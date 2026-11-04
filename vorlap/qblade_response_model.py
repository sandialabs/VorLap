"""Response-aware parked-VIV forcing for the VorLap--QBlade bridge.

This module contains a deliberately small, single-generalized-mode model.  It
is disabled by default and returns an *additive* global nodal force, so it can
be layered on the existing fixed-body VorLap force without changing that
force path.

Let ``phi_i`` be a dimensionless mode shape normalized to unit maximum (the
tip for the intended cantilever use), ``e_i`` the supplied global force
direction, ``v_i`` the QBlade structural velocity [m/s], and ``w_i > 0`` the
optional projection weight.  The measured generalized modal velocity [m/s]
is the weighted least-squares projection

``qdot_raw = sum(w_i*phi_i*dot(v_i,e_i)) / sum(w_i*phi_i**2)``.

Uniform weights are retained as a backward-compatible default.  Qualification
cases should normally supply generalized-mass quadrature weights, for example
nodal mass times tributary length, so that nonmodal motion does not receive an
arbitrary node-count weighting.

Rigid drift must not be interpreted as vortex-shedding motion.  The projected
velocity therefore passes through the stable second-order bandpass

``H(s) = 2*zeta_b*omega_n*s / (s**2 + 2*zeta_b*omega_n*s + omega_n**2)``,

which has zero DC gain and unity gain with zero phase at ``f_n``.  Its state is
initialized at the DC equilibrium and advanced exactly for a linearly varying
input over each accepted time step.  The state
``s_rms = EMA(qdot_filtered**2)`` [m2/s2] is then advanced exactly for a
piecewise-constant endpoint value over ``dt``:

``s_rms(t+dt) = exp(-dt/tau)*s_rms(t)``
``                 + (1-exp(-dt/tau))*qdot_filtered**2``.

For a response dominated by the configured natural frequency ``f_n`` [Hz],
the estimated peak modal displacement is
``A = sqrt(2*s)/(2*pi*f_n)`` [m].  Each local scalar force is the Scanlan-style
velocity-proportional term

``F_i = 0.5*rho*U_i*D_i*l_i*H1*a(Ur_i)``
``      *(1-(A/(D_ref*a_sat))**2) * (phi_i*qdot_filtered)`` [N],

where ``rho`` is kg/m3, ``U`` is m/s, and ``D`` and ``l`` are m.  ``H1`` and
the smooth reduced-velocity activation ``a`` are dimensionless.  Positive
``H1`` therefore gives negative aerodynamic damping (force in phase with
modal velocity, positive work) below ``A/D_ref = a_sat`` and positive damping
(negative work) above it.  This is an engineering energy-balance closure, not
a resolved wake oscillator or a standalone certification validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numbers
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from scipy.linalg import expm


_CHECKPOINT_SCHEMA = "vorlap-qblade-response-model-v2"
_TIME_ABS_TOL_S = 1.0e-12
_FILTER_TRANSITION_CACHE_LIMIT = 16

RESPONSE_MODEL_SCHEMA_VERSION_RELATIVE_CROSSFLOW = 1
RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW = 2
AERODYNAMIC_SPEED_BASIS_RELATIVE_CROSSFLOW = "relative_crossflow"
AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW = (
    "declared_ambient_crossflow"
)
RESPONSE_SECTION_WIDTH_BASIS_RELATIVE_CROSSFLOW = "relative_crossflow_angle"
RESPONSE_SECTION_WIDTH_BASIS_DECLARED_AMBIENT_CROSSFLOW = (
    "declared_ambient_crossflow_angle"
)
RESPONSE_SECTION_WIDTH_BASIS_EXPLICIT = "explicit_response_section_width_m"

_STARTUP_SEED_SCHEMA_VERSION = 1
_STARTUP_SEED_TAPER = "hann_sine_squared"
_STARTUP_SEED_CARRIER = "natural_frequency"
_STARTUP_SEED_NODAL_WEIGHTING = "projection_mass_times_mode"
_STARTUP_SEED_LIMIT_SOURCE = "response_model_max_abs_nodal_force_n"


@dataclass(frozen=True)
class QBladeStartupSeedConfig:
    """Deterministic finite-duration modal startup excitation.

    The seed acceleration is

    ``a_s(t) = a_0 sin(pi*t/T)^2 sin(2*pi*f_n*t)`` for ``0 <= t < T``

    and is identically zero otherwise, with ``T=duration_cycles/f_n``.  Nodal
    forces are ``m_i*phi_i*a_s`` where ``m_i`` are the response model's
    configured ``projection_weights`` interpreted as nodal mass quadrature
    [kg].  ``a_0`` is solved once so an ideal linear SDOF with the configured
    natural frequency and structural damping finishes the pulse with the
    requested free-response envelope ``target_amplitude_ratio*D_ref``.

    The string fields deliberately make the waveform, carrier, distribution,
    and force-limit provenance portable rather than leaving them as implicit
    implementation details.  Only the enumerated values below are supported.
    """

    enabled: bool = False
    duration_cycles: float = 5.0
    target_amplitude_ratio: float = 0.005
    structural_damping_ratio: float = 0.0
    schema_version: int = _STARTUP_SEED_SCHEMA_VERSION
    taper: str = _STARTUP_SEED_TAPER
    carrier: str = _STARTUP_SEED_CARRIER
    nodal_weighting: str = _STARTUP_SEED_NODAL_WEIGHTING
    nodal_force_limit_source: str = _STARTUP_SEED_LIMIT_SOURCE


@dataclass(frozen=True)
class QBladeResponseModelConfig:
    """Configuration for one parked-VIV generalized mode.

    Parameters use SI units. ``mode_shape`` is associated one-to-one with
    ``node_ids`` and is internally normalized by its maximum absolute value,
    making the result invariant to a nonzero scalar rescaling or sign reversal
    of the supplied shape. ``reference_length_m`` is the modal reference
    dimension used in reduced velocity and the response ratio ``A/D_ref``.

    The activation is zero outside
    ``[reduced_velocity_low, reduced_velocity_high]``, one at
    ``reduced_velocity_peak``, and uses cubic smoothstep ramps between those
    points. ``max_abs_gain`` bounds the complete signed dimensionless product
    ``H1*activation*saturation`` before force evaluation;
    ``max_abs_nodal_force_n`` then bounds each signed nodal force.
    ``force_direction`` declares whether bridge integration supplies the
    rotating section ``normal`` (default) or ``chord`` basis.  The standalone
    model still consumes the explicit basis passed to :meth:`step`.

    ``projection_weights`` are optional positive weights in ``node_ids``
    order.  The recommended qualification choice is generalized-mass
    quadrature (mass times tributary length); omitted weights are uniform and
    this fallback is explicitly identified in diagnostics.
    ``bandpass_damping_ratio`` is the dimensionless ``zeta_b`` in the filter
    equation above and must lie strictly between zero and one.

    ``enabled`` defaults to ``False`` so constructing a calibrated model does
    not alter the legacy force path until explicitly requested.
    """

    node_ids: Tuple[str, ...]
    mode_shape: Tuple[float, ...]
    natural_frequency_hz: float
    reference_length_m: float
    reduced_velocity_low: float
    reduced_velocity_peak: float
    reduced_velocity_high: float
    aerodynamic_damping_gain_h1: float
    saturation_amplitude_ratio: float
    rms_time_constant_s: float
    max_abs_gain: float
    max_abs_nodal_force_n: float
    force_direction: str = "normal"
    projection_weights: Tuple[float, ...] | None = None
    bandpass_damping_ratio: float = 0.1
    enabled: bool = False
    startup_seed: QBladeStartupSeedConfig | None = None
    schema_version: int = RESPONSE_MODEL_SCHEMA_VERSION_RELATIVE_CROSSFLOW
    aerodynamic_speed_basis: str = AERODYNAMIC_SPEED_BASIS_RELATIVE_CROSSFLOW
    response_section_width_m: Tuple[float, ...] | float | None = None


@dataclass(frozen=True)
class QBladeResponseDiagnostics:
    """Typed scalar diagnostics for one response-model evaluation.

    ``raw_qdot_mps`` is the weighted structural projection and
    ``filtered_qdot_mps`` is its narrowband component used by both the force
    and amplitude estimator.  ``qdot_mps`` is retained as an alias for the
    filtered value.

    Power follows the structural convention ``sum(F dot v)`` [W]: positive
    values add structural energy and negative values remove it.
    ``modal_power_w`` uses only the reconstructed modal velocity, whereas
    ``structural_power_w`` uses the complete supplied structural velocity.
    """

    enabled: bool
    time_s: float
    dt_s: float
    reduced_velocity_min: float
    reduced_velocity_max: float
    activation_min: float
    activation_max: float
    raw_qdot_mps: float
    filtered_qdot_mps: float
    qdot_mps: float
    qdot_rms_mps: float
    projection_weighting: str
    projection_weight_min: float
    projection_weight_max: float
    amplitude_estimate_m: float
    amplitude_ratio: float
    saturation_factor: float
    raw_gain_min: float
    raw_gain_max: float
    applied_gain_min: float
    applied_gain_max: float
    gain_limited_nodes: int
    force_limited_nodes: int
    modal_power_w: float
    structural_power_w: float
    max_abs_nodal_force_n: float


@dataclass(frozen=True)
class QBladeSeededResponseDiagnostics(QBladeResponseDiagnostics):
    """Response diagnostics augmented with finite startup-seed evidence.

    A distinct subclass keeps the default, unseeded diagnostics and SIL
    history byte-for-byte compatible.  Raw seed force quantities equal the
    applied seed quantities whenever both seed and total ``force_limited``
    counters are zero, which is the required qualification condition.
    """

    startup_seed_enabled: bool
    startup_seed_active: bool
    startup_seed_elapsed_cycles: float
    startup_seed_duration_cycles: float
    startup_seed_duration_s: float
    startup_seed_taper: float
    startup_seed_target_amplitude_ratio: float
    startup_seed_target_amplitude_m: float
    startup_seed_ideal_terminal_amplitude_m: float
    startup_seed_modal_acceleration_scale_mps2: float
    startup_seed_modal_acceleration_mps2: float
    startup_seed_raw_generalized_force_n: float
    startup_seed_raw_max_abs_nodal_force_n: float
    startup_seed_force_limited_nodes: int
    startup_seed_modal_power_w: float


@dataclass(frozen=True)
class QBladeResponseDiagnosticsV2(QBladeResponseDiagnostics):
    """Schema-v2 response diagnostics with explicit speed/width provenance.

    This is deliberately a distinct subclass.  Schema-v1 continues to emit
    :class:`QBladeResponseDiagnostics` with its historical field order and
    serialized shape, while v2 history gains the provenance required to audit
    the response-only aerodynamic closure.
    """

    response_model_schema_version: int
    aerodynamic_speed_basis: str
    response_declared_ambient_crossflow_speed_min_mps: float
    response_declared_ambient_crossflow_speed_max_mps: float
    response_relative_crossflow_speed_min_mps: float
    response_relative_crossflow_speed_max_mps: float
    response_section_width_basis: str
    response_section_width_min_m: float
    response_section_width_max_m: float
    configured_response_reduced_velocity_min: float
    configured_response_reduced_velocity_max: float
    realized_response_reduced_velocity_min: float
    realized_response_reduced_velocity_max: float
    speed_basis_invariant_violations: int
    activation_invariant_violations: int


@dataclass(frozen=True)
class QBladeSeededResponseDiagnosticsV2(QBladeResponseDiagnosticsV2):
    """Schema-v2 provenance plus deterministic startup-seed diagnostics."""

    startup_seed_enabled: bool
    startup_seed_active: bool
    startup_seed_elapsed_cycles: float
    startup_seed_duration_cycles: float
    startup_seed_duration_s: float
    startup_seed_taper: float
    startup_seed_target_amplitude_ratio: float
    startup_seed_target_amplitude_m: float
    startup_seed_ideal_terminal_amplitude_m: float
    startup_seed_modal_acceleration_scale_mps2: float
    startup_seed_modal_acceleration_mps2: float
    startup_seed_raw_generalized_force_n: float
    startup_seed_raw_max_abs_nodal_force_n: float
    startup_seed_force_limited_nodes: int
    startup_seed_modal_power_w: float


@dataclass(frozen=True)
class QBladeResponseResult:
    """Additive global nodal forces [N] and their response diagnostics."""

    additive_force_n: np.ndarray
    diagnostics: QBladeResponseDiagnostics


def _smooth_reduced_velocity_activation(
    reduced_velocity: np.ndarray,
    low: float,
    peak: float,
    high: float,
) -> np.ndarray:
    """Return a C1-continuous low--peak--high activation in ``[0, 1]``."""

    value = np.asarray(reduced_velocity, dtype=float)
    activation = np.zeros_like(value)
    rising = (value > low) & (value < peak)
    falling = (value >= peak) & (value < high)
    rise_coordinate = (value[rising] - low) / (peak - low)
    fall_coordinate = (high - value[falling]) / (high - peak)
    activation[rising] = rise_coordinate**2 * (3.0 - 2.0 * rise_coordinate)
    activation[falling] = fall_coordinate**2 * (3.0 - 2.0 * fall_coordinate)
    activation[value == peak] = 1.0
    return activation


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _node_vector(value: Any, n_nodes: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n_nodes, float(array), dtype=float)
    if array.shape != (n_nodes,):
        raise ValueError(f"{name} must be scalar or have shape ({n_nodes},).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.array(array, dtype=float, copy=True)


class QBladeResponseModel:
    """Stateful amplitude-dependent aerodynamic damping for one mode.

    Call :meth:`step` exactly once per accepted structural time level.  An
    exact repeat at the same time returns the cached force and does not advance
    the RMS state, which makes nonlinear-solver retries idempotent.  A changed
    input at the same time, or a decreasing time, raises ``ValueError`` rather
    than silently corrupting the energy state.

    Direction rows are normalized on every call and may vary with the rotating
    or deforming structure.  Step node order may differ from configuration
    order; IDs are used to map the fixed mode shape and the returned force uses
    the step order.
    """

    def __init__(self, config: QBladeResponseModelConfig):
        self.config = config
        self._validate_config(config)

        self._node_ids = tuple(str(node_id) for node_id in config.node_ids)
        raw_mode = np.asarray(config.mode_shape, dtype=float)
        normalized_mode = raw_mode / float(np.max(np.abs(raw_mode)))
        mode_by_node = dict(zip(self._node_ids, normalized_mode))
        first_nonzero = next(
            mode_by_node[node_id]
            for node_id in sorted(self._node_ids)
            if mode_by_node[node_id] != 0.0
        )
        # A mode and its sign reversal are the same physical basis.  Fixing a
        # deterministic orientation also makes filter checkpoint state
        # portable across equivalent input representations.
        self._mode_shape = normalized_mode * math.copysign(1.0, first_nonzero)
        self._mode_by_node = dict(zip(self._node_ids, self._mode_shape))
        if config.projection_weights is None:
            self._projection_weights = np.ones(len(self._node_ids), dtype=float)
            self._projection_weighting = "uniform_default"
        else:
            self._projection_weights = np.asarray(
                config.projection_weights,
                dtype=float,
            ).copy()
            self._projection_weighting = "configured"
        self._weight_by_node = dict(zip(self._node_ids, self._projection_weights))
        if config.response_section_width_m is None:
            self._response_section_width_m: np.ndarray | None = None
        else:
            self._response_section_width_m = _node_vector(
                config.response_section_width_m,
                len(self._node_ids),
                "response_section_width_m",
            )

        self._startup_seed_enabled = bool(
            config.startup_seed is not None and config.startup_seed.enabled
        )
        self._startup_seed_duration_s = 0.0
        self._startup_seed_target_amplitude_m = 0.0
        self._startup_seed_unit_terminal_amplitude_m_per_mps2 = 0.0
        self._startup_seed_modal_acceleration_scale_mps2 = 0.0
        self._startup_seed_ideal_terminal_amplitude_m = 0.0
        if self._startup_seed_enabled:
            seed = config.startup_seed
            if seed is None:  # pragma: no cover - protected by expression above
                raise RuntimeError("Enabled startup seed configuration is missing.")
            self._startup_seed_duration_s = (
                float(seed.duration_cycles) / float(config.natural_frequency_hz)
            )
            self._startup_seed_target_amplitude_m = (
                float(seed.target_amplitude_ratio)
                * float(config.reference_length_m)
            )
            self._startup_seed_unit_terminal_amplitude_m_per_mps2 = (
                self._ideal_seed_terminal_amplitude_for_unit_acceleration(
                    natural_frequency_hz=float(config.natural_frequency_hz),
                    structural_damping_ratio=float(
                        seed.structural_damping_ratio
                    ),
                    duration_s=self._startup_seed_duration_s,
                )
            )
            if self._startup_seed_unit_terminal_amplitude_m_per_mps2 <= 0.0:
                raise ValueError(
                    "startup_seed produced a degenerate ideal terminal response."
                )
            self._startup_seed_modal_acceleration_scale_mps2 = (
                self._startup_seed_target_amplitude_m
                / self._startup_seed_unit_terminal_amplitude_m_per_mps2
            )
            self._startup_seed_ideal_terminal_amplitude_m = (
                self._startup_seed_modal_acceleration_scale_mps2
                * self._startup_seed_unit_terminal_amplitude_m_per_mps2
            )
        self._config_fingerprint = self._fingerprint_config()

        self._last_time_s: float | None = None
        self._filter_position_state = 0.0
        self._filter_rate_state = 0.0
        self._filter_previous_input_mps: float | None = None
        self._filter_transition_cache: dict[float, np.ndarray] = {}
        self._qdot_mean_square_m2ps2 = 0.0
        self._last_input_signature: str | None = None
        self._last_result: QBladeResponseResult | None = None

    @property
    def last_diagnostics(self) -> QBladeResponseDiagnostics | None:
        """Diagnostics from the last accepted step, or ``None`` before start."""
        if self._last_result is None:
            return None
        return self._last_result.diagnostics

    @property
    def startup_seed_enabled(self) -> bool:
        """Whether the deterministic finite-duration startup seed is active."""

        return self._startup_seed_enabled

    @property
    def schema_version(self) -> int:
        """Portable response-model schema version."""

        return int(self.config.schema_version)

    @property
    def aerodynamic_speed_basis(self) -> str:
        """Declared crossflow-speed basis used by activation and force."""

        return str(self.config.aerodynamic_speed_basis)

    @property
    def response_section_width_basis(self) -> str:
        """Provenance label for the response-only section width."""

        if self._response_section_width_m is not None:
            return RESPONSE_SECTION_WIDTH_BASIS_EXPLICIT
        if (
            self.aerodynamic_speed_basis
            == AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW
        ):
            return RESPONSE_SECTION_WIDTH_BASIS_DECLARED_AMBIENT_CROSSFLOW
        return RESPONSE_SECTION_WIDTH_BASIS_RELATIVE_CROSSFLOW

    @property
    def response_section_width_m(self) -> np.ndarray | None:
        """Explicit per-node response width, or ``None`` for flow-derived width."""

        if self._response_section_width_m is None:
            return None
        return np.array(self._response_section_width_m, dtype=float, copy=True)

    @property
    def config_fingerprint(self) -> str:
        """Deterministic identity used to reject incompatible checkpoints."""

        return self._config_fingerprint

    @staticmethod
    def _validate_config(config: QBladeResponseModelConfig) -> None:
        node_ids = tuple(str(node_id) for node_id in config.node_ids)
        if not node_ids:
            raise ValueError("node_ids must contain at least one node.")
        if any(not node_id for node_id in node_ids):
            raise ValueError("node_ids must not contain empty IDs.")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be unique.")

        mode_shape = np.asarray(config.mode_shape, dtype=float)
        if mode_shape.shape != (len(node_ids),):
            raise ValueError("mode_shape must contain one scalar for each node_id.")
        if not np.all(np.isfinite(mode_shape)):
            raise ValueError("mode_shape must contain only finite values.")
        if float(np.max(np.abs(mode_shape))) <= 0.0:
            raise ValueError("mode_shape must contain at least one nonzero value.")
        if config.projection_weights is not None:
            projection_weights = np.asarray(config.projection_weights, dtype=float)
            if projection_weights.shape != (len(node_ids),):
                raise ValueError(
                    "projection_weights must contain one scalar for each node_id."
                )
            if (
                np.any(~np.isfinite(projection_weights))
                or np.any(projection_weights <= 0.0)
            ):
                raise ValueError("projection_weights must be finite and positive.")
        if (
            isinstance(config.schema_version, bool)
            or not isinstance(config.schema_version, numbers.Integral)
            or int(config.schema_version)
            not in {
                RESPONSE_MODEL_SCHEMA_VERSION_RELATIVE_CROSSFLOW,
                RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW,
            }
        ):
            raise ValueError("schema_version must be 1 or 2.")
        if (
            int(config.schema_version)
            == RESPONSE_MODEL_SCHEMA_VERSION_RELATIVE_CROSSFLOW
            and config.response_section_width_m is not None
        ):
            raise ValueError(
                "response_section_width_m is supported only for response schema_version 2."
            )
        if config.response_section_width_m is not None:
            raw_width = config.response_section_width_m
            raw_width_values = (
                (raw_width,)
                if np.asarray(raw_width).ndim == 0
                else tuple(raw_width)
            )
            if any(isinstance(value, (bool, np.bool_)) for value in raw_width_values):
                raise TypeError(
                    "response_section_width_m values must be real numeric values, not boolean."
                )
            response_width = np.asarray(raw_width, dtype=float)
            if response_width.ndim == 0:
                response_width = np.full(len(node_ids), float(response_width))
            if response_width.shape != (len(node_ids),):
                raise ValueError(
                    "response_section_width_m must be scalar or contain one value for each node_id."
                )
            if np.any(~np.isfinite(response_width)) or np.any(response_width <= 0.0):
                raise ValueError(
                    "response_section_width_m must contain finite positive values."
                )
        expected_speed_basis = (
            AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW
            if int(config.schema_version)
            == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
            else AERODYNAMIC_SPEED_BASIS_RELATIVE_CROSSFLOW
        )
        if (
            not isinstance(config.aerodynamic_speed_basis, str)
            or config.aerodynamic_speed_basis != expected_speed_basis
        ):
            raise ValueError(
                "aerodynamic_speed_basis must be "
                f"{expected_speed_basis!r} for response schema_version "
                f"{int(config.schema_version)}."
            )

        positive_fields = {
            "natural_frequency_hz": config.natural_frequency_hz,
            "reference_length_m": config.reference_length_m,
            "saturation_amplitude_ratio": config.saturation_amplitude_ratio,
            "rms_time_constant_s": config.rms_time_constant_s,
            "max_abs_gain": config.max_abs_gain,
            "max_abs_nodal_force_n": config.max_abs_nodal_force_n,
            "bandpass_damping_ratio": config.bandpass_damping_ratio,
        }
        for name, raw_value in positive_fields.items():
            value = _finite_scalar(raw_value, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")

        h1 = _finite_scalar(
            config.aerodynamic_damping_gain_h1,
            "aerodynamic_damping_gain_h1",
        )
        if h1 < 0.0:
            raise ValueError("aerodynamic_damping_gain_h1 must be non-negative.")
        if str(config.force_direction) not in {"normal", "chord"}:
            raise ValueError("force_direction must be either 'normal' or 'chord'.")
        if not isinstance(config.enabled, bool):
            raise TypeError("enabled must be a boolean.")
        bandpass_damping = _finite_scalar(
            config.bandpass_damping_ratio,
            "bandpass_damping_ratio",
        )
        if bandpass_damping <= 0.0 or bandpass_damping >= 1.0:
            raise ValueError("bandpass_damping_ratio must be between zero and one.")

        low = _finite_scalar(config.reduced_velocity_low, "reduced_velocity_low")
        peak = _finite_scalar(config.reduced_velocity_peak, "reduced_velocity_peak")
        high = _finite_scalar(config.reduced_velocity_high, "reduced_velocity_high")
        if low < 0.0 or not low < peak < high:
            raise ValueError(
                "Reduced-velocity bounds must satisfy 0 <= low < peak < high."
            )

        seed = config.startup_seed
        if seed is not None:
            if not isinstance(seed, QBladeStartupSeedConfig):
                raise TypeError(
                    "startup_seed must be a QBladeStartupSeedConfig or None."
                )
            if not isinstance(seed.enabled, bool):
                raise TypeError("startup_seed.enabled must be a boolean.")
            if (
                isinstance(seed.schema_version, bool)
                or not isinstance(seed.schema_version, numbers.Integral)
                or seed.schema_version != 1
            ):
                raise ValueError("startup_seed.schema_version must be 1.")
            if isinstance(seed.duration_cycles, bool) or not isinstance(
                seed.duration_cycles,
                numbers.Real,
            ):
                raise TypeError(
                    "startup_seed.duration_cycles must be a real numeric value, not boolean."
                )
            duration_cycles = _finite_scalar(
                seed.duration_cycles,
                "startup_seed.duration_cycles",
            )
            if duration_cycles <= 0.0:
                raise ValueError("startup_seed.duration_cycles must be positive.")
            if isinstance(seed.target_amplitude_ratio, bool) or not isinstance(
                seed.target_amplitude_ratio,
                numbers.Real,
            ):
                raise TypeError(
                    "startup_seed.target_amplitude_ratio must be a real numeric value, not boolean."
                )
            target_ratio = _finite_scalar(
                seed.target_amplitude_ratio,
                "startup_seed.target_amplitude_ratio",
            )
            if target_ratio <= 0.0 or target_ratio > 1.0:
                raise ValueError(
                    "startup_seed.target_amplitude_ratio must be in (0, 1]."
                )
            if isinstance(seed.structural_damping_ratio, bool) or not isinstance(
                seed.structural_damping_ratio,
                numbers.Real,
            ):
                raise TypeError(
                    "startup_seed.structural_damping_ratio must be a real numeric value, not boolean."
                )
            structural_damping = _finite_scalar(
                seed.structural_damping_ratio,
                "startup_seed.structural_damping_ratio",
            )
            if structural_damping < 0.0 or structural_damping >= 1.0:
                raise ValueError(
                    "startup_seed.structural_damping_ratio must be in [0, 1)."
                )
            expected_strings = {
                "taper": (seed.taper, _STARTUP_SEED_TAPER),
                "carrier": (seed.carrier, _STARTUP_SEED_CARRIER),
                "nodal_weighting": (
                    seed.nodal_weighting,
                    _STARTUP_SEED_NODAL_WEIGHTING,
                ),
                "nodal_force_limit_source": (
                    seed.nodal_force_limit_source,
                    _STARTUP_SEED_LIMIT_SOURCE,
                ),
            }
            for name, (actual, expected) in expected_strings.items():
                if not isinstance(actual, str) or actual != expected:
                    raise ValueError(
                        f"startup_seed.{name} must be {expected!r}."
                    )
            if seed.enabled:
                if not config.enabled:
                    raise ValueError(
                        "startup_seed.enabled requires response model enabled=True."
                    )
                if config.projection_weights is None:
                    raise ValueError(
                        "startup_seed.enabled requires projection_weights as nodal mass quadrature [kg]."
                    )

    def _fingerprint_config(self) -> str:
        node_data = sorted(
            (node_id, float(mode), float(weight))
            for node_id, mode, weight in zip(
                self._node_ids,
                self._mode_shape,
                self._projection_weights,
            )
        )
        payload = {
            "node_ids": [node_id for node_id, _, _ in node_data],
            # Mode sign, scale, and input ordering are physical invariances.
            "mode_shape": [value for _, value, _ in node_data],
            "projection_weights": [weight for _, _, weight in node_data],
            "projection_weighting": self._projection_weighting,
            "natural_frequency_hz": float(self.config.natural_frequency_hz),
            "reference_length_m": float(self.config.reference_length_m),
            "reduced_velocity_low": float(self.config.reduced_velocity_low),
            "reduced_velocity_peak": float(self.config.reduced_velocity_peak),
            "reduced_velocity_high": float(self.config.reduced_velocity_high),
            "aerodynamic_damping_gain_h1": float(
                self.config.aerodynamic_damping_gain_h1
            ),
            "saturation_amplitude_ratio": float(
                self.config.saturation_amplitude_ratio
            ),
            "rms_time_constant_s": float(self.config.rms_time_constant_s),
            "max_abs_gain": float(self.config.max_abs_gain),
            "max_abs_nodal_force_n": float(self.config.max_abs_nodal_force_n),
            "force_direction": str(self.config.force_direction),
            "bandpass_damping_ratio": float(self.config.bandpass_damping_ratio),
            "enabled": bool(self.config.enabled),
        }
        if (
            self.schema_version
            == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        ):
            payload.update(
                {
                    "schema_version": self.schema_version,
                    "aerodynamic_speed_basis": self.aerodynamic_speed_basis,
                    "response_section_width_m": (
                        None
                        if self._response_section_width_m is None
                        else sorted(
                            (node_id, float(width))
                            for node_id, width in zip(
                                self._node_ids,
                                self._response_section_width_m,
                            )
                        )
                    ),
                }
            )
        if (
            self.config.startup_seed is not None
            and self.config.startup_seed.enabled
        ):
            payload["startup_seed"] = asdict(self.config.startup_seed)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _ideal_seed_terminal_amplitude_for_unit_acceleration(
        *,
        natural_frequency_hz: float,
        structural_damping_ratio: float,
        duration_s: float,
    ) -> float:
        """Return ideal terminal free-response envelope per 1 m/s2 seed.

        The Hann-windowed carrier is represented exactly as three sinusoids,

        ``0.5 sin(w*t) - 0.25 sin((w+2*pi/T)*t)``
        ``                 - 0.25 sin((w-2*pi/T)*t)``.

        An augmented constant-coefficient state transition therefore computes
        the endpoint without timestep-dependent quadrature.  The returned
        amplitude is the coefficient of the freely decaying underdamped
        response at the instant the forcing becomes identically zero.
        """

        omega = 2.0 * math.pi * natural_frequency_hz
        modulation = 2.0 * math.pi / duration_s
        forcing_terms = (
            (0.5, omega),
            (-0.25, omega + modulation),
            (-0.25, omega - modulation),
        )
        state_size = 2 + 2 * len(forcing_terms)
        dynamics = np.zeros((state_size, state_size), dtype=float)
        dynamics[0, 1] = 1.0
        dynamics[1, 0] = -omega**2
        dynamics[1, 1] = -2.0 * structural_damping_ratio * omega
        initial = np.zeros(state_size, dtype=float)
        for term_index, (coefficient, forcing_frequency) in enumerate(
            forcing_terms
        ):
            sine_index = 2 + 2 * term_index
            cosine_index = sine_index + 1
            dynamics[1, sine_index] = coefficient
            dynamics[sine_index, cosine_index] = forcing_frequency
            dynamics[cosine_index, sine_index] = -forcing_frequency
            initial[cosine_index] = 1.0

        terminal = expm(dynamics * duration_s) @ initial
        displacement_m = float(terminal[0])
        velocity_mps = float(terminal[1])
        damped_omega = omega * math.sqrt(1.0 - structural_damping_ratio**2)
        quadrature_displacement_m = (
            velocity_mps
            + structural_damping_ratio * omega * displacement_m
        ) / damped_omega
        amplitude_m = math.hypot(displacement_m, quadrature_displacement_m)
        if not math.isfinite(amplitude_m):
            raise ValueError("startup_seed ideal terminal amplitude is non-finite.")
        return amplitude_m

    def _startup_seed_modal_acceleration(
        self,
        time_s: float,
    ) -> tuple[bool, float, float, float]:
        """Return ``(active, elapsed_cycles, taper, acceleration)``.

        Absolute simulation time makes the waveform deterministic across
        restarts.  The branch at ``t >= T`` guarantees an exact floating-point
        zero after the declared seed interval, including its endpoint.
        """

        if not self._startup_seed_enabled:
            return False, 0.0, 0.0, 0.0
        duration_s = self._startup_seed_duration_s
        if time_s < 0.0:
            return False, 0.0, 0.0, 0.0
        if time_s >= duration_s:
            seed = self.config.startup_seed
            if seed is None:  # pragma: no cover - protected state invariant
                raise RuntimeError("Enabled startup seed configuration is missing.")
            return False, float(seed.duration_cycles), 0.0, 0.0
        natural_frequency_hz = float(self.config.natural_frequency_hz)
        elapsed_cycles = time_s * natural_frequency_hz
        taper = math.sin(math.pi * time_s / duration_s) ** 2
        carrier = math.sin(2.0 * math.pi * natural_frequency_hz * time_s)
        acceleration_mps2 = (
            self._startup_seed_modal_acceleration_scale_mps2 * taper * carrier
        )
        return True, elapsed_cycles, taper, acceleration_mps2

    def _filter_projected_velocity(self, raw_qdot_mps: float, dt_s: float) -> float:
        """Advance the modal-velocity bandpass and return its endpoint [m/s].

        The continuous state ``x`` satisfies

        ``x1_dot = x2``
        ``x2_dot = qdot_raw - 2*zeta_b*omega_n*x2 - omega_n**2*x1``
        ``qdot_filtered = 2*zeta_b*omega_n*x2``.

        At the first accepted sample, ``x1=qdot_raw/omega_n**2`` and ``x2=0``
        place the filter exactly at its DC equilibrium.  Later propagation
        uses a matrix exponential augmented with the input and its constant
        within-step slope, so the update is exact for linear interpolation of
        consecutive projected-velocity samples.
        """

        omega_n = 2.0 * math.pi * float(self.config.natural_frequency_hz)
        damping = float(self.config.bandpass_damping_ratio)
        if self._filter_previous_input_mps is None:
            self._filter_position_state = raw_qdot_mps / omega_n**2
            self._filter_rate_state = 0.0
            self._filter_previous_input_mps = raw_qdot_mps
            return 0.0

        if dt_s <= 0.0:  # pragma: no cover - duplicate steps return earlier
            raise RuntimeError("Bandpass state requires a positive accepted time step.")

        transition = self._filter_transition_cache.get(dt_s)
        if transition is None:
            augmented_dynamics = np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [-omega_n**2, -2.0 * damping * omega_n, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
            transition = expm(augmented_dynamics * dt_s)
            if len(self._filter_transition_cache) >= _FILTER_TRANSITION_CACHE_LIMIT:
                self._filter_transition_cache.pop(next(iter(self._filter_transition_cache)))
            self._filter_transition_cache[dt_s] = transition

        previous_input = float(self._filter_previous_input_mps)
        input_slope = (raw_qdot_mps - previous_input) / dt_s
        augmented_state = transition @ np.array(
            [
                self._filter_position_state,
                self._filter_rate_state,
                previous_input,
                input_slope,
            ],
            dtype=float,
        )
        self._filter_position_state = float(augmented_state[0])
        self._filter_rate_state = float(augmented_state[1])
        self._filter_previous_input_mps = raw_qdot_mps
        return 2.0 * damping * omega_n * self._filter_rate_state

    @staticmethod
    def _input_signature(
        time_s: float,
        node_ids: Tuple[str, ...],
        direction_basis: np.ndarray,
        structural_velocity_mps: np.ndarray,
        effective_velocity_mps: np.ndarray,
        section_width_m: np.ndarray,
        local_length_m: np.ndarray,
        fluid_density_kgpm3: float,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray([time_s, fluid_density_kgpm3], dtype="<f8").tobytes())
        for node_id in node_ids:
            encoded = node_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
            digest.update(encoded)
        for array in (
            direction_basis,
            structural_velocity_mps,
            effective_velocity_mps,
            section_width_m,
            local_length_m,
        ):
            canonical = np.ascontiguousarray(array, dtype="<f8")
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _input_signature_v2(
        time_s: float,
        node_ids: Tuple[str, ...],
        direction_basis: np.ndarray,
        structural_velocity_mps: np.ndarray,
        declared_ambient_crossflow_speed_mps: np.ndarray,
        relative_crossflow_speed_mps: np.ndarray,
        section_width_m: np.ndarray,
        local_length_m: np.ndarray,
        fluid_density_kgpm3: float,
        activation_override: np.ndarray | None = None,
    ) -> str:
        """Hash all schema-v2 physical inputs without changing v1 hashes."""

        digest = hashlib.sha256()
        digest.update(np.asarray([time_s, fluid_density_kgpm3], dtype="<f8").tobytes())
        for node_id in node_ids:
            encoded = node_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
            digest.update(encoded)
        for array in (
            direction_basis,
            structural_velocity_mps,
            declared_ambient_crossflow_speed_mps,
            relative_crossflow_speed_mps,
            section_width_m,
            local_length_m,
        ):
            canonical = np.ascontiguousarray(array, dtype="<f8")
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes())
        if activation_override is not None:
            digest.update(b"spectral_activation_override\0")
            canonical = np.ascontiguousarray(activation_override, dtype="<f8")
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes())
        return digest.hexdigest()

    def step(
        self,
        time_s: float,
        *,
        node_ids: Sequence[str],
        direction_basis: Any,
        structural_velocity_mps: Any,
        effective_velocity_mps: Any,
        section_width_m: Any,
        local_length_m: Any,
        fluid_density_kgpm3: float,
        declared_ambient_crossflow_speed_mps: Any | None = None,
        relative_crossflow_speed_mps: Any | None = None,
        response_section_width_basis: str | None = None,
        activation_override: Any | None = None,
    ) -> QBladeResponseResult:
        """Evaluate additive nodal forces at one accepted time level.

        ``direction_basis`` and ``structural_velocity_mps`` have shape
        ``(n_nodes, 3)`` and are global unit directions and velocities [m/s].
        Direction rows are normalized internally. ``effective_velocity_mps``
        is the non-negative local cross-flow speed ``U`` [m/s].
        ``section_width_m`` is the local cross-flow dimension ``D`` [m],
        ``local_length_m`` is tributary span ``l`` [m], and density is kg/m3.

        The returned array has shape ``(n_nodes, 3)`` in the supplied node
        order and is additive: callers retain responsibility for summing it
        with, and applying the same global/local convention as, other forces.
        """

        current_time = _finite_scalar(time_s, "time_s")
        step_node_ids = tuple(str(node_id) for node_id in node_ids)
        if len(step_node_ids) != len(self._node_ids):
            raise ValueError("node_ids must contain every configured node exactly once.")
        if (
            len(set(step_node_ids)) != len(step_node_ids)
            or set(step_node_ids) != set(self._node_ids)
        ):
            raise ValueError("node_ids must contain every configured node exactly once.")
        n_nodes = len(step_node_ids)

        directions = np.asarray(direction_basis, dtype=float)
        velocities = np.asarray(structural_velocity_mps, dtype=float)
        if directions.shape != (n_nodes, 3):
            raise ValueError(f"direction_basis must have shape ({n_nodes}, 3).")
        if velocities.shape != (n_nodes, 3):
            raise ValueError(f"structural_velocity_mps must have shape ({n_nodes}, 3).")
        if not np.all(np.isfinite(directions)):
            raise ValueError("direction_basis must contain only finite values.")
        if not np.all(np.isfinite(velocities)):
            raise ValueError("structural_velocity_mps must contain only finite values.")
        direction_norm = np.linalg.norm(directions, axis=1)
        if np.any(direction_norm <= 0.0):
            raise ValueError("Every direction_basis row must be nonzero.")
        directions = np.array(directions / direction_norm[:, None], dtype=float, copy=True)
        velocities = np.array(velocities, dtype=float, copy=True)

        effective_velocity = _node_vector(
            effective_velocity_mps,
            n_nodes,
            "effective_velocity_mps",
        )
        is_schema_v2 = (
            self.schema_version
            == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        )
        if is_schema_v2:
            if declared_ambient_crossflow_speed_mps is None:
                raise ValueError(
                    "Schema-v2 response forcing requires declared_ambient_crossflow_speed_mps."
                )
            declared_ambient_crossflow_speed = _node_vector(
                declared_ambient_crossflow_speed_mps,
                n_nodes,
                "declared_ambient_crossflow_speed_mps",
            )
            if relative_crossflow_speed_mps is None:
                raise ValueError(
                    "Schema-v2 response forcing requires relative_crossflow_speed_mps diagnostics."
                )
            relative_crossflow_speed = _node_vector(
                relative_crossflow_speed_mps,
                n_nodes,
                "relative_crossflow_speed_mps",
            )
            spectral_activation = (
                None
                if activation_override is None
                else _node_vector(
                    activation_override,
                    n_nodes,
                    "activation_override",
                )
            )
            if spectral_activation is not None and np.any(spectral_activation < 0.0):
                raise ValueError("activation_override must be non-negative.")
        else:
            if activation_override is not None:
                raise ValueError("activation_override is supported only by response schema_version 2.")
            spectral_activation = None
            # The optional v2-only inputs do not change schema-v1 evaluation
            # or duplicate signatures.
            declared_ambient_crossflow_speed = np.array(
                effective_velocity,
                dtype=float,
                copy=True,
            )
            relative_crossflow_speed = np.array(
                effective_velocity,
                dtype=float,
                copy=True,
            )
        section_width = _node_vector(section_width_m, n_nodes, "section_width_m")
        local_length = _node_vector(local_length_m, n_nodes, "local_length_m")
        density = _finite_scalar(fluid_density_kgpm3, "fluid_density_kgpm3")
        if np.any(effective_velocity < 0.0):
            raise ValueError("effective_velocity_mps must be non-negative.")
        if np.any(declared_ambient_crossflow_speed < 0.0):
            raise ValueError(
                "declared_ambient_crossflow_speed_mps must be non-negative."
            )
        if np.any(relative_crossflow_speed < 0.0):
            raise ValueError("relative_crossflow_speed_mps must be non-negative.")
        if is_schema_v2 and not np.array_equal(
            effective_velocity,
            declared_ambient_crossflow_speed,
        ):
            # Fail before any accepted-time, filter, RMS, or duplicate-cache
            # state changes.  V2 has one authoritative response speed.
            raise ValueError(
                "Schema-v2 effective_velocity_mps must exactly equal "
                "declared_ambient_crossflow_speed_mps."
            )
        if np.any(section_width <= 0.0):
            raise ValueError("section_width_m must be positive.")
        if np.any(local_length < 0.0):
            raise ValueError("local_length_m must be non-negative.")
        if density <= 0.0:
            raise ValueError("fluid_density_kgpm3 must be positive.")
        if is_schema_v2:
            expected_width_basis = self.response_section_width_basis
            width_basis = (
                expected_width_basis
                if response_section_width_basis is None
                else response_section_width_basis
            )
            if not isinstance(width_basis, str) or width_basis != expected_width_basis:
                raise ValueError(
                    "response_section_width_basis must be "
                    f"{expected_width_basis!r} for this response configuration."
                )
            signature = self._input_signature_v2(
                current_time,
                step_node_ids,
                directions,
                velocities,
                declared_ambient_crossflow_speed,
                relative_crossflow_speed,
                section_width,
                local_length,
                density,
                spectral_activation,
            )
        else:
            width_basis = RESPONSE_SECTION_WIDTH_BASIS_RELATIVE_CROSSFLOW
            signature = self._input_signature(
                current_time,
                step_node_ids,
                directions,
                velocities,
                effective_velocity,
                section_width,
                local_length,
                density,
            )
        if self._last_time_s is not None:
            if current_time < self._last_time_s - _TIME_ABS_TOL_S:
                raise ValueError(
                    f"Response-model time decreased from {self._last_time_s:.17g} "
                    f"to {current_time:.17g} s."
                )
            if math.isclose(current_time, self._last_time_s, rel_tol=0.0, abs_tol=_TIME_ABS_TOL_S):
                if signature != self._last_input_signature:
                    raise ValueError("Response-model inputs changed at the same time level.")
                if self._last_result is None:  # pragma: no cover - protected state invariant
                    raise RuntimeError("Response-model duplicate cache is incomplete.")
                return self._copy_result(self._last_result)

        dt_s = 0.0 if self._last_time_s is None else current_time - self._last_time_s
        phi = np.asarray(
            [self._mode_by_node[node_id] for node_id in step_node_ids],
            dtype=float,
        )
        weights = np.asarray(
            [self._weight_by_node[node_id] for node_id in step_node_ids],
            dtype=float,
        )
        measured_directional_velocity = np.sum(velocities * directions, axis=1)
        projection_denominator = float(np.dot(weights * phi, phi))
        raw_qdot_mps = float(
            np.dot(weights * phi, measured_directional_velocity)
            / projection_denominator
        )
        filtered_qdot_mps = self._filter_projected_velocity(raw_qdot_mps, dt_s)

        if dt_s > 0.0:
            decay = math.exp(-dt_s / float(self.config.rms_time_constant_s))
            self._qdot_mean_square_m2ps2 = (
                decay * self._qdot_mean_square_m2ps2
                + (1.0 - decay) * filtered_qdot_mps**2
            )
        qdot_rms_mps = math.sqrt(max(0.0, self._qdot_mean_square_m2ps2))
        amplitude_estimate_m = (
            math.sqrt(2.0) * qdot_rms_mps
            / (2.0 * math.pi * float(self.config.natural_frequency_hz))
        )
        amplitude_ratio = amplitude_estimate_m / float(self.config.reference_length_m)
        saturation_factor = 1.0 - (
            amplitude_ratio / float(self.config.saturation_amplitude_ratio)
        ) ** 2

        response_speed = (
            declared_ambient_crossflow_speed if is_schema_v2 else effective_velocity
        )
        reduced_velocity = response_speed / (
            float(self.config.natural_frequency_hz) * float(self.config.reference_length_m)
        )
        activation = _smooth_reduced_velocity_activation(
            reduced_velocity,
            float(self.config.reduced_velocity_low),
            float(self.config.reduced_velocity_peak),
            float(self.config.reduced_velocity_high),
        )
        if spectral_activation is not None:
            activation = spectral_activation
        configured_response_reduced_velocity = reduced_velocity
        raw_gain = (
            float(self.config.aerodynamic_damping_gain_h1)
            * activation
            * saturation_factor
        )
        gain_limit = float(self.config.max_abs_gain)
        applied_gain = np.clip(raw_gain, -gain_limit, gain_limit)
        gain_limited = np.abs(raw_gain) > gain_limit

        modal_velocity = phi * filtered_qdot_mps
        seed_active = False
        seed_elapsed_cycles = 0.0
        seed_taper = 0.0
        seed_modal_acceleration_mps2 = 0.0
        seed_force_scalar = np.zeros(n_nodes, dtype=float)
        seed_force_limited = np.zeros(n_nodes, dtype=bool)
        if self.config.enabled:
            aerodynamic_force_scalar = (
                0.5
                * density
                * response_speed
                * section_width
                * local_length
                * applied_gain
                * modal_velocity
            )
            force_limit = float(self.config.max_abs_nodal_force_n)
            if self._startup_seed_enabled:
                (
                    seed_active,
                    seed_elapsed_cycles,
                    seed_taper,
                    seed_modal_acceleration_mps2,
                ) = self._startup_seed_modal_acceleration(current_time)
                seed_force_scalar = (
                    weights * phi * seed_modal_acceleration_mps2
                )
                seed_force_limited = np.abs(seed_force_scalar) > force_limit
                raw_total_force_scalar = (
                    aerodynamic_force_scalar + seed_force_scalar
                )
                force_limited = np.abs(raw_total_force_scalar) > force_limit
                force_scalar = np.clip(
                    raw_total_force_scalar,
                    -force_limit,
                    force_limit,
                )
            else:
                # Retain the exact legacy arithmetic path when no seed is
                # configured; disabled-by-default means zero output drift.
                force_limited = np.abs(aerodynamic_force_scalar) > force_limit
                force_scalar = np.clip(
                    aerodynamic_force_scalar,
                    -force_limit,
                    force_limit,
                )
        else:
            force_scalar = np.zeros(n_nodes, dtype=float)
            force_limited = np.zeros(n_nodes, dtype=bool)
        additive_force = force_scalar[:, None] * directions

        diagnostics_kwargs: dict[str, Any] = dict(
            enabled=bool(self.config.enabled),
            time_s=current_time,
            dt_s=dt_s,
            reduced_velocity_min=float(np.min(reduced_velocity)),
            reduced_velocity_max=float(np.max(reduced_velocity)),
            activation_min=float(np.min(activation)),
            activation_max=float(np.max(activation)),
            raw_qdot_mps=raw_qdot_mps,
            filtered_qdot_mps=filtered_qdot_mps,
            qdot_mps=filtered_qdot_mps,
            qdot_rms_mps=qdot_rms_mps,
            projection_weighting=self._projection_weighting,
            projection_weight_min=float(np.min(weights)),
            projection_weight_max=float(np.max(weights)),
            amplitude_estimate_m=amplitude_estimate_m,
            amplitude_ratio=amplitude_ratio,
            saturation_factor=saturation_factor,
            raw_gain_min=float(np.min(raw_gain)),
            raw_gain_max=float(np.max(raw_gain)),
            applied_gain_min=float(np.min(applied_gain)),
            applied_gain_max=float(np.max(applied_gain)),
            gain_limited_nodes=int(np.count_nonzero(gain_limited)),
            force_limited_nodes=int(np.count_nonzero(force_limited)),
            modal_power_w=float(np.dot(force_scalar, modal_velocity)),
            structural_power_w=float(np.sum(additive_force * velocities)),
            max_abs_nodal_force_n=float(np.max(np.abs(force_scalar))),
        )
        diagnostics_v2_kwargs: dict[str, Any] = {}
        if is_schema_v2:
            diagnostics_v2_kwargs = {
                "response_model_schema_version": self.schema_version,
                "aerodynamic_speed_basis": self.aerodynamic_speed_basis,
                "response_declared_ambient_crossflow_speed_min_mps": float(
                    np.min(declared_ambient_crossflow_speed)
                ),
                "response_declared_ambient_crossflow_speed_max_mps": float(
                    np.max(declared_ambient_crossflow_speed)
                ),
                "response_relative_crossflow_speed_min_mps": float(
                    np.min(relative_crossflow_speed)
                ),
                "response_relative_crossflow_speed_max_mps": float(
                    np.max(relative_crossflow_speed)
                ),
                "response_section_width_basis": width_basis,
                "response_section_width_min_m": float(np.min(section_width)),
                "response_section_width_max_m": float(np.max(section_width)),
                "configured_response_reduced_velocity_min": float(
                    np.min(configured_response_reduced_velocity)
                ),
                "configured_response_reduced_velocity_max": float(
                    np.max(configured_response_reduced_velocity)
                ),
                "realized_response_reduced_velocity_min": float(
                    np.min(reduced_velocity)
                ),
                "realized_response_reduced_velocity_max": float(
                    np.max(reduced_velocity)
                ),
                "speed_basis_invariant_violations": 0,
                "activation_invariant_violations": 0,
            }
        if self._startup_seed_enabled:
            seed = self.config.startup_seed
            if seed is None:  # pragma: no cover - protected state invariant
                raise RuntimeError("Enabled startup seed configuration is missing.")
            diagnostics_type = (
                QBladeSeededResponseDiagnosticsV2
                if is_schema_v2
                else QBladeSeededResponseDiagnostics
            )
            diagnostics = diagnostics_type(
                **diagnostics_kwargs,
                **diagnostics_v2_kwargs,
                startup_seed_enabled=True,
                startup_seed_active=seed_active,
                startup_seed_elapsed_cycles=seed_elapsed_cycles,
                startup_seed_duration_cycles=float(seed.duration_cycles),
                startup_seed_duration_s=self._startup_seed_duration_s,
                startup_seed_taper=seed_taper,
                startup_seed_target_amplitude_ratio=float(
                    seed.target_amplitude_ratio
                ),
                startup_seed_target_amplitude_m=(
                    self._startup_seed_target_amplitude_m
                ),
                startup_seed_ideal_terminal_amplitude_m=(
                    self._startup_seed_ideal_terminal_amplitude_m
                ),
                startup_seed_modal_acceleration_scale_mps2=(
                    self._startup_seed_modal_acceleration_scale_mps2
                ),
                startup_seed_modal_acceleration_mps2=(
                    seed_modal_acceleration_mps2
                ),
                startup_seed_raw_generalized_force_n=float(
                    np.dot(phi, seed_force_scalar)
                ),
                startup_seed_raw_max_abs_nodal_force_n=float(
                    np.max(np.abs(seed_force_scalar))
                ),
                startup_seed_force_limited_nodes=int(
                    np.count_nonzero(seed_force_limited)
                ),
                startup_seed_modal_power_w=float(
                    np.dot(seed_force_scalar, modal_velocity)
                ),
            )
        else:
            diagnostics_type = (
                QBladeResponseDiagnosticsV2
                if is_schema_v2
                else QBladeResponseDiagnostics
            )
            diagnostics = diagnostics_type(
                **diagnostics_kwargs,
                **diagnostics_v2_kwargs,
            )
        result = QBladeResponseResult(
            additive_force_n=np.array(additive_force, dtype=float, copy=True),
            diagnostics=diagnostics,
        )
        self._last_time_s = current_time
        self._last_input_signature = signature
        self._last_result = self._copy_result(result)
        return self._copy_result(result)

    @staticmethod
    def _copy_result(result: QBladeResponseResult) -> QBladeResponseResult:
        return QBladeResponseResult(
            additive_force_n=np.array(result.additive_force_n, dtype=float, copy=True),
            diagnostics=result.diagnostics,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        """Return JSON-serializable state sufficient for exact restart/retry."""

        cached_result: dict[str, Any] | None = None
        if self._last_result is not None:
            cached_result = {
                "additive_force_n": self._last_result.additive_force_n.tolist(),
                "diagnostics": asdict(self._last_result.diagnostics),
            }
        state = {
            "schema": _CHECKPOINT_SCHEMA,
            "config_fingerprint": self._config_fingerprint,
            "last_time_s": self._last_time_s,
            "bandpass_filter": {
                "position_state": self._filter_position_state,
                "rate_state": self._filter_rate_state,
                "previous_input_mps": self._filter_previous_input_mps,
            },
            "qdot_mean_square_m2ps2": self._qdot_mean_square_m2ps2,
            "last_input_signature": self._last_input_signature,
            "last_result": cached_result,
        }
        if (
            self.schema_version
            == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        ):
            state.update(
                {
                    "response_model_schema_version": self.schema_version,
                    "aerodynamic_speed_basis": self.aerodynamic_speed_basis,
                    "response_section_width_basis": (
                        self.response_section_width_basis
                    ),
                }
            )
        if self._startup_seed_enabled:
            diagnostics = (
                None if self._last_result is None else self._last_result.diagnostics
            )
            if diagnostics is not None and not isinstance(
                diagnostics,
                (
                    QBladeSeededResponseDiagnostics,
                    QBladeSeededResponseDiagnosticsV2,
                ),
            ):
                raise RuntimeError(
                    "Seeded response-model cache lacks startup-seed diagnostics."
                )
            state["startup_seed"] = {
                "schema_version": _STARTUP_SEED_SCHEMA_VERSION,
                "waveform": (
                    "sin(pi*t/T)^2*sin(2*pi*natural_frequency_hz*t)"
                ),
                "duration_s": self._startup_seed_duration_s,
                "target_amplitude_m": self._startup_seed_target_amplitude_m,
                "unit_terminal_amplitude_m_per_mps2": (
                    self._startup_seed_unit_terminal_amplitude_m_per_mps2
                ),
                "modal_acceleration_scale_mps2": (
                    self._startup_seed_modal_acceleration_scale_mps2
                ),
                "ideal_terminal_amplitude_m": (
                    self._startup_seed_ideal_terminal_amplitude_m
                ),
                "last_active": (
                    None
                    if diagnostics is None
                    else diagnostics.startup_seed_active
                ),
                "last_modal_acceleration_mps2": (
                    None
                    if diagnostics is None
                    else diagnostics.startup_seed_modal_acceleration_mps2
                ),
            }
        return state

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore state from :meth:`checkpoint_state` after strict validation."""

        if state.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported QBlade response-model checkpoint schema.")
        if state.get("config_fingerprint") != self._config_fingerprint:
            raise ValueError("QBlade response-model checkpoint configuration mismatch.")
        if (
            self.schema_version
            == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        ):
            if state.get("response_model_schema_version") != self.schema_version:
                raise ValueError(
                    "QBlade response-model checkpoint schema-version mismatch."
                )
            if state.get("aerodynamic_speed_basis") != self.aerodynamic_speed_basis:
                raise ValueError("QBlade response-model checkpoint speed-basis mismatch.")
            if (
                state.get("response_section_width_basis")
                != self.response_section_width_basis
            ):
                raise ValueError("QBlade response-model checkpoint width-basis mismatch.")

        last_time_raw = state.get("last_time_s")
        last_time = None if last_time_raw is None else _finite_scalar(last_time_raw, "last_time_s")
        filter_raw = state.get("bandpass_filter")
        if not isinstance(filter_raw, Mapping):
            raise ValueError("bandpass_filter must be a mapping.")
        filter_position = _finite_scalar(
            filter_raw.get("position_state"),
            "bandpass_filter.position_state",
        )
        filter_rate = _finite_scalar(
            filter_raw.get("rate_state"),
            "bandpass_filter.rate_state",
        )
        filter_input_raw = filter_raw.get("previous_input_mps")
        filter_previous_input = (
            None
            if filter_input_raw is None
            else _finite_scalar(
                filter_input_raw,
                "bandpass_filter.previous_input_mps",
            )
        )
        mean_square = _finite_scalar(
            state.get("qdot_mean_square_m2ps2", 0.0),
            "qdot_mean_square_m2ps2",
        )
        if mean_square < 0.0:
            raise ValueError("qdot_mean_square_m2ps2 must be non-negative.")
        signature_raw = state.get("last_input_signature")
        signature = None if signature_raw is None else str(signature_raw)

        cached_raw = state.get("last_result")
        cached_result: QBladeResponseResult | None = None
        if cached_raw is not None:
            if not isinstance(cached_raw, Mapping):
                raise ValueError("last_result must be a mapping or null.")
            force = np.asarray(cached_raw.get("additive_force_n"), dtype=float)
            if force.shape != (len(self._node_ids), 3) or not np.all(np.isfinite(force)):
                raise ValueError("Checkpoint additive_force_n has an invalid shape or value.")
            diagnostics_raw = cached_raw.get("diagnostics")
            if not isinstance(diagnostics_raw, Mapping):
                raise ValueError("Checkpoint diagnostics must be a mapping.")
            try:
                diagnostics_values = dict(diagnostics_raw)
                if (
                    self.schema_version
                    == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
                ):
                    diagnostics_type = (
                        QBladeSeededResponseDiagnosticsV2
                        if "startup_seed_enabled" in diagnostics_values
                        else QBladeResponseDiagnosticsV2
                    )
                else:
                    diagnostics_type = (
                        QBladeSeededResponseDiagnostics
                        if "startup_seed_enabled" in diagnostics_values
                        else QBladeResponseDiagnostics
                    )
                diagnostics = diagnostics_type(**diagnostics_values)
            except (TypeError, ValueError) as exc:
                raise ValueError("Checkpoint response diagnostics are invalid.") from exc
            cached_result = QBladeResponseResult(
                additive_force_n=np.array(force, dtype=float, copy=True),
                diagnostics=diagnostics,
            )

        if last_time is None:
            if signature is not None or cached_result is not None:
                raise ValueError("An unstarted checkpoint cannot contain a duplicate cache.")
            if filter_previous_input is not None:
                raise ValueError("An unstarted checkpoint cannot contain filter input state.")
            if filter_position != 0.0 or filter_rate != 0.0 or mean_square != 0.0:
                raise ValueError("An unstarted checkpoint must contain zero dynamic state.")
        elif signature is None or cached_result is None:
            raise ValueError("A started checkpoint must contain the complete duplicate cache.")
        elif filter_previous_input is None:
            raise ValueError("A started checkpoint must contain filter input state.")

        seed_state = state.get("startup_seed")
        if self._startup_seed_enabled:
            if not isinstance(seed_state, Mapping):
                raise ValueError(
                    "Enabled startup seed requires checkpoint startup_seed state."
                )
            if (
                isinstance(seed_state.get("schema_version"), bool)
                or seed_state.get("schema_version") != _STARTUP_SEED_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Unsupported response-model startup-seed checkpoint schema."
                )
            if seed_state.get("waveform") != (
                "sin(pi*t/T)^2*sin(2*pi*natural_frequency_hz*t)"
            ):
                raise ValueError("Checkpoint startup-seed waveform mismatch.")
            expected_seed_scalars = {
                "duration_s": self._startup_seed_duration_s,
                "target_amplitude_m": self._startup_seed_target_amplitude_m,
                "unit_terminal_amplitude_m_per_mps2": (
                    self._startup_seed_unit_terminal_amplitude_m_per_mps2
                ),
                "modal_acceleration_scale_mps2": (
                    self._startup_seed_modal_acceleration_scale_mps2
                ),
                "ideal_terminal_amplitude_m": (
                    self._startup_seed_ideal_terminal_amplitude_m
                ),
            }
            for name, expected in expected_seed_scalars.items():
                actual = _finite_scalar(
                    seed_state.get(name),
                    f"startup_seed.{name}",
                )
                if actual != expected:
                    raise ValueError(
                        f"Checkpoint startup-seed {name} mismatch."
                    )
            last_active = seed_state.get("last_active")
            last_acceleration = seed_state.get("last_modal_acceleration_mps2")
            if last_time is None:
                if last_active is not None or last_acceleration is not None:
                    raise ValueError(
                        "Unstarted startup-seed checkpoint cannot contain last values."
                    )
            else:
                if not isinstance(last_active, bool):
                    raise ValueError(
                        "Started startup-seed checkpoint requires boolean last_active."
                    )
                last_acceleration_value = _finite_scalar(
                    last_acceleration,
                    "startup_seed.last_modal_acceleration_mps2",
                )
                if not isinstance(
                    cached_result.diagnostics,
                    (
                        QBladeSeededResponseDiagnostics,
                        QBladeSeededResponseDiagnosticsV2,
                    ),
                ):
                    raise ValueError(
                        "Seeded checkpoint requires startup-seed response diagnostics."
                    )
                if (
                    last_active
                    != cached_result.diagnostics.startup_seed_active
                    or last_acceleration_value
                    != cached_result.diagnostics.startup_seed_modal_acceleration_mps2
                ):
                    raise ValueError(
                        "Checkpoint startup-seed last values do not match cached diagnostics."
                    )
        elif seed_state is not None:
            raise ValueError(
                "Checkpoint contains startup-seed state for an unseeded model."
            )

        self._last_time_s = last_time
        self._filter_position_state = filter_position
        self._filter_rate_state = filter_rate
        self._filter_previous_input_mps = filter_previous_input
        self._filter_transition_cache = {}
        self._qdot_mean_square_m2ps2 = mean_square
        self._last_input_signature = signature
        self._last_result = cached_result


__all__ = [
    "QBladeResponseDiagnostics",
    "QBladeResponseDiagnosticsV2",
    "QBladeResponseModel",
    "QBladeResponseModelConfig",
    "QBladeResponseResult",
    "QBladeSeededResponseDiagnostics",
    "QBladeSeededResponseDiagnosticsV2",
    "QBladeStartupSeedConfig",
]
