"""Response-aware one-mode aerodynamic-power closure for parked blade VIV.

This module implements a deliberately bounded reduced-order model for the
Grinderslev P100--I50 comparison.  It is a *calibration-based* closure: the
cycle-mean aerodynamic power is read from a forced-motion power curve and is
not predicted from VorLap, CFD, or a wake oscillator.  Consequently, matching
the calibration curve is not independent CFD verification, physical
validation, or certification evidence.

Let ``phi_i`` be a dimensionless mode shape normalized to unit maximum,
``m_i`` positive nodal mass/quadrature weights, ``v_i`` the structural
velocity in the declared force direction, and ``a_i`` non-negative
aerodynamic tributary weights.  The raw generalized velocity is

``qdot_raw = sum(m_i phi_i v_i) / sum(m_i phi_i**2)``.

It passes through the narrow-band filter

``H(s) = 2 zeta_b omega_n s / (s**2 + 2 zeta_b omega_n s + omega_n**2)``

before being used by the closure.  This rejects rigid drift (zero DC gain),
has unit gain and zero phase at ``omega_n``, and limits the feedback to the
calibrated mode.  An exponential mean square of the filtered velocity gives

``A = sqrt(2 <qdot_f**2>) / omega_n``.

For the interpolated cycle-mean power ``P_aero(A)``, the signed generalized
velocity gain and force are

``c_aero = 2 P_aero(A) / (omega_n**2 A**2)``,
``Q_aero = c_aero qdot_f``.

Positive power therefore injects structural energy and negative power removes
it.  The first table interval uses the explicit regularization
``P(A)=P_1(A/A_1)**2`` so the small-amplitude gain is finite.  All later
intervals are linear.  Amplitudes outside the tabulated domain are rejected.

The nodal scalar force is distributed as

``F_i = Q_aero a_i phi_i / sum(a_j phi_j**2)``,

which guarantees ``sum(phi_i F_i)=Q_aero`` before an explicitly diagnosed
limit.  The overall scale of ``a_i`` is immaterial.  The model is disabled by
default.  Default limit handling raises instead of silently clipping; an
explicit ``limit_action='clip'`` is available for diagnostic sensitivity runs
and reports every active limit and the resulting power shortfall.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

_CHECKPOINT_SCHEMA = "vorlap-qblade-modal-power-model-v1"
_TIME_ABS_TOL_S = 1.0e-12
_FLOAT_ABS_TOL = 1.0e-12


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _finite_tuple(values: Sequence[float], name: str) -> Tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric sequence.") from exc
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be a non-empty finite numeric sequence.")
    return result


@dataclass(frozen=True)
class TabulatedAerodynamicPowerCurve:
    """Tip-amplitude versus cycle-mean aerodynamic power in SI units.

    The curve must start at exactly ``(0 m, 0 W)``, initially inject energy,
    and contain a negative-power point.  Those checks are intentional for this
    self-excited, amplitude-limited P100--I50 closure: a table without a
    stabilizing branch cannot define the selected limit-cycle comparison.
    """

    amplitude_m: Tuple[float, ...]
    power_w: Tuple[float, ...]
    source_label: str = "calibrated forced-motion power curve"
    source_sha256: str = ""

    def __post_init__(self) -> None:
        amplitude = _finite_tuple(self.amplitude_m, "amplitude_m")
        power = _finite_tuple(self.power_w, "power_w")
        object.__setattr__(self, "amplitude_m", amplitude)
        object.__setattr__(self, "power_w", power)
        source_label = str(self.source_label).strip()
        object.__setattr__(self, "source_label", source_label)
        source_sha256 = str(self.source_sha256).strip().lower()
        object.__setattr__(self, "source_sha256", source_sha256)

        if len(amplitude) != len(power):
            raise ValueError("amplitude_m and power_w must have equal length.")
        if len(amplitude) < 3:
            raise ValueError(
                "The aerodynamic-power table requires at least three points."
            )
        if amplitude[0] != 0.0 or power[0] != 0.0:
            raise ValueError("The aerodynamic-power table must start at (0 m, 0 W).")
        if any(value < 0.0 for value in amplitude):
            raise ValueError("Aerodynamic-power amplitudes must be non-negative.")
        if any(right <= left for left, right in zip(amplitude[:-1], amplitude[1:])):
            raise ValueError(
                "Aerodynamic-power amplitudes must be strictly increasing."
            )
        if power[1] <= 0.0 or not any(value > 0.0 for value in power[1:]):
            raise ValueError(
                "The aerodynamic-power table must initially inject energy."
            )
        if not any(value < 0.0 for value in power[1:]):
            raise ValueError(
                "The aerodynamic-power table requires a negative-power stabilizing point."
            )
        if not source_label:
            raise ValueError("source_label must not be empty.")
        if source_sha256 and (
            len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise ValueError(
                "source_sha256 must be empty or a 64-character hex digest."
            )

    @classmethod
    def from_csv(
        cls,
        path: Any,
        *,
        expected_series: Optional[str] = None,
    ) -> "TabulatedAerodynamicPowerCurve":
        """Load one curve from a strict figure-digitization CSV.

        Required columns are ``tip_amplitude_m`` and ``aerodynamic_power_w``.
        If a ``series`` column is present, all rows must name one series; an
        ``expected_series`` additionally pins that identifier.  Malformed or
        non-finite rows are rejected rather than skipped.
        """

        csv_path = Path(path)
        amplitude = []
        power = []
        series_values = set()
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"tip_amplitude_m", "aerodynamic_power_w"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    "Power-curve CSV must contain tip_amplitude_m and "
                    "aerodynamic_power_w columns."
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    amplitude.append(
                        _finite_scalar(
                            row["tip_amplitude_m"], f"row {row_number} amplitude"
                        )
                    )
                    power.append(
                        _finite_scalar(
                            row["aerodynamic_power_w"], f"row {row_number} power"
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Malformed aerodynamic-power CSV row {row_number}."
                    ) from exc
                if "series" in row:
                    series_values.add(str(row["series"]).strip())

        if not amplitude:
            raise ValueError("Power-curve CSV contains no data rows.")
        if len(series_values) > 1:
            raise ValueError("Power-curve CSV mixes more than one series.")
        loaded_series = next(iter(series_values), "")
        if expected_series is not None and loaded_series != str(expected_series):
            raise ValueError(
                f"Power-curve series {loaded_series!r} does not match "
                f"{str(expected_series)!r}."
            )
        label = loaded_series or csv_path.stem
        source_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        return cls(
            tuple(amplitude),
            tuple(power),
            source_label=label,
            source_sha256=source_sha256,
        )

    @property
    def maximum_amplitude_m(self) -> float:
        """Largest calibrated amplitude; extrapolation is not permitted."""

        return self.amplitude_m[-1]

    def evaluate(self, amplitude_m: float) -> float:
        """Return interpolated cycle-mean power [W] at a non-negative amplitude."""

        amplitude = _finite_scalar(amplitude_m, "amplitude_m")
        if amplitude < 0.0:
            raise ValueError("amplitude_m must be non-negative.")
        if amplitude > self.maximum_amplitude_m + _FLOAT_ABS_TOL:
            raise ValueError(
                f"Amplitude {amplitude:.17g} m exceeds the calibrated power-curve "
                f"domain ending at {self.maximum_amplitude_m:.17g} m."
            )
        amplitude = min(amplitude, self.maximum_amplitude_m)
        first_amplitude = self.amplitude_m[1]
        if amplitude <= first_amplitude:
            value = self.power_w[1] * (amplitude / first_amplitude) ** 2
        else:
            value = float(np.interp(amplitude, self.amplitude_m, self.power_w))
        if not math.isfinite(value):  # pragma: no cover - constructor guards inputs
            raise FloatingPointError(
                "Aerodynamic-power interpolation became non-finite."
            )
        return value

    def equivalent_velocity_gain(self, amplitude_m: float, omega_radps: float) -> float:
        """Return ``2 P(A)/(omega**2 A**2)`` [kg/s] with a finite origin limit."""

        amplitude = _finite_scalar(amplitude_m, "amplitude_m")
        omega = _finite_scalar(omega_radps, "omega_radps")
        if amplitude < 0.0 or omega <= 0.0:
            raise ValueError(
                "amplitude_m must be non-negative and omega_radps positive."
            )
        if amplitude <= _FLOAT_ABS_TOL:
            first_amplitude = self.amplitude_m[1]
            gain = 2.0 * self.power_w[1] / (omega**2 * first_amplitude**2)
        else:
            gain = 2.0 * self.evaluate(amplitude) / (omega**2 * amplitude**2)
        if not math.isfinite(gain):
            raise FloatingPointError(
                "Equivalent aerodynamic velocity gain is non-finite."
            )
        return gain

    def positive_zero_crossings_m(self) -> Tuple[float, ...]:
        """Return positive zero-power crossings within the calibrated domain."""

        roots = []
        for left_a, right_a, left_p, right_p in zip(
            self.amplitude_m[:-1],
            self.amplitude_m[1:],
            self.power_w[:-1],
            self.power_w[1:],
        ):
            if left_a > 0.0 and left_p == 0.0:
                roots.append(left_a)
            if left_p * right_p < 0.0:
                roots.append(left_a - left_p * (right_a - left_a) / (right_p - left_p))
            if right_p == 0.0 and right_a > 0.0:
                roots.append(right_a)
        return _deduplicate_sorted(roots)

    def structural_equilibria_m(
        self,
        structural_power_coefficient_wpm2: float,
    ) -> Tuple[float, ...]:
        """Return nonzero roots of ``P_aero(A) = C A**2`` within the table."""

        coefficient = _finite_scalar(
            structural_power_coefficient_wpm2,
            "structural_power_coefficient_wpm2",
        )
        if coefficient < 0.0:
            raise ValueError("structural_power_coefficient_wpm2 must be non-negative.")

        def residual(amplitude: float) -> float:
            return self.evaluate(amplitude) - coefficient * amplitude**2

        roots = []
        for left, right in zip(self.amplitude_m[:-1], self.amplitude_m[1:]):
            left_value = residual(left)
            right_value = residual(right)
            if left > 0.0 and left_value == 0.0:
                roots.append(left)
            if left_value * right_value < 0.0:
                roots.append(brentq(residual, left, right, xtol=1.0e-13, rtol=1.0e-13))
            if right > 0.0 and right_value == 0.0:
                roots.append(right)
        return _deduplicate_sorted(roots)


def _deduplicate_sorted(values: Sequence[float]) -> Tuple[float, ...]:
    result = []
    for value in sorted(float(item) for item in values):
        if not result or not math.isclose(
            value,
            result[-1],
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class QBladeModalPowerModelConfig:
    """Configuration for the one-mode tabulated-power closure.

    ``modal_mass_weights_kg`` are required positive mass/quadrature weights;
    there is no uniform fallback for qualification work.  The
    ``aerodynamic_tributary_weights`` may be proportional to tributary span,
    area, or another declared aerodynamic quadrature, and may contain zeros as
    long as the modal distribution denominator is positive.

    All three force/power limits are finite and positive.  ``limit_action`` is
    ``"error"`` by default.  Selecting ``"clip"`` is explicit permission for
    a diagnostic clipped run; diagnostics retain raw and applied quantities.
    ``force_direction`` is metadata used by bridge integration and is expected
    to be ``"chord"`` for the first-edgewise P100--I50 comparison.
    """

    node_ids: Tuple[str, ...]
    mode_shape: Tuple[float, ...]
    modal_mass_weights_kg: Tuple[float, ...]
    aerodynamic_tributary_weights: Tuple[float, ...]
    natural_frequency_hz: float
    power_curve: Optional[TabulatedAerodynamicPowerCurve]
    rms_time_constant_s: float
    max_abs_target_power_w: float
    max_abs_generalized_force_n: float
    max_abs_nodal_force_n: float
    bandpass_damping_ratio: float = 0.1
    force_direction: str = "chord"
    limit_action: str = "error"
    enabled: bool = False
    external_target_source_fingerprint: str = ""


@dataclass(frozen=True)
class QBladeModalPowerDiagnostics:
    """Scalar diagnostics for one accepted model evaluation.

    ``target_cycle_mean_power_w`` is the raw table value.  The applied target
    can differ only under explicit clipping.  Realized modal and structural
    power are reported both instantaneously and as exponential means using the
    amplitude estimator's time constant. ``instantaneous_modal_power_w`` is
    the controller-state quantity ``Q*qdot_filtered``;
    ``instantaneous_structural_power_w`` is the exact realized nodal work
    ``sum(F_i dot v_i)``. Positive power adds structural energy; negative
    power removes it.
    """

    enabled: bool
    time_s: float
    dt_s: float
    raw_qdot_mps: float
    filtered_qdot_mps: float
    qdot_rms_mps: float
    amplitude_estimate_m: float
    target_cycle_mean_power_w: float
    applied_target_cycle_mean_power_w: float
    equivalent_velocity_gain_kgps: float
    raw_generalized_force_n: float
    applied_generalized_force_n: float
    force_distribution_denominator: float
    force_scale: float
    target_power_limited: bool
    generalized_force_limited: bool
    nodal_force_limited_nodes: int
    instantaneous_modal_power_w: float
    instantaneous_structural_power_w: float
    realized_modal_power_ema_w: float
    realized_structural_power_ema_w: float
    max_abs_nodal_force_n: float
    projection_weight_min_kg: float
    projection_weight_max_kg: float
    aerodynamic_weight_min: float
    aerodynamic_weight_max: float
    dynamic_conjugate_weight_override: bool = False
    dynamic_conjugate_weight_min: float = 0.0
    dynamic_conjugate_weight_max: float = 0.0
    instantaneous_controller_state_power_w: float = 0.0
    instantaneous_realized_work_w: float = 0.0
    controller_state_power_ema_w: float = 0.0
    realized_work_ema_w: float = 0.0


@dataclass(frozen=True)
class QBladeModalPowerResult:
    """Additive global nodal force [N] and typed diagnostics."""

    additive_force_n: np.ndarray
    diagnostics: QBladeModalPowerDiagnostics


@dataclass(frozen=True)
class ExternalModalPowerTarget:
    """Externally evaluated power target for the energy-conjugate actuator.

    ``origin_equivalent_velocity_gain_kgps`` is the finite analytic
    ``A -> 0`` limit of ``2P/(omega**2 A**2)``.  ``context_fingerprint`` must
    bind every map query input not already present in the modal step arrays.
    """

    target_cycle_mean_power_w: float
    origin_equivalent_velocity_gain_kgps: float
    context_fingerprint: str


class QBladeModalPowerLimitError(RuntimeError):
    """Raised when a configured fail-closed power or force limit is exceeded."""

    def __init__(self, metric: str, raw_value: float, limit: float):
        self.metric = str(metric)
        self.raw_value = float(raw_value)
        self.limit = float(limit)
        super().__init__(
            f"Modal-power {self.metric} limit exceeded: "
            f"|{self.raw_value:.17g}| > {self.limit:.17g}."
        )


class QBladeModalPowerModel:
    """Stateful, idempotent one-mode aerodynamic-power closure.

    Call :meth:`step` once per accepted structural time level.  Exact retries
    are idempotent.  Inputs are canonicalized by node ID, so configuration and
    call ordering do not change the projection, force, checkpoint identity, or
    returned physics.  Results are always returned in the current call order.
    """

    def __init__(self, config: QBladeModalPowerModelConfig):
        self.config = config
        self._validate_config(config)

        raw_ids = tuple(str(node_id) for node_id in config.node_ids)
        self._canonical_node_ids = tuple(sorted(raw_ids))
        raw_mode_by_node = dict(zip(raw_ids, (float(v) for v in config.mode_shape)))
        maximum_mode = max(abs(value) for value in raw_mode_by_node.values())
        normalized_by_node = {
            node_id: value / maximum_mode for node_id, value in raw_mode_by_node.items()
        }
        first_nonzero = next(
            normalized_by_node[node_id]
            for node_id in self._canonical_node_ids
            if normalized_by_node[node_id] != 0.0
        )
        orientation = math.copysign(1.0, first_nonzero)
        self._mode_shape = np.asarray(
            [
                normalized_by_node[node_id] * orientation
                for node_id in self._canonical_node_ids
            ],
            dtype=float,
        )

        mass_by_node = dict(
            zip(raw_ids, (float(v) for v in config.modal_mass_weights_kg))
        )
        aero_by_node = dict(
            zip(raw_ids, (float(v) for v in config.aerodynamic_tributary_weights))
        )
        self._modal_mass_weights_kg = np.asarray(
            [mass_by_node[node_id] for node_id in self._canonical_node_ids],
            dtype=float,
        )
        self._aerodynamic_weights = np.asarray(
            [aero_by_node[node_id] for node_id in self._canonical_node_ids],
            dtype=float,
        )
        self._projection_denominator = float(
            np.dot(self._modal_mass_weights_kg * self._mode_shape, self._mode_shape)
        )
        self._distribution_denominator = float(
            np.dot(self._aerodynamic_weights * self._mode_shape, self._mode_shape)
        )
        if self._distribution_denominator <= 0.0:
            raise ValueError(
                "aerodynamic_tributary_weights must overlap a nonzero mode-shape value."
            )

        self._config_fingerprint = self._fingerprint_config()
        self._transition_cache = {}
        self._last_time_s = None  # type: Optional[float]
        self._filter_position_state = 0.0
        self._filter_rate_state = 0.0
        self._filter_previous_input_mps = None  # type: Optional[float]
        self._qdot_mean_square_m2ps2 = 0.0
        self._modal_power_ema_w = 0.0
        self._structural_power_ema_w = 0.0
        self._last_input_signature = None  # type: Optional[str]
        self._last_force_canonical_n = None  # type: Optional[np.ndarray]
        self._last_diagnostics = None  # type: Optional[QBladeModalPowerDiagnostics]

    @property
    def last_diagnostics(self) -> Optional[QBladeModalPowerDiagnostics]:
        """Last accepted diagnostics, or ``None`` before the first step."""

        return self._last_diagnostics

    @staticmethod
    def _validate_config(config: QBladeModalPowerModelConfig) -> None:
        node_ids = tuple(str(node_id) for node_id in config.node_ids)
        if not node_ids or any(not node_id for node_id in node_ids):
            raise ValueError("node_ids must contain at least one non-empty node ID.")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be unique.")
        n_nodes = len(node_ids)

        mode = np.asarray(config.mode_shape, dtype=float)
        mass = np.asarray(config.modal_mass_weights_kg, dtype=float)
        aero = np.asarray(config.aerodynamic_tributary_weights, dtype=float)
        if mode.shape != (n_nodes,) or not np.all(np.isfinite(mode)):
            raise ValueError("mode_shape must provide one finite scalar per node_id.")
        if float(np.max(np.abs(mode))) <= 0.0:
            raise ValueError("mode_shape must contain at least one nonzero value.")
        if (
            mass.shape != (n_nodes,)
            or np.any(~np.isfinite(mass))
            or np.any(mass <= 0.0)
        ):
            raise ValueError(
                "modal_mass_weights_kg must provide one finite positive value per node_id."
            )
        if aero.shape != (n_nodes,) or np.any(~np.isfinite(aero)) or np.any(aero < 0.0):
            raise ValueError(
                "aerodynamic_tributary_weights must provide one finite non-negative "
                "value per node_id."
            )
        external_fingerprint = str(config.external_target_source_fingerprint).lower()
        if config.power_curve is None:
            if len(external_fingerprint) != 64 or any(
                character not in "0123456789abcdef"
                for character in external_fingerprint
            ):
                raise ValueError(
                    "An external-target modal-power model requires a SHA-256 "
                    "external_target_source_fingerprint."
                )
        elif not isinstance(config.power_curve, TabulatedAerodynamicPowerCurve):
            raise ValueError(
                "power_curve must be a TabulatedAerodynamicPowerCurve or null."
            )
        elif external_fingerprint:
            raise ValueError(
                "A tabulated power curve and an external target source are mutually exclusive."
            )

        positive_fields = {
            "natural_frequency_hz": config.natural_frequency_hz,
            "rms_time_constant_s": config.rms_time_constant_s,
            "max_abs_target_power_w": config.max_abs_target_power_w,
            "max_abs_generalized_force_n": config.max_abs_generalized_force_n,
            "max_abs_nodal_force_n": config.max_abs_nodal_force_n,
            "bandpass_damping_ratio": config.bandpass_damping_ratio,
        }
        for name, raw_value in positive_fields.items():
            value = _finite_scalar(raw_value, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        damping = float(config.bandpass_damping_ratio)
        if damping >= 1.0:
            raise ValueError("bandpass_damping_ratio must be between zero and one.")
        if str(config.force_direction) not in {"normal", "chord"}:
            raise ValueError("force_direction must be either 'normal' or 'chord'.")
        if str(config.limit_action) not in {"error", "clip"}:
            raise ValueError("limit_action must be either 'error' or 'clip'.")
        if not isinstance(config.enabled, bool):
            raise TypeError("enabled must be a boolean.")

    def _fingerprint_config(self) -> str:
        node_records = [
            {
                "node_id": node_id,
                "mode": float(mode),
                "modal_mass_weight_kg": float(mass),
                "aerodynamic_weight": float(aero),
            }
            for node_id, mode, mass, aero in zip(
                self._canonical_node_ids,
                self._mode_shape,
                self._modal_mass_weights_kg,
                self._aerodynamic_weights,
            )
        ]
        payload = {
            "nodes": node_records,
            "natural_frequency_hz": float(self.config.natural_frequency_hz),
            "power_curve_amplitude_m": (
                None
                if self.config.power_curve is None
                else list(self.config.power_curve.amplitude_m)
            ),
            "power_curve_power_w": (
                None
                if self.config.power_curve is None
                else list(self.config.power_curve.power_w)
            ),
            "power_curve_source_label": (
                None
                if self.config.power_curve is None
                else self.config.power_curve.source_label
            ),
            "power_curve_source_sha256": (
                None
                if self.config.power_curve is None
                else self.config.power_curve.source_sha256
            ),
            "external_target_source_fingerprint": str(
                self.config.external_target_source_fingerprint
            ).lower(),
            "rms_time_constant_s": float(self.config.rms_time_constant_s),
            "max_abs_target_power_w": float(self.config.max_abs_target_power_w),
            "max_abs_generalized_force_n": float(
                self.config.max_abs_generalized_force_n
            ),
            "max_abs_nodal_force_n": float(self.config.max_abs_nodal_force_n),
            "bandpass_damping_ratio": float(self.config.bandpass_damping_ratio),
            "force_direction": str(self.config.force_direction),
            "limit_action": str(self.config.limit_action),
            "enabled": bool(self.config.enabled),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def _bandpass_candidate(
        self,
        raw_qdot_mps: float,
        dt_s: float,
    ) -> Tuple[float, float, float, float]:
        """Return candidate position, rate, prior-input, and filtered output."""

        omega = 2.0 * math.pi * float(self.config.natural_frequency_hz)
        damping = float(self.config.bandpass_damping_ratio)
        if self._filter_previous_input_mps is None:
            return raw_qdot_mps / omega**2, 0.0, raw_qdot_mps, 0.0
        if dt_s <= 0.0:  # pragma: no cover - duplicate step returns earlier
            raise RuntimeError("Bandpass state requires a positive accepted time step.")

        transition = self._transition_cache.get(dt_s)
        if transition is None:
            dynamics = np.asarray(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [-(omega**2), -2.0 * damping * omega, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
            transition = expm(dynamics * dt_s)
            self._transition_cache[dt_s] = transition
        previous_input = float(self._filter_previous_input_mps)
        input_slope = (raw_qdot_mps - previous_input) / dt_s
        state = transition @ np.asarray(
            [
                self._filter_position_state,
                self._filter_rate_state,
                previous_input,
                input_slope,
            ],
            dtype=float,
        )
        position = float(state[0])
        rate = float(state[1])
        filtered = 2.0 * damping * omega * rate
        if not all(math.isfinite(value) for value in (position, rate, filtered)):
            raise FloatingPointError("Modal-velocity bandpass state became non-finite.")
        return position, rate, raw_qdot_mps, filtered

    @staticmethod
    def _canonical_input_signature(
        time_s: float,
        node_ids: Tuple[str, ...],
        directions: np.ndarray,
        velocities: np.ndarray,
        dynamic_conjugate_weights: np.ndarray | None = None,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray([time_s], dtype="<f8").tobytes())
        for node_id in node_ids:
            encoded = node_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
            digest.update(encoded)
        for array in (directions, velocities):
            canonical = np.ascontiguousarray(array, dtype="<f8")
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes())
        if dynamic_conjugate_weights is None:
            digest.update(b"configured_projection_and_distribution_weights")
        else:
            digest.update(b"dynamic_energy_conjugate_weights")
            canonical = np.ascontiguousarray(dynamic_conjugate_weights, dtype="<f8")
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes())
        return digest.hexdigest()

    def _bounded(self, metric: str, value: float, limit: float) -> Tuple[float, bool]:
        if not math.isfinite(value):
            raise FloatingPointError(f"Modal-power {metric} became non-finite.")
        if abs(value) <= limit:
            return value, False
        if self.config.limit_action == "error":
            raise QBladeModalPowerLimitError(metric, value, limit)
        return math.copysign(limit, value), True

    def step(
        self,
        time_s: float,
        *,
        node_ids: Sequence[str],
        direction_basis: Any,
        structural_velocity_mps: Any,
        energy_conjugate_weights: Any | None = None,
        target_power_evaluator: Optional[
            Callable[[float], ExternalModalPowerTarget]
        ] = None,
    ) -> QBladeModalPowerResult:
        """Evaluate additive global nodal forces at one accepted time level.

        ``direction_basis`` and ``structural_velocity_mps`` must have shape
        ``(n_nodes, 3)``.  Direction rows are normalized internally.  The
        output has the current ``node_ids`` order and must be added to any base
        aerodynamic force by the caller.  When supplied,
        ``energy_conjugate_weights`` replaces both the configured modal
        projection and aerodynamic lift-back weights for this accepted step:
        the resulting Petrov coordinate and its adjoint satisfy
        ``sum(F_i dot v_i) = Q*qdot`` exactly, including non-modal velocity.
        """

        current_time = _finite_scalar(time_s, "time_s")
        step_node_ids = tuple(str(node_id) for node_id in node_ids)
        if (
            len(step_node_ids) != len(self._canonical_node_ids)
            or len(set(step_node_ids)) != len(step_node_ids)
            or set(step_node_ids) != set(self._canonical_node_ids)
        ):
            raise ValueError(
                "node_ids must contain every configured node exactly once."
            )
        n_nodes = len(step_node_ids)
        directions_step = np.asarray(direction_basis, dtype=float)
        velocities_step = np.asarray(structural_velocity_mps, dtype=float)
        if directions_step.shape != (n_nodes, 3):
            raise ValueError(f"direction_basis must have shape ({n_nodes}, 3).")
        if velocities_step.shape != (n_nodes, 3):
            raise ValueError(f"structural_velocity_mps must have shape ({n_nodes}, 3).")
        if not np.all(np.isfinite(directions_step)):
            raise ValueError("direction_basis must contain only finite values.")
        if not np.all(np.isfinite(velocities_step)):
            raise ValueError("structural_velocity_mps must contain only finite values.")
        norms = np.linalg.norm(directions_step, axis=1)
        if np.any(norms <= 0.0) or np.any(~np.isfinite(norms)):
            raise ValueError("Every direction_basis row must have finite nonzero norm.")
        directions_step = np.array(
            directions_step / norms[:, None], dtype=float, copy=True
        )
        velocities_step = np.array(velocities_step, dtype=float, copy=True)

        index_by_node = {node_id: index for index, node_id in enumerate(step_node_ids)}
        canonical_indices = [
            index_by_node[node_id] for node_id in self._canonical_node_ids
        ]
        directions = directions_step[canonical_indices]
        velocities = velocities_step[canonical_indices]
        dynamic_weights = None
        if energy_conjugate_weights is not None:
            weights_step = np.asarray(energy_conjugate_weights, dtype=float)
            if weights_step.shape != (n_nodes,):
                raise ValueError(
                    f"energy_conjugate_weights must have shape ({n_nodes},)."
                )
            if np.any(~np.isfinite(weights_step)) or np.any(weights_step < 0.0):
                raise ValueError(
                    "energy_conjugate_weights must be finite and non-negative."
                )
            dynamic_weights = np.array(
                weights_step[canonical_indices], dtype=float, copy=True
            )
            dynamic_denominator = float(
                np.dot(dynamic_weights * self._mode_shape, self._mode_shape)
            )
            if not math.isfinite(dynamic_denominator) or dynamic_denominator <= 0.0:
                raise ValueError(
                    "energy_conjugate_weights must overlap a nonzero mode shape."
                )
            projection_weights = dynamic_weights
            distribution_weights = dynamic_weights
            projection_denominator = dynamic_denominator
            distribution_denominator = dynamic_denominator
        else:
            projection_weights = self._modal_mass_weights_kg
            distribution_weights = self._aerodynamic_weights
            projection_denominator = self._projection_denominator
            distribution_denominator = self._distribution_denominator
        base_signature = self._canonical_input_signature(
            current_time,
            self._canonical_node_ids,
            directions,
            velocities,
            dynamic_weights,
        )

        external_target_mode = self.config.power_curve is None
        if external_target_mode != (target_power_evaluator is not None):
            raise ValueError(
                "target_power_evaluator is required exactly for external-target "
                "modal-power configurations."
            )

        def evaluate_external(amplitude_m: float) -> ExternalModalPowerTarget:
            if target_power_evaluator is None:  # pragma: no cover - guarded above
                raise RuntimeError("External target evaluator is unavailable.")
            target = target_power_evaluator(float(amplitude_m))
            if not isinstance(target, ExternalModalPowerTarget):
                raise TypeError(
                    "target_power_evaluator must return ExternalModalPowerTarget."
                )
            target_power = _finite_scalar(
                target.target_cycle_mean_power_w,
                "external target_cycle_mean_power_w",
            )
            origin_gain = _finite_scalar(
                target.origin_equivalent_velocity_gain_kgps,
                "external origin_equivalent_velocity_gain_kgps",
            )
            context = str(target.context_fingerprint).lower()
            if len(context) != 64 or any(
                character not in "0123456789abcdef" for character in context
            ):
                raise ValueError("External target context_fingerprint must be SHA-256.")
            return ExternalModalPowerTarget(target_power, origin_gain, context)

        def external_signature(target: ExternalModalPowerTarget) -> str:
            digest = hashlib.sha256()
            digest.update(base_signature.encode("ascii"))
            digest.update(
                np.asarray(
                    [
                        target.target_cycle_mean_power_w,
                        target.origin_equivalent_velocity_gain_kgps,
                    ],
                    dtype="<f8",
                ).tobytes()
            )
            digest.update(target.context_fingerprint.encode("ascii"))
            return digest.hexdigest()

        if self._last_time_s is not None:
            if current_time < self._last_time_s - _TIME_ABS_TOL_S:
                raise ValueError(
                    f"Modal-power model time decreased from {self._last_time_s:.17g} "
                    f"to {current_time:.17g} s."
                )
            if math.isclose(
                current_time,
                self._last_time_s,
                rel_tol=0.0,
                abs_tol=_TIME_ABS_TOL_S,
            ):
                if external_target_mode:
                    if self._last_diagnostics is None:
                        raise RuntimeError(
                            "External modal-power duplicate cache is incomplete."
                        )
                    duplicate_target = evaluate_external(
                        self._last_diagnostics.amplitude_estimate_m
                    )
                    signature = external_signature(duplicate_target)
                else:
                    signature = base_signature
                if signature != self._last_input_signature:
                    raise ValueError(
                        "Modal-power model inputs changed at the same time level."
                    )
                return self._cached_result_for_order(step_node_ids)

        dt_s = 0.0 if self._last_time_s is None else current_time - self._last_time_s
        directional_velocity = np.sum(velocities * directions, axis=1)
        raw_qdot = float(
            np.dot(
                projection_weights * self._mode_shape,
                directional_velocity,
            )
            / projection_denominator
        )
        if not math.isfinite(raw_qdot):
            raise FloatingPointError(
                "Projected generalized velocity became non-finite."
            )

        filter_position, filter_rate, filter_input, filtered_qdot = (
            self._bandpass_candidate(raw_qdot, dt_s)
        )
        mean_square = self._qdot_mean_square_m2ps2
        if dt_s > 0.0:
            decay = math.exp(-dt_s / float(self.config.rms_time_constant_s))
            mean_square = decay * mean_square + (1.0 - decay) * filtered_qdot**2
        if not math.isfinite(mean_square) or mean_square < 0.0:
            raise FloatingPointError("Modal-velocity mean square became invalid.")
        qdot_rms = math.sqrt(mean_square)
        omega = 2.0 * math.pi * float(self.config.natural_frequency_hz)
        amplitude = math.sqrt(2.0) * qdot_rms / omega
        external_target = None
        if external_target_mode:
            external_target = evaluate_external(amplitude)
            target_power = external_target.target_cycle_mean_power_w
            signature = external_signature(external_target)
        else:
            if self.config.power_curve is None:  # pragma: no cover - guarded above
                raise RuntimeError("Tabulated modal-power curve is unavailable.")
            target_power = self.config.power_curve.evaluate(amplitude)
            signature = base_signature

        target_limited = False
        generalized_limited = False
        nodal_limited_count = 0
        force_scale = 1.0
        if self.config.enabled:
            applied_target, target_limited = self._bounded(
                "target power",
                target_power,
                float(self.config.max_abs_target_power_w),
            )
            if amplitude <= _FLOAT_ABS_TOL:
                if external_target is None:
                    if self.config.power_curve is None:  # pragma: no cover
                        raise RuntimeError(
                            "Tabulated modal-power curve is unavailable."
                        )
                    velocity_gain = self.config.power_curve.equivalent_velocity_gain(
                        amplitude,
                        omega,
                    )
                else:
                    velocity_gain = external_target.origin_equivalent_velocity_gain_kgps
            else:
                velocity_gain = 2.0 * applied_target / (omega**2 * amplitude**2)
            if not math.isfinite(velocity_gain):
                raise FloatingPointError("Aerodynamic velocity gain became non-finite.")
            raw_generalized_force = velocity_gain * filtered_qdot
            bounded_generalized_force, generalized_limited = self._bounded(
                "generalized force",
                raw_generalized_force,
                float(self.config.max_abs_generalized_force_n),
            )
            force_scalar = (
                bounded_generalized_force
                * distribution_weights
                * self._mode_shape
                / distribution_denominator
            )
            if not np.all(np.isfinite(force_scalar)):
                raise FloatingPointError(
                    "Distributed modal-power force became non-finite."
                )
            raw_max_nodal = float(np.max(np.abs(force_scalar)))
            nodal_limit = float(self.config.max_abs_nodal_force_n)
            if raw_max_nodal > nodal_limit:
                nodal_limited_count = int(
                    np.count_nonzero(np.abs(force_scalar) > nodal_limit)
                )
                if self.config.limit_action == "error":
                    raise QBladeModalPowerLimitError(
                        "nodal force",
                        raw_max_nodal,
                        nodal_limit,
                    )
                force_scale = nodal_limit / raw_max_nodal
                force_scalar = force_scalar * force_scale
        else:
            applied_target = 0.0
            velocity_gain = 0.0
            raw_generalized_force = 0.0
            force_scalar = np.zeros(n_nodes, dtype=float)

        applied_generalized_force = float(np.dot(self._mode_shape, force_scalar))
        force_global = force_scalar[:, None] * directions
        instantaneous_modal_power = applied_generalized_force * filtered_qdot
        instantaneous_structural_power = float(np.sum(force_global * velocities))
        if not all(
            math.isfinite(value)
            for value in (
                applied_generalized_force,
                instantaneous_modal_power,
                instantaneous_structural_power,
            )
        ):
            raise FloatingPointError(
                "Realized modal-power diagnostics became non-finite."
            )

        modal_power_ema = self._modal_power_ema_w
        structural_power_ema = self._structural_power_ema_w
        if dt_s > 0.0:
            decay = math.exp(-dt_s / float(self.config.rms_time_constant_s))
            modal_power_ema = (
                decay * modal_power_ema + (1.0 - decay) * instantaneous_modal_power
            )
            structural_power_ema = (
                decay * structural_power_ema
                + (1.0 - decay) * instantaneous_structural_power
            )
        diagnostics = QBladeModalPowerDiagnostics(
            enabled=bool(self.config.enabled),
            time_s=current_time,
            dt_s=dt_s,
            raw_qdot_mps=raw_qdot,
            filtered_qdot_mps=filtered_qdot,
            qdot_rms_mps=qdot_rms,
            amplitude_estimate_m=amplitude,
            target_cycle_mean_power_w=target_power,
            applied_target_cycle_mean_power_w=applied_target,
            equivalent_velocity_gain_kgps=velocity_gain,
            raw_generalized_force_n=raw_generalized_force,
            applied_generalized_force_n=applied_generalized_force,
            force_distribution_denominator=distribution_denominator,
            force_scale=force_scale,
            target_power_limited=target_limited,
            generalized_force_limited=generalized_limited,
            nodal_force_limited_nodes=nodal_limited_count,
            instantaneous_modal_power_w=instantaneous_modal_power,
            instantaneous_structural_power_w=instantaneous_structural_power,
            realized_modal_power_ema_w=modal_power_ema,
            realized_structural_power_ema_w=structural_power_ema,
            max_abs_nodal_force_n=float(np.max(np.abs(force_scalar))),
            projection_weight_min_kg=float(np.min(self._modal_mass_weights_kg)),
            projection_weight_max_kg=float(np.max(self._modal_mass_weights_kg)),
            aerodynamic_weight_min=float(np.min(self._aerodynamic_weights)),
            aerodynamic_weight_max=float(np.max(self._aerodynamic_weights)),
            dynamic_conjugate_weight_override=dynamic_weights is not None,
            dynamic_conjugate_weight_min=(
                0.0 if dynamic_weights is None else float(np.min(dynamic_weights))
            ),
            dynamic_conjugate_weight_max=(
                0.0 if dynamic_weights is None else float(np.max(dynamic_weights))
            ),
            instantaneous_controller_state_power_w=instantaneous_modal_power,
            instantaneous_realized_work_w=instantaneous_structural_power,
            controller_state_power_ema_w=modal_power_ema,
            realized_work_ema_w=structural_power_ema,
        )

        # Commit only after all interpolation, limit, and finite checks pass.
        self._last_time_s = current_time
        self._filter_position_state = filter_position
        self._filter_rate_state = filter_rate
        self._filter_previous_input_mps = filter_input
        self._qdot_mean_square_m2ps2 = mean_square
        self._modal_power_ema_w = modal_power_ema
        self._structural_power_ema_w = structural_power_ema
        self._last_input_signature = signature
        self._last_force_canonical_n = np.array(force_global, dtype=float, copy=True)
        self._last_diagnostics = diagnostics
        return self._cached_result_for_order(step_node_ids)

    def _cached_result_for_order(
        self,
        step_node_ids: Sequence[str],
    ) -> QBladeModalPowerResult:
        if self._last_force_canonical_n is None or self._last_diagnostics is None:
            raise RuntimeError("Modal-power duplicate cache is incomplete.")
        canonical_index = {
            node_id: index for index, node_id in enumerate(self._canonical_node_ids)
        }
        force = np.asarray(
            [
                self._last_force_canonical_n[canonical_index[str(node_id)]]
                for node_id in step_node_ids
            ],
            dtype=float,
        )
        return QBladeModalPowerResult(
            additive_force_n=np.array(force, dtype=float, copy=True),
            diagnostics=self._last_diagnostics,
        )

    def checkpoint_state(self) -> Mapping[str, Any]:
        """Return JSON-serializable state sufficient for exact replay."""

        cached = None
        if (
            self._last_force_canonical_n is not None
            and self._last_diagnostics is not None
        ):
            cached = {
                "canonical_additive_force_n": self._last_force_canonical_n.tolist(),
                "diagnostics": asdict(self._last_diagnostics),
            }
        return {
            "schema": _CHECKPOINT_SCHEMA,
            "config_fingerprint": self._config_fingerprint,
            "last_time_s": self._last_time_s,
            "bandpass_filter": {
                "position_state": self._filter_position_state,
                "rate_state": self._filter_rate_state,
                "previous_input_mps": self._filter_previous_input_mps,
            },
            "qdot_mean_square_m2ps2": self._qdot_mean_square_m2ps2,
            "modal_power_ema_w": self._modal_power_ema_w,
            "structural_power_ema_w": self._structural_power_ema_w,
            "last_input_signature": self._last_input_signature,
            "last_result": cached,
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpoint after strict schema, configuration, and finite checks."""

        if state.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported QBlade modal-power checkpoint schema.")
        if state.get("config_fingerprint") != self._config_fingerprint:
            raise ValueError("QBlade modal-power checkpoint configuration mismatch.")
        last_time_raw = state.get("last_time_s")
        last_time = (
            None
            if last_time_raw is None
            else _finite_scalar(last_time_raw, "last_time_s")
        )
        filter_state = state.get("bandpass_filter")
        if not isinstance(filter_state, Mapping):
            raise ValueError("bandpass_filter must be a mapping.")
        position = _finite_scalar(filter_state.get("position_state"), "filter position")
        rate = _finite_scalar(filter_state.get("rate_state"), "filter rate")
        prior_input_raw = filter_state.get("previous_input_mps")
        prior_input = (
            None
            if prior_input_raw is None
            else _finite_scalar(prior_input_raw, "filter previous input")
        )
        mean_square = _finite_scalar(
            state.get("qdot_mean_square_m2ps2"),
            "qdot_mean_square_m2ps2",
        )
        if mean_square < 0.0:
            raise ValueError("qdot_mean_square_m2ps2 must be non-negative.")
        modal_power_ema = _finite_scalar(
            state.get("modal_power_ema_w"),
            "modal_power_ema_w",
        )
        structural_power_ema = _finite_scalar(
            state.get("structural_power_ema_w"),
            "structural_power_ema_w",
        )
        signature_raw = state.get("last_input_signature")
        signature = None if signature_raw is None else str(signature_raw)

        cached_raw = state.get("last_result")
        cached_force = None
        cached_diagnostics = None
        if cached_raw is not None:
            if not isinstance(cached_raw, Mapping):
                raise ValueError("last_result must be a mapping or null.")
            cached_force = np.asarray(
                cached_raw.get("canonical_additive_force_n"),
                dtype=float,
            )
            if cached_force.shape != (len(self._canonical_node_ids), 3) or not np.all(
                np.isfinite(cached_force)
            ):
                raise ValueError(
                    "Checkpoint canonical force has invalid shape or values."
                )
            diagnostics_raw = cached_raw.get("diagnostics")
            if not isinstance(diagnostics_raw, Mapping):
                raise ValueError("Checkpoint diagnostics must be a mapping.")
            expected_names = {
                field.name for field in fields(QBladeModalPowerDiagnostics)
            }
            if set(diagnostics_raw) != expected_names:
                raise ValueError(
                    "Checkpoint diagnostics fields do not match the schema."
                )
            for key, value in diagnostics_raw.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"Checkpoint diagnostic {key} is non-finite.")
            try:
                cached_diagnostics = QBladeModalPowerDiagnostics(
                    **dict(diagnostics_raw)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("Checkpoint diagnostics are invalid.") from exc

        if last_time is None:
            if any(
                value is not None
                for value in (prior_input, signature, cached_force, cached_diagnostics)
            ):
                raise ValueError(
                    "An unstarted checkpoint cannot contain dynamic cache state."
                )
            if any(
                value != 0.0
                for value in (
                    position,
                    rate,
                    mean_square,
                    modal_power_ema,
                    structural_power_ema,
                )
            ):
                raise ValueError(
                    "An unstarted checkpoint must contain zero dynamic state."
                )
        else:
            if any(
                value is None
                for value in (prior_input, signature, cached_force, cached_diagnostics)
            ):
                raise ValueError(
                    "A started checkpoint requires complete dynamic cache state."
                )
            if not math.isclose(
                cached_diagnostics.time_s,
                last_time,
                rel_tol=0.0,
                abs_tol=_TIME_ABS_TOL_S,
            ):
                raise ValueError(
                    "Checkpoint diagnostic time does not match last_time_s."
                )
            if cached_diagnostics.enabled is not bool(self.config.enabled):
                raise ValueError("Checkpoint diagnostic enabled state is inconsistent.")

        self._last_time_s = last_time
        self._filter_position_state = position
        self._filter_rate_state = rate
        self._filter_previous_input_mps = prior_input
        self._qdot_mean_square_m2ps2 = mean_square
        self._modal_power_ema_w = modal_power_ema
        self._structural_power_ema_w = structural_power_ema
        self._last_input_signature = signature
        self._last_force_canonical_n = (
            None
            if cached_force is None
            else np.array(cached_force, dtype=float, copy=True)
        )
        self._last_diagnostics = cached_diagnostics


__all__ = [
    "ExternalModalPowerTarget",
    "QBladeModalPowerDiagnostics",
    "QBladeModalPowerLimitError",
    "QBladeModalPowerModel",
    "QBladeModalPowerModelConfig",
    "QBladeModalPowerResult",
    "TabulatedAerodynamicPowerCurve",
]
