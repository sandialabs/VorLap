"""Structure-conditioned partition of airfoil spectra for H-VAWT lock-in.

The legacy VorLap source reconstructs every interpolated airfoil harmonic at
its measured Strouhal frequency.  The first H-VAWT response closure instead
discarded those spectra and used a cylinder-derived reduced-velocity window.
This module supplies the conservative bridge between those two descriptions:

* interpolate ``CL`` spectra upstream at the instantaneous Reynolds number and
  angle of attack;
* convert each Strouhal number to a dimensional source frequency;
* partition near-resonant harmonic energy among audited structural modes;
* pull only that captured energy to the corresponding modal frequency; and
* retain the uncaptured amplitude and phase as an energy-conserving residual.

For the frozen cylinder calibration, ``St=0.2`` and ``|CL|=0.3``.  Therefore
the default frequency-ratio window is the calibrated reduced-velocity window
multiplied by 0.2, and the captured spectral gain is normalized by 0.3.  A
single cylinder harmonic consequently reproduces the existing activation
curve exactly.  Airfoil spectra are not flattened to that cylinder: their
measured frequency locations and energy determine the available modal gain.

This is a spectrum partition, not an independent aerodynamic force source.
The caller must reconstruct the residual spectrum once and apply the captured
modal contribution once.  QBlade remains responsible for structural Rayleigh
damping; the modal damping balance reported here is a diagnostic consistency
check, not an additional damping term.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numbers
from typing import Any, Sequence

import numpy as np

from .force_model import (
    force_from_chord_normal,
    harmonic_force_channel_from_angle,
    section_flow_state,
    strouhal_reference_length,
)
from .interpolation import (
    interpolate_fft_spectra_batch_packed,
    prepare_fft_spectra_packed_tensor,
)
from .qblade_engineering_lockin import capture_weight
from .qblade_source_coverage import SourceCoverageRow
from .structs import AirfoilFFT


CYLINDER_REFERENCE_STROUHAL = 0.2
CYLINDER_REFERENCE_LIFT_AMPLITUDE = 0.3
CYLINDER_REDUCED_VELOCITY_LOW = 5.3625308212165494
CYLINDER_REDUCED_VELOCITY_PEAK = 6.245825
CYLINDER_REDUCED_VELOCITY_HIGH = 6.982801851062782
DEFAULT_CAPTURE_FREQUENCY_RATIO_LOW = (
    CYLINDER_REFERENCE_STROUHAL * CYLINDER_REDUCED_VELOCITY_LOW
)
DEFAULT_CAPTURE_FREQUENCY_RATIO_PEAK = (
    CYLINDER_REFERENCE_STROUHAL * CYLINDER_REDUCED_VELOCITY_PEAK
)
DEFAULT_CAPTURE_FREQUENCY_RATIO_HIGH = (
    CYLINDER_REFERENCE_STROUHAL * CYLINDER_REDUCED_VELOCITY_HIGH
)

_EPS = 1.0e-12


def _strict_real(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be a real numeric value.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _positive(value: Any, label: str) -> float:
    result = _strict_real(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return result


def _readonly(value: Any, *, dtype: Any = float) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def smooth_capture_amplitude(
    frequency_ratio: Any,
    low: float,
    peak: float,
    high: float,
) -> np.ndarray:
    """Return the calibrated C1-continuous capture-amplitude window.

    The input ratio is the undisturbed source frequency divided by the
    structural natural frequency.  The result is an amplitude fraction in
    ``[0, 1]``; its square is the corresponding captured spectral-energy
    fraction.
    """

    low_value = _strict_real(low, "low")
    peak_value = _strict_real(peak, "peak")
    high_value = _strict_real(high, "high")
    if low_value < 0.0 or not low_value < peak_value < high_value:
        raise ValueError("Capture bounds must satisfy 0 <= low < peak < high.")
    ratio = np.asarray(frequency_ratio, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("frequency_ratio must be finite and non-negative.")
    capture = np.zeros_like(ratio)
    rising = (ratio > low_value) & (ratio < peak_value)
    falling = (ratio >= peak_value) & (ratio < high_value)
    rise = (ratio[rising] - low_value) / (peak_value - low_value)
    fall = (high_value - ratio[falling]) / (high_value - peak_value)
    capture[rising] = rise**2 * (3.0 - 2.0 * rise)
    capture[falling] = fall**2 * (3.0 - 2.0 * fall)
    capture[ratio == peak_value] = 1.0
    return capture


@dataclass(frozen=True)
class SpectralLockInCaptureConfig:
    """Frozen spectral normalization and frequency-capture window."""

    reference_strouhal: float = CYLINDER_REFERENCE_STROUHAL
    reference_lift_amplitude: float = CYLINDER_REFERENCE_LIFT_AMPLITUDE
    frequency_ratio_low: float = DEFAULT_CAPTURE_FREQUENCY_RATIO_LOW
    frequency_ratio_peak: float = DEFAULT_CAPTURE_FREQUENCY_RATIO_PEAK
    frequency_ratio_high: float = DEFAULT_CAPTURE_FREQUENCY_RATIO_HIGH
    maximum_spectral_gain: float = 10.0
    minimum_modal_participation: float = 1.0e-12
    capture_window: str = "cubic_smoothstep"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ValueError("spectral capture schema_version must be integer 1 or 2.")
        expected_window = (
            "cubic_smoothstep"
            if self.schema_version == 1
            else "compact_sine_squared"
        )
        if self.capture_window != expected_window:
            raise ValueError(
                f"spectral capture schema {self.schema_version} requires "
                f"capture_window={expected_window!r}."
            )
        reference_st = _positive(self.reference_strouhal, "reference_strouhal")
        reference_amp = _positive(
            self.reference_lift_amplitude,
            "reference_lift_amplitude",
        )
        low = _strict_real(self.frequency_ratio_low, "frequency_ratio_low")
        peak = _strict_real(self.frequency_ratio_peak, "frequency_ratio_peak")
        high = _strict_real(self.frequency_ratio_high, "frequency_ratio_high")
        if low < 0.0 or not low < peak < high:
            raise ValueError(
                "Frequency-ratio bounds must satisfy 0 <= low < peak < high."
            )
        maximum_gain = _positive(self.maximum_spectral_gain, "maximum_spectral_gain")
        minimum_participation = _strict_real(
            self.minimum_modal_participation,
            "minimum_modal_participation",
        )
        if not 0.0 <= minimum_participation < 1.0:
            raise ValueError("minimum_modal_participation must be in [0, 1).")
        object.__setattr__(self, "reference_strouhal", reference_st)
        object.__setattr__(self, "reference_lift_amplitude", reference_amp)
        object.__setattr__(self, "frequency_ratio_low", low)
        object.__setattr__(self, "frequency_ratio_peak", peak)
        object.__setattr__(self, "frequency_ratio_high", high)
        object.__setattr__(self, "maximum_spectral_gain", maximum_gain)
        object.__setattr__(
            self,
            "minimum_modal_participation",
            minimum_participation,
        )


@dataclass(frozen=True)
class SpectralLockInModeConfig:
    """Audited structural data needed by one captured response mode."""

    tracked_mode: int
    natural_frequency_hz: float
    generalized_modal_mass_kg: float
    structural_damping_ratio: float
    reference_length_m: float
    aerodynamic_damping_gain_h1: float
    saturation_amplitude_ratio: float

    def __post_init__(self) -> None:
        if type(self.tracked_mode) is not int or self.tracked_mode <= 0:
            raise ValueError("tracked_mode must be a positive integer.")
        for name in (
            "natural_frequency_hz",
            "generalized_modal_mass_kg",
            "reference_length_m",
            "saturation_amplitude_ratio",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        damping = _strict_real(
            self.structural_damping_ratio,
            "structural_damping_ratio",
        )
        if not 0.0 <= damping < 1.0:
            raise ValueError("structural_damping_ratio must be in [0, 1).")
        h1 = _strict_real(
            self.aerodynamic_damping_gain_h1,
            "aerodynamic_damping_gain_h1",
        )
        if h1 < 0.0:
            raise ValueError("aerodynamic_damping_gain_h1 must be non-negative.")
        object.__setattr__(self, "structural_damping_ratio", damping)
        object.__setattr__(self, "aerodynamic_damping_gain_h1", h1)


@dataclass(frozen=True)
class SpectralPartitionResult:
    """Immutable energy partition for one interpolated lift spectrum."""

    source_frequency_hz: np.ndarray
    source_amplitude: np.ndarray
    source_phase_rad: np.ndarray
    residual_amplitude: np.ndarray
    capture_amplitude_fraction: np.ndarray
    capture_energy_fraction_by_mode: np.ndarray
    captured_amplitude_by_node_mode: np.ndarray
    spectral_gain_by_node_mode: np.ndarray
    source_centroid_frequency_hz_by_node_mode: np.ndarray
    pulled_frequency_hz_by_node_mode: np.ndarray
    total_source_energy_by_node: np.ndarray
    total_residual_energy_by_node: np.ndarray
    total_captured_energy_by_node: np.ndarray
    maximum_energy_closure_error: float

    def __post_init__(self) -> None:
        for name in (
            "source_frequency_hz",
            "source_amplitude",
            "source_phase_rad",
            "residual_amplitude",
            "capture_amplitude_fraction",
            "capture_energy_fraction_by_mode",
            "captured_amplitude_by_node_mode",
            "spectral_gain_by_node_mode",
            "source_centroid_frequency_hz_by_node_mode",
            "pulled_frequency_hz_by_node_mode",
            "total_source_energy_by_node",
            "total_residual_energy_by_node",
            "total_captured_energy_by_node",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        closure = _strict_real(
            self.maximum_energy_closure_error,
            "maximum_energy_closure_error",
        )
        if closure < 0.0:
            raise ValueError("maximum_energy_closure_error must be non-negative.")
        object.__setattr__(self, "maximum_energy_closure_error", closure)


@dataclass(frozen=True)
class ModalDampingBalance:
    """Zero-amplitude modal energy balance implied by a spectral partition."""

    tracked_mode: int
    structural_damping_coefficient_nspm: float
    captured_aerodynamic_damping_coefficient_nspm: float
    aerodynamic_to_structural_damping_ratio: float
    linear_lockin_available: bool
    predicted_equilibrium_amplitude_m: float
    predicted_equilibrium_amplitude_ratio: float


@dataclass(frozen=True)
class HybridSpectralStepResult:
    """Residual source force and captured-energy diagnostics for one step."""

    residual_force_global_n: np.ndarray
    partition: SpectralPartitionResult
    base_aoa_deg: np.ndarray
    base_reynolds: np.ndarray
    base_effective_crossflow_speed_mps: np.ndarray
    base_section_width_m: np.ndarray
    applied_residual_lift_energy_by_node: np.ndarray
    truncated_residual_lift_energy_by_node: np.ndarray
    truncated_residual_drag_energy_by_node: np.ndarray
    maximum_residual_nodal_force_n: float
    residual_force_power_w: float

    def __post_init__(self) -> None:
        for name in (
            "residual_force_global_n",
            "base_aoa_deg",
            "base_reynolds",
            "base_effective_crossflow_speed_mps",
            "base_section_width_m",
            "applied_residual_lift_energy_by_node",
            "truncated_residual_lift_energy_by_node",
            "truncated_residual_drag_energy_by_node",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        if not isinstance(self.partition, SpectralPartitionResult):
            raise TypeError("partition must be SpectralPartitionResult.")
        for name in ("maximum_residual_nodal_force_n", "residual_force_power_w"):
            object.__setattr__(self, name, _strict_real(getattr(self, name), name))


def _canonical_array_bytes(value: Any) -> bytes:
    array = np.ascontiguousarray(value, dtype="<f8")
    return np.asarray(array.shape, dtype="<i8").tobytes() + array.tobytes()


def airfoil_fft_content_sha256(airfoil: AirfoilFFT) -> str:
    """Return a path-independent identity for the parsed airfoil FFT tensors."""

    if not isinstance(airfoil, AirfoilFFT):
        raise TypeError("airfoil must be AirfoilFFT.")
    digest = hashlib.sha256()
    digest.update(b"vorlap-airfoil-fft-content-v1\0")
    digest.update(airfoil.name.encode("utf-8") + b"\0")
    digest.update(airfoil.reynolds_model.encode("utf-8") + b"\0")
    digest.update(_canonical_array_bytes(airfoil.Re))
    digest.update(_canonical_array_bytes(airfoil.AOA))
    digest.update(_canonical_array_bytes([airfoil.Thickness]))
    for field in AirfoilFFT._FIELDS:
        for suffix in ("ST", "Amp", "Pha"):
            digest.update(_canonical_array_bytes(getattr(airfoil, f"{field}_{suffix}")))
    return digest.hexdigest()


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HVAWTSpectralLockInSource:
    """Stateful phase-continuous residual source plus modal energy capture.

    The source spectra are interpolated at the base-flow AoA/Re.  Their
    capture frequency is evaluated using the rigid-motion-only response flow,
    preventing flexible velocity from directly moving the lock-in window.
    Residual lift and all fluctuating drag are reconstructed with their
    uncaptured amplitudes; DC coefficients are intentionally omitted because
    native QBlade aerodynamics supplies the mean load.
    """

    def __init__(
        self,
        *,
        node_ids: Sequence[str],
        airfoils_by_node: Sequence[AirfoilFFT],
        component_ids_by_node: Sequence[str] | None = None,
        modes: Sequence[SpectralLockInModeConfig],
        modal_participation: Any,
        n_frequency_entries: int,
        fluid_density_kgpm3: float,
        fluid_dynamic_viscosity_pas: float,
        source_bounds_policy: str = "raise",
        source_frequency_cap_hz: float | None = None,
        sample_period_s: float | None = None,
        capture: SpectralLockInCaptureConfig | None = None,
    ) -> None:
        self.node_ids = tuple(node_ids)
        if (
            not self.node_ids
            or len(set(self.node_ids)) != len(self.node_ids)
            or any(not isinstance(value, str) or not value for value in self.node_ids)
        ):
            raise ValueError("node_ids must be unique nonempty strings.")
        airfoils = tuple(airfoils_by_node)
        if len(airfoils) != len(self.node_ids) or any(
            not isinstance(value, AirfoilFFT) for value in airfoils
        ):
            raise TypeError("airfoils_by_node must contain one AirfoilFFT per node.")
        component_ids = (
            self.node_ids
            if component_ids_by_node is None
            else tuple(component_ids_by_node)
        )
        if (
            len(component_ids) != len(self.node_ids)
            or any(not isinstance(value, str) or not value for value in component_ids)
        ):
            raise ValueError(
                "component_ids_by_node must contain one nonempty string per node."
            )
        self.modes = tuple(modes)
        if not self.modes or any(
            not isinstance(value, SpectralLockInModeConfig) for value in self.modes
        ):
            raise TypeError("modes must contain SpectralLockInModeConfig objects.")
        if type(n_frequency_entries) is not int or n_frequency_entries < 1:
            raise ValueError("n_frequency_entries must be a positive integer.")
        self.n_frequency_entries = n_frequency_entries
        self.fluid_density_kgpm3 = _positive(
            fluid_density_kgpm3,
            "fluid_density_kgpm3",
        )
        self.fluid_dynamic_viscosity_pas = _positive(
            fluid_dynamic_viscosity_pas,
            "fluid_dynamic_viscosity_pas",
        )
        if source_bounds_policy not in {"raise", "warn", "clip"}:
            raise ValueError("source_bounds_policy must be raise, warn, or clip.")
        self.source_bounds_policy = source_bounds_policy
        self.source_frequency_cap_hz = (
            None
            if source_frequency_cap_hz is None
            else _positive(source_frequency_cap_hz, "source_frequency_cap_hz")
        )
        self.sample_period_s = (
            None
            if sample_period_s is None
            else _positive(sample_period_s, "sample_period_s")
        )
        self.capture = capture or SpectralLockInCaptureConfig()
        if not isinstance(self.capture, SpectralLockInCaptureConfig):
            raise TypeError("capture must be SpectralLockInCaptureConfig.")
        participation = np.asarray(modal_participation, dtype=float)
        expected_participation = (len(self.node_ids), len(self.modes))
        if participation.shape != expected_participation:
            raise ValueError(
                f"modal_participation must have shape {expected_participation}."
            )
        if np.any(~np.isfinite(participation)) or np.any(participation < 0.0):
            raise ValueError("modal_participation must be finite and non-negative.")
        self.modal_participation = _readonly(participation)

        groups: dict[str, dict[str, Any]] = {}
        for index, airfoil in enumerate(airfoils):
            identity = airfoil_fft_content_sha256(airfoil)
            if identity not in groups:
                groups[identity] = {"airfoil": airfoil, "indices": []}
            groups[identity]["indices"].append(index)
        self._groups = tuple(
            {
                "identity": identity,
                "name": str(row["airfoil"].name),
                "airfoil": row["airfoil"],
                "indices": np.asarray(row["indices"], dtype=int),
                "depth": min(
                    n_frequency_entries,
                    int(row["airfoil"].CL_ST.shape[2]),
                    int(row["airfoil"].CD_ST.shape[2]),
                ),
                "packed_tensor": prepare_fft_spectra_packed_tensor(
                    row["airfoil"],
                    ["CL", "CD"],
                    n_freq_depth=min(
                        n_frequency_entries,
                        int(row["airfoil"].CL_ST.shape[2]),
                        int(row["airfoil"].CD_ST.shape[2]),
                    ),
                ),
            }
            for identity, row in sorted(groups.items())
        )
        coverage_groups: dict[str, dict[str, Any]] = {}
        source_re_min = np.empty(len(self.node_ids), dtype=float)
        source_re_max = np.empty(len(self.node_ids), dtype=float)
        source_aoa_min = np.empty(len(self.node_ids), dtype=float)
        source_aoa_max = np.empty(len(self.node_ids), dtype=float)
        source_force_coefficient_max = np.empty(len(self.node_ids), dtype=float)
        reynolds_models: list[str] = []
        airfoil_labels: list[str] = []
        for index, (component, airfoil) in enumerate(zip(component_ids, airfoils)):
            identity = airfoil_fft_content_sha256(airfoil)
            depth = min(
                n_frequency_entries,
                int(airfoil.CL_ST.shape[2]),
                int(airfoil.CD_ST.shape[2]),
            )
            source_cf_sum = np.sum(np.abs(airfoil.CF_Amp[:, :, :depth]), axis=2)
            source_cl_sum = np.sum(np.abs(airfoil.CL_Amp[:, :, :depth]), axis=2)
            source_cd_sum = np.sum(np.abs(airfoil.CD_Amp[:, :, :depth]), axis=2)
            source_re_min[index] = float(np.min(airfoil.Re))
            source_re_max[index] = float(np.max(airfoil.Re))
            source_aoa_min[index] = float(np.min(airfoil.AOA))
            source_aoa_max[index] = float(np.max(airfoil.AOA))
            source_force_coefficient_max[index] = max(
                float(np.max(source_cf_sum)),
                float(np.max(np.hypot(source_cl_sum, source_cd_sum))),
            )
            reynolds_models.append(str(airfoil.reynolds_model))
            airfoil_labels.append(f"{airfoil.name}:{identity}")
            if component not in coverage_groups:
                coverage_groups[component] = {
                    "component": component,
                    "indices": [],
                }
            coverage_groups[component]["indices"].append(index)
        self._coverage_groups = tuple(
            {
                **row,
                "indices": np.asarray(row["indices"], dtype=int),
                "airfoil_id": ",".join(
                    sorted({airfoil_labels[index] for index in row["indices"]})
                ),
                "reynolds_model": ",".join(
                    sorted({reynolds_models[index] for index in row["indices"]})
                ),
            }
            for _component, row in sorted(coverage_groups.items())
        )
        self._source_re_min_by_node = _readonly(source_re_min)
        self._source_re_max_by_node = _readonly(source_re_max)
        self._source_aoa_min_by_node = _readonly(source_aoa_min)
        self._source_aoa_max_by_node = _readonly(source_aoa_max)
        self._source_force_coefficient_max_by_node = _readonly(
            source_force_coefficient_max
        )
        self._reynolds_models_by_node = tuple(reynolds_models)
        self._reynolds_independent_by_node = _readonly(
            [
                value == "replicated_reynolds_independent"
                for value in reynolds_models
            ],
            dtype=bool,
        )
        self.airfoil_identity_by_node = tuple(
            airfoil_fft_content_sha256(value) for value in airfoils
        )
        identity_payload = {
            "schema": "vorlap-hvawt-spectral-lockin-source-v1",
            "node_ids": list(self.node_ids),
            "component_ids_by_node": list(component_ids),
            "airfoil_identity_by_node": list(self.airfoil_identity_by_node),
            "modes": [asdict(value) for value in self.modes],
            "modal_participation_sha256": hashlib.sha256(
                _canonical_array_bytes(self.modal_participation)
            ).hexdigest(),
            "n_frequency_entries": self.n_frequency_entries,
            "fluid_density_kgpm3": self.fluid_density_kgpm3,
            "fluid_dynamic_viscosity_pas": self.fluid_dynamic_viscosity_pas,
            "source_bounds_policy": self.source_bounds_policy,
            "source_frequency_cap_hz": self.source_frequency_cap_hz,
            "sample_period_s": self.sample_period_s,
            "capture": asdict(self.capture),
        }
        self.config_fingerprint = _semantic_sha256(identity_payload)
        state_shape = (len(self.node_ids), self.n_frequency_entries)
        self._phase_angle_cl_rad = np.zeros(state_shape, dtype=float)
        self._phase_angle_cd_rad = np.zeros(state_shape, dtype=float)
        self._phase_frequency_cl_hz = np.zeros(state_shape, dtype=float)
        self._phase_frequency_cd_hz = np.zeros(state_shape, dtype=float)
        self._last_time_s: float | None = None
        self._last_input_signature: str | None = None
        self._last_result: HybridSpectralStepResult | None = None
        self._last_coverage_sample: tuple[dict[str, Any], ...] | None = None
        self._coverage_last_sample_index: int | None = None
        self._coverage_rows: dict[str, dict[str, Any]] = {}
        self.performance_timing = None

    def _input_signature(self, time_s: float, arrays: Sequence[np.ndarray]) -> str:
        digest = hashlib.sha256()
        digest.update(b"vorlap-hvawt-spectral-step-v1\0")
        digest.update(_canonical_array_bytes([time_s]))
        for value in arrays:
            digest.update(_canonical_array_bytes(value))
        return digest.hexdigest()

    def _coverage_sample(
        self,
        *,
        time_s: float,
        flow_reynolds: np.ndarray,
        flow_aoa_deg: np.ndarray,
        frequency_cl_hz: np.ndarray,
        frequency_cd_hz: np.ndarray,
        amplitude_cl: np.ndarray,
        amplitude_cd: np.ndarray,
        section_dynamic_force_n: np.ndarray,
    ) -> tuple[dict[str, Any], ...]:
        """Prepare immutable scalar source-coverage evidence for one step."""

        sample_period = self.sample_period_s
        nyquist_hz = (
            0.0
            if sample_period is None
            else 0.5 / _positive(sample_period, "sample_period_s")
        )
        active_cl = np.abs(amplitude_cl) > _EPS
        active_cd = np.abs(amplitude_cd) > _EPS
        active_cl[:, 0] = False
        active_cd[:, 0] = False
        if self.source_frequency_cap_hz is None:
            truncate_cl = np.zeros(frequency_cl_hz.shape, dtype=bool)
            truncate_cd = np.zeros(frequency_cd_hz.shape, dtype=bool)
        else:
            truncate_cl = frequency_cl_hz > self.source_frequency_cap_hz
            truncate_cd = frequency_cd_hz > self.source_frequency_cap_hz
            truncate_cl[:, 0] = False
            truncate_cd[:, 0] = False
        node_frequency_max = np.maximum(
            np.max(
                np.where(active_cl & ~truncate_cl, frequency_cl_hz, 0.0),
                axis=1,
            ),
            np.max(
                np.where(active_cd & ~truncate_cd, frequency_cd_hz, 0.0),
                axis=1,
            ),
        )
        unfiltered_node_frequency_max = np.maximum(
            np.max(np.where(active_cl, frequency_cl_hz, 0.0), axis=1),
            np.max(np.where(active_cd, frequency_cd_hz, 0.0), axis=1),
        )
        truncated_count_by_node = np.count_nonzero(
            truncate_cl & active_cl,
            axis=1,
        ) + np.count_nonzero(truncate_cd & active_cd, axis=1)
        rows: list[dict[str, Any]] = []
        for group in self._coverage_groups:
            indices = group["indices"]
            reynolds = flow_reynolds[indices]
            aoa = flow_aoa_deg[indices]
            re_min = self._source_re_min_by_node[indices]
            re_max = self._source_re_max_by_node[indices]
            aoa_min = self._source_aoa_min_by_node[indices]
            aoa_max = self._source_aoa_max_by_node[indices]
            reynolds_independent = self._reynolds_independent_by_node[indices]
            re_below = reynolds < re_min
            re_lower_warning = re_below & reynolds_independent
            re_out = (reynolds > re_max) | (re_below & ~reynolds_independent)
            aoa_out = (aoa < aoa_min) | (aoa > aoa_max)

            dynamic_force = np.asarray(section_dynamic_force_n[indices], dtype=float)
            rows.append(
                {
                    "time_s": time_s,
                    "identity": str(group["component"]),
                    "component": str(group["component"]),
                    "airfoil_id": str(group["airfoil_id"]),
                    "reynolds_model": str(group["reynolds_model"]),
                    "node_count": int(indices.size),
                    "indices": indices,
                    "dynamic_force_n": dynamic_force,
                    "aoa_min_deg": float(np.min(aoa)),
                    "aoa_max_deg": float(np.max(aoa)),
                    "source_aoa_min_deg": float(np.min(aoa_min)),
                    "source_aoa_max_deg": float(np.max(aoa_max)),
                    "re_min": float(np.min(reynolds)),
                    "re_max": float(np.max(reynolds)),
                    "source_re_min": float(np.min(re_min)),
                    "source_re_max": float(np.max(re_max)),
                    "freq_min_hz": 0.0,
                    "freq_max_hz": float(np.max(node_frequency_max[indices])),
                    "unfiltered_freq_max_hz": float(
                        np.max(unfiltered_node_frequency_max[indices])
                    ),
                    "frequency_truncated_count": int(
                        np.sum(truncated_count_by_node[indices])
                    ),
                    "nyquist_hz": nyquist_hz,
                    "source_force_coefficient_max_abs": float(
                        np.max(self._source_force_coefficient_max_by_node[indices])
                    ),
                    "re_lower_bound_warning_count": int(
                        np.count_nonzero(re_lower_warning)
                    ),
                    "re_out_of_range_count": int(np.count_nonzero(re_out)),
                    "aoa_out_of_range_count": int(np.count_nonzero(aoa_out)),
                    "aliased_frequency_count": (
                        0
                        if nyquist_hz <= 0.0
                        else int(
                            np.count_nonzero(node_frequency_max[indices] > nyquist_hz)
                        )
                    ),
                    "source_clip_count": int(np.count_nonzero(re_out | aoa_out)),
                    # All contributing arrays have already passed finite-value
                    # validation in the flow and interpolation kernels.
                    "nonfinite_count": 0,
                }
            )
        return tuple(rows)

    @staticmethod
    def _new_coverage_row(sample: dict[str, Any]) -> dict[str, Any]:
        return {
            "identity": sample["identity"],
            "component": sample["component"],
            "airfoil_id": sample["airfoil_id"],
            "reynolds_model": sample["reynolds_model"],
            "samples": 0,
            "node_samples": 0,
            "aoa_min_deg": sample["aoa_min_deg"],
            "aoa_max_deg": sample["aoa_max_deg"],
            "source_aoa_min_deg": sample["source_aoa_min_deg"],
            "source_aoa_max_deg": sample["source_aoa_max_deg"],
            "re_min": sample["re_min"],
            "re_max": sample["re_max"],
            "source_re_min": sample["source_re_min"],
            "source_re_max": sample["source_re_max"],
            "freq_min_hz": sample["freq_min_hz"],
            "freq_max_hz": sample["freq_max_hz"],
            "unfiltered_freq_max_hz": sample["unfiltered_freq_max_hz"],
            "frequency_truncated_count": 0,
            "nyquist_hz": sample["nyquist_hz"],
            "max_nodal_force_n": 0.0,
            "force_coefficient_max_abs": 0.0,
            "max_response_nodal_force_n": 0.0,
            "response_force_coefficient_max_abs": 0.0,
            "max_total_nodal_force_n": 0.0,
            "total_force_coefficient_max_abs": 0.0,
            "source_force_coefficient_max_abs": sample[
                "source_force_coefficient_max_abs"
            ],
            "max_structural_velocity_mps": 0.0,
            "force_power_sum_w": 0.0,
            "force_power_count": 0,
            "force_power_max_abs_w": 0.0,
            "re_lower_bound_warning_count": 0,
            "re_out_of_range_count": 0,
            "aoa_out_of_range_count": 0,
            "aliased_frequency_count": 0,
            "source_clip_count": 0,
            "force_coefficient_out_of_range_count": 0,
            "nonfinite_count": 0,
        }

    def commit_source_coverage(
        self,
        *,
        sample_index: int,
        applied_residual_force_global_n: Any,
        total_force_global_n: Any,
        response_force_global_n: Any,
        structural_velocity_global_mps: Any,
    ) -> None:
        """Commit prepared coverage only after the coupled step is accepted."""

        samples = self._last_coverage_sample
        if samples is None:
            return
        if type(sample_index) is not int or sample_index < 0:
            raise ValueError("H-VAWT source-coverage sample_index is invalid.")
        if self._coverage_last_sample_index == sample_index:
            return
        expected_sample_index = (
            0
            if self._coverage_last_sample_index is None
            else self._coverage_last_sample_index + 1
        )
        if sample_index != expected_sample_index:
            raise ValueError(
                "H-VAWT source-coverage sample index is not contiguous."
            )
        shape = (len(self.node_ids), 3)
        residual_force = np.asarray(applied_residual_force_global_n, dtype=float)
        total_force = np.asarray(total_force_global_n, dtype=float)
        response_force = np.asarray(response_force_global_n, dtype=float)
        structural_velocity = np.asarray(structural_velocity_global_mps, dtype=float)
        if any(
            value.shape != shape or np.any(~np.isfinite(value))
            for value in (
                residual_force,
                total_force,
                response_force,
                structural_velocity,
            )
        ):
            raise ValueError("H-VAWT source-coverage force inputs are invalid.")

        candidate = {identity: dict(row) for identity, row in self._coverage_rows.items()}
        residual_norm = np.linalg.norm(residual_force, axis=1)
        response_norm = np.linalg.norm(response_force, axis=1)
        total_norm = np.linalg.norm(total_force, axis=1)
        structural_speed = np.linalg.norm(structural_velocity, axis=1)
        force_power_all = np.einsum("ij,ij->i", total_force, structural_velocity)
        dynamic_force_all = np.empty(len(self.node_ids), dtype=float)
        for sample in samples:
            dynamic_force_all[sample["indices"]] = sample["dynamic_force_n"]
        residual_coeff_all = np.divide(
            residual_norm,
            dynamic_force_all,
            out=np.zeros_like(residual_norm),
            where=dynamic_force_all > _EPS,
        )
        response_coeff_all = np.divide(
            response_norm,
            dynamic_force_all,
            out=np.zeros_like(response_norm),
            where=dynamic_force_all > _EPS,
        )
        total_coeff_all = np.divide(
            total_norm,
            dynamic_force_all,
            out=np.zeros_like(total_norm),
            where=dynamic_force_all > _EPS,
        )
        coefficient_tolerance_all = (
            64.0
            * np.finfo(float).eps
            * np.maximum(1.0, self._source_force_coefficient_max_by_node)
        )
        for sample in samples:
            identity = str(sample["identity"])
            indices = sample["indices"]
            row = candidate.setdefault(identity, self._new_coverage_row(sample))
            residual_norm_group = residual_norm[indices]
            response_norm_group = response_norm[indices]
            total_norm_group = total_norm[indices]
            residual_coeff = residual_coeff_all[indices]
            response_coeff = response_coeff_all[indices]
            total_coeff = total_coeff_all[indices]
            force_power = force_power_all[indices]

            row["samples"] += 1
            row["node_samples"] += int(sample["node_count"])
            for name in (
                "aoa_min_deg",
                "re_min",
                "freq_min_hz",
            ):
                row[name] = min(float(row[name]), float(sample[name]))
            for name in (
                "aoa_max_deg",
                "re_max",
                "freq_max_hz",
                "unfiltered_freq_max_hz",
                "source_force_coefficient_max_abs",
            ):
                row[name] = max(float(row[name]), float(sample[name]))
            if float(sample["nyquist_hz"]) > 0.0:
                row["nyquist_hz"] = min(
                    float(row["nyquist_hz"]),
                    float(sample["nyquist_hz"]),
                )
            for name in (
                "frequency_truncated_count",
                "re_lower_bound_warning_count",
                "re_out_of_range_count",
                "aoa_out_of_range_count",
                "aliased_frequency_count",
                "source_clip_count",
                "nonfinite_count",
            ):
                row[name] += int(sample[name])
            row["force_coefficient_out_of_range_count"] += int(
                np.count_nonzero(
                    residual_coeff
                    > self._source_force_coefficient_max_by_node[indices]
                    + coefficient_tolerance_all[indices]
                )
            )
            row["max_nodal_force_n"] = max(
                float(row["max_nodal_force_n"]),
                float(np.max(residual_norm_group)),
            )
            row["force_coefficient_max_abs"] = max(
                float(row["force_coefficient_max_abs"]),
                float(np.max(residual_coeff)),
            )
            row["max_response_nodal_force_n"] = max(
                float(row["max_response_nodal_force_n"]),
                float(np.max(response_norm_group)),
            )
            row["response_force_coefficient_max_abs"] = max(
                float(row["response_force_coefficient_max_abs"]),
                float(np.max(response_coeff)),
            )
            row["max_total_nodal_force_n"] = max(
                float(row["max_total_nodal_force_n"]),
                float(np.max(total_norm_group)),
            )
            row["total_force_coefficient_max_abs"] = max(
                float(row["total_force_coefficient_max_abs"]),
                float(np.max(total_coeff)),
            )
            row["max_structural_velocity_mps"] = max(
                float(row["max_structural_velocity_mps"]),
                float(np.max(structural_speed[indices])),
            )
            row["force_power_sum_w"] += float(np.sum(force_power))
            row["force_power_count"] += int(indices.size)
            row["force_power_max_abs_w"] = max(
                float(row["force_power_max_abs_w"]),
                float(np.max(np.abs(force_power))),
            )
        self._coverage_rows = candidate
        self._coverage_last_sample_index = sample_index

    def source_coverage_summary(self) -> list[dict[str, Any]]:
        """Return accumulated H-VAWT spectral coverage for run receipts."""

        rows: list[dict[str, Any]] = []
        for identity, stored in sorted(self._coverage_rows.items()):
            row = dict(stored)
            row.pop("identity")
            count = int(row.pop("force_power_count"))
            force_power_sum = float(row.pop("force_power_sum_w"))
            row["force_power_mean_w"] = force_power_sum / count if count else 0.0
            freq_max = float(row["freq_max_hz"])
            nyquist = float(row["nyquist_hz"])
            row["nyquist_ratio"] = (
                nyquist / freq_max if freq_max > _EPS and nyquist > 0.0 else 1.0e12
            )
            rows.append(SourceCoverageRow.from_mapping(row).as_dict())
        return rows

    def step(
        self,
        time_s: float,
        *,
        base_relative_velocity_global_mps: Any,
        response_effective_crossflow_speed_mps: Any,
        response_section_width_m: Any,
        chord_unit_global: Any,
        normal_unit_global: Any,
        chord_m: Any,
        thickness_ratio: Any,
        local_length_m: Any,
        structural_velocity_global_mps: Any,
    ) -> HybridSpectralStepResult:
        time_value = _strict_real(time_s, "time_s")
        count = len(self.node_ids)

        def vectors(value: Any, label: str) -> np.ndarray:
            array = np.asarray(value, dtype=float)
            if array.shape != (count, 3) or np.any(~np.isfinite(array)):
                raise ValueError(f"{label} must be finite with shape ({count}, 3).")
            return array

        def scalars(value: Any, label: str, *, positive: bool) -> np.ndarray:
            array = np.asarray(value, dtype=float)
            if array.shape != (count,) or np.any(~np.isfinite(array)):
                raise ValueError(f"{label} must be finite with shape ({count},).")
            if positive and np.any(array <= 0.0):
                raise ValueError(f"{label} must be positive.")
            if not positive and np.any(array < 0.0):
                raise ValueError(f"{label} must be non-negative.")
            return array

        base_velocity = vectors(
            base_relative_velocity_global_mps,
            "base_relative_velocity_global_mps",
        )
        chord = vectors(chord_unit_global, "chord_unit_global")
        normal = vectors(normal_unit_global, "normal_unit_global")
        structural_velocity = vectors(
            structural_velocity_global_mps,
            "structural_velocity_global_mps",
        )
        response_speed = scalars(
            response_effective_crossflow_speed_mps,
            "response_effective_crossflow_speed_mps",
            positive=False,
        )
        response_width = scalars(
            response_section_width_m,
            "response_section_width_m",
            positive=True,
        )
        chord_length = scalars(chord_m, "chord_m", positive=True)
        thickness = scalars(thickness_ratio, "thickness_ratio", positive=False)
        length = scalars(local_length_m, "local_length_m", positive=False)
        signature = self._input_signature(
            time_value,
            (
                base_velocity,
                response_speed,
                response_width,
                chord,
                normal,
                chord_length,
                thickness,
                length,
                structural_velocity,
            ),
        )
        if self._last_time_s is not None:
            if time_value < self._last_time_s - 1.0e-12:
                raise ValueError("Hybrid spectral-source time decreased.")
            if math.isclose(time_value, self._last_time_s, rel_tol=0.0, abs_tol=1.0e-12):
                if signature != self._last_input_signature:
                    raise ValueError("Hybrid spectral-source inputs changed at the same time level.")
                if self._last_result is None:
                    raise RuntimeError("Hybrid spectral-source duplicate cache is incomplete.")
                return self._last_result

        timing = self.performance_timing
        started_ns = timing.now_ns() if timing is not None else 0
        flow = section_flow_state(
            base_velocity,
            chord,
            normal,
            chord_length,
            length,
            self.fluid_density_kgpm3,
            self.fluid_dynamic_viscosity_pas,
        )
        base_width = strouhal_reference_length(
            chord_length,
            flow.aoa_deg,
            thickness,
        )
        shape = (count, self.n_frequency_entries)
        st_cl = np.zeros(shape, dtype=float)
        amp_cl = np.zeros(shape, dtype=float)
        phase_cl = np.zeros(shape, dtype=float)
        st_cd = np.zeros(shape, dtype=float)
        amp_cd = np.zeros(shape, dtype=float)
        phase_cd = np.zeros(shape, dtype=float)
        for group in self._groups:
            indices = group["indices"]
            depth = group["depth"]
            spectra = interpolate_fft_spectra_batch_packed(
                group["airfoil"],
                flow.reynolds[indices],
                flow.aoa_deg[indices],
                ["CL", "CD"],
                n_freq_depth=depth,
                bounds_policy=self.source_bounds_policy,
                packed_tensor=group["packed_tensor"],
            )
            cl = spectra["CL"]
            cd = spectra["CD"]
            st_cl[indices, :depth], amp_cl[indices, :depth], phase_cl[indices, :depth] = cl
            st_cd[indices, :depth], amp_cd[indices, :depth], phase_cd[indices, :depth] = cd

        if timing is not None:
            timing.record_since("cl_cd_spectral_interpolation", started_ns)
            started_ns = timing.now_ns()

        partition = partition_lift_spectrum(
            strouhal=st_cl,
            amplitude=amp_cl,
            phase_rad=phase_cl,
            crossflow_speed_mps=response_speed,
            reference_width_m=response_width,
            modes=self.modes,
            modal_participation=self.modal_participation,
            capture=self.capture,
        )
        # The airfoil spectrum remains tied to the actual section inflow.  Only
        # the capture decision uses the rigid-body-relative response flow above;
        # otherwise a flexible-velocity perturbation could move spectral energy
        # into the structural mode *and* silently retune the uncaptured source.
        # Keeping these frequencies separate is the core hybrid-model split:
        # base flow drives the residual airfoil spectrum, while response flow
        # determines which part of that spectrum is pulled into lock-in.
        frequency_cl = st_cl * np.divide(
            flow.effective_velocity_mps,
            base_width,
            out=np.zeros_like(flow.effective_velocity_mps),
            where=base_width > 0.0,
        )[:, None]
        frequency_cd = st_cd * np.divide(
            flow.effective_velocity_mps,
            base_width,
            out=np.zeros_like(flow.effective_velocity_mps),
            where=base_width > 0.0,
        )[:, None]
        if timing is not None:
            timing.record_since(
                "captured_residual_partition_and_reconstruction",
                started_ns,
            )
            started_ns = timing.now_ns()
        coverage_sample = self._coverage_sample(
            time_s=time_value,
            flow_reynolds=flow.reynolds,
            flow_aoa_deg=flow.aoa_deg,
            frequency_cl_hz=frequency_cl,
            frequency_cd_hz=frequency_cd,
            amplitude_cl=amp_cl,
            amplitude_cd=amp_cd,
            section_dynamic_force_n=flow.section_dynamic_force_n,
        )
        if timing is not None:
            timing.record_since("source_coverage_preparation", started_ns)
            started_ns = timing.now_ns()
        residual_cl = np.array(partition.residual_amplitude, copy=True)
        residual_cd = np.array(amp_cd, copy=True)
        truncated_lift_energy = np.zeros(count, dtype=float)
        truncated_drag_energy = np.zeros(count, dtype=float)
        if self.source_frequency_cap_hz is not None:
            truncate_cl = frequency_cl > self.source_frequency_cap_hz
            truncate_cd = frequency_cd > self.source_frequency_cap_hz
            truncate_cl[:, 0] = False
            truncate_cd[:, 0] = False
            truncated_lift_energy = np.sum(
                np.where(truncate_cl, residual_cl**2, 0.0),
                axis=1,
            )
            truncated_drag_energy = np.sum(
                np.where(truncate_cd, residual_cd**2, 0.0),
                axis=1,
            )
            frequency_cl[truncate_cl] = 0.0
            frequency_cd[truncate_cd] = 0.0
            residual_cl[truncate_cl] = 0.0
            residual_cd[truncate_cd] = 0.0

        first = self._last_time_s is None
        if first:
            angle_cl = 2.0 * math.pi * frequency_cl * time_value + phase_cl
            angle_cd = 2.0 * math.pi * frequency_cd * time_value + phase_cd
        else:
            dt_s = time_value - float(self._last_time_s)
            angle_cl = self._phase_angle_cl_rad + (
                math.pi * (self._phase_frequency_cl_hz + frequency_cl) * dt_s
            )
            angle_cd = self._phase_angle_cd_rad + (
                math.pi * (self._phase_frequency_cd_hz + frequency_cd) * dt_s
            )
        chord_force = harmonic_force_channel_from_angle(
            flow.section_dynamic_force_n,
            residual_cd,
            angle_cd,
            include_mean_load=False,
        )
        normal_force = harmonic_force_channel_from_angle(
            flow.section_dynamic_force_n,
            residual_cl,
            angle_cl,
            include_mean_load=False,
        )
        residual_force = force_from_chord_normal(
            chord_force,
            normal_force,
            chord,
            normal,
        )
        if np.any(~np.isfinite(residual_force)):
            raise FloatingPointError("Residual spectral force became non-finite.")
        result = HybridSpectralStepResult(
            residual_force_global_n=residual_force,
            partition=partition,
            base_aoa_deg=flow.aoa_deg,
            base_reynolds=flow.reynolds,
            base_effective_crossflow_speed_mps=flow.effective_velocity_mps,
            base_section_width_m=base_width,
            applied_residual_lift_energy_by_node=np.sum(
                residual_cl[:, 1:] ** 2,
                axis=1,
            ),
            truncated_residual_lift_energy_by_node=truncated_lift_energy,
            truncated_residual_drag_energy_by_node=truncated_drag_energy,
            maximum_residual_nodal_force_n=float(
                np.max(np.linalg.norm(residual_force, axis=1))
            ),
            residual_force_power_w=float(np.sum(residual_force * structural_velocity)),
        )
        if timing is not None:
            timing.record_since(
                "captured_residual_partition_and_reconstruction",
                started_ns,
            )

        self._phase_angle_cl_rad = np.array(angle_cl, copy=True)
        self._phase_angle_cd_rad = np.array(angle_cd, copy=True)
        self._phase_frequency_cl_hz = np.array(frequency_cl, copy=True)
        self._phase_frequency_cd_hz = np.array(frequency_cd, copy=True)
        self._last_time_s = time_value
        self._last_input_signature = signature
        self._last_coverage_sample = coverage_sample
        # Both result dataclasses are frozen and every contained array was
        # copied into a read-only buffer in ``__post_init__``.  Sharing this
        # immutable result between the cache and caller avoids two complete
        # copies per accepted timestep without exposing mutable model state.
        self._last_result = result
        return result

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "config_fingerprint": self.config_fingerprint,
            "last_time_s": self._last_time_s,
            "last_input_signature": self._last_input_signature,
            "phase_angle_cl_rad": self._phase_angle_cl_rad.tolist(),
            "phase_angle_cd_rad": self._phase_angle_cd_rad.tolist(),
            "phase_frequency_cl_hz": self._phase_frequency_cl_hz.tolist(),
            "phase_frequency_cd_hz": self._phase_frequency_cd_hz.tolist(),
            "coverage_last_sample_index": self._coverage_last_sample_index,
            "coverage_rows": [
                dict(row) for _identity, row in sorted(self._coverage_rows.items())
            ],
        }

    def transaction_snapshot(
        self,
    ) -> tuple[object, ...]:
        """Return the accepted in-process state without JSON conversion.

        Successful steps replace phase arrays and the cached result rather
        than mutating the accepted objects.  Retaining those objects by
        reference therefore provides exact rollback while persistent restart
        checkpoints continue to use ``checkpoint_state``.
        """

        return (
            self._phase_angle_cl_rad,
            self._phase_angle_cd_rad,
            self._phase_frequency_cl_hz,
            self._phase_frequency_cd_hz,
            self._last_time_s,
            self._last_input_signature,
            self._last_result,
            self._last_coverage_sample,
            self._coverage_last_sample_index,
            self._coverage_rows,
        )

    def restore_transaction_snapshot(
        self,
        snapshot: tuple[object, ...],
    ) -> None:
        """Restore a snapshot after a downstream coupled-step rejection."""

        if not isinstance(snapshot, tuple) or len(snapshot) != 10:
            raise TypeError("Hybrid spectral transaction snapshot is invalid.")
        (
            self._phase_angle_cl_rad,
            self._phase_angle_cd_rad,
            self._phase_frequency_cl_hz,
            self._phase_frequency_cd_hz,
            self._last_time_s,
            self._last_input_signature,
            self._last_result,
            self._last_coverage_sample,
            self._coverage_last_sample_index,
            self._coverage_rows,
        ) = snapshot  # type: ignore[assignment]

    def restore_checkpoint_state(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError("Hybrid spectral-source checkpoint must be a mapping.")
        expected = {
            "schema_version",
            "config_fingerprint",
            "last_time_s",
            "last_input_signature",
            "phase_angle_cl_rad",
            "phase_angle_cd_rad",
            "phase_frequency_cl_hz",
            "phase_frequency_cd_hz",
            "coverage_last_sample_index",
            "coverage_rows",
        }
        if state.get("schema_version") != 2 or set(state) != expected:
            raise ValueError("Hybrid spectral-source checkpoint keys differ.")
        if state["config_fingerprint"] != self.config_fingerprint:
            raise ValueError("Hybrid spectral-source checkpoint identity mismatch.")
        last_time = state["last_time_s"]
        signature = state["last_input_signature"]
        if (last_time is None) != (signature is None):
            raise ValueError("Hybrid spectral-source checkpoint time/signature mismatch.")
        if last_time is not None:
            last_time = _strict_real(last_time, "last_time_s")
            if not isinstance(signature, str) or len(signature) != 64:
                raise ValueError("Hybrid spectral-source checkpoint signature is invalid.")
        shape = (len(self.node_ids), self.n_frequency_entries)
        restored: dict[str, np.ndarray] = {}
        for key in (
            "phase_angle_cl_rad",
            "phase_angle_cd_rad",
            "phase_frequency_cl_hz",
            "phase_frequency_cd_hz",
        ):
            array = np.asarray(state[key], dtype=float)
            if array.shape != shape or np.any(~np.isfinite(array)):
                raise ValueError(f"Hybrid spectral-source checkpoint {key} is invalid.")
            if "frequency" in key and np.any(array < 0.0):
                raise ValueError(f"Hybrid spectral-source checkpoint {key} is negative.")
            restored[key] = np.array(array, copy=True)
        coverage_last_index = state["coverage_last_sample_index"]
        raw_rows = state["coverage_rows"]
        if not isinstance(raw_rows, list):
            raise TypeError("Hybrid spectral-source coverage_rows must be a list.")
        if (coverage_last_index is None) != (not raw_rows):
            raise ValueError("Hybrid spectral-source coverage index/rows mismatch.")
        if coverage_last_index is not None and (
            type(coverage_last_index) is not int or coverage_last_index < 0
        ):
            raise ValueError("Hybrid spectral-source coverage index is invalid.")

        valid_groups = {
            str(group["component"]): group for group in self._coverage_groups
        }
        row_keys = {
            "identity",
            "component",
            "airfoil_id",
            "reynolds_model",
            "samples",
            "node_samples",
            "aoa_min_deg",
            "aoa_max_deg",
            "source_aoa_min_deg",
            "source_aoa_max_deg",
            "re_min",
            "re_max",
            "source_re_min",
            "source_re_max",
            "freq_min_hz",
            "freq_max_hz",
            "unfiltered_freq_max_hz",
            "frequency_truncated_count",
            "nyquist_hz",
            "max_nodal_force_n",
            "force_coefficient_max_abs",
            "max_response_nodal_force_n",
            "response_force_coefficient_max_abs",
            "max_total_nodal_force_n",
            "total_force_coefficient_max_abs",
            "source_force_coefficient_max_abs",
            "max_structural_velocity_mps",
            "force_power_sum_w",
            "force_power_count",
            "force_power_max_abs_w",
            "re_lower_bound_warning_count",
            "re_out_of_range_count",
            "aoa_out_of_range_count",
            "aliased_frequency_count",
            "source_clip_count",
            "force_coefficient_out_of_range_count",
            "nonfinite_count",
        }
        restored_rows: dict[str, dict[str, Any]] = {}
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                raise TypeError("Hybrid spectral-source coverage row must be a mapping.")
            if set(raw_row) != row_keys:
                raise ValueError("Hybrid spectral-source coverage row keys differ.")
            identity = raw_row.get("identity")
            if identity not in valid_groups or identity in restored_rows:
                raise ValueError("Hybrid spectral-source coverage identity is invalid.")
            if raw_row.get("component") != identity:
                raise ValueError("Hybrid spectral-source coverage component is invalid.")
            group = valid_groups[str(identity)]
            if (
                raw_row.get("airfoil_id") != group["airfoil_id"]
                or raw_row.get("reynolds_model") != group["reynolds_model"]
            ):
                raise ValueError("Hybrid spectral-source coverage source identity differs.")
            if type(raw_row.get("samples")) is not int or raw_row["samples"] < 1:
                raise ValueError("Hybrid spectral-source coverage samples are invalid.")
            if raw_row.get("node_samples") != (
                raw_row["samples"] * int(group["indices"].size)
            ):
                raise ValueError("Hybrid spectral-source coverage node_samples is invalid.")
            row = dict(raw_row)
            for name, value in row.items():
                if name in {"identity", "component", "airfoil_id", "reynolds_model"}:
                    if not isinstance(value, str) or (
                        name != "reynolds_model" and not value
                    ):
                        raise ValueError(
                            f"Hybrid spectral-source coverage {name} is invalid."
                        )
                elif name.endswith("_count") or name in {"samples", "node_samples"}:
                    if type(value) is not int or value < 0:
                        raise ValueError(
                            f"Hybrid spectral-source coverage {name} is invalid."
                        )
                elif isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise ValueError(
                        f"Hybrid spectral-source coverage {name} is invalid."
                    )
                elif not math.isfinite(float(value)):
                    raise ValueError(
                        f"Hybrid spectral-source coverage {name} is nonfinite."
                    )
            restored_rows[str(identity)] = row

        if raw_rows:
            expected_identities = set(valid_groups)
            if set(restored_rows) != expected_identities:
                raise ValueError(
                    "Hybrid spectral-source coverage component set is incomplete."
                )
            expected_samples = int(coverage_last_index) + 1
            if any(
                row["samples"] != expected_samples
                for row in restored_rows.values()
            ):
                raise ValueError(
                    "Hybrid spectral-source coverage sample/index counts differ."
                )

        # Commit only after the complete checkpoint, including coverage, has
        # been parsed and validated.  A malformed receipt cannot partially
        # mutate accepted phase/filter state.
        self._phase_angle_cl_rad = restored["phase_angle_cl_rad"]
        self._phase_angle_cd_rad = restored["phase_angle_cd_rad"]
        self._phase_frequency_cl_hz = restored["phase_frequency_cl_hz"]
        self._phase_frequency_cd_hz = restored["phase_frequency_cd_hz"]
        self._last_time_s = last_time
        self._last_input_signature = signature
        self._last_result = None
        self._last_coverage_sample = None
        self._coverage_last_sample_index = coverage_last_index
        self._coverage_rows = restored_rows


def _validated_spectrum(
    strouhal: Any,
    amplitude: Any,
    phase_rad: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = np.asarray(strouhal, dtype=float)
    amp = np.asarray(amplitude, dtype=float)
    phase = np.asarray(phase_rad, dtype=float)
    if st.ndim != 2 or amp.shape != st.shape or phase.shape != st.shape:
        raise ValueError(
            "strouhal, amplitude, and phase_rad must share shape [node, harmonic]."
        )
    if st.shape[0] < 1 or st.shape[1] < 1:
        raise ValueError("The spectrum must contain at least one node and DC entry.")
    if np.any(~np.isfinite(st)) or np.any(~np.isfinite(amp)) or np.any(~np.isfinite(phase)):
        raise ValueError("Spectrum arrays must contain only finite values.")
    if np.any(st < 0.0) or np.any(amp[:, 1:] < 0.0):
        raise ValueError(
            "Strouhal numbers and harmonic amplitudes must be non-negative; "
            "the DC coefficient may be signed."
        )
    if np.any(np.abs(st[:, 0]) > _EPS):
        raise ValueError("Spectral index zero must be the DC entry with St=0.")
    active_harmonic = amp[:, 1:] > _EPS
    if np.any(active_harmonic & (st[:, 1:] <= _EPS)):
        raise ValueError("Nonzero harmonic amplitudes require positive Strouhal numbers.")
    return st, amp, phase


def partition_lift_spectrum(
    *,
    strouhal: Any,
    amplitude: Any,
    phase_rad: Any,
    crossflow_speed_mps: Any,
    reference_width_m: Any,
    modes: Sequence[SpectralLockInModeConfig],
    modal_participation: Any | None = None,
    capture: SpectralLockInCaptureConfig | None = None,
) -> SpectralPartitionResult:
    """Partition measured lift-spectrum energy between modes and residual.

    ``modal_participation`` has shape ``[node, mode]`` and is used only to
    resolve competition when capture windows overlap.  With one eligible mode,
    its calibrated capture fraction is unchanged at every nonzero modal site.
    This avoids inadvertently multiplying the physical mode-shape weighting
    that is already present in the generalized force projection.
    """

    st, amp, phase = _validated_spectrum(strouhal, amplitude, phase_rad)
    node_count, harmonic_count = st.shape
    mode_rows = tuple(modes)
    if not mode_rows or any(
        not isinstance(mode, SpectralLockInModeConfig) for mode in mode_rows
    ):
        raise TypeError("modes must contain SpectralLockInModeConfig objects.")
    if len({mode.tracked_mode for mode in mode_rows}) != len(mode_rows):
        raise ValueError("tracked_mode values must be unique.")
    capture_config = capture or SpectralLockInCaptureConfig()
    if not isinstance(capture_config, SpectralLockInCaptureConfig):
        raise TypeError("capture must be SpectralLockInCaptureConfig.")

    speed = np.asarray(crossflow_speed_mps, dtype=float)
    width = np.asarray(reference_width_m, dtype=float)
    if speed.shape != (node_count,) or width.shape != (node_count,):
        raise ValueError("crossflow_speed_mps and reference_width_m must have shape [node].")
    if np.any(~np.isfinite(speed)) or np.any(speed < 0.0):
        raise ValueError("crossflow_speed_mps must be finite and non-negative.")
    if np.any(~np.isfinite(width)) or np.any(width <= 0.0):
        raise ValueError("reference_width_m must be finite and positive.")
    if modal_participation is None:
        participation = np.ones((node_count, len(mode_rows)), dtype=float)
    else:
        participation = np.asarray(modal_participation, dtype=float)
        if participation.shape != (node_count, len(mode_rows)):
            raise ValueError("modal_participation must have shape [node, mode].")
        if np.any(~np.isfinite(participation)) or np.any(participation < 0.0):
            raise ValueError("modal_participation must be finite and non-negative.")
        maximum = np.max(participation, axis=0)
        if np.any(maximum <= 0.0):
            raise ValueError("Every mode must have positive participation at at least one node.")
        participation = participation / maximum[None, :]

    source_frequency = st * np.divide(
        speed,
        width,
        out=np.zeros_like(speed),
        where=width > 0.0,
    )[:, None]
    natural_frequency = np.asarray(
        [mode.natural_frequency_hz for mode in mode_rows],
        dtype=float,
    )
    ratio = source_frequency[:, :, None] / natural_frequency[None, None, :]
    if capture_config.capture_window == "compact_sine_squared":
        capture_amplitude = capture_weight(
            ratio,
            low=capture_config.frequency_ratio_low,
            peak=capture_config.frequency_ratio_peak,
            high=capture_config.frequency_ratio_high,
        )
    else:
        capture_amplitude = smooth_capture_amplitude(
            ratio,
            capture_config.frequency_ratio_low,
            capture_config.frequency_ratio_peak,
            capture_config.frequency_ratio_high,
        )
    capture_amplitude[:, 0, :] = 0.0
    capture_amplitude = np.where(amp[:, :, None] > _EPS, capture_amplitude, 0.0)

    eligible = participation > capture_config.minimum_modal_participation
    demand = capture_amplitude**2
    weighted_demand = demand * participation[:, None, :] * eligible[:, None, :]
    total_requested = np.minimum(1.0, np.sum(demand * eligible[:, None, :], axis=2))
    denominator = np.sum(weighted_demand, axis=2)
    energy_fraction = np.divide(
        weighted_demand,
        denominator[:, :, None],
        out=np.zeros_like(weighted_demand),
        where=denominator[:, :, None] > 0.0,
    ) * total_requested[:, :, None]
    total_capture_fraction = np.sum(energy_fraction, axis=2)
    if np.any(total_capture_fraction > 1.0 + 5.0e-15):
        raise FloatingPointError("Captured spectral energy exceeds source energy.")
    total_capture_fraction = np.clip(total_capture_fraction, 0.0, 1.0)

    residual_amplitude = np.array(amp, copy=True)
    residual_amplitude[:, 1:] *= np.sqrt(
        np.maximum(0.0, 1.0 - total_capture_fraction[:, 1:])
    )
    harmonic_energy = amp**2
    captured_energy = harmonic_energy[:, :, None] * energy_fraction
    captured_amplitude = np.sqrt(np.sum(captured_energy, axis=1))
    spectral_gain = captured_amplitude / capture_config.reference_lift_amplitude
    if np.any(spectral_gain > capture_config.maximum_spectral_gain + 1.0e-12):
        observed = float(np.max(spectral_gain))
        raise ValueError(
            f"Spectral gain {observed:.17g} exceeds maximum_spectral_gain "
            f"{capture_config.maximum_spectral_gain:.17g}."
        )

    energy_weighted_frequency = np.sum(
        captured_energy * source_frequency[:, :, None],
        axis=1,
    )
    captured_energy_by_node_mode = np.sum(captured_energy, axis=1)
    source_centroid = np.divide(
        energy_weighted_frequency,
        captured_energy_by_node_mode,
        out=np.zeros_like(energy_weighted_frequency),
        where=captured_energy_by_node_mode > 0.0,
    )
    pulled_frequency = np.where(
        captured_energy_by_node_mode > 0.0,
        natural_frequency[None, :],
        0.0,
    )

    # DC is a signed/mean coefficient in the source schema and is neither
    # captured nor counted as oscillatory energy.
    source_energy_by_node = np.sum(harmonic_energy[:, 1:], axis=1)
    residual_energy_by_node = np.sum(residual_amplitude[:, 1:] ** 2, axis=1)
    captured_energy_by_node = np.sum(captured_energy[:, 1:, :], axis=(1, 2))
    closure_error = np.abs(
        source_energy_by_node - residual_energy_by_node - captured_energy_by_node
    )
    maximum_closure = float(np.max(closure_error))
    tolerance = 2.0e-12 * max(1.0, float(np.max(source_energy_by_node)))
    if maximum_closure > tolerance:
        raise FloatingPointError(
            f"Spectral energy partition error {maximum_closure:.17g} exceeds {tolerance:.17g}."
        )

    return SpectralPartitionResult(
        source_frequency_hz=source_frequency,
        source_amplitude=amp,
        source_phase_rad=phase,
        residual_amplitude=residual_amplitude,
        capture_amplitude_fraction=capture_amplitude,
        capture_energy_fraction_by_mode=energy_fraction,
        captured_amplitude_by_node_mode=captured_amplitude,
        spectral_gain_by_node_mode=spectral_gain,
        source_centroid_frequency_hz_by_node_mode=source_centroid,
        pulled_frequency_hz_by_node_mode=pulled_frequency,
        total_source_energy_by_node=source_energy_by_node,
        total_residual_energy_by_node=residual_energy_by_node,
        total_captured_energy_by_node=captured_energy_by_node,
        maximum_energy_closure_error=maximum_closure,
    )


def evaluate_modal_damping_balance(
    *,
    partition: SpectralPartitionResult,
    modes: Sequence[SpectralLockInModeConfig],
    real_mode_shape: Any,
    section_dynamic_prefactor_nspm: Any,
) -> tuple[ModalDampingBalance, ...]:
    """Compare captured aerodynamic negative damping with Rayleigh damping.

    ``section_dynamic_prefactor_nspm`` is ``0.5*rho*U*D*length`` and has the
    units required by the response force ``F=prefactor*H1*gain*phi*qdot``.
    The returned coefficient is therefore the generalized modal coefficient in
    N s/m.  It is diagnostic only; QBlade already applies the structural term.
    """

    if not isinstance(partition, SpectralPartitionResult):
        raise TypeError("partition must be SpectralPartitionResult.")
    mode_rows = tuple(modes)
    if not mode_rows or any(
        not isinstance(mode, SpectralLockInModeConfig) for mode in mode_rows
    ):
        raise TypeError("modes must contain SpectralLockInModeConfig objects.")
    node_count, mode_count = partition.spectral_gain_by_node_mode.shape
    if len(mode_rows) != mode_count:
        raise ValueError("Mode count differs from the spectral partition.")
    shape = np.asarray(real_mode_shape, dtype=float)
    if shape.shape != (node_count, 3, mode_count):
        raise ValueError("real_mode_shape must have shape [node, 3, mode].")
    if np.any(~np.isfinite(shape)):
        raise ValueError("real_mode_shape must be finite.")
    prefactor = np.asarray(section_dynamic_prefactor_nspm, dtype=float)
    if prefactor.shape != (node_count,):
        raise ValueError("section_dynamic_prefactor_nspm must have shape [node].")
    if np.any(~np.isfinite(prefactor)) or np.any(prefactor < 0.0):
        raise ValueError("section_dynamic_prefactor_nspm must be finite and non-negative.")

    shape_norm_squared = np.sum(shape**2, axis=1)
    result: list[ModalDampingBalance] = []
    for mode_index, mode in enumerate(mode_rows):
        omega = 2.0 * math.pi * mode.natural_frequency_hz
        structural = (
            2.0
            * mode.structural_damping_ratio
            * mode.generalized_modal_mass_kg
            * omega
        )
        aerodynamic = float(
            np.sum(
                prefactor
                * mode.aerodynamic_damping_gain_h1
                * partition.spectral_gain_by_node_mode[:, mode_index]
                * shape_norm_squared[:, mode_index]
            )
        )
        if structural > 0.0:
            damping_ratio = aerodynamic / structural
            available = aerodynamic > structural
        else:
            damping_ratio = math.inf if aerodynamic > 0.0 else 0.0
            available = aerodynamic > 0.0
        equilibrium_ratio = (
            mode.saturation_amplitude_ratio
            * math.sqrt(max(0.0, 1.0 - structural / aerodynamic))
            if aerodynamic > structural and aerodynamic > 0.0
            else 0.0
        )
        result.append(
            ModalDampingBalance(
                tracked_mode=mode.tracked_mode,
                structural_damping_coefficient_nspm=structural,
                captured_aerodynamic_damping_coefficient_nspm=aerodynamic,
                aerodynamic_to_structural_damping_ratio=damping_ratio,
                linear_lockin_available=available,
                predicted_equilibrium_amplitude_m=(
                    equilibrium_ratio * mode.reference_length_m
                ),
                predicted_equilibrium_amplitude_ratio=equilibrium_ratio,
            )
        )
    return tuple(result)


__all__ = [
    "CYLINDER_REFERENCE_LIFT_AMPLITUDE",
    "CYLINDER_REFERENCE_STROUHAL",
    "CYLINDER_REDUCED_VELOCITY_HIGH",
    "CYLINDER_REDUCED_VELOCITY_LOW",
    "CYLINDER_REDUCED_VELOCITY_PEAK",
    "DEFAULT_CAPTURE_FREQUENCY_RATIO_HIGH",
    "DEFAULT_CAPTURE_FREQUENCY_RATIO_LOW",
    "DEFAULT_CAPTURE_FREQUENCY_RATIO_PEAK",
    "HVAWTSpectralLockInSource",
    "HybridSpectralStepResult",
    "ModalDampingBalance",
    "SpectralLockInCaptureConfig",
    "SpectralLockInModeConfig",
    "SpectralPartitionResult",
    "airfoil_fft_content_sha256",
    "evaluate_modal_damping_balance",
    "partition_lift_spectrum",
    "smooth_capture_amplitude",
]
