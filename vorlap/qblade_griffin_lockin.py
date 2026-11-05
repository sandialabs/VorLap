"""Static-spectrum-scaled Gao--Scanlan reciprocal-Griffin VIV closure.

The fixed-body spectrum supplies two things only: the shedding-frequency ratio
and a sinusoidal-equivalent fluctuating cross-flow coefficient.  Motion-correlated
force per unit span is

``f = F_s * w * (G_c sign(v) - G_v v/U)``,

where ``F_s=0.5*rho*U**2*D*C_eff`` and ``w`` is a compact frequency-capture
weight.  The Coulomb-like term supplies positive cycle work at small amplitude;
the velocity-linear term supplies restoring aerodynamic damping at large
amplitude.  For harmonic velocity with peak ``V``,

``<f v> = F_s*w*(2*G_c*V/pi - G_v*V**2/(2U))``.

Balancing this work with ``0.5*c_s*V**2`` produces the reciprocal-amplitude
Griffin trend.  The QBlade bridge may smooth ``sign(v)`` at the origin, but the
closed-form calibration and validation use the declared ideal-cycle law.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MODEL_TYPE = "static_spectrum_gao_scanlan_griffin_v1"
CALIBRATION_SCHEMA = "vorlap-gao-scanlan-griffin-calibration-v1"


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class GriffinTrainingPoint:
    natural_frequency_hz: float
    mass_per_span_kgpm: float
    structural_damping_ratio: float
    air_density_kgpm3: float
    crossflow_depth_m: float
    reduced_frequency_omega_d_over_u: float
    peak_amplitude_over_depth: float

    def __post_init__(self) -> None:
        for name in (
            "natural_frequency_hz",
            "mass_per_span_kgpm",
            "structural_damping_ratio",
            "air_density_kgpm3",
            "crossflow_depth_m",
            "reduced_frequency_omega_d_over_u",
            "peak_amplitude_over_depth",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive.")

    @property
    def omega_radps(self) -> float:
        return 2.0 * math.pi * self.natural_frequency_hz

    @property
    def wind_speed_mps(self) -> float:
        return (
            self.omega_radps
            * self.crossflow_depth_m
            / self.reduced_frequency_omega_d_over_u
        )

    @property
    def structural_damping_kgpspm(self) -> float:
        return (
            2.0
            * self.structural_damping_ratio
            * self.mass_per_span_kgpm
            * self.omega_radps
        )


@dataclass(frozen=True)
class GriffinFit:
    coulomb_work_gain: float
    velocity_damping_gain: float
    maximum_training_relative_error: float
    rank: int


def fit_griffin_gains(
    points: Sequence[GriffinTrainingPoint],
    *,
    sinusoidal_equivalent_coefficient: float,
) -> GriffinFit:
    """Fit ``G_c`` and ``G_v`` by linear least squares in reciprocal amplitude."""

    coefficient = _finite(
        sinusoidal_equivalent_coefficient, "sinusoidal_equivalent_coefficient"
    )
    if coefficient <= 0.0 or len(points) < 2:
        raise ValueError(
            "A positive force coefficient and at least two points are required."
        )
    matrix = []
    target = []
    for point in points:
        speed = point.wind_speed_mps
        omega = point.omega_radps
        force_scale = (
            0.5
            * point.air_density_kgpm3
            * speed**2
            * point.crossflow_depth_m
            * coefficient
        )
        # 1/A = a*(1/Gc) + b*(Gv/Gc), with A in metres.
        matrix.append(
            [
                math.pi * omega * point.structural_damping_kgpspm / (4.0 * force_scale),
                math.pi * omega / (4.0 * speed),
            ]
        )
        target.append(1.0 / (point.peak_amplitude_over_depth * point.crossflow_depth_m))
    solution, _, rank, _ = np.linalg.lstsq(
        np.asarray(matrix, dtype=float), np.asarray(target, dtype=float), rcond=None
    )
    inverse_coulomb_gain, velocity_over_coulomb = map(float, solution)
    if rank != 2 or inverse_coulomb_gain <= 0.0 or velocity_over_coulomb <= 0.0:
        raise ValueError("Training points do not identify positive Griffin gains.")
    coulomb = 1.0 / inverse_coulomb_gain
    velocity = velocity_over_coulomb * coulomb
    predictions = [
        predict_stable_amplitude_ratio(
            parameters=GriffinLockInParameters(
                coulomb_work_gain=coulomb,
                velocity_damping_gain=velocity,
                frequency_ratio_low=1.0,
                frequency_ratio_peak=1.166666667,
                frequency_ratio_high=1.21875,
                regularization_velocity_over_wind=1.0e-5,
            ),
            natural_frequency_hz=point.natural_frequency_hz,
            mass_per_span_kgpm=point.mass_per_span_kgpm,
            structural_damping_ratio=point.structural_damping_ratio,
            air_density_kgpm3=point.air_density_kgpm3,
            crossflow_depth_m=point.crossflow_depth_m,
            wind_speed_mps=point.wind_speed_mps,
            sinusoidal_equivalent_coefficient=coefficient,
            frequency_ratio=1.166666667,
        )
        for point in points
    ]
    errors = [
        abs(predicted - point.peak_amplitude_over_depth)
        / point.peak_amplitude_over_depth
        for predicted, point in zip(predictions, points, strict=True)
    ]
    return GriffinFit(
        coulomb_work_gain=coulomb,
        velocity_damping_gain=velocity,
        maximum_training_relative_error=max(errors),
        rank=int(rank),
    )


@dataclass(frozen=True)
class GriffinLockInParameters:
    coulomb_work_gain: float
    velocity_damping_gain: float
    frequency_ratio_low: float
    frequency_ratio_peak: float
    frequency_ratio_high: float
    regularization_velocity_over_wind: float

    def __post_init__(self) -> None:
        for name in (
            "coulomb_work_gain",
            "velocity_damping_gain",
            "frequency_ratio_low",
            "frequency_ratio_peak",
            "frequency_ratio_high",
            "regularization_velocity_over_wind",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if not (
            self.frequency_ratio_low
            < self.frequency_ratio_peak
            < self.frequency_ratio_high
        ):
            raise ValueError("Frequency bounds must satisfy low < peak < high.")
        if self.regularization_velocity_over_wind >= 1.0e-2:
            raise ValueError(
                "Origin regularization must remain a small velocity ratio."
            )

    def capture_weight(self, frequency_ratio: float) -> float:
        ratio = _finite(frequency_ratio, "frequency_ratio")
        if ratio <= self.frequency_ratio_low or ratio >= self.frequency_ratio_high:
            return 0.0
        if ratio <= self.frequency_ratio_peak:
            phase = (ratio - self.frequency_ratio_low) / (
                self.frequency_ratio_peak - self.frequency_ratio_low
            )
            return math.sin(0.5 * math.pi * phase) ** 2
        phase = (ratio - self.frequency_ratio_peak) / (
            self.frequency_ratio_high - self.frequency_ratio_peak
        )
        return math.cos(0.5 * math.pi * phase) ** 2


def motion_correlated_force_per_span(
    *,
    parameters: GriffinLockInParameters,
    modal_velocity_mps: float,
    wind_speed_mps: float,
    air_density_kgpm3: float,
    crossflow_depth_m: float,
    sinusoidal_equivalent_coefficient: float,
    frequency_ratio: float,
    regularized: bool = True,
) -> float:
    """Return cross-flow force per span; positive work means force follows velocity."""

    velocity = _finite(modal_velocity_mps, "modal_velocity_mps")
    speed = _finite(wind_speed_mps, "wind_speed_mps")
    density = _finite(air_density_kgpm3, "air_density_kgpm3")
    depth = _finite(crossflow_depth_m, "crossflow_depth_m")
    coefficient = _finite(
        sinusoidal_equivalent_coefficient, "sinusoidal_equivalent_coefficient"
    )
    if speed <= 0.0 or density <= 0.0 or depth <= 0.0 or coefficient < 0.0:
        raise ValueError("Flow, geometry, and spectral coefficient are invalid.")
    weight = parameters.capture_weight(frequency_ratio)
    if weight == 0.0 or coefficient == 0.0 or velocity == 0.0:
        return 0.0
    if regularized:
        epsilon = parameters.regularization_velocity_over_wind * speed
        direction = velocity / math.sqrt(velocity**2 + epsilon**2)
    else:
        direction = math.copysign(1.0, velocity)
    force_scale = 0.5 * density * speed**2 * depth * coefficient * weight
    return force_scale * (
        parameters.coulomb_work_gain * direction
        - parameters.velocity_damping_gain * velocity / speed
    )


def ideal_cycle_mean_aerodynamic_power_per_span(
    *,
    parameters: GriffinLockInParameters,
    peak_velocity_mps: float,
    wind_speed_mps: float,
    air_density_kgpm3: float,
    crossflow_depth_m: float,
    sinusoidal_equivalent_coefficient: float,
    frequency_ratio: float,
) -> float:
    peak_velocity = _finite(peak_velocity_mps, "peak_velocity_mps")
    if peak_velocity < 0.0:
        raise ValueError("peak_velocity_mps must be non-negative.")
    speed = _finite(wind_speed_mps, "wind_speed_mps")
    density = _finite(air_density_kgpm3, "air_density_kgpm3")
    depth = _finite(crossflow_depth_m, "crossflow_depth_m")
    coefficient = _finite(
        sinusoidal_equivalent_coefficient, "sinusoidal_equivalent_coefficient"
    )
    if speed <= 0.0 or density <= 0.0 or depth <= 0.0 or coefficient < 0.0:
        raise ValueError("Flow, geometry, and spectral coefficient are invalid.")
    force_scale = (
        0.5
        * density
        * speed**2
        * depth
        * coefficient
        * parameters.capture_weight(frequency_ratio)
    )
    return force_scale * (
        2.0 * parameters.coulomb_work_gain * peak_velocity / math.pi
        - parameters.velocity_damping_gain * peak_velocity**2 / (2.0 * speed)
    )


def predict_stable_amplitude_ratio(
    *,
    parameters: GriffinLockInParameters,
    natural_frequency_hz: float,
    mass_per_span_kgpm: float,
    structural_damping_ratio: float,
    air_density_kgpm3: float,
    crossflow_depth_m: float,
    wind_speed_mps: float,
    sinusoidal_equivalent_coefficient: float,
    frequency_ratio: float,
) -> float:
    frequency = _finite(natural_frequency_hz, "natural_frequency_hz")
    mass = _finite(mass_per_span_kgpm, "mass_per_span_kgpm")
    damping = _finite(structural_damping_ratio, "structural_damping_ratio")
    density = _finite(air_density_kgpm3, "air_density_kgpm3")
    depth = _finite(crossflow_depth_m, "crossflow_depth_m")
    speed = _finite(wind_speed_mps, "wind_speed_mps")
    coefficient = _finite(
        sinusoidal_equivalent_coefficient, "sinusoidal_equivalent_coefficient"
    )
    if min(frequency, mass, damping, density, depth, speed) <= 0.0 or coefficient < 0.0:
        raise ValueError("Structure, flow, and geometry values are invalid.")
    weight = parameters.capture_weight(frequency_ratio)
    if weight == 0.0 or coefficient == 0.0:
        return 0.0
    omega = 2.0 * math.pi * frequency
    structural = 2.0 * damping * mass * omega
    force_scale = 0.5 * density * speed**2 * depth * coefficient * weight
    peak_velocity = (
        4.0
        * force_scale
        * parameters.coulomb_work_gain
        / (
            math.pi
            * (structural + force_scale * parameters.velocity_damping_gain / speed)
        )
    )
    return peak_velocity / (omega * depth)


@dataclass(frozen=True)
class GriffinCalibration:
    raw: Mapping[str, Any]
    fingerprint: str
    parameters: GriffinLockInParameters

    @classmethod
    def from_json(cls, path: Path) -> "GriffinCalibration":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "schema",
            "status",
            "model_type",
            "method",
            "parameters",
            "training",
            "sources",
            "claim_boundary",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("Griffin calibration keys differ from the schema.")
        if (
            raw["schema"] != CALIBRATION_SCHEMA
            or raw["status"] != "trained_4x1_ready_for_untouched_validation"
            or raw["model_type"] != MODEL_TYPE
        ):
            raise ValueError("Unsupported or improperly staged Griffin calibration.")
        values = raw["parameters"]
        parameters = GriffinLockInParameters(
            coulomb_work_gain=float(values["coulomb_work_gain"]),
            velocity_damping_gain=float(values["velocity_damping_gain"]),
            frequency_ratio_low=float(values["frequency_ratio_low"]),
            frequency_ratio_peak=float(values["frequency_ratio_peak"]),
            frequency_ratio_high=float(values["frequency_ratio_high"]),
            regularization_velocity_over_wind=float(
                values["regularization_velocity_over_wind"]
            ),
        )
        if float(raw["training"]["maximum_relative_error"]) > 0.01:
            raise ValueError(
                "Griffin training fit does not reproduce its declared rows."
            )
        if raw["claim_boundary"]["liu_5x1_used_for_parameter_fitting"] is not False:
            raise ValueError("Untouched Liu response was used for fitting.")
        return cls(
            raw=dict(raw),
            fingerprint=semantic_sha256(raw),
            parameters=parameters,
        )


__all__ = [
    "CALIBRATION_SCHEMA",
    "MODEL_TYPE",
    "GriffinCalibration",
    "GriffinFit",
    "GriffinLockInParameters",
    "GriffinTrainingPoint",
    "fit_griffin_gains",
    "ideal_cycle_mean_aerodynamic_power_per_span",
    "motion_correlated_force_per_span",
    "predict_stable_amplitude_ratio",
    "semantic_sha256",
]
