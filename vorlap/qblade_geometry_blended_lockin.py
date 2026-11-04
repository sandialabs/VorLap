"""Geometry-blended static-spectrum VIV work closure.

The closure combines two experimentally observed limiting mechanisms without
using phase from a stationary force record:

* bluff rectangular sections use reciprocal Griffin work,
  ``Gc*sign(v)-Gv*v/U``;
* thin airfoil/plate sections use Scanlan-style negative damping with signed
  saturation, ``Ha*(1-(A/(lambda*D))**2)*v/U``.

The interpolation coordinate is the body-fixed streamwise breadth divided by
the cross-flow depth.  It is a declared geometry input, not a fitted study ID.
The stationary spectrum supplies only force magnitude and frequency capture.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .qblade_engineering_lockin import capture_weight

MODEL_TYPE = "static_spectrum_geometry_blended_work_v2"
CALIBRATION_SCHEMA = "vorlap-geometry-blended-viv-calibration-v1"


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
class GeometryBlendedLockInParameters:
    reciprocal_coulomb_gain: float
    reciprocal_velocity_damping_gain: float
    airfoil_velocity_feedback_gain: float
    airfoil_limiting_amplitude_ratio: float
    frequency_ratio_low: float
    frequency_ratio_peak: float
    frequency_ratio_high: float
    airfoil_frequency_scale: float
    airfoil_breadth_depth_ratio: float
    reciprocal_breadth_depth_ratio: float
    regularization_velocity_over_wind: float

    def __post_init__(self) -> None:
        for name in (
            "reciprocal_coulomb_gain",
            "reciprocal_velocity_damping_gain",
            "airfoil_velocity_feedback_gain",
            "airfoil_limiting_amplitude_ratio",
            "frequency_ratio_low",
            "frequency_ratio_peak",
            "frequency_ratio_high",
            "airfoil_frequency_scale",
            "airfoil_breadth_depth_ratio",
            "reciprocal_breadth_depth_ratio",
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
        if not (
            self.airfoil_breadth_depth_ratio
            < self.reciprocal_breadth_depth_ratio
        ):
            raise ValueError("Geometry anchors must be strictly ordered.")
        if not 0.5 < self.airfoil_frequency_scale <= 1.0:
            raise ValueError("Airfoil frequency scale must lie in (0.5, 1].")
        if self.regularization_velocity_over_wind >= 1.0e-2:
            raise ValueError("Origin regularization must remain small.")

    def airfoil_work_fraction(self, breadth_depth_ratio: float) -> float:
        ratio = _finite(breadth_depth_ratio, "breadth_depth_ratio")
        if ratio <= 0.0:
            raise ValueError("breadth_depth_ratio must be positive.")
        return float(
            np.clip(
                (self.reciprocal_breadth_depth_ratio - ratio)
                / (
                    self.reciprocal_breadth_depth_ratio
                    - self.airfoil_breadth_depth_ratio
                ),
                0.0,
                1.0,
            )
        )

    def frequency_scale(self, breadth_depth_ratio: float) -> float:
        fraction = self.airfoil_work_fraction(breadth_depth_ratio)
        return 1.0 - fraction * (1.0 - self.airfoil_frequency_scale)

    def frequency_bounds(
        self, breadth_depth_ratio: float
    ) -> tuple[float, float, float]:
        scale = self.frequency_scale(breadth_depth_ratio)
        return (
            scale * self.frequency_ratio_low,
            scale * self.frequency_ratio_peak,
            scale * self.frequency_ratio_high,
        )

    def capture_weight(
        self, frequency_ratio: Any, breadth_depth_ratio: float
    ) -> np.ndarray:
        low, peak, high = self.frequency_bounds(breadth_depth_ratio)
        return capture_weight(frequency_ratio, low=low, peak=peak, high=high)


def cycle_mean_aerodynamic_power_per_span(
    *,
    parameters: GeometryBlendedLockInParameters,
    peak_velocity_mps: float,
    amplitude_ratio: float,
    wind_speed_mps: float,
    air_density_kgpm3: float,
    crossflow_depth_m: float,
    sinusoidal_equivalent_coefficient: float,
    frequency_ratio: float,
    breadth_depth_ratio: float,
) -> float:
    velocity = _finite(peak_velocity_mps, "peak_velocity_mps")
    amplitude = _finite(amplitude_ratio, "amplitude_ratio")
    speed = _finite(wind_speed_mps, "wind_speed_mps")
    density = _finite(air_density_kgpm3, "air_density_kgpm3")
    depth = _finite(crossflow_depth_m, "crossflow_depth_m")
    coefficient = _finite(
        sinusoidal_equivalent_coefficient,
        "sinusoidal_equivalent_coefficient",
    )
    if velocity < 0.0 or amplitude < 0.0:
        raise ValueError("Peak velocity and amplitude must be non-negative.")
    if min(speed, density, depth) <= 0.0 or coefficient < 0.0:
        raise ValueError("Flow, geometry, and coefficient inputs are invalid.")
    weight = float(parameters.capture_weight([frequency_ratio], breadth_depth_ratio)[0])
    force_scale = 0.5 * density * speed**2 * depth * coefficient * weight
    airfoil = parameters.airfoil_work_fraction(breadth_depth_ratio)
    reciprocal = 1.0 - airfoil
    return force_scale * (
        reciprocal
        * (
            2.0 * parameters.reciprocal_coulomb_gain * velocity / math.pi
            - parameters.reciprocal_velocity_damping_gain
            * velocity**2
            / (2.0 * speed)
        )
        + airfoil
        * parameters.airfoil_velocity_feedback_gain
        * (1.0 - (amplitude / parameters.airfoil_limiting_amplitude_ratio) ** 2)
        * velocity**2
        / (2.0 * speed)
    )


def motion_correlated_force_per_span(
    *,
    parameters: GeometryBlendedLockInParameters,
    modal_velocity_mps: float,
    amplitude_ratio: float,
    wind_speed_mps: float,
    air_density_kgpm3: float,
    crossflow_depth_m: float,
    sinusoidal_equivalent_coefficient: float,
    frequency_ratio: float,
    breadth_depth_ratio: float,
    regularized: bool = True,
) -> float:
    velocity = _finite(modal_velocity_mps, "modal_velocity_mps")
    amplitude = _finite(amplitude_ratio, "amplitude_ratio")
    speed = _finite(wind_speed_mps, "wind_speed_mps")
    density = _finite(air_density_kgpm3, "air_density_kgpm3")
    depth = _finite(crossflow_depth_m, "crossflow_depth_m")
    coefficient = _finite(
        sinusoidal_equivalent_coefficient,
        "sinusoidal_equivalent_coefficient",
    )
    if amplitude < 0.0 or min(speed, density, depth) <= 0.0 or coefficient < 0.0:
        raise ValueError("Closure inputs are invalid.")
    weight = float(parameters.capture_weight([frequency_ratio], breadth_depth_ratio)[0])
    if weight == 0.0 or coefficient == 0.0 or velocity == 0.0:
        return 0.0
    if regularized:
        epsilon = parameters.regularization_velocity_over_wind * speed
        direction = velocity / math.sqrt(velocity**2 + epsilon**2)
    else:
        direction = math.copysign(1.0, velocity)
    airfoil = parameters.airfoil_work_fraction(breadth_depth_ratio)
    reciprocal = 1.0 - airfoil
    force_scale = 0.5 * density * speed**2 * depth * coefficient * weight
    return force_scale * (
        reciprocal
        * (
            parameters.reciprocal_coulomb_gain * direction
            - parameters.reciprocal_velocity_damping_gain * velocity / speed
        )
        + airfoil
        * parameters.airfoil_velocity_feedback_gain
        * (1.0 - (amplitude / parameters.airfoil_limiting_amplitude_ratio) ** 2)
        * velocity
        / speed
    )


def predict_stable_amplitude_ratio(
    *,
    parameters: GeometryBlendedLockInParameters,
    natural_frequency_hz: float,
    mass_per_span_kgpm: float,
    structural_damping_ratio: float,
    air_density_kgpm3: float,
    crossflow_depth_m: float,
    wind_speed_mps: float,
    sinusoidal_equivalent_coefficient: float,
    frequency_ratio: float,
    breadth_depth_ratio: float,
) -> float:
    frequency = _finite(natural_frequency_hz, "natural_frequency_hz")
    mass = _finite(mass_per_span_kgpm, "mass_per_span_kgpm")
    damping = _finite(structural_damping_ratio, "structural_damping_ratio")
    speed = _finite(wind_speed_mps, "wind_speed_mps")
    depth = _finite(crossflow_depth_m, "crossflow_depth_m")
    density = _finite(air_density_kgpm3, "air_density_kgpm3")
    coefficient = _finite(
        sinusoidal_equivalent_coefficient,
        "sinusoidal_equivalent_coefficient",
    )
    if min(frequency, mass, damping, speed, depth, density) <= 0.0 or coefficient < 0.0:
        raise ValueError("Structure, flow, and geometry inputs are invalid.")
    weight = float(parameters.capture_weight([frequency_ratio], breadth_depth_ratio)[0])
    if weight == 0.0 or coefficient == 0.0:
        return 0.0
    omega = 2.0 * math.pi * frequency
    structural_damping = 2.0 * damping * mass * omega
    force_scale = 0.5 * density * speed**2 * depth * coefficient * weight

    def balance(amplitude: float) -> float:
        velocity = omega * depth * amplitude
        return cycle_mean_aerodynamic_power_per_span(
            parameters=parameters,
            peak_velocity_mps=velocity,
            amplitude_ratio=amplitude,
            wind_speed_mps=speed,
            air_density_kgpm3=density,
            crossflow_depth_m=depth,
            sinusoidal_equivalent_coefficient=coefficient,
            frequency_ratio=frequency_ratio,
            breadth_depth_ratio=breadth_depth_ratio,
        ) - 0.5 * structural_damping * velocity**2

    airfoil = parameters.airfoil_work_fraction(breadth_depth_ratio)
    if airfoil == 1.0:
        zero_amplitude_feedback = (
            force_scale * parameters.airfoil_velocity_feedback_gain / speed
        )
        if zero_amplitude_feedback <= structural_damping:
            return 0.0
        return parameters.airfoil_limiting_amplitude_ratio * math.sqrt(
            1.0 - structural_damping / zero_amplitude_feedback
        )
    lower = 1.0e-14
    if balance(lower) <= 0.0:
        return 0.0
    upper = max(0.1, 2.0 * parameters.airfoil_limiting_amplitude_ratio)
    while balance(upper) > 0.0 and upper < 10.0:
        upper *= 2.0
    if balance(upper) > 0.0:
        raise ValueError("Closure has no stable positive-work crossing.")
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if balance(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


@dataclass(frozen=True)
class GeometryBlendedCalibration:
    raw: Mapping[str, Any]
    fingerprint: str
    parameters: GeometryBlendedLockInParameters

    @classmethod
    def from_json(cls, path: Path) -> "GeometryBlendedCalibration":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {
            "schema",
            "status",
            "model_type",
            "method",
            "parameters",
            "training",
            "protected_validation",
            "sources",
            "applicability",
            "claim_boundary",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError("Geometry-blended calibration keys differ from schema.")
        if (
            raw["schema"] != CALIBRATION_SCHEMA
            or raw["status"] != "trained_ready_for_protected_validation"
            or raw["model_type"] != MODEL_TYPE
        ):
            raise ValueError("Unsupported geometry-blended calibration status.")
        p = raw["parameters"]
        parameters = GeometryBlendedLockInParameters(
            reciprocal_coulomb_gain=p["reciprocal_coulomb_gain"],
            reciprocal_velocity_damping_gain=p[
                "reciprocal_velocity_damping_gain"
            ],
            airfoil_velocity_feedback_gain=p["airfoil_velocity_feedback_gain"],
            airfoil_limiting_amplitude_ratio=p[
                "airfoil_limiting_amplitude_ratio"
            ],
            frequency_ratio_low=p["frequency_ratio_low"],
            frequency_ratio_peak=p["frequency_ratio_peak"],
            frequency_ratio_high=p["frequency_ratio_high"],
            airfoil_frequency_scale=p["airfoil_frequency_scale"],
            airfoil_breadth_depth_ratio=p["airfoil_breadth_depth_ratio"],
            reciprocal_breadth_depth_ratio=p[
                "reciprocal_breadth_depth_ratio"
            ],
            regularization_velocity_over_wind=p[
                "regularization_velocity_over_wind"
            ],
        )
        if raw["protected_validation"]["liu_5x1_used_for_fitting"] is not False:
            raise ValueError("Protected Liu response was used for fitting.")
        training = raw["training"]
        if (
            training["maximum_reciprocal_peak_relative_error"] > 0.30
            or training["maximum_airfoil_peak_relative_error"] > 0.30
            or training["airfoil_peak_location_relative_error"] > 0.10
        ):
            raise ValueError("Training closure exceeds its declared gates.")
        return cls(raw=dict(raw), fingerprint=semantic_sha256(raw), parameters=parameters)


__all__ = [
    "CALIBRATION_SCHEMA",
    "MODEL_TYPE",
    "GeometryBlendedCalibration",
    "GeometryBlendedLockInParameters",
    "cycle_mean_aerodynamic_power_per_span",
    "motion_correlated_force_per_span",
    "predict_stable_amplitude_ratio",
    "semantic_sha256",
]
