"""Engineering spectral negative-damping closure for bluff-body lock-in.

The fixed-body spectrum supplies shedding frequency and available fluctuating
lift.  A two-parameter Scanlan--Vickery feedback law stretches that energy
into a velocity-correlated modal force.  The law is deliberately small: a
zero-amplitude feedback coefficient and a limiting amplitude.  Structural
mass, stiffness, and damping determine whether lock-in exists and its stable
equilibrium.

Positive ``aerodynamic_damping_coefficient_kgps`` in this module denotes a
force *with* modal velocity and therefore negative aerodynamic damping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MODEL_TYPE = "single_mode_airfoil_engineering_lockin_v1"
CALIBRATION_SCHEMA = "vorlap-ehrmann2014-engineering-lockin-calibration-v1"


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


def capture_weight(
    frequency_ratio: Any,
    *,
    low: float,
    peak: float,
    high: float,
) -> np.ndarray:
    """Smooth compact lock-in window evaluated for every spectral harmonic.

    The sine-squared/cosine-squared ramps are continuous with zero slope at
    the two edges and unit value at the structural frequency.
    """

    lo = _finite(low, "low")
    pk = _finite(peak, "peak")
    hi = _finite(high, "high")
    if not 0.0 < lo < pk < hi:
        raise ValueError("capture bounds must satisfy 0 < low < peak < high.")
    ratio = np.asarray(frequency_ratio, dtype=float)
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ValueError("frequency_ratio must be finite and non-negative.")
    weight = np.zeros_like(ratio)
    rising = (ratio > lo) & (ratio <= pk)
    falling = (ratio > pk) & (ratio < hi)
    weight[rising] = np.sin(0.5 * math.pi * (ratio[rising] - lo) / (pk - lo)) ** 2
    weight[falling] = np.cos(0.5 * math.pi * (ratio[falling] - pk) / (hi - pk)) ** 2
    weight[ratio == pk] = 1.0
    return weight


@dataclass(frozen=True)
class CapturedSpectrum:
    coefficient_rss: float
    centroid_frequency_ratio: float
    dominant_frequency_ratio: float
    active_harmonics: int
    total_harmonic_energy: float
    captured_harmonic_energy: float


def captured_spectral_coefficient(
    strouhal: Sequence[float],
    harmonic_amplitude: Sequence[float],
    *,
    effective_speed_mps: float,
    strouhal_reference_length_m: float,
    natural_frequency_hz: float,
    low: float,
    peak: float,
    high: float,
) -> CapturedSpectrum:
    """Return the RMS-like coefficient captured from the full static spectrum."""

    st = np.asarray(strouhal, dtype=float)
    amplitude = np.asarray(harmonic_amplitude, dtype=float)
    if st.ndim != 1 or amplitude.shape != st.shape:
        raise ValueError("strouhal and harmonic_amplitude must be equal 1-D arrays.")
    if np.any(~np.isfinite(st)) or np.any(st < 0.0):
        raise ValueError("strouhal must be finite and non-negative.")
    if np.any(~np.isfinite(amplitude)):
        raise ValueError("harmonic_amplitude must be finite.")
    # Index zero stores the signed mean coefficient in the legacy HDF5
    # contract.  Only positive-frequency entries are harmonic amplitudes.
    if np.any(amplitude[st > 0.0] < 0.0):
        raise ValueError("Positive-frequency harmonic amplitudes must be non-negative.")
    speed = _finite(effective_speed_mps, "effective_speed_mps")
    width = _finite(strouhal_reference_length_m, "strouhal_reference_length_m")
    frequency = _finite(natural_frequency_hz, "natural_frequency_hz")
    if speed <= 0.0 or width <= 0.0 or frequency <= 0.0:
        raise ValueError("speed, reference length, and frequency must be positive.")
    active = (st > 0.0) & (amplitude > 0.0)
    ratio = st * speed / (width * frequency)
    weights = capture_weight(ratio, low=low, peak=peak, high=high)
    total_energy = float(np.sum(amplitude[active] ** 2))
    captured_energy_by_harmonic = (weights * amplitude) ** 2
    captured_energy = float(np.sum(captured_energy_by_harmonic[active]))
    if captured_energy > 0.0:
        centroid = float(np.sum(ratio * captured_energy_by_harmonic) / captured_energy)
        dominant = float(ratio[int(np.argmax(captured_energy_by_harmonic))])
    else:
        centroid = 0.0
        dominant = 0.0
    return CapturedSpectrum(
        coefficient_rss=math.sqrt(max(0.0, captured_energy)),
        centroid_frequency_ratio=centroid,
        dominant_frequency_ratio=dominant,
        active_harmonics=int(np.count_nonzero(active & (weights > 0.0))),
        total_harmonic_energy=total_energy,
        captured_harmonic_energy=captured_energy,
    )


@dataclass(frozen=True)
class EndpointFit:
    limiting_amplitude_ratio: float
    critical_mass_damping: float
    intercept_amplitude_ratio_squared: float
    slope_per_mass_damping: float

    def predict_amplitude_ratio(self, mass_damping: float) -> float:
        md = _finite(mass_damping, "mass_damping")
        value = (
            self.intercept_amplitude_ratio_squared + self.slope_per_mass_damping * md
        )
        return math.sqrt(max(0.0, value))


def fit_endpoint_saturation(
    first_mass_damping: float,
    first_amplitude_ratio: float,
    last_mass_damping: float,
    last_amplitude_ratio: float,
) -> EndpointFit:
    """Fit ``A^2=lambda^2*(1-md/mdcrit)`` to exactly two endpoints."""

    x0 = _finite(first_mass_damping, "first_mass_damping")
    x1 = _finite(last_mass_damping, "last_mass_damping")
    y0 = _finite(first_amplitude_ratio, "first_amplitude_ratio")
    y1 = _finite(last_amplitude_ratio, "last_amplitude_ratio")
    if not 0.0 < x0 < x1 or y0 <= 0.0 or y1 <= 0.0 or y1 >= y0:
        raise ValueError(
            "Endpoint data must have increasing damping and decreasing amplitude."
        )
    slope = (y1**2 - y0**2) / (x1 - x0)
    intercept = y0**2 - slope * x0
    if slope >= 0.0 or intercept <= 0.0:
        raise ValueError(
            "Endpoint fit does not define a saturating negative-damping law."
        )
    critical = -intercept / slope
    if critical <= x1:
        raise ValueError(
            "Critical mass damping must exceed the highest calibration point."
        )
    return EndpointFit(
        limiting_amplitude_ratio=math.sqrt(intercept),
        critical_mass_damping=critical,
        intercept_amplitude_ratio_squared=intercept,
        slope_per_mass_damping=slope,
    )


def feedback_gain_from_threshold(
    *,
    critical_mass_damping: float,
    captured_lift_coefficient: float,
    effective_reduced_velocity: float,
) -> float:
    """Return dimensionless wake-work stretch ``H`` from the onset threshold."""

    critical = _finite(critical_mass_damping, "critical_mass_damping")
    coefficient = _finite(captured_lift_coefficient, "captured_lift_coefficient")
    reduced_velocity = _finite(effective_reduced_velocity, "effective_reduced_velocity")
    if critical <= 0.0 or coefficient <= 0.0 or reduced_velocity <= 0.0:
        raise ValueError(
            "Threshold, coefficient, and reduced velocity must be positive."
        )
    return 2.0 * math.pi**2 * critical / (coefficient * reduced_velocity)


def critical_mass_damping(
    *,
    feedback_gain: float,
    captured_lift_coefficient: float,
    effective_reduced_velocity: float,
) -> float:
    gain = _finite(feedback_gain, "feedback_gain")
    coefficient = _finite(captured_lift_coefficient, "captured_lift_coefficient")
    reduced_velocity = _finite(effective_reduced_velocity, "effective_reduced_velocity")
    if gain < 0.0 or coefficient < 0.0 or reduced_velocity < 0.0:
        raise ValueError(
            "Gain, coefficient, and reduced velocity must be non-negative."
        )
    return gain * coefficient * reduced_velocity / (2.0 * math.pi**2)


@dataclass(frozen=True)
class EngineeringLockInCalibration:
    raw: Mapping[str, Any]
    fingerprint: str

    @classmethod
    def from_json(cls, path: Path) -> "EngineeringLockInCalibration":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("Engineering lock-in calibration must be a JSON object.")
        required = {
            "schema",
            "status",
            "model_type",
            "profile",
            "sources",
            "facility",
            "spectral_capture",
            "amplitude_calibration",
            "holdout_assessment",
            "claim_boundary",
        }
        if set(raw) != required:
            raise ValueError(
                "Engineering lock-in calibration keys differ from the schema."
            )
        if (
            raw["schema"] != CALIBRATION_SCHEMA
            or raw["status"] != "calibrated_holdout_passed"
            or raw["model_type"] != MODEL_TYPE
        ):
            raise ValueError(
                "Unsupported or unqualified engineering lock-in calibration."
            )
        spectral = raw["spectral_capture"]
        amplitude = raw["amplitude_calibration"]
        if not isinstance(spectral, Mapping) or not isinstance(amplitude, Mapping):
            raise TypeError("Calibration spectral/amplitude blocks must be mappings.")
        fit = fit_endpoint_saturation(
            amplitude["calibration_points"][0]["mass_damping"],
            amplitude["calibration_points"][0]["amplitude_over_chord"],
            amplitude["calibration_points"][1]["mass_damping"],
            amplitude["calibration_points"][1]["amplitude_over_chord"],
        )
        for key, expected in {
            "limiting_amplitude_ratio": fit.limiting_amplitude_ratio,
            "critical_mass_damping": fit.critical_mass_damping,
            "intercept_amplitude_ratio_squared": fit.intercept_amplitude_ratio_squared,
            "slope_per_mass_damping": fit.slope_per_mass_damping,
        }.items():
            if not math.isclose(
                float(amplitude[key]), expected, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(
                    f"Calibration {key} is inconsistent with endpoint data."
                )
        modal_reference = amplitude.get("reference_modal_structure")
        if not isinstance(modal_reference, Mapping):
            raise ValueError("Calibration lacks its reference modal structure.")
        expected_feedback = float(
            modal_reference["structural_damping_coefficient_at_onset_kgps"]
        ) / float(modal_reference["unit_gain_aerodynamic_feedback_coefficient_kgps"])
        if not math.isclose(
            float(amplitude["feedback_gain"]),
            expected_feedback,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Calibration feedback_gain is inconsistent with modal onset."
            )
        facility = raw["facility"]
        expected_velocity_scale = 1.0 / (
            1.0 - float(facility["reported_tunnel_blockage_fraction"])
        )
        if not math.isclose(
            float(facility["effective_velocity_scale"]),
            expected_velocity_scale,
            rel_tol=1e-15,
            abs_tol=0.0,
        ):
            raise ValueError(
                "Facility effective_velocity_scale is inconsistent with blockage."
            )
        if (
            max(
                float(row["absolute_relative_error"])
                for row in raw["holdout_assessment"]["points"]
            )
            > 0.30
        ):
            raise ValueError("Engineering lock-in holdout exceeds the 30% requirement.")
        return cls(raw=dict(raw), fingerprint=semantic_sha256(raw))

    @property
    def feedback_gain(self) -> float:
        return float(self.raw["amplitude_calibration"]["feedback_gain"])

    @property
    def limiting_amplitude_ratio(self) -> float:
        return float(self.raw["amplitude_calibration"]["limiting_amplitude_ratio"])

    @property
    def velocity_scale(self) -> float:
        return float(self.raw["facility"]["effective_velocity_scale"])

    @property
    def capture_bounds(self) -> tuple[float, float, float]:
        capture = self.raw["spectral_capture"]
        return (
            float(capture["frequency_ratio_low"]),
            float(capture["frequency_ratio_peak"]),
            float(capture["frequency_ratio_high"]),
        )

    @property
    def transferable_parameters(self) -> "EngineeringLockInTransferParameters":
        """Return the facility-independent engineering transfer contract.

        The Ehrmann tunnel blockage correction is required to reproduce the
        calibration experiment, but it is not a material or wake property and
        must not be applied to a turbine's already local relative velocity.
        This deliberately narrow view makes that distinction enforceable: it
        contains the fitted dimensionless work/saturation parameters and
        spectral-source identity, but no facility velocity scale.
        """

        capture = self.raw["spectral_capture"]
        source = self.raw["sources"]["stationary_spectrum"]
        return EngineeringLockInTransferParameters(
            calibration_fingerprint=self.fingerprint,
            profile=str(self.raw["profile"]),
            static_spectrum_sha256=str(source["sha256"]).lower(),
            static_spectrum_reynolds_model=str(source["reynolds_model"]),
            feedback_gain=self.feedback_gain,
            limiting_amplitude_ratio=self.limiting_amplitude_ratio,
            frequency_ratio_low=float(capture["frequency_ratio_low"]),
            frequency_ratio_peak=float(capture["frequency_ratio_peak"]),
            frequency_ratio_high=float(capture["frequency_ratio_high"]),
            capture_window=str(capture["window"]),
        )


@dataclass(frozen=True)
class EngineeringLockInTransferParameters:
    """Facility-independent subset permitted for a new structure/flow."""

    calibration_fingerprint: str
    profile: str
    static_spectrum_sha256: str
    static_spectrum_reynolds_model: str
    feedback_gain: float
    limiting_amplitude_ratio: float
    frequency_ratio_low: float
    frequency_ratio_peak: float
    frequency_ratio_high: float
    capture_window: str

    def __post_init__(self) -> None:
        if self.capture_window != "compact_sine_squared":
            raise ValueError("Unsupported engineering spectral capture window.")
        if self.static_spectrum_reynolds_model != "replicated_reynolds_independent":
            raise ValueError("Unsupported engineering static-spectrum Reynolds model.")
        if not self.profile:
            raise ValueError("Engineering transfer profile must be nonempty.")
        for name in ("calibration_fingerprint", "static_spectrum_sha256"):
            value = str(getattr(self, name)).lower()
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
        if self.feedback_gain <= 0.0 or self.limiting_amplitude_ratio <= 0.0:
            raise ValueError("Engineering transfer gain and limiting amplitude must be positive.")
        if not (
            0.0
            < self.frequency_ratio_low
            < self.frequency_ratio_peak
            < self.frequency_ratio_high
        ):
            raise ValueError("Engineering transfer capture bounds are invalid.")


__all__ = [
    "CALIBRATION_SCHEMA",
    "MODEL_TYPE",
    "CapturedSpectrum",
    "EndpointFit",
    "EngineeringLockInCalibration",
    "EngineeringLockInTransferParameters",
    "capture_weight",
    "captured_spectral_coefficient",
    "critical_mass_damping",
    "feedback_gain_from_threshold",
    "fit_endpoint_saturation",
    "semantic_sha256",
]
