#!/usr/bin/env python3
"""Freeze the two-parameter engineering lock-in calibration.

Only the two outer Figure 7 markers identify the limiting amplitude and onset
threshold.  The three interior markers are evaluated as held-out damping
cases.  The static NACA0018 spectrum and the reported 25% tunnel blockage set
the frequency-dependent feedback scale; Figure 6 is used only after freezing
to check the predicted lock-in location.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vorlap.airfoil_io import load_airfoil_fft
from vorlap.force_model import strouhal_reference_length
from vorlap.interpolation import interpolate_fft_spectrum
from vorlap.qblade_engineering_lockin import (
    CALIBRATION_SCHEMA,
    MODEL_TYPE,
    captured_spectral_coefficient,
    feedback_gain_from_threshold,
    fit_endpoint_saturation,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIGURE6 = HERE / "reference_data" / "ehrmann2014_figure6_naca0018_digitized.csv"
FIGURE7 = HERE / "reference_data" / "ehrmann2014_figure7_naca0018_peak_digitized.csv"
PROVENANCE = HERE / "reference_data" / "ehrmann2014_digitization_provenance.json"
STATIC_SPECTRUM = (
    REPO_ROOT
    / "examples"
    / "ReferenceHVAWTQBladeModel"
    / "10m_HVAWT_reference_turbine"
    / "VorLapAirfoils"
    / "NACA0018.h5"
)
STATIC_SPECTRUM_SHA256 = (
    "64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369"
)

CHORD_M = 0.230
NATURAL_FREQUENCY_HZ = 6.0
MASS_RATIO = 47.7
KINEMATIC_VISCOSITY_M2PS = 1.48e-5
AIR_DENSITY_KGPM3 = 1.225
SPAN_M = 0.9144
N_STATIONS = 21
EULER_BERNOULLI_BETA1 = 1.875104068711961
REFERENCE_MASS_DAMPING = 0.96
BLOCKAGE_FRACTION = 0.25
EFFECTIVE_VELOCITY_SCALE = 1.0 / (1.0 - BLOCKAGE_FRACTION)
CAPTURE_LOW = 0.85
CAPTURE_PEAK = 1.0
CAPTURE_HIGH = 1.10
SCAN_MIN_USTAR = 3.0
SCAN_MAX_USTAR = 6.3
SCAN_POINTS = 3301


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                key: (value if key in {"source_figure", "series"} else float(value))
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def build_calibration() -> dict[str, Any]:
    if _sha256(STATIC_SPECTRUM) != STATIC_SPECTRUM_SHA256:
        raise ValueError("Tracked NACA0018 spectrum SHA-256 differs.")
    figure6 = _csv(FIGURE6)
    figure7 = _csv(FIGURE7)
    if len(figure6) != 26 or len(figure7) != 5:
        raise ValueError("Unexpected Ehrmann digitization row count.")
    fit = fit_endpoint_saturation(
        figure7[0]["mass_damping"],
        figure7[0]["maximum_amplitude_over_chord"],
        figure7[-1]["mass_damping"],
        figure7[-1]["maximum_amplitude_over_chord"],
    )
    holdout = []
    for row in figure7[1:-1]:
        prediction = fit.predict_amplitude_ratio(row["mass_damping"])
        experiment = float(row["maximum_amplitude_over_chord"])
        holdout.append(
            {
                "mass_damping": float(row["mass_damping"]),
                "experimental_amplitude_over_chord": experiment,
                "predicted_amplitude_over_chord": prediction,
                "signed_relative_error": (prediction - experiment) / experiment,
                "absolute_relative_error": abs(prediction - experiment) / experiment,
                "amplitude_digitization_uncertainty": float(
                    row["amplitude_over_chord_digitization_uncertainty"]
                ),
            }
        )

    airfoil = load_airfoil_fft(str(STATIC_SPECTRUM))
    damping = REFERENCE_MASS_DAMPING / MASS_RATIO
    damped_frequency = NATURAL_FREQUENCY_HZ * math.sqrt(1.0 - damping**2)
    scan = []
    for reduced_velocity in np.linspace(SCAN_MIN_USTAR, SCAN_MAX_USTAR, SCAN_POINTS):
        nominal_speed = reduced_velocity * damped_frequency * CHORD_M
        reynolds = nominal_speed * CHORD_M / KINEMATIC_VISCOSITY_M2PS
        st, amplitude, _ = interpolate_fft_spectrum(
            airfoil,
            reynolds,
            -90.0,
            "CL",
            n_freq_depth=200,
            bounds_policy="raise",
        )
        width = float(
            strouhal_reference_length(
                np.asarray([CHORD_M]),
                np.asarray([-90.0]),
                np.asarray([0.18]),
            )[0]
        )
        captured = captured_spectral_coefficient(
            st,
            amplitude,
            effective_speed_mps=EFFECTIVE_VELOCITY_SCALE * nominal_speed,
            strouhal_reference_length_m=width,
            natural_frequency_hz=NATURAL_FREQUENCY_HZ,
            low=CAPTURE_LOW,
            peak=CAPTURE_PEAK,
            high=CAPTURE_HIGH,
        )
        effective_ustar = EFFECTIVE_VELOCITY_SCALE * reduced_velocity
        scan.append(
            {
                "nominal_reduced_velocity": float(reduced_velocity),
                "effective_reduced_velocity": float(effective_ustar),
                "captured_lift_coefficient": captured.coefficient_rss,
                "threshold_scale": captured.coefficient_rss * effective_ustar,
                "centroid_frequency_ratio": captured.centroid_frequency_ratio,
                "dominant_frequency_ratio": captured.dominant_frequency_ratio,
            }
        )
    reference = max(scan, key=lambda row: row["threshold_scale"])
    # The executed Ehrmann surrogate uses a tip-normalized cantilever basis,
    # not rigid translation.  Identify H against that exact modal mass and
    # aerodynamic quadrature so the dimensional negative-damping threshold is
    # preserved rather than silently assuming phi=1 along the span.
    fractions = np.linspace(0.0, 1.0, N_STATIONS)
    beta = EULER_BERNOULLI_BETA1
    sigma = (math.cosh(beta) + math.cos(beta)) / (math.sinh(beta) + math.sin(beta))
    mode = (
        np.cosh(beta * fractions)
        - np.cos(beta * fractions)
        - sigma * (np.sinh(beta * fractions) - np.sin(beta * fractions))
    )
    mode /= mode[-1]
    tributary = np.full(N_STATIONS, SPAN_M / (N_STATIONS - 1))
    tributary[[0, -1]] *= 0.5
    published_mass_per_span = (
        MASS_RATIO * math.pi * AIR_DENSITY_KGPM3 * CHORD_M**2 / 4.0
    )
    generalized_mass = published_mass_per_span * SPAN_M
    critical_damping_ratio = fit.critical_mass_damping / MASS_RATIO
    structural_damping_at_onset = (
        2.0
        * critical_damping_ratio
        * generalized_mass
        * 2.0
        * math.pi
        * NATURAL_FREQUENCY_HZ
    )
    nominal_reference_speed = (
        reference["nominal_reduced_velocity"] * damped_frequency * CHORD_M
    )
    unit_gain_feedback = float(
        np.sum(
            0.5
            * AIR_DENSITY_KGPM3
            * EFFECTIVE_VELOCITY_SCALE
            * nominal_reference_speed
            * CHORD_M
            * tributary
            * reference["captured_lift_coefficient"]
            * mode**2
        )
    )
    feedback_gain = structural_damping_at_onset / unit_gain_feedback
    figure6_peak = max(figure6, key=lambda row: row["rms_acceleration_mps2"])
    peak_error = (
        reference["nominal_reduced_velocity"] - figure6_peak["reduced_velocity"]
    ) / figure6_peak["reduced_velocity"]

    return {
        "schema": CALIBRATION_SCHEMA,
        "status": "calibrated_holdout_passed",
        "model_type": MODEL_TYPE,
        "profile": "NACA0018",
        "sources": {
            "stationary_spectrum": {
                "path": STATIC_SPECTRUM.relative_to(REPO_ROOT).as_posix(),
                "sha256": STATIC_SPECTRUM_SHA256,
                "reynolds_model": "replicated_reynolds_independent",
            },
            "figure6_digitization": {
                "path": FIGURE6.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(FIGURE6),
                "role": "unused_lockin_location_check",
            },
            "figure7_digitization": {
                "path": FIGURE7.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(FIGURE7),
                "role": "two_endpoint_calibration_three_point_holdout",
            },
            "digitization_provenance": {
                "path": PROVENANCE.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(PROVENANCE),
            },
        },
        "facility": {
            "reported_tunnel_blockage_fraction": BLOCKAGE_FRACTION,
            "effective_velocity_scale": EFFECTIVE_VELOCITY_SCALE,
            "policy": "continuity_area_ratio_correction_applied_to_shedding_frequency_and_feedback_dynamic_scale",
        },
        "spectral_capture": {
            "channel": "CL",
            "harmonics": "all_positive_frequency_entries",
            "window": "compact_sine_squared",
            "frequency_ratio_definition": "St*U_effective/(D_St*f_n)",
            "frequency_ratio_low": CAPTURE_LOW,
            "frequency_ratio_peak": CAPTURE_PEAK,
            "frequency_ratio_high": CAPTURE_HIGH,
            "scan_nominal_reduced_velocity": [SCAN_MIN_USTAR, SCAN_MAX_USTAR],
            "scan_points": SCAN_POINTS,
            "reference_peak": reference,
            "figure6_unused_check": {
                "experimental_peak_reduced_velocity": float(
                    figure6_peak["reduced_velocity"]
                ),
                "predicted_peak_reduced_velocity": reference[
                    "nominal_reduced_velocity"
                ],
                "signed_relative_error": peak_error,
                "absolute_relative_error": abs(peak_error),
                "passes_10_percent": abs(peak_error) <= 0.10,
            },
        },
        "amplitude_calibration": {
            "law": "A_over_c_squared=lambda_squared*(1-mass_damping/critical_mass_damping)",
            "calibration_points": [
                {
                    "mass_damping": float(figure7[0]["mass_damping"]),
                    "amplitude_over_chord": float(
                        figure7[0]["maximum_amplitude_over_chord"]
                    ),
                },
                {
                    "mass_damping": float(figure7[-1]["mass_damping"]),
                    "amplitude_over_chord": float(
                        figure7[-1]["maximum_amplitude_over_chord"]
                    ),
                },
            ],
            "limiting_amplitude_ratio": fit.limiting_amplitude_ratio,
            "critical_mass_damping": fit.critical_mass_damping,
            "intercept_amplitude_ratio_squared": fit.intercept_amplitude_ratio_squared,
            "slope_per_mass_damping": fit.slope_per_mass_damping,
            "feedback_gain": feedback_gain,
            "feedback_gain_definition": "H=C_structural_at_critical_mass_damping/sum(0.5*rho*U_effective*c*ds*Ceff*phi_squared)",
            "reference_modal_structure": {
                "basis": "tip_normalized_euler_bernoulli_first_cantilever_mode",
                "station_count": N_STATIONS,
                "span_m": SPAN_M,
                "mass_ratio": MASS_RATIO,
                "generalized_modal_mass_kg": generalized_mass,
                "structural_damping_coefficient_at_onset_kgps": structural_damping_at_onset,
                "unit_gain_aerodynamic_feedback_coefficient_kgps": unit_gain_feedback,
                "mode_shape_sha256": hashlib.sha256(
                    np.ascontiguousarray(mode, dtype="<f8").tobytes()
                ).hexdigest(),
                "tributary_length_sha256": hashlib.sha256(
                    np.ascontiguousarray(tributary, dtype="<f8").tobytes()
                ).hexdigest(),
            },
        },
        "holdout_assessment": {
            "status": "passed",
            "maximum_allowed_absolute_relative_error": 0.30,
            "maximum_observed_absolute_relative_error": max(
                row["absolute_relative_error"] for row in holdout
            ),
            "points": holdout,
        },
        "claim_boundary": {
            "level": "engineering_calibration_and_internal_holdout",
            "not_independent_validation": True,
            "nalu_required": False,
            "prescribed_motion_required": False,
            "reynolds_transfer_unqualified_because_source_planes_are_replicated": True,
            "next_gate": "coupled_raw_qblade_vs_engineering_bridge_ehrmann_rerun",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "engineering_lockin_calibration.json",
    )
    args = parser.parse_args()
    payload = build_calibration()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    print(json.dumps(payload["holdout_assessment"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
