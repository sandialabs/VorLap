#!/usr/bin/env python3
"""Fit a spectrum-scaled saturation to 3:2 data and test 4:1 data untouched."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vorlap.qblade_engineering_lockin import EngineeringLockInCalibration


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CALIBRATION = REPO / "examples/ehrmann2014_naca0018/engineering_lockin_calibration.json"
CALIBRATION_SHA256 = "d96c0b73a708b918360a15e79224f6a3a3f4e80702edf721bc606501a46fa02d"
TRAINING_3X2 = REPO / "examples/mannini2016_rectangular_3x2/reference_data/digitization_provenance.json"
DATA_4X1 = HERE / "reference_data/gao2025_table5_training.csv"
PROVENANCE_4X1 = HERE / "reference_data/training_provenance.json"
SCHEMA = "vorlap-spectrum-scaled-saturation-transfer-screen-v1"
MAX_RELATIVE_ERROR = 0.30


class AssessmentError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows_4x1() -> list[dict[str, float]]:
    rows = list(csv.DictReader(DATA_4X1.open(newline="", encoding="utf-8")))
    if len(rows) != 3 or {row["evidence_role"] for row in rows} != {"training"}:
        raise AssessmentError("Expected exactly three declared 4:1 training rows.")
    keys = (
        "scruton_number",
        "natural_frequency_hz",
        "mass_per_span_kgpm",
        "structural_damping_ratio",
        "air_density_kgpm3",
        "reduced_frequency_omega_d_over_u",
        "peak_amplitude_over_depth",
    )
    return [{key: float(row[key]) for key in keys} for row in rows]


def assess() -> dict[str, Any]:
    if _sha256(CALIBRATION) != CALIBRATION_SHA256:
        raise AssessmentError("Frozen Ehrmann calibration bytes differ.")
    calibration = EngineeringLockInCalibration.from_json(CALIBRATION)
    source_3x2 = json.loads(TRAINING_3X2.read_text(encoding="utf-8"))
    source_4x1 = json.loads(PROVENANCE_4X1.read_text(encoding="utf-8"))
    if (
        source_3x2["source"]["static_body_input_at_re_58200"]
        ["sinusoidal_equivalent_lift_coefficient"]
        != 0.656
        or source_4x1["status"]
        != "frozen_as_training_after_prior_model_failure"
    ):
        raise AssessmentError("Training roles or static inputs differ.")

    # Revision hypothesis: retain the v1 signed Rayleigh saturation, but allow
    # its limiting displacement to scale with the fixed-body fluctuating-force
    # coefficient.  The one exponent is fitted only to the failed 3:2 case.
    mannini = {
        "depth_m": 0.077,
        "span_m": 0.986,
        "natural_frequency_hz": 9.01,
        "mass_kg": 5.526,
        "structural_damping_ratio": 0.0135,
        "air_density_kgpm3": 1.20,
        "strouhal_number": 0.106,
        "frequency_ratio_at_peak": 1.166666667,
        "coefficient": 0.656,
        "target_amplitude_ratio": 0.04921875,
    }
    omega = 2.0 * math.pi * mannini["natural_frequency_hz"]
    speed = (
        mannini["frequency_ratio_at_peak"]
        * mannini["natural_frequency_hz"]
        * mannini["depth_m"]
        / mannini["strouhal_number"]
    )
    training_structural = (
        2.0
        * mannini["structural_damping_ratio"]
        * mannini["mass_kg"]
        * omega
    )
    training_aerodynamic = (
        0.5
        * mannini["air_density_kgpm3"]
        * speed
        * mannini["depth_m"]
        * mannini["span_m"]
        * calibration.feedback_gain
        * mannini["coefficient"]
    )
    if training_aerodynamic <= training_structural:
        raise AssessmentError("3:2 training case has no negative-damping branch.")
    required_limit = mannini["target_amplitude_ratio"] / math.sqrt(
        1.0 - training_structural / training_aerodynamic
    )
    reference_coefficient = float(
        calibration.raw["spectral_capture"]["reference_peak"]
        ["captured_lift_coefficient"]
    )
    exponent = math.log(required_limit / calibration.limiting_amplitude_ratio) / math.log(
        mannini["coefficient"] / reference_coefficient
    )

    rows = []
    coefficient_4x1 = float(
        source_4x1["static_source"]["sinusoidal_equivalent_lift_coefficient"]
    )
    depth = float(source_4x1["experiment"]["crossflow_depth_m"])
    span = float(source_4x1["experiment"]["span_m"])
    effective_limit = calibration.limiting_amplitude_ratio * (
        coefficient_4x1 / reference_coefficient
    ) ** exponent
    for row in _rows_4x1():
        omega = 2.0 * math.pi * row["natural_frequency_hz"]
        speed = omega * depth / row["reduced_frequency_omega_d_over_u"]
        aerodynamic = (
            0.5
            * row["air_density_kgpm3"]
            * speed
            * depth
            * span
            * calibration.feedback_gain
            * coefficient_4x1
        )
        structural = (
            2.0
            * row["structural_damping_ratio"]
            * row["mass_per_span_kgpm"]
            * span
            * omega
        )
        prediction = effective_limit * math.sqrt(
            max(0.0, 1.0 - structural / aerodynamic)
        )
        target = row["peak_amplitude_over_depth"]
        error = abs(prediction - target) / target
        rows.append(
            {
                **row,
                "wind_speed_mps": speed,
                "predicted_peak_amplitude_over_depth": prediction,
                "absolute_relative_error": error,
                "passes_30_percent": error <= MAX_RELATIVE_ERROR,
            }
        )
    passed = all(row["passes_30_percent"] for row in rows)
    return {
        "schema": SCHEMA,
        "status": "passed_untouched_4x1" if passed else "failed_untouched_4x1",
        "pass": passed,
        "revision": {
            "name": "fixed_body_force_scaled_rayleigh_saturation",
            "hypothesis": "lambda_eff=lambda_ref*(Ceff/Cref)^q",
            "single_fitted_parameter": "q",
            "q": exponent,
            "frequency_ratio_at_peak": mannini["frequency_ratio_at_peak"],
            "reference_coefficient": reference_coefficient,
            "fitted_only_to": "Mannini 3:2 Sc=87.7 peak",
            "not_fitted_to_4x1": True,
        },
        "training_3x2": {
            "required_limiting_amplitude_ratio": required_limit,
            "zero_amplitude_aerodynamic_coefficient_kgps": training_aerodynamic,
            "structural_damping_coefficient_kgps": training_structural,
        },
        "untouched_4x1_results": rows,
        "assessment": {
            "maximum_allowed_relative_error": MAX_RELATIVE_ERROR,
            "maximum_observed_relative_error": max(
                row["absolute_relative_error"] for row in rows
            ),
            "failure_interpretation": (
                "A fixed Rayleigh saturation shape remains nearly insensitive to "
                "structural damping and cannot reproduce the reciprocal-amplitude "
                "Griffin trend."
            ),
        },
        "sources": {
            "calibration_sha256": CALIBRATION_SHA256,
            "mannini_training_sha256": _sha256(TRAINING_3X2),
            "marra_table_sha256": _sha256(DATA_4X1),
            "marra_provenance_sha256": _sha256(PROVENANCE_4X1),
        },
        "claim_boundary": {
            "independent_4x1_assessment": True,
            "failure_preserved_before_next_revision": True,
            "4x1_may_now_be_used_for_training": not passed,
            "does_not_validate_hvawt_or_fatigue": True,
        },
    }


def _plot(report: dict[str, Any], path: Path) -> None:
    rows = report["untouched_4x1_results"]
    sc = [row["scruton_number"] for row in rows]
    target = [row["peak_amplitude_over_depth"] for row in rows]
    prediction = [row["predicted_peak_amplitude_over_depth"] for row in rows]
    fig, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    axis.plot(sc, target, "o-", color="#222222", label="Marra experiment")
    axis.plot(sc, prediction, "s--", color="#b21f2d", label="3:2-trained revision")
    axis.set(xlabel="Scruton number", ylabel=r"Peak $A_y/D$", xlim=(0, 82), ylim=(0, 0.07))
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    axis.text(
        0.98,
        0.95,
        f"Maximum error = {100*report['assessment']['maximum_observed_relative_error']:.0f}%\n30% gate: FAIL",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "evidence")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = assess()
    report_path = args.output_dir / "scaled_saturation_screen.json"
    plot_path = args.output_dir / "scaled_saturation_screen.pdf"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _plot(report, plot_path)
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in sorted((plot_path, report_path), key=lambda item: item.name)
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
