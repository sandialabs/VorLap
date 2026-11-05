#!/usr/bin/env python3
"""Fit the Gao--Scanlan reciprocal-Griffin closure to the declared 4:1 rows."""

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

from vorlap.qblade_griffin_lockin import (
    CALIBRATION_SCHEMA,
    MODEL_TYPE,
    GriffinTrainingPoint,
    fit_griffin_gains,
    predict_stable_amplitude_ratio,
    GriffinLockInParameters,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "reference_data/gao2025_table5_training.csv"
PROVENANCE = HERE / "reference_data/training_provenance.json"
MANNINI_DATA = (
    REPO
    / "examples/mannini2016_rectangular_3x2/reference_data/mannini2015_figure4_sc87p7_digitized.csv"
)
LIU_PROVENANCE = (
    REPO / "examples/liu2024_rectangular_5x1/reference_data/validation_provenance.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows() -> list[dict[str, Any]]:
    rows = list(csv.DictReader(DATA.open(newline="", encoding="utf-8")))
    if len(rows) != 3 or {row["evidence_role"] for row in rows} != {"training"}:
        raise ValueError("Expected exactly three 4:1 training rows.")
    return rows


def _training_points() -> list[GriffinTrainingPoint]:
    return [
        GriffinTrainingPoint(
            natural_frequency_hz=float(row["natural_frequency_hz"]),
            mass_per_span_kgpm=float(row["mass_per_span_kgpm"]),
            structural_damping_ratio=float(row["structural_damping_ratio"]),
            air_density_kgpm3=float(row["air_density_kgpm3"]),
            crossflow_depth_m=0.075,
            reduced_frequency_omega_d_over_u=float(
                row["reduced_frequency_omega_d_over_u"]
            ),
            peak_amplitude_over_depth=float(row["peak_amplitude_over_depth"]),
        )
        for row in _rows()
    ]


def build_calibration() -> dict[str, Any]:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    liu = json.loads(LIU_PROVENANCE.read_text(encoding="utf-8"))
    if (
        provenance["status"] != "frozen_as_training_after_prior_model_failure"
        or liu["status"] != "frozen_before_fitting_4x1_training_closure"
    ):
        raise ValueError("Evidence roles were not frozen before calibration.")
    coefficient = float(
        provenance["static_source"]["sinusoidal_equivalent_lift_coefficient"]
    )
    points = _training_points()
    fit = fit_griffin_gains(points, sinusoidal_equivalent_coefficient=coefficient)
    parameters = GriffinLockInParameters(
        coulomb_work_gain=fit.coulomb_work_gain,
        velocity_damping_gain=fit.velocity_damping_gain,
        # These three frequency ratios were taken only from the previously
        # failed 3:2 response: onset, peak, and post-peak collapse.
        frequency_ratio_low=1.0,
        frequency_ratio_peak=1.166666667,
        frequency_ratio_high=1.21875,
        regularization_velocity_over_wind=1.0e-5,
    )
    predictions = []
    for row, point in zip(_rows(), points, strict=True):
        predicted = predict_stable_amplitude_ratio(
            parameters=parameters,
            natural_frequency_hz=point.natural_frequency_hz,
            mass_per_span_kgpm=point.mass_per_span_kgpm,
            structural_damping_ratio=point.structural_damping_ratio,
            air_density_kgpm3=point.air_density_kgpm3,
            crossflow_depth_m=point.crossflow_depth_m,
            wind_speed_mps=point.wind_speed_mps,
            sinusoidal_equivalent_coefficient=coefficient,
            frequency_ratio=parameters.frequency_ratio_peak,
        )
        target = point.peak_amplitude_over_depth
        predictions.append(
            {
                "scruton_number": float(row["scruton_number"]),
                "target_peak_amplitude_over_depth": target,
                "predicted_peak_amplitude_over_depth": predicted,
                "absolute_relative_error": abs(predicted - target) / target,
            }
        )
    return {
        "schema": CALIBRATION_SCHEMA,
        "status": "trained_4x1_ready_for_untouched_validation",
        "model_type": MODEL_TYPE,
        "method": {
            "name": "static-spectrum-scaled Gao--Scanlan reciprocal-Griffin work closure",
            "force_per_span": "0.5*rho*U^2*D*Ceff*w*(Gc*sign(v)-Gv*v/U)",
            "cycle_mean_aerodynamic_power": "0.5*rho*U^2*D*Ceff*w*(2*Gc*V/pi-Gv*V^2/(2U))",
            "structural_power": "0.5*(2*zeta*m*omega)*V^2",
            "positive_power_sign": "aerodynamic force does work on the structure",
            "static_spectrum_role": "shedding frequency and sinusoidal-equivalent force scale only",
            "motion_phase_from_static_spectrum": False,
        },
        "parameters": {
            "coulomb_work_gain": fit.coulomb_work_gain,
            "velocity_damping_gain": fit.velocity_damping_gain,
            "frequency_ratio_low": parameters.frequency_ratio_low,
            "frequency_ratio_peak": parameters.frequency_ratio_peak,
            "frequency_ratio_high": parameters.frequency_ratio_high,
            "regularization_velocity_over_wind": parameters.regularization_velocity_over_wind,
        },
        "training": {
            "study": "Marra et al. 4:1 rectangular cylinder",
            "row_count": len(points),
            "response_quantity": "peak harmonic displacement amplitude Ay/D",
            "static_sinusoidal_equivalent_coefficient": coefficient,
            "fit_space": "linear least squares in reciprocal dimensional amplitude",
            "rank": fit.rank,
            "maximum_relative_error": max(
                row["absolute_relative_error"] for row in predictions
            ),
            "rows": predictions,
            "frequency_window_source": "previously failed Mannini 3:2 response",
        },
        "sources": {
            "marra_training_csv": {
                "path": str(DATA.relative_to(REPO)),
                "sha256": _sha256(DATA),
            },
            "marra_provenance": {
                "path": str(PROVENANCE.relative_to(REPO)),
                "sha256": _sha256(PROVENANCE),
            },
            "mannini_frequency_training": {
                "path": str(MANNINI_DATA.relative_to(REPO)),
                "sha256": _sha256(MANNINI_DATA),
            },
            "liu_validation_freeze": {
                "path": str(LIU_PROVENANCE.relative_to(REPO)),
                "sha256": _sha256(LIU_PROVENANCE),
            },
        },
        "claim_boundary": {
            "marra_4x1_is_training": True,
            "mannini_3x2_is_training": True,
            "liu_5x1_used_for_parameter_fitting": False,
            "ready_for_untouched_liu_assessment": True,
            "does_not_validate_hvawt_or_fatigue": True,
        },
    }


def _plot(payload: dict[str, Any], path: Path) -> None:
    rows = payload["training"]["rows"]
    x = [row["scruton_number"] for row in rows]
    target = [row["target_peak_amplitude_over_depth"] for row in rows]
    predicted = [row["predicted_peak_amplitude_over_depth"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].plot(x, target, "o-", color="#222222", label="Experiment")
    axes[0].plot(x, predicted, "s--", color="#2468a2", label="Fitted closure")
    axes[0].set(
        xlabel="Scruton number", ylabel=r"Peak $A_y/D$", xlim=(0, 82), ylim=(0, 0.07)
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(
        x, [1.0 / value for value in target], "o-", color="#222222", label="Experiment"
    )
    axes[1].plot(
        x,
        [1.0 / value for value in predicted],
        "s--",
        color="#2468a2",
        label="Fitted closure",
    )
    axes[1].set(
        xlabel="Scruton number", ylabel=r"Reciprocal peak $D/A_y$", xlim=(0, 82)
    )
    axes[1].grid(True, alpha=0.25)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE / "griffin_lockin_calibration.json"
    )
    parser.add_argument(
        "--plot", type=Path, default=HERE / "evidence/griffin_training_fit.pdf"
    )
    args = parser.parse_args(argv)
    payload = build_calibration()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _plot(payload, args.plot)
    (args.plot.parent / "griffin_training_SHA256SUMS").write_text(
        f"{_sha256(args.output)}  {args.output.name}\n"
        f"{_sha256(args.plot)}  {args.plot.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
