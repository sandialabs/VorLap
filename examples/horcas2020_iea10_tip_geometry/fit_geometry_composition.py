#!/usr/bin/env python3
"""Fit a turbine work-margin closure without reading the COMB holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAINING = HERE / "training_targets.csv"
PROTOCOL = HERE / "validation_protocol.json"
FU_TARGETS = REPO / "examples/fu2026_iea10/targets.csv"
FU_V1_MODEL = REPO / "examples/fu2026_iea10/evidence/turbine_spectral_v1/model.json"
FU_V1_PREDICTIONS = REPO / "examples/fu2026_iea10/evidence/turbine_spectral_v1/predictions.csv"
FU_V1_ASSESSMENT = REPO / "examples/fu2026_iea10/evidence/turbine_spectral_v1/assessment.json"
POWER_CURVE = REPO / "examples/grinderslev2022_iea10/reference_data/p100_i50_figure6_power_digitized.csv"
STATIC_SPECTRUM = REPO / "data/airfoils/ffa_w3_211.h5"
DEFAULT_OUTPUT = HERE / "evidence/combined_tip_v1"

SCHEMA = "vorlap-turbine-geometry-composed-negative-damping-v1"
MODEL_TYPE = "static_spectrum_geometry_composed_negative_damping"
TRAINING_SHA = "95517f2aa93b56f61d91147859f977288ba32572e9de91407e27eec8770de28b"
PROTOCOL_SHA = "1b9be3e7ca7d15d73ff268a6e82429125b5c0ee60c3cd9951328f0c4d04a1947"
FU_TARGETS_SHA = "64e958175ade5423bf51c15ed68756a8458199fc06601b60205f2bf14ff39ff1"
FU_V1_MODEL_SHA = "71c5ee621c50b92201163762bd61832adb7f9409465e69687b4f15bb9fe50eb8"
FU_V1_PREDICTIONS_SHA = "bf2273c04f1e491671ff3af7c6715d03329cf4a45b9c1334ab07afa958c3fde9"
FU_V1_ASSESSMENT_SHA = "ac1522d91d3f44564d479fc79551230e3dd9d74a592520a77ec70845f3e28dd9"
POWER_CURVE_SHA = "016ce5265b77dc0c9e577d7548e453d09f67ef732dda817618f79909546f323d"
STATIC_SPECTRUM_SHA = "7f968617b10dab530bba5418b0802b885195f585d70d6218ea0083d939dc95dc"
SATURATION_M = 1.57
TRAINING_GEOMETRIES = ("BASE", "EXT", "SWEEP", "DIHE")
COMPOSITION_GEOMETRIES = ("EXT", "SWEEP", "DIHE")
INCLINATIONS = (40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 65.0, 70.0)


class FitError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FitError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _power_margin(amplitude_m: float, saturation_m: float) -> float:
    _require(math.isfinite(amplitude_m) and 0.0 <= amplitude_m < saturation_m, "Amplitude is outside the signed-saturation branch.")
    return (amplitude_m / saturation_m) ** 2


def _feedback_ratio(margin: float) -> float:
    _require(0.0 <= margin < 1.0, "Work margin must be in [0,1).")
    return 1.0 / (1.0 - margin)


def _verify_inputs() -> None:
    expected = {
        TRAINING: TRAINING_SHA,
        PROTOCOL: PROTOCOL_SHA,
        FU_TARGETS: FU_TARGETS_SHA,
        FU_V1_MODEL: FU_V1_MODEL_SHA,
        FU_V1_PREDICTIONS: FU_V1_PREDICTIONS_SHA,
        FU_V1_ASSESSMENT: FU_V1_ASSESSMENT_SHA,
        POWER_CURVE: POWER_CURVE_SHA,
        STATIC_SPECTRUM: STATIC_SPECTRUM_SHA,
    }
    for path, digest in expected.items():
        _require(_sha(path) == digest, f"Training/source identity changed: {path}")
    assessment = json.loads(FU_V1_ASSESSMENT.read_text(encoding="utf-8"))
    _require(assessment.get("status") == "failed", "The frozen Fu failure must precede reuse as training.")
    curve = _rows(POWER_CURVE)
    zero_crossings = [
        float(row["tip_amplitude_m"])
        for row in curve
        if float(row["tip_amplitude_m"]) > 0.0
        and float(row["aerodynamic_power_w"]) == 0.0
    ]
    _require(zero_crossings == [SATURATION_M], "Signed-power saturation identity changed.")


def fit_model() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit using only the 44-row training snapshot and previously failed Fu data."""
    _verify_inputs()
    training = _rows(TRAINING)
    _require(len(training) == 44, "Expected exactly 44 Horcas training rows.")
    _require({row["geometry"] for row in training} == set(TRAINING_GEOMETRIES), "Training geometry set changed.")
    _require(all(row["role"] == "training" for row in training), "Holdout row leaked into training snapshot.")
    _require(all(not row["case_id"].startswith("COMB_") for row in training), "COMB holdout leaked into fitting.")

    curves: dict[str, dict[float, float]] = {geometry: {} for geometry in TRAINING_GEOMETRIES}
    margins: dict[str, list[dict[str, float]]] = {geometry: [] for geometry in TRAINING_GEOMETRIES}
    for row in training:
        geometry = row["geometry"]
        inclination = float(row["inclination_deg"])
        amplitude = float(row["peak_edge_amplitude_m"])
        margin = _power_margin(amplitude, SATURATION_M)
        curves[geometry][inclination] = amplitude
        margins[geometry].append(
            {
                "inclination_deg": inclination,
                "stable_peak_amplitude_m": amplitude,
                "normalized_positive_work_margin": margin,
                "zero_amplitude_feedback_over_structural_damping": _feedback_ratio(margin),
            }
        )
    for geometry in TRAINING_GEOMETRIES:
        _require(tuple(curves[geometry]) == INCLINATIONS, f"Training inclination grid changed for {geometry}.")

    predictions: list[dict[str, Any]] = []
    for inclination in INCLINATIONS:
        component_margins = {
            geometry: _power_margin(curves[geometry][inclination], SATURATION_M)
            for geometry in COMPOSITION_GEOMETRIES
        }
        combined_margin = sum(component_margins.values()) / len(component_margins)
        amplitude = SATURATION_M * math.sqrt(combined_margin)
        predictions.append(
            {
                "case_id": f"COMB_I{inclination:05.1f}".replace(".", "p"),
                "inclination_deg": inclination,
                "predicted_peak_edge_amplitude_m": amplitude,
                "predicted_response_frequency_over_fn": 1.0,
                "combined_normalized_positive_work_margin": combined_margin,
                "zero_amplitude_feedback_over_structural_damping": _feedback_ratio(combined_margin),
                **{f"{key.lower()}_work_margin": value for key, value in component_margins.items()},
            }
        )

    fu_rows = _rows(FU_TARGETS)
    _require(len(fu_rows) == 20, "Expected the complete 20-row failed Fu matrix as training.")
    fu_by_config: dict[str, list[dict[str, str]]] = {}
    for row in fu_rows:
        fu_by_config.setdefault(row["config"], []).append(row)
    fu_training: list[dict[str, Any]] = []
    for config, rows in sorted(fu_by_config.items()):
        saturation = max(SATURATION_M, 1.05 * max(float(row["Ay_m"]) for row in rows))
        for row in rows:
            amplitude = float(row["Ay_m"])
            margin = _power_margin(amplitude, saturation)
            fu_training.append(
                {
                    "case_id": row["case_id"],
                    "config": config,
                    "wind_speed_mps": float(row["U_mps"]),
                    "stable_peak_amplitude_m": amplitude,
                    "configuration_saturation_amplitude_m": saturation,
                    "normalized_positive_work_margin": margin,
                    "zero_amplitude_feedback_over_structural_damping": _feedback_ratio(margin),
                }
            )

    model: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "frozen_before_horcas_combined_geometry_evaluation",
        "model_type": MODEL_TYPE,
        "method": {
            "name": "static-spectrum work-equivalent negative damping with quadratic saturation and geometry-composed work margin",
            "generalized_force": "Q=c_a0*(1-(A/A_sat)^2)*qdot_filtered",
            "stable_equilibrium": "mu=1-c_s/c_a0=(A/A_sat)^2",
            "geometry_composition": "mu_COMB=(mu_EXT+mu_SWEEP+mu_DIHE)/3",
            "amplitude_prediction": "A_COMB=A_sat*sqrt(mu_COMB)=rms(A_EXT,A_SWEEP,A_DIHE)",
            "frequency_prediction": "f_response/f_n=1 inside the resolved work-positive branch",
            "static_spectrum_role": "Fixed-body FFA-W3 shedding frequency and available force scale remain the exogenous input; the trained work-margin surface stretches that scale into motion-correlated negative damping.",
            "stationary_spectrum_phase_used": False,
            "positive_power_sign": "positive supplemental force-velocity work supplies negative aerodynamic damping",
        },
        "parameters": {
            "grinderslev_signed_power_saturation_tip_amplitude_m": SATURATION_M,
            "geometry_margin_weights": {geometry: 1.0 / 3.0 for geometry in COMPOSITION_GEOMETRIES},
            "resolved_viv_frequency_ratio": 1.0,
        },
        "training": {
            "horcas2020_primitive_geometry_rows": margins,
            "fu2026_speed_orientation_rows_after_preserved_failure": fu_training,
            "fu_row_count": len(fu_training),
            "horcas_row_count": len(training),
        },
        "protected_validation": {
            "study": "Horcas et al. 2020 IEA-10-MW combined sweep-plus-dihedral tip",
            "geometry": "COMB",
            "case_count": 11,
            "comb_rows_used_for_fitting": False,
            "protocol_path": "examples/horcas2020_iea10_tip_geometry/validation_protocol.json",
            "protocol_sha256": PROTOCOL_SHA,
        },
        "sources": {
            "horcas_training_snapshot": {"path": str(TRAINING.relative_to(REPO)), "sha256": TRAINING_SHA},
            "fu_targets_reclassified_as_training_after_failure": {"path": str(FU_TARGETS.relative_to(REPO)), "sha256": FU_TARGETS_SHA},
            "fu_v1_frozen_model": {"path": str(FU_V1_MODEL.relative_to(REPO)), "sha256": FU_V1_MODEL_SHA},
            "fu_v1_frozen_predictions": {"path": str(FU_V1_PREDICTIONS.relative_to(REPO)), "sha256": FU_V1_PREDICTIONS_SHA},
            "fu_v1_failed_assessment": {"path": str(FU_V1_ASSESSMENT.relative_to(REPO)), "sha256": FU_V1_ASSESSMENT_SHA},
            "grinderslev_signed_power_curve": {"path": str(POWER_CURVE.relative_to(REPO)), "sha256": POWER_CURVE_SHA},
            "ffa_w3_static_spectrum": {"path": str(STATIC_SPECTRUM.relative_to(REPO)), "sha256": STATIC_SPECTRUM_SHA},
            "fit_code": {"path": str(Path(__file__).resolve().relative_to(REPO)), "sha256": _sha(Path(__file__).resolve())},
        },
        "claim_boundary": {
            "trained_semi_empirical_turbine_closure": True,
            "validated_before_holdout_evaluation": False,
            "fatigue_validated": False,
        },
    }
    fingerprint_payload = dict(model)
    model["model_fingerprint"] = _canonical_fingerprint(fingerprint_payload)
    for row in predictions:
        row["model_fingerprint"] = model["model_fingerprint"]
    return model, predictions


def write_frozen_model(output: Path = DEFAULT_OUTPUT) -> tuple[Path, Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, predictions = fit_model()
    model_path = output / "model.json"
    prediction_path = output / "predictions.csv"
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = (
        "case_id",
        "model_fingerprint",
        "inclination_deg",
        "predicted_peak_edge_amplitude_m",
        "predicted_response_frequency_over_fn",
        "combined_normalized_positive_work_margin",
        "zero_amplitude_feedback_over_structural_damping",
        "ext_work_margin",
        "sweep_work_margin",
        "dihe_work_margin",
    )
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    return model_path, prediction_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model, predictions = write_frozen_model(args.output)
    print(json.dumps({"model": str(model), "predictions": str(predictions)}, indent=2))


if __name__ == "__main__":
    main()
