#!/usr/bin/env python3
"""Assess the frozen geometry-composed closure on the COMB HFM holdout."""

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

from .validate import load_reference

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence/combined_tip_v1"
MODEL_PATH = EVIDENCE / "model.json"
PREDICTIONS_PATH = EVIDENCE / "predictions.csv"
ASSESSMENT_PATH = EVIDENCE / "assessment.json"
FIGURE_PATH = EVIDENCE / "validation.pdf"

SCHEMA = "horcas2020-combined-tip-validation-assessment-v1"
MODEL_SHA = "656a2dac90eda862ec02d6e893679b1118ca5043d962c19c518b978ba202a0dc"
PREDICTIONS_SHA = "62b6f38008263a1f0c76d772a4a8b97666c7906189e3539a07f85f5dc35ddb69"
EXPECTED_MODEL_FINGERPRINT = "8b067d6f3a049f665c451f545a25dc8a0345b0a5bc829a3715f691038f899891"


class AssessmentError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssessmentError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolved_interval(inclinations: list[float], amplitudes: list[float], fraction: float) -> dict[str, Any]:
    _require(len(inclinations) == len(amplitudes) and amplitudes, "Invalid interval arrays.")
    peak_index = max(range(len(amplitudes)), key=amplitudes.__getitem__)
    threshold = fraction * amplitudes[peak_index]
    active = [value >= threshold for value in amplitudes]
    left = peak_index
    right = peak_index
    while left > 0 and active[left - 1]:
        left -= 1
    while right + 1 < len(active) and active[right + 1]:
        right += 1
    return {
        "threshold_m": threshold,
        "indices": list(range(left, right + 1)),
        "inclinations_deg": inclinations[left : right + 1],
        "onset_deg": inclinations[left],
        "collapse_deg": inclinations[right],
        "point_count": right - left + 1,
    }


def assess() -> dict[str, Any]:
    _require(_sha(MODEL_PATH) == MODEL_SHA, "Frozen model SHA-256 mismatch.")
    _require(_sha(PREDICTIONS_PATH) == PREDICTIONS_SHA, "Frozen predictions SHA-256 mismatch.")
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    _require(model.get("model_fingerprint") == EXPECTED_MODEL_FINGERPRINT, "Model fingerprint mismatch.")
    _require(model.get("status") == "frozen_before_horcas_combined_geometry_evaluation", "Model was not frozen before evaluation.")
    _require(model["protected_validation"]["comb_rows_used_for_fitting"] is False, "COMB leaked into fitting.")
    reference = load_reference()
    protocol = reference["protocol"]
    targets = reference["by_geometry"]["COMB"]
    predictions = _rows(PREDICTIONS_PATH)
    _require(len(targets) == len(predictions) == 11, "Expected exactly 11 holdout comparisons.")
    target_by_id = {row["case_id"]: row for row in targets}
    prediction_by_id = {row["case_id"]: row for row in predictions}
    _require(set(target_by_id) == set(prediction_by_id), "Prediction case IDs differ from holdout IDs.")
    _require({row["model_fingerprint"] for row in predictions} == {EXPECTED_MODEL_FINGERPRINT}, "Prediction fingerprint mismatch.")

    scored: list[dict[str, Any]] = []
    for target in targets:
        prediction = prediction_by_id[target["case_id"]]
        target_amplitude = float(target["peak_edge_amplitude_m"])
        predicted_amplitude = float(prediction["predicted_peak_edge_amplitude_m"])
        target_frequency = float(target["response_frequency_over_fn"])
        predicted_frequency = float(prediction["predicted_response_frequency_over_fn"])
        scored.append(
            {
                "case_id": target["case_id"],
                "inclination_deg": float(target["inclination_deg"]),
                "target_peak_edge_amplitude_m": target_amplitude,
                "predicted_peak_edge_amplitude_m": predicted_amplitude,
                "amplitude_error_m": predicted_amplitude - target_amplitude,
                "absolute_amplitude_error_m": abs(predicted_amplitude - target_amplitude),
                "target_response_frequency_over_fn": target_frequency,
                "predicted_response_frequency_over_fn": predicted_frequency,
                "frequency_ratio_absolute_error": abs(predicted_frequency - target_frequency),
            }
        )

    inclinations = [row["inclination_deg"] for row in scored]
    target_amplitudes = [row["target_peak_edge_amplitude_m"] for row in scored]
    predicted_amplitudes = [row["predicted_peak_edge_amplitude_m"] for row in scored]
    target_peak_index = max(range(len(scored)), key=target_amplitudes.__getitem__)
    predicted_peak_index = max(range(len(scored)), key=predicted_amplitudes.__getitem__)
    target_peak = target_amplitudes[target_peak_index]
    predicted_peak = predicted_amplitudes[predicted_peak_index]
    rmse = math.sqrt(sum((prediction - target) ** 2 for prediction, target in zip(predicted_amplitudes, target_amplitudes)) / len(scored))
    nrmse = rmse / target_peak

    resolved_fraction = float(protocol["resolved_viv_definition"]["fraction_of_geometry_peak"])
    target_interval = _resolved_interval(inclinations, target_amplitudes, resolved_fraction)
    predicted_interval = _resolved_interval(inclinations, predicted_amplitudes, resolved_fraction)
    frequency_rows = [scored[index] for index in target_interval["indices"]]

    acceptance = protocol["acceptance"]
    curve_pass = nrmse <= float(acceptance["maximum_curve_nrmse_by_target_peak"])
    peak_tolerance = (
        float(acceptance["maximum_peak_amplitude_relative_error_after_digitization_allowance"]) * target_peak
        + float(acceptance["peak_digitization_allowance_m"])
    )
    peak_amplitude_pass = abs(predicted_peak - target_peak) <= peak_tolerance
    peak_inclination_error = abs(inclinations[predicted_peak_index] - inclinations[target_peak_index])
    peak_inclination_pass = peak_inclination_error <= float(acceptance["maximum_peak_inclination_error_deg"])
    onset_error = abs(predicted_interval["onset_deg"] - target_interval["onset_deg"])
    collapse_error = abs(predicted_interval["collapse_deg"] - target_interval["collapse_deg"])
    interval_pass = max(onset_error, collapse_error) <= float(acceptance["maximum_interval_endpoint_error_deg"])
    frequency_limit = float(acceptance["maximum_frequency_ratio_absolute_error"])
    frequency_pass = all(row["frequency_ratio_absolute_error"] <= frequency_limit for row in frequency_rows)
    all_pass = all((curve_pass, peak_amplitude_pass, peak_inclination_pass, interval_pass, frequency_pass))

    amplitude_multiplier = 1.30
    additive_floor = max(0.0, max(target - amplitude_multiplier * prediction for target, prediction in zip(target_amplitudes, predicted_amplitudes)))
    saturation = float(model["parameters"]["grinderslev_signed_power_saturation_tip_amplitude_m"])
    envelope_values = [amplitude_multiplier * prediction + additive_floor for prediction in predicted_amplitudes]
    _require(all(envelope + 1.0e-14 >= target for envelope, target in zip(envelope_values, target_amplitudes)), "Derived conservative envelope does not cover all holdout rows.")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if all_pass else "failed",
        "claim": "passed_multi_condition_turbine_hfm_geometry_validation" if all_pass else "no_turbine_geometry_validation_claim",
        "model_fingerprint": EXPECTED_MODEL_FINGERPRINT,
        "source_identities": {
            "model_sha256": MODEL_SHA,
            "predictions_sha256": PREDICTIONS_SHA,
            "targets_sha256": reference["targets_sha256"],
            "provenance_sha256": reference["provenance_sha256"],
            "assessment_code_sha256": _sha(Path(__file__).resolve()),
        },
        "matrix": {
            "training_geometry_count": 4,
            "training_case_count": 44,
            "heldout_geometry": "COMB",
            "heldout_case_count": 11,
            "all_holdout_rows_reported": len(scored) == 11,
        },
        "metrics": {
            "curve_rmse_m": rmse,
            "curve_nrmse_by_target_peak": nrmse,
            "target_peak_amplitude_m": target_peak,
            "predicted_peak_amplitude_m": predicted_peak,
            "peak_amplitude_absolute_error_m": abs(predicted_peak - target_peak),
            "peak_amplitude_relative_error": abs(predicted_peak - target_peak) / target_peak,
            "peak_amplitude_tolerance_m": peak_tolerance,
            "target_peak_inclination_deg": inclinations[target_peak_index],
            "predicted_peak_inclination_deg": inclinations[predicted_peak_index],
            "peak_inclination_error_deg": peak_inclination_error,
            "target_resolved_interval": target_interval,
            "predicted_resolved_interval": predicted_interval,
            "interval_onset_error_deg": onset_error,
            "interval_collapse_error_deg": collapse_error,
            "maximum_resolved_frequency_ratio_error": max(row["frequency_ratio_absolute_error"] for row in frequency_rows),
        },
        "gates": {
            "curve_nrmse": curve_pass,
            "peak_amplitude": peak_amplitude_pass,
            "peak_inclination": peak_inclination_pass,
            "finite_interval": interval_pass,
            "resolved_frequency": frequency_pass,
            "all": all_pass,
        },
        "conservative_engineering_envelope": {
            "status": "post_validation_discrepancy_bound_not_independent_validation",
            "formula": "A_design=1.30*A_model+delta_A",
            "multiplicative_factor": amplitude_multiplier,
            "additive_floor_m": additive_floor,
            "additive_floor_over_signed_saturation": additive_floor / saturation,
            "covers_all_11_heldout_rows": True,
            "fatigue_use_boundary": "The bound may be scaled with the local model saturation for conservative H-VAWT screening; it is not itself a fatigue validation and must be applied to cycle-load amplitude before rainflow/DEL calculation.",
        },
        "rows": scored,
        "claim_boundary": protocol["claim_boundary"],
    }
    return result


def write_figure(result: dict[str, Any], path: Path = FIGURE_PATH) -> Path:
    rows = result["rows"]
    inclination = [row["inclination_deg"] for row in rows]
    target_amplitude = [row["target_peak_edge_amplitude_m"] for row in rows]
    predicted_amplitude = [row["predicted_peak_edge_amplitude_m"] for row in rows]
    target_frequency = [row["target_response_frequency_over_fn"] for row in rows]
    predicted_frequency = [row["predicted_response_frequency_over_fn"] for row in rows]

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), constrained_layout=True)
    axes[0].plot(inclination, target_amplitude, "o-", color="#1f4e79", lw=1.6, ms=4.5, label="Horcas HFM, COMB (held out)")
    axes[0].plot(inclination, predicted_amplitude, "s--", color="#c55a11", lw=1.5, ms=4.0, label="Frozen hybrid closure")
    axes[0].axvspan(result["metrics"]["target_resolved_interval"]["onset_deg"], result["metrics"]["target_resolved_interval"]["collapse_deg"], color="#1f4e79", alpha=0.10, lw=0)
    axes[0].set(xlabel="Inflow inclination, $I$ (deg)", ylabel="Edgewise peak amplitude (m)", title="(a) Response amplitude")
    axes[0].set_xlim(39, 71)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].plot(inclination, target_frequency, "o", color="#1f4e79", ms=4.5, label="Horcas HFM, COMB (held out)")
    axes[1].plot(inclination, predicted_frequency, "--", color="#c55a11", lw=1.5, label="Lock-in prediction")
    axes[1].axhspan(0.9, 1.1, color="#6aa84f", alpha=0.10, lw=0)
    axes[1].set(xlabel="Inflow inclination, $I$ (deg)", ylabel="$f_y/f_n$", title="(b) Response frequency")
    axes[1].set_xlim(39, 71)
    axes[1].set_ylim(0.88, 1.14)
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(frameon=False, loc="upper right")
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessment", type=Path, default=ASSESSMENT_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args()
    result = assess()
    args.assessment.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_figure(result, args.figure)
    print(json.dumps({"assessment": str(args.assessment), "figure": str(args.figure), "status": result["status"], "metrics": result["metrics"], "gates": result["gates"]}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
