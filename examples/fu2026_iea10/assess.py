#!/usr/bin/env python3
"""Assess frozen model predictions against the Fu multi-condition HFM matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.fu2026_iea10.validate import load_reference


PROTOCOL_SCHEMA = "fu2026-iea10-multicondition-validation-protocol-v1"
ASSESSMENT_SCHEMA = "fu2026-iea10-multicondition-validation-assessment-v1"
PREDICTION_COLUMNS = (
    "case_id",
    "model_fingerprint",
    "Ay_pred_over_cbar",
    "fy_pred_over_fn2",
)


class AssessmentError(ValueError):
    """Raised when predictions or the frozen protocol are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssessmentError(message)


def _reject_json_constant(value: str) -> None:
    raise AssessmentError(f"Non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise AssessmentError(f"Could not load strict JSON from {path}: {exc}") from exc
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise AssessmentError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AssessmentError(f"{name} must be a finite real number.") from exc
    _require(math.isfinite(result), f"{name} must be finite.")
    return result


def load_protocol(path: Path = ROOT / "validation_protocol.json") -> dict[str, Any]:
    protocol = _load_json(path)
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "Unexpected protocol schema.")
    _require(
        protocol.get("status") == "frozen_before_turbine_model_evaluation",
        "Protocol is not frozen before evaluation.",
    )
    reference = load_reference(ROOT)
    reference_meta = protocol.get("reference")
    _require(isinstance(reference_meta, dict), "Protocol reference must be an object.")
    _require(
        reference_meta.get("targets_sha256") == reference["targets_sha256"],
        "Protocol target hash does not match the frozen Fu reference.",
    )
    _require(
        reference_meta.get("source_pdf_sha256")
        == reference["provenance"]["publication"]["vector_source"]["sha256"],
        "Protocol source-PDF hash does not match the Fu reference.",
    )

    rows_by_id = reference["rows_by_id"]
    scored = protocol.get("primary_scored_case_ids")
    training_linked = protocol.get("training_linked_case_ids")
    _require(isinstance(scored, list), "Scored case IDs must be a list.")
    _require(isinstance(training_linked, list), "Training-linked case IDs must be a list.")
    _require(len(scored) == 16 and len(set(scored)) == 16, "Expected 16 unique scored cases.")
    _require(
        len(training_linked) == 4 and len(set(training_linked)) == 4,
        "Expected four unique training-linked cases.",
    )
    _require(not set(scored) & set(training_linked), "Evidence roles overlap.")
    _require(
        set(scored) | set(training_linked) == set(rows_by_id),
        "Evidence roles do not cover the exact 20-row Fu matrix.",
    )
    _require(
        all(int(rows_by_id[case_id]["U_mps"]) != 18 for case_id in scored),
        "A U=18 m/s row is incorrectly scored as independent of training.",
    )
    _require(
        all(int(rows_by_id[case_id]["U_mps"]) == 18 for case_id in training_linked),
        "Training-linked rows must be exactly the U=18 m/s cases.",
    )
    active_threshold = _finite_float(
        protocol["acceptance"]["active_target_threshold_Ay_over_cbar"],
        "active threshold",
    )
    active_count = sum(
        float(rows_by_id[case_id]["Ay_over_cbar"]) >= active_threshold
        for case_id in scored
    )
    _require(
        active_count >= protocol["matrix_requirements"]["minimum_scoreable_viv_points"],
        "The frozen matrix has too few active VIV points.",
    )
    return {"protocol": protocol, "reference": reference}


def _load_predictions(path: Path) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(
                tuple(reader.fieldnames or ()) == PREDICTION_COLUMNS,
                "Prediction CSV columns or ordering changed.",
            )
            rows = list(reader)
    except OSError as exc:
        raise AssessmentError(f"Could not load predictions from {path}: {exc}") from exc
    _require(all(None not in row for row in rows), "Prediction CSV has extra fields.")
    _require(
        all(all(value != "" for value in row.values()) for row in rows),
        "Prediction CSV contains an empty field.",
    )
    return rows


def assess_predictions(
    predictions_path: Path,
    protocol_path: Path = ROOT / "validation_protocol.json",
) -> dict[str, Any]:
    loaded = load_protocol(protocol_path)
    protocol = loaded["protocol"]
    reference = loaded["reference"]
    targets = reference["rows_by_id"]
    predictions = _load_predictions(predictions_path)
    scored_ids = tuple(protocol["primary_scored_case_ids"])
    _require(len(predictions) == len(scored_ids), "Expected exactly 16 prediction rows.")
    prediction_ids = [row["case_id"] for row in predictions]
    _require(len(set(prediction_ids)) == len(prediction_ids), "Prediction case IDs repeat.")
    _require(
        set(prediction_ids) == set(scored_ids),
        "Predictions must contain exactly the 16 primary scored cases.",
    )
    fingerprints = {row["model_fingerprint"] for row in predictions}
    _require(len(fingerprints) == 1, "All predictions must use one frozen model fingerprint.")
    model_fingerprint = next(iter(fingerprints))
    _require(
        len(model_fingerprint) == 64
        and all(character in "0123456789abcdef" for character in model_fingerprint),
        "Model fingerprint must be a lowercase SHA-256 string.",
    )

    acceptance = protocol["acceptance"]
    active_threshold = float(acceptance["active_target_threshold_Ay_over_cbar"])
    amplitude_fraction = float(
        acceptance["maximum_active_amplitude_relative_error_after_digitization_allowance"]
    )
    inactive_limit = float(acceptance["maximum_inactive_predicted_Ay_over_cbar"])
    frequency_limit = float(acceptance["maximum_frequency_ratio_absolute_error"])
    peak_speed_limit = float(acceptance["maximum_configuration_peak_speed_error_mps"])

    rows_by_id: dict[str, dict[str, Any]] = {}
    row_results: list[dict[str, Any]] = []
    for prediction in predictions:
        case_id = prediction["case_id"]
        target = targets[case_id]
        amplitude = _finite_float(
            prediction["Ay_pred_over_cbar"], f"{case_id} predicted amplitude"
        )
        frequency = _finite_float(
            prediction["fy_pred_over_fn2"], f"{case_id} predicted frequency ratio"
        )
        _require(amplitude >= 0.0, f"{case_id} predicted amplitude must be non-negative.")
        _require(frequency >= 0.0, f"{case_id} predicted frequency must be non-negative.")
        target_amplitude = float(target["Ay_over_cbar"])
        target_frequency = float(target["fy_over_fn2"])
        digitization = float(target["Ac_unc"])
        active = target_amplitude >= active_threshold
        absolute_amplitude_error = abs(amplitude - target_amplitude)
        relative_error_after_allowance = (
            max(0.0, absolute_amplitude_error - digitization) / target_amplitude
            if active
            else None
        )
        amplitude_pass = (
            relative_error_after_allowance <= amplitude_fraction
            if active
            else amplitude <= inactive_limit
        )
        frequency_error = abs(frequency - target_frequency) if active else None
        frequency_pass = frequency_error <= frequency_limit if active else True
        row_result = {
            "case_id": case_id,
            "config": target["config"],
            "wind_speed_mps": int(target["U_mps"]),
            "target_active": active,
            "target_Ay_over_cbar": target_amplitude,
            "predicted_Ay_over_cbar": amplitude,
            "absolute_amplitude_error": absolute_amplitude_error,
            "relative_amplitude_error_after_digitization_allowance": (
                relative_error_after_allowance
            ),
            "amplitude_or_classification_pass": amplitude_pass,
            "target_fy_over_fn2": target_frequency,
            "predicted_fy_over_fn2": frequency,
            "frequency_ratio_absolute_error": frequency_error,
            "frequency_pass": frequency_pass,
            "pass": amplitude_pass and frequency_pass,
        }
        rows_by_id[case_id] = row_result
        row_results.append(row_result)

    configuration_results: list[dict[str, Any]] = []
    for config in sorted({targets[case_id]["config"] for case_id in scored_ids}):
        config_rows = [row for row in row_results if row["config"] == config]
        _require(len(config_rows) == 4, f"{config} must have four scored speeds.")
        target_peak = max(config_rows, key=lambda row: row["target_Ay_over_cbar"])
        predicted_peak = max(config_rows, key=lambda row: row["predicted_Ay_over_cbar"])
        peak_speed_error = abs(
            predicted_peak["wind_speed_mps"] - target_peak["wind_speed_mps"]
        )
        configuration_results.append(
            {
                "config": config,
                "target_peak_speed_mps": target_peak["wind_speed_mps"],
                "predicted_peak_speed_mps": predicted_peak["wind_speed_mps"],
                "peak_speed_error_mps": peak_speed_error,
                "peak_speed_pass": peak_speed_error <= peak_speed_limit,
            }
        )

    active_rows = [row for row in row_results if row["target_active"]]
    inactive_rows = [row for row in row_results if not row["target_active"]]
    gates = {
        "exact_16_case_matrix": len(row_results) == 16,
        "minimum_10_active_viv_points": len(active_rows) >= 10,
        "all_active_amplitudes_within_30_percent_plus_digitization": all(
            row["amplitude_or_classification_pass"] for row in active_rows
        ),
        "all_inactive_cases_below_0p02_cbar": all(
            row["amplitude_or_classification_pass"] for row in inactive_rows
        ),
        "all_active_frequencies_within_0p10_fn": all(
            row["frequency_pass"] for row in active_rows
        ),
        "all_configuration_peak_speeds_within_one_interval": all(
            row["peak_speed_pass"] for row in configuration_results
        ),
    }
    overall_pass = all(gates.values())
    return {
        "schema": ASSESSMENT_SCHEMA,
        "status": "passed" if overall_pass else "failed",
        "pass": overall_pass,
        "claim": (
            "passed_multi_condition_turbine_hfm_validation"
            if overall_pass
            else "no_turbine_validation_claim"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "targets_sha256": reference["targets_sha256"],
        "predictions_sha256": sha256_file(predictions_path),
        "model_fingerprint": model_fingerprint,
        "scored_case_count": len(row_results),
        "active_viv_case_count": len(active_rows),
        "inactive_case_count": len(inactive_rows),
        "training_linked_case_ids_not_scored": protocol["training_linked_case_ids"],
        "gates": gates,
        "row_results": sorted(row_results, key=lambda row: row["case_id"]),
        "configuration_results": configuration_results,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "validation_protocol.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assessment = assess_predictions(args.predictions, args.protocol)
    encoded = json.dumps(assessment, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if assessment["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
