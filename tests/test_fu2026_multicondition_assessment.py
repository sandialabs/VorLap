from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from examples.fu2026_iea10 import assess


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "examples" / "fu2026_iea10"
MODEL_FINGERPRINT = "1" * 64


def _prediction_rows() -> list[dict[str, str]]:
    loaded = assess.load_protocol()
    targets = loaded["reference"]["rows_by_id"]
    return [
        {
            "case_id": case_id,
            "model_fingerprint": MODEL_FINGERPRINT,
            "Ay_pred_over_cbar": targets[case_id]["Ay_over_cbar"],
            "fy_pred_over_fn2": targets[case_id]["fy_over_fn2"],
        }
        for case_id in loaded["protocol"]["primary_scored_case_ids"]
    ]


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=assess.PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_protocol_freezes_four_configurations_four_speeds_and_ten_active_points() -> None:
    loaded = assess.load_protocol()
    protocol = loaded["protocol"]
    targets = loaded["reference"]["rows_by_id"]
    scored = protocol["primary_scored_case_ids"]
    training_linked = protocol["training_linked_case_ids"]

    assert len(scored) == 16
    assert len(training_linked) == 4
    assert {int(targets[case_id]["U_mps"]) for case_id in scored} == {6, 10, 14, 22}
    assert {int(targets[case_id]["U_mps"]) for case_id in training_linked} == {18}
    assert {targets[case_id]["config"] for case_id in scored} == {
        "P80A310",
        "P90A310",
        "P95A310",
        "P95A320",
    }
    assert sum(float(targets[case_id]["Ay_over_cbar"]) >= 0.01 for case_id in scored) == 10
    assert protocol["claim_boundary"]["h_vawt_promotion"].startswith(
        "The H-VAWT pilot may be submitted only after"
    )


def test_exact_predictions_pass_every_gate(tmp_path: Path) -> None:
    predictions = _write_predictions(tmp_path / "predictions.csv", _prediction_rows())
    result = assess.assess_predictions(predictions)

    assert result["status"] == "passed"
    assert result["pass"] is True
    assert result["claim"] == "passed_multi_condition_turbine_hfm_validation"
    assert result["scored_case_count"] == 16
    assert result["active_viv_case_count"] == 10
    assert result["inactive_case_count"] == 6
    assert all(result["gates"].values())
    assert all(row["pass"] for row in result["row_results"])


@pytest.mark.parametrize(
    ("case_id", "field", "replacement", "failed_gate"),
    [
        pytest.param(
            "P80A310_U14",
            "Ay_pred_over_cbar",
            "0.70",
            "all_active_amplitudes_within_30_percent_plus_digitization",
            id="active-amplitude",
        ),
        pytest.param(
            "P80A310_U22",
            "Ay_pred_over_cbar",
            "0.03",
            "all_inactive_cases_below_0p02_cbar",
            id="inactive-classification",
        ),
        pytest.param(
            "P90A310_U14",
            "fy_pred_over_fn2",
            "0.80",
            "all_active_frequencies_within_0p10_fn",
            id="frequency",
        ),
        pytest.param(
            "P95A320_U22",
            "Ay_pred_over_cbar",
            "0.50",
            "all_configuration_peak_speeds_within_one_interval",
            id="peak-speed",
        ),
    ],
)
def test_each_physical_gate_fails_closed(
    tmp_path: Path,
    case_id: str,
    field: str,
    replacement: str,
    failed_gate: str,
) -> None:
    rows = _prediction_rows()
    for row in rows:
        if row["case_id"] == case_id:
            row[field] = replacement
    predictions = _write_predictions(tmp_path / "predictions.csv", rows)
    result = assess.assess_predictions(predictions)

    assert result["pass"] is False
    assert result["claim"] == "no_turbine_validation_claim"
    assert result["gates"][failed_gate] is False


def test_missing_case_mixed_model_and_training_linked_row_are_rejected(tmp_path: Path) -> None:
    rows = _prediction_rows()
    predictions = _write_predictions(tmp_path / "missing.csv", rows[:-1])
    with pytest.raises(assess.AssessmentError, match="exactly 16"):
        assess.assess_predictions(predictions)

    rows = _prediction_rows()
    rows[0]["model_fingerprint"] = "2" * 64
    predictions = _write_predictions(tmp_path / "mixed.csv", rows)
    with pytest.raises(assess.AssessmentError, match="one frozen model"):
        assess.assess_predictions(predictions)

    rows = _prediction_rows()
    rows[0]["case_id"] = "P80A310_U18"
    predictions = _write_predictions(tmp_path / "training.csv", rows)
    with pytest.raises(assess.AssessmentError, match="exactly the 16"):
        assess.assess_predictions(predictions)


def test_protocol_target_hash_and_role_mutations_are_rejected(tmp_path: Path) -> None:
    protocol_path = tmp_path / "validation_protocol.json"
    shutil.copy2(REFERENCE_ROOT / "validation_protocol.json", protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["reference"]["targets_sha256"] = "0" * 64
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(assess.AssessmentError, match="target hash"):
        assess.load_protocol(protocol_path)

    protocol = json.loads(
        (REFERENCE_ROOT / "validation_protocol.json").read_text(encoding="utf-8")
    )
    protocol["primary_scored_case_ids"][0] = "P80A310_U18"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(assess.AssessmentError, match="overlap|U=18"):
        assess.load_protocol(protocol_path)
