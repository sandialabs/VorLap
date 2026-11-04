from __future__ import annotations

import csv
import importlib
from pathlib import Path

import pytest

assessment = importlib.import_module("examples.horcas2020_iea10_tip_geometry.assess_geometry_composition")


def test_frozen_combined_geometry_passes_every_preregistered_gate() -> None:
    result = assessment.assess()
    assert result["status"] == "passed"
    assert result["claim"] == "passed_multi_condition_turbine_hfm_geometry_validation"
    assert result["matrix"] == {
        "training_geometry_count": 4,
        "training_case_count": 44,
        "heldout_geometry": "COMB",
        "heldout_case_count": 11,
        "all_holdout_rows_reported": True,
    }
    assert all(result["gates"].values())
    assert len(result["rows"]) == 11


def test_metrics_are_quantified_over_the_complete_curve() -> None:
    metrics = assessment.assess()["metrics"]
    assert metrics["curve_nrmse_by_target_peak"] == pytest.approx(0.1292258126, rel=1e-8)
    assert metrics["peak_amplitude_relative_error"] == pytest.approx(0.0283676017, rel=1e-8)
    assert metrics["target_peak_inclination_deg"] == 55.0
    assert metrics["predicted_peak_inclination_deg"] == 55.0
    assert metrics["target_resolved_interval"]["inclinations_deg"] == [55.0, 57.5]
    assert metrics["predicted_resolved_interval"]["inclinations_deg"] == [52.5, 55.0, 57.5]
    assert metrics["maximum_resolved_frequency_ratio_error"] < 0.04


def test_post_validation_envelope_is_conservative_but_not_promoted_to_fatigue_validation() -> None:
    result = assessment.assess()
    envelope = result["conservative_engineering_envelope"]
    assert envelope["covers_all_11_heldout_rows"] is True
    assert envelope["multiplicative_factor"] == 1.30
    assert envelope["additive_floor_m"] > 0.0
    assert result["claim_boundary"]["does_not_support"].find("fatigue life") >= 0


def test_prediction_identity_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assessment, "PREDICTIONS_SHA", "0" * 64)
    with pytest.raises(assessment.AssessmentError, match="predictions SHA-256 mismatch"):
        assessment.assess()


def test_figure_is_generated_from_all_11_rows(tmp_path: Path) -> None:
    result = assessment.assess()
    path = assessment.write_figure(result, tmp_path / "validation.pdf")
    assert path.stat().st_size > 10_000
    with assessment.PREDICTIONS_PATH.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 11
