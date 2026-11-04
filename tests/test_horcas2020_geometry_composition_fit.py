from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "examples" / "horcas2020_iea10_tip_geometry"
SPEC = importlib.util.spec_from_file_location("horcas2020_fit", ROOT / "fit_geometry_composition.py")
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit
SPEC.loader.exec_module(fit)


def test_fit_uses_only_the_separate_44_row_training_snapshot() -> None:
    model, predictions = fit.fit_model()
    assert model["training"]["horcas_row_count"] == 44
    assert model["training"]["fu_row_count"] == 20
    assert model["protected_validation"]["comb_rows_used_for_fitting"] is False
    assert len(predictions) == 11
    training_text = fit.TRAINING.read_text(encoding="utf-8")
    assert "COMB_" not in training_text
    assert all(row["case_id"].startswith("COMB_") for row in predictions)


def test_work_margin_composition_is_exact_and_energy_consistent() -> None:
    model, predictions = fit.fit_model()
    saturation = model["parameters"]["grinderslev_signed_power_saturation_tip_amplitude_m"]
    for row in predictions:
        component = [row["ext_work_margin"], row["sweep_work_margin"], row["dihe_work_margin"]]
        expected_margin = sum(component) / 3.0
        assert row["combined_normalized_positive_work_margin"] == pytest.approx(expected_margin, abs=1e-15)
        assert row["predicted_peak_edge_amplitude_m"] == pytest.approx(saturation * math.sqrt(expected_margin), abs=1e-15)
        assert row["zero_amplitude_feedback_over_structural_damping"] == pytest.approx(1.0 / (1.0 - expected_margin), abs=1e-15)
    assert model["method"]["stationary_spectrum_phase_used"] is False


def test_written_model_and_predictions_reproduce(tmp_path: Path) -> None:
    model_path, prediction_path = fit.write_frozen_model(tmp_path)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    with prediction_path.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    assert model["status"] == "frozen_before_horcas_combined_geometry_evaluation"
    assert len(model["model_fingerprint"]) == 64
    assert len(predictions) == 11
    assert {row["model_fingerprint"] for row in predictions} == {model["model_fingerprint"]}


def test_out_of_range_amplitude_and_source_drift_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(fit.FitError, match="outside the signed-saturation branch"):
        fit._power_margin(fit.SATURATION_M, fit.SATURATION_M)
    monkeypatch.setattr(fit, "TRAINING_SHA", "0" * 64)
    with pytest.raises(fit.FitError, match="Training/source identity changed"):
        fit.fit_model()
