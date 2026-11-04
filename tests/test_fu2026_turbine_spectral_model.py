from __future__ import annotations

from pathlib import Path

import pytest

from examples.fu2026_iea10 import turbine_spectral_model as model


def test_fit_is_training_only_and_reconstructs_all_four_training_rows() -> None:
    frozen = model.fit_model()
    assert frozen["schema"] == model.MODEL_SCHEMA
    assert frozen["protected_validation"]["fu_non18_response_values_used_for_fit"] is False
    assert len(frozen["training"]) == 4
    for row in frozen["training"]:
        assert row["reconstructed_amplitude_m"] == pytest.approx(
            row["edgewise_peak_amplitude_m"], rel=1.0e-12, abs=1.0e-12
        )
        assert row["coefficient_kgps_per_unit_gain"] > 0.0
        assert row["motion_correlated_gain"] > 0.0


def test_prediction_uses_exact_16_case_ids_and_one_fingerprint(tmp_path: Path) -> None:
    frozen = model.fit_model()
    model_path = tmp_path / "model.json"
    model._write_json(model_path, frozen)
    predictions = model.predict_model(model_path)
    assert len(predictions) == 16
    assert len({row["case_id"] for row in predictions}) == 16
    assert {row["model_fingerprint"] for row in predictions} == {
        frozen["model_fingerprint"]
    }
    assert all(row["Ay_pred_over_cbar"] >= 0.0 for row in predictions)
    assert all(row["fy_pred_over_fn2"] in (0.0, 1.0) for row in predictions)


def test_model_tamper_is_rejected(tmp_path: Path) -> None:
    frozen = model.fit_model()
    frozen["parameters"]["saturation_tip_amplitude_m"] += 0.1
    path = tmp_path / "tampered.json"
    model._write_json(path, frozen)
    with pytest.raises(model.TurbineModelError, match="fingerprint"):
        model.predict_model(path)


def test_prediction_does_not_load_fu_response_targets(monkeypatch, tmp_path: Path) -> None:
    frozen = model.fit_model()
    path = tmp_path / "model.json"
    model._write_json(path, frozen)

    original_read_text = Path.read_text

    def guarded_read_text(self: Path, *args, **kwargs):
        assert self.name != "targets.csv"
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    assert len(model.predict_model(path)) == 16
