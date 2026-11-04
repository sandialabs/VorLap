from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import pytest

from examples.horcas2020_iea10_tip_geometry import reproduce_unified_gain
from vorlap.unified_viv_gain import (
    AIRFOIL_BRANCH,
    OPPOSING_DAMPING,
    SPECTRAL_BASIS,
    TURBINE_BRANCH,
    UnifiedVIVGainError,
    UnifiedVIVGainModel,
)


REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / "run_headless/config/unified_viv_gain_model_v1.json"
TURBINE_MODEL_PATH = (
    REPO
    / "examples/horcas2020_iea10_tip_geometry/evidence/combined_tip_v1/model.json"
)
SIGNED_WORK_MODEL_PATH = (
    REPO / "examples/fu2026_iea10/evidence/signed_work_shape_v1/model.json"
)


def _model() -> UnifiedVIVGainModel:
    return UnifiedVIVGainModel.from_json(MODEL_PATH, repository_root=REPO)


def test_airfoil_anchor_and_common_c_a0_conversion_are_exact() -> None:
    state = _model().evaluate_airfoil(
        profile="NACA0018",
        kinematic_family="quasi_static_azimuth",
    )
    assert state.branch == AIRFOIL_BRANCH
    assert state.gain_basis == SPECTRAL_BASIS
    assert state.feedback_gain == 143.1939823335363
    assert state.limiting_amplitude_ratio == 0.020343253217205814
    assert state.zero_amplitude_feedback(spectral_basis=2.5) == (
        143.1939823335363 * 2.5
    )
    assert state.static_spectrum_sha256 == (
        "64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369"
    )


def test_all_44_primitive_turbine_anchors_preserve_margin_ratio_and_amplitude() -> None:
    evaluator = _model()
    frozen = json.loads(TURBINE_MODEL_PATH.read_text(encoding="utf-8"))
    rows_by_geometry = frozen["training"]["horcas2020_primitive_geometry_rows"]
    for geometry, rows in rows_by_geometry.items():
        for row in rows:
            state = evaluator.evaluate_turbine_geometry(
                profile_family="FFA-W3_family",
                geometry_family=geometry,
                inclination_deg=row["inclination_deg"],
            )
            assert state.branch == TURBINE_BRANCH
            assert state.gain_basis == OPPOSING_DAMPING
            assert state.work_margin == row["normalized_positive_work_margin"]
            assert (
                state.zero_amplitude_gain
                == row["zero_amplitude_feedback_over_structural_damping"]
            )
            assert state.zero_amplitude_feedback(opposing_damping=3.0) == (
                state.zero_amplitude_gain * 3.0
            )
            # Reversing A -> mu -> A can incur one binary64 rounding step.
            assert state.stable_equilibrium_amplitude_m() == pytest.approx(
                row["stable_peak_amplitude_m"], rel=0.0, abs=2.5e-16
            )


def test_all_20_fu_training_anchors_preserve_original_gain_states() -> None:
    evaluator = _model()
    frozen = json.loads(TURBINE_MODEL_PATH.read_text(encoding="utf-8"))
    rows = frozen["training"][
        "fu2026_speed_orientation_rows_after_preserved_failure"
    ]
    assert len(rows) == 20
    for row in rows:
        state = evaluator.evaluate_turbine_training_condition(
            profile_family="FFA-W3_family",
            configuration=row["config"],
            wind_speed_mps=row["wind_speed_mps"],
        )
        assert state.work_margin == row["normalized_positive_work_margin"]
        assert (
            state.zero_amplitude_gain
            == row["zero_amplitude_feedback_over_structural_damping"]
        )
        assert (
            state.reference_length_m
            == row["configuration_saturation_amplitude_m"]
        )
        assert state.stable_equilibrium_amplitude_m() == pytest.approx(
            row["stable_peak_amplitude_m"], rel=0.0, abs=2.5e-16
        )


def test_all_20_fu_signed_work_anchors_preserve_active_inactive_transfer() -> None:
    evaluator = _model()
    frozen = json.loads(SIGNED_WORK_MODEL_PATH.read_text(encoding="utf-8"))
    assert len(frozen["cases"]) == 20
    states = []
    for row in frozen["cases"]:
        state = evaluator.evaluate_turbine_signed_work_condition(
            profile_family="FFA-W3_family",
            configuration=row["configuration"],
            wind_speed_mps=row["wind_speed_mps"],
        )
        states.append(state)
        assert state.active is row["active_branch"]
        assert (
            state.zero_amplitude_gain
            == row["transferred_feedback_over_opposing_damping"]
        )
        assert (
            state.reference_length_m
            == row["transferred_limiting_amplitude_m"]
        )
        if state.active:
            assert state.stable_equilibrium_amplitude_m() == pytest.approx(
                row["equilibrium_amplitude_m"], rel=0.0, abs=2.5e-16
            )
        else:
            assert state.work_margin is None
            assert state.zero_amplitude_gain == 0.0
    assert sum(state.active is True for state in states) == 12
    assert sum(state.active is False for state in states) == 8


def test_unified_replay_is_exact_for_all_11_comb_predictions_and_metrics() -> None:
    result = reproduce_unified_gain.reproduce()
    assert result["status"] == "passed"
    assert result["prediction_parity"] == {
        "comparison": "exact_binary64_after_decimal_parse",
        "all_11_rows_exact": True,
        "row_count": 11,
    }
    assert result["gates"]["all"] is True
    assert result["metrics"]["curve_nrmse_by_target_peak"] == (
        0.12922581236076258
    )
    assert result["metrics"]["maximum_resolved_frequency_ratio_error"] == (
        0.039160768999999984
    )


def test_schema_source_and_domain_fail_closed(tmp_path: Path) -> None:
    raw = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    raw["claim_boundary"]["hvawt_experimentally_validated"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(UnifiedVIVGainError, match="fingerprint changed"):
        UnifiedVIVGainModel.from_json(changed, repository_root=REPO)

    evaluator = _model()
    with pytest.raises(UnifiedVIVGainError, match="Profile is outside"):
        evaluator.evaluate_airfoil(
            profile="NACA0021",
            kinematic_family="quasi_static_azimuth",
        )
    with pytest.raises(UnifiedVIVGainError, match="anchor grid"):
        evaluator.evaluate_turbine_geometry(
            profile_family="FFA-W3_family",
            geometry_family="COMB",
            inclination_deg=56.0,
        )

    staged = tmp_path / "staged"
    staged.mkdir()
    shutil.copy2(MODEL_PATH, staged / MODEL_PATH.name)
    calibration = REPO / "run_headless/config/geometry_blended_airfoil_calibration.json"
    staged_calibration = staged / calibration.name
    shutil.copy2(calibration, staged_calibration)
    staged_calibration.write_bytes(staged_calibration.read_bytes() + b"\n")
    staged_model = UnifiedVIVGainModel.from_json(
        staged / MODEL_PATH.name,
        repository_root=REPO,
    )
    with pytest.raises(UnifiedVIVGainError, match="SHA-256 changed"):
        staged_model.evaluate_airfoil(
            profile="NACA0018",
            kinematic_family="quasi_static_azimuth",
        )


def test_quasi_static_rotational_velocity_ratios_are_bounded_at_planned_winds() -> None:
    radius_m = 5.27
    omega_radps = 2.0 * math.pi / 60.0
    tip_speed_mps = omega_radps * radius_m
    assert tip_speed_mps == pytest.approx(0.551873109480607, rel=0.0, abs=1.0e-15)
    assert tip_speed_mps / 8.25 == pytest.approx(
        0.06689371024007357, rel=0.0, abs=1.0e-15
    )
    assert tip_speed_mps / 12.2 == pytest.approx(
        0.04523550077709893, rel=0.0, abs=1.0e-15
    )
    assert max(tip_speed_mps / wind for wind in (8.25, 12.2)) < 0.07
