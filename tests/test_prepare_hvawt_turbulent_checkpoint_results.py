from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "paper_drafts/NAWEA_QBladeBridge/analysis/prepare_hvawt_turbulent_checkpoint_results_v2.py"
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vorlap-turbulent-checkpoint-test-mpl")
SPEC = importlib.util.spec_from_file_location(
    "prepare_hvawt_turbulent_checkpoint_results_v2", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion)


def test_load_descriptor_accepts_only_expected_signed_blade_channels():
    assert promotion._load_descriptor(
        "X_l~Mom.~BLD_1~pos~0.230~[Nm]"
    ) == {
        "axis": "X",
        "quantity": "moment",
        "position": 0.23,
        "unit": "Nm",
    }
    assert promotion._load_descriptor("Y_l~For.~BLD_1~pos~0.500~[N]") == {
        "axis": "Y",
        "quantity": "force",
        "position": 0.5,
        "unit": "N",
    }
    assert promotion._load_descriptor("Time~[s]") is None
    assert promotion._load_descriptor("X_l~Mom.~BLD_2~pos~0.230~[Nm]") is None


def test_del_reduction_uses_declared_fixed_equivalent_cycle_count():
    damage = 600.0 * 2.5**10
    assert promotion._del_from_damage(damage, 10.0, 600.0) == pytest.approx(2.5)
    assert promotion._del_from_damage(0.0, 10.0, 600.0) == 0.0
    with pytest.raises(promotion.CheckpointPromotionError):
        promotion._del_from_damage(-1.0, 10.0, 600.0)


def test_headline_selection_separates_ratio_and_absolute_scale():
    rows = [
        {
            "row_id": "ratio_force",
            "wind_speed_mps": 11.25,
            "channel": "Y force",
            "axis": "Y",
            "quantity": "force",
            "position": 0.23,
            "unit": "N",
            "wohler_exponent": 10.0,
            "raw_del": 10.0,
            "complete_del": 100.0,
            "del_ratio": 10.0,
        },
        {
            "row_id": "ratio_moment",
            "wind_speed_mps": 11.25,
            "channel": "X moment",
            "axis": "X",
            "quantity": "moment",
            "position": 0.23,
            "unit": "Nm",
            "wohler_exponent": 10.0,
            "raw_del": 40.0,
            "complete_del": 320.0,
            "del_ratio": 8.0,
        },
        {
            "row_id": "absolute_moment",
            "wind_speed_mps": 35.0,
            "channel": "Y moment",
            "axis": "Y",
            "quantity": "moment",
            "position": 0.23,
            "unit": "Nm",
            "wohler_exponent": 10.0,
            "raw_del": 3700.0,
            "complete_del": 4200.0,
            "del_ratio": 4200.0 / 3700.0,
        },
    ]
    selected = promotion._select_headline_rows(pd.DataFrame(rows))
    assert selected["all_channel_ratio"]["row_id"] == "ratio_force"
    assert selected["primary_ratio"]["row_id"] == "ratio_moment"
    assert selected["largest_complete_del"]["row_id"] == "absolute_moment"


def test_hash_bound_identity_rejects_changed_file(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("reviewed\n", encoding="utf-8")
    record = promotion._identity(artifact, base=tmp_path)
    assert promotion._verify_identity(tmp_path, record, label="artifact") == artifact
    artifact.write_text("changed\n", encoding="utf-8")
    with pytest.raises(promotion.CheckpointPromotionError, match="Byte count changed|SHA-256 changed"):
        promotion._verify_identity(tmp_path, record, label="artifact")


def _synthetic_historical_context_rows():
    current_rows = []
    aggregate_rows = []
    same_seed_rows = []
    for wind_index, wind_speed in enumerate(promotion.EXPECTED_WINDS, start=1):
        for axis_index, axis in enumerate(("X", "Y", "Z"), start=1):
            baseline = 10.0 * wind_index + axis_index
            current_rows.append(
                {
                    "wind_speed_mps": wind_speed,
                    "axis": axis,
                    "quantity": "moment",
                    "position": 0.23,
                    "wohler_exponent": 10.0,
                    "raw_del": 0.9 * baseline,
                    "complete_del": 1.8 * baseline,
                    "del_ratio": 2.0,
                }
            )
            aggregate_rows.append(
                {
                    "position": 0.23,
                    "quantity": "moment",
                    "axis": f"{axis}_l",
                    "wind_speed_mps": wind_speed,
                    "seed_count": 3,
                    "baseline_del_seed_aggregate": baseline,
                    "vorlap_del_seed_aggregate": 1.25 * baseline,
                    "del_ratio": 1.25,
                    "paired_ratio_seed_min": 1.2,
                    "paired_ratio_seed_max": 1.3,
                    "paired_ratio_seed_sample_std": 0.05,
                }
            )
            same_seed_rows.append(
                {
                    "seed": promotion.CHECKPOINT_SEED,
                    "position": 0.23,
                    "quantity": "moment",
                    "axis": axis,
                    "wind_speed_mps": wind_speed,
                    "signed_del_baseline": baseline,
                    "signed_del_vorlap": 1.2 * baseline,
                    "equivalent_cycles_baseline": promotion.EQUIVALENT_CYCLES,
                    "equivalent_cycles_vorlap": promotion.EQUIVALENT_CYCLES,
                    "smoke_only": False,
                    "postprocess_warnings": "",
                }
            )
    return (
        pd.DataFrame(current_rows),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(same_seed_rows),
    )


def test_historical_context_keeps_current_and_historical_pairings_distinct():
    current, aggregate, same_seed = _synthetic_historical_context_rows()
    result = promotion._merge_historical_context(current, aggregate, same_seed)
    assert len(result) == len(promotion.EXPECTED_WINDS) * 3
    assert result["current_complete_ratio"].eq(2.0).all()
    assert result["historical_one_way_ratio"].eq(1.25).all()
    assert result["historical_same_seed_one_way_ratio"].eq(1.2).all()
    assert result["current_raw_over_historical_baseline"].to_numpy() == pytest.approx(
        0.9
    )
    assert "current_effect_over_historical_effect" not in result


def test_historical_context_rejects_missing_same_seed_coverage():
    current, aggregate, same_seed = _synthetic_historical_context_rows()
    with pytest.raises(
        promotion.CheckpointPromotionError,
        match="same-seed rows do not satisfy",
    ):
        promotion._merge_historical_context(
            current,
            aggregate,
            same_seed.iloc[:-1].copy(),
        )


def test_contextual_weighting_adds_hybrid_on_one_common_wind_basis(tmp_path):
    current, aggregate, same_seed = _synthetic_historical_context_rows()
    context = promotion._merge_historical_context(current, aggregate, same_seed)
    weighting = pd.DataFrame(
        {
            "wind_speed_mps": promotion.HISTORICAL_WINDS,
            "fatigue_weight": [
                float(index) for index in range(1, len(promotion.HISTORICAL_WINDS) + 1)
            ],
        }
    )
    result = promotion._compute_contextual_weighting(context, weighting)

    assert len(result) == len(promotion.EXPECTED_WINDS) * 3
    assert set(result["path"]) == {
        "standalone_qblade",
        "vorlap_one_way",
        "complete_hybrid",
    }
    assert 19.25 not in set(result["wind_speed_mps"])
    assert result.groupby("path")["pseudo_damage_fraction"].sum().to_numpy() == (
        pytest.approx([1.0, 1.0, 1.0], abs=1e-12)
    )
    assert set(result["evidence_basis"]) == {
        "current checkpoint, seed 4433456",
        "historical campaign, seed 4433456",
    }

    figure = tmp_path / "weighting.pdf"
    promotion._plot_contextual_weighting(result, figure)
    assert figure.is_file()
    assert figure.stat().st_size > 0


def test_contextual_weighting_rejects_missing_common_wind():
    current, aggregate, same_seed = _synthetic_historical_context_rows()
    context = promotion._merge_historical_context(current, aggregate, same_seed)
    weighting = pd.DataFrame(
        {
            "wind_speed_mps": promotion.HISTORICAL_WINDS[:-1],
            "fatigue_weight": [1.0] * (len(promotion.HISTORICAL_WINDS) - 1),
        }
    )
    with pytest.raises(
        promotion.CheckpointPromotionError,
        match="Common-wind weighting context is incomplete",
    ):
        promotion._compute_contextual_weighting(context, weighting)
