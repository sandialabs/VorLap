from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_PATH = ROOT / "tests/test_prepare_unified_hvawt_paper_results_v2.py"
SCRIPT = (
    ROOT
    / "paper_drafts/NAWEA_QBladeBridge/analysis/select_hvawt_sensitivity_rows.py"
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vorlap-sensitivity-selection-test-mpl")

SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "_vorlap_test_recovered_promotion_support", SUPPORT_PATH
)
assert SUPPORT_SPEC is not None and SUPPORT_SPEC.loader is not None
support = importlib.util.module_from_spec(SUPPORT_SPEC)
SUPPORT_SPEC.loader.exec_module(support)

SPEC = importlib.util.spec_from_file_location(
    "select_hvawt_sensitivity_rows", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(payload: dict) -> str:
    content = dict(payload)
    content.pop("fingerprint", None)
    return selection.canonical_fingerprint(content)


def _campaign_with_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    support._write_recovered_fixture(campaign, monkeypatch)
    selection.promoter.promote(
        campaign, campaign / selection.PAPER_EXPORT_NAME
    )
    return campaign


def test_freezes_two_distinct_rows_before_any_sensitivity_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign_with_promotion(tmp_path, monkeypatch)
    target = campaign / selection.SELECTION_RECEIPT_NAME

    preflight = selection.build_selection_receipt(campaign)

    assert not target.exists()
    assert preflight["status"] == selection.SELECTION_STATUS
    assert preflight["candidate_physical_row_count"] == 33
    assert preflight["candidate_channel_row_count"] == 99
    assert preflight["case_generation_performed"] is False
    assert preflight["simulation_execution_performed"] is False
    assert "not fatigue life" in preflight["selection_rule"]["metric_scope"]
    selected = preflight["selected_physical_rows"]
    assert len(selected) == 2
    assert len({row["row_id"] for row in selected}) == 2
    assert selected[0]["selection_role"].startswith("governing_complete")
    assert selected[0]["row_id"] == "dlc6p4_U35p00ms_s4433456"
    assert selected[0]["channel"]["axis"] == "Z"
    assert selected[1]["row_id"] == "dlc6p4_U35p00ms_s4433457"
    assert selected[1]["channel"]["axis"] == "Z"
    assert all("dlc6p4" in row["row_id"] for row in selected)
    assert all(len(row["case_indices"]) == 3 for row in selected)

    written = selection.write_selection_receipt(campaign)
    first_bytes = target.read_bytes()
    repeated = selection.write_selection_receipt(campaign)
    verified = selection.verify_selection_receipt(campaign)

    assert written == repeated == verified
    assert target.read_bytes() == first_bytes
    assert written["fingerprint"] == _fingerprint(written)
    assert written["selection_source"]["sha256"] == _sha(SCRIPT)


def test_selection_rejects_changed_promotion_output_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign_with_promotion(tmp_path, monkeypatch)
    path = (
        campaign
        / selection.PAPER_EXPORT_NAME
        / selection.PRIMARY_REALIZATION_NAME
    )
    path.write_text(path.read_text() + "\n")

    with pytest.raises(
        selection.HVAWTSensitivitySelectionError, match="identity changed"
    ):
        selection.build_selection_receipt(campaign)


def test_selection_cli_preflight_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    campaign = _campaign_with_promotion(tmp_path, monkeypatch)
    target = campaign / selection.SELECTION_RECEIPT_NAME

    return_code = selection.main(
        ["preflight", "--campaign-root", str(campaign)]
    )
    output = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert output["status"] == "ready_to_freeze_two_physical_rows"
    assert output["writes_performed"] == 0
    assert output["case_generation_performed"] is False
    assert output["simulation_execution_performed"] is False
    assert len(output["selected_physical_rows"]) == 2
    assert not target.exists()


def test_selection_rejects_internally_inconsistent_realization_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign_with_promotion(tmp_path, monkeypatch)
    output = campaign / selection.PAPER_EXPORT_NAME
    path = output / selection.PRIMARY_REALIZATION_NAME
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["complete_to_one_way_ratio"] = str(
        float(rows[0]["complete_to_one_way_ratio"]) + 0.1
    )
    support.fixture_support._write_csv(path, rows)
    receipt_path = output / selection.PAPER_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"]["primary_realizations"] = selection.promoter._identity(path)
    receipt.pop("fingerprint")
    receipt["fingerprint"] = selection.canonical_fingerprint(receipt)
    support._write_json(receipt_path, receipt)

    with pytest.raises(
        selection.HVAWTSensitivitySelectionError, match="inconsistent derived"
    ):
        selection.build_selection_receipt(campaign)


def test_selection_rejects_noninteger_seed_even_when_truncation_would_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign_with_promotion(tmp_path, monkeypatch)
    output = campaign / selection.PAPER_EXPORT_NAME
    path = output / selection.PRIMARY_REALIZATION_NAME
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["seed"] = str(float(rows[0]["seed"]) + 0.5)
    support.fixture_support._write_csv(path, rows)
    receipt_path = output / selection.PAPER_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"]["primary_realizations"] = selection.promoter._identity(path)
    receipt.pop("fingerprint")
    receipt["fingerprint"] = selection.canonical_fingerprint(receipt)
    support._write_json(receipt_path, receipt)

    with pytest.raises(
        selection.HVAWTSensitivitySelectionError, match="seed is not an integer"
    ):
        selection.build_selection_receipt(campaign)


def _numeric_row(
    row_id: str,
    *,
    complete_term: float,
    one_way_term: float,
    complete_value: float,
    one_way_value: float,
    wind: float,
    seed: int,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "axis": "X",
        "wind_speed_mps": wind,
        "seed": seed,
        "fatigue_weight": 1.0,
        "values": {
            "raw": 1.0,
            "one_way": one_way_value,
            "complete": complete_value,
        },
        "ratios": {
            "one_way_to_raw": one_way_value,
            "complete_to_raw": complete_value,
            "complete_to_one_way": complete_value / one_way_value,
        },
        "terms": {
            "raw": 1.0,
            "one_way": one_way_term,
            "complete": complete_term,
        },
    }


def test_ratio_selection_ignores_large_but_immaterial_ratio() -> None:
    rows = [
        _numeric_row(
            "governing",
            complete_term=90.0,
            one_way_term=90.0,
            complete_value=10.0,
            one_way_value=10.0,
            wind=35.0,
            seed=1,
        ),
        _numeric_row(
            "tiny_huge_ratio",
            complete_term=1.0e-8,
            one_way_term=1.0e-20,
            complete_value=1.0,
            one_way_value=1.0e-6,
            wind=3.0,
            seed=2,
        ),
        _numeric_row(
            "material_ratio",
            complete_term=10.0,
            one_way_term=5.0,
            complete_value=4.0,
            one_way_value=2.0,
            wind=11.25,
            seed=3,
        ),
    ]

    result = selection.select_rows(rows, {"X": {"complete": 10.0}})

    assert result["governing_weighted_contributor"]["row_id"] == "governing"
    assert result["largest_meaningful_composition_ratio"]["row_id"] == (
        "material_ratio"
    )


def test_selection_fails_without_a_distinct_material_ratio_row() -> None:
    rows = [
        _numeric_row(
            "governing",
            complete_term=100.0,
            one_way_term=100.0,
            complete_value=10.0,
            one_way_value=10.0,
            wind=35.0,
            seed=1,
        ),
        _numeric_row(
            "immaterial",
            complete_term=1.0e-12,
            one_way_term=1.0e-12,
            complete_value=1.0,
            one_way_value=0.1,
            wind=3.0,
            seed=2,
        ),
    ]

    with pytest.raises(
        selection.HVAWTSensitivitySelectionError, match="No distinct material"
    ):
        selection.select_rows(rows, {"X": {"complete": 10.0}})
