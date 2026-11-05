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
OLD_TEST = ROOT / "tests/test_prepare_unified_hvawt_paper_results.py"
SCRIPT = (
    ROOT
    / "paper_drafts/NAWEA_QBladeBridge/analysis/prepare_unified_hvawt_paper_results_v2.py"
)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vorlap-recovered-paper-test-mpl")

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_vorlap_test_unified_hvawt_paper_results_v1", OLD_TEST
)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
fixture_support = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixture_support)

SPEC = importlib.util.spec_from_file_location(
    "prepare_unified_hvawt_paper_results_v2", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion)


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_identity(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_recovered_fixture(
    campaign: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict, dict]:
    fixture_support._write_fixture(campaign)
    manifest_path = campaign / "campaign_manifest.json"
    original_path = campaign / promotion.recovery.ORIGINAL_ANALYSIS_NAME
    manifest = json.loads(manifest_path.read_text())
    helper = promotion.LEGACY_PROMOTER_PATH
    manifest["inputs"] = {
        "orchestration_sources": {
            "paper_result_promotion": {
                "path": helper.relative_to(ROOT).as_posix(),
                "sha256": _sha(helper),
                "size_bytes": helper.stat().st_size,
            }
        }
    }
    manifest.pop("fingerprint")
    manifest["fingerprint"] = _fingerprint(manifest)
    _write_json(manifest_path, manifest)

    original = json.loads(original_path.read_text())
    original["campaign_fingerprint"] = manifest["fingerprint"]
    _write_json(original_path, original)

    receipt_path = campaign / promotion.recovery.RECOVERY_RECEIPT_NAME
    receipt = {
        "schema": promotion.recovery.RECOVERY_SCHEMA,
        "status": promotion.recovery.RECOVERY_STATUS,
        "campaign_fingerprint": manifest["fingerprint"],
        "original_analysis": _relative_identity(original_path, campaign),
        "original_complete_case_count": 68,
        "recovered_case_count": 34,
        "accepted_case_count": 102,
        "recovered_case_indices": list(promotion.recovery.EXPECTED_ONE_WAY_INDICES),
        "simulation_rerun_count": 0,
        "modified_artifact_count": 0,
        "comparator_semantics": {
            "one_way_label": "closure-off legacy spectral comparator",
            "explicit_unified_response_closure_active": False,
            "strict_open_loop_one_way": False,
            "kinematic_relative_velocity_feedback_retained": True,
            "one_way_to_complete": (
                "change in force composition and partition including "
                "response-correlated work; not isolated additive negative damping"
            ),
        },
        "claim_boundary": (
            "Artifact recovery only; engineering application, not experimental "
            "validation, fatigue life, or certification evidence."
        ),
    }
    receipt["fingerprint"] = _fingerprint(receipt)
    _write_json(receipt_path, receipt)

    cases = []
    for index, case in enumerate(manifest["cases"]):
        cases.append(
            {
                "case_index": index,
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "scenario": case["scenario"],
                "status": "complete",
                "audit_disposition": (
                    "recovered_exact_inert_response_sentinel"
                    if case["scenario"] == promotion.sealed.SCENARIOS[1]
                    else "unchanged_frozen_terminal_audit"
                ),
            }
        )
    analysis = {
        "schema": promotion.EXPECTED_ANALYSIS_SCHEMA,
        "status": promotion.recovery.RECOVERED_ANALYSIS_STATUS,
        "campaign_fingerprint": manifest["fingerprint"],
        "expected_case_count": 102,
        "complete_case_count": 102,
        "all_three_scenarios_complete": True,
        "original_analysis": receipt["original_analysis"],
        "recovery_receipt": _relative_identity(receipt_path, campaign),
        "recovered_case_count": 34,
        "recovered_case_indices": list(promotion.recovery.EXPECTED_ONE_WAY_INDICES),
        "cases": cases,
        "claim_boundary": receipt["claim_boundary"],
    }
    analysis["fingerprint"] = _fingerprint(analysis)
    _write_json(campaign / promotion.recovery.RECOVERED_ANALYSIS_NAME, analysis)

    def load_fixture_receipt(*_args, **_kwargs):
        live = json.loads(receipt_path.read_text())
        content = dict(live)
        recorded = content.pop("fingerprint", None)
        if recorded != _fingerprint(content):
            raise promotion.recovery.HVAWTOneWayRecoveryError(
                "One-way recovery receipt fingerprint is invalid."
            )
        return live

    monkeypatch.setattr(
        promotion.recovery, "load_recovery_receipt", load_fixture_receipt
    )
    return manifest, receipt, analysis


def _rewrite_linked_campaign_records(campaign: Path, manifest: dict) -> None:
    """Refresh the additive fixture after a manifest-only provenance change."""

    manifest_path = campaign / "campaign_manifest.json"
    original_path = campaign / promotion.recovery.ORIGINAL_ANALYSIS_NAME
    receipt_path = campaign / promotion.recovery.RECOVERY_RECEIPT_NAME
    analysis_path = campaign / promotion.recovery.RECOVERED_ANALYSIS_NAME
    manifest.pop("fingerprint", None)
    manifest["fingerprint"] = _fingerprint(manifest)
    _write_json(manifest_path, manifest)

    original = json.loads(original_path.read_text())
    original["campaign_fingerprint"] = manifest["fingerprint"]
    _write_json(original_path, original)

    receipt = json.loads(receipt_path.read_text())
    receipt["campaign_fingerprint"] = manifest["fingerprint"]
    receipt["original_analysis"] = _relative_identity(original_path, campaign)
    receipt.pop("fingerprint", None)
    receipt["fingerprint"] = _fingerprint(receipt)
    _write_json(receipt_path, receipt)

    analysis = json.loads(analysis_path.read_text())
    analysis["campaign_fingerprint"] = manifest["fingerprint"]
    analysis["original_analysis"] = receipt["original_analysis"]
    analysis["recovery_receipt"] = _relative_identity(receipt_path, campaign)
    analysis.pop("fingerprint", None)
    analysis["fingerprint"] = _fingerprint(analysis)
    _write_json(analysis_path, analysis)


def test_recovered_preflight_accepts_only_v3_and_restores_helper_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    manifest, receipt, analysis = _write_recovered_fixture(campaign, monkeypatch)
    original_loader = promotion.legacy._load_json
    original_schema = promotion.legacy.ANALYSIS_SCHEMA

    report = promotion.preflight(campaign)

    assert report["status"] == "ready_for_three_full_record_postprocessors"
    assert report["campaign_fingerprint"] == manifest["fingerprint"]
    assert report["analysis_fingerprint"] == analysis["fingerprint"]
    assert report["recovery_fingerprint"] == receipt["fingerprint"]
    assert report["physical_row_count"] == 34
    assert report["complete_case_count"] == 102
    assert report["writes_performed"] == 0
    assert report["source_comparison"]["sha256_matches_manifest"] is True
    assert promotion.legacy._load_json is original_loader
    assert promotion.legacy.ANALYSIS_SCHEMA == original_schema


def test_recovered_promotion_binds_inputs_outputs_and_evidence_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _manifest, receipt, analysis = _write_recovered_fixture(campaign, monkeypatch)
    output = tmp_path / "paper_export_recovered_v2"

    result = promotion.promote(campaign, output)

    assert result["status"] == "promoted_from_recovered_102_complete_matched_cases"
    summary = json.loads(
        (output / "hvawt_three_path_results_summary.json").read_text()
    )
    assert summary["schema"] == promotion.PAPER_SCHEMA
    assert summary["case_count"] == 102
    assert summary["analysis_fingerprint"] == analysis["fingerprint"]
    assert summary["recovery_fingerprint"] == receipt["fingerprint"]
    assert summary["comparator_semantics"]["strict_open_loop_one_way"] is False
    assert summary["primary_realization_row_count"] == 99
    assert summary["wind_bin_contribution_row_count"] == 33
    assert "not fatigue life" in summary["contribution_metric_scope"]
    realizations = list(
        csv.DictReader(
            (output / "hvawt_three_path_primary_realizations.csv").open()
        )
    )
    assert len(realizations) == 99
    assert {row["axis"] for row in realizations} == {"X", "Y", "Z"}
    assert all(
        row["metric_scope"] == "conditional_w_i_L_i_power_m_not_fatigue_life"
        for row in realizations
    )
    for speed in promotion.legacy.EXPECTED_SPEEDS:
        for axis in ("X", "Y", "Z"):
            selected = [
                row
                for row in realizations
                if float(row["wind_speed_mps"]) == speed and row["axis"] == axis
            ]
            assert len(selected) == 3
            assert len({row["seed"] for row in selected}) == 3
    contributions = list(
        csv.DictReader(
            (output / "hvawt_three_path_wind_bin_contributions.csv").open()
        )
    )
    assert len(contributions) == 33
    for axis in ("X", "Y", "Z"):
        for scenario in promotion.legacy.SCENARIOS:
            assert sum(
                float(row[f"{scenario}_pseudo_damage_fraction"])
                for row in contributions
                if row["axis"] == axis
            ) == pytest.approx(1.0, abs=1.0e-12)
    fragment = (output / "hvawt_matched_three_path_results.tex").read_text()
    assert "structural-motion-induced relative" in fragment
    assert "not an isolated response-force or negative-damping ablation" in fragment
    assert "fixed-body spectral forcing" not in fragment

    promoted = json.loads(
        (output / "hvawt_three_path_promotion_receipt.json").read_text()
    )
    assert promoted["schema"] == promotion.PROMOTION_RECEIPT_SCHEMA
    assert promoted["analysis_schema"] == promotion.EXPECTED_ANALYSIS_SCHEMA
    assert promoted["recovery_schema"] == promotion.recovery.RECOVERY_SCHEMA
    assert promoted["promoted_case_count"] == 102
    assert promoted["matched_physical_row_count"] == 34
    assert promoted["source_comparison"]["sha256_matches_manifest"] is True
    assert "recovered_analysis_launcher" in promoted["inputs"]
    assert "full_record_postprocessor" in promoted["inputs"]
    assert "primary_realizations" in promoted["outputs"]
    assert "wind_bin_contributions" in promoted["outputs"]
    fingerprint_payload = dict(promoted)
    recorded_fingerprint = fingerprint_payload.pop("fingerprint")
    assert recorded_fingerprint == _fingerprint(fingerprint_payload)
    for record in promoted["outputs"].values():
        path = Path(record["path"])
        assert path.stat().st_size == record["bytes"]
        assert _sha(path) == record["sha256"]
    first_hashes = {path.name: _sha(path) for path in output.iterdir()}

    promotion.promote(campaign, output)

    second_hashes = {path.name: _sha(path) for path in output.iterdir()}
    assert second_hashes == first_hashes


def test_recovered_preflight_records_postcampaign_helper_drift_without_treating_it_as_physics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    manifest, _receipt, _analysis = _write_recovered_fixture(campaign, monkeypatch)
    recorded = manifest["inputs"]["orchestration_sources"][
        "paper_result_promotion"
    ]
    recorded["sha256"] = "f" * 64
    recorded["size_bytes"] = int(recorded["size_bytes"]) - 1
    _rewrite_linked_campaign_records(campaign, manifest)

    report = promotion.preflight(campaign)

    comparison = report["source_comparison"]
    assert comparison["sha256_matches_manifest"] is False
    assert comparison["size_matches_manifest"] is False
    assert "post-campaign paper tool" in comparison["interpretation"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(schema=promotion.BASE_ANALYSIS_SCHEMA), "v3"),
        (lambda value: value.update(complete_case_count=101), "102-case v3"),
        (
            lambda value: value["cases"].__setitem__(1, dict(value["cases"][0])),
            "differs from the manifest",
        ),
        (
            lambda value: value.update(recovered_case_count=33),
            "102-case v3",
        ),
    ),
)
def test_recovered_promotion_rejects_wrong_analysis_without_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_recovered_fixture(campaign, monkeypatch)
    analysis_path = campaign / promotion.recovery.RECOVERED_ANALYSIS_NAME
    analysis = json.loads(analysis_path.read_text())
    mutation(analysis)
    analysis.pop("fingerprint")
    analysis["fingerprint"] = _fingerprint(analysis)
    _write_json(analysis_path, analysis)
    output = tmp_path / "paper_export_recovered_v2"

    with pytest.raises(promotion.RecoveredPromotionError, match=message):
        promotion.promote(campaign, output)
    assert not output.exists()


def test_recovered_preflight_rejects_tampered_recovery_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_recovered_fixture(campaign, monkeypatch)
    receipt_path = campaign / promotion.recovery.RECOVERY_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    receipt["accepted_case_count"] = 101
    _write_json(receipt_path, receipt)

    with pytest.raises(promotion.RecoveredPromotionError, match="Identity changed"):
        promotion.preflight(campaign)


def test_recovered_preflight_restores_helper_after_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_recovered_fixture(campaign, monkeypatch)
    original_loader = promotion.legacy._load_json
    original_schema = promotion.legacy.ANALYSIS_SCHEMA
    monkeypatch.setattr(
        promotion.legacy,
        "_validate_campaign",
        lambda _root: (_ for _ in ()).throw(
            promotion.legacy.PromotionError("injected helper failure")
        ),
    )

    with pytest.raises(promotion.RecoveredPromotionError, match="injected"):
        promotion.preflight(campaign)
    assert promotion.legacy._load_json is original_loader
    assert promotion.legacy.ANALYSIS_SCHEMA == original_schema


def test_recovered_analysis_launcher_is_analysis_only_and_ordered() -> None:
    launcher = (
        ROOT / "run_headless/submit_hvawt_recovered_dlc_analysis.sj"
    ).read_text()

    assert "recover_hvawt_completed_one_way.py verify" in launcher
    assert "recover_hvawt_completed_one_way.py analyze" in launcher
    assert "--preflight-only" in launcher
    assert launcher.index("--preflight-only") < launcher.index(
        "postprocess_parked_demo.py"
    )
    assert launcher.count("postprocess_parked_demo.py") == 1
    assert "raw_vs_one_way raw_vs_complete one_way_vs_complete" in launcher
    assert "--min-fatigue-samples 80000" in launcher
    assert "--min-fatigue-duration 600" in launcher
    assert "--min-rainflow-cycles 10" in launcher
    assert "prepare_unified_hvawt_paper_results_v2.py" in launcher
    assert "#SBATCH --mail-type=ALL" in launcher
    assert "#SBATCH --licenses" not in launcher
    forbidden = ("run_qblade_sim", "#SBATCH --array", "sbatch ", "srun ")
    assert all(token not in launcher for token in forbidden)
