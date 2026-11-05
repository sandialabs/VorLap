from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from vorlap import hvawt_complete_dlc_campaign as sealed
from vorlap import hvawt_complete_dlc_recovery as recovery


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _inert_response() -> dict[str, object]:
    return {
        "model_type": "none",
        **{name: 0 for name in recovery.INERT_INTEGER_FIELDS},
        **{name: 0.0 for name in recovery.INERT_FLOAT_FIELDS},
    }


def _source_coverage() -> list[dict[str, object]]:
    return [
        {
            "component": component,
            "max_nodal_force_n": float(index + 1),
            "max_response_nodal_force_n": 0.0,
            "max_total_nodal_force_n": float(index + 1),
            "force_coefficient_max_abs": 0.1 * float(index + 1),
            "response_force_coefficient_max_abs": 0.0,
            "total_force_coefficient_max_abs": 0.1 * float(index + 1),
        }
        for index, component in enumerate(sealed.HVAWT_V8_COMPONENT_NODE_COUNTS)
    ]


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    root = tmp_path / "campaign"
    root.mkdir()
    cases: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    runtime = {"schema": sealed.RUNTIME_SCHEMA, "runtime": "fixture"}
    runtime_fingerprint = recovery.canonical_fingerprint(runtime)
    for index in range(102):
        row_index = index // 3
        scenario = sealed.SCENARIOS[index % 3]
        row_id = f"row_{row_index:02d}"
        case_id = f"{row_id}__{scenario}"
        result_dir = Path("results") / row_id / scenario
        case = {
            "index": index,
            "case_id": case_id,
            "row_id": row_id,
            "row_index": row_index,
            "scenario": scenario,
            "wind_speed_mps": float(3 + row_index),
            "seed": 4433456 + (row_index % 3),
            "common_physics_fingerprint": f"{row_index:064x}",
            "force_contract": dict(sealed.SCENARIO_FORCE_CONTRACT[scenario]),
            "result_dir": result_dir.as_posix(),
            "sim": (Path("cases") / row_id / scenario / "case.sim").as_posix(),
            "result_name": f"{case_id}_qblade_results",
        }
        cases.append(case)
        if scenario == sealed.SCENARIOS[1]:
            analysis_rows.append(
                {
                    "case_id": case_id,
                    "row_id": row_id,
                    "scenario": scenario,
                    "status": "blocked",
                    "reason": recovery.EXACT_CLASSIFICATION_REASON,
                }
            )
            directory = root / result_dir
            directory.mkdir(parents=True)
            result = directory / f"{case['result_name']}.txt"
            result.write_text("Time~[s]\tload\n0\t1\n")
            history = directory / "case_sil_vorlap_history.csv"
            history.write_text(
                "step_index,time_s,generated_max_abs_force_component_n,"
                "applied_max_abs_force_component_n\n0,0,1,1\n"
            )
            summary = {
                "sil_vorlap_response_diagnostics_summary": _inert_response(),
                "sil_vorlap_source_coverage": _source_coverage(),
                "sil_vorlap_history": {
                    "mode": "compact",
                    "path": str(history.resolve()),
                    "file_identity": recovery._identity(history),
                },
            }
            _write_json(directory / "case_run_summary.json", summary)
            _write_json(
                directory / "case_metadata.json",
                {
                    "schema": "vorlap-hvawt-complete-dlc-case-metadata-v1",
                    "campaign_fingerprint": "a" * 64,
                    "case_id": case_id,
                    "case_index": index,
                    "row_id": row_id,
                    "scenario": scenario,
                    "force_contract": case["force_contract"],
                    "runtime_identity_fingerprint": runtime_fingerprint,
                },
            )
            lock = root / ".case_locks" / f"{case_id}.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock.touch()
        else:
            analysis_rows.append(
                {
                    "case_id": case_id,
                    "row_id": row_id,
                    "scenario": scenario,
                    "status": "complete",
                    "receipt": {"fixture": index},
                }
            )
    manifest = {
        "schema": sealed.CAMPAIGN_SCHEMA,
        "fingerprint": "a" * 64,
        "cases": cases,
        "inputs": {
            "orchestration_sources": {
                "campaign_model": {"sha256": "1" * 64},
                "paper_result_promotion": {"sha256": "2" * 64},
            }
        },
    }
    analysis = {
        "schema": sealed.ANALYSIS_SCHEMA,
        "status": "incomplete",
        "campaign_fingerprint": manifest["fingerprint"],
        "expected_case_count": 102,
        "complete_case_count": 68,
        "all_three_scenarios_complete": False,
        "cases": analysis_rows,
    }
    _write_json(root / "campaign_manifest.json", manifest)
    _write_json(root / recovery.ORIGINAL_ANALYSIS_NAME, analysis)
    _write_json(root / sealed.RUNTIME_RECEIPT_NAME, runtime)
    _write_json(
        root / "complementary_execution_authorization.json",
        {"fingerprint": "b" * 64},
    )
    _write_json(
        root / sealed.U50_SMOKE_RECEIPT_NAME,
        {"fingerprint": "c" * 64},
    )

    state = {"campaign_root": root, "manifest": manifest, "valid": True}
    monkeypatch.setattr(recovery, "_validate_campaign", lambda *_a, **_k: state)
    monkeypatch.setattr(sealed, "_load_runtime", lambda *_a, **_k: runtime)
    monkeypatch.setattr(
        recovery,
        "_load_continuation_authorization",
        lambda *_a, **_k: {"fingerprint": "b" * 64},
    )
    monkeypatch.setattr(
        sealed,
        "_load_u50_smoke_receipt",
        lambda *_a, **_k: {"fingerprint": "c" * 64},
    )

    def frozen_audit(*, root: Path, manifest: dict, case: dict) -> dict[str, object]:
        del root, manifest
        if case["scenario"] == sealed.SCENARIOS[1]:
            raise sealed.HVAWTCompleteDLCError(recovery.EXACT_FROZEN_ERROR)
        return {"fixture_terminal_receipt": case["index"]}

    monkeypatch.setattr(sealed, "_terminal_case_audit", frozen_audit)
    return root, manifest


def test_preflight_is_read_only_and_identifies_exact_34_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _manifest = _fixture(tmp_path, monkeypatch)
    before = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())

    report = recovery.preflight_recovery(root)

    after = sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())
    assert report["recoverable_candidate_count"] == 34
    assert tuple(report["recoverable_candidate_indices"]) == tuple(range(1, 102, 3))
    assert report["writes_performed"] == 0
    assert report["simulation_actions_available"] is False
    assert before == after


def test_recovery_writes_one_idempotent_receipt_and_versioned_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _manifest = _fixture(tmp_path, monkeypatch)

    receipt = recovery.write_recovery_receipt(root)
    receipt_path = root / recovery.RECOVERY_RECEIPT_NAME
    first_bytes = receipt_path.read_bytes()
    repeated = recovery.write_recovery_receipt(root)

    assert repeated == receipt
    assert receipt_path.read_bytes() == first_bytes
    assert receipt["status"] == recovery.RECOVERY_STATUS
    assert receipt["original_complete_case_count"] == 68
    assert receipt["recovered_case_count"] == 34
    assert receipt["accepted_case_count"] == 102
    assert receipt["simulation_rerun_count"] == 0
    assert receipt["modified_artifact_count"] == 0
    assert len(receipt["cases"]) == 102
    assert len(receipt["recovered_cases"]) == 34
    assert all(
        row["strict_open_loop_one_way"] is False
        and row["kinematic_relative_velocity_feedback_retained"] is True
        for row in receipt["recovered_cases"]
    )

    analysis = recovery.write_recovered_analysis(root)
    assert analysis["schema"] == recovery.RECOVERED_ANALYSIS_SCHEMA
    assert analysis["complete_case_count"] == 102
    assert analysis["recovered_case_count"] == 34
    assert len(analysis["cases"]) == 102
    assert (root / recovery.ORIGINAL_ANALYSIS_NAME).is_file()
    assert (root / recovery.RECOVERED_ANALYSIS_NAME).is_file()


@pytest.mark.parametrize(
    "field", (*recovery.INERT_INTEGER_FIELDS, *recovery.INERT_FLOAT_FIELDS)
)
def test_inert_response_rejects_every_nonzero_field(field: str) -> None:
    payload = _inert_response()
    payload[field] = 1
    with pytest.raises(recovery.HVAWTOneWayRecoveryError, match=field):
        recovery.audit_inert_response_summary(payload)


@pytest.mark.parametrize("mutation", ("missing", "extra", "wrong_model", "nan", "bool"))
def test_inert_response_rejects_schema_and_type_mutations(mutation: str) -> None:
    payload = _inert_response()
    if mutation == "missing":
        payload.pop(recovery.INERT_FLOAT_FIELDS[0])
    elif mutation == "extra":
        payload["unexpected"] = 0
    elif mutation == "wrong_model":
        payload["model_type"] = "active"
    elif mutation == "nan":
        payload[recovery.INERT_FLOAT_FIELDS[0]] = math.nan
    elif mutation == "bool":
        payload[recovery.INERT_INTEGER_FIELDS[0]] = False
    with pytest.raises(recovery.HVAWTOneWayRecoveryError):
        recovery.audit_inert_response_summary(payload)


@pytest.mark.parametrize(
    "field",
    ("max_response_nodal_force_n", "response_force_coefficient_max_abs"),
)
def test_source_coverage_rejects_nonzero_response_fields(field: str) -> None:
    rows = _source_coverage()
    rows[0][field] = 1.0
    with pytest.raises(recovery.HVAWTOneWayRecoveryError, match=field):
        recovery.audit_one_way_source_coverage(rows)


def test_source_coverage_rejects_total_force_mismatch_and_duplicate_component() -> None:
    mismatch = _source_coverage()
    mismatch[0]["max_total_nodal_force_n"] = 2.0
    with pytest.raises(recovery.HVAWTOneWayRecoveryError, match="differs"):
        recovery.audit_one_way_source_coverage(mismatch)

    duplicate = _source_coverage()
    duplicate[1]["component"] = duplicate[0]["component"]
    with pytest.raises(recovery.HVAWTOneWayRecoveryError, match="component"):
        recovery.audit_one_way_source_coverage(duplicate)


@pytest.mark.parametrize(
    "error",
    (
        "Earlier result identity mismatch.",
        recovery.EXACT_FROZEN_ERROR + " extra",
        "One-way result unexpectedly activated response closure.",
    ),
)
def test_recovery_rejects_any_different_frozen_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: str,
) -> None:
    root, manifest = _fixture(tmp_path, monkeypatch)

    def wrong_audit(**kwargs: object) -> dict[str, object]:
        case = kwargs["case"]
        assert isinstance(case, dict)
        if case["scenario"] == sealed.SCENARIOS[1]:
            raise sealed.HVAWTCompleteDLCError(error)
        return {"fixture_terminal_receipt": case["index"]}

    monkeypatch.setattr(sealed, "_terminal_case_audit", wrong_audit)
    with pytest.raises(recovery.HVAWTOneWayRecoveryError, match="earlier/different"):
        recovery.build_recovery_receipt(root)


def test_recovery_rejects_response_columns_in_compact_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = _fixture(tmp_path, monkeypatch)
    case = manifest["cases"][1]
    directory = root / case["result_dir"]
    history = directory / "case_sil_vorlap_history.csv"
    history.write_text(
        "step_index,time_s,response_hidden_force,"
        "generated_max_abs_force_component_n,applied_max_abs_force_component_n\n"
        "0,0,0,1,1\n"
    )
    summary_path = directory / "case_run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["sil_vorlap_history"]["file_identity"] = recovery._identity(history)
    _write_json(summary_path, summary)

    with pytest.raises(recovery.HVAWTOneWayRecoveryError, match="response columns"):
        recovery.build_recovery_receipt(root)


def test_original_analysis_must_be_exact_68_of_102(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = _fixture(tmp_path, monkeypatch)
    analysis_path = root / recovery.ORIGINAL_ANALYSIS_NAME
    analysis = json.loads(analysis_path.read_text())
    analysis["complete_case_count"] = 69
    _write_json(analysis_path, analysis)

    with pytest.raises(recovery.HVAWTOneWayRecoveryError, match="68/102"):
        recovery._validate_original_analysis(root, manifest)


def test_recovery_module_has_no_execution_or_archive_calls() -> None:
    source = Path(recovery.__file__).read_text()
    assert "execute_case(" not in source
    assert "_classify_case_state(" not in source
    assert "_archive_partial(" not in source
    assert "subprocess" not in source


def test_recovery_launcher_is_nonarray_audit_only_and_scheduler_compliant() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "run_headless/submit_hvawt_one_way_audit_recovery.sj"
    ).read_text()

    assert "#SBATCH --array" not in launcher
    assert "#SBATCH --licenses" not in launcher
    assert "#SBATCH --mail-type=ALL" in launcher
    assert "recover_hvawt_completed_one_way.py preflight" in launcher
    assert "recover_hvawt_completed_one_way.py recover" in launcher
    assert launcher.index("preflight") < launcher.index("recover \\")
    for forbidden in (
        "run_qblade_sim.py",
        "execute_case(",
        "_archive_partial(",
        "sbatch ",
        "srun ",
    ):
        assert forbidden not in launcher


def test_loaded_receipt_rejects_changed_bound_recovery_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _manifest = _fixture(tmp_path, monkeypatch)
    source = tmp_path / "fixture_recovery_source.py"
    source.write_text("VERSION = 1\n")
    recovery.write_recovery_receipt(root, recovery_sources={"fixture": source})
    source.write_text("VERSION = 2\n")

    with pytest.raises(recovery.HVAWTOneWayRecoveryError, match="Identity changed"):
        recovery.load_recovery_receipt(root, verify_live_artifacts=False)
