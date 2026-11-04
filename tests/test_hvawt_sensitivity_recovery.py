from __future__ import annotations

from contextlib import contextmanager
import csv
import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from vorlap import hvawt_sensitivity_campaign as sensitivity
from vorlap import hvawt_sensitivity_recovery as recovery
from vorlap.hvawt_response_pilot import HVAWTResponsePilotError


ROOT = Path(__file__).resolve().parents[1]
RESULT_FILTER = ROOT / "run_headless/config/parked_fatigue_results.filter"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _filter_header() -> list[str]:
    channels = [
        line.strip()
        for line in RESULT_FILTER.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]
    return ["~".join(channel.split()) for channel in channels]


def _small_grid_plan() -> tuple[float, float, int, int]:
    timestep = recovery.EXPECTED_HALF_TIMESTEP_S
    store_from = 100.0
    start = int(__import__("math").ceil(store_from / timestep - 1.0e-10))
    total = start + 12
    return timestep, store_from, start, total


def _write_result(
    path: Path,
    *,
    mutate: str | None = None,
) -> tuple[float, float, int]:
    timestep, store_from, start, total = _small_grid_plan()
    rows: list[list[str]] = []
    for offset, sample_index in enumerate(range(start, total)):
        lower, upper, _exact, _accumulation, _rounding = (
            recovery._rendered_time_candidates(
                sample_index=sample_index, timestep_s=timestep
            )
        )
        token = upper if offset % 2 and lower != upper else lower
        rows.append([token, *(["1.00000E+00"] * 25)])
    if mutate == "deleted":
        rows.pop(4)
    elif mutate == "duplicate":
        rows[4][0] = rows[3][0]
    elif mutate == "impossible_subtoken":
        rows[4][0] = f"{float(rows[4][0]) + 1.0e-4:.6E}"
    elif mutate == "shifted":
        rows[4][0] = f"{float(rows[4][0]) + 1.0e-3:.5E}"
    elif mutate == "nonfinite":
        rows[4][8] = "nan"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["QBlade result fixture"])
        writer.writerow(["generated for rendered-time-grid audit"])
        writer.writerow(_filter_header())
        writer.writerows(rows)
    return timestep, store_from, total


def test_renderer_audit_accepts_only_the_configured_qblade_tokens(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.txt"
    timestep, store_from, total = _write_result(path)

    audit = recovery.audit_qblade_export_time_grid(
        path,
        result_filter=RESULT_FILTER,
        timestep_s=timestep,
        total_steps=total,
        store_from_s=store_from,
    )

    assert audit["schema"] == "vorlap-qblade-rendered-time-grid-audit-v1"
    assert audit["row_count"] == 12
    assert audit["channel_count"] == 26
    assert audit["strictly_increasing_exported_time"] is True
    assert audit["maximum_renderer_mismatch_s"] == 0.0
    assert audit["maximum_time_grid_error_s"] <= audit["time_grid_tolerance_s"]


@pytest.mark.parametrize(
    "mutation",
    ("deleted", "duplicate", "impossible_subtoken", "shifted", "nonfinite"),
)
def test_renderer_audit_rejects_adversarial_time_or_data_changes(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / f"{mutation}.txt"
    timestep, store_from, total = _write_result(path, mutate=mutation)

    with pytest.raises(recovery.HVAWTSensitivityRecoveryError):
        recovery.audit_qblade_export_time_grid(
            path,
            result_filter=RESULT_FILTER,
            timestep_s=timestep,
            total_steps=total,
            store_from_s=store_from,
        )


def test_renderer_rejects_wrong_configured_timestep(tmp_path: Path) -> None:
    path = tmp_path / "wrong_timestep.txt"
    timestep, store_from, total = _write_result(path)
    with pytest.raises(
        recovery.HVAWTSensitivityRecoveryError,
        match="inconsistent with the configured grid",
    ):
        recovery.audit_qblade_export_time_grid(
            path,
            result_filter=RESULT_FILTER,
            timestep_s=timestep * 1.001,
            total_steps=total,
            store_from_s=store_from,
        )


def test_renderer_rejects_header_change(tmp_path: Path) -> None:
    path = tmp_path / "header.txt"
    timestep, store_from, total = _write_result(path)
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[2] = rows[2].replace("Time~[s]", "Time~changed~[s]", 1)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(
        recovery.HVAWTSensitivityRecoveryError, match="header differs"
    ):
        recovery.audit_qblade_export_time_grid(
            path,
            result_filter=RESULT_FILTER,
            timestep_s=timestep,
            total_steps=total,
            store_from_s=store_from,
        )


def test_half_step_scope_requires_exact_observed_grid(monkeypatch) -> None:
    monkeypatch.setattr(
        recovery,
        "audit_qblade_export_time_grid",
        lambda *_a, **_k: {
            "row_count": recovery.EXPECTED_HALF_RETAINED_ROWS,
            "maximum_time_grid_error_s": recovery.OBSERVED_EXPORT_ERROR_S,
        },
    )
    result = recovery._recovery_result_auditor(
        Path("unused"),
        result_filter=RESULT_FILTER,
        timestep_s=recovery.EXPECTED_HALF_TIMESTEP_S,
        total_steps=recovery.EXPECTED_HALF_TOTAL_STEPS,
        store_from_s=sensitivity.STORE_FROM_S,
    )
    assert result["row_count"] == recovery.EXPECTED_HALF_RETAINED_ROWS

    monkeypatch.setattr(
        recovery,
        "audit_qblade_export_time_grid",
        lambda *_a, **_k: {
            "row_count": recovery.EXPECTED_HALF_RETAINED_ROWS,
            "maximum_time_grid_error_s": 4.0e-4,
        },
    )
    with pytest.raises(
        recovery.HVAWTSensitivityRecoveryError,
        match="exact observed export grid",
    ):
        recovery._recovery_result_auditor(
            Path("unused"),
            result_filter=RESULT_FILTER,
            timestep_s=recovery.EXPECTED_HALF_TIMESTEP_S,
            total_steps=recovery.EXPECTED_HALF_TOTAL_STEPS,
            store_from_s=sensitivity.STORE_FROM_S,
        )


def test_nominal_step_still_uses_the_frozen_auditor(monkeypatch) -> None:
    observed: dict[str, Any] = {}

    def frozen(path: Path, **kwargs: Any) -> dict[str, object]:
        observed.update({"path": path, **kwargs})
        return {"frozen": True}

    monkeypatch.setattr(recovery, "_FROZEN_RESULT_AUDITOR", frozen)
    result = recovery._recovery_result_auditor(
        Path("nominal.txt"),
        result_filter=RESULT_FILTER,
        timestep_s=0.0075,
        total_steps=88_000,
        store_from_s=60.0,
    )
    assert result == {"frozen": True}
    assert observed["timestep_s"] == 0.0075


def _receipt_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = tmp_path / "campaign"
    root.mkdir()
    cases: list[dict[str, Any]] = []
    for index in range(6):
        variant = (
            recovery.EXPECTED_VARIANT
            if index in recovery.EXPECTED_RECOVERED_INDICES
            else ("complete_response_off" if index % 3 == 0 else "complete_lower_cap")
        )
        row_id = "row_a" if index < 3 else "row_b"
        case_id = f"{row_id}__{variant}"
        result_dir = Path("results") / row_id / variant
        total_steps = (
            recovery.EXPECTED_HALF_TOTAL_STEPS
            if index in recovery.EXPECTED_RECOVERED_INDICES
            else 88_000
        )
        case = {
            "index": index,
            "case_id": case_id,
            "row_id": row_id,
            "variant": variant,
            "timestep_s": (
                recovery.EXPECTED_HALF_TIMESTEP_S
                if index in recovery.EXPECTED_RECOVERED_INDICES
                else 0.0075
            ),
            "total_steps": total_steps,
            "retained_rows": (
                recovery.EXPECTED_HALF_RETAINED_ROWS
                if index in recovery.EXPECTED_RECOVERED_INDICES
                else 80_000
            ),
            "result_dir": result_dir.as_posix(),
            "sim": (Path("cases") / row_id / variant / "case.sim").as_posix(),
            "result_name": f"{case_id}_qblade_results",
        }
        cases.append(case)
        directory = root / result_dir
        directory.mkdir(parents=True)
        result = directory / f"{case['result_name']}.txt"
        result.write_text("terminal result\n")
        history = directory / "case_sil_vorlap_history.csv"
        history.write_text("step_index,time_s\n0,0\n")
        summary = {
            "failed": False,
            "stopped": False,
            "completed_steps": total_steps,
            "requested_steps": total_steps,
            "exported_results": True,
            "sil_vorlap_history": {
                "mode": "compact",
                "path": str(history.resolve()),
                "file_identity": sensitivity._identity(history),
            },
        }
        _write_json(directory / "case_run_summary.json", summary)
        _write_json(directory / "case_metadata.json", {"case_id": case_id})
        lock = root / ".case_locks" / f"{case_id}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.touch()

    manifest = {
        "fingerprint": recovery.EXPECTED_CAMPAIGN_FINGERPRINT,
        "cases": cases,
    }
    runtime = {"runtime": "fixture"}
    smoke = {"fingerprint": recovery.EXPECTED_SMOKE_FINGERPRINT}
    authorization = {
        "fingerprint": recovery.EXPECTED_AUTHORIZATION_FINGERPRINT,
        "authorized_case_indices": list(range(6)),
        "fresh_start_only": True,
    }
    state = {
        "campaign_root": root,
        "manifest": manifest,
        "valid": True,
        "runtime": runtime,
        "runtime_fingerprint": recovery.EXPECTED_RUNTIME_FINGERPRINT,
        "smoke": smoke,
        "authorization": authorization,
    }
    _write_json(root / sensitivity.MANIFEST_NAME, manifest)
    _write_json(root / sensitivity.RUNTIME_RECEIPT_NAME, runtime)
    _write_json(root / sensitivity.SMOKE_RECEIPT_NAME, smoke)
    _write_json(root / sensitivity.AUTHORIZATION_NAME, authorization)
    monkeypatch.setattr(recovery, "_load_recovery_state", lambda *_a, **_k: state)

    def terminal_audit(*, root: Path, manifest: dict, case: dict) -> dict[str, object]:
        del root, manifest
        index = case["index"]
        if (
            index in recovery.EXPECTED_RECOVERED_INDICES
            and sensitivity._audit_qblade_fatigue_result
            is recovery._FROZEN_RESULT_AUDITOR
        ):
            raise HVAWTResponsePilotError(recovery.EXACT_FROZEN_ERROR)
        return {
            "fixture_terminal_receipt": index,
            "result_audit": {
                "row_count": case["retained_rows"],
                "maximum_time_grid_error_s": (
                    recovery.OBSERVED_EXPORT_ERROR_S
                    if index in recovery.EXPECTED_RECOVERED_INDICES
                    else 0.0
                ),
            },
            "compact_history_grid": {
                "rows": case["total_steps"],
                "maximum_time_grid_error_s": 0.0,
            },
        }

    monkeypatch.setattr(sensitivity, "_terminal_case_audit", terminal_audit)
    return root, manifest, state


def test_receipt_accepts_exact_four_unchanged_two_recovered_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _manifest, _state = _receipt_fixture(tmp_path, monkeypatch)

    receipt = recovery.write_recovery_receipt(root)
    first_bytes = (root / recovery.RECOVERY_RECEIPT_NAME).read_bytes()
    repeated = recovery.write_recovery_receipt(root)

    assert repeated == receipt
    assert (root / recovery.RECOVERY_RECEIPT_NAME).read_bytes() == first_bytes
    assert receipt["accepted_case_count"] == 6
    assert receipt["unchanged_frozen_audit_case_count"] == 4
    assert receipt["recovered_case_count"] == 2
    assert tuple(receipt["recovered_case_indices"]) == (1, 4)
    assert receipt["simulation_rerun_count"] == 0
    assert receipt["modified_artifact_count"] == 0
    assert receipt["artifact_identities_before"] == receipt[
        "artifact_identities_after"
    ]
    assert [row["audit_disposition"] for row in receipt["terminal_cases"]] == [
        "unchanged_frozen_terminal_audit",
        "recovered_qblade_text_timestamp_representation_only",
        "unchanged_frozen_terminal_audit",
        "unchanged_frozen_terminal_audit",
        "recovered_qblade_text_timestamp_representation_only",
        "unchanged_frozen_terminal_audit",
    ]


def test_receipt_rejects_a_different_half_step_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _manifest, _state = _receipt_fixture(tmp_path, monkeypatch)

    def wrong_terminal(*, root: Path, manifest: dict, case: dict) -> dict[str, object]:
        del root, manifest
        if case["index"] in recovery.EXPECTED_RECOVERED_INDICES:
            raise HVAWTResponsePilotError("different terminal failure")
        return {"fixture": case["index"]}

    monkeypatch.setattr(sensitivity, "_terminal_case_audit", wrong_terminal)
    with pytest.raises(
        recovery.HVAWTSensitivityRecoveryError,
        match="earlier or different gate",
    ):
        recovery.build_recovery_receipt(root)


def test_receipt_detects_any_terminal_artifact_modification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, _state = _receipt_fixture(tmp_path, monkeypatch)
    original = sensitivity._terminal_case_audit
    changed = False

    def mutating_terminal(*, root: Path, manifest: dict, case: dict) -> dict[str, object]:
        nonlocal changed
        try:
            return original(root=root, manifest=manifest, case=case)
        finally:
            if (
                case["index"] == 4
                and sensitivity._audit_qblade_fatigue_result
                is recovery._recovery_result_auditor
                and not changed
            ):
                changed = True
                result = recovery._case_paths(root=root, case=case)["result"]
                result.write_text(result.read_text() + "changed\n")

    monkeypatch.setattr(sensitivity, "_terminal_case_audit", mutating_terminal)
    with pytest.raises(
        recovery.HVAWTSensitivityRecoveryError,
        match="artifact changed",
    ):
        recovery.build_recovery_receipt(root)
    assert changed is True
    assert len(manifest["cases"]) == 6


@pytest.mark.parametrize("tamper", ("contract", "source", "artifact"))
def test_loaded_receipt_rejects_recomputed_or_live_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    root, manifest, _state = _receipt_fixture(tmp_path, monkeypatch)
    receipt = recovery.write_recovery_receipt(root)
    if tamper == "contract":
        receipt["recovered_case_indices"] = [1, 3]
        content = dict(receipt)
        content.pop("fingerprint")
        receipt["fingerprint"] = recovery.canonical_fingerprint(content)
        _write_json(root / recovery.RECOVERY_RECEIPT_NAME, receipt)
    elif tamper == "source":
        receipt["recovery_sources"]["recovery_model"]["sha256"] = "0" * 64
        content = dict(receipt)
        content.pop("fingerprint")
        receipt["fingerprint"] = recovery.canonical_fingerprint(content)
        _write_json(root / recovery.RECOVERY_RECEIPT_NAME, receipt)
    else:
        result = recovery._case_paths(
            root=root, case=manifest["cases"][0]
        )["result"]
        result.write_text(result.read_text() + "tampered\n")

    with pytest.raises(recovery.HVAWTSensitivityRecoveryError):
        recovery.load_recovery_receipt(root, verify_live_artifacts=False)


def test_scoped_replacements_restore_frozen_objects_after_exception() -> None:
    frozen_auditor = sensitivity._audit_qblade_fatigue_result
    frozen_writer = sensitivity._write_new_or_exact_json
    frozen_schema = sensitivity.ANALYSIS_SCHEMA

    with pytest.raises(RuntimeError):
        with recovery._scoped_recovery_result_audit():
            assert (
                sensitivity._audit_qblade_fatigue_result
                is recovery._recovery_result_auditor
            )
            raise RuntimeError("stop")
    assert sensitivity._audit_qblade_fatigue_result is frozen_auditor

    with pytest.raises(RuntimeError):
        with recovery._capture_analysis_write():
            assert sensitivity._write_new_or_exact_json is not frozen_writer
            raise RuntimeError("stop")
    assert sensitivity._write_new_or_exact_json is frozen_writer

    with pytest.raises(RuntimeError):
        with recovery._scoped_recovered_analysis_schema():
            assert sensitivity.ANALYSIS_SCHEMA == recovery.RECOVERED_ANALYSIS_SCHEMA
            raise RuntimeError("stop")
    assert sensitivity.ANALYSIS_SCHEMA == frozen_schema


def test_recovered_analysis_uses_original_analyzer_without_writing_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _manifest, state = _receipt_fixture(tmp_path, monkeypatch)
    receipt = {
        "fingerprint": "r" * 64,
        "claim_boundary": "audit recovery only",
        "terminal_cases": [
            {
                "case_index": index,
                "case_id": case["case_id"],
                "audit_disposition": (
                    "recovered_qblade_text_timestamp_representation_only"
                    if index in recovery.EXPECTED_RECOVERED_INDICES
                    else "unchanged_frozen_terminal_audit"
                ),
            }
            for index, case in enumerate(state["manifest"]["cases"])
        ],
    }
    monkeypatch.setattr(recovery, "load_recovery_receipt", lambda *_a, **_k: receipt)
    _write_json(root / recovery.RECOVERY_RECEIPT_NAME, receipt)

    def analyze(
        campaign: Path, *, base_campaign_root: Path, output: Path
    ) -> dict[str, object]:
        del campaign, base_campaign_root
        assert (
            sensitivity._audit_qblade_fatigue_result
            is recovery._recovery_result_auditor
        )
        payload: dict[str, object] = {
            "schema": "old-v1",
            "status": "completed",
            "campaign_fingerprint": recovery.EXPECTED_CAMPAIGN_FINGERPRINT,
            "complete_case_count": 6,
            "expected_case_count": 6,
            "all_execution_audits_passed": True,
            "claim_boundary": "existing claim boundary",
        }
        payload["fingerprint"] = recovery.canonical_fingerprint(payload)
        return sensitivity._write_new_or_exact_json(
            output,
            payload,
            label="captured fixture",
            mismatch_message="unused",
        )

    monkeypatch.setattr(sensitivity, "analyze_campaign", analyze)
    report = recovery.build_recovered_analysis(
        root, base_campaign_root=tmp_path / "base"
    )

    assert report["schema"] == recovery.RECOVERED_ANALYSIS_SCHEMA
    assert report["recovery_fingerprint"] == receipt["fingerprint"]
    assert tuple(report["recovered_case_indices"]) == (1, 4)
    assert report["simulation_rerun_count"] == 0
    assert report["modified_terminal_artifact_count"] == 0
    assert not (root / sensitivity.ANALYSIS_NAME).exists()
    assert not (root / recovery.RECOVERED_ANALYSIS_NAME).exists()
    content = dict(report)
    fingerprint = content.pop("fingerprint")
    assert fingerprint == recovery.canonical_fingerprint(content)


def test_recovered_promotion_binds_outer_receipt_and_restores_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _manifest, _state = _receipt_fixture(tmp_path, monkeypatch)
    recovery_receipt = {"fingerprint": "r" * 64, "claim_boundary": "bounded"}
    analysis = {"fingerprint": "a" * 64}
    analysis_path = root / recovery.RECOVERED_ANALYSIS_NAME
    _write_json(root / recovery.RECOVERY_RECEIPT_NAME, recovery_receipt)
    _write_json(analysis_path, analysis)
    monkeypatch.setattr(
        recovery,
        "_load_recovered_analysis",
        lambda *_a, **_k: (analysis_path, analysis, recovery_receipt),
    )
    original_schema = sensitivity.ANALYSIS_SCHEMA

    def promote(
        campaign: Path, *, analysis_path: Path, output_dir: Path
    ) -> dict[str, object]:
        del campaign
        assert analysis_path == root / recovery.RECOVERED_ANALYSIS_NAME
        assert sensitivity.ANALYSIS_SCHEMA == recovery.RECOVERED_ANALYSIS_SCHEMA
        output_dir.mkdir(exist_ok=True)
        inner: dict[str, object] = {
            "schema": sensitivity.PROMOTION_SCHEMA,
            "analysis_fingerprint": analysis["fingerprint"],
            "outputs": {},
        }
        inner["fingerprint"] = recovery.canonical_fingerprint(inner)
        _write_json(output_dir / sensitivity.PROMOTION_RECEIPT_NAME, inner)
        return inner

    monkeypatch.setattr(sensitivity, "promote_analysis", promote)
    first = recovery.promote_recovered_analysis(root)
    repeated = recovery.promote_recovered_analysis(root)

    assert repeated == first
    assert first["schema"] == recovery.RECOVERED_PROMOTION_SCHEMA
    assert first["recovery_fingerprint"] == recovery_receipt["fingerprint"]
    assert first["analysis_fingerprint"] == analysis["fingerprint"]
    assert first["simulation_rerun_count"] == 0
    assert first["modified_terminal_artifact_count"] == 0
    assert sensitivity.ANALYSIS_SCHEMA == original_schema
    content = dict(first)
    fingerprint = content.pop("fingerprint")
    assert fingerprint == recovery.canonical_fingerprint(content)


def test_new_recovery_interfaces_have_no_simulation_or_archive_path() -> None:
    module = (ROOT / "vorlap/hvawt_sensitivity_recovery.py").read_text()
    for forbidden in ("subprocess", "execute_case", "_archive_partial", "shutil.move"):
        assert forbidden not in module

    for relative in (
        "run_headless/submit_hvawt_sensitivity_audit_recovery.sj",
        "run_headless/submit_hvawt_sensitivity_recovered_analysis.sj",
    ):
        script = (ROOT / relative).read_text().lower()
        assert "#sbatch --array" not in script
        assert "qblade_root" not in script
        assert "hvawt_sensitivity_campaign.py run" not in script
        assert "archive_partial" not in script


def test_frozen_sensitivity_sources_remain_unmodified_by_recovery_patch() -> None:
    status = __import__("subprocess").run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "vorlap/hvawt_sensitivity_campaign.py",
            "vorlap/hvawt_response_pilot.py",
            "run_headless/hvawt_sensitivity_campaign.py",
            "run_headless/submit_hvawt_sensitivity_analysis.sj",
            "run_headless/submit_hvawt_sensitivity_flight.sj",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
