from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "paper_drafts/NAWEA_QBladeBridge/analysis/prepare_hvawt_sensitivity_paper_results_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_hvawt_sensitivity_paper_results_v2", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
paper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paper
SPEC.loader.exec_module(paper)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path, display: str) -> dict[str, object]:
    return {
        "path": display,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _handoff(
    tmp_path: Path, *, all_passed: bool = False
) -> tuple[Path, Path, Path]:
    root = tmp_path / "campaign"
    root.mkdir()
    export = root / paper.RECOVERED_EXPORT_NAME
    export.mkdir()

    recovery_claim = "read-only two-case recovery; no validation"
    recovery: dict[str, object] = {
        "schema": paper.RECOVERY_SCHEMA,
        "status": paper.RECOVERY_STATUS,
        "campaign_fingerprint": "campaign",
        "accepted_case_count": 6,
        "unchanged_frozen_audit_case_count": 4,
        "recovered_case_count": 2,
        "recovered_case_indices": [1, 4],
        "simulation_rerun_count": 0,
        "modified_artifact_count": 0,
        "claim_boundary": recovery_claim,
    }
    recovery["fingerprint"] = paper.canonical_fingerprint(recovery)
    recovery_path = root / paper.RECOVERY_RECEIPT_NAME
    _write_json(recovery_path, recovery)
    recovery_identity = _identity(recovery_path, paper.RECOVERY_RECEIPT_NAME)

    nominal = {
        paper.common.ROW_ORDER[0]: 4000.0,
        paper.common.ROW_ORDER[1]: 1200.0,
    }
    factors = {
        "nominal_raw": 0.5,
        "nominal_one_way": 0.7,
        "nominal_complete": 1.0,
        "complete_response_off": 0.995,
        "complete_half_timestep": 1.108,
        "complete_lower_cap": 1.02,
    }
    metrics = []
    for row_id in paper.common.ROW_ORDER:
        for label, factor in factors.items():
            metrics.append(
                {
                    "row_id": row_id,
                    "label": label,
                    "del_Nm": nominal[row_id] * factor,
                    "monitored_channel_absolute_extreme_Nm": (
                        nominal[row_id] * 1.5 * factor
                    ),
                }
            )
    gates = [
        {
            "row_id": paper.common.ROW_ORDER[0],
            "family": "complete_path_numerical_load_sensitivity",
            "metric": "complete_half_timestep_del_Nm",
            "comparison": "complete_half_timestep_vs_nominal_complete_log_ratio",
            "observed": math.log(1.108),
            "limit": math.log(1.1),
            "passed": all_passed,
        },
        {
            "row_id": paper.common.ROW_ORDER[1],
            "family": "complete_path_numerical_load_sensitivity",
            "metric": "complete_lower_cap_del_Nm",
            "comparison": "complete_lower_cap_vs_nominal_complete_log_ratio",
            "observed": math.log(1.02),
            "limit": math.log(1.1),
            "passed": True,
        },
        {
            "row_id": paper.common.ROW_ORDER[0],
            "family": "response_ablation_finite_record_sensitivity",
            "metric": "response_off_damage_rate_del",
            "comparison": "duration_normalized_damage_rate_del_480_to_600s",
            "observed": 0.01,
            "limit": 0.05,
            "passed": True,
        },
        {
            "row_id": paper.common.ROW_ORDER[1],
            "family": "response_ablation_finite_record_sensitivity",
            "metric": "response_off_damage_rate_del",
            "comparison": "duration_normalized_damage_rate_del_480_to_600s",
            "observed": 0.01,
            "limit": 0.05,
            "passed": True,
        },
    ]
    ratios = [
        {"row_id": row_id, "setting": setting}
        for row_id in paper.common.ROW_ORDER
        for setting in ("nominal", "response_off", "half_timestep", "lower_cap")
    ]
    modal = [
        {
            "row_id": row_id,
            "mode": mode,
            "intervariant_frequency_gate_applied": mode == 12,
        }
        for row_id in paper.common.ROW_ORDER
        for mode in (6, 12)
    ]
    attribution = [
        {
            "row_id": row_id,
            "response_off_del_log_effect": 0.005,
            "two_tested_complete_path_del_log_envelope": 0.1,
            "response_off_monitored_extreme_log_effect": 0.004,
            "two_tested_complete_path_monitored_extreme_log_envelope": 0.05,
        }
        for row_id in paper.common.ROW_ORDER
    ]
    passed_count = sum(row["passed"] is True for row in gates)
    analysis: dict[str, object] = {
        "schema": paper.RECOVERED_ANALYSIS_SCHEMA,
        "status": (
            paper.common.PASS_STATUS
            if all_passed
            else paper.common.FAIL_STATUS
        ),
        "campaign_fingerprint": "campaign",
        "complete_case_count": 6,
        "expected_case_count": 6,
        "all_execution_audits_passed": True,
        "response_ablation_execution_audits_passed": True,
        "all_complete_path_numerical_gates_passed": all_passed,
        "all_response_ablation_diagnostics_passed": True,
        "all_preregistered_screening_gates_passed": all_passed,
        "response_ablation_effect_closeness_gate_applied": False,
        "passed_gate_count": passed_count,
        "gate_count": len(gates),
        "metrics": metrics,
        "ratio_records": ratios,
        "gates": gates,
        "modal_diagnostics": modal,
        "modal_applicability": {
            "diagnostic_record_count": len(modal),
            "intervariant_gate_applied_count": 2,
            "intervariant_gate_not_applied_count": 2,
        },
        "response_attribution": attribution,
        "claim_boundary": "two selected U35 rows only",
        "recovery_receipt": recovery_identity,
        "recovery_fingerprint": recovery["fingerprint"],
        "recovery_claim_boundary": recovery_claim,
        "recovered_case_count": 2,
        "recovered_case_indices": [1, 4],
        "simulation_rerun_count": 0,
        "modified_terminal_artifact_count": 0,
    }
    analysis["fingerprint"] = paper.canonical_fingerprint(analysis)
    analysis_path = root / paper.RECOVERED_ANALYSIS_NAME
    _write_json(analysis_path, analysis)

    source_rows = {
        "metrics": metrics,
        "gates": gates,
        "ratio_context": ratios,
        "modal_diagnostics": modal,
        "response_attribution": attribution,
    }
    for role, rows in source_rows.items():
        _write_csv(export / paper.common.EXPECTED_EXPORTS[role], rows)
    summary: dict[str, object] = {
        "schema": paper.common.SUMMARY_SCHEMA,
        "status": analysis["status"],
        "analysis_fingerprint": analysis["fingerprint"],
        "all_complete_path_numerical_gates_passed": all_passed,
        "all_response_ablation_diagnostics_passed": True,
        "response_ablation_execution_audits_passed": True,
        "all_preregistered_screening_gates_passed": all_passed,
        "response_ablation_effect_closeness_gate_applied": False,
        "passed_gate_count": passed_count,
        "gate_count": len(gates),
        "modal_applicability": analysis["modal_applicability"],
        "ratio_records": ratios,
        "response_attribution": attribution,
        "claim_boundary": analysis["claim_boundary"],
    }
    summary["fingerprint"] = paper.canonical_fingerprint(summary)
    _write_json(export / paper.common.EXPECTED_EXPORTS["summary"], summary)

    outputs = {
        role: _identity(
            export / filename, f"{paper.RECOVERED_EXPORT_NAME}/{filename}"
        )
        for role, filename in paper.common.EXPECTED_EXPORTS.items()
    }
    inner: dict[str, object] = {
        "schema": paper.common.PROMOTION_SCHEMA,
        "status": (
            paper.common.PROMOTED_PASS_STATUS
            if all_passed
            else paper.common.PROMOTED_FAIL_STATUS
        ),
        "campaign_fingerprint": "campaign",
        "analysis": _identity(analysis_path, paper.RECOVERED_ANALYSIS_NAME),
        "analysis_fingerprint": analysis["fingerprint"],
        "outputs": outputs,
        "claim_boundary": analysis["claim_boundary"],
    }
    inner["fingerprint"] = paper.canonical_fingerprint(inner)
    inner_path = export / paper.common.PROMOTION_RECEIPT_NAME
    _write_json(inner_path, inner)

    outer: dict[str, object] = {
        "schema": paper.RECOVERED_PROMOTION_SCHEMA,
        "status": paper.OUTER_STATUS,
        "campaign_fingerprint": "campaign",
        "recovery_receipt": recovery_identity,
        "recovery_fingerprint": recovery["fingerprint"],
        "analysis": _identity(analysis_path, paper.RECOVERED_ANALYSIS_NAME),
        "analysis_fingerprint": analysis["fingerprint"],
        "inner_promotion": _identity(
            inner_path,
            f"{paper.RECOVERED_EXPORT_NAME}/{paper.common.PROMOTION_RECEIPT_NAME}",
        ),
        "inner_promotion_fingerprint": inner["fingerprint"],
        "outputs": outputs,
        "recovered_case_indices": [1, 4],
        "simulation_rerun_count": 0,
        "modified_terminal_artifact_count": 0,
        "claim_boundary": recovery_claim,
    }
    outer["fingerprint"] = paper.canonical_fingerprint(outer)
    outer_path = root / paper.RECOVERED_PROMOTION_RECEIPT_NAME
    _write_json(outer_path, outer)
    return analysis_path, export, outer_path


def test_recovered_failed_handoff_installs_atomically_and_is_idempotent(tmp_path):
    analysis, export, outer = _handoff(tmp_path)
    paper_dir = tmp_path / "paper"
    (paper_dir / "generated").mkdir(parents=True)

    first = paper.install_paper_package(
        analysis_path=analysis,
        export_dir=export,
        recovery_promotion=outer,
        paper_dir=paper_dir,
    )
    second = paper.install_paper_package(
        analysis_path=analysis,
        export_dir=export,
        recovery_promotion=outer,
        paper_dir=paper_dir,
    )

    assert second == first
    assert first["status"] == (
        "installed_claim_bounded_failed_recovered_targeted_sensitivity_screen"
    )
    package = paper_dir / "generated" / paper.PACKAGE_NAME
    fragment = (package / paper.GENERATED_NAMES["fragment"]).read_text()
    assert "passes 3 of 4 preregistered gates" in fragment
    assert "half-timestep DEL difference 10.8" in fragment
    assert "do not resolve a negative-damping contribution" in fragment
    assert f"generated/{paper.PACKAGE_NAME}" in fragment
    assert (package / paper.RECOVERED_PROMOTION_RECEIPT_NAME).is_file()
    assert (package / paper.RECOVERY_RECEIPT_NAME).is_file()
    assert not list((paper_dir / "generated").glob(".*.preparing.*"))


def test_recovered_handoff_rejects_tampered_outer_receipt(tmp_path):
    analysis, export, outer = _handoff(tmp_path)
    payload = json.loads(outer.read_text())
    payload["simulation_rerun_count"] = 1
    _write_json(outer, payload)

    with pytest.raises(
        paper.SensitivityPaperRecoveryError,
        match="fingerprint is invalid",
    ):
        paper.validate_handoff(
            analysis_path=analysis,
            export_dir=export,
            recovery_promotion=outer,
        )


def test_recovered_handoff_rejects_changed_recovery_bytes(tmp_path):
    analysis, export, outer = _handoff(tmp_path)
    recovery = analysis.parent / paper.RECOVERY_RECEIPT_NAME
    recovery.write_text(recovery.read_text() + " ", encoding="utf-8")

    with pytest.raises(
        paper.SensitivityPaperRecoveryError,
        match="identity does not match local bytes",
    ):
        paper.validate_handoff(
            analysis_path=analysis,
            export_dir=export,
            recovery_promotion=outer,
        )


def test_recovered_package_refuses_changed_overwrite(tmp_path):
    analysis, export, outer = _handoff(tmp_path)
    paper_dir = tmp_path / "paper"
    (paper_dir / "generated").mkdir(parents=True)
    paper.install_paper_package(
        analysis_path=analysis,
        export_dir=export,
        recovery_promotion=outer,
        paper_dir=paper_dir,
    )
    fragment = (
        paper_dir
        / "generated"
        / paper.PACKAGE_NAME
        / paper.GENERATED_NAMES["fragment"]
    )
    fragment.write_text("changed\n", encoding="utf-8")

    with pytest.raises(
        paper.SensitivityPaperRecoveryError,
        match="differs; use a new versioned package",
    ):
        paper.install_paper_package(
            analysis_path=analysis,
            export_dir=export,
            recovery_promotion=outer,
            paper_dir=paper_dir,
        )
    assert fragment.read_text(encoding="utf-8") == "changed\n"
