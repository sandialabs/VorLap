from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
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
    / "paper_drafts/NAWEA_QBladeBridge/analysis/prepare_hvawt_sensitivity_paper_results.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_hvawt_sensitivity_paper_results", SCRIPT
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _analysis(*, passed: bool) -> dict[str, object]:
    nominal = {
        paper.ROW_ORDER[0]: 4000.0,
        paper.ROW_ORDER[1]: 1200.0,
    }
    factors = {
        "nominal_raw": 0.5,
        "nominal_one_way": 0.7,
        "nominal_complete": 1.0,
        "complete_response_off": 0.8,
        "complete_half_timestep": 1.01,
        "complete_lower_cap": 0.99,
    }
    metrics = []
    for row_id in paper.ROW_ORDER:
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
    gates = []
    for row_id in paper.ROW_ORDER:
        gates.extend(
            [
                {
                    "row_id": row_id,
                    "family": "complete_path_numerical_load_sensitivity",
                    "metric": "half_timestep_del_Nm",
                    "passed": passed or row_id == paper.ROW_ORDER[1],
                },
                {
                    "row_id": row_id,
                    "family": "response_ablation_finite_record_sensitivity",
                    "metric": "response_off_del_480_to_600s",
                    "passed": True,
                },
            ]
        )
    ratios = [
        {"row_id": row_id, "setting": setting}
        for row_id in paper.ROW_ORDER
        for setting in ("nominal", "response_off", "half_timestep", "lower_cap")
    ]
    modal = [
        {
            "row_id": row_id,
            "mode": mode,
            "intervariant_frequency_gate_applied": mode == 12,
        }
        for row_id in paper.ROW_ORDER
        for mode in (6, 12)
    ]
    attribution = [
        {
            "row_id": row_id,
            "response_off_del_log_effect": 0.2,
            "two_tested_complete_path_del_log_envelope": 0.01,
        }
        for row_id in paper.ROW_ORDER
    ]
    report: dict[str, object] = {
        "schema": paper.ANALYSIS_SCHEMA,
        "status": paper.PASS_STATUS if passed else paper.FAIL_STATUS,
        "campaign_fingerprint": "campaign",
        "complete_case_count": 6,
        "expected_case_count": 6,
        "all_execution_audits_passed": True,
        "response_ablation_execution_audits_passed": True,
        "all_complete_path_numerical_gates_passed": passed,
        "all_response_ablation_diagnostics_passed": True,
        "all_preregistered_screening_gates_passed": passed,
        "response_ablation_effect_closeness_gate_applied": False,
        "passed_gate_count": sum(row["passed"] is True for row in gates),
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
    }
    report["fingerprint"] = paper.canonical_fingerprint(report)
    return report


def _handoff(tmp_path: Path, *, passed: bool = True) -> tuple[Path, Path]:
    analysis = _analysis(passed=passed)
    analysis_path = tmp_path / paper.ANALYSIS_NAME
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    export = tmp_path / "paper_export_sensitivity_v1"
    export.mkdir()
    source_rows = {
        "metrics": analysis["metrics"],
        "gates": analysis["gates"],
        "ratio_context": analysis["ratio_records"],
        "modal_diagnostics": analysis["modal_diagnostics"],
        "response_attribution": analysis["response_attribution"],
    }
    for role, rows in source_rows.items():
        _write_csv(export / paper.EXPECTED_EXPORTS[role], rows)
    summary: dict[str, object] = {
        "schema": paper.SUMMARY_SCHEMA,
        "status": analysis["status"],
        "analysis_fingerprint": analysis["fingerprint"],
        "all_complete_path_numerical_gates_passed": analysis[
            "all_complete_path_numerical_gates_passed"
        ],
        "all_response_ablation_diagnostics_passed": True,
        "response_ablation_execution_audits_passed": True,
        "all_preregistered_screening_gates_passed": analysis[
            "all_preregistered_screening_gates_passed"
        ],
        "response_ablation_effect_closeness_gate_applied": False,
        "passed_gate_count": analysis["passed_gate_count"],
        "gate_count": analysis["gate_count"],
        "modal_applicability": analysis["modal_applicability"],
        "ratio_records": analysis["ratio_records"],
        "response_attribution": analysis["response_attribution"],
        "claim_boundary": analysis["claim_boundary"],
    }
    summary["fingerprint"] = paper.canonical_fingerprint(summary)
    summary_path = export / paper.EXPECTED_EXPORTS["summary"]
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs = {
        role: _identity(export / filename, f"paper_export_sensitivity_v1/{filename}")
        for role, filename in paper.EXPECTED_EXPORTS.items()
    }
    receipt: dict[str, object] = {
        "schema": paper.PROMOTION_SCHEMA,
        "status": (
            paper.PROMOTED_PASS_STATUS if passed else paper.PROMOTED_FAIL_STATUS
        ),
        "campaign_fingerprint": "campaign",
        "analysis": _identity(analysis_path, paper.ANALYSIS_NAME),
        "analysis_fingerprint": analysis["fingerprint"],
        "outputs": outputs,
        "claim_boundary": analysis["claim_boundary"],
    }
    receipt["fingerprint"] = paper.canonical_fingerprint(receipt)
    (export / paper.PROMOTION_RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return analysis_path, export


def test_passing_handoff_installs_atomically_and_is_idempotent(tmp_path):
    analysis, export = _handoff(tmp_path, passed=True)
    paper_dir = tmp_path / "paper"
    (paper_dir / "generated").mkdir(parents=True)

    first = paper.install_paper_package(
        analysis_path=analysis, export_dir=export, paper_dir=paper_dir
    )
    second = paper.install_paper_package(
        analysis_path=analysis, export_dir=export, paper_dir=paper_dir
    )

    assert second == first
    assert first["status"] == "installed_passing_targeted_sensitivity_screen"
    package = paper_dir / "generated" / paper.PACKAGE_NAME
    fragment = (package / paper.GENERATED_NAMES["fragment"]).read_text()
    table = (package / paper.GENERATED_NAMES["table"]).read_text()
    assert "All preregistered complete-path numerical" in fragment
    assert "not full timestep convergence or bandwidth independence" in fragment
    assert "4433458, local $Y$" in table
    assert "-20.0\\,\\%" in table
    assert "Response off" in table
    assert "4000" in table
    assert not list((paper_dir / "generated").glob(".*.preparing.*"))


def test_failed_gate_is_promoted_with_narrowed_claim(tmp_path):
    analysis, export = _handoff(tmp_path, passed=False)
    paper_dir = tmp_path / "paper"
    (paper_dir / "generated").mkdir(parents=True)

    receipt = paper.install_paper_package(
        analysis_path=analysis, export_dir=export, paper_dir=paper_dir
    )

    fragment = (
        paper_dir
        / "generated"
        / paper.PACKAGE_NAME
        / paper.GENERATED_NAMES["fragment"]
    ).read_text()
    assert receipt["status"] == "installed_claim_bounded_failed_targeted_sensitivity_screen"
    assert "passes 3 of 4 preregistered gates" in fragment
    assert "does not support a numerical-robustness claim" in fragment
    assert r"complete\_path\_numerical\_load\_sensitivity" in fragment


def test_tampered_promoted_csv_is_rejected(tmp_path):
    analysis, export = _handoff(tmp_path, passed=True)
    metrics = export / paper.EXPECTED_EXPORTS["metrics"]
    metrics.write_text(metrics.read_text() + "tampered\n", encoding="utf-8")

    with pytest.raises(
        paper.SensitivityPaperError,
        match="identity does not match local bytes",
    ):
        paper.validate_handoff(analysis_path=analysis, export_dir=export)


def test_existing_changed_paper_package_is_not_overwritten(tmp_path):
    analysis, export = _handoff(tmp_path, passed=True)
    paper_dir = tmp_path / "paper"
    (paper_dir / "generated").mkdir(parents=True)
    paper.install_paper_package(
        analysis_path=analysis, export_dir=export, paper_dir=paper_dir
    )
    fragment = (
        paper_dir
        / "generated"
        / paper.PACKAGE_NAME
        / paper.GENERATED_NAMES["fragment"]
    )
    fragment.write_text("changed\n", encoding="utf-8")

    with pytest.raises(
        paper.SensitivityPaperError,
        match="differs; use a versioned package",
    ):
        paper.install_paper_package(
            analysis_path=analysis, export_dir=export, paper_dir=paper_dir
        )
    assert fragment.read_text(encoding="utf-8") == "changed\n"
