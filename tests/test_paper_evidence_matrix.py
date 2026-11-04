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
CHECKER_PATH = (
    REPO
    / "paper_drafts"
    / "NAWEA_QBladeBridge"
    / "analysis"
    / "check_evidence_matrix.py"
)
MATRIX_PATH = (
    REPO
    / "paper_drafts"
    / "NAWEA_QBladeBridge"
    / "analysis"
    / "evidence_matrix.json"
)
HVAWT_SMOKE_PATH = REPO / "run_headless" / "config" / "hvawt_response_active_smoke_evidence.json"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_evidence_matrix", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["check_evidence_matrix"] = module
    spec.loader.exec_module(module)
    return module


def write_matrix(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def valid_row(**updates):
    row = {
        "claim_id": "demo_claim",
        "claim": "The demo claim has local evidence.",
        "status": "local_complete",
        "evidence_type": ["test"],
        "current_evidence": "A local test file exists.",
        "evidence_refs": ["evidence.txt"],
        "required_final_artifact": "Archived test output.",
        "blocks_submission": False,
        "notes": "",
    }
    row.update(updates)
    return row


def test_evidence_matrix_checker_accepts_valid_local_row(tmp_path):
    checker = load_checker()
    (tmp_path / "evidence.txt").write_text("ok\n", encoding="utf-8")
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, [valid_row()])

    assert checker.validate_evidence_matrix(matrix, root=tmp_path) == []
    assert checker.main(["--matrix", str(matrix), "--root", str(tmp_path)]) == 0


def test_evidence_matrix_checker_rejects_unmarked_blocker_and_missing_ref(tmp_path):
    checker = load_checker()
    matrix = tmp_path / "matrix.json"
    write_matrix(
        matrix,
        [
            valid_row(
                claim_id="native_callback",
                status="negative_evidence",
                evidence_refs=["missing.txt"],
                blocks_submission=False,
            )
        ],
    )

    issues = checker.validate_evidence_matrix(matrix, root=tmp_path)

    assert {issue.code for issue in issues} == {
        "blocking_status_not_marked",
        "missing_local_reference",
    }


def write_citation_files(
    root: Path,
    *,
    tex: str = "A supported claim \\citep{Alpha,Beta}.\n",
    bib: str = "@article{Alpha,\n}\n@misc{Beta,\n}\n",
    evidence: str | None = None,
) -> None:
    if evidence is None:
        evidence = (
            "# Citation Evidence Table\n\n"
            "| Manuscript claim | Citation keys | Sources checked | What the sources support | Status and action |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| A supported claim. | `Alpha`, `Beta` | Local fixtures. | Supports the claim. | OK. |\n\n"
            "## Required Manuscript Guardrails\n\n"
            "- Do not use background citations as bridge validation.\n"
        )
    (root / "NAWEA2026_QBladeBridge.tex").write_text(tex, encoding="utf-8")
    (root / "sources.bib").write_text(bib, encoding="utf-8")
    (root / "citation_evidence.md").write_text(evidence, encoding="utf-8")


def citation_support_row(**updates):
    return valid_row(
        claim_id="citation_support",
        status="pending_final",
        evidence_type=["citation_table"],
        evidence_refs=["citation_evidence.md", "sources.bib"],
        blocks_submission=True,
        current_evidence="Citation evidence has been checked for local consistency.",
        required_final_artifact="Final citation refresh.",
        **updates,
    )


def test_evidence_matrix_checker_accepts_complete_citation_support_row(tmp_path):
    checker = load_checker()
    write_citation_files(tmp_path)
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, [citation_support_row()])

    assert checker.validate_evidence_matrix(matrix, root=tmp_path) == []


def test_evidence_matrix_checker_rejects_missing_citation_support(tmp_path):
    checker = load_checker()
    write_citation_files(
        tmp_path,
        bib="@article{Alpha,\n}\n",
        evidence=(
            "# Citation Evidence Table\n\n"
            "| Manuscript claim | Citation keys | Sources checked | What the sources support | Status and action |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| A partly supported claim. | `Alpha` | Local fixtures. | Supports part of the claim. | OK. |\n\n"
            "## Required Manuscript Guardrails\n\n"
            "- Do not use background citations as bridge validation.\n"
        ),
    )
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, [citation_support_row()])

    issues = checker.validate_evidence_matrix(matrix, root=tmp_path)

    assert {
        "missing_citation_bib_entry",
        "missing_citation_evidence_key",
    } <= {issue.code for issue in issues}


def test_evidence_matrix_checker_requires_citation_guardrails(tmp_path):
    checker = load_checker()
    write_citation_files(
        tmp_path,
        evidence=(
            "# Citation Evidence Table\n\n"
            "| Manuscript claim | Citation keys | Sources checked | What the sources support | Status and action |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| A supported claim. | `Alpha`, `Beta` | Local fixtures. | Supports the claim. | OK. |\n"
        ),
    )
    matrix = tmp_path / "matrix.json"
    write_matrix(matrix, [citation_support_row()])

    issues = checker.validate_evidence_matrix(matrix, root=tmp_path)

    assert {issue.code for issue in issues} == {"missing_citation_guardrails"}


def test_current_evidence_matrix_is_valid_with_only_external_release_blocking():
    checker = load_checker()

    issues = checker.validate_evidence_matrix(MATRIX_PATH)
    ready_issues = checker.validate_evidence_matrix(MATRIX_PATH, require_submission_ready=True)

    assert issues == []
    blocking_claims = {
        issue.claim_id
        for issue in ready_issues
        if issue.code == "submission_blocked"
    }
    assert blocking_claims == {"archival_and_human_release"}
    assert {
        "native_qblade_callback",
        "native_sil_offline_parity",
        "qblade_geometry_position_proof",
        "baseline_identity",
        "structural_model_qualification",
        "ultimate_margin_products",
        "signed_load_channel_products",
        "production_source_coverage",
        "production_fatigue_trends",
        "spectral_force_content",
        "citation_support",
        "placeholder_free_submission",
    }.isdisjoint(blocking_claims)
    assert {issue.code for issue in ready_issues} == {"submission_blocked"}
    assert (
        checker.main(
            [
                "--matrix",
                str(MATRIX_PATH),
                "--require-submission-ready",
                "--json",
            ]
        )
        == 1
    )


def test_tracked_hvawt_active_smoke_evidence_preserves_claim_boundary():
    evidence = json.loads(HVAWT_SMOKE_PATH.read_text(encoding="utf-8"))

    assert evidence["schema"] == "vorlap-hvawt-response-active-smoke-evidence-v1"
    assert evidence["status"] == "pass"
    assert "does not measure Rayleigh decay" in evidence["claim_boundary"]
    assert all(evidence["checks"].values())
    assert [row["role"] for row in evidence["cases"]] == [
        "off_response",
        "mode6_active_band",
    ]
    off_response, active = evidence["cases"]
    assert off_response["maximum_response_nodal_force_n"] == 0.0
    assert off_response["native_callback_count"] == 0
    assert active["maximum_response_nodal_force_n"] > 0.0
    assert active["mode6_activation_max"] > 0.99
    assert active["mode12_activation_max"] == 0.0
    assert active["native_callback_count"] == 0
