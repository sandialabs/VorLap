import importlib.util
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
    / "check_submission_ready.py"
)


def load_checker():
    spec = importlib.util.spec_from_file_location("check_submission_ready", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["check_submission_ready"] = module
    spec.loader.exec_module(module)
    return module


def test_submission_placeholder_checker_accepts_clean_tex(tmp_path):
    checker = load_checker()
    tex = tmp_path / "clean.tex"
    tex.write_text(
        "\\section{Results}\n"
        "The final production matrix completed without placeholder markers.\n"
        "\\includegraphics{figures/final_fatigue_trends.pdf}\n",
        encoding="utf-8",
    )

    assert checker.scan_tex_placeholders(tex) == []
    assert checker.main(["--tex", str(tex)]) == 0


def test_submission_placeholder_checker_finds_blocking_markers(tmp_path):
    checker = load_checker()
    tex = tmp_path / "draft.tex"
    tex.write_text(
        "\\todoresult{final fatigue ratio}\n"
        "\\tododata{final source-coverage table}\n"
        "\\placeholderfig{production fatigue trend}\n"
        "\\includegraphics{figures/parked_10s_placeholder_del.pdf}\n",
        encoding="utf-8",
    )

    findings = checker.scan_tex_placeholders(tex)

    assert [finding.code for finding in findings] == [
        "todoresult_command",
        "tododata_command",
        "placeholderfig_command",
        "placeholder_figure_include",
    ]
    assert checker.main(["--tex", str(tex), "--json"]) == 1


def test_submission_checker_flags_unsupported_bandwidth_claims(tmp_path):
    checker = load_checker()
    tex = tmp_path / "draft.tex"
    tex.write_text(
        "The campaign includes amplitude/bandwidth sensitivity cases.\n"
        "The method uses an effective frequency-overlap band, $\\Delta f_{\\mathrm{eff}}$.\n"
        "\\placeholderfig{Plot frequency-band or lock-in acceptance assumptions.}\n",
        encoding="utf-8",
    )

    findings = checker.scan_tex_placeholders(tex)

    assert {
        "unsupported_bandwidth_sensitivity",
        "unsupported_overlap_band_parameter",
        "unsupported_lockin_acceptance",
    } <= {finding.code for finding in findings}


def test_submission_checker_flags_short_run_overclaims(tmp_path):
    checker = load_checker()
    tex = tmp_path / "draft.tex"
    tex.write_text(
        "The one-second SIL replay provides final fatigue evidence.\n"
        "The ten-second parked sample is production evidence.\n"
        "The 10 s diagnostic demonstrates fatigue-grade results.\n",
        encoding="utf-8",
    )

    findings = checker.scan_tex_placeholders(tex)

    assert {
        "unsupported_one_second_production_claim",
        "unsupported_ten_second_production_claim",
    } <= {finding.code for finding in findings}


def test_current_nawea_bridge_draft_has_no_submission_placeholders():
    checker = load_checker()
    tex = REPO / "paper_drafts" / "NAWEA_QBladeBridge" / "NAWEA2026_QBladeBridge.tex"

    findings = checker.scan_tex_placeholders(tex)

    assert findings == []
