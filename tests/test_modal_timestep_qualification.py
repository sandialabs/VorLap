from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "run_headless"
    / "build_modal_timestep_qualification.py"
)


def base_command(out: Path) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--out",
        str(out),
        "--modal-source",
        "reviewed QBlade modal analysis",
        "--modal-method",
        "non-spinning full structural model",
        "--source-frequency-basis",
        "reviewed retained VorLap source bound",
        "--source-frequency-truncation",
        "nyquist",
        "--timestep",
        "0.0075",
    ]
    for frequency in (1, 2, 3, 4, 5):
        command.extend(["--modal-frequency-hz", str(frequency)])
    return command


def test_qualification_passes_only_with_review_and_numerical_margin(tmp_path):
    out = tmp_path / "qualification.json"
    completed = subprocess.run(
        base_command(out)
        + [
            "--source-max-frequency-hz",
            "5",
            "--approved",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert completed.returncode == 0
    assert payload["qualified"] is True
    assert payload["numerical_checks_passed"] is True
    assert payload["modal_frequency_count"] == 5
    assert payload["source_samples_per_cycle_min"] > 10
    assert payload["source_frequency_truncation"] == "nyquist"
    assert len(payload["qualification_sha256"]) == 64


def test_qualification_rejects_current_underresolved_high_frequency_example(tmp_path):
    out = tmp_path / "qualification.json"
    completed = subprocess.run(
        base_command(out)
        + [
            "--source-max-frequency-hz",
            "102.5",
            "--approved",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    issue_codes = {row["code"] for row in payload["issues"]}

    assert completed.returncode == 2
    assert payload["qualified"] is False
    assert "source_timestep_underresolved" in issue_codes
    assert "source_nyquist_margin_too_small" in issue_codes
    assert "modal_band_does_not_cover_source" in issue_codes


def test_qualification_reads_runtime_source_coverage_and_hashes_it(tmp_path):
    coverage = tmp_path / "source_coverage.csv"
    coverage.write_text(
        "scenario,case,component,freq_max_hz\n"
        "vorlap,U05,BLD_1,3.5\n"
        "vorlap,U05,STR_1,4.5\n",
        encoding="utf-8",
    )
    out = tmp_path / "qualification.json"
    subprocess.run(
        base_command(out)
        + [
            "--source-coverage",
            str(coverage),
            "--approved",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["source_max_frequency_hz"] == 4.5
    assert payload["source_frequency_provenance"]["rows_with_frequency"] == 2
    assert len(payload["source_frequency_provenance"]["sha256"]) == 64


def test_qualification_combines_and_hashes_multiple_source_coverage_files(tmp_path):
    low = tmp_path / "source_coverage_U05.csv"
    low.write_text("component,freq_max_hz\nBLD_1,60\nSTR_1_1,61\n", encoding="utf-8")
    high = tmp_path / "source_coverage_U50.csv"
    high.write_text("component,freq_max_hz\nBLD_1,66.6\nSTR_1_1,65\n", encoding="utf-8")
    out = tmp_path / "qualification.json"
    completed = subprocess.run(
        base_command(out)
        + [
            "--source-coverage",
            str(low),
            "--source-coverage",
            str(high),
            "--min-source-samples-per-cycle",
            "2",
            "--min-nyquist-ratio",
            "1",
            "--min-modal-band-ratio",
            "0",
            "--approved",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    provenance = payload["source_frequency_provenance"]

    assert completed.returncode == 0
    assert payload["qualified"] is True
    assert payload["source_max_frequency_hz"] == 66.6
    assert provenance["type"] == "source_coverage_csv_set"
    assert provenance["file_count"] == 2
    assert provenance["rows_with_frequency"] == 4
    assert [record["rows_with_frequency"] for record in provenance["files"]] == [2, 2]
    assert all(len(record["sha256"]) == 64 for record in provenance["files"])


def test_qualification_rejects_duplicate_source_coverage_files(tmp_path):
    coverage = tmp_path / "source_coverage.csv"
    coverage.write_text("component,freq_max_hz\nBLD_1,5\n", encoding="utf-8")
    out = tmp_path / "qualification.json"
    completed = subprocess.run(
        base_command(out)
        + [
            "--source-coverage",
            str(coverage),
            "--source-coverage",
            str(coverage),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "must be unique" in completed.stderr


def test_qualification_reads_selected_owens_campbell_row(tmp_path):
    campbell = tmp_path / "campbell_frequencies.csv"
    campbell.write_text(
        "rpm,mode_01_hz,mode_02_hz,mode_03_hz,mode_04_hz,mode_05_hz\n"
        "0.0,1.0,2.0,3.0,4.0,5.0\n"
        "1.0,1.1,2.1,3.1,4.1,5.1\n",
        encoding="utf-8",
    )
    out = tmp_path / "qualification.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--out",
        str(out),
        "--modal-source",
        "OWENS/GXBeam",
        "--modal-method",
        "rotating changed-geometry model",
        "--campbell-csv",
        str(campbell),
        "--campbell-rpm",
        "1",
        "--source-max-frequency-hz",
        "5",
        "--source-frequency-basis",
        "test bound",
        "--approved",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["modal_frequencies_hz"] == [1.1, 2.1, 3.1, 4.1, 5.1]
    assert payload["modal_frequency_provenance"]["selected_rpm"] == 1.0
    assert len(payload["modal_frequency_provenance"]["sha256"]) == 64
