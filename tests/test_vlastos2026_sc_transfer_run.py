from __future__ import annotations

import csv
import copy
import json
from pathlib import Path

import pytest

from examples.vlastos2026_cantilever import engineering_sc_transfer as transfer
from examples.vlastos2026_cantilever import engineering_sc_transfer_run as runner
from examples.vlastos2026_cantilever import plot_engineering_sc_transfer_results as plotter


@pytest.fixture()
def transfer_campaign(tmp_path: Path) -> tuple[Path, dict]:
    manifest_path = transfer.generate_transfer_campaign(tmp_path / "campaign")
    return manifest_path, transfer.validate_transfer_campaign(manifest_path)


def _readiness(tmp_path: Path) -> tuple[Path, dict]:
    value = {"assessment_fingerprint_sha256": "a" * 64}
    path = tmp_path / "calibration_readiness.json"
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path, value


@pytest.mark.requires_qblade
def test_generated_campaign_binds_runtime_contract(transfer_campaign) -> None:
    _, manifest = transfer_campaign
    contract = manifest["runtime_contract"]
    assert contract["schema"] == "vlastos2026-ambient-response-runtime-contract-v1"
    assert contract["runner"]["path"] == "run_headless/run_qblade_sim.py"
    assert contract["result_filter"]["path"] == "cantilever_results.filter"
    assert "examples/vlastos2026_cantilever/engineering_sc_transfer_run.py" in (
        manifest["implementation_sources"]
    )


@pytest.mark.requires_qblade
def test_execution_plan_binds_transfer_readiness_and_runtime(
    transfer_campaign,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = transfer_campaign
    readiness_path, readiness = _readiness(tmp_path)
    monkeypatch.setattr(
        runner,
        "_passing_readiness",
        lambda _: (
            readiness,
            {"path": str(readiness_path), "sha256": "b" * 64, "bytes": 71},
        ),
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "_validate_v2_results_root",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "_ambient_v2_plan_runtime_identities",
        lambda *args, **kwargs: {"runtime": "pinned"},
    )
    plan = runner.build_execution_plan(
        manifest_path,
        readiness_path,
        tmp_path / "results",
        qblade_root=tmp_path / "qblade",
        checkpoint_every=1000,
        threads=2,
    )
    assert plan["schema"] == runner.PLAN_SCHEMA
    assert plan["execution_intent"] == runner.EXECUTION_INTENT
    assert plan["independent_validation"] is False
    assert plan["transfer_definition_fingerprint_sha256"] == (
        transfer.EXPECTED_DEFINITION_FINGERPRINT_SHA256
    )
    assert plan["campaign_manifest_fingerprint_sha256"] == manifest[
        "manifest_fingerprint_sha256"
    ]
    assert plan["calibration_readiness_fingerprint_sha256"] == "a" * 64
    assert plan["runtime_identities"] == {"runtime": "pinned"}
    assert len(plan["effective_parameter_fingerprints_by_case"]) == 8
    assert len(plan["commands"]) == 8
    assert all(row["role"] == "transfer_tuning_engineering_screen" for row in plan["commands"])
    assert all("--checkpoint-every" in row["argv"] for row in plan["commands"])
    payload = dict(plan)
    payload.pop("plan_fingerprint_sha256")
    assert plan["plan_fingerprint_sha256"] == transfer.canonical_sha256(payload)


def test_execute_requires_acknowledgement() -> None:
    with pytest.raises(RuntimeError, match="explicit acknowledgement"):
        runner.execute_case(
            {},
            Path("manifest.json"),
            Path("readiness.json"),
            case_index=0,
            acknowledge_transfer_tuning_screen=False,
        )


def test_execute_dispatches_only_requested_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "campaign" / "campaign_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}\n", encoding="utf-8")
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text("{}\n", encoding="utf-8")
    commands = [
        {
            "case_id": f"case_{index}",
            "role": "transfer_tuning_engineering_screen",
            "argv": ["python", "run.py", "--output-dir", str(tmp_path / f"case_{index}")],
        }
        for index in range(8)
    ]
    supplied = {"commands": commands, "results_root": str(tmp_path / "results")}
    campaign = {
        "cases": [
            {"case_id": f"case_{index}", "parameter_file": "config.json"}
            for index in range(8)
        ]
    }
    monkeypatch.setattr(runner, "_canonical_plan_from_supplied", lambda *args: supplied)
    monkeypatch.setattr(
        runner.engineering_sc_transfer,
        "validate_transfer_campaign",
        lambda _: campaign,
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "_assert_v2_case_outputs_fresh",
        lambda *args: None,
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "_campaign_tree_snapshot",
        lambda *args: (("frozen",),),
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "_preflight_response_runtime",
        lambda *args: None,
    )
    dispatched: list[list[str]] = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda argv, check: dispatched.append(list(argv)),
    )
    post: list[str] = []
    monkeypatch.setattr(
        runner,
        "_assert_transfer_post_dispatch",
        lambda *args: post.append(args[-1]["case_id"]),
    )
    case_id = runner.execute_case(
        supplied,
        manifest_path,
        readiness_path,
        case_index=3,
        acknowledge_transfer_tuning_screen=True,
    )
    assert case_id == "case_3"
    assert dispatched == [commands[3]["argv"]]
    assert post == ["case_3"]


@pytest.mark.requires_qblade
def test_incomplete_analysis_is_explicitly_not_evaluated(
    transfer_campaign,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = transfer_campaign
    readiness_path, readiness = _readiness(tmp_path)
    monkeypatch.setattr(
        runner,
        "_passing_readiness",
        lambda _: (readiness, {"path": str(readiness_path)}),
    )
    results = tmp_path / "results"
    first = results / manifest["cases"][0]["case_id"]
    first.mkdir(parents=True)
    (first / "result.txt").write_text("partial\n", encoding="utf-8")
    assessment = runner.analyze_results(
        manifest_path,
        readiness_path,
        results,
        tmp_path / "analysis",
    )
    assert assessment["status"] == "not_evaluated"
    assert assessment["evaluated"] is False
    assert assessment["pass"] is False
    assert assessment["completed_case_count"] == 1
    assert len(assessment["missing_case_ids"]) == 7
    assert assessment["independent_validation"] is False
    stored = json.loads(
        (tmp_path / "analysis" / "engineering_sc_transfer_assessment.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored == assessment


@pytest.mark.requires_qblade
def test_complete_acceptance_reuses_declared_engineering_tolerances(
    transfer_campaign,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = transfer_campaign
    readiness = {"assessment_fingerprint_sha256": "a" * 64}
    amplitudes = {2.94: {6.25: 0.40, 6.8: 0.36}, 7.02: {6.2: 0.14, 6.25: 0.13}}
    rows = []
    audits = []
    for entry in manifest["cases"]:
        sc = float(entry["scruton_number"])
        u_star = float(entry["reduced_velocity"])
        amplitude = amplitudes[sc][u_star]
        if entry["timestep_level"] == "dt":
            amplitude += 0.005
        rows.append(
            {
                "case_id": entry["case_id"],
                "scruton_number": sc,
                "role": entry["role"],
                "reduced_velocity": u_star,
                "timestep_level": entry["timestep_level"],
                "tip_peak_amplitude_over_diameter": amplitude,
                "terminal_dominant_frequency_over_fn": 1.0,
                "runtime_audit_evaluated": True,
                "runtime_audit_pass": True,
                "stationarity_evaluated": True,
                "stationarity_pass": True,
            }
        )
        audits.append({"case_id": entry["case_id"]})
    monkeypatch.setattr(
        runner.engineering_holdout,
        "_runtime_audit_pass",
        lambda *args, **kwargs: True,
    )
    assessment = runner._assess_complete_results(
        manifest,
        readiness,
        rows,
        audits,
        manifest_path=manifest_path,
        results_root=tmp_path / "results",
    )
    assert assessment["status"] == "pass"
    assert assessment["pass"] is True
    assert assessment["independent_validation"] is False
    assert assessment["checks"]["all_timestep_pairs_pass"] is True
    assert assessment["checks"]["both_peak_amplitudes_pass"] is True
    tolerances = [
        row["peak_amplitude_tolerance"]
        for row in assessment["by_scruton_number"]
    ]
    assert tolerances == pytest.approx([0.115, 0.055])

    response_csv = tmp_path / "response_by_case.csv"
    with response_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assessment_path = tmp_path / "engineering_sc_transfer_assessment.json"
    assessment_path.write_text(json.dumps(assessment) + "\n", encoding="utf-8")
    figure = plotter.draw_coupled_transfer_screen(
        assessment_path,
        response_csv,
        tmp_path / "engineering_sc_transfer_qblade_screen.pdf",
    )
    assert figure.read_bytes().startswith(b"%PDF")


def test_cli_help_exposes_plan_run_and_analyze(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for command in ("plan", "run", "analyze"):
        assert command in output
