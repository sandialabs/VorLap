from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from examples.vlastos2026_cantilever import engineering_holdout as holdout
from examples.vlastos2026_cantilever import engineering_holdout_run as runner


def _full_plan(tmp_path: Path) -> dict:
    selected = holdout.engineering_screen_spec()["exact_case_ids"]
    commands = [
        {
            "case_id": case_id,
            "role": "held_out",
            "argv": ["python", "run.py", "--output-dir", str(tmp_path / case_id)],
        }
        for case_id in selected
    ]
    commands.append(
        {
            "case_id": "ambientv2_sc2p94_u4p3_dt",
            "role": "held_out",
            "argv": ["python", "run.py"],
        }
    )
    return {
        "results_root": str(tmp_path / "results"),
        "qblade_root": str(tmp_path / "qblade"),
        "checkpoint_every": 0,
        "threads": 4,
        "postfreeze_analytical_gate_status": "fail",
        "commands": commands,
    }


def _patch_plan_inputs(monkeypatch, tmp_path: Path, full_plan: dict) -> tuple[Path, Path]:
    manifest = tmp_path / "campaign_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    readiness = tmp_path / "readiness.json"
    readiness.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_load_passing_readiness",
        lambda path: (
            {"assessment_fingerprint_sha256": "a" * 64},
            {"path": str(readiness), "sha256": "b" * 64, "bytes": 3},
        ),
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "build_run_plan",
        lambda *args, **kwargs: copy.deepcopy(full_plan),
    )
    return manifest, readiness


def test_plan_selects_exact_eight_in_frozen_order(monkeypatch, tmp_path):
    full = _full_plan(tmp_path)
    manifest, readiness = _patch_plan_inputs(monkeypatch, tmp_path, full)
    plan = runner.build_exact_eight_plan(
        manifest,
        readiness,
        tmp_path / "results",
        qblade_root=tmp_path / "qblade",
    )
    expected = holdout.engineering_screen_spec()["exact_case_ids"]
    assert plan["exact_case_ids"] == expected
    assert [row["case_id"] for row in plan["commands"]] == expected
    assert len(plan["commands"]) == 8
    assert plan["strict_hgast_gate_status"] == "fail"
    without_fingerprint = dict(plan)
    without_fingerprint.pop("plan_fingerprint_sha256")
    assert plan["plan_fingerprint_sha256"] == holdout.semantic_sha256(
        without_fingerprint
    )


def test_execute_requires_explicit_failed_gate_acknowledgement():
    with pytest.raises(RuntimeError, match="explicit acknowledgement"):
        runner.execute_exact_eight_plan(
            {},
            Path("manifest.json"),
            Path("readiness.json"),
            acknowledge_failed_strict_hgast_gate=False,
        )


def test_execute_rejects_any_plan_drift_before_dispatch(monkeypatch, tmp_path):
    supplied = {"schema": runner.PLAN_SCHEMA, "tampered": True}
    monkeypatch.setattr(
        runner,
        "_canonical_plan_from_supplied",
        lambda *args, **kwargs: {"schema": runner.PLAN_SCHEMA},
    )
    called = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: called.append(args))
    with pytest.raises(RuntimeError, match="differs from the canonical plan"):
        runner.execute_exact_eight_plan(
            supplied,
            tmp_path / "manifest.json",
            tmp_path / "readiness.json",
            acknowledge_failed_strict_hgast_gate=True,
        )
    assert called == []


def test_execute_dispatches_only_selected_commands(monkeypatch, tmp_path):
    full = _full_plan(tmp_path)
    manifest = tmp_path / "campaign_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    selected = full["commands"][:-1]
    supplied = {
        "results_root": full["results_root"],
        "qblade_root": full["qblade_root"],
        "checkpoint_every": 0,
        "threads": 4,
        "commands": selected,
    }
    campaign = {
        "cases": [
            {"case_id": row["case_id"], "parameter_file": "config.json"}
            for row in selected
        ]
    }
    monkeypatch.setattr(
        runner,
        "_canonical_plan_from_supplied",
        lambda *args, **kwargs: supplied,
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "_load_campaign",
        lambda path: (tmp_path, campaign),
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "build_run_plan",
        lambda *args, **kwargs: full,
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "_validate_ambient_v2_run_plan_shape",
        lambda *args, **kwargs: None,
    )
    fresh_calls = []
    monkeypatch.setattr(
        runner.response_benchmark,
        "_assert_v2_case_outputs_fresh",
        lambda plan, commands: fresh_calls.append([row["case_id"] for row in commands]),
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "_campaign_tree_snapshot",
        lambda root: ("frozen",),
    )
    monkeypatch.setattr(
        runner.response_benchmark,
        "_preflight_response_runtime",
        lambda path: None,
    )
    post = []
    monkeypatch.setattr(
        runner.response_benchmark,
        "_assert_v2_post_dispatch",
        lambda *args: post.append(args[-1]["case_id"]),
    )
    dispatched = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda argv, check: dispatched.append((argv, check)),
    )
    runner.execute_exact_eight_plan(
        supplied,
        manifest,
        tmp_path / "readiness.json",
        acknowledge_failed_strict_hgast_gate=True,
    )
    expected = holdout.engineering_screen_spec()["exact_case_ids"]
    assert [argv[0][-1] for argv in dispatched] == [
        str(tmp_path / case_id) for case_id in expected
    ]
    assert all(check is True for _, check in dispatched)
    assert post == expected
    assert fresh_calls[0] == expected
    assert fresh_calls[1:] == [[case_id] for case_id in expected]


def test_result_root_requires_exact_eight_nonempty_directories(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    for case_id in holdout.engineering_screen_spec()["exact_case_ids"]:
        directory = root / case_id
        directory.mkdir()
        (directory / "result.txt").write_text("data\n", encoding="utf-8")
    runner._require_exact_result_directories(root)
    extra = root / "unexpected"
    extra.mkdir()
    (extra / "result.txt").write_text("data\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        runner._require_exact_result_directories(root)


def test_response_csv_loader_preserves_scientific_types(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text(
        "case_id,scruton_number,reduced_velocity,tip_peak_amplitude_over_diameter,"
        "terminal_dominant_frequency_over_fn,runtime_audit_evaluated,"
        "runtime_audit_pass,stationarity_evaluated,stationarity_pass\n"
        "case,2.94,6.25,0.35,1.01,True,True,True,True\n",
        encoding="utf-8",
    )
    row = runner._load_response_rows(path)[0]
    assert row["scruton_number"] == pytest.approx(2.94)
    assert row["tip_peak_amplitude_over_diameter"] == pytest.approx(0.35)
    assert row["runtime_audit_pass"] is True
    assert row["stationarity_pass"] is True


def test_cli_help_exposes_full_workflow(capsys):
    with pytest.raises(SystemExit) as exc:
        runner._parser().parse_args(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for command in ("readiness", "plan", "execute", "analyze"):
        assert command in output
