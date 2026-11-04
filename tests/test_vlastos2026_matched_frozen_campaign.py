import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1]
EXAMPLE = HERE / "examples" / "vlastos2026_cantilever"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_module(
    "vlastos2026_response_benchmark_for_matched_frozen",
    EXAMPLE / "response_benchmark.py",
)
matched = _load_module(
    "vlastos2026_matched_frozen_campaign",
    EXAMPLE / "matched_frozen_campaign.py",
)


@pytest.fixture(scope="module")
def campaigns(tmp_path_factory):
    root = tmp_path_factory.mktemp("matched_frozen")
    parent_manifest = benchmark.generate_campaign(root / "parent")
    frozen_manifest = matched.generate_campaign(root / "frozen", parent_manifest)
    return root, parent_manifest, frozen_manifest


@pytest.mark.requires_qblade
def test_generation_is_exactly_parent_matched_fine_dt_and_plan_is_sequential(campaigns):
    root, parent_manifest, frozen_manifest = campaigns
    state = matched.validate_campaign(frozen_manifest, parent_manifest)
    manifest = state["manifest"]
    parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
    parent_by_id = {
        row["case_id"]: row
        for row in parent["cases"]
        if row["timestep_level"] == "dt_half"
    }

    assert state["case_count"] == 36
    assert manifest["scope"].startswith(
        "legacy frozen-phase bridge comparator amplitude-skill diagnostic"
    )
    assert manifest["qblade_mapping"]["legacy_frozen_phase_runtime_forcing"][
        "strict_open_loop_one_way"
    ] is False
    assert manifest["policy"]["predeclared_cap_natural_cycles"] == 240
    assert manifest["policy"]["terminal_stationarity_window_cycles"] == 40.0
    assert manifest["policy"]["total_requested_steps"] == 2_198_568
    assert manifest["policy"]["maximum_unforced_transient_residual_ratio_after_cap"] == pytest.approx(
        math.exp(-2.0 * math.pi * 0.0027 * 240)
    )
    # Schema-v1 embeds absolute coefficient-source paths. Its file hash changes
    # with checkout location; the physical comparator contract is an exact copy
    # of every parent fine-case source, independently hashed from its bytes.
    parent_source_hashes = {
        hashlib.sha256(
            (parent_manifest.parent / "cases" / case_id / "model/VorLapAirfoils/CYLINDER.h5").read_bytes()
        ).hexdigest()
        for case_id in parent_by_id
    }
    assert len(parent_source_hashes) == 1
    expected_source_hash = parent_source_hashes.pop()
    assert manifest["source_binding"] == {
        "airfoil_id": "CYLINDER",
        "identical_to_every_parent_fine_case": True,
        "source_fingerprint_sha256": expected_source_hash,
    }
    assert manifest["declared_runtime_config_deltas"] == [
        "force_scale",
        "response_model.enabled",
        "response_model.startup_seed.enabled",
    ]

    observed_grid = []
    for entry in manifest["cases"]:
        parent_entry = parent_by_id[entry["case_id"]]
        frozen_source = state["campaign_root"] / "cases" / entry["case_id"] / "model/VorLapAirfoils/CYLINDER.h5"
        assert hashlib.sha256(frozen_source.read_bytes()).hexdigest() == expected_source_hash
        observed_grid.append((entry["scruton_number"], entry["reduced_velocity"]))
        for name in (
            "scruton_number",
            "reduced_velocity",
            "mean_inflow_mps",
            "reynolds_number",
            "timestep_s",
            "steps_per_natural_cycle",
            "structure",
            "sim_path",
            "parameter_file",
        ):
            assert entry[name] == parent_entry[name]
        endpoint = matched.frozen_endpoint_for_cycles(float(entry["scruton_number"]))
        assert entry["timestep_level"] == "dt_half"
        assert entry["timestep_s"] == 0.00025
        assert entry["num_steps"] == endpoint["dt_half_steps"]
        assert entry["aligned_nominal_step_count"] == endpoint["dt_steps"]
        assert entry["num_steps"] == 2 * entry["aligned_nominal_step_count"]
        assert entry["duration_s"] == endpoint["endpoint_time_s"]
        assert entry["actual_endpoint_natural_cycles"] >= 240.0
        assert entry["last_exported_sample_natural_cycles"] >= 240.0
        assert entry["force_scale"] == 1.0
        assert entry["response_model_enabled"] is False
        assert entry["startup_seed_enabled"] is False
        config = json.loads(
            (state["campaign_root"] / entry["parameter_file"]).read_text(
                encoding="utf-8"
            )
        )
        assert config["force_scale"] == 1.0
        assert config["response_model"]["enabled"] is False
        assert config["response_model"]["startup_seed"]["enabled"] is False
        assert config["airfoil_source_manifest"][0]["sha256"] == entry[
            "source_fingerprint_sha256"
        ]
    assert observed_grid == [
        (sc, u_star)
        for sc in sorted(matched.stationary.STRUCTURES)
        for u_star in matched.stationary.REDUCED_VELOCITIES
    ]

    plan = matched.build_run_plan(
        frozen_manifest,
        parent_manifest,
        root / "planned_results",
        qblade_root=HERE / "run_headless" / "QBladeCE_2.0.9.7",
    )
    assert plan["dry_plan"] is True
    assert plan["execution_function_provided"] is True
    assert plan["execution_requires_explicit_cli_gate"] == "run --execute"
    assert plan["execution_order"] == "strictly sequential in sequence_index order"
    assert plan["maximum_concurrent_cases"] == 1
    assert plan["threads_per_qblade_run"] == 4
    assert plan["thread_environment"] == matched.EXECUTION_THREAD_ENVIRONMENT
    assert plan["sil_vorlap_history_mode"] == "compact"
    assert plan["total_requested_steps"] == 2_198_568
    assert len(plan["commands"]) == 36
    assert [row["sequence_index"] for row in plan["commands"]] == list(range(36))
    for row in plan["commands"]:
        argv = row["argv"]
        assert argv[argv.index("--threads") + 1] == "4"
        assert argv[argv.index("--sil-vorlap-history-mode") + 1] == "compact"
        assert "--execute" not in argv
        assert "--checkpoint-every" not in argv


@pytest.mark.requires_qblade
def test_execution_input_bindings_are_cross_mount_portable(
    campaigns, tmp_path, monkeypatch
):
    root, parent_manifest, frozen_manifest = campaigns
    manifest = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    assert manifest["execution_inputs"]["runner"]["path"] == (
        "run_headless/run_qblade_sim.py"
    )
    assert manifest["execution_inputs"]["result_filter"]["path"] == (
        "examples/vlastos2026_cantilever/cantilever_results.filter"
    )

    alternate_root = tmp_path / "work" / "VorLap"
    alternate_here = alternate_root / "examples" / "vlastos2026_cantilever"
    alternate_runner = alternate_root / "run_headless" / "run_qblade_sim.py"
    alternate_runner.parent.mkdir(parents=True)
    alternate_here.mkdir(parents=True)
    alternate_runner.write_bytes(
        (HERE / "run_headless" / "run_qblade_sim.py").read_bytes()
    )
    (alternate_here / "cantilever_results.filter").write_bytes(
        (EXAMPLE / "cantilever_results.filter").read_bytes()
    )
    monkeypatch.setattr(matched, "REPO_ROOT", alternate_root)
    monkeypatch.setattr(matched, "HERE", alternate_here)

    state = matched.validate_campaign(frozen_manifest, parent_manifest)
    assert state["valid"] is True
    plan = matched.build_run_plan(
        frozen_manifest,
        parent_manifest,
        root / "alternate_mount_results",
        qblade_root=HERE / "run_headless" / "QBladeCE_2.0.9.7",
    )
    assert Path(plan["commands"][0]["argv"][1]) == alternate_runner


@pytest.mark.requires_qblade
def test_explicit_executor_is_hash_rebuilt_sequential_and_audited(
    campaigns, tmp_path, monkeypatch
):
    _root, parent_manifest, frozen_manifest = campaigns
    results_root = tmp_path / "executed_results"
    qblade_root = HERE / "run_headless" / "QBladeCE_2.0.9.7"
    plan = matched.build_run_plan(
        frozen_manifest,
        parent_manifest,
        results_root,
        qblade_root=qblade_root,
    )
    calls = []
    preflights = []

    def fake_subprocess_run(argv, *, check, cwd, env):
        assert check is True
        assert Path(cwd) == matched.REPO_ROOT
        assert env["OMP_NUM_THREADS"] == "4"
        assert env["OPENBLAS_NUM_THREADS"] == "1"
        assert env["MKL_NUM_THREADS"] == "1"
        assert env["NUMEXPR_NUM_THREADS"] == "1"
        output = Path(argv[argv.index("--output-dir") + 1])
        output.mkdir(parents=True)
        calls.append(list(argv))

    def fake_runtime_preflight(config_path):
        preflights.append(Path(config_path))

    def fake_runtime_audit(*, entry, result_directory, **_kwargs):
        case_id = str(entry["case_id"])
        identity = {
            "path": str(Path(result_directory) / "synthetic"),
            "sha256": _digest(case_id),
            "bytes": 1,
        }
        return {
            "pass": True,
            "result_identity": identity,
            "run_summary_identity": identity,
            "history_identity": identity,
            "runtime_build_fingerprint": matched._runtime_build_fingerprint(
                _runtime_summary(), case_id=case_id
            ),
        }

    monkeypatch.setattr(matched.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(matched, "_runtime_preflight", fake_runtime_preflight)
    monkeypatch.setattr(matched, "audit_frozen_runtime_case", fake_runtime_audit)

    dry_plan_path = tmp_path / "dry_plan.json"
    assert (
        matched.main(
            [
                "run",
                "--campaign",
                str(frozen_manifest),
                "--parent-campaign",
                str(parent_manifest),
                "--results",
                str(results_root),
                "--plan-output",
                str(dry_plan_path),
                "--qblade-root",
                str(qblade_root),
            ]
        )
        == 0
    )
    assert dry_plan_path.is_file()
    assert calls == []

    tampered = json.loads(json.dumps(plan))
    tampered["commands"][0]["argv"][-1] = "full"
    tampered_payload = dict(tampered)
    tampered_payload.pop("run_plan_payload_sha256")
    tampered["run_plan_payload_sha256"] = matched.sha256_payload(tampered_payload)
    with pytest.raises(matched.MatchedFrozenError, match="no longer matches"):
        matched.execute_run_plan(
            tampered,
            frozen_manifest,
            parent_manifest,
            results_root,
            qblade_root=qblade_root,
        )

    receipt_path = matched.execute_run_plan(
        plan,
        frozen_manifest,
        parent_manifest,
        results_root,
        qblade_root=qblade_root,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert len(calls) == 36
    assert len(set(preflights)) == 36
    assert receipt["completed_case_count"] == 36
    assert receipt["all_runtime_audits_pass"] is True
    assert receipt["maximum_concurrent_cases"] == 1
    assert receipt["threads_per_qblade_run"] == 4
    assert receipt["thread_environment"] == matched.EXECUTION_THREAD_ENVIRONMENT
    assert [row["case_id"] for row in receipt["cases"]] == [
        row["case_id"] for row in plan["commands"]
    ]


def test_generation_and_validation_reject_existing_or_tampered_inputs(campaigns):
    root, parent_manifest, frozen_manifest = campaigns
    with pytest.raises(FileExistsError, match="replace existing campaign"):
        matched.generate_campaign(root / "frozen", parent_manifest)
    with pytest.raises(matched.MatchedFrozenError, match="immutable campaign inputs"):
        matched.generate_campaign(
            parent_manifest.parent / "forbidden_comparator", parent_manifest
        )

    nonempty_results = root / "nonempty_results"
    nonempty_results.mkdir()
    (nonempty_results / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty results root"):
        matched.build_run_plan(
            frozen_manifest,
            parent_manifest,
            nonempty_results,
            qblade_root=HERE / "run_headless" / "QBladeCE_2.0.9.7",
        )
    for forbidden_results in (
        frozen_manifest.parent / "results",
        parent_manifest.parent / "results",
    ):
        with pytest.raises(
            matched.MatchedFrozenError, match="immutable campaign inputs"
        ):
            matched.build_run_plan(
                frozen_manifest,
                parent_manifest,
                forbidden_results,
                qblade_root=HERE / "run_headless" / "QBladeCE_2.0.9.7",
            )
    with pytest.raises(matched.MatchedFrozenError, match="Run-plan output"):
        matched.main(
            [
                "plan",
                "--campaign",
                str(frozen_manifest),
                "--parent-campaign",
                str(parent_manifest),
                "--results",
                str(root / "planned_results"),
                "--output",
                str(frozen_manifest.parent / "forbidden_plan.json"),
            ]
        )
    with pytest.raises(matched.MatchedFrozenError, match="case-results root"):
        matched.main(
            [
                "plan",
                "--campaign",
                str(frozen_manifest),
                "--parent-campaign",
                str(parent_manifest),
                "--results",
                str(root / "planned_results"),
                "--output",
                str(root / "planned_results" / "forbidden_plan.json"),
            ]
        )

    manifest = json.loads(frozen_manifest.read_text(encoding="utf-8"))
    source = (
        frozen_manifest.parent
        / "cases"
        / manifest["cases"][0]["case_id"]
        / "model"
        / "VorLapAirfoils"
        / "CYLINDER.h5"
    )
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"tamper")
        with pytest.raises(matched.MatchedFrozenError, match="identity changed"):
            matched.validate_campaign(frozen_manifest, parent_manifest)
    finally:
        source.write_bytes(original)
    assert matched.validate_campaign(frozen_manifest, parent_manifest)["valid"] is True

    original_manifest = frozen_manifest.read_bytes()
    semantic_tampers = (
        ("scope", lambda value: value.__setitem__("scope", "broadened claim")),
        (
            "parent manifest schema binding",
            lambda value: value["parent_campaign"].__setitem__(
                "manifest_schema", "vlastos2026-response-benchmark-future"
            ),
        ),
        (
            "source binding",
            lambda value: value["source_binding"].__setitem__(
                "identical_to_every_parent_fine_case", False
            ),
        ),
        (
            "policy",
            lambda value: value["policy"].__setitem__(
                "transient_rationale_equation", "unsupported"
            ),
        ),
        (
            "case residual",
            lambda value: value["cases"][0].__setitem__(
                "unforced_structural_transient_residual_ratio_after_cap", 0.0
            ),
        ),
    )
    for _label, mutate in semantic_tampers:
        tampered = json.loads(original_manifest.decode("utf-8"))
        mutate(tampered)
        tampered.pop("manifest_payload_sha256")
        tampered["manifest_payload_sha256"] = matched.sha256_payload(tampered)
        try:
            frozen_manifest.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with pytest.raises(matched.MatchedFrozenError):
                matched.validate_campaign(frozen_manifest, parent_manifest)
        finally:
            frozen_manifest.write_bytes(original_manifest)
    assert matched.validate_campaign(frozen_manifest, parent_manifest)["valid"] is True


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _runtime_summary(*, qblade_sha256: str = "a" * 64, update_count: int = 0):
    bridge_sha256 = "b" * 64
    return {
        "reproducibility": {
            "qblade": {"library": {"sha256": qblade_sha256}},
            "bridge_library": {"sha256": bridge_sha256},
            "vorlap_python_runtime_fingerprint": "c" * 64,
        },
        "native_callback_evidence": {
            "callback_block_present": True,
            "callback_block_error": None,
            "config_error": None,
            "library_file_exists": True,
            "library_symbol_check_available": True,
            "library_symbol_check_error": None,
            "library_missing_required_symbols": [],
            "library_file_hash": {"sha256": bridge_sha256},
            "update_count_observed_max": update_count,
            "update_init_count": 0,
            "update_init_seen": False,
            "update_log_entry_count": 0,
            "runtime_initialized_message_seen": False,
        },
    }


def test_runtime_fingerprint_binds_builds_and_proves_native_callback_zero():
    fingerprint = matched._runtime_build_fingerprint(
        _runtime_summary(), case_id="synthetic"
    )
    assert fingerprint["qblade_library_sha256"] == "a" * 64
    assert fingerprint["bridge_library_sha256"] == "b" * 64
    assert fingerprint["vorlap_python_runtime_fingerprint_sha256"] == "c" * 64
    assert fingerprint["native_callback_zero_pass"] is True
    with pytest.raises(matched.MatchedFrozenError, match="Native callback"):
        matched._runtime_build_fingerprint(
            _runtime_summary(update_count=1), case_id="synthetic"
        )


def _reported_identity(path: Path, portable_path: str) -> dict[str, object]:
    return {
        "path": portable_path,
        "sha256": matched.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_frozen_runtime_fixture(
    *,
    state: dict[str, object],
    result_directory: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    entry = state["manifest"]["cases"][0]
    campaign_root = state["campaign_root"]
    case_id = str(entry["case_id"])
    sim_relative = str(entry["sim_path"])
    config_relative = str(entry["parameter_file"])
    sim_path = campaign_root / sim_relative
    config_path = campaign_root / config_relative
    turbine_raw = matched._read_unique_qblade_setting(sim_path, "TURBFILE")
    turbine_relative = (
        Path(sim_relative).parent / Path(turbine_raw)
    ).as_posix()
    turbine_path = campaign_root / turbine_relative
    result_filter = EXAMPLE / "cantilever_results.filter"
    expected_steps = int(entry["num_steps"])

    result_directory.mkdir(parents=True)
    result_path = result_directory / "result.txt"
    history_path = result_directory / "result_sil_vorlap_history.csv"
    summary_path = result_directory / "synthetic_run_summary.json"
    result_path.write_text("synthetic result\n", encoding="utf-8")
    history_path.write_text("synthetic compact history\n", encoding="utf-8")

    source_row = {field: 0 for field in matched.response._SOURCE_ZERO_FIELDS}
    source_row.update(
        {
            "component": "BLD_1",
            "airfoil_id": "CYLINDER",
            "samples": expected_steps,
            "node_samples": expected_steps * matched.EXPECTED_NODE_COUNT,
            "max_response_nodal_force_n": 0.0,
            "response_force_coefficient_max_abs": 0.0,
        }
    )
    summary = _runtime_summary()
    summary["reproducibility"].update(
        {
            "input_files": {
                "sim": _reported_identity(sim_path, f"/container/{sim_relative}"),
                "vorlap_config": _reported_identity(
                    config_path, f"/container/{config_relative}"
                ),
                "turbine": _reported_identity(
                    turbine_path, f"/container/{turbine_relative}"
                ),
                "result_filter": _reported_identity(
                    result_filter,
                    "/container/examples/vlastos2026_cantilever/"
                    "cantilever_results.filter",
                ),
            },
            "command_line": [
                "/usr/bin/python3",
                "/container/run_headless/run_qblade_sim.py",
                "--sim",
                f"/container/{sim_relative}",
                "--output-dir",
                f"/container/results/{case_id}",
                "--result-name",
                "result",
                "--result-filter",
                "/container/examples/vlastos2026_cantilever/"
                "cantilever_results.filter",
                "--threads",
                str(matched.THREADS_PER_RUN),
                "--sil-vorlap",
                "--coupling-mode",
                "sil-vorlap",
                "--sil-vorlap-history-mode",
                "compact",
            ],
            "loader_environment": {
                "OMP_NUM_THREADS": str(matched.THREADS_PER_RUN)
            },
            "qblade_case_settings": {
                "MEANINF": str(entry["mean_inflow_mps"]),
                "TIMESTEP": str(entry["timestep_s"]),
                "NUMTIMESTEPS": str(expected_steps),
                "TURBFILE": turbine_raw,
            },
        }
    )
    summary.update(
        {
            "sim": f"/container/{sim_relative}",
            "sil_vorlap_parameter_file": f"/container/{config_relative}",
            "qblade_execution_mode": "step-loop",
            "coupling_mode": "sil-vorlap",
            "sil_vorlap": True,
            "sil_vorlap_updates": expected_steps,
            "sil_constant_force": False,
            "sil_constant_force_updates": 0,
            "sil_constant_force_actions": [],
            "sil_constant_force_max_abs_force": None,
            "force_sign_probe_observations": [],
            "failed": False,
            "stopped": False,
            "completed_steps": expected_steps,
            "requested_steps": expected_steps,
            "threads": matched.THREADS_PER_RUN,
            "use_opencl": False,
            "result_file": f"/container/results/{case_id}/result.txt",
            "result_file_identity": _reported_identity(
                result_path, f"/container/results/{case_id}/result.txt"
            ),
            "sil_vorlap_source_coverage": [source_row],
            "sil_vorlap_response_diagnostics_summary": {
                "model_type": "modal_negative_damping",
                "samples": 0,
                "max_response_nodal_force_n": 0.0,
                "gain_limited_steps": 0,
                "force_limited_steps": 0,
                "gain_limited_node_samples": 0,
                "force_limited_node_samples": 0,
            },
            "sil_vorlap_history_mode": "compact",
            "sil_vorlap_history": {
                "mode": "compact",
                "schema": matched.COMPACT_HISTORY_SCHEMA,
                "vector_digest_schema": matched.FORCE_VECTOR_DIGEST_SCHEMA,
                "streaming": True,
                "retained_rows": 0,
                "discarded_uncheckpointed_tail_bytes": 0,
                "node_count": matched.EXPECTED_NODE_COUNT,
                "rows": expected_steps,
                "columns": 1,
                "path": f"/container/results/{case_id}/{history_path.name}",
                "file_identity": _reported_identity(
                    history_path,
                    f"/container/results/{case_id}/{history_path.name}",
                ),
            },
            "qblade_export_time_diagnostics": {"samples": expected_steps},
        }
    )
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return entry, summary, summary_path


def test_runtime_audit_rejects_wrong_source_response_and_execution_semantics(
    campaigns, tmp_path, monkeypatch
):
    _root, parent_manifest, frozen_manifest = campaigns
    state = matched.validate_campaign(frozen_manifest, parent_manifest)
    entry, valid_summary, summary_path = _write_frozen_runtime_fixture(
        state=state,
        result_directory=tmp_path / "frozen_runtime",
    )
    expected_steps = int(entry["num_steps"])
    monkeypatch.setattr(
        matched,
        "audit_compact_frozen_history",
        lambda *_args, **_kwargs: {
            "pass": True,
            "samples": expected_steps,
            "columns": 1,
        },
    )

    def run_audit():
        return matched.audit_frozen_runtime_case(
            campaign_root=state["campaign_root"],
            file_index=state["file_index"],
            entry=entry,
            result_directory=summary_path.parent,
        )

    assert run_audit()["pass"] is True

    adversarial_mutations = (
        ("native full-run mode", lambda value: value.__setitem__("qblade_execution_mode", "run-full-simulation")),
        ("missing SIL loop", lambda value: value.__setitem__("sil_vorlap", False)),
        ("constant force enabled", lambda value: value.__setitem__("sil_constant_force", True)),
        ("constant force updates", lambda value: value.__setitem__("sil_constant_force_updates", 1)),
        ("constant force actions", lambda value: value.__setitem__("sil_constant_force_actions", [{"force": 1.0}])),
        ("constant force maximum", lambda value: value.__setitem__("sil_constant_force_max_abs_force", 1.0)),
        ("constant force observations", lambda value: value.__setitem__("force_sign_probe_observations", [{"observed": True}])),
        (
            "constant force command",
            lambda value: value["reproducibility"]["command_line"].append(
                "--sil-constant-force"
            ),
        ),
        ("wrong component", lambda value: value["sil_vorlap_source_coverage"][0].__setitem__("component", "BLD_2")),
        ("wrong airfoil", lambda value: value["sil_vorlap_source_coverage"][0].__setitem__("airfoil_id", "NACA0018")),
        ("wrong step samples", lambda value: value["sil_vorlap_source_coverage"][0].__setitem__("samples", expected_steps - 1)),
        ("wrong node samples", lambda value: value["sil_vorlap_source_coverage"][0].__setitem__("node_samples", expected_steps)),
        ("response force present", lambda value: value["sil_vorlap_source_coverage"][0].__setitem__("max_response_nodal_force_n", 1.0e-12)),
        ("response coefficient present", lambda value: value["sil_vorlap_source_coverage"][0].__setitem__("response_force_coefficient_max_abs", 1.0e-12)),
        (
            "extra source row",
            lambda value: value["sil_vorlap_source_coverage"].append(
                dict(value["sil_vorlap_source_coverage"][0])
            ),
        ),
        ("response model samples", lambda value: value["sil_vorlap_response_diagnostics_summary"].__setitem__("samples", 1)),
        ("response summary force", lambda value: value["sil_vorlap_response_diagnostics_summary"].__setitem__("max_response_nodal_force_n", 1.0e-12)),
    )
    for _label, mutate in adversarial_mutations:
        tampered = json.loads(json.dumps(valid_summary, allow_nan=False))
        mutate(tampered)
        summary_path.write_text(
            json.dumps(tampered, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(matched.MatchedFrozenError):
            run_audit()

    summary_path.write_text(
        json.dumps(valid_summary, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert run_audit()["pass"] is True


def _compact_history_rows(path: Path, *, tamper_lag: bool = False):
    node_ids = [f"BLD_1_{index / 20:.6f}" for index in range(21)]
    component_count = 63
    zero_prefix = (
        matched.FORCE_VECTOR_DIGEST_SCHEMA.encode("ascii")
        + b"\0"
        + struct.pack("!Q", component_count)
    )
    zero_digest = hashlib.sha256(
        zero_prefix + struct.pack("!63d", *([0.0] * 63))
    ).hexdigest()
    generated = [_digest(f"generated-{index}") for index in range(3)]
    rows = []
    for index in range(3):
        row = {
            "force_history_schema": matched.COMPACT_HISTORY_SCHEMA,
            "force_vector_digest_schema": matched.FORCE_VECTOR_DIGEST_SCHEMA,
            "force_vector_component_count": component_count,
            "step_index": index,
            "time_s": 0.01 * index,
            "generated_step_index": index,
            "generated_time_s": 0.01 * index,
            "applied_source_step_index": "" if index == 0 else index - 1,
            "applied_source_time_s": "" if index == 0 else 0.01 * (index - 1),
            "applied_startup_zero": index == 0,
            "applied_force_lag_steps": 1,
            "qblade_timestep": index,
            "azimuth_deg": 0.0,
            "generated_force_vector_sha256": generated[index],
            "applied_force_vector_sha256": (
                zero_digest if index == 0 else generated[index - 1]
            ),
            "structural_velocity_vector_sha256": _digest(f"velocity-{index}"),
            "applied_force_dot_structural_velocity_w": 0.0,
            "applied_power_sign_convention": matched.APPLIED_POWER_SIGN_CONVENTION,
        }
        for prefix in ("generated_", "applied_"):
            row[prefix + "max_force_node_id"] = node_ids[-1]
            for name in matched.COMPACT_FORCE_METRICS:
                row[prefix + name] = 0.1 if name.startswith("max_abs") else 0.0
        rows.append(row)
    if tamper_lag:
        rows[2]["applied_force_vector_sha256"] = _digest("wrong-source")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return node_ids


def test_compact_frozen_history_binds_exact_lag_and_rejects_tampering(tmp_path):
    valid_path = tmp_path / "valid.csv"
    node_ids = _compact_history_rows(valid_path)
    audit = matched.audit_compact_frozen_history(
        valid_path,
        timestep_s=0.01,
        expected_samples=3,
        node_ids=node_ids,
    )
    assert audit["pass"] is True
    assert audit["samples"] == 3
    assert audit["force_vector_component_count"] == 63
    assert audit["one_step_lag_digest_pass"] is True
    assert audit["response_columns_absent"] is True

    tampered_path = tmp_path / "tampered.csv"
    _compact_history_rows(tampered_path, tamper_lag=True)
    with pytest.raises(matched.MatchedFrozenError, match="lag digest changed"):
        matched.audit_compact_frozen_history(
            tampered_path,
            timestep_s=0.01,
            expected_samples=3,
            node_ids=node_ids,
        )


def test_analysis_exposes_matched_skill_but_never_fatigue_conservatism(
    campaigns, tmp_path, monkeypatch
):
    _root, parent_manifest, frozen_manifest = campaigns
    state = matched.validate_campaign(frozen_manifest, parent_manifest)
    parent = state["parent_manifest"]
    fine_by_id = {
        row["case_id"]: row
        for row in parent["cases"]
        if row["timestep_level"] == "dt_half"
    }
    frozen_results = tmp_path / "frozen_results"
    response_results = tmp_path / "response_results"
    entry_by_id = {}
    for frozen_entry in state["manifest"]["cases"]:
        case_id = frozen_entry["case_id"]
        entry_by_id[case_id] = frozen_entry
        for root in (frozen_results, response_results):
            directory = root / case_id
            directory.mkdir(parents=True)
            (directory / "result.txt").write_text("synthetic\n", encoding="utf-8")
            (directory / "synthetic_run_summary.json").write_text(
                json.dumps(_runtime_summary(), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (directory / "synthetic_sil_vorlap_history.csv").write_text(
                "synthetic\n", encoding="utf-8"
            )

    mismatch_runtime = {"enabled": True}

    def fake_frozen_audit(*, entry, result_directory, **_kwargs):
        result = result_directory / "result.txt"
        summary = result_directory / "synthetic_run_summary.json"
        history = result_directory / "synthetic_sil_vorlap_history.csv"
        runtime_summary = _runtime_summary(
            qblade_sha256="d" * 64
            if mismatch_runtime["enabled"]
            else "a" * 64
        )
        return {
            "pass": True,
            "result_identity": matched._identity(result),
            "run_summary_identity": matched._identity(summary),
            "history_identity": matched._identity(history),
            "runtime_build_fingerprint": matched._runtime_build_fingerprint(
                runtime_summary, case_id=str(entry["case_id"])
            ),
        }

    def fake_response_audit(result_directory, **_kwargs):
        return {
            "evaluated": True,
            "provenance_pass": True,
            "execution_pass": True,
            "source_coverage_complete": True,
            "source_limits_pass": True,
            "response_summary_limits_pass": True,
            "startup_seed_summary_pass": True,
            "startup_seed_response": {"pass": True},
            "history": {
                "runtime_model_pass": True,
                "limit_headroom_pass": True,
                "compact_lag_provenance_pass": True,
                "startup_seed_integrity_pass": True,
                "response_estimator_stationarity_pass": True,
            },
            "summary_path": str(result_directory / "synthetic_run_summary.json"),
            "history_path": str(
                result_directory / "synthetic_sil_vorlap_history.csv"
            ),
            "problems": [],
        }

    figure9 = matched.response._reference_path("sc5p18_figure9_digitized.csv")
    hgast_u, hgast_a = matched.response._read_figure9_series(
        figure9, matched.response.CALIBRATION_SERIES
    )

    def response_curve(sc, u_star):
        if sc == 5.18:
            return float(np.interp(u_star, hgast_u, hgast_a))
        if sc == 2.94:
            return 0.54 * math.exp(-((u_star - 6.8) / 0.55) ** 2)
        return 0.15 * math.exp(-((u_star - 6.2) / 0.55) ** 2)

    def fake_amplitude(result_path, **_kwargs):
        case_id = Path(result_path).parent.name
        entry = entry_by_id[case_id]
        sc = float(entry["scruton_number"])
        u_star = float(entry["reduced_velocity"])
        is_frozen = "frozen_results" in Path(result_path).parts
        amplitude = 0.015 if is_frozen else response_curve(sc, u_star)
        return {
            "tip_peak_amplitude_over_diameter": amplitude,
            "terminal_dominant_frequency_over_fn": (
                0.2 * u_star if is_frozen else 1.0
            ),
            "terminal_mode_band_power_fraction": 0.02 if is_frozen else 0.98,
            "stationarity_pass": True,
        }

    monkeypatch.setattr(matched, "audit_frozen_runtime_case", fake_frozen_audit)
    monkeypatch.setattr(matched.response, "runtime_case_audit", fake_response_audit)
    monkeypatch.setattr(
        matched.response, "response_amplitude_from_result", fake_amplitude
    )

    for protected_output in (
        frozen_results / "forbidden_analysis",
        response_results / "forbidden_analysis",
    ):
        with pytest.raises(
            matched.MatchedFrozenError, match="results input"
        ):
            matched.analyze_campaign(
                frozen_manifest,
                parent_manifest,
                frozen_results,
                (response_results,),
                protected_output,
            )

    with pytest.raises(
        matched.MatchedFrozenError, match="runtime build differs"
    ):
        matched.analyze_campaign(
            frozen_manifest,
            parent_manifest,
            frozen_results,
            (response_results,),
            tmp_path / "mismatched_analysis",
        )
    mismatch_runtime["enabled"] = False
    output = tmp_path / "analysis"
    summary_path = matched.analyze_campaign(
        frozen_manifest,
        parent_manifest,
        frozen_results,
        (response_results,),
        output,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "diagnostic_complete"
    assert summary["runtime_evidence"]["completed_case_count"] == 36
    assert summary["runtime_evidence"]["exact_sc_u_fine_dt_co_location_pass"] is True
    assert summary["runtime_evidence"]["exact_runtime_build_co_location_pass"] is True
    assert summary["runtime_evidence"]["single_runtime_build_across_all_cases_pass"] is True
    assert summary["runtime_evidence"]["native_callback_zero_pass"] is True
    assert summary["runtime_evidence"]["all_terminal_stationarity_pass"] is True
    assert summary["diagnostic_skill"]["sc5p18_hgast_curve"]["evaluated"] is True
    assert (
        summary["diagnostic_skill"]["sc5p18_hgast_curve"][
            "skill_over_legacy_frozen_phase"
        ]
        > 0.9
    )
    assert (
        summary["diagnostic_skill"][
            "hgast_all_sc_peak_diagnostic_gate_pass"
        ]
        is True
    )
    assert (
        summary["diagnostic_skill"][
            "hgast_all_sc_peak_diagnostic_gate_is_validation"
        ]
        is False
    )
    assert summary["diagnostic_skill"]["figure10_experiment_rows_enter_gate"] is False
    assert len(summary["peak_reference_comparisons"]) == 6
    assert summary["claim_boundary"] == {
        "certification_suitability_claimed": False,
        "experimental_validation_claimed": False,
        "fatigue_conservative_validation": False,
        "fatigue_conservatism_evaluated": False,
        "reason": (
            "tip-amplitude skill does not establish local stress ranges, cycle counts, DEL, "
            "Miner damage, conservative fatigue bias, or experimental validation from "
            "non-co-located Figure 10 amplitude-only rows"
        ),
    }
    rows = list(
        csv.DictReader(
            (output / "legacy_frozen_phase_comparison_by_case.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert len(rows) == 36
    assert all(row["co_location_pass"] == "True" for row in rows)
    assert all(row["runtime_build_co_location_pass"] == "True" for row in rows)
    held_out = [
        row
        for row in summary["peak_reference_comparisons"]
        if row["role"] == "held_out_diagnostic"
    ]
    assert held_out
    assert all(
        "non-co-located held-out model-parameter peak-location metadata"
        in row["reference_location_source"]
        for row in held_out
    )
    assert all(row["reference_amplitude_location_colocated"] is False for row in held_out)
    assert all(row["figure10_amplitude_only_non_colocated"] is True for row in held_out)
    assert all(
        row["eligible_for_experimental_validation_claim"] is False
        and row["experimental_validation_assessment"]
        == "not_evaluated_non_colocated_figure10_amplitude_only"
        for row in held_out
    )
    held_out_experiment = [
        row for row in held_out if row["reference_series"] == "experiment_lupi2021"
    ]
    assert held_out_experiment
    assert all(
        row["eligible_for_hgast_all_sc_peak_diagnostic_gate"] is False
        for row in held_out_experiment
    )
    assert all("pass" not in key for row in held_out_experiment for key in row)
    assert (output / "legacy_frozen_phase_reference_peak_skill.csv").is_file()
    with pytest.raises(FileExistsError, match="nonempty analysis output"):
        matched.analyze_campaign(
            frozen_manifest,
            parent_manifest,
            frozen_results,
            (response_results,),
            output,
        )
