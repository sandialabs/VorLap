from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import pytest

from vorlap.qblade_bridge_config import QBladeBridgeConfig
from vorlap.qblade_bridge_service import (
    QBladeBridgeRunServiceResult,
    QBladeBridgeService,
    _maybe_apply_reference_stable_support_patch,
    find_qblade_gui_executable,
    write_run_geometry_audit_reported_positions,
)
from vorlap.qblade_case_prep import QBladeCasePrepRequest, QBladeCasePrepResult
from vorlap.qblade_files import read_qblade_keyword
from vorlap.qblade_nodes import position_channel_name
from vorlap.qblade_run import QBladeRunRequest, QBladeRunResult
from vorlap.qblade_runtime import GEOMETRY_AUDIT_COLUMNS


def bridge_config(tmp_path: Path, **overrides: object) -> QBladeBridgeConfig:
    qblade_root = tmp_path / "QBladeCE"
    values: dict[str, object] = {
        "sim_path": tmp_path / "case.sim",
        "qblade_root": qblade_root,
        "output_dir": tmp_path / "out",
        "airfoil_dir": tmp_path / "airfoils",
        "result_name": "case_results",
        "n_freq_depth": 20,
        "force_scale": 1.0,
        "include_mean_load": True,
        "coupling_mode": "sil-vorlap",
        "steps": 4,
        "threads": 2,
        "manage_loader_env": False,
    }
    values.update(overrides)
    return QBladeBridgeConfig(**values).validated()


def write_minimal_qblade_model(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    aero_dir = model_dir / "Aero"
    structure_dir = model_dir / "Structure"
    aero_dir.mkdir()
    structure_dir.mkdir()
    sim = model_dir / "case.sim"
    trb = model_dir / "case.trb"
    bld = aero_dir / "case.bld"
    structural = structure_dir / "case.str"
    sim.write_text(
        "case.trb TURBFILE - turbine file\n"
        "5 MEANINF - mean inflow\n"
        "0 INITIAL_AZIMUTH - initial azimuth\n"
        "0 PRESCRIBETYPE - rpm prescribe mode\n"
        "0 RPMPRESCRIBED - rpm\n"
        "100 NUMTIMESTEPS - steps\n"
        "0.1 TIMESTEP - timestep\n"
        "0 STOREFROM - store from\n"
        "false STOREREPLAY - store replay\n",
        encoding="utf-8",
    )
    trb.write_text(
        "Aero/case.bld BLADEFILE - blade file\n"
        "1 NUMBLADES - blades\n"
        "Structure/case.str STRUCTURALFILE - structural file\n",
        encoding="utf-8",
    )
    bld.write_text(
        "1 NUMBLADES - blades\n"
        "HEIGHT_[m] CHORD_[m] RADIUS_[m] TOFFSET_[m] TWIST_[deg] CIRCANGLE_[deg] P_AXIS_[-] POLAR_FILE\n"
        "0.0 0.10 0.50 0.00 0.0 0.0 0.25 Polars/NACA_0018.plr\n"
        "1.0 0.10 0.50 0.00 0.0 0.0 0.25 Polars/NACA_0018.plr\n",
        encoding="utf-8",
    )
    structural.write_text("BLD_1_0.000000\nBLD_1_1.000000\n", encoding="utf-8")
    return sim


def write_airfoil_name(airfoil_dir: Path, name: str = "NACA0018") -> None:
    airfoil_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(airfoil_dir / f"{name}.h5", "w") as h5:
        h5.create_dataset("Airfoilname", data=name.encode("utf-8"))


def fake_prep_result(request: QBladeCasePrepRequest, tmp_path: Path) -> QBladeCasePrepResult:
    config = tmp_path / "case" / "Control" / "vorlap_qblade_external.json"
    audit = config.with_name("vorlap_qblade_geometry_audit.csv")
    turbine = tmp_path / "case" / "case.trb"
    structural = tmp_path / "case" / "case.str"
    sim = Path(request.sim_path)
    return QBladeCasePrepResult(
        sim_path=sim,
        turbine_path=turbine,
        structural_path=structural,
        parameter_file_path=config,
        geometry_audit_path=audit,
        sim_path_for_config="case.sim",
        airfoil_dir_for_config="airfoils",
        node_source=request.node_source,
        node_ids=("BLD_1_0.000000", "BLD_1_1.000000"),
        structural_node_ids=("BLD_1_0.000000", "BLD_1_1.000000"),
        converted_node_ids=("BLD_1_0.000000",),
        convert_include_struts=request.include_struts,
        convert_include_tower=request.include_tower,
        swap_size=15,
        component_count=1,
        runtime_config={
            "force_scale": request.force_scale,
            "include_mean_load": request.include_mean_load,
            "source_bounds_policy": request.source_bounds_policy,
        },
        planned_write_paths=(config, turbine, structural, audit),
        written_files=() if request.dry_run else (config, audit, turbine, structural),
        dry_run=request.dry_run,
    )


def fake_run_result(request: QBladeRunRequest, tmp_path: Path) -> QBladeRunResult:
    if request.message_callback is not None:
        request.message_callback("Completed 4/4 timesteps")
    summary_path = Path(request.output_dir) / "case_run_summary.json"
    result_path = Path(request.output_dir) / "case_results.txt"
    run_log_path = Path(request.output_dir) / "case_qblade.log"
    summary = {
        "requested_steps": request.steps,
        "completed_steps": request.steps,
        "stopped": False,
        "stop_message": None,
        "failed": False,
        "failure_message": None,
        "coupling_mode": request.coupling_mode,
        "qblade_execution_mode": request.qblade_execution_mode,
        "result_file": str(result_path),
        "run_log": str(run_log_path),
        "sil_vorlap_history": str(Path(request.output_dir) / "case_history.csv"),
        "sil_vorlap_message": "VorLap QBlade runtime initialized.",
        "sil_vorlap_max_abs_force": 42.0,
        "sil_vorlap_velocity_diagnostics": {
            "axis_stats": {
                "X": {"samples": 4, "max_abs": 1.2},
                "Y": {"samples": 4, "max_abs": 0.1},
                "Z": {"samples": 4, "max_abs": 0.0},
            },
            "all_zero_axes": ["Z"],
            "all_zero_channels": ["Z_g Position BLD_1_0.000000 [m/s]"],
        },
        "sil_vorlap_timing_diagnostics": {"updates": 4},
        "sil_vorlap_cache_stats": {"hits": 3, "misses": 1},
        "sil_vorlap_source_bounds_policy": "raise",
        "sil_vorlap_source_coverage": [
            {
                "component": "BLD_1",
                "samples": 4,
                "node_samples": 8,
                "aoa_min_deg": -2.5,
                "aoa_max_deg": 4.0,
                "max_structural_velocity_mps": 1.1,
                "source_clip_count": 0,
            }
        ],
        "native_callback_evidence": {"log_file": str(Path(request.output_dir) / "bridge.log")},
        "qblade_export_time_diagnostics": {"available": True},
        "restart": {
            "latest_checkpoint": str(Path(request.output_dir) / "checkpoints" / "latest_checkpoint.json"),
            "latest_checkpoint_completed_steps": request.steps,
            "latest_checkpoint_project": str(Path(request.output_dir) / "checkpoints" / "case_step_0000004.qpr"),
            "resume_requested": bool(request.resume_from_checkpoint or request.resume_checkpoint),
            "resume_start_step": 2 if request.resume_from_checkpoint else 0,
        },
    }
    return QBladeRunResult(
        summary=summary,
        summary_path=summary_path,
        result_path=result_path,
        run_log_path=run_log_path,
    )


def test_bridge_service_validation_can_report_qblade_install_errors(tmp_path):
    root = tmp_path / "QBladeCE"
    root.mkdir()
    service = QBladeBridgeService()
    config = bridge_config(tmp_path, qblade_root=root)

    result = service.validate_config(config, check_qblade_install=True)

    assert result.ok is False
    assert len(result.errors) == 1
    assert "Incomplete QBlade root" in result.errors[0]
    assert "missing QBlade SIL_Interface directory" in result.errors[0]
    assert "missing QBlade ControllerFiles directory" in result.errors[0]
    assert "missing QBlade shared library" in result.errors[0]


def test_bridge_service_validation_reports_broken_qblade_model_references(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("missing.trb TURBFILE - turbine file\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    airfoils.mkdir()
    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        airfoil_dir=airfoils,
        output_dir=tmp_path / "out",
    )
    service = QBladeBridgeService()

    result = service.validate_config(config, check_paths=True, check_qblade_model=True)

    assert result.ok is False
    assert result.errors == (f"QBlade .sim TURBFILE does not exist: {(tmp_path / 'missing.trb').resolve()}",)


def test_bridge_service_validation_reports_missing_airfoil_fft_files(tmp_path):
    sim = write_minimal_qblade_model(tmp_path / "model")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    airfoils.mkdir()
    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        airfoil_dir=airfoils,
        output_dir=tmp_path / "out",
    )
    service = QBladeBridgeService()

    result = service.validate_config(config, check_paths=True, check_qblade_model=True)

    assert result.ok is False
    assert result.errors == (f"No airfoil FFT .h5 files were found in: {airfoils.resolve()}",)


def test_bridge_service_validation_reports_duplicate_structural_node_ids(tmp_path):
    model_dir = tmp_path / "model"
    sim = write_minimal_qblade_model(model_dir)
    (model_dir / "Structure" / "case.str").write_text(
        "BLD_1_0.000000\n"
        "BLD_1_0.000000\n"
        "BLD_1_1.000000\n",
        encoding="utf-8",
    )
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    write_airfoil_name(airfoils)
    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        airfoil_dir=airfoils,
        output_dir=tmp_path / "out",
    )
    service = QBladeBridgeService()

    result = service.validate_config(config, check_paths=True, check_qblade_model=True)

    assert result.ok is False
    assert len(result.errors) == 1
    assert "Duplicate structural node id(s)" in result.errors[0]
    assert "BLD_1_0.000000" in result.errors[0]


def test_bridge_service_prepare_case_returns_gui_artifacts_and_diagnostics(tmp_path):
    captured: list[QBladeCasePrepRequest] = []

    def prepare(request: QBladeCasePrepRequest) -> QBladeCasePrepResult:
        captured.append(request)
        return fake_prep_result(request, tmp_path)

    config = bridge_config(tmp_path, include_tower=True)
    service = QBladeBridgeService(case_preparer=prepare)

    result = service.prepare_case(config, dry_run=True)

    assert captured
    assert captured[0].include_mean_load is True
    assert captured[0].include_tower is True
    assert captured[0].dry_run is True
    assert result.state == "planned"
    assert result.artifacts["runtime_config"].endswith("vorlap_qblade_external.json")
    assert result.artifacts["geometry_audit"].endswith("vorlap_qblade_geometry_audit.csv")
    assert result.diagnostics["node_count"] == 2
    assert result.diagnostics["swap_size"] == 15
    assert result.diagnostics["dry_run"] is True
    assert result.diagnostics["geometry_audit_exists"] is False
    assert result.diagnostics["geometry_audit_duplicate_node_count"] == 0


def test_bridge_service_prepare_case_summarizes_geometry_audit_status(tmp_path):
    def prepare(request: QBladeCasePrepRequest) -> QBladeCasePrepResult:
        result = fake_prep_result(request, tmp_path)
        audit = result.geometry_audit_path
        assert audit is not None
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(
            "requested_node_id,basis_status,basis_warnings\n"
            "BLD_1_0.000000,ok,\n"
            "BLD_1_1.000000,bad_basis,chord_dot_normal=0.2\n",
            encoding="utf-8",
        )
        return result

    config = bridge_config(tmp_path)
    service = QBladeBridgeService(case_preparer=prepare)

    result = service.prepare_case(config)

    assert result.diagnostics["geometry_audit_exists"] is True
    assert result.diagnostics["geometry_audit_rows"] == 2
    assert result.diagnostics["geometry_audit_basis_error_count"] == 1
    assert result.diagnostics["geometry_audit_basis_warning_count"] == 1
    assert result.diagnostics["geometry_audit_basis_error_nodes"] == ["BLD_1_1.000000"]
    assert result.diagnostics["geometry_audit_duplicate_node_count"] == 0


def test_bridge_service_prepare_case_copies_model_without_overrides_by_default(tmp_path):
    model_dir = tmp_path / "model"
    sim = write_minimal_qblade_model(model_dir)
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    write_airfoil_name(airfoils)
    captured: list[QBladeCasePrepRequest] = []

    def prepare(request: QBladeCasePrepRequest) -> QBladeCasePrepResult:
        captured.append(request)
        return fake_prep_result(request, tmp_path)

    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        output_dir=tmp_path / "out",
        airfoil_dir=airfoils,
        duration_s=None,
        timestep_s=None,
        steps=None,
    )
    service = QBladeBridgeService(case_preparer=prepare)

    result = service.prepare_case(config, check_paths=True)

    assert captured
    working_sim = Path(result.artifacts["working_sim"])
    manifest_path = Path(result.artifacts["working_copy_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert captured[0].sim_path == working_sim
    assert result.config.sim_path == working_sim
    assert working_sim.is_file()
    assert Path(result.artifacts["source_sim"]) == sim.resolve()
    assert manifest["source_sim"] == str(sim.resolve())
    assert manifest["working_sim"] == str(working_sim)
    assert manifest["qblade_keyword_patches"] == []
    assert manifest["case_preparation_planned_write_paths"]
    assert manifest["case_preparation_written_files"]
    assert all(row["hash"] is None or row["hash"]["sha256"] for row in manifest["case_preparation_written_hashes"])
    assert {row["relative_path"] for row in manifest["copied_files"]} >= {
        "case.sim",
        "case.trb",
        "Aero/case.bld",
        "Structure/case.str",
    }
    assert result.diagnostics["qblade_model_override_count"] == 0
    assert result.diagnostics["working_copy_file_count"] >= 4
    assert result.diagnostics["original_model_preserved"] is True


def test_bridge_service_prepare_case_can_patch_original_when_explicitly_requested(tmp_path):
    model_dir = tmp_path / "model"
    sim = write_minimal_qblade_model(model_dir)
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    write_airfoil_name(airfoils)
    captured: list[QBladeCasePrepRequest] = []

    def prepare(request: QBladeCasePrepRequest) -> QBladeCasePrepResult:
        captured.append(request)
        return fake_prep_result(request, tmp_path)

    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        output_dir=tmp_path / "out",
        airfoil_dir=airfoils,
        patch_original_model=True,
    )
    service = QBladeBridgeService(case_preparer=prepare)

    result = service.prepare_case(config, check_paths=True)

    assert captured
    assert captured[0].sim_path == sim
    assert result.config.sim_path == sim
    assert "working_sim" not in result.artifacts
    assert "working_copy_manifest" not in result.artifacts


def test_bridge_service_prepare_case_uses_generated_working_copy_for_model_overrides(tmp_path):
    model_dir = tmp_path / "model"
    sim = write_minimal_qblade_model(model_dir)
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    write_airfoil_name(airfoils)
    captured: list[QBladeCasePrepRequest] = []

    def prepare(request: QBladeCasePrepRequest) -> QBladeCasePrepResult:
        captured.append(request)
        return fake_prep_result(request, tmp_path)

    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        output_dir=tmp_path / "out",
        airfoil_dir=airfoils,
        wind_speed=9.0,
        steps=12,
    )
    service = QBladeBridgeService(case_preparer=prepare)

    result = service.prepare_case(config, check_paths=True)

    assert captured
    working_sim = Path(result.artifacts["working_sim"])
    assert captured[0].sim_path == working_sim
    assert result.config.sim_path == working_sim
    assert read_qblade_keyword(working_sim, "MEANINF") == "9"
    assert read_qblade_keyword(working_sim, "NUMTIMESTEPS") == "12"
    assert read_qblade_keyword(sim, "MEANINF") == "5"
    manifest = json.loads(Path(result.artifacts["working_copy_manifest"]).read_text(encoding="utf-8"))
    assert {patch["keyword"] for patch in manifest["qblade_keyword_patches"]} == {"MEANINF", "NUMTIMESTEPS"}
    assert str(working_sim) in manifest["modified_files"]
    assert manifest["case_preparation_written_files"]
    assert result.diagnostics["qblade_model_override_count"] == 2


def test_bridge_service_run_case_forwards_progress_and_summarizes_outputs(tmp_path):
    captured: list[QBladeRunRequest] = []

    def run(request: QBladeRunRequest) -> QBladeRunResult:
        captured.append(request)
        return fake_run_result(request, tmp_path)

    config = bridge_config(tmp_path)
    messages: list[str] = []
    service = QBladeBridgeService(runner=run)

    result = service.run_case(config, message_callback=messages.append)

    assert captured
    assert captured[0].steps == 4
    assert captured[0].sil_vorlap is True
    assert captured[0].message_callback is not None
    assert messages == ["Completed 4/4 timesteps"]
    assert result.state == "complete"
    assert result.artifacts["summary"].endswith("case_run_summary.json")
    assert result.artifacts["native_callback_log"].endswith("bridge.log")
    assert result.artifacts["sil_vorlap_source_coverage"].endswith("case_run_summary_source_coverage.csv")
    with Path(result.artifacts["sil_vorlap_source_coverage"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "component": "BLD_1",
            "airfoil_id": "",
            "reynolds_model": "",
            "samples": "4",
            "node_samples": "8",
            "aoa_min_deg": "-2.5",
            "aoa_max_deg": "4.0",
            "source_aoa_min_deg": "",
            "source_aoa_max_deg": "",
            "re_min": "",
            "re_max": "",
            "source_re_min": "",
            "source_re_max": "",
            "freq_min_hz": "",
            "freq_max_hz": "",
            "unfiltered_freq_max_hz": "",
            "frequency_truncated_count": "",
            "nyquist_hz": "",
            "nyquist_ratio": "",
            "max_nodal_force_n": "",
            "force_coefficient_max_abs": "",
            "max_response_nodal_force_n": "",
            "response_force_coefficient_max_abs": "",
            "max_total_nodal_force_n": "",
            "total_force_coefficient_max_abs": "",
            "source_force_coefficient_max_abs": "",
            "max_structural_velocity_mps": "1.1",
            "force_power_mean_w": "",
            "force_power_max_abs_w": "",
            "re_lower_bound_warning_count": "",
            "re_out_of_range_count": "",
            "aoa_out_of_range_count": "",
            "aliased_frequency_count": "",
            "source_clip_count": "0",
            "force_coefficient_out_of_range_count": "",
            "nonfinite_count": "",
        }
    ]
    assert result.diagnostics["completed_steps"] == 4
    assert result.diagnostics["sil_vorlap_max_abs_force"] == 42.0
    assert result.diagnostics["sil_vorlap_max_structural_velocity_mps"] == 1.2
    assert result.diagnostics["sil_vorlap_cache"] == "3/1"
    assert result.diagnostics["sil_vorlap_velocity_warning_count"] == 2
    assert result.diagnostics["sil_vorlap_source_coverage_rows"] == 1
    assert result.diagnostics["source_bounds_policy"] == "raise"
    assert result.diagnostics["checkpoint_latest"].endswith("latest_checkpoint.json")
    assert result.diagnostics["checkpoint_latest_completed_steps"] == 4
    assert result.diagnostics["checkpoint_latest_project"].endswith("case_step_0000004.qpr")
    assert result.diagnostics["resume_requested"] is False
    assert result.diagnostics["resume_start_step"] == 0


def test_bridge_service_run_case_isolated_uses_subprocess_summary(tmp_path):
    commands: list[tuple[list[str], dict[str, object]]] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = ["subprocess progress\n"]

        def poll(self):
            return 0

        def wait(self):
            return 0

    config = bridge_config(tmp_path, python_executable=sys.executable)

    def launcher(command, **kwargs):
        commands.append((list(command), kwargs))
        request = config.to_run_request()
        result = fake_run_result(request, tmp_path)
        result.summary_path.parent.mkdir(parents=True, exist_ok=True)
        result.summary_path.write_text(json.dumps(result.summary), encoding="utf-8")
        result.result_path.write_text("Time~[s]\n0\n", encoding="utf-8")
        result.run_log_path.write_text("log\n", encoding="utf-8")
        return FakeProcess()

    messages: list[str] = []
    service = QBladeBridgeService(launcher=launcher)

    result = service.run_case_isolated(config, message_callback=messages.append)

    assert result.state == "complete"
    assert result.diagnostics["isolated_qblade_process"] is True
    assert result.diagnostics["completed_steps"] == 4
    assert result.artifacts["summary"].endswith("case_run_summary.json")
    assert messages[0].startswith("Launching isolated QBlade SIL runner:")
    assert "subprocess progress" in messages
    assert commands
    command, kwargs = commands[0]
    assert command[0] == sys.executable
    assert str(Path("run_headless") / "run_qblade_sim.py") in command[1]
    assert "--sil-vorlap" in command
    assert kwargs["stdout"] == subprocess.PIPE
    assert kwargs["stderr"] == subprocess.STDOUT


def test_bridge_service_run_case_tails_qblade_log_to_message_callback(tmp_path):
    def run(request: QBladeRunRequest) -> QBladeRunResult:
        run_log = Path(request.output_dir) / f"{Path(request.sim_path).stem}_qblade.log"
        run_log.parent.mkdir(parents=True, exist_ok=True)
        run_log.write_text(
            "initializing turbine\n"
            "VorLap callback loaded\n",
            encoding="utf-8",
        )
        return fake_run_result(request, tmp_path)

    config = bridge_config(tmp_path)
    messages: list[str] = []
    service = QBladeBridgeService(runner=run)

    result = service.run_case(config, message_callback=messages.append)

    assert "QBlade log: initializing turbine" in messages
    assert "QBlade log: VorLap callback loaded" in messages
    assert "Completed 4/4 timesteps" in messages
    assert result.diagnostics["qblade_log_tail_enabled"] is True
    assert result.diagnostics["qblade_log_tail_line_count"] == 2
    assert result.diagnostics["qblade_log_tail_path"].endswith("case_qblade.log")


def test_bridge_service_run_case_writes_bridge_quick_plots(tmp_path):
    pytest.importorskip("matplotlib")

    def run(request: QBladeRunRequest) -> QBladeRunResult:
        result = fake_run_result(request, tmp_path)
        history_path = Path(result.summary["sil_vorlap_history"])
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            "step_index,time_s,max_force_norm_n,max_force_node_id\n"
            "0,0.0,0.0,BLD_1_0.000000\n"
            "1,0.1,10.0,BLD_1_0.000000\n"
            "2,0.2,5.0,BLD_1_1.000000\n",
            encoding="utf-8",
        )
        result.result_path.parent.mkdir(parents=True, exist_ok=True)
        result.result_path.write_text(
            "QBlade synthetic export\n"
            "Time~[s]\tX_g~Pos.~BLD_1~pos~0.000~[m]\tRotSpeed~[rpm]\n"
            "0.0\t0.0\t1.0\n"
            "0.1\t0.02\t1.1\n"
            "0.2\t0.03\t1.2\n",
            encoding="utf-8",
        )
        result.summary["sil_vorlap_source_coverage"] = [
            {
                "component": "BLD_1",
                "samples": 3,
                "node_samples": 6,
                "aoa_min_deg": -2.5,
                "aoa_max_deg": 4.0,
                "source_aoa_min_deg": -15.0,
                "source_aoa_max_deg": 15.0,
                "re_min": 100000.0,
                "re_max": 250000.0,
                "source_re_min": 50000.0,
                "source_re_max": 500000.0,
                "nyquist_ratio": 2.0,
                "max_nodal_force_n": 12.5,
                "future_metric": "preserved",
            },
            {
                "component": "BLD_2",
                "samples": 3,
                "node_samples": 6,
                "aoa_min_deg": -1.0,
                "aoa_max_deg": 3.5,
                "source_aoa_min_deg": -12.0,
                "source_aoa_max_deg": 12.0,
                "re_min": 120000.0,
                "re_max": 260000.0,
                "source_re_min": 50000.0,
                "source_re_max": 500000.0,
                "nyquist_ratio": 0.8,
                "max_nodal_force_n": 9.0,
            },
        ]
        return result

    config = bridge_config(tmp_path)
    service = QBladeBridgeService(runner=run)

    result = service.run_case(config)

    plot_keys = {
        "bridge_quick_plot_max_force",
        "bridge_quick_plot_aoa_re",
        "bridge_quick_plot_nyquist",
        "bridge_quick_plot_force_history",
        "bridge_quick_plot_qblade_response",
    }
    assert result.diagnostics["bridge_quick_plot_count"] == 5
    assert result.diagnostics["bridge_quick_plot_errors"] == []
    assert result.diagnostics["bridge_quick_plot_force_history_column"] == "max_force_norm_n"
    assert result.diagnostics["bridge_quick_plot_qblade_response_channel"] == "X_g~Pos.~BLD_1~pos~0.000~[m]"
    assert result.diagnostics["sil_vorlap_source_coverage"][0]["freq_max_hz"] is None
    assert result.diagnostics["sil_vorlap_source_coverage"][0]["future_metric"] == "preserved"
    assert Path(result.artifacts["bridge_quick_plot_dir"]).is_dir()
    for key in plot_keys:
        path = Path(result.artifacts[key])
        assert path.is_file(), key
        assert path.suffix == ".png"


def test_bridge_service_run_case_updates_geometry_audit_with_qblade_reported_positions(tmp_path):
    def run(request: QBladeRunRequest) -> QBladeRunResult:
        result = fake_run_result(request, tmp_path)
        control_dir = tmp_path / "case" / "Control"
        control_dir.mkdir(parents=True)
        parameter_file = control_dir / "vorlap_qblade_external.json"
        audit_file = control_dir / "vorlap_qblade_geometry_audit.csv"
        parameter_file.write_text(
            json.dumps({"geometry_audit_file": "vorlap_qblade_geometry_audit.csv"}),
            encoding="utf-8",
        )
        with audit_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=GEOMETRY_AUDIT_COLUMNS)
            writer.writeheader()
            writer.writerow(
                {
                    "requested_node_id": "BLD_1_0.000000",
                    "action_id": "BLD_1",
                    "action_position": 0.0,
                    "qblade_reported_available": "false",
                    "vorlap_x_m": 1.0,
                    "vorlap_y_m": 2.0,
                    "vorlap_z_m": 3.0,
                    "basis_status": "ok",
                }
            )
            writer.writerow(
                {
                    "requested_node_id": "BLD_1_1.000000",
                    "action_id": "BLD_1",
                    "action_position": 1.0,
                    "qblade_reported_available": "false",
                    "vorlap_x_m": 0.0,
                    "vorlap_y_m": 0.0,
                    "vorlap_z_m": 0.0,
                    "basis_status": "ok",
                }
            )
        result.result_path.parent.mkdir(parents=True, exist_ok=True)
        result.result_path.write_text(
            "QBlade synthetic export\n"
            "Time~[s]\tX_g~Pos.~BLD_1~pos~0.000~[m]\tY_g~Pos.~BLD_1~pos~0.000~[m]\t"
            "Z_g~Pos.~BLD_1~pos~0.000~[m]\n"
            "0.0\t1.00003\t2.0\t2.99998\n",
            encoding="utf-8",
        )
        result.summary["sil_vorlap_parameter_file"] = str(parameter_file)
        result.summary["sil_vorlap_geometry_audit_file"] = str(audit_file)
        return result

    config = bridge_config(tmp_path)
    service = QBladeBridgeService(runner=run)

    result = service.run_case(config)

    assert result.artifacts["geometry_audit"].endswith("vorlap_qblade_geometry_audit.csv")
    assert result.diagnostics["geometry_audit_exists"] is True
    assert result.diagnostics["geometry_audit_rows"] == 2
    assert result.diagnostics["geometry_audit_reported_updated"] is True
    assert result.diagnostics["geometry_audit_reported_available_count"] == 1
    assert result.diagnostics["geometry_audit_reported_missing_count"] == 1
    assert result.diagnostics["geometry_audit_reported_error_count"] == 0
    assert result.diagnostics["geometry_audit_reported_max_delta_m"] == pytest.approx(
        ((3.0e-5) ** 2 + (2.0e-5) ** 2) ** 0.5
    )
    with Path(result.artifacts["geometry_audit"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["qblade_reported_available"] == "true"
    assert rows[0]["qblade_reported_x_m"] == "1.00003"
    assert rows[0]["delta_reported_m"]
    assert rows[1]["qblade_reported_available"] == "false"


def test_reported_geometry_audit_can_preserve_hash_pinned_prepared_input(tmp_path):
    prepared = tmp_path / "prepared_geometry_audit.csv"
    with prepared.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GEOMETRY_AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "requested_node_id": "BLD_1_0.000000",
                "action_id": "BLD_1",
                "action_position": 0.0,
                "qblade_reported_available": "false",
                "vorlap_x_m": 1.0,
                "vorlap_y_m": 2.0,
                "vorlap_z_m": 3.0,
                "basis_status": "ok",
            }
        )
    prepared_bytes = prepared.read_bytes()
    result_path = tmp_path / "result.txt"
    result_path.write_text(
        "QBlade synthetic export\n"
        "Time~[s]\tX_g~Pos.~BLD_1~pos~0.000~[m]\t"
        "Y_g~Pos.~BLD_1~pos~0.000~[m]\tZ_g~Pos.~BLD_1~pos~0.000~[m]\n"
        "0.0\t1.0\t2.0\t3.0\n",
        encoding="utf-8",
    )
    result = QBladeRunResult(
        summary={"sil_vorlap_geometry_audit_file": str(prepared)},
        summary_path=tmp_path / "summary.json",
        result_path=result_path,
        run_log_path=tmp_path / "run.log",
    )
    reported = tmp_path / "reported_geometry_audit.csv"

    diagnostics = write_run_geometry_audit_reported_positions(
        result,
        output_path=reported,
    )

    assert prepared.read_bytes() == prepared_bytes
    assert diagnostics["geometry_audit_reported_path"] == str(reported)
    assert diagnostics["geometry_audit_reported_updated"] is True
    with reported.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["qblade_reported_available"] == "true"
    assert rows[0]["qblade_reported_x_m"] == "1.0"


def test_bridge_service_run_case_reports_fitted_geometry_frame(tmp_path):
    points = [
        ("BLD_1_0.000000", 0.0, (0.0, 0.0, 0.0)),
        ("BLD_1_0.500000", 0.5, (1.0, 0.0, 0.0)),
        ("BLD_1_1.000000", 1.0, (0.0, 1.0, 0.0)),
    ]

    def transformed(point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = point
        return (y, -x, z - 3.0)

    def run(request: QBladeRunRequest) -> QBladeRunResult:
        result = fake_run_result(request, tmp_path)
        control_dir = tmp_path / "case" / "Control"
        control_dir.mkdir(parents=True)
        parameter_file = control_dir / "vorlap_qblade_external.json"
        audit_file = control_dir / "vorlap_qblade_geometry_audit.csv"
        parameter_file.write_text(
            json.dumps({"geometry_audit_file": "vorlap_qblade_geometry_audit.csv"}),
            encoding="utf-8",
        )
        with audit_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=GEOMETRY_AUDIT_COLUMNS)
            writer.writeheader()
            for node_id, action_position, point in points:
                writer.writerow(
                    {
                        "requested_node_id": node_id,
                        "action_id": "BLD_1",
                        "action_position": action_position,
                        "qblade_reported_available": "false",
                        "vorlap_x_m": point[0],
                        "vorlap_y_m": point[1],
                        "vorlap_z_m": point[2],
                        "basis_status": "ok",
                    }
                )
        header = ["Time~[s]"]
        values = ["0.0"]
        for _node_id, action_position, point in points:
            qblade_point = transformed(point)
            for axis, value in zip("XYZ", qblade_point, strict=True):
                header.append(position_channel_name({"id": "BLD_1", "position": action_position, "direction": axis}))
                values.append(str(value))
        result.result_path.parent.mkdir(parents=True, exist_ok=True)
        result.result_path.write_text(
            "QBlade synthetic export\n" + "\t".join(header) + "\n" + "\t".join(values) + "\n",
            encoding="utf-8",
        )
        result.summary["sil_vorlap_parameter_file"] = str(parameter_file)
        result.summary["sil_vorlap_geometry_audit_file"] = str(audit_file)
        return result

    service = QBladeBridgeService(runner=run)
    result = service.run_case(bridge_config(tmp_path))

    assert result.diagnostics["geometry_audit_reported_available_count"] == 3
    assert result.diagnostics["geometry_audit_reported_error_count"] == 3
    assert result.diagnostics["geometry_audit_reported_frame_fit_status"] == "within_tolerance"
    assert result.diagnostics["geometry_audit_reported_frame_fit_sample_count"] == 3
    assert result.diagnostics["geometry_audit_reported_frame_fit_max_residual_m"] == pytest.approx(0.0, abs=1.0e-12)
    assert result.diagnostics["geometry_audit_reported_frame_fit_translation_m"] == pytest.approx([0.0, 0.0, -3.0])
    with Path(result.artifacts["geometry_audit"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["qblade_frame_fit_available"] == "true"
    assert float(rows[0]["delta_reported_frame_fit_m"]) == pytest.approx(0.0, abs=1.0e-12)


def test_bridge_service_run_case_forwards_stop_callback_and_reports_stopped_state(tmp_path):
    captured: list[QBladeRunRequest] = []

    def run(request: QBladeRunRequest) -> QBladeRunResult:
        captured.append(request)
        result = fake_run_result(request, tmp_path)
        result.summary.update(
            {
                "completed_steps": 2,
                "stopped": True,
                "stop_message": "QBlade simulation stopped by user at 2/4 completed timesteps.",
            }
        )
        return result

    config = bridge_config(tmp_path)
    service = QBladeBridgeService(runner=run)

    result = service.run_case(config, stop_requested=lambda: True)

    assert captured
    assert captured[0].stop_requested is not None
    assert captured[0].stop_requested() is True
    assert result.state == "stopped"
    assert result.diagnostics["stopped"] is True
    assert result.diagnostics["stop_message"] == "QBlade simulation stopped by user at 2/4 completed timesteps."
    assert result.diagnostics["failed"] is False


def test_bridge_service_run_case_patches_generated_working_copy_for_model_overrides(tmp_path):
    model_dir = tmp_path / "model"
    sim = write_minimal_qblade_model(model_dir)
    (model_dir / "sidecar.txt").write_text("copied with model\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    write_airfoil_name(airfoils)
    captured: list[QBladeRunRequest] = []

    def run(request: QBladeRunRequest) -> QBladeRunResult:
        captured.append(request)
        return fake_run_result(request, tmp_path)

    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        output_dir=tmp_path / "out",
        airfoil_dir=airfoils,
        wind_speed=8.5,
        azimuth_deg=45.0,
        rpm=1.25,
        timestep_s=0.25,
        steps=12,
        store_from_s=0.5,
        store_replay=True,
    )
    service = QBladeBridgeService(runner=run)

    result = service.run_case(config, check_paths=True)

    assert captured
    working_sim = Path(result.artifacts["working_sim"])
    assert captured[0].sim_path == working_sim
    assert result.config.sim_path == working_sim
    assert working_sim.is_file()
    assert (working_sim.parent / "sidecar.txt").is_file()
    assert Path(result.artifacts["source_sim"]) == sim.resolve()
    assert result.artifacts["working_model_dir"] == str((tmp_path / "out" / "qblade_model_working").resolve())
    assert result.diagnostics["qblade_model_override_scope"] == "generated_working_copy"
    assert result.diagnostics["original_model_preserved"] is True
    assert result.diagnostics["qblade_model_override_count"] == 8
    assert read_qblade_keyword(working_sim, "MEANINF") == "8.5"
    assert read_qblade_keyword(working_sim, "INITIAL_AZIMUTH") == "45"
    assert read_qblade_keyword(working_sim, "PRESCRIBETYPE") == "1"
    assert read_qblade_keyword(working_sim, "RPMPRESCRIBED") == "1.25"
    assert read_qblade_keyword(working_sim, "NUMTIMESTEPS") == "12"
    assert read_qblade_keyword(working_sim, "TIMESTEP") == "0.25"
    assert read_qblade_keyword(working_sim, "STOREFROM") == "0.5"
    assert read_qblade_keyword(working_sim, "STOREREPLAY") == "true"
    assert read_qblade_keyword(sim, "MEANINF") == "5"
    assert read_qblade_keyword(sim, "NUMTIMESTEPS") == "100"


def test_bridge_service_run_case_prepares_active_working_copy_before_runner(tmp_path):
    model_dir = tmp_path / "model"
    sim = write_minimal_qblade_model(model_dir)
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    write_airfoil_name(airfoils)
    prep_requests: list[QBladeCasePrepRequest] = []
    run_requests: list[QBladeRunRequest] = []

    def prepare(request: QBladeCasePrepRequest) -> QBladeCasePrepResult:
        prep_requests.append(request)
        return fake_prep_result(request, tmp_path)

    def run(request: QBladeRunRequest) -> QBladeRunResult:
        run_requests.append(request)
        return fake_run_result(request, tmp_path)

    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        output_dir=tmp_path / "out",
        airfoil_dir=airfoils,
        timestep_s=0.0075,
        steps=2,
    )
    service = QBladeBridgeService(case_preparer=prepare, runner=run)

    result = service.run_case(config, check_paths=True)

    assert prep_requests
    assert run_requests
    assert Path(prep_requests[0].sim_path) == Path(run_requests[0].sim_path)
    assert Path(run_requests[0].sim_path).parent == tmp_path / "out" / "qblade_model_working"
    assert result.diagnostics["prepared_before_run"] is True
    assert result.artifacts["runtime_config"].endswith("vorlap_qblade_external.json")


def test_reference_stable_support_patch_requires_explicit_request(tmp_path, monkeypatch):
    sim = tmp_path / "baseline_reference.sim"
    turbine = tmp_path / "reference_model" / "reference.trb"
    structure = tmp_path / "reference_model" / "Structure" / "reference_main.str"
    structure.parent.mkdir(parents=True)
    turbine.write_text(
        "Structure/reference_main.str STRUCTURALFILE - structural model\n",
        encoding="utf-8",
    )
    structure.write_text(
        "legacy_sub.str SUBFILE - substructure\n",
        encoding="utf-8",
    )
    sim.write_text(
        "reference_model/reference.trb TURBFILE - turbine model\n"
        "5 RELAXSTEPS - static relaxation steps\n"
        "0 RELAXTIMESTEP - relaxation timestep\n"
        "10 RAMPUP - rampup\n"
        "5 ADDDAMP - damping\n"
        "0.0075 TIMESTEP - timestep\n",
        encoding="utf-8",
    )
    config = QBladeBridgeConfig(
        sim_path=sim,
        qblade_root=tmp_path / "QBladeCE",
        output_dir=tmp_path / "out",
        airfoil_dir=tmp_path / "airfoils",
        timestep_s=0.0075,
    )

    monkeypatch.delenv("VORLAP_STABLE_PATCH_REMOVE_SUBFILE", raising=False)
    assert _maybe_apply_reference_stable_support_patch(config) is None
    assert read_qblade_keyword(structure, "SUBFILE") == "legacy_sub.str"
    assert read_qblade_keyword(sim, "RAMPUP") == "10"
    monkeypatch.setenv("VORLAP_STABLE_PATCH_REMOVE_SUBFILE", "1")
    manifest = _maybe_apply_reference_stable_support_patch(config)

    assert manifest == tmp_path / "reference_model" / "Control" / "vorlap_stable_support_patch.json"
    assert manifest.is_file()
    assert read_qblade_keyword(sim, "RELAXSTEPS") == "0"
    assert read_qblade_keyword(sim, "RELAXTIMESTEP") == "0.0075"
    assert read_qblade_keyword(sim, "RAMPUP") == "0"
    assert read_qblade_keyword(sim, "ADDDAMP") == "0"
    assert read_qblade_keyword(structure, "SUBFILE") is None
    assert "legacy_sub.str" not in structure.read_text(encoding="utf-8")


def test_bridge_service_model_overrides_reject_output_inside_source_model(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = model_dir / "runs" / "out"
    sim = write_minimal_qblade_model(model_dir)
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    write_airfoil_name(airfoils)
    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        output_dir=output_dir,
        airfoil_dir=airfoils,
        wind_speed=8.5,
    )
    service = QBladeBridgeService(runner=lambda request: fake_run_result(request, tmp_path))

    with pytest.raises(ValueError, match="output_dir outside the selected model directory"):
        service.run_case(config, check_paths=True)


def test_bridge_service_run_sign_probe_prepares_working_copy_and_writes_report(tmp_path):
    model_dir = tmp_path / "model"
    sim = write_minimal_qblade_model(model_dir)
    (model_dir / "sidecar.txt").write_text("copied with model\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    airfoils = tmp_path / "airfoils"
    write_airfoil_name(airfoils)
    prep_requests: list[QBladeCasePrepRequest] = []
    run_requests: list[QBladeRunRequest] = []

    def prepare(request: QBladeCasePrepRequest) -> QBladeCasePrepResult:
        prep_requests.append(request)
        return fake_prep_result(request, tmp_path)

    def run(request: QBladeRunRequest) -> QBladeRunResult:
        run_requests.append(request)
        spec = request.sil_constant_force_specs[0]
        _node, axis, force_text, _local = spec.split(":")
        force = float(force_text)
        delta = force * 1.0e-6
        summary_path = Path(request.output_dir) / "case_run_summary.json"
        result_path = Path(request.output_dir) / f"{request.result_name}.txt"
        run_log_path = Path(request.output_dir) / "case_qblade.log"
        return QBladeRunResult(
            summary={
                "requested_steps": request.steps,
                "completed_steps": request.steps,
                "failed": False,
                "exported_results": True,
                "coupling_mode": "sil-constant-force",
                "result_file": str(result_path),
                "force_sign_probe_observations": [
                    {
                        "status": "same_sign" if force > 0.0 else "opposite_sign",
                        "observed_position_channel": f"{axis}_g Position BLD_1_0.230455 [m]",
                        "delta_position_m": delta,
                    }
                ],
            },
            summary_path=summary_path,
            result_path=result_path,
            run_log_path=run_log_path,
        )

    config = bridge_config(
        tmp_path,
        sim_path=sim,
        qblade_root=qblade_root,
        output_dir=tmp_path / "out",
        airfoil_dir=airfoils,
        steps=None,
    )
    service = QBladeBridgeService(case_preparer=prepare, runner=run)

    result = service.run_sign_probe(
        config,
        axes="X Z",
        force_n=750.0,
        steps=25,
        check_paths=True,
    )

    assert prep_requests
    working_sim = Path(result.artifacts["working_sim"])
    assert prep_requests[0].sim_path == working_sim
    assert working_sim.is_file()
    assert (working_sim.parent / "sidecar.txt").is_file()
    assert len(run_requests) == 4
    assert all(request.coupling_mode == "sil-constant-force" for request in run_requests)
    assert all(request.sil_vorlap is False for request in run_requests)
    assert all(request.steps == 25 for request in run_requests)
    assert run_requests[0].sil_constant_force_specs == ("BLD_1_0.000000:X:750:global",)
    assert run_requests[1].sil_constant_force_specs == ("BLD_1_0.000000:X:-750:global",)
    assert Path(run_requests[0].case_metadata_path).is_file()
    assert result.state == "complete"
    assert result.diagnostics["node"] == "BLD_1_0.000000"
    assert result.diagnostics["axes"] == ["X", "Z"]
    assert result.diagnostics["force_n"] == 750.0
    assert result.diagnostics["probes_completed"] == 4
    assert result.diagnostics["failures"] == 0
    assert result.diagnostics["passed"] is True
    assert len(result.diagnostics["sign_probe_report_rows"]) == 4
    assert result.diagnostics["sign_probe_report_rows"][0]["axis"] == "X"
    assert result.diagnostics["sign_probe_report_rows"][0]["sign"] == "positive"
    assert result.diagnostics["sign_probe_report_rows"][0]["paired_passed"] == "true"
    assert result.diagnostics["sign_probe_report_rows"][0]["observed_position_channel"] == "X_g Position BLD_1_0.230455 [m]"
    assert result.diagnostics["qblade_model_override_count"] == 0
    assert Path(result.artifacts["sign_probe_plan"]).is_file()
    assert Path(result.artifacts["sign_probe_report_csv"]).is_file()
    report = json.loads(Path(result.artifacts["sign_probe_report_json"]).read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["rows"] == 4
    assert all(row["paired_passed"] == "true" for row in report["probes"])


def test_bridge_service_run_sign_probe_rejects_bad_inputs(tmp_path):
    config = bridge_config(tmp_path)
    service = QBladeBridgeService()

    with pytest.raises(ValueError, match="Unsupported sign-probe axis"):
        service.run_sign_probe(config, axes="X Q")

    with pytest.raises(ValueError, match="positive and finite"):
        service.run_sign_probe(config, force_n=0.0)

    with pytest.raises(ValueError, match="steps must be positive"):
        service.run_sign_probe(config, steps=0)


def test_bridge_service_setup_batch_delegates_and_reports_manifest_path(tmp_path):
    args = SimpleNamespace(run_root=tmp_path / "run")

    def setup(_args):
        return {"cases": [{"index": 0}, {"index": 1}]}

    service = QBladeBridgeService(batch_setup=setup)

    result = service.setup_batch(args)

    assert result.case_count == 2
    assert result.manifest["cases"] == [{"index": 0}, {"index": 1}]
    assert result.artifacts["manifest"] == str(args.run_root / "cases.json")


def test_bridge_service_build_bridge_delegates_to_builder_and_reports_artifacts(tmp_path):
    qblade_root = tmp_path / "QBladeCE"
    controller_files = qblade_root / "ControllerFiles"
    controller_files.mkdir(parents=True)
    python = Path(sys.executable)
    config = bridge_config(tmp_path, qblade_root=qblade_root, python_executable=python)
    captured: list[object] = []

    def builder(args) -> None:
        captured.append(args)
        (args.qblade_root / "ControllerFiles" / "libvorlap_qblade_bridge.so").write_text(
            "bridge placeholder\n",
            encoding="utf-8",
        )

    service = QBladeBridgeService(bridge_builder=builder)

    result = service.build_bridge(config, force=True)

    assert captured
    args = captured[0]
    assert args.qblade_root == qblade_root.resolve()
    assert args.run_root == Path(config.output_dir).resolve()
    assert args.vorlap_root == Path(__file__).resolve().parents[1]
    assert args.python == str(python)
    assert args.force_bridge_build is True
    assert args.skip_bridge_build is False
    assert args.skip_vorlap_install is True
    assert result.state == "built"
    assert result.artifacts["controller_files"] == str(controller_files.resolve())
    assert result.artifacts["build_dir"] == str(Path(config.output_dir).resolve() / "build" / "qblade_external_linux")
    assert result.artifacts["bridge_library"] == str((controller_files / "libvorlap_qblade_bridge.so").resolve())
    assert result.diagnostics["force"] is True
    assert result.diagnostics["python"] == str(python)
    assert result.diagnostics["python_executable"] == str(python)
    assert result.diagnostics["python_version"] == sys.version.split()[0]
    assert result.diagnostics["numpy_version"]
    assert result.diagnostics["python_probe_error"] is None
    assert "cmake_version" in result.diagnostics
    assert "cmake_error" in result.diagnostics
    assert "compiler" in result.diagnostics
    assert "compiler_error" in result.diagnostics
    assert result.diagnostics["qblade_ld_library_path"]
    expected_loader_key = "PATH" if os.name == "nt" else "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    assert result.diagnostics["qblade_loader_path_key"] == expected_loader_key
    assert result.diagnostics["qblade_loader_path"] == result.diagnostics["qblade_ld_library_path"]
    assert result.diagnostics["bridge_library_exists"] is True
    assert result.diagnostics["bridge_library_loadable"] is False
    assert result.diagnostics["bridge_library_load_error"]


def test_bridge_service_builds_qblade_gui_launch_plan_and_can_launch_with_fake_launcher(tmp_path):
    qblade_root = tmp_path / "QBladeCE"
    libraries = qblade_root / "Libraries"
    controller = qblade_root / "ControllerFiles"
    for path in (libraries, controller):
        path.mkdir(parents=True)
    executable = qblade_root / "QBlade"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    config = bridge_config(tmp_path, qblade_root=qblade_root, qblade_binary=None)

    assert find_qblade_gui_executable(qblade_root) == executable.resolve()

    launched: list[dict[str, object]] = []

    def launcher(command, *, cwd, env, **kwargs):
        launched.append({"command": command, "cwd": cwd, "env": env, "kwargs": kwargs})
        return "fake-process"

    service = QBladeBridgeService(launcher=launcher)
    plan, process = service.launch_qblade_gui(config, extra_args=("--inspect",), env={"PATH": "/bin"})

    assert process == "fake-process"
    assert plan.executable == executable.resolve()
    assert plan.command == (str(executable.resolve()), "--inspect")
    assert plan.cwd == qblade_root.resolve()
    assert plan.env["QBLADE_ROOT"] == str(qblade_root.resolve())
    assert plan.env["OMP_NUM_THREADS"] == "2"
    assert str(qblade_root.resolve()) in plan.env["LD_LIBRARY_PATH"]
    assert str(libraries.resolve()) in plan.env["LD_LIBRARY_PATH"]
    assert str(qblade_root.resolve()) in plan.env["PATH"]
    assert str(libraries.resolve()) in plan.env["PATH"]
    assert plan.env["PATH"].endswith("/bin")
    assert launched[0]["command"] == list(plan.command)
    assert launched[0]["cwd"] == str(qblade_root.resolve())
    if os.name == "nt":
        assert launched[0]["kwargs"]["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert launched[0]["kwargs"]["start_new_session"] is True


def test_bridge_service_launch_plan_prefers_completed_project_from_last_run(tmp_path):
    qblade_root = tmp_path / "QBladeCE"
    (qblade_root / "Libraries").mkdir(parents=True)
    (qblade_root / "ControllerFiles").mkdir()
    executable = qblade_root / "QBlade"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    config = bridge_config(tmp_path, qblade_root=qblade_root, qblade_binary=executable)
    stored_project = tmp_path / "out" / "case_completed.qpr"
    checkpoint_project = tmp_path / "out" / "checkpoints" / "case_step_0000004.qpr"
    stored_project.parent.mkdir(parents=True)
    checkpoint_project.parent.mkdir(parents=True)
    stored_project.write_text("completed project\n", encoding="utf-8")
    checkpoint_project.write_text("checkpoint project\n", encoding="utf-8")
    run_result = QBladeRunResult(
        summary={
            "stored_project": str(stored_project),
            "restart": {"latest_checkpoint_project": str(checkpoint_project)},
        },
        summary_path=tmp_path / "out" / "case_run_summary.json",
        result_path=tmp_path / "out" / "case_results.txt",
        run_log_path=tmp_path / "out" / "case_qblade.log",
    )
    bridge_result = QBladeBridgeRunServiceResult(
        config=config,
        run_result=run_result,
        state="complete",
        artifacts={"stored_project": str(stored_project)},
        diagnostics={"checkpoint_latest_project": str(checkpoint_project)},
    )

    plan = QBladeBridgeService().build_qblade_gui_launch_plan(config, run_result=bridge_result)

    assert plan.project_path == stored_project.resolve()
    assert plan.project_source == "completed run project"
    assert plan.command == (str(executable.resolve()), str(stored_project.resolve()))


def test_bridge_service_launch_plan_falls_back_to_latest_checkpoint_project(tmp_path):
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    executable = qblade_root / "QBlade"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    checkpoint_dir = tmp_path / "out" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_project = checkpoint_dir / "case_step_0000004.qpr"
    checkpoint_project.write_text("checkpoint project\n", encoding="utf-8")
    latest = checkpoint_dir / "latest_checkpoint.json"
    latest.write_text(json.dumps({"qblade_project": str(checkpoint_project)}) + "\n", encoding="utf-8")
    config = bridge_config(
        tmp_path,
        qblade_root=qblade_root,
        qblade_binary=executable,
        checkpoint_dir=checkpoint_dir,
    )

    plan = QBladeBridgeService().build_qblade_gui_launch_plan(config)

    assert plan.project_path == checkpoint_project.resolve()
    assert plan.project_source == "latest restart checkpoint"
    assert plan.command == (str(executable.resolve()), str(checkpoint_project.resolve()))


def test_find_qblade_gui_executable_accepts_version_named_binary(tmp_path):
    qblade_root = tmp_path / "QBladeCE_2.0.9.7"
    qblade_root.mkdir()
    executable = qblade_root / qblade_root.name
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    assert find_qblade_gui_executable(qblade_root) == executable.resolve()


def test_bridge_service_can_terminate_launched_process(monkeypatch):
    terminated: list[tuple[object, float]] = []
    process = object()

    def terminate(value, *, timeout_s=10.0):
        terminated.append((value, timeout_s))
        return -15

    monkeypatch.setattr("vorlap.qblade_bridge_service.terminate_process_tree", terminate)
    service = QBladeBridgeService()

    assert service.terminate_launched_process(process, timeout_s=0.5) == -15
    assert terminated == [(process, 0.5)]


def test_bridge_service_opens_existing_artifact_with_fake_launcher(tmp_path):
    artifact = tmp_path / "case_run_summary.json"
    artifact.write_text("{}\n", encoding="utf-8")
    launched: list[dict[str, object]] = []

    def launcher(command, *args, **kwargs):
        launched.append({"command": command, "args": args, "kwargs": kwargs})
        return "fake-open-process"

    service = QBladeBridgeService(launcher=launcher)
    plan, process = service.open_artifact(artifact)

    assert process == "fake-open-process"
    assert plan.artifact == artifact.resolve()
    assert str(artifact.resolve()) == plan.command[-1]
    assert launched == [{"command": list(plan.command), "args": (), "kwargs": {}}]

    with pytest.raises(FileNotFoundError, match="Artifact path does not exist"):
        service.build_artifact_open_plan(tmp_path / "missing.txt")

    with pytest.raises(ValueError, match="Select an artifact path"):
        service.build_artifact_open_plan(" ")


def test_bridge_service_launch_plan_requires_existing_executable(tmp_path):
    config = bridge_config(tmp_path, qblade_binary=tmp_path / "missing-qblade")
    service = QBladeBridgeService()

    with pytest.raises(FileNotFoundError, match="QBlade GUI executable does not exist"):
        service.build_qblade_gui_launch_plan(config)
