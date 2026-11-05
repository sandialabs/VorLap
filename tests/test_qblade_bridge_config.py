from __future__ import annotations

import json
from pathlib import Path

import pytest

from vorlap.qblade_bridge_config import QBladeBridgeConfig


def bridge_config(tmp_path: Path, **overrides: object) -> QBladeBridgeConfig:
    values: dict[str, object] = {
        "sim_path": tmp_path / "case.sim",
        "qblade_root": tmp_path / "QBladeCE",
        "output_dir": tmp_path / "out",
        "airfoil_dir": tmp_path / "airfoils",
        "qblade_binary": tmp_path / "QBlade",
        "python_executable": tmp_path / ".venv" / "bin" / "python",
        "result_name": "case_results",
        "n_freq_depth": 20,
        "force_scale": 1.0,
        "coupling_mode": "sil-vorlap",
        "duration_s": 1.0,
        "timestep_s": 0.25,
        "threads": 4,
        "use_opencl": True,
        "opencl_device": 2,
        "opencl_group_size": 64,
        "checkpoint_every": 10,
        "record_sil_vorlap_history": True,
        "manage_loader_env": False,
    }
    values.update(overrides)
    return QBladeBridgeConfig(**values).validated()


def write_valid_path_inputs(tmp_path: Path) -> None:
    (tmp_path / "case.sim").write_text("simulation placeholder\n", encoding="utf-8")
    (tmp_path / "QBladeCE").mkdir()
    (tmp_path / "airfoils").mkdir()
    (tmp_path / "QBlade").write_text("#!/bin/sh\n", encoding="utf-8")


def write_checkpoint_metadata(path: Path, **overrides: object) -> Path:
    project = path.with_suffix(".qpr")
    project.parent.mkdir(parents=True, exist_ok=True)
    project.write_text("QBlade checkpoint project\n", encoding="utf-8")
    metadata: dict[str, object] = {
        "sim": str(path.parents[1] / "case.sim"),
        "qblade_root": str(path.parents[1] / "QBladeCE"),
        "qblade_project": str(project),
        "coupling_mode": "sil-vorlap",
        "result_name": "case_results",
        "requested_steps": 4,
        "completed_steps": 2,
        "threads": 4,
        "use_opencl": True,
    }
    metadata.update(overrides)
    path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    return path


def test_bridge_config_serializes_round_trip_and_computes_steps(tmp_path):
    config = bridge_config(tmp_path, include_mean_load=True, python_path_entries=("extra",))

    restored = QBladeBridgeConfig.from_json(config.to_json())

    assert restored.as_dict() == config.as_dict()
    assert restored.as_dict()["include_mean_load"] is True
    assert restored.as_dict()["source_bounds_policy"] == "raise"
    assert restored.as_dict()["python_path_entries"] == ["extra"]
    assert restored.effective_steps() == 4


def test_bridge_config_from_gui_state_normalizes_strings(tmp_path):
    config = QBladeBridgeConfig.from_gui_state(
        {
            "sim": tmp_path / "case.sim",
            "qblade_root": tmp_path / "QBladeCE",
            "run_output_dir": tmp_path / "out",
            "airfoils": tmp_path / "airfoils",
            "include_struts": "false",
            "include_tower": "true",
            "include_mean_load": "yes",
            "source_bounds_policy": "RAISE",
            "diagnostic_force_scale": "true",
            "force_scale": "2.5",
            "duration": "2.0",
            "timestep": "0.5",
            "threads": "8",
            "use_opencl": "on",
            "device": "3",
            "group_size": "16",
            "wind_speed": "9.5",
            "rpm": "1.25",
            "azimuth": "45",
            "fluid_density": "1.3",
            "fluid_dynamicviscosity": "1.7e-5",
            "rotation_axis": ("0", "0", "1"),
            "rotation_axis_offset": "1 2 3",
            "inflow_vec": ("1", "1", "0"),
            "store_from": "0.5",
            "store_replay": "false",
            "coupling_mode": "qblade-external",
            "qblade_execution_mode": "run-full-simulation",
            "checkpoint_every": "20",
            "checkpoint_dir": tmp_path / "checkpoints",
            "resume_from_checkpoint": "true",
            "resume_checkpoint": tmp_path / "checkpoints" / "latest_checkpoint.json",
            "resume_initialize": "yes",
            "manage_loader_env": "false",
            "patch_original_model": "true",
        }
    )

    assert config.include_struts is False
    assert config.include_tower is True
    assert config.include_mean_load is True
    assert config.source_bounds_policy == "raise"
    assert config.force_scale == 2.5
    assert config.diagnostic_force_scale is True
    assert config.effective_steps() == 4
    assert config.threads == 8
    assert config.use_opencl is True
    assert config.opencl_device == 3
    assert config.opencl_group_size == 16
    assert config.wind_speed == 9.5
    assert config.rpm == 1.25
    assert config.azimuth_deg == 45.0
    assert config.fluid_density == 1.3
    assert config.fluid_dynamicviscosity == 1.7e-5
    assert config.rotation_axis == (0.0, 0.0, 1.0)
    assert config.rotation_axis_offset == (1.0, 2.0, 3.0)
    assert config.inflow_vec == (1.0, 1.0, 0.0)
    assert config.store_from_s == 0.5
    assert config.store_replay is False
    assert config.coupling_mode == "qblade-external"
    assert config.qblade_execution_mode == "run-full-simulation"
    assert config.checkpoint_every == 20
    assert str(config.checkpoint_dir) == str(tmp_path / "checkpoints")
    assert config.resume_from_checkpoint is True
    assert str(config.resume_checkpoint) == str(tmp_path / "checkpoints" / "latest_checkpoint.json")
    assert config.resume_initialize is True
    assert config.manage_loader_env is False
    assert config.patch_original_model is True


def test_bridge_config_rejects_bad_force_scale_without_diagnostic_label(tmp_path):
    config = QBladeBridgeConfig(
        sim_path=tmp_path / "case.sim",
        qblade_root=tmp_path / "QBladeCE",
        output_dir=tmp_path / "out",
        airfoil_dir=tmp_path / "airfoils",
        force_scale=2.0,
    )

    with pytest.raises(ValueError, match="diagnostic-force-scale"):
        config.validated()


def test_bridge_config_rejects_bad_source_bounds_policy(tmp_path):
    config = QBladeBridgeConfig(
        sim_path=tmp_path / "case.sim",
        qblade_root=tmp_path / "QBladeCE",
        output_dir=tmp_path / "out",
        airfoil_dir=tmp_path / "airfoils",
        source_bounds_policy="extrapolate",  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="source_bounds_policy"):
        config.validated()


def test_bridge_config_builds_case_prep_and_run_requests(tmp_path):
    config = bridge_config(
        tmp_path,
        include_mean_load=True,
        source_bounds_policy="raise",
        include_tower=True,
        include_tower_nodes=True,
        sample_step=0.01,
        fluid_density=1.3,
        fluid_dynamicviscosity=1.7e-5,
        rotation_axis=(0.0, 0.0, 1.0),
        rotation_axis_offset=(1.0, 2.0, 3.0),
        inflow_vec=(1.0, 1.0, 0.0),
        log_file="bridge.log",
        geometry_audit_file="audit.csv",
        case_metadata_path=tmp_path / "case_metadata.json",
        store_project=True,
    )

    prep = config.to_case_prep_request()
    run = config.to_run_request()

    assert prep.sim_path == config.sim_path
    assert prep.airfoil_dir == config.airfoil_dir
    assert prep.include_tower is True
    assert prep.include_tower_nodes is True
    assert prep.include_mean_load is True
    assert prep.source_bounds_policy == "raise"
    assert prep.sample_step == 0.01
    assert prep.fluid_density == 1.3
    assert prep.fluid_dynamicviscosity == 1.7e-5
    assert prep.rotation_axis == (0.0, 0.0, 1.0)
    assert prep.rotation_axis_offset == (1.0, 2.0, 3.0)
    assert prep.inflow_vec == (1.0, 1.0, 0.0)
    assert prep.log_file == "bridge.log"
    assert prep.geometry_audit_file == "audit.csv"

    assert run.sim_path == config.sim_path
    assert run.qblade_root == config.qblade_root
    assert run.output_dir == config.output_dir
    assert run.steps == 4
    assert run.threads == 4
    assert run.device == 2
    assert run.group_size == 64
    assert run.use_opencl is True
    assert run.checkpoint_every == 10
    assert run.sil_vorlap is True
    assert run.coupling_mode == "sil-vorlap"
    assert run.qblade_execution_mode == "step-loop"
    assert run.record_sil_vorlap_history is True
    assert run.case_metadata_path == tmp_path / "case_metadata.json"
    assert run.store_project is True
    assert run.manage_loader_env is False


def test_bridge_config_builds_prepare_and_run_cli_args(tmp_path):
    config = bridge_config(
        tmp_path,
        include_struts=False,
        include_tower=True,
        include_mean_load=True,
        source_bounds_policy="clip",
        debug=True,
        python_path_entries=("extra",),
        include_default_python_path=False,
        sample_step=0.05,
        store_project=True,
    )

    prepare_args = config.to_cli_args("prepare")
    run_args = config.to_cli_args("run")

    assert prepare_args[prepare_args.index("--sim") + 1] == str(config.sim_path)
    assert prepare_args[prepare_args.index("--airfoils") + 1] == str(config.airfoil_dir)
    assert "--no-include-struts" in prepare_args
    assert "--include-tower" in prepare_args
    assert "--include-mean-load" in prepare_args
    assert prepare_args[prepare_args.index("--source-bounds-policy") + 1] == "clip"
    assert "--debug" in prepare_args
    assert prepare_args[prepare_args.index("--sample-step") + 1] == "0.05"
    assert prepare_args[prepare_args.index("--python-path-entry") + 1] == "extra"
    assert "--no-default-python-path" in prepare_args

    assert run_args[run_args.index("--qblade-root") + 1] == str(config.qblade_root)
    assert run_args[run_args.index("--steps") + 1] == "4"
    assert run_args[run_args.index("--threads") + 1] == "4"
    assert run_args[run_args.index("--device") + 1] == "2"
    assert run_args[run_args.index("--group-size") + 1] == "64"
    assert "--use-opencl" in run_args
    assert "--store-project" in run_args
    assert "--native-run-full-simulation" not in run_args

    native_args = bridge_config(
        tmp_path,
        coupling_mode="qblade-external",
        qblade_execution_mode="run-full-simulation",
        checkpoint_every=0,
    ).to_cli_args("run")
    assert "--native-run-full-simulation" in native_args
    assert native_args[native_args.index("--coupling-mode") + 1] == "qblade-external"
    assert "--sil-vorlap" in run_args
    assert "--record-sil-vorlap-history" in run_args


def test_bridge_config_path_validation_reports_missing_inputs(tmp_path):
    config = bridge_config(tmp_path)

    errors = config.validation_errors(check_paths=True)

    assert any("QBlade simulation file does not exist" in error for error in errors)
    assert any("QBlade root directory does not exist" in error for error in errors)
    assert any("VorLap airfoil directory does not exist" in error for error in errors)
    assert any("QBlade executable does not exist" in error for error in errors)


def test_bridge_config_path_validation_accepts_creatable_output_dir(tmp_path):
    write_valid_path_inputs(tmp_path)
    config = bridge_config(tmp_path, output_dir=tmp_path / "runs" / "case")

    assert config.validation_errors(check_paths=True) == ()


def test_bridge_config_path_validation_rejects_unusable_output_paths(tmp_path):
    write_valid_path_inputs(tmp_path)
    output_file = tmp_path / "out-is-a-file"
    output_file.write_text("not a directory\n", encoding="utf-8")
    blocked_parent = tmp_path / "blocked-parent"
    blocked_parent.write_text("not a directory\n", encoding="utf-8")

    errors = bridge_config(tmp_path, output_dir=output_file).validation_errors(check_paths=True)
    nested_errors = bridge_config(tmp_path, output_dir=blocked_parent / "out").validation_errors(check_paths=True)

    assert f"QBlade output directory is not a directory: {output_file}" in errors
    assert any("parent is not a directory" in error and str(blocked_parent) in error for error in nested_errors)


def test_bridge_config_validates_resume_checkpoint_metadata_before_run(tmp_path):
    write_valid_path_inputs(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint = checkpoint_dir / "latest_checkpoint.json"
    config = bridge_config(
        tmp_path,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=True,
    )

    errors = config.validation_errors(check_paths=True)

    assert errors == (f"Resume checkpoint metadata does not exist: {checkpoint.resolve()}",)

    write_checkpoint_metadata(checkpoint)

    assert config.validation_errors(check_paths=True) == ()

    write_checkpoint_metadata(
        checkpoint,
        sim=str(tmp_path / "different.sim"),
        requested_steps=99,
    )

    errors = config.validation_errors(check_paths=True)

    assert any("Resume checkpoint sim mismatch" in error for error in errors)
    assert any("Resume checkpoint requested_steps mismatch" in error for error in errors)


def test_bridge_config_rejects_non_positive_sample_step(tmp_path):
    config = QBladeBridgeConfig(
        sim_path=tmp_path / "case.sim",
        qblade_root=tmp_path / "QBladeCE",
        output_dir=tmp_path / "out",
        airfoil_dir=tmp_path / "airfoils",
        sample_step=0.0,
    )

    with pytest.raises(ValueError, match="sample_step must be positive"):
        config.validated()


def test_bridge_config_rejects_invalid_model_override_values(tmp_path):
    config = QBladeBridgeConfig(
        sim_path=tmp_path / "case.sim",
        qblade_root=tmp_path / "QBladeCE",
        output_dir=tmp_path / "out",
        airfoil_dir=tmp_path / "airfoils",
        wind_speed=-1.0,
        store_from_s=-0.1,
    )

    errors = config.validation_errors()

    assert "wind_speed must be non-negative when provided." in errors
    assert "store_from_s must be non-negative when provided." in errors


@pytest.mark.parametrize("mode,native", [("sil-vorlap", False), ("qblade-external", True)])
def test_native_case_keeps_paths_anchored_when_qblade_relocates_parameters(tmp_path, mode, native):
    config = bridge_config(tmp_path, coupling_mode=mode)
    assert config.to_case_prep_request().absolute_source_parameter_dir is native
