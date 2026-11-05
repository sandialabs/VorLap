from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import h5py

from vorlap.gui.tabs import QBladeBridgeTab
from vorlap.gui.tabs import qblade_bridge as bridge_module
from vorlap.gui.tabs.qblade_bridge import (
    BRIDGE_USER_ACTIONS,
    BRIDGE_SCENARIOS,
    bridge_config_warnings,
    bridge_scenario_config,
    build_status_rows,
    infer_model_airfoil_dir,
    prepare_status_rows,
    run_status_rows,
    sign_probe_status_rows,
    viv_parameter_status_rows,
)
from vorlap.gui.state import VorLapGuiState
from vorlap.qblade_bridge_config import QBladeBridgeConfig
from vorlap.qblade_bridge_service import QBladeBridgeService


class DummyValue:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class DummyStatus:
    def __init__(self) -> None:
        self.value = ""
        self.history: list[str] = []

    def set(self, value):
        self.value = value
        self.history.append(value)


class DummyLog:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def write(self, message):
        self.messages.append(message)


class DummyButton:
    def __init__(self) -> None:
        self.state = None

    def configure(self, *, state):
        self.state = state


def minimal_bridge_config(tmp_path: Path, **overrides: object) -> QBladeBridgeConfig:
    values: dict[str, object] = {
        "sim_path": tmp_path / "case.sim",
        "qblade_root": tmp_path / "QBladeCE",
        "output_dir": tmp_path / "out",
        "airfoil_dir": tmp_path / "airfoils",
        "coupling_mode": "sil-vorlap",
        "manage_loader_env": False,
    }
    values.update(overrides)
    return QBladeBridgeConfig(**values).validated()


def write_minimal_valid_bridge_paths(tmp_path: Path) -> None:
    aero_dir = tmp_path / "Aero"
    structure_dir = tmp_path / "Structure"
    aero_dir.mkdir()
    structure_dir.mkdir()
    (tmp_path / "case.sim").write_text(
        "case.trb TURBFILE - turbine file\n"
        "100 NUMTIMESTEPS - steps\n",
        encoding="utf-8",
    )
    (tmp_path / "case.trb").write_text(
        "Aero/case.bld BLADEFILE - blade file\n"
        "1 NUMBLADES - blades\n"
        "Structure/case.str STRUCTURALFILE - structural file\n",
        encoding="utf-8",
    )
    (aero_dir / "case.bld").write_text(
        "1 NUMBLADES - blades\n"
        "HEIGHT_[m] CHORD_[m] RADIUS_[m] TOFFSET_[m] TWIST_[deg] CIRCANGLE_[deg] P_AXIS_[-] POLAR_FILE\n"
        "0.0 0.10 0.50 0.00 0.0 0.0 0.25 Polars/NACA_0018.plr\n"
        "1.0 0.10 0.50 0.00 0.0 0.0 0.25 Polars/NACA_0018.plr\n",
        encoding="utf-8",
    )
    (structure_dir / "case.str").write_text("BLD_1_0.000000\nBLD_1_1.000000\n", encoding="utf-8")
    (tmp_path / "QBladeCE").mkdir()
    airfoil_dir = tmp_path / "airfoils"
    airfoil_dir.mkdir()
    with h5py.File(airfoil_dir / "NACA0018.h5", "w") as h5:
        h5.create_dataset("Airfoilname", data=b"NACA0018")


def test_qblade_bridge_tab_seeds_model_path_from_shared_state():
    state = VorLapGuiState()
    state.set_qblade_sim_path("/tmp/first.sim")
    tab = object.__new__(QBladeBridgeTab)
    tab.app = SimpleNamespace(state=state)
    tab.qblade_sim_path = DummyValue("")
    tab._state_qblade_sim_path = ""
    inferred: list[tuple[str, bool]] = []
    tab._apply_model_airfoil_inference = lambda path, only_if_default=False: inferred.append(
        (str(path), bool(only_if_default))
    ) or False

    QBladeBridgeTab._seed_qblade_sim_from_state(tab, state.qblade_sim_path)
    QBladeBridgeTab._on_app_state_changed(tab, "qblade_sim_path", "/tmp/second.sim")

    assert tab.qblade_sim_path.get() == "/tmp/second.sim"
    assert inferred == [("/tmp/first.sim", True), ("/tmp/second.sim", True)]

    tab.qblade_sim_path.set("/tmp/user-selected.sim")
    QBladeBridgeTab._on_app_state_changed(tab, "qblade_sim_path", "/tmp/third.sim")

    assert tab.qblade_sim_path.get() == "/tmp/user-selected.sim"
    assert inferred == [("/tmp/first.sim", True), ("/tmp/second.sim", True)]


def test_qblade_bridge_tab_apply_defaults_fills_bridge_paths(monkeypatch, tmp_path):
    sim = tmp_path / "model" / "case.sim"
    root = tmp_path / "QBladeCE_2.0.9.7"
    executable = root / root.name
    sim.parent.mkdir()
    root.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.delenv("QBLADE_ROOT", raising=False)
    monkeypatch.delenv("QBLADE_BINARY", raising=False)
    monkeypatch.delenv("QBLADE_EXECUTABLE", raising=False)
    monkeypatch.setattr(bridge_module, "default_qblade_sim_path", lambda: sim)
    monkeypatch.setattr(bridge_module, "default_qblade_root", lambda: root)
    monkeypatch.setattr(bridge_module, "find_qblade_gui_executable", lambda value: executable)

    tab = object.__new__(QBladeBridgeTab)
    tab.airfoil_dir_var = DummyValue("")
    tab.output_dir_var = DummyValue("")
    tab.checkpoint_dir_var = DummyValue("")
    tab.resume_checkpoint_var = DummyValue("")
    tab.qblade_sim_path = DummyValue("")
    tab.qblade_root_var = DummyValue("")
    tab.qblade_binary_var = DummyValue("")
    tab.store_project_var = DummyValue(False)
    tab.store_replay_var = DummyValue("")
    tab.checkpoint_every_var = DummyValue("")
    tab._state_qblade_sim_path = ""
    tab.app = SimpleNamespace(state=SimpleNamespace(qblade_sim_path=""))
    inferred: list[tuple[Path, bool]] = []
    tab._apply_model_airfoil_inference = lambda path, only_if_default=False: inferred.append(
        (Path(path), bool(only_if_default))
    ) or False

    QBladeBridgeTab._apply_defaults(tab)

    assert tab.qblade_sim_path.get() == str(sim)
    assert tab.qblade_root_var.get() == str(root)
    assert tab.qblade_binary_var.get() == str(executable)
    assert tab.output_dir_var.get().endswith("run_headless/runs/gui_qblade_bridge")
    assert tab.checkpoint_dir_var.get().endswith("run_headless/runs/gui_qblade_bridge/checkpoints")
    assert tab.resume_checkpoint_var.get() == ""
    assert tab.store_project_var.get() is True
    assert tab.store_replay_var.get() == "true"
    assert tab.checkpoint_every_var.get() == "200"
    assert inferred == [(sim, True)]


def test_default_python_executable_prefers_repo_venv(monkeypatch, tmp_path):
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setattr(bridge_module.vorlap, "repo_dir", str(tmp_path))
    monkeypatch.delenv("VORLAP_PYTHON", raising=False)
    monkeypatch.delenv("PYTHON_HOST", raising=False)
    monkeypatch.delenv("PYTHON_EXE", raising=False)

    assert bridge_module.default_python_executable() == str(python)


def test_default_python_executable_avoids_frozen_gui_executable(monkeypatch, tmp_path):
    app_exe = tmp_path / "VorLapGui"
    app_exe.write_text("#!/bin/sh\n", encoding="utf-8")
    app_exe.chmod(0o755)
    fallback_python = tmp_path / "bin" / "python3"
    fallback_python.parent.mkdir()
    fallback_python.write_text("#!/bin/sh\n", encoding="utf-8")
    fallback_python.chmod(0o755)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app_exe))
    monkeypatch.setattr(sys, "_base_executable", str(app_exe), raising=False)
    monkeypatch.setattr(bridge_module.vorlap, "repo_dir", str(tmp_path))
    monkeypatch.delenv("VORLAP_PYTHON", raising=False)
    monkeypatch.delenv("PYTHON_HOST", raising=False)
    monkeypatch.delenv("PYTHON_EXE", raising=False)
    monkeypatch.setattr(bridge_module.shutil, "which", lambda command: str(fallback_python) if command == "python3" else None)

    assert bridge_module.default_python_executable() == str(fallback_python)


def test_qblade_bridge_tab_user_actions_are_bridge_workflows_only():
    assert BRIDGE_USER_ACTIONS == (
        "Validate",
        "Prepare case",
        "Build bridge",
        "Run SIL bridge",
        "Run QBlade-driven",
        "Open QBlade GUI",
        "Stop",
    )
    assert "Run native proof" not in BRIDGE_USER_ACTIONS
    assert "Run sign probe" not in BRIDGE_USER_ACTIONS


def make_headless_worker_tab():
    tab = object.__new__(QBladeBridgeTab)
    tab._worker_active = False
    tab._worker = None
    tab._stop_event = threading.Event()
    tab.status_var = DummyStatus()
    action_buttons = [DummyButton(), DummyButton()]
    stop_button = DummyButton()
    tab._action_buttons = action_buttons
    tab.stop_button = stop_button
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.local_log = DummyLog()
    app_messages: list[str] = []
    tab.app = SimpleNamespace(log=app_messages.append)
    callbacks: list[object] = []
    main_thread = threading.get_ident()

    def schedule(_delay_ms, callback):
        assert threading.get_ident() == main_thread, "Worker must not access Tk"
        callbacks.append(callback)

    tab.after = schedule
    return tab, callbacks, inserted, action_buttons, stop_button, app_messages


def test_qblade_bridge_tab_collect_config_uses_typed_bridge_config(tmp_path):
    tab = object.__new__(QBladeBridgeTab)
    tab.qblade_sim_path = DummyValue(tmp_path / "case.sim")
    tab.qblade_root_path = DummyValue(tmp_path / "QBladeCE")
    tab.qblade_binary_path = DummyValue(tmp_path / "QBlade")
    tab.python_path = DummyValue(tmp_path / ".venv" / "bin" / "python")
    tab.airfoil_dir_path = DummyValue(tmp_path / "airfoils")
    tab.output_dir_path = DummyValue(tmp_path / "out")
    tab.coupling_var = DummyValue("none")
    tab.node_source_var = DummyValue("structural")
    tab.include_struts_var = DummyValue(True)
    tab.include_tower_var = DummyValue(False)
    tab.include_tower_nodes_var = DummyValue(False)
    tab.diagnostic_force_scale_var = DummyValue(True)
    tab.include_mean_load_var = DummyValue(True)
    tab.source_bounds_policy_var = DummyValue("raise")
    tab.fluid_density_var = DummyValue("1.3")
    tab.fluid_dynamicviscosity_var = DummyValue("1.7e-5")
    tab.rotation_axis_x_var = DummyValue("0")
    tab.rotation_axis_y_var = DummyValue("0")
    tab.rotation_axis_z_var = DummyValue("1")
    tab.rotation_axis_offset_x_var = DummyValue("1")
    tab.rotation_axis_offset_y_var = DummyValue("2")
    tab.rotation_axis_offset_z_var = DummyValue("3")
    tab.inflow_vec_x_var = DummyValue("1")
    tab.inflow_vec_y_var = DummyValue("1")
    tab.inflow_vec_z_var = DummyValue("0")
    tab.debug_var = DummyValue(True)
    tab.record_history_var = DummyValue(True)
    tab.force_scale_var = DummyValue("2.0")
    tab.n_freq_depth_var = DummyValue("20")
    tab.duration_var = DummyValue("1.5")
    tab.timestep_var = DummyValue("0.25")
    tab.steps_var = DummyValue("")
    tab.wind_speed_var = DummyValue("8.5")
    tab.azimuth_var = DummyValue("45")
    tab.rpm_var = DummyValue("1.25")
    tab.store_from_var = DummyValue("0.5")
    tab.store_replay_var = DummyValue("true")
    tab.threads_var = DummyValue("4")
    tab.use_opencl_var = DummyValue(True)
    tab.device_var = DummyValue("2")
    tab.group_size_var = DummyValue("64")
    tab.allow_partial_var = DummyValue(True)
    tab.store_project_var = DummyValue(True)
    tab.patch_original_model_var = DummyValue(True)
    tab.checkpoint_every_var = DummyValue("10")
    tab.checkpoint_dir_path = DummyValue(tmp_path / "checkpoints")
    tab.resume_from_checkpoint_var = DummyValue(True)
    tab.resume_checkpoint_path = DummyValue(tmp_path / "checkpoints" / "latest_checkpoint.json")
    tab.resume_initialize_var = DummyValue(True)
    tab.sample_step_var = DummyValue("0.05")
    tab.progress_every_var = DummyValue("5")

    config = QBladeBridgeTab.collect_config(tab)

    assert str(config.sim_path) == str(tmp_path / "case.sim")
    assert str(config.qblade_root) == str(tmp_path / "QBladeCE")
    assert str(config.qblade_binary) == str(tmp_path / "QBlade")
    assert str(config.python_executable) == str(tmp_path / ".venv" / "bin" / "python")
    assert config.coupling_mode == "sil-vorlap"
    assert config.include_mean_load is True
    assert config.source_bounds_policy == "raise"
    assert config.fluid_density is None
    assert config.fluid_dynamicviscosity is None
    assert config.rotation_axis is None
    assert config.rotation_axis_offset is None
    assert config.inflow_vec is None
    assert config.force_scale == 2.0
    assert config.diagnostic_force_scale is True
    assert config.effective_steps() == 6
    assert config.threads == 4
    assert config.use_opencl is True
    assert config.opencl_device == 2
    assert config.opencl_group_size == 64
    assert config.allow_partial is True
    assert config.store_project is True
    assert config.patch_original_model is True
    assert config.checkpoint_every == 10
    assert str(config.checkpoint_dir) == str(tmp_path / "checkpoints")
    assert config.resume_from_checkpoint is True
    assert str(config.resume_checkpoint) == str(tmp_path / "checkpoints" / "latest_checkpoint.json")
    assert config.resume_initialize is True
    assert config.sample_step == 0.05
    assert config.wind_speed is None
    assert config.azimuth_deg is None
    assert config.rpm == 1.25
    assert config.store_from_s == 0.5
    assert config.store_replay is True
    assert config.progress_every == 5
    assert config.record_sil_vorlap_history is True


def test_viv_parameter_status_rows_marks_inferred_and_overridden(tmp_path):
    inferred = minimal_bridge_config(tmp_path)
    overridden = minimal_bridge_config(
        tmp_path,
        fluid_density=1.3,
        fluid_dynamicviscosity=1.7e-5,
        rotation_axis=(0.0, 0.0, 1.0),
        rotation_axis_offset=(1.0, 2.0, 3.0),
        inflow_vec=(1.0, 1.0, 0.0),
    )

    assert viv_parameter_status_rows(inferred) == []
    rows = dict(viv_parameter_status_rows(overridden))
    assert rows["Fluid density"] == "manual override: 1.3 kg/m^3"
    assert rows["Dynamic viscosity"] == "manual override: 1.7e-05 Pa s"
    assert rows["Rotation axis"] == "manual override: 0.0, 0.0, 1.0"
    assert rows["Axis offset"] == "manual override: 1.0, 2.0, 3.0 m"
    assert "Inflow vector" not in rows


def test_bridge_scenario_config_maps_run_modes(tmp_path):
    config = minimal_bridge_config(tmp_path, force_scale=1.5, diagnostic_force_scale=True)

    nominal = bridge_scenario_config(config, "nominal VorLap")
    baseline = bridge_scenario_config(config, "baseline/no coupling")
    identity = bridge_scenario_config(config, "identity force scale 0")
    sensitivity = bridge_scenario_config(config, "diagnostic amplitude sensitivity")

    assert BRIDGE_SCENARIOS[0] == "nominal VorLap"
    assert nominal.coupling_mode == "sil-vorlap"
    assert nominal.force_scale == 1.5
    assert baseline.coupling_mode == "none"
    assert baseline.force_scale == 0.0
    assert baseline.diagnostic_force_scale is False
    assert identity.coupling_mode == "sil-vorlap"
    assert identity.force_scale == 0.0
    assert identity.diagnostic_force_scale is False
    assert sensitivity.coupling_mode == "sil-vorlap"
    assert sensitivity.force_scale == 1.5
    assert sensitivity.diagnostic_force_scale is True


def test_bridge_config_warnings_flag_coarse_timesteps(tmp_path):
    assert bridge_config_warnings(minimal_bridge_config(tmp_path, timestep_s=0.05)) == ()

    warnings = bridge_config_warnings(minimal_bridge_config(tmp_path, timestep_s=0.25))

    assert len(warnings) == 1
    assert "0.25 s" in warnings[0]
    assert "0.1 s bridge smoke-test range" in warnings[0]


def test_infer_model_airfoil_dir_prefers_direct_then_model_child(tmp_path):
    model_dir = tmp_path / "qblade_case"
    sim = model_dir / "case.sim"
    nested = model_dir / "reference_model" / "VorLapAirfoils"
    nested.mkdir(parents=True)
    sim.write_text("simulation placeholder\n", encoding="utf-8")

    assert infer_model_airfoil_dir(sim) == nested.resolve()

    direct = model_dir / "VorLapAirfoils"
    direct.mkdir()

    assert infer_model_airfoil_dir(sim) == direct.resolve()
    assert infer_model_airfoil_dir(tmp_path / "missing.sim") is None


def test_qblade_bridge_tab_selected_model_infers_local_airfoil_dir(tmp_path):
    model_dir = tmp_path / "qblade_case"
    sim = model_dir / "case.sim"
    local_airfoils = model_dir / "reference_model" / "VorLapAirfoils"
    local_airfoils.mkdir(parents=True)
    sim.write_text("simulation placeholder\n", encoding="utf-8")
    tab = object.__new__(QBladeBridgeTab)
    tab.airfoil_dir_path = DummyValue("")
    tab.local_log = DummyLog()
    app_messages: list[str] = []
    tab.app = SimpleNamespace(log=app_messages.append)

    QBladeBridgeTab._on_qblade_sim_selected(tab, str(sim))

    assert tab.airfoil_dir_path.get() == str(local_airfoils.resolve())
    assert tab.local_log.messages == [f"Using model-local VorLap airfoils: {local_airfoils.resolve()}\n"]
    assert app_messages == tab.local_log.messages


def test_prepare_status_rows_render_key_artifacts_and_diagnostics():
    result = SimpleNamespace(
        state="prepared",
        artifacts={
            "runtime_config": "/tmp/case/Control/vorlap_qblade_external.json",
            "geometry_audit": "/tmp/case/Control/vorlap_qblade_geometry_audit.csv",
            "working_sim": "/tmp/out/qblade_model_working/case.sim",
            "working_copy_manifest": "/tmp/out/qblade_model_working_manifest.json",
        },
        diagnostics={
            "node_count": 3,
            "swap_size": 22,
            "component_count": 2,
            "node_source": "structural",
            "source_bounds_policy": "raise",
            "geometry_audit_exists": True,
            "geometry_audit_rows": 3,
            "geometry_audit_basis_error_count": 1,
            "geometry_audit_basis_warning_count": 2,
            "geometry_audit_basis_error_nodes": ["BLD_1_1.000000"],
            "geometry_audit_duplicate_node_count": 0,
            "qblade_model_override_count": 2,
            "working_copy_file_count": 4,
            "dry_run": False,
        },
    )

    rows = dict(prepare_status_rows(result))

    assert rows["State"] == "prepared"
    assert rows["Runtime config"].endswith("vorlap_qblade_external.json")
    assert rows["Geometry audit"].endswith("vorlap_qblade_geometry_audit.csv")
    assert rows["Working sim"].endswith("qblade_model_working/case.sim")
    assert rows["Working manifest"].endswith("qblade_model_working_manifest.json")
    assert rows["Node count"] == "3"
    assert rows["Swap size"] == "22"
    assert rows["Component count"] == "2"
    assert rows["Node source"] == "structural"
    assert rows["Source policy"] == "raise"
    assert rows["Geometry audit exists"] == "True"
    assert rows["Geometry audit rows"] == "3"
    assert rows["Geometry basis errors"] == "1"
    assert rows["Geometry basis warnings"] == "2"
    assert rows["Geometry basis error nodes"] == "BLD_1_1.000000"
    assert rows["Geometry duplicate nodes"] == "0"
    assert rows["Model overrides"] == "2"
    assert rows["Working copy files"] == "4"
    assert rows["Dry run"] == "False"


def test_run_status_rows_render_progress_and_outputs():
    result = SimpleNamespace(
        state="complete",
        artifacts={
            "summary": "/tmp/out/case_run_summary.json",
            "result_file": "/tmp/out/case_results.txt",
            "run_log": "/tmp/out/case_qblade.log",
            "sil_vorlap_history": "/tmp/out/history.csv",
            "sil_vorlap_source_coverage": "/tmp/out/case_run_summary_source_coverage.csv",
            "bridge_quick_plot_dir": "/tmp/out/case_run_summary_quick_plots",
            "bridge_quick_plot_max_force": "/tmp/out/case_run_summary_quick_plots/max_nodal_force_by_component.png",
            "bridge_quick_plot_aoa_re": "/tmp/out/case_run_summary_quick_plots/aoa_re_source_limits.png",
            "bridge_quick_plot_nyquist": "/tmp/out/case_run_summary_quick_plots/nyquist_margin.png",
            "bridge_quick_plot_force_history": "/tmp/out/case_run_summary_quick_plots/selected_force_history.png",
            "bridge_quick_plot_qblade_response": "/tmp/out/case_run_summary_quick_plots/selected_qblade_response.png",
            "working_sim": "/tmp/out/qblade_model_working/case.sim",
            "working_copy_manifest": "/tmp/out/qblade_model_working_manifest.json",
        },
        diagnostics={
            "completed_steps": 10,
            "requested_steps": 10,
            "coupling_mode": "sil-vorlap",
            "sil_vorlap_source_coverage_rows": 2,
            "source_bounds_policy": "raise",
            "geometry_audit_reported_available_count": 39,
            "geometry_audit_reported_missing_count": 0,
            "geometry_audit_reported_error_count": 39,
            "geometry_audit_reported_max_delta_m": 8.19254,
            "geometry_audit_reported_frame_fit_status": "within_tolerance",
            "geometry_audit_reported_frame_fit_sample_count": 39,
            "geometry_audit_reported_frame_fit_max_residual_m": 0.059659,
            "geometry_audit_reported_frame_fit_mean_residual_m": 0.036411,
            "geometry_audit_reported_frame_fit_translation_m": [0.0, 0.0, -3.258147],
            "qblade_model_override_count": 2,
            "working_copy_file_count": 4,
            "stopped": False,
            "failed": False,
            "sil_vorlap_max_abs_force": 123.4,
            "sil_vorlap_max_structural_velocity_mps": 1.2,
            "sil_vorlap_cache": "3/1",
            "sil_vorlap_velocity_warning_count": 2,
            "qblade_log_tail_path": "/tmp/out/case_qblade.log",
            "qblade_log_tail_line_count": 5,
            "bridge_quick_plot_count": 5,
            "bridge_quick_plot_errors": [],
            "bridge_quick_plot_force_history_column": "max_force_norm_n",
            "bridge_quick_plot_qblade_response_channel": "X_g~Pos.~BLD_1~pos~0.000~[m]",
            "sil_vorlap_velocity_diagnostics": {
                "axis_stats": {
                    "X": {"samples": 4, "min": -0.5, "max": 1.2, "max_abs": 1.2, "all_zero": False},
                    "Y": {"samples": 4, "min": 0.0, "max": 0.1, "max_abs": 0.1, "all_zero": False},
                    "Z": {"samples": 4, "min": 0.0, "max": 0.0, "max_abs": 0.0, "all_zero": True},
                },
                "all_zero_axes": ["Z"],
                "all_zero_channels": ["Z_g Position BLD_1_0.000000 [m/s]"],
            },
            "sil_vorlap_source_coverage": [
                {
                    "component": "BLD_1",
                    "samples": 4,
                    "node_samples": 8,
                    "aoa_min_deg": -2.5,
                    "aoa_max_deg": 4.0,
                    "source_aoa_min_deg": -15.0,
                    "source_aoa_max_deg": 15.0,
                    "re_min": 100000.0,
                    "re_max": 250000.0,
                    "source_re_min": 50000.0,
                    "source_re_max": 500000.0,
                    "freq_min_hz": 0.0,
                    "freq_max_hz": 2.5,
                    "nyquist_hz": 5.0,
                    "nyquist_ratio": 2.0,
                    "max_nodal_force_n": 12.5,
                    "force_coefficient_max_abs": 0.4,
                    "source_force_coefficient_max_abs": 0.8,
                    "max_structural_velocity_mps": 1.1,
                    "force_power_mean_w": 0.2,
                    "force_power_max_abs_w": 0.7,
                    "re_out_of_range_count": 0,
                    "aoa_out_of_range_count": 1,
                    "aliased_frequency_count": 0,
                    "source_clip_count": 1,
                    "force_coefficient_out_of_range_count": 0,
                    "nonfinite_count": 0,
                }
            ],
            "native_callback_evidence": {
                "callback_block_present": True,
                "turbine_file_resolved": "/tmp/case/reference_model.trb",
                "callback_block_error": None,
                "library_stem": "libvorlap_qblade_bridge",
                "function_name": "update",
                "array_size": "15",
                "parameter_file_exists": True,
                "library_symbol_check_available": True,
                "library_exported_required_symbols": ["update", "update_init", "update_message"],
                "library_missing_required_symbols": [],
                "update_init_seen": True,
                "update_init_count": 1,
                "update_log_entry_count": 10,
                "update_count_observed_max": 10,
                "log_file": "/tmp/out/bridge.log",
                "python_env_before_import_seen": True,
                "python_env_after_import_seen": True,
                "python_env_parse_errors": [],
                "numpy_version": "2.2.6",
                "numpy_file": "/tmp/venv/site-packages/numpy/__init__.py",
                "vorlap_file": "/tmp/VorLap/vorlap/__init__.py",
                "vorlap_qblade_runtime_file": "/tmp/VorLap/vorlap/qblade_runtime.py",
            },
            "checkpoint_latest": "/tmp/out/checkpoints/latest_checkpoint.json",
            "checkpoint_latest_completed_steps": 8,
            "checkpoint_latest_project": "/tmp/out/checkpoints/case_step_0000008.qpr",
            "resume_requested": True,
            "resume_start_step": 4,
            "sil_vorlap_message": "VorLap QBlade runtime initialized.",
        },
    )

    rows = dict(run_status_rows(result))

    assert rows["State"] == "complete"
    assert rows["Summary"].endswith("case_run_summary.json")
    assert rows["Result file"].endswith("case_results.txt")
    assert rows["Run log"].endswith("case_qblade.log")
    assert rows["QBlade log tail"].endswith("case_qblade.log")
    assert rows["QBlade log lines"] == "5"
    assert rows["SIL history"].endswith("history.csv")
    assert rows["Source coverage"].endswith("case_run_summary_source_coverage.csv")
    assert rows["Quick plot directory"].endswith("case_run_summary_quick_plots")
    assert rows["Plot max force"].endswith("max_nodal_force_by_component.png")
    assert rows["Plot AOA/Re limits"].endswith("aoa_re_source_limits.png")
    assert rows["Plot Nyquist margin"].endswith("nyquist_margin.png")
    assert rows["Plot force history"].endswith("selected_force_history.png")
    assert rows["Plot QBlade response"].endswith("selected_qblade_response.png")
    assert rows["Quick plot count"] == "5"
    assert rows["Quick plot errors"] == ""
    assert rows["Force history column"] == "max_force_norm_n"
    assert rows["QBlade response channel"] == "X_g~Pos.~BLD_1~pos~0.000~[m]"
    assert rows["Working sim"].endswith("qblade_model_working/case.sim")
    assert rows["Working manifest"].endswith("qblade_model_working_manifest.json")
    assert rows["Completed steps"] == "10"
    assert rows["Requested steps"] == "10"
    assert rows["Coupling"] == "sil-vorlap"
    assert rows["Source coverage rows"] == "2"
    assert rows["Source policy"] == "raise"
    assert rows["Geometry reported positions"] == "available 39; missing 0; direct errors 39"
    assert rows["Geometry direct max delta [m]"] == "8.19254"
    assert rows["Geometry fitted frame"] == "status within_tolerance; samples 39; max residual 0.059659 m; mean residual 0.036411 m"
    assert rows["Geometry fitted translation [m]"] == "0.0, 0.0, -3.258147"
    assert rows["Model overrides"] == "2"
    assert rows["Working copy files"] == "4"
    assert rows["Stopped"] == "False"
    assert rows["Failed"] == "False"
    assert rows["Max force [N]"] == "123.4"
    assert rows["Max structural velocity [m/s]"] == "1.2"
    assert rows["Cache hits/misses"] == "3/1"
    assert rows["Velocity warnings"] == "2"
    assert rows["Velocity X [m/s]"] == "-0.5..1.2; max abs 1.2; samples 4; all zero False"
    assert rows["Velocity Z [m/s]"] == "0..0; max abs 0; samples 4; all zero True"
    assert rows["Velocity zero axes"] == "Z"
    assert rows["Velocity zero channels"] == "Z_g Position BLD_1_0.000000 [m/s]"
    assert rows["Source BLD_1 samples"] == "4 samples, 8 node samples"
    assert rows["Source BLD_1 AOA [deg]"] == "-2.5..4 (source -15..15)"
    assert rows["Source BLD_1 Re"] == "100000..250000 (source 50000..500000)"
    assert rows["Source BLD_1 frequency [Hz]"] == "0..2.5; Nyquist 5; ratio 2"
    assert rows["Source BLD_1 load"] == "max force 12.5 N; C 0.4/0.8; max Vstruct 1.1 m/s; power mean/max 0.2/0.7 W"
    assert rows["Source BLD_1 counts"] == "Re out 0; AOA out 1; clips 1; alias 0; coeff out 0; nonfinite 0"
    assert rows["Native callback block"] == "present True; turbine /tmp/case/reference_model.trb; error "
    assert rows["Native callback target"] == "library libvorlap_qblade_bridge; function update; array 15; parameter exists True"
    assert rows["Native library symbols"] == "checked True; exported update, update_init, update_message; missing none"
    assert rows["Native update_init"] == "seen True; count 1"
    assert rows["Native update calls"] == "log entries 10; observed max 10"
    assert rows["Native callback log"] == "/tmp/out/bridge.log"
    assert rows["Native Python env"] == "before import True; after import True; parse errors none"
    assert rows["Native imports"].startswith("NumPy 2.2.6 at /tmp/venv/site-packages/numpy/__init__.py")
    assert rows["Latest checkpoint"].endswith("latest_checkpoint.json")
    assert rows["Latest checkpoint step"] == "8"
    assert rows["Checkpoint project"].endswith("case_step_0000008.qpr")
    assert rows["Resume requested"] == "True"
    assert rows["Resume start step"] == "4"
    assert rows["Runtime message"] == "VorLap QBlade runtime initialized."


def test_build_status_rows_render_bridge_artifacts():
    result = SimpleNamespace(
        state="built",
        artifacts={
            "controller_files": "/tmp/QBladeCE/ControllerFiles",
            "build_dir": "/tmp/out/build/qblade_external_linux",
            "bridge_library": "/tmp/QBladeCE/ControllerFiles/libvorlap_qblade_bridge.so",
        },
        diagnostics={
            "python": "/tmp/.venv/bin/python",
            "python_executable": "/tmp/.venv/bin/python",
            "python_version": "3.11.9",
            "numpy_version": "2.2.6",
            "numpy_error": "",
            "python_probe_error": "",
            "cmake_version": "cmake version 3.28.0",
            "cmake_error": "",
            "compiler": "cc (example) 1.0",
            "compiler_error": "",
            "qblade_ld_library_path": "/tmp/QBladeCE:/tmp/QBladeCE/ControllerFiles",
            "qblade_loader_path_key": "LD_LIBRARY_PATH",
            "qblade_loader_path": "/tmp/QBladeCE:/tmp/QBladeCE/ControllerFiles",
            "force": False,
            "bridge_library_exists": True,
            "bridge_library_loadable": False,
            "bridge_library_load_error": "file too short",
        },
    )

    rows = dict(build_status_rows(result))

    assert rows["State"] == "built"
    assert rows["ControllerFiles"].endswith("ControllerFiles")
    assert rows["Build directory"].endswith("qblade_external_linux")
    assert rows["Bridge library"].endswith("libvorlap_qblade_bridge.so")
    assert rows["Python"].endswith("python")
    assert rows["Python executable"].endswith("python")
    assert rows["Python version"] == "3.11.9"
    assert rows["NumPy version"] == "2.2.6"
    assert rows["CMake version"].startswith("cmake version")
    assert rows["Compiler"].startswith("cc")
    assert rows["QBlade loader variable"] == "LD_LIBRARY_PATH"
    assert rows["QBlade loader path"].endswith("ControllerFiles")
    assert rows["QBlade library path"].endswith("ControllerFiles")
    assert rows["Force rebuild"] == "False"
    assert rows["Library exists"] == "True"
    assert rows["Library loadable"] == "False"
    assert rows["Library load error"] == "file too short"


def test_sign_probe_status_rows_render_report_artifacts():
    result = SimpleNamespace(
        state="complete",
        artifacts={
            "sign_probe_plan": "/tmp/out/sign_probes/force_sign_probe_plan.json",
            "sign_probe_report_csv": "/tmp/out/sign_probes/force_sign_probe_report.csv",
            "sign_probe_report_json": "/tmp/out/sign_probes/force_sign_probe_report.json",
            "sign_probe_output_root": "/tmp/out/sign_probes",
        },
        diagnostics={
            "node": "BLD_1_0.000000",
            "axes": ["X", "Z"],
            "force_n": 750.0,
            "local": False,
            "probes_completed": 4,
            "failures": 0,
            "passed": True,
            "geometry_audit_exists": True,
            "geometry_audit_rows": 3,
            "geometry_audit_basis_error_count": 0,
            "geometry_audit_basis_warning_count": 1,
            "geometry_audit_basis_error_nodes": [],
            "geometry_audit_duplicate_node_count": 0,
            "sign_probe_report_rows": [
                {
                    "node": "BLD_1_0.000000",
                    "axis": "X",
                    "sign": "positive",
                    "passed": "true",
                    "paired_passed": "true",
                    "delta_position_m": 0.00075,
                    "paired_response_delta_m": 0.0015,
                    "observation_status": "same_sign",
                    "observed_position_channel": "X_g Position BLD_1_0.000000 [m]",
                    "issue": "",
                },
                {
                    "node": "BLD_1_0.000000",
                    "axis": "Z",
                    "sign": "negative",
                    "passed": "false",
                    "paired_passed": "false",
                    "delta_position_m": "",
                    "paired_response_delta_m": "",
                    "observation_status": "missing_position_channel",
                    "observed_position_channel": "",
                    "issue": "missing_position_channel",
                },
            ],
        },
    )

    rows = dict(sign_probe_status_rows(result))

    assert rows["State"] == "complete"
    assert rows["Plan"].endswith("force_sign_probe_plan.json")
    assert rows["Report CSV"].endswith("force_sign_probe_report.csv")
    assert rows["Report JSON"].endswith("force_sign_probe_report.json")
    assert rows["Output root"].endswith("sign_probes")
    assert rows["Node"] == "BLD_1_0.000000"
    assert rows["Axes"] == "X, Z"
    assert rows["Force [N]"] == "750.0"
    assert rows["Local"] == "False"
    assert rows["Probes completed"] == "4"
    assert rows["Failures"] == "0"
    assert rows["Passed"] == "True"
    assert rows["Geometry audit exists"] == "True"
    assert rows["Geometry audit rows"] == "3"
    assert rows["Geometry basis errors"] == "0"
    assert rows["Geometry basis warnings"] == "1"
    assert rows["Geometry basis error nodes"] == ""
    assert rows["Geometry duplicate nodes"] == "0"
    assert (
        rows["Probe BLD_1_0.000000 X positive"]
        == "passed true; pair true; delta 0.00075 m; pair delta 0.0015 m; status same_sign; "
        "channel X_g Position BLD_1_0.000000 [m]; issue none"
    )
    assert (
        rows["Probe BLD_1_0.000000 Z negative"]
        == "passed false; pair false; delta  m; pair delta  m; status missing_position_channel; "
        "channel missing position channel; issue missing_position_channel"
    )


def test_qblade_bridge_tab_prepare_complete_sets_rows_and_status():
    tab = object.__new__(QBladeBridgeTab)
    tab._worker_active = True
    tab.status_var = DummyStatus()
    action_buttons = [DummyButton(), DummyButton()]
    stop_button = DummyButton()
    tab._action_buttons = action_buttons
    tab.stop_button = stop_button
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.local_log = DummyLog()
    app_messages: list[str] = []
    tab.app = SimpleNamespace(log=app_messages.append)
    result = SimpleNamespace(
        state="prepared",
        artifacts={
            "runtime_config": "/tmp/case/Control/vorlap_qblade_external.json",
            "geometry_audit": "/tmp/case/Control/vorlap_qblade_geometry_audit.csv",
            "working_sim": "/tmp/out/qblade_model_working/case.sim",
            "working_copy_manifest": "/tmp/out/qblade_model_working_manifest.json",
        },
        diagnostics={
            "node_count": 3,
            "swap_size": 22,
            "component_count": 2,
            "node_source": "structural",
            "source_bounds_policy": "raise",
            "geometry_audit_exists": True,
            "geometry_audit_rows": 3,
            "qblade_model_override_count": 2,
            "working_copy_file_count": 4,
            "dry_run": False,
        },
    )

    QBladeBridgeTab._on_prepare_complete(tab, result)

    assert tab._worker_active is False
    assert tab.status_var.value == "prepared"
    assert action_buttons[0].state == "normal"
    assert stop_button.state == "disabled"
    assert ("State", "prepared") in inserted
    assert ("Runtime config", "/tmp/case/Control/vorlap_qblade_external.json") in inserted
    assert ("Working manifest", "/tmp/out/qblade_model_working_manifest.json") in inserted
    assert tab.local_log.messages == ["QBlade bridge case prepared.\n"]
    assert app_messages == tab.local_log.messages


def test_qblade_bridge_tab_build_complete_sets_rows_and_status():
    tab = object.__new__(QBladeBridgeTab)
    tab._worker_active = True
    tab.status_var = DummyStatus()
    action_buttons = [DummyButton(), DummyButton()]
    stop_button = DummyButton()
    tab._action_buttons = action_buttons
    tab.stop_button = stop_button
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.local_log = DummyLog()
    app_messages: list[str] = []
    tab.app = SimpleNamespace(log=app_messages.append)
    result = SimpleNamespace(
        state="built",
        artifacts={
            "controller_files": "/tmp/QBladeCE/ControllerFiles",
            "build_dir": "/tmp/out/build/qblade_external_linux",
            "bridge_library": "/tmp/QBladeCE/ControllerFiles/libvorlap_qblade_bridge.so",
        },
        diagnostics={
            "python": "/tmp/.venv/bin/python",
            "force": False,
            "bridge_library_exists": True,
        },
    )

    QBladeBridgeTab._on_build_complete(tab, result)

    assert tab._worker_active is False
    assert tab.status_var.value == "built"
    assert action_buttons[0].state == "normal"
    assert stop_button.state == "disabled"
    assert ("State", "built") in inserted
    assert tab.local_log.messages == ["QBlade bridge library built.\n"]
    assert app_messages == tab.local_log.messages


def test_qblade_bridge_tab_sign_probe_complete_sets_rows_and_status():
    tab = object.__new__(QBladeBridgeTab)
    tab._worker_active = True
    tab.status_var = DummyStatus()
    action_buttons = [DummyButton(), DummyButton()]
    stop_button = DummyButton()
    tab._action_buttons = action_buttons
    tab.stop_button = stop_button
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.local_log = DummyLog()
    app_messages: list[str] = []
    tab.app = SimpleNamespace(log=app_messages.append)
    result = SimpleNamespace(
        state="complete",
        artifacts={
            "sign_probe_plan": "/tmp/out/sign_probes/force_sign_probe_plan.json",
            "sign_probe_report_csv": "/tmp/out/sign_probes/force_sign_probe_report.csv",
            "sign_probe_report_json": "/tmp/out/sign_probes/force_sign_probe_report.json",
            "sign_probe_output_root": "/tmp/out/sign_probes",
        },
        diagnostics={
            "node": "BLD_1_0.000000",
            "axes": ["X", "Z"],
            "force_n": 750.0,
            "local": False,
            "probes_completed": 4,
            "failures": 0,
            "passed": True,
            "sign_probe_report_rows": [
                {
                    "node": "BLD_1_0.000000",
                    "axis": "X",
                    "sign": "positive",
                    "passed": "true",
                    "paired_passed": "true",
                    "delta_position_m": 0.00075,
                    "paired_response_delta_m": 0.0015,
                    "observation_status": "same_sign",
                    "observed_position_channel": "X_g Position BLD_1_0.000000 [m]",
                    "issue": "",
                },
            ],
        },
    )

    QBladeBridgeTab._on_sign_probe_complete(tab, result)

    assert tab._worker_active is False
    assert tab.status_var.value == "sign probe complete"
    assert action_buttons[0].state == "normal"
    assert stop_button.state == "disabled"
    assert ("State", "complete") in inserted
    assert ("Report CSV", "/tmp/out/sign_probes/force_sign_probe_report.csv") in inserted
    assert (
        "Probe BLD_1_0.000000 X positive",
        "passed true; pair true; delta 0.00075 m; pair delta 0.0015 m; status same_sign; "
        "channel X_g Position BLD_1_0.000000 [m]; issue none",
    ) in inserted
    assert tab.local_log.messages == ["QBlade sign probe complete.\n"]
    assert app_messages == tab.local_log.messages


def test_qblade_bridge_tab_prepare_reports_invalid_config_before_worker(monkeypatch):
    tab = object.__new__(QBladeBridgeTab)
    tab.status_var = DummyStatus()
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.collect_config = lambda: (_ for _ in ()).throw(ValueError("bad force scale"))
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr("vorlap.gui.tabs.qblade_bridge.messagebox.showerror", lambda title, message: errors.append((title, message)))

    QBladeBridgeTab.prepare_case(tab)

    assert started == []
    assert tab.status_var.value == "validation failed"
    assert inserted == [("Error", "bad force scale")]
    assert errors == [("Validation failed", "bad force scale")]


def test_qblade_bridge_tab_validate_inputs_renders_coarse_timestep_warning(tmp_path):
    config = minimal_bridge_config(tmp_path, timestep_s=0.25)

    class Service:
        def validate_config(self, value, *, check_paths=False, check_qblade_install=False, check_qblade_model=False):
            return SimpleNamespace(ok=True, qblade_gui_executable="/tmp/QBlade")

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.collect_config = lambda: config
    tab.status_var = DummyStatus()
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.local_log = DummyLog()
    app_messages: list[str] = []
    tab.app = SimpleNamespace(log=app_messages.append)

    QBladeBridgeTab.validate_inputs(tab)

    rows = dict(inserted)
    assert tab.status_var.value == "valid"
    assert rows["State"] == "valid"
    assert rows["QBlade GUI"] == "/tmp/QBlade"
    assert "0.25 s" in rows["Warning"]
    assert tab.local_log.messages == ["Bridge inputs validated.\n"]
    assert app_messages == tab.local_log.messages


def test_qblade_bridge_tab_open_qblade_gui_renders_launch_plan(tmp_path):
    config = minimal_bridge_config(tmp_path)
    executable = tmp_path / "QBladeCE" / "QBlade"
    cwd = tmp_path / "QBladeCE"
    case_path = tmp_path / "case.sim"
    project_path = tmp_path / "out" / "case_completed.qpr"
    plan = SimpleNamespace(
        executable=executable,
        command=(str(executable), str(project_path)),
        cwd=cwd,
        case_path_hint=case_path,
        project_path=project_path,
        project_source="completed run project",
    )
    last_run = SimpleNamespace(state="complete")

    class Service:
        def launch_qblade_gui(self, value, *, run_result=None):
            assert value is config
            assert run_result is last_run
            return plan, "fake-process"

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.collect_config = lambda: config
    tab._last_run_result = last_run
    tab.status_var = DummyStatus()
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.local_log = DummyLog()
    app_messages: list[str] = []
    tab.app = SimpleNamespace(log=app_messages.append)

    QBladeBridgeTab.open_qblade_gui(tab)

    rows = dict(inserted)
    assert tab.status_var.value == "qblade launched"
    assert rows["Executable"] == str(executable)
    assert rows["Command"] == f"{executable} {project_path}"
    assert rows["Working directory"] == str(cwd)
    assert rows["Case path"] == str(case_path)
    assert rows["Opened project"] == str(project_path)
    assert rows["Project source"] == "completed run project"
    assert rows["Run visibility"] == "QBlade GUI inspection is separate from the automated VorLap run."
    assert tab.local_log.messages == [f"Launched QBlade GUI: {executable} {project_path}\n"]
    assert app_messages == tab.local_log.messages


def test_qblade_bridge_tab_prepare_uses_sil_coupling_only(tmp_path):
    config = minimal_bridge_config(tmp_path, force_scale=1.0)
    validated: list[QBladeBridgeConfig] = []

    class Service:
        def validate_config(self, value, *, check_paths=False, check_qblade_install=False, check_qblade_model=False):
            validated.append(value)
            return SimpleNamespace(errors=())

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.collect_config = lambda: config
    tab.scenario_var = DummyValue("identity force scale 0")
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)

    QBladeBridgeTab.prepare_case(tab)

    assert started
    assert len(validated) == 1
    assert validated[0].coupling_mode == "sil-vorlap"
    assert validated[0].force_scale == 1.0
    assert validated[0].diagnostic_force_scale is False


def test_qblade_bridge_tab_run_validates_paths_before_worker(tmp_path):
    config = minimal_bridge_config(tmp_path)
    calls: list[tuple[object, bool, bool, bool]] = []

    class Service:
        def validate_config(self, value, *, check_paths=False, check_qblade_install=False, check_qblade_model=False):
            calls.append((value, check_paths, check_qblade_install, check_qblade_model))
            return SimpleNamespace(errors=("QBlade simulation file does not exist: case.sim",))

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.status_var = DummyStatus()
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.collect_config = lambda: config
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)

    QBladeBridgeTab.run_sil_bridge(tab)

    assert started == []
    assert len(calls) == 1
    validated_config, check_paths, check_qblade_install, check_qblade_model = calls[0]
    assert validated_config.coupling_mode == "sil-vorlap"
    assert check_paths is True
    assert check_qblade_install is False
    assert check_qblade_model is True
    assert tab.status_var.value == "validation failed"
    assert inserted == [("Error", "QBlade simulation file does not exist: case.sim")]


def test_qblade_bridge_tab_run_uses_sil_coupling_only(tmp_path):
    config = minimal_bridge_config(tmp_path, force_scale=1.0)
    validated: list[QBladeBridgeConfig] = []
    isolated_calls: list[tuple[QBladeBridgeConfig, bool]] = []

    class Service:
        def validate_config(self, value, *, check_paths=False, check_qblade_install=False, check_qblade_model=False):
            validated.append(value)
            return SimpleNamespace(errors=())

        def run_case_isolated(self, value, *, message_callback=None, stop_requested=None, check_paths=False):
            isolated_calls.append((value, check_paths))
            return "isolated-result"

        def run_case(self, *args, **kwargs):
            raise AssertionError("Run SIL bridge should not load QBlade in the GUI process")

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.collect_config = lambda: config
    tab.scenario_var = DummyValue("baseline/no coupling")
    tab._stop_event = threading.Event()
    tab._threadsafe_log = lambda message: None
    tab._on_run_complete = lambda result: None
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)

    QBladeBridgeTab.run_sil_bridge(tab)

    assert started
    assert len(validated) == 1
    assert validated[0].coupling_mode == "sil-vorlap"
    assert validated[0].force_scale == 1.0
    state, work, on_success = started[0]
    assert state == "running"
    assert work() == "isolated-result"
    assert on_success == tab._on_run_complete
    assert isolated_calls == [(validated[0], True)]


def test_qblade_bridge_tab_qblade_driven_selects_qblade_full_run_mode(tmp_path):
    config = minimal_bridge_config(
        tmp_path,
        coupling_mode="sil-vorlap",
        checkpoint_every=10,
        checkpoint_dir=tmp_path / "checkpoints",
        resume_from_checkpoint=True,
        resume_checkpoint=tmp_path / "checkpoints" / "latest_checkpoint.json",
        resume_initialize=True,
        allow_partial=True,
    )
    validated: list[QBladeBridgeConfig] = []

    class Service:
        def validate_config(self, value, *, check_paths=False, check_qblade_install=False, check_qblade_model=False):
            validated.append(value)
            return SimpleNamespace(errors=())

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.collect_config = lambda: config
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)

    QBladeBridgeTab.run_qblade_driven_bridge(tab)

    assert started
    assert len(validated) == 1
    assert validated[0].coupling_mode == "qblade-external"
    assert validated[0].qblade_execution_mode == "run-full-simulation"
    assert validated[0].checkpoint_every == 0
    assert validated[0].checkpoint_dir is None
    assert validated[0].resume_from_checkpoint is False
    assert validated[0].resume_checkpoint is None
    assert validated[0].resume_initialize is False
    assert validated[0].allow_partial is False
    state, work, on_success = started[0]
    assert state == "running QBlade-driven"
    assert on_success == tab._on_run_complete


def test_qblade_bridge_tab_native_proof_alias_uses_qblade_driven_path(tmp_path):
    tab = object.__new__(QBladeBridgeTab)
    called: list[bool] = []
    tab.run_qblade_driven_bridge = lambda: called.append(True)

    QBladeBridgeTab.run_native_callback_proof(tab)

    assert called == [True]


def test_qblade_bridge_tab_run_reports_bad_output_dir_before_worker(tmp_path):
    write_minimal_valid_bridge_paths(tmp_path)
    output_file = tmp_path / "out-is-a-file"
    output_file.write_text("not a directory\n", encoding="utf-8")
    config = minimal_bridge_config(tmp_path, output_dir=output_file)
    tab = object.__new__(QBladeBridgeTab)
    tab.service = QBladeBridgeService()
    tab.status_var = DummyStatus()
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.collect_config = lambda: config
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)

    QBladeBridgeTab.run_sil_bridge(tab)

    assert started == []
    assert tab.status_var.value == "validation failed"
    assert inserted == [("Error", f"QBlade output directory is not a directory: {output_file}")]


def test_qblade_bridge_tab_run_rejects_resume_checkpoint_mismatch_before_worker(tmp_path):
    write_minimal_valid_bridge_paths(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    project = checkpoint_dir / "case_step_0000004.qpr"
    project.write_text("checkpoint project\n", encoding="utf-8")
    checkpoint = checkpoint_dir / "latest_checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "sim": str(tmp_path / "different.sim"),
                "qblade_root": str(tmp_path / "QBladeCE"),
                "qblade_project": str(project),
                "coupling_mode": "sil-vorlap",
                "completed_steps": 4,
                "threads": 1,
                "use_opencl": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = minimal_bridge_config(
        tmp_path,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=True,
    )
    tab = object.__new__(QBladeBridgeTab)
    tab.service = QBladeBridgeService()
    tab.status_var = DummyStatus()
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.collect_config = lambda: config
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)

    QBladeBridgeTab.run_sil_bridge(tab)

    assert started == []
    assert tab.status_var.value == "validation failed"
    assert len(inserted) == 1
    assert inserted[0][0] == "Error"
    assert "Resume checkpoint sim mismatch" in inserted[0][1]


def test_qblade_bridge_tab_build_validates_qblade_install_before_worker(tmp_path):
    config = minimal_bridge_config(tmp_path)
    calls: list[tuple[object, bool, bool, bool]] = []

    class Service:
        def validate_config(self, value, *, check_paths=False, check_qblade_install=False, check_qblade_model=False):
            calls.append((value, check_paths, check_qblade_install, check_qblade_model))
            return SimpleNamespace(errors=("Incomplete QBlade root.",))

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.status_var = DummyStatus()
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.collect_config = lambda: config
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)

    QBladeBridgeTab.build_bridge(tab)

    assert started == []
    assert len(calls) == 1
    assert calls[0][1] is False
    assert calls[0][2] is True
    assert calls[0][3] is False
    assert tab.status_var.value == "validation failed"
    assert inserted == [("Error", "Incomplete QBlade root.")]


def test_qblade_bridge_tab_sign_probe_parses_inputs_before_worker(tmp_path, monkeypatch):
    config = minimal_bridge_config(tmp_path)

    class Service:
        def validate_config(self, value, *, check_paths=False, check_qblade_install=False, check_qblade_model=False):
            return SimpleNamespace(errors=())

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.status_var = DummyStatus()
    inserted: list[tuple[str, str]] = []
    tab.result_tree = SimpleNamespace(
        get_children=lambda: [],
        delete=lambda _item: None,
        insert=lambda _parent, _index, text, values: inserted.append((text, values[0])),
    )
    tab.collect_config = lambda: config
    tab.sign_probe_force_var = DummyValue("not-a-number")
    tab.sign_probe_steps_var = DummyValue("20")
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr("vorlap.gui.tabs.qblade_bridge.messagebox.showerror", lambda title, message: errors.append((title, message)))

    QBladeBridgeTab.run_sign_probe(tab)

    assert started == []
    assert tab.status_var.value == "validation failed"
    assert inserted and inserted[0][0] == "Error"
    assert "could not convert string to float" in inserted[0][1]
    assert errors and errors[0][0] == "Validation failed"


def test_qblade_bridge_tab_sign_probe_applies_selected_scenario(tmp_path):
    config = minimal_bridge_config(tmp_path, force_scale=1.0, diagnostic_force_scale=True)
    validated: list[QBladeBridgeConfig] = []

    class Service:
        def validate_config(self, value, *, check_paths=False, check_qblade_install=False, check_qblade_model=False):
            validated.append(value)
            return SimpleNamespace(errors=())

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.collect_config = lambda: config
    tab.sign_probe_scenario_var = DummyValue("identity force scale 0")
    tab.sign_probe_node_var = DummyValue("")
    tab.sign_probe_axes_var = DummyValue("X Y")
    tab.sign_probe_force_var = DummyValue("500")
    tab.sign_probe_steps_var = DummyValue("20")
    tab.sign_probe_local_var = DummyValue(False)
    started: list[tuple[object, ...]] = []
    tab._start_worker = lambda *args: started.append(args)

    QBladeBridgeTab.run_sign_probe(tab)

    assert started
    assert len(validated) == 1
    assert validated[0].coupling_mode == "sil-vorlap"
    assert validated[0].force_scale == 0.0
    assert validated[0].diagnostic_force_scale is False


def test_qblade_bridge_tab_stop_action_marks_active_run_stopping():
    tab = object.__new__(QBladeBridgeTab)
    tab._worker_active = True
    tab._stop_event = threading.Event()
    tab.status_var = DummyStatus()
    tab.local_log = DummyLog()
    app_messages: list[str] = []
    tab.app = SimpleNamespace(log=app_messages.append)

    QBladeBridgeTab.stop_action(tab)

    assert tab._stop_event.is_set()
    assert tab.status_var.value == "stopping"
    assert tab.local_log.messages == [
        "Stop requested; the active bridge run will stop at the next timestep boundary.\n"
    ]
    assert app_messages == tab.local_log.messages


def test_qblade_bridge_tab_control_states_enable_stop_only_while_active():
    tab = object.__new__(QBladeBridgeTab)
    action_buttons = [DummyButton(), DummyButton()]
    stop_button = DummyButton()
    tab._action_buttons = action_buttons
    tab.stop_button = stop_button

    QBladeBridgeTab._set_controls_enabled(tab, False)

    assert [button.state for button in action_buttons] == ["disabled", "disabled"]
    assert stop_button.state == "normal"

    QBladeBridgeTab._set_controls_enabled(tab, True)

    assert [button.state for button in action_buttons] == ["normal", "normal"]
    assert stop_button.state == "disabled"


def test_qblade_bridge_tab_worker_runs_fake_progress_and_completion():
    tab, callbacks, inserted, action_buttons, stop_button, app_messages = make_headless_worker_tab()
    result = SimpleNamespace(
        state="complete",
        artifacts={
            "summary": "/tmp/out/case_run_summary.json",
            "result_file": "/tmp/out/case_results.txt",
            "run_log": "/tmp/out/case_qblade.log",
            "sil_vorlap_history": "/tmp/out/history.csv",
            "sil_vorlap_source_coverage": "/tmp/out/case_run_summary_source_coverage.csv",
            "working_sim": "/tmp/out/qblade_model_working/case.sim",
        },
        diagnostics={
            "completed_steps": 2,
            "requested_steps": 2,
            "coupling_mode": "sil-vorlap",
            "sil_vorlap_source_coverage_rows": 1,
            "qblade_model_override_count": 0,
            "stopped": False,
            "failed": False,
            "sil_vorlap_max_abs_force": 12.5,
            "sil_vorlap_message": "progress completed",
        },
    )

    def work():
        QBladeBridgeTab._threadsafe_log(tab, "progress 1/2")
        QBladeBridgeTab._threadsafe_log(tab, "progress 2/2")
        return result

    QBladeBridgeTab._start_worker(
        tab,
        "running",
        work,
        lambda value: QBladeBridgeTab._on_run_complete(tab, value),
    )
    assert tab._worker_active is True
    assert tab.status_var.value == "running"
    assert [button.state for button in action_buttons] == ["disabled", "disabled"]
    assert stop_button.state == "normal"

    tab._worker.join(timeout=2)
    assert not tab._worker.is_alive()
    assert len(callbacks) == 1
    for callback in list(callbacks):
        callback()

    assert tab._worker_active is False
    assert tab._last_run_result is result
    assert tab.status_var.history == ["running", "complete"]
    assert [button.state for button in action_buttons] == ["normal", "normal"]
    assert stop_button.state == "disabled"
    assert ("State", "complete") in inserted
    assert ("Completed steps", "2") in inserted
    assert tab.local_log.messages == [
        "progress 1/2\n",
        "progress 2/2\n",
        "QBlade bridge run complete.\n",
    ]
    assert app_messages == tab.local_log.messages


def test_qblade_bridge_tab_worker_failure_restores_controls_with_deferred_after():
    tab, callbacks, inserted, action_buttons, stop_button, app_messages = make_headless_worker_tab()

    def work():
        raise RuntimeError("runner exploded")

    QBladeBridgeTab._start_worker(
        tab,
        "running",
        work,
        lambda _value: None,
    )
    assert tab._worker_active is True
    assert tab.status_var.value == "running"

    tab._worker.join(timeout=2)
    assert not tab._worker.is_alive()
    assert len(callbacks) == 1
    callbacks[0]()

    assert tab._worker_active is False
    assert tab.status_var.history == ["running", "failed"]
    assert [button.state for button in action_buttons] == ["normal", "normal"]
    assert stop_button.state == "disabled"
    assert inserted == [("Error", "runner exploded")]
    assert tab.local_log.messages == ["QBlade bridge action failed: runner exploded\n"]
    assert app_messages == tab.local_log.messages


def test_qblade_bridge_tab_open_selected_artifact_uses_selected_result_value(tmp_path):
    artifact = tmp_path / "case_run_summary.json"
    artifact.write_text("{}\n", encoding="utf-8")
    opened: list[str] = []

    class Service:
        def open_artifact(self, value):
            opened.append(value)
            return SimpleNamespace(artifact=Path(value), command=("open", value)), "fake-process"

    tab = object.__new__(QBladeBridgeTab)
    tab.service = Service()
    tab.status_var = DummyStatus()
    tab.result_tree = SimpleNamespace(
        selection=lambda: ("row-1",),
        item=lambda _item: {"values": (str(artifact),)},
    )
    tab.local_log = DummyLog()
    app_messages: list[str] = []
    tab.app = SimpleNamespace(log=app_messages.append)

    QBladeBridgeTab.open_selected_artifact(tab)

    assert opened == [str(artifact)]
    assert tab.status_var.value == "artifact opened"
    assert tab.local_log.messages == [f"Opened artifact: {artifact}\n"]
    assert app_messages == tab.local_log.messages


def test_qblade_bridge_tab_open_selected_artifact_reports_missing_selection(monkeypatch):
    tab = object.__new__(QBladeBridgeTab)
    tab.status_var = DummyStatus()
    tab.result_tree = SimpleNamespace(selection=lambda: ())
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr("vorlap.gui.tabs.qblade_bridge.messagebox.showerror", lambda title, message: errors.append((title, message)))

    QBladeBridgeTab.open_selected_artifact(tab)

    assert tab.status_var.value == "open failed"
    assert errors == [("Open artifact failed", "Select an artifact row first.")]
