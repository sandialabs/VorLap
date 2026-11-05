#!/usr/bin/env python3
"""QBlade bridge workflow tab for the VorLap GUI."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
import math
import os
import shutil
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import vorlap
from vorlap.qblade_batch import reference_case_dir_for_root, reference_sim_name_for_root
from vorlap.qblade_bridge_config import QBladeBridgeConfig
from vorlap.qblade_bridge_service import (
    QBladeBridgeBuildResult,
    QBladeBridgePrepareResult,
    QBladeBridgeRunServiceResult,
    QBladeBridgeService,
    QBladeBridgeSignProbeResult,
    find_qblade_gui_executable,
)

from ..widgets import PathEntry, ScrollableFrame, ScrollText


BRIDGE_SCENARIOS = (
    "nominal VorLap",
    "baseline/no coupling",
    "identity force scale 0",
    "diagnostic amplitude sensitivity",
)
COARSE_TIMESTEP_WARNING_S = 0.1
BRIDGE_USER_ACTIONS = (
    "Validate",
    "Prepare case",
    "Build bridge",
    "Run SIL bridge",
    "Run QBlade-driven",
    "Open QBlade GUI",
    "Stop",
)
DEFAULT_QBLADE_DURATION_S = "60"
DEFAULT_QBLADE_TIMESTEP_S = "0.0075"
DEFAULT_QBLADE_STEPS = "8000"
DEFAULT_QBLADE_RPM = "1"
DEFAULT_QBLADE_STORE_FROM_S = "0"
DEFAULT_QBLADE_SAMPLE_STEP_S = DEFAULT_QBLADE_TIMESTEP_S
DEFAULT_QBLADE_CHECKPOINT_EVERY = "200"


def _display_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(_display_value(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={_display_value(item)}" for key, item in value.items())
    return str(value)


def _display_number(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _display_value(value)
    if math.isfinite(number):
        return f"{number:.6g}"
    if math.isinf(number):
        return "inf" if number > 0.0 else "-inf"
    return "nan"


def _display_range(row: dict[str, object], min_key: str, max_key: str) -> str:
    lo = _display_number(row.get(min_key))
    hi = _display_number(row.get(max_key))
    if lo and hi:
        return f"{lo}..{hi}"
    return lo or hi


def _display_list_preview(value: object, *, limit: int = 4) -> str:
    if not isinstance(value, (list, tuple)):
        return _display_value(value)
    shown = [_display_value(item) for item in value[:limit]]
    if len(value) > limit:
        shown.append(f"... ({len(value)} total)")
    return ", ".join(shown)


def _source_coverage_status_rows(diagnostics: dict[str, object]) -> list[tuple[str, str]]:
    coverage = diagnostics.get("sil_vorlap_source_coverage")
    if not isinstance(coverage, list):
        return []
    rows: list[tuple[str, str]] = []
    for index, raw in enumerate(coverage, start=1):
        if not isinstance(raw, dict):
            continue
        component = _display_value(raw.get("component") or f"component {index}")
        aoa = _display_range(raw, "aoa_min_deg", "aoa_max_deg")
        source_aoa = _display_range(raw, "source_aoa_min_deg", "source_aoa_max_deg")
        reynolds = _display_range(raw, "re_min", "re_max")
        source_reynolds = _display_range(raw, "source_re_min", "source_re_max")
        frequency = _display_range(raw, "freq_min_hz", "freq_max_hz")
        samples = _display_number(raw.get("samples"))
        node_samples = _display_number(raw.get("node_samples"))
        rows.extend(
            [
                (f"Source {component} samples", f"{samples} samples, {node_samples} node samples"),
                (f"Source {component} AOA [deg]", f"{aoa} (source {source_aoa})" if source_aoa else aoa),
                (f"Source {component} Re", f"{reynolds} (source {source_reynolds})" if source_reynolds else reynolds),
                (
                    f"Source {component} frequency [Hz]",
                    (
                        f"{frequency}; Nyquist {_display_number(raw.get('nyquist_hz'))}; "
                        f"ratio {_display_number(raw.get('nyquist_ratio'))}"
                    ),
                ),
                (
                    f"Source {component} load",
                    (
                        f"max force {_display_number(raw.get('max_nodal_force_n'))} N; "
                        f"C {_display_number(raw.get('force_coefficient_max_abs'))}/"
                        f"{_display_number(raw.get('source_force_coefficient_max_abs'))}; "
                        f"max Vstruct {_display_number(raw.get('max_structural_velocity_mps'))} m/s; "
                        f"power mean/max {_display_number(raw.get('force_power_mean_w'))}/"
                        f"{_display_number(raw.get('force_power_max_abs_w'))} W"
                    ),
                ),
                (
                    f"Source {component} counts",
                    (
                        f"Re out {_display_number(raw.get('re_out_of_range_count'))}; "
                        f"AOA out {_display_number(raw.get('aoa_out_of_range_count'))}; "
                        f"clips {_display_number(raw.get('source_clip_count'))}; "
                        f"alias {_display_number(raw.get('aliased_frequency_count'))}; "
                        f"coeff out {_display_number(raw.get('force_coefficient_out_of_range_count'))}; "
                        f"nonfinite {_display_number(raw.get('nonfinite_count'))}"
                    ),
                ),
            ]
        )
    return rows


def _velocity_status_rows(diagnostics: dict[str, object]) -> list[tuple[str, str]]:
    velocity = diagnostics.get("sil_vorlap_velocity_diagnostics")
    if not isinstance(velocity, dict):
        return []
    rows: list[tuple[str, str]] = []
    axis_stats = velocity.get("axis_stats")
    if isinstance(axis_stats, dict):
        for axis in sorted(axis_stats):
            stat = axis_stats.get(axis)
            if not isinstance(stat, dict):
                continue
            rows.append(
                (
                    f"Velocity {axis} [m/s]",
                    (
                        f"{_display_number(stat.get('min'))}..{_display_number(stat.get('max'))}; "
                        f"max abs {_display_number(stat.get('max_abs'))}; "
                        f"samples {_display_number(stat.get('samples'))}; "
                        f"all zero {_display_value(stat.get('all_zero'))}"
                    ),
                )
            )
    rows.append(("Velocity zero axes", _display_list_preview(velocity.get("all_zero_axes"))))
    rows.append(("Velocity zero channels", _display_list_preview(velocity.get("all_zero_channels"))))
    return rows


def _native_callback_status_rows(diagnostics: dict[str, object]) -> list[tuple[str, str]]:
    evidence = diagnostics.get("native_callback_evidence")
    if not isinstance(evidence, dict):
        return []
    parse_errors = _display_list_preview(evidence.get("python_env_parse_errors")) or "none"
    rows = [
        (
            "Native callback block",
            (
                f"present {_display_value(evidence.get('callback_block_present'))}; "
                f"turbine {_display_value(evidence.get('turbine_file_resolved'))}; "
                f"error {_display_value(evidence.get('callback_block_error'))}"
            ),
        ),
        (
            "Native callback target",
            (
                f"library {_display_value(evidence.get('library_stem'))}; "
                f"function {_display_value(evidence.get('function_name'))}; "
                f"array {_display_number(evidence.get('array_size'))}; "
                f"parameter exists {_display_value(evidence.get('parameter_file_exists'))}"
            ),
        ),
        (
            "Native library symbols",
            (
                f"checked {_display_value(evidence.get('library_symbol_check_available'))}; "
                f"exported {_display_list_preview(evidence.get('library_exported_required_symbols')) or 'none'}; "
                f"missing {_display_list_preview(evidence.get('library_missing_required_symbols')) or 'none'}"
            ),
        ),
        (
            "Native update_init",
            (
                f"seen {_display_value(evidence.get('update_init_seen'))}; "
                f"count {_display_number(evidence.get('update_init_count'))}"
            ),
        ),
        (
            "Native update calls",
            (
                f"log entries {_display_number(evidence.get('update_log_entry_count'))}; "
                f"observed max {_display_number(evidence.get('update_count_observed_max'))}"
            ),
        ),
        ("Native callback log", _display_value(evidence.get("log_file"))),
        (
            "Native Python env",
            (
                f"before import {_display_value(evidence.get('python_env_before_import_seen'))}; "
                f"after import {_display_value(evidence.get('python_env_after_import_seen'))}; "
                f"parse errors {parse_errors}"
            ),
        ),
        (
            "Native imports",
            (
                f"NumPy {_display_value(evidence.get('numpy_version'))} at {_display_value(evidence.get('numpy_file'))}; "
                f"VorLap {_display_value(evidence.get('vorlap_file'))}; "
                f"runtime {_display_value(evidence.get('vorlap_qblade_runtime_file'))}"
            ),
        ),
    ]
    return rows


def default_airfoil_dir() -> Path:
    """Return the repo default airfoil directory used by the bridge tab."""
    return Path(vorlap.repo_dir) / "data" / "airfoils"


def default_qblade_sim_path() -> Path | None:
    """Return the packaged reference QBlade model when available."""
    root = Path(vorlap.repo_dir)
    try:
        candidate = reference_case_dir_for_root(root) / reference_sim_name_for_root(root)
    except FileNotFoundError:
        return None
    return candidate if candidate.is_file() else None


def default_qblade_root() -> Path | None:
    """Return the repo-local QBlade CE root when available."""
    for name in ("QBladeCE_2.0.9.7", "QBladeCE_2.0.9.6"):
        candidate = Path(vorlap.repo_dir) / "run_headless" / name
        if candidate.is_dir():
            return candidate
    return None


def _python_candidate_exists(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if os.sep not in text and (os.altsep is None or os.altsep not in text):
        resolved = shutil.which(text)
        return resolved or None
    path = Path(text).expanduser()
    return str(path) if path.is_file() else None


def _is_current_frozen_app_executable(value: str | Path | None) -> bool:
    if not getattr(sys, "frozen", False) or value in (None, ""):
        return False
    try:
        return Path(value).expanduser().resolve(strict=False) == Path(sys.executable).resolve(strict=False)
    except OSError:
        return str(value) == sys.executable


def default_python_executable() -> str:
    """Return a Python interpreter suitable for child bridge work, not the GUI app."""
    for env_name in ("VORLAP_PYTHON", "PYTHON_HOST", "PYTHON_EXE"):
        candidate = _python_candidate_exists(os.environ.get(env_name))
        if candidate is not None and not _is_current_frozen_app_executable(candidate):
            return candidate

    if os.name == "nt":
        venv_candidates = (
            Path(vorlap.repo_dir) / ".venv" / "Scripts" / "python.exe",
            Path(vorlap.repo_dir) / ".venv" / "bin" / "python",
        )
    else:
        venv_candidates = (
            Path(vorlap.repo_dir) / ".venv" / "bin" / "python",
            Path(vorlap.repo_dir) / ".venv" / "Scripts" / "python.exe",
        )
    for candidate_path in venv_candidates:
        candidate = _python_candidate_exists(candidate_path)
        if candidate is not None:
            return candidate

    base_executable = getattr(sys, "_base_executable", None)
    candidate = _python_candidate_exists(base_executable)
    if candidate is not None and not _is_current_frozen_app_executable(candidate):
        return candidate

    if not getattr(sys, "frozen", False):
        candidate = _python_candidate_exists(sys.executable)
        if candidate is not None:
            return candidate

    for command in ("python3", "python"):
        candidate = _python_candidate_exists(command)
        if candidate is not None and not _is_current_frozen_app_executable(candidate):
            return candidate
    return sys.executable


def infer_model_airfoil_dir(sim_path: str | Path) -> Path | None:
    """Find a model-local VorLapAirfoils directory near a selected QBlade .sim file."""
    if sim_path in (None, ""):
        return None
    resolved_sim = Path(sim_path).expanduser().resolve()
    if not resolved_sim.is_file():
        return None
    model_dir = resolved_sim.parent
    candidates = [model_dir / "VorLapAirfoils"]
    candidates.extend(sorted(model_dir.glob("*/VorLapAirfoils")))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def prepare_status_rows(result: QBladeBridgePrepareResult) -> list[tuple[str, str]]:
    """Rows shown after a bridge case preparation or dry-run plan."""
    artifacts = result.artifacts
    diagnostics = result.diagnostics
    rows = [
        ("State", result.state),
        ("Runtime config", artifacts.get("runtime_config")),
        ("Geometry audit", artifacts.get("geometry_audit")),
        ("Working sim", artifacts.get("working_sim")),
        ("Working manifest", artifacts.get("working_copy_manifest")),
        ("Node count", diagnostics.get("node_count")),
        ("Swap size", diagnostics.get("swap_size")),
        ("Component count", diagnostics.get("component_count")),
        ("Node source", diagnostics.get("node_source")),
        ("Source policy", diagnostics.get("source_bounds_policy")),
        ("Geometry audit exists", diagnostics.get("geometry_audit_exists")),
        ("Geometry audit rows", diagnostics.get("geometry_audit_rows")),
        ("Geometry basis errors", diagnostics.get("geometry_audit_basis_error_count")),
        ("Geometry basis warnings", diagnostics.get("geometry_audit_basis_warning_count")),
        ("Geometry basis error nodes", diagnostics.get("geometry_audit_basis_error_nodes")),
        ("Geometry duplicate nodes", diagnostics.get("geometry_audit_duplicate_node_count")),
        ("Model overrides", diagnostics.get("qblade_model_override_count")),
        ("Working copy files", diagnostics.get("working_copy_file_count")),
        ("Dry run", diagnostics.get("dry_run")),
    ]
    return [(key, _display_value(value)) for key, value in rows]


def run_status_rows(result: QBladeBridgeRunServiceResult) -> list[tuple[str, str]]:
    """Rows shown after a one-case QBlade/VorLap run."""
    artifacts = result.artifacts
    diagnostics = result.diagnostics
    rows = [
        ("State", result.state),
        ("Summary", artifacts.get("summary")),
        ("Result file", artifacts.get("result_file")),
        ("Run log", artifacts.get("run_log")),
        ("Stored project", artifacts.get("stored_project")),
        ("QBlade log tail", diagnostics.get("qblade_log_tail_path")),
        ("QBlade log lines", diagnostics.get("qblade_log_tail_line_count")),
        ("SIL history", artifacts.get("sil_vorlap_history")),
        ("Source coverage", artifacts.get("sil_vorlap_source_coverage")),
        ("Quick plot directory", artifacts.get("bridge_quick_plot_dir")),
        ("Plot max force", artifacts.get("bridge_quick_plot_max_force")),
        ("Plot AOA/Re limits", artifacts.get("bridge_quick_plot_aoa_re")),
        ("Plot Nyquist margin", artifacts.get("bridge_quick_plot_nyquist")),
        ("Plot force history", artifacts.get("bridge_quick_plot_force_history")),
        ("Plot QBlade response", artifacts.get("bridge_quick_plot_qblade_response")),
        ("Quick plot count", diagnostics.get("bridge_quick_plot_count")),
        ("Quick plot errors", diagnostics.get("bridge_quick_plot_errors")),
        ("Force history column", diagnostics.get("bridge_quick_plot_force_history_column")),
        ("QBlade response channel", diagnostics.get("bridge_quick_plot_qblade_response_channel")),
        ("Working sim", artifacts.get("working_sim")),
        ("Working manifest", artifacts.get("working_copy_manifest")),
        ("Completed steps", diagnostics.get("completed_steps")),
        ("Requested steps", diagnostics.get("requested_steps")),
        ("Coupling", diagnostics.get("coupling_mode")),
        ("QBlade execution", diagnostics.get("qblade_execution_mode")),
        ("Source coverage rows", diagnostics.get("sil_vorlap_source_coverage_rows")),
        ("Source policy", diagnostics.get("source_bounds_policy")),
        (
            "Geometry reported positions",
            (
                f"available {_display_number(diagnostics.get('geometry_audit_reported_available_count'))}; "
                f"missing {_display_number(diagnostics.get('geometry_audit_reported_missing_count'))}; "
                f"direct errors {_display_number(diagnostics.get('geometry_audit_reported_error_count'))}"
            ),
        ),
        ("Geometry direct max delta [m]", diagnostics.get("geometry_audit_reported_max_delta_m")),
        (
            "Geometry fitted frame",
            (
                f"status {_display_value(diagnostics.get('geometry_audit_reported_frame_fit_status'))}; "
                f"samples {_display_number(diagnostics.get('geometry_audit_reported_frame_fit_sample_count'))}; "
                f"max residual {_display_number(diagnostics.get('geometry_audit_reported_frame_fit_max_residual_m'))} m; "
                f"mean residual {_display_number(diagnostics.get('geometry_audit_reported_frame_fit_mean_residual_m'))} m"
            ),
        ),
        ("Geometry fitted translation [m]", diagnostics.get("geometry_audit_reported_frame_fit_translation_m")),
        ("Model overrides", diagnostics.get("qblade_model_override_count")),
        ("Working copy files", diagnostics.get("working_copy_file_count")),
        ("Stopped", diagnostics.get("stopped")),
        ("Failed", diagnostics.get("failed")),
        ("Max force [N]", diagnostics.get("sil_vorlap_max_abs_force")),
        ("Max structural velocity [m/s]", diagnostics.get("sil_vorlap_max_structural_velocity_mps")),
        ("Cache hits/misses", diagnostics.get("sil_vorlap_cache")),
        ("Velocity warnings", diagnostics.get("sil_vorlap_velocity_warning_count")),
        ("Latest checkpoint", diagnostics.get("checkpoint_latest")),
        ("Latest checkpoint step", diagnostics.get("checkpoint_latest_completed_steps")),
        ("Checkpoint project", diagnostics.get("checkpoint_latest_project")),
        ("Resume requested", diagnostics.get("resume_requested")),
        ("Resume start step", diagnostics.get("resume_start_step")),
        ("Runtime message", diagnostics.get("sil_vorlap_message")),
    ]
    rows.extend(_velocity_status_rows(diagnostics))
    rows.extend(_source_coverage_status_rows(diagnostics))
    rows.extend(_native_callback_status_rows(diagnostics))
    return [(key, _display_value(value)) for key, value in rows]


def build_status_rows(result: QBladeBridgeBuildResult) -> list[tuple[str, str]]:
    """Rows shown after a QBlade bridge library build."""
    artifacts = result.artifacts
    diagnostics = result.diagnostics
    rows = [
        ("State", result.state),
        ("ControllerFiles", artifacts.get("controller_files")),
        ("Build directory", artifacts.get("build_dir")),
        ("Bridge library", artifacts.get("bridge_library")),
        ("Python", diagnostics.get("python")),
        ("Python executable", diagnostics.get("python_executable")),
        ("Python version", diagnostics.get("python_version")),
        ("NumPy version", diagnostics.get("numpy_version")),
        ("NumPy error", diagnostics.get("numpy_error")),
        ("Python probe error", diagnostics.get("python_probe_error")),
        ("CMake version", diagnostics.get("cmake_version")),
        ("CMake error", diagnostics.get("cmake_error")),
        ("Compiler", diagnostics.get("compiler")),
        ("Compiler error", diagnostics.get("compiler_error")),
        ("QBlade loader variable", diagnostics.get("qblade_loader_path_key")),
        ("QBlade loader path", diagnostics.get("qblade_loader_path")),
        ("QBlade library path", diagnostics.get("qblade_ld_library_path")),
        ("Force rebuild", diagnostics.get("force")),
        ("Library exists", diagnostics.get("bridge_library_exists")),
        ("Library loadable", diagnostics.get("bridge_library_loadable")),
        ("Library load error", diagnostics.get("bridge_library_load_error")),
    ]
    return [(key, _display_value(value)) for key, value in rows]


def sign_probe_status_rows(result: QBladeBridgeSignProbeResult) -> list[tuple[str, str]]:
    """Rows shown after a force sign-probe diagnostic."""
    artifacts = result.artifacts
    diagnostics = result.diagnostics
    rows = [
        ("State", result.state),
        ("Plan", artifacts.get("sign_probe_plan")),
        ("Report CSV", artifacts.get("sign_probe_report_csv")),
        ("Report JSON", artifacts.get("sign_probe_report_json")),
        ("Output root", artifacts.get("sign_probe_output_root")),
        ("Node", diagnostics.get("node")),
        ("Axes", diagnostics.get("axes")),
        ("Force [N]", diagnostics.get("force_n")),
        ("Local", diagnostics.get("local")),
        ("Probes completed", diagnostics.get("probes_completed")),
        ("Failures", diagnostics.get("failures")),
        ("Passed", diagnostics.get("passed")),
        ("Geometry audit exists", diagnostics.get("geometry_audit_exists")),
        ("Geometry audit rows", diagnostics.get("geometry_audit_rows")),
        ("Geometry basis errors", diagnostics.get("geometry_audit_basis_error_count")),
        ("Geometry basis warnings", diagnostics.get("geometry_audit_basis_warning_count")),
        ("Geometry basis error nodes", diagnostics.get("geometry_audit_basis_error_nodes")),
        ("Geometry duplicate nodes", diagnostics.get("geometry_audit_duplicate_node_count")),
    ]
    rows.extend(_sign_probe_report_status_rows(diagnostics))
    return [(key, _display_value(value)) for key, value in rows]


def _sign_probe_report_status_rows(diagnostics: dict[str, object]) -> list[tuple[str, str]]:
    report_rows = diagnostics.get("sign_probe_report_rows")
    if not isinstance(report_rows, list):
        return []
    rows: list[tuple[str, str]] = []
    for index, raw in enumerate(report_rows, start=1):
        if not isinstance(raw, dict):
            continue
        node = _display_value(raw.get("node")) or "node"
        axis = _display_value(raw.get("axis")) or "axis"
        sign = _display_value(raw.get("sign")) or str(index)
        channel = _display_value(raw.get("observed_position_channel")) or "missing position channel"
        issue = _display_value(raw.get("issue")) or "none"
        value = (
            f"passed {_display_value(raw.get('passed'))}; "
            f"pair {_display_value(raw.get('paired_passed'))}; "
            f"delta {_display_number(raw.get('delta_position_m'))} m; "
            f"pair delta {_display_number(raw.get('paired_response_delta_m'))} m; "
            f"status {_display_value(raw.get('observation_status'))}; "
            f"channel {channel}; "
            f"issue {issue}"
        )
        rows.append((f"Probe {node} {axis} {sign}", value))
    return rows


def bridge_scenario_config(config: QBladeBridgeConfig, scenario: str) -> QBladeBridgeConfig:
    """Apply the selected GUI run scenario to a typed bridge config."""
    if scenario == "nominal VorLap":
        return replace(config, coupling_mode="sil-vorlap")
    if scenario == "baseline/no coupling":
        return replace(config, coupling_mode="none", force_scale=0.0, diagnostic_force_scale=False)
    if scenario == "identity force scale 0":
        return replace(config, coupling_mode="sil-vorlap", force_scale=0.0, diagnostic_force_scale=False)
    if scenario == "diagnostic amplitude sensitivity":
        return replace(config, coupling_mode="sil-vorlap", diagnostic_force_scale=True)
    raise ValueError(f"Unsupported QBlade bridge scenario: {scenario}")


def bridge_config_warnings(config: QBladeBridgeConfig) -> tuple[str, ...]:
    """Return non-blocking GUI warnings for bridge run settings."""
    warnings: list[str] = []
    if config.timestep_s is not None and float(config.timestep_s) > COARSE_TIMESTEP_WARNING_S:
        warnings.append(
            f"QBlade timestep {float(config.timestep_s):.6g} s exceeds the "
            f"{COARSE_TIMESTEP_WARNING_S:.6g} s bridge smoke-test range; verify stability before using results."
        )
    return tuple(warnings)


def viv_parameter_status_rows(config: QBladeBridgeConfig) -> list[tuple[str, str]]:
    """Rows describing manually overridden physical parameters."""
    fields = [
        ("Fluid density", config.fluid_density, "kg/m^3"),
        ("Dynamic viscosity", config.fluid_dynamicviscosity, "Pa s"),
        ("Rotation axis", config.rotation_axis, ""),
        ("Axis offset", config.rotation_axis_offset, "m"),
    ]
    rows: list[tuple[str, str]] = []
    for label, value, unit in fields:
        if value is None:
            continue
        suffix = f" {unit}" if unit else ""
        rows.append((label, f"manual override: {_display_value(value)}{suffix}"))
    return rows


class QBladeBridgeTab(ttk.Frame):
    """GUI controls for the QBlade/VorLap bridge workflow."""

    def __init__(self, master, app, service: QBladeBridgeService | None = None):
        super().__init__(master)
        self.app = app
        self.service = service or QBladeBridgeService()
        self._worker: threading.Thread | None = None
        self._worker_active = False
        self._worker_poll_id = None
        self._stop_event = threading.Event()
        self._state_listener = None
        self._state_qblade_sim_path = ""
        self._last_run_result: QBladeBridgeRunServiceResult | None = None
        self._build_variables()
        self._build()
        self._subscribe_app_state()
        self._apply_defaults()

    def destroy(self):
        self._stop_event.set()
        if self._worker_poll_id is not None:
            self.after_cancel(self._worker_poll_id)
            self._worker_poll_id = None
        state = getattr(self.app, "state", None)
        if self._state_listener is not None and hasattr(state, "unsubscribe"):
            state.unsubscribe(self._state_listener)
            self._state_listener = None
        super().destroy()

    def _subscribe_app_state(self) -> None:
        state = getattr(self.app, "state", None)
        if hasattr(state, "subscribe"):
            self._state_listener = state.subscribe(self._on_app_state_changed)

    def _on_app_state_changed(self, field_name: str, value: object) -> None:
        if field_name == "qblade_sim_path":
            self._seed_qblade_sim_from_state(str(value or ""))

    def _build_variables(self) -> None:
        self.qblade_root_var = tk.StringVar()
        self.qblade_binary_var = tk.StringVar()
        self.python_var = tk.StringVar(value=default_python_executable())
        self.airfoil_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.scenario_var = tk.StringVar(value=BRIDGE_SCENARIOS[0])
        self.node_source_var = tk.StringVar(value="structural")
        self.include_struts_var = tk.BooleanVar(value=True)
        self.include_tower_var = tk.BooleanVar(value=False)
        self.include_tower_nodes_var = tk.BooleanVar(value=False)
        self.diagnostic_force_scale_var = tk.BooleanVar(value=False)
        self.include_mean_load_var = tk.BooleanVar(value=False)
        self.source_bounds_policy_var = tk.StringVar(value="raise")
        self.debug_var = tk.BooleanVar(value=False)
        self.record_history_var = tk.BooleanVar(value=False)
        self.use_opencl_var = tk.BooleanVar(value=False)
        self.allow_partial_var = tk.BooleanVar(value=False)
        self.store_project_var = tk.BooleanVar(value=True)
        self.patch_original_model_var = tk.BooleanVar(value=False)
        self.force_scale_var = tk.StringVar(value="1.0")
        self.n_freq_depth_var = tk.StringVar(value="20")
        self.sample_step_var = tk.StringVar(value=DEFAULT_QBLADE_SAMPLE_STEP_S)
        self.duration_var = tk.StringVar(value=DEFAULT_QBLADE_DURATION_S)
        self.timestep_var = tk.StringVar(value=DEFAULT_QBLADE_TIMESTEP_S)
        self.steps_var = tk.StringVar(value=DEFAULT_QBLADE_STEPS)
        self.rpm_var = tk.StringVar(value=DEFAULT_QBLADE_RPM)
        self.store_from_var = tk.StringVar(value=DEFAULT_QBLADE_STORE_FROM_S)
        self.store_replay_var = tk.StringVar(value="true")
        self.sign_probe_node_var = tk.StringVar(value="")
        self.sign_probe_axes_var = tk.StringVar(value="X Y Z")
        self.sign_probe_force_var = tk.StringVar(value="1000")
        self.sign_probe_steps_var = tk.StringVar(value="200")
        self.sign_probe_local_var = tk.BooleanVar(value=False)
        self.sign_probe_scenario_var = tk.StringVar(value=BRIDGE_SCENARIOS[0])
        self.threads_var = tk.StringVar(value="1")
        self.device_var = tk.StringVar(value="1")
        self.group_size_var = tk.StringVar(value="32")
        self.checkpoint_every_var = tk.StringVar(value=DEFAULT_QBLADE_CHECKPOINT_EVERY)
        self.checkpoint_dir_var = tk.StringVar()
        self.resume_checkpoint_var = tk.StringVar()
        self.resume_from_checkpoint_var = tk.BooleanVar(value=False)
        self.resume_initialize_var = tk.BooleanVar(value=False)
        self.progress_every_var = tk.StringVar(value="100")
        self.status_var = tk.StringVar(value="idle")

    def _build(self) -> None:
        self.scrollable = ScrollableFrame(self)
        self.scrollable.pack(fill="both", expand=True)
        self.content = self.scrollable.content
        self.content.columnconfigure(0, weight=1)

        paths = ttk.LabelFrame(self.content, text="QBlade Model And Runtime")
        paths.grid(row=0, column=0, sticky="ew", pady=(10, 6))
        paths.columnconfigure(1, weight=1)
        self.qblade_sim_path = self._path_row(
            paths,
            0,
            "QBlade simulation (.sim)",
            "file",
            "Select QBlade simulation",
            on_select=self._on_qblade_sim_selected,
        )
        self.qblade_root_path = self._path_row(
            paths,
            1,
            "QBlade root",
            "dir",
            "Select QBlade root",
            textvariable=self.qblade_root_var,
        )
        self.qblade_binary_path = self._path_row(
            paths,
            2,
            "QBlade executable",
            "file",
            "Select QBlade executable",
            textvariable=self.qblade_binary_var,
        )
        self.python_path = self._path_row(
            paths,
            3,
            "Python executable",
            "file",
            "Select Python executable",
            textvariable=self.python_var,
        )
        self.airfoil_dir_path = self._path_row(
            paths,
            4,
            "VorLap airfoils",
            "dir",
            "Select airfoil directory",
            textvariable=self.airfoil_dir_var,
        )
        self.output_dir_path = self._path_row(
            paths,
            5,
            "Output directory",
            "dir",
            "Select output directory",
            textvariable=self.output_dir_var,
        )

        prep = ttk.LabelFrame(self.content, text="Bridge Preparation")
        prep.grid(row=1, column=0, sticky="ew", pady=(10, 6))
        for col in range(6):
            prep.columnconfigure(col, weight=1 if col in (1, 3, 5) else 0)
        self._combo_row(prep, 0, "Node source", self.node_source_var, ("structural", "converted"))
        self._combo_row(prep, 0, "Source policy", self.source_bounds_policy_var, ("warn", "raise", "clip"), column=2)
        ttk.Checkbutton(prep, text="Patch original", variable=self.patch_original_model_var).grid(row=0, column=4, sticky="w")
        ttk.Checkbutton(prep, text="Struts", variable=self.include_struts_var).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(prep, text="Tower", variable=self.include_tower_var).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(prep, text="Tower nodes", variable=self.include_tower_nodes_var).grid(row=1, column=2, sticky="w")
        ttk.Checkbutton(prep, text="Diagnostic scale", variable=self.diagnostic_force_scale_var).grid(row=1, column=3, sticky="w")
        ttk.Checkbutton(prep, text="Mean load", variable=self.include_mean_load_var).grid(row=1, column=4, sticky="w")
        ttk.Checkbutton(prep, text="Debug", variable=self.debug_var).grid(row=1, column=5, sticky="w")
        self._entry_row(prep, 2, "Force scale [-]", self.force_scale_var, column=0)
        self._entry_row(prep, 2, "Frequency depth [count]", self.n_freq_depth_var, column=2)
        self._entry_row(prep, 2, "Sample step [s]", self.sample_step_var, column=4)

        run = ttk.LabelFrame(self.content, text="Run Controls")
        run.grid(row=2, column=0, sticky="ew", pady=(10, 6))
        for col in range(6):
            run.columnconfigure(col, weight=1 if col in (1, 3, 5) else 0)
        self._entry_row(run, 0, "Duration [s]", self.duration_var, column=0)
        self._entry_row(run, 0, "Timestep [s]", self.timestep_var, column=2)
        self._entry_row(run, 0, "Steps [count]", self.steps_var, column=4)
        self._entry_row(run, 1, "Rotor speed [rpm]", self.rpm_var, column=0)
        self._entry_row(run, 1, "Store from [s]", self.store_from_var, column=2)
        self._combo_row(run, 1, "Store replay", self.store_replay_var, ("true", "false"), column=4)
        self._entry_row(run, 2, "Threads [count]", self.threads_var, column=0)
        self._entry_row(run, 2, "OpenCL device", self.device_var, column=2)
        self._entry_row(run, 2, "Group size [count]", self.group_size_var, column=4)
        self._entry_row(run, 3, "Progress every [steps]", self.progress_every_var, column=0)
        ttk.Checkbutton(run, text="OpenCL", variable=self.use_opencl_var).grid(row=3, column=2, sticky="w")
        ttk.Checkbutton(run, text="Partial export", variable=self.allow_partial_var).grid(row=3, column=3, sticky="w")
        ttk.Checkbutton(run, text="SIL history", variable=self.record_history_var).grid(row=3, column=4, sticky="w")
        ttk.Checkbutton(run, text="Store project", variable=self.store_project_var).grid(row=3, column=5, sticky="w")
        self._entry_row(run, 4, "Checkpoint every [steps]", self.checkpoint_every_var, column=0)
        ttk.Checkbutton(run, text="Resume latest", variable=self.resume_from_checkpoint_var).grid(row=4, column=2, sticky="w")
        ttk.Checkbutton(run, text="Resume init", variable=self.resume_initialize_var).grid(row=4, column=3, sticky="w")
        ttk.Label(run, text="Checkpoint dir").grid(row=5, column=0, sticky="w", padx=(0, 6), pady=2)
        self.checkpoint_dir_path = PathEntry(
            run,
            kind="dir",
            title="Select checkpoint directory",
            textvariable=self.checkpoint_dir_var,
        )
        self.checkpoint_dir_path.grid(row=5, column=1, columnspan=5, sticky="ew", pady=2)
        ttk.Label(run, text="Resume checkpoint").grid(row=6, column=0, sticky="w", padx=(0, 6), pady=2)
        self.resume_checkpoint_path = PathEntry(
            run,
            kind="file",
            title="Select resume checkpoint JSON",
            textvariable=self.resume_checkpoint_var,
        )
        self.resume_checkpoint_path.grid(row=6, column=1, columnspan=5, sticky="ew", pady=2)

        actions = ttk.Frame(self.content)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 6))
        action_specs = (
            ("validate_button", self.validate_inputs),
            ("prepare_button", self.prepare_case),
            ("build_button", self.build_bridge),
            ("run_button", self.run_sil_bridge),
            ("qblade_driven_button", self.run_qblade_driven_bridge),
            ("open_button", self.open_qblade_gui),
            ("stop_button", self.stop_action),
        )
        for col, (label, (attribute, command)) in enumerate(zip(BRIDGE_USER_ACTIONS, action_specs)):
            state = "disabled" if attribute == "stop_button" else "normal"
            button = ttk.Button(actions, text=label, command=command, state=state)
            setattr(self, attribute, button)
            button.grid(row=0, column=col, padx=(0, 6), sticky="w")
        status_col = len(action_specs)
        ttk.Label(actions, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=status_col, sticky="e")
        actions.columnconfigure(status_col, weight=1)
        self._action_buttons = [
            self.validate_button,
            self.prepare_button,
            self.build_button,
            self.run_button,
            self.qblade_driven_button,
            self.open_button,
        ]

        results = ttk.LabelFrame(self.content, text="Diagnostics And Artifacts")
        results.grid(row=4, column=0, sticky="nsew", pady=(10, 6))
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        self.result_tree = ttk.Treeview(results, columns=("value",), show="tree headings", height=10)
        self.result_tree.heading("#0", text="Item")
        self.result_tree.heading("value", text="Value")
        self.result_tree.column("#0", width=190, anchor="w")
        self.result_tree.column("value", width=840, anchor="w")
        result_scroll = ttk.Scrollbar(results, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=result_scroll.set)
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        result_scroll.grid(row=0, column=1, sticky="ns")
        self.open_artifact_button = ttk.Button(results, text="Open selected artifact", command=self.open_selected_artifact)
        self.open_artifact_button.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._action_buttons.append(self.open_artifact_button)

        log_frame = ttk.LabelFrame(self.content, text="Bridge Log")
        log_frame.grid(row=5, column=0, sticky="nsew", pady=(10, 6))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.local_log = ScrollText(log_frame, height=6)
        self.local_log.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    def _path_row(self, parent, row, label, kind, title, textvariable=None, on_select=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        entry = PathEntry(parent, kind=kind, title=title, textvariable=textvariable, on_select=on_select)
        entry.grid(row=row, column=1, columnspan=5, sticky="ew", pady=2)
        return entry

    def _entry_row(self, parent, row, label, variable, *, column):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(parent, textvariable=variable, width=12).grid(row=row, column=column + 1, sticky="ew", padx=(0, 10), pady=2)

    def _combo_row(self, parent, row, label, variable, values, *, column=0):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 6), pady=2)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=18)
        combo.grid(row=row, column=column + 1, sticky="ew", padx=(0, 10), pady=2)
        return combo

    def _apply_defaults(self) -> None:
        output_dir = Path(vorlap.repo_dir) / "run_headless" / "runs" / "gui_qblade_bridge"
        self.airfoil_dir_var.set(str(default_airfoil_dir()))
        self.output_dir_var.set(str(output_dir))
        self.checkpoint_dir_var.set(str(output_dir / "checkpoints"))
        self.resume_checkpoint_var.set("")
        self.store_project_var.set(True)
        self.store_replay_var.set("true")
        self.checkpoint_every_var.set(DEFAULT_QBLADE_CHECKPOINT_EVERY)

        default_sim = default_qblade_sim_path()
        if default_sim is not None:
            self._state_qblade_sim_path = str(default_sim)
            self.qblade_sim_path.set(str(default_sim))
            self._apply_model_airfoil_inference(default_sim, only_if_default=True)

        qblade_root = os.environ.get("QBLADE_ROOT")
        if qblade_root:
            self.qblade_root_var.set(qblade_root)
        else:
            default_root = default_qblade_root()
            if default_root is not None:
                self.qblade_root_var.set(str(default_root))

        qblade_binary = os.environ.get("QBLADE_BINARY") or os.environ.get("QBLADE_EXECUTABLE")
        if qblade_binary:
            self.qblade_binary_var.set(qblade_binary)
        else:
            root_text = self.qblade_root_var.get().strip()
            if root_text:
                executable = find_qblade_gui_executable(root_text)
                if executable is not None:
                    self.qblade_binary_var.set(str(executable))

        state = getattr(self.app, "state", None)
        self._seed_qblade_sim_from_state(getattr(state, "qblade_sim_path", ""))

    def _seed_qblade_sim_from_state(self, sim_path: str | Path | None) -> None:
        sim_path_text = str(sim_path or "").strip()
        if not sim_path_text:
            return
        current = self.qblade_sim_path.get().strip()
        if current and current != self._state_qblade_sim_path:
            return
        self._state_qblade_sim_path = sim_path_text
        self.qblade_sim_path.set(sim_path_text)
        self._apply_model_airfoil_inference(sim_path_text, only_if_default=True)

    def _on_qblade_sim_selected(self, sim_path: str) -> None:
        if self._apply_model_airfoil_inference(sim_path):
            self._log(f"Using model-local VorLap airfoils: {self.airfoil_dir_path.get()}\n")

    def _apply_model_airfoil_inference(self, sim_path: str | Path, *, only_if_default: bool = False) -> bool:
        inferred = infer_model_airfoil_dir(sim_path)
        if inferred is None:
            return False
        if only_if_default:
            current = str(self.airfoil_dir_path.get()).strip()
            if current:
                try:
                    if Path(current).expanduser().resolve() != default_airfoil_dir().resolve():
                        return False
                except OSError:
                    return False
        self.airfoil_dir_path.set(str(inferred))
        return True

    def on_tab_selected(self):
        try:
            self.scrollable.refresh()
        except Exception:
            pass

    def collect_config(self) -> QBladeBridgeConfig:
        python_executable = self.python_path.get()
        if _is_current_frozen_app_executable(python_executable):
            python_executable = default_python_executable()
            self.python_path.set(python_executable)
        state = {
            "sim_path": self.qblade_sim_path.get(),
            "qblade_root": self.qblade_root_path.get(),
            "qblade_binary": self.qblade_binary_path.get(),
            "python_executable": python_executable,
            "airfoil_dir": self.airfoil_dir_path.get(),
            "output_dir": self.output_dir_path.get(),
            "coupling_mode": "sil-vorlap",
            "node_source": self.node_source_var.get(),
            "include_struts": self.include_struts_var.get(),
            "include_tower": self.include_tower_var.get(),
            "include_tower_nodes": self.include_tower_nodes_var.get(),
            "diagnostic_force_scale": self.diagnostic_force_scale_var.get(),
            "include_mean_load": self.include_mean_load_var.get(),
            "source_bounds_policy": self.source_bounds_policy_var.get(),
            "debug": self.debug_var.get(),
            "record_sil_vorlap_history": self.record_history_var.get(),
            "force_scale": self.force_scale_var.get(),
            "n_freq_depth": self.n_freq_depth_var.get(),
            "sample_step": self.sample_step_var.get(),
            "duration_s": self.duration_var.get(),
            "timestep_s": self.timestep_var.get(),
            "steps": self.steps_var.get(),
            "rpm": self.rpm_var.get(),
            "store_from_s": self.store_from_var.get(),
            "store_replay": self.store_replay_var.get(),
            "threads": self.threads_var.get(),
            "use_opencl": self.use_opencl_var.get(),
            "opencl_device": self.device_var.get(),
            "opencl_group_size": self.group_size_var.get(),
            "allow_partial": self.allow_partial_var.get(),
            "store_project": self.store_project_var.get(),
            "checkpoint_every": self.checkpoint_every_var.get(),
            "checkpoint_dir": self.checkpoint_dir_path.get(),
            "resume_from_checkpoint": self.resume_from_checkpoint_var.get(),
            "resume_checkpoint": self.resume_checkpoint_path.get(),
            "resume_initialize": self.resume_initialize_var.get(),
            "progress_every": self.progress_every_var.get(),
            "patch_original_model": self.patch_original_model_var.get(),
        }
        return QBladeBridgeConfig.from_gui_state(state)

    def validate_inputs(self) -> None:
        try:
            config = self.collect_config()
            result = self.service.validate_config(
                config,
                check_paths=True,
                check_qblade_install=True,
                check_qblade_model=True,
            )
        except Exception as exc:
            self._show_validation_error(str(exc))
            return
        if result.ok:
            rows = [("State", "valid"), ("QBlade GUI", result.qblade_gui_executable)]
            rows.extend(("Warning", warning) for warning in bridge_config_warnings(config))
            rows.extend(viv_parameter_status_rows(config))
            self._set_status("valid")
            self._set_rows([(key, _display_value(value)) for key, value in rows])
            self._log("Bridge inputs validated.\n")
        else:
            self._set_status("validation failed")
            self._set_rows([("Error", error) for error in result.errors])

    def prepare_case(self) -> None:
        config = self._validated_config(
            check_paths=True,
            check_qblade_model=True,
            transform=lambda value: replace(value, coupling_mode="sil-vorlap"),
        )
        if config is None:
            return
        self._start_worker(
            "preparing",
            lambda: self.service.prepare_case(config, check_paths=True),
            self._on_prepare_complete,
        )

    def build_bridge(self) -> None:
        config = self._validated_config(check_qblade_install=True)
        if config is None:
            return
        self._start_worker(
            "building",
            lambda: self.service.build_bridge(config),
            self._on_build_complete,
        )

    def run_sign_probe(self) -> None:
        config = self._validated_config(
            check_paths=True,
            check_qblade_model=True,
            transform=lambda value: bridge_scenario_config(value, self._selected_sign_probe_scenario()),
        )
        if config is None:
            return
        try:
            force_n = float(self.sign_probe_force_var.get())
            steps = int(self.sign_probe_steps_var.get())
        except Exception as exc:
            self._show_validation_error(str(exc))
            return

        def work():
            return self.service.run_sign_probe(
                config,
                node_id=self.sign_probe_node_var.get() or None,
                axes=self.sign_probe_axes_var.get(),
                force_n=force_n,
                local=self.sign_probe_local_var.get(),
                steps=steps,
                message_callback=self._threadsafe_log,
                stop_requested=self._stop_event.is_set,
                check_paths=True,
            )

        self._start_worker("running sign probe", work, self._on_sign_probe_complete)

    def run_sil_bridge(self) -> None:
        config = self._validated_config(
            check_paths=True,
            check_qblade_model=True,
            transform=lambda value: replace(value, coupling_mode="sil-vorlap"),
        )
        if config is None:
            return
        self._last_run_result = None

        def work():
            return self.service.run_case_isolated(
                config,
                message_callback=self._threadsafe_log,
                stop_requested=self._stop_event.is_set,
                check_paths=True,
            )

        self._start_worker("running", work, self._on_run_complete)

    def _selected_scenario(self) -> str:
        scenario_var = getattr(self, "scenario_var", None)
        if scenario_var is None:
            return BRIDGE_SCENARIOS[0]
        return str(scenario_var.get())

    def _selected_sign_probe_scenario(self) -> str:
        scenario_var = getattr(self, "sign_probe_scenario_var", None)
        if scenario_var is None:
            return BRIDGE_SCENARIOS[0]
        return str(scenario_var.get())

    def run_qblade_driven_bridge(self) -> None:
        config = self._validated_config(
            check_paths=True,
            check_qblade_model=True,
            transform=lambda value: replace(
                value,
                coupling_mode="qblade-external",
                qblade_execution_mode="run-full-simulation",
                checkpoint_every=0,
                checkpoint_dir=None,
                resume_from_checkpoint=False,
                resume_checkpoint=None,
                resume_initialize=False,
                allow_partial=False,
            ),
        )
        if config is None:
            return
        self._last_run_result = None

        def work():
            return self.service.run_case(
                config,
                message_callback=self._threadsafe_log,
                stop_requested=self._stop_event.is_set,
                check_paths=True,
            )

        self._start_worker("running QBlade-driven", work, self._on_run_complete)

    def run_native_callback_proof(self) -> None:
        self.run_qblade_driven_bridge()

    def _validated_config(
        self,
        *,
        check_paths: bool = False,
        check_qblade_install: bool = False,
        check_qblade_model: bool = False,
        transform=None,
    ) -> QBladeBridgeConfig | None:
        try:
            config = self.collect_config()
            if transform is not None:
                config = transform(config)
            result = self.service.validate_config(
                config,
                check_paths=check_paths,
                check_qblade_install=check_qblade_install,
                check_qblade_model=check_qblade_model,
            )
        except Exception as exc:
            self._show_validation_error(str(exc))
            return None
        if result.errors:
            self._set_status("validation failed")
            self._set_rows([("Error", error) for error in result.errors])
            return None
        return config

    def _show_validation_error(self, message: str) -> None:
        self._set_status("validation failed")
        self._set_rows([("Error", message)])
        messagebox.showerror("Validation failed", message)

    def stop_action(self) -> None:
        if not self._worker_active:
            self._set_status("idle")
            return
        if not self._stop_event.is_set():
            self._stop_event.set()
            self._set_status("stopping")
            self._log("Stop requested; the active bridge run will stop at the next timestep boundary.\n")

    def open_qblade_gui(self) -> None:
        try:
            plan, _process = self.service.launch_qblade_gui(
                self.collect_config(),
                run_result=getattr(self, "_last_run_result", None),
            )
        except Exception as exc:
            self._set_status("launch failed")
            self._set_rows([("Error", str(exc))])
            messagebox.showerror("QBlade launch failed", str(exc))
            return
        self._set_status("qblade launched")
        self._set_rows(
            [
                ("Executable", str(plan.executable)),
                ("Command", " ".join(plan.command)),
                ("Working directory", str(plan.cwd)),
                ("Case path", str(plan.case_path_hint)),
                ("Opened project", "" if plan.project_path is None else str(plan.project_path)),
                ("Project source", "" if plan.project_source is None else str(plan.project_source)),
                ("Run visibility", "QBlade GUI inspection is separate from the automated VorLap run."),
            ]
        )
        self._log(f"Launched QBlade GUI: {' '.join(plan.command)}\n")

    def open_selected_artifact(self) -> None:
        try:
            value = self._selected_result_value()
            plan, _process = self.service.open_artifact(value)
        except Exception as exc:
            self._set_status("open failed")
            messagebox.showerror("Open artifact failed", str(exc))
            return
        self._set_status("artifact opened")
        self._log(f"Opened artifact: {plan.artifact}\n")

    def _selected_result_value(self) -> str:
        selection = self.result_tree.selection()
        if not selection:
            raise ValueError("Select an artifact row first.")
        item = self.result_tree.item(selection[0])
        values = item.get("values") or ()
        value = str(values[0]).strip() if values else ""
        if not value:
            raise ValueError("Selected row has no artifact path.")
        return value

    def _start_worker(self, state: str, work, on_success) -> None:
        if self._worker_active:
            messagebox.showwarning("Run active", "A QBlade bridge action is already running.")
            return
        self._stop_event.clear()
        self._worker_events = Queue()
        self._worker_active = True
        self._set_controls_enabled(False)
        self._set_status(state)

        def worker():
            try:
                result = work()
            except Exception as exc:
                self._worker_events.put((self._on_worker_failed, exc))
                return
            self._worker_events.put((on_success, result))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self._worker_poll_id = self.after(30, self._poll_worker_events)

    def _poll_worker_events(self) -> None:
        """Drain worker messages without requiring workers to enter Tcl/Tk."""
        self._worker_poll_id = None
        try:
            while True:
                callback, value = self._worker_events.get_nowait()
                callback(value)
        except Empty:
            if self._worker_active:
                self._worker_poll_id = self.after(30, self._poll_worker_events)

    def _on_prepare_complete(self, result: QBladeBridgePrepareResult) -> None:
        self._worker_active = False
        self._set_controls_enabled(True)
        self._set_status(result.state)
        self._set_rows(prepare_status_rows(result))
        self._log(f"QBlade bridge case {result.state}.\n")

    def _on_build_complete(self, result: QBladeBridgeBuildResult) -> None:
        self._worker_active = False
        self._set_controls_enabled(True)
        self._set_status(result.state)
        self._set_rows(build_status_rows(result))
        self._log(f"QBlade bridge library {result.state}.\n")

    def _on_sign_probe_complete(self, result: QBladeBridgeSignProbeResult) -> None:
        self._worker_active = False
        self._set_controls_enabled(True)
        self._set_status(f"sign probe {result.state}")
        self._set_rows(sign_probe_status_rows(result))
        self._log(f"QBlade sign probe {result.state}.\n")

    def _on_run_complete(self, result: QBladeBridgeRunServiceResult) -> None:
        self._worker_active = False
        self._set_controls_enabled(True)
        self._last_run_result = result
        self._set_status(result.state)
        self._set_rows(run_status_rows(result))
        self._log(f"QBlade bridge run {result.state}.\n")

    def _on_worker_failed(self, exc: Exception) -> None:
        self._worker_active = False
        self._set_controls_enabled(True)
        self._set_status("failed")
        self._set_rows([("Error", str(exc))])
        self._log(f"QBlade bridge action failed: {exc}\n")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self._action_buttons:
            button.configure(state=state)
        stop_button = getattr(self, "stop_button", None)
        if stop_button is not None:
            stop_button.configure(state="disabled" if enabled else "normal")

    def _set_status(self, value: str) -> None:
        self.status_var.set(value)

    def _set_rows(self, rows: list[tuple[str, str]]) -> None:
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        for key, value in rows:
            self.result_tree.insert("", "end", text=key, values=(value,))

    def _threadsafe_log(self, message: str) -> None:
        self._worker_events.put((self._log, message + ("" if message.endswith("\n") else "\n")))

    def _log(self, message: str) -> None:
        try:
            self.local_log.write(message)
        except Exception:
            pass
        try:
            self.app.log(message)
        except Exception:
            pass


__all__ = [
    "QBladeBridgeTab",
    "BRIDGE_SCENARIOS",
    "BRIDGE_USER_ACTIONS",
    "bridge_scenario_config",
    "bridge_config_warnings",
    "build_status_rows",
    "default_airfoil_dir",
    "default_python_executable",
    "default_qblade_root",
    "default_qblade_sim_path",
    "infer_model_airfoil_dir",
    "prepare_status_rows",
    "run_status_rows",
    "sign_probe_status_rows",
    "viv_parameter_status_rows",
]
