"""GUI-facing service facade for QBlade/VorLap bridge workflows."""

from __future__ import annotations

import csv
import ctypes
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

from .qblade_batch import apply_stable_support_patch, parse_bool_env, turbine_dir_for_sim
from .qblade_batch import build_bridge as build_qblade_bridge
from .qblade_batch import setup as setup_qblade_batch
from .qblade_batch import validate_qblade_root
from .qblade_bridge_config import QBladeBridgeConfig
from .qblade_case_prep import (
    QBladeCasePrepRequest,
    QBladeCasePrepResult,
    prepare_qblade_external_case,
    validate_component_airfoils,
    validate_qblade_model_references,
)
from .qblade_conversion import convert_qblade_to_vorlap_inputs
from .qblade_files import read_qblade_keyword, replace_qblade_keyword_file
from .qblade_nodes import infer_geometry_inclusions, position_channel_name
from .qblade_process import launch_isolated_process, terminate_process_tree
from .qblade_run import QBladeRunRequest, QBladeRunResult, run_qblade_case
from .qblade_runtime import GEOMETRY_AUDIT_COLUMNS, SOURCE_COVERAGE_COLUMNS, SourceCoverageRow
from .qblade_run_support import atomic_write_json, file_sha256, find_bridge_library, prepend_env_path, qblade_library_dirs
from .qblade_sil import normalize_qblade_header, read_qblade_result_numeric_columns


_GEOMETRY_REPORTED_TOLERANCE_M = 1.0e-4
_GEOMETRY_FRAME_FIT_TOLERANCE_M = 0.10
_GEOMETRY_AXES = ("X", "Y", "Z")


@dataclass(frozen=True)
class QBladeBridgeValidationResult:
    config: QBladeBridgeConfig
    errors: tuple[str, ...]
    qblade_gui_executable: Path | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class QBladeBridgePrepareResult:
    config: QBladeBridgeConfig
    prep_result: QBladeCasePrepResult
    state: str
    artifacts: dict[str, object]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class QBladeBridgeRunServiceResult:
    config: QBladeBridgeConfig
    run_result: QBladeRunResult
    state: str
    artifacts: dict[str, object]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class QBladeBridgeSignProbeResult:
    config: QBladeBridgeConfig
    state: str
    prep_result: QBladeCasePrepResult
    run_results: tuple[QBladeRunResult, ...]
    artifacts: dict[str, object]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class QBladeBridgeBatchSetupResult:
    manifest: dict[str, object]
    case_count: int
    artifacts: dict[str, object]


@dataclass(frozen=True)
class QBladeBridgeBuildResult:
    config: QBladeBridgeConfig
    state: str
    artifacts: dict[str, object]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class QBladeGuiLaunchPlan:
    executable: Path
    command: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    case_path_hint: Path
    project_path: Path | None = None
    project_source: str | None = None


@dataclass(frozen=True)
class QBladeArtifactOpenPlan:
    artifact: Path
    command: tuple[str, ...]


@dataclass(frozen=True)
class _MaterializedBridgeConfig:
    config: QBladeBridgeConfig
    artifacts: dict[str, object]
    diagnostics: dict[str, object]


class _QBladeLogTailer:
    """Forward QBlade log lines through the service callback while a run is active."""

    def __init__(
        self,
        path: Path,
        callback: Callable[[str], None] | None,
        *,
        poll_s: float = 0.05,
    ) -> None:
        self.path = Path(path)
        self.callback = callback
        self.poll_s = float(poll_s)
        self.forwarded_lines = 0
        self._offset = 0
        self._buffer = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.callback is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.callback is None:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._drain(final=True)

    def diagnostics(self) -> dict[str, object]:
        return {
            "qblade_log_tail_path": str(self.path),
            "qblade_log_tail_line_count": self.forwarded_lines,
            "qblade_log_tail_enabled": self.callback is not None,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._drain()
            self._stop.wait(self.poll_s)

    def _drain(self, *, final: bool = False) -> None:
        text = ""
        if self.path.is_file():
            size = self.path.stat().st_size
            if size < self._offset:
                self._offset = 0
                self._buffer = ""
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                text = handle.read()
                self._offset = handle.tell()
        elif not final:
            return

        chunk = self._buffer + text
        if not chunk:
            return
        if final or chunk.endswith(("\n", "\r")):
            lines = chunk.splitlines()
            self._buffer = ""
        else:
            lines = chunk.splitlines()
            self._buffer = lines.pop() if lines else chunk

        for line in lines:
            clean = line.strip()
            if not clean:
                continue
            if self.callback is not None:
                self.callback(f"QBlade log: {clean}")
            self.forwarded_lines += 1


def _expected_qblade_run_log_path(request: QBladeRunRequest) -> Path:
    output_dir = Path(request.output_dir).expanduser().resolve()
    sim_stem = Path(request.sim_path).expanduser().stem
    return output_dir / f"{sim_stem}_qblade.log"


def _expected_qblade_summary_path(request: QBladeRunRequest) -> Path:
    output_dir = Path(request.output_dir).expanduser().resolve()
    sim_stem = Path(request.sim_path).expanduser().stem
    return output_dir / f"{sim_stem}_run_summary.json"


def _qblade_run_result_from_summary(summary_path: Path) -> QBladeRunResult:
    summary_path = summary_path.expanduser().resolve(strict=False)
    if not summary_path.is_file():
        raise FileNotFoundError(f"QBlade run summary was not written: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"QBlade run summary is not a JSON object: {summary_path}")
    result_file = summary.get("result_file")
    run_log = summary.get("run_log")
    if not result_file:
        raise ValueError(f"QBlade run summary is missing result_file: {summary_path}")
    if not run_log:
        raise ValueError(f"QBlade run summary is missing run_log: {summary_path}")
    return QBladeRunResult(
        summary=summary,
        summary_path=summary_path,
        result_path=Path(str(result_file)).expanduser().resolve(strict=False),
        run_log_path=Path(str(run_log)).expanduser().resolve(strict=False),
    )


def find_qblade_gui_executable(qblade_root: str | Path) -> Path | None:
    """Best-effort detection of a QBlade GUI executable below a QBlade root."""
    root = Path(qblade_root)
    candidates = [
        root / root.name,
        root / "QBlade",
        root / "QBladeCE",
        root / "QBlade.exe",
        root / "QBladeCE.exe",
        root / "Binaries" / "QBlade",
        root / "Binaries" / "QBladeCE",
        root / "Binaries" / "QBlade.exe",
        root / "Binaries" / "QBladeCE.exe",
        root / "QBlade.app" / "Contents" / "MacOS" / "QBlade",
        root / "QBladeCE.app" / "Contents" / "MacOS" / "QBladeCE",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _existing_path(raw_path: object, *, base: Path | None = None) -> Path | None:
    if raw_path in (None, ""):
        return None
    try:
        path = Path(str(raw_path)).expanduser()
    except TypeError:
        return None
    if not path.is_absolute() and base is not None:
        path = base / path
    path = path.resolve(strict=False)
    return path if path.is_file() else None


def _checkpoint_project_from_metadata(metadata_path: Path) -> Path | None:
    metadata_path = metadata_path.expanduser().resolve(strict=False)
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(metadata, Mapping):
        return None
    return _existing_path(metadata.get("qblade_project"), base=metadata_path.parent)


def _project_from_run_context(run_context: object) -> tuple[Path | None, str | None]:
    if run_context is None:
        return None, None
    summary: Mapping[str, object] | None = None
    artifacts: Mapping[str, object] | None = None
    diagnostics: Mapping[str, object] | None = None
    if isinstance(run_context, QBladeBridgeRunServiceResult):
        summary = run_context.run_result.summary
        artifacts = run_context.artifacts
        diagnostics = run_context.diagnostics
    elif isinstance(run_context, QBladeRunResult):
        summary = run_context.summary
    elif isinstance(run_context, Mapping):
        summary = run_context

    if artifacts is not None:
        project = _existing_path(artifacts.get("stored_project"))
        if project is not None:
            return project, "completed run project"
    if summary is not None:
        project = _existing_path(summary.get("stored_project"))
        if project is not None:
            return project, "completed run project"
        restart = summary.get("restart")
        if isinstance(restart, Mapping):
            project = _existing_path(restart.get("latest_checkpoint_project"))
            if project is not None:
                return project, "latest restart checkpoint"
    if diagnostics is not None:
        project = _existing_path(diagnostics.get("checkpoint_latest_project"))
        if project is not None:
            return project, "latest restart checkpoint"
    return None, None


def _project_from_config_paths(config: QBladeBridgeConfig) -> tuple[Path | None, str | None]:
    output_dir = Path(config.output_dir).expanduser().resolve(strict=False)
    sim_stem = Path(config.sim_path).stem
    project = _existing_path(output_dir / f"{sim_stem}_completed.qpr")
    if project is not None:
        return project, "completed run project"

    checkpoint_dir = (
        Path(config.checkpoint_dir).expanduser().resolve(strict=False)
        if config.checkpoint_dir not in (None, "")
        else output_dir / "checkpoints"
    )
    project = _checkpoint_project_from_metadata(checkpoint_dir / "latest_checkpoint.json")
    if project is not None:
        return project, "latest restart checkpoint"
    return None, None


def _prep_artifacts(
    result: QBladeCasePrepResult,
    materialized: _MaterializedBridgeConfig | None = None,
) -> dict[str, object]:
    artifacts = {
        "sim": str(result.sim_path),
        "turbine": str(result.turbine_path),
        "structural": str(result.structural_path),
        "runtime_config": str(result.parameter_file_path),
        "geometry_audit": None if result.geometry_audit_path is None else str(result.geometry_audit_path),
        "planned_write_paths": [str(path) for path in result.planned_write_paths],
        "written_files": [str(path) for path in result.written_files],
    }
    if materialized is not None:
        artifacts.update(materialized.artifacts)
    return artifacts


def _prep_diagnostics(
    result: QBladeCasePrepResult,
    materialized: _MaterializedBridgeConfig | None = None,
) -> dict[str, object]:
    diagnostics = {
        "node_source": result.node_source,
        "node_count": len(result.node_ids),
        "structural_node_count": len(result.structural_node_ids),
        "converted_node_count": len(result.converted_node_ids),
        "swap_size": result.swap_size,
        "component_count": result.component_count,
        "include_struts": result.convert_include_struts,
        "include_tower": result.convert_include_tower,
        "dry_run": result.dry_run,
        "source_bounds_policy": result.runtime_config.get("source_bounds_policy"),
        "source_frequency_cap_hz": result.runtime_config.get("source_frequency_cap_hz"),
    }
    diagnostics.update(_geometry_audit_diagnostics(result.geometry_audit_path, result.node_ids))
    if materialized is not None:
        diagnostics.update(materialized.diagnostics)
    return diagnostics


def _duplicate_value_count(values: Sequence[object]) -> int:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            duplicates.add(text)
        seen.add(text)
    return len(duplicates)


def _geometry_audit_diagnostics(audit_path: Path | None, node_ids: Sequence[str]) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "geometry_audit_exists": False,
        "geometry_audit_rows": None,
        "geometry_audit_basis_error_count": None,
        "geometry_audit_basis_warning_count": None,
        "geometry_audit_basis_error_nodes": [],
        "geometry_audit_duplicate_node_count": _duplicate_value_count(node_ids),
    }
    if audit_path is None:
        return diagnostics
    path = Path(audit_path)
    diagnostics["geometry_audit_exists"] = path.is_file()
    if not path.is_file():
        return diagnostics

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    error_nodes: list[str] = []
    warning_count = 0
    for row in rows:
        status = str(row.get("basis_status", "")).strip().lower()
        warnings_text = str(row.get("basis_warnings", "")).strip()
        if status and status != "ok":
            error_nodes.append(str(row.get("requested_node_id", "")).strip() or "?")
        if warnings_text:
            warning_count += 1
    diagnostics.update(
        {
            "geometry_audit_rows": len(rows),
            "geometry_audit_basis_error_count": len(error_nodes),
            "geometry_audit_basis_warning_count": warning_count,
            "geometry_audit_basis_error_nodes": error_nodes[:8],
        }
    )
    return diagnostics


def _geometry_audit_path_from_summary(summary: dict[str, object]) -> Path | None:
    for key in ("sil_vorlap_geometry_audit_file", "geometry_audit_file", "geometry_audit"):
        raw_path = summary.get(key)
        if isinstance(raw_path, str) and raw_path.strip():
            return Path(raw_path).expanduser().resolve(strict=False)
    return None


def _run_artifacts(
    result: QBladeRunResult,
    materialized: _MaterializedBridgeConfig | None = None,
    quick_plots: dict[str, str] | None = None,
) -> dict[str, object]:
    summary = result.summary
    source_coverage_path = _write_source_coverage_report(result)
    quick_plot_artifacts = quick_plots or {}
    artifacts = {
        "summary": str(result.summary_path),
        "result_file": str(result.result_path),
        "run_log": str(result.run_log_path),
        "stored_project": summary.get("stored_project"),
        "sil_vorlap_history": summary.get("sil_vorlap_history"),
        "sil_vorlap_source_coverage": None if source_coverage_path is None else str(source_coverage_path),
        "geometry_audit": summary.get("sil_vorlap_geometry_audit_file"),
        "bridge_quick_plot_dir": quick_plot_artifacts.get("bridge_quick_plot_dir"),
        "bridge_quick_plot_max_force": quick_plot_artifacts.get("bridge_quick_plot_max_force"),
        "bridge_quick_plot_aoa_re": quick_plot_artifacts.get("bridge_quick_plot_aoa_re"),
        "bridge_quick_plot_nyquist": quick_plot_artifacts.get("bridge_quick_plot_nyquist"),
        "bridge_quick_plot_force_history": quick_plot_artifacts.get("bridge_quick_plot_force_history"),
        "bridge_quick_plot_qblade_response": quick_plot_artifacts.get("bridge_quick_plot_qblade_response"),
        "native_callback_log": summary.get("native_callback_evidence", {}).get("log_file")
        if isinstance(summary.get("native_callback_evidence"), dict)
        else None,
    }
    if materialized is not None:
        artifacts.update(materialized.artifacts)
    return artifacts


def _run_diagnostics(
    result: QBladeRunResult,
    materialized: _MaterializedBridgeConfig | None = None,
    quick_plot_diagnostics: dict[str, object] | None = None,
    geometry_diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = result.summary
    source_coverage = _source_coverage_rows(summary)
    restart = summary.get("restart")
    restart_info = restart if isinstance(restart, dict) else {}
    diagnostics = {
        "requested_steps": summary.get("requested_steps"),
        "completed_steps": summary.get("completed_steps"),
        "stopped": summary.get("stopped"),
        "stop_message": summary.get("stop_message"),
        "failed": summary.get("failed"),
        "failure_message": summary.get("failure_message"),
        "coupling_mode": summary.get("coupling_mode"),
        "qblade_execution_mode": summary.get("qblade_execution_mode"),
        "stored_project": summary.get("stored_project"),
        "sil_vorlap_message": summary.get("sil_vorlap_message"),
        "sil_vorlap_max_abs_force": summary.get("sil_vorlap_max_abs_force"),
        "sil_vorlap_max_structural_velocity_mps": _max_structural_velocity_mps(summary, source_coverage),
        "sil_vorlap_velocity_warning_count": _velocity_warning_count(summary),
        "sil_vorlap_cache": _cache_label(summary),
        "sil_vorlap_velocity_diagnostics": summary.get("sil_vorlap_velocity_diagnostics"),
        "sil_vorlap_timing_diagnostics": summary.get("sil_vorlap_timing_diagnostics"),
        "sil_vorlap_cache_stats": summary.get("sil_vorlap_cache_stats"),
        "sil_vorlap_source_coverage": source_coverage,
        "sil_vorlap_source_coverage_rows": len(source_coverage),
        "source_bounds_policy": summary.get("sil_vorlap_source_bounds_policy"),
        "source_frequency_cap_hz": summary.get("sil_vorlap_source_frequency_cap_hz"),
        "checkpoint_latest": restart_info.get("latest_checkpoint"),
        "checkpoint_latest_project": restart_info.get("latest_checkpoint_project"),
        "checkpoint_latest_completed_steps": restart_info.get("latest_checkpoint_completed_steps"),
        "resume_requested": restart_info.get("resume_requested"),
        "resume_start_step": restart_info.get("resume_start_step"),
        "native_callback_evidence": summary.get("native_callback_evidence"),
        "qblade_export_time_diagnostics": summary.get("qblade_export_time_diagnostics"),
    }
    geometry_audit_path = _geometry_audit_path_from_summary(summary)
    if geometry_audit_path is not None:
        diagnostics.update(_geometry_audit_diagnostics(geometry_audit_path, ()))
    if geometry_diagnostics is not None:
        diagnostics.update(geometry_diagnostics)
    if quick_plot_diagnostics is not None:
        diagnostics.update(quick_plot_diagnostics)
    if materialized is not None:
        diagnostics.update(materialized.diagnostics)
    return diagnostics


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _model_override_values(config: QBladeBridgeConfig) -> dict[str, float | int | str]:
    overrides: dict[str, float | int | str] = {}
    if config.wind_speed is not None:
        overrides["MEANINF"] = float(config.wind_speed)
    if config.azimuth_deg is not None:
        overrides["INITIAL_AZIMUTH"] = float(config.azimuth_deg)
    if config.rpm is not None:
        overrides["PRESCRIBETYPE"] = 1
        overrides["RPMPRESCRIBED"] = float(config.rpm)
    if config.timestep_s is not None:
        overrides["TIMESTEP"] = float(config.timestep_s)
    if config.effective_steps() is not None:
        overrides["NUMTIMESTEPS"] = int(config.effective_steps())
    if config.store_from_s is not None:
        overrides["STOREFROM"] = float(config.store_from_s)
    if config.store_replay is not None:
        overrides["STOREREPLAY"] = "true" if config.store_replay else "false"
    return overrides


def _default_vorlap_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _platform_loader_path_key() -> str:
    if os.name == "nt":
        return "PATH"
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


SIGN_PROBE_AXES = {"X", "Y", "Z"}
SIGN_PROBE_REPORT_COLUMNS = (
    "node",
    "axis",
    "sign",
    "force_n",
    "local",
    "output_dir",
    "summary_file",
    "result_file",
    "requested_steps",
    "completed_steps",
    "failed",
    "exported_results",
    "observation_status",
    "observed_position_channel",
    "delta_position_m",
    "paired_response_delta_m",
    "paired_passed",
    "passed",
    "issue",
)


def _reference_stable_support_manifest(sim_path: Path) -> Path:
    return turbine_dir_for_sim(sim_path) / "Control" / "vorlap_stable_support_patch.json"


def _maybe_apply_reference_stable_support_patch(config: QBladeBridgeConfig) -> Path | None:
    """Apply a support-model workaround only when explicitly requested."""
    if not parse_bool_env(os.environ.get("VORLAP_STABLE_PATCH_REMOVE_SUBFILE"), default=False):
        return None
    sim_path = Path(config.sim_path).expanduser().resolve()
    case_dir = sim_path.parent

    timestep = config.timestep_s
    if timestep is None:
        raw_timestep = read_qblade_keyword(sim_path, "TIMESTEP")
        timestep = 0.0075 if raw_timestep in (None, "") else float(raw_timestep)
    apply_stable_support_patch(
        case_dir,
        sim_path,
        timestep=float(timestep),
        rampup=0.0,
        add_damp=0.0,
        str_iterations=None,
    )
    return _reference_stable_support_manifest(sim_path)


def _parse_sign_probe_axes(axes: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(axes, str):
        values = [part.upper() for part in re.split(r"[\s,]+", axes.strip()) if part]
    else:
        values = [str(part).upper() for part in axes]
    if not values:
        raise ValueError("At least one sign-probe axis is required.")
    bad = [axis for axis in values if axis not in SIGN_PROBE_AXES]
    if bad:
        raise ValueError(f"Unsupported sign-probe axis/axes: {', '.join(bad)}.")
    return tuple(values)


def _safe_label(text: object) -> str:
    label = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text).strip())
    return label.strip("_") or "probe"


def _default_sign_probe_node(node_ids: Sequence[str]) -> str:
    for node_id in node_ids:
        if str(node_id).startswith("BLD_1_"):
            return str(node_id)
    for node_id in node_ids:
        if str(node_id).startswith("BLD_"):
            return str(node_id)
    if node_ids:
        return str(node_ids[0])
    raise ValueError("No QBlade nodes are available for a force sign probe.")


def _write_json(path: Path, data: dict[str, object]) -> None:
    atomic_write_json(path, data)


def _working_copy_file_rows(source_dir: Path, working_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_path in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        relative = source_path.relative_to(source_dir)
        working_path = working_dir / relative
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "source": str(source_path),
                "copy": str(working_path),
                "source_hash": file_sha256(source_path),
                "copy_hash": file_sha256(working_path),
            }
        )
    return rows


def _write_working_copy_manifest(
    *,
    manifest_path: Path,
    source_sim: Path,
    working_sim: Path,
    patches: Sequence[dict[str, object]],
) -> None:
    source_dir = source_sim.parent
    working_dir = working_sim.parent
    copied_files = _working_copy_file_rows(source_dir, working_dir)
    patched_files = sorted({str(patch.get("file")) for patch in patches if patch.get("file")})
    _write_json(
        manifest_path,
        {
            "source_model_dir": str(source_dir),
            "working_model_dir": str(working_dir),
            "source_sim": str(source_sim),
            "working_sim": str(working_sim),
            "copied_files": copied_files,
            "modified_files": patched_files,
            "qblade_keyword_patches": [dict(patch) for patch in patches],
        },
    )


def _finalize_working_copy_manifest(manifest_path: Path, prep_result: QBladeCasePrepResult) -> None:
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planned = [str(path) for path in prep_result.planned_write_paths]
    written = [str(path) for path in prep_result.written_files]
    existing_modified = set(str(path) for path in manifest.get("modified_files", []) if path)
    existing_modified.update(written)
    manifest["case_preparation_planned_write_paths"] = planned
    manifest["case_preparation_written_files"] = written
    manifest["case_preparation_written_hashes"] = [
        {
            "path": str(path),
            "hash": file_sha256(path),
        }
        for path in prep_result.written_files
    ]
    manifest["modified_files"] = sorted(existing_modified)
    _write_json(manifest_path, manifest)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _source_coverage_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows = summary.get("sil_vorlap_source_coverage")
    if not isinstance(rows, list):
        return []
    return [SourceCoverageRow.normalize_mapping(row) for row in rows if isinstance(row, dict)]


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_complete_position_sample(
    columns: dict[str, list[float]],
    channel_names: Sequence[str],
) -> tuple[int | None, tuple[float, float, float] | None]:
    if len(channel_names) != len(_GEOMETRY_AXES):
        return None, None
    sample_count = min((len(columns.get(name, [])) for name in channel_names), default=0)
    for sample_index in range(sample_count):
        values: list[float] = []
        for channel_name in channel_names:
            channel_values = columns.get(channel_name, [])
            value = _float_or_none(channel_values[sample_index] if sample_index < len(channel_values) else None)
            if value is None:
                break
            values.append(value)
        if len(values) == len(_GEOMETRY_AXES):
            return sample_index, (values[0], values[1], values[2])
    return None, None


def _qblade_position_channels_for_geometry_row(
    row: dict[str, object],
    columns: dict[str, list[float]],
) -> tuple[tuple[str, str, str] | None, list[str]]:
    action_id = str(row.get("action_id", "")).strip()
    action_position = _float_or_none(row.get("action_position"))
    if not action_id or action_position is None:
        return None, ["missing action_id/action_position"]

    normalized = {normalize_qblade_header(name): name for name in columns}
    channel_names: list[str] = []
    missing: list[str] = []
    for axis in _GEOMETRY_AXES:
        expected = position_channel_name({"id": action_id, "position": action_position, "direction": axis})
        actual = normalized.get(normalize_qblade_header(expected))
        if actual is None:
            missing.append(expected)
        else:
            channel_names.append(actual)
    if missing:
        return None, missing
    return (channel_names[0], channel_names[1], channel_names[2]), []


def _rigid_frame_fit_diagnostics(
    vorlap_points: Sequence[Sequence[float]],
    qblade_points: Sequence[Sequence[float]],
) -> tuple[dict[str, object], np.ndarray | None, np.ndarray | None]:
    diagnostics: dict[str, object] = {
        "geometry_audit_reported_frame_fit_available": False,
        "geometry_audit_reported_frame_fit_status": "insufficient_points",
        "geometry_audit_reported_frame_fit_sample_count": len(vorlap_points),
        "geometry_audit_reported_frame_fit_rank": None,
        "geometry_audit_reported_frame_fit_rotation": None,
        "geometry_audit_reported_frame_fit_translation_m": None,
        "geometry_audit_reported_frame_fit_determinant": None,
        "geometry_audit_reported_frame_fit_reflection_corrected": False,
        "geometry_audit_reported_frame_fit_max_residual_m": None,
        "geometry_audit_reported_frame_fit_mean_residual_m": None,
        "geometry_audit_reported_frame_fit_rms_residual_m": None,
        "geometry_audit_reported_frame_fit_tolerance_m": _GEOMETRY_FRAME_FIT_TOLERANCE_M,
    }
    if len(vorlap_points) < 3 or len(qblade_points) < 3:
        return diagnostics, None, None
    p = np.asarray(vorlap_points, dtype=float)
    q = np.asarray(qblade_points, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        diagnostics["geometry_audit_reported_frame_fit_status"] = "invalid_shape"
        return diagnostics, None, None
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        diagnostics["geometry_audit_reported_frame_fit_status"] = "nonfinite_points"
        return diagnostics, None, None

    p_centroid = p.mean(axis=0)
    q_centroid = q.mean(axis=0)
    p_centered = p - p_centroid
    q_centered = q - q_centroid
    rank = int(np.linalg.matrix_rank(p_centered))
    diagnostics["geometry_audit_reported_frame_fit_rank"] = rank
    if rank < 2:
        diagnostics["geometry_audit_reported_frame_fit_status"] = "insufficient_rank"
        return diagnostics, None, None

    left, _singular_values, right_t = np.linalg.svd(p_centered.T @ q_centered)
    rotation = right_t.T @ left.T
    reflection_corrected = False
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
        reflection_corrected = True
    translation = q_centroid - rotation @ p_centroid
    fitted = (rotation @ p.T).T + translation
    residuals = np.linalg.norm(q - fitted, axis=1)
    max_residual = float(np.max(residuals))
    diagnostics.update(
        {
            "geometry_audit_reported_frame_fit_available": True,
            "geometry_audit_reported_frame_fit_status": (
                "within_tolerance"
                if max_residual <= _GEOMETRY_FRAME_FIT_TOLERANCE_M
                else "outside_tolerance"
            ),
            "geometry_audit_reported_frame_fit_rotation": rotation.tolist(),
            "geometry_audit_reported_frame_fit_translation_m": translation.tolist(),
            "geometry_audit_reported_frame_fit_determinant": float(np.linalg.det(rotation)),
            "geometry_audit_reported_frame_fit_reflection_corrected": reflection_corrected,
            "geometry_audit_reported_frame_fit_max_residual_m": max_residual,
            "geometry_audit_reported_frame_fit_mean_residual_m": float(np.mean(residuals)),
            "geometry_audit_reported_frame_fit_rms_residual_m": float(math.sqrt(np.mean(residuals * residuals))),
        }
    )
    return diagnostics, fitted, residuals


def write_run_geometry_audit_reported_positions(
    result: QBladeRunResult,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Write QBlade-reported positions to a geometry-audit CSV.

    ``output_path`` keeps a hash-pinned prepared audit immutable by writing the
    observed columns beside the run artifacts.  The default retains the
    historical in-place behavior for interactive bridge-service callers.
    """
    diagnostics: dict[str, object] = {
        "geometry_audit_reported_path": None,
        "geometry_audit_reported_updated": False,
        "geometry_audit_reported_available_count": None,
        "geometry_audit_reported_missing_count": None,
        "geometry_audit_reported_error_count": None,
        "geometry_audit_reported_error_nodes": [],
        "geometry_audit_reported_max_delta_m": None,
        "geometry_audit_reported_tolerance_m": _GEOMETRY_REPORTED_TOLERANCE_M,
        "geometry_audit_reported_update_error": None,
        "geometry_audit_reported_frame_fit_available": False,
        "geometry_audit_reported_frame_fit_status": None,
        "geometry_audit_reported_frame_fit_sample_count": None,
        "geometry_audit_reported_frame_fit_max_residual_m": None,
        "geometry_audit_reported_frame_fit_mean_residual_m": None,
        "geometry_audit_reported_frame_fit_rms_residual_m": None,
        "geometry_audit_reported_frame_fit_tolerance_m": _GEOMETRY_FRAME_FIT_TOLERANCE_M,
    }
    prepared_audit_path = _geometry_audit_path_from_summary(result.summary)
    if prepared_audit_path is None:
        return diagnostics
    reported_audit_path = (
        prepared_audit_path
        if output_path is None
        else Path(output_path).expanduser().resolve(strict=False)
    )
    diagnostics["geometry_audit_reported_path"] = str(reported_audit_path)
    if not prepared_audit_path.is_file():
        return diagnostics

    try:
        with prepared_audit_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        columns = read_qblade_result_numeric_columns(result.result_path)
        if not columns:
            diagnostics.update(
                {
                    "geometry_audit_reported_available_count": 0,
                    "geometry_audit_reported_missing_count": len(rows),
                    "geometry_audit_reported_error_count": 0,
                }
            )
            return diagnostics

        available_count = 0
        missing_count = 0
        error_nodes: list[str] = []
        deltas: list[float] = []
        matched_rows: list[dict[str, object]] = []
        matched_vorlap_points: list[tuple[float, float, float]] = []
        matched_qblade_points: list[tuple[float, float, float]] = []
        for row in rows:
            channel_names, _missing_channels = _qblade_position_channels_for_geometry_row(row, columns)
            if channel_names is None:
                row["qblade_reported_available"] = "false"
                row["qblade_frame_fit_available"] = "false"
                missing_count += 1
                continue
            sample_index, qblade_xyz = _first_complete_position_sample(columns, channel_names)
            if sample_index is None or qblade_xyz is None:
                row["qblade_reported_available"] = "false"
                row["qblade_frame_fit_available"] = "false"
                missing_count += 1
                continue
            vorlap_xyz = (
                _float_or_none(row.get("vorlap_x_m")),
                _float_or_none(row.get("vorlap_y_m")),
                _float_or_none(row.get("vorlap_z_m")),
            )
            row.update(
                {
                    "qblade_reported_available": "true",
                    "qblade_reported_x_m": qblade_xyz[0],
                    "qblade_reported_y_m": qblade_xyz[1],
                    "qblade_reported_z_m": qblade_xyz[2],
                }
            )
            available_count += 1
            if any(value is None for value in vorlap_xyz):
                row["delta_reported_m"] = ""
                row["qblade_frame_fit_available"] = "false"
                continue
            delta = math.sqrt(
                sum((float(qblade_xyz[index]) - float(vorlap_xyz[index])) ** 2 for index in range(3))
            )
            row["delta_reported_m"] = delta
            deltas.append(delta)
            matched_rows.append(row)
            matched_vorlap_points.append((float(vorlap_xyz[0]), float(vorlap_xyz[1]), float(vorlap_xyz[2])))
            matched_qblade_points.append((float(qblade_xyz[0]), float(qblade_xyz[1]), float(qblade_xyz[2])))
            if delta > _GEOMETRY_REPORTED_TOLERANCE_M:
                error_nodes.append(str(row.get("requested_node_id", "")).strip() or "?")

        frame_fit, fitted_points, frame_residuals = _rigid_frame_fit_diagnostics(
            matched_vorlap_points,
            matched_qblade_points,
        )
        diagnostics.update(frame_fit)
        if fitted_points is not None and frame_residuals is not None:
            for row, fitted_point, residual in zip(matched_rows, fitted_points, frame_residuals, strict=True):
                row.update(
                    {
                        "qblade_frame_fit_available": "true",
                        "qblade_frame_fit_x_m": float(fitted_point[0]),
                        "qblade_frame_fit_y_m": float(fitted_point[1]),
                        "qblade_frame_fit_z_m": float(fitted_point[2]),
                        "delta_reported_frame_fit_m": float(residual),
                    }
                )

        known = list(GEOMETRY_AUDIT_COLUMNS)
        extras = sorted({str(key) for row in rows for key in row if str(key) not in GEOMETRY_AUDIT_COLUMNS})
        fieldnames = [*known, *extras]
        reported_audit_path.parent.mkdir(parents=True, exist_ok=True)
        with reported_audit_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})

        diagnostics.update(
            {
                "geometry_audit_reported_updated": True,
                "geometry_audit_reported_available_count": available_count,
                "geometry_audit_reported_missing_count": missing_count,
                "geometry_audit_reported_error_count": len(error_nodes),
                "geometry_audit_reported_error_nodes": error_nodes[:8],
                "geometry_audit_reported_max_delta_m": max(deltas) if deltas else None,
            }
        )
    except Exception as exc:
        diagnostics["geometry_audit_reported_update_error"] = str(exc)
    return diagnostics


def _max_structural_velocity_mps(summary: dict[str, object], source_coverage: Sequence[dict[str, object]]) -> float | None:
    maxima: list[float] = []
    velocity = summary.get("sil_vorlap_velocity_diagnostics")
    if isinstance(velocity, dict):
        axis_stats = velocity.get("axis_stats")
        if isinstance(axis_stats, dict):
            for stat in axis_stats.values():
                if isinstance(stat, dict):
                    value = _float_or_none(stat.get("max_abs"))
                    if value is not None:
                        maxima.append(value)
    for row in source_coverage:
        value = _float_or_none(row.get("max_structural_velocity_mps"))
        if value is not None:
            maxima.append(value)
    return max(maxima) if maxima else None


def _velocity_warning_count(summary: dict[str, object]) -> int | None:
    velocity = summary.get("sil_vorlap_velocity_diagnostics")
    if not isinstance(velocity, dict):
        return None
    all_zero_axes = velocity.get("all_zero_axes")
    all_zero_channels = velocity.get("all_zero_channels")
    axis_count = len(all_zero_axes) if isinstance(all_zero_axes, list) else 0
    channel_count = len(all_zero_channels) if isinstance(all_zero_channels, list) else 0
    return axis_count + channel_count


def _cache_label(summary: dict[str, object]) -> str | None:
    cache_stats = summary.get("sil_vorlap_cache_stats")
    if not isinstance(cache_stats, dict):
        return None
    hits = int(cache_stats.get("hits", 0) or 0)
    misses = int(cache_stats.get("misses", 0) or 0)
    return f"{hits}/{misses}"


def _run_version_command(command: Sequence[str], *, timeout_s: float = 10.0) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        return None, str(exc)
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip()).strip()
    first_line = output.splitlines()[0] if output else ""
    if completed.returncode != 0:
        return None, first_line or f"exit status {completed.returncode}"
    return first_line, None


def _python_runtime_diagnostics(python_executable: str | Path) -> dict[str, object]:
    code = (
        "import json, sys\n"
        "try:\n"
        "    import numpy\n"
        "    numpy_version = numpy.__version__\n"
        "    numpy_error = None\n"
        "except Exception as exc:\n"
        "    numpy_version = None\n"
        "    numpy_error = str(exc)\n"
        "print(json.dumps({\n"
        "    'python_executable': sys.executable,\n"
        "    'python_version': sys.version.split()[0],\n"
        "    'numpy_version': numpy_version,\n"
        "    'numpy_error': numpy_error,\n"
        "}))\n"
    )
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except Exception as exc:
        return {
            "python_executable": str(python_executable),
            "python_version": None,
            "numpy_version": None,
            "numpy_error": None,
            "python_probe_error": str(exc),
        }
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or f"exit status {completed.returncode}").strip()
        return {
            "python_executable": str(python_executable),
            "python_version": None,
            "numpy_version": None,
            "numpy_error": None,
            "python_probe_error": message,
        }
    try:
        parsed = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "python_executable": str(python_executable),
            "python_version": None,
            "numpy_version": None,
            "numpy_error": None,
            "python_probe_error": f"Could not parse Python probe output: {exc}",
        }
    parsed["python_probe_error"] = None
    return parsed


def _compiler_version() -> tuple[str | None, str | None]:
    candidates = []
    if os.environ.get("CC"):
        candidates.append(os.environ["CC"])
    candidates.extend(["cc", "gcc", "clang"])
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        executable = shutil.which(candidate)
        if executable is None:
            continue
        version, error = _run_version_command([executable, "--version"])
        if version is not None:
            return version, None
        if error:
            return None, error
    return None, "not found"


def _shared_library_load_diagnostics(path: Path | None) -> tuple[bool | None, str | None]:
    if path is None:
        return None, "bridge library not found"
    try:
        ctypes.CDLL(str(path))
    except Exception as exc:
        return False, str(exc)
    return True, None


def _write_source_coverage_report(result: QBladeRunResult) -> Path | None:
    rows = _source_coverage_rows(result.summary)
    if not rows:
        return None

    known = list(SOURCE_COVERAGE_COLUMNS)
    extras = sorted({str(key) for row in rows for key in row if str(key) not in SOURCE_COVERAGE_COLUMNS})
    fieldnames = [*known, *extras]
    report_path = result.summary_path.with_name(f"{result.summary_path.stem}_source_coverage.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    return report_path


def _plot_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plot_label(value: object, fallback: str) -> str:
    text = str(value).strip() if value not in (None, "") else ""
    return text or fallback


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_plot(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def _write_source_force_plot(rows: Sequence[dict[str, object]], plot_dir: Path, plt) -> Path | None:
    values = [
        (_plot_label(row.get("component"), f"component {index}"), _plot_float(row.get("max_nodal_force_n")))
        for index, row in enumerate(rows, start=1)
    ]
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        return None
    labels, force_values = zip(*values)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(labels, force_values, color="#2563eb")
    ax.set_ylabel("Max nodal force [N]")
    ax.set_title("Bridge Max Nodal Force")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = plot_dir / "max_nodal_force_by_component.png"
    _save_plot(fig, path)
    plt.close(fig)
    return path


def _write_aoa_re_plot(rows: Sequence[dict[str, object]], plot_dir: Path, plt) -> Path | None:
    labels: list[str] = []
    aoa_actual: list[tuple[float, float] | None] = []
    aoa_source: list[tuple[float, float] | None] = []
    re_actual: list[tuple[float, float] | None] = []
    re_source: list[tuple[float, float] | None] = []
    for index, row in enumerate(rows, start=1):
        label = _plot_label(row.get("component"), f"component {index}")
        actual_aoa = (_plot_float(row.get("aoa_min_deg")), _plot_float(row.get("aoa_max_deg")))
        source_aoa = (_plot_float(row.get("source_aoa_min_deg")), _plot_float(row.get("source_aoa_max_deg")))
        actual_re = (_plot_float(row.get("re_min")), _plot_float(row.get("re_max")))
        source_re = (_plot_float(row.get("source_re_min")), _plot_float(row.get("source_re_max")))
        has_any = any(value is not None for pair in (actual_aoa, source_aoa, actual_re, source_re) for value in pair)
        if not has_any:
            continue
        labels.append(label)
        aoa_actual.append(actual_aoa if all(value is not None for value in actual_aoa) else None)  # type: ignore[arg-type]
        aoa_source.append(source_aoa if all(value is not None for value in source_aoa) else None)  # type: ignore[arg-type]
        re_actual.append(actual_re if all(value is not None for value in actual_re) else None)  # type: ignore[arg-type]
        re_source.append(source_re if all(value is not None for value in source_re) else None)  # type: ignore[arg-type]
    if not labels:
        return None

    x = list(range(len(labels)))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    for axis, actual, source, ylabel, title in (
        (axes[0], aoa_actual, aoa_source, "AOA [deg]", "Angle of Attack Range"),
        (axes[1], re_actual, re_source, "Re [-]", "Reynolds Range"),
    ):
        for idx, interval in enumerate(source):
            if interval is not None:
                axis.vlines(x[idx], interval[0], interval[1], color="#9ca3af", linewidth=6, alpha=0.55, label="source" if idx == 0 else "")
        for idx, interval in enumerate(actual):
            if interval is not None:
                axis.vlines(x[idx], interval[0], interval[1], color="#dc2626", linewidth=2.5, label="actual" if idx == 0 else "")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    fig.tight_layout()
    path = plot_dir / "aoa_re_source_limits.png"
    _save_plot(fig, path)
    plt.close(fig)
    return path


def _write_nyquist_plot(rows: Sequence[dict[str, object]], plot_dir: Path, plt) -> Path | None:
    values = [
        (_plot_label(row.get("component"), f"component {index}"), _plot_float(row.get("nyquist_ratio")))
        for index, row in enumerate(rows, start=1)
    ]
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        return None
    labels, ratios = zip(*values)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = ["#16a34a" if ratio >= 1.0 else "#dc2626" for ratio in ratios]
    ax.bar(labels, ratios, color=colors)
    ax.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--", label="ratio = 1")
    ax.set_ylabel("Nyquist / max source frequency [-]")
    ax.set_title("Bridge Nyquist Margin")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best")
    fig.tight_layout()
    path = plot_dir / "nyquist_margin.png"
    _save_plot(fig, path)
    plt.close(fig)
    return path


def _read_numeric_csv_columns(path: Path) -> dict[str, list[float]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns: dict[str, list[float]] = {str(name): [] for name in (reader.fieldnames or [])}
        for row in reader:
            for key in columns:
                value = _plot_float(row.get(key))
                if value is not None:
                    columns[key].append(value)
    return columns


def _write_force_history_plot(summary: dict[str, object], plot_dir: Path, plt) -> tuple[Path | None, str | None]:
    history_value = summary.get("sil_vorlap_history")
    history_path = Path(str(history_value)).expanduser() if history_value not in (None, "") else None
    if history_path is None:
        return None, None
    columns = _read_numeric_csv_columns(history_path)
    if not columns:
        return None, None
    x_name = "time_s" if columns.get("time_s") else "step_index"
    y_name = "max_force_norm_n" if columns.get("max_force_norm_n") else "max_abs_force_component_n"
    x_values = columns.get(x_name, [])
    y_values = columns.get(y_name, [])
    count = min(len(x_values), len(y_values))
    if count < 1:
        return None, None
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(x_values[:count], y_values[:count], color="#2563eb", linewidth=1.8)
    ax.set_xlabel("Time [s]" if x_name == "time_s" else "Step [-]")
    ax.set_ylabel("Force [N]")
    ax.set_title("Bridge Selected Force History")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plot_dir / "selected_force_history.png"
    _save_plot(fig, path)
    plt.close(fig)
    return path, y_name


def _qblade_time_column(columns: dict[str, list[float]]) -> str | None:
    for name in columns:
        normalized = normalize_qblade_header(name)
        if normalized == "Time [s]" or normalized.startswith("Time "):
            return name
    return None


def _qblade_response_column(columns: dict[str, list[float]], time_column: str | None) -> str | None:
    candidates = [name for name in columns if name != time_column and columns.get(name)]
    position_candidates = [
        name
        for name in candidates
        if " Pos" in normalize_qblade_header(name) or "Position" in normalize_qblade_header(name)
    ]
    return (position_candidates or candidates or [None])[0]


def _write_qblade_response_plot(result: QBladeRunResult, plot_dir: Path, plt) -> tuple[Path | None, str | None]:
    columns = read_qblade_result_numeric_columns(result.result_path)
    if not columns:
        return None, None
    time_column = _qblade_time_column(columns)
    response_column = _qblade_response_column(columns, time_column)
    if response_column is None:
        return None, None
    y_values = columns.get(response_column, [])
    x_values = columns.get(time_column, []) if time_column is not None else list(range(len(y_values)))
    count = min(len(x_values), len(y_values))
    if count < 1:
        return None, response_column
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(x_values[:count], y_values[:count], color="#7c3aed", linewidth=1.8)
    ax.set_xlabel("Time [s]" if time_column is not None else "Sample [-]")
    ax.set_ylabel(normalize_qblade_header(response_column))
    ax.set_title("Bridge Selected QBlade Response")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = plot_dir / "selected_qblade_response.png"
    _save_plot(fig, path)
    plt.close(fig)
    return path, response_column


def _write_bridge_quick_plots(result: QBladeRunResult) -> tuple[dict[str, str], dict[str, object]]:
    artifacts: dict[str, str] = {}
    diagnostics: dict[str, object] = {
        "bridge_quick_plot_count": 0,
        "bridge_quick_plot_errors": [],
        "bridge_quick_plot_force_history_column": None,
        "bridge_quick_plot_qblade_response_channel": None,
    }
    plot_dir = result.summary_path.with_name(f"{result.summary_path.stem}_quick_plots")
    rows = _source_coverage_rows(result.summary)
    try:
        plt = _load_pyplot()
    except Exception as exc:
        diagnostics["bridge_quick_plot_errors"] = [f"matplotlib unavailable: {exc}"]
        return artifacts, diagnostics

    plot_specs = [
        ("bridge_quick_plot_max_force", lambda: _write_source_force_plot(rows, plot_dir, plt)),
        ("bridge_quick_plot_aoa_re", lambda: _write_aoa_re_plot(rows, plot_dir, plt)),
        ("bridge_quick_plot_nyquist", lambda: _write_nyquist_plot(rows, plot_dir, plt)),
    ]
    errors: list[str] = []
    for key, writer in plot_specs:
        try:
            path = writer()
        except Exception as exc:
            errors.append(f"{key}: {exc}")
            continue
        if path is not None:
            artifacts[key] = str(path)

    try:
        path, column = _write_force_history_plot(result.summary, plot_dir, plt)
        diagnostics["bridge_quick_plot_force_history_column"] = column
        if path is not None:
            artifacts["bridge_quick_plot_force_history"] = str(path)
    except Exception as exc:
        errors.append(f"bridge_quick_plot_force_history: {exc}")

    try:
        path, channel = _write_qblade_response_plot(result, plot_dir, plt)
        diagnostics["bridge_quick_plot_qblade_response_channel"] = channel
        if path is not None:
            artifacts["bridge_quick_plot_qblade_response"] = str(path)
    except Exception as exc:
        errors.append(f"bridge_quick_plot_qblade_response: {exc}")

    if artifacts:
        artifacts["bridge_quick_plot_dir"] = str(plot_dir)
    diagnostics["bridge_quick_plot_count"] = len([key for key in artifacts if key != "bridge_quick_plot_dir"])
    diagnostics["bridge_quick_plot_errors"] = errors
    return artifacts, diagnostics


def _sign_probe_row(probe: dict[str, object], result: QBladeRunResult) -> dict[str, object]:
    summary = result.summary
    row: dict[str, object] = {column: "" for column in SIGN_PROBE_REPORT_COLUMNS}
    row.update(
        {
            "node": probe["node"],
            "axis": probe["axis"],
            "sign": probe["sign"],
            "force_n": probe["force_n"],
            "local": str(bool(probe["local"])).lower(),
            "output_dir": str(probe["output_dir"]),
            "summary_file": str(result.summary_path),
            "result_file": str(result.result_path),
            "requested_steps": summary.get("requested_steps"),
            "completed_steps": summary.get("completed_steps"),
            "failed": str(bool(summary.get("failed"))).lower(),
            "exported_results": str(bool(summary.get("exported_results"))).lower(),
            "passed": "false",
        }
    )
    observations = summary.get("force_sign_probe_observations")
    observation = observations[0] if isinstance(observations, list) and observations else {}
    if isinstance(observation, dict):
        row["observation_status"] = observation.get("status", "")
        row["observed_position_channel"] = observation.get("observed_position_channel", "")
        row["delta_position_m"] = observation.get("delta_position_m", "")
    if summary.get("coupling_mode") != "sil-constant-force":
        row["issue"] = "wrong_coupling_mode"
    elif bool(summary.get("failed")):
        row["issue"] = "failed_run"
    elif not bool(summary.get("exported_results")):
        row["issue"] = "results_not_exported"
    elif int(summary.get("completed_steps", 0) or 0) < 1:
        row["issue"] = "no_completed_steps"
    elif not row["observed_position_channel"]:
        row["issue"] = "missing_position_channel"
    elif row["delta_position_m"] == "":
        row["issue"] = "missing_delta_position"
    else:
        row["passed"] = "true"
    return row


def _row_float(row: dict[str, object], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _apply_sign_probe_pair_checks(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, object]]] = {}
    for row in rows:
        key = (str(row["node"]), str(row["axis"]), str(row["local"]))
        grouped.setdefault(key, {})[str(row["sign"])] = row

    for pair in grouped.values():
        positive = pair.get("positive")
        negative = pair.get("negative")
        if positive is None or negative is None:
            for row in pair.values():
                row["paired_passed"] = "false"
                row["passed"] = "false"
                row["issue"] = "missing_positive_or_negative_pair"
            continue
        response_delta = _row_float(positive, "delta_position_m") - _row_float(negative, "delta_position_m")
        passed = math.isfinite(response_delta) and response_delta > 1.0e-9
        for row in (positive, negative):
            row["paired_response_delta_m"] = response_delta
            row["paired_passed"] = str(passed).lower()
            if not passed:
                row["passed"] = "false"
                row["issue"] = "paired_response_not_positive"


def _write_sign_probe_report(rows: list[dict[str, object]], output_root: Path) -> dict[str, object]:
    _apply_sign_probe_pair_checks(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "force_sign_probe_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIGN_PROBE_REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if str(row.get("passed", "")).lower() != "true"]
    report = {
        "passed": not failures,
        "rows": len(rows),
        "failures": len(failures),
        "report_csv": str(csv_path),
        "probes": rows,
    }
    _write_json(output_root / "force_sign_probe_report.json", report)
    return report


class QBladeBridgeService:
    """Orchestrate bridge validation, preparation, runs, and GUI launch plans."""

    def __init__(
        self,
        *,
        case_preparer: Callable[[QBladeCasePrepRequest], QBladeCasePrepResult] = prepare_qblade_external_case,
        runner: Callable[[QBladeRunRequest], QBladeRunResult] = run_qblade_case,
        batch_setup: Callable[[Any], dict[str, object]] = setup_qblade_batch,
        bridge_builder: Callable[[Any], None] = build_qblade_bridge,
        launcher: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.case_preparer = case_preparer
        self.runner = runner
        self.batch_setup = batch_setup
        self.bridge_builder = bridge_builder
        self.launcher = launcher

    def _run_with_qblade_log_tail(
        self,
        request: QBladeRunRequest,
        message_callback: Callable[[str], None] | None,
    ) -> tuple[QBladeRunResult, dict[str, object]]:
        tailer = _QBladeLogTailer(_expected_qblade_run_log_path(request), message_callback)
        tailer.start()
        try:
            result = self.runner(request)
        finally:
            tailer.stop()
        return result, tailer.diagnostics()

    def _run_qblade_subprocess(
        self,
        config: QBladeBridgeConfig,
        *,
        message_callback: Callable[[str], None] | None,
        stop_requested: Callable[[], bool] | None,
    ) -> QBladeRunResult:
        request = config.to_run_request()
        python = Path(str(config.python_executable or sys.executable)).expanduser()
        runner = _default_vorlap_root() / "run_headless" / "run_qblade_sim.py"
        if not runner.is_file():
            raise FileNotFoundError(f"QBlade runner script does not exist: {runner}")
        command = [str(python), str(runner), *config.to_cli_args("run")]
        env = os.environ.copy()
        prepend_env_path(env, "PYTHONPATH", [_default_vorlap_root()])
        if message_callback is not None:
            message_callback(f"Launching isolated QBlade SIL runner: {' '.join(command)}")
        process = launch_isolated_process(
            command,
            cwd=_default_vorlap_root(),
            env=env,
            launcher=self.launcher,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines: list[str] = []

        def read_output() -> None:
            stream = getattr(process, "stdout", None)
            if stream is None:
                return
            try:
                for line in stream:
                    clean = str(line).rstrip()
                    if clean:
                        output_lines.append(clean)
                        if message_callback is not None:
                            message_callback(clean)
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        stopped = False
        while True:
            poll = getattr(process, "poll", None)
            returncode = poll() if poll is not None else None
            if returncode is not None:
                break
            if stop_requested is not None and stop_requested():
                stopped = True
                terminate_process_tree(process)
                break
            time.sleep(0.1)
        wait = getattr(process, "wait", None)
        returncode = wait() if wait is not None else 0
        reader.join(timeout=2.0)
        if stopped:
            raise RuntimeError("QBlade SIL subprocess stopped by user request.")
        if returncode:
            tail = "\n".join(output_lines[-20:])
            detail = f"\n{tail}" if tail else ""
            raise RuntimeError(f"QBlade SIL subprocess failed with exit status {returncode}.{detail}")
        return _qblade_run_result_from_summary(_expected_qblade_summary_path(request))

    def validate_config(
        self,
        config: QBladeBridgeConfig,
        *,
        check_paths: bool = False,
        check_qblade_install: bool = False,
        check_qblade_model: bool = False,
    ) -> QBladeBridgeValidationResult:
        errors = list(config.validation_errors(check_paths=check_paths))
        if check_qblade_install and not any("QBlade root" in error for error in errors):
            try:
                validate_qblade_root(Path(config.qblade_root))
            except Exception as exc:
                errors.append(str(exc))
        if check_qblade_model and not any("QBlade simulation file" in error for error in errors):
            model_validation = validate_qblade_model_references(
                config.sim_path,
                include_tower_nodes=config.include_tower_nodes,
            )
            errors.extend(model_validation.errors)
            if not model_validation.errors and not any("VorLap airfoil directory" in error for error in errors):
                try:
                    structural_node_ids = model_validation.structural_node_ids if config.node_source == "structural" else ()
                    include_struts, include_tower = infer_geometry_inclusions(
                        structural_node_ids,
                        config.include_struts,
                        config.include_tower,
                    )
                    components, _viv_params, _node_ids = convert_qblade_to_vorlap_inputs(
                        str(config.sim_path),
                        default_airfoil_id=config.default_airfoil_id,
                        include_struts=include_struts,
                        include_tower=include_tower,
                        tower_airfoil_id=config.tower_airfoil_id,
                    )
                    validate_component_airfoils(components, config.airfoil_dir)
                except Exception as exc:
                    errors.append(str(exc))
        executable = Path(config.qblade_binary).resolve() if config.qblade_binary else find_qblade_gui_executable(config.qblade_root)
        return QBladeBridgeValidationResult(
            config=config,
            errors=tuple(errors),
            qblade_gui_executable=executable,
        )

    def prepare_case(
        self,
        config: QBladeBridgeConfig,
        *,
        dry_run: bool = False,
        check_paths: bool = False,
    ) -> QBladeBridgePrepareResult:
        self._raise_for_validation(config, check_paths=check_paths, check_qblade_model=check_paths)
        materialized = (
            None
            if dry_run or not check_paths or config.patch_original_model
            else self._materialize_model_overrides(config, force_copy=True)
        )
        active_config = config if materialized is None else materialized.config
        stable_support_manifest = None
        if not dry_run and check_paths:
            stable_support_manifest = _maybe_apply_reference_stable_support_patch(active_config)
        request = active_config.to_case_prep_request()
        if dry_run:
            request = replace(request, dry_run=True)
        result = self.case_preparer(request)
        if materialized is not None:
            manifest = materialized.artifacts.get("working_copy_manifest")
            if manifest is not None:
                _finalize_working_copy_manifest(Path(str(manifest)), result)
        artifacts = _prep_artifacts(result, materialized)
        diagnostics = _prep_diagnostics(result, materialized)
        if stable_support_manifest is not None:
            artifacts["stable_support_patch"] = str(stable_support_manifest)
            diagnostics["stable_support_patch"] = True
            diagnostics["stable_support_patch_manifest"] = str(stable_support_manifest)
        return QBladeBridgePrepareResult(
            config=active_config,
            prep_result=result,
            state="planned" if result.dry_run else "prepared",
            artifacts=artifacts,
            diagnostics=diagnostics,
        )

    def run_case(
        self,
        config: QBladeBridgeConfig,
        *,
        message_callback: Callable[[str], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        check_paths: bool = False,
    ) -> QBladeBridgeRunServiceResult:
        self._raise_for_validation(config, check_paths=check_paths, check_qblade_model=check_paths)
        materialized = self._materialize_model_overrides(config) if check_paths else None
        active_config = config if materialized is None else materialized.config
        prep_result = None
        stable_support_manifest = None
        if check_paths and active_config.coupling_mode in ("sil-vorlap", "qblade-external"):
            stable_support_manifest = _maybe_apply_reference_stable_support_patch(active_config)
            prep_result = self.case_preparer(active_config.to_case_prep_request())
        request = replace(
            active_config.to_run_request(),
            message_callback=message_callback,
            stop_requested=stop_requested,
        )
        result, log_tail_diagnostics = self._run_with_qblade_log_tail(request, message_callback)
        stopped = bool(result.summary.get("stopped"))
        failed = bool(result.summary.get("failed"))
        geometry_diagnostics = write_run_geometry_audit_reported_positions(result)
        quick_plot_artifacts, quick_plot_diagnostics = _write_bridge_quick_plots(result)
        artifacts = _run_artifacts(result, materialized, quick_plots=quick_plot_artifacts)
        diagnostics = _run_diagnostics(
            result,
            materialized,
            quick_plot_diagnostics=quick_plot_diagnostics,
            geometry_diagnostics=geometry_diagnostics,
        )
        if prep_result is not None:
            artifacts["runtime_config"] = str(prep_result.parameter_file_path)
            if prep_result.geometry_audit_path is not None:
                artifacts["geometry_audit"] = str(prep_result.geometry_audit_path)
            diagnostics["prepared_before_run"] = True
            diagnostics["prepared_node_count"] = len(prep_result.node_ids)
        if stable_support_manifest is not None:
            artifacts["stable_support_patch"] = str(stable_support_manifest)
            diagnostics["stable_support_patch"] = True
            diagnostics["stable_support_patch_manifest"] = str(stable_support_manifest)
        diagnostics.update(log_tail_diagnostics)
        return QBladeBridgeRunServiceResult(
            config=active_config,
            run_result=result,
            state="stopped" if stopped else ("failed" if failed else "complete"),
            artifacts=artifacts,
            diagnostics=diagnostics,
        )

    def run_case_isolated(
        self,
        config: QBladeBridgeConfig,
        *,
        message_callback: Callable[[str], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        check_paths: bool = False,
    ) -> QBladeBridgeRunServiceResult:
        """Run QBlade in a child Python process so GUI state cannot be reset by QBlade/Qt."""
        self._raise_for_validation(config, check_paths=check_paths, check_qblade_model=check_paths)
        materialized = self._materialize_model_overrides(config) if check_paths else None
        active_config = config if materialized is None else materialized.config
        prep_result = None
        stable_support_manifest = None
        if check_paths and active_config.coupling_mode in ("sil-vorlap", "qblade-external"):
            stable_support_manifest = _maybe_apply_reference_stable_support_patch(active_config)
            prep_result = self.case_preparer(active_config.to_case_prep_request())
        result = self._run_qblade_subprocess(
            active_config,
            message_callback=message_callback,
            stop_requested=stop_requested,
        )
        stopped = bool(result.summary.get("stopped"))
        failed = bool(result.summary.get("failed"))
        geometry_diagnostics = write_run_geometry_audit_reported_positions(result)
        quick_plot_artifacts, quick_plot_diagnostics = _write_bridge_quick_plots(result)
        artifacts = _run_artifacts(result, materialized, quick_plots=quick_plot_artifacts)
        diagnostics = _run_diagnostics(
            result,
            materialized,
            quick_plot_diagnostics=quick_plot_diagnostics,
            geometry_diagnostics=geometry_diagnostics,
        )
        if prep_result is not None:
            artifacts["runtime_config"] = str(prep_result.parameter_file_path)
            if prep_result.geometry_audit_path is not None:
                artifacts["geometry_audit"] = str(prep_result.geometry_audit_path)
            diagnostics["prepared_before_run"] = True
            diagnostics["prepared_node_count"] = len(prep_result.node_ids)
        if stable_support_manifest is not None:
            artifacts["stable_support_patch"] = str(stable_support_manifest)
            diagnostics["stable_support_patch"] = True
            diagnostics["stable_support_patch_manifest"] = str(stable_support_manifest)
        request = active_config.to_run_request()
        diagnostics.update(
            {
                "qblade_log_tail_path": str(_expected_qblade_run_log_path(request)),
                "qblade_log_tail_line_count": 0,
                "qblade_log_tail_enabled": False,
                "isolated_qblade_process": True,
            }
        )
        return QBladeBridgeRunServiceResult(
            config=active_config,
            run_result=result,
            state="stopped" if stopped else ("failed" if failed else "complete"),
            artifacts=artifacts,
            diagnostics=diagnostics,
        )

    def run_sign_probe(
        self,
        config: QBladeBridgeConfig,
        *,
        node_id: str | None = None,
        axes: str | Sequence[str] = ("X", "Y", "Z"),
        force_n: float = 1000.0,
        local: bool = False,
        steps: int | None = 200,
        message_callback: Callable[[str], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        check_paths: bool = False,
    ) -> QBladeBridgeSignProbeResult:
        self._raise_for_validation(config, check_paths=check_paths, check_qblade_model=check_paths)
        axes_tuple = _parse_sign_probe_axes(axes)
        force = float(force_n)
        if not math.isfinite(force) or force <= 0.0:
            raise ValueError("Sign-probe force_n must be positive and finite.")
        requested_steps = config.effective_steps() if steps is None else int(steps)
        if requested_steps is None:
            requested_steps = 200
        if requested_steps < 1:
            raise ValueError("Sign-probe steps must be positive.")

        materialized = self._materialize_model_overrides(config, force_copy=check_paths) if check_paths else None
        active_config = config if materialized is None else materialized.config
        stable_support_manifest = _maybe_apply_reference_stable_support_patch(active_config) if check_paths else None
        prep_result = self.case_preparer(active_config.to_case_prep_request())
        node = str(node_id).strip() if node_id else _default_sign_probe_node(prep_result.node_ids)
        output_root = Path(active_config.output_dir).expanduser().resolve() / "sign_probes"
        output_root.mkdir(parents=True, exist_ok=True)

        plan: dict[str, object] = {
            "created_unix": time.time(),
            "sim": str(active_config.sim_path),
            "qblade_root": str(active_config.qblade_root),
            "output_root": str(output_root),
            "node": node,
            "axes": list(axes_tuple),
            "force_magnitude_n": force,
            "local": bool(local),
            "steps": requested_steps,
            "probes": [],
        }
        run_results: list[QBladeRunResult] = []
        rows: list[dict[str, object]] = []
        stopped = False
        base_request = active_config.to_run_request()

        for axis in axes_tuple:
            for sign_name, sign_multiplier in (("positive", 1.0), ("negative", -1.0)):
                if stop_requested is not None and stop_requested():
                    stopped = True
                    break
                value = sign_multiplier * force
                local_flag = "local" if local else "global"
                force_spec = f"{node}:{axis}:{value:.9g}:{local_flag}"
                label = _safe_label(f"{node}_{axis}_{sign_name}")
                output_dir = output_root / label
                result_name = f"{label}_qblade_results"
                metadata_path = output_dir / "case_metadata.json"
                probe = {
                    "node": node,
                    "axis": axis,
                    "sign": sign_name,
                    "force_n": value,
                    "local": bool(local),
                    "force_spec": force_spec,
                    "output_dir": str(output_dir),
                    "result_name": result_name,
                    "metadata_file": str(metadata_path),
                }
                plan["probes"].append(probe)
                _write_json(
                    metadata_path,
                    {
                        "force_sign_probe": probe,
                        "source_sim": str(active_config.sim_path),
                    },
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                request = replace(
                    base_request,
                    output_dir=output_dir,
                    result_name=result_name,
                    steps=requested_steps,
                    sil_vorlap=False,
                    sil_constant_force_specs=(force_spec,),
                    coupling_mode="sil-constant-force",
                    case_metadata_path=metadata_path,
                    message_callback=message_callback,
                    stop_requested=stop_requested,
                )
                run_result, _log_tail_diagnostics = self._run_with_qblade_log_tail(request, message_callback)
                run_results.append(run_result)
                rows.append(_sign_probe_row(probe, run_result))
            if stopped:
                break

        plan_path = output_root / "force_sign_probe_plan.json"
        _write_json(plan_path, plan)
        report = _write_sign_probe_report(rows, output_root)
        failed = bool(report["failures"])
        state = "stopped" if stopped else ("failed" if failed else "complete")
        artifacts = {
            "sign_probe_plan": str(plan_path),
            "sign_probe_report_csv": report["report_csv"],
            "sign_probe_report_json": str(output_root / "force_sign_probe_report.json"),
            "sign_probe_output_root": str(output_root),
            "runtime_config": str(prep_result.parameter_file_path),
            "geometry_audit": None if prep_result.geometry_audit_path is None else str(prep_result.geometry_audit_path),
        }
        diagnostics = {
            "node": node,
            "axes": list(axes_tuple),
            "force_n": force,
            "local": bool(local),
            "requested_steps": requested_steps,
            "probes_requested": len(plan["probes"]),
            "probes_completed": len(run_results),
            "failures": int(report["failures"]),
            "passed": bool(report["passed"]),
            "sign_probe_report_rows": [dict(row) for row in report["probes"]],
            "stopped": stopped,
        }
        diagnostics.update(_geometry_audit_diagnostics(prep_result.geometry_audit_path, prep_result.node_ids))
        if materialized is not None:
            artifacts.update(materialized.artifacts)
            diagnostics.update(materialized.diagnostics)
        if stable_support_manifest is not None:
            artifacts["stable_support_patch"] = str(stable_support_manifest)
            diagnostics["stable_support_patch"] = True
            diagnostics["stable_support_patch_manifest"] = str(stable_support_manifest)
        return QBladeBridgeSignProbeResult(
            config=active_config,
            state=state,
            prep_result=prep_result,
            run_results=tuple(run_results),
            artifacts=artifacts,
            diagnostics=diagnostics,
        )

    def setup_batch(self, args: Any) -> QBladeBridgeBatchSetupResult:
        manifest = self.batch_setup(args)
        cases = manifest.get("cases", [])
        case_count = len(cases) if isinstance(cases, list) else 0
        run_root = getattr(args, "run_root", None)
        return QBladeBridgeBatchSetupResult(
            manifest=manifest,
            case_count=case_count,
            artifacts={
                "run_root": None if run_root is None else str(run_root),
                "manifest": None if run_root is None else str(Path(run_root) / "cases.json"),
            },
        )

    def build_bridge(self, config: QBladeBridgeConfig, *, force: bool = False) -> QBladeBridgeBuildResult:
        self._raise_for_validation(config, check_paths=False)
        qblade_root = Path(config.qblade_root).expanduser().resolve()
        run_root = Path(config.output_dir).expanduser().resolve()
        args = type(
            "BridgeBuildArgs",
            (),
            {
                "qblade_root": qblade_root,
                "run_root": run_root,
                "vorlap_root": _default_vorlap_root(),
                "python": str(config.python_executable or sys.executable),
                "skip_bridge_build": False,
                "force_bridge_build": bool(force),
                "skip_vorlap_install": True,
            },
        )()
        self.bridge_builder(args)
        install_dir = qblade_root / "ControllerFiles"
        build_dir = run_root / "build" / "qblade_external_linux"
        bridge_library = find_bridge_library(qblade_root, config.library_stem)
        python_info = _python_runtime_diagnostics(args.python)
        cmake_version, cmake_error = _run_version_command(["cmake", "--version"])
        compiler_version, compiler_error = _compiler_version()
        library_dirs = qblade_library_dirs(qblade_root)
        qblade_library_path = os.pathsep.join(str(path) for path in library_dirs)
        loader_path_key = _platform_loader_path_key()
        bridge_loadable, bridge_load_error = _shared_library_load_diagnostics(bridge_library)
        return QBladeBridgeBuildResult(
            config=config,
            state="built",
            artifacts={
                "controller_files": str(install_dir),
                "build_dir": str(build_dir),
                "bridge_library": None if bridge_library is None else str(bridge_library),
            },
            diagnostics={
                "force": bool(force),
                "python": str(args.python),
                "python_executable": python_info.get("python_executable"),
                "python_version": python_info.get("python_version"),
                "numpy_version": python_info.get("numpy_version"),
                "numpy_error": python_info.get("numpy_error"),
                "python_probe_error": python_info.get("python_probe_error"),
                "cmake_version": cmake_version,
                "cmake_error": cmake_error,
                "compiler": compiler_version,
                "compiler_error": compiler_error,
                "qblade_ld_library_path": qblade_library_path,
                "qblade_loader_path_key": loader_path_key,
                "qblade_loader_path": qblade_library_path,
                "bridge_library_exists": bridge_library is not None,
                "bridge_library_loadable": bridge_loadable,
                "bridge_library_load_error": bridge_load_error,
            },
        )

    def build_qblade_gui_launch_plan(
        self,
        config: QBladeBridgeConfig,
        *,
        run_result: QBladeBridgeRunServiceResult | QBladeRunResult | Mapping[str, object] | None = None,
        extra_args: Sequence[str | Path] = (),
        env: dict[str, str] | None = None,
    ) -> QBladeGuiLaunchPlan:
        self._raise_for_validation(config, check_paths=False)
        executable = Path(config.qblade_binary).resolve() if config.qblade_binary else find_qblade_gui_executable(config.qblade_root)
        if executable is None:
            raise FileNotFoundError(
                "No QBlade GUI executable configured or auto-detected. Set qblade_binary in QBladeBridgeConfig."
            )
        if not executable.is_file():
            raise FileNotFoundError(f"QBlade GUI executable does not exist: {executable}")
        launch_env = os.environ.copy() if env is None else dict(env)
        qblade_root = Path(config.qblade_root).resolve()
        library_dirs = qblade_library_dirs(qblade_root)
        prepend_env_path(launch_env, "LD_LIBRARY_PATH", library_dirs)
        prepend_env_path(launch_env, "DYLD_LIBRARY_PATH", library_dirs)
        prepend_env_path(launch_env, "PATH", library_dirs)
        launch_env["QBLADE_ROOT"] = str(qblade_root)
        launch_env["OMP_NUM_THREADS"] = str(config.threads)
        project_path, project_source = _project_from_run_context(run_result)
        if project_path is None:
            project_path, project_source = _project_from_config_paths(config)
        command_args = [str(project_path)] if project_path is not None else []
        command_args.extend(str(arg) for arg in extra_args)
        return QBladeGuiLaunchPlan(
            executable=executable,
            command=(str(executable), *command_args),
            cwd=qblade_root,
            env=launch_env,
            case_path_hint=Path(config.sim_path).resolve(),
            project_path=project_path,
            project_source=project_source,
        )

    def launch_qblade_gui(
        self,
        config: QBladeBridgeConfig,
        *,
        run_result: QBladeBridgeRunServiceResult | QBladeRunResult | Mapping[str, object] | None = None,
        extra_args: Sequence[str | Path] = (),
        env: dict[str, str] | None = None,
    ) -> tuple[QBladeGuiLaunchPlan, Any]:
        plan = self.build_qblade_gui_launch_plan(config, run_result=run_result, extra_args=extra_args, env=env)
        process = launch_isolated_process(
            plan.command,
            cwd=str(plan.cwd),
            env=plan.env,
            launcher=self.launcher,
        )
        return plan, process

    def terminate_launched_process(self, process: Any, *, timeout_s: float = 10.0) -> int | None:
        return terminate_process_tree(process, timeout_s=timeout_s)

    def build_artifact_open_plan(self, artifact_path: str | Path) -> QBladeArtifactOpenPlan:
        path_text = str(artifact_path).strip()
        if not path_text:
            raise ValueError("Select an artifact path to open.")
        artifact = Path(path_text).expanduser().resolve()
        if not artifact.exists():
            raise FileNotFoundError(f"Artifact path does not exist: {artifact}")
        if sys.platform == "darwin":
            command = ("open", str(artifact))
        elif os.name == "nt":
            command = ("cmd", "/c", "start", "", str(artifact))
        else:
            command = ("xdg-open", str(artifact))
        return QBladeArtifactOpenPlan(artifact=artifact, command=command)

    def open_artifact(self, artifact_path: str | Path) -> tuple[QBladeArtifactOpenPlan, Any]:
        plan = self.build_artifact_open_plan(artifact_path)
        process = self.launcher(list(plan.command))
        return plan, process

    def _raise_for_validation(
        self,
        config: QBladeBridgeConfig,
        *,
        check_paths: bool,
        check_qblade_model: bool = False,
    ) -> None:
        validation = self.validate_config(
            config,
            check_paths=check_paths,
            check_qblade_model=check_qblade_model,
        )
        if validation.errors:
            raise ValueError("; ".join(validation.errors))

    def _materialize_model_overrides(
        self,
        config: QBladeBridgeConfig,
        *,
        force_copy: bool = False,
    ) -> _MaterializedBridgeConfig | None:
        overrides = _model_override_values(config)
        if not overrides and not force_copy:
            return None

        source_sim = Path(config.sim_path).expanduser().resolve()
        source_dir = source_sim.parent
        output_dir = Path(config.output_dir).expanduser().resolve()
        if _is_relative_to(output_dir, source_dir):
            raise ValueError(
                "QBlade model overrides require output_dir outside the selected model directory "
                "so the generated working copy cannot recurse into the source model."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        working_dir = output_dir / "qblade_model_working"
        if working_dir.exists():
            shutil.rmtree(working_dir)
        shutil.copytree(source_dir, working_dir, symlinks=True)
        working_sim = working_dir / source_sim.name
        if not working_sim.is_file():
            raise FileNotFoundError(f"Copied QBlade simulation file is missing: {working_sim}")

        patches = [
            replace_qblade_keyword_file(working_sim, keyword, value, comment=None)
            for keyword, value in overrides.items()
        ]
        manifest_path = output_dir / "qblade_model_working_manifest.json"
        _write_working_copy_manifest(
            manifest_path=manifest_path,
            source_sim=source_sim,
            working_sim=working_sim,
            patches=patches,
        )
        active_config = replace(config, sim_path=working_sim)
        patch_values = {str(patch["keyword"]): str(patch["new_value"]) for patch in patches}
        return _MaterializedBridgeConfig(
            config=active_config,
            artifacts={
                "source_sim": str(source_sim),
                "working_model_dir": str(working_dir),
                "working_sim": str(working_sim),
                "working_copy_manifest": str(manifest_path),
                "qblade_model_patches": patches,
            },
            diagnostics={
                "qblade_model_override_count": len(patches),
                "qblade_model_overrides": patch_values,
                "qblade_model_override_scope": "generated_working_copy",
                "original_model_preserved": True,
                "working_copy_file_count": len(_working_copy_file_rows(source_dir, working_dir)),
            },
        )


__all__ = [
    "QBladeArtifactOpenPlan",
    "QBladeBridgeBatchSetupResult",
    "QBladeBridgeBuildResult",
    "QBladeBridgePrepareResult",
    "QBladeBridgeRunServiceResult",
    "QBladeBridgeSignProbeResult",
    "QBladeBridgeService",
    "QBladeBridgeValidationResult",
    "QBladeGuiLaunchPlan",
    "find_qblade_gui_executable",
    "write_run_geometry_audit_reported_positions",
]
