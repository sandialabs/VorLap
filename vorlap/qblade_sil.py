"""Reusable QBlade SIL coupling and result-diagnostic helpers."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import struct
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

from .qblade_files import (
    read_qblade_float_keyword,
    read_qblade_keyword,
    resolve_qblade_path,
)
from .qblade_nodes import (
    action_position_label,
    action_target_from_node_id,
    position_channel_name,
)
from .performance_timing import AggregatePerformanceTiming, profiling_requested


SYNTHETIC_SIL_INPUTS = {"Time [s]", "Timestep [-]"}
STRUCTURAL_VELOCITY_RE = re.compile(r"^(?P<axis>[XYZ])_g Vel\. .+ \[m/s\]$")
TIMING_SAMPLE_LIMIT = 20
AXES = {"X", "Y", "Z"}
QBLADE_MISSING_DATA_PROBE = "__VORLAP_INTENTIONALLY_MISSING_CUSTOM_DATA__"
SIL_FORCE_PACKET_SCHEMA = "vorlap-sil-force-packet-v1"
SIL_FORCE_HISTORY_SCHEMA = "vorlap-sil-force-history-v2"
SIL_COMPACT_FORCE_HISTORY_SCHEMA = "vorlap-sil-force-history-compact-v1"
SIL_FORCE_VECTOR_DIGEST_SCHEMA = "vorlap-force-vector-sha256-v1"
SIL_COMPACT_HISTORY_CHECKPOINT_SCHEMA = "vorlap-sil-compact-history-checkpoint-v1"
SIL_COUPLER_CHECKPOINT_SCHEMA = "vorlap-sil-coupler-checkpoint-v2"
SIL_HISTORY_MODES = {"off", "full", "compact"}


def resolve_sil_history_mode(*, record_history: bool, history_mode: str | None) -> str:
    """Resolve the legacy full-history flag and the explicit history mode."""
    if history_mode is None:
        return "full" if record_history else "off"
    mode = str(history_mode).strip().lower()
    if mode not in SIL_HISTORY_MODES:
        raise ValueError(
            "SIL VorLap history mode must be one of off, full, or compact; "
            f"got {history_mode!r}."
        )
    if record_history and mode != "full":
        raise ValueError(
            "The legacy record_history flag selects full history and cannot be combined "
            f"with explicit history mode {mode!r}."
        )
    return mode


@dataclass(frozen=True)
class SilForcePacket:
    """Immutable force vector plus the SIL step and time that generated it."""

    generation_step: int | None
    generation_time_s: float | None
    values: tuple[float, ...]
    startup_zero: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SIL_FORCE_PACKET_SCHEMA,
            "generation_step": self.generation_step,
            "generation_time_s": self.generation_time_s,
            "values": list(self.values),
            "startup_zero": bool(self.startup_zero),
        }


def qblade_action_target_from_node(node_id: str) -> tuple[str, float]:
    """Parse a QBlade node id into the ADDFORCE target id and position."""
    try:
        return action_target_from_node_id(node_id)
    except ValueError as exc:
        raise ValueError(f"Unsupported QBlade node id for force probe: {node_id!r}") from exc


def parse_bool_flag(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"true", "1", "yes", "local"}:
        return True
    if value in {"false", "0", "no", "global"}:
        return False
    raise ValueError(f"Expected true/false/local/global for LOCAL flag, got {text!r}.")


def parse_constant_force_action(spec: str) -> dict[str, object]:
    """Parse NODE:AXIS:VALUE[:LOCAL] into a QBlade setExternalAction request."""
    parts = [part.strip() for part in str(spec).split(":")]
    if len(parts) not in (3, 4):
        raise ValueError(f"Use NODE:AXIS:VALUE[:LOCAL] for --sil-constant-force, got {spec!r}.")
    node_id, axis, value_text = parts[:3]
    axis = axis.upper()
    if axis not in AXES:
        raise ValueError(f"Constant force axis must be one of X, Y, Z; got {axis!r}.")
    value = float(value_text)
    if not math.isfinite(value):
        raise ValueError(f"Constant force value must be finite; got {value_text!r}.")
    local = parse_bool_flag(parts[3]) if len(parts) == 4 else False
    action_id, position = qblade_action_target_from_node(node_id)
    return {
        "source_node_id": node_id,
        "action": "ADDFORCE",
        "id": action_id,
        "position": position,
        "direction": axis,
        "value": value,
        "local": local,
    }


def qblade_node_position_label(action_id: str, position: float) -> str:
    return action_position_label(action_id, position)


def qblade_position_channel_name(action: dict[str, object]) -> str:
    return position_channel_name(action)


def normalize_qblade_header(text: str) -> str:
    return " ".join(str(text).replace("~", " ").split())


def read_qblade_result_numeric_columns(path: Path) -> dict[str, list[float]]:
    """Read a QBlade tab-separated result export into numeric columns."""
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header: list[str] | None = None
    start = 0
    for index, line in enumerate(lines):
        cells = line.rstrip("\n").split("\t")
        if any(
            normalize_qblade_header(cell).startswith("Time ")
            or normalize_qblade_header(cell) == "Time [s]"
            for cell in cells
        ):
            header = cells
            start = index + 1
            break
    if header is None:
        return {}
    columns = {name: [] for name in header}
    for line in lines[start:]:
        if not line.strip():
            continue
        cells = line.rstrip("\n").split("\t")
        if len(cells) != len(header):
            continue
        try:
            parsed = [float(cell) for cell in cells]
        except ValueError:
            continue
        for name, value in zip(header, parsed):
            columns[name].append(value)
    return columns


def qblade_result_time_diagnostics(path: Path) -> dict[str, object]:
    """Summarize exported QBlade result timestamps when a Time column exists."""
    columns = read_qblade_result_numeric_columns(path)
    normalized = {normalize_qblade_header(name): name for name in columns}
    source_column = normalized.get("Time [s]")
    if source_column is None:
        for normalized_name, original_name in normalized.items():
            if normalized_name.startswith("Time "):
                source_column = original_name
                break
    if source_column is None:
        return {"available": False, "column": None, "samples": 0}
    values = [float(value) for value in columns.get(source_column, []) if math.isfinite(value)]
    if not values:
        return {"available": False, "column": source_column, "samples": 0}
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return {
        "available": True,
        "column": source_column,
        "samples": len(values),
        "first_time_s": values[0],
        "last_time_s": values[-1],
        "min_step_s": min(diffs) if diffs else None,
        "max_step_s": max(diffs) if diffs else None,
        "duration_s": values[-1] - values[0],
    }


def qblade_result_rpm_diagnostics(
    path: Path,
    *,
    expected_rpm: float,
    start_time_s: float = 0.1,
) -> dict[str, object]:
    """Summarize the exported low-speed-shaft RPM after a startup exclusion."""
    columns = read_qblade_result_numeric_columns(path)
    normalized = {normalize_qblade_header(name): name for name in columns}
    rpm_column = normalized.get("LSS Rpm [rpm]")
    if rpm_column is None:
        for normalized_name, original_name in normalized.items():
            lowered_name = normalized_name.lower()
            if "lss" in lowered_name and "rpm" in lowered_name:
                rpm_column = original_name
                break
    time_column = normalized.get("Time [s]")
    if rpm_column is None or time_column is None:
        return {
            "available": False,
            "column": rpm_column,
            "time_column": time_column,
            "samples": 0,
            "start_time_s": float(start_time_s),
            "expected_rpm": float(expected_rpm),
        }
    times = columns.get(time_column, [])
    rpm_values = columns.get(rpm_column, [])
    retained = [
        float(rpm)
        for time_value, rpm in zip(times, rpm_values)
        if math.isfinite(float(time_value))
        and math.isfinite(float(rpm))
        and float(time_value) + 1.0e-12 >= float(start_time_s)
    ]
    if not retained:
        return {
            "available": False,
            "column": rpm_column,
            "time_column": time_column,
            "samples": 0,
            "start_time_s": float(start_time_s),
            "expected_rpm": float(expected_rpm),
        }
    deviations = [abs(value - float(expected_rpm)) for value in retained]
    return {
        "available": True,
        "column": rpm_column,
        "time_column": time_column,
        "samples": len(retained),
        "start_time_s": float(start_time_s),
        "expected_rpm": float(expected_rpm),
        "mean_rpm": sum(retained) / len(retained),
        "min_rpm": min(retained),
        "max_rpm": max(retained),
        "max_abs_error_rpm": max(deviations),
    }


def force_sign_probe_observations(
    result_path: Path,
    actions: list[dict[str, object]],
    *,
    tolerance: float = 1.0e-9,
) -> list[dict[str, object]]:
    """Summarize exported position-channel response to constant force probes."""
    columns = read_qblade_result_numeric_columns(result_path)
    normalized = {normalize_qblade_header(name): name for name in columns}
    observations: list[dict[str, object]] = []
    for action in actions:
        expected_channel = qblade_position_channel_name(action)
        source_column = normalized.get(normalize_qblade_header(expected_channel))
        row: dict[str, object] = {
            "source_node_id": action["source_node_id"],
            "action_id": action["id"],
            "position": action["position"],
            "direction": action["direction"],
            "force_n": action["value"],
            "local": action["local"],
            "expected_position_channel": expected_channel,
            "observed_position_channel": source_column,
            "status": "missing_position_channel" if source_column is None else "unknown",
        }
        if source_column is None:
            observations.append(row)
            continue
        values = [value for value in columns.get(source_column, []) if math.isfinite(value)]
        if len(values) < 2:
            row["status"] = "insufficient_samples"
            observations.append(row)
            continue
        initial = float(values[0])
        final = float(values[-1])
        delta = final - initial
        force = float(action["value"])
        row.update(
            {
                "initial_position_m": initial,
                "final_position_m": final,
                "delta_position_m": delta,
                "max_abs_position_change_m": max(abs(float(value) - initial) for value in values),
            }
        )
        if abs(delta) <= tolerance:
            row["status"] = "zero_or_unresolved_response"
        elif force * delta > 0.0:
            row["status"] = "same_sign"
        else:
            row["status"] = "opposite_sign"
        observations.append(row)
    return observations


def find_qblade_external_parameter_file(sim_path: Path) -> Path:
    turbine_file = read_qblade_keyword(sim_path, "TURBFILE")
    if turbine_file is None:
        raise ValueError(f"Could not find TURBFILE in {sim_path}")
    turbine_path = resolve_qblade_path(sim_path, turbine_file)
    if not turbine_path.is_file():
        raise FileNotFoundError(f"QBlade turbine definition not found: {turbine_path}")

    parameter_file = read_qblade_keyword(turbine_path, "LIBPARAMETERFILE_1")
    if parameter_file is None:
        raise ValueError(f"Could not find LIBPARAMETERFILE_1 in {turbine_path}")
    parameter_path = resolve_qblade_path(turbine_path, parameter_file)
    if not parameter_path.is_file():
        raise FileNotFoundError(f"VorLap external config not found: {parameter_path}")
    return parameter_path


class SilTimingDiagnostics:
    """Collect compact SIL timing evidence without storing every timestep."""

    def __init__(self, *, sample_limit: int = TIMING_SAMPLE_LIMIT) -> None:
        self.sample_limit = int(sample_limit)
        self.samples = 0
        self.qblade_time_available_count = 0
        self.qblade_time_missing_count = 0
        self.wrapper_time_min_s: float | None = None
        self.wrapper_time_max_s: float | None = None
        self.applied_force_time_min_s: float | None = None
        self.applied_force_time_max_s: float | None = None
        self.azimuth_min_deg: float | None = None
        self.azimuth_max_deg: float | None = None
        self.time_delta_min_s: float | None = None
        self.time_delta_max_s: float | None = None
        self.time_delta_max_abs_s = 0.0
        self.first_samples: list[dict[str, object]] = []
        self.last_sample: dict[str, object] | None = None

    @staticmethod
    def _update_minmax(current_min: float | None, current_max: float | None, value: float) -> tuple[float, float]:
        if current_min is None or current_max is None:
            return value, value
        return min(current_min, value), max(current_max, value)

    def record(
        self,
        *,
        step_index: int,
        wrapper_time_s: float,
        wrapper_timestep: float,
        qblade_time_pre_advance_s: float | None,
        azimuth_deg: float,
        applied_force_time_s: float | None,
        applied_force_source_step: int | None = None,
        applied_force_startup_zero: bool = False,
        applied_force_lag_steps: int = 0,
    ) -> None:
        row: dict[str, object] = {
            "step_index": int(step_index),
            "wrapper_time_s": float(wrapper_time_s),
            "wrapper_timestep": float(wrapper_timestep),
            "qblade_time_pre_advance_s": qblade_time_pre_advance_s,
            "azimuth_deg": float(azimuth_deg),
            "applied_force_time_s": (
                None if applied_force_time_s is None else float(applied_force_time_s)
            ),
            "applied_force_source_step": (
                None if applied_force_source_step is None else int(applied_force_source_step)
            ),
            "applied_force_startup_zero": bool(applied_force_startup_zero),
            "applied_force_lag_steps": int(applied_force_lag_steps),
            "force_application_order": "before_advance",
        }
        if qblade_time_pre_advance_s is None:
            self.qblade_time_missing_count += 1
            row["time_delta_qblade_minus_wrapper_s"] = None
        else:
            self.qblade_time_available_count += 1
            delta = float(qblade_time_pre_advance_s) - float(wrapper_time_s)
            row["time_delta_qblade_minus_wrapper_s"] = delta
            self.time_delta_min_s, self.time_delta_max_s = self._update_minmax(
                self.time_delta_min_s,
                self.time_delta_max_s,
                delta,
            )
            self.time_delta_max_abs_s = max(self.time_delta_max_abs_s, abs(delta))

        self.samples += 1
        self.wrapper_time_min_s, self.wrapper_time_max_s = self._update_minmax(
            self.wrapper_time_min_s,
            self.wrapper_time_max_s,
            float(wrapper_time_s),
        )
        if applied_force_time_s is not None:
            self.applied_force_time_min_s, self.applied_force_time_max_s = self._update_minmax(
                self.applied_force_time_min_s,
                self.applied_force_time_max_s,
                float(applied_force_time_s),
            )
        self.azimuth_min_deg, self.azimuth_max_deg = self._update_minmax(
            self.azimuth_min_deg,
            self.azimuth_max_deg,
            float(azimuth_deg),
        )
        if len(self.first_samples) < self.sample_limit:
            self.first_samples.append(dict(row))
        self.last_sample = dict(row)

    def as_dict(self) -> dict[str, object]:
        return {
            "time_level": "pre_advance_explicit",
            "samples": self.samples,
            "qblade_time_available_count": self.qblade_time_available_count,
            "qblade_time_missing_count": self.qblade_time_missing_count,
            "wrapper_time_min_s": self.wrapper_time_min_s,
            "wrapper_time_max_s": self.wrapper_time_max_s,
            "applied_force_time_min_s": self.applied_force_time_min_s,
            "applied_force_time_max_s": self.applied_force_time_max_s,
            "azimuth_min_deg": self.azimuth_min_deg,
            "azimuth_max_deg": self.azimuth_max_deg,
            "time_delta_qblade_minus_wrapper_min_s": self.time_delta_min_s,
            "time_delta_qblade_minus_wrapper_max_s": self.time_delta_max_s,
            "time_delta_qblade_minus_wrapper_max_abs_s": self.time_delta_max_abs_s,
            "first_samples": self.first_samples,
            "last_sample": self.last_sample,
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore compact timing diagnostics from checkpoint metadata."""
        self.samples = int(state.get("samples", 0) or 0)
        self.qblade_time_available_count = int(state.get("qblade_time_available_count", 0) or 0)
        self.qblade_time_missing_count = int(state.get("qblade_time_missing_count", 0) or 0)
        self.wrapper_time_min_s = state.get("wrapper_time_min_s")  # type: ignore[assignment]
        self.wrapper_time_max_s = state.get("wrapper_time_max_s")  # type: ignore[assignment]
        self.applied_force_time_min_s = state.get("applied_force_time_min_s")  # type: ignore[assignment]
        self.applied_force_time_max_s = state.get("applied_force_time_max_s")  # type: ignore[assignment]
        self.azimuth_min_deg = state.get("azimuth_min_deg")  # type: ignore[assignment]
        self.azimuth_max_deg = state.get("azimuth_max_deg")  # type: ignore[assignment]
        self.time_delta_min_s = state.get("time_delta_qblade_minus_wrapper_min_s")  # type: ignore[assignment]
        self.time_delta_max_s = state.get("time_delta_qblade_minus_wrapper_max_s")  # type: ignore[assignment]
        self.time_delta_max_abs_s = float(state.get("time_delta_qblade_minus_wrapper_max_abs_s", 0.0) or 0.0)
        first_samples = state.get("first_samples", [])
        self.first_samples = list(first_samples) if isinstance(first_samples, list) else []
        last_sample = state.get("last_sample")
        self.last_sample = dict(last_sample) if isinstance(last_sample, dict) else None


class SilVorLapCoupler:
    """Evaluate VorLap in the Python SIL loop and apply forces through QBlade."""

    def __init__(
        self,
        sim_path: Path,
        *,
        record_history: bool = False,
        history_mode: str | None = None,
        history_path: Path | None = None,
        resume_history: bool = False,
        force_application_lag_steps: int = 1,
    ) -> None:
        import numpy as np
        from .qblade_runtime import VorLapQBladeRuntime

        self.np = np
        self.force_application_lag_steps = int(force_application_lag_steps)
        if self.force_application_lag_steps < 0:
            raise ValueError("force_application_lag_steps must be nonnegative.")
        self.timestep = read_qblade_float_keyword(sim_path, "TIMESTEP")
        self.parameter_file = find_qblade_external_parameter_file(sim_path)
        self.config = json.loads(self.parameter_file.read_text(encoding="utf-8"))
        self.runtime = VorLapQBladeRuntime.from_qblade_config(str(self.parameter_file))
        self.use_qblade_local_inflow = self.runtime.use_qblade_local_inflow
        self.hvawt_response_adapter = None
        raw_hvawt_response = self.config.get("hvawt_response_model")
        raw_single_mode_spectral = self.config.get("single_mode_spectral_lockin")
        raw_single_mode_work_map = self.config.get("single_mode_airfoil_work_map")
        raw_single_mode_engineering = self.config.get(
            "single_mode_engineering_lockin"
        )
        configured_response_adapters = sum(
            value is not None
            for value in (
                raw_hvawt_response,
                raw_single_mode_spectral,
                raw_single_mode_work_map,
                raw_single_mode_engineering,
            )
        )
        if configured_response_adapters > 1:
            raise ValueError(
                "A SIL case may configure exactly one H-VAWT, single-mode "
                "spectral, work-map, or engineering lock-in response adapter."
            )
        if raw_hvawt_response is not None:
            from .qblade_hvawt_sil import HVAWTSilResponseAdapter

            self.hvawt_response_adapter = HVAWTSilResponseAdapter(
                raw_config=raw_hvawt_response,
                parameter_file=self.parameter_file,
                runtime=self.runtime,
                sample_period_s=self.timestep,
            )
        elif raw_single_mode_spectral is not None:
            from .qblade_single_mode_spectral_sil import (
                QBladeSingleModeSpectralSilAdapter,
            )

            self.hvawt_response_adapter = QBladeSingleModeSpectralSilAdapter(
                raw_config=raw_single_mode_spectral,
                parameter_file=self.parameter_file,
                runtime=self.runtime,
            )
        elif raw_single_mode_work_map is not None:
            from .qblade_single_mode_work_map_sil import (
                QBladeSingleModeWorkMapSilAdapter,
            )

            self.hvawt_response_adapter = QBladeSingleModeWorkMapSilAdapter(
                raw_config=raw_single_mode_work_map,
                parameter_file=self.parameter_file,
                runtime=self.runtime,
            )
        elif raw_single_mode_engineering is not None:
            from .qblade_single_mode_engineering_lockin_sil import (
                QBladeSingleModeEngineeringLockInSilAdapter,
            )

            self.hvawt_response_adapter = QBladeSingleModeEngineeringLockInSilAdapter(
                raw_config=raw_single_mode_engineering,
                parameter_file=self.parameter_file,
                runtime=self.runtime,
            )
        self.swap = np.zeros(max(int(self.config.get("swap_size", 0)), self.runtime.swap_size), dtype=float)
        self.input_rows = list(self.config.get("external_input_rows", []))
        self.output_rows = list(self.config.get("external_output_rows", []))
        if not self.input_rows:
            raise ValueError(f"{self.parameter_file} does not define external_input_rows.")
        if not self.output_rows:
            raise ValueError(f"{self.parameter_file} does not define external_output_rows.")
        self._configure_input_diagnostics()
        self._validate_runtime_input_storage(sim_path)
        self._timing = SilTimingDiagnostics()
        self._performance_timing = (
            AggregatePerformanceTiming() if profiling_requested() else None
        )
        if self.hvawt_response_adapter is not None:
            self.hvawt_response_adapter.performance_timing = self._performance_timing
        layout = self.runtime.layout
        self._time_offset = layout[str(self.config.get("time_block", "time"))].offset
        self._timestep_offset = layout[str(self.config.get("timestep_block", "timestep"))].offset
        self._azimuth_offset = layout[str(self.config.get("azimuth_block", "azimuth_deg"))].offset
        self._velocity_spec = layout[str(self.config.get("velocity_block", "velocity"))]
        self._force_spec = layout[str(self.config.get("force_block", "force"))]
        self._force_application_queue = [
            self._startup_zero_force_packet() for _ in range(self.force_application_lag_steps)
        ]
        self.updates = 0
        self.max_abs_force = 0.0
        self.last_message = self.runtime.message
        self.history_mode = resolve_sil_history_mode(
            record_history=bool(record_history),
            history_mode=history_mode,
        )
        self.record_history = self.history_mode != "off"
        if self.record_history and any(bool(row.get("local", False)) for row in self.output_rows):
            raise ValueError(
                "SIL applied-force work history requires global force actions because configured "
                "structural velocities are in the global frame."
            )
        self.node_ids = [str(node_id) for node_id in self.config.get("node_ids", [])]
        if self.use_qblade_local_inflow:
            self._configure_local_inflow_positions()
        self.history_rows: list[dict[str, object]] = []
        self._initialize_compact_history_state()
        if history_path is not None:
            self.configure_history_output(history_path, resume=resume_history)

    def _configure_local_inflow_positions(self) -> None:
        """Require current global positions in the same order as nodal forces."""
        if self.hvawt_response_adapter is not None and tuple(self.node_ids) != tuple(
            self.runtime.controller.node_ids
        ):
            raise ValueError(
                "Local inflow with a response adapter requires external node_ids "
                "in controller/modal order; regenerate the external tables."
            )
        spec = self.runtime.layout.get("position_global_m")
        if spec is None or spec.shape != (len(self.node_ids), 3):
            raise ValueError(
                "use_qblade_local_inflow requires a position_global_m block "
                "with one global XYZ position per node; regenerate the external tables."
            )
        rows_by_index = {int(row["index"]): str(row["variable"]) for row in self.input_rows}
        for inode, node_id in enumerate(self.node_ids):
            target, position = qblade_action_target_from_node(node_id)
            for iaxis, axis in enumerate(("X", "Y", "Z")):
                expected = position_channel_name(
                    {"id": target, "position": position, "direction": axis}
                )
                if rows_by_index.get(spec.offset + 3 * inode + iaxis) != expected:
                    raise ValueError(
                        "use_qblade_local_inflow requires current nodal positions "
                        f"in node_ids order; missing or incorrect input '{expected}'."
                    )
        self._local_inflow_position_spec = spec

    def _initialize_compact_history_state(self) -> None:
        self._compact_history_path: Path | None = None
        self._compact_history_resume = False
        self._compact_history_restored = False
        self._compact_history_handle = None
        self._compact_history_columns: tuple[str, ...] = ()
        self._compact_history_rows_written = 0
        self._compact_history_size_bytes = 0
        self._compact_history_hasher = hashlib.sha256()
        self._compact_generated_digest_window: list[dict[str, object]] = []
        self._compact_history_discarded_tail_bytes = 0

    def configure_history_output(self, path: Path, *, resume: bool = False) -> None:
        """Bind compact history to one immutable output path before stepping."""
        if self._configured_history_mode() != "compact":
            raise ValueError("A streaming history output path is only valid in compact mode.")
        if self.updates or self._compact_history_rows_written or self._compact_history_handle is not None:
            raise RuntimeError("Compact SIL history output must be configured before stepping.")
        resolved = Path(path).expanduser().resolve()
        if self._compact_history_path is not None and resolved != self._compact_history_path:
            raise RuntimeError(
                "Compact SIL history output is already bound to a different path: "
                f"{self._compact_history_path}."
            )
        if not resume and resolved.exists():
            raise FileExistsError(
                "Refusing to overwrite an existing compact SIL history outside the run-output "
                f"lifecycle: {resolved}"
            )
        self._compact_history_path = resolved
        self._compact_history_resume = bool(resume)

    @staticmethod
    def _qblade_value(qblade, variable: str) -> float:
        # QBlade's documented SIL examples pass a finite normalized position.
        # These configured names already identify an exact stored channel, so
        # position zero is ignored for scalar/exact-name retrieval.
        return float(qblade.getCustomData_at_num(variable.encode("utf-8"), 0.0, 0))

    @classmethod
    def _optional_qblade_value(cls, qblade, variable: str) -> float | None:
        value = cls._qblade_value(qblade, variable)
        return value if math.isfinite(value) else None

    @staticmethod
    def _new_velocity_stat() -> dict[str, float | int | bool | None]:
        return {
            "samples": 0,
            "min": None,
            "max": None,
            "max_abs": 0.0,
            "all_zero": True,
        }

    @staticmethod
    def _update_velocity_stat(stat: dict[str, float | int | bool | None], value: float) -> None:
        samples = int(stat["samples"])
        stat["samples"] = samples + 1
        stat["min"] = value if samples == 0 else min(float(stat["min"]), value)
        stat["max"] = value if samples == 0 else max(float(stat["max"]), value)
        stat["max_abs"] = max(float(stat["max_abs"]), abs(value))
        stat["all_zero"] = bool(stat["all_zero"]) and abs(value) <= 1.0e-12

    def _configure_input_diagnostics(self) -> None:
        self.qblade_input_variables = sorted(
            {
                str(row["variable"])
                for row in self.input_rows
                if str(row["variable"]) not in SYNTHETIC_SIL_INPUTS
            }
        )
        self.velocity_axis_by_variable: dict[str, str] = {}
        self.velocity_axis_stats = {axis: self._new_velocity_stat() for axis in ("X", "Y", "Z")}
        self.velocity_channel_stats: dict[str, dict[str, float | int | bool | None]] = {}
        for variable in self.qblade_input_variables:
            match = STRUCTURAL_VELOCITY_RE.match(variable)
            if not match:
                continue
            self.velocity_axis_by_variable[variable] = match.group("axis")
            self.velocity_channel_stats[variable] = self._new_velocity_stat()

    @staticmethod
    def _validated_velocity_stat(
        raw: object,
        *,
        expected_samples: int,
        label: str,
    ) -> dict[str, float | int | bool | None]:
        if not isinstance(raw, dict) or set(raw) != {
            "samples", "min", "max", "max_abs", "all_zero"
        }:
            raise ValueError(f"Corrupt SIL checkpoint: {label} keys differ.")
        if type(raw["samples"]) is not int or raw["samples"] != expected_samples:
            raise ValueError(f"Corrupt SIL checkpoint: {label} sample count differs.")
        if not isinstance(raw["all_zero"], bool):
            raise ValueError(f"Corrupt SIL checkpoint: {label} all_zero is invalid.")
        maximum_abs = raw["max_abs"]
        if (
            isinstance(maximum_abs, bool)
            or not isinstance(maximum_abs, (int, float))
            or not math.isfinite(float(maximum_abs))
            or float(maximum_abs) < 0.0
        ):
            raise ValueError(f"Corrupt SIL checkpoint: {label} max_abs is invalid.")
        if expected_samples == 0:
            if raw["min"] is not None or raw["max"] is not None or float(maximum_abs) != 0.0 or raw["all_zero"] is not True:
                raise ValueError(f"Corrupt SIL checkpoint: empty {label} is invalid.")
            return dict(raw)
        extrema = (raw["min"], raw["max"])
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in extrema
        ):
            raise ValueError(f"Corrupt SIL checkpoint: {label} extrema are invalid.")
        minimum, maximum = (float(value) for value in extrema)
        expected_max_abs = max(abs(minimum), abs(maximum))
        if minimum > maximum or not math.isclose(
            float(maximum_abs),
            expected_max_abs,
            rel_tol=0.0,
            abs_tol=64.0 * math.ulp(max(1.0, expected_max_abs)),
        ):
            raise ValueError(f"Corrupt SIL checkpoint: {label} extrema disagree.")
        if raw["all_zero"] is not (float(maximum_abs) <= 1.0e-12):
            raise ValueError(f"Corrupt SIL checkpoint: {label} zero flag disagrees.")
        return dict(raw)

    def _prepare_velocity_restore(
        self,
        raw: object,
        *,
        updates: int,
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        if not isinstance(raw, dict) or set(raw) != {
            "velocity_channel_count",
            "axis_stats",
            "channel_stats",
            "all_zero_axes",
            "all_zero_channels",
        }:
            raise ValueError("Corrupt SIL checkpoint: velocity diagnostics keys differ.")
        if raw["velocity_channel_count"] != len(self.velocity_channel_stats):
            raise ValueError("Corrupt SIL checkpoint: velocity channel count differs.")
        axis_stats = raw["axis_stats"]
        channel_stats = raw["channel_stats"]
        if not isinstance(axis_stats, dict) or set(axis_stats) != {"X", "Y", "Z"}:
            raise ValueError("Corrupt SIL checkpoint: velocity axes differ.")
        if not isinstance(channel_stats, dict) or set(channel_stats) != set(self.velocity_channel_stats):
            raise ValueError("Corrupt SIL checkpoint: velocity channels differ.")
        restored_channels = {
            name: self._validated_velocity_stat(
                channel_stats[name], expected_samples=updates, label=f"velocity channel {name}"
            )
            for name in self.velocity_channel_stats
        }
        restored_axes: dict[str, dict[str, object]] = {}
        for axis in ("X", "Y", "Z"):
            axis_channel_count = sum(
                value == axis for value in self.velocity_axis_by_variable.values()
            )
            restored_axes[axis] = self._validated_velocity_stat(
                axis_stats[axis],
                expected_samples=updates * axis_channel_count,
                label=f"velocity axis {axis}",
            )
        expected_zero_channels = sorted(
            name for name, stat in restored_channels.items() if stat["all_zero"]
        )
        expected_zero_axes = [
            axis for axis in ("X", "Y", "Z") if restored_axes[axis]["all_zero"]
        ]
        if raw["all_zero_channels"] != expected_zero_channels or raw["all_zero_axes"] != expected_zero_axes:
            raise ValueError("Corrupt SIL checkpoint: velocity zero-channel summary differs.")
        return restored_axes, restored_channels

    def _validate_runtime_input_storage(self, sim_path: Path) -> None:
        """Require FILTERFILE to retain every QBlade channel consumed by SIL."""
        filter_value = read_qblade_keyword(sim_path, "FILTERFILE")
        if filter_value is None or not filter_value.strip():
            return
        filter_path = resolve_qblade_path(sim_path, filter_value)
        if not filter_path.is_file():
            raise FileNotFoundError(f"QBlade FILTERFILE does not exist: {filter_path}")
        channels = {
            line.strip()
            for line in filter_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "!"))
        }
        missing = [variable for variable in self.qblade_input_variables if variable not in channels]
        if missing:
            preview = "; ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f"; ... ({len(missing)} total)"
            raise RuntimeError(
                "QBlade FILTERFILE omits SIL-VorLap runtime input channel(s): "
                f"{preview}{suffix}. Use the generated SIL storage filter for FILTERFILE and "
                "reserve the compact fatigue filter for exportResults()."
            )

    def validate_channels(self, qblade) -> None:
        """Fail before advancement if a configured QBlade SIL channel is unavailable."""
        missing_or_nonfinite = []
        observed: dict[str, float] = {}
        for variable in self.qblade_input_variables:
            value = self._qblade_value(qblade, variable)
            observed[variable] = value
            if not math.isfinite(value):
                missing_or_nonfinite.append(variable)
        if missing_or_nonfinite:
            preview = "; ".join(missing_or_nonfinite[:8])
            suffix = "" if len(missing_or_nonfinite) <= 8 else f"; ... ({len(missing_or_nonfinite)} total)"
            raise RuntimeError(
                "Missing or non-finite QBlade SIL input channel(s): "
                f"{preview}{suffix}. A misspelled or unavailable channel must be fixed before running."
            )
        missing_sentinel = self._qblade_value(qblade, QBLADE_MISSING_DATA_PROBE)
        if observed and math.isfinite(missing_sentinel) and all(
            value == missing_sentinel for value in observed.values()
        ):
            raise RuntimeError(
                "QBlade returned its missing-custom-data sentinel "
                f"{missing_sentinel:g} for all {len(observed)} configured SIL input channels. "
                "Check FILTERFILE coverage and the generated external-input names."
            )

    def _record_velocity_input(self, variable: str, value: float) -> None:
        axis = self.velocity_axis_by_variable.get(variable)
        if axis is None:
            return
        self._update_velocity_stat(self.velocity_axis_stats[axis], value)
        self._update_velocity_stat(self.velocity_channel_stats[variable], value)

    def velocity_diagnostics(self) -> dict[str, object]:
        all_zero_channels = [
            variable
            for variable, stat in sorted(self.velocity_channel_stats.items())
            if int(stat["samples"]) == 0 or bool(stat["all_zero"])
        ]
        all_zero_axes = [
            axis
            for axis, stat in self.velocity_axis_stats.items()
            if int(stat["samples"]) == 0 or bool(stat["all_zero"])
        ]
        return {
            "velocity_channel_count": len(self.velocity_channel_stats),
            "axis_stats": {
                axis: dict(stat) for axis, stat in self.velocity_axis_stats.items()
            },
            "channel_stats": {
                variable: dict(stat)
                for variable, stat in sorted(self.velocity_channel_stats.items())
            },
            "all_zero_axes": all_zero_axes,
            "all_zero_channels": all_zero_channels,
        }

    def timing_diagnostics(self) -> dict[str, object]:
        return self._timing.as_dict()

    def performance_timing(self) -> dict[str, object] | None:
        timing = getattr(self, "_performance_timing", None)
        if timing is None:
            return None
        return timing.as_dict()

    def attach_performance_timing(self, timing: AggregatePerformanceTiming) -> None:
        """Attach the runner-owned timer whose origin precedes construction."""
        self._performance_timing = timing
        if self.hvawt_response_adapter is not None:
            self.hvawt_response_adapter.performance_timing = timing

    def record_performance_stage(self, stage: str, elapsed_ns: int) -> None:
        timing = getattr(self, "_performance_timing", None)
        if timing is not None:
            timing.record_ns(stage, elapsed_ns)

    def source_coverage(self) -> list[dict[str, object]]:
        adapter = getattr(self, "hvawt_response_adapter", None)
        if adapter is not None:
            coverage = getattr(adapter, "source_coverage", None)
            return [] if not callable(coverage) else list(coverage())
        return self.runtime.source_coverage_summary(self.timestep)

    def _startup_zero_force_packet(self) -> SilForcePacket:
        return SilForcePacket(
            generation_step=None,
            generation_time_s=None,
            values=tuple(0.0 for _ in range(self._force_spec.size)),
            startup_zero=True,
        )

    def _generated_force_packet(
        self,
        force_values,
        *,
        generation_step: int,
        generation_time_s: float,
    ) -> SilForcePacket:
        values = self.np.asarray(force_values, dtype=float).reshape(-1)
        if values.size != self._force_spec.size:
            raise RuntimeError(
                f"Generated force vector has {values.size} components; expected {self._force_spec.size}."
            )
        if not bool(self.np.all(self.np.isfinite(values))):
            raise RuntimeError("VorLap generated a non-finite force vector.")
        if isinstance(generation_step, bool) or int(generation_step) < 0:
            raise RuntimeError(f"Invalid force generation step: {generation_step!r}.")
        generation_time_s = float(generation_time_s)
        if not math.isfinite(generation_time_s):
            raise RuntimeError(f"Invalid force generation time: {generation_time_s!r}.")
        return SilForcePacket(
            generation_step=int(generation_step),
            generation_time_s=generation_time_s,
            values=tuple(float(value) for value in values),
            startup_zero=False,
        )

    def _packet_from_checkpoint(self, raw_packet: object, *, queue_index: int) -> SilForcePacket:
        if not isinstance(raw_packet, dict):
            raise ValueError(
                "Corrupt SIL force queue: every entry must be a force-packet object; "
                f"entry {queue_index} is {type(raw_packet).__name__}."
            )
        if raw_packet.get("schema") != SIL_FORCE_PACKET_SCHEMA:
            raise ValueError(
                "Corrupt or legacy SIL force queue: packet schema must be "
                f"{SIL_FORCE_PACKET_SCHEMA!r} at entry {queue_index}."
            )
        startup_zero = raw_packet.get("startup_zero")
        if not isinstance(startup_zero, bool):
            raise ValueError(
                f"Corrupt SIL force queue: startup_zero must be boolean at entry {queue_index}."
            )
        raw_values = raw_packet.get("values")
        if not isinstance(raw_values, list) or len(raw_values) != self._force_spec.size:
            raise ValueError(
                "Corrupt SIL force queue: packet vector length at entry "
                f"{queue_index} must be {self._force_spec.size}."
            )
        try:
            values_array = self.np.asarray(raw_values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Corrupt SIL force queue: nonnumeric packet vector at entry {queue_index}."
            ) from exc
        if not bool(self.np.all(self.np.isfinite(values_array))):
            raise ValueError(
                f"Corrupt SIL force queue: non-finite packet vector at entry {queue_index}."
            )

        generation_step = raw_packet.get("generation_step")
        generation_time_s = raw_packet.get("generation_time_s")
        if startup_zero:
            if generation_step is not None or generation_time_s is not None:
                raise ValueError(
                    "Corrupt SIL force queue: startup-zero packets must have null generation "
                    f"step/time at entry {queue_index}."
                )
            if bool(self.np.any(values_array != 0.0)):
                raise ValueError(
                    f"Corrupt SIL force queue: startup-zero packet {queue_index} is not exactly zero."
                )
            return self._startup_zero_force_packet()

        if isinstance(generation_step, bool) or not isinstance(generation_step, int) or generation_step < 0:
            raise ValueError(
                f"Corrupt SIL force queue: invalid generation_step at entry {queue_index}."
            )
        if isinstance(generation_time_s, bool) or not isinstance(generation_time_s, (int, float)):
            raise ValueError(
                f"Corrupt SIL force queue: invalid generation_time_s at entry {queue_index}."
            )
        generation_time_s = float(generation_time_s)
        if not math.isfinite(generation_time_s):
            raise ValueError(
                f"Corrupt SIL force queue: non-finite generation_time_s at entry {queue_index}."
            )
        return SilForcePacket(
            generation_step=int(generation_step),
            generation_time_s=generation_time_s,
            values=tuple(float(value) for value in values_array),
            startup_zero=False,
        )

    def _force_queue_from_checkpoint(
        self,
        state: dict[str, object],
        *,
        updates: int,
    ) -> list[SilForcePacket]:
        if state.get("force_packet_schema") != SIL_FORCE_PACKET_SCHEMA:
            raise ValueError(
                "SIL checkpoint is missing the required force-packet schema; refusing to "
                "resume without generated-vs-applied force provenance."
            )
        raw_lag = state.get("force_application_lag_steps")
        if isinstance(raw_lag, bool) or not isinstance(raw_lag, int) or raw_lag < 0:
            raise ValueError("Corrupt SIL checkpoint: force_application_lag_steps is invalid.")
        if raw_lag != self.force_application_lag_steps:
            raise ValueError(
                "SIL checkpoint force lag does not match the configured run: "
                f"checkpoint={raw_lag}, configured={self.force_application_lag_steps}."
            )
        raw_queue = state.get("force_application_queue")
        if not isinstance(raw_queue, list) or len(raw_queue) != raw_lag:
            observed = None if not isinstance(raw_queue, list) else len(raw_queue)
            raise ValueError(
                "Corrupt SIL force queue: expected exactly "
                f"{raw_lag} packet(s), observed {observed}."
            )
        queue = [
            self._packet_from_checkpoint(raw_packet, queue_index=index)
            for index, raw_packet in enumerate(raw_queue)
        ]
        expected_steps = list(range(max(0, updates - raw_lag), updates))
        expected_startup_count = raw_lag - len(expected_steps)
        if any(not packet.startup_zero for packet in queue[:expected_startup_count]) or any(
            packet.startup_zero for packet in queue[expected_startup_count:]
        ):
            raise ValueError(
                "Corrupt SIL force queue: startup-zero packets are not the expected leading prefix."
            )
        observed_steps = [
            packet.generation_step for packet in queue[expected_startup_count:]
        ]
        if observed_steps != expected_steps:
            raise ValueError(
                "Corrupt SIL force queue: generation steps do not match completed updates; "
                f"expected {expected_steps}, observed {observed_steps}."
            )
        for packet in queue[expected_startup_count:]:
            expected_time = float(packet.generation_step) * self.timestep  # type: ignore[arg-type]
            if not math.isclose(
                float(packet.generation_time_s),
                expected_time,
                rel_tol=1.0e-12,
                abs_tol=max(1.0e-12, abs(self.timestep) * 1.0e-12),
            ):
                raise ValueError(
                    "Corrupt SIL force queue: generation time is inconsistent with generation "
                    f"step {packet.generation_step}."
                )
        return queue

    def _flush_compact_history(self, *, durable: bool) -> None:
        if self._compact_history_handle is not None:
            self._compact_history_handle.flush()
            if durable:
                os.fsync(self._compact_history_handle.fileno())
        if self._compact_history_rows_written:
            if self._compact_history_path is None or not self._compact_history_path.is_file():
                raise RuntimeError("Compact SIL history disappeared during the run.")
            observed_size = self._compact_history_path.stat().st_size
            if observed_size != self._compact_history_size_bytes:
                raise RuntimeError(
                    "Compact SIL history size changed outside the writer: "
                    f"expected {self._compact_history_size_bytes}, observed {observed_size}."
                )

    def _compact_history_checkpoint_state(self) -> dict[str, object]:
        if self._compact_history_path is None:
            raise RuntimeError(
                "Compact SIL history cannot be checkpointed without a configured output path."
            )
        if self.history_rows:
            raise RuntimeError("Compact SIL history unexpectedly retained in-memory row dictionaries.")
        if self._compact_history_rows_written != self.updates:
            raise RuntimeError(
                "Compact SIL history row count does not match completed updates: "
                f"rows={self._compact_history_rows_written}, updates={self.updates}."
            )
        self._flush_compact_history(durable=True)
        expected_sha = self._compact_history_hasher.hexdigest()
        if not self._compact_history_size_bytes and expected_sha != hashlib.sha256(b"").hexdigest():
            raise RuntimeError("Compact SIL history hasher is inconsistent for an empty history.")
        return {
            "schema": SIL_COMPACT_HISTORY_CHECKPOINT_SCHEMA,
            "path": str(self._compact_history_path),
            "rows": int(self._compact_history_rows_written),
            "columns": list(self._compact_history_columns),
            "size_bytes": int(self._compact_history_size_bytes),
            "sha256": expected_sha,
            "generated_digest_window": [
                dict(entry) for entry in self._compact_generated_digest_window
            ],
        }

    def checkpoint_state(self) -> dict[str, object]:
        velocity_state = self.velocity_diagnostics()
        mode = self._configured_history_mode()
        result: dict[str, object] = {
            "schema": SIL_COUPLER_CHECKPOINT_SCHEMA,
            "updates": int(self.updates),
            "timestep_s": float(self.timestep),
            "use_qblade_local_inflow": bool(getattr(self, "use_qblade_local_inflow", False)),
            "max_abs_force": float(self.max_abs_force),
            "last_message": self.last_message,
            "velocity_diagnostics": velocity_state,
            "timing_diagnostics": self.timing_diagnostics(),
            "runtime": self.runtime.checkpoint_state(),
            "record_history": bool(self.record_history),
            "history_mode": self.history_mode,
            "history_rows": self.history_rows if mode == "full" else [],
            "force_packet_schema": SIL_FORCE_PACKET_SCHEMA,
            "force_application_lag_steps": int(self.force_application_lag_steps),
            "force_application_queue": [packet.as_dict() for packet in self._force_application_queue],
        }
        if mode == "compact":
            result["compact_history"] = self._compact_history_checkpoint_state()
        if getattr(self, "hvawt_response_adapter", None) is not None:
            result["hvawt_response_state"] = self.hvawt_response_adapter.checkpoint_state()
        return result

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != SIL_COUPLER_CHECKPOINT_SCHEMA:
            raise ValueError(
                "Corrupt or legacy SIL checkpoint: schema/timestep identity is incompatible."
            )
        checkpoint_local_inflow = state.get("use_qblade_local_inflow", False)
        if not isinstance(checkpoint_local_inflow, bool) or checkpoint_local_inflow != bool(
            getattr(self, "use_qblade_local_inflow", False)
        ):
            raise ValueError("SIL checkpoint local-inflow mode differs from this run.")
        raw_updates = state.get("updates")
        if isinstance(raw_updates, bool) or not isinstance(raw_updates, int) or raw_updates < 0:
            raise ValueError("Corrupt SIL checkpoint: updates must be a nonnegative integer.")
        updates = int(raw_updates)
        checkpoint_timestep = state.get("timestep_s")
        if (
            isinstance(checkpoint_timestep, bool)
            or not isinstance(checkpoint_timestep, (int, float))
            or not math.isclose(
                float(checkpoint_timestep),
                float(self.timestep),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("Corrupt SIL checkpoint: timestep identity differs.")
        raw_max_abs_force = state.get("max_abs_force")
        if isinstance(raw_max_abs_force, bool) or not isinstance(raw_max_abs_force, (int, float)):
            raise ValueError("Corrupt SIL checkpoint: max_abs_force must be finite and nonnegative.")
        max_abs_force = float(raw_max_abs_force)
        if not math.isfinite(max_abs_force) or max_abs_force < 0.0:
            raise ValueError("Corrupt SIL checkpoint: max_abs_force must be finite and nonnegative.")
        checkpoint_record_history = state.get("record_history")
        if not isinstance(checkpoint_record_history, bool):
            raise ValueError("Corrupt SIL checkpoint: record_history must be boolean.")
        if checkpoint_record_history != self.record_history:
            raise ValueError(
                "SIL checkpoint history mode does not match this run: "
                f"checkpoint={checkpoint_record_history}, configured={self.record_history}."
            )
        checkpoint_history_mode = state.get("history_mode")
        if checkpoint_history_mode is None:
            # Backward-compatible interpretation of checkpoints written before
            # compact history existed. A recorded legacy history was always full.
            checkpoint_history_mode = "full" if checkpoint_record_history else "off"
        if checkpoint_history_mode not in SIL_HISTORY_MODES:
            raise ValueError("Corrupt SIL checkpoint: history_mode is invalid.")
        if (checkpoint_history_mode != "off") != checkpoint_record_history:
            raise ValueError(
                "Corrupt SIL checkpoint: record_history and history_mode disagree."
            )
        if checkpoint_history_mode != self.history_mode:
            raise ValueError(
                "SIL checkpoint history mode does not match this run: "
                f"checkpoint={checkpoint_history_mode}, configured={self.history_mode}."
            )
        runtime_state = state.get("runtime")
        if not isinstance(runtime_state, dict):
            raise ValueError("Corrupt SIL checkpoint: runtime state is missing or malformed.")
        restored_queue = self._force_queue_from_checkpoint(state, updates=updates)
        history_rows = state.get("history_rows")
        if not isinstance(history_rows, list) or any(not isinstance(row, dict) for row in history_rows):
            raise ValueError("Corrupt SIL checkpoint: history_rows must be a list of objects.")
        restored_history_rows = [dict(row) for row in history_rows]
        compact_restore: dict[str, object] | None = None
        if self.history_mode == "full":
            if len(restored_history_rows) != updates:
                raise ValueError(
                    "Corrupt SIL checkpoint: force-history row count must equal completed updates."
                )
            self._validate_history_rows(restored_history_rows)
            self._validate_force_queue_against_history(
                restored_queue,
                restored_history_rows,
                updates=updates,
            )
        elif self.history_mode == "compact":
            if restored_history_rows:
                raise ValueError(
                    "Corrupt SIL checkpoint: compact history must not contain in-memory rows."
                )
            compact_restore = self._prepare_compact_history_restore(
                state.get("compact_history"),
                updates=updates,
                restored_queue=restored_queue,
            )
        elif restored_history_rows:
            raise ValueError("Corrupt SIL checkpoint: history rows exist while history recording is disabled.")

        last_message = state.get("last_message")
        velocity = state.get("velocity_diagnostics")
        timing = state.get("timing_diagnostics")
        restored_velocity_axes, restored_velocity_channels = (
            self._prepare_velocity_restore(velocity, updates=updates)
        )
        timing_keys = set(self._timing.as_dict())
        if not isinstance(timing, dict) or set(timing) != timing_keys:
            raise ValueError("Corrupt SIL checkpoint: timing diagnostics keys differ.")
        for name in (
            "samples",
            "qblade_time_available_count",
            "qblade_time_missing_count",
        ):
            if type(timing[name]) is not int or timing[name] < 0:
                raise ValueError(f"Corrupt SIL checkpoint: timing {name} is invalid.")
        if (
            timing["samples"] != updates
            or timing["qblade_time_available_count"]
            + timing["qblade_time_missing_count"]
            != updates
        ):
            raise ValueError("Corrupt SIL checkpoint: timing sample counts differ.")
        for name in (
            "wrapper_time_min_s",
            "wrapper_time_max_s",
            "applied_force_time_min_s",
            "applied_force_time_max_s",
            "azimuth_min_deg",
            "azimuth_max_deg",
            "time_delta_qblade_minus_wrapper_min_s",
            "time_delta_qblade_minus_wrapper_max_s",
        ):
            value = timing[name]
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"Corrupt SIL checkpoint: timing {name} is invalid.")
        maximum_delta = timing["time_delta_qblade_minus_wrapper_max_abs_s"]
        if (
            isinstance(maximum_delta, bool)
            or not isinstance(maximum_delta, (int, float))
            or not math.isfinite(float(maximum_delta))
            or float(maximum_delta) < 0.0
        ):
            raise ValueError("Corrupt SIL checkpoint: timing maximum delta is invalid.")
        first_samples = timing["first_samples"]
        last_sample = timing["last_sample"]
        if (
            not isinstance(first_samples, list)
            or len(first_samples) > self._timing.sample_limit
            or any(not isinstance(row, dict) for row in first_samples)
            or (last_sample is not None and not isinstance(last_sample, dict))
            or ((updates == 0) != (last_sample is None))
        ):
            raise ValueError("Corrupt SIL checkpoint: timing samples are invalid.")
        restored_timing = SilTimingDiagnostics(sample_limit=self._timing.sample_limit)
        restored_timing.restore(timing)

        hvawt_response_state = state.get("hvawt_response_state")
        if getattr(self, "hvawt_response_adapter", None) is None:
            if hvawt_response_state is not None:
                raise ValueError(
                    "Legacy SIL checkpoint cannot contain H-VAWT response state."
                )
        elif not isinstance(hvawt_response_state, dict):
            raise ValueError(
                "H-VAWT SIL checkpoint is missing hvawt_response_state."
            )

        # Runtime restoration can reject an otherwise structurally valid
        # checkpoint. Keep the coupler's local counters and diagnostics at
        # their pre-restore values until that authoritative restore succeeds.
        # Compact-history activation (which may truncate an audited,
        # uncheckpointed tail) remains later still.
        adapter = getattr(self, "hvawt_response_adapter", None)
        previous_runtime_state = self.runtime.checkpoint_state()
        previous_adapter_state = (
            None if adapter is None else adapter.checkpoint_state()
        )
        try:
            self.runtime.restore_checkpoint_state(runtime_state)
            if adapter is not None:
                adapter.restore_checkpoint_state(hvawt_response_state)
        except Exception:
            # Treat runtime and response state as one transaction.  Either
            # subordinate restore may perform validation after staging state,
            # so put both objects back before exposing the failure.
            self.runtime.restore_checkpoint_state(previous_runtime_state)
            if adapter is not None and previous_adapter_state is not None:
                adapter.restore_checkpoint_state(previous_adapter_state)
            raise

        self.updates = updates
        self.max_abs_force = max_abs_force
        if isinstance(last_message, str):
            self.last_message = last_message

        self.velocity_axis_stats = restored_velocity_axes
        self.velocity_channel_stats = restored_velocity_channels

        self._timing = restored_timing

        self._force_application_queue = restored_queue
        self.history_rows = restored_history_rows
        if compact_restore is not None:
            self._activate_compact_history_restore(compact_restore)

    def _record_timing_sample(
        self,
        qblade,
        step_index: int,
        applied_packet: SilForcePacket,
    ) -> None:
        wrapper_time = float(self.swap[self._time_offset])
        wrapper_timestep = float(self.swap[self._timestep_offset])
        azimuth = float(self.swap[self._azimuth_offset])
        qblade_time = self._optional_qblade_value(qblade, "Time [s]")
        self._timing.record(
            step_index=step_index,
            wrapper_time_s=wrapper_time,
            wrapper_timestep=wrapper_timestep,
            qblade_time_pre_advance_s=qblade_time,
            azimuth_deg=azimuth,
            applied_force_time_s=applied_packet.generation_time_s,
            applied_force_source_step=applied_packet.generation_step,
            applied_force_startup_zero=applied_packet.startup_zero,
            applied_force_lag_steps=self.force_application_lag_steps,
        )

    def _fill_inputs(self, qblade, step_index: int) -> None:
        for row in self.input_rows:
            idx = int(row["index"])
            variable = str(row["variable"])
            if variable == "Time [s]":
                value = step_index * self.timestep
            elif variable == "Timestep [-]":
                value = step_index
            else:
                value = self._qblade_value(qblade, variable)
            if not math.isfinite(value):
                raise RuntimeError(
                    f"QBlade returned missing or non-finite SIL value for '{variable}': {value}. "
                    "Check that the generated external input table matches available QBlade channels."
                )
            self.swap[idx] = value
            self._record_velocity_input(variable, value)

    def _force_packets_for_step(
        self,
        current_force_values,
        *,
        generation_step: int,
        generation_time_s: float,
    ) -> tuple[SilForcePacket, SilForcePacket]:
        generated_packet = self._generated_force_packet(
            current_force_values,
            generation_step=generation_step,
            generation_time_s=generation_time_s,
        )
        if self.force_application_lag_steps <= 0:
            if self._force_application_queue:
                raise RuntimeError("Internal SIL force queue is nonempty for zero-lag application.")
            return generated_packet, generated_packet
        if len(self._force_application_queue) != self.force_application_lag_steps:
            raise RuntimeError(
                "Internal SIL force queue length changed unexpectedly: "
                f"expected {self.force_application_lag_steps}, observed "
                f"{len(self._force_application_queue)}."
            )
        applied_packet = self._force_application_queue.pop(0)
        if not isinstance(applied_packet, SilForcePacket):
            raise RuntimeError("Internal SIL force queue contains a mutable or untyped entry.")
        self._force_application_queue.append(generated_packet)
        return generated_packet, applied_packet

    def _apply_outputs(self, qblade, force_values=None) -> None:
        if force_values is not None:
            force_values = self.np.asarray(force_values, dtype=float).reshape(-1)
        for row in self.output_rows:
            idx = int(row["index"])
            if force_values is None:
                value = float(self.swap[idx])
            else:
                force_idx = idx - self._force_spec.offset
                if force_idx < 0 or force_idx >= force_values.size:
                    raise RuntimeError(f"Output row index {idx} is outside the configured force block.")
                value = float(force_values[force_idx])
            if not math.isfinite(value):
                raise RuntimeError(f"VorLap returned non-finite force at swap index {idx}: {value}")
            qblade.setExternalAction(
                str(row["action"]).encode("utf-8"),
                str(row["id"]).encode("utf-8"),
                value,
                float(row["position"]),
                str(row["direction"]).encode("utf-8"),
                bool(row["local"]),
                0,
            )

    @staticmethod
    def _safe_column_token(text: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")

    def _force_metrics(self, force_array) -> dict[str, object]:
        force_array = self.np.asarray(force_array, dtype=float).reshape((-1, 3))
        force_norms = self.np.linalg.norm(force_array, axis=1) if force_array.size else self.np.zeros(0)
        max_node_idx = int(self.np.argmax(force_norms)) if force_norms.size else -1
        max_force = force_array[max_node_idx] if max_node_idx >= 0 else self.np.zeros(3)
        max_node_id = self.node_ids[max_node_idx] if 0 <= max_node_idx < len(self.node_ids) else ""
        return {
            "max_abs_force_component_n": (
                float(self.np.max(self.np.abs(force_array))) if force_array.size else 0.0
            ),
            "max_force_norm_n": float(force_norms[max_node_idx]) if max_node_idx >= 0 else 0.0,
            "max_force_node_id": max_node_id,
            "max_force_x_n": float(max_force[0]),
            "max_force_y_n": float(max_force[1]),
            "max_force_z_n": float(max_force[2]),
        }

    def _compact_force_metrics(self, force_array) -> dict[str, object]:
        """Return fixed-width whole-vector metrics for compact history rows."""
        force_array = self.np.asarray(force_array, dtype=float).reshape((-1, 3))
        metrics = self._force_metrics(force_array)
        total = self.np.sum(force_array, axis=0) if force_array.size else self.np.zeros(3)
        metrics.update(
            {
                "total_force_x_n": float(total[0]),
                "total_force_y_n": float(total[1]),
                "total_force_z_n": float(total[2]),
                "force_vector_l1_n": float(self.np.sum(self.np.abs(force_array))),
                "force_vector_l2_n": float(self.np.linalg.norm(force_array.reshape(-1))),
            }
        )
        return metrics

    @staticmethod
    def _force_vector_sha256(values) -> str:
        """Hash an ordered force vector using a platform-independent binary encoding."""
        numeric = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Cannot digest a non-finite SIL force vector.")
        payload = SIL_FORCE_VECTOR_DIGEST_SCHEMA.encode("ascii") + b"\0"
        payload += struct.pack("!Q", len(numeric))
        if numeric:
            payload += struct.pack(f"!{len(numeric)}d", *numeric)
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _history_digest(row: dict[str, object], key: str, *, row_index: int) -> str:
        value = row.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(
                f"Corrupt SIL force history: row {row_index} field {key!r} is not a SHA-256 digest."
            )
        return value

    def _configured_history_mode(self) -> str:
        """Return mode while tolerating legacy test doubles built with ``__new__``."""
        mode = getattr(self, "history_mode", None)
        if mode is None:
            return "full" if bool(getattr(self, "record_history", False)) else "off"
        if mode not in SIL_HISTORY_MODES:
            raise ValueError(f"Invalid configured SIL history mode: {mode!r}.")
        return str(mode)

    @staticmethod
    def _history_number(row: dict[str, object], key: str, *, row_index: int) -> float:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Corrupt SIL force history: row {row_index} field {key!r} is not numeric.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Corrupt SIL force history: row {row_index} field {key!r} is non-finite.")
        return value

    def _history_force_vector(
        self,
        row: dict[str, object],
        prefix: str,
        *,
        row_index: int,
    ):
        values: list[float] = []
        for node_id in self.node_ids:
            token = self._safe_column_token(node_id)
            for axis in ("x", "y", "z"):
                values.append(
                    self._history_number(
                        row,
                        f"{prefix}_{token}_{axis}_n",
                        row_index=row_index,
                    )
                )
        return self.np.asarray(values, dtype=float)

    def _validate_history_rows(self, rows: list[dict[str, object]]) -> None:
        mode = self._configured_history_mode()
        if mode == "full":
            self._validate_full_history_rows(rows)
            return
        if mode == "compact":
            if rows:
                raise ValueError(
                    "Corrupt compact SIL force history: streamed rows must not be retained in memory."
                )
            return
        if rows:
            raise ValueError(
                "Corrupt SIL force history: rows exist while history recording is disabled."
            )

    @staticmethod
    def _compact_metric_names() -> tuple[str, ...]:
        return (
            "max_abs_force_component_n",
            "max_force_norm_n",
            "max_force_x_n",
            "max_force_y_n",
            "max_force_z_n",
            "total_force_x_n",
            "total_force_y_n",
            "total_force_z_n",
            "force_vector_l1_n",
            "force_vector_l2_n",
        )

    def _validate_compact_history_row(
        self,
        row: dict[str, object],
        *,
        row_index: int,
        generated_window: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Validate one compact row and return its bounded generated-digest window."""
        required_metric_names = self._compact_metric_names()
        if row.get("force_history_schema") != SIL_COMPACT_FORCE_HISTORY_SCHEMA:
            raise ValueError(
                f"Corrupt compact SIL force history: row {row_index} has an invalid history schema."
            )
        if row.get("force_vector_digest_schema") != SIL_FORCE_VECTOR_DIGEST_SCHEMA:
            raise ValueError(
                f"Corrupt compact SIL force history: row {row_index} has an invalid vector digest schema."
            )
        component_count = row.get("force_vector_component_count")
        if (
            isinstance(component_count, bool)
            or not isinstance(component_count, int)
            or component_count != self._force_spec.size
        ):
            raise ValueError(
                f"Corrupt compact SIL force history: row {row_index} has an invalid force-vector size."
            )
        step_index = row.get("step_index")
        generated_step = row.get("generated_step_index")
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or step_index != row_index
            or generated_step != step_index
        ):
            raise ValueError(
                f"Corrupt compact SIL force history: nonsequential generated step at row {row_index}."
            )
        generated_time = self._history_number(row, "generated_time_s", row_index=row_index)
        wrapper_time = self._history_number(row, "time_s", row_index=row_index)
        expected_time = row_index * self.timestep
        if not math.isclose(
            generated_time,
            wrapper_time,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            generated_time,
            expected_time,
            rel_tol=1.0e-12,
            abs_tol=max(1.0e-12, abs(self.timestep) * 1.0e-12),
        ):
            raise ValueError(
                f"Corrupt compact SIL force history: generated time is inconsistent at row {row_index}."
            )
        self._history_number(row, "qblade_timestep", row_index=row_index)
        self._history_number(row, "azimuth_deg", row_index=row_index)
        if row.get("applied_force_lag_steps") != self.force_application_lag_steps:
            raise ValueError(
                f"Corrupt compact SIL force history: applied lag is inconsistent at row {row_index}."
            )

        generated_digest = self._history_digest(
            row,
            "generated_force_vector_sha256",
            row_index=row_index,
        )
        applied_digest = self._history_digest(
            row,
            "applied_force_vector_sha256",
            row_index=row_index,
        )
        expected_source_index = row_index - self.force_application_lag_steps
        startup_zero = row.get("applied_startup_zero")
        if not isinstance(startup_zero, bool):
            raise ValueError(
                f"Corrupt compact SIL force history: startup flag is not boolean at row {row_index}."
            )
        if expected_source_index < 0:
            if (
                not startup_zero
                or row.get("applied_source_step_index") is not None
                or row.get("applied_source_time_s") is not None
            ):
                raise ValueError(
                    f"Corrupt compact SIL force history: startup provenance is invalid at row {row_index}."
                )
            zero_digest = self._force_vector_sha256(
                (0.0 for _ in range(self._force_spec.size))
            )
            if applied_digest != zero_digest:
                raise ValueError(
                    f"Corrupt compact SIL force history: startup force digest is not exactly zero at row {row_index}."
                )
        else:
            if startup_zero or row.get("applied_source_step_index") != expected_source_index:
                raise ValueError(
                    f"Corrupt compact SIL force history: applied source step is invalid at row {row_index}."
                )
            if self.force_application_lag_steps == 0:
                expected_source = {
                    "step_index": row_index,
                    "generated_time_s": generated_time,
                    "generated_force_vector_sha256": generated_digest,
                }
            else:
                if len(generated_window) != self.force_application_lag_steps:
                    raise ValueError(
                        "Corrupt compact SIL force history: generated digest lag window is incomplete "
                        f"at row {row_index}."
                    )
                expected_source = generated_window[0]
            source_time = self._history_number(
                row,
                "applied_source_time_s",
                row_index=row_index,
            )
            if expected_source.get("step_index") != expected_source_index or not math.isclose(
                source_time,
                float(expected_source["generated_time_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"Corrupt compact SIL force history: applied source time is invalid at row {row_index}."
                )
            if applied_digest != expected_source.get("generated_force_vector_sha256"):
                raise ValueError(
                    "Corrupt compact SIL force history: applied force digest does not equal its "
                    f"generated source at row {row_index}."
                )

        if row.get("applied_power_sign_convention") != (
            "positive_when_applied_force_does_work_on_current_structural_velocity"
        ):
            raise ValueError(
                f"Corrupt compact SIL force history: applied-power convention is invalid at row {row_index}."
            )
        self._history_number(
            row,
            "applied_force_dot_structural_velocity_w",
            row_index=row_index,
        )
        self._history_digest(
            row,
            "structural_velocity_vector_sha256",
            row_index=row_index,
        )
        for prefix in ("generated_", "applied_"):
            node_id = row.get(f"{prefix}max_force_node_id")
            if not isinstance(node_id, str) or node_id not in self.node_ids:
                raise ValueError(
                    "Corrupt compact SIL force history: row "
                    f"{row_index} field {prefix + 'max_force_node_id'!r} is invalid."
                )
            for metric_name in required_metric_names:
                self._history_number(
                    row,
                    f"{prefix}{metric_name}",
                    row_index=row_index,
                )

        response_fields = {
            key: value
            for key, value in row.items()
            if key.startswith("response_") and key != "response_model_type"
        }
        if response_fields:
            model_type = row.get("response_model_type")
            if not isinstance(model_type, str) or not model_type.strip():
                raise ValueError(
                    "Corrupt compact SIL force history: response diagnostics lack a model type "
                    f"at row {row_index}."
                )
            for key, value in response_fields.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(float(value)):
                        raise ValueError(
                            "Corrupt compact SIL force history: response diagnostic "
                            f"{key!r} is non-finite at row {row_index}."
                        )
                    continue
                if isinstance(value, str) and value.strip() and "\n" not in value and "\r" not in value:
                    continue
                raise ValueError(
                    "Corrupt compact SIL force history: response diagnostic "
                    f"{key!r} has unsupported type at row {row_index}."
                )

        generated_entry = {
            "step_index": row_index,
            "generated_time_s": generated_time,
            "generated_force_vector_sha256": generated_digest,
        }
        if self.force_application_lag_steps == 0:
            return []
        return [*generated_window, generated_entry][-self.force_application_lag_steps :]

    @staticmethod
    def _compact_csv_bytes(
        columns: tuple[str, ...],
        row: dict[str, object] | None,
        *,
        include_header: bool,
    ) -> bytes:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise")
        if include_header:
            writer.writeheader()
        if row is not None:
            writer.writerow(row)
        return stream.getvalue().encode("utf-8")

    def _write_compact_history_row(self, row: dict[str, object]) -> None:
        if self._compact_history_path is None:
            raise RuntimeError(
                "Compact SIL history requires a configured output path before the first step."
            )
        if self._compact_history_resume and self._compact_history_handle is None:
            raise RuntimeError(
                "Compact SIL history resume state must be restored before timestep advancement."
            )
        invalid_multiline_fields = [
            key
            for key, value in row.items()
            if isinstance(value, str) and ("\n" in value or "\r" in value)
        ]
        if invalid_multiline_fields:
            raise RuntimeError(
                "Compact SIL history does not permit multiline string fields: "
                + ", ".join(invalid_multiline_fields)
            )
        row_columns = tuple(row)
        if not self._compact_history_columns:
            self._compact_history_columns = row_columns
        elif row_columns != self._compact_history_columns:
            raise RuntimeError(
                "Compact SIL history columns changed during the run; refusing an ambiguous CSV schema."
            )
        next_window = self._validate_compact_history_row(
            row,
            row_index=self._compact_history_rows_written,
            generated_window=self._compact_generated_digest_window,
        )
        include_header = self._compact_history_rows_written == 0
        payload = self._compact_csv_bytes(
            self._compact_history_columns,
            row,
            include_header=include_header,
        )
        if self._compact_history_handle is None:
            self._compact_history_path.parent.mkdir(parents=True, exist_ok=True)
            restored_path_exists = self._compact_history_path.exists()
            if self._compact_history_restored:
                if self._compact_history_size_bytes and not restored_path_exists:
                    raise RuntimeError(
                        "Compact SIL history disappeared after restart validation; refusing "
                        "to create a headerless replacement for a nonempty checkpoint prefix."
                    )
                if restored_path_exists:
                    observed_size = self._compact_history_path.stat().st_size
                    if observed_size != self._compact_history_size_bytes:
                        raise RuntimeError(
                            "Compact SIL history size changed after restart validation: "
                            f"expected {self._compact_history_size_bytes}, observed {observed_size}."
                        )
            mode = (
                "ab"
                if self._compact_history_rows_written
                or (self._compact_history_restored and restored_path_exists)
                else "xb"
            )
            self._compact_history_handle = self._compact_history_path.open(mode)
            self._compact_history_restored = False
        written = self._compact_history_handle.write(payload)
        if written != len(payload):
            raise OSError(
                "Compact SIL history write was incomplete: "
                f"wrote {written} of {len(payload)} bytes."
            )
        self._compact_history_hasher.update(payload)
        self._compact_history_size_bytes += len(payload)
        self._compact_history_rows_written += 1
        self._compact_generated_digest_window = next_window

    @staticmethod
    def _compact_csv_scalar(value: str) -> object:
        if value == "True":
            return True
        if value == "False":
            return False
        try:
            numeric = float(value)
        except ValueError:
            return value
        if not math.isfinite(numeric):
            raise ValueError("Compact SIL history contains a non-finite response diagnostic.")
        return numeric

    def _parse_compact_csv_row(
        self,
        raw: dict[str, str],
        *,
        row_index: int,
    ) -> dict[str, object]:
        row: dict[str, object] = dict(raw)
        required_ints = (
            "force_vector_component_count",
            "step_index",
            "generated_step_index",
            "applied_force_lag_steps",
        )
        nullable_ints = ("applied_source_step_index",)
        required_floats = (
            "time_s",
            "generated_time_s",
            "qblade_timestep",
            "azimuth_deg",
            "applied_force_dot_structural_velocity_w",
            *(
                f"{prefix}{metric}"
                for prefix in ("generated_", "applied_")
                for metric in self._compact_metric_names()
            ),
        )
        nullable_floats = ("applied_source_time_s",)
        try:
            for key in required_ints:
                value = raw[key]
                parsed = int(value)
                if str(parsed) != value:
                    raise ValueError
                row[key] = parsed
            for key in nullable_ints:
                value = raw[key]
                if value == "":
                    row[key] = None
                else:
                    parsed = int(value)
                    if str(parsed) != value:
                        raise ValueError
                    row[key] = parsed
            for key in required_floats:
                row[key] = float(raw[key])
            for key in nullable_floats:
                row[key] = None if raw[key] == "" else float(raw[key])
            if raw["applied_startup_zero"] not in {"True", "False"}:
                raise ValueError
            row["applied_startup_zero"] = raw["applied_startup_zero"] == "True"
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Corrupt compact SIL force history: row {row_index} has an invalid typed field."
            ) from exc
        for key, value in raw.items():
            if key.startswith("response_") and key != "response_model_type":
                row[key] = self._compact_csv_scalar(value)
        return row

    def _audit_compact_history_prefix(
        self,
        path: Path,
        *,
        size_bytes: int,
        expected_rows: int,
        expected_columns: tuple[str, ...],
    ) -> tuple[object, list[dict[str, object]], int]:
        if not path.is_file():
            raise FileNotFoundError(f"Compact SIL history file is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size < size_bytes:
            raise ValueError(
                "Compact SIL history is shorter than its checkpoint prefix: "
                f"expected at least {size_bytes} bytes, observed {actual_size}."
            )
        hasher = hashlib.sha256()
        generated_window: list[dict[str, object]] = []
        observed_rows = 0
        remaining = size_bytes
        with path.open("rb") as handle:
            header: tuple[str, ...] = ()
            while remaining:
                raw_line = handle.readline(remaining + 1)
                if not raw_line or len(raw_line) > remaining or not raw_line.endswith(b"\n"):
                    raise ValueError(
                        "Compact SIL history checkpoint prefix does not end on a complete CSV row."
                    )
                remaining -= len(raw_line)
                hasher.update(raw_line)
                try:
                    cells = next(csv.reader([raw_line.decode("utf-8")]))
                except (UnicodeDecodeError, csv.Error) as exc:
                    raise ValueError("Compact SIL history contains invalid CSV data.") from exc
                if not header:
                    header = tuple(cells)
                    if header != expected_columns:
                        raise ValueError(
                            "Compact SIL history header does not match checkpoint columns."
                        )
                    continue
                if len(cells) != len(header):
                    raise ValueError(
                        f"Compact SIL history row {observed_rows} has the wrong column count."
                    )
                parsed = self._parse_compact_csv_row(
                    dict(zip(header, cells)),
                    row_index=observed_rows,
                )
                generated_window = self._validate_compact_history_row(
                    parsed,
                    row_index=observed_rows,
                    generated_window=generated_window,
                )
                observed_rows += 1
        if remaining != 0 or observed_rows != expected_rows:
            raise ValueError(
                "Compact SIL history row count does not match checkpoint state: "
                f"expected {expected_rows}, observed {observed_rows}."
            )
        if expected_rows > 0 and not expected_columns:
            raise ValueError("Compact SIL history checkpoint lacks CSV columns.")
        return hasher, generated_window, actual_size - size_bytes

    def _prepare_compact_history_restore(
        self,
        raw_state: object,
        *,
        updates: int,
        restored_queue: list[SilForcePacket],
    ) -> dict[str, object]:
        if not self._compact_history_resume or self._compact_history_path is None:
            raise ValueError(
                "Compact SIL checkpoint resume requires the same configured streaming output path."
            )
        if not isinstance(raw_state, dict):
            raise ValueError("Corrupt SIL checkpoint: compact_history state is missing.")
        if raw_state.get("schema") != SIL_COMPACT_HISTORY_CHECKPOINT_SCHEMA:
            raise ValueError("Corrupt SIL checkpoint: compact history schema is invalid.")
        raw_path = raw_state.get("path")
        if not isinstance(raw_path, str) or Path(raw_path).expanduser().resolve() != self._compact_history_path:
            raise ValueError(
                "SIL checkpoint compact history path does not match this run: "
                f"checkpoint={raw_path!r}, configured={str(self._compact_history_path)!r}."
            )
        rows = raw_state.get("rows")
        size_bytes = raw_state.get("size_bytes")
        sha256 = raw_state.get("sha256")
        columns = raw_state.get("columns")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows != updates:
            raise ValueError(
                "Corrupt SIL checkpoint: compact history rows must equal completed updates."
            )
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError("Corrupt SIL checkpoint: compact history size is invalid.")
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError("Corrupt SIL checkpoint: compact history SHA-256 is invalid.")
        if (
            not isinstance(columns, list)
            or any(not isinstance(column, str) or not column for column in columns)
            or len(columns) != len(set(columns))
        ):
            raise ValueError("Corrupt SIL checkpoint: compact history columns are invalid.")
        expected_columns = tuple(columns)
        if rows > 0 and (size_bytes <= 0 or not expected_columns):
            raise ValueError(
                "Corrupt SIL checkpoint: nonempty compact history lacks bytes or columns."
            )
        if rows == 0 and (size_bytes != 0 or expected_columns):
            raise ValueError(
                "Corrupt SIL checkpoint: empty compact history has bytes or columns."
            )

        if self._compact_history_path.exists():
            hasher, audited_window, tail_bytes = self._audit_compact_history_prefix(
                self._compact_history_path,
                size_bytes=size_bytes,
                expected_rows=rows,
                expected_columns=expected_columns,
            )
        elif size_bytes == 0 and rows == 0:
            hasher = hashlib.sha256()
            audited_window = []
            tail_bytes = 0
        else:
            raise FileNotFoundError(
                f"Compact SIL checkpoint history is missing: {self._compact_history_path}"
            )
        if hasher.hexdigest() != sha256:
            raise ValueError(
                "Compact SIL history checkpoint prefix SHA-256 does not match checkpoint state."
            )

        raw_window = raw_state.get("generated_digest_window")
        if not isinstance(raw_window, list) or any(
            not isinstance(entry, dict) for entry in raw_window
        ):
            raise ValueError(
                "Corrupt SIL checkpoint: compact generated-digest window is invalid."
            )
        expected_window_length = min(self.force_application_lag_steps, updates)
        if len(raw_window) != expected_window_length:
            raise ValueError(
                "Corrupt SIL checkpoint: compact generated-digest window has the wrong length."
            )
        normalized_window: list[dict[str, object]] = []
        expected_window_steps = list(range(updates - expected_window_length, updates))
        for expected_step, raw_entry in zip(expected_window_steps, raw_window):
            step = raw_entry.get("step_index")
            generated_time = raw_entry.get("generated_time_s")
            digest = raw_entry.get("generated_force_vector_sha256")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step != expected_step
                or isinstance(generated_time, bool)
                or not isinstance(generated_time, (int, float))
                or not math.isfinite(float(generated_time))
                or not math.isclose(
                    float(generated_time),
                    expected_step * self.timestep,
                    rel_tol=1.0e-12,
                    abs_tol=max(1.0e-12, abs(self.timestep) * 1.0e-12),
                )
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(
                    "Corrupt SIL checkpoint: compact generated-digest window entry is invalid."
                )
            normalized_window.append(
                {
                    "step_index": int(step),
                    "generated_time_s": float(generated_time),
                    "generated_force_vector_sha256": digest,
                }
            )
        if normalized_window != audited_window:
            raise ValueError(
                "Compact SIL checkpoint digest window does not match the audited CSV prefix."
            )

        queue_packets = [packet for packet in restored_queue if not packet.startup_zero]
        if len(queue_packets) != len(normalized_window):
            raise ValueError(
                "Compact SIL checkpoint force queue does not match its digest window length."
            )
        for packet, entry in zip(queue_packets, normalized_window):
            if packet.generation_step != entry["step_index"] or self._force_vector_sha256(
                packet.values
            ) != entry["generated_force_vector_sha256"]:
                raise ValueError(
                    "Corrupt SIL checkpoint: queued force packet does not match its compact "
                    f"generated-history digest at step {packet.generation_step}."
                )
        return {
            "path": self._compact_history_path,
            "rows": rows,
            "columns": expected_columns,
            "size_bytes": size_bytes,
            "hasher": hasher,
            "generated_digest_window": normalized_window,
            "tail_bytes": tail_bytes,
        }

    def _activate_compact_history_restore(self, prepared: dict[str, object]) -> None:
        path = Path(prepared["path"])
        size_bytes = int(prepared["size_bytes"])
        tail_bytes = int(prepared["tail_bytes"])
        if tail_bytes:
            with path.open("r+b") as handle:
                handle.truncate(size_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        path.parent.mkdir(parents=True, exist_ok=True)
        self._compact_history_handle = None
        self._compact_history_path = path
        self._compact_history_resume = False
        self._compact_history_restored = True
        self._compact_history_columns = tuple(prepared["columns"])
        self._compact_history_rows_written = int(prepared["rows"])
        self._compact_history_size_bytes = size_bytes
        self._compact_history_hasher = prepared["hasher"]
        self._compact_generated_digest_window = [
            dict(entry) for entry in prepared["generated_digest_window"]
        ]
        self._compact_history_discarded_tail_bytes += tail_bytes

    def _validate_force_queue_against_history(
        self,
        queue: list[SilForcePacket],
        rows: list[dict[str, object]],
        *,
        updates: int,
    ) -> None:
        """Bind checkpoint queue vectors to the generated history tail before resuming."""
        if not queue:
            return
        first_generated_queue_index = max(0, len(queue) - updates)
        for packet in queue[first_generated_queue_index:]:
            source_index = packet.generation_step
            if source_index is None or source_index >= len(rows):
                raise ValueError(
                    "Corrupt SIL checkpoint: queued force packet lacks a matching history row."
                )
            row = rows[source_index]
            generated = self._history_force_vector(
                row,
                "generated_force",
                row_index=source_index,
            )
            if not bool(self.np.array_equal(self.np.asarray(packet.values), generated)):
                raise ValueError(
                    "Corrupt SIL checkpoint: queued force packet does not match its full "
                    f"generated-history vector at step {source_index}."
                )

    def _validate_full_history_rows(self, rows: list[dict[str, object]]) -> None:
        """Fail closed if generated/applied force provenance is internally inconsistent."""
        generated_vectors = []
        for row_index, row in enumerate(rows):
            if row.get("force_history_schema") != SIL_FORCE_HISTORY_SCHEMA:
                raise ValueError(
                    f"Corrupt SIL force history: row {row_index} has an invalid history schema."
                )
            if row.get("legacy_force_columns_semantics") != "generated_force_alias":
                raise ValueError(
                    "Corrupt SIL force history: legacy force-column semantics are missing or invalid "
                    f"at row {row_index}."
                )
            step_index = row.get("step_index")
            generated_step = row.get("generated_step_index")
            if (
                isinstance(step_index, bool)
                or not isinstance(step_index, int)
                or step_index != row_index
                or generated_step != step_index
            ):
                raise ValueError(
                    f"Corrupt SIL force history: nonsequential generated step at row {row_index}."
                )
            generated_time = self._history_number(row, "generated_time_s", row_index=row_index)
            wrapper_time = self._history_number(row, "time_s", row_index=row_index)
            expected_time = row_index * self.timestep
            if not math.isclose(generated_time, wrapper_time, rel_tol=0.0, abs_tol=1.0e-12) or not math.isclose(
                generated_time,
                expected_time,
                rel_tol=1.0e-12,
                abs_tol=max(1.0e-12, abs(self.timestep) * 1.0e-12),
            ):
                raise ValueError(
                    f"Corrupt SIL force history: generated time is inconsistent at row {row_index}."
                )
            lag_steps = row.get("applied_force_lag_steps")
            if lag_steps != self.force_application_lag_steps:
                raise ValueError(
                    f"Corrupt SIL force history: applied lag is inconsistent at row {row_index}."
                )

            generated = self._history_force_vector(row, "generated_force", row_index=row_index)
            legacy = self._history_force_vector(row, "force", row_index=row_index)
            applied = self._history_force_vector(row, "applied_force", row_index=row_index)
            if not bool(self.np.array_equal(legacy, generated)):
                raise ValueError(
                    f"Corrupt SIL force history: legacy force aliases differ at row {row_index}."
                )
            generated_vectors.append(generated)

            expected_source_index = row_index - self.force_application_lag_steps
            startup_zero = row.get("applied_startup_zero")
            if not isinstance(startup_zero, bool):
                raise ValueError(
                    f"Corrupt SIL force history: startup flag is not boolean at row {row_index}."
                )
            if expected_source_index < 0:
                if not startup_zero or row.get("applied_source_step_index") is not None or row.get(
                    "applied_source_time_s"
                ) is not None:
                    raise ValueError(
                        f"Corrupt SIL force history: startup provenance is invalid at row {row_index}."
                    )
                if bool(self.np.any(applied != 0.0)):
                    raise ValueError(
                        f"Corrupt SIL force history: startup force is not exactly zero at row {row_index}."
                    )
            else:
                if startup_zero or row.get("applied_source_step_index") != expected_source_index:
                    raise ValueError(
                        f"Corrupt SIL force history: applied source step is invalid at row {row_index}."
                    )
                source_time = self._history_number(
                    row,
                    "applied_source_time_s",
                    row_index=row_index,
                )
                expected_source_time = self._history_number(
                    rows[expected_source_index],
                    "generated_time_s",
                    row_index=expected_source_index,
                )
                if not math.isclose(source_time, expected_source_time, rel_tol=0.0, abs_tol=1.0e-12):
                    raise ValueError(
                        f"Corrupt SIL force history: applied source time is invalid at row {row_index}."
                    )
                if not bool(self.np.array_equal(applied, generated_vectors[expected_source_index])):
                    raise ValueError(
                        "Corrupt SIL force history: applied force does not equal its generated source "
                        f"at row {row_index}."
                    )

            velocity_values: list[float] = []
            for node_id in self.node_ids:
                token = self._safe_column_token(node_id)
                for axis in ("x", "y", "z"):
                    velocity_values.append(
                        self._history_number(
                            row,
                            f"vel_{token}_{axis}_mps",
                            row_index=row_index,
                        )
                    )
            observed_work_rate = self._history_number(
                row,
                "applied_force_dot_structural_velocity_w",
                row_index=row_index,
            )
            if row.get("applied_power_sign_convention") != (
                "positive_when_applied_force_does_work_on_current_structural_velocity"
            ):
                raise ValueError(
                    f"Corrupt SIL force history: applied-power convention is invalid at row {row_index}."
                )
            expected_work_rate = float(self.np.dot(applied, self.np.asarray(velocity_values)))
            if not math.isclose(
                observed_work_rate,
                expected_work_rate,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"Corrupt SIL force history: applied F dot velocity is invalid at row {row_index}."
                )

            response_fields = {
                key: value
                for key, value in row.items()
                if key.startswith("response_") and key != "response_model_type"
            }
            if response_fields:
                model_type = row.get("response_model_type")
                if not isinstance(model_type, str) or not model_type.strip():
                    raise ValueError(
                        "Corrupt SIL force history: response diagnostics lack a model type "
                        f"at row {row_index}."
                    )
                for key, value in response_fields.items():
                    if isinstance(value, bool):
                        continue
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if not math.isfinite(float(value)):
                            raise ValueError(
                                "Corrupt SIL force history: response diagnostic "
                                f"{key!r} is non-finite at row {row_index}."
                            )
                        continue
                    if isinstance(value, str) and value.strip():
                        continue
                    raise ValueError(
                        "Corrupt SIL force history: response diagnostic "
                        f"{key!r} has unsupported type at row {row_index}."
                    )

    def _response_history_fields(self) -> dict[str, object]:
        if getattr(self, "hvawt_response_adapter", None) is not None:
            return self.hvawt_response_adapter.history_fields()
        diagnostics = getattr(self.runtime, "last_response_diagnostics", None)
        if diagnostics is None:
            return {}
        if not is_dataclass(diagnostics):
            raise RuntimeError("Response diagnostics must be a dataclass for SIL history.")
        model_type = str(getattr(self.runtime, "response_model_type", "")).strip()
        if not model_type or model_type == "none":
            raise RuntimeError("Response diagnostics require a concrete response_model_type.")
        result: dict[str, object] = {"response_model_type": model_type}
        for name, value in asdict(diagnostics).items():
            key = f"response_{name}"
            if isinstance(value, bool):
                result[key] = value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise RuntimeError(f"Response diagnostic {name!r} is non-finite.")
                result[key] = value
            elif isinstance(value, str) and value.strip():
                result[key] = value
            else:
                raise RuntimeError(
                    f"Response diagnostic {name!r} has unsupported SIL-history type."
                )
        return result

    def _record_history_sample(
        self,
        step_index: int,
        generated_packet: SilForcePacket,
        applied_packet: SilForcePacket,
    ) -> None:
        if not self.record_history:
            return
        if any(bool(row.get("local", False)) for row in self.output_rows):
            raise RuntimeError(
                "SIL applied-force work rate requires global force actions because the structural "
                "velocity history is in the global frame."
            )
        generated_force_array = self.np.asarray(generated_packet.values, dtype=float).reshape((-1, 3))
        applied_force_array = self.np.asarray(applied_packet.values, dtype=float).reshape((-1, 3))
        velocity_start = self._velocity_spec.offset
        velocity_stop = velocity_start + self._velocity_spec.size
        velocity_array = self.np.asarray(self.swap[velocity_start:velocity_stop], dtype=float).reshape((-1, 3))
        if generated_force_array.shape != applied_force_array.shape:
            raise RuntimeError("Generated and applied SIL force packet sizes differ.")
        expected_shape = (len(self.node_ids), 3)
        if generated_force_array.shape != expected_shape or velocity_array.shape != expected_shape:
            raise RuntimeError(
                "SIL history requires one three-component force and structural-velocity vector per "
                f"configured node; expected {expected_shape}, observed forces "
                f"{generated_force_array.shape} and velocities {velocity_array.shape}."
            )
        if not bool(self.np.all(self.np.isfinite(velocity_array))):
            raise RuntimeError("Current structural velocity contains a non-finite value.")
        if self._configured_history_mode() == "compact":
            generated_metrics = self._compact_force_metrics(generated_force_array)
            applied_metrics = self._compact_force_metrics(applied_force_array)
            row: dict[str, object] = {
                "force_history_schema": SIL_COMPACT_FORCE_HISTORY_SCHEMA,
                "force_vector_digest_schema": SIL_FORCE_VECTOR_DIGEST_SCHEMA,
                "force_vector_component_count": int(generated_force_array.size),
                "generated_force_vector_sha256": self._force_vector_sha256(
                    generated_packet.values
                ),
                "applied_force_vector_sha256": self._force_vector_sha256(
                    applied_packet.values
                ),
                "structural_velocity_vector_sha256": self._force_vector_sha256(
                    velocity_array.reshape(-1)
                ),
                "step_index": int(step_index),
                "time_s": float(self.swap[self._time_offset]),
                "generated_step_index": generated_packet.generation_step,
                "generated_time_s": generated_packet.generation_time_s,
                "applied_source_step_index": applied_packet.generation_step,
                "applied_source_time_s": applied_packet.generation_time_s,
                "applied_startup_zero": bool(applied_packet.startup_zero),
                "applied_force_lag_steps": int(self.force_application_lag_steps),
                "qblade_timestep": float(self.swap[self._timestep_offset]),
                "azimuth_deg": float(self.swap[self._azimuth_offset]),
                "applied_force_dot_structural_velocity_w": float(
                    self.np.sum(applied_force_array * velocity_array)
                ),
                "applied_power_sign_convention": (
                    "positive_when_applied_force_does_work_on_current_structural_velocity"
                ),
            }
            row.update(self._response_history_fields())
            for key, value in generated_metrics.items():
                row[f"generated_{key}"] = value
            for key, value in applied_metrics.items():
                row[f"applied_{key}"] = value
            self._write_compact_history_row(row)
            return
        generated_metrics = self._force_metrics(generated_force_array)
        applied_metrics = self._force_metrics(applied_force_array)
        row: dict[str, object] = {
            "force_history_schema": SIL_FORCE_HISTORY_SCHEMA,
            "legacy_force_columns_semantics": "generated_force_alias",
            "step_index": int(step_index),
            "time_s": float(self.swap[self._time_offset]),
            "generated_step_index": generated_packet.generation_step,
            "generated_time_s": generated_packet.generation_time_s,
            "applied_source_step_index": applied_packet.generation_step,
            "applied_source_time_s": applied_packet.generation_time_s,
            "applied_startup_zero": bool(applied_packet.startup_zero),
            "applied_force_lag_steps": int(self.force_application_lag_steps),
            "qblade_timestep": float(self.swap[self._timestep_offset]),
            "azimuth_deg": float(self.swap[self._azimuth_offset]),
            "applied_force_dot_structural_velocity_w": float(
                self.np.sum(applied_force_array * velocity_array)
            ),
            "applied_power_sign_convention": (
                "positive_when_applied_force_does_work_on_current_structural_velocity"
            ),
        }
        row.update(self._response_history_fields())
        for key, value in generated_metrics.items():
            row[key] = value
            row[f"generated_{key}"] = value
        for key, value in applied_metrics.items():
            row[f"applied_{key}"] = value
        for inode, node_id in enumerate(self.node_ids):
            token = self._safe_column_token(node_id)
            for axis_index, axis in enumerate(("x", "y", "z")):
                generated_value = float(generated_force_array[inode, axis_index])
                row[f"vel_{token}_{axis}_mps"] = float(velocity_array[inode, axis_index])
                row[f"generated_force_{token}_{axis}_n"] = generated_value
                row[f"applied_force_{token}_{axis}_n"] = float(
                    applied_force_array[inode, axis_index]
                )
                # Backward-compatible alias: legacy force_* columns always mean generated force.
                row[f"force_{token}_{axis}_n"] = generated_value
        self.history_rows.append(row)

    def write_history(self, path: Path) -> dict[str, object] | None:
        if not self.record_history:
            return None
        mode = self._configured_history_mode()
        if mode == "compact":
            target = Path(path).expanduser().resolve()
            if self._compact_history_path is None or target != self._compact_history_path:
                raise RuntimeError(
                    "Compact SIL history finalization path does not match its configured stream: "
                    f"configured={self._compact_history_path}, requested={target}."
                )
            if self.history_rows:
                raise RuntimeError(
                    "Compact SIL history unexpectedly retained in-memory row dictionaries."
                )
            if self._compact_history_rows_written != self.updates:
                raise RuntimeError(
                    "Compact SIL history row count does not match completed updates at finalization."
                )
            self.close_history(durable=True)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb"):
                    pass
            audited_hasher, audited_window, tail_bytes = self._audit_compact_history_prefix(
                target,
                size_bytes=self._compact_history_size_bytes,
                expected_rows=self._compact_history_rows_written,
                expected_columns=self._compact_history_columns,
            )
            if tail_bytes:
                raise RuntimeError(
                    "Compact SIL history gained bytes outside the streaming writer before finalization."
                )
            if audited_hasher.hexdigest() != self._compact_history_hasher.hexdigest():
                raise RuntimeError(
                    "Compact SIL history SHA-256 differs from the streamed byte identity."
                )
            if audited_window != self._compact_generated_digest_window:
                raise RuntimeError(
                    "Compact SIL history digest tail differs from the streaming provenance state."
                )
            return {
                "path": str(target),
                "rows": int(self._compact_history_rows_written),
                "columns": len(self._compact_history_columns),
                "node_count": len(self.node_ids),
                "mode": "compact",
                "schema": SIL_COMPACT_FORCE_HISTORY_SCHEMA,
                "vector_digest_schema": SIL_FORCE_VECTOR_DIGEST_SCHEMA,
                "streaming": True,
                "retained_rows": 0,
                "discarded_uncheckpointed_tail_bytes": int(
                    self._compact_history_discarded_tail_bytes
                ),
            }
        self._validate_history_rows(self.history_rows)
        columns: list[str] = []
        seen: set[str] = set()
        for row in self.history_rows:
            for key in row:
                if key not in seen:
                    columns.append(key)
                    seen.add(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.history_rows)
        result: dict[str, object] = {
            "path": str(path),
            "rows": len(self.history_rows),
            "columns": len(columns),
            "node_count": len(self.node_ids),
            "mode": mode,
            "schema": SIL_FORCE_HISTORY_SCHEMA,
        }
        if mode == "full":
            result["legacy_force_columns_semantics"] = "generated_force_alias"
        return result

    def close_history(self, *, durable: bool = False) -> None:
        """Flush and close the compact stream; full history has no open writer."""
        handle = getattr(self, "_compact_history_handle", None)
        if handle is None:
            return
        try:
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        finally:
            handle.close()
            self._compact_history_handle = None

    def step(
        self, qblade, step_index: int, *, ambient_velocity_global_mps=None,
    ) -> str:
        # Consume a complete QBlade input snapshot before writing any VorLap
        # forces back to the shared swap block for this timestep.
        timing = getattr(self, "_performance_timing", None)
        started_ns = timing.now_ns() if timing is not None else 0
        if ambient_velocity_global_mps is not None and not getattr(
            self, "use_qblade_local_inflow", False
        ):
            raise ValueError(
                "Explicit nodal wind samples require use_qblade_local_inflow=true."
            )
        if getattr(self, "use_qblade_local_inflow", False) and ambient_velocity_global_mps is None:
            raise RuntimeError(
                "use_qblade_local_inflow requires explicit nodal wind samples; "
                "the automatic ambient/induced-flow sampler is not selected yet."
            )
        wind_inputs = {}
        if ambient_velocity_global_mps is not None:
            wind = self.np.asarray(ambient_velocity_global_mps, dtype=float)
            if wind.shape != (len(self.node_ids), 3) or not self.np.isfinite(wind).all():
                raise ValueError("Nodal wind samples must be finite XYZ vectors in node_ids order.")
            wind_inputs["ambient_velocity_global_mps"] = wind
        self._fill_inputs(qblade, step_index)
        if timing is not None:
            timing.record_since(
                "qblade_state_reads_and_channel_conversion",
                started_ns,
            )
        force_start = self._force_spec.offset
        force_stop = force_start + self._force_spec.size
        if getattr(self, "hvawt_response_adapter", None) is None:
            self.runtime.update(self.swap, **wind_inputs)
            current_force_values = self.np.array(
                self.swap[force_start:force_stop], copy=True
            )
            self.last_message = self.runtime.message
        else:
            # QBlade 2.0.9.7 exposes the initialized state at wrapper calls 0
            # and 1 before its time channel advances.  Keep the response-core
            # time on that physical grid while retaining the legacy wrapper
            # time/step fields for force-lag provenance.
            response_time_s = max(0.0, (step_index - 1) * self.timestep)
            response_force = self.hvawt_response_adapter.step(
                self.swap,
                response_time_s=response_time_s,
                **wind_inputs,
            )
            current_force_values = self.np.asarray(
                response_force, dtype=float
            ).reshape(-1)
            self.swap[force_start:force_stop] = current_force_values
            self.last_message = (
                "Applied audited incremental airfoil response closure "
                f"at step {step_index}."
            )
        generated_packet, applied_packet = self._force_packets_for_step(
            current_force_values,
            generation_step=step_index,
            generation_time_s=float(self.swap[self._time_offset]),
        )
        self._record_timing_sample(qblade, step_index, applied_packet)
        if (
            getattr(self, "hvawt_response_adapter", None) is None
            and self.runtime.sample_step is None
        ):
            self.runtime.record_source_coverage_sample(self.timestep)
        if generated_packet.values:
            self.max_abs_force = max(
                self.max_abs_force,
                max(abs(value) for value in generated_packet.values),
            )
        started_ns = timing.now_ns() if timing is not None else 0
        self._apply_outputs(qblade, applied_packet.values)
        if timing is not None:
            timing.record_since("qblade_force_write_and_application", started_ns)
        started_ns = timing.now_ns() if timing is not None else 0
        self._record_history_sample(step_index, generated_packet, applied_packet)
        if timing is not None:
            timing.record_since(
                "compact_history_construction_and_writing",
                started_ns,
            )
        adapter = getattr(self, "hvawt_response_adapter", None)
        commit_coverage = getattr(adapter, "commit_source_coverage", None)
        if callable(commit_coverage):
            started_ns = timing.now_ns() if timing is not None else 0
            commit_coverage(step_index=step_index)
            if timing is not None:
                timing.record_since("source_coverage_aggregation", started_ns)
        self.updates += 1
        return self.last_message


class SilConstantForceCoupler:
    """Apply user-specified constant ADDFORCE actions through QBlade SIL."""

    def __init__(self, actions: list[dict[str, object]]) -> None:
        if not actions:
            raise ValueError("At least one constant-force action is required.")
        self.actions = actions
        self.updates = 0
        self.max_abs_force = max(abs(float(action["value"])) for action in actions)

    def step(self, qblade, step_index: int) -> str:
        for action in self.actions:
            value = float(action["value"])
            if not math.isfinite(value):
                raise RuntimeError(f"Constant force action has non-finite value: {action}")
            qblade.setExternalAction(
                str(action["action"]).encode("utf-8"),
                str(action["id"]).encode("utf-8"),
                value,
                float(action["position"]),
                str(action["direction"]).encode("utf-8"),
                bool(action["local"]),
                0,
            )
        self.updates += 1
        return f"Applied {len(self.actions)} constant SIL force action(s) at step {step_index}."


__all__ = [
    "AXES",
    "SIL_COMPACT_HISTORY_CHECKPOINT_SCHEMA",
    "SIL_COMPACT_FORCE_HISTORY_SCHEMA",
    "SYNTHETIC_SIL_INPUTS",
    "STRUCTURAL_VELOCITY_RE",
    "SIL_FORCE_HISTORY_SCHEMA",
    "SIL_FORCE_PACKET_SCHEMA",
    "SIL_FORCE_VECTOR_DIGEST_SCHEMA",
    "SIL_HISTORY_MODES",
    "TIMING_SAMPLE_LIMIT",
    "SilConstantForceCoupler",
    "SilForcePacket",
    "SilTimingDiagnostics",
    "SilVorLapCoupler",
    "find_qblade_external_parameter_file",
    "force_sign_probe_observations",
    "normalize_qblade_header",
    "parse_bool_flag",
    "parse_constant_force_action",
    "qblade_action_target_from_node",
    "qblade_node_position_label",
    "qblade_position_channel_name",
    "qblade_result_time_diagnostics",
    "read_qblade_result_numeric_columns",
    "resolve_sil_history_mode",
]
