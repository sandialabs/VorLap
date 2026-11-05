"""Typed configuration helpers for GUI-facing QBlade bridge workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .qblade_case_prep import QBladeCasePrepRequest, QBladeNodeSource, validate_force_scale
from .qblade_run import QBladeCouplingMode, QBladeExecutionMode, QBladeRunRequest


BridgeCliAction = Literal["prepare", "run"]
SourceBoundsPolicy = Literal["warn", "raise", "clip"]


def _normalize_source_bounds_policy(value: Any) -> SourceBoundsPolicy:
    policy = str(value).strip().lower()
    if policy not in ("warn", "raise", "clip"):
        raise ValueError("source_bounds_policy must be one of: 'warn', 'raise', or 'clip'.")
    return policy  # type: ignore[return-value]


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field} must be a boolean value.")


def _optional_bool(value: Any, *, field: str) -> bool | None:
    if value in (None, ""):
        return None
    return _as_bool(value, field=field)


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value: Any, *, field: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number.") from exc


def _optional_float_vector(value: Any, *, field: str, length: int = 3) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        parts = [part for part in value.replace(",", " ").split() if part]
    else:
        try:
            parts = list(value)
        except TypeError as exc:
            raise ValueError(f"{field} must contain {length} numeric components.") from exc
    if not parts or all(part in (None, "") for part in parts):
        return None
    if len(parts) != length or any(part in (None, "") for part in parts):
        raise ValueError(f"{field} must contain exactly {length} numeric components.")
    try:
        vector = tuple(float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain only numeric components.") from exc
    if any(not math.isfinite(component) for component in vector):
        raise ValueError(f"{field} components must be finite.")
    return vector


def _optional_int(value: Any, *, field: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer.") from exc


def _sequence(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _state_value(state: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in state:
            return state[name]
    return default


def _output_dir_validation_error(output_dir: str | Path) -> str | None:
    """Return a non-mutating validation error for an output path, if any."""
    path = Path(output_dir).expanduser()
    if path.exists():
        if not path.is_dir():
            return f"QBlade output directory is not a directory: {output_dir}"
        if not os.access(path, os.W_OK | os.X_OK):
            return f"QBlade output directory is not writable: {output_dir}"
        return None

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.exists():
        return f"Cannot create QBlade output directory {output_dir}; no existing parent directory was found."
    if not parent.is_dir():
        return f"Cannot create QBlade output directory {output_dir}; parent is not a directory: {parent}"
    if not os.access(parent, os.W_OK | os.X_OK):
        return f"Cannot create QBlade output directory {output_dir}; parent is not writable: {parent}"
    return None


def _resolved_checkpoint_path(config: "QBladeBridgeConfig") -> Path | None:
    if config.resume_checkpoint not in (None, ""):
        return Path(config.resume_checkpoint).expanduser().resolve()
    if not config.resume_from_checkpoint:
        return None
    if config.checkpoint_dir not in (None, ""):
        return Path(config.checkpoint_dir).expanduser().resolve() / "latest_checkpoint.json"
    return Path(config.output_dir).expanduser().resolve() / "checkpoints" / "latest_checkpoint.json"


def _metadata_path_value(metadata: Mapping[str, Any], key: str) -> Path | None:
    value = metadata.get(key)
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser().resolve()


def _resume_checkpoint_validation_errors(config: "QBladeBridgeConfig") -> list[str]:
    checkpoint_path = _resolved_checkpoint_path(config)
    if checkpoint_path is None:
        return []
    if not checkpoint_path.is_file():
        return [f"Resume checkpoint metadata does not exist: {checkpoint_path}"]

    try:
        metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Could not read resume checkpoint metadata {checkpoint_path}: {exc}"]
    if not isinstance(metadata, dict):
        return [f"Resume checkpoint metadata must be a JSON object: {checkpoint_path}"]

    errors: list[str] = []
    qblade_project = metadata.get("qblade_project")
    if qblade_project in (None, ""):
        errors.append(f"Resume checkpoint metadata does not include qblade_project: {checkpoint_path}")
    else:
        project_path = Path(str(qblade_project)).expanduser()
        if not project_path.is_absolute():
            project_path = (checkpoint_path.parent / project_path).resolve()
        if not project_path.is_file():
            errors.append(f"Resume checkpoint QBlade project does not exist: {project_path}")

    expected_paths = {
        "sim": Path(config.sim_path).expanduser().resolve(),
        "qblade_root": Path(config.qblade_root).expanduser().resolve(),
    }
    for key, expected in expected_paths.items():
        actual = _metadata_path_value(metadata, key)
        if actual is not None and actual != expected:
            errors.append(f"Resume checkpoint {key} mismatch: checkpoint has {actual}, current config has {expected}")

    expected_values: dict[str, object | None] = {
        "coupling_mode": config.coupling_mode,
        "result_name": config.result_name,
        "threads": int(config.threads),
        "use_opencl": bool(config.use_opencl),
    }
    effective_steps = config.effective_steps()
    if effective_steps is not None:
        expected_values["requested_steps"] = int(effective_steps)
    for key, expected in expected_values.items():
        actual = metadata.get(key)
        if expected is None or actual is None:
            continue
        if actual != expected:
            errors.append(f"Resume checkpoint {key} mismatch: checkpoint has {actual!r}, current config has {expected!r}")

    return errors


@dataclass(frozen=True)
class QBladeBridgeConfig:
    """Shared bridge configuration for GUI, CLI, tests, and run summaries."""

    sim_path: str | Path
    qblade_root: str | Path
    output_dir: str | Path
    airfoil_dir: str | Path
    qblade_binary: str | Path | None = None
    python_executable: str | Path | None = None
    run_root: str | Path | None = None
    result_name: str | None = None
    library_stem: str = "libvorlap_qblade_bridge"
    function_name: str = "update"
    parameter_file: str | Path | None = None
    sample_step: float | None = None
    default_airfoil_id: str = "default"
    include_struts: bool = True
    include_tower: bool = False
    node_source: QBladeNodeSource = "structural"
    include_tower_nodes: bool = False
    tower_airfoil_id: str = "cylinder"
    n_freq_depth: int | None = None
    force_scale: float = 1.0
    diagnostic_force_scale: bool = False
    include_mean_load: bool = False
    source_bounds_policy: SourceBoundsPolicy = "raise"
    source_frequency_cap_hz: float | None = None
    fluid_density: float | None = None
    fluid_dynamicviscosity: float | None = None
    rotation_axis: Sequence[float] | None = None
    rotation_axis_offset: Sequence[float] | None = None
    inflow_vec: Sequence[float] | None = None
    debug: bool = False
    log_file: str | Path | None = None
    python_path_entries: Sequence[str | Path] = ()
    include_default_python_path: bool = True
    geometry_audit_file: str | Path | None = None
    write_geometry_audit: bool = True
    coupling_mode: QBladeCouplingMode = "sil-vorlap"
    qblade_execution_mode: QBladeExecutionMode = "step-loop"
    record_sil_vorlap_history: bool = False
    duration_s: float | None = None
    timestep_s: float | None = None
    steps: int | None = None
    rpm: float | None = None
    wind_speed: float | None = None
    azimuth_deg: float | None = None
    store_from_s: float | None = None
    store_replay: bool | None = None
    threads: int = 1
    use_opencl: bool = False
    opencl_device: int = 1
    opencl_group_size: int = 32
    store_project: bool = False
    checkpoint_every: int = 0
    checkpoint_dir: str | Path | None = None
    resume_from_checkpoint: bool = False
    resume_checkpoint: str | Path | None = None
    resume_initialize: bool = False
    allow_partial: bool = False
    sil_constant_force_specs: Sequence[str] = ()
    case_metadata_path: str | Path | None = None
    progress_every: int = 100
    manage_loader_env: bool = True
    patch_original_model: bool = False

    @classmethod
    def from_gui_state(cls, state: Mapping[str, Any]) -> "QBladeBridgeConfig":
        """Build a typed config from GUI-state dictionaries or widget-model values."""
        return cls(
            sim_path=_state_value(state, "sim_path", "sim"),
            qblade_root=_state_value(state, "qblade_root"),
            output_dir=_state_value(state, "output_dir", "run_output_dir"),
            airfoil_dir=_state_value(state, "airfoil_dir", "airfoils"),
            qblade_binary=_optional_str(_state_value(state, "qblade_binary", "qblade_executable")),
            python_executable=_optional_str(_state_value(state, "python_executable", "python")),
            run_root=_optional_str(_state_value(state, "run_root")),
            result_name=_optional_str(_state_value(state, "result_name")),
            library_stem=str(_state_value(state, "library_stem", default="libvorlap_qblade_bridge")),
            function_name=str(_state_value(state, "function_name", default="update")),
            parameter_file=_optional_str(_state_value(state, "parameter_file")),
            sample_step=_optional_float(_state_value(state, "sample_step"), field="sample_step"),
            default_airfoil_id=str(_state_value(state, "default_airfoil_id", default="default")),
            include_struts=_as_bool(_state_value(state, "include_struts", default=True), field="include_struts"),
            include_tower=_as_bool(_state_value(state, "include_tower", default=False), field="include_tower"),
            node_source=str(_state_value(state, "node_source", default="structural")),  # type: ignore[arg-type]
            include_tower_nodes=_as_bool(
                _state_value(state, "include_tower_nodes", default=False),
                field="include_tower_nodes",
            ),
            tower_airfoil_id=str(_state_value(state, "tower_airfoil_id", default="cylinder")),
            n_freq_depth=_optional_int(_state_value(state, "n_freq_depth"), field="n_freq_depth"),
            force_scale=float(_state_value(state, "force_scale", default=1.0)),
            diagnostic_force_scale=_as_bool(
                _state_value(state, "diagnostic_force_scale", default=False),
                field="diagnostic_force_scale",
            ),
            include_mean_load=_as_bool(
                _state_value(state, "include_mean_load", default=False),
                field="include_mean_load",
            ),
            source_bounds_policy=_normalize_source_bounds_policy(
                _state_value(state, "source_bounds_policy", default="raise")
            ),
            source_frequency_cap_hz=_optional_float(
                _state_value(state, "source_frequency_cap_hz"),
                field="source_frequency_cap_hz",
            ),
            fluid_density=_optional_float(_state_value(state, "fluid_density"), field="fluid_density"),
            fluid_dynamicviscosity=_optional_float(
                _state_value(state, "fluid_dynamicviscosity", "fluid_dynamic_viscosity"),
                field="fluid_dynamicviscosity",
            ),
            rotation_axis=_optional_float_vector(_state_value(state, "rotation_axis"), field="rotation_axis"),
            rotation_axis_offset=_optional_float_vector(
                _state_value(state, "rotation_axis_offset"),
                field="rotation_axis_offset",
            ),
            inflow_vec=_optional_float_vector(_state_value(state, "inflow_vec"), field="inflow_vec"),
            debug=_as_bool(_state_value(state, "debug", default=False), field="debug"),
            log_file=_optional_str(_state_value(state, "log_file")),
            python_path_entries=_sequence(_state_value(state, "python_path_entries", "python_path_entry")),
            include_default_python_path=_as_bool(
                _state_value(state, "include_default_python_path", default=True),
                field="include_default_python_path",
            ),
            geometry_audit_file=_optional_str(_state_value(state, "geometry_audit_file")),
            write_geometry_audit=_as_bool(
                _state_value(state, "write_geometry_audit", default=True),
                field="write_geometry_audit",
            ),
            coupling_mode=str(_state_value(state, "coupling_mode", "coupling", default="sil-vorlap")),  # type: ignore[arg-type]
            qblade_execution_mode=str(  # type: ignore[arg-type]
                _state_value(state, "qblade_execution_mode", default="step-loop")
            ),
            record_sil_vorlap_history=_as_bool(
                _state_value(state, "record_sil_vorlap_history", default=False),
                field="record_sil_vorlap_history",
            ),
            duration_s=_optional_float(_state_value(state, "duration_s", "duration"), field="duration_s"),
            timestep_s=_optional_float(_state_value(state, "timestep_s", "timestep"), field="timestep_s"),
            steps=_optional_int(_state_value(state, "steps", "qblade_steps"), field="steps"),
            rpm=_optional_float(_state_value(state, "rpm"), field="rpm"),
            wind_speed=_optional_float(_state_value(state, "wind_speed", "mean_inflow"), field="wind_speed"),
            azimuth_deg=_optional_float(_state_value(state, "azimuth_deg", "azimuth"), field="azimuth_deg"),
            store_from_s=_optional_float(_state_value(state, "store_from_s", "store_from"), field="store_from_s"),
            store_replay=_optional_bool(_state_value(state, "store_replay"), field="store_replay"),
            threads=int(_state_value(state, "threads", default=1)),
            use_opencl=_as_bool(_state_value(state, "use_opencl", default=False), field="use_opencl"),
            opencl_device=int(_state_value(state, "opencl_device", "device", default=1)),
            opencl_group_size=int(_state_value(state, "opencl_group_size", "group_size", default=32)),
            store_project=_as_bool(_state_value(state, "store_project", default=False), field="store_project"),
            checkpoint_every=int(_state_value(state, "checkpoint_every", default=0)),
            checkpoint_dir=_optional_str(_state_value(state, "checkpoint_dir")),
            resume_from_checkpoint=_as_bool(
                _state_value(state, "resume_from_checkpoint", default=False),
                field="resume_from_checkpoint",
            ),
            resume_checkpoint=_optional_str(_state_value(state, "resume_checkpoint")),
            resume_initialize=_as_bool(
                _state_value(state, "resume_initialize", default=False),
                field="resume_initialize",
            ),
            allow_partial=_as_bool(_state_value(state, "allow_partial", default=False), field="allow_partial"),
            sil_constant_force_specs=_sequence(_state_value(state, "sil_constant_force_specs", "sil_constant_force")),
            case_metadata_path=_optional_str(_state_value(state, "case_metadata_path", "case_metadata")),
            progress_every=int(_state_value(state, "progress_every", default=100)),
            manage_loader_env=_as_bool(
                _state_value(state, "manage_loader_env", default=True),
                field="manage_loader_env",
            ),
            patch_original_model=_as_bool(
                _state_value(state, "patch_original_model", default=False),
                field="patch_original_model",
            ),
        ).validated()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QBladeBridgeConfig":
        return cls.from_gui_state(data)

    @classmethod
    def from_json(cls, text: str) -> "QBladeBridgeConfig":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("QBlade bridge config JSON must contain an object.")
        return cls.from_dict(data)

    def as_dict(self) -> dict[str, object]:
        return {
            "sim_path": str(self.sim_path),
            "qblade_root": str(self.qblade_root),
            "output_dir": str(self.output_dir),
            "airfoil_dir": str(self.airfoil_dir),
            "qblade_binary": None if self.qblade_binary is None else str(self.qblade_binary),
            "python_executable": None if self.python_executable is None else str(self.python_executable),
            "run_root": None if self.run_root is None else str(self.run_root),
            "result_name": self.result_name,
            "library_stem": self.library_stem,
            "function_name": self.function_name,
            "parameter_file": None if self.parameter_file is None else str(self.parameter_file),
            "sample_step": self.sample_step,
            "default_airfoil_id": self.default_airfoil_id,
            "include_struts": self.include_struts,
            "include_tower": self.include_tower,
            "node_source": self.node_source,
            "include_tower_nodes": self.include_tower_nodes,
            "tower_airfoil_id": self.tower_airfoil_id,
            "n_freq_depth": self.n_freq_depth,
            "force_scale": self.force_scale,
            "diagnostic_force_scale": self.diagnostic_force_scale,
            "include_mean_load": self.include_mean_load,
            "source_bounds_policy": self.source_bounds_policy,
            "source_frequency_cap_hz": self.source_frequency_cap_hz,
            "fluid_density": self.fluid_density,
            "fluid_dynamicviscosity": self.fluid_dynamicviscosity,
            "rotation_axis": None if self.rotation_axis is None else [float(value) for value in self.rotation_axis],
            "rotation_axis_offset": None
            if self.rotation_axis_offset is None
            else [float(value) for value in self.rotation_axis_offset],
            "inflow_vec": None if self.inflow_vec is None else [float(value) for value in self.inflow_vec],
            "debug": self.debug,
            "log_file": None if self.log_file is None else str(self.log_file),
            "python_path_entries": [str(path) for path in self.python_path_entries],
            "include_default_python_path": self.include_default_python_path,
            "geometry_audit_file": None if self.geometry_audit_file is None else str(self.geometry_audit_file),
            "write_geometry_audit": self.write_geometry_audit,
            "coupling_mode": self.coupling_mode,
            "qblade_execution_mode": self.qblade_execution_mode,
            "record_sil_vorlap_history": self.record_sil_vorlap_history,
            "duration_s": self.duration_s,
            "timestep_s": self.timestep_s,
            "steps": self.steps,
            "rpm": self.rpm,
            "wind_speed": self.wind_speed,
            "azimuth_deg": self.azimuth_deg,
            "store_from_s": self.store_from_s,
            "store_replay": self.store_replay,
            "threads": self.threads,
            "use_opencl": self.use_opencl,
            "opencl_device": self.opencl_device,
            "opencl_group_size": self.opencl_group_size,
            "store_project": self.store_project,
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_dir": None if self.checkpoint_dir is None else str(self.checkpoint_dir),
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "resume_checkpoint": None if self.resume_checkpoint is None else str(self.resume_checkpoint),
            "resume_initialize": self.resume_initialize,
            "allow_partial": self.allow_partial,
            "sil_constant_force_specs": list(self.sil_constant_force_specs),
            "case_metadata_path": None if self.case_metadata_path is None else str(self.case_metadata_path),
            "progress_every": self.progress_every,
            "manage_loader_env": self.manage_loader_env,
            "patch_original_model": self.patch_original_model,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"

    def validation_errors(self, *, check_paths: bool = False) -> tuple[str, ...]:
        errors: list[str] = []
        for field_name, value in (
            ("sim_path", self.sim_path),
            ("qblade_root", self.qblade_root),
            ("output_dir", self.output_dir),
            ("airfoil_dir", self.airfoil_dir),
        ):
            if value in (None, ""):
                errors.append(f"{field_name} is required.")
        if self.node_source not in ("structural", "converted"):
            errors.append(f"node_source must be 'structural' or 'converted'; got {self.node_source!r}.")
        if self.coupling_mode not in ("sil-vorlap", "sil-constant-force", "qblade-external", "none"):
            errors.append(f"Unsupported coupling_mode: {self.coupling_mode!r}.")
        if self.qblade_execution_mode not in ("step-loop", "run-full-simulation"):
            errors.append(f"Unsupported qblade_execution_mode: {self.qblade_execution_mode!r}.")
        if self.qblade_execution_mode == "run-full-simulation" and self.coupling_mode != "qblade-external":
            errors.append("qblade_execution_mode='run-full-simulation' requires coupling_mode='qblade-external'.")
        try:
            _normalize_source_bounds_policy(self.source_bounds_policy)
        except ValueError as exc:
            errors.append(str(exc))
        for field_name, value in (
            ("fluid_density", self.fluid_density),
            ("fluid_dynamicviscosity", self.fluid_dynamicviscosity),
        ):
            if value is not None:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    errors.append(f"{field_name} must be numeric when provided.")
                    continue
                if not math.isfinite(number) or number <= 0.0:
                    errors.append(f"{field_name} must be positive and finite when provided.")
        for field_name, value in (
            ("rotation_axis", self.rotation_axis),
            ("rotation_axis_offset", self.rotation_axis_offset),
            ("inflow_vec", self.inflow_vec),
        ):
            try:
                vector = _optional_float_vector(value, field=field_name)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if field_name in {"rotation_axis", "inflow_vec"} and vector is not None:
                norm = math.sqrt(sum(component * component for component in vector))
                if norm <= 1.0e-12:
                    errors.append(f"{field_name} must have non-zero magnitude when provided.")
        try:
            validate_force_scale(float(self.force_scale), bool(self.diagnostic_force_scale))
        except ValueError as exc:
            errors.append(str(exc))
        if self.n_freq_depth is not None and int(self.n_freq_depth) < 1:
            errors.append("n_freq_depth must be positive when provided.")
        if self.source_frequency_cap_hz is not None:
            cap = float(self.source_frequency_cap_hz)
            if not math.isfinite(cap) or cap <= 0.0:
                errors.append("source_frequency_cap_hz must be positive and finite when provided.")
        if self.steps is not None and int(self.steps) < 1:
            errors.append("steps must be positive when provided.")
        if self.duration_s is not None and float(self.duration_s) <= 0.0:
            errors.append("duration_s must be positive when provided.")
        if self.timestep_s is not None and float(self.timestep_s) <= 0.0:
            errors.append("timestep_s must be positive when provided.")
        if self.sample_step is not None and float(self.sample_step) <= 0.0:
            errors.append("sample_step must be positive when provided.")
        for field_name, value in (
            ("wind_speed", self.wind_speed),
            ("rpm", self.rpm),
            ("azimuth_deg", self.azimuth_deg),
        ):
            if value is not None and not math.isfinite(float(value)):
                errors.append(f"{field_name} must be finite when provided.")
        if self.wind_speed is not None and float(self.wind_speed) < 0.0:
            errors.append("wind_speed must be non-negative when provided.")
        if self.store_from_s is not None:
            if not math.isfinite(float(self.store_from_s)):
                errors.append("store_from_s must be finite when provided.")
            elif float(self.store_from_s) < 0.0:
                errors.append("store_from_s must be non-negative when provided.")
        if int(self.threads) < 1:
            errors.append("threads must be positive.")
        if int(self.opencl_group_size) < 1:
            errors.append("opencl_group_size must be positive.")
        if int(self.checkpoint_every) < 0:
            errors.append("checkpoint_every must be non-negative.")
        if int(self.progress_every) < 0:
            errors.append("progress_every must be non-negative.")
        if self.coupling_mode == "sil-constant-force" and not self.sil_constant_force_specs:
            errors.append("sil-constant-force mode requires at least one force specification.")
        if self.coupling_mode != "sil-constant-force" and self.sil_constant_force_specs:
            errors.append("sil_constant_force_specs require coupling_mode='sil-constant-force'.")

        if check_paths:
            if not Path(self.sim_path).is_file():
                errors.append(f"QBlade simulation file does not exist: {self.sim_path}")
            if not Path(self.qblade_root).is_dir():
                errors.append(f"QBlade root directory does not exist: {self.qblade_root}")
            if not Path(self.airfoil_dir).is_dir():
                errors.append(f"VorLap airfoil directory does not exist: {self.airfoil_dir}")
            output_error = _output_dir_validation_error(self.output_dir)
            if output_error is not None:
                errors.append(output_error)
            if self.qblade_binary is not None and not Path(self.qblade_binary).is_file():
                errors.append(f"QBlade executable does not exist: {self.qblade_binary}")
            errors.extend(_resume_checkpoint_validation_errors(self))
        return tuple(errors)

    def validated(self, *, check_paths: bool = False) -> "QBladeBridgeConfig":
        errors = self.validation_errors(check_paths=check_paths)
        if errors:
            raise ValueError("; ".join(errors))
        source_bounds_policy = _normalize_source_bounds_policy(self.source_bounds_policy)
        replacements: dict[str, object] = {}
        if source_bounds_policy != self.source_bounds_policy:
            replacements["source_bounds_policy"] = source_bounds_policy
        if self.fluid_density is not None:
            replacements["fluid_density"] = float(self.fluid_density)
        if self.fluid_dynamicviscosity is not None:
            replacements["fluid_dynamicviscosity"] = float(self.fluid_dynamicviscosity)
        if self.source_frequency_cap_hz is not None:
            replacements["source_frequency_cap_hz"] = float(self.source_frequency_cap_hz)
        for field_name in ("rotation_axis", "rotation_axis_offset", "inflow_vec"):
            vector = _optional_float_vector(getattr(self, field_name), field=field_name)
            if vector is not None:
                replacements[field_name] = vector
        return replace(self, **replacements) if replacements else self

    def effective_steps(self) -> int | None:
        if self.steps is not None:
            return int(self.steps)
        if self.duration_s is not None and self.timestep_s is not None:
            return max(1, int(math.ceil(float(self.duration_s) / float(self.timestep_s))))
        return None

    def to_case_prep_request(self) -> QBladeCasePrepRequest:
        self.validated()
        return QBladeCasePrepRequest(
            sim_path=self.sim_path,
            airfoil_dir=self.airfoil_dir,
            library_stem=self.library_stem,
            function_name=self.function_name,
            parameter_file=self.parameter_file,
            sample_step=self.sample_step,
            default_airfoil_id=self.default_airfoil_id,
            include_struts=self.include_struts,
            include_tower=self.include_tower,
            node_source=self.node_source,
            include_tower_nodes=self.include_tower_nodes,
            tower_airfoil_id=self.tower_airfoil_id,
            n_freq_depth=self.n_freq_depth,
            force_scale=self.force_scale,
            diagnostic_force_scale=self.diagnostic_force_scale,
            include_mean_load=self.include_mean_load,
            source_bounds_policy=self.source_bounds_policy,
            source_frequency_cap_hz=self.source_frequency_cap_hz,
            fluid_density=self.fluid_density,
            fluid_dynamicviscosity=self.fluid_dynamicviscosity,
            rotation_axis=self.rotation_axis,
            rotation_axis_offset=self.rotation_axis_offset,
            inflow_vec=self.inflow_vec,
            debug=self.debug,
            log_file=self.log_file,
            python_path_entries=self.python_path_entries,
            include_default_python_path=self.include_default_python_path,
            geometry_audit_file=self.geometry_audit_file,
            write_geometry_audit=self.write_geometry_audit,
        )

    def to_run_request(self) -> QBladeRunRequest:
        self.validated()
        return QBladeRunRequest(
            sim_path=self.sim_path,
            qblade_root=self.qblade_root,
            output_dir=self.output_dir,
            result_name=self.result_name,
            steps=self.effective_steps(),
            threads=self.threads,
            device=self.opencl_device,
            group_size=self.opencl_group_size,
            use_opencl=self.use_opencl,
            store_project=self.store_project,
            checkpoint_every=self.checkpoint_every,
            checkpoint_dir=self.checkpoint_dir,
            resume_from_checkpoint=self.resume_from_checkpoint,
            resume_checkpoint=self.resume_checkpoint,
            resume_initialize=self.resume_initialize,
            allow_partial=self.allow_partial,
            sil_vorlap=self.coupling_mode == "sil-vorlap",
            sil_constant_force_specs=tuple(self.sil_constant_force_specs),
            coupling_mode=self.coupling_mode,
            qblade_execution_mode=self.qblade_execution_mode,
            case_metadata_path=self.case_metadata_path,
            progress_every=self.progress_every,
            record_sil_vorlap_history=self.record_sil_vorlap_history,
            manage_loader_env=self.manage_loader_env,
        )

    def to_cli_args(self, action: BridgeCliAction) -> list[str]:
        if action == "prepare":
            return self._prepare_cli_args()
        if action == "run":
            return self._run_cli_args()
        raise ValueError(f"Unsupported bridge CLI action: {action!r}.")

    def _prepare_cli_args(self) -> list[str]:
        self.validated()
        args = [
            "--sim",
            str(self.sim_path),
            "--airfoils",
            str(self.airfoil_dir),
            "--library-stem",
            self.library_stem,
            "--function-name",
            self.function_name,
            "--default-airfoil-id",
            self.default_airfoil_id,
            "--node-source",
            self.node_source,
            "--tower-airfoil-id",
            self.tower_airfoil_id,
            "--force-scale",
            str(self.force_scale),
        ]
        optional_values = [
            ("--parameter-file", self.parameter_file),
            ("--sample-step", self.sample_step),
            ("--n-freq-depth", self.n_freq_depth),
            ("--log-file", self.log_file),
            ("--geometry-audit-file", self.geometry_audit_file),
            ("--fluid-density", self.fluid_density),
            ("--fluid-dynamicviscosity", self.fluid_dynamicviscosity),
        ]
        for flag, value in optional_values:
            if value is not None:
                args.extend([flag, str(value)])
        for flag, value in (
            ("--rotation-axis", self.rotation_axis),
            ("--rotation-axis-offset", self.rotation_axis_offset),
            ("--inflow-vec", self.inflow_vec),
        ):
            if value is not None:
                args.extend([flag, *(str(float(component)) for component in value)])
        args.append("--include-struts" if self.include_struts else "--no-include-struts")
        args.append("--include-tower" if self.include_tower else "--no-include-tower")
        args.append("--include-tower-nodes" if self.include_tower_nodes else "--no-include-tower-nodes")
        if self.diagnostic_force_scale:
            args.append("--diagnostic-force-scale")
        args.append("--include-mean-load" if self.include_mean_load else "--no-include-mean-load")
        args.extend(["--source-bounds-policy", self.source_bounds_policy])
        args.append("--debug" if self.debug else "--no-debug")
        for entry in self.python_path_entries:
            args.extend(["--python-path-entry", str(entry)])
        if not self.include_default_python_path:
            args.append("--no-default-python-path")
        if not self.write_geometry_audit:
            args.append("--skip-geometry-audit")
        return args

    def _run_cli_args(self) -> list[str]:
        self.validated()
        args = [
            "--sim",
            str(self.sim_path),
            "--qblade-root",
            str(self.qblade_root),
            "--output-dir",
            str(self.output_dir),
            "--threads",
            str(self.threads),
            "--device",
            str(self.opencl_device),
            "--group-size",
            str(self.opencl_group_size),
            "--coupling-mode",
            self.coupling_mode,
            "--progress-every",
            str(self.progress_every),
        ]
        optional_values = [
            ("--result-name", self.result_name),
            ("--steps", self.effective_steps()),
            ("--checkpoint-every", self.checkpoint_every if self.checkpoint_every else None),
            ("--checkpoint-dir", self.checkpoint_dir),
            ("--resume-checkpoint", self.resume_checkpoint),
            ("--case-metadata", self.case_metadata_path),
        ]
        for flag, value in optional_values:
            if value is not None:
                args.extend([flag, str(value)])
        if self.use_opencl:
            args.append("--use-opencl")
        if self.store_project:
            args.append("--store-project")
        if self.resume_from_checkpoint:
            args.append("--resume-from-checkpoint")
        if self.resume_initialize:
            args.append("--resume-initialize")
        if self.allow_partial:
            args.append("--allow-partial")
        if self.qblade_execution_mode == "run-full-simulation":
            args.append("--native-run-full-simulation")
        if self.coupling_mode == "sil-vorlap":
            args.append("--sil-vorlap")
        for spec in self.sil_constant_force_specs:
            args.extend(["--sil-constant-force", str(spec)])
        if self.record_sil_vorlap_history:
            args.append("--record-sil-vorlap-history")
        return args


__all__ = [
    "BridgeCliAction",
    "QBladeBridgeConfig",
    "SourceBoundsPolicy",
]
