"""QBlade external-library config and ASCII patch helpers."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .qblade_external_table import (
    ExternalLibraryTableSpec,
    build_external_library_table_spec,
    build_hvawt_response_table_spec,
)
from .qblade_files import format_qblade_keyword_line, parse_qblade_keyword_line
from .qblade_nodes import infer_geometry_inclusions


def _format_keyword_line(value: str, keyword: str, comment: str) -> str:
    """Render a QBlade keyword/value line with the usual ASCII alignment."""
    return format_qblade_keyword_line(value, keyword, comment)


def build_qblade_external_config(
    *,
    sim_path: str,
    airfoil_dir: str,
    node_ids: Sequence[str],
    library_stem: str = "libvorlap_qblade_bridge",
    function_name: str = "update",
    parameter_file: Optional[str] = None,
    sample_step: Optional[float] = None,
    default_airfoil_id: str = "default",
    include_struts: bool = True,
    include_tower: bool = True,
    tower_airfoil_id: str = "cylinder",
    n_freq_depth: Optional[int] = None,
    force_scale: float = 1.0,
    include_mean_load: bool = False,
    source_bounds_policy: str = "warn",
    source_frequency_cap_hz: Optional[float] = None,
    response_model: Optional[Mapping[str, Any]] = None,
    hvawt_response_model: Optional[Mapping[str, Any]] = None,
    use_qblade_local_inflow: bool = False,
    fluid_density: Optional[float] = None,
    fluid_dynamicviscosity: Optional[float] = None,
    rotation_axis: Optional[Sequence[float]] = None,
    rotation_axis_offset: Optional[Sequence[float]] = None,
    inflow_vec: Optional[Sequence[float]] = None,
    source_parameter_dir: Optional[str] = None,
    python_path_entries: Optional[Sequence[str]] = None,
    debug: bool = False,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the JSON config consumed by the Python runtime and C++ bridge."""
    if response_model is not None and hvawt_response_model is not None:
        raise ValueError(
            "The single-mode response_model and two-mode hvawt_response_model are mutually exclusive."
        )
    if not isinstance(use_qblade_local_inflow, bool):
        raise ValueError("use_qblade_local_inflow must be a JSON boolean.")
    table_spec = (
        build_hvawt_response_table_spec(node_ids)
        if hvawt_response_model is not None or use_qblade_local_inflow
        else build_external_library_table_spec(node_ids)
    )
    include_struts, include_tower = infer_geometry_inclusions(node_ids, include_struts, include_tower)
    config: Dict[str, Any] = {
        # Keep these paths as provided so case-prep can store model-relative
        # values. Runtime resolution is handled by from_qblade_config().
        "sim_path": str(sim_path),
        "airfoil_dir": str(airfoil_dir),
        "library_stem": str(library_stem),
        "function_name": str(function_name),
        "default_airfoil_id": str(default_airfoil_id),
        "include_struts": bool(include_struts),
        "include_tower": bool(include_tower),
        "tower_airfoil_id": str(tower_airfoil_id),
        "force_scale": float(force_scale),
        "include_mean_load": bool(include_mean_load),
        "source_bounds_policy": str(source_bounds_policy),
        "debug": bool(debug),
        "node_ids": list(table_spec.node_ids),
        "swap_size": int(table_spec.swap_size),
        "swap_layout": [
            {"name": name, "offset": spec.offset, "shape": list(spec.shape)}
            for name, spec in table_spec.swap_layout.items()
        ],
        "external_input_rows": [
            {"index": idx, "variable": variable}
            for idx, variable in table_spec.input_rows
        ],
        "external_output_rows": [
            {
                "index": idx,
                "action": action,
                "id": target_id,
                "position": position,
                "direction": direction,
                "local": local,
            }
            for idx, action, target_id, position, direction, local in table_spec.output_rows
        ],
    }
    if parameter_file is not None:
        config["parameter_file"] = str(parameter_file)
    if use_qblade_local_inflow:
        config["use_qblade_local_inflow"] = True
    if sample_step is not None:
        config["sample_step"] = float(sample_step)
    if n_freq_depth is not None:
        config["n_freq_depth"] = int(n_freq_depth)
    if source_frequency_cap_hz is not None:
        source_frequency_cap_hz = float(source_frequency_cap_hz)
        if not math.isfinite(source_frequency_cap_hz) or source_frequency_cap_hz <= 0.0:
            raise ValueError("source_frequency_cap_hz must be finite and positive when provided.")
        config["source_frequency_cap_hz"] = source_frequency_cap_hz
    if response_model is not None:
        if not isinstance(response_model, Mapping):
            raise TypeError("response_model must be a mapping when provided.")
        # Round-trip through JSON here so case preparation fails before QBlade
        # is patched if a model configuration is not portable/checkpointable.
        try:
            response_model_copy = json.loads(json.dumps(dict(response_model), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("response_model must contain finite JSON-serializable values.") from exc
        config["response_model"] = response_model_copy
    if hvawt_response_model is not None:
        if not isinstance(hvawt_response_model, Mapping):
            raise TypeError("hvawt_response_model must be a mapping when provided.")
        try:
            hvawt_response_copy = json.loads(
                json.dumps(dict(hvawt_response_model), allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "hvawt_response_model must contain finite JSON-serializable values."
            ) from exc
        config["hvawt_response_model"] = hvawt_response_copy
    if fluid_density is not None:
        config["fluid_density"] = float(fluid_density)
    if fluid_dynamicviscosity is not None:
        config["fluid_dynamicviscosity"] = float(fluid_dynamicviscosity)
    for key, value in (
        ("rotation_axis", rotation_axis),
        ("rotation_axis_offset", rotation_axis_offset),
        ("inflow_vec", inflow_vec),
    ):
        if value is not None:
            config[key] = [float(component) for component in value]
    if source_parameter_dir is not None:
        config["source_parameter_dir"] = str(source_parameter_dir)
    if python_path_entries is not None:
        config["python_path_entries"] = [str(entry) for entry in python_path_entries]
    if log_file is not None:
        config["log_file"] = str(log_file)
    return config


def write_qblade_external_config(path: str, config: Mapping[str, Any]) -> str:
    """Write the runtime config to JSON with stable formatting."""
    out_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")
    return out_path


def _external_in_table_text(rows: Iterable[Tuple[int, str]]) -> str:
    lines = [
        "EXTERNAL_1_IN",
        "SWAP DATA",
    ]
    for idx, variable in rows:
        lines.append(f"{idx}    \"{variable}\"")
    return "\n".join(lines)


def _external_out_table_text(rows: Iterable[Tuple[int, str, str, float, str, bool]]) -> str:
    lines = [
        "EXTERNAL_1_OUT",
        "SWAP  ACTION     ID     POS  DIR  LOCAL",
    ]
    for idx, action, target_id, position, direction, local in rows:
        local_text = "true" if local else "false"
        lines.append(f"{idx}    {action:<9} {target_id:<8} {position:.6f} {direction:<4} {local_text}")
    return "\n".join(lines)


def _replace_keyword_line(lines: List[str], keyword: str, value: str, comment: str) -> bool:
    for i, line in enumerate(lines):
        parsed = parse_qblade_keyword_line(line.rstrip("\n"))
        if parsed is not None and parsed[0] == keyword:
            lines[i] = _format_keyword_line(value, keyword, comment)
            return True
    return False


def _qblade_section_header(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("----------------------------------------") and stripped.endswith("---")


def _replace_qblade_section(lines: List[str], header: str, replacement: Sequence[str]) -> List[str]:
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(header))
    except StopIteration:
        return [*lines, *replacement]

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if _qblade_section_header(lines[i]):
            end = i
            break
    return [*lines[:start], *replacement, *lines[end:]]


_EXTERNAL_INPUT_ROW_RE = re.compile(r'^\s*\d+\s+"[^"]+"\s*$')
_EXTERNAL_OUTPUT_ROW_RE = re.compile(
    r"^\s*\d+\s+\S+\s+\S+\s+[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s+\S+\s+(?:true|false)\s*$",
    re.IGNORECASE,
)


def _external_table_end(lines: Sequence[str], start: int, *, header: str) -> int:
    if start + 1 >= len(lines):
        return start + 1

    row_pattern: re.Pattern[str]
    if header == "EXTERNAL_1_IN":
        expected_subheader = "SWAP DATA"
        row_pattern = _EXTERNAL_INPUT_ROW_RE
    elif header == "EXTERNAL_1_OUT":
        expected_subheader = "SWAP  ACTION"
        row_pattern = _EXTERNAL_OUTPUT_ROW_RE
    else:  # pragma: no cover - helper is currently used for EXTERNAL_1 only.
        raise ValueError(f"Unsupported external table header: {header}")

    if not lines[start + 1].strip().startswith(expected_subheader):
        raise ValueError(f"Cannot safely replace {header}: unexpected table subheader {lines[start + 1]!r}.")

    end = start + 2
    while end < len(lines):
        line = lines[end]
        stripped = line.strip()
        if not stripped or stripped.startswith("EXTERNAL_") or _qblade_section_header(line):
            break
        if not row_pattern.match(line):
            raise ValueError(f"Cannot safely replace {header}: unexpected table row {line!r}.")
        end += 1
    return end


def _remove_external_table(lines: List[str], header: str) -> List[str]:
    result = list(lines)
    while True:
        try:
            start = next(i for i, line in enumerate(result) if line.strip() == header)
        except StopIteration:
            return result
        end = _external_table_end(result, start, header=header)
        del result[start:end]


def strip_qblade_external_library_section(text: str) -> str:
    """Return a turbine definition without scenario-specific library wiring."""

    lines = text.splitlines()
    external_header = (
        "----------------------------------------External Libraries"
        "-----------------------------------------------------------"
    )
    matches = [
        index for index, line in enumerate(lines) if line.startswith(external_header)
    ]
    if len(matches) > 1:
        raise ValueError("QBlade turbine definition has duplicate external-library sections.")
    if matches:
        start = matches[0]
        end = len(lines)
        for index in range(start + 1, len(lines)):
            if _qblade_section_header(lines[index]):
                end = index
                break
        del lines[start:end]
    return "\n".join(lines).rstrip() + "\n"


def strip_qblade_external_tables(text: str) -> str:
    """Return a structural definition without scenario-specific swap tables."""

    lines = text.splitlines()
    lines = _remove_external_table(lines, "EXTERNAL_1_OUT")
    lines = _remove_external_table(lines, "EXTERNAL_1_IN")
    return "\n".join(lines).rstrip() + "\n"


def patch_qblade_turbine_definition(
    text: str,
    *,
    library_stem: str,
    function_name: str,
    swap_size: int,
    parameter_file: str,
) -> str:
    """Patch a turbine definition ASCII file for VorLap external-library use."""
    lines = text.splitlines()
    _replace_keyword_line(
        lines,
        "CONTROLLERTYPE",
        "0",
        "the type of turbine controller 0 = none, 1 = BLADED, 2 = DTU, 3 = TUB",
    )
    _replace_keyword_line(lines, "CONTROLLERFILE", "", "the controller file name, WITHOUT file ending (.dll or .so ) - leave blank if unused")
    _replace_keyword_line(lines, "PARAMETERFILE", "", "the controller parameter file name (leave blank if unused)")

    external_header = "----------------------------------------External Libraries-----------------------------------------------------------"
    external_section = [
        external_header,
        _format_keyword_line(library_stem, "LIBFILE_1", "the library file name, WITHOUT file ending (.dll or .so )"),
        _format_keyword_line(function_name, "LIBFUNCTION_1", "the library function name that should be called every timestep"),
        _format_keyword_line(str(int(swap_size)), "LIBARRAYSIZE_1", "the library swap array size for data exchange"),
        _format_keyword_line(parameter_file, "LIBPARAMETERFILE_1", "the library parameter file name (leave blank if unused)"),
        "",
    ]

    lines = _replace_qblade_section(lines, external_header, external_section)
    return "\n".join(lines).rstrip() + "\n"


def patch_qblade_structural_definition(
    text: str,
    *,
    table_spec: ExternalLibraryTableSpec,
) -> str:
    """Append or replace the EXTERNAL_1 input/output tables in a structural file."""
    lines = text.splitlines()
    lines = _remove_external_table(lines, "EXTERNAL_1_OUT")
    lines = _remove_external_table(lines, "EXTERNAL_1_IN")
    while lines and not lines[-1].strip():
        lines.pop()

    lines.extend(
        [
            "",
            _external_in_table_text(table_spec.input_rows),
            "",
            _external_out_table_text(table_spec.output_rows),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "build_qblade_external_config",
    "patch_qblade_structural_definition",
    "patch_qblade_turbine_definition",
    "strip_qblade_external_library_section",
    "strip_qblade_external_tables",
    "write_qblade_external_config",
]
