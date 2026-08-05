"""Runtime helpers for QBlade external-library integration."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .fileio import convert_qblade_to_vorlap_inputs, load_airfoil_fft
from .qblade_controller import QBladeController, SwapArrayAdapter, SwapBlockSpec
from .structs import AirfoilFFT, Component


_EPS = 1.0e-12
_QBLD_KEYVAL_RE = re.compile(r"^(?P<value>.*?)\s+(?P<keyword>[A-Z][A-Z0-9_]+)\s*(?:-.*)?$")
_BLD_NODE_RE = re.compile(r"^BLD_(?P<blade>\d+)_(?P<pos>[-+]?\d*\.?\d+)$")
_STR_NODE_RE = re.compile(r"^STR_(?P<strut>\d+)_(?P<blade>\d+)_(?P<pos>[-+]?\d*\.?\d+)$")
_TWR_NODE_RE = re.compile(r"^TWR_(?P<section>\d+)_(?P<pos>[-+]?\d*\.?\d+)$")
_TWR_SHORT_RE = re.compile(r"^TWR_(?P<pos>[-+]?\d*\.?\d+)$")


@dataclass(frozen=True)
class ExternalLibraryTableSpec:
    """Describe the QBlade swap layout and the corresponding table rows."""

    swap_layout: Dict[str, SwapBlockSpec]
    input_rows: List[Tuple[int, str]]
    output_rows: List[Tuple[int, str, str, float, str, bool]]
    swap_size: int
    node_ids: Tuple[str, ...]


def _resolve_path(base_file: str, candidate: str, extra_bases: Optional[Sequence[str]] = None) -> str:
    """Resolve a path relative to the parameter file directory or fallback bases."""
    candidate = str(candidate).strip()
    if not candidate:
        return ""
    if os.path.isabs(candidate):
        return candidate

    candidate_bases = [os.path.dirname(os.path.abspath(base_file))]
    for base in extra_bases or ():
        if not base:
            continue
        candidate_bases.append(os.path.abspath(str(base)))

    for base in candidate_bases:
        resolved = os.path.normpath(os.path.join(base, candidate))
        if os.path.exists(resolved):
            return resolved

    return os.path.normpath(os.path.join(candidate_bases[0], candidate))


def _format_keyword_line(value: str, keyword: str, comment: str) -> str:
    """Render a QBlade keyword/value line with the usual ASCII alignment."""
    return f"{str(value):<50}{keyword:<18} - {comment}"


def _normalize_shape(shape: Union[None, int, Sequence[int], Tuple[int, ...]]) -> Tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        return () if shape == 0 else (int(shape),)
    shape_tuple = tuple(int(dim) for dim in shape)
    if len(shape_tuple) == 1 and shape_tuple[0] == 0:
        return ()
    if any(dim <= 0 for dim in shape_tuple):
        raise ValueError("swap block dimensions must be positive.")
    return shape_tuple


def build_external_library_table_spec(
    node_ids: Sequence[str],
    *,
    time_block: str = "time",
    timestep_block: str = "timestep",
    azimuth_block: str = "azimuth_deg",
    velocity_block: str = "velocity",
    force_block: str = "force",
    velocity_offset: int = 3,
) -> ExternalLibraryTableSpec:
    """
    Build the default QBlade external-library swap layout.

    The layout is:
      0 -> Time [s]
      1 -> Timestep [-]
      2 -> LSS Azimuthal Pos. [deg]
      3.. -> per-node global velocity components
      ... -> per-node global force components
    """
    node_ids_tuple = tuple(str(node_id) for node_id in node_ids)
    node_count = len(node_ids_tuple)
    if node_count < 1:
        raise ValueError("node_ids must contain at least one entry.")

    layout = {
        time_block: SwapBlockSpec(offset=0, shape=()),
        timestep_block: SwapBlockSpec(offset=1, shape=()),
        azimuth_block: SwapBlockSpec(offset=2, shape=()),
        velocity_block: SwapBlockSpec(offset=velocity_offset, shape=(node_count, 3)),
        force_block: SwapBlockSpec(offset=velocity_offset + 3 * node_count, shape=(node_count, 3)),
    }

    input_rows: List[Tuple[int, str]] = [
        (0, "Time [s]"),
        (1, "Timestep [-]"),
        (2, "LSS Azimuthal Pos. [deg]"),
    ]
    for inode, node_id in enumerate(node_ids_tuple):
        for icomp, axis in enumerate(("X", "Y", "Z")):
            idx = velocity_offset + 3 * inode + icomp
            input_rows.append((idx, _external_input_velocity_name(node_id, axis)))

    output_rows: List[Tuple[int, str, str, float, str, bool]] = []
    force_offset = velocity_offset + 3 * node_count
    for inode, node_id in enumerate(node_ids_tuple):
        action_id, action_position = _external_output_target(node_id)
        for icomp, axis in enumerate(("X", "Y", "Z")):
            idx = force_offset + 3 * inode + icomp
            output_rows.append((idx, "ADDFORCE", action_id, action_position, axis, False))

    swap_size = force_offset + 3 * node_count
    return ExternalLibraryTableSpec(
        swap_layout=layout,
        input_rows=input_rows,
        output_rows=output_rows,
        swap_size=swap_size,
        node_ids=node_ids_tuple,
    )


def _node_position_label(node_id: str) -> str:
    """Render the human-readable QBlade variable suffix for a node id."""
    action_id, position = _external_output_target(node_id)
    if action_id.startswith("BLD_"):
        return f"{action_id} pos {position:.3f}"
    if action_id.startswith("STR_"):
        return f"{action_id} pos {position:.3f}"
    if action_id.startswith("TWR"):
        return f"TWR pos {position:.3f}"
    if action_id.startswith("TRQ"):
        return f"TRQ pos {position:.3f}"
    return f"{action_id} pos {position:.3f}"


def _external_input_velocity_name(node_id: str, axis: str) -> str:
    label = _node_position_label(node_id)
    return f"{axis}_g Vel. {label} [m/s]"


def _external_output_target(node_id: str) -> Tuple[str, float]:
    """Return `(target_id, normalized_position)` for a QBlade external action row."""
    comp_id, position = _component_id_and_position(node_id)
    return comp_id, position


def _component_id_and_position(node_id: str) -> Tuple[str, float]:
    """Parse a QBlade node id into `(component_id, normalized_position)`."""
    node = str(node_id).strip()

    match = _BLD_NODE_RE.match(node)
    if match:
        return f"BLD_{match.group('blade')}", float(match.group("pos"))

    match = _STR_NODE_RE.match(node)
    if match:
        return f"STR_{match.group('strut')}_{match.group('blade')}", float(match.group("pos"))

    match = _TWR_NODE_RE.match(node)
    if match:
        return f"TWR_{match.group('section')}", float(match.group("pos"))

    match = _TWR_SHORT_RE.match(node)
    if match:
        return "TWR_1", float(match.group("pos"))

    raise ValueError(f"Unsupported QBlade node id format: '{node_id}'.")


def load_airfoil_fft_directory(airfoil_dir: str) -> Dict[str, AirfoilFFT]:
    """Load all HDF5 airfoil FFT files in a directory."""
    if not os.path.isdir(airfoil_dir):
        raise FileNotFoundError(f"Airfoil directory does not exist: {airfoil_dir}")
    affts: Dict[str, AirfoilFFT] = {}
    for file in sorted(glob.glob(os.path.join(airfoil_dir, "*.h5"))):
        afft = load_airfoil_fft(file)
        affts[afft.name] = afft
    if not affts:
        raise FileNotFoundError(f"No airfoil FFT .h5 files were found in: {airfoil_dir}")
    if "default" not in affts:
        affts["default"] = next(iter(affts.values()))
    return affts


def _coerce_layout(layout: Mapping[str, Union[SwapBlockSpec, Sequence[int], Tuple[int, Sequence[int]], int]]) -> Dict[str, SwapBlockSpec]:
    coerced: Dict[str, SwapBlockSpec] = {}
    for name, spec in layout.items():
        if isinstance(spec, SwapBlockSpec):
            coerced[str(name)] = spec
        elif isinstance(spec, int):
            coerced[str(name)] = SwapBlockSpec(offset=spec)
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
            offset = int(spec[0])
            shape = _normalize_shape(spec[1])
            coerced[str(name)] = SwapBlockSpec(offset=offset, shape=shape)
        else:
            raise TypeError(f"Unsupported swap layout entry for '{name}'.")
    return coerced


def _infer_geometry_inclusions(
    node_ids: Sequence[str],
    include_struts: bool,
    include_tower: bool,
) -> Tuple[bool, bool]:
    """Force inclusion of geometry classes required by the requested node ids."""
    node_ids_tuple = tuple(str(node_id) for node_id in node_ids)
    requires_struts = any(node_id.startswith("STR_") for node_id in node_ids_tuple)
    requires_tower = any(node_id.startswith("TWR_") for node_id in node_ids_tuple)
    return bool(include_struts or requires_struts), bool(include_tower or requires_tower)


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
    source_parameter_dir: Optional[str] = None,
    debug: bool = False,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the JSON config consumed by the Python runtime and C++ bridge."""
    table_spec = build_external_library_table_spec(node_ids)
    include_struts, include_tower = _infer_geometry_inclusions(node_ids, include_struts, include_tower)
    config: Dict[str, Any] = {
        # Keep sim_path as provided so case-prep can store a model-relative path.
        # Runtime resolution is handled by from_qblade_config() via _resolve_path().
        "sim_path": str(sim_path),
        # Keep airfoil_dir as provided so case-prep can store a model-relative path.
        # Runtime resolution is handled by from_qblade_config() via _resolve_path().
        "airfoil_dir": str(airfoil_dir),
        "library_stem": str(library_stem),
        "function_name": str(function_name),
        "default_airfoil_id": str(default_airfoil_id),
        "include_struts": bool(include_struts),
        "include_tower": bool(include_tower),
        "tower_airfoil_id": str(tower_airfoil_id),
        "force_scale": float(force_scale),
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
    if sample_step is not None:
        config["sample_step"] = float(sample_step)
    if n_freq_depth is not None:
        config["n_freq_depth"] = int(n_freq_depth)
    if source_parameter_dir is not None:
        config["source_parameter_dir"] = os.path.abspath(str(source_parameter_dir))
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


def _load_config_json(param_file: str) -> Dict[str, Any]:
    with open(param_file, "r", encoding="utf-8") as f:
        return json.load(f)


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
    pattern = re.compile(rf"^(?P<value>.*?)\s+{re.escape(keyword)}\s*(?:-.*)?$")
    for i, line in enumerate(lines):
        if pattern.match(line.rstrip("\n")):
            lines[i] = _format_keyword_line(value, keyword, comment)
            return True
    return False


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

    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(external_header))
        lines = lines[:start]
    except StopIteration:
        pass
    lines.extend(external_section)
    return "\n".join(lines).rstrip() + "\n"


def patch_qblade_structural_definition(
    text: str,
    *,
    table_spec: ExternalLibraryTableSpec,
) -> str:
    """Append or replace the EXTERNAL_1 input/output tables in a structural file."""
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("EXTERNAL_1_IN"))
        lines = lines[:start]
    except StopIteration:
        pass
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


def _canonical_node_id(comp_id: str, position: float) -> str:
    return f"{comp_id}_{float(position):.6f}"


def _component_span_coordinate(shape_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(shape_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid component shape_xyz with shape {xyz.shape}.")
    if xyz.shape[0] == 1:
        return np.array([0.0], dtype=float)
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cumulative[-1])
    if total <= _EPS:
        return np.linspace(0.0, 1.0, xyz.shape[0], dtype=float)
    return cumulative / total


def _interp_along_span(values: np.ndarray, span: np.ndarray, query: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.interp(query, span, arr)
    if arr.ndim == 2:
        out = np.empty((query.shape[0], arr.shape[1]), dtype=float)
        for j in range(arr.shape[1]):
            out[:, j] = np.interp(query, span, arr[:, j])
        return out
    raise ValueError(f"Unsupported interpolation rank: {arr.ndim}")


def _resample_component(component: Component, positions: Sequence[float]) -> Component:
    query = np.asarray(positions, dtype=float).reshape(-1)
    query = np.clip(query, 0.0, 1.0)
    span = _component_span_coordinate(component.shape_xyz)

    shape_xyz = _interp_along_span(np.asarray(component.shape_xyz, dtype=float), span, query)
    chord = _interp_along_span(np.asarray(component.chord, dtype=float), span, query)
    twist = _interp_along_span(np.asarray(component.twist, dtype=float), span, query)
    thickness = _interp_along_span(np.asarray(component.thickness, dtype=float), span, query)
    offset = _interp_along_span(np.asarray(component.offset, dtype=float), span, query)

    airfoil_ids_raw = list(getattr(component, "airfoil_ids", []) or [])
    if not airfoil_ids_raw:
        airfoil_ids_raw = ["default"] * len(span)
    nearest = np.argmin(np.abs(span[None, :] - query[:, None]), axis=1)
    airfoil_ids = [str(airfoil_ids_raw[i]) for i in nearest]

    chord_vector = np.asarray(component.chord_vector, dtype=float)
    normal_vector = np.asarray(component.normal_vector, dtype=float)
    if chord_vector.ndim == 2 and chord_vector.shape[0] == len(span):
        chord_vector = _interp_along_span(chord_vector, span, query)
    else:
        chord_vector = np.zeros((query.shape[0], 3), dtype=float)
    if normal_vector.ndim == 2 and normal_vector.shape[0] == len(span):
        normal_vector = _interp_along_span(normal_vector, span, query)
    else:
        normal_vector = np.zeros((query.shape[0], 3), dtype=float)

    return Component(
        id=str(component.id),
        translation=np.asarray(component.translation, dtype=float).copy(),
        rotation=np.asarray(component.rotation, dtype=float).copy(),
        pitch=np.asarray(component.pitch, dtype=float).copy(),
        shape_xyz=np.asarray(shape_xyz, dtype=float),
        shape_xyz_global=np.zeros((query.shape[0], 3), dtype=float),
        chord=np.asarray(chord, dtype=float),
        twist=np.asarray(twist, dtype=float),
        thickness=np.asarray(thickness, dtype=float),
        offset=np.asarray(offset, dtype=float),
        airfoil_ids=airfoil_ids,
        chord_vector=np.asarray(chord_vector, dtype=float),
        normal_vector=np.asarray(normal_vector, dtype=float),
    )


def _build_selected_components(
    components: Sequence[Component], requested_node_ids: Sequence[str]
) -> Tuple[List[Component], Tuple[str, ...], Tuple[str, ...]]:
    """Build resampled components for selected node ids and return canonical node orders."""
    component_map = {str(comp.id): comp for comp in components}
    requested_canonical: List[str] = []
    position_map: Dict[str, Dict[float, str]] = {}

    for node_id in requested_node_ids:
        comp_id, position = _component_id_and_position(node_id)
        canonical = _canonical_node_id(comp_id, position)
        requested_canonical.append(canonical)
        position_map.setdefault(comp_id, {})
        position_map[comp_id].setdefault(float(position), canonical)

    selected_components: List[Component] = []
    controller_node_ids: List[str] = []
    for source in components:
        comp_id = str(source.id)
        if comp_id not in position_map:
            continue
        pos_pairs = sorted(position_map[comp_id].items(), key=lambda item: item[0])
        positions = [pos for pos, _ in pos_pairs]
        canon = [node for _, node in pos_pairs]
        selected_components.append(_resample_component(source, positions))
        controller_node_ids.extend(canon)

    missing = sorted(set(position_map.keys()) - {str(comp.id) for comp in selected_components})
    if missing:
        raise KeyError(f"Requested node ids reference unknown components: {missing}")

    return selected_components, tuple(controller_node_ids), tuple(requested_canonical)


def _build_node_permutations(
    swap_node_ids: Sequence[str], controller_node_ids: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return `(swap_to_controller, controller_to_swap)` index permutations."""
    if len(swap_node_ids) != len(controller_node_ids):
        raise ValueError("Swap node list and controller node list must have the same length.")

    ctrl_index: Dict[str, int] = {}
    for i, node_id in enumerate(controller_node_ids):
        if node_id in ctrl_index:
            raise ValueError(f"Duplicate controller node id detected: '{node_id}'")
        ctrl_index[node_id] = i

    swap_to_controller = np.empty(len(swap_node_ids), dtype=int)
    for i, node_id in enumerate(swap_node_ids):
        if node_id not in ctrl_index:
            raise KeyError(f"Swap node id '{node_id}' not found in controller nodes.")
        swap_to_controller[i] = ctrl_index[node_id]

    controller_to_swap = np.empty(len(controller_node_ids), dtype=int)
    for swap_idx, ctrl_idx in enumerate(swap_to_controller):
        controller_to_swap[ctrl_idx] = swap_idx

    return swap_to_controller, controller_to_swap


class VorLapQBladeRuntime:
    """
    Stateful runtime object used by the embedded QBlade bridge.

    The runtime keeps the VorLap controller alive across timesteps, caches the
    swap-array layout, and exposes a small diagnostic message buffer for the
    optional QBlade `update_message` callback.
    """

    def __init__(
        self,
        controller: QBladeController,
        swap_layout: Mapping[str, Union[SwapBlockSpec, Sequence[int], Tuple[int, Sequence[int]], int]],
        *,
        time_block: str = "time",
        timestep_block: str = "timestep",
        azimuth_block: str = "azimuth_deg",
        velocity_block: str = "velocity",
        force_block: str = "force",
        sample_step: Optional[float] = None,
        parameter_file: Optional[str] = None,
        swap_to_controller_idx: Optional[Sequence[int]] = None,
        controller_to_swap_idx: Optional[Sequence[int]] = None,
        debug: bool = False,
        resolved_sim_path: Optional[str] = None,
        resolved_airfoil_dir: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> None:
        if controller is None:
            raise ValueError("controller must not be None.")
        self.controller = controller
        self.time_block = str(time_block)
        self.timestep_block = str(timestep_block)
        self.azimuth_block = str(azimuth_block)
        self.velocity_block = str(velocity_block)
        self.force_block = str(force_block)
        self.sample_step = None if sample_step is None else float(sample_step)
        self.parameter_file = parameter_file
        self.debug = bool(debug)
        self.resolved_sim_path = None if resolved_sim_path is None else str(resolved_sim_path)
        self.resolved_airfoil_dir = None if resolved_airfoil_dir is None else str(resolved_airfoil_dir)
        self.log_file = None if log_file is None else str(log_file)
        self._layout = _coerce_layout(swap_layout)
        self._adapter: Optional[SwapArrayAdapter] = None
        if (swap_to_controller_idx is None) != (controller_to_swap_idx is None):
            raise ValueError("swap_to_controller_idx and controller_to_swap_idx must be provided together.")
        self._swap_to_controller_idx = (
            None if swap_to_controller_idx is None else np.asarray(swap_to_controller_idx, dtype=int).reshape(-1)
        )
        self._controller_to_swap_idx = (
            None if controller_to_swap_idx is None else np.asarray(controller_to_swap_idx, dtype=int).reshape(-1)
        )
        self._controller_velocity_buffer: Optional[np.ndarray] = None
        self._controller_structural_velocity_buffer: Optional[np.ndarray] = None
        self._swap_force_buffer: Optional[np.ndarray] = None
        inflow_dir = np.asarray(self.controller.viv_params.inflow_vec, dtype=float).reshape(3)
        inflow_speed = float(np.asarray(self.controller.viv_params.inflow_speeds, dtype=float).reshape(-1)[0])
        self._ambient_velocity_global = inflow_dir * inflow_speed
        if self.debug:
            sim_name = os.path.basename(self.resolved_sim_path) if self.resolved_sim_path else "?"
            airfoil_name = self.resolved_airfoil_dir if self.resolved_airfoil_dir else "?"
            self._last_message = (
                f"VorLap init ok; sim={sim_name}; airfoils={airfoil_name}; "
                f"nodes={self.controller.node_ids[0] if self.controller.node_ids else '?'}.. "
                f"count={len(self.controller.node_ids) if self.controller.node_ids else 0}"
            )
        else:
            self._last_message = "VorLap QBlade runtime initialized."
        self._append_log_line(self._last_message)

    def _append_log_line(self, message: str) -> None:
        if not self.log_file:
            return
        log_path = os.path.abspath(self.log_file)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")

    @classmethod
    def from_qblade_config(cls, param_file: str) -> "VorLapQBladeRuntime":
        """Build a runtime directly from the JSON config written by the case-prep script."""
        param_path = os.path.abspath(param_file)
        cfg = _load_config_json(param_path)

        source_parameter_dir = cfg.get("source_parameter_dir")
        extra_bases = [] if source_parameter_dir in (None, "") else [str(source_parameter_dir)]

        sim_path = _resolve_path(param_path, cfg["sim_path"], extra_bases=extra_bases)
        airfoil_dir = _resolve_path(param_path, cfg["airfoil_dir"], extra_bases=extra_bases)
        log_file = cfg.get("log_file")
        log_file = None if log_file in (None, "") else _resolve_path(param_path, str(log_file), extra_bases=extra_bases)

        default_airfoil_id = str(cfg.get("default_airfoil_id", "default"))
        tower_airfoil_id = str(cfg.get("tower_airfoil_id", "cylinder"))
        n_freq_depth = cfg.get("n_freq_depth", None)
        n_freq_depth = None if n_freq_depth is None else int(n_freq_depth)
        force_scale = float(cfg.get("force_scale", 1.0))
        debug = bool(cfg.get("debug", False))

        requested_node_ids_raw = cfg.get("node_ids")
        if requested_node_ids_raw is None:
            requested_node_ids: Tuple[str, ...] = ()
        else:
            requested_node_ids = tuple(str(node_id) for node_id in requested_node_ids_raw)
        include_struts, include_tower = _infer_geometry_inclusions(
            requested_node_ids,
            bool(cfg.get("include_struts", True)),
            bool(cfg.get("include_tower", True)),
        )

        components, viv_params, node_ids = convert_qblade_to_vorlap_inputs(
            sim_path,
            default_airfoil_id=default_airfoil_id,
            include_struts=include_struts,
            include_tower=include_tower,
            tower_airfoil_id=tower_airfoil_id,
        )
        affts = load_airfoil_fft_directory(airfoil_dir)
        viv_params.airfoil_folder = airfoil_dir

        if not requested_node_ids:
            requested_node_ids = tuple(str(node_id) for node_id in node_ids)
        if not requested_node_ids:
            raise ValueError("Config file does not define any node_ids.")

        swap_layout_cfg = cfg.get("swap_layout")
        if not isinstance(swap_layout_cfg, list) or not swap_layout_cfg:
            raise ValueError("Config file does not define a valid swap_layout.")
        swap_layout: Dict[str, Union[SwapBlockSpec, Sequence[int], Tuple[int, Sequence[int]], int]] = {}
        for entry in swap_layout_cfg:
            if not isinstance(entry, Mapping):
                raise TypeError("swap_layout entries must be mapping objects.")
            name = str(entry["name"])
            offset = int(entry["offset"])
            shape = _normalize_shape(entry.get("shape", ()))
            swap_layout[name] = SwapBlockSpec(offset=offset, shape=shape)

        selected_components, controller_node_ids, swap_node_ids = _build_selected_components(
            components, requested_node_ids
        )
        controller = QBladeController.from_components(
            selected_components,
            affts,
            viv_params=viv_params,
            n_freq_depth=n_freq_depth,
            node_ids=controller_node_ids,
            force_scale=force_scale,
        )
        swap_to_controller_idx, controller_to_swap_idx = _build_node_permutations(
            swap_node_ids, controller.node_ids
        )

        return cls(
            controller=controller,
            swap_layout=swap_layout,
            time_block=str(cfg.get("time_block", "time")),
            timestep_block=str(cfg.get("timestep_block", "timestep")),
            azimuth_block=str(cfg.get("azimuth_block", "azimuth_deg")),
            velocity_block=str(cfg.get("velocity_block", "velocity")),
            force_block=str(cfg.get("force_block", "force")),
            sample_step=cfg.get("sample_step"),
            parameter_file=param_path,
            swap_to_controller_idx=swap_to_controller_idx,
            controller_to_swap_idx=controller_to_swap_idx,
            debug=debug,
            resolved_sim_path=sim_path,
            resolved_airfoil_dir=airfoil_dir,
            log_file=log_file,
        )

    @classmethod
    def from_qblade_sim(
        cls,
        sim_path: str,
        airfoil_dir: str,
        *,
        default_airfoil_id: str = "default",
        include_struts: bool = True,
        include_tower: bool = True,
        tower_airfoil_id: str = "cylinder",
        n_freq_depth: Optional[int] = None,
        force_scale: float = 1.0,
    ) -> "VorLapQBladeRuntime":
        """Construct a runtime from a QBlade simulation definition and an airfoil FFT directory."""
        components, viv_params, node_ids = convert_qblade_to_vorlap_inputs(
            sim_path,
            default_airfoil_id=default_airfoil_id,
            include_struts=include_struts,
            include_tower=include_tower,
            tower_airfoil_id=tower_airfoil_id,
        )
        affts = load_airfoil_fft_directory(airfoil_dir)
        viv_params.airfoil_folder = airfoil_dir
        controller = QBladeController.from_components(
            components,
            affts,
            viv_params=viv_params,
            n_freq_depth=n_freq_depth,
            node_ids=node_ids,
            force_scale=force_scale,
        )
        table_spec = build_external_library_table_spec(node_ids)
        return cls(
            controller=controller,
            swap_layout=table_spec.swap_layout,
            sample_step=None,
            debug=False,
            resolved_sim_path=sim_path,
            resolved_airfoil_dir=airfoil_dir,
            log_file=None,
        )

    @property
    def message(self) -> str:
        return self._last_message

    @property
    def swap_size(self) -> int:
        return max(spec.offset + spec.size for spec in self._layout.values())

    @property
    def layout(self) -> Dict[str, SwapBlockSpec]:
        return dict(self._layout)

    def _adapter_for(self, avr_swap: np.ndarray) -> SwapArrayAdapter:
        if self._adapter is None:
            self._adapter = SwapArrayAdapter(avr_swap, self._layout)
            return self._adapter
        self._adapter.rebind(avr_swap)
        return self._adapter

    def update(self, avr_swap: np.ndarray) -> np.ndarray:
        """
        Evaluate VorLap for a single QBlade timestep.

        The passed array must contain the input blocks defined in the runtime
        config and will be modified in place with the returned force vectors.
        """
        swap = np.asarray(avr_swap)
        if swap.ndim != 1:
            raise ValueError("avr_swap must be a one-dimensional buffer.")
        if swap.size < self.swap_size:
            raise ValueError(f"avr_swap is too small: expected at least {self.swap_size}, got {swap.size}.")
        adapter = self._adapter_for(swap)

        if self._swap_to_controller_idx is None:
            structural_velocity = np.asarray(adapter.read(self.velocity_block), dtype=float)
            relative_velocity = self._ambient_velocity_global[None, :] - structural_velocity
            forces = self.controller.step(
                {
                    "velocity": relative_velocity,
                    "time": adapter.scalar(self.time_block),
                    "azimuth_deg": adapter.scalar(self.azimuth_block),
                }
            )
            adapter.write(self.force_block, forces)
        else:
            velocity_swap = np.asarray(adapter.read(self.velocity_block), dtype=float)
            n_controller = int(self._controller_to_swap_idx.size)
            n_swap = int(self._swap_to_controller_idx.size)
            if self._controller_velocity_buffer is None or self._controller_velocity_buffer.shape != (n_controller, 3):
                self._controller_velocity_buffer = np.empty((n_controller, 3), dtype=float)
            if self._controller_structural_velocity_buffer is None or self._controller_structural_velocity_buffer.shape != (n_controller, 3):
                self._controller_structural_velocity_buffer = np.empty((n_controller, 3), dtype=float)
            if self._swap_force_buffer is None or self._swap_force_buffer.shape != (n_swap, 3):
                self._swap_force_buffer = np.empty((n_swap, 3), dtype=float)

            np.take(
                velocity_swap,
                self._controller_to_swap_idx,
                axis=0,
                out=self._controller_structural_velocity_buffer,
            )
            np.subtract(
                self._ambient_velocity_global[None, :],
                self._controller_structural_velocity_buffer,
                out=self._controller_velocity_buffer,
            )
            forces = self.controller.step(
                {
                    "velocity": self._controller_velocity_buffer,
                    "time": adapter.scalar(self.time_block),
                    "azimuth_deg": adapter.scalar(self.azimuth_block),
                }
            )
            np.take(
                forces,
                self._swap_to_controller_idx,
                axis=0,
                out=self._swap_force_buffer,
            )
            adapter.write(self.force_block, self._swap_force_buffer)

        time = adapter.scalar(self.time_block)
        azimuth = adapter.scalar(self.azimuth_block)
        if self.debug:
            force_norm = np.linalg.norm(forces, axis=1)
            max_idx = int(np.argmax(force_norm)) if force_norm.size else 0
            max_force = float(force_norm[max_idx]) if force_norm.size else 0.0
            max_node = self.controller.node_ids[max_idx] if force_norm.size else "?"
            structural_velocity = np.asarray(adapter.read(self.velocity_block), dtype=float)
            max_structural_speed = float(np.max(np.linalg.norm(structural_velocity, axis=1))) if structural_velocity.size else 0.0
            max_relative_speed = float(np.max(np.linalg.norm(self._ambient_velocity_global[None, :] - structural_velocity, axis=1))) if structural_velocity.size else 0.0
            self._last_message = (
                f"VorLap dbg t={time:.6g}s az={azimuth:.6g}deg "
                f"max|F|={max_force:.6g}N node={max_node} "
                f"scale={self.controller.force_scale:.6g} "
                f"max|Vstruct|={max_structural_speed:.6g}m/s "
                f"max|Vrel|={max_relative_speed:.6g}m/s"
            )
        else:
            self._last_message = (
                f"VorLap update complete: t={time:.6g} s, azimuth={azimuth:.6g} deg, "
                f"nodes={forces.shape[0]}"
            )
        self._append_log_line(self._last_message)
        return forces

    def update_message(self) -> str:
        """Return the last human-readable status string."""
        return self._last_message


def create_runtime(param_file: str) -> VorLapQBladeRuntime:
    """Public factory used by the C++ bridge."""
    return VorLapQBladeRuntime.from_qblade_config(param_file)


__all__ = [
    "ExternalLibraryTableSpec",
    "VorLapQBladeRuntime",
    "build_external_library_table_spec",
    "build_qblade_external_config",
    "create_runtime",
    "load_airfoil_fft_directory",
    "patch_qblade_structural_definition",
    "patch_qblade_turbine_definition",
    "write_qblade_external_config",
]
