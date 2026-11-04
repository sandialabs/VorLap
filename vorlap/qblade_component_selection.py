"""QBlade node-selection and component-resampling helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .geometry_frames import rotationMatrix
from .qblade_nodes import action_target_from_node_id, canonical_node_id
from .structs import Component


_EPS = 1.0e-12


def component_span_coordinate(shape_xyz: np.ndarray) -> np.ndarray:
    """Return normalized cumulative span coordinates for component nodes."""
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


def interp_along_span(values: np.ndarray, span: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Interpolate 1D or 2D node arrays along normalized component span."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.interp(query, span, arr)
    if arr.ndim == 2:
        out = np.empty((query.shape[0], arr.shape[1]), dtype=float)
        for j in range(arr.shape[1]):
            out[:, j] = np.interp(query, span, arr[:, j])
        return out
    raise ValueError(f"Unsupported interpolation rank: {arr.ndim}")


def resample_component(component: Component, positions: Sequence[float]) -> Component:
    """Return a component sampled at normalized QBlade action positions."""
    query = np.asarray(positions, dtype=float).reshape(-1)
    if np.any(~np.isfinite(query)) or np.any((query < 0.0) | (query > 1.0)):
        raise ValueError("Component resampling positions must be finite and within [0, 1].")
    span = component_span_coordinate(component.shape_xyz)

    shape_xyz = interp_along_span(np.asarray(component.shape_xyz, dtype=float), span, query)
    chord = interp_along_span(np.asarray(component.chord, dtype=float), span, query)
    twist = interp_along_span(np.asarray(component.twist, dtype=float), span, query)
    thickness = interp_along_span(np.asarray(component.thickness, dtype=float), span, query)
    offset = interp_along_span(np.asarray(component.offset, dtype=float), span, query)

    airfoil_ids_raw = list(getattr(component, "airfoil_ids", []) or [])
    if not airfoil_ids_raw:
        airfoil_ids_raw = ["default"] * len(span)
    nearest = np.argmin(np.abs(span[None, :] - query[:, None]), axis=1)
    airfoil_ids = [str(airfoil_ids_raw[i]) for i in nearest]

    chord_vector = np.asarray(component.chord_vector, dtype=float)
    normal_vector = np.asarray(component.normal_vector, dtype=float)
    if chord_vector.ndim == 2 and chord_vector.shape[0] == len(span):
        chord_vector = interp_along_span(chord_vector, span, query)
    else:
        chord_vector = np.zeros((query.shape[0], 3), dtype=float)
    if normal_vector.ndim == 2 and normal_vector.shape[0] == len(span):
        normal_vector = interp_along_span(normal_vector, span, query)
    else:
        normal_vector = np.zeros((query.shape[0], 3), dtype=float)

    # Interpolating a direction field and the curved centerline independently
    # introduces small chord/span and normal/span errors.  Reconstruct the
    # right-handed section basis against the *resampled* global centerline so
    # that the vectors sent to VorLap are consistent with the QBlade action
    # locations used by the geometry audit and runtime coupling.  Preserve
    # legacy/custom vector pairs that were not section bases to begin with.
    frame_valid = (np.linalg.norm(chord_vector, axis=1) > _EPS) & (
        np.linalg.norm(normal_vector, axis=1) > _EPS
    )
    rotation = rotationMatrix(np.asarray(component.rotation, dtype=float).reshape(3))
    source_chord = np.asarray(component.chord_vector, dtype=float)
    source_normal = np.asarray(component.normal_vector, dtype=float)
    source_global_shape = (rotation @ np.asarray(component.shape_xyz, dtype=float).T).T
    source_span = component_span_vectors_global(source_global_shape)
    source_is_section_basis = (
        source_chord.shape == source_normal.shape == source_span.shape
        and np.allclose(np.linalg.norm(source_chord, axis=1), 1.0, atol=1.0e-8)
        and np.allclose(np.linalg.norm(source_normal, axis=1), 1.0, atol=1.0e-8)
        and np.allclose(np.sum(source_chord * source_normal, axis=1), 0.0, atol=1.0e-8)
        and np.allclose(np.sum(source_chord * source_span, axis=1), 0.0, atol=1.0e-8)
        and np.allclose(np.sum(source_normal * source_span, axis=1), 0.0, atol=1.0e-8)
        and np.allclose(
            np.sum(source_span * np.cross(source_chord, source_normal), axis=1),
            1.0,
            atol=1.0e-8,
        )
    )
    if np.any(frame_valid) and source_is_section_basis:
        global_shape = (rotation @ np.asarray(shape_xyz, dtype=float).T).T
        global_span = component_span_vectors_global(global_shape)
        projected_chord = chord_vector - np.sum(chord_vector * global_span, axis=1)[:, None] * global_span
        projected_norm = np.linalg.norm(projected_chord, axis=1)
        usable = frame_valid & (projected_norm > _EPS)
        chord_vector[usable] = projected_chord[usable] / projected_norm[usable, None]
        normal_vector[usable] = np.cross(global_span[usable], chord_vector[usable])
        normal_vector[usable] /= np.linalg.norm(normal_vector[usable], axis=1)[:, None]
        chord_vector[frame_valid & ~usable] = 0.0
        normal_vector[frame_valid & ~usable] = 0.0

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


def build_selected_components(
    components: Sequence[Component], requested_node_ids: Sequence[str]
) -> Tuple[List[Component], Tuple[str, ...], Tuple[str, ...]]:
    """Build resampled components for selected node ids and return canonical node orders."""
    position_map: Dict[str, Dict[float, str]] = {}
    requested_canonical: List[str] = []

    for node_id in requested_node_ids:
        comp_id, position = action_target_from_node_id(node_id)
        canonical = canonical_node_id(comp_id, position)
        requested_canonical.append(canonical)
        position_map.setdefault(comp_id, {})
        position_map[comp_id].setdefault(float(position), canonical)

    if len(set(requested_canonical)) != len(requested_canonical):
        duplicates = sorted({node for node in requested_canonical if requested_canonical.count(node) > 1})
        raise ValueError(f"Duplicate requested node id(s) after canonicalization: {', '.join(duplicates)}")

    selected_components: List[Component] = []
    controller_node_ids: List[str] = []
    for source in components:
        comp_id = str(source.id)
        if comp_id not in position_map:
            continue
        pos_pairs = sorted(position_map[comp_id].items(), key=lambda item: item[0])
        positions = [pos for pos, _ in pos_pairs]
        canon = [node for _, node in pos_pairs]
        selected_components.append(resample_component(source, positions))
        controller_node_ids.extend(canon)

    missing = sorted(set(position_map.keys()) - {str(comp.id) for comp in selected_components})
    if missing:
        raise KeyError(f"Requested node ids reference unknown components: {missing}")

    return selected_components, tuple(controller_node_ids), tuple(requested_canonical)


def build_node_permutations(
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


def component_span_vectors_global(shape_xyz_global: np.ndarray) -> np.ndarray:
    """Compute per-node span directions from prepared global component positions."""
    xyz = np.asarray(shape_xyz_global, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Invalid prepared component coordinates with shape {xyz.shape}.")
    npts = xyz.shape[0]
    if npts < 1:
        raise ValueError("Prepared component coordinates must contain at least one node.")
    if npts == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=float)
    spans = np.empty_like(xyz)
    for i in range(npts):
        if i == 0:
            delta = xyz[1] - xyz[0]
        elif i == npts - 1:
            delta = xyz[-1] - xyz[-2]
        else:
            delta = 0.5 * ((xyz[i + 1] - xyz[i]) + (xyz[i] - xyz[i - 1]))
        norm = float(np.linalg.norm(delta))
        if norm <= _EPS:
            spans[i] = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            spans[i] = delta / norm
    return spans


__all__ = [
    "build_node_permutations",
    "build_selected_components",
    "component_span_coordinate",
    "component_span_vectors_global",
    "interp_along_span",
    "resample_component",
]
