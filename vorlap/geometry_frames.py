"""Geometry-frame preparation and rotation helpers for VorLap."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .structs import Component


_EPS = 1.0e-12


def normalize(vector: np.ndarray, name: str) -> np.ndarray:
    """Return a unit vector and raise on zero magnitude input."""
    vec = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= _EPS:
        raise ValueError(f"{name} must have non-zero magnitude.")
    return vec / norm


def rotate_vector(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a 3D vector around a given axis using the right-hand rule.
    """
    theta = math.radians(angle_deg)
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= _EPS:
        raise ValueError("Rotation axis must have non-zero magnitude.")
    k = axis / axis_norm
    v = np.asarray(vec, dtype=float)
    return v * math.cos(theta) + np.cross(k, v) * math.sin(theta) + k * np.dot(k, v) * (1 - math.cos(theta))


def rotationMatrix(euler: np.ndarray) -> np.ndarray:
    """
    Compute a 3x3 rotation matrix from Euler angles using the ZYX convention.
    """
    cz, sz = math.cos(math.radians(euler[2])), math.sin(math.radians(euler[2]))
    cy, sy = math.cos(math.radians(euler[1])), math.sin(math.radians(euler[1]))
    cx, sx = math.cos(math.radians(euler[0])), math.sin(math.radians(euler[0]))

    Rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    Ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ]
    )

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ]
    )

    return Rz @ Ry @ Rx


def rotate_point_about_axis(point: np.ndarray, axis_offset: np.ndarray, axis_unit: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a point about an axis passing through `axis_offset`."""
    offset = np.asarray(axis_offset, dtype=float)
    return offset + rotate_vector(np.asarray(point, dtype=float) - offset, axis_unit, angle_deg)


def rotate_vectors_about_axes(vectors: np.ndarray, axes: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Rotate an array of vectors around per-row axes using Rodrigues' formula."""
    vec = np.asarray(vectors, dtype=float)
    axis = np.asarray(axes, dtype=float)
    angles = np.deg2rad(np.asarray(angles_deg, dtype=float).reshape(-1))
    if vec.shape != axis.shape:
        raise ValueError("vectors and axes must have matching shapes.")
    if vec.ndim != 2 or vec.shape[1] != 3:
        raise ValueError("vectors must have shape [n, 3].")
    if axis.ndim != 2 or axis.shape[1] != 3:
        raise ValueError("axes must have shape [n, 3].")
    if angles.shape[0] != vec.shape[0]:
        raise ValueError("angles_deg must have one value per vector.")

    axis_norm = np.linalg.norm(axis, axis=1)
    if np.any(axis_norm <= _EPS):
        raise ValueError("Rotation axes must have non-zero magnitude.")
    axis_unit = axis / axis_norm[:, None]

    c = np.cos(angles)[:, None]
    s = np.sin(angles)[:, None]
    one_c = 1.0 - c
    return (
        vec * c
        + np.cross(axis_unit, vec) * s
        + axis_unit * np.sum(axis_unit * vec, axis=1)[:, None] * one_c
    )


def segment_span_vectors(shape_xyz: np.ndarray) -> np.ndarray:
    """Compute per-node span directions from component centerline points."""
    xyz = np.asarray(shape_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("shape_xyz must have shape [n, 3].")
    npts = xyz.shape[0]
    if npts < 1:
        raise ValueError("shape_xyz must contain at least one node.")
    if npts == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=float)

    span = np.empty_like(xyz)
    for i in range(npts):
        if i == 0:
            delta = xyz[1] - xyz[0]
        elif i == npts - 1:
            delta = xyz[-1] - xyz[-2]
        else:
            delta = 0.5 * ((xyz[i + 1] - xyz[i]) + (xyz[i] - xyz[i - 1]))
        norm = float(np.linalg.norm(delta))
        if norm <= _EPS:
            span[i] = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            span[i] = delta / norm
    return span


def pick_reference_axis(span: np.ndarray) -> np.ndarray:
    """Choose a stable reference axis that is not parallel to the span vector."""
    span_u = normalize(span, "span")
    refs = (
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )
    for ref in refs:
        if abs(float(np.dot(span_u, ref))) < 0.9:
            return ref
    return refs[0]


def validate_component_geometry_arrays(component: Component, npts: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validated per-node chord, twist, thickness, and offset arrays."""
    chord = np.asarray(component.chord, dtype=float).reshape(-1)
    twist = np.asarray(component.twist, dtype=float).reshape(-1)
    thickness = np.asarray(component.thickness, dtype=float).reshape(-1)
    offset = np.asarray(component.offset, dtype=float).reshape(-1)
    if not (len(chord) == len(twist) == len(thickness) == len(offset) == npts):
        raise ValueError(
            f"Component '{component.id}' has inconsistent vector lengths: "
            f"xyz={npts}, chord={len(chord)}, twist={len(twist)}, "
            f"thickness={len(thickness)}, offset={len(offset)}."
        )
    for name, arr in (("chord", chord), ("twist", twist), ("thickness", thickness), ("offset", offset)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Component '{component.id}' {name} values must be finite.")
    if np.any(chord <= _EPS):
        raise ValueError(f"Component '{component.id}' chord values must be positive.")
    if np.any(thickness < 0.0):
        raise ValueError(f"Component '{component.id}' thickness values must be non-negative.")
    return chord, twist, thickness, offset


def build_fallback_frame(component: Component) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer global unit chord/normal vectors and node positions from component geometry."""
    xyz = np.asarray(component.shape_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Component '{component.id}' has invalid shape_xyz with shape {xyz.shape}.")
    if not np.all(np.isfinite(xyz)):
        raise ValueError(f"Component '{component.id}' shape_xyz values must be finite.")

    npts = xyz.shape[0]
    _chord, twist, _thickness, _offset = validate_component_geometry_arrays(component, npts)
    pitch = np.asarray(component.pitch, dtype=float).reshape(-1)
    if not np.all(np.isfinite(pitch)):
        raise ValueError(f"Component '{component.id}' pitch values must be finite.")
    pitch_val = float(pitch[0]) if pitch.size else 0.0
    twist_eff = twist + pitch_val

    span_local = segment_span_vectors(xyz)
    chord_local = np.empty((npts, 3), dtype=float)
    normal_local = np.empty((npts, 3), dtype=float)

    for i in range(npts):
        span = span_local[i]
        ref = pick_reference_axis(span)
        chord = np.cross(ref, span)
        chord_norm = float(np.linalg.norm(chord))
        if chord_norm <= _EPS:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
            chord = np.cross(ref, span)
            chord_norm = float(np.linalg.norm(chord))
        if chord_norm <= _EPS:
            chord = np.array([1.0, 0.0, 0.0], dtype=float)
            chord_norm = 1.0
        chord /= chord_norm

        normal = np.cross(span, chord)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= _EPS:
            normal = np.array([0.0, 1.0, 0.0], dtype=float)
            normal_norm = 1.0
        normal /= normal_norm

        chord_local[i] = rotate_vectors_about_axes(
            chord[None, :],
            span[None, :],
            np.array([twist_eff[i]], dtype=float),
        )[0]
        normal_local[i] = rotate_vectors_about_axes(
            normal[None, :],
            span[None, :],
            np.array([twist_eff[i]], dtype=float),
        )[0]

    R = rotationMatrix(np.asarray(component.rotation, dtype=float).reshape(3))
    translation = np.asarray(component.translation, dtype=float).reshape(3)
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(translation)):
        raise ValueError(f"Component '{component.id}' rotation and translation must be finite.")

    chord_global = (R @ chord_local.T).T
    normal_global = (R @ normal_local.T).T
    global_pos = (R @ xyz.T).T + translation[None, :]
    return chord_global, normal_global, global_pos


def as_valid_frame_array(component: Component, attr_name: str, npts: int) -> Optional[np.ndarray]:
    """Return a finite [n,3] frame array, or None when placeholders need fallback."""
    raw = getattr(component, attr_name, None)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (npts, 3) or not np.all(np.isfinite(arr)):
        return None
    return arr


def prepare_component_frames(components: List[Component]) -> List[Component]:
    """
    Populate solver-critical global node positions and section basis vectors.
    """
    if not components:
        raise ValueError("At least one component is required.")

    for component in components:
        xyz = np.asarray(component.shape_xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Component '{component.id}' has invalid shape_xyz with shape {xyz.shape}.")
        npts = xyz.shape[0]

        fallback_chord, fallback_normal, global_pos = build_fallback_frame(component)
        chord_vec = as_valid_frame_array(component, "chord_vector", npts)
        normal_vec = as_valid_frame_array(component, "normal_vector", npts)

        chord_final = fallback_chord.copy()
        normal_final = fallback_normal.copy()
        if chord_vec is not None and normal_vec is not None:
            chord_norm = np.linalg.norm(chord_vec, axis=1)
            normal_norm = np.linalg.norm(normal_vec, axis=1)
            provided_good = (chord_norm > _EPS) & (normal_norm > _EPS)
            if np.any(provided_good):
                chord_final[provided_good] = chord_vec[provided_good] / chord_norm[provided_good][:, None]
                normal_final[provided_good] = normal_vec[provided_good] / normal_norm[provided_good][:, None]

        component.shape_xyz_global = np.asarray(global_pos, dtype=float)
        component.chord_vector = np.asarray(chord_final, dtype=float)
        component.normal_vector = np.asarray(normal_final, dtype=float)

    return components


__all__ = [
    "as_valid_frame_array",
    "build_fallback_frame",
    "normalize",
    "pick_reference_axis",
    "prepare_component_frames",
    "rotate_point_about_axis",
    "rotate_vector",
    "rotate_vectors_about_axes",
    "rotationMatrix",
    "segment_span_vectors",
    "validate_component_geometry_arrays",
]
