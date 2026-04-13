"""Stateful per-timestep controller utilities for QBlade-style coupling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .computations import _compute_local_segment_length, _resolve_airfoil, rotationMatrix
from .structs import AirfoilFFT, Component, VIV_Params

_EPS = 1.0e-12


def _normalize_vector(vector: np.ndarray, name: str) -> np.ndarray:
    """Return a unit-length 3-vector and raise if the input is degenerate."""
    vec = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= _EPS:
        raise ValueError(f"{name} must have non-zero magnitude.")
    return vec / norm


def _axis_angle_matrix(axis_unit: np.ndarray, angle_deg: float) -> np.ndarray:
    """Return a 3x3 Rodrigues rotation matrix for a unit axis and angle in degrees."""
    axis = _normalize_vector(axis_unit, "axis_unit")
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    one_c = 1.0 - c
    x, y, z = axis
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _rotate_vectors_about_axes(vectors: np.ndarray, axes: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
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


def _segment_span_vectors(shape_xyz: np.ndarray) -> np.ndarray:
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


def _pick_reference_axis(span: np.ndarray) -> np.ndarray:
    """Choose a stable reference axis that is not parallel to the span vector."""
    span_u = _normalize_vector(span, "span")
    refs = (
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([1.0, 0.0, 0.0], dtype=float),
    )
    for ref in refs:
        if abs(float(np.dot(span_u, ref))) < 0.9:
            return ref
    return refs[0]


def _build_fallback_frame(component: Component) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer global chord and normal vectors when the component does not provide them."""
    xyz = np.asarray(component.shape_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Component '{component.id}' has invalid shape_xyz with shape {xyz.shape}.")

    npts = xyz.shape[0]
    twist = np.asarray(component.twist, dtype=float).reshape(-1)
    if twist.shape[0] != npts:
        raise ValueError(f"Component '{component.id}' twist length does not match shape_xyz.")
    pitch = np.asarray(component.pitch, dtype=float).reshape(-1)
    pitch_val = float(pitch[0]) if pitch.size else 0.0
    twist_eff = twist + pitch_val

    span_local = _segment_span_vectors(xyz)
    chord_local = np.empty((npts, 3), dtype=float)
    normal_local = np.empty((npts, 3), dtype=float)

    for i in range(npts):
        span = span_local[i]
        ref = _pick_reference_axis(span)
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

        chord_local[i] = _rotate_vectors_about_axes(chord[None, :], span[None, :], np.array([twist_eff[i]], dtype=float))[0]
        normal_local[i] = _rotate_vectors_about_axes(normal[None, :], span[None, :], np.array([twist_eff[i]], dtype=float))[0]

    R = rotationMatrix(np.asarray(component.rotation, dtype=float).reshape(3))
    translation = np.asarray(component.translation, dtype=float).reshape(3)
    chord_global = (R @ chord_local.T).T
    normal_global = (R @ normal_local.T).T
    global_pos = (R @ xyz.T).T + translation[None, :]
    return chord_global, normal_global, global_pos


def _normalize_airfoil_ids(airfoil_ids: Sequence[str], npts: int) -> List[str]:
    """Pad or trim airfoil ids to match the component node count."""
    ids = [str(v) for v in airfoil_ids] if airfoil_ids is not None else []
    if len(ids) < npts:
        fill_id = ids[-1] if ids else "default"
        ids = ids + [fill_id] * (npts - len(ids))
    return ids[:npts]


@dataclass(frozen=True)
class SwapBlockSpec:
    """Describe a contiguous block within a flat QBlade-style swap array."""

    offset: int
    shape: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        offset = int(self.offset)
        if offset < 0:
            raise ValueError("offset must be non-negative.")
        shape_raw = self.shape
        if isinstance(shape_raw, numbers.Integral):
            shape = (int(shape_raw),)
        else:
            try:
                shape_tuple = tuple(shape_raw)
            except TypeError as exc:
                raise TypeError("shape must be an int or a sequence of ints.") from exc
            shape = tuple(int(dim) for dim in shape_tuple)
        if len(shape) == 1 and shape[0] == 0:
            shape = ()
        if any(dim <= 0 for dim in shape):
            raise ValueError("shape dimensions must be positive.")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "shape", shape)

    @property
    def size(self) -> int:
        """Number of floats spanned by the block."""
        if not self.shape:
            return 1
        size = 1
        for dim in self.shape:
            size *= dim
        return int(size)


class SwapArrayAdapter:
    """
    Zero-copy adapter for a flat QBlade-style `float* avrSwap` buffer.

    Blocks are defined by a mapping of names to `SwapBlockSpec` objects or
    `(offset, shape)` tuples. The adapter returns views into the underlying
    array so callers can read and write without extra copies.
    """

    def __init__(self, avr_swap: np.ndarray, layout: Mapping[str, Union[SwapBlockSpec, Tuple[int, Tuple[int, ...]], Tuple[int, int], int]]):
        swap = np.asarray(avr_swap)
        if swap.dtype.kind not in ("f",):
            swap = np.asarray(avr_swap, dtype=float)
        self._swap = swap
        if self._swap.ndim != 1:
            raise ValueError("avr_swap must be a one-dimensional array.")
        self._layout: Dict[str, SwapBlockSpec] = {str(name): self._coerce_spec(spec) for name, spec in layout.items()}
        self._validate_layout()

    @classmethod
    def from_layout(
        cls,
        avr_swap: np.ndarray,
        layout: Mapping[str, Union[SwapBlockSpec, Tuple[int, Tuple[int, ...]], Tuple[int, int], int]],
    ) -> "SwapArrayAdapter":
        """Construct an adapter from a flat buffer and a block layout mapping."""
        return cls(avr_swap, layout)

    @staticmethod
    def _coerce_spec(spec: Union[SwapBlockSpec, Tuple[int, Tuple[int, ...]], Tuple[int, int], int]) -> SwapBlockSpec:
        if isinstance(spec, SwapBlockSpec):
            return spec
        if isinstance(spec, numbers.Integral):
            return SwapBlockSpec(offset=spec)
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
            offset = int(spec[0])
            shape = spec[1]
            if isinstance(shape, numbers.Integral):
                shape_tuple: Tuple[int, ...] = (int(shape),)
            else:
                shape_tuple = tuple(int(dim) for dim in shape)
            return SwapBlockSpec(offset=offset, shape=shape_tuple)
        raise TypeError("layout entries must be SwapBlockSpec, int, or (offset, shape) tuples.")

    def _validate_layout(self) -> None:
        used = np.zeros(self._swap.size, dtype=bool)
        for name, spec in self._layout.items():
            end = spec.offset + spec.size
            if end > self._swap.size:
                raise ValueError(
                    f"Swap block '{name}' exceeds avr_swap length: offset {spec.offset}, size {spec.size}, total {self._swap.size}."
                )
            if np.any(used[spec.offset:end]):
                raise ValueError(f"Swap block '{name}' overlaps an existing block.")
            used[spec.offset:end] = True

    @property
    def swap(self) -> np.ndarray:
        """Underlying flat swap buffer."""
        return self._swap

    @property
    def layout(self) -> Dict[str, SwapBlockSpec]:
        """Return a shallow copy of the configured block layout."""
        return dict(self._layout)

    @property
    def names(self) -> Tuple[str, ...]:
        """Configured block names."""
        return tuple(self._layout.keys())

    def block(self, name: str) -> np.ndarray:
        """Return a mutable view into a named block."""
        if name not in self._layout:
            raise KeyError(f"Unknown swap block '{name}'.")
        spec = self._layout[name]
        start = spec.offset
        stop = start + spec.size
        view = self._swap[start:stop]
        return view.reshape(spec.shape if spec.shape else ())

    def __getitem__(self, name: str) -> np.ndarray:
        return self.block(name)

    def __setitem__(self, name: str, values: np.ndarray) -> None:
        self.write(name, values)

    def read(self, name: str, copy: bool = False) -> np.ndarray:
        """Read a block, optionally copying it out of the buffer."""
        arr = self.block(name)
        return np.array(arr, copy=True) if copy else arr

    def scalar(self, name: str) -> float:
        """Read a scalar block as a Python float."""
        arr = self.block(name)
        if arr.size != 1:
            raise ValueError(f"Swap block '{name}' is not scalar.")
        return float(np.asarray(arr).reshape(-1)[0])

    def write(self, name: str, values: np.ndarray) -> None:
        """Write values into a named block."""
        target = self.block(name)
        arr = np.asarray(values, dtype=float)
        if target.shape != arr.shape:
            raise ValueError(f"Block '{name}' expects shape {target.shape}, got {arr.shape}.")
        target[...] = arr

    def rebind(self, avr_swap: np.ndarray) -> None:
        """Point the adapter at a new flat swap buffer with the same layout."""
        swap = np.asarray(avr_swap)
        if swap.dtype.kind not in ("f",):
            swap = np.asarray(avr_swap, dtype=float)
        if swap.ndim != 1:
            raise ValueError("avr_swap must be a one-dimensional array.")
        if swap.size != self._swap.size:
            raise ValueError(f"avr_swap size changed from {self._swap.size} to {swap.size}.")
        self._swap = swap


class VorLapQBladeController:
    """
    Stateful per-timestep controller for QBlade-style external force calls.

    The controller performs one-time geometry preparation, caches airfoil
    interpolators, and evaluates a single timestep with vectorized NumPy
    operations. If a step is called with exactly the same inputs as the
    previous call, the cached force output is returned unchanged.
    """

    def __init__(
        self,
        components: Sequence[Component],
        airfoils: Mapping[str, AirfoilFFT],
        viv_params: Optional[VIV_Params] = None,
        *,
        n_freq_depth: Optional[int] = None,
        node_ids: Optional[Sequence[str]] = None,
        force_scale: float = 1.0,
    ) -> None:
        if not components:
            raise ValueError("At least one component is required.")
        if not airfoils:
            raise ValueError("At least one airfoil FFT dataset is required.")

        self.viv_params = viv_params if viv_params is not None else VIV_Params()
        self.components = tuple(components)
        self.airfoils = dict(airfoils)
        # Backward-compatible alias for "airfoil FFTs" naming used elsewhere.
        self.affts = self.airfoils
        self.node_ids = tuple(str(n) for n in node_ids) if node_ids is not None else None
        self.n_freq_depth = int(self.viv_params.n_freq_depth if n_freq_depth is None else n_freq_depth)
        if self.n_freq_depth < 1:
            raise ValueError("n_freq_depth must be >= 1.")
        self.force_scale = float(force_scale)
        if not math.isfinite(self.force_scale) or self.force_scale < 0.0:
            raise ValueError("force_scale must be a finite, non-negative scalar.")

        self.fluid_density = float(self.viv_params.fluid_density)
        self.fluid_dynamicviscosity = float(self.viv_params.fluid_dynamicviscosity)
        self.rotation_axis = _normalize_vector(self.viv_params.rotation_axis, "rotation_axis")
        self.rotation_axis_offset = np.asarray(self.viv_params.rotation_axis_offset, dtype=float).reshape(3)

        self._prepare_static_state()
        self._cached_signature = None
        self._cached_forces = self._force_buffer.view()
        self._cached_forces.setflags(write=False)

    @classmethod
    def from_components(
        cls,
        components: Sequence[Component],
        airfoils: Mapping[str, AirfoilFFT],
        viv_params: Optional[VIV_Params] = None,
        *,
        n_freq_depth: Optional[int] = None,
        node_ids: Optional[Sequence[str]] = None,
        force_scale: float = 1.0,
    ) -> "VorLapQBladeController":
        """Construct a controller from already-converted VorLap components."""
        return cls(
            components=components,
            airfoils=airfoils,
            viv_params=viv_params,
            n_freq_depth=n_freq_depth,
            node_ids=node_ids,
            force_scale=force_scale,
        )

    @classmethod
    def from_qblade_sim(
        cls,
        sim_path: str,
        airfoils: Mapping[str, AirfoilFFT],
        *,
        default_airfoil_id: str = "default",
        include_struts: bool = True,
        include_tower: bool = True,
        tower_airfoil_id: str = "cylinder",
        n_freq_depth: Optional[int] = None,
        force_scale: float = 1.0,
    ) -> "VorLapQBladeController":
        """Build a controller directly from a QBlade `.sim` file."""
        from .fileio import convert_qblade_to_vorlap_inputs

        components, viv_params, node_ids = convert_qblade_to_vorlap_inputs(
            sim_path,
            default_airfoil_id=default_airfoil_id,
            include_struts=include_struts,
            include_tower=include_tower,
            tower_airfoil_id=tower_airfoil_id,
        )
        return cls(
            components=components,
            airfoils=airfoils,
            viv_params=viv_params,
            n_freq_depth=n_freq_depth,
            node_ids=node_ids,
            force_scale=force_scale,
        )

    def _prepare_static_state(self) -> None:
        """Flatten component geometry into node-wise arrays and cache airfoil groups."""
        node_count = sum(np.asarray(comp.shape_xyz, dtype=float).shape[0] for comp in self.components)
        if node_count < 1:
            raise ValueError("Controller requires at least one node.")
        if self.node_ids is not None and len(self.node_ids) != node_count:
            raise ValueError("node_ids length must match the flattened node count.")

        self._node_count = int(node_count)
        self._base_chord = np.empty((node_count, 3), dtype=float)
        self._base_normal = np.empty((node_count, 3), dtype=float)
        self._global_pos = np.empty((node_count, 3), dtype=float)
        self._chord = np.empty(node_count, dtype=float)
        self._local_length = np.empty(node_count, dtype=float)
        self._base_node_ids: List[str] = []
        self._node_airfoil_ids: List[str] = []

        node_cursor = 0
        for icomp, comp in enumerate(self.components):
            xyz = np.asarray(comp.shape_xyz, dtype=float)
            if xyz.ndim != 2 or xyz.shape[1] != 3:
                raise ValueError(f"Component '{comp.id}' has invalid shape_xyz with shape {xyz.shape}.")

            npts = xyz.shape[0]
            chord = np.asarray(comp.chord, dtype=float).reshape(-1)
            twist = np.asarray(comp.twist, dtype=float).reshape(-1)
            thickness = np.asarray(comp.thickness, dtype=float).reshape(-1)
            offset = np.asarray(comp.offset, dtype=float).reshape(-1)
            if not (len(chord) == len(twist) == len(thickness) == len(offset) == npts):
                raise ValueError(
                    f"Component '{comp.id}' has inconsistent vector lengths: "
                    f"xyz={npts}, chord={len(chord)}, twist={len(twist)}, thickness={len(thickness)}, offset={len(offset)}."
                )

            airfoil_ids = _normalize_airfoil_ids(getattr(comp, "airfoil_ids", None), npts)

            chord_vec = np.asarray(comp.chord_vector, dtype=float) if getattr(comp, "chord_vector", None) is not None else None
            normal_vec = np.asarray(comp.normal_vector, dtype=float) if getattr(comp, "normal_vector", None) is not None else None

            fallback_chord, fallback_normal, global_pos = _build_fallback_frame(comp)

            use_provided = (
                chord_vec is not None
                and normal_vec is not None
                and chord_vec.shape == (npts, 3)
                and normal_vec.shape == (npts, 3)
            )
            if use_provided:
                chord_norm = np.linalg.norm(chord_vec, axis=1)
                normal_norm = np.linalg.norm(normal_vec, axis=1)
                provided_good = (chord_norm > _EPS) & (normal_norm > _EPS)
                chord_final = fallback_chord.copy()
                normal_final = fallback_normal.copy()
                if np.any(provided_good):
                    chord_final[provided_good] = chord_vec[provided_good] / chord_norm[provided_good][:, None]
                    normal_final[provided_good] = normal_vec[provided_good] / normal_norm[provided_good][:, None]
            else:
                chord_final = fallback_chord
                normal_final = fallback_normal

            for ipt in range(npts):
                self._base_chord[node_cursor + ipt] = chord_final[ipt]
                self._base_normal[node_cursor + ipt] = normal_final[ipt]
                self._global_pos[node_cursor + ipt] = global_pos[ipt]
                self._chord[node_cursor + ipt] = float(chord[ipt])
                self._local_length[node_cursor + ipt] = float(_compute_local_segment_length(xyz, ipt))
                self._node_airfoil_ids.append(airfoil_ids[ipt])
                if self.node_ids is None:
                    self._base_node_ids.append(f"{comp.id}_{ipt + 1}")
                else:
                    self._base_node_ids.append(str(self.node_ids[node_cursor + ipt]))

            node_cursor += npts

        self.node_ids = tuple(self._base_node_ids)

        warned_missing_airfoils: set = set()
        group_indices: Dict[str, List[int]] = {}
        group_airfoils: Dict[str, AirfoilFFT] = {}
        for idx, airfoil_id in enumerate(self._node_airfoil_ids):
            group_indices.setdefault(airfoil_id, []).append(idx)
            if airfoil_id not in group_airfoils:
                group_airfoils[airfoil_id] = _resolve_airfoil(
                    self.airfoils,
                    airfoil_id,
                    warned_missing_airfoils=warned_missing_airfoils,
                )

        self._groups = []
        for airfoil_id, indices in group_indices.items():
            afft = group_airfoils[airfoil_id]
            afft._cache_interpolators()
            depth = min(
                self.n_freq_depth,
                int(np.asarray(afft.CL_ST).shape[2]),
                int(np.asarray(afft.CD_ST).shape[2]),
            )
            if depth < 1:
                raise ValueError(f"Airfoil '{airfoil_id}' does not contain any usable spectrum depth.")
            self._groups.append(
                {
                    "airfoil_id": airfoil_id,
                    "afft": afft,
                    "indices": np.asarray(indices, dtype=int),
                    "depth": depth,
                }
            )

        self._force_buffer = np.empty((node_count, 3), dtype=float)
        self._force_buffer.fill(0.0)

    def _coerce_velocity(self, node_data: Union[np.ndarray, Mapping[str, np.ndarray]], velocity: Optional[np.ndarray]) -> np.ndarray:
        """Normalize node velocities into an [nnodes, 3] array."""
        if velocity is None:
            if isinstance(node_data, Mapping):
                for key in ("velocity", "relative_velocity", "inflow_velocity", "flow_velocity"):
                    if key in node_data:
                        velocity = node_data[key]
                        break
                else:
                    raise KeyError(
                        "node_data must contain one of: velocity, relative_velocity, inflow_velocity, flow_velocity."
                    )
            else:
                velocity = node_data

        vel = np.asarray(velocity, dtype=float)
        if vel.ndim == 1:
            if vel.size == 3:
                vel = np.broadcast_to(vel.reshape(1, 3), (self._node_count, 3))
            elif vel.size == 3 * self._node_count:
                vel = vel.reshape(self._node_count, 3)
            else:
                raise ValueError(
                    f"Velocity vector must have length 3 or {3 * self._node_count}; got {vel.size}."
                )
        elif vel.ndim == 2:
            if vel.shape == (self._node_count, 3):
                pass
            elif vel.shape == (3, self._node_count):
                vel = vel.T
            else:
                raise ValueError(
                    f"Velocity array must have shape [{self._node_count}, 3] or [3, {self._node_count}]; got {vel.shape}."
                )
        else:
            raise ValueError("Velocity input must be one- or two-dimensional.")
        if np.any(~np.isfinite(vel)):
            raise ValueError("Velocity input contains non-finite values.")
        return np.asarray(vel, dtype=float)

    @staticmethod
    def _coerce_scalar(
        node_data: Union[np.ndarray, Mapping[str, np.ndarray]],
        value: Optional[Union[float, np.ndarray]],
        keys: Tuple[str, ...],
        default: float,
        name: str,
    ) -> float:
        """Normalize a scalar control input from either kwargs or a mapping."""
        if value is None and isinstance(node_data, Mapping):
            for key in keys:
                if key in node_data:
                    value = node_data[key]
                    break
        if value is None:
            return float(default)
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != 1:
            raise ValueError(f"{name} must be a scalar value.")
        scalar = float(arr[0])
        if not math.isfinite(scalar):
            raise ValueError(f"{name} must be finite.")
        return scalar

    def step(
        self,
        node_data: Union[np.ndarray, Mapping[str, np.ndarray]],
        *,
        time: Optional[Union[float, np.ndarray]] = None,
        azimuth_deg: Optional[Union[float, np.ndarray]] = None,
        velocity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate one timestep and return per-node global force vectors.

        Args:
            node_data: Mapping or array containing per-node velocity vectors.
            time: Scalar time used to reconstruct the harmonic force signal.
            azimuth_deg: Scalar rotor azimuth used to rotate geometry and inflow.
            velocity: Optional explicit velocity override.

        Returns:
            A read-only array with shape `[nnodes, 3]`.
        """
        t = self._coerce_scalar(node_data, time, ("time", "t"), 0.0, "time")
        az = self._coerce_scalar(node_data, azimuth_deg, ("azimuth_deg", "azimuth", "rotation_deg"), 0.0, "azimuth_deg")
        vel = self._coerce_velocity(node_data, velocity)

        if self._cached_signature is not None:
            cached_t, cached_az, cached_vel = self._cached_signature
            if cached_t == t and cached_az == az and np.array_equal(cached_vel, vel):
                return self._cached_forces
        vel_snapshot = np.array(vel, copy=True)

        from .interpolation import interpolate_fft_spectrum_batch

        rot = _axis_angle_matrix(self.rotation_axis, az)
        chord = self._base_chord @ rot.T
        normal = self._base_normal @ rot.T

        v_chord = np.einsum("ij,ij->i", vel, chord)
        v_normal = np.einsum("ij,ij->i", vel, normal)

        aoa_deg = np.degrees(np.arctan2(v_normal, v_chord))
        v_eff = np.hypot(v_normal, v_chord)
        reynolds = self.fluid_density * v_eff * self._chord / self.fluid_dynamicviscosity
        dynamic_pressure = 0.5 * self.fluid_density * (v_eff ** 2) * self._chord * self._local_length
        st_length = np.maximum(self._chord * np.abs(np.sin(np.deg2rad(aoa_deg))), _EPS)
        yaw_deg = np.degrees(np.arctan2(chord[:, 1], chord[:, 0]))
        roll_deg = np.degrees(np.arctan2(normal[:, 2], normal[:, 1]))

        self._force_buffer.fill(0.0)

        for group in self._groups:
            idx = group["indices"]
            afft = group["afft"]
            depth = group["depth"]

            st_cl, amp_cl, pha_cl = interpolate_fft_spectrum_batch(afft, reynolds[idx], aoa_deg[idx], "CL", n_freq_depth=depth)
            st_cd, amp_cd, pha_cd = interpolate_fft_spectrum_batch(afft, reynolds[idx], aoa_deg[idx], "CD", n_freq_depth=depth)

            scale = v_eff[idx] / st_length[idx]
            drag = dynamic_pressure[idx] * amp_cd[:, 0]
            lift = dynamic_pressure[idx] * amp_cl[:, 0]

            if depth > 1:
                omega_cl_t = 2.0 * np.pi * (st_cl[:, 1:] * scale[:, None]) * t + pha_cl[:, 1:]
                omega_cd_t = 2.0 * np.pi * (st_cd[:, 1:] * scale[:, None]) * t + pha_cd[:, 1:]
                lift = lift + np.sum((dynamic_pressure[idx][:, None] * amp_cl[:, 1:]) * np.cos(omega_cl_t), axis=1)
                drag = drag + np.sum((dynamic_pressure[idx][:, None] * amp_cd[:, 1:]) * np.cos(omega_cd_t), axis=1)

            yaw = np.deg2rad(yaw_deg[idx])
            roll = np.deg2rad(roll_deg[idx])
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            cos_roll = np.cos(roll)
            sin_roll = np.sin(roll)

            self._force_buffer[idx, 0] = drag * cos_yaw - lift * cos_roll * sin_yaw
            self._force_buffer[idx, 1] = drag * sin_yaw + lift * cos_roll * cos_yaw
            self._force_buffer[idx, 2] = lift * sin_roll

        if self.force_scale != 1.0:
            self._force_buffer *= self.force_scale

        self._cached_signature = (t, az, vel_snapshot)
        return self._cached_forces

    def step_from_swap(
        self,
        swap: SwapArrayAdapter,
        *,
        velocity_block: str = "velocity",
        time_block: str = "time",
        azimuth_block: str = "azimuth_deg",
        force_block: str = "force",
    ) -> np.ndarray:
        """
        Read inputs from a swap adapter, evaluate a step, and write forces back.
        """
        if not isinstance(swap, SwapArrayAdapter):
            raise TypeError("swap must be a SwapArrayAdapter instance.")
        node_data = {
            "velocity": swap.read(velocity_block),
            "time": swap.scalar(time_block),
            "azimuth_deg": swap.scalar(azimuth_block),
        }
        forces = self.step(node_data)
        swap.write(force_block, forces)
        return forces


VorLapController = VorLapQBladeController
QBladeController = VorLapQBladeController


__all__ = [
    "QBladeController",
    "SwapArrayAdapter",
    "SwapBlockSpec",
    "VorLapController",
    "VorLapQBladeController",
]
