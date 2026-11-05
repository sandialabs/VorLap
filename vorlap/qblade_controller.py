"""Stateful per-timestep controller utilities for QBlade-style coupling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy as _copy
from dataclasses import dataclass
import math
import numbers
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .force_model import (
    compute_local_segment_length,
    force_from_chord_normal,
    harmonic_force_channel,
    harmonic_force_channel_from_angle,
    resolve_airfoil,
    section_flow_state,
    strouhal_reference_length,
)
from .geometry_frames import build_fallback_frame, rotationMatrix
from .qblade_response_model import (
    AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW,
    QBladeResponseDiagnostics,
    QBladeResponseDiagnosticsV2,
    QBladeResponseModel,
    QBladeSeededResponseDiagnostics,
    QBladeSeededResponseDiagnosticsV2,
    RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW,
)
from .qblade_modal_power_model import (
    QBladeModalPowerDiagnostics,
    QBladeModalPowerModel,
)
from .structs import AirfoilFFT, Component, VIV_Params

_EPS = 1.0e-12
SOURCE_BOUNDS_POLICIES = {"warn", "raise", "clip"}
ResponseModel = Union[QBladeResponseModel, QBladeModalPowerModel]
ResponseDiagnostics = Union[
    QBladeResponseDiagnostics,
    QBladeModalPowerDiagnostics,
]


def _normalize_source_bounds_policy(policy: str) -> str:
    normalized = str(policy).strip().lower()
    if normalized not in SOURCE_BOUNDS_POLICIES:
        raise ValueError("source_bounds_policy must be one of: 'warn', 'raise', or 'clip'.")
    return normalized


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


@dataclass(frozen=True)
class ControllerSourceState:
    """Per-node source-grid and force diagnostics from the last controller step."""

    node_ids: np.ndarray
    component_ids: np.ndarray
    airfoil_ids: np.ndarray
    source_airfoil_ids: np.ndarray
    reynolds_models: np.ndarray
    aoa_deg: np.ndarray
    reynolds: np.ndarray
    freq_min_hz: np.ndarray
    freq_max_hz: np.ndarray
    unfiltered_freq_max_hz: np.ndarray
    frequency_truncated_count: np.ndarray
    # Legacy source-bound metrics; these intentionally exclude response force.
    force_norm_n: np.ndarray
    force_coefficient_abs: np.ndarray
    response_force_norm_n: np.ndarray
    response_force_coefficient_abs: np.ndarray
    total_force_norm_n: np.ndarray
    total_force_coefficient_abs: np.ndarray
    source_aoa_min_deg: np.ndarray
    source_aoa_max_deg: np.ndarray
    source_re_min: np.ndarray
    source_re_max: np.ndarray
    source_force_coefficient_max_abs: np.ndarray

    def __post_init__(self) -> None:
        node_ids = np.asarray(self.node_ids, dtype=object).reshape(-1)
        object.__setattr__(self, "node_ids", np.array(node_ids, copy=True))
        node_count = int(node_ids.size)
        for name in ("component_ids", "airfoil_ids", "source_airfoil_ids", "reynolds_models"):
            values = np.asarray(getattr(self, name), dtype=object).reshape(-1)
            if values.size != node_count:
                raise ValueError(f"{name} length must match node_ids.")
            object.__setattr__(self, name, np.array(values, copy=True))
        for name in (
            "aoa_deg",
            "reynolds",
            "freq_min_hz",
            "freq_max_hz",
            "unfiltered_freq_max_hz",
            "force_norm_n",
            "force_coefficient_abs",
            "response_force_norm_n",
            "response_force_coefficient_abs",
            "total_force_norm_n",
            "total_force_coefficient_abs",
            "source_aoa_min_deg",
            "source_aoa_max_deg",
            "source_re_min",
            "source_re_max",
            "source_force_coefficient_max_abs",
        ):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if values.size != node_count:
                raise ValueError(f"{name} length must match node_ids.")
            object.__setattr__(self, name, np.array(values, copy=True))
        truncated = np.asarray(self.frequency_truncated_count, dtype=int).reshape(-1)
        if truncated.size != node_count:
            raise ValueError("frequency_truncated_count length must match node_ids.")
        if np.any(truncated < 0):
            raise ValueError("frequency_truncated_count values must be non-negative.")
        object.__setattr__(self, "frequency_truncated_count", np.array(truncated, copy=True))

    @property
    def node_count(self) -> int:
        return int(self.node_ids.size)

    def copy(self) -> "ControllerSourceState":
        return type(self)(**self.as_dict(copy=True))

    def as_dict(self, *, copy: bool = True) -> Dict[str, Any]:
        data = {
            "node_ids": self.node_ids,
            "component_ids": self.component_ids,
            "airfoil_ids": self.airfoil_ids,
            "source_airfoil_ids": self.source_airfoil_ids,
            "reynolds_models": self.reynolds_models,
            "aoa_deg": self.aoa_deg,
            "reynolds": self.reynolds,
            "freq_min_hz": self.freq_min_hz,
            "freq_max_hz": self.freq_max_hz,
            "unfiltered_freq_max_hz": self.unfiltered_freq_max_hz,
            "frequency_truncated_count": self.frequency_truncated_count,
            "force_norm_n": self.force_norm_n,
            "force_coefficient_abs": self.force_coefficient_abs,
            "response_force_norm_n": self.response_force_norm_n,
            "response_force_coefficient_abs": self.response_force_coefficient_abs,
            "total_force_norm_n": self.total_force_norm_n,
            "total_force_coefficient_abs": self.total_force_coefficient_abs,
            "source_aoa_min_deg": self.source_aoa_min_deg,
            "source_aoa_max_deg": self.source_aoa_max_deg,
            "source_re_min": self.source_re_min,
            "source_re_max": self.source_re_max,
            "source_force_coefficient_max_abs": self.source_force_coefficient_max_abs,
        }
        if not copy:
            return dict(data)
        return {key: np.array(value, copy=True) for key, value in data.items()}


class VorLapQBladeController:
    """
    Stateful per-timestep controller for QBlade-style external force calls.

    The controller performs one-time geometry preparation, groups airfoil
    spectra by node, and evaluates a single timestep with vectorized NumPy
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
        include_mean_load: bool = False,
        source_bounds_policy: str = "warn",
        source_frequency_cap_hz: Optional[float] = None,
        response_model: Optional[ResponseModel] = None,
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
        self.include_mean_load = bool(include_mean_load)
        self.source_bounds_policy = _normalize_source_bounds_policy(source_bounds_policy)
        self.source_frequency_cap_hz = None if source_frequency_cap_hz is None else float(source_frequency_cap_hz)
        if self.source_frequency_cap_hz is not None and (
            not math.isfinite(self.source_frequency_cap_hz) or self.source_frequency_cap_hz <= 0.0
        ):
            raise ValueError("source_frequency_cap_hz must be finite and positive when provided.")
        if response_model is not None and not isinstance(
            response_model,
            (QBladeResponseModel, QBladeModalPowerModel),
        ):
            raise TypeError(
                "response_model must be a QBladeResponseModel, "
                "QBladeModalPowerModel, or None."
            )
        self.response_model = response_model

        self.fluid_density = float(self.viv_params.fluid_density)
        self.fluid_dynamicviscosity = float(self.viv_params.fluid_dynamicviscosity)
        self.rotation_axis = _normalize_vector(self.viv_params.rotation_axis, "rotation_axis")
        self.geometry_reference_azimuth_deg = float(self.viv_params.geometry_reference_azimuth_deg)
        if not math.isfinite(self.geometry_reference_azimuth_deg):
            raise ValueError("geometry_reference_azimuth_deg must be finite.")
        self.rotation_axis_offset = np.asarray(self.viv_params.rotation_axis_offset, dtype=float).reshape(3)

        self._prepare_static_state()
        if self.response_model is not None:
            response_node_ids = tuple(str(node_id) for node_id in self.response_model.config.node_ids)
            if response_node_ids != self.node_ids:
                raise ValueError(
                    "response_model node_ids must exactly match controller node_ids and order."
                )
        self._response_model_enabled = bool(
            self.response_model is not None and self.response_model.config.enabled
        )
        self._phase_time_s: Optional[float] = None
        self._phase_angle_cl_rad: Optional[np.ndarray] = None
        self._phase_angle_cd_rad: Optional[np.ndarray] = None
        self._phase_frequency_cl_hz: Optional[np.ndarray] = None
        self._phase_frequency_cd_hz: Optional[np.ndarray] = None
        if self._response_model_enabled:
            phase_shape = (self._node_count, self.n_freq_depth)
            self._phase_angle_cl_rad = np.zeros(phase_shape, dtype=float)
            self._phase_angle_cd_rad = np.zeros(phase_shape, dtype=float)
            self._phase_frequency_cl_hz = np.zeros(phase_shape, dtype=float)
            self._phase_frequency_cd_hz = np.zeros(phase_shape, dtype=float)
        self._cached_signature = None
        self._cached_forces = self._force_buffer.view()
        self._cached_forces.setflags(write=False)
        self._cache_hits = 0
        self._cache_misses = 0
        self._last_source_state: Optional[ControllerSourceState] = None
        self._last_response_diagnostics: Optional[ResponseDiagnostics] = None
        if isinstance(self.response_model, QBladeResponseModel):
            response_model_type = "modal_negative_damping"
        elif isinstance(self.response_model, QBladeModalPowerModel):
            response_model_type = "modal_aerodynamic_power"
        else:
            response_model_type = "none"
        self._response_diagnostics_summary: Dict[str, Any] = {
            "model_type": response_model_type,
            "samples": 0,
            "gain_limited_steps": 0,
            "force_limited_steps": 0,
            "gain_limited_node_samples": 0,
            "force_limited_node_samples": 0,
            "max_gain_limited_nodes": 0,
            "max_force_limited_nodes": 0,
            "max_amplitude_ratio": 0.0,
            "max_abs_modal_power_w": 0.0,
            "max_response_nodal_force_n": 0.0,
            "target_power_limited_steps": 0,
            "generalized_force_limited_steps": 0,
            "nodal_force_limited_steps": 0,
            "nodal_force_limited_node_samples": 0,
            "max_nodal_force_limited_nodes": 0,
            "max_amplitude_estimate_m": 0.0,
            "max_abs_target_cycle_mean_power_w": 0.0,
            "max_abs_applied_target_cycle_mean_power_w": 0.0,
            "max_abs_instantaneous_modal_power_w": 0.0,
            "max_abs_instantaneous_structural_power_w": 0.0,
            "max_abs_realized_modal_power_w": 0.0,
            "max_abs_realized_structural_power_w": 0.0,
        }
        if (
            isinstance(self.response_model, QBladeResponseModel)
            and self.response_model.schema_version
            == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        ):
            self._response_diagnostics_summary.update(
                {
                    "response_model_schema_version": self.response_model.schema_version,
                    "aerodynamic_speed_basis": self.response_model.aerodynamic_speed_basis,
                    "response_section_width_basis": (
                        self.response_model.response_section_width_basis
                    ),
                    "response_config_fingerprint_sha256": (
                        self.response_model.config_fingerprint
                    ),
                    "response_declared_ambient_crossflow_speed_min_mps": None,
                    "response_declared_ambient_crossflow_speed_max_mps": 0.0,
                    "response_relative_crossflow_speed_min_mps": None,
                    "response_relative_crossflow_speed_max_mps": 0.0,
                    "response_section_width_min_m": None,
                    "response_section_width_max_m": 0.0,
                    "configured_response_reduced_velocity_min": None,
                    "configured_response_reduced_velocity_max": 0.0,
                    "realized_response_reduced_velocity_min": None,
                    "realized_response_reduced_velocity_max": 0.0,
                    "speed_basis_invariant_violations": 0,
                    "activation_invariant_violations": 0,
                    "speed_basis_invariant_violation_steps": 0,
                    "activation_invariant_violation_steps": 0,
                }
            )
        if (
            isinstance(self.response_model, QBladeResponseModel)
            and self.response_model.startup_seed_enabled
        ):
            seed = self.response_model.config.startup_seed
            if seed is None:  # pragma: no cover - protected model invariant
                raise RuntimeError("Enabled startup seed configuration is missing.")
            self._response_diagnostics_summary.update(
                {
                    "startup_seed_samples": 0,
                    "startup_seed_active_steps": 0,
                    "startup_seed_force_limited_steps": 0,
                    "startup_seed_force_limited_node_samples": 0,
                    "max_startup_seed_force_limited_nodes": 0,
                    "max_abs_startup_seed_modal_acceleration_mps2": 0.0,
                    "max_abs_startup_seed_raw_generalized_force_n": 0.0,
                    "max_startup_seed_raw_nodal_force_n": 0.0,
                    "startup_seed_target_amplitude_ratio": float(
                        seed.target_amplitude_ratio
                    ),
                    "startup_seed_target_amplitude_m": (
                        float(seed.target_amplitude_ratio)
                        * float(self.response_model.config.reference_length_m)
                    ),
                }
            )

    @property
    def cache_hits(self) -> int:
        """Number of exact-input force evaluations served from the cache."""
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        """Number of force evaluations that recomputed the force buffer."""
        return self._cache_misses

    @property
    def cache_stats(self) -> Dict[str, int]:
        """Return exact-input cache counters for logging and diagnostics."""
        return {"hits": self._cache_hits, "misses": self._cache_misses}

    @property
    def last_source_state(self) -> Optional[Dict[str, Any]]:
        """Return the last per-node aerodynamic source state as a compatibility dict."""
        if self._last_source_state is None:
            return None
        return self._last_source_state.as_dict(copy=True)

    @property
    def last_source_diagnostics(self) -> Optional[ControllerSourceState]:
        """Return the last per-node aerodynamic source state as a typed object."""
        if self._last_source_state is None:
            return None
        return self._last_source_state.copy()

    @property
    def response_model_enabled(self) -> bool:
        """Whether response-aware forcing and phase integration are active."""
        return self._response_model_enabled

    @property
    def response_model_type(self) -> str:
        """Stable portable type name for the configured response closure."""
        return str(self._response_diagnostics_summary["model_type"])

    @property
    def last_response_diagnostics(self) -> Optional[ResponseDiagnostics]:
        """Typed diagnostics from the most recent response-force evaluation."""
        return self._last_response_diagnostics

    @property
    def response_diagnostics_summary(self) -> Dict[str, Any]:
        """Cumulative clipping and response extrema over accepted model steps."""
        return dict(self._response_diagnostics_summary)

    def _record_response_diagnostics(
        self,
        diagnostics: ResponseDiagnostics,
    ) -> None:
        """Accumulate model-specific limits and physically named power extrema."""

        summary = self._response_diagnostics_summary
        if isinstance(diagnostics, QBladeResponseDiagnostics):
            gain_limited = int(diagnostics.gain_limited_nodes)
            force_limited = int(diagnostics.force_limited_nodes)
            target_power_limited = False
            generalized_force_limited = False
            nodal_force_limited = force_limited
            amplitude_estimate_m = float(diagnostics.amplitude_estimate_m)
            amplitude_ratio = float(diagnostics.amplitude_ratio)
            target_power_w = 0.0
            applied_target_power_w = 0.0
            instantaneous_modal_power_w = float(diagnostics.modal_power_w)
            instantaneous_structural_power_w = float(
                diagnostics.structural_power_w
            )
            realized_modal_power_w = instantaneous_modal_power_w
            realized_structural_power_w = instantaneous_structural_power_w
            if isinstance(diagnostics, QBladeResponseDiagnosticsV2):
                for summary_name, diagnostic_name in (
                    (
                        "response_declared_ambient_crossflow_speed_min_mps",
                        "response_declared_ambient_crossflow_speed_min_mps",
                    ),
                    (
                        "response_relative_crossflow_speed_min_mps",
                        "response_relative_crossflow_speed_min_mps",
                    ),
                    (
                        "response_section_width_min_m",
                        "response_section_width_min_m",
                    ),
                    (
                        "configured_response_reduced_velocity_min",
                        "configured_response_reduced_velocity_min",
                    ),
                    (
                        "realized_response_reduced_velocity_min",
                        "realized_response_reduced_velocity_min",
                    ),
                ):
                    candidate = float(getattr(diagnostics, diagnostic_name))
                    previous = summary[summary_name]
                    summary[summary_name] = (
                        candidate
                        if previous is None
                        else min(float(previous), candidate)
                    )
                for summary_name, diagnostic_name in (
                    (
                        "response_declared_ambient_crossflow_speed_max_mps",
                        "response_declared_ambient_crossflow_speed_max_mps",
                    ),
                    (
                        "response_relative_crossflow_speed_max_mps",
                        "response_relative_crossflow_speed_max_mps",
                    ),
                    (
                        "response_section_width_max_m",
                        "response_section_width_max_m",
                    ),
                    (
                        "configured_response_reduced_velocity_max",
                        "configured_response_reduced_velocity_max",
                    ),
                    (
                        "realized_response_reduced_velocity_max",
                        "realized_response_reduced_velocity_max",
                    ),
                ):
                    summary[summary_name] = max(
                        float(summary[summary_name]),
                        float(getattr(diagnostics, diagnostic_name)),
                    )
                speed_violations = int(
                    diagnostics.speed_basis_invariant_violations
                )
                activation_violations = int(
                    diagnostics.activation_invariant_violations
                )
                summary["speed_basis_invariant_violations"] += speed_violations
                summary["activation_invariant_violations"] += activation_violations
                summary["speed_basis_invariant_violation_steps"] += int(
                    speed_violations > 0
                )
                summary["activation_invariant_violation_steps"] += int(
                    activation_violations > 0
                )
        elif isinstance(diagnostics, QBladeModalPowerDiagnostics):
            gain_limited = 0
            force_limited = int(diagnostics.nodal_force_limited_nodes)
            target_power_limited = bool(diagnostics.target_power_limited)
            generalized_force_limited = bool(
                diagnostics.generalized_force_limited
            )
            nodal_force_limited = int(diagnostics.nodal_force_limited_nodes)
            amplitude_estimate_m = float(diagnostics.amplitude_estimate_m)
            amplitude_ratio = 0.0
            target_power_w = float(diagnostics.target_cycle_mean_power_w)
            applied_target_power_w = float(
                diagnostics.applied_target_cycle_mean_power_w
            )
            instantaneous_modal_power_w = float(
                diagnostics.instantaneous_modal_power_w
            )
            instantaneous_structural_power_w = float(
                diagnostics.instantaneous_structural_power_w
            )
            realized_modal_power_w = float(
                diagnostics.realized_modal_power_ema_w
            )
            realized_structural_power_w = float(
                diagnostics.realized_structural_power_ema_w
            )
        else:  # pragma: no cover - constructor restricts response-model types
            raise TypeError("Unsupported response diagnostics type.")

        summary["samples"] += 1
        summary["gain_limited_steps"] += int(gain_limited > 0)
        summary["force_limited_steps"] += int(force_limited > 0)
        summary["gain_limited_node_samples"] += gain_limited
        summary["force_limited_node_samples"] += force_limited
        summary["max_gain_limited_nodes"] = max(
            int(summary["max_gain_limited_nodes"]),
            gain_limited,
        )
        summary["max_force_limited_nodes"] = max(
            int(summary["max_force_limited_nodes"]),
            force_limited,
        )
        summary["max_amplitude_ratio"] = max(
            float(summary["max_amplitude_ratio"]),
            amplitude_ratio,
        )
        summary["max_abs_modal_power_w"] = max(
            float(summary["max_abs_modal_power_w"]),
            abs(instantaneous_modal_power_w),
        )
        summary["max_response_nodal_force_n"] = max(
            float(summary["max_response_nodal_force_n"]),
            float(diagnostics.max_abs_nodal_force_n),
        )
        summary["target_power_limited_steps"] += int(target_power_limited)
        summary["generalized_force_limited_steps"] += int(
            generalized_force_limited
        )
        summary["nodal_force_limited_steps"] += int(nodal_force_limited > 0)
        summary["nodal_force_limited_node_samples"] += nodal_force_limited
        summary["max_nodal_force_limited_nodes"] = max(
            int(summary["max_nodal_force_limited_nodes"]),
            nodal_force_limited,
        )
        summary["max_amplitude_estimate_m"] = max(
            float(summary["max_amplitude_estimate_m"]),
            amplitude_estimate_m,
        )
        power_extrema = {
            "max_abs_target_cycle_mean_power_w": target_power_w,
            "max_abs_applied_target_cycle_mean_power_w": applied_target_power_w,
            "max_abs_instantaneous_modal_power_w": instantaneous_modal_power_w,
            "max_abs_instantaneous_structural_power_w": instantaneous_structural_power_w,
            "max_abs_realized_modal_power_w": realized_modal_power_w,
            "max_abs_realized_structural_power_w": realized_structural_power_w,
        }
        for name, value in power_extrema.items():
            summary[name] = max(float(summary[name]), abs(float(value)))

        if isinstance(
            diagnostics,
            (QBladeSeededResponseDiagnostics, QBladeSeededResponseDiagnosticsV2),
        ):
            seed_limited = int(diagnostics.startup_seed_force_limited_nodes)
            summary["startup_seed_samples"] += 1
            summary["startup_seed_active_steps"] += int(
                diagnostics.startup_seed_active
            )
            summary["startup_seed_force_limited_steps"] += int(
                seed_limited > 0
            )
            summary["startup_seed_force_limited_node_samples"] += seed_limited
            summary["max_startup_seed_force_limited_nodes"] = max(
                int(summary["max_startup_seed_force_limited_nodes"]),
                seed_limited,
            )
            summary["max_abs_startup_seed_modal_acceleration_mps2"] = max(
                float(summary["max_abs_startup_seed_modal_acceleration_mps2"]),
                abs(float(diagnostics.startup_seed_modal_acceleration_mps2)),
            )
            summary["max_abs_startup_seed_raw_generalized_force_n"] = max(
                float(summary["max_abs_startup_seed_raw_generalized_force_n"]),
                abs(float(diagnostics.startup_seed_raw_generalized_force_n)),
            )
            summary["max_startup_seed_raw_nodal_force_n"] = max(
                float(summary["max_startup_seed_raw_nodal_force_n"]),
                float(diagnostics.startup_seed_raw_max_abs_nodal_force_n),
            )

    def reset_cache_stats(self) -> None:
        """Reset exact-input cache counters without invalidating cached forces."""
        self._cache_hits = 0
        self._cache_misses = 0

    def _return_forces(self, copy: bool) -> np.ndarray:
        """Return either a stable copy or the internal read-only force view."""
        if copy:
            return np.array(self._cached_forces, copy=True)
        return self._cached_forces

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
        include_mean_load: bool = False,
        source_bounds_policy: str = "warn",
        source_frequency_cap_hz: Optional[float] = None,
        response_model: Optional[ResponseModel] = None,
    ) -> "VorLapQBladeController":
        """Construct a controller from already-converted VorLap components."""
        return cls(
            components=components,
            airfoils=airfoils,
            viv_params=viv_params,
            n_freq_depth=n_freq_depth,
            node_ids=node_ids,
            force_scale=force_scale,
            include_mean_load=include_mean_load,
            source_bounds_policy=source_bounds_policy,
            source_frequency_cap_hz=source_frequency_cap_hz,
            response_model=response_model,
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
        include_mean_load: bool = False,
        source_bounds_policy: str = "warn",
        source_frequency_cap_hz: Optional[float] = None,
        response_model: Optional[ResponseModel] = None,
    ) -> "VorLapQBladeController":
        """Build a controller directly from a QBlade `.sim` file."""
        from .qblade_conversion import convert_qblade_to_vorlap_inputs

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
            include_mean_load=include_mean_load,
            source_bounds_policy=source_bounds_policy,
            source_frequency_cap_hz=source_frequency_cap_hz,
            response_model=response_model,
        )

    def _prepare_static_state(self) -> None:
        """Flatten component geometry into node-wise arrays and group airfoil data."""
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
        self._thickness = np.empty(node_count, dtype=float)
        self._local_length = np.empty(node_count, dtype=float)
        self._source_aoa_min = np.empty(node_count, dtype=float)
        self._source_aoa_max = np.empty(node_count, dtype=float)
        self._source_re_min = np.empty(node_count, dtype=float)
        self._source_re_max = np.empty(node_count, dtype=float)
        self._source_airfoil_ids = np.empty(node_count, dtype=object)
        self._source_reynolds_models = np.empty(node_count, dtype=object)
        self._source_force_coefficient_max_abs = np.empty(node_count, dtype=float)
        self._base_node_ids: List[str] = []
        self._node_airfoil_ids: List[str] = []
        self._node_component_ids: List[str] = []

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

            fallback_chord, fallback_normal, global_pos = build_fallback_frame(comp)

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
                self._thickness[node_cursor + ipt] = float(thickness[ipt])
                self._local_length[node_cursor + ipt] = float(compute_local_segment_length(xyz, ipt))
                self._node_airfoil_ids.append(airfoil_ids[ipt])
                self._node_component_ids.append(str(comp.id))
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
                group_airfoils[airfoil_id] = resolve_airfoil(
                    self.airfoils,
                    airfoil_id,
                    warned_missing_airfoils=warned_missing_airfoils,
                )

        self._groups = []
        for airfoil_id, indices in group_indices.items():
            afft = group_airfoils[airfoil_id]
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
            idx = np.asarray(indices, dtype=int)
            self._source_aoa_min[idx] = float(np.min(afft.AOA))
            self._source_aoa_max[idx] = float(np.max(afft.AOA))
            self._source_re_min[idx] = float(np.min(afft.Re))
            self._source_re_max[idx] = float(np.max(afft.Re))
            self._source_airfoil_ids[idx] = str(afft.name)
            self._source_reynolds_models[idx] = str(getattr(afft, "reynolds_model", ""))
            source_cf_sum = np.sum(np.abs(afft.CF_Amp), axis=2)
            source_cl_sum = np.sum(np.abs(afft.CL_Amp), axis=2)
            source_cd_sum = np.sum(np.abs(afft.CD_Amp), axis=2)
            source_coeff = max(
                float(np.max(source_cf_sum)),
                float(np.max(np.hypot(source_cl_sum, source_cd_sum))),
            )
            self._source_force_coefficient_max_abs[idx] = source_coeff

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

    def _coerce_structural_velocity(
        self,
        node_data: Union[np.ndarray, Mapping[str, np.ndarray]],
        structural_velocity: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Read structural velocity only when the response model requires it."""
        if not self._response_model_enabled:
            return None
        if structural_velocity is None and isinstance(node_data, Mapping):
            for key in ("structural_velocity", "structure_velocity", "motion_velocity"):
                if key in node_data:
                    structural_velocity = node_data[key]
                    break
        if structural_velocity is None:
            raise KeyError(
                "Response-aware forcing requires node_data['structural_velocity']."
            )
        return self._coerce_velocity(node_data, structural_velocity)

    def _coerce_declared_ambient_velocity(
        self,
        node_data: Union[np.ndarray, Mapping[str, np.ndarray]],
        declared_ambient_velocity: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Read the explicit global ambient velocity for response schema v2."""

        if not isinstance(self.response_model, QBladeResponseModel):
            return None
        if not self._response_model_enabled:
            return None
        if (
            self.response_model.schema_version
            != RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        ):
            return None
        if declared_ambient_velocity is None and isinstance(node_data, Mapping):
            for key in (
                "declared_ambient_velocity",
                "ambient_velocity",
                "declared_inflow_velocity",
            ):
                if key in node_data:
                    declared_ambient_velocity = node_data[key]
                    break
        if declared_ambient_velocity is None:
            raise KeyError(
                "Response schema_version 2 requires "
                "node_data['declared_ambient_velocity']."
            )
        return self._coerce_velocity(
            declared_ambient_velocity,
            declared_ambient_velocity,
        )

    def checkpoint_state(self) -> Dict[str, Any]:
        """Return JSON-serializable phase, response, cache, and diagnostic state."""
        phase_state: Optional[Dict[str, Any]] = None
        if self._response_model_enabled:
            phase_state = {
                "time_s": self._phase_time_s,
                "angle_cl_rad": self._phase_angle_cl_rad.tolist(),
                "angle_cd_rad": self._phase_angle_cd_rad.tolist(),
                "frequency_cl_hz": self._phase_frequency_cl_hz.tolist(),
                "frequency_cd_hz": self._phase_frequency_cd_hz.tolist(),
            }

        cached_state: Optional[Dict[str, Any]] = None
        if self._cached_signature is not None:
            cached_t = float(self._cached_signature[0])
            cached_azimuth = float(self._cached_signature[1])
            cached_state = {
                "time_s": cached_t,
                "azimuth_deg": cached_azimuth,
                "relative_velocity_mps": np.asarray(
                    self._cached_signature[2], dtype=float
                ).tolist(),
                "structural_velocity_mps": (
                    None
                    if len(self._cached_signature) < 4
                    else np.asarray(self._cached_signature[3], dtype=float).tolist()
                ),
                "force_n": np.asarray(self._cached_forces, dtype=float).tolist(),
            }
            if (
                isinstance(self.response_model, QBladeResponseModel)
                and self.response_model.schema_version
                == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
            ):
                cached_state["declared_ambient_velocity_mps"] = (
                    None
                    if len(self._cached_signature) < 5
                    else np.asarray(
                        self._cached_signature[4], dtype=float
                    ).tolist()
                )

        source_state: Optional[Dict[str, Any]] = None
        if self._last_source_state is not None:
            source_state = {
                key: np.asarray(value).tolist()
                for key, value in self._last_source_state.as_dict(copy=False).items()
            }

        return {
            "schema": "vorlap-qblade-controller-state-v1",
            "response_model_enabled": self._response_model_enabled,
            "phase_state": phase_state,
            "response_model_state": (
                None
                if self.response_model is None
                else self.response_model.checkpoint_state()
            ),
            "cached_step": cached_state,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "last_source_state": source_state,
            "response_diagnostics_summary": self.response_diagnostics_summary,
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Validate a checkpoint completely, then commit it transactionally."""
        if state.get("schema") != "vorlap-qblade-controller-state-v1":
            raise ValueError("Unsupported QBlade controller checkpoint schema.")
        if bool(state.get("response_model_enabled", False)) != self._response_model_enabled:
            raise ValueError("QBlade controller response-model checkpoint mismatch.")
        is_response_schema_v2 = (
            isinstance(self.response_model, QBladeResponseModel)
            and self.response_model.schema_version
            == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        )

        restored_phase_time = self._phase_time_s
        restored_phase_arrays: dict[str, np.ndarray] | None = None
        restored_response_diagnostics = self._last_response_diagnostics
        restored_source_state = self._last_source_state
        restored_summary = dict(self._response_diagnostics_summary)
        restored_cached_signature: tuple[Any, ...] | None = None
        restored_force: np.ndarray | None = None

        phase_state = state.get("phase_state")
        if self._response_model_enabled:
            if not isinstance(phase_state, Mapping):
                raise ValueError("Enabled response model requires checkpoint phase_state.")
            phase_shape = (self._node_count, self.n_freq_depth)
            restored_arrays = {}
            for state_name in (
                "angle_cl_rad",
                "angle_cd_rad",
                "frequency_cl_hz",
                "frequency_cd_hz",
            ):
                array = np.asarray(phase_state.get(state_name), dtype=float)
                if array.shape != phase_shape or np.any(~np.isfinite(array)):
                    raise ValueError(
                        f"Checkpoint {state_name} must be finite with shape {phase_shape}."
                    )
                restored_arrays[state_name] = np.array(array, copy=True)
            phase_time_raw = phase_state.get("time_s")
            if phase_time_raw is None:
                phase_time = None
            else:
                phase_time = float(phase_time_raw)
                if not math.isfinite(phase_time):
                    raise ValueError("Checkpoint phase time must be finite or null.")
            restored_phase_time = phase_time
            restored_phase_arrays = restored_arrays
        elif phase_state is not None:
            raise ValueError("Disabled response model cannot restore phase_state.")

        response_state = state.get("response_model_state")
        if self.response_model is None:
            if response_state is not None:
                raise ValueError("Checkpoint contains response state for an absent model.")
        else:
            if not isinstance(response_state, Mapping):
                raise ValueError("Checkpoint response_model_state must be a mapping.")
            response_model_probe = _copy.deepcopy(self.response_model)
            response_model_probe.restore_checkpoint_state(response_state)
            restored_response_diagnostics = response_model_probe.last_diagnostics

        source_state = state.get("last_source_state")
        if source_state is None:
            restored_source_state = None
        elif isinstance(source_state, Mapping):
            try:
                restored_source_state = ControllerSourceState(**dict(source_state))
            except (TypeError, ValueError) as exc:
                raise ValueError("Checkpoint last_source_state is invalid.") from exc
        else:
            raise ValueError("Checkpoint last_source_state must be a mapping or null.")

        summary_state = state.get("response_diagnostics_summary")
        if self._response_model_enabled and not isinstance(summary_state, Mapping):
            raise ValueError(
                "Enabled response checkpoint requires response_diagnostics_summary."
            )
        if isinstance(summary_state, Mapping):
            restored_model_type = str(
                summary_state.get("model_type", restored_summary["model_type"])
            )
            if restored_model_type != restored_summary["model_type"]:
                raise ValueError(
                    "Checkpoint response diagnostics model_type does not match."
                )
            restored_summary["model_type"] = restored_model_type
            if is_response_schema_v2:
                for name in (
                    "response_model_schema_version",
                    "aerodynamic_speed_basis",
                    "response_section_width_basis",
                    "response_config_fingerprint_sha256",
                ):
                    expected = restored_summary[name]
                    actual = summary_state.get(name, expected)
                    if actual != expected:
                        raise ValueError(f"Checkpoint {name} mismatch.")
                    restored_summary[name] = expected
            for name in (
                "samples",
                "gain_limited_steps",
                "force_limited_steps",
                "gain_limited_node_samples",
                "force_limited_node_samples",
                "max_gain_limited_nodes",
                "max_force_limited_nodes",
                "target_power_limited_steps",
                "generalized_force_limited_steps",
                "nodal_force_limited_steps",
                "nodal_force_limited_node_samples",
                "max_nodal_force_limited_nodes",
            ):
                value = int(summary_state.get(name, 0) or 0)
                if value < 0:
                    raise ValueError(f"Checkpoint {name} must be non-negative.")
                restored_summary[name] = value
            if is_response_schema_v2:
                for name in (
                    "speed_basis_invariant_violations",
                    "activation_invariant_violations",
                    "speed_basis_invariant_violation_steps",
                    "activation_invariant_violation_steps",
                ):
                    value = int(summary_state.get(name, 0) or 0)
                    if value < 0:
                        raise ValueError(f"Checkpoint {name} must be non-negative.")
                    restored_summary[name] = value
            for name in (
                "max_amplitude_ratio",
                "max_abs_modal_power_w",
                "max_response_nodal_force_n",
                "max_amplitude_estimate_m",
                "max_abs_target_cycle_mean_power_w",
                "max_abs_applied_target_cycle_mean_power_w",
                "max_abs_instantaneous_modal_power_w",
                "max_abs_instantaneous_structural_power_w",
                "max_abs_realized_modal_power_w",
                "max_abs_realized_structural_power_w",
            ):
                value = float(summary_state.get(name, 0.0) or 0.0)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(f"Checkpoint {name} must be finite and non-negative.")
                restored_summary[name] = value
            if is_response_schema_v2:
                for name in (
                    "response_declared_ambient_crossflow_speed_min_mps",
                    "response_relative_crossflow_speed_min_mps",
                    "response_section_width_min_m",
                    "configured_response_reduced_velocity_min",
                    "realized_response_reduced_velocity_min",
                ):
                    raw_value = summary_state.get(name)
                    if raw_value is None:
                        if int(restored_summary["samples"]) != 0:
                            raise ValueError(
                                f"Checkpoint {name} is missing after response samples."
                            )
                        restored_summary[name] = None
                    else:
                        value = float(raw_value)
                        if not math.isfinite(value) or value < 0.0:
                            raise ValueError(
                                f"Checkpoint {name} must be finite and non-negative."
                            )
                        restored_summary[name] = value
                for name in (
                    "response_declared_ambient_crossflow_speed_max_mps",
                    "response_relative_crossflow_speed_max_mps",
                    "response_section_width_max_m",
                    "configured_response_reduced_velocity_max",
                    "realized_response_reduced_velocity_max",
                ):
                    value = float(summary_state.get(name, 0.0) or 0.0)
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(
                            f"Checkpoint {name} must be finite and non-negative."
                        )
                    restored_summary[name] = value
            seed_summary_names = {
                name
                for name in summary_state
                if str(name).startswith("startup_seed_")
                or str(name).startswith("max_startup_seed_")
                or str(name).startswith("max_abs_startup_seed_")
            }
            seeded_model = "startup_seed_samples" in restored_summary
            if seeded_model:
                seed_integer_names = (
                    "startup_seed_samples",
                    "startup_seed_active_steps",
                    "startup_seed_force_limited_steps",
                    "startup_seed_force_limited_node_samples",
                    "max_startup_seed_force_limited_nodes",
                )
                for name in seed_integer_names:
                    value = int(summary_state.get(name, 0) or 0)
                    if value < 0:
                        raise ValueError(
                            f"Checkpoint {name} must be non-negative."
                        )
                    restored_summary[name] = value
                for name in (
                    "max_abs_startup_seed_modal_acceleration_mps2",
                    "max_abs_startup_seed_raw_generalized_force_n",
                    "max_startup_seed_raw_nodal_force_n",
                ):
                    value = float(summary_state.get(name, 0.0) or 0.0)
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(
                            f"Checkpoint {name} must be finite and non-negative."
                        )
                    restored_summary[name] = value
                for name in (
                    "startup_seed_target_amplitude_ratio",
                    "startup_seed_target_amplitude_m",
                ):
                    value = float(summary_state.get(name))
                    if not math.isfinite(value) or value != float(
                        restored_summary[name]
                    ):
                        raise ValueError(f"Checkpoint {name} mismatch.")
            elif seed_summary_names:
                raise ValueError(
                    "Unseeded response model cannot restore startup-seed summary fields."
                )
        cached_state = state.get("cached_step")
        if cached_state is not None:
            if not isinstance(cached_state, Mapping):
                raise ValueError("Checkpoint cached_step must be a mapping or null.")
            cached_t = float(cached_state.get("time_s"))
            cached_azimuth = float(cached_state.get("azimuth_deg"))
            relative_velocity = np.asarray(
                cached_state.get("relative_velocity_mps"), dtype=float
            )
            force = np.asarray(cached_state.get("force_n"), dtype=float)
            expected_vector_shape = (self._node_count, 3)
            if (
                not math.isfinite(cached_t)
                or not math.isfinite(cached_azimuth)
                or relative_velocity.shape != expected_vector_shape
                or force.shape != expected_vector_shape
                or np.any(~np.isfinite(relative_velocity))
                or np.any(~np.isfinite(force))
            ):
                raise ValueError("Checkpoint cached_step contains invalid values.")
            structural_raw = cached_state.get("structural_velocity_mps")
            if self._response_model_enabled:
                structural_velocity = np.asarray(structural_raw, dtype=float)
                if (
                    structural_velocity.shape != expected_vector_shape
                    or np.any(~np.isfinite(structural_velocity))
                ):
                    raise ValueError(
                        "Enabled response checkpoint requires cached structural velocity."
                    )
                restored_signature = (
                    cached_t,
                    cached_azimuth,
                    np.array(relative_velocity, copy=True),
                    np.array(structural_velocity, copy=True),
                )
                ambient_raw = cached_state.get("declared_ambient_velocity_mps")
                if is_response_schema_v2 and ambient_raw is not None:
                    ambient_velocity = np.asarray(ambient_raw, dtype=float)
                    if (
                        ambient_velocity.shape != expected_vector_shape
                        or np.any(~np.isfinite(ambient_velocity))
                    ):
                        raise ValueError(
                            "Checkpoint cached ambient velocity is invalid."
                        )
                    restored_signature += (
                        np.array(ambient_velocity, copy=True),
                    )
                elif is_response_schema_v2:
                    raise ValueError(
                        "Response schema_version 2 checkpoint requires cached ambient velocity."
                    )
                restored_cached_signature = restored_signature
            else:
                if structural_raw is not None:
                    raise ValueError(
                        "Disabled response checkpoint cannot cache structural velocity."
                    )
                restored_cached_signature = (
                    cached_t,
                    cached_azimuth,
                    np.array(relative_velocity, copy=True),
                )
            restored_force = np.array(force, copy=True)

        restored_cache_hits = int(state.get("cache_hits", 0) or 0)
        restored_cache_misses = int(state.get("cache_misses", 0) or 0)
        if restored_cache_hits < 0 or restored_cache_misses < 0:
            raise ValueError("Checkpoint cache counters must be non-negative.")

        # Every fallible conversion and model-specific provenance check has
        # succeeded.  Commit the complete state without exposing a mixed
        # pre-/post-restore controller after rejection.
        if self.response_model is not None:
            self.response_model.restore_checkpoint_state(response_state)
            restored_response_diagnostics = self.response_model.last_diagnostics
        if restored_phase_arrays is not None:
            self._phase_time_s = restored_phase_time
            self._phase_angle_cl_rad = restored_phase_arrays["angle_cl_rad"]
            self._phase_angle_cd_rad = restored_phase_arrays["angle_cd_rad"]
            self._phase_frequency_cl_hz = restored_phase_arrays[
                "frequency_cl_hz"
            ]
            self._phase_frequency_cd_hz = restored_phase_arrays[
                "frequency_cd_hz"
            ]
        self._last_response_diagnostics = restored_response_diagnostics
        self._last_source_state = restored_source_state
        self._response_diagnostics_summary = restored_summary
        self._cached_signature = restored_cached_signature
        if restored_force is not None:
            self._force_buffer[...] = restored_force
        self._cache_hits = restored_cache_hits
        self._cache_misses = restored_cache_misses

    def step(
        self,
        node_data: Union[np.ndarray, Mapping[str, np.ndarray]],
        *,
        time: Optional[Union[float, np.ndarray]] = None,
        azimuth_deg: Optional[Union[float, np.ndarray]] = None,
        velocity: Optional[np.ndarray] = None,
        structural_velocity: Optional[np.ndarray] = None,
        declared_ambient_velocity: Optional[np.ndarray] = None,
        copy: bool = True,
    ) -> np.ndarray:
        """
        Evaluate one timestep and return per-node global force vectors.

        Args:
            node_data: Mapping or array containing per-node velocity vectors.
            time: Scalar time used to reconstruct the harmonic force signal.
            azimuth_deg: Scalar rotor azimuth used to rotate geometry and inflow.
            velocity: Optional explicit relative-flow velocity override [m/s].
            structural_velocity: Optional structural velocity [m/s], required
                when response-aware forcing is enabled.
            declared_ambient_velocity: Explicit ambient velocity [m/s] before
                flexible structural motion is subtracted. Required by response
                schema v2 and used only by the response closure.
            copy: If true, return a stable caller-owned copy. If false, return
                the internal read-only view used by the runtime fast path.

        Returns:
            An array with shape `[nnodes, 3]`. Public calls receive a copy by
            default; callers using `copy=False` must not store the returned view
            across later non-cached calls.
        """
        t = self._coerce_scalar(node_data, time, ("time", "t"), 0.0, "time")
        az = self._coerce_scalar(node_data, azimuth_deg, ("azimuth_deg", "azimuth", "rotation_deg"), 0.0, "azimuth_deg")
        vel = self._coerce_velocity(node_data, velocity)
        structural_vel = self._coerce_structural_velocity(
            node_data,
            structural_velocity,
        )
        declared_ambient_vel = self._coerce_declared_ambient_velocity(
            node_data,
            declared_ambient_velocity,
        )

        if self._cached_signature is not None:
            cached_t, cached_az, cached_vel = self._cached_signature[:3]
            structural_matches = (
                not self._response_model_enabled
                or np.array_equal(self._cached_signature[3], structural_vel)
            )
            cached_ambient = (
                None
                if len(self._cached_signature) < 5
                else self._cached_signature[4]
            )
            ambient_matches = (
                cached_ambient is None
                and declared_ambient_vel is None
            ) or (
                cached_ambient is not None
                and declared_ambient_vel is not None
                and np.array_equal(cached_ambient, declared_ambient_vel)
            )
            if (
                cached_t == t
                and cached_az == az
                and np.array_equal(cached_vel, vel)
                and structural_matches
                and ambient_matches
            ):
                self._cache_hits += 1
                return self._return_forces(copy)
        if self._response_model_enabled and self._phase_time_s is not None:
            if t < self._phase_time_s - 1.0e-12:
                raise ValueError(
                    f"Phase-continuous force time decreased from "
                    f"{self._phase_time_s:.17g} to {t:.17g} s."
                )
            if math.isclose(t, self._phase_time_s, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    "Phase-continuous force inputs changed at the same time level."
                )
        vel_snapshot = np.array(vel, copy=True)
        structural_vel_snapshot = (
            None if structural_vel is None else np.array(structural_vel, copy=True)
        )
        declared_ambient_vel_snapshot = (
            None
            if declared_ambient_vel is None
            else np.array(declared_ambient_vel, copy=True)
        )
        from .interpolation import interpolate_fft_spectrum_batch

        rot = _axis_angle_matrix(self.rotation_axis, az - self.geometry_reference_azimuth_deg)
        chord = self._base_chord @ rot.T
        normal = self._base_normal @ rot.T

        flow = section_flow_state(
            vel,
            chord,
            normal,
            self._chord,
            self._local_length,
            self.fluid_density,
            self.fluid_dynamicviscosity,
        )
        aoa_deg = flow.aoa_deg
        v_eff = flow.effective_velocity_mps
        reynolds = flow.reynolds
        section_dynamic_force_n = flow.section_dynamic_force_n
        st_length = strouhal_reference_length(self._chord, aoa_deg, self._thickness)
        response_effective_velocity = v_eff
        response_section_width = st_length
        response_declared_ambient_crossflow_speed = v_eff
        response_relative_crossflow_speed = v_eff
        response_section_width_basis = "relative_crossflow_angle"
        if isinstance(self.response_model, QBladeResponseModel):
            response_relative_crossflow_speed = v_eff
            if declared_ambient_vel is None:
                response_declared_ambient_crossflow_speed = v_eff
                ambient_aoa_deg = aoa_deg
            else:
                ambient_flow = section_flow_state(
                    declared_ambient_vel,
                    chord,
                    normal,
                    self._chord,
                    self._local_length,
                    self.fluid_density,
                    self.fluid_dynamicviscosity,
                )
                response_declared_ambient_crossflow_speed = (
                    ambient_flow.effective_velocity_mps
                )
                ambient_aoa_deg = ambient_flow.aoa_deg
            if (
                self.response_model.aerodynamic_speed_basis
                == AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW
            ):
                response_effective_velocity = (
                    response_declared_ambient_crossflow_speed
                )
            explicit_response_width = self.response_model.response_section_width_m
            if explicit_response_width is not None:
                response_section_width = explicit_response_width
            elif (
                self.response_model.schema_version
                == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
            ):
                response_section_width = strouhal_reference_length(
                    self._chord,
                    ambient_aoa_deg,
                    self._thickness,
                )
            response_section_width_basis = (
                self.response_model.response_section_width_basis
            )
        freq_min_hz = np.zeros(self._node_count, dtype=float)
        freq_max_hz = np.zeros(self._node_count, dtype=float)
        unfiltered_freq_max_hz = np.zeros(self._node_count, dtype=float)
        frequency_truncated_count = np.zeros(self._node_count, dtype=int)

        # Build the entire step into a candidate buffer.  A response-model
        # limit or interpolation error must leave the last accepted cache,
        # force vector, phase, and diagnostics untouched.
        candidate_force = np.zeros_like(self._force_buffer)
        if self._response_model_enabled:
            phase_angle_cl = np.array(self._phase_angle_cl_rad, copy=True)
            phase_angle_cd = np.array(self._phase_angle_cd_rad, copy=True)
            phase_frequency_cl = np.array(self._phase_frequency_cl_hz, copy=True)
            phase_frequency_cd = np.array(self._phase_frequency_cd_hz, copy=True)
            first_phase_step = self._phase_time_s is None
            phase_dt_s = 0.0 if first_phase_step else t - self._phase_time_s

        for group in self._groups:
            idx = group["indices"]
            afft = group["afft"]
            depth = group["depth"]

            st_cl, amp_cl, pha_cl = interpolate_fft_spectrum_batch(
                afft,
                reynolds[idx],
                aoa_deg[idx],
                "CL",
                n_freq_depth=depth,
                bounds_policy=self.source_bounds_policy,
            )
            st_cd, amp_cd, pha_cd = interpolate_fft_spectrum_batch(
                afft,
                reynolds[idx],
                aoa_deg[idx],
                "CD",
                n_freq_depth=depth,
                bounds_policy=self.source_bounds_policy,
            )

            scale = v_eff[idx] / st_length[idx]
            freq_cl = st_cl * scale[:, None]
            freq_cd = st_cd * scale[:, None]
            unfiltered_freq_max_hz[idx] = np.maximum(np.max(freq_cl, axis=1), np.max(freq_cd, axis=1))
            if self.source_frequency_cap_hz is not None:
                truncate_cl = freq_cl > self.source_frequency_cap_hz
                truncate_cd = freq_cd > self.source_frequency_cap_hz
                # Index zero is the DC coefficient, not an oscillatory tone.
                truncate_cl[:, 0] = False
                truncate_cd[:, 0] = False
                frequency_truncated_count[idx] = np.count_nonzero(
                    truncate_cl & (np.abs(amp_cl) > _EPS), axis=1
                ) + np.count_nonzero(truncate_cd & (np.abs(amp_cd) > _EPS), axis=1)
                freq_cl = np.where(truncate_cl, 0.0, freq_cl)
                freq_cd = np.where(truncate_cd, 0.0, freq_cd)
                amp_cl = np.where(truncate_cl, 0.0, amp_cl)
                amp_cd = np.where(truncate_cd, 0.0, amp_cd)
            freq_min_hz[idx] = np.minimum(np.min(freq_cl, axis=1), np.min(freq_cd, axis=1))
            freq_max_hz[idx] = np.maximum(np.max(freq_cl, axis=1), np.max(freq_cd, axis=1))
            if self._response_model_enabled:
                if first_phase_step:
                    angle_cd = 2.0 * np.pi * freq_cd * t + pha_cd
                    angle_cl = 2.0 * np.pi * freq_cl * t + pha_cl
                else:
                    angle_cd = phase_angle_cd[idx, :depth] + (
                        np.pi
                        * (phase_frequency_cd[idx, :depth] + freq_cd)
                        * phase_dt_s
                    )
                    angle_cl = phase_angle_cl[idx, :depth] + (
                        np.pi
                        * (phase_frequency_cl[idx, :depth] + freq_cl)
                        * phase_dt_s
                    )
                phase_angle_cd[idx, :depth] = angle_cd
                phase_angle_cl[idx, :depth] = angle_cl
                phase_frequency_cd[idx, :depth] = freq_cd
                phase_frequency_cl[idx, :depth] = freq_cl
                chord_force = harmonic_force_channel_from_angle(
                    section_dynamic_force_n[idx],
                    amp_cd,
                    angle_cd,
                    include_mean_load=self.include_mean_load,
                )
                normal_force = harmonic_force_channel_from_angle(
                    section_dynamic_force_n[idx],
                    amp_cl,
                    angle_cl,
                    include_mean_load=self.include_mean_load,
                )
            else:
                chord_force = harmonic_force_channel(
                    section_dynamic_force_n[idx],
                    freq_cd,
                    amp_cd,
                    pha_cd,
                    t,
                    include_mean_load=self.include_mean_load,
                )
                normal_force = harmonic_force_channel(
                    section_dynamic_force_n[idx],
                    freq_cl,
                    amp_cl,
                    pha_cl,
                    t,
                    include_mean_load=self.include_mean_load,
                )

            candidate_force[idx, :] = force_from_chord_normal(
                chord_force,
                normal_force,
                chord[idx],
                normal[idx],
            )

        if self.force_scale != 1.0:
            candidate_force *= self.force_scale

        base_force_norm = np.linalg.norm(candidate_force, axis=1)
        base_force_coefficient = np.divide(
            base_force_norm,
            section_dynamic_force_n,
            out=np.zeros_like(base_force_norm),
            where=section_dynamic_force_n > _EPS,
        )
        response_force_norm = np.zeros(self._node_count, dtype=float)
        response_force_coefficient = np.zeros(self._node_count, dtype=float)

        if self._response_model_enabled:
            response_direction_basis = (
                normal
                if self.response_model.config.force_direction == "normal"
                else chord
            )
            if isinstance(self.response_model, QBladeResponseModel):
                if (
                    self.response_model.schema_version
                    == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
                ):
                    response_result = self.response_model.step(
                        t,
                        node_ids=self.node_ids,
                        direction_basis=response_direction_basis,
                        structural_velocity_mps=structural_vel,
                        effective_velocity_mps=response_effective_velocity,
                        section_width_m=response_section_width,
                        local_length_m=self._local_length,
                        fluid_density_kgpm3=self.fluid_density,
                        declared_ambient_crossflow_speed_mps=(
                            response_declared_ambient_crossflow_speed
                        ),
                        relative_crossflow_speed_mps=(
                            response_relative_crossflow_speed
                        ),
                        response_section_width_basis=(
                            response_section_width_basis
                        ),
                    )
                else:
                    # Keep the pre-v2 call shape and duplicate signature
                    # exactly intact for omitted/schema-1 configurations.
                    response_result = self.response_model.step(
                        t,
                        node_ids=self.node_ids,
                        direction_basis=response_direction_basis,
                        structural_velocity_mps=structural_vel,
                        effective_velocity_mps=v_eff,
                        section_width_m=st_length,
                        local_length_m=self._local_length,
                        fluid_density_kgpm3=self.fluid_density,
                    )
            elif isinstance(self.response_model, QBladeModalPowerModel):
                response_result = self.response_model.step(
                    t,
                    node_ids=self.node_ids,
                    direction_basis=response_direction_basis,
                    structural_velocity_mps=structural_vel,
                )
            else:  # pragma: no cover - constructor restricts model types
                raise RuntimeError("Enabled response model has an unsupported type.")
            candidate_force += response_result.additive_force_n
            response_force_norm = np.linalg.norm(
                response_result.additive_force_n,
                axis=1,
            )
            response_force_coefficient = np.divide(
                response_force_norm,
                section_dynamic_force_n,
                out=np.zeros_like(response_force_norm),
                where=section_dynamic_force_n > _EPS,
            )
            candidate_response_diagnostics = response_result.diagnostics
        else:
            candidate_response_diagnostics = None

        total_force_norm = np.linalg.norm(candidate_force, axis=1)
        total_force_coefficient = np.divide(
            total_force_norm,
            section_dynamic_force_n,
            out=np.zeros_like(total_force_norm),
            where=section_dynamic_force_n > _EPS,
        )
        candidate_source_state = ControllerSourceState(
            node_ids=np.asarray(self.node_ids, dtype=object),
            component_ids=np.asarray(self._node_component_ids, dtype=object),
            airfoil_ids=np.asarray(self._node_airfoil_ids, dtype=object),
            source_airfoil_ids=self._source_airfoil_ids,
            reynolds_models=self._source_reynolds_models,
            aoa_deg=aoa_deg,
            reynolds=reynolds,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            unfiltered_freq_max_hz=unfiltered_freq_max_hz,
            frequency_truncated_count=frequency_truncated_count,
            force_norm_n=base_force_norm,
            force_coefficient_abs=np.abs(base_force_coefficient),
            response_force_norm_n=response_force_norm,
            response_force_coefficient_abs=np.abs(response_force_coefficient),
            total_force_norm_n=total_force_norm,
            total_force_coefficient_abs=np.abs(total_force_coefficient),
            source_aoa_min_deg=self._source_aoa_min,
            source_aoa_max_deg=self._source_aoa_max,
            source_re_min=self._source_re_min,
            source_re_max=self._source_re_max,
            source_force_coefficient_max_abs=self._source_force_coefficient_max_abs,
        )

        # Commit only after the complete candidate step, including the
        # response model's fail-closed checks, has succeeded.
        self._force_buffer[...] = candidate_force
        self._last_source_state = candidate_source_state
        self._last_response_diagnostics = candidate_response_diagnostics
        if self._response_model_enabled:
            self._record_response_diagnostics(candidate_response_diagnostics)
            self._phase_time_s = t
            self._phase_angle_cl_rad = phase_angle_cl
            self._phase_angle_cd_rad = phase_angle_cd
            self._phase_frequency_cl_hz = phase_frequency_cl
            self._phase_frequency_cd_hz = phase_frequency_cd
        self._cache_misses += 1
        if self._response_model_enabled:
            response_signature = (
                t,
                az,
                vel_snapshot,
                structural_vel_snapshot,
            )
            if declared_ambient_vel_snapshot is not None:
                response_signature += (declared_ambient_vel_snapshot,)
            self._cached_signature = response_signature
        else:
            self._cached_signature = (t, az, vel_snapshot)
        return self._return_forces(copy)

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
        forces = self.step(node_data, copy=False)
        swap.write(force_block, forces)
        return forces


VorLapController = VorLapQBladeController
QBladeController = VorLapQBladeController


__all__ = [
    "ControllerSourceState",
    "QBladeController",
    "SwapArrayAdapter",
    "SwapBlockSpec",
    "VorLapController",
    "VorLapQBladeController",
]
