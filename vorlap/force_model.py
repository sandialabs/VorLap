"""Shared aerodynamic force-model kernels for VorLap computations and QBlade coupling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import warnings

import numpy as np

from .structs import AirfoilFFT


_EPS = 1.0e-12


@dataclass(frozen=True)
class SectionFlowState:
    """Local section flow quantities used by offline and coupled force paths."""

    velocity_chord_mps: np.ndarray
    velocity_normal_mps: np.ndarray
    aoa_deg: np.ndarray
    effective_velocity_mps: np.ndarray
    reynolds: np.ndarray
    section_dynamic_force_n: np.ndarray


def resolve_airfoil(
    affts: Dict[str, AirfoilFFT],
    airfoil_id: str,
    warned_missing_airfoils: Optional[set] = None,
) -> AirfoilFFT:
    """Resolve an airfoil id with fallback to `default`."""
    if airfoil_id in affts:
        return affts[airfoil_id]
    if "default" in affts:
        warn_once = warned_missing_airfoils is not None
        should_warn = True
        if warn_once:
            if airfoil_id in warned_missing_airfoils:
                should_warn = False
            else:
                warned_missing_airfoils.add(airfoil_id)
        if should_warn:
            default_name = getattr(affts["default"], "name", "default")
            warnings.warn(
                f"Airfoil '{airfoil_id}' not found; using default airfoil '{default_name}'.",
                RuntimeWarning,
                stacklevel=2,
            )
        return affts["default"]
    raise KeyError(f"Airfoil '{airfoil_id}' was not found and no 'default' airfoil is available.")


def compute_local_segment_length(shape_xyz: np.ndarray, ipt: int) -> float:
    """
    Compute a nodal span length using neighboring points.

    For interior points this is the average of adjacent segment lengths, giving
    a partition of unity over the span. For end points this is half of the
    adjacent segment. For a single-node component, 1.0 is used to keep the node
    active in force calculations.
    """
    n_pts = shape_xyz.shape[0]
    if n_pts <= 1:
        return 1.0
    if ipt == 0:
        return 0.5 * float(np.linalg.norm(shape_xyz[1, :] - shape_xyz[0, :]))
    if ipt == n_pts - 1:
        return 0.5 * float(np.linalg.norm(shape_xyz[-1, :] - shape_xyz[-2, :]))

    left = float(np.linalg.norm(shape_xyz[ipt, :] - shape_xyz[ipt - 1, :]))
    right = float(np.linalg.norm(shape_xyz[ipt + 1, :] - shape_xyz[ipt, :]))
    return 0.5 * (left + right)


def strouhal_reference_length(
    chord: np.ndarray,
    aoa_deg: np.ndarray,
    thickness_ratio: np.ndarray,
) -> np.ndarray:
    """
    Return the dimensional length used for Strouhal scaling.

    VorLap uses the projected cross-flow chord length, `chord * |sin(AOA)|`.
    At low AOA the projected chord can collapse below the physical airfoil
    thickness, so the relative thickness is used as a lower bound.
    """
    chord_arr, aoa_arr, thickness_arr = np.broadcast_arrays(
        np.asarray(chord, dtype=float),
        np.asarray(aoa_deg, dtype=float),
        np.asarray(thickness_ratio, dtype=float),
    )
    if np.any(~np.isfinite(chord_arr)) or np.any(~np.isfinite(aoa_arr)) or np.any(~np.isfinite(thickness_arr)):
        raise ValueError("chord, AOA, and thickness values must be finite.")
    if np.any(chord_arr <= _EPS):
        raise ValueError("chord values must be positive.")
    if np.any(thickness_arr < 0.0):
        raise ValueError("thickness ratios must be non-negative.")

    projected_ratio = np.abs(np.sin(np.deg2rad(aoa_arr)))
    reference_ratio = np.maximum(projected_ratio, np.maximum(thickness_arr, _EPS))
    return chord_arr * reference_ratio


def section_flow_state(
    velocity_mps: np.ndarray,
    chord_unit: np.ndarray,
    normal_unit: np.ndarray,
    chord_m: np.ndarray,
    local_length_m: np.ndarray,
    fluid_density_kgpm3: float,
    fluid_dynamicviscosity_pa_s: float,
) -> SectionFlowState:
    """Project velocity into the section basis and return unit-labelled force inputs."""
    velocity = np.asarray(velocity_mps, dtype=float)
    chord_vec = np.asarray(chord_unit, dtype=float)
    normal_vec = np.asarray(normal_unit, dtype=float)
    chord = np.asarray(chord_m, dtype=float)
    local_length = np.asarray(local_length_m, dtype=float)
    rho = float(fluid_density_kgpm3)
    mu = float(fluid_dynamicviscosity_pa_s)

    if velocity.shape[-1:] != (3,) or chord_vec.shape[-1:] != (3,) or normal_vec.shape[-1:] != (3,):
        raise ValueError("velocity_mps, chord_unit, and normal_unit must have a trailing length-3 axis.")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(chord_vec)) or not np.all(np.isfinite(normal_vec)):
        raise ValueError("velocity_mps, chord_unit, and normal_unit must be finite.")
    if not np.all(np.isfinite(chord)) or np.any(chord <= _EPS):
        raise ValueError("chord_m values must be finite and positive.")
    if not np.all(np.isfinite(local_length)) or np.any(local_length < 0.0):
        raise ValueError("local_length_m values must be finite and non-negative.")
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("fluid_density_kgpm3 must be finite and positive.")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("fluid_dynamicviscosity_pa_s must be finite and positive.")

    velocity_chord = np.sum(velocity * chord_vec, axis=-1)
    velocity_normal = np.sum(velocity * normal_vec, axis=-1)
    aoa_deg = np.degrees(np.arctan2(velocity_normal, velocity_chord))
    effective_velocity = np.hypot(velocity_normal, velocity_chord)
    reynolds = rho * effective_velocity * chord / mu
    section_dynamic_force = 0.5 * rho * effective_velocity**2 * chord * local_length
    return SectionFlowState(
        velocity_chord_mps=velocity_chord,
        velocity_normal_mps=velocity_normal,
        aoa_deg=aoa_deg,
        effective_velocity_mps=effective_velocity,
        reynolds=reynolds,
        section_dynamic_force_n=section_dynamic_force,
    )


def harmonic_force_channel(
    section_dynamic_force_n: np.ndarray,
    frequency_hz: np.ndarray,
    amplitude_coefficient: np.ndarray,
    phase_rad: np.ndarray,
    time_s: np.ndarray,
    *,
    include_mean_load: bool,
) -> np.ndarray:
    """Reconstruct one chord/normal force channel from harmonic coefficients."""
    dynamic_force = np.asarray(section_dynamic_force_n, dtype=float)
    frequency = np.asarray(frequency_hz, dtype=float)
    amplitude = np.asarray(amplitude_coefficient, dtype=float)
    phase = np.asarray(phase_rad, dtype=float)
    time = np.asarray(time_s, dtype=float)

    try:
        frequency, amplitude, phase = np.broadcast_arrays(frequency, amplitude, phase)
    except ValueError as exc:
        raise ValueError("frequency_hz, amplitude_coefficient, and phase_rad must be broadcast-compatible.") from exc
    if amplitude.ndim < 1 or amplitude.shape[-1] < 1:
        raise ValueError("harmonic coefficient arrays must include at least one frequency depth.")
    if (
        not np.all(np.isfinite(dynamic_force))
        or not np.all(np.isfinite(frequency))
        or not np.all(np.isfinite(amplitude))
        or not np.all(np.isfinite(phase))
        or not np.all(np.isfinite(time))
    ):
        raise ValueError("harmonic force inputs must be finite.")

    base_shape = amplitude.shape[:-1]
    try:
        dynamic_force = np.broadcast_to(dynamic_force, base_shape)
    except ValueError as exc:
        raise ValueError("section_dynamic_force_n must broadcast to the harmonic coefficient base shape.") from exc

    if time.ndim == 0:
        if include_mean_load:
            force = dynamic_force * amplitude[..., 0]
        else:
            force = np.zeros(base_shape, dtype=float)
        if amplitude.shape[-1] > 1:
            force = force + np.sum(
                dynamic_force[..., None]
                * amplitude[..., 1:]
                * np.cos(2.0 * np.pi * frequency[..., 1:] * float(time) + phase[..., 1:]),
                axis=-1,
            )
        return force

    time_shape = time.shape
    dynamic_time = dynamic_force.reshape((1,) * time.ndim + base_shape)
    if include_mean_load:
        force = dynamic_time * amplitude[..., 0]
    else:
        force = np.zeros(time_shape + base_shape, dtype=float)
    if amplitude.shape[-1] > 1:
        time_for_harmonics = time.reshape(time_shape + (1,) * amplitude.ndim)
        force = force + np.sum(
            dynamic_time[..., None]
            * amplitude[..., 1:]
            * np.cos(2.0 * np.pi * frequency[..., 1:] * time_for_harmonics + phase[..., 1:]),
            axis=-1,
        )
    return force


def harmonic_force_channel_from_angle(
    section_dynamic_force_n: np.ndarray,
    amplitude_coefficient: np.ndarray,
    phase_angle_rad: np.ndarray,
    *,
    include_mean_load: bool,
) -> np.ndarray:
    """Reconstruct a harmonic channel from complete accumulated angles.

    ``phase_angle_rad[..., k]`` is the instantaneous angle of tone ``k``.
    Unlike :func:`harmonic_force_channel`, this function does not form
    ``2*pi*f(t)*t`` and is therefore suitable when a stateful model changes
    frequency in time. Index zero remains the DC coefficient and its angle is
    ignored.
    """
    dynamic_force = np.asarray(section_dynamic_force_n, dtype=float)
    amplitude = np.asarray(amplitude_coefficient, dtype=float)
    angle = np.asarray(phase_angle_rad, dtype=float)
    try:
        amplitude, angle = np.broadcast_arrays(amplitude, angle)
    except ValueError as exc:
        raise ValueError("amplitude_coefficient and phase_angle_rad must be broadcast-compatible.") from exc
    if amplitude.ndim < 1 or amplitude.shape[-1] < 1:
        raise ValueError("harmonic coefficient arrays must include at least one frequency depth.")
    if not np.all(np.isfinite(dynamic_force)) or not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(angle)):
        raise ValueError("harmonic force inputs must be finite.")

    base_shape = amplitude.shape[:-1]
    try:
        dynamic_force = np.broadcast_to(dynamic_force, base_shape)
    except ValueError as exc:
        raise ValueError("section_dynamic_force_n must broadcast to the harmonic coefficient base shape.") from exc

    force = dynamic_force * amplitude[..., 0] if include_mean_load else np.zeros(base_shape, dtype=float)
    if amplitude.shape[-1] > 1:
        force = force + np.sum(
            dynamic_force[..., None] * amplitude[..., 1:] * np.cos(angle[..., 1:]),
            axis=-1,
        )
    return force


def force_from_chord_normal(
    chord_force: np.ndarray,
    normal_force: np.ndarray,
    chord_unit: np.ndarray,
    normal_unit: np.ndarray,
) -> np.ndarray:
    """Assemble global force vectors from chord/normal forces and section basis vectors."""
    return (
        np.asarray(chord_force, dtype=float)[..., None] * np.asarray(chord_unit, dtype=float)
        + np.asarray(normal_force, dtype=float)[..., None] * np.asarray(normal_unit, dtype=float)
    )


__all__ = [
    "compute_local_segment_length",
    "force_from_chord_normal",
    "harmonic_force_channel",
    "harmonic_force_channel_from_angle",
    "resolve_airfoil",
    "SectionFlowState",
    "section_flow_state",
    "strouhal_reference_length",
]
