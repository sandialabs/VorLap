"""Stationary VorLap sweep orchestration for force, moment, and overlap outputs."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .force_model import (
    compute_local_segment_length,
    force_from_chord_normal,
    resolve_airfoil,
    section_flow_state,
    strouhal_reference_length,
)
from .geometry_frames import normalize, prepare_component_frames, rotate_point_about_axis, rotate_vector
from .spectral_reconstruction import reconstruct_signal
from .structs import AirfoilFFT, Component, VIV_Params


def _finite_1d(values: np.ndarray, name: str, *, min_size: int = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} value(s).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} values must be finite.")
    return arr


def _time_vector(values: np.ndarray, name: str, *, min_size: int = 1) -> np.ndarray:
    arr = _finite_1d(values, name, min_size=min_size)
    if arr.size > 1 and np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} values must be strictly increasing.")
    return arr


def _compute_thrust_torque_spectrum_impl(
    components: List[Component],
    affts: Dict[str, AirfoilFFT],
    viv_params: VIV_Params,
    natfreqs: np.ndarray,
    spectrum_lookup,
    include_mean_load: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared implementation for standard and optimized spectrum solve paths."""
    if not components:
        raise ValueError("At least one component is required.")
    if not affts:
        raise ValueError("At least one airfoil FFT dataset is required.")
    prepare_component_frames(components)

    inflow_speeds = _finite_1d(viv_params.inflow_speeds, "inflow_speeds")
    if np.any(inflow_speeds < 0.0):
        raise ValueError("inflow_speeds values must be non-negative.")
    azimuths = _finite_1d(viv_params.azimuths, "azimuths")
    output_time = _time_vector(viv_params.output_time, "output_time")
    inflow_unit = normalize(viv_params.inflow_vec, "inflow_vec")
    rotation_axis = normalize(viv_params.rotation_axis, "rotation_axis")
    axis_offset = np.asarray(viv_params.rotation_axis_offset, dtype=float)
    if axis_offset.shape != (3,) or not np.all(np.isfinite(axis_offset)):
        raise ValueError("rotation_axis_offset must contain three finite values.")
    fluid_density = float(viv_params.fluid_density)
    fluid_dynamicviscosity = float(viv_params.fluid_dynamicviscosity)
    n_harmonic = int(viv_params.n_harmonic)
    amplitude_coeff_cutoff = float(viv_params.amplitude_coeff_cutoff)
    n_freq_depth = int(viv_params.n_freq_depth)
    freq_min = float(viv_params.freq_min)
    freq_max = float(viv_params.freq_max)
    if not math.isfinite(freq_min) or freq_min < 0.0:
        raise ValueError("freq_min must be finite and non-negative.")
    if not (math.isfinite(freq_max) or math.isinf(freq_max)):
        raise ValueError("freq_max must be finite or infinity.")
    if freq_max < freq_min:
        raise ValueError("freq_max must be greater than or equal to freq_min.")

    natfreqs = np.asarray(natfreqs, dtype=float).reshape(-1)
    natfreqs = natfreqs[np.isfinite(natfreqs) & (natfreqs > 0.0)]

    n_inflow = len(inflow_speeds)
    n_az = len(azimuths)

    total_global_force_vector = np.zeros((n_inflow, n_az, 3))
    total_global_moment_vector = np.zeros((n_inflow, n_az, 3))
    percdiff_matrix = np.ones((n_inflow, n_az)) * 1000.0
    percdiff_info = np.full((n_inflow, n_az), None, dtype=object)

    total_nodes = sum(comp.shape_xyz.shape[0] for comp in components)
    global_force_vector_nodes = np.zeros((output_time.size, 3, total_nodes))

    output_azimuth = float(viv_params.output_azimuth_vinf[0])
    output_inflow = float(viv_params.output_azimuth_vinf[1])
    warned_missing_airfoils: set = set()

    for i_inflow, inflow_speed in enumerate(inflow_speeds):
        Vinf = inflow_unit * inflow_speed

        for j_azi, azimuth in enumerate(azimuths):
            inode = 0
            for comp in components:
                n_pts = comp.shape_xyz.shape[0]

                for ipt in range(n_pts):
                    inode += 1

                    global_pos = np.asarray(comp.shape_xyz_global[ipt], dtype=float)
                    chord = float(comp.chord[ipt])
                    thickness = float(comp.thickness[ipt])

                    afft = resolve_airfoil(
                        affts,
                        comp.airfoil_ids[ipt],
                        warned_missing_airfoils=warned_missing_airfoils,
                    )

                    chord_vector = np.asarray(comp.chord_vector[ipt, :], dtype=float)
                    normal_vector = np.asarray(comp.normal_vector[ipt, :], dtype=float)
                    chord_unit = normalize(chord_vector, f"chord_vector[{comp.id}:{ipt}]")
                    normal_unit = normalize(normal_vector, f"normal_vector[{comp.id}:{ipt}]")
                    chord_rotated = rotate_vector(chord_unit, rotation_axis, azimuth)
                    normal_rotated = rotate_vector(normal_unit, rotation_axis, azimuth)

                    local_length = compute_local_segment_length(comp.shape_xyz, ipt)
                    flow = section_flow_state(
                        Vinf,
                        chord_rotated,
                        normal_rotated,
                        chord,
                        local_length,
                        fluid_density,
                        fluid_dynamicviscosity,
                    )
                    aoa_deg = float(flow.aoa_deg)
                    V_eff = float(flow.effective_velocity_mps)
                    Re = float(flow.reynolds)
                    q = float(flow.section_dynamic_force_n)

                    spectra = spectrum_lookup(afft, Re, aoa_deg, n_freq_depth)
                    ST_cl, amps_cl, phases_cl = spectra["CL"]
                    ST_cd, amps_cd, phases_cd = spectra["CD"]
                    ST_cf, amps_cf, _phases_cf = spectra["CF"]

                    if include_mean_load:
                        normal_force = amps_cl[0] * q
                        chord_force = amps_cd[0] * q
                    else:
                        normal_force = 0.0
                        chord_force = 0.0

                    global_force_vector = force_from_chord_normal(
                        chord_force,
                        normal_force,
                        chord_rotated,
                        normal_rotated,
                    )

                    total_global_force_vector[i_inflow, j_azi, :] += global_force_vector

                    # Physical moment is r x F about the rotation-axis offset.
                    global_pos_rotated = rotate_point_about_axis(global_pos, axis_offset, rotation_axis, azimuth)
                    moment_arm = global_pos_rotated - axis_offset
                    total_global_moment_vector[i_inflow, j_azi, :] += np.cross(moment_arm, global_force_vector)

                    st_length = float(strouhal_reference_length(chord, aoa_deg, thickness))
                    frequencies_cf = ST_cf * (V_eff / st_length)

                    # Record worst-case overlap while skipping the DC component.
                    for lstrouhal in range(1, min(n_freq_depth, len(frequencies_cf))):
                        if amps_cf[lstrouhal] <= amplitude_coeff_cutoff:
                            continue
                        shedding_freq = float(frequencies_cf[lstrouhal])
                        if not np.isfinite(shedding_freq):
                            continue
                        if shedding_freq < freq_min or shedding_freq > freq_max:
                            continue

                        for jnatfreq, natfreq in enumerate(natfreqs):
                            for kharmonic in range(1, n_harmonic + 1):
                                target_freq = natfreq * kharmonic
                                percdiff = (shedding_freq - target_freq) / target_freq * 100.0
                                abs_percdiff = abs(percdiff)
                                if percdiff_matrix[i_inflow, j_azi] > abs_percdiff:
                                    percdiff_matrix[i_inflow, j_azi] = abs_percdiff
                                    percdiff_info[i_inflow, j_azi] = (
                                        f"{percdiff} percdiff Occurs for NatFreq: {natfreqs[jnatfreq]} at Harmonic: "
                                        f"{kharmonic} with Shedding frequency: {shedding_freq} "
                                        f"(Strouhaul {ST_cf[lstrouhal]} depth {lstrouhal}) "
                                        f"AmplitudeCoeff: {amps_cf[lstrouhal]} in Comp: {comp.id} at pt#: {ipt+1} "
                                        f"aoa(deg): {aoa_deg}, Re: {Re}"
                                    )

                    if (
                        np.isclose(output_azimuth, azimuth)
                        and np.isclose(output_inflow, inflow_speed)
                    ):
                        normal_signal = reconstruct_signal(
                            ST_cl * (V_eff / st_length), amps_cl, phases_cl, output_time
                        )
                        chord_signal = reconstruct_signal(
                            ST_cd * (V_eff / st_length), amps_cd, phases_cd, output_time
                        )
                        if not include_mean_load:
                            normal_signal = normal_signal - amps_cl[0]
                            chord_signal = chord_signal - amps_cd[0]

                        global_force_vector_list = force_from_chord_normal(
                            chord_signal * q,
                            normal_signal * q,
                            chord_rotated,
                            normal_rotated,
                        )
                        global_force_vector_nodes[:, :, inode - 1] = global_force_vector_list

    return (
        percdiff_matrix,
        percdiff_info,
        total_global_force_vector,
        total_global_moment_vector,
        global_force_vector_nodes,
    )


def compute_thrust_torque_spectrum_optimized(
    components: List[Component],
    affts: Dict[str, AirfoilFFT],
    viv_params: VIV_Params,
    natfreqs: np.ndarray,
    include_mean_load: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute force, moment, overlap metrics, and node-level reconstructed loads.

    This variant uses the optimized vectorized interpolation backend.
    """
    from .interpolation import interpolate_fft_spectrum_optimized

    def lookup(afft: AirfoilFFT, Re: float, aoa_deg: float, n_freq_depth: int):
        return interpolate_fft_spectrum_optimized(afft, Re, aoa_deg, ["CL", "CD", "CF"], n_freq_depth=n_freq_depth)

    return _compute_thrust_torque_spectrum_impl(
        components,
        affts,
        viv_params,
        natfreqs,
        lookup,
        bool(include_mean_load),
    )


def compute_thrust_torque_spectrum(
    components: List[Component],
    affts: Dict[str, AirfoilFFT],
    viv_params: VIV_Params,
    natfreqs: np.ndarray,
    include_mean_load: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute force, moment, overlap metrics, and node-level reconstructed loads.

    This variant uses the baseline per-field interpolation backend.
    """
    from .interpolation import interpolate_fft_spectrum

    def lookup(afft: AirfoilFFT, Re: float, aoa_deg: float, n_freq_depth: int):
        st_cl, amp_cl, pha_cl = interpolate_fft_spectrum(afft, Re, aoa_deg, "CL", n_freq_depth=n_freq_depth)
        st_cd, amp_cd, pha_cd = interpolate_fft_spectrum(afft, Re, aoa_deg, "CD", n_freq_depth=n_freq_depth)
        st_cf, amp_cf, pha_cf = interpolate_fft_spectrum(afft, Re, aoa_deg, "CF", n_freq_depth=n_freq_depth)
        return {"CL": (st_cl, amp_cl, pha_cl), "CD": (st_cd, amp_cd, pha_cd), "CF": (st_cf, amp_cf, pha_cf)}

    return _compute_thrust_torque_spectrum_impl(
        components,
        affts,
        viv_params,
        natfreqs,
        lookup,
        bool(include_mean_load),
    )


__all__ = [
    "compute_thrust_torque_spectrum",
    "compute_thrust_torque_spectrum_optimized",
]
