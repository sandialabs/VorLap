"""Time-varying VorLap force-history orchestration."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .force_model import (
    compute_local_segment_length,
    force_from_chord_normal,
    resolve_airfoil,
    section_flow_state,
    strouhal_reference_length,
)
from .geometry_frames import normalize, prepare_component_frames, rotate_point_about_axis, rotate_vector
from .spectral_reconstruction import reconstruct_nonstationary_signal
from .structs import AirfoilFFT, Component, InflowTimeSeries, VIV_Params


def coerce_inflow_profile(inflow_profile: Union[InflowTimeSeries, Dict[str, np.ndarray]]) -> InflowTimeSeries:
    """Normalize accepted inflow profile inputs into an InflowTimeSeries."""
    if isinstance(inflow_profile, InflowTimeSeries):
        return inflow_profile

    if isinstance(inflow_profile, dict):
        if {"time", "inflow_speeds", "inflow_directions"}.issubset(inflow_profile.keys()):
            return InflowTimeSeries(
                time=np.asarray(inflow_profile["time"], dtype=float),
                inflow_speeds=np.asarray(inflow_profile["inflow_speeds"], dtype=float),
                inflow_directions=np.asarray(inflow_profile["inflow_directions"], dtype=float),
            )
        raise ValueError(
            "Dictionary inflow profile must contain keys: "
            "`time`, `inflow_speeds`, `inflow_directions`."
        )

    raise TypeError(
        "inflow_profile must be an InflowTimeSeries or a dictionary with "
        "`time`, `inflow_speeds`, and `inflow_directions`."
    )


def _compute_time_varying_force_history_impl(
    components: List[Component],
    affts: Dict[str, AirfoilFFT],
    viv_params: VIV_Params,
    inflow_profile: Union[InflowTimeSeries, Dict[str, np.ndarray]],
    spectrum_lookup,
    azimuth_deg: Optional[float],
    smoothing_cycles: float,
    n_freq_depth: Optional[int],
    include_mean_load: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared implementation for nonstationary force reconstruction."""
    if not components:
        raise ValueError("At least one component is required.")
    if not affts:
        raise ValueError("At least one airfoil FFT dataset is required.")
    prepare_component_frames(components)

    profile = coerce_inflow_profile(inflow_profile)
    time = profile.time
    inflow_speeds = profile.inflow_speeds
    inflow_directions = profile.inflow_directions

    n_freq_depth_eff = int(viv_params.n_freq_depth if n_freq_depth is None else n_freq_depth)
    if n_freq_depth_eff < 1:
        raise ValueError("n_freq_depth must be >= 1.")

    rotation_axis = normalize(viv_params.rotation_axis, "rotation_axis")
    axis_offset = np.asarray(viv_params.rotation_axis_offset, dtype=float)
    fluid_density = float(viv_params.fluid_density)
    fluid_dynamicviscosity = float(viv_params.fluid_dynamicviscosity)
    azimuth = float(viv_params.output_azimuth_vinf[0] if azimuth_deg is None else azimuth_deg)

    n_time = len(time)
    total_nodes = sum(comp.shape_xyz.shape[0] for comp in components)

    total_global_force_vector = np.zeros((n_time, 3), dtype=float)
    total_global_moment_vector = np.zeros((n_time, 3), dtype=float)
    global_force_vector_nodes = np.zeros((n_time, 3, total_nodes), dtype=float)

    Vinf = inflow_directions * inflow_speeds[:, None]

    inode = 0
    warned_missing_airfoils: set = set()
    for comp in components:
        n_pts = comp.shape_xyz.shape[0]

        for ipt in range(n_pts):
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
            aoa_deg = flow.aoa_deg
            V_eff = flow.effective_velocity_mps
            Re = flow.reynolds
            q = flow.section_dynamic_force_n

            spectra = spectrum_lookup(afft, Re, aoa_deg, n_freq_depth_eff)
            ST_cl, amps_cl, phases_cl = spectra["CL"]
            ST_cd, amps_cd, phases_cd = spectra["CD"]

            st_length = strouhal_reference_length(chord, aoa_deg, thickness)
            cl_freqs = ST_cl * (V_eff / st_length)[:, None]
            cd_freqs = ST_cd * (V_eff / st_length)[:, None]

            normal_force_amps = amps_cl * q[:, None]
            chord_force_amps = amps_cd * q[:, None]
            if not include_mean_load:
                normal_force_amps[:, 0] = 0.0
                chord_force_amps[:, 0] = 0.0

            normal_signal = reconstruct_nonstationary_signal(
                cl_freqs,
                normal_force_amps,
                phases_cl,
                time,
                smoothing_cycles=smoothing_cycles,
            )
            chord_signal = reconstruct_nonstationary_signal(
                cd_freqs,
                chord_force_amps,
                phases_cd,
                time,
                smoothing_cycles=smoothing_cycles,
            )

            global_force_vector_list = force_from_chord_normal(
                chord_signal,
                normal_signal,
                chord_rotated,
                normal_rotated,
            )

            global_force_vector_nodes[:, :, inode] = global_force_vector_list
            total_global_force_vector += global_force_vector_list
            global_pos_rotated = rotate_point_about_axis(global_pos, axis_offset, rotation_axis, azimuth)
            moment_arm = global_pos_rotated - axis_offset
            total_global_moment_vector += np.cross(moment_arm[None, :], global_force_vector_list)
            inode += 1

    return time, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes


def compute_time_varying_force_history_optimized(
    components: List[Component],
    affts: Dict[str, AirfoilFFT],
    viv_params: VIV_Params,
    inflow_profile: Union[InflowTimeSeries, Dict[str, np.ndarray]],
    azimuth_deg: Optional[float] = None,
    smoothing_cycles: float = 0.25,
    n_freq_depth: Optional[int] = None,
    include_mean_load: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute nonstationary node/global forces for time-varying inflow speed and direction.
    """
    from .interpolation import interpolate_fft_spectrum_batch

    def lookup(afft: AirfoilFFT, Re: np.ndarray, aoa_deg: np.ndarray, n_depth: int):
        st_cl, amp_cl, pha_cl = interpolate_fft_spectrum_batch(afft, Re, aoa_deg, "CL", n_freq_depth=n_depth)
        st_cd, amp_cd, pha_cd = interpolate_fft_spectrum_batch(afft, Re, aoa_deg, "CD", n_freq_depth=n_depth)
        return {"CL": (st_cl, amp_cl, pha_cl), "CD": (st_cd, amp_cd, pha_cd)}

    return _compute_time_varying_force_history_impl(
        components=components,
        affts=affts,
        viv_params=viv_params,
        inflow_profile=inflow_profile,
        spectrum_lookup=lookup,
        azimuth_deg=azimuth_deg,
        smoothing_cycles=smoothing_cycles,
        n_freq_depth=n_freq_depth,
        include_mean_load=bool(include_mean_load),
    )


def compute_time_varying_force_history(
    components: List[Component],
    affts: Dict[str, AirfoilFFT],
    viv_params: VIV_Params,
    inflow_profile: Union[InflowTimeSeries, Dict[str, np.ndarray]],
    azimuth_deg: Optional[float] = None,
    smoothing_cycles: float = 0.25,
    n_freq_depth: Optional[int] = None,
    include_mean_load: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Baseline nonstationary force-history computation.
    """
    from .interpolation import interpolate_fft_spectrum

    def lookup(afft: AirfoilFFT, Re: np.ndarray, aoa_deg: np.ndarray, n_depth: int):
        n_time = Re.shape[0]
        k_depth = min(
            int(n_depth),
            afft.CL_ST.shape[2],
            afft.CD_ST.shape[2],
        )
        st_cl = np.zeros((n_time, k_depth), dtype=float)
        amp_cl = np.zeros((n_time, k_depth), dtype=float)
        pha_cl = np.zeros((n_time, k_depth), dtype=float)
        st_cd = np.zeros((n_time, k_depth), dtype=float)
        amp_cd = np.zeros((n_time, k_depth), dtype=float)
        pha_cd = np.zeros((n_time, k_depth), dtype=float)
        for it in range(n_time):
            st_cl[it, :], amp_cl[it, :], pha_cl[it, :] = interpolate_fft_spectrum(
                afft, Re[it], aoa_deg[it], "CL", n_freq_depth=k_depth
            )
            st_cd[it, :], amp_cd[it, :], pha_cd[it, :] = interpolate_fft_spectrum(
                afft, Re[it], aoa_deg[it], "CD", n_freq_depth=k_depth
            )
        return {"CL": (st_cl, amp_cl, pha_cl), "CD": (st_cd, amp_cd, pha_cd)}

    return _compute_time_varying_force_history_impl(
        components=components,
        affts=affts,
        viv_params=viv_params,
        inflow_profile=inflow_profile,
        spectrum_lookup=lookup,
        azimuth_deg=azimuth_deg,
        smoothing_cycles=smoothing_cycles,
        n_freq_depth=n_freq_depth,
        include_mean_load=bool(include_mean_load),
    )


__all__ = [
    "coerce_inflow_profile",
    "compute_time_varying_force_history",
    "compute_time_varying_force_history_optimized",
]
