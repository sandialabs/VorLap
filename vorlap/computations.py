"""
Utility functions for the VorLap package.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings

from .structs import AirfoilFFT, Component, VIV_Params


_EPS = 1.0e-12


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    """Return a unit vector and raise on zero magnitude input."""
    vec = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= _EPS:
        raise ValueError(f"{name} must have non-zero magnitude.")
    return vec / norm


def _resolve_airfoil(affts: Dict[str, AirfoilFFT], airfoil_id: str) -> AirfoilFFT:
    """Resolve an airfoil id with fallback to `default`."""
    if airfoil_id in affts:
        return affts[airfoil_id]
    if "default" in affts:
        return affts["default"]
    raise KeyError(f"Airfoil '{airfoil_id}' was not found and no 'default' airfoil is available.")


def _compute_local_segment_length(shape_xyz: np.ndarray, ipt: int) -> float:
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


def _compute_thrust_torque_spectrum_impl(
    components: List[Component],
    affts: Dict[str, AirfoilFFT],
    viv_params: VIV_Params,
    natfreqs: np.ndarray,
    spectrum_lookup,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared implementation for standard and optimized spectrum solve paths."""
    if not components:
        raise ValueError("At least one component is required.")
    if not affts:
        raise ValueError("At least one airfoil FFT dataset is required.")

    inflow_speeds = np.asarray(viv_params.inflow_speeds, dtype=float)
    azimuths = np.asarray(viv_params.azimuths, dtype=float)
    inflow_unit = _normalize(viv_params.inflow_vec, "inflow_vec")
    rotation_axis = _normalize(viv_params.rotation_axis, "rotation_axis")
    axis_offset = np.asarray(viv_params.rotation_axis_offset, dtype=float)
    fluid_density = float(viv_params.fluid_density)
    fluid_dynamicviscosity = float(viv_params.fluid_dynamicviscosity)
    n_harmonic = int(viv_params.n_harmonic)
    amplitude_coeff_cutoff = float(viv_params.amplitude_coeff_cutoff)
    n_freq_depth = int(viv_params.n_freq_depth)

    natfreqs = np.asarray(natfreqs, dtype=float).reshape(-1)
    natfreqs = natfreqs[np.isfinite(natfreqs) & (natfreqs > 0.0)]

    n_inflow = len(inflow_speeds)
    n_az = len(azimuths)

    total_global_force_vector = np.zeros((n_inflow, n_az, 3))
    total_global_moment_vector = np.zeros((n_inflow, n_az, 3))
    percdiff_matrix = np.ones((n_inflow, n_az)) * 1000.0
    percdiff_info = np.full((n_inflow, n_az), None, dtype=object)

    total_nodes = sum(comp.shape_xyz.shape[0] for comp in components)
    global_force_vector_nodes = np.zeros((len(viv_params.output_time), 3, total_nodes))

    output_azimuth = float(viv_params.output_azimuth_vinf[0])
    output_inflow = float(viv_params.output_azimuth_vinf[1])

    for i_inflow, inflow_speed in enumerate(inflow_speeds):
        Vinf = inflow_unit * inflow_speed

        for j_azi, azimuth in enumerate(azimuths):
            # Rotating inflow by -azimuth is equivalent to rotating the structure by +azimuth.
            Vin_rotated = rotate_vector(Vinf, rotation_axis, -azimuth)

            inode = 0
            for comp in components:
                n_pts = comp.shape_xyz.shape[0]

                for ipt in range(n_pts):
                    inode += 1

                    global_pos = np.asarray(comp.shape_xyz_global[ipt], dtype=float)
                    chord = float(comp.chord[ipt])

                    afft = _resolve_airfoil(affts, comp.airfoil_ids[ipt])

                    chord_vector = np.asarray(comp.chord_vector[ipt, :], dtype=float)
                    normal_vector = np.asarray(comp.normal_vector[ipt, :], dtype=float)
                    chord_unit = _normalize(chord_vector, f"chord_vector[{comp.id}:{ipt}]")
                    normal_unit = _normalize(normal_vector, f"normal_vector[{comp.id}:{ipt}]")

                    V_chord = float(np.dot(Vin_rotated, chord_unit))
                    V_normal = float(np.dot(Vin_rotated, normal_unit))

                    aoa_rad = math.atan2(V_normal, V_chord)
                    aoa_deg = math.degrees(aoa_rad)
                    V_eff = math.sqrt(V_normal**2 + V_chord**2)
                    Re = fluid_density * V_eff * chord / fluid_dynamicviscosity
                    local_length = _compute_local_segment_length(comp.shape_xyz, ipt)
                    q = 0.5 * fluid_density * V_eff**2 * chord * local_length

                    spectra = spectrum_lookup(afft, Re, aoa_deg, n_freq_depth)
                    ST_cl, amps_cl, phases_cl = spectra["CL"]
                    ST_cd, amps_cd, phases_cd = spectra["CD"]
                    ST_cf, amps_cf, _phases_cf = spectra["CF"]

                    lifts = amps_cl[0] * q
                    drags = amps_cd[0] * q

                    # We use structure-fixed yaw and azimuth-adjusted roll to retain
                    # the same frame convention used by existing VorLap verification scripts.
                    local_yaw = math.degrees(math.atan2(chord_vector[1], chord_vector[0]))
                    normal_vector_rotated = rotate_vector(normal_vector, rotation_axis, azimuth)
                    local_roll = math.degrees(math.atan2(normal_vector_rotated[2], normal_vector_rotated[1]))

                    local_force_vector = np.array([drags, lifts, 0.0], dtype=float)
                    force_vector_rolled = rotate_vector(local_force_vector, np.array([1.0, 0.0, 0.0]), local_roll)
                    global_force_vector = rotate_vector(force_vector_rolled, np.array([0.0, 0.0, 1.0]), local_yaw)

                    total_global_force_vector[i_inflow, j_azi, :] += global_force_vector

                    # Physical moment is r × F about the rotation-axis offset.
                    moment_arm = global_pos - axis_offset
                    total_global_moment_vector[i_inflow, j_azi, :] += np.cross(moment_arm, global_force_vector)

                    st_length = max(chord * abs(math.sin(math.radians(aoa_deg))), _EPS)
                    frequencies_cf = ST_cf * (V_eff / st_length)

                    # Record worst-case overlap while skipping the DC component.
                    for lstrouhal in range(1, min(n_freq_depth, len(frequencies_cf))):
                        if amps_cf[lstrouhal] <= amplitude_coeff_cutoff:
                            continue
                        shedding_freq = float(frequencies_cf[lstrouhal])
                        if not np.isfinite(shedding_freq):
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
                        cl_signal = reconstruct_signal(
                            ST_cl * (V_eff / st_length), amps_cl, phases_cl, viv_params.output_time
                        )
                        cd_signal = reconstruct_signal(
                            ST_cd * (V_eff / st_length), amps_cd, phases_cd, viv_params.output_time
                        )

                        local_force_vectors = np.column_stack([cd_signal * q, cl_signal * q, np.zeros_like(cl_signal)])
                        force_vector_rolled_list = np.array(
                            [rotate_vector(local_force, np.array([1.0, 0.0, 0.0]), local_roll) for local_force in local_force_vectors]
                        )
                        global_force_vector_list = np.array(
                            [rotate_vector(force_rolled, np.array([0.0, 0.0, 1.0]), local_yaw) for force_rolled in force_vector_rolled_list]
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute force, moment, overlap metrics, and node-level reconstructed loads.

    This variant uses the optimized vectorized interpolation backend.
    """
    from .interpolation import interpolate_fft_spectrum_optimized

    def lookup(afft: AirfoilFFT, Re: float, aoa_deg: float, n_freq_depth: int):
        return interpolate_fft_spectrum_optimized(afft, Re, aoa_deg, ["CL", "CD", "CF"], n_freq_depth=n_freq_depth)

    return _compute_thrust_torque_spectrum_impl(components, affts, viv_params, natfreqs, lookup)


def compute_thrust_torque_spectrum(
    components: List[Component],
    affts: Dict[str, AirfoilFFT],
    viv_params: VIV_Params,
    natfreqs: np.ndarray,
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

    return _compute_thrust_torque_spectrum_impl(components, affts, viv_params, natfreqs, lookup)


# """
#     reconstruct_signal(freqs::Vector{Float64}, amps::Vector{Float64}, phases::Vector{Float64}, t::Vector{Float64})

# Reconstructs a time-domain signal from frequency, amplitude (peak), and phase (radians).
# - Assumes the DC term is included as `freqs[i]==0` with `amps[i]` equal to the mean value.
# - For f>0, `amps[i]` is the peak amplitude (not RMS).

# # Arguments
# - `freqs`: Frequency vector (Hz)
# - `amps`: Amplitudes corresponding to each frequency
# - `phases`: Phase offsets (radians) corresponding to each frequency
# - `t`: Time vector (s)

# # Returns
# - `signal`: Time-domain signal as a vector of Float64
# """
# function reconstruct_signal(freqs::Vector{Float64}, amps::Vector{Float64}, phases::Vector{Float64}, tvec::Vector{Float64})

#     @assert length(freqs) == length(amps) == length(phases) "freqs/amps/phases must be same length"
#     @assert length(tvec) ≥ 2 "tvec must have at least 2 samples"

#     dt = tvec[2] - tvec[1]
#     fs = 1/dt
#     fnyq = fs/2
#     if maximum(freqs) > fnyq + eps(fnyq)
#         @warn "Max frequency $(maximum(freqs)) exceeds Nyquist $(fnyq) Hz implied by tvec; reconstruction will alias."
#     end

#     signal = zeros(Float64, length(tvec))

#     # DC (mean)
#     for (A, f) in zip(amps, freqs)
#         if iszero(f)
#             signal .+= A
#         end
#     end

#     # Oscillatory terms (cosine with FFT phases; amps = peak)
#     ω = 2π .* freqs
#     for i in eachindex(freqs)
#         f = freqs[i]
#         if f > 0.0
#             A = amps[i]
#             φ = phases[i]
#             signal += A .* cos.(ω[i] .* tvec .+ φ)
#         end
#     end
#     return signal
# end

def reconstruct_signal(freqs: np.ndarray,
                       amps: np.ndarray,
                       phases: np.ndarray,
                       tvec: np.ndarray) -> np.ndarray:
    """
    Reconstruct a time-domain signal from frequency, peak amplitude, and phase (radians).

    Assumptions / conventions (matching the Julia version):
    - DC term(s): entries where freqs == 0 carry the mean value in `amps`; all such entries are summed.
    - For f > 0, `amps` are **peak** amplitudes (not RMS) and phases follow a cosine convention: cos(ωt + φ).
    - Negative frequencies, if present, are ignored (assumed redundant w.r.t. positive freqs + phases).

    Args:
        freqs  : array of frequencies [Hz]
        amps   : array of peak amplitudes corresponding to each frequency
        phases : array of phase offsets [rad] corresponding to each frequency
        tvec   : time vector [s] (must have at least 2 samples)

    Returns:
        signal : reconstructed time-domain signal (float64), shape = (len(tvec),)
    """
    freqs  = np.asarray(freqs, dtype=np.float64)
    amps   = np.asarray(amps, dtype=np.float64)
    phases = np.asarray(phases, dtype=np.float64)
    tvec   = np.asarray(tvec, dtype=np.float64)

    if not (len(freqs) == len(amps) == len(phases)):
        raise ValueError("freqs/amps/phases must be the same length")
    if tvec.size < 2:
        raise ValueError("tvec must have at least 2 samples")

    dt = tvec[1] - tvec[0]
    if dt <= 0.0:
        raise ValueError("tvec must be strictly increasing with positive spacing")
    fs = 1.0 / dt
    fnyq = 0.5 * fs

    # Nyquist check with epsilon tolerance
    eps = np.finfo(np.float64).eps
    fmax = float(np.max(freqs)) if freqs.size else 0.0
    if fmax > (fnyq + eps * max(1.0, fnyq)):
        warnings.warn(
            (
                f"Max frequency {fmax:.6g} Hz exceeds Nyquist {fnyq:.6g} Hz implied by tvec; "
                "reconstruction may alias."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    signal = np.zeros(tvec.shape[0], dtype=np.float64)

    # DC term(s): sum all freqs == 0
    # Use a tolerance for zero comparison to be robust to tiny numerical noise.
    zero_mask = np.isclose(freqs, 0.0, rtol=0.0, atol=eps)
    if np.any(zero_mask):
        signal += np.sum(amps[zero_mask])

    # Oscillatory terms: f > 0 using cosine convention
    pos_mask = freqs > 0.0
    if np.any(pos_mask):
        fpos = freqs[pos_mask]
        Apos = amps[pos_mask]
        Ppos = phases[pos_mask]
        # ω = 2π f
        omega = 2.0 * np.pi * fpos
        # Efficient broadcasting: (n_pos, 1) * (1, n_t) + (n_pos, 1)
        # then sum over rows to get (n_t,)
        # But to keep memory modest, loop over components (usually sparse spectral lines).
        for w, A, phi in zip(omega, Apos, Ppos):
            signal += A * np.cos(w * tvec + phi)

    return signal


#@profile
def rotate_vector(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates a 3D vector `vec` around a given `axis` by `angle_deg` degrees using Rodrigues' rotation formula.

    Args:
        vec: Vector to rotate (length-3, any direction).
        axis: Rotation axis (length-3, not required to be normalized).
        angle_deg: Rotation angle in degrees (positive is right-hand rule about the axis).

    Returns:
        Rotated 3D vector (length-3).
    """
    θ = math.radians(angle_deg)
    axis = np.asarray(axis, dtype=float)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= _EPS:
        raise ValueError("Rotation axis must have non-zero magnitude.")
    k = axis / axis_norm
    v = np.asarray(vec, dtype=float)
    return v * math.cos(θ) + np.cross(k, v) * math.sin(θ) + k * np.dot(k, v) * (1 - math.cos(θ))


def rotationMatrix(euler: np.ndarray) -> np.ndarray:
    """
    Computes a 3×3 rotation matrix from Euler angles using the ZYX convention (yaw–pitch–roll), where:
    - Z: yaw (heading)
    - Y: pitch (elevation)
    - X: roll (bank)

    The angles are provided in **degrees** and applied in **Z–Y–X** order (extrinsic frame), meaning:
    1. Rotate about global Z axis (yaw)
    2. Then about the global Y axis (pitch)
    3. Then about the global X axis (roll)

    Args:
        euler: Euler angles `[roll, pitch, yaw]` in **degrees**.

    Returns:
        R_global: A 3×3 rotation matrix for transforming a local vector into the global frame.

    Example:
        ```python
        euler = np.array([30.0, 15.0, 60.0])  # roll, pitch, yaw in degrees
        R = rotationMatrix(euler)
        v_local = np.array([1.0, 0.0, 0.0])
        v_global = R @ v_local
        ```
    """
    cz, sz = math.cos(math.radians(euler[2])), math.sin(math.radians(euler[2]))
    cy, sy = math.cos(math.radians(euler[1])), math.sin(math.radians(euler[1]))
    cx, sx = math.cos(math.radians(euler[0])), math.sin(math.radians(euler[0]))
    
    Rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    Ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy]
    ])
    
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx]
    ])
    
    R_global = Rz @ Ry @ Rx
    return R_global


# Import from fileio to avoid circular imports
# Import at runtime to avoid circular imports
def interpolate_fft_spectrum(afft, Re_val, AOA_val, field, n_freq_depth=None):
    """
    This is a wrapper function to avoid circular imports.
    The actual implementation is in fileio.py.
    """
    from .interpolation import interpolate_fft_spectrum as _interpolate_fft_spectrum
    return _interpolate_fft_spectrum(afft, Re_val, AOA_val, field, n_freq_depth)
