"""
Utility functions for the VorLap package.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union, Any

from .structs import AirfoilFFT, Component, VIV_Params


#@profile
def compute_thrust_torque_spectrum_optimized(components: List[Component], 
                                           affts: Dict[str, AirfoilFFT],
                                           viv_params: VIV_Params,
                                           natfreqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized version of compute_thrust_torque_spectrum using cached interpolators and vectorized operations.
    
    Same interface and outputs as the original function, but with significant performance improvements.
    """
    from .interpolation import interpolate_fft_spectrum_optimized
    
    # Pre-cache interpolators for all airfoils
    for afft in affts.values():
        afft._cache_interpolators()
    
    inflow_speeds = viv_params.inflow_speeds
    azimuths = viv_params.azimuths
    rotation_axis = viv_params.rotation_axis
    fluid_density = viv_params.fluid_density
    fluid_dynamicviscosity = viv_params.fluid_dynamicviscosity
    n_harmonic = viv_params.n_harmonic
    amplitude_coeff_cutoff = viv_params.amplitude_coeff_cutoff
    n_freq_depth = viv_params.n_freq_depth

    n_inflow = len(inflow_speeds)
    n_az = len(azimuths)

    total_global_force_vector = np.zeros((n_inflow, n_az, 3))
    total_global_moment_vector = np.zeros((n_inflow, n_az, 3))
    percdiff_matrix = np.ones((n_inflow, n_az)) * 1000
    percdiff_info = np.empty((n_inflow, n_az), dtype=object)

    total_nodes = sum([comp.shape_xyz.shape[0] for comp in components])
    global_force_vector_nodes = np.zeros((len(viv_params.output_time), 3, total_nodes))

    for i_inflow in range(n_inflow):
        Vinf = np.array(viv_params.inflow_vec) / np.linalg.norm(viv_params.inflow_vec) * inflow_speeds[i_inflow]
        
        for j_azi in range(n_az):
            # Negative since rotating the inflow in the negative direction is the same as rotating the structure in the positive
            Vin_rotated = rotate_vector(Vinf, rotation_axis, -azimuths[j_azi])
            
            inode = 0
            for comp in components:
                N_pts = comp.shape_xyz.shape[0]
                
                for ipt in range(N_pts):
                    inode += 1
                    
                    global_pos = comp.shape_xyz_global[ipt]
                    chord = comp.chord[ipt]
                    
                    afid = comp.airfoil_ids[ipt]
                    afft = affts.get(afid, affts["default"])  # get the id, or use the default
                    
                    chord_vector = comp.chord_vector[ipt, :]
                    normal_vector = comp.normal_vector[ipt, :]
                    
                    V_chord = np.dot(Vin_rotated, chord_vector / np.linalg.norm(chord_vector))
                    V_normal = np.dot(Vin_rotated, normal_vector / np.linalg.norm(normal_vector))
                    
                    aoa_rad = math.atan2(V_normal, V_chord)  # Using atan2 for correct quadrant
                    aoa_deg = math.degrees(aoa_rad)
                    V_eff = math.sqrt(V_normal**2 + V_chord**2)  # Fundamental assumption that spanwise flow doesn't impact lift and drag
                    Re = fluid_density * V_eff * chord / fluid_dynamicviscosity

                    if ipt == 0:
                        local_length = 0.0 #TODO: is there a better way to do this without going full on nodes and elements? Maybe just tell people this is how it is calculated
                        # print("AOA: ",aoa_deg, "Azi: ",azimuths[j_azi], "Vinf: ",Vinf, "Veff: ",V_eff)
                    else:
                        local_length = np.linalg.norm(comp.shape_xyz[ipt, :] - comp.shape_xyz[ipt-1, :])

                    q = 0.5 * fluid_density * V_eff**2 * chord * local_length
                    
                    # Optimized interpolation: get all three fields at once
                    results = interpolate_fft_spectrum_optimized(afft, Re, aoa_deg, ['CL', 'CD', 'CF'], n_freq_depth=n_freq_depth)
                    ST_cl, amps_cl, phases_cl = results['CL']
                    ST_cd, amps_cd, phases_cd = results['CD']
                    ST_cf, amps_cf, phases_cf = results['CF']
                    
                    Lifts = amps_cl[0] * q
                    Drags = amps_cd[0] * q

                    # print("CL: ",amps_cl[0], "q: ", q, "Lifts: ", Lifts)
                    # print("CD: ",amps_cd[0], "q: ", q, "Drags: ", Drags)

                    # Calculate the global force vector
                    chord_vector_rotated = rotate_vector(chord_vector, rotation_axis, azimuths[j_azi]*0) # no need to double rotate, keep at 0.
                    local_yaw = math.degrees(math.atan2(chord_vector_rotated[1], chord_vector_rotated[0]))
                    normal_vector_rotated = rotate_vector(normal_vector, rotation_axis, azimuths[j_azi])
                    local_roll = math.degrees(math.atan2(normal_vector_rotated[2], normal_vector_rotated[1]))
                    local_force_vector = np.array([Drags, Lifts, 0.0])
                    force_vector_rolled = rotate_vector(local_force_vector, np.array([1.0, 0, 0]), local_roll)
                    global_force_vector = rotate_vector(force_vector_rolled, np.array([0, 0, 1.0]), local_yaw)
                    
                    total_global_force_vector[i_inflow, j_azi, :] += global_force_vector
                    total_global_moment_vector[i_inflow, j_azi, :] += global_force_vector * global_pos

                    # print("total_global_force_vector: ", total_global_force_vector[i_inflow, j_azi, :])
                    
                    STlength = chord * abs(math.sin(math.radians(aoa_deg)))
                    frequencies_cf = ST_cf * (V_eff / STlength)
                    
                    # Record the worst case overlap, and where it happened
                    for lstrouhaul in range(min(n_freq_depth, len(frequencies_cf))):
                        if amps_cf[lstrouhaul] > amplitude_coeff_cutoff:
                            for jnatfreq in range(natfreqs.shape[0]):
                                for kharmonic in range(1, n_harmonic + 1):
                                    percdiff = (frequencies_cf[lstrouhaul] - natfreqs[jnatfreq] * kharmonic) / (natfreqs[jnatfreq] * kharmonic) * 100
                                    
                                    if percdiff_matrix[i_inflow, j_azi] > abs(percdiff):
                                        percdiff_matrix[i_inflow, j_azi] = abs(percdiff)
                                        percdiff_info[i_inflow, j_azi] = f"{percdiff} percdiff Occurs for NatFreq: {natfreqs[jnatfreq]} at Harmonic: {kharmonic} with Shedding frequency: {frequencies_cf[lstrouhaul]} (Strouhaul depth {lstrouhaul}) AmplitudeCoeff: {amps_cf[lstrouhaul]} in Comp: {comp.id} at pt#: {ipt+1}"
                    
                    # Output data for just the requested point
                    if viv_params.output_azimuth_vinf[0] == azimuths[j_azi] and viv_params.output_azimuth_vinf[1] == inflow_speeds[i_inflow]:
                        # Recreate the time signal for the sampled ST information
                        cl_signal = reconstruct_signal(ST_cl * (V_eff / STlength), amps_cl, phases_cl, viv_params.output_time)
                        cd_signal = reconstruct_signal(ST_cd * (V_eff / STlength), amps_cd, phases_cd, viv_params.output_time)
                        
                        # Create force vectors for each time point
                        local_force_vectors = []
                        for cd, cl in zip(cd_signal, cl_signal):
                            local_force_vectors.append(np.array([cd * q, cl * q, 0.0]))
                        
                        # Rotate force vectors
                        force_vector_rolled_list = [rotate_vector(local_force, np.array([1.0, 0, 0]), local_roll) for local_force in local_force_vectors]
                        global_force_vector_list = [rotate_vector(force_rolled, np.array([0, 0, 1.0]), local_yaw) for force_rolled in force_vector_rolled_list]
                        
                        # Store in the output array
                        global_force_vector_nodes[:, :, inode-1] = np.array(global_force_vector_list)

    return percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes


#@profile
def compute_thrust_torque_spectrum(components: List[Component], 
                                  affts: Dict[str, AirfoilFFT],
                                  viv_params: VIV_Params,
                                  natfreqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the mean thrust and torque, as well as their frequency-domain spectra, over a range of inflow speeds and azimuthal orientations.

    Args:
        components: List of structural components including geometry, orientation, and segment-wise parameters.
        affts: Dictionary mapping airfoil IDs to their FFT-derived lift/drag/moment spectra.
        viv_params: Encapsulates all analysis parameters, including inflow, rotation axis, fluid properties, and plotting settings.
        natfreqs: Natural frequencies to compare against.

    Returns:
        percdiff_matrix: Matrix of percent differences between shedding frequencies and natural frequencies.
        percdiff_info: Matrix of strings with information about the worst percent differences.
        total_global_force_vector: Total global force vector for each inflow speed and azimuth.
        total_global_moment_vector: Total global moment vector for each inflow speed and azimuth.
        global_force_vector_nodes: Global force vector for each node over time.

    Notes:
        - Airfoil FFT data is interpolated by Reynolds number and angle of attack.
        - The segment's local AOA determines how CL and CD spectra are rotated into global inflow and torque components.
    """
    inflow_speeds = viv_params.inflow_speeds
    azimuths = viv_params.azimuths
    rotation_axis = viv_params.rotation_axis
    fluid_density = viv_params.fluid_density
    fluid_dynamicviscosity = viv_params.fluid_dynamicviscosity
    n_harmonic = viv_params.n_harmonic
    amplitude_coeff_cutoff = viv_params.amplitude_coeff_cutoff
    n_freq_depth = viv_params.n_freq_depth

    n_inflow = len(inflow_speeds)
    n_az = len(azimuths)

    total_global_force_vector = np.zeros((n_inflow, n_az, 3))
    total_global_moment_vector = np.zeros((n_inflow, n_az, 3))
    percdiff_matrix = np.ones((n_inflow, n_az)) * 1000
    percdiff_info = np.empty((n_inflow, n_az), dtype=object)

    total_nodes = sum([comp.shape_xyz.shape[0] for comp in components])
    global_force_vector_nodes = np.zeros((len(viv_params.output_time), 3, total_nodes))

    for i_inflow in range(n_inflow):
        Vinf = np.array(viv_params.inflow_vec) / np.linalg.norm(viv_params.inflow_vec) * inflow_speeds[i_inflow]
        
        for j_azi in range(n_az):
            # Negative since rotating the inflow in the negative direction is the same as rotating the structure in the positive
            Vin_rotated = rotate_vector(Vinf, rotation_axis, -azimuths[j_azi])
            
            inode = 0
            for comp in components:
                N_pts = comp.shape_xyz.shape[0]
                
                for ipt in range(N_pts):
                    inode += 1
                    
                    global_pos = comp.shape_xyz_global[ipt]
                    chord = comp.chord[ipt]
                    
                    afid = comp.airfoil_ids[ipt]
                    afft = affts.get(afid, affts["default"])  # get the id, or use the default
                    
                    chord_vector = comp.chord_vector[ipt, :]
                    normal_vector = comp.normal_vector[ipt, :]
                    
                    V_chord = np.dot(Vin_rotated, chord_vector / np.linalg.norm(chord_vector))
                    V_normal = np.dot(Vin_rotated, normal_vector / np.linalg.norm(normal_vector))
                    
                    aoa_rad = math.atan2(V_normal, V_chord)  # Using atan2 for correct quadrant
                    aoa_deg = math.degrees(aoa_rad)
                    V_eff = math.sqrt(V_normal**2 + V_chord**2)  # Fundamental assumption that spanwise flow doesn't impact lift and drag
                    Re = fluid_density * V_eff * chord / fluid_dynamicviscosity
                    q = 0.5 * fluid_density * V_eff**2 * chord
                    
                    # Interpolate FFT spectrum
                    ST_cl, amps_cl, phases_cl = interpolate_fft_spectrum(afft, Re, aoa_deg, 'CL', n_freq_depth=n_freq_depth)
                    ST_cd, amps_cd, phases_cd = interpolate_fft_spectrum(afft, Re, aoa_deg, 'CD', n_freq_depth=n_freq_depth)
                    ST_cf, amps_cf, phases_cf = interpolate_fft_spectrum(afft, Re, aoa_deg, 'CF', n_freq_depth=n_freq_depth)
                    
                    Lifts = amps_cl[0] * q
                    Drags = amps_cd[0] * q
                    
                    # Calculate the global force vector
                    chord_vector_rotated = rotate_vector(chord_vector, rotation_axis, azimuths[j_azi])
                    local_yaw = math.degrees(math.atan2(chord_vector_rotated[1], chord_vector_rotated[0]))
                    normal_vector_rotated = rotate_vector(normal_vector, rotation_axis, azimuths[j_azi])
                    local_roll = math.degrees(math.atan2(normal_vector_rotated[2], normal_vector_rotated[1]))
                    local_force_vector = np.array([Drags, Lifts, 0.0])
                    force_vector_rolled = rotate_vector(local_force_vector, np.array([1.0, 0, 0]), local_roll)
                    global_force_vector = rotate_vector(force_vector_rolled, np.array([0, 0, 1.0]), local_yaw)
                    
                    total_global_force_vector[i_inflow, j_azi, :] += global_force_vector
                    total_global_moment_vector[i_inflow, j_azi, :] += global_force_vector * global_pos
                    
                    STlength = chord * abs(math.sin(math.radians(aoa_deg)))
                    frequencies_cf = ST_cf * (V_eff / STlength)
                    
                    # Record the worst case overlap, and where it happened
                    for lstrouhaul in range(min(n_freq_depth, len(frequencies_cf))):
                        if amps_cf[lstrouhaul] > amplitude_coeff_cutoff:
                            for jnatfreq in range(natfreqs.shape[0]):
                                for kharmonic in range(1, n_harmonic + 1):
                                    percdiff = (frequencies_cf[lstrouhaul] - natfreqs[jnatfreq] * kharmonic) / (natfreqs[jnatfreq] * kharmonic) * 100
                                    
                                    if percdiff_matrix[i_inflow, j_azi] > abs(percdiff):
                                        percdiff_matrix[i_inflow, j_azi] = abs(percdiff)
                                        percdiff_info[i_inflow, j_azi] = f"{percdiff} percdiff Occurs for NatFreq: {natfreqs[jnatfreq]} at Harmonic: {kharmonic} with Shedding frequency: {frequencies_cf[lstrouhaul]} (Strouhaul depth {lstrouhaul}) AmplitudeCoeff: {amps_cf[lstrouhaul]} in Comp: {comp.id} at pt#: {ipt+1}"
                    
                    # Output data for just the requested point
                    if viv_params.output_azimuth_vinf[0] == azimuths[j_azi] and viv_params.output_azimuth_vinf[1] == inflow_speeds[i_inflow]:
                        # Recreate the time signal for the sampled ST information
                        cl_signal = reconstruct_signal(ST_cl * (V_eff / STlength), amps_cl, phases_cl, viv_params.output_time)
                        cd_signal = reconstruct_signal(ST_cd * (V_eff / STlength), amps_cd, phases_cd, viv_params.output_time)
                        
                        # Create force vectors for each time point
                        local_force_vectors = []
                        for cd, cl in zip(cd_signal, cl_signal):
                            local_force_vectors.append(np.array([cd * q, cl * q, 0.0]))
                        
                        # Rotate force vectors
                        force_vector_rolled_list = [rotate_vector(local_force, np.array([1.0, 0, 0]), local_roll) for local_force in local_force_vectors]
                        global_force_vector_list = [rotate_vector(force_rolled, np.array([0, 0, 1.0]), local_yaw) for force_rolled in force_vector_rolled_list]
                        
                        # Store in the output array
                        global_force_vector_nodes[:, :, inode-1] = np.array(global_force_vector_list)

    return percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes


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

    Assumptions / conventions:
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
    fs = 1.0 / dt
    fnyq = 0.5 * fs

    # Nyquist check with epsilon tolerance
    eps = np.finfo(np.float64).eps
    fmax = float(np.max(freqs)) if freqs.size else 0.0
    if fmax > (fnyq + eps * max(1.0, fnyq)):
        print(f"Warning: Max frequency {fmax:.6g} Hz exceeds Nyquist {fnyq:.6g} Hz implied by tvec; reconstruction may alias.")

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
    k = axis / np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)  # Normalize axis
    v = vec
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
