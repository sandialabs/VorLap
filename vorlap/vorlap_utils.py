"""
Utility functions for the VorLap package.
"""

import numpy as np
import math
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional, Union, Any
from .structs import AirfoilFFT, Component, VIV_Params
import os


#@profile
def compute_thrust_torque_spectrum_optimized(components: List[Component], 
                                           affts: Dict[str, AirfoilFFT],
                                           viv_params: VIV_Params,
                                           natfreqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized version of compute_thrust_torque_spectrum using cached interpolators and vectorized operations.
    
    Same interface and outputs as the original function, but with significant performance improvements.
    """
    from .fileio import interpolate_fft_spectrum_optimized
    
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
                    q = 0.5 * fluid_density * V_eff**2 * chord
                    
                    # Optimized interpolation: get all three fields at once
                    results = interpolate_fft_spectrum_optimized(afft, Re, aoa_deg, ['CL', 'CD', 'CF'], n_freq_depth=n_freq_depth)
                    ST_cl, amps_cl, phases_cl = results['CL']
                    ST_cd, amps_cd, phases_cd = results['CD']
                    ST_cf, amps_cf, phases_cf = results['CF']
                    
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


#@profile
def reconstruct_signal(freqs: np.ndarray, amps: np.ndarray, phases: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Reconstructs a time-domain signal from FFT amplitude, frequency, and phase data.

    Args:
        freqs: Frequency vector (Hz)
        amps: Amplitudes corresponding to each frequency
        phases: Phase offsets (radians) corresponding to each frequency
        tvec: Time vector (s)

    Returns:
        signal: Time-domain signal as a vector of float
    """
    dt = tvec[1] - tvec[0]
    if np.max(freqs) > 2/dt:
        print(f"Warning: The maximum frequency after the applied input cutoff parameters is {np.max(freqs)}, while the maximum nyquist frequency possible in the input time vector is {2/dt}")
    
    signal = np.zeros(len(tvec))
    for i, f in enumerate(freqs):
        if f == 0:
            signal += amps[i]  # DC term
        else:
            signal += amps[i] * np.sin(2 * np.pi * f * tvec + phases[i])
    
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


def calc_structure_vectors_andplot(components: List[Component], viv_params: VIV_Params):
    """
    Calculates structure vectors and creates a plot.

    Args:
        components: List of structural components.
        viv_params: Configuration parameters.

    Returns:
        fig: Plotly figure object.
    """
    from .fileio import load_airfoil_coords, resample_airfoil
    
    # Create a new 3D figure
    fig = go.Figure()
    
    # Draw rotation axis
    axis_len = max([np.max(comp.shape_xyz) for comp in components]) * 1.2
    origin = viv_params.rotation_axis_offset
    arrow = viv_params.rotation_axis * axis_len + origin
    
    fig.add_trace(go.Scatter3d(
        x=[origin[0], arrow[0]],
        y=[origin[1], arrow[1]],
        z=[origin[2], arrow[2]],
        mode='lines',
        line=dict(color='black', width=2),
        name='Rotation Axis'
    ))
    
    # Draw inflow vector
    inflow_origin = np.array([-axis_len/1.5, 0.0, axis_len/2])
    inflow_arrow = viv_params.inflow_vec * axis_len * 0.5 + inflow_origin
    
    fig.add_trace(go.Scatter3d(
        x=[inflow_origin[0], inflow_arrow[0]],
        y=[inflow_origin[1], inflow_arrow[1]],
        z=[inflow_origin[2], inflow_arrow[2]],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Inflow'
    ))
    
    absmin = min(min(inflow_origin), min(inflow_arrow))
    absmin = min(absmin, min(min(origin), min(arrow)))
    absmax = max(max(inflow_origin), max(inflow_arrow))
    absmax = max(absmax, max(max(origin), max(arrow)))
    
    for cidx, comp in enumerate(components):
        color = viv_params.plot_cycle[cidx % len(viv_params.plot_cycle)]
        N_Airfoils = comp.shape_xyz.shape[0]
        N_Af_coords = 200
        af_coords_local = np.zeros((N_Airfoils, N_Af_coords, 3))
        chordline_local = np.zeros((N_Airfoils, 2, 3))  # 2 start and stop, 3 xyz of each
        normalline_local = np.zeros((N_Airfoils, 2, 3))  # 2 start and stop, 3 xyz of each
        pitch = comp.pitch
        
        for ipt in range(comp.shape_xyz.shape[0]):
            pt = comp.shape_xyz[ipt, :]
            chord = comp.chord[ipt]
            twist = comp.twist[ipt] + pitch[0]
            thickness = comp.thickness[ipt]
            offset = comp.offset[ipt]
            
            # Each component's direction is defined by the comp.rotation rx, ry, rz angles, where 0,0,0 is pointing straight update
            # Let's create a local point cloud of xyz airfoil points, based on the shape input, then rotate that into position
            airfoil2d = load_airfoil_coords(f"{viv_params.airfoil_folder}{comp.airfoil_ids[ipt]}.csv")
            xy_scaled = resample_airfoil(airfoil2d, npoints=N_Af_coords)
            xy_scaled[:, 0] = xy_scaled[:, 0] * chord - chord * offset
            xy_scaled[:, 1] = xy_scaled[:, 1] * thickness * chord
            
            R_twist = np.array([
                [math.cos(math.radians(twist)), -math.sin(math.radians(twist))],
                [math.sin(math.radians(twist)), math.cos(math.radians(twist))]
            ])
            
            xy_scaled_twisted = (R_twist @ xy_scaled.T).T
            xy_scaled_twisted_translated = np.column_stack([
                xy_scaled_twisted[:, 0] + pt[0],
                xy_scaled_twisted[:, 1] + pt[1],
            ])
            
            af_coords_local[ipt, :, :] = np.column_stack([
                xy_scaled_twisted_translated[:, 0],
                xy_scaled_twisted_translated[:, 1],
                np.zeros(xy_scaled_twisted_translated.shape[0]) + pt[2]
            ])
            
            chordline_scaled_twisted = (R_twist @ np.array([[0, 0], [2*chord, 0]]).T).T
            chordline_scaled_twisted_translated = np.column_stack([
                chordline_scaled_twisted[:, 0] + pt[0],
                chordline_scaled_twisted[:, 1] + pt[1],
            ])
            
            chordline_local[ipt, :, :] = np.column_stack([
                chordline_scaled_twisted_translated[:, 0],
                chordline_scaled_twisted_translated[:, 1],
                np.zeros(chordline_scaled_twisted_translated.shape[0]) + pt[2]
            ])
            
            normalline_scaled_twisted = (R_twist @ np.array([[0, 0], [0, 2*chord]]).T).T
            
            # Calculate the local skew/sweep angle
            if ipt == 0:
                d_xyz = comp.shape_xyz[ipt+1, :] - comp.shape_xyz[ipt, :]
            elif ipt == comp.shape_xyz.shape[0] - 1:
                d_xyz = comp.shape_xyz[ipt, :] - comp.shape_xyz[ipt-1, :]
            else:
                d_xyz1 = comp.shape_xyz[ipt+1, :] - comp.shape_xyz[ipt, :]
                d_xyz2 = comp.shape_xyz[ipt, :] - comp.shape_xyz[ipt-1, :]
                d_xyz = (d_xyz1 + d_xyz2) / 2
            
            skew = math.atan2(d_xyz[2], d_xyz[1])
            R_skew = rotationMatrix(np.array([math.degrees(skew) - 90, 0.0, 0.0]))
            
            normalline_scaled_twisted3D = np.column_stack([
                normalline_scaled_twisted[:, 0],
                normalline_scaled_twisted[:, 1],
                np.zeros(normalline_scaled_twisted.shape[0])
            ])
            
            normalline_scaled_twisted_skewed = (R_skew @ normalline_scaled_twisted3D.T).T
            normalline_scaled_twisted_skewed_translated = np.column_stack([
                normalline_scaled_twisted_skewed[:, 0] + pt[0],
                normalline_scaled_twisted_skewed[:, 1] + pt[1],
                normalline_scaled_twisted_skewed[:, 2] + pt[2]
            ])
            
            normalline_local[ipt, :, :] = normalline_scaled_twisted_skewed_translated
        
        # Now that the local point cloud is generated, let's rotate and move it into position
        # Use Fortran-style (column-major) order to match Julia's reshape behavior
        af_cloud_local = af_coords_local.reshape(-1, af_coords_local.shape[2], order='F')
        chordline_cloud_local = chordline_local.reshape(-1, chordline_local.shape[2], order='F')
        normalline_cloud_local = normalline_local.reshape(-1, normalline_local.shape[2], order='F')
        
        euler = comp.rotation
        R_global = rotationMatrix(euler)
        
        af_coords_global = (R_global @ af_cloud_local.T).T
        af_coords_global[:, 0] += comp.translation[0]
        af_coords_global[:, 1] += comp.translation[1]
        af_coords_global[:, 2] += comp.translation[2]
        
        chordline_global = (R_global @ chordline_cloud_local.T).T
        chordline_global[:, 0] += comp.translation[0]
        chordline_global[:, 1] += comp.translation[1]
        chordline_global[:, 2] += comp.translation[2]
        
        normalline_global = (R_global @ normalline_cloud_local.T).T
        normalline_global[:, 0] += comp.translation[0]
        normalline_global[:, 1] += comp.translation[1]
        normalline_global[:, 2] += comp.translation[2]
        
        # Add airfoil surface to the plot
        fig.add_trace(go.Scatter3d(
            x=af_coords_global[:, 0],
            y=af_coords_global[:, 1],
            z=af_coords_global[:, 2],
            mode='lines',
            line=dict(color=color),
            name=f'Component {comp.id}'
        ))
        
        # Add chord and normal lines
        halfIdx = int(chordline_global.shape[0] / 2)
        for idx in range(halfIdx):
            # Update component vectors
            comp.chord_vector[idx, :] = chordline_global[halfIdx + idx, :] - chordline_global[idx, :]
            comp.normal_vector[idx, :] = normalline_global[halfIdx + idx, :] - normalline_global[idx, :]
            comp.shape_xyz_global[idx, :] = chordline_global[idx, :]
            
            # Add chord line
            fig.add_trace(go.Scatter3d(
                x=[chordline_global[idx, 0], chordline_global[halfIdx + idx, 0]],
                y=[chordline_global[idx, 1], chordline_global[halfIdx + idx, 1]],
                z=[chordline_global[idx, 2], chordline_global[halfIdx + idx, 2]],
                mode='lines',
                line=dict(color=color),
                showlegend=False
            ))
            
            # Add normal line
            fig.add_trace(go.Scatter3d(
                x=[normalline_global[idx, 0], normalline_global[halfIdx + idx, 0]],
                y=[normalline_global[idx, 1], normalline_global[halfIdx + idx, 1]],
                z=[normalline_global[idx, 2], normalline_global[halfIdx + idx, 2]],
                mode='lines',
                line=dict(color=color, dash='dash'),
                showlegend=False
            ))
    
    # Set layout with equal axis scaling
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[absmin, absmax]),
            yaxis=dict(range=[absmin, absmax]),
            zaxis=dict(range=[absmin, absmax])
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    # Display the figure
    fig.show()
    
    return fig


# Import from fileio to avoid circular imports
# Import at runtime to avoid circular imports
def interpolate_fft_spectrum(afft, Re_val, AOA_val, field, n_freq_depth=None):
    """
    This is a wrapper function to avoid circular imports.
    The actual implementation is in fileio.py.
    """
    from .fileio import interpolate_fft_spectrum as _interpolate_fft_spectrum
    return _interpolate_fft_spectrum(afft, Re_val, AOA_val, field, n_freq_depth)
