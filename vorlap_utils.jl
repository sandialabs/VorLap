# === Thrust and Torque Spectrum ===
"""
    compute_thrust_torque_spectrum(components::Vector{Component}, affts::Dict{String, AirfoilFFT}, viv_params::VIV_Params)

Computes the mean thrust and torque, as well as their frequency-domain spectra, over a range of inflow speeds and azimuthal orientations.

# Arguments
- `components::Vector{Component}`: List of structural components including geometry, orientation, and segment-wise parameters.
- `affts::Dict{String, AirfoilFFT}`: Dictionary mapping airfoil IDs to their FFT-derived lift/drag/moment spectra.
- `viv_params::VIV_Params`: Encapsulates all analysis parameters, including inflow, rotation axis, fluid properties, and plotting settings.

# Fields in `viv_params`
- `fluid_density::Float64`: Fluid density [kg/m³].
- `fluid_dynamicviscosity::Float64`: Fluid dynamic viscosity [Pa·s].
- `rotation_axis::Vector{Float64}`: Global axis of rotation for torque projection (unit vector).
- `inflow_vec::Vector{Float64}`: Inflow direction in global coordinates (assumed constant).
- `azimuths::Vector{Float64}`: Azimuth sweep angles [degrees] applied to each component.
- `inflow_speeds::Vector{Float64}`: Uniform flow speeds [m/s] to simulate.
- Other fields are not directly used in this function.

# Frame Convention
- Inflow is aligned with the **+x global axis**, unless overridden via `inflow_vec`.
- Lift acts normal to local inflow and drag acts parallel to it.
- Forces from each segment are resolved globally and summed.
- Torque is projected onto the specified `rotation_axis`.

# Returns
- `thrust_mean::Matrix{Float64}`: Mean thrust [N] over (inflow speed × azimuth) grid.
- `torque_mean::Matrix{Float64}`: Mean torque [N·m] over same grid, projected about `rotation_axis`.
- `thrust_spectrum::Matrix{Float64}`: Amplitude spectrum [N] of total thrust per inflow speed.
- `torque_spectrum::Matrix{Float64}`: Amplitude spectrum [N·m] of total torque per inflow speed.

# Units
- Force: Newtons [N]
- Torque: Newton-meters [N·m]
- Velocity: meters per second [m/s]
- Angle: degrees input, radians internal
- Frequency: Hertz [Hz]

# Notes
- Airfoil FFT data is interpolated by Reynolds number and angle of attack.
- The segment's local AOA determines how CL and CD spectra are rotated into global inflow and torque components.
"""
function compute_thrust_torque_spectrum(components, affts::Dict{String, AirfoilFFT},viv_params,natfreqs)

    inflow_speeds = viv_params.inflow_speeds
    azimuths = viv_params.azimuths
    rotation_axis = viv_params.rotation_axis
    fluid_density = viv_params.fluid_density
    fluid_dynamicviscosity = viv_params.fluid_dynamicviscosity
    n_harmonic = viv_params.n_harmonic
    amplitude_coeff_cutoff = viv_params.amplitude_coeff_cutoff
    n_freq_depth = viv_params.n_freq_depth

    n_inflow = length(inflow_speeds)
    n_az = length(azimuths)

    total_global_force_vector = zeros(n_inflow, n_az,3)
    total_global_moment_vector = zeros(n_inflow, n_az,3)
    percdiff_matrix = ones(n_inflow, n_az).*1000
    percdiff_info = Matrix{String}(undef, n_inflow, n_az)

    total_nodes = sum([size(comp.shape_xyz, 1) for comp in components])
    global_force_vector_nodes = zeros(length(viv_params.output_time),3,total_nodes)
    for i_inflow = 1:n_inflow
        
        Vinf = LinearAlgebra.normalize(viv_params.inflow_vec).*inflow_speeds[i_inflow]
        for j_azi = 1:n_az

            Vin_rotated = rotate_vector(Vinf,rotation_axis,-azimuths[j_azi]) # Negative since rotating the inflow in the negative direction is the same as rotating the structure in the positive

            inode = 0
            for comp in components

                N_pts = size(comp.shape_xyz, 1)

                for ipt in 1:N_pts

                    inode += 1

                    global_pos = comp.shape_xyz_global[ipt,:]

                    chord = comp.chord[ipt]
                    
                    afid = comp.airfoil_ids[ipt]
                    afft = get(affts, afid, affts["default"]) #get the id, or use the default #TODO: input the default?

                    chord_vector = comp.chord_vector[ipt,:]
                    normal_vector = comp.normal_vector[ipt,:]

                    V_chord = LinearAlgebra.dot(Vin_rotated,LinearAlgebra.normalize(chord_vector))
                    V_normal = LinearAlgebra.dot(Vin_rotated,LinearAlgebra.normalize(normal_vector))

                    aoa_rad = atan(V_normal,V_chord) #TODO: check at negative aoa
                    aoa_deg = rad2deg(aoa_rad)
                    V_eff = sqrt(V_normal^2+V_chord^2) #Fundamental assumption that spanwise flow doesn't impact lift and drag
                    Re = fluid_density * V_eff * chord / fluid_dynamicviscosity
                    q = 0.5 * fluid_density * V_eff^2 * chord

                    #TODO: re-use interpolation instances since that is taking up the bulk of run time cycles
                    ST_cl, amps_cl, phases_cl = interpolate_fft_spectrum(afft, Re, aoa_deg, :CL; n_freq_depth)
                    ST_cd, amps_cd, phases_cd = interpolate_fft_spectrum(afft, Re, aoa_deg, :CD; n_freq_depth)
                    ST_cf, amps_cf, phases_cf = interpolate_fft_spectrum(afft, Re, aoa_deg, :CF; n_freq_depth)
                    Lifts = amps_cl[1].*q
                    Drags = amps_cd[1].*q

                    # Calculate the global force vector (fx, fy, fz at each point, we'll multiply by the offset and integrate later to get thrust and torque about the rotation axis)
                    # lift is in the direction perpendicular to the flow, rotated by the roll angle, immune to pitch. Drag is in the direction of the flow, but rotated by the yaw angle since we are doing inflow in the direction of the airfoil not thickness or chord increases. 
                    # Per the 0,0,0 definition of a component, it is vertical span with the airfoil going into the flow. So, drag is simply 
                    # We take the lift and drag vectors and combine them rotate them 
                    # NOTE: this has the inflow vector hard coded as [1,0,0]
                    # If we rotate the chord and normal lines by the rotation angle, we can get the yaw from the chord line x,y values. Then we can get the roll from the normal y z, 
                    # TODO: should this be rotated by the azimuth to get into the global frame? or is the system effectively rotating and the inflow is stationary
                    chord_vector_rotated = rotate_vector(chord_vector,rotation_axis,azimuths[j_azi])
                    local_yaw = atand(chord_vector_rotated[2],chord_vector_rotated[1])
                    normal_vector_rotated = rotate_vector(normal_vector,rotation_axis,azimuths[j_azi])
                    local_roll = atand(normal_vector_rotated[3],normal_vector_rotated[2])
                    local_force_vector = [Drags, Lifts, zero(Lifts)]
                    force_vector_rolled = rotate_vector(local_force_vector,[1.0,0,0],local_roll)
                    global_force_vector = rotate_vector(force_vector_rolled,[0,0,1.0],local_yaw)
                    total_global_force_vector[i_inflow,j_azi,:] = total_global_force_vector[i_inflow,j_azi,:] + global_force_vector
                    total_global_moment_vector[i_inflow,j_azi,:] = total_global_moment_vector[i_inflow,j_azi,:] + (global_force_vector .* global_pos)
                    
                    STlength = chord*abs(sind(aoa_deg))#
                    frequencies_cf = ST_cf .* (V_eff/STlength)
                    # record the worst case overlap, and where it happened
                    for lstrouhaul = 1:n_freq_depth#axes(frequencies_cf,1)
                        if amps_cf[lstrouhaul]>amplitude_coeff_cutoff
                            for jnatfreq in axes(natfreqs,1)
                                for kharmonic = 1:n_harmonic
                                    percdiff = (frequencies_cf[lstrouhaul]-natfreqs[jnatfreq]*kharmonic)./(natfreqs[jnatfreq]*kharmonic)*100
                                    
                                    if percdiff_matrix[i_inflow,j_azi]>abs(percdiff)
                                        percdiff_matrix[i_inflow,j_azi] = abs(percdiff)
                                        percdiff_info[i_inflow,j_azi] = "$percdiff percdiff Occurs for NatFreq: $(natfreqs[jnatfreq]) at Harmonic: $kharmonic with Shedding frequency: $(frequencies_cf[lstrouhaul]) (Strouhaul depth $lstrouhaul) AmplitudeCoeff: $(amps_cf[lstrouhaul]) in Comp: $(comp.id) at pt#: $(ipt)"
                                    end
                                end
                            end
                        end
                    end
                    if viv_params.output_azimuth_vinf[1] == azimuths[j_azi] && viv_params.output_azimuth_vinf[2] == inflow_speeds[i_inflow] # output data for just the requested point
                        # recreate the time signal for the sampled ST information
                        cl_signal = reconstruct_signal(ST_cl .* (V_eff/STlength), amps_cl, phases_cl, viv_params.output_time)
                        cd_signal = reconstruct_signal(ST_cd .* (V_eff/STlength), amps_cd, phases_cd, viv_params.output_time)
                        # local_force_vector = [cd_signal.*q, cl_signal.*q, zero(cl_signal)]
                        local_force_vector = [[cd*q, cl*q, 0.0] for (cd, cl) in zip(cd_signal, cl_signal)]
                        force_vector_rolled = rotate_vector.(local_force_vector,Ref([1.0,0,0]),Ref(local_roll))
                        global_force_vector = rotate_vector.(force_vector_rolled,Ref([0,0,1.0]),Ref(local_yaw))
                        global_force_vector_nodes[:,1:3,inode] = hcat(global_force_vector...)'
                    end
                end
            end
        end
    end

    # return thrust_mean, torque_mean, thrust_spectrum, torque_spectrum
    return percdiff_matrix, percdiff_info,total_global_force_vector,total_global_moment_vector,global_force_vector_nodes
end

"""
    reconstruct_signal(freqs::Vector{Float64}, amps::Vector{Float64}, phases::Vector{Float64}, t::Vector{Float64})

Reconstructs a time-domain signal from FFT amplitude, frequency, and phase data.

# Arguments
- `freqs`: Frequency vector (Hz)
- `amps`: Amplitudes corresponding to each frequency
- `phases`: Phase offsets (radians) corresponding to each frequency
- `t`: Time vector (s)

# Returns
- `signal`: Time-domain signal as a vector of Float64
"""
function reconstruct_signal(freqs::Vector{Float64}, amps::Vector{Float64}, phases::Vector{Float64}, tvec::Vector{Float64})
    dt = tvec[2] - tvec[1]
    if maximum(freqs) > 2/dt
        @warn "The maximum frequency after the applied input cutoff parameters is $(maximum(freqs)), while the maximum nyquist frequency possible in the input time vector is $(2/(dt))"
    end

    signal = zeros(length(tvec))
    for (i, f) in enumerate(freqs)

        if iszero(f)
            signal .+= amps[i]  # DC term
        else
            signal .+= amps[i] .* sin.(2π * f .* tvec .+ phases[i])
        end
    end
    return signal
end


"""
    rotate_vector(vec::Vector{Float64}, axis::Vector{Float64}, angle_deg::Float64) -> Vector{Float64}

Rotates a 3D vector `vec` around a given `axis` by `angle_deg` degrees using Rodrigues' rotation formula.

# Parameters
- `vec`: Vector to rotate (length-3, any direction).
- `axis`: Rotation axis (length-3, not required to be normalized).
- `angle_deg`: Rotation angle in degrees (positive is right-hand rule about the axis).

# Returns
- Rotated 3D vector (length-3, Float64).
"""
function rotate_vector(vec::Vector{Float64}, axis::Vector{Float64}, angle_deg::Float64)
    θ = deg2rad(angle_deg)
    k = axis ./ sqrt(axis[1]^2+axis[2]^2+axis[3]^2) # Normalize axis
    v = vec
    return v * cos(θ) + LinearAlgebra.cross(k, v) * sin(θ) + k * LinearAlgebra.dot(k, v) * (1 - cos(θ))
end


"""
    rotationMatrix(euler::AbstractVector{<:Real}) -> Matrix{Float64}

Computes a 3×3 rotation matrix from Euler angles using the ZYX convention (yaw–pitch–roll), where:
- Z: yaw (heading)
- Y: pitch (elevation)
- X: roll (bank)

The angles are provided in **degrees** and applied in **Z–Y–X** order (extrinsic frame), meaning:
1. Rotate about global Z axis (yaw)
2. Then about the global Y axis (pitch)
3. Then about the global X axis (roll)

# Arguments
- `euler::Vector{Float64}`: Euler angles `[roll, pitch, yaw]` in **degrees**.

# Returns
- `R_global::Matrix{Float64}`: A 3×3 rotation matrix for transforming a local vector into the global frame.

# Example
```julia
euler = [30.0, 15.0, 60.0]  # roll, pitch, yaw in degrees
R = rotationMatrix(euler)
v_local = [1.0, 0.0, 0.0]
v_global = R * v_local
````
"""
function rotationMatrix(euler::Vector{Float64})
    cz, sz = cosd(euler[3]), sind(euler[3])
    cy, sy = cosd(euler[2]), sind(euler[2])
    cx, sx = cosd(euler[1]), sind(euler[1])
    Rz = [cz -sz 0.0; sz cz 0.0; 0.0 0.0 1.0]
    Ry = [cy 0.0 sy; 0.0 1.0 0.0; -sy 0.0 cy]
    Rx = [1.0 0.0 0.0; 0.0 cx -sx; 0.0 sx cx]
    R_global = Rz * Ry * Rx
    return R_global
end


function calc_structure_vectors_andplot!(components::Vector{Component}, viv_params::VIV_Params)
    plt = Plots.plot(; aspect_ratio=:equal, legend=false)

    # Draw rotation axis #TODO apply this in the calculations
    axis_len = maximum(vcat([maximum(comp.shape_xyz) for comp in components]...)) * 1.2
    origin = viv_params.rotation_axis_offset
    arrow = viv_params.rotation_axis .* axis_len .+ origin
    Plots.plot!(plt, [origin[1], arrow[1]], [origin[2], arrow[2]], [origin[3], arrow[3]],
          lw=2, color=:black, label="Rotation Axis")
    
    # Draw inflow vector
    inflow_origin = [-axis_len/1.5, 0.0, axis_len/2]  # or viv_params.rotation_axis_offset
    inflow_arrow = viv_params.inflow_vec .* axis_len .* 0.5 .+ inflow_origin
    Plots.plot!(plt,
    [inflow_origin[1], inflow_arrow[1]],
    [inflow_origin[2], inflow_arrow[2]],
    [inflow_origin[3], inflow_arrow[3]],
    lw=2, color=:blue, label="Inflow")

    absmin = min(minimum(inflow_origin),minimum(inflow_arrow))
    absmin = min(absmin,min(minimum(origin),minimum(arrow)))
    absmax = max(maximum(inflow_origin),maximum(inflow_arrow))
    absmax = max(absmax,max(maximum(origin),maximum(arrow)))
    for (cidx, comp) in enumerate(components)
        color = viv_params.plot_cycle[mod1(cidx, length(viv_params.plot_cycle))]
        N_Airfoils = size(comp.shape_xyz, 1)
        N_Af_coords = 200
        af_coords_local = zeros(N_Airfoils,N_Af_coords,3)
        chordline_local = zeros(N_Airfoils,2,3) # 2 start and stop, 3 xyz of each
        normalline_local = zeros(N_Airfoils,2,3) # 2 start and stop, 3 xyz of each
        pitch = comp.pitch
        for ipt in 1:size(comp.shape_xyz, 1)
            pt = comp.shape_xyz[ipt, :]
            chord = comp.chord[ipt]
            twist = comp.twist[ipt]+pitch[1]
            thickness = comp.thickness[ipt]
            offset = comp.offset[ipt]
            
            # Each component's direction is defined by the comp.rotation rx, ry, rz angles, where 0,0,0 is pointing straight update
            # Let's create a local point cloud of xyz airfoil points, based on the shape input, then rotate that into position
            airfoil2d = load_airfoil_coords("$(viv_params.airfoil_folder)$(comp.airfoil_ids[ipt]).csv")
            xy_scaled = resample_airfoil(airfoil2d;npoints=N_Af_coords)
            xy_scaled[:, 1] = xy_scaled[:, 1] .* chord .- chord*offset
            xy_scaled[:, 2] = xy_scaled[:, 2] .* thickness .* chord
            R_twist = [cosd(twist) -sind(twist);
                       sind(twist)  cosd(twist)]
            xy_scaled_twisted = (R_twist*xy_scaled')'
            xy_scaled_twisted_translated = [xy_scaled_twisted[:,1].+pt[1]  xy_scaled_twisted[:,2].+pt[2]]
            af_coords_local[ipt,:,:] = [xy_scaled_twisted_translated[:,1] xy_scaled_twisted_translated[:,2] zeros(size(xy_scaled_twisted_translated,1)).+pt[3]]

            chordline_scaled_twisted = (R_twist*[0 0;2*chord 0]')'
            chordline_scaled_twisted_translated = [chordline_scaled_twisted[:,1].+pt[1]  chordline_scaled_twisted[:,2].+pt[2]]
            chordline_local[ipt,:,:] = [chordline_scaled_twisted_translated[:,1] chordline_scaled_twisted_translated[:,2] zeros(size(chordline_scaled_twisted_translated,1)).+pt[3]]

            normalline_scaled_twisted = (R_twist*[0 0;0 2*chord]')'
            # Calculate the local skew/sweep angle
            if ipt == 1
                d_xyz = comp.shape_xyz[ipt+1, :]-comp.shape_xyz[ipt, :]
            elseif ipt == size(comp.shape_xyz, 1)
                d_xyz = comp.shape_xyz[ipt, :]-comp.shape_xyz[ipt-1, :]
            else
                d_xyz1 = comp.shape_xyz[ipt+1, :]-comp.shape_xyz[ipt, :]
                d_xyz2 = comp.shape_xyz[ipt, :]-comp.shape_xyz[ipt-1, :]
                d_xyz = (d_xyz1 + d_xyz2)./2
            end
            skew = atan(d_xyz[3],d_xyz[2])
            R_skew = rotationMatrix([skew*180/pi-90,0.0,0.0])
            normalline_scaled_twisted3D = [normalline_scaled_twisted[:,1] normalline_scaled_twisted[:,2] zeros(size(normalline_scaled_twisted,1))]
            normalline_scaled_twisted_skewed = (R_skew*normalline_scaled_twisted3D')'
            normalline_scaled_twisted_skewed_translated = [normalline_scaled_twisted_skewed[:,1].+pt[1]  normalline_scaled_twisted_skewed[:,2].+pt[2] normalline_scaled_twisted_skewed[:,3].+pt[3]]
            normalline_local[ipt,:,:] = normalline_scaled_twisted_skewed_translated
        end
        # Now that the local point cloud is generated, let's rotate and move it into position
        af_cloud_local = reshape(af_coords_local, :, size(af_coords_local, 3)) 
        chordline_cloud_local = reshape(chordline_local, :, size(chordline_local, 3)) 
        normalline_cloud_local = reshape(normalline_local, :, size(normalline_local, 3)) 

        euler = comp.rotation
        R_global = rotationMatrix(euler)

        af_coords_global = (R_global*af_cloud_local')'
        af_coords_global[:,1] .+= comp.translation[1]
        af_coords_global[:,2] .+= comp.translation[2]
        af_coords_global[:,3] .+= comp.translation[3]

        chordline_global = (R_global*chordline_cloud_local')'
        chordline_global[:,1] .+= comp.translation[1]
        chordline_global[:,2] .+= comp.translation[2]
        chordline_global[:,3] .+= comp.translation[3]

        normalline_global = (R_global*normalline_cloud_local')'
        normalline_global[:,1] .+= comp.translation[1]
        normalline_global[:,2] .+= comp.translation[2]
        normalline_global[:,3] .+= comp.translation[3]

        # Manually enforce equal axis scaling
        xlims = Plots.extrema(af_coords_global[:, 1])
        ylims = Plots.extrema(af_coords_global[:, 2])
        zlims = Plots.extrema(af_coords_global[:, 3])

        absmin = min(absmin,minimum([xlims[1],ylims[1],zlims[1]]))
        absmax = max(absmax,maximum([xlims[2],ylims[2],zlims[2]]))

        Plots.plot!(plt, af_coords_global[:, 1], af_coords_global[:, 2], af_coords_global[:, 3])#; color=color)

        halfIdx = Int(size(chordline_global,1)/2)
        for idx = 1:halfIdx
            components[cidx].chord_vector[idx,:] = chordline_global[halfIdx+idx,:] - chordline_global[idx,:]
            components[cidx].normal_vector[idx,:] = normalline_global[halfIdx+idx,:] - normalline_global[idx,:]
            components[cidx].shape_xyz_global[idx,:] = chordline_global[idx,:]

            linex = [chordline_global[idx,1],chordline_global[halfIdx+idx,1]]
            liney = [chordline_global[idx,2],chordline_global[halfIdx+idx,2]]
            linez = [chordline_global[idx,3],chordline_global[halfIdx+idx,3]]
            Plots.plot!(plt, linex, liney, linez, color=color)

            linex = [normalline_global[idx,1],normalline_global[halfIdx+idx,1]]
            liney = [normalline_global[idx,2],normalline_global[halfIdx+idx,2]]
            linez = [normalline_global[idx,3],normalline_global[halfIdx+idx,3]]
            Plots.plot!(plt, linex, liney, linez, linestyle = :dash, color=color, aspect_ratio=:equal)
        end

    end

    # Plots.plot!(plt;xlims=(absmin,absmax),ylims=(absmin,absmax),zlims=(absmin,absmax))

    Plots.display(plt)

end
