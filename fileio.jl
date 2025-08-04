"""
    load_components_from_csv(dir::String)

Loads all component geometry and metadata from CSV files in the given directory.

# Parameters
- `dir`: Path to a directory containing CSV files. Each file must follow a two-header format.

# Expected CSV Format
1. First data row contains: `id`, `translation_x`, `translation_y`, `translation_z`, `rotation_x`, `rotation_y`, `rotation_z`
2. Second header row: column names for vectors — must include `x`, `y`, `z`, `chord`, `twist`, `thickness`, and optional `airfoil_id`
3. Remaining rows: vector data for each blade segment or shape point

# Returns
- `Vector{Component}`: List of parsed `Component` structs containing all geometry and configuration data

# Notes
- If `airfoil_id` is missing, `"default"` will be used for all segments in that component
- All transformations are centered at the origin and adjusted by top-level translation/rotation
- All components are assumed to have the span oriented in the z-direction
"""
function load_components_from_csv(dir::String)
    files = filter(f -> endswith(f, ".csv"), readdir(dir; join=true))
    components = Component[]
    for file in files
        raw = CSV.File(file; header=false) |> collect

        # Extract top-level metadata
        id = String(raw[2][1])
        tx = parse(Float64, String(raw[2][2]))
        ty = parse(Float64, String(raw[2][3]))
        tz = parse(Float64, String(raw[2][4]))
        rx = parse(Float64, String(raw[2][5]))
        ry = parse(Float64, String(raw[2][6]))
        rz = parse(Float64, String(raw[2][7]))
        pitch = parse(Float64, String(raw[2][8]))

        # Locate secondary header (usually row 3)
        colnames = map(Symbol, raw[3])
        df = DataFrames.DataFrame(raw[4:end], colnames)

        xyz = hcat(parse.(Float64, String.(df.x)),
           parse.(Float64, String.(df.y)),
           parse.(Float64, String.(df.z)))

        chord = parse.(Float64, String.(df.chord))
        twist = parse.(Float64, String.(df.twist))
        thickness = parse.(Float64, String.(df.thickness))
        offset = parse.(Float64, String.(df.offset))
        airfoil_ids = "airfoil_id" in names(df) ? convert(Vector{String}, df.airfoil_id) : fill("default", DataFrames.nrow(df))

        chord_vec = zeros(size(xyz,1),3) #place holder, will get filled in later
        norm_vec = zeros(size(xyz,1),3) #place holder, will get filled in later
        xyz_global = zero(xyz) #place holder, will get filled in later
        push!(components, Component(id, [tx, ty, tz], [rx, ry, rz], [pitch],
                                    xyz, xyz_global, chord, twist, thickness, offset, airfoil_ids,chord_vec,norm_vec))
    end
    return components
end

"""
    load_airfoil_fft(path::String) -> AirfoilFFT

Loads a processed airfoil unsteady FFT dataset from an HDF5 file.

# Expected HDF5 File Format

The file must contain the following datasets:

- `Airfoilname` :: String — Name of the airfoil (e.g., "NACA0012")
- `Re` :: Vector{Float64} — Reynolds number values (assumed constant across all entries)
- `Thickness` :: Vector{Float64} — Thickness ratio(s) used
- `AOA` :: Vector{Float64} — Angle of attack values in degrees
- `CL_ST`, `CD_ST`, `CM_ST`, `CF_ST` :: 3D Arrays [Re x AOA x freq] — Strouhal numbers for each force/moment
- `CL_Amp`, `CD_Amp`, `CM_Amp`, `CF_Amp` :: 3D Arrays [Re x AOA x freq] — FFT amplitudes for lift, drag, moment, and combined force
- `CL_Pha`, `CD_Pha`, `CM_Pha`, `CF_Pha` :: 3D Arrays [Re x AOA x freq] — FFT phases in radians for each quantity

# Assumptions
- All arrays must share dimensions [Re, AOA, NFreq], where the frequency dimension is sorted by the amplitude
- Phase data is in radians.
- Struhaul data represents unsteady aerodynamics due to vortex shedding.
- No ragged or missing data is allowed.
"""
function load_airfoil_fft(path::String)
    HDF5.h5open(path, "r") do h5
        name      = haskey(h5, "Airfoilname") ? HDF5.read(h5["Airfoilname"]) : basename(path)
        Re        = HDF5.read(h5["Re"])
        Thickness = HDF5.read(h5["Thickness"])
        AOA       = HDF5.read(h5["AOA"])

        CL_ST     = HDF5.read(h5["CL_ST"])
        CD_ST     = HDF5.read(h5["CD_ST"])
        CM_ST     = HDF5.read(h5["CM_ST"])
        CF_ST     = HDF5.read(h5["CF_ST"])

        CL_Amp    = HDF5.read(h5["CL_Amp"])
        CD_Amp    = HDF5.read(h5["CD_Amp"])
        CM_Amp    = HDF5.read(h5["CM_Amp"])
        CF_Amp    = HDF5.read(h5["CF_Amp"])

        CL_Pha    = HDF5.read(h5["CL_Pha"])
        CD_Pha    = HDF5.read(h5["CD_Pha"])
        CM_Pha    = HDF5.read(h5["CM_Pha"])
        CF_Pha    = HDF5.read(h5["CF_Pha"])

        return AirfoilFFT(name, Re, AOA, Thickness,
                          CL_ST, CD_ST, CM_ST, CF_ST,
                          CL_Amp, CD_Amp, CM_Amp, CF_Amp,
                          CL_Pha, CD_Pha, CM_Pha, CF_Pha)
    end
end


"""
    load_airfoil_coords(path::String="")

Loads an airfoil shape from a 2-column text file (x, z), normalized to unit chord length.
If no file is specified, or if loading fails, returns a built-in 200-point Clark Y airfoil shape.

# Parameters
- `path`: Optional path to a text file with two columns: x and z coordinates.

# Returns
- `xy`: Nx2 matrix of normalized (x, y) coordinates representing the airfoil surface.

# Notes
- If loading from file, x-coordinates are normalized to span [0, 1].
- The default fallback airfoil is a symmetric approximation of the Clark Y shape.
- This airfoil is primarily used for visualization, not aerodynamic calculations.
"""
function load_airfoil_coords(afpath::String="")

    if !isempty(afpath) && isfile(afpath)
        xy = DelimitedFiles.readdlm(afpath,',',Float64,skipstart = 0)
        xy[:, 1] .-= minimum(xy[:, 1])
        xy[:, 1] ./= maximum(xy[:, 1])
        xy[:, 2] ./= (maximum(xy[:, 2])-minimum(xy[:, 2]))
        return xy
    else
        @warn "Could not load airfoil file used for plotting. Falling back to default Clark Y profile for plotting."

        # Fallback Clark Y airfoil coordinates (from airfoiltools.com)
        xy = [
        1.0000000 0.0
        0.9900000 0.0029690
        0.9800000 0.0053335
        0.9700000 0.0076868
        0.9600000 0.0100232
        0.9400000 0.0146239
        0.9200000 0.0191156
        0.9000000 0.0235025
        0.8800000 0.0277891
        0.8600000 0.0319740
        0.8400000 0.0360536
        0.8200000 0.0400245
        0.8000000 0.0438836
        0.7800000 0.0476281
        0.7600000 0.0512565
        0.7400000 0.0547675
        0.7200000 0.0581599
        0.7000000 0.0614329
        0.6800000 0.0645843
        0.6600000 0.0676046
        0.6400000 0.0704822
        0.6200000 0.0732055
        0.6000000 0.0757633
        0.5800000 0.0781451
        0.5600000 0.0803480
        0.5400000 0.0823712
        0.5200000 0.0842145
        0.5000000 0.0858772
        0.4800000 0.0873572
        0.4600000 0.0886427
        0.4400000 0.0897175
        0.4200000 0.0905657
        0.4000000 0.0911712
        0.3800000 0.0915212
        0.3600000 0.0916266
        0.3400000 0.0915079
        0.3200000 0.0911857
        0.3000000 0.0906804
        0.2800000 0.0900016
        0.2600000 0.0890840
        0.2400000 0.0878308
        0.2200000 0.0861433
        0.2000000 0.0839202
        0.1800000 0.0810687
        0.1600000 0.0775707
        0.1400000 0.0734360
        0.1200000 0.0686204
        0.1000000 0.0629981
        0.0800000 0.0564308
        0.0600000 0.0487571
        0.0500000 0.0442753
        0.0400000 0.0391283
        0.0300000 0.0330215
        0.0200000 0.0253735
        0.0120000 0.0178581
        0.0080000 0.0137350
        0.0040000 0.0089238
        0.0020000 0.0058025
        0.0010000 0.0037271
        0.0005000 0.0023390
        0.0000000 0.0000000	
        0.0005000 -.0046700	
        0.0010000 -.0059418	
        0.0020000 -.0078113	
        0.0040000 -.0105126	
        0.0080000 -.0142862	
        0.0120000 -.0169733	
        0.0200000 -.0202723	
        0.0300000 -.0226056	
        0.0400000 -.0245211	
        0.0500000 -.0260452	
        0.0600000 -.0271277	
        0.0800000 -.0284595	
        0.1000000 -.0293786	
        0.1200000 -.0299633	
        0.1400000 -.0302404	
        0.1600000 -.0302546	
        0.1800000 -.0300490	
        0.2000000 -.0296656	
        0.2200000 -.0291445	
        0.2400000 -.0285181	
        0.2600000 -.0278164	
        0.2800000 -.0270696	
        0.3000000 -.0263079	
        0.3200000 -.0255565	
        0.3400000 -.0248176	
        0.3600000 -.0240870	
        0.3800000 -.0233606	
        0.4000000 -.0226341	
        0.4200000 -.0219042	
        0.4400000 -.0211708	
        0.4600000 -.0204353	
        0.4800000 -.0196986	
        0.5000000 -.0189619	
        0.5200000 -.0182262	
        0.5400000 -.0174914	
        0.5600000 -.0167572	
        0.5800000 -.0160232	
        0.6000000 -.0152893	
        0.6200000 -.0145551	
        0.6400000 -.0138207	
        0.6600000 -.0130862	
        0.6800000 -.0123515	
        0.7000000 -.0116169	
        0.7200000 -.0108823	
        0.7400000 -.0101478	
        0.7600000 -.0094133	
        0.7800000 -.0086788	
        0.8000000 -.0079443	
        0.8200000 -.0072098	
        0.8400000 -.0064753	
        0.8600000 -.0057408	
        0.8800000 -.0050063	
        0.9000000 -.0042718	
        0.9200000 -.0035373	
        0.9400000 -.0028028	
        0.9600000 -.0020683	
        0.9700000 -.0017011	
        0.9800000 -.0013339	
        0.9900000 -.0009666	
        1.0 0
        ]
        xy[:, 1] .-= minimum(xy[:, 1])
        xy[:, 1] ./= maximum(xy[:, 1])
        xy[:, 2] ./= (maximum(xy[:, 2])-minimum(xy[:, 2]))
        return xy
    end
end

"""
    resample_airfoil(xy::Matrix{Float64}, npoints::Int=200) -> Matrix{Float64}

Resamples the given airfoil shape by:
1. Identifying leading (min x) and trailing (max x) edges.
2. Splitting into upper and lower surfaces.
3. Interpolating both surfaces using `npoints` uniformly spaced x-values.
4. Recombining to produce a smooth resampled airfoil shape.

# Inputs
- `xy`: Nx2 matrix of (x, y) airfoil coordinates, not assumed to start at TE or LE.
- `npoints`: Number of points used in interpolation (default: 200).

# Returns
- `xy_resampled`: Matrix{Float64}, size ≈ 2npoints x 2, resampled and recombined airfoil shape.
"""
function resample_airfoil(xy::Matrix{Float64}; npoints::Int=200)
    x = xy[:, 1]
    y = xy[:, 2]

    # Find leading edge as the point with minimum x
    le_idx = argmin(x)
    x_le = x[le_idx]

    # Split into upper and lower surfaces
    upper = xy[1:le_idx, :]
    lower = xy[le_idx:end, :]

    # Sort upper surface by descending x
    upper_sort_idx = sortperm(upper[:, 1], rev=false)
    upper_sorted = upper[upper_sort_idx, :]

    # Sort lower surface by ascending x
    lower_sort_idx = sortperm(lower[:, 1], rev=false)
    lower_sorted = lower[lower_sort_idx, :]

    x_upper = upper_sorted[:, 1]
    y_upper = upper_sorted[:, 2]
    x_lower = lower_sorted[:, 1]
    y_lower = lower_sorted[:, 2]

    # Create common x-grid
    x_min = maximum([minimum(x_upper), minimum(x_lower)])
    x_max = minimum([maximum(x_upper), maximum(x_lower)])
    x_resample = range(x_min, x_max; length=round(Int,npoints/2))

    # Interpolate both surfaces
    itp_upper = Interpolations.LinearInterpolation(x_upper, y_upper, extrapolation_bc=Interpolations.Line())
    itp_lower = Interpolations.LinearInterpolation(x_lower, y_lower, extrapolation_bc=Interpolations.Line())
    y_upper_resampled = itp_upper.(x_resample)
    y_lower_resampled = itp_lower.(x_resample)

    # Recombine: upper reversed to preserve typical TE-to-LE-to-TE convention
    x_combined = vcat(reverse(x_resample), x_resample)
    y_combined = vcat(reverse(y_upper_resampled), y_lower_resampled)
    return hcat(x_combined, y_combined)
end



"""
    interpolate_fft_spectrum(afft, Re_val, AOA_val, field)

Interpolates the FFT amplitude and phase spectra for a given Reynolds number `Re_val`
and angle of attack `AOA_val` using bilinear interpolation over the stored Re × AOA grid.

# Parameters
- `afft`: AirfoilFFT struct containing FFT results.
- `Re_val`: Desired Reynolds number.
- `AOA_val`: Desired angle of attack (degrees).
- `field`: Symbol (:CL, :CD, or :CM) indicating which force coefficient to interpolate.

# Returns
- `freqs`: Vector of frequency values.
- `amp_out`: Vector of interpolated amplitudes at each frequency.
- `phase_out`: Vector of interpolated phases at each frequency.

# Notes
- The interpolation is performed independently at each frequency index in the spectrum.
- Assumes consistent frequency axis across the full 3D data structure.
- Returns values suitable for reconstructing time-domain or frequency-domain force estimates.
"""
function interpolate_fft_spectrum(afft::AirfoilFFT, Re_val::Float64, AOA_val::Float64, field::Symbol;n_freq_depth=nothing)
    STs, amps, phases = if field == :CL
        afft.CL_ST, afft.CL_Amp, afft.CL_Pha
    elseif field == :CD
        afft.CD_ST, afft.CD_Amp, afft.CD_Pha
    elseif field == :CM
        afft.CM_ST, afft.CM_Amp, afft.CM_Pha
    elseif field == :CF
        afft.CF_ST, afft.CF_Amp, afft.CF_Pha
    else
        error("Invalid field symbol: $field")
    end

    if isnothing(n_freq_depth)
        n_freq_depth = length(afft.freqs)
    end

    ST_out = Vector{Float64}(undef, n_freq_depth)
    amp_out = Vector{Float64}(undef, n_freq_depth)
    phase_out = Vector{Float64}(undef, n_freq_depth)

    for k in 1:n_freq_depth
        st_itp = Interpolations.interpolate((afft.Re, afft.AOA), STs[:, :, k], Interpolations.Gridded(Interpolations.Linear()))
        st_itp = Interpolations.extrapolate(st_itp, Interpolations.Flat()) 
        amp_itp = Interpolations.interpolate((afft.Re, afft.AOA), amps[:, :, k], Interpolations.Gridded(Interpolations.Linear()))
        amp_itp = Interpolations.extrapolate(amp_itp, Interpolations.Flat()) 
        pha_itp = Interpolations.interpolate((afft.Re, afft.AOA), phases[:, :, k], Interpolations.Gridded(Interpolations.Linear()))
        pha_itp = Interpolations.extrapolate(pha_itp, Interpolations.Flat()) 
        ST_out[k] = st_itp(Re_val,AOA_val)
        amp_out[k] = amp_itp(Re_val,AOA_val)
        phase_out[k] = pha_itp(Re_val,AOA_val)
    end

    return ST_out, amp_out, phase_out
end


function write_force_time_series(filename::String, output_time::Vector{Float64}, global_force_vector_nodes::Array{Float64, 3})
    ntime, _, nnodes = size(global_force_vector_nodes)

    open(filename, "w") do io
        # --- Write header ---
        header = ["time"]
        for n in 1:nnodes
            push!(header, "node$(n)x")
            push!(header, "node$(n)y")
            push!(header, "node$(n)z")
        end
        println(io, join(header, ", "))

        # --- Write each time row ---
        for t in 1:ntime
            row = [output_time[t]]
            for n in 1:nnodes
                fx = global_force_vector_nodes[t, 1, n]
                fy = global_force_vector_nodes[t, 2, n]
                fz = global_force_vector_nodes[t, 3, n]
                append!(row, (fx, fy, fz))
            end
            println(io, join(row, ", "))
        end
    end
end