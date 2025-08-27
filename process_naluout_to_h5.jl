import HDF5, FFTW, DelimitedFiles, Printf, LinearAlgebra
import Statistics:mean
using DSP: hann
using FFTW

import Plots
Plots.plotlyjs()  # switch backend to PlotlyJS for interactive 3D plots
Plots.closeall()
const localpath = splitdir(@__FILE__)[1]
include("$localpath/VorLap.jl")

# dat_folder now points to the AIRFOIL directory that contains RE* subfolders.
dat_folder = "$localpath/airfoils/NACA0018/"          # (unchanged var name, but used as root)  ### CHANGED
h5_path     = "$localpath/airfoils/NACA0018_fft.h5"

# --- constants / knobs (same names as before) ---
Vinf_fallback = 2.0   # used only if no RE subfolders are found                     ### NEW
chord = 1.0
span  = 4.0
fluid_density   = 1.2
fluid_viscosity = 9.0e-06
NFreq   = 200
minFreq = 0.0
genplots = true
sampledT_startcutoff = 10.3

# === discover RE subfolders and parse Re from names ===                               ### NEW
# Accepts names like RE5_00E5 or RE4_00E4.
# Per your request/example, "RE4_00E4" is interpreted as 4.00E5 (×10 over the literal E4).
function parse_re_from_dirname(name::AbstractString)::Float64
    m = match(r"^RE(\d+)_([0-9]+)E(\d+)$", name)
    m === nothing && return NaN
    base = m.captures[1]
    frac = m.captures[2]
    exp  = parse(Int, m.captures[3])
    val = parse(Float64, string(base, ".", frac, "E", exp))
    return val
end

all_entries = readdir(dat_folder; join=true)
re_dirs = filter(p -> isdir(p) && occursin(r"^RE\d+_\d+E\d+$", basename(p)), all_entries)

# If no RE* subfolders, fall back to treating dat_folder as a single-Re directory
if isempty(re_dirs)
    @info "No RE* subfolders found; using $dat_folder as a single-Re directory."
    re_dirs = [dat_folder]
end

Re = [parse_re_from_dirname(basename(d)) for d in re_dirs]
# For non-matching names (NaN), use the legacy single Vinf fallback
for i in eachindex(Re)
    if isnan(Re[i])
        Re[i] = fluid_density*Vinf_fallback*chord/fluid_viscosity
        @warn "Folder $(basename(re_dirs[i])) didn't match RE* pattern; using fallback Re=$(Re[i])"
    end
end

# Helper: list .dat files in a directory (direct or in ./data_files)                   ### NEW
function dat_files_in(dir::AbstractString)
    fs = filter(f -> endswith(f, ".dat"), readdir(dir; join=true))
    if isempty(fs)
        d2 = joinpath(dir, "data_files")
        if isdir(d2)
            fs = filter(f -> endswith(f, ".dat"), readdir(d2; join=true))
        end
    end
    return sort(fs)
end

# Use the first RE folder to define the AOA/file set (and order)
base_files = dat_files_in(re_dirs[1])
N_AOA = length(base_files)
if N_AOA == 0
    error("No .dat files found in $(re_dirs[1]) (or its data_files subdir).")
end

# --- allocate outputs (now truly multi-Re) ---
NRe = length(Re)
single_Re = (NRe == 1)

AOA = zeros(Float64, N_AOA)
thickness = 0.0 #placeholder, is set elsewhere
airfoil_name = "placeholder"

CL_ST = zeros(NRe,N_AOA,NFreq)
CD_ST = zeros(NRe,N_AOA,NFreq)
CM_ST = zeros(NRe,N_AOA,NFreq)
CF_ST = zeros(NRe,N_AOA,NFreq)
CL_Amp = zeros(NRe,N_AOA,NFreq)
CD_Amp = zeros(NRe,N_AOA,NFreq)
CM_Amp = zeros(NRe,N_AOA,NFreq)
CF_Amp = zeros(NRe,N_AOA,NFreq)
CL_Pha = zeros(NRe,N_AOA,NFreq)
CD_Pha = zeros(NRe,N_AOA,NFreq)
CM_Pha = zeros(NRe,N_AOA,NFreq)
CF_Pha = zeros(NRe,N_AOA,NFreq)

# === main loops: over Re folders, then AOA files (aligned by basename) ===            ### NEW
for (iRe, re_dir) in enumerate(re_dirs)
    # Inflow velocity from Re, chord, density, viscosity
    Vinf = Re[iRe] * fluid_viscosity / (fluid_density * chord)

    # Map basenames -> full paths so we can align with base_files ordering
    files_i = dat_files_in(re_dir)
    filemap  = Dict(basename(f) => f for f in files_i)

    for (iaoa, f0) in enumerate(base_files)
        b0 = basename(f0)
        file = get(filemap, b0, nothing)
        if file === nothing
            @warn "AOA file '$b0' not found in $(re_dir); skipping this (iRe=$iRe, iaoa=$iaoa)."
            continue
        end

        println("Processing (Re idx=$iRe) $(file)")
        filename = split(basename(file), ".")[1]
        AOA_str = replace(split(filename, "_")[end], r"[^\d\-]+" => "")

        if iRe == 1
            AOA[iaoa] = parse(Float64, AOA_str)
            global airfoil_name = join(split(filename, "_")[1:end-1], "_")
            try
                thickness_str = replace(split(airfoil_name, "_")[end], r"[^\d\-]+" => "")
                global thickness = parse(Float64, thickness_str)./1000
            catch
                global thickness = 0.0
            end
        else
            # light consistency check
            aoa_chk = try parse(Float64, AOA_str) catch; NaN end
            if !isnan(aoa_chk) && abs(aoa_chk - AOA[iaoa]) > 1e-9
                @warn "AOA mismatch at iaoa=$iaoa between base set ($(AOA[iaoa])) and $(re_dir) ($aoa_chk)."
            end
        end

        global NFreq

        # Read data (skip headers)
        data = []
        try
            data = DelimitedFiles.readdlm(file, skipstart=1)
        catch
            @warn "Skipping $(file), doesn't appear to be readable or doesn't contain data"
            continue
        end
        timefull = data[:, 1]
        if length(timefull) < 6001
            @warn "Skipping $(file): timefull length $(length(timefull)) < 6001"
            continue
        end
        dt = diff(timefull[1:2])[1] # assumes fixed time step
        fpx, fpy = data[:, 2], data[:, 3]
        fvx, fvy = data[:, 5], data[:, 6]
        mty      = data[:, 9]

        q = 0.5*fluid_density*Vinf^2*chord*span
        CL = (fpy .+ fvy) ./ q # +x is inflow direction
        CD = (fpx .+ fvx) ./ q
        CF = sqrt.(CD.^2+CL.^2)
        CM = (mty) ./ q

        # function uses Vinf from outer scope (per-Re)
        function compute_fft(signal::Vector{Float64}, dt::Float64, chord::Float64, aoa::Float64)
            N  = length(signal)
            fs = 1/dt
            half_N = div(N, 2)

            # --- DC and demean for spectral parts ---
            mean_amp = mean(signal)
            x  = signal .- mean_amp

            # --- Windowed path (ONLY for PSD + power sorting) ---
            w  = hann(N)
            U  = sum(w.^2) / N
            xw = w .* x
            Xw = FFTW.fft(xw)

            freqs = (0:half_N-1) ./ (N * dt)

            # One-sided PSD (power/Hz) from WINDOWED spectrum
            Xpos_w = Xw[1:half_N]
            S = (abs.(Xpos_w).^2) ./ (fs * N * U)
            if half_N > 2
                S[2:end-1] .*= 2
            end
            df = fs / N
            power = S .* df                      # per-bin power for sorting

            # --- Unwindowed path (for reconstruction amplitudes & phases) ---
            X = FFTW.fft(x)                      # FFT of demeaned, UNwindowed signal
            Xpos = X[1:half_N]

            # Amplitude scaling like your original (peak amplitude per cosine)
            amps = abs.(Xpos) ./ N
            if half_N > 2
                amps[2:end-1] .*= 2              # one-sided doubling (no DC/Nyquist)
            end

            # Phases from UNwindowed FFT
            phases = angle.(Xpos)

            # Enforce your DC convention: mean at 0 Hz, phase 0
            amps[1]   = mean_amp
            phases[1] = 0.0

            # --- Sort all except DC by POWER (from windowed PSD) ---
            if length(power) > 1
                perm_peakpow = sortperm(power[2:end]; rev=true) .+ 1  # skip DC bin
                freqs_sorted  = vcat(0.0, freqs[perm_peakpow])
                amps_sorted   = vcat(mean_amp, amps[perm_peakpow])
                phases_sorted = vcat(0.0, phases[perm_peakpow])
            else
                freqs_sorted  = [0.0]
                amps_sorted   = [mean_amp]
                phases_sorted = [0.0]
            end

            STlength   = chord*abs(sind(aoa))
            ST_sorted  = freqs_sorted * STlength / Vinf

            # Return: unsorted FFT bins (for reconstruction) + sorted-by-power view
            return collect(freqs), amps, phases, power, ST_sorted, amps_sorted, phases_sorted
        end

        freqs_cl, amps_cl, phases_cl, power_cl, ST_sorted_cl, amps_sorted_cl, phases_sorted_cl = compute_fft(CL,dt,chord,AOA[iaoa])
        freqs_cd, amps_cd, phases_cd, power_cd, ST_sorted_cd, amps_sorted_cd, phases_sorted_cd = compute_fft(CD,dt,chord,AOA[iaoa])
        freqs_cm, amps_cm, phases_cm, power_cm, ST_sorted_cm, amps_sorted_cm, phases_sorted_cm = compute_fft(CM,dt,chord,AOA[iaoa])
        freqs_cf, amps_cf, phases_cf, power_cf, ST_sorted_cf, amps_sorted_cf, phases_sorted_cf = compute_fft(CF,dt,chord,AOA[iaoa])

        if genplots

            idx_start = max(1, round(Int, sampledT_startcutoff/dt))
            timeplot = Plots.plot()
            # signal = VorLap.reconstruct_signal(freqs_cf, amps_cf, phases_cf, timefull)
            # Plots.plot!(timeplot, timefull[idx_start:end], signal[idx_start:end])
            Plots.plot!(timeplot, timefull[idx_start:end], CF[idx_start:end],
                xlabel="Time (s)", ylabel="CF", legend=true,
                title="(CF), AOA: $(AOA[iaoa]) (Re=$(Re[iRe]))", lw=2)
            Plots.display(timeplot)

            timeplot = Plots.plot(timefull[idx_start:end], CL[idx_start:end],
                xlabel="Time (s)", ylabel="CL", legend=true,
                title="(CL), AOA: $(AOA[iaoa]) (Re=$(Re[iRe]))", lw=2)
            Plots.display(timeplot)

            timeplot = Plots.plot(timefull[idx_start:end], CD[idx_start:end],
                xlabel="Time (s)", ylabel="CD", legend=true,
                title="(CD), AOA: $(AOA[iaoa]) (Re=$(Re[iRe]))", lw=2)
            Plots.display(timeplot)

            cfbode = Plots.plot(freqs_cf[2:NFreq], power_cf[2:NFreq],
                xlabel="Frequency (Hz)", ylabel="PSD",
                title="(CF), AOA: $(AOA[iaoa]) (Re=$(Re[iRe]))",
                label="Original", marker=:cross, legend=false, lw=2)
            Plots.display(cfbode)

            cfbode = Plots.plot(freqs_cl[2:NFreq], power_cl[2:NFreq],
                xlabel="Frequency (Hz)", ylabel="PSD",
                title="(CL), AOA: $(AOA[iaoa]) (Re=$(Re[iRe]))",
                label="Original", marker=:cross, legend=false, lw=2)
            Plots.display(cfbode)

            cfbode = Plots.plot(freqs_cd[2:NFreq], power_cd[2:NFreq],
                xlabel="Frequency (Hz)", ylabel="PSD",
                title="(CD), AOA: $(AOA[iaoa]) (Re=$(Re[iRe]))",
                label="Original", marker=:cross, legend=false, lw=2)
            Plots.display(cfbode)

        end

        # === Store (cap by available length) ===
        if NFreq > length(ST_sorted_cl)
            @warn "Number of frequencies ($NFreq) exceeds available ($(length(ST_sorted_cl)))"
            global NFreq = length(ST_sorted_cl)
        end
        CL_ST[iRe,iaoa,1:NFreq] = ST_sorted_cl[1:NFreq]
        CD_ST[iRe,iaoa,1:NFreq] = ST_sorted_cd[1:NFreq]
        CM_ST[iRe,iaoa,1:NFreq] = ST_sorted_cm[1:NFreq]
        CF_ST[iRe,iaoa,1:NFreq] = ST_sorted_cf[1:NFreq]
        CL_Amp[iRe,iaoa,1:NFreq] = amps_sorted_cl[1:NFreq]
        CD_Amp[iRe,iaoa,1:NFreq] = amps_sorted_cd[1:NFreq]
        CM_Amp[iRe,iaoa,1:NFreq] = amps_sorted_cm[1:NFreq]
        CF_Amp[iRe,iaoa,1:NFreq] = amps_sorted_cf[1:NFreq]
        CL_Pha[iRe,iaoa,1:NFreq] = phases_sorted_cl[1:NFreq]
        CD_Pha[iRe,iaoa,1:NFreq] = phases_sorted_cd[1:NFreq]
        CM_Pha[iRe,iaoa,1:NFreq] = phases_sorted_cm[1:NFreq]
        CF_Pha[iRe,iaoa,1:NFreq] = phases_sorted_cf[1:NFreq]
    end
end

# If only one Re, duplicate second slice to satisfy downstream expectations
if single_Re
    CL_ST[2,:,:] = CL_ST[1,:,:]
    CD_ST[2,:,:] = CD_ST[1,:,:]
    CM_ST[2,:,:] = CM_ST[1,:,:]
    CF_ST[2,:,:] = CF_ST[1,:,:]
    CL_Amp[2,:,:] = CL_Amp[1,:,:]
    CD_Amp[2,:,:] = CD_Amp[1,:,:]
    CM_Amp[2,:,:] = CM_Amp[1,:,:]
    CF_Amp[2,:,:] = CF_Amp[1,:,:]
    CL_Pha[2,:,:] = CL_Pha[1,:,:]
    CD_Pha[2,:,:] = CD_Pha[1,:,:]
    CM_Pha[2,:,:] = CM_Pha[1,:,:]
    CF_Pha[2,:,:] = CF_Pha[1,:,:]
end

# Sort by AOA (shared across Re)
aoa_sort_idx = sortperm(AOA)
AOA_sort = AOA[aoa_sort_idx]
CL_ST_sort = CL_ST[:,aoa_sort_idx,:]
CD_ST_sort = CD_ST[:,aoa_sort_idx,:]
CM_ST_sort = CM_ST[:,aoa_sort_idx,:]
CF_ST_sort = CF_ST[:,aoa_sort_idx,:]
CL_Amp_sort = CL_Amp[:,aoa_sort_idx,:]
CD_Amp_sort = CD_Amp[:,aoa_sort_idx,:]
CM_Amp_sort = CM_Amp[:,aoa_sort_idx,:]
CF_Amp_sort = CF_Amp[:,aoa_sort_idx,:]
CL_Pha_sort = CL_Pha[:,aoa_sort_idx,:]
CD_Pha_sort = CD_Pha[:,aoa_sort_idx,:]
CM_Pha_sort = CM_Pha[:,aoa_sort_idx,:]
CF_Pha_sort = CF_Pha[:,aoa_sort_idx,:]

# === Sort by Reynolds number (ascending) ===
re_sort_idx = sortperm(Re)                 # numeric ascending
Re_sort = Re[re_sort_idx]

CL_ST_sort = CL_ST_sort[re_sort_idx, :, :]
CD_ST_sort = CD_ST_sort[re_sort_idx, :, :]
CM_ST_sort = CM_ST_sort[re_sort_idx, :, :]
CF_ST_sort = CF_ST_sort[re_sort_idx, :, :]

CL_Amp_sort = CL_Amp_sort[re_sort_idx, :, :]
CD_Amp_sort = CD_Amp_sort[re_sort_idx, :, :]
CM_Amp_sort = CM_Amp_sort[re_sort_idx, :, :]
CF_Amp_sort = CF_Amp_sort[re_sort_idx, :, :]

CL_Pha_sort = CL_Pha_sort[re_sort_idx, :, :]
CD_Pha_sort = CD_Pha_sort[re_sort_idx, :, :]
CM_Pha_sort = CM_Pha_sort[re_sort_idx, :, :]
CF_Pha_sort = CF_Pha_sort[re_sort_idx, :, :]

if genplots


    ire_indices = single_Re ? [1] : collect(1:size(CF_ST_sort, 1))
    for iRe in ire_indices
        plot_ = Plots.plot()
        plot_2 = Plots.plot()
        for ist = 2:1:20
            Plots.plot!(plot_, AOA_sort, CF_ST_sort[iRe,:,ist],
                xlabel="AOA (deg)", ylabel="ST (CF)", title="ST Re=$(round(Re_sort[iRe]; sigdigits=5))",
                label="ist=$ist",
                ylims=(0.0,0.5),
                marker=:cross, legend=true, lw=2)

            Plots.plot!(plot_2, AOA_sort, CF_Amp_sort[iRe,:,ist],
                xlabel="AOA (Hz)", ylabel="Amp (CF)", title="Amp Re=$(round(Re_sort[iRe]; sigdigits=5))",
                label="ist=$ist",
                marker=:cross, legend=true, lw=2)
        end
        Plots.display(plot_2)
         Plots.display(plot_)
    end

end

HDF5.h5open(h5_path, "w") do file
    HDF5.write(file,"Airfoilname", airfoil_name)
    HDF5.write(file,"Re", Re_sort)
    HDF5.write(file,"Thickness",thickness)
    HDF5.write(file,"AOA", AOA_sort)
    HDF5.write(file,"CL_ST", CL_ST_sort)
    HDF5.write(file,"CD_ST", CD_ST_sort)
    HDF5.write(file,"CM_ST", CM_ST_sort)
    HDF5.write(file,"CF_ST", CF_ST_sort)
    HDF5.write(file,"CL_Amp", CL_Amp_sort)
    HDF5.write(file,"CD_Amp", CD_Amp_sort)
    HDF5.write(file,"CM_Amp", CM_Amp_sort)
    HDF5.write(file,"CF_Amp", CF_Amp_sort)
    HDF5.write(file,"CL_Pha", CL_Pha_sort)
    HDF5.write(file,"CD_Pha", CD_Pha_sort)
    HDF5.write(file,"CM_Pha", CM_Pha_sort)
    HDF5.write(file,"CF_Pha", CF_Pha_sort)
end
