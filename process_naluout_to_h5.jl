import HDF5, FFTW, DelimitedFiles, Printf, LinearAlgebra
import Statistics:mean

import Plots
Plots.plotlyjs()  # switch backend to PlotlyJS for interactive 3D plots
Plots.closeall()
const localpath = splitdir(@__FILE__)[1]


# function process_dat_folder(dat_folder::String, h5_path::String; 
#     Re::Float64=1e7,
#     Vinf=70.0,
#     chord=1.0,
#     span=4.0,
#     fluid_density=1.2,
#     fluid_viscosity=9.0e-06,
#     freq_vector=range(0, stop=50, length=100))
# dat_folder = "$localpath/airfoils/2024_Ganesh_VIV_Paper_Data/ffa_data_files_ftt_160/ffa_w3_211/data_files"
# dat_folder = "$localpath/airfoils/2024_Ganesh_VIV_Paper_Data/flat_plate_000/data_files"
dat_folder = "$localpath/airfoils/cylinder/data_files"
h5_path = "$localpath/airfoils/cylinder_fft.h5"

#TODO: add thickness?
Vinf=2.0
chord=1.0
span=4.0
fluid_density=1.2
fluid_viscosity=9.0e-06
NFreq = 200
minFreq = 0.0 #Hz frequency cutoff for Strouhaul number calculation
genplots = true

sampledT_startcutoff = 70.0

reynolds = fluid_density*Vinf*chord/fluid_viscosity


files = filter(f -> endswith(f, ".dat"), readdir(dat_folder; join=true))
N_AOA = length(files)

thickness = 0.0 #placeholder, is set elsewhere
airfoil_name = "placeholder"

# Figure out the number of AOAs
Re = [reynolds] #TODO: other reynolds?
NRe = length(Re) #TODO
single_Re=false
if NRe == 1
    Re = [Re[1],Re[1]+1e-6]
    NRe = 2
    single_Re=true
end
AOA = zeros(N_AOA)
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

iRe = 1 #TODO: other Reynolds
for (iaoa,file) in enumerate(files)
    # iaoa = 5
    # file = files[iaoa]
    
    println("Processing $file")
    filename = split(basename(file), ".")[1]
    AOA_str = replace(split(filename, "_")[end], r"[^\d\-]+" => "")
    AOA[iaoa] = parse(Float64, AOA_str)
    global airfoil_name = join(split(filename, "_")[1:end-1], "_")
    try
        thickness_str = replace(split(airfoil_name, "_")[end], r"[^\d\-]+" => "")
        global thickness = parse(Float64, thickness_str)./1000
    catch
        global thickness = 0.0
    end

    global NFreq

    # Read data (skip headers)
    data = DelimitedFiles.readdlm(file, skipstart=1)
    timefull = data[:, 1]
    dt = diff(timefull[1:2])[1] #NOTE: assumes that the CFD solver uses a fixed time step
    fpx, fpy = data[:, 2], data[:, 3]
    fvx, fvy = data[:, 5], data[:, 6]
    mty = data[:, 9]

    q = 0.5*fluid_density*Vinf^2*chord*span
    CL = (fpy .+ fvy) ./ q #pressure and viscous forces. #NOTE: mesh x is assumed to be in the same direction as the flow, +x
    CD = (fpx .+ fvx) ./ q 
    CF = sqrt.(CD.^2+CL.^2)
    CM = (mty) ./ q

    # # Calculate the number of flow-throughs, and cut out the first
    time_flowthrough = chord/Vinf #chord_m/m*s = second/chord
    total_flowthroughs = timefull[end]/time_flowthrough # seconds * chord/seconds = chords

    function compute_fft(signal::Vector{Float64}, dt::Float64,chord::Float64,aoa::Float64)
        N = length(signal)
        signal_used = signal # .- mean(signal) maximum difference in 2:end is 1.6e-16
        fft_vals = FFTW.fft(signal_used)
        half_N = div(N, 2)
        
        # One-sided frequency axis
        freqs = (0:half_N-1) ./ (N * dt)
        
        # Amplitude spectrum (one-sided)
        amps = abs.(fft_vals[1:half_N]) ./ N
        amps[2:end-1] .*= 2  # Double non-DC, non-Nyquist components

        # Phase spectrum
        phases = angle.(fft_vals[1:half_N])

        # Strouhaul number calculation
        # Sort by the largest to smallest amplitudes
        perm_peakamp = sortperm(amps;rev=true) #largest to smallest
        freqs_sorted = freqs[perm_peakamp]
        amps_sorted = amps[perm_peakamp]
        phases_sorted = phases[perm_peakamp]
        STlength = chord*abs(sind(aoa))# NOTE: ST seems to be dominated by the major axis length, not planform length+thickness*chord*abs(cosd(AOA[iaoa]))
        ST_sorted = freqs_sorted * STlength / Vinf
        # st = freqs * STlength / Vinf
        # freq = st*Vinf/length
        
        return collect(freqs), amps, phases, ST_sorted, amps_sorted, phases_sorted
    end


    freqs_cl, amps_cl, phases_cl, ST_sorted_cl, amps_sorted_cl, phases_sorted_cl = compute_fft(CL,dt,chord,AOA[iaoa])
    freqs_cd, amps_cd, phases_cd, ST_sorted_cd, amps_sorted_cd, phases_sorted_cd = compute_fft(CD,dt,chord,AOA[iaoa])
    freqs_cm, amps_cm, phases_cm, ST_sorted_cm, amps_sorted_cm, phases_sorted_cm = compute_fft(CM,dt,chord,AOA[iaoa])
    freqs_cf, amps_cf, phases_cf, ST_sorted_cf, amps_sorted_cf, phases_sorted_cf = compute_fft(CF,dt,chord,AOA[iaoa])

    # testsignal = reconstruct_signal(freqs_cf,amps_cf,phases_cf,timefull)

    if genplots
        idx_start = max(1,round(Int,sampledT_startcutoff/dt))

        timeplot = Plots.plot()
        # Plots.plot!(timeplot,timefull[idx_start:end],testsignal[idx_start:end])

        Plots.plot!(timeplot,timefull[idx_start:end],CF[idx_start:end],
        xlabel = "Time (s)",
        ylabel = "CF",
        legend = true,
        title = "(CF), AOA: $(AOA[iaoa])",
        lw = 2)
        Plots.display(timeplot)


        cfbode = Plots.plot(freqs_cd[2:NFreq], amps_cf[2:NFreq],
        #  xscale = :log10,
        xlabel = "Frequency (Hz)",
        ylabel = "Magnitude (CF)",
        title = "(CF), AOA: $(AOA[iaoa])",
        label = "Original",
        # xlim = (0,100),
        marker = :cross,
        legend = false,
        lw = 2)
        Plots.display(cfbode)

        #  sleep(0.5)
    end

    # === Store Contents From File, Only Save Frequency up to  ===
    if NFreq>length(ST_sorted_cl)
        @warn "Number of frequencies exceeds those available"
        NFreq=length(ST_sorted_cl)
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
# Sort the AOA vector and its dependencies
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

if genplots
    plot_ = Plots.plot()
    plot_2 = Plots.plot()
    for ist = 2:5:22#:5:50
        Plots.plot!(plot_,AOA_sort, CF_ST_sort[1,:,ist],
        #  xscale = :log10,
        xlabel = "AOA (deg)",
        ylabel = "ST (CF)",
        title = "ST",
        label = "$ist",
        # xlim = (0,100),
        marker = :cross,
        legend = true,
        lw = 2)

        Plots.plot!(plot_2,AOA_sort, CF_Amp_sort[1,:,ist],
        #  xscale = :log10,
        xlabel = "AOA (Hz)",
        ylabel = "Amp (CF)",
        title = "Amp",
        label = "$ist",
        # xlim = (0,100),
        marker = :cross,
        legend = true,
        lw = 2)

    end
    Plots.display(plot_2)
    Plots.display(plot_)

    #  sleep(0.5)
end

HDF5.h5open(h5_path, "w") do file
    HDF5.write(file,"Airfoilname", airfoil_name)
    HDF5.write(file,"Re", Re)
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
