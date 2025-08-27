path = splitdir(@__FILE__)[1]
include("../VorLap.jl")
import .VorLap
import DelimitedFiles
import HDF5 #TODO remove this once debugging done
import Plots
using Test

Plots.plotlyjs()  # switch backend to PlotlyJS for interactive 3D plots

# === Top-Level Inputs ===
viv_params = VorLap.VIV_Params(
    fluid_density=1.225,
    fluid_dynamicviscosity=1.81e-5,
    rotation_axis=[0.0, 0.0, 1.0],
    rotation_axis_offset=[0.0, 0.0, 0.0],
    inflow_vec=[1.0, 0.0, 0.0],
    # plot_cycle=["#348ABD", "#A60628", "#009E73", "#7A68A6", "#D55E00", "#CC79A7"],
    azimuths=collect(0:5:255.0),
    inflow_speeds=collect(2.0:0.5:50.0),
    # freq_min=0.0,
    # freq_max=Inf,
    n_harmonic = 2,
    output_time = collect(0.0:0.001:0.01), #s
    output_azimuth_vinf = (5.0, 2.0), #used to limit the case where the relatively expensive output signal reconstruction is done
    amplitude_coeff_cutoff = 0.2,
    n_freq_depth = 10,
    airfoil_folder="$path/../airfoils/",
)

nodal_force_time_file = "$path/forces_output.csv"
nodal_force_time_file_unit = "$path/forces_output_unit.csv"

#TODO: single airfoil verification with test case
#TODO: translate for GUI

# First thing, we want to load a CSV that contains the parked natural frequencies 
natfreqs = DelimitedFiles.readdlm("$path/../natural_frequencies.csv",',',Float64,skipstart = 0)[:]

# upload a series CSVs that define each component's overall position and rotation, and the shape, chord, twist, thickness, and (optional) airfoil data used 
components = VorLap.load_components_from_csv("$path/../componentsHVAWT/")

# If an airfoil file is specified, read that in, otherwise use the default

affts = Dict{String, VorLap.AirfoilFFT}()
for file in filter(f -> endswith(f, ".h5"), readdir(viv_params.airfoil_folder, join=true))
    afft = VorLap.load_airfoil_fft(file)
    affts[afft.name] = afft
end

#TODO: implement placeholder general model, and warn if using it since the airfoil isn't known
if !haskey(affts, "default")
    affts["default"] = first(values(affts))
end

# assemble each component into a full structure, and we need the rotation axis
# plot the full structure surface with a generic airfoil shape
VorLap.calc_structure_vectors_andplot!(components, viv_params)

# Calculate the angle of attack relative to inflow for each azimuth angle about the rotation axis
# Also calculate the tilt in the direction of the inflow, which changes the perceived inflow velocity normal to the airfoil
# calculate the mean thrust and torque about the rotation axis for each azimuth angle to create a surface plot of inflow velocity vs azimuth vs value
percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector,global_force_vector_nodes = VorLap.compute_thrust_torque_spectrum(components,affts,viv_params,natfreqs)
VorLap.write_force_time_series(nodal_force_time_file, viv_params.output_time, global_force_vector_nodes)
# VorLap.write_force_time_series(nodal_force_time_file_unit, viv_params.output_time, global_force_vector_nodes)

data_logged = log10.(max.(percdiff_matrix, 1e-12))  # avoid log(0)
tick_vals = [1.0, 5.0, 20.0, 50.0, 100.0]  # example: 0.001% to 100%
tick_positions = log10.(tick_vals)  # positions in the log scale
plot_ = Plots.heatmap(
    viv_params.azimuths,
    viv_params.inflow_speeds,
    percdiff_matrix;
    clim = (0, 50), 
    xlabel = "Azimuth (deg)",
    ylabel = "Inflow (m/s)",
    title = "Worst Percent Difference",
    colorbar_title = "Freq % Diff",
    interpolate = true,
    c = Plots.cgrad(:viridis, rev=true),
    background_color=:transparent
)
Plots.display(plot_)
Plots.savefig(plot_, "$path/../figs/worst_percent_diff.pdf")

plot_ = Plots.heatmap(
    viv_params.azimuths,
    viv_params.inflow_speeds,
    total_global_force_vector[:,:,1];
    # clim = (0, 50), 
    xlabel = "Azimuth (deg)",
    ylabel = "Inflow (m/s)",
    title = "Fx",
    colorbar_title = "Freq % Diff",
    interpolate = true,
    c = Plots.cgrad(:viridis, rev=true),
    background_color=:transparent
)
Plots.display(plot_)
Plots.savefig(plot_, "$path/../figs/Fx.pdf")

plot_ = Plots.heatmap(
    viv_params.azimuths,
    viv_params.inflow_speeds,
    total_global_force_vector[:,:,2];
    # clim = (0, 50), 
    xlabel = "Azimuth (deg)",
    ylabel = "Inflow (m/s)",
    title = "Fy",
    colorbar_title = "Force (N)",
    interpolate = true,
    c = Plots.cgrad(:viridis, rev=true),
    background_color=:transparent
)
Plots.display(plot_)
Plots.savefig(plot_, "$path/../figs/Fy.pdf")

plot_ = Plots.heatmap(
    viv_params.azimuths,
    viv_params.inflow_speeds,
    total_global_force_vector[:,:,3];
    # clim = (0, 50), 
    xlabel = "Azimuth (deg)",
    ylabel = "Inflow (m/s)",
    title = "Fz",
    colorbar_title = "Force (N)",
    interpolate = true,
    c = Plots.cgrad(:viridis, rev=true),
    background_color=:transparent
)
Plots.display(plot_)
Plots.savefig(plot_, "$path/../figs/Fz.pdf")

plot_ = Plots.heatmap(
    viv_params.azimuths,
    viv_params.inflow_speeds,
    total_global_moment_vector[:,:,1];
    # clim = (0, 50), 
    xlabel = "Azimuth (deg)",
    ylabel = "Inflow (m/s)",
    title = "Mx",
    colorbar_title = "Moment (N-m)",
    interpolate = true,
    c = Plots.cgrad(:viridis, rev=true),
    background_color=:transparent
)
Plots.display(plot_)
Plots.savefig(plot_, "$path/../figs/Fx.pdf")

plot_ = Plots.heatmap(
    viv_params.azimuths,
    viv_params.inflow_speeds,
    total_global_moment_vector[:,:,2];
    # clim = (0, 50), 
    xlabel = "Azimuth (deg)",
    ylabel = "Inflow (m/s)",
    title = "My",
    colorbar_title = "Moment (N-m)",
    interpolate = true,
    c = Plots.cgrad(:viridis, rev=true),
    background_color=:transparent
)
Plots.display(plot_)
Plots.savefig(plot_, "$path/../figs/Fy.pdf")

plot_ = Plots.heatmap(
    viv_params.azimuths,
    viv_params.inflow_speeds,
    total_global_moment_vector[:,:,3];
    # clim = (0, 50), 
    xlabel = "Azimuth (deg)",
    ylabel = "Inflow (m/s)",
    title = "Mz",
    colorbar_title = "Moment (N-m)",
    interpolate = true,
    c = Plots.cgrad(:viridis, rev=true),
    background_color=:transparent
)
Plots.display(plot_)
Plots.savefig(plot_, "$path/../figs/Mz.pdf")

#test
# Load in the unit CSV file and the just saved CSV File, and verify they are the same
new_file = DelimitedFiles.readdlm(nodal_force_time_file,',',Float64,skipstart = 1)
unit_file = DelimitedFiles.readdlm("$path/forces_output_unit.csv",',',Float64,skipstart = 1)

for irow in 1:size(new_file)[1]
    for icol in 1:size(new_file)[2]
        @test isapprox(new_file[irow,icol],unit_file[irow,icol],atol=max(1e-6,abs(unit_file[irow,icol]).*1e-5))
    end
end
