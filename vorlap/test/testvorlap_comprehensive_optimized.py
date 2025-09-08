import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import vorlap.graphics

# Add the parent directory to the path so we can import vorlap
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import vorlap

# Get path (equivalent to Julia: path = splitdir(@__FILE__)[1])
path = os.path.dirname(os.path.abspath(__file__))

# === Top-Level Inputs ===
viv_params = vorlap.VIV_Params(
    fluid_density=1.225,
    fluid_dynamicviscosity=1.81e-5,
    rotation_axis=np.array([0.0, 0.0, 1.0]),
    rotation_axis_offset=np.array([0.0, 0.0, 0.0]),
    inflow_vec=np.array([1.0, 0.0, 0.0]),
    azimuths=np.arange(10, 360, 30),  # collect(0:5:255.0)
    inflow_speeds=np.arange(2.0, 50.0, 10),  # collect(2.0:0.5:50.0)
    n_harmonic=2,
    output_time=np.arange(0.0, 0.011, 0.001),  # collect(0.0:0.001:0.01)
    output_azimuth_vinf=(10.0, 42.0),
    amplitude_coeff_cutoff=0.2,
    n_freq_depth=10,
    airfoil_folder=f"{path}/../../airfoils/"
)

nodal_force_time_file = f"{path}/forces_output_optimized.csv"

# First thing, we want to load a CSV that contains the parked natural frequencies 
natfreqs = np.loadtxt(f"{path}/../../natural_frequencies.csv", delimiter=',')

# upload a series CSVs that define each component's overall position and rotation, and the shape, chord, twist, thickness, and (optional) airfoil data used 
components = vorlap.load_components_from_csv(f"{path}/../../componentsHVAWT/")

# If an airfoil file is specified, read that in, otherwise use the default
affts = {}
airfoil_folder = viv_params.airfoil_folder

import glob
for file in glob.glob(os.path.join(airfoil_folder, "*.h5")):
    afft = vorlap.load_airfoil_fft(file)
    affts[afft.name] = afft

# TODO: implement placeholder general model, and warn if using it since the airfoil isn't known
if "default" not in affts:
    affts["default"] = next(iter(affts.values()))

# assemble each component into a full structure, and we need the rotation axis
# plot the full structure surface with a generic airfoil shape
vorlap.graphics.calc_structure_vectors_andplot(components, viv_params)

# Calculate the angle of attack relative to inflow for each azimuth angle about the rotation axis
# Also calculate the tilt in the direction of the inflow, which changes the perceived inflow velocity normal to the airfoil
# calculate the mean thrust and torque about the rotation axis for each azimuth angle to create a surface plot of inflow velocity vs azimuth vs value
import time

print("Running OPTIMIZED compute_thrust_torque_spectrum...")
start_time = time.time()

percdiff_matrix, percdiff_info, total_global_force_vector, total_global_moment_vector, global_force_vector_nodes = vorlap.compute_thrust_torque_spectrum_optimized(components, affts, viv_params, natfreqs)
end_time = time.time()
execution_time = end_time - start_time
print(f"OPTIMIZED compute_thrust_torque_spectrum execution time: {execution_time:.4f} seconds")

vorlap.write_force_time_series(nodal_force_time_file, viv_params.output_time, global_force_vector_nodes)

data_logged = np.log10(np.maximum(percdiff_matrix, 1e-12))  # avoid log(0)
tick_vals = [1.0, 5.0, 20.0, 50.0, 100.0]  # example: 0.001% to 100%
tick_positions = np.log10(tick_vals)  # positions in the log scale

# Create figures directory
figs_dir = f"{path}/../pyfigs_optimized"
os.makedirs(figs_dir, exist_ok=True)

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(percdiff_matrix, 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r', vmin=0, vmax=50)
plt.colorbar(label='Freq % Diff')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Worst Percent Difference (Optimized)')
plt.savefig(f"{figs_dir}/worst_percent_diff_optimized.pdf", bbox_inches='tight')
plt.show()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_force_vector[:, :, 0], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Force (N)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Fx (Optimized)')
plt.savefig(f"{figs_dir}/Fx_optimized.pdf", bbox_inches='tight')
plt.show()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_force_vector[:, :, 1], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Force (N)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Fy (Optimized)')
plt.savefig(f"{figs_dir}/Fy_optimized.pdf", bbox_inches='tight')
plt.show()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_force_vector[:, :, 2], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Force (N)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Fz (Optimized)')
plt.savefig(f"{figs_dir}/Fz_optimized.pdf", bbox_inches='tight')
plt.show()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_moment_vector[:, :, 0], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Moment (N-m)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Mx (Optimized)')
plt.savefig(f"{figs_dir}/Mx_optimized.pdf", bbox_inches='tight')
plt.show()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_moment_vector[:, :, 1], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Moment (N-m)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('My (Optimized)')
plt.savefig(f"{figs_dir}/My_optimized.pdf", bbox_inches='tight')
plt.show()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_moment_vector[:, :, 2], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Moment (N-m)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Mz (Optimized)')
plt.savefig(f"{figs_dir}/Mz_optimized.pdf", bbox_inches='tight')
plt.show()

# Compare with unit test file if available
print("\nComparing optimized results with unit test...")

new_file = np.loadtxt(nodal_force_time_file, delimiter=',', skiprows=1)
unit_file = np.loadtxt(f"{path}/forces_output_unit.csv", delimiter=',', skiprows=1)

for irow in range(new_file.shape[0]):
    for icol in range(new_file.shape[1]):
        # Python equivalent of Julia's @test isapprox
        tolerance = max(1e-6, abs(unit_file[irow, icol]) * 1e-5)
        assert np.isclose(new_file[irow, icol], unit_file[irow, icol], atol=tolerance), \
            f"Mismatch at [{irow}, {icol}]: {new_file[irow, icol]} vs {unit_file[irow, icol]}"

