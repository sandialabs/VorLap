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
    azimuths=np.arange(4, 360, 8),  # collect(0:5:255.0)
    inflow_speeds=np.array([7.386,7.3878,7.388]),#np.arange(2.0, 50.0, 4.0),  # collect(2.0:0.5:50.0)
    n_harmonic=2,
    output_time=np.arange(0.0, 0.011, 0.001),  # collect(0.0:0.001:0.01)
    output_azimuth_vinf=(10.0, 6.0),
    amplitude_coeff_cutoff=0.002,
    n_freq_depth=10,
    airfoil_folder=f"{path}/../data/airfoils/"
)

nodal_force_time_file = f"{path}/forces_output_single_blade.csv"

# First thing, we want to load a CSV that contains the parked natural frequencies 
natfreqs = np.loadtxt(f"{path}/../data/natural_frequencies.csv", delimiter=',')

# upload a series CSVs that define each component's overall position and rotation, and the shape, chord, twist, thickness, and (optional) airfoil data used 
components = vorlap.load_components_from_csv(f"{path}/../data/components/componentsSingle/")

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
figs_dir = f"{path}/../pyfigs_single"
os.makedirs(figs_dir, exist_ok=True)

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(percdiff_matrix, 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r', vmin=0, vmax=50)
plt.colorbar(label='Freq % Diff')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Worst Percent Difference')
plt.savefig(f"{figs_dir}/worst_percent_diff_single_blade.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_force_vector[:, :, 0], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Force (N)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Fx')
plt.savefig(f"{figs_dir}/Fx_single_blade.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_force_vector[:, :, 1], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Force (N)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Fy')
plt.savefig(f"{figs_dir}/Fy_single_blade.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_force_vector[:, :, 2], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Force (N)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Fz')
plt.savefig(f"{figs_dir}/Fz_single_blade.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_moment_vector[:, :, 0], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Moment (N-m)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Mx')
plt.savefig(f"{figs_dir}/Mx_single_blade.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_moment_vector[:, :, 1], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Moment (N-m)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('My')
plt.savefig(f"{figs_dir}/My_single_blade.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(10, 8))
plt.imshow(total_global_moment_vector[:, :, 2], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Moment (N-m)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Mz')
plt.savefig(f"{figs_dir}/Mz_single_blade.pdf", bbox_inches='tight')
plt.close()

# Compare with unit test file if available
print("\nComparing results with unit test...")

# AOA -4
cl_4 = -0.402
cd_4 = 0.024
q = 0.5*1.225*7.3878**2*1.0*80
lift = q*cl_4
drag = q*cd_4

aoa = viv_params.azimuths[0]
# print(aoa)
fx = total_global_force_vector[1, 0, 0] #inflow, azi, dof
fy = total_global_force_vector[1, 0, 1]

tolerance = max(1e-2, abs(drag) * 0.05)
# print(f"aoa -4 Mismatch at drag vs fx: {drag} vs {fx} {(drag-fx)/drag*100}% diff")
assert np.isclose(fx, drag, atol=tolerance), \
            f"aoa -4 Mismatch at drag vs fx: {drag} vs {fx}"

tolerance = max(1e-2, abs(lift) * 0.05)
# print(f"aoa -4 Mismatch at lift vs fy: {lift} vs {fy} {(lift-fy)/lift*100}% diff")
assert np.isclose(fy, lift, atol=tolerance), \
            f"aoa -4 Mismatch at lift vs fy: {lift} vs {fy}"

# AOA -52
cl_52 = -1.5
cd_52 = 1.98
q = 0.5*1.225*7.3878**2*1.0*80
lift = q*cl_52
drag = q*cd_52
aoa = viv_params.azimuths[6]
# print(aoa)
fx = total_global_force_vector[1, 6, 0] #azi, inflow, dof
fy = total_global_force_vector[1, 6, 1]

tolerance = max(1e-2, abs(drag) * 0.05)
# print(f"aoa -52 Mismatch at drag vs fx: {drag} vs {fx} {(drag-fx)/drag*100}% diff")
assert np.isclose(fx, drag, atol=tolerance), \
            f"aoa -52 Mismatch at drag vs fx: {drag} vs {fx}"

tolerance = max(1e-2, abs(lift) * 0.05)
# print(f"aoa -52 Mismatch at lift vs fy: {lift} vs {fy} {(lift-fy)/lift*100}% diff")
assert np.isclose(fy, lift, atol=tolerance), \
            f"aoa -52 Mismatch at lift vs fy: {lift} vs {fy}"


