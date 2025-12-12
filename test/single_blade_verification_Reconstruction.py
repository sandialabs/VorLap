import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib as mpl
plot_cycle = ["#348ABD", "#A60628", "#009E73", "#7A68A6", "#D55E00", "#CC79A7"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=plot_cycle)

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
    azimuths=np.arange(0, 360, 24),  # collect(0:5:255.0)
    inflow_speeds=np.arange(0.0, 80.0, 5.0),  # collect(2.0:0.5:50.0)
    n_harmonic=1,
    output_time=np.arange(0.0, 2.0, 0.01),  # collect(0.0:0.001:0.01)
    output_azimuth_vinf=(216.0, 75.0),
    amplitude_coeff_cutoff=0.2,
    n_freq_depth=20,
    airfoil_folder=f"{path}/../data/airfoils/"
)

nodal_force_time_file = f"{path}/forces_output_single_blade_120deg_6mps_Reconstruction.csv"

# First thing, we want to load a CSV that contains the parked natural frequencies 
natfreqs = np.loadtxt(f"{path}/../data/natural_frequencies_Reconstruction.csv", delimiter=',')

# upload a series CSVs that define each component's overall position and rotation, and the shape, chord, twist, thickness, and (optional) airfoil data used 
components = vorlap.load_components_from_csv(f"{path}/../data/components/componentsSingle_Reconstruction/")

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
figs_dir = f"{path}/../pyfigs_single_Reconstruction"
os.makedirs(figs_dir, exist_ok=True)

plot_ = plt.figure(figsize=(4.5,3))
plt.imshow(percdiff_matrix, 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r', vmin=0, vmax=50)
plt.colorbar(label='Percent Difference in Frequencies')
plt.xlabel('Azimuth (deg)')
plt.ylabel(r'Inflow (m s$^{-1}$)')
# plt.title('Worst Percent Difference')
plt.savefig(f"{figs_dir}/worst_percent_diff_single_blade_Reconstruction.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(4.5,3))
plt.imshow(total_global_force_vector[:, :, 0], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Force (N)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Fx')
plt.savefig(f"{figs_dir}/Fx_single_blade_Reconstruction.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(4.5,3))
plt.imshow(total_global_force_vector[:, :, 1], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Force (N)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Fy')
plt.savefig(f"{figs_dir}/Fy_single_blade_Reconstruction.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(4.5,3))
plt.imshow(total_global_force_vector[:, :, 2], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Force (N)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Fz')
plt.savefig(f"{figs_dir}/Fz_single_blade_Reconstruction.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(4.5,3))
plt.imshow(total_global_moment_vector[:, :, 0], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Moment (N-m)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Mx')
plt.savefig(f"{figs_dir}/Mx_single_blade_Reconstruction.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(4.5,3))
plt.imshow(total_global_moment_vector[:, :, 1], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Moment (N-m)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('My')
plt.savefig(f"{figs_dir}/My_single_blade_Reconstruction.pdf", bbox_inches='tight')
plt.close()

plot_ = plt.figure(figsize=(4.5,3))
plt.imshow(total_global_moment_vector[:, :, 2], 
           extent=[viv_params.azimuths[0], viv_params.azimuths[-1], 
                  viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
           aspect='auto', origin='lower', cmap='viridis_r')
plt.colorbar(label='Moment (N-m)')
plt.xlabel('Azimuth (deg)')
plt.ylabel('Inflow (m/s)')
plt.title('Mz')
plt.savefig(f"{figs_dir}/Mz_single_blade_Reconstruction.pdf", bbox_inches='tight')
plt.close()

# Compare with unit test file if available
print("\nComparing results with unit test...")

# --- Load CSV file ---
# Replace with your actual file path

def safe_load_dat(path: str):
    """Load .dat with one header row (skiprows=1). Returns np.ndarray or None."""
    try:
        arr = np.loadtxt(path, skiprows=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        print(f"Warning: Skipping {path}, unreadable or no data.")
        return None

df = pd.read_csv(nodal_force_time_file, skipinitialspace=True)
data = safe_load_dat(f"{path}/../data/airfoils/2024_Ganesh_VIV_Paper_Data/ffa_data_files_ftt_160/ffa_w3_211/RE1_00E7/ffa_w3_211_144.dat")
# data = safe_load_dat(f"{path}/../data/airfoils/NALURuns/NACA0018/RE5_00E5/NACA0018_164.dat")
fpx, fpy = data[:, 1], data[:, 2]
fvx, fvy = data[:, 4], data[:, 5]
mty      = data[:, 8]
timefull = data[:, 0]
lift = fpy + fvy

# --- Plot ---
plt.figure(figsize=(4.5, 3))
q = 0.5*1.225*75.0**2*1.0*4.0 #Note that the original CFD span was 4, which these loads in vorlap are for each node and are per unit length, and this case just has 2 nodes.
plt.plot(timefull, -lift, 'k',label="Shreyas CFD", linewidth=2)
plt.plot(df["time"], df["node2y"]*4, label="VorLap Reconstructed", linewidth=2)
# plt.plot(viv_params.output_time,global_force_vector_nodes[:,1,1]*4, label="VorLap Reconstructed2", linewidth=2)

# --- Labels and legend ---
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
# plt.title("Node 2 X and Y vs Time")
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.xlim([0.0,2.0])
plt.ylim([0.0,40000.0])
plt.savefig(f"{figs_dir}/ReconstructedForce_Reconstruction.pdf", bbox_inches='tight', transparent=True)
plt.show()