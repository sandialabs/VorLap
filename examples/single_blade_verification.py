"""Single-blade verification example using repository reference data."""

from pathlib import Path

import numpy as np

from vorlap.computations import compute_thrust_torque_spectrum_optimized
from vorlap.fileio import load_airfoil_fft, load_components_from_csv
from vorlap.graphics import calc_structure_vectors_andplot
from vorlap.structs import VIV_Params


def run():
    repo = Path(__file__).resolve().parents[1]

    viv_params = VIV_Params(
        fluid_density=1.225,
        fluid_dynamicviscosity=1.81e-5,
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        rotation_axis_offset=np.array([0.0, 0.0, 0.0]),
        inflow_vec=np.array([1.0, 0.0, 0.0]),
        azimuths=np.array([4.0, 52.0]),
        inflow_speeds=np.array([7.3878]),
        output_time=np.array([0.0, 0.01]),
        output_azimuth_vinf=(999.0, 999.0),
        n_harmonic=2,
        amplitude_coeff_cutoff=0.002,
        n_freq_depth=10,
        airfoil_folder=str(repo / "data" / "airfoils") + "/",
    )

    natfreqs = np.loadtxt(repo / "data" / "natural_frequencies.csv", delimiter=",")
    components = load_components_from_csv(str(repo / "data" / "components" / "componentsSingle"))

    afft = load_airfoil_fft(str(repo / "data" / "airfoils" / "NACA0018.h5"))
    affts = {afft.name: afft, "default": afft}

    # Populates component chord/normal/global vectors used by the solver.
    calc_structure_vectors_andplot(components, viv_params, show_plot=False, return_fig=False, save_path=None)

    _, _, total_force, _, _ = compute_thrust_torque_spectrum_optimized(
        components, affts, viv_params, natfreqs
    )

    q = 0.5 * 1.225 * 7.3878**2 * 1.0 * 80.0
    expected_drag_4 = q * 0.024
    expected_lift_4 = q * -0.402
    expected_drag_52 = q * 1.98
    expected_lift_52 = q * -1.5

    print("AOA -4 deg case:")
    print("  Fx (VorLap):", total_force[0, 0, 0], "Expected:", expected_drag_4)
    print("  Fy (VorLap):", total_force[0, 0, 1], "Expected:", expected_lift_4)
    print("AOA -52 deg case:")
    print("  Fx (VorLap):", total_force[0, 1, 0], "Expected:", expected_drag_52)
    print("  Fy (VorLap):", total_force[0, 1, 1], "Expected:", expected_lift_52)


if __name__ == "__main__":
    run()
