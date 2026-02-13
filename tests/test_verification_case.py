from pathlib import Path

import numpy as np

from vorlap.computations import compute_thrust_torque_spectrum_optimized
from vorlap.fileio import load_airfoil_fft, load_components_from_csv
from vorlap.graphics import calc_structure_vectors_andplot
from vorlap.structs import VIV_Params


def test_single_blade_verification_case_matches_expected_force_levels():
    """
    Regression-style verification test based on the existing single-blade case.

    The expected CL/CD values come from the team's documented verification script.
    """
    repo = Path(__file__).resolve().parents[1]

    viv_params = VIV_Params(
        fluid_density=1.225,
        fluid_dynamicviscosity=1.81e-5,
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        rotation_axis_offset=np.array([0.0, 0.0, 0.0]),
        inflow_vec=np.array([1.0, 0.0, 0.0]),
        azimuths=np.array([4.0, 52.0]),
        inflow_speeds=np.array([7.3878]),
        n_harmonic=2,
        output_time=np.array([0.0, 0.01]),
        output_azimuth_vinf=(999.0, 999.0),  # disable reconstruction path for speed
        amplitude_coeff_cutoff=0.002,
        n_freq_depth=10,
        airfoil_folder=str(repo / "data" / "airfoils") + "/",
    )

    natfreqs = np.loadtxt(repo / "data" / "natural_frequencies.csv", delimiter=",")
    components = load_components_from_csv(str(repo / "data" / "components" / "componentsSingle"))

    afft = load_airfoil_fft(str(repo / "data" / "airfoils" / "NACA0018.h5"))
    affts = {afft.name: afft, "default": afft}

    calc_structure_vectors_andplot(components, viv_params, show_plot=False, return_fig=False, save_path=None)
    _, _, total_force, _, _ = compute_thrust_torque_spectrum_optimized(components, affts, viv_params, natfreqs)

    q = 0.5 * 1.225 * 7.3878**2 * 1.0 * 80.0

    # AOA = -4 deg
    expected_drag_4 = q * 0.024
    expected_lift_4 = q * -0.402
    np.testing.assert_allclose(total_force[0, 0, 0], expected_drag_4, rtol=0.05, atol=1e-2)
    np.testing.assert_allclose(total_force[0, 0, 1], expected_lift_4, rtol=0.05, atol=1e-2)

    # AOA = -52 deg
    expected_drag_52 = q * 1.98
    expected_lift_52 = q * -1.5
    np.testing.assert_allclose(total_force[0, 1, 0], expected_drag_52, rtol=0.05, atol=1e-2)
    np.testing.assert_allclose(total_force[0, 1, 1], expected_lift_52, rtol=0.05, atol=1e-2)
