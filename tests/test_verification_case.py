from pathlib import Path

import numpy as np
import pytest

from vorlap.computations import compute_thrust_torque_spectrum_optimized, rotate_vector
from vorlap.fileio import load_airfoil_fft, load_components_from_csv
from vorlap.interpolation import AirfoilRangeWarning
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

    with pytest.warns(RuntimeWarning) as caught:
        afft = load_airfoil_fft(str(repo / "data" / "airfoils" / "NACA0018.h5"))
    assert any("Repairing legacy airfoil FFT" in str(warning.message) for warning in caught)
    affts = {afft.name: afft, "default": afft}

    with pytest.warns(AirfoilRangeWarning, match="outside source grid"):
        _, _, total_force, _, _ = compute_thrust_torque_spectrum_optimized(components, affts, viv_params, natfreqs)

    q = 0.5 * 1.225 * 7.3878**2 * 1.0 * 80.0
    chord_unit = components[0].chord_vector[0] / np.linalg.norm(components[0].chord_vector[0])
    normal_unit = components[0].normal_vector[0] / np.linalg.norm(components[0].normal_vector[0])

    def projected_force_components(azimuth_deg: float, force: np.ndarray) -> tuple[float, float]:
        chord_rotated = rotate_vector(chord_unit, viv_params.rotation_axis, azimuth_deg)
        normal_rotated = rotate_vector(normal_unit, viv_params.rotation_axis, azimuth_deg)
        return float(np.dot(force, chord_rotated)), float(np.dot(force, normal_rotated))

    # AOA = -4 deg
    expected_drag_4 = q * 0.024
    expected_lift_4 = q * -0.402
    drag_4, lift_4 = projected_force_components(4.0, total_force[0, 0, :])
    np.testing.assert_allclose(drag_4, expected_drag_4, rtol=0.05, atol=1e-2)
    np.testing.assert_allclose(lift_4, expected_lift_4, rtol=0.05, atol=1e-2)

    # AOA = -52 deg
    expected_drag_52 = q * 1.98
    expected_lift_52 = q * -1.5
    drag_52, lift_52 = projected_force_components(52.0, total_force[0, 1, :])
    np.testing.assert_allclose(drag_52, expected_drag_52, rtol=0.05, atol=1e-2)
    np.testing.assert_allclose(lift_52, expected_lift_52, rtol=0.05, atol=1e-2)
