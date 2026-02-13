import numpy as np
import pytest

from vorlap.computations import (
    compute_thrust_torque_spectrum,
    compute_thrust_torque_spectrum_optimized,
    reconstruct_signal,
    rotate_vector,
)

from conftest import make_component, make_constant_airfoil_fft, make_viv_params


def test_rotate_vector_right_hand_rule():
    vec = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    rotated = rotate_vector(vec, axis, 90.0)
    np.testing.assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_rotate_vector_zero_axis_raises():
    with pytest.raises(ValueError, match="non-zero magnitude"):
        rotate_vector(np.array([1.0, 0.0, 0.0]), np.zeros(3), 30.0)


def test_reconstruct_signal_dc_and_harmonic():
    t = np.linspace(0.0, 1.0, 21)
    freqs = np.array([0.0, 2.0])
    amps = np.array([1.5, 0.25])
    phases = np.array([0.0, 0.0])
    signal = reconstruct_signal(freqs, amps, phases, t)
    expected = 1.5 + 0.25 * np.cos(2.0 * np.pi * 2.0 * t)
    np.testing.assert_allclose(signal, expected, atol=1e-12)


def test_reconstruct_signal_requires_monotonic_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        reconstruct_signal(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([1.0, 0.5]),
        )


def test_compute_spectrum_uses_cross_product_moment_and_segment_weights():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft()
    affts = {"default": afft}
    viv_params = make_viv_params()
    natfreqs = np.array([1.0])

    percdiff_matrix, percdiff_info, total_force, total_moment, node_forces = compute_thrust_torque_spectrum_optimized(
        [component], affts, viv_params, natfreqs
    )

    # Two nodes, each with q=2.0 and [drag,lift,0]=[4,2,0]
    np.testing.assert_allclose(total_force[0, 0, :], np.array([8.0, 4.0, 0.0]), atol=1e-12)

    # Moment about origin uses r x F. Node at z=2 contributes [-4, 8, 0].
    np.testing.assert_allclose(total_moment[0, 0, :], np.array([-4.0, 8.0, 0.0]), atol=1e-12)

    # Output reconstruction should be available for requested azimuth/inflow.
    np.testing.assert_allclose(node_forces[:, 0, 0], np.array([4.0, 4.0]), atol=1e-12)
    np.testing.assert_allclose(node_forces[:, 1, 0], np.array([2.0, 2.0]), atol=1e-12)
    assert np.isfinite(percdiff_matrix[0, 0])
    assert percdiff_info[0, 0] is not None


def test_standard_and_optimized_paths_match_on_synthetic_case():
    component = make_component(n_nodes=3, span=3.0, airfoil_id="default")
    afft = make_constant_airfoil_fft()
    affts = {"default": afft}
    viv_params = make_viv_params()
    natfreqs = np.array([1.0, 2.0])

    out_std = compute_thrust_torque_spectrum([component], affts, viv_params, natfreqs)
    out_opt = compute_thrust_torque_spectrum_optimized([component], affts, viv_params, natfreqs)

    for arr_std, arr_opt in zip(out_std, out_opt):
        if isinstance(arr_std, np.ndarray) and arr_std.dtype != object:
            np.testing.assert_allclose(arr_std, arr_opt, atol=1e-12)


def test_compute_spectrum_missing_default_airfoil_raises():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="unknown")
    affts = {"known": make_constant_airfoil_fft(name="known")}
    viv_params = make_viv_params()

    with pytest.raises(KeyError, match="no 'default'"):
        compute_thrust_torque_spectrum_optimized([component], affts, viv_params, np.array([1.0]))
