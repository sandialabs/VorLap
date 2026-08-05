import numpy as np
import pytest

from vorlap.computations import (
    _strouhal_reference_length,
    compute_time_varying_force_history,
    compute_time_varying_force_history_optimized,
    compute_thrust_torque_spectrum,
    compute_thrust_torque_spectrum_optimized,
    reconstruct_nonstationary_signal,
    reconstruct_signal,
    rotate_vector,
)
from vorlap.structs import InflowTimeSeries

from conftest import make_component, make_constant_airfoil_fft, make_viv_params


def test_rotate_vector_right_hand_rule():
    vec = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    rotated = rotate_vector(vec, axis, 90.0)
    np.testing.assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_rotate_vector_zero_axis_raises():
    with pytest.raises(ValueError, match="non-zero magnitude"):
        rotate_vector(np.array([1.0, 0.0, 0.0]), np.zeros(3), 30.0)


def test_strouhal_reference_length_uses_thickness_at_low_aoa():
    chord = np.array([2.0, 2.0, 2.0])
    aoa = np.array([0.0, 10.0, 30.0])
    thickness = np.array([0.12, 0.12, 0.12])
    expected = chord * np.array([0.12, np.sin(np.deg2rad(10.0)), 0.5])

    np.testing.assert_allclose(
        _strouhal_reference_length(chord, aoa, thickness),
        expected,
        atol=1.0e-12,
    )


def test_reconstruct_signal_dc_and_harmonic():
    t = np.linspace(0.0, 1.0, 21)
    freqs = np.array([0.0, 2.0])
    amps = np.array([1.5, 0.25])
    phases = np.array([0.0, 0.0])
    signal = reconstruct_signal(freqs, amps, phases, t)
    expected = 1.5 + 0.25 * np.cos(2.0 * np.pi * 2.0 * t)
    np.testing.assert_allclose(signal, expected, atol=1e-12)


def test_reconstruct_signal_includes_nyquist_once():
    t = np.arange(16, dtype=float) * 0.1
    freqs = np.array([0.0, 5.0])
    amps = np.array([0.3, 0.4])
    phases = np.array([0.0, 0.0])

    signal = reconstruct_signal(freqs, amps, phases, t)
    expected = 0.3 + 0.4 * np.cos(2.0 * np.pi * 5.0 * t)

    np.testing.assert_allclose(signal, expected, atol=1e-12)


def test_reconstruct_signal_requires_monotonic_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        reconstruct_signal(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([1.0, 0.5]),
        )


def test_reconstruct_nonstationary_signal_reduces_to_stationary_case():
    t = np.linspace(0.0, 1.0, 21)
    freqs = np.column_stack([np.zeros_like(t), np.full_like(t, 2.0)])
    amps = np.column_stack([np.full_like(t, 1.5), np.full_like(t, 0.25)])
    phases = np.column_stack([np.zeros_like(t), np.full_like(t, 0.2)])

    signal = reconstruct_nonstationary_signal(freqs, amps, phases, t, smoothing_cycles=0.0)
    expected = 1.5 + 0.25 * np.cos(2.0 * np.pi * 2.0 * t + 0.2)
    np.testing.assert_allclose(signal, expected, atol=1e-12)


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


def test_compute_spectrum_missing_airfoil_warns_and_falls_back_to_default():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="cylinder")
    default_afft = make_constant_airfoil_fft(name="NACA0018")
    affts = {"default": default_afft}
    viv_params = make_viv_params()

    with pytest.warns(RuntimeWarning, match="using default airfoil 'NACA0018'"):
        compute_thrust_torque_spectrum_optimized([component], affts, viv_params, np.array([1.0]))


def test_time_varying_force_history_matches_single_case_reconstruction():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    afft.CL_Amp[:, :, 1] = 0.2
    afft.CD_Amp[:, :, 1] = 0.1
    afft.CL_Pha[:, :, 1] = 0.3
    afft.CD_Pha[:, :, 1] = -0.4
    affts = {"default": afft}

    viv_params = make_viv_params()
    viv_params.output_time = np.linspace(0.0, 1.0, 101)
    viv_params.output_azimuth_vinf = (0.0, 2.0)
    natfreqs = np.array([1.0])

    _, _, _, _, node_forces_static = compute_thrust_torque_spectrum_optimized(
        [component], affts, viv_params, natfreqs
    )

    inflow_dir = viv_params.inflow_vec / np.linalg.norm(viv_params.inflow_vec)
    inflow_profile = InflowTimeSeries(
        time=viv_params.output_time,
        inflow_speeds=np.full(viv_params.output_time.shape[0], 2.0),
        inflow_directions=np.tile(inflow_dir, (viv_params.output_time.shape[0], 1)),
    )

    _, _, _, node_forces_time_varying = compute_time_varying_force_history_optimized(
        [component],
        affts,
        viv_params,
        inflow_profile,
        azimuth_deg=0.0,
        smoothing_cycles=0.0,
    )

    np.testing.assert_allclose(node_forces_time_varying, node_forces_static, atol=1e-12)


def test_time_varying_standard_and_optimized_paths_match():
    component = make_component(n_nodes=3, span=3.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    afft.CL_Amp[:, :, 1] = 0.15
    afft.CD_Amp[:, :, 1] = 0.05
    affts = {"default": afft}
    viv_params = make_viv_params()

    inflow_profile = InflowTimeSeries(
        time=np.array([0.0, 0.2, 0.4, 0.8, 1.0]),
        inflow_speeds=np.array([2.0, 4.0, 3.0, 5.0, 2.5]),
        inflow_directions=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.7, 0.7, 0.0],
                [0.0, 1.0, 0.0],
                [-0.7, 0.7, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
    )

    out_std = compute_time_varying_force_history(
        [component], affts, viv_params, inflow_profile, azimuth_deg=0.0, smoothing_cycles=0.0
    )
    out_opt = compute_time_varying_force_history_optimized(
        [component], affts, viv_params, inflow_profile, azimuth_deg=0.0, smoothing_cycles=0.0
    )

    for arr_std, arr_opt in zip(out_std, out_opt):
        np.testing.assert_allclose(arr_std, arr_opt, atol=1e-12)
