import warnings

import numpy as np
import pytest

from vorlap import QBladeController
from vorlap.computations import (
    _force_from_chord_normal,
    _strouhal_reference_length,
    compute_time_varying_force_history,
    compute_time_varying_force_history_optimized,
    compute_thrust_torque_spectrum,
    compute_thrust_torque_spectrum_optimized,
    prepare_component_frames,
    reconstruct_nonstationary_signal,
    reconstruct_signal,
    rotate_vector,
    rotationMatrix,
)
from vorlap.force_model import (
    SectionFlowState,
    force_from_chord_normal,
    harmonic_force_channel,
    harmonic_force_channel_from_angle,
    section_flow_state,
    strouhal_reference_length,
)
from vorlap.geometry_frames import (
    prepare_component_frames as geometry_prepare_component_frames,
    rotate_vector as geometry_rotate_vector,
    rotationMatrix as geometry_rotationMatrix,
)
from vorlap.spectral_reconstruction import (
    dominant_spectral_peaks,
    one_sided_power_spectrum,
    reconstruct_nonstationary_signal as spectral_reconstruct_nonstationary_signal,
    reconstruct_signal as spectral_reconstruct_signal,
    verify_expected_spectral_content,
    verify_frequency_band_spectral_content,
)
from vorlap.stationary_sweep import (
    compute_thrust_torque_spectrum as stationary_compute_thrust_torque_spectrum,
    compute_thrust_torque_spectrum_optimized as stationary_compute_thrust_torque_spectrum_optimized,
)
from vorlap.structs import InflowTimeSeries
from vorlap.time_varying_history import (
    compute_time_varying_force_history as history_compute_time_varying_force_history,
    compute_time_varying_force_history_optimized as history_compute_time_varying_force_history_optimized,
)

from conftest import make_component, make_constant_airfoil_fft, make_viv_params


def _component_with_3d_section_basis(n_nodes: int = 3) -> object:
    """Return a component with non-planar chord/normal basis vectors."""
    component = make_component(n_nodes=n_nodes, span=2.0, airfoil_id="default")
    chord = np.array([1.0, 0.0, 0.25], dtype=float)
    chord /= np.linalg.norm(chord)
    normal = np.array([0.25, 0.0, -1.0], dtype=float)
    normal /= np.linalg.norm(normal)
    assert np.isclose(float(np.dot(chord, normal)), 0.0)
    component.chord_vector = np.tile(chord, (n_nodes, 1))
    component.normal_vector = np.tile(normal, (n_nodes, 1))
    return component


def _rotated_3d_basis_component(component_id: str, base_azimuth_deg: float, n_nodes: int = 2) -> object:
    component = _component_with_3d_section_basis(n_nodes=n_nodes)
    axis = np.array([0.0, 0.0, 1.0], dtype=float)
    component.id = component_id
    component.chord_vector = np.vstack(
        [rotate_vector(row, axis, base_azimuth_deg) for row in component.chord_vector]
    )
    component.normal_vector = np.vstack(
        [rotate_vector(row, axis, base_azimuth_deg) for row in component.normal_vector]
    )
    component.translation = rotate_vector(np.array([3.0, 0.0, 0.0], dtype=float), axis, base_azimuth_deg)
    return component


def test_rotate_vector_right_hand_rule():
    vec = np.array([1.0, 0.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    rotated = rotate_vector(vec, axis, 90.0)
    np.testing.assert_allclose(rotated, np.array([0.0, 1.0, 0.0]), atol=1e-12)


def test_rotate_vector_zero_axis_raises():
    with pytest.raises(ValueError, match="non-zero magnitude"):
        rotate_vector(np.array([1.0, 0.0, 0.0]), np.zeros(3), 30.0)


def test_geometry_frame_module_matches_computations_compatibility_exports():
    assert rotate_vector is geometry_rotate_vector
    assert prepare_component_frames is geometry_prepare_component_frames
    assert rotationMatrix is geometry_rotationMatrix

    euler = np.array([15.0, -5.0, 40.0], dtype=float)
    np.testing.assert_allclose(rotationMatrix(euler), geometry_rotationMatrix(euler), atol=1e-12)


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


def test_reconstruct_signal_requires_uniform_time_spacing():
    with pytest.raises(ValueError, match="uniform spacing"):
        reconstruct_signal(
            np.array([0.0, 1.0]),
            np.array([1.0, 0.25]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.1, 0.35]),
        )


def test_reconstruct_nonstationary_signal_reduces_to_stationary_case():
    t = np.linspace(0.0, 1.0, 21)
    freqs = np.column_stack([np.zeros_like(t), np.full_like(t, 2.0)])
    amps = np.column_stack([np.full_like(t, 1.5), np.full_like(t, 0.25)])
    phases = np.column_stack([np.zeros_like(t), np.full_like(t, 0.2)])

    signal = reconstruct_nonstationary_signal(freqs, amps, phases, t, smoothing_cycles=0.0)
    expected = 1.5 + 0.25 * np.cos(2.0 * np.pi * 2.0 * t + 0.2)
    np.testing.assert_allclose(signal, expected, atol=1e-12)


def test_spectral_content_identifies_expected_force_frequencies():
    t = np.arange(0.0, 4.0, 0.01)
    signal = 5.0 + 2.0 * np.cos(2.0 * np.pi * 1.5 * t + 0.2) + 0.5 * np.cos(2.0 * np.pi * 3.0 * t - 0.1)

    frequencies, power = one_sided_power_spectrum(t, signal)
    peaks = dominant_spectral_peaks(t, signal, top_n=2)
    expected = verify_expected_spectral_content(t, signal, np.array([1.5, 3.0]), min_power_fraction=0.01)
    band = verify_frequency_band_spectral_content(t, signal, min_frequency_hz=1.0, max_frequency_hz=3.5)
    out_of_band = verify_frequency_band_spectral_content(
        t,
        signal,
        min_frequency_hz=4.0,
        max_frequency_hz=5.0,
        min_band_power_fraction=0.5,
    )

    assert frequencies[0] == pytest.approx(0.0)
    assert float(power[0]) == pytest.approx(0.0, abs=1.0e-12)
    assert [round(peak["frequency_hz"], 6) for peak in peaks] == [1.5, 3.0]
    assert expected["passed"] is True
    assert all(row["passed"] for row in expected["comparisons"])
    assert band["passed"] is True
    assert out_of_band["passed"] is False


def test_spectral_reconstruction_module_matches_computations_compatibility_exports():
    assert reconstruct_signal is spectral_reconstruct_signal
    assert reconstruct_nonstationary_signal is spectral_reconstruct_nonstationary_signal


def test_time_varying_history_module_matches_computations_compatibility_exports():
    assert compute_time_varying_force_history is history_compute_time_varying_force_history
    assert compute_time_varying_force_history_optimized is history_compute_time_varying_force_history_optimized


def test_stationary_sweep_module_matches_computations_compatibility_exports():
    assert compute_thrust_torque_spectrum is stationary_compute_thrust_torque_spectrum
    assert compute_thrust_torque_spectrum_optimized is stationary_compute_thrust_torque_spectrum_optimized


def test_strouhal_reference_length_uses_projected_chord_with_thickness_floor():
    chord = np.array([2.0, 2.0, 2.0], dtype=float)
    aoa = np.array([0.0, 10.0, 30.0], dtype=float)
    thickness = np.array([0.12, 0.12, 0.12], dtype=float)

    st_length = _strouhal_reference_length(chord, aoa, thickness)

    expected = chord * np.array([0.12, np.sin(np.deg2rad(10.0)), np.sin(np.deg2rad(30.0))])
    np.testing.assert_allclose(st_length, expected, atol=1e-12)
    np.testing.assert_allclose(strouhal_reference_length(chord, aoa, thickness), expected, atol=1e-12)


def test_section_flow_state_reports_unit_labelled_quantities():
    velocity = np.array([[3.0, 4.0, 0.0]], dtype=float)
    chord_unit = np.array([[1.0, 0.0, 0.0]], dtype=float)
    normal_unit = np.array([[0.0, 1.0, 0.0]], dtype=float)

    state = section_flow_state(
        velocity,
        chord_unit,
        normal_unit,
        chord_m=np.array([2.0]),
        local_length_m=np.array([0.5]),
        fluid_density_kgpm3=1.2,
        fluid_dynamicviscosity_pa_s=2.0e-5,
    )

    assert isinstance(state, SectionFlowState)
    np.testing.assert_allclose(state.velocity_chord_mps, [3.0])
    np.testing.assert_allclose(state.velocity_normal_mps, [4.0])
    np.testing.assert_allclose(state.aoa_deg, [np.degrees(np.arctan2(4.0, 3.0))])
    np.testing.assert_allclose(state.effective_velocity_mps, [5.0])
    np.testing.assert_allclose(state.reynolds, [1.2 * 5.0 * 2.0 / 2.0e-5])
    np.testing.assert_allclose(state.section_dynamic_force_n, [0.5 * 1.2 * 25.0 * 2.0 * 0.5])


def test_harmonic_force_channel_reconstructs_coefficients_with_mean_control():
    dynamic_force = np.array([10.0, 20.0], dtype=float)
    frequency = np.array([[0.0, 2.0], [0.0, 4.0]], dtype=float)
    amplitude = np.array([[1.5, 0.25], [2.0, 0.5]], dtype=float)
    phase = np.zeros_like(amplitude)

    with_mean = harmonic_force_channel(
        dynamic_force,
        frequency,
        amplitude,
        phase,
        time_s=0.0,
        include_mean_load=True,
    )
    no_mean = harmonic_force_channel(
        dynamic_force,
        frequency,
        amplitude,
        phase,
        time_s=0.0,
        include_mean_load=False,
    )
    history = harmonic_force_channel(
        10.0,
        frequency[0],
        amplitude[0],
        phase[0],
        time_s=np.array([0.0, 0.125], dtype=float),
        include_mean_load=False,
    )

    np.testing.assert_allclose(with_mean, dynamic_force * np.array([1.75, 2.5]))
    np.testing.assert_allclose(no_mean, dynamic_force * np.array([0.25, 0.5]))
    np.testing.assert_allclose(history, [2.5, 0.0], atol=1.0e-12)


def test_harmonic_force_channel_from_angle_matches_constant_frequency_reconstruction():
    dynamic_force = np.array([2.0, 3.0])
    frequency = np.array([[0.0, 1.5, 2.5], [0.0, 1.0, 3.0]])
    amplitude = np.array([[0.4, 0.2, 0.1], [0.3, 0.5, 0.25]])
    source_phase = np.array([[0.0, 0.1, -0.2], [0.0, -0.3, 0.4]])
    time = 0.37
    angle = 2.0 * np.pi * frequency * time + source_phase

    expected = harmonic_force_channel(
        dynamic_force,
        frequency,
        amplitude,
        source_phase,
        time,
        include_mean_load=True,
    )
    actual = harmonic_force_channel_from_angle(
        dynamic_force,
        amplitude,
        angle,
        include_mean_load=True,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)


def test_harmonic_force_channel_from_angle_can_suppress_dc_term():
    actual = harmonic_force_channel_from_angle(
        np.array([4.0]),
        np.array([[10.0, 0.5]]),
        np.array([[99.0, np.pi / 2.0]]),
        include_mean_load=False,
    )

    np.testing.assert_allclose(actual, 0.0, atol=1.0e-12)


def test_force_model_kernel_matches_legacy_computations_alias():
    chord_force = np.array([1.0, 2.0])
    normal_force = np.array([3.0, 4.0])
    chord_unit = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    normal_unit = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    expected = np.array([[1.0, 0.0, 3.0], [4.0, 2.0, 0.0]])

    np.testing.assert_allclose(
        force_from_chord_normal(chord_force, normal_force, chord_unit, normal_unit),
        expected,
    )
    np.testing.assert_allclose(
        _force_from_chord_normal(chord_force, normal_force, chord_unit, normal_unit),
        expected,
    )


def test_prepare_component_frames_fills_csv_style_placeholders():
    component = make_component(n_nodes=3, span=2.0, airfoil_id="default")
    component.translation = np.array([1.0, 2.0, 3.0], dtype=float)
    component.rotation = np.array([0.0, 0.0, 90.0], dtype=float)
    component.shape_xyz_global = np.zeros_like(component.shape_xyz)
    component.chord_vector = np.zeros_like(component.chord_vector)
    component.normal_vector = np.zeros_like(component.normal_vector)

    prepare_component_frames([component])

    expected_xyz = np.column_stack(
        [
            np.ones(3),
            np.full(3, 2.0),
            np.linspace(3.0, 5.0, 3),
        ]
    )
    np.testing.assert_allclose(component.shape_xyz_global, expected_xyz, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(component.chord_vector, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(component.normal_vector, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(component.chord_vector * component.normal_vector, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(component.chord_vector[0], np.array([0.0, 1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(component.normal_vector[0], np.array([-1.0, 0.0, 0.0]), atol=1e-12)


def test_compute_spectrum_prepares_zero_frames_without_plotting():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.shape_xyz_global = np.zeros_like(component.shape_xyz)
    component.chord_vector = np.zeros_like(component.chord_vector)
    component.normal_vector = np.zeros_like(component.normal_vector)
    affts = {"default": make_constant_airfoil_fft()}
    viv_params = make_viv_params()

    _, _, total_force, _, _ = compute_thrust_torque_spectrum_optimized(
        [component], affts, viv_params, np.array([1.0])
    )

    np.testing.assert_allclose(total_force[0, 0, :], np.array([8.0, 4.0, 0.0]), atol=1e-12)
    assert np.all(np.linalg.norm(component.chord_vector, axis=1) > 0.0)
    assert np.all(np.linalg.norm(component.normal_vector, axis=1) > 0.0)


def test_compute_spectrum_rejects_empty_or_invalid_sweeps():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    affts = {"default": make_constant_airfoil_fft()}

    viv = make_viv_params()
    viv.azimuths = np.array([], dtype=float)
    with pytest.raises(ValueError, match="azimuths"):
        compute_thrust_torque_spectrum_optimized([component], affts, viv, np.array([0.1]))

    viv = make_viv_params()
    viv.inflow_speeds = np.array([-1.0], dtype=float)
    with pytest.raises(ValueError, match="inflow_speeds values must be non-negative"):
        compute_thrust_torque_spectrum_optimized([component], affts, viv, np.array([0.1]))

    viv = make_viv_params()
    viv.output_time = np.array([0.1, 0.0], dtype=float)
    with pytest.raises(ValueError, match="output_time values must be strictly increasing"):
        compute_thrust_torque_spectrum_optimized([component], affts, viv, np.array([0.1]))


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


def test_batch_and_qblade_controller_match_nonzero_azimuth():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=1)
    affts = {"default": afft}
    viv_params = make_viv_params()
    viv_params.azimuths = np.array([30.0], dtype=float)
    viv_params.output_azimuth_vinf = (30.0, 2.0)
    natfreqs = np.array([1.0])

    _, _, total_force, _, node_forces = compute_thrust_torque_spectrum_optimized(
        [component], affts, viv_params, natfreqs
    )

    controller = QBladeController.from_components(
        [component],
        affts,
        viv_params=viv_params,
        n_freq_depth=1,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        include_mean_load=True,
    )
    inflow_dir = viv_params.inflow_vec / np.linalg.norm(viv_params.inflow_vec)
    velocity = np.tile(inflow_dir * viv_params.inflow_speeds[0], (2, 1))
    controller_forces = controller.step({"velocity": velocity, "time": 0.0, "azimuth_deg": 30.0})

    np.testing.assert_allclose(total_force[0, 0, :], np.sum(controller_forces, axis=0), atol=1e-12)
    np.testing.assert_allclose(node_forces[0, :, :].T, controller_forces, atol=1e-12)


def test_batch_and_qblade_controller_match_multiple_azimuths_with_3d_section_basis():
    azimuths = np.array([0.0, 30.0, 90.0, 120.0, 180.0], dtype=float)
    afft = make_constant_airfoil_fft(n_freq=1)
    affts = {"default": afft}
    node_ids = [f"BLD_1_{position:.6f}" for position in np.linspace(0.0, 1.0, 3)]

    for azimuth in azimuths:
        component = _component_with_3d_section_basis(n_nodes=3)
        viv_params = make_viv_params()
        viv_params.azimuths = np.array([azimuth], dtype=float)
        viv_params.output_azimuth_vinf = (float(azimuth), 2.0)
        natfreqs = np.array([1.0])

        _, _, total_force, _, node_forces = compute_thrust_torque_spectrum_optimized(
            [component], affts, viv_params, natfreqs
        )

        controller = QBladeController.from_components(
            [component],
            affts,
            viv_params=viv_params,
            n_freq_depth=1,
            node_ids=node_ids,
            include_mean_load=True,
        )
        inflow_dir = viv_params.inflow_vec / np.linalg.norm(viv_params.inflow_vec)
        velocity = np.tile(inflow_dir * viv_params.inflow_speeds[0], (len(node_ids), 1))
        controller_forces = controller.step({"velocity": velocity, "time": 0.0, "azimuth_deg": azimuth})

        np.testing.assert_allclose(total_force[0, 0, :], np.sum(controller_forces, axis=0), atol=1e-12)
        np.testing.assert_allclose(node_forces[0, :, :].T, controller_forces, atol=1e-12)


def test_time_varying_force_history_matches_qblade_controller_nonzero_azimuth():
    component = _component_with_3d_section_basis(n_nodes=3)
    afft = make_constant_airfoil_fft(n_freq=2, st_h1=0.25)
    afft.CL_Amp[:, :, 1] = 0.2
    afft.CD_Amp[:, :, 1] = 0.1
    afft.CL_Pha[:, :, 1] = 0.3
    afft.CD_Pha[:, :, 1] = -0.4
    affts = {"default": afft}
    viv_params = make_viv_params()
    viv_params.output_time = np.linspace(0.0, 0.8, 81)
    azimuth = 45.0
    inflow_dir = viv_params.inflow_vec / np.linalg.norm(viv_params.inflow_vec)
    inflow_profile = InflowTimeSeries(
        time=viv_params.output_time,
        inflow_speeds=np.full(viv_params.output_time.size, viv_params.inflow_speeds[0]),
        inflow_directions=np.tile(inflow_dir, (viv_params.output_time.size, 1)),
    )

    _, total_force, _, node_forces = compute_time_varying_force_history_optimized(
        [component],
        affts,
        viv_params,
        inflow_profile,
        azimuth_deg=azimuth,
        smoothing_cycles=0.0,
        include_mean_load=False,
    )

    node_ids = [f"BLD_1_{position:.6f}" for position in np.linspace(0.0, 1.0, 3)]
    controller = QBladeController.from_components(
        [component],
        affts,
        viv_params=viv_params,
        n_freq_depth=2,
        node_ids=node_ids,
    )
    velocity = np.tile(inflow_dir * viv_params.inflow_speeds[0], (len(node_ids), 1))
    controller_history = np.stack(
        [
            controller.step({"velocity": velocity, "time": time, "azimuth_deg": azimuth})
            for time in viv_params.output_time
        ],
        axis=0,
    )

    np.testing.assert_allclose(node_forces.transpose(0, 2, 1), controller_history, atol=1e-12)
    np.testing.assert_allclose(total_force, np.sum(controller_history, axis=1), atol=1e-12)


def test_stationary_reconstruction_can_match_oscillatory_only_qblade_controller():
    component = _component_with_3d_section_basis(n_nodes=2)
    afft = make_constant_airfoil_fft(n_freq=2, st_h1=0.5)
    afft.CL_Amp[:, :, 1] = 0.2
    afft.CD_Amp[:, :, 1] = 0.1
    afft.CL_Pha[:, :, 1] = 0.0
    afft.CD_Pha[:, :, 1] = 0.0
    affts = {"default": afft}
    viv_params = make_viv_params()
    viv_params.azimuths = np.array([30.0], dtype=float)
    viv_params.output_azimuth_vinf = (30.0, 2.0)
    viv_params.output_time = np.array([0.0, 0.05], dtype=float)

    _, _, total_force, _, node_forces = compute_thrust_torque_spectrum_optimized(
        [component],
        affts,
        viv_params,
        np.array([1.0]),
        include_mean_load=False,
    )

    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    controller = QBladeController.from_components(
        [component],
        affts,
        viv_params=viv_params,
        n_freq_depth=2,
        node_ids=node_ids,
    )
    inflow_dir = viv_params.inflow_vec / np.linalg.norm(viv_params.inflow_vec)
    velocity = np.tile(inflow_dir * viv_params.inflow_speeds[0], (len(node_ids), 1))
    controller_history = np.stack(
        [
            controller.step({"velocity": velocity, "time": time, "azimuth_deg": 30.0})
            for time in viv_params.output_time
        ],
        axis=0,
    )

    np.testing.assert_allclose(total_force[0, 0, :], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(node_forces.transpose(0, 2, 1), controller_history, atol=1e-12)


def test_three_blade_symmetry_maps_under_120_degree_azimuth_shift():
    affts = {"default": make_constant_airfoil_fft(n_freq=1)}
    viv_params = make_viv_params()
    viv_params.output_time = np.array([0.0, 0.25], dtype=float)
    natfreqs = np.array([1.0])

    def solve_nodes(azimuth: float) -> np.ndarray:
        components = [
            _rotated_3d_basis_component("BLD_1", 0.0),
            _rotated_3d_basis_component("BLD_2", 120.0),
            _rotated_3d_basis_component("BLD_3", 240.0),
        ]
        viv_params.azimuths = np.array([azimuth], dtype=float)
        viv_params.output_azimuth_vinf = (azimuth, 2.0)
        _, _, _, _, node_forces = compute_thrust_torque_spectrum_optimized(
            components,
            affts,
            viv_params,
            natfreqs,
        )
        return node_forces[0].T.reshape(3, 2, 3)

    forces_at_0 = solve_nodes(0.0)
    forces_at_120 = solve_nodes(120.0)

    np.testing.assert_allclose(forces_at_120[0], forces_at_0[1], atol=1e-12)
    np.testing.assert_allclose(forces_at_120[1], forces_at_0[2], atol=1e-12)
    np.testing.assert_allclose(forces_at_120[2], forces_at_0[0], atol=1e-12)


def test_low_aoa_strouhal_scaling_does_not_alias():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    affts = {"default": afft}
    viv_params = make_viv_params()
    viv_params.inflow_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    viv_params.output_time = np.linspace(0.0, 1.0, 1001)
    viv_params.output_azimuth_vinf = (0.0, 2.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_thrust_torque_spectrum_optimized([component], affts, viv_params, np.array([1.0]))

    alias_warnings = [
        warning for warning in caught
        if issubclass(warning.category, RuntimeWarning) and "alias" in str(warning.message)
    ]
    assert not alias_warnings


def test_qblade_controller_uses_low_aoa_thickness_for_strouhal_scaling():
    component = make_component(n_nodes=1, span=1.0, airfoil_id="default")
    component.thickness[:] = 0.2
    afft = make_constant_airfoil_fft(n_freq=2, cl_dc=0.0, cd_dc=0.0, st_h1=0.5)
    afft.CL_Amp[:, :, 1] = 1.0
    affts = {"default": afft}
    viv_params = make_viv_params()
    viv_params.inflow_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    viv_params.inflow_speeds = np.array([2.0], dtype=float)

    controller = QBladeController.from_components(
        [component],
        affts,
        viv_params=viv_params,
        n_freq_depth=2,
        node_ids=["BLD_1_0.000000"],
    )
    force = controller.step(
        {
            "velocity": np.array([[2.0, 0.0, 0.0]], dtype=float),
            "time": 0.1,
            "azimuth_deg": 0.0,
        }
    )

    # AOA = 0 deg, so L_st = chord * thickness = 0.2 m.
    # f = St * V / L_st = 0.5 * 2 / 0.2 = 5 Hz and q = 2 N.
    # At t = 0.1 s, cos(2*pi*5*t) = -1, giving -2 N in the normal direction.
    np.testing.assert_allclose(force[0], np.array([0.0, -2.0, 0.0]), atol=1e-12)


def test_frequency_band_filters_overlap_candidates():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2, cf_h1_amp=0.1, st_h1=0.5)
    affts = {"default": afft}
    viv_params = make_viv_params()
    viv_params.inflow_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    viv_params.output_time = np.linspace(0.0, 1.0, 1001)
    viv_params.freq_min = 1.5
    viv_params.freq_max = 2.0

    percdiff_matrix, percdiff_info, *_ = compute_thrust_torque_spectrum_optimized(
        [component], affts, viv_params, np.array([1.0])
    )

    np.testing.assert_allclose(percdiff_matrix, 1000.0)
    assert percdiff_info[0, 0] is None


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


def test_time_varying_force_history_prepares_zero_frames_without_plotting():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.shape_xyz_global = np.zeros_like(component.shape_xyz)
    component.chord_vector = np.zeros_like(component.chord_vector)
    component.normal_vector = np.zeros_like(component.normal_vector)
    affts = {"default": make_constant_airfoil_fft(n_freq=1)}
    viv_params = make_viv_params()
    inflow_dir = viv_params.inflow_vec / np.linalg.norm(viv_params.inflow_vec)
    inflow_profile = InflowTimeSeries(
        time=np.array([0.0, 0.25], dtype=float),
        inflow_speeds=np.array([2.0, 2.0], dtype=float),
        inflow_directions=np.tile(inflow_dir, (2, 1)),
    )

    _, total_force, _, node_forces = compute_time_varying_force_history_optimized(
        [component],
        affts,
        viv_params,
        inflow_profile,
        azimuth_deg=0.0,
        smoothing_cycles=0.0,
    )

    np.testing.assert_allclose(total_force, np.tile(np.array([[8.0, 4.0, 0.0]]), (2, 1)), atol=1e-12)
    np.testing.assert_allclose(node_forces[:, 0, :], 4.0, atol=1e-12)
    np.testing.assert_allclose(node_forces[:, 1, :], 2.0, atol=1e-12)


def test_time_varying_standard_and_optimized_paths_match():
    component = make_component(n_nodes=3, span=3.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2, st_h1=0.1)
    afft.CL_Amp[:, :, 1] = 0.15
    afft.CD_Amp[:, :, 1] = 0.05
    affts = {"default": afft}
    viv_params = make_viv_params()

    inflow_profile = InflowTimeSeries(
        time=np.array([0.0, 0.05, 0.10, 0.20, 0.25]),
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
