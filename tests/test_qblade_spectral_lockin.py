import math
import copy
from pathlib import Path

import numpy as np
import pytest

from vorlap.airfoil_io import load_airfoil_fft
from vorlap.interpolation import interpolate_fft_spectrum_batch
from vorlap.qblade_spectral_lockin import (
    CYLINDER_REFERENCE_LIFT_AMPLITUDE,
    CYLINDER_REFERENCE_STROUHAL,
    CYLINDER_REDUCED_VELOCITY_HIGH,
    CYLINDER_REDUCED_VELOCITY_LOW,
    CYLINDER_REDUCED_VELOCITY_PEAK,
    DEFAULT_CAPTURE_FREQUENCY_RATIO_HIGH,
    DEFAULT_CAPTURE_FREQUENCY_RATIO_LOW,
    DEFAULT_CAPTURE_FREQUENCY_RATIO_PEAK,
    HVAWTSpectralLockInSource,
    SpectralLockInCaptureConfig,
    SpectralLockInModeConfig,
    evaluate_modal_damping_balance,
    partition_lift_spectrum,
    smooth_capture_amplitude,
)
from vorlap.qblade_engineering_lockin import capture_weight
from vorlap.qblade_source_coverage import SOURCE_COVERAGE_COLUMNS
from vorlap.structs import AirfoilFFT


REPO_ROOT = Path(__file__).resolve().parents[1]


def _mode(
    tracked_mode=6,
    natural_frequency_hz=2.0,
    structural_damping_ratio=0.01,
):
    return SpectralLockInModeConfig(
        tracked_mode=tracked_mode,
        natural_frequency_hz=natural_frequency_hz,
        generalized_modal_mass_kg=100.0,
        structural_damping_ratio=structural_damping_ratio,
        reference_length_m=0.5,
        aerodynamic_damping_gain_h1=3.0,
        saturation_amplitude_ratio=0.3,
    )


def _synthetic_airfoil(*, cl_amplitude=0.3, cd_amplitude=0.0):
    re = np.array([1.0e5, 1.0e6])
    aoa = np.array([-90.0, 90.0])
    shape = (2, 2, 2)

    def field(st_value, amplitude_value):
        st = np.zeros(shape)
        amp = np.zeros(shape)
        phase = np.zeros(shape)
        st[:, :, 1] = st_value
        amp[:, :, 1] = amplitude_value
        return st, amp, phase

    cl = field(0.2, cl_amplitude)
    cd = field(0.3, cd_amplitude)
    zero = field(0.4, 0.0)
    return AirfoilFFT(
        name="synthetic",
        Re=re,
        AOA=aoa,
        Thickness=0.18,
        CL_ST=cl[0],
        CD_ST=cd[0],
        CM_ST=zero[0],
        CF_ST=zero[0],
        CL_Amp=cl[1],
        CD_Amp=cd[1],
        CM_Amp=zero[1],
        CF_Amp=zero[1],
        CL_Pha=cl[2],
        CD_Pha=cd[2],
        CM_Pha=zero[2],
        CF_Pha=zero[2],
    )


def _spectral_source(*, airfoil=None, mode=None, source_frequency_cap_hz=None):
    return HVAWTSpectralLockInSource(
        node_ids=("node",),
        airfoils_by_node=((airfoil or _synthetic_airfoil()),),
        modes=((mode or _mode()),),
        modal_participation=np.ones((1, 1)),
        n_frequency_entries=2,
        fluid_density_kgpm3=1.2,
        fluid_dynamic_viscosity_pas=1.8e-5,
        source_bounds_policy="raise",
        source_frequency_cap_hz=source_frequency_cap_hz,
    )


def _source_step(source, time_s, *, response_reduced_velocity, base_speed=6.0):
    mode = source.modes[0]
    width = 0.5
    response_speed = response_reduced_velocity * mode.natural_frequency_hz * width
    return source.step(
        time_s,
        base_relative_velocity_global_mps=np.array([[base_speed, 0.0, 0.0]]),
        response_effective_crossflow_speed_mps=np.array([response_speed]),
        response_section_width_m=np.array([width]),
        chord_unit_global=np.array([[1.0, 0.0, 0.0]]),
        normal_unit_global=np.array([[0.0, 1.0, 0.0]]),
        chord_m=np.array([0.5]),
        thickness_ratio=np.array([0.18]),
        local_length_m=np.array([1.0]),
        structural_velocity_global_mps=np.zeros((1, 3)),
    )


def _cylinder_partition(reduced_velocity, *, amplitude=0.3, modes=None):
    modes = (_mode(),) if modes is None else tuple(modes)
    frequency = modes[0].natural_frequency_hz
    width = 0.5
    speed = reduced_velocity * frequency * width
    return partition_lift_spectrum(
        strouhal=np.array([[0.0, CYLINDER_REFERENCE_STROUHAL]]),
        amplitude=np.array([[0.0, amplitude]]),
        phase_rad=np.zeros((1, 2)),
        crossflow_speed_mps=np.array([speed]),
        reference_width_m=np.array([width]),
        modes=modes,
    )


def test_default_frequency_ratios_are_exact_cylinder_mapping():
    assert DEFAULT_CAPTURE_FREQUENCY_RATIO_LOW == pytest.approx(
        CYLINDER_REFERENCE_STROUHAL * CYLINDER_REDUCED_VELOCITY_LOW,
        rel=0.0,
        abs=0.0,
    )
    assert DEFAULT_CAPTURE_FREQUENCY_RATIO_PEAK == pytest.approx(
        CYLINDER_REFERENCE_STROUHAL * CYLINDER_REDUCED_VELOCITY_PEAK,
        rel=0.0,
        abs=0.0,
    )
    assert DEFAULT_CAPTURE_FREQUENCY_RATIO_HIGH == pytest.approx(
        CYLINDER_REFERENCE_STROUHAL * CYLINDER_REDUCED_VELOCITY_HIGH,
        rel=0.0,
        abs=0.0,
    )


@pytest.mark.parametrize(
    ("reduced_velocity", "expected"),
    [
        (CYLINDER_REDUCED_VELOCITY_LOW, 0.0),
        (CYLINDER_REDUCED_VELOCITY_PEAK, 1.0),
        (CYLINDER_REDUCED_VELOCITY_HIGH, 0.0),
        (4.0, 0.0),
        (8.0, 0.0),
    ],
)
def test_single_cylinder_bin_reproduces_existing_activation(reduced_velocity, expected):
    result = _cylinder_partition(reduced_velocity)
    assert result.spectral_gain_by_node_mode[0, 0] == pytest.approx(expected)
    assert result.maximum_energy_closure_error <= 1.0e-15


def test_cylinder_rising_branch_matches_same_smoothstep():
    reduced_velocity = 0.5 * (
        CYLINDER_REDUCED_VELOCITY_LOW + CYLINDER_REDUCED_VELOCITY_PEAK
    )
    result = _cylinder_partition(reduced_velocity)
    expected = smooth_capture_amplitude(
        np.array([CYLINDER_REFERENCE_STROUHAL * reduced_velocity]),
        DEFAULT_CAPTURE_FREQUENCY_RATIO_LOW,
        DEFAULT_CAPTURE_FREQUENCY_RATIO_PEAK,
        DEFAULT_CAPTURE_FREQUENCY_RATIO_HIGH,
    )[0]
    assert result.spectral_gain_by_node_mode[0, 0] == pytest.approx(expected)


def test_schema_v2_uses_exact_engineering_sine_squared_capture():
    capture = SpectralLockInCaptureConfig(
        schema_version=2,
        capture_window="compact_sine_squared",
        reference_lift_amplitude=1.0,
        frequency_ratio_low=0.85,
        frequency_ratio_peak=1.0,
        frequency_ratio_high=1.1,
    )
    ratios = np.array([0.85, 0.925, 1.0, 1.05, 1.1])
    result = partition_lift_spectrum(
        strouhal=np.column_stack((np.zeros(ratios.size), ratios)),
        amplitude=np.column_stack((np.zeros(ratios.size), np.ones(ratios.size))),
        phase_rad=np.zeros((ratios.size, 2)),
        crossflow_speed_mps=np.ones(ratios.size),
        reference_width_m=np.ones(ratios.size),
        modes=(_mode(natural_frequency_hz=1.0),),
        capture=capture,
    )
    expected = capture_weight(ratios, low=0.85, peak=1.0, high=1.1)
    np.testing.assert_allclose(
        result.captured_amplitude_by_node_mode[:, 0], expected, atol=1.0e-15
    )
    np.testing.assert_allclose(
        result.spectral_gain_by_node_mode[:, 0], expected, atol=1.0e-15
    )
    assert result.maximum_energy_closure_error <= 1.0e-15


def test_capture_schema_and_window_are_bound_together():
    with pytest.raises(ValueError, match="schema 1 requires"):
        SpectralLockInCaptureConfig(capture_window="compact_sine_squared")
    with pytest.raises(ValueError, match="schema 2 requires"):
        SpectralLockInCaptureConfig(schema_version=2)


def test_off_resonant_spectrum_is_retained_exactly():
    st = np.array([[0.0, 0.1, 0.2, 0.4]])
    amp = np.array([[0.2, 0.4, 0.5, 0.6]])
    phase = np.array([[0.0, 0.1, -0.2, 0.3]])
    result = partition_lift_spectrum(
        strouhal=st,
        amplitude=amp,
        phase_rad=phase,
        crossflow_speed_mps=np.array([1.0]),
        reference_width_m=np.array([1.0]),
        modes=(_mode(natural_frequency_hz=10.0),),
    )
    np.testing.assert_array_equal(result.residual_amplitude, amp)
    np.testing.assert_array_equal(result.source_phase_rad, phase)
    assert result.spectral_gain_by_node_mode[0, 0] == 0.0


def test_full_capture_removes_harmonic_from_residual_without_touching_dc():
    result = _cylinder_partition(CYLINDER_REDUCED_VELOCITY_PEAK)
    assert result.residual_amplitude[0, 0] == 0.0
    assert result.residual_amplitude[0, 1] == pytest.approx(0.0, abs=1.0e-15)
    assert result.total_source_energy_by_node[0] == pytest.approx(0.3**2)
    assert result.total_residual_energy_by_node[0] == pytest.approx(0.0)
    assert result.total_captured_energy_by_node[0] == pytest.approx(0.3**2)


def test_dc_mean_is_never_captured_or_removed():
    fn = 2.0
    width = 0.5
    speed = CYLINDER_REDUCED_VELOCITY_PEAK * fn * width
    result = partition_lift_spectrum(
        strouhal=np.array([[0.0, 0.2]]),
        amplitude=np.array([[0.75, 0.3]]),
        phase_rad=np.array([[1.2, 0.0]]),
        crossflow_speed_mps=np.array([speed]),
        reference_width_m=np.array([width]),
        modes=(_mode(natural_frequency_hz=fn),),
    )
    assert result.residual_amplitude[0, 0] == 0.75
    assert result.capture_energy_fraction_by_mode[0, 0, 0] == 0.0


def test_two_mode_overlap_conserves_energy_and_uses_participation_only_for_allocation():
    modes = (
        _mode(6, natural_frequency_hz=2.0),
        _mode(12, natural_frequency_hz=2.0),
    )
    result = partition_lift_spectrum(
        strouhal=np.array([[0.0, 0.2], [0.0, 0.2]]),
        amplitude=np.array([[0.0, 0.3], [0.0, 0.3]]),
        phase_rad=np.zeros((2, 2)),
        crossflow_speed_mps=np.full(
            2,
            CYLINDER_REDUCED_VELOCITY_PEAK * 2.0 * 0.5,
        ),
        reference_width_m=np.full(2, 0.5),
        modes=modes,
        modal_participation=np.array([[1.0, 0.25], [0.25, 1.0]]),
    )
    fraction = result.capture_energy_fraction_by_mode[:, 1, :]
    np.testing.assert_allclose(np.sum(fraction, axis=1), 1.0)
    assert fraction[0, 0] > fraction[0, 1]
    assert fraction[1, 1] > fraction[1, 0]
    np.testing.assert_allclose(
        result.total_source_energy_by_node,
        result.total_residual_energy_by_node + result.total_captured_energy_by_node,
        rtol=0.0,
        atol=1.0e-15,
    )


def test_spectral_amplitude_scales_captured_gain_not_frequency_window():
    base = _cylinder_partition(CYLINDER_REDUCED_VELOCITY_PEAK, amplitude=0.3)
    energetic = _cylinder_partition(CYLINDER_REDUCED_VELOCITY_PEAK, amplitude=0.9)
    assert base.spectral_gain_by_node_mode[0, 0] == pytest.approx(1.0)
    assert energetic.spectral_gain_by_node_mode[0, 0] == pytest.approx(3.0)
    assert energetic.source_centroid_frequency_hz_by_node_mode[0, 0] == pytest.approx(
        base.source_centroid_frequency_hz_by_node_mode[0, 0]
    )


def test_modal_damping_balance_uses_known_structure_without_adding_damping():
    mode = _mode(structural_damping_ratio=0.01)
    partition = _cylinder_partition(CYLINDER_REDUCED_VELOCITY_PEAK, modes=(mode,))
    shape = np.zeros((1, 3, 1))
    shape[0, 0, 0] = 1.0
    balance = evaluate_modal_damping_balance(
        partition=partition,
        modes=(mode,),
        real_mode_shape=shape,
        section_dynamic_prefactor_nspm=np.array([100.0]),
    )[0]
    expected_structural = 2.0 * 0.01 * 100.0 * 2.0 * math.pi * 2.0
    assert balance.structural_damping_coefficient_nspm == pytest.approx(
        expected_structural
    )
    assert balance.captured_aerodynamic_damping_coefficient_nspm == pytest.approx(300.0)
    assert balance.linear_lockin_available is True
    expected_ratio = 0.3 * math.sqrt(1.0 - expected_structural / 300.0)
    assert balance.predicted_equilibrium_amplitude_ratio == pytest.approx(expected_ratio)
    assert balance.predicted_equilibrium_amplitude_m == pytest.approx(0.5 * expected_ratio)


def test_stateful_source_removes_captured_lift_and_retains_off_band_lift():
    captured_source = _spectral_source()
    captured = _source_step(
        captured_source,
        0.0,
        response_reduced_velocity=CYLINDER_REDUCED_VELOCITY_PEAK,
    )
    assert captured.partition.spectral_gain_by_node_mode[0, 0] == pytest.approx(1.0)
    assert captured.maximum_residual_nodal_force_n == pytest.approx(0.0, abs=1.0e-14)

    residual_source = _spectral_source()
    residual = _source_step(
        residual_source,
        0.0,
        response_reduced_velocity=4.0,
    )
    dynamic_force = 0.5 * 1.2 * 6.0**2 * 0.5 * 1.0
    assert residual.partition.spectral_gain_by_node_mode[0, 0] == 0.0
    assert residual.residual_force_global_n[0, 1] == pytest.approx(
        dynamic_force * CYLINDER_REFERENCE_LIFT_AMPLITUDE
    )
    assert residual.residual_force_global_n[0, 0] == pytest.approx(0.0)


def test_stateful_source_retains_drag_while_capturing_lift():
    source = _spectral_source(
        airfoil=_synthetic_airfoil(cl_amplitude=0.3, cd_amplitude=0.1)
    )
    result = _source_step(
        source,
        0.0,
        response_reduced_velocity=CYLINDER_REDUCED_VELOCITY_PEAK,
    )
    dynamic_force = 0.5 * 1.2 * 6.0**2 * 0.5 * 1.0
    assert result.residual_force_global_n[0, 0] == pytest.approx(
        dynamic_force * 0.1
    )
    assert result.residual_force_global_n[0, 1] == pytest.approx(0.0, abs=1.0e-14)


def test_residual_frequency_uses_base_flow_while_capture_uses_response_flow():
    source = _spectral_source()
    result = _source_step(
        source,
        0.0,
        response_reduced_velocity=4.0,
        base_speed=6.0,
    )
    state = source.checkpoint_state()

    capture_frequency = result.partition.source_frequency_hz[0, 1]
    residual_frequency = state["phase_frequency_cl_hz"][0][1]
    expected_capture = 0.2 * (4.0 * 2.0 * 0.5) / 0.5
    expected_residual = (
        0.2
        * 6.0
        / float(result.base_section_width_m[0])
    )
    assert capture_frequency == pytest.approx(expected_capture)
    assert residual_frequency == pytest.approx(expected_residual)
    assert residual_frequency != pytest.approx(capture_frequency)


def test_frequency_cap_reports_applied_and_discarded_residual_energy_exactly():
    source = _spectral_source(source_frequency_cap_hz=5.0)
    result = _source_step(
        source,
        0.0,
        response_reduced_velocity=4.0,
        base_speed=6.0,
    )
    source_energy = result.partition.total_source_energy_by_node[0]
    captured_energy = result.partition.total_captured_energy_by_node[0]
    applied_energy = result.applied_residual_lift_energy_by_node[0]
    truncated_energy = result.truncated_residual_lift_energy_by_node[0]

    assert result.residual_force_global_n[0, 1] == pytest.approx(0.0)
    assert applied_energy == pytest.approx(0.0)
    assert truncated_energy == pytest.approx(0.3**2)
    assert source_energy == pytest.approx(
        captured_energy + applied_energy + truncated_energy
    )


def test_stateful_source_is_phase_continuous_and_checkpoint_restart_exact():
    uninterrupted = _spectral_source(
        airfoil=_synthetic_airfoil(cl_amplitude=0.3, cd_amplitude=0.1)
    )
    _source_step(uninterrupted, 0.0, response_reduced_velocity=4.0)
    checkpoint = uninterrupted.checkpoint_state()
    expected = _source_step(uninterrupted, 0.05, response_reduced_velocity=4.0)

    restarted = _spectral_source(
        airfoil=_synthetic_airfoil(cl_amplitude=0.3, cd_amplitude=0.1)
    )
    restarted.restore_checkpoint_state(checkpoint)
    observed = _source_step(restarted, 0.05, response_reduced_velocity=4.0)
    np.testing.assert_allclose(
        observed.residual_force_global_n,
        expected.residual_force_global_n,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        observed.partition.spectral_gain_by_node_mode,
        expected.partition.spectral_gain_by_node_mode,
        rtol=0.0,
        atol=0.0,
    )


def test_stateful_source_duplicate_is_exact_and_changed_same_time_rejected():
    source = _spectral_source()
    first = _source_step(source, 0.0, response_reduced_velocity=4.0)
    duplicate = _source_step(source, 0.0, response_reduced_velocity=4.0)
    assert duplicate is first
    assert not duplicate.residual_force_global_n.flags.writeable
    np.testing.assert_array_equal(
        duplicate.residual_force_global_n,
        first.residual_force_global_n,
    )
    with pytest.raises(ValueError, match="changed at the same time"):
        _source_step(
            source,
            0.0,
            response_reduced_velocity=4.0,
            base_speed=6.1,
        )


def test_stateful_source_checkpoint_rejects_identity_and_shape_mutations():
    source = _spectral_source()
    _source_step(source, 0.0, response_reduced_velocity=4.0)
    state = source.checkpoint_state()
    bad_identity = dict(state)
    bad_identity["config_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        _spectral_source().restore_checkpoint_state(bad_identity)
    bad_shape = dict(state)
    bad_shape["phase_angle_cl_rad"] = []
    with pytest.raises(ValueError, match="phase_angle_cl_rad is invalid"):
        _spectral_source().restore_checkpoint_state(bad_shape)


def test_source_coverage_counts_distinct_wrapper_steps_at_same_response_time():
    source = _spectral_source()
    source.sample_period_s = 0.05
    result = _source_step(source, 0.0, response_reduced_velocity=4.0)
    zero = np.zeros((1, 3))
    residual = np.asarray(result.residual_force_global_n)
    source.commit_source_coverage(
        sample_index=0,
        applied_residual_force_global_n=residual,
        response_force_global_n=zero,
        total_force_global_n=residual,
        structural_velocity_global_mps=zero,
    )
    duplicate = _source_step(source, 0.0, response_reduced_velocity=4.0)
    assert duplicate is result
    source.commit_source_coverage(
        sample_index=1,
        applied_residual_force_global_n=residual,
        response_force_global_n=zero,
        total_force_global_n=residual,
        structural_velocity_global_mps=zero,
    )

    rows = source.source_coverage_summary()
    assert len(rows) == 1
    assert tuple(rows[0]) == SOURCE_COVERAGE_COLUMNS
    assert rows[0]["samples"] == 2
    assert rows[0]["node_samples"] == 2
    assert rows[0]["max_nodal_force_n"] > 0.0
    assert rows[0]["max_response_nodal_force_n"] == 0.0
    assert rows[0]["max_total_nodal_force_n"] == rows[0]["max_nodal_force_n"]


def test_source_coverage_checkpoint_restore_is_exact_and_atomic():
    source = _spectral_source()
    source.sample_period_s = 0.05
    result = _source_step(source, 0.0, response_reduced_velocity=4.0)
    residual = np.asarray(result.residual_force_global_n)
    zero = np.zeros((1, 3))
    source.commit_source_coverage(
        sample_index=0,
        applied_residual_force_global_n=residual,
        response_force_global_n=zero,
        total_force_global_n=residual,
        structural_velocity_global_mps=zero,
    )
    checkpoint = source.checkpoint_state()
    restarted = _spectral_source()
    restarted.sample_period_s = 0.05
    restarted.restore_checkpoint_state(copy.deepcopy(checkpoint))
    assert restarted.checkpoint_state() == checkpoint
    assert restarted.source_coverage_summary() == source.source_coverage_summary()

    before = restarted.checkpoint_state()
    malformed = copy.deepcopy(checkpoint)
    malformed["coverage_rows"][0]["node_samples"] += 1
    with pytest.raises(ValueError, match="node_samples"):
        restarted.restore_checkpoint_state(malformed)
    assert restarted.checkpoint_state() == before


def test_real_naca0018_spectra_drive_different_finite_captured_energy():
    airfoil = load_airfoil_fft(str(REPO_ROOT / "data" / "airfoils" / "NACA0018.h5"))
    reynolds = np.array([500_000.0, 500_000.0])
    aoa = np.array([0.0, 15.0])
    st, amp, phase = interpolate_fft_spectrum_batch(
        airfoil,
        reynolds,
        aoa,
        "CL",
        bounds_policy="raise",
    )
    harmonic_energy = amp[:, 1:] ** 2
    dominant = 1 + np.argmax(harmonic_energy, axis=1)
    dominant_st = st[np.arange(2), dominant]
    fn = 2.5
    width = np.full(2, 0.5)
    speed = (
        DEFAULT_CAPTURE_FREQUENCY_RATIO_PEAK * fn * width / dominant_st
    )
    result = partition_lift_spectrum(
        strouhal=st,
        amplitude=amp,
        phase_rad=phase,
        crossflow_speed_mps=speed,
        reference_width_m=width,
        modes=(_mode(natural_frequency_hz=fn),),
    )
    assert np.all(np.isfinite(result.spectral_gain_by_node_mode))
    assert np.all(result.spectral_gain_by_node_mode > 0.0)
    assert result.spectral_gain_by_node_mode[0, 0] != pytest.approx(
        result.spectral_gain_by_node_mode[1, 0],
        rel=1.0e-3,
    )
    np.testing.assert_allclose(
        result.total_source_energy_by_node,
        result.total_residual_energy_by_node + result.total_captured_energy_by_node,
        rtol=0.0,
        atol=2.0e-12,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"strouhal": np.array([[0.1, 0.2]])},
        {"amplitude": np.array([[0.0, -0.3]])},
        {"phase_rad": np.array([[0.0, np.nan]])},
        {"crossflow_speed_mps": np.array([-1.0])},
        {"reference_width_m": np.array([0.0])},
    ],
)
def test_invalid_spectral_inputs_fail_closed(mutation):
    values = dict(
        strouhal=np.array([[0.0, 0.2]]),
        amplitude=np.array([[0.0, 0.3]]),
        phase_rad=np.zeros((1, 2)),
        crossflow_speed_mps=np.array([6.0]),
        reference_width_m=np.array([0.5]),
        modes=(_mode(),),
    )
    values.update(mutation)
    with pytest.raises((TypeError, ValueError)):
        partition_lift_spectrum(**values)


def test_maximum_spectral_gain_is_fail_closed_not_silently_clipped():
    with pytest.raises(ValueError, match="exceeds maximum_spectral_gain"):
        _cylinder_partition(CYLINDER_REDUCED_VELOCITY_PEAK, amplitude=3.3)


def test_config_rejects_boolean_schema_and_inverted_window():
    with pytest.raises(ValueError, match="integer 1"):
        SpectralLockInCaptureConfig(schema_version=True)
    with pytest.raises(ValueError, match="low < peak < high"):
        SpectralLockInCaptureConfig(
            frequency_ratio_low=1.2,
            frequency_ratio_peak=1.1,
        )


def test_reference_amplitude_is_the_frozen_cylinder_value():
    config = SpectralLockInCaptureConfig()
    assert config.reference_lift_amplitude == CYLINDER_REFERENCE_LIFT_AMPLITUDE
