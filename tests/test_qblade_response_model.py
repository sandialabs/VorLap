from dataclasses import fields, replace
import copy
import json
import math

import numpy as np
import pytest

from vorlap.qblade_response_model import (
    AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW,
    QBladeResponseDiagnostics,
    QBladeResponseModel,
    QBladeResponseModelConfig,
)


def _config(**overrides):
    config = QBladeResponseModelConfig(
        node_ids=("root",),
        mode_shape=(1.0,),
        natural_frequency_hz=1.0,
        reference_length_m=1.0,
        reduced_velocity_low=4.0,
        reduced_velocity_peak=5.0,
        reduced_velocity_high=8.0,
        aerodynamic_damping_gain_h1=2.0,
        saturation_amplitude_ratio=0.1,
        rms_time_constant_s=1.0,
        max_abs_gain=100.0,
        max_abs_nodal_force_n=1.0e6,
        bandpass_damping_ratio=0.2,
    )
    return replace(config, **overrides)


def _step_inputs(
    *,
    node_ids=("root",),
    structural_velocity=((0.1, 0.0, 0.0),),
    effective_velocity=5.0,
    section_width=0.2,
    local_length=1.5,
    density=1.2,
):
    return {
        "node_ids": node_ids,
        "direction_basis": np.tile([1.0, 0.0, 0.0], (len(node_ids), 1)),
        "structural_velocity_mps": np.asarray(structural_velocity, dtype=float),
        "effective_velocity_mps": effective_velocity,
        "section_width_m": section_width,
        "local_length_m": local_length,
        "fluid_density_kgpm3": density,
    }


def _run_one_node_signal(model, signal, *, dt_s, duration_s):
    results = []
    n_steps = int(round(duration_s / dt_s))
    for index in range(n_steps + 1):
        time_s = index * dt_s
        velocity_mps = float(signal(time_s))
        results.append(
            model.step(
                time_s,
                **_step_inputs(
                    structural_velocity=((velocity_mps, 0.0, 0.0),)
                ),
            )
        )
    return results


def _measure_bandpass(frequency_hz, dt_s):
    model = QBladeResponseModel(
        _config(
            enabled=True,
            aerodynamic_damping_gain_h1=0.0,
            saturation_amplitude_ratio=100.0,
        )
    )
    duration_s = max(12.0, 6.0 / frequency_hz)
    results = _run_one_node_signal(
        model,
        lambda time_s: math.sin(2.0 * math.pi * frequency_hz * time_s),
        dt_s=dt_s,
        duration_s=duration_s,
    )
    time = np.arange(len(results), dtype=float) * dt_s
    filtered = np.asarray(
        [result.diagnostics.filtered_qdot_mps for result in results],
        dtype=float,
    )
    window_start = duration_s - min(4.0 / frequency_hz, 4.0)
    keep = time >= window_start
    angle = 2.0 * math.pi * frequency_hz * time[keep]
    design = np.column_stack((np.sin(angle), np.cos(angle), np.ones_like(angle)))
    sine_coefficient, cosine_coefficient, _ = np.linalg.lstsq(
        design,
        filtered[keep],
        rcond=None,
    )[0]
    gain = math.hypot(sine_coefficient, cosine_coefficient)
    phase_rad = math.atan2(cosine_coefficient, sine_coefficient)
    return gain, phase_rad


def test_model_is_disabled_by_default_and_zero_state_returns_zero_force():
    disabled = QBladeResponseModel(_config())
    disabled_result = disabled.step(0.0, **_step_inputs())

    assert disabled.config.enabled is False
    assert disabled_result.diagnostics.enabled is False
    assert disabled_result.diagnostics.raw_qdot_mps == pytest.approx(0.1)
    assert disabled_result.diagnostics.filtered_qdot_mps == 0.0
    assert disabled_result.diagnostics.projection_weighting == "uniform_default"
    np.testing.assert_array_equal(disabled_result.additive_force_n, np.zeros((1, 3)))

    enabled = QBladeResponseModel(_config(enabled=True))
    zero_result = enabled.step(
        0.0,
        **_step_inputs(structural_velocity=((0.0, 0.0, 0.0),)),
    )
    assert isinstance(zero_result.diagnostics, QBladeResponseDiagnostics)
    assert zero_result.diagnostics.raw_qdot_mps == 0.0
    assert zero_result.diagnostics.qdot_mps == 0.0
    assert zero_result.diagnostics.qdot_rms_mps == 0.0
    assert zero_result.diagnostics.modal_power_w == 0.0
    np.testing.assert_array_equal(zero_result.additive_force_n, np.zeros((1, 3)))


def test_force_adds_energy_below_saturation_at_natural_frequency():
    model = QBladeResponseModel(
        _config(enabled=True, rms_time_constant_s=0.3)
    )
    results = _run_one_node_signal(
        model,
        lambda time_s: 0.1 * math.sin(2.0 * math.pi * time_s),
        dt_s=0.01,
        duration_s=8.0,
    )
    settled = results[-100:]

    assert max(result.diagnostics.amplitude_ratio for result in settled) < 0.1
    assert min(result.diagnostics.saturation_factor for result in settled) > 0.0
    assert np.mean([result.diagnostics.modal_power_w for result in settled]) > 0.0
    assert np.mean([result.diagnostics.structural_power_w for result in settled]) > 0.0


def test_force_removes_energy_above_saturation_at_natural_frequency():
    model = QBladeResponseModel(
        _config(enabled=True, rms_time_constant_s=0.2)
    )
    results = _run_one_node_signal(
        model,
        lambda time_s: 1.0 * math.sin(2.0 * math.pi * time_s),
        dt_s=0.01,
        duration_s=8.0,
    )
    settled = results[-100:]

    assert min(result.diagnostics.amplitude_ratio for result in settled) > 0.1
    assert max(result.diagnostics.saturation_factor for result in settled) < 0.0
    assert np.mean([result.diagnostics.modal_power_w for result in settled]) < 0.0
    assert np.mean([result.diagnostics.structural_power_w for result in settled]) < 0.0


@pytest.mark.parametrize(
    ("reduced_velocity", "expected_activation"),
    [
        (3.0, 0.0),
        (4.0, 0.0),
        (4.5, 0.5),
        (5.0, 1.0),
        (6.5, 0.5),
        (8.0, 0.0),
        (9.0, 0.0),
    ],
)
def test_smooth_reduced_velocity_activation(reduced_velocity, expected_activation):
    model = QBladeResponseModel(_config(enabled=True))
    model.step(
        0.0,
        **_step_inputs(
            structural_velocity=((0.0, 0.0, 0.0),),
            effective_velocity=reduced_velocity,
        ),
    )
    result = model.step(
        0.25,
        **_step_inputs(
            structural_velocity=((0.1, 0.0, 0.0),),
            effective_velocity=reduced_velocity,
        ),
    )

    assert result.diagnostics.activation_min == pytest.approx(expected_activation)
    assert result.diagnostics.activation_max == pytest.approx(expected_activation)
    if expected_activation == 0.0:
        np.testing.assert_array_equal(result.additive_force_n, np.zeros((1, 3)))
    else:
        assert np.max(np.abs(result.additive_force_n)) > 0.0


def test_weighted_projection_uses_positive_configured_weights_and_diagnoses_choice():
    weighted = QBladeResponseModel(
        _config(
            node_ids=("root", "tip"),
            mode_shape=(0.5, 1.0),
            projection_weights=(1.0, 3.0),
            enabled=True,
        )
    )
    result = weighted.step(
        0.0,
        **_step_inputs(
            node_ids=("root", "tip"),
            structural_velocity=((2.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ),
    )

    expected_qdot = (1.0 * 0.5 * 2.0) / (1.0 * 0.5**2 + 3.0 * 1.0**2)
    assert result.diagnostics.raw_qdot_mps == pytest.approx(expected_qdot)
    assert result.diagnostics.filtered_qdot_mps == 0.0
    assert result.diagnostics.projection_weighting == "configured"
    assert result.diagnostics.projection_weight_min == 1.0
    assert result.diagnostics.projection_weight_max == 3.0


@pytest.mark.parametrize(
    "weights",
    [(1.0,), (0.0, 1.0), (-1.0, 1.0), (math.nan, 1.0)],
)
def test_projection_weights_must_match_nodes_and_be_positive(weights):
    with pytest.raises(ValueError, match="projection_weights"):
        QBladeResponseModel(
            _config(
                node_ids=("root", "tip"),
                mode_shape=(0.5, 1.0),
                projection_weights=weights,
            )
        )


def test_node_order_mode_scale_sign_and_weight_mapping_do_not_change_force():
    common = dict(
        natural_frequency_hz=1.0,
        reference_length_m=1.0,
        reduced_velocity_low=4.0,
        reduced_velocity_peak=5.0,
        reduced_velocity_high=8.0,
        aerodynamic_damping_gain_h1=2.0,
        saturation_amplitude_ratio=0.2,
        rms_time_constant_s=0.5,
        max_abs_gain=100.0,
        max_abs_nodal_force_n=1.0e6,
        bandpass_damping_ratio=0.2,
        enabled=True,
    )
    canonical_model = QBladeResponseModel(
        QBladeResponseModelConfig(
            node_ids=("root", "tip"),
            mode_shape=(0.5, 1.0),
            projection_weights=(2.0, 3.0),
            **common,
        )
    )
    reordered_scaled_model = QBladeResponseModel(
        QBladeResponseModelConfig(
            node_ids=("tip", "root"),
            mode_shape=(-10.0, -5.0),
            projection_weights=(3.0, 2.0),
            **common,
        )
    )

    canonical = reordered = None
    canonical_inputs = None
    for index in range(401):
        time_s = index * 0.01
        qdot_mps = 0.2 * math.sin(2.0 * math.pi * time_s)
        canonical_inputs = _step_inputs(
            node_ids=("root", "tip"),
            structural_velocity=(
                (0.5 * qdot_mps, 0.0, 0.0),
                (qdot_mps, 0.0, 0.0),
            ),
            effective_velocity=np.array([5.0, 5.0]),
            section_width=np.array([0.2, 0.3]),
            local_length=np.array([1.0, 1.5]),
        )
        canonical = canonical_model.step(time_s, **canonical_inputs)
        reordered = reordered_scaled_model.step(
            time_s,
            **_step_inputs(
                node_ids=("tip", "root"),
                structural_velocity=(
                    (qdot_mps, 0.0, 0.0),
                    (0.5 * qdot_mps, 0.0, 0.0),
                ),
                effective_velocity=np.array([5.0, 5.0]),
                section_width=np.array([0.3, 0.2]),
                local_length=np.array([1.5, 1.0]),
            ),
        )

    assert canonical is not None and reordered is not None and canonical_inputs is not None
    np.testing.assert_allclose(canonical.additive_force_n, reordered.additive_force_n[::-1])
    assert canonical.diagnostics.raw_qdot_mps == pytest.approx(
        reordered.diagnostics.raw_qdot_mps
    )
    assert canonical.diagnostics.filtered_qdot_mps == pytest.approx(
        reordered.diagnostics.filtered_qdot_mps
    )
    assert canonical.diagnostics.modal_power_w == pytest.approx(
        reordered.diagnostics.modal_power_w
    )

    # Equivalent node ordering, mode scaling, and mode sign share checkpoint
    # identity because the internal mode orientation is canonical.
    restored_equivalent = QBladeResponseModel(reordered_scaled_model.config)
    restored_equivalent.restore_checkpoint_state(canonical_model.checkpoint_state())
    duplicate = restored_equivalent.step(4.0, **canonical_inputs)
    np.testing.assert_allclose(duplicate.additive_force_n, canonical.additive_force_n)


def test_constant_modal_velocity_is_rejected_as_dc_drift():
    model = QBladeResponseModel(_config(enabled=True))
    results = _run_one_node_signal(
        model,
        lambda _time_s: 2.0,
        dt_s=0.1,
        duration_s=10.0,
    )
    final = results[-1]

    assert final.diagnostics.raw_qdot_mps == pytest.approx(2.0)
    assert abs(final.diagnostics.filtered_qdot_mps) < 1.0e-12
    assert final.diagnostics.qdot_rms_mps < 1.0e-12
    assert final.diagnostics.max_abs_nodal_force_n < 1.0e-12


def test_slow_modal_velocity_drift_is_strongly_attenuated():
    model = QBladeResponseModel(_config(enabled=True))
    results = _run_one_node_signal(
        model,
        lambda time_s: 0.01 * time_s,
        dt_s=0.05,
        duration_s=20.0,
    )
    final = results[-1]

    assert final.diagnostics.raw_qdot_mps == pytest.approx(0.2)
    assert abs(final.diagnostics.filtered_qdot_mps) < 0.002
    assert final.diagnostics.qdot_rms_mps < 0.002


def test_bandpass_has_unity_gain_and_zero_phase_at_natural_frequency():
    gain, phase_rad = _measure_bandpass(1.0, 0.01)

    assert gain == pytest.approx(1.0, abs=5.0e-4)
    assert phase_rad == pytest.approx(0.0, abs=5.0e-4)


@pytest.mark.parametrize("frequency_hz", [0.1, 5.0])
def test_bandpass_attenuates_off_frequency_motion(frequency_hz):
    gain, _ = _measure_bandpass(frequency_hz, 0.01)
    damping = 0.2
    frequency_ratio = frequency_hz / 1.0
    expected_gain = (
        2.0 * damping * frequency_ratio
        / math.sqrt(
            (1.0 - frequency_ratio**2) ** 2
            + (2.0 * damping * frequency_ratio) ** 2
        )
    )

    assert gain < 0.1
    assert gain == pytest.approx(expected_gain, rel=0.03, abs=5.0e-4)


def test_bandpass_frequency_response_converges_with_timestep():
    coarse_gain, coarse_phase = _measure_bandpass(1.0, 0.02)
    fine_gain, fine_phase = _measure_bandpass(1.0, 0.01)

    assert coarse_gain == pytest.approx(fine_gain, abs=0.002)
    assert coarse_phase == pytest.approx(fine_phase, abs=0.002)


def test_duplicate_step_is_idempotent_and_returns_protected_copy():
    model = QBladeResponseModel(_config(enabled=True))
    inputs = _step_inputs()
    first = model.step(0.0, **inputs)
    state_before_duplicate = model.checkpoint_state()
    first.additive_force_n[:] = 999.0

    duplicate = model.step(0.0, **inputs)

    assert model.checkpoint_state() == state_before_duplicate
    expected_diagnostics = QBladeResponseDiagnostics(
        **state_before_duplicate["last_result"]["diagnostics"]
    )
    assert duplicate.diagnostics == expected_diagnostics
    assert duplicate.additive_force_n[0, 0] != 999.0


def test_same_time_changed_inputs_and_decreasing_time_fail_without_state_change():
    model = QBladeResponseModel(_config(enabled=True))
    model.step(1.0, **_step_inputs())
    accepted_state = model.checkpoint_state()

    with pytest.raises(ValueError, match="changed at the same time"):
        model.step(
            1.0,
            **_step_inputs(structural_velocity=((0.2, 0.0, 0.0),)),
        )
    assert model.checkpoint_state() == accepted_state

    with pytest.raises(ValueError, match="time decreased"):
        model.step(0.5, **_step_inputs())
    assert model.checkpoint_state() == accepted_state


def test_checkpoint_json_roundtrip_preserves_filter_and_future_step():
    config = _config(enabled=True, rms_time_constant_s=0.4)
    original = QBladeResponseModel(config)
    accepted = None
    for index in range(76):
        time_s = index * 0.02
        velocity = 0.3 * math.sin(2.0 * math.pi * time_s)
        accepted = original.step(
            time_s,
            **_step_inputs(structural_velocity=((velocity, 0.0, 0.0),)),
        )
    assert accepted is not None
    checkpoint = json.loads(json.dumps(original.checkpoint_state()))
    assert checkpoint["bandpass_filter"]["previous_input_mps"] is not None
    assert abs(checkpoint["bandpass_filter"]["rate_state"]) > 0.0

    restored = QBladeResponseModel(config)
    restored.restore_checkpoint_state(checkpoint)
    duplicate_velocity = 0.3 * math.sin(2.0 * math.pi * 1.5)
    duplicate_inputs = _step_inputs(
        structural_velocity=((duplicate_velocity, 0.0, 0.0),)
    )
    duplicate = restored.step(1.5, **duplicate_inputs)
    np.testing.assert_allclose(duplicate.additive_force_n, accepted.additive_force_n)
    assert duplicate.diagnostics == accepted.diagnostics

    future_time = 1.57
    future_velocity = 0.3 * math.sin(2.0 * math.pi * future_time)
    future_inputs = _step_inputs(
        structural_velocity=((future_velocity, 0.0, 0.0),)
    )
    original_future = original.step(future_time, **future_inputs)
    restored_future = restored.step(future_time, **future_inputs)
    np.testing.assert_allclose(restored_future.additive_force_n, original_future.additive_force_n)
    assert restored_future.diagnostics == original_future.diagnostics
    assert restored.checkpoint_state() == original.checkpoint_state()


def test_exact_ema_uses_filtered_endpoint_velocity_analytic_update():
    tau_s = 2.0
    dt_s = 0.25
    model = QBladeResponseModel(
        _config(enabled=True, rms_time_constant_s=tau_s)
    )
    model.step(
        0.0,
        **_step_inputs(structural_velocity=((0.0, 0.0, 0.0),)),
    )
    result = model.step(
        dt_s,
        **_step_inputs(structural_velocity=((1.0, 0.0, 0.0),)),
    )

    filtered = result.diagnostics.filtered_qdot_mps
    expected_mean_square = filtered**2 * (1.0 - math.exp(-dt_s / tau_s))
    assert result.diagnostics.qdot_rms_mps == pytest.approx(
        math.sqrt(expected_mean_square)
    )
    assert model.checkpoint_state()["qdot_mean_square_m2ps2"] == pytest.approx(
        expected_mean_square
    )


def test_hard_gain_and_nodal_force_limits_are_applied_and_reported():
    model = QBladeResponseModel(
        _config(
            enabled=True,
            aerodynamic_damping_gain_h1=10.0,
            saturation_amplitude_ratio=100.0,
            max_abs_gain=0.25,
            max_abs_nodal_force_n=0.2,
        )
    )
    results = _run_one_node_signal(
        model,
        lambda time_s: 4.0 * math.sin(2.0 * math.pi * time_s),
        dt_s=0.01,
        duration_s=4.25,
    )
    result = results[-1]

    assert result.diagnostics.raw_gain_max == pytest.approx(10.0, rel=1.0e-4)
    assert result.diagnostics.applied_gain_max == pytest.approx(0.25)
    assert result.diagnostics.gain_limited_nodes == 1
    assert result.diagnostics.force_limited_nodes == 1
    assert result.additive_force_n[0, 0] == pytest.approx(0.2)
    assert result.diagnostics.max_abs_nodal_force_n == pytest.approx(0.2)


def test_force_direction_and_filter_configuration_are_validated_and_checkpointed():
    normal = QBladeResponseModel(_config())
    chord = QBladeResponseModel(_config(force_direction="chord"))
    wider_filter = QBladeResponseModel(_config(bandpass_damping_ratio=0.3))

    assert normal.config.force_direction == "normal"
    assert chord.config.force_direction == "chord"
    assert (
        normal.checkpoint_state()["config_fingerprint"]
        != chord.checkpoint_state()["config_fingerprint"]
    )
    assert (
        normal.checkpoint_state()["config_fingerprint"]
        != wider_filter.checkpoint_state()["config_fingerprint"]
    )
    with pytest.raises(ValueError, match="force_direction"):
        QBladeResponseModel(_config(force_direction="span"))
    with pytest.raises(ValueError, match="bandpass_damping_ratio"):
        QBladeResponseModel(_config(bandpass_damping_ratio=0.0))
    with pytest.raises(ValueError, match="bandpass_damping_ratio"):
        QBladeResponseModel(_config(bandpass_damping_ratio=1.0))


def test_schema_v2_uses_one_declared_ambient_speed_for_activation_and_prefactor():
    model = QBladeResponseModel(
        _config(
            enabled=True,
            schema_version=2,
            aerodynamic_speed_basis=(
                AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW
            ),
            reduced_velocity_low=0.0,
            reduced_velocity_peak=5.0,
            reduced_velocity_high=10.0,
            saturation_amplitude_ratio=100.0,
        )
    )
    common = dict(
        effective_velocity_mps=5.0,
        declared_ambient_crossflow_speed_mps=5.0,
        relative_crossflow_speed_mps=3.0,
        response_section_width_basis="declared_ambient_crossflow_angle",
    )
    model.step(
        0.0,
        **{
            **_step_inputs(structural_velocity=((0.0, 0.0, 0.0),)),
            **common,
        },
    )
    result = model.step(
        0.25,
        **{
            **_step_inputs(structural_velocity=((0.1, 0.0, 0.0),)),
            **common,
        },
    )

    diagnostics = result.diagnostics
    assert diagnostics.aerodynamic_speed_basis == "declared_ambient_crossflow"
    assert diagnostics.response_declared_ambient_crossflow_speed_min_mps == 5.0
    assert diagnostics.response_declared_ambient_crossflow_speed_max_mps == 5.0
    assert diagnostics.response_relative_crossflow_speed_min_mps == 3.0
    assert diagnostics.response_relative_crossflow_speed_max_mps == 3.0
    assert diagnostics.configured_response_reduced_velocity_min == 5.0
    assert diagnostics.realized_response_reduced_velocity_min == 5.0
    assert diagnostics.activation_min == 1.0
    assert diagnostics.speed_basis_invariant_violations == 0
    assert diagnostics.activation_invariant_violations == 0
    expected_force = (
        0.5
        * 1.2
        * 5.0
        * 0.2
        * 1.5
        * diagnostics.applied_gain_min
        * diagnostics.filtered_qdot_mps
    )
    assert result.additive_force_n[0, 0] == pytest.approx(expected_force)


def test_schema_v2_spectral_activation_override_replaces_only_lockin_window():
    model = QBladeResponseModel(
        _config(
            enabled=True,
            schema_version=2,
            aerodynamic_speed_basis="declared_ambient_crossflow",
            reduced_velocity_low=100.0,
            reduced_velocity_peak=101.0,
            reduced_velocity_high=102.0,
            saturation_amplitude_ratio=100.0,
        )
    )
    common = dict(
        effective_velocity_mps=5.0,
        declared_ambient_crossflow_speed_mps=5.0,
        relative_crossflow_speed_mps=3.0,
        response_section_width_basis="declared_ambient_crossflow_angle",
        activation_override=0.25,
    )
    model.step(
        0.0,
        **{
            **_step_inputs(structural_velocity=((0.0, 0.0, 0.0),)),
            **common,
        },
    )
    result = model.step(
        0.25,
        **{
            **_step_inputs(structural_velocity=((0.1, 0.0, 0.0),)),
            **common,
        },
    )
    assert result.diagnostics.activation_min == pytest.approx(0.25)
    assert result.diagnostics.activation_max == pytest.approx(0.25)
    assert result.diagnostics.configured_response_reduced_velocity_min == 5.0
    assert np.max(np.abs(result.additive_force_n)) > 0.0

    with pytest.raises(ValueError, match="activation_override"):
        QBladeResponseModel(_config(enabled=True)).step(
            0.0, **_step_inputs(), activation_override=0.5
        )


def test_schema_v2_requires_explicit_ambient_inputs_and_rejects_mismatch_atomically():
    config = _config(
        enabled=True,
        schema_version=2,
        aerodynamic_speed_basis="declared_ambient_crossflow",
    )
    with pytest.raises(ValueError, match="declared_ambient_crossflow_speed_mps"):
        QBladeResponseModel(config).step(0.0, **_step_inputs())

    model = QBladeResponseModel(config)
    model.step(
        0.0,
        **_step_inputs(effective_velocity=5.0),
        declared_ambient_crossflow_speed_mps=5.0,
        relative_crossflow_speed_mps=3.0,
        response_section_width_basis="declared_ambient_crossflow_angle",
    )
    before = copy.deepcopy(model.checkpoint_state())
    with pytest.raises(ValueError, match="must exactly equal"):
        model.step(
            0.25,
            **_step_inputs(effective_velocity=3.0),
            declared_ambient_crossflow_speed_mps=5.0,
            relative_crossflow_speed_mps=3.0,
            response_section_width_basis="declared_ambient_crossflow_angle",
        )
    assert model.checkpoint_state() == before


def test_schema_and_speed_basis_are_fail_closed_and_reject_cross_version_restore():
    with pytest.raises(ValueError, match="aerodynamic_speed_basis"):
        QBladeResponseModel(_config(schema_version=2))
    with pytest.raises(ValueError, match="aerodynamic_speed_basis"):
        QBladeResponseModel(
            _config(
                schema_version=1,
                aerodynamic_speed_basis="declared_ambient_crossflow",
            )
        )

    legacy = QBladeResponseModel(_config())
    ambient = QBladeResponseModel(
        _config(
            schema_version=2,
            aerodynamic_speed_basis="declared_ambient_crossflow",
        )
    )
    assert legacy.config_fingerprint != ambient.config_fingerprint
    with pytest.raises(ValueError, match="configuration mismatch"):
        ambient.restore_checkpoint_state(legacy.checkpoint_state())


def test_explicit_response_width_is_typed_and_fingerprinted():
    explicit = QBladeResponseModel(
        _config(
            schema_version=2,
            aerodynamic_speed_basis="declared_ambient_crossflow",
            response_section_width_m=0.05,
        )
    )
    baseline = QBladeResponseModel(
        _config(
            schema_version=2,
            aerodynamic_speed_basis="declared_ambient_crossflow",
        )
    )
    assert explicit.response_section_width_basis == "explicit_response_section_width_m"
    np.testing.assert_array_equal(explicit.response_section_width_m, [0.05])
    assert explicit.config_fingerprint != baseline.config_fingerprint
    with pytest.raises(ValueError, match="response_section_width_m"):
        QBladeResponseModel(_config(response_section_width_m=0.0))
    with pytest.raises(TypeError, match="not boolean"):
        QBladeResponseModel(
            _config(
                schema_version=2,
                aerodynamic_speed_basis="declared_ambient_crossflow",
                response_section_width_m=(True,),
            )
        )


def test_schema_v1_fingerprint_signature_diagnostics_and_checkpoint_are_frozen():
    expected_fields = (
        "enabled",
        "time_s",
        "dt_s",
        "reduced_velocity_min",
        "reduced_velocity_max",
        "activation_min",
        "activation_max",
        "raw_qdot_mps",
        "filtered_qdot_mps",
        "qdot_mps",
        "qdot_rms_mps",
        "projection_weighting",
        "projection_weight_min",
        "projection_weight_max",
        "amplitude_estimate_m",
        "amplitude_ratio",
        "saturation_factor",
        "raw_gain_min",
        "raw_gain_max",
        "applied_gain_min",
        "applied_gain_max",
        "gain_limited_nodes",
        "force_limited_nodes",
        "modal_power_w",
        "structural_power_w",
        "max_abs_nodal_force_n",
    )
    assert tuple(field.name for field in fields(QBladeResponseDiagnostics)) == (
        expected_fields
    )

    legacy = QBladeResponseModel(_config(enabled=True))
    with_v2_only_kwargs = QBladeResponseModel(_config(enabled=True))
    assert legacy.config_fingerprint == (
        "30830e795b548bdc32e1a7be0dcb919371632c45d72e51e76746da87eb68edc2"
    )
    legacy_result = legacy.step(0.0, **_step_inputs())
    compatibility_result = with_v2_only_kwargs.step(
        0.0,
        **_step_inputs(),
        declared_ambient_crossflow_speed_mps=999.0,
        relative_crossflow_speed_mps=0.0,
        response_section_width_basis="ignored-v2-only-input",
    )
    assert type(legacy_result.diagnostics) is QBladeResponseDiagnostics
    np.testing.assert_array_equal(
        compatibility_result.additive_force_n,
        legacy_result.additive_force_n,
    )
    assert compatibility_result.diagnostics == legacy_result.diagnostics
    assert with_v2_only_kwargs.checkpoint_state() == legacy.checkpoint_state()

    checkpoint = legacy.checkpoint_state()
    assert tuple(checkpoint) == (
        "schema",
        "config_fingerprint",
        "last_time_s",
        "bandpass_filter",
        "qdot_mean_square_m2ps2",
        "last_input_signature",
        "last_result",
    )
    assert checkpoint["last_input_signature"] == (
        "f526ef3864c91d2b5e28a27f73c14d541a9719552ea96963abd707e8903ca1cd"
    )
