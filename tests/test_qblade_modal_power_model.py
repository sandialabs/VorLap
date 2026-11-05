from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from vorlap.qblade_modal_power_model import (
    ExternalModalPowerTarget,
    QBladeModalPowerDiagnostics,
    QBladeModalPowerLimitError,
    QBladeModalPowerModel,
    QBladeModalPowerModelConfig,
    TabulatedAerodynamicPowerCurve,
)

REFERENCE_CURVE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "grinderslev2022_iea10"
    / "reference_data"
    / "p100_i50_figure6_power_digitized.csv"
)
SERIES = "p100_i50_forced_motion"
NODE_IDS = ("root", "mid", "tip")
MODE_BY_NODE = {"root": 0.2, "mid": 0.6, "tip": 1.0}


def _curve():
    return TabulatedAerodynamicPowerCurve.from_csv(
        REFERENCE_CURVE,
        expected_series=SERIES,
    )


def _config(**overrides):
    config = QBladeModalPowerModelConfig(
        node_ids=NODE_IDS,
        mode_shape=(0.2, 0.6, 1.0),
        modal_mass_weights_kg=(800.0, 400.0, 100.0),
        aerodynamic_tributary_weights=(0.5, 1.0, 1.5),
        natural_frequency_hz=0.67,
        power_curve=_curve(),
        rms_time_constant_s=4.0,
        max_abs_target_power_w=5.0e3,
        max_abs_generalized_force_n=1.0e6,
        max_abs_nodal_force_n=1.0e6,
        bandpass_damping_ratio=0.2,
    )
    return replace(config, **overrides)


def _external_config(**overrides):
    values = {
        "power_curve": None,
        "external_target_source_fingerprint": "a" * 64,
        "enabled": True,
    }
    values.update(overrides)
    return _config(**values)


def _modal_step_inputs(
    time_s,
    *,
    amplitude_m=1.0,
    frequency_ratio=1.0,
    node_ids=NODE_IDS,
    dc_velocity_mps=0.0,
    orthogonal_velocity_mps=0.0,
):
    omega = 2.0 * math.pi * 0.67 * frequency_ratio
    qdot = omega * amplitude_m * math.cos(omega * time_s) + dc_velocity_mps
    directions = []
    velocities = []
    for node_id in node_ids:
        phi = MODE_BY_NODE[node_id]
        directions.append((2.0, 0.0, 0.0))  # normalization is intentional
        velocities.append((phi * qdot, orthogonal_velocity_mps, 0.0))
    return {
        "node_ids": node_ids,
        "direction_basis": np.asarray(directions, dtype=float),
        "structural_velocity_mps": np.asarray(velocities, dtype=float),
    }


def _run_harmonic(
    model,
    amplitude_m,
    *,
    dt_s=0.02,
    duration_s=80.0,
    frequency_ratio=1.0,
):
    results = []
    for index in range(int(round(duration_s / dt_s)) + 1):
        time_s = index * dt_s
        results.append(
            model.step(
                time_s,
                **_modal_step_inputs(
                    time_s,
                    amplitude_m=amplitude_m,
                    frequency_ratio=frequency_ratio,
                ),
            )
        )
    return results


def test_digitized_p100_i50_curve_retains_power_and_equilibrium_targets():
    curve = _curve()

    assert curve.source_label == SERIES
    assert (
        curve.source_sha256
        == "016ce5265b77dc0c9e577d7548e453d09f67ef732dda817618f79909546f323d"
    )
    assert curve.evaluate(0.0) == 0.0
    # The quadratic origin continuation is explicit and finite, while every
    # digitized knot is retained exactly.
    assert curve.evaluate(0.05) == pytest.approx(11.25)
    assert curve.evaluate(1.0) == pytest.approx(1390.0)
    assert curve.evaluate(1.585) == pytest.approx(-22.5)
    assert curve.positive_zero_crossings_m() == pytest.approx((1.57,))

    equilibrium_540 = curve.structural_equilibria_m(540.0)
    equilibrium_495p8 = curve.structural_equilibria_m(495.8)
    assert equilibrium_540 == pytest.approx((1.2455587568,), abs=1.0e-9)
    assert equilibrium_495p8 == pytest.approx((1.2586411831,), abs=1.0e-9)
    # The figure-derived curve supports an approximately 1.25--1.26 m
    # intersection for the two published damping-power coefficients.  This is
    # close to the reported 1.24 m forced-motion point, but it is deliberately
    # not relabeled as the separate 1.18 m FSI result.
    assert 1.24 <= equilibrium_540[0] <= 1.26
    assert 1.24 <= equilibrium_495p8[0] <= 1.27
    # The forced-motion zero crossing (1.57 m) is within 0.06 m of the
    # separately reported zero-damping FSI amplitude (about 1.63 m).
    assert abs(curve.positive_zero_crossings_m()[0] - 1.63) <= 0.06 + 1.0e-12


@pytest.mark.parametrize(
    ("amplitude", "power", "match"),
    [
        ((0.0, 0.2, 0.1), (0.0, 20.0, -1.0), "strictly increasing"),
        ((0.0, 0.1, 0.2), (1.0, 20.0, -1.0), "start"),
        ((0.0, 0.1, 0.2), (0.0, -1.0, -2.0), "initially inject"),
        ((0.0, 0.1, 0.2), (0.0, 1.0, 0.0), "negative-power"),
        ((0.0, 0.1, 0.2), (0.0, math.nan, -1.0), "finite"),
    ],
)
def test_bad_power_tables_fail_visibly(amplitude, power, match):
    with pytest.raises(ValueError, match=match):
        TabulatedAerodynamicPowerCurve(amplitude, power)


def test_configuration_requires_boolean_enablement_and_fingerprints_curve_provenance():
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        QBladeModalPowerModel(_config(enabled="false"))

    first = QBladeModalPowerModel(_config())
    relabeled_curve = replace(_curve(), source_label="independent relabel")
    second = QBladeModalPowerModel(_config(power_curve=relabeled_curve))
    assert (
        first.checkpoint_state()["config_fingerprint"]
        != second.checkpoint_state()["config_fingerprint"]
    )

    external = QBladeModalPowerModel(_external_config())
    assert (
        external.checkpoint_state()["config_fingerprint"]
        != first.checkpoint_state()["config_fingerprint"]
    )
    with pytest.raises(ValueError, match="requires a SHA-256"):
        QBladeModalPowerModel(_external_config(external_target_source_fingerprint=""))
    with pytest.raises(ValueError, match="mutually exclusive"):
        QBladeModalPowerModel(_config(external_target_source_fingerprint="b" * 64))


def test_external_target_path_is_energy_conjugate_and_restart_exact():
    model = QBladeModalPowerModel(_external_config())

    def evaluator(amplitude_m):
        target = 100.0 * amplitude_m**2
        return ExternalModalPowerTarget(
            target_cycle_mean_power_w=target,
            origin_equivalent_velocity_gain_kgps=(200.0 / (2.0 * math.pi * 0.67) ** 2),
            context_fingerprint="c" * 64,
        )

    results = []
    dt_s = 0.02
    for index in range(3001):
        time_s = index * dt_s
        results.append(
            model.step(
                time_s,
                **_modal_step_inputs(time_s, amplitude_m=0.5),
                target_power_evaluator=evaluator,
            )
        )
    final = results[-1].diagnostics
    assert final.amplitude_estimate_m == pytest.approx(0.5, rel=0.03)
    assert final.target_cycle_mean_power_w == pytest.approx(
        100.0 * final.amplitude_estimate_m**2
    )
    settled = [item.diagnostics.instantaneous_modal_power_w for item in results[-500:]]
    assert np.mean(settled) == pytest.approx(
        final.target_cycle_mean_power_w,
        rel=0.04,
    )

    checkpoint = model.checkpoint_state()
    restarted = QBladeModalPowerModel(_external_config())
    restarted.restore_checkpoint_state(checkpoint)
    next_time = 3001 * dt_s
    expected = model.step(
        next_time,
        **_modal_step_inputs(next_time, amplitude_m=0.5),
        target_power_evaluator=evaluator,
    )
    actual = restarted.step(
        next_time,
        **_modal_step_inputs(next_time, amplitude_m=0.5),
        target_power_evaluator=evaluator,
    )
    np.testing.assert_array_equal(expected.additive_force_n, actual.additive_force_n)
    assert expected.diagnostics == actual.diagnostics


def test_dynamic_spectral_dual_is_energy_conjugate_and_signature_bound():
    model = QBladeModalPowerModel(_external_config())

    def evaluator(amplitude_m):
        return ExternalModalPowerTarget(
            target_cycle_mean_power_w=50.0 * amplitude_m**2,
            origin_equivalent_velocity_gain_kgps=2.0,
            context_fingerprint="9" * 64,
        )

    weights = np.asarray((0.25, 2.0, 4.0), dtype=float)
    directions = np.tile(np.asarray((1.0, 0.0, 0.0)), (3, 1))
    model.step(
        0.0,
        node_ids=NODE_IDS,
        direction_basis=directions,
        structural_velocity_mps=np.zeros((3, 3)),
        energy_conjugate_weights=weights,
        target_power_evaluator=evaluator,
    )
    # This deliberately is not a pure modal velocity field.  Using the same
    # spectrum-derived dual for observation and force lift-back must still
    # preserve exact instantaneous virtual work.
    directional_velocity = np.asarray((1.0, -0.35, 0.2), dtype=float)
    velocities = directions * directional_velocity[:, None]
    result = model.step(
        0.05,
        node_ids=NODE_IDS,
        direction_basis=directions,
        structural_velocity_mps=velocities,
        energy_conjugate_weights=weights,
        target_power_evaluator=evaluator,
    )
    mode = np.asarray(tuple(MODE_BY_NODE[node_id] for node_id in NODE_IDS))
    denominator = float(np.dot(weights * mode, mode))
    expected_qdot = float(np.dot(weights * mode, directional_velocity) / denominator)
    generalized_force = result.diagnostics.applied_generalized_force_n
    expected_scalar_force = generalized_force * weights * mode / denominator
    np.testing.assert_allclose(result.additive_force_n[:, 0], expected_scalar_force)
    assert result.diagnostics.raw_qdot_mps == pytest.approx(expected_qdot)
    assert result.diagnostics.instantaneous_structural_power_w == pytest.approx(
        generalized_force * expected_qdot, rel=2.0e-14, abs=1.0e-14
    )
    assert result.diagnostics.instantaneous_controller_state_power_w == pytest.approx(
        result.diagnostics.instantaneous_modal_power_w
    )
    assert result.diagnostics.instantaneous_realized_work_w == pytest.approx(
        result.diagnostics.instantaneous_structural_power_w
    )
    assert result.diagnostics.dynamic_conjugate_weight_override is True
    assert result.diagnostics.dynamic_conjugate_weight_min == pytest.approx(0.25)
    assert result.diagnostics.dynamic_conjugate_weight_max == pytest.approx(4.0)

    duplicate = model.step(
        0.05,
        node_ids=NODE_IDS,
        direction_basis=directions,
        structural_velocity_mps=velocities,
        energy_conjugate_weights=weights,
        target_power_evaluator=evaluator,
    )
    np.testing.assert_array_equal(duplicate.additive_force_n, result.additive_force_n)
    with pytest.raises(ValueError, match="changed at the same time"):
        model.step(
            0.05,
            node_ids=NODE_IDS,
            direction_basis=directions,
            structural_velocity_mps=velocities,
            energy_conjugate_weights=np.asarray((0.5, 2.0, 4.0)),
            target_power_evaluator=evaluator,
        )


def test_external_target_same_time_context_change_is_rejected_atomically():
    model = QBladeModalPowerModel(_external_config())

    def target(context):
        return lambda amplitude: ExternalModalPowerTarget(
            target_cycle_mean_power_w=amplitude**2,
            origin_equivalent_velocity_gain_kgps=1.0,
            context_fingerprint=context * 64,
        )

    inputs = _modal_step_inputs(0.0, amplitude_m=0.1)
    first = model.step(0.0, **inputs, target_power_evaluator=target("d"))
    duplicate = model.step(0.0, **inputs, target_power_evaluator=target("d"))
    np.testing.assert_array_equal(first.additive_force_n, duplicate.additive_force_n)
    state = model.checkpoint_state()
    with pytest.raises(ValueError, match="changed at the same time"):
        model.step(0.0, **inputs, target_power_evaluator=target("e"))
    assert model.checkpoint_state() == state


def test_external_target_is_required_exactly_for_external_configuration():
    inputs = _modal_step_inputs(0.0)
    with pytest.raises(ValueError, match="required exactly"):
        QBladeModalPowerModel(_external_config()).step(0.0, **inputs)
    with pytest.raises(ValueError, match="required exactly"):
        QBladeModalPowerModel(_config()).step(
            0.0,
            **inputs,
            target_power_evaluator=lambda _amplitude: ExternalModalPowerTarget(
                0.0, 0.0, "f" * 64
            ),
        )


def test_power_curve_loader_rejects_malformed_rows_and_out_of_domain(tmp_path):
    malformed = tmp_path / "bad.csv"
    malformed.write_text(
        "tip_amplitude_m,aerodynamic_power_w,series\n"
        "0,0,case\n"
        "0.1,not-a-number,case\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed"):
        TabulatedAerodynamicPowerCurve.from_csv(malformed, expected_series="case")

    curve = _curve()
    with pytest.raises(ValueError, match="exceeds the calibrated"):
        curve.evaluate(1.61)
    with pytest.raises(ValueError, match="non-negative"):
        curve.evaluate(-0.01)
    with pytest.raises(ValueError, match="source_sha256"):
        TabulatedAerodynamicPowerCurve(
            (0.0, 0.1, 0.2),
            (0.0, 1.0, -1.0),
            source_sha256="not-a-digest",
        )


def test_model_is_opt_in_and_mass_projection_is_energy_conjugate():
    model = QBladeModalPowerModel(_config())
    velocities = np.asarray(
        [
            (1.0, 3.0, 0.0),
            (0.5, 3.0, 0.0),
            (-0.25, 3.0, 0.0),
        ]
    )
    result = model.step(
        0.0,
        node_ids=NODE_IDS,
        direction_basis=np.tile((1.0, 0.0, 0.0), (3, 1)),
        structural_velocity_mps=velocities,
    )
    phi = np.asarray((0.2, 0.6, 1.0))
    mass = np.asarray((800.0, 400.0, 100.0))
    expected = float(np.dot(mass * phi, velocities[:, 0]) / np.dot(mass * phi, phi))

    assert model.config.enabled is False
    assert result.diagnostics.enabled is False
    assert result.diagnostics.raw_qdot_mps == pytest.approx(expected)
    assert result.diagnostics.filtered_qdot_mps == 0.0
    assert result.diagnostics.applied_target_cycle_mean_power_w == 0.0
    np.testing.assert_array_equal(result.additive_force_n, np.zeros((3, 3)))


def test_cycle_mean_modal_and_structural_work_follow_calibration_table():
    model = QBladeModalPowerModel(_config(enabled=True))
    results = _run_harmonic(model, 1.0)
    final = results[-1].diagnostics

    assert final.amplitude_estimate_m == pytest.approx(1.0, rel=0.02)
    expected_power = model.config.power_curve.evaluate(final.amplitude_estimate_m)
    assert final.target_cycle_mean_power_w == pytest.approx(expected_power)
    assert final.target_power_limited is False
    assert final.generalized_force_limited is False
    assert final.nodal_force_limited_nodes == 0
    settled = [result.diagnostics for result in results[-1000:]]
    mean_target = np.mean(
        [diagnostic.target_cycle_mean_power_w for diagnostic in settled]
    )
    mean_modal_work = np.mean(
        [diagnostic.instantaneous_modal_power_w for diagnostic in settled]
    )
    mean_structural_work = np.mean(
        [diagnostic.instantaneous_structural_power_w for diagnostic in settled]
    )
    assert mean_modal_work == pytest.approx(mean_target, rel=0.01)
    assert mean_structural_work == pytest.approx(mean_target, rel=0.01)
    # Endpoint EMAs retain a small twice-frequency ripple, but still make any
    # material target/realized mismatch directly visible to a live caller.
    assert final.realized_modal_power_ema_w == pytest.approx(
        final.target_cycle_mean_power_w,
        rel=0.05,
    )
    assert final.realized_structural_power_ema_w == pytest.approx(
        final.target_cycle_mean_power_w,
        rel=0.05,
    )

    # The normalized aerodynamic distribution is exactly generalized-force
    # conjugate before a limit, independent of the structural mass weights.
    assert final.applied_generalized_force_n == pytest.approx(
        final.raw_generalized_force_n,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_negative_power_branch_removes_modal_energy():
    # Use a long RMS window to keep the estimated 1.585-m harmonic inside the
    # 1.60-m calibrated domain despite the unavoidable twice-frequency ripple.
    model = QBladeModalPowerModel(_config(enabled=True, rms_time_constant_s=20.0))
    results = _run_harmonic(
        model,
        1.585,
        dt_s=0.02,
        duration_s=180.0,
    )
    final = results[-1].diagnostics

    assert 1.57 < final.amplitude_estimate_m < 1.60
    assert final.target_cycle_mean_power_w < 0.0
    assert final.equivalent_velocity_gain_kgps < 0.0
    assert final.realized_modal_power_ema_w < 0.0
    assert final.realized_structural_power_ema_w < 0.0


def test_dc_and_off_band_motion_are_rejected():
    dc_model = QBladeModalPowerModel(_config(enabled=True, rms_time_constant_s=1.0))
    dc_results = []
    for index in range(401):
        time_s = index * 0.05
        dc_results.append(
            dc_model.step(
                time_s,
                **_modal_step_inputs(
                    time_s,
                    amplitude_m=0.0,
                    dc_velocity_mps=2.0,
                ),
            )
        )
    dc_final = dc_results[-1].diagnostics
    assert dc_final.raw_qdot_mps == pytest.approx(2.0)
    assert abs(dc_final.filtered_qdot_mps) < 1.0e-12
    assert dc_final.amplitude_estimate_m < 1.0e-12
    assert dc_final.max_abs_nodal_force_n < 1.0e-12

    off_band = QBladeModalPowerModel(_config(enabled=True, rms_time_constant_s=1.0))
    results = _run_harmonic(
        off_band,
        0.5,
        dt_s=0.02,
        duration_s=80.0,
        frequency_ratio=5.0,
    )
    settled = results[-200:]
    raw_rms = math.sqrt(
        np.mean([result.diagnostics.raw_qdot_mps**2 for result in settled])
    )
    filtered_rms = math.sqrt(
        np.mean([result.diagnostics.filtered_qdot_mps**2 for result in settled])
    )
    frequency_ratio = 5.0
    damping = 0.2
    expected_gain = (
        2.0
        * damping
        * frequency_ratio
        / math.sqrt(
            (1.0 - frequency_ratio**2) ** 2 + (2.0 * damping * frequency_ratio) ** 2
        )
    )
    assert filtered_rms / raw_rms == pytest.approx(expected_gain, rel=0.03)
    assert filtered_rms / raw_rms < 0.1


def test_power_and_filter_response_converge_with_timestep():
    coarse = QBladeModalPowerModel(_config(enabled=True))
    fine = QBladeModalPowerModel(_config(enabled=True))
    coarse_final = _run_harmonic(coarse, 0.75, dt_s=0.02)[-1].diagnostics
    fine_final = _run_harmonic(fine, 0.75, dt_s=0.01)[-1].diagnostics

    assert coarse_final.amplitude_estimate_m == pytest.approx(
        fine_final.amplitude_estimate_m,
        rel=0.005,
    )
    assert coarse_final.target_cycle_mean_power_w == pytest.approx(
        fine_final.target_cycle_mean_power_w,
        rel=0.005,
    )
    assert coarse_final.realized_modal_power_ema_w == pytest.approx(
        fine_final.realized_modal_power_ema_w,
        rel=0.01,
    )


def test_node_order_mode_scale_and_sign_do_not_change_result_or_checkpoint_identity():
    canonical = QBladeModalPowerModel(_config(enabled=True, rms_time_constant_s=1.0))
    reordered_config = _config(
        node_ids=("tip", "root", "mid"),
        mode_shape=(-10.0, -2.0, -6.0),
        modal_mass_weights_kg=(100.0, 800.0, 400.0),
        aerodynamic_tributary_weights=(1.5, 0.5, 1.0),
        enabled=True,
        rms_time_constant_s=1.0,
    )
    reordered = QBladeModalPowerModel(reordered_config)

    first = second = None
    for index in range(301):
        time_s = index * 0.02
        first = canonical.step(time_s, **_modal_step_inputs(time_s, amplitude_m=0.5))
        second = reordered.step(
            time_s,
            **_modal_step_inputs(
                time_s,
                amplitude_m=0.5,
                node_ids=("tip", "root", "mid"),
            ),
        )
    assert first is not None and second is not None
    np.testing.assert_array_equal(
        first.additive_force_n, second.additive_force_n[[1, 2, 0]]
    )
    assert first.diagnostics == second.diagnostics
    assert (
        canonical.checkpoint_state()["config_fingerprint"]
        == reordered.checkpoint_state()["config_fingerprint"]
    )

    # An exact same-time retry may use a different node order; it neither
    # advances state nor changes the physical force mapping.
    state_before = canonical.checkpoint_state()
    duplicate = canonical.step(
        6.0,
        **_modal_step_inputs(
            6.0,
            amplitude_m=0.5,
            node_ids=("tip", "root", "mid"),
        ),
    )
    assert canonical.checkpoint_state() == state_before
    np.testing.assert_array_equal(
        duplicate.additive_force_n, first.additive_force_n[[2, 0, 1]]
    )


def test_checkpoint_json_roundtrip_preserves_exact_duplicate_and_future_replay():
    config = _config(enabled=True, rms_time_constant_s=1.0)
    original = QBladeModalPowerModel(config)
    accepted = None
    for index in range(251):
        time_s = index * 0.02
        accepted = original.step(
            time_s,
            **_modal_step_inputs(time_s, amplitude_m=0.7),
        )
    assert accepted is not None
    serialized = json.loads(json.dumps(original.checkpoint_state()))

    restored = QBladeModalPowerModel(config)
    restored.restore_checkpoint_state(serialized)
    duplicate = restored.step(
        5.0,
        **_modal_step_inputs(5.0, amplitude_m=0.7),
    )
    np.testing.assert_array_equal(duplicate.additive_force_n, accepted.additive_force_n)
    assert duplicate.diagnostics == accepted.diagnostics

    future_time = 5.07
    future_inputs = _modal_step_inputs(future_time, amplitude_m=0.7)
    original_future = original.step(future_time, **future_inputs)
    restored_future = restored.step(future_time, **future_inputs)
    np.testing.assert_array_equal(
        restored_future.additive_force_n,
        original_future.additive_force_n,
    )
    assert restored_future.diagnostics == original_future.diagnostics
    assert restored.checkpoint_state() == original.checkpoint_state()


def test_limit_error_is_transactional_and_explicit_clipping_is_visible():
    fail_closed = QBladeModalPowerModel(
        _config(
            enabled=True,
            rms_time_constant_s=0.5,
            max_abs_target_power_w=100.0,
        )
    )
    fail_closed.step(0.0, **_modal_step_inputs(0.0, amplitude_m=1.0))
    rejected = False
    for index in range(1, 201):
        time_s = index * 0.02
        before_rejected_step = fail_closed.checkpoint_state()
        try:
            fail_closed.step(
                time_s,
                **_modal_step_inputs(time_s, amplitude_m=1.0),
            )
        except QBladeModalPowerLimitError as exc:
            assert "target power" in str(exc)
            assert fail_closed.checkpoint_state() == before_rejected_step
            rejected = True
            break
    assert rejected
    # The rejected step is transactional; it does not commit filter or energy
    # state beyond the last accepted below-limit sample.
    assert fail_closed.checkpoint_state()["last_time_s"] < 4.0

    clipped = QBladeModalPowerModel(
        _config(
            enabled=True,
            rms_time_constant_s=0.5,
            max_abs_target_power_w=500.0,
            max_abs_generalized_force_n=100.0,
            max_abs_nodal_force_n=20.0,
            limit_action="clip",
        )
    )
    results = _run_harmonic(
        clipped,
        1.0,
        dt_s=0.02,
        duration_s=20.0,
    )
    limited = [
        result.diagnostics
        for result in results
        if result.diagnostics.target_power_limited
        and result.diagnostics.generalized_force_limited
        and result.diagnostics.nodal_force_limited_nodes > 0
    ]
    assert limited
    diagnostic = limited[-1]
    assert diagnostic.target_cycle_mean_power_w > 500.0
    assert diagnostic.applied_target_cycle_mean_power_w == pytest.approx(500.0)
    assert diagnostic.force_scale < 1.0
    assert diagnostic.max_abs_nodal_force_n == pytest.approx(20.0)
    assert abs(diagnostic.applied_generalized_force_n) < 100.0
    assert diagnostic.realized_modal_power_ema_w < 500.0


def test_nonfinite_inputs_and_tampered_checkpoint_fail_visibly():
    model = QBladeModalPowerModel(_config(enabled=True))
    bad_velocity = _modal_step_inputs(0.0)
    bad_velocity["structural_velocity_mps"][1, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        model.step(0.0, **bad_velocity)

    clean = QBladeModalPowerModel(_config(enabled=True))
    clean.step(0.0, **_modal_step_inputs(0.0))
    checkpoint = json.loads(json.dumps(clean.checkpoint_state()))
    checkpoint["qdot_mean_square_m2ps2"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        QBladeModalPowerModel(_config(enabled=True)).restore_checkpoint_state(
            checkpoint
        )

    assert isinstance(clean.last_diagnostics, QBladeModalPowerDiagnostics)
