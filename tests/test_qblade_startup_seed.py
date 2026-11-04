from dataclasses import replace
import json
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from conftest import make_component, make_constant_airfoil_fft, make_viv_params
from vorlap.qblade_controller import VorLapQBladeController
from vorlap.qblade_response_model import (
    QBladeResponseDiagnostics,
    QBladeResponseModel,
    QBladeResponseModelConfig,
    QBladeSeededResponseDiagnostics,
    QBladeStartupSeedConfig,
)
from vorlap.qblade_runtime import _response_model_from_config


def _seed_config(**overrides):
    config = QBladeResponseModelConfig(
        node_ids=("root", "tip"),
        mode_shape=(0.5, 1.0),
        natural_frequency_hz=2.0,
        reference_length_m=0.05,
        reduced_velocity_low=4.0,
        reduced_velocity_peak=5.0,
        reduced_velocity_high=8.0,
        aerodynamic_damping_gain_h1=0.0,
        saturation_amplitude_ratio=0.25,
        rms_time_constant_s=0.2,
        max_abs_gain=100.0,
        max_abs_nodal_force_n=100.0,
        force_direction="normal",
        projection_weights=(2.0, 3.0),
        bandpass_damping_ratio=0.2,
        enabled=True,
        startup_seed=QBladeStartupSeedConfig(
            enabled=True,
            duration_cycles=5.0,
            target_amplitude_ratio=0.005,
            structural_damping_ratio=0.0028,
        ),
    )
    return replace(config, **overrides)


def _inputs(structural_velocity=None):
    if structural_velocity is None:
        structural_velocity = np.zeros((2, 3), dtype=float)
    return {
        "node_ids": ("root", "tip"),
        "direction_basis": np.tile([1.0, 0.0, 0.0], (2, 1)),
        "structural_velocity_mps": np.asarray(structural_velocity, dtype=float),
        "effective_velocity_mps": np.full(2, 5.0),
        "section_width_m": np.full(2, 0.05),
        "local_length_m": np.full(2, 0.5),
        "fluid_density_kgpm3": 1.225,
    }


def _portable_response_config():
    return {
        "type": "modal_negative_damping",
        "schema_version": 1,
        "enabled": True,
        "node_ids": ["root", "tip"],
        "mode_shape": [0.5, 1.0],
        "projection_weights": [2.0, 3.0],
        "natural_frequency_hz": 2.0,
        "reference_length_m": 0.05,
        "reduced_velocity_activation": [4.0, 5.0, 8.0],
        "aerodynamic_damping_gain_h1": 0.0,
        "saturation_amplitude_ratio": 0.25,
        "rms_time_constant_s": 0.2,
        "max_abs_gain": 100.0,
        "max_abs_nodal_force_n": 100.0,
        "force_direction": "normal",
        "bandpass_damping_ratio": 0.2,
        "startup_seed": {
            "schema_version": 1,
            "enabled": True,
            "duration_cycles": 5.0,
            "target_amplitude_ratio": 0.005,
            "structural_damping_ratio": 0.0028,
            "taper": "hann_sine_squared",
            "carrier": "natural_frequency",
            "nodal_weighting": "projection_mass_times_mode",
            "nodal_force_limit_source": (
                "response_model_max_abs_nodal_force_n"
            ),
        },
    }


def test_five_cycle_hann_seed_waveform_and_mass_modal_distribution():
    model = QBladeResponseModel(_seed_config())
    frequency_hz = model.config.natural_frequency_hz
    duration_s = 5.0 / frequency_hz
    time_s = 2.25 / frequency_hz

    result = model.step(time_s, **_inputs())
    diagnostics = result.diagnostics

    assert isinstance(diagnostics, QBladeSeededResponseDiagnostics)
    expected_taper = math.sin(math.pi * time_s / duration_s) ** 2
    expected_acceleration = (
        diagnostics.startup_seed_modal_acceleration_scale_mps2
        * expected_taper
        * math.sin(2.0 * math.pi * frequency_hz * time_s)
    )
    assert diagnostics.startup_seed_enabled is True
    assert diagnostics.startup_seed_active is True
    assert diagnostics.startup_seed_elapsed_cycles == pytest.approx(2.25)
    assert diagnostics.startup_seed_duration_cycles == 5.0
    assert diagnostics.startup_seed_duration_s == pytest.approx(duration_s)
    assert diagnostics.startup_seed_taper == pytest.approx(expected_taper)
    assert diagnostics.startup_seed_modal_acceleration_mps2 == pytest.approx(
        expected_acceleration,
        rel=2.0e-15,
        abs=1.0e-15,
    )

    mode = np.array([0.5, 1.0])
    nodal_mass = np.array([2.0, 3.0])
    expected_force = nodal_mass * mode * expected_acceleration
    np.testing.assert_allclose(
        result.additive_force_n[:, 0],
        expected_force,
        rtol=2.0e-15,
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(
        result.additive_force_n[:, 1:],
        np.zeros((2, 2)),
    )
    assert diagnostics.startup_seed_raw_generalized_force_n == pytest.approx(
        np.dot(mode, expected_force)
    )
    assert diagnostics.startup_seed_force_limited_nodes == 0
    assert diagnostics.force_limited_nodes == 0
    assert diagnostics.startup_seed_raw_max_abs_nodal_force_n < (
        model.config.max_abs_nodal_force_n
    )


def test_seed_scale_realizes_requested_ideal_terminal_free_response():
    model = QBladeResponseModel(_seed_config())
    probe = model.step(0.0, **_inputs()).diagnostics
    assert isinstance(probe, QBladeSeededResponseDiagnostics)
    omega = 2.0 * math.pi * model.config.natural_frequency_hz
    damping = model.config.startup_seed.structural_damping_ratio
    duration_s = probe.startup_seed_duration_s
    acceleration_scale = probe.startup_seed_modal_acceleration_scale_mps2

    def rhs(time_s, state):
        displacement_m, velocity_mps = state
        if 0.0 <= time_s < duration_s:
            acceleration_mps2 = (
                acceleration_scale
                * math.sin(math.pi * time_s / duration_s) ** 2
                * math.sin(omega * time_s)
            )
        else:
            acceleration_mps2 = 0.0
        return (
            velocity_mps,
            acceleration_mps2
            - 2.0 * damping * omega * velocity_mps
            - omega**2 * displacement_m,
        )

    integration = solve_ivp(
        rhs,
        (0.0, duration_s),
        (0.0, 0.0),
        rtol=2.0e-12,
        atol=1.0e-14,
        max_step=duration_s / 2000.0,
    )
    assert integration.success
    displacement_m, velocity_mps = integration.y[:, -1]
    damped_omega = omega * math.sqrt(1.0 - damping**2)
    terminal_envelope_m = math.hypot(
        displacement_m,
        (velocity_mps + damping * omega * displacement_m) / damped_omega,
    )
    target_m = 0.005 * model.config.reference_length_m

    assert probe.startup_seed_target_amplitude_ratio == pytest.approx(0.005)
    assert probe.startup_seed_target_amplitude_m == pytest.approx(target_m)
    assert probe.startup_seed_ideal_terminal_amplitude_m == pytest.approx(
        target_m,
        rel=2.0e-15,
    )
    assert terminal_envelope_m == pytest.approx(target_m, rel=2.0e-9)


def test_seed_is_exactly_zero_at_and_after_five_cycles():
    model = QBladeResponseModel(_seed_config())
    duration_s = 5.0 / model.config.natural_frequency_hz

    active = model.step(duration_s - 0.01, **_inputs())
    at_cutoff = model.step(duration_s, **_inputs())
    after_cutoff = model.step(duration_s + 0.123, **_inputs())

    assert active.diagnostics.startup_seed_active is True
    assert active.diagnostics.startup_seed_modal_acceleration_mps2 != 0.0
    for result in (at_cutoff, after_cutoff):
        diagnostics = result.diagnostics
        assert diagnostics.startup_seed_active is False
        assert diagnostics.startup_seed_taper == 0.0
        assert diagnostics.startup_seed_modal_acceleration_mps2 == 0.0
        assert diagnostics.startup_seed_raw_generalized_force_n == 0.0
        assert diagnostics.startup_seed_raw_max_abs_nodal_force_n == 0.0
        np.testing.assert_array_equal(
            result.additive_force_n,
            np.zeros((2, 3)),
        )


def test_seed_checkpoint_restart_and_duplicate_are_exactly_idempotent():
    config = _seed_config()
    original = QBladeResponseModel(config)
    inputs = _inputs(
        structural_velocity=np.array(
            [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
            dtype=float,
        )
    )
    accepted = None
    for time_s in (0.0, 0.05, 0.11, 0.29, 0.47):
        accepted = original.step(time_s, **inputs)
    assert accepted is not None
    checkpoint = json.loads(json.dumps(original.checkpoint_state()))
    assert checkpoint["startup_seed"]["schema_version"] == 1
    assert checkpoint["startup_seed"]["last_active"] is True
    assert checkpoint["startup_seed"]["last_modal_acceleration_mps2"] == (
        accepted.diagnostics.startup_seed_modal_acceleration_mps2
    )

    restored = QBladeResponseModel(config)
    restored.restore_checkpoint_state(checkpoint)
    before_duplicate = restored.checkpoint_state()
    duplicate = restored.step(0.47, **inputs)
    assert restored.checkpoint_state() == before_duplicate
    np.testing.assert_array_equal(
        duplicate.additive_force_n,
        accepted.additive_force_n,
    )
    assert duplicate.diagnostics == accepted.diagnostics

    for time_s in (0.61, 1.2, 2.5, 2.7):
        expected = original.step(time_s, **inputs)
        actual = restored.step(time_s, **inputs)
        np.testing.assert_array_equal(actual.additive_force_n, expected.additive_force_n)
        assert actual.diagnostics == expected.diagnostics
        assert restored.checkpoint_state() == original.checkpoint_state()


def _seeded_controller():
    component = make_component(n_nodes=2, span=1.0, airfoil_id="default")
    component.id = "BLD_1"
    return VorLapQBladeController(
        components=[component],
        airfoils={"default": make_constant_airfoil_fft(n_freq=2)},
        viv_params=make_viv_params(),
        node_ids=("root", "tip"),
        n_freq_depth=2,
        force_scale=0.0,
        include_mean_load=False,
        response_model=QBladeResponseModel(_seed_config()),
    )


def test_controller_checkpoints_seed_limits_and_summary_exactly():
    original = _seeded_controller()
    node_data = {
        "velocity": np.tile([0.0, 5.0, 0.0], (2, 1)),
        "structural_velocity": np.zeros((2, 3), dtype=float),
        "azimuth_deg": 0.0,
    }
    accepted = None
    for time_s in (0.0, 0.1, 0.3, 0.7):
        accepted = original.step({**node_data, "time": time_s})
    assert accepted is not None
    summary = original.response_diagnostics_summary
    assert summary["startup_seed_samples"] == 4
    assert summary["startup_seed_active_steps"] == 4
    assert summary["startup_seed_force_limited_steps"] == 0
    assert summary["startup_seed_force_limited_node_samples"] == 0
    assert summary["max_startup_seed_raw_nodal_force_n"] > 0.0
    assert summary["startup_seed_target_amplitude_ratio"] == pytest.approx(0.005)

    checkpoint = json.loads(json.dumps(original.checkpoint_state()))
    restored = _seeded_controller()
    restored.restore_checkpoint_state(checkpoint)
    assert restored.response_diagnostics_summary == summary
    before_duplicate = restored.checkpoint_state()
    duplicate = restored.step({**node_data, "time": 0.7})
    np.testing.assert_array_equal(duplicate, accepted)
    after_duplicate = restored.checkpoint_state()
    assert after_duplicate["response_model_state"] == (
        before_duplicate["response_model_state"]
    )
    assert after_duplicate["response_diagnostics_summary"] == (
        before_duplicate["response_diagnostics_summary"]
    )
    assert after_duplicate["cache_hits"] == before_duplicate["cache_hits"] + 1

    expected = original.step({**node_data, "time": 0.9})
    actual = restored.step({**node_data, "time": 0.9})
    np.testing.assert_array_equal(actual, expected)
    assert restored.response_diagnostics_summary == (
        original.response_diagnostics_summary
    )


def test_default_and_explicitly_disabled_seed_preserve_legacy_state_exactly():
    default_config = replace(_seed_config(), startup_seed=None)
    disabled_config = replace(
        default_config,
        startup_seed=QBladeStartupSeedConfig(enabled=False),
    )
    default_model = QBladeResponseModel(default_config)
    disabled_model = QBladeResponseModel(disabled_config)
    assert default_model.startup_seed_enabled is False
    assert disabled_model.startup_seed_enabled is False

    for time_s in (0.0, 0.1, 0.25, 0.7):
        structural_velocity = np.array(
            [
                [0.02 * math.sin(4.0 * math.pi * time_s), 0.0, 0.0],
                [0.04 * math.sin(4.0 * math.pi * time_s), 0.0, 0.0],
            ]
        )
        inputs = _inputs(structural_velocity)
        default_result = default_model.step(time_s, **inputs)
        disabled_result = disabled_model.step(time_s, **inputs)
        assert type(default_result.diagnostics) is QBladeResponseDiagnostics
        assert type(disabled_result.diagnostics) is QBladeResponseDiagnostics
        np.testing.assert_array_equal(
            disabled_result.additive_force_n,
            default_result.additive_force_n,
        )
        assert disabled_result.diagnostics == default_result.diagnostics
    assert disabled_model.checkpoint_state() == default_model.checkpoint_state()


def test_seed_uses_response_force_limit_and_exposes_clipping():
    model = QBladeResponseModel(
        _seed_config(max_abs_nodal_force_n=1.0e-8)
    )
    result = model.step(2.25 / model.config.natural_frequency_hz, **_inputs())
    diagnostics = result.diagnostics

    assert diagnostics.startup_seed_raw_max_abs_nodal_force_n > 1.0e-8
    assert diagnostics.startup_seed_force_limited_nodes > 0
    assert diagnostics.force_limited_nodes > 0
    assert diagnostics.max_abs_nodal_force_n == pytest.approx(1.0e-8)
    assert np.max(np.abs(result.additive_force_n)) == pytest.approx(1.0e-8)


def test_portable_runtime_seed_config_is_explicit_and_validated():
    raw = _portable_response_config()
    model = _response_model_from_config(raw, ("root", "tip"))

    assert isinstance(model, QBladeResponseModel)
    assert model.startup_seed_enabled is True
    assert model.config.startup_seed == QBladeStartupSeedConfig(
        enabled=True,
        duration_cycles=5.0,
        target_amplitude_ratio=0.005,
        structural_damping_ratio=0.0028,
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("enabled", 1, TypeError),
        ("duration_cycles", True, TypeError),
        ("duration_cycles", "5.0", TypeError),
        ("duration_cycles", 0.0, ValueError),
        ("duration_cycles", math.nan, ValueError),
        ("target_amplitude_ratio", 0.0, ValueError),
        ("target_amplitude_ratio", 1.01, ValueError),
        ("structural_damping_ratio", -0.1, ValueError),
        ("structural_damping_ratio", 1.0, ValueError),
        ("taper", "boxcar", ValueError),
        ("carrier", "vortex_shedding", ValueError),
        ("nodal_weighting", "uniform", ValueError),
        ("nodal_force_limit_source", "none", ValueError),
        ("schema_version", 1.0, ValueError),
    ],
)
def test_portable_runtime_seed_rejects_invalid_values(field, value, error):
    raw = _portable_response_config()
    raw["startup_seed"][field] = value

    with pytest.raises(error, match="startup_seed"):
        _response_model_from_config(raw, ("root", "tip"))


def test_enabled_seed_requires_explicit_nodal_mass_quadrature():
    raw = _portable_response_config()
    raw.pop("projection_weights")

    with pytest.raises(ValueError, match="projection_weights.*mass quadrature"):
        _response_model_from_config(raw, ("root", "tip"))


def test_portable_runtime_seed_rejects_unknown_provenance_fields():
    raw = _portable_response_config()
    raw["startup_seed"]["target_amplitdue_ratio"] = 0.005

    with pytest.raises(ValueError, match="unsupported field.*target_amplitdue"):
        _response_model_from_config(raw, ("root", "tip"))
