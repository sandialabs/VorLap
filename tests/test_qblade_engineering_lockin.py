import json
import math
from pathlib import Path

import numpy as np
import pytest

from examples.ehrmann2014_naca0018.calibrate_engineering_lockin import (
    build_calibration,
)
from vorlap.qblade_engineering_lockin import (
    EngineeringLockInCalibration,
    capture_weight,
    captured_spectral_coefficient,
    critical_mass_damping,
    feedback_gain_from_threshold,
    fit_endpoint_saturation,
)


def test_compact_capture_uses_every_harmonic_and_excludes_dc():
    ratio = np.asarray([0.0, 0.85, 0.925, 1.0, 1.05, 1.1, 1.2])
    weight = capture_weight(ratio, low=0.85, peak=1.0, high=1.1)
    np.testing.assert_allclose(weight[[0, 1, 3, 5, 6]], [0, 0, 1, 0, 0])
    assert 0.0 < weight[2] < 1.0
    assert 0.0 < weight[4] < 1.0

    result = captured_spectral_coefficient(
        [0.0, 0.09, 0.10, 0.105, 0.14],
        [-0.4, 0.2, 0.3, 0.4, 9.0],
        effective_speed_mps=10.0,
        strouhal_reference_length_m=1.0,
        natural_frequency_hz=1.0,
        low=0.85,
        peak=1.0,
        high=1.1,
    )
    expected = capture_weight(
        np.asarray([0.9, 1.0, 1.05]), low=0.85, peak=1.0, high=1.1
    ) * np.asarray([0.2, 0.3, 0.4])
    assert result.coefficient_rss == pytest.approx(float(np.linalg.norm(expected)))
    assert result.active_harmonics == 3
    assert result.total_harmonic_energy == pytest.approx(
        0.2**2 + 0.3**2 + 0.4**2 + 9.0**2
    )


def test_endpoint_fit_predicts_three_unseen_ehrmann_markers_within_30_percent():
    fit = fit_endpoint_saturation(1.007724, 0.01236635, 1.478621, 0.00556788)
    assert fit.limiting_amplitude_ratio == pytest.approx(0.020343253217205814)
    assert fit.critical_mass_damping == pytest.approx(1.5983534133928563)
    heldout = (
        (1.138759, 0.01195601),
        (1.264000, 0.01036376),
        (1.375724, 0.00889484),
    )
    errors = [
        abs(fit.predict_amplitude_ratio(md) - value) / value for md, value in heldout
    ]
    np.testing.assert_allclose(
        errors,
        [0.08760037239637868, 0.10222081537558944, 0.14643395776926113],
    )
    assert max(errors) < 0.30


def test_feedback_threshold_round_trip_and_structural_equilibrium():
    gain = feedback_gain_from_threshold(
        critical_mass_damping=1.5983534133928563,
        captured_lift_coefficient=0.03925587880977077,
        effective_reduced_velocity=6.284,
    )
    assert gain == pytest.approx(127.897387460612)
    assert critical_mass_damping(
        feedback_gain=gain,
        captured_lift_coefficient=0.03925587880977077,
        effective_reduced_velocity=6.284,
    ) == pytest.approx(1.5983534133928563)

    mass = 2.2
    omega = 2.0 * math.pi * 6.0
    damping = 0.02
    structural = 2.0 * damping * mass * omega
    zero_feedback = 2.0 * structural
    limiting = 0.004
    equilibrium = limiting * math.sqrt(1.0 - structural / zero_feedback)
    saturation = 1.0 - (equilibrium / limiting) ** 2
    assert zero_feedback * saturation == pytest.approx(structural)


def test_tracked_calibration_recomputes_and_validates(tmp_path):
    payload = build_calibration()
    assert payload["spectral_capture"]["figure6_unused_check"]["passes_10_percent"]
    assert (
        payload["spectral_capture"]["figure6_unused_check"]["absolute_relative_error"]
        < 0.03
    )
    assert (
        payload["holdout_assessment"]["maximum_observed_absolute_relative_error"] < 0.15
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    calibration = EngineeringLockInCalibration.from_json(path)
    assert calibration.feedback_gain == pytest.approx(510.5219625924112)
    assert calibration.velocity_scale == pytest.approx(4.0 / 3.0)
    transfer = calibration.transferable_parameters
    assert transfer.feedback_gain == calibration.feedback_gain
    assert transfer.limiting_amplitude_ratio == calibration.limiting_amplitude_ratio
    assert transfer.capture_window == "compact_sine_squared"
    assert transfer.static_spectrum_sha256 == (
        payload["sources"]["stationary_spectrum"]["sha256"]
    )
    assert not hasattr(transfer, "velocity_scale")
    assert not hasattr(transfer, "reported_tunnel_blockage_fraction")

    tampered = json.loads(path.read_text())
    tampered["amplitude_calibration"]["critical_mass_damping"] += 0.01
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="critical_mass_damping"):
        EngineeringLockInCalibration.from_json(path)
