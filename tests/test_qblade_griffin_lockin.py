from __future__ import annotations

import json
import math

import numpy as np
import pytest

from examples.marra2015_rectangular_4x1.calibrate_griffin_lockin import (
    build_calibration,
)
from examples.liu2024_rectangular_5x1.assess_griffin_model import assess
from vorlap.qblade_griffin_lockin import (
    GriffinCalibration,
    GriffinLockInParameters,
    GriffinTrainingPoint,
    fit_griffin_gains,
    ideal_cycle_mean_aerodynamic_power_per_span,
    motion_correlated_force_per_span,
    predict_stable_amplitude_ratio,
)

PARAMETERS = GriffinLockInParameters(
    coulomb_work_gain=0.6977211891570517,
    velocity_damping_gain=18.883554312927426,
    frequency_ratio_low=1.0,
    frequency_ratio_peak=1.166666667,
    frequency_ratio_high=1.21875,
    regularization_velocity_over_wind=1.0e-5,
)


def _training_points() -> list[GriffinTrainingPoint]:
    return [
        GriffinTrainingPoint(7.97, 7.089, 0.00058, 1.19, 0.075, 0.6549, 0.06551),
        GriffinTrainingPoint(7.87, 7.292, 0.00650, 1.22, 0.075, 0.7364, 0.02920),
        GriffinTrainingPoint(7.88, 7.292, 0.02340, 1.22, 0.075, 0.7733, 0.01106),
    ]


def test_fit_recovers_reciprocal_griffin_gains_and_training_rows() -> None:
    fit = fit_griffin_gains(
        _training_points(), sinusoidal_equivalent_coefficient=math.sqrt(2.0) * 0.64
    )
    assert fit.rank == 2
    assert fit.coulomb_work_gain == pytest.approx(0.6977211891570517)
    assert fit.velocity_damping_gain == pytest.approx(18.883554312927426)
    assert fit.maximum_training_relative_error < 0.003


def test_cycle_work_has_injection_and_restoring_branches() -> None:
    common = dict(
        parameters=PARAMETERS,
        wind_speed_mps=5.0,
        air_density_kgpm3=1.2,
        crossflow_depth_m=0.075,
        sinusoidal_equivalent_coefficient=0.9,
        frequency_ratio=PARAMETERS.frequency_ratio_peak,
    )
    assert (
        ideal_cycle_mean_aerodynamic_power_per_span(peak_velocity_mps=0.02, **common)
        > 0.0
    )
    assert (
        ideal_cycle_mean_aerodynamic_power_per_span(peak_velocity_mps=1.0, **common)
        < 0.0
    )
    assert (
        ideal_cycle_mean_aerodynamic_power_per_span(peak_velocity_mps=0.0, **common)
        == 0.0
    )


def test_force_is_odd_zero_outside_capture_and_dimensionally_finite() -> None:
    common = dict(
        parameters=PARAMETERS,
        wind_speed_mps=5.0,
        air_density_kgpm3=1.2,
        crossflow_depth_m=0.075,
        sinusoidal_equivalent_coefficient=0.9,
        frequency_ratio=PARAMETERS.frequency_ratio_peak,
    )
    positive = motion_correlated_force_per_span(modal_velocity_mps=0.1, **common)
    negative = motion_correlated_force_per_span(modal_velocity_mps=-0.1, **common)
    assert math.isfinite(positive)
    assert negative == pytest.approx(-positive)
    assert (
        motion_correlated_force_per_span(
            modal_velocity_mps=0.1,
            **{**common, "frequency_ratio": PARAMETERS.frequency_ratio_low},
        )
        == 0.0
    )
    assert motion_correlated_force_per_span(modal_velocity_mps=0.0, **common) == 0.0


def test_analytic_stable_amplitude_balances_cycle_work() -> None:
    frequency = 7.87
    mass = 7.292
    damping = 0.0065
    depth = 0.075
    density = 1.22
    speed = 2.0 * math.pi * frequency * depth / 0.7364
    coefficient = math.sqrt(2.0) * 0.64
    amplitude_ratio = predict_stable_amplitude_ratio(
        parameters=PARAMETERS,
        natural_frequency_hz=frequency,
        mass_per_span_kgpm=mass,
        structural_damping_ratio=damping,
        air_density_kgpm3=density,
        crossflow_depth_m=depth,
        wind_speed_mps=speed,
        sinusoidal_equivalent_coefficient=coefficient,
        frequency_ratio=PARAMETERS.frequency_ratio_peak,
    )
    peak_velocity = amplitude_ratio * depth * 2.0 * math.pi * frequency
    aerodynamic_power = ideal_cycle_mean_aerodynamic_power_per_span(
        parameters=PARAMETERS,
        peak_velocity_mps=peak_velocity,
        wind_speed_mps=speed,
        air_density_kgpm3=density,
        crossflow_depth_m=depth,
        sinusoidal_equivalent_coefficient=coefficient,
        frequency_ratio=PARAMETERS.frequency_ratio_peak,
    )
    structural_coefficient = 2.0 * damping * mass * 2.0 * math.pi * frequency
    assert aerodynamic_power == pytest.approx(
        0.5 * structural_coefficient * peak_velocity**2, rel=1e-13
    )
    assert amplitude_ratio == pytest.approx(0.029166348261197697)


def test_smoothed_force_converges_to_analytic_cycle_work() -> None:
    speed = 5.0
    peak_velocity = 0.1
    time = np.linspace(0.0, 100.0, 400_001)
    velocity = peak_velocity * np.cos(2.0 * math.pi * time)
    force = np.asarray(
        [
            motion_correlated_force_per_span(
                parameters=PARAMETERS,
                modal_velocity_mps=value,
                wind_speed_mps=speed,
                air_density_kgpm3=1.2,
                crossflow_depth_m=0.075,
                sinusoidal_equivalent_coefficient=0.9,
                frequency_ratio=PARAMETERS.frequency_ratio_peak,
            )
            for value in velocity
        ]
    )
    numerical = float(np.trapezoid(force * velocity, time) / (time[-1] - time[0]))
    analytic = ideal_cycle_mean_aerodynamic_power_per_span(
        parameters=PARAMETERS,
        peak_velocity_mps=peak_velocity,
        wind_speed_mps=speed,
        air_density_kgpm3=1.2,
        crossflow_depth_m=0.075,
        sinusoidal_equivalent_coefficient=0.9,
        frequency_ratio=PARAMETERS.frequency_ratio_peak,
    )
    assert numerical == pytest.approx(analytic, rel=2e-5)


def test_calibration_is_4x1_only_and_strictly_loadable(tmp_path) -> None:
    payload = build_calibration()
    assert payload["training"]["maximum_relative_error"] < 0.003
    assert payload["claim_boundary"]["liu_5x1_used_for_parameter_fitting"] is False
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    loaded = GriffinCalibration.from_json(path)
    assert loaded.parameters.coulomb_work_gain == pytest.approx(0.6977211891570517)

    tampered = json.loads(path.read_text())
    tampered["claim_boundary"]["liu_5x1_used_for_parameter_fitting"] = True
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="used for fitting"):
        GriffinCalibration.from_json(path)


def test_untouched_liu_5x1_validation_passes_without_retuning() -> None:
    report = assess()
    assert report["status"] == "passed_untouched_independent_5x1_validation"
    assert report["pass"] is True
    assert report["model"]["retuned_for_liu_5x1"] is False
    assert report["prediction"]["peak_absolute_relative_error"] < 0.20
    assert report["prediction"]["onset_absolute_relative_error"] < 0.05
    assert all(report["gates"].values())
    assert report["claim_boundary"]["qualifies_model_for_hvawt_pilot_preparation"]
    assert report["claim_boundary"]["validates_fatigue_loads"] is False
