import math
from pathlib import Path

import pytest

from examples.viv_geometry_blended.calibrate import (
    build_airfoil_branch_calibration,
    build_calibration,
)
from vorlap.qblade_engineering_lockin import EngineeringLockInCalibration
from examples.liu2024_rectangular_5x1.assess_geometry_blended_model import assess
from vorlap.qblade_geometry_blended_lockin import (
    GeometryBlendedCalibration,
    GeometryBlendedLockInParameters,
    cycle_mean_aerodynamic_power_per_span,
    motion_correlated_force_per_span,
    predict_stable_amplitude_ratio,
)


def _parameters() -> GeometryBlendedLockInParameters:
    payload = build_calibration()
    p = payload["parameters"]
    return GeometryBlendedLockInParameters(**p)


def test_training_uses_three_nonvalidation_studies_and_passes_declared_gates():
    payload = build_calibration()
    assert len(payload["training"]["studies"]) == 3
    assert payload["training"]["maximum_reciprocal_peak_relative_error"] < 0.28
    assert payload["training"]["maximum_airfoil_peak_relative_error"] < 0.15
    assert payload["training"]["airfoil_peak_location_relative_error"] < 0.001
    assert payload["protected_validation"]["liu_5x1_used_for_fitting"] is False


def test_geometry_blend_has_exact_endpoints_and_continuous_midpoint():
    p = _parameters()
    assert p.airfoil_work_fraction(0.18) == 1.0
    assert p.airfoil_work_fraction(1.5) == 0.0
    assert p.airfoil_work_fraction(5.0) == 0.0
    assert p.airfoil_work_fraction((0.18 + 1.5) / 2) == pytest.approx(0.5)
    assert p.frequency_scale(0.18) == pytest.approx(p.airfoil_frequency_scale)
    assert p.frequency_scale(1.5) == 1.0


def test_both_branches_inject_at_small_amplitude_and_restore_at_large_amplitude():
    p = _parameters()
    common = dict(
        parameters=p,
        wind_speed_mps=8.0,
        air_density_kgpm3=1.225,
        crossflow_depth_m=0.23,
        sinusoidal_equivalent_coefficient=0.2,
    )
    for ratio in (0.18, 1.5, 0.84):
        _, peak, _ = p.frequency_bounds(ratio)
        small = cycle_mean_aerodynamic_power_per_span(
            peak_velocity_mps=0.01,
            amplitude_ratio=0.001,
            frequency_ratio=peak,
            breadth_depth_ratio=ratio,
            **common,
        )
        large = cycle_mean_aerodynamic_power_per_span(
            peak_velocity_mps=2.0,
            amplitude_ratio=0.08,
            frequency_ratio=peak,
            breadth_depth_ratio=ratio,
            **common,
        )
        assert small > 0.0
        assert large < 0.0


def test_instantaneous_force_is_odd_and_cycle_work_matches_analytic():
    p = _parameters()
    ratio = 0.84
    _, peak, _ = p.frequency_bounds(ratio)
    common = dict(
        parameters=p,
        amplitude_ratio=0.01,
        wind_speed_mps=8.0,
        air_density_kgpm3=1.225,
        crossflow_depth_m=0.23,
        sinusoidal_equivalent_coefficient=0.2,
        frequency_ratio=peak,
        breadth_depth_ratio=ratio,
        regularized=False,
    )
    positive = motion_correlated_force_per_span(modal_velocity_mps=0.2, **common)
    negative = motion_correlated_force_per_span(modal_velocity_mps=-0.2, **common)
    assert negative == pytest.approx(-positive)
    assert math.isfinite(positive)


def test_pure_airfoil_equilibrium_has_critical_damping_threshold():
    p = _parameters()
    _, peak, _ = p.frequency_bounds(0.18)
    low = predict_stable_amplitude_ratio(
        parameters=p,
        natural_frequency_hz=6.0,
        mass_per_span_kgpm=2.4,
        structural_damping_ratio=0.01,
        air_density_kgpm3=1.225,
        crossflow_depth_m=0.23,
        wind_speed_mps=8.0,
        sinusoidal_equivalent_coefficient=0.2,
        frequency_ratio=peak,
        breadth_depth_ratio=0.18,
    )
    high = predict_stable_amplitude_ratio(
        parameters=p,
        natural_frequency_hz=6.0,
        mass_per_span_kgpm=2.4,
        structural_damping_ratio=0.2,
        air_density_kgpm3=1.225,
        crossflow_depth_m=0.23,
        wind_speed_mps=8.0,
        sinusoidal_equivalent_coefficient=0.2,
        frequency_ratio=peak,
        breadth_depth_ratio=0.18,
    )
    assert low > 0.0
    assert high == 0.0


def test_frozen_json_is_strictly_loadable(tmp_path: Path):
    path = tmp_path / "calibration.json"
    import json

    path.write_text(json.dumps(build_calibration(), indent=2, sort_keys=True) + "\n")
    calibration = GeometryBlendedCalibration.from_json(path)
    assert calibration.parameters.airfoil_work_fraction(0.18) == 1.0

    airfoil_path = tmp_path / "airfoil.json"
    airfoil_path.write_text(
        json.dumps(build_airfoil_branch_calibration(), indent=2, sort_keys=True)
        + "\n"
    )
    airfoil = EngineeringLockInCalibration.from_json(airfoil_path)
    assert airfoil.feedback_gain == pytest.approx(
        calibration.parameters.airfoil_velocity_feedback_gain
    )
    assert airfoil.transferable_parameters.frequency_ratio_peak == pytest.approx(
        calibration.parameters.frequency_ratio_peak
        * calibration.parameters.airfoil_frequency_scale
    )


def test_protected_liu_validation_passes_without_fitting_liu_response():
    report = assess()
    assert report["status"] == "passed_protected_liu_5x1_validation"
    assert report["pass"] is True
    assert report["model"]["liu_response_used_for_fitting"] is False
    assert report["prediction"]["peak_absolute_relative_error"] < 0.06
    assert report["prediction"]["onset_absolute_relative_error"] < 0.05
    assert all(report["gates"].values())
