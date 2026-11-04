from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_headless.postprocess_parked_demo import (
    equivalent_load_with_cycles,
    turning_points,
)
from vorlap.qblade_full_structure_output import MonitorPoint, build_monitor_plan
from vorlap.qblade_surface_strain_del import (
    MonitorBinding,
    OwensElement,
    axial_surface_normal_strain_history,
    bind_monitors_to_owens,
    equivalent_load_with_cycles_fast,
    read_airfoil_coordinates,
    read_owens_model,
    read_reference_pitch_axes,
    recover_generalized_strain,
    surface_points_for_binding,
    turning_points_vectorized,
    validate_dense_result_history,
    validate_pitch_axis,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SIM = (
    REPO_ROOT
    / "examples"
    / "ReferenceHVAWTQBladeModel"
    / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
)
REFERENCE_AIRFOIL = (
    REPO_ROOT
    / "examples"
    / "ReferenceHVAWTQBladeModel"
    / "10m_HVAWT_reference_turbine"
    / "Aero"
    / "Airfoils"
    / "NACA_0018_RFoil.afl"
)
REFERENCE_BLADE = (
    REPO_ROOT
    / "examples"
    / "ReferenceHVAWTQBladeModel"
    / "10m_HVAWT_reference_turbine"
    / "Aero"
    / "10m_HVAWT_reference_rotor.bld"
)


@pytest.fixture
def archived_owens_model() -> Path:
    """Historical geometry integration is optional; no private model is vendored."""
    configured = os.environ.get("VORLAP_ARCHIVE_RUNS")
    root = Path(configured).expanduser() if configured else REPO_ROOT / "run_headless/runs"
    model = root / "vorlap_changed_geometry_trial_hardened_v2_20260731/owens_modal"
    missing = [name for name in ("nodes.csv", "elements.csv", "visual_sections.csv")
               if not (model / name).is_file()]
    if missing:
        message = (f"Historical OWENS geometry requires {model}: missing {', '.join(missing)}. "
                   "Set VORLAP_ARCHIVE_RUNS to the retained run archive to enable this integration test.")
        if configured:
            pytest.fail(message)
        pytest.skip(message)
    return model


def _synthetic_binding() -> MonitorBinding:
    """A unit beam with coupled positive-definite stiffness, independent of designs."""
    start = np.zeros(3)
    stop = np.array([1., 0., 0.])
    matrix = np.diag([2., 3., 4., 5., 6., 7.])
    matrix[0, 4] = matrix[4, 0] = .2
    element = OwensElement(1, 4, 1, 2, 1., start, stop, np.eye(3), matrix,
                           .5, .25, .5, .25)
    monitor = MonitorPoint("BLD_1", "blade", 1, .5, "synthetic-section")
    return MonitorBinding(0, monitor, element, 0., 1., start, stop, .5, .25, .5, .25)


def test_fast_rainflow_matches_campaign_implementation() -> None:
    rng = np.random.default_rng(321)
    cases = [
        np.asarray([0.0, 2.0, 0.0]),
        np.asarray([0.0, 0.0, 2.0, 2.0, 0.0]),
        np.cumsum(rng.normal(size=2000)),
        np.sin(np.linspace(0.0, 30.0 * math.pi, 3001))
        + 0.2 * np.sin(np.linspace(0.0, 117.0 * math.pi, 3001)),
    ]
    for values in cases:
        np.testing.assert_array_equal(
            turning_points_vectorized(values),
            turning_points(values),
        )
        expected = equivalent_load_with_cycles(values, 10.0, 600.0)
        actual = equivalent_load_with_cycles_fast(values, 10.0, 600.0)
        assert actual[1] == expected[1]
        assert math.isclose(actual[0], expected[0], rel_tol=2.0e-14, abs_tol=0.0)


def test_fast_rainflow_rejects_nonfinite_history() -> None:
    values = np.asarray([0.0, 2.0, math.nan, 0.0])
    with pytest.raises(ValueError, match="non-finite"):
        turning_points_vectorized(values)
    with pytest.raises(ValueError, match="non-finite"):
        equivalent_load_with_cycles_fast(values, 10.0, 600.0)


@pytest.mark.parametrize(
    ("values", "wohler_exponent"),
    [
        (np.asarray([-1.0e6, 1.0e6, -1.0e6]), 100.0),
        (np.asarray([-1.0e308, 1.0e308, -1.0e308]), 2.0),
    ],
)
def test_fast_rainflow_rejects_finite_overflow(
    values: np.ndarray,
    wohler_exponent: float,
) -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        equivalent_load_with_cycles_fast(values, wohler_exponent, 600.0)


def test_monitor_binding_covers_full_visible_structure(archived_owens_model) -> None:
    monitors, _ = build_monitor_plan(REFERENCE_SIM)
    elements = read_owens_model(archived_owens_model)
    bindings = bind_monitors_to_owens(monitors, elements)
    assert len(bindings) == 289
    assert {binding.monitor.family for binding in bindings} == {
        "blade",
        "strut",
        "tower",
        "torque_tube",
        "substructure",
    }
    assert [
        binding.element.element_id
        for binding in bindings
        if binding.monitor.component == "BLD_1"
    ] == list(range(35, 77))
    sub_1 = [binding for binding in bindings if binding.monitor.component == "SUB_1"]
    sub_2 = [binding for binding in bindings if binding.monitor.component == "SUB_2"]
    assert len(sub_1) == 10
    assert len(sub_2) == 7
    assert sub_1[0].segment_start_position == 0.0
    assert sub_1[-1].segment_stop_position == 1.0
    assert sub_2[0].segment_start_position == 0.0
    assert sub_2[-1].segment_stop_position == 1.0


def test_surface_coordinates_use_actual_airfoil_and_circular_tower(archived_owens_model) -> None:
    monitors, _ = build_monitor_plan(REFERENCE_SIM)
    bindings = bind_monitors_to_owens(monitors, read_owens_model(archived_owens_model))
    airfoil = read_airfoil_coordinates(REFERENCE_AIRFOIL)

    blade = next(binding for binding in bindings if binding.monitor.component == "BLD_1")
    blade_points = surface_points_for_binding(
        blade,
        airfoil,
        pitch_axis_fraction=0.25,
        circle_points=48,
    )
    assert len(blade_points) == 35
    assert min(point.local_y_m for point in blade_points) == -0.375
    assert max(point.local_y_m for point in blade_points) == 0.125
    assert math.isclose(
        max(abs(point.local_z_m) for point in blade_points),
        0.045015,
        rel_tol=1.0e-12,
    )

    tower = next(binding for binding in bindings if binding.monitor.component == "TWR")
    tower_points = surface_points_for_binding(
        tower,
        airfoil,
        pitch_axis_fraction=0.25,
        circle_points=48,
    )
    assert len(tower_points) == 48
    assert {point.perimeter_kind for point in tower_points} == {"circle"}
    radii = [math.hypot(point.local_y_m, point.local_z_m) for point in tower_points]
    assert np.ptp(radii) < 1.0e-12
    assert math.isclose(radii[0], 0.5, rel_tol=1.0e-12)

    noncircular = replace(tower, start_half_z_m=0.4)
    with pytest.raises(ValueError, match="not circular"):
        surface_points_for_binding(
            noncircular,
            airfoil,
            pitch_axis_fraction=0.25,
            circle_points=48,
        )


def test_reference_pitch_axis_is_read_and_enforced() -> None:
    axes = read_reference_pitch_axes(REFERENCE_BLADE)
    assert axes.size == 25
    np.testing.assert_array_equal(axes, np.full(25, 0.25))
    validate_pitch_axis(REFERENCE_BLADE, 0.25)
    with pytest.raises(ValueError, match="does not match"):
        validate_pitch_axis(REFERENCE_BLADE, 0.30)


def test_generalized_strain_uses_qblade_to_gxbeam_axis_mapping() -> None:
    binding = _synthetic_binding()
    channels = [channel.replace(" ", "~") for channel in binding.monitor.result_channels()]
    force = np.asarray([10.0, 20.0, -5.0])
    moment_x = np.asarray([3.0, -7.0, 2.0])
    moment_y = np.asarray([-4.0, 8.0, 1.0])
    data = pd.DataFrame(
        {
            channels[0]: force,
            channels[1]: moment_x,
            channels[2]: moment_y,
        }
    )
    generalized = recover_generalized_strain(data, binding)
    loads = np.zeros((6, force.size))
    loads[0] = force
    loads[4] = moment_x
    loads[5] = moment_y
    np.testing.assert_allclose(
        binding.element.stiffness @ generalized,
        loads,
        rtol=2.0e-13,
        atol=2.0e-12,
    )


def test_surface_strain_matches_hand_calculation_for_diagonal_stiffness() -> None:
    binding = _synthetic_binding()
    stiffness = np.diag([10.0, 1.0, 1.0, 1.0, 20.0, 40.0])
    binding = replace(binding, element=replace(binding.element, stiffness=stiffness))
    channels = [channel.replace(" ", "~") for channel in binding.monitor.result_channels()]
    data = pd.DataFrame(
        {
            channels[0]: [10.0, 20.0],
            channels[1]: [4.0, 8.0],
            channels[2]: [8.0, 16.0],
        }
    )
    generalized = recover_generalized_strain(data, binding)
    actual = axial_surface_normal_strain_history(
        generalized,
        local_y_m=0.5,
        local_z_m=0.25,
    )
    # epsilon = F/EA - (Mx/EIyy) z + (My/EIzz) y.
    np.testing.assert_allclose(actual, [1.05, 2.10], rtol=0.0, atol=1.0e-14)


def _one_monitor_history(*, sample_count: int = 80_000) -> tuple[pd.DataFrame, list]:
    monitor = build_monitor_plan(REFERENCE_SIM)[0][0]
    time_s = 60.0 + 0.0075 * np.arange(sample_count)
    data = pd.DataFrame(
        {
            "Time~[s]": time_s,
            "LSS~Rpm~[rpm]": np.ones(sample_count),
            **{
                channel.replace(" ", "~"): np.zeros(sample_count)
                for channel in monitor.result_channels()
            },
        }
    )
    return data, [monitor]


def test_dense_history_gate_accepts_exact_production_shape() -> None:
    data, monitors = _one_monitor_history()
    report = validate_dense_result_history(
        data,
        monitors,
        expected_sample_count=80_000,
        expected_start_s=60.0,
        expected_timestep_s=0.0075,
        expected_rpm=1.0,
        rpm_tolerance=0.01,
        minimum_duration_s=600.0,
        allow_short=False,
    )
    assert report["observed_result_column_count"] == 5
    assert report["sample_count"] == 80_000
    assert report["sampled_duration_s"] == pytest.approx(600.0)
    assert report["timestep_max_error_s"] < 1.0e-12
    assert report["rpm_max_error"] == 0.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.iloc[:-1].copy(), "samples; expected"),
        (
            lambda data: data.assign(**{"Time~[s]": data["Time~[s]"] + 1.0}),
            "starts at",
        ),
        (
            lambda data: data.assign(
                **{
                    "Time~[s]": data["Time~[s]"]
                    + np.r_[np.zeros(len(data) - 1), 0.001]
                }
            ),
            "timestep differs",
        ),
        (
            lambda data: data.assign(
                **{"LSS~Rpm~[rpm]": np.full(len(data), 1.02)}
            ),
            "RPM differs",
        ),
        (lambda data: data.drop(columns=data.columns[-1]), "columns do not match"),
    ],
)
def test_dense_history_gate_rejects_production_mismatch(mutation, message: str) -> None:
    data, monitors = _one_monitor_history()
    with pytest.raises(ValueError, match=message):
        validate_dense_result_history(
            mutation(data),
            monitors,
            expected_sample_count=80_000,
            expected_start_s=60.0,
            expected_timestep_s=0.0075,
            expected_rpm=1.0,
            rpm_tolerance=0.01,
            minimum_duration_s=600.0,
            allow_short=False,
        )
