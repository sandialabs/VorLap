import numpy as np
import pytest

import vorlap
from conftest import make_component, make_constant_airfoil_fft, make_viv_params
from vorlap import QBladeController, SwapArrayAdapter, SwapBlockSpec


def _build_controller(node_ids=None):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    viv_params = make_viv_params()
    viv_params.output_time = np.array([0.0, 0.25], dtype=float)
    if node_ids is None:
        node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    return QBladeController(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=node_ids,
        n_freq_depth=2,
    )


def test_qblade_controller_is_reexported_from_package_root():
    assert vorlap.QBladeController is QBladeController


def test_step_returns_per_node_forces():
    controller = _build_controller()
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    forces = controller.step(
        {
            "velocity": velocities,
            "time": 0.0,
            "azimuth_deg": 0.0,
        }
    )

    assert forces.shape == (2, 3)
    assert np.isfinite(forces).all()
    assert not forces.flags.writeable


def test_exact_input_cache_skips_recompute(monkeypatch):
    controller = _build_controller()
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    import vorlap.interpolation as interpolation

    calls = {"count": 0}
    original = interpolation.interpolate_fft_spectrum_batch

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(interpolation, "interpolate_fft_spectrum_batch", wrapped)

    first = controller.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})
    first_calls = calls["count"]
    assert first_calls > 0

    second = controller.step({"velocity": velocities.copy(), "time": 0.0, "azimuth_deg": 0.0})
    assert calls["count"] == first_calls
    assert second is first

    controller.step({"velocity": velocities, "time": 0.1, "azimuth_deg": 0.0})
    assert calls["count"] > first_calls


def test_constructor_validates_node_ids_length():
    with pytest.raises(ValueError, match="node_ids length must match"):
        _build_controller(node_ids=["BLD_1_0.000000"])


def test_step_validates_bad_velocity_input():
    controller = _build_controller()

    with pytest.raises(ValueError, match="Velocity vector must have length"):
        controller.step({"velocity": np.array([1.0, 2.0]), "time": 0.0, "azimuth_deg": 0.0})

    with pytest.raises(ValueError, match="contains non-finite"):
        controller.step(
            {
                "velocity": np.array([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
                "time": 0.0,
                "azimuth_deg": 0.0,
            }
        )


def test_swap_array_adapter_round_trip_with_controller():
    controller = _build_controller()
    swap = np.zeros(14, dtype=float)
    layout = {
        "velocity": SwapBlockSpec(offset=0, shape=(2, 3)),
        "time": SwapBlockSpec(offset=6, shape=()),
        "azimuth_deg": SwapBlockSpec(offset=7, shape=()),
        "force": SwapBlockSpec(offset=8, shape=(2, 3)),
    }
    adapter = SwapArrayAdapter.from_layout(swap, layout)

    velocity = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    adapter.write("velocity", velocity)
    adapter.write("time", 0.0)
    adapter.write("azimuth_deg", 0.0)

    direct = controller.step({"velocity": velocity, "time": 0.0, "azimuth_deg": 0.0})
    via_swap = controller.step_from_swap(adapter)

    np.testing.assert_allclose(via_swap, direct, atol=1e-12)
    np.testing.assert_allclose(adapter.read("force"), direct, atol=1e-12)


def test_swap_array_adapter_validates_overlap_and_bounds():
    with pytest.raises(ValueError, match="overlaps"):
        SwapArrayAdapter(
            np.zeros(10, dtype=float),
            {
                "a": SwapBlockSpec(offset=0, shape=(6,)),
                "b": SwapBlockSpec(offset=5, shape=(3,)),
            },
        )

    with pytest.raises(ValueError, match="exceeds avr_swap length"):
        SwapArrayAdapter(
            np.zeros(10, dtype=float),
            {
                "a": SwapBlockSpec(offset=9, shape=(2,)),
            },
        )
