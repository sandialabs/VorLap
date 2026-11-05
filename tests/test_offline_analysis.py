from __future__ import annotations

import warnings

import numpy as np
import pytest

from vorlap.offline_analysis import VorLapOfflineAnalysisConfig, run_offline_analysis
from vorlap.structs import Component, InflowTimeSeries, VIV_Params


@pytest.mark.parametrize("frequencies", [[0.0], [-1.0], [float("nan")], [float("inf")], []])
def test_invalid_modal_frequencies_fail_before_loading_inputs(frequencies):
    with pytest.raises(ValueError, match="frequencies"):
        run_offline_analysis(
            VorLapOfflineAnalysisConfig(
                viv_params=VIV_Params(),
                natural_frequencies=np.asarray(frequencies),
                inflow_profile_path="not-loaded.csv",
            )
        )


def _component(component_id: str = "blade") -> Component:
    shape_xyz = np.array([[0.0, 0.0, 0.0]], dtype=float)
    return Component(
        id=component_id,
        translation=np.zeros(3),
        rotation=np.zeros(3),
        pitch=np.array([0.0]),
        shape_xyz=shape_xyz,
        shape_xyz_global=shape_xyz.copy(),
        chord=np.ones(1),
        twist=np.zeros(1),
        thickness=np.full(1, 0.12),
        offset=np.zeros(1),
        airfoil_ids=["default"],
        chord_vector=np.zeros((1, 3)),
        normal_vector=np.zeros((1, 3)),
    )


def _patch_fast_analysis(monkeypatch, module, tmp_path):
    calls: list[tuple[str, object]] = []

    def fake_load_inflow(path):
        calls.append(("load_inflow", path))
        return InflowTimeSeries(
            time=np.array([0.0, 1.0]),
            inflow_speeds=np.array([2.0, 2.0]),
            inflow_directions=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )

    def fake_stationary(*args, **kwargs):
        warnings.warn("source clipped", RuntimeWarning)
        return (
            np.array([[1.0]]),
            {"case": "stationary"},
            np.zeros((1, 1, 3)),
            np.zeros((1, 1, 3)),
            np.zeros((2, 3, 1)),
        )

    def fake_time_varying(*args, **kwargs):
        return (
            np.array([0.0, 1.0]),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.ones((2, 3, 1)),
        )

    def fake_write_force(path, time, forces):
        calls.append(("write_force", path))
        assert time.shape == (2,)
        assert forces.shape == (2, 3, 1)

    def fake_write_loading(path, time, forces, node_ids, local):
        calls.append(("write_loading", (path, tuple(node_ids), local)))

    monkeypatch.setattr(module, "load_inflow_time_series", fake_load_inflow)
    monkeypatch.setattr(module, "load_airfoils_from_folder", lambda _folder: {"default": object()})
    monkeypatch.setattr(module.graphics, "calc_structure_vectors_andplot", lambda *args, **kwargs: calls.append(("frames", None)))
    monkeypatch.setattr(module, "compute_thrust_torque_spectrum_optimized", fake_stationary)
    monkeypatch.setattr(module, "compute_time_varying_force_history_optimized", fake_time_varying)
    monkeypatch.setattr(module, "write_force_time_series", fake_write_force)
    monkeypatch.setattr(module, "write_qblade_loading_file", fake_write_loading)
    return calls


def test_run_offline_analysis_returns_typed_result_and_artifacts(monkeypatch, tmp_path):
    import vorlap.offline_analysis as offline

    calls = _patch_fast_analysis(monkeypatch, offline, tmp_path)
    messages: list[str] = []
    component = _component()
    viv = VIV_Params(azimuths=np.array([0.0]), inflow_speeds=np.array([2.0]), output_time=np.array([0.0, 1.0]))

    result = run_offline_analysis(
        VorLapOfflineAnalysisConfig(
            viv_params=viv,
            natural_frequencies=np.array([1.0]),
            inflow_profile_path=tmp_path / "inflow.csv",
            components=[component],
            save_dir=tmp_path / "out",
        ),
        progress_callback=messages.append,
    )

    assert result["viv_params"] is viv
    assert result.components == (component,)
    assert result.warnings == ("source clipped",)
    assert result.artifacts["force_time_series"].endswith("forces_output.csv")
    assert result.metadata["stationary_include_mean_load"] is True
    assert result.metadata["time_varying_include_mean_load"] is True
    assert ("frames", None) in calls
    assert any(message.startswith("Running thrust/torque") for message in messages)
    assert not any(call[0] == "write_loading" for call in calls)


def test_run_offline_analysis_qblade_geometry_writes_loading_file(monkeypatch, tmp_path):
    import vorlap.offline_analysis as offline

    calls = _patch_fast_analysis(monkeypatch, offline, tmp_path)
    component = _component("BLD_1")
    qblade_viv = VIV_Params(
        fluid_density=1.3,
        fluid_dynamicviscosity=1.7e-5,
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        inflow_vec=np.array([0.0, 1.0, 0.0]),
    )
    monkeypatch.setattr(
        offline,
        "convert_qblade_to_vorlap_inputs",
        lambda *args, **kwargs: ([component], qblade_viv, ["BLD_1_0.000000"]),
    )
    viv = VIV_Params(azimuths=np.array([0.0]), inflow_speeds=np.array([2.0]), output_time=np.array([0.0, 1.0]))

    result = run_offline_analysis(
        VorLapOfflineAnalysisConfig(
            viv_params=viv,
            natural_frequencies=np.array([1.0]),
            inflow_profile_path=tmp_path / "inflow.csv",
            use_qblade_geometry=True,
            qblade_sim_path=tmp_path / "case.sim",
            save_dir=tmp_path / "out",
            qblade_loading_path="loads.txt",
        )
    )

    assert result.components == (component,)
    assert result.qblade_node_ids == ("BLD_1_0.000000",)
    assert result.qblade_sim_path.endswith("case.sim")
    assert viv.fluid_density == 1.3
    assert result.artifacts["qblade_loading_file"].endswith("loads.txt")
    assert any(
        call == ("write_loading", (str(tmp_path / "out" / "loads.txt"), ("BLD_1_0.000000",), False))
        for call in calls
    )
