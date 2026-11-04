from dataclasses import is_dataclass

import numpy as np
import pytest

from vorlap.structs import AirfoilFFT, Component, InflowTimeSeries, VIV_Params, VorLapAnalysisResult


def _component_kwargs(n_nodes=2):
    shape_xyz = np.column_stack(
        [
            np.zeros(n_nodes, dtype=float),
            np.zeros(n_nodes, dtype=float),
            np.arange(n_nodes, dtype=float),
        ]
    )
    return {
        "id": "blade",
        "translation": np.zeros(3, dtype=float),
        "rotation": np.zeros(3, dtype=float),
        "pitch": np.array([0.0], dtype=float),
        "shape_xyz": shape_xyz,
        "shape_xyz_global": shape_xyz.copy(),
        "chord": np.ones(n_nodes, dtype=float),
        "twist": np.zeros(n_nodes, dtype=float),
        "thickness": np.full(n_nodes, 0.12, dtype=float),
        "offset": np.zeros(n_nodes, dtype=float),
        "airfoil_ids": np.array(["default"] * n_nodes, dtype=object),
        "chord_vector": np.tile([[1.0, 0.0, 0.0]], (n_nodes, 1)),
        "normal_vector": np.tile([[0.0, 1.0, 0.0]], (n_nodes, 1)),
    }


def test_core_structs_are_dataclasses():
    assert is_dataclass(AirfoilFFT)
    assert is_dataclass(Component)
    assert is_dataclass(VIV_Params)
    assert is_dataclass(InflowTimeSeries)
    assert is_dataclass(VorLapAnalysisResult)


def test_component_validates_node_aligned_arrays_and_exposes_node_count():
    component = Component(**_component_kwargs(n_nodes=3))

    assert component.node_count == 3
    assert component.airfoil_ids == ["default", "default", "default"]

    bad = _component_kwargs(n_nodes=2)
    bad["chord"] = np.ones(1)
    with pytest.raises(ValueError, match="chord length must match shape_xyz rows"):
        Component(**bad)


def test_component_validates_vector_shapes():
    bad = _component_kwargs()
    bad["translation"] = np.zeros(2)
    with pytest.raises(ValueError, match="translation must contain exactly three values"):
        Component(**bad)

    bad = _component_kwargs()
    bad["normal_vector"] = np.zeros((2, 2))
    with pytest.raises(ValueError, match="normal_vector must have shape"):
        Component(**bad)


def test_vorlap_analysis_result_preserves_dict_like_access_for_gui_plots():
    viv_params = VIV_Params(azimuths=np.array([0.0]), inflow_speeds=np.array([2.0]))
    result = VorLapAnalysisResult(
        percdiff_matrix=np.array([[1.0]]),
        percdiff_info={"case": "test"},
        total_global_force_vector=np.zeros((1, 1, 3)),
        total_global_moment_vector=np.zeros((1, 1, 3)),
        global_force_vector_nodes_selected_case=np.zeros((2, 3, 1)),
        global_force_vector_nodes=np.ones((2, 3, 1)),
        inflow_time=np.array([0.0, 1.0]),
        time_varying_total_global_force_vector=np.zeros((2, 3)),
        time_varying_total_global_moment_vector=np.zeros((2, 3)),
        qblade_node_ids=["BLD_1_0.000000"],
        qblade_sim_path="case.sim",
        viv_params=viv_params,
        warnings=["source clipped"],
        artifacts={"force_time_series": "forces.csv"},
        execution_time_s="1.25",
        metadata={"stationary_include_mean_load": True},
    )

    assert result["viv_params"] is viv_params
    assert result.get("missing", "fallback") == "fallback"
    assert result.qblade_node_ids == ("BLD_1_0.000000",)
    assert result.warnings == ("source clipped",)
    assert result.artifacts == {"force_time_series": "forces.csv"}
    assert result.execution_time_s == pytest.approx(1.25)
    assert result["metadata"] == {"stationary_include_mean_load": True}
    assert "percdiff_matrix" in result


def test_viv_params_default_mutables_are_not_shared():
    first = VIV_Params()
    second = VIV_Params()

    first.rotation_axis[0] = 99.0
    first.rotation_axis_offset[0] = 99.0
    first.inflow_vec[0] = 99.0
    first.azimuths[0] = 99.0
    first.inflow_speeds[0] = 99.0
    first.output_time[0] = 99.0
    first.plot_cycle[0] = "#000000"

    np.testing.assert_allclose(second.rotation_axis, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(second.rotation_axis_offset, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(second.inflow_vec, np.array([1.0, 0.0, 0.0]))
    assert second.azimuths[0] == pytest.approx(0.0)
    assert second.inflow_speeds[0] == pytest.approx(2.0)
    assert second.output_time[0] == pytest.approx(0.0)
    assert second.plot_cycle[0] == "#348ABD"


def test_viv_params_copies_caller_owned_mutables():
    rotation_axis = np.array([0.0, 0.0, 1.0])
    rotation_axis_offset = np.array([1.0, 2.0, 3.0])
    inflow_vec = np.array([1.0, 1.0, 0.0])
    azimuths = np.array([0.0, 30.0])
    inflow_speeds = np.array([5.0, 7.0])
    output_time = np.array([0.0, 0.25])
    plot_cycle = ["#111111", "#222222"]

    params = VIV_Params(
        rotation_axis=rotation_axis,
        rotation_axis_offset=rotation_axis_offset,
        inflow_vec=inflow_vec,
        plot_cycle=plot_cycle,
        azimuths=azimuths,
        inflow_speeds=inflow_speeds,
        output_time=output_time,
    )

    rotation_axis[:] = 99.0
    rotation_axis_offset[:] = 99.0
    inflow_vec[:] = 99.0
    azimuths[:] = 99.0
    inflow_speeds[:] = 99.0
    output_time[:] = 99.0
    plot_cycle[0] = "#999999"

    np.testing.assert_allclose(params.rotation_axis, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(params.rotation_axis_offset, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(params.inflow_vec, np.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(params.azimuths, np.array([0.0, 30.0]))
    np.testing.assert_allclose(params.inflow_speeds, np.array([5.0, 7.0]))
    np.testing.assert_allclose(params.output_time, np.array([0.0, 0.25]))
    assert params.plot_cycle == ["#111111", "#222222"]


def test_viv_params_validates_vector_and_output_pair_shapes():
    with pytest.raises(ValueError, match="exactly three"):
        VIV_Params(rotation_axis=np.array([1.0, 0.0]))

    with pytest.raises(ValueError, match="exactly two"):
        VIV_Params(output_azimuth_vinf=(0.0, 1.0, 2.0))
