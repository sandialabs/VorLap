from pathlib import Path

import numpy as np
import pytest

from vorlap.graphics import calc_structure_vectors_andplot
from vorlap.structs import VIV_Params

from conftest import make_component


def test_calc_structure_vectors_andplot_handles_single_node_without_mutating_frames(tmp_path: Path):
    airfoil_dir = tmp_path / "airfoils"
    airfoil_dir.mkdir()
    # Minimal closed profile with two columns.
    np.savetxt(
        airfoil_dir / "default.csv",
        np.array([[1.0, 0.0], [0.5, 0.1], [0.0, 0.0], [0.5, -0.1], [1.0, 0.0]]),
        delimiter=",",
    )

    comp = make_component(n_nodes=1, span=0.0, airfoil_id="default")
    comp.shape_xyz_global = np.zeros_like(comp.shape_xyz_global)
    comp.chord_vector = np.zeros_like(comp.chord_vector)
    comp.normal_vector = np.zeros_like(comp.normal_vector)
    original_global = comp.shape_xyz_global.copy()
    original_chord = comp.chord_vector.copy()
    original_normal = comp.normal_vector.copy()
    viv = VIV_Params(airfoil_folder=str(airfoil_dir) + "/", output_time=np.array([0.0, 1.0]))

    fig = calc_structure_vectors_andplot([comp], viv, show_plot=False, return_fig=True, save_path=None)
    assert fig is not None
    np.testing.assert_allclose(comp.shape_xyz_global, original_global)
    np.testing.assert_allclose(comp.chord_vector, original_chord)
    np.testing.assert_allclose(comp.normal_vector, original_normal)
    assert fig.layout.meta["airfoil_coordinate_warnings"] == []


def test_calc_structure_vectors_andplot_exposes_airfoil_fallback(tmp_path: Path):
    comp = make_component(n_nodes=1, span=0.0, airfoil_id="missing")
    viv = VIV_Params(airfoil_folder=str(tmp_path), output_time=np.array([0.0, 1.0]))

    with pytest.warns(UserWarning, match="Falling back"):
        fig = calc_structure_vectors_andplot([comp], viv, show_plot=False, return_fig=True, save_path=None)

    assert fig is not None
    warnings_meta = fig.layout.meta["airfoil_coordinate_warnings"]
    assert len(warnings_meta) == 1
    assert "missing.csv" in warnings_meta[0]


def test_calc_structure_vectors_andplot_requires_components():
    viv = VIV_Params()
    with pytest.raises(ValueError, match="At least one component"):
        calc_structure_vectors_andplot([], viv, show_plot=False, return_fig=False)
