from pathlib import Path

import numpy as np
import pytest

from vorlap.graphics import calc_structure_vectors_andplot
from vorlap.structs import VIV_Params

from conftest import make_component


def test_calc_structure_vectors_andplot_handles_single_node(tmp_path: Path):
    airfoil_dir = tmp_path / "airfoils"
    airfoil_dir.mkdir()
    # Minimal closed profile with two columns.
    np.savetxt(
        airfoil_dir / "default.csv",
        np.array([[1.0, 0.0], [0.5, 0.1], [0.0, 0.0], [0.5, -0.1], [1.0, 0.0]]),
        delimiter=",",
    )

    comp = make_component(n_nodes=1, span=0.0, airfoil_id="default")
    viv = VIV_Params(airfoil_folder=str(airfoil_dir) + "/", output_time=np.array([0.0, 1.0]))

    fig = calc_structure_vectors_andplot([comp], viv, show_plot=False, return_fig=True, save_path=None)
    assert fig is not None
    assert np.linalg.norm(comp.chord_vector[0]) > 0.0
    assert np.linalg.norm(comp.normal_vector[0]) > 0.0


def test_calc_structure_vectors_andplot_requires_components():
    viv = VIV_Params()
    with pytest.raises(ValueError, match="At least one component"):
        calc_structure_vectors_andplot([], viv, show_plot=False, return_fig=False)
