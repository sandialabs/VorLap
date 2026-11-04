"""Physical checks on the independent public example, not paper gold updates."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'examples/ReferenceHVAWTQBladeModel/generate_public_sections.py'
spec = importlib.util.spec_from_file_location('public_sections', SOURCE)
sections = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sections)


def test_blade_matches_published_reference_and_positive_section():
    row, detail = sections.blade_section()
    assert row[1:3] == pytest.approx([1.21e5, 2.87e6], rel=1e-12)
    assert detail['skin_thickness_m'] > 0
    assert detail['total_cap_area_m2'] > 0
    assert all(v > 0 for v in row[:6])
    # Effective bending/mass properties must describe the same area.
    area = row[3] / detail['E_pa']
    assert row[0] / detail['density_kg_m3'] == pytest.approx(area)
    assert row[0]*(row[9]*.5)**2 == pytest.approx(row[1]*detail['density_kg_m3']/detail['E_pa'])
    # Independent quadrature refinement, well below file precision needs.
    refined, _ = sections.blade_section(8193)
    assert np.asarray(row[:6]) == pytest.approx(refined[:6], rel=2e-6)


def test_tube_recovers_solid_circular_shaft_and_thin_wall_limits():
    solid = sections.tube_section(.1, 0)
    assert solid[9:11] == pytest.approx([.25, .25])
    assert solid[4] / solid[1] == pytest.approx(1/1.3)  # GJ/EI, J=2I
    wall = sections.tube_section(.1, .099999)
    assert wall[9] == pytest.approx(1/(8**.5), rel=2e-5)


def test_generated_inputs_are_current():
    files, _ = sections.render()
    root = SOURCE.parent/'10m_HVAWT_reference_turbine/Structure'
    for name, content in files.items():
        assert (root/name).read_text() == content
