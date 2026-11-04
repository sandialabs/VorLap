from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_bidadi2024_ffa_spectral_bundle.py"
)


def load_builder():
    scripts = SCRIPT.parent
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("build_bidadi2024_bundle", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def _write_minimal_sources(root: Path, builder) -> None:
    header = "Time Fpx Fpy Fpz Fvx Fvy Fvz Mtx Mty Mtz Y+min Y+max\n"
    for profile in builder.PROFILES:
        directory = root / f"ffa_w3_{profile}" / "data_files"
        directory.mkdir(parents=True)
        for angle in builder.EXPECTED_AOA_DEG:
            (directory / f"ffa_w3_{profile}_{angle}.dat").write_text(
                header + "0 0 0 0 0 0 0 0 0 0 0 0\n", encoding="utf-8"
            )


def test_source_inventory_requires_exact_complete_profile_grid(tmp_path):
    builder = load_builder()
    _write_minimal_sources(tmp_path, builder)
    rows = builder.source_inventory(tmp_path)
    assert len(rows) == 7 * 59
    assert rows[0]["path"].startswith("ffa_w3_211/data_files/")
    assert rows[-1]["profile"] == "FFA-W3-500"

    missing = tmp_path / "ffa_w3_301" / "data_files" / "ffa_w3_301_90.dat"
    missing.unlink()
    with pytest.raises(ValueError, match="59-angle grid"):
        builder.source_inventory(tmp_path)


def test_source_inventory_rejects_wrong_force_contract(tmp_path):
    builder = load_builder()
    _write_minimal_sources(tmp_path, builder)
    source = tmp_path / "ffa_w3_211" / "data_files" / "ffa_w3_211_0.dat"
    source.write_text("Time lift drag\n0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="force-history header"):
        builder.source_inventory(tmp_path)
