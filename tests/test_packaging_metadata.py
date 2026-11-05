from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_declares_entry_points_and_dependencies():
    repo = Path(__file__).resolve().parents[1]
    data = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))

    project = data["project"]
    assert project["name"] == "vorlap"
    assert project["requires-python"] == ">=3.9"
    assert {"numpy", "scipy", "pandas", "h5py", "plotly"} <= set(project["dependencies"])
    assert set(project["optional-dependencies"]) >= {"dev", "docs", "gui"}
    assert project["scripts"] == {
        "vorlap-gui": "vorlap.gui:main",
        "vorlap-qblade-prepare": "scripts.prepare_qblade_external_case:main",
        "vorlap-qblade-run": "run_headless.run_qblade_sim:main",
    }


def test_pyproject_data_files_cover_actual_repo_data_layout():
    repo = Path(__file__).resolve().parents[1]
    data = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = [
        pattern
        for entries in data["tool"]["setuptools"]["data-files"].values()
        for pattern in entries
    ]

    configured_files = {
        path.relative_to(repo).as_posix()
        for pattern in patterns
        for path in repo.glob(pattern)
        if path.is_file()
    }
    required_files = {
        path.relative_to(repo).as_posix()
        for path in (repo / "data").rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".h5", ".json"}
    }

    assert required_files
    assert required_files <= configured_files


def test_setup_py_delegates_to_pyproject_metadata():
    repo = Path(__file__).resolve().parents[1]
    setup_text = (repo / "setup.py").read_text(encoding="utf-8")

    assert "setup()" in setup_text
    assert "package_data" not in setup_text
    assert "data/airfoils" not in setup_text
