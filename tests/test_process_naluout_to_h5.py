from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from vorlap.airfoil_io import load_airfoil_fft


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "process_naluout_to_h5.py"


def load_converter():
    spec = importlib.util.spec_from_file_location("process_naluout_to_h5", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["process_naluout_to_h5"] = module
    spec.loader.exec_module(module)
    return module


def test_compute_fft_uses_thickness_floor_at_zero_aoa():
    converter = load_converter()
    dt = 0.01
    time = np.arange(256, dtype=float) * dt
    signal = 0.2 + 0.5 * np.cos(2.0 * np.pi * 5.0 * time)

    _, _, _, _, strouhal, amps, _ = converter.compute_fft(
        signal,
        dt,
        chord=2.0,
        aoa_deg=0.0,
        vinf=10.0,
        thickness_ratio=0.12,
        low_freq_skip=0,
    )

    assert strouhal[0] == 0.0
    assert amps[0] == pytest.approx(np.mean(signal))
    dominant = int(np.argmax(amps[1:]) + 1)
    assert strouhal[dominant] == pytest.approx(5.0 * 2.0 * 0.12 / 10.0, rel=2.0e-2)


def test_compute_fft_rejects_zero_reference_length():
    converter = load_converter()

    with pytest.raises(ValueError, match="reference length collapsed"):
        converter.compute_fft(
            np.ones(32),
            0.01,
            chord=1.0,
            aoa_deg=0.0,
            vinf=2.0,
            thickness_ratio=0.0,
            low_freq_skip=0,
        )


def test_converter_cli_writes_hdf5_with_source_provenance(tmp_path):
    converter = load_converter()
    input_dir = tmp_path / "source" / "RE1_0E5" / "data_files"
    input_dir.mkdir(parents=True)
    dt = 0.01
    time = np.arange(128, dtype=float) * dt
    data = np.zeros((time.size, 9), dtype=float)
    data[:, 0] = time
    data[:, 2] = 0.2 + 0.05 * np.cos(2.0 * np.pi * 5.0 * time)
    data[:, 5] = 0.03 * np.cos(2.0 * np.pi * 7.0 * time)
    data[:, 1] = 0.1 + 0.02 * np.sin(2.0 * np.pi * 4.0 * time)
    data[:, 4] = 0.01 * np.sin(2.0 * np.pi * 8.0 * time)
    data[:, 8] = 0.04 * np.cos(2.0 * np.pi * 6.0 * time)
    source = input_dir / "testfoil_120_0.dat"
    np.savetxt(source, data, header="t fpx fpy x fvx fvy y z mty")
    output = tmp_path / "testfoil.h5"

    summary = converter.main(
        [
            "--input-dir",
            str(tmp_path / "source"),
            "--output",
            str(output),
            "--airfoil-name",
            "testfoil_120",
            "--thickness",
            "0.12",
            "--n-freq",
            "4",
            "--low-freq-skip",
            "0",
            "--min-samples",
            "32",
            "--no-plots",
        ]
    )

    assert summary["source_files"] == 1
    assert summary["resampled_source_files"] == 0
    with h5py.File(output, "r") as h5:
        assert h5["Airfoilname"][()].decode("utf-8") == "testfoil_120"
        assert h5["Thickness"][()] == pytest.approx(0.12)
        assert h5["Re"].shape == (1,)
        assert h5["CL_ST"].shape == (1, 1, 4)
        assert h5["CL_ST"][0, 0, 0] == 0.0
        assert h5.attrs["generator"] == "scripts/process_naluout_to_h5.py"
        assert len(h5.attrs["source_data_sha256"]) == 64
        source_rows = json.loads(h5.attrs["source_files_json"])
        assert source_rows[0]["path"] == str(source.resolve())
        assert source_rows[0]["samples"] == 128

    airfoil = load_airfoil_fft(output)
    assert airfoil.name == "testfoil_120"
    assert airfoil.Re.tolist() == pytest.approx([1.0e5])


def test_flat_folder_re_and_resampled_time_history(tmp_path):
    converter = load_converter()
    input_dir = tmp_path / "naca" / "data_files"
    input_dir.mkdir(parents=True)
    dt = 0.01
    time = np.arange(128, dtype=float) * dt
    time[64:] += 0.001
    data = np.zeros((time.size, 9), dtype=float)
    data[:, 0] = time
    data[:, 1] = 0.05 + 0.02 * np.sin(2.0 * np.pi * 4.0 * time)
    data[:, 2] = 0.20 + 0.04 * np.cos(2.0 * np.pi * 5.0 * time)
    data[:, 4] = 0.01 * np.sin(2.0 * np.pi * 6.0 * time)
    data[:, 5] = 0.02 * np.cos(2.0 * np.pi * 7.0 * time)
    data[:, 8] = 0.03 * np.cos(2.0 * np.pi * 8.0 * time)
    source = input_dir / "NACA0018_4.dat"
    np.savetxt(source, data, header="t fpx fpy x fvx fvy y z mty")
    output = tmp_path / "NACA0018.h5"

    summary = converter.main(
        [
            "--input-dir",
            str(tmp_path / "naca"),
            "--output",
            str(output),
            "--airfoil-name",
            "NACA0018",
            "--re",
            "5e5",
            "--n-freq",
            "4",
            "--low-freq-skip",
            "0",
            "--min-samples",
            "32",
            "--no-plots",
        ]
    )

    assert summary["thickness_ratio"] == pytest.approx(0.18)
    assert summary["reynolds"] == pytest.approx([5.0e5])
    assert summary["resampled_source_files"] == 1
    airfoil = load_airfoil_fft(output)
    assert airfoil.Re.tolist() == pytest.approx([5.0e5])
    assert airfoil.AOA.tolist() == pytest.approx([4.0])
