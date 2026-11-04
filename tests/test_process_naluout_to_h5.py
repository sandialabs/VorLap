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
EXTENSION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "extend_replicated_reynolds_grid.py"
)
PLOT_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "plot_naluout_per_aoa.py"
)


def load_converter():
    spec = importlib.util.spec_from_file_location("process_naluout_to_h5", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["process_naluout_to_h5"] = module
    spec.loader.exec_module(module)
    return module


def load_plotter():
    spec = importlib.util.spec_from_file_location("plot_naluout_per_aoa", PLOT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["plot_naluout_per_aoa"] = module
    spec.loader.exec_module(module)
    return module


def load_reynolds_extender():
    spec = importlib.util.spec_from_file_location(
        "extend_replicated_reynolds_grid", EXTENSION_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["extend_replicated_reynolds_grid"] = module
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


def test_compute_fft_even_length_retains_nyquist_without_doubling():
    converter = load_converter()
    n_samples = 16
    dt = 0.1
    time = np.arange(n_samples, dtype=float) * dt
    nyquist_hz = 0.5 / dt
    signal = 0.3 + 0.4 * np.cos(2.0 * np.pi * nyquist_hz * time)

    freqs, amps, phases, _, _, _, _ = converter.compute_fft(
        signal,
        dt,
        chord=1.0,
        aoa_deg=90.0,
        vinf=1.0,
        thickness_ratio=0.12,
        low_freq_skip=0,
    )

    assert freqs[-1] == pytest.approx(nyquist_hz)
    assert amps[0] == pytest.approx(0.3)
    assert amps[-1] == pytest.approx(0.4)
    assert np.cos(phases[-1]) == pytest.approx(1.0)


def test_compute_fft_odd_length_doubles_highest_positive_bin():
    converter = load_converter()
    n_samples = 9
    dt = 1.0 / n_samples
    time = np.arange(n_samples, dtype=float) * dt
    highest_positive_hz = 4.0
    signal = 0.3 + 0.4 * np.cos(2.0 * np.pi * highest_positive_hz * time)

    freqs, amps, _, _, _, _, _ = converter.compute_fft(
        signal,
        dt,
        chord=1.0,
        aoa_deg=90.0,
        vinf=1.0,
        thickness_ratio=0.12,
        low_freq_skip=0,
    )

    assert freqs[-1] == pytest.approx(highest_positive_hz)
    assert amps[-1] == pytest.approx(0.4)


def test_plot_psd_odd_length_doubles_highest_positive_bin():
    plotter = load_plotter()
    n_samples = 9
    dt = 1.0 / n_samples
    time = np.arange(n_samples, dtype=float) * dt
    signal = np.cos(2.0 * np.pi * 4.0 * time)

    frequencies, psd = plotter.one_sided_psd(signal, dt)
    spectrum = np.fft.rfft((signal - np.mean(signal)) * np.hanning(n_samples))
    unscaled = (dt / np.sum(np.hanning(n_samples) ** 2)) * np.abs(spectrum) ** 2

    assert frequencies[-1] == pytest.approx(4.0)
    assert psd[-1] == pytest.approx(2.0 * unscaled[-1])


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
    data = np.zeros((time.size, 10), dtype=float)
    data[:, 0] = time
    data[:, 2] = 0.2 + 0.05 * np.cos(2.0 * np.pi * 5.0 * time)
    data[:, 5] = 0.03 * np.cos(2.0 * np.pi * 7.0 * time)
    data[:, 1] = 0.1 + 0.02 * np.sin(2.0 * np.pi * 4.0 * time)
    data[:, 4] = 0.01 * np.sin(2.0 * np.pi * 8.0 * time)
    data[:, 9] = 0.04 * np.cos(2.0 * np.pi * 6.0 * time)
    source = input_dir / "testfoil_120_0.dat"
    np.savetxt(source, data, header="t fpx fpy fpz fvx fvy fvz mtx mty mtz")
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
            "--initial-timestep-skip",
            "0",
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
        assert source_rows[0]["skipped_initial_steps"] == 0
        assert source_rows[0]["samples"] == 128

    airfoil = load_airfoil_fft(output)
    assert airfoil.name == "testfoil_120"
    assert airfoil.Re.tolist() == pytest.approx([1.0e5])


def test_converter_can_embed_portable_relative_source_paths(tmp_path):
    converter = load_converter()
    root = tmp_path / "source_bundle"
    data_dir = root / "ffa_w3_211" / "data_files"
    data_dir.mkdir(parents=True)
    time = np.arange(64, dtype=float) * 0.01
    data = np.zeros((time.size, 10), dtype=float)
    data[:, 0] = time
    data[:, 1] = np.cos(2.0 * np.pi * 3.0 * time)
    data[:, 2] = np.sin(2.0 * np.pi * 4.0 * time)
    source = data_dir / "ffa_w3_211_0.dat"
    np.savetxt(source, data, header="t fpx fpy fpz fvx fvy fvz mtx mty mtz")
    output = tmp_path / "portable.h5"

    converter.main(
        [
            "--input-dir",
            str(root / "ffa_w3_211"),
            "--source-path-root",
            str(root),
            "--output",
            str(output),
            "--re",
            "1e7",
            "--n-freq",
            "4",
            "--low-freq-skip",
            "0",
            "--min-samples",
            "32",
            "--initial-timestep-skip",
            "0",
            "--no-plots",
        ]
    )

    with h5py.File(output, "r") as h5:
        rows = json.loads(h5.attrs["source_files_json"])
        assert rows[0]["path"] == "ffa_w3_211/data_files/ffa_w3_211_0.dat"
        assert rows[0]["path_kind"] == "relative_to_source_path_root"


def test_force_loader_can_retain_exact_final_window(tmp_path):
    converter = load_converter()
    path = tmp_path / "history.dat"
    time = np.arange(100, dtype=float) * 0.01
    data = np.zeros((time.size, 10), dtype=float)
    data[:, 0] = time
    np.savetxt(path, data, header="t fpx fpy fpz fvx fvy fvz mtx mty mtz")

    window, info = converter.load_force_history(
        path,
        min_samples=20,
        initial_timestep_skip=0,
        final_timestep_count=40,
    )
    assert window.shape == (40, 10)
    assert info.skipped_initial_steps == 60
    assert info.time_start_s == pytest.approx(0.60)
    with pytest.raises(ValueError, match="mutually exclusive"):
        converter.load_force_history(
            path,
            min_samples=20,
            initial_timestep_skip=1,
            final_timestep_count=40,
        )


def test_flat_folder_re_and_resampled_time_history(tmp_path):
    converter = load_converter()
    input_dir = tmp_path / "naca" / "data_files"
    input_dir.mkdir(parents=True)
    dt = 0.01
    time = np.arange(128, dtype=float) * dt
    time[64:] += 0.001
    data = np.zeros((time.size, 10), dtype=float)
    data[:, 0] = time
    data[:, 1] = 0.05 + 0.02 * np.sin(2.0 * np.pi * 4.0 * time)
    data[:, 2] = 0.20 + 0.04 * np.cos(2.0 * np.pi * 5.0 * time)
    data[:, 4] = 0.01 * np.sin(2.0 * np.pi * 6.0 * time)
    data[:, 5] = 0.02 * np.cos(2.0 * np.pi * 7.0 * time)
    data[:, 9] = 0.03 * np.cos(2.0 * np.pi * 8.0 * time)
    source = input_dir / "NACA0018_4.dat"
    np.savetxt(source, data, header="t fpx fpy fpz fvx fvy fvz mtx mty mtz")
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
            "--initial-timestep-skip",
            "0",
            "--no-plots",
        ]
    )

    assert summary["thickness_ratio"] == pytest.approx(0.18)
    assert summary["reynolds"] == pytest.approx([5.0e5])
    assert summary["resampled_source_files"] == 1
    airfoil = load_airfoil_fft(output)
    assert airfoil.Re.tolist() == pytest.approx([5.0e5])
    assert airfoil.AOA.tolist() == pytest.approx([4.0])


def test_converter_can_declare_reynolds_independent_replication_grid(tmp_path):
    converter = load_converter()
    input_dir = tmp_path / "naca" / "data_files"
    input_dir.mkdir(parents=True)
    dt = 0.01
    time = np.arange(128, dtype=float) * dt
    data = np.zeros((time.size, 10), dtype=float)
    data[:, 0] = time
    data[:, 1] = 0.05 + 0.02 * np.sin(2.0 * np.pi * 4.0 * time)
    data[:, 2] = 0.20 + 0.04 * np.cos(2.0 * np.pi * 5.0 * time)
    data[:, 4] = 0.01 * np.sin(2.0 * np.pi * 6.0 * time)
    data[:, 5] = 0.02 * np.cos(2.0 * np.pi * 7.0 * time)
    data[:, 9] = 0.03 * np.cos(2.0 * np.pi * 8.0 * time)
    np.savetxt(
        input_dir / "NACA0018_4.dat",
        data,
        header="t fpx fpy fpz fvx fvy fvz mtx mty mtz",
    )
    output = tmp_path / "NACA0018.h5"
    justification = "Conservative Re-independent parked-DLC source model"

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
            "--initial-timestep-skip",
            "0",
            "--replicate-re-grid",
            "3e4 5e5 2e6",
            "--replicate-re-justification",
            justification,
            "--no-plots",
        ]
    )

    assert summary["source_reynolds"] == pytest.approx([5.0e5])
    assert summary["reynolds"] == pytest.approx([3.0e4, 5.0e5, 2.0e6])
    assert summary["reynolds_model"] == "replicated_reynolds_independent"
    airfoil = load_airfoil_fft(output)
    assert airfoil.reynolds_model == "replicated_reynolds_independent"
    with h5py.File(output, "r") as h5:
        assert h5["Re"][:].tolist() == pytest.approx([3.0e4, 5.0e5, 2.0e6])
        assert h5["Thickness"][()] == pytest.approx(0.18)
        assert h5.attrs["reynolds_model"] == "replicated_reynolds_independent"
        assert h5.attrs["reynolds_replication_justification"] == justification
        config = json.loads(h5.attrs["processing_config_json"])
        assert config["source_reynolds"] == pytest.approx([5.0e5])
        assert config["output_reynolds"] == pytest.approx([3.0e4, 5.0e5, 2.0e6])
        assert config["replicate_re_justification"] == justification
        for name in ("CL_ST", "CL_Amp", "CD_ST", "CF_Amp"):
            np.testing.assert_allclose(h5[name][0], h5[name][1])
            np.testing.assert_allclose(h5[name][1], h5[name][2])


def test_reynolds_replication_requires_explicit_justification():
    converter = load_converter()
    args = converter.parse_args(["--replicate-re-grid", "3e4 5e5 2e6"])

    with pytest.raises(ValueError, match="requires a nonempty"):
        converter.config_from_args(args)


def write_replicated_parent_h5(path: Path, *, mismatch: bool = False) -> None:
    parent_grid = [30_000.0, 500_000.0, 2_000_000.0]
    spectrum = np.arange(12, dtype=np.float64).reshape(2, 6)
    replicated = np.stack([spectrum, spectrum, spectrum])
    if mismatch:
        replicated[2, 0, 0] += 1.0
    processing = {
        "airfoil_name": "NACA0018",
        "source_reynolds": [500_000.0],
        "output_reynolds": parent_grid,
        "replicate_re_grid": parent_grid,
        "replicate_re_justification": "approved parent",
        "reynolds_model": "replicated_reynolds_independent",
    }
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "Airfoilname", data=np.array("NACA0018", dtype=h5py.string_dtype("utf-8"))
        )
        h5.create_dataset("Re", data=parent_grid)
        h5.create_dataset("AOA", data=[-180.0, 180.0])
        h5.create_dataset("Thickness", data=0.18)
        h5.create_dataset("CL_Amp", data=replicated)
        h5.create_dataset("CL_ST", data=replicated + 0.5)
        h5.create_dataset("CL_Pha", data=replicated - 0.5)
        h5.attrs["generator"] = "scripts/process_naluout_to_h5.py"
        h5.attrs["source_data_sha256"] = "b" * 64
        h5.attrs["source_files_json"] = "[]"
        h5.attrs["processing_config_json"] = json.dumps(processing, sort_keys=True)
        h5.attrs["reynolds_model"] = "replicated_reynolds_independent"
        h5.attrs["reynolds_replication_justification"] = "approved parent"


def test_exact_reynolds_envelope_extension_preserves_all_non_re_datasets(tmp_path):
    extender = load_reynolds_extender()
    parent = tmp_path / "parent.h5"
    output = tmp_path / "extended.h5"
    write_replicated_parent_h5(parent)
    parent_sha256 = extender.sha256_file(parent)

    report = extender.extend_replicated_reynolds_grid(
        parent=parent,
        output=output,
        expected_parent_sha256=parent_sha256,
        reynolds=[10_000.0, 500_000.0, 2_000_000.0],
        justification="U03 reached chord Re=19,213; extend the replicated envelope to Re=10,000.",
    )

    assert report["non_re_datasets_preserved_exactly"] is True
    assert report["parent_sha256"] == parent_sha256
    with h5py.File(parent, "r") as source, h5py.File(output, "r") as revised:
        assert revised["Re"][:].tolist() == pytest.approx(
            [10_000.0, 500_000.0, 2_000_000.0]
        )
        assert set(source) == set(revised)
        for name in source:
            if name != "Re":
                assert np.array_equal(source[name][...], revised[name][...]), name
        processing = json.loads(revised.attrs["processing_config_json"])
        assert processing["replicate_re_parent_sha256"] == parent_sha256
        assert processing["reynolds_envelope_extension_exact"] is True
        assert processing["reynolds_envelope_parent_grid"] == pytest.approx(
            [30_000.0, 500_000.0, 2_000_000.0]
        )
        assert processing["reynolds_envelope_extension_generator"] == (
            "scripts/extend_replicated_reynolds_grid.py"
        )


def test_exact_reynolds_envelope_extension_rejects_nonidentical_re_planes(tmp_path):
    extender = load_reynolds_extender()
    parent = tmp_path / "nonidentical.h5"
    write_replicated_parent_h5(parent, mismatch=True)

    with pytest.raises(ValueError, match="not exactly replicated across Re"):
        extender.extend_replicated_reynolds_grid(
            parent=parent,
            output=tmp_path / "should_not_exist.h5",
            expected_parent_sha256=extender.sha256_file(parent),
            reynolds=[10_000.0, 500_000.0, 2_000_000.0],
            justification="test",
        )


def test_close_periodic_aoa_grid_appends_positive_180_copy():
    converter = load_converter()
    aoa = np.arange(-180.0, 180.0, 4.0)
    values = np.arange(2 * aoa.size * 3, dtype=float).reshape(2, aoa.size, 3)
    arrays = {"CL_ST": values, "CL_Amp": values + 100.0}

    closed, closed_aoa, endpoint_added = converter.close_periodic_aoa_grid(
        arrays,
        aoa,
    )

    assert endpoint_added is True
    assert closed_aoa[0] == pytest.approx(-180.0)
    assert closed_aoa[-2:].tolist() == pytest.approx([176.0, 180.0])
    for name, original in arrays.items():
        assert closed[name].shape == (2, aoa.size + 1, 3)
        np.testing.assert_allclose(closed[name][:, :-1, :], original)
        np.testing.assert_allclose(closed[name][:, -1, :], original[:, 0, :])


def test_close_periodic_aoa_grid_rejects_incomplete_circle():
    converter = load_converter()
    aoa = np.array([-180.0, -120.0, -60.0, 0.0, 60.0])
    arrays = {"CL_ST": np.zeros((1, aoa.size, 2))}

    with pytest.raises(ValueError, match="does not reach the periodic seam"):
        converter.close_periodic_aoa_grid(arrays, aoa)


def test_load_experimental_overlay_reads_headerless_swalwell_tables(tmp_path):
    converter = load_converter()
    tables = {
        "Lift_St_RE2p7E5.csv": [[40.0, 0.18], [20.0, 0.20]],
        "Drag_St_RE2p7E5.csv": [[20.0, 0.20], [40.0, 0.18]],
        "Lift_PSD_RE2p7E5.csv": [[20.0, 1.5], [40.0, 2.5]],
        "Drag_PSD_RE2p7E5.csv": [[20.0, 0.5], [40.0, 0.8]],
    }
    for name, rows in tables.items():
        np.savetxt(tmp_path / name, np.asarray(rows), delimiter=",")

    overlay = converter.load_experimental_overlay(tmp_path)

    assert overlay.label == "Swalwell et al. NACA0021 (Re=2.7E5)"
    assert overlay.lift_st[:, 0].tolist() == [20.0, 40.0]
    assert overlay.lift_st[:, 1].tolist() == pytest.approx([0.20, 0.18])
    assert overlay.drag_psd[:, 1].tolist() == pytest.approx([0.0005, 0.0008])
    assert overlay.psd_scale == pytest.approx(1.0e-3)
    assert len(overlay.source_files) == 4


def test_initial_timestep_skip_drops_startup_rows_before_resampling(tmp_path):
    converter = load_converter()
    dt = 0.1
    time = np.arange(12, dtype=float) * dt
    data = np.zeros((time.size, 10), dtype=float)
    data[:, 0] = time
    data[:, 2] = np.arange(time.size, dtype=float)
    source = tmp_path / "startup_0.dat"
    np.savetxt(source, data, header="t fpx fpy fpz fvx fvy fvz mtx mty mtz")

    trimmed, info = converter.load_force_history(
        source,
        min_samples=4,
        initial_timestep_skip=5,
    )

    assert info.source_samples == 12
    assert info.skipped_initial_steps == 5
    assert info.samples == 7
    assert info.time_start_s == pytest.approx(0.5)
    assert trimmed[0, 0] == pytest.approx(0.5)
    assert trimmed[0, 2] == pytest.approx(5.0)
