"""Fundamental numerical contracts used by the archived 2025 NAWEA drivers.

These run without the paper checkout, plotting, or a structural solver.
The full historical figure/sweep regression belongs to the paper archive.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

import vorlap
from conftest import make_component, make_constant_airfoil_fft, make_viv_params
from scripts.process_naluout_to_h5 import compute_fft, main as convert_histories


def test_converter_uses_pitching_moment_and_only_shifts_non_dc_strouhal(tmp_path):
    """Mtz/(q*c), signed DC and the paper's non-DC +.07 offset end to end."""
    time = np.arange(128) / 128
    cm = -0.3 + 0.4 * np.cos(2 * np.pi * 5 * time + 0.6)
    source = tmp_path / "source"
    source.mkdir()
    data = np.zeros((128, 10))
    data[:, 0] = time
    q = 0.5 * 1.25 * 10.0**2 * 2.0 * 3.0
    data[:, 1] = q * 0.2
    data[:, 2] = q * 0.5
    data[:, 8] = q * 93.0  # Mty must not be interpreted as pitching moment.
    data[:, 9] = q * 2.0 * cm
    np.savetxt(source / "test_30.dat", data, header="Time Fpx Fpy Fpz Fvx Fvy Fvz Mtx Mty Mtz")
    target = tmp_path / "test.h5"
    convert_histories([
        "--input-dir", str(source), "--output", str(target), "--airfoil-name", "test",
        "--thickness", "0.18", "--chord", "2", "--span", "3", "--fluid-density", "1.25",
        "--fluid-viscosity", "0.01", "--re", "2500", "--initial-timestep-skip", "0",
        "--min-samples", "16", "--n-freq", "9", "--low-freq-skip", "0", "--st-offset", "0.07", "--no-plots",
    ])
    airfoil = vorlap.load_airfoil_fft(str(target))
    st = airfoil.CM_ST[0, 0].copy()
    assert st[0] == 0.0
    assert airfoil.CM_Amp[0, 0, 0] == pytest.approx(-0.3, abs=1e-12)
    st[1:] -= 0.07
    # c*sin(30deg)=1 m; undo only the explicit paper tuning before replay.
    actual = vorlap.reconstruct_signal(st * 10.0, airfoil.CM_Amp[0, 0], airfoil.CM_Pha[0, 0], time)
    np.testing.assert_allclose(actual, cm, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("samples", [128, 129])
@pytest.mark.parametrize("aoa,length_fraction", [(0.0, 0.18), (60.0, np.sqrt(3) / 2)])
def test_fft_to_dimensional_reconstruction_preserves_signed_mean_and_phase(samples, aoa, length_fraction):
    """Test the preprocessing/reconstruction interface after changing U and chord.

    The analytic record includes a negative mean, a phase-bearing low harmonic,
    and the highest positive bin (unpaired Nyquist for even-length records).
    """
    time = np.arange(samples) / samples
    upper_phase = 0.0 if samples % 2 == 0 else 0.4
    signal = -0.7 + 0.3 * np.cos(2 * np.pi * 7 * time - 0.6) + 0.12 * np.cos(2 * np.pi * (samples // 2) * time + upper_phase)
    _, _, _, _, st, amplitude, phase = compute_fft(
        signal, 1 / samples, chord=2.0, aoa_deg=aoa, vinf=20.0, thickness_ratio=0.18, low_freq_skip=0
    )
    # New speed 50 m/s and chord 4 m imply a 1.25x dimensional frequency.
    new_time = time / 1.25
    actual = vorlap.reconstruct_signal(st * 50.0 / (4.0 * length_fraction), amplitude, phase, new_time)
    np.testing.assert_allclose(actual, signal, rtol=1e-11, atol=1e-11)
    assert st[0] == 0.0
    assert amplitude[0] == pytest.approx(-0.7, abs=1e-12)


@pytest.mark.parametrize("solver", [vorlap.compute_thrust_torque_spectrum, vorlap.compute_thrust_torque_spectrum_optimized])
@pytest.mark.parametrize("include_mean", [True, False])
def test_multicomponent_history_integrates_to_physical_force_and_moment(solver, include_mean):
    """Analytic vertical blade: rotation, node weights, offsets, N and N m.

    A translated 4 m blade is discretized nonuniformly, preceded by a second
    component. Its node 2 must survive concatenation. Total force equals the
    uniform line-load integral and torque equals its independent centroid load.
    """
    dummy = make_component(n_nodes=2, span=1.0, airfoil_id="quiet")
    dummy.id = "Arm1"
    blade = make_component(n_nodes=3, span=4.0)
    blade.id = "Blade3"
    blade.shape_xyz[:, 2] = [0.0, 1.0, 4.0]
    blade.translation = np.array([3.0, 2.0, 0.0])
    # Force preparation from geometry, independent of graphics or preset frames.
    blade.chord_vector[:] = 0.0
    blade.normal_vector[:] = 0.0
    airfoil = make_constant_airfoil_fft(cl_dc=-0.4, cd_dc=0.2, cl_h1_amp=0.3, cd_h1_amp=0.1, st_h1=0.2)
    airfoil.CL_Pha[..., 1] = 0.6
    airfoil.CD_Pha[..., 1] = -0.3
    quiet = make_constant_airfoil_fft(cl_dc=0.0, cd_dc=0.0, cf_h1_amp=0.0)
    p = make_viv_params()
    p.fluid_density = 1.25
    p.fluid_dynamicviscosity = 1e-5
    p.inflow_vec = np.array([1.0, 0.0, 0.0])
    p.azimuths = np.array([60.0])
    p.inflow_speeds = np.array([6.0])
    p.rotation_axis_offset = np.array([1.0, 1.0, 0.0])
    p.output_azimuth_vinf = (60.0, 6.0)
    f = 0.2 * 6.0 / (np.sqrt(3) / 2)
    p.output_time = np.arange(256) / (256 * f)
    _, _, mean_force, mean_moment, nodes = solver([dummy, blade], {"default": airfoil, "quiet": quiet}, p, np.array([f]), include_mean_load=include_mean)

    c = np.array([0.5, np.sqrt(3) / 2, 0.0])
    n = np.array([-np.sqrt(3) / 2, 0.5, 0.0])
    q_per_m = 0.5 * 1.25 * 6.0**2
    cd = 0.1 * np.cos(2 * np.pi * f * p.output_time - 0.3) + (0.2 if include_mean else 0.0)
    cl = 0.3 * np.cos(2 * np.pi * f * p.output_time + 0.6) - (0.4 if include_mean else 0.0)
    force_per_m = q_per_m * (cd[:, None] * c + cl[:, None] * n)
    # Trapezoidal support lengths for the [0,1,4] grid: [.5,2,1.5] m.
    expected_nodes = force_per_m[:, :, None] * np.array([0.5, 2.0, 1.5])
    np.testing.assert_allclose(nodes[:, :, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(nodes[:, :, 2:], expected_nodes, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(nodes[:, :, 3], 2.0 * force_per_m, rtol=1e-11, atol=1e-11)
    total = 4.0 * q_per_m * (0.2 * c - 0.4 * n) if include_mean else np.zeros(3)
    np.testing.assert_allclose(mean_force[0, 0], total, atol=1e-11)
    np.testing.assert_allclose(nodes.sum(axis=2).mean(axis=0), total, atol=1e-11)
    # Rotated centroid (2,1,2) about the offset, written out independently.
    lever = np.array([1 - np.sqrt(3) / 2, np.sqrt(3) + 0.5, 2.0])
    np.testing.assert_allclose(mean_moment[0, 0], np.cross(lever, total), atol=1e-11)


@pytest.mark.parametrize("solver", [vorlap.compute_thrust_torque_spectrum, vorlap.compute_thrust_torque_spectrum_optimized])
def test_overlap_grid_preserves_frequency_scaling_and_peak_selection(solver):
    """A 2x3 sweep exercises low-AoA thickness, harmonic matching and cutoff."""
    airfoil = make_constant_airfoil_fft(cf_dc=100.0, cf_h1_amp=0.6, st_h1=0.2, n_freq=3)
    airfoil.CF_ST[..., 2] = 0.3
    airfoil.CF_Amp[..., 2] = 0.1  # Would change overlap if cutoff were ignored.
    component = make_component(span=2.0)
    component.thickness[:] = 0.2
    p = make_viv_params()
    p.inflow_vec = np.array([1., 0., 0.])
    p.inflow_speeds = np.array([5.0, 10.0])
    p.azimuths = np.array([0.0, 30.0, 90.0])
    p.amplitude_coeff_cutoff = 0.5
    p.n_freq_depth = 3
    p.output_azimuth_vinf = (90., 10.)
    p.n_harmonic = 2
    actual = solver([component], {"default": airfoil}, p, np.array([2.0]))[0]
    # Selected frequencies: [[5,2,1],[10,4,2]] Hz; targets 2 and 4 Hz.
    np.testing.assert_allclose(actual, [[25., 0., 50.], [150., 0., 0.]], atol=1e-11)
    p.freq_min, p.freq_max = 1.5, 4.5
    filtered = solver([component], {"default": airfoil}, p, np.array([2.0]))[0]
    np.testing.assert_allclose(filtered, [[1000., 0., 1000.], [1000., 0., 0.]], atol=1e-11)
    p.n_freq_depth = 1
    only_dc = solver([component], {"default": airfoil}, p, np.array([2.0]))[0]
    np.testing.assert_array_equal(only_dc, np.full((2, 3), 1000.0))


def test_archived_ffa_verification_waveform_and_correlation(tmp_path):
    """Retain the actual nine-entry paper reconstruction, beyond rounded r=.90."""
    with np.load(Path(__file__).parent / "fixtures/nawea2025/ffa_verification.npz") as fixture:
        h5_path = tmp_path / "ffa_w3_211.h5"
        with h5py.File(h5_path, "w") as h5:
            for name in ("Re", "AOA", "Thickness"):
                h5[name] = fixture[name]
            for field in ("CL", "CD", "CM", "CF"):
                for suffix in ("ST", "Amp", "Pha"):
                    h5[field + "_" + suffix] = fixture[field + "_" + suffix]
        airfoil = vorlap.load_airfoil_fft(str(h5_path))
        frequency = airfoil.CL_ST[0, 0] * 75.0 / np.sin(np.deg2rad(144.0))
        actual = -0.5 * 1.225 * 75.0**2 * 4.0 * vorlap.reconstruct_signal(
            frequency, airfoil.CL_Amp[0, 0], airfoil.CL_Pha[0, 0], fixture["time"]
        )
        np.testing.assert_allclose(actual, fixture["reconstructed"], rtol=1e-10, atol=1e-8)
        correlation = np.corrcoef(actual, fixture["original"])[0, 1]
        assert correlation == pytest.approx(0.9040041661508004, abs=1e-8)
