#!/usr/bin/env python3
"""Regenerate the computational figures and checks for the 2025 NAWEA paper.

The published PDFs are archival assets and are not overwritten.  This driver
writes scientifically equivalent figures and a machine-readable summary to a
separate build directory using the reviewer-requested spectral conventions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import warnings
import zipfile

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.getenv("TMPDIR", "/tmp")) / "vorlap-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.getenv("TMPDIR", "/tmp")) / "vorlap-cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import vorlap


COLORS = ["#348ABD", "#A60628", "#009E73", "#7A68A6", "#D55E00", "#CC79A7"]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORS)

PAPER_FIGURES = (
    "fig1_CL_time_AOA164NACA0018_Re5e_05.pdf",
    "fig2_CL_psd_AOA164NACA0018_Re5e_05.pdf",
    "fig3_CD_time_AOA164NACA0018_Re5e_05.pdf",
    "fig4_CD_psd_AOA164NACA0018_Re5e_05.pdf",
    "fig5_NACA0018STCF_summary_Re5e_05.pdf",
    "fig6_NACA0018AmpCF_summary_Re5e_05.pdf",
    "fig7_HVAWT_3D.png",
    "fig8_ReconstructedForce_Reconstruction.pdf",
    "fig9_worst_percent_diff_single_blade_Reconstruction.pdf",
    "fig14_torque.pdf",
    "fig15_worst_percent_diff.pdf",
    "fig16_ReconstructedForce.pdf",
    "CL_CD_time_AOA84NACA0018_Re5e_05.pdf",
    "CL_CD_psd_AOA84NACA0018_Re5e_05.pdf",
)
CORRECTED_COMPANION_FIGURES = (
    "fig14_torque_corrected.pdf",
    "fig16_ReconstructedForce_blade3_node2.pdf",
)

NACA_ARCHIVE_SHA256 = "a46218ceebb238a0377299e865a850d35ba2d94be3fdc0908da66d19ac41bc93"
LEGACY_NACA_H5_SHA256 = "8f6eee280f6f67353018d7aa13551d481a7fc7db52ac8c461a77f306dace054e"
PAPER_NACA_HISTORIES_SHA256 = "bd30575e8b1a4f2c9c64c3e991cbe53308ee4a7837c2dc6d04b7c4870266cac2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_converter():
    path = REPO / "scripts" / "process_naluout_to_h5.py"
    spec = importlib.util.spec_from_file_location("process_naluout_to_h5", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def save_figure(fig: plt.Figure, path: Path, *, transparent: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150, transparent=transparent)
    plt.close(fig)
    return str(path)


def prepare_naca_source(output_dir: Path) -> tuple[Path, dict[str, object]]:
    archive = REPO / "scripts" / "NACA18.zip"
    source_root = output_dir / "source"
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(source_root)

    converter = load_converter()
    h5_path = output_dir / "NACA0018_Re5e5_rebuilt.h5"
    summary = converter.main(
        [
            "--input-dir",
            str(source_root / "NACA0018"),
            "--output",
            str(h5_path),
            "--airfoil-name",
            "NACA0018",
            "--thickness",
            "0.18",
            "--span",
            "1.0",
            "--re",
            "5e5",
            "--symmetric-append",
            "--st-offset",
            "0.07",
            "--low-freq-skip",
            "30",
            "--initial-timestep-skip",
            "1000",
            "--no-plots",
        ]
    )

    with h5py.File(h5_path, "r") as h5:
        assert float(h5["Thickness"][()]) == 0.18
        for name in ("CL_ST", "CD_ST", "CM_ST", "CF_ST"):
            assert np.all(h5[name][..., 0] == 0.0)
            assert np.all(h5[name][..., 1:] >= 0.07)

    summary["archive_sha256"] = sha256(archive)
    summary["rebuilt_h5_sha256"] = sha256(h5_path)
    return source_root / "NACA0018", summary


def naca_history_figures(figures_dir: Path) -> dict[str, object]:
    converter = load_converter()
    rho = 1.2
    viscosity = 9.0e-6
    reynolds = 5.0e5
    chord = 1.0
    span = 4.0
    vinf = reynolds * viscosity / (rho * chord)
    archive = PAPER_DIR / "inputs" / "naca_re5e5_paper_histories.zip"
    if sha256(archive) != PAPER_NACA_HISTORIES_SHA256:
        raise RuntimeError("Publication-era NACA history archive hash does not match provenance")
    results: dict[str, object] = {
        "vinf_m_s": vinf,
        "reynolds": reynolds,
        "source_span_m": span,
        "archive_sha256": sha256(archive),
    }

    histories: dict[int, dict[str, np.ndarray]] = {}
    member_hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive) as zipped:
        for aoa in (84, 164):
            member = f"NACA0018_{aoa}.dat"
            payload = zipped.read(member)
            member_hashes[member] = hashlib.sha256(payload).hexdigest()
            with zipped.open(member) as stream:
                data = np.loadtxt(stream, skiprows=1)
            q = 0.5 * rho * vinf**2 * chord * span
            histories[aoa] = {
                "time": data[:, 0],
                "CL": (data[:, 2] + data[:, 5]) / q,
                "CD": (data[:, 1] + data[:, 4]) / q,
            }
    results["member_sha256"] = member_hashes

    def psd(values: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        freqs, _, _, power, _, _, _ = converter.compute_fft(
            values,
            float(np.median(np.diff(time))),
            chord,
            90.0,
            vinf,
            0.18,
            low_freq_skip=0,
        )
        return freqs, power

    # Match the original plotting cutoff while retaining the complete record
    # for the full-record PSD calculation below.
    def publication_window(history: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        keep = (history["time"] >= 10.3) & (history["time"] <= 102.0)
        return {key: values[keep] for key, values in history.items()}

    full_hist = histories[164]
    hist = publication_window(full_hist)
    fig, axis_left = plt.subplots(figsize=(9.9, 5.9))
    axis_right = axis_left.twinx()
    axis_left.plot(hist["time"], hist["CL"], label="CL", color=COLORS[0])
    axis_right.plot(hist["time"], hist["CD"], label="CD", color=COLORS[1])
    axis_left.set_xlabel("Time (s)")
    axis_left.set_ylabel("Lift Coefficient (CL)")
    axis_right.set_ylabel("Drag Coefficient (CD)")
    axis_left.legend(loc="upper left")
    axis_right.legend(loc="upper right")
    axis_left.grid(alpha=0.25)
    save_figure(fig, figures_dir / PAPER_FIGURES[0])

    f_cl, p_cl = psd(full_hist["CL"], full_hist["time"])
    f_cd, p_cd = psd(full_hist["CD"], full_hist["time"])
    fig, ax = plt.subplots(figsize=(9.9, 5.9))
    keep = (f_cl >= 0.09) & (f_cl <= 2.0)
    ax.plot(f_cl[keep], p_cl[keep], marker="x", label="CL PSD")
    ax.plot(f_cd[keep], p_cd[keep], marker="x", label="CD PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_xlim(0.09, 2.0)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / PAPER_FIGURES[1])

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(hist["time"], hist["CD"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CD")
    ax.set_title("NACA0018 (CD), AOA: 164.0 (Re=5e+05)")
    save_figure(fig, figures_dir / PAPER_FIGURES[2])

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(f_cd[keep], p_cd[keep])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (CD)")
    ax.set_title("NACA0018 (CD), AOA: 164.0 (Re=5e+05)")
    save_figure(fig, figures_dir / PAPER_FIGURES[3])

    full_hist = histories[84]
    hist = publication_window(full_hist)
    fig, axis_left = plt.subplots(figsize=(9.9, 5.9))
    axis_right = axis_left.twinx()
    axis_left.plot(hist["time"], hist["CL"], label="CL", color=COLORS[0])
    axis_right.plot(hist["time"], hist["CD"], label="CD", color=COLORS[1])
    axis_left.set_xlabel("Time (s)")
    axis_left.set_ylabel("Lift Coefficient (CL)")
    axis_right.set_ylabel("Drag Coefficient (CD)")
    axis_left.legend(loc="upper left")
    axis_right.legend(loc="upper right")
    axis_left.grid(alpha=0.25)
    save_figure(fig, figures_dir / "CL_CD_time_AOA84NACA0018_Re5e_05.pdf")

    f_cl, p_cl = psd(full_hist["CL"], full_hist["time"])
    f_cd, p_cd = psd(full_hist["CD"], full_hist["time"])
    keep = (f_cl >= 0.09) & (f_cl <= 2.0)
    fig, ax = plt.subplots(figsize=(9.9, 5.9))
    ax.plot(f_cl[keep], p_cl[keep], label="CL PSD")
    ax.plot(f_cd[keep], p_cd[keep], label="CD PSD")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_xlim(0.09, 2.0)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "CL_CD_psd_AOA84NACA0018_Re5e_05.pdf")

    results["aoa_164_cl_mean"] = float(np.mean(histories[164]["CL"]))
    results["aoa_164_cd_mean"] = float(np.mean(histories[164]["CD"]))
    results["aoa_84_cl_mean"] = float(np.mean(histories[84]["CL"]))
    results["aoa_84_cd_mean"] = float(np.mean(histories[84]["CD"]))
    return results


def corrected_legacy_naca(output_dir: Path) -> tuple[Path, dict[str, object]]:
    source = REPO / "data" / "airfoils" / "NACA0018.h5"
    target = output_dir / "NACA0018_paper_spectra_corrected.h5"
    shutil.copy2(source, target)
    with h5py.File(target, "r+") as h5:
        old_thickness = float(h5["Thickness"][()])
        h5["Thickness"][...] = 0.18
        old_dc = {}
        for name in ("CL_ST", "CD_ST", "CM_ST", "CF_ST"):
            old_dc[name] = sorted(float(value) for value in np.unique(h5[name][..., 0]))
            h5[name][..., 0] = 0.0
        h5.attrs["derived_from_sha256"] = sha256(source)
        h5.attrs["paper_release_corrections"] = json.dumps(
            {
                "thickness_ratio": 0.18,
                "dc_strouhal": 0.0,
                "nonzero_strouhal_unchanged": True,
            },
            sort_keys=True,
        )
    return target, {
        "legacy_h5_sha256": sha256(source),
        "corrected_h5_sha256": sha256(target),
        "old_thickness_ratio": old_thickness,
        "old_dc_strouhal": old_dc,
        "new_thickness_ratio": 0.18,
        "new_dc_strouhal": 0.0,
    }


def spectral_summary_figures(naca_h5: Path, figures_dir: Path) -> None:
    """Plot the historical processed spectra while omitting the corrected DC bin."""
    with h5py.File(naca_h5, "r") as h5:
        reynolds = np.asarray(h5["Re"], dtype=float)
        aoa = np.asarray(h5["AOA"], dtype=float)
        i_re = int(np.argmin(np.abs(reynolds - 5.0e5)))
        st = np.asarray(h5["CF_ST"][i_re, :, 1:30], dtype=float)
        amp = np.asarray(h5["CF_Amp"][i_re, :, 1:30], dtype=float)

    for values, ylabel, target, ylim in (
        (st, "Strouhal number based on CF", "fig5_NACA0018STCF_summary_Re5e_05.pdf", (0.0, 0.5)),
        (amp, "Frequency amplitude based on CF", "fig6_NACA0018AmpCF_summary_Re5e_05.pdf", None),
    ):
        fig, ax = plt.subplots(figsize=(9.9, 5.9))
        for index in range(values.shape[1]):
            ax.plot(
                aoa,
                values[:, index],
                marker="x",
                color=COLORS[0],
                linewidth=0.0,
                markersize=4,
            )
        ax.set_xlabel("AoA (°)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(float(np.min(aoa)), float(np.max(aoa)))
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0.0, 0.7)
        ax.grid(alpha=0.25)
        save_figure(fig, figures_dir / target)


def initialize_geometry(components, viv_params) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        vorlap.graphics.calc_structure_vectors_andplot(components, viv_params, show_plot=False)


def plot_geometry(components, figures_dir: Path) -> None:
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    for component in components:
        xyz = component.shape_xyz_global
        color = COLORS[0] if component.id.lower().startswith("blade") else "0.25"
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=2.0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=24.0, azim=-72.0)
    save_figure(fig, figures_dir / "fig7_HVAWT_3D.png", transparent=True)


def verification_figures(figures_dir: Path) -> dict[str, object]:
    components = vorlap.load_components_from_csv(
        str(REPO / "data" / "components" / "componentsSingle_Reconstruction")
    )
    for component in components:
        component.airfoil_ids = ["ffa_w3_211"] * len(component.airfoil_ids)

    # Use centers of complete 1 ms comparison intervals.  The raw history
    # starts at 0.13335 ms with a startup impulse, so the first incomplete
    # interval is not extrapolated into the reviewer-requested 0--2 s metric.
    time = np.arange(0.0005, 2.0, 0.001)
    viv_params = vorlap.VIV_Params(
        fluid_density=1.225,
        fluid_dynamicviscosity=1.81e-5,
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        rotation_axis_offset=np.array([0.0, 0.0, 0.0]),
        inflow_vec=np.array([1.0, 0.0, 0.0]),
        azimuths=np.arange(0.0, 360.0, 24.0),
        inflow_speeds=np.arange(0.0, 80.0, 5.0),
        n_harmonic=1,
        output_time=time,
        output_azimuth_vinf=(216.0, 75.0),
        amplitude_coeff_cutoff=0.2,
        n_freq_depth=20,
        airfoil_folder=str(REPO / "data" / "airfoils") + os.sep,
    )
    initialize_geometry(components, viv_params)
    afft = vorlap.load_airfoil_fft(str(REPO / "data" / "airfoils" / "ffa_w3_211.h5"))
    outputs = vorlap.compute_thrust_torque_spectrum_optimized(
        components,
        {"ffa_w3_211": afft, "default": afft},
        viv_params,
        np.array([15.0]),
    )
    percdiff = outputs[0]

    raw = np.loadtxt(
        REPO
        / "data"
        / "airfoils"
        / "2024_Ganesh_VIV_Paper_Data"
        / "ffa_data_files_ftt_160"
        / "ffa_w3_211"
        / "RE1_00E7"
        / "ffa_w3_211_144.dat",
        skiprows=1,
    )
    raw_force = -(raw[:, 2] + raw[:, 5])
    original = np.interp(time, raw[:, 0], raw_force)
    aoa_index = int(np.flatnonzero(np.isclose(afft.AOA, 144.0))[0])
    retained_depth = 9  # DC plus the eight dominant oscillatory peaks used in the plotted comparison.
    st_length = abs(np.sin(np.deg2rad(144.0)))
    frequencies = afft.CL_ST[0, aoa_index, :retained_depth] * 75.0 / st_length
    coefficient = vorlap.reconstruct_signal(
        frequencies,
        afft.CL_Amp[0, aoa_index, :retained_depth],
        afft.CL_Pha[0, aoa_index, :retained_depth],
        time,
    )
    q = 0.5 * 1.225 * 75.0**2 * 1.0 * 4.0
    reconstructed = -coefficient * q
    correlation = float(
        np.corrcoef(
            original - np.mean(original),
            reconstructed - np.mean(reconstructed),
        )[0, 1]
    )

    fig, ax = plt.subplots(figsize=(10.75, 5.98))
    ax.plot(time, original, color="black", label="Original", linewidth=2.0)
    ax.plot(time, reconstructed, label="Reconstructed", linewidth=2.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend()
    ax.set_xlim(0.0, 2.0)
    save_figure(fig, figures_dir / "fig8_ReconstructedForce_Reconstruction.pdf")

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    image = ax.imshow(
        percdiff,
        extent=[viv_params.azimuths[0], viv_params.azimuths[-1], viv_params.inflow_speeds[0], viv_params.inflow_speeds[-1]],
        aspect="auto",
        origin="lower",
        cmap="viridis_r",
        vmin=0.0,
        vmax=50.0,
    )
    fig.colorbar(image, ax=ax, label="Percent Difference in Frequencies")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel(r"Inflow (m s$^{-1}$)")
    save_figure(fig, figures_dir / "fig9_worst_percent_diff_single_blade_Reconstruction.pdf")

    az_index = int(np.flatnonzero(np.isclose(viv_params.azimuths, 216.0))[0])
    return {
        "modal_frequency_hz": 15.0,
        "pearson_r_demeaned_0_2s": correlation,
        "pearson_r_reported_2dp": f"{correlation:.2f}",
        "retained_spectral_entries_including_dc": retained_depth,
        "source_file_aoa_deg": 144.0,
        "structural_convention_aoa_deg": -144.0,
        "minimum_percent_difference": float(np.min(percdiff)),
        "minimum_percent_difference_at_216deg": float(np.min(percdiff[:, az_index])),
        "airfoil_id": "ffa_w3_211",
    }


def published_legacy_torque_grid(components, affts, viv_params) -> np.ndarray:
    """Reproduce Fig. 14's historical elementwise ``r * F`` plotting quantity.

    This is deliberately isolated from the corrected library calculation, which
    uses the physical cross product ``r x F`` and is emitted as a second figure.
    """
    from vorlap.interpolation import interpolate_fft_spectrum_optimized

    torque = np.zeros((len(viv_params.inflow_speeds), len(viv_params.azimuths)), dtype=float)
    inflow_unit = np.asarray(viv_params.inflow_vec, dtype=float)
    inflow_unit /= np.linalg.norm(inflow_unit)
    for i_speed, speed in enumerate(viv_params.inflow_speeds):
        inflow = inflow_unit * speed
        for i_azimuth, azimuth in enumerate(viv_params.azimuths):
            rotated_inflow = vorlap.rotate_vector(inflow, viv_params.rotation_axis, -azimuth)
            elementwise_moment = np.zeros(3, dtype=float)
            for component in components:
                for i_node in range(component.shape_xyz.shape[0]):
                    chord = float(component.chord[i_node])
                    airfoil = affts.get(component.airfoil_ids[i_node], affts["default"])
                    chord_vector = component.chord_vector[i_node]
                    normal_vector = component.normal_vector[i_node]
                    v_chord = float(np.dot(rotated_inflow, chord_vector / np.linalg.norm(chord_vector)))
                    v_normal = float(np.dot(rotated_inflow, normal_vector / np.linalg.norm(normal_vector)))
                    aoa_deg = math.degrees(math.atan2(v_normal, v_chord))
                    v_eff = math.hypot(v_normal, v_chord)
                    reynolds = viv_params.fluid_density * v_eff * chord / viv_params.fluid_dynamicviscosity
                    if i_node == 0:
                        local_length = 0.0
                    else:
                        local_length = float(
                            np.linalg.norm(component.shape_xyz[i_node] - component.shape_xyz[i_node - 1])
                        )
                    q = 0.5 * viv_params.fluid_density * v_eff**2 * chord * local_length
                    spectra = interpolate_fft_spectrum_optimized(
                        airfoil,
                        reynolds,
                        aoa_deg,
                        ["CL", "CD"],
                        n_freq_depth=viv_params.n_freq_depth,
                    )
                    lift = spectra["CL"][1][0] * q
                    drag = spectra["CD"][1][0] * q
                    local_yaw = math.degrees(math.atan2(chord_vector[1], chord_vector[0]))
                    normal_rotated = vorlap.rotate_vector(normal_vector, viv_params.rotation_axis, azimuth)
                    local_roll = math.degrees(math.atan2(normal_rotated[2], normal_rotated[1]))
                    force_rolled = vorlap.rotate_vector(
                        np.array([drag, lift, 0.0]), np.array([1.0, 0.0, 0.0]), local_roll
                    )
                    global_force = vorlap.rotate_vector(
                        force_rolled, np.array([0.0, 0.0, 1.0]), local_yaw
                    )
                    elementwise_moment += global_force * component.shape_xyz_global[i_node]
            torque[i_speed, i_azimuth] = elementwise_moment[2]
    return torque


def reference_turbine_figures(naca_h5: Path, figures_dir: Path) -> dict[str, object]:
    components = vorlap.load_components_from_csv(
        str(REPO / "data" / "components" / "componentsHVAWTReference")
    )
    inflow_speeds = np.arange(1.0, 17.0, 1.0)
    output_time = np.arange(0.0, 10.011, 0.001)
    viv_params = vorlap.VIV_Params(
        fluid_density=1.225,
        fluid_dynamicviscosity=1.81e-5,
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        rotation_axis_offset=np.array([0.0, 0.0, 0.0]),
        inflow_vec=np.array([1.0, 0.0, 0.0]),
        azimuths=np.arange(0.0, 125.0, 5.0),
        inflow_speeds=inflow_speeds,
        n_harmonic=1,
        output_time=output_time,
        output_azimuth_vinf=(60.0, 6.0),
        amplitude_coeff_cutoff=0.5,
        n_freq_depth=10,
        airfoil_folder=str(REPO / "data" / "airfoils") + os.sep,
    )
    initialize_geometry(components, viv_params)
    plot_geometry(components, figures_dir)
    naca = vorlap.load_airfoil_fft(str(naca_h5))
    ffa = vorlap.load_airfoil_fft(str(REPO / "data" / "airfoils" / "ffa_w3_211.h5"))
    affts = {"NACA0018": naca, "ffa_w3_211": ffa, "default": naca}
    natural_frequencies = np.loadtxt(
        REPO / "data" / "natural_frequencies_Reference_Turbine.csv", delimiter=","
    )
    percdiff, _, _, moments, node_forces = vorlap.compute_thrust_torque_spectrum_optimized(
        components, affts, viv_params, natural_frequencies
    )

    torque = moments[:, :, 2]
    legacy_torque = published_legacy_torque_grid(components, affts, viv_params)
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    if float(np.min(legacy_torque)) < 0.0 < float(np.max(legacy_torque)):
        norm = mcolors.TwoSlopeNorm(
            vmin=float(np.min(legacy_torque)),
            vcenter=0.0,
            vmax=float(np.max(legacy_torque)),
        )
    else:
        norm = None
    image = ax.imshow(
        legacy_torque,
        extent=[viv_params.azimuths[0], viv_params.azimuths[-1], inflow_speeds[0], inflow_speeds[-1]],
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        norm=norm,
    )
    fig.colorbar(image, ax=ax, label="Moment (N m)")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel(r"Inflow (m s$^{-1}$)")
    ax.set_title("Static Torque")
    save_figure(fig, figures_dir / "fig14_torque.pdf")

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    image = ax.imshow(
        torque,
        extent=[viv_params.azimuths[0], viv_params.azimuths[-1], inflow_speeds[0], inflow_speeds[-1]],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="Physical torque r × F (N m)")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel(r"Inflow (m s$^{-1}$)")
    ax.set_title("Corrected Static Torque")
    save_figure(fig, figures_dir / "fig14_torque_corrected.pdf")

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    image = ax.imshow(
        percdiff,
        extent=[viv_params.azimuths[0], viv_params.azimuths[-1], inflow_speeds[0], inflow_speeds[-1]],
        aspect="auto",
        origin="lower",
        cmap="viridis_r",
        vmin=0.0,
        vmax=50.0,
    )
    fig.colorbar(image, ax=ax, label="Percent Difference in Frequencies")
    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel(r"Inflow (m s$^{-1}$)")
    save_figure(fig, figures_dir / "fig15_worst_percent_diff.pdf")

    # The archived Fig. 16 script selected global node index 1 (the second
    # Arm1 node) and used the archival input in which every Strouhal entry,
    # including DC, had the illustrative +0.07 shift.  Preserve that exact
    # compatibility path under the publication filename.  The corrected
    # Blade3/node2 selection is emitted separately from the corrected input.
    archival_naca = vorlap.load_airfoil_fft(str(REPO / "data" / "airfoils" / "NACA0018.h5"))
    compatibility_params = vorlap.VIV_Params(
        fluid_density=viv_params.fluid_density,
        fluid_dynamicviscosity=viv_params.fluid_dynamicviscosity,
        rotation_axis=viv_params.rotation_axis,
        rotation_axis_offset=viv_params.rotation_axis_offset,
        inflow_vec=viv_params.inflow_vec,
        azimuths=np.array([60.0]),
        inflow_speeds=np.array([6.0]),
        n_harmonic=viv_params.n_harmonic,
        output_time=output_time,
        output_azimuth_vinf=(60.0, 6.0),
        amplitude_coeff_cutoff=viv_params.amplitude_coeff_cutoff,
        n_freq_depth=viv_params.n_freq_depth,
        airfoil_folder=viv_params.airfoil_folder,
    )
    _, _, _, _, compatibility_forces = vorlap.compute_thrust_torque_spectrum_optimized(
        components,
        {"NACA0018": archival_naca, "ffa_w3_211": ffa, "default": archival_naca},
        compatibility_params,
        natural_frequencies,
    )
    publication_node = 1
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ax.plot(output_time, compatibility_forces[:, 0, publication_node], label="x Force", linewidth=2.0)
    ax.plot(output_time, compatibility_forces[:, 1, publication_node], label="y Force", linewidth=2.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force per Span (N/m)")
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, figures_dir / "fig16_ReconstructedForce.pdf")

    blade3 = next(index for index, component in enumerate(components) if component.id == "Blade3")
    node_offset = sum(component.shape_xyz.shape[0] for component in components[:blade3])
    blade3_node2 = node_offset + 1
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.plot(output_time, node_forces[:, 0, blade3_node2], label="x Force", linewidth=2.0)
    ax.plot(output_time, node_forces[:, 1, blade3_node2], label="y Force", linewidth=2.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force per Span (N/m)")
    ax.legend()
    save_figure(fig, figures_dir / "fig16_ReconstructedForce_blade3_node2.pdf")

    assert np.array_equal(inflow_speeds, np.arange(1.0, 17.0, 1.0))
    return {
        "inflow_speeds_m_s": inflow_speeds.tolist(),
        "azimuths_deg": viv_params.azimuths.tolist(),
        "natural_frequencies_hz": np.asarray(natural_frequencies).reshape(-1).tolist(),
        "minimum_percent_difference": float(np.min(percdiff)),
        "torque_min_nm": float(np.min(torque)),
        "torque_max_nm": float(np.max(torque)),
        "published_legacy_elementwise_moment_min": float(np.min(legacy_torque)),
        "published_legacy_elementwise_moment_max": float(np.max(legacy_torque)),
        "published_figure16_component": components[0].id,
        "published_figure16_node": 2,
        "corrected_reconstructed_component": "Blade3",
        "corrected_reconstructed_node": 2,
    }


def validate_release_results(summary: dict[str, object]) -> dict[str, object]:
    """Fail the release run when a reviewer-sensitive result drifts."""
    verification = summary["verification"]
    reference = summary["reference_turbine"]
    corrected = summary["corrected_legacy_spectral_input"]
    processor = summary["naca_processor"]
    histories = summary["naca_histories"]

    checks = {
        "paper_pearson_r_is_0.90": verification["pearson_r_reported_2dp"] == "0.90",
        "verification_frequency_is_15_hz": verification["modal_frequency_hz"] == 15.0,
        "verification_finds_subpercent_overlap": verification["minimum_percent_difference"] < 1.0,
        "reference_speed_grid_is_1_through_16_m_s": reference["inflow_speeds_m_s"]
        == list(np.arange(1.0, 17.0, 1.0)),
        "reference_finds_subpercent_overlap": reference["minimum_percent_difference"] < 1.0,
        "naca_thickness_is_0.18": corrected["new_thickness_ratio"] == 0.18,
        "dc_strouhal_is_zero": corrected["new_dc_strouhal"] == 0.0,
        "naca_archive_hash_matches": processor["archive_sha256"] == NACA_ARCHIVE_SHA256,
        "paper_naca_history_hash_matches": histories["archive_sha256"]
        == PAPER_NACA_HISTORIES_SHA256,
        "legacy_naca_hash_matches": corrected["legacy_h5_sha256"] == LEGACY_NACA_H5_SHA256,
        "library_torque_uses_physical_sign": reference["torque_max_nm"] < 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Paper-release acceptance checks failed: {failed}")
    return {"status": "passed", "checks": checks}


def run(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_dir, processor = prepare_naca_source(output_dir)
    histories = naca_history_figures(figures_dir)
    corrected_h5, corrected = corrected_legacy_naca(output_dir)
    spectral_summary_figures(corrected_h5, figures_dir)
    verification = verification_figures(figures_dir)
    reference = reference_turbine_figures(corrected_h5, figures_dir)

    expected_figures = PAPER_FIGURES + CORRECTED_COMPANION_FIGURES
    missing = [name for name in expected_figures if not (figures_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Expected figure outputs were not created: {missing}")

    summary = {
        "schema_version": 1,
        "code_base_commit": "191d49b801c563ea088b886814e3369b26ed086b",
        "paper_commit": "1b206f3f36d3a5fbfa28ea241759e3671201cd35",
        "paper_figures_regenerated": list(PAPER_FIGURES),
        "external_archival_figures": [
            "fig10_mode1.png",
            "fig11_mode2.png",
            "fig12_mode3.png",
            "fig13_mode4.png",
        ],
        "naca_processor": processor,
        "naca_histories": histories,
        "corrected_legacy_spectral_input": corrected,
        "verification": verification,
        "reference_turbine": reference,
    }
    summary["acceptance"] = validate_release_results(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER_DIR / "build",
        help="Generated HDF5, figures, and summary (default: paper/2025-nawea/build).",
    )
    args = parser.parse_args(argv)
    run(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
