#!/usr/bin/env python3
"""Screen VorLap AmpCF/STCF forcing against the qualified structural modes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.modal_overlap import assess_peak_coverage, rank_local_peaks, screen_modal_overlap
from vorlap.airfoil_io import load_airfoil_fft
from vorlap.qblade_component_selection import build_selected_components
from vorlap.qblade_conversion import convert_qblade_to_vorlap_inputs
from vorlap.qblade_nodes import extract_structural_node_ids


DEFAULT_SIM = REPO_ROOT / "examples/ReferenceHVAWTQBladeModel/baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
DEFAULT_STRUCTURE = REPO_ROOT / (
    "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/Structure/"
    "10m_HVAWT_reference_main.str"
)
DEFAULT_AIRFOIL_DIR = REPO_ROOT / (
    "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/VorLapAirfoils"
)
DEFAULT_CAMPBELL = REPO_ROOT / "run_headless/config/changed_geometry_campbell_frequencies.csv"
DEFAULT_MATRIX = REPO_ROOT / "run_headless/config/parked_dlc_matrix_draft.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_campbell_modes(path: Path, rpm: float) -> tuple[list[float], float, list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "rpm" not in rows[0]:
        raise ValueError(f"Campbell CSV has no data or rpm column: {path}")
    selected = min(rows, key=lambda row: abs(float(row["rpm"]) - float(rpm)))
    selected_rpm = float(selected["rpm"])
    if not math.isclose(selected_rpm, float(rpm), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"Campbell CSV has no row at {rpm:g} RPM; nearest is {selected_rpm:g} RPM.")
    columns = [name for name in selected if name != "rpm" and name.endswith("_hz")]
    modes = [float(selected[name]) for name in columns]
    if not modes or any(not math.isfinite(value) or value <= 0.0 for value in modes):
        raise ValueError("Selected Campbell row has no finite positive mode frequencies.")
    return modes, selected_rpm, columns


def matrix_dlc64_speeds(path: Path) -> list[float]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    speeds: set[float] = set()
    for group in spec.get("groups", []):
        if str(group.get("dlc")) == "6.4":
            speeds.update(float(value) for value in group.get("wind_speeds_mps", []))
    if not speeds:
        raise ValueError(f"No DLC 6.4 speeds found in matrix spec: {path}")
    return sorted(speeds)


def rayleigh_pdf(speed_mps: float, average_speed_mps: float) -> float:
    return (
        math.pi
        * speed_mps
        / (2.0 * average_speed_mps**2)
        * math.exp(-math.pi * speed_mps**2 / (4.0 * average_speed_mps**2))
    )


def normalized_id(value: str) -> str:
    return "".join(character for character in str(value).upper() if character.isalnum())


def load_used_airfoils(
    airfoil_dir: Path, components: list[Any]
) -> tuple[dict[str, Any], list[Path]]:
    required_ids = sorted(
        {str(airfoil_id) for component in components for airfoil_id in component.airfoil_ids}
    )
    required_normalized = {normalized_id(value) for value in required_ids}
    candidate_paths = [
        path
        for path in sorted(airfoil_dir.glob("*.h5"))
        if normalized_id(path.stem) in required_normalized
    ]
    if not candidate_paths:
        raise FileNotFoundError(
            f"No HDF5 source in {airfoil_dir} matches required airfoil ids {required_ids}."
        )
    airfoils: dict[str, Any] = {}
    for path in candidate_paths:
        airfoil = load_airfoil_fft(str(path))
        airfoils[airfoil.name] = airfoil
        airfoils[path.stem] = airfoil
    normalized_loaded = {normalized_id(key): value for key, value in airfoils.items()}
    for required in required_ids:
        normalized = normalized_id(required)
        if normalized not in normalized_loaded:
            raise KeyError(f"No loaded HDF5 source matches required airfoil id '{required}'.")
        airfoils[required] = normalized_loaded[normalized]
    airfoils["default"] = airfoils[required_ids[0]]
    return airfoils, candidate_paths


def unique_peak_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_speed: dict[float, dict[str, Any]] = {}
    for group in groups:
        for row in group:
            by_speed.setdefault(float(row["wind_speed_mps"]), row)
    return [by_speed[speed] for speed in sorted(by_speed)]


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "wind_speed_mps",
            "overlap_score_n",
            "downtime_probability_density_per_mps",
            "downtime_weighted_overlap_score_n_per_mps",
            "minimum_detuning_percent",
            "within_overlap_band_candidate_count",
            "source_clip_query_fraction",
        )
    }


def peak_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **compact_row(row),
        "best_force_overlap": row.get("best_force_overlap"),
        "best_frequency_match": row.get("best_frequency_match"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    compact = [compact_row(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compact[0]))
        writer.writeheader()
        writer.writerows(compact)


def write_plot(path: Path, rows: list[dict[str, Any]], matrix_speeds: list[float], peaks: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    wind = np.asarray([row["wind_speed_mps"] for row in rows], dtype=float)
    load_score = np.asarray([row["overlap_score_n"] for row in rows], dtype=float)
    weighted = np.asarray([row["downtime_weighted_overlap_score_n_per_mps"] for row in rows], dtype=float)
    density = np.asarray([row["downtime_probability_density_per_mps"] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True, constrained_layout=True)
    axes[0].plot(wind, load_score, color="tab:blue", linewidth=1.5, label="Force-overlap score")
    axes[0].set_ylabel("Overlap score [N]")
    axes[0].grid(True, alpha=0.25)

    weighted_norm = weighted / np.max(weighted) if np.max(weighted) > 0.0 else weighted
    density_norm = density / np.max(density) if np.max(density) > 0.0 else density
    axes[1].plot(wind, weighted_norm, color="tab:red", linewidth=1.5, label="Downtime-weighted overlap")
    axes[1].plot(wind, density_norm, color="black", linewidth=1.0, linestyle="--", label="Downtime density")
    axes[1].set_ylabel("Normalized screening metric [-]")
    axes[1].set_xlabel("10-minute mean wind speed [m/s]")
    axes[1].grid(True, alpha=0.25)

    for axis in axes:
        for index, speed in enumerate(matrix_speeds):
            axis.axvline(speed, color="0.65", linewidth=0.8, linestyle=":", label="DLC matrix speed" if index == 0 else None)
        for index, row in enumerate(peaks):
            axis.axvline(
                float(row["wind_speed_mps"]),
                color="tab:orange",
                linewidth=0.9,
                alpha=0.75,
                label="Ranked overlap peak" if index == 0 else None,
            )
        axis.legend(loc="best", fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_review(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sim_path = args.sim.resolve()
    structural_path = args.structural_file.resolve()
    airfoil_dir = args.airfoil_dir.resolve()
    campbell_path = args.campbell_csv.resolve()
    matrix_path = args.matrix_spec.resolve()

    modes, selected_rpm, mode_columns = load_campbell_modes(campbell_path, args.rpm)
    source_components, viv_params, _all_node_ids = convert_qblade_to_vorlap_inputs(
        str(sim_path), include_struts=True, include_tower=False
    )
    requested_node_ids = extract_structural_node_ids(
        structural_path.read_text(encoding="utf-8"), include_tower_nodes=False
    )
    components, controller_node_ids, requested_canonical = build_selected_components(
        source_components, requested_node_ids
    )
    if set(controller_node_ids) != set(requested_canonical):
        raise ValueError("Selected VorLap nodes do not match the structural action-node list.")

    airfoils, used_airfoil_paths = load_used_airfoils(airfoil_dir, components)
    viv_params.inflow_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    wind_speeds = np.arange(
        float(args.wind_min),
        float(args.wind_max) + 0.5 * float(args.wind_step),
        float(args.wind_step),
        dtype=float,
    )
    azimuths = np.arange(0.0, 360.0, float(args.azimuth_step), dtype=float)
    rows = screen_modal_overlap(
        components,
        airfoils,
        modes,
        wind_speeds,
        azimuths,
        node_ids=controller_node_ids,
        rpm=args.rpm,
        inflow_vector=(1.0, 0.0, 0.0),
        rotation_axis=viv_params.rotation_axis,
        rotation_axis_offset=viv_params.rotation_axis_offset,
        fluid_density_kg_m3=args.air_density,
        fluid_kinematic_viscosity_m2_s=args.kinematic_viscosity,
        n_freq_depth=args.n_freq_depth,
        amplitude_coefficient_cutoff=args.amplitude_cutoff,
        source_frequency_cap_hz=args.source_frequency_cap,
        overlap_band_fraction=args.overlap_band_fraction,
    )
    downtime_fraction = 1.0 - float(args.availability)
    for row in rows:
        density = downtime_fraction * rayleigh_pdf(float(row["wind_speed_mps"]), float(args.vave))
        row["downtime_probability_density_per_mps"] = density
        row["downtime_weighted_overlap_score_n_per_mps"] = density * float(row["overlap_score_n"])

    load_peaks = rank_local_peaks(
        rows,
        "overlap_score_n",
        max_peaks=args.max_peaks_per_metric,
        minimum_relative_score=args.minimum_relative_peak_score,
        minimum_spacing_mps=args.minimum_peak_spacing,
    )
    downtime_peaks = rank_local_peaks(
        rows,
        "downtime_weighted_overlap_score_n_per_mps",
        max_peaks=args.max_peaks_per_metric,
        minimum_relative_score=args.minimum_relative_peak_score,
        minimum_spacing_mps=args.minimum_peak_spacing,
    )
    required_peaks = unique_peak_rows(load_peaks, downtime_peaks)
    matrix_speeds = matrix_dlc64_speeds(matrix_path)
    coverage = assess_peak_coverage(
        required_peaks,
        matrix_speeds,
        overlap_band_fraction=args.overlap_band_fraction,
        minimum_tolerance_mps=args.minimum_coverage_tolerance,
    )
    recommended = sorted(set(matrix_speeds + coverage["uncovered_peak_wind_speeds_mps"]))
    recommended_coverage = assess_peak_coverage(
        required_peaks,
        recommended,
        overlap_band_fraction=args.overlap_band_fraction,
        minimum_tolerance_mps=args.minimum_coverage_tolerance,
    )
    rows_by_speed = {float(row["wind_speed_mps"]): row for row in rows}
    missing_matrix_speeds = [speed for speed in matrix_speeds if speed not in rows_by_speed]
    if missing_matrix_speeds:
        raise ValueError(
            "Matrix speed(s) are not represented on the screening grid: "
            + ", ".join(f"{speed:g}" for speed in missing_matrix_speeds)
        )
    matrix_source_checks = [
        {
            "wind_speed_mps": speed,
            "source_clip_query_fraction": float(rows_by_speed[speed]["source_clip_query_fraction"]),
            "passed": bool(rows_by_speed[speed]["source_clip_query_fraction"] == 0.0),
        }
        for speed in matrix_speeds
    ]
    required_peak_source_checks = [
        {
            "wind_speed_mps": float(row["wind_speed_mps"]),
            "source_clip_query_fraction": float(row["source_clip_query_fraction"]),
            "passed": bool(row["source_clip_query_fraction"] == 0.0),
        }
        for row in required_peaks
    ]
    source_domain_checks = {
        "passed": all(check["passed"] for check in matrix_source_checks + required_peak_source_checks),
        "current_matrix_speeds": matrix_source_checks,
        "required_peak_speeds": required_peak_source_checks,
        "note": (
            "The full 1--35 m/s sweep may clip low-Re source queries below the selected matrix; "
            "passing requires zero clipping at every selected matrix speed and ranked peak."
        ),
    }

    report: dict[str, Any] = {
        "schema_version": 1,
        "passed": bool(coverage["passed"] and source_domain_checks["passed"]),
        "purpose": "Risk-informed preselection of DLC 6.4 mean wind speeds from VorLap/modal overlap.",
        "method": (
            "For each mean wind speed, 1 RPM azimuth, and production VorLap action node, "
            "interpolate alternating AmpCF/STCF tones, dimensionalize STCF with the local "
            "relative flow and VorLap reference length, and compare each retained tone with "
            "the qualified Campbell modes. The force-overlap score is nodal harmonic-force "
            "amplitude times a Gaussian detuning kernel. The downtime-weighted score also "
            "multiplies by the project 1-percent Rayleigh downtime density."
        ),
        "limitations": (
            "This is a conservative frequency/amplitude screen, not a modal-response or lock-in "
            "prediction. It does not include generalized modal force, mode-shape participation, "
            "structural damping, turbulence broadening, or aeroelastic feedback. Final evidence "
            "comes from the time-domain QBlade baseline/VorLap runs."
        ),
        "screening_parameters": {
            "rpm": float(args.rpm),
            "wind_min_mps": float(args.wind_min),
            "wind_max_mps": float(args.wind_max),
            "wind_step_mps": float(args.wind_step),
            "azimuth_step_deg": float(args.azimuth_step),
            "n_freq_depth": int(args.n_freq_depth),
            "amplitude_coefficient_cutoff": float(args.amplitude_cutoff),
            "source_frequency_cap_hz": float(args.source_frequency_cap),
            "overlap_band_fraction": float(args.overlap_band_fraction),
            "minimum_coverage_tolerance_mps": float(args.minimum_coverage_tolerance),
            "minimum_relative_peak_score": float(args.minimum_relative_peak_score),
            "minimum_peak_spacing_mps": float(args.minimum_peak_spacing),
            "air_density_kg_m3": float(args.air_density),
            "air_kinematic_viscosity_m2_s": float(args.kinematic_viscosity),
            "availability": float(args.availability),
            "annual_downtime_fraction": downtime_fraction,
            "vave_mps": float(args.vave),
            "node_count": len(controller_node_ids),
            "azimuth_count": len(azimuths),
            "wind_speed_count": len(wind_speeds),
        },
        "mode_basis": {
            "campbell_csv": display_path(campbell_path),
            "campbell_csv_sha256": sha256_file(campbell_path),
            "selected_rpm": selected_rpm,
            "mode_columns": mode_columns,
            "natural_frequencies_hz": modes,
        },
        "source_basis": {
            "qblade_sim": display_path(sim_path),
            "qblade_sim_sha256": sha256_file(sim_path),
            "qblade_structural_file": display_path(structural_path),
            "qblade_structural_file_sha256": sha256_file(structural_path),
            "airfoil_files": [
                {"path": display_path(path), "sha256": sha256_file(path)}
                for path in used_airfoil_paths
            ],
            "node_ids": list(controller_node_ids),
            "inflow_vector": [1.0, 0.0, 0.0],
        },
        "matrix_spec": {
            "path": display_path(matrix_path),
            "current_dlc64_wind_speeds_mps": matrix_speeds,
        },
        "ranked_force_overlap_peaks": [peak_record(row) for row in load_peaks],
        "ranked_downtime_weighted_overlap_peaks": [peak_record(row) for row in downtime_peaks],
        "required_peak_wind_speeds_mps": [float(row["wind_speed_mps"]) for row in required_peaks],
        "current_matrix_coverage": coverage,
        "source_domain_checks": source_domain_checks,
        "recommended_dlc64_wind_speeds_mps": recommended,
        "recommended_matrix_coverage": recommended_coverage,
        "sweep": [compact_row(row) for row in rows],
    }
    report["review_sha256"] = sha256_payload(report)
    return report, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--structural-file", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--airfoil-dir", type=Path, default=DEFAULT_AIRFOIL_DIR)
    parser.add_argument("--campbell-csv", type=Path, default=DEFAULT_CAMPBELL)
    parser.add_argument("--matrix-spec", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--rpm", type=float, default=1.0)
    parser.add_argument("--wind-min", type=float, default=1.0)
    parser.add_argument("--wind-max", type=float, default=35.0)
    parser.add_argument("--wind-step", type=float, default=0.25)
    parser.add_argument("--azimuth-step", type=float, default=5.0)
    parser.add_argument("--n-freq-depth", type=int, default=20)
    parser.add_argument("--amplitude-cutoff", type=float, default=0.002)
    parser.add_argument("--source-frequency-cap", type=float, default=0.5 / 0.0075)
    parser.add_argument("--overlap-band-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-coverage-tolerance", type=float, default=0.5)
    parser.add_argument("--minimum-relative-peak-score", type=float, default=0.2)
    parser.add_argument("--minimum-peak-spacing", type=float, default=0.75)
    parser.add_argument("--max-peaks-per-metric", type=int, default=6)
    parser.add_argument("--air-density", type=float, default=1.0296)
    parser.add_argument("--kinematic-viscosity", type=float, default=1.647e-5)
    parser.add_argument("--availability", type=float, default=0.99)
    parser.add_argument("--vave", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.wind_step <= 0.0 or args.wind_max < args.wind_min:
            raise ValueError("Wind bounds require wind_step > 0 and wind_max >= wind_min.")
        if args.azimuth_step <= 0.0 or args.azimuth_step > 360.0:
            raise ValueError("azimuth_step must be in (0, 360].")
        if not 0.0 <= args.availability <= 1.0:
            raise ValueError("availability must be between zero and one.")
        report, rows = build_review(args)
        if args.out is not None:
            write_json(args.out, report)
        if args.csv is not None:
            write_csv(args.csv, rows)
        if args.plot is not None:
            required = [
                row for row in rows if float(row["wind_speed_mps"]) in report["required_peak_wind_speeds_mps"]
            ]
            write_plot(
                args.plot,
                rows,
                report["matrix_spec"]["current_dlc64_wind_speeds_mps"],
                required,
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
