"""Generate a QBlade LOADINGFILE from VorLap time-varying inflow reconstruction."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import vorlap


def _load_airfoils(airfoil_folder: str):
    affts = {}
    for file in glob.glob(os.path.join(airfoil_folder, "*.h5")):
        afft = vorlap.load_airfoil_fft(file)
        affts[afft.name] = afft
    if "default" not in affts and affts:
        affts["default"] = next(iter(affts.values()))
    return affts


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, help="Path to a QBlade .sim file.")
    parser.add_argument(
        "--inflow",
        default=str(Path(vorlap.repo_dir) / "data" / "inflow_profile.csv"),
        help="CSV with time-varying inflow: time,inflow_speed,inflow_direction_deg",
    )
    parser.add_argument(
        "--airfoils",
        default=str(Path(vorlap.repo_dir) / "data" / "airfoils"),
        help="Folder containing VorLap airfoil FFT .h5 files.",
    )
    parser.add_argument(
        "--output",
        default="qblade_external_loading.txt",
        help="Output QBlade loading file path.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=None,
        help="Optional fixed azimuth for reconstruction (deg).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    components, viv_params, qblade_node_ids = vorlap.convert_qblade_to_vorlap_inputs(args.sim)
    inflow_profile = vorlap.load_inflow_time_series(args.inflow)
    affts = _load_airfoils(args.airfoils)
    if not affts:
        raise RuntimeError(f"No airfoil FFT .h5 files found in: {args.airfoils}")
    viv_params.airfoil_folder = os.path.join(args.airfoils, "")

    azimuth = args.azimuth if args.azimuth is not None else viv_params.output_azimuth_vinf[0]

    # Populate global geometry and local chord/normal vectors needed by reconstruction.
    vorlap.calc_structure_vectors_andplot(components, viv_params, show_plot=False)

    time, _total_force, _total_moment, node_forces = vorlap.compute_time_varying_force_history_optimized(
        components=components,
        affts=affts,
        viv_params=viv_params,
        inflow_profile=inflow_profile,
        azimuth_deg=azimuth,
        smoothing_cycles=0.25,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vorlap.write_qblade_loading_file(
        str(output_path),
        time,
        node_forces,
        qblade_node_ids,
        local=False,
    )
    print(f"Wrote QBlade loading file: {output_path}")
    print(f"Load targets: {len(qblade_node_ids)}")
    print("Set this file in your QBlade .sim TURB_1 LOADINGFILE field.")


if __name__ == "__main__":
    main()
