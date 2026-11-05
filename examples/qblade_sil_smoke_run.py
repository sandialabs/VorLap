"""Run a short Python SIL QBlade/VorLap smoke case with an installed QBlade root."""

from __future__ import annotations

import argparse
from pathlib import Path

import vorlap
from vorlap.qblade_batch import (
    reference_case_dir_for_root,
    reference_sim_name_for_root,
    vorlap_airfoil_dir_for_sim,
)
from vorlap.qblade_bridge_config import QBladeBridgeConfig
from vorlap.qblade_bridge_service import QBladeBridgeService


def default_reference_sim() -> Path:
    root = Path(vorlap.repo_dir)
    return reference_case_dir_for_root(root) / reference_sim_name_for_root(root)


def default_airfoil_dir(sim_path: Path) -> Path:
    return vorlap_airfoil_dir_for_sim(sim_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qblade-root", required=True, help="QBlade installation root containing SIL_Interface.")
    parser.add_argument("--sim", default=str(default_reference_sim()), help="QBlade .sim file to run.")
    parser.add_argument("--airfoils", default=None, help="VorLap airfoil FFT directory.")
    parser.add_argument("--output-dir", default="run_headless/runs/example_sil_smoke", help="Run output directory.")
    parser.add_argument("--steps", type=int, default=20, help="Short smoke-run timestep count.")
    parser.add_argument("--timestep", type=float, default=None, help="Optional QBlade timestep override for the working copy.")
    parser.add_argument("--wind-speed", type=float, default=None, help="Optional QBlade MEANINF override for the working copy.")
    parser.add_argument(
        "--source-bounds-policy",
        choices=("warn", "raise", "clip"),
        default="raise",
        help="Runtime Re/AOA source-grid policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_path = Path(args.sim).expanduser().resolve()
    airfoil_dir = Path(args.airfoils).expanduser().resolve() if args.airfoils else default_airfoil_dir(sim_path)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(vorlap.repo_dir) / output_dir

    config = QBladeBridgeConfig(
        sim_path=sim_path,
        qblade_root=Path(args.qblade_root).expanduser().resolve(),
        output_dir=output_dir,
        airfoil_dir=airfoil_dir,
        coupling_mode="sil-vorlap",
        steps=args.steps,
        timestep_s=args.timestep,
        wind_speed=args.wind_speed,
        source_bounds_policy=args.source_bounds_policy,
        debug=True,
        record_sil_vorlap_history=True,
        progress_every=max(1, min(5, args.steps)),
        manage_loader_env=True,
    ).validated()

    service = QBladeBridgeService()
    prep = service.prepare_case(config, check_paths=True)
    print(f"Prepared working sim: {prep.artifacts.get('working_sim')}")
    print(f"Working-copy manifest: {prep.artifacts.get('working_copy_manifest')}")

    result = service.run_case(prep.config, check_paths=False, message_callback=print)
    print(f"Run state: {result.state}")
    print(f"Summary: {result.artifacts.get('summary')}")
    print(f"Result file: {result.artifacts.get('result_file')}")
    print(f"Source coverage: {result.artifacts.get('sil_vorlap_source_coverage')}")


if __name__ == "__main__":
    main()
