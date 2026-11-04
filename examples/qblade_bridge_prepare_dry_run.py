"""Dry-run QBlade external-library case preparation without launching QBlade."""

from __future__ import annotations

import argparse
from pathlib import Path

import vorlap
from vorlap.qblade_batch import (
    reference_case_dir_for_root,
    reference_sim_name_for_root,
    vorlap_airfoil_dir_for_sim,
)
from vorlap.qblade_case_prep import QBladeCasePrepRequest, prepare_qblade_external_case


def default_reference_sim() -> Path:
    root = Path(vorlap.repo_dir)
    return reference_case_dir_for_root(root) / reference_sim_name_for_root(root)


def default_airfoil_dir(sim_path: Path) -> Path:
    return vorlap_airfoil_dir_for_sim(sim_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim",
        default=str(default_reference_sim()),
        help="QBlade .sim file to inspect.",
    )
    parser.add_argument(
        "--airfoils",
        default=None,
        help="VorLap airfoil FFT directory. Defaults to model-local VorLapAirfoils.",
    )
    parser.add_argument(
        "--node-source",
        choices=("structural", "converted"),
        default="structural",
        help="Use QBlade structural node ids or converted VorLap geometry node ids.",
    )
    parser.add_argument(
        "--source-bounds-policy",
        choices=("warn", "raise", "clip"),
        default="raise",
        help="Runtime Re/AOA source-grid policy to write into the generated config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_path = Path(args.sim).expanduser().resolve()
    airfoil_dir = Path(args.airfoils).expanduser().resolve() if args.airfoils else default_airfoil_dir(sim_path)

    result = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim_path,
            airfoil_dir=airfoil_dir,
            node_source=args.node_source,
            source_bounds_policy=args.source_bounds_policy,
            include_default_python_path=False,
            dry_run=True,
        )
    )

    print(f"Dry-run QBlade bridge preparation: {result.sim_path}")
    print(f"Node source: {result.node_source}")
    print(f"Nodes: {len(result.node_ids)}")
    print(f"Swap size: {result.swap_size}")
    print(f"Source policy: {result.runtime_config['source_bounds_policy']}")
    print("Planned writes:")
    for path in result.planned_write_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
