#!/usr/bin/env python3
"""Prepare a runnable copy of the bundled VorLap/QBlade reference sample."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.qblade_batch import copy_reference_case, vorlap_airfoil_dir_for_sim
from vorlap.qblade_files import replace_qblade_keyword_file


_KEYWORD_COMMENTS = {
    "MEANINF": "the mean inflow velocity, overridden if a windfield or hubheight file is use",
    "NUMTIMESTEPS": "the number of timesteps",
    "TIMESTEP": "the timestep size in [s]",
    "PRESCRIBETYPE": "rotor RPM prescribe type (0 = ramp-up; 1 = whole sim; 3 = no RPM prescribed)",
    "RPMPRESCRIBED": "the prescribed rotor RPM [-]",
    "STOREFROM": "the simulation stores data from this point in time, in [s]",
    "STOREREPLAY": "store a replay of the simulation (warning, large memory will be required) [bool]",
    "WNDTYPE": "use a number: 0 = steady; 1 = windfield; 2 = hubheight",
    "WNDNAME": "filename of the turbsim input file, mann input file or hubheight file (with extension), leave blank if unused",
    "RAMPUP": "the rampup time for the structural model",
    "ADDDAMP": "the initial time with additional damping",
    "STRITERATIONS": "number of iterations for the time integration (used when integrator is HHT or Euler)",
}


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {text!r}.")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    run_headless_root = Path(__file__).resolve().parent
    default_vorlap = repo_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vorlap-root",
        default=os.environ.get("VORLAP_ROOT", str(default_vorlap)),
        help="VorLap repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(run_headless_root / "runs" / "vorlap_sample" / "case"),
        help="Directory where the prepared QBlade case will be written.",
    )
    parser.add_argument("--mean-inflow", type=float, default=6.0, help="QBlade MEANINF value.")
    parser.add_argument("--rpm", type=float, default=0.0, help="QBlade RPMPRESCRIBED value.")
    parser.add_argument("--timesteps", type=int, default=2000, help="QBlade NUMTIMESTEPS value.")
    parser.add_argument("--timestep", type=float, default=0.0075, help="QBlade TIMESTEP value.")
    parser.add_argument("--store-from", type=float, default=0.0, help="QBlade STOREFROM value.")
    parser.add_argument("--store-replay", type=parse_bool, default=False, help="QBlade STOREREPLAY value.")
    parser.add_argument("--n-freq-depth", type=int, default=20, help="VorLap spectral depth.")
    parser.add_argument("--force-scale", type=float, default=1.0, help="VorLap force multiplier.")
    parser.add_argument(
        "--diagnostic-force-scale",
        action="store_true",
        help="Allow --force-scale values above 1.0 for explicit diagnostic/amplitude-sensitivity runs.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for VorLap prep.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def replace_qblade_keyword(path: Path, keyword: str, value: float | int | str) -> None:
    comment = _KEYWORD_COMMENTS.get(keyword, "")
    replace_qblade_keyword_file(path, keyword, value, comment=comment)


def validate_force_scale(force_scale: float, diagnostic_force_scale: bool = False) -> None:
    """Require explicit labeling for force scales above the nominal physical value."""
    if not math.isfinite(force_scale) or force_scale < 0.0:
        raise ValueError("--force-scale must be finite and non-negative.")
    if force_scale > 1.0 and not diagnostic_force_scale:
        raise ValueError(
            "--force-scale values greater than 1.0 require --diagnostic-force-scale "
            "so exaggerated forcing cannot be mistaken for a nominal physical run."
        )


def run_vorlap_case_prep(
    python_exe: str,
    vorlap_root: Path,
    sim_path: Path,
    n_freq_depth: int,
    force_scale: float,
    diagnostic_force_scale: bool,
) -> None:
    script = vorlap_root / "scripts" / "prepare_qblade_external_case.py"
    airfoils = vorlap_airfoil_dir_for_sim(sim_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{vorlap_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            python_exe,
            str(script),
            "--sim",
            str(sim_path),
            "--airfoils",
            str(airfoils),
            "--node-source",
            "structural",
            "--n-freq-depth",
            str(n_freq_depth),
            "--force-scale",
            str(force_scale),
            "--debug",
        ] + (["--diagnostic-force-scale"] if diagnostic_force_scale else []),
        check=True,
        env=env,
    )


def main() -> None:
    args = parse_args()
    vorlap_root = Path(args.vorlap_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    validate_force_scale(args.force_scale, args.diagnostic_force_scale)

    sim_path = copy_reference_case(vorlap_root, output_dir, args.force)
    replace_qblade_keyword(sim_path, "MEANINF", args.mean_inflow)
    replace_qblade_keyword(sim_path, "PRESCRIBETYPE", 1)
    replace_qblade_keyword(sim_path, "RPMPRESCRIBED", args.rpm)
    replace_qblade_keyword(sim_path, "NUMTIMESTEPS", args.timesteps)
    replace_qblade_keyword(sim_path, "TIMESTEP", args.timestep)
    replace_qblade_keyword(sim_path, "STOREFROM", args.store_from)
    replace_qblade_keyword(sim_path, "STOREREPLAY", str(args.store_replay).lower())

    run_vorlap_case_prep(
        args.python,
        vorlap_root,
        sim_path,
        args.n_freq_depth,
        args.force_scale,
        args.diagnostic_force_scale,
    )
    print(f"Prepared VorLap/QBlade case: {sim_path}")


if __name__ == "__main__":
    main()
