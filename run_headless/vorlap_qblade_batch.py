#!/usr/bin/env python3
"""Prepare and run VorLap parked-load QBlade batches on a headless cluster."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.qblade_batch import (
    QBladeBatchRunOptions,
    apply_stable_support_patch,
    azimuth_label,
    build_bridge,
    build_case_manifest,
    completed_filtered_export_ready,
    comparable_case_manifest,
    copy_reference_case,
    find_bridge_library,
    find_qblade_library,
    install_vorlap,
    load_manifest,
    manifest_matches_setup_inputs,
    parse_azimuths,
    parse_float_spec,
    parse_wind_speeds,
    prepend_env_path,
    prepare_cases,
    qblade_library_dirs,
    relative_case_path,
    replace_qblade_keyword,
    replace_structural_keyword,
    run_checked,
    setup,
    setup_lock,
    speed_label,
    validate_force_scale,
    validate_qblade_root,
    WAKE_PROFILE_CHOICES,
    WAKE_PROFILE_REFERENCE,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_headless_root() -> Path:
    return Path(__file__).resolve().parent


def default_python(vorlap_root: Path) -> str:
    venv_python = vorlap_root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def default_qblade_root(root: Path) -> Path:
    if "QBLADE_ROOT" in os.environ:
        return Path(os.environ["QBLADE_ROOT"])
    for candidate in (run_headless_root() / "QBladeCE_2.0.9.7", root / "QBladeCE_2.0.9.7"):
        if candidate.is_dir():
            return candidate
    return run_headless_root() / "QBladeCE_2.0.9.7"


def parse_bool(text: str | bool) -> bool:
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {text!r}.")


def run_case(args: argparse.Namespace, manifest: dict, index: int) -> None:
    cases = manifest["cases"]
    if index < 0 or index >= len(cases):
        raise IndexError(f"Case index {index} is outside 0..{len(cases) - 1}.")
    case = cases[index]
    result_dir = Path(case["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    case_metadata_path = result_dir / "case_metadata.json"
    case_metadata_path.write_text(
        json.dumps(
            {
                "case": case,
                "manifest_settings": manifest.get("settings", {}),
                "manifest_created_unix": manifest.get("created_unix"),
                "batch_command": [sys.executable, *sys.argv],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runner = Path(__file__).with_name("run_qblade_sim.py")
    run_options = QBladeBatchRunOptions.from_args(args)
    command = run_options.command_for_case(
        case,
        runner_path=runner,
        case_metadata_path=case_metadata_path,
    )
    run_checked(command, env=run_options.env())


def list_cases(manifest: dict) -> None:
    for case in manifest["cases"]:
        print(
            f"{case['index']:3d}  {case['label']:>20s}  "
            f"U={case['wind_speed']:g} m/s  psi={case.get('azimuth_deg', math.nan):g} deg"
        )


def parse_args() -> argparse.Namespace:
    root = repo_root()
    initial_vorlap = Path(os.environ.get("VORLAP_ROOT", root)).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("setup", "run", "run-all", "run-existing", "recover-export", "list"),
    )
    parser.add_argument("--run-root", type=Path, default=run_headless_root() / "runs" / "vorlap_parked")
    parser.add_argument("--qblade-root", type=Path, default=default_qblade_root(root))
    parser.add_argument("--vorlap-root", type=Path, default=initial_vorlap)
    parser.add_argument("--python", default=os.environ.get("PYTHON_EXE", default_python(initial_vorlap)))
    parser.add_argument("--wind-speeds", default=os.environ.get("WIND_SPEEDS", "linspace:2:20:13"))
    parser.add_argument(
        "--azimuths",
        default=os.environ.get("QBLADE_AZIMUTHS", "270"),
        help="Initial azimuths in deg. Accepts lists, linspace:start:stop:count, or range:start:stop:step.",
    )
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "16")))
    parser.add_argument(
        "--use-opencl",
        action="store_true",
        default=parse_bool(os.environ.get("QBLADE_USE_OPENCL", "false")),
        help="Ask QBlade to use OpenCL acceleration in run commands.",
    )
    parser.add_argument("--opencl-device", type=int, default=int(os.environ.get("QBLADE_OPENCL_DEVICE", "1")))
    parser.add_argument("--opencl-group-size", type=int, default=int(os.environ.get("QBLADE_OPENCL_GROUP_SIZE", "32")))
    parser.add_argument("--qblade-steps", type=int, default=None, help="Optional short-run override.")
    parser.add_argument("--timesteps", type=int, default=int(os.environ.get("QBLADE_TIMESTEPS", "2000")))
    parser.add_argument("--timestep", type=float, default=float(os.environ.get("QBLADE_TIMESTEP", "0.0075")))
    parser.add_argument("--store-from", type=float, default=float(os.environ.get("QBLADE_STORE_FROM", "0.0")))
    parser.add_argument(
        "--store-replay",
        type=parse_bool,
        default=parse_bool(os.environ.get("QBLADE_STORE_REPLAY", "false")),
        help="Set QBlade STOREREPLAY in generated .sim files.",
    )
    parser.add_argument(
        "--result-filter",
        type=Path,
        default=None
        if not os.environ.get("QBLADE_RESULT_FILTER", "").strip()
        else Path(os.environ["QBLADE_RESULT_FILTER"]),
        help=(
            "Compact QBlade export filter. SIL-VorLap setup also generates a storage filter "
            "containing the bridge input channels for .sim FILTERFILE, while run commands "
            "pass this compact filter to exportResults()."
        ),
    )
    parser.add_argument(
        "--full-structure-strain-output",
        action="store_true",
        default=parse_bool(os.environ.get("QBLADE_FULL_STRUCTURE_STRAIN_OUTPUT", "false")),
        help=(
            "Generate dense Fz/Mx/My output stations on every blade, strut, upper/lower "
            "tower, and torque-tube family for chordwise surface normal-strain DEL recovery."
        ),
    )
    parser.add_argument("--rpm", type=float, default=float(os.environ.get("QBLADE_RPM", "0.0")))
    parser.add_argument("--rampup", type=float, default=float(os.environ.get("QBLADE_RAMPUP", "0.0")))
    parser.add_argument("--add-damp", type=float, default=float(os.environ.get("QBLADE_ADD_DAMP", "0.0")))
    parser.add_argument(
        "--str-iterations",
        type=int,
        default=None if "QBLADE_STR_ITERATIONS" not in os.environ else int(os.environ["QBLADE_STR_ITERATIONS"]),
    )
    parser.add_argument("--wind-type", type=int, default=int(os.environ.get("QBLADE_WIND_TYPE", "0")))
    parser.add_argument(
        "--wind-name",
        default=os.environ.get("QBLADE_WIND_NAME", ""),
        help=(
            "Optional QBlade wind file name. May include {speed}, {speed_label}, "
            "{azimuth}, {azimuth_label}, and {label} placeholders."
        ),
    )
    parser.add_argument(
        "--horizontal-inflow-angle",
        type=float,
        default=float(os.environ.get("QBLADE_HORIZONTAL_INFLOW_ANGLE", "0.0")),
    )
    parser.add_argument(
        "--vertical-inflow-angle",
        type=float,
        default=float(os.environ.get("QBLADE_VERTICAL_INFLOW_ANGLE", "0.0")),
    )
    parser.add_argument(
        "--shear-exponent",
        type=float,
        default=float(os.environ.get("QBLADE_SHEAR_EXPONENT", "0.23")),
    )
    parser.add_argument(
        "--reference-height",
        type=float,
        default=float(os.environ.get("QBLADE_REFERENCE_HEIGHT", "18.0")),
    )
    parser.add_argument(
        "--air-density",
        type=float,
        default=float(os.environ.get("QBLADE_AIR_DENSITY", "1.0296")),
    )
    parser.add_argument(
        "--air-kinematic-viscosity",
        type=float,
        default=float(os.environ.get("QBLADE_AIR_KINEMATIC_VISCOSITY", "1.647e-5")),
    )
    parser.add_argument(
        "--wake-profile",
        choices=WAKE_PROFILE_CHOICES,
        default=os.environ.get("QBLADE_WAKE_PROFILE", WAKE_PROFILE_REFERENCE),
        help=(
            "Generated-case LLFVW profile. 'reference' preserves the tracked .trb; "
            "'parked-reduced-v1' uses timestep-counted wake zones with bounded far-wake retention; "
            "'parked-reduced-v2' additionally selects the accepted 10D cutoff, Oye TF=8, and "
            "UNH-derived vortex-core parameters."
        ),
    )
    parser.add_argument("--n-freq-depth", type=int, default=int(os.environ.get("VORLAP_N_FREQ_DEPTH", "20")))
    parser.add_argument(
        "--source-bounds-policy",
        choices=("warn", "raise", "clip"),
        default=os.environ.get("VORLAP_SOURCE_BOUNDS_POLICY", "warn"),
        help=(
            "How VorLap interpolation handles Re/AOA queries outside the airfoil FFT source grid. "
            "Use 'raise' for strict source-coverage debugging; the campaign default is 'warn' so "
            "runs complete while source-coverage diagnostics record clipping."
        ),
    )
    parser.add_argument(
        "--source-frequency-truncation",
        choices=("none", "nyquist"),
        default=os.environ.get("VORLAP_SOURCE_FREQUENCY_TRUNCATION", "none"),
        help=(
            "Optional fixed source-band policy. 'nyquist' discards VorLap oscillatory tones "
            "above 0.5/TIMESTEP; the default preserves the unfiltered source spectrum."
        ),
    )
    parser.add_argument("--force-scale", type=float, default=float(os.environ.get("VORLAP_FORCE_SCALE", "1.0")))
    parser.add_argument(
        "--diagnostic-force-scale",
        action="store_true",
        help="Allow --force-scale values above 1.0 for explicit diagnostic/amplitude-sensitivity runs.",
    )
    parser.add_argument("--progress-every", type=int, default=int(os.environ.get("QBLADE_PROGRESS_EVERY", "250")))
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(os.environ.get("QBLADE_CHECKPOINT_EVERY", "0")),
        help="Store/reuse QBlade project restart checkpoints every N completed timesteps; 0 disables.",
    )
    parser.add_argument(
        "--resume-initialize",
        action="store_true",
        help="Call initializeSimulation() after loading a restart checkpoint project.",
    )
    parser.add_argument(
        "--completed-checkpoint-export-only",
        action="store_true",
        help=(
            "Recover a final checkpoint by exporting filtered results without initialization "
            "or timestep advancement. Intended for the recover-export command."
        ),
    )
    parser.add_argument(
        "--coupling",
        choices=("sil-vorlap", "qblade-external", "none"),
        default=os.environ.get("VORLAP_QBLADE_COUPLING", "sil-vorlap"),
        help=(
            "Coupling path used by the QBlade runner. 'sil-vorlap' evaluates VorLap "
            "from Python and applies forces with setExternalAction(); 'qblade-external' "
            "relies on QBlade's .trb external-library callback."
        ),
    )
    parser.add_argument(
        "--allow-coarse-timestep",
        action="store_true",
        help="Allow QBLADE_TIMESTEP values above 0.01 s despite observed structural divergence.",
    )
    parser.add_argument("--force-setup", action="store_true")
    parser.add_argument("--force-bridge-build", action="store_true")
    parser.add_argument(
        "--skip-stable-support-patch",
        action="store_true",
        help="Leave the VorLap reference support structure unmodified. The current exported model fails Linux SIL initialization without the patch.",
    )
    parser.add_argument("--skip-vorlap-install", action="store_true")
    parser.add_argument("--skip-bridge-build", action="store_true")
    parser.add_argument("--store-project", action="store_true")
    parser.add_argument(
        "--native-run-full-simulation",
        action="store_true",
        help=(
            "For qblade-external diagnostics, call QBlade runFullSimulation() "
            "instead of the wrapper timestep loop."
        ),
    )
    parser.add_argument(
        "--record-sil-vorlap-history",
        action="store_true",
        help="For SIL VorLap cases, write per-step velocity-input and force-output history CSVs.",
    )
    parser.add_argument(
        "--sil-vorlap-history-mode",
        choices=("off", "full", "compact"),
        default=None,
        help=(
            "Explicit per-step SIL history mode passed to run_qblade_sim.py. Use compact "
            "for fixed-width qualification histories; the legacy recording flag remains full mode."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Export completed timesteps if QBlade reports a timestep failure.",
    )
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    args.qblade_root = args.qblade_root.resolve()
    args.vorlap_root = args.vorlap_root.resolve()
    if args.result_filter is not None:
        args.result_filter = args.result_filter.expanduser().resolve()
    validate_force_scale(args.force_scale, args.diagnostic_force_scale)
    return args


def main() -> None:
    args = parse_args()
    if args.timestep > 0.01 and not args.allow_coarse_timestep:
        raise ValueError(
            f"Refusing coarse QBlade timestep {args.timestep:g} s. The reference "
            "headless case has been observed to diverge at 0.1 s; use the stable "
            "0.0075 s example timestep or pass --allow-coarse-timestep for an explicit diagnostic run."
        )
    if args.command == "setup":
        manifest = setup(args)
        list_cases(manifest)
        return

    if args.command == "list":
        manifest = load_manifest(args)
        list_cases(manifest)
        return

    if args.command in {"run-existing", "recover-export"}:
        manifest_path = args.run_root / "cases.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{args.command} requires an existing case manifest and never performs setup: {manifest_path}"
            )
        if args.result_filter is None:
            raise ValueError(f"{args.command} requires --result-filter.")
        args.completed_checkpoint_export_only = args.command == "recover-export"
        index = args.case_index
        if index is None:
            array_index = os.environ.get("SLURM_ARRAY_TASK_ID")
            index = int(array_index) if array_index is not None else 0
        manifest = load_manifest(args)
        if args.command == "recover-export" and completed_filtered_export_ready(
            manifest["cases"][index],
            args.result_filter,
            require_completed_checkpoint_export_only=True,
        ):
            print(
                "Completed filtered checkpoint export is already present and hash-verified; "
                "skipping QBlade reload/export.",
                flush=True,
            )
            return
        run_case(args, manifest, index)
        return

    if args.command == "run":
        index = args.case_index
        if index is None:
            array_index = os.environ.get("SLURM_ARRAY_TASK_ID")
            index = int(array_index) if array_index is not None else 0
        manifest = setup(args)
        if args.result_filter is not None and completed_filtered_export_ready(
            manifest["cases"][index], args.result_filter
        ):
            print(
                "Completed filtered result is already present and hash-verified; "
                "skipping the completed QBlade case.",
                flush=True,
            )
            return
        run_case(args, manifest, index)
        return

    if args.command == "run-all":
        manifest = setup(args)
        for case in manifest["cases"]:
            if args.result_filter is not None and completed_filtered_export_ready(
                case, args.result_filter
            ):
                print(
                    f"Completed filtered result for {case['label']} is already present and "
                    "hash-verified; skipping the completed QBlade case.",
                    flush=True,
                )
                continue
            run_case(args, manifest, int(case["index"]))


if __name__ == "__main__":
    main()
