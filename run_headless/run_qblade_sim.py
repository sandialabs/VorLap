#!/usr/bin/env python3
"""Run a QBlade simulation definition through the QBlade SIL interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.qblade_files import (
    read_qblade_float_keyword as _read_qblade_float_keyword,
    read_qblade_keyword as _read_qblade_keyword,
    resolve_qblade_path as _resolve_qblade_path,
)
from vorlap.qblade_bridge_service import write_run_geometry_audit_reported_positions
from vorlap.qblade_sil import (
    SilConstantForceCoupler,
    SilTimingDiagnostics,
    SilVorLapCoupler,
    find_qblade_external_parameter_file,
    force_sign_probe_observations,
    normalize_qblade_header,
    parse_bool_flag,
    parse_constant_force_action,
    qblade_action_target_from_node,
    qblade_node_position_label,
    qblade_position_channel_name,
    qblade_result_time_diagnostics,
    read_qblade_result_numeric_columns,
)
from vorlap.qblade_run_support import (
    atomic_write_json,
    build_reproducibility_metadata,
    checkpoint_directory,
    clear_native_bridge_log,
    ensure_qblade_loader_env,
    file_sha256,
    find_bridge_library,
    find_qblade_library,
    git_metadata,
    load_json_file,
    load_qblade_api,
    load_resume_checkpoint,
    load_vorlap_config_for_sim,
    native_bridge_log_path,
    native_callback_evidence,
    package_versions,
    preflight_qblade_wind_input,
    prepend_env_path,
    prune_restart_checkpoints,
    qblade_case_settings,
    qblade_library_dirs,
    selected_environment,
    vorlap_package_metadata,
    write_restart_checkpoint,
)
from vorlap.qblade_run import QBladeRunRequest, QBladeRunResult, run_qblade_case


def parse_args() -> argparse.Namespace:
    default_qblade = Path(__file__).resolve().parent / "QBladeCE_2.0.9.7"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", required=True, help="QBlade .sim file to run.")
    parser.add_argument(
        "--qblade-root",
        default=os.environ.get("QBLADE_ROOT", str(default_qblade)),
        help="QBlade CE root containing libQBladeCE_*.so and SIL_Interface.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for exported results, run log, and summary metadata.",
    )
    parser.add_argument(
        "--result-name",
        default=None,
        help="Exported QBlade result base name. Defaults to '<sim stem>_results'.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override NUMTIMESTEPS from the .sim file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", "1")),
        help="QBlade/OpenMP thread count.",
    )
    parser.add_argument("--device", type=int, default=1, help="QBlade OpenCL device index.")
    parser.add_argument("--group-size", type=int, default=32, help="QBlade OpenCL group size.")
    parser.add_argument(
        "--use-opencl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable QBlade OpenCL acceleration if the cluster node supports it.",
    )
    parser.add_argument(
        "--store-project",
        action="store_true",
        help="Also store a completed QBlade project in the output directory.",
    )
    parser.add_argument(
        "--result-filter",
        default=os.environ.get("QBLADE_RESULT_FILTER"),
        help=(
            "QBlade result-filter file passed to exportResults(). This limits the exported "
            "columns; generated .sim FILTERFILE storage is configured by the batch setup."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(os.environ.get("QBLADE_CHECKPOINT_EVERY", "0")),
        help=(
            "Store a QBlade project checkpoint every N completed timesteps. "
            "Use 0 to disable wrapper-level checkpointing."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for QBlade project checkpoints. Defaults to <output-dir>/checkpoints.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from the latest checkpoint metadata in --checkpoint-dir when present.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Specific checkpoint JSON metadata file to resume from.",
    )
    parser.add_argument(
        "--resume-initialize",
        action="store_true",
        help=(
            "Call initializeSimulation() after loading a checkpoint project. "
            "The default leaves the stored project state untouched."
        ),
    )
    parser.add_argument(
        "--completed-checkpoint-export-only",
        action="store_true",
        help=(
            "Load a hash-verified checkpoint that already reached every requested timestep and "
            "perform only a filtered result export. No initialization or timestep advancement occurs."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Export the completed portion if QBlade reports a timestep failure after data has been produced.",
    )
    parser.add_argument(
        "--sil-vorlap",
        action="store_true",
        help=(
            "Drive the VorLap bridge from this Python SIL loop using the generated "
            "Control/vorlap_qblade_external.json and QBlade setExternalAction()."
        ),
    )
    parser.add_argument(
        "--sil-constant-force",
        action="append",
        default=[],
        metavar="NODE:AXIS:VALUE[:LOCAL]",
        help=(
            "Apply a constant diagnostic ADDFORCE through the Python SIL loop. "
            "NODE uses QBlade node ids such as BLD_1_0.230455, AXIS is X/Y/Z, "
            "VALUE is in N, and optional LOCAL is true/false/local/global. "
            "Repeat the option for multi-axis sign probes."
        ),
    )
    parser.add_argument(
        "--sil-modal-burst-plan",
        default=None,
        help=(
            "Apply an immutable scheduled H-VAWT modal burst through the Python SIL "
            "step loop. The JSON plan hash-binds its external layout, full 12-mode "
            "action basis, and force dual."
        ),
    )
    parser.add_argument(
        "--coupling-mode",
        choices=(
            "sil-vorlap",
            "sil-constant-force",
            "sil-modal-burst",
            "qblade-external",
            "none",
        ),
        default=None,
        help=(
            "Explicit coupling label written to run metadata. Defaults to sil-vorlap "
            "when --sil-vorlap is set, sil-constant-force when --sil-constant-force "
            "is used, sil-modal-burst when --sil-modal-burst-plan is used, otherwise "
            "qblade-external."
        ),
    )
    parser.add_argument(
        "--native-run-full-simulation",
        action="store_true",
        help=(
            "Diagnostic mode for qblade-external proof runs: call QBlade "
            "runFullSimulation() instead of advancing timesteps from the Python wrapper. "
            "This lets QBlade own the native external-library callback phase."
        ),
    )
    parser.add_argument(
        "--case-metadata",
        default=None,
        help="Optional JSON sidecar with batch/campaign case settings to embed in the run summary.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N timesteps. Use 0 to disable.",
    )
    parser.add_argument(
        "--record-sil-vorlap-history",
        action="store_true",
        help="Write per-step SIL VorLap velocity-input and force-output history CSVs for parity plots.",
    )
    parser.add_argument(
        "--sil-vorlap-history-mode",
        choices=("off", "full", "compact"),
        default=None,
        help=(
            "Explicit SIL history mode. 'compact' retains fixed-width response, force-work, "
            "aggregate-force, and cryptographic lag-provenance diagnostics without per-node "
            "vectors. The legacy --record-sil-vorlap-history flag remains exact full mode."
        ),
    )
    return parser.parse_args()


def read_qblade_keyword(path: Path, keyword: str) -> str | None:
    return _read_qblade_keyword(path, keyword)


def read_float_keyword(path: Path, keyword: str) -> float:
    return _read_qblade_float_keyword(path, keyword)


def resolve_qblade_path(base_file: Path, value: str) -> Path:
    return _resolve_qblade_path(base_file, value)


def finalize_run_geometry_audit(result: QBladeRunResult) -> dict[str, object]:
    """Record QBlade-reported node positions and persist their diagnostics."""
    reported_audit_path = result.result_path.with_name(
        f"{result.result_path.stem}_geometry_audit_reported.csv"
    )
    diagnostics = write_run_geometry_audit_reported_positions(
        result,
        output_path=reported_audit_path,
    )
    result.summary["geometry_audit_reported_diagnostics"] = diagnostics
    atomic_write_json(result.summary_path, result.summary)
    return diagnostics


def main() -> None:
    args = parse_args()
    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=args.sim,
            qblade_root=args.qblade_root,
            output_dir=args.output_dir,
            result_name=args.result_name,
            steps=args.steps,
            threads=args.threads,
            device=args.device,
            group_size=args.group_size,
            use_opencl=args.use_opencl,
            store_project=args.store_project,
            result_filter=args.result_filter,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            resume_checkpoint=args.resume_checkpoint,
            resume_initialize=args.resume_initialize,
            completed_checkpoint_export_only=args.completed_checkpoint_export_only,
            allow_partial=args.allow_partial,
            sil_vorlap=args.sil_vorlap,
            sil_modal_burst_plan=args.sil_modal_burst_plan,
            sil_constant_force_specs=tuple(args.sil_constant_force),
            coupling_mode=args.coupling_mode,
            qblade_execution_mode="run-full-simulation" if args.native_run_full_simulation else "step-loop",
            case_metadata_path=args.case_metadata,
            progress_every=args.progress_every,
            record_sil_vorlap_history=args.record_sil_vorlap_history,
            sil_vorlap_history_mode=args.sil_vorlap_history_mode,
            command_line=[sys.executable, *sys.argv],
            message_callback=lambda message: print(message, flush=True),
        )
    )
    finalize_run_geometry_audit(result)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
