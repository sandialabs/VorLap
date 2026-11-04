#!/usr/bin/env python3
"""Generate, run, analyze, and gate sealed H-VAWT response campaigns."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.hvawt_response_pilot import (
    analyze_campaign,
    authorize_full_rerun,
    bind_flight_runtime_identity,
    command_for_case,
    execute_case,
    generate_campaign,
    render_pilot_result_artifacts,
    validate_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate", help="Generate a sealed preflight or full input tree."
    )
    generate.add_argument("--campaign-root", type=Path, required=True)
    generate.add_argument(
        "--rayleigh-analysis",
        "--rayleigh-selection",
        dest="rayleigh_analysis",
        type=Path,
        required=True,
        help="Completed decay analysis or source-defined Rayleigh selection artifact.",
    )
    generate.add_argument(
        "--runtime-vorlap-root",
        type=Path,
        default=REPO_ROOT,
        help="VorLap checkout/import path that QBlade's embedded Python will use at runtime.",
    )
    generate.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "run_headless/config/hvawt_response_campaign_spec.json",
    )
    generate.add_argument(
        "--campaign-kind",
        choices=(
            "two_case_120s_preflight",
            "two_speed_120s_h0_complete_test",
            "full_660s_resonance_sweep",
        ),
        default="full_660s_resonance_sweep",
        help=(
            "Select the exact native/hybrid preflight, two-speed matched "
            "H0/complete-v8 demonstration, or 20-case full campaign."
        ),
    )

    validate = commands.add_parser(
        "validate", help="Rehash and validate generated inputs."
    )
    validate.add_argument("--campaign-root", type=Path, required=True)

    bind_runtime = commands.add_parser(
        "bind-runtime",
        help="Create or verify the exact native QBlade/bridge/Python runtime receipt.",
    )
    bind_runtime.add_argument("--campaign-root", type=Path, required=True)
    bind_runtime.add_argument("--qblade-root", type=Path, required=True)

    list_cases = commands.add_parser("list", help="List the sealed campaign indices.")
    list_cases.add_argument("--campaign-root", type=Path, required=True)

    common_run = argparse.ArgumentParser(add_help=False)
    common_run.add_argument("--campaign-root", type=Path, required=True)
    common_run.add_argument("--qblade-root", type=Path, required=True)
    common_run.add_argument("--case-index", type=int, default=None)
    common_run.add_argument("--threads", type=int, default=1)
    common_run.add_argument("--checkpoint-every", type=int, default=0)
    common_run.add_argument("--progress-every", type=int, default=1000)
    common_run.add_argument(
        "--resume",
        action="store_true",
        help="Rejected for actual-aero cases; retained only to fail old commands clearly.",
    )
    common_run.add_argument("--python", default=sys.executable)
    common_run.add_argument(
        "--archive-partial-and-restart",
        action="store_true",
        help="Preserve a blocked partial attempt under attempt_archive and rerun cleanly.",
    )
    commands.add_parser(
        "show-command", parents=[common_run], help="Print one exact case command."
    )
    commands.add_parser(
        "run", parents=[common_run], help="Run one explicit/Slurm-array pilot case."
    )

    analyze = commands.add_parser("analyze", help="Analyze available pilot results.")
    analyze.add_argument("--campaign-root", type=Path, required=True)
    analyze.add_argument("--output", type=Path, default=None)

    plot = commands.add_parser(
        "plot",
        help="Render a complete pilot analysis into paper-facing PDF and CSV artifacts.",
    )
    plot.add_argument("--campaign-root", type=Path, required=True)
    plot.add_argument("--analysis", type=Path, default=None)
    plot.add_argument("--output-pdf", type=Path, default=None)
    plot.add_argument("--output-csv", type=Path, default=None)

    authorize = commands.add_parser(
        "authorize-full-rerun",
        help="Promote the prepared 68-run plan only after the pilot passes.",
    )
    authorize.add_argument("--campaign-root", type=Path, required=True)
    authorize.add_argument("--analysis", type=Path, default=None)
    authorize.add_argument("--output", type=Path, default=None)
    return parser


def _case_index(args: argparse.Namespace) -> int:
    if args.case_index is not None:
        return int(args.case_index)
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError(
            "Provide --case-index or run inside a Slurm array with SLURM_ARRAY_TASK_ID."
        )
    return int(raw)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        path = generate_campaign(
            campaign_root=args.campaign_root,
            rayleigh_analysis_path=args.rayleigh_analysis,
            repo_root=REPO_ROOT,
            spec_path=args.spec,
            runtime_vorlap_root=args.runtime_vorlap_root,
            campaign_kind=args.campaign_kind,
        )
        print(path)
        return 0
    if args.command == "validate":
        state = validate_campaign(args.campaign_root)
        print(
            json.dumps(
                {
                    "valid": state["valid"],
                    "campaign_root": str(state["campaign_root"]),
                    "case_count": len(state["manifest"]["cases"]),
                    "fingerprint": state["manifest"]["fingerprint"],
                    "full_rerun_execution_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "bind-runtime":
        receipt = bind_flight_runtime_identity(
            args.campaign_root, qblade_root=args.qblade_root
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "list":
        state = validate_campaign(args.campaign_root)
        for case in state["manifest"]["cases"]:
            print(
                f"{case['index']:2d}  {case['case_id']}  "
                f"U={case['wind_speed_mps']:g} m/s  dt={case['timestep_s']:g} s  "
                f"{case['scenario']}"
            )
        return 0
    if args.command in {"show-command", "run"}:
        index = _case_index(args)
        if args.command == "show-command":
            command = command_for_case(
                args.campaign_root,
                case_index=index,
                qblade_root=args.qblade_root,
                python_executable=args.python,
                threads=args.threads,
                checkpoint_every=args.checkpoint_every,
                progress_every=args.progress_every,
                resume=args.resume,
            )
            print(shlex.join(command))
        else:
            execute_case(
                args.campaign_root,
                case_index=index,
                qblade_root=args.qblade_root,
                python_executable=args.python,
                threads=args.threads,
                checkpoint_every=args.checkpoint_every,
                progress_every=args.progress_every,
                resume=args.resume,
                archive_partial_and_restart=args.archive_partial_and_restart,
            )
        return 0
    if args.command == "analyze":
        path = analyze_campaign(args.campaign_root, output_path=args.output)
        report = json.loads(path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {"path": str(path), "status": report["status"]},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["status"] in {
            "preflight_passed_complete_hybrid_screen",
            "two_speed_test_passed_terminal_audit",
            "pilot_passed_complete_hybrid_screen",
        } else 2
    if args.command == "plot":
        artifacts = render_pilot_result_artifacts(
            args.campaign_root,
            analysis_path=args.analysis,
            output_pdf=args.output_pdf,
            output_csv=args.output_csv,
        )
        print(json.dumps(artifacts, indent=2, sort_keys=True))
        return 0
    if args.command == "authorize-full-rerun":
        path = authorize_full_rerun(
            args.campaign_root,
            analysis_path=args.analysis,
            output_path=args.output,
        )
        print(path)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
