#!/usr/bin/env python3
"""Prepare, execute, and audit the matched three-path H-VAWT DLC campaign."""

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

from vorlap.hvawt_complete_dlc_campaign import (  # noqa: E402
    analyze_campaign,
    analyze_checkpoint,
    bind_checkpoint_authorization,
    bind_execution_authorization,
    bind_runtime,
    classify_case,
    command_for_case,
    execute_case,
    execute_row,
    execute_u50_smoke,
    generate_campaign,
    validate_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser(
        "generate", help="Generate the sealed 34-row, 102-case campaign."
    )
    generate.add_argument("--campaign-root", type=Path, required=True)
    generate.add_argument("--rayleigh-selection", type=Path, required=True)
    generate.add_argument("--runtime-vorlap-root", type=Path, default=REPO_ROOT)
    generate.add_argument("--approved-manifest", type=Path, default=None)

    validate = commands.add_parser(
        "validate", help="Rehash and validate all campaign inputs."
    )
    validate.add_argument("--campaign-root", type=Path, required=True)
    validate.add_argument(
        "--sealed-only",
        action="store_true",
        help="Validate sealed manifests/current code without rehashing external winds/cases.",
    )

    bind = commands.add_parser(
        "bind-runtime", help="Bind the exact successful Flight Python/QBlade runtime."
    )
    bind.add_argument("--campaign-root", type=Path, required=True)
    bind.add_argument("--qblade-root", type=Path, required=True)

    authorize = commands.add_parser(
        "bind-authorization",
        help="Bind the passed 20-case pilot before turbulent execution.",
    )
    authorize.add_argument("--campaign-root", type=Path, required=True)
    authorize.add_argument("--pilot-campaign-root", type=Path, required=True)

    checkpoint_authorize = commands.add_parser(
        "bind-checkpoint-authorization",
        help="Bind the reviewed uniform evidence to the exact turbulent 20-case subset.",
    )
    checkpoint_authorize.add_argument("--campaign-root", type=Path, required=True)
    checkpoint_authorize.add_argument("--evidence", type=Path, default=None)

    list_cases = commands.add_parser("list", help="List all 102 ordered cases.")
    list_cases.add_argument("--campaign-root", type=Path, required=True)

    common_case = argparse.ArgumentParser(add_help=False)
    common_case.add_argument("--campaign-root", type=Path, required=True)
    common_case.add_argument("--case-index", type=int, default=None)

    classify = commands.add_parser(
        "classify", parents=[common_case], help="Classify one result without writing."
    )

    common_run = argparse.ArgumentParser(add_help=False, parents=[common_case])
    common_run.add_argument("--qblade-root", type=Path, required=True)
    common_run.add_argument("--threads", type=int, default=1)
    common_run.add_argument("--checkpoint-every", type=int, default=24_000)
    common_run.add_argument("--progress-every", type=int, default=1000)
    common_run.add_argument("--python", default=sys.executable)
    commands.add_parser(
        "show-command", parents=[common_run], help="Print one exact QBlade command."
    )
    commands.add_parser(
        "run", parents=[common_run], help="Run one case fresh or reuse an exact result."
    )
    commands.add_parser(
        "run-checkpoint",
        parents=[common_run],
        help="Run one case under the narrow turbulent 20-case authorization.",
    )

    smoke = commands.add_parser(
        "smoke-u50", help="Run the four-step U50 complete force-ceiling smoke."
    )
    smoke.add_argument("--campaign-root", type=Path, required=True)
    smoke.add_argument("--qblade-root", type=Path, required=True)
    smoke.add_argument("--threads", type=int, default=1)
    smoke.add_argument("--progress-every", type=int, default=0)
    smoke.add_argument("--python", default=sys.executable)

    run_row = commands.add_parser(
        "run-row", help="Run one raw/one-way/complete row sequentially."
    )
    run_row.add_argument("--campaign-root", type=Path, required=True)
    run_row.add_argument("--row-index", type=int, default=None)
    run_row.add_argument("--qblade-root", type=Path, required=True)
    run_row.add_argument("--threads", type=int, default=1)
    run_row.add_argument("--checkpoint-every", type=int, default=24_000)
    run_row.add_argument("--progress-every", type=int, default=1000)
    run_row.add_argument("--python", default=sys.executable)

    analyze = commands.add_parser(
        "analyze", help="Audit all 102 executions and write analysis.json."
    )
    analyze.add_argument("--campaign-root", type=Path, required=True)
    analyze.add_argument("--output", type=Path, default=None)
    analyze_checkpoint_parser = commands.add_parser(
        "analyze-checkpoint",
        help="Audit the exact 20 turbulent raw/complete checkpoint cases.",
    )
    analyze_checkpoint_parser.add_argument("--campaign-root", type=Path, required=True)
    analyze_checkpoint_parser.add_argument("--output", type=Path, default=None)
    return parser


def _array_index(explicit: int | None, *, name: str) -> int:
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError(
            f"Provide --{name.replace('_', '-')} or run inside a Slurm array."
        )
    return int(raw)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        path = generate_campaign(
            args.campaign_root,
            rayleigh_selection=args.rayleigh_selection,
            repository=REPO_ROOT,
            runtime_repository=args.runtime_vorlap_root,
            approved_manifest_path=args.approved_manifest,
        )
        print(path)
        return 0
    if args.command == "validate":
        state = validate_campaign(
            args.campaign_root,
            rehash_external_winds=not args.sealed_only,
            rehash_case_assets=not args.sealed_only,
            deep_case_validation=not args.sealed_only,
        )
        print(
            json.dumps(
                {
                    "valid": state["valid"],
                    "campaign_root": str(state["campaign_root"]),
                    "fingerprint": state["manifest"]["fingerprint"],
                    "row_count": state["manifest"]["row_count"],
                    "case_count": state["manifest"]["case_count"],
                    "external_winds_rehashed": not args.sealed_only,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "bind-runtime":
        receipt = bind_runtime(args.campaign_root, qblade_root=args.qblade_root)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "bind-authorization":
        receipt = bind_execution_authorization(
            args.campaign_root, pilot_campaign=args.pilot_campaign_root
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "bind-checkpoint-authorization":
        receipt = bind_checkpoint_authorization(
            args.campaign_root, evidence_path=args.evidence
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "list":
        state = validate_campaign(
            args.campaign_root,
            rehash_external_winds=False,
            rehash_case_assets=False,
            deep_case_validation=False,
        )
        for case in state["manifest"]["cases"]:
            print(
                f"{case['index']:3d}  row={case['row_index']:2d}  "
                f"{case['scenario']:25s}  {case['row_id']}"
            )
        return 0
    if args.command == "classify":
        index = _array_index(args.case_index, name="case_index")
        print(
            json.dumps(
                classify_case(args.campaign_root, case_index=index),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command in {"show-command", "run", "run-checkpoint"}:
        index = _array_index(args.case_index, name="case_index")
        if args.command == "show-command":
            command = command_for_case(
                args.campaign_root,
                case_index=index,
                qblade_root=args.qblade_root,
                python_executable=args.python,
                threads=args.threads,
                checkpoint_every=args.checkpoint_every,
                progress_every=args.progress_every,
            )
            print(shlex.join(command))
        else:
            receipt = execute_case(
                args.campaign_root,
                case_index=index,
                qblade_root=args.qblade_root,
                python_executable=args.python,
                threads=args.threads,
                checkpoint_every=args.checkpoint_every,
                progress_every=args.progress_every,
                authorization_scope=(
                    "turbulent_20case_checkpoint"
                    if args.command == "run-checkpoint"
                    else "full"
                ),
            )
            print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "smoke-u50":
        receipt = execute_u50_smoke(
            args.campaign_root,
            qblade_root=args.qblade_root,
            python_executable=args.python,
            threads=args.threads,
            progress_every=args.progress_every,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.command == "run-row":
        index = _array_index(args.row_index, name="row_index")
        receipts = execute_row(
            args.campaign_root,
            row_index=index,
            qblade_root=args.qblade_root,
            python_executable=args.python,
            threads=args.threads,
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
        )
        print(json.dumps(receipts, indent=2, sort_keys=True))
        return 0
    if args.command == "analyze":
        report = analyze_campaign(args.campaign_root, output=args.output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "complete_case_count": report["complete_case_count"],
                    "expected_case_count": report["expected_case_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["all_three_scenarios_complete"] is True else 2
    if args.command == "analyze-checkpoint":
        report = analyze_checkpoint(args.campaign_root, output=args.output)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "complete_checkpoint_case_count": report[
                        "complete_checkpoint_case_count"
                    ],
                    "expected_checkpoint_case_count": report[
                        "expected_checkpoint_case_count"
                    ],
                    "all_pairs_complete": report["all_pairs_complete"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["all_checkpoint_cases_complete"] is True else 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
