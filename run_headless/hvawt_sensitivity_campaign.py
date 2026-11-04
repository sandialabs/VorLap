#!/usr/bin/env python3
"""Prepare, execute, audit, and promote the targeted H-VAWT sensitivities."""

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

from vorlap.hvawt_sensitivity_campaign import (  # noqa: E402
    analyze_campaign,
    bind_execution_authorization,
    bind_runtime,
    classify_case,
    command_for_case,
    execute_case,
    execute_smoke,
    generate_campaign,
    promote_analysis,
    validate_base_preflight,
    validate_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight", help="Read-only login-safe accepted-v3 source check."
    )
    preflight.add_argument("--base-campaign-root", type=Path, required=True)

    generate = commands.add_parser(
        "generate", help="Generate the separate sealed six-case campaign."
    )
    generate.add_argument("--campaign-root", type=Path, required=True)
    generate.add_argument("--base-campaign-root", type=Path, required=True)

    validate = commands.add_parser(
        "validate", help="Validate the sealed sensitivity inputs."
    )
    validate.add_argument("--campaign-root", type=Path, required=True)
    validate.add_argument("--skip-case-rehash", action="store_true")

    bind = commands.add_parser(
        "bind-runtime", help="Bind the exact successful Flight runtime."
    )
    bind.add_argument("--campaign-root", type=Path, required=True)
    bind.add_argument("--qblade-root", type=Path, required=True)

    smoke = commands.add_parser(
        "smoke", help="Run all six four-step input/runtime smokes."
    )
    smoke.add_argument("--campaign-root", type=Path, required=True)
    smoke.add_argument("--qblade-root", type=Path, required=True)
    smoke.add_argument("--python", default=sys.executable)
    smoke.add_argument("--threads", type=int, required=True)

    authorize = commands.add_parser(
        "authorize", help="Authorize the six exact cases after smoke review."
    )
    authorize.add_argument("--campaign-root", type=Path, required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Generate, validate, bind, smoke, and authorize without production runs.",
    )
    prepare.add_argument("--campaign-root", type=Path, required=True)
    prepare.add_argument("--base-campaign-root", type=Path, required=True)
    prepare.add_argument("--qblade-root", type=Path, required=True)
    prepare.add_argument("--python", default=sys.executable)
    prepare.add_argument("--threads", type=int, required=True)

    list_cases = commands.add_parser("list", help="List six ordered cases.")
    list_cases.add_argument("--campaign-root", type=Path, required=True)

    for name, help_text in (
        ("classify", "Classify one result without writing."),
        ("show-command", "Print one exact sealed QBlade command."),
        ("run", "Run one case fresh or reuse an exact terminal result."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--campaign-root", type=Path, required=True)
        command.add_argument("--case-index", type=int, default=None)
        if name in {"show-command", "run"}:
            command.add_argument("--qblade-root", type=Path, required=True)
            command.add_argument("--python", default=sys.executable)
            command.add_argument("--threads", type=int, required=True)

    analyze = commands.add_parser(
        "analyze", help="Audit all six results and evaluate frozen gates."
    )
    analyze.add_argument("--campaign-root", type=Path, required=True)
    analyze.add_argument("--base-campaign-root", type=Path, required=True)
    analyze.add_argument("--output", type=Path, default=None)

    promote = commands.add_parser(
        "promote", help="Write the hash-bound sensitivity paper handoff."
    )
    promote.add_argument("--campaign-root", type=Path, required=True)
    promote.add_argument("--analysis", type=Path, default=None)
    promote.add_argument("--output-dir", type=Path, default=None)
    return parser


def _case_index(explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError("Provide --case-index or run inside a Slurm array.")
    return int(raw)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        state = validate_base_preflight(args.base_campaign_root)
        _print(
            {
                "schema": "vorlap-hvawt-sensitivity-preflight-v1",
                "status": "accepted_v3_source_ready_for_scheduled_preparation",
                "base_campaign_root": state["base_campaign_root"],
                "base_campaign_fingerprint": state["manifest"]["fingerprint"],
                "selection_fingerprint": state["selection"]["fingerprint"],
                "selected_complete_case_indices": [98, 92],
                "selected_nominal_case_indices": [90, 91, 92, 96, 97, 98],
                "writes_performed": 0,
            }
        )
        return 0
    if args.command == "generate":
        print(
            generate_campaign(
                args.campaign_root,
                base_campaign_root=args.base_campaign_root,
                repository=REPO_ROOT,
            )
        )
        return 0
    if args.command == "validate":
        state = validate_campaign(
            args.campaign_root,
            rehash_case_assets=not args.skip_case_rehash,
        )
        _print(
            {
                "valid": state["valid"],
                "campaign_root": state["campaign_root"],
                "fingerprint": state["manifest"]["fingerprint"],
                "case_count": state["manifest"]["case_count"],
                "case_assets_rehashed": not args.skip_case_rehash,
            }
        )
        return 0
    if args.command == "bind-runtime":
        _print(bind_runtime(args.campaign_root, qblade_root=args.qblade_root))
        return 0
    if args.command == "smoke":
        _print(
            execute_smoke(
                args.campaign_root,
                qblade_root=args.qblade_root,
                python_executable=args.python,
                threads=args.threads,
            )
        )
        return 0
    if args.command == "authorize":
        _print(bind_execution_authorization(args.campaign_root))
        return 0
    if args.command == "prepare":
        generate_campaign(
            args.campaign_root,
            base_campaign_root=args.base_campaign_root,
            repository=REPO_ROOT,
        )
        bind_runtime(args.campaign_root, qblade_root=args.qblade_root)
        execute_smoke(
            args.campaign_root,
            qblade_root=args.qblade_root,
            python_executable=args.python,
            threads=args.threads,
        )
        _print(bind_execution_authorization(args.campaign_root))
        return 0
    if args.command == "list":
        state = validate_campaign(
            args.campaign_root, rehash_case_assets=False
        )
        for case in state["manifest"]["cases"]:
            print(
                f"{case['index']:2d} row={case['row_id']} "
                f"variant={case['variant']} dt={case['timestep_s']:.5f} "
                f"cap={case['source_frequency_cap_hz']:.12g}"
            )
        return 0
    if args.command == "classify":
        _print(
            classify_case(
                args.campaign_root, case_index=_case_index(args.case_index)
            )
        )
        return 0
    if args.command in {"show-command", "run"}:
        index = _case_index(args.case_index)
        if args.command == "show-command":
            print(
                shlex.join(
                    command_for_case(
                        args.campaign_root,
                        case_index=index,
                        qblade_root=args.qblade_root,
                        python_executable=args.python,
                        threads=args.threads,
                    )
                )
            )
        else:
            _print(
                execute_case(
                    args.campaign_root,
                    case_index=index,
                    qblade_root=args.qblade_root,
                    python_executable=args.python,
                    threads=args.threads,
                )
            )
        return 0
    if args.command == "analyze":
        _print(
            analyze_campaign(
                args.campaign_root,
                base_campaign_root=args.base_campaign_root,
                output=args.output,
            )
        )
        return 0
    if args.command == "promote":
        _print(
            promote_analysis(
                args.campaign_root,
                analysis_path=args.analysis,
                output_dir=args.output_dir,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
