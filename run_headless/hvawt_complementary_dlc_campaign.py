#!/usr/bin/env python3
"""Authorize and execute the reviewed 82-case H-VAWT continuation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from hvawt_complementary_dlc_support import (  # noqa: E402
    bind_complementary_authorization,
    execute_case,
    execute_u50_smoke,
    verify_sealed_orchestration_sources,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight-sources",
        help="Verify sealed campaign-generation sources before calling sbatch.",
    )
    preflight.add_argument("--campaign-root", type=Path, required=True)

    authorize = commands.add_parser(
        "bind-authorization",
        help="Bind the reviewed 20-case checkpoint before the remaining 82 cases.",
    )
    authorize.add_argument("--campaign-root", type=Path, required=True)
    authorize.add_argument("--checkpoint-review", type=Path, default=None)

    smoke = commands.add_parser(
        "smoke-u50", help="Run the exact sealed four-step U50 complete-path smoke."
    )
    smoke.add_argument("--campaign-root", type=Path, required=True)
    smoke.add_argument("--qblade-root", type=Path, required=True)
    smoke.add_argument("--threads", type=int, default=1)
    smoke.add_argument("--progress-every", type=int, default=0)
    smoke.add_argument("--python", default=sys.executable)

    run = commands.add_parser(
        "run", help="Run or exactly reuse one sealed campaign case."
    )
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--qblade-root", type=Path, required=True)
    run.add_argument("--case-index", type=int, default=None)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--checkpoint-every", type=int, default=24_000)
    run.add_argument("--progress-every", type=int, default=1000)
    run.add_argument("--python", default=sys.executable)
    return parser


def _case_index(explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise ValueError("Provide --case-index or run inside a Slurm array.")
    return int(raw)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight-sources":
        receipt = verify_sealed_orchestration_sources(
            args.campaign_root,
            repository=REPOSITORY,
        )
    elif args.command == "bind-authorization":
        receipt = bind_complementary_authorization(
            args.campaign_root,
            evidence_path=args.checkpoint_review,
            repository=REPOSITORY,
        )
    elif args.command == "smoke-u50":
        receipt = execute_u50_smoke(
            args.campaign_root,
            qblade_root=args.qblade_root,
            python_executable=args.python,
            threads=args.threads,
            progress_every=args.progress_every,
        )
    elif args.command == "run":
        receipt = execute_case(
            args.campaign_root,
            case_index=_case_index(args.case_index),
            qblade_root=args.qblade_root,
            python_executable=args.python,
            threads=args.threads,
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
