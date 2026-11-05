#!/usr/bin/env python3
"""Audit and recover completed H-VAWT one-way artifacts without rerunning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from vorlap.hvawt_complete_dlc_recovery import (  # noqa: E402
    HVAWTOneWayRecoveryError,
    load_recovery_receipt,
    preflight_recovery,
    write_recovered_analysis,
    write_recovery_receipt,
)


RECOVERY_SOURCES = {
    "recovery_cli": Path(__file__).resolve(),
    "recovery_submission": (
        REPOSITORY / "run_headless/submit_hvawt_one_way_audit_recovery.sj"
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("preflight", "Validate the 68/102 recovery preconditions without writing."),
        ("recover", "Audit all artifacts and write one recovery receipt."),
        ("verify", "Reproduce an existing recovery receipt from live artifacts."),
        ("analyze", "Write the versioned recovered 102-case analysis."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--campaign-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight_recovery(args.campaign_root)
        elif args.command == "recover":
            result = write_recovery_receipt(
                args.campaign_root, recovery_sources=RECOVERY_SOURCES
            )
        elif args.command == "verify":
            result = load_recovery_receipt(
                args.campaign_root,
                verify_live_artifacts=True,
                recovery_sources=RECOVERY_SOURCES,
            )
        elif args.command == "analyze":
            result = write_recovered_analysis(
                args.campaign_root, recovery_sources=RECOVERY_SOURCES
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (HVAWTOneWayRecoveryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
