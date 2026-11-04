#!/usr/bin/env python3
"""Build, validate, inspect, or select rows from a parked DLC manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
VORLAP_ROOT = SCRIPT_DIR.parent
if str(VORLAP_ROOT) not in sys.path:
    sys.path.insert(0, str(VORLAP_ROOT))

from vorlap.dlc_manifest import (  # noqa: E402
    DLCManifestError,
    build_manifest,
    disk_preflight,
    load_manifest,
    manifest_summary,
    selected_row,
    write_json,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Expand a compact JSON matrix specification.")
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument(
        "--require-approved",
        action="store_true",
        help="Fail unless the generated manifest is approved and passes production seed/weight gates.",
    )

    validate = subparsers.add_parser("validate", help="Validate an existing manifest.")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--allow-draft", action="store_true")

    count = subparsers.add_parser("count", help="Print the number of manifest rows.")
    count.add_argument("--manifest", type=Path, required=True)
    count.add_argument("--allow-draft", action="store_true")

    row = subparsers.add_parser("row", help="Print one manifest row as JSON.")
    row.add_argument("--manifest", type=Path, required=True)
    row.add_argument("--index", type=int, required=True)
    row.add_argument("--allow-draft", action="store_true")

    disk = subparsers.add_parser("disk-check", help="Fail if the wind filesystem lacks estimated capacity.")
    disk.add_argument("--manifest", type=Path, required=True)
    disk.add_argument("--allow-draft", action="store_true")
    disk.add_argument("--required-factor", type=float, default=1.25)
    disk.add_argument("--available-bytes", type=int, default=None, help=argparse.SUPPRESS)

    root = subparsers.add_parser("root", help="Print the manifest wind or run root.")
    root.add_argument("--manifest", type=Path, required=True)
    root.add_argument("--allow-draft", action="store_true")
    root.add_argument("--kind", choices=("wind", "run"), required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "build":
            spec = json.loads(args.spec.read_text(encoding="utf-8"))
            manifest = build_manifest(spec)
            if args.require_approved:
                from vorlap.dlc_manifest import validate_manifest

                validate_manifest(manifest, require_approved=True)
            write_json(args.out, manifest)
            print(json.dumps(manifest_summary(manifest), indent=2, sort_keys=True))
            return 0

        manifest = load_manifest(args.manifest, require_approved=not args.allow_draft)
        if args.command == "validate":
            print(json.dumps(manifest_summary(manifest), indent=2, sort_keys=True))
        elif args.command == "count":
            print(len(manifest["rows"]))
        elif args.command == "row":
            print(json.dumps(selected_row(manifest, args.index), indent=2, sort_keys=True))
        elif args.command == "disk-check":
            print(
                json.dumps(
                    disk_preflight(
                        manifest,
                        required_factor=args.required_factor,
                        available_bytes=args.available_bytes,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "root":
            print(manifest[f"{args.kind}_root"])
        return 0
    except (DLCManifestError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
