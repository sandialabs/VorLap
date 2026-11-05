#!/usr/bin/env python3
"""Build a hashed production archive manifest for a parked VorLap/QBlade run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


PRODUCTION_ARCHIVE_MANIFEST_DEFAULT = "production_archive_manifest.json"
REQUIRED_CATEGORIES = (
    "audit_json",
    "bridge_log",
    "figure_provenance",
    "postprocess_csv",
    "qblade_log",
    "run_summary",
)
CATEGORY_GLOBS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("campaign_settings", ("campaign_settings.json",)),
    ("case_manifest", ("*/cases.json",)),
    ("dlc_manifest_row", ("dlc_manifest_row.json",)),
    ("execution_environment", ("execution_environment_*.json",)),
    ("setup_manifest", ("*/setup_complete.json",)),
    ("audit_json", ("audits/*.json",)),
    ("bridge_log", ("**/vorlap_qblade_debug.log", "**/vorlap_qblade_bridge_cpp.log")),
    ("figure_provenance", ("postprocess/figure_provenance.json", "postprocess/figure_provenance*.json")),
    ("postprocess_csv", ("postprocess/*.csv",)),
    ("postprocess_figure", ("postprocess/*.png", "postprocess/*.svg")),
    ("postprocess_tex", ("postprocess/*.tex",)),
    ("qblade_log", ("**/*_qblade.log", "**/debugLog.txt", "logs/*qblade*.log")),
    ("qblade_result", ("**/*_qblade_results.txt",)),
    ("run_metadata", ("**/case_metadata.json",)),
    ("run_summary", ("**/*_run_summary.json",)),
    ("wind_record", ("wind_record.json",)),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _iter_category_files(run_root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(run_root.glob(pattern)):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _artifact_entry(run_root: Path, path: Path, category: str) -> dict[str, object]:
    relative = path.relative_to(run_root).as_posix()
    return {
        "category": category,
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_manifest(run_root: Path, *, archive_id: str | None = None) -> dict[str, object]:
    root = run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {root}")

    artifacts: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for category, patterns in CATEGORY_GLOBS:
        for path in _iter_category_files(root, patterns):
            if not _is_relative_to(path.resolve(), root):
                continue
            relative = path.relative_to(root).as_posix()
            if relative in seen_paths:
                continue
            seen_paths.add(relative)
            artifacts.append(_artifact_entry(root, path, category))

    artifacts.sort(key=lambda item: (str(item["category"]), str(item["path"])))
    category_counts: dict[str, int] = {}
    for item in artifacts:
        category = str(item["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    return {
        "archive_id": archive_id or root.name,
        "run_root": str(root),
        "required_categories": list(REQUIRED_CATEGORIES),
        "missing_required_categories": [
            category for category in REQUIRED_CATEGORIES if category_counts.get(category, 0) <= 0
        ],
        "category_counts": dict(sorted(category_counts.items())),
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Completed campaign run root.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output manifest path. Defaults to RUN_ROOT/{PRODUCTION_ARCHIVE_MANIFEST_DEFAULT}.",
    )
    parser.add_argument("--archive-id", default=None, help="Stable archive identifier. Defaults to the run-root directory name.")
    parser.add_argument(
        "--fail-missing-required",
        action="store_true",
        help="Exit nonzero when any required final-audit archive category has no files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_path = (args.out or (run_root / PRODUCTION_ARCHIVE_MANIFEST_DEFAULT)).resolve()
    manifest = build_manifest(run_root, archive_id=args.archive_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    missing = manifest.get("missing_required_categories", [])
    if missing:
        print(f"Missing required archive categories: {', '.join(str(item) for item in missing)}")
        if args.fail_missing_required:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
