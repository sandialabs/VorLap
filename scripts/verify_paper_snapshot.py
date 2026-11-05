#!/usr/bin/env python3
"""Verify or archive the exact separate manuscript inputs for this release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "reproducibility/sil_paper_manifest.json"


def verify_snapshot(paper_dir: Path, manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = paper_dir.resolve()
    issues = []
    for entry in manifest["files"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid snapshot path: {relative}")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Snapshot path escapes its root: {relative}")
        if not path.is_file():
            issues.append(f"missing: {relative}")
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            issues.append(f"changed: {relative}")
    if issues:
        raise ValueError("Paper snapshot does not match this release:\n" + "\n".join(issues))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper_drafts/NAWEA_QBladeBridge")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=Path, help="Write a verified, deterministic ZIP without Git history.")
    args = parser.parse_args()
    manifest = verify_snapshot(args.paper_dir, args.manifest)
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects an existing deposit from replacement.
        with zipfile.ZipFile(args.archive, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in manifest["files"]:
                info = zipfile.ZipInfo(entry["path"], date_time=(2026, 10, 28, 0, 0, 0))
                info.create_system = 3
                info.external_attr = entry["mode"] << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, (args.paper_dir / entry["path"]).read_bytes())
        digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        args.archive.with_suffix(args.archive.suffix + ".sha256").write_text(
            f"{digest}  {args.archive.name}\n", encoding="utf-8",
        )
        print(f"Wrote {args.archive}; SHA-256 {digest}")
    print(f"Verified {len(manifest['files'])} paper snapshot files.")


if __name__ == "__main__":
    main()
