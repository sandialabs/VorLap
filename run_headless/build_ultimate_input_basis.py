#!/usr/bin/env python3
"""Draft an ultimate input-basis manifest from reviewed allowables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ULTIMATE_INPUT_BASIS_DEFAULT = "ultimate_input_basis.json"
DEFAULT_ALLOWABLES_CANDIDATES = (
    "postprocess/ultimate_allowables.csv",
    "postprocess/ultimate_allowables_template.csv",
)
REQUIRED_ALLOWABLES_COLUMNS = (
    "blade",
    "position",
    "quantity",
    "axis",
    "unit",
    "allowable_abs",
    "load_factor",
    "resistance_factor",
)


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _campaign_allowables(run_root: Path) -> Path | None:
    settings_path = run_root / "campaign_settings.json"
    if not settings_path.is_file():
        return None
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw = payload.get("ultimate_allowables")
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw.strip())
    return path if path.is_absolute() else run_root / path


def default_allowables_path(run_root: Path) -> Path:
    archived = _campaign_allowables(run_root)
    if archived is not None:
        return archived
    for relative in DEFAULT_ALLOWABLES_CANDIDATES:
        path = run_root / relative
        if path.is_file():
            return path
    return run_root / DEFAULT_ALLOWABLES_CANDIDATES[0]


def audit_allowables(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Allowables CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in REQUIRED_ALLOWABLES_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"Allowables CSV is missing required column(s): {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Allowables CSV has no rows: {path}")

    valid_rows = 0
    for row in rows:
        if any(not str(row.get(column, "")).strip() for column in ("blade", "quantity", "axis", "unit")):
            continue
        try:
            position = float(row.get("position", ""))
            allowable = float(row.get("allowable_abs", ""))
            load_factor = float(row.get("load_factor", ""))
            resistance_factor = float(row.get("resistance_factor", ""))
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in (position, allowable, load_factor, resistance_factor)):
            if allowable > 0.0 and load_factor > 0.0 and resistance_factor > 0.0:
                valid_rows += 1
    if valid_rows <= 0:
        raise ValueError("Allowables CSV has no complete rows with positive allowable_abs, load_factor, and resistance_factor.")
    return {"rows": len(rows), "valid_rows": valid_rows}


def build_manifest(
    run_root: Path,
    *,
    allowables: Path | None = None,
    basis_id: str | None = None,
    allowable_type: str,
    channel_mapping_basis: str,
    stress_recovery_or_local_load_basis: str,
    load_factor_basis: str,
    resistance_factor_basis: str,
    units_convention: str,
) -> dict[str, Any]:
    root = run_root.resolve()
    allowables_path = (allowables or default_allowables_path(root)).resolve()
    allowables_report = audit_allowables(allowables_path)
    return {
        "basis_id": basis_id or f"{root.name}-ultimate-input-basis",
        "review_status": "draft_generated_from_allowables",
        "allowables_artifact": _relative_or_absolute(root, allowables_path),
        "allowables_sha256": sha256_file(allowables_path),
        "allowables_summary": allowables_report,
        "allowable_type": allowable_type,
        "channel_mapping_basis": channel_mapping_basis,
        "stress_recovery_or_local_load_basis": stress_recovery_or_local_load_basis,
        "load_factor_basis": load_factor_basis,
        "resistance_factor_basis": resistance_factor_basis,
        "units_convention": units_convention,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Campaign run root.")
    parser.add_argument(
        "--allowables",
        type=Path,
        default=None,
        help="Completed allowables CSV. Defaults to campaign_settings.json ultimate_allowables or postprocess/ultimate_allowables.csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output manifest path. Defaults to RUN_ROOT/{ULTIMATE_INPUT_BASIS_DEFAULT}.",
    )
    parser.add_argument("--basis-id", default=None, help="Basis identifier. Defaults to '<run-root-name>-ultimate-input-basis'.")
    parser.add_argument("--allowable-type", default="DRAFT: reviewed local QBlade load allowable")
    parser.add_argument("--channel-mapping-basis", default="DRAFT: allowables keyed by blade, position, quantity, axis, and unit")
    parser.add_argument(
        "--stress-recovery-or-local-load-basis",
        default="DRAFT: local-load allowable basis; no material stress recovery documented by this generator",
    )
    parser.add_argument("--load-factor-basis", default="DRAFT: load_factor column in the completed allowables CSV")
    parser.add_argument("--resistance-factor-basis", default="DRAFT: resistance_factor column in the completed allowables CSV")
    parser.add_argument("--units-convention", default="DRAFT: QBlade exported local load units")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_path = (args.out or (run_root / ULTIMATE_INPUT_BASIS_DEFAULT)).resolve()
    if out_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing manifest without --force: {out_path}")
    manifest = build_manifest(
        run_root,
        allowables=args.allowables,
        basis_id=args.basis_id,
        allowable_type=args.allowable_type,
        channel_mapping_basis=args.channel_mapping_basis,
        stress_recovery_or_local_load_basis=args.stress_recovery_or_local_load_basis,
        load_factor_basis=args.load_factor_basis,
        resistance_factor_basis=args.resistance_factor_basis,
        units_convention=args.units_convention,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
