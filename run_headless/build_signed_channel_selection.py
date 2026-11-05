#!/usr/bin/env python3
"""Draft a signed fatigue-channel selection manifest from postprocess outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SIGNED_CHANNEL_SELECTION_DEFAULT = "signed_channel_selection.json"
SIGNED_DEL_COMPARISON_DEFAULT = "postprocess/signed_del_comparison.csv"
CRITICAL_CASE_SUMMARY_DEFAULT = "postprocess/critical_case_summary.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV artifact: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV artifact has no header: {path}")
        return list(reader)


def row_float(row: dict[str, str], column: str, default: float = math.nan) -> float:
    try:
        return float(row.get(column, ""))
    except (TypeError, ValueError):
        return default


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("case", "")).strip(),
        f"{row_float(row, 'position'):.9g}",
        str(row.get("quantity", "")).strip(),
        str(row.get("axis", "")).strip(),
        str(row.get("unit", "")).strip(),
        str(row.get("channel_vorlap", "")).strip(),
    )


def select_critical_row(critical_rows: list[dict[str, str]]) -> dict[str, str]:
    if not critical_rows:
        raise ValueError("critical_case_summary.csv has no rows.")
    finite_rows = [
        (row_float(row, "signed_del_ratio", -math.inf), row)
        for row in critical_rows
    ]
    return max(finite_rows, key=lambda item: item[0])[1]


def selection_from_row(row: dict[str, str], *, critical: bool) -> dict[str, Any]:
    selection: dict[str, Any] = {
        "case": str(row.get("case", "")).strip(),
        "position": row_float(row, "position"),
        "quantity": str(row.get("quantity", "")).strip(),
        "axis": str(row.get("axis", "")).strip(),
        "unit": str(row.get("unit", "")).strip(),
        "channel_vorlap": str(row.get("channel_vorlap", "")).strip(),
        "critical": critical,
    }
    if str(row.get("channel_baseline", "")).strip():
        selection["channel_baseline"] = str(row.get("channel_baseline", "")).strip()
    if math.isfinite(row_float(row, "signed_del_ratio")):
        selection["signed_del_ratio"] = row_float(row, "signed_del_ratio")
    if math.isfinite(row_float(row, "signed_delta_pct")):
        selection["signed_delta_pct"] = row_float(row, "signed_delta_pct")
    return selection


def campaign_wohler_exponent(run_root: Path) -> float | None:
    settings_path = run_root / "campaign_settings.json"
    if not settings_path.is_file():
        return None
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        value = float(payload.get("wohler_exponent"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def build_manifest(
    run_root: Path,
    *,
    selection_id: str | None = None,
    wohler_exponent: float | None = None,
) -> dict[str, Any]:
    root = run_root.resolve()
    signed_path = root / SIGNED_DEL_COMPARISON_DEFAULT
    critical_path = root / CRITICAL_CASE_SUMMARY_DEFAULT
    signed_rows = read_csv_rows(signed_path)
    critical_rows = read_csv_rows(critical_path)
    if not signed_rows:
        raise ValueError("signed_del_comparison.csv has no rows.")
    critical_row = select_critical_row(critical_rows)
    critical_key = row_key(critical_row)
    if not any(row_key(row) == critical_key for row in signed_rows):
        raise ValueError("Selected critical-case row does not match signed_del_comparison.csv.")

    exponent = wohler_exponent if wohler_exponent is not None else campaign_wohler_exponent(root)
    if exponent is None:
        exponent = 10.0

    return {
        "selection_id": selection_id or f"{root.name}-signed-channel-selection",
        "review_status": "draft_generated_from_postprocess",
        "normalization": {
            "metric": "signed_del_ratio",
            "baseline": "no_sil_baseline",
            "comparison": "vorlap_over_baseline",
            "wohler_exponent": float(exponent),
            "reference_cycle_convention": "rainflow cycles from retained QBlade result records",
            "transient_removal": "as configured by QBlade STOREFROM and postprocess inputs",
            "mean_stress_assumption": "not applied to load-channel DEL ratios",
        },
        "artifacts": {
            "signed_del_comparison": SIGNED_DEL_COMPARISON_DEFAULT,
            "critical_case_summary": CRITICAL_CASE_SUMMARY_DEFAULT,
        },
        "selection_method": "critical_case_summary row with maximum signed_del_ratio",
        "reports_stress_or_reserve_factors": False,
        "selected_channels": [selection_from_row(critical_row, critical=True)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Campaign run root with postprocess outputs.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output manifest path. Defaults to RUN_ROOT/{SIGNED_CHANNEL_SELECTION_DEFAULT}.",
    )
    parser.add_argument("--selection-id", default=None, help="Selection identifier. Defaults to '<run-root-name>-signed-channel-selection'.")
    parser.add_argument("--wohler-exponent", type=float, default=None, help="Override Wohler exponent written to the normalization block.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_path = (args.out or (run_root / SIGNED_CHANNEL_SELECTION_DEFAULT)).resolve()
    if out_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing manifest without --force: {out_path}")
    manifest = build_manifest(
        run_root,
        selection_id=args.selection_id,
        wohler_exponent=args.wohler_exponent,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
