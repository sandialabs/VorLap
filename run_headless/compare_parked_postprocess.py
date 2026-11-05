#!/usr/bin/env python3
"""Compare literal raw-file postprocessing with corrected campaign artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ComparisonError(ValueError):
    """Postprocess artifacts cannot be compared safely."""


FILE_SPECS = {
    "signed_del_comparison.csv": {
        "keys": [
            "case",
            "wind_speed_mps",
            "azimuth_deg",
            "position",
            "quantity",
            "axis",
            "unit",
            "channel_baseline",
            "channel_vorlap",
        ],
        "ignore": set(),
        "allowed_candidate_only": set(),
    },
    "extreme_channel_metrics.csv": {
        "keys": [
            "scenario",
            "case",
            "wind_speed_mps",
            "azimuth_deg",
            "blade",
            "position",
            "quantity",
            "axis",
            "unit",
            "channel",
        ],
        "ignore": {"result_file"},
        "allowed_candidate_only": set(),
    },
    "source_coverage.csv": {
        "keys": ["scenario", "case", "component"],
        "ignore": {"summary_file"},
        "allowed_candidate_only": {
            "airfoil_id",
            "reynolds_model",
            "re_lower_bound_warning_count",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_key(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value)


def keyed_rows(table: pd.DataFrame, keys: list[str], path: Path) -> dict[tuple[Any, ...], int]:
    missing = set(keys) - set(table.columns)
    if missing:
        raise ComparisonError(f"{path} is missing key columns: {', '.join(sorted(missing))}")
    result: dict[tuple[Any, ...], int] = {}
    for index, row in table.iterrows():
        key = tuple(normalized_key(row[name]) for name in keys)
        if key in result:
            raise ComparisonError(f"{path} has a duplicate comparison key: {key}")
        result[key] = int(index)
    return result


def compare_column(
    reference: pd.Series,
    candidate: pd.Series,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    numeric = pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(candidate)
    if numeric and not pd.api.types.is_bool_dtype(reference):
        ref = pd.to_numeric(reference, errors="raise").to_numpy(dtype=float)
        cand = pd.to_numeric(candidate, errors="raise").to_numpy(dtype=float)
        finite = np.isfinite(ref) & np.isfinite(cand)
        same_nonfinite = (~np.isfinite(ref)) & (~np.isfinite(cand)) & (
            np.isnan(ref) == np.isnan(cand)
        )
        close = np.zeros(len(ref), dtype=bool)
        close[finite] = np.isclose(ref[finite], cand[finite], rtol=rtol, atol=atol)
        close[~finite] = same_nonfinite[~finite]
        absolute = np.abs(cand[finite] - ref[finite])
        scale = np.maximum.reduce(
            [np.abs(ref[finite]), np.abs(cand[finite]), np.full(np.count_nonzero(finite), 1.0e-300)]
        )
        relative = absolute / scale
        return {
            "comparison_type": "numeric",
            "compared_count": len(ref),
            "mismatch_count": int(np.count_nonzero(~close)),
            "max_absolute_error": float(np.max(absolute)) if len(absolute) else 0.0,
            "max_relative_error": float(np.max(relative)) if len(relative) else 0.0,
            "passed": bool(np.all(close)),
        }
    ref_text = reference.map(lambda value: "" if pd.isna(value) else str(value))
    cand_text = candidate.map(lambda value: "" if pd.isna(value) else str(value))
    equal = ref_text.to_numpy() == cand_text.to_numpy()
    return {
        "comparison_type": "text",
        "compared_count": len(reference),
        "mismatch_count": int(np.count_nonzero(~equal)),
        "max_absolute_error": math.nan,
        "max_relative_error": math.nan,
        "passed": bool(np.all(equal)),
    }


def compare_artifact(
    reference_path: Path,
    candidate_path: Path,
    keys: list[str],
    ignored: set[str],
    allowed_candidate_only: set[str] | None = None,
    *,
    row_id: str,
    rtol: float,
    atol: float,
) -> list[dict[str, Any]]:
    if not reference_path.is_file() or not candidate_path.is_file():
        raise ComparisonError(
            f"Missing comparison input: {reference_path} or {candidate_path}"
        )
    reference = pd.read_csv(reference_path)
    candidate = pd.read_csv(candidate_path)
    ref_rows = keyed_rows(reference, keys, reference_path)
    cand_rows = keyed_rows(candidate, keys, candidate_path)
    if set(ref_rows) != set(cand_rows):
        raise ComparisonError(
            f"Comparison key coverage differs for {row_id} {reference_path.name}: "
            f"reference={len(ref_rows)}, candidate={len(cand_rows)}"
        )
    ordered_keys = sorted(ref_rows, key=lambda key: tuple(str(value) for value in key))
    ref_order = [ref_rows[key] for key in ordered_keys]
    cand_order = [cand_rows[key] for key in ordered_keys]
    comparable = (set(reference.columns) & set(candidate.columns)) - set(keys) - ignored
    allowed_candidate_only = allowed_candidate_only or set()
    reference_only = set(reference.columns) - set(candidate.columns) - ignored
    unexpected_candidate_only = (
        set(candidate.columns) - set(reference.columns) - ignored - allowed_candidate_only
    )
    missing_columns = reference_only | unexpected_candidate_only
    if missing_columns:
        raise ComparisonError(
            f"Comparison schemas differ for {row_id} {reference_path.name}: "
            + ", ".join(sorted(missing_columns))
        )
    rows: list[dict[str, Any]] = []
    for column in sorted(comparable):
        result = compare_column(
            reference.loc[ref_order, column].reset_index(drop=True),
            candidate.loc[cand_order, column].reset_index(drop=True),
            rtol=rtol,
            atol=atol,
        )
        rows.append(
            {
                "row_id": row_id,
                "artifact": reference_path.name,
                "column": column,
                **result,
            }
        )
    return rows


def validate_raw_manifest(
    path: Path,
    equivalent_cycles: float,
    wohler_exponent: float,
) -> tuple[int, int]:
    if not path.is_file():
        raise ComparisonError(f"Raw postprocess manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True or payload.get("source_mode") != "raw_qblade_exports":
        raise ComparisonError(f"Raw postprocess provenance did not pass: {path}")
    for name, expected in (
        ("equivalent_cycles", equivalent_cycles),
        ("wohler_exponent", wohler_exponent),
    ):
        if not math.isclose(float(payload.get(name, math.nan)), expected, abs_tol=1.0e-12):
            raise ComparisonError(f"Raw postprocess {name} differs in {path}")
    inputs = payload.get("inputs")
    outputs = payload.get("outputs")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise ComparisonError(f"Raw postprocess hash maps are missing: {path}")
    input_bytes = 0
    for record in inputs.values():
        input_path = Path(str(record.get("path", ""))) if isinstance(record, dict) else Path()
        expected = record.get("sha256") if isinstance(record, dict) else None
        if not input_path.is_file() or sha256_file(input_path) != expected:
            raise ComparisonError(f"Raw input hash mismatch: {input_path}")
        input_bytes += input_path.stat().st_size
    for name in FILE_SPECS:
        record = outputs.get(name)
        output_path = path.parent / name
        expected = record.get("sha256") if isinstance(record, dict) else None
        if not output_path.is_file() or sha256_file(output_path) != expected:
            raise ComparisonError(f"Raw output hash mismatch: {output_path}")
    return len(inputs), input_bytes


def compare_campaign(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "approved":
        raise ComparisonError(f"Manifest is not approved: {manifest_path}")
    rows = manifest.get("rows", [])
    if int(manifest.get("row_count", -1)) != len(rows):
        raise ComparisonError(f"Manifest row count is inconsistent: {manifest_path}")
    campaign_root = args.campaign_root.expanduser().resolve()
    details: list[dict[str, Any]] = []
    raw_file_count = 0
    raw_bytes = 0
    for row in rows:
        row_id = str(row["row_id"])
        run_root = campaign_root / Path(str(row["run_root"])).name
        reference_dir = run_root / args.reference_dir_name
        candidate_dir = run_root / args.candidate_dir_name
        count, byte_count = validate_raw_manifest(
            candidate_dir / "postprocess_manifest.json",
            args.equivalent_cycles,
            args.wohler_exponent,
        )
        raw_file_count += count
        raw_bytes += byte_count
        for name, spec in FILE_SPECS.items():
            details.extend(
                compare_artifact(
                    reference_dir / name,
                    candidate_dir / name,
                    spec["keys"],
                    spec["ignore"],
                    spec["allowed_candidate_only"],
                    row_id=row_id,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            )
    detail_path = args.out_csv.expanduser().resolve()
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(details).to_csv(detail_path, index=False, lineterminator="\n")
    numeric = [row for row in details if row["comparison_type"] == "numeric"]
    summary = {
        "schema_version": 1,
        "passed": all(row["passed"] for row in details),
        "method": "keyed_raw_rerun_vs_exact_sufficient_statistic_reconstruction",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "campaign_row_count": len(rows),
        "raw_input_file_count": raw_file_count,
        "raw_input_bytes": raw_bytes,
        "reference_dir_name": args.reference_dir_name,
        "candidate_dir_name": args.candidate_dir_name,
        "wohler_exponent": args.wohler_exponent,
        "equivalent_cycles": args.equivalent_cycles,
        "rtol": args.rtol,
        "atol": args.atol,
        "column_comparison_count": len(details),
        "value_comparison_count": sum(int(row["compared_count"]) for row in details),
        "mismatch_count": sum(int(row["mismatch_count"]) for row in details),
        "max_absolute_error": max(
            (float(row["max_absolute_error"]) for row in numeric),
            default=0.0,
        ),
        "max_relative_error": max(
            (float(row["max_relative_error"]) for row in numeric),
            default=0.0,
        ),
        "detail_csv": str(detail_path),
        "detail_csv_sha256": sha256_file(detail_path),
    }
    summary_path = args.out_json.expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--reference-dir-name", default="postprocess_neq600")
    parser.add_argument("--candidate-dir-name", default="postprocess_raw_neq600")
    parser.add_argument("--wohler-exponent", type=float, default=10.0)
    parser.add_argument("--equivalent-cycles", type=float, default=600.0)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = compare_campaign(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["passed"] else 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
