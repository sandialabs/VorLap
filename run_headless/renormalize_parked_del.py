#!/usr/bin/env python3
"""Recover fixed-reference DELs from legacy cycle-normalized postprocessing.

The legacy postprocessor stored both

    L_old = (damage / observed_rainflow_cycles) ** (1 / m)

and the observed rainflow-cycle count.  Those sufficient statistics recover
the exact DEL for a fixed reference count without the raw time series:

    L_ref = L_old * (observed_rainflow_cycles / N_eq) ** (1 / m).

This script preserves the source postprocess directory and writes only the
three inputs required by campaign aggregation plus a hash-audited correction
manifest in a separate directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

class RenormalizationError(ValueError):
    """A legacy postprocess artifact cannot be corrected safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_reference_del(
    legacy_del: float,
    observed_cycles: float,
    wohler_exponent: float,
    equivalent_cycles: float,
) -> float:
    values = (legacy_del, observed_cycles, wohler_exponent, equivalent_cycles)
    if not all(math.isfinite(value) for value in values):
        raise RenormalizationError("DEL correction inputs must be finite.")
    if legacy_del < 0.0 or observed_cycles < 0.0:
        raise RenormalizationError("DEL and observed cycles must be nonnegative.")
    if wohler_exponent <= 0.0 or equivalent_cycles <= 0.0:
        raise RenormalizationError("Wohler exponent and equivalent cycles must be positive.")
    if legacy_del == 0.0 or observed_cycles == 0.0:
        return 0.0
    return legacy_del * (observed_cycles / equivalent_cycles) ** (
        1.0 / wohler_exponent
    )


def correction_error(
    legacy_del: float,
    corrected_del: float,
    observed_cycles: float,
    wohler_exponent: float,
    equivalent_cycles: float,
) -> float:
    legacy_damage = legacy_del**wohler_exponent * observed_cycles
    corrected_damage = corrected_del**wohler_exponent * equivalent_cycles
    scale = max(abs(legacy_damage), abs(corrected_damage), 1.0)
    return abs(corrected_damage - legacy_damage) / scale


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RenormalizationError(f"Required source artifact is missing: {path}")
    table = pd.read_csv(path)
    if table.empty:
        raise RenormalizationError(f"Required source artifact is empty: {path}")
    return table


def write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n")


def renormalize_signed_table(
    table: pd.DataFrame,
    wohler_exponent: float,
    equivalent_cycles: float,
) -> tuple[pd.DataFrame, float]:
    required = {
        "signed_del_baseline",
        "signed_del_vorlap",
        "rainflow_cycles_baseline",
        "rainflow_cycles_vorlap",
    }
    missing = required - set(table.columns)
    if missing:
        raise RenormalizationError(
            "Legacy signed DEL table is missing: " + ", ".join(sorted(missing))
        )
    corrected = table.copy()
    max_error = 0.0
    for scenario in ("baseline", "vorlap"):
        del_name = f"signed_del_{scenario}"
        cycle_name = f"rainflow_cycles_{scenario}"
        values: list[float] = []
        for legacy, cycles in zip(
            pd.to_numeric(table[del_name], errors="raise"),
            pd.to_numeric(table[cycle_name], errors="raise"),
        ):
            value = fixed_reference_del(
                float(legacy),
                float(cycles),
                wohler_exponent,
                equivalent_cycles,
            )
            max_error = max(
                max_error,
                correction_error(
                    float(legacy),
                    value,
                    float(cycles),
                    wohler_exponent,
                    equivalent_cycles,
                ),
            )
            values.append(value)
        corrected[del_name] = values
        corrected[f"equivalent_cycles_{scenario}"] = equivalent_cycles
    baseline = pd.to_numeric(corrected["signed_del_baseline"], errors="raise")
    vorlap = pd.to_numeric(corrected["signed_del_vorlap"], errors="raise")
    corrected["signed_delta_pct"] = [
        100.0 * (coupled - base) / base if base > 0.0 else math.nan
        for base, coupled in zip(baseline, vorlap)
    ]
    return corrected, max_error


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def renormalize_row(
    run_root: Path,
    source_dir_name: str,
    output_dir_name: str,
    wohler_exponent: float,
    equivalent_cycles: float,
) -> dict[str, Any]:
    source_dir = run_root / source_dir_name
    output_dir = run_root / output_dir_name
    signed_source = source_dir / "signed_del_comparison.csv"
    extreme_source = source_dir / "extreme_channel_metrics.csv"
    coverage_source = source_dir / "source_coverage.csv"

    signed, max_error = renormalize_signed_table(
        read_csv(signed_source),
        wohler_exponent,
        equivalent_cycles,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    signed_output = output_dir / signed_source.name
    extreme_output = output_dir / extreme_source.name
    coverage_output = output_dir / coverage_source.name
    write_csv(signed, signed_output)
    write_csv(read_csv(extreme_source), extreme_output)
    write_csv(read_csv(coverage_source), coverage_output)

    inputs = {
        path.name: artifact_record(path)
        for path in (signed_source, extreme_source, coverage_source)
    }
    outputs = {
        path.name: artifact_record(path)
        for path in (signed_output, extreme_output, coverage_output)
    }
    report = {
        "schema_version": 1,
        "passed": max_error <= 5.0e-12,
        "method": "exact_sufficient_statistic_renormalization",
        "formula": "L_ref=L_old*(observed_cycles/equivalent_cycles)^(1/m)",
        "run_root": str(run_root),
        "source_dir_name": source_dir_name,
        "output_dir_name": output_dir_name,
        "wohler_exponent": wohler_exponent,
        "equivalent_cycles": equivalent_cycles,
        "signed_row_count": len(signed),
        "max_relative_damage_reconstruction_error": max_error,
        "inputs": inputs,
        "outputs": outputs,
    }
    report_path = output_dir / "renormalization_manifest.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise RenormalizationError(
            f"DEL damage reconstruction tolerance failed: {report_path}"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-dir-name", default="postprocess")
    parser.add_argument("--output-dir-name", default="postprocess_neq600")
    parser.add_argument("--wohler-exponent", type=float, default=10.0)
    parser.add_argument("--equivalent-cycles", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_path = args.manifest.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "approved":
            raise RenormalizationError(
                f"Manifest is not approved: {manifest_path}"
            )
        if int(manifest.get("row_count", -1)) != len(manifest.get("rows", [])):
            raise RenormalizationError(
                f"Manifest row count is inconsistent: {manifest_path}"
            )
        campaign_root = args.campaign_root.expanduser().resolve()
        reports = []
        for row in manifest["rows"]:
            local_root = campaign_root / Path(str(row["run_root"])).name
            reports.append(
                renormalize_row(
                    local_root,
                    args.source_dir_name,
                    args.output_dir_name,
                    args.wohler_exponent,
                    args.equivalent_cycles,
                )
            )
        result = {
            "passed": all(report["passed"] for report in reports),
            "row_count": len(reports),
            "wohler_exponent": args.wohler_exponent,
            "equivalent_cycles": args.equivalent_cycles,
            "maximum_relative_damage_reconstruction_error": max(
                report["max_relative_damage_reconstruction_error"]
                for report in reports
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 2
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
