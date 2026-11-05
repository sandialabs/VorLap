#!/usr/bin/env python3
"""Build a reviewed modal/source-band/timestep qualification artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
VORLAP_ROOT = SCRIPT_DIR.parent
if str(VORLAP_ROOT) not in sys.path:
    sys.path.insert(0, str(VORLAP_ROOT))

from vorlap.dlc_manifest import sha256_file, sha256_payload, write_json  # noqa: E402


def source_frequency_from_csv(path: Path) -> tuple[float, int]:
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            raw = row.get("freq_max_hz")
            if raw in (None, ""):
                continue
            value = float(raw)
            if math.isfinite(value) and value > 0.0:
                values.append(value)
    if not values:
        raise ValueError(f"No finite positive freq_max_hz values found in {path}.")
    return max(values), len(values)


def provenance_path(path: Path) -> str:
    """Use a checkout-relative path when the evidence is tracked with VorLap."""
    try:
        return str(path.relative_to(VORLAP_ROOT))
    except ValueError:
        return str(path)


def modal_frequencies_from_campbell(path: Path, target_rpm: float, tolerance: float) -> tuple[list[float], float]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "rpm" not in rows[0]:
        raise ValueError(f"Campbell CSV has no RPM rows: {path}.")
    selected = min(rows, key=lambda row: abs(float(row["rpm"]) - target_rpm))
    selected_rpm = float(selected["rpm"])
    if abs(selected_rpm - target_rpm) > tolerance:
        raise ValueError(
            f"Campbell CSV has no RPM within {tolerance:g} of {target_rpm:g}; nearest is {selected_rpm:g}."
        )
    modal = [
        float(value)
        for name, value in selected.items()
        if name.startswith("mode_") and name.endswith("_hz") and value not in (None, "")
    ]
    if not modal:
        raise ValueError(f"Campbell CSV row at {selected_rpm:g} RPM contains no mode_*_hz values: {path}.")
    return modal, selected_rpm


def build_report(args: argparse.Namespace) -> dict:
    if args.campbell_csv is not None:
        campbell_path = args.campbell_csv.expanduser().resolve()
        modal, selected_rpm = modal_frequencies_from_campbell(
            campbell_path,
            float(args.campbell_rpm),
            float(args.campbell_rpm_tolerance),
        )
        modal_provenance = {
            "type": "owens_campbell_csv",
            "path": provenance_path(campbell_path),
            "sha256": sha256_file(campbell_path),
            "selected_rpm": selected_rpm,
        }
    else:
        modal = list(args.modal_frequency_hz or [])
        modal_provenance = {
            "type": "explicit_frequencies",
            "count": len(modal),
        }
    modal = sorted(set(float(value) for value in modal))
    if not modal or any(not math.isfinite(value) or value <= 0.0 for value in modal):
        raise ValueError("Supply finite positive --modal-frequency-hz values.")
    timestep = float(args.timestep)
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("--timestep must be finite and positive.")
    if args.source_coverage:
        source_paths = [path.expanduser().resolve() for path in args.source_coverage]
        if len(set(source_paths)) != len(source_paths):
            raise ValueError("Each --source-coverage CSV must be unique.")
        coverage_records = []
        for source_path in source_paths:
            file_max, file_rows = source_frequency_from_csv(source_path)
            coverage_records.append(
                {
                    "path": provenance_path(source_path),
                    "sha256": sha256_file(source_path),
                    "rows_with_frequency": file_rows,
                    "max_frequency_hz": file_max,
                }
            )
        source_max = max(float(record["max_frequency_hz"]) for record in coverage_records)
        source_rows = sum(int(record["rows_with_frequency"]) for record in coverage_records)
        if len(coverage_records) == 1:
            source_provenance = {
                "type": "source_coverage_csv",
                **coverage_records[0],
            }
        else:
            source_provenance = {
                "type": "source_coverage_csv_set",
                "files": coverage_records,
                "file_count": len(coverage_records),
                "rows_with_frequency": source_rows,
                "max_frequency_hz": source_max,
            }
    elif args.source_max_frequency_hz is not None:
        source_max = float(args.source_max_frequency_hz)
        source_provenance = {
            "type": "explicit_bound",
            "value_hz": source_max,
            "basis": args.source_frequency_basis,
        }
    else:
        raise ValueError("Supply --source-coverage or --source-max-frequency-hz.")
    if not math.isfinite(source_max) or source_max <= 0.0:
        raise ValueError("Source maximum frequency must be finite and positive.")

    modal_max = max(modal)
    nyquist = 0.5 / timestep
    modal_samples = 1.0 / (timestep * modal_max)
    source_samples = 1.0 / (timestep * source_max)
    nyquist_ratio = nyquist / source_max
    modal_band_ratio = modal_max / source_max
    issues: list[dict[str, str | float]] = []
    if len(modal) < args.min_modal_count:
        issues.append(
            {
                "code": "insufficient_modal_count",
                "message": f"Only {len(modal)} modes were reviewed; require at least {args.min_modal_count}.",
            }
        )
    if modal_samples < args.min_modal_samples_per_cycle:
        issues.append(
            {
                "code": "modal_timestep_underresolved",
                "message": (
                    f"Fastest reviewed mode has {modal_samples:.3g} samples/cycle; "
                    f"require {args.min_modal_samples_per_cycle:g}."
                ),
            }
        )
    if source_samples < args.min_source_samples_per_cycle:
        issues.append(
            {
                "code": "source_timestep_underresolved",
                "message": (
                    f"Maximum retained source frequency has {source_samples:.3g} samples/cycle; "
                    f"require {args.min_source_samples_per_cycle:g}."
                ),
            }
        )
    if nyquist_ratio < args.min_nyquist_ratio:
        issues.append(
            {
                "code": "source_nyquist_margin_too_small",
                "message": (
                    f"Nyquist/source ratio is {nyquist_ratio:.3g}; require {args.min_nyquist_ratio:g}."
                ),
            }
        )
    if modal_band_ratio < args.min_modal_band_ratio:
        issues.append(
            {
                "code": "modal_band_does_not_cover_source",
                "message": (
                    f"Highest reviewed mode/source maximum ratio is {modal_band_ratio:.3g}; "
                    f"require {args.min_modal_band_ratio:g}. Truncate the source band or review more modes."
                ),
            }
        )

    numerical_passed = not issues
    qualified = bool(args.approved and numerical_passed)
    report = {
        "schema_version": 1,
        "qualified": qualified,
        "numerical_checks_passed": numerical_passed,
        "human_approved": bool(args.approved),
        "modal_source": args.modal_source,
        "modal_method": args.modal_method,
        "modal_frequency_provenance": modal_provenance,
        "review_note": args.review_note,
        "modal_frequencies_hz": modal,
        "modal_frequency_count": len(modal),
        "source_frequency_provenance": source_provenance,
        "source_frequency_truncation": args.source_frequency_truncation,
        "timestep_s": timestep,
        "source_max_frequency_hz": source_max,
        "highest_reviewed_modal_frequency_hz": modal_max,
        "nyquist_frequency_hz": nyquist,
        "modal_samples_per_cycle_min": modal_samples,
        "source_samples_per_cycle_min": source_samples,
        "nyquist_to_source_ratio": nyquist_ratio,
        "modal_band_to_source_ratio": modal_band_ratio,
        "thresholds": {
            "min_modal_count": args.min_modal_count,
            "min_modal_samples_per_cycle": args.min_modal_samples_per_cycle,
            "min_source_samples_per_cycle": args.min_source_samples_per_cycle,
            "min_nyquist_ratio": args.min_nyquist_ratio,
            "min_modal_band_ratio": args.min_modal_band_ratio,
        },
        "recommended_max_timestep_s": min(
            1.0 / (args.min_modal_samples_per_cycle * modal_max),
            1.0 / (args.min_source_samples_per_cycle * source_max),
        ),
        "issues": issues,
    }
    report["qualification_sha256"] = sha256_payload(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    modal = parser.add_mutually_exclusive_group(required=True)
    modal.add_argument("--modal-frequency-hz", type=float, action="append")
    modal.add_argument("--campbell-csv", type=Path)
    parser.add_argument("--campbell-rpm", type=float, default=1.0)
    parser.add_argument("--campbell-rpm-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--modal-source", required=True)
    parser.add_argument("--modal-method", required=True)
    parser.add_argument("--review-note", default="")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-coverage",
        type=Path,
        action="append",
        help="Source-coverage CSV; repeat to qualify against multiple returned pilot records.",
    )
    source.add_argument("--source-max-frequency-hz", type=float)
    parser.add_argument("--source-frequency-basis", default="")
    parser.add_argument(
        "--source-frequency-truncation",
        choices=("none", "nyquist"),
        default="none",
        help="Source-frequency policy used by the pilot that supplied the retained frequency bound.",
    )
    parser.add_argument("--timestep", type=float, default=0.0075)
    parser.add_argument("--min-modal-count", type=int, default=5)
    parser.add_argument("--min-modal-samples-per-cycle", type=float, default=20.0)
    parser.add_argument("--min-source-samples-per-cycle", type=float, default=10.0)
    parser.add_argument("--min-nyquist-ratio", type=float, default=1.25)
    parser.add_argument("--min-modal-band-ratio", type=float, default=1.0)
    parser.add_argument(
        "--approved",
        action="store_true",
        help="Mark qualified only when all numerical checks pass and the modal/source basis has been reviewed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
        write_json(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["qualified"] or not args.approved else 2
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
