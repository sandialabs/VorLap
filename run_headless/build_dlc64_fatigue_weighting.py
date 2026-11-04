#!/usr/bin/env python3
"""Build an IEC 61400-1 DLC 6.4 non-power-production weighting review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rayleigh_cdf(speed_mps: float, average_speed_mps: float) -> float:
    """IEC Rayleigh cumulative probability for a 10-minute mean wind speed."""
    return 1.0 - math.exp(-math.pi * (speed_mps / (2.0 * average_speed_mps)) ** 2)


def build_review(args: argparse.Namespace) -> dict:
    vref = float(args.vref)
    vave = float(args.vave) if args.vave is not None else 0.2 * vref
    availability = float(args.availability)
    annual_hours = float(args.annual_hours)
    edges = [float(value) for value in args.bin_upper_speed]
    values = {
        "vref": vref,
        "vave": vave,
        "annual_hours": annual_hours,
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if not math.isfinite(availability) or not 0.0 <= availability < 1.0:
        raise ValueError("--availability must be at least zero and below one.")
    if not edges or any(not math.isfinite(value) or value <= 0.0 for value in edges):
        raise ValueError("--bin-upper-speed values must be finite and positive.")
    if edges != sorted(set(edges)):
        raise ValueError("--bin-upper-speed values must be unique and strictly increasing.")
    dlc64_limit = 0.7 * vref
    if edges[-1] > dlc64_limit + 1.0e-12:
        raise ValueError(f"DLC 6.4 bin edges cannot exceed 0.7 Vref = {dlc64_limit:g} m/s.")

    bins = []
    lower = 0.0
    for upper in edges:
        bin_probability = rayleigh_cdf(upper, vave) - rayleigh_cdf(lower, vave)
        non_power_probability = (1.0 - availability) * bin_probability
        bins.append(
            {
                "representative_upper_speed_mps": upper,
                "lower_edge_mps": lower,
                "upper_edge_mps": upper,
                "rayleigh_probability": bin_probability,
                "downtime_probability": non_power_probability,
                "non_power_probability": non_power_probability,
                "expected_non_power_hours_per_year": non_power_probability * annual_hours,
            }
        )
        lower = upper

    total_probability = sum(row["non_power_probability"] for row in bins)
    for row in bins:
        row["conditional_non_power_weight"] = row["non_power_probability"] / total_probability
    report = {
        "schema_version": 2,
        "passed": True,
        "standard_basis": "IEC 61400-1 DLC 6.4 NTM below 0.7 Vref",
        "wind_distribution": "IEC standard-class Rayleigh distribution",
        "weighting_formula": (
            "The project-assumed non-power fraction is distributed over all wind speeds "
            "according to the IEC standard-class Rayleigh distribution."
        ),
        "study_assumption": (
            "Availability is a project input, not an IEC-prescribed value. The current candidate "
            "uses 0.99 availability: one 1-percent-of-year parked window that may occur at any "
            "wind speed, distributed according to the wind-speed probability."
        ),
        "vref_mps": vref,
        "vave_mps": vave,
        "vave_to_vref_ratio": vave / vref,
        "dlc64_max_speed_mps": dlc64_limit,
        "availability": availability,
        "annual_non_power_fraction": 1.0 - availability,
        "annual_hours": annual_hours,
        "expected_total_non_power_hours_per_year": (1.0 - availability) * annual_hours,
        "expected_total_non_power_days_per_year": (1.0 - availability) * annual_hours / 24.0,
        "bins": bins,
        "fatigue_weights_by_speed": {
            f"{row['representative_upper_speed_mps']:g}": row["non_power_probability"]
            for row in bins
        },
        "total_non_power_probability_through_dlc64_limit": total_probability,
        "expected_non_power_hours_per_year_through_dlc64_limit": total_probability * annual_hours,
    }
    report["review_sha256"] = sha256_payload(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--vref", type=float, default=50.0)
    parser.add_argument("--vave", type=float)
    parser.add_argument("--availability", type=float, default=0.99)
    parser.add_argument("--annual-hours", type=float, default=8760.0)
    parser.add_argument(
        "--bin-upper-speed",
        type=float,
        action="append",
        default=None,
        help="Upper edge and conservative representative speed for one DLC 6.4 bin.",
    )
    args = parser.parse_args()
    if args.bin_upper_speed is None:
        # Risk-informed upper representatives from screen_vorlap_modal_overlap.py.
        args.bin_upper_speed = [3.0, 7.0, 8.0, 8.75, 11.25, 13.0, 14.75, 18.0, 19.25, 21.5, 35.0]
    return args


def main() -> int:
    args = parse_args()
    try:
        report = build_review(args)
        if args.out is not None:
            write_json(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
