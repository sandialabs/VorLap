#!/usr/bin/env python3
"""Aggregate completed sampled-DLC rows into weighted fatigue and ultimate CSVs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
VORLAP_ROOT = SCRIPT_DIR.parent
if str(VORLAP_ROOT) not in sys.path:
    sys.path.insert(0, str(VORLAP_ROOT))

from vorlap.dlc_manifest import (  # noqa: E402
    DLCManifestError,
    load_manifest,
    sha256_file,
    validate_wind_record,
    write_json,
)
from vorlap.execution_provenance import validate_execution_provenance  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DLCManifestError(f"Required aggregation input is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise DLCManifestError(f"Required aggregation input is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DLCManifestError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise DLCManifestError(f"{name} must be finite.")
    return result


def false_text(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "0", "false", "no"}


def archive_matches(run_root: Path, manifest: dict, row: dict, index: int) -> None:
    path = run_root / "dlc_manifest_row.json"
    if not path.is_file():
        raise DLCManifestError(f"Selected-row archive is missing: {path}")
    archive = json.loads(path.read_text(encoding="utf-8"))
    if (
        archive.get("manifest_sha256") != manifest["manifest_sha256"]
        or archive.get("row_index") != index
        or archive.get("row") != row
    ):
        raise DLCManifestError(f"Selected-row archive does not match approved row {row['row_id']}.")


def result_audit_passes(run_root: Path) -> dict[str, Any]:
    path = run_root / "audits" / "results_audit.json"
    if not path.is_file():
        raise DLCManifestError(f"Results audit is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise DLCManifestError(f"Results audit did not pass: {path}")
    return payload


def execution_provenance_passes(
    run_root: Path,
    manifest: dict,
    row: dict,
    index: int,
) -> None:
    for phase in ("wind", "qblade"):
        path = run_root / f"execution_environment_{phase}.json"
        if not path.is_file():
            raise DLCManifestError(f"{phase} execution provenance is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_execution_provenance(
            payload,
            manifest=manifest,
            row=row,
            row_index=index,
            phase=phase,
            require_runtime_artifacts=True,
        )


def validate_derived_postprocess(
    postprocess_dir: Path,
    output_paths: tuple[Path, ...],
    equivalent_cycles: float,
    wohler_exponent: float,
) -> str:
    raw_manifest_path = postprocess_dir / "postprocess_manifest.json"
    correction_path = postprocess_dir / "renormalization_manifest.json"
    if raw_manifest_path.is_file():
        provenance_path = raw_manifest_path
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        mode = str(provenance.get("source_mode", ""))
        if mode != "raw_qblade_exports":
            raise DLCManifestError(
                f"Unsupported raw postprocess source mode {mode!r}: {provenance_path}"
            )
        inputs = provenance.get("inputs")
        if not isinstance(inputs, dict) or not inputs:
            raise DLCManifestError(
                f"Raw postprocess has no input hash map: {provenance_path}"
            )
        for record in inputs.values():
            input_path = Path(str(record.get("path", ""))) if isinstance(record, dict) else Path()
            expected = record.get("sha256") if isinstance(record, dict) else None
            if not input_path.is_file() or expected != sha256_file(input_path):
                raise DLCManifestError(
                    f"Raw postprocess input hash mismatch for {input_path}."
                )
        exponent = as_float(
            provenance.get("wohler_exponent"),
            "raw postprocess wohler_exponent",
        )
        if not math.isclose(exponent, wohler_exponent, abs_tol=1.0e-12):
            raise DLCManifestError(
                f"Raw postprocess uses the wrong Wohler exponent: {provenance_path}"
            )
    elif correction_path.is_file():
        provenance_path = correction_path
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        mode = "exact_sufficient_statistic_renormalization"
    else:
        raise DLCManifestError(
            "Derived postprocess directory has neither postprocess_manifest.json nor "
            f"renormalization_manifest.json: {postprocess_dir}"
        )
    if provenance.get("passed") is not True:
        raise DLCManifestError(f"Derived postprocess did not pass: {provenance_path}")
    observed_cycles = as_float(
        provenance.get("equivalent_cycles"),
        "derived postprocess equivalent_cycles",
    )
    if not math.isclose(observed_cycles, equivalent_cycles, abs_tol=1.0e-9):
        raise DLCManifestError(
            f"Derived postprocess uses the wrong equivalent-cycle count: {provenance_path}"
        )
    outputs = provenance.get("outputs", {})
    if not isinstance(outputs, dict):
        raise DLCManifestError(
            f"Derived postprocess has no output hash map: {provenance_path}"
        )
    for path in output_paths:
        record = outputs.get(path.name)
        expected = record.get("sha256") if isinstance(record, dict) else None
        if expected != sha256_file(path):
            raise DLCManifestError(f"Derived postprocess hash mismatch for {path}.")
    return mode


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    if args.offline_derived_mirror:
        if args.campaign_root is None or args.postprocess_dir_name == "postprocess":
            raise DLCManifestError(
                "--offline-derived-mirror requires --campaign-root and a "
                "separate corrected --postprocess-dir-name."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "approved":
            raise DLCManifestError("Offline derived aggregation requires an approved manifest.")
        if int(manifest.get("row_count", -1)) != len(manifest.get("rows", [])):
            raise DLCManifestError("Offline manifest row count is inconsistent.")
    else:
        manifest = load_manifest(manifest_path, require_approved=True)
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    signed_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    audit_inventory_rows: list[dict[str, Any]] = []
    postprocess_provenance_modes: set[str] = set()
    equivalent_cycles = as_float(args.equivalent_cycles, "--equivalent-cycles")
    if equivalent_cycles <= 0.0:
        raise DLCManifestError("--equivalent-cycles must be positive.")
    exponent = float(args.wohler_exponent)
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise DLCManifestError("--wohler-exponent must be finite and positive.")

    for index, row in enumerate(manifest["rows"]):
        run_root = (
            Path(str(row["run_root"]))
            if args.campaign_root is None
            else args.campaign_root.expanduser().resolve()
            / Path(str(row["run_root"])).name
        )
        postprocess_dir = run_root / args.postprocess_dir_name
        archive_matches(run_root, manifest, row, index)
        if not args.offline_derived_mirror:
            validate_wind_record(row, manifest)
            execution_provenance_passes(run_root, manifest, row, index)
        audit_payload = result_audit_passes(run_root)
        issues = audit_payload.get("issues", [])
        if not isinstance(issues, list):
            raise DLCManifestError(
                f"Results audit issues must be a list: {run_root}"
            )
        warnings = [
            issue for issue in issues
            if isinstance(issue, dict)
            and str(issue.get("severity", "")).lower() == "warning"
        ]
        source_rows = read_csv(postprocess_dir / "source_coverage.csv")
        lower_re_warning_samples = sum(
            int(round(as_float(
                source.get("re_lower_bound_warning_count", 0) or 0,
                "source lower-Re warning count",
            )))
            for source in source_rows
        )
        source_re_min = min(
            as_float(source.get("re_min"), "source coverage re_min")
            for source in source_rows
        )
        recovery_records = sorted(
            path
            for directory in (run_root, run_root / "audits")
            if directory.is_dir()
            for path in directory.glob("*provenance*recovery*.json")
        )
        audit_inventory_rows.append(
            {
                "row_id": row["row_id"],
                "dlc": row["dlc"],
                "seed": row["seed"],
                "wind_speed_mps": row["wind_speed_mps"],
                "result_audit_passed": True,
                "result_audit_sha256": sha256_file(
                    run_root / "audits" / "results_audit.json"
                ),
                "warning_count": len(warnings),
                "warning_codes": ";".join(
                    sorted(str(issue.get("code", "")) for issue in warnings)
                ),
                "warning_messages": " | ".join(
                    str(issue.get("message", "")) for issue in warnings
                ),
                "source_re_min": source_re_min,
                "source_re_lower_bound_warning_samples": (
                    lower_re_warning_samples
                ),
                "provenance_recovery_record_count": len(recovery_records),
                "provenance_recovery_records": ";".join(
                    str(path.relative_to(run_root)) for path in recovery_records
                ),
            }
        )
        signed_path = postprocess_dir / "signed_del_comparison.csv"
        extreme_path = postprocess_dir / "extreme_channel_metrics.csv"
        if args.postprocess_dir_name != "postprocess":
            postprocess_provenance_modes.add(
                validate_derived_postprocess(
                    postprocess_dir,
                    (signed_path, extreme_path, postprocess_dir / "source_coverage.csv"),
                    equivalent_cycles,
                    exponent,
                )
            )
        for source in read_csv(signed_path):
            if not false_text(source.get("smoke_only")):
                raise DLCManifestError(f"Smoke-only DEL row found in production input: {signed_path}")
            if str(source.get("postprocess_warnings", "") or "").strip():
                raise DLCManifestError(f"Warning-bearing DEL row found in production input: {signed_path}")
            speed = as_float(source.get("wind_speed_mps"), "signed DEL wind_speed_mps")
            if not math.isclose(speed, float(row["wind_speed_mps"]), abs_tol=1.0e-9):
                raise DLCManifestError(f"Signed DEL speed does not match manifest row {row['row_id']}.")
            for name in (
                "equivalent_cycles_baseline",
                "equivalent_cycles_vorlap",
            ):
                value = as_float(source.get(name), f"signed DEL {name}")
                if not math.isclose(value, equivalent_cycles, abs_tol=1.0e-9):
                    raise DLCManifestError(
                        f"Signed DEL {name}={value:g} does not match the frozen "
                        f"equivalent-cycle count {equivalent_cycles:g}."
                    )
            enriched = dict(source)
            enriched.update(
                {
                    "row_id": row["row_id"],
                    "dlc": row["dlc"],
                    "seed": row["seed"],
                    "fatigue_weight": row["fatigue_weight"],
                }
            )
            signed_rows.append(enriched)
        for source in read_csv(extreme_path):
            if not false_text(source.get("smoke_only")):
                raise DLCManifestError(f"Smoke-only extreme row found in production input: {extreme_path}")
            if str(source.get("postprocess_warnings", "") or "").strip():
                raise DLCManifestError(f"Warning-bearing extreme row found in production input: {extreme_path}")
            speed = as_float(source.get("wind_speed_mps"), "extreme wind_speed_mps")
            if not math.isclose(speed, float(row["wind_speed_mps"]), abs_tol=1.0e-9):
                raise DLCManifestError(f"Extreme speed does not match manifest row {row['row_id']}.")
            enriched = dict(source)
            enriched.update({"row_id": row["row_id"], "dlc": row["dlc"], "seed": row["seed"]})
            extreme_rows.append(enriched)

    if not signed_rows or not extreme_rows:
        raise DLCManifestError("No production fatigue or ultimate rows were collected.")

    signed_fields = ["row_id", "dlc", "seed", "fatigue_weight"] + [
        name for name in signed_rows[0] if name not in {"row_id", "dlc", "seed", "fatigue_weight"}
    ]
    extreme_fields = ["row_id", "dlc", "seed"] + [
        name for name in extreme_rows[0] if name not in {"row_id", "dlc", "seed"}
    ]
    write_csv(output_dir / "signed_del_rows.csv", signed_rows, signed_fields)
    write_csv(output_dir / "extreme_channel_rows.csv", extreme_rows, extreme_fields)
    write_csv(
        output_dir / "campaign_audit_inventory.csv",
        audit_inventory_rows,
        list(audit_inventory_rows[0]),
    )

    channel_fields = (
        "position",
        "quantity",
        "axis",
        "unit",
        "channel_baseline",
        "channel_vorlap",
    )
    by_channel_condition: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in signed_rows:
        if str(row["dlc"]) != "6.4":
            continue
        key = tuple(row.get(name, "") for name in channel_fields) + (as_float(row["wind_speed_mps"], "wind speed"),)
        by_channel_condition[key].append(row)

    condition_rows: list[dict[str, Any]] = []
    by_channel: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for key, rows in sorted(by_channel_condition.items(), key=lambda item: tuple(map(str, item[0]))):
        channel_key = key[:-1]
        speed = float(key[-1])
        seeds = {int(row["seed"]) for row in rows}
        expected_seeds = {
            int(manifest_row["seed"])
            for manifest_row in manifest["rows"]
            if manifest_row["dlc"] == "6.4"
            and math.isclose(float(manifest_row["wind_speed_mps"]), speed, abs_tol=1.0e-9)
        }
        if seeds != expected_seeds:
            raise DLCManifestError(f"Fatigue channel at U={speed:g} does not cover every approved seed.")
        weights = {as_float(row["fatigue_weight"], "fatigue_weight") for row in rows}
        if len(weights) != 1:
            raise DLCManifestError(f"Fatigue weights differ across U={speed:g} seed rows.")
        baseline_values = [as_float(row["signed_del_baseline"], "signed_del_baseline") for row in rows]
        vorlap_values = [as_float(row["signed_del_vorlap"], "signed_del_vorlap") for row in rows]
        if any(value < 0.0 for value in baseline_values + vorlap_values):
            raise DLCManifestError("Signed-channel DEL magnitudes must be nonnegative.")
        paired_ratios = [
            vorlap / baseline if baseline > 0.0 else math.nan
            for baseline, vorlap in zip(baseline_values, vorlap_values)
        ]
        finite_paired_ratios = [value for value in paired_ratios if math.isfinite(value)]
        baseline_seed_del = (sum(value**exponent for value in baseline_values) / len(rows)) ** (1.0 / exponent)
        vorlap_seed_del = (sum(value**exponent for value in vorlap_values) / len(rows)) ** (1.0 / exponent)
        condition = {
            **dict(zip(channel_fields, channel_key)),
            "wind_speed_mps": speed,
            "seed_count": len(rows),
            "fatigue_weight": next(iter(weights)),
            "equivalent_cycles": equivalent_cycles,
            "signed_del_baseline_seed_aggregate": baseline_seed_del,
            "signed_del_vorlap_seed_aggregate": vorlap_seed_del,
            "signed_del_baseline_seed_min": min(baseline_values),
            "signed_del_baseline_seed_max": max(baseline_values),
            "signed_del_vorlap_seed_min": min(vorlap_values),
            "signed_del_vorlap_seed_max": max(vorlap_values),
            "paired_ratio_seed_mean": (
                statistics.fmean(finite_paired_ratios) if finite_paired_ratios else math.nan
            ),
            "paired_ratio_finite_seed_count": len(finite_paired_ratios),
            "paired_ratio_seed_min": (
                min(finite_paired_ratios) if finite_paired_ratios else math.nan
            ),
            "paired_ratio_seed_max": (
                max(finite_paired_ratios) if finite_paired_ratios else math.nan
            ),
            "paired_ratio_seed_sample_std": (
                statistics.stdev(finite_paired_ratios)
                if len(finite_paired_ratios) > 1
                else math.nan
            ),
            "ratio_vorlap_to_baseline": (
                vorlap_seed_del / baseline_seed_del if baseline_seed_del > 0.0 else math.nan
            ),
        }
        condition_rows.append(condition)
        by_channel[channel_key].append(condition)

    weighted_rows: list[dict[str, Any]] = []
    for channel_key, rows in sorted(by_channel.items(), key=lambda item: tuple(map(str, item[0]))):
        weight_sum = sum(float(row["fatigue_weight"]) for row in rows)
        if weight_sum <= 0.0:
            raise DLCManifestError("Fatigue weights must sum to a positive value.")
        baseline = (
            sum(
                float(row["fatigue_weight"])
                * float(row["signed_del_baseline_seed_aggregate"]) ** exponent
                for row in rows
            )
            / weight_sum
        ) ** (1.0 / exponent)
        vorlap = (
            sum(
                float(row["fatigue_weight"])
                * float(row["signed_del_vorlap_seed_aggregate"]) ** exponent
                for row in rows
            )
            / weight_sum
        ) ** (1.0 / exponent)
        weighted_rows.append(
            {
                **dict(zip(channel_fields, channel_key)),
                "wind_speed_bin_count": len(rows),
                "weight_sum": weight_sum,
                "wohler_exponent": exponent,
                "equivalent_cycles": equivalent_cycles,
                "signed_del_baseline_weighted": baseline,
                "signed_del_vorlap_weighted": vorlap,
                "ratio_vorlap_to_baseline": vorlap / baseline if baseline > 0.0 else math.nan,
            }
        )

    if not weighted_rows:
        raise DLCManifestError("No DLC 6.4 fatigue rows were available for weighted aggregation.")
    condition_fields = list(condition_rows[0])
    weighted_fields = list(weighted_rows[0])
    write_csv(output_dir / "fatigue_seed_aggregates.csv", condition_rows, condition_fields)
    write_csv(output_dir / "fatigue_weighted_signed_del.csv", weighted_rows, weighted_fields)

    ultimate_by_channel: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in extreme_rows:
        key = tuple(str(row.get(name, "")) for name in ("scenario", "blade", "position", "quantity", "axis", "unit", "channel"))
        value = as_float(row.get("max_abs"), "extreme max_abs")
        current = ultimate_by_channel.get(key)
        if current is None or value > float(current["max_abs"]):
            ultimate_by_channel[key] = {
                "scenario": key[0],
                "blade": key[1],
                "position": key[2],
                "quantity": key[3],
                "axis": key[4],
                "unit": key[5],
                "channel": key[6],
                "max_abs": value,
                "critical_row_id": row["row_id"],
                "critical_dlc": row["dlc"],
                "critical_seed": row["seed"],
                "critical_wind_speed_mps": row["wind_speed_mps"],
                "critical_case": row.get("case", ""),
            }
    ultimate_rows = list(ultimate_by_channel.values())
    write_csv(output_dir / "ultimate_envelope.csv", ultimate_rows, list(ultimate_rows[0]))

    artifacts = {}
    for name in (
        "signed_del_rows.csv",
        "extreme_channel_rows.csv",
        "fatigue_seed_aggregates.csv",
        "fatigue_weighted_signed_del.csv",
        "ultimate_envelope.csv",
        "campaign_audit_inventory.csv",
    ):
        path = output_dir / name
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    summary = {
        "schema_version": 1,
        "passed": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "row_count": len(manifest["rows"]),
        "signed_del_input_rows": len(signed_rows),
        "extreme_input_rows": len(extreme_rows),
        "fatigue_channel_count": len(weighted_rows),
        "ultimate_channel_count": len(ultimate_rows),
        "wohler_exponent": exponent,
        "equivalent_cycles": equivalent_cycles,
        "postprocess_dir_name": args.postprocess_dir_name,
        "offline_derived_mirror": bool(args.offline_derived_mirror),
        "postprocess_provenance_modes": sorted(postprocess_provenance_modes),
        "audit_warning_row_count": sum(
            int(row["warning_count"]) > 0 for row in audit_inventory_rows
        ),
        "source_re_lower_bound_warning_samples": sum(
            int(row["source_re_lower_bound_warning_samples"])
            for row in audit_inventory_rows
        ),
        "provenance_recovery_row_count": sum(
            int(row["provenance_recovery_record_count"]) > 0
            for row in audit_inventory_rows
        ),
        "artifacts": artifacts,
    }
    write_json(output_dir / "aggregation_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        help="Optional local mirror that contains one directory per manifest row.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wohler-exponent", type=float, default=10.0)
    parser.add_argument("--equivalent-cycles", type=float, default=600.0)
    parser.add_argument("--postprocess-dir-name", default="postprocess")
    parser.add_argument(
        "--offline-derived-mirror",
        action="store_true",
        help=(
            "Aggregate a local mirror from separately hash-audited corrected "
            "postprocess inputs when cluster-only wind/runtime files are absent."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        print(json.dumps(aggregate(args), indent=2, sort_keys=True))
        return 0
    except (DLCManifestError, OSError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
