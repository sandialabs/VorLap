#!/usr/bin/env python3
"""Postprocess parked VorLap/QBlade campaign results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class CaseResult:
    scenario: str
    label: str
    wind_speed: float
    azimuth_deg: float
    path: Path
    data: pd.DataFrame


@dataclass(frozen=True)
class FatigueQualityThresholds:
    min_samples: int
    min_duration_s: float
    min_rainflow_cycles: float


SOURCE_COVERAGE_COLUMNS = [
    "scenario",
    "case",
    "component",
    "airfoil_id",
    "reynolds_model",
    "wind_speed_mps",
    "azimuth_deg",
    "aoa_min_deg",
    "aoa_max_deg",
    "source_aoa_min_deg",
    "source_aoa_max_deg",
    "re_min",
    "re_max",
    "source_re_min",
    "source_re_max",
    "freq_min_hz",
    "freq_max_hz",
    "unfiltered_freq_max_hz",
    "frequency_truncated_count",
    "nyquist_hz",
    "nyquist_ratio",
    "max_nodal_force_n",
    "force_coefficient_max_abs",
    "source_force_coefficient_max_abs",
    "max_structural_velocity_mps",
    "force_power_mean_w",
    "force_power_max_abs_w",
    "re_lower_bound_warning_count",
    "re_out_of_range_count",
    "aoa_out_of_range_count",
    "aliased_frequency_count",
    "source_clip_count",
    "force_coefficient_out_of_range_count",
    "nonfinite_count",
    "samples",
    "node_samples",
    "summary_file",
]

RAINFLOW_VALIDATION_COLUMNS = [
    "case",
    "description",
    "expected_cycles",
    "actual_cycles",
    "expected_damage",
    "actual_damage",
    "equivalent_cycles",
    "expected_del",
    "actual_del",
    "actual_ranges",
    "actual_counts",
    "passed",
]

COLOR_CYCLE = ["#348ABD", "#A60628", "#009E73", "#7A68A6", "#D55E00", "#CC79A7"]
DURATION_THRESHOLD_TOLERANCE_S = 0.01
DEFAULT_EQUIVALENT_CYCLES = 600.0
MAX_TIME_GRID_ABSOLUTE_TOLERANCE_S = 5.01e-4
TIME_GRID_RELATIVE_TO_STEP_TOLERANCE = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_raw_postprocess_manifest(
    path: Path,
    cases: list[CaseResult],
    scenarios: list[str],
    comparison_baseline: str,
    comparison_coupled: str,
    wohler_exponent: float,
    equivalent_cycles: float,
    thresholds: FatigueQualityThresholds,
    outputs: list[Path],
) -> dict[str, object]:
    present_scenarios = {case.scenario for case in cases}
    report: dict[str, object] = {
        "schema_version": 1,
        "passed": all(scenario in present_scenarios for scenario in scenarios),
        "source_mode": "raw_qblade_exports",
        "scenarios": scenarios,
        "comparison_baseline": comparison_baseline,
        "comparison_coupled": comparison_coupled,
        "wohler_exponent": wohler_exponent,
        "equivalent_cycles": equivalent_cycles,
        "quality_thresholds": {
            "min_samples": thresholds.min_samples,
            "min_duration_s": thresholds.min_duration_s,
            "min_rainflow_cycles": thresholds.min_rainflow_cycles,
        },
        "inputs": {
            f"{case.scenario}:{case.label}": {
                "scenario": case.scenario,
                "case": case.label,
                **artifact_record(case.path),
            }
            for case in cases
        },
        "outputs": {output.name: artifact_record(output) for output in outputs},
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["passed"] is not True:
        missing = sorted(set(scenarios) - present_scenarios)
        raise RuntimeError(
            "Raw postprocess provenance is incomplete; missing scenario(s): "
            + ", ".join(missing)
        )
    return report


def lighten_hex_color(color: str, fraction: float = 0.34) -> str:
    color = color.lstrip("#")
    rgb = np.asarray([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    rgb = rgb + (1.0 - rgb) * fraction
    return "#" + "".join(f"{int(round(channel * 255)):02X}" for channel in rgb)


def comparison_markevery(sample_count: int) -> int:
    return max(1, int(np.ceil(max(sample_count, 1) / 24.0)))


def parse_float_list(spec: str) -> list[float]:
    return [float(part) for part in re.split(r"[\s,]+", spec.strip()) if part]


def parse_scenarios(spec: str) -> list[str]:
    scenarios = [part for part in re.split(r"[\s,]+", spec.strip()) if part]
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("Scenario list contains duplicates.")
    return scenarios


def select_comparison_pair(
    scenarios: list[str],
    comparison_baseline: str | None,
    comparison_coupled: str | None,
) -> tuple[str, str]:
    if comparison_baseline is None:
        comparison_baseline = "identity" if "identity" in scenarios and "vorlap" in scenarios else scenarios[0]
    if comparison_coupled is None:
        comparison_coupled = "vorlap" if "vorlap" in scenarios and comparison_baseline != "vorlap" else None
    if comparison_coupled is None:
        candidates = [scenario for scenario in scenarios if scenario != comparison_baseline]
        if not candidates:
            raise ValueError("At least two distinct scenarios are required for DEL comparison.")
        comparison_coupled = candidates[0]
    if comparison_baseline == comparison_coupled:
        raise ValueError("Comparison baseline and coupled scenarios must be distinct.")
    missing = [scenario for scenario in (comparison_baseline, comparison_coupled) if scenario not in scenarios]
    if missing:
        raise ValueError(f"Comparison scenario(s) not listed in --scenarios: {', '.join(missing)}.")
    return comparison_baseline, comparison_coupled


def _strict_finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    nonfinite = np.flatnonzero(~np.isfinite(array))
    if nonfinite.size:
        preview = ", ".join(str(int(index)) for index in nonfinite[:5])
        raise ValueError(f"{name} contains non-finite sample(s) at index {preview}")
    return array


def read_qblade_results(
    path: Path,
    *,
    expected_columns: list[str] | None = None,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read one QBlade export without coercing or discarding invalid input."""
    if not path.is_file():
        raise FileNotFoundError(path)
    numeric_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            next(reader)
            next(reader)
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(
                f"QBlade result must contain two metadata rows, a header, and data: {path}"
            ) from exc
        columns = [column.strip() for column in header]
        if not columns or any(not column for column in columns):
            raise ValueError(f"QBlade result contains an empty column name: {path}")
        duplicate_columns = sorted(
            {column for column in columns if columns.count(column) > 1}
        )
        if duplicate_columns:
            raise ValueError(
                f"QBlade result contains duplicate column name(s): {', '.join(duplicate_columns)}"
            )
        if expected_columns is not None and columns != list(expected_columns):
            missing = [column for column in expected_columns if column not in columns]
            extra = [column for column in columns if column not in expected_columns]
            mismatch = next(
                (
                    index
                    for index, (expected, observed) in enumerate(
                        zip(expected_columns, columns)
                    )
                    if expected != observed
                ),
                min(len(expected_columns), len(columns)),
            )
            raise ValueError(
                "QBlade result columns do not exactly match the generated result filter: "
                f"first_mismatch={mismatch}, missing={missing[:5]}, extra={extra[:5]}"
            )
        required_columns = list(required_columns or [])
        missing_required = [
            column for column in required_columns if columns.count(column) != 1
        ]
        if missing_required:
            raise ValueError(
                "QBlade result is missing required unique fatigue column(s): "
                + ", ".join(missing_required)
            )
        for line_number, row in enumerate(reader, start=4):
            if len(row) != len(columns):
                raise ValueError(
                    f"QBlade result row {line_number} has {len(row)} fields; expected {len(columns)}"
                )
            numeric_row: list[float] = []
            for column, token in zip(columns, row):
                stripped = token.strip()
                if not stripped:
                    raise ValueError(
                        f"QBlade result row {line_number}, column {column!r} is empty"
                    )
                try:
                    value = float(stripped)
                except ValueError as exc:
                    raise ValueError(
                        f"QBlade result row {line_number}, column {column!r} is not numeric: {stripped!r}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(
                        f"QBlade result row {line_number}, column {column!r} is non-finite: {stripped!r}"
                    )
                numeric_row.append(value)
            numeric_rows.append(numeric_row)
    if not numeric_rows:
        raise ValueError(f"QBlade result has no data rows: {path}")
    return pd.DataFrame(numeric_rows, columns=columns, dtype=float)


def fatigue_required_columns(blade: str, locations: list[float]) -> list[str]:
    columns = ["Time~[s]"]
    for position in locations:
        position_text = f"{float(position):.3f}"
        for quantity, unit in (("For.", "N"), ("Mom.", "Nm")):
            columns.extend(
                f"{axis}_l~{quantity}~{blade}~pos~{position_text}~[{unit}]"
                for axis in ("X", "Y", "Z")
            )
    return columns


def _resolved_case_file(manifest_path: Path, raw_path: object, *, label: str) -> Path:
    path = Path(str(raw_path))
    if path.is_file():
        return path
    candidate = manifest_path.parent / "results" / label / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(candidate)


def _resolved_result_filter(
    manifest_path: Path,
    case: dict,
    *,
    label: str,
) -> Path | None:
    raw_filter = case.get("result_filter")
    if raw_filter is None or not str(raw_filter).strip():
        return None
    path = Path(str(raw_filter))
    candidates = [
        path,
        manifest_path.parent / "cases" / label / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(candidates[-1])


def result_filter_columns(path: Path) -> list[str]:
    channels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]
    if not channels:
        raise ValueError(f"QBlade result filter has no channels: {path}")
    duplicates = sorted({channel for channel in channels if channels.count(channel) > 1})
    if duplicates:
        raise ValueError(
            f"QBlade result filter contains duplicate channel(s): {', '.join(duplicates)}"
        )
    return [re.sub(r"\s+", "~", channel) for channel in channels]


def _settings_positive_integer(value: object, *, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Campaign {name} must be a positive integer") from exc
    if not math.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"Campaign {name} must be a positive integer")
    return int(numeric)


def result_history_plan(settings: dict) -> tuple[np.ndarray, float]:
    try:
        timestep = float(settings["timestep_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Campaign settings must define numeric timestep_s") from exc
    if "timesteps" not in settings:
        raise ValueError("Campaign settings must define timesteps")
    planned_steps = _settings_positive_integer(settings["timesteps"], name="timesteps")
    override = settings.get("qblade_steps_override")
    expected_steps = (
        planned_steps
        if override is None
        else _settings_positive_integer(override, name="qblade_steps_override")
    )
    store_from = float(settings.get("store_from_s", 0.0))
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("Campaign timestep_s must be finite and positive")
    if not math.isfinite(store_from) or store_from < 0.0:
        raise ValueError("Campaign store_from_s must be finite and nonnegative")
    start_index = max(0, int(math.ceil(store_from / timestep - 1.0e-10)))
    if start_index >= expected_steps:
        raise ValueError("Campaign store_from_s leaves no retained result samples")
    expected_time = np.arange(start_index, expected_steps, dtype=float) * timestep
    tolerance = min(
        MAX_TIME_GRID_ABSOLUTE_TOLERANCE_S,
        max(1.0e-10, TIME_GRID_RELATIVE_TO_STEP_TOLERANCE * timestep),
    )
    return expected_time, tolerance


def validate_result_history(data: pd.DataFrame, settings: dict, *, path: Path) -> None:
    if "Time~[s]" not in data.columns:
        raise ValueError(f"QBlade result has no Time~[s] column: {path}")
    values = data.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"QBlade result contains non-finite numeric data: {path}")
    time = _strict_finite_vector(data["Time~[s]"].to_numpy(float), name="QBlade time")
    if time.size < 2:
        raise ValueError(f"QBlade result must contain at least two retained samples: {path}")
    differences = np.diff(time)
    if np.any(differences <= 0.0):
        raise ValueError(f"QBlade result time must be strictly increasing with no duplicates: {path}")
    expected_time, tolerance = result_history_plan(settings)
    if time.size != expected_time.size:
        raise ValueError(
            f"QBlade result has {time.size} retained rows; expected exactly {expected_time.size}: {path}"
        )
    error = np.abs(time - expected_time)
    max_error = float(np.max(error))
    if max_error > tolerance:
        index = int(np.argmax(error))
        raise ValueError(
            "QBlade result time grid differs from the planned grid: "
            f"row={index}, observed={time[index]:.12g}, expected={expected_time[index]:.12g}, "
            f"error={max_error:.6g} s, tolerance={tolerance:.6g} s"
        )


def _validate_campaign_plan(settings: dict, scenarios: list[str]) -> list[dict]:
    if not settings:
        raise FileNotFoundError("campaign_settings.json is required for fail-closed postprocessing")
    configured_scenarios = settings.get("scenario_names")
    if not isinstance(configured_scenarios, list) or not configured_scenarios:
        raise ValueError("Campaign settings have no scenario_names plan")
    if any(not isinstance(name, str) or not name for name in configured_scenarios):
        raise ValueError("Campaign scenario_names plan contains an invalid name")
    if len(set(configured_scenarios)) != len(configured_scenarios):
        raise ValueError("Campaign scenario_names plan contains duplicates")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("Requested scenario list contains duplicates")
    unknown = [scenario for scenario in scenarios if scenario not in configured_scenarios]
    if unknown:
        raise ValueError(f"Requested scenario(s) are absent from campaign plan: {', '.join(unknown)}")
    planned_cases = settings.get("planned_cases")
    if not isinstance(planned_cases, list) or not planned_cases:
        raise ValueError("Campaign settings have no planned_cases matrix")
    labels = [str(case.get("label", "")) for case in planned_cases if isinstance(case, dict)]
    if len(labels) != len(planned_cases) or any(not label for label in labels):
        raise ValueError("Campaign planned_cases contain an invalid or missing label")
    if len(set(labels)) != len(labels):
        raise ValueError("Campaign planned_cases contain duplicate labels")
    planned_per_scenario = settings.get("planned_case_count_per_scenario")
    if planned_per_scenario is not None and _settings_positive_integer(
        planned_per_scenario,
        name="planned_case_count_per_scenario",
    ) != len(planned_cases):
        raise ValueError("Campaign planned_case_count_per_scenario does not match planned_cases")
    planned_total = settings.get("planned_case_count_total")
    if planned_total is not None and _settings_positive_integer(
        planned_total,
        name="planned_case_count_total",
    ) != len(configured_scenarios) * len(planned_cases):
        raise ValueError("Campaign planned_case_count_total does not match the planned matrix")
    return planned_cases


def load_campaign(
    run_root: Path,
    scenarios: list[str],
    *,
    settings: dict | None = None,
    required_columns: list[str] | None = None,
) -> list[CaseResult]:
    settings = load_campaign_settings(run_root) if settings is None else settings
    planned_cases = _validate_campaign_plan(settings, scenarios)
    planned_labels = [str(case["label"]) for case in planned_cases]
    cases: list[CaseResult] = []
    for scenario in scenarios:
        manifest_path = run_root / scenario / "cases.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing scenario manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_cases = manifest.get("cases")
        if not isinstance(manifest_cases, list):
            raise ValueError(f"Scenario manifest has no cases list: {manifest_path}")
        observed_labels = [str(case.get("label", "")) for case in manifest_cases if isinstance(case, dict)]
        if observed_labels != planned_labels:
            raise ValueError(
                f"Scenario {scenario!r} case matrix is incomplete or reordered: "
                f"observed={observed_labels}, expected={planned_labels}"
            )
        for planned_case, case in zip(planned_cases, manifest_cases):
            label = str(case["label"])
            result_name = str(case.get("result_name", "")).strip()
            if not result_name:
                raise ValueError(f"Scenario {scenario!r} case {label!r} has no result_name")
            result_path = _resolved_case_file(
                manifest_path,
                Path(str(case.get("result_dir", ""))) / f"{result_name}.txt",
                label=label,
            )
            filter_path = _resolved_result_filter(manifest_path, case, label=label)
            expected_filter_hash = str(settings.get("result_filter_sha256", "") or "")
            planned_filter = str(settings.get("result_filter", "") or "").strip()
            if filter_path is None:
                if expected_filter_hash or planned_filter:
                    raise ValueError(
                        f"Scenario {scenario!r} case {label!r} is missing its planned result filter"
                    )
                expected_columns = None
            else:
                expected_columns = result_filter_columns(filter_path)
                if expected_filter_hash and sha256_file(filter_path) != expected_filter_hash:
                    raise ValueError(
                        f"Generated result filter hash does not match campaign settings: {filter_path}"
                    )
            data = read_qblade_results(
                result_path,
                expected_columns=expected_columns,
                required_columns=required_columns,
            )
            validate_result_history(data, settings, path=result_path)
            wind_speed = float(case["wind_speed"])
            expected_wind_speed = float(planned_case["wind_speed_mps"])
            azimuth_deg = case_azimuth_deg(case)
            expected_azimuth_deg = float(planned_case["parked_azimuth_deg"])
            if not math.isclose(wind_speed, expected_wind_speed, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    f"Scenario {scenario!r} case {label!r} wind speed {wind_speed} "
                    f"does not match planned {expected_wind_speed}"
                )
            if not math.isclose(azimuth_deg, expected_azimuth_deg, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    f"Scenario {scenario!r} case {label!r} azimuth {azimuth_deg} "
                    f"does not match planned {expected_azimuth_deg}"
                )
            cases.append(
                CaseResult(
                    scenario=scenario,
                    label=label,
                    wind_speed=wind_speed,
                    azimuth_deg=azimuth_deg,
                    path=result_path,
                    data=data,
                )
            )
    expected_case_count = len(scenarios) * len(planned_cases)
    if len(cases) != expected_case_count:
        raise RuntimeError(
            f"Loaded {len(cases)} cases; expected complete matrix of {expected_case_count}"
        )
    return cases


def load_campaign_settings(run_root: Path) -> dict:
    settings_path = run_root / "campaign_settings.json"
    if not settings_path.is_file():
        return {}
    return json.loads(settings_path.read_text(encoding="utf-8"))


def case_azimuth_deg(case: dict) -> float:
    """Return explicit parked azimuth, leaving legacy manifests unlabelled."""
    if "azimuth_deg" not in case:
        return math.nan
    return float(case["azimuth_deg"])


def load_completion_summary(run_root: Path, scenarios: list[str], timestep: float) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        manifest_path = run_root / scenario / "cases.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest.get("cases", []):
            result_dir = Path(case["result_dir"])
            if not result_dir.is_dir():
                result_dir = manifest_path.parent / "results" / case["label"]
            summary_files = sorted(result_dir.glob("*_run_summary.json"))
            summary = json.loads(summary_files[0].read_text(encoding="utf-8")) if summary_files else {}
            requested_steps = int(summary.get("requested_steps", summary.get("steps", 0)) or 0)
            completed_steps = int(summary.get("completed_steps", requested_steps) or 0)
            completed_time_s = completed_steps * timestep if math.isfinite(timestep) else math.nan
            if not math.isfinite(completed_time_s):
                completed_time_s = sampled_export_duration_seconds(
                    summary.get("qblade_export_time_diagnostics", {})
                )
            rows.append(
                {
                    "scenario": scenario,
                    "case": case["label"],
                    "wind_speed_mps": float(case["wind_speed"]),
                    "azimuth_deg": case_azimuth_deg(case),
                    "requested_steps": requested_steps,
                    "completed_steps": completed_steps,
                    "completed_time_s": completed_time_s,
                    "failed": bool(summary.get("failed", not bool(summary))),
                    "failure_message": summary.get("failure_message", "missing run summary" if not summary else ""),
                    "exported_results": bool(summary.get("exported_results", False)),
                    "summary_file": str(summary_files[0]) if summary_files else "",
                }
            )
    return pd.DataFrame(rows)


def load_source_coverage(run_root: Path, scenarios: list[str]) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        manifest_path = run_root / scenario / "cases.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest.get("cases", []):
            result_dir = Path(case["result_dir"])
            if not result_dir.is_dir():
                result_dir = manifest_path.parent / "results" / case["label"]
            summary_files = sorted(result_dir.glob("*_run_summary.json"))
            if not summary_files:
                continue
            summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
            coverage_rows = summary.get("sil_vorlap_source_coverage", [])
            if not isinstance(coverage_rows, list):
                continue
            for coverage in coverage_rows:
                if not isinstance(coverage, dict):
                    continue
                row = {
                    "scenario": scenario,
                    "case": case["label"],
                    "wind_speed_mps": float(case["wind_speed"]),
                    "azimuth_deg": case_azimuth_deg(case),
                    "summary_file": str(summary_files[0]),
                }
                row.update(coverage)
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=SOURCE_COVERAGE_COLUMNS)
    table = pd.DataFrame(rows)
    for column in SOURCE_COVERAGE_COLUMNS:
        if column not in table.columns:
            if column in {"scenario", "case", "component", "airfoil_id", "reynolds_model", "summary_file"}:
                table[column] = ""
            elif column == "re_lower_bound_warning_count":
                table[column] = 0
            else:
                table[column] = math.nan
    return table[SOURCE_COVERAGE_COLUMNS].sort_values(["scenario", "wind_speed_mps", "azimuth_deg", "case", "component"])


def sampled_duration_seconds(values: np.ndarray) -> float:
    time = _strict_finite_vector(values, name="sample time")
    if time.size < 2:
        return 0.0
    differences = np.diff(time)
    if np.any(differences <= 0.0):
        raise ValueError("sample time must be strictly increasing with no duplicates")
    sample_period = float(np.median(differences))
    return float(time[-1] - time[0] + sample_period)


def sampled_export_duration_seconds(diagnostics: object) -> float:
    if not isinstance(diagnostics, dict):
        return math.nan
    samples = int(diagnostics.get("samples", 0) or 0)
    first = diagnostics.get("first_time_s")
    last = diagnostics.get("last_time_s")
    if samples >= 2 and first is not None and last is not None:
        try:
            first_f = float(first)
            last_f = float(last)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(first_f) and math.isfinite(last_f) and last_f >= first_f:
                sample_period = (last_f - first_f) / max(samples - 1, 1)
                return float(last_f - first_f + sample_period)
    duration = diagnostics.get("duration_s")
    try:
        duration_f = float(duration)
    except (TypeError, ValueError):
        return math.nan
    return duration_f if math.isfinite(duration_f) else math.nan


def time_duration_seconds(data: pd.DataFrame) -> float:
    if "Time~[s]" not in data.columns:
        raise KeyError("Required QBlade time column Time~[s] is missing")
    return sampled_duration_seconds(data["Time~[s]"].to_numpy(float))


def _warning_text(warnings: list[str]) -> str:
    return "; ".join(warnings)


def meets_minimum_duration(duration_s: float, minimum_s: float) -> bool:
    """Accept records no more than one hundredth of a second below the target."""
    return math.isfinite(duration_s) and duration_s >= minimum_s - DURATION_THRESHOLD_TOLERANCE_S


def completion_warnings(row: pd.Series, thresholds: FatigueQualityThresholds) -> list[str]:
    warnings: list[str] = []
    completed_steps = int(row.get("completed_steps", 0) or 0)
    completed_time_s = float(row.get("completed_time_s", math.nan))
    if completed_steps < thresholds.min_samples:
        warnings.append(f"completed_steps<{thresholds.min_samples}")
    if not meets_minimum_duration(completed_time_s, thresholds.min_duration_s):
        warnings.append(f"completed_time_s<{thresholds.min_duration_s:g}")
    if bool(row.get("failed", False)):
        warnings.append("partial_or_failed_run")
    return warnings


def annotate_completion_quality(
    completion: pd.DataFrame,
    thresholds: FatigueQualityThresholds,
) -> pd.DataFrame:
    annotated = completion.copy()
    if annotated.empty:
        annotated["smoke_only"] = []
        annotated["postprocess_warnings"] = []
        return annotated
    warning_lists = [completion_warnings(row, thresholds) for _, row in annotated.iterrows()]
    annotated["smoke_only"] = [bool(warnings) for warnings in warning_lists]
    annotated["postprocess_warnings"] = [_warning_text(warnings) for warnings in warning_lists]
    return annotated


def find_vector_columns(
    columns: list[str],
    *,
    blade: str,
    position: float,
    quantity: str,
    unit: str,
) -> tuple[float, dict[str, str]]:
    pattern = re.compile(
        rf"^(?P<axis>[XYZ])_l~{re.escape(quantity)}~{re.escape(blade)}~pos~"
        rf"(?P<pos>[0-9.]+)~\[{re.escape(unit)}\]$"
    )
    by_position: dict[float, dict[str, str]] = {}
    for column in columns:
        match = pattern.match(column)
        if not match:
            continue
        by_position.setdefault(float(match.group("pos")), {})[match.group("axis")] = column
    if not by_position:
        raise KeyError(f"No {quantity} {unit} channels found for {blade}")
    nearest = min(by_position, key=lambda value: abs(value - position))
    axis_columns = by_position[nearest]
    missing = [axis for axis in ("X", "Y", "Z") if axis not in axis_columns]
    if missing:
        raise KeyError(f"Missing {quantity} components {missing} for {blade} pos {nearest:.3f}")
    return nearest, axis_columns


def vector_resultant(data: pd.DataFrame, columns: dict[str, str]) -> np.ndarray:
    values = np.vstack(
        [
            _strict_finite_vector(
                data[columns[axis]].to_numpy(float),
                name=f"vector channel {columns[axis]}",
            )
            for axis in ("X", "Y", "Z")
        ]
    )
    return np.sqrt(np.sum(values * values, axis=0))


def turning_points(values: np.ndarray) -> np.ndarray:
    samples = _strict_finite_vector(values, name="rainflow history")
    if samples.size <= 2:
        return samples
    compact = [float(samples[0])]
    for value in samples[1:]:
        if value != compact[-1]:
            compact.append(float(value))
    if len(compact) <= 2:
        return np.asarray(compact, dtype=float)
    points = [compact[0]]
    for i in range(1, len(compact) - 1):
        prev_value = compact[i - 1]
        value = compact[i]
        next_value = compact[i + 1]
        if (value - prev_value) * (next_value - value) <= 0.0:
            points.append(value)
    points.append(compact[-1])
    return np.asarray(points, dtype=float)


def _finite_absolute_range(first: float, second: float, *, name: str) -> float:
    value = abs(float(second) - float(first))
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} became non-finite")
    return value


def rainflow_ranges(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = turning_points(values)
    if points.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    stack: list[float] = []
    ranges: list[float] = []
    counts: list[float] = []
    for point in points:
        stack.append(float(point))
        while len(stack) >= 3:
            old_range = _finite_absolute_range(
                stack[-2],
                stack[-3],
                name="rainflow range",
            )
            new_range = _finite_absolute_range(
                stack[-1],
                stack[-2],
                name="rainflow range",
            )
            if new_range < old_range:
                break
            if old_range > 0.0:
                ranges.append(old_range)
                counts.append(0.5 if len(stack) == 3 else 1.0)
            if len(stack) == 3:
                stack.pop(0)
            else:
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for i in range(len(stack) - 1):
        residual_range = _finite_absolute_range(
            stack[i + 1],
            stack[i],
            name="rainflow residual range",
        )
        if residual_range > 0.0:
            ranges.append(residual_range)
            counts.append(0.5)
    return np.asarray(ranges, dtype=float), np.asarray(counts, dtype=float)


def reference_rainflow_ranges(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independently structured four-point reference counter.

    This routine is intentionally not used for production DEL calculation. It
    removes enclosed reversal pairs from a working list, whereas
    :func:`rainflow_ranges` uses a streaming three-point stack.
    """
    samples = _strict_finite_vector(values, name="reference rainflow history")
    if samples.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    compact: list[float] = []
    for raw_value in samples:
        value = float(raw_value)
        if not compact or value != compact[-1]:
            compact.append(value)
    if len(compact) <= 2:
        reversals = compact
    else:
        reversals = [compact[0]]
        previous_slope = compact[1] - compact[0]
        for index in range(1, len(compact) - 1):
            next_slope = compact[index + 1] - compact[index]
            if previous_slope * next_slope <= 0.0:
                reversals.append(compact[index])
            previous_slope = next_slope
        reversals.append(compact[-1])

    work = reversals.copy()
    closed_ranges: list[float] = []
    index = 0
    while len(work) >= 4 and index <= len(work) - 4:
        first, second, third, fourth = work[index : index + 4]
        left_range = _finite_absolute_range(first, second, name="reference rainflow range")
        enclosed_range = _finite_absolute_range(
            second,
            third,
            name="reference rainflow range",
        )
        right_range = _finite_absolute_range(third, fourth, name="reference rainflow range")
        if enclosed_range <= left_range and enclosed_range <= right_range:
            if enclosed_range > 0.0:
                closed_ranges.append(enclosed_range)
            del work[index + 1 : index + 3]
            index = max(0, index - 2)
        else:
            index += 1

    residual_ranges = [
        _finite_absolute_range(
            first,
            second,
            name="reference rainflow residual range",
        )
        for first, second in zip(work, work[1:])
        if second != first
    ]
    ranges = np.asarray([*closed_ranges, *residual_ranges], dtype=float)
    counts = np.asarray(
        [*[1.0] * len(closed_ranges), *[0.5] * len(residual_ranges)],
        dtype=float,
    )
    return ranges, counts


def aggregate_range_count_multiset(
    ranges: np.ndarray,
    counts: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    range_values = _strict_finite_vector(ranges, name="rainflow ranges")
    count_values = _strict_finite_vector(counts, name="rainflow counts")
    if range_values.shape != count_values.shape:
        raise ValueError("rainflow ranges and counts must have matching shapes")
    aggregated: dict[float, float] = {}
    for cycle_range, count in zip(range_values, count_values):
        if cycle_range <= 0.0:
            raise ValueError("rainflow ranges must be positive")
        if count <= 0.0:
            raise ValueError("rainflow counts must be positive")
        key = float(cycle_range)
        aggregated[key] = aggregated.get(key, 0.0) + float(count)
    return tuple(sorted(aggregated.items()))


def _range_count_multisets_close(
    actual: tuple[tuple[float, float], ...],
    expected: tuple[tuple[float, float], ...],
    *,
    rtol: float,
    atol: float,
) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        math.isclose(actual_range, expected_range, rel_tol=rtol, abs_tol=atol)
        and math.isclose(actual_count, expected_count, rel_tol=rtol, abs_tol=atol)
        for (actual_range, actual_count), (expected_range, expected_count) in zip(
            actual, expected
        )
    )


def rainflow_damage(values: np.ndarray, wohler_exponent: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return direct pseudo-damage A_m, cycle count, ranges, and counts."""
    if not math.isfinite(wohler_exponent) or wohler_exponent <= 0.0:
        raise ValueError("wohler_exponent must be finite and positive")
    ranges, counts = rainflow_ranges(values)
    if ranges.size == 0 or np.sum(counts) <= 0.0:
        return 0.0, 0.0, ranges, counts
    with np.errstate(over="ignore", invalid="ignore"):
        terms = counts * np.power(ranges, wohler_exponent)
    if np.any(~np.isfinite(terms)):
        raise FloatingPointError("rainflow damage terms became non-finite")
    damage = float(np.sum(terms))
    cycle_count = float(np.sum(counts))
    if not math.isfinite(damage) or not math.isfinite(cycle_count):
        raise FloatingPointError("rainflow damage accumulation became non-finite")
    return damage, cycle_count, ranges, counts


def equivalent_load_with_cycles(
    values: np.ndarray,
    wohler_exponent: float,
    equivalent_cycles: float = DEFAULT_EQUIVALENT_CYCLES,
) -> tuple[float, float]:
    if not math.isfinite(equivalent_cycles) or equivalent_cycles <= 0.0:
        raise ValueError("equivalent_cycles must be finite and positive")
    damage, cycle_count, _, _ = rainflow_damage(values, wohler_exponent)
    if damage <= 0.0 or cycle_count <= 0.0:
        return 0.0, cycle_count
    return (
        float(np.power(damage / equivalent_cycles, 1.0 / wohler_exponent)),
        cycle_count,
    )


def equivalent_load_from_pseudo_damage(
    pseudo_damage_a_m: float,
    wohler_exponent: float,
    equivalent_cycles: float,
) -> float:
    """Convert the direct range-power sum A_m to a fixed-cycle DEL."""

    damage = float(pseudo_damage_a_m)
    if not math.isfinite(damage) or damage < 0.0:
        raise ValueError("pseudo_damage_a_m must be finite and non-negative")
    if not math.isfinite(wohler_exponent) or wohler_exponent <= 0.0:
        raise ValueError("wohler_exponent must be finite and positive")
    if not math.isfinite(equivalent_cycles) or equivalent_cycles <= 0.0:
        raise ValueError("equivalent_cycles must be finite and positive")
    if damage == 0.0:
        return 0.0
    result = float(np.power(damage / equivalent_cycles, 1.0 / wohler_exponent))
    if not math.isfinite(result):
        raise FloatingPointError("equivalent load became non-finite")
    return result


def equivalent_load(
    values: np.ndarray,
    wohler_exponent: float,
    equivalent_cycles: float = DEFAULT_EQUIVALENT_CYCLES,
) -> float:
    load, _ = equivalent_load_with_cycles(
        values,
        wohler_exponent,
        equivalent_cycles,
    )
    return load


def rainflow_validation_table(
    wohler_exponent: float,
    equivalent_cycles: float = DEFAULT_EQUIVALENT_CYCLES,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Validate full range-count multisets against frozen and reference cases."""
    cases = [
        {
            "case": "constant_signal",
            "description": "constant signal has no damaging ranges",
            "values": np.asarray([5.0, 5.0, 5.0], dtype=float),
            "expected_range_counts": (),
        },
        {
            "case": "single_closed_cycle_range_2",
            "description": "one closed reversal pair with range 2",
            "values": np.asarray([-1.0, 1.0, -1.0], dtype=float),
            "expected_range_counts": ((2.0, 1.0),),
        },
        {
            "case": "two_closed_cycles_range_2",
            "description": "two closed reversal pairs with range 2",
            "values": np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0], dtype=float),
            "expected_range_counts": ((2.0, 2.0),),
        },
        {
            "case": "open_half_cycle_range_3",
            "description": "one residual half cycle with range 3",
            "values": np.asarray([0.0, 3.0], dtype=float),
            "expected_range_counts": ((3.0, 0.5),),
        },
        {
            "case": "nested_reversals",
            "description": "nested reversals retain complete range-count histogram",
            "values": np.asarray([0.0, 5.0, 1.0, 4.0, 2.0, 3.0, 0.0], dtype=float),
            "expected_range_counts": ((1.0, 1.0), (3.0, 1.0), (5.0, 1.0)),
        },
        {
            "case": "astm_style_mixed_history",
            "description": "frozen ASTM-style mixed closed and residual cycles",
            "values": np.asarray(
                [
                    0.0,
                    -150.0,
                    100.0,
                    0.0,
                    130.0,
                    -120.0,
                    110.0,
                    -50.0,
                    80.0,
                    0.0,
                    150.0,
                    -100.0,
                    100.0,
                    0.0,
                ],
                dtype=float,
            ),
            "expected_range_counts": (
                (80.0, 1.0),
                (100.0, 1.5),
                (150.0, 0.5),
                (160.0, 1.0),
                (200.0, 0.5),
                (250.0, 1.5),
                (300.0, 0.5),
            ),
        },
    ]
    rows = []
    for case in cases:
        damage, cycles, ranges, counts = rainflow_damage(case["values"], wohler_exponent)
        reference_ranges, reference_counts = reference_rainflow_ranges(case["values"])
        actual_multiset = aggregate_range_count_multiset(ranges, counts)
        reference_multiset = aggregate_range_count_multiset(
            reference_ranges,
            reference_counts,
        )
        expected_multiset = tuple(case["expected_range_counts"])
        expected_cycles = float(sum(count for _, count in expected_multiset))
        expected_damage = float(
            sum(count * math.pow(cycle_range, wohler_exponent) for cycle_range, count in expected_multiset)
        )
        expected_del = (
            math.pow(
                expected_damage / equivalent_cycles,
                1.0 / wohler_exponent,
            )
            if expected_damage > 0.0
            else 0.0
        )
        actual_del, _ = equivalent_load_with_cycles(
            case["values"],
            wohler_exponent,
            equivalent_cycles,
        )
        passed = all(
            [
                _range_count_multisets_close(
                    actual_multiset,
                    expected_multiset,
                    rtol=rtol,
                    atol=atol,
                ),
                _range_count_multisets_close(
                    reference_multiset,
                    expected_multiset,
                    rtol=rtol,
                    atol=atol,
                ),
                math.isclose(cycles, expected_cycles, rel_tol=rtol, abs_tol=atol),
                math.isclose(damage, expected_damage, rel_tol=rtol, abs_tol=atol),
                math.isclose(actual_del, expected_del, rel_tol=rtol, abs_tol=atol),
            ]
        )
        rows.append(
            {
                "case": case["case"],
                "description": case["description"],
                "expected_cycles": expected_cycles,
                "actual_cycles": cycles,
                "expected_damage": expected_damage,
                "actual_damage": damage,
                "equivalent_cycles": equivalent_cycles,
                "expected_del": expected_del,
                "actual_del": actual_del,
                "actual_ranges": json.dumps(ranges.tolist()),
                "actual_counts": json.dumps(counts.tolist()),
                "passed": passed,
            }
        )
    return pd.DataFrame(rows, columns=RAINFLOW_VALIDATION_COLUMNS)


def fatigue_metric_warnings(
    *,
    samples: int,
    duration_s: float,
    force_cycles: float,
    moment_cycles: float,
    thresholds: FatigueQualityThresholds,
) -> list[str]:
    warnings: list[str] = []
    if samples < thresholds.min_samples:
        warnings.append(f"samples<{thresholds.min_samples}")
    if not meets_minimum_duration(duration_s, thresholds.min_duration_s):
        warnings.append(f"duration_s<{thresholds.min_duration_s:g}")
    if force_cycles < thresholds.min_rainflow_cycles:
        warnings.append(f"force_cycles<{thresholds.min_rainflow_cycles:g}")
    if moment_cycles < thresholds.min_rainflow_cycles:
        warnings.append(f"moment_cycles<{thresholds.min_rainflow_cycles:g}")
    return warnings


def signed_metric_warnings(
    *,
    samples: int,
    duration_s: float,
    cycles: float,
    thresholds: FatigueQualityThresholds,
) -> list[str]:
    warnings: list[str] = []
    if samples < thresholds.min_samples:
        warnings.append(f"samples<{thresholds.min_samples}")
    if not meets_minimum_duration(duration_s, thresholds.min_duration_s):
        warnings.append(f"duration_s<{thresholds.min_duration_s:g}")
    if cycles < thresholds.min_rainflow_cycles:
        warnings.append(f"cycles<{thresholds.min_rainflow_cycles:g}")
    return warnings


def rms(values: np.ndarray) -> float:
    samples = _strict_finite_vector(values, name="RMS history")
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


def latex_escape(text: object) -> str:
    value = str(text)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        value = value.replace(old, new)
    return value


def compute_metrics(
    cases: list[CaseResult],
    locations: list[float],
    blade: str,
    wohler: float,
    thresholds: FatigueQualityThresholds,
    equivalent_cycles: float = DEFAULT_EQUIVALENT_CYCLES,
) -> pd.DataFrame:
    rows = []
    for case in cases:
        columns = list(case.data.columns)
        samples = int(len(case.data))
        duration_s = time_duration_seconds(case.data)
        for location in locations:
            try:
                actual_force_pos, force_cols = find_vector_columns(
                    columns,
                    blade=blade,
                    position=location,
                    quantity="For.",
                    unit="N",
                )
                actual_moment_pos, moment_cols = find_vector_columns(
                    columns,
                    blade=blade,
                    position=location,
                    quantity="Mom.",
                    unit="Nm",
                )
            except KeyError as exc:
                raise ValueError(
                    f"Missing required fatigue vector for {case.scenario} {case.label} "
                    f"at requested position {location}: {exc}"
                ) from exc
            force = vector_resultant(case.data, force_cols)
            moment = vector_resultant(case.data, moment_cols)
            force_damage, force_cycles, _, _ = rainflow_damage(force, wohler)
            moment_damage, moment_cycles, _, _ = rainflow_damage(moment, wohler)
            force_del = equivalent_load_from_pseudo_damage(
                force_damage, wohler, equivalent_cycles
            )
            moment_del = equivalent_load_from_pseudo_damage(
                moment_damage, wohler, equivalent_cycles
            )
            warnings = fatigue_metric_warnings(
                samples=samples,
                duration_s=duration_s,
                force_cycles=force_cycles,
                moment_cycles=moment_cycles,
                thresholds=thresholds,
            )
            rows.append(
                {
                    "scenario": case.scenario,
                    "case": case.label,
                    "wind_speed_mps": case.wind_speed,
                    "azimuth_deg": case.azimuth_deg,
                    "blade": blade,
                    "requested_position": location,
                    "force_position": actual_force_pos,
                    "moment_position": actual_moment_pos,
                    "force_del_N": force_del,
                    "moment_del_Nm": moment_del,
                    "force_pseudo_damage_A_m": force_damage,
                    "moment_pseudo_damage_A_m": moment_damage,
                    "wohler_exponent": wohler,
                    "force_rainflow_cycles": force_cycles,
                    "moment_rainflow_cycles": moment_cycles,
                    "equivalent_cycles": equivalent_cycles,
                    "force_rms_N": rms(force),
                    "moment_rms_Nm": rms(moment),
                    "force_max_N": float(np.max(force)) if force.size else 0.0,
                    "moment_max_Nm": float(np.max(moment)) if moment.size else 0.0,
                    "samples": samples,
                    "duration_s": duration_s,
                    "smoke_only": bool(warnings),
                    "postprocess_warnings": _warning_text(warnings),
                    "result_file": str(case.path),
                }
            )
    return pd.DataFrame(rows)


def compute_signed_channel_metrics(
    cases: list[CaseResult],
    locations: list[float],
    blade: str,
    wohler: float,
    thresholds: FatigueQualityThresholds,
    equivalent_cycles: float = DEFAULT_EQUIVALENT_CYCLES,
) -> pd.DataFrame:
    rows = []
    quantity_specs = [
        ("force", "For.", "N"),
        ("moment", "Mom.", "Nm"),
    ]
    for case in cases:
        columns = list(case.data.columns)
        samples = int(len(case.data))
        duration_s = time_duration_seconds(case.data)
        for location in locations:
            for quantity_name, quantity_token, unit in quantity_specs:
                try:
                    actual_position, axis_columns = find_vector_columns(
                        columns,
                        blade=blade,
                        position=location,
                        quantity=quantity_token,
                        unit=unit,
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"Missing required signed fatigue vector for {case.scenario} "
                        f"{case.label}, {quantity_name}, requested position {location}: {exc}"
                    ) from exc
                for axis in ("X", "Y", "Z"):
                    channel = axis_columns[axis]
                    values = _strict_finite_vector(
                        case.data[channel].to_numpy(float),
                        name=f"signed channel {channel}",
                    )
                    pseudo_damage, cycles, _, _ = rainflow_damage(values, wohler)
                    signed_del = equivalent_load_from_pseudo_damage(
                        pseudo_damage,
                        wohler,
                        equivalent_cycles,
                    )
                    warnings = signed_metric_warnings(
                        samples=samples,
                        duration_s=duration_s,
                        cycles=cycles,
                        thresholds=thresholds,
                    )
                    rows.append(
                        {
                            "scenario": case.scenario,
                            "case": case.label,
                            "wind_speed_mps": case.wind_speed,
                            "azimuth_deg": case.azimuth_deg,
                            "blade": blade,
                            "requested_position": location,
                            "position": actual_position,
                            "quantity": quantity_name,
                            "axis": axis,
                            "unit": unit,
                            "channel": channel,
                            "signed_del": signed_del,
                            "pseudo_damage_A_m": pseudo_damage,
                            "wohler_exponent": wohler,
                            "rainflow_cycles": cycles,
                            "equivalent_cycles": equivalent_cycles,
                            "rms": rms(values),
                            "mean": float(np.mean(values)) if values.size else math.nan,
                            "min": float(np.min(values)) if values.size else math.nan,
                            "max": float(np.max(values)) if values.size else math.nan,
                            "samples": samples,
                            "duration_s": duration_s,
                            "smoke_only": bool(warnings),
                            "postprocess_warnings": _warning_text(warnings),
                            "result_file": str(case.path),
                        }
                    )
    columns = [
        "scenario",
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "blade",
        "requested_position",
        "position",
        "quantity",
        "axis",
        "unit",
        "channel",
        "signed_del",
        "rainflow_cycles",
        "equivalent_cycles",
        "rms",
        "mean",
        "min",
        "max",
        "samples",
        "duration_s",
        "smoke_only",
        "postprocess_warnings",
        "result_file",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["scenario", "wind_speed_mps", "azimuth_deg", "position", "quantity", "axis"]
    )


def comparison_table(metrics: pd.DataFrame, baseline: str, coupled: str, *, include_smoke: bool = False) -> pd.DataFrame:
    key_cols = ["case", "wind_speed_mps", "azimuth_deg", "force_position"]
    base = metrics[metrics["scenario"] == baseline].copy()
    comp = metrics[metrics["scenario"] == coupled].copy()
    merged = base.merge(comp, on=key_cols, suffixes=("_baseline", "_vorlap"))
    rows = []
    for _, row in merged.iterrows():
        force_base = float(row["force_del_N_baseline"])
        force_vorlap = float(row["force_del_N_vorlap"])
        moment_base = float(row["moment_del_Nm_baseline"])
        moment_vorlap = float(row["moment_del_Nm_vorlap"])
        smoke_only = bool(row.get("smoke_only_baseline", False)) or bool(row.get("smoke_only_vorlap", False))
        warnings = _warning_text(
            [
                str(value)
                for value in (
                    row.get("postprocess_warnings_baseline", ""),
                    row.get("postprocess_warnings_vorlap", ""),
                )
                if str(value)
            ]
        )
        if smoke_only and not include_smoke:
            continue
        rows.append(
            {
                "case": row["case"],
                "wind_speed_mps": row["wind_speed_mps"],
                "azimuth_deg": row["azimuth_deg"],
                "position": row["force_position"],
                "force_del_baseline_N": force_base,
                "force_del_vorlap_N": force_vorlap,
                "force_delta_pct": 100.0 * (force_vorlap - force_base) / force_base if force_base else math.nan,
                "moment_del_baseline_Nm": moment_base,
                "moment_del_vorlap_Nm": moment_vorlap,
                "moment_delta_pct": 100.0 * (moment_vorlap - moment_base) / moment_base if moment_base else math.nan,
                "equivalent_cycles_baseline": row["equivalent_cycles_baseline"],
                "equivalent_cycles_vorlap": row["equivalent_cycles_vorlap"],
                "smoke_only": smoke_only,
                "postprocess_warnings": warnings,
            }
        )
    columns = [
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "force_del_baseline_N",
        "force_del_vorlap_N",
        "force_delta_pct",
        "moment_del_baseline_Nm",
        "moment_del_vorlap_Nm",
        "moment_delta_pct",
        "equivalent_cycles_baseline",
        "equivalent_cycles_vorlap",
        "smoke_only",
        "postprocess_warnings",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["wind_speed_mps", "azimuth_deg", "position"])


def signed_comparison_table(metrics: pd.DataFrame, baseline: str, coupled: str, *, include_smoke: bool = False) -> pd.DataFrame:
    key_cols = ["case", "wind_speed_mps", "azimuth_deg", "position", "quantity", "axis"]
    columns = [
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "quantity",
        "axis",
        "unit",
        "channel_baseline",
        "channel_vorlap",
        "signed_del_baseline",
        "signed_del_vorlap",
        "signed_delta_pct",
        "rainflow_cycles_baseline",
        "rainflow_cycles_vorlap",
        "equivalent_cycles_baseline",
        "equivalent_cycles_vorlap",
        "smoke_only",
        "postprocess_warnings",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    base = metrics[metrics["scenario"] == baseline].copy()
    comp = metrics[metrics["scenario"] == coupled].copy()
    merged = base.merge(comp, on=key_cols, suffixes=("_baseline", "_vorlap"))
    rows = []
    for _, row in merged.iterrows():
        del_base = float(row["signed_del_baseline"])
        del_vorlap = float(row["signed_del_vorlap"])
        smoke_only = bool(row.get("smoke_only_baseline", False)) or bool(row.get("smoke_only_vorlap", False))
        warnings = _warning_text(
            [
                str(value)
                for value in (
                    row.get("postprocess_warnings_baseline", ""),
                    row.get("postprocess_warnings_vorlap", ""),
                )
                if str(value)
            ]
        )
        if smoke_only and not include_smoke:
            continue
        rows.append(
            {
                "case": row["case"],
                "wind_speed_mps": row["wind_speed_mps"],
                "azimuth_deg": row["azimuth_deg"],
                "position": row["position"],
                "quantity": row["quantity"],
                "axis": row["axis"],
                "unit": row.get("unit_baseline", row.get("unit_vorlap", "")),
                "channel_baseline": row["channel_baseline"],
                "channel_vorlap": row["channel_vorlap"],
                "signed_del_baseline": del_base,
                "signed_del_vorlap": del_vorlap,
                "signed_delta_pct": 100.0 * (del_vorlap - del_base) / del_base if del_base else math.nan,
                "rainflow_cycles_baseline": row["rainflow_cycles_baseline"],
                "rainflow_cycles_vorlap": row["rainflow_cycles_vorlap"],
                "equivalent_cycles_baseline": row["equivalent_cycles_baseline"],
                "equivalent_cycles_vorlap": row["equivalent_cycles_vorlap"],
                "smoke_only": smoke_only,
                "postprocess_warnings": warnings,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["wind_speed_mps", "azimuth_deg", "position", "quantity", "axis"]
    )


def baseline_identity_resultant_table(metrics: pd.DataFrame, *, include_smoke: bool = False) -> pd.DataFrame:
    """Compare no-SIL baseline against bridge-active zero-force identity."""
    table = comparison_table(metrics, "baseline", "identity", include_smoke=include_smoke)
    return table.rename(
        columns={
            "force_del_vorlap_N": "force_del_identity_N",
            "force_delta_pct": "force_identity_delta_pct",
            "moment_del_vorlap_Nm": "moment_del_identity_Nm",
            "moment_delta_pct": "moment_identity_delta_pct",
            "equivalent_cycles_vorlap": "equivalent_cycles_identity",
        }
    )


def baseline_identity_signed_table(metrics: pd.DataFrame, *, include_smoke: bool = False) -> pd.DataFrame:
    """Compare signed channels for no-SIL baseline and zero-force bridge identity."""
    table = signed_comparison_table(metrics, "baseline", "identity", include_smoke=include_smoke)
    return table.rename(
        columns={
            "channel_vorlap": "channel_identity",
            "signed_del_vorlap": "signed_del_identity",
            "signed_delta_pct": "signed_identity_delta_pct",
            "rainflow_cycles_vorlap": "rainflow_cycles_identity",
            "equivalent_cycles_vorlap": "equivalent_cycles_identity",
        }
    )


def extreme_channel_table(signed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return signed-channel extrema for ultimate-load screening inputs.

    These are load or moment extrema, not material utilization ratios. Optional
    reserve factors require a documented user-supplied allowable map.
    """
    columns = [
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
        "min",
        "max",
        "max_abs",
        "samples",
        "duration_s",
        "smoke_only",
        "postprocess_warnings",
        "result_file",
    ]
    if signed_metrics.empty:
        return pd.DataFrame(columns=columns)
    table = signed_metrics.copy()
    table["max_abs"] = np.maximum(np.abs(table["min"].astype(float)), np.abs(table["max"].astype(float)))
    return table[columns].sort_values(["scenario", "wind_speed_mps", "azimuth_deg", "max_abs"], ascending=[True, True, True, False])


ULTIMATE_ALLOWABLE_TEMPLATE_COLUMNS = [
    "blade",
    "position",
    "quantity",
    "axis",
    "unit",
    "allowable_abs",
    "allowable_label",
    "load_factor",
    "resistance_factor",
    "notes",
]


ULTIMATE_MARGIN_COLUMNS = [
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
    "max_abs",
    "allowable_abs",
    "allowable_label",
    "load_factor",
    "resistance_factor",
    "factored_demand",
    "factored_allowable",
    "utilization",
    "reserve_factor",
    "ultimate_status",
    "postprocess_warnings",
    "result_file",
]


def ultimate_allowables_template(extreme_channels: pd.DataFrame) -> pd.DataFrame:
    """Write one fill-in row per monitored signed channel for ultimate checks."""
    if extreme_channels.empty:
        return pd.DataFrame(columns=ULTIMATE_ALLOWABLE_TEMPLATE_COLUMNS)
    keys = ["blade", "position", "quantity", "axis", "unit"]
    table = extreme_channels[keys].copy()
    table["position"] = pd.to_numeric(table["position"], errors="coerce")
    table = table.drop_duplicates().sort_values(keys).reset_index(drop=True)
    table["allowable_abs"] = ""
    table["allowable_label"] = ""
    table["load_factor"] = 1.0
    table["resistance_factor"] = 1.0
    table["notes"] = ""
    return table[ULTIMATE_ALLOWABLE_TEMPLATE_COLUMNS]


def load_ultimate_allowables(path: Path | None) -> pd.DataFrame:
    """Read optional signed-channel ultimate allowables."""
    if path is None:
        return pd.DataFrame(columns=ULTIMATE_ALLOWABLE_TEMPLATE_COLUMNS)
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    missing = [column for column in ("blade", "position", "quantity", "axis", "unit", "allowable_abs") if column not in table.columns]
    if missing:
        raise ValueError(f"Ultimate allowables file is missing required column(s): {', '.join(missing)}")
    for column in ULTIMATE_ALLOWABLE_TEMPLATE_COLUMNS:
        if column not in table.columns:
            table[column] = "" if column in {"allowable_label", "notes"} else 1.0
    return table[ULTIMATE_ALLOWABLE_TEMPLATE_COLUMNS]


def _ultimate_key_frame(table: pd.DataFrame) -> pd.DataFrame:
    keyed = table.copy()
    keyed["position"] = pd.to_numeric(keyed["position"], errors="coerce")
    keyed["_position_key"] = keyed["position"].round(6)
    for column in ("blade", "quantity", "axis", "unit"):
        keyed[column] = keyed[column].astype(str)
    return keyed


def ultimate_margin_table(extreme_channels: pd.DataFrame, allowables: pd.DataFrame) -> pd.DataFrame:
    """Compute optional utilization/reserve factors from signed-channel extrema.

    The calculation is intentionally simple: factored demand is
    `load_factor * max_abs`; factored allowable is
    `resistance_factor * allowable_abs`. The result is only as meaningful as the
    user-supplied mapping from local QBlade load channels to material allowables.
    """
    if extreme_channels.empty:
        return pd.DataFrame(columns=ULTIMATE_MARGIN_COLUMNS)

    demands = _ultimate_key_frame(extreme_channels)
    if allowables.empty:
        merged = demands.copy()
        merged["allowable_abs"] = math.nan
        merged["allowable_label"] = ""
        merged["load_factor"] = 1.0
        merged["resistance_factor"] = 1.0
        merged["notes"] = "missing allowable"
    else:
        allowable_keys = ["blade", "_position_key", "quantity", "axis", "unit"]
        allow = _ultimate_key_frame(allowables)
        duplicates = allow.duplicated(allowable_keys, keep=False)
        if duplicates.any():
            duplicate_rows = allow.loc[duplicates, allowable_keys].drop_duplicates()
            preview = duplicate_rows.head(5).to_dict(orient="records")
            raise ValueError(f"Ultimate allowables contain duplicate channel keys: {preview}")
        merged = demands.merge(
            allow.drop(columns=["position"]),
            on=allowable_keys,
            how="left",
            suffixes=("", "_allowable"),
        )

    for column, default in (("allowable_abs", math.nan), ("load_factor", 1.0), ("resistance_factor", 1.0)):
        merged[column] = pd.to_numeric(merged.get(column, default), errors="coerce")
    merged["load_factor"] = merged["load_factor"].fillna(1.0)
    merged["resistance_factor"] = merged["resistance_factor"].fillna(1.0)
    if "allowable_label" in merged.columns:
        merged["allowable_label"] = merged["allowable_label"].fillna("").astype(str)
    else:
        merged["allowable_label"] = ""

    max_abs = pd.to_numeric(merged["max_abs"], errors="coerce")
    allowable = pd.to_numeric(merged["allowable_abs"], errors="coerce")
    load_factor = pd.to_numeric(merged["load_factor"], errors="coerce")
    resistance_factor = pd.to_numeric(merged["resistance_factor"], errors="coerce")
    merged["factored_demand"] = max_abs * load_factor
    merged["factored_allowable"] = allowable * resistance_factor
    merged["utilization"] = np.where(
        (merged["factored_allowable"] > 0.0) & np.isfinite(merged["factored_allowable"]),
        merged["factored_demand"] / merged["factored_allowable"],
        np.nan,
    )
    merged["reserve_factor"] = np.where(
        (merged["factored_demand"] > 0.0) & np.isfinite(merged["factored_demand"]),
        merged["factored_allowable"] / merged["factored_demand"],
        math.inf,
    )

    statuses = []
    warnings = []
    if "postprocess_warnings" in merged.columns:
        source_warnings = merged["postprocess_warnings"].fillna("").astype(str).tolist()
    else:
        source_warnings = [""] * len(merged)
    for source_warning, (_, row) in zip(source_warnings, merged.iterrows()):
        row_warnings = [item for item in [source_warning] if item]
        demand = float(row["factored_demand"]) if np.isfinite(row["factored_demand"]) else math.nan
        row_allowable = float(row["factored_allowable"]) if np.isfinite(row["factored_allowable"]) else math.nan
        if not math.isfinite(demand) or demand < 0.0:
            status = "invalid_demand"
            row_warnings.append("invalid ultimate demand")
        elif not math.isfinite(float(row["allowable_abs"])) or float(row["allowable_abs"]) <= 0.0:
            status = "missing_allowable"
            row_warnings.append("missing positive allowable_abs")
        elif float(row["load_factor"]) <= 0.0 or float(row["resistance_factor"]) <= 0.0:
            status = "invalid_factor"
            row_warnings.append("load_factor and resistance_factor must be positive")
        elif not math.isfinite(row_allowable) or row_allowable <= 0.0:
            status = "invalid_allowable"
            row_warnings.append("invalid factored allowable")
        elif float(row["utilization"]) > 1.0:
            status = "exceeds_allowable"
        else:
            status = "ok"
        statuses.append(status)
        warnings.append(_warning_text(row_warnings))

    merged["ultimate_status"] = statuses
    merged["postprocess_warnings"] = warnings
    for column in ULTIMATE_MARGIN_COLUMNS:
        if column not in merged.columns:
            merged[column] = "" if column in {"allowable_label", "ultimate_status", "postprocess_warnings", "result_file"} else math.nan
    return merged[ULTIMATE_MARGIN_COLUMNS].sort_values(
        ["ultimate_status", "scenario", "wind_speed_mps", "azimuth_deg", "utilization"],
        ascending=[True, True, True, True, False],
    )


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\hline",
        r"Case & $U$ & $\psi$ & $s/R$ & $DEL_F^0$ & $DEL_F^{VL}$ & $\Delta F$ & $DEL_M^0$ & $DEL_M^{VL}$ & $\Delta M$ \\",
        r"{} & {[m/s]} & {[deg]} & {[-]} & {[N]} & {[N]} & {[\%]} & {[Nm]} & {[Nm]} & {[\%]} \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{latex_escape(row['case'])} & {row['wind_speed_mps']:.1f} & {row['azimuth_deg']:.1f} & {row['position']:.3f} & "
            f"{row['force_del_baseline_N']:.3g} & {row['force_del_vorlap_N']:.3g} & {row['force_delta_pct']:.2f} & "
            f"{row['moment_del_baseline_Nm']:.3g} & {row['moment_del_vorlap_Nm']:.3g} & {row['moment_delta_pct']:.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_signed_latex_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\scriptsize",
        r"\begin{tabular}{lrrrlrrrr}",
        r"\hline",
        r"Case & $U$ & $\psi$ & $s/R$ & Channel & $DEL^0$ & $DEL^{VL}$ & $\Delta$ & Cycles \\",
        r"{} & {[m/s]} & {[deg]} & {[-]} & {} & {} & {} & {[\%]} & {[--]} \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        channel = f"{row['quantity']} {row['axis']} [{row['unit']}]"
        cycles = min(float(row["rainflow_cycles_baseline"]), float(row["rainflow_cycles_vorlap"]))
        lines.append(
            f"{latex_escape(row['case'])} & {row['wind_speed_mps']:.1f} & {row['azimuth_deg']:.1f} & "
            f"{row['position']:.3f} & {latex_escape(channel)} & "
            f"{row['signed_del_baseline']:.3g} & {row['signed_del_vorlap']:.3g} & "
            f"{row['signed_delta_pct']:.2f} & {cycles:.1f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_completion_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\scriptsize",
        r"\begin{tabular}{llrrrrl}",
        r"\hline",
        r"Scenario & Case & $U$ & $\psi$ & Steps & $t_{used}$ & Status \\",
        r"{} & {} & {[m/s]} & {[deg]} & {[--]} & {[s]} & {} \\",
        r"\hline",
    ]
    for _, row in table.sort_values(["scenario", "wind_speed_mps", "azimuth_deg"]).iterrows():
        status = "partial" if bool(row["failed"]) else "complete"
        if bool(row.get("smoke_only", False)):
            status = f"{status}; smoke-only"
        if row["failure_message"]:
            status = f"{status}: {row['failure_message']}"
        warnings = str(row.get("postprocess_warnings", ""))
        if warnings:
            status = f"{status}; {warnings}"
        if not bool(row["exported_results"]):
            status = f"{status}; not exported"
        lines.append(
            f"{latex_escape(row['scenario'])} & {latex_escape(row['case'])} & "
            f"{row['wind_speed_mps']:.1f} & {row['azimuth_deg']:.1f} & "
            f"{int(row['completed_steps'])}/{int(row['requested_steps'])} & "
            f"{row['completed_time_s']:.2f} & {latex_escape(status)} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_histories(cases: list[CaseResult], locations: list[float], blade: str, out_dir: Path) -> list[Path]:
    plots: list[Path] = []
    by_label: dict[str, list[CaseResult]] = {}
    for case in cases:
        by_label.setdefault(case.label, []).append(case)

    for label, label_cases in sorted(by_label.items()):
        rows = len(locations)
        fig, axes = plt.subplots(rows, 1, figsize=(8.0, max(2.2 * rows, 3.0)), sharex=True)
        axes_array = np.atleast_1d(axes)
        for axis, location in zip(axes_array, locations):
            for case_index, case in enumerate(sorted(label_cases, key=lambda item: item.scenario)):
                time = case.data["Time~[s]"].to_numpy(float)
                try:
                    _, cols = find_vector_columns(
                        list(case.data.columns),
                        blade=blade,
                        position=location,
                        quantity="For.",
                        unit="N",
                    )
                except KeyError as exc:
                    print(f"WARNING: plot skip {case.scenario} {case.label} location {location}: {exc}", file=sys.stderr)
                    continue
                force = vector_resultant(case.data, cols)
                color = COLOR_CYCLE[case_index % len(COLOR_CYCLE)]
                is_coupled = case.scenario in {"sil_vorlap", "vorlap"} or "sil" in case.scenario.lower()
                if is_coupled:
                    axis.plot(
                        time,
                        force,
                        color=lighten_hex_color(color, 0.24),
                        linestyle="--",
                        marker="o",
                        markevery=comparison_markevery(len(time)),
                        markersize=2.8,
                        markerfacecolor="white",
                        markeredgewidth=0.8,
                        alpha=0.86,
                        linewidth=1.2,
                        zorder=3,
                        label=case.scenario,
                    )
                else:
                    axis.plot(time, force, color=color, linewidth=1.35, zorder=2, label=case.scenario)
            axis.set_ylabel(f"|F| pos {location:.3f} [N]")
            axis.grid(True, color="0.86", linewidth=0.6)
        axes_array[-1].set_xlabel("Time [s]")
        axes_array[0].legend(loc="best")
        fig.suptitle(f"Blade force resultants, {label}")
        fig.tight_layout()
        plot_path = out_dir / f"force_history_{label}.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        plots.append(plot_path)
    return plots


def plot_del_comparison(table: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    if table.empty:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No paper-grade paired comparison cases",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    else:
        for group_index, ((position, azimuth), group) in enumerate(table.groupby(["position", "azimuth_deg"])):
            ordered = group.sort_values("wind_speed_mps")
            label = f"{position:.3f}, psi={azimuth:.0f}"
            color = COLOR_CYCLE[group_index % len(COLOR_CYCLE)]
            marker_every = comparison_markevery(len(ordered))
            axes[0].plot(ordered["wind_speed_mps"], ordered["force_del_vorlap_N"], "-", color=color, linewidth=1.4, zorder=2, label=f"{label} VL")
            axes[0].plot(
                ordered["wind_speed_mps"],
                ordered["force_del_baseline_N"],
                "--",
                color=lighten_hex_color(color, 0.34),
                marker="o",
                markevery=marker_every,
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.8,
                alpha=0.86,
                linewidth=1.0,
                zorder=3,
                label=f"{label} base",
            )
            axes[1].plot(ordered["wind_speed_mps"], ordered["moment_del_vorlap_Nm"], "-", color=color, linewidth=1.4, zorder=2, label=f"{label} VL")
            axes[1].plot(
                ordered["wind_speed_mps"],
                ordered["moment_del_baseline_Nm"],
                "--",
                color=lighten_hex_color(color, 0.34),
                marker="o",
                markevery=marker_every,
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.8,
                alpha=0.86,
                linewidth=1.0,
                zorder=3,
                label=f"{label} base",
            )
    axes[0].set_ylabel("Force DEL [N]")
    axes[1].set_ylabel("Moment DEL [Nm]")
    for axis in axes:
        axis.set_xlabel("Wind speed [m/s]")
        axis.grid(True, color="0.86", linewidth=0.6)
    if not table.empty:
        axes[1].legend(loc="best", fontsize=7)
    fig.tight_layout()
    path = out_dir / "del_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _pivot_for_heatmap(table: pd.DataFrame, value_column: str, *, reducer: str = "max") -> pd.DataFrame:
    if table.empty or value_column not in table.columns:
        return pd.DataFrame()
    if "wind_speed_mps" not in table.columns or "azimuth_deg" not in table.columns:
        return pd.DataFrame()
    values = table.copy()
    values["wind_speed_mps"] = pd.to_numeric(values["wind_speed_mps"], errors="coerce")
    values["azimuth_deg"] = pd.to_numeric(values["azimuth_deg"], errors="coerce")
    values[value_column] = pd.to_numeric(values[value_column], errors="coerce")
    values = values[np.isfinite(values["wind_speed_mps"]) & np.isfinite(values["azimuth_deg"])]
    if values.empty:
        return pd.DataFrame()
    if reducer == "min":
        aggregate = "min"
    elif reducer == "mean":
        aggregate = "mean"
    else:
        aggregate = "max"
    pivot = values.pivot_table(
        index="azimuth_deg",
        columns="wind_speed_mps",
        values=value_column,
        aggfunc=aggregate,
    )
    return pivot.sort_index().sort_index(axis=1)


def _draw_heatmap(
    axis,
    pivot: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if pivot.empty:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel("Wind speed [m/s]")
        axis.set_ylabel("Parked azimuth [deg]")
        return
    image = axis.imshow(pivot.to_numpy(float), origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("Wind speed [m/s]")
    axis.set_ylabel("Parked azimuth [deg]")
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels([f"{float(value):g}" for value in pivot.columns], rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels([f"{float(value):g}" for value in pivot.index])
    colorbar = axis.figure.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label(colorbar_label)


def plot_signed_del_heatmap(table: pd.DataFrame, out_dir: Path) -> Path:
    """Plot the max signed DEL ratio over monitored channels for each parked state."""
    path = out_dir / "signed_del_heatmap.png"
    plot_table = table.copy()
    if not plot_table.empty:
        baseline = plot_table["signed_del_baseline"].astype(float)
        coupled = plot_table["signed_del_vorlap"].astype(float)
        plot_table["signed_del_ratio"] = np.where(baseline > 0.0, coupled / baseline, np.nan)
    pivot = _pivot_for_heatmap(plot_table, "signed_del_ratio")
    finite_values = pivot.to_numpy(float)
    finite_values = finite_values[np.isfinite(finite_values)]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    _draw_heatmap(
        axis,
        pivot,
        title="Max signed-channel DEL ratio",
        colorbar_label="DEL ratio [-]",
        cmap="magma",
        vmin=1.0 if finite_values.size and float(np.max(finite_values)) >= 1.0 else None,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_completion_heatmap(completion: pd.DataFrame, out_dir: Path) -> Path:
    """Plot completed fraction by scenario, wind speed, and parked azimuth."""
    path = out_dir / "completion_heatmap.png"
    scenarios = sorted(completion["scenario"].dropna().unique()) if not completion.empty and "scenario" in completion else []
    if not scenarios:
        scenarios = [""]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.8), squeeze=False)
    for axis, scenario in zip(axes[0], scenarios):
        if completion.empty:
            subset = pd.DataFrame()
        else:
            subset = completion[completion["scenario"] == scenario].copy()
            requested = subset["requested_steps"].astype(float)
            completed = subset["completed_steps"].astype(float)
            subset["completed_fraction"] = np.where(requested > 0.0, completed / requested, np.nan)
        pivot = _pivot_for_heatmap(subset, "completed_fraction", reducer="min")
        _draw_heatmap(
            axis,
            pivot,
            title=f"Completion: {scenario}" if scenario else "Completion",
            colorbar_label="Completed fraction [-]",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def source_coverage_case_summary(source_coverage: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "min_nyquist_ratio",
        "max_force_coefficient_ratio",
        "max_aoa_margin_deg",
        "max_re_margin",
        "max_nodal_force_n",
        "max_structural_velocity_mps",
        "force_power_max_abs_w",
        "re_lower_bound_warning_count",
        "violation_count",
    ]
    if source_coverage.empty:
        return pd.DataFrame(columns=columns)
    table = source_coverage.copy()
    if "re_lower_bound_warning_count" not in table.columns:
        table["re_lower_bound_warning_count"] = 0
    for column in [
        "nyquist_ratio",
        "force_coefficient_max_abs",
        "source_force_coefficient_max_abs",
        "aoa_min_deg",
        "aoa_max_deg",
        "source_aoa_min_deg",
        "source_aoa_max_deg",
        "re_min",
        "re_max",
        "source_re_min",
        "source_re_max",
        "max_nodal_force_n",
        "max_structural_velocity_mps",
        "force_power_max_abs_w",
        "re_lower_bound_warning_count",
        *SOURCE_COVERAGE_COUNT_COLUMNS_FOR_POSTPROCESS,
    ]:
        if column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    denominator = table["source_force_coefficient_max_abs"].replace(0.0, np.nan)
    table["force_coefficient_ratio"] = table["force_coefficient_max_abs"] / denominator
    table["aoa_margin_deg"] = np.maximum(
        table["source_aoa_min_deg"] - table["aoa_min_deg"],
        table["aoa_max_deg"] - table["source_aoa_max_deg"],
    )
    table["re_margin"] = np.maximum(
        table["source_re_min"] - table["re_min"],
        table["re_max"] - table["source_re_max"],
    )
    count_cols = [column for column in SOURCE_COVERAGE_COUNT_COLUMNS_FOR_POSTPROCESS if column in table.columns]
    table["violation_count"] = table[count_cols].sum(axis=1) if count_cols else 0
    grouped = table.groupby(["scenario", "case", "wind_speed_mps", "azimuth_deg"], dropna=False)
    summary = grouped.agg(
        min_nyquist_ratio=("nyquist_ratio", "min"),
        max_force_coefficient_ratio=("force_coefficient_ratio", "max"),
        max_aoa_margin_deg=("aoa_margin_deg", "max"),
        max_re_margin=("re_margin", "max"),
        max_nodal_force_n=("max_nodal_force_n", "max"),
        max_structural_velocity_mps=("max_structural_velocity_mps", "max"),
        force_power_max_abs_w=("force_power_max_abs_w", "max"),
        re_lower_bound_warning_count=("re_lower_bound_warning_count", "sum"),
        violation_count=("violation_count", "sum"),
    )
    return summary.reset_index()[columns].sort_values(["scenario", "wind_speed_mps", "azimuth_deg", "case"])


SOURCE_COVERAGE_COUNT_COLUMNS_FOR_POSTPROCESS = [
    "re_out_of_range_count",
    "aoa_out_of_range_count",
    "aliased_frequency_count",
    "source_clip_count",
    "force_coefficient_out_of_range_count",
    "nonfinite_count",
]


def plot_source_coverage_sanity(summary: pd.DataFrame, out_dir: Path) -> Path:
    """Plot campaign source-coverage margins used for paper evidence review."""
    path = out_dir / "source_coverage_sanity.png"
    scenario = "vorlap" if "vorlap" in set(summary.get("scenario", [])) else None
    plot_table = summary if scenario is None else summary[summary["scenario"] == scenario]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    panels = [
        ("min_nyquist_ratio", "Min Nyquist ratio", "ratio [-]", "viridis", None, None, "min"),
        ("max_force_coefficient_ratio", "Max coefficient/source ratio", "ratio [-]", "magma", 0.0, None, "max"),
        ("max_aoa_margin_deg", "Max AOA source-margin exceedance", "deg", "coolwarm", None, None, "max"),
        ("violation_count", "Source-coverage violation count", "count [-]", "plasma", 0.0, None, "max"),
    ]
    for axis, (column, title, label, cmap, vmin, vmax, reducer) in zip(axes.ravel(), panels):
        pivot = _pivot_for_heatmap(plot_table, column, reducer=reducer)
        _draw_heatmap(axis, pivot, title=title, colorbar_label=label, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


CRITICAL_CASE_COLUMNS = [
    "case",
    "wind_speed_mps",
    "azimuth_deg",
    "position",
    "quantity",
    "axis",
    "unit",
    "channel_baseline",
    "channel_vorlap",
    "signed_del_baseline",
    "signed_del_vorlap",
    "signed_del_ratio",
    "signed_delta_pct",
    "rainflow_cycles_baseline",
    "rainflow_cycles_vorlap",
    "equivalent_cycles_baseline",
    "equivalent_cycles_vorlap",
    "postprocess_warnings",
    "selection_reason",
    "min_nyquist_ratio",
    "max_force_coefficient_ratio",
    "force_power_max_abs_w",
    "source_coverage_violation_count",
]


def critical_case_summary(signed_table: pd.DataFrame, source_summary: pd.DataFrame) -> pd.DataFrame:
    """Select the paper-facing critical signed-channel case for deep-dive plots."""
    if signed_table.empty:
        return pd.DataFrame(columns=CRITICAL_CASE_COLUMNS)
    table = signed_table.copy()
    table["signed_del_baseline"] = pd.to_numeric(table["signed_del_baseline"], errors="coerce")
    table["signed_del_vorlap"] = pd.to_numeric(table["signed_del_vorlap"], errors="coerce")
    table = table[np.isfinite(table["signed_del_vorlap"])]
    if table.empty:
        return pd.DataFrame(columns=CRITICAL_CASE_COLUMNS)
    table["signed_del_ratio"] = np.where(
        table["signed_del_baseline"] > 0.0,
        table["signed_del_vorlap"] / table["signed_del_baseline"],
        np.nan,
    )
    finite_ratio = table[np.isfinite(table["signed_del_ratio"])]
    if not finite_ratio.empty:
        selected = finite_ratio.sort_values(["signed_del_ratio", "signed_del_vorlap"], ascending=[False, False]).iloc[0].copy()
        selected["selection_reason"] = "max_signed_del_ratio"
    else:
        selected = table.sort_values("signed_del_vorlap", ascending=False).iloc[0].copy()
        selected["signed_del_ratio"] = math.nan
        selected["selection_reason"] = "max_signed_del_vorlap_no_finite_baseline_ratio"

    if not source_summary.empty:
        source = source_summary.copy()
        for column in ("wind_speed_mps", "azimuth_deg"):
            source[column] = pd.to_numeric(source[column], errors="coerce")
        mask = (
            (source["case"].astype(str) == str(selected["case"]))
            & np.isclose(source["wind_speed_mps"].astype(float), float(selected["wind_speed_mps"]), rtol=1.0e-9, atol=1.0e-9)
            & np.isclose(source["azimuth_deg"].astype(float), float(selected["azimuth_deg"]), rtol=1.0e-9, atol=1.0e-9)
        )
        matched = source[mask]
        if not matched.empty:
            source_row = matched.iloc[0]
            selected["min_nyquist_ratio"] = source_row.get("min_nyquist_ratio", math.nan)
            selected["max_force_coefficient_ratio"] = source_row.get("max_force_coefficient_ratio", math.nan)
            selected["force_power_max_abs_w"] = source_row.get("force_power_max_abs_w", math.nan)
            selected["source_coverage_violation_count"] = source_row.get("violation_count", math.nan)

    for column in CRITICAL_CASE_COLUMNS:
        if column not in selected:
            selected[column] = "" if column in {"case", "quantity", "axis", "unit", "channel_baseline", "channel_vorlap", "postprocess_warnings", "selection_reason"} else math.nan
    return pd.DataFrame([{column: selected[column] for column in CRITICAL_CASE_COLUMNS}], columns=CRITICAL_CASE_COLUMNS)


def _case_lookup(cases: list[CaseResult], scenario: str, label: str) -> CaseResult | None:
    for case in cases:
        if case.scenario == scenario and case.label == label:
            return case
    return None


def _channel_time_series(case: CaseResult | None, channel: str) -> tuple[np.ndarray, np.ndarray]:
    if case is None or not channel or channel not in case.data.columns or "Time~[s]" not in case.data.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    time = case.data["Time~[s]"].to_numpy(float)
    values = case.data[channel].to_numpy(float)
    mask = np.isfinite(time) & np.isfinite(values)
    time = time[mask]
    values = values[mask]
    if time.size:
        order = np.argsort(time)
        time = time[order]
        values = values[order]
    return time, values


def _psd(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if time.size < 3 or values.size < 3:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    sample_period = float(np.median(dt))
    if sample_period <= 0.0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    centered = values - float(np.mean(values))
    freq = np.fft.rfftfreq(centered.size, d=sample_period)
    spectrum = np.fft.rfft(centered)
    sample_rate = 1.0 / sample_period
    power = (np.abs(spectrum) ** 2) / (sample_rate * centered.size)
    if power.size > 2:
        power[1:-1] *= 2.0
    return freq, power


def _plot_series(
    axis,
    time: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    linewidth: float = 1.1,
    color: str | None = None,
    linestyle: str = "-",
    marker: str | None = None,
    alpha: float = 1.0,
    zorder: int = 2,
) -> None:
    if time.size and values.size:
        axis.plot(
            time,
            values,
            linewidth=linewidth,
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=comparison_markevery(len(time)) if marker else None,
            markersize=2.8 if marker else None,
            markerfacecolor="white" if marker else None,
            markeredgewidth=0.8 if marker else None,
            alpha=alpha,
            zorder=zorder,
        )


def plot_critical_case_diagnostic(
    cases: list[CaseResult],
    critical_summary: pd.DataFrame,
    out_dir: Path,
    *,
    comparison_baseline: str,
    comparison_coupled: str,
    wohler_exponent: float,
) -> tuple[Path, Path]:
    """Write critical signed-channel time, PSD, and rainflow diagnostic artifacts."""
    plot_path = out_dir / "critical_case_diagnostic.png"
    rainflow_path = out_dir / "critical_case_rainflow_ranges.csv"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    if critical_summary.empty:
        for axis in axes.ravel():
            axis.text(0.5, 0.5, "No paper-grade critical signed channel", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
        pd.DataFrame(columns=["range", "count"]).to_csv(rainflow_path, index=False)
    else:
        row = critical_summary.iloc[0]
        label = str(row["case"])
        baseline_case = _case_lookup(cases, comparison_baseline, label)
        coupled_case = _case_lookup(cases, comparison_coupled, label)
        channel_baseline = str(row.get("channel_baseline", ""))
        channel_coupled = str(row.get("channel_vorlap", ""))
        time_base, values_base = _channel_time_series(baseline_case, channel_baseline)
        time_coupled, values_coupled = _channel_time_series(coupled_case, channel_coupled)

        channel_label = f"{row['quantity']} {row['axis']} at s/R={float(row['position']):.3f} [{row['unit']}]"
        axes[0, 0].set_title("Critical signed-channel history")
        _plot_series(axes[0, 0], time_base, values_base, label=comparison_baseline, linewidth=1.0, color=COLOR_CYCLE[0], zorder=2)
        _plot_series(
            axes[0, 0],
            time_coupled,
            values_coupled,
            label=comparison_coupled,
            linewidth=1.2,
            color=lighten_hex_color(COLOR_CYCLE[1], 0.24),
            linestyle="--",
            marker="o",
            alpha=0.86,
            zorder=3,
        )
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel(channel_label)
        axes[0, 0].grid(True, color="0.86", linewidth=0.6)
        if time_base.size or time_coupled.size:
            axes[0, 0].legend(loc="best", fontsize=8)

        axes[0, 1].set_title("Signed-channel PSD")
        for index, (series_time, series_values, series_label) in enumerate(
            (
                (time_base, values_base, comparison_baseline),
                (time_coupled, values_coupled, comparison_coupled),
            )
        ):
            freq, power = _psd(series_time, series_values)
            if freq.size and power.size:
                if index == 0:
                    axes[0, 1].semilogy(freq, power, linewidth=1.0, color=COLOR_CYCLE[0], zorder=2, label=series_label)
                else:
                    axes[0, 1].semilogy(
                        freq,
                        power,
                        linewidth=1.0,
                        color=lighten_hex_color(COLOR_CYCLE[1], 0.24),
                        linestyle="--",
                        marker="o",
                        markevery=comparison_markevery(len(freq)),
                        markersize=2.5,
                        markerfacecolor="white",
                        markeredgewidth=0.7,
                        alpha=0.86,
                        zorder=3,
                        label=series_label,
                    )
        axes[0, 1].set_xlabel("Frequency [Hz]")
        axes[0, 1].set_ylabel(f"PSD [{row['unit']}^2/Hz]")
        axes[0, 1].grid(True, color="0.86", linewidth=0.6)
        if axes[0, 1].has_data():
            axes[0, 1].legend(loc="best", fontsize=8)

        ranges, counts = rainflow_ranges(values_coupled)
        pd.DataFrame({"range": ranges, "count": counts}).to_csv(rainflow_path, index=False)
        axes[1, 0].set_title(f"Rainflow ranges, m={wohler_exponent:g}")
        if ranges.size:
            axes[1, 0].hist(ranges, weights=counts, bins=min(max(6, int(np.sqrt(ranges.size))), 30), color="0.28", edgecolor="white")
        else:
            axes[1, 0].text(0.5, 0.5, "No counted ranges", ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_xlabel(f"Range [{row['unit']}]")
        axes[1, 0].set_ylabel("Cycle count [-]")
        axes[1, 0].grid(True, color="0.90", linewidth=0.5)

        axes[1, 1].set_axis_off()
        context = [
            f"Case: {label}",
            f"U={float(row['wind_speed_mps']):g} m/s, psi={float(row['azimuth_deg']):g} deg",
            f"Selected by: {row['selection_reason']}",
            f"DEL ratio: {float(row['signed_del_ratio']):.4g}" if math.isfinite(float(row["signed_del_ratio"])) else "DEL ratio: n/a",
            f"{comparison_coupled} DEL: {float(row['signed_del_vorlap']):.4g} {row['unit']}",
            f"Cycles: {float(row['rainflow_cycles_vorlap']):.4g}",
            f"Min Nyquist ratio: {float(row['min_nyquist_ratio']):.4g}" if math.isfinite(float(row["min_nyquist_ratio"])) else "Min Nyquist ratio: n/a",
            f"Max force power: {float(row['force_power_max_abs_w']):.4g} W" if math.isfinite(float(row["force_power_max_abs_w"])) else "Max force power: n/a",
            f"Source violations: {float(row['source_coverage_violation_count']):.0f}" if math.isfinite(float(row["source_coverage_violation_count"])) else "Source violations: n/a",
        ]
        warnings = str(row.get("postprocess_warnings", ""))
        if warnings:
            context.append(f"Warnings: {warnings}")
        axes[1, 1].text(0.02, 0.98, "\n".join(context), ha="left", va="top", transform=axes[1, 1].transAxes, family="monospace", fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path, rainflow_path


def write_report(
    out_dir: Path,
    history_plots: list[Path],
    del_plot: Path,
    review_plots: list[tuple[Path, str]],
    table_path: Path,
    completion_table_path: Path,
    rainflow_validation_path: Path,
    rainflow_validation: pd.DataFrame,
    completion: pd.DataFrame,
    metrics: pd.DataFrame,
    diagnostics_table_path: Path,
    signed_table_path: Path,
    signed_diagnostics_table_path: Path,
    comparison_baseline: str,
    comparison_coupled: str,
) -> Path:
    figure_lines = []
    for plot in history_plots:
        figure_lines.extend(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=0.92\linewidth]{{{plot.name}}}",
                rf"\caption{{Blade local force resultants for {latex_escape(plot.stem.replace('force_history_', ''))}.}}",
                r"\end{figure}",
                "",
            ]
        )
    review_figure_lines = []
    for plot, caption in review_plots:
        review_figure_lines.extend(
            [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width=0.92\linewidth]{{{plot.name}}}",
                rf"\caption{{{latex_escape(caption)}}}",
                r"\end{figure}",
                "",
            ]
        )
    any_partial = bool(completion["failed"].any()) if not completion.empty else False
    smoke_completion_count = int(completion["smoke_only"].sum()) if "smoke_only" in completion else 0
    smoke_metric_count = int(metrics["smoke_only"].sum()) if "smoke_only" in metrics else 0
    completion_note = (
        "QBlade reported structural divergence or NaN geometry before the requested duration in at least one case; "
        "the plots and damage-equivalent loads below use only the exported completed timesteps."
        if any_partial
        else "All listed cases completed the requested number of QBlade timesteps."
    )
    fatigue_quality_note = (
        f"The postprocessor marked {smoke_completion_count} completion rows and {smoke_metric_count} metric rows as "
        "smoke-only because they did not satisfy the configured sample, duration, or rainflow-cycle thresholds. "
        f"Smoke-only rows are excluded from the DEL table and plot below. The full diagnostic comparison is written to "
        f"{latex_escape(diagnostics_table_path.name)}."
    )
    signed_note = (
        "Signed local QBlade force and moment channels are postprocessed separately from vector resultants. "
        f"The paper-facing signed-channel comparison is written to {latex_escape(signed_table_path.name)}; "
        f"the full signed diagnostic table, including smoke-only or low-cycle rows, is written to "
        f"{latex_escape(signed_diagnostics_table_path.name)}."
    )
    validation_passed = bool(rainflow_validation["passed"].all()) if "passed" in rainflow_validation else False
    equivalent_cycles = (
        float(rainflow_validation["equivalent_cycles"].iloc[0])
        if not rainflow_validation.empty
        and "equivalent_cycles" in rainflow_validation
        else math.nan
    )
    rainflow_note = (
        "The local rainflow counter's complete range-count multisets were checked against frozen constant, "
        "closed-cycle, residual half-cycle, nested-reversal, and ASTM-style mixed histories, plus an independently "
        f"structured four-point reference counter. Non-finite samples are rejected. DEL uses a fixed equivalent-cycle count of "
        f"{equivalent_cycles:g}. The validation table is written to "
        f"{latex_escape(rainflow_validation_path.name)} and all rows passed."
        if validation_passed
        else "The local rainflow validation table contains a failed row; do not use these DEL products for fatigue claims."
    )
    report = "\n".join(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=0.8in]{geometry}",
            r"\usepackage{graphicx}",
            r"\usepackage{booktabs}",
            r"\usepackage{siunitx}",
            r"\title{Parked VorLap/QBlade Headless Campaign Summary}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            "This report is generated by the headless parked-load workflow. "
            f"It compares the {latex_escape(comparison_baseline)} scenario against the "
            f"{latex_escape(comparison_coupled)} scenario for damage-equivalent loads, while the completion table lists all loaded scenarios. "
            "The current generated case uses the stabilized support patch documented in the runner README.",
            "",
            r"\section*{Run Completion}",
            completion_note,
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\input{{{completion_table_path.name}}}",
            r"\caption{Requested and completed QBlade timesteps used for each exported result file.}",
            r"\end{table}",
            "",
            r"\section*{Damage Equivalent Load Summary}",
            fatigue_quality_note,
            "",
            r"\section*{Rainflow Validation}",
            rainflow_note,
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\input{{{table_path.name}}}",
            rf"\caption{{Rainflow damage-equivalent force and moment resultants for BLD\_1 stations. Superscript 0 is the {latex_escape(comparison_baseline)} comparison baseline and VL is {latex_escape(comparison_coupled)}.}}",
            r"\end{table}",
            "",
            r"\section*{Signed Channel DEL Summary}",
            signed_note,
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\input{{{signed_table_path.name}}}",
            rf"\caption{{Rainflow damage-equivalent signed local force and moment components for BLD\_1 stations. Superscript 0 is the {latex_escape(comparison_baseline)} comparison baseline and VL is {latex_escape(comparison_coupled)}.}}",
            r"\end{table}",
            "",
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=0.92\linewidth]{{{del_plot.name}}}",
            r"\caption{Damage-equivalent resultant force and moment trends with wind speed.}",
            r"\end{figure}",
            "",
            r"\section*{Campaign Review Figures}",
            *review_figure_lines,
            "",
            r"\section*{Force Histories}",
            *figure_lines,
            r"\end{document}",
            "",
        ]
    )
    report_path = out_dir / "parked_campaign_report.tex"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/vorlap_parked_demo"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--scenarios", default="baseline vorlap")
    parser.add_argument("--comparison-baseline", default=None)
    parser.add_argument("--comparison-coupled", default=None)
    parser.add_argument("--locations", default="0.230 0.500 0.770 0.940")
    parser.add_argument("--blade", default="BLD_1")
    parser.add_argument("--wohler-exponent", type=float, default=10.0)
    parser.add_argument(
        "--equivalent-cycles",
        type=float,
        default=DEFAULT_EQUIVALENT_CYCLES,
        help="Fixed reference cycle count used in every DEL denominator.",
    )
    parser.add_argument("--min-fatigue-samples", type=int, default=1000)
    parser.add_argument("--min-fatigue-duration", type=float, default=10.0)
    parser.add_argument("--min-rainflow-cycles", type=float, default=10.0)
    parser.add_argument(
        "--ultimate-allowables",
        type=Path,
        default=None,
        help=(
            "Optional CSV mapping signed local load channels to positive ultimate allowables. "
            "If omitted, the postprocessor still writes ultimate_allowables_template.csv."
        ),
    )
    parser.add_argument(
        "--include-smoke-in-del",
        action="store_true",
        help="Include smoke-only rows in del_comparison.csv, the LaTeX DEL table, and DEL plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or (run_root / "postprocess")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = parse_scenarios(args.scenarios)
    locations = parse_float_list(args.locations)
    thresholds = FatigueQualityThresholds(
        min_samples=args.min_fatigue_samples,
        min_duration_s=args.min_fatigue_duration,
        min_rainflow_cycles=args.min_rainflow_cycles,
    )

    settings = load_campaign_settings(run_root)
    required_result_columns = fatigue_required_columns(args.blade, locations)
    cases = load_campaign(
        run_root,
        scenarios,
        settings=settings,
        required_columns=required_result_columns,
    )
    available_scenarios = [scenario for scenario in scenarios if any(case.scenario == scenario for case in cases)]
    comparison_scenarios = (
        scenarios
        if args.comparison_baseline is not None or args.comparison_coupled is not None
        else available_scenarios
    )
    comparison_baseline, comparison_coupled = select_comparison_pair(
        comparison_scenarios,
        args.comparison_baseline,
        args.comparison_coupled,
    )
    timestep = float(settings.get("timestep_s", math.nan))
    completion = annotate_completion_quality(load_completion_summary(run_root, scenarios, timestep), thresholds)
    completion_path = out_dir / "completion_summary.csv"
    completion.to_csv(completion_path, index=False)
    completion_table_path = out_dir / "completion_table.tex"
    write_completion_table(completion, completion_table_path)

    rainflow_validation = rainflow_validation_table(
        args.wohler_exponent,
        args.equivalent_cycles,
    )
    rainflow_validation_path = out_dir / "rainflow_validation.csv"
    rainflow_validation.to_csv(rainflow_validation_path, index=False)
    if not bool(rainflow_validation["passed"].all()):
        raise RuntimeError(f"Rainflow validation failed; see {rainflow_validation_path}")

    source_coverage = load_source_coverage(run_root, scenarios)
    source_coverage_path = out_dir / "source_coverage.csv"
    source_coverage.to_csv(source_coverage_path, index=False)

    source_coverage_summary = source_coverage_case_summary(source_coverage)
    source_coverage_summary_path = out_dir / "source_coverage_case_summary.csv"
    source_coverage_summary.to_csv(source_coverage_summary_path, index=False)

    metrics = compute_metrics(
        cases,
        locations,
        args.blade,
        args.wohler_exponent,
        thresholds,
        args.equivalent_cycles,
    )
    metrics_path = out_dir / "summary_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    signed_metrics = compute_signed_channel_metrics(
        cases,
        locations,
        args.blade,
        args.wohler_exponent,
        thresholds,
        args.equivalent_cycles,
    )
    signed_metrics_path = out_dir / "signed_channel_metrics.csv"
    signed_metrics.to_csv(signed_metrics_path, index=False)

    extreme_channels = extreme_channel_table(signed_metrics)
    extreme_channels_path = out_dir / "extreme_channel_metrics.csv"
    extreme_channels.to_csv(extreme_channels_path, index=False)

    ultimate_template = ultimate_allowables_template(extreme_channels)
    ultimate_template_path = out_dir / "ultimate_allowables_template.csv"
    ultimate_template.to_csv(ultimate_template_path, index=False)

    ultimate_allowables = load_ultimate_allowables(None if args.ultimate_allowables is None else args.ultimate_allowables.resolve())
    ultimate_margins = ultimate_margin_table(extreme_channels, ultimate_allowables)
    ultimate_margins_path = out_dir / "ultimate_margin_metrics.csv"
    ultimate_margins.to_csv(ultimate_margins_path, index=False)

    diagnostics_table = comparison_table(metrics, comparison_baseline, comparison_coupled, include_smoke=True)
    diagnostics_comparison_path = out_dir / "del_comparison_diagnostics.csv"
    diagnostics_table.to_csv(diagnostics_comparison_path, index=False)

    signed_diagnostics_table = signed_comparison_table(
        signed_metrics,
        comparison_baseline,
        comparison_coupled,
        include_smoke=True,
    )
    signed_diagnostics_path = out_dir / "signed_del_comparison_diagnostics.csv"
    signed_diagnostics_table.to_csv(signed_diagnostics_path, index=False)

    identity_diagnostics = baseline_identity_resultant_table(metrics, include_smoke=True)
    identity_diagnostics_path = out_dir / "baseline_identity_comparison_diagnostics.csv"
    identity_diagnostics.to_csv(identity_diagnostics_path, index=False)

    signed_identity_diagnostics = baseline_identity_signed_table(signed_metrics, include_smoke=True)
    signed_identity_diagnostics_path = out_dir / "signed_baseline_identity_comparison_diagnostics.csv"
    signed_identity_diagnostics.to_csv(signed_identity_diagnostics_path, index=False)

    table = comparison_table(metrics, comparison_baseline, comparison_coupled, include_smoke=args.include_smoke_in_del)
    comparison_path = out_dir / "del_comparison.csv"
    table.to_csv(comparison_path, index=False)
    latex_table_path = out_dir / "del_table.tex"
    write_latex_table(table, latex_table_path)

    identity_table = baseline_identity_resultant_table(metrics, include_smoke=args.include_smoke_in_del)
    identity_comparison_path = out_dir / "baseline_identity_comparison.csv"
    identity_table.to_csv(identity_comparison_path, index=False)

    signed_table = signed_comparison_table(
        signed_metrics,
        comparison_baseline,
        comparison_coupled,
        include_smoke=args.include_smoke_in_del,
    )
    signed_comparison_path = out_dir / "signed_del_comparison.csv"
    signed_table.to_csv(signed_comparison_path, index=False)
    signed_latex_table_path = out_dir / "signed_del_table.tex"
    write_signed_latex_table(signed_table, signed_latex_table_path)

    signed_identity_table = baseline_identity_signed_table(signed_metrics, include_smoke=args.include_smoke_in_del)
    signed_identity_comparison_path = out_dir / "signed_baseline_identity_comparison.csv"
    signed_identity_table.to_csv(signed_identity_comparison_path, index=False)

    critical_summary = critical_case_summary(signed_table, source_coverage_summary)
    critical_summary_path = out_dir / "critical_case_summary.csv"
    critical_summary.to_csv(critical_summary_path, index=False)

    history_plots = plot_histories(cases, locations, args.blade, out_dir)
    del_plot = plot_del_comparison(table, out_dir)
    signed_heatmap = plot_signed_del_heatmap(signed_table, out_dir)
    completion_heatmap = plot_completion_heatmap(completion, out_dir)
    source_sanity_plot = plot_source_coverage_sanity(source_coverage_summary, out_dir)
    critical_plot, critical_rainflow_path = plot_critical_case_diagnostic(
        cases,
        critical_summary,
        out_dir,
        comparison_baseline=comparison_baseline,
        comparison_coupled=comparison_coupled,
        wohler_exponent=args.wohler_exponent,
    )
    review_plots = [
        (completion_heatmap, "Completion fraction by scenario, wind speed, and parked azimuth."),
        (signed_heatmap, "Maximum signed-channel DEL ratio by wind speed and parked azimuth."),
        (source_sanity_plot, "Source-coverage and sampling sanity diagnostics by wind speed and parked azimuth."),
        (critical_plot, "Critical signed-channel time-history, spectrum, rainflow, and source-coverage diagnostic."),
    ]
    report_path = write_report(
        out_dir,
        history_plots,
        del_plot,
        review_plots,
        latex_table_path,
        completion_table_path,
        rainflow_validation_path,
        rainflow_validation,
        completion,
        metrics,
        diagnostics_comparison_path,
        signed_latex_table_path,
        signed_diagnostics_path,
        comparison_baseline,
        comparison_coupled,
    )

    postprocess_manifest_path = out_dir / "postprocess_manifest.json"
    write_raw_postprocess_manifest(
        postprocess_manifest_path,
        cases,
        scenarios,
        comparison_baseline,
        comparison_coupled,
        args.wohler_exponent,
        args.equivalent_cycles,
        thresholds,
        [signed_comparison_path, extreme_channels_path, source_coverage_path],
    )

    print(
        json.dumps(
            {
                "completion": str(completion_path),
                "completion_table": str(completion_table_path),
                "rainflow_validation": str(rainflow_validation_path),
                "equivalent_cycles": args.equivalent_cycles,
                "metrics": str(metrics_path),
                "signed_metrics": str(signed_metrics_path),
                "extreme_channel_metrics": str(extreme_channels_path),
                "ultimate_allowables_template": str(ultimate_template_path),
                "ultimate_allowables": None if args.ultimate_allowables is None else str(args.ultimate_allowables.resolve()),
                "ultimate_margin_metrics": str(ultimate_margins_path),
                "source_coverage_case_summary": str(source_coverage_summary_path),
                "critical_case_summary": str(critical_summary_path),
                "critical_case_rainflow_ranges": str(critical_rainflow_path),
                "comparison": str(comparison_path),
                "comparison_baseline": comparison_baseline,
                "comparison_coupled": comparison_coupled,
                "comparison_diagnostics": str(diagnostics_comparison_path),
                "baseline_identity_comparison": str(identity_comparison_path),
                "baseline_identity_comparison_diagnostics": str(identity_diagnostics_path),
                "signed_comparison": str(signed_comparison_path),
                "signed_comparison_diagnostics": str(signed_diagnostics_path),
                "signed_baseline_identity_comparison": str(signed_identity_comparison_path),
                "signed_baseline_identity_comparison_diagnostics": str(signed_identity_diagnostics_path),
                "signed_latex_table": str(signed_latex_table_path),
                "latex_table": str(latex_table_path),
                "source_coverage": str(source_coverage_path),
                "postprocess_manifest": str(postprocess_manifest_path),
                "report": str(report_path),
                "plots": [str(path) for path in [*history_plots, del_plot, completion_heatmap, signed_heatmap, source_sanity_plot, critical_plot]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
