#!/usr/bin/env python3
"""Build final spectral PSD comparison rows from explicit time-history artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

VORLAP_ROOT = Path(__file__).resolve().parents[1]
if str(VORLAP_ROOT) not in sys.path:
    sys.path.insert(0, str(VORLAP_ROOT))

from vorlap.spectral_reconstruction import one_sided_power_spectrum


SPECTRAL_PSD_COMPARISON_DEFAULT = "postprocess/spectral_psd_comparison.csv"
SPECTRAL_PSD_SOURCE_SPEC_DEFAULT = "postprocess/spectral_psd_sources.csv"
SPECTRAL_PSD_COMPARISON_REQUIRED_COLUMNS = (
    "scenario",
    "case",
    "component",
    "channel",
    "source",
    "expected_frequency_hz",
    "peak_frequency_hz",
    "frequency_error_hz",
    "band_power_fraction",
    "peak_power",
    "passed",
    "artifact",
)
SPECTRAL_PSD_SOURCE_SPEC_REQUIRED_COLUMNS = (
    "scenario",
    "case",
    "component",
    "channel",
    "source",
    "expected_frequency_hz",
    "artifact",
    "time_column",
    "value_column",
)
SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES = ("analytic", "offline", "sil", "qblade")


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _row_float(row: dict[str, Any], column: str) -> float:
    try:
        value = float(row.get(column, ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Source spec row has nonnumeric {column!r}: {row.get(column)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Source spec row has nonfinite {column!r}: {row.get(column)!r}")
    return value


def _resolve_path(root: Path, raw_path: str) -> Path:
    if not str(raw_path).strip():
        raise ValueError("Source spec row has an empty artifact path.")
    path = Path(str(raw_path).strip())
    return path if path.is_absolute() else root / path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def read_source_spec(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing spectral PSD source spec: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON source spec must be a list of rows or an object with a 'rows' list.")
        return [dict(row) for row in rows]
    return [dict(row) for row in read_csv_rows(path)]


def validate_source_spec_row(row: dict[str, Any], index: int) -> None:
    missing = [column for column in SPECTRAL_PSD_SOURCE_SPEC_REQUIRED_COLUMNS if not str(row.get(column, "")).strip()]
    if missing:
        raise ValueError(f"Source spec row {index} is missing required column(s): {', '.join(missing)}")
    source = str(row.get("source", "")).strip().lower()
    if source not in SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES:
        raise ValueError(f"Source spec row {index} has unsupported source {source!r}.")


def load_time_history(path: Path, *, time_column: str, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing PSD source artifact: {path}")
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"PSD source artifact has no rows: {path}")
    fieldnames = set(rows[0])
    missing = [column for column in (time_column, value_column) if column not in fieldnames]
    if missing:
        raise ValueError(f"PSD source artifact {path} is missing required column(s): {', '.join(missing)}")

    time: list[float] = []
    values: list[float] = []
    for row_index, row in enumerate(rows, start=2):
        try:
            t = float(row.get(time_column, ""))
            value = float(row.get(value_column, ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PSD source artifact {path} has nonnumeric data at row {row_index}.") from exc
        if not math.isfinite(t) or not math.isfinite(value):
            raise ValueError(f"PSD source artifact {path} has nonfinite data at row {row_index}.")
        time.append(t)
        values.append(value)
    return np.asarray(time, dtype=float), np.asarray(values, dtype=float)


def compute_psd_metrics(
    time: np.ndarray,
    values: np.ndarray,
    *,
    expected_frequency_hz: float,
    band_tolerance_hz: float,
    min_total_power: float,
) -> dict[str, float]:
    if expected_frequency_hz <= 0.0:
        raise ValueError("expected_frequency_hz must be positive.")
    if band_tolerance_hz < 0.0:
        raise ValueError("band_tolerance_hz must be nonnegative.")
    frequency_hz, power = one_sided_power_spectrum(time, values)
    positive = frequency_hz > 0.0
    total_power = float(np.sum(power[positive]))
    if total_power <= float(min_total_power):
        raise ValueError("PSD source artifact has no measurable positive-frequency dynamic power.")

    positive_indices = np.flatnonzero(positive)
    peak_index = int(positive_indices[int(np.argmax(power[positive_indices]))])
    band = positive & (np.abs(frequency_hz - expected_frequency_hz) <= band_tolerance_hz + np.finfo(float).eps)
    band_power = float(np.sum(power[band]))
    peak_frequency = float(frequency_hz[peak_index])
    return {
        "expected_frequency_hz": float(expected_frequency_hz),
        "peak_frequency_hz": peak_frequency,
        "frequency_error_hz": abs(peak_frequency - float(expected_frequency_hz)),
        "band_power_fraction": band_power / total_power,
        "peak_power": float(power[peak_index]),
        "total_dynamic_power": total_power,
    }


def comparison_row_from_source(
    run_root: Path,
    row: dict[str, Any],
    *,
    index: int,
    max_frequency_error_hz: float,
    min_band_power_fraction: float,
    band_tolerance_hz: float,
    min_total_power: float,
) -> dict[str, str]:
    validate_source_spec_row(row, index)
    artifact = _resolve_path(run_root, str(row["artifact"]))
    time, values = load_time_history(
        artifact,
        time_column=str(row["time_column"]).strip(),
        value_column=str(row["value_column"]).strip(),
    )
    expected_frequency = _row_float(row, "expected_frequency_hz")
    metrics = compute_psd_metrics(
        time,
        values,
        expected_frequency_hz=expected_frequency,
        band_tolerance_hz=band_tolerance_hz,
        min_total_power=min_total_power,
    )
    passed = (
        metrics["frequency_error_hz"] <= max_frequency_error_hz
        and metrics["band_power_fraction"] >= min_band_power_fraction
    )
    return {
        "scenario": str(row["scenario"]).strip(),
        "case": str(row["case"]).strip(),
        "component": str(row["component"]).strip(),
        "channel": str(row["channel"]).strip(),
        "source": str(row["source"]).strip().lower(),
        "expected_frequency_hz": _format_float(metrics["expected_frequency_hz"]),
        "peak_frequency_hz": _format_float(metrics["peak_frequency_hz"]),
        "frequency_error_hz": _format_float(metrics["frequency_error_hz"]),
        "band_power_fraction": _format_float(metrics["band_power_fraction"]),
        "peak_power": _format_float(metrics["peak_power"]),
        "passed": "true" if passed else "false",
        "artifact": _relative_or_absolute(run_root, artifact),
    }


def build_comparison_rows(
    run_root: Path,
    source_spec: Path,
    *,
    max_frequency_error_hz: float = 0.05,
    min_band_power_fraction: float = 0.80,
    band_tolerance_hz: float | None = None,
    min_total_power: float = 1.0e-20,
) -> list[dict[str, str]]:
    root = run_root.resolve()
    spec_path = source_spec.resolve()
    tolerance = max_frequency_error_hz if band_tolerance_hz is None else band_tolerance_hz
    if max_frequency_error_hz < 0.0:
        raise ValueError("max_frequency_error_hz must be nonnegative.")
    if min_band_power_fraction < 0.0 or min_band_power_fraction > 1.0:
        raise ValueError("min_band_power_fraction must be between 0 and 1.")

    rows = read_source_spec(spec_path)
    if not rows:
        raise ValueError(f"Spectral PSD source spec has no rows: {spec_path}")
    return [
        comparison_row_from_source(
            root,
            row,
            index=index,
            max_frequency_error_hz=max_frequency_error_hz,
            min_band_power_fraction=min_band_power_fraction,
            band_tolerance_hz=tolerance,
            min_total_power=min_total_power,
        )
        for index, row in enumerate(rows, start=2)
    ]


def write_comparison_csv(path: Path, rows: list[dict[str, str]], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing spectral PSD comparison without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPECTRAL_PSD_COMPARISON_REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SPECTRAL_PSD_COMPARISON_REQUIRED_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Campaign run root.")
    parser.add_argument(
        "--source-spec",
        "--spec",
        type=Path,
        default=None,
        help=(
            "CSV or JSON rows describing source histories. Defaults to "
            f"RUN_ROOT/{SPECTRAL_PSD_SOURCE_SPEC_DEFAULT}."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output CSV path. Defaults to RUN_ROOT/{SPECTRAL_PSD_COMPARISON_DEFAULT}.",
    )
    parser.add_argument("--max-frequency-error-hz", type=float, default=0.05)
    parser.add_argument("--min-band-power-fraction", type=float, default=0.80)
    parser.add_argument(
        "--band-tolerance-hz",
        type=float,
        default=None,
        help="Frequency half-width used for band-power fraction. Defaults to --max-frequency-error-hz.",
    )
    parser.add_argument("--min-total-power", type=float, default=1.0e-20)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing comparison CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    source_spec = (args.source_spec or (run_root / SPECTRAL_PSD_SOURCE_SPEC_DEFAULT)).resolve()
    out_path = (args.out or (run_root / SPECTRAL_PSD_COMPARISON_DEFAULT)).resolve()
    rows = build_comparison_rows(
        run_root,
        source_spec,
        max_frequency_error_hz=args.max_frequency_error_hz,
        min_band_power_fraction=args.min_band_power_fraction,
        band_tolerance_hz=args.band_tolerance_hz,
        min_total_power=args.min_total_power,
    )
    write_comparison_csv(out_path, rows, force=args.force)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
