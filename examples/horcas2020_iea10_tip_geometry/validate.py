#!/usr/bin/env python3
"""Validate the frozen Horcas 2020 multi-geometry response package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SCHEMA = "horcas2020-iea10-tip-geometry-reference-v1"
PROTOCOL_SCHEMA = "horcas2020-iea10-tip-geometry-validation-protocol-v1"
TARGET_SHA = "ffa8b9f134744ed966df576dccca9ae771808412ef0f752129edf0df1d1e1967"
PROVENANCE_SHA = "cafd90dee9107209a1f0c12cd8ee08dbce7d592d8d373d2f61d0f70ec64b8450"
SOURCE_SHA = "cbd11afc906ed268a8cbb7589fe370df2ea2fa218833c8e823ccfbbd4942a89d"
DOI = "10.1063/5.0004005"
GEOMETRIES = {
    "BASE": (0, 0, 0, 0.67, 2.25, "training"),
    "EXT": (5, 0, 0, 0.64, 2.15, "training"),
    "SWEEP": (5, 20, 0, 0.64, 2.13, "training"),
    "DIHE": (5, 0, 20, 0.62, 2.08, "training"),
    "COMB": (5, 20, 20, 0.62, 2.06, "locked_holdout"),
}
INCLINATIONS = (40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 65.0, 70.0)
COLUMNS = (
    "case_id",
    "geometry",
    "inclination_deg",
    "extension_percent",
    "sweep_deg",
    "dihedral_deg",
    "first_edgewise_frequency_hz",
    "first_edgewise_log_decrement_percent",
    "peak_to_peak_edge_amplitude_m",
    "peak_edge_amplitude_m",
    "response_frequency_over_fn",
    "role",
    "frequency_scoreable",
)


class ValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValidationError(f"Non-finite JSON constant: {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Could not load {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{path.name} must contain an object.")
    return payload


def _number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as exc:
        raise ValidationError(f"Invalid {key} in {row.get('case_id')}") from exc
    _require(math.isfinite(value), f"Non-finite {key} in {row.get('case_id')}")
    return value


def load_reference(root: Path = HERE, source_pdf: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    targets = root / "targets.csv"
    provenance_path = root / "provenance.json"
    protocol_path = root / "validation_protocol.json"
    _require(_sha(targets) == TARGET_SHA, "Target CSV SHA-256 mismatch.")
    _require(_sha(provenance_path) == PROVENANCE_SHA, "Provenance SHA-256 mismatch.")
    provenance = _strict_json(provenance_path)
    protocol = _strict_json(protocol_path)
    _require(provenance.get("schema") == SCHEMA, "Unexpected provenance schema.")
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "Unexpected protocol schema.")
    publication = provenance.get("publication", {})
    _require(publication.get("doi") == DOI, "Publication DOI changed.")
    source = publication.get("source", {})
    _require(source.get("sha256") == SOURCE_SHA, "Source PDF identity changed.")
    _require(source.get("figure") == 19, "Source figure changed.")
    if source_pdf is not None:
        _require(_sha(Path(source_pdf)) == SOURCE_SHA, "Provided source PDF SHA-256 mismatch.")
    _require(protocol["reference"]["targets_sha256"] == TARGET_SHA, "Protocol target identity changed.")
    _require(protocol["reference"]["provenance_sha256"] == PROVENANCE_SHA, "Protocol provenance identity changed.")
    _require(protocol["primary_holdout_geometry"] == "COMB", "Holdout geometry changed.")

    with targets.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == COLUMNS, "Target columns changed.")
        rows = list(reader)
    _require(len(rows) == 55, "Expected exactly 55 geometry-condition rows.")
    seen: set[str] = set()
    by_geometry: dict[str, list[dict[str, str]]] = {key: [] for key in GEOMETRIES}
    for row in rows:
        case_id = row["case_id"]
        _require(case_id not in seen, f"Duplicate case ID: {case_id}")
        seen.add(case_id)
        geometry = row["geometry"]
        _require(geometry in GEOMETRIES, f"Unexpected geometry: {geometry}")
        extension, sweep, dihedral, frequency, decrement, role = GEOMETRIES[geometry]
        inclination = _number(row, "inclination_deg")
        expected_id = f"{geometry}_I{inclination:05.1f}".replace(".", "p")
        _require(case_id == expected_id, f"Noncanonical case ID: {case_id}")
        _require(_number(row, "extension_percent") == extension, f"Extension changed in {case_id}")
        _require(_number(row, "sweep_deg") == sweep, f"Sweep changed in {case_id}")
        _require(_number(row, "dihedral_deg") == dihedral, f"Dihedral changed in {case_id}")
        _require(_number(row, "first_edgewise_frequency_hz") == frequency, f"Frequency changed in {case_id}")
        _require(_number(row, "first_edgewise_log_decrement_percent") == decrement, f"Damping changed in {case_id}")
        peak_to_peak = _number(row, "peak_to_peak_edge_amplitude_m")
        peak = _number(row, "peak_edge_amplitude_m")
        _require(peak_to_peak >= 0.0 and peak >= 0.0, f"Negative amplitude in {case_id}")
        _require(abs(peak - 0.5 * peak_to_peak) <= 5.1e-10, f"Peak conversion changed in {case_id}")
        _require(row["role"] == role, f"Evidence role changed in {case_id}")
        scoreable = row["frequency_scoreable"]
        _require(scoreable in {"true", "false"}, f"Invalid scoreable flag in {case_id}")
        if scoreable == "true":
            ratio = _number(row, "response_frequency_over_fn")
            _require(0.8 <= ratio <= 1.2, f"Frequency ratio out of range in {case_id}")
        else:
            _require(row["response_frequency_over_fn"] == "NA", f"Missing frequency must be NA in {case_id}")
        by_geometry[geometry].append(row)

    for geometry, group in by_geometry.items():
        _require(len(group) == 11, f"Expected 11 rows for {geometry}.")
        _require(tuple(_number(row, "inclination_deg") for row in group) == INCLINATIONS, f"Inclination grid changed for {geometry}.")
    _require(sum(row["role"] == "training" for row in rows) == 44, "Training count changed.")
    _require(sum(row["role"] == "locked_holdout" for row in rows) == 11, "Holdout count changed.")
    return {
        "rows": rows,
        "by_geometry": by_geometry,
        "provenance": provenance,
        "protocol": protocol,
        "targets_sha256": TARGET_SHA,
        "provenance_sha256": PROVENANCE_SHA,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--source-pdf", type=Path)
    args = parser.parse_args()
    result = load_reference(args.root, args.source_pdf)
    print(json.dumps({"valid": True, "rows": len(result["rows"]), "training": 44, "holdout": 11}, indent=2))


if __name__ == "__main__":
    main()
