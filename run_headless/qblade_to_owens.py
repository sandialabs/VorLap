#!/usr/bin/env python3
"""Convert a QBlade structural model and optionally run an OWENS/GXBeam Campbell sweep.

The converter writes a solver-neutral, numeric CSV bundle.  The companion Julia
driver consumes that bundle with the OWENS project environment, runs GXBeam's
rotating eigenanalysis, and writes ParaView mode-shape collections.

This is intentionally a structural converter.  Aerodynamic, hydrodynamic, and
controller loads are not included in the eigenanalysis.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Iterable, Sequence


SECTION_COLUMNS = (
    "fraction",
    "mass_per_length",
    "eix",
    "eiy",
    "ea",
    "gj",
    "ga",
    "structural_pitch_deg",
    "ksx",
    "ksy",
    "rgx",
    "rgy",
    "xcm",
    "ycm",
    "xce",
    "yce",
    "xcs",
    "ycs",
)

COMPONENT_IDS = {
    "substructure": 1,
    "tower": 2,
    "torque_tube": 3,
    "blade": 4,
    "lower_strut": 5,
    "upper_strut": 6,
    "rigid_connector": 90,
    "released_connector": 91,
    "bearing": 92,
    "drivetrain": 93,
}

KIND_IDS = {"physical": 1, "rigid": 2, "released": 3, "bearing": 4, "drivetrain": 5}

DAMPING_CANDIDATES = {
    "qblade_2_0_9_7_documented": {
        "substructure": "SUBELEMENTS zero-based row[19] DAMP; optional row[20] aerodynamic drag",
        "blade_strut": "optional station column 19 distributed beta; scalar RAYLEIGHDMP fallback",
        "evidence": "QBlade 2.0.9.7 documentation",
        "sources": [
            "https://docs.qblade.org/src/user/turbine/substructure.html#substructure-definition",
            "https://docs.qblade.org/src/user/turbine/structure.html#structural-blade-and-strut-definition",
        ],
    },
    "reference_header_intended": {
        "substructure": "conflicting local header interpretation: row[19] Drag and row[20] DAMP",
        "blade_strut": "optional station column 19 distributed beta; scalar RAYLEIGHDMP fallback",
        "evidence": "reference-file header only; active-binary behavior unresolved",
        "sources": ["tracked substructure source row and header serialized in conversion_manifest.json"],
    },
    "qblade_2_0_4_archived_scalar": {
        "substructure": "SUBELEMENTS zero-based row[19] DAMP; optional row[20] aerodynamic drag",
        "blade_strut": "file-level scalar RAYLEIGHDMP; optional station column ignored",
        "evidence": "archived QBlade 2.0.4 StrModel.cpp",
        "sources": ["StrModel.cpp: row[19] DAMP, optional row[20] DRAG; scalar section RAYLEIGHDMP"],
    },
    "legacy_converter_header_scalar": {
        "substructure": "row[20] interpreted as DAMP from the conflicting local header",
        "blade_strut": "file-level scalar RAYLEIGHDMP",
        "evidence": "legacy VorLap converter behavior retained only as an audit candidate",
        "sources": ["pre-correction VorLap converter projection"],
    },
}


def _clean(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig", errors="strict").replace("\r", "").splitlines()


def _find_value(lines: Sequence[str], key: str, *, required: bool = True, default: str = "") -> str:
    for raw in lines:
        tokens = _clean(raw).split()
        for index, token in enumerate(tokens):
            if token == key:
                if index > 0:
                    return tokens[index - 1]
                if index + 1 < len(tokens):
                    return tokens[index + 1]
    if required:
        raise ValueError(f"Required QBlade key {key!r} was not found")
    return default


def _float_value(lines: Sequence[str], key: str, *, required: bool = True, default: float = 0.0) -> float:
    value = _find_value(lines, key, required=required, default=str(default))
    return float(value)


def _int_value(lines: Sequence[str], key: str, *, required: bool = True, default: int = 0) -> int:
    return int(float(_find_value(lines, key, required=required, default=str(default))))


def _bool_value(lines: Sequence[str], key: str, *, required: bool = True, default: bool = False) -> bool:
    value = _find_value(lines, key, required=required, default=str(default)).lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"QBlade key {key!r} has non-Boolean value {value!r}")


def _numeric_rows_after_header(lines: Sequence[str], header_token: str, minimum_columns: int) -> list[list[float]]:
    header_index = next(
        (index for index, line in enumerate(lines) if header_token in _clean(line).split()),
        None,
    )
    if header_index is None:
        raise ValueError(f"Numeric table header {header_token!r} was not found")
    rows: list[list[float]] = []
    for raw in lines[header_index + 1 :]:
        body = _clean(raw)
        if not body:
            if rows:
                break
            continue
        values: list[float] = []
        for token in body.split():
            try:
                values.append(float(token))
            except ValueError:
                break
        if len(values) < minimum_columns:
            if rows:
                break
            continue
        rows.append(values)
    if not rows:
        raise ValueError(f"Numeric table {header_token!r} contained no rows")
    return rows


def _numeric_rows_after_header_with_source(
    lines: Sequence[str], header_token: str, minimum_columns: int
) -> tuple[list[str], list[list[float]], list[int]]:
    """Parse a numeric section table without discarding its source identity.

    Line numbers are one-based so an artifact can be checked directly against
    the archived QBlade input.  Keeping the complete numeric rows is important:
    column 19 has different documented meanings for different structure-table
    families and must not be silently truncated by the mechanical-property
    parser.
    """

    header_index = next(
        (index for index, line in enumerate(lines) if header_token in _clean(line).split()),
        None,
    )
    if header_index is None:
        raise ValueError(f"Numeric table header {header_token!r} was not found")
    header = _clean(lines[header_index]).split()
    rows: list[list[float]] = []
    line_numbers: list[int] = []
    for index, raw in enumerate(lines[header_index + 1 :], start=header_index + 1):
        body = _clean(raw)
        if not body:
            if rows:
                break
            continue
        values: list[float] = []
        for token in body.split():
            try:
                values.append(float(token))
            except ValueError:
                break
        if len(values) < minimum_columns:
            if rows:
                break
            continue
        rows.append(values)
        line_numbers.append(index + 1)
    if not rows:
        raise ValueError(f"Numeric table {header_token!r} contained no rows")
    return header, rows, line_numbers


def _numeric_table(lines: Sequence[str], marker: str, minimum_columns: int) -> list[list[float]]:
    marker_index = next(
        (index for index, line in enumerate(lines) if _clean(line).split()[:1] == [marker]),
        None,
    )
    if marker_index is None:
        raise ValueError(f"Numeric table marker {marker!r} was not found")
    rows: list[list[float]] = []
    for raw in lines[marker_index + 1 :]:
        body = _clean(raw)
        if not body:
            if rows:
                break
            continue
        values: list[float] = []
        for token in body.split():
            try:
                values.append(float(token))
            except ValueError:
                break
        if len(values) >= minimum_columns:
            rows.append(values)
        elif rows:
            break
    return rows


def _numeric_table_with_source(
    lines: Sequence[str], marker: str, minimum_columns: int
) -> tuple[list[list[float]], int, list[str], int, list[int], list[str]]:
    """Parse a marker table and retain marker/header/row source references."""

    marker_index = next(
        (index for index, line in enumerate(lines) if _clean(line).split()[:1] == [marker]),
        None,
    )
    if marker_index is None:
        raise ValueError(f"Numeric table marker {marker!r} was not found")
    header_index = next(
        (
            index
            for index in range(marker_index + 1, len(lines))
            if _clean(lines[index])
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"Numeric table marker {marker!r} has no header")
    header = _clean(lines[header_index]).split()
    rows: list[list[float]] = []
    line_numbers: list[int] = []
    raw_rows: list[str] = []
    for index in range(header_index + 1, len(lines)):
        body = _clean(lines[index])
        if not body:
            if rows:
                break
            continue
        values: list[float] = []
        for token in body.split():
            try:
                values.append(float(token))
            except ValueError:
                break
        if len(values) >= minimum_columns:
            rows.append(values)
            line_numbers.append(index + 1)
            raw_rows.append(lines[index])
        elif rows:
            break
    if not rows:
        raise ValueError(f"Numeric table {marker!r} contained no rows")
    return rows, marker_index + 1, header, header_index + 1, line_numbers, raw_rows


def _added_masses(lines: Sequence[str]) -> list[tuple[float, float]]:
    masses: list[tuple[float, float]] = []
    pattern = re.compile(r"^ADDMASS_([-+0-9.eE]+)(?:_([-+0-9.eE]+))?$")
    for raw in lines:
        tokens = _clean(raw).split()
        for index, token in enumerate(tokens):
            match = pattern.match(token)
            if not match:
                continue
            fraction = float(match.group(1))
            if match.group(2) is not None:
                mass = float(match.group(2))
            elif index + 1 < len(tokens):
                mass = float(tokens[index + 1])
            else:
                raise ValueError(f"Added mass at fraction {fraction} has no mass value")
            if not 0.0 <= fraction <= 1.0 or mass <= 0.0:
                raise ValueError(f"Invalid added mass definition: fraction={fraction}, mass={mass}")
            masses.append((fraction, mass))
    return masses


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path_label(path: Path) -> str:
    """Return a checkout-independent label for a source artifact."""

    resolved = path.resolve()
    parts = resolved.parts
    for marker in ("run_headless", "examples", "tests"):
        if marker in parts:
            return Path(*parts[parts.index(marker) :]).as_posix()
    return resolved.name


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _interp(rows: Sequence[Sequence[float]], fraction: float, column: int) -> float:
    fractions = [row[0] for row in rows]
    if fraction <= fractions[0]:
        return float(rows[0][column])
    if fraction >= fractions[-1]:
        return float(rows[-1][column])
    upper = bisect.bisect_right(fractions, fraction)
    lower = upper - 1
    span = fractions[upper] - fractions[lower]
    weight = (fraction - fractions[lower]) / span
    return float(rows[lower][column] + weight * (rows[upper][column] - rows[lower][column]))


def _interp_geometry(rows: Sequence[dict[str, float]], height: float, key: str) -> float:
    heights = [row["height"] for row in rows]
    if height <= heights[0]:
        return rows[0][key]
    if height >= heights[-1]:
        return rows[-1][key]
    upper = bisect.bisect_right(heights, height)
    lower = upper - 1
    weight = (height - heights[lower]) / (heights[upper] - heights[lower])
    return rows[lower][key] + weight * (rows[upper][key] - rows[lower][key])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Sequence[float]) -> list[float]:
    magnitude = _norm(a)
    if magnitude <= 1.0e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return [value / magnitude for value in a]


def _frame(start: Sequence[float], stop: Sequence[float], preferred_y: Sequence[float]) -> list[list[float]]:
    axis_x = _unit([stop[i] - start[i] for i in range(3)])
    projected_y = [preferred_y[i] - _dot(preferred_y, axis_x) * axis_x[i] for i in range(3)]
    if _norm(projected_y) < 1.0e-8:
        fallback = [1.0, 0.0, 0.0] if abs(axis_x[0]) < 0.9 else [0.0, 1.0, 0.0]
        projected_y = [fallback[i] - _dot(fallback, axis_x) * axis_x[i] for i in range(3)]
    axis_y = _unit(projected_y)
    axis_z = _unit(_cross(axis_x, axis_y))
    return [[axis_x[row], axis_y[row], axis_z[row]] for row in range(3)]


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def _transpose(a: Sequence[Sequence[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*a)]


def _add_matrices(*matrices: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[sum(matrix[i][j] for matrix in matrices) for j in range(6)] for i in range(6)]


def _diagonal(values: Sequence[float]) -> list[list[float]]:
    return [[float(values[i]) if i == j else 0.0 for j in range(len(values))] for i in range(len(values))]


def _skew(vector: Sequence[float]) -> list[list[float]]:
    x, y, z = vector
    return [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]


def _offset_transform(y: float, z: float) -> list[list[float]]:
    transform = _diagonal([1.0] * 6)
    skew = _skew([0.0, y, z])
    for i in range(3):
        for j in range(3):
            transform[i][j + 3] = -skew[i][j]
    return transform


def _rotation6(angle_deg: float) -> list[list[float]]:
    angle = math.radians(angle_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    rotation = [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    result = [[0.0] * 6 for _ in range(6)]
    for block in (0, 3):
        for i in range(3):
            for j in range(3):
                result[block + i][block + j] = rotation[i][j]
    return result


def _rotate_matrix(matrix: Sequence[Sequence[float]], angle_deg: float) -> list[list[float]]:
    rotation = _rotation6(angle_deg)
    return _matmul(_matmul(rotation, matrix), _transpose(rotation))


def _translate_matrix(matrix: Sequence[Sequence[float]], y: float, z: float) -> list[list[float]]:
    transform = _offset_transform(y, z)
    return _matmul(_matmul(_transpose(transform), matrix), transform)


def _physical_section_mass_matrix(
    rho_a: float,
    rg_x_normalized: float,
    rg_y_normalized: float,
    chord_or_diameter: float,
    section_rotation_deg: float,
    mass_y: float,
    mass_z: float,
) -> list[list[float]]:
    """Build a positive-definite section mass matrix from QBlade properties.

    The QBlade files label RGX/RGY as orthogonal radii of gyration and XCM/YCM
    as the center-of-mass offset.  Interpret the radii as principal inertias at
    that center, rotate the inertia tensor, and transport it to the beam
    reference line.  This is the physically consistent interpretation needed
    by GXBeam's generalized eigenproblem.  QBlade's current Chrono call instead
    rotates the radii as coordinates and omits this parallel-axis transport;
    that literal matrix is indefinite for sections in this model.
    """
    if rg_x_normalized == 0.0:
        rg_x_normalized = math.sqrt(0.2)
    if rg_y_normalized == 0.0:
        rg_y_normalized = math.sqrt(0.2)
    rg_x = rg_x_normalized * chord_or_diameter
    rg_y = rg_y_normalized * chord_or_diameter

    inertia_at_center = _diagonal(
        [
            rho_a,
            rho_a,
            rho_a,
            rho_a * (rg_x**2 + rg_y**2),
            rho_a * rg_x**2,
            rho_a * rg_y**2,
        ]
    )
    rotated = _rotate_matrix(inertia_at_center, section_rotation_deg)
    return _translate_matrix(rotated, mass_y, mass_z)


def _flatten(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def _section_matrices(
    row: Sequence[float],
    *,
    chord_or_diameter: float,
    pitch_axis: float,
    geometry_twist_deg: float,
    stiffness_tuner: float,
    mass_tuner: float,
) -> tuple[list[list[float]], list[list[float]]]:
    rho_a = row[1] * mass_tuner
    eix = row[2] * stiffness_tuner
    eiy = row[3] * stiffness_tuner
    ea = row[4] * stiffness_tuner
    gj = row[5] * stiffness_tuner
    ga = row[6] * stiffness_tuner
    section_rotation = row[7] - geometry_twist_deg
    ga_y = ga * row[8]
    ga_z = ga * row[9]

    def offset(x_value: float, y_value: float) -> tuple[float, float]:
        x = (x_value + pitch_axis - 0.5) * chord_or_diameter
        y = y_value * chord_or_diameter
        angle = math.radians(-geometry_twist_deg)
        return x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)

    mass_y, mass_z = offset(row[12], row[13])
    elastic_y, elastic_z = offset(row[14], row[15])
    shear_y, shear_z = offset(row[16], row[17])

    elastic = _rotate_matrix(_diagonal([ea, 0.0, 0.0, 0.0, eix, eiy]), section_rotation)
    shear = _rotate_matrix(_diagonal([0.0, ga_y, ga_z, gj, 0.0, 0.0]), section_rotation)
    stiffness = _add_matrices(
        _translate_matrix(elastic, elastic_y, elastic_z),
        _translate_matrix(shear, shear_y, shear_z),
    )

    mass = _physical_section_mass_matrix(
        rho_a,
        row[10],
        row[11],
        chord_or_diameter,
        section_rotation,
        mass_y,
        mass_z,
    )
    return stiffness, mass


def _parse_section_file(path: Path) -> dict[str, object]:
    lines = _lines(path)
    header, rows, row_line_numbers = _numeric_rows_after_header_with_source(
        lines, "MASSD_[kg/m]", 18
    )
    if len(rows) < 2:
        raise ValueError(f"Structural property file must contain at least two stations: {path}")
    if any(rows[index + 1][0] <= rows[index][0] for index in range(len(rows) - 1)):
        raise ValueError(f"Structural fractions must be strictly increasing: {path}")
    if rows[0][0] > 1.0e-9 or rows[-1][0] < 1.0 - 1.0e-9:
        raise ValueError(f"Structural property table must cover fractions 0 through 1: {path}")
    station_beta_columns = [
        index
        for index, token in enumerate(header)
        if token.lower().startswith("rayleighdmp_vec")
    ]
    if len(station_beta_columns) > 1:
        raise ValueError(f"Multiple RAYLEIGHDMP_Vec columns were found: {path}")
    station_beta_column = station_beta_columns[0] if station_beta_columns else None
    if station_beta_column is not None and station_beta_column != 18:
        raise ValueError(
            f"RAYLEIGHDMP_Vec must be optional zero-based column 18, found "
            f"column {station_beta_column}: {path}"
        )
    if station_beta_column is not None and any(
        len(row) <= station_beta_column for row in rows
    ):
        raise ValueError(f"RAYLEIGHDMP_Vec header is present but a station value is missing: {path}")
    scalar_beta = _float_value(lines, "RAYLEIGHDMP", required=False, default=0.0)
    return {
        "path": path,
        "rows": rows,
        "header": header,
        "header_line_number": next(
            index + 1
            for index, line in enumerate(lines)
            if "MASSD_[kg/m]" in _clean(line).split()
        ),
        "row_line_numbers": row_line_numbers,
        "station_beta_column": station_beta_column,
        "stiffness_tuner": _float_value(lines, "STIFFTUNER", required=False, default=1.0),
        "mass_tuner": _float_value(lines, "MASSTUNER", required=False, default=1.0),
        "rayleigh": scalar_beta,
        "disc": _int_value(lines, "DISC", required=False, default=0),
        "disctype": _int_value(lines, "DISCTYPE", required=False, default=-1),
        "added_masses": _added_masses(lines),
    }


def _section_damping_candidates(
    data: dict[str, object], fraction: float
) -> tuple[dict[str, float], dict[str, object]]:
    """Return explicit station-distributed and legacy scalar beta candidates."""

    rows = data["rows"]
    station_column = data["station_beta_column"]
    scalar_beta = float(data["rayleigh"])
    if station_column is None:
        distributed_beta = scalar_beta
        policy = "scalar_RAYLEIGHDMP_fallback_no_station_column"
    else:
        distributed_beta = _interp(rows, fraction, int(station_column))
        policy = "linear_station_RAYLEIGHDMP_Vec"

    fractions = [float(row[0]) for row in rows]
    upper = bisect.bisect_right(fractions, fraction)
    if upper == 0:
        lower = upper = 0
    elif upper >= len(fractions):
        lower = upper = len(fractions) - 1
    elif abs(fractions[upper - 1] - fraction) <= 1.0e-12:
        lower = upper = upper - 1
    else:
        lower = upper - 1
    line_numbers = data["row_line_numbers"]
    provenance = {
        "source_label": _portable_path_label(Path(data["path"])),
        "source_table": "section_properties",
        "source_header_line": int(data["header_line_number"]),
        "source_row_line_lower": int(line_numbers[lower]),
        "source_row_line_upper": int(line_numbers[upper]),
        "source_fraction": float(fraction),
        "distributed_policy": policy,
        "scalar_beta_s": scalar_beta,
        "station_beta_column_zero_based": station_column,
    }
    return (
        {
            "qblade_2_0_9_7_documented": distributed_beta,
            "reference_header_intended": distributed_beta,
            "qblade_2_0_4_archived_scalar": scalar_beta,
            "legacy_converter_header_scalar": scalar_beta,
        },
        provenance,
    )


def _subelement_damping_candidates(row: Sequence[float]) -> dict[str, float | None]:
    if len(row) < 20:
        raise ValueError(
            "QBlade SUBELEMENTS requires 20 base numeric entries through documented DAMP"
        )
    documented_beta = float(row[19])
    header_intended_beta = float(row[20]) if len(row) > 20 else None
    return {
        "qblade_2_0_9_7_documented": documented_beta,
        "reference_header_intended": header_intended_beta,
        "qblade_2_0_4_archived_scalar": documented_beta,
        "legacy_converter_header_scalar": header_intended_beta,
    }


def _parse_blade(path: Path) -> tuple[list[dict[str, float]], list[dict[str, float | str]]]:
    lines = _lines(path)
    header_index = next(index for index, line in enumerate(lines) if "HEIGHT_[m]" in line)
    header = _clean(lines[header_index]).split()
    rows: list[dict[str, float]] = []
    for raw in lines[header_index + 1 :]:
        body = _clean(raw)
        if not body:
            if rows:
                break
            continue
        tokens = body.split()
        try:
            numeric = [float(value) for value in tokens[: len(header) - 1]]
        except ValueError:
            break
        if len(numeric) < 7:
            break
        # QBlade 2.0.9 adds a tangential-offset column between radius and twist.
        if len(numeric) >= 7:
            height, chord, radius, tangential_offset, twist, circular_angle, pitch_axis = numeric[:7]
        else:  # pragma: no cover - guarded above, retained for format documentation
            height, chord, radius, twist, circular_angle, pitch_axis = numeric[:6]
            tangential_offset = 0.0
        rows.append(
            {
                "height": height,
                "chord": chord,
                "radius": radius,
                "tangential_offset": tangential_offset,
                "twist": twist,
                "circular_angle": circular_angle,
                "pitch_axis": pitch_axis,
            }
        )
    if len(rows) < 2:
        raise ValueError(f"Blade geometry table is incomplete: {path}")

    struts: list[dict[str, float | str]] = []
    block_pattern = re.compile(r"^STRUT_(\d+)$")
    for index, raw in enumerate(lines):
        match = block_pattern.match(_clean(raw))
        if not match:
            continue
        end_marker = f"END_STRUT_{match.group(1)}"
        end = next(i for i in range(index + 1, len(lines)) if _clean(lines[i]) == end_marker)
        block = lines[index + 1 : end]
        struts.append(
            {
                "id": int(match.group(1)),
                "name": _find_value(block, "NAME_STR"),
                "chord_hub": _float_value(block, "CHORDHUB_STR"),
                "chord_blade": _float_value(block, "CHORDBLD_STR"),
                "angle_deg": _float_value(block, "ANGLE_STR"),
                "height_blade": _float_value(block, "HGTBLD_STR"),
                "height_hub": _float_value(block, "HGTHUB_STR"),
                "distance_hub": _float_value(block, "DSTHUB_STR"),
                "pitch_axis_hub": _float_value(block, "PAXISHUB_STR", required=False, default=0.25),
                "pitch_axis_blade": _float_value(block, "PAXISBLD_STR", required=False, default=0.25),
            }
        )
    return rows, struts


class ModelBuilder:
    def __init__(self, *, penalty_factor: float, release_ratio: float) -> None:
        self.nodes: list[dict[str, object]] = []
        self.elements: list[dict[str, object]] = []
        self.point_masses: dict[int, list[list[float]]] = {}
        self.fixed_nodes: list[tuple[int, list[int]]] = []
        self.penalty_factor = penalty_factor
        self.release_ratio = release_ratio
        self._physical_scales = [1.0] * 6

    def add_node(
        self,
        xyz: Sequence[float],
        *,
        label: str,
        component: str,
        frame: Sequence[Sequence[float]] | None = None,
        half_y: float = 0.02,
        half_z: float = 0.02,
    ) -> int:
        node_id = len(self.nodes) + 1
        if frame is None:
            frame = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        self.nodes.append(
            {
                "id": node_id,
                "xyz": [float(value) for value in xyz],
                "label": label,
                "component": component,
                "frame": [list(row) for row in frame],
                "half_y": float(max(half_y, 1.0e-3)),
                "half_z": float(max(half_z, 1.0e-3)),
            }
        )
        return node_id

    def update_visual(self, node_id: int, frame: Sequence[Sequence[float]], half_y: float, half_z: float) -> None:
        node = self.nodes[node_id - 1]
        if half_y * half_z >= float(node["half_y"]) * float(node["half_z"]):
            node["frame"] = [list(row) for row in frame]
            node["half_y"] = float(half_y)
            node["half_z"] = float(half_z)

    def add_element(
        self,
        start: int,
        stop: int,
        *,
        component: str,
        kind: str,
        frame: Sequence[Sequence[float]],
        stiffness: Sequence[Sequence[float]],
        mass: Sequence[Sequence[float]],
        damping: float,
        damping_candidates: dict[str, float | None] | None = None,
        damping_provenance: dict[str, object] | None = None,
        length: float | None = None,
    ) -> int:
        if length is None:
            p1 = self.nodes[start - 1]["xyz"]
            p2 = self.nodes[stop - 1]["xyz"]
            length = _norm([p2[i] - p1[i] for i in range(3)])
        if kind == "physical":
            for index in range(6):
                self._physical_scales[index] = max(self._physical_scales[index], abs(stiffness[index][index]))
        self.elements.append(
            {
                "id": len(self.elements) + 1,
                "start": start,
                "stop": stop,
                "length": float(length),
                "component": component,
                "kind": kind,
                "frame": [list(row) for row in frame],
                "stiffness": [list(row) for row in stiffness],
                "mass": [list(row) for row in mass],
                "damping": [float(damping)] * 6,
                "damping_candidates": {
                    candidate_id: (
                        None
                        if value is None
                        else float(value)
                    )
                    for candidate_id, value in (
                        damping_candidates
                        if damping_candidates is not None
                        else {candidate_id: damping for candidate_id in DAMPING_CANDIDATES}
                    ).items()
                },
                "damping_provenance": damping_provenance or {
                    "source_label": "generated_connector",
                    "source_table": "none",
                    "source_header_line": 0,
                    "source_row_line_lower": 0,
                    "source_row_line_upper": 0,
                    "source_fraction": 0.0,
                    "distributed_policy": "zero_generated_connector",
                    "scalar_beta_s": 0.0,
                    "station_beta_column_zero_based": None,
                },
            }
        )
        return len(self.elements)

    def add_penalty_element(
        self,
        start: int,
        stop: int,
        *,
        component: str,
        kind: str,
        frame: Sequence[Sequence[float]],
        released_dofs: Iterable[int] = (),
    ) -> None:
        values = [scale * self.penalty_factor for scale in self._physical_scales]
        for dof in released_dofs:
            # Keep the released-DOF regularization independent of the rigid-link
            # penalty.  Otherwise tightening a rigid constraint also (wrongly)
            # stiffens every QBlade hinge release and changes physical modes.
            values[dof] = self._physical_scales[dof] * self.release_ratio
        self.add_element(
            start,
            stop,
            component=component,
            kind=kind,
            frame=frame,
            stiffness=_diagonal(values),
            mass=_diagonal([1.0e-9] * 6),
            damping=0.0,
            length=1.0,
        )

    def add_point_mass(self, node_id: int, mass: float, *, inertia_global: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        matrix = self.point_masses.setdefault(node_id, [[0.0] * 6 for _ in range(6)])
        for index in range(3):
            matrix[index][index] += mass
            matrix[index + 3][index + 3] += float(inertia_global[index])


def _unique(values: Iterable[float], tolerance: float = 1.0e-10) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        value = min(1.0, max(0.0, float(value)))
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def _nearest_node(fractions: Sequence[float], nodes: Sequence[int], target: float) -> int:
    index = min(range(len(fractions)), key=lambda i: abs(fractions[i] - target))
    if abs(fractions[index] - target) > 1.0e-7:
        raise ValueError(f"Required connection fraction {target} was not inserted in the component mesh")
    return nodes[index]


def _write_bundle(output_dir: Path, builder: ModelBuilder, manifest: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "nodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node_id", "x_m", "y_m", "z_m"))
        for node in builder.nodes:
            writer.writerow((node["id"], *node["xyz"]))

    # Keep solver input numeric while retaining a deterministic bridge from
    # QBlade component coordinates to the converted GXBeam mesh.  Labels are
    # generated by the converter (not scraped from a visualization file) and
    # are therefore safe to use when interpolating mode shapes to external-
    # action locations without changing the accepted eigenproblem mesh.
    with (output_dir / "node_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node_id", "component", "label"))
        for node in builder.nodes:
            writer.writerow((node["id"], node["component"], node["label"]))

    frame_headers = [f"frame_{i}{j}" for i in range(1, 4) for j in range(1, 4)]
    matrix_headers = [f"{prefix}_{i}{j}" for prefix in ("k", "m") for i in range(1, 7) for j in range(1, 7)]
    with (output_dir / "elements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("element_id", "start_node", "stop_node", "length_m", "component_id", "kind_id", *frame_headers, *matrix_headers, *(f"damping_{i}" for i in range(1, 7)))
        )
        for element in builder.elements:
            writer.writerow(
                (
                    element["id"],
                    element["start"],
                    element["stop"],
                    element["length"],
                    COMPONENT_IDS[element["component"]],
                    KIND_IDS[element["kind"]],
                    *_flatten(element["frame"]),
                    *_flatten(element["stiffness"]),
                    *_flatten(element["mass"]),
                    *element["damping"],
                )
            )

    candidate_ids = tuple(DAMPING_CANDIDATES)
    with (output_dir / "element_damping_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "element_id",
                "component_id",
                "kind_id",
                "source_label",
                "source_table",
                "source_header_line",
                "source_row_line_lower",
                "source_row_line_upper",
                "source_fraction",
                "distributed_policy",
                "scalar_beta_s",
                "station_beta_column_zero_based",
                *(f"{candidate_id}_beta_s" for candidate_id in candidate_ids),
            )
        )
        for element in builder.elements:
            source = element["damping_provenance"]
            writer.writerow(
                (
                    element["id"],
                    COMPONENT_IDS[element["component"]],
                    KIND_IDS[element["kind"]],
                    source["source_label"],
                    source["source_table"],
                    source["source_header_line"],
                    source["source_row_line_lower"],
                    source["source_row_line_upper"],
                    source["source_fraction"],
                    source["distributed_policy"],
                    source["scalar_beta_s"],
                    ""
                    if source["station_beta_column_zero_based"] is None
                    else source["station_beta_column_zero_based"],
                    *(
                        "" if element["damping_candidates"][candidate_id] is None else element["damping_candidates"][candidate_id]
                        for candidate_id in candidate_ids
                    ),
                )
            )

    with (output_dir / "point_masses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node_id", *(f"mass_{i}{j}" for i in range(1, 7) for j in range(1, 7))))
        for node_id, matrix in sorted(builder.point_masses.items()):
            writer.writerow((node_id, *_flatten(matrix)))

    with (output_dir / "fixed_nodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node_id", "ux", "uy", "uz", "rx", "ry", "rz"))
        for node_id, dofs in builder.fixed_nodes:
            writer.writerow((node_id, *dofs))

    with (output_dir / "visual_sections.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("node_id", *frame_headers, "half_y_m", "half_z_m"))
        for node in builder.nodes:
            writer.writerow((node["id"], *_flatten(node["frame"]), node["half_y"], node["half_z"]))

    manifest["mesh"] = {
        "node_count": len(builder.nodes),
        "element_count": len(builder.elements),
        "physical_element_count": sum(element["kind"] == "physical" for element in builder.elements),
        "connector_element_count": sum(element["kind"] != "physical" for element in builder.elements),
        "point_mass_node_count": len(builder.point_masses),
        "component_ids": COMPONENT_IDS,
        "kind_ids": KIND_IDS,
        "fixed_nodes": [node for node, _ in builder.fixed_nodes],
    }
    distributed_mass = sum(
        float(element["mass"][0][0]) * float(element["length"])
        for element in builder.elements
        if element["kind"] == "physical"
    )
    point_mass = sum(matrix[0][0] for matrix in builder.point_masses.values())
    manifest["mass_checks"] = {
        "distributed_mass_kg": distributed_mass,
        "point_mass_kg": point_mass,
        "total_mass_kg": distributed_mass + point_mass,
    }
    (output_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def convert_qblade_model(
    sim_file: Path,
    output_dir: Path,
    *,
    penalty_factor: float = 1.0e4,
    release_ratio: float = 1.0e-10,
    initial_azimuth_deg: float | None = None,
) -> dict[str, object]:
    sim_file = sim_file.resolve()
    sim_lines = _lines(sim_file)
    turbine_file = _resolve(sim_file.parent, _find_value(sim_lines, "TURBFILE"))
    turbine_lines = _lines(turbine_file)
    blade_file = _resolve(turbine_file.parent, _find_value(turbine_lines, "BLADEFILE"))
    structural_file = _resolve(turbine_file.parent, _find_value(turbine_lines, "STRUCTURALFILE"))
    structural_lines = _lines(structural_file)
    structural_dir = structural_file.parent

    number_blades = _int_value(structural_lines, "NUMBLD")
    blade_files = [_resolve(structural_dir, _find_value(structural_lines, f"BLDFILE_{index}")) for index in range(1, number_blades + 1)]
    strut_files: list[Path] = []
    strut_index = 1
    while True:
        value = _find_value(structural_lines, f"STRTFILE_{strut_index}", required=False)
        if not value:
            break
        strut_files.append(_resolve(structural_dir, value))
        strut_index += 1
    tower_file = _resolve(structural_dir, _find_value(structural_lines, "TWRFILE"))
    torque_tube_file = _resolve(structural_dir, _find_value(structural_lines, "TRQTBFILE"))
    substructure_file = _resolve(structural_dir, _find_value(structural_lines, "SUBFILE"))

    blade_geometry, strut_geometry = _parse_blade(blade_file)
    if len(strut_files) != len(strut_geometry):
        raise ValueError(f"QBlade defines {len(strut_geometry)} aerodynamic struts but {len(strut_files)} structural strut files")
    blade_sections = [_parse_section_file(path) for path in blade_files]
    strut_sections = [_parse_section_file(path) for path in strut_files]
    tower_sections = _parse_section_file(tower_file)
    torque_sections = _parse_section_file(torque_tube_file)

    sub_lines = _lines(substructure_file)
    interface = _numeric_table(sub_lines, "TP_INTERFACE_POS", 3)[0][:3]
    sub_joints = {int(row[0]): row[1:4] for row in _numeric_table(sub_lines, "SUBJOINTS", 4)}
    (
        sub_element_rows,
        sub_marker_line,
        sub_header,
        sub_header_line,
        sub_row_lines,
        sub_raw_rows,
    ) = _numeric_table_with_source(sub_lines, "SUBELEMENTS", 20)
    sub_elements = {int(row[0]): row for row in sub_element_rows}
    sub_element_source = {
        int(row[0]): {
            "line_number": line_number,
            "raw_text": raw_text,
            "values": row,
        }
        for row, line_number, raw_text in zip(
            sub_element_rows, sub_row_lines, sub_raw_rows
        )
    }
    sub_members = _numeric_table(sub_lines, "SUBMEMBERS", 10)
    sub_constraints = _numeric_table(sub_lines, "SUBCONSTRAINTS", 12)
    sub_stiffness_tuner = _float_value(sub_lines, "STIFFTUNER", required=False, default=1.0)
    sub_mass_tuner = _float_value(sub_lines, "MASSTUNER", required=False, default=1.0)

    builder = ModelBuilder(penalty_factor=penalty_factor, release_ratio=release_ratio)
    source_files = [sim_file, turbine_file, blade_file, structural_file, *blade_files, *strut_files, tower_file, torque_tube_file, substructure_file]
    source_files = list(dict.fromkeys(path.resolve() for path in source_files))
    source_labels = [_portable_path_label(path) for path in source_files]
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("Portable source labels are not unique; source artifact identity would be ambiguous")

    # Substructure members are assembled with shared QBlade joint nodes.
    sub_node_ids: dict[int, int] = {}
    for joint_id, xyz in sub_joints.items():
        frame = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        sub_node_ids[joint_id] = builder.add_node(xyz, label=f"sub_joint_{joint_id}", component="substructure", frame=frame, half_y=0.5, half_z=0.5)
    for member in sub_members:
        _, joint_a, joint_b, property_id, _, _, _, _, _, maximum_length = member[:10]
        row = sub_elements[int(property_id)]
        sub_candidates = _subelement_damping_candidates(row)
        documented_beta = sub_candidates["qblade_2_0_9_7_documented"]
        source_row = sub_element_source[int(property_id)]
        sub_damping_provenance = {
            "source_label": _portable_path_label(substructure_file),
            "source_table": "SUBELEMENTS",
            "source_header_line": sub_header_line,
            "source_row_line_lower": source_row["line_number"],
            "source_row_line_upper": source_row["line_number"],
            "source_fraction": 0.0,
            "distributed_policy": "constant_SUBELEMENTS_property",
            "scalar_beta_s": documented_beta,
            "station_beta_column_zero_based": 19,
        }
        start_xyz = sub_joints[int(joint_a)]
        stop_xyz = sub_joints[int(joint_b)]
        preferred_y = [1.0, 0.0, 0.0]
        member_frame = _frame(start_xyz, stop_xyz, preferred_y)
        diameter = row[18]
        section_row = [0.0, *row[1:18]]
        stiffness, mass = _section_matrices(
            section_row,
            chord_or_diameter=diameter,
            pitch_axis=0.5,
            geometry_twist_deg=0.0,
            stiffness_tuner=sub_stiffness_tuner,
            mass_tuner=sub_mass_tuner,
        )
        builder.add_element(
            sub_node_ids[int(joint_a)],
            sub_node_ids[int(joint_b)],
            component="substructure",
            kind="physical",
            frame=member_frame,
            stiffness=stiffness,
            mass=mass,
            damping=documented_beta,
            damping_candidates=sub_candidates,
            damping_provenance=sub_damping_provenance,
        )
        builder.update_visual(sub_node_ids[int(joint_a)], member_frame, diameter / 2.0, diameter / 2.0)
        builder.update_visual(sub_node_ids[int(joint_b)], member_frame, diameter / 2.0, diameter / 2.0)
    for constraint in sub_constraints:
        joint_id = int(constraint[1])
        ground_connection = int(constraint[4])
        dofs = [int(value) for value in constraint[6:12]]
        if ground_connection and any(dofs):
            builder.fixed_nodes.append((sub_node_ids[joint_id], dofs))
    if not builder.fixed_nodes:
        raise ValueError("The QBlade substructure has no ground boundary condition")

    origin = interface
    tower_height = _float_value(structural_lines, "TWRHEIGHT")
    torque_height = _float_value(structural_lines, "TRQTBHEIGHT")
    torque_clearance = _float_value(structural_lines, "TRQTBCLEAR")
    rotor_clearance = _float_value(structural_lines, "RTRCLEAR")
    hub_position = _float_value(structural_lines, "HUBPOS")
    bearing_relative = _float_value(structural_lines, "TRQTBCONN")
    bearing_height = torque_clearance + bearing_relative

    def add_vertical_component(
        name: str,
        data: dict[str, object],
        bottom: float,
        height: float,
        extra_heights: Iterable[float],
        start_node: int | None = None,
    ) -> tuple[list[float], list[int]]:
        rows = data["rows"]
        disc = max(int(data["disc"]), 2)
        fractions = _unique(
            [index / (disc - 1) for index in range(disc)]
            + [row[0] for row in rows]
            + [(value - bottom) / height for value in extra_heights if bottom - 1.0e-9 <= value <= bottom + height + 1.0e-9]
            + [fraction for fraction, _ in data["added_masses"]]
        )
        node_ids: list[int] = []
        for index, fraction in enumerate(fractions):
            xyz = [origin[0], origin[1], origin[2] + bottom + height * fraction]
            diameter = _interp(rows, fraction, 18) if len(rows[0]) > 18 else 0.1
            frame = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
            if index == 0 and start_node is not None:
                node_id = start_node
                builder.update_visual(node_id, frame, diameter / 2.0, diameter / 2.0)
            else:
                node_id = builder.add_node(xyz, label=f"{name}_{fraction:.8f}", component=name, frame=frame, half_y=diameter / 2.0, half_z=diameter / 2.0)
            node_ids.append(node_id)
        for index in range(len(fractions) - 1):
            midpoint = (fractions[index] + fractions[index + 1]) / 2.0
            row = [_interp(rows, midpoint, column) for column in range(len(rows[0]))]
            diameter = row[18] if len(row) > 18 else 0.1
            stiffness, mass = _section_matrices(
                row,
                chord_or_diameter=diameter,
                pitch_axis=0.5,
                geometry_twist_deg=0.0,
                stiffness_tuner=float(data["stiffness_tuner"]),
                mass_tuner=float(data["mass_tuner"]),
            )
            frame = _frame(builder.nodes[node_ids[index] - 1]["xyz"], builder.nodes[node_ids[index + 1] - 1]["xyz"], [1.0, 0.0, 0.0])
            candidates, damping_source = _section_damping_candidates(data, midpoint)
            builder.add_element(
                node_ids[index], node_ids[index + 1], component=name, kind="physical",
                frame=frame, stiffness=stiffness, mass=mass,
                damping=candidates["qblade_2_0_9_7_documented"],
                damping_candidates=candidates, damping_provenance=damping_source,
            )
        for fraction, mass in data["added_masses"]:
            builder.add_point_mass(_nearest_node(fractions, node_ids, fraction), mass)
        return fractions, node_ids

    interface_node = min(sub_node_ids.values(), key=lambda node: _norm([builder.nodes[node - 1]["xyz"][i] - origin[i] for i in range(3)]))
    strut_hub_heights = [rotor_clearance + float(strut["height_hub"]) for strut in strut_geometry]
    tower_fractions, tower_nodes = add_vertical_component(
        "tower",
        tower_sections,
        0.0,
        tower_height,
        [hub_position, bearing_height, *strut_hub_heights],
        start_node=interface_node,
    )
    torque_fractions, torque_nodes = add_vertical_component(
        "torque_tube",
        torque_sections,
        torque_clearance,
        torque_height,
        [hub_position, bearing_height, *strut_hub_heights],
    )

    total_blade_height = blade_geometry[-1]["height"] - blade_geometry[0]["height"]
    source_initial_azimuth = _float_value(
        sim_lines, "INITIAL_AZIMUTH", required=False, default=0.0
    )
    initial_azimuth = (
        source_initial_azimuth
        if initial_azimuth_deg is None
        else float(initial_azimuth_deg)
    )
    phase0 = math.radians(initial_azimuth + 90.0)
    blade_meshes: list[tuple[list[float], list[int]]] = []
    strut_meshes: list[list[tuple[list[float], list[int]]]] = []

    for blade_index in range(number_blades):
        phase = phase0 + 2.0 * math.pi * blade_index / number_blades
        radial = [math.cos(phase), math.sin(phase), 0.0]
        tangent = [-math.sin(phase), math.cos(phase), 0.0]
        section_data = blade_sections[blade_index]
        connection_fractions = [
            (float(strut["height_blade"]) - blade_geometry[0]["height"]) / total_blade_height for strut in strut_geometry
        ]
        fractions = _unique(
            [row[0] for row in section_data["rows"]]
            + connection_fractions
            + [fraction for fraction, _ in section_data["added_masses"]]
        )
        node_ids: list[int] = []
        for fraction in fractions:
            height = blade_geometry[0]["height"] + fraction * total_blade_height
            radius = _interp_geometry(blade_geometry, height, "radius")
            tangential_offset = _interp_geometry(blade_geometry, height, "tangential_offset")
            xyz = [
                origin[0] + radius * radial[0] + tangential_offset * tangent[0],
                origin[1] + radius * radial[1] + tangential_offset * tangent[1],
                origin[2] + rotor_clearance + height,
            ]
            chord = _interp_geometry(blade_geometry, height, "chord")
            if node_ids:
                local_frame = _frame(builder.nodes[node_ids[-1] - 1]["xyz"], xyz, tangent)
            else:
                local_frame = [[0.0, tangent[0], -radial[0]], [0.0, tangent[1], -radial[1]], [1.0, 0.0, 0.0]]
            node_ids.append(builder.add_node(xyz, label=f"blade_{blade_index + 1}_{fraction:.8f}", component="blade", frame=local_frame, half_y=chord / 2.0, half_z=0.09 * chord / 2.0))
        for index in range(len(fractions) - 1):
            midpoint = (fractions[index] + fractions[index + 1]) / 2.0
            height = blade_geometry[0]["height"] + midpoint * total_blade_height
            chord = _interp_geometry(blade_geometry, height, "chord")
            pitch_axis = _interp_geometry(blade_geometry, height, "pitch_axis")
            twist = _interp_geometry(blade_geometry, height, "twist")
            row = [_interp(section_data["rows"], midpoint, column) for column in range(len(section_data["rows"][0]))]
            stiffness, mass = _section_matrices(row, chord_or_diameter=chord, pitch_axis=pitch_axis, geometry_twist_deg=twist, stiffness_tuner=float(section_data["stiffness_tuner"]), mass_tuner=float(section_data["mass_tuner"]))
            local_frame = _frame(builder.nodes[node_ids[index] - 1]["xyz"], builder.nodes[node_ids[index + 1] - 1]["xyz"], tangent)
            candidates, damping_source = _section_damping_candidates(section_data, midpoint)
            builder.add_element(
                node_ids[index], node_ids[index + 1], component="blade", kind="physical",
                frame=local_frame, stiffness=stiffness, mass=mass,
                damping=candidates["qblade_2_0_9_7_documented"],
                damping_candidates=candidates, damping_provenance=damping_source,
            )
            builder.update_visual(node_ids[index], local_frame, chord / 2.0, 0.09 * chord / 2.0)
            builder.update_visual(node_ids[index + 1], local_frame, chord / 2.0, 0.09 * chord / 2.0)
        for fraction, mass in section_data["added_masses"]:
            builder.add_point_mass(_nearest_node(fractions, node_ids, fraction), mass)
        blade_meshes.append((fractions, node_ids))

        blade_struts: list[tuple[list[float], list[int]]] = []
        for strut_index, (geometry, section_data) in enumerate(zip(strut_geometry, strut_sections)):
            root = [
                origin[0] + float(geometry["distance_hub"]) * radial[0],
                origin[1] + float(geometry["distance_hub"]) * radial[1],
                origin[2] + rotor_clearance + float(geometry["height_hub"]),
            ]
            blade_height = float(geometry["height_blade"])
            blade_radius = _interp_geometry(blade_geometry, blade_height, "radius")
            blade_offset = _interp_geometry(blade_geometry, blade_height, "tangential_offset")
            tip = [
                origin[0] + blade_radius * radial[0] + blade_offset * tangent[0],
                origin[1] + blade_radius * radial[1] + blade_offset * tangent[1],
                origin[2] + rotor_clearance + blade_height,
            ]
            fractions_s = _unique([row[0] for row in section_data["rows"]] + [fraction for fraction, _ in section_data["added_masses"]])
            node_ids_s: list[int] = []
            for fraction in fractions_s:
                xyz = [root[i] + fraction * (tip[i] - root[i]) for i in range(3)]
                chord = float(geometry["chord_hub"]) + fraction * (float(geometry["chord_blade"]) - float(geometry["chord_hub"]))
                local_frame = _frame(root, tip, tangent)
                component = "lower_strut" if strut_index == 0 else "upper_strut"
                node_ids_s.append(builder.add_node(xyz, label=f"strut_{strut_index + 1}_blade_{blade_index + 1}_{fraction:.8f}", component=component, frame=local_frame, half_y=chord / 2.0, half_z=0.09 * chord / 2.0))
            for index in range(len(fractions_s) - 1):
                midpoint = (fractions_s[index] + fractions_s[index + 1]) / 2.0
                chord = float(geometry["chord_hub"]) + midpoint * (float(geometry["chord_blade"]) - float(geometry["chord_hub"]))
                pitch_axis = float(geometry["pitch_axis_hub"]) + midpoint * (float(geometry["pitch_axis_blade"]) - float(geometry["pitch_axis_hub"]))
                twist = -float(geometry["angle_deg"])
                row = [_interp(section_data["rows"], midpoint, column) for column in range(len(section_data["rows"][0]))]
                stiffness, mass = _section_matrices(row, chord_or_diameter=chord, pitch_axis=pitch_axis, geometry_twist_deg=twist, stiffness_tuner=float(section_data["stiffness_tuner"]), mass_tuner=float(section_data["mass_tuner"]))
                local_frame = _frame(root, tip, tangent)
                component = "lower_strut" if strut_index == 0 else "upper_strut"
                candidates, damping_source = _section_damping_candidates(section_data, midpoint)
                builder.add_element(
                    node_ids_s[index], node_ids_s[index + 1], component=component,
                    kind="physical", frame=local_frame, stiffness=stiffness, mass=mass,
                    damping=candidates["qblade_2_0_9_7_documented"],
                    damping_candidates=candidates, damping_provenance=damping_source,
                )
            for fraction, mass in section_data["added_masses"]:
                builder.add_point_mass(_nearest_node(fractions_s, node_ids_s, fraction), mass)
            blade_struts.append((fractions_s, node_ids_s))
        strut_meshes.append(blade_struts)

    # Apply QBlade connector topology after all physical stiffness scales are known.
    constraints_rows = _numeric_table(structural_lines, "STRUT_BLADE_CONSTRAINTS", 9)
    constraint_by_pair = {(int(row[0]), int(row[1])): [int(value) for value in row[2:8]] for row in constraints_rows}
    vertical_frame = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    for blade_index, blade_struts in enumerate(strut_meshes, start=1):
        phase = phase0 + 2.0 * math.pi * (blade_index - 1) / number_blades
        radial = [math.cos(phase), math.sin(phase), 0.0]
        tangent = [-math.sin(phase), math.cos(phase), 0.0]
        for strut_index, (_, strut_nodes) in enumerate(blade_struts, start=1):
            geometry = strut_geometry[strut_index - 1]
            torque_fraction = (rotor_clearance + float(geometry["height_hub"]) - torque_clearance) / torque_height
            tube_node = _nearest_node(torque_fractions, torque_nodes, torque_fraction)
            radial_frame = _frame(builder.nodes[tube_node - 1]["xyz"], builder.nodes[strut_nodes[0] - 1]["xyz"], tangent)
            builder.add_penalty_element(tube_node, strut_nodes[0], component="rigid_connector", kind="rigid", frame=radial_frame)

            blade_fraction = (float(geometry["height_blade"]) - blade_geometry[0]["height"]) / total_blade_height
            blade_node = _nearest_node(blade_meshes[blade_index - 1][0], blade_meshes[blade_index - 1][1], blade_fraction)
            dofs = constraint_by_pair.get((strut_index, blade_index), [1, 1, 1, 1, 1, 1])
            released = [index for index, constrained in enumerate(dofs) if not constrained]
            connector_frame = _frame(
                [builder.nodes[blade_node - 1]["xyz"][i] - (vertical_frame[i][0] * 1.0e-6) for i in range(3)],
                builder.nodes[blade_node - 1]["xyz"],
                tangent,
            )
            builder.add_penalty_element(blade_node, strut_nodes[-1], component="released_connector" if released else "rigid_connector", kind="released" if released else "rigid", frame=connector_frame, released_dofs=released)

    hub_tower_node = _nearest_node(tower_fractions, tower_nodes, hub_position / tower_height)
    hub_tube_node = _nearest_node(torque_fractions, torque_nodes, (hub_position - torque_clearance) / torque_height)
    bearing_tower_node = _nearest_node(tower_fractions, tower_nodes, bearing_height / tower_height)
    bearing_tube_node = _nearest_node(torque_fractions, torque_nodes, (bearing_height - torque_clearance) / torque_height)
    drivetrain_dynamic = _bool_value(structural_lines, "DRTRDOF", required=False, default=False)
    if drivetrain_dynamic:
        raise ValueError("DRTRDOF=true is not yet supported; a drivetrain torsional spring/mass state must be added explicitly")
    builder.add_penalty_element(hub_tower_node, hub_tube_node, component="drivetrain", kind="drivetrain", frame=vertical_frame)
    builder.add_penalty_element(bearing_tower_node, bearing_tube_node, component="bearing", kind="bearing", frame=vertical_frame, released_dofs=(3,))
    hub_inertia = _float_value(structural_lines, "HUBINER", required=False, default=0.0)
    generator_inertia = _float_value(structural_lines, "GENINER", required=False, default=0.0)
    gearbox_ratio = _float_value(structural_lines, "GBRATIO", required=False, default=1.0)
    hub_mass = _float_value(structural_lines, "HUBMASS", required=False, default=0.0)
    builder.add_point_mass(hub_tube_node, hub_mass, inertia_global=(0.0, 0.0, hub_inertia + generator_inertia * gearbox_ratio**2))

    manifest: dict[str, object] = {
        "schema": "vorlap-qblade-to-owens-v1",
        "converter": _portable_path_label(Path(__file__)),
        "converter_sha256": _sha256(Path(__file__).resolve()),
        "inputs": {label: _sha256(path) for label, path in zip(source_labels, source_files)},
        "input_sizes_bytes": {
            label: path.stat().st_size for label, path in zip(source_labels, source_files)
        },
        "runtime_provenance": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "qblade": {
            "sim_file": _portable_path_label(sim_file),
            "turbine_file": _portable_path_label(turbine_file),
            "structural_file": _portable_path_label(structural_file),
            "blade_file": _portable_path_label(blade_file),
            "number_blades": number_blades,
            "number_struts_per_blade": len(strut_geometry),
            "source_initial_azimuth_deg": source_initial_azimuth,
            "initial_azimuth_deg": initial_azimuth,
            "initial_azimuth_override_deg": initial_azimuth_deg,
            "rotor_clearance_relative_to_tower_base_m": rotor_clearance,
            "tower_base_global_m": origin,
            "hub_inertia_kg_m2": hub_inertia,
            "generator_inertia_reflected_kg_m2": generator_inertia * gearbox_ratio**2,
        },
        "mapping": {
            "units": "SI",
            "gxbeam_axes": "local x follows each beam; local y follows QBlade section X/chord; local z completes a right-handed frame",
            "stiffness_order": ["EA", "GAy", "GAz", "GJ", "EIyy", "EIzz"],
            "qblade_eix_mapping": "EIx -> GXBeam EIyy (QBlade section X maps to GXBeam local y)",
            "qblade_eiy_mapping": "EIy -> GXBeam EIzz (QBlade section Y maps to GXBeam local z)",
            "sectional_mass_matrix": "Positive-definite GXBeam convention: RGX/RGY are orthogonal principal radii at the center of mass, followed by section rotation and parallel-axis transport to the beam reference line",
            "penalty_factor": penalty_factor,
            "released_dof_stiffness_ratio_relative_to_physical_scale": release_ratio,
            "gravity_in_converter": "not applied; selected by Julia runner",
        },
        "damping_semantics": {
            "status": "unresolved_pending_paired_active_binary_decay_probe",
            "selected_damping_candidate_id": None,
            "elements_csv_storage_candidate_id": "qblade_2_0_9_7_documented",
            "elements_csv_runtime_use": "stored for traceability only; the Campbell solve sets structural_damping=false",
            "candidates": DAMPING_CANDIDATES,
            "qblade_2_0_9_7_documented_contract": {
                "subelements": "20 base numeric entries: zero-based row[19] is stiffness-proportional DAMP; optional row[20] is aerodynamic drag",
                "section_properties": "optional zero-based station column 18 named RAYLEIGHDMP_Vec is linearly interpolated; scalar RAYLEIGHDMP is the fallback",
            },
            "qblade_2_0_4_archived_contract": {
                "subelements": "zero-based row[19] is DAMP and optional row[20] is DRAG",
                "section_properties": "file-level scalar RAYLEIGHDMP is retained as the archived-code candidate",
            },
            "reference_header_conflict": {
                "source_label": _portable_path_label(substructure_file),
                "marker_line_number": sub_marker_line,
                "header_line_number": sub_header_line,
                "header_tokens": sub_header,
                "header_raw_text": sub_lines[sub_header_line - 1],
                "source_rows": [sub_element_source[key] for key in sorted(sub_element_source)],
                "interpretation": "The local header labels row[19] Drag and row[20] DAMP, contrary to the documented and archived parser order; both projections remain candidates until the paired decay discriminator is run.",
            },
            "section_source_tables": [
                {
                    "source_label": _portable_path_label(Path(data["path"])),
                    "header_line_number": int(data["header_line_number"]),
                    "header_tokens": data["header"],
                    "station_beta_column_zero_based": data["station_beta_column"],
                    "scalar_rayleigh_beta_s": data["rayleigh"],
                    "row_line_numbers": data["row_line_numbers"],
                    "raw_numeric_rows": data["rows"],
                }
                for data in [*blade_sections, *strut_sections, tower_sections, torque_sections]
            ],
        },
        "limitations": [
            "QBlade Chrono multipoint constraints are represented by massless high-stiffness GXBeam connector elements.",
            "Released connector coordinates use a finite regularization stiffness relative to the corresponding physical stiffness scale.",
            "DRTRDOF=false is represented by a torsion-locked drivetrain connector because rotor speed is prescribed.",
            "QBlade aerodynamic, hydrodynamic, controller, and turbulent-wind loads are not part of this structural eigenanalysis.",
            "The Campbell solve follows OWENS' GXBeam path with structural damping disabled; unresolved QBlade Rayleigh candidates are retained separately and projected only in postprocessing.",
            "The OWENS/GXBeam rotating-frame Campbell formulation applies angular velocity to the complete converted assembly, as does OWENS.AutoCampbellDiagram's GX branch.",
            "Only fully fixed QBlade ground constraints are currently supported by the Julia runner.",
            "Visualization sections are NACA-like ellipses/circles and do not alter stiffness or mass matrices.",
            "The archived QBlade source was used to map normalized centers, pitch-axis offsets, tuners, and radii of gyration.",
            "The sectional mass matrix intentionally uses a positive-definite tensor rotation and parallel-axis transport. A literal reproduction of QBlade's current Chrono call is indefinite for sections in this model because it rotates the radii as coordinates and omits center-of-mass inertia transport.",
        ],
    }
    _write_bundle(output_dir, builder, manifest)
    return manifest


def _find_owens_root(explicit: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get("OWENS_ROOT"):
        candidates.append(Path(os.environ["OWENS_ROOT"]))
    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        (
            repo_root / "OWENS_Toolkit",
            repo_root.parent / "OWENS_Toolkit",
            repo_root.parents[1] / "_archive" / "OWENS_Toolkit",
            Path.home() / "Documents" / "coderepos" / "_archive" / "OWENS_Toolkit",
        )
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "OWENS.jl" / "Project.toml").is_file():
            return candidate
    return None


def _parse_frequencies(path: Path) -> tuple[list[float], list[list[float]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    return [float(row[0]) for row in rows[1:]], [[float(value) for value in row[1:]] for row in rows[1:]]


def _plot_campbell(output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"WARNING: matplotlib is unavailable; Campbell CSV and VTK files were written, but no PNG/PDF plot was made: {exc}", file=sys.stderr)
        return
    rpms, frequencies = _parse_frequencies(output_dir / "campbell_frequencies.csv")
    if not frequencies:
        raise ValueError("Campbell frequency output is empty")
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    mode_count = len(frequencies[0])
    mode_colors = plt.get_cmap("turbo", mode_count)
    for mode in range(mode_count):
        axis.plot(
            rpms,
            [row[mode] for row in frequencies],
            color=mode_colors(mode),
            linewidth=1.35,
            label=f"Mode {mode + 1}",
        )
    for multiple, style in ((1, "--"), (3, ":"), (6, "-.")):
        axis.plot(rpms, [multiple * rpm / 60.0 for rpm in rpms], color="0.25", linestyle=style, linewidth=0.9, label=f"{multiple}P")
    if min(rpms) <= 1.0 <= max(rpms):
        axis.axvline(1.0, color="black", linewidth=0.8, alpha=0.55, label="Parked-study 1 RPM")
    axis.set_xlabel("Rotor speed (RPM)")
    axis.set_ylabel("Natural frequency (Hz)")
    axis.set_title("VorLap 10 m H-VAWT: OWENS/GXBeam Campbell diagram")
    axis.grid(True, linewidth=0.45, alpha=0.35)
    if min(rpms) < max(rpms):
        axis.set_xlim(min(rpms), max(rpms))
    axis.set_ylim(bottom=0.0)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=7.5, frameon=False)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"campbell_diagram.{suffix}", dpi=220)
    plt.close(figure)


def _command_output(command: Sequence[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def _execution_provenance(project: Path, julia: str) -> dict[str, object]:
    """Capture version/commit state needed to reproduce a Campbell artifact."""

    try:
        owens_commit = _command_output(("git", "rev-parse", "HEAD"), cwd=project)
        owens_status = _command_output(("git", "status", "--porcelain"), cwd=project)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot establish OWENS git provenance at {project}: {exc}") from exc
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "julia_version_output": _command_output((julia, "--version")),
        "owens_git_commit": owens_commit,
        "owens_git_dirty": bool(owens_status),
        "owens_git_status_sha256": hashlib.sha256(owens_status.encode("utf-8")).hexdigest(),
    }


def run_owens(
    output_dir: Path,
    *,
    owens_root: Path,
    julia: str,
    rpms: str,
    modes: int,
    save_rpms: str,
    save_modes: str,
    vtk_modes: str,
    mode_scaling: float,
    vtk_steps: int,
    gravity: bool,
) -> dict[str, object]:
    project = owens_root / "OWENS.jl"
    runner = Path(__file__).with_name("run_owens_campbell.jl")
    command = [
        julia,
        f"--project={project}",
        str(runner),
        "--model-dir",
        str(output_dir),
        "--rpms",
        rpms,
        "--modes",
        str(modes),
        "--save-rpms",
        save_rpms,
        "--save-modes",
        save_modes,
        "--vtk-modes",
        vtk_modes,
        "--mode-scaling",
        str(mode_scaling),
        "--vtk-steps",
        str(vtk_steps),
        "--gravity",
        "1" if gravity else "0",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    _plot_campbell(output_dir)
    request = {
        "schema": "vorlap-owens-campbell-request-v1",
        "conversion_manifest_sha256": _sha256(output_dir / "conversion_manifest.json"),
        "runner_sha256": _sha256(runner),
        "owens_project": "OWENS.jl",
        "owens_project_toml_sha256": _sha256(project / "Project.toml"),
        "owens_manifest_toml_sha256": _sha256(project / "Manifest.toml"),
        "julia_executable": Path(julia).name,
        "execution_provenance": _execution_provenance(project, julia),
        "rpms": rpms,
        "modes": modes,
        "save_rpms": save_rpms,
        "save_modes": save_modes,
        "vtk_modes": vtk_modes,
        "mode_scaling": mode_scaling,
        "vtk_steps": vtk_steps,
        "gravity": gravity,
        "structural_damping": False,
    }
    (output_dir / "owens_request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return request


def _request_is_reusable(output_dir: Path, expected: dict[str, object]) -> bool:
    request_file = output_dir / "owens_request.json"
    required = (
        output_dir / "node_metadata.csv",
        output_dir / "element_damping_candidates.csv",
        output_dir / "campbell_frequencies.csv",
        output_dir / "campbell_eigenvalues.csv",
        output_dir / "campbell_mode_tracking.csv",
        output_dir / "campbell_mode_shapes.csv",
        output_dir / "campbell_mode_properties.csv",
        output_dir / "campbell_mode_energy_by_component.csv",
        output_dir / "campbell_mode_energy_by_element.csv",
        output_dir / "campbell_mode_orthogonality.csv",
        output_dir / "owens_run_metadata.txt",
        output_dir / "paraview" / "converted_qblade_geometry.vtu",
    )
    if not request_file.is_file() or not all(path.is_file() for path in required):
        return False
    try:
        actual = json.loads(request_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("convert", "run"))
    parser.add_argument("--sim", type=Path, required=True, help="QBlade .sim file to convert")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for the converted model and Campbell outputs")
    parser.add_argument("--penalty-factor", type=float, default=1.0e4, help="Connector stiffness relative to the stiffest physical section")
    parser.add_argument("--release-ratio", type=float, default=1.0e-10, help="Regularized released-DOF stiffness relative to the physical stiffness scale")
    parser.add_argument(
        "--initial-azimuth-deg",
        type=float,
        help="Override INITIAL_AZIMUTH for this structural conversion while retaining the canonical source .sim hash",
    )
    parser.add_argument("--owens-root", type=Path, help="OWENS_Toolkit root; alternatively set OWENS_ROOT")
    parser.add_argument("--julia", default=os.environ.get("JULIA", "julia"))
    parser.add_argument("--if-available", action="store_true", help="Return successfully after conversion when Julia/OWENS is unavailable")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse a complete Campbell result only when its inputs and run options match exactly")
    parser.add_argument("--rpms", default="0,0.25,0.5,0.75,1,1.5,2,3,5,7.5,10")
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--save-rpms", default="0,1", help="RPM values for ParaView mode-shape animation")
    parser.add_argument(
        "--save-modes",
        default="1,2,3,4,5,6,7,8,9,10,11,12",
        help="One-based physical modes whose complete action/energy evidence is saved",
    )
    parser.add_argument(
        "--vtk-modes",
        default="6,12",
        help="Subset of saved modes for which ParaView animation files are written",
    )
    parser.add_argument("--mode-scaling", type=float, default=3.0)
    parser.add_argument("--vtk-steps", type=int, default=36)
    parser.add_argument("--no-gravity", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.penalty_factor <= 1.0 or not 0.0 < args.release_ratio < 1.0:
        raise ValueError("penalty-factor must exceed one and release-ratio must lie between zero and one")
    if args.modes < 1 or args.vtk_steps < 4:
        raise ValueError("modes must be positive and vtk-steps must be at least four")
    manifest = convert_qblade_model(
        args.sim,
        args.output_dir,
        penalty_factor=args.penalty_factor,
        release_ratio=args.release_ratio,
        initial_azimuth_deg=args.initial_azimuth_deg,
    )
    mass = manifest["mass_checks"]
    print(
        f"Converted {manifest['mesh']['node_count']} nodes and {manifest['mesh']['element_count']} elements; "
        f"structural-plus-point mass={mass['total_mass_kg']:.3f} kg."
    )
    if args.command == "convert":
        return 0
    owens_root = _find_owens_root(args.owens_root)
    julia = shutil.which(args.julia)
    if owens_root is None or julia is None:
        message = "OWENS/GXBeam run skipped because " + ("OWENS_ROOT was not found" if owens_root is None else f"Julia executable {args.julia!r} was not found")
        if args.if_available:
            print(message)
            return 0
        raise RuntimeError(message)
    output_dir = args.output_dir.resolve()
    expected_request = {
        "schema": "vorlap-owens-campbell-request-v1",
        "conversion_manifest_sha256": _sha256(output_dir / "conversion_manifest.json"),
        "runner_sha256": _sha256(Path(__file__).with_name("run_owens_campbell.jl")),
        "owens_project": "OWENS.jl",
        "owens_project_toml_sha256": _sha256(owens_root / "OWENS.jl" / "Project.toml"),
        "owens_manifest_toml_sha256": _sha256(owens_root / "OWENS.jl" / "Manifest.toml"),
        "julia_executable": Path(julia).name,
        "execution_provenance": _execution_provenance(
            owens_root / "OWENS.jl", julia
        ),
        "rpms": args.rpms,
        "modes": args.modes,
        "save_rpms": args.save_rpms,
        "save_modes": args.save_modes,
        "vtk_modes": args.vtk_modes,
        "mode_scaling": args.mode_scaling,
        "vtk_steps": args.vtk_steps,
        "gravity": not args.no_gravity,
        "structural_damping": False,
    }
    if args.reuse_existing and _request_is_reusable(output_dir, expected_request):
        print(f"Reusing matching OWENS/GXBeam Campbell outputs: {output_dir}")
        return 0
    run_owens(
        output_dir,
        owens_root=owens_root,
        julia=julia,
        rpms=args.rpms,
        modes=args.modes,
        save_rpms=args.save_rpms,
        save_modes=args.save_modes,
        vtk_modes=args.vtk_modes,
        mode_scaling=args.mode_scaling,
        vtk_steps=args.vtk_steps,
        gravity=not args.no_gravity,
    )
    print(f"OWENS/GXBeam Campbell outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
