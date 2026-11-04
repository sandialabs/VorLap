"""QBlade model definition parsers used by VorLap conversion and validation."""

from __future__ import annotations

import os
import re
from typing import Dict, List

import numpy as np

from .qblade_files import (
    QBLADE_KEYVALUE_RE,
    first_qblade_keyword_value,
    parse_qblade_keywords,
    resolve_qblade_path_text,
)


def _to_float(value: str, default: float = 0.0) -> float:
    """Best-effort string-to-float conversion with a fallback default."""
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def _to_int(value: str, default: int = 0) -> int:
    """Best-effort string-to-int conversion with a fallback default."""
    try:
        return int(round(float(str(value).strip())))
    except Exception:
        return int(default)


def load_qblade_simulation_definition(sim_path: str) -> Dict[str, object]:
    """
    Load key fields from a QBlade `.sim` file.

    Returns a dictionary with parsed and path-resolved entries used by VorLap.
    """
    parsed = parse_qblade_keywords(sim_path)

    turbfile_raw = first_qblade_keyword_value(parsed, "TURBFILE", "")
    turbfile_path = (
        resolve_qblade_path_text(sim_path, turbfile_raw) if turbfile_raw else ""
    )

    density = _to_float(
        first_qblade_keyword_value(parsed, "DENSITYAIR", "1.225"), 1.225
    )
    nu = _to_float(
        first_qblade_keyword_value(parsed, "VISCOSITYAIR", "1.81e-5"), 1.81e-5
    )
    mu = density * nu

    hor_angle_deg = _to_float(
        first_qblade_keyword_value(parsed, "HORANGLE", "0.0"), 0.0
    )
    vert_angle_deg = _to_float(
        first_qblade_keyword_value(parsed, "VERTANGLE", "0.0"), 0.0
    )
    mean_inflow = _to_float(first_qblade_keyword_value(parsed, "MEANINF", "0.0"), 0.0)

    timestep = _to_float(first_qblade_keyword_value(parsed, "TIMESTEP", "0.01"), 0.01)
    numtimesteps = max(
        1, _to_int(first_qblade_keyword_value(parsed, "NUMTIMESTEPS", "1"), 1)
    )
    initial_azimuth = _to_float(
        first_qblade_keyword_value(parsed, "INITIAL_AZIMUTH", "0.0"), 0.0
    )
    initial_yaw = _to_float(
        first_qblade_keyword_value(parsed, "INITIAL_YAW", "0.0"), 0.0
    )

    globpos_x = _to_float(first_qblade_keyword_value(parsed, "GLOBPOS_X", "0.0"), 0.0)
    globpos_y = _to_float(first_qblade_keyword_value(parsed, "GLOBPOS_Y", "0.0"), 0.0)
    globpos_z = _to_float(first_qblade_keyword_value(parsed, "GLOBPOS_Z", "0.0"), 0.0)

    return {
        "sim_path": os.path.abspath(sim_path),
        "turbfile": turbfile_raw,
        "turbfile_path": turbfile_path,
        "fluid_density": density,
        "fluid_kinematic_viscosity": nu,
        "fluid_dynamic_viscosity": mu,
        "hor_angle_deg": hor_angle_deg,
        "vert_angle_deg": vert_angle_deg,
        "mean_inflow": mean_inflow,
        "timestep": timestep,
        "numtimesteps": numtimesteps,
        "initial_azimuth_deg": initial_azimuth,
        "initial_yaw_deg": initial_yaw,
        "global_position": np.array([globpos_x, globpos_y, globpos_z], dtype=float),
    }


def load_qblade_turbine_definition(trb_path: str) -> Dict[str, object]:
    """Load key fields from a QBlade `.trb` turbine definition file."""
    parsed = parse_qblade_keywords(trb_path)

    bladefile_raw = first_qblade_keyword_value(parsed, "BLADEFILE", "")
    bladefile_path = (
        resolve_qblade_path_text(trb_path, bladefile_raw) if bladefile_raw else ""
    )
    structuralfile_raw = first_qblade_keyword_value(parsed, "STRUCTURALFILE", "")
    structuralfile_path = (
        resolve_qblade_path_text(trb_path, structuralfile_raw)
        if structuralfile_raw
        else ""
    )

    tower_height = _to_float(
        first_qblade_keyword_value(parsed, "TOWERHEIGHT", "0.0"), 0.0
    )
    overhang = _to_float(first_qblade_keyword_value(parsed, "OVERHANG", "0.0"), 0.0)
    shaft_tilt = _to_float(first_qblade_keyword_value(parsed, "SHAFTTILT", "0.0"), 0.0)
    if structuralfile_path and os.path.isfile(structuralfile_path):
        structural = parse_qblade_keywords(structuralfile_path)
        tower_height = _to_float(
            first_qblade_keyword_value(structural, "TWRHEIGHT", str(tower_height)),
            tower_height,
        )
        overhang = _to_float(
            first_qblade_keyword_value(structural, "OVERHANG", str(overhang)),
            overhang,
        )
        shaft_tilt = _to_float(
            first_qblade_keyword_value(structural, "SHFTTILT", str(shaft_tilt)),
            shaft_tilt,
        )

    return {
        "trb_path": os.path.abspath(trb_path),
        "bladefile": bladefile_raw,
        "bladefile_path": bladefile_path,
        "num_blades": max(
            1, _to_int(first_qblade_keyword_value(parsed, "NUMBLADES", "1"), 1)
        ),
        "turbine_type": _to_int(first_qblade_keyword_value(parsed, "TURBTYPE", "1"), 1),
        "structuralfile": structuralfile_raw,
        "structuralfile_path": structuralfile_path,
        "tower_height": tower_height,
        "overhang": overhang,
        "shaft_tilt_deg": shaft_tilt,
        "tower_top_radius": _to_float(
            first_qblade_keyword_value(parsed, "TOWERTOPRAD", "0.0"), 0.0
        ),
        "tower_bottom_radius": _to_float(
            first_qblade_keyword_value(parsed, "TOWERBOTRAD", "0.0"), 0.0
        ),
    }


def load_qblade_blade_definition(bld_path: str) -> Dict[str, object]:
    """
    Load blade and strut geometry tables from a QBlade `.bld` definition file.

    Returns a dictionary with `num_blades`, `blade_rows`, and `strut_rows`.
    """
    if not os.path.isfile(bld_path):
        raise FileNotFoundError(f"QBlade blade file does not exist: {bld_path}")

    with open(bld_path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    parsed = parse_qblade_keywords(bld_path)

    blade_header_idx = -1
    for i, raw in enumerate(lines):
        is_vawt_header = "HEIGHT_[m]" in raw and "CHORD_[m]" in raw
        is_hawt_header = "POS_[m]" in raw and "CHORD_[m]" in raw
        if is_vawt_header or is_hawt_header:
            blade_header_idx = i
            break
    if blade_header_idx < 0:
        raise ValueError(
            f"Could not locate blade data table in QBlade blade file: {bld_path}"
        )

    blade_columns = re.split(r"\s+", lines[blade_header_idx].strip())
    blade_rows: List[Dict[str, str]] = []
    for i in range(blade_header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            if blade_rows:
                break
            continue
        if line.startswith("-") or line.startswith("STRUT_") or "Strut Data" in line:
            break
        tokens = re.split(r"\s+", line)
        if len(tokens) < len(blade_columns):
            break
        row = {blade_columns[j]: tokens[j] for j in range(len(blade_columns))}
        blade_rows.append(row)

    strut_rows: List[Dict[str, str]] = []
    i = 0
    while i < len(lines):
        marker = lines[i].strip()
        if re.fullmatch(r"STRUT_\d+", marker):
            row: Dict[str, str] = {}
            row["STRUT_MARKER"] = marker
            i += 1
            while i < len(lines):
                end_marker = lines[i].strip()
                if end_marker.startswith("END_STRUT_"):
                    break
                match = QBLADE_KEYVALUE_RE.match(lines[i].strip())
                if match:
                    row[match.group("keyword").strip()] = match.group("value").strip()
                i += 1
            strut_rows.append(row)
        i += 1

    return {
        "bld_path": os.path.abspath(bld_path),
        "rotor_type": first_qblade_keyword_value(parsed, "ROTORTYPE", "")
        .strip()
        .upper(),
        "num_blades": max(
            1, _to_int(first_qblade_keyword_value(parsed, "NUMBLADES", "1"), 1)
        ),
        "blade_rows": blade_rows,
        "strut_rows": strut_rows,
    }


__all__ = [
    "load_qblade_blade_definition",
    "load_qblade_simulation_definition",
    "load_qblade_turbine_definition",
]
