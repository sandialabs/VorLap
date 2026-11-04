#!/usr/bin/env python3
"""Audit a TurbSim box against the QBlade VAWT aerodynamic swept domain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VORLAP_ROOT = SCRIPT_DIR.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def tagged_value(path: Path, tag: str) -> str:
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if tag in fields:
            return fields[0].strip('"')
    raise ValueError(f"{path} has no {tag} value.")


def blade_rows(path: Path) -> list[tuple[float, float, float]]:
    rows = []
    active = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.strip().startswith("HEIGHT_[m]"):
            active = True
            continue
        if active and raw.strip().startswith("-"):
            if rows:
                break
            continue
        if not active or not raw.strip():
            continue
        fields = raw.split()
        try:
            rows.append((float(fields[0]), float(fields[1]), float(fields[2])))
        except (IndexError, ValueError):
            if rows:
                break
    if not rows:
        raise ValueError(f"No VAWT blade table found in {path}.")
    return rows


def strut_definitions(path: Path) -> list[dict[str, float]]:
    required = {
        "CHORDHUB_STR",
        "CHORDBLD_STR",
        "HGTBLD_STR",
        "HGTHUB_STR",
        "DSTHUB_STR",
    }
    definitions: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("STRUT_"):
            if current is not None:
                raise ValueError(f"Nested strut definition in {path}.")
            current = {}
            continue
        if stripped.startswith("END_STRUT_"):
            if current is None or set(current) != required:
                missing = sorted(required - set(current or {}))
                raise ValueError(f"Incomplete strut definition in {path}; missing {missing}.")
            definitions.append(current)
            current = None
            continue
        if current is None:
            continue
        fields = stripped.split()
        if len(fields) >= 2 and fields[1] in required:
            current[fields[1]] = float(fields[0])
    if current is not None:
        raise ValueError(f"Unterminated strut definition in {path}.")
    if not definitions:
        raise ValueError(f"No VAWT strut definitions found in {path}.")
    return definitions


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(VORLAP_ROOT))
    except ValueError:
        return str(path)


def build_report(args: argparse.Namespace) -> dict:
    turbine = args.turbine.expanduser().resolve()
    blade = args.blade.expanduser().resolve()
    simulation = args.simulation.expanduser().resolve()
    rows = blade_rows(blade)
    struts = strut_definitions(blade)
    clearance = float(tagged_value(turbine, "CLEARANCE"))
    blade_count = int(tagged_value(turbine, "NUMBLADES"))
    heights = [row[0] for row in rows]
    chords = [row[1] for row in rows]
    radii = [row[2] for row in rows]
    max_radius = max(radii)
    max_chord = max(chords)
    max_strut_chord = max(
        max(row["CHORDHUB_STR"], row["CHORDBLD_STR"]) for row in struts
    )
    max_strut_hub_radius = max(row["DSTHUB_STR"] for row in struts)
    # Adding one full chord is deliberately conservative for arbitrary chord orientation.
    swept_radius = max(
        max_radius + max_chord,
        max(max_radius, max_strut_hub_radius) + max_strut_chord,
    )
    strut_heights = [
        value
        for row in struts
        for value in (row["HGTBLD_STR"], row["HGTHUB_STR"])
    ]
    rotor_bottom = clearance + min(heights + strut_heights)
    rotor_top = clearance + max(heights + strut_heights)

    grid_width = float(args.grid_width)
    grid_height = float(args.grid_height)
    hub_height = float(args.hub_height)
    ref_height = float(args.ref_height)
    num_y = int(args.num_grid_y)
    num_z = int(args.num_grid_z)
    min_margin = float(args.min_margin)
    if min(num_y, num_z) < 2:
        raise ValueError("Grid dimensions must be at least two.")
    numeric = (grid_width, grid_height, hub_height, ref_height, min_margin)
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric):
        raise ValueError("Grid dimensions, heights, and minimum margin must be finite and positive.")

    lateral_min = -0.5 * grid_width
    lateral_max = 0.5 * grid_width
    vertical_min = hub_height - 0.5 * grid_height
    vertical_max = hub_height + 0.5 * grid_height
    lateral_margin = min(lateral_max - swept_radius, -swept_radius - lateral_min)
    lower_margin = rotor_bottom - vertical_min
    upper_margin = vertical_max - rotor_top
    spacing_y = grid_width / (num_y - 1)
    spacing_z = grid_height / (num_z - 1)
    sim_ref_height = float(tagged_value(simulation, "REFHEIGHT"))
    template_horizontal_angle = float(tagged_value(simulation, "HORANGLE"))
    template_vertical_angle = float(tagged_value(simulation, "VERTANGLE"))
    horizontal_angle = float(args.horizontal_inflow_angle)
    vertical_angle = float(args.vertical_inflow_angle)
    issues = []
    if lateral_margin < min_margin:
        issues.append("lateral_margin_below_minimum")
    if lower_margin < min_margin:
        issues.append("lower_vertical_margin_below_minimum")
    if upper_margin < min_margin:
        issues.append("upper_vertical_margin_below_minimum")
    if spacing_y > max_radius / 4.0 or spacing_z > max_radius / 4.0:
        issues.append("grid_spacing_exceeds_quarter_radius")
    if not math.isclose(ref_height, sim_ref_height, abs_tol=1.0e-12):
        issues.append("turbsim_and_qblade_reference_heights_differ")
    if horizontal_angle != 0.0 or vertical_angle != 0.0:
        issues.append("nonzero_inflow_angle_requires_rotated_domain_review")

    report = {
        "schema_version": 1,
        "passed": not issues,
        "scope": (
            "QBlade turbulent inflow coverage for the rotating VAWT blades, struts, and all "
            "VorLap aerodynamic action points. The fixed foundation and lower non-aerodynamic "
            "support are outside this sampling requirement."
        ),
        "source_files": {
            "turbine": {"path": portable_path(turbine), "sha256": sha256_file(turbine)},
            "blade": {"path": portable_path(blade), "sha256": sha256_file(blade)},
            "simulation": {"path": portable_path(simulation), "sha256": sha256_file(simulation)},
        },
        "qblade_conventions": {
            "wind_box_cross_plane": "lateral y by vertical z",
            "center_height_parameter": "TurbSim HubHt",
            "reference_height_parameter": "TurbSim RefHt equals QBlade REFHEIGHT",
            "wind_auto_shift": (
                "QBlade WINDAUTOSHIFT is a time shift of rotor diameter divided by mean wind "
                "speed; it does not change the cross-plane bounds."
            ),
            "horizontal_inflow_angle_deg": horizontal_angle,
            "vertical_inflow_angle_deg": vertical_angle,
            "source_template_horizontal_inflow_angle_deg": template_horizontal_angle,
            "source_template_vertical_inflow_angle_deg": template_vertical_angle,
            "production_override": (
                "The campaign writes HORANGLE and VERTANGLE from the archived manifest-row "
                "environment; the production values above supersede the source template."
            ),
        },
        "aerodynamic_geometry": {
            "blade_height_min_m": min(heights),
            "blade_height_max_m": max(heights),
            "clearance_m": clearance,
            "rotor_bottom_m": rotor_bottom,
            "rotor_top_m": rotor_top,
            "maximum_blade_radius_m": max_radius,
            "maximum_chord_m": max_chord,
            "strut_definitions_per_blade": len(struts),
            "total_strut_count": blade_count * len(struts),
            "strut_endpoint_height_min_m": min(strut_heights),
            "strut_endpoint_height_max_m": max(strut_heights),
            "maximum_strut_chord_m": max_strut_chord,
            "maximum_strut_hub_radius_m": max_strut_hub_radius,
            "conservative_swept_radius_m": swept_radius,
        },
        "grid": {
            "num_grid_y": num_y,
            "num_grid_z": num_z,
            "grid_width_m": grid_width,
            "grid_height_m": grid_height,
            "hub_height_m": hub_height,
            "ref_height_m": ref_height,
            "lateral_bounds_m": [lateral_min, lateral_max],
            "vertical_bounds_m": [vertical_min, vertical_max],
            "spacing_y_m": spacing_y,
            "spacing_z_m": spacing_z,
        },
        "margins": {
            "required_minimum_m": min_margin,
            "lateral_each_side_m": lateral_margin,
            "below_rotor_m": lower_margin,
            "above_rotor_m": upper_margin,
        },
        "issues": issues,
    }
    report["review_sha256"] = sha256_payload(report)
    return report


def parse_args() -> argparse.Namespace:
    model = VORLAP_ROOT / "examples" / "ReferenceHVAWTQBladeModel"
    turbine_root = model / "10m_HVAWT_reference_turbine"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--turbine", type=Path, default=turbine_root / "10m_HVAWT_reference_turbine.trb")
    parser.add_argument("--blade", type=Path, default=turbine_root / "Aero" / "10m_HVAWT_reference_rotor.bld")
    parser.add_argument("--simulation", type=Path, default=model / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim")
    parser.add_argument("--num-grid-y", type=int, default=26)
    parser.add_argument("--num-grid-z", type=int, default=38)
    parser.add_argument("--grid-width", type=float, default=26.0)
    parser.add_argument("--grid-height", type=float, default=26.0)
    parser.add_argument("--hub-height", type=float, default=18.0)
    parser.add_argument("--ref-height", type=float, default=18.0)
    parser.add_argument("--horizontal-inflow-angle", type=float, default=0.0)
    parser.add_argument("--vertical-inflow-angle", type=float, default=0.0)
    parser.add_argument("--min-margin", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_report(args)
        if args.out is not None:
            write_json(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 2
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
