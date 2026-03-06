"""
File input/output operations for the VorLap package.
"""

import os
import csv
import numpy as np
import pandas as pd
import h5py
from typing import List, Dict, Tuple, Optional
import re
import warnings
from .structs import AirfoilFFT, Component, InflowTimeSeries, VIV_Params


_QBLADE_KEYVALUE_RE = re.compile(r"^(?P<value>.*?)\s+(?P<keyword>[A-Z][A-Z0-9_]+)\s*(?:-.*)?$")


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


def _resolve_qblade_path(base_file: str, candidate: str) -> str:
    """Resolve a QBlade relative file path against its parent file location."""
    candidate = str(candidate).strip()
    if not candidate:
        return ""
    if os.path.isabs(candidate):
        return candidate
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(base_file)), candidate))


def _parse_qblade_keywords(path: str) -> Dict[str, List[str]]:
    """
    Parse QBlade keyword/value lines into a dictionary of keyword -> list of values.

    QBlade definition files generally store one value followed by an uppercase keyword
    and then a comment field.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"QBlade file does not exist: {path}")

    parsed: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("-"):
                continue
            m = _QBLADE_KEYVALUE_RE.match(line)
            if not m:
                continue
            keyword = m.group("keyword").strip()
            value = m.group("value").strip()
            parsed.setdefault(keyword, []).append(value)
    return parsed


def _first_keyword_value(parsed: Dict[str, List[str]], keyword: str, default: str = "") -> str:
    values = parsed.get(keyword, [])
    return values[0] if values else default


def _infer_airfoil_id_from_polar(polar_path: str, default_airfoil_id: str = "default") -> str:
    """
    Infer a VorLap airfoil identifier from a QBlade polar file path.

    Examples:
        `Polars/NACA_0018_RFoil_MultiRePolar.plr` -> `NACA0018`
        `Polars/flat_plate_MultiRePolar.plr` -> `default`
    """
    base = os.path.basename(str(polar_path))
    if not base:
        return default_airfoil_id

    m = re.search(r"(NACA[_-]?\d{4})", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).replace("_", "").replace("-", "").upper()
    return default_airfoil_id


def _infer_thickness_from_airfoil_id(airfoil_id: str, default_thickness: float = 0.12) -> float:
    """Infer a thickness ratio from an identifier such as `NACA0018`."""
    m = re.search(r"(\d{4})$", str(airfoil_id))
    if m:
        return max(0.01, min(1.0, float(m.group(1)[2:]) / 100.0))
    return float(default_thickness)


def load_qblade_simulation_definition(sim_path: str) -> Dict[str, object]:
    """
    Load key fields from a QBlade `.sim` file.

    Returns a dictionary with parsed and path-resolved entries used by VorLap.
    """
    parsed = _parse_qblade_keywords(sim_path)

    turbfile_raw = _first_keyword_value(parsed, "TURBFILE", "")
    turbfile_path = _resolve_qblade_path(sim_path, turbfile_raw) if turbfile_raw else ""

    density = _to_float(_first_keyword_value(parsed, "DENSITYAIR", "1.225"), 1.225)
    nu = _to_float(_first_keyword_value(parsed, "VISCOSITYAIR", "1.81e-5"), 1.81e-5)
    mu = density * nu

    hor_angle_deg = _to_float(_first_keyword_value(parsed, "HORANGLE", "0.0"), 0.0)
    vert_angle_deg = _to_float(_first_keyword_value(parsed, "VERTANGLE", "0.0"), 0.0)
    mean_inflow = _to_float(_first_keyword_value(parsed, "MEANINF", "0.0"), 0.0)

    timestep = _to_float(_first_keyword_value(parsed, "TIMESTEP", "0.01"), 0.01)
    numtimesteps = max(1, _to_int(_first_keyword_value(parsed, "NUMTIMESTEPS", "1"), 1))
    initial_azimuth = _to_float(_first_keyword_value(parsed, "INITIAL_AZIMUTH", "0.0"), 0.0)

    globpos_x = _to_float(_first_keyword_value(parsed, "GLOBPOS_X", "0.0"), 0.0)
    globpos_y = _to_float(_first_keyword_value(parsed, "GLOBPOS_Y", "0.0"), 0.0)
    globpos_z = _to_float(_first_keyword_value(parsed, "GLOBPOS_Z", "0.0"), 0.0)

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
        "global_position": np.array([globpos_x, globpos_y, globpos_z], dtype=float),
    }


def load_qblade_turbine_definition(trb_path: str) -> Dict[str, object]:
    """Load key fields from a QBlade `.trb` turbine definition file."""
    parsed = _parse_qblade_keywords(trb_path)

    bladefile_raw = _first_keyword_value(parsed, "BLADEFILE", "")
    bladefile_path = _resolve_qblade_path(trb_path, bladefile_raw) if bladefile_raw else ""
    structuralfile_raw = _first_keyword_value(parsed, "STRUCTURALFILE", "")
    structuralfile_path = _resolve_qblade_path(trb_path, structuralfile_raw) if structuralfile_raw else ""

    return {
        "trb_path": os.path.abspath(trb_path),
        "bladefile": bladefile_raw,
        "bladefile_path": bladefile_path,
        "num_blades": max(1, _to_int(_first_keyword_value(parsed, "NUMBLADES", "1"), 1)),
        "structuralfile": structuralfile_raw,
        "structuralfile_path": structuralfile_path,
        "tower_height": _to_float(_first_keyword_value(parsed, "TOWERHEIGHT", "0.0"), 0.0),
        "tower_top_radius": _to_float(_first_keyword_value(parsed, "TOWERTOPRAD", "0.0"), 0.0),
        "tower_bottom_radius": _to_float(_first_keyword_value(parsed, "TOWERBOTRAD", "0.0"), 0.0),
    }


def load_qblade_blade_definition(bld_path: str) -> Dict[str, object]:
    """
    Load blade and strut geometry tables from a QBlade `.bld` definition file.

    Returns:
        Dictionary with fields:
            - `num_blades`
            - `blade_rows` (list of dicts)
            - `strut_rows` (list of dicts)
    """
    if not os.path.isfile(bld_path):
        raise FileNotFoundError(f"QBlade blade file does not exist: {bld_path}")

    with open(bld_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    parsed = _parse_qblade_keywords(bld_path)

    blade_header_idx = -1
    for i, raw in enumerate(lines):
        if "HEIGHT_[m]" in raw and "CHORD_[m]" in raw:
            blade_header_idx = i
            break
    if blade_header_idx < 0:
        raise ValueError(f"Could not locate blade data table in QBlade blade file: {bld_path}")

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
                m = _QBLADE_KEYVALUE_RE.match(lines[i].strip())
                if m:
                    row[m.group("keyword").strip()] = m.group("value").strip()
                i += 1
            strut_rows.append(row)
        i += 1

    return {
        "bld_path": os.path.abspath(bld_path),
        "num_blades": max(1, _to_int(_first_keyword_value(parsed, "NUMBLADES", "1"), 1)),
        "blade_rows": blade_rows,
        "strut_rows": strut_rows,
    }


def convert_qblade_to_vorlap_inputs(
    sim_path: str,
    default_airfoil_id: str = "default",
    include_struts: bool = True,
    include_tower: bool = True,
    tower_airfoil_id: str = "cylinder",
) -> Tuple[List[Component], VIV_Params, List[str]]:
    """
    Convert a QBlade `.sim` setup into equivalent VorLap components and baseline parameters.

    This supports the "existing LOADINGFILE path" workflow where VorLap reconstructs
    non-mean nodal loads and exports them for QBlade external-loading ingestion.

    Returns:
        components: VorLap components equivalent to QBlade blade/strut geometry.
        viv_params: Baseline parameters inferred from QBlade simulation/turbine files.
        qblade_node_ids: Per-node QBlade location IDs aligned with VorLap node ordering.

    Args:
        include_struts: Include strut components when available in the blade definition.
        include_tower: Include a tower component inferred from `.trb` tower fields.
    """
    sim = load_qblade_simulation_definition(sim_path)
    if not sim["turbfile_path"]:
        raise ValueError("QBlade .sim file does not specify TURBFILE.")

    trb = load_qblade_turbine_definition(str(sim["turbfile_path"]))
    if not trb["bladefile_path"]:
        raise ValueError("QBlade .trb file does not specify BLADEFILE.")

    bld = load_qblade_blade_definition(str(trb["bladefile_path"]))

    blade_rows = bld["blade_rows"]
    if not blade_rows:
        raise ValueError("QBlade blade table is empty.")

    heights = np.array([_to_float(r.get("HEIGHT_[m]", "0.0"), 0.0) for r in blade_rows], dtype=float)
    chords = np.array([_to_float(r.get("CHORD_[m]", "0.0"), 0.0) for r in blade_rows], dtype=float)
    radii = np.array([_to_float(r.get("RADIUS_[m]", "0.0"), 0.0) for r in blade_rows], dtype=float)
    twists = np.array([_to_float(r.get("TWIST_[deg]", "0.0"), 0.0) for r in blade_rows], dtype=float)
    offsets = np.array([_to_float(r.get("P_AXIS_[-]", "0.25"), 0.25) for r in blade_rows], dtype=float)
    circangles = np.array([_to_float(r.get("CIRCANGLE_[deg]", "0.0"), 0.0) for r in blade_rows], dtype=float)
    polar_files = [str(r.get("POLAR_FILE", "")).strip() for r in blade_rows]

    hmin = float(np.min(heights))
    hspan = float(np.max(heights) - hmin)
    if hspan <= 0.0:
        blade_norm_pos = np.zeros_like(heights)
    else:
        blade_norm_pos = (heights - hmin) / hspan

    num_blades = int(trb["num_blades"]) if trb["num_blades"] else int(bld["num_blades"])
    num_blades = max(1, num_blades)

    components: List[Component] = []
    qblade_node_ids: List[str] = []

    base_airfoil_id = _infer_airfoil_id_from_polar(polar_files[0] if polar_files else "", default_airfoil_id)
    base_thickness = _infer_thickness_from_airfoil_id(base_airfoil_id, default_thickness=0.12)

    for iblade in range(num_blades):
        azimuth_deg = (360.0 / num_blades) * iblade + float(np.mean(circangles))
        comp_airfoil_ids = [
            _infer_airfoil_id_from_polar(pf, default_airfoil_id) if pf else base_airfoil_id
            for pf in polar_files
        ]
        comp_thickness = np.array(
            [_infer_thickness_from_airfoil_id(afid, default_thickness=base_thickness) for afid in comp_airfoil_ids],
            dtype=float,
        )

        shape_xyz = np.column_stack([np.zeros_like(heights), radii, heights])
        n_nodes = shape_xyz.shape[0]
        component = Component(
            id=f"BLD_{iblade + 1}",
            translation=np.zeros(3, dtype=float),
            rotation=np.array([0.0, 0.0, azimuth_deg], dtype=float),
            pitch=np.array([0.0], dtype=float),
            shape_xyz=shape_xyz,
            shape_xyz_global=np.zeros((n_nodes, 3), dtype=float),
            chord=chords.copy(),
            twist=twists.copy(),
            thickness=comp_thickness,
            offset=offsets.copy(),
            airfoil_ids=comp_airfoil_ids,
            chord_vector=np.zeros((n_nodes, 3), dtype=float),
            normal_vector=np.zeros((n_nodes, 3), dtype=float),
        )
        components.append(component)
        for p in blade_norm_pos:
            qblade_node_ids.append(f"BLD_{iblade + 1}_{float(p):.6f}")

    if include_struts and bld["strut_rows"]:
        order = np.argsort(heights)
        h_sorted = heights[order]
        r_sorted = radii[order]

        for istruct, srow in enumerate(bld["strut_rows"]):
            chord_hub = _to_float(srow.get("CHORDHUB_STR", "0.0"), 0.0)
            chord_bld = _to_float(srow.get("CHORDBLD_STR", "0.0"), 0.0)
            hgt_hub = _to_float(srow.get("HGTHUB_STR", "0.0"), 0.0)
            hgt_bld = _to_float(srow.get("HGTBLD_STR", "0.0"), 0.0)
            dst_hub = _to_float(srow.get("DSTHUB_STR", "0.0"), 0.0)
            angle_str = _to_float(srow.get("ANGLE_STR", "0.0"), 0.0)
            paxis_hub = _to_float(srow.get("PAXISHUB_STR", "0.25"), 0.25)
            paxis_bld = _to_float(srow.get("PAXISBLD_STR", "0.25"), 0.25)
            polar_str = str(srow.get("POLAR_STR", "")).strip()

            radius_bld = float(np.interp(hgt_bld, h_sorted, r_sorted))
            strut_airfoil_id = _infer_airfoil_id_from_polar(polar_str, default_airfoil_id)
            strut_thickness = _infer_thickness_from_airfoil_id(strut_airfoil_id, default_thickness=0.10)

            for iblade in range(num_blades):
                azimuth_deg = (360.0 / num_blades) * iblade
                shape_xyz = np.array(
                    [
                        [0.0, dst_hub, hgt_hub],
                        [0.0, radius_bld, hgt_bld],
                    ],
                    dtype=float,
                )
                component = Component(
                    id=f"STR_{istruct + 1}_{iblade + 1}",
                    translation=np.zeros(3, dtype=float),
                    rotation=np.array([0.0, 0.0, azimuth_deg], dtype=float),
                    pitch=np.array([0.0], dtype=float),
                    shape_xyz=shape_xyz,
                    shape_xyz_global=np.zeros((2, 3), dtype=float),
                    chord=np.array([chord_hub, chord_bld], dtype=float),
                    twist=np.array([angle_str, angle_str], dtype=float),
                    thickness=np.array([strut_thickness, strut_thickness], dtype=float),
                    offset=np.array([paxis_hub, paxis_bld], dtype=float),
                    airfoil_ids=[strut_airfoil_id, strut_airfoil_id],
                    chord_vector=np.zeros((2, 3), dtype=float),
                    normal_vector=np.zeros((2, 3), dtype=float),
                )
                components.append(component)
                qblade_node_ids.append(f"STR_{istruct + 1}_{iblade + 1}_0.000000")
                qblade_node_ids.append(f"STR_{istruct + 1}_{iblade + 1}_1.000000")

    if include_tower:
        tower_height = float(trb.get("tower_height", 0.0) or 0.0)
        tower_top_radius = float(trb.get("tower_top_radius", 0.0) or 0.0)
        tower_bottom_radius = float(trb.get("tower_bottom_radius", 0.0) or 0.0)
        if tower_height > 0.0 and (tower_top_radius > 0.0 or tower_bottom_radius > 0.0):
            n_tower = max(6, len(heights))
            # Align tower top with the blade hub reference height in the imported geometry.
            hub_z_ref = float(np.mean(heights)) if heights.size else 0.0
            t = np.linspace(0.0, 1.0, n_tower, dtype=float)
            z = hub_z_ref - tower_height + tower_height * t
            radii_tower = tower_bottom_radius + (tower_top_radius - tower_bottom_radius) * t
            chord_tower = np.maximum(2.0 * radii_tower, 1.0e-3)

            tower_component = Component(
                id="TWR_1",
                translation=np.zeros(3, dtype=float),
                rotation=np.zeros(3, dtype=float),
                pitch=np.array([0.0], dtype=float),
                shape_xyz=np.column_stack([np.zeros_like(z), np.zeros_like(z), z]),
                shape_xyz_global=np.zeros((n_tower, 3), dtype=float),
                chord=chord_tower,
                twist=np.zeros(n_tower, dtype=float),
                thickness=np.ones(n_tower, dtype=float),
                offset=np.full(n_tower, 0.5, dtype=float),
                airfoil_ids=[tower_airfoil_id] * n_tower,
                chord_vector=np.zeros((n_tower, 3), dtype=float),
                normal_vector=np.zeros((n_tower, 3), dtype=float),
            )
            components.append(tower_component)
            for p in t:
                qblade_node_ids.append(f"TWR_1_{float(p):.6f}")

    hor_rad = np.deg2rad(float(sim["hor_angle_deg"]))
    vert_rad = np.deg2rad(float(sim["vert_angle_deg"]))
    inflow_vec = np.array(
        [
            np.cos(vert_rad) * np.cos(hor_rad),
            np.cos(vert_rad) * np.sin(hor_rad),
            np.sin(vert_rad),
        ],
        dtype=float,
    )
    if np.linalg.norm(inflow_vec) <= 1.0e-12:
        inflow_vec = np.array([1.0, 0.0, 0.0], dtype=float)

    timestep = float(sim["timestep"])
    numsteps = int(sim["numtimesteps"])
    output_time = np.arange(0.0, (numsteps + 1) * timestep, timestep, dtype=float)

    viv_params = VIV_Params(
        fluid_density=float(sim["fluid_density"]),
        fluid_dynamicviscosity=float(sim["fluid_dynamic_viscosity"]),
        rotation_axis=np.array([0.0, 0.0, 1.0], dtype=float),
        rotation_axis_offset=np.asarray(sim["global_position"], dtype=float),
        inflow_vec=inflow_vec,
        azimuths=np.arange(0.0, 360.0, 5.0, dtype=float),
        inflow_speeds=np.array([max(0.0, float(sim["mean_inflow"]))], dtype=float),
        output_time=output_time,
        n_harmonic=2,
        amplitude_coeff_cutoff=0.002,
        n_freq_depth=10,
        output_azimuth_vinf=(float(sim["initial_azimuth_deg"]), max(0.0, float(sim["mean_inflow"]))),
    )

    return components, viv_params, qblade_node_ids


def write_components_to_csv(dir_path: str, components: List[Component]) -> List[str]:
    """
    Write VorLap components to per-component CSV files compatible with `load_components_from_csv`.

    Args:
        dir_path: Output directory for component CSV files.
        components: Components to write.

    Returns:
        List of absolute file paths written.
    """
    if not components:
        raise ValueError("No components were provided for export.")

    os.makedirs(dir_path, exist_ok=True)
    written_files: List[str] = []

    for icomp, comp in enumerate(components):
        comp_id = str(comp.id) if getattr(comp, "id", None) else f"component_{icomp+1}"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", comp_id).strip("_") or f"component_{icomp+1}"
        out_path = os.path.abspath(os.path.join(dir_path, f"{safe_id}.csv"))

        xyz = np.asarray(comp.shape_xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"Component '{comp_id}' has invalid shape_xyz with shape {xyz.shape}.")

        npts = xyz.shape[0]
        chord = np.asarray(comp.chord, dtype=float).reshape(-1)
        twist = np.asarray(comp.twist, dtype=float).reshape(-1)
        thickness = np.asarray(comp.thickness, dtype=float).reshape(-1)
        offset = np.asarray(comp.offset, dtype=float).reshape(-1)
        if not (len(chord) == len(twist) == len(thickness) == len(offset) == npts):
            raise ValueError(
                f"Component '{comp_id}' has inconsistent vector lengths: "
                f"xyz={npts}, chord={len(chord)}, twist={len(twist)}, thickness={len(thickness)}, offset={len(offset)}."
            )

        airfoil_ids = list(comp.airfoil_ids) if getattr(comp, "airfoil_ids", None) is not None else []
        if len(airfoil_ids) < npts:
            fill_id = airfoil_ids[-1] if airfoil_ids else "default"
            airfoil_ids = airfoil_ids + [fill_id] * (npts - len(airfoil_ids))
        else:
            airfoil_ids = airfoil_ids[:npts]

        translation = np.asarray(comp.translation, dtype=float).reshape(-1)
        rotation = np.asarray(comp.rotation, dtype=float).reshape(-1)
        pitch = np.asarray(comp.pitch, dtype=float).reshape(-1)
        pitch_value = float(pitch[0]) if pitch.size else 0.0
        if translation.size != 3 or rotation.size != 3:
            raise ValueError(f"Component '{comp_id}' has invalid translation/rotation dimensions.")

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z", "pitch"])
            writer.writerow(
                [
                    comp_id,
                    f"{translation[0]:.9g}",
                    f"{translation[1]:.9g}",
                    f"{translation[2]:.9g}",
                    f"{rotation[0]:.9g}",
                    f"{rotation[1]:.9g}",
                    f"{rotation[2]:.9g}",
                    f"{pitch_value:.9g}",
                ]
            )
            writer.writerow(["x", "y", "z", "chord", "twist", "thickness", "offset", "airfoil_id"])
            for i in range(npts):
                writer.writerow(
                    [
                        f"{xyz[i, 0]:.9g}",
                        f"{xyz[i, 1]:.9g}",
                        f"{xyz[i, 2]:.9g}",
                        f"{chord[i]:.9g}",
                        f"{twist[i]:.9g}",
                        f"{thickness[i]:.9g}",
                        f"{offset[i]:.9g}",
                        str(airfoil_ids[i]),
                    ]
                )

        written_files.append(out_path)

    return written_files


def write_qblade_loading_file(
    filename: str,
    time: np.ndarray,
    global_force_vector_nodes: np.ndarray,
    node_ids: List[str],
    local: bool = False,
    node_torque_vector_nodes: Optional[np.ndarray] = None,
) -> None:
    """
    Write a QBlade external loading file compatible with `LOADINGFILE`.

    File format per node block:
        NODE_ID [LOCAL]
        t Fx Fy Fz Mx My Mz
        ...

    QBlade linearly interpolates between time rows during simulation.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    force = np.asarray(global_force_vector_nodes, dtype=float)
    if force.ndim != 3 or force.shape[1] != 3:
        raise ValueError("global_force_vector_nodes must have shape [ntime, 3, nnodes].")
    if force.shape[0] != time.shape[0]:
        raise ValueError("time length must match global_force_vector_nodes time dimension.")
    if len(node_ids) != force.shape[2]:
        raise ValueError("node_ids length must match global_force_vector_nodes node dimension.")

    if node_torque_vector_nodes is None:
        torque = np.zeros_like(force)
    else:
        torque = np.asarray(node_torque_vector_nodes, dtype=float)
        if torque.shape != force.shape:
            raise ValueError("node_torque_vector_nodes must match shape [ntime, 3, nnodes].")

    suffix = " LOCAL" if local else ""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("// VorLap-generated QBlade external loading file\n")
        f.write("// Format per block: ID [LOCAL], then rows of time Fx Fy Fz Mx My Mz\n\n")
        for inode, node_id in enumerate(node_ids):
            f.write(f"{node_id}{suffix}\n")
            for it, t in enumerate(time):
                fx, fy, fz = force[it, 0, inode], force[it, 1, inode], force[it, 2, inode]
                mx, my, mz = torque[it, 0, inode], torque[it, 1, inode], torque[it, 2, inode]
                f.write(f"{t:.9g} {fx:.9g} {fy:.9g} {fz:.9g} {mx:.9g} {my:.9g} {mz:.9g}\n")
            f.write("\n")


def load_components_from_csv(dir_path: str) -> List[Component]:
    """
    Loads all component geometry and metadata from CSV files in the given directory.

    Args:
        dir_path: Path to a directory containing CSV files. Each file must follow a two-header format.

    Returns:
        List of parsed Component objects containing all geometry and configuration data.

    Expected CSV Format:
        1. First data row contains: `id`, `translation_x`, `translation_y`, `translation_z`, `rotation_x`, `rotation_y`, `rotation_z`
        2. Second header row: column names for vectors — must include `x`, `y`, `z`, `chord`, `twist`, `thickness`, and optional `airfoil_id`
        3. Remaining rows: vector data for each blade segment or shape point

    Notes:
        - If `airfoil_id` is missing, `"default"` will be used for all segments in that component
        - All transformations are centered at the origin and adjusted by top-level translation/rotation
        - All components are assumed to have the span oriented in the z-direction
    """
    import glob

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Components directory does not exist: {dir_path}")

    files = glob.glob(os.path.join(dir_path, "*.csv"))
    files.sort()  # Sort files for consistent ordering
    if not files:
        raise FileNotFoundError(f"No component CSV files were found in {dir_path}")

    components = []
    
    for file in files:
        # Read the raw CSV data
        with open(file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 4:
            raise ValueError(f"Component CSV is too short: {file}")

        # Extract top-level metadata (row 2)
        metadata = lines[1].strip().split(',')
        if len(metadata) < 8:
            raise ValueError(f"Component metadata row must have at least 8 values in {file}")
        id_str = metadata[0]
        tx = float(metadata[1])
        ty = float(metadata[2])
        tz = float(metadata[3])
        rx = float(metadata[4])
        ry = float(metadata[5])
        rz = float(metadata[6])
        pitch = float(metadata[7])
        
        # Get column names (row 3)
        colnames = [col.strip() for col in lines[2].strip().split(',')]
        
        # Read the data rows (from row 4 onwards)
        data = []
        for line in lines[3:]:
            if line.strip():  # Skip empty lines
                data.append(line.strip().split(','))
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=colnames)
        required_cols = {"x", "y", "z", "chord", "twist", "thickness", "offset"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {file}: {sorted(missing)}")
        
        # Extract vectors
        xyz = np.column_stack([
            df['x'].astype(float).values,
            df['y'].astype(float).values,
            df['z'].astype(float).values
        ])
        
        chord = df['chord'].astype(float).values
        twist = df['twist'].astype(float).values
        thickness = df['thickness'].astype(float).values
        offset = df['offset'].astype(float).values
        
        if xyz.size == 0:
            raise ValueError(f"Component {id_str} in {file} has no geometry rows")

        # Handle optional airfoil_id column
        if 'airfoil_id' in df.columns:
            airfoil_ids = df['airfoil_id'].astype(str).values.tolist()
        else:
            airfoil_ids = ['default'] * len(df)
        
        # Create placeholders for vectors that will be filled later
        chord_vec = np.zeros((xyz.shape[0], 3))
        norm_vec = np.zeros((xyz.shape[0], 3))
        xyz_global = np.zeros_like(xyz)
        
        # Create and add the component
        component = Component(
            id=id_str,
            translation=np.array([tx, ty, tz]),
            rotation=np.array([rx, ry, rz]),
            pitch=np.array([pitch]),
            shape_xyz=xyz,
            shape_xyz_global=xyz_global,
            chord=chord,
            twist=twist,
            thickness=thickness,
            offset=offset,
            airfoil_ids=airfoil_ids,
            chord_vector=chord_vec,
            normal_vector=norm_vec
        )
        
        components.append(component)
    
    return components


def load_airfoil_fft(path: str) -> AirfoilFFT:
    """
    Loads a processed airfoil unsteady FFT dataset from an HDF5 file.

    Args:
        path: Path to the HDF5 file containing the airfoil FFT data.

    Returns:
        AirfoilFFT object containing the loaded data.

    Expected HDF5 File Format:
        The file must contain the following datasets:
        - `Airfoilname` :: String — Name of the airfoil (e.g., "NACA0012")
        - `Re` :: Vector{Float64} — Reynolds number values (assumed constant across all entries)
        - `Thickness` :: Vector{Float64} — Thickness ratio(s) used
        - `AOA` :: Vector{Float64} — Angle of attack values in degrees
        - `CL_ST`, `CD_ST`, `CM_ST`, `CF_ST` :: 3D Arrays [Re x AOA x freq] — Strouhal numbers for each force/moment
        - `CL_Amp`, `CD_Amp`, `CM_Amp`, `CF_Amp` :: 3D Arrays [Re x AOA x freq] — FFT amplitudes for lift, drag, moment, and combined force
        - `CL_Pha`, `CD_Pha`, `CM_Pha`, `CF_Pha` :: 3D Arrays [Re x AOA x freq] — FFT phases in radians for each quantity

    Assumptions:
        - All arrays must share dimensions [Re, AOA, NFreq], where the frequency dimension is sorted by the amplitude
        - Phase data is in radians.
        - Struhaul data represents unsteady aerodynamics due to vortex shedding.
        - No ragged or missing data is allowed.
    """
    with h5py.File(path, 'r') as h5:
        name = h5['Airfoilname'][()] if 'Airfoilname' in h5 else os.path.basename(path)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
            
        Re = np.asarray(h5['Re'][()], dtype=float).reshape(-1)
        Thickness = np.asarray(h5['Thickness'][()])
        AOA = np.asarray(h5['AOA'][()], dtype=float).reshape(-1)
        
        CL_ST = h5['CL_ST'][()]
        CD_ST = h5['CD_ST'][()]
        CM_ST = h5['CM_ST'][()]
        CF_ST = h5['CF_ST'][()]
        
        CL_Amp = h5['CL_Amp'][()]
        CD_Amp = h5['CD_Amp'][()]
        CM_Amp = h5['CM_Amp'][()]
        CF_Amp = h5['CF_Amp'][()]
        
        CL_Pha = h5['CL_Pha'][()]
        CD_Pha = h5['CD_Pha'][()]
        CM_Pha = h5['CM_Pha'][()]
        CF_Pha = h5['CF_Pha'][()]
        
        expected_prefix = (len(Re), len(AOA))

        def orient_dimensions(arr: np.ndarray, arr_name: str) -> np.ndarray:
            """Orient tensor dimensions to [Re, AOA, freq]."""
            arr = np.asarray(arr)
            if arr.ndim != 3:
                raise ValueError(f"{arr_name} must be 3D, got shape {arr.shape}")
            if arr.shape[:2] == expected_prefix:
                return arr

            for perm in (
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
            ):
                candidate = np.transpose(arr, perm)
                if candidate.shape[:2] == expected_prefix:
                    warnings.warn(
                        f"Transposing {arr_name} from shape {arr.shape} to {candidate.shape}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return candidate

            raise ValueError(
                f"Could not orient {arr_name} to [Re, AOA, freq]. "
                f"Got shape {arr.shape}, expected prefix {expected_prefix}."
            )

        arrays = {
            "CL_ST": orient_dimensions(CL_ST, "CL_ST"),
            "CD_ST": orient_dimensions(CD_ST, "CD_ST"),
            "CM_ST": orient_dimensions(CM_ST, "CM_ST"),
            "CF_ST": orient_dimensions(CF_ST, "CF_ST"),
            "CL_Amp": orient_dimensions(CL_Amp, "CL_Amp"),
            "CD_Amp": orient_dimensions(CD_Amp, "CD_Amp"),
            "CM_Amp": orient_dimensions(CM_Amp, "CM_Amp"),
            "CF_Amp": orient_dimensions(CF_Amp, "CF_Amp"),
            "CL_Pha": orient_dimensions(CL_Pha, "CL_Pha"),
            "CD_Pha": orient_dimensions(CD_Pha, "CD_Pha"),
            "CM_Pha": orient_dimensions(CM_Pha, "CM_Pha"),
            "CF_Pha": orient_dimensions(CF_Pha, "CF_Pha"),
        }

        common_depth = min(arr.shape[2] for arr in arrays.values())
        for arr_name, arr in list(arrays.items()):
            if arr.shape[2] != common_depth:
                warnings.warn(
                    f"Trimming {arr_name} frequency depth from {arr.shape[2]} to {common_depth} for consistency.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            arrays[arr_name] = arr[:, :, :common_depth]

        if np.isscalar(Thickness) or np.ndim(Thickness) == 0:
            thickness_value = float(np.asarray(Thickness, dtype=float))
        else:
            thickness_arr = np.asarray(Thickness, dtype=float).reshape(-1)
            thickness_value = float(thickness_arr[0]) if thickness_arr.size else 0.0
        
        return AirfoilFFT(
            name=name,
            Re=Re,
            AOA=AOA,
            Thickness=thickness_value,
            CL_ST=arrays["CL_ST"],
            CD_ST=arrays["CD_ST"],
            CM_ST=arrays["CM_ST"],
            CF_ST=arrays["CF_ST"],
            CL_Amp=arrays["CL_Amp"],
            CD_Amp=arrays["CD_Amp"],
            CM_Amp=arrays["CM_Amp"],
            CF_Amp=arrays["CF_Amp"],
            CL_Pha=arrays["CL_Pha"],
            CD_Pha=arrays["CD_Pha"],
            CM_Pha=arrays["CM_Pha"],
            CF_Pha=arrays["CF_Pha"]
        )


def load_airfoil_coords(afpath: str = "") -> np.ndarray:
    """
    Loads an airfoil shape from a 2-column text file (x, z), normalized to unit chord length.
    If no file is specified, or if loading fails, returns a built-in 200-point Clark Y airfoil shape.

    Args:
        afpath: Optional path to a text file with two columns: x and z coordinates.

    Returns:
        xy: Nx2 matrix of normalized (x, y) coordinates representing the airfoil surface.

    Notes:
        - If loading from file, x-coordinates are normalized to span [0, 1].
        - The default fallback airfoil is a symmetric approximation of the Clark Y shape.
        - This airfoil is primarily used for visualization, not aerodynamic calculations.
    """
    if afpath and os.path.isfile(afpath):
        try:
            xy = np.loadtxt(afpath, delimiter=',')
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError(f"Expected Nx2 coordinates, got {xy.shape}")
            xy[:, 0] -= np.min(xy[:, 0])
            x_max = np.max(xy[:, 0])
            y_span = np.max(xy[:, 1]) - np.min(xy[:, 1])
            if x_max <= 0.0 or y_span <= 0.0:
                raise ValueError("Airfoil coordinates have zero span in x or y.")
            xy[:, 0] /= x_max
            xy[:, 1] /= y_span
            return xy
        except Exception as e:
            warnings.warn(f"Could not load airfoil file: {e}. Falling back to default Clark Y profile for plotting.")
    else:
        warnings.warn("Could not load airfoil file used for plotting. Falling back to default Clark Y profile for plotting.")
    
    # Fallback Clark Y airfoil coordinates (from airfoiltools.com)
    xy = np.array([
        [1.0000000, 0.0],
        [0.9900000, 0.0029690],
        [0.9800000, 0.0053335],
        [0.9700000, 0.0076868],
        [0.9600000, 0.0100232],
        [0.9400000, 0.0146239],
        [0.9200000, 0.0191156],
        [0.9000000, 0.0235025],
        [0.8800000, 0.0277891],
        [0.8600000, 0.0319740],
        [0.8400000, 0.0360536],
        [0.8200000, 0.0400245],
        [0.8000000, 0.0438836],
        [0.7800000, 0.0476281],
        [0.7600000, 0.0512565],
        [0.7400000, 0.0547675],
        [0.7200000, 0.0581599],
        [0.7000000, 0.0614329],
        [0.6800000, 0.0645843],
        [0.6600000, 0.0676046],
        [0.6400000, 0.0704822],
        [0.6200000, 0.0732055],
        [0.6000000, 0.0757633],
        [0.5800000, 0.0781451],
        [0.5600000, 0.0803480],
        [0.5400000, 0.0823712],
        [0.5200000, 0.0842145],
        [0.5000000, 0.0858772],
        [0.4800000, 0.0873572],
        [0.4600000, 0.0886427],
        [0.4400000, 0.0897175],
        [0.4200000, 0.0905657],
        [0.4000000, 0.0911712],
        [0.3800000, 0.0915212],
        [0.3600000, 0.0916266],
        [0.3400000, 0.0915079],
        [0.3200000, 0.0911857],
        [0.3000000, 0.0906804],
        [0.2800000, 0.0900016],
        [0.2600000, 0.0890840],
        [0.2400000, 0.0878308],
        [0.2200000, 0.0861433],
        [0.2000000, 0.0839202],
        [0.1800000, 0.0810687],
        [0.1600000, 0.0775707],
        [0.1400000, 0.0734360],
        [0.1200000, 0.0686204],
        [0.1000000, 0.0629981],
        [0.0800000, 0.0564308],
        [0.0600000, 0.0487571],
        [0.0500000, 0.0442753],
        [0.0400000, 0.0391283],
        [0.0300000, 0.0330215],
        [0.0200000, 0.0253735],
        [0.0120000, 0.0178581],
        [0.0080000, 0.0137350],
        [0.0040000, 0.0089238],
        [0.0020000, 0.0058025],
        [0.0010000, 0.0037271],
        [0.0005000, 0.0023390],
        [0.0000000, 0.0000000],
        [0.0005000, -0.0046700],
        [0.0010000, -0.0059418],
        [0.0020000, -0.0078113],
        [0.0040000, -0.0105126],
        [0.0080000, -0.0142862],
        [0.0120000, -0.0169733],
        [0.0200000, -0.0202723],
        [0.0300000, -0.0226056],
        [0.0400000, -0.0245211],
        [0.0500000, -0.0260452],
        [0.0600000, -0.0271277],
        [0.0800000, -0.0284595],
        [0.1000000, -0.0293786],
        [0.1200000, -0.0299633],
        [0.1400000, -0.0302404],
        [0.1600000, -0.0302546],
        [0.1800000, -0.0300490],
        [0.2000000, -0.0296656],
        [0.2200000, -0.0291445],
        [0.2400000, -0.0285181],
        [0.2600000, -0.0278164],
        [0.2800000, -0.0270696],
        [0.3000000, -0.0263079],
        [0.3200000, -0.0255565],
        [0.3400000, -0.0248176],
        [0.3600000, -0.0240870],
        [0.3800000, -0.0233606],
        [0.4000000, -0.0226341],
        [0.4200000, -0.0219042],
        [0.4400000, -0.0211708],
        [0.4600000, -0.0204353],
        [0.4800000, -0.0196986],
        [0.5000000, -0.0189619],
        [0.5200000, -0.0182262],
        [0.5400000, -0.0174914],
        [0.5600000, -0.0167572],
        [0.5800000, -0.0160232],
        [0.6000000, -0.0152893],
        [0.6200000, -0.0145551],
        [0.6400000, -0.0138207],
        [0.6600000, -0.0130862],
        [0.6800000, -0.0123515],
        [0.7000000, -0.0116169],
        [0.7200000, -0.0108823],
        [0.7400000, -0.0101478],
        [0.7600000, -0.0094133],
        [0.7800000, -0.0086788],
        [0.8000000, -0.0079443],
        [0.8200000, -0.0072098],
        [0.8400000, -0.0064753],
        [0.8600000, -0.0057408],
        [0.8800000, -0.0050063],
        [0.9000000, -0.0042718],
        [0.9200000, -0.0035373],
        [0.9400000, -0.0028028],
        [0.9600000, -0.0020683],
        [0.9700000, -0.0017011],
        [0.9800000, -0.0013339],
        [0.9900000, -0.0009666],
        [1.0, 0]
    ])
    
    xy[:, 0] -= np.min(xy[:, 0])
    xy[:, 0] /= np.max(xy[:, 0])
    xy[:, 1] /= (np.max(xy[:, 1]) - np.min(xy[:, 1]))
    
    return xy


def load_inflow_time_series(path: str) -> InflowTimeSeries:
    """
    Load a time-varying inflow profile from CSV.

    Required columns:
        - `time`
        - `inflow_speed`

    Direction column options:
        1) `inflow_direction_deg` (degrees CCW from +X in the global XY plane), or
        2) `inflow_dir_x`, `inflow_dir_y`, and optional `inflow_dir_z`.

    Args:
        path: CSV file path.

    Returns:
        InflowTimeSeries with validated and normalized direction vectors.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Inflow profile CSV does not exist: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Inflow profile CSV is empty: {path}")

    column_map = {str(col).strip().lower(): col for col in df.columns}

    def _find_column(*candidates: str):
        for name in candidates:
            key = name.strip().lower()
            if key in column_map:
                return column_map[key]
        return None

    time_col = _find_column("time")
    speed_col = _find_column("inflow_speed", "speed", "vinf")
    if time_col is None or speed_col is None:
        raise ValueError(
            "Inflow profile CSV must include `time` and `inflow_speed` columns "
            "(aliases: `speed`, `vinf`)."
        )

    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    inflow_speeds = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=float)

    dir_deg_col = _find_column("inflow_direction_deg", "direction_deg", "inflow_direction")
    dir_x_col = _find_column("inflow_dir_x", "direction_x", "dir_x")
    dir_y_col = _find_column("inflow_dir_y", "direction_y", "dir_y")
    dir_z_col = _find_column("inflow_dir_z", "direction_z", "dir_z")

    if dir_deg_col is not None:
        direction_deg = pd.to_numeric(df[dir_deg_col], errors="coerce").to_numpy(dtype=float)
        direction_rad = np.deg2rad(direction_deg)
        inflow_directions = np.column_stack(
            [
                np.cos(direction_rad),
                np.sin(direction_rad),
                np.zeros_like(direction_rad),
            ]
        )
    elif dir_x_col is not None and dir_y_col is not None:
        dir_x = pd.to_numeric(df[dir_x_col], errors="coerce").to_numpy(dtype=float)
        dir_y = pd.to_numeric(df[dir_y_col], errors="coerce").to_numpy(dtype=float)
        if dir_z_col is None:
            dir_z = np.zeros_like(dir_x)
        else:
            dir_z = pd.to_numeric(df[dir_z_col], errors="coerce").to_numpy(dtype=float)
        inflow_directions = np.column_stack([dir_x, dir_y, dir_z])
    else:
        raise ValueError(
            "Inflow profile CSV must include either `inflow_direction_deg` "
            "or direction vectors (`inflow_dir_x`, `inflow_dir_y`, optional `inflow_dir_z`)."
        )

    order = np.argsort(time)
    time = time[order]
    inflow_speeds = inflow_speeds[order]
    inflow_directions = inflow_directions[order, :]

    return InflowTimeSeries(time=time, inflow_speeds=inflow_speeds, inflow_directions=inflow_directions)


def write_force_time_series(filename: str, output_time: np.ndarray, global_force_vector_nodes: np.ndarray) -> None:
    """
    Writes force time series data to a CSV file.

    Args:
        filename: Path to the output CSV file.
        output_time: Vector of time points.
        global_force_vector_nodes: Array of force vectors for each node at each time point.

    Returns:
        None
    """
    output_time = np.asarray(output_time, dtype=float)
    global_force_vector_nodes = np.asarray(global_force_vector_nodes, dtype=float)
    if global_force_vector_nodes.ndim != 3 or global_force_vector_nodes.shape[1] != 3:
        raise ValueError("global_force_vector_nodes must have shape [ntime, 3, nnodes].")

    ntime, _, nnodes = global_force_vector_nodes.shape
    if output_time.shape[0] != ntime:
        raise ValueError("output_time length must match global_force_vector_nodes time dimension.")
    
    with open(filename, "w") as f:
        # Write header
        header = ["time"]
        for n in range(nnodes):
            header.extend([f"node{n+1}x", f"node{n+1}y", f"node{n+1}z"])
        f.write(", ".join(header) + "\n")
        
        # Write each time row
        for t in range(ntime):
            row = [str(output_time[t])]
            for n in range(nnodes):
                fx = global_force_vector_nodes[t, 0, n]
                fy = global_force_vector_nodes[t, 1, n]
                fz = global_force_vector_nodes[t, 2, n]
                row.extend([str(fx), str(fy), str(fz)])
            f.write(", ".join(row) + "\n")
