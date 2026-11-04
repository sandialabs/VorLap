"""Convert QBlade model definitions into VorLap components and parameters."""

from __future__ import annotations

import os
import math
from pathlib import Path
import re
from typing import List, Tuple

import numpy as np

from .geometry_frames import (
    rotate_vectors_about_axes,
    rotationMatrix,
    segment_span_vectors,
)
from .qblade_parser import (
    _to_float,
    load_qblade_blade_definition,
    load_qblade_simulation_definition,
    load_qblade_turbine_definition,
)
from .qblade_files import read_qblade_keyword
from .structs import Component, VIV_Params

_FRAME_EPS = 1.0e-12


def _hawt_blade_frame_vectors(
    shape_xyz: np.ndarray,
    twist_deg: np.ndarray,
    azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return QBlade-aligned HAWT chord/normal vectors in global axes.

    At zero azimuth and twist QBlade's blade-result frame has ``+X_b`` in
    the rotor-normal direction and ``+Y_b`` along the chord toward the
    trailing edge.  VorLap stores its aerodynamic basis in
    ``(chord, normal, span)`` order and requires that ordering to be
    right-handed.  The corresponding zero-angle VorLap basis is therefore
    ``chord=+Y_b`` and ``normal=-X_b``.  The sign conversion is intentional:
    a global/QBlade ``+X_b`` action is a negative VorLap-normal action.

    The QBlade blade table may include sweep and prebend, so the reference
    chord is projected into each local plane before section twist and rotor
    azimuth are applied.
    """
    xyz = np.asarray(shape_xyz, dtype=float)
    twist = np.asarray(twist_deg, dtype=float).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or twist.shape[0] != xyz.shape[0]:
        raise ValueError(
            "HAWT frame inputs must contain aligned [n,3] geometry and [n] twist arrays."
        )

    span_local = segment_span_vectors(xyz)
    chord_reference = np.tile(np.array([0.0, 1.0, 0.0], dtype=float), (len(xyz), 1))
    chord_local = (
        chord_reference
        - np.sum(chord_reference * span_local, axis=1)[:, None] * span_local
    )
    chord_norm = np.linalg.norm(chord_local, axis=1)
    if np.any(chord_norm <= _FRAME_EPS):
        raise ValueError(
            "A HAWT blade span is locally parallel to QBlade's chord reference axis."
        )
    chord_local /= chord_norm[:, None]
    normal_local = np.cross(span_local, chord_local)
    normal_local /= np.linalg.norm(normal_local, axis=1)[:, None]

    chord_local = rotate_vectors_about_axes(chord_local, span_local, twist)
    normal_local = rotate_vectors_about_axes(normal_local, span_local, twist)
    rotor_rotation = rotationMatrix(
        np.array([float(azimuth_deg), 0.0, 0.0], dtype=float)
    )
    chord_global = (rotor_rotation @ chord_local.T).T
    normal_global = (rotor_rotation @ normal_local.T).T
    return chord_global, normal_global


def _infer_airfoil_id_from_polar(
    polar_path: str, default_airfoil_id: str = "default"
) -> str:
    """
    Infer a VorLap airfoil identifier from a QBlade polar file path.

    Examples:
        `Polars/NACA_0018_RFoil_MultiRePolar.plr` -> `NACA0018`
        `Polars/flat_plate_MultiRePolar.plr` -> `default`
    """
    base = os.path.basename(str(polar_path))
    if not base:
        return default_airfoil_id

    match = re.search(r"(NACA[_-]?\d{4})", base, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", "").replace("-", "").upper()
    match = re.search(r"(FFA[_-]?W\d[_-]?\d{3})", base, flags=re.IGNORECASE)
    if match:
        return re.sub(r"[-]+", "_", match.group(1)).upper()
    match = re.search(r"(IEA10[_-]STATION[_-]\d+)", base, flags=re.IGNORECASE)
    if match:
        return re.sub(r"[-]+", "_", match.group(1)).upper()
    if "CYLINDER" in base.upper():
        return "CYLINDER"
    return default_airfoil_id


def _infer_thickness_from_airfoil_id(
    airfoil_id: str, default_thickness: float = 0.12
) -> float:
    """Infer a thickness ratio from an identifier such as `NACA0018`."""
    if str(airfoil_id).strip().upper() == "CYLINDER":
        return 1.0
    match = re.search(r"(\d{4})$", str(airfoil_id))
    if match:
        return max(0.01, min(1.0, float(match.group(1)[2:]) / 100.0))
    match = re.search(r"(\d{3})$", str(airfoil_id))
    if match:
        return max(0.01, min(1.0, float(match.group(1)) / 1000.0))
    return float(default_thickness)


def _thickness_from_polar(
    polar_path: str, *, blade_path: str, fallback_airfoil_id: str
) -> float:
    """Read the native QBlade polar thickness independently of spectral ID."""
    candidate = (Path(blade_path).resolve().parent / polar_path).resolve()
    if candidate.is_file():
        raw = read_qblade_keyword(candidate, "THICKNESS")
        if raw is not None:
            value = float(raw) / 100.0
            if math.isfinite(value) and 0.0 < value <= 1.5:
                return value
            raise ValueError(f"Invalid THICKNESS in QBlade polar {candidate}: {raw}")
    return _infer_thickness_from_airfoil_id(fallback_airfoil_id)


def convert_qblade_to_vorlap_inputs(
    sim_path: str,
    default_airfoil_id: str = "default",
    include_struts: bool = True,
    include_tower: bool = True,
    tower_airfoil_id: str = "cylinder",
) -> Tuple[List[Component], VIV_Params, List[str]]:
    """
    Convert a QBlade `.sim` setup into equivalent VorLap components and baseline parameters.

    Returns:
        components: VorLap components equivalent to QBlade blade/strut geometry.
        viv_params: Baseline parameters inferred from QBlade simulation/turbine files.
        qblade_node_ids: Per-node QBlade location IDs aligned with VorLap node ordering.
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

    turbine_type = int(trb.get("turbine_type", 1))
    rotor_type = str(bld.get("rotor_type", "")).upper()
    is_hawt = turbine_type == 0 or rotor_type == "HAWT"
    yaw_deg = float(sim.get("initial_yaw_deg", 0.0) or 0.0)
    shaft_tilt_deg = float(trb.get("shaft_tilt_deg", 0.0) or 0.0)
    fixed_hawt_rotation = rotationMatrix(
        np.array([0.0, shaft_tilt_deg, yaw_deg], dtype=float)
    )
    hawt_hub_translation = np.array(
        [0.0, 0.0, float(trb.get("tower_height", 0.0) or 0.0)], dtype=float
    ) + fixed_hawt_rotation @ np.array(
        [float(trb.get("overhang", 0.0) or 0.0), 0.0, 0.0], dtype=float
    )

    span_positions = np.array(
        [
            _to_float(row.get("POS_[m]", row.get("HEIGHT_[m]", "0.0")), 0.0)
            for row in blade_rows
        ],
        dtype=float,
    )
    chords = np.array(
        [_to_float(row.get("CHORD_[m]", "0.0"), 0.0) for row in blade_rows], dtype=float
    )
    radii = np.array(
        [_to_float(row.get("RADIUS_[m]", "0.0"), 0.0) for row in blade_rows],
        dtype=float,
    )
    twists = np.array(
        [_to_float(row.get("TWIST_[deg]", "0.0"), 0.0) for row in blade_rows],
        dtype=float,
    )
    offsets = np.array(
        [
            _to_float(row.get("TAXIS_[-]", row.get("P_AXIS_[-]", "0.25")), 0.25)
            for row in blade_rows
        ],
        dtype=float,
    )
    circangles = np.array(
        [_to_float(row.get("CIRCANGLE_[deg]", "0.0"), 0.0) for row in blade_rows],
        dtype=float,
    )
    in_plane_offsets = np.array(
        [_to_float(row.get("OFFSET_X_[m]", "0.0"), 0.0) for row in blade_rows],
        dtype=float,
    )
    out_of_plane_offsets = np.array(
        [_to_float(row.get("OFFSET_Y_[m]", "0.0"), 0.0) for row in blade_rows],
        dtype=float,
    )
    polar_files = [str(row.get("POLAR_FILE", "")).strip() for row in blade_rows]

    span_min = float(np.min(span_positions))
    span = float(np.max(span_positions) - span_min)
    if span <= 0.0:
        blade_norm_pos = np.zeros_like(span_positions)
    else:
        blade_norm_pos = (span_positions - span_min) / span

    num_blades = int(trb["num_blades"]) if trb["num_blades"] else int(bld["num_blades"])
    num_blades = max(1, num_blades)

    components: List[Component] = []
    qblade_node_ids: List[str] = []

    base_airfoil_id = _infer_airfoil_id_from_polar(
        polar_files[0] if polar_files else "", default_airfoil_id
    )
    base_thickness = _thickness_from_polar(
        polar_files[0],
        blade_path=str(trb["bladefile_path"]),
        fallback_airfoil_id=base_airfoil_id,
    )

    for iblade in range(num_blades):
        azimuth_deg = (
            float(sim.get("initial_azimuth_deg", 0.0) or 0.0)
            + (360.0 / num_blades) * iblade
            + float(np.mean(circangles))
        )
        comp_airfoil_ids = [
            (
                _infer_airfoil_id_from_polar(polar_file, default_airfoil_id)
                if polar_file
                else base_airfoil_id
            )
            for polar_file in polar_files
        ]
        comp_thickness = np.array(
            [
                (
                    _thickness_from_polar(
                        polar_file,
                        blade_path=str(trb["bladefile_path"]),
                        fallback_airfoil_id=afid,
                    )
                    if polar_file
                    else base_thickness
                )
                for polar_file, afid in zip(polar_files, comp_airfoil_ids)
            ],
            dtype=float,
        )

        if is_hawt:
            # QBlade HAWT station offsets are in-plane (global y at zero
            # azimuth) and out-of-plane (global x); the blade span is +z.
            # Rotor azimuth is a right-handed rotation about the hub x-axis.
            shape_xyz = np.column_stack(
                [out_of_plane_offsets, in_plane_offsets, span_positions]
            )
            component_rotation = np.array(
                [azimuth_deg, shaft_tilt_deg, yaw_deg], dtype=float
            )
            chord_vector, normal_vector = _hawt_blade_frame_vectors(
                shape_xyz, twists, azimuth_deg
            )
            chord_vector = (fixed_hawt_rotation @ chord_vector.T).T
            normal_vector = (fixed_hawt_rotation @ normal_vector.T).T
        else:
            shape_xyz = np.column_stack(
                [np.zeros_like(span_positions), radii, span_positions]
            )
            component_rotation = np.array([0.0, 0.0, azimuth_deg], dtype=float)
            chord_vector = np.zeros((shape_xyz.shape[0], 3), dtype=float)
            normal_vector = np.zeros((shape_xyz.shape[0], 3), dtype=float)
        n_nodes = shape_xyz.shape[0]
        component = Component(
            id=f"BLD_{iblade + 1}",
            translation=(
                hawt_hub_translation.copy() if is_hawt else np.zeros(3, dtype=float)
            ),
            rotation=component_rotation,
            pitch=np.array([0.0], dtype=float),
            shape_xyz=shape_xyz,
            shape_xyz_global=np.zeros((n_nodes, 3), dtype=float),
            chord=chords.copy(),
            twist=twists.copy(),
            thickness=comp_thickness,
            offset=offsets.copy(),
            airfoil_ids=comp_airfoil_ids,
            chord_vector=chord_vector,
            normal_vector=normal_vector,
        )
        components.append(component)
        for position in blade_norm_pos:
            qblade_node_ids.append(f"BLD_{iblade + 1}_{float(position):.6f}")

    if include_struts and not is_hawt and bld["strut_rows"]:
        order = np.argsort(span_positions)
        h_sorted = span_positions[order]
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
            strut_airfoil_id = _infer_airfoil_id_from_polar(
                polar_str, default_airfoil_id
            )
            strut_thickness = _infer_thickness_from_airfoil_id(
                strut_airfoil_id, default_thickness=0.10
            )

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
            n_tower = max(6, len(span_positions))
            hub_z_ref = float(np.mean(span_positions)) if span_positions.size else 0.0
            t = np.linspace(0.0, 1.0, n_tower, dtype=float)
            z = hub_z_ref - tower_height + tower_height * t
            radii_tower = (
                tower_bottom_radius + (tower_top_radius - tower_bottom_radius) * t
            )
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
            for position in t:
                qblade_node_ids.append(f"TWR_1_{float(position):.6f}")

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
        rotation_axis=(
            fixed_hawt_rotation @ np.array([1.0, 0.0, 0.0], dtype=float)
            if is_hawt
            else np.array([0.0, 0.0, 1.0], dtype=float)
        ),
        rotation_axis_offset=np.asarray(sim["global_position"], dtype=float)
        + (hawt_hub_translation if is_hawt else np.zeros(3, dtype=float)),
        inflow_vec=inflow_vec,
        azimuths=np.arange(0.0, 360.0, 5.0, dtype=float),
        inflow_speeds=np.array([max(0.0, float(sim["mean_inflow"]))], dtype=float),
        output_time=output_time,
        n_harmonic=2,
        amplitude_coeff_cutoff=0.002,
        n_freq_depth=10,
        output_azimuth_vinf=(
            float(sim["initial_azimuth_deg"]),
            max(0.0, float(sim["mean_inflow"])),
        ),
    )

    return components, viv_params, qblade_node_ids


__all__ = [
    "convert_qblade_to_vorlap_inputs",
]
