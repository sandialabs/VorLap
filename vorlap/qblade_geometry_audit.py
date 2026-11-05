"""QBlade/VorLap geometry-audit row and CSV helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from typing import Sequence

import numpy as np

from .geometry_frames import prepare_component_frames
from .qblade_component_selection import (
    build_node_permutations,
    build_selected_components,
    component_span_vectors_global,
)
from .qblade_external_table import build_external_library_table_spec, validate_unique_node_ids
from .qblade_nodes import action_target_from_node_id
from .structs import Component


GEOMETRY_AUDIT_COLUMNS = (
    "swap_index",
    "controller_index",
    "requested_node_id",
    "canonical_node_id",
    "controller_node_id",
    "action_id",
    "action_position",
    "force_swap_x",
    "force_swap_y",
    "force_swap_z",
    "velocity_swap_x",
    "velocity_swap_y",
    "velocity_swap_z",
    "qblade_reported_available",
    "qblade_reported_x_m",
    "qblade_reported_y_m",
    "qblade_reported_z_m",
    "vorlap_x_m",
    "vorlap_y_m",
    "vorlap_z_m",
    "delta_reported_m",
    "chord_x",
    "chord_y",
    "chord_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "span_x",
    "span_y",
    "span_z",
    "chord_norm",
    "normal_norm",
    "span_norm",
    "chord_dot_normal",
    "chord_dot_span",
    "normal_dot_span",
    "basis_determinant",
    "basis_status",
    "basis_warnings",
)


@dataclass(frozen=True)
class GeometryAuditRow:
    """Typed pre-run QBlade/VorLap geometry audit row with stable CSV keys."""

    swap_index: int
    controller_index: int
    requested_node_id: str
    canonical_node_id: str
    controller_node_id: str
    action_id: str
    action_position: float
    force_swap_x: int
    force_swap_y: int
    force_swap_z: int
    velocity_swap_x: int
    velocity_swap_y: int
    velocity_swap_z: int
    qblade_reported_available: bool
    qblade_reported_x_m: float | None
    qblade_reported_y_m: float | None
    qblade_reported_z_m: float | None
    vorlap_x_m: float
    vorlap_y_m: float
    vorlap_z_m: float
    delta_reported_m: float | None
    chord_x: float
    chord_y: float
    chord_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    span_x: float
    span_y: float
    span_z: float
    chord_norm: float
    normal_norm: float
    span_norm: float
    chord_dot_normal: float
    chord_dot_span: float
    normal_dot_span: float
    basis_determinant: float
    basis_status: str
    basis_warnings: str

    def as_dict(self) -> dict[str, object]:
        row = {key: getattr(self, key) for key in GEOMETRY_AUDIT_COLUMNS}
        for key in (
            "qblade_reported_x_m",
            "qblade_reported_y_m",
            "qblade_reported_z_m",
            "delta_reported_m",
        ):
            if row[key] is None:
                row[key] = ""
        return row


def _audit_basis_status(
    chord: np.ndarray,
    normal: np.ndarray,
    span: np.ndarray,
    *,
    tolerance: float,
) -> tuple[str, str, dict[str, float]]:
    chord_norm = float(np.linalg.norm(chord))
    normal_norm = float(np.linalg.norm(normal))
    span_norm = float(np.linalg.norm(span))
    chord_dot_normal = float(np.dot(chord, normal))
    chord_dot_span = float(np.dot(chord, span))
    normal_dot_span = float(np.dot(normal, span))
    determinant = float(np.dot(span, np.cross(chord, normal)))
    metrics = {
        "chord_norm": chord_norm,
        "normal_norm": normal_norm,
        "span_norm": span_norm,
        "chord_dot_normal": chord_dot_normal,
        "chord_dot_span": chord_dot_span,
        "normal_dot_span": normal_dot_span,
        "basis_determinant": determinant,
    }
    warnings: list[str] = []
    for name, value in metrics.items():
        if not math.isfinite(value):
            warnings.append(f"{name} nonfinite")
    if abs(chord_norm - 1.0) > tolerance:
        warnings.append("chord norm not unit")
    if abs(normal_norm - 1.0) > tolerance:
        warnings.append("normal norm not unit")
    if abs(span_norm - 1.0) > tolerance:
        warnings.append("span norm not unit")
    if abs(chord_dot_normal) > tolerance:
        warnings.append("chord/normal not orthogonal")
    if abs(chord_dot_span) > tolerance:
        warnings.append("chord/span not orthogonal")
    if abs(normal_dot_span) > tolerance:
        warnings.append("normal/span not orthogonal")
    if determinant <= 0.0 or abs(determinant - 1.0) > tolerance:
        warnings.append("basis determinant not right-handed unit")
    return ("ok" if not warnings else "error"), "; ".join(warnings), metrics


def build_qblade_geometry_audit_rows(
    components: Sequence[Component],
    requested_node_ids: Sequence[str],
    *,
    tolerance: float = 1.0e-8,
) -> list[dict[str, object]]:
    """Return a per-node geometry and swap-mapping audit table.

    The rows are ordered by the QBlade swap/request order, while also recording
    the controller index used internally after canonicalization and component
    resampling. QBlade-reported coordinates are intentionally blank here; those
    require a completed QBlade run with position output enabled.
    """
    requested = tuple(str(node_id) for node_id in requested_node_ids)
    validate_unique_node_ids(requested, "geometry audit")
    table_spec = build_external_library_table_spec(requested)
    selected_components, controller_node_ids, swap_node_ids = build_selected_components(components, requested)
    prepare_component_frames(selected_components)
    swap_to_controller_idx, _controller_to_swap_idx = build_node_permutations(swap_node_ids, controller_node_ids)

    prepared: dict[str, tuple[Component, int, np.ndarray]] = {}
    controller_index = 0
    for component in selected_components:
        spans = component_span_vectors_global(component.shape_xyz_global)
        npts = np.asarray(component.shape_xyz_global).shape[0]
        for inode in range(npts):
            node_id = controller_node_ids[controller_index]
            prepared[node_id] = (component, inode, spans[inode])
            controller_index += 1

    rows: list[dict[str, object]] = []
    velocity_offset = table_spec.swap_layout["velocity"].offset
    force_offset = table_spec.swap_layout["force"].offset
    for swap_index, requested_node_id in enumerate(requested):
        canonical = swap_node_ids[swap_index]
        controller_idx = int(swap_to_controller_idx[swap_index])
        controller_node_id = controller_node_ids[controller_idx]
        component, inode, span = prepared[controller_node_id]
        chord = np.asarray(component.chord_vector[inode], dtype=float)
        normal = np.asarray(component.normal_vector[inode], dtype=float)
        position = np.asarray(component.shape_xyz_global[inode], dtype=float)
        action_id, action_position = action_target_from_node_id(canonical)
        status, warnings, metrics = _audit_basis_status(chord, normal, span, tolerance=tolerance)
        rows.append(
            GeometryAuditRow(
                swap_index=swap_index,
                controller_index=controller_idx,
                requested_node_id=requested_node_id,
                canonical_node_id=canonical,
                controller_node_id=controller_node_id,
                action_id=action_id,
                action_position=action_position,
                force_swap_x=force_offset + 3 * swap_index,
                force_swap_y=force_offset + 3 * swap_index + 1,
                force_swap_z=force_offset + 3 * swap_index + 2,
                velocity_swap_x=velocity_offset + 3 * swap_index,
                velocity_swap_y=velocity_offset + 3 * swap_index + 1,
                velocity_swap_z=velocity_offset + 3 * swap_index + 2,
                qblade_reported_available=False,
                qblade_reported_x_m=None,
                qblade_reported_y_m=None,
                qblade_reported_z_m=None,
                vorlap_x_m=float(position[0]),
                vorlap_y_m=float(position[1]),
                vorlap_z_m=float(position[2]),
                delta_reported_m=None,
                chord_x=float(chord[0]),
                chord_y=float(chord[1]),
                chord_z=float(chord[2]),
                normal_x=float(normal[0]),
                normal_y=float(normal[1]),
                normal_z=float(normal[2]),
                span_x=float(span[0]),
                span_y=float(span[1]),
                span_z=float(span[2]),
                chord_norm=metrics["chord_norm"],
                normal_norm=metrics["normal_norm"],
                span_norm=metrics["span_norm"],
                chord_dot_normal=metrics["chord_dot_normal"],
                chord_dot_span=metrics["chord_dot_span"],
                normal_dot_span=metrics["normal_dot_span"],
                basis_determinant=metrics["basis_determinant"],
                basis_status=status,
                basis_warnings=warnings,
            ).as_dict()
        )
    return rows


def write_qblade_geometry_audit(
    path: str,
    components: Sequence[Component],
    requested_node_ids: Sequence[str],
    *,
    tolerance: float = 1.0e-8,
) -> str:
    """Write the per-node geometry/swap audit CSV used by setup audits."""
    rows = build_qblade_geometry_audit_rows(components, requested_node_ids, tolerance=tolerance)
    out_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(GEOMETRY_AUDIT_COLUMNS),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in GEOMETRY_AUDIT_COLUMNS})
    return out_path


__all__ = [
    "GEOMETRY_AUDIT_COLUMNS",
    "GeometryAuditRow",
    "build_qblade_geometry_audit_rows",
    "write_qblade_geometry_audit",
]
