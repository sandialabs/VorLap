"""QBlade external-library swap-table layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

from .qblade_controller import SwapBlockSpec
from .qblade_nodes import (
    action_target_from_node_id,
    position_channel_name,
    velocity_channel_name,
)


NodeID = str
SwapLayout = Dict[str, SwapBlockSpec]
ExternalInputRow = Tuple[int, str]
ExternalOutputRow = Tuple[int, str, str, float, str, bool]


@dataclass(frozen=True)
class ExternalLibraryTableSpec:
    """Describe the QBlade swap layout and the corresponding table rows."""

    swap_layout: SwapLayout
    input_rows: List[ExternalInputRow]
    output_rows: List[ExternalOutputRow]
    swap_size: int
    node_ids: Tuple[NodeID, ...]


def normalize_shape(shape: Union[None, int, Sequence[int], Tuple[int, ...]]) -> Tuple[int, ...]:
    """Normalize configured swap-block shapes into tuple form."""
    if shape is None:
        return ()
    if isinstance(shape, int):
        return () if shape == 0 else (int(shape),)
    shape_tuple = tuple(int(dim) for dim in shape)
    if len(shape_tuple) == 1 and shape_tuple[0] == 0:
        return ()
    if any(dim <= 0 for dim in shape_tuple):
        raise ValueError("swap block dimensions must be positive.")
    return shape_tuple


def validate_unique_node_ids(node_ids: Sequence[str], context: str) -> None:
    """Reject exact duplicate node ids before they can alias swap rows."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for node_id in node_ids:
        text = str(node_id)
        if text in seen and text not in duplicates:
            duplicates.append(text)
        seen.add(text)
    if duplicates:
        raise ValueError(f"Duplicate {context} node id(s): {', '.join(duplicates)}")


def build_external_library_table_spec(
    node_ids: Sequence[str],
    *,
    time_block: str = "time",
    timestep_block: str = "timestep",
    azimuth_block: str = "azimuth_deg",
    velocity_block: str = "velocity",
    force_block: str = "force",
    velocity_offset: int = 3,
) -> ExternalLibraryTableSpec:
    """
    Build the default QBlade external-library swap layout.

    The layout is:
      0 -> Time [s]
      1 -> Timestep [-]
      2 -> LSS Azimuthal Pos. [deg]
      3.. -> per-node global velocity components
      ... -> per-node global force components
    """
    node_ids_tuple = tuple(str(node_id) for node_id in node_ids)
    validate_unique_node_ids(node_ids_tuple, "external-library table")
    node_count = len(node_ids_tuple)
    if node_count < 1:
        raise ValueError("node_ids must contain at least one entry.")

    layout = {
        time_block: SwapBlockSpec(offset=0, shape=()),
        timestep_block: SwapBlockSpec(offset=1, shape=()),
        azimuth_block: SwapBlockSpec(offset=2, shape=()),
        velocity_block: SwapBlockSpec(offset=velocity_offset, shape=(node_count, 3)),
        force_block: SwapBlockSpec(offset=velocity_offset + 3 * node_count, shape=(node_count, 3)),
    }

    input_rows: List[Tuple[int, str]] = [
        (0, "Time [s]"),
        (1, "Timestep [-]"),
        (2, "LSS Azimuthal Pos. [deg]"),
    ]
    for inode, node_id in enumerate(node_ids_tuple):
        for icomp, axis in enumerate(("X", "Y", "Z")):
            idx = velocity_offset + 3 * inode + icomp
            input_rows.append((idx, velocity_channel_name(node_id, axis)))

    output_rows: List[Tuple[int, str, str, float, str, bool]] = []
    force_offset = velocity_offset + 3 * node_count
    for inode, node_id in enumerate(node_ids_tuple):
        action_id, action_position = action_target_from_node_id(node_id)
        for icomp, axis in enumerate(("X", "Y", "Z")):
            idx = force_offset + 3 * inode + icomp
            output_rows.append((idx, "ADDFORCE", action_id, action_position, axis, False))

    swap_size = force_offset + 3 * node_count
    return ExternalLibraryTableSpec(
        swap_layout=layout,
        input_rows=input_rows,
        output_rows=output_rows,
        swap_size=swap_size,
        node_ids=node_ids_tuple,
    )


def build_hvawt_response_table_spec(
    node_ids: Sequence[str],
    *,
    rpm_block: str = "lss_rpm",
    position_block: str = "position_global_m",
) -> ExternalLibraryTableSpec:
    """Append RPM and current global positions to the legacy 237-value table.

    Existing time, step, azimuth, velocity, and force indices remain byte-for-
    byte compatible.  For 39 H-VAWT sites, RPM is index 237, positions occupy
    238:354, and the resulting swap size is 355.
    """

    base = build_external_library_table_spec(node_ids)
    node_ids_tuple = base.node_ids
    rpm_offset = base.swap_size
    position_offset = rpm_offset + 1
    layout = dict(base.swap_layout)
    if rpm_block in layout or position_block in layout:
        raise ValueError("H-VAWT response block names collide with the legacy table.")
    layout[rpm_block] = SwapBlockSpec(offset=rpm_offset, shape=())
    layout[position_block] = SwapBlockSpec(
        offset=position_offset,
        shape=(len(node_ids_tuple), 3),
    )
    input_rows = [*base.input_rows, (rpm_offset, "LSS Rpm [rpm]")]
    for inode, node_id in enumerate(node_ids_tuple):
        action_id, action_position = action_target_from_node_id(node_id)
        for icomp, axis in enumerate(("X", "Y", "Z")):
            input_rows.append(
                (
                    position_offset + 3 * inode + icomp,
                    position_channel_name(
                        {
                            "id": action_id,
                            "position": action_position,
                            "direction": axis,
                        }
                    ),
                )
            )
    return ExternalLibraryTableSpec(
        swap_layout=layout,
        input_rows=input_rows,
        output_rows=list(base.output_rows),
        swap_size=position_offset + 3 * len(node_ids_tuple),
        node_ids=node_ids_tuple,
    )


__all__ = [
    "ExternalInputRow",
    "ExternalLibraryTableSpec",
    "ExternalOutputRow",
    "NodeID",
    "SwapLayout",
    "build_external_library_table_spec",
    "build_hvawt_response_table_spec",
    "normalize_shape",
    "validate_unique_node_ids",
]
