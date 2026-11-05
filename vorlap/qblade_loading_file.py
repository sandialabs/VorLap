"""QBlade external loading-file read/write and parity helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


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
    if time.size == 0:
        raise ValueError("time must contain at least one value.")
    if not np.all(np.isfinite(time)):
        raise ValueError("time values must be finite.")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError("time values must be strictly increasing for QBlade loading-file interpolation.")
    if not np.all(np.isfinite(force)):
        raise ValueError("global_force_vector_nodes values must be finite.")

    if node_torque_vector_nodes is None:
        torque = np.zeros_like(force)
    else:
        torque = np.asarray(node_torque_vector_nodes, dtype=float)
        if torque.shape != force.shape:
            raise ValueError("node_torque_vector_nodes must match shape [ntime, 3, nnodes].")
        if not np.all(np.isfinite(torque)):
            raise ValueError("node_torque_vector_nodes values must be finite.")

    suffix = " LOCAL" if local else ""
    with open(filename, "w", encoding="utf-8") as handle:
        handle.write("// VorLap-generated QBlade external loading file\n")
        handle.write("// Format per block: ID [LOCAL], then rows of time Fx Fy Fz Mx My Mz\n\n")
        for inode, node_id in enumerate(node_ids):
            handle.write(f"{node_id}{suffix}\n")
            for it, t in enumerate(time):
                fx, fy, fz = force[it, 0, inode], force[it, 1, inode], force[it, 2, inode]
                mx, my, mz = torque[it, 0, inode], torque[it, 1, inode], torque[it, 2, inode]
                handle.write(f"{t:.9g} {fx:.9g} {fy:.9g} {fz:.9g} {mx:.9g} {my:.9g} {mz:.9g}\n")
            handle.write("\n")


def read_qblade_loading_file(filename: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse a QBlade external loading file into node blocks.

    The returned mapping preserves file order and stores each node's `local`
    flag plus an `n x 7` float array with columns `t Fx Fy Fz Mx My Mz`.
    """
    blocks: Dict[str, Dict[str, Any]] = {}
    current_node: Optional[str] = None
    current_local = False
    current_rows: List[List[float]] = []

    def flush_current() -> None:
        nonlocal current_node, current_local, current_rows
        if current_node is None:
            return
        if not current_rows:
            raise ValueError(f"Loading-file node '{current_node}' has no data rows.")
        blocks[current_node] = {
            "local": current_local,
            "data": np.asarray(current_rows, dtype=float),
        }
        current_node = None
        current_local = False
        current_rows = []

    with open(filename, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            parts = line.split()
            try:
                row = [float(part) for part in parts]
            except ValueError:
                flush_current()
                if len(parts) not in (1, 2):
                    raise ValueError(f"Invalid loading-file node header at line {line_number}: {line}")
                node_id = parts[0]
                if node_id in blocks:
                    raise ValueError(f"Duplicate loading-file node block: {node_id}")
                local_flag = False
                if len(parts) == 2:
                    if parts[1].upper() != "LOCAL":
                        raise ValueError(f"Unsupported loading-file node flag at line {line_number}: {parts[1]}")
                    local_flag = True
                current_node = node_id
                current_local = local_flag
                current_rows = []
                continue
            if current_node is None:
                raise ValueError(f"Loading-file data row appears before a node header at line {line_number}.")
            if len(row) != 7:
                raise ValueError(f"Loading-file data row at line {line_number} must have 7 columns.")
            if not np.all(np.isfinite(row)):
                raise ValueError(f"Loading-file data row at line {line_number} contains non-finite values.")
            current_rows.append(row)

    flush_current()
    return blocks


def compare_qblade_loading_file(
    filename: str,
    time: np.ndarray,
    global_force_vector_nodes: np.ndarray,
    node_ids: List[str],
    *,
    local: Optional[bool] = None,
    node_torque_vector_nodes: Optional[np.ndarray] = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-9,
) -> Dict[str, Any]:
    """Compare a QBlade loading file against expected nodal force/torque arrays."""
    expected_time = np.asarray(time, dtype=float).reshape(-1)
    expected_force = np.asarray(global_force_vector_nodes, dtype=float)
    if expected_force.ndim != 3 or expected_force.shape[1] != 3:
        raise ValueError("global_force_vector_nodes must have shape [ntime, 3, nnodes].")
    if expected_force.shape[0] != expected_time.shape[0]:
        raise ValueError("time length must match global_force_vector_nodes time dimension.")
    if len(node_ids) != expected_force.shape[2]:
        raise ValueError("node_ids length must match global_force_vector_nodes node dimension.")
    if expected_time.size == 0:
        raise ValueError("time must contain at least one value.")
    if not np.all(np.isfinite(expected_time)):
        raise ValueError("time values must be finite.")
    if not np.all(np.isfinite(expected_force)):
        raise ValueError("global_force_vector_nodes values must be finite.")

    if node_torque_vector_nodes is None:
        expected_torque = np.zeros_like(expected_force)
    else:
        expected_torque = np.asarray(node_torque_vector_nodes, dtype=float)
        if expected_torque.shape != expected_force.shape:
            raise ValueError("node_torque_vector_nodes must match shape [ntime, 3, nnodes].")
        if not np.all(np.isfinite(expected_torque)):
            raise ValueError("node_torque_vector_nodes values must be finite.")

    blocks = read_qblade_loading_file(filename)
    expected_nodes = [str(node_id) for node_id in node_ids]
    parsed_nodes = list(blocks)
    missing_nodes = [node_id for node_id in expected_nodes if node_id not in blocks]
    extra_nodes = [node_id for node_id in parsed_nodes if node_id not in expected_nodes]
    local_mismatches: List[str] = []
    shape_mismatches: List[str] = []
    time_mismatches: List[str] = []
    force_mismatches: List[str] = []
    torque_mismatches: List[str] = []
    max_time_error = 0.0
    max_force_error = 0.0
    max_torque_error = 0.0
    compared_rows = 0

    for inode, node_id in enumerate(expected_nodes):
        block = blocks.get(node_id)
        if block is None:
            continue
        if local is not None and bool(block["local"]) != bool(local):
            local_mismatches.append(node_id)
        data = np.asarray(block["data"], dtype=float)
        if data.shape != (expected_time.size, 7):
            shape_mismatches.append(node_id)
            continue
        compared_rows += int(data.shape[0])
        time_error = np.abs(data[:, 0] - expected_time)
        force_error = np.abs(data[:, 1:4] - expected_force[:, :, inode])
        torque_error = np.abs(data[:, 4:7] - expected_torque[:, :, inode])
        max_time_error = max(max_time_error, float(np.max(time_error)))
        max_force_error = max(max_force_error, float(np.max(force_error)))
        max_torque_error = max(max_torque_error, float(np.max(torque_error)))
        if not np.allclose(data[:, 0], expected_time, rtol=rtol, atol=atol):
            time_mismatches.append(node_id)
        if not np.allclose(data[:, 1:4], expected_force[:, :, inode], rtol=rtol, atol=atol):
            force_mismatches.append(node_id)
        if not np.allclose(data[:, 4:7], expected_torque[:, :, inode], rtol=rtol, atol=atol):
            torque_mismatches.append(node_id)

    node_order_matches = parsed_nodes == expected_nodes
    passed = not (
        missing_nodes
        or extra_nodes
        or local_mismatches
        or shape_mismatches
        or time_mismatches
        or force_mismatches
        or torque_mismatches
        or not node_order_matches
    )
    return {
        "passed": passed,
        "node_count": len(expected_nodes),
        "compared_rows": compared_rows,
        "node_order_matches": node_order_matches,
        "missing_nodes": missing_nodes,
        "extra_nodes": extra_nodes,
        "local_mismatches": local_mismatches,
        "shape_mismatches": shape_mismatches,
        "time_mismatches": time_mismatches,
        "force_mismatches": force_mismatches,
        "torque_mismatches": torque_mismatches,
        "max_time_abs_error_s": max_time_error,
        "max_force_abs_error_n": max_force_error,
        "max_torque_abs_error_nm": max_torque_error,
    }


__all__ = [
    "compare_qblade_loading_file",
    "read_qblade_loading_file",
    "write_qblade_loading_file",
]
