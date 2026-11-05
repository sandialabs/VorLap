"""Inflow profile and force time-series CSV helpers."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .structs import InflowTimeSeries


def load_inflow_time_series(path: str) -> InflowTimeSeries:
    """
    Load a time-varying inflow profile from CSV.

    Required columns:
        - `time`
        - `inflow_speed`

    Direction column options:
        1) `inflow_direction_deg` (degrees CCW from +X in the global XY plane), or
        2) `inflow_dir_x`, `inflow_dir_y`, and optional `inflow_dir_z`.
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
    Write force time series data to a CSV file.

    Args:
        filename: Path to the output CSV file.
        output_time: Vector of time points.
        global_force_vector_nodes: Array of force vectors for each node at each time point.
    """
    output_time = np.asarray(output_time, dtype=float)
    global_force_vector_nodes = np.asarray(global_force_vector_nodes, dtype=float)
    if global_force_vector_nodes.ndim != 3 or global_force_vector_nodes.shape[1] != 3:
        raise ValueError("global_force_vector_nodes must have shape [ntime, 3, nnodes].")

    ntime, _, nnodes = global_force_vector_nodes.shape
    if output_time.shape[0] != ntime:
        raise ValueError("output_time length must match global_force_vector_nodes time dimension.")

    with open(filename, "w", encoding="utf-8") as handle:
        header = ["time"]
        for n in range(nnodes):
            header.extend([f"node{n+1}x", f"node{n+1}y", f"node{n+1}z"])
        handle.write(", ".join(header) + "\n")

        for t in range(ntime):
            row = [str(output_time[t])]
            for n in range(nnodes):
                fx = global_force_vector_nodes[t, 0, n]
                fy = global_force_vector_nodes[t, 1, n]
                fz = global_force_vector_nodes[t, 2, n]
                row.extend([str(fx), str(fy), str(fz)])
            handle.write(", ".join(row) + "\n")


__all__ = [
    "load_inflow_time_series",
    "write_force_time_series",
]
