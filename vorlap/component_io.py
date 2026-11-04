"""VorLap component CSV read/write helpers."""

from __future__ import annotations

import csv
import glob
import os
import re
from typing import List

import numpy as np

from .structs import Component


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

        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
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


def _metadata_row(path: str, raw_line: str | None) -> list[str]:
    if raw_line is None:
        raise ValueError(f"Component CSV is too short: {path}")
    rows = list(csv.reader([raw_line]))
    return rows[0] if rows else []


def load_components_from_csv(dir_path: str) -> List[Component]:
    """
    Loads all component geometry and metadata from CSV files in the given directory.

    Args:
        dir_path: Path to a directory containing CSV files. Each file must follow a two-header format.

    Returns:
        List of parsed Component objects containing all geometry and configuration data.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Components directory does not exist: {dir_path}")

    files = glob.glob(os.path.join(dir_path, "*.csv"))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No component CSV files were found in {dir_path}")

    components = []

    for file in files:
        with open(file, "r", newline="", encoding="utf-8") as handle:
            _metadata_header = handle.readline()
            metadata = _metadata_row(file, handle.readline())
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

            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Component CSV is too short: {file}")
            colnames = [col.strip() for col in reader.fieldnames]
            required_cols = {"x", "y", "z", "chord", "twist", "thickness", "offset"}
            missing = required_cols.difference(colnames)
            if missing:
                raise ValueError(f"Missing required columns in {file}: {sorted(missing)}")
            rows = [
                {str(key).strip(): value for key, value in row.items()}
                for row in reader
                if any(str(value).strip() for value in row.values() if value is not None)
            ]

        if not rows:
            raise ValueError(f"Component {id_str} in {file} has no geometry rows")

        xyz = np.column_stack(
            [
                [float(row["x"]) for row in rows],
                [float(row["y"]) for row in rows],
                [float(row["z"]) for row in rows],
            ]
        )
        chord = np.array([float(row["chord"]) for row in rows], dtype=float)
        twist = np.array([float(row["twist"]) for row in rows], dtype=float)
        thickness = np.array([float(row["thickness"]) for row in rows], dtype=float)
        offset = np.array([float(row["offset"]) for row in rows], dtype=float)

        if "airfoil_id" in colnames:
            airfoil_ids = [str(row.get("airfoil_id", "default") or "default") for row in rows]
        else:
            airfoil_ids = ["default"] * len(rows)

        chord_vec = np.zeros((xyz.shape[0], 3))
        norm_vec = np.zeros((xyz.shape[0], 3))
        xyz_global = np.zeros_like(xyz)

        components.append(
            Component(
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
                normal_vector=norm_vec,
            )
        )

    return components


__all__ = [
    "load_components_from_csv",
    "write_components_to_csv",
]
