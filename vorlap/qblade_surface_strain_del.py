"""Recover dense QBlade surface normal-strain DEL fields.

QBlade exports section resultants, not a chordwise stress or strain field.  A
surface field can nevertheless be recovered for the beam model by combining
the retained local axial force and two local bending moments with the same
section stiffness matrices used by the QBlade-to-OWENS conversion.  This
module binds every dense QBlade monitor to a physical converted beam segment,
solves the generalized section strain, and rainflow-counts the axial strain at
the actual NACA 0018 perimeter (or at a circular support perimeter).

The result is a response DEL field.  It is not material stress, fatigue life,
damage, utilization, or a replacement for a sectional finite-element model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from .qblade_full_structure_output import (
    MonitorPoint,
    build_monitor_plan,
    filter_channels,
)

if TYPE_CHECKING:
    import pandas as pd


SCHEMA = "vorlap-qblade-surface-normal-strain-del-v1"
AIRFOIL_PERIMETER_KIND = "airfoil"
CIRCLE_PERIMETER_KIND = "circle"
PHYSICAL_KIND_ID = 1
COMPONENT_IDS = {
    "substructure": 1,
    "tower": 2,
    "torque_tube": 3,
    "blade": 4,
    "lower_strut": 5,
    "upper_strut": 6,
}


@dataclass(frozen=True)
class OwensElement:
    element_id: int
    component_id: int
    start_node: int
    stop_node: int
    length_m: float
    start_xyz_m: np.ndarray
    stop_xyz_m: np.ndarray
    frame: np.ndarray
    stiffness: np.ndarray
    start_half_y_m: float
    start_half_z_m: float
    stop_half_y_m: float
    stop_half_z_m: float


@dataclass(frozen=True)
class MonitorBinding:
    monitor_index: int
    monitor: MonitorPoint
    element: OwensElement
    segment_start_position: float
    segment_stop_position: float
    segment_start_xyz_m: np.ndarray
    segment_stop_xyz_m: np.ndarray
    start_half_y_m: float
    start_half_z_m: float
    stop_half_y_m: float
    stop_half_z_m: float


@dataclass(frozen=True)
class SurfacePoint:
    perimeter_index: int
    perimeter_kind: str
    profile_x: float
    profile_y: float
    angle_deg: float
    local_y_m: float
    local_z_m: float
    segment_start_xyz_m: np.ndarray
    segment_stop_xyz_m: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampled_duration_seconds(time_s: np.ndarray) -> float:
    finite = np.asarray(time_s, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0
    differences = np.diff(finite)
    positive = differences[np.isfinite(differences) & (differences > 0.0)]
    if positive.size == 0:
        return 0.0
    return float(finite[-1] - finite[0] + np.median(positive))


def read_qblade_results(path: Path) -> "pd.DataFrame":
    import pandas as pd

    if not Path(path).is_file():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, sep="\t", skiprows=2, engine="python")
    unnamed = [name for name in data.columns if str(name).startswith("Unnamed:")]
    if unnamed:
        data = data.drop(columns=unnamed)
    return data.apply(pd.to_numeric, errors="coerce")


def turning_points_vectorized(values: np.ndarray) -> np.ndarray:
    """Return the same turning points as the campaign rainflow implementation."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1:
        raise ValueError("rainflow history must be one-dimensional")
    nonfinite = np.flatnonzero(~np.isfinite(samples))
    if nonfinite.size:
        preview = ", ".join(str(int(index)) for index in nonfinite[:5])
        raise ValueError(
            f"rainflow history contains non-finite sample(s) at index {preview}"
        )
    if samples.size <= 2:
        return samples
    compact = samples[np.r_[True, samples[1:] != samples[:-1]]]
    if compact.size <= 2:
        return compact
    with np.errstate(over="ignore", invalid="ignore"):
        first_difference = compact[1:-1] - compact[:-2]
        second_difference = compact[2:] - compact[1:-1]
    if np.any(~np.isfinite(first_difference)) or np.any(~np.isfinite(second_difference)):
        raise FloatingPointError("rainflow turning-point difference became non-finite")
    reverses = (
        (first_difference == 0.0)
        | (second_difference == 0.0)
        | (np.signbit(first_difference) != np.signbit(second_difference))
    )
    extrema = compact[1:-1][reverses]
    return np.concatenate((compact[:1], extrema, compact[-1:]))


def _finite_range_and_damage(
    first: float,
    second: float,
    count: float,
    wohler_exponent: float,
) -> tuple[float, float]:
    cycle_range = abs(float(second) - float(first))
    if not math.isfinite(cycle_range):
        raise FloatingPointError("rainflow range became non-finite")
    try:
        increment = float(count) * cycle_range**wohler_exponent
    except OverflowError as exc:
        raise FloatingPointError("rainflow damage increment became non-finite") from exc
    if not math.isfinite(increment):
        raise FloatingPointError("rainflow damage increment became non-finite")
    return cycle_range, increment


def equivalent_load_with_cycles_fast(
    values: np.ndarray,
    wohler_exponent: float,
    equivalent_cycles: float,
) -> tuple[float, float]:
    if not math.isfinite(wohler_exponent) or wohler_exponent <= 0.0:
        raise ValueError("wohler_exponent must be finite and positive")
    if not math.isfinite(equivalent_cycles) or equivalent_cycles <= 0.0:
        raise ValueError("equivalent_cycles must be finite and positive")
    points = turning_points_vectorized(values)
    if points.size < 2:
        return 0.0, 0.0
    stack: list[float] = []
    damage = 0.0
    cycle_count = 0.0
    for raw in points:
        stack.append(float(raw))
        while len(stack) >= 3:
            old_range, _ = _finite_range_and_damage(
                stack[-3], stack[-2], 0.0, wohler_exponent
            )
            new_range, _ = _finite_range_and_damage(
                stack[-2], stack[-1], 0.0, wohler_exponent
            )
            if new_range < old_range:
                break
            if old_range > 0.0:
                count = 0.5 if len(stack) == 3 else 1.0
                _, increment = _finite_range_and_damage(
                    stack[-3], stack[-2], count, wohler_exponent
                )
                damage += increment
                if not math.isfinite(damage):
                    raise FloatingPointError("rainflow damage accumulation became non-finite")
                cycle_count += count
            if len(stack) == 3:
                stack.pop(0)
            else:
                last = stack.pop()
                stack.pop()
                stack.pop()
                stack.append(last)
    for first, second in zip(stack, stack[1:]):
        residual_range, increment = _finite_range_and_damage(
            first, second, 0.5, wohler_exponent
        )
        if residual_range > 0.0:
            damage += increment
            if not math.isfinite(damage):
                raise FloatingPointError("rainflow damage accumulation became non-finite")
            cycle_count += 0.5
    if damage <= 0.0 or cycle_count <= 0.0:
        return 0.0, cycle_count
    equivalent_load = float((damage / equivalent_cycles) ** (1.0 / wohler_exponent))
    if not math.isfinite(equivalent_load):
        raise FloatingPointError("equivalent load became non-finite")
    return equivalent_load, cycle_count


def _matrix(row: dict[str, str], prefix: str, size: int) -> np.ndarray:
    return np.asarray(
        [
            [float(row[f"{prefix}_{i}{j}"]) for j in range(1, size + 1)]
            for i in range(1, size + 1)
        ],
        dtype=float,
    )


def read_owens_model(model_dir: Path) -> list[OwensElement]:
    model_dir = Path(model_dir).resolve()
    with (model_dir / "nodes.csv").open(newline="", encoding="utf-8") as stream:
        nodes = {
            int(row["node_id"]): np.asarray(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])],
                dtype=float,
            )
            for row in csv.DictReader(stream)
        }
    with (model_dir / "visual_sections.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        visuals = {
            int(row["node_id"]): (
                float(row["half_y_m"]),
                float(row["half_z_m"]),
            )
            for row in csv.DictReader(stream)
        }
    elements: list[OwensElement] = []
    with (model_dir / "elements.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["kind_id"]) != PHYSICAL_KIND_ID:
                continue
            start_node = int(row["start_node"])
            stop_node = int(row["stop_node"])
            stiffness = _matrix(row, "k", 6)
            if not np.allclose(stiffness, stiffness.T, rtol=1.0e-10, atol=1.0e-6):
                raise ValueError(f"OWENS element {row['element_id']} stiffness is not symmetric")
            eigenvalues = np.linalg.eigvalsh(stiffness)
            if np.min(eigenvalues) <= 0.0:
                raise ValueError(
                    f"OWENS element {row['element_id']} stiffness is not positive definite"
                )
            elements.append(
                OwensElement(
                    element_id=int(row["element_id"]),
                    component_id=int(row["component_id"]),
                    start_node=start_node,
                    stop_node=stop_node,
                    length_m=float(row["length_m"]),
                    start_xyz_m=nodes[start_node],
                    stop_xyz_m=nodes[stop_node],
                    frame=_matrix(row, "frame", 3),
                    stiffness=stiffness,
                    start_half_y_m=visuals[start_node][0],
                    start_half_z_m=visuals[start_node][1],
                    stop_half_y_m=visuals[stop_node][0],
                    stop_half_z_m=visuals[stop_node][1],
                )
            )
    elements.sort(key=lambda element: element.element_id)
    if not elements:
        raise ValueError(f"No physical OWENS elements found in {model_dir}")
    return elements


def _runs(elements: Sequence[OwensElement], component_id: int) -> list[list[OwensElement]]:
    selected = [element for element in elements if element.component_id == component_id]
    runs: list[list[OwensElement]] = []
    for element in selected:
        if not runs or element.element_id != runs[-1][-1].element_id + 1:
            runs.append([element])
        else:
            runs[-1].append(element)
    return runs


def _monitor_groups(monitors: Sequence[MonitorPoint]) -> dict[str, list[tuple[int, MonitorPoint]]]:
    groups: dict[str, list[tuple[int, MonitorPoint]]] = {}
    for index, monitor in enumerate(monitors):
        groups.setdefault(monitor.component, []).append((index, monitor))
    for rows in groups.values():
        rows.sort(key=lambda item: item[1].position)
    return groups


def _interpolate(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    return first + fraction * (second - first)


def _bind_exact_run(
    monitor_rows: Sequence[tuple[int, MonitorPoint]],
    elements: Sequence[OwensElement],
    *,
    tolerance: float,
) -> list[MonitorBinding]:
    if len(monitor_rows) != len(elements):
        raise ValueError(
            f"Monitor/element count mismatch for {monitor_rows[0][1].component}: "
            f"{len(monitor_rows)} versus {len(elements)}"
        )
    total_length = sum(element.length_m for element in elements)
    if total_length <= 0.0:
        raise ValueError("Bound OWENS component has zero length")
    bindings: list[MonitorBinding] = []
    distance = 0.0
    for (monitor_index, monitor), element in zip(monitor_rows, elements):
        start = distance / total_length
        stop = (distance + element.length_m) / total_length
        midpoint = 0.5 * (start + stop)
        if not math.isclose(midpoint, monitor.position, abs_tol=tolerance):
            raise ValueError(
                f"QBlade/OWENS station mismatch for {monitor.component}: "
                f"monitor={monitor.position:.12g}, element midpoint={midpoint:.12g}"
            )
        bindings.append(
            MonitorBinding(
                monitor_index=monitor_index,
                monitor=monitor,
                element=element,
                segment_start_position=start,
                segment_stop_position=stop,
                segment_start_xyz_m=element.start_xyz_m,
                segment_stop_xyz_m=element.stop_xyz_m,
                start_half_y_m=element.start_half_y_m,
                start_half_z_m=element.start_half_z_m,
                stop_half_y_m=element.stop_half_y_m,
                stop_half_z_m=element.stop_half_z_m,
            )
        )
        distance += element.length_m
    return bindings


def _bind_substructure_member(
    monitor_rows: Sequence[tuple[int, MonitorPoint]],
    element: OwensElement,
    *,
    tolerance: float,
) -> list[MonitorBinding]:
    positions = [monitor.position for _, monitor in monitor_rows]
    if len(positions) == 1:
        boundaries = [0.0, 1.0]
    else:
        boundaries = [0.0]
        boundaries.extend(0.5 * (a + b) for a, b in zip(positions, positions[1:]))
        boundaries.append(1.0)
    bindings: list[MonitorBinding] = []
    for index, (monitor_index, monitor) in enumerate(monitor_rows):
        start = boundaries[index]
        stop = boundaries[index + 1]
        if not math.isclose(0.5 * (start + stop), monitor.position, abs_tol=tolerance):
            raise ValueError(
                f"Substructure monitor {monitor.component} {monitor.position} is not centered "
                f"in its visualization interval [{start}, {stop}]"
            )
        bindings.append(
            MonitorBinding(
                monitor_index=monitor_index,
                monitor=monitor,
                element=element,
                segment_start_position=start,
                segment_stop_position=stop,
                segment_start_xyz_m=_interpolate(
                    element.start_xyz_m, element.stop_xyz_m, start
                ),
                segment_stop_xyz_m=_interpolate(
                    element.start_xyz_m, element.stop_xyz_m, stop
                ),
                start_half_y_m=(
                    element.start_half_y_m
                    + start * (element.stop_half_y_m - element.start_half_y_m)
                ),
                start_half_z_m=(
                    element.start_half_z_m
                    + start * (element.stop_half_z_m - element.start_half_z_m)
                ),
                stop_half_y_m=(
                    element.start_half_y_m
                    + stop * (element.stop_half_y_m - element.start_half_y_m)
                ),
                stop_half_z_m=(
                    element.start_half_z_m
                    + stop * (element.stop_half_z_m - element.start_half_z_m)
                ),
            )
        )
    return bindings


def bind_monitors_to_owens(
    monitors: Sequence[MonitorPoint],
    elements: Sequence[OwensElement],
    *,
    tolerance: float = 2.0e-7,
) -> list[MonitorBinding]:
    groups = _monitor_groups(monitors)
    expected_components = {
        "TWR",
        "TRQ",
        "SUB_1",
        "SUB_2",
        "BLD_1",
        "BLD_2",
        "BLD_3",
        "STR_1_1",
        "STR_1_2",
        "STR_1_3",
        "STR_2_1",
        "STR_2_2",
        "STR_2_3",
    }
    if set(groups) != expected_components:
        raise ValueError(
            "Unexpected full-structure monitor components: "
            f"{sorted(set(groups) ^ expected_components)}"
        )

    blade_runs = _runs(elements, COMPONENT_IDS["blade"])
    lower_runs = _runs(elements, COMPONENT_IDS["lower_strut"])
    upper_runs = _runs(elements, COMPONENT_IDS["upper_strut"])
    tower_runs = _runs(elements, COMPONENT_IDS["tower"])
    torque_runs = _runs(elements, COMPONENT_IDS["torque_tube"])
    sub_runs = _runs(elements, COMPONENT_IDS["substructure"])
    if [len(blade_runs), len(lower_runs), len(upper_runs)] != [3, 3, 3]:
        raise ValueError("OWENS blade/strut physical elements do not form three body runs")
    if len(tower_runs) != 1 or len(torque_runs) != 1:
        raise ValueError("OWENS tower and torque tube must each form one physical run")
    if len(sub_runs) != 1 or len(sub_runs[0]) != 2:
        raise ValueError("OWENS substructure must contain the two reference tower members")

    bindings: list[MonitorBinding] = []
    bindings.extend(_bind_exact_run(groups["TWR"], tower_runs[0], tolerance=tolerance))
    bindings.extend(_bind_exact_run(groups["TRQ"], torque_runs[0], tolerance=tolerance))
    for blade_index in range(1, 4):
        bindings.extend(
            _bind_exact_run(
                groups[f"BLD_{blade_index}"],
                blade_runs[blade_index - 1],
                tolerance=tolerance,
            )
        )
        bindings.extend(
            _bind_exact_run(
                groups[f"STR_1_{blade_index}"],
                lower_runs[blade_index - 1],
                tolerance=tolerance,
            )
        )
        bindings.extend(
            _bind_exact_run(
                groups[f"STR_2_{blade_index}"],
                upper_runs[blade_index - 1],
                tolerance=tolerance,
            )
        )
    bindings.extend(
        _bind_substructure_member(groups["SUB_1"], sub_runs[0][0], tolerance=tolerance)
    )
    bindings.extend(
        _bind_substructure_member(groups["SUB_2"], sub_runs[0][1], tolerance=tolerance)
    )
    bindings.sort(key=lambda binding: binding.monitor_index)
    if [binding.monitor_index for binding in bindings] != list(range(len(monitors))):
        raise ValueError("Monitor bindings are incomplete or duplicated")
    return bindings


def read_airfoil_coordinates(path: Path) -> np.ndarray:
    coordinates: list[tuple[float, float]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        tokens = raw.split()
        if len(tokens) < 2:
            continue
        try:
            coordinates.append((float(tokens[0]), float(tokens[1])))
        except ValueError:
            continue
    values = np.asarray(coordinates, dtype=float)
    if values.shape[0] < 20 or values.shape[1] != 2:
        raise ValueError(f"Incomplete airfoil coordinates in {path}")
    if np.any(values[:, 0] < -1.0e-9) or np.any(values[:, 0] > 1.0 + 1.0e-9):
        raise ValueError(f"Airfoil x/c values are outside [0, 1] in {path}")
    return values


def read_reference_pitch_axes(path: Path) -> np.ndarray:
    """Read every blade and strut pitch-axis fraction from a QBlade .bld file."""
    lines = Path(path).read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        header = next(index for index, line in enumerate(lines) if "HEIGHT_[m]" in line)
    except StopIteration as exc:
        raise ValueError(f"No blade geometry table found in {path}") from exc

    values: list[float] = []
    started = False
    for raw in lines[header + 1 :]:
        tokens = raw.split("//", 1)[0].split()
        if not tokens:
            if started:
                break
            continue
        try:
            row = [float(token) for token in tokens[:7]]
        except ValueError:
            if started:
                break
            continue
        if len(row) < 7:
            if started:
                break
            continue
        values.append(row[6])
        started = True

    for raw in lines:
        tokens = raw.split("//", 1)[0].split()
        if len(tokens) >= 2 and tokens[1] in {"PAXISHUB_STR", "PAXISBLD_STR"}:
            values.append(float(tokens[0]))
    axes = np.asarray(values, dtype=float)
    if axes.size == 0 or not np.all(np.isfinite(axes)):
        raise ValueError(f"No finite blade/strut pitch axes found in {path}")
    if np.any(axes < 0.0) or np.any(axes > 1.0):
        raise ValueError(f"Blade/strut pitch axes are outside [0, 1] in {path}")
    return axes


def validate_pitch_axis(path: Path, requested: float, tolerance: float = 1.0e-10) -> None:
    axes = read_reference_pitch_axes(path)
    if not np.allclose(axes, requested, rtol=0.0, atol=tolerance):
        unique = sorted({float(value) for value in axes})
        raise ValueError(
            f"Requested pitch-axis fraction {requested:.12g} does not match "
            f"the QBlade blade/strut definitions {unique} in {path}"
        )


def _global_surface_point(
    center: np.ndarray,
    frame: np.ndarray,
    local_y: float,
    local_z: float,
) -> np.ndarray:
    return center + frame[:, 1] * local_y + frame[:, 2] * local_z


def _circular_radius(half_y: float, half_z: float, *, label: str) -> float:
    if not math.isclose(half_y, half_z, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError(
            f"{label} is not circular in the OWENS visualization model: "
            f"half_y={half_y:.12g}, half_z={half_z:.12g}"
        )
    if not half_y > 0.0:
        raise ValueError(f"{label} has a nonpositive section radius")
    return half_y


def surface_points_for_binding(
    binding: MonitorBinding,
    airfoil_coordinates: np.ndarray,
    *,
    pitch_axis_fraction: float,
    circle_points: int,
) -> list[SurfacePoint]:
    if not 0.0 <= pitch_axis_fraction <= 1.0:
        raise ValueError("pitch_axis_fraction must lie in [0, 1]")
    monitor = binding.monitor
    is_airfoil = monitor.family in {"blade", "strut"}
    if is_airfoil:
        profile = [
            (index, AIRFOIL_PERIMETER_KIND, float(x), float(z), math.nan)
            for index, (x, z) in enumerate(airfoil_coordinates)
        ]
    else:
        if circle_points < 12:
            raise ValueError("circle_points must be at least 12")
        profile = [
            (
                index,
                CIRCLE_PERIMETER_KIND,
                math.nan,
                math.nan,
                360.0 * index / circle_points,
            )
            for index in range(circle_points)
        ]

    points: list[SurfacePoint] = []
    for perimeter_index, kind, profile_x, profile_y, angle_deg in profile:
        if is_airfoil:
            chord_start = 2.0 * binding.start_half_y_m
            chord_stop = 2.0 * binding.stop_half_y_m
            chord_mid = 0.5 * (chord_start + chord_stop)
            start_y = (pitch_axis_fraction - profile_x) * chord_start
            start_z = profile_y * chord_start
            stop_y = (pitch_axis_fraction - profile_x) * chord_stop
            stop_z = profile_y * chord_stop
            local_y = (pitch_axis_fraction - profile_x) * chord_mid
            local_z = profile_y * chord_mid
        else:
            angle = math.radians(angle_deg)
            radius_start = _circular_radius(
                binding.start_half_y_m,
                binding.start_half_z_m,
                label=f"{monitor.component} segment start",
            )
            radius_stop = _circular_radius(
                binding.stop_half_y_m,
                binding.stop_half_z_m,
                label=f"{monitor.component} segment stop",
            )
            radius_mid = 0.5 * (radius_start + radius_stop)
            start_y = radius_start * math.cos(angle)
            start_z = radius_start * math.sin(angle)
            stop_y = radius_stop * math.cos(angle)
            stop_z = radius_stop * math.sin(angle)
            local_y = radius_mid * math.cos(angle)
            local_z = radius_mid * math.sin(angle)
        points.append(
            SurfacePoint(
                perimeter_index=perimeter_index,
                perimeter_kind=kind,
                profile_x=profile_x,
                profile_y=profile_y,
                angle_deg=angle_deg,
                local_y_m=local_y,
                local_z_m=local_z,
                segment_start_xyz_m=_global_surface_point(
                    binding.segment_start_xyz_m, binding.element.frame, start_y, start_z
                ),
                segment_stop_xyz_m=_global_surface_point(
                    binding.segment_stop_xyz_m, binding.element.frame, stop_y, stop_z
                ),
            )
        )
    return points


def qblade_column(channel: str) -> str:
    return channel.replace(" ", "~")


def validate_dense_result_history(
    data: "pd.DataFrame",
    monitors: Sequence[MonitorPoint],
    *,
    expected_sample_count: int,
    expected_start_s: float,
    expected_timestep_s: float,
    expected_rpm: float,
    rpm_tolerance: float,
    minimum_duration_s: float,
    allow_short: bool,
) -> dict[str, float | int]:
    """Verify that a dense export is the planned retained production history."""
    expected_columns = [qblade_column(channel) for channel in filter_channels(monitors)]
    observed_columns = [str(column) for column in data.columns]
    if observed_columns != expected_columns:
        missing = [column for column in expected_columns if column not in observed_columns]
        extra = [column for column in observed_columns if column not in expected_columns]
        mismatch = next(
            (
                index
                for index, (expected, observed) in enumerate(
                    zip(expected_columns, observed_columns)
                )
                if expected != observed
            ),
            min(len(expected_columns), len(observed_columns)),
        )
        raise ValueError(
            "Dense result columns do not match the ordered full-structure plan: "
            f"expected={len(expected_columns)}, observed={len(observed_columns)}, "
            f"first_mismatch={mismatch}, missing={missing[:5]}, extra={extra[:5]}"
        )
    if expected_sample_count < 2:
        raise ValueError("expected_sample_count must be at least two")
    if expected_timestep_s <= 0.0 or not math.isfinite(expected_timestep_s):
        raise ValueError("expected_timestep_s must be finite and positive")
    if rpm_tolerance < 0.0 or not math.isfinite(rpm_tolerance):
        raise ValueError("rpm_tolerance must be finite and nonnegative")

    time_s = np.asarray(data["Time~[s]"], dtype=float)
    rpm = np.asarray(data["LSS~Rpm~[rpm]"], dtype=float)
    if time_s.size < 2 or not np.all(np.isfinite(time_s)):
        raise ValueError("Dense result has an invalid time vector")
    if rpm.shape != time_s.shape or not np.all(np.isfinite(rpm)):
        raise ValueError("Dense result has an invalid RPM vector")
    differences = np.diff(time_s)
    if np.any(differences <= 0.0):
        raise ValueError("Dense result time is not strictly increasing")
    duration_s = sampled_duration_seconds(time_s)
    timestep_median = float(np.median(differences))
    timestep_max_error = float(np.max(np.abs(differences - expected_timestep_s)))
    rpm_max_error = float(np.max(np.abs(rpm - expected_rpm)))
    if not allow_short:
        if time_s.size != expected_sample_count:
            raise ValueError(
                f"Dense result has {time_s.size} samples; expected {expected_sample_count}"
            )
        if not math.isclose(time_s[0], expected_start_s, abs_tol=1.0e-7):
            raise ValueError(
                f"Dense result starts at {time_s[0]:.12g} s; expected "
                f"{expected_start_s:.12g} s"
            )
        if timestep_max_error > 1.0e-7:
            raise ValueError(
                f"Dense result timestep differs from {expected_timestep_s:.12g} s "
                f"by as much as {timestep_max_error:.3g} s"
            )
        if duration_s < minimum_duration_s - 0.01:
            raise ValueError(
                f"Dense result retained duration {duration_s:.12g} s is below "
                f"the required {minimum_duration_s:.12g} s"
            )
        if rpm_max_error > rpm_tolerance:
            raise ValueError(
                f"Dense result RPM differs from {expected_rpm:.12g} by as much "
                f"as {rpm_max_error:.6g} RPM"
            )
    return {
        "expected_result_column_count": len(expected_columns),
        "observed_result_column_count": len(observed_columns),
        "sample_count": int(time_s.size),
        "first_time_s": float(time_s[0]),
        "last_time_s": float(time_s[-1]),
        "sampled_duration_s": duration_s,
        "timestep_median_s": timestep_median,
        "timestep_max_error_s": timestep_max_error,
        "rpm_min": float(np.min(rpm)),
        "rpm_mean": float(np.mean(rpm)),
        "rpm_max": float(np.max(rpm)),
        "rpm_max_error": rpm_max_error,
    }


def recover_generalized_strain(
    data: "pd.DataFrame",
    binding: MonitorBinding,
) -> np.ndarray:
    channels = [qblade_column(channel) for channel in binding.monitor.result_channels()]
    missing = [channel for channel in channels if channel not in data]
    if missing:
        raise KeyError(
            f"Missing dense result channels for {binding.monitor.component}: {missing}"
        )
    loads = np.zeros((6, len(data)), dtype=float)
    loads[0, :] = np.asarray(data[channels[0]], dtype=float)
    loads[4, :] = np.asarray(data[channels[1]], dtype=float)
    loads[5, :] = np.asarray(data[channels[2]], dtype=float)
    if not np.all(np.isfinite(loads)):
        raise ValueError(f"Nonfinite QBlade resultants for {binding.monitor.component}")
    generalized = np.linalg.solve(binding.element.stiffness, loads)
    if not np.all(np.isfinite(generalized)):
        raise ValueError(f"Nonfinite recovered strains for {binding.monitor.component}")
    return generalized


def axial_surface_normal_strain_history(
    generalized_strain: np.ndarray,
    *,
    local_y_m: float,
    local_z_m: float,
) -> np.ndarray:
    """Recover beam-axis normal strain at one QBlade section-surface point."""
    values = np.asarray(generalized_strain, dtype=float)
    if values.ndim != 2 or values.shape[0] != 6:
        raise ValueError("generalized_strain must have shape (6, sample_count)")
    return (
        values[0, :]
        - values[4, :] * float(local_z_m)
        + values[5, :] * float(local_y_m)
    )


def compute_scenario_surface_dels(
    data: "pd.DataFrame",
    bindings: Sequence[MonitorBinding],
    airfoil_coordinates: np.ndarray,
    *,
    pitch_axis_fraction: float,
    circle_points: int,
    wohler_exponent: float,
    equivalent_cycles: float,
) -> tuple[list[list[dict[str, float | int | str]]], dict[str, float | int]]:
    result: list[list[dict[str, float | int | str]]] = []
    cycle_min = math.inf
    cycle_max = -math.inf
    for binding in bindings:
        generalized = recover_generalized_strain(data, binding)
        surface_points = surface_points_for_binding(
            binding,
            airfoil_coordinates,
            pitch_axis_fraction=pitch_axis_fraction,
            circle_points=circle_points,
        )
        monitor_rows: list[dict[str, float | int | str]] = []
        for point in surface_points:
            # QBlade local Z maps to GXBeam local x, QBlade local X to GXBeam
            # local y, and QBlade local Y to GXBeam local z.  The axial strain
            # at (y,z) is e0 - kappa_y*z + kappa_z*y.
            history = axial_surface_normal_strain_history(
                generalized,
                local_y_m=point.local_y_m,
                local_z_m=point.local_z_m,
            )
            value, cycles = equivalent_load_with_cycles_fast(
                history,
                wohler_exponent,
                equivalent_cycles,
            )
            cycle_min = min(cycle_min, cycles)
            cycle_max = max(cycle_max, cycles)
            monitor_rows.append(
                {
                    "perimeter_index": point.perimeter_index,
                    "perimeter_kind": point.perimeter_kind,
                    "profile_x": point.profile_x,
                    "profile_y": point.profile_y,
                    "angle_deg": point.angle_deg,
                    "local_y_m": point.local_y_m,
                    "local_z_m": point.local_z_m,
                    "segment_start_x_m": point.segment_start_xyz_m[0],
                    "segment_start_y_m": point.segment_start_xyz_m[1],
                    "segment_start_z_m": point.segment_start_xyz_m[2],
                    "segment_stop_x_m": point.segment_stop_xyz_m[0],
                    "segment_stop_y_m": point.segment_stop_xyz_m[1],
                    "segment_stop_z_m": point.segment_stop_xyz_m[2],
                    "strain_del": value,
                    "rainflow_cycles": cycles,
                }
            )
        result.append(monitor_rows)
    return result, {
        "minimum_rainflow_cycles": 0.0 if math.isinf(cycle_min) else cycle_min,
        "maximum_rainflow_cycles": 0.0 if math.isinf(cycle_max) else cycle_max,
    }


FIELD_COLUMNS = [
    "monitor_index",
    "component",
    "result_component",
    "family",
    "member_index",
    "station_position",
    "segment_start_position",
    "segment_stop_position",
    "owens_element_id",
    "owens_component_id",
    "perimeter_index",
    "perimeter_kind",
    "profile_x",
    "profile_y",
    "angle_deg",
    "local_y_m",
    "local_z_m",
    "segment_start_x_m",
    "segment_start_y_m",
    "segment_start_z_m",
    "segment_stop_x_m",
    "segment_stop_y_m",
    "segment_stop_z_m",
    "baseline_strain_del",
    "vorlap_strain_del",
    "ratio_vorlap_to_baseline",
    "baseline_rainflow_cycles",
    "vorlap_rainflow_cycles",
]


def combine_scenario_fields(
    bindings: Sequence[MonitorBinding],
    baseline: Sequence[Sequence[dict[str, float | int | str]]],
    vorlap: Sequence[Sequence[dict[str, float | int | str]]],
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    if len(baseline) != len(bindings) or len(vorlap) != len(bindings):
        raise ValueError("Scenario surface fields do not match monitor bindings")
    rows: list[dict[str, float | int | str]] = []
    undefined = 0
    for binding, baseline_points, vorlap_points in zip(bindings, baseline, vorlap):
        if len(baseline_points) != len(vorlap_points):
            raise ValueError(f"Perimeter mismatch for {binding.monitor.component}")
        for base, coupled in zip(baseline_points, vorlap_points):
            if base["perimeter_index"] != coupled["perimeter_index"]:
                raise ValueError(f"Perimeter ordering mismatch for {binding.monitor.component}")
            baseline_del = float(base["strain_del"])
            vorlap_del = float(coupled["strain_del"])
            if baseline_del > 0.0:
                ratio = vorlap_del / baseline_del
            else:
                ratio = math.nan
                undefined += 1
            rows.append(
                {
                    "monitor_index": binding.monitor_index,
                    "component": binding.monitor.component,
                    "result_component": (
                        binding.monitor.result_component or binding.monitor.component
                    ),
                    "family": binding.monitor.family,
                    "member_index": binding.monitor.member_index,
                    "station_position": binding.monitor.position,
                    "segment_start_position": binding.segment_start_position,
                    "segment_stop_position": binding.segment_stop_position,
                    "owens_element_id": binding.element.element_id,
                    "owens_component_id": binding.element.component_id,
                    **{
                        key: base[key]
                        for key in (
                            "perimeter_index",
                            "perimeter_kind",
                            "profile_x",
                            "profile_y",
                            "angle_deg",
                            "local_y_m",
                            "local_z_m",
                            "segment_start_x_m",
                            "segment_start_y_m",
                            "segment_start_z_m",
                            "segment_stop_x_m",
                            "segment_stop_y_m",
                            "segment_stop_z_m",
                        )
                    },
                    "baseline_strain_del": baseline_del,
                    "vorlap_strain_del": vorlap_del,
                    "ratio_vorlap_to_baseline": ratio,
                    "baseline_rainflow_cycles": base["rainflow_cycles"],
                    "vorlap_rainflow_cycles": coupled["rainflow_cycles"],
                }
            )
    return rows, {
        "surface_field_row_count": len(rows),
        "undefined_ratio_count": undefined,
    }


def write_field_csv(path: Path, rows: Iterable[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_surface_strain_del_field(
    *,
    baseline_results: Path,
    vorlap_results: Path,
    reference_sim: Path,
    owens_model_dir: Path,
    airfoil_file: Path,
    output_csv: Path,
    output_manifest: Path,
    wohler_exponent: float = 10.0,
    equivalent_cycles: float = 600.0,
    minimum_duration_s: float = 600.0,
    expected_sample_count: int = 80_000,
    expected_start_s: float = 60.0,
    expected_timestep_s: float = 0.0075,
    expected_rpm: float = 1.0,
    rpm_tolerance: float = 0.01,
    pitch_axis_fraction: float = 0.25,
    circle_points: int = 48,
    allow_short: bool = False,
) -> dict[str, object]:
    baseline_results = Path(baseline_results).resolve()
    vorlap_results = Path(vorlap_results).resolve()
    monitors, monitor_metadata = build_monitor_plan(Path(reference_sim).resolve())
    blade_file = Path(str(monitor_metadata["blade_file"])).resolve()
    validate_pitch_axis(blade_file, pitch_axis_fraction)
    elements = read_owens_model(Path(owens_model_dir).resolve())
    bindings = bind_monitors_to_owens(monitors, elements)
    airfoil_coordinates = read_airfoil_coordinates(Path(airfoil_file).resolve())

    scenario_fields: dict[str, list[list[dict[str, float | int | str]]]] = {}
    scenario_reports: dict[str, dict[str, object]] = {}
    reference_time: np.ndarray | None = None
    for scenario, result_path in (
        ("baseline", baseline_results),
        ("vorlap", vorlap_results),
    ):
        data = read_qblade_results(result_path)
        history_report = validate_dense_result_history(
            data,
            monitors,
            expected_sample_count=expected_sample_count,
            expected_start_s=expected_start_s,
            expected_timestep_s=expected_timestep_s,
            expected_rpm=expected_rpm,
            rpm_tolerance=rpm_tolerance,
            minimum_duration_s=minimum_duration_s,
            allow_short=allow_short,
        )
        time_s = np.asarray(data["Time~[s]"], dtype=float)
        if reference_time is None:
            reference_time = time_s.copy()
        elif not np.array_equal(reference_time, time_s):
            raise ValueError("Baseline and VorLap result time vectors are not identical")
        surface, fatigue_report = compute_scenario_surface_dels(
            data,
            bindings,
            airfoil_coordinates,
            pitch_axis_fraction=pitch_axis_fraction,
            circle_points=circle_points,
            wohler_exponent=wohler_exponent,
            equivalent_cycles=equivalent_cycles,
        )
        scenario_fields[scenario] = surface
        scenario_reports[scenario] = {
            "result_file": str(result_path),
            "result_file_sha256": sha256_file(result_path),
            **history_report,
            **fatigue_report,
        }
        del data

    rows, field_report = combine_scenario_fields(
        bindings,
        scenario_fields["baseline"],
        scenario_fields["vorlap"],
    )
    write_field_csv(output_csv, rows)
    finite_ratios = np.asarray(
        [float(row["ratio_vorlap_to_baseline"]) for row in rows], dtype=float
    )
    finite_ratios = finite_ratios[np.isfinite(finite_ratios)]
    if finite_ratios.size == 0:
        raise ValueError("No finite surface DEL ratios were recovered")
    component_rows: dict[str, int] = {}
    for row in rows:
        component = str(row["component"])
        component_rows[component] = component_rows.get(component, 0) + 1
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "quantity": "axial surface normal-strain damage-equivalent range",
        "ratio": "VorLap strain DEL / baseline strain DEL",
        "interpretation": (
            "Beam-section response metric; not stress, fatigue damage, fatigue life, "
            "or structural utilization"
        ),
        "strain_recovery": {
            "generalized_load_order": ["Fz_QBlade", "0", "0", "0", "Mx_QBlade", "My_QBlade"],
            "axis_mapping": (
                "QBlade local Z -> GXBeam x; QBlade local X -> GXBeam y; "
                "QBlade local Y -> GXBeam z"
            ),
            "equation": "epsilon_x(y,z) = e0 - kappa_y*z + kappa_z*y",
            "stiffness_source": str(Path(owens_model_dir).resolve() / "elements.csv"),
        },
        "fatigue": {
            "rainflow": "ASTM-style four-point range counting shared with parked campaign postprocessor",
            "wohler_exponent": wohler_exponent,
            "equivalent_cycles": equivalent_cycles,
            "minimum_duration_s": minimum_duration_s,
            "expected_sample_count": expected_sample_count,
            "expected_start_s": expected_start_s,
            "expected_timestep_s": expected_timestep_s,
            "expected_rpm": expected_rpm,
            "rpm_tolerance": rpm_tolerance,
            "short_run_override": allow_short,
        },
        "surface_discretization": {
            "airfoil_file": str(Path(airfoil_file).resolve()),
            "airfoil_file_sha256": sha256_file(Path(airfoil_file).resolve()),
            "airfoil_perimeter_points": int(airfoil_coordinates.shape[0]),
            "airfoil_pitch_axis_fraction": pitch_axis_fraction,
            "qblade_blade_file": str(blade_file),
            "qblade_blade_file_sha256": sha256_file(blade_file),
            "circular_perimeter_points": circle_points,
            "monitor_count": len(monitors),
            "component_monitor_counts": monitor_metadata["component_monitor_counts"],
            "surface_field_rows_by_component": component_rows,
        },
        "inputs": {
            "reference_sim": str(Path(reference_sim).resolve()),
            "reference_sim_sha256": sha256_file(Path(reference_sim).resolve()),
            "owens_model_dir": str(Path(owens_model_dir).resolve()),
            "baseline_results": scenario_reports["baseline"],
            "vorlap_results": scenario_reports["vorlap"],
        },
        "output": {
            "field_csv": str(Path(output_csv).resolve()),
            "field_csv_sha256": sha256_file(Path(output_csv).resolve()),
            **field_report,
            "finite_ratio_min": float(np.min(finite_ratios)),
            "finite_ratio_max": float(np.max(finite_ratios)),
            "finite_ratio_p01": float(np.quantile(finite_ratios, 0.01)),
            "finite_ratio_median": float(np.median(finite_ratios)),
            "finite_ratio_p99": float(np.quantile(finite_ratios, 0.99)),
        },
    }
    write_manifest(output_manifest, manifest)
    return manifest


__all__ = [
    "AIRFOIL_PERIMETER_KIND",
    "CIRCLE_PERIMETER_KIND",
    "FIELD_COLUMNS",
    "MonitorBinding",
    "OwensElement",
    "SCHEMA",
    "SurfacePoint",
    "axial_surface_normal_strain_history",
    "bind_monitors_to_owens",
    "build_surface_strain_del_field",
    "combine_scenario_fields",
    "compute_scenario_surface_dels",
    "equivalent_load_with_cycles_fast",
    "qblade_column",
    "read_airfoil_coordinates",
    "read_owens_model",
    "read_reference_pitch_axes",
    "read_qblade_results",
    "recover_generalized_strain",
    "sampled_duration_seconds",
    "surface_points_for_binding",
    "turning_points_vectorized",
    "validate_pitch_axis",
    "validate_dense_result_history",
    "write_field_csv",
]
