"""Generate dense QBlade result channels for full-structure strain DEL recovery.

The ordinary parked-fatigue filter stores four load stations on Blade 1.  That
is sufficient for the original section-resultant comparison, but it cannot
support a field on every blade, strut, and central-support segment.  This
module builds one output station at the midpoint of every physical section
interval used by the QBlade-to-OWENS conversion for the blades, struts, upper
tower, and torque tube.  The lower substructure tower is sampled at no more
than one-metre spacing because its two long QBlade members would otherwise
appear as only two broad color bands.  It stores the three local resultants
required to recover axial surface strain,

    epsilon = Fz / EA - Mx * y / EIx + My * x / EIy,

where ``x`` and ``y`` are measured from the section elastic center.  The
resulting strain time history can be rainflow-counted at many perimeter points
to obtain a genuinely chordwise-varying DEL ratio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Sequence

from .qblade_files import read_qblade_keyword, resolve_qblade_path


SCHEMA = "vorlap-qblade-full-structure-strain-output-v2"
OUTPUT_MODE = "full-visible-structure-surface-normal-strain"
SUBSTRUCTURE_MAX_MONITOR_SPACING_M = 1.0
_OUTPUT_FLAGS_END = "LAC_OUT"
_EXTERNAL_INPUT_START = "EXTERNAL_1_IN"
_COMMENT_PREFIXES = ("#", "!")


@dataclass(frozen=True)
class MonitorPoint:
    component: str
    family: str
    member_index: int
    position: float
    section_file: str
    result_component: str | None = None

    @property
    def output_token(self) -> str:
        return f"{self.component}_{self.position:.9f}"

    @property
    def result_position(self) -> str:
        return f"{self.position:.3f}"

    def result_channels(self) -> tuple[str, str, str]:
        position = self.result_position
        component = self.result_component or self.component
        return (
            f"Z_l For. {component} pos {position} [N]",
            f"X_l Mom. {component} pos {position} [Nm]",
            f"Y_l Mom. {component} pos {position} [Nm]",
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _clean(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _require_keyword(path: Path, keyword: str) -> str:
    value = read_qblade_keyword(path, keyword)
    if value is None or not value.strip():
        raise ValueError(f"Could not find {keyword} in {path}")
    return value.strip()


def _float_keyword(path: Path, keyword: str) -> float:
    return float(_require_keyword(path, keyword))


def _int_keyword(path: Path, keyword: str) -> int:
    return int(float(_require_keyword(path, keyword)))


def _unique(values: Iterable[float], tolerance: float = 1.0e-10) -> list[float]:
    result: list[float] = []
    for raw in sorted(float(value) for value in values):
        value = min(1.0, max(0.0, raw))
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def _midpoints(fractions: Sequence[float], *, label: str) -> list[float]:
    if len(fractions) < 2:
        raise ValueError(f"{label} must contain at least two structural fractions")
    if not math.isclose(fractions[0], 0.0, abs_tol=1.0e-9):
        raise ValueError(f"{label} does not start at normalized position zero")
    if not math.isclose(fractions[-1], 1.0, abs_tol=1.0e-9):
        raise ValueError(f"{label} does not end at normalized position one")
    return [(first + second) / 2.0 for first, second in zip(fractions, fractions[1:])]


def _section_fractions(path: Path) -> list[float]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        header = next(index for index, line in enumerate(lines) if "MASSD_[kg/m]" in line)
    except StopIteration as exc:
        raise ValueError(f"No structural property table found in {path}") from exc
    fractions: list[float] = []
    started = False
    for raw in lines[header + 1 :]:
        body = _clean(raw)
        if not body:
            if started:
                break
            continue
        try:
            values = [float(token) for token in body.split()]
        except ValueError:
            if started:
                break
            continue
        if len(values) < 19:
            if started:
                break
            continue
        fractions.append(values[0])
        started = True
    if len(fractions) < 2:
        raise ValueError(f"Incomplete structural property table in {path}")
    if any(second <= first for first, second in zip(fractions, fractions[1:])):
        raise ValueError(f"Structural fractions are not strictly increasing in {path}")
    return fractions


def _added_mass_fractions(path: Path) -> list[float]:
    pattern = re.compile(r"^\s*ADDMASS_([-+0-9.eE]+)(?:\s|$)")
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        match = pattern.match(_clean(line))
        if match:
            values.append(float(match.group(1)))
    return values


def _blade_geometry(path: Path) -> tuple[list[float], list[dict[str, float]]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        header = next(index for index, line in enumerate(lines) if "HEIGHT_[m]" in line)
    except StopIteration as exc:
        raise ValueError(f"No blade geometry table found in {path}") from exc
    heights: list[float] = []
    for raw in lines[header + 1 :]:
        body = _clean(raw)
        if not body:
            if heights:
                break
            continue
        tokens = body.split()
        try:
            values = [float(token) for token in tokens[:7]]
        except ValueError:
            break
        if len(values) < 7:
            break
        heights.append(values[0])
    if len(heights) < 2 or heights[-1] <= heights[0]:
        raise ValueError(f"Incomplete blade geometry table in {path}")

    struts: list[dict[str, float]] = []
    start_pattern = re.compile(r"^STRUT_(\d+)$")
    for index, raw in enumerate(lines):
        match = start_pattern.match(_clean(raw))
        if not match:
            continue
        strut_id = int(match.group(1))
        end_marker = f"END_STRUT_{strut_id}"
        try:
            end = next(
                row for row in range(index + 1, len(lines)) if _clean(lines[row]) == end_marker
            )
        except StopIteration as exc:
            raise ValueError(f"Missing {end_marker} in {path}") from exc
        values: dict[str, float] = {"id": float(strut_id)}
        for line in lines[index + 1 : end]:
            body = _clean(line)
            tokens = body.split()
            if len(tokens) < 2:
                continue
            try:
                number = float(tokens[0])
            except ValueError:
                continue
            values[tokens[1]] = number
        for keyword in ("HGTBLD_STR", "HGTHUB_STR"):
            if keyword not in values:
                raise ValueError(f"Strut {strut_id} has no {keyword} in {path}")
        struts.append(values)
    if not struts:
        raise ValueError(f"No aerodynamic struts found in {path}")
    return heights, struts


def _numeric_table(path: Path, marker: str, minimum_columns: int) -> list[list[float]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        start = next(
            index
            for index, line in enumerate(lines)
            if _clean(line).split() and _clean(line).split()[0] == marker
        )
    except StopIteration as exc:
        raise ValueError(f"Could not find {marker} in {path}") from exc
    rows: list[list[float]] = []
    started = False
    for raw in lines[start + 1 :]:
        body = _clean(raw)
        if not body:
            if started:
                break
            continue
        values: list[float] = []
        for token in body.split():
            try:
                values.append(float(token))
            except ValueError:
                break
        if len(values) < minimum_columns:
            if started:
                break
            continue
        rows.append(values)
        started = True
    if not rows:
        raise ValueError(f"No numeric rows found after {marker} in {path}")
    return rows


def _substructure_member_geometry(path: Path) -> list[dict[str, float | str]]:
    joints = {
        int(row[0]): row[1:4]
        for row in _numeric_table(path, "SUBJOINTS", 4)
    }
    member_result_names: dict[int, str] = {}
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if _clean(line).split() and _clean(line).split()[0] == "SUBMEMBERS"
    )
    for raw in lines[start + 1 :]:
        tokens = _clean(raw).split()
        numeric: list[float] = []
        for token in tokens[:10]:
            try:
                numeric.append(float(token))
            except ValueError:
                break
        if len(numeric) == 10:
            member_id = int(numeric[0])
            # QBlade 2.0.9.7 uses the first whitespace-delimited member-name
            # token in exported variable names, even though sensor input uses ID.
            member_result_names[member_id] = (
                f"SUB_{tokens[10]}" if len(tokens) > 10 else f"SUB_{member_id}"
            )

    members: list[dict[str, float | str]] = []
    for row in _numeric_table(path, "SUBMEMBERS", 10):
        member_id = int(row[0])
        joint_a = int(row[1])
        joint_b = int(row[2])
        if joint_a not in joints or joint_b not in joints:
            raise ValueError(
                f"SUBMEMBER {member_id} references an undefined joint in {path}"
            )
        delta = [second - first for first, second in zip(joints[joint_a], joints[joint_b])]
        length = math.sqrt(sum(value * value for value in delta))
        if not length > 0.0:
            raise ValueError(f"SUBMEMBER {member_id} has zero length in {path}")
        members.append(
            {
                "id": float(member_id),
                "joint_a": float(joint_a),
                "joint_b": float(joint_b),
                "length_m": length,
                "result_component": member_result_names[member_id],
            }
        )
    return members


def _component_files(sim_path: Path) -> dict[str, object]:
    turbine_file = resolve_qblade_path(sim_path, _require_keyword(sim_path, "TURBFILE"))
    structural_file = resolve_qblade_path(
        turbine_file, _require_keyword(turbine_file, "STRUCTURALFILE")
    )
    blade_file = resolve_qblade_path(turbine_file, _require_keyword(turbine_file, "BLADEFILE"))
    number_blades = _int_keyword(structural_file, "NUMBLD")
    blade_sections = [
        resolve_qblade_path(
            structural_file,
            _require_keyword(structural_file, f"BLDFILE_{index}"),
        )
        for index in range(1, number_blades + 1)
    ]
    strut_sections: list[Path] = []
    index = 1
    while True:
        value = read_qblade_keyword(structural_file, f"STRTFILE_{index}")
        if value is None or not value.strip():
            break
        strut_sections.append(resolve_qblade_path(structural_file, value.strip()))
        index += 1
    tower_section = resolve_qblade_path(
        structural_file, _require_keyword(structural_file, "TWRFILE")
    )
    torque_section = resolve_qblade_path(
        structural_file, _require_keyword(structural_file, "TRQTBFILE")
    )
    substructure_file = resolve_qblade_path(
        structural_file, _require_keyword(structural_file, "SUBFILE")
    )
    return {
        "sim": sim_path.resolve(),
        "turbine": turbine_file.resolve(),
        "structural": structural_file.resolve(),
        "blade": blade_file.resolve(),
        "number_blades": number_blades,
        "blade_sections": blade_sections,
        "strut_sections": strut_sections,
        "tower_section": tower_section,
        "torque_section": torque_section,
        "substructure": substructure_file,
    }


def build_monitor_plan(sim_path: Path) -> tuple[list[MonitorPoint], dict[str, object]]:
    """Return one monitor at every converted physical-element midpoint."""
    files = _component_files(Path(sim_path).resolve())
    structural_file = Path(files["structural"])
    blade_file = Path(files["blade"])
    blade_heights, strut_geometry = _blade_geometry(blade_file)
    blade_height = blade_heights[-1] - blade_heights[0]
    if len(files["strut_sections"]) != len(strut_geometry):
        raise ValueError(
            "The structural and aerodynamic strut counts differ: "
            f"{len(files['strut_sections'])} versus {len(strut_geometry)}"
        )

    monitors: list[MonitorPoint] = []
    counts: dict[str, int] = {}
    blade_connection_fractions = [
        (row["HGTBLD_STR"] - blade_heights[0]) / blade_height for row in strut_geometry
    ]
    for blade_index, section_file in enumerate(files["blade_sections"], start=1):
        section_path = Path(section_file)
        fractions = _unique(
            _section_fractions(section_path)
            + _added_mass_fractions(section_path)
            + blade_connection_fractions
        )
        positions = _midpoints(fractions, label=f"BLD_{blade_index}")
        component = f"BLD_{blade_index}"
        counts[component] = len(positions)
        monitors.extend(
            MonitorPoint(component, "blade", blade_index, position, str(section_path))
            for position in positions
        )

    for blade_index in range(1, int(files["number_blades"]) + 1):
        for strut_index, section_file in enumerate(files["strut_sections"], start=1):
            section_path = Path(section_file)
            fractions = _unique(
                _section_fractions(section_path) + _added_mass_fractions(section_path)
            )
            component = f"STR_{strut_index}_{blade_index}"
            positions = _midpoints(fractions, label=component)
            counts[component] = len(positions)
            monitors.extend(
                MonitorPoint(component, "strut", strut_index, position, str(section_path))
                for position in positions
            )

    tower_section = Path(files["tower_section"])
    tower_height = _float_keyword(structural_file, "TWRHEIGHT")
    torque_clearance = _float_keyword(structural_file, "TRQTBCLEAR")
    bearing_height = torque_clearance + _float_keyword(structural_file, "TRQTBCONN")
    extra_heights = [
        _float_keyword(structural_file, "HUBPOS"),
        bearing_height,
        *(
            _float_keyword(structural_file, "RTRCLEAR") + row["HGTHUB_STR"]
            for row in strut_geometry
        ),
    ]
    disc_value = read_qblade_keyword(tower_section, "DISC")
    disc = max(2, 0 if disc_value is None else int(float(disc_value)))
    tower_fractions = _unique(
        [index / (disc - 1) for index in range(disc)]
        + _section_fractions(tower_section)
        + _added_mass_fractions(tower_section)
        + [
            height / tower_height
            for height in extra_heights
            if -1.0e-9 <= height <= tower_height + 1.0e-9
        ]
    )
    tower_positions = _midpoints(tower_fractions, label="TWR")
    counts["TWR"] = len(tower_positions)
    monitors.extend(
        MonitorPoint("TWR", "tower", 1, position, str(tower_section))
        for position in tower_positions
    )

    torque_section = Path(files["torque_section"])
    torque_height = _float_keyword(structural_file, "TRQTBHEIGHT")
    torque_disc_value = read_qblade_keyword(torque_section, "DISC")
    torque_disc = max(
        2,
        0 if torque_disc_value is None else int(float(torque_disc_value)),
    )
    torque_fractions = _unique(
        [index / (torque_disc - 1) for index in range(torque_disc)]
        + _section_fractions(torque_section)
        + _added_mass_fractions(torque_section)
        + [
            (height - torque_clearance) / torque_height
            for height in extra_heights
            if torque_clearance - 1.0e-9
            <= height
            <= torque_clearance + torque_height + 1.0e-9
        ]
    )
    torque_positions = _midpoints(torque_fractions, label="TRQ")
    counts["TRQ"] = len(torque_positions)
    monitors.extend(
        MonitorPoint("TRQ", "torque_tube", 1, position, str(torque_section))
        for position in torque_positions
    )

    substructure_file = Path(files["substructure"])
    for member in _substructure_member_geometry(substructure_file):
        member_index = int(member["id"])
        interval_count = max(
            1,
            math.ceil(member["length_m"] / SUBSTRUCTURE_MAX_MONITOR_SPACING_M),
        )
        positions = [
            (index + 0.5) / interval_count for index in range(interval_count)
        ]
        component = f"SUB_{member_index}"
        counts[component] = len(positions)
        monitors.extend(
            MonitorPoint(
                component,
                "substructure",
                member_index,
                position,
                str(substructure_file),
                str(member["result_component"]),
            )
            for position in positions
        )

    rounded_keys = [(monitor.component, monitor.result_position) for monitor in monitors]
    if len(rounded_keys) != len(set(rounded_keys)):
        duplicates = sorted(key for key in set(rounded_keys) if rounded_keys.count(key) > 1)
        raise ValueError(
            "Two physical-element midpoints collapse to the same QBlade three-decimal "
            f"result label: {duplicates}"
        )
    metadata = {
        "schema": SCHEMA,
        "output_mode": OUTPUT_MODE,
        "sim": str(files["sim"]),
        "turbine_file": str(files["turbine"]),
        "structural_file": str(files["structural"]),
        "blade_file": str(files["blade"]),
        "monitor_count": len(monitors),
        "component_monitor_counts": counts,
        "resultant_channels_per_monitor": 3,
        "strain_recovery": {
            "quantity": "axial surface strain",
            "formula": "epsilon = Fz/EA - Mx*y/EIx + My*x/EIy",
            "coordinates": "QBlade local section x/y measured from the elastic center",
            "stored_resultants": ["Z_l force", "X_l moment", "Y_l moment"],
            "substructure_max_monitor_spacing_m": SUBSTRUCTURE_MAX_MONITOR_SPACING_M,
        },
        "source_hashes": {
            str(path): _sha256(Path(path))
            for path in (
                files["sim"],
                files["turbine"],
                files["structural"],
                files["blade"],
                *files["blade_sections"],
                *files["strut_sections"],
                files["tower_section"],
                files["torque_section"],
                files["substructure"],
            )
        },
        "monitors": [asdict(monitor) for monitor in monitors],
    }
    return monitors, metadata


def filter_channels(monitors: Sequence[MonitorPoint]) -> list[str]:
    channels = ["Time [s]", "LSS Rpm [rpm]"]
    for monitor in monitors:
        channels.extend(monitor.result_channels())
    if len(channels) != len(set(channels)):
        raise ValueError("Generated full-structure result filter contains duplicate channels")
    return channels


def write_full_structure_filter(
    sim_path: Path,
    filter_path: Path,
    *,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    monitors, metadata = build_monitor_plan(sim_path)
    channels = filter_channels(monitors)
    _atomic_write_text(filter_path, "\n".join(channels) + "\n")
    metadata = {
        **metadata,
        "result_filter": str(filter_path.resolve()),
        "result_filter_sha256": _sha256(filter_path),
        "result_filter_channel_count": len(channels),
    }
    if metadata_path is not None:
        _atomic_write_text(
            metadata_path,
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )
    return metadata


def _filter_file_channels(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(_COMMENT_PREFIXES)
    ]


def configure_full_structure_outputs(
    sim_path: Path,
    result_filter: Path,
    *,
    metadata_path: Path | None = None,
) -> dict[str, object]:
    """Patch one generated case and verify its result filter matches the plan."""
    monitors, metadata = build_monitor_plan(sim_path)
    expected_channels = filter_channels(monitors)
    observed_channels = _filter_file_channels(result_filter)
    if observed_channels != expected_channels:
        raise ValueError(
            f"Generated case monitor plan does not match {result_filter}; "
            f"expected {len(expected_channels)} ordered channels and found "
            f"{len(observed_channels)}"
        )
    structural_file = Path(metadata["structural_file"])
    lines = structural_file.read_text(encoding="utf-8", errors="strict").splitlines()
    try:
        flags_end = next(index for index, line in enumerate(lines) if _OUTPUT_FLAGS_END in line)
        external_start = next(
            index
            for index in range(flags_end + 1, len(lines))
            if _clean(lines[index]) == _EXTERNAL_INPUT_START
        )
    except StopIteration as exc:
        raise ValueError(
            f"Could not locate the QBlade structural output block in {structural_file}"
        ) from exc
    structural_monitors = [
        monitor for monitor in monitors if monitor.family != "substructure"
    ]
    replacement = ["", *(monitor.output_token for monitor in structural_monitors), ""]
    updated = lines[: flags_end + 1] + replacement + lines[external_start:]
    _atomic_write_text(structural_file, "\n".join(updated) + "\n")

    substructure_file = Path(
        next(
            monitor.section_file
            for monitor in monitors
            if monitor.family == "substructure"
        )
    )
    sub_lines = substructure_file.read_text(
        encoding="utf-8", errors="strict"
    ).splitlines()
    generated_sub_token = re.compile(r"^SUB_\d+_[-+0-9.eE]+$")
    sub_lines = [
        line for line in sub_lines if not generated_sub_token.match(_clean(line))
    ]
    sub_tokens = [
        monitor.output_token
        for monitor in monitors
        if monitor.family == "substructure"
    ]
    while sub_lines and not _clean(sub_lines[-1]):
        sub_lines.pop()
    # QBlade must have parsed SUBMEMBERS before it can resolve SUB_<MemID> sensors.
    # Append member sensors after all substructure tables, matching the official
    # exemplar. Joint sensors can be declared earlier, but member sensors placed
    # immediately after SUBJOINTS are silently ignored by QBlade 2.0.9.7.
    updated_sub = [*sub_lines, "", *sub_tokens, ""]
    _atomic_write_text(substructure_file, "\n".join(updated_sub) + "\n")
    metadata = {
        **metadata,
        "scope": "generated_case_only",
        "result_filter": str(result_filter.resolve()),
        "result_filter_sha256": _sha256(result_filter),
        "result_filter_channel_count": len(expected_channels),
        "patched_structural_file_sha256": _sha256(structural_file),
        "patched_substructure_file": str(substructure_file),
        "patched_substructure_file_sha256": _sha256(substructure_file),
        "external_swap_block_modified": False,
    }
    if metadata_path is not None:
        _atomic_write_text(
            metadata_path,
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )
    return metadata


__all__ = [
    "MonitorPoint",
    "OUTPUT_MODE",
    "SCHEMA",
    "SUBSTRUCTURE_MAX_MONITOR_SPACING_M",
    "build_monitor_plan",
    "configure_full_structure_outputs",
    "filter_channels",
    "write_full_structure_filter",
]
