#!/usr/bin/env python3
"""Generate and audit the frozen NACA0018 prescribed-motion Nalu campaign.

The campaign is deliberately split into four Flight stages:

1. short, full-node scaling probes at 2/4/8 nodes;
2. one zero-amplitude ALE precursor at each Reynolds number;
3. one complete moving-mesh case used as the motion/history preflight; and
4. the remaining prescribed-motion matrix.

No command in this module invokes Nalu-Wind.  The Slurm entry points own that
operation; this module generates immutable inputs, validates them, converts
completed surface-force output to the canonical paired-history contract, and
enforces the measured-runtime gates.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REQUEST_PATH = HERE / "work_map_data_request.json"
NALU_ASSET_DIR = HERE / "nalu"
TEMPLATE_PATH = NALU_ASSET_DIR / "template.yaml"
HYPRE_PATH = NALU_ASSET_DIR / "hypre_file.yaml"

CAMPAIGN_SCHEMA = "vorlap-naca0018-nalu-prescribed-motion-campaign-v1"
RESULTS_SCHEMA = "vorlap-naca0018-nalu-prescribed-motion-results-v1"
SCALING_SCHEMA = "vorlap-naca0018-nalu-scaling-report-v1"
MESH_AUDIT_SCHEMA = "vorlap-naca0018-exodus-geometry-audit-v1"
HISTORY_COLUMNS = (
    "time_s",
    "displacement_m",
    "velocity_mps",
    "crossflow_force_n",
)

EXPECTED_REYNOLDS = (64_000.0, 100_000.0, 140_000.0)
EXPECTED_K = (0.45, 0.65, 0.8, 1.05)
EXPECTED_NONZERO_AMPLITUDE_RATIOS = (0.005, 0.0125, 0.025, 0.05)
EXPECTED_INCIDENCE_DEG = -90.0
EXPECTED_DIRECTION_DEG = 0.0
PRECURSOR_PHASE_FACTOR = 20.0
TARGET_ST_MAX = 0.4
TARGET_SAMPLES_AT_ST_MAX = 120
DISCARD_CYCLES = 5
RETAINED_CYCLES = 8
TOTAL_CYCLES = DISCARD_CYCLES + RETAINED_CYCLES
SCALING_PROBE_UPDATES = 64
SCALING_REYNOLDS = 64_000.0
SUPPORTED_NODE_COUNTS = (2, 4, 8)
PHYSICAL_CORES_PER_FLIGHT_NODE = 112
BASELINE_NODES = 2
TARGET_WALLTIME_HOURS = 2.0
MAX_WALLTIME_HOURS = 8.0
SCALING_RATE_SAFETY_FACTOR = 1.25
ARCHIVED_TWO_NODE_STEP_SECONDS = 19.6343
ARCHIVED_TWO_NODE_STEP_RANGE_SECONDS = (17.2712, 21.81)
ARCHIVED_RATE_SOURCE = (
    "airfoil_vortexshedding_mesh_run_post_nalu/nalu_runs/failedsim/"
    "log.out and log_failed.out; Nalu-Wind v2.2.1, 224 ranks, "
    "four early completed steps"
)
EXPECTED_NALU_GIT_COMMIT = "ffa9de729df2a11b5241fdeb7628e7fab9f48f9b"

WALLCLOCK_RE = re.compile(
    r"WallClockTime:\s+(?P<step>\d+).*?Total:\s*(?P<seconds>[0-9.eE+\-]+)"
)
STEP_RE = re.compile(r"Time Step Count:\s*(\d+)\s+Current Time:\s*([0-9.eE+\-]+)")
TERMINAL_COMPLETION_RE = re.compile(
    r"Simulation Shall Complete:\s*time/timestep:\s*"
    r"(?P<time>[0-9.eE+\-]+)\s*/\s*(?P<step>\d+)"
)
NALU_VERSION_RE = re.compile(r"Nalu-Wind Version:\s*(\S+)")
NALU_COMMIT_RE = re.compile(r"Nalu-Wind GIT Commit SHA:\s*(\S+)")
FAILURE_PATTERNS = (
    re.compile(r"\bMPI_ABORT\b", re.IGNORECASE),
    re.compile(r"terminate called", re.IGNORECASE),
    re.compile(r"what\(\):", re.IGNORECASE),
    re.compile(r"(^|\s)ERROR:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"segmentation fault", re.IGNORECASE),
)


class CampaignError(RuntimeError):
    """Raised when generation or a fail-closed campaign check fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nalu_executable_identity(path: Path) -> dict[str, str]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CampaignError(f"Cannot resolve the Nalu executable: {path}") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise CampaignError(
            f"Nalu executable is not a regular executable file: {resolved}"
        )
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _recorded_nalu_executable_identity(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise CampaignError("A scaling report has a malformed naluX identity.")
    path = value.get("path")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or os.path.normpath(path) != path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CampaignError("A scaling report has a malformed naluX identity.")
    return {"path": path, "sha256": digest}


def _validated_log_runtime(
    log: Path,
    *,
    expected_solver: Mapping[str, Any],
    executable: Mapping[str, Any],
    root: Path,
    expected_terminal_step: int,
) -> dict[str, Any]:
    if log.is_symlink() or not log.is_file():
        raise CampaignError(f"Completed Nalu run lacks a regular log: {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    versions = set(NALU_VERSION_RE.findall(text))
    commits = set(NALU_COMMIT_RE.findall(text))
    if versions != {expected_solver["version"]}:
        raise CampaignError(
            f"Completed log does not prove exact Nalu-Wind v2.2.1: {log}"
        )
    if commits != {expected_solver["expected_git_commit"]}:
        raise CampaignError(
            f"Completed log does not prove the frozen Nalu-Wind commit: {log}"
        )
    terminal_step = _terminal_completion_step(log)
    if terminal_step != int(expected_terminal_step):
        raise CampaignError(
            f"Completed log lacks the exact terminal Nalu step {expected_terminal_step}: {log}"
        )
    return {
        "status": "verified",
        "nalu_executable": dict(executable),
        "observed_solver_version": next(iter(versions)),
        "observed_solver_git_commit": next(iter(commits)),
        "terminal_step": terminal_step,
        "log": {"path": _relative(log, root), "sha256": _sha256(log)},
    }


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve_campaign_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=2.0e-13, abs_tol=1.0e-14
    ):
        raise CampaignError(
            f"{label} changed: expected {expected!r}, found {actual!r}."
        )


def _load_request() -> tuple[dict[str, Any], str]:
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    if request.get("schema") != "vorlap-naca0018-prescribed-motion-data-request-v1":
        raise CampaignError("The prescribed-motion request schema is not frozen v1.")
    axes = request["frozen_axes"]
    if tuple(map(float, axes["reynolds_number"])) != EXPECTED_REYNOLDS:
        raise CampaignError(
            "The requested Reynolds grid differs from the executable freeze."
        )
    if tuple(map(float, axes["reduced_frequency_k"])) != EXPECTED_K:
        raise CampaignError(
            "The requested reduced-frequency grid differs from the executable freeze."
        )
    if tuple(map(float, axes["incidence_deg"])) != (EXPECTED_INCIDENCE_DEG,):
        raise CampaignError(
            "The Nalu workflow requires the frozen -90 degree mesh incidence."
        )
    if tuple(map(float, axes["direction_deg_mod_180"])) != (EXPECTED_DIRECTION_DEG,):
        raise CampaignError(
            "The Nalu workflow requires the frozen cross-flow direction."
        )
    amplitudes = tuple(map(float, axes["amplitude_ratio"]))
    if amplitudes != (0.0,) + EXPECTED_NONZERO_AMPLITUDE_RATIOS:
        raise CampaignError(
            "The requested amplitude grid differs from the executable freeze."
        )
    minimum = request["minimum_accepted_history"]
    if int(minimum["discard_cycles"]) != DISCARD_CYCLES:
        raise CampaignError("The request no longer discards exactly five cycles.")
    if int(minimum["complete_cycles_after_discard"]) != RETAINED_CYCLES:
        raise CampaignError("The request no longer retains exactly eight cycles.")
    if int(minimum["samples_per_cycle"]) > TARGET_SAMPLES_AT_ST_MAX:
        raise CampaignError(
            "The requested sampling minimum exceeds the frozen Nalu grid."
        )
    flight = request["flight_parallel_plan"]
    if (
        flight.get("preferred_target_walltime") != "02:00:00"
        or flight.get("maximum_requested_walltime") != "08:00:00"
        or flight.get("node_selection_policy")
        != "smallest_2h_passing_size_among_2_4_8_else_smallest_8h_passing_size"
    ):
        raise CampaignError("The measured Nalu node-selection policy differs.")
    fluid = request["frozen_fluid_and_section"]
    _assert_close(float(fluid["reference_chord_m"]), 0.23, "reference chord")
    _assert_close(float(fluid["section_span_m"]), 4.0, "section span")
    return request, _sha256(REQUEST_PATH)


def _load_template() -> dict[str, Any]:
    value = yaml.safe_load(TEMPLATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "realms" not in value:
        raise CampaignError(f"Invalid Nalu template: {TEMPLATE_PATH}")
    return value


def _tag(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace("-", "m").replace(".", "p")


def _precursor_id(reynolds: float) -> str:
    return f"precursor_re{int(round(reynolds)):06d}"


def _case_id(reynolds: float, reduced_frequency: float, amplitude_ratio: float) -> str:
    return (
        f"re{int(round(reynolds)):06d}_k{_tag(reduced_frequency, 2)}_"
        f"a{_tag(amplitude_ratio, 4)}"
    )


def _physics(request: Mapping[str, Any]) -> dict[str, float]:
    fluid = request["frozen_fluid_and_section"]
    return {
        "fluid_density_kgpm3": float(fluid["fluid_density_kgpm3"]),
        "fluid_dynamic_viscosity_pas": float(fluid["fluid_dynamic_viscosity_pas"]),
        "reference_chord_m": float(fluid["reference_chord_m"]),
        "section_span_m": float(fluid["section_span_m"]),
    }


def _flow_values(
    reynolds: float, physics: Mapping[str, float]
) -> dict[str, float | int]:
    rho = physics["fluid_density_kgpm3"]
    mu = physics["fluid_dynamic_viscosity_pas"]
    chord = physics["reference_chord_m"]
    speed = reynolds * mu / (rho * chord)
    restart_time = PRECURSOR_PHASE_FACTOR * math.pi * chord / speed
    maximum_dt = chord / (TARGET_SAMPLES_AT_ST_MAX * TARGET_ST_MAX * speed)
    precursor_steps = math.ceil(restart_time / maximum_dt)
    precursor_dt = restart_time / precursor_steps
    return {
        "wind_speed_mps": speed,
        "precursor_restart_time_s": restart_time,
        "precursor_steps": precursor_steps,
        "precursor_time_step_s": precursor_dt,
    }


def _motion_values(
    reynolds: float,
    reduced_frequency: float,
    amplitude_ratio: float,
    physics: Mapping[str, float],
) -> dict[str, float | int]:
    flow = _flow_values(reynolds, physics)
    speed = float(flow["wind_speed_mps"])
    chord = physics["reference_chord_m"]
    frequency = reduced_frequency * speed / (math.pi * chord)
    period = 1.0 / frequency
    samples_per_cycle = math.ceil(
        max(
            100.0,
            TARGET_SAMPLES_AT_ST_MAX * TARGET_ST_MAX * math.pi / reduced_frequency,
        )
    )
    time_step = period / samples_per_cycle
    updates = TOTAL_CYCLES * samples_per_cycle + 1
    restart_time = float(flow["precursor_restart_time_s"])
    exact_phase_cycles = frequency * restart_time
    if not math.isclose(exact_phase_cycles, round(exact_phase_cycles), abs_tol=2.0e-13):
        raise CampaignError(
            f"The start phase for k={reduced_frequency:g} is not an integer cycle."
        )
    return {
        **flow,
        "frequency_hz": frequency,
        "period_s": period,
        "amplitude_m": amplitude_ratio * chord,
        "samples_per_cycle": samples_per_cycle,
        "time_step_s": time_step,
        "updates_after_restart": updates,
        "relative_history_start_s": time_step,
        "relative_history_end_s": updates * time_step,
        "phase_cycles_at_restart": exact_phase_cycles,
    }


def _turbulence(speed: float, chord: float) -> tuple[float, float]:
    intensity = 0.01
    cmu = 0.09
    length_scale = 0.07 * chord
    turbulent_ke = 1.5 * (speed * intensity) ** 2
    specific_dissipation = math.sqrt(turbulent_ke) / (cmu**0.25 * length_scale)
    return turbulent_ke, specific_dissipation


def _patch_common_deck(
    template: Mapping[str, Any],
    *,
    mesh: Path,
    speed: float,
    time_step: float,
    termination_step_count: int,
    physics: Mapping[str, float],
    output_file: Path,
    force_file: Path,
    restart_file: Path,
    output_frequency: int,
    restart_frequency: int,
    restart_start: int,
) -> dict[str, Any]:
    deck = copy.deepcopy(template)
    integrator = deck["Time_Integrators"][0]["StandardTimeIntegrator"]
    integrator["start_time"] = 0.0
    integrator["time_step_count"] = 0
    integrator["time_step"] = float(time_step)
    integrator["termination_step_count"] = int(termination_step_count)

    for solver in deck["linear_solvers"]:
        solver["hypre_cfg_file"] = str(HYPRE_PATH.resolve())

    realm = deck["realms"][0]
    realm["mesh"] = str(mesh.resolve())
    turbulent_ke, specific_dissipation = _turbulence(
        speed, physics["reference_chord_m"]
    )
    for bc in realm["boundary_conditions"]:
        if "inflow_user_data" in bc:
            bc["inflow_user_data"]["velocity"] = [speed, 0.0, 0.0]
            bc["inflow_user_data"]["turbulent_ke"] = turbulent_ke
            bc["inflow_user_data"]["specific_dissipation_rate"] = specific_dissipation
        if "open_user_data" in bc:
            bc["open_user_data"]["turbulent_ke"] = turbulent_ke
            bc["open_user_data"]["specific_dissipation_rate"] = specific_dissipation
    initial = realm["initial_conditions"][0]["value"]
    initial["velocity"] = [speed, 0.0, 0.0]
    initial["turbulent_ke"] = turbulent_ke
    initial["specific_dissipation_rate"] = specific_dissipation
    specifications = realm["material_properties"]["specifications"]
    next(row for row in specifications if row["name"] == "density")["value"] = physics[
        "fluid_density_kgpm3"
    ]
    next(row for row in specifications if row["name"] == "viscosity")["value"] = (
        physics["fluid_dynamic_viscosity_pas"]
    )
    realm["output"]["output_data_base_name"] = str(output_file.resolve())
    realm["output"]["output_frequency"] = int(output_frequency)
    realm["post_processing"][0]["output_file_name"] = str(force_file.resolve())
    realm["restart"]["restart_data_base_name"] = str(restart_file.resolve())
    realm["restart"]["restart_frequency"] = int(restart_frequency)
    realm["restart"]["restart_start"] = int(restart_start)
    return deck


def _precursor_deck(
    template: Mapping[str, Any],
    *,
    original_mesh: Path,
    case_dir: Path,
    flow: Mapping[str, float | int],
    physics: Mapping[str, float],
    updates: int | None = None,
) -> dict[str, Any]:
    precursor_steps = int(flow["precursor_steps"])
    step_count = precursor_steps if updates is None else int(updates)
    deck = _patch_common_deck(
        template,
        mesh=original_mesh,
        speed=float(flow["wind_speed_mps"]),
        time_step=float(flow["precursor_time_step_s"]),
        termination_step_count=step_count,
        physics=physics,
        output_file=case_dir / "results" / "flow.e",
        force_file=case_dir / "full_surface_force.dat",
        restart_file=case_dir / "restart" / "precursor.rst",
        output_frequency=step_count,
        restart_frequency=step_count,
        restart_start=step_count,
    )
    realm = deck["realms"][0]
    realm["mesh_transformation"] = [
        {
            "name": "scale_original_naca0018_once",
            "mesh_parts": ["all_blocks"],
            "motion": [
                {
                    "type": "scaling",
                    "factor": [
                        physics["reference_chord_m"],
                        physics["reference_chord_m"],
                        1.0,
                    ],
                    # MotionScalingKernel v2.2.1 reads the key "centroid"
                    # (despite older manual examples that say "origin").
                    "centroid": [0.0, 0.0, 0.0],
                }
            ],
        }
    ]
    # A zero-amplitude ALE precursor deliberately declares the same moving-mesh
    # fields/states as its children.  A truly static precursor does not.
    realm["mesh_motion"] = [
        {
            "name": "stationary_zero_amplitude_ale",
            "mesh_parts": ["all_blocks"],
            "motion": [
                {
                    "type": "oscillation",
                    "start_time": 0.0,
                    "period": physics["reference_chord_m"]
                    / float(flow["wind_speed_mps"]),
                    "amplitude": 0.0,
                    "direction": [0.0, 1.0, 0.0],
                }
            ],
        }
    ]
    return deck


def _motion_deck(
    template: Mapping[str, Any],
    *,
    precursor_restart: Path,
    case_dir: Path,
    values: Mapping[str, float | int],
    physics: Mapping[str, float],
) -> dict[str, Any]:
    precursor_steps = int(values["precursor_steps"])
    samples = int(values["samples_per_cycle"])
    updates = int(values["updates_after_restart"])
    deck = _patch_common_deck(
        template,
        mesh=precursor_restart,
        speed=float(values["wind_speed_mps"]),
        time_step=float(values["time_step_s"]),
        termination_step_count=precursor_steps + updates,
        physics=physics,
        output_file=case_dir / "results" / "flow.e",
        force_file=case_dir / "full_surface_force.dat",
        restart_file=case_dir / "restart" / "motion.rst",
        output_frequency=samples,
        restart_frequency=samples,
        restart_start=precursor_steps,
    )
    realm = deck["realms"][0]
    realm["output"]["output_start"] = precursor_steps + 1
    realm["restart"]["restart_time"] = float(values["precursor_restart_time_s"])
    realm["mesh_motion"] = [
        {
            "name": "prescribed_crossflow_translation",
            "mesh_parts": ["all_blocks"],
            "motion": [
                {
                    "type": "oscillation",
                    "start_time": float(values["precursor_restart_time_s"]),
                    "period": float(values["period_s"]),
                    "amplitude": float(values["amplitude_m"]),
                    "direction": [0.0, 1.0, 0.0],
                }
            ],
        }
    ]
    if "mesh_transformation" in realm:
        raise CampaignError(
            "Motion child unexpectedly retained the precursor mesh scaling."
        )
    return deck


def _file_identity(path: Path, root: Path | None = None) -> dict[str, Any]:
    value = str(path.resolve()) if root is None else _relative(path, root)
    return {"path": value, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _decode_exodus_names(values: Any) -> list[str]:
    import numpy as np

    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    names: list[str] = []
    for row in array:
        raw = np.asarray(row).astype("S1", copy=False).tobytes()
        names.append(raw.decode("utf-8", errors="replace").rstrip("\x00 "))
    return names


def _audit_exodus_variables(mesh: Path, variables: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    if "eb_names" not in variables or "ss_names" not in variables:
        raise CampaignError("Exodus mesh lacks eb_names or ss_names metadata.")
    block_names = _decode_exodus_names(variables["eb_names"][:])
    side_names = _decode_exodus_names(variables["ss_names"][:])
    required = {"fluid-hex", "wing", "wing-pp", "inlet", "outlet", "front", "back"}
    observed = set(block_names) | set(side_names)
    missing = sorted(required - observed)
    if missing:
        raise CampaignError(f"Exodus mesh is missing required parts: {missing}")

    if all(name in variables for name in ("coordx", "coordy", "coordz")):
        coordinates = tuple(
            np.asarray(variables[name][:]) for name in ("coordx", "coordy", "coordz")
        )
    elif "coord" in variables:
        packed = np.asarray(variables["coord"][:])
        if packed.shape[0] < 3:
            raise CampaignError("Exodus coordinate array is not three-dimensional.")
        coordinates = (packed[0], packed[1], packed[2])
    else:
        raise CampaignError("Exodus mesh lacks coordx/coordy/coordz or coord.")

    connect_names = sorted(
        (name for name in variables if re.fullmatch(r"connect\d+", name)),
        key=lambda name: int(name[7:]),
    )
    if not connect_names:
        raise CampaignError("Exodus mesh has no element connectivity arrays.")
    cumulative: list[tuple[int, int, str]] = []
    first = 1
    for name in connect_names:
        count = int(variables[name].shape[0])
        cumulative.append((first, first + count - 1, name))
        first += count

    face_nodes = {
        1: (0, 1, 5, 4),
        2: (1, 2, 6, 5),
        3: (2, 3, 7, 6),
        4: (0, 4, 7, 3),
        5: (0, 3, 2, 1),
        6: (4, 5, 6, 7),
    }
    surface_nodes: set[int] = set()
    for surface_name in ("wing", "wing-pp"):
        surface_index = side_names.index(surface_name) + 1
        element_key = f"elem_ss{surface_index}"
        side_key = f"side_ss{surface_index}"
        if element_key not in variables or side_key not in variables:
            raise CampaignError(
                f"Exodus side set {surface_name} lacks element/side membership."
            )
        elements = np.asarray(variables[element_key][:], dtype=np.int64).ravel()
        sides = np.asarray(variables[side_key][:], dtype=np.int64).ravel()
        if len(elements) != len(sides) or not len(elements):
            raise CampaignError(
                f"Exodus side set {surface_name} is empty or malformed."
            )
        for element, side in zip(elements, sides):
            block = next(
                (row for row in cumulative if row[0] <= element <= row[1]), None
            )
            if block is None or int(side) not in face_nodes:
                raise CampaignError(
                    f"Cannot resolve {surface_name} element {element}/side {side}."
                )
            first_element, _, connectivity_name = block
            connectivity = np.asarray(
                variables[connectivity_name][int(element) - first_element],
                dtype=np.int64,
            ).ravel()
            if len(connectivity) != 8:
                raise CampaignError(
                    f"Expected HEX8 on {surface_name}; {connectivity_name} has {len(connectivity)} nodes/element."
                )
            surface_nodes.update(
                int(connectivity[offset]) - 1 for offset in face_nodes[int(side)]
            )

    node_index = np.fromiter(sorted(surface_nodes), dtype=np.int64)
    if not len(node_index):
        raise CampaignError("Resolved no nodes on combined [wing, wing-pp] surface.")
    bounds = []
    extents = []
    for coordinate in coordinates:
        values = np.asarray(coordinate[node_index], dtype=float)
        lower, upper = float(np.min(values)), float(np.max(values))
        bounds.append([lower, upper])
        extents.append(upper - lower)
    x_extent, y_extent, z_extent = extents
    if not (0.12 <= x_extent <= 0.24 and 0.95 <= y_extent <= 1.05):
        raise CampaignError(
            "Combined airfoil surface is not the chord-1 NACA0018 rotated onto global Y: "
            f"x/y extents are {x_extent:.9g}/{y_extent:.9g}."
        )
    if not math.isclose(z_extent, 4.0, rel_tol=0.0, abs_tol=2.0e-5):
        raise CampaignError(
            f"Airfoil surface span is {z_extent:.9g} m, not the frozen 4 m."
        )
    y_min, y_max = bounds[1]
    rotated_minus_90 = abs(y_max) <= 2.0e-3 and -1.01 <= y_min <= -0.99
    if not rotated_minus_90:
        raise CampaignError(
            "Airfoil chord is global Y but not the frozen -90-degree orientation "
            f"(surface y bounds {y_min:.9g}, {y_max:.9g})."
        )
    return {
        "element_block_names": block_names,
        "side_set_names": side_names,
        "combined_force_surface": ["wing", "wing-pp"],
        "combined_force_surface_node_count": len(surface_nodes),
        "source_surface_bounds_xyz_m": bounds,
        "source_surface_extents_xyz_m": extents,
        "scaled_surface_extents_xyz_m": [0.23 * x_extent, 0.23 * y_extent, z_extent],
        "chord_axis": "global_Y",
        "incidence_orientation_deg": -90.0,
        "source_chord_m": y_extent,
        "source_span_m": z_extent,
    }


def audit_mesh(campaign: Path) -> dict[str, Any]:
    campaign = campaign.expanduser().resolve()
    validate_campaign(campaign)
    manifest = _load_manifest(campaign)
    existing = campaign / "mesh_audit.json"
    if (
        existing.is_file()
        and _mesh_audit_status(campaign, manifest)["status"] == "pass"
    ):
        return json.loads(existing.read_text(encoding="utf-8"))
    mesh = Path(manifest["mesh"]["path"])
    details: dict[str, Any] | None = None
    backend = ""
    scipy_error: Exception | None = None
    try:
        from scipy.io import netcdf_file

        with netcdf_file(mesh, "r", mmap=True) as database:
            details = _audit_exodus_variables(mesh, database.variables)
        backend = "scipy.io.netcdf_file"
    except Exception as exc:  # NetCDF4/HDF5 Exodus falls through to h5py.
        scipy_error = exc
    if details is None:
        try:
            import h5py

            with h5py.File(mesh, "r") as database:
                details = _audit_exodus_variables(mesh, database)
            backend = "h5py"
        except Exception as exc:
            raise CampaignError(
                f"Could not audit Exodus mesh with scipy ({scipy_error}) or h5py ({exc})."
            ) from exc
    report = {
        "schema": MESH_AUDIT_SCHEMA,
        "status": "pass",
        "campaign_fingerprint": manifest["fingerprint"],
        "mesh": _file_identity(mesh),
        "reader_backend": backend,
        **details,
    }
    _write_json(campaign / "mesh_audit.json", report)
    return report


def _mesh_audit_status(campaign: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = campaign / "mesh_audit.json"
    if not path.is_file():
        return {"status": "pending", "path": "mesh_audit.json"}
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != MESH_AUDIT_SCHEMA
        or report.get("status") != "pass"
        or report.get("campaign_fingerprint") != manifest["fingerprint"]
        or report.get("mesh", {}).get("sha256") != manifest["mesh"]["sha256"]
    ):
        raise CampaignError("Mesh audit is invalid or stale.")
    return {"status": "pass", "path": "mesh_audit.json", "sha256": _sha256(path)}


def _canonical_history_entry(case_id: str, updates: int) -> dict[str, Any]:
    return {
        "path": f"histories/{case_id}_nalu.csv",
        "status": "pending",
        "sha256": None,
        "rows_expected": int(updates),
        "columns": list(HISTORY_COLUMNS),
    }


def generate_campaign(
    campaign: Path, mesh: Path, expected_mesh_sha256: str
) -> dict[str, Any]:
    request, request_sha = _load_request()
    physics = _physics(request)
    mesh = mesh.expanduser().resolve()
    if not mesh.is_file():
        raise CampaignError(f"NACA0018 Exodus mesh is unavailable: {mesh}")
    expected = expected_mesh_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise CampaignError("--mesh-sha256 must be an explicit 64-character SHA-256.")
    observed = _sha256(mesh)
    if observed != expected:
        raise CampaignError(
            f"Mesh SHA-256 mismatch for {mesh}: expected {expected}, observed {observed}."
        )
    campaign = campaign.expanduser().resolve()
    if campaign.exists():
        raise CampaignError(f"Refusing to replace an existing campaign: {campaign}")
    campaign.mkdir(parents=True)
    template = _load_template()

    precursors: list[dict[str, Any]] = []
    precursor_by_re: dict[float, dict[str, Any]] = {}
    for index, reynolds in enumerate(EXPECTED_REYNOLDS):
        flow = _flow_values(reynolds, physics)
        identifier = _precursor_id(reynolds)
        case_dir = campaign / "precursors" / identifier
        yaml_path = case_dir / f"{identifier}.yaml"
        deck = _precursor_deck(
            template,
            original_mesh=mesh,
            case_dir=case_dir,
            flow=flow,
            physics=physics,
        )
        _write_yaml(yaml_path, deck)
        row = {
            "index": index,
            "precursor_id": identifier,
            "reynolds_number": reynolds,
            **flow,
            "mesh_scaling_xyz": [0.23, 0.23, 1.0],
            "zero_amplitude_ale": True,
            "yaml": _relative(yaml_path, campaign),
            "yaml_sha256": _sha256(yaml_path),
            "log": _relative(case_dir / "log.out", campaign),
            "force_output": _relative(case_dir / "full_surface_force.dat", campaign),
            "restart_output": _relative(
                case_dir / "restart" / "precursor.rst", campaign
            ),
        }
        precursors.append(row)
        precursor_by_re[reynolds] = row

    cases: list[dict[str, Any]] = []
    for reynolds in EXPECTED_REYNOLDS:
        precursor = precursor_by_re[reynolds]
        precursor_restart = campaign / str(precursor["restart_output"])
        for reduced_frequency in EXPECTED_K:
            for amplitude_ratio in EXPECTED_NONZERO_AMPLITUDE_RATIOS:
                values = _motion_values(
                    reynolds, reduced_frequency, amplitude_ratio, physics
                )
                identifier = _case_id(reynolds, reduced_frequency, amplitude_ratio)
                case_dir = campaign / "motion_cases" / identifier
                yaml_path = case_dir / f"{identifier}.yaml"
                deck = _motion_deck(
                    template,
                    precursor_restart=precursor_restart,
                    case_dir=case_dir,
                    values=values,
                    physics=physics,
                )
                _write_yaml(yaml_path, deck)
                cases.append(
                    {
                        "index": len(cases),
                        "case_id": identifier,
                        "kind": "prescribed_motion",
                        "incidence_deg": EXPECTED_INCIDENCE_DEG,
                        "reynolds_number": reynolds,
                        "reduced_frequency_k": reduced_frequency,
                        "amplitude_ratio": amplitude_ratio,
                        "direction_deg_mod_180": EXPECTED_DIRECTION_DEG,
                        "wind_speed_mps": values["wind_speed_mps"],
                        "frequency_hz": values["frequency_hz"],
                        "period_s": values["period_s"],
                        "amplitude_m": values["amplitude_m"],
                        "precursor_id": precursor["precursor_id"],
                        "precursor_restart_time_s": values["precursor_restart_time_s"],
                        "precursor_restart_step": values["precursor_steps"],
                        "phase_cycles_at_restart": values["phase_cycles_at_restart"],
                        "samples_per_cycle": values["samples_per_cycle"],
                        "discard_cycles": DISCARD_CYCLES,
                        "retained_cycles": RETAINED_CYCLES,
                        "updates_after_restart": values["updates_after_restart"],
                        "relative_history_start_s": values["relative_history_start_s"],
                        "relative_history_end_s": values["relative_history_end_s"],
                        "raw_force_time_start_expected_s": float(
                            values["precursor_restart_time_s"]
                        )
                        + float(values["time_step_s"]),
                        "raw_force_time_end_expected_s": float(
                            values["precursor_restart_time_s"]
                        )
                        + float(values["relative_history_end_s"]),
                        "time_step_s": values["time_step_s"],
                        "yaml": _relative(yaml_path, campaign),
                        "yaml_sha256": _sha256(yaml_path),
                        "log": _relative(case_dir / "log.out", campaign),
                        "force_output": _relative(
                            case_dir / "full_surface_force.dat", campaign
                        ),
                        "restart_output": _relative(
                            case_dir / "restart" / "motion.rst", campaign
                        ),
                        "canonical_history": _canonical_history_entry(
                            identifier, int(values["updates_after_restart"])
                        ),
                    }
                )

    # Scaling probes use the same scaled, zero-amplitude ALE physics and time
    # step as each precursor.  Node count only changes the Flight execution;
    # separate paths prevent concurrent probes from appending to one output.
    scaling_cases: list[dict[str, Any]] = []
    for nodes in SUPPORTED_NODE_COUNTS:
        for reynolds in (SCALING_REYNOLDS,):
            flow = _flow_values(reynolds, physics)
            probe_dir = (
                campaign
                / "preflight"
                / "scaling"
                / f"re{int(reynolds):06d}_nodes{nodes}"
            )
            yaml_path = probe_dir / "scaling.yaml"
            deck = _precursor_deck(
                template,
                original_mesh=mesh,
                case_dir=probe_dir,
                flow=flow,
                physics=physics,
                updates=SCALING_PROBE_UPDATES,
            )
            _write_yaml(yaml_path, deck)
            scaling_cases.append(
                {
                    "nodes": nodes,
                    "ranks": nodes * PHYSICAL_CORES_PER_FLIGHT_NODE,
                    "reynolds_number": reynolds,
                    "updates": SCALING_PROBE_UPDATES,
                    "yaml": _relative(yaml_path, campaign),
                    "yaml_sha256": _sha256(yaml_path),
                    "log": _relative(probe_dir / "log.out", campaign),
                    "report": _relative(probe_dir / "scaling_report.json", campaign),
                }
            )

    preflight_case = max(
        cases,
        key=lambda row: (
            int(row["updates_after_restart"]),
            float(row["amplitude_ratio"]),
            float(row["reynolds_number"]),
        ),
    )
    manifest: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "generated_awaiting_scaling_preflight",
        "request": {
            "path": str(REQUEST_PATH.resolve()),
            "sha256": request_sha,
            "schema": request["schema"],
        },
        "solver": {
            "name": "Nalu-Wind",
            "version": "v2.2.1",
            "release_source": "official_github_v2.2.1",
            "expected_git_commit": EXPECTED_NALU_GIT_COMMIT,
            "turbulence_model": "sst_iddes",
            "transition_model": "none",
            "model_scope_caveat": (
                "SST-IDDES is the recovered original high-fidelity workflow; no transition "
                "model is enabled, so laminar-transition fidelity at Re=64000 is not claimed."
            ),
        },
        "assets": {
            "template": _file_identity(TEMPLATE_PATH),
            "hypre": _file_identity(HYPRE_PATH),
        },
        "mesh": {
            "path": str(mesh),
            "sha256": observed,
            "source_chord_m": 1.0,
            "source_span_m": 4.0,
            "transformation_xyz": [0.23, 0.23, 1.0],
            "transformation_stage": "precursor_only",
            "required_parts": [
                "fluid-hex",
                "wing",
                "wing-pp",
                "inlet",
                "outlet",
                "front",
                "back",
            ],
        },
        "fluid_and_section": physics,
        "numerics": {
            "precursor_phase_factor": PRECURSOR_PHASE_FACTOR,
            "precursor_steps": int(precursors[0]["precursor_steps"]),
            "target_st_max": TARGET_ST_MAX,
            "target_samples_at_st_max": TARGET_SAMPLES_AT_ST_MAX,
            "discard_cycles": DISCARD_CYCLES,
            "retained_cycles": RETAINED_CYCLES,
            "updates_rule": "13*samples_per_cycle+1",
            "force_surface": ["wing", "wing-pp"],
            "crossflow_force_expression": "Fpy+Fvy",
        },
        "history_contract": {
            "time_origin": "precursor_restart",
            "first_row": "dt",
            "last_row": "(13*samples_per_cycle+1)*dt",
            "complete_cycle_intervals": TOTAL_CYCLES,
            "raw_absolute_time_rule": "t0+j*dt for j=1..13*N+1",
            "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
            "columns": list(HISTORY_COLUMNS),
            "float_encoding": ".17g",
            "pairing_policy": "exact_array_equal_time_displacement_velocity_no_interpolation",
            "force_units": "N_total_over_4m_span",
        },
        "flight": {
            "physical_cores_per_node": PHYSICAL_CORES_PER_FLIGHT_NODE,
            "baseline_nodes_per_case": BASELINE_NODES,
            "baseline_ranks_per_case": BASELINE_NODES * PHYSICAL_CORES_PER_FLIGHT_NODE,
            "supported_scaling_nodes": list(SUPPORTED_NODE_COUNTS),
            "maximum_walltime_hours": MAX_WALLTIME_HOURS,
            "production_requires_measured_scaling_gate": True,
            "archived_two_node_step_seconds": ARCHIVED_TWO_NODE_STEP_SECONDS,
            "archived_two_node_step_range_seconds": list(
                ARCHIVED_TWO_NODE_STEP_RANGE_SECONDS
            ),
            "archived_rate_source": ARCHIVED_RATE_SOURCE,
        },
        "preflight": {
            "scaling_probe_updates": SCALING_PROBE_UPDATES,
            "scaling_reynolds_number": SCALING_REYNOLDS,
            "scaling_similarity_basis": (
                "The mesh, nondimensional time step, and step count are identical across Re; "
                "the Re=64000 probe is the conservative representative, and a complete moving "
                "case is still required before the matrix."
            ),
            "motion_case_id": preflight_case["case_id"],
            "motion_case_index": preflight_case["index"],
            "motion_preflight_required_before_matrix": True,
            "scaling_cases": scaling_cases,
        },
        "precursors": precursors,
        "cases": cases,
    }
    payload = copy.deepcopy(manifest)
    manifest["fingerprint"] = _canonical_json_sha256(payload)
    _write_json(campaign / "campaign_manifest.json", manifest)
    return manifest


def _load_manifest(campaign: Path) -> dict[str, Any]:
    path = campaign / "campaign_manifest.json"
    if not path.is_file():
        raise CampaignError(f"Campaign manifest is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError("Campaign manifest must be a JSON object.")
    return value


def _restart_files(base: Path) -> list[Path]:
    return sorted(
        path
        for path in base.parent.glob(f"{base.name}*")
        if path.is_file() and path.stat().st_size > 0
    )


def _max_logged_step(log: Path) -> int:
    if not log.is_file():
        return -1
    maximum = -1
    for match in STEP_RE.finditer(log.read_text(encoding="utf-8", errors="replace")):
        maximum = max(maximum, int(match.group(1)))
    return maximum


def _terminal_completion_step(log: Path) -> int | None:
    if not log.is_file():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    markers = list(TERMINAL_COMPLETION_RE.finditer(text))
    if not markers:
        return None
    marker = markers[-1]
    steps = list(STEP_RE.finditer(text))
    if steps and marker.start() < steps[-1].end():
        return None
    try:
        terminal_time = float(marker.group("time"))
    except ValueError:
        return None
    if not math.isfinite(terminal_time):
        return None
    return int(marker.group("step"))


def _log_failure(log: Path) -> str | None:
    if not log.is_file():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    for pattern in FAILURE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _validate_deck_common(
    deck: Mapping[str, Any],
    *,
    expected_speed: float,
    physics: Mapping[str, float],
) -> Mapping[str, Any]:
    realm = deck["realms"][0]
    if realm["post_processing"] != [
        {
            "type": "surface",
            "physics": "surface_force_and_moment",
            "output_file_name": realm["post_processing"][0]["output_file_name"],
            "frequency": 1,
            "parameters": [0.0, 0.0, 0.0],
            "target_name": ["wing", "wing-pp"],
        }
    ]:
        raise CampaignError("Nalu force output must contain only full [wing, wing-pp].")
    inflow = next(
        bc["inflow_user_data"]
        for bc in realm["boundary_conditions"]
        if "inflow_user_data" in bc
    )
    _assert_close(float(inflow["velocity"][0]), expected_speed, "inflow speed")
    if list(map(float, inflow["velocity"][1:])) != [0.0, 0.0]:
        raise CampaignError("Nalu inflow must remain global +X.")
    specifications = realm["material_properties"]["specifications"]
    density = next(row for row in specifications if row["name"] == "density")
    viscosity = next(row for row in specifications if row["name"] == "viscosity")
    _assert_close(density["value"], physics["fluid_density_kgpm3"], "density")
    _assert_close(
        viscosity["value"], physics["fluid_dynamic_viscosity_pas"], "viscosity"
    )
    if realm["solution_options"].get("turbulence_model") != "sst_iddes":
        raise CampaignError("Nalu turbulence model must remain SST-IDDES.")
    source_terms = next(
        (
            option["source_terms"]
            for option in realm["solution_options"]["options"]
            if "source_terms" in option
        ),
        None,
    )
    if source_terms != {
        "continuity": "gcl",
        "momentum": "gcl",
        "turbulent_ke": "gcl",
        "specific_dissipation_rate": "gcl",
    }:
        raise CampaignError(
            "Moving-mesh GCL sources must remain active for every solved equation."
        )
    if any(
        "Transition" in next(iter(system), "")
        for system in realm["equation_systems"]["systems"]
    ):
        raise CampaignError(
            "The recovered Nalu workflow must not silently enable transition."
        )
    return realm


def validate_campaign(campaign: Path) -> dict[str, Any]:
    campaign = campaign.expanduser().resolve()
    manifest = _load_manifest(campaign)
    if manifest.get("schema") != CAMPAIGN_SCHEMA:
        raise CampaignError("Unexpected Nalu campaign schema.")
    fingerprint = manifest.get("fingerprint")
    payload = copy.deepcopy(manifest)
    payload.pop("fingerprint", None)
    if fingerprint != _canonical_json_sha256(payload):
        raise CampaignError("Campaign manifest fingerprint mismatch.")
    request, request_sha = _load_request()
    if manifest["request"]["sha256"] != request_sha:
        raise CampaignError("The frozen work-map request changed after generation.")
    for name, path in (("template", TEMPLATE_PATH), ("hypre", HYPRE_PATH)):
        if manifest["assets"][name] != _file_identity(path):
            raise CampaignError(f"Tracked Nalu {name} asset changed after generation.")
    mesh = Path(manifest["mesh"]["path"])
    if not mesh.is_file():
        raise CampaignError(f"Bound NACA0018 mesh is unavailable: {mesh}")
    if _sha256(mesh) != manifest["mesh"]["sha256"]:
        raise CampaignError("Bound NACA0018 mesh SHA-256 mismatch.")
    if len(manifest.get("precursors", [])) != 3:
        raise CampaignError("Campaign must contain exactly three precursors.")
    if len(manifest.get("cases", [])) != 48:
        raise CampaignError("Campaign must contain exactly 48 moving cases.")
    physics = _physics(request)
    expected_vertices = {
        (reynolds, reduced_frequency, amplitude)
        for reynolds in EXPECTED_REYNOLDS
        for reduced_frequency in EXPECTED_K
        for amplitude in EXPECTED_NONZERO_AMPLITUDE_RATIOS
    }
    actual_vertices: set[tuple[float, float, float]] = set()
    precursor_by_id = {row["precursor_id"]: row for row in manifest["precursors"]}
    for precursor in manifest["precursors"]:
        reynolds = float(precursor["reynolds_number"])
        flow = _flow_values(reynolds, physics)
        for key in (
            "wind_speed_mps",
            "precursor_restart_time_s",
            "precursor_time_step_s",
        ):
            _assert_close(float(precursor[key]), float(flow[key]), f"precursor {key}")
        if int(precursor["precursor_steps"]) != int(flow["precursor_steps"]):
            raise CampaignError("Precursor step count changed.")
        yaml_path = campaign / precursor["yaml"]
        if _sha256(yaml_path) != precursor["yaml_sha256"]:
            raise CampaignError(f"Precursor YAML identity mismatch: {yaml_path}")
        deck = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        realm = _validate_deck_common(
            deck, expected_speed=float(flow["wind_speed_mps"]), physics=physics
        )
        if Path(realm["mesh"]).resolve() != mesh.resolve():
            raise CampaignError("Precursor does not read the bound original mesh.")
        scaling = realm.get("mesh_transformation", [])[0]["motion"][0]
        if scaling != {
            "type": "scaling",
            "factor": [0.23, 0.23, 1.0],
            "centroid": [0.0, 0.0, 0.0],
        }:
            raise CampaignError("Precursor one-time mesh scaling is not frozen.")
        motion = realm.get("mesh_motion", [])[0]["motion"][0]
        if motion["type"] != "oscillation" or float(motion["amplitude"]) != 0.0:
            raise CampaignError("Precursor must use zero-amplitude ALE oscillation.")
        if realm["restart"].get("restart_time") is not None:
            raise CampaignError("A precursor may not read a restart.")
    for case in manifest["cases"]:
        vertex = (
            float(case["reynolds_number"]),
            float(case["reduced_frequency_k"]),
            float(case["amplitude_ratio"]),
        )
        if vertex in actual_vertices:
            raise CampaignError(f"Duplicate prescribed-motion vertex: {vertex}")
        actual_vertices.add(vertex)
        values = _motion_values(*vertex, physics)
        for key in (
            "wind_speed_mps",
            "frequency_hz",
            "period_s",
            "amplitude_m",
            "precursor_restart_time_s",
            "time_step_s",
            "relative_history_start_s",
            "relative_history_end_s",
            "phase_cycles_at_restart",
        ):
            _assert_close(
                float(case[key]), float(values[key]), f"{case['case_id']} {key}"
            )
        _assert_close(
            case["raw_force_time_start_expected_s"],
            float(values["precursor_restart_time_s"]) + float(values["time_step_s"]),
            f"{case['case_id']} raw force start",
        )
        _assert_close(
            case["raw_force_time_end_expected_s"],
            float(values["precursor_restart_time_s"])
            + float(values["relative_history_end_s"]),
            f"{case['case_id']} raw force end",
        )
        for key, expected in (
            ("samples_per_cycle", values["samples_per_cycle"]),
            ("updates_after_restart", values["updates_after_restart"]),
            ("precursor_restart_step", values["precursor_steps"]),
            ("discard_cycles", DISCARD_CYCLES),
            ("retained_cycles", RETAINED_CYCLES),
        ):
            if int(case[key]) != int(expected):
                raise CampaignError(f"{case['case_id']} {key} changed.")
        if int(case["samples_per_cycle"]) < 100:
            raise CampaignError("A motion case has fewer than 100 samples per cycle.")
        history = case["canonical_history"]
        if history["columns"] != list(HISTORY_COLUMNS):
            raise CampaignError("Canonical history columns changed.")
        if int(history["rows_expected"]) != int(case["updates_after_restart"]):
            raise CampaignError("Canonical history row count changed.")
        yaml_path = campaign / case["yaml"]
        if _sha256(yaml_path) != case["yaml_sha256"]:
            raise CampaignError(f"Motion YAML identity mismatch: {yaml_path}")
        deck = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        realm = _validate_deck_common(
            deck, expected_speed=float(case["wind_speed_mps"]), physics=physics
        )
        if "mesh_transformation" in realm:
            raise CampaignError("Motion child would scale its precursor restart twice.")
        precursor = precursor_by_id[case["precursor_id"]]
        expected_restart = campaign / precursor["restart_output"]
        if Path(realm["mesh"]).resolve() != expected_restart.resolve():
            raise CampaignError("Motion child is not bound to its precursor restart.")
        _assert_close(
            realm["restart"]["restart_time"],
            case["precursor_restart_time_s"],
            "motion restart time",
        )
        motion = realm["mesh_motion"][0]["motion"][0]
        if motion["type"] != "oscillation" or motion["direction"] != [0.0, 1.0, 0.0]:
            raise CampaignError("Motion child is not global-Y native ALE oscillation.")
        _assert_close(
            motion["start_time"], case["precursor_restart_time_s"], "motion start"
        )
        _assert_close(motion["period"], case["period_s"], "motion period")
        _assert_close(motion["amplitude"], case["amplitude_m"], "motion amplitude")
        integrator = deck["Time_Integrators"][0]["StandardTimeIntegrator"]
        expected_stop = int(case["precursor_restart_step"]) + int(
            case["updates_after_restart"]
        )
        if int(integrator["termination_step_count"]) != expected_stop:
            raise CampaignError("Motion child does not execute 13*N+1 updates.")
    if actual_vertices != expected_vertices:
        raise CampaignError("The moving-case vertex grid is incomplete.")
    for probe in manifest["preflight"]["scaling_cases"]:
        path = campaign / probe["yaml"]
        if _sha256(path) != probe["yaml_sha256"]:
            raise CampaignError(f"Scaling YAML identity mismatch: {path}")
    return {
        "valid": True,
        "schema": CAMPAIGN_SCHEMA,
        "fingerprint": fingerprint,
        "precursor_count": len(manifest["precursors"]),
        "motion_case_count": len(manifest["cases"]),
        "mesh_sha256": manifest["mesh"]["sha256"],
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise CampaignError("Cannot compute a percentile of no samples.")
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def record_scaling(
    campaign: Path,
    *,
    nodes: int,
    reynolds: float,
    log: Path,
    elapsed_seconds: float,
    nalu_executable: Path,
) -> dict[str, Any]:
    validate_campaign(campaign)
    manifest = _load_manifest(campaign)
    if nodes not in SUPPORTED_NODE_COUNTS:
        raise CampaignError(f"Scaling nodes must be one of {SUPPORTED_NODE_COUNTS}.")
    probe = next(
        (
            row
            for row in manifest["preflight"]["scaling_cases"]
            if int(row["nodes"]) == nodes
            and float(row["reynolds_number"]) == float(reynolds)
        ),
        None,
    )
    if probe is None:
        raise CampaignError("No generated scaling case matches nodes/Re.")
    expected_log = campaign / probe["log"]
    if log.expanduser().resolve() != expected_log.resolve():
        raise CampaignError(f"Scaling log path must be {expected_log}.")
    text = expected_log.read_text(encoding="utf-8", errors="replace")
    versions = set(NALU_VERSION_RE.findall(text))
    commits = set(NALU_COMMIT_RE.findall(text))
    expected_solver = manifest["solver"]
    executable_identity = _nalu_executable_identity(nalu_executable)
    _scaling_gate(campaign, manifest, current_nalu_executable=executable_identity)
    if versions != {expected_solver["version"]}:
        raise CampaignError(
            "Scaling log does not prove the frozen Nalu-Wind v2.2.1 runtime."
        )
    if commits != {expected_solver["expected_git_commit"]}:
        raise CampaignError(
            "Scaling log does not prove the frozen Nalu-Wind commit identity."
        )
    samples = [float(match.group("seconds")) for match in WALLCLOCK_RE.finditer(text)]
    if len(samples) < 8:
        raise CampaignError(
            f"Scaling log has {len(samples)} completed step timings; require at least 8."
        )
    if _log_failure(expected_log):
        raise CampaignError("Scaling log contains a Nalu failure marker.")
    max_step = _max_logged_step(expected_log)
    if max_step != SCALING_PROBE_UPDATES:
        raise CampaignError(
            f"Scaling probe stopped at step {max_step}; expected {SCALING_PROBE_UPDATES}."
        )
    terminal_step = _terminal_completion_step(expected_log)
    if terminal_step != SCALING_PROBE_UPDATES:
        raise CampaignError(
            "Scaling log lacks the exact normal-completion terminal banner."
        )
    p95 = _percentile(samples, 0.95)
    bounded = max(statistics.mean(samples), p95) * SCALING_RATE_SAFETY_FACTOR
    initialization = max(0.0, float(elapsed_seconds) - sum(samples))
    precursor_steps = max(int(row["precursor_steps"]) for row in manifest["precursors"])
    motion_steps = max(int(row["updates_after_restart"]) for row in manifest["cases"])
    precursor_hours = (initialization + bounded * precursor_steps) / 3600.0
    motion_hours = (initialization + bounded * motion_steps) / 3600.0
    projected_hours = max(precursor_hours, motion_hours)
    report = {
        "schema": SCALING_SCHEMA,
        "status": "pass" if projected_hours <= MAX_WALLTIME_HOURS else "fail",
        "campaign_fingerprint": manifest["fingerprint"],
        "nodes": nodes,
        "ranks": nodes * PHYSICAL_CORES_PER_FLIGHT_NODE,
        "reynolds_number": float(reynolds),
        "probe_updates": SCALING_PROBE_UPDATES,
        "observed_solver_version": next(iter(versions)),
        "observed_solver_git_commit": next(iter(commits)),
        "terminal_step": terminal_step,
        "nalu_executable": executable_identity,
        "timing_sample_count": len(samples),
        "mean_step_seconds": statistics.mean(samples),
        "p95_step_seconds": p95,
        "maximum_step_seconds": max(samples),
        "rate_safety_factor": SCALING_RATE_SAFETY_FACTOR,
        "bounded_step_seconds": bounded,
        "elapsed_seconds": float(elapsed_seconds),
        "estimated_initialization_seconds": initialization,
        "projected_precursor_hours": precursor_hours,
        "projected_longest_motion_hours": motion_hours,
        "target_walltime_hours": TARGET_WALLTIME_HOURS,
        "target_walltime_pass": projected_hours <= TARGET_WALLTIME_HOURS,
        "maximum_walltime_hours": MAX_WALLTIME_HOURS,
        "maximum_walltime_pass": projected_hours <= MAX_WALLTIME_HOURS,
        "archived_two_node_caveat": {
            "observed_step_seconds": ARCHIVED_TWO_NODE_STEP_SECONDS,
            "observed_range_seconds": list(ARCHIVED_TWO_NODE_STEP_RANGE_SECONDS),
            "source": ARCHIVED_RATE_SOURCE,
            "note": "The archived evidence is a first-step observation, not a production rate guarantee.",
        },
        "log": _relative(expected_log, campaign),
        "log_sha256": _sha256(expected_log),
    }
    _write_json(campaign / probe["report"], report)
    return report


def _scaling_gate(
    campaign: Path,
    manifest: Mapping[str, Any],
    *,
    current_nalu_executable: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for probe in manifest["preflight"]["scaling_cases"]:
        path = campaign / probe["report"]
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        projected = max(
            float(report.get("projected_precursor_hours", math.inf)),
            float(report.get("projected_longest_motion_hours", math.inf)),
        )
        executable = _recorded_nalu_executable_identity(report.get("nalu_executable"))
        if (
            report.get("schema") != SCALING_SCHEMA
            or report.get("campaign_fingerprint") != manifest["fingerprint"]
            or report.get("log_sha256") != _sha256(campaign / report["log"])
            or report.get("target_walltime_hours") != TARGET_WALLTIME_HOURS
            or report.get("target_walltime_pass")
            is not (projected <= TARGET_WALLTIME_HOURS)
            or report.get("maximum_walltime_hours") != MAX_WALLTIME_HOURS
            or report.get("maximum_walltime_pass")
            is not (projected <= MAX_WALLTIME_HOURS)
            or report.get("terminal_step") != SCALING_PROBE_UPDATES
            or _terminal_completion_step(campaign / report["log"])
            != SCALING_PROBE_UPDATES
            or report.get("status")
            != ("pass" if projected <= MAX_WALLTIME_HOURS else "fail")
        ):
            raise CampaignError(f"Invalid or stale scaling report: {path}")
        report["nalu_executable"] = executable
        reports.append(report)
    executable_identities = {
        json.dumps(row["nalu_executable"], sort_keys=True, separators=(",", ":"))
        for row in reports
    }
    if len(executable_identities) > 1:
        raise CampaignError(
            "The 2/4/8-node scaling probes used different naluX executables."
        )
    bound_executable = None if not reports else dict(reports[0]["nalu_executable"])
    if (
        current_nalu_executable is not None
        and bound_executable is not None
        and dict(current_nalu_executable) != bound_executable
    ):
        raise CampaignError(
            "The current naluX executable differs from the scaling identity."
        )

    by_nodes: dict[str, Any] = {}
    for nodes in SUPPORTED_NODE_COUNTS:
        node_reports = [row for row in reports if int(row["nodes"]) == nodes]
        reynolds = {float(row["reynolds_number"]) for row in node_reports}
        complete = reynolds == {SCALING_REYNOLDS}
        target_passed = complete and all(
            row["target_walltime_pass"] is True for row in node_reports
        )
        maximum_passed = complete and all(
            row["maximum_walltime_pass"] is True for row in node_reports
        )
        by_nodes[str(nodes)] = {
            "complete_reynolds_sweep": complete,
            "target_walltime_pass": target_passed,
            "maximum_walltime_pass": maximum_passed,
            "pass": maximum_passed,
            "reports": len(node_reports),
            "worst_projected_hours": max(
                (
                    max(
                        float(row["projected_precursor_hours"]),
                        float(row["projected_longest_motion_hours"]),
                    )
                    for row in node_reports
                ),
                default=None,
            ),
        }
    all_probes_complete = len(reports) == len(SUPPORTED_NODE_COUNTS)
    target_nodes = [
        nodes
        for nodes in SUPPORTED_NODE_COUNTS
        if by_nodes[str(nodes)]["target_walltime_pass"]
    ]
    fallback_nodes = [
        nodes
        for nodes in SUPPORTED_NODE_COUNTS
        if by_nodes[str(nodes)]["maximum_walltime_pass"]
    ]
    if not all_probes_complete:
        recommended = None
        selection_tier = None
    elif target_nodes:
        recommended = min(target_nodes)
        selection_tier = "target_2h"
    elif fallback_nodes:
        recommended = min(fallback_nodes)
        selection_tier = "fallback_8h"
    else:
        recommended = None
        selection_tier = None
    recommended_walltime = (
        "02:00:00"
        if selection_tier == "target_2h"
        else "08:00:00" if selection_tier == "fallback_8h" else None
    )
    return {
        "status": "pass" if recommended is not None else "awaiting_passing_sweep",
        "required_probe_count": len(SUPPORTED_NODE_COUNTS),
        "complete_probe_count": len(reports),
        "recommended_nodes": recommended,
        "selection_tier": selection_tier,
        "recommended_walltime": recommended_walltime,
        "recommended_ranks": (
            None
            if recommended is None
            else recommended * PHYSICAL_CORES_PER_FLIGHT_NODE
        ),
        "target_walltime_hours": TARGET_WALLTIME_HOURS,
        "maximum_walltime_hours": MAX_WALLTIME_HOURS,
        "by_nodes": by_nodes,
        "archived_two_node_step_seconds": ARCHIVED_TWO_NODE_STEP_SECONDS,
        "archived_two_node_step_range_seconds": list(
            ARCHIVED_TWO_NODE_STEP_RANGE_SECONDS
        ),
        "archived_rate_source": ARCHIVED_RATE_SOURCE,
        "runtime_identity": {
            "status": "verified" if all_probes_complete else "pending",
            "observed_versions": sorted(
                {str(row["observed_solver_version"]) for row in reports}
            ),
            "observed_git_commits": sorted(
                {str(row["observed_solver_git_commit"]) for row in reports}
            ),
            "nalu_executable": bound_executable,
        },
    }


def _force_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    if not path.is_file():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if len(fields) < 6:
            continue
        try:
            row = [float(value) for value in fields]
        except ValueError:
            continue
        if len(row) >= 6 and all(math.isfinite(value) for value in row[:6]):
            rows.append(row)
    rows.sort(key=lambda row: row[0])
    return rows


def _select_expected_force_rows(
    rows: Sequence[Sequence[float]], case: Mapping[str, Any]
) -> list[Sequence[float]] | None:
    restart = float(case["precursor_restart_time_s"])
    dt = float(case["time_step_s"])
    count = int(case["updates_after_restart"])
    selected: list[Sequence[float]] = []
    cursor = 0
    for index in range(1, count + 1):
        expected = restart + index * dt
        # SurfaceForceAndMomentAlgorithm uses std::setprecision(6) with the
        # default float format.  Match that exact serialized grid rather than
        # admitting a broad numerical tolerance that could hide a shifted row.
        serialized_expected = float(format(expected, ".6g"))
        tolerance = max(2.0e-12, 2.0e-12 * abs(serialized_expected))
        while (
            cursor < len(rows)
            and float(rows[cursor][0]) < serialized_expected - tolerance
        ):
            cursor += 1
        candidates = rows[cursor : cursor + 2]
        if not candidates:
            return None
        best_offset, best = min(
            enumerate(candidates),
            key=lambda item: abs(float(item[1][0]) - serialized_expected),
        )
        if abs(float(best[0]) - serialized_expected) > tolerance:
            return None
        selected.append(best)
        cursor += best_offset + 1
    return selected


def _write_history(
    path: Path, case: Mapping[str, Any], force_rows: Sequence[Sequence[float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    dt = float(case["time_step_s"])
    amplitude = float(case["amplitude_m"])
    frequency = float(case["frequency_hz"])
    omega = 2.0 * math.pi * frequency
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HISTORY_COLUMNS)
        for index, force in enumerate(force_rows, start=1):
            relative_time = index * dt
            displacement = amplitude * math.sin(omega * relative_time)
            velocity = amplitude * omega * math.cos(omega * relative_time)
            crossflow_force = float(force[2]) + float(force[5])
            writer.writerow(
                (
                    format(relative_time, ".17g"),
                    format(displacement, ".17g"),
                    format(velocity, ".17g"),
                    format(crossflow_force, ".17g"),
                )
            )
    temporary.replace(path)


def campaign_status(
    campaign: Path,
    *,
    write: bool = True,
    nalu_executable: Path | None = None,
) -> dict[str, Any]:
    campaign = campaign.expanduser().resolve()
    validate_campaign(campaign)
    manifest = _load_manifest(campaign)
    current_executable = (
        None if nalu_executable is None else _nalu_executable_identity(nalu_executable)
    )
    scaling_gate = _scaling_gate(
        campaign, manifest, current_nalu_executable=current_executable
    )
    bound_executable = scaling_gate["runtime_identity"]["nalu_executable"]
    precursor_status: list[dict[str, Any]] = []
    complete_precursors: set[str] = set()
    for precursor in manifest["precursors"]:
        log = campaign / precursor["log"]
        restart = campaign / precursor["restart_output"]
        maximum_step = _max_logged_step(log)
        failure = _log_failure(log)
        complete = (
            failure is None
            and maximum_step == int(precursor["precursor_steps"])
            and _terminal_completion_step(log) == int(precursor["precursor_steps"])
            and bool(_restart_files(restart))
        )
        if complete:
            if bound_executable is None:
                raise CampaignError(
                    f"Completed precursor {precursor['precursor_id']} lacks a bound naluX identity."
                )
            status = "complete"
            complete_precursors.add(precursor["precursor_id"])
            runtime_identity = _validated_log_runtime(
                log,
                expected_solver=manifest["solver"],
                executable=bound_executable,
                root=campaign,
                expected_terminal_step=int(precursor["precursor_steps"]),
            )
        elif failure:
            status = "failed"
            runtime_identity = None
        elif log.is_file():
            status = "running_or_incomplete"
            runtime_identity = None
        else:
            status = "pending"
            runtime_identity = None
        precursor_status.append(
            {
                "precursor_id": precursor["precursor_id"],
                "reynolds_number": precursor["reynolds_number"],
                "status": status,
                "maximum_logged_step": maximum_step,
                "required_step": precursor["precursor_steps"],
                "restart_file_count": len(_restart_files(restart)),
                "failure_marker": failure,
                "runtime_identity": runtime_identity,
            }
        )

    case_results: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        force_path = campaign / case["force_output"]
        log_path = campaign / case["log"]
        failure = _log_failure(log_path)
        rows = _force_rows(force_path)
        selected = _select_expected_force_rows(rows, case)
        expected_terminal_step = int(case["precursor_restart_step"]) + int(
            case["updates_after_restart"]
        )
        history = copy.deepcopy(case["canonical_history"])
        if failure:
            solver_status = "failed"
            runtime_identity = None
        elif (
            selected is not None
            and _terminal_completion_step(log_path) == expected_terminal_step
        ):
            if bound_executable is None:
                raise CampaignError(
                    f"Completed motion {case['case_id']} lacks a bound naluX identity."
                )
            solver_status = "complete"
            runtime_identity = _validated_log_runtime(
                log_path,
                expected_solver=manifest["solver"],
                executable=bound_executable,
                root=campaign,
                expected_terminal_step=expected_terminal_step,
            )
            history_path = campaign / history["path"]
            if write:
                _write_history(history_path, case, selected)
                history["status"] = "complete"
                history["sha256"] = _sha256(history_path)
                history["rows"] = len(selected)
            elif history_path.is_file() and not history_path.is_symlink():
                history["status"] = "complete"
                history["sha256"] = _sha256(history_path)
                history["rows"] = len(selected)
        elif log_path.is_file() or force_path.is_file():
            solver_status = "running_or_incomplete"
            runtime_identity = None
        else:
            solver_status = "pending"
            runtime_identity = None
        result = copy.deepcopy(case)
        result["solver_status"] = solver_status
        result["failure_marker"] = failure
        result["observed_force_rows"] = len(rows)
        result["force_output_sha256"] = (
            _sha256(force_path) if force_path.is_file() else None
        )
        result["selected_force_rows"] = 0 if selected is None else len(selected)
        result["raw_force_time_start_observed_s"] = (
            None if selected is None else float(selected[0][0])
        )
        result["raw_force_time_end_observed_s"] = (
            None if selected is None else float(selected[-1][0])
        )
        result["canonical_history"] = history
        result["runtime_identity"] = runtime_identity
        case_results.append(result)

    complete_cases = sum(row["solver_status"] == "complete" for row in case_results)
    failed_cases = sum(row["solver_status"] == "failed" for row in case_results)
    failed_precursors = sum(row["status"] == "failed" for row in precursor_status)
    preflight_id = manifest["preflight"]["motion_case_id"]
    preflight_result = next(
        row for row in case_results if row["case_id"] == preflight_id
    )
    mesh_audit = _mesh_audit_status(campaign, manifest)
    all_prerequisites = (
        mesh_audit["status"] == "pass"
        and scaling_gate["status"] == "pass"
        and len(complete_precursors) == len(manifest["precursors"])
    )
    if failed_cases or failed_precursors:
        overall = "failed"
    elif complete_cases == len(case_results) and all_prerequisites:
        overall = "complete"
    elif complete_cases:
        overall = "incomplete"
    else:
        overall = "awaiting_results"
    results = {
        "schema": RESULTS_SCHEMA,
        "status": overall,
        "campaign_fingerprint": manifest["fingerprint"],
        "campaign_manifest": "campaign_manifest.json",
        "request": manifest["request"],
        "mesh": manifest["mesh"],
        "solver": manifest["solver"],
        "fluid_and_section": manifest["fluid_and_section"],
        "force_contract": {
            "surface": manifest["numerics"]["force_surface"],
            "crossflow_expression": manifest["numerics"]["crossflow_force_expression"],
            "sign": "positive_fluid_on_body_global_positive_Y",
            "units": "N_total_over_declared_4m_span",
        },
        "history_contract": manifest["history_contract"],
        "mesh_audit": mesh_audit,
        "scaling_gate": scaling_gate,
        "precursors": precursor_status,
        "motion_preflight": {
            "case_id": preflight_id,
            "case_index": manifest["preflight"]["motion_case_index"],
            "status": preflight_result["solver_status"],
            "history_sha256": preflight_result["canonical_history"].get("sha256"),
        },
        "summary": {
            "complete_precursors": len(complete_precursors),
            "required_precursors": len(manifest["precursors"]),
            "complete_motion_cases": complete_cases,
            "failed_motion_cases": failed_cases,
            "failed_precursors": failed_precursors,
            "required_motion_cases": len(case_results),
        },
        "cases": case_results,
    }
    if write:
        _write_json(campaign / "nalu_results_manifest.json", results)
    return results


def _enforce_status_gates(
    status: Mapping[str, Any],
    *,
    allocated_nodes: int | None,
    require_mesh_audit: bool,
    require_scaling: bool,
    require_precursors: bool,
    require_motion_preflight: bool,
) -> None:
    if require_mesh_audit and status["mesh_audit"]["status"] != "pass":
        raise CampaignError("The Exodus part/geometry audit has not passed.")
    gate = status["scaling_gate"]
    if require_scaling and gate["status"] != "pass":
        raise CampaignError(
            "No complete 2/4/8-node scaling sweep yields a two-hour target or eight-hour fallback."
        )
    if allocated_nodes is not None:
        if gate["status"] != "pass":
            raise CampaignError(
                "Cannot check allocation before the scaling gate passes."
            )
        if int(allocated_nodes) != int(gate["recommended_nodes"]):
            raise CampaignError(
                f"Measured scaling selects {gate['recommended_nodes']} nodes per case "
                f"at tier {gate['selection_tier']}; "
                f"this allocation has {allocated_nodes}."
            )
    if require_precursors:
        summary = status["summary"]
        if summary["complete_precursors"] != summary["required_precursors"]:
            raise CampaignError(
                f"Precursors incomplete: {summary['complete_precursors']}/"
                f"{summary['required_precursors']}."
            )
    if require_motion_preflight and status["motion_preflight"]["status"] != "complete":
        raise CampaignError(
            f"Motion preflight {status['motion_preflight']['case_id']} is not complete."
        )


def _case_path(
    campaign: Path,
    *,
    stage: str,
    index: int | None,
    nodes: int | None,
    reynolds: float | None,
    skip_complete: bool,
) -> Path:
    manifest = _load_manifest(campaign)
    if stage == "scaling":
        if nodes is None or reynolds is None:
            raise CampaignError("Scaling path requires --nodes and --reynolds.")
        row = next(
            (
                item
                for item in manifest["preflight"]["scaling_cases"]
                if int(item["nodes"]) == nodes
                and float(item["reynolds_number"]) == float(reynolds)
            ),
            None,
        )
        if row is None:
            raise CampaignError("No matching generated scaling input.")
    elif stage == "precursor":
        if index is None or not 0 <= index < len(manifest["precursors"]):
            raise CampaignError("Precursor index is out of range 0..2.")
        row = manifest["precursors"][index]
        if skip_complete:
            status = campaign_status(campaign, write=False)
            current = status["precursors"][index]
            if current["status"] == "complete":
                raise SystemExit(10)
    elif stage == "motion":
        if index is None or not 0 <= index < len(manifest["cases"]):
            raise CampaignError("Motion index is out of range 0..47.")
        row = manifest["cases"][index]
        if skip_complete:
            status = campaign_status(campaign, write=False)
            current = status["cases"][index]
            if current["solver_status"] == "complete":
                raise SystemExit(10)
    elif stage == "motion-preflight":
        row = manifest["cases"][int(manifest["preflight"]["motion_case_index"])]
        if skip_complete:
            status = campaign_status(campaign, write=False)
            current = status["cases"][int(row["index"])]
            if current["solver_status"] == "complete":
                raise SystemExit(10)
    else:
        raise CampaignError(f"Unsupported stage: {stage}")
    return campaign / row["yaml"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate an immutable campaign")
    generate.add_argument("--campaign", type=Path, required=True)
    generate.add_argument("--mesh", type=Path, required=True)
    generate.add_argument("--mesh-sha256", required=True)

    validate = subparsers.add_parser("validate", help="Validate all frozen inputs")
    validate.add_argument("--campaign", type=Path, required=True)

    mesh_audit = subparsers.add_parser(
        "audit-mesh", help="Audit Exodus parts, chord orientation, and span"
    )
    mesh_audit.add_argument("--campaign", type=Path, required=True)

    status = subparsers.add_parser(
        "status", help="Audit outputs and write results manifest"
    )
    status.add_argument("--campaign", type=Path, required=True)
    status.add_argument("--no-write", action="store_true")
    status.add_argument("--allocated-nodes", type=int)
    status.add_argument("--require-mesh-audit", action="store_true")
    status.add_argument("--require-scaling", action="store_true")
    status.add_argument("--require-precursors", action="store_true")
    status.add_argument("--require-motion-preflight", action="store_true")
    status.add_argument("--nalu-executable", type=Path)

    scaling = subparsers.add_parser(
        "record-scaling", help="Record one executed scaling probe"
    )
    scaling.add_argument("--campaign", type=Path, required=True)
    scaling.add_argument("--nodes", type=int, required=True)
    scaling.add_argument("--reynolds", type=float, required=True)
    scaling.add_argument("--log", type=Path, required=True)
    scaling.add_argument("--elapsed-seconds", type=float, required=True)
    scaling.add_argument("--nalu-executable", type=Path, required=True)

    path = subparsers.add_parser("case-path", help="Print one generated Nalu YAML path")
    path.add_argument("--campaign", type=Path, required=True)
    path.add_argument(
        "--stage",
        choices=("scaling", "precursor", "motion", "motion-preflight"),
        required=True,
    )
    path.add_argument("--index", type=int)
    path.add_argument("--nodes", type=int)
    path.add_argument("--reynolds", type=float)
    path.add_argument("--skip-complete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            value = generate_campaign(args.campaign, args.mesh, args.mesh_sha256)
            output: object = {
                "campaign": str(args.campaign.expanduser().resolve()),
                "fingerprint": value["fingerprint"],
                "precursors": len(value["precursors"]),
                "motion_cases": len(value["cases"]),
                "motion_preflight_case": value["preflight"]["motion_case_id"],
            }
        elif args.command == "validate":
            output = validate_campaign(args.campaign)
        elif args.command == "audit-mesh":
            output = audit_mesh(args.campaign)
        elif args.command == "status":
            output = campaign_status(
                args.campaign,
                write=not args.no_write,
                nalu_executable=args.nalu_executable,
            )
            _enforce_status_gates(
                output,
                allocated_nodes=args.allocated_nodes,
                require_mesh_audit=args.require_mesh_audit,
                require_scaling=args.require_scaling,
                require_precursors=args.require_precursors,
                require_motion_preflight=args.require_motion_preflight,
            )
        elif args.command == "record-scaling":
            output = record_scaling(
                args.campaign,
                nodes=args.nodes,
                reynolds=args.reynolds,
                log=args.log,
                elapsed_seconds=args.elapsed_seconds,
                nalu_executable=args.nalu_executable,
            )
        elif args.command == "case-path":
            path = _case_path(
                args.campaign.expanduser().resolve(),
                stage=args.stage,
                index=args.index,
                nodes=args.nodes,
                reynolds=args.reynolds,
                skip_complete=args.skip_complete,
            )
            print(path.resolve())
            return 0
        else:  # pragma: no cover
            raise CampaignError(f"Unknown command: {args.command}")
    except CampaignError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
