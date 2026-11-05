#!/usr/bin/env python3
"""Assemble paired Nalu/QBlade histories into the work-map build manifest.

The two prescribed-motion campaigns are executed independently, but their
canonical histories must be byte-identical in time, displacement, and
velocity.  This command verifies that association, snapshots the exact input
bytes into one immutable bundle, and writes the manifest consumed by
``build_airfoil_work_map``.  It never evaluates work or reads the Ehrmann
free-response measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any, Mapping, Sequence

from examples.ehrmann2014_naca0018 import nalu_prescribed_motion as nalu_producer
from examples.ehrmann2014_naca0018 import qblade_prescribed_motion as qblade_producer

HERE = Path(__file__).resolve().parent
REQUEST_PATH = HERE / "work_map_data_request.json"
IDENTIFICATION_SCHEMA = "vorlap-airfoil-work-map-identification-manifest-v2"
BUNDLE_SCHEMA = "vorlap-naca0018-work-map-identification-bundle-v1"
NALU_RESULTS_SCHEMA = "vorlap-naca0018-nalu-prescribed-motion-results-v1"
QBLADE_RESULTS_SCHEMA = "vorlap-naca0018-qblade-prescribed-motion-results-v1"
HISTORY_COLUMNS = (
    "time_s",
    "displacement_m",
    "velocity_mps",
    "crossflow_force_n",
)
PHYSICAL_KEYS = (
    "fluid_density_kgpm3",
    "fluid_dynamic_viscosity_pas",
    "reference_chord_m",
    "section_span_m",
)
VERTEX_KEYS = (
    "incidence_deg",
    "reynolds_number",
    "reduced_frequency_k",
    "amplitude_ratio",
    "direction_deg_mod_180",
)
EXPECTED_REYNOLDS = (64_000.0, 100_000.0, 140_000.0)
EXPECTED_K = (0.45, 0.65, 0.8, 1.05)
EXPECTED_AMPLITUDES = (0.005, 0.0125, 0.025, 0.05)
EXPECTED_NALU_COMMIT = "ffa9de729df2a11b5241fdeb7628e7fab9f48f9b"
EXPECTED_QBLADE_LIBRARY_SHA256 = (
    "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
)
EXPECTED_QBLADE_DEPENDENCY_FINGERPRINT = (
    "c50c6909317c754caec4130ccbc8e2ed3f705f810f07da11e7ce3ad003499089"
)


class AssemblyError(RuntimeError):
    """Raised when paired evidence cannot form the frozen identification set."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AssemblyError(f"{label} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"Cannot parse {label}: {path}") from exc
    if not isinstance(value, dict):
        raise AssemblyError(f"{label} must contain a JSON object.")
    return value


def _relative_member(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise AssemblyError(f"{label} must be a relative path string.")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise AssemblyError(f"{label} must be a canonical relative path.")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise AssemblyError(f"{label} must identify a regular non-symlink file.")
    return path


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise AssemblyError(f"{label} must be finite.")
    return result


def _assert_close(actual: object, expected: float, label: str) -> float:
    value = _finite(actual, label)
    if not math.isclose(value, expected, rel_tol=2.0e-13, abs_tol=1.0e-14):
        raise AssemblyError(f"{label} changed: expected {expected!r}, found {value!r}.")
    return value


def _expected_physical() -> dict[str, float]:
    request = _read_json(REQUEST_PATH, "tracked data request")
    physical = request.get("frozen_fluid_and_section")
    if not isinstance(physical, Mapping):
        raise AssemblyError("Tracked request lacks frozen fluid/section values.")
    return {key: _finite(physical.get(key), f"tracked {key}") for key in PHYSICAL_KEYS}


def _validate_physical(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or any(key not in value for key in PHYSICAL_KEYS):
        raise AssemblyError(f"{label} lacks the frozen fluid/section contract.")
    expected = _expected_physical()
    observed = {key: _finite(value[key], f"{label} {key}") for key in PHYSICAL_KEYS}
    if observed != expected:
        raise AssemblyError(
            f"{label} fluid/section values differ from the tracked request."
        )
    return observed


def _identity_file(root: Path, identity: object, label: str) -> Path:
    if not isinstance(identity, Mapping):
        raise AssemblyError(f"{label} lacks a file identity.")
    path = _relative_member(root, identity.get("path"), label)
    if identity.get("sha256") != _sha256(path):
        raise AssemblyError(f"{label} SHA-256 differs.")
    if "bytes" in identity and identity.get("bytes") != path.stat().st_size:
        raise AssemblyError(f"{label} byte count differs.")
    return path


def _recorded_nalu_executable(identity: object, label: str) -> dict[str, str]:
    if not isinstance(identity, Mapping) or set(identity) != {"path", "sha256"}:
        raise AssemblyError(f"{label} must contain exact canonical path/SHA-256 keys.")
    path = identity.get("path")
    digest = identity.get("sha256")
    pure = PurePosixPath(path) if isinstance(path, str) else PurePosixPath()
    if (
        not isinstance(path, str)
        or not pure.is_absolute()
        or str(pure) != path
        or any(part in {"", ".", ".."} for part in pure.parts)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise AssemblyError(f"{label} executable identity is malformed.")
    return {"path": path, "sha256": digest}


def _validate_completed_nalu_runtime(
    root: Path,
    value: object,
    expected_executable: Mapping[str, str],
    expected_terminal_step: int,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "status",
        "nalu_executable",
        "observed_solver_version",
        "observed_solver_git_commit",
        "terminal_step",
        "log",
    }:
        raise AssemblyError(f"{label} runtime identity keys differ.")
    if (
        value.get("status") != "verified"
        or _recorded_nalu_executable(value.get("nalu_executable"), label)
        != dict(expected_executable)
        or value.get("observed_solver_version") != "v2.2.1"
        or value.get("observed_solver_git_commit") != EXPECTED_NALU_COMMIT
        or value.get("terminal_step") != expected_terminal_step
    ):
        raise AssemblyError(f"{label} runtime identity differs.")
    log = _identity_file(root, value.get("log"), f"{label} log")
    text = log.read_text(encoding="utf-8", errors="replace")
    if (
        set(nalu_producer.NALU_VERSION_RE.findall(text)) != {"v2.2.1"}
        or set(nalu_producer.NALU_COMMIT_RE.findall(text)) != {EXPECTED_NALU_COMMIT}
        or nalu_producer._terminal_completion_step(log) != expected_terminal_step
    ):
        raise AssemblyError(f"{label} log does not prove the exact Nalu runtime.")


def _load_campaign(root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = root / "campaign_manifest.json"
    return path, _read_json(path, f"{label} campaign manifest")


def _tag(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace("-", "m").replace(".", "p")


def _case_id(reynolds: float, reduced_frequency: float, amplitude_ratio: float) -> str:
    return (
        f"re{int(round(reynolds)):06d}_k{_tag(reduced_frequency, 2)}_"
        f"a{_tag(amplitude_ratio, 4)}"
    )


def _motion_values(
    reynolds: float,
    reduced_frequency: float,
    amplitude_ratio: float,
    physical: Mapping[str, float],
) -> dict[str, float | int]:
    density = physical["fluid_density_kgpm3"]
    viscosity = physical["fluid_dynamic_viscosity_pas"]
    chord = physical["reference_chord_m"]
    speed = reynolds * viscosity / (density * chord)
    frequency = reduced_frequency * speed / (math.pi * chord)
    period = 1.0 / frequency
    samples = math.ceil(max(100.0, 120.0 * 0.4 * math.pi / reduced_frequency))
    timestep = period / samples
    rows = 13 * samples + 1
    precursor_restart_time = 20.0 * math.pi * chord / speed
    precursor_maximum_dt = chord / (120.0 * 0.4 * speed)
    precursor_steps = math.ceil(precursor_restart_time / precursor_maximum_dt)
    return {
        "wind_speed_mps": speed,
        "frequency_hz": frequency,
        "period_s": period,
        "amplitude_m": amplitude_ratio * chord,
        "samples_per_cycle": samples,
        "time_step_s": timestep,
        "expected_result_rows": rows,
        "updates_after_restart": rows,
        "wrapper_steps": rows + 1,
        "relative_history_start_s": timestep,
        "relative_history_end_s": rows * timestep,
        "precursor_restart_time_s": precursor_restart_time,
        "precursor_steps": precursor_steps,
    }


def _validate_vertex_and_grid(
    row: Mapping[str, Any], physical: Mapping[str, float], label: str
) -> tuple[float, float, float, float, float, dict[str, float | int]]:
    vertex = tuple(_finite(row.get(key), f"{label} {key}") for key in VERTEX_KEYS)
    incidence, reynolds, reduced_frequency, amplitude_ratio, direction = vertex
    if (
        incidence != -90.0
        or reynolds not in EXPECTED_REYNOLDS
        or reduced_frequency not in EXPECTED_K
        or amplitude_ratio not in EXPECTED_AMPLITUDES
        or direction != 0.0
    ):
        raise AssemblyError(f"{label} is outside the frozen vertex grid.")
    expected_id = _case_id(reynolds, reduced_frequency, amplitude_ratio)
    if row.get("case_id") != expected_id:
        raise AssemblyError(
            f"{label} case_id is not canonical; expected {expected_id}."
        )
    values = _motion_values(reynolds, reduced_frequency, amplitude_ratio, physical)
    return (*vertex, values)


def _copy_snapshot(source: Path, target: Path, output: Path) -> dict[str, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "path": target.relative_to(output).as_posix(),
        "sha256": _sha256(target),
    }


def _case_index(payload: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 48:
        raise AssemblyError(f"{label} must contain exactly 48 cases.")
    result: dict[str, Mapping[str, Any]] = {}
    for row in cases:
        if not isinstance(row, Mapping):
            raise AssemblyError(f"{label} contains a malformed case row.")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in result:
            raise AssemblyError(f"{label} case IDs must be unique non-empty strings.")
        result[case_id] = row
    return result


def _history(
    root: Path, row: Mapping[str, Any], label: str
) -> tuple[Path, str, tuple[tuple[str, str, str], ...], int]:
    identity = row.get("canonical_history")
    if not isinstance(identity, Mapping):
        raise AssemblyError(f"{label} lacks canonical_history identity.")
    if identity.get("status") != "complete" or identity.get("columns") != list(
        HISTORY_COLUMNS
    ):
        raise AssemblyError(f"{label} canonical history is not complete or canonical.")
    path = _relative_member(root, identity.get("path"), f"{label} history")
    digest = _sha256(path)
    if identity.get("sha256") != digest:
        raise AssemblyError(f"{label} canonical history SHA-256 differs.")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != HISTORY_COLUMNS:
            raise AssemblyError(f"{label} history columns differ.")
        motion: list[tuple[str, str, str]] = []
        count = 0
        for line, item in enumerate(reader, start=2):
            if None in item or set(item) != set(HISTORY_COLUMNS):
                raise AssemblyError(f"{label} history row {line} is malformed.")
            for key in HISTORY_COLUMNS:
                try:
                    number = float(item[key])
                except (TypeError, ValueError) as exc:
                    raise AssemblyError(
                        f"{label} history row {line} {key} is not numeric."
                    ) from exc
                if not math.isfinite(number):
                    raise AssemblyError(
                        f"{label} history row {line} {key} is not finite."
                    )
            motion.append(
                (item["time_s"], item["displacement_m"], item["velocity_mps"])
            )
            count += 1
    if count != identity.get("rows") or count != identity.get("rows_expected"):
        raise AssemblyError(f"{label} canonical history row count differs.")
    return path, digest, tuple(motion), count


def _validate_results(
    path: Path, *, schema: str, label: str
) -> tuple[Path, dict[str, Any], dict[str, Mapping[str, Any]]]:
    path = path.expanduser().resolve()
    payload = _read_json(path, label)
    if payload.get("schema") != schema or payload.get("status") != "complete":
        raise AssemblyError(f"{label} is not a complete {schema} artifact.")
    request = payload.get("request")
    if not isinstance(request, Mapping) or request.get("sha256") != _sha256(
        REQUEST_PATH
    ):
        raise AssemblyError(f"{label} is not bound to the tracked data request.")
    _validate_physical(payload.get("fluid_and_section"), label)
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise AssemblyError(f"{label} lacks a results summary.")
    completed = summary.get("complete_motion_cases", summary.get("complete_cases"))
    failed = summary.get("failed_motion_cases", summary.get("failed_cases"))
    required = summary.get("required_motion_cases", summary.get("required_cases"))
    if (completed, failed, required) != (48, 0, 48):
        raise AssemblyError(f"{label} is not a clean complete 48-case result set.")
    return path.parent, payload, _case_index(payload, label)


def _validate_nalu_producer(
    root: Path,
    results: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        nalu_producer.validate_campaign(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise AssemblyError(f"Nalu campaign validation failed: {exc}") from exc
    campaign_path, campaign = _load_campaign(root, "Nalu")
    if campaign.get("schema") != nalu_producer.CAMPAIGN_SCHEMA:
        raise AssemblyError("Nalu campaign schema differs.")
    if results.get("campaign_manifest") != "campaign_manifest.json" or results.get(
        "campaign_fingerprint"
    ) != campaign.get("fingerprint"):
        raise AssemblyError("Nalu result/campaign identity differs.")
    if results.get("request") != campaign.get("request"):
        raise AssemblyError("Nalu result/campaign request identity differs.")
    physical = _validate_physical(results.get("fluid_and_section"), "Nalu results")
    if results.get("fluid_and_section") != campaign.get("fluid_and_section"):
        raise AssemblyError("Nalu result/campaign fluid identity differs.")

    solver = results.get("solver")
    required_solver = {
        "name": "Nalu-Wind",
        "version": "v2.2.1",
        "expected_git_commit": EXPECTED_NALU_COMMIT,
        "turbulence_model": "sst_iddes",
        "transition_model": "none",
    }
    if not isinstance(solver, Mapping) or any(
        solver.get(key) != value for key, value in required_solver.items()
    ):
        raise AssemblyError(
            "Nalu solver/model identity differs from the frozen contract."
        )
    if solver != campaign.get("solver"):
        raise AssemblyError("Nalu result/campaign solver identity differs.")

    mesh = results.get("mesh")
    if not isinstance(mesh, Mapping) or mesh != campaign.get("mesh"):
        raise AssemblyError("Nalu result/campaign mesh identity differs.")
    required_parts = [
        "fluid-hex",
        "wing",
        "wing-pp",
        "inlet",
        "outlet",
        "front",
        "back",
    ]
    if (
        mesh.get("transformation_xyz") != [0.23, 0.23, 1.0]
        or mesh.get("transformation_stage") != "precursor_only"
        or mesh.get("required_parts") != required_parts
    ):
        raise AssemblyError("Nalu mesh transformation or part contract differs.")
    _assert_close(mesh.get("source_chord_m"), 1.0, "Nalu source chord")
    _assert_close(mesh.get("source_span_m"), 4.0, "Nalu source span")

    expected_force = {
        "surface": ["wing", "wing-pp"],
        "crossflow_expression": "Fpy+Fvy",
        "sign": "positive_fluid_on_body_global_positive_Y",
        "units": "N_total_over_declared_4m_span",
    }
    if results.get("force_contract") != expected_force:
        raise AssemblyError("Nalu force surface/sign/units contract differs.")
    expected_history = {
        "time_origin": "precursor_restart",
        "first_row": "dt",
        "last_row": "(13*samples_per_cycle+1)*dt",
        "complete_cycle_intervals": 13,
        "raw_absolute_time_rule": "t0+j*dt for j=1..13*N+1",
        "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
        "columns": list(HISTORY_COLUMNS),
        "float_encoding": ".17g",
        "pairing_policy": "exact_array_equal_time_displacement_velocity_no_interpolation",
        "force_units": "N_total_over_4m_span",
    }
    if (
        results.get("history_contract") != expected_history
        or campaign.get("history_contract") != expected_history
    ):
        raise AssemblyError("Nalu canonical history contract differs.")

    mesh_identity = results.get("mesh_audit")
    mesh_audit_path = _identity_file(root, mesh_identity, "Nalu mesh audit")
    mesh_audit = _read_json(mesh_audit_path, "Nalu mesh audit")
    if (
        mesh_audit.get("schema") != nalu_producer.MESH_AUDIT_SCHEMA
        or mesh_audit.get("status") != "pass"
        or mesh_audit.get("campaign_fingerprint") != campaign.get("fingerprint")
        or mesh_audit.get("mesh", {}).get("sha256") != mesh.get("sha256")
        or mesh_audit.get("combined_force_surface") != ["wing", "wing-pp"]
        or mesh_audit.get("chord_axis") != "global_Y"
        or mesh_audit.get("incidence_orientation_deg") != -90.0
    ):
        raise AssemblyError("Nalu mesh audit does not prove the frozen geometry.")
    _assert_close(mesh_audit.get("source_chord_m"), 1.0, "Nalu audited source chord")
    _assert_close(mesh_audit.get("source_span_m"), 4.0, "Nalu audited source span")

    gate = results.get("scaling_gate")
    if not isinstance(gate, Mapping):
        raise AssemblyError("Nalu results lack the scaling gate.")
    recommended = gate.get("recommended_nodes")
    runtime_identity = gate.get("runtime_identity")
    expected_executable = (
        _recorded_nalu_executable(
            runtime_identity.get("nalu_executable"), "Nalu scaling gate"
        )
        if isinstance(runtime_identity, Mapping)
        else None
    )
    if (
        gate.get("status") != "pass"
        or gate.get("required_probe_count") != 3
        or gate.get("complete_probe_count") != 3
        or recommended not in (2, 4, 8)
        or gate.get("recommended_ranks") != int(recommended) * 112
        or gate.get("target_walltime_hours") != 2.0
        or gate.get("maximum_walltime_hours") != 8.0
        or not isinstance(runtime_identity, Mapping)
        or set(runtime_identity)
        != {
            "status",
            "observed_versions",
            "observed_git_commits",
            "nalu_executable",
        }
        or runtime_identity.get("status") != "verified"
        or runtime_identity.get("observed_versions") != ["v2.2.1"]
        or runtime_identity.get("observed_git_commits") != [EXPECTED_NALU_COMMIT]
        or expected_executable is None
    ):
        raise AssemblyError(
            "Nalu scaling/runtime gate differs from the frozen contract."
        )
    reports: list[tuple[int, Path]] = []
    scaling_rows = campaign.get("preflight", {}).get("scaling_cases")
    if not isinstance(scaling_rows, list) or len(scaling_rows) != 3:
        raise AssemblyError("Nalu campaign lacks the 2/4/8-node scaling sweep.")
    observed_nodes: set[int] = set()
    target_nodes: list[int] = []
    fallback_nodes: list[int] = []
    if not isinstance(gate.get("by_nodes"), Mapping) or set(gate["by_nodes"]) != {
        "2",
        "4",
        "8",
    }:
        raise AssemblyError("Nalu scaling gate node coverage differs.")
    for row in scaling_rows:
        nodes = int(row.get("nodes", -1))
        report_path = _relative_member(
            root, row.get("report"), f"Nalu {nodes}-node scaling report"
        )
        report = _read_json(report_path, f"Nalu {nodes}-node scaling report")
        log_path = _identity_file(
            root,
            {"path": report.get("log"), "sha256": report.get("log_sha256")},
            f"Nalu {nodes}-node scaling log",
        )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if (
            nodes not in (2, 4, 8)
            or report.get("schema") != nalu_producer.SCALING_SCHEMA
            or report.get("campaign_fingerprint") != campaign.get("fingerprint")
            or report.get("nodes") != nodes
            or report.get("ranks") != 112 * nodes
            or report.get("reynolds_number") != 64_000.0
            or report.get("probe_updates") != 64
            or report.get("observed_solver_version") != "v2.2.1"
            or report.get("observed_solver_git_commit") != EXPECTED_NALU_COMMIT
            or report.get("terminal_step") != nalu_producer.SCALING_PROBE_UPDATES
            or _recorded_nalu_executable(
                report.get("nalu_executable"), f"Nalu {nodes}-node scaling report"
            )
            != expected_executable
            or report.get("target_walltime_hours") != 2.0
            or report.get("maximum_walltime_hours") != 8.0
            or set(nalu_producer.NALU_VERSION_RE.findall(log_text)) != {"v2.2.1"}
            or set(nalu_producer.NALU_COMMIT_RE.findall(log_text))
            != {EXPECTED_NALU_COMMIT}
            or nalu_producer._terminal_completion_step(log_path)
            != nalu_producer.SCALING_PROBE_UPDATES
        ):
            raise AssemblyError(f"Nalu {nodes}-node scaling report differs.")
        projected = max(
            _finite(
                report.get("projected_precursor_hours"), "projected precursor hours"
            ),
            _finite(
                report.get("projected_longest_motion_hours"), "projected motion hours"
            ),
        )
        target_pass = projected <= 2.0
        maximum_pass = projected <= 8.0
        expected_status = "pass" if maximum_pass else "fail"
        if (
            report.get("status") != expected_status
            or report.get("target_walltime_pass") is not target_pass
            or report.get("maximum_walltime_pass") is not maximum_pass
        ):
            raise AssemblyError(f"Nalu {nodes}-node scaling verdict is inconsistent.")
        gate_row = gate.get("by_nodes", {}).get(str(nodes), {})
        if (
            gate_row.get("complete_reynolds_sweep") is not True
            or gate_row.get("reports") != 1
            or gate_row.get("target_walltime_pass") is not target_pass
            or gate_row.get("maximum_walltime_pass") is not maximum_pass
            or gate_row.get("pass") is not maximum_pass
        ):
            raise AssemblyError(f"Nalu {nodes}-node scaling gate row differs.")
        if target_pass:
            target_nodes.append(nodes)
        if maximum_pass:
            fallback_nodes.append(nodes)
        observed_nodes.add(nodes)
        reports.append((nodes, report_path))
    expected_tier = "target_2h" if target_nodes else "fallback_8h"
    selected_nodes = target_nodes if target_nodes else fallback_nodes
    expected_walltime = "02:00:00" if target_nodes else "08:00:00"
    if (
        observed_nodes != {2, 4, 8}
        or not selected_nodes
        or recommended != min(selected_nodes)
        or gate.get("selection_tier") != expected_tier
        or gate.get("recommended_walltime") != expected_walltime
    ):
        raise AssemblyError(
            "Nalu recommended node count is inconsistent with its reports."
        )

    precursors = results.get("precursors")
    if (
        not isinstance(precursors, list)
        or len(precursors) != 3
        or any(row.get("status") != "complete" for row in precursors)
    ):
        raise AssemblyError("Nalu precursors are not all complete.")
    campaign_precursors = campaign.get("precursors")
    if not isinstance(campaign_precursors, list) or len(campaign_precursors) != 3:
        raise AssemblyError("Nalu campaign precursor contract is incomplete.")
    campaign_precursors_by_id = {
        row.get("precursor_id"): row
        for row in campaign_precursors
        if isinstance(row, Mapping) and isinstance(row.get("precursor_id"), str)
    }
    if len(campaign_precursors_by_id) != 3:
        raise AssemblyError("Nalu campaign precursor IDs are incomplete.")
    for row in precursors:
        source = campaign_precursors_by_id.get(row.get("precursor_id"))
        if (
            source is None
            or row.get("required_step") != source.get("precursor_steps")
            or row.get("maximum_logged_step") != source.get("precursor_steps")
        ):
            raise AssemblyError("Nalu precursor terminal-step evidence differs.")
        _validate_completed_nalu_runtime(
            root,
            row.get("runtime_identity"),
            expected_executable,
            int(source["precursor_steps"]),
            f"Nalu precursor {row.get('precursor_id')}",
        )
    preflight = results.get("motion_preflight")
    expected_preflight = campaign.get("preflight", {})
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("status") != "complete"
        or preflight.get("case_id") != expected_preflight.get("motion_case_id")
        or preflight.get("case_index") != expected_preflight.get("motion_case_index")
    ):
        raise AssemblyError(
            "Nalu prescribed-motion preflight is incomplete or inconsistent."
        )
    if results.get("summary") != {
        "complete_precursors": 3,
        "required_precursors": 3,
        "complete_motion_cases": 48,
        "failed_motion_cases": 0,
        "failed_precursors": 0,
        "required_motion_cases": 48,
    }:
        raise AssemblyError("Nalu completion summary differs.")

    campaign_cases = _case_index(campaign, "Nalu campaign")
    for case_id, row in cases.items():
        source = campaign_cases.get(case_id)
        if source is None:
            raise AssemblyError(
                f"Nalu result case {case_id} is absent from its campaign."
            )
        *_, values = _validate_vertex_and_grid(source, physical, f"Nalu {case_id}")
        if (
            row.get("solver_status") != "complete"
            or row.get("failure_marker") is not None
        ):
            raise AssemblyError(f"Nalu case {case_id} is not a clean completed solve.")
        _validate_completed_nalu_runtime(
            root,
            row.get("runtime_identity"),
            expected_executable,
            int(source["precursor_restart_step"])
            + int(source["updates_after_restart"]),
            f"Nalu case {case_id}",
        )
        for key in (
            "wind_speed_mps",
            "frequency_hz",
            "period_s",
            "amplitude_m",
            "time_step_s",
            "relative_history_start_s",
            "relative_history_end_s",
            "precursor_restart_time_s",
        ):
            _assert_close(source.get(key), float(values[key]), f"Nalu {case_id} {key}")
        if (
            source.get("samples_per_cycle") != values["samples_per_cycle"]
            or source.get("updates_after_restart") != values["updates_after_restart"]
            or source.get("precursor_restart_step") != values["precursor_steps"]
            or row.get("precursor_restart_step") != values["precursor_steps"]
            or source.get("discard_cycles") != 5
            or source.get("retained_cycles") != 8
            or row.get("selected_force_rows") != values["updates_after_restart"]
        ):
            raise AssemblyError(f"Nalu case {case_id} derived grid differs.")
        for key in (*VERTEX_KEYS, "discard_cycles"):
            if row.get(key) != source.get(key):
                raise AssemblyError(
                    f"Nalu result/campaign case {case_id} {key} differs."
                )
        history = row.get("canonical_history")
        source_history = source.get("canonical_history")
        if (
            not isinstance(history, Mapping)
            or not isinstance(source_history, Mapping)
            or history.get("path") != source_history.get("path")
            or history.get("rows_expected") != values["updates_after_restart"]
            or history.get("rows") != values["updates_after_restart"]
            or history.get("columns") != list(HISTORY_COLUMNS)
            or history.get("status") != "complete"
        ):
            raise AssemblyError(f"Nalu case {case_id} history identity differs.")
    preflight_case = cases[str(preflight["case_id"])]
    if preflight.get("history_sha256") != preflight_case.get(
        "canonical_history", {}
    ).get("sha256"):
        raise AssemblyError("Nalu preflight history identity differs.")
    return {
        "campaign": campaign_path,
        "mesh_audit": mesh_audit_path,
        "scaling_reports": sorted(reports),
    }


def _validate_qblade_producer(
    root: Path,
    results: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        qblade_producer.validate_campaign(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise AssemblyError(f"QBlade campaign validation failed: {exc}") from exc
    campaign_path, campaign = _load_campaign(root, "QBlade")
    if campaign.get("schema") != qblade_producer.CAMPAIGN_SCHEMA:
        raise AssemblyError("QBlade campaign schema differs.")
    if results.get("campaign_manifest") != "campaign_manifest.json" or results.get(
        "campaign_fingerprint"
    ) != campaign.get("fingerprint"):
        raise AssemblyError("QBlade result/campaign identity differs.")
    if results.get("request") != campaign.get("request"):
        raise AssemblyError("QBlade result/campaign request identity differs.")
    physical = _validate_physical(results.get("fluid_and_section"), "QBlade results")
    if results.get("fluid_and_section") != campaign.get("fluid_and_section"):
        raise AssemblyError("QBlade result/campaign fluid identity differs.")

    runtime = results.get("runtime")
    if not isinstance(runtime, Mapping) or runtime != campaign.get("runtime"):
        raise AssemblyError("QBlade result/campaign runtime identity differs.")
    library = runtime.get("library")
    if (
        runtime.get("version") != "2.0.9.7"
        or not isinstance(library, Mapping)
        or library.get("sha256") != EXPECTED_QBLADE_LIBRARY_SHA256
        or runtime.get("runtime_dependency_fingerprint")
        != EXPECTED_QBLADE_DEPENDENCY_FINGERPRINT
    ):
        raise AssemblyError(
            "QBlade runtime/library identity differs from the frozen contract."
        )

    expected_geometry = {
        "profile": "NACA0018",
        "blade_count": 1,
        "station_count": 21,
        "chord_axis_global": [0.0, 1.0, 0.0],
        "normal_axis_global": [-1.0, 0.0, 0.0],
        "ambient_wind_global": [1.0, 0.0, 0.0],
        "qblade_internal_incidence_deg": 90.0,
        "vorlap_relative_wind_incidence_deg": -90.0,
        "mapping": (
            "QBlade +90 uses the opposite relative-wind sign convention from "
            "VorLap atan2(U.normal,U.chord)=-90; pitch and twist remain zero"
        ),
        "hub_y_equals_global_y": "zero_yaw_zero_shaft_tilt_zero_global_rotation",
    }
    expected_native = {
        "enabled": True,
        "dynamic_stall_type": 1,
        "unsteady_noncirculatory_aero": False,
        "two_point_lift_drag_correction": True,
        "wake_type": "unsteady_BEM",
        "bem_tip_loss": False,
        "tower_shadow": False,
        "rotation_rpm": 0.0,
        "internal_storage_filter": None,
        "export_filter": "results.filter_exact_10_channels",
        "filter_rationale": (
            "QBlade 2.0.9.7 silently omits the requested panel AoA/Re "
            "diagnostics when FILTERFILE is active during simulation; retain "
            "native channels internally and filter only at exportResults"
        ),
    }
    expected_external = {
        "controller_type": 0,
        "external_library_entries": 0,
        "sil_vorlap": False,
        "sil_constant_force": False,
        "expected_native_callback_updates": 0,
    }
    expected_history = {
        "first_row": "dt",
        "last_row": "(13*samples_per_cycle+1)*dt",
        "wrapper_steps_rule": "13*samples_per_cycle+2",
        "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
        "columns": list(HISTORY_COLUMNS),
        "float_encoding": ".17g",
        "force_channel": qblade_producer.FORCE_CHANNEL,
        "force_frame": "hub_Y_equals_global_crossflow_Y",
        "force_sign": "positive_fluid_on_body_in_positive_global_Y",
        "force_units": "N_total_over_4m_span",
    }
    for key, expected in (
        ("geometry_and_frames", expected_geometry),
        ("native_aero", expected_native),
        ("external_forces", expected_external),
    ):
        if results.get(key) != expected or campaign.get(key) != expected:
            raise AssemblyError(f"QBlade {key} contract differs.")
    if (
        campaign.get("history_contract") != expected_history
        or results.get("history_contract") != expected_history
    ):
        raise AssemblyError("QBlade canonical history/force contract differs.")
    execution = campaign.get("execution")
    if not isinstance(execution, Mapping) or (
        execution.get("threads_per_case"),
        execution.get("total_case_count"),
        execution.get("preflight_cpu_slots"),
        execution.get("matrix_cpu_slots"),
        execution.get("local_docker_max_cpu_slots"),
    ) != (1, 48, 1, 47, 8):
        raise AssemblyError("QBlade staged one-core execution contract differs.")
    expected_preflight_contract = {
        "case_index": qblade_producer.PREFLIGHT_CASE_INDEX,
        "case_id": qblade_producer.PREFLIGHT_CASE_ID,
        "selection_basis": (
            "maximum samples-per-cycle, maximum A/c, and maximum Reynolds "
            "number: longest-grid, largest-motion, high-Re representative"
        ),
        "required_before_matrix": True,
        "required_scheduler_schema": qblade_producer.PREFLIGHT_SCHEDULER_SCHEMA,
        "required_scheduler_stage": "preflight",
        "required_scheduler_status": "complete",
        "required_case_audit_schema": (
            "vorlap-naca0018-qblade-prescribed-motion-case-audit-v1"
        ),
    }
    if campaign.get("preflight_gate") != expected_preflight_contract:
        raise AssemblyError("QBlade preflight-gate campaign contract differs.")
    try:
        preflight_evidence = qblade_producer.validate_preflight_evidence(root, campaign)
    except (OSError, ValueError, RuntimeError) as exc:
        raise AssemblyError(
            f"QBlade preflight evidence validation failed: {exc}"
        ) from exc
    expected_preflight = {
        "campaign_contract": campaign.get("preflight_gate"),
        **preflight_evidence,
    }
    if results.get("preflight_gate") != expected_preflight:
        raise AssemblyError("QBlade result/preflight evidence identity differs.")
    preflight_report_path = _identity_file(
        root,
        preflight_evidence.get("scheduler_report"),
        "QBlade preflight scheduler report",
    )
    if results.get("summary") != {
        "required_cases": 48,
        "complete_cases": 48,
        "failed_cases": 0,
    }:
        raise AssemblyError("QBlade completion summary differs.")

    campaign_cases = _case_index(campaign, "QBlade campaign")
    audits: list[tuple[str, Path]] = []
    for case_id, row in cases.items():
        source = campaign_cases.get(case_id)
        if source is None:
            raise AssemblyError(
                f"QBlade result case {case_id} is absent from its campaign."
            )
        *_, values = _validate_vertex_and_grid(source, physical, f"QBlade {case_id}")
        if source.get("qblade_internal_incidence_deg") != 90.0:
            raise AssemblyError(f"QBlade case {case_id} incidence mapping differs.")
        for key in (
            "wind_speed_mps",
            "frequency_hz",
            "period_s",
            "amplitude_m",
            "time_step_s",
            "relative_history_start_s",
            "relative_history_end_s",
        ):
            _assert_close(
                source.get(key), float(values[key]), f"QBlade {case_id} {key}"
            )
        if (
            source.get("samples_per_cycle") != values["samples_per_cycle"]
            or source.get("expected_result_rows") != values["expected_result_rows"]
            or source.get("wrapper_steps") != values["wrapper_steps"]
            or source.get("discard_cycles") != 5
            or source.get("retained_cycles") != 8
        ):
            raise AssemblyError(f"QBlade case {case_id} derived grid differs.")
        for key in (*VERTEX_KEYS, "discard_cycles"):
            if row.get(key) != source.get(key):
                raise AssemblyError(
                    f"QBlade result/campaign case {case_id} {key} differs."
                )
        history = row.get("canonical_history")
        source_history = source.get("canonical_history")
        if (
            not isinstance(history, Mapping)
            or not isinstance(source_history, Mapping)
            or history.get("path") != source_history.get("path")
            or history.get("status") != "complete"
            or history.get("rows") != values["expected_result_rows"]
            or history.get("rows_expected") != values["expected_result_rows"]
            or history.get("columns") != list(HISTORY_COLUMNS)
        ):
            raise AssemblyError(f"QBlade case {case_id} history identity differs.")
        audit_path = _identity_file(
            root, row.get("case_audit"), f"QBlade {case_id} case audit"
        )
        audit = _read_json(audit_path, f"QBlade {case_id} case audit")
        expected_runtime = {
            "qblade_library_sha256": EXPECTED_QBLADE_LIBRARY_SHA256,
            "qblade_runtime_dependency_fingerprint": EXPECTED_QBLADE_DEPENDENCY_FINGERPRINT,
            "threads": 1,
            "coupling_mode": "none",
            "external_callback_updates": 0,
        }
        expected_channels = {
            "raw_export_rows": values["wrapper_steps"],
            "discarded_wrapper_t0_rows": 1,
            "prescribed_motion": qblade_producer.PLATFORM_Y_CHANNEL,
            "rigid_blade_position": qblade_producer.BLADE_TIP_Y_CHANNEL,
            "crossflow_force": qblade_producer.FORCE_CHANNEL,
            "crossflow_force_units": "N_total_over_4m_span",
            "crossflow_force_sign": "positive_fluid_on_body_positive_global_Y",
            "qblade_aoa": qblade_producer.AOA_CHANNEL,
            "qblade_reynolds": qblade_producer.REYNOLDS_CHANNEL,
        }
        if (
            audit.get("schema")
            != "vorlap-naca0018-qblade-prescribed-motion-case-audit-v1"
            or audit.get("status") != "complete"
            or audit.get("case_id") != case_id
            or audit.get("campaign_fingerprint") != campaign.get("fingerprint")
            or audit.get("runtime") != expected_runtime
            or audit.get("channel_contract") != expected_channels
            or audit.get("canonical_history") != history
        ):
            raise AssemblyError(f"QBlade case audit contract differs for {case_id}.")
        for identity_name in ("run_summary", "raw_result"):
            _identity_file(
                root, audit.get(identity_name), f"QBlade {case_id} {identity_name}"
            )
        frame = audit.get("geometry_frame_audit")
        if not isinstance(frame, Mapping):
            raise AssemblyError(f"QBlade case {case_id} lacks frame audit evidence.")
        if (
            frame.get("qblade_internal_incidence_target_deg") != 90.0
            or frame.get("vorlap_incidence_deg") != -90.0
            or abs(
                _finite(frame.get("median_abs_qblade_aoa_deg"), "QBlade median AoA")
                - 90.0
            )
            > 2.0
            or _finite(
                frame.get("max_abs_qblade_aoa_deviation_from_90_deg"),
                "QBlade AoA deviation",
            )
            > 12.0
            or abs(
                _finite(frame.get("median_qblade_panel_reynolds"), "QBlade Reynolds")
                - float(source["reynolds_number"])
            )
            / float(source["reynolds_number"])
            > 0.03
            or _finite(
                frame.get("analytic_platform_max_abs_error_m"), "QBlade motion error"
            )
            > max(2.0e-9, 2.0e-5 * float(source["amplitude_m"]))
            or _finite(
                frame.get("tip_follows_platform_max_abs_error_m"),
                "QBlade rigid-tip error",
            )
            > 2.0e-6
        ):
            raise AssemblyError(f"QBlade case {case_id} frame/motion audit differs.")
        statistics = audit.get("force_statistics_n")
        if not isinstance(statistics, Mapping) or any(
            not math.isfinite(
                _finite(statistics.get(key), f"QBlade {case_id} force {key}")
            )
            for key in ("minimum", "maximum", "mean", "rms")
        ):
            raise AssemblyError(f"QBlade case {case_id} force statistics are invalid.")
        audits.append((case_id, audit_path))
    return {
        "campaign": campaign_path,
        "preflight_scheduler_report": preflight_report_path,
        "case_audits": sorted(audits),
    }


def assemble(
    *,
    nalu_results: Path,
    qblade_results: Path,
    static_spectrum: Path,
    output: Path,
) -> Path:
    """Snapshot and associate the complete prescribed-motion evidence bundle."""

    nalu_root, nalu, high_cases = _validate_results(
        nalu_results,
        schema=NALU_RESULTS_SCHEMA,
        label="Nalu results manifest",
    )
    qblade_root, qblade, base_cases = _validate_results(
        qblade_results,
        schema=QBLADE_RESULTS_SCHEMA,
        label="QBlade results manifest",
    )
    if set(high_cases) != set(base_cases):
        raise AssemblyError("Nalu and QBlade case IDs differ.")
    nalu_provenance = _validate_nalu_producer(nalu_root, nalu, high_cases)
    qblade_provenance = _validate_qblade_producer(qblade_root, qblade, base_cases)
    if any(
        _finite(nalu["fluid_and_section"][key], f"Nalu {key}")
        != _finite(qblade["fluid_and_section"][key], f"QBlade {key}")
        for key in PHYSICAL_KEYS
    ):
        raise AssemblyError("Nalu and QBlade fluid/section identities differ.")

    spectrum = static_spectrum.expanduser().resolve()
    if spectrum.is_symlink() or not spectrum.is_file():
        raise AssemblyError(
            "The stationary spectrum must be a regular non-symlink file."
        )
    output = output.expanduser().resolve()
    if output.exists():
        raise AssemblyError(f"Refusing to replace an existing bundle: {output}")
    manifest_cases: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []
    try:
        (output / "histories" / "high_fidelity").mkdir(parents=True)
        (output / "histories" / "exact_runtime_base").mkdir(parents=True)
        shutil.copy2(REQUEST_PATH, output / REQUEST_PATH.name)
        spectrum_target = output / "stationary_spectrum" / spectrum.name
        spectrum_target.parent.mkdir()
        shutil.copy2(spectrum, spectrum_target)

        nalu_snapshot_root = output / "provenance" / "nalu"
        qblade_snapshot_root = output / "provenance" / "qblade"
        high_provenance = {
            "kind": "nalu_wind_prescribed_motion",
            "results_manifest": _copy_snapshot(
                Path(nalu_results).expanduser().resolve(),
                nalu_snapshot_root / "nalu_results_manifest.json",
                output,
            ),
            "campaign_manifest": _copy_snapshot(
                nalu_provenance["campaign"],
                nalu_snapshot_root / "campaign_manifest.json",
                output,
            ),
            "mesh_audit": _copy_snapshot(
                nalu_provenance["mesh_audit"],
                nalu_snapshot_root / "mesh_audit.json",
                output,
            ),
            "scaling_reports": [
                {
                    "nodes": nodes,
                    "file": _copy_snapshot(
                        path,
                        nalu_snapshot_root / f"scaling_nodes{nodes}.json",
                        output,
                    ),
                }
                for nodes, path in nalu_provenance["scaling_reports"]
            ],
        }
        base_provenance = {
            "kind": "native_qblade_prescribed_motion",
            "results_manifest": _copy_snapshot(
                Path(qblade_results).expanduser().resolve(),
                qblade_snapshot_root / "qblade_results.json",
                output,
            ),
            "campaign_manifest": _copy_snapshot(
                qblade_provenance["campaign"],
                qblade_snapshot_root / "campaign_manifest.json",
                output,
            ),
            "preflight_scheduler_report": _copy_snapshot(
                qblade_provenance["preflight_scheduler_report"],
                qblade_snapshot_root / "preflight_scheduler_report.json",
                output,
            ),
            "case_audits": [
                {
                    "case_id": case_id,
                    "file": _copy_snapshot(
                        path,
                        qblade_snapshot_root / "case_audits" / f"{case_id}.json",
                        output,
                    ),
                }
                for case_id, path in qblade_provenance["case_audits"]
            ],
        }
        producer_provenance = {
            "high_fidelity": high_provenance,
            "exact_runtime_base": base_provenance,
        }
        for case_id in sorted(high_cases):
            high = high_cases[case_id]
            base = base_cases[case_id]
            high_vertex = tuple(
                _finite(high.get(key), f"{case_id} Nalu {key}") for key in VERTEX_KEYS
            )
            base_vertex = tuple(
                _finite(base.get(key), f"{case_id} QBlade {key}") for key in VERTEX_KEYS
            )
            if high_vertex != base_vertex:
                raise AssemblyError(f"Paired vertex differs for {case_id}.")
            high_path, high_sha, high_motion, high_rows = _history(
                nalu_root, high, f"{case_id} Nalu"
            )
            base_path, base_sha, base_motion, base_rows = _history(
                qblade_root, base, f"{case_id} QBlade"
            )
            if high_motion != base_motion or high_rows != base_rows:
                raise AssemblyError(
                    f"Paired time/displacement/velocity bytes differ for {case_id}."
                )
            high_target = output / "histories" / "high_fidelity" / f"{case_id}.csv"
            base_target = output / "histories" / "exact_runtime_base" / f"{case_id}.csv"
            shutil.copy2(high_path, high_target)
            shutil.copy2(base_path, base_target)
            discard = high.get("discard_cycles")
            if type(discard) is not int or discard != base.get("discard_cycles"):
                raise AssemblyError(
                    f"Paired discard-cycle policy differs for {case_id}."
                )
            manifest_cases.append(
                {
                    "case_id": case_id,
                    **{
                        key: high_vertex[index] for index, key in enumerate(VERTEX_KEYS)
                    },
                    "discard_cycles": discard,
                    "high_fidelity_history": {
                        "path": high_target.relative_to(output).as_posix(),
                        "sha256": high_sha,
                    },
                    "exact_runtime_base_history": {
                        "path": base_target.relative_to(output).as_posix(),
                        "sha256": base_sha,
                    },
                }
            )
            bundle_rows.append(
                {
                    "case_id": case_id,
                    "vertex": list(high_vertex),
                    "rows": high_rows,
                    "motion_grid_sha256": hashlib.sha256(
                        json.dumps(high_motion, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "high_fidelity_sha256": high_sha,
                    "exact_runtime_base_sha256": base_sha,
                }
            )

        physical = {
            key: _finite(nalu["fluid_and_section"][key], key) for key in PHYSICAL_KEYS
        }
        manifest = {
            "schema": IDENTIFICATION_SCHEMA,
            "status": "complete",
            "profile": "NACA0018",
            "request": {
                "path": REQUEST_PATH.name,
                "sha256": _sha256(output / REQUEST_PATH.name),
            },
            "static_spectrum": {
                "path": spectrum_target.relative_to(output).as_posix(),
                "sha256": _sha256(spectrum_target),
            },
            "producer_provenance": producer_provenance,
            "fluid_and_section": {
                **physical,
                "static_spectrum_frequency_depth": 200,
            },
            "cases": manifest_cases,
        }
        manifest_path = output / "identification_manifest.json"
        _write_json(manifest_path, manifest)
        bundle = {
            "schema": BUNDLE_SCHEMA,
            "status": "complete",
            "identification_manifest": {
                "path": manifest_path.name,
                "sha256": _sha256(manifest_path),
            },
            "producer_provenance": producer_provenance,
            "case_count": len(bundle_rows),
            "cases": bundle_rows,
            "claim_boundary": {
                "ehrmann_free_response_used": False,
                "static_fft_phase_used_for_work": False,
                "incremental_force": "high_fidelity_minus_exact_runtime_base",
            },
        }
        _write_json(output / "bundle_report.json", bundle)
    except BaseException:
        # Leave evidence intact only when it is internally complete.  This root
        # was created by this function and did not exist on entry.
        shutil.rmtree(output)
        raise
    return output / "identification_manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nalu-results", type=Path, required=True)
    parser.add_argument("--qblade-results", type=Path, required=True)
    parser.add_argument("--static-spectrum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = assemble(
            nalu_results=args.nalu_results,
            qblade_results=args.qblade_results,
            static_spectrum=args.static_spectrum,
            output=args.output,
        )
    except AssemblyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
