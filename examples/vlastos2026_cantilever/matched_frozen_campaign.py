#!/usr/bin/env python3
"""Build and assess a legacy frozen-phase bridge comparator campaign.

This workflow derives one fine-timestep legacy frozen-phase comparator case
from every
``(Scruton number, reduced velocity)`` pair in a semantically validated
stationary response-aware campaign.  Geometry, structure, inflow, QBlade/VorLap
mapping, and the ``CYLINDER.h5`` source are copied byte-for-byte.  The only
runtime-model changes are ``force_scale: 0 -> 1``, response-model disablement,
and startup-seed disablement.  Simulation duration is replaced by a
predeclared, exactly aligned 240-natural-cycle cap.

Generation and run planning never execute QBlade.  Analysis is deliberately a
matched amplitude-skill diagnostic: it compares terminal stationary response
amplitudes with hGAST/experiment data where locally provenance-bound, the
legacy frozen-phase bridge comparator, and zero VIV.  It does not establish
conservative
fatigue loads, experimental validation, or certification suitability.

"Frozen" describes the prescribed source spectrum/phase law, not a strict
open-loop force history: the legacy VorLap load path still uses QBlade
structural velocity in ``U_relative = U_inflow - U_structure`` and therefore
retains kinematic force-magnitude feedback.  The response-aware modal closure
and its seed are the portions disabled here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import response_benchmark as response
import stationary_campaign_protocol as stationary


SCHEMA = "vlastos2026-legacy-frozen-phase-bridge-campaign-v1"
RUN_PLAN_SCHEMA = "vlastos2026-legacy-frozen-phase-bridge-run-plan-v1"
ANALYSIS_SCHEMA = "vlastos2026-legacy-frozen-phase-bridge-analysis-v1"
PARENT_MANIFEST_SCHEMA = response.SCHEMA
CAP_NATURAL_CYCLES = 240
TERMINAL_WINDOW_CYCLES = 40.0
THREADS_PER_RUN = 4
EXECUTION_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": str(THREADS_PER_RUN),
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
FORCE_SCALE = 1.0
EXPECTED_CASE_COUNT = 36
EXPECTED_NODE_COUNT = 21
EXPECTED_FORCE_COMPONENT_COUNT = 3 * EXPECTED_NODE_COUNT
COMPACT_HISTORY_SCHEMA = "vorlap-sil-force-history-compact-v1"
FORCE_VECTOR_DIGEST_SCHEMA = "vorlap-force-vector-sha256-v1"
APPLIED_POWER_SIGN_CONVENTION = (
    "positive_when_applied_force_does_work_on_current_structural_velocity"
)
CAMPAIGN_SCOPE = (
    "legacy frozen-phase bridge comparator amplitude-skill diagnostic with retained "
    "kinematic relative-velocity feedback; not strict open-loop one-way, "
    "fatigue-conservative validation, resolved-wake validation, or certification evidence"
)


class MatchedFrozenError(ValueError):
    """Raised when immutable legacy frozen-phase evidence does not validate."""


def frozen_endpoint_for_cycles(scruton_number: float) -> dict[str, Any]:
    """Return an aligned endpoint whose *exported sample span* covers 240 cycles.

    QBlade exports ``NUMTIMESTEPS`` samples at ``0, dt, ..., (N-1)dt``.  The
    stationary protocol's shared endpoint is based on ``N*dt``.  Add one
    nominal step (two fine steps) only when needed so the last exported sample,
    rather than merely the post-step endpoint, reaches the declared cap.
    """

    sc = next(
        (
            float(value)
            for value in stationary.STRUCTURES
            if math.isclose(
                float(scruton_number), float(value), rel_tol=0.0, abs_tol=1.0e-12
            )
        ),
        None,
    )
    if sc is None:
        raise MatchedFrozenError(f"Unsupported Scruton number: {scruton_number!r}")
    base = stationary.endpoint_for_cycles(sc, CAP_NATURAL_CYCLES)
    coarse_steps = int(base["dt_steps"])
    frequency_numerator, frequency_denominator = stationary.STRUCTURES[sc][
        "natural_frequency_fraction"
    ]
    target_integer = CAP_NATURAL_CYCLES * 1_000_000_000 * int(
        frequency_denominator
    )
    while (
        (2 * coarse_steps - 1)
        * stationary.FINE_TIMESTEP_NS
        * int(frequency_numerator)
        < target_integer
    ):
        coarse_steps += 1
    fine_steps = 2 * coarse_steps
    endpoint_time_ns = coarse_steps * stationary.COARSE_TIMESTEP_NS
    observed_span_numerator = (
        (fine_steps - 1)
        * stationary.FINE_TIMESTEP_NS
        * int(frequency_numerator)
    )
    observed_span_cycles = observed_span_numerator / (
        1_000_000_000 * int(frequency_denominator)
    )
    if observed_span_numerator < target_integer:
        raise MatchedFrozenError("Internal exported-span alignment error.")
    return {
        "evaluation_cycles": CAP_NATURAL_CYCLES,
        "endpoint_time_ns": endpoint_time_ns,
        "endpoint_time_s": endpoint_time_ns / 1.0e9,
        "dt_steps": coarse_steps,
        "dt_half_steps": fine_steps,
        "last_exported_sample_time_s": (fine_steps - 1)
        * stationary.FINE_TIMESTEP_S,
        "last_exported_sample_natural_cycles": observed_span_cycles,
        "fine_steps_added_beyond_endpoint_ceiling": fine_steps
        - int(base["dt_half_steps"]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchedFrozenError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise MatchedFrozenError(f"{label} must contain one JSON object: {path}")
    return value


def _safe_relative_path(root: Path, raw: Any, label: str) -> tuple[Path, str]:
    text = str(raw)
    relative = PurePosixPath(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise MatchedFrozenError(f"{label} has an unsafe relative path: {text!r}.")
    candidate = root.joinpath(*relative.parts)
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise MatchedFrozenError(f"{label} escapes its campaign root: {text!r}.") from exc
    return candidate, relative.as_posix()


def _identity(path: Path, *, relative: str | None = None) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": relative if relative is not None else path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _reject_write_within_immutable_roots(
    path: Path,
    immutable_roots: Iterable[Path],
    *,
    label: str,
) -> None:
    candidate = Path(path).expanduser().resolve()
    for raw_root in immutable_roots:
        root = Path(raw_root).expanduser().resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        raise MatchedFrozenError(
            f"{label} may not be inside immutable campaign inputs: {candidate}"
        )


def _reject_plan_within_results(path: Path, results_root: Path) -> None:
    candidate = Path(path).expanduser().resolve()
    root = Path(results_root).expanduser().resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return
    raise MatchedFrozenError(
        f"Run-plan output may not be inside the case-results root: {candidate}"
    )


def _manifest_file_rows(root: Path) -> list[dict[str, Any]]:
    excluded = {"campaign_manifest.json"}
    return [
        _identity(path, relative=path.relative_to(root).as_posix())
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name not in excluded
    ]


def _execution_input_record() -> dict[str, dict[str, Any]]:
    paths = {
        "runner": REPO_ROOT / "run_headless" / "run_qblade_sim.py",
        "result_filter": HERE / "cantilever_results.filter",
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise MatchedFrozenError(f"Required execution input is unavailable: {path}")
        result[name] = _identity(
            path, relative=path.relative_to(REPO_ROOT).as_posix()
        )
    return result


def _manifest_file_index(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    raw_rows = manifest.get("files")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise MatchedFrozenError(
            "Legacy frozen-phase manifest has no immutable file list."
        )
    result: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise MatchedFrozenError(
                "Legacy frozen-phase file identity is not an object."
            )
        path, relative = _safe_relative_path(root, raw.get("path"), "Manifest file")
        if relative in result:
            raise MatchedFrozenError(f"Duplicate manifest file identity: {relative}")
        if not path.is_file():
            raise MatchedFrozenError(f"Manifest file is missing: {relative}")
        try:
            expected_bytes = int(raw.get("bytes", -1))
        except (TypeError, ValueError) as exc:
            raise MatchedFrozenError(f"Invalid byte count for {relative}.") from exc
        actual = _identity(path, relative=relative)
        if expected_bytes != actual["bytes"] or raw.get("sha256") != actual["sha256"]:
            raise MatchedFrozenError(f"Immutable file identity changed: {relative}")
        result[relative] = actual
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "campaign_manifest.json"
    }
    if actual_paths != set(result):
        raise MatchedFrozenError(
            "Legacy frozen-phase input tree contains undeclared or missing files: "
            f"undeclared={sorted(actual_paths - set(result))}, "
            f"missing={sorted(set(result) - actual_paths)}."
        )
    return result


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise MatchedFrozenError(f"Refusing to replace existing output: {destination}") from exc
    return destination


_QBLADE_SETTING = re.compile(r"^(?P<lead>\s*)(?P<value>\S+)(?P<tail>\s+{name}\b.*)$")


def _replace_unique_qblade_setting(path: Path, name: str, value: Any) -> None:
    text = Path(path).read_text(encoding="utf-8")
    Path(path).write_text(
        _replace_qblade_setting_text(text, name, value), encoding="utf-8"
    )


def _replace_qblade_setting_text(text: str, name: str, value: Any) -> str:
    """Return one setting replacement without mutating an immutable input tree."""

    pattern = re.compile(_QBLADE_SETTING.pattern.format(name=re.escape(name)))
    lines = str(text).splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if pattern.match(line.rstrip("\r\n"))]
    if len(matches) != 1:
        raise MatchedFrozenError(
            f"Expected exactly one {name} setting; found {len(matches)}."
        )
    index = matches[0]
    ending = "\r\n" if lines[index].endswith("\r\n") else ("\n" if lines[index].endswith("\n") else "")
    body = lines[index][: -len(ending)] if ending else lines[index]
    match = pattern.match(body)
    if match is None:  # pragma: no cover - protected by the match search
        raise MatchedFrozenError(f"Cannot replace {name}.")
    lines[index] = f"{match.group('lead')}{value}{match.group('tail')}{ending}"
    return "".join(lines)


def _read_unique_qblade_setting(path: Path, name: str) -> str:
    pattern = re.compile(_QBLADE_SETTING.pattern.format(name=re.escape(name)))
    matches = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is not None:
            matches.append(match.group("value"))
    if len(matches) != 1:
        raise MatchedFrozenError(
            f"Expected exactly one {name} setting in {path}; found {len(matches)}."
        )
    return matches[0]


def _config_delta_paths(parent: Any, frozen: Any, prefix: str = "") -> set[str]:
    if isinstance(parent, Mapping) and isinstance(frozen, Mapping):
        if set(parent) != set(frozen):
            return {prefix or "<root-key-schema>"}
        result: set[str] = set()
        for key in parent:
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_config_delta_paths(parent[key], frozen[key], child_prefix))
        return result
    if isinstance(parent, list) and isinstance(frozen, list):
        if len(parent) != len(frozen):
            return {prefix}
        result = set()
        for index, (left, right) in enumerate(zip(parent, frozen)):
            result.update(_config_delta_paths(left, right, f"{prefix}[{index}]"))
        return result
    return set() if parent == frozen else {prefix}


EXPECTED_CONFIG_DELTAS = {
    "force_scale",
    "response_model.enabled",
    "response_model.startup_seed.enabled",
}


def _disabled_frozen_config(parent_config: Mapping[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(parent_config, allow_nan=False))
    response_model = config.get("response_model")
    if not isinstance(response_model, dict):
        raise MatchedFrozenError("Parent runtime config has no response_model object.")
    startup_seed = response_model.get("startup_seed")
    if not isinstance(startup_seed, dict):
        raise MatchedFrozenError("Parent response model has no startup_seed object.")
    if config.get("force_scale") != 0.0:
        raise MatchedFrozenError("Parent response case does not have force_scale=0.")
    if response_model.get("enabled") is not True or startup_seed.get("enabled") is not True:
        raise MatchedFrozenError("Parent response case is not response/seed enabled.")
    config["force_scale"] = FORCE_SCALE
    response_model["enabled"] = False
    startup_seed["enabled"] = False
    if _config_delta_paths(parent_config, config) != EXPECTED_CONFIG_DELTAS:
        raise MatchedFrozenError("Frozen runtime config changed outside the declared three fields.")
    return config


def _source_identity_from_config(
    config: Mapping[str, Any], source_path: Path
) -> dict[str, Any]:
    source_manifest = config.get("airfoil_source_manifest")
    if not isinstance(source_manifest, list) or len(source_manifest) != 1:
        raise MatchedFrozenError("Runtime config must contain exactly one airfoil source identity.")
    row = source_manifest[0]
    if not isinstance(row, Mapping) or row.get("airfoil_id") != "CYLINDER" or row.get("file") != "CYLINDER.h5":
        raise MatchedFrozenError("Runtime source identity is not the frozen CYLINDER source.")
    actual = _identity(source_path)
    if row.get("sha256") != actual["sha256"] or int(row.get("bytes", -1)) != actual["bytes"]:
        raise MatchedFrozenError("Runtime source fingerprint does not match CYLINDER.h5.")
    return {
        "airfoil_id": "CYLINDER",
        "file": "CYLINDER.h5",
        "sha256": actual["sha256"],
        "bytes": actual["bytes"],
    }


def _parent_pair_index(protocol: Mapping[str, Any]) -> dict[tuple[float, float], Mapping[str, Any]]:
    rows = protocol.get("case_pairs")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASE_COUNT:
        raise MatchedFrozenError("Parent stationary protocol does not contain 36 case pairs.")
    result: dict[tuple[float, float], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise MatchedFrozenError("Parent stationary pair is not an object.")
        key = (float(row["scruton_number"]), float(row["reduced_velocity"]))
        if key in result:
            raise MatchedFrozenError(f"Duplicate parent stationary pair: {key}")
        result[key] = row
    return result


def _fine_parent_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = [
        entry
        for entry in manifest.get("cases", [])
        if isinstance(entry, Mapping) and entry.get("timestep_level") == "dt_half"
    ]
    entries.sort(key=lambda row: (float(row["scruton_number"]), float(row["reduced_velocity"])))
    expected_grid = [
        (float(sc), float(u_star))
        for sc in sorted(stationary.STRUCTURES)
        for u_star in stationary.REDUCED_VELOCITIES
    ]
    observed_grid = [
        (float(entry["scruton_number"]), float(entry["reduced_velocity"]))
        for entry in entries
    ]
    if observed_grid != expected_grid:
        raise MatchedFrozenError("Parent fine-timestep cases do not match the frozen 3x12 grid.")
    return entries


def _qblade_mapping_record(parent_manifest: Mapping[str, Any]) -> dict[str, Any]:
    parent_mapping = parent_manifest.get("qblade_mapping")
    if not isinstance(parent_mapping, Mapping):
        raise MatchedFrozenError("Parent campaign has no QBlade mapping object.")
    parent_copy = json.loads(json.dumps(parent_mapping, allow_nan=False))
    return {
        "parent_mapping": parent_copy,
        "parent_mapping_sha256": sha256_payload(parent_copy),
        "preserved_geometry_structure_inflow_frames_and_external_rows": True,
        "legacy_frozen_phase_runtime_forcing": {
            "persistent_fixed_wake_source": "enabled with force_scale=1",
            "response_model": "configured for provenance but disabled",
            "startup_seed": "disabled",
            "motion_induced_velocity_feedback": (
                "retained in the base VorLap relative-velocity and force calculation; "
                "the response-aware modal closure alone is disabled"
            ),
            "strict_open_loop_one_way": False,
        },
    }


def _campaign_policy(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(entries) != EXPECTED_CASE_COUNT:
        raise MatchedFrozenError("Comparator policy requires exactly 36 cases.")
    minimum_damping = min(
        float(entry["structure"]["damping_ratio"]) for entry in entries
    )
    return {
        "case_count": EXPECTED_CASE_COUNT,
        "total_requested_steps": sum(int(entry["num_steps"]) for entry in entries),
        "total_simulated_seconds": sum(float(entry["duration_s"]) for entry in entries),
        "grid": "3 Scruton numbers x 12 reduced velocities x fine timestep only",
        "timestep_s": stationary.FINE_TIMESTEP_S,
        "predeclared_cap_natural_cycles": CAP_NATURAL_CYCLES,
        "terminal_stationarity_window_cycles": TERMINAL_WINDOW_CYCLES,
        "terminal_stationarity_gate": (
            "last 40 natural cycles split 20+20 using response_benchmark stationarity thresholds"
        ),
        "endpoint_alignment": (
            "fine steps equal twice an integer nominal-step count at a shared integer-nanosecond endpoint; "
            "up to one extra nominal step ensures exported samples 0..N-1 span at least 240 cycles"
        ),
        "minimum_structural_damping_ratio": minimum_damping,
        "maximum_unforced_transient_residual_ratio_after_cap": math.exp(
            -2.0 * math.pi * minimum_damping * CAP_NATURAL_CYCLES
        ),
        "transient_rationale_equation": "exp(-2*pi*zeta*N)",
        "transient_rationale_boundary": (
            "planning rationale for the forced structural transient; measured terminal stationarity controls use"
        ),
        "force_scale": FORCE_SCALE,
        "response_model_enabled": False,
        "startup_seed_enabled": False,
        "sil_vorlap_history_mode": "compact",
        "run_execution": "four-thread sequential cases",
    }


def _parent_case_files(
    parent_files: Mapping[str, Mapping[str, Any]], case_id: str
) -> dict[str, Mapping[str, Any]]:
    prefix = f"cases/{case_id}/"
    result = {
        path[len(prefix) :]: identity
        for path, identity in parent_files.items()
        if path.startswith(prefix)
    }
    if not result:
        raise MatchedFrozenError(f"Parent manifest has no files for case {case_id}.")
    return result


def _frozen_case_files(case_root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(case_root).as_posix(): _identity(
            path, relative=path.relative_to(case_root).as_posix()
        )
        for path in sorted(item for item in case_root.rglob("*") if item.is_file())
    }


def _runtime_preflight(config_path: Path) -> None:
    from vorlap.qblade_runtime import VorLapQBladeRuntime

    runtime = VorLapQBladeRuntime.from_qblade_config(str(config_path))
    model = getattr(runtime.controller, "response_model", None)
    if model is None:
        raise MatchedFrozenError("Frozen runtime lost its explicit response-model identity.")
    if bool(getattr(runtime.controller, "response_model_enabled", True)):
        raise MatchedFrozenError("Frozen runtime still has response-aware forcing enabled.")
    if bool(getattr(model, "startup_seed_enabled", True)):
        raise MatchedFrozenError("Frozen runtime still has startup-seed forcing enabled.")
    if not math.isclose(float(runtime.controller.force_scale), FORCE_SCALE, rel_tol=0.0, abs_tol=0.0):
        raise MatchedFrozenError("Frozen runtime did not consume force_scale=1.")


def generate_campaign(output: Path, parent_manifest_path: Path) -> Path:
    """Create an immutable 36-case fine-dt frozen campaign; never run QBlade."""

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing campaign directory: {output}")
    parent_manifest_path = Path(parent_manifest_path).expanduser().resolve()
    parent_root, parent_manifest = response._load_campaign(parent_manifest_path)
    if parent_manifest.get("schema") != PARENT_MANIFEST_SCHEMA:
        raise MatchedFrozenError(
            "Parent response campaign does not use the exact supported manifest schema."
        )
    _reject_write_within_immutable_roots(
        output,
        (parent_root,),
        label="Comparator campaign output",
    )
    try:
        protocol = stationary.build_protocol(parent_manifest_path)
        stationary.validate_protocol(protocol, parent_manifest_path)
    except stationary.ProtocolError as exc:
        raise MatchedFrozenError("Parent stationary response campaign does not validate.") from exc
    parent_files = parent_manifest["_verified_manifest_files"]
    pair_index = _parent_pair_index(protocol)
    fine_entries = _fine_parent_entries(parent_manifest)

    output.mkdir(parents=True)
    protocol_path = output / "parent_stationary_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frozen_entries: list[dict[str, Any]] = []
    source_hashes: set[str] = set()

    for parent_entry in fine_entries:
        case_id = str(parent_entry["case_id"])
        sc = float(parent_entry["scruton_number"])
        u_star = float(parent_entry["reduced_velocity"])
        parent_case_root = parent_root / "cases" / case_id
        frozen_case_root = output / "cases" / case_id
        shutil.copytree(parent_case_root, frozen_case_root, copy_function=shutil.copy2)

        sim_relative = str(parent_entry["sim_path"])
        parameter_relative = str(parent_entry["parameter_file"])
        frozen_sim = output.joinpath(*PurePosixPath(sim_relative).parts)
        frozen_config_path = output.joinpath(*PurePosixPath(parameter_relative).parts)
        parent_sim = parent_root.joinpath(*PurePosixPath(sim_relative).parts)
        parent_config_path = parent_root.joinpath(*PurePosixPath(parameter_relative).parts)

        endpoint = frozen_endpoint_for_cycles(sc)
        fine_steps = int(endpoint["dt_half_steps"])
        _replace_unique_qblade_setting(frozen_sim, "NUMTIMESTEPS", fine_steps)
        parent_steps = _read_unique_qblade_setting(parent_sim, "NUMTIMESTEPS")
        restored_sim = _replace_qblade_setting_text(
            frozen_sim.read_text(encoding="utf-8"), "NUMTIMESTEPS", parent_steps
        )
        if restored_sim.encode("utf-8") != parent_sim.read_bytes():
            raise MatchedFrozenError(
                f"Case {case_id} simulation changed outside NUMTIMESTEPS."
            )

        parent_config = _load_json_object(parent_config_path, f"parent config {case_id}")
        frozen_config = _disabled_frozen_config(parent_config)
        frozen_config_path.write_text(
            json.dumps(frozen_config, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        source_path = frozen_case_root / "model" / "VorLapAirfoils" / "CYLINDER.h5"
        source_identity = _source_identity_from_config(frozen_config, source_path)
        source_hashes.add(str(source_identity["sha256"]))
        _runtime_preflight(frozen_config_path)

        parent_case_files = _parent_case_files(parent_files, case_id)
        frozen_case_files = _frozen_case_files(frozen_case_root)
        if set(parent_case_files) != set(frozen_case_files):
            raise MatchedFrozenError(f"Case {case_id} copied file set changed.")
        permitted_changed_relative = {
            Path(sim_relative).relative_to(Path("cases") / case_id).as_posix(),
            Path(parameter_relative).relative_to(Path("cases") / case_id).as_posix(),
        }
        unchanged_rows = []
        changed_rows = []
        for relative in sorted(parent_case_files):
            parent_identity = parent_case_files[relative]
            frozen_identity = frozen_case_files[relative]
            identities_differ = parent_identity["sha256"] != frozen_identity["sha256"]
            if relative in permitted_changed_relative and identities_differ:
                changed_rows.append(
                    {
                        "path": relative,
                        "parent_sha256": parent_identity["sha256"],
                        "frozen_sha256": frozen_identity["sha256"],
                    }
                )
            else:
                if (
                    parent_identity["sha256"] != frozen_identity["sha256"]
                    or int(parent_identity["size_bytes"]) != int(frozen_identity["bytes"])
                ):
                    raise MatchedFrozenError(f"Copied input changed: {case_id}/{relative}")
                unchanged_rows.append(
                    {
                        "path": relative,
                        "sha256": frozen_identity["sha256"],
                        "bytes": frozen_identity["bytes"],
                    }
                )

        parent_pair = pair_index[(sc, u_star)]
        parent_binding = parent_pair["cases"]["dt_half"]
        if parent_binding["case_id"] != case_id:
            raise MatchedFrozenError(f"Parent protocol/case mismatch for {case_id}.")
        structure = dict(parent_entry["structure"])
        natural_frequency_hz = float(structure["natural_frequency_hz"])
        actual_cycles = float(endpoint["endpoint_time_s"]) * natural_frequency_hz
        residual_ratio = math.exp(
            -2.0
            * math.pi
            * float(structure["damping_ratio"])
            * CAP_NATURAL_CYCLES
        )
        entry = dict(parent_entry)
        entry.update(
            {
                "parent_case_id": case_id,
                "parent_case_binding_sha256": parent_binding["case_binding_sha256"],
                "parent_case_manifest_entry_sha256": parent_binding[
                    "case_manifest_entry_sha256"
                ],
                "parent_case_input_bundle_sha256": parent_binding[
                    "case_input_bundle_sha256"
                ],
                "parent_sim_sha256": parent_binding["sim_sha256"],
                "parent_config_sha256": parent_binding["config_sha256"],
                "num_steps": fine_steps,
                "duration_s": float(endpoint["endpoint_time_s"]),
                "duration_policy": "predeclared_240_natural_cycle_cap",
                "stationary_maximum_cycles": CAP_NATURAL_CYCLES,
                "actual_endpoint_natural_cycles": actual_cycles,
                "last_exported_sample_natural_cycles": float(
                    endpoint["last_exported_sample_natural_cycles"]
                ),
                "fine_steps_added_for_exported_span": int(
                    endpoint["fine_steps_added_beyond_endpoint_ceiling"]
                ),
                "endpoint_time_ns": int(endpoint["endpoint_time_ns"]),
                "aligned_nominal_step_count": int(endpoint["dt_steps"]),
                "response_model_config": frozen_config["response_model"],
                "force_scale": FORCE_SCALE,
                "response_model_enabled": False,
                "startup_seed_enabled": False,
                "source_fingerprint_sha256": source_identity["sha256"],
                "source_bytes": source_identity["bytes"],
                "unforced_structural_transient_residual_ratio_after_cap": residual_ratio,
                "unchanged_parent_files": unchanged_rows,
                "declared_changed_files": changed_rows,
            }
        )
        frozen_entries.append(entry)

    if len(source_hashes) != 1:
        raise MatchedFrozenError(
            f"Matched cases do not share one source fingerprint: {sorted(source_hashes)}"
        )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": CAMPAIGN_SCOPE,
        "parent_campaign": {
            "recorded_manifest_path": parent_manifest_path.as_posix(),
            "manifest_schema": parent_manifest["schema"],
            "manifest_sha256": sha256_file(parent_manifest_path),
            "stationary_protocol_file": protocol_path.relative_to(output).as_posix(),
            "stationary_protocol_file_sha256": sha256_file(protocol_path),
            "stationary_protocol_sha256": protocol["protocol_sha256"],
            "coefficient_fingerprint_sha256": parent_manifest["calibration_policy"][
                "coefficient_fingerprint_sha256"
            ],
        },
        "policy": _campaign_policy(frozen_entries),
        "source_binding": {
            "airfoil_id": "CYLINDER",
            "source_fingerprint_sha256": next(iter(source_hashes)),
            "identical_to_every_parent_fine_case": True,
        },
        "declared_runtime_config_deltas": sorted(EXPECTED_CONFIG_DELTAS),
        "declared_simulation_delta": "NUMTIMESTEPS only",
        "execution_inputs": _execution_input_record(),
        "qblade_mapping": _qblade_mapping_record(parent_manifest),
        "cases": frozen_entries,
        "files": _manifest_file_rows(output),
    }
    manifest = {**payload, "manifest_payload_sha256": sha256_payload(payload)}
    manifest_path = output / "campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_campaign(manifest_path, parent_manifest_path)
    return manifest_path


def _resolve_parent_manifest(
    manifest: Mapping[str, Any], parent_manifest_path: Path | None
) -> Path:
    if parent_manifest_path is None:
        raw = manifest.get("parent_campaign", {}).get("recorded_manifest_path")
        parent_manifest_path = Path(str(raw))
    path = Path(parent_manifest_path).expanduser().resolve()
    if not path.is_file():
        raise MatchedFrozenError(
            "The hash-bound parent response manifest is unavailable; supply its current path."
        )
    return path


def _load_matched_manifest(
    manifest_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json_object(path, "legacy frozen-phase manifest")
    if manifest.get("schema") != SCHEMA:
        raise MatchedFrozenError(
            f"Unsupported legacy frozen-phase schema in {path}."
        )
    supplied_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if supplied_hash != sha256_payload(payload):
        raise MatchedFrozenError("Legacy frozen-phase manifest self-hash changed.")
    root = path.parent
    file_index = _manifest_file_index(root, manifest)
    return root, manifest, file_index


def validate_campaign(
    manifest_path: Path, parent_manifest_path: Path | None = None
) -> dict[str, Any]:
    """Fail closed unless parent, copied inputs, deltas, and 36-case grid match."""

    root, manifest, file_index = _load_matched_manifest(manifest_path)
    parent_path = _resolve_parent_manifest(manifest, parent_manifest_path)
    parent_root, parent_manifest = response._load_campaign(parent_path)
    parent_binding = manifest.get("parent_campaign")
    if not isinstance(parent_binding, Mapping):
        raise MatchedFrozenError("Matched campaign has no parent binding.")
    if (
        parent_manifest.get("schema") != PARENT_MANIFEST_SCHEMA
        or parent_binding.get("manifest_schema") != PARENT_MANIFEST_SCHEMA
    ):
        raise MatchedFrozenError(
            "Parent response manifest schema binding changed."
        )
    if parent_binding.get("manifest_sha256") != sha256_file(parent_path):
        raise MatchedFrozenError("Parent response manifest hash changed.")
    protocol_relative = str(parent_binding.get("stationary_protocol_file", ""))
    protocol_path, protocol_relative = _safe_relative_path(
        root, protocol_relative, "Parent stationary protocol"
    )
    if protocol_relative not in file_index:
        raise MatchedFrozenError("Parent stationary protocol is not hash-bound as an input.")
    if parent_binding.get("stationary_protocol_file_sha256") != sha256_file(protocol_path):
        raise MatchedFrozenError("Stored parent stationary protocol hash changed.")
    protocol = _load_json_object(protocol_path, "parent stationary protocol")
    try:
        stationary.validate_protocol(protocol, parent_path)
    except stationary.ProtocolError as exc:
        raise MatchedFrozenError("Stored protocol no longer validates the parent campaign.") from exc
    if parent_binding.get("stationary_protocol_sha256") != protocol.get("protocol_sha256"):
        raise MatchedFrozenError("Parent stationary protocol self-hash binding changed.")
    if parent_binding.get("coefficient_fingerprint_sha256") != parent_manifest[
        "calibration_policy"
    ]["coefficient_fingerprint_sha256"]:
        raise MatchedFrozenError("Parent response coefficient fingerprint changed.")

    parent_files = parent_manifest["_verified_manifest_files"]
    parent_fine = {str(entry["case_id"]): entry for entry in _fine_parent_entries(parent_manifest)}
    pair_index = _parent_pair_index(protocol)
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != EXPECTED_CASE_COUNT:
        raise MatchedFrozenError("Matched campaign must contain exactly 36 cases.")
    seen_grid: set[tuple[float, float]] = set()
    source_hashes: set[str] = set()
    case_bindings: list[dict[str, Any]] = []
    for raw_entry in raw_cases:
        if not isinstance(raw_entry, Mapping):
            raise MatchedFrozenError("Matched-frozen case entry is not an object.")
        entry = dict(raw_entry)
        case_id = str(entry.get("case_id", ""))
        if case_id not in parent_fine or entry.get("parent_case_id") != case_id:
            raise MatchedFrozenError(f"Case {case_id!r} is not co-located with a parent fine case.")
        parent_entry = parent_fine[case_id]
        sc = float(entry["scruton_number"])
        u_star = float(entry["reduced_velocity"])
        key = (sc, u_star)
        if key in seen_grid:
            raise MatchedFrozenError(
                f"Duplicate legacy frozen-phase grid point: {key}"
            )
        seen_grid.add(key)
        endpoint = frozen_endpoint_for_cycles(sc)
        exact_matches = (
            "scruton_number",
            "qblade_implied_scruton_number",
            "scruton_relative_difference_percent",
            "role",
            "reduced_velocity",
            "mean_inflow_mps",
            "reynolds_number",
            "timestep_level",
            "timestep_s",
            "steps_per_natural_cycle",
            "structure",
            "sim_path",
            "parameter_file",
            "coefficient_fingerprint_sha256",
        )
        for name in exact_matches:
            if entry.get(name) != parent_entry.get(name):
                raise MatchedFrozenError(f"Case {case_id} is not co-located in {name}.")
        if entry.get("timestep_level") != "dt_half" or not math.isclose(
            float(entry["timestep_s"]), stationary.FINE_TIMESTEP_S, rel_tol=0.0, abs_tol=0.0
        ):
            raise MatchedFrozenError(f"Case {case_id} is not the exact fine timestep.")
        expected_scalars = {
            "num_steps": int(endpoint["dt_half_steps"]),
            "duration_s": float(endpoint["endpoint_time_s"]),
            "stationary_maximum_cycles": CAP_NATURAL_CYCLES,
            "endpoint_time_ns": int(endpoint["endpoint_time_ns"]),
            "aligned_nominal_step_count": int(endpoint["dt_steps"]),
            "actual_endpoint_natural_cycles": float(endpoint["endpoint_time_s"])
            * float(entry["structure"]["natural_frequency_hz"]),
            "last_exported_sample_natural_cycles": float(
                endpoint["last_exported_sample_natural_cycles"]
            ),
            "fine_steps_added_for_exported_span": int(
                endpoint["fine_steps_added_beyond_endpoint_ceiling"]
            ),
            "force_scale": FORCE_SCALE,
            "unforced_structural_transient_residual_ratio_after_cap": math.exp(
                -2.0
                * math.pi
                * float(entry["structure"]["damping_ratio"])
                * CAP_NATURAL_CYCLES
            ),
        }
        for name, expected in expected_scalars.items():
            if entry.get(name) != expected:
                raise MatchedFrozenError(f"Case {case_id} has invalid {name}.")
        if (
            entry.get("duration_policy") != "predeclared_240_natural_cycle_cap"
            or entry.get("response_model_enabled") is not False
            or entry.get("startup_seed_enabled") is not False
        ):
            raise MatchedFrozenError(f"Case {case_id} has invalid forcing policy.")

        parent_pair = pair_index[key]["cases"]["dt_half"]
        binding_fields = {
            "parent_case_binding_sha256": "case_binding_sha256",
            "parent_case_manifest_entry_sha256": "case_manifest_entry_sha256",
            "parent_case_input_bundle_sha256": "case_input_bundle_sha256",
            "parent_sim_sha256": "sim_sha256",
            "parent_config_sha256": "config_sha256",
        }
        for child_name, parent_name in binding_fields.items():
            if entry.get(child_name) != parent_pair.get(parent_name):
                raise MatchedFrozenError(f"Case {case_id} parent binding changed: {child_name}.")

        sim_path, sim_relative = _safe_relative_path(root, entry["sim_path"], f"Case {case_id} sim")
        config_path, config_relative = _safe_relative_path(
            root, entry["parameter_file"], f"Case {case_id} config"
        )
        if sim_relative not in file_index or config_relative not in file_index:
            raise MatchedFrozenError(f"Case {case_id} primary inputs are not hash-bound.")
        parent_sim = parent_root.joinpath(*PurePosixPath(str(parent_entry["sim_path"])).parts)
        parent_config_path = parent_root.joinpath(
            *PurePosixPath(str(parent_entry["parameter_file"])).parts
        )
        if int(float(_read_unique_qblade_setting(sim_path, "NUMTIMESTEPS"))) != int(
            endpoint["dt_half_steps"]
        ):
            raise MatchedFrozenError(f"Case {case_id} simulation step count changed.")
        if _read_unique_qblade_setting(sim_path, "TIMESTEP") != _read_unique_qblade_setting(
            parent_sim, "TIMESTEP"
        ) or _read_unique_qblade_setting(sim_path, "MEANINF") != _read_unique_qblade_setting(
            parent_sim, "MEANINF"
        ):
            raise MatchedFrozenError(f"Case {case_id} timestep or inflow is not parent-matched.")
        restored_sim = _replace_qblade_setting_text(
            sim_path.read_text(encoding="utf-8"),
            "NUMTIMESTEPS",
            _read_unique_qblade_setting(parent_sim, "NUMTIMESTEPS"),
        )
        if restored_sim.encode("utf-8") != parent_sim.read_bytes():
            raise MatchedFrozenError(f"Case {case_id} sim differs beyond NUMTIMESTEPS.")

        parent_config = _load_json_object(parent_config_path, f"parent config {case_id}")
        frozen_config = _load_json_object(config_path, f"frozen config {case_id}")
        if _config_delta_paths(parent_config, frozen_config) != EXPECTED_CONFIG_DELTAS:
            raise MatchedFrozenError(f"Case {case_id} runtime config has undeclared changes.")
        expected_config = _disabled_frozen_config(parent_config)
        if frozen_config != expected_config or entry.get("response_model_config") != frozen_config[
            "response_model"
        ]:
            raise MatchedFrozenError(f"Case {case_id} frozen runtime config changed.")
        case_root = root / "cases" / case_id
        source_path = case_root / "model" / "VorLapAirfoils" / "CYLINDER.h5"
        source_identity = _source_identity_from_config(frozen_config, source_path)
        source_hashes.add(source_identity["sha256"])
        if (
            entry.get("source_fingerprint_sha256") != source_identity["sha256"]
            or int(entry.get("source_bytes", -1)) != source_identity["bytes"]
        ):
            raise MatchedFrozenError(f"Case {case_id} source binding changed.")

        parent_case_files = _parent_case_files(parent_files, case_id)
        frozen_case_files = _frozen_case_files(case_root)
        if set(parent_case_files) != set(frozen_case_files):
            raise MatchedFrozenError(f"Case {case_id} file set differs from its parent.")
        relative_sim = Path(sim_relative).relative_to(Path("cases") / case_id).as_posix()
        relative_config = Path(config_relative).relative_to(Path("cases") / case_id).as_posix()
        permitted_changed = {relative_sim, relative_config}
        actual_changed = {
            relative
            for relative in parent_case_files
            if parent_case_files[relative]["sha256"]
            != frozen_case_files[relative]["sha256"]
        }
        if relative_config not in actual_changed or not actual_changed.issubset(
            permitted_changed
        ):
            raise MatchedFrozenError(
                f"Case {case_id} has an invalid changed-file set: {sorted(actual_changed)}"
            )
        for relative in parent_case_files:
            same = parent_case_files[relative]["sha256"] == frozen_case_files[relative]["sha256"]
            if same != (relative not in actual_changed):
                raise MatchedFrozenError(f"Case {case_id} file delta is invalid: {relative}")
        if entry.get("declared_changed_files") != [
            {
                "path": relative,
                "parent_sha256": parent_case_files[relative]["sha256"],
                "frozen_sha256": frozen_case_files[relative]["sha256"],
            }
            for relative in sorted(actual_changed)
        ]:
            raise MatchedFrozenError(f"Case {case_id} changed-file ledger differs.")
        expected_unchanged = [
            {
                "path": relative,
                "sha256": frozen_case_files[relative]["sha256"],
                "bytes": frozen_case_files[relative]["bytes"],
            }
            for relative in sorted(set(frozen_case_files) - actual_changed)
        ]
        if entry.get("unchanged_parent_files") != expected_unchanged:
            raise MatchedFrozenError(f"Case {case_id} unchanged-file ledger differs.")
        _runtime_preflight(config_path)
        case_bindings.append(
            {
                "case_id": case_id,
                "scruton_number": sc,
                "reduced_velocity": u_star,
                "sim_sha256": file_index[sim_relative]["sha256"],
                "config_sha256": file_index[config_relative]["sha256"],
                "source_sha256": source_identity["sha256"],
                "parent_case_binding_sha256": parent_pair["case_binding_sha256"],
            }
        )

    expected_grid = {
        (float(sc), float(u_star))
        for sc in stationary.STRUCTURES
        for u_star in stationary.REDUCED_VELOCITIES
    }
    if seen_grid != expected_grid or len(source_hashes) != 1:
        raise MatchedFrozenError("Matched campaign grid or source fingerprint is incomplete.")
    expected_source_binding = {
        "airfoil_id": "CYLINDER",
        "source_fingerprint_sha256": next(iter(source_hashes)),
        "identical_to_every_parent_fine_case": True,
    }
    if manifest.get("source_binding") != expected_source_binding:
        raise MatchedFrozenError("Top-level comparator source binding changed.")
    if manifest.get("scope") != CAMPAIGN_SCOPE:
        raise MatchedFrozenError("Comparator claim scope changed.")
    if manifest.get("policy") != _campaign_policy(raw_cases):
        raise MatchedFrozenError("Comparator physics/numerics policy changed.")
    if manifest.get("declared_runtime_config_deltas") != sorted(EXPECTED_CONFIG_DELTAS):
        raise MatchedFrozenError("Declared runtime-config delta set changed.")
    if manifest.get("declared_simulation_delta") != "NUMTIMESTEPS only":
        raise MatchedFrozenError("Declared simulation delta changed.")
    if manifest.get("qblade_mapping") != _qblade_mapping_record(parent_manifest):
        raise MatchedFrozenError(
            "Legacy frozen-phase QBlade mapping record changed."
        )
    if manifest.get("execution_inputs") != _execution_input_record():
        raise MatchedFrozenError(
            "Legacy frozen-phase runner or result-filter identity changed."
        )
    return {
        "valid": True,
        "manifest_path": Path(manifest_path).expanduser().resolve().as_posix(),
        "manifest_sha256": sha256_file(Path(manifest_path).expanduser().resolve()),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "parent_manifest_path": parent_path.as_posix(),
        "parent_manifest_sha256": sha256_file(parent_path),
        "parent_protocol_sha256": protocol["protocol_sha256"],
        "source_fingerprint_sha256": next(iter(source_hashes)),
        "case_count": len(case_bindings),
        "case_bindings": case_bindings,
        "campaign_root": root,
        "manifest": manifest,
        "file_index": file_index,
        "parent_root": parent_root,
        "parent_manifest": parent_manifest,
    }


def build_run_plan(
    manifest_path: Path,
    parent_manifest_path: Path,
    results_root: Path,
    *,
    qblade_root: Path,
) -> dict[str, Any]:
    """Return a hash-bound, sequential, four-thread dry plan; never execute it."""

    state = validate_campaign(manifest_path, parent_manifest_path)
    results_root = Path(results_root).expanduser().resolve()
    for immutable_root in (state["campaign_root"], state["parent_root"]):
        try:
            results_root.relative_to(Path(immutable_root).resolve())
        except ValueError:
            continue
        raise MatchedFrozenError(
            f"Results root may not be inside immutable campaign inputs: {results_root}"
        )
    if results_root.exists() and (
        not results_root.is_dir() or any(results_root.iterdir())
    ):
        raise FileExistsError(f"Refusing a nonempty results root: {results_root}")
    qblade_root = Path(qblade_root).expanduser().resolve()
    if not qblade_root.is_dir():
        raise MatchedFrozenError(f"QBlade root is unavailable: {qblade_root}")
    runner = REPO_ROOT / "run_headless" / "run_qblade_sim.py"
    result_filter = HERE / "cantilever_results.filter"
    root = state["campaign_root"]
    commands = []
    cases = sorted(
        state["manifest"]["cases"],
        key=lambda row: (float(row["scruton_number"]), float(row["reduced_velocity"])),
    )
    for sequence_index, entry in enumerate(cases):
        case_id = str(entry["case_id"])
        case_output = results_root / case_id
        argv = [
            sys.executable,
            str(runner),
            "--sim",
            str(root.joinpath(*PurePosixPath(str(entry["sim_path"])).parts)),
            "--qblade-root",
            str(qblade_root),
            "--output-dir",
            str(case_output),
            "--result-name",
            "result",
            "--result-filter",
            str(result_filter),
            "--threads",
            str(THREADS_PER_RUN),
            "--sil-vorlap",
            "--coupling-mode",
            "sil-vorlap",
            "--sil-vorlap-history-mode",
            "compact",
        ]
        commands.append(
            {
                "sequence_index": sequence_index,
                "case_id": case_id,
                "scruton_number": entry["scruton_number"],
                "reduced_velocity": entry["reduced_velocity"],
                "argv": argv,
                "shell_command_for_display_only": shlex.join(argv),
            }
        )
    payload = {
        "schema": RUN_PLAN_SCHEMA,
        "dry_plan": True,
        "execution_function_provided": True,
        "execution_requires_explicit_cli_gate": "run --execute",
        "execution_order": "strictly sequential in sequence_index order",
        "maximum_concurrent_cases": 1,
        "threads_per_qblade_run": THREADS_PER_RUN,
        "thread_environment": EXECUTION_THREAD_ENVIRONMENT,
        "sil_vorlap_history_mode": "compact",
        "checkpoint_every": 0,
        "total_requested_steps": state["manifest"]["policy"][
            "total_requested_steps"
        ],
        "total_simulated_seconds": state["manifest"]["policy"][
            "total_simulated_seconds"
        ],
        "campaign_manifest_path": Path(manifest_path).expanduser().resolve().as_posix(),
        "campaign_manifest_sha256": state["manifest_sha256"],
        "campaign_manifest_payload_sha256": state["manifest_payload_sha256"],
        "parent_manifest_sha256": state["parent_manifest_sha256"],
        "parent_protocol_sha256": state["parent_protocol_sha256"],
        "source_fingerprint_sha256": state["source_fingerprint_sha256"],
        "execution_inputs": state["manifest"]["execution_inputs"],
        "results_root": results_root.as_posix(),
        "commands": commands,
    }
    return {**payload, "run_plan_payload_sha256": sha256_payload(payload)}


def execute_run_plan(
    plan: Mapping[str, Any],
    manifest_path: Path,
    parent_manifest_path: Path,
    results_root: Path,
    *,
    qblade_root: Path,
) -> Path:
    """Execute the exact rebuilt plan sequentially after an explicit CLI gate."""

    supplied = dict(plan)
    supplied_hash = supplied.pop("run_plan_payload_sha256", None)
    if supplied.get("schema") != RUN_PLAN_SCHEMA or supplied_hash != sha256_payload(
        supplied
    ):
        raise MatchedFrozenError("Run plan schema or self-hash changed.")
    expected = build_run_plan(
        manifest_path,
        parent_manifest_path,
        results_root,
        qblade_root=qblade_root,
    )
    if dict(plan) != expected:
        raise MatchedFrozenError(
            "Run plan no longer matches immutable inputs and current execution paths."
        )
    state = validate_campaign(manifest_path, parent_manifest_path)
    entries = {str(row["case_id"]): row for row in state["manifest"]["cases"]}
    commands = list(expected["commands"])
    if len(commands) != EXPECTED_CASE_COUNT:
        raise MatchedFrozenError("Executable comparator plan must contain exactly 36 cases.")

    # Complete every cheap configuration check before creating the results root.
    for row in commands:
        case_id = str(row["case_id"])
        entry = entries.get(case_id)
        if entry is None:
            raise MatchedFrozenError(f"Run plan contains an unknown case: {case_id}.")
        config_path = state["campaign_root"].joinpath(
            *PurePosixPath(str(entry["parameter_file"])).parts
        )
        _runtime_preflight(config_path)

    results_root = Path(results_root).expanduser().resolve()
    if results_root.exists() and (
        not results_root.is_dir() or any(results_root.iterdir())
    ):
        raise FileExistsError(f"Refusing a nonempty results root: {results_root}")
    results_root.mkdir(parents=True, exist_ok=True)
    execution_environment = dict(os.environ)
    execution_environment.update(EXECUTION_THREAD_ENVIRONMENT)
    completed: list[dict[str, Any]] = []
    for row in commands:
        case_id = str(row["case_id"])
        case_output = results_root / case_id
        if case_output.exists():
            raise FileExistsError(
                f"Refusing to replace an existing case result: {case_output}"
            )
        subprocess.run(
            list(row["argv"]),
            check=True,
            cwd=REPO_ROOT,
            env=execution_environment,
        )
        if not case_output.is_dir():
            raise MatchedFrozenError(
                f"Runner returned without creating the case output: {case_output}"
            )
        audit = audit_frozen_runtime_case(
            campaign_root=state["campaign_root"],
            file_index=state["file_index"],
            entry=entries[case_id],
            result_directory=case_output,
        )
        completed.append(
            {
                "sequence_index": row["sequence_index"],
                "case_id": case_id,
                "result_identity": audit["result_identity"],
                "run_summary_identity": audit["run_summary_identity"],
                "history_identity": audit["history_identity"],
                "runtime_build_fingerprint": audit[
                    "runtime_build_fingerprint"
                ],
                "runtime_audit_pass": bool(audit["pass"]),
            }
        )
    receipt_payload = {
        "schema": "vlastos2026-legacy-frozen-phase-bridge-execution-receipt-v1",
        "execution_order": expected["execution_order"],
        "maximum_concurrent_cases": expected["maximum_concurrent_cases"],
        "threads_per_qblade_run": expected["threads_per_qblade_run"],
        "thread_environment": expected["thread_environment"],
        "run_plan_payload_sha256": expected["run_plan_payload_sha256"],
        "campaign_manifest_sha256": state["manifest_sha256"],
        "parent_manifest_sha256": state["parent_manifest_sha256"],
        "source_fingerprint_sha256": state["source_fingerprint_sha256"],
        "completed_case_count": len(completed),
        "all_runtime_audits_pass": all(
            row["runtime_audit_pass"] for row in completed
        ),
        "cases": completed,
    }
    receipt = {
        **receipt_payload,
        "execution_receipt_payload_sha256": sha256_payload(receipt_payload),
    }
    return _write_json_exclusive(
        results_root / "legacy_frozen_phase_execution_receipt.json",
        receipt,
    )


def _portable_path_has_suffix(value: Any, expected: str) -> bool:
    observed = PurePosixPath(str(value).replace("\\", "/"))
    suffix = PurePosixPath(str(expected).replace("\\", "/"))
    return (
        bool(observed.parts)
        and bool(suffix.parts)
        and len(observed.parts) >= len(suffix.parts)
        and observed.parts[-len(suffix.parts) :] == suffix.parts
    )


def _reported_identity_matches(
    reported: Any,
    actual_path: Path,
    *,
    expected_suffix: str,
    manifest_identity: Mapping[str, Any] | None = None,
) -> bool:
    if not isinstance(reported, Mapping):
        return False
    actual_sha = sha256_file(actual_path)
    actual_bytes = actual_path.stat().st_size
    if not _portable_path_has_suffix(reported.get("path"), expected_suffix):
        return False
    if reported.get("sha256") != actual_sha:
        return False
    try:
        if int(reported.get("size_bytes", -1)) != actual_bytes:
            return False
    except (TypeError, ValueError):
        return False
    if manifest_identity is not None:
        return (
            manifest_identity.get("sha256") == actual_sha
            and int(manifest_identity.get("bytes", -1)) == actual_bytes
        )
    return True


def _unique_matching_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1 or not matches[0].is_file():
        raise MatchedFrozenError(
            f"Expected exactly one {label} in {directory}; found {len(matches)}."
        )
    return matches[0]


def _parse_bool_text(value: str, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise MatchedFrozenError(f"Compact history field {label} is not boolean: {value!r}")


def _parse_finite_float(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MatchedFrozenError(f"Compact history field {label} is not numeric.") from exc
    if not math.isfinite(number):
        raise MatchedFrozenError(f"Compact history field {label} is non-finite.")
    return number


def _parse_integer(value: str, label: str) -> int:
    try:
        numeric = float(value)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MatchedFrozenError(f"Compact history field {label} is not an integer.") from exc
    if numeric != integer:
        raise MatchedFrozenError(f"Compact history field {label} is not an integer.")
    return integer


def _validate_sha256(value: str, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise MatchedFrozenError(f"Compact history field {label} is not lowercase SHA-256.")
    return text


def _zero_force_digest(component_count: int) -> str:
    prefix = FORCE_VECTOR_DIGEST_SCHEMA.encode("ascii") + b"\0" + struct.pack(
        "!Q", component_count
    )
    return hashlib.sha256(
        prefix + struct.pack(f"!{component_count}d", *([0.0] * component_count))
    ).hexdigest()


COMPACT_FORCE_METRICS = (
    "max_abs_force_component_n",
    "max_force_norm_n",
    "max_force_x_n",
    "max_force_y_n",
    "max_force_z_n",
    "total_force_x_n",
    "total_force_y_n",
    "total_force_z_n",
    "force_vector_l1_n",
    "force_vector_l2_n",
)


def audit_compact_frozen_history(
    history_path: Path,
    *,
    timestep_s: float,
    expected_samples: int,
    node_ids: Sequence[str],
) -> dict[str, Any]:
    """Stream and validate compact one-step-lag/source history in O(1) memory."""

    history_path = Path(history_path)
    component_count = 3 * len(node_ids)
    if component_count != EXPECTED_FORCE_COMPONENT_COUNT:
        raise MatchedFrozenError("Frozen history must use the exact 21-node force mapping.")
    required = {
        "force_history_schema",
        "force_vector_digest_schema",
        "force_vector_component_count",
        "step_index",
        "time_s",
        "generated_step_index",
        "generated_time_s",
        "applied_source_step_index",
        "applied_source_time_s",
        "applied_startup_zero",
        "applied_force_lag_steps",
        "qblade_timestep",
        "azimuth_deg",
        "generated_force_vector_sha256",
        "applied_force_vector_sha256",
        "structural_velocity_vector_sha256",
        "applied_force_dot_structural_velocity_w",
        "applied_power_sign_convention",
    }
    for prefix in ("generated_", "applied_"):
        required.add(prefix + "max_force_node_id")
        required.update(prefix + name for name in COMPACT_FORCE_METRICS)
    prior_generated_digest: str | None = None
    prior_generated_time: float | None = None
    rows = 0
    maximum_generated_force_component = 0.0
    with history_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if not columns or len(columns) != len(set(columns)):
            raise MatchedFrozenError("Compact history has missing or duplicate columns.")
        missing = sorted(required - set(columns))
        if missing:
            raise MatchedFrozenError(f"Compact history is missing columns: {missing}")
        response_columns = [
            name for name in columns if name == "response_model_type" or name.startswith("response_")
        ]
        if response_columns:
            raise MatchedFrozenError(
                "Frozen compact history contains response diagnostics despite response disablement."
            )
        zero_digest = _zero_force_digest(component_count)
        node_set = {str(value) for value in node_ids}
        for row_index, row in enumerate(reader):
            rows += 1
            if row["force_history_schema"] != COMPACT_HISTORY_SCHEMA:
                raise MatchedFrozenError(f"Compact history schema changed at row {row_index}.")
            if row["force_vector_digest_schema"] != FORCE_VECTOR_DIGEST_SCHEMA:
                raise MatchedFrozenError(f"Force-vector digest schema changed at row {row_index}.")
            if _parse_integer(row["force_vector_component_count"], "component count") != component_count:
                raise MatchedFrozenError(f"Force-vector size changed at row {row_index}.")
            if _parse_integer(row["step_index"], "step_index") != row_index or _parse_integer(
                row["generated_step_index"], "generated_step_index"
            ) != row_index:
                raise MatchedFrozenError(f"Compact history step sequence changed at row {row_index}.")
            time_s = _parse_finite_float(row["time_s"], "time_s")
            generated_time = _parse_finite_float(row["generated_time_s"], "generated_time_s")
            expected_time = row_index * timestep_s
            tolerance = max(1.0e-12, abs(expected_time) * 1.0e-12)
            if not math.isclose(time_s, expected_time, rel_tol=1.0e-12, abs_tol=tolerance) or not math.isclose(
                generated_time, expected_time, rel_tol=1.0e-12, abs_tol=tolerance
            ):
                raise MatchedFrozenError(f"Compact history time grid changed at row {row_index}.")
            if _parse_integer(row["applied_force_lag_steps"], "applied_force_lag_steps") != 1:
                raise MatchedFrozenError("Matched frozen history requires the declared one-step SIL lag.")
            generated_digest = _validate_sha256(
                row["generated_force_vector_sha256"], "generated_force_vector_sha256"
            )
            applied_digest = _validate_sha256(
                row["applied_force_vector_sha256"], "applied_force_vector_sha256"
            )
            _validate_sha256(
                row["structural_velocity_vector_sha256"],
                "structural_velocity_vector_sha256",
            )
            startup_zero = _parse_bool_text(row["applied_startup_zero"], "applied_startup_zero")
            if row_index == 0:
                if (
                    not startup_zero
                    or row["applied_source_step_index"].strip()
                    or row["applied_source_time_s"].strip()
                    or applied_digest != zero_digest
                ):
                    raise MatchedFrozenError("Compact history startup-zero provenance changed.")
            else:
                if startup_zero or _parse_integer(
                    row["applied_source_step_index"], "applied_source_step_index"
                ) != row_index - 1:
                    raise MatchedFrozenError(f"Compact history lag index changed at row {row_index}.")
                source_time = _parse_finite_float(
                    row["applied_source_time_s"], "applied_source_time_s"
                )
                if (
                    prior_generated_digest is None
                    or prior_generated_time is None
                    or applied_digest != prior_generated_digest
                    or not math.isclose(source_time, prior_generated_time, rel_tol=0.0, abs_tol=1.0e-12)
                ):
                    raise MatchedFrozenError(f"Compact history lag digest changed at row {row_index}.")
            if row["applied_power_sign_convention"] != APPLIED_POWER_SIGN_CONVENTION:
                raise MatchedFrozenError("Compact history applied-power sign convention changed.")
            _parse_finite_float(
                row["applied_force_dot_structural_velocity_w"],
                "applied_force_dot_structural_velocity_w",
            )
            _parse_finite_float(row["qblade_timestep"], "qblade_timestep")
            _parse_finite_float(row["azimuth_deg"], "azimuth_deg")
            for prefix in ("generated_", "applied_"):
                if row[prefix + "max_force_node_id"] not in node_set:
                    raise MatchedFrozenError(f"Compact history node identity changed at row {row_index}.")
                for name in COMPACT_FORCE_METRICS:
                    value = _parse_finite_float(row[prefix + name], prefix + name)
                    if name in {
                        "max_abs_force_component_n",
                        "max_force_norm_n",
                        "force_vector_l1_n",
                        "force_vector_l2_n",
                    } and value < 0.0:
                        raise MatchedFrozenError(f"Compact force norm is negative at row {row_index}.")
            maximum_generated_force_component = max(
                maximum_generated_force_component,
                _parse_finite_float(
                    row["generated_max_abs_force_component_n"],
                    "generated_max_abs_force_component_n",
                ),
            )
            prior_generated_digest = generated_digest
            prior_generated_time = generated_time
    if rows != expected_samples:
        raise MatchedFrozenError(
            f"Compact history row count differs: expected {expected_samples}, observed {rows}."
        )
    if maximum_generated_force_component <= 0.0:
        raise MatchedFrozenError("Frozen source generated no nonzero force over the completed run.")
    return {
        "pass": True,
        "samples": rows,
        "columns": len(columns),
        "history_schema": COMPACT_HISTORY_SCHEMA,
        "vector_digest_schema": FORCE_VECTOR_DIGEST_SCHEMA,
        "force_vector_component_count": component_count,
        "one_step_lag_digest_pass": True,
        "response_columns_absent": True,
        "maximum_generated_force_component_n": maximum_generated_force_component,
        "file_identity": _identity(history_path),
    }


def _manifest_identity_for(
    file_index: Mapping[str, Mapping[str, Any]], relative: str
) -> Mapping[str, Any]:
    identity = file_index.get(PurePosixPath(relative).as_posix())
    if not isinstance(identity, Mapping):
        raise MatchedFrozenError(f"Manifest lacks required file identity: {relative}")
    return identity


def _runtime_build_fingerprint(
    summary: Mapping[str, Any], *, case_id: str
) -> dict[str, Any]:
    """Return code/library identity after proving the native callback stayed idle."""

    reproducibility = summary.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise MatchedFrozenError(
            f"Run summary has no reproducibility object for {case_id}."
        )
    qblade = reproducibility.get("qblade")
    qblade_library = qblade.get("library") if isinstance(qblade, Mapping) else None
    bridge_library = reproducibility.get("bridge_library")

    def require_sha256(value: Any, label: str) -> str:
        text = str(value)
        if re.fullmatch(r"[0-9a-f]{64}", text) is None:
            raise MatchedFrozenError(
                f"Run summary has no valid {label} SHA-256 for {case_id}."
            )
        return text

    if not isinstance(qblade_library, Mapping) or not isinstance(
        bridge_library, Mapping
    ):
        raise MatchedFrozenError(
            f"Run summary lacks QBlade or bridge-library identity for {case_id}."
        )
    qblade_sha256 = require_sha256(
        qblade_library.get("sha256"), "QBlade library"
    )
    bridge_sha256 = require_sha256(
        bridge_library.get("sha256"), "bridge library"
    )
    python_sha256 = require_sha256(
        reproducibility.get("vorlap_python_runtime_fingerprint"),
        "VorLap Python runtime fingerprint",
    )

    native = summary.get("native_callback_evidence")
    if not isinstance(native, Mapping):
        raise MatchedFrozenError(
            f"Run summary has no native-callback evidence for {case_id}."
        )
    library_file_hash = native.get("library_file_hash")
    native_checks = {
        "callback_block_present": native.get("callback_block_present") is True,
        "callback_block_error_absent": native.get("callback_block_error") is None,
        "config_error_absent": native.get("config_error") is None,
        "library_file_exists": native.get("library_file_exists") is True,
        "library_symbol_check_available": native.get(
            "library_symbol_check_available"
        )
        is True,
        "library_symbol_check_error_absent": native.get(
            "library_symbol_check_error"
        )
        is None,
        "required_symbols_present": native.get("library_missing_required_symbols")
        == [],
        "library_hash_matches_bridge": isinstance(library_file_hash, Mapping)
        and library_file_hash.get("sha256") == bridge_sha256,
        "update_count_zero": native.get("update_count_observed_max") == 0,
        "update_init_count_zero": native.get("update_init_count") == 0,
        "update_init_not_seen": native.get("update_init_seen") is False,
        "update_log_entry_count_zero": native.get("update_log_entry_count") == 0,
        "runtime_initialized_message_not_seen": native.get(
            "runtime_initialized_message_seen"
        )
        is False,
    }
    failed_checks = [name for name, passed in native_checks.items() if not passed]
    if failed_checks:
        raise MatchedFrozenError(
            f"Native callback was not proven idle for {case_id}: {failed_checks}."
        )
    payload = {
        "qblade_library_sha256": qblade_sha256,
        "bridge_library_sha256": bridge_sha256,
        "vorlap_python_runtime_fingerprint_sha256": python_sha256,
        "native_callback_zero_evidence": native_checks,
        "native_callback_zero_pass": True,
    }
    return {**payload, "fingerprint_sha256": sha256_payload(payload)}


def audit_frozen_runtime_case(
    *,
    campaign_root: Path,
    file_index: Mapping[str, Mapping[str, Any]],
    entry: Mapping[str, Any],
    result_directory: Path,
) -> dict[str, Any]:
    """Bind a completed frozen run to immutable inputs and compact lag evidence."""

    case_id = str(entry["case_id"])
    result_directory = Path(result_directory)
    result_path = result_directory / "result.txt"
    if not result_path.is_file():
        raise MatchedFrozenError(f"Frozen result is missing for {case_id}: {result_path}")
    summary_path = _unique_matching_file(result_directory, "*_run_summary.json", "run summary")
    history_path = _unique_matching_file(
        result_directory, "*_sil_vorlap_history.csv", "compact SIL history"
    )
    summary = _load_json_object(summary_path, f"frozen run summary {case_id}")
    runtime_build_fingerprint = _runtime_build_fingerprint(
        summary, case_id=case_id
    )
    expected_steps = int(entry["num_steps"])
    expected_timestep = float(entry["timestep_s"])
    sim_relative = str(entry["sim_path"])
    config_relative = str(entry["parameter_file"])
    sim_path = campaign_root.joinpath(*PurePosixPath(sim_relative).parts)
    config_path = campaign_root.joinpath(*PurePosixPath(config_relative).parts)
    config = _load_json_object(config_path, f"frozen config {case_id}")
    if (
        config.get("force_scale") != FORCE_SCALE
        or config.get("response_model", {}).get("enabled") is not False
        or config.get("response_model", {}).get("startup_seed", {}).get("enabled") is not False
    ):
        raise MatchedFrozenError(f"Frozen forcing configuration changed for {case_id}.")
    input_files = summary.get("reproducibility", {}).get("input_files")
    if not isinstance(input_files, Mapping):
        raise MatchedFrozenError(f"Run summary has no input identities for {case_id}.")
    if not _reported_identity_matches(
        input_files.get("sim"),
        sim_path,
        expected_suffix=sim_relative,
        manifest_identity=_manifest_identity_for(file_index, sim_relative),
    ) or not _reported_identity_matches(
        input_files.get("vorlap_config"),
        config_path,
        expected_suffix=config_relative,
        manifest_identity=_manifest_identity_for(file_index, config_relative),
    ):
        raise MatchedFrozenError(f"Run input identity changed for {case_id}.")
    result_filter_path = HERE / "cantilever_results.filter"
    if not _reported_identity_matches(
        input_files.get("result_filter"),
        result_filter_path,
        expected_suffix="examples/vlastos2026_cantilever/cantilever_results.filter",
    ):
        raise MatchedFrozenError(f"Run result-filter identity changed for {case_id}.")
    command_line = summary.get("reproducibility", {}).get("command_line")
    if not isinstance(command_line, list) or not all(
        isinstance(value, str) for value in command_line
    ):
        raise MatchedFrozenError(f"Run summary has no reproducible command line for {case_id}.")

    def command_value(option: str) -> str:
        indices = [index for index, value in enumerate(command_line) if value == option]
        if len(indices) != 1 or indices[0] + 1 >= len(command_line):
            raise MatchedFrozenError(
                f"Run command line does not contain one {option} value for {case_id}."
            )
        return command_line[indices[0] + 1]

    if (
        command_value("--threads") != str(THREADS_PER_RUN)
        or command_value("--coupling-mode") != "sil-vorlap"
        or command_value("--sil-vorlap-history-mode") != "compact"
        or command_value("--result-name") != "result"
        or "--sil-vorlap" not in command_line
        or "--checkpoint-every" in command_line
        or not _portable_path_has_suffix(
            command_value("--sim"), sim_relative
        )
        or not _portable_path_has_suffix(
            command_value("--result-filter"),
            "examples/vlastos2026_cantilever/cantilever_results.filter",
        )
    ):
        raise MatchedFrozenError(f"Run command policy changed for {case_id}.")
    loader_environment = summary.get("reproducibility", {}).get(
        "loader_environment"
    )
    if not isinstance(loader_environment, Mapping) or str(
        loader_environment.get("OMP_NUM_THREADS")
    ) != str(THREADS_PER_RUN):
        raise MatchedFrozenError(f"Run thread environment changed for {case_id}.")
    qblade_settings = summary.get("reproducibility", {}).get("qblade_case_settings")
    if not isinstance(qblade_settings, Mapping):
        raise MatchedFrozenError(f"Run summary has no QBlade settings for {case_id}.")
    for name, expected in {
        "MEANINF": float(entry["mean_inflow_mps"]),
        "TIMESTEP": expected_timestep,
        "NUMTIMESTEPS": float(expected_steps),
    }.items():
        try:
            observed = float(qblade_settings[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise MatchedFrozenError(f"Run summary lacks QBlade {name} for {case_id}.") from exc
        if not math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise MatchedFrozenError(f"Run QBlade {name} changed for {case_id}.")
    turbine_raw = str(qblade_settings.get("TURBFILE", ""))
    if not turbine_raw:
        raise MatchedFrozenError(f"Run summary lacks QBlade TURBFILE for {case_id}.")
    turbine_relative = (
        PurePosixPath(sim_relative).parent / PurePosixPath(turbine_raw)
    ).as_posix()
    turbine_path = campaign_root.joinpath(*PurePosixPath(turbine_relative).parts)
    if not _reported_identity_matches(
        input_files.get("turbine"),
        turbine_path,
        expected_suffix=turbine_relative,
        manifest_identity=_manifest_identity_for(file_index, turbine_relative),
    ):
        raise MatchedFrozenError(f"Run turbine identity changed for {case_id}.")
    if (
        bool(summary.get("failed"))
        or bool(summary.get("stopped"))
        or int(summary.get("completed_steps", -1)) != expected_steps
        or int(summary.get("requested_steps", -1)) != expected_steps
        or int(summary.get("sil_vorlap_updates", -1)) != expected_steps
        or summary.get("qblade_execution_mode") != "step-loop"
        or summary.get("sil_vorlap") is not True
        or summary.get("coupling_mode") != "sil-vorlap"
        or int(summary.get("threads", -1)) != THREADS_PER_RUN
        or bool(summary.get("use_opencl"))
    ):
        raise MatchedFrozenError(f"Frozen run did not complete exactly for {case_id}.")
    if (
        summary.get("sil_constant_force") is not False
        or int(summary.get("sil_constant_force_updates", -1)) != 0
        or summary.get("sil_constant_force_actions") != []
        or summary.get("sil_constant_force_max_abs_force") is not None
        or summary.get("force_sign_probe_observations") != []
        or "--sil-constant-force" in command_line
    ):
        raise MatchedFrozenError(
            f"Frozen run did not prove the constant-force path was inactive for {case_id}."
        )
    if not _reported_identity_matches(
        summary.get("result_file_identity"),
        result_path,
        expected_suffix="result.txt",
    ) or not _portable_path_has_suffix(summary.get("result_file"), "result.txt"):
        raise MatchedFrozenError(f"Run result identity changed for {case_id}.")
    if not _portable_path_has_suffix(summary.get("sim"), sim_relative) or not _portable_path_has_suffix(
        summary.get("sil_vorlap_parameter_file"), config_relative
    ):
        raise MatchedFrozenError(f"Run input paths are not case-bound for {case_id}.")
    source_coverage = summary.get("sil_vorlap_source_coverage")
    if (
        not isinstance(source_coverage, list)
        or len(source_coverage) != 1
        or not isinstance(source_coverage[0], Mapping)
    ):
        raise MatchedFrozenError(
            f"Run summary does not have exactly one source-coverage row for {case_id}."
        )
    source_row = source_coverage[0]
    node_ids = config.get("node_ids")
    if not isinstance(node_ids, list) or len(node_ids) != EXPECTED_NODE_COUNT:
        raise MatchedFrozenError(f"Frozen runtime node mapping changed for {case_id}.")
    expected_node_samples = expected_steps * len(node_ids)
    try:
        source_semantics_pass = (
            source_row.get("component") == "BLD_1"
            and source_row.get("airfoil_id") == "CYLINDER"
            and int(source_row.get("samples", -1)) == expected_steps
            and int(source_row.get("node_samples", -1)) == expected_node_samples
            and float(source_row.get("max_response_nodal_force_n", math.nan)) == 0.0
            and float(
                source_row.get("response_force_coefficient_max_abs", math.nan)
            )
            == 0.0
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise MatchedFrozenError(
            f"Frozen source-coverage semantics are invalid for {case_id}."
        ) from exc
    if not source_semantics_pass:
        raise MatchedFrozenError(
            f"Frozen source coverage is not exact BLD_1/CYLINDER zero-response coverage for {case_id}."
        )
    source_limit_counts: dict[str, int] = {}
    for field in response._SOURCE_ZERO_FIELDS:
        try:
            values = [int(row[field]) for row in source_coverage if isinstance(row, Mapping)]
        except (KeyError, TypeError, ValueError) as exc:
            raise MatchedFrozenError(f"Source coverage lacks {field} for {case_id}.") from exc
        if len(values) != len(source_coverage):
            raise MatchedFrozenError(f"Source coverage lacks {field} for {case_id}.")
        source_limit_counts[field] = sum(values)
    if any(source_limit_counts.values()):
        raise MatchedFrozenError(f"Frozen source coverage/limits failed for {case_id}.")
    response_summary = summary.get("sil_vorlap_response_diagnostics_summary")
    if not isinstance(response_summary, Mapping):
        raise MatchedFrozenError(f"Run summary lacks response-disable diagnostics for {case_id}.")
    try:
        response_samples = int(response_summary.get("samples", -1))
        maximum_response_force = float(
            response_summary.get("max_response_nodal_force_n", math.nan)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise MatchedFrozenError(
            f"Response-disable diagnostics are invalid for {case_id}."
        ) from exc
    if (
        response_summary.get("model_type") != "modal_negative_damping"
        or response_samples != 0
        or maximum_response_force != 0.0
    ):
        raise MatchedFrozenError(f"Response model was not demonstrably disabled for {case_id}.")
    limit_fields = (
        "gain_limited_steps",
        "force_limited_steps",
        "gain_limited_node_samples",
        "force_limited_node_samples",
    )
    if any(int(response_summary.get(name, -1)) != 0 for name in limit_fields):
        raise MatchedFrozenError(f"Disabled response diagnostics are inconsistent for {case_id}.")
    history_metadata = summary.get("sil_vorlap_history")
    if (
        summary.get("sil_vorlap_history_mode") != "compact"
        or not isinstance(history_metadata, Mapping)
        or history_metadata.get("mode") != "compact"
        or history_metadata.get("schema") != COMPACT_HISTORY_SCHEMA
        or history_metadata.get("vector_digest_schema") != FORCE_VECTOR_DIGEST_SCHEMA
        or history_metadata.get("streaming") is not True
        or int(history_metadata.get("retained_rows", -1)) != 0
        or int(history_metadata.get("discarded_uncheckpointed_tail_bytes", -1)) != 0
        or int(history_metadata.get("node_count", -1)) != EXPECTED_NODE_COUNT
    ):
        raise MatchedFrozenError(f"Compact history metadata changed for {case_id}.")
    history = audit_compact_frozen_history(
        history_path,
        timestep_s=expected_timestep,
        expected_samples=expected_steps,
        node_ids=config["node_ids"],
    )
    if (
        int(history_metadata.get("rows", -1)) != history["samples"]
        or int(history_metadata.get("columns", -1)) != history["columns"]
        or not _reported_identity_matches(
            history_metadata.get("file_identity"),
            history_path,
            expected_suffix=history_path.name,
        )
        or not _portable_path_has_suffix(history_metadata.get("path"), history_path.name)
    ):
        raise MatchedFrozenError(f"Compact history identity changed for {case_id}.")
    time_diagnostics = summary.get("qblade_export_time_diagnostics")
    if not isinstance(time_diagnostics, Mapping) or int(time_diagnostics.get("samples", -1)) != expected_steps:
        raise MatchedFrozenError(f"QBlade export sample count changed for {case_id}.")
    return {
        "pass": True,
        "case_id": case_id,
        "result_path": result_path.resolve().as_posix(),
        "result_identity": _identity(result_path),
        "run_summary_identity": _identity(summary_path),
        "history_identity": _identity(history_path),
        "summary_path": summary_path.resolve().as_posix(),
        "runtime_build_fingerprint": runtime_build_fingerprint,
        "source_limit_counts": source_limit_counts,
        "response_model_disabled": True,
        "startup_seed_disabled": True,
        "history": history,
    }


def _response_runtime_integrity_pass(audit: Mapping[str, Any]) -> bool:
    history = audit.get("history")
    seed_response = audit.get("startup_seed_response")
    return bool(
        audit.get("evaluated")
        and audit.get("provenance_pass")
        and audit.get("execution_pass")
        and audit.get("source_coverage_complete")
        and audit.get("source_limits_pass")
        and audit.get("response_summary_limits_pass")
        and isinstance(history, Mapping)
        and history.get("runtime_model_pass")
        and history.get("limit_headroom_pass")
        and history.get("compact_lag_provenance_pass")
        and history.get("startup_seed_integrity_pass")
        and audit.get("startup_seed_summary_pass")
        and isinstance(seed_response, Mapping)
        and seed_response.get("pass")
    )


def _locate_response_result_directory(
    roots: Sequence[Path], case_id: str
) -> Path:
    matches = [Path(root).expanduser().resolve() / case_id for root in roots]
    matches = [path for path in matches if path.is_dir()]
    if len(matches) != 1:
        raise MatchedFrozenError(
            f"Expected one response result directory for {case_id}; found {len(matches)}."
        )
    return matches[0]


def _interpolate_curve(rows: Sequence[Mapping[str, Any]], target: float, field: str) -> float:
    ordered = sorted(rows, key=lambda row: float(row["reduced_velocity"]))
    u_star = np.asarray([float(row["reduced_velocity"]) for row in ordered], dtype=float)
    values = np.asarray([float(row[field]) for row in ordered], dtype=float)
    if target < u_star[0] or target > u_star[-1]:
        raise MatchedFrozenError(f"Matched curve does not span reference U*={target}.")
    return float(np.interp(target, u_star, values))


def _absolute_error_skill(response_error: float, baseline_error: float) -> float | None:
    if baseline_error == 0.0:
        return None
    return 1.0 - response_error / baseline_error


def _peak_reference_rows(
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    figure9_path: Path,
    figure10_path: Path,
    model_parameters_path: Path,
) -> list[dict[str, Any]]:
    peaks = response._load_figure10_peaks(figure10_path)
    structures = response.load_structural_cases(model_parameters_path)
    hgast_u, hgast_a = response._read_figure9_series(
        figure9_path, response.CALIBRATION_SERIES
    )
    experiment_u, experiment_a = response._read_figure9_series(
        figure9_path, "experiment_lupi2021"
    )
    calibration_locations = {
        "hgast_hartlen_currie": float(hgast_u[int(np.argmax(hgast_a))]),
        "experiment_lupi2021": float(experiment_u[int(np.argmax(experiment_a))]),
    }
    result = []
    for sc in sorted(stationary.STRUCTURES):
        sc_rows = [row for row in comparison_rows if float(row["scruton_number"]) == sc]
        if len(sc_rows) != len(stationary.REDUCED_VELOCITIES):
            raise MatchedFrozenError(f"Matched comparison curve is incomplete for Sc={sc:.2f}.")
        for series in ("hgast_hartlen_currie", "experiment_lupi2021"):
            reference_amplitude, uncertainty = peaks[(sc, series)]
            if sc == response.CALIBRATION_SC:
                reference_u = calibration_locations[series]
                location_source = "sampled maximum of hash-bound Figure 9 series"
                amplitude_location_colocated = True
                reference_context = (
                    "Figure 10 peak-amplitude row evaluated at the sampled peak U* of "
                    "the paired Figure 9 reduced-velocity/amplitude series"
                )
            else:
                reference_u = structures[sc].u_star_max_figure_digitized
                location_source = (
                    "non-co-located held-out model-parameter peak-location metadata; "
                    "Figure 10 supplies amplitude only"
                )
                amplitude_location_colocated = False
                reference_context = (
                    "Figure 10 amplitude-only row evaluated at a separate model-parameter "
                    "U* location; this non-co-located diagnostic is not validation evidence"
                )
            response_value = _interpolate_curve(
                sc_rows, reference_u, "response_amplitude_over_diameter"
            )
            frozen_value = _interpolate_curve(
                sc_rows,
                reference_u,
                "legacy_frozen_phase_amplitude_over_diameter",
            )
            response_error = abs(response_value - reference_amplitude)
            frozen_error = abs(frozen_value - reference_amplitude)
            zero_error = abs(reference_amplitude)
            skill_frozen = _absolute_error_skill(response_error, frozen_error)
            skill_zero = _absolute_error_skill(response_error, zero_error)
            result.append(
                {
                    "scruton_number": sc,
                    "role": "calibration" if sc == response.CALIBRATION_SC else "held_out_diagnostic",
                    "reference_series": series,
                    "reference_reduced_velocity": reference_u,
                    "reference_location_source": location_source,
                    "reference_amplitude_source": (
                        "hash-bound Figure 10 peak-amplitude row"
                    ),
                    "reference_context": reference_context,
                    "reference_amplitude_location_colocated": amplitude_location_colocated,
                    "figure10_amplitude_only_non_colocated": not amplitude_location_colocated,
                    "reference_amplitude_over_diameter": reference_amplitude,
                    "digitization_uncertainty": uncertainty,
                    "response_at_reference_over_diameter": response_value,
                    "legacy_frozen_phase_at_reference_over_diameter": frozen_value,
                    "zero_viv_at_reference_over_diameter": 0.0,
                    "response_absolute_error": response_error,
                    "legacy_frozen_phase_absolute_error": frozen_error,
                    "zero_viv_absolute_error": zero_error,
                    "response_skill_over_legacy_frozen_phase": skill_frozen,
                    "response_skill_over_zero_viv": skill_zero,
                    "minimum_diagnostic_skill": response.MATERIAL_IMPROVEMENT_MINIMUM_SKILL,
                    "skill_over_legacy_frozen_phase_diagnostic_threshold_met": skill_frozen is not None
                    and skill_frozen >= response.MATERIAL_IMPROVEMENT_MINIMUM_SKILL,
                    "skill_over_zero_viv_diagnostic_threshold_met": skill_zero is not None
                    and skill_zero >= response.MATERIAL_IMPROVEMENT_MINIMUM_SKILL,
                    "eligible_for_hgast_all_sc_peak_diagnostic_gate": (
                        series == "hgast_hartlen_currie"
                    ),
                    "eligible_for_experimental_validation_claim": False,
                    "experimental_validation_assessment": (
                        "not_evaluated_matched_diagnostic_only"
                        if amplitude_location_colocated
                        else "not_evaluated_non_colocated_figure10_amplitude_only"
                    ),
                    "scope": (
                        "Figure 9-supported peak-location amplitude-error diagnostic using "
                        "the Figure 10 peak summary; not independent validation or fatigue "
                        "conservatism"
                        if amplitude_location_colocated
                        else "non-co-located Figure 10 amplitude-only diagnostic at a separate "
                        "model-parameter location; never an experimental-validation pass claim"
                    ),
                }
            )
    return result


def _curve_reference_skill(
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    figure9_path: Path,
    series: str,
) -> dict[str, Any]:
    selected = sorted(
        [
            row
            for row in comparison_rows
            if float(row["scruton_number"]) == response.CALIBRATION_SC
        ],
        key=lambda row: float(row["reduced_velocity"]),
    )
    reference_u, reference_a = response._read_figure9_series(figure9_path, series)
    metrics = response.material_improvement_metrics(
        reference_u=reference_u,
        reference_amplitude=reference_a,
        response_u=[float(row["reduced_velocity"]) for row in selected],
        response_amplitude=[
            float(row["response_amplitude_over_diameter"]) for row in selected
        ],
        frozen_u=[float(row["reduced_velocity"]) for row in selected],
        frozen_amplitude=[
            float(row["legacy_frozen_phase_amplitude_over_diameter"])
            for row in selected
        ],
    )
    metrics = {
        key.replace("frozen", "legacy_frozen_phase"): value
        for key, value in metrics.items()
    }
    metrics.update(
        {
            "reference_series": series,
            "reference_file_sha256": sha256_file(figure9_path),
            "baseline_pairing": (
                "response-aware, legacy frozen-phase bridge comparator, and "
                "zero-VIV evaluated on the shared reference U* grid"
            ),
            "scope": (
                "Sc=5.18 matched stationary amplitude-curve skill diagnostic; not "
                "independent validation or fatigue conservatism"
            ),
        }
    )
    return metrics


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise MatchedFrozenError(f"Refusing to write empty comparison table: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def analyze_campaign(
    manifest_path: Path,
    parent_manifest_path: Path,
    frozen_results_root: Path,
    response_results_roots: Sequence[Path],
    output: Path,
) -> Path:
    """Compare 36 matched stationary cases after fail-closed runtime auditing."""

    state = validate_campaign(manifest_path, parent_manifest_path)
    output = Path(output).expanduser().resolve()
    frozen_results_root = Path(frozen_results_root).expanduser().resolve()
    resolved_response_roots = tuple(
        Path(root).expanduser().resolve() for root in response_results_roots
    )
    protected_output_roots = (
        (state["campaign_root"], "immutable legacy frozen-phase campaign"),
        (state["parent_root"], "immutable parent response campaign"),
        (frozen_results_root, "legacy frozen-phase results input"),
        *(
            (root, "response-aware results input")
            for root in resolved_response_roots
        ),
    )
    for protected_root, protected_label in protected_output_roots:
        try:
            output.relative_to(Path(protected_root).resolve())
        except ValueError:
            continue
        raise MatchedFrozenError(
            f"Analysis output may not be inside {protected_label}: {output}"
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing a nonempty analysis output: {output}")
    if not frozen_results_root.is_dir():
        raise MatchedFrozenError(f"Frozen results root is unavailable: {frozen_results_root}")
    if not resolved_response_roots:
        raise MatchedFrozenError("At least one response results root is required.")
    parent_manifest = state["parent_manifest"]
    parent_root = state["parent_root"]
    parent_files = parent_manifest["_verified_manifest_files"]
    fine_parent = {str(entry["case_id"]): entry for entry in _fine_parent_entries(parent_manifest)}
    frozen_entries = sorted(
        state["manifest"]["cases"],
        key=lambda row: (float(row["scruton_number"]), float(row["reduced_velocity"])),
    )
    comparison_rows: list[dict[str, Any]] = []
    artifact_bindings: list[dict[str, Any]] = []
    runtime_build_fingerprints: list[dict[str, Any]] = []
    all_stationary = True
    for frozen_entry in frozen_entries:
        case_id = str(frozen_entry["case_id"])
        response_entry = fine_parent.get(case_id)
        if response_entry is None:
            raise MatchedFrozenError(f"No exactly co-located response case for {case_id}.")
        for field in (
            "scruton_number",
            "reduced_velocity",
            "mean_inflow_mps",
            "timestep_s",
            "structure",
            "sim_path",
            "parameter_file",
        ):
            if frozen_entry.get(field) != response_entry.get(field):
                raise MatchedFrozenError(f"Response/frozen co-location changed in {field}: {case_id}")

        frozen_directory = frozen_results_root / case_id
        frozen_audit = audit_frozen_runtime_case(
            campaign_root=state["campaign_root"],
            file_index=state["file_index"],
            entry=frozen_entry,
            result_directory=frozen_directory,
        )
        response_directory = _locate_response_result_directory(
            resolved_response_roots, case_id
        )
        response_result = response_directory / "result.txt"
        response_audit = response.runtime_case_audit(
            response_directory,
            entry=response_entry,
            campaign_root=parent_root,
            manifest_files=parent_files,
            result_path=response_result,
            expected_history_mode="compact",
        )
        if not _response_runtime_integrity_pass(response_audit):
            raise MatchedFrozenError(
                f"Response runtime integrity failed for {case_id}: "
                + "; ".join(str(item) for item in response_audit.get("problems", []))
            )
        response_summary_path = Path(str(response_audit["summary_path"]))
        response_summary = _load_json_object(
            response_summary_path, f"response run summary {case_id}"
        )
        response_runtime_build = _runtime_build_fingerprint(
            response_summary, case_id=case_id
        )
        frozen_runtime_build = frozen_audit["runtime_build_fingerprint"]
        if response_runtime_build != frozen_runtime_build:
            raise MatchedFrozenError(
                f"Response/comparator runtime build differs for {case_id}."
            )
        runtime_build_fingerprints.append(response_runtime_build)
        response_metrics = response.response_amplitude_from_result(
            response_result,
            natural_frequency_hz=float(response_entry["structure"]["natural_frequency_hz"]),
            response_window_cycles=TERMINAL_WINDOW_CYCLES,
            expected_timestep_s=float(response_entry["timestep_s"]),
            expected_samples=int(response_entry["num_steps"]),
        )
        frozen_result = frozen_directory / "result.txt"
        frozen_metrics = response.response_amplitude_from_result(
            frozen_result,
            natural_frequency_hz=float(frozen_entry["structure"]["natural_frequency_hz"]),
            response_window_cycles=TERMINAL_WINDOW_CYCLES,
            expected_timestep_s=float(frozen_entry["timestep_s"]),
            expected_samples=int(frozen_entry["num_steps"]),
        )
        response_stationary = bool(response_metrics["stationarity_pass"]) and bool(
            response_audit["history"].get("response_estimator_stationarity_pass")
        )
        frozen_stationary = bool(frozen_metrics["stationarity_pass"])
        all_stationary = all_stationary and response_stationary and frozen_stationary
        comparison_rows.append(
            {
                "case_id": case_id,
                "scruton_number": frozen_entry["scruton_number"],
                "role": frozen_entry["role"],
                "reduced_velocity": frozen_entry["reduced_velocity"],
                "mean_inflow_mps": frozen_entry["mean_inflow_mps"],
                "timestep_s": frozen_entry["timestep_s"],
                "terminal_window_cycles": TERMINAL_WINDOW_CYCLES,
                "co_location_pass": True,
                "runtime_build_co_location_pass": True,
                "runtime_build_fingerprint_sha256": response_runtime_build[
                    "fingerprint_sha256"
                ],
                "native_callback_zero_pass": True,
                "source_fingerprint_sha256": frozen_entry["source_fingerprint_sha256"],
                "response_amplitude_over_diameter": response_metrics[
                    "tip_peak_amplitude_over_diameter"
                ],
                "legacy_frozen_phase_amplitude_over_diameter": frozen_metrics[
                    "tip_peak_amplitude_over_diameter"
                ],
                "zero_viv_amplitude_over_diameter": 0.0,
                "response_minus_legacy_frozen_phase_amplitude_over_diameter": float(
                    response_metrics["tip_peak_amplitude_over_diameter"]
                )
                - float(frozen_metrics["tip_peak_amplitude_over_diameter"]),
                "response_frequency_over_fn": response_metrics[
                    "terminal_dominant_frequency_over_fn"
                ],
                "legacy_frozen_phase_frequency_over_fn": frozen_metrics[
                    "terminal_dominant_frequency_over_fn"
                ],
                "response_mode_band_power_fraction": response_metrics[
                    "terminal_mode_band_power_fraction"
                ],
                "legacy_frozen_phase_mode_band_power_fraction": frozen_metrics[
                    "terminal_mode_band_power_fraction"
                ],
                "response_stationarity_pass": response_stationary,
                "legacy_frozen_phase_stationarity_pass": frozen_stationary,
                "response_runtime_integrity_pass": True,
                "legacy_frozen_phase_runtime_integrity_pass": frozen_audit["pass"],
            }
        )
        artifact_bindings.append(
            {
                "case_id": case_id,
                "response_result": response._actual_file_identity(response_result),
                "response_run_summary": response._actual_file_identity(
                    Path(response_audit["summary_path"])
                ),
                "response_history": response._actual_file_identity(
                    Path(response_audit["history_path"])
                ),
                "legacy_frozen_phase_result": frozen_audit["result_identity"],
                "legacy_frozen_phase_run_summary": frozen_audit[
                    "run_summary_identity"
                ],
                "legacy_frozen_phase_history": frozen_audit["history_identity"],
                "runtime_build_fingerprint_sha256": response_runtime_build[
                    "fingerprint_sha256"
                ],
            }
        )

    if len(comparison_rows) != EXPECTED_CASE_COUNT:
        raise MatchedFrozenError("Matched analysis did not collect exactly 36 cases.")
    unique_runtime_builds = {
        row["fingerprint_sha256"] for row in runtime_build_fingerprints
    }
    if len(unique_runtime_builds) != 1:
        raise MatchedFrozenError(
            "Completed matched cases did not use one common runtime build."
        )
    runtime_build_fingerprint = runtime_build_fingerprints[0]
    reference_files = parent_manifest["reference"]["reference_files"]
    figure9_path = response._resolved_reference_path(
        reference_files["sc5p18_figure9_digitized.csv"]
    )
    figure10_path = response._resolved_reference_path(
        reference_files["all_sc_figure10_peak_digitized.csv"]
    )
    model_parameters_path = response._resolved_reference_path(
        reference_files["vlastos2026_model_parameters.csv"]
    )
    if all_stationary:
        hgast_curve_skill = _curve_reference_skill(
            comparison_rows,
            figure9_path=figure9_path,
            series=response.CALIBRATION_SERIES,
        )
        experiment_curve_skill = _curve_reference_skill(
            comparison_rows,
            figure9_path=figure9_path,
            series="experiment_lupi2021",
        )
        peak_reference_rows = _peak_reference_rows(
            comparison_rows,
            figure9_path=figure9_path,
            figure10_path=figure10_path,
            model_parameters_path=model_parameters_path,
        )
        status = "diagnostic_complete"
    else:
        hgast_curve_skill = {
            "evaluated": False,
            "pass": False,
            "reason": "all matched response and frozen terminal windows must be stationary",
        }
        experiment_curve_skill = dict(hgast_curve_skill)
        peak_reference_rows = []
        status = "incomplete_nonstationary"
    hgast_peak_rows = [
        row
        for row in peak_reference_rows
        if row["reference_series"] == "hgast_hartlen_currie"
        and row["eligible_for_hgast_all_sc_peak_diagnostic_gate"]
    ]
    hgast_peak_screen_pass = bool(hgast_peak_rows) and all(
        bool(row["skill_over_legacy_frozen_phase_diagnostic_threshold_met"])
        and bool(row["skill_over_zero_viv_diagnostic_threshold_met"])
        for row in hgast_peak_rows
    )
    summary_payload = {
        "schema": ANALYSIS_SCHEMA,
        "status": status,
        "scope": (
            "matched stationary amplitude-skill diagnostic against response-aware and the legacy "
            "frozen-phase bridge comparator, "
            "zero-VIV, and hGAST; experimental Figure 10 amplitude-only rows are retained "
            "only as explicitly non-co-located context"
        ),
        "claim_boundary": {
            "fatigue_conservatism_evaluated": False,
            "fatigue_conservative_validation": False,
            "experimental_validation_claimed": False,
            "certification_suitability_claimed": False,
            "reason": (
                "tip-amplitude skill does not establish local stress ranges, cycle counts, DEL, "
                "Miner damage, conservative fatigue bias, or experimental validation from "
                "non-co-located Figure 10 amplitude-only rows"
            ),
        },
        "campaign_binding": {
            "legacy_frozen_phase_manifest_path": Path(manifest_path).expanduser().resolve().as_posix(),
            "legacy_frozen_phase_manifest_sha256": state["manifest_sha256"],
            "legacy_frozen_phase_manifest_payload_sha256": state[
                "manifest_payload_sha256"
            ],
            "parent_response_manifest_path": Path(parent_manifest_path).expanduser().resolve().as_posix(),
            "parent_response_manifest_sha256": state["parent_manifest_sha256"],
            "parent_stationary_protocol_sha256": state["parent_protocol_sha256"],
            "source_fingerprint_sha256": state["source_fingerprint_sha256"],
        },
        "runtime_evidence": {
            "expected_case_count": EXPECTED_CASE_COUNT,
            "completed_case_count": len(comparison_rows),
            "exact_sc_u_fine_dt_co_location_pass": True,
            "exact_runtime_build_co_location_pass": True,
            "single_runtime_build_across_all_cases_pass": True,
            "native_callback_zero_pass": True,
            "runtime_build_fingerprint": runtime_build_fingerprint,
            "all_runtime_integrity_pass": True,
            "all_terminal_stationarity_pass": all_stationary,
            "terminal_window_cycles": TERMINAL_WINDOW_CYCLES,
            "frozen_cap_natural_cycles": CAP_NATURAL_CYCLES,
            "sil_vorlap_history_mode": "compact",
            "one_step_lag_digest_bound": True,
        },
        "diagnostic_skill": {
            "minimum_skill_threshold": response.MATERIAL_IMPROVEMENT_MINIMUM_SKILL,
            "sc5p18_hgast_curve": hgast_curve_skill,
            "sc5p18_experiment_curve": experiment_curve_skill,
            "hgast_all_sc_peak_diagnostic_gate_pass": hgast_peak_screen_pass,
            "hgast_all_sc_peak_diagnostic_gate_is_validation": False,
            "figure10_experiment_rows_enter_gate": False,
            "interpretation": (
                "positive skill means smaller matched amplitude error than the named baseline; "
                "the all-Sc gate is hGAST-only and diagnostic; Figure 10 experimental rows "
                "are non-co-located context, not validation or a fatigue safety factor"
            ),
        },
        "reference_files": {
            "figure9_sha256": sha256_file(figure9_path),
            "figure10_sha256": sha256_file(figure10_path),
            "model_parameters_sha256": sha256_file(model_parameters_path),
        },
        "case_artifact_bindings": artifact_bindings,
        "peak_reference_comparisons": peak_reference_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    comparison_path = output / "legacy_frozen_phase_comparison_by_case.csv"
    _write_csv(comparison_path, comparison_rows)
    output_identities: dict[str, Any] = {
        comparison_path.name: _identity(
            comparison_path, relative=comparison_path.name
        )
    }
    if peak_reference_rows:
        peak_path = output / "legacy_frozen_phase_reference_peak_skill.csv"
        _write_csv(peak_path, peak_reference_rows)
        output_identities[peak_path.name] = _identity(
            peak_path, relative=peak_path.name
        )
    summary_payload["analysis_outputs"] = output_identities
    summary = {
        **summary_payload,
        "analysis_payload_sha256": sha256_payload(summary_payload),
    }
    summary_path = _write_json_exclusive(
        output / "legacy_frozen_phase_skill_summary.json", summary
    )
    return summary_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generation = subparsers.add_parser(
        "generate", help="Generate immutable legacy frozen-phase comparator inputs."
    )
    generation.add_argument("--parent-campaign", type=Path, required=True)
    generation.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate parent and frozen inputs.")
    validate.add_argument("--campaign", type=Path, required=True)
    validate.add_argument("--parent-campaign", type=Path, required=True)

    plan = subparsers.add_parser("plan", help="Write a sequential four-thread dry run plan.")
    plan.add_argument("--campaign", type=Path, required=True)
    plan.add_argument("--parent-campaign", type=Path, required=True)
    plan.add_argument("--results", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument(
        "--qblade-root",
        type=Path,
        default=REPO_ROOT / "run_headless" / "QBladeCE_2.0.9.7",
    )

    run = subparsers.add_parser(
        "run", help="Write the exact plan and run it only with --execute."
    )
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--parent-campaign", type=Path, required=True)
    run.add_argument("--results", type=Path, required=True)
    run.add_argument("--plan-output", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument(
        "--qblade-root",
        type=Path,
        default=REPO_ROOT / "run_headless" / "QBladeCE_2.0.9.7",
    )

    analysis = subparsers.add_parser("analyze", help="Analyze complete matched stationary results.")
    analysis.add_argument("--campaign", type=Path, required=True)
    analysis.add_argument("--parent-campaign", type=Path, required=True)
    analysis.add_argument("--frozen-results", type=Path, required=True)
    analysis.add_argument("--response-results", type=Path, nargs="+", required=True)
    analysis.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        manifest = generate_campaign(args.output, args.parent_campaign)
        print(json.dumps({"status": "valid", "manifest": str(manifest)}, sort_keys=True))
        return 0
    if args.command == "validate":
        state = validate_campaign(args.campaign, args.parent_campaign)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "case_count": state["case_count"],
                    "manifest_sha256": state["manifest_sha256"],
                    "source_fingerprint_sha256": state["source_fingerprint_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "plan":
        _reject_write_within_immutable_roots(
            args.output,
            (Path(args.campaign).parent, Path(args.parent_campaign).parent),
            label="Run-plan output",
        )
        _reject_plan_within_results(args.output, args.results)
        plan = build_run_plan(
            args.campaign,
            args.parent_campaign,
            args.results,
            qblade_root=args.qblade_root,
        )
        _write_json_exclusive(args.output, plan)
        print(json.dumps({"status": "dry_plan", "commands": len(plan["commands"])}, sort_keys=True))
        return 0
    if args.command == "run":
        _reject_write_within_immutable_roots(
            args.plan_output,
            (Path(args.campaign).parent, Path(args.parent_campaign).parent),
            label="Run-plan output",
        )
        _reject_plan_within_results(args.plan_output, args.results)
        plan = build_run_plan(
            args.campaign,
            args.parent_campaign,
            args.results,
            qblade_root=args.qblade_root,
        )
        plan_path = _write_json_exclusive(args.plan_output, plan)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "status": "dry_plan",
                        "commands": len(plan["commands"]),
                        "plan": str(plan_path),
                        "execution_requires": "--execute",
                    },
                    sort_keys=True,
                )
            )
            return 0
        receipt = execute_run_plan(
            plan,
            args.campaign,
            args.parent_campaign,
            args.results,
            qblade_root=args.qblade_root,
        )
        print(
            json.dumps(
                {
                    "status": "executed",
                    "commands": len(plan["commands"]),
                    "plan": str(plan_path),
                    "receipt": str(receipt),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "analyze":
        summary = analyze_campaign(
            args.campaign,
            args.parent_campaign,
            args.frozen_results,
            args.response_results,
            args.output,
        )
        print(json.dumps({"status": "analyzed", "summary": str(summary)}, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
