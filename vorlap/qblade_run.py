"""Importable one-case QBlade run service."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Literal

from .qblade_files import read_qblade_keyword, resolve_qblade_path
from .qblade_modal_decay import SilModalBurstCoupler, _APPLIED_RECORD_KEYS
from .qblade_run_support import (
    build_checkpoint_identity,
    build_reproducibility_metadata,
    canonical_fingerprint,
    checkpoint_directory,
    clear_native_bridge_log,
    ensure_qblade_loader_env,
    file_sha256,
    load_json_file,
    load_qblade_api,
    load_resume_checkpoint,
    load_strict_modal_external_config,
    load_vorlap_config_for_sim,
    native_callback_evidence,
    preflight_qblade_wind_input,
    require_zero_native_callback_evidence,
    require_modal_native_bridge_runtime_contract,
    require_modal_qblade_runtime_dependency_contract,
    require_modal_runtime_identity_contract,
    qblade_runtime_dependency_manifest,
    validate_loaded_modal_burst_resume_position,
    validate_modal_burst_resume_position_record,
    write_restart_checkpoint,
    atomic_write_json,
)
from .qblade_sil import (
    SilConstantForceCoupler,
    SilVorLapCoupler,
    force_sign_probe_observations,
    parse_constant_force_action,
    qblade_result_time_diagnostics,
    qblade_result_rpm_diagnostics,
    resolve_sil_history_mode,
)
from .performance_timing import AggregatePerformanceTiming, profiling_requested


QBladeCouplingMode = Literal[
    "sil-vorlap",
    "sil-constant-force",
    "sil-modal-burst",
    "qblade-external",
    "none",
]
QBladeExecutionMode = Literal["step-loop", "run-full-simulation"]


@dataclass(frozen=True)
class QBladeRunRequest:
    """Inputs for running one QBlade case through the SIL wrapper."""

    sim_path: str | Path
    qblade_root: str | Path
    output_dir: str | Path
    result_name: str | None = None
    steps: int | None = None
    threads: int = 1
    device: int = 1
    group_size: int = 32
    use_opencl: bool = False
    store_project: bool = False
    result_filter: str | Path | None = None
    checkpoint_every: int = 0
    checkpoint_dir: str | Path | None = None
    resume_from_checkpoint: bool = False
    resume_checkpoint: str | Path | None = None
    resume_initialize: bool = False
    completed_checkpoint_export_only: bool = False
    allow_partial: bool = False
    sil_vorlap: bool = False
    sil_modal_burst_plan: str | Path | None = None
    sil_constant_force_specs: Sequence[str] = ()
    coupling_mode: QBladeCouplingMode | None = None
    qblade_execution_mode: QBladeExecutionMode = "step-loop"
    case_metadata_path: str | Path | None = None
    case_metadata: dict | None = None
    progress_every: int = 100
    record_sil_vorlap_history: bool = False
    sil_vorlap_history_mode: Literal["off", "full", "compact"] | None = None
    command_line: Sequence[str] = ()
    manage_loader_env: bool = True
    qblade_api_loader: Callable[[Path], tuple[Any, Path]] | None = None
    message_callback: Callable[[str], None] | None = None
    stop_requested: Callable[[], bool] | None = None


@dataclass(frozen=True)
class QBladeRunResult:
    """Outputs from running one QBlade case."""

    summary: dict[str, object]
    summary_path: Path
    result_path: Path
    run_log_path: Path


@dataclass(frozen=True)
class QBladeModalBurstRunSummary:
    """Typed provenance for the deliberately narrow modal-decay SIL path."""

    enabled: bool
    plan_file: str | None
    identity: dict[str, object] | None
    case_contract: dict[str, object] | None
    timing_witnesses: list[dict[str, object]]
    finalized: bool
    final_summary: dict[str, object] | None
    runtime_contract_final: dict[str, object] | None
    runtime_identity_audit_final: dict[str, object] | None
    runtime_dependency_fingerprint: str | None
    loaded_resume_position_audit: dict[str, object] | None
    physical_consumption_claim: str


@dataclass(frozen=True)
class QBladeRunSummary:
    """Stable JSON payload for one QBlade/VorLap run summary."""

    sim: str
    qblade_root: str
    qblade_library: str
    coupling_mode: QBladeCouplingMode
    qblade_execution_mode: QBladeExecutionMode
    requested_steps: int
    completed_steps: int
    stopped: bool
    stop_message: str | None
    failed: bool
    failure_message: str | None
    allow_partial: bool
    threads: int
    use_opencl: bool
    restart: dict[str, object]
    stored_project: str | None
    result_filter: str | None
    result_filter_identity: dict[str, object] | None
    result_file: str
    result_file_identity: dict[str, object] | None
    exported_results: bool
    run_log: str
    sil_vorlap: bool
    sil_vorlap_updates: int
    sil_vorlap_parameter_file: str | None
    sil_vorlap_geometry_audit_file: str | None
    sil_vorlap_max_abs_force: float | None
    sil_vorlap_message: str | None
    sil_vorlap_velocity_diagnostics: dict[str, object] | None
    sil_vorlap_timing_diagnostics: dict[str, object] | None
    sil_vorlap_cache_stats: dict[str, int] | None
    sil_vorlap_response_diagnostics_summary: dict[str, object] | None
    sil_vorlap_source_bounds_policy: str | None
    sil_vorlap_source_frequency_cap_hz: float | None
    sil_vorlap_source_coverage: list[dict[str, object]]
    sil_vorlap_history_mode: str
    sil_vorlap_history: dict[str, object] | None
    sil_constant_force: bool
    sil_constant_force_updates: int
    sil_constant_force_actions: Sequence[dict[str, object]]
    sil_constant_force_max_abs_force: float | None
    force_sign_probe_observations: list[dict[str, object]]
    qblade_export_time_diagnostics: dict[str, object]
    qblade_export_rpm_diagnostics: dict[str, object]
    native_callback_evidence: dict[str, object]
    reproducibility: dict[str, object]
    elapsed_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _emit(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _resolve_result_filter(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QBlade result filter does not exist: {path}")
    channels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]
    if not channels:
        raise ValueError(f"QBlade result filter has no variable names: {path}")
    if len(channels) != len(set(channels)):
        raise ValueError(f"QBlade result filter contains duplicate variable names: {path}")
    return path


def _validate_completed_checkpoint_export_only(
    checkpoint: dict[str, object] | None,
    *,
    requested_steps: int,
    sim_path: Path,
    output_dir: Path,
    result_name: str,
    coupling_mode: str,
) -> dict[str, object]:
    if checkpoint is None:
        raise FileNotFoundError(
            "--completed-checkpoint-export-only requires an existing final checkpoint."
        )
    completed_steps = int(checkpoint.get("completed_steps", 0) or 0)
    checkpoint_requested_steps = int(checkpoint.get("requested_steps", 0) or 0)
    if completed_steps != requested_steps or checkpoint_requested_steps != requested_steps:
        raise ValueError(
            "Completed-checkpoint export recovery requires completed_steps == "
            f"requested_steps == {requested_steps}; checkpoint records "
            f"{completed_steps}/{checkpoint_requested_steps}."
        )
    expected_strings = {
        "sim": str(sim_path),
        "output_dir": str(output_dir),
        "result_name": result_name,
        "coupling_mode": coupling_mode,
    }
    mismatches = [
        f"{key}={checkpoint.get(key)!r}, expected {expected!r}"
        for key, expected in expected_strings.items()
        if str(checkpoint.get(key, "")) != expected
    ]
    if mismatches:
        raise ValueError(
            "Completed-checkpoint export recovery rejected case metadata mismatch: "
            + "; ".join(mismatches)
        )
    return checkpoint


def _restore_sil_vorlap_resume_state(
    sil_vorlap: SilVorLapCoupler | None,
    resume_checkpoint: dict[str, object] | None,
    *,
    completed_checkpoint_export_only: bool,
) -> None:
    """Require causal SIL state whenever a resumed run can execute wrapper logic."""
    if sil_vorlap is None or resume_checkpoint is None or completed_checkpoint_export_only:
        return
    sil_state = resume_checkpoint.get("sil_vorlap_state")
    if not isinstance(sil_state, dict):
        raise ValueError(
            "SIL-VorLap resume checkpoint is missing a valid sil_vorlap_state; refusing to "
            "resume with a fresh force queue or runtime state."
        )
    sil_vorlap.restore_checkpoint_state(sil_state)


def _restore_modal_burst_resume_state(
    modal_burst: SilModalBurstCoupler | None,
    resume_checkpoint: dict[str, object] | None,
) -> None:
    """Require the exact scheduled-force state before any resumed wrapper call."""

    if modal_burst is None or resume_checkpoint is None:
        return
    modal_state = resume_checkpoint.get("sil_modal_burst_state")
    if not isinstance(modal_state, dict):
        raise ValueError(
            "Modal-burst resume checkpoint is missing a valid sil_modal_burst_state; "
            "refusing to resume with a fresh schedule."
        )
    modal_burst.restore_checkpoint_state(modal_state)


def _restore_modal_timing_witnesses(
    modal_burst: SilModalBurstCoupler | None,
    resume_checkpoint: dict[str, object] | None,
    *,
    plan_path: Path | None,
) -> dict[int, dict[str, object]]:
    if modal_burst is None or resume_checkpoint is None:
        return {}
    raw = resume_checkpoint.get("sil_modal_burst_timing_witnesses")
    if not isinstance(raw, list):
        raise ValueError(
            "Modal-burst resume checkpoint is missing timing witnesses."
        )
    witnesses: dict[int, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Modal-burst checkpoint timing witness must be an object.")
        if set(item) != set(_APPLIED_RECORD_KEYS):
            raise ValueError(
                "Modal-burst checkpoint timing witness must contain the exact applied-record "
                "schema keys."
            )
        index = item.get("wrapper_step_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Modal-burst checkpoint timing witness has an invalid wrapper index.")
        if index in witnesses:
            raise ValueError(
                f"Modal-burst checkpoint contains duplicate timing witness {index}."
            )
        witnesses[index] = dict(item)
    expected_indices = {0} if modal_burst.wrapper_updates == 1 else {0, 1}
    if modal_burst.wrapper_updates > 0:
        expected_indices.add(modal_burst.wrapper_updates - 1)
    if set(witnesses) != expected_indices:
        raise ValueError(
            "Modal-burst checkpoint timing witnesses do not contain exactly wrapper 0, "
            "wrapper 1 when available, and the latest wrapper."
        )
    observed_fingerprint = resume_checkpoint.get(
        "sil_modal_burst_timing_witnesses_fingerprint"
    )
    if observed_fingerprint != canonical_fingerprint(raw):
        raise ValueError("Modal-burst checkpoint timing-witness fingerprint is invalid.")
    if plan_path is None:
        raise ValueError("Modal-burst resume requires the immutable plan path.")

    class _WitnessQBlade:
        def __init__(self, record: dict[str, object]) -> None:
            self.record = record
            self.action_count = 0

        def getCustomData_at_num(
            self, variable: bytes, _position: float, _index: int
        ) -> float:
            if variable == b"Time [s]":
                return float(self.record["qblade_time_pre_advance_s"])
            if variable == b"LSS Azimuthal Pos. [deg]":
                return float(self.record["azimuth_deg"])
            return float("nan")

        def setExternalAction(self, *_args) -> None:
            self.action_count += 1

    # Re-execute the deterministic wrapper-0/1 design/application contract on a
    # fresh coupler.  The latest witness is independently reconstructed by the
    # core checkpoint-state restore below.
    witness_clone = SilModalBurstCoupler.from_plan_file(plan_path)
    for index in sorted(expected_indices & {0, 1}):
        fake = _WitnessQBlade(witnesses[index])
        witness_clone.step(fake, index)
        if fake.action_count != 117 or witness_clone.last_record != witnesses[index]:
            raise ValueError(
                f"Modal-burst checkpoint timing witness {index} is not deterministic."
            )
    if modal_burst.last_record is not None:
        last = witnesses[max(witnesses)]
        if last != modal_burst.last_record:
            raise ValueError(
                "Modal-burst checkpoint latest timing witness does not match coupler state."
            )
    validate_modal_burst_resume_position_record(
        resume_checkpoint.get("sil_modal_burst_resume_position"),
        completed_steps=modal_burst.wrapper_updates,
        timing_witnesses=[witnesses[index] for index in sorted(witnesses)],
    )
    return witnesses


_MODAL_CHECKPOINT_TOP_LEVEL_KEYS = frozenset(
    {
        "version",
        "created_unix",
        "reason",
        "sim",
        "qblade_root",
        "output_dir",
        "result_name",
        "coupling_mode",
        "requested_steps",
        "completed_steps",
        "qblade_project",
        "qblade_project_sha256",
        "qblade_project_size_bytes",
        "threads",
        "use_opencl",
        "checkpoint_identity",
        "checkpoint_identity_fingerprint",
        "qblade_resume_mode",
        "qblade_resume_note",
        "previous_checkpoint",
        "sil_modal_burst_state",
        "sil_modal_burst_timing_witnesses",
        "sil_modal_burst_timing_witnesses_fingerprint",
        "zero_native_callback_evidence",
        "zero_native_callback_evidence_fingerprint",
        "native_bridge_runtime_audit",
        "native_bridge_runtime_audit_fingerprint",
        "qblade_runtime_dependency_audit",
        "qblade_runtime_dependency_audit_fingerprint",
        "modal_runtime_identity_audit",
        "modal_runtime_identity_audit_fingerprint",
        "sil_modal_burst_resume_position",
        "sil_modal_burst_resume_position_fingerprint",
        "_metadata_path",
    }
)


def _validate_modal_checkpoint_envelope(
    checkpoint: dict[str, object],
    *,
    sim_path: Path,
    qblade_root: Path,
    output_dir: Path,
    result_name: str,
    requested_steps: int,
    threads: int,
    use_opencl: bool,
    checkpoint_identity: dict[str, object],
    modal_plan_path: Path,
) -> None:
    if set(checkpoint) != set(_MODAL_CHECKPOINT_TOP_LEVEL_KEYS):
        raise ValueError(
            "Modal-burst v4 checkpoint must contain the exact top-level schema fields; "
            f"missing={sorted(set(_MODAL_CHECKPOINT_TOP_LEVEL_KEYS) - set(checkpoint))}, "
            f"unexpected={sorted(set(checkpoint) - set(_MODAL_CHECKPOINT_TOP_LEVEL_KEYS))}."
        )
    exact = {
        "version": 4,
        "sim": str(sim_path),
        "qblade_root": str(qblade_root),
        "output_dir": str(output_dir),
        "result_name": result_name,
        "coupling_mode": "sil-modal-burst",
        "requested_steps": int(requested_steps),
        "threads": int(threads),
        "use_opencl": bool(use_opencl),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_identity_fingerprint": checkpoint_identity.get("fingerprint"),
        "qblade_resume_mode": "loadProject",
    }
    mismatches = [
        key for key, expected in exact.items() if checkpoint.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "Modal-burst v4 checkpoint case/schema mismatch for: "
            + ", ".join(mismatches)
        )
    if checkpoint.get("reason") not in {"periodic", "stopped", "final"}:
        raise ValueError("Modal-burst v4 checkpoint reason is invalid.")
    completed = checkpoint.get("completed_steps")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise ValueError("Modal-burst v4 completed_steps must be an integer.")
    if completed < 1 or completed > requested_steps:
        raise ValueError("Modal-burst v4 completed_steps is outside the planned range.")
    evidence = checkpoint.get("zero_native_callback_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Modal-burst v4 checkpoint lacks zero-native callback evidence.")
    evidence_fingerprint = checkpoint.get(
        "zero_native_callback_evidence_fingerprint"
    )
    if evidence_fingerprint != canonical_fingerprint(evidence):
        raise ValueError("Modal-burst v4 zero-native evidence fingerprint is invalid.")
    require_zero_native_callback_evidence(
        evidence,
        context="modal-burst resume checkpoint",
    )
    current_evidence = native_callback_evidence(sim_path, qblade_root=qblade_root)
    require_zero_native_callback_evidence(
        current_evidence,
        context="modal-burst resume preflight",
        reject_existing_log=True,
    )
    if evidence != current_evidence:
        raise ValueError(
            "Modal-burst v4 zero-native evidence no longer matches the staged case/runtime."
        )
    bridge_audit = checkpoint.get("native_bridge_runtime_audit")
    if (
        not isinstance(bridge_audit, dict)
        or checkpoint.get("native_bridge_runtime_audit_fingerprint")
        != canonical_fingerprint(bridge_audit)
    ):
        raise ValueError("Modal-burst v4 native-bridge runtime audit is invalid.")
    current_bridge_audit = require_modal_native_bridge_runtime_contract(
        sim_path,
        qblade_root,
    )
    if bridge_audit != current_bridge_audit:
        raise ValueError(
            "Modal-burst v4 native-bridge runtime audit no longer matches the staged "
            "case/runtime."
        )
    dependency_audit = checkpoint.get("qblade_runtime_dependency_audit")
    if (
        not isinstance(dependency_audit, dict)
        or checkpoint.get("qblade_runtime_dependency_audit_fingerprint")
        != canonical_fingerprint(dependency_audit)
    ):
        raise ValueError("Modal-burst v4 QBlade dependency audit is invalid.")
    current_dependency_audit = require_modal_qblade_runtime_dependency_contract(
        qblade_root,
        checkpoint_identity.get("qblade_runtime_dependency_fingerprint"),
    )
    if dependency_audit != current_dependency_audit:
        raise ValueError(
            "Modal-burst v4 QBlade dependency audit no longer matches the staged runtime."
        )
    runtime_identity_audit = checkpoint.get("modal_runtime_identity_audit")
    if (
        not isinstance(runtime_identity_audit, dict)
        or set(runtime_identity_audit)
        != {
            "schema",
            "passed",
            "plan_file",
            "modal_burst_identity_fingerprint",
            "checkpoint_identity_fingerprint",
        }
        or checkpoint.get("modal_runtime_identity_audit_fingerprint")
        != canonical_fingerprint(runtime_identity_audit)
    ):
        raise ValueError("Modal-burst v4 complete runtime-identity audit is invalid.")
    expected_runtime_identity_audit = {
        "schema": "vorlap-qblade-modal-runtime-identity-audit-v1",
        "passed": True,
        "plan_file": str(modal_plan_path.resolve()),
        "modal_burst_identity_fingerprint": checkpoint_identity[
            "modal_burst_identity"
        ]["fingerprint"],
        "checkpoint_identity_fingerprint": checkpoint_identity["fingerprint"],
    }
    if runtime_identity_audit != expected_runtime_identity_audit:
        raise ValueError(
            "Modal-burst v4 complete runtime-identity audit does not match this plan/runtime."
        )
    resume_position = checkpoint.get("sil_modal_burst_resume_position")
    if (
        not isinstance(resume_position, dict)
        or checkpoint.get("sil_modal_burst_resume_position_fingerprint")
        != canonical_fingerprint(resume_position)
    ):
        raise ValueError("Modal-burst v4 resume-position fingerprint is invalid.")


def _retain_modal_timing_witness(
    witnesses: dict[int, dict[str, object]],
    record: dict[str, object],
) -> None:
    index = int(record["wrapper_step_index"])
    witnesses[index] = json.loads(json.dumps(record, sort_keys=True))
    keep = {0, 1, max(witnesses)}
    for stale in set(witnesses) - keep:
        del witnesses[stale]


def _ordered_modal_timing_witnesses(
    witnesses: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    return [witnesses[index] for index in sorted(witnesses)]


def _discard_contaminated_modal_checkpoint(
    checkpoint: dict[str, object] | None,
    checkpoint_dir: Path,
) -> list[str]:
    """Remove only the task-owned checkpoint pair that failed the final callback audit."""

    removed: list[str] = []
    if checkpoint is not None:
        candidates = [
            Path(str(checkpoint.get("_metadata_path", ""))),
            Path(str(checkpoint.get("qblade_project", ""))),
        ]
        for candidate in candidates:
            if not str(candidate) or candidate.name in {"", "."}:
                continue
            try:
                candidate.resolve().relative_to(checkpoint_dir.resolve())
            except ValueError:
                continue
            if candidate.is_file():
                candidate.unlink()
                removed.append(str(candidate))
    latest_pointer = checkpoint_dir / "latest_checkpoint.json"
    if latest_pointer.is_file():
        latest_pointer.unlink()
        removed.append(str(latest_pointer))
    return removed


def _sil_vorlap_cache_stats(sil_vorlap: SilVorLapCoupler | None) -> dict[str, int] | None:
    if sil_vorlap is None:
        return None
    runtime = getattr(sil_vorlap, "runtime", None)
    controller = getattr(runtime, "controller", None)
    cache_stats = getattr(controller, "cache_stats", {}) or {}
    return {
        "hits": int(cache_stats.get("hits", 0) or 0),
        "misses": int(cache_stats.get("misses", 0) or 0),
    }


def _sil_vorlap_response_diagnostics_summary(
    sil_vorlap: SilVorLapCoupler | None,
) -> dict[str, object] | None:
    """Return the response-closure audit block without exposing mutable state."""

    if sil_vorlap is None:
        return None
    hvawt_adapter = getattr(sil_vorlap, "hvawt_response_adapter", None)
    if hvawt_adapter is not None:
        return dict(hvawt_adapter.diagnostics_summary())
    runtime = getattr(sil_vorlap, "runtime", None)
    summary = getattr(runtime, "response_diagnostics_summary", {}) or {}
    if not isinstance(summary, dict):
        summary = dict(summary)
    return dict(summary)


def _sil_vorlap_geometry_audit_file(sil_vorlap: SilVorLapCoupler | None) -> str | None:
    if sil_vorlap is None:
        return None
    config = getattr(sil_vorlap, "config", None)
    parameter_file = getattr(sil_vorlap, "parameter_file", None)
    if not isinstance(config, dict) or parameter_file is None:
        return None
    raw_path = config.get("geometry_audit_file")
    if not raw_path:
        return None
    return str(resolve_qblade_path(Path(parameter_file), str(raw_path)))


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _float_triplet(value: object) -> tuple[float, float, float] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        raw = list(value)  # type: ignore[arg-type]
    except TypeError:
        return None
    if len(raw) != 3:
        return None
    values = tuple(_finite_float(item) for item in raw)
    if any(item is None for item in values):
        return None
    return values  # type: ignore[return-value]


def _sil_vorlap_force_scale(sil_vorlap: SilVorLapCoupler | None) -> float | None:
    if sil_vorlap is None:
        return None
    runtime = getattr(sil_vorlap, "runtime", None)
    controller = getattr(runtime, "controller", None)
    return _finite_float(getattr(controller, "force_scale", None))


def _sil_vorlap_max_relative_speed(sil_vorlap: SilVorLapCoupler | None) -> float | None:
    if sil_vorlap is None:
        return None
    runtime = getattr(sil_vorlap, "runtime", None)
    ambient = _float_triplet(getattr(runtime, "_ambient_velocity_global", None))
    structural = getattr(runtime, "_last_controller_structural_velocity", None)
    if ambient is None or structural is None:
        return None
    if hasattr(structural, "tolist"):
        structural = structural.tolist()
    max_speed = 0.0
    rows = 0
    for row in structural:
        velocity = _float_triplet(row)
        if velocity is None:
            continue
        relative = tuple(ambient[i] - velocity[i] for i in range(3))
        max_speed = max(max_speed, math.sqrt(sum(component * component for component in relative)))
        rows += 1
    return max_speed if rows else None


def _sil_vorlap_timing_progress_fields(sil_vorlap: SilVorLapCoupler | None) -> list[str]:
    if sil_vorlap is None:
        return []
    timing = sil_vorlap.timing_diagnostics()
    last_sample = timing.get("last_sample") if isinstance(timing, dict) else None
    if not isinstance(last_sample, dict):
        return []

    fields: list[str] = []
    wrapper_time = _finite_float(last_sample.get("wrapper_time_s"))
    timestep = _finite_float(last_sample.get("wrapper_timestep"))
    azimuth = _finite_float(last_sample.get("azimuth_deg"))
    qblade_delta = _finite_float(last_sample.get("time_delta_qblade_minus_wrapper_s"))
    if wrapper_time is not None:
        fields.append(f"t={wrapper_time:.6g} s")
    if timestep is not None:
        fields.append(f"qblade step={timestep:.6g}")
    if azimuth is not None:
        fields.append(f"az={azimuth:.6g} deg")
    if qblade_delta is not None:
        fields.append(f"qblade dt={qblade_delta:.6g} s")
    return fields


def _sil_vorlap_progress_diagnostics(sil_vorlap: SilVorLapCoupler | None) -> str | None:
    if sil_vorlap is None:
        return None

    velocity = sil_vorlap.velocity_diagnostics()
    axis_stats = velocity.get("axis_stats", {}) if isinstance(velocity, dict) else {}
    max_abs_velocity = 0.0
    if isinstance(axis_stats, dict):
        for stat in axis_stats.values():
            if isinstance(stat, dict):
                max_abs_velocity = max(max_abs_velocity, float(stat.get("max_abs", 0.0) or 0.0))

    source_rows = sil_vorlap.source_coverage()
    source_samples = 0
    source_clips = 0
    for row in source_rows:
        if isinstance(row, dict):
            source_samples += int(row.get("samples", 0) or 0)
            source_clips += int(row.get("source_clip_count", 0) or 0)

    cache_stats = _sil_vorlap_cache_stats(sil_vorlap) or {"hits": 0, "misses": 0}
    cache_hits = cache_stats["hits"]
    cache_misses = cache_stats["misses"]
    channel_warnings = len(velocity.get("all_zero_channels", ())) if isinstance(velocity, dict) else 0
    axis_warnings = len(velocity.get("all_zero_axes", ())) if isinstance(velocity, dict) else 0

    fields = [
        f"max|F|={float(sil_vorlap.max_abs_force):.6g} N",
        f"max|Vstruct|={max_abs_velocity:.6g} m/s",
    ]
    max_relative_speed = _sil_vorlap_max_relative_speed(sil_vorlap)
    if max_relative_speed is not None:
        fields.append(f"max|Vrel|={max_relative_speed:.6g} m/s")
    force_scale = _sil_vorlap_force_scale(sil_vorlap)
    if force_scale is not None:
        fields.append(f"force scale={force_scale:.6g}")
    fields.extend(_sil_vorlap_timing_progress_fields(sil_vorlap))
    fields.extend(
        [
            f"cache={cache_hits}/{cache_misses}",
            f"source rows={len(source_rows)}",
            f"source samples={source_samples}",
            f"source clips={source_clips}",
            f"velocity warnings={channel_warnings} channel/{axis_warnings} axis",
        ]
    )
    return "VorLap diagnostics: " + ", ".join(fields)


def _progress_message(completed_steps: int, steps: int, sil_vorlap: SilVorLapCoupler | None) -> str:
    message = f"Completed {completed_steps}/{steps} timesteps"
    diagnostics = _sil_vorlap_progress_diagnostics(sil_vorlap)
    return message if diagnostics is None else f"{message}; {diagnostics}"


def _read_sim_declared_steps(sim_path: Path, *, strict: bool = False) -> int:
    if not strict:
        raw_steps = read_qblade_keyword(sim_path, "NUMTIMESTEPS")
        if raw_steps is None:
            raise ValueError(f"Could not find NUMTIMESTEPS in {sim_path}")
        steps = int(float(raw_steps))
        if steps < 1:
            raise ValueError("QBlade NUMTIMESTEPS must be positive.")
        return steps
    declaration = re.compile(
        r"^[ \t]*(?P<value>\S+)[ \t]+NUMTIMESTEPS(?:[ \t]+-.*)?[ \t]*$"
    )
    keyword = re.compile(r"(?<!\S)NUMTIMESTEPS(?!\S)")
    matches: list[str] = []
    malformed: list[str] = []
    for line in sim_path.read_text(encoding="utf-8", errors="strict").splitlines():
        if keyword.search(line) is None:
            continue
        match = declaration.fullmatch(line)
        if match is None:
            malformed.append(line)
        else:
            matches.append(match.group("value"))
    if malformed or len(matches) != 1:
        raise ValueError(
            "QBlade NUMTIMESTEPS must have exactly one unambiguous declaration; "
            f"values={matches!r}, malformed={malformed!r}."
        )
    raw_steps = matches[0]
    try:
        numeric_steps = Decimal(raw_steps)
    except InvalidOperation as exc:
        raise ValueError("QBlade NUMTIMESTEPS must be a finite integral token.") from exc
    if not numeric_steps.is_finite() or numeric_steps != numeric_steps.to_integral_value():
        raise ValueError("QBlade NUMTIMESTEPS must be a finite integral token.")
    steps = int(numeric_steps)
    if steps < 1:
        raise ValueError("QBlade NUMTIMESTEPS must be positive.")
    return steps


def _resolve_requested_steps(
    sim_path: Path,
    requested_steps: int | None,
    *,
    sim_declared_steps: int | None = None,
) -> int:
    if requested_steps is not None:
        steps = int(requested_steps)
    else:
        steps = _read_sim_declared_steps(sim_path) if sim_declared_steps is None else int(sim_declared_steps)
    if steps < 1:
        raise ValueError("--steps must be positive.")
    return steps


def _resolve_coupling_mode(
    *,
    explicit: QBladeCouplingMode | None,
    sil_vorlap: bool,
    modal_burst_plan: str | Path | None,
    constant_force_actions: Sequence[dict[str, object]],
) -> QBladeCouplingMode:
    if modal_burst_plan is not None and not str(modal_burst_plan).strip():
        raise ValueError("--sil-modal-burst-plan was supplied but is empty or whitespace.")
    modal_burst = modal_burst_plan is not None
    enabled_paths = int(bool(sil_vorlap)) + int(bool(constant_force_actions)) + int(modal_burst)
    if enabled_paths > 1:
        raise ValueError(
            "--sil-vorlap, --sil-constant-force, and --sil-modal-burst-plan are "
            "separate coupling paths; run them as separate cases."
        )
    if sil_vorlap and explicit not in (None, "sil-vorlap"):
        raise ValueError("--sil-vorlap requires --coupling-mode sil-vorlap when an explicit coupling label is provided.")
    if constant_force_actions and explicit not in (None, "sil-constant-force"):
            raise ValueError("--sil-constant-force requires --coupling-mode sil-constant-force when an explicit coupling label is provided.")
    if modal_burst and explicit not in (None, "sil-modal-burst"):
        raise ValueError(
            "--sil-modal-burst-plan requires --coupling-mode sil-modal-burst when an "
            "explicit coupling label is provided."
        )
    if explicit == "sil-vorlap" and not sil_vorlap:
        raise ValueError("--coupling-mode sil-vorlap requires --sil-vorlap.")
    if explicit == "sil-constant-force" and not constant_force_actions:
        raise ValueError("--coupling-mode sil-constant-force requires at least one --sil-constant-force action.")
    if explicit == "sil-modal-burst" and not modal_burst:
        raise ValueError(
            "--coupling-mode sil-modal-burst requires --sil-modal-burst-plan."
        )
    if explicit is not None:
        return explicit
    if sil_vorlap:
        return "sil-vorlap"
    if constant_force_actions:
        return "sil-constant-force"
    if modal_burst:
        return "sil-modal-burst"
    return "qblade-external"


def _required_numeric_keyword(
    sim_path: Path,
    keyword: str,
    expected: float,
    *,
    tolerance: float = 1.0e-12,
) -> float:
    raw = read_qblade_keyword(sim_path, keyword)
    if raw is None:
        raise ValueError(f"Modal-burst case is missing required {keyword} in {sim_path}.")
    try:
        observed = float(raw)
    except ValueError as exc:
        raise ValueError(f"Modal-burst case has nonnumeric {keyword}={raw!r}.") from exc
    if not math.isfinite(observed) or not math.isclose(
        observed,
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"Modal-burst case requires {keyword}={expected:.17g}; observed {raw!r}."
        )
    return observed


def _required_false_keyword(sim_path: Path, keyword: str) -> str:
    raw = read_qblade_keyword(sim_path, keyword)
    if raw is None or str(raw).strip().lower() not in {"0", "false"}:
        raise ValueError(
            f"Modal-burst case requires {keyword}=false/0; observed {raw!r}."
        )
    return str(raw)


def _required_blank_keyword(sim_path: Path, keyword: str) -> None:
    matches: list[str] = []
    pattern = re.compile(
        rf"^(?P<value>.*?)[ \t]+{re.escape(keyword)}(?:[ \t]+-.*)?$"
    )
    for line in sim_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if match is not None:
            matches.append(match.group("value").strip())
    if matches != [""]:
        raise ValueError(
            f"Modal-burst case requires exactly one blank {keyword}; observed {matches!r}."
        )


def _validate_modal_burst_case_contract(
    *,
    sim_path: Path,
    modal_burst: SilModalBurstCoupler,
    qblade_root: Path,
    requested_steps: int,
    sim_declared_steps: int,
) -> dict[str, object]:
    """Fail closed unless the .sim and legacy bridge are inert and plan-identical."""

    plan = modal_burst.plan
    if requested_steps != plan.total_wrapper_updates:
        raise ValueError(
            "Modal-burst requested steps must equal plan.total_wrapper_updates: "
            f"requested={requested_steps}, plan={plan.total_wrapper_updates}."
        )
    if sim_declared_steps != plan.total_wrapper_updates:
        raise ValueError(
            "Modal-burst NUMTIMESTEPS must equal plan.total_wrapper_updates: "
            f"sim={sim_declared_steps}, plan={plan.total_wrapper_updates}."
        )

    observed_numeric = {
        "TIMESTEP": _required_numeric_keyword(
            sim_path,
            "TIMESTEP",
            plan.timestep_s,
            tolerance=max(1.0e-14, abs(plan.timestep_s) * 1.0e-12),
        ),
        "PRESCRIBETYPE": _required_numeric_keyword(sim_path, "PRESCRIBETYPE", 1.0),
        "RPMPRESCRIBED": _required_numeric_keyword(
            sim_path,
            "RPMPRESCRIBED",
            plan.rotor_speed_rpm,
            tolerance=1.0e-10,
        ),
        "INITIAL_AZIMUTH": _required_numeric_keyword(
            sim_path,
            "INITIAL_AZIMUTH",
            plan.azimuth_reference_deg,
            tolerance=1.0e-10,
        ),
        "INCLUDEAERO": _required_numeric_keyword(sim_path, "INCLUDEAERO", 0.0),
        "INCLUDEHYDRO": _required_numeric_keyword(sim_path, "INCLUDEHYDRO", 0.0),
        "WNDTYPE": _required_numeric_keyword(sim_path, "WNDTYPE", 0.0),
        "MEANINF": _required_numeric_keyword(sim_path, "MEANINF", 0.0),
        "RAMPUP": _required_numeric_keyword(sim_path, "RAMPUP", 0.0),
        "ADDDAMP": _required_numeric_keyword(sim_path, "ADDDAMP", 0.0),
        "RELAXSTEPS": _required_numeric_keyword(sim_path, "RELAXSTEPS", 0.0),
        "STOREFROM": _required_numeric_keyword(sim_path, "STOREFROM", 0.0),
        "ISOFFSHORE": _required_numeric_keyword(sim_path, "ISOFFSHORE", 0.0),
        "WAKEINTERACTION": _required_numeric_keyword(
            sim_path, "WAKEINTERACTION", 0.0
        ),
        "SURF_CURR_U": _required_numeric_keyword(sim_path, "SURF_CURR_U", 0.0),
        "SUB_CURR_U": _required_numeric_keyword(sim_path, "SUB_CURR_U", 0.0),
        "SHORE_CURR_U": _required_numeric_keyword(sim_path, "SHORE_CURR_U", 0.0),
    }
    store_replay = _required_false_keyword(sim_path, "STOREREPLAY")
    blank_inputs = (
        "EVENTFILE",
        "LOADINGFILE",
        "SIMFILE",
        "MOTIONFILE",
        "WAVEFILE",
        "MOORINGSYSTEM",
    )
    for keyword in blank_inputs:
        _required_blank_keyword(sim_path, keyword)

    parameter_file, config = load_strict_modal_external_config(sim_path)
    force_scale = _finite_float(config.get("force_scale"))
    if force_scale is None or force_scale != 0.0:
        raise ValueError(
            "Modal-burst case requires legacy VorLap force_scale exactly 0.0; "
            f"observed {config.get('force_scale')!r}."
        )
    if config.get("include_mean_load", False) is not False:
        raise ValueError("Modal-burst case requires legacy include_mean_load=false.")
    response_model = config.get("response_model")
    response_inactive = response_model is None or (
        isinstance(response_model, dict) and response_model.get("enabled") is False
    )
    if not response_inactive:
        raise ValueError(
            "Modal-burst case requires the legacy response_model to be absent or explicitly "
            "disabled."
        )
    external_artifacts = [
        identity
        for identity in modal_burst.artifact_identities
        if identity.get("role") == "external_config"
    ]
    if len(external_artifacts) != 1:
        raise ValueError("Modal-burst plan must bind exactly one external_config artifact.")
    external_identity = external_artifacts[0]
    parameter_identity = file_sha256(parameter_file)
    if parameter_identity is None:
        raise FileNotFoundError(parameter_file)
    if (
        Path(str(external_identity["path"])).resolve() != parameter_file.resolve()
        or external_identity.get("sha256") != parameter_identity.get("sha256")
        or external_identity.get("size_bytes") != parameter_identity.get("size_bytes")
    ):
        raise ValueError(
            "Modal-burst plan external_config must be the exact staged "
            "LIBPARAMETERFILE_1 bytes used by this case."
        )
    try:
        parameter_file.resolve().relative_to(sim_path.parent.resolve())
    except ValueError as exc:
        raise ValueError(
            "Modal-burst LIBPARAMETERFILE_1 must reside inside the staged case tree."
        ) from exc
    raw_env_log_dir = os.environ.get("VORLAP_QBLADE_BRIDGE_LOG_DIR")
    if raw_env_log_dir is not None and raw_env_log_dir != "":
        raise ValueError(
            "Modal-burst runs prohibit any raw nonempty "
            "VORLAP_QBLADE_BRIDGE_LOG_DIR value."
        )
    source_parameter_dir = config.get("source_parameter_dir")
    if not isinstance(source_parameter_dir, str) or not source_parameter_dir.strip():
        raise ValueError("Modal-burst legacy source_parameter_dir must be a staged path.")
    source_dir_candidate = Path(source_parameter_dir).expanduser()
    if not source_dir_candidate.is_absolute():
        raise ValueError(
            "Modal-burst source_parameter_dir must be an absolute path because the C++ "
            "bridge uses it directly before falling back to the parameter-file parent."
        )
    source_dir = source_dir_candidate.resolve()
    if source_dir != parameter_file.parent.resolve():
        raise ValueError(
            "Modal-burst source_parameter_dir must resolve exactly to the staged config "
            "directory."
        )
    if not source_dir.is_dir() or not os.access(source_dir, os.W_OK):
        raise ValueError("Modal-burst staged callback-log directory must be writable.")

    bridge_runtime_audit = require_modal_native_bridge_runtime_contract(
        sim_path,
        qblade_root,
    )
    qblade_dependencies, qblade_dependency_fingerprint = (
        qblade_runtime_dependency_manifest(qblade_root)
    )
    callback_evidence = native_callback_evidence(sim_path, qblade_root=qblade_root)
    callback_contract = {
        "library_stem": "libvorlap_qblade_bridge",
        "function_name": "update",
        "array_size": "237",
        "parameter_file_resolved": str(parameter_file.resolve()),
    }
    callback_mismatches = {
        key: {"expected": expected, "observed": callback_evidence.get(key)}
        for key, expected in callback_contract.items()
        if str(callback_evidence.get(key)) != str(expected)
    }
    if callback_mismatches:
        raise ValueError(
            "Modal-burst turbine native callback contract mismatch: "
            + json.dumps(callback_mismatches, sort_keys=True)
        )
    if callback_evidence.get("callback_block_present") is not True:
        raise ValueError("Modal-burst turbine must retain the complete audited callback block.")
    if callback_evidence.get("library_file_exists") is not True:
        raise ValueError(
            "Modal-burst QBlade root must contain the configured native bridge library so "
            "zero-callback evidence cannot be produced by a missing library."
        )
    if (
        callback_evidence.get("library_symbol_check_available") is not True
        or callback_evidence.get("library_missing_required_symbols") != []
        or callback_evidence.get("library_exported_required_symbols")
        != ["update", "update_init", "update_message"]
    ):
        raise ValueError(
            "Modal-burst native bridge must export the exact update, update_init, and "
            "update_message callback symbols."
        )
    require_zero_native_callback_evidence(
        callback_evidence,
        context="modal-burst preflight",
        reject_existing_log=True,
    )
    return {
        "schema": "vorlap-qblade-modal-burst-case-contract-v1",
        "passed": True,
        "requested_steps_equal_plan": True,
        "sim_steps_equal_plan": True,
        "numeric_settings": observed_numeric,
        "store_replay": store_replay,
        "blank_alternate_inputs": list(blank_inputs),
        "legacy_external_config": file_sha256(parameter_file),
        "legacy_force_scale": force_scale,
        "legacy_include_mean_load": False,
        "legacy_response_model_inactive": True,
        "plan_external_config_is_live_parameter_file": True,
        "source_parameter_dir": str(source_dir),
        "native_bridge_runtime_audit": bridge_runtime_audit,
        "callback_contract": callback_contract,
        "callback_library": callback_evidence.get("library_file_hash"),
        "qblade_runtime_dependencies": qblade_dependencies,
        "qblade_runtime_dependency_fingerprint": qblade_dependency_fingerprint,
        "native_callback_requirement": "exactly zero init and update callbacks",
    }


def _resolve_qblade_execution_mode(value: QBladeExecutionMode | str) -> QBladeExecutionMode:
    mode = str(value)
    if mode not in ("step-loop", "run-full-simulation"):
        raise ValueError(
            "qblade_execution_mode must be one of: 'step-loop' or 'run-full-simulation'."
        )
    return mode  # type: ignore[return-value]


def _require_modal_output_roots_outside_case(
    *,
    sim_path: Path,
    output_dir: Path,
    checkpoint_dir: Path,
) -> None:
    """Keep mutable run products outside the immutable staged input tree."""

    case_root = sim_path.resolve().parent
    for label, candidate in (
        ("output_dir", output_dir.resolve()),
        ("checkpoint_dir", checkpoint_dir.resolve()),
    ):
        try:
            candidate.relative_to(case_root)
        except ValueError:
            continue
        raise ValueError(
            f"Modal-burst {label} must reside outside the immutable staged case root "
            f"{case_root}; observed {candidate}."
        )


def _validate_qblade_execution_mode(
    *,
    execution_mode: QBladeExecutionMode,
    coupling_mode: QBladeCouplingMode,
    request: QBladeRunRequest,
    constant_force_actions: Sequence[dict[str, object]],
    requested_steps: int,
    sim_declared_steps: int,
) -> None:
    if execution_mode == "step-loop":
        return
    if coupling_mode != "qblade-external":
        raise ValueError("--native-run-full-simulation requires --coupling-mode qblade-external.")
    if request.sil_vorlap:
        raise ValueError("--native-run-full-simulation cannot be combined with --sil-vorlap.")
    if constant_force_actions:
        raise ValueError("--native-run-full-simulation cannot be combined with --sil-constant-force.")
    if int(requested_steps) != int(sim_declared_steps):
        raise ValueError(
            "--native-run-full-simulation uses NUMTIMESTEPS from the .sim file; "
            f"requested {requested_steps} steps but the .sim declares {sim_declared_steps}."
        )
    if request.checkpoint_every:
        raise ValueError("--native-run-full-simulation does not support wrapper-level checkpoints.")
    if request.resume_from_checkpoint or request.resume_checkpoint is not None:
        raise ValueError("--native-run-full-simulation does not support wrapper-level resume.")
    if request.allow_partial:
        raise ValueError("--native-run-full-simulation cannot export partial results.")


def run_qblade_case(request: QBladeRunRequest) -> QBladeRunResult:
    """Run one QBlade simulation definition and write the standard run summary."""
    started = time.time()
    full_run_started_ns = time.perf_counter_ns()
    aggregate_timing = (
        AggregatePerformanceTiming(started_ns=full_run_started_ns)
        if request.sil_vorlap and profiling_requested()
        else None
    )
    sim_path = Path(request.sim_path).resolve()
    qblade_root = Path(request.qblade_root).resolve()
    output_dir = Path(request.output_dir).resolve()
    checkpoint_args = type(
        "CheckpointArgs",
        (),
        {
            "checkpoint_dir": request.checkpoint_dir,
            "resume_checkpoint": request.resume_checkpoint,
            "resume_from_checkpoint": request.resume_from_checkpoint,
        },
    )()
    checkpoint_dir = checkpoint_directory(checkpoint_args, output_dir)
    if request.sil_modal_burst_plan is not None:
        if not str(request.sil_modal_burst_plan).strip():
            raise ValueError(
                "--sil-modal-burst-plan was supplied but is empty or whitespace."
            )
        _require_modal_output_roots_outside_case(
            sim_path=sim_path,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if request.manage_loader_env:
        ensure_qblade_loader_env(qblade_root)

    if not sim_path.is_file():
        raise FileNotFoundError(f"QBlade simulation file not found: {sim_path}")
    preflight_qblade_wind_input(sim_path)

    sim_declared_steps = _read_sim_declared_steps(
        sim_path,
        strict=request.sil_modal_burst_plan is not None,
    )
    steps = _resolve_requested_steps(sim_path, request.steps, sim_declared_steps=sim_declared_steps)
    constant_force_actions = [
        parse_constant_force_action(spec)
        for spec in request.sil_constant_force_specs
    ]
    coupling_mode = _resolve_coupling_mode(
        explicit=request.coupling_mode,
        sil_vorlap=request.sil_vorlap,
        modal_burst_plan=request.sil_modal_burst_plan,
        constant_force_actions=constant_force_actions,
    )
    modal_plan_path = (
        None
        if request.sil_modal_burst_plan is None
        else Path(request.sil_modal_burst_plan).expanduser().resolve()
    )
    sil_modal_burst = (
        None
        if coupling_mode != "sil-modal-burst" or modal_plan_path is None
        else SilModalBurstCoupler.from_plan_file(modal_plan_path)
    )
    modal_case_contract = (
        None
        if sil_modal_burst is None
        else _validate_modal_burst_case_contract(
            sim_path=sim_path,
            modal_burst=sil_modal_burst,
            qblade_root=qblade_root,
            requested_steps=steps,
            sim_declared_steps=sim_declared_steps,
        )
    )
    if sil_modal_burst is not None:
        if request.allow_partial:
            raise ValueError("Modal-burst runs prohibit --allow-partial.")
        if request.completed_checkpoint_export_only:
            raise ValueError(
                "Modal-burst runs prohibit --completed-checkpoint-export-only because the "
                "schedule must be finalized by the active coupler."
            )
    sil_vorlap_history_mode = resolve_sil_history_mode(
        record_history=bool(request.record_sil_vorlap_history),
        history_mode=request.sil_vorlap_history_mode,
    )
    if not request.sil_vorlap and sil_vorlap_history_mode != "off":
        raise ValueError("SIL VorLap history recording requires sil_vorlap=True.")
    qblade_execution_mode = _resolve_qblade_execution_mode(request.qblade_execution_mode)
    _validate_qblade_execution_mode(
        execution_mode=qblade_execution_mode,
        coupling_mode=coupling_mode,
        request=request,
        constant_force_actions=constant_force_actions,
        requested_steps=steps,
        sim_declared_steps=sim_declared_steps,
    )
    case_metadata = request.case_metadata
    if case_metadata is None and request.case_metadata_path is not None:
        case_metadata = load_json_file(Path(request.case_metadata_path).resolve())

    result_name = request.result_name or f"{sim_path.stem}_results"
    if result_name.endswith(".txt"):
        result_name = result_name[:-4]
    result_filter = _resolve_result_filter(request.result_filter)
    if request.completed_checkpoint_export_only:
        if result_filter is None:
            raise ValueError(
                "--completed-checkpoint-export-only requires --result-filter so recovery "
                "cannot repeat an unbounded all-channel export."
            )
        if request.resume_initialize:
            raise ValueError(
                "--completed-checkpoint-export-only cannot be combined with --resume-initialize."
            )
        if not (request.resume_from_checkpoint or request.resume_checkpoint is not None):
            raise ValueError(
                "--completed-checkpoint-export-only requires --resume-from-checkpoint or "
                "--resume-checkpoint."
            )
        if sil_vorlap_history_mode == "compact":
            raise ValueError(
                "--completed-checkpoint-export-only cannot reconstruct a streamed compact SIL history."
            )
    result_path = output_dir / f"{result_name}.txt"
    sil_vorlap_history_path = output_dir / f"{result_name}_sil_vorlap_history.csv"
    if result_path.exists():
        result_path.unlink()
    # Every generated case contains the native callback table. Clear any stale
    # evidence even for SIL/none runs so a zero callback count is meaningful.
    if not request.completed_checkpoint_export_only and sil_modal_burst is None:
        clear_native_bridge_log(sim_path)
    if int(request.checkpoint_every) < 0:
        raise ValueError("--checkpoint-every must be non-negative.")

    checkpoint_identity = build_checkpoint_identity(
        sim_path=sim_path,
        qblade_root=qblade_root,
        coupling_mode=coupling_mode,
        requested_steps=steps,
        threads=int(request.threads),
        use_opencl=bool(request.use_opencl),
        device=int(request.device),
        group_size=int(request.group_size),
        constant_force_actions=constant_force_actions,
        qblade_execution_mode=qblade_execution_mode,
        resume_initialize=bool(request.resume_initialize),
        allow_partial=bool(request.allow_partial),
        modal_burst_identity=(
            None
            if sil_modal_burst is None
            else sil_modal_burst.checkpoint_identity()
        ),
    )
    if sil_modal_burst is not None and modal_case_contract is not None:
        expected_dependency_fingerprint = modal_case_contract.get(
            "qblade_runtime_dependency_fingerprint"
        )
        if (
            checkpoint_identity.get("qblade_runtime_dependency_fingerprint")
            != expected_dependency_fingerprint
        ):
            raise RuntimeError(
                "Modal-burst checkpoint identity does not match the preflight QBlade "
                "runtime dependency fingerprint."
            )
    resume_checkpoint = load_resume_checkpoint(
        checkpoint_args,
        checkpoint_dir,
        expected_identity=None if request.completed_checkpoint_export_only else checkpoint_identity,
    )
    if request.completed_checkpoint_export_only:
        resume_checkpoint = _validate_completed_checkpoint_export_only(
            resume_checkpoint,
            requested_steps=steps,
            sim_path=sim_path,
            output_dir=output_dir,
            result_name=result_name,
            coupling_mode=coupling_mode,
        )
    if sil_modal_burst is not None and resume_checkpoint is not None:
        _validate_modal_checkpoint_envelope(
            resume_checkpoint,
            sim_path=sim_path,
            qblade_root=qblade_root,
            output_dir=output_dir,
            result_name=result_name,
            requested_steps=steps,
            threads=int(request.threads),
            use_opencl=bool(request.use_opencl),
            checkpoint_identity=checkpoint_identity,
            modal_plan_path=modal_plan_path,
        )
    resume_start_step = 0
    if resume_checkpoint is not None:
        resume_start_step = int(resume_checkpoint.get("completed_steps", 0) or 0)
        if resume_start_step < 0 or resume_start_step > steps:
            raise ValueError(
                f"Checkpoint completed_steps={resume_start_step} is outside requested range 0..{steps}."
            )
    if sil_vorlap_history_mode == "compact" and resume_checkpoint is None:
        sil_vorlap_history_path.unlink(missing_ok=True)

    qblade_loader = request.qblade_api_loader or load_qblade_api
    qblade, library_path = qblade_loader(qblade_root)
    run_log = output_dir / f"{sim_path.stem}_qblade.log"
    sil_vorlap = (
        SilVorLapCoupler(
            sim_path,
            record_history=request.record_sil_vorlap_history,
            history_mode=sil_vorlap_history_mode,
            history_path=(
                sil_vorlap_history_path if sil_vorlap_history_mode == "compact" else None
            ),
            resume_history=resume_checkpoint is not None,
        )
        if request.sil_vorlap
        else None
    )
    if aggregate_timing is not None and sil_vorlap is not None:
        sil_vorlap.attach_performance_timing(aggregate_timing)
    _restore_sil_vorlap_resume_state(
        sil_vorlap,
        resume_checkpoint,
        completed_checkpoint_export_only=bool(request.completed_checkpoint_export_only),
    )
    _restore_modal_burst_resume_state(sil_modal_burst, resume_checkpoint)
    if sil_modal_burst is not None and sil_modal_burst.wrapper_updates != resume_start_step:
        raise ValueError(
            "Modal-burst checkpoint state must satisfy wrapper_updates == completed_steps; "
            f"observed {sil_modal_burst.wrapper_updates} != {resume_start_step}."
        )
    modal_timing_witnesses = _restore_modal_timing_witnesses(
        sil_modal_burst,
        resume_checkpoint,
        plan_path=modal_plan_path,
    )
    sil_constant_force = SilConstantForceCoupler(constant_force_actions) if constant_force_actions else None
    if sil_constant_force is not None and resume_start_step:
        sil_constant_force.updates = resume_start_step
    if aggregate_timing is not None:
        aggregate_timing.record_since(
            "runner_preparation_and_coupler_construction",
            full_run_started_ns,
        )
    previous_cwd = Path.cwd()
    completed_steps = resume_start_step
    failure_message: str | None = None
    stop_message: str | None = None
    stopped = False
    exported_results = False
    latest_checkpoint = resume_checkpoint
    stored_project: Path | None = None
    modal_loaded_resume_position_audit: dict[str, object] | None = None
    initialization_started_ns = time.perf_counter_ns()

    try:
        qblade.setLogFile(str(run_log).encode("utf-8"))
        qblade.createInstance(int(request.device), int(request.group_size))
        qblade.setOmpNumThreads(int(request.threads))
        qblade.setUseOpenCl(bool(request.use_opencl))

        if resume_checkpoint is None:
            os.chdir(sim_path.parent)
            qblade.loadSimDefinition(sim_path.name.encode("utf-8"))
            qblade.initializeSimulation()
        else:
            project_path = Path(str(resume_checkpoint["qblade_project"]))
            os.chdir(project_path.parent)
            _emit(
                request.message_callback,
                f"Resuming QBlade project checkpoint at step {resume_start_step}/{steps}: {project_path}",
            )
            qblade.loadProject(str(project_path).encode("utf-8"))
            if request.resume_initialize:
                qblade.initializeSimulation()
        if sil_vorlap is not None and not request.completed_checkpoint_export_only:
            sil_vorlap.validate_channels(qblade)
        if sil_modal_burst is not None:
            sil_modal_burst.validate_channels(qblade)
            if resume_checkpoint is not None:
                modal_loaded_resume_position_audit = (
                    validate_loaded_modal_burst_resume_position(
                        qblade,
                        sil_modal_burst,
                        resume_start_step,
                        _ordered_modal_timing_witnesses(modal_timing_witnesses),
                        resume_checkpoint["sil_modal_burst_resume_position"],
                    )
                )
        if aggregate_timing is not None:
            aggregate_timing.record_ns(
                "initialization",
                time.perf_counter_ns() - initialization_started_ns,
            )

        if qblade_execution_mode == "run-full-simulation":
            if request.stop_requested is not None and request.stop_requested():
                stopped = True
                stop_message = f"QBlade simulation stopped by user at {completed_steps}/{steps} completed timesteps."
                _emit(request.message_callback, stop_message)
            else:
                run_full = getattr(qblade, "runFullSimulation", None)
                if run_full is None:
                    raise RuntimeError("The loaded QBlade SIL wrapper does not expose runFullSimulation().")
                _emit(request.message_callback, f"Starting QBlade runFullSimulation() for {steps} declared timesteps.")
                if not run_full():
                    failure_message = "QBlade runFullSimulation() failed."
                    raise RuntimeError(failure_message)
                completed_steps = steps
                _emit(request.message_callback, f"Completed {completed_steps}/{steps} timesteps via runFullSimulation().")
        else:
            for i in range(completed_steps, steps):
                if request.stop_requested is not None and request.stop_requested():
                    stopped = True
                    stop_message = f"QBlade simulation stopped by user at {completed_steps}/{steps} completed timesteps."
                    _emit(request.message_callback, stop_message)
                    break
                if sil_vorlap is not None:
                    sil_vorlap.step(qblade, i)
                if sil_modal_burst is not None:
                    sil_modal_burst.step(qblade, i)
                    if sil_modal_burst.last_record is None:  # pragma: no cover - core invariant.
                        raise RuntimeError("Modal-burst step completed without an application record.")
                    _retain_modal_timing_witness(
                        modal_timing_witnesses,
                        sil_modal_burst.last_record,
                    )
                if sil_constant_force is not None:
                    sil_constant_force.step(qblade, i)
                advance_started_ns = time.perf_counter_ns()
                advanced = qblade.advanceTurbineSimulation()
                if aggregate_timing is not None:
                    aggregate_timing.record_ns(
                        "qblade_timestep_advancement",
                        time.perf_counter_ns() - advance_started_ns,
                    )
                if not advanced:
                    failure_message = f"QBlade simulation failed at timestep {i}."
                    if not request.allow_partial:
                        raise RuntimeError(failure_message)
                    if sil_vorlap is not None or sil_modal_burst is not None:
                        coupling_label = (
                            "SIL VorLap runtime and force history"
                            if sil_vorlap is not None
                            else "modal-burst schedule and force history"
                        )
                        raise RuntimeError(
                            f"{failure_message} The {coupling_label} were "
                            "already advanced for this timestep, so a partial export or restart "
                            "checkpoint would not be causally consistent with the QBlade project."
                        )
                    if completed_steps == 0:
                        raise RuntimeError(f"{failure_message} No completed timestep is available to export.")
                    _emit(request.message_callback, f"{failure_message} Exporting {completed_steps} completed timesteps.")
                    break
                completed_steps = i + 1
                if request.progress_every and completed_steps % int(request.progress_every) == 0:
                    _emit(request.message_callback, _progress_message(completed_steps, steps, sil_vorlap))
                if request.checkpoint_every and completed_steps % int(request.checkpoint_every) == 0:
                    latest_checkpoint = write_restart_checkpoint(
                        qblade=qblade,
                        checkpoint_dir=checkpoint_dir,
                        sim_path=sim_path,
                        qblade_root=qblade_root,
                        output_dir=output_dir,
                        result_name=result_name,
                        coupling_mode=coupling_mode,
                        requested_steps=steps,
                        completed_steps=completed_steps,
                        threads=int(request.threads),
                        use_opencl=bool(request.use_opencl),
                        checkpoint_identity=checkpoint_identity,
                        resume_checkpoint=latest_checkpoint,
                        sil_vorlap=sil_vorlap,
                        sil_modal_burst=sil_modal_burst,
                        sil_modal_burst_plan_path=modal_plan_path,
                        sil_modal_burst_timing_witnesses=(
                            _ordered_modal_timing_witnesses(modal_timing_witnesses)
                        ),
                        reason="periodic",
                    )
                    _emit(request.message_callback, f"Wrote restart checkpoint at {completed_steps}/{steps} timesteps.")

        if request.checkpoint_every and completed_steps > 0:
            last_checkpoint_step = 0 if latest_checkpoint is None else int(latest_checkpoint.get("completed_steps", 0) or 0)
            if last_checkpoint_step < completed_steps:
                latest_checkpoint = write_restart_checkpoint(
                    qblade=qblade,
                    checkpoint_dir=checkpoint_dir,
                    sim_path=sim_path,
                    qblade_root=qblade_root,
                    output_dir=output_dir,
                    result_name=result_name,
                    coupling_mode=coupling_mode,
                    requested_steps=steps,
                    completed_steps=completed_steps,
                    threads=int(request.threads),
                    use_opencl=bool(request.use_opencl),
                    checkpoint_identity=checkpoint_identity,
                    resume_checkpoint=latest_checkpoint,
                    sil_vorlap=sil_vorlap,
                    sil_modal_burst=sil_modal_burst,
                    sil_modal_burst_plan_path=modal_plan_path,
                    sil_modal_burst_timing_witnesses=(
                        _ordered_modal_timing_witnesses(modal_timing_witnesses)
                    ),
                    reason="stopped" if stopped else ("final" if failure_message is None else "partial_failure"),
                )
                _emit(request.message_callback, f"Wrote restart checkpoint at {completed_steps}/{steps} timesteps.")

        modal_final_summary = None
        if sil_modal_burst is not None and failure_message is None and not stopped:
            modal_final_summary = sil_modal_burst.finalize()

        if request.store_project and failure_message is None and not stopped:
            project_path = output_dir / f"{sim_path.stem}_completed.qpr"
            qblade.storeProject(str(project_path).encode("utf-8"))
            if not project_path.is_file() or project_path.stat().st_size <= 0:
                project_path.unlink(missing_ok=True)
                raise RuntimeError(f"QBlade did not write a nonempty completed project: {project_path}")
            stored_project = project_path

        if completed_steps > 0:
            filter_bytes = b"" if result_filter is None else str(result_filter).encode("utf-8")
            if request.completed_checkpoint_export_only:
                _emit(
                    request.message_callback,
                    "Exporting completed QBlade checkpoint without initialization or timestep advancement "
                    f"using result filter {result_filter}.",
                )
            export_started_ns = time.perf_counter_ns()
            qblade.exportResults(
                0,
                str(output_dir).encode("utf-8"),
                result_name.encode("utf-8"),
                filter_bytes,
            )
            if aggregate_timing is not None:
                aggregate_timing.record_ns(
                    "terminal_result_export",
                    time.perf_counter_ns() - export_started_ns,
                )
            if not result_path.is_file() or result_path.stat().st_size <= 0:
                result_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "QBlade did not write a nonempty result export"
                    + (" using filter " + str(result_filter) if result_filter is not None else "")
                    + f": {result_path}"
                )
            exported_results = True
        elif stopped:
            _emit(request.message_callback, "Stop requested before any timestep completed; no QBlade results were exported.")
    finally:
        os.chdir(previous_cwd)
        try:
            qblade.unload()
        finally:
            if sil_vorlap is not None:
                close_history = getattr(sil_vorlap, "close_history", None)
                if callable(close_history):
                    close_history(durable=True)

    sil_vorlap_history = None
    if sil_vorlap is not None:
        sil_vorlap_history = sil_vorlap.write_history(
            sil_vorlap_history_path
        )
        if sil_vorlap_history is not None:
            history_path = Path(str(sil_vorlap_history["path"]))
            sil_vorlap_history = {
                **sil_vorlap_history,
                "file_identity": file_sha256(history_path),
            }

    command_line = list(request.command_line) if request.command_line else [sys.executable, *sys.argv]
    raw_expected_rpm = read_qblade_keyword(sim_path, "RPMPRESCRIBED")
    raw_store_from = read_qblade_keyword(sim_path, "STOREFROM")
    expected_rpm = 0.0 if raw_expected_rpm is None else float(raw_expected_rpm)
    store_from_s = 0.0 if raw_store_from is None else float(raw_store_from)
    rpm_audit_start_s = max(store_from_s, 0.1)
    native_evidence = native_callback_evidence(sim_path, qblade_root=qblade_root)
    unexpected_native_message: str | None = None
    discarded_contaminated_checkpoint_files: list[str] = []
    discarded_contaminated_stored_project: str | None = None
    modal_final_runtime_contract: dict[str, object] | None = None
    modal_final_dependency_audit: dict[str, object] | None = None
    modal_final_runtime_identity_audit: dict[str, object] | None = None
    if sil_modal_burst is not None:
        try:
            if modal_plan_path is None:  # pragma: no cover - construction invariant.
                raise RuntimeError("Modal-burst final audit lacks its immutable plan path.")
            modal_final_runtime_contract = (
                require_modal_native_bridge_runtime_contract(sim_path, qblade_root)
            )
            modal_final_dependency_audit = (
                require_modal_qblade_runtime_dependency_contract(
                    qblade_root,
                    checkpoint_identity.get(
                        "qblade_runtime_dependency_fingerprint"
                    ),
                )
            )
            require_zero_native_callback_evidence(
                native_evidence,
                context="modal-burst final audit",
                reject_existing_log=True,
            )
            modal_final_runtime_identity_audit = (
                require_modal_runtime_identity_contract(
                    sim_path=sim_path,
                    qblade_root=qblade_root,
                    plan_path=modal_plan_path,
                    expected_checkpoint_identity=checkpoint_identity,
                )
            )
        except Exception as exc:
            unexpected_native_message = str(exc)
            failure_message = unexpected_native_message
            if stored_project is not None and stored_project.exists():
                discarded_contaminated_stored_project = str(stored_project)
                stored_project.unlink()
                stored_project = None
            discarded_contaminated_checkpoint_files = (
                _discard_contaminated_modal_checkpoint(latest_checkpoint, checkpoint_dir)
            )
            latest_checkpoint = None
    elif coupling_mode != "qblade-external":
        native_init_count = int(native_evidence.get("update_init_count", 0) or 0)
        native_update_count = max(
            int(native_evidence.get("update_count", 0) or 0),
            int(native_evidence.get("update_count_observed_max", 0) or 0),
        )
        if native_init_count or native_update_count:
            unexpected_native_message = (
                f"Unexpected native callback activity in coupling mode {coupling_mode!r}: "
                f"update_init={native_init_count}, update={native_update_count}. "
                "Refusing a run that may have applied external forces twice."
            )
            failure_message = unexpected_native_message
    summary_payload = QBladeRunSummary(
        sim=str(sim_path),
        qblade_root=str(qblade_root),
        qblade_library=str(library_path),
        coupling_mode=coupling_mode,
        qblade_execution_mode=qblade_execution_mode,
        requested_steps=steps,
        completed_steps=completed_steps,
        stopped=stopped,
        stop_message=stop_message,
        failed=failure_message is not None,
        failure_message=failure_message,
        allow_partial=bool(request.allow_partial),
        threads=int(request.threads),
        use_opencl=bool(request.use_opencl),
        restart={
            "checkpoint_every": int(request.checkpoint_every),
            "checkpoint_dir": str(checkpoint_dir),
            "resume_requested": bool(request.resume_from_checkpoint or request.resume_checkpoint),
            "resume_checkpoint": None if resume_checkpoint is None else resume_checkpoint.get("_metadata_path"),
            "resume_rejection": getattr(checkpoint_args, "_checkpoint_rejection", None),
            "resume_start_step": int(resume_start_step),
            "resume_initialize": bool(request.resume_initialize),
            "latest_checkpoint": None if latest_checkpoint is None else latest_checkpoint.get("_metadata_path"),
            "latest_checkpoint_completed_steps": None
            if latest_checkpoint is None
            else int(latest_checkpoint.get("completed_steps", 0) or 0),
            "latest_checkpoint_project": None if latest_checkpoint is None else latest_checkpoint.get("qblade_project"),
            "qblade_resume_mode": "loadProject",
            "checkpoint_identity_fingerprint": (
                checkpoint_identity.get("fingerprint")
                if not request.completed_checkpoint_export_only or resume_checkpoint is None
                else resume_checkpoint.get("checkpoint_identity_fingerprint")
            ),
            "checkpoint_current_identity_fingerprint": checkpoint_identity.get("fingerprint"),
            "checkpoint_observed_identity_fingerprint": None
            if resume_checkpoint is None
            else resume_checkpoint.get("checkpoint_identity_fingerprint"),
            "completed_checkpoint_export_only": bool(request.completed_checkpoint_export_only),
            "identity_policy": (
                "validated_current_identity"
                if not request.completed_checkpoint_export_only
                else "verified_qpr_hash_size_complete_steps_and_case_binding_no_timestep_advance"
            ),
        },
        stored_project=None if stored_project is None else str(stored_project),
        result_filter=None if result_filter is None else str(result_filter),
        result_filter_identity=None if result_filter is None else file_sha256(result_filter),
        result_file=str(result_path),
        result_file_identity=file_sha256(result_path) if exported_results else None,
        exported_results=exported_results,
        run_log=str(run_log),
        sil_vorlap=bool(sil_vorlap is not None),
        sil_vorlap_updates=0 if sil_vorlap is None else sil_vorlap.updates,
        sil_vorlap_parameter_file=None if sil_vorlap is None else str(sil_vorlap.parameter_file),
        sil_vorlap_geometry_audit_file=_sil_vorlap_geometry_audit_file(sil_vorlap),
        sil_vorlap_max_abs_force=None if sil_vorlap is None else sil_vorlap.max_abs_force,
        sil_vorlap_message=None if sil_vorlap is None else sil_vorlap.last_message,
        sil_vorlap_velocity_diagnostics=None if sil_vorlap is None else sil_vorlap.velocity_diagnostics(),
        sil_vorlap_timing_diagnostics=None if sil_vorlap is None else sil_vorlap.timing_diagnostics(),
        sil_vorlap_cache_stats=_sil_vorlap_cache_stats(sil_vorlap),
        sil_vorlap_response_diagnostics_summary=(
            _sil_vorlap_response_diagnostics_summary(sil_vorlap)
        ),
        sil_vorlap_source_bounds_policy=None
        if sil_vorlap is None
        else getattr(sil_vorlap.runtime, "source_bounds_policy", None),
        sil_vorlap_source_frequency_cap_hz=None
        if sil_vorlap is None
        else getattr(sil_vorlap.runtime, "source_frequency_cap_hz", None),
        sil_vorlap_source_coverage=[] if sil_vorlap is None else sil_vorlap.source_coverage(),
        sil_vorlap_history_mode=sil_vorlap_history_mode,
        sil_vorlap_history=sil_vorlap_history,
        sil_constant_force=bool(sil_constant_force is not None),
        sil_constant_force_updates=0 if sil_constant_force is None else sil_constant_force.updates,
        sil_constant_force_actions=constant_force_actions,
        sil_constant_force_max_abs_force=None if sil_constant_force is None else sil_constant_force.max_abs_force,
        force_sign_probe_observations=[]
        if sil_constant_force is None
        else force_sign_probe_observations(result_path, constant_force_actions),
        qblade_export_time_diagnostics=qblade_result_time_diagnostics(result_path),
        qblade_export_rpm_diagnostics=qblade_result_rpm_diagnostics(
            result_path,
            expected_rpm=expected_rpm,
            start_time_s=rpm_audit_start_s,
        ),
        native_callback_evidence=native_evidence,
        reproducibility=build_reproducibility_metadata(
            command_line=command_line,
            sim_path=sim_path,
            output_dir=output_dir,
            qblade_root=qblade_root,
            qblade_library=library_path,
            coupling_mode=coupling_mode,
            requested_steps=steps,
            completed_steps=completed_steps,
            threads=int(request.threads),
            use_opencl=bool(request.use_opencl),
            qblade_execution_mode=qblade_execution_mode,
            case_metadata=case_metadata,
            result_filter=result_filter,
            modal_burst=sil_modal_burst is not None,
        ),
        elapsed_seconds=time.time() - started,
    )
    summary = summary_payload.as_dict()
    if aggregate_timing is not None:
        summary["performance_timing"] = aggregate_timing.as_dict()
    if sil_modal_burst is not None:
        summary["restart"]["discarded_contaminated_checkpoint_files"] = (
            discarded_contaminated_checkpoint_files
        )
        summary["restart"]["discarded_contaminated_stored_project"] = (
            discarded_contaminated_stored_project
        )
        modal_summary = QBladeModalBurstRunSummary(
            enabled=True,
            plan_file=str(Path(request.sil_modal_burst_plan).expanduser().resolve()),
            identity=sil_modal_burst.checkpoint_identity(),
            case_contract=modal_case_contract,
            timing_witnesses=_ordered_modal_timing_witnesses(modal_timing_witnesses),
            finalized=modal_final_summary is not None,
            final_summary=modal_final_summary,
            runtime_contract_final=modal_final_runtime_contract,
            runtime_identity_audit_final=modal_final_runtime_identity_audit,
            runtime_dependency_fingerprint=(
                None
                if modal_final_dependency_audit is None
                else str(modal_final_dependency_audit.get("fingerprint"))
            ),
            loaded_resume_position_audit=modal_loaded_resume_position_audit,
            physical_consumption_claim=(
                "The runner proves exactly 117 global setExternalAction calls were completed "
                "in each wrapper call before advance, with same-call force generation and no "
                "software queue. It does not claim that QBlade physically consumed those "
                "actions; paired structural-response analysis is required."
            ),
        )
        summary["sil_modal_burst"] = asdict(modal_summary)
    summary_path = output_dir / f"{sim_path.stem}_run_summary.json"
    atomic_write_json(summary_path, summary)
    if unexpected_native_message is not None:
        raise RuntimeError(unexpected_native_message)
    return QBladeRunResult(
        summary=summary,
        summary_path=summary_path,
        result_path=result_path,
        run_log_path=run_log,
    )


__all__ = [
    "QBladeCouplingMode",
    "QBladeExecutionMode",
    "QBladeRunRequest",
    "QBladeRunResult",
    "QBladeRunSummary",
    "QBladeModalBurstRunSummary",
    "run_qblade_case",
]
