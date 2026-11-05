"""Recover two completed H-VAWT sensitivity artifacts without rerunning QBlade.

The sealed sensitivity campaign and its original result auditor remain unchanged.
Recovery is limited to the two half-timestep cases whose QBlade text exports
completed but were rejected because the frozen audit scaled a text-rounding
tolerance with the physical timestep.  The replacement audit accepts only the
deterministic ``%.5E`` timestamp tokens that QBlade can emit from the configured
grid, then reuses the complete frozen terminal audit for every other check.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import csv
import fcntl
import math
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

from . import hvawt_sensitivity_campaign as sensitivity
from .hvawt_response_pilot import (
    HVAWTResponsePilotError,
    _history_path_from_summary,
    _result_filter_channels,
)
from .qblade_run_support import atomic_write_json, canonical_fingerprint


RECOVERY_SCHEMA = "vorlap-hvawt-targeted-sensitivity-time-grid-recovery-v2"
RECOVERY_STATUS = "accepted_completed_half_timestep_artifacts_without_rerun"
RECOVERY_RECEIPT_NAME = "sensitivity_terminal_audit_recovery_v2.json"
RECOVERED_ANALYSIS_SCHEMA = (
    "vorlap-hvawt-targeted-sensitivity-analysis-recovered-v2"
)
RECOVERED_ANALYSIS_NAME = "sensitivity_analysis_recovered_v2.json"
RECOVERED_PROMOTION_SCHEMA = (
    "vorlap-hvawt-targeted-sensitivity-promotion-recovered-v2"
)
RECOVERED_PROMOTION_RECEIPT_NAME = "sensitivity_promotion_recovered_v2.json"
RECOVERED_EXPORT_NAME = "paper_export_sensitivity_recovered_v2"

EXPECTED_CAMPAIGN_FINGERPRINT = (
    "e434783d5389f9ae0145045b595e7f6f76fdb5d0670d7d297cf91cf5c17cce2e"
)
EXPECTED_RUNTIME_FINGERPRINT = (
    "d7b0a69f3bb8465c8a24e7adec362607f70ae6a5326a2c9bbf03e74a52b80938"
)
EXPECTED_SMOKE_FINGERPRINT = (
    "99c595b1f657b6b5848a1e5b18929a94d87839ae0d2f5e471a05ca165e121b17"
)
EXPECTED_AUTHORIZATION_FINGERPRINT = (
    "3af3e9ae5807523897827a51d6f8c039cdf9733826251cf9bc733654c302c93c"
)
EXPECTED_RECOVERED_INDICES = (1, 4)
EXPECTED_UNCHANGED_INDICES = (0, 2, 3, 5)
EXPECTED_VARIANT = "complete_half_timestep"
EXPECTED_HALF_TIMESTEP_S = 0.00375
EXPECTED_HALF_TOTAL_STEPS = 176_000
EXPECTED_HALF_RETAINED_ROWS = 160_000
FROZEN_HALF_TIMESTEP_TOLERANCE_S = 0.000375
OBSERVED_EXPORT_ERROR_S = 0.0005000000001018634
EXACT_FROZEN_ERROR = (
    "QBlade fatigue result time grid exceeds tolerance: "
    "0.0005000000001018634 > 0.000375 s."
)
EXACT_CLASSIFICATION_REASON = f"partial_or_invalid:{EXACT_FROZEN_ERROR}"
QBLADE_TIME_FORMAT = "%.5E"
QBLADE_TIME_FORMAT_DECIMALS = 5

_FROZEN_RESULT_AUDITOR = sensitivity._audit_qblade_fatigue_result


class HVAWTSensitivityRecoveryError(RuntimeError):
    """Raised when the exact read-only recovery contract is not satisfied."""


def _validate_fingerprint(payload: Mapping[str, Any], *, label: str) -> None:
    content = dict(payload)
    recorded = content.pop("fingerprint", None)
    if recorded != canonical_fingerprint(content):
        raise HVAWTSensitivityRecoveryError(f"{label} fingerprint is invalid.")


def _verify_absolute_identity(record: object, *, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise HVAWTSensitivityRecoveryError(f"Missing identity for {label}.")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise HVAWTSensitivityRecoveryError(
            f"Invalid absolute identity for {label}."
        )
    path = Path(raw).resolve()
    if not path.is_file():
        raise HVAWTSensitivityRecoveryError(f"Missing {label}: {path}")
    observed = sensitivity._identity(path)
    if observed != dict(record):
        raise HVAWTSensitivityRecoveryError(f"Identity changed for {label}.")
    return path


def _verify_relative_identity(
    root: Path, record: object, *, label: str
) -> Path:
    try:
        return sensitivity._verify_relative_identity(root, record, label=label)
    except sensitivity.HVAWTSensitivityError as exc:
        raise HVAWTSensitivityRecoveryError(str(exc)) from exc


def _declared_source_identities(
    recovery_sources: Mapping[str, Path] | None,
) -> dict[str, dict[str, object]]:
    repository = Path(sensitivity.__file__).resolve().parents[1]
    declared: dict[str, Path] = {
        "frozen_sensitivity_campaign": Path(sensitivity.__file__).resolve(),
        "frozen_result_auditor": (
            repository / "vorlap/hvawt_response_pilot.py"
        ).resolve(),
        "recovery_model": Path(__file__).resolve(),
    }
    if recovery_sources:
        declared.update(
            {
                str(role): Path(path).expanduser().resolve()
                for role, path in recovery_sources.items()
            }
        )
    if any(not role or not path.is_file() for role, path in declared.items()):
        raise HVAWTSensitivityRecoveryError(
            "Recovery source declarations are incomplete or missing."
        )
    return {
        role: sensitivity._identity(path)
        for role, path in sorted(declared.items())
    }


def _load_recovery_state(campaign: Path) -> dict[str, Any]:
    state = sensitivity.validate_campaign(campaign, rehash_case_assets=False)
    root = state["campaign_root"]
    manifest = state["manifest"]
    if manifest.get("fingerprint") != EXPECTED_CAMPAIGN_FINGERPRINT:
        raise HVAWTSensitivityRecoveryError(
            "Recovery is pinned to the accepted six-case sensitivity campaign."
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise HVAWTSensitivityRecoveryError(
            "Recovery requires the exact ordered six-case manifest."
        )
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or case.get("index") != index:
            raise HVAWTSensitivityRecoveryError(
                "Sensitivity case indices or ordering changed."
            )
        if index in EXPECTED_RECOVERED_INDICES:
            if (
                case.get("variant") != EXPECTED_VARIANT
                or not math.isclose(
                    float(case.get("timestep_s", math.nan)),
                    EXPECTED_HALF_TIMESTEP_S,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                or case.get("total_steps") != EXPECTED_HALF_TOTAL_STEPS
                or case.get("retained_rows") != EXPECTED_HALF_RETAINED_ROWS
            ):
                raise HVAWTSensitivityRecoveryError(
                    f"Recoverable case {index} is not the frozen half-timestep case."
                )
        elif index not in EXPECTED_UNCHANGED_INDICES:
            raise HVAWTSensitivityRecoveryError(
                f"Unexpected sensitivity case index {index}."
            )

    runtime = sensitivity._load_runtime(root)
    runtime_fingerprint = canonical_fingerprint(runtime)
    smoke = sensitivity.execute_smoke_receipt(root, manifest)
    authorization = sensitivity._load_authorization(root, manifest)
    if (
        runtime_fingerprint != EXPECTED_RUNTIME_FINGERPRINT
        or smoke.get("fingerprint") != EXPECTED_SMOKE_FINGERPRINT
        or authorization.get("fingerprint")
        != EXPECTED_AUTHORIZATION_FINGERPRINT
        or authorization.get("authorized_case_indices") != list(range(6))
        or authorization.get("fresh_start_only") is not True
    ):
        raise HVAWTSensitivityRecoveryError(
            "Runtime, smoke, or fresh-start authorization differs from Phase F."
        )
    if (root / sensitivity.ANALYSIS_NAME).exists() or (
        root / sensitivity.PAPER_EXPORT_NAME
    ).exists():
        raise HVAWTSensitivityRecoveryError(
            "Unversioned sensitivity analysis or promotion exists; stop for review."
        )
    return {
        **state,
        "runtime": runtime,
        "runtime_fingerprint": runtime_fingerprint,
        "smoke": smoke,
        "authorization": authorization,
    }


def _case_paths(
    *,
    root: Path,
    case: Mapping[str, Any],
    summary: Mapping[str, Any] | None = None,
    verify_history_identity: bool = True,
) -> dict[str, Path]:
    result_dir = root / str(case["result_dir"])
    sim_path = root / str(case["sim"])
    summary_path = result_dir / f"{sim_path.stem}_run_summary.json"
    payload = (
        sensitivity._load_json(summary_path, label=f"summary {case['case_id']}")
        if summary is None
        else summary
    )
    if verify_history_identity:
        try:
            history = _history_path_from_summary(payload, result_dir)
        except HVAWTResponsePilotError as exc:
            raise HVAWTSensitivityRecoveryError(str(exc)) from exc
    else:
        history_record = payload.get("sil_vorlap_history")
        raw_history = (
            history_record.get("path")
            if isinstance(history_record, Mapping)
            else None
        )
        if not isinstance(raw_history, str) or not raw_history:
            raise HVAWTSensitivityRecoveryError(
                f"Case {case['index']} lacks a compact-history path."
            )
        history = Path(raw_history)
    if not history.is_file():
        relocated = result_dir / history.name
        if relocated.is_file():
            history = relocated
    paths = {
        "summary": summary_path,
        "result": result_dir / f"{case['result_name']}.txt",
        "metadata": result_dir / "case_metadata.json",
        "history": history,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise HVAWTSensitivityRecoveryError(
            "Sensitivity terminal artifacts are missing: " + ", ".join(missing)
        )
    root_resolved = root.resolve()
    for role, path in paths.items():
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise HVAWTSensitivityRecoveryError(
                f"Sensitivity {role} is linked instead of an immutable plain file."
            )
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError as exc:
            raise HVAWTSensitivityRecoveryError(
                f"Sensitivity {role} escapes the campaign root."
            ) from exc
    return paths


def _artifact_identities(
    *, root: Path, case: Mapping[str, Any]
) -> dict[str, dict[str, object]]:
    return {
        role: sensitivity._relative_identity(path, root)
        for role, path in _case_paths(root=root, case=case).items()
    }


def _rendered_time_candidates(
    *, sample_index: int, timestep_s: float
) -> tuple[str, str, float, float, float]:
    """Return the at-most-two QBlade tokens allowed by repeated addition."""

    if sample_index < 0 or not math.isfinite(timestep_s) or timestep_s <= 0.0:
        raise HVAWTSensitivityRecoveryError("Invalid configured time-grid input.")
    unit_roundoff = 0.5 * sys.float_info.epsilon
    n_unit_roundoff = sample_index * unit_roundoff
    if n_unit_roundoff >= 1.0:
        raise HVAWTSensitivityRecoveryError(
            "Configured time grid exceeds the binary64 accumulation bound."
        )
    exact_time = sample_index * timestep_s
    gamma_n = n_unit_roundoff / (1.0 - n_unit_roundoff)
    accumulation_bound = math.nextafter(
        (gamma_n + unit_roundoff) * abs(exact_time), math.inf
    )
    if sample_index == 0:
        accumulation_bound = 0.0
    lower = QBLADE_TIME_FORMAT % (exact_time - accumulation_bound)
    upper = QBLADE_TIME_FORMAT % (exact_time + accumulation_bound)
    scale = max(abs(exact_time), timestep_s)
    printed_unit = 10.0 ** (
        math.floor(math.log10(scale)) - QBLADE_TIME_FORMAT_DECIMALS
    )
    rounding_bound = 0.5 * printed_unit + accumulation_bound
    parse_slack = 16.0 * sys.float_info.epsilon * max(
        1.0, abs(float(lower)), abs(float(upper))
    )
    if abs(float(upper) - float(lower)) > printed_unit + 2.0 * parse_slack:
        raise HVAWTSensitivityRecoveryError(
            "Binary64 interval spans more than two QBlade export tokens."
        )
    return lower, upper, exact_time, accumulation_bound, rounding_bound


def audit_qblade_export_time_grid(
    path: Path,
    *,
    result_filter: Path,
    timestep_s: float,
    total_steps: int,
    store_from_s: float,
) -> dict[str, object]:
    """Audit the exact QBlade ``%.5E`` tokens against the configured grid."""

    expected_header = [
        "~".join(channel.split())
        for channel in _result_filter_channels(result_filter)
    ]
    start_index = max(
        0, int(math.ceil(store_from_s / timestep_s - 1.0e-10))
    )
    expected_rows = int(total_steps) - start_index
    if expected_rows < 2:
        raise HVAWTSensitivityRecoveryError(
            "QBlade recovery audit requires at least two retained rows."
        )

    row_count = 0
    maximum_abs_load = 0.0
    maximum_time_error = 0.0
    maximum_accumulation_bound = 0.0
    maximum_rounding_bound = 0.0
    two_token_samples = 0
    first_time: float | None = None
    last_time: float | None = None
    minimum_adjacent = math.inf
    maximum_adjacent = -math.inf
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            next(reader)
            next(reader)
            header = [cell.strip() for cell in next(reader)]
        except StopIteration as exc:
            raise HVAWTSensitivityRecoveryError(
                "QBlade result lacks metadata, header, or numeric rows."
            ) from exc
        if header != expected_header:
            raise HVAWTSensitivityRecoveryError(
                "QBlade result header differs from the frozen 26-channel filter."
            )
        for line_number, row in enumerate(reader, start=4):
            if len(row) != len(header):
                raise HVAWTSensitivityRecoveryError(
                    f"QBlade result row {line_number} has {len(row)} fields; "
                    f"expected {len(header)}."
                )
            tokens = [token.strip() for token in row]
            try:
                values = [float(token) for token in tokens]
            except ValueError as exc:
                raise HVAWTSensitivityRecoveryError(
                    f"QBlade result row {line_number} is not numeric."
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise HVAWTSensitivityRecoveryError(
                    f"QBlade result row {line_number} is non-finite."
                )
            sample_index = start_index + row_count
            lower, upper, exact, accumulation, rounding = (
                _rendered_time_candidates(
                    sample_index=sample_index, timestep_s=timestep_s
                )
            )
            if tokens[0] not in {lower, upper}:
                raise HVAWTSensitivityRecoveryError(
                    "QBlade exported time is inconsistent with the configured "
                    f"grid and {QBLADE_TIME_FORMAT} rendering at sample "
                    f"{sample_index}: observed {tokens[0]!r}, admissible "
                    f"token(s) {lower!r} and {upper!r}."
                )
            if lower != upper:
                two_token_samples += 1
            value_time = values[0]
            if last_time is not None:
                adjacent = value_time - last_time
                if adjacent <= 0.0:
                    raise HVAWTSensitivityRecoveryError(
                        "QBlade result time is not strictly increasing."
                    )
                minimum_adjacent = min(minimum_adjacent, adjacent)
                maximum_adjacent = max(maximum_adjacent, adjacent)
            if first_time is None:
                first_time = value_time
            last_time = value_time
            maximum_time_error = max(
                maximum_time_error, abs(value_time - exact)
            )
            maximum_accumulation_bound = max(
                maximum_accumulation_bound, accumulation
            )
            maximum_rounding_bound = max(maximum_rounding_bound, rounding)
            maximum_abs_load = max(
                maximum_abs_load,
                max(abs(value) for value in values[2:]),
            )
            row_count += 1

    if row_count != expected_rows:
        raise HVAWTSensitivityRecoveryError(
            f"QBlade result has {row_count} retained rows; expected {expected_rows}."
        )
    return {
        "schema": "vorlap-qblade-rendered-time-grid-audit-v1",
        "channel_count": len(expected_header),
        "row_count": row_count,
        "expected_row_count": expected_rows,
        "first_time_s": first_time,
        "last_time_s": last_time,
        "maximum_time_grid_error_s": maximum_time_error,
        "time_grid_tolerance_s": maximum_rounding_bound,
        "maximum_abs_structural_load_channel_value": maximum_abs_load,
        "export_format": QBLADE_TIME_FORMAT,
        "acceptance_rule": (
            "each raw timestamp token exactly matches one of the at-most-two "
            "configured-grid tokens at its binary64 repeated-addition bounds"
        ),
        "strictly_increasing_exported_time": True,
        "minimum_exported_adjacent_time_s": minimum_adjacent,
        "maximum_exported_adjacent_time_s": maximum_adjacent,
        "two_token_rounding_boundary_sample_count": two_token_samples,
        "maximum_repeated_addition_bound_s": maximum_accumulation_bound,
        "maximum_renderer_mismatch_s": 0.0,
        "frozen_half_timestep_tolerance_s": FROZEN_HALF_TIMESTEP_TOLERANCE_S,
        "observed_failed_export_error_s": OBSERVED_EXPORT_ERROR_S,
        "recovery_scope": "QBlade textual timestamp representation only",
    }


def _recovery_result_auditor(
    path: Path,
    *,
    result_filter: Path,
    timestep_s: float,
    total_steps: int,
    store_from_s: float,
) -> dict[str, object]:
    if not math.isclose(
        timestep_s,
        EXPECTED_HALF_TIMESTEP_S,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        return _FROZEN_RESULT_AUDITOR(
            path,
            result_filter=result_filter,
            timestep_s=timestep_s,
            total_steps=total_steps,
            store_from_s=store_from_s,
        )
    if (
        total_steps != EXPECTED_HALF_TOTAL_STEPS
        or not math.isclose(
            store_from_s,
            sensitivity.STORE_FROM_S,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise HVAWTSensitivityRecoveryError(
            "Half-timestep recovery was requested outside its frozen run plan."
        )
    audit = audit_qblade_export_time_grid(
        path,
        result_filter=result_filter,
        timestep_s=timestep_s,
        total_steps=total_steps,
        store_from_s=store_from_s,
    )
    if (
        audit["row_count"] != EXPECTED_HALF_RETAINED_ROWS
        or audit["maximum_time_grid_error_s"] != OBSERVED_EXPORT_ERROR_S
    ):
        raise HVAWTSensitivityRecoveryError(
            "Half-timestep result does not reproduce the exact observed export grid."
        )
    return audit


@contextmanager
def _scoped_recovery_result_audit() -> Iterator[None]:
    if sensitivity._audit_qblade_fatigue_result is not _FROZEN_RESULT_AUDITOR:
        raise HVAWTSensitivityRecoveryError(
            "Sensitivity result auditor was already replaced; recovery stopped."
        )
    sensitivity._audit_qblade_fatigue_result = _recovery_result_auditor
    try:
        yield
    finally:
        sensitivity._audit_qblade_fatigue_result = _FROZEN_RESULT_AUDITOR


@contextmanager
def _locked_cases(root: Path, manifest: Mapping[str, Any]) -> Iterator[None]:
    lock_root = root / ".case_locks"
    with ExitStack() as stack:
        for case in manifest["cases"]:
            path = lock_root / f"{case['case_id']}.lock"
            if not path.is_file():
                raise HVAWTSensitivityRecoveryError(
                    f"Missing completed-case execution lock: {path}"
                )
            stream = stack.enter_context(path.open("r", encoding="utf-8"))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HVAWTSensitivityRecoveryError(
                    f"Case {case['index']} has an active writer; recovery stopped."
                ) from exc
        yield


def preflight_recovery(campaign: Path) -> dict[str, object]:
    """Perform a lightweight preflight without hashing terminal artifacts."""

    state = _load_recovery_state(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    summaries: list[dict[str, object]] = []
    for case in manifest["cases"]:
        paths = _case_paths(
            root=root, case=case, verify_history_identity=False
        )
        summary = sensitivity._load_json(
            paths["summary"], label=f"summary {case['case_id']}"
        )
        if (
            summary.get("failed") is not False
            or summary.get("stopped") is not False
            or summary.get("completed_steps") != case["total_steps"]
            or summary.get("requested_steps") != case["total_steps"]
            or summary.get("exported_results") is not True
        ):
            raise HVAWTSensitivityRecoveryError(
                f"Case {case['index']} does not have a completed run summary."
            )
        summaries.append(
            {
                "case_index": case["index"],
                "case_id": case["case_id"],
                "variant": case["variant"],
                "completed_steps": summary["completed_steps"],
            }
        )
    return {
        "schema": "vorlap-hvawt-targeted-sensitivity-recovery-preflight-v1",
        "status": "ready_for_scheduled_read_only_terminal_audit",
        "campaign_root": str(root),
        "campaign_fingerprint": manifest["fingerprint"],
        "runtime_fingerprint": state["runtime_fingerprint"],
        "smoke_fingerprint": state["smoke"]["fingerprint"],
        "authorization_fingerprint": state["authorization"]["fingerprint"],
        "accepted_candidate_count": 6,
        "recovery_candidate_indices": list(EXPECTED_RECOVERED_INDICES),
        "cases": summaries,
        "writes_performed": 0,
        "simulation_actions_available": False,
    }


def build_recovery_receipt(
    campaign: Path,
    *,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Deep-audit all six cases and construct a deterministic receipt."""

    state = _load_recovery_state(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    before = {
        str(int(case["index"])): _artifact_identities(root=root, case=case)
        for case in manifest["cases"]
    }
    terminal_cases: list[dict[str, object]] = []
    recovered_cases: list[dict[str, object]] = []
    for case in manifest["cases"]:
        index = int(case["index"])
        try:
            frozen_receipt = sensitivity._terminal_case_audit(
                root=root, manifest=manifest, case=case
            )
        except Exception as exc:
            if index not in EXPECTED_RECOVERED_INDICES:
                raise HVAWTSensitivityRecoveryError(
                    f"Unchanged case {index} failed its frozen audit: {exc}"
                ) from exc
            if (
                not isinstance(exc, HVAWTResponsePilotError)
                or str(exc) != EXACT_FROZEN_ERROR
            ):
                raise HVAWTSensitivityRecoveryError(
                    f"Case {index} failed an earlier or different gate: {exc}"
                ) from exc
            with _scoped_recovery_result_audit():
                recovered_receipt = sensitivity._terminal_case_audit(
                    root=root, manifest=manifest, case=case
                )
            recovered = {
                "case_index": index,
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "variant": case["variant"],
                "original_error_class": type(exc).__name__,
                "original_error_message": str(exc),
                "original_classification_reason": EXACT_CLASSIFICATION_REASON,
                "receipt": recovered_receipt,
                "artifacts": before[str(index)],
            }
            recovered["fingerprint"] = canonical_fingerprint(recovered)
            recovered_cases.append(recovered)
            terminal = {
                "case_index": index,
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "variant": case["variant"],
                "status": "complete",
                "audit_disposition": (
                    "recovered_qblade_text_timestamp_representation_only"
                ),
                "receipt": recovered_receipt,
            }
        else:
            if index in EXPECTED_RECOVERED_INDICES:
                raise HVAWTSensitivityRecoveryError(
                    f"Case {index} did not reproduce the exact frozen failure."
                )
            terminal = {
                "case_index": index,
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "variant": case["variant"],
                "status": "complete",
                "audit_disposition": "unchanged_frozen_terminal_audit",
                "receipt": frozen_receipt,
            }
        terminal_cases.append(terminal)

    if (
        tuple(row["case_index"] for row in recovered_cases)
        != EXPECTED_RECOVERED_INDICES
        or len(terminal_cases) != 6
    ):
        raise HVAWTSensitivityRecoveryError(
            "Recovery did not produce the exact four-unchanged/two-recovered set."
        )
    after = {
        str(int(case["index"])): _artifact_identities(root=root, case=case)
        for case in manifest["cases"]
    }
    modified = sum(
        before[str(index)][role] != after[str(index)][role]
        for index in range(6)
        for role in before[str(index)]
    )
    if modified:
        raise HVAWTSensitivityRecoveryError(
            "A terminal simulation artifact changed during recovery."
        )

    report: dict[str, object] = {
        "schema": RECOVERY_SCHEMA,
        "status": RECOVERY_STATUS,
        "campaign_fingerprint": manifest["fingerprint"],
        "campaign_manifest": sensitivity._relative_identity(
            root / sensitivity.MANIFEST_NAME, root
        ),
        "runtime": sensitivity._relative_identity(
            root / sensitivity.RUNTIME_RECEIPT_NAME, root
        ),
        "runtime_fingerprint": state["runtime_fingerprint"],
        "smoke_receipt": sensitivity._relative_identity(
            root / sensitivity.SMOKE_RECEIPT_NAME, root
        ),
        "smoke_fingerprint": state["smoke"]["fingerprint"],
        "execution_authorization": sensitivity._relative_identity(
            root / sensitivity.AUTHORIZATION_NAME, root
        ),
        "authorization_fingerprint": state["authorization"]["fingerprint"],
        "recovery_sources": _declared_source_identities(recovery_sources),
        "accepted_case_count": 6,
        "unchanged_frozen_audit_case_count": 4,
        "recovered_case_count": 2,
        "unchanged_case_indices": list(EXPECTED_UNCHANGED_INDICES),
        "recovered_case_indices": list(EXPECTED_RECOVERED_INDICES),
        "recovered_case_ids": [row["case_id"] for row in recovered_cases],
        "simulation_rerun_count": 0,
        "modified_artifact_count": modified,
        "terminal_cases": terminal_cases,
        "recovered_cases": recovered_cases,
        "artifact_identities_before": before,
        "artifact_identities_after": after,
        "time_grid_recovery_policy": {
            "scope": "two half-timestep QBlade textual time columns only",
            "frozen_error": EXACT_FROZEN_ERROR,
            "frozen_tolerance_s": FROZEN_HALF_TIMESTEP_TOLERANCE_S,
            "observed_maximum_error_s": OBSERVED_EXPORT_ERROR_S,
            "export_format": QBLADE_TIME_FORMAT,
            "acceptance": (
                "exact raw-token match to the at-most-two deterministic renders "
                "allowed by the configured grid and binary64 accumulation bound"
            ),
            "compact_history_time_tolerance_s": 1.0e-9,
        },
        "claim_boundary": (
            "Read-only audit recovery of two completed numerical-sensitivity "
            "artifacts. No simulation was rerun or modified, and this creates "
            "no new physical validation, fatigue-life, or certification evidence."
        ),
    }
    report["fingerprint"] = canonical_fingerprint(report)
    return report


def write_recovery_receipt(
    campaign: Path,
    *,
    output: Path | None = None,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    state = _load_recovery_state(campaign)
    root = state["campaign_root"]
    target = (
        root / RECOVERY_RECEIPT_NAME
        if output is None
        else Path(output).expanduser().resolve()
    )
    if target != root / RECOVERY_RECEIPT_NAME:
        raise HVAWTSensitivityRecoveryError(
            f"Recovery receipt must be {root / RECOVERY_RECEIPT_NAME}."
        )
    with _locked_cases(root, state["manifest"]):
        report = build_recovery_receipt(
            root, recovery_sources=recovery_sources
        )
        if target.exists():
            existing = sensitivity._load_json(
                target, label="existing sensitivity recovery receipt"
            )
            if existing != report:
                raise HVAWTSensitivityRecoveryError(
                    "Existing recovery receipt differs from the live audit."
                )
            return existing
        atomic_write_json(target, report)
        recorded = sensitivity._load_json(
            target, label="written sensitivity recovery receipt"
        )
        if recorded != report:
            raise HVAWTSensitivityRecoveryError(
                "Written recovery receipt differs from its audited payload."
            )
        return recorded


def load_recovery_receipt(
    campaign: Path,
    *,
    receipt_path: Path | None = None,
    verify_live_artifacts: bool = True,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    state = _load_recovery_state(campaign)
    root = state["campaign_root"]
    path = (
        root / RECOVERY_RECEIPT_NAME
        if receipt_path is None
        else Path(receipt_path).expanduser().resolve()
    )
    if path != root / RECOVERY_RECEIPT_NAME:
        raise HVAWTSensitivityRecoveryError(
            "Recovery receipt path differs from the campaign-root contract."
        )
    receipt = sensitivity._load_json(path, label="sensitivity recovery receipt")
    _validate_fingerprint(receipt, label="Sensitivity recovery receipt")
    if (
        receipt.get("schema") != RECOVERY_SCHEMA
        or receipt.get("status") != RECOVERY_STATUS
        or receipt.get("campaign_fingerprint") != EXPECTED_CAMPAIGN_FINGERPRINT
        or receipt.get("runtime_fingerprint") != EXPECTED_RUNTIME_FINGERPRINT
        or receipt.get("smoke_fingerprint") != EXPECTED_SMOKE_FINGERPRINT
        or receipt.get("authorization_fingerprint")
        != EXPECTED_AUTHORIZATION_FINGERPRINT
        or receipt.get("accepted_case_count") != 6
        or receipt.get("unchanged_frozen_audit_case_count") != 4
        or receipt.get("recovered_case_count") != 2
        or tuple(receipt.get("unchanged_case_indices", ()))
        != EXPECTED_UNCHANGED_INDICES
        or tuple(receipt.get("recovered_case_indices", ()))
        != EXPECTED_RECOVERED_INDICES
        or receipt.get("recovered_case_ids")
        != [
            state["manifest"]["cases"][index]["case_id"]
            for index in EXPECTED_RECOVERED_INDICES
        ]
        or receipt.get("simulation_rerun_count") != 0
        or receipt.get("modified_artifact_count") != 0
    ):
        raise HVAWTSensitivityRecoveryError(
            "Sensitivity recovery receipt contract is invalid."
        )
    terminal_cases = receipt.get("terminal_cases")
    recovered_cases = receipt.get("recovered_cases")
    before = receipt.get("artifact_identities_before")
    after = receipt.get("artifact_identities_after")
    if (
        not isinstance(terminal_cases, list)
        or len(terminal_cases) != 6
        or not isinstance(recovered_cases, list)
        or len(recovered_cases) != 2
        or not isinstance(before, Mapping)
        or before != after
    ):
        raise HVAWTSensitivityRecoveryError(
            "Sensitivity recovery case or artifact records are invalid."
        )
    expected_dispositions = {
        index: (
            "recovered_qblade_text_timestamp_representation_only"
            if index in EXPECTED_RECOVERED_INDICES
            else "unchanged_frozen_terminal_audit"
        )
        for index in range(6)
    }
    for index, (record, planned) in enumerate(
        zip(terminal_cases, state["manifest"]["cases"], strict=True)
    ):
        if (
            not isinstance(record, Mapping)
            or record.get("case_index") != index
            or record.get("case_id") != planned["case_id"]
            or record.get("row_id") != planned["row_id"]
            or record.get("variant") != planned["variant"]
            or record.get("status") != "complete"
            or record.get("audit_disposition") != expected_dispositions[index]
        ):
            raise HVAWTSensitivityRecoveryError(
                f"Sensitivity terminal recovery record {index} is invalid."
            )
    for record, index in zip(
        recovered_cases, EXPECTED_RECOVERED_INDICES, strict=True
    ):
        if not isinstance(record, Mapping):
            raise HVAWTSensitivityRecoveryError(
                "Recovered case record is not an object."
            )
        _validate_fingerprint(record, label=f"Recovered case {index}")
        if (
            record.get("case_index") != index
            or record.get("case_id")
            != state["manifest"]["cases"][index]["case_id"]
            or record.get("variant") != EXPECTED_VARIANT
            or record.get("original_error_class")
            != "HVAWTResponsePilotError"
            or record.get("original_error_message") != EXACT_FROZEN_ERROR
            or record.get("original_classification_reason")
            != EXACT_CLASSIFICATION_REASON
        ):
            raise HVAWTSensitivityRecoveryError(
                f"Recovered case {index} does not retain the exact frozen failure."
            )
    policy = receipt.get("time_grid_recovery_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("frozen_error") != EXACT_FROZEN_ERROR
        or policy.get("frozen_tolerance_s")
        != FROZEN_HALF_TIMESTEP_TOLERANCE_S
        or policy.get("observed_maximum_error_s") != OBSERVED_EXPORT_ERROR_S
        or policy.get("export_format") != QBLADE_TIME_FORMAT
        or policy.get("compact_history_time_tolerance_s") != 1.0e-9
    ):
        raise HVAWTSensitivityRecoveryError(
            "Sensitivity time-grid recovery policy changed."
        )
    for index in range(6):
        artifacts = before.get(str(index))
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "summary",
            "result",
            "metadata",
            "history",
        }:
            raise HVAWTSensitivityRecoveryError(
                f"Sensitivity artifact records for case {index} are invalid."
            )
        for role, record in artifacts.items():
            _verify_relative_identity(root, record, label=f"case {index} {role}")
    for role in (
        "campaign_manifest",
        "runtime",
        "smoke_receipt",
        "execution_authorization",
    ):
        _verify_relative_identity(root, receipt.get(role), label=role)
    sources = receipt.get("recovery_sources")
    if not isinstance(sources, Mapping) or not sources:
        raise HVAWTSensitivityRecoveryError(
            "Recovery receipt does not bind its source files."
        )
    for role, record in sources.items():
        _verify_absolute_identity(record, label=f"recovery source {role}")
    if verify_live_artifacts:
        with _locked_cases(root, state["manifest"]):
            reproduced = build_recovery_receipt(
                root, recovery_sources=recovery_sources
            )
        if reproduced != receipt:
            raise HVAWTSensitivityRecoveryError(
                "Live artifacts no longer reproduce the recovery receipt."
            )
    return receipt


@contextmanager
def _capture_analysis_write() -> Iterator[dict[str, Any]]:
    captured: dict[str, Any] = {}
    original = sensitivity._write_new_or_exact_json

    def capture(
        path: Path,
        payload: Mapping[str, Any],
        *,
        label: str,
        mismatch_message: str,
    ) -> dict[str, Any]:
        del label, mismatch_message
        captured["path"] = Path(path).resolve()
        captured["payload"] = dict(payload)
        return dict(payload)

    sensitivity._write_new_or_exact_json = capture
    try:
        yield captured
    finally:
        sensitivity._write_new_or_exact_json = original


def build_recovered_analysis(
    campaign: Path,
    *,
    base_campaign_root: Path,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    state = _load_recovery_state(campaign)
    root = state["campaign_root"]
    receipt = load_recovery_receipt(
        root,
        verify_live_artifacts=True,
        recovery_sources=recovery_sources,
    )
    with _locked_cases(root, state["manifest"]):
        with _scoped_recovery_result_audit(), _capture_analysis_write() as captured:
            report = sensitivity.analyze_campaign(
                root,
                base_campaign_root=base_campaign_root,
                output=root / RECOVERED_ANALYSIS_NAME,
            )
    if captured.get("payload") != report or captured.get("path") != (
        root / RECOVERED_ANALYSIS_NAME
    ).resolve():
        raise HVAWTSensitivityRecoveryError(
            "Recovered analysis did not use the captured versioned output."
        )
    report = dict(report)
    report.pop("fingerprint", None)
    report["schema"] = RECOVERED_ANALYSIS_SCHEMA
    report["analysis_variant"] = "read_only_half_timestep_time_grid_recovery_v2"
    report["recovery_receipt"] = sensitivity._relative_identity(
        root / RECOVERY_RECEIPT_NAME, root
    )
    report["recovery_fingerprint"] = receipt["fingerprint"]
    report["recovered_case_count"] = 2
    report["recovered_case_indices"] = list(EXPECTED_RECOVERED_INDICES)
    report["simulation_rerun_count"] = 0
    report["modified_terminal_artifact_count"] = 0
    report["terminal_audit_dispositions"] = [
        {
            "case_index": row["case_index"],
            "case_id": row["case_id"],
            "audit_disposition": row["audit_disposition"],
        }
        for row in receipt["terminal_cases"]
    ]
    report["recovery_claim_boundary"] = receipt["claim_boundary"]
    report["fingerprint"] = canonical_fingerprint(report)
    return report


def write_recovered_analysis(
    campaign: Path,
    *,
    base_campaign_root: Path,
    output: Path | None = None,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    root = _load_recovery_state(campaign)["campaign_root"]
    target = (
        root / RECOVERED_ANALYSIS_NAME
        if output is None
        else Path(output).expanduser().resolve()
    )
    if target != root / RECOVERED_ANALYSIS_NAME:
        raise HVAWTSensitivityRecoveryError(
            f"Recovered analysis must be {root / RECOVERED_ANALYSIS_NAME}."
        )
    report = build_recovered_analysis(
        root,
        base_campaign_root=base_campaign_root,
        recovery_sources=recovery_sources,
    )
    if target.exists():
        existing = sensitivity._load_json(
            target, label="existing recovered sensitivity analysis"
        )
        if existing != report:
            raise HVAWTSensitivityRecoveryError(
                "Existing recovered analysis differs from the live evidence."
            )
        return existing
    atomic_write_json(target, report)
    recorded = sensitivity._load_json(
        target, label="written recovered sensitivity analysis"
    )
    if recorded != report:
        raise HVAWTSensitivityRecoveryError(
            "Written recovered analysis differs from its audited payload."
        )
    return recorded


def _load_recovered_analysis(
    campaign: Path,
    *,
    recovery_sources: Mapping[str, Path] | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    state = _load_recovery_state(campaign)
    root = state["campaign_root"]
    receipt = load_recovery_receipt(
        root,
        verify_live_artifacts=True,
        recovery_sources=recovery_sources,
    )
    path = root / RECOVERED_ANALYSIS_NAME
    analysis = sensitivity._load_json(path, label="recovered sensitivity analysis")
    _validate_fingerprint(analysis, label="Recovered sensitivity analysis")
    if (
        analysis.get("schema") != RECOVERED_ANALYSIS_SCHEMA
        or analysis.get("campaign_fingerprint") != EXPECTED_CAMPAIGN_FINGERPRINT
        or analysis.get("complete_case_count") != 6
        or analysis.get("expected_case_count") != 6
        or analysis.get("all_execution_audits_passed") is not True
        or analysis.get("recovery_fingerprint") != receipt["fingerprint"]
        or analysis.get("recovered_case_count") != 2
        or tuple(analysis.get("recovered_case_indices", ()))
        != EXPECTED_RECOVERED_INDICES
        or analysis.get("simulation_rerun_count") != 0
        or analysis.get("modified_terminal_artifact_count") != 0
        or analysis.get("recovery_receipt")
        != sensitivity._relative_identity(root / RECOVERY_RECEIPT_NAME, root)
    ):
        raise HVAWTSensitivityRecoveryError(
            "Recovered sensitivity analysis contract is invalid."
        )
    return path, analysis, receipt


@contextmanager
def _scoped_recovered_analysis_schema() -> Iterator[None]:
    original = sensitivity.ANALYSIS_SCHEMA
    sensitivity.ANALYSIS_SCHEMA = RECOVERED_ANALYSIS_SCHEMA
    try:
        yield
    finally:
        sensitivity.ANALYSIS_SCHEMA = original


def promote_recovered_analysis(
    campaign: Path,
    *,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    state = _load_recovery_state(campaign)
    root = state["campaign_root"]
    analysis_path, analysis, receipt = _load_recovered_analysis(
        root, recovery_sources=recovery_sources
    )
    output_dir = root / RECOVERED_EXPORT_NAME
    with _scoped_recovered_analysis_schema():
        inner = sensitivity.promote_analysis(
            root,
            analysis_path=analysis_path,
            output_dir=output_dir,
        )
    inner_path = output_dir / sensitivity.PROMOTION_RECEIPT_NAME
    if sensitivity._load_json(
        inner_path, label="inner recovered sensitivity promotion"
    ) != inner:
        raise HVAWTSensitivityRecoveryError(
            "Inner recovered promotion receipt changed after publication."
        )
    report: dict[str, object] = {
        "schema": RECOVERED_PROMOTION_SCHEMA,
        "status": "promoted_recovered_sensitivity_analysis_with_explicit_provenance",
        "campaign_fingerprint": EXPECTED_CAMPAIGN_FINGERPRINT,
        "recovery_receipt": sensitivity._relative_identity(
            root / RECOVERY_RECEIPT_NAME, root
        ),
        "recovery_fingerprint": receipt["fingerprint"],
        "analysis": sensitivity._relative_identity(analysis_path, root),
        "analysis_fingerprint": analysis["fingerprint"],
        "inner_promotion": sensitivity._relative_identity(inner_path, root),
        "inner_promotion_fingerprint": inner["fingerprint"],
        "outputs": inner["outputs"],
        "recovered_case_indices": list(EXPECTED_RECOVERED_INDICES),
        "simulation_rerun_count": 0,
        "modified_terminal_artifact_count": 0,
        "claim_boundary": receipt["claim_boundary"],
    }
    report["fingerprint"] = canonical_fingerprint(report)
    target = root / RECOVERED_PROMOTION_RECEIPT_NAME
    if target.exists():
        existing = sensitivity._load_json(
            target, label="existing recovered promotion receipt"
        )
        if existing != report:
            raise HVAWTSensitivityRecoveryError(
                "Existing recovered promotion receipt differs from live outputs."
            )
        return existing
    atomic_write_json(target, report)
    recorded = sensitivity._load_json(
        target, label="written recovered promotion receipt"
    )
    if recorded != report:
        raise HVAWTSensitivityRecoveryError(
            "Written recovered promotion receipt differs from live outputs."
        )
    return recorded


__all__ = [
    "EXACT_CLASSIFICATION_REASON",
    "EXACT_FROZEN_ERROR",
    "EXPECTED_RECOVERED_INDICES",
    "HVAWTSensitivityRecoveryError",
    "RECOVERED_ANALYSIS_NAME",
    "RECOVERED_ANALYSIS_SCHEMA",
    "RECOVERED_EXPORT_NAME",
    "RECOVERED_PROMOTION_RECEIPT_NAME",
    "RECOVERY_RECEIPT_NAME",
    "RECOVERY_SCHEMA",
    "audit_qblade_export_time_grid",
    "build_recovered_analysis",
    "build_recovery_receipt",
    "load_recovery_receipt",
    "preflight_recovery",
    "promote_recovered_analysis",
    "write_recovered_analysis",
    "write_recovery_receipt",
]
