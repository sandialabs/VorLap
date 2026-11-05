"""Recover completed H-VAWT one-way artifacts rejected by one audit shape.

This module is intentionally additive.  It never calls a QBlade execution path,
never archives a result, and never edits the sealed campaign or its terminal
artifacts.  Recovery is allowed only when the unchanged frozen terminal audit
reaches its exact final one-way response-diagnostics error and the nonempty
diagnostics object is proven to be the exact inert sentinel emitted by the SIL
runtime.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import csv
import fcntl
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

from . import hvawt_complete_dlc_campaign as sealed
from .hvawt_response_pilot import _history_path_from_summary, _identity, _load_json
from .qblade_run_support import atomic_write_json, canonical_fingerprint


RECOVERY_SCHEMA = "vorlap-hvawt-one-way-audit-recovery-v1"
RECOVERY_STATUS = "accepted_completed_one_way_artifacts_without_rerun"
RECOVERED_ANALYSIS_SCHEMA = "vorlap-hvawt-complete-dlc-analysis-recovered-v3"
RECOVERED_ANALYSIS_STATUS = "complete_ready_for_load_postprocess"
RECOVERY_RECEIPT_NAME = "one_way_audit_recovery_receipt.json"
RECOVERED_ANALYSIS_NAME = "analysis_recovered_v3.json"
ORIGINAL_ANALYSIS_NAME = "analysis.json"
EXACT_FROZEN_ERROR = "One-way result unexpectedly activated a response closure."
EXACT_CLASSIFICATION_REASON = f"partial_or_invalid:{EXACT_FROZEN_ERROR}"
EXPECTED_ONE_WAY_INDICES = tuple(range(1, 102, 3))

INERT_INTEGER_FIELDS = (
    "samples",
    "gain_limited_steps",
    "force_limited_steps",
    "gain_limited_node_samples",
    "force_limited_node_samples",
    "max_gain_limited_nodes",
    "max_force_limited_nodes",
    "target_power_limited_steps",
    "generalized_force_limited_steps",
    "nodal_force_limited_steps",
    "nodal_force_limited_node_samples",
    "max_nodal_force_limited_nodes",
)
INERT_FLOAT_FIELDS = (
    "max_amplitude_ratio",
    "max_abs_modal_power_w",
    "max_response_nodal_force_n",
    "max_amplitude_estimate_m",
    "max_abs_target_cycle_mean_power_w",
    "max_abs_applied_target_cycle_mean_power_w",
    "max_abs_instantaneous_modal_power_w",
    "max_abs_instantaneous_structural_power_w",
    "max_abs_realized_modal_power_w",
    "max_abs_realized_structural_power_w",
)
INERT_RESPONSE_FIELDS = frozenset(
    ("model_type", *INERT_INTEGER_FIELDS, *INERT_FLOAT_FIELDS)
)


class HVAWTOneWayRecoveryError(RuntimeError):
    """Raised when a completed case is not exactly recoverable."""


def _relative_identity(path: Path, root: Path) -> dict[str, object]:
    value = _identity(path)
    try:
        relative = Path(value["path"]).resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise HVAWTOneWayRecoveryError(
            f"Recovery artifact lies outside the campaign root: {path}"
        ) from exc
    return {
        "path": relative.as_posix(),
        "sha256": value["sha256"],
        "size_bytes": value["size_bytes"],
    }


def _verify_relative_identity(
    root: Path, record: object, *, label: str
) -> Path:
    if not isinstance(record, Mapping):
        raise HVAWTOneWayRecoveryError(f"Missing identity for {label}.")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise HVAWTOneWayRecoveryError(f"Invalid relative identity for {label}.")
    base = root.resolve()
    candidate = (base / raw).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HVAWTOneWayRecoveryError(
            f"Identity for {label} escapes the campaign root."
        ) from exc
    if not candidate.is_file():
        raise HVAWTOneWayRecoveryError(f"Missing {label}: {candidate}")
    observed = _identity(candidate)
    if (
        observed["sha256"] != record.get("sha256")
        or observed["size_bytes"] != record.get("size_bytes")
    ):
        raise HVAWTOneWayRecoveryError(f"Identity changed for {label}.")
    return candidate


def _verify_absolute_identity(record: object, *, label: str) -> Path:
    """Verify a receipt-bound source without silently relocating it."""

    if not isinstance(record, Mapping):
        raise HVAWTOneWayRecoveryError(f"Missing identity for {label}.")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise HVAWTOneWayRecoveryError(f"Invalid absolute identity for {label}.")
    candidate = Path(raw).resolve()
    if not candidate.is_file():
        raise HVAWTOneWayRecoveryError(f"Missing {label}: {candidate}")
    observed = _identity(candidate)
    if (
        observed["path"] != str(candidate)
        or observed["sha256"] != record.get("sha256")
        or observed["size_bytes"] != record.get("size_bytes")
    ):
        raise HVAWTOneWayRecoveryError(f"Identity changed for {label}.")
    return candidate


def _validate_fingerprint(payload: Mapping[str, Any], *, label: str) -> None:
    content = dict(payload)
    recorded = content.pop("fingerprint", None)
    if recorded != canonical_fingerprint(content):
        raise HVAWTOneWayRecoveryError(f"{label} fingerprint is invalid.")


@contextmanager
def _recovery_validation_scope() -> Iterator[None]:
    """Validate the frozen auditor while excluding unused postprocessors.

    The v3 manifest binds several execution and paper-orchestration sources.
    Paper sources evolved after execution, but the recovery path calls none of
    them.  Keep the manifest-bound campaign model as the exact live authority;
    all other source roles remain recorded in the manifest and are reported as
    deliberately unused by the additive recovery receipt.
    """

    original = sealed.ORCHESTRATION_SOURCE_RELATIVE
    campaign_model = original.get("campaign_model")
    if campaign_model is None:
        raise HVAWTOneWayRecoveryError(
            "Frozen campaign does not declare its campaign-model source."
        )
    sealed.ORCHESTRATION_SOURCE_RELATIVE = {"campaign_model": campaign_model}
    try:
        yield
    finally:
        sealed.ORCHESTRATION_SOURCE_RELATIVE = original


def _validate_campaign(
    campaign: Path, *, deep: bool, rehash_external_winds: bool
) -> dict[str, Any]:
    with _recovery_validation_scope():
        return sealed.validate_campaign(
            campaign,
            rehash_external_winds=rehash_external_winds,
            rehash_case_assets=deep,
            deep_case_validation=deep,
        )


def _load_continuation_authorization(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        from run_headless import hvawt_complementary_dlc_support as continuation

        return continuation._load_complementary_authorization(
            root,
            manifest,
            repository=Path(sealed.__file__).resolve().parents[1],
        )
    except Exception as exc:
        raise HVAWTOneWayRecoveryError(
            f"Complementary execution authorization is invalid: {exc}"
        ) from exc


def _validate_original_analysis(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    path = root / ORIGINAL_ANALYSIS_NAME
    analysis = _load_json(path, label="original complete-DLC analysis")
    if (
        analysis.get("schema") != sealed.ANALYSIS_SCHEMA
        or analysis.get("status") != "incomplete"
        or analysis.get("campaign_fingerprint") != manifest.get("fingerprint")
        or analysis.get("expected_case_count") != 102
        or analysis.get("complete_case_count") != 68
        or analysis.get("all_three_scenarios_complete") is not False
    ):
        raise HVAWTOneWayRecoveryError(
            "Original analysis is not the exact 68/102 incomplete v2 report."
        )
    rows = analysis.get("cases")
    cases = manifest.get("cases")
    if not isinstance(rows, list) or not isinstance(cases, list) or len(rows) != 102:
        raise HVAWTOneWayRecoveryError(
            "Original analysis does not contain all 102 ordered case rows."
        )
    complete_counts = {scenario: 0 for scenario in sealed.SCENARIOS}
    blocked: list[int] = []
    for index, (row, case) in enumerate(zip(rows, cases, strict=True)):
        if not isinstance(row, Mapping) or not isinstance(case, Mapping):
            raise HVAWTOneWayRecoveryError(
                f"Original analysis row {index} is not an object."
            )
        for key in ("case_id", "row_id", "scenario"):
            if row.get(key) != case.get(key):
                raise HVAWTOneWayRecoveryError(
                    f"Original analysis row {index} changes {key}."
                )
        scenario = str(case.get("scenario"))
        if index in EXPECTED_ONE_WAY_INDICES:
            if (
                scenario != sealed.SCENARIOS[1]
                or row.get("status") != "blocked"
                or row.get("reason") != EXACT_CLASSIFICATION_REASON
            ):
                raise HVAWTOneWayRecoveryError(
                    f"Original one-way analysis row {index} has a different failure."
                )
            blocked.append(index)
        else:
            if row.get("status") != "complete" or not isinstance(
                row.get("receipt"), Mapping
            ):
                raise HVAWTOneWayRecoveryError(
                    f"Original non-one-way analysis row {index} is not complete."
                )
            if scenario not in complete_counts:
                raise HVAWTOneWayRecoveryError(
                    f"Original analysis row {index} has an unknown scenario."
                )
            complete_counts[scenario] += 1
    if tuple(blocked) != EXPECTED_ONE_WAY_INDICES or complete_counts != {
        sealed.SCENARIOS[0]: 34,
        sealed.SCENARIOS[1]: 0,
        sealed.SCENARIOS[2]: 34,
    }:
        raise HVAWTOneWayRecoveryError(
            "Original analysis case composition differs from 34 raw, 34 blocked "
            "one-way, and 34 complete cases."
        )
    return path, analysis


def audit_inert_response_summary(summary: object) -> dict[str, object]:
    if not isinstance(summary, Mapping) or set(summary) != INERT_RESPONSE_FIELDS:
        raise HVAWTOneWayRecoveryError(
            "One-way response diagnostics differ from the exact inert schema."
        )
    if summary.get("model_type") != "none":
        raise HVAWTOneWayRecoveryError(
            "One-way response diagnostics do not declare model_type='none'."
        )
    for name in INERT_INTEGER_FIELDS:
        if type(summary.get(name)) is not int or summary[name] != 0:
            raise HVAWTOneWayRecoveryError(
                f"One-way inert response count {name!r} is not integer zero."
            )
    for name in INERT_FLOAT_FIELDS:
        value = summary.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HVAWTOneWayRecoveryError(
                f"One-way inert response value {name!r} is not numeric."
            )
        if not math.isfinite(float(value)) or float(value) != 0.0:
            raise HVAWTOneWayRecoveryError(
                f"One-way inert response value {name!r} is not finite zero."
            )
    return {
        "model_type": "none",
        "integer_zero_field_count": len(INERT_INTEGER_FIELDS),
        "numeric_zero_field_count": len(INERT_FLOAT_FIELDS),
        "explicit_unified_response_closure_active": False,
    }


def audit_one_way_source_coverage(rows: object) -> dict[str, object]:
    if not isinstance(rows, list) or len(rows) != len(
        sealed.HVAWT_V8_COMPONENT_NODE_COUNTS
    ):
        raise HVAWTOneWayRecoveryError(
            "One-way source coverage does not contain all nine components."
        )
    expected_components = set(sealed.HVAWT_V8_COMPONENT_NODE_COUNTS)
    observed: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise HVAWTOneWayRecoveryError(
                f"One-way source coverage row {index} is not an object."
            )
        component = row.get("component")
        if (
            not isinstance(component, str)
            or component not in expected_components
            or component in observed
        ):
            raise HVAWTOneWayRecoveryError(
                "One-way source coverage component set is invalid."
            )
        observed.add(component)
        for field in (
            "max_response_nodal_force_n",
            "response_force_coefficient_max_abs",
        ):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HVAWTOneWayRecoveryError(
                    f"One-way source coverage {field!r} is not numeric."
                )
            if not math.isfinite(float(value)) or float(value) != 0.0:
                raise HVAWTOneWayRecoveryError(
                    f"One-way source coverage {field!r} is not finite zero."
                )
        for total, base in (
            ("max_total_nodal_force_n", "max_nodal_force_n"),
            ("total_force_coefficient_max_abs", "force_coefficient_max_abs"),
        ):
            if row.get(total) != row.get(base):
                raise HVAWTOneWayRecoveryError(
                    f"One-way source coverage {total!r} differs from {base!r}."
                )
    if observed != expected_components:
        raise HVAWTOneWayRecoveryError(
            "One-way source coverage component set is incomplete."
        )
    return {
        "component_count": len(observed),
        "components": sorted(observed),
        "response_force_fields_exactly_zero": True,
        "total_force_equals_legacy_spectral_force": True,
    }


def _case_paths(
    *, root: Path, case: Mapping[str, Any], summary: Mapping[str, Any] | None = None
) -> dict[str, Path]:
    result_dir = root / str(case["result_dir"])
    sim_path = root / str(case["sim"])
    summary_path = result_dir / f"{sim_path.stem}_run_summary.json"
    payload = (
        _load_json(summary_path, label=f"summary {case['case_id']}")
        if summary is None
        else summary
    )
    return {
        "summary": summary_path,
        "result": result_dir / f"{case['result_name']}.txt",
        "metadata": result_dir / "case_metadata.json",
        "history": _history_path_from_summary(payload, result_dir),
    }


def _artifact_identities(
    *, root: Path, case: Mapping[str, Any], summary: Mapping[str, Any] | None = None
) -> dict[str, dict[str, object]]:
    return {
        name: _relative_identity(path, root)
        for name, path in _case_paths(root=root, case=case, summary=summary).items()
    }


def _audit_one_way_case(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
    original_analysis_row: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, object]:
    index = case.get("index")
    if type(index) is not int or index not in EXPECTED_ONE_WAY_INDICES:
        raise HVAWTOneWayRecoveryError("Recoverable one-way case index is invalid.")
    if case.get("scenario") != sealed.SCENARIOS[1]:
        raise HVAWTOneWayRecoveryError("Recovery was offered a non-one-way case.")
    if case.get("force_contract") != sealed.SCENARIO_FORCE_CONTRACT[
        sealed.SCENARIOS[1]
    ]:
        raise HVAWTOneWayRecoveryError(
            f"One-way case {index} does not retain the exact 1/0/0 force contract."
        )
    if (
        original_analysis_row.get("status") != "blocked"
        or original_analysis_row.get("reason") != EXACT_CLASSIFICATION_REASON
    ):
        raise HVAWTOneWayRecoveryError(
            f"One-way case {index} lacks the exact original audit failure."
        )
    try:
        sealed._terminal_case_audit(root=root, manifest=manifest, case=case)
    except sealed.HVAWTCompleteDLCError as exc:
        if str(exc) != EXACT_FROZEN_ERROR:
            raise HVAWTOneWayRecoveryError(
                f"One-way case {index} failed an earlier/different frozen gate: {exc}"
            ) from exc
    except Exception as exc:
        raise HVAWTOneWayRecoveryError(
            f"One-way case {index} raised the wrong frozen-audit exception: {exc}"
        ) from exc
    else:
        raise HVAWTOneWayRecoveryError(
            f"One-way case {index} did not reproduce the exact frozen audit error."
        )

    paths = _case_paths(root=root, case=case)
    summary = _load_json(paths["summary"], label=f"summary {case['case_id']}")
    metadata = _load_json(paths["metadata"], label=f"metadata {case['case_id']}")
    expected_metadata = {
        "schema": "vorlap-hvawt-complete-dlc-case-metadata-v1",
        "campaign_fingerprint": manifest["fingerprint"],
        "case_id": case["case_id"],
        "case_index": index,
        "row_id": case["row_id"],
        "scenario": sealed.SCENARIOS[1],
        "force_contract": sealed.SCENARIO_FORCE_CONTRACT[sealed.SCENARIOS[1]],
        "runtime_identity_fingerprint": canonical_fingerprint(runtime),
    }
    if metadata != expected_metadata:
        raise HVAWTOneWayRecoveryError(
            f"One-way case {index} metadata/runtime identity is not exact."
        )
    response_audit = audit_inert_response_summary(
        summary.get("sil_vorlap_response_diagnostics_summary")
    )
    coverage_audit = audit_one_way_source_coverage(
        summary.get("sil_vorlap_source_coverage")
    )
    with paths["history"].open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), None)
    if not isinstance(header, list) or not header:
        raise HVAWTOneWayRecoveryError(
            f"One-way case {index} compact history has no header."
        )
    response_columns = [name for name in header if name.startswith("response_")]
    if response_columns:
        raise HVAWTOneWayRecoveryError(
            f"One-way case {index} compact history contains response columns."
        )
    return {
        "case_index": index,
        "case_id": case["case_id"],
        "row_id": case["row_id"],
        "scenario": case["scenario"],
        "wind_speed_mps": case["wind_speed_mps"],
        "seed": case["seed"],
        "common_physics_fingerprint": case["common_physics_fingerprint"],
        "force_contract": case["force_contract"],
        "original_error_class": "HVAWTCompleteDLCError",
        "original_error_message": EXACT_FROZEN_ERROR,
        "original_classification_reason": EXACT_CLASSIFICATION_REASON,
        "prior_frozen_terminal_gates_passed": True,
        "response_sentinel_audit": response_audit,
        "source_coverage_response_audit": coverage_audit,
        "compact_history_response_column_count": 0,
        "artifacts": _artifact_identities(root=root, case=case, summary=summary),
        "explicit_unified_response_closure_active": False,
        "strict_open_loop_one_way": False,
        "kinematic_relative_velocity_feedback_retained": True,
    }


def preflight_recovery(campaign: Path) -> dict[str, object]:
    """Perform the lightweight, read-only recovery preflight."""

    state = _validate_campaign(
        campaign, deep=False, rehash_external_winds=False
    )
    root = state["campaign_root"]
    manifest = state["manifest"]
    analysis_path, _analysis = _validate_original_analysis(root, manifest)
    runtime = sealed._load_runtime(root)
    authorization = _load_continuation_authorization(root, manifest)
    u50 = sealed._load_u50_smoke_receipt(root, manifest)
    return {
        "schema": "vorlap-hvawt-one-way-audit-recovery-preflight-v1",
        "status": "ready_for_read_only_terminal_artifact_audit",
        "campaign_root": str(root),
        "campaign_fingerprint": manifest["fingerprint"],
        "original_analysis": _relative_identity(analysis_path, root),
        "original_complete_case_count": 68,
        "recoverable_candidate_count": 34,
        "recoverable_candidate_indices": list(EXPECTED_ONE_WAY_INDICES),
        "runtime": _relative_identity(root / sealed.RUNTIME_RECEIPT_NAME, root),
        "complementary_authorization": _relative_identity(
            root / "complementary_execution_authorization.json", root
        ),
        "complementary_authorization_fingerprint": authorization["fingerprint"],
        "u50_smoke": _relative_identity(root / sealed.U50_SMOKE_RECEIPT_NAME, root),
        "u50_smoke_fingerprint": u50["fingerprint"],
        "runtime_identity_fingerprint": canonical_fingerprint(runtime),
        "writes_performed": 0,
        "simulation_actions_available": False,
    }


def build_recovery_receipt(
    campaign: Path,
    *,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Audit every terminal artifact and construct a deterministic receipt."""

    state = _validate_campaign(campaign, deep=True, rehash_external_winds=True)
    root = state["campaign_root"]
    manifest = state["manifest"]
    analysis_path, analysis = _validate_original_analysis(root, manifest)
    runtime = sealed._load_runtime(root)
    authorization = _load_continuation_authorization(root, manifest)
    u50 = sealed._load_u50_smoke_receipt(root, manifest)
    original_rows = analysis["cases"]
    cases: list[dict[str, object]] = []
    recovered: list[dict[str, object]] = []
    before: dict[int, dict[str, dict[str, object]]] = {}
    scenario_counts = {scenario: 0 for scenario in sealed.SCENARIOS}
    for index, case in enumerate(manifest["cases"]):
        if index in EXPECTED_ONE_WAY_INDICES:
            one_way = _audit_one_way_case(
                root=root,
                manifest=manifest,
                case=case,
                original_analysis_row=original_rows[index],
                runtime=runtime,
            )
            before[index] = one_way["artifacts"]  # type: ignore[assignment]
            recovered.append(one_way)
            terminal: dict[str, object] = {
                "case_index": index,
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "scenario": case["scenario"],
                "status": "complete",
                "audit_disposition": "recovered_exact_inert_response_sentinel",
                "recovery_case_fingerprint": canonical_fingerprint(one_way),
            }
        else:
            try:
                receipt = sealed._terminal_case_audit(
                    root=root, manifest=manifest, case=case
                )
            except Exception as exc:
                raise HVAWTOneWayRecoveryError(
                    f"Previously complete case {index} no longer passes: {exc}"
                ) from exc
            terminal = {
                "case_index": index,
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "scenario": case["scenario"],
                "status": "complete",
                "audit_disposition": "unchanged_frozen_terminal_audit",
                "receipt": receipt,
            }
        scenario_counts[str(case["scenario"])] += 1
        cases.append(terminal)
    if (
        len(recovered) != 34
        or tuple(row["case_index"] for row in recovered) != EXPECTED_ONE_WAY_INDICES
        or len(cases) != 102
        or scenario_counts != {scenario: 34 for scenario in sealed.SCENARIOS}
    ):
        raise HVAWTOneWayRecoveryError(
            "Recovery did not produce the exact 34-by-three-path composition."
        )
    after = {
        index: _artifact_identities(root=root, case=manifest["cases"][index])
        for index in EXPECTED_ONE_WAY_INDICES
    }
    modified = sum(before[index] != after[index] for index in EXPECTED_ONE_WAY_INDICES)
    if modified:
        raise HVAWTOneWayRecoveryError(
            "A one-way terminal artifact changed during the recovery audit."
        )

    repository = Path(sealed.__file__).resolve().parents[1]
    declared_sources: dict[str, Path] = {
        "frozen_terminal_auditor": Path(sealed.__file__).resolve(),
        "recovery_model": Path(__file__).resolve(),
    }
    if recovery_sources:
        declared_sources.update(
            {name: Path(path).expanduser().resolve() for name, path in recovery_sources.items()}
        )
    source_identities = {name: _identity(path) for name, path in declared_sources.items()}
    frozen_roles = dict(manifest["inputs"]["orchestration_sources"])
    report: dict[str, object] = {
        "schema": RECOVERY_SCHEMA,
        "status": RECOVERY_STATUS,
        "campaign_fingerprint": manifest["fingerprint"],
        "campaign_manifest": _relative_identity(root / "campaign_manifest.json", root),
        "original_analysis": _relative_identity(analysis_path, root),
        "runtime": _relative_identity(root / sealed.RUNTIME_RECEIPT_NAME, root),
        "runtime_identity_fingerprint": canonical_fingerprint(runtime),
        "complementary_authorization": _relative_identity(
            root / "complementary_execution_authorization.json", root
        ),
        "complementary_authorization_fingerprint": authorization["fingerprint"],
        "u50_smoke": _relative_identity(root / sealed.U50_SMOKE_RECEIPT_NAME, root),
        "u50_smoke_fingerprint": u50["fingerprint"],
        "frozen_orchestration_source_records": frozen_roles,
        "live_source_validation_scope": ["campaign_model"],
        "unused_manifest_bound_orchestration_roles": sorted(
            set(frozen_roles) - {"campaign_model"}
        ),
        "recovery_sources": source_identities,
        "original_complete_case_count": 68,
        "recovered_case_count": 34,
        "accepted_case_count": 102,
        "recovered_case_indices": list(EXPECTED_ONE_WAY_INDICES),
        "simulation_rerun_count": 0,
        "modified_artifact_count": modified,
        "cases": cases,
        "recovered_cases": recovered,
        "comparator_semantics": {
            "one_way_label": "closure-off legacy spectral comparator",
            "explicit_unified_response_closure_active": False,
            "strict_open_loop_one_way": False,
            "kinematic_relative_velocity_feedback_retained": True,
            "raw_to_one_way": "full imposed legacy spectral forcing",
            "one_way_to_complete": (
                "change in force composition and partition including "
                "response-correlated work; not isolated additive negative damping"
            ),
            "raw_to_complete": "total hybrid-model effect",
        },
        "claim_boundary": (
            "Artifact recovery of completed closure-off legacy spectral comparator; "
            "no simulation rerun and no new validation. The matched H-VAWT result "
            "remains an engineering application, not experimental validation, "
            "fatigue life, or certification evidence."
        ),
        "repository": str(repository),
    }
    report["fingerprint"] = canonical_fingerprint(report)
    return report


@contextmanager
def _locked_one_way_cases(root: Path, manifest: Mapping[str, Any]) -> Iterator[None]:
    """Block execution while reading or publishing recovery evidence."""

    lock_root = root / ".case_locks"
    with ExitStack() as stack:
        for order, index in enumerate(EXPECTED_ONE_WAY_INDICES):
            case = manifest["cases"][index]
            path = lock_root / f"{case['case_id']}.lock"
            if not path.is_file():
                raise HVAWTOneWayRecoveryError(
                    f"Missing completed-case execution lock: {path}"
                )
            stream = stack.enter_context(path.open("r", encoding="utf-8"))
            operation = fcntl.LOCK_EX if order == 0 else fcntl.LOCK_SH
            try:
                fcntl.flock(stream.fileno(), operation | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HVAWTOneWayRecoveryError(
                    f"One-way case {index} has an active writer; recovery stopped."
                ) from exc
        yield


def write_recovery_receipt(
    campaign: Path,
    *,
    output: Path | None = None,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Create or exactly reuse the sole additive recovery receipt."""

    initial = _validate_campaign(campaign, deep=False, rehash_external_winds=False)
    root = initial["campaign_root"]
    target = (
        root / RECOVERY_RECEIPT_NAME
        if output is None
        else Path(output).expanduser().resolve()
    )
    if target.parent != root or target.name != RECOVERY_RECEIPT_NAME:
        raise HVAWTOneWayRecoveryError(
            f"Recovery receipt must be {root / RECOVERY_RECEIPT_NAME}."
        )
    with _locked_one_way_cases(root, initial["manifest"]):
        report = build_recovery_receipt(
            root, recovery_sources=recovery_sources
        )
        if target.exists():
            existing = _load_json(target, label="existing one-way recovery receipt")
            if existing != report:
                raise HVAWTOneWayRecoveryError(
                    "Existing recovery receipt differs from the live read-only audit."
                )
            return existing
        atomic_write_json(target, report)
        recorded = _load_json(target, label="written one-way recovery receipt")
        if recorded != report:
            raise HVAWTOneWayRecoveryError(
                "Written recovery receipt does not match its audited payload."
            )
        return recorded


def load_recovery_receipt(
    campaign: Path,
    *,
    receipt_path: Path | None = None,
    verify_live_artifacts: bool = True,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Load and optionally reproduce the recovery receipt from live artifacts."""

    initial = _validate_campaign(campaign, deep=False, rehash_external_winds=False)
    root = initial["campaign_root"]
    path = (
        root / RECOVERY_RECEIPT_NAME
        if receipt_path is None
        else Path(receipt_path).expanduser().resolve()
    )
    if path != root / RECOVERY_RECEIPT_NAME:
        raise HVAWTOneWayRecoveryError(
            "Recovery receipt path differs from the campaign-root contract."
        )
    receipt = _load_json(path, label="one-way recovery receipt")
    _validate_fingerprint(receipt, label="One-way recovery receipt")
    if (
        receipt.get("schema") != RECOVERY_SCHEMA
        or receipt.get("status") != RECOVERY_STATUS
        or receipt.get("campaign_fingerprint") != initial["manifest"]["fingerprint"]
        or receipt.get("original_complete_case_count") != 68
        or receipt.get("recovered_case_count") != 34
        or receipt.get("accepted_case_count") != 102
        or tuple(receipt.get("recovered_case_indices", ()))
        != EXPECTED_ONE_WAY_INDICES
        or receipt.get("simulation_rerun_count") != 0
        or receipt.get("modified_artifact_count") != 0
    ):
        raise HVAWTOneWayRecoveryError("One-way recovery receipt contract is invalid.")
    for role in (
        "campaign_manifest",
        "original_analysis",
        "runtime",
        "complementary_authorization",
        "u50_smoke",
    ):
        _verify_relative_identity(root, receipt.get(role), label=role)
    recorded_sources = receipt.get("recovery_sources")
    if not isinstance(recorded_sources, Mapping) or not recorded_sources:
        raise HVAWTOneWayRecoveryError(
            "One-way recovery receipt does not bind its live recovery sources."
        )
    for role, record in recorded_sources.items():
        if not isinstance(role, str) or not role:
            raise HVAWTOneWayRecoveryError(
                "One-way recovery receipt contains an invalid source role."
            )
        _verify_absolute_identity(record, label=f"recovery source {role}")
    if verify_live_artifacts:
        with _locked_one_way_cases(root, initial["manifest"]):
            reproduced = build_recovery_receipt(
                root, recovery_sources=recovery_sources
            )
        if reproduced != receipt:
            raise HVAWTOneWayRecoveryError(
                "Live artifacts no longer reproduce the recovery receipt."
            )
    return receipt


def build_recovered_analysis(
    campaign: Path,
    *,
    receipt_path: Path | None = None,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Construct a versioned 102-case analysis without editing analysis.json."""

    initial = _validate_campaign(campaign, deep=False, rehash_external_winds=False)
    root = initial["campaign_root"]
    manifest = initial["manifest"]
    receipt = load_recovery_receipt(
        root,
        receipt_path=receipt_path,
        verify_live_artifacts=True,
        recovery_sources=recovery_sources,
    )
    cases = receipt.get("cases")
    if not isinstance(cases, list) or len(cases) != 102:
        raise HVAWTOneWayRecoveryError(
            "Recovery receipt does not contain 102 terminal case records."
        )
    analysis_cases: list[dict[str, object]] = []
    for index, (record, case) in enumerate(
        zip(cases, manifest["cases"], strict=True)
    ):
        if (
            not isinstance(record, Mapping)
            or record.get("case_index") != index
            or record.get("case_id") != case["case_id"]
            or record.get("row_id") != case["row_id"]
            or record.get("scenario") != case["scenario"]
            or record.get("status") != "complete"
        ):
            raise HVAWTOneWayRecoveryError(
                f"Recovery terminal record {index} differs from the manifest."
            )
        analysis_cases.append(
            {
                "case_index": index,
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "scenario": case["scenario"],
                "status": "complete",
                "audit_disposition": record["audit_disposition"],
            }
        )
    recovery_path = root / RECOVERY_RECEIPT_NAME
    report: dict[str, object] = {
        "schema": RECOVERED_ANALYSIS_SCHEMA,
        "status": RECOVERED_ANALYSIS_STATUS,
        "campaign_fingerprint": manifest["fingerprint"],
        "expected_case_count": 102,
        "complete_case_count": 102,
        "all_three_scenarios_complete": True,
        "original_analysis": receipt["original_analysis"],
        "recovery_receipt": _relative_identity(recovery_path, root),
        "recovered_case_count": 34,
        "recovered_case_indices": list(EXPECTED_ONE_WAY_INDICES),
        "cases": analysis_cases,
        "claim_boundary": receipt["claim_boundary"],
    }
    report["fingerprint"] = canonical_fingerprint(report)
    return report


def write_recovered_analysis(
    campaign: Path,
    *,
    output: Path | None = None,
    receipt_path: Path | None = None,
    recovery_sources: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    initial = _validate_campaign(campaign, deep=False, rehash_external_winds=False)
    root = initial["campaign_root"]
    target = (
        root / RECOVERED_ANALYSIS_NAME
        if output is None
        else Path(output).expanduser().resolve()
    )
    if target.parent != root or target.name != RECOVERED_ANALYSIS_NAME:
        raise HVAWTOneWayRecoveryError(
            f"Recovered analysis must be {root / RECOVERED_ANALYSIS_NAME}."
        )
    report = build_recovered_analysis(
        root,
        receipt_path=receipt_path,
        recovery_sources=recovery_sources,
    )
    if target.exists():
        existing = _load_json(target, label="existing recovered analysis")
        if existing != report:
            raise HVAWTOneWayRecoveryError(
                "Existing recovered analysis differs from the live recovery evidence."
            )
        return existing
    atomic_write_json(target, report)
    return report


__all__ = [
    "EXACT_CLASSIFICATION_REASON",
    "EXACT_FROZEN_ERROR",
    "EXPECTED_ONE_WAY_INDICES",
    "HVAWTOneWayRecoveryError",
    "RECOVERED_ANALYSIS_NAME",
    "RECOVERED_ANALYSIS_SCHEMA",
    "RECOVERY_RECEIPT_NAME",
    "RECOVERY_SCHEMA",
    "audit_inert_response_summary",
    "audit_one_way_source_coverage",
    "build_recovered_analysis",
    "build_recovery_receipt",
    "load_recovery_receipt",
    "preflight_recovery",
    "write_recovered_analysis",
    "write_recovery_receipt",
]
