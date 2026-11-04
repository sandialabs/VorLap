"""Sealed targeted numerical-sensitivity campaign for the matched H-VAWT study.

The accepted 102-case campaign is immutable.  This module copies only the
preselected case inputs into a separate root, applies an explicit single-axis
numerical or force-partition change, and binds every source and result by hash.
It never writes inside the accepted campaign.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from run_headless.postprocess_parked_demo import (
    rainflow_damage,
    read_qblade_results,
)

from .dlc_manifest import (
    sha256_payload,
    validate_wind_record,
)
from .hvawt_complete_dlc_campaign import (
    APPROVED_MANIFEST_CANONICAL_SHA256,
    APPROVED_MANIFEST_FILE_SHA256,
    BRIDGE_LIBRARY_SHA256,
    QBLADE_LIBRARY_SHA256,
    RUNTIME_RECEIPT_NAME,
    SOURCE_FREQUENCY_CAP_HZ,
    STORE_FROM_S,
    DURATION_S,
    TIMESTEP_S,
    TOTAL_STEPS,
    build_runtime_identity as build_base_runtime_identity,
)
from .hvawt_response_pilot import (
    _audit_qblade_fatigue_result,
    _history_path_from_summary,
    analyze_history,
)
from .qblade_files import read_qblade_keyword, replace_qblade_keyword_file
from .qblade_run_support import (
    atomic_write_json,
    canonical_fingerprint,
    qblade_runtime_asset_fingerprint,
    require_zero_native_callback_evidence,
)
from .qblade_source_coverage import (
    HVAWT_V8_COMPONENT_NODE_COUNTS,
    validate_complete_source_coverage,
    validate_complete_velocity_diagnostics,
)


CAMPAIGN_SCHEMA = "vorlap-hvawt-targeted-sensitivity-campaign-v1"
ANALYSIS_SCHEMA = "vorlap-hvawt-targeted-sensitivity-analysis-v1"
PROMOTION_SCHEMA = "vorlap-hvawt-targeted-sensitivity-promotion-v1"
AUTHORIZATION_SCHEMA = "vorlap-hvawt-targeted-sensitivity-authorization-v1"
SMOKE_SCHEMA = "vorlap-hvawt-targeted-sensitivity-smoke-v1"
BASE_CAMPAIGN_SCHEMA = "vorlap-hvawt-complete-dlc-campaign-v1"
BASE_CAMPAIGN_FINGERPRINT = (
    "10a397bfcfdf93499e3a0e7417db1814f7e9e39e44982fd7066b83343cd25ec1"
)
SELECTION_SCHEMA = "vorlap-hvawt-sensitivity-selection-v1"
SELECTION_STATUS = "two_physical_rows_frozen_before_sensitivity_generation"
SELECTION_FINGERPRINT = (
    "55a0622ee828175667a63f8379115b4ace317f36585834d9c8be4cdf97252c1e"
)
SELECTION_NAME = "hvawt_sensitivity_selection_v1.json"
RECOVERY_NAME = "one_way_audit_recovery_receipt.json"
RECOVERED_ANALYSIS_NAME = "analysis_recovered_v3.json"
RECOVERY_FINGERPRINT = (
    "98046e00445d8c53638ccd2139aa47c9f607245e7fffd38e84c08d0351b184e5"
)
RECOVERED_ANALYSIS_FINGERPRINT = (
    "39a4cdf1b247841be48ddfe8d006f90ecce17ff1be85c93569746636695fe0da"
)
BASE_PROMOTION_FILE_SHA256 = (
    "884a933dc1d6ce1f35e890206cda58d1af6c7be039f3f06f6be26ded2526db4f"
)
BASE_PROMOTION_FINGERPRINT = (
    "0ff5f6b1bf1ad165cb3e928467df8004ad7736acf614ff18eed0285e4216dd1a"
)
BASE_CAMPAIGN_PAPER_HELPER_SHA256 = (
    "fcb4b90d5408846d757f58133bf65728bc96ca87ccec8fe8a5df1175a7d46877"
)
BASE_POSTCAMPAIGN_PAPER_HELPER_SHA256 = (
    "b03326383e4e1ea772941b6f2634a6b623d7764bb9ff7d1d1d24946dd2ecda60"
)
PROMOTION_RECEIPT_RELATIVE = Path(
    "paper_export_recovered_v2/hvawt_three_path_promotion_receipt.json"
)
MODAL_QUALIFICATION_RELATIVE = Path(
    "run_headless/config/modal_timestep_qualification.json"
)
PROTOCOL_RELATIVE = Path(
    "run_headless/config/hvawt_sensitivity_protocol_v1.json"
)
MODAL_QUALIFICATION_FILE_SHA256 = (
    "456c0b83206bb05ceb038e8591696b9ff01a309268f570abb81b11d47ab6b29f"
)
MODAL_QUALIFICATION_FINGERPRINT = (
    "2050f544e788250f899621a0cea9ee1de28502341dfe544925a85324f14ff944"
)
MANIFEST_NAME = "sensitivity_campaign_manifest.json"
SMOKE_RECEIPT_NAME = "sensitivity_smoke_receipt.json"
AUTHORIZATION_NAME = "sensitivity_execution_authorization.json"
ANALYSIS_NAME = "sensitivity_analysis_v1.json"
PAPER_EXPORT_NAME = "paper_export_sensitivity_v1"
PROMOTION_RECEIPT_NAME = "hvawt_sensitivity_promotion_receipt.json"

NOMINAL_CAP_HZ = SOURCE_FREQUENCY_CAP_HZ
LOWER_CAP_HZ = 0.25 / TIMESTEP_S
HALF_TIMESTEP_S = TIMESTEP_S / 2.0
HALF_TIMESTEP_STEPS = TOTAL_STEPS * 2
NOMINAL_RETAINED_ROWS = int(round((DURATION_S - STORE_FROM_S) / TIMESTEP_S))
HALF_RETAINED_ROWS = int(round((DURATION_S - STORE_FROM_S) / HALF_TIMESTEP_S))
HIGHEST_REVIEWED_MODE_HZ = 4.572593676110874
MODE_FREQUENCIES_HZ = {6: 2.5115545017693828, 12: 4.572593676110874}
WOHLER_EXPONENT = 10.0
EQUIVALENT_CYCLES = 600.0
LOAD_RELATIVE_LIMIT = 0.10
MODE_RELATIVE_LIMIT = 0.05
MODE_ABSOLUTE_BAND_LIMIT = 0.10
MODE_ACTIVE_DAMPING_RATIO_THRESHOLD = 1.0
FINAL_120_RELATIVE_LIMIT = 0.05

SELECTED_ROWS = (
    {
        "row_id": "dlc6p4_U35p00ms_s4433458",
        "seed": 4_433_458,
        "channel_axis": "Y",
        "complete_case_index": 98,
        "selection_role": "governing_complete_weighted_del_power_contributor",
    },
    {
        "row_id": "dlc6p4_U35p00ms_s4433456",
        "seed": 4_433_456,
        "channel_axis": "X",
        "complete_case_index": 92,
        "selection_role": "largest_material_complete_to_one_way_composition_ratio",
    },
)

RAW_CONTRACT = {
    "coupling_mode": "none",
    "legacy_vorlap_force_scale": 0.0,
    "residual_source_scale": 0.0,
    "response_force_scale": 0.0,
}
ONE_WAY_CONTRACT = {
    "coupling_mode": "sil-vorlap",
    "legacy_vorlap_force_scale": 1.0,
    "residual_source_scale": 0.0,
    "response_force_scale": 0.0,
}
COMPLETE_CONTRACT = {
    "coupling_mode": "sil-vorlap",
    "legacy_vorlap_force_scale": 0.0,
    "residual_source_scale": 1.0,
    "response_force_scale": 1.0,
}
RESPONSE_OFF_CONTRACT = {
    **COMPLETE_CONTRACT,
    "response_force_scale": 0.0,
}

VARIANTS = (
    {
        "name": "complete_response_off",
        "source_scenario": "complete_hybrid_v8",
        "timestep_s": TIMESTEP_S,
        "total_steps": TOTAL_STEPS,
        "retained_rows": NOMINAL_RETAINED_ROWS,
        "source_frequency_cap_hz": NOMINAL_CAP_HZ,
        "force_contract": RESPONSE_OFF_CONTRACT,
        "role": "response_ablation_predeclared_diagnostic",
    },
    {
        "name": "complete_half_timestep",
        "source_scenario": "complete_hybrid_v8",
        "timestep_s": HALF_TIMESTEP_S,
        "total_steps": HALF_TIMESTEP_STEPS,
        "retained_rows": HALF_RETAINED_ROWS,
        "source_frequency_cap_hz": NOMINAL_CAP_HZ,
        "force_contract": COMPLETE_CONTRACT,
        "role": "half_timestep_same_physical_bandwidth",
    },
    {
        "name": "complete_lower_cap",
        "source_scenario": "complete_hybrid_v8",
        "timestep_s": TIMESTEP_S,
        "total_steps": TOTAL_STEPS,
        "retained_rows": NOMINAL_RETAINED_ROWS,
        "source_frequency_cap_hz": LOWER_CAP_HZ,
        "force_contract": COMPLETE_CONTRACT,
        "role": "lower_cap_four_samples_per_shortest_retained_cycle",
    },
)

ORCHESTRATION_SOURCE_RELATIVE = {
    "campaign_model": Path("vorlap/hvawt_sensitivity_campaign.py"),
    "campaign_cli": Path("run_headless/hvawt_sensitivity_campaign.py"),
    "preparation_submission": Path(
        "run_headless/submit_hvawt_sensitivity_prepare.sj"
    ),
    "execution_submission": Path(
        "run_headless/submit_hvawt_sensitivity_flight.sj"
    ),
    "analysis_submission": Path(
        "run_headless/submit_hvawt_sensitivity_analysis.sj"
    ),
    "qblade_runner": Path("run_headless/run_qblade_sim.py"),
    "postprocessor": Path("run_headless/postprocess_parked_demo.py"),
    "sensitivity_protocol": PROTOCOL_RELATIVE,
}


class HVAWTSensitivityError(RuntimeError):
    """Raised when targeted sensitivity provenance or physics changes."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise HVAWTSensitivityError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HVAWTSensitivityError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise HVAWTSensitivityError(f"{label} must be a JSON object.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _relative_identity(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _verify_relative_identity(
    root: Path, record: object, *, label: str
) -> Path:
    if not isinstance(record, Mapping):
        raise HVAWTSensitivityError(f"Missing identity for {label}.")
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise HVAWTSensitivityError(f"Invalid path identity for {label}.")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise HVAWTSensitivityError(f"{label} escapes its root.") from exc
    if (
        not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise HVAWTSensitivityError(f"Identity changed for {label}.")
    return path


def _validate_fingerprint(payload: Mapping[str, Any], *, label: str) -> None:
    content = dict(payload)
    observed = content.pop("fingerprint", None)
    if observed != canonical_fingerprint(content):
        raise HVAWTSensitivityError(f"{label} fingerprint is invalid.")


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _scenario_cases(
    manifest: Mapping[str, Any], row_id: str
) -> dict[str, Mapping[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise HVAWTSensitivityError("Base campaign case list is invalid.")
    selected = {
        str(case.get("scenario")): case
        for case in cases
        if isinstance(case, Mapping) and case.get("row_id") == row_id
    }
    if set(selected) != {
        "raw_qblade",
        "one_way_spectral_vorlap",
        "complete_hybrid_v8",
    }:
        raise HVAWTSensitivityError(f"Base trio is incomplete for {row_id}.")
    return selected


def validate_base_preflight(base_campaign_root: Path) -> dict[str, Any]:
    """Perform a login-safe, read-only check without hashing large results."""

    root = Path(base_campaign_root).expanduser().resolve()
    manifest = _load_json(root / "campaign_manifest.json", label="base manifest")
    _validate_fingerprint(manifest, label="Base campaign")
    if (
        manifest.get("schema") != BASE_CAMPAIGN_SCHEMA
        or manifest.get("fingerprint") != BASE_CAMPAIGN_FINGERPRINT
        or manifest.get("case_count") != 102
        or manifest.get("row_count") != 34
    ):
        raise HVAWTSensitivityError("Base campaign is not the accepted 102-case plan.")
    selection = _load_json(root / SELECTION_NAME, label="sensitivity selection")
    _validate_fingerprint(selection, label="Sensitivity selection")
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("status") != SELECTION_STATUS
        or selection.get("fingerprint") != SELECTION_FINGERPRINT
        or selection.get("campaign_fingerprint") != BASE_CAMPAIGN_FINGERPRINT
        or selection.get("case_generation_performed") is not False
        or selection.get("simulation_execution_performed") is not False
    ):
        raise HVAWTSensitivityError("Sensitivity selection receipt is not accepted.")
    rows = selection.get("selected_physical_rows")
    if not isinstance(rows, list) or len(rows) != len(SELECTED_ROWS):
        raise HVAWTSensitivityError("Sensitivity selection must contain two rows.")
    for expected, observed in zip(SELECTED_ROWS, rows, strict=True):
        channel = observed.get("channel") if isinstance(observed, Mapping) else None
        if (
            not isinstance(observed, Mapping)
            or observed.get("row_id") != expected["row_id"]
            or observed.get("seed") != expected["seed"]
            or observed.get("complete_case_index") != expected["complete_case_index"]
            or observed.get("selection_role") != expected["selection_role"]
            or not isinstance(channel, Mapping)
            or channel.get("axis") != expected["channel_axis"]
            or float(channel.get("position", math.nan)) != 0.23
        ):
            raise HVAWTSensitivityError("Frozen sensitivity row mapping changed.")
        trio = _scenario_cases(manifest, str(expected["row_id"]))
        complete = trio["complete_hybrid_v8"]
        if (
            complete.get("index") != expected["complete_case_index"]
            or complete.get("seed") != expected["seed"]
            or float(complete.get("wind_speed_mps", math.nan)) != 35.0
        ):
            raise HVAWTSensitivityError("Selected base complete case changed.")
        for scenario, case in trio.items():
            case_dir = root / str(case.get("case_dir", ""))
            result_dir = root / str(case.get("result_dir", ""))
            sim = root / str(case.get("sim", ""))
            summary = result_dir / f"{sim.stem}_run_summary.json"
            result = result_dir / f"{case.get('result_name')}.txt"
            if not case_dir.is_dir() or not sim.is_file():
                raise HVAWTSensitivityError(
                    f"Missing selected base case input: {expected['row_id']}/{scenario}"
                )
            if not summary.is_file() or not result.is_file():
                raise HVAWTSensitivityError(
                    f"Missing selected base result: {expected['row_id']}/{scenario}"
                )
    recovery = _load_json(root / RECOVERY_NAME, label="recovery receipt")
    analysis = _load_json(
        root / RECOVERED_ANALYSIS_NAME, label="recovered analysis"
    )
    promotion_path = root / PROMOTION_RECEIPT_RELATIVE
    promotion = _load_json(promotion_path, label="paper promotion receipt")
    _validate_fingerprint(recovery, label="Recovery receipt")
    _validate_fingerprint(analysis, label="Recovered analysis")
    _validate_fingerprint(promotion, label="Base paper promotion receipt")
    source_comparison = promotion.get("source_comparison")
    manifest_helper = (
        source_comparison.get("manifest_record")
        if isinstance(source_comparison, Mapping)
        else None
    )
    postcampaign_helper = (
        source_comparison.get("live_postcampaign_helper")
        if isinstance(source_comparison, Mapping)
        else None
    )
    if (
        recovery.get("fingerprint") != RECOVERY_FINGERPRINT
        or recovery.get("status")
        != "accepted_completed_one_way_artifacts_without_rerun"
        or recovery.get("recovered_case_count") != 34
        or recovery.get("simulation_rerun_count") != 0
        or recovery.get("modified_artifact_count") != 0
        or analysis.get("fingerprint") != RECOVERED_ANALYSIS_FINGERPRINT
        or analysis.get("status") != "complete_ready_for_load_postprocess"
        or analysis.get("complete_case_count") != 102
        or analysis.get("expected_case_count") != 102
        or _sha256(promotion_path) != BASE_PROMOTION_FILE_SHA256
        or promotion.get("fingerprint") != BASE_PROMOTION_FINGERPRINT
        or promotion.get("status")
        != "promoted_from_recovered_102_complete_matched_cases"
        or not isinstance(source_comparison, Mapping)
        or source_comparison.get("sha256_matches_manifest") is not False
        or source_comparison.get("size_matches_manifest") is not False
        or not isinstance(manifest_helper, Mapping)
        or manifest_helper.get("sha256")
        != BASE_CAMPAIGN_PAPER_HELPER_SHA256
        or not isinstance(postcampaign_helper, Mapping)
        or postcampaign_helper.get("sha256")
        != BASE_POSTCAMPAIGN_PAPER_HELPER_SHA256
    ):
        raise HVAWTSensitivityError(
            "Accepted recovery, analysis, or promotion evidence changed."
        )
    return {
        "base_campaign_root": root,
        "manifest": manifest,
        "selection": selection,
        "recovery": recovery,
    }


def validate_selected_base_inputs(
    base: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Deep-check only the immutable inputs used by the six new cases.

    The accepted base campaign records a campaign-era paper formatter.  Its
    later, receipt-bound replacement is intentionally not simulation physics,
    so rerunning the original whole-campaign validator would incorrectly reject
    the accepted 102-case result.  This check instead hashes every copied base
    input needed to interpret the selected cases and rehashes the two external
    TurbSim records actually used by this additive campaign.
    """

    root = Path(base["base_campaign_root"]).resolve()
    manifest = base["manifest"]
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise HVAWTSensitivityError("Base campaign input identities are missing.")
    verified_inputs: dict[str, Path] = {}
    for role in (
        "approved_manifest",
        "campaign_spec",
        "response_model",
        "rayleigh_selection",
        "result_filter",
    ):
        verified_inputs[role] = _verify_relative_identity(
            root,
            inputs.get(role),
            label=f"selected base input {role}",
        )
    approved_path = verified_inputs["approved_manifest"]
    if _sha256(approved_path) != APPROVED_MANIFEST_FILE_SHA256:
        raise HVAWTSensitivityError("Copied approved-manifest bytes changed.")
    approved = _load_json(approved_path, label="copied approved DLC manifest")
    if (
        approved.get("status") != "approved"
        or approved.get("row_count") != 34
        or approved.get("manifest_sha256")
        != APPROVED_MANIFEST_CANONICAL_SHA256
        or not isinstance(approved.get("rows"), list)
        or len(approved["rows"]) != 34
    ):
        raise HVAWTSensitivityError("Copied approved manifest identity changed.")
    approved_rows = {
        str(row["row_id"]): row
        for row in approved.get("rows", [])
        if isinstance(row, dict) and row.get("row_id")
    }
    wind_identities = manifest.get("wind_records")
    if not isinstance(wind_identities, Mapping):
        raise HVAWTSensitivityError("Base wind-record identities are missing.")
    selected_records: dict[str, dict[str, Any]] = {}
    for selected in SELECTED_ROWS:
        row_id = str(selected["row_id"])
        source_row = approved_rows.get(row_id)
        if not isinstance(source_row, dict):
            raise HVAWTSensitivityError(
                f"Selected row {row_id} is absent from the approved DLC matrix."
            )
        record_path = _verify_relative_identity(
            root,
            wind_identities.get(row_id),
            label=f"selected base wind record {row_id}",
        )
        copied_record = _load_json(
            record_path, label=f"selected copied wind record {row_id}"
        )
        payload = dict(copied_record)
        archived_digest = payload.pop("record_sha256", None)
        if (
            copied_record.get("schema_version") != 1
            or copied_record.get("row_id") != row_id
            or copied_record.get("manifest_sha256")
            != APPROVED_MANIFEST_CANONICAL_SHA256
            or archived_digest != sha256_payload(payload)
            or copied_record.get("wind_file") != source_row.get("wind_file")
            or copied_record.get("seed") != source_row.get("seed")
            or copied_record.get("dlc") != source_row.get("dlc")
            or copied_record.get("wind_speed_mps")
            != source_row.get("wind_speed_mps")
        ):
            raise HVAWTSensitivityError(
                f"Copied wind record contract changed for {row_id}."
            )
        try:
            observed_record = validate_wind_record(source_row, approved)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise HVAWTSensitivityError(
                f"External wind record changed for selected row {row_id}."
            ) from exc
        if copied_record != observed_record:
            raise HVAWTSensitivityError(
                f"External wind record changed for selected row {row_id}."
            )
        selected_records[row_id] = copied_record
    if set(selected_records) != {str(row["row_id"]) for row in SELECTED_ROWS}:
        raise HVAWTSensitivityError("Selected wind-record set is incomplete.")
    return selected_records


def _case_result_paths(
    root: Path, case: Mapping[str, Any]
) -> dict[str, Path]:
    result_dir = root / str(case["result_dir"])
    sim_path = root / str(case["sim"])
    values = {
        "result": result_dir / f"{case['result_name']}.txt",
        "summary": result_dir / f"{sim_path.stem}_run_summary.json",
        "metadata": result_dir / "case_metadata.json",
    }
    if case["scenario"] != "raw_qblade":
        summary = _load_json(values["summary"], label=f"base summary {case['case_id']}")
        values["history"] = _history_path_from_summary(summary, result_dir)
        if not values["history"].is_file():
            local_history = result_dir / values["history"].name
            if local_history.is_file():
                values["history"] = local_history
    return values


def _accepted_selected_artifact_records(
    recovery: Mapping[str, Any],
) -> dict[int, dict[str, Mapping[str, Any]]]:
    """Extract immutable artifact identities for the six selected nominal cases."""

    selected_indices = {
        int(index)
        for row in SELECTED_ROWS
        for index in (
            int(row["complete_case_index"]) - 2,
            int(row["complete_case_index"]) - 1,
            int(row["complete_case_index"]),
        )
    }
    accepted_cases = recovery.get("cases")
    recovered_cases = recovery.get("recovered_cases")
    if not isinstance(accepted_cases, list) or not isinstance(recovered_cases, list):
        raise HVAWTSensitivityError("Recovery artifact records are missing.")
    accepted_by_index = {
        int(row["case_index"]): row
        for row in accepted_cases
        if isinstance(row, Mapping) and row.get("case_index") in selected_indices
    }
    recovered_by_index = {
        int(row["case_index"]): row
        for row in recovered_cases
        if isinstance(row, Mapping) and row.get("case_index") in selected_indices
    }
    if set(accepted_by_index) != selected_indices:
        raise HVAWTSensitivityError(
            "Recovery receipt does not cover all selected nominal cases."
        )
    records: dict[int, dict[str, Mapping[str, Any]]] = {}
    for case_index in sorted(selected_indices):
        accepted = accepted_by_index[case_index]
        scenario = accepted.get("scenario")
        if scenario == "one_way_spectral_vorlap":
            recovered = recovered_by_index.get(case_index)
            artifacts = recovered.get("artifacts") if recovered else None
            if (
                not isinstance(recovered, Mapping)
                or recovered.get("case_id") != accepted.get("case_id")
                or recovered.get("row_id") != accepted.get("row_id")
                or recovered.get("scenario") != scenario
                or not isinstance(artifacts, Mapping)
            ):
                raise HVAWTSensitivityError(
                    "Selected one-way recovery artifact record is invalid."
                )
            source = artifacts
        else:
            receipt = accepted.get("receipt")
            if not isinstance(receipt, Mapping):
                raise HVAWTSensitivityError(
                    "Selected raw/complete accepted artifact record is invalid."
                )
            source = dict(receipt)
            if scenario == "complete_hybrid_v8":
                history_audit = receipt.get("coupled_history_audit")
                history = (
                    history_audit.get("history")
                    if isinstance(history_audit, Mapping)
                    else None
                )
                if not isinstance(history, Mapping):
                    raise HVAWTSensitivityError(
                        "Selected complete history identity is missing."
                    )
                source["history"] = history
        required_roles = {"result", "summary", "metadata"}
        if scenario != "raw_qblade":
            required_roles.add("history")
        selected_records = {
            role: source[role]
            for role in required_roles
            if isinstance(source.get(role), Mapping)
        }
        if set(selected_records) != required_roles:
            raise HVAWTSensitivityError(
                "Selected accepted artifact identity set is incomplete."
            )
        records[case_index] = selected_records
    return records


def validate_selected_nominal_evidence(
    base: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, object]]]:
    """Hash accepted selected artifacts and reproduce their frozen channel DELs."""

    base_root = Path(base["base_campaign_root"]).resolve()
    manifest = base["manifest"]
    selection = base["selection"]
    recovery = base["recovery"]
    accepted_records = _accepted_selected_artifact_records(recovery)
    selection_rows = {
        str(row["row_id"]): row
        for row in selection["selected_physical_rows"]
        if isinstance(row, Mapping)
    }
    baseline_results: dict[str, dict[str, dict[str, object]]] = {}
    for selected in SELECTED_ROWS:
        row_id = str(selected["row_id"])
        selection_row = selection_rows.get(row_id)
        if not isinstance(selection_row, Mapping):
            raise HVAWTSensitivityError("Selected nominal DEL record is missing.")
        trio = _scenario_cases(manifest, row_id)
        baseline_results[row_id] = {}
        expected_dels = selection_row.get("absolute_del_Nm")
        expected_indices = selection_row.get("case_indices")
        expected_ids = selection_row.get("case_ids")
        if not all(
            isinstance(value, Mapping)
            for value in (expected_dels, expected_indices, expected_ids)
        ):
            raise HVAWTSensitivityError("Selected nominal mapping is incomplete.")
        for scenario, del_key in (
            ("raw_qblade", "raw"),
            ("one_way_spectral_vorlap", "one_way"),
            ("complete_hybrid_v8", "complete"),
        ):
            case = trio[scenario]
            case_index = int(case["index"])
            if (
                expected_indices.get(scenario) != case_index
                or expected_ids.get(scenario) != case.get("case_id")
            ):
                raise HVAWTSensitivityError(
                    "Selection case mapping changed before sensitivity sealing."
                )
            paths = _case_result_paths(base_root, case)
            records = accepted_records.get(case_index)
            if not isinstance(records, Mapping) or set(paths) != set(records):
                raise HVAWTSensitivityError(
                    "Selected result path roles differ from accepted recovery."
                )
            verified: dict[str, dict[str, object]] = {}
            for role, path in paths.items():
                accepted_path = _verify_relative_identity(
                    base_root,
                    records[role],
                    label=f"accepted selected {row_id}/{scenario}/{role}",
                )
                if accepted_path != path.resolve():
                    raise HVAWTSensitivityError(
                        "Selected artifact path differs from accepted recovery."
                    )
                verified[role] = dict(records[role])
            channel = (
                f"{selected['channel_axis']}_l~Mom.~BLD_1~pos~0.230~[Nm]"
            )
            metric = _load_metric(
                path=paths["result"],
                identity_root=base_root,
                channel=channel,
                timestep_s=TIMESTEP_S,
                retained_rows=NOMINAL_RETAINED_ROWS,
                label=f"accepted_{scenario}",
                row_id=row_id,
                source_kind="accepted_v3_reproduced_before_sealing",
            )
            if not math.isclose(
                float(metric["del_Nm"]),
                float(expected_dels[del_key]),
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            ):
                raise HVAWTSensitivityError(
                    "Selected nominal DEL does not reproduce the frozen receipt."
                )
            verified["frozen_selected_channel_del_Nm"] = {
                "value": float(expected_dels[del_key]),
                "channel": channel,
                "wohler_exponent": WOHLER_EXPONENT,
                "equivalent_cycles": EQUIVALENT_CYCLES,
            }
            baseline_results[row_id][scenario] = verified
    return baseline_results


def _patch_runtime_config(
    path: Path, *, variant: Mapping[str, Any]
) -> None:
    config = _load_json(path, label="copied VorLap runtime config")
    source_scenario = str(variant["source_scenario"])
    if not math.isclose(
        float(config.get("sample_step", math.nan)),
        TIMESTEP_S,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise HVAWTSensitivityError("Source runtime sample step is not nominal.")
    if not math.isclose(
        float(config.get("source_frequency_cap_hz", math.nan)),
        NOMINAL_CAP_HZ,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise HVAWTSensitivityError("Source runtime frequency cap is not nominal.")
    config["sample_step"] = float(variant["timestep_s"])
    config["source_frequency_cap_hz"] = float(variant["source_frequency_cap_hz"])
    # Keep staged model artifacts relocatable and prevent any read through the
    # accepted v3 case's absolute Control directory.
    config["source_parameter_dir"] = "."
    hvawt = config.get("hvawt_response_model")
    if source_scenario == "one_way_spectral_vorlap":
        if hvawt is not None or float(config.get("force_scale", math.nan)) != 1.0:
            raise HVAWTSensitivityError("Source one-way runtime contract changed.")
    else:
        engineering = (
            hvawt.get("engineering_lockin")
            if isinstance(hvawt, dict)
            else None
        )
        if (
            float(config.get("force_scale", math.nan)) != 0.0
            or not isinstance(engineering, dict)
            or engineering.get("residual_source_scale") != 1.0
            or engineering.get("response_force_scale") != 1.0
            or engineering.get("velocity_basis")
            != "local_rigid_body_relative_unscaled"
        ):
            raise HVAWTSensitivityError("Source complete runtime contract changed.")
        engineering["response_force_scale"] = float(
            variant["force_contract"]["response_force_scale"]
        )
    atomic_write_json(path, config)


def _orchestration_identities(repository: Path) -> dict[str, dict[str, object]]:
    repository = Path(repository).expanduser().resolve()
    return {
        role: _relative_identity(repository / relative, repository)
        for role, relative in ORCHESTRATION_SOURCE_RELATIVE.items()
    }


def _tree_inventory(root: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise HVAWTSensitivityError(f"Case tree contains a symlink: {path}")
        if not path.is_file():
            continue
        if path.stat().st_nlink != 1:
            raise HVAWTSensitivityError(f"Case tree contains a hard link: {path}")
        inventory[path.relative_to(root).as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    if not inventory:
        raise HVAWTSensitivityError(f"Case tree is empty: {root}")
    return inventory


def generate_campaign(
    campaign_root: Path,
    *,
    base_campaign_root: Path,
    repository: Path | None = None,
) -> Path:
    """Generate six complete-path sensitivity cases in a separate root."""

    repo = _repository() if repository is None else Path(repository).resolve()
    base = validate_base_preflight(base_campaign_root)
    base_root = base["base_campaign_root"]
    base_manifest = base["manifest"]
    validate_selected_base_inputs(base)
    baseline_results = validate_selected_nominal_evidence(base)
    protocol_path = repo / PROTOCOL_RELATIVE
    protocol = _load_json(protocol_path, label="sensitivity protocol")
    protocol_gates = protocol.get("acceptance_gates")
    protocol_claim_boundary = protocol.get("claim_boundary")
    if (
        protocol.get("schema")
        != "vorlap-hvawt-targeted-sensitivity-protocol-v1"
        or protocol.get("status")
        != "frozen_before_case_generation_or_execution"
        or not isinstance(protocol_gates, Mapping)
        or not isinstance(protocol_claim_boundary, str)
        or not protocol_claim_boundary.strip()
    ):
        raise HVAWTSensitivityError(
            "Sensitivity protocol is incomplete before generation."
        )
    root = Path(campaign_root).expanduser().resolve(strict=False)
    if root.exists():
        raise FileExistsError(f"Sensitivity campaign already exists: {root}")
    (root / "inputs").mkdir(parents=True)
    (root / "cases").mkdir()
    (root / "results").mkdir()
    (root / "smoke_results").mkdir()

    snapshot_sources = {
        "base_campaign_manifest": base_root / "campaign_manifest.json",
        "selection_receipt": base_root / SELECTION_NAME,
        "recovery_receipt": base_root / RECOVERY_NAME,
        "recovered_analysis": base_root / RECOVERED_ANALYSIS_NAME,
        "base_promotion_receipt": base_root / PROMOTION_RECEIPT_RELATIVE,
        "modal_timestep_qualification": repo
        / MODAL_QUALIFICATION_RELATIVE,
        "sensitivity_protocol": protocol_path,
    }
    snapshots: dict[str, dict[str, object]] = {}
    for role, source in snapshot_sources.items():
        if (
            role == "modal_timestep_qualification"
            and _sha256(source) != MODAL_QUALIFICATION_FILE_SHA256
        ):
            raise HVAWTSensitivityError(
                "Modal/timestep qualification source changed."
            )
        destination = root / "inputs" / source.name
        shutil.copy2(source, destination)
        snapshots[role] = _relative_identity(destination, root)

    base_filter = base_root / str(base_manifest["inputs"]["result_filter"]["path"])
    copied_filter = root / "inputs" / base_filter.name
    shutil.copy2(base_filter, copied_filter)
    snapshots["result_filter"] = _relative_identity(copied_filter, root)

    cases: list[dict[str, object]] = []
    for selected in SELECTED_ROWS:
        row_id = str(selected["row_id"])
        trio = _scenario_cases(base_manifest, row_id)
        for variant in VARIANTS:
            source_scenario = str(variant["source_scenario"])
            source_case = trio[source_scenario]
            source_dir = base_root / str(source_case["case_dir"])
            source_sim = base_root / str(source_case["sim"])
            _assets, source_runtime_fingerprint = qblade_runtime_asset_fingerprint(
                source_sim
            )
            if source_runtime_fingerprint != source_case.get(
                "runtime_asset_fingerprint"
            ):
                raise HVAWTSensitivityError(
                    f"Base runtime assets changed for {source_case['case_id']}."
                )
            variant_name = str(variant["name"])
            case_dir = root / "cases" / row_id / variant_name
            source_inventory = _tree_inventory(source_dir)
            shutil.copytree(source_dir, case_dir, symlinks=False)
            copied_inventory = _tree_inventory(case_dir)
            if copied_inventory != source_inventory:
                raise HVAWTSensitivityError(
                    f"Copied source bytes changed before patching {case_dir}."
                )
            sim_relative = source_sim.relative_to(source_dir)
            sim_path = case_dir / sim_relative
            replace_qblade_keyword_file(
                sim_path, "RELAXTIMESTEP", float(variant["timestep_s"])
            )
            replace_qblade_keyword_file(
                sim_path, "TIMESTEP", float(variant["timestep_s"])
            )
            replace_qblade_keyword_file(
                sim_path, "NUMTIMESTEPS", int(variant["total_steps"])
            )
            runtime_config = source_case.get("runtime_config")
            if source_scenario == "raw_qblade":
                if runtime_config is not None:
                    raise HVAWTSensitivityError("Raw source unexpectedly has SIL config.")
            else:
                if not isinstance(runtime_config, str):
                    raise HVAWTSensitivityError("Coupled source lacks runtime config.")
                _patch_runtime_config(
                    case_dir / runtime_config, variant=variant
                )
            target_inventory = _tree_inventory(case_dir)
            changed_files = sorted(
                path
                for path in set(source_inventory) | set(target_inventory)
                if source_inventory.get(path) != target_inventory.get(path)
            )
            allowed_changes = {sim_relative.as_posix()}
            if isinstance(runtime_config, str):
                allowed_changes.add(runtime_config)
            if not set(changed_files).issubset(allowed_changes):
                raise HVAWTSensitivityError(
                    f"Case mutation exceeded its allowlist: {changed_files}"
                )
            required_changes = (
                {sim_relative.as_posix(), str(runtime_config)}
                if variant["timestep_s"] == HALF_TIMESTEP_S
                else {str(runtime_config)}
            )
            if not required_changes.issubset(changed_files):
                raise HVAWTSensitivityError(
                    f"Case did not apply its required controlled changes: {case_dir}"
                )
            runtime_assets, runtime_fingerprint = qblade_runtime_asset_fingerprint(
                sim_path
            )
            case_id = f"{row_id}__{variant_name}"
            cases.append(
                {
                    "index": len(cases),
                    "row_id": row_id,
                    "seed": selected["seed"],
                    "wind_speed_mps": 35.0,
                    "selection_role": selected["selection_role"],
                    "channel_axis": selected["channel_axis"],
                    "channel": (
                        f"{selected['channel_axis']}_l~Mom.~BLD_1~"
                        "pos~0.230~[Nm]"
                    ),
                    "variant": variant_name,
                    "variant_role": variant["role"],
                    "source_scenario": source_scenario,
                    "source_case_id": source_case["case_id"],
                    "source_case_index": source_case["index"],
                    "source_case_runtime_asset_fingerprint": (
                        source_runtime_fingerprint
                    ),
                    "source_tree_fingerprint": canonical_fingerprint(
                        source_inventory
                    ),
                    "target_tree_fingerprint": canonical_fingerprint(
                        target_inventory
                    ),
                    "controlled_mutation_files": changed_files,
                    "controlled_mutation_allowlist": sorted(allowed_changes),
                    "case_id": case_id,
                    "case_dir": case_dir.relative_to(root).as_posix(),
                    "sim": sim_path.relative_to(root).as_posix(),
                    "runtime_config": runtime_config,
                    "storage_filter": source_case["storage_filter"],
                    "result_filter": copied_filter.relative_to(root).as_posix(),
                    "result_dir": (
                        root / "results" / row_id / variant_name
                    ).relative_to(root).as_posix(),
                    "result_name": f"{case_id}_qblade_results",
                    "timestep_s": variant["timestep_s"],
                    "wind_timestep_s": TIMESTEP_S,
                    "wind_to_qblade_timestep_ratio": (
                        TIMESTEP_S / float(variant["timestep_s"])
                    ),
                    "coarse_wind_interpolation_acknowledged": (
                        variant["timestep_s"] == HALF_TIMESTEP_S
                    ),
                    "total_steps": variant["total_steps"],
                    "retained_rows": variant["retained_rows"],
                    "duration_s": DURATION_S,
                    "store_from_s": STORE_FROM_S,
                    "source_frequency_cap_hz": variant[
                        "source_frequency_cap_hz"
                    ],
                    "checkpoint_every": (
                        48_000
                        if variant["timestep_s"] == HALF_TIMESTEP_S
                        else 24_000
                    ),
                    "progress_every": (
                        2_000
                        if variant["timestep_s"] == HALF_TIMESTEP_S
                        else 1_000
                    ),
                    "force_contract": dict(variant["force_contract"]),
                    "runtime_asset_count": len(runtime_assets),
                    "runtime_asset_fingerprint": runtime_fingerprint,
                }
            )
    if len(cases) != 6 or [case["index"] for case in cases] != list(range(6)):
        raise HVAWTSensitivityError("Targeted campaign must contain exactly six cases.")

    manifest: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "prepared_not_run",
        "sealed_inputs": True,
        "case_count": 6,
        "selected_row_count": 2,
        "variants_per_row": [str(row["name"]) for row in VARIANTS],
        "base_campaign_fingerprint": BASE_CAMPAIGN_FINGERPRINT,
        "selection_fingerprint": SELECTION_FINGERPRINT,
        "base_campaign_root_hint": str(base_root),
        "inputs": {
            **snapshots,
            "orchestration_sources": _orchestration_identities(repo),
        },
        "numerical_contract": {
            "nominal_timestep_s": TIMESTEP_S,
            "half_timestep_s": HALF_TIMESTEP_S,
            "nominal_source_frequency_cap_hz": NOMINAL_CAP_HZ,
            "lower_source_frequency_cap_hz": LOWER_CAP_HZ,
            "lower_cap_basis": (
                "one-half nominal Nyquist; four samples per shortest retained "
                "cycle and 7.29 times the highest reviewed structural mode"
            ),
            "highest_reviewed_mode_hz": HIGHEST_REVIEWED_MODE_HZ,
            "modal_timestep_qualification_fingerprint": (
                MODAL_QUALIFICATION_FINGERPRINT
            ),
            "wind_record_timestep_s": TIMESTEP_S,
            "half_step_wind_handling": (
                "QBlade advances at 0.00375 s while interpolating the unchanged "
                "TurbSim wind record generated on the accepted 0.0075 s grid"
            ),
            "duration_s": DURATION_S,
            "store_from_s": STORE_FROM_S,
        },
        "acceptance_gates": dict(protocol_gates),
        "baseline_results": baseline_results,
        "cases": cases,
        "claim_boundary": protocol_claim_boundary,
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    target = root / MANIFEST_NAME
    atomic_write_json(target, manifest)
    validate_campaign(root, rehash_case_assets=True)
    return target


def _manifest_path(value: Path) -> Path:
    path = Path(value).expanduser().resolve()
    return path / MANIFEST_NAME if path.is_dir() else path


def _require_numeric_keyword(path: Path, keyword: str, expected: float) -> None:
    raw = read_qblade_keyword(path, keyword)
    try:
        observed = float(raw) if raw is not None else math.nan
    except ValueError as exc:
        raise HVAWTSensitivityError(f"{path}: {keyword} is not numeric.") from exc
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-10):
        raise HVAWTSensitivityError(
            f"{path}: {keyword}={observed:g}, expected {expected:g}."
        )


def validate_campaign(
    value: Path, *, rehash_case_assets: bool = True
) -> dict[str, Any]:
    manifest_path = _manifest_path(value)
    root = manifest_path.parent
    manifest = _load_json(manifest_path, label="sensitivity manifest")
    _validate_fingerprint(manifest, label="Sensitivity campaign")
    if (
        manifest.get("schema") != CAMPAIGN_SCHEMA
        or manifest.get("status") != "prepared_not_run"
        or manifest.get("sealed_inputs") is not True
        or manifest.get("case_count") != 6
        or manifest.get("selected_row_count") != 2
        or manifest.get("base_campaign_fingerprint")
        != BASE_CAMPAIGN_FINGERPRINT
        or manifest.get("selection_fingerprint") != SELECTION_FINGERPRINT
        or manifest.get("variants_per_row")
        != [str(row["name"]) for row in VARIANTS]
    ):
        raise HVAWTSensitivityError("Sensitivity campaign header changed.")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise HVAWTSensitivityError("Sensitivity campaign inputs are missing.")
    for role in (
        "base_campaign_manifest",
        "selection_receipt",
        "recovery_receipt",
        "recovered_analysis",
        "base_promotion_receipt",
        "modal_timestep_qualification",
        "sensitivity_protocol",
        "result_filter",
    ):
        _verify_relative_identity(root, inputs.get(role), label=role)
    selection = _load_json(
        root / str(inputs["selection_receipt"]["path"]),
        label="copied selection receipt",
    )
    _validate_fingerprint(selection, label="Copied selection receipt")
    if selection.get("fingerprint") != SELECTION_FINGERPRINT:
        raise HVAWTSensitivityError("Copied selection fingerprint changed.")
    modal_path = root / str(inputs["modal_timestep_qualification"]["path"])
    modal = _load_json(modal_path, label="copied modal/timestep qualification")
    if (
        _sha256(modal_path) != MODAL_QUALIFICATION_FILE_SHA256
        or modal.get("qualification_sha256")
        != MODAL_QUALIFICATION_FINGERPRINT
        or modal.get("qualified") is not True
        or modal.get("timestep_s") != TIMESTEP_S
    ):
        raise HVAWTSensitivityError(
            "Copied modal/timestep qualification changed."
        )
    protocol_path = root / str(inputs["sensitivity_protocol"]["path"])
    protocol = _load_json(protocol_path, label="copied sensitivity protocol")
    if (
        protocol.get("schema")
        != "vorlap-hvawt-targeted-sensitivity-protocol-v1"
        or protocol.get("status")
        != "frozen_before_case_generation_or_execution"
        or protocol.get("case_count") != 6
        or protocol.get("base_campaign_fingerprint")
        != BASE_CAMPAIGN_FINGERPRINT
        or protocol.get("selection_fingerprint") != SELECTION_FINGERPRINT
        or _identity(protocol_path)["sha256"]
        != _identity(_repository() / PROTOCOL_RELATIVE)["sha256"]
        or manifest.get("acceptance_gates")
        != protocol.get("acceptance_gates")
        or manifest.get("claim_boundary") != protocol.get("claim_boundary")
    ):
        raise HVAWTSensitivityError("Copied sensitivity protocol changed.")
    orchestration = inputs.get("orchestration_sources")
    expected_orchestration = _orchestration_identities(_repository())
    if orchestration != expected_orchestration:
        raise HVAWTSensitivityError("Sensitivity orchestration source changed.")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 6:
        raise HVAWTSensitivityError("Sensitivity case list changed.")
    expected_pairs = [
        (str(selected["row_id"]), str(variant["name"]))
        for selected in SELECTED_ROWS
        for variant in VARIANTS
    ]
    observed_pairs = [
        (str(case.get("row_id")), str(case.get("variant")))
        for case in cases
        if isinstance(case, Mapping)
    ]
    if observed_pairs != expected_pairs:
        raise HVAWTSensitivityError("Sensitivity case ordering changed.")
    variant_by_name = {str(row["name"]): row for row in VARIANTS}
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or case.get("index") != index:
            raise HVAWTSensitivityError("Sensitivity case index changed.")
        variant = variant_by_name[str(case["variant"])]
        if (
            case.get("source_scenario") != variant["source_scenario"]
            or case.get("force_contract") != variant["force_contract"]
            or case.get("timestep_s") != variant["timestep_s"]
            or case.get("wind_timestep_s") != TIMESTEP_S
            or case.get("wind_to_qblade_timestep_ratio")
            != TIMESTEP_S / float(variant["timestep_s"])
            or case.get("coarse_wind_interpolation_acknowledged")
            is not (variant["timestep_s"] == HALF_TIMESTEP_S)
            or case.get("total_steps") != variant["total_steps"]
            or case.get("retained_rows") != variant["retained_rows"]
            or case.get("source_frequency_cap_hz")
            != variant["source_frequency_cap_hz"]
            or case.get("checkpoint_every")
            != (
                48_000
                if variant["timestep_s"] == HALF_TIMESTEP_S
                else 24_000
            )
            or case.get("progress_every")
            != (
                2_000
                if variant["timestep_s"] == HALF_TIMESTEP_S
                else 1_000
            )
        ):
            raise HVAWTSensitivityError(
                f"Sensitivity variant contract changed for case {index}."
            )
        case_dir = root / str(case["case_dir"])
        sim_path = root / str(case["sim"])
        if not case_dir.is_dir() or not sim_path.is_file():
            raise HVAWTSensitivityError(f"Missing case assets for case {index}.")
        inventory = _tree_inventory(case_dir)
        if (
            canonical_fingerprint(inventory)
            != case.get("target_tree_fingerprint")
            or not isinstance(case.get("controlled_mutation_files"), list)
            or not set(case["controlled_mutation_files"]).issubset(
                set(case.get("controlled_mutation_allowlist", []))
            )
        ):
            raise HVAWTSensitivityError(
                f"Controlled mutation inventory changed for case {index}."
            )
        _require_numeric_keyword(sim_path, "TIMESTEP", float(case["timestep_s"]))
        _require_numeric_keyword(
            sim_path, "RELAXTIMESTEP", float(case["timestep_s"])
        )
        _require_numeric_keyword(
            sim_path, "NUMTIMESTEPS", float(case["total_steps"])
        )
        _require_numeric_keyword(sim_path, "STOREFROM", STORE_FROM_S)
        runtime_config = case.get("runtime_config")
        if case["source_scenario"] == "raw_qblade":
            if runtime_config is not None:
                raise HVAWTSensitivityError("Raw half-timestep case activates SIL.")
        else:
            config = _load_json(
                case_dir / str(runtime_config),
                label=f"runtime config case {index}",
            )
            if (
                not math.isclose(
                    float(config.get("sample_step", math.nan)),
                    float(case["timestep_s"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                or not math.isclose(
                    float(config.get("source_frequency_cap_hz", math.nan)),
                    float(case["source_frequency_cap_hz"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or config.get("source_parameter_dir") != "."
            ):
                raise HVAWTSensitivityError("Runtime numerical controls changed.")
            hvawt = config.get("hvawt_response_model")
            if case["source_scenario"] == "one_way_spectral_vorlap":
                if hvawt is not None or float(config.get("force_scale", -1.0)) != 1.0:
                    raise HVAWTSensitivityError("One-way force contract changed.")
            else:
                engineering = (
                    hvawt.get("engineering_lockin")
                    if isinstance(hvawt, Mapping)
                    else None
                )
                if (
                    float(config.get("force_scale", -1.0)) != 0.0
                    or not isinstance(engineering, Mapping)
                    or engineering.get("residual_source_scale") != 1.0
                    or engineering.get("response_force_scale")
                    != case["force_contract"]["response_force_scale"]
                    or engineering.get("velocity_basis")
                    != "local_rigid_body_relative_unscaled"
                ):
                    raise HVAWTSensitivityError("Complete force partition changed.")
        if rehash_case_assets:
            assets, fingerprint = qblade_runtime_asset_fingerprint(sim_path)
            if (
                fingerprint != case.get("runtime_asset_fingerprint")
                or len(assets) != case.get("runtime_asset_count")
            ):
                raise HVAWTSensitivityError(
                    f"Runtime assets changed for sensitivity case {index}."
                )
    return {
        "valid": True,
        "campaign_root": root,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def build_runtime_identity(qblade_root: Path) -> dict[str, object]:
    identity = build_base_runtime_identity(qblade_root)
    if (
        identity["qblade_library"]["sha256"] != QBLADE_LIBRARY_SHA256
        or identity["bridge_library"]["sha256"] != BRIDGE_LIBRARY_SHA256
    ):
        raise HVAWTSensitivityError("Flight runtime library identity changed.")
    return {
        "schema": "vorlap-hvawt-targeted-sensitivity-runtime-v1",
        "base_runtime": identity,
        "sensitivity_orchestration_sources": _orchestration_identities(
            _repository()
        ),
    }


def bind_runtime(campaign: Path, *, qblade_root: Path) -> dict[str, object]:
    state = validate_campaign(campaign)
    root = state["campaign_root"]
    observed = build_runtime_identity(qblade_root)
    path = root / RUNTIME_RECEIPT_NAME
    if path.exists():
        recorded = _load_json(path, label="sensitivity runtime receipt")
        if recorded != observed:
            raise HVAWTSensitivityError("Bound Flight runtime changed.")
    else:
        atomic_write_json(path, observed)
    return observed


def _load_runtime(root: Path) -> dict[str, Any]:
    receipt = _load_json(root / RUNTIME_RECEIPT_NAME, label="runtime receipt")
    base_runtime = receipt.get("base_runtime")
    qblade_library = (
        base_runtime.get("qblade_library")
        if isinstance(base_runtime, Mapping)
        else None
    )
    bridge_library = (
        base_runtime.get("bridge_library")
        if isinstance(base_runtime, Mapping)
        else None
    )
    if (
        receipt.get("schema")
        != "vorlap-hvawt-targeted-sensitivity-runtime-v1"
        or not isinstance(base_runtime, Mapping)
        or base_runtime.get("schema")
        != "vorlap-hvawt-complete-dlc-flight-runtime-v1"
        or not isinstance(qblade_library, Mapping)
        or qblade_library.get("sha256") != QBLADE_LIBRARY_SHA256
        or not isinstance(bridge_library, Mapping)
        or bridge_library.get("sha256") != BRIDGE_LIBRARY_SHA256
        or receipt.get("sensitivity_orchestration_sources")
        != _orchestration_identities(_repository())
    ):
        raise HVAWTSensitivityError("Sensitivity runtime receipt is invalid.")
    return receipt


def command_for_case(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
    python_executable: str = sys.executable,
    threads: int = 1,
    checkpoint_every: int | None = None,
    progress_every: int | None = None,
) -> list[str]:
    state = validate_campaign(campaign, rehash_case_assets=False)
    cases = state["manifest"]["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise IndexError("Sensitivity case index must be in 0..5.")
    if isinstance(threads, bool) or not 1 <= int(threads) <= 112:
        raise HVAWTSensitivityError("QBlade threads must be in 1..112.")
    case = cases[case_index]
    root = state["campaign_root"]
    sim_path = root / str(case["sim"])
    _assets, fingerprint = qblade_runtime_asset_fingerprint(sim_path)
    if fingerprint != case["runtime_asset_fingerprint"]:
        raise HVAWTSensitivityError("Selected sensitivity case assets changed.")
    result_dir = root / str(case["result_dir"])
    sealed_checkpoint = int(case["checkpoint_every"])
    sealed_progress = int(case["progress_every"])
    if checkpoint_every is not None and int(checkpoint_every) != sealed_checkpoint:
        raise HVAWTSensitivityError(
            "Checkpoint interval differs from the sealed physical-time policy."
        )
    if progress_every is not None and int(progress_every) != sealed_progress:
        raise HVAWTSensitivityError(
            "Progress interval differs from the sealed physical-time policy."
        )
    command = [
        str(python_executable),
        str(_repository() / "run_headless/run_qblade_sim.py"),
        "--sim",
        str(sim_path),
        "--qblade-root",
        str(Path(qblade_root).expanduser().resolve()),
        "--output-dir",
        str(result_dir),
        "--result-name",
        str(case["result_name"]),
        "--result-filter",
        str(root / str(case["result_filter"])),
        "--threads",
        str(int(threads)),
        "--coupling-mode",
        str(case["force_contract"]["coupling_mode"]),
        "--checkpoint-every",
        str(sealed_checkpoint),
        "--progress-every",
        str(sealed_progress),
        "--case-metadata",
        str(result_dir / "case_metadata.json"),
    ]
    if case["force_contract"]["coupling_mode"] == "sil-vorlap":
        command.extend(("--sil-vorlap", "--sil-vorlap-history-mode", "compact"))
    return command


def _replace_command_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise HVAWTSensitivityError(f"Command lacks option {option}.") from exc
    command[index + 1] = value


def _read_history_force_metrics(path: Path) -> dict[str, float]:
    maximum_response = 0.0
    maximum_residual = 0.0
    maximum_total = 0.0
    maximum_power_closure = 0.0
    maximum_generalized_power = 0.0
    maximum_nodal_power = 0.0
    maximum_total_residual_error = 0.0
    rows = 0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "response_engineering_maximum_abs_nodal_force_n",
            "response_spectral_applied_residual_maximum_nodal_force_n",
            "response_total_maximum_nodal_force_n",
            "response_engineering_generalized_power_w",
            "response_engineering_nodal_flexible_power_w",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise HVAWTSensitivityError("Compact history lacks force closure columns.")
        for row in reader:
            try:
                response = float(
                    row["response_engineering_maximum_abs_nodal_force_n"]
                )
                residual = float(
                    row[
                        "response_spectral_applied_residual_maximum_nodal_force_n"
                    ]
                )
                total = float(row["response_total_maximum_nodal_force_n"])
                generalized = float(
                    row["response_engineering_generalized_power_w"]
                )
                nodal = float(row["response_engineering_nodal_flexible_power_w"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HVAWTSensitivityError("History force row is invalid.") from exc
            if not all(
                math.isfinite(value)
                for value in (response, residual, total, generalized, nodal)
            ):
                raise HVAWTSensitivityError("History force row is nonfinite.")
            maximum_response = max(maximum_response, abs(response))
            maximum_residual = max(maximum_residual, abs(residual))
            maximum_total = max(maximum_total, abs(total))
            maximum_generalized_power = max(
                maximum_generalized_power, abs(generalized)
            )
            maximum_nodal_power = max(maximum_nodal_power, abs(nodal))
            maximum_power_closure = max(
                maximum_power_closure, abs(generalized - nodal)
            )
            maximum_total_residual_error = max(
                maximum_total_residual_error, abs(total - residual)
            )
            rows += 1
    tolerance = 1.0e-10 * max(
        1.0, maximum_generalized_power, maximum_nodal_power
    )
    return {
        "rows": rows,
        "maximum_response_force_n": maximum_response,
        "maximum_residual_force_n": maximum_residual,
        "maximum_total_force_n": maximum_total,
        "maximum_engineering_generalized_power_w": maximum_generalized_power,
        "maximum_engineering_nodal_power_w": maximum_nodal_power,
        "maximum_virtual_work_error_w": maximum_power_closure,
        "virtual_work_tolerance_w": tolerance,
        "maximum_total_residual_force_error_n": maximum_total_residual_error,
    }


def _validate_complete_force_partition_metrics(
    metrics: Mapping[str, Any],
    *,
    response_force_scale: float,
    require_response_signal: bool,
) -> None:
    """Fail closed on residual/response partition and virtual-work evidence."""

    required = {
        "maximum_response_force_n",
        "maximum_residual_force_n",
        "maximum_total_force_n",
        "maximum_engineering_generalized_power_w",
        "maximum_engineering_nodal_power_w",
        "maximum_virtual_work_error_w",
        "virtual_work_tolerance_w",
        "maximum_total_residual_force_error_n",
    }
    if not required.issubset(metrics):
        raise HVAWTSensitivityError("Complete force-partition metrics are incomplete.")
    values = {name: float(metrics[name]) for name in required}
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        raise HVAWTSensitivityError("Complete force-partition metric is invalid.")
    if (
        values["maximum_residual_force_n"] <= 0.0
        or values["maximum_total_force_n"] <= 0.0
    ):
        raise HVAWTSensitivityError("Complete partition lost residual or total force.")
    if (
        values["maximum_virtual_work_error_w"]
        > values["virtual_work_tolerance_w"]
    ):
        raise HVAWTSensitivityError("Complete response virtual-work closure failed.")
    if response_force_scale == 0.0:
        if (
            values["maximum_response_force_n"] != 0.0
            or values["maximum_engineering_generalized_power_w"] != 0.0
            or values["maximum_engineering_nodal_power_w"] != 0.0
            or values["maximum_total_residual_force_error_n"] > 1.0e-12
        ):
            raise HVAWTSensitivityError(
                "Response-off case applied response work or double counted force."
            )
    elif response_force_scale == 1.0:
        if require_response_signal and values["maximum_response_force_n"] <= 0.0:
            raise HVAWTSensitivityError(
                "Complete case never applied response force."
            )
    else:
        raise HVAWTSensitivityError("Unsupported response-force scale.")


def _audit_compact_history_grid(
    path: Path, *, timestep_s: float, total_steps: int
) -> dict[str, object]:
    required = {
        "step_index",
        "time_s",
        "qblade_timestep",
        "generated_max_abs_force_component_n",
        "applied_max_abs_force_component_n",
    }
    rows = 0
    maximum_generated = 0.0
    maximum_applied = 0.0
    maximum_time_error = 0.0
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise HVAWTSensitivityError(
                "Compact history lacks force and time-grid columns."
            )
        for row_index, row in enumerate(reader):
            try:
                step = int(row["step_index"])
                time_s = float(row["time_s"])
                qblade_step = float(row["qblade_timestep"])
                generated = float(row["generated_max_abs_force_component_n"])
                applied = float(row["applied_max_abs_force_component_n"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HVAWTSensitivityError(
                    f"Compact history row {row_index} is invalid."
                ) from exc
            if not all(
                math.isfinite(value)
                for value in (time_s, qblade_step, generated, applied)
            ):
                raise HVAWTSensitivityError("Compact history is nonfinite.")
            if step != row_index or not math.isclose(
                qblade_step,
                float(row_index),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise HVAWTSensitivityError(
                    "Compact history step indices are not sequential."
                )
            maximum_time_error = max(
                maximum_time_error, abs(time_s - row_index * timestep_s)
            )
            maximum_generated = max(maximum_generated, abs(generated))
            maximum_applied = max(maximum_applied, abs(applied))
            rows += 1
    if rows != total_steps or maximum_time_error > 1.0e-9:
        raise HVAWTSensitivityError(
            "Compact history length or physical time grid changed."
        )
    if maximum_generated <= 0.0 or maximum_applied <= 0.0:
        raise HVAWTSensitivityError("Compact history never applied a force.")
    return {
        "rows": rows,
        "maximum_time_grid_error_s": maximum_time_error,
        "maximum_generated_force_component_n": maximum_generated,
        "maximum_applied_force_component_n": maximum_applied,
    }


def _terminal_case_audit(
    *, root: Path, manifest: Mapping[str, Any], case: Mapping[str, Any]
) -> dict[str, object]:
    result_dir = root / str(case["result_dir"])
    sim_path = root / str(case["sim"])
    summary_path = result_dir / f"{sim_path.stem}_run_summary.json"
    result_path = result_dir / f"{case['result_name']}.txt"
    metadata_path = result_dir / "case_metadata.json"
    if not all(path.is_file() for path in (summary_path, result_path, metadata_path)):
        raise HVAWTSensitivityError("Terminal result, summary, or metadata is missing.")
    summary = _load_json(summary_path, label=f"summary {case['case_id']}")
    metadata = _load_json(metadata_path, label=f"metadata {case['case_id']}")
    if (
        summary.get("failed") is not False
        or summary.get("stopped") is not False
        or summary.get("allow_partial") is not False
        or summary.get("exported_results") is not True
        or summary.get("qblade_execution_mode") != "step-loop"
        or summary.get("requested_steps") != case["total_steps"]
        or summary.get("completed_steps") != case["total_steps"]
        or summary.get("coupling_mode")
        != case["force_contract"]["coupling_mode"]
        or metadata.get("campaign_fingerprint") != manifest["fingerprint"]
        or metadata.get("case_id") != case["case_id"]
        or metadata.get("case_index") != case["index"]
        or metadata.get("row_id") != case["row_id"]
        or metadata.get("variant") != case["variant"]
        or metadata.get("source_scenario") != case["source_scenario"]
        or metadata.get("force_contract") != case["force_contract"]
        or metadata.get("runtime_identity_fingerprint")
        != canonical_fingerprint(_load_runtime(root))
    ):
        raise HVAWTSensitivityError("Terminal execution receipt is mismatched.")
    restart = summary.get("restart")
    if (
        not isinstance(restart, Mapping)
        or restart.get("checkpoint_every") != case["checkpoint_every"]
        or restart.get("resume_requested") is not False
        or restart.get("resume_checkpoint") is not None
        or restart.get("resume_start_step") != 0
    ):
        raise HVAWTSensitivityError("Sensitivity result used continuation.")
    native_callback_evidence = summary.get("native_callback_evidence")
    if not isinstance(native_callback_evidence, dict):
        raise HVAWTSensitivityError(
            "Terminal native-callback evidence is missing or invalid."
        )
    require_zero_native_callback_evidence(
        native_callback_evidence,
        context=f"sensitivity case {case['case_id']}",
    )
    reproducibility = summary.get("reproducibility")
    qblade_block = (
        reproducibility.get("qblade")
        if isinstance(reproducibility, Mapping)
        else None
    )
    qblade_library = (
        qblade_block.get("library")
        if isinstance(qblade_block, Mapping)
        else None
    )
    runtime_identity = _load_runtime(root)
    base_runtime = runtime_identity.get("base_runtime")
    expected_bridge = (
        base_runtime.get("bridge_library")
        if isinstance(base_runtime, Mapping)
        else None
    )
    if (
        not isinstance(reproducibility, Mapping)
        or not isinstance(qblade_library, Mapping)
        or qblade_library.get("sha256") != QBLADE_LIBRARY_SHA256
        or not isinstance(expected_bridge, Mapping)
        or expected_bridge.get("sha256") != BRIDGE_LIBRARY_SHA256
        or reproducibility.get("runtime_asset_fingerprint")
        != case["runtime_asset_fingerprint"]
    ):
        raise HVAWTSensitivityError(
            "Terminal runtime or staged-asset provenance changed."
        )
    result_identity = summary.get("result_file_identity")
    observed_result = _identity(result_path)
    if (
        not isinstance(result_identity, Mapping)
        or result_identity.get("sha256") != observed_result["sha256"]
        or result_identity.get("size_bytes") != observed_result["size_bytes"]
    ):
        raise HVAWTSensitivityError("Result identity is invalid.")
    result_audit = _audit_qblade_fatigue_result(
        result_path,
        result_filter=root / str(case["result_filter"]),
        timestep_s=float(case["timestep_s"]),
        total_steps=int(case["total_steps"]),
        store_from_s=STORE_FROM_S,
    )
    scenario = str(case["source_scenario"])
    updates = int(summary.get("sil_vorlap_updates", 0) or 0)
    if scenario == "raw_qblade":
        if updates != 0 or summary.get("sil_vorlap") is not False:
            raise HVAWTSensitivityError("Raw sensitivity case applied SIL forcing.")
        history_record = None
        history_metrics = None
        force_metrics = None
    else:
        if (
            updates != case["total_steps"]
            or summary.get("sil_vorlap") is not True
            or summary.get("sil_vorlap_history_mode") != "compact"
            or summary.get("sil_vorlap_source_bounds_policy") != "raise"
            or not math.isclose(
                float(summary.get("sil_vorlap_source_frequency_cap_hz", math.nan)),
                float(case["source_frequency_cap_hz"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise HVAWTSensitivityError("Coupled sensitivity execution changed.")
        validate_complete_source_coverage(
            summary.get("sil_vorlap_source_coverage"),
            expected_samples=int(case["total_steps"]),
            component_node_counts=HVAWT_V8_COMPONENT_NODE_COUNTS,
        )
        validate_complete_velocity_diagnostics(
            summary.get("sil_vorlap_velocity_diagnostics"),
            expected_samples=int(case["total_steps"]),
            node_count=sum(HVAWT_V8_COMPONENT_NODE_COUNTS.values()),
        )
        coverage = summary["sil_vorlap_source_coverage"]
        if any(
            float(row["freq_max_hz"])
            > float(case["source_frequency_cap_hz"]) + 1.0e-9
            for row in coverage
        ):
            raise HVAWTSensitivityError("Retained source exceeds declared cap.")
        history_path = _history_path_from_summary(summary, result_dir)
        if not history_path.is_file():
            candidate = result_dir / history_path.name
            if candidate.is_file():
                history_path = candidate
        grid_audit = _audit_compact_history_grid(
            history_path,
            timestep_s=float(case["timestep_s"]),
            total_steps=int(case["total_steps"]),
        )
        if scenario == "one_way_spectral_vorlap":
            history_metrics = None
            force_metrics = None
        else:
            history_metrics = analyze_history(
                history_path,
                tail_start_s=STORE_FROM_S,
                tail_duration_s=DURATION_S - STORE_FROM_S,
            )
            force_metrics = _read_history_force_metrics(history_path)
            if (
                history_metrics["rows"] != case["total_steps"]
                or history_metrics["spectral_residual_source_scale"] != 1.0
                or history_metrics["maximum_spectral_energy_closure_error"]
                > history_metrics["spectral_energy_closure_tolerance"]
            ):
                raise HVAWTSensitivityError(
                    "Complete force or energy closure failed."
                )
            response_scale = float(
                case["force_contract"]["response_force_scale"]
            )
            _validate_complete_force_partition_metrics(
                force_metrics,
                response_force_scale=response_scale,
                require_response_signal=True,
            )
        history_record = _relative_identity(history_path, root)
    return {
        "summary": _relative_identity(summary_path, root),
        "result": _relative_identity(result_path, root),
        "metadata": _relative_identity(metadata_path, root),
        "history": history_record,
        "compact_history_grid": (
            None if scenario == "raw_qblade" else grid_audit
        ),
        "result_audit": result_audit,
        "history_metrics": history_metrics,
        "force_metrics": force_metrics,
    }


def classify_case(campaign: Path, *, case_index: int) -> dict[str, object]:
    state = validate_campaign(campaign, rehash_case_assets=False)
    cases = state["manifest"]["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise IndexError("Sensitivity case index must be in 0..5.")
    case = cases[case_index]
    result_dir = state["campaign_root"] / str(case["result_dir"])
    if not result_dir.exists() or (
        result_dir.is_dir() and not any(result_dir.iterdir())
    ):
        return {"case_id": case["case_id"], "status": "runnable"}
    if not result_dir.is_dir():
        return {
            "case_id": case["case_id"],
            "status": "blocked",
            "reason": "result_path_not_directory",
        }
    try:
        receipt = _terminal_case_audit(
            root=state["campaign_root"],
            manifest=state["manifest"],
            case=case,
        )
    except Exception as exc:
        return {
            "case_id": case["case_id"],
            "status": "blocked",
            "reason": f"partial_or_invalid:{exc}",
        }
    return {"case_id": case["case_id"], "status": "complete", "receipt": receipt}


def _archive_partial(root: Path, case: Mapping[str, Any], reason: str) -> Path:
    source = root / str(case["result_dir"])
    archive_root = root / "aborted_attempts" / str(case["case_id"])
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"attempt_{len(list(archive_root.glob('attempt_*'))) + 1:03d}"
    shutil.move(str(source), destination)
    atomic_write_json(
        destination / "archive_reason.json",
        {"case_id": case["case_id"], "reason": reason},
    )
    return destination


def _load_authorization(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_json(root / AUTHORIZATION_NAME, label="execution authorization")
    _validate_fingerprint(receipt, label="Sensitivity authorization")
    if (
        receipt.get("schema") != AUTHORIZATION_SCHEMA
        or receipt.get("status") != "authorized_after_6_case_input_smoke"
        or receipt.get("campaign_fingerprint") != manifest.get("fingerprint")
        or receipt.get("authorized_case_indices") != list(range(6))
        or receipt.get("authorized_case_ids")
        != [str(case["case_id"]) for case in manifest["cases"]]
        or receipt.get("fresh_start_only") is not True
    ):
        raise HVAWTSensitivityError("Sensitivity execution authorization is invalid.")
    _verify_relative_identity(
        root, receipt.get("smoke_receipt"), label="sensitivity smoke receipt"
    )
    runtime_path = _verify_relative_identity(
        root,
        receipt.get("runtime_receipt"),
        label="sensitivity authorization runtime receipt",
    )
    runtime = _load_runtime(root)
    if (
        runtime_path != (root / RUNTIME_RECEIPT_NAME).resolve()
        or receipt.get("runtime_identity_fingerprint")
        != canonical_fingerprint(runtime)
    ):
        raise HVAWTSensitivityError(
            "Sensitivity authorization runtime binding changed."
        )
    smoke = execute_smoke_receipt(root, manifest)
    if receipt.get("smoke_fingerprint") != smoke.get("fingerprint"):
        raise HVAWTSensitivityError(
            "Sensitivity authorization does not bind the accepted smoke."
        )
    return receipt


def execute_case(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
    python_executable: str,
    threads: int,
    checkpoint_every: int | None = None,
    progress_every: int | None = None,
) -> dict[str, object]:
    state = validate_campaign(campaign, rehash_case_assets=False)
    root = state["campaign_root"]
    manifest = state["manifest"]
    runtime = _load_runtime(root)
    if runtime != build_runtime_identity(qblade_root):
        raise HVAWTSensitivityError("Current Flight runtime differs from binding.")
    _load_authorization(root, manifest)
    cases = manifest["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise IndexError("Sensitivity case index must be in 0..5.")
    case = cases[case_index]
    lock_root = root / ".case_locks"
    lock_root.mkdir(exist_ok=True)
    with (lock_root / f"{case['case_id']}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HVAWTSensitivityError("Sensitivity case is already running.") from exc
        classification = classify_case(root, case_index=case_index)
        if classification["status"] == "complete":
            return classification
        if classification["status"] == "blocked":
            _archive_partial(root, case, str(classification.get("reason")))
        result_dir = root / str(case["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema": "vorlap-hvawt-targeted-sensitivity-case-metadata-v1",
            "campaign_fingerprint": manifest["fingerprint"],
            "case_id": case["case_id"],
            "case_index": case_index,
            "row_id": case["row_id"],
            "variant": case["variant"],
            "source_scenario": case["source_scenario"],
            "force_contract": case["force_contract"],
            "runtime_identity_fingerprint": canonical_fingerprint(runtime),
        }
        atomic_write_json(result_dir / "case_metadata.json", metadata)
        command = command_for_case(
            root,
            case_index=case_index,
            qblade_root=qblade_root,
            python_executable=python_executable,
            threads=threads,
            checkpoint_every=checkpoint_every,
            progress_every=progress_every,
        )
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(threads))
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        subprocess.run(command, check=True, env=env)
        terminal = classify_case(root, case_index=case_index)
        if terminal["status"] != "complete":
            raise HVAWTSensitivityError(
                f"Sensitivity case did not produce a terminal receipt: {terminal}"
            )
        return terminal


def execute_smoke(
    campaign: Path,
    *,
    qblade_root: Path,
    python_executable: str,
    threads: int,
    progress_every: int = 0,
) -> dict[str, object]:
    """Run four fresh steps for every planned input without using result dirs."""

    state = validate_campaign(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    runtime = _load_runtime(root)
    if runtime != build_runtime_identity(qblade_root):
        raise HVAWTSensitivityError("Current Flight runtime differs from binding.")
    receipt_path = root / SMOKE_RECEIPT_NAME
    if receipt_path.exists():
        existing = _load_json(receipt_path, label="sensitivity smoke receipt")
        _validate_fingerprint(existing, label="Sensitivity smoke receipt")
        if (
            existing.get("schema") != SMOKE_SCHEMA
            or existing.get("status") != "passed_all_6_four_step_input_smokes"
            or existing.get("campaign_fingerprint") != manifest["fingerprint"]
        ):
            raise HVAWTSensitivityError("Existing smoke receipt is invalid.")
        for case in existing.get("cases", []):
            if not isinstance(case, Mapping):
                raise HVAWTSensitivityError("Smoke case receipt is invalid.")
            for role in ("summary", "result", "metadata"):
                _verify_relative_identity(
                    root,
                    case.get(role),
                    label=f"smoke {case.get('case_id')} {role}",
                )
            if case.get("history") is not None:
                _verify_relative_identity(
                    root,
                    case.get("history"),
                    label=f"smoke {case.get('case_id')} history",
                )
        return existing

    case_receipts: list[dict[str, object]] = []
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(int(threads))
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    for case in manifest["cases"]:
        case_index = int(case["index"])
        output = root / "smoke_results" / str(case["case_id"])
        if output.exists():
            raise HVAWTSensitivityError(
                f"Smoke output exists without a receipt: {output}"
            )
        output.mkdir(parents=True)
        metadata_path = output / "case_metadata.json"
        metadata = {
            "schema": "vorlap-hvawt-targeted-sensitivity-smoke-metadata-v1",
            "campaign_fingerprint": manifest["fingerprint"],
            "case_id": case["case_id"],
            "case_index": case_index,
            "row_id": case["row_id"],
            "variant": case["variant"],
            "source_scenario": case["source_scenario"],
            "force_contract": case["force_contract"],
            "runtime_identity_fingerprint": canonical_fingerprint(runtime),
            "smoke_only": True,
        }
        atomic_write_json(metadata_path, metadata)
        command = command_for_case(
            root,
            case_index=case_index,
            qblade_root=qblade_root,
            python_executable=python_executable,
            threads=threads,
        )
        _replace_command_option(command, "--checkpoint-every", "0")
        _replace_command_option(
            command, "--progress-every", str(int(progress_every))
        )
        _replace_command_option(command, "--output-dir", str(output))
        _replace_command_option(command, "--case-metadata", str(metadata_path))
        command.extend(("--steps", "4"))
        subprocess.run(command, check=True, env=env)
        sim_path = root / str(case["sim"])
        summary_path = output / f"{sim_path.stem}_run_summary.json"
        result_path = output / f"{case['result_name']}.txt"
        summary = _load_json(
            summary_path, label=f"smoke summary {case['case_id']}"
        )
        if (
            summary.get("failed") is not False
            or summary.get("stopped") is not False
            or summary.get("exported_results") is not True
            or summary.get("requested_steps") != 4
            or summary.get("completed_steps") != 4
            or summary.get("coupling_mode")
            != case["force_contract"]["coupling_mode"]
        ):
            raise HVAWTSensitivityError(
                f"Four-step smoke failed for {case['case_id']}."
            )
        history_record = None
        maximum_residual = None
        maximum_response = None
        maximum_total = None
        force_metrics = None
        if case["source_scenario"] == "raw_qblade":
            if (
                int(summary.get("sil_vorlap_updates", 0) or 0) != 0
                or summary.get("sil_vorlap") is not False
            ):
                raise HVAWTSensitivityError("Raw smoke applied SIL forcing.")
        else:
            if (
                summary.get("sil_vorlap") is not True
                or summary.get("sil_vorlap_updates") != 4
                or not math.isclose(
                    float(
                        summary.get(
                            "sil_vorlap_source_frequency_cap_hz", math.nan
                        )
                    ),
                    float(case["source_frequency_cap_hz"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ):
                raise HVAWTSensitivityError("Coupled smoke contract changed.")
            history_path = _history_path_from_summary(summary, output)
            if not history_path.is_file():
                candidate = output / history_path.name
                if candidate.is_file():
                    history_path = candidate
            if case["source_scenario"] == "complete_hybrid_v8":
                force_metrics = _read_history_force_metrics(history_path)
                maximum_residual = force_metrics["maximum_residual_force_n"]
                maximum_response = force_metrics["maximum_response_force_n"]
                maximum_total = force_metrics["maximum_total_force_n"]
                _validate_complete_force_partition_metrics(
                    force_metrics,
                    response_force_scale=float(
                        case["force_contract"]["response_force_scale"]
                    ),
                    require_response_signal=False,
                )
            history_record = _relative_identity(history_path, root)
        case_receipts.append(
            {
                "case_index": case_index,
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "variant": case["variant"],
                "source_scenario": case["source_scenario"],
                "force_contract": case["force_contract"],
                "completed_steps": 4,
                "maximum_residual_force_n": maximum_residual,
                "maximum_response_force_n": maximum_response,
                "maximum_total_force_n": maximum_total,
                "force_metrics": force_metrics,
                "summary": _relative_identity(summary_path, root),
                "result": _relative_identity(result_path, root),
                "metadata": _relative_identity(metadata_path, root),
                "history": history_record,
            }
        )
    receipt: dict[str, object] = {
        "schema": SMOKE_SCHEMA,
        "status": "passed_all_6_four_step_input_smokes",
        "claim_boundary": (
            "Input/runtime/finite-force smoke only; not load, response, fatigue, "
            "robustness, validation, or certification evidence"
        ),
        "campaign_fingerprint": manifest["fingerprint"],
        "runtime_receipt": _relative_identity(
            root / RUNTIME_RECEIPT_NAME, root
        ),
        "runtime_identity_fingerprint": canonical_fingerprint(runtime),
        "case_count": 6,
        "completed_steps_per_case": 4,
        "cases": case_receipts,
    }
    receipt["fingerprint"] = canonical_fingerprint(receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


def bind_execution_authorization(campaign: Path) -> dict[str, object]:
    """Authorize only the exact six cases after all input smokes pass."""

    state = validate_campaign(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    runtime = _load_runtime(root)
    smoke = execute_smoke_receipt(root, manifest)
    receipt_path = root / AUTHORIZATION_NAME
    receipt: dict[str, object] = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "authorized_after_6_case_input_smoke",
        "campaign_fingerprint": manifest["fingerprint"],
        "authorized_case_indices": list(range(6)),
        "authorized_case_ids": [
            str(case["case_id"]) for case in manifest["cases"]
        ],
        "smoke_receipt": _relative_identity(root / SMOKE_RECEIPT_NAME, root),
        "smoke_fingerprint": smoke["fingerprint"],
        "runtime_receipt": _relative_identity(root / RUNTIME_RECEIPT_NAME, root),
        "runtime_identity_fingerprint": canonical_fingerprint(runtime),
        "fresh_start_only": True,
        "claim_boundary": (
            "Execution authorization for targeted numerical sensitivity and "
            "response ablation only"
        ),
    }
    receipt["fingerprint"] = canonical_fingerprint(receipt)
    if receipt_path.exists():
        existing = _load_json(
            receipt_path, label="existing sensitivity authorization"
        )
        if existing != receipt:
            raise HVAWTSensitivityError(
                "Existing sensitivity authorization differs."
            )
        return existing
    atomic_write_json(receipt_path, receipt)
    return receipt


def execute_smoke_receipt(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _load_json(
        root / SMOKE_RECEIPT_NAME, label="sensitivity smoke receipt"
    )
    _validate_fingerprint(receipt, label="Sensitivity smoke receipt")
    cases = receipt.get("cases")
    runtime_path = _verify_relative_identity(
        root, receipt.get("runtime_receipt"), label="smoke runtime receipt"
    )
    runtime = _load_runtime(root)
    if (
        receipt.get("schema") != SMOKE_SCHEMA
        or receipt.get("status") != "passed_all_6_four_step_input_smokes"
        or receipt.get("campaign_fingerprint") != manifest.get("fingerprint")
        or receipt.get("case_count") != 6
        or receipt.get("completed_steps_per_case") != 4
        or not isinstance(cases, list)
        or len(cases) != 6
        or not all(isinstance(case, Mapping) for case in cases)
        or [case["case_index"] for case in cases] != list(range(6))
        or runtime_path != (root / RUNTIME_RECEIPT_NAME).resolve()
        or receipt.get("runtime_identity_fingerprint")
        != canonical_fingerprint(runtime)
    ):
        raise HVAWTSensitivityError("Sensitivity smoke receipt is invalid.")
    for recorded, planned in zip(cases, manifest["cases"], strict=True):
        if (
            recorded.get("case_id") != planned["case_id"]
            or recorded.get("row_id") != planned["row_id"]
            or recorded.get("variant") != planned["variant"]
            or recorded.get("source_scenario") != planned["source_scenario"]
            or recorded.get("force_contract") != planned["force_contract"]
            or recorded.get("completed_steps") != 4
        ):
            raise HVAWTSensitivityError(
                "Sensitivity smoke case ordering or semantics changed."
            )
        verified_paths = {
            role: _verify_relative_identity(
                root,
                recorded.get(role),
                label=f"smoke {recorded.get('case_id')} {role}",
            )
            for role in ("summary", "result", "metadata")
        }
        summary = _load_json(
            verified_paths["summary"], label=f"smoke summary {planned['case_id']}"
        )
        metadata = _load_json(
            verified_paths["metadata"], label=f"smoke metadata {planned['case_id']}"
        )
        if (
            summary.get("failed") is not False
            or summary.get("stopped") is not False
            or summary.get("exported_results") is not True
            or summary.get("requested_steps") != 4
            or summary.get("completed_steps") != 4
            or summary.get("coupling_mode")
            != planned["force_contract"]["coupling_mode"]
            or metadata.get("campaign_fingerprint") != manifest["fingerprint"]
            or metadata.get("case_id") != planned["case_id"]
            or metadata.get("case_index") != planned["index"]
            or metadata.get("row_id") != planned["row_id"]
            or metadata.get("variant") != planned["variant"]
            or metadata.get("source_scenario") != planned["source_scenario"]
            or metadata.get("force_contract") != planned["force_contract"]
            or metadata.get("runtime_identity_fingerprint")
            != canonical_fingerprint(runtime)
            or metadata.get("smoke_only") is not True
        ):
            raise HVAWTSensitivityError("Sensitivity smoke artifact semantics changed.")
        history_record = recorded.get("history")
        if planned["source_scenario"] == "complete_hybrid_v8":
            history_path = _verify_relative_identity(
                root,
                history_record,
                label=f"smoke {planned['case_id']} history",
            )
            observed_force = _read_history_force_metrics(history_path)
            if recorded.get("force_metrics") != observed_force:
                raise HVAWTSensitivityError(
                    "Sensitivity smoke force metrics changed."
                )
            _validate_complete_force_partition_metrics(
                observed_force,
                response_force_scale=float(
                    planned["force_contract"]["response_force_scale"]
                ),
                require_response_signal=False,
            )
            if (
                recorded.get("maximum_residual_force_n")
                != observed_force["maximum_residual_force_n"]
                or recorded.get("maximum_response_force_n")
                != observed_force["maximum_response_force_n"]
                or recorded.get("maximum_total_force_n")
                != observed_force["maximum_total_force_n"]
            ):
                raise HVAWTSensitivityError(
                    "Sensitivity smoke force summary changed."
                )
        elif history_record is not None or recorded.get("force_metrics") is not None:
            raise HVAWTSensitivityError(
                "Non-complete sensitivity smoke recorded complete force metrics."
            )
    return receipt


def _load_metric(
    *,
    path: Path,
    identity_root: Path,
    channel: str,
    timestep_s: float,
    retained_rows: int,
    label: str,
    row_id: str,
    source_kind: str,
) -> dict[str, object]:
    data = read_qblade_results(path, required_columns=[channel])
    if len(data) != retained_rows:
        raise HVAWTSensitivityError(
            f"{label} retained {len(data)} rows; expected {retained_rows}."
        )
    values = data[channel].to_numpy(float)
    if np.any(~np.isfinite(values)):
        raise HVAWTSensitivityError(f"{label} target load is nonfinite.")
    damage, cycles, _ranges, _counts = rainflow_damage(
        values, WOHLER_EXPONENT
    )
    if damage <= 0.0 or cycles < 10.0:
        raise HVAWTSensitivityError(
            f"{label} target load has insufficient positive rainflow content."
        )
    full_del = float((damage / EQUIVALENT_CYCLES) ** (1.0 / WOHLER_EXPONENT))
    count_480 = int(round(480.0 / timestep_s))
    if count_480 <= 0 or count_480 >= len(values):
        raise HVAWTSensitivityError(f"{label} cannot form 480/600 s windows.")
    damage_480, cycles_480, _ranges, _counts = rainflow_damage(
        values[:count_480], WOHLER_EXPONENT
    )
    if damage_480 <= 0.0 or cycles_480 < 10.0:
        raise HVAWTSensitivityError(
            f"{label} 480 s prefix has insufficient rainflow content."
        )
    del_480 = float((damage_480 / 480.0) ** (1.0 / WOHLER_EXPONENT))
    extreme_480 = float(np.max(np.abs(values[:count_480])))
    extreme_600 = float(np.max(np.abs(values)))
    if min(full_del, del_480, extreme_480, extreme_600) <= 0.0:
        raise HVAWTSensitivityError(f"{label} target metric is nonpositive.")
    return {
        "row_id": row_id,
        "label": label,
        "source_kind": source_kind,
        "channel": channel,
        "timestep_s": timestep_s,
        "retained_rows": retained_rows,
        "wohler_exponent": WOHLER_EXPONENT,
        "equivalent_cycles": EQUIVALENT_CYCLES,
        "rainflow_cycles": cycles,
        "del_Nm": full_del,
        "minimum_Nm": float(np.min(values)),
        "maximum_Nm": float(np.max(values)),
        "monitored_channel_absolute_extreme_Nm": extreme_600,
        "damage_rate_del_480_Nm": del_480,
        "damage_rate_del_600_Nm": full_del,
        "monitored_channel_absolute_extreme_480_Nm": extreme_480,
        "monitored_channel_absolute_extreme_600_Nm": extreme_600,
        "finite_record_del_relative_change_480_to_600s": abs(
            full_del / del_480 - 1.0
        ),
        "finite_record_monitored_extreme_relative_change_480_to_600s": abs(
            extreme_600 / extreme_480 - 1.0
        ),
        "result": _relative_identity(path, identity_root),
    }


def _symmetric_change(left: float, right: float) -> float:
    if left <= 0.0 or right <= 0.0:
        raise HVAWTSensitivityError("Ratio comparison requires positive values.")
    return abs(math.log(left / right))


def _gate(
    gates: list[dict[str, object]],
    *,
    row_id: str,
    family: str,
    metric: str,
    observed: float,
    limit: float,
    comparison: str,
) -> None:
    if not math.isfinite(observed):
        raise HVAWTSensitivityError(f"Gate {family}/{metric} is nonfinite.")
    gates.append(
        {
            "row_id": row_id,
            "family": family,
            "metric": metric,
            "comparison": comparison,
            "observed": observed,
            "limit": limit,
            "passed": observed <= limit,
        }
    )


def _ratio_row(
    *,
    row_id: str,
    setting: str,
    raw: Mapping[str, Any],
    one_way: Mapping[str, Any],
    complete: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "setting": setting,
        "raw_del_Nm": raw["del_Nm"],
        "one_way_del_Nm": one_way["del_Nm"],
        "complete_del_Nm": complete["del_Nm"],
        "one_way_to_raw": float(one_way["del_Nm"]) / float(raw["del_Nm"]),
        "complete_to_raw": float(complete["del_Nm"]) / float(raw["del_Nm"]),
        "complete_to_one_way": float(complete["del_Nm"])
        / float(one_way["del_Nm"]),
    }


def _append_complete_path_numerical_load_gates(
    gates: list[dict[str, object]],
    *,
    row_id: str,
    metric_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    """Gate only timestep/cap perturbations; response-off is an ablation."""

    log_load_limit = math.log1p(LOAD_RELATIVE_LIMIT)
    nominal_metric = metric_by_key[(row_id, "nominal_complete")]
    for variant_label in (
        "complete_half_timestep",
        "complete_lower_cap",
    ):
        variant_metric = metric_by_key[(row_id, variant_label)]
        for metric_name in (
            "del_Nm",
            "monitored_channel_absolute_extreme_Nm",
        ):
            _gate(
                gates,
                row_id=row_id,
                family="complete_path_numerical_load_sensitivity",
                metric=f"{variant_label}_{metric_name}",
                observed=_symmetric_change(
                    float(variant_metric[metric_name]),
                    float(nominal_metric[metric_name]),
                ),
                limit=log_load_limit,
                comparison=f"{variant_label}_vs_nominal_complete_log_ratio",
            )


def _append_finite_record_sensitivity_gates(
    gates: list[dict[str, object]],
    *,
    row_id: str,
    family: str,
    label: str,
    metric: Mapping[str, Any],
) -> None:
    """Compare duration-normalized DEL and record maxima at 480 and 600 s."""

    _gate(
        gates,
        row_id=row_id,
        family=family,
        metric=f"{label}_damage_rate_del",
        observed=float(
            metric["finite_record_del_relative_change_480_to_600s"]
        ),
        limit=FINAL_120_RELATIVE_LIMIT,
        comparison="duration_normalized_damage_rate_del_480_to_600s",
    )
    _gate(
        gates,
        row_id=row_id,
        family=family,
        metric=f"{label}_monitored_channel_absolute_extreme",
        observed=float(
            metric[
                "finite_record_monitored_extreme_relative_change_480_to_600s"
            ]
        ),
        limit=FINAL_120_RELATIVE_LIMIT,
        comparison="monitored_record_maximum_magnitude_480_to_600s",
    )


def _append_modal_diagnostics_and_gates(
    gates: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    *,
    row_id: str,
    nominal_history: Mapping[str, Any],
    variant_histories: Mapping[str, Mapping[str, Any]],
) -> None:
    """Record activity applicability and gate only meaningful frequencies."""

    nominal_payload = nominal_history.get("modes")
    if not isinstance(nominal_payload, Mapping):
        raise HVAWTSensitivityError(
            "Nominal complete history lacks modal diagnostics."
        )
    nominal_modes: dict[int, dict[str, object]] = {}
    for mode, natural_frequency in MODE_FREQUENCIES_HZ.items():
        nominal_mode = nominal_payload.get(str(mode))
        if not isinstance(nominal_mode, Mapping):
            raise HVAWTSensitivityError(
                f"Nominal complete history lacks mode {mode}."
            )
        activation = float(nominal_mode.get("maximum_activation", math.nan))
        frequency = float(nominal_mode.get("dominant_frequency_hz", math.nan))
        if (
            not math.isfinite(activation)
            or activation < 0.0
            or not math.isfinite(frequency)
            or frequency <= 0.0
        ):
            raise HVAWTSensitivityError(
                "Nominal complete modal diagnostic is invalid."
            )
        active = activation > MODE_ACTIVE_DAMPING_RATIO_THRESHOLD
        nominal_modes[mode] = {
            "activation": activation,
            "frequency_hz": frequency,
            "active": active,
        }
        if active:
            _gate(
                gates,
                row_id=row_id,
                family="internal_filtered_modal_velocity_frequency_band",
                metric=f"nominal_complete_mode{mode}_natural_frequency",
                observed=abs(frequency / natural_frequency - 1.0),
                limit=MODE_ABSOLUTE_BAND_LIMIT,
                comparison=(
                    "filtered_modal_velocity_dominant_vs_reviewed_structural_"
                    "frequency_not_physical_lockin_validation"
                ),
            )

    for variant_label, variant_history in variant_histories.items():
        variant_payload = variant_history.get("modes")
        if not isinstance(variant_payload, Mapping):
            raise HVAWTSensitivityError(
                f"{variant_label} lacks modal diagnostics."
            )
        for mode, natural_frequency in MODE_FREQUENCIES_HZ.items():
            nominal_mode = nominal_modes[mode]
            variant_mode = variant_payload.get(str(mode))
            if not isinstance(variant_mode, Mapping):
                raise HVAWTSensitivityError(
                    f"{variant_label} lacks mode {mode}."
                )
            variant_activation = float(
                variant_mode.get("maximum_activation", math.nan)
            )
            variant_frequency = float(
                variant_mode.get("dominant_frequency_hz", math.nan)
            )
            if (
                not math.isfinite(variant_activation)
                or variant_activation < 0.0
                or not math.isfinite(variant_frequency)
                or variant_frequency <= 0.0
            ):
                raise HVAWTSensitivityError(
                    f"{variant_label} modal diagnostic is invalid."
                )
            nominal_active = bool(nominal_mode["active"])
            variant_active = (
                variant_activation > MODE_ACTIVE_DAMPING_RATIO_THRESHOLD
            )
            _gate(
                gates,
                row_id=row_id,
                family="complete_path_active_mode_classification_consistency",
                metric=f"{variant_label}_mode{mode}_active_classification",
                observed=float(nominal_active != variant_active),
                limit=0.0,
                comparison=(
                    "engineering_aerodynamic_to_structural_damping_ratio"
                    f"_above_{MODE_ACTIVE_DAMPING_RATIO_THRESHOLD:g}"
                ),
            )
            if variant_active:
                _gate(
                    gates,
                    row_id=row_id,
                    family="internal_filtered_modal_velocity_frequency_band",
                    metric=f"{variant_label}_mode{mode}_natural_frequency",
                    observed=abs(variant_frequency / natural_frequency - 1.0),
                    limit=MODE_ABSOLUTE_BAND_LIMIT,
                    comparison=(
                        "filtered_modal_velocity_dominant_vs_reviewed_structural_"
                        "frequency_not_physical_lockin_validation"
                    ),
                )
            intervariant_applicable = nominal_active and variant_active
            if intervariant_applicable:
                _gate(
                    gates,
                    row_id=row_id,
                    family=(
                        "complete_path_filtered_modal_velocity_frequency_"
                        "sensitivity"
                    ),
                    metric=f"{variant_label}_mode{mode}_intervariant",
                    observed=abs(
                        variant_frequency
                        / float(nominal_mode["frequency_hz"])
                        - 1.0
                    ),
                    limit=MODE_RELATIVE_LIMIT,
                    comparison=f"{variant_label}_vs_nominal_complete",
                )
            if intervariant_applicable:
                not_applicable_reason = None
            elif nominal_active:
                not_applicable_reason = "variant_below_activation_threshold"
            elif variant_active:
                not_applicable_reason = "nominal_below_activation_threshold"
            else:
                not_applicable_reason = (
                    "nominal_and_variant_below_activation_threshold"
                )
            diagnostics.append(
                {
                    "row_id": row_id,
                    "mode": mode,
                    "comparison_variant": variant_label,
                    "reviewed_structural_frequency_hz": natural_frequency,
                    "activation_threshold": MODE_ACTIVE_DAMPING_RATIO_THRESHOLD,
                    "nominal_maximum_activation": nominal_mode["activation"],
                    "variant_maximum_activation": variant_activation,
                    "nominal_active": nominal_active,
                    "variant_active": variant_active,
                    "active_classification_matches": (
                        nominal_active == variant_active
                    ),
                    "nominal_dominant_frequency_hz": nominal_mode[
                        "frequency_hz"
                    ],
                    "variant_dominant_frequency_hz": variant_frequency,
                    "nominal_absolute_band_gate_applied": nominal_active,
                    "variant_absolute_band_gate_applied": variant_active,
                    "intervariant_frequency_gate_applied": (
                        intervariant_applicable
                    ),
                    "intervariant_not_applicable_reason": not_applicable_reason,
                    "interpretation": (
                        "Internal filtered-modal-velocity consistency; not "
                        "physical lock-in validation."
                    ),
                }
            )


def _base_history_metrics(
    *,
    base_root: Path,
    baseline_records: Mapping[str, Any],
) -> dict[str, object]:
    history_path = _verify_relative_identity(
        base_root,
        baseline_records.get("history"),
        label="nominal complete compact history",
    )
    return analyze_history(
        history_path,
        tail_start_s=STORE_FROM_S,
        tail_duration_s=DURATION_S - STORE_FROM_S,
    )


def analyze_campaign(
    campaign: Path,
    *,
    base_campaign_root: Path,
    output: Path | None = None,
) -> dict[str, object]:
    """Audit all new cases and evaluate preregistered screening diagnostics."""

    state = validate_campaign(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    base = validate_base_preflight(base_campaign_root)
    base_root = base["base_campaign_root"]
    if base["manifest"]["fingerprint"] != manifest["base_campaign_fingerprint"]:
        raise HVAWTSensitivityError("Analysis base campaign fingerprint changed.")
    terminal_cases: list[dict[str, object]] = []
    terminal_by_variant: dict[tuple[str, str], Mapping[str, Any]] = {}
    for case in manifest["cases"]:
        classification = classify_case(root, case_index=int(case["index"]))
        if classification["status"] != "complete":
            raise HVAWTSensitivityError(
                f"Sensitivity case is not complete: {classification}"
            )
        record = {
            "case_index": case["index"],
            "case_id": case["case_id"],
            "row_id": case["row_id"],
            "variant": case["variant"],
            **classification,
        }
        terminal_cases.append(record)
        terminal_by_variant[(str(case["row_id"]), str(case["variant"]))] = record

    metrics: list[dict[str, object]] = []
    metric_by_key: dict[tuple[str, str], dict[str, object]] = {}
    ratios: list[dict[str, object]] = []
    gates: list[dict[str, object]] = []
    attribution: list[dict[str, object]] = []
    modal_diagnostics: list[dict[str, object]] = []
    baseline_records_all = manifest.get("baseline_results")
    if not isinstance(baseline_records_all, Mapping):
        raise HVAWTSensitivityError("Baseline result identities are missing.")
    case_by_key = {
        (str(case["row_id"]), str(case["variant"])): case
        for case in manifest["cases"]
    }
    for selected in SELECTED_ROWS:
        row_id = str(selected["row_id"])
        channel = (
            f"{selected['channel_axis']}_l~Mom.~BLD_1~pos~0.230~[Nm]"
        )
        baseline_records = baseline_records_all.get(row_id)
        if not isinstance(baseline_records, Mapping):
            raise HVAWTSensitivityError(
                f"Baseline result identities are missing for {row_id}."
            )
        for scenario, label in (
            ("raw_qblade", "nominal_raw"),
            ("one_way_spectral_vorlap", "nominal_one_way"),
            ("complete_hybrid_v8", "nominal_complete"),
        ):
            records = baseline_records.get(scenario)
            if not isinstance(records, Mapping):
                raise HVAWTSensitivityError(
                    f"Baseline {scenario} identities are missing."
                )
            file_roles = {"result", "summary", "metadata"}
            if scenario != "raw_qblade":
                file_roles.add("history")
            if set(records) != file_roles | {"frozen_selected_channel_del_Nm"}:
                raise HVAWTSensitivityError(
                    f"Baseline {scenario} evidence roles changed."
                )
            for role in sorted(file_roles):
                _verify_relative_identity(
                    base_root,
                    records[role],
                    label=f"{row_id} nominal {scenario} {role}",
                )
            result_path = _verify_relative_identity(
                base_root,
                records.get("result"),
                label=f"{row_id} nominal {scenario} result",
            )
            metric = _load_metric(
                path=result_path,
                identity_root=base_root,
                channel=channel,
                timestep_s=TIMESTEP_S,
                retained_rows=NOMINAL_RETAINED_ROWS,
                label=label,
                row_id=row_id,
                source_kind="accepted_v3",
            )
            frozen_del = records["frozen_selected_channel_del_Nm"]
            if (
                not isinstance(frozen_del, Mapping)
                or frozen_del.get("channel") != channel
                or frozen_del.get("wohler_exponent") != WOHLER_EXPONENT
                or frozen_del.get("equivalent_cycles") != EQUIVALENT_CYCLES
                or not math.isclose(
                    float(metric["del_Nm"]),
                    float(frozen_del.get("value", math.nan)),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                )
            ):
                raise HVAWTSensitivityError(
                    "Baseline selected-channel DEL no longer matches its frozen value."
                )
            metrics.append(metric)
            metric_by_key[(row_id, label)] = metric
        _append_finite_record_sensitivity_gates(
            gates,
            row_id=row_id,
            family="complete_path_nominal_finite_record_sensitivity",
            label="nominal_complete",
            metric=metric_by_key[(row_id, "nominal_complete")],
        )
        for variant in VARIANTS:
            name = str(variant["name"])
            case = case_by_key[(row_id, name)]
            terminal = terminal_by_variant[(row_id, name)]
            receipt = terminal["receipt"]
            result_path = _verify_relative_identity(
                root,
                receipt["result"],
                label=f"{row_id} {name} result",
            )
            metric = _load_metric(
                path=result_path,
                identity_root=root,
                channel=channel,
                timestep_s=float(case["timestep_s"]),
                retained_rows=int(case["retained_rows"]),
                label=name,
                row_id=row_id,
                source_kind="targeted_sensitivity",
            )
            metrics.append(metric)
            metric_by_key[(row_id, name)] = metric
            finite_record_family = (
                "response_ablation_finite_record_sensitivity"
                if name == "complete_response_off"
                else "complete_path_finite_record_sensitivity"
            )
            _append_finite_record_sensitivity_gates(
                gates,
                row_id=row_id,
                family=finite_record_family,
                label=name,
                metric=metric,
            )

        nominal = _ratio_row(
            row_id=row_id,
            setting="nominal",
            raw=metric_by_key[(row_id, "nominal_raw")],
            one_way=metric_by_key[(row_id, "nominal_one_way")],
            complete=metric_by_key[(row_id, "nominal_complete")],
        )
        half = _ratio_row(
            row_id=row_id,
            setting="half_timestep_complete_with_nominal_denominators",
            raw=metric_by_key[(row_id, "nominal_raw")],
            one_way=metric_by_key[(row_id, "nominal_one_way")],
            complete=metric_by_key[(row_id, "complete_half_timestep")],
        )
        lower = _ratio_row(
            row_id=row_id,
            setting="lower_cap_complete_with_nominal_denominators",
            raw=metric_by_key[(row_id, "nominal_raw")],
            one_way=metric_by_key[(row_id, "nominal_one_way")],
            complete=metric_by_key[(row_id, "complete_lower_cap")],
        )
        response_off = _ratio_row(
            row_id=row_id,
            setting="response_off_complete_with_nominal_denominators",
            raw=metric_by_key[(row_id, "nominal_raw")],
            one_way=metric_by_key[(row_id, "nominal_one_way")],
            complete=metric_by_key[(row_id, "complete_response_off")],
        )
        for row in (half, lower, response_off):
            row["denominator_scope"] = (
                "accepted nominal raw and one-way reused; contextual numerator "
                "sensitivity, not matched three-path numerical convergence"
            )
            row["diagnostic_only"] = True
            row["complete_numerator_del_log_change_from_nominal"] = (
                _symmetric_change(
                    float(row["complete_del_Nm"]),
                    float(nominal["complete_del_Nm"]),
                )
            )
        nominal["denominator_scope"] = "accepted matched nominal three-path result"
        nominal["diagnostic_only"] = False
        ratios.extend((nominal, response_off, half, lower))
        _append_complete_path_numerical_load_gates(
            gates,
            row_id=row_id,
            metric_by_key=metric_by_key,
        )

        complete_records = baseline_records["complete_hybrid_v8"]
        baseline_history = _base_history_metrics(
            base_root=base_root,
            baseline_records=complete_records,
        )
        variant_histories: dict[str, Mapping[str, Any]] = {}
        for variant_label in ("complete_half_timestep", "complete_lower_cap"):
            variant_history = terminal_by_variant[
                (row_id, variant_label)
            ]["receipt"]["history_metrics"]
            if not isinstance(variant_history, Mapping):
                raise HVAWTSensitivityError(
                    f"{variant_label} lacks complete history metrics."
                )
            variant_histories[variant_label] = variant_history
        _append_modal_diagnostics_and_gates(
            gates,
            modal_diagnostics,
            row_id=row_id,
            nominal_history=baseline_history,
            variant_histories=variant_histories,
        )
        metric_keys = (
            ("del_Nm", "del"),
            ("monitored_channel_absolute_extreme_Nm", "monitored_extreme"),
        )
        effect_values: dict[str, object] = {"row_id": row_id}
        for metric_name, short_name in metric_keys:
            nominal_value = float(
                metric_by_key[(row_id, "nominal_complete")][metric_name]
            )
            response_off_value = float(
                metric_by_key[(row_id, "complete_response_off")][metric_name]
            )
            half_value = float(
                metric_by_key[(row_id, "complete_half_timestep")][metric_name]
            )
            lower_value = float(
                metric_by_key[(row_id, "complete_lower_cap")][metric_name]
            )
            response_delta = _symmetric_change(response_off_value, nominal_value)
            numerical_delta = max(
                _symmetric_change(half_value, nominal_value),
                _symmetric_change(lower_value, nominal_value),
            )
            effect_values[f"response_off_{short_name}_log_effect"] = response_delta
            effect_values[
                f"two_tested_complete_path_{short_name}_log_envelope"
            ] = numerical_delta
            effect_values[
                f"response_off_{short_name}_effect_larger_than_two_tested_"
                "complete_path_variations"
            ] = response_delta > numerical_delta
        effect_values["interpretation"] = (
            "response-work ablation within the complete residual partition; "
            "effect magnitude is diagnostic, not a closeness gate or isolated "
            "negative-damping validation"
        )
        attribution.append(
            effect_values
        )

    modal_intervariant_count = sum(
        row["intervariant_frequency_gate_applied"] is True
        for row in modal_diagnostics
    )
    modal_applicability: dict[str, object] = {
        "diagnostic_record_count": len(modal_diagnostics),
        "intervariant_gate_applied_count": modal_intervariant_count,
        "intervariant_gate_not_applied_count": (
            len(modal_diagnostics) - modal_intervariant_count
        ),
        "interpretation": (
            "Internal filtered-modal-velocity consistency diagnostic; no "
            "physical lock-in validation."
        ),
        "intervariant_applicability_statement": (
            "Applied only where the same reviewed mode exceeded the activation "
            "threshold in both nominal and variant histories."
            if modal_intervariant_count
            else "Not applicable: no reviewed mode exceeded the activation "
            "threshold in both nominal and variant histories."
        ),
    }
    passed_count = sum(row["passed"] is True for row in gates)
    numerical_gates = [
        row
        for row in gates
        if not str(row["family"]).startswith("response_ablation_")
    ]
    response_gates = [
        row
        for row in gates
        if str(row["family"]).startswith("response_ablation_")
    ]
    numerical_passed = bool(numerical_gates) and all(
        row["passed"] is True for row in numerical_gates
    )
    response_diagnostics_passed = bool(response_gates) and all(
        row["passed"] is True for row in response_gates
    )
    all_screening_gates_passed = numerical_passed and response_diagnostics_passed
    report: dict[str, object] = {
        "schema": ANALYSIS_SCHEMA,
        "status": (
            "completed_targeted_screen_with_complete_path_numerical_gates_"
            "passed_and_response_ablation_audited"
            if all_screening_gates_passed
            else "completed_targeted_screen_with_failed_preregistered_diagnostics"
        ),
        "campaign_fingerprint": manifest["fingerprint"],
        "base_campaign_fingerprint": BASE_CAMPAIGN_FINGERPRINT,
        "complete_case_count": len(terminal_cases),
        "expected_case_count": 6,
        "all_execution_audits_passed": True,
        "response_ablation_execution_audits_passed": True,
        "all_complete_path_numerical_gates_passed": numerical_passed,
        "all_response_ablation_diagnostics_passed": response_diagnostics_passed,
        "all_preregistered_screening_gates_passed": all_screening_gates_passed,
        "response_ablation_effect_closeness_gate_applied": False,
        "passed_gate_count": passed_count,
        "gate_count": len(gates),
        "complete_path_numerical_gate_count": len(numerical_gates),
        "response_ablation_diagnostic_gate_count": len(response_gates),
        "terminal_cases": terminal_cases,
        "metrics": metrics,
        "ratio_records": ratios,
        "gates": gates,
        "modal_diagnostics": modal_diagnostics,
        "modal_applicability": modal_applicability,
        "response_attribution": attribution,
        "claim_boundary": manifest["claim_boundary"],
    }
    report["fingerprint"] = canonical_fingerprint(report)
    target = (
        root / ANALYSIS_NAME
        if output is None
        else Path(output).expanduser().resolve()
    )
    try:
        target.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise HVAWTSensitivityError(
            "Sensitivity analysis output must remain inside the campaign root."
        ) from exc
    return _write_new_or_exact_json(
        target,
        report,
        label="sensitivity analysis",
        mismatch_message=(
            "Existing sensitivity analysis differs; use a versioned output."
        ),
    )


def _write_new_or_exact_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
    mismatch_message: str,
) -> dict[str, Any]:
    """Atomically create a JSON artifact or accept only an exact retry."""

    if path.exists():
        existing = _load_json(path, label=f"existing {label}")
        if existing != payload:
            raise HVAWTSensitivityError(mismatch_message)
        return existing
    value = dict(payload)
    atomic_write_json(path, value)
    return value


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise HVAWTSensitivityError(f"Refusing to write empty CSV: {path}")
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def _planned_relative_identity(
    staged_path: Path, final_path: Path, root: Path
) -> dict[str, object]:
    """Describe staged bytes at their final campaign-relative destination."""

    if not staged_path.is_file():
        raise HVAWTSensitivityError(
            f"Staged promotion output is missing: {staged_path}"
        )
    final = final_path.resolve(strict=False)
    try:
        relative = final.relative_to(root.resolve())
    except ValueError as exc:
        raise HVAWTSensitivityError(
            "Sensitivity promotion output escapes the campaign root."
        ) from exc
    return {
        "path": relative.as_posix(),
        "size_bytes": staged_path.stat().st_size,
        "sha256": _sha256(staged_path),
    }


def _validate_existing_promotion(
    *,
    target: Path,
    root: Path,
    manifest: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    analysis: Mapping[str, Any],
    final_outputs: Mapping[str, Path],
) -> dict[str, object]:
    """Return an exact idempotent promotion, or fail before changing bytes."""

    if not target.is_dir() or target.is_symlink():
        raise HVAWTSensitivityError(
            "Existing sensitivity promotion target is not a plain directory."
        )
    receipt_path = target / PROMOTION_RECEIPT_NAME
    receipt = _load_json(
        receipt_path, label="existing sensitivity promotion receipt"
    )
    _validate_fingerprint(receipt, label="Existing sensitivity promotion")
    expected_status = (
        "promoted_complete_path_numerical_pass_and_audited_response_ablation"
        if analysis["all_preregistered_screening_gates_passed"]
        else "promoted_claim_bounded_failed_diagnostic_sensitivity_screen"
    )
    outputs = receipt.get("outputs")
    if (
        receipt.get("schema") != PROMOTION_SCHEMA
        or receipt.get("status") != expected_status
        or receipt.get("campaign_fingerprint") != manifest["fingerprint"]
        or receipt.get("analysis") != source_identity
        or receipt.get("analysis_fingerprint") != analysis["fingerprint"]
        or receipt.get("claim_boundary") != analysis["claim_boundary"]
        or not isinstance(outputs, Mapping)
        or set(outputs) != set(final_outputs)
    ):
        raise HVAWTSensitivityError(
            "Existing sensitivity promotion differs from this analysis."
        )
    for role, final_path in final_outputs.items():
        observed = _verify_relative_identity(
            root, outputs[role], label=f"promoted sensitivity {role}"
        )
        if observed != final_path.resolve():
            raise HVAWTSensitivityError(
                f"Promoted sensitivity {role} path changed."
            )
    return receipt


def promote_analysis(
    campaign: Path,
    *,
    analysis_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Promote a complete pass or bounded failed-gate screen by exact hashes."""

    state = validate_campaign(campaign, rehash_case_assets=False)
    root = state["campaign_root"]
    manifest = state["manifest"]
    source = (
        root / ANALYSIS_NAME
        if analysis_path is None
        else Path(analysis_path).expanduser().resolve()
    )
    analysis = _load_json(source, label="sensitivity analysis")
    _validate_fingerprint(analysis, label="Sensitivity analysis")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("campaign_fingerprint") != manifest["fingerprint"]
        or analysis.get("complete_case_count") != 6
        or analysis.get("expected_case_count") != 6
        or analysis.get("all_execution_audits_passed") is not True
        or analysis.get("response_ablation_execution_audits_passed") is not True
        or analysis.get("response_ablation_effect_closeness_gate_applied")
        is not False
        or not isinstance(analysis.get("modal_diagnostics"), list)
        or not analysis["modal_diagnostics"]
        or not isinstance(analysis.get("modal_applicability"), Mapping)
        or analysis.get("status")
        not in {
            "completed_targeted_screen_with_complete_path_numerical_gates_"
            "passed_and_response_ablation_audited",
            "completed_targeted_screen_with_failed_preregistered_diagnostics",
        }
    ):
        raise HVAWTSensitivityError("Sensitivity analysis is not promotable.")
    try:
        source.relative_to(root.resolve())
    except ValueError as exc:
        raise HVAWTSensitivityError(
            "Sensitivity analysis must be inside the campaign root."
        ) from exc
    source_identity = _relative_identity(source, root)
    target_candidate = (
        root / PAPER_EXPORT_NAME
        if output_dir is None
        else Path(output_dir).expanduser()
    )
    if target_candidate.is_symlink():
        raise HVAWTSensitivityError(
            "Sensitivity promotion target may not be a symlink."
        )
    target = target_candidate.resolve(strict=False)
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise HVAWTSensitivityError(
            "Sensitivity promotion target escapes the campaign root."
        ) from exc
    if target == root.resolve() or not target.parent.is_dir():
        raise HVAWTSensitivityError(
            "Sensitivity promotion requires an existing campaign-owned parent."
        )
    filenames = {
        "metrics": "hvawt_sensitivity_metrics_v1.csv",
        "gates": "hvawt_sensitivity_gates_v1.csv",
        "ratio_context": "hvawt_sensitivity_ratio_context_v1.csv",
        "modal_diagnostics": "hvawt_sensitivity_modal_diagnostics_v1.csv",
        "response_attribution": "hvawt_response_attribution_v1.csv",
        "summary": "hvawt_sensitivity_summary_v1.json",
    }
    final_outputs = {
        role: target / filename for role, filename in filenames.items()
    }
    if target.exists():
        return _validate_existing_promotion(
            target=target,
            root=root,
            manifest=manifest,
            source_identity=source_identity,
            analysis=analysis,
            final_outputs=final_outputs,
        )
    stage = target.with_name(
        f".{target.name}.preparing.{os.getpid()}."
        f"{str(analysis['fingerprint'])[:12]}"
    )
    if stage.exists():
        raise HVAWTSensitivityError(
            f"Sensitivity promotion staging path already exists: {stage}"
        )
    stage.mkdir()
    try:
        staged_outputs = {
            role: stage / filename for role, filename in filenames.items()
        }
        _write_csv(staged_outputs["metrics"], list(analysis["metrics"]))
        _write_csv(staged_outputs["gates"], list(analysis["gates"]))
        _write_csv(
            staged_outputs["ratio_context"], list(analysis["ratio_records"])
        )
        _write_csv(
            staged_outputs["modal_diagnostics"],
            list(analysis["modal_diagnostics"]),
        )
        _write_csv(
            staged_outputs["response_attribution"],
            list(analysis["response_attribution"]),
        )
        summary: dict[str, object] = {
            "schema": "vorlap-hvawt-targeted-sensitivity-summary-v1",
            "status": analysis["status"],
            "analysis_fingerprint": analysis["fingerprint"],
            "all_complete_path_numerical_gates_passed": analysis[
                "all_complete_path_numerical_gates_passed"
            ],
            "all_response_ablation_diagnostics_passed": analysis[
                "all_response_ablation_diagnostics_passed"
            ],
            "response_ablation_execution_audits_passed": True,
            "all_preregistered_screening_gates_passed": analysis[
                "all_preregistered_screening_gates_passed"
            ],
            "response_ablation_effect_closeness_gate_applied": False,
            "passed_gate_count": analysis["passed_gate_count"],
            "gate_count": analysis["gate_count"],
            "modal_applicability": analysis["modal_applicability"],
            "ratio_records": analysis["ratio_records"],
            "response_attribution": analysis["response_attribution"],
            "claim_boundary": analysis["claim_boundary"],
        }
        summary["fingerprint"] = canonical_fingerprint(summary)
        atomic_write_json(staged_outputs["summary"], summary)
        outputs = {
            role: _planned_relative_identity(
                staged_outputs[role], final_outputs[role], root
            )
            for role in filenames
        }
        receipt: dict[str, object] = {
            "schema": PROMOTION_SCHEMA,
            "status": (
                "promoted_complete_path_numerical_pass_and_audited_response_ablation"
                if analysis["all_preregistered_screening_gates_passed"]
                else "promoted_claim_bounded_failed_diagnostic_sensitivity_screen"
            ),
            "campaign_fingerprint": manifest["fingerprint"],
            "analysis": source_identity,
            "analysis_fingerprint": analysis["fingerprint"],
            "outputs": outputs,
            "claim_boundary": analysis["claim_boundary"],
        }
        receipt["fingerprint"] = canonical_fingerprint(receipt)
        atomic_write_json(stage / PROMOTION_RECEIPT_NAME, receipt)
        os.replace(stage, target)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return _validate_existing_promotion(
        target=target,
        root=root,
        manifest=manifest,
        source_identity=source_identity,
        analysis=analysis,
        final_outputs=final_outputs,
    )


__all__ = [
    "ANALYSIS_NAME",
    "ANALYSIS_SCHEMA",
    "AUTHORIZATION_NAME",
    "CAMPAIGN_SCHEMA",
    "HVAWTSensitivityError",
    "LOWER_CAP_HZ",
    "MANIFEST_NAME",
    "PAPER_EXPORT_NAME",
    "PROMOTION_RECEIPT_NAME",
    "SELECTED_ROWS",
    "SMOKE_RECEIPT_NAME",
    "VARIANTS",
    "bind_runtime",
    "analyze_campaign",
    "build_runtime_identity",
    "classify_case",
    "command_for_case",
    "execute_case",
    "execute_smoke",
    "generate_campaign",
    "promote_analysis",
    "bind_execution_authorization",
    "validate_base_preflight",
    "validate_campaign",
]
