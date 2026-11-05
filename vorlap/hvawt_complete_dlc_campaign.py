"""Sealed matched raw/one-way/complete H-VAWT TurbSim campaign.

The archived raw and one-way results remain useful historical evidence, but
they do not share the corrected structural damping, startup settings, or
static-spectrum identity of the complete H-VAWT model.  This module therefore
materializes all three scenarios from one common source for every approved DLC
row.  Runs start fresh and may write bounded recovery snapshots; continuation
from those snapshots remains disabled because project restart has not yet
preserved the coupled structural state reliably for these cases.
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
from typing import Any, Mapping, Sequence

from .dlc_manifest import (
    load_manifest as load_dlc_manifest,
    sha256_payload,
    validate_wind_record,
)
from .hvawt_response_pilot import (
    ANALYSIS_SCHEMA as PILOT_ANALYSIS_SCHEMA,
    FLIGHT_RESTART_SAVE_INTERVAL_STEPS,
    FULL_CAMPAIGN_KIND,
    FULL_RERUN_SCHEMA,
    FULL_RERUN_SCENARIOS,
    HVAWTResponsePilotError,
    RESULT_FILTER_CHANNEL_COUNT,
    RESULT_FILTER_SHA256,
    RESULT_FILTER_SOURCE_RELATIVE,
    _apply_engineering_rayleigh_inputs,
    _audit_qblade_fatigue_result,
    _identity,
    _load_json,
    _rayleigh_source_case,
    _result_filter_channels,
    _stage_calibrated_static_spectrum,
    _verify_identity,
    _history_path_from_summary,
    analyze_history,
    validate_campaign as validate_pilot_campaign,
    validate_campaign_spec,
    validate_response_model,
)
from .qblade_batch import (
    qblade_aerodynamic_reference_issues,
    qblade_filter_channels,
    turbine_definition_path_for_sim,
    vorlap_airfoil_dir_for_sim,
    wake_profile_settings,
    write_sil_storage_filter,
)
from .qblade_case_prep import QBladeCasePrepRequest, prepare_qblade_external_case
from .qblade_files import read_qblade_keyword, replace_qblade_keyword_file
from .qblade_external_config import (
    strip_qblade_external_library_section,
    strip_qblade_external_tables,
)
from .qblade_run_support import (
    atomic_write_json,
    canonical_fingerprint,
    find_bridge_library,
    find_qblade_library,
    qblade_runtime_asset_fingerprint,
    qblade_runtime_dependency_manifest,
    require_zero_native_callback_evidence,
    vorlap_python_runtime_manifest,
)
from .qblade_source_coverage import (
    HVAWT_V8_COMPONENT_NODE_COUNTS,
    validate_complete_source_coverage,
    validate_complete_velocity_diagnostics,
)

CAMPAIGN_SCHEMA = "vorlap-hvawt-complete-dlc-campaign-v1"
RUNTIME_SCHEMA = "vorlap-hvawt-complete-dlc-flight-runtime-v1"
ANALYSIS_SCHEMA = "vorlap-hvawt-complete-dlc-analysis-v2"
APPROVED_MANIFEST_RELATIVE = Path("run_headless/config/parked_dlc_matrix_approved.json")
APPROVED_MANIFEST_FILE_SHA256 = (
    "e71b38cd336e4db5ef2eae00b43a3d4f3e41a7cec2c0866e87165a765d456fc9"
)
APPROVED_MANIFEST_CANONICAL_SHA256 = (
    "75ff52e6ef7a1b4a52c57149cba1ba85b26b36968bfddc53b5279120c7d43c0b"
)
CAMPAIGN_SPEC_RELATIVE = Path("run_headless/config/hvawt_response_campaign_spec.json")
TURBULENT_CHECKPOINT_POLICY_RELATIVE = Path(
    "run_headless/config/hvawt_turbulent_20case_checkpoint_policy.json"
)
UNIFORM_EVIDENCE_RECEIPT_RELATIVE = Path(
    "paper_drafts/NAWEA_QBladeBridge/analysis/hvawt_uniform_pilot_paper_receipt.json"
)
RESPONSE_MODEL_RELATIVE = Path(
    "run_headless/config/hvawt_response_model_v8_complete.json"
)
QBLADE_LIBRARY_SHA256 = (
    "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
)
BRIDGE_LIBRARY_SHA256 = (
    "e5988ccc2f023714be9dd40c44359a8312f97fd1e65cb9e741456da2a0142a07"
)
SCENARIOS = (
    "raw_qblade",
    "one_way_spectral_vorlap",
    "complete_hybrid_v8",
)
SCENARIO_FORCE_CONTRACT = {
    SCENARIOS[0]: {
        "coupling_mode": "none",
        "legacy_vorlap_force_scale": 0.0,
        "residual_source_scale": 0.0,
        "response_force_scale": 0.0,
    },
    SCENARIOS[1]: {
        "coupling_mode": "sil-vorlap",
        "legacy_vorlap_force_scale": 1.0,
        "residual_source_scale": 0.0,
        "response_force_scale": 0.0,
    },
    SCENARIOS[2]: {
        "coupling_mode": "sil-vorlap",
        "legacy_vorlap_force_scale": 0.0,
        "residual_source_scale": 1.0,
        "response_force_scale": 1.0,
    },
}
TIMESTEP_S = 0.0075
DURATION_S = 660.0
STORE_FROM_S = 60.0
TOTAL_STEPS = 88_000
RETAINED_ROWS = 80_000
RPM = 1.0
INITIAL_AZIMUTH_DEG = 0.0
RAMPUP_S = 10.0
ADDITIONAL_DAMPING_S = 5.0
WAKE_PROFILE = "parked-reduced-v2"
N_FREQUENCY_ENTRIES = 200
SOURCE_FREQUENCY_CAP_HZ = 0.5 / TIMESTEP_S
RUNTIME_RECEIPT_NAME = "flight_runtime_identity.json"
EXECUTION_AUTHORIZATION_NAME = "full_rerun_execution_authorization.json"
U50_SMOKE_RECEIPT_NAME = "u50_complete_four_step_smoke.json"
EXECUTION_AUTHORIZATION_SCHEMA = "vorlap-hvawt-complete-dlc-execution-authorization-v1"
CHECKPOINT_AUTHORIZATION_NAME = "turbulent_20case_execution_authorization.json"
CHECKPOINT_AUTHORIZATION_SCHEMA = (
    "vorlap-hvawt-turbulent-20case-execution-authorization-v1"
)
CHECKPOINT_ANALYSIS_SCHEMA = "vorlap-hvawt-turbulent-20case-analysis-v1"
CHECKPOINT_CASE_INDICES = (
    0,
    2,
    9,
    11,
    18,
    20,
    27,
    29,
    36,
    38,
    45,
    47,
    54,
    56,
    63,
    65,
    81,
    83,
    90,
    92,
)
CHECKPOINT_WIND_SPEEDS_MPS = (
    3.0,
    7.0,
    8.0,
    8.75,
    11.25,
    13.0,
    14.75,
    18.0,
    21.5,
    35.0,
)
CHECKPOINT_SEED = 4_433_456
CHECKPOINT_SCENARIOS = (SCENARIOS[0], SCENARIOS[2])

ORCHESTRATION_SOURCE_RELATIVE = {
    "campaign_model": Path("vorlap/hvawt_complete_dlc_campaign.py"),
    "campaign_cli": Path("run_headless/hvawt_complete_dlc_campaign.py"),
    "full_submission": Path("run_headless/submit_hvawt_complete_dlc_flight.sj"),
    "analysis_submission": Path("run_headless/submit_hvawt_complete_dlc_analysis.sj"),
    "checkpoint_submission": Path(
        "run_headless/submit_hvawt_turbulent_20case_checkpoint.sj"
    ),
    "checkpoint_analysis_submission": Path(
        "run_headless/submit_hvawt_turbulent_20case_checkpoint_analysis.sj"
    ),
    "u50_smoke_submission": Path("run_headless/submit_hvawt_complete_dlc_u50_smoke.sj"),
    "qblade_runner": Path("run_headless/run_qblade_sim.py"),
    "paper_result_promotion": Path(
        "paper_drafts/NAWEA_QBladeBridge/analysis/prepare_unified_hvawt_paper_results.py"
    ),
}


class HVAWTCompleteDLCError(RuntimeError):
    """Raised when a matched H-VAWT DLC contract is violated."""


def _relative_identity(path: Path, root: Path) -> dict[str, object]:
    identity = _identity(path)
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
    }


def _require_numeric_keyword(path: Path, keyword: str, expected: float) -> None:
    raw = read_qblade_keyword(path, keyword)
    try:
        observed = float(raw) if raw is not None else math.nan
    except ValueError as exc:
        raise HVAWTCompleteDLCError(
            f"{path}: {keyword} is not numeric: {raw!r}."
        ) from exc
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-10):
        raise HVAWTCompleteDLCError(
            f"{path}: {keyword}={raw!r}, expected {expected:.17g}."
        )


def _approved_manifest(repository: Path, path: Path | None = None) -> tuple[Path, dict]:
    source = (
        repository / APPROVED_MANIFEST_RELATIVE
        if path is None
        else Path(path).expanduser().resolve()
    )
    if _identity(source)["sha256"] != APPROVED_MANIFEST_FILE_SHA256:
        raise HVAWTCompleteDLCError(
            "Approved parked-DLC manifest bytes differ from the frozen source."
        )
    manifest = load_dlc_manifest(source, require_approved=True)
    if (
        manifest.get("manifest_sha256") != APPROVED_MANIFEST_CANONICAL_SHA256
        or manifest.get("row_count") != 34
        or len(manifest.get("rows", [])) != 34
        or not math.isclose(
            float(manifest.get("timestep_s", math.nan)),
            TIMESTEP_S,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(manifest.get("duration_s", math.nan)),
            DURATION_S,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(manifest.get("store_from_s", math.nan)),
            STORE_FROM_S,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise HVAWTCompleteDLCError(
            "Approved parked-DLC matrix differs from the frozen 34-row campaign."
        )
    if any(
        not math.isclose(float(row.get("rpm", math.nan)), RPM, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(
            float(row.get("parked_azimuth_deg", math.nan)),
            INITIAL_AZIMUTH_DEG,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for row in manifest.get("rows", [])
        if isinstance(row, Mapping)
    ):
        raise HVAWTCompleteDLCError(
            "Approved parked-DLC rows differ from the frozen 1-RPM, zero-azimuth basis."
        )
    execution = manifest.get("execution_settings")
    if (
        not isinstance(execution, Mapping)
        or execution.get("wake_profile") != WAKE_PROFILE
        or execution.get("result_filter_sha256") != RESULT_FILTER_SHA256
        or execution.get("source_frequency_truncation") != "nyquist"
        or execution.get("source_bounds_policy") != "raise"
    ):
        raise HVAWTCompleteDLCError("Approved parked-DLC execution contract changed.")
    return source, manifest


def build_case_plan(
    campaign_root: Path, approved: Mapping[str, Any]
) -> list[dict[str, object]]:
    """Expand 34 approved wind rows into ordered matched scenario trios."""

    root = Path(campaign_root).expanduser().resolve(strict=False)
    rows: list[dict[str, object]] = []
    source_rows = approved.get("rows")
    if not isinstance(source_rows, list) or len(source_rows) != 34:
        raise HVAWTCompleteDLCError("Exactly 34 approved source rows are required.")
    for row_index, source_row in enumerate(source_rows):
        if not isinstance(source_row, Mapping):
            raise HVAWTCompleteDLCError("Approved source rows must be mappings.")
        row_id = str(source_row.get("row_id", ""))
        if not row_id:
            raise HVAWTCompleteDLCError("Approved source row lacks row_id.")
        for scenario in SCENARIOS:
            case_id = f"{row_id}__{scenario}"
            case_dir = root / "cases" / row_id / scenario
            rows.append(
                {
                    "index": len(rows),
                    "row_index": row_index,
                    "row_id": row_id,
                    "case_id": case_id,
                    "scenario": scenario,
                    "dlc": source_row.get("dlc"),
                    "seed": source_row.get("seed"),
                    "wind_speed_mps": source_row.get("wind_speed_mps"),
                    "wind_file": source_row.get("wind_file"),
                    "fatigue_weight": source_row.get("fatigue_weight"),
                    "include_in_fatigue": source_row.get("include_in_fatigue"),
                    "include_in_ultimate": source_row.get("include_in_ultimate"),
                    "rotor_speed_rpm": RPM,
                    "initial_azimuth_deg": INITIAL_AZIMUTH_DEG,
                    "timestep_s": TIMESTEP_S,
                    "duration_s": DURATION_S,
                    "store_from_s": STORE_FROM_S,
                    "total_steps": TOTAL_STEPS,
                    "case_dir": case_dir.relative_to(root).as_posix(),
                    "sim": (case_dir / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim")
                    .relative_to(root)
                    .as_posix(),
                    "result_dir": (root / "results" / row_id / scenario)
                    .relative_to(root)
                    .as_posix(),
                    "result_name": f"{case_id}_qblade_results",
                    "force_contract": dict(SCENARIO_FORCE_CONTRACT[scenario]),
                }
            )
    if len(rows) != 102 or [row["index"] for row in rows] != list(range(102)):
        raise HVAWTCompleteDLCError("Matched DLC expansion must produce 102 cases.")
    return rows


def _load_turbulent_checkpoint_policy(
    repository: Path, path: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    source = (
        repository / TURBULENT_CHECKPOINT_POLICY_RELATIVE
        if path is None
        else Path(path).expanduser().resolve()
    )
    policy = _load_json(source, label="turbulent 20-case checkpoint policy")
    pairs = policy.get("pairs")
    if (
        policy.get("schema")
        != "vorlap-hvawt-turbulent-20case-checkpoint-policy-v1"
        or policy.get("status") != "approved_for_scoped_execution_attempt"
        or policy.get("case_count") != len(CHECKPOINT_CASE_INDICES)
        or tuple(policy.get("case_indices", [])) != CHECKPOINT_CASE_INDICES
        or tuple(policy.get("scenarios", [])) != CHECKPOINT_SCENARIOS
        or int(policy.get("seed", -1)) != CHECKPOINT_SEED
        or tuple(float(value) for value in policy.get("wind_speeds_mps", []))
        != CHECKPOINT_WIND_SPEEDS_MPS
        or not isinstance(pairs, list)
        or len(pairs) != len(CHECKPOINT_WIND_SPEEDS_MPS)
    ):
        raise HVAWTCompleteDLCError(
            "Turbulent checkpoint policy differs from the reviewed 20-case contract."
        )
    expected_pairs = tuple(
        (
            wind,
            CHECKPOINT_SEED,
            CHECKPOINT_CASE_INDICES[2 * index],
            CHECKPOINT_CASE_INDICES[2 * index + 1],
        )
        for index, wind in enumerate(CHECKPOINT_WIND_SPEEDS_MPS)
    )
    observed_pairs = tuple(
        (
            float(row.get("wind_speed_mps", math.nan)),
            int(row.get("seed", -1)),
            int(row.get("raw_case_index", -1)),
            int(row.get("complete_case_index", -1)),
        )
        for row in pairs
        if isinstance(row, Mapping)
    )
    if observed_pairs != expected_pairs:
        raise HVAWTCompleteDLCError(
            "Turbulent checkpoint pair mapping differs from the reviewed selection."
        )
    return source, policy


def validate_turbulent_checkpoint_selection(
    manifest: Mapping[str, Any], policy: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Return the exact 10 raw/complete TurbSim pairs inside the 102-case plan."""

    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 102:
        raise HVAWTCompleteDLCError(
            "Turbulent checkpoint requires the sealed 102-case campaign."
        )
    selected: list[Mapping[str, Any]] = []
    for pair_index, wind in enumerate(CHECKPOINT_WIND_SPEEDS_MPS):
        raw_index = CHECKPOINT_CASE_INDICES[2 * pair_index]
        complete_index = CHECKPOINT_CASE_INDICES[2 * pair_index + 1]
        raw = cases[raw_index]
        complete = cases[complete_index]
        if (
            not isinstance(raw, Mapping)
            or not isinstance(complete, Mapping)
            or raw.get("index") != raw_index
            or complete.get("index") != complete_index
            or raw.get("row_id") != complete.get("row_id")
            or raw.get("scenario") != SCENARIOS[0]
            or complete.get("scenario") != SCENARIOS[2]
            or int(raw.get("seed", -1)) != CHECKPOINT_SEED
            or int(complete.get("seed", -1)) != CHECKPOINT_SEED
            or not math.isclose(
                float(raw.get("wind_speed_mps", math.nan)),
                wind,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(complete.get("wind_speed_mps", math.nan)),
                wind,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or raw.get("common_physics_fingerprint")
            != complete.get("common_physics_fingerprint")
        ):
            raise HVAWTCompleteDLCError(
                f"Turbulent checkpoint pair {pair_index} does not match the sealed campaign."
            )
        selected.extend((raw, complete))
    if (
        tuple(int(case["index"]) for case in selected) != CHECKPOINT_CASE_INDICES
        or tuple(policy.get("case_indices", [])) != CHECKPOINT_CASE_INDICES
        or any(case.get("scenario") == SCENARIOS[1] for case in selected)
    ):
        raise HVAWTCompleteDLCError("Turbulent checkpoint selection is inconsistent.")
    return selected


def _patch_dlc_sim(
    sim_path: Path,
    *,
    source_row: Mapping[str, Any],
    result_filter_name: str,
) -> None:
    wind_settings = {
        "PRESCRIBETYPE": 1,
        "RPMPRESCRIBED": RPM,
        "INITIAL_AZIMUTH": INITIAL_AZIMUTH_DEG,
        "RELAXSTEPS": 0,
        "RELAXTIMESTEP": TIMESTEP_S,
        "INCLUDEAERO": 1,
        "INCLUDEHYDRO": 0,
        "TIMESTEP": TIMESTEP_S,
        "NUMTIMESTEPS": TOTAL_STEPS,
        "RAMPUP": RAMPUP_S,
        "ADDDAMP": ADDITIONAL_DAMPING_S,
        "WNDTYPE": 1,
        "WNDNAME": str(source_row["wind_file"]),
        "STITCHINGTYPE": 0,
        "WINDAUTOSHIFT": "true",
        "SHIFTTIME": 0.0,
        "MEANINF": float(source_row["wind_speed_mps"]),
        "HORANGLE": 0.0,
        "VERTANGLE": 0.0,
        "SHEAREXP": 0.23,
        "REFHEIGHT": 18.0,
        "DENSITYAIR": 1.0296,
        "VISCOSITYAIR": 1.647e-5,
        "STOREFROM": STORE_FROM_S,
        "STOREREPLAY": "false",
        "FILTERFILE": result_filter_name,
    }
    for keyword, value in wind_settings.items():
        replace_qblade_keyword_file(sim_path, keyword, value)


def _apply_wake_profile(sim_path: Path) -> dict[str, object]:
    turbine = turbine_definition_path_for_sim(sim_path)
    patches = wake_profile_settings(WAKE_PROFILE)
    for keyword, value in patches.items():
        replace_qblade_keyword_file(turbine, keyword, value)
    return {
        "profile": WAKE_PROFILE,
        "settings": patches,
        "turbine": _common_physics_asset_identity(turbine, sim_path.parent),
    }


def _common_physics_asset_identity(path: Path, case_dir: Path) -> dict[str, object]:
    """Hash physical content while excluding only external-coupling plumbing."""

    suffix = path.suffix.lower()
    normalization = "exact_bytes"
    if suffix == ".trb":
        text = path.read_text(encoding="utf-8", errors="strict")
        payload = strip_qblade_external_library_section(text).encode("utf-8")
        normalization = "qblade_turbine_without_external_library_section"
    elif suffix == ".str":
        text = path.read_text(encoding="utf-8", errors="strict")
        payload = strip_qblade_external_tables(text).encode("utf-8")
        normalization = "qblade_structure_without_external_swap_tables"
    else:
        payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(case_dir.resolve()).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "normalization": normalization,
    }


def _physics_assets(case_dir: Path, sim_path: Path) -> list[dict[str, object]]:
    """Hash every shared aerodynamic/structural/source asset in one case.

    Control files, the turbine external-library section, structural swap
    tables, and storage filters intentionally differ by scenario. The physical
    turbine, airfoil spectra, polars, and structural definitions must not.
    """

    files: list[Path] = []
    for relative in (
        "10m_HVAWT_reference_turbine/Aero",
        "10m_HVAWT_reference_turbine/Structure",
        "10m_HVAWT_reference_turbine/VorLapAirfoils",
    ):
        directory = case_dir / relative
        if not directory.is_dir():
            raise HVAWTCompleteDLCError(
                f"Missing common-physics directory: {directory}"
            )
        files.extend(path for path in directory.rglob("*") if path.is_file())
    files.append(turbine_definition_path_for_sim(sim_path))
    return [
        _common_physics_asset_identity(path, case_dir) for path in sorted(files)
    ]


def _common_physics_contract(
    *,
    case_dir: Path,
    sim_path: Path,
    source_row: Mapping[str, Any],
    wind_record: Mapping[str, Any],
    rayleigh_normalization: Mapping[str, Any] | None,
    wake_audit: Mapping[str, Any],
) -> dict[str, object]:
    keywords = {
        key: read_qblade_keyword(sim_path, key)
        for key in (
            "PRESCRIBETYPE",
            "RPMPRESCRIBED",
            "INITIAL_AZIMUTH",
            "RELAXSTEPS",
            "RELAXTIMESTEP",
            "INCLUDEAERO",
            "INCLUDEHYDRO",
            "TIMESTEP",
            "NUMTIMESTEPS",
            "RAMPUP",
            "ADDDAMP",
            "WNDTYPE",
            "WNDNAME",
            "STITCHINGTYPE",
            "WINDAUTOSHIFT",
            "SHIFTTIME",
            "MEANINF",
            "HORANGLE",
            "VERTANGLE",
            "SHEAREXP",
            "REFHEIGHT",
            "DENSITYAIR",
            "VISCOSITYAIR",
            "STOREFROM",
            "STOREREPLAY",
        )
    }
    contract = {
        "row_id": source_row["row_id"],
        "wind_record_sha256": wind_record["record_sha256"],
        "wind_file_sha256": wind_record["sha256"],
        "wind_file_bytes": wind_record["bytes"],
        "simulation_keywords": keywords,
        "rayleigh_normalization": rayleigh_normalization,
        "wake_profile": wake_audit["profile"],
        "wake_settings": wake_audit["settings"],
        "wake_turbine_identity": wake_audit["turbine"],
        "physical_assets": _physics_assets(case_dir, sim_path),
    }
    contract["fingerprint"] = canonical_fingerprint(contract)
    return contract


def _prepare_scenario(
    *,
    runtime_repository: Path,
    case: dict[str, object],
    case_dir: Path,
    sim_path: Path,
    response_model: Mapping[str, Any],
    model_source: Path,
    result_filter: Path,
) -> None:
    scenario = str(case["scenario"])
    case_filter = case_dir / result_filter.name
    shutil.copy2(result_filter, case_filter)
    if scenario == SCENARIOS[0]:
        replace_qblade_keyword_file(sim_path, "FILTERFILE", case_filter.name)
        case.update(
            {
                "runtime_config": None,
                "storage_filter": case_filter.name,
                "storage_filter_sha256": _identity(case_filter)["sha256"],
                "storage_filter_channel_count": RESULT_FILTER_CHANNEL_COUNT,
            }
        )
        return

    one_way = scenario == SCENARIOS[1]
    prepared = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim_path,
            airfoil_dir=vorlap_airfoil_dir_for_sim(sim_path),
            sample_step=TIMESTEP_S,
            node_source="structural",
            include_struts=True,
            include_tower=False,
            n_freq_depth=N_FREQUENCY_ENTRIES,
            force_scale=1.0 if one_way else 0.0,
            include_mean_load=False,
            source_bounds_policy="raise",
            source_frequency_cap_hz=SOURCE_FREQUENCY_CAP_HZ,
            hvawt_response_model=None if one_way else response_model,
            hvawt_response_artifact_dir=None if one_way else model_source.parent,
            inflow_vec=(1.0, 0.0, 0.0),
            include_default_python_path=True,
            default_python_path=runtime_repository,
            absolute_source_parameter_dir=True,
            debug=False,
        )
    )
    storage_filter = case_dir / "hvawt_complete_dlc_sil_runtime.filter"
    storage = write_sil_storage_filter(
        result_filter=case_filter,
        storage_filter=storage_filter,
        parameter_file=prepared.parameter_file_path,
    )
    replace_qblade_keyword_file(sim_path, "FILTERFILE", storage_filter.name)
    case.update(
        {
            "runtime_config": prepared.parameter_file_path.relative_to(
                case_dir
            ).as_posix(),
            "storage_filter": storage_filter.relative_to(case_dir).as_posix(),
            "storage_filter_sha256": storage["sha256"],
            "storage_filter_channel_count": storage["channel_count"],
        }
    )


def generate_campaign(
    campaign_root: Path,
    *,
    rayleigh_selection: Path,
    repository: Path | None = None,
    runtime_repository: Path | None = None,
    approved_manifest_path: Path | None = None,
) -> Path:
    """Materialize the exact 102-case campaign on the wind-field filesystem."""

    repo = (
        Path(__file__).resolve().parents[1]
        if repository is None
        else Path(repository).expanduser().resolve()
    )
    runtime_repo = (
        repo
        if runtime_repository is None
        else Path(runtime_repository).expanduser().resolve()
    )
    approved_source, approved = _approved_manifest(repo, approved_manifest_path)
    wind_records: dict[str, Mapping[str, Any]] = {}
    for source_row in approved["rows"]:
        wind_records[str(source_row["row_id"])] = validate_wind_record(
            source_row, approved
        )

    spec_source = repo / CAMPAIGN_SPEC_RELATIVE
    model_source = repo / RESPONSE_MODEL_RELATIVE
    spec = _load_json(spec_source, label="H-VAWT response campaign spec")
    response_model = _load_json(model_source, label="H-VAWT complete model")
    validate_campaign_spec(spec)
    validate_response_model(response_model, spec)
    source_case, rayleigh = _rayleigh_source_case(rayleigh_selection)
    source_sim_relative = str(rayleigh["selected_sim_relative_path"])
    source_sim = source_case / source_sim_relative
    issues = qblade_aerodynamic_reference_issues(source_sim)
    if issues:
        raise HVAWTCompleteDLCError("Incomplete source case: " + "; ".join(issues))

    root = Path(campaign_root).expanduser().resolve(strict=False)
    if root.exists():
        raise FileExistsError(f"Complete DLC campaign already exists: {root}")
    (root / "inputs" / "wind_records").mkdir(parents=True)
    (root / "cases").mkdir()
    (root / "results").mkdir()
    copied_approved = root / "inputs" / approved_source.name
    copied_spec = root / "inputs" / spec_source.name
    copied_model = root / "inputs" / model_source.name
    copied_rayleigh = root / "inputs" / Path(rayleigh_selection).name
    result_filter_source = repo / RESULT_FILTER_SOURCE_RELATIVE
    if _identity(result_filter_source)["sha256"] != RESULT_FILTER_SHA256:
        raise HVAWTCompleteDLCError("Fatigue result-filter identity changed.")
    _result_filter_channels(result_filter_source)
    copied_filter = root / "inputs" / result_filter_source.name
    shutil.copy2(approved_source, copied_approved)
    shutil.copy2(spec_source, copied_spec)
    shutil.copy2(model_source, copied_model)
    shutil.copy2(Path(rayleigh_selection).expanduser().resolve(), copied_rayleigh)
    shutil.copy2(result_filter_source, copied_filter)
    for row_id, record in wind_records.items():
        atomic_write_json(root / "inputs" / "wind_records" / f"{row_id}.json", record)

    cases = build_case_plan(root, approved)
    source_rows = {str(row["row_id"]): row for row in approved["rows"]}
    for case in cases:
        row_id = str(case["row_id"])
        source_row = source_rows[row_id]
        case_dir = root / str(case["case_dir"])
        shutil.copytree(source_case, case_dir, symlinks=False)
        normalization = _apply_engineering_rayleigh_inputs(case_dir, rayleigh)
        sim_path = case_dir / source_sim_relative
        expected_sim = root / str(case["sim"])
        if sim_path != expected_sim:
            raise HVAWTCompleteDLCError(
                f"Source simulation path changed for {case['case_id']}."
            )
        spectrum_staging = _stage_calibrated_static_spectrum(
            repository=repo,
            model_source=model_source,
            response_model=response_model,
            sim_path=sim_path,
            case_dir=case_dir,
        )
        _patch_dlc_sim(
            sim_path,
            source_row=source_row,
            result_filter_name=copied_filter.name,
        )
        wake_audit = _apply_wake_profile(sim_path)
        _prepare_scenario(
            runtime_repository=runtime_repo,
            case=case,
            case_dir=case_dir,
            sim_path=sim_path,
            response_model=response_model,
            model_source=model_source,
            result_filter=copied_filter,
        )
        runtime_assets, runtime_fingerprint = qblade_runtime_asset_fingerprint(sim_path)
        common = _common_physics_contract(
            case_dir=case_dir,
            sim_path=sim_path,
            source_row=source_row,
            wind_record=wind_records[row_id],
            rayleigh_normalization=normalization,
            wake_audit=wake_audit,
        )
        case.update(
            {
                "rayleigh_input_normalization": normalization,
                "static_spectrum_staging": spectrum_staging,
                "runtime_asset_fingerprint": runtime_fingerprint,
                "runtime_asset_count": len(runtime_assets),
                "common_physics_fingerprint": common["fingerprint"],
                "common_physics_contract": common,
                "result_filter": copied_filter.relative_to(root).as_posix(),
            }
        )

    for row_id in source_rows:
        row_cases = [case for case in cases if case["row_id"] == row_id]
        fingerprints = {case["common_physics_fingerprint"] for case in row_cases}
        if len(row_cases) != 3 or len(fingerprints) != 1:
            raise HVAWTCompleteDLCError(
                f"Matched trio {row_id} does not share one common-physics fingerprint."
            )

    inputs = {
        "approved_manifest": _relative_identity(copied_approved, root),
        "campaign_spec": _relative_identity(copied_spec, root),
        "response_model": _relative_identity(copied_model, root),
        "rayleigh_selection": _relative_identity(copied_rayleigh, root),
        "result_filter": _relative_identity(copied_filter, root),
        "orchestration_sources": {
            role: _identity(repo / relative)
            for role, relative in ORCHESTRATION_SOURCE_RELATIVE.items()
        },
    }
    manifest: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "prepared_not_run",
        "sealed_inputs": True,
        "case_count": len(cases),
        "row_count": 34,
        "scenarios": list(SCENARIOS),
        "matrix": {
            "duration_s": DURATION_S,
            "store_from_s": STORE_FROM_S,
            "timestep_s": TIMESTEP_S,
            "total_steps": TOTAL_STEPS,
            "retained_rows": RETAINED_ROWS,
            "rpm": RPM,
            "initial_azimuth_deg": INITIAL_AZIMUTH_DEG,
            "wake_profile": WAKE_PROFILE,
        },
        "approved_manifest_sha256": APPROVED_MANIFEST_CANONICAL_SHA256,
        "inputs": inputs,
        "rayleigh_selection": rayleigh,
        "wind_records": {
            row_id: _relative_identity(
                root / "inputs" / "wind_records" / f"{row_id}.json", root
            )
            for row_id in wind_records
        },
        "comparison_contract": {
            "raw": "native QBlade aerodynamics with no supplemental force",
            "one_way": "native QBlade plus full fixed-body static-spectrum VorLap forcing without response feedback",
            "complete": "native QBlade plus uncaptured static-spectrum residual plus motion-correlated lock-in feedback",
            "matched_inputs": "same wind bytes, structural/aero bytes, timestep, wake, startup, and load channels within each row",
            "historical_archive_role": "context_only_not_exact_control_for_this_corrected_campaign",
        },
        "cases": cases,
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    manifest_path = root / "campaign_manifest.json"
    atomic_write_json(manifest_path, manifest)
    _write_postprocess_layout(root, manifest)
    validate_campaign(manifest_path, rehash_external_winds=True)
    return manifest_path


def _write_postprocess_layout(root: Path, manifest: Mapping[str, Any]) -> None:
    raw_cases = [case for case in manifest["cases"] if case["scenario"] == SCENARIOS[0]]
    planned_cases = [
        {
            "label": case["row_id"],
            "speed_label": case["row_id"],
            "azimuth_label": "A000p00deg",
            "wind_speed_mps": case["wind_speed_mps"],
            "parked_azimuth_deg": INITIAL_AZIMUTH_DEG,
        }
        for case in raw_cases
    ]
    result_filter = root / str(manifest["inputs"]["result_filter"]["path"])
    settings = {
        "schema": "vorlap-hvawt-complete-dlc-postprocess-layout-v1",
        "timestep_s": TIMESTEP_S,
        "duration_s": DURATION_S,
        "store_from_s": STORE_FROM_S,
        "timesteps": TOTAL_STEPS,
        "qblade_steps_override": None,
        "scenario_names": list(SCENARIOS),
        "scenario_roles": {
            SCENARIOS[0]: "native_qblade_no_supplemental_force",
            SCENARIOS[1]: "native_qblade_plus_full_static_spectral_force",
            SCENARIOS[2]: "native_qblade_plus_residual_spectrum_plus_lockin_feedback",
        },
        "scenario_couplings": {
            scenario: contract["coupling_mode"]
            for scenario, contract in SCENARIO_FORCE_CONTRACT.items()
        },
        "scenario_force_scales": {
            scenario: contract["legacy_vorlap_force_scale"]
            for scenario, contract in SCENARIO_FORCE_CONTRACT.items()
        },
        "planned_cases": planned_cases,
        "planned_case_count_per_scenario": 34,
        "planned_case_count_total": 102,
        "result_filter": str(result_filter),
        "result_filter_sha256": RESULT_FILTER_SHA256,
        "rpm": RPM,
        "wake_profile": WAKE_PROFILE,
        "source_bounds_policy": "raise",
        "source_frequency_truncation": "nyquist",
        "source_frequency_cap_hz": SOURCE_FREQUENCY_CAP_HZ,
        "campaign_fingerprint": manifest["fingerprint"],
    }
    atomic_write_json(root / "campaign_settings.json", settings)
    for scenario in SCENARIOS:
        scenario_root = root / scenario
        scenario_root.mkdir()
        selected = [case for case in manifest["cases"] if case["scenario"] == scenario]
        cases = []
        for case in selected:
            cases.append(
                {
                    "index": len(cases),
                    "label": case["row_id"],
                    "wind_speed": case["wind_speed_mps"],
                    "azimuth_deg": INITIAL_AZIMUTH_DEG,
                    "case_dir": str(root / case["case_dir"]),
                    "sim": str(root / case["sim"]),
                    "result_dir": str(root / case["result_dir"]),
                    "result_name": case["result_name"],
                    "result_filter": str(root / case["result_filter"]),
                }
            )
        atomic_write_json(
            scenario_root / "cases.json",
            {
                "scenario": scenario,
                "campaign_fingerprint": manifest["fingerprint"],
                "settings": settings,
                "cases": cases,
            },
        )


def _manifest_path(value: Path) -> Path:
    candidate = Path(value).expanduser().resolve()
    return candidate / "campaign_manifest.json" if candidate.is_dir() else candidate


def _load_copied_wind_records(
    *, root: Path, manifest: Mapping[str, Any], approved: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    identities = manifest.get("wind_records")
    if not isinstance(identities, Mapping):
        raise HVAWTCompleteDLCError("Sealed wind-record identities are missing.")
    records: dict[str, dict[str, Any]] = {}
    for source_row in approved["rows"]:
        row_id = str(source_row["row_id"])
        identity = identities.get(row_id)
        if not isinstance(identity, Mapping):
            raise HVAWTCompleteDLCError(f"Copied wind record is missing for {row_id}.")
        path = _verify_identity(root, identity, label=f"wind record {row_id}")
        record = _load_json(path, label=f"copied wind record {row_id}")
        payload = dict(record)
        archived_digest = payload.pop("record_sha256", None)
        if (
            record.get("schema_version") != 1
            or record.get("row_id") != row_id
            or record.get("manifest_sha256") != APPROVED_MANIFEST_CANONICAL_SHA256
            or archived_digest != sha256_payload(payload)
            or record.get("wind_file") != source_row["wind_file"]
            or record.get("seed") != source_row["seed"]
            or record.get("dlc") != source_row["dlc"]
            or record.get("wind_speed_mps") != source_row["wind_speed_mps"]
        ):
            raise HVAWTCompleteDLCError(
                f"Copied wind record contract changed for {row_id}."
            )
        records[row_id] = record
    if set(records) != set(identities):
        raise HVAWTCompleteDLCError("Sealed wind-record set is not exactly 34 rows.")
    return records


def validate_campaign(
    value: Path,
    *,
    rehash_external_winds: bool = True,
    rehash_case_assets: bool = True,
    deep_case_validation: bool = True,
) -> dict[str, Any]:
    manifest_path = _manifest_path(value)
    root = manifest_path.parent
    manifest = _load_json(manifest_path, label="complete DLC campaign manifest")
    if (
        manifest.get("schema") != CAMPAIGN_SCHEMA
        or manifest.get("sealed_inputs") is not True
        or manifest.get("case_count") != 102
        or manifest.get("row_count") != 34
        or tuple(manifest.get("scenarios", [])) != SCENARIOS
    ):
        raise HVAWTCompleteDLCError("Complete DLC campaign header is invalid.")
    expected_matrix = {
        "duration_s": DURATION_S,
        "store_from_s": STORE_FROM_S,
        "timestep_s": TIMESTEP_S,
        "total_steps": TOTAL_STEPS,
        "retained_rows": RETAINED_ROWS,
        "rpm": RPM,
        "initial_azimuth_deg": INITIAL_AZIMUTH_DEG,
        "wake_profile": WAKE_PROFILE,
    }
    if (
        manifest.get("status") != "prepared_not_run"
        or manifest.get("matrix") != expected_matrix
        or manifest.get("approved_manifest_sha256")
        != APPROVED_MANIFEST_CANONICAL_SHA256
    ):
        raise HVAWTCompleteDLCError("Complete DLC matrix contract changed.")
    fingerprint = manifest.get("fingerprint")
    payload = dict(manifest)
    payload.pop("fingerprint", None)
    if fingerprint != canonical_fingerprint(payload):
        raise HVAWTCompleteDLCError("Complete DLC campaign fingerprint is invalid.")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise HVAWTCompleteDLCError("Complete DLC campaign inputs are missing.")
    for key in (
        "approved_manifest",
        "campaign_spec",
        "response_model",
        "rayleigh_selection",
        "result_filter",
    ):
        identity = inputs.get(key)
        if not isinstance(identity, Mapping):
            raise HVAWTCompleteDLCError(f"Campaign input {key!r} is missing.")
        _verify_identity(root, identity, label=f"complete DLC input {key}")
    copied_approved = root / str(inputs["approved_manifest"]["path"])
    if _identity(copied_approved)["sha256"] != APPROVED_MANIFEST_FILE_SHA256:
        raise HVAWTCompleteDLCError("Copied approved-manifest bytes changed.")
    approved = load_dlc_manifest(copied_approved, require_approved=True)
    if approved.get("manifest_sha256") != APPROVED_MANIFEST_CANONICAL_SHA256:
        raise HVAWTCompleteDLCError("Copied approved manifest changed.")
    copied_filter = root / str(inputs["result_filter"]["path"])
    if (
        _identity(copied_filter)["sha256"] != RESULT_FILTER_SHA256
        or len(_result_filter_channels(copied_filter)) != RESULT_FILTER_CHANNEL_COUNT
    ):
        raise HVAWTCompleteDLCError("Copied result filter changed.")
    copied_spec = root / str(inputs["campaign_spec"]["path"])
    copied_model = root / str(inputs["response_model"]["path"])
    repository = Path(__file__).resolve().parents[1]
    if (
        _identity(copied_spec)["sha256"]
        != _identity(repository / CAMPAIGN_SPEC_RELATIVE)["sha256"]
        or _identity(copied_model)["sha256"]
        != _identity(repository / RESPONSE_MODEL_RELATIVE)["sha256"]
    ):
        raise HVAWTCompleteDLCError(
            "Staged H-VAWT spec/model bytes differ from the executing checkout."
        )
    copied_spec_payload = _load_json(copied_spec, label="copied campaign spec")
    copied_model_payload = _load_json(copied_model, label="copied response model")
    validate_campaign_spec(copied_spec_payload)
    validate_response_model(copied_model_payload, copied_spec_payload)
    orchestration = inputs.get("orchestration_sources")
    if not isinstance(orchestration, Mapping):
        raise HVAWTCompleteDLCError("Orchestration identities are missing.")
    for role, relative in ORCHESTRATION_SOURCE_RELATIVE.items():
        recorded = orchestration.get(role)
        if not isinstance(recorded, Mapping) or recorded != _identity(
            repository / relative
        ):
            raise HVAWTCompleteDLCError(
                f"Orchestration source {role!r} differs from campaign generation."
            )
    source_rows = {str(row["row_id"]): row for row in approved["rows"]}
    copied_wind_records = _load_copied_wind_records(
        root=root, manifest=manifest, approved=approved
    )
    if rehash_external_winds:
        for source_row in approved["rows"]:
            observed = validate_wind_record(source_row, approved)
            if copied_wind_records[str(source_row["row_id"])] != observed:
                raise HVAWTCompleteDLCError(
                    f"External wind record changed for {source_row['row_id']}."
                )

    cases = manifest.get("cases")
    expected = build_case_plan(root, approved)
    if not isinstance(cases, list) or len(cases) != len(expected):
        raise HVAWTCompleteDLCError("Complete DLC case count changed.")
    observed_order = [
        (case.get("row_id"), case.get("scenario"), case.get("index"))
        for case in cases
        if isinstance(case, Mapping)
    ]
    expected_order = [
        (case["row_id"], case["scenario"], case["index"]) for case in expected
    ]
    if observed_order != expected_order:
        raise HVAWTCompleteDLCError("Complete DLC case ordering changed.")
    if not deep_case_validation:
        return {
            "valid": True,
            "campaign_root": root,
            "manifest_path": manifest_path,
            "manifest": manifest,
        }
    by_row: dict[str, set[str]] = {row_id: set() for row_id in source_rows}
    for case, expected_case in zip(cases, expected, strict=True):
        if not isinstance(case, Mapping):
            raise HVAWTCompleteDLCError("Complete DLC case row is invalid.")
        for key, expected_value in expected_case.items():
            if case.get(key) != expected_value:
                raise HVAWTCompleteDLCError(
                    f"Case {case.get('case_id', '<unknown>')} field {key!r} changed."
                )
        case_dir = root / str(case["case_dir"])
        sim_path = root / str(case["sim"])
        storage = case_dir / str(case["storage_filter"])
        if not sim_path.is_file() or not storage.is_file():
            raise FileNotFoundError(f"Incomplete case assets for {case['case_id']}.")
        row = source_rows[str(case["row_id"])]
        expected_keywords = {
            "PRESCRIBETYPE": 1.0,
            "RPMPRESCRIBED": RPM,
            "INITIAL_AZIMUTH": INITIAL_AZIMUTH_DEG,
            "RELAXSTEPS": 0.0,
            "RELAXTIMESTEP": TIMESTEP_S,
            "INCLUDEAERO": 1.0,
            "INCLUDEHYDRO": 0.0,
            "TIMESTEP": TIMESTEP_S,
            "NUMTIMESTEPS": float(TOTAL_STEPS),
            "RAMPUP": RAMPUP_S,
            "ADDDAMP": ADDITIONAL_DAMPING_S,
            "WNDTYPE": 1.0,
            "STITCHINGTYPE": 0.0,
            "SHIFTTIME": 0.0,
            "MEANINF": float(row["wind_speed_mps"]),
            "STOREFROM": STORE_FROM_S,
        }
        for keyword, expected_value in expected_keywords.items():
            _require_numeric_keyword(sim_path, keyword, expected_value)
        if read_qblade_keyword(sim_path, "WNDNAME") != str(row["wind_file"]):
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} uses a different TurbSim file."
            )
        if read_qblade_keyword(sim_path, "WINDAUTOSHIFT") != "true":
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} changed automatic TurbSim advection."
            )
        if read_qblade_keyword(sim_path, "FILTERFILE") != storage.name:
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} uses a different storage filter."
            )
        storage_identity = _identity(storage)
        if (
            storage_identity["sha256"] != case.get("storage_filter_sha256")
            or len(qblade_filter_channels(storage))
            != case.get("storage_filter_channel_count")
            or case.get("result_filter") != inputs["result_filter"]["path"]
        ):
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} storage/result-filter contract changed."
            )
        contract = SCENARIO_FORCE_CONTRACT[str(case["scenario"])]
        if case.get("force_contract") != contract:
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} force contract changed."
            )
        runtime_config = case.get("runtime_config")
        if case["scenario"] == SCENARIOS[0]:
            if runtime_config is not None:
                raise HVAWTCompleteDLCError("Raw QBlade case must not activate SIL.")
        else:
            config_path = case_dir / str(runtime_config)
            config = _load_json(config_path, label=f"runtime config {case['case_id']}")
            hvawt = config.get("hvawt_response_model")
            if case["scenario"] == SCENARIOS[1]:
                if (
                    float(config.get("force_scale", math.nan)) != 1.0
                    or hvawt is not None
                ):
                    raise HVAWTCompleteDLCError(
                        "One-way case must use full static-spectrum force and no response model."
                    )
            else:
                engineering = (
                    hvawt.get("engineering_lockin")
                    if isinstance(hvawt, Mapping)
                    else None
                )
                if (
                    float(config.get("force_scale", math.nan)) != 0.0
                    or not isinstance(engineering, Mapping)
                    or engineering.get("residual_source_scale") != 1.0
                    or engineering.get("response_force_scale") != 1.0
                ):
                    raise HVAWTCompleteDLCError(
                        "Complete case does not enable the exact residual and response paths."
                    )
        staging = case.get("static_spectrum_staging")
        if not isinstance(staging, Mapping):
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} lacks static-spectrum staging evidence."
            )
        staged_source = staging.get("destination")
        if not isinstance(staged_source, Mapping):
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} lacks staged NACA0018 identity."
            )
        staged_path = _verify_identity(
            case_dir,
            staged_source,
            label=f"staged NACA0018 source for {case['case_id']}",
        )
        if (
            _identity(staged_path)["sha256"]
            != copied_spec_payload["static_spectrum_sha256"]
        ):
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} uses different static-spectrum bytes."
            )
        if rehash_case_assets:
            _assets, runtime_fingerprint = qblade_runtime_asset_fingerprint(sim_path)
            if runtime_fingerprint != case.get("runtime_asset_fingerprint"):
                raise HVAWTCompleteDLCError(
                    f"Case {case['case_id']} runtime assets changed."
                )
        common = case.get("common_physics_contract")
        normalization = case.get("rayleigh_input_normalization")
        turbine = turbine_definition_path_for_sim(sim_path)
        observed_wake = {
            "profile": WAKE_PROFILE,
            "settings": wake_profile_settings(WAKE_PROFILE),
            "turbine": _common_physics_asset_identity(turbine, case_dir),
        }
        observed_common = _common_physics_contract(
            case_dir=case_dir,
            sim_path=sim_path,
            source_row=row,
            wind_record=copied_wind_records[str(case["row_id"])],
            rayleigh_normalization=(
                normalization if isinstance(normalization, Mapping) else None
            ),
            wake_audit=observed_wake,
        )
        if (
            not isinstance(common, Mapping)
            or common.get("fingerprint") != case.get("common_physics_fingerprint")
            or canonical_fingerprint(
                {k: v for k, v in common.items() if k != "fingerprint"}
            )
            != common.get("fingerprint")
            or common != observed_common
        ):
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} common-physics contract is invalid."
            )
        by_row[str(case["row_id"])].add(str(case["common_physics_fingerprint"]))
    if any(len(fingerprints) != 1 for fingerprints in by_row.values()):
        raise HVAWTCompleteDLCError(
            "One or more raw/one-way/complete trios do not share common physics."
        )
    return {
        "valid": True,
        "campaign_root": root,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def build_runtime_identity(qblade_root: Path) -> dict[str, object]:
    root = Path(qblade_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or root.name != "QBladeCE_2.0.9.7":
        raise HVAWTCompleteDLCError(f"Unexpected QBlade runtime root: {root}")
    library = find_qblade_library(root)
    bridge = find_bridge_library(root, "vorlap_qblade_bridge")
    if bridge is None or not bridge.is_file() or bridge.is_symlink():
        raise HVAWTCompleteDLCError("VorLap-QBlade bridge library is unavailable.")
    if (
        _identity(library)["sha256"] != QBLADE_LIBRARY_SHA256
        or _identity(bridge)["sha256"] != BRIDGE_LIBRARY_SHA256
    ):
        raise HVAWTCompleteDLCError("Flight QBlade or bridge library identity changed.")
    _deps, dependency_fingerprint = qblade_runtime_dependency_manifest(root)
    _python, python_fingerprint = vorlap_python_runtime_manifest()
    repository = Path(__file__).resolve().parents[1]
    return {
        "schema": RUNTIME_SCHEMA,
        "qblade_library": _identity(library),
        "bridge_library": _identity(bridge),
        "qblade_runtime_dependency_fingerprint": dependency_fingerprint,
        "vorlap_python_runtime_fingerprint": python_fingerprint,
        "orchestration_sources": {
            role: _identity(repository / relative)
            for role, relative in ORCHESTRATION_SOURCE_RELATIVE.items()
        },
    }


def bind_runtime(campaign: Path, *, qblade_root: Path) -> dict[str, object]:
    state = validate_campaign(campaign, rehash_external_winds=True)
    root = state["campaign_root"]
    observed = build_runtime_identity(qblade_root)
    lock_path = root / ".flight_runtime_identity.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt_path = root / RUNTIME_RECEIPT_NAME
        if receipt_path.exists():
            recorded = _load_json(receipt_path, label="complete DLC runtime receipt")
            if recorded != observed:
                raise HVAWTCompleteDLCError(
                    "Flight runtime differs from the campaign's first bound runtime."
                )
        else:
            atomic_write_json(receipt_path, observed)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return observed


def _validate_fingerprint(payload: Mapping[str, Any], *, label: str) -> None:
    recorded = payload.get("fingerprint")
    content = dict(payload)
    content.pop("fingerprint", None)
    if recorded != canonical_fingerprint(content):
        raise HVAWTCompleteDLCError(f"{label} fingerprint is invalid.")


def bind_execution_authorization(
    campaign: Path, *, pilot_campaign: Path
) -> dict[str, object]:
    """Bind a passed 20-case uniform-flow pilot before turbulent execution."""

    state = validate_campaign(
        campaign,
        rehash_external_winds=False,
        rehash_case_assets=True,
        deep_case_validation=True,
    )
    root = state["campaign_root"]
    manifest = state["manifest"]
    pilot_state = validate_pilot_campaign(pilot_campaign)
    pilot_root = pilot_state["campaign_root"]
    pilot_manifest = pilot_state["manifest"]
    pilot_matrix = pilot_manifest.get("matrix")
    if (
        not isinstance(pilot_matrix, Mapping)
        or pilot_matrix.get("campaign_kind") != FULL_CAMPAIGN_KIND
        or pilot_manifest.get("case_count") != 20
    ):
        raise HVAWTCompleteDLCError(
            "Execution authorization requires the complete 20-case uniform-flow pilot."
        )
    full_inputs = manifest["inputs"]
    pilot_inputs = pilot_manifest.get("inputs")
    if not isinstance(pilot_inputs, Mapping) or any(
        pilot_inputs.get(role, {}).get("sha256")
        != full_inputs.get(role, {}).get("sha256")
        for role in ("campaign_spec", "response_model")
    ):
        raise HVAWTCompleteDLCError(
            "Pilot and turbulent campaign do not use identical spec/model bytes."
        )

    analysis_path = pilot_root / "pilot_analysis.json"
    authorization_path = pilot_root / "full_rerun_plan_authorized.json"
    analysis = _load_json(analysis_path, label="passed H-VAWT pilot analysis")
    authorization = _load_json(
        authorization_path, label="authorized H-VAWT full-rerun plan"
    )
    _validate_fingerprint(analysis, label="Pilot analysis")
    _validate_fingerprint(authorization, label="Full-rerun authorization")
    cases = analysis.get("cases")
    cross_gates = analysis.get("cross_case_gates")
    if (
        analysis.get("schema") != PILOT_ANALYSIS_SCHEMA
        or analysis.get("status") != "pilot_passed_complete_hybrid_screen"
        or analysis.get("campaign_kind") != FULL_CAMPAIGN_KIND
        or analysis.get("campaign_fingerprint") != pilot_manifest["fingerprint"]
        or analysis.get("complete_case_count") != 20
        or analysis.get("expected_case_count") != 20
        or analysis.get("passed_case_count") != 20
        or analysis.get("full_rerun_authorized") is not True
        or not isinstance(cases, list)
        or len(cases) != 20
        or any(
            row.get("passed") is not True
            or row.get("execution_receipt_pass") is not True
            for row in cases
            if isinstance(row, Mapping)
        )
        or any(not isinstance(row, Mapping) for row in cases)
        or not isinstance(cross_gates, list)
        or not cross_gates
        or any(
            not isinstance(row, Mapping) or row.get("passed") is not True
            for row in cross_gates
        )
    ):
        raise HVAWTCompleteDLCError(
            "The complete 20-case uniform-flow pilot has not passed all gates."
        )
    if (
        authorization.get("schema") != FULL_RERUN_SCHEMA
        or authorization.get("status") != "authorized_for_matched_three_path_execution"
        or authorization.get("execution_authorized") is not True
        or authorization.get("launch_implemented") is not True
        or authorization.get("campaign_fingerprint") != pilot_manifest["fingerprint"]
        or authorization.get("expected_runs") != 102
        or tuple(authorization.get("scenarios", [])) != FULL_RERUN_SCENARIOS
        or authorization.get("pilot_analysis") != _identity(analysis_path)
        or authorization.get("source_manifest", {}).get("sha256")
        != APPROVED_MANIFEST_FILE_SHA256
    ):
        raise HVAWTCompleteDLCError(
            "Full-rerun authorization is not the exact 102-case three-path contract."
        )

    snapshot_root = root / "execution_authorization"
    lock_path = root / ".execution_authorization.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt_path = root / EXECUTION_AUTHORIZATION_NAME
        if receipt_path.exists():
            recorded = _load_json(receipt_path, label="execution authorization receipt")
            _load_execution_authorization(root, manifest)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return recorded
        if snapshot_root.exists():
            raise HVAWTCompleteDLCError(
                "Authorization snapshots exist without their sealed receipt."
            )
        snapshot_root.mkdir()
        snapshots = {
            "pilot_campaign_manifest": pilot_state["manifest_path"],
            "pilot_analysis": analysis_path,
            "full_rerun_authorization": authorization_path,
        }
        identities: dict[str, dict[str, object]] = {}
        for role, source in snapshots.items():
            destination = snapshot_root / Path(source).name
            shutil.copy2(source, destination)
            identities[role] = _relative_identity(destination, root)
        receipt: dict[str, object] = {
            "schema": EXECUTION_AUTHORIZATION_SCHEMA,
            "campaign_fingerprint": manifest["fingerprint"],
            "pilot_campaign_fingerprint": pilot_manifest["fingerprint"],
            "expected_case_count": 102,
            "scenarios": list(SCENARIOS),
            "pilot_status": analysis["status"],
            "snapshots": identities,
        }
        receipt["fingerprint"] = canonical_fingerprint(receipt)
        atomic_write_json(receipt_path, receipt)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return receipt


def _validate_uniform_checkpoint_evidence(
    repository: Path, evidence_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    receipt = _load_json(evidence_path, label="uniform H-VAWT paper evidence receipt")
    _validate_fingerprint(receipt, label="Uniform H-VAWT paper evidence receipt")
    generated = receipt.get("generated_artifacts")
    summary_identity = generated.get("summary") if isinstance(generated, Mapping) else None
    if (
        receipt.get("schema") != "vorlap-hvawt-uniform-pilot-paper-receipt-v2"
        or receipt.get("status")
        != "complete_v11_flag_with_descriptive_periodicity_followup"
        or receipt.get("raw_histories_transferred") is not True
        or receipt.get("local_history_reanalysis_performed") is not True
        or not isinstance(summary_identity, Mapping)
    ):
        raise HVAWTCompleteDLCError(
            "Uniform evidence receipt is not the reviewed local-replay and periodicity result."
        )
    summary_path = repository / str(summary_identity.get("path", ""))
    _verify_identity(repository, summary_identity, label="uniform paper summary")
    summary = _load_json(summary_path, label="uniform H-VAWT paper summary")
    periodicity = summary.get("periodicity_summary")
    if (
        summary.get("status")
        != "complete_v11_flag_with_descriptive_periodicity_followup"
        or summary.get("analysis_schema") != PILOT_ANALYSIS_SCHEMA
        or summary.get("analysis_status") != "pilot_failed_complete_hybrid_screen"
        or summary.get("campaign_kind") != FULL_CAMPAIGN_KIND
        or summary.get("case_count") != 20
        or summary.get("complete_case_count") != 20
        or summary.get("source_clip_count") != 0
        or summary.get("source_nonfinite_count") != 0
        or not math.isclose(
            float(summary.get("maximum_native_control_supplemental_nodal_force_n", math.nan)),
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not isinstance(periodicity, Mapping)
        or periodicity.get("final_revolution_record_peak_count") != 0
        or any(
            int(periodicity.get(name, 99)) >= 4
            for name in (
                "late_mean_longest_consecutive_increase_count_max",
                "late_peak_longest_consecutive_increase_count_max",
                "late_raw_velocity_rms_longest_consecutive_increase_count_max",
            )
        )
    ):
        raise HVAWTCompleteDLCError(
            "Uniform evidence does not support the reviewed checkpoint-attempt decision."
        )
    return receipt, summary_path, summary


def bind_checkpoint_authorization(
    campaign: Path, *, evidence_path: Path | None = None
) -> dict[str, object]:
    """Bind a narrow 20-case turbulent raw/complete execution authorization."""

    state = validate_campaign(
        campaign,
        rehash_external_winds=False,
        rehash_case_assets=True,
        deep_case_validation=True,
    )
    root = state["campaign_root"]
    manifest = state["manifest"]
    repository = Path(__file__).resolve().parents[1]
    policy_path, policy = _load_turbulent_checkpoint_policy(repository)
    selected = validate_turbulent_checkpoint_selection(manifest, policy)
    source_evidence = (
        repository / UNIFORM_EVIDENCE_RECEIPT_RELATIVE
        if evidence_path is None
        else Path(evidence_path).expanduser().resolve()
    )
    evidence, summary_path, summary = _validate_uniform_checkpoint_evidence(
        repository, source_evidence
    )

    snapshot_root = root / "checkpoint_execution_authorization"
    lock_path = root / ".checkpoint_execution_authorization.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt_path = root / CHECKPOINT_AUTHORIZATION_NAME
        if receipt_path.exists():
            recorded = _load_json(
                receipt_path, label="turbulent checkpoint authorization"
            )
            _load_checkpoint_authorization(root, manifest)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return recorded
        if snapshot_root.exists():
            raise HVAWTCompleteDLCError(
                "Checkpoint authorization snapshots exist without their sealed receipt."
            )
        snapshot_root.mkdir()
        snapshots = {
            "checkpoint_policy": policy_path,
            "uniform_evidence_receipt": source_evidence,
            "uniform_evidence_summary": summary_path,
        }
        identities: dict[str, dict[str, object]] = {}
        for role, source in snapshots.items():
            destination = snapshot_root / Path(source).name
            shutil.copy2(source, destination)
            identities[role] = _relative_identity(destination, root)
        receipt: dict[str, object] = {
            "schema": CHECKPOINT_AUTHORIZATION_SCHEMA,
            "status": "authorized_for_turbulent_raw_complete_checkpoint_attempt",
            "campaign_fingerprint": manifest["fingerprint"],
            "full_campaign_case_count": 102,
            "authorized_case_count": len(CHECKPOINT_CASE_INDICES),
            "authorized_case_indices": list(CHECKPOINT_CASE_INDICES),
            "scenarios": list(CHECKPOINT_SCENARIOS),
            "seed": CHECKPOINT_SEED,
            "wind_speeds_mps": list(CHECKPOINT_WIND_SPEEDS_MPS),
            "u50_smoke_required": False,
            "uniform_evidence_status": summary["status"],
            "uniform_evidence_fingerprint": evidence["fingerprint"],
            "snapshots": identities,
            "selected_cases": [
                {
                    "index": int(case["index"]),
                    "case_id": case["case_id"],
                    "row_id": case["row_id"],
                    "scenario": case["scenario"],
                    "wind_speed_mps": case["wind_speed_mps"],
                    "seed": case["seed"],
                    "common_physics_fingerprint": case[
                        "common_physics_fingerprint"
                    ],
                }
                for case in selected
            ],
            "claim_boundary": (
                "Scoped execution of 10 matched raw/complete TurbSim pairs. This is "
                "not a three-path result, seed-convergence result, complete weighted "
                "DLC result, physical validation, fatigue life, or certification evidence."
            ),
        }
        receipt["fingerprint"] = canonical_fingerprint(receipt)
        atomic_write_json(receipt_path, receipt)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return receipt


def _load_checkpoint_authorization(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _load_json(
        root / CHECKPOINT_AUTHORIZATION_NAME,
        label="turbulent checkpoint authorization",
    )
    _validate_fingerprint(receipt, label="Turbulent checkpoint authorization")
    repository = Path(__file__).resolve().parents[1]
    _policy_path, policy = _load_turbulent_checkpoint_policy(repository)
    selected = validate_turbulent_checkpoint_selection(manifest, policy)
    expected_cases = [
        {
            "index": int(case["index"]),
            "case_id": case["case_id"],
            "row_id": case["row_id"],
            "scenario": case["scenario"],
            "wind_speed_mps": case["wind_speed_mps"],
            "seed": case["seed"],
            "common_physics_fingerprint": case["common_physics_fingerprint"],
        }
        for case in selected
    ]
    if (
        receipt.get("schema") != CHECKPOINT_AUTHORIZATION_SCHEMA
        or receipt.get("status")
        != "authorized_for_turbulent_raw_complete_checkpoint_attempt"
        or receipt.get("campaign_fingerprint") != manifest.get("fingerprint")
        or receipt.get("full_campaign_case_count") != 102
        or receipt.get("authorized_case_count") != len(CHECKPOINT_CASE_INDICES)
        or tuple(receipt.get("authorized_case_indices", []))
        != CHECKPOINT_CASE_INDICES
        or tuple(receipt.get("scenarios", [])) != CHECKPOINT_SCENARIOS
        or receipt.get("seed") != CHECKPOINT_SEED
        or tuple(float(value) for value in receipt.get("wind_speeds_mps", []))
        != CHECKPOINT_WIND_SPEEDS_MPS
        or receipt.get("u50_smoke_required") is not False
        or receipt.get("selected_cases") != expected_cases
    ):
        raise HVAWTCompleteDLCError(
            "Turbulent checkpoint authorization receipt is invalid."
        )
    snapshots = receipt.get("snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != {
        "checkpoint_policy",
        "uniform_evidence_receipt",
        "uniform_evidence_summary",
    }:
        raise HVAWTCompleteDLCError(
            "Turbulent checkpoint authorization snapshots are incomplete."
        )
    for role, identity in snapshots.items():
        if not isinstance(identity, Mapping):
            raise HVAWTCompleteDLCError(
                f"Turbulent checkpoint authorization snapshot {role!r} is invalid."
            )
        _verify_identity(root, identity, label=f"checkpoint authorization {role}")
    return receipt


def _load_execution_authorization(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _load_json(
        root / EXECUTION_AUTHORIZATION_NAME,
        label="complete DLC execution authorization",
    )
    _validate_fingerprint(receipt, label="Execution authorization")
    if (
        receipt.get("schema") != EXECUTION_AUTHORIZATION_SCHEMA
        or receipt.get("campaign_fingerprint") != manifest.get("fingerprint")
        or receipt.get("expected_case_count") != 102
        or tuple(receipt.get("scenarios", [])) != SCENARIOS
        or receipt.get("pilot_status") != "pilot_passed_complete_hybrid_screen"
    ):
        raise HVAWTCompleteDLCError("Execution authorization receipt is invalid.")
    snapshots = receipt.get("snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != {
        "pilot_campaign_manifest",
        "pilot_analysis",
        "full_rerun_authorization",
    }:
        raise HVAWTCompleteDLCError("Execution authorization snapshots are incomplete.")
    for role, identity in snapshots.items():
        if not isinstance(identity, Mapping):
            raise HVAWTCompleteDLCError(
                f"Execution authorization snapshot {role!r} is invalid."
            )
        _verify_identity(root, identity, label=f"execution authorization {role}")
    return receipt


def _load_runtime(root: Path) -> dict[str, Any]:
    receipt = _load_json(root / RUNTIME_RECEIPT_NAME, label="runtime receipt")
    if receipt.get("schema") != RUNTIME_SCHEMA:
        raise HVAWTCompleteDLCError("Complete DLC runtime receipt is invalid.")
    return receipt


def _replace_command_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise HVAWTCompleteDLCError(
            f"Generated QBlade command lacks required option {option}."
        ) from exc
    command[index + 1] = value


def command_for_case(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
    python_executable: str = sys.executable,
    threads: int = 1,
    checkpoint_every: int = FLIGHT_RESTART_SAVE_INTERVAL_STEPS,
    progress_every: int = 1000,
) -> list[str]:
    state = validate_campaign(
        campaign,
        rehash_external_winds=False,
        rehash_case_assets=False,
        deep_case_validation=False,
    )
    cases = state["manifest"]["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise IndexError("Complete DLC case index is out of range.")
    if isinstance(threads, bool) or not 1 <= int(threads) <= 112:
        raise HVAWTCompleteDLCError("QBlade threads must be in 1..112.")
    if int(checkpoint_every) < 0 or int(progress_every) < 0:
        raise HVAWTCompleteDLCError(
            "Checkpoint and progress intervals must be nonnegative."
        )
    case = cases[case_index]
    root = state["campaign_root"]
    sim_path = root / str(case["sim"])
    _assets, runtime_fingerprint = qblade_runtime_asset_fingerprint(sim_path)
    if runtime_fingerprint != case.get("runtime_asset_fingerprint"):
        raise HVAWTCompleteDLCError(
            f"Selected case {case['case_id']} runtime assets changed."
        )
    result_dir = root / str(case["result_dir"])
    command = [
        str(python_executable),
        str(Path(__file__).resolve().parents[1] / "run_headless/run_qblade_sim.py"),
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
        str(int(checkpoint_every)),
        "--progress-every",
        str(int(progress_every)),
        "--case-metadata",
        str(result_dir / "case_metadata.json"),
    ]
    if case["force_contract"]["coupling_mode"] == "sil-vorlap":
        command.extend(("--sil-vorlap", "--sil-vorlap-history-mode", "compact"))
    return command


def _load_u50_smoke_receipt(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _load_json(
        root / U50_SMOKE_RECEIPT_NAME, label="U50 complete four-step smoke"
    )
    _validate_fingerprint(receipt, label="U50 smoke receipt")
    cases = manifest.get("cases")
    expected = cases[101] if isinstance(cases, list) and len(cases) == 102 else None
    if (
        not isinstance(expected, Mapping)
        or receipt.get("schema") != "vorlap-hvawt-complete-dlc-four-step-smoke-v2"
        or receipt.get("status") != "passed_input_runtime_finite_force_smoke_only"
        or receipt.get("campaign_fingerprint") != manifest.get("fingerprint")
        or receipt.get("case_index") != 101
        or receipt.get("case_id") != expected.get("case_id")
        or expected.get("scenario") != SCENARIOS[2]
        or not math.isclose(
            float(expected.get("wind_speed_mps", math.nan)),
            50.0,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or receipt.get("completed_steps") != 4
        or not math.isfinite(
            float(receipt.get("maximum_total_supplemental_nodal_force_n", math.nan))
        )
        or float(receipt.get("maximum_total_supplemental_nodal_force_n", 0.0))
        <= 0.0
        or not math.isfinite(
            float(receipt.get("maximum_applied_residual_force_n", math.nan))
        )
        or float(receipt.get("maximum_applied_residual_force_n", 0.0)) <= 0.0
    ):
        raise HVAWTCompleteDLCError("U50 complete four-step smoke receipt is invalid.")
    for role in ("summary", "history"):
        identity = receipt.get(role)
        if not isinstance(identity, Mapping):
            raise HVAWTCompleteDLCError(f"U50 smoke {role} identity is missing.")
        _verify_identity(root, identity, label=f"U50 smoke {role}")
    return receipt


def execute_u50_smoke(
    campaign: Path,
    *,
    qblade_root: Path,
    python_executable: str,
    threads: int,
    progress_every: int = 0,
) -> dict[str, object]:
    """Run four exact U50 complete steps before admitting the long matrix."""

    state = validate_campaign(
        campaign,
        rehash_external_winds=False,
        rehash_case_assets=False,
        deep_case_validation=False,
    )
    root = state["campaign_root"]
    manifest = state["manifest"]
    runtime = _load_runtime(root)
    _load_execution_authorization(root, manifest)
    if runtime != build_runtime_identity(qblade_root):
        raise HVAWTCompleteDLCError(
            "Current runtime differs from the bound Flight runtime."
        )
    receipt_path = root / U50_SMOKE_RECEIPT_NAME
    if receipt_path.exists():
        return _load_u50_smoke_receipt(root, manifest)
    case = manifest["cases"][101]
    if case["scenario"] != SCENARIOS[2] or not math.isclose(
        float(case["wind_speed_mps"]), 50.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise HVAWTCompleteDLCError("Case 101 is not the frozen U50 complete case.")
    output = root / "smoke_results" / str(case["case_id"])
    lock_path = root / ".u50_smoke.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if receipt_path.exists():
            recorded = _load_u50_smoke_receipt(root, manifest)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return recorded
        if output.exists():
            archive_root = root / "aborted_attempts" / "u50_four_step_smoke"
            archive_root.mkdir(parents=True, exist_ok=True)
            sequence = len(list(archive_root.glob("attempt_*"))) + 1
            shutil.move(str(output), archive_root / f"attempt_{sequence:03d}")
        output.mkdir(parents=True)
        metadata_path = output / "case_metadata.json"
        atomic_write_json(
            metadata_path,
            {
                "schema": "vorlap-hvawt-complete-dlc-smoke-metadata-v1",
                "campaign_fingerprint": manifest["fingerprint"],
                "case_id": case["case_id"],
                "case_index": 101,
                "force_contract": case["force_contract"],
            },
        )
        command = command_for_case(
            campaign,
            case_index=101,
            qblade_root=qblade_root,
            python_executable=python_executable,
            threads=threads,
            checkpoint_every=0,
            progress_every=progress_every,
        )
        _replace_command_option(command, "--output-dir", str(output))
        _replace_command_option(command, "--case-metadata", str(metadata_path))
        command.extend(("--steps", "4"))
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(threads))
        for name in (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = "1"
        subprocess.run(command, check=True, env=env)

        sim_path = root / str(case["sim"])
        summary_path = output / f"{sim_path.stem}_run_summary.json"
        summary = _load_json(summary_path, label="U50 smoke summary")
        history_path = _history_path_from_summary(summary, output)
        with history_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if (
            summary.get("failed") is not False
            or summary.get("stopped") is not False
            or summary.get("exported_results") is not True
            or summary.get("requested_steps") != 4
            or summary.get("completed_steps") != 4
            or summary.get("sil_vorlap_updates") != 4
            or summary.get("coupling_mode") != "sil-vorlap"
            or len(rows) != 4
        ):
            raise HVAWTCompleteDLCError("U50 smoke did not complete four exact steps.")
        try:
            residual_scales = [
                float(row["response_spectral_residual_source_scale"]) for row in rows
            ]
            total_forces = [
                float(row["response_total_maximum_nodal_force_n"]) for row in rows
            ]
            residual_forces = [
                float(row["response_spectral_applied_residual_maximum_nodal_force_n"])
                for row in rows
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise HVAWTCompleteDLCError(
                "U50 smoke history lacks complete-composition force diagnostics."
            ) from exc
        maximum_total = max(total_forces)
        maximum_residual = max(residual_forces)
        if (
            any(not math.isfinite(value) for value in total_forces + residual_forces)
            or any(value != 1.0 for value in residual_scales)
            or maximum_residual <= 0.0
            or maximum_total <= 0.0
        ):
            raise HVAWTCompleteDLCError(
                "U50 smoke produced nonfinite/nonpositive total or residual force."
            )
        receipt: dict[str, object] = {
            "schema": "vorlap-hvawt-complete-dlc-four-step-smoke-v2",
            "status": "passed_input_runtime_finite_force_smoke_only",
            "claim_boundary": "four-step input/runtime/finite-force smoke; not response, load, fatigue, or validation evidence",
            "campaign_fingerprint": manifest["fingerprint"],
            "case_index": 101,
            "case_id": case["case_id"],
            "completed_steps": 4,
            "maximum_total_supplemental_nodal_force_n": maximum_total,
            "maximum_applied_residual_force_n": maximum_residual,
            "summary": _relative_identity(summary_path, root),
            "history": _relative_identity(history_path, root),
        }
        receipt["fingerprint"] = canonical_fingerprint(receipt)
        atomic_write_json(receipt_path, receipt)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return receipt


def _audit_coupled_history(
    *, summary: Mapping[str, Any], result_dir: Path
) -> dict[str, object]:
    """Audit the common compact-history contract for a coupled DLC case."""

    history_meta = summary.get("sil_vorlap_history")
    if (
        not isinstance(history_meta, Mapping)
        or history_meta.get("mode") != "compact"
        or history_meta.get("rows") != TOTAL_STEPS
    ):
        raise HVAWTCompleteDLCError(
            "Coupled result lacks an exact full-length compact history receipt."
        )
    history_path = _history_path_from_summary(summary, result_dir)
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
    with history_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise HVAWTCompleteDLCError(
                "Coupled compact history lacks its force/time audit columns."
            )
        for row_index, row in enumerate(reader):
            try:
                step = int(row["step_index"])
                time_s = float(row["time_s"])
                qblade_step = float(row["qblade_timestep"])
                generated = float(row["generated_max_abs_force_component_n"])
                applied = float(row["applied_max_abs_force_component_n"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HVAWTCompleteDLCError(
                    f"Coupled compact history row {row_index} is not numeric."
                ) from exc
            if not all(
                math.isfinite(value)
                for value in (time_s, qblade_step, generated, applied)
            ):
                raise HVAWTCompleteDLCError(
                    f"Coupled compact history row {row_index} is non-finite."
                )
            if step != row_index or not math.isclose(
                qblade_step, float(row_index), rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise HVAWTCompleteDLCError(
                    "Coupled compact history step indices are not sequential."
                )
            maximum_time_error = max(
                maximum_time_error, abs(time_s - row_index * TIMESTEP_S)
            )
            maximum_generated = max(maximum_generated, abs(generated))
            maximum_applied = max(maximum_applied, abs(applied))
            rows += 1
    if rows != TOTAL_STEPS or maximum_time_error > 1.0e-9:
        raise HVAWTCompleteDLCError(
            "Coupled compact history length or time grid differs from the sealed plan."
        )
    if maximum_generated <= 0.0 or maximum_applied <= 0.0:
        raise HVAWTCompleteDLCError(
            "Coupled compact history never generated and applied a nonzero force."
        )
    return {
        "history": _relative_identity(history_path, result_dir.parents[2]),
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
    if (
        not summary_path.is_file()
        or not result_path.is_file()
        or not metadata_path.is_file()
    ):
        raise HVAWTCompleteDLCError("Terminal summary, export, or metadata is missing.")
    summary = _load_json(summary_path, label=f"summary {case['case_id']}")
    metadata = _load_json(metadata_path, label=f"metadata {case['case_id']}")
    if (
        summary.get("failed") is not False
        or summary.get("stopped") is not False
        or summary.get("allow_partial") is not False
        or summary.get("exported_results") is not True
        or summary.get("qblade_execution_mode") != "step-loop"
        or summary.get("requested_steps") != TOTAL_STEPS
        or summary.get("completed_steps") != TOTAL_STEPS
        or summary.get("coupling_mode") != case["force_contract"]["coupling_mode"]
        or metadata.get("campaign_fingerprint") != manifest["fingerprint"]
        or metadata.get("case_id") != case["case_id"]
        or metadata.get("force_contract") != case["force_contract"]
    ):
        raise HVAWTCompleteDLCError(
            "Terminal execution receipt is incomplete or mismatched."
        )
    restart = summary.get("restart")
    if (
        not isinstance(restart, Mapping)
        or int(restart.get("checkpoint_every", -1))
        not in {0, FLIGHT_RESTART_SAVE_INTERVAL_STEPS}
        or restart.get("resume_requested") is not False
        or restart.get("resume_checkpoint") is not None
        or restart.get("resume_start_step") != 0
    ):
        raise HVAWTCompleteDLCError(
            "Actual-aero case used a prohibited checkpoint interval or continuation."
        )
    reproducibility = summary.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise HVAWTCompleteDLCError("QBlade result lacks reproducibility evidence.")
    qblade_block = reproducibility.get("qblade")
    qblade = qblade_block.get("library") if isinstance(qblade_block, Mapping) else None
    if not isinstance(qblade, Mapping) or qblade.get("sha256") != QBLADE_LIBRARY_SHA256:
        raise HVAWTCompleteDLCError("QBlade result runtime identity changed.")
    if reproducibility.get("runtime_asset_fingerprint") != case.get(
        "runtime_asset_fingerprint"
    ):
        raise HVAWTCompleteDLCError("Executed QBlade case assets differ from staging.")
    result_identity = summary.get("result_file_identity")
    observed_result = _identity(result_path)
    if (
        not isinstance(result_identity, Mapping)
        or result_identity.get("sha256") != observed_result["sha256"]
        or result_identity.get("size_bytes") != observed_result["size_bytes"]
    ):
        raise HVAWTCompleteDLCError("QBlade result-file identity is invalid.")
    callback_evidence = summary.get("native_callback_evidence")
    if not isinstance(callback_evidence, dict):
        raise HVAWTCompleteDLCError("Native-callback zero evidence is missing.")
    require_zero_native_callback_evidence(
        callback_evidence, context=f"complete DLC case {case['case_id']}"
    )
    audit = _audit_qblade_fatigue_result(
        result_path,
        result_filter=root / str(case["result_filter"]),
        timestep_s=TIMESTEP_S,
        total_steps=TOTAL_STEPS,
        store_from_s=STORE_FROM_S,
    )
    if audit["maximum_abs_structural_load_channel_value"] <= 0.0:
        raise HVAWTCompleteDLCError(
            "QBlade structural load export is identically zero."
        )
    scenario = str(case["scenario"])
    updates = int(summary.get("sil_vorlap_updates", 0) or 0)
    max_force = float(summary.get("sil_vorlap_max_abs_force", 0.0) or 0.0)
    if scenario == SCENARIOS[0]:
        if updates != 0 or max_force != 0.0 or summary.get("sil_vorlap") is not False:
            raise HVAWTCompleteDLCError(
                "Raw QBlade result contains supplemental forcing."
            )
    else:
        if (
            updates != TOTAL_STEPS
            or max_force <= 0.0
            or summary.get("sil_vorlap") is not True
            or summary.get("sil_vorlap_history_mode") != "compact"
            or summary.get("sil_vorlap_source_bounds_policy") != "raise"
            or not math.isclose(
                float(summary.get("sil_vorlap_source_frequency_cap_hz", math.nan)),
                SOURCE_FREQUENCY_CAP_HZ,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise HVAWTCompleteDLCError(
                "Coupled scenario did not apply VorLap for every step."
            )
        coverage = summary.get("sil_vorlap_source_coverage")
        try:
            validate_complete_source_coverage(
                coverage,
                expected_samples=TOTAL_STEPS,
                component_node_counts=HVAWT_V8_COMPONENT_NODE_COUNTS,
            )
            validate_complete_velocity_diagnostics(
                summary.get("sil_vorlap_velocity_diagnostics"),
                expected_samples=TOTAL_STEPS,
                node_count=sum(HVAWT_V8_COMPONENT_NODE_COUNTS.values()),
            )
        except (TypeError, ValueError) as exc:
            raise HVAWTCompleteDLCError(
                f"Coupled scenario source/velocity audit failed: {exc}"
            ) from exc
        history_audit = _audit_coupled_history(summary=summary, result_dir=result_dir)
    if scenario == SCENARIOS[1]:
        if summary.get("sil_vorlap_response_diagnostics_summary") not in (None, {}):
            raise HVAWTCompleteDLCError(
                "One-way result unexpectedly activated a response closure."
            )
    if scenario == SCENARIOS[2]:
        response = summary.get("sil_vorlap_response_diagnostics_summary")
        if not isinstance(response, Mapping):
            raise HVAWTCompleteDLCError("Complete result lacks response diagnostics.")
        spectral = response.get("spectral")
        history_path = _history_path_from_summary(summary, result_dir)
        history_metrics = analyze_history(
            history_path,
            tail_start_s=STORE_FROM_S,
            tail_duration_s=DURATION_S - STORE_FROM_S,
        )
        if (
            response.get("model_type") != "hvawt_airfoil_engineering_lockin_hybrid"
            or not isinstance(spectral, Mapping)
            or spectral.get("residual_source_scale") != 1.0
            or not math.isfinite(
                float(response.get("total_maximum_nodal_force_n", math.nan))
            )
            or float(response.get("total_maximum_nodal_force_n", 0.0)) <= 0.0
            or history_metrics.get("spectral_residual_source_scale") != 1.0
            or not math.isfinite(
                float(
                    history_metrics.get(
                        "maximum_applied_residual_force_n", math.nan
                    )
                )
            )
            or float(history_metrics.get("maximum_applied_residual_force_n", 0.0))
            <= 0.0
            or not math.isfinite(
                float(
                    history_metrics.get(
                        "maximum_total_supplemental_nodal_force_n", math.nan
                    )
                )
            )
            or float(
                history_metrics.get(
                    "maximum_total_supplemental_nodal_force_n", 0.0
                )
            )
            <= 0.0
            or not math.isfinite(
                float(
                    history_metrics.get(
                        "maximum_spectral_energy_closure_error", math.nan
                    )
                )
            )
            or not math.isfinite(
                float(
                    history_metrics.get(
                        "spectral_energy_closure_tolerance", math.nan
                    )
                )
            )
            or float(
                history_metrics.get("maximum_spectral_energy_closure_error", math.inf)
            )
            > float(history_metrics.get("spectral_energy_closure_tolerance", -math.inf))
        ):
            raise HVAWTCompleteDLCError(
                "Complete result violates its three-path force contract."
            )
    else:
        history_metrics = None
    return {
        "summary": _relative_identity(summary_path, root),
        "result": _relative_identity(result_path, root),
        "metadata": _relative_identity(metadata_path, root),
        "result_audit": audit,
        "sil_vorlap_updates": updates,
        "maximum_supplemental_force_n": max_force,
        "coupled_history_audit": None if scenario == SCENARIOS[0] else history_audit,
        "complete_history_metrics": history_metrics,
    }


def _classify_case_state(
    state: Mapping[str, Any], *, case_index: int
) -> dict[str, object]:
    root = state["campaign_root"]
    cases = state["manifest"]["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise IndexError("Complete DLC case index is out of range.")
    case = cases[case_index]
    result_dir = root / str(case["result_dir"])
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
        receipt = _terminal_case_audit(root=root, manifest=state["manifest"], case=case)
    except (HVAWTCompleteDLCError, HVAWTResponsePilotError, OSError, ValueError) as exc:
        return {
            "case_id": case["case_id"],
            "status": "blocked",
            "reason": f"partial_or_invalid:{exc}",
        }
    return {"case_id": case["case_id"], "status": "complete", "receipt": receipt}


def classify_case(campaign: Path, *, case_index: int) -> dict[str, object]:
    state = validate_campaign(
        campaign,
        rehash_external_winds=False,
        rehash_case_assets=False,
        deep_case_validation=False,
    )
    return _classify_case_state(state, case_index=case_index)


def _archive_partial(root: Path, case: Mapping[str, Any], reason: str) -> Path:
    source = root / str(case["result_dir"])
    archive_root = root / "aborted_attempts" / str(case["case_id"])
    archive_root.mkdir(parents=True, exist_ok=True)
    sequence = len(list(archive_root.glob("attempt_*"))) + 1
    destination = archive_root / f"attempt_{sequence:03d}"
    shutil.move(str(source), destination)
    atomic_write_json(
        destination / "archive_reason.json",
        {"case_id": case["case_id"], "reason": reason},
    )
    return destination


def execute_case(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
    python_executable: str,
    threads: int,
    checkpoint_every: int = FLIGHT_RESTART_SAVE_INTERVAL_STEPS,
    progress_every: int = 1000,
    authorization_scope: str = "full",
) -> dict[str, object]:
    state = validate_campaign(
        campaign,
        rehash_external_winds=False,
        rehash_case_assets=False,
        deep_case_validation=False,
    )
    root = state["campaign_root"]
    manifest = state["manifest"]
    runtime = _load_runtime(root)
    cases = manifest["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise IndexError("Complete DLC case index is out of range.")
    if authorization_scope == "full":
        _load_execution_authorization(root, manifest)
        _load_u50_smoke_receipt(root, manifest)
    elif authorization_scope == "turbulent_20case_checkpoint":
        if case_index not in CHECKPOINT_CASE_INDICES:
            raise HVAWTCompleteDLCError(
                f"Case {case_index} is outside the authorized turbulent checkpoint."
            )
        _load_checkpoint_authorization(root, manifest)
    else:
        raise HVAWTCompleteDLCError(
            f"Unknown complete-DLC authorization scope: {authorization_scope!r}."
        )
    if runtime != build_runtime_identity(qblade_root):
        raise HVAWTCompleteDLCError(
            "Current runtime differs from the bound Flight runtime."
        )
    case = cases[case_index]
    lock_root = root / ".case_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{case['case_id']}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} already has an active execution lock."
            ) from exc
        classification = _classify_case_state(state, case_index=case_index)
        if classification["status"] == "complete":
            return classification
        if classification["status"] == "blocked":
            _archive_partial(root, case, str(classification.get("reason", "blocked")))
        result_dir = root / str(case["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "schema": "vorlap-hvawt-complete-dlc-case-metadata-v1",
            "campaign_fingerprint": manifest["fingerprint"],
            "case_id": case["case_id"],
            "case_index": case_index,
            "row_id": case["row_id"],
            "scenario": case["scenario"],
            "force_contract": case["force_contract"],
            "runtime_identity_fingerprint": canonical_fingerprint(runtime),
        }
        atomic_write_json(result_dir / "case_metadata.json", metadata)
        command = command_for_case(
            campaign,
            case_index=case_index,
            qblade_root=qblade_root,
            python_executable=python_executable,
            threads=threads,
            checkpoint_every=checkpoint_every,
            progress_every=progress_every,
        )
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(threads))
        for name in (
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = "1"
        subprocess.run(command, check=True, env=env)
        terminal = _classify_case_state(state, case_index=case_index)
        if terminal["status"] != "complete":
            raise HVAWTCompleteDLCError(
                f"Case {case['case_id']} did not produce an exact terminal receipt: {terminal}"
            )
        return terminal


def execute_row(
    campaign: Path,
    *,
    row_index: int,
    qblade_root: Path,
    python_executable: str,
    threads: int,
    checkpoint_every: int = FLIGHT_RESTART_SAVE_INTERVAL_STEPS,
    progress_every: int = 1000,
) -> list[dict[str, object]]:
    if row_index < 0 or row_index >= 34:
        raise IndexError("Complete DLC row index must be in 0..33.")
    receipts = []
    for case_index in range(row_index * 3, row_index * 3 + 3):
        receipts.append(
            execute_case(
                campaign,
                case_index=case_index,
                qblade_root=qblade_root,
                python_executable=python_executable,
                threads=threads,
                checkpoint_every=checkpoint_every,
                progress_every=progress_every,
            )
        )
    return receipts


def analyze_campaign(
    campaign: Path, *, output: Path | None = None
) -> dict[str, object]:
    state = validate_campaign(
        campaign, rehash_external_winds=True, rehash_case_assets=True
    )
    root = state["campaign_root"]
    manifest = state["manifest"]
    cases = []
    for case in manifest["cases"]:
        classification = _classify_case_state(state, case_index=int(case["index"]))
        cases.append(
            {
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "scenario": case["scenario"],
                **classification,
            }
        )
    complete_count = sum(row["status"] == "complete" for row in cases)
    report: dict[str, object] = {
        "schema": ANALYSIS_SCHEMA,
        "status": (
            "complete_ready_for_load_postprocess"
            if complete_count == 102
            else "incomplete"
        ),
        "campaign_fingerprint": manifest["fingerprint"],
        "expected_case_count": 102,
        "complete_case_count": complete_count,
        "all_three_scenarios_complete": complete_count == 102,
        "cases": cases,
        "postprocess_commands": [
            f"{sys.executable} run_headless/postprocess_parked_demo.py --run-root {root} --out-dir {root / 'postprocess_raw_vs_one_way'} --scenarios '{' '.join(SCENARIOS)}' --comparison-baseline {SCENARIOS[0]} --comparison-coupled {SCENARIOS[1]} --min-fatigue-samples 80000 --min-fatigue-duration 600 --min-rainflow-cycles 10",
            f"{sys.executable} run_headless/postprocess_parked_demo.py --run-root {root} --out-dir {root / 'postprocess_raw_vs_complete'} --scenarios '{' '.join(SCENARIOS)}' --comparison-baseline {SCENARIOS[0]} --comparison-coupled {SCENARIOS[2]} --min-fatigue-samples 80000 --min-fatigue-duration 600 --min-rainflow-cycles 10",
            f"{sys.executable} run_headless/postprocess_parked_demo.py --run-root {root} --out-dir {root / 'postprocess_one_way_vs_complete'} --scenarios '{' '.join(SCENARIOS)}' --comparison-baseline {SCENARIOS[1]} --comparison-coupled {SCENARIOS[2]} --min-fatigue-samples 80000 --min-fatigue-duration 600 --min-rainflow-cycles 10",
        ],
        "claim_boundary": "matched engineering load comparison; not H-VAWT experimental validation or certification evidence",
    }
    output_path = (
        root / "analysis.json"
        if output is None
        else Path(output).expanduser().resolve()
    )
    atomic_write_json(output_path, report)
    return report


def analyze_checkpoint(
    campaign: Path, *, output: Path | None = None
) -> dict[str, object]:
    """Audit the exact 20-case turbulent raw/complete checkpoint in place."""

    state = validate_campaign(
        campaign,
        rehash_external_winds=False,
        rehash_case_assets=False,
        deep_case_validation=False,
    )
    root = state["campaign_root"]
    manifest = state["manifest"]
    authorization = _load_checkpoint_authorization(root, manifest)
    repository = Path(__file__).resolve().parents[1]
    _policy_path, policy = _load_turbulent_checkpoint_policy(repository)
    selected = validate_turbulent_checkpoint_selection(manifest, policy)
    cases: list[dict[str, object]] = []
    for case in selected:
        classification = _classify_case_state(
            state, case_index=int(case["index"])
        )
        cases.append(
            {
                "case_index": int(case["index"]),
                "case_id": case["case_id"],
                "row_id": case["row_id"],
                "scenario": case["scenario"],
                "wind_speed_mps": case["wind_speed_mps"],
                "seed": case["seed"],
                **classification,
            }
        )
    complete_count = sum(row["status"] == "complete" for row in cases)
    pairs = []
    for pair_index, wind in enumerate(CHECKPOINT_WIND_SPEEDS_MPS):
        pair = cases[2 * pair_index : 2 * pair_index + 2]
        pairs.append(
            {
                "wind_speed_mps": wind,
                "seed": CHECKPOINT_SEED,
                "raw_case_index": pair[0]["case_index"],
                "complete_case_index": pair[1]["case_index"],
                "both_complete": all(row["status"] == "complete" for row in pair),
            }
        )
    report: dict[str, object] = {
        "schema": CHECKPOINT_ANALYSIS_SCHEMA,
        "status": (
            "complete_turbulent_raw_complete_execution_checkpoint"
            if complete_count == len(CHECKPOINT_CASE_INDICES)
            else "incomplete_turbulent_raw_complete_execution_checkpoint"
        ),
        "campaign_fingerprint": manifest["fingerprint"],
        "authorization_fingerprint": authorization["fingerprint"],
        "full_campaign_case_count": 102,
        "expected_checkpoint_case_count": len(CHECKPOINT_CASE_INDICES),
        "complete_checkpoint_case_count": complete_count,
        "all_checkpoint_cases_complete": complete_count
        == len(CHECKPOINT_CASE_INDICES),
        "all_pairs_complete": all(pair["both_complete"] for pair in pairs),
        "scenarios": list(CHECKPOINT_SCENARIOS),
        "case_indices": list(CHECKPOINT_CASE_INDICES),
        "cases": cases,
        "pairs": pairs,
        "reuse_contract": policy["reuse_contract"],
        "claim_boundary": (
            "Terminal execution audit of 10 matched raw/complete TurbSim pairs. "
            "Load comparisons and descriptive turbulent-response diagnostics require "
            "separate postprocessing. This is not a three-path result, seed-convergence "
            "result, complete weighted DLC result, physical validation, fatigue life, "
            "or certification evidence."
        ),
    }
    report["fingerprint"] = canonical_fingerprint(report)
    output_path = (
        root / "turbulent_20case_analysis.json"
        if output is None
        else Path(output).expanduser().resolve()
    )
    atomic_write_json(output_path, report)
    return report


__all__ = [
    "ANALYSIS_SCHEMA",
    "CAMPAIGN_SCHEMA",
    "HVAWTCompleteDLCError",
    "SCENARIOS",
    "analyze_campaign",
    "analyze_checkpoint",
    "bind_checkpoint_authorization",
    "bind_execution_authorization",
    "bind_runtime",
    "build_case_plan",
    "build_runtime_identity",
    "classify_case",
    "command_for_case",
    "execute_case",
    "execute_row",
    "execute_u50_smoke",
    "generate_campaign",
    "validate_campaign",
    "validate_turbulent_checkpoint_selection",
]
