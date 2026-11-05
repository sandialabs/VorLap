"""Prepare and qualify the matched H-VAWT complete-hybrid campaign.

The pilot keeps native QBlade aerodynamics enabled in every case.  The native
control cases instantiate the same airfoil-spectral and modal observer as the
hybrid cases but set both supplemental force paths to exactly zero.  Complete
cases combine native QBlade aerodynamics, the uncaptured zero-mean static
spectral residual, and the motion-correlated geometry-blended lock-in force.
Consequently the controls remain exact native-QBlade references while the
complete cases exercise the three-path load composition intended for H-VAWT
engineering application.
"""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .qblade_batch import (
    qblade_aerodynamic_reference_issues,
    vorlap_airfoil_dir_for_sim,
    write_sil_storage_filter,
)
from .qblade_case_prep import QBladeCasePrepRequest, prepare_qblade_external_case
from .qblade_files import read_qblade_keyword, replace_qblade_keyword_file
from .qblade_hvawt_response import separate_hvawt_flows
from .qblade_hvawt_sil import HVAWTSilResponseAdapter
from .qblade_runtime import VorLapQBladeRuntime
from .qblade_single_mode_work_map_sil import _lift_unit
from .qblade_run_support import (
    atomic_write_json,
    canonical_fingerprint,
    file_sha256,
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

CAMPAIGN_SCHEMA = "vorlap-hvawt-response-pilot-campaign-v8"
ANALYSIS_SCHEMA = "vorlap-hvawt-response-pilot-analysis-v11"
FULL_RERUN_SCHEMA = "vorlap-hvawt-response-full-rerun-plan-v8"
FULL_RERUN_SCENARIOS = (
    "raw_qblade",
    "one_way_spectral_vorlap",
    "complete_hybrid_v8",
)
RAYLEIGH_ANALYSIS_SCHEMA = "vorlap-hvawt-modal-decay-analysis-v3"
RAYLEIGH_SELECTION_SCHEMA = "vorlap-hvawt-modal-damping-selection-v2"
RAYLEIGH_DOCUMENTED_SELECTION_BASIS = (
    "qblade_parser_semantics_and_bounded_engineering_damping_assumption"
)
RAYLEIGH_SELECTED_VARIANT = "C20"
DOCUMENTED_MODE_DAMPING = {
    6: {
        "frequency_hz": 2.5115545017693828,
        "generalized_modal_mass_kg": 337.9307091909933,
        "rayleigh_energy_projected_damping_ratio": 0.007448479057063753,
    },
    12: {
        "frequency_hz": 4.572593676110874,
        "generalized_modal_mass_kg": 123.99012430136153,
        "rayleigh_energy_projected_damping_ratio": 0.009503978382703563,
    },
}
SCENARIOS = (
    "native_qblade_aero_only",
    "native_qblade_aero_plus_geometry_blended_airfoil_hybrid",
)
PILOT_WINDS_MPS = (3.0, 6.5, 7.25, 8.25, 8.5, 9.25, 11.2, 12.2, 13.2, 16.0)
PILOT_TIMESTEPS_S = (0.0075,)
PILOT_DURATION_S = 660.0
PILOT_TAIL_DURATION_S = 600.0
PILOT_RPM = 1.0
PILOT_AZIMUTH_DEG = 0.0
PREFLIGHT_WINDS_MPS = (8.25,)
PREFLIGHT_TIMESTEPS_S = (0.0075,)
PREFLIGHT_DURATION_S = 120.0
PREFLIGHT_TAIL_DURATION_S = 60.0
TWO_SPEED_TEST_WINDS_MPS = (8.25, 12.2)
TWO_SPEED_TEST_TIMESTEPS_S = (0.0075,)
TWO_SPEED_TEST_DURATION_S = 120.0
TWO_SPEED_TEST_TAIL_DURATION_S = 60.0
TWO_SPEED_TEST_SCENARIOS = SCENARIOS
FULL_CAMPAIGN_KIND = "full_660s_resonance_sweep"
PREFLIGHT_CAMPAIGN_KIND = "two_case_120s_preflight"
TWO_SPEED_TEST_CAMPAIGN_KIND = "two_speed_120s_h0_complete_test"
PREFLIGHT_PREPASS_FINGERPRINT = (
    "83bc403f61aaaf4a4a9ec1f462dacdcc6143211863df9160779619356f4a8414"
)
TWO_SPEED_TEST_PREPASS_FINGERPRINT = (
    "4793988ff9cc87f357524221dc4bed93cd9d54ee41eeabc90745d32f5341b493"
)
FLIGHT_RESTART_SAVE_INTERVAL_SIMULATED_S = 180.0
FLIGHT_RESTART_SAVE_INTERVAL_STEPS = 24_000
RESULT_FILTER_SOURCE_RELATIVE = Path("run_headless/config/parked_fatigue_results.filter")
RESULT_FILTER_SHA256 = (
    "e3f051c4e9a347de39ebbd5d10836edc02ec194c7895302b22ef59c8dac29f0b"
)
RESULT_FILTER_CHANNEL_COUNT = 26
STATIONARITY_TAIL_SPLIT_FRACTION = 0.5
TURBINE_VALIDATION_SCHEMA = "horcas2020-combined-tip-validation-assessment-v1"
TURBINE_VALIDATION_ARTIFACT = (
    "examples/horcas2020_iea10_tip_geometry/evidence/combined_tip_v1/assessment.json"
)
TURBINE_VALIDATION_SHA256 = (
    "be956c38f5672ada0a09a6a7905df48245fe9d99830dfce722447ec86109e46b"
)
TURBINE_VALIDATION_MODEL_FINGERPRINT = (
    "8b067d6f3a049f665c451f545a25dc8a0345b0a5bc829a3715f691038f899891"
)
FLIGHT_RUNTIME_IDENTITY_SCHEMA = "vorlap-hvawt-flight-runtime-identity-v1"
FLIGHT_RUNTIME_IDENTITY_NAME = "flight_qblade_runtime_identity.json"


def _result_filter_channels(path: Path) -> list[str]:
    channels = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]
    duplicates = sorted(
        {channel for channel in channels if channels.count(channel) > 1}
    )
    if duplicates:
        raise HVAWTResponsePilotError(
            "H-VAWT result filter contains duplicate channels: "
            + ", ".join(duplicates)
        )
    if len(channels) != RESULT_FILTER_CHANNEL_COUNT:
        raise HVAWTResponsePilotError(
            "H-VAWT result filter must contain exactly "
            f"{RESULT_FILTER_CHANNEL_COUNT} channels; observed {len(channels)}."
        )
    if channels[:2] != ["Time [s]", "LSS Rpm [rpm]"]:
        raise HVAWTResponsePilotError(
            "H-VAWT result filter must begin with Time and LSS RPM."
        )
    return channels


def _audit_qblade_fatigue_result(
    path: Path,
    *,
    result_filter: Path,
    timestep_s: float,
    total_steps: int,
    store_from_s: float,
) -> dict[str, object]:
    """Strictly audit the retained 26-channel QBlade load export."""

    expected_header = [
        re.sub(r"\s+", "~", row) for row in _result_filter_channels(result_filter)
    ]
    start_index = max(0, int(math.ceil(store_from_s / timestep_s - 1.0e-10)))
    expected_rows = int(total_steps) - start_index
    if expected_rows < 2:
        raise HVAWTResponsePilotError(
            "H-VAWT fatigue export plan retains fewer than two rows."
        )
    row_count = 0
    maximum_abs_load = 0.0
    maximum_time_error = 0.0
    first_time: float | None = None
    last_time: float | None = None
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            next(reader)
            next(reader)
            header = [cell.strip() for cell in next(reader)]
        except StopIteration as exc:
            raise HVAWTResponsePilotError(
                "QBlade fatigue result lacks metadata, header, or numeric rows."
            ) from exc
        if header != expected_header:
            raise HVAWTResponsePilotError(
                "QBlade fatigue result header differs from the frozen 26-channel filter."
            )
        for line_number, row in enumerate(reader, start=4):
            if len(row) != len(header):
                raise HVAWTResponsePilotError(
                    f"QBlade fatigue result row {line_number} has {len(row)} fields; "
                    f"expected {len(header)}."
                )
            try:
                values = [float(token.strip()) for token in row]
            except ValueError as exc:
                raise HVAWTResponsePilotError(
                    f"QBlade fatigue result row {line_number} is not numeric."
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise HVAWTResponsePilotError(
                    f"QBlade fatigue result row {line_number} is non-finite."
                )
            expected_time = (start_index + row_count) * timestep_s
            time_error = abs(values[0] - expected_time)
            maximum_time_error = max(maximum_time_error, time_error)
            if row_count > 0 and last_time is not None and values[0] <= last_time:
                raise HVAWTResponsePilotError(
                    "QBlade fatigue result time is not strictly increasing."
                )
            if first_time is None:
                first_time = values[0]
            last_time = values[0]
            maximum_abs_load = max(
                maximum_abs_load,
                max(abs(value) for value in values[2:]),
            )
            row_count += 1
    time_tolerance = min(5.01e-4, max(1.0e-10, 0.1 * timestep_s))
    if row_count != expected_rows:
        raise HVAWTResponsePilotError(
            f"QBlade fatigue result has {row_count} retained rows; expected {expected_rows}."
        )
    if maximum_time_error > time_tolerance:
        raise HVAWTResponsePilotError(
            "QBlade fatigue result time grid exceeds tolerance: "
            f"{maximum_time_error} > {time_tolerance} s."
        )
    return {
        "channel_count": len(expected_header),
        "row_count": row_count,
        "expected_row_count": expected_rows,
        "first_time_s": first_time,
        "last_time_s": last_time,
        "maximum_time_grid_error_s": maximum_time_error,
        "time_grid_tolerance_s": time_tolerance,
        "maximum_abs_structural_load_channel_value": maximum_abs_load,
    }


def _campaign_matrix(spec: Mapping[str, Any]) -> dict[str, object]:
    """Return the exact execution matrix selected by a sealed v8 spec."""

    pilot = spec.get("pilot")
    if not isinstance(pilot, Mapping):
        raise HVAWTResponsePilotError("Campaign spec lacks pilot settings.")
    kind = str(pilot.get("campaign_kind", ""))
    if kind == FULL_CAMPAIGN_KIND:
        expected = {
            "winds": PILOT_WINDS_MPS,
            "timesteps": PILOT_TIMESTEPS_S,
            "scenarios": SCENARIOS,
            "duration_s": PILOT_DURATION_S,
            "tail_duration_s": PILOT_TAIL_DURATION_S,
        }
    elif kind == PREFLIGHT_CAMPAIGN_KIND:
        expected = {
            "winds": PREFLIGHT_WINDS_MPS,
            "timesteps": PREFLIGHT_TIMESTEPS_S,
            "scenarios": SCENARIOS,
            "duration_s": PREFLIGHT_DURATION_S,
            "tail_duration_s": PREFLIGHT_TAIL_DURATION_S,
        }
    elif kind == TWO_SPEED_TEST_CAMPAIGN_KIND:
        expected = {
            "winds": TWO_SPEED_TEST_WINDS_MPS,
            "timesteps": TWO_SPEED_TEST_TIMESTEPS_S,
            "scenarios": TWO_SPEED_TEST_SCENARIOS,
            "duration_s": TWO_SPEED_TEST_DURATION_S,
            "tail_duration_s": TWO_SPEED_TEST_TAIL_DURATION_S,
        }
    else:
        raise HVAWTResponsePilotError(
            "pilot.campaign_kind must select the frozen 120 s preflight, "
            "two-speed matched H0/complete-v8 test, or 660 s resonance sweep."
        )
    return {"kind": kind, **expected}


def _effective_campaign_spec(
    source: Mapping[str, Any], *, campaign_kind: str | None
) -> dict[str, Any]:
    """Derive the two-case preflight from the frozen full-campaign spec."""

    spec = json.loads(json.dumps(source, allow_nan=False))
    if campaign_kind is None or campaign_kind == FULL_CAMPAIGN_KIND:
        return spec
    if campaign_kind not in {
        PREFLIGHT_CAMPAIGN_KIND,
        TWO_SPEED_TEST_CAMPAIGN_KIND,
    }:
        raise HVAWTResponsePilotError(
            f"Unsupported H-VAWT campaign kind: {campaign_kind!r}."
        )
    two_speed = campaign_kind == TWO_SPEED_TEST_CAMPAIGN_KIND
    winds = TWO_SPEED_TEST_WINDS_MPS if two_speed else PREFLIGHT_WINDS_MPS
    timesteps = (
        TWO_SPEED_TEST_TIMESTEPS_S if two_speed else PREFLIGHT_TIMESTEPS_S
    )
    scenarios = TWO_SPEED_TEST_SCENARIOS if two_speed else SCENARIOS
    pilot = spec["pilot"]
    pilot.update(
        {
            "campaign_kind": campaign_kind,
            "wind_speeds_mps": list(winds),
            "time_steps_s": list(timesteps),
            "scenarios": list(scenarios),
            "duration_s": (
                TWO_SPEED_TEST_DURATION_S if two_speed else PREFLIGHT_DURATION_S
            ),
            "assessment_tail_duration_s": (
                TWO_SPEED_TEST_TAIL_DURATION_S
                if two_speed
                else PREFLIGHT_TAIL_DURATION_S
            ),
            "expected_runs": len(winds) * len(timesteps) * len(scenarios),
            "predicted_amplitude_target_by_selected_wind_and_mode_m": [
                row
                for row in pilot[
                    "predicted_amplitude_target_by_selected_wind_and_mode_m"
                ]
                if any(
                    math.isclose(
                        float(row["wind_speed_mps"]),
                        wind,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    for wind in winds
                )
            ],
        }
    )
    if two_speed:
        pilot["static_spectrum_prepass"]["selected_wind_roles"] = [
            "mode6_peak_matched_h0_complete_demonstration",
            "mode12_peak_matched_h0_complete_demonstration",
        ]
        pilot["static_spectrum_prepass"]["fingerprint"] = (
            TWO_SPEED_TEST_PREPASS_FINGERPRINT
        )
    else:
        pilot["static_spectrum_prepass"]["selected_wind_roles"] = [
            "mode6_peak_matched_native_hybrid_preflight"
        ]
        pilot["static_spectrum_prepass"]["fingerprint"] = (
            PREFLIGHT_PREPASS_FINGERPRINT
        )
    spec["execution"] = {
        **spec["execution"],
        "array_pair_count": 4 if two_speed else 1,
        "maximum_concurrent_cases_per_node": 1,
        "allocated_physical_cores_per_array_task": 112,
        "submission_entrypoint": (
            "run_headless/submit_hvawt_two_speed_120s_test.sj"
            if two_speed
            else "run_headless/submit_hvawt_response_preflight.sj"
        ),
    }
    spec["status"] = (
        "ready_for_native_x86_two_speed_test_submission"
        if two_speed
        else "ready_for_native_x86_preflight_submission"
    )
    return spec


class HVAWTResponsePilotError(ValueError):
    """Raised when a pilot input or qualification artifact is inconsistent."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HVAWTResponsePilotError(
            f"{label} is not valid JSON: {candidate}"
        ) from exc
    if not isinstance(payload, dict):
        raise HVAWTResponsePilotError(
            f"{label} must contain one JSON object: {candidate}"
        )
    return payload


def _identity(path: Path) -> dict[str, object]:
    value = file_sha256(Path(path).resolve())
    if value is None:
        raise FileNotFoundError(path)
    return {
        "path": str(Path(path).resolve()),
        "sha256": value["sha256"],
        "size_bytes": value["size_bytes"],
    }


def _relative_identity(path: Path, root: Path) -> dict[str, object]:
    value = _identity(path)
    return {
        "path": Path(path).resolve().relative_to(root.resolve()).as_posix(),
        "sha256": value["sha256"],
        "size_bytes": value["size_bytes"],
    }


def build_flight_runtime_identity(qblade_root: Path) -> dict[str, object]:
    """Hash the exact native runtime and orchestration sources used on Flight."""

    root = Path(qblade_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or root.name != "QBladeCE_2.0.9.7":
        raise HVAWTResponsePilotError(
            f"Expected a real QBladeCE_2.0.9.7 runtime directory: {root}"
        )
    library = find_qblade_library(root)
    bridge = find_bridge_library(root, "vorlap_qblade_bridge")
    if bridge is None or bridge.is_symlink() or not bridge.is_file():
        raise HVAWTResponsePilotError(
            "The native VorLap-QBlade bridge library is unavailable or symlinked."
        )
    _dependencies, dependency_fingerprint = qblade_runtime_dependency_manifest(root)
    _python_assets, python_fingerprint = vorlap_python_runtime_manifest()
    repository = Path(__file__).resolve().parents[1]
    source_paths = {
        "pilot_model": Path(__file__).resolve(),
        "pilot_cli": repository / "run_headless/hvawt_response_pilot.py",
        "full_submission": repository / "run_headless/submit_hvawt_response_pilot.sj",
        "preflight_submission": repository
        / "run_headless/submit_hvawt_response_preflight.sj",
        "analysis_submission": repository
        / "run_headless/submit_hvawt_response_analysis.sj",
        "qblade_runner": repository / "run_headless/run_qblade_sim.py",
    }
    return {
        "schema": FLIGHT_RUNTIME_IDENTITY_SCHEMA,
        "qblade_root_name": root.name,
        "qblade_library": _identity(library),
        "qblade_runtime_dependency_fingerprint": dependency_fingerprint,
        "bridge_library": _identity(bridge),
        "vorlap_python_runtime_fingerprint": python_fingerprint,
        "orchestration_sources": {
            role: _identity(path) for role, path in source_paths.items()
        },
    }


def bind_flight_runtime_identity(
    campaign: Path, *, qblade_root: Path
) -> dict[str, object]:
    """Create or verify one immutable Flight runtime receipt for a campaign."""

    state = validate_campaign(campaign)
    root = state["campaign_root"]
    observed = build_flight_runtime_identity(qblade_root)
    lock_path = root / ".flight_qblade_runtime_identity.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt_path = root / FLIGHT_RUNTIME_IDENTITY_NAME
        if receipt_path.exists():
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise HVAWTResponsePilotError(
                    "Flight runtime identity receipt is not a regular file."
                )
            recorded = _load_json(receipt_path, label="Flight runtime identity")
            if recorded != observed:
                raise HVAWTResponsePilotError(
                    "Flight runtime differs from the campaign's first native run."
                )
        else:
            atomic_write_json(receipt_path, observed)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return observed


def _load_flight_runtime_identity(root: Path) -> dict[str, Any]:
    receipt = _load_json(
        root / FLIGHT_RUNTIME_IDENTITY_NAME, label="Flight runtime identity"
    )
    if receipt.get("schema") != FLIGHT_RUNTIME_IDENTITY_SCHEMA:
        raise HVAWTResponsePilotError("Flight runtime identity schema is invalid.")
    return receipt


def _verify_identity(root: Path, identity: Mapping[str, object], *, label: str) -> Path:
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise HVAWTResponsePilotError(f"{label} path must be campaign-relative.")
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise HVAWTResponsePilotError(
            f"{label} path escapes the campaign root."
        ) from exc
    observed = _identity(path)
    if observed["sha256"] != identity.get("sha256") or observed[
        "size_bytes"
    ] != identity.get("size_bytes"):
        raise HVAWTResponsePilotError(f"{label} identity mismatch: {path}")
    return path


def _assert_close_sequence(
    observed: Sequence[object], expected: Sequence[float], *, label: str
) -> None:
    if len(observed) != len(expected) or any(
        not math.isclose(float(actual), reference, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, reference in zip(observed, expected)
    ):
        raise HVAWTResponsePilotError(f"{label} must equal {list(expected)!r}.")


def _selected_wind_amplitude_targets(
    pilot: Mapping[str, Any],
) -> dict[tuple[float, str], float]:
    winds_expected = tuple(float(value) for value in pilot.get("wind_speeds_mps", []))
    rows = pilot.get("predicted_amplitude_target_by_selected_wind_and_mode_m")
    if not isinstance(rows, list) or len(rows) != len(winds_expected):
        raise HVAWTResponsePilotError(
            "Pilot must contain one prepass amplitude-target row per selected wind."
        )
    winds: list[object] = []
    targets: dict[tuple[float, str], float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise HVAWTResponsePilotError(
                "Pilot selected-wind amplitude targets must be JSON objects."
            )
        wind = float(row.get("wind_speed_mps", math.nan))
        winds.append(wind)
        for mode in ("6", "12"):
            value = float(row.get(f"mode{mode}_equilibrium_amplitude_m", math.nan))
            if not math.isfinite(value) or value < 0.0:
                raise HVAWTResponsePilotError(
                    f"Pilot U={wind:g} mode-{mode} amplitude target must be finite and nonnegative."
                )
            targets[(wind, mode)] = value
    _assert_close_sequence(
        winds,
        winds_expected,
        label="pilot selected-wind amplitude-target winds",
    )
    return targets


def _validate_cross_profile_turbine_validation(
    declaration: Mapping[str, Any],
    assessment: Mapping[str, Any] | None = None,
) -> None:
    """Bind the H-VAWT screen to the frozen 11-condition turbine holdout."""

    expected_text = {
        "artifact": TURBINE_VALIDATION_ARTIFACT,
        "artifact_sha256": TURBINE_VALIDATION_SHA256,
        "claim": "multi_condition_high_fidelity_numerical_wind_turbine_validation",
        "evidence_role": "primary_heldout_cross_profile_validation",
        "heldout_geometry": "COMB",
        "model_fingerprint": TURBINE_VALIDATION_MODEL_FINGERPRINT,
        "operating_condition_variable": "inclination_deg",
        "source_fidelity": "ellipsys3d_hawc2_high_fidelity_wind_turbine_fsi",
        "status": "passed",
    }
    for key, expected in expected_text.items():
        if declaration.get(key) != expected:
            raise HVAWTResponsePilotError(
                f"cross_profile_turbine_validation.{key} must equal {expected!r}."
            )
    expected_counts = {
        "operating_condition_count": 11,
        "resolved_half_peak_viv_point_count": 2,
        "training_geometry_count": 4,
    }
    for key, expected in expected_counts.items():
        if declaration.get(key) != expected:
            raise HVAWTResponsePilotError(
                f"cross_profile_turbine_validation.{key} must equal {expected}."
            )
    metrics = declaration.get("metrics")
    expected_metrics = {
        "curve_nrmse_by_target_peak": 0.12922581236076258,
        "maximum_resolved_frequency_ratio_error": 0.039160768999999984,
        "peak_amplitude_relative_error": 0.028367601686628965,
        "peak_inclination_error_deg": 0.0,
        "resolved_interval_onset_error_deg": 2.5,
    }
    if not isinstance(metrics, Mapping):
        raise HVAWTResponsePilotError(
            "Cross-profile turbine validation metrics are missing."
        )
    for key, expected in expected_metrics.items():
        if not math.isclose(
            float(metrics.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise HVAWTResponsePilotError(
                f"cross_profile_turbine_validation.metrics.{key} differs from the frozen holdout."
            )

    if assessment is None:
        return
    matrix = assessment.get("matrix")
    observed_metrics = assessment.get("metrics")
    gates = assessment.get("gates")
    envelope = assessment.get("conservative_engineering_envelope")
    if (
        assessment.get("schema") != TURBINE_VALIDATION_SCHEMA
        or assessment.get("status") != "passed"
        or assessment.get("claim")
        != "passed_multi_condition_turbine_hfm_geometry_validation"
        or assessment.get("model_fingerprint") != TURBINE_VALIDATION_MODEL_FINGERPRINT
        or not isinstance(matrix, Mapping)
        or matrix.get("heldout_geometry") != "COMB"
        or matrix.get("heldout_case_count") != 11
        or matrix.get("training_geometry_count") != 4
        or matrix.get("all_holdout_rows_reported") is not True
        or not isinstance(gates, Mapping)
        or gates.get("all") is not True
        or not isinstance(observed_metrics, Mapping)
        or not isinstance(envelope, Mapping)
    ):
        raise HVAWTResponsePilotError(
            "The cross-profile turbine validation artifact is not the passing "
            "11-condition heldout COMB assessment."
        )
    observed = {
        "curve_nrmse_by_target_peak": observed_metrics.get(
            "curve_nrmse_by_target_peak"
        ),
        "maximum_resolved_frequency_ratio_error": observed_metrics.get(
            "maximum_resolved_frequency_ratio_error"
        ),
        "peak_amplitude_relative_error": observed_metrics.get(
            "peak_amplitude_relative_error"
        ),
        "peak_inclination_error_deg": observed_metrics.get(
            "peak_inclination_error_deg"
        ),
        "resolved_interval_onset_error_deg": observed_metrics.get(
            "interval_onset_error_deg"
        ),
    }
    for key, expected in expected_metrics.items():
        if not math.isclose(
            float(observed.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1.0e-15
        ):
            raise HVAWTResponsePilotError(
                f"Cross-profile turbine assessment metric {key!r} changed."
            )
    target_interval = observed_metrics.get("target_resolved_interval")
    if (
        not isinstance(target_interval, Mapping)
        or target_interval.get("point_count") != 2
        or envelope.get("covers_all_11_heldout_rows") is not True
        or envelope.get("multiplicative_factor") != 1.3
        or not math.isclose(
            float(envelope.get("additive_floor_over_signed_saturation", math.nan)),
            0.10265640699352364,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or envelope.get("status")
        != "post_validation_discrepancy_bound_not_independent_validation"
    ):
        raise HVAWTResponsePilotError(
            "The cross-profile turbine VIV interval or conservative envelope changed."
        )


def _cross_profile_turbine_validation_source(
    repository: Path, spec: Mapping[str, Any]
) -> Path:
    declaration = spec.get("cross_profile_turbine_validation")
    if not isinstance(declaration, Mapping):
        raise HVAWTResponsePilotError(
            "Campaign spec lacks cross-profile turbine validation evidence."
        )
    _validate_cross_profile_turbine_validation(declaration)
    source = (repository.resolve() / str(declaration["artifact"])).resolve()
    try:
        source.relative_to(repository.resolve())
    except ValueError as exc:
        raise HVAWTResponsePilotError(
            "Cross-profile turbine validation artifact escapes the repository."
        ) from exc
    identity = _identity(source)
    if identity["sha256"] != declaration["artifact_sha256"]:
        raise HVAWTResponsePilotError(
            "Cross-profile turbine validation artifact SHA-256 mismatch."
        )
    assessment = _load_json(source, label="cross-profile turbine validation")
    _validate_cross_profile_turbine_validation(declaration, assessment)
    return source


def validate_campaign_spec(spec: Mapping[str, Any]) -> None:
    """Require one of the two frozen native-x86 response campaigns."""

    if spec.get("schema") != "vorlap-hvawt-response-campaign-spec-v8":
        raise HVAWTResponsePilotError(
            "Unexpected H-VAWT response campaign-spec schema."
        )
    pilot = spec.get("pilot")
    acceptance = spec.get("acceptance")
    full = spec.get("full_rerun")
    execution = spec.get("execution")
    if not isinstance(pilot, Mapping) or not isinstance(acceptance, Mapping):
        raise HVAWTResponsePilotError(
            "Campaign spec lacks pilot or acceptance settings."
        )
    if not isinstance(full, Mapping):
        raise HVAWTResponsePilotError("Campaign spec lacks the full-rerun plan.")
    if not isinstance(execution, Mapping):
        raise HVAWTResponsePilotError("Campaign spec lacks native execution settings.")
    matrix = _campaign_matrix(spec)
    winds = tuple(float(value) for value in matrix["winds"])
    timesteps = tuple(float(value) for value in matrix["timesteps"])
    scenarios = tuple(str(value) for value in matrix["scenarios"])
    duration_s = float(matrix["duration_s"])
    tail_duration_s = float(matrix["tail_duration_s"])
    expected_runs = len(winds) * len(timesteps) * len(scenarios)
    turbine_validation = spec.get("cross_profile_turbine_validation")
    envelope = spec.get("conservative_response_envelope")
    if not isinstance(turbine_validation, Mapping):
        raise HVAWTResponsePilotError(
            "Campaign spec lacks cross-profile turbine validation evidence."
        )
    _validate_cross_profile_turbine_validation(turbine_validation)
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("formula") != "A_design=1.30*A_model+0.10265640699352364*A_sat"
        or envelope.get("multiplicative_factor") != 1.3
        or not math.isclose(
            float(envelope.get("additive_floor_over_signed_saturation", math.nan)),
            0.10265640699352364,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or envelope.get("covers_all_11_heldout_conditions") is not True
        or envelope.get("fatigue_validated") is not False
        or envelope.get("application")
        != "postprocess_screening_only_does_not_modify_runtime_force_or_validation_predictions"
    ):
        raise HVAWTResponsePilotError(
            "Campaign spec conservative response envelope differs from the "
            "11-condition heldout turbine evidence."
        )
    _assert_close_sequence(
        pilot.get("wind_speeds_mps", []), winds, label="pilot winds"
    )
    _assert_close_sequence(
        pilot.get("time_steps_s", []), timesteps, label="pilot timesteps"
    )
    if tuple(pilot.get("scenarios", [])) != scenarios:
        raise HVAWTResponsePilotError(
            f"Pilot scenarios must equal {scenarios!r} for {matrix['kind']}."
        )
    required_equal = {
        "duration_s": duration_s,
        "assessment_tail_duration_s": tail_duration_s,
        "initial_azimuth_deg": PILOT_AZIMUTH_DEG,
        "expected_runs": expected_runs,
    }
    for key, expected in required_equal.items():
        if not math.isclose(
            float(pilot.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise HVAWTResponsePilotError(f"pilot.{key} must equal {expected!r}.")
    if pilot.get("native_control_external_force_policy") != (
        "same_geometry_blended_airfoil_observer_with_zero_response_force_scale_and_zero_residual_scale"
    ):
        raise HVAWTResponsePilotError(
            "Pilot native controls must use the same hybrid observer with both "
            "external-force paths exactly zero."
        )
    peaks = pilot.get("lock_in_peak_wind_speed_by_mode_mps")
    if not isinstance(peaks, Mapping):
        raise HVAWTResponsePilotError("Pilot lock-in peak winds are missing.")
    expected_peaks = {
        str(mode): float(value)
        for mode, value in peaks.items()
        if str(mode) in {"6", "12"}
    }
    if set(expected_peaks) != {"6", "12"}:
        raise HVAWTResponsePilotError(
            "Pilot peak winds must identify both tracked modes."
        )
    required_selected_peaks = (
        {"6", "12"}
        if matrix["kind"]
        in {FULL_CAMPAIGN_KIND, TWO_SPEED_TEST_CAMPAIGN_KIND}
        else {"6"}
    )
    if any(expected_peaks[mode] not in winds for mode in required_selected_peaks):
        raise HVAWTResponsePilotError(
            "Campaign must include every peak required by its execution role."
        )
    for mode, expected in expected_peaks.items():
        if not math.isclose(
            float(peaks.get(mode, math.nan)), expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise HVAWTResponsePilotError(
                f"Pilot mode-{mode} peak wind must equal {expected:g} m/s."
            )
    targets = pilot.get("predicted_peak_amplitude_target_by_mode_m")
    if not isinstance(targets, Mapping):
        raise HVAWTResponsePilotError(
            "Pilot geometry-blended airfoil amplitude targets are missing."
        )
    expected_targets = {
        mode: float(targets.get(mode, math.nan)) for mode in ("6", "12")
    }
    if any(
        not math.isfinite(value) or value < 0.0 for value in expected_targets.values()
    ):
        raise HVAWTResponsePilotError(
            "Pilot geometry-blended airfoil peak targets are invalid."
        )
    selected_targets = _selected_wind_amplitude_targets(pilot)
    for mode, peak_wind in expected_peaks.items():
        if peak_wind not in winds:
            continue
        if not math.isclose(
            selected_targets[(peak_wind, mode)],
            expected_targets[mode],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise HVAWTResponsePilotError(
                f"Pilot mode-{mode} selected-wind target does not reproduce its peak target."
            )
    prepass = pilot.get("static_spectrum_prepass")
    if not isinstance(prepass, Mapping) or (
        prepass.get("velocity_basis") != "local_rigid_body_relative_unscaled"
        or prepass.get("facility_velocity_scale_applied") != 1.0
        or prepass.get("wind_scan_step_mps") != 0.05
        or prepass.get("azimuth_increment_deg") != 30.0
    ):
        raise HVAWTResponsePilotError(
            "Pilot static-spectrum prepass contract is invalid."
        )
    gates = {
        "maximum_peak_amplitude_error_fraction_of_target": 0.3,
        "peak_amplitude_absolute_allowance_fraction_of_reference_length": 0.0,
        "maximum_tail_half_peak_amplitude_relative_difference": 0.1,
        "inactive_closure_only_amplitude_reference_m": 0.0005,
        "maximum_inactive_tail_half_peak_amplitude_absolute_difference_m": 0.0005,
        "stationarity_tail_split_fraction": STATIONARITY_TAIL_SPLIT_FRACTION,
        "maximum_modal_peak_frequency_relative_error": 0.1,
        "required_legacy_vorlap_force_scale": 0.0,
        "required_rpm": PILOT_RPM,
        "required_rpm_tolerance": 0.001,
    }
    for key, expected in gates.items():
        if not math.isclose(
            float(acceptance.get(key, math.nan)), expected, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise HVAWTResponsePilotError(f"acceptance.{key} must equal {expected!r}.")
    if acceptance.get("native_qblade_aerodynamics_required") is not True:
        raise HVAWTResponsePilotError("Native QBlade aerodynamics must remain enabled.")
    if acceptance.get("required_external_viv_mode") != "response_closure":
        raise HVAWTResponsePilotError("The pilot must use response_closure mode.")
    if (
        acceptance.get(
            "required_hybrid_response_shape_against_all_selected_wind_targets"
        )
        is not False
        or acceptance.get("required_observed_modal_peak_wind_agreement") is not False
        or acceptance.get("closure_only_amplitude_targets_are_diagnostic") is not True
        or acceptance.get("required_active_branch_stationarity_and_frequency")
        is not True
        or acceptance.get("required_hybrid_residual_source_scale") != 1.0
    ):
        raise HVAWTResponsePilotError(
            "The complete campaign must keep closure-only amplitudes diagnostic, "
            "require active-branch stationarity/frequency, and apply the residual "
            "spectral source exactly once."
        )
    if full.get("expand_only_after_pilot_pass") is not True:
        raise HVAWTResponsePilotError("Full rerun expansion must remain pilot-gated.")
    if (
        int(full.get("row_count", -1)) != 34
        or int(full.get("expected_new_runs_if_controls_are_rerun", -1)) != 102
        or tuple(full.get("scenarios", [])) != FULL_RERUN_SCENARIOS
        or full.get("execution_entrypoint")
        != "run_headless/submit_hvawt_complete_dlc_flight.sj"
        or full.get("analysis_entrypoint")
        != "run_headless/submit_hvawt_complete_dlc_analysis.sj"
    ):
        raise HVAWTResponsePilotError(
            "Full rerun must remain 34 rows by the exact raw, one-way, and "
            "complete scenarios."
        )
    if full.get("execution_authorized") is not False or (
        full.get("execution_status") != "prepared_waiting_for_pilot_pass"
    ):
        raise HVAWTResponsePilotError(
            "Full rerun must remain unauthorized before pilot acceptance."
        )
    if matrix["kind"] == FULL_CAMPAIGN_KIND:
        campaign_execution = {
            "array_pair_count": 10,
            "maximum_concurrent_cases_per_node": 1,
            "allocated_physical_cores_per_array_task": 112,
            "submission_entrypoint": "run_headless/submit_hvawt_response_pilot.sj",
        }
        expected_status = "ready_for_native_x86_full_submission"
    elif matrix["kind"] == TWO_SPEED_TEST_CAMPAIGN_KIND:
        campaign_execution = {
            "array_pair_count": 4,
            "maximum_concurrent_cases_per_node": 1,
            "allocated_physical_cores_per_array_task": 112,
            "submission_entrypoint": "run_headless/submit_hvawt_two_speed_120s_test.sj",
        }
        expected_status = "ready_for_native_x86_two_speed_test_submission"
    else:
        campaign_execution = {
            "array_pair_count": 1,
            "maximum_concurrent_cases_per_node": 1,
            "allocated_physical_cores_per_array_task": 112,
            "submission_entrypoint": "run_headless/submit_hvawt_response_preflight.sj",
        }
        expected_status = "ready_for_native_x86_preflight_submission"
    required_execution = {
        "architecture": "native_linux_x86_64",
        "qblade_threads_per_case": 112,
        "numa_domains": 8,
        "node_physical_cores": 112,
        "qblade_library_sha256": (
            "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
        ),
        "result_filter_channel_count": RESULT_FILTER_CHANNEL_COUNT,
        "result_filter_sha256": RESULT_FILTER_SHA256,
        "result_filter_role": "matched_26_channel_parked_fatigue_load_export",
        **campaign_execution,
    }
    for key, expected in required_execution.items():
        if execution.get(key) != expected:
            raise HVAWTResponsePilotError(f"execution.{key} must equal {expected!r}.")
    if spec.get("status") != expected_status:
        raise HVAWTResponsePilotError(
            "Campaign spec is not released for its native-x86 execution role."
        )
    if spec.get("scope") != (
        "complete_three_path_hvawt_engineering_application_screen_"
        "not_hvawt_validation_or_certification"
    ):
        raise HVAWTResponsePilotError(
            "Campaign scope must identify the complete three-path H-VAWT "
            "engineering application without claiming H-VAWT validation."
        )
    model = spec.get("model")
    if not isinstance(model, Mapping) or (
        model.get("response_model_type") != "hvawt_airfoil_engineering_lockin_hybrid"
        or model.get("spectral_source")
        != "NACA0018_static_CL_compact_sine_squared_capture"
        or model.get("load_composition")
        != "native_qblade_aerodynamics_plus_uncaptured_static_spectral_residual_plus_motion_correlated_geometry_blended_lockin"
        or model.get("unified_gain_model")
        != "run_headless/config/unified_viv_gain_model_v1.json"
        or model.get("unified_gain_model_fingerprint")
        != "40cbbced0d0be36d23e3e016556b063e43e72a88020b4d6b76a9af4a12bce4c2"
        or model.get("unified_gain_model_sha256")
        != "2da5e305b4dec79dfd3e569ad5c069f55bea387ca7cf85d33e03592e68f42ff9"
    ):
        raise HVAWTResponsePilotError(
            "Campaign spec does not identify the hybrid spectral model."
        )


def validate_response_model(model: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    modes = model.get("modes")
    force_sources = model.get("force_sources")
    if (
        model.get("schema_version") != 3
        or model.get("type") != "hvawt_airfoil_engineering_lockin_hybrid"
        or model.get("enabled") is not True
    ):
        raise HVAWTResponsePilotError(
            "Response model must be the enabled geometry-blended airfoil endpoint hybrid."
        )
    if not isinstance(modes, list) or [row.get("tracked_mode") for row in modes] != [
        6,
        12,
    ]:
        raise HVAWTResponsePilotError(
            "Response model must contain modes 6 and 12 in order."
        )
    if not isinstance(force_sources, Mapping) or (
        force_sources.get("native_qblade_aerodynamics") != "required_enabled"
        or force_sources.get("external_viv_mode") != "response_closure"
        or force_sources.get("include_mean_load") is not False
        or force_sources.get("coefficient_role") != "incremental_viv_screening"
    ):
        raise HVAWTResponsePilotError(
            "Response-model force-source contract is not the frozen composition."
        )
    calibration_fingerprint = str(spec.get("calibration_fingerprint_sha256", ""))
    if len(calibration_fingerprint) != 64:
        raise HVAWTResponsePilotError("Campaign calibration fingerprint is invalid.")
    for row in modes:
        mode = int(row["tracked_mode"])
        expected = DOCUMENTED_MODE_DAMPING[mode][
            "rayleigh_energy_projected_damping_ratio"
        ]
        if not math.isclose(
            float(row.get("structural_damping_ratio", math.nan)),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise HVAWTResponsePilotError(
                f"Mode {mode} engineering damping differs from the frozen Rayleigh projection."
            )
    engineering = model.get("engineering_lockin")
    if not isinstance(engineering, Mapping) or (
        engineering.get("schema_version") != 2
        or engineering.get("source_channel") != "CL"
        or engineering.get("residual_source_scale") != 1.0
        or engineering.get("response_force_scale") != 1.0
        or engineering.get("velocity_basis") != "local_rigid_body_relative_unscaled"
        or engineering.get("static_spectrum_sha256")
        != spec.get("static_spectrum_sha256")
        or engineering.get("unified_gain_model_path")
        != "unified_viv_gain_model_v1.json"
        or engineering.get("unified_gain_model_sha256")
        != spec.get("model", {}).get("unified_gain_model_sha256")
        or engineering.get("unified_gain_branch")
        != "naca0018_quasi_static_azimuth"
    ):
        raise HVAWTResponsePilotError(
            "Response model lacks the frozen airfoil static-spectrum contract."
        )
    model_spec = spec.get("model")
    if not isinstance(model_spec, Mapping) or (
        model_spec.get("geometry_blended_branch") != "airfoil_endpoint"
        or model_spec.get("response_config")
        != "run_headless/config/hvawt_response_model_v8_complete.json"
        or not math.isclose(
            float(model_spec.get("body_thickness_over_depth", math.nan)),
            0.18,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise HVAWTResponsePilotError(
            "Response model is not bound to the final geometry-blended airfoil endpoint."
        )


def _documented_selection_modes(
    selection: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = selection.get("modes")
    if not isinstance(rows, list) or len(rows) != 2:
        raise HVAWTResponsePilotError(
            "Documented Rayleigh selection must contain modes 6 and 12."
        )
    mode_map = {
        row.get("tracked_mode"): row
        for row in rows
        if isinstance(row, Mapping) and type(row.get("tracked_mode")) is int
    }
    if set(mode_map) != set(DOCUMENTED_MODE_DAMPING):
        raise HVAWTResponsePilotError(
            "Documented Rayleigh selection must contain modes 6 and 12 exactly once."
        )
    for mode, expected in DOCUMENTED_MODE_DAMPING.items():
        row = mode_map[mode]
        for key, target in expected.items():
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isclose(
                    float(value), target, rel_tol=1.0e-12, abs_tol=1.0e-12
                )
            ):
                raise HVAWTResponsePilotError(
                    f"Documented Rayleigh mode {mode} {key} differs from the frozen projection."
                )
        if row.get("qblade_effective_decay_damping_ratio") is not None:
            raise HVAWTResponsePilotError(
                "Documented input selection may not claim a measured QBlade decay ratio."
            )
    return [dict(mode_map[mode]) for mode in sorted(mode_map)]


def _rayleigh_source_case(analysis_path: Path) -> tuple[Path, dict[str, object]]:
    analysis = _load_json(analysis_path, label="Rayleigh damping evidence")
    documented = analysis.get("schema") == RAYLEIGH_SELECTION_SCHEMA
    documented_modes: list[dict[str, object]] | None = None
    if documented:
        fingerprint = analysis.get("fingerprint")
        fingerprint_payload = dict(analysis)
        fingerprint_payload.pop("fingerprint", None)
        if fingerprint != canonical_fingerprint(fingerprint_payload):
            raise HVAWTResponsePilotError(
                "Documented Rayleigh selection fingerprint is inconsistent."
            )
        if (
            analysis.get("status") != "selected"
            or analysis.get("selection_basis") != RAYLEIGH_DOCUMENTED_SELECTION_BASIS
            or analysis.get("selected_structural_encoding") != RAYLEIGH_SELECTED_VARIANT
            or analysis.get("parser_semantics") != "fixed_20th_numeric_token_is_damping"
            or analysis.get("damping_source")
            != "distributed_station_beta_with_normalized_scalar_fallback"
        ):
            raise HVAWTResponsePilotError(
                "Pilot generation requires the corrected documented C20 Rayleigh selection."
            )
        documented_modes = _documented_selection_modes(analysis)
        source = analysis.get("source_case")
        if not isinstance(source, Mapping):
            raise HVAWTResponsePilotError(
                "Documented Rayleigh selection lacks its source case."
            )
        campaign_root = (
            Path(str(source.get("campaign_root", ""))).expanduser().resolve()
        )
        expected_fingerprint = source.get("campaign_fingerprint")
    else:
        campaign_root = (
            Path(str(analysis.get("campaign_root", ""))).expanduser().resolve()
        )
        expected_fingerprint = analysis.get("campaign_fingerprint")
    classification = analysis.get("classification")
    if not documented and (
        analysis.get("schema") != RAYLEIGH_ANALYSIS_SCHEMA
        or analysis.get("status") != "complete_interpretable"
        or not isinstance(classification, Mapping)
        or classification.get("selection_allowed") is not True
        or classification.get("selected_unambiguous_substructure_encoding")
        != RAYLEIGH_SELECTED_VARIANT
    ):
        raise HVAWTResponsePilotError(
            "Pilot generation requires a complete, interpretable Rayleigh discriminator selecting C20."
        )
    manifest = _load_json(
        campaign_root / "campaign_manifest.json",
        label="Rayleigh campaign manifest",
    )
    if manifest.get("fingerprint") != expected_fingerprint:
        raise HVAWTResponsePilotError(
            "Rayleigh analysis is not bound to its campaign manifest."
        )
    candidates = [
        row
        for row in manifest.get("cases", [])
        if isinstance(row, Mapping)
        and row.get("variant") == RAYLEIGH_SELECTED_VARIANT
        and row.get("role") == "baseline"
    ]
    if len(candidates) != 1:
        raise HVAWTResponsePilotError(
            "Rayleigh campaign must contain one C20 baseline source case."
        )
    case = candidates[0]
    if documented:
        source = analysis["source_case"]
        if (
            source.get("selected_case_id") != case.get("case_id")
            or source.get("selected_variant") != RAYLEIGH_SELECTED_VARIANT
            or source.get("selected_sim_relative_path") != case.get("sim_relative_path")
        ):
            raise HVAWTResponsePilotError(
                "Documented Rayleigh source-case binding is inconsistent."
            )
    case_root = campaign_root / "cases" / str(case["case_id"])
    sim_path = case_root / str(case["sim_relative_path"])
    if not sim_path.is_file():
        raise FileNotFoundError(sim_path)
    return case_root, {
        "analysis": _identity(Path(analysis_path).expanduser().resolve()),
        "campaign_manifest": _identity(campaign_root / "campaign_manifest.json"),
        "campaign_fingerprint": manifest["fingerprint"],
        "selected_variant": RAYLEIGH_SELECTED_VARIANT,
        "selected_case_id": case["case_id"],
        "selected_sim_relative_path": case["sim_relative_path"],
        "selection_basis": analysis.get(
            "selection_basis", "active_qblade_decay_discriminator"
        ),
        "damping_source": (
            analysis.get("damping_source")
            if documented
            else classification.get("damping_source")
        ),
        "effective_c20_damping_ratio": (
            None if documented else classification.get("effective_c20_damping_ratio")
        ),
        "modes": documented_modes,
        "engineering_scalar_beta_s": analysis.get("engineering_scalar_beta_s"),
        "engineering_scalar_components": analysis.get("engineering_scalar_components"),
        "selected_substructure_relative_path": (
            analysis.get("source_case", {}).get("selected_substructure_relative_path")
            if documented
            else None
        ),
        "normalized_scalar_relative_paths": (
            analysis.get("source_case", {}).get("normalized_scalar_relative_paths")
            if documented
            else None
        ),
    }


def _replace_c20_damping(path: Path, beta_s: float) -> dict[str, object]:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    try:
        table_index = next(
            i for i, line in enumerate(lines) if line.strip() == "SUBELEMENTS"
        )
    except StopIteration as exc:
        raise HVAWTResponsePilotError(
            "Selected substructure lacks SUBELEMENTS."
        ) from exc
    header_index = table_index + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1
    if lines[header_index].split()[-2:] != ["DIA_[m]", "DAMP[-]"]:
        raise HVAWTResponsePilotError("Selected substructure is not canonical C20.")
    replacement = f"{beta_s:.12g}"
    changed = 0
    for index in range(header_index + 1, len(lines)):
        if lines[index].strip() == "SUBMEMBERS":
            break
        content = lines[index].rstrip("\r\n")
        ending = lines[index][len(content) :]
        spans = list(re.finditer(r"\S+", content))
        if not spans:
            continue
        try:
            float(spans[0].group())
        except ValueError:
            continue
        if len(spans) != 20 or not math.isclose(
            float(spans[19].group()), 0.02, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise HVAWTResponsePilotError(
                "Canonical C20 substructure row must contain DAMP=0.02 before normalization."
            )
        target = spans[19]
        lines[index] = (
            content[: target.start()] + replacement + content[target.end() :] + ending
        )
        changed += 1
    if changed == 0:
        raise HVAWTResponsePilotError(
            "Selected substructure has no C20 rows to normalize."
        )
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return {
        "relative_path": path.name,
        "source_beta_s": 0.02,
        "selected_beta_s": beta_s,
        "row_count": changed,
    }


def _apply_engineering_rayleigh_inputs(
    case_root: Path,
    rayleigh: Mapping[str, object],
) -> dict[str, object] | None:
    if rayleigh.get("selection_basis") != RAYLEIGH_DOCUMENTED_SELECTION_BASIS:
        return None
    beta = rayleigh.get("engineering_scalar_beta_s")
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isclose(float(beta), 0.0011, rel_tol=0.0, abs_tol=1.0e-15)
    ):
        raise HVAWTResponsePilotError(
            "Engineering Rayleigh scalar beta must equal 0.0011 s."
        )
    sub_relative = rayleigh.get("selected_substructure_relative_path")
    scalar_relative = rayleigh.get("normalized_scalar_relative_paths")
    if (
        not isinstance(sub_relative, str)
        or not isinstance(scalar_relative, list)
        or not all(isinstance(value, str) and value for value in scalar_relative)
    ):
        raise HVAWTResponsePilotError(
            "Rayleigh selection lacks its normalization paths."
        )
    sub_audit = _replace_c20_damping(case_root / sub_relative, float(beta))
    scalar_audits: list[dict[str, object]] = []
    for relative in scalar_relative:
        path = case_root / relative
        old = read_qblade_keyword(path, "RAYLEIGHDMP")
        replace_qblade_keyword_file(path, "RAYLEIGHDMP", f"{float(beta):.12g}")
        scalar_audits.append(
            {
                "relative_path": relative,
                "source_value": old,
                "selected_beta_s": float(beta),
            }
        )
    return {
        "policy": "distributed_station_beta_with_normalized_scalar_fallback",
        "selected_scalar_beta_s": float(beta),
        "substructure": sub_audit,
        "file_scalars": scalar_audits,
    }


def _case_label(wind_speed_mps: float, timestep_s: float, scenario: str) -> str:
    wind = f"U{wind_speed_mps:012.9f}ms".replace(".", "p")
    dt = f"dt{int(round(timestep_s * 1.0e6)):06d}us"
    suffix = "native" if scenario == SCENARIOS[0] else "response"
    return f"{wind}_{dt}_{suffix}"


def build_pilot_cases(
    campaign_root: Path, spec: Mapping[str, Any]
) -> list[dict[str, object]]:
    """Return the exact ordered wind by timestep by scenario plan."""

    validate_campaign_spec(spec)
    matrix = _campaign_matrix(spec)
    winds = tuple(float(value) for value in matrix["winds"])
    timesteps = tuple(float(value) for value in matrix["timesteps"])
    scenarios = tuple(str(value) for value in matrix["scenarios"])
    duration_s = float(matrix["duration_s"])
    tail_duration_s = float(matrix["tail_duration_s"])
    root = Path(campaign_root).expanduser().resolve(strict=False)
    rows: list[dict[str, object]] = []
    sim_name = "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
    for wind in winds:
        for timestep in timesteps:
            steps = int(round(duration_s / timestep))
            if not math.isclose(
                steps * timestep, duration_s, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise HVAWTResponsePilotError(
                    "Pilot duration is not an integer number of timesteps."
                )
            for scenario in scenarios:
                label = _case_label(wind, timestep, scenario)
                rows.append(
                    {
                        "index": len(rows),
                        "case_id": label,
                        "scenario": scenario,
                        "wind_speed_mps": wind,
                        "timestep_s": timestep,
                        "duration_s": duration_s,
                        "assessment_tail_duration_s": tail_duration_s,
                        "rotor_speed_rpm": PILOT_RPM,
                        "initial_azimuth_deg": PILOT_AZIMUTH_DEG,
                        "total_steps": steps,
                        "case_dir": f"cases/{label}",
                        "sim": f"cases/{label}/{sim_name}",
                        "result_dir": f"results/{label}",
                        "result_name": f"{label}_qblade_results",
                    }
                )
    expected_count = len(winds) * len(timesteps) * len(scenarios)
    if len(rows) != expected_count:
        raise AssertionError(
            f"Internal response expansion did not produce {expected_count} cases."
        )
    return rows


def _zero_gain_observer(response_model: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(response_model, allow_nan=False))
    copied["engineering_lockin"]["response_force_scale"] = 0.0
    copied["engineering_lockin"]["residual_source_scale"] = 0.0
    return copied


def _stage_calibrated_static_spectrum(
    *,
    repository: Path,
    model_source: Path,
    response_model: Mapping[str, Any],
    sim_path: Path,
    case_dir: Path,
) -> dict[str, object]:
    """Replace the copied case's active NACA0018 file with the frozen bytes.

    The frozen source-case archive predates the Reynolds-envelope extension used
    by the final geometry-blended airfoil branch.  Campaign preparation therefore cannot trust
    the archive's same-named HDF5 file.  Resolve the calibration's tracked source
    inside the repository, verify both declared identities, copy it into the
    extracted case, and record the preserved unused HDF5 sources separately.
    """

    repo = Path(repository).expanduser().resolve()
    model_path = Path(model_source).expanduser().resolve()
    engineering = response_model.get("engineering_lockin")
    if not isinstance(engineering, Mapping):
        raise HVAWTResponsePilotError(
            "Geometry-blended response model lacks engineering_lockin settings."
        )
    calibration_name = engineering.get("calibration_path")
    if (
        not isinstance(calibration_name, str)
        or not calibration_name
        or Path(calibration_name).is_absolute()
        or Path(calibration_name).name != calibration_name
    ):
        raise HVAWTResponsePilotError(
            "Airfoil-branch calibration_path must be one model-relative filename."
        )
    calibration_path = (model_path.parent / calibration_name).resolve()
    expected_calibration_sha = engineering.get("calibration_sha256")
    calibration_identity = _identity(calibration_path)
    if calibration_identity["sha256"] != expected_calibration_sha:
        raise HVAWTResponsePilotError(
            "Airfoil-branch calibration bytes differ from the response-model identity."
        )
    calibration_payload = _load_json(
        calibration_path, label="geometry-blended airfoil calibration"
    )
    calibration_sources = calibration_payload.get("sources")
    stationary_source = (
        calibration_sources.get("stationary_spectrum")
        if isinstance(calibration_sources, Mapping)
        else None
    )
    source_relative = (
        stationary_source.get("path")
        if isinstance(stationary_source, Mapping)
        else None
    )
    if (
        not isinstance(source_relative, str)
        or not source_relative
        or Path(source_relative).is_absolute()
    ):
        raise HVAWTResponsePilotError(
            "Airfoil-branch stationary-spectrum path must be repository-relative."
        )
    source = (repo / source_relative).resolve()
    try:
        source.relative_to(repo)
    except ValueError as exc:
        raise HVAWTResponsePilotError(
            "Calibration stationary-spectrum path escapes the repository."
        ) from exc
    source_identity = _identity(source)
    expected_source_sha = engineering.get("static_spectrum_sha256")
    if source_identity["sha256"] != expected_source_sha:
        raise HVAWTResponsePilotError(
            "Tracked stationary-spectrum bytes differ from the airfoil-branch model identity."
        )

    airfoil_dir = vorlap_airfoil_dir_for_sim(Path(sim_path).resolve())
    destination = airfoil_dir / source.name
    before = _identity(destination) if destination.is_file() else None
    if source != destination:
        shutil.copy2(source, destination)
    after = _identity(destination)
    if after["sha256"] != expected_source_sha:
        raise HVAWTResponsePilotError(
            "Staged stationary-spectrum bytes differ from the calibrated identity."
        )
    preserved = [
        _relative_identity(path, case_dir)
        for path in sorted(airfoil_dir.glob("*.h5"))
        if path.resolve() != destination.resolve()
    ]
    return {
        "policy": "replace_archived_naca0018_with_exact_geometry_blended_airfoil_source",
        "calibration": calibration_identity,
        "tracked_source": {
            "path": source.relative_to(repo).as_posix(),
            "sha256": source_identity["sha256"],
            "size_bytes": source_identity["size_bytes"],
        },
        "destination": _relative_identity(destination, case_dir),
        "archived_destination_before_replacement": before,
        "other_hdf5_sources_preserved": preserved,
    }


def _stable_prepass_float(value: float) -> float:
    """Remove immaterial BLAS/platform roundoff before sealing prepass evidence."""

    result = float(value)
    if not math.isfinite(result):
        raise HVAWTResponsePilotError(
            "Static-spectrum prepass produced a nonfinite value."
        )
    return float(format(result, ".10g"))


def run_geometry_blended_airfoil_prepass(
    parameter_file: Path,
    *,
    selected_winds_mps: Sequence[float] | None = None,
) -> dict[str, object]:
    """Scan rigid H-VAWT flow without QBlade at the selected campaign winds.

    Every section uses the same local, unscaled rigid-body-relative velocity as
    the airfoil endpoint of the final geometry-blended closure.  The scan
    cycle-averages the zero-amplitude feedback coefficient over twelve
    azimuths and combines it with each tracked mode's audited mass, stiffness,
    and damping.  The Ehrmann facility velocity correction is calibration-only
    and is not transferred into this rotating-turbine path.
    """

    path = Path(parameter_file).expanduser().resolve()
    runtime = VorLapQBladeRuntime.from_qblade_config(str(path))
    config = _load_json(path, label="geometry-blended prepass runtime config")
    response_model = config.get("hvawt_response_model")
    if not isinstance(response_model, Mapping):
        raise HVAWTResponsePilotError("Prepass runtime lacks hvawt_response_model.")
    adapter = HVAWTSilResponseAdapter(
        raw_config=response_model,
        parameter_file=path,
        runtime=runtime,
    )
    if adapter.schema_version != 3 or adapter.transfer is None:
        raise HVAWTResponsePilotError(
            "Static-spectrum prepass requires the geometry-blended airfoil endpoint."
        )
    controller = runtime.controller
    node_count = len(adapter.node_ids)
    axis = np.asarray(adapter.core.config.kinematics.axis_unit_global, dtype=float)
    omega = axis * (2.0 * math.pi * PILOT_RPM / 60.0)
    wind_grid = np.round(np.arange(2.0, 20.0 + 0.025, 0.05), 10)
    azimuth_grid = np.arange(0.0, 360.0, 30.0)
    rows: list[dict[str, float]] = []
    source_time = 0.0
    for wind in wind_grid:
        coefficients: list[list[float]] = []
        for azimuth in azimuth_grid:
            rotation = adapter.core.config.kinematics.rotation_body_to_global(
                float(azimuth)
            )
            positions = np.asarray(controller._global_pos, dtype=float) @ rotation.T
            rigid_velocity = np.cross(
                np.broadcast_to(omega, (node_count, 3)),
                positions,
            )
            chord = np.asarray(controller._base_chord, dtype=float) @ rotation.T
            normal = np.asarray(controller._base_normal, dtype=float) @ rotation.T
            ambient = np.array([float(wind), 0.0, 0.0], dtype=float)
            flows = separate_hvawt_flows(
                ambient_velocity_global_mps=ambient,
                total_structural_velocity_global_mps=rigid_velocity,
                rigid_body_velocity_global_mps=rigid_velocity,
                chord_unit_global=chord,
                normal_unit_global=normal,
                chord_m=controller._chord,
                thickness_ratio=controller._thickness,
            )
            spectral = adapter.spectral_source.step(
                source_time,
                base_relative_velocity_global_mps=(
                    flows.response_relative_velocity_global_mps
                ),
                response_effective_crossflow_speed_mps=(
                    flows.response_effective_crossflow_speed_mps
                ),
                response_section_width_m=flows.response_section_width_m,
                chord_unit_global=chord,
                normal_unit_global=normal,
                chord_m=controller._chord,
                thickness_ratio=controller._thickness,
                local_length_m=controller._local_length,
                structural_velocity_global_mps=rigid_velocity,
            )
            response_velocity = flows.response_relative_velocity_global_mps
            lift_unit = _lift_unit(
                chord,
                normal,
                np.sum(response_velocity * chord, axis=1),
                np.sum(response_velocity * normal, axis=1),
                flows.response_effective_crossflow_speed_mps,
            )
            shape_global = np.einsum(
                "ab,ibm->iam",
                rotation,
                adapter.core.config.modal_basis.real_shape_body_per_m,
            )
            section_force = (
                0.5
                * float(controller.fluid_density)
                * flows.response_effective_crossflow_speed_mps
                * np.asarray(controller._chord, dtype=float)
                * np.asarray(controller._local_length, dtype=float)
            )
            azimuth_coefficients: list[float] = []
            for mode_index in range(2):
                shape = shape_global[:, :, mode_index]
                participation = np.linalg.norm(shape, axis=1)
                direction = np.divide(
                    shape,
                    participation[:, None],
                    out=np.zeros_like(shape),
                    where=participation[:, None] > 0.0,
                )
                projection = np.abs(np.sum(lift_unit * direction, axis=1))
                azimuth_coefficients.append(
                    adapter.transfer.feedback_gain
                    * float(
                        np.sum(
                            section_force
                            * spectral.partition.captured_amplitude_by_node_mode[
                                :, mode_index
                            ]
                            * projection
                            * participation**2
                        )
                    )
                )
            coefficients.append(azimuth_coefficients)
            source_time += 0.001
        mean_coefficient = np.mean(np.asarray(coefficients, dtype=float), axis=0)
        row: dict[str, float] = {"wind_speed_mps": _stable_prepass_float(float(wind))}
        for mode_index, mode in enumerate(adapter.core.config.modes):
            structural = (
                2.0
                * adapter.structural_damping_ratios[mode_index]
                * mode.generalized_modal_mass_kg
                * math.sqrt(
                    mode.generalized_modal_stiffness_npm
                    / mode.generalized_modal_mass_kg
                )
            )
            ratio = float(mean_coefficient[mode_index] / structural)
            target = (
                adapter.transfer.limiting_amplitude_ratio
                * mode.reference_length_m
                * math.sqrt(max(0.0, 1.0 - 1.0 / ratio))
                if ratio > 1.0
                else 0.0
            )
            row[f"mode{mode.tracked_mode}_damping_ratio"] = _stable_prepass_float(ratio)
            row[f"mode{mode.tracked_mode}_equilibrium_amplitude_m"] = (
                _stable_prepass_float(target)
            )
        rows.append(row)

    peak6 = max(rows, key=lambda row: row["mode6_damping_ratio"])
    peak12 = max(rows, key=lambda row: row["mode12_damping_ratio"])
    mode6_active = [row for row in rows if row["mode6_damping_ratio"] > 1.0]
    mode12_active = [row for row in rows if row["mode12_damping_ratio"] > 1.0]
    if not mode6_active or not mode12_active:
        raise HVAWTResponsePilotError(
            "Static-spectrum prepass found no active interval for both tracked modes."
        )
    lower_mode6 = math.ceil(mode6_active[0]["wind_speed_mps"] * 2.0) / 2.0
    upper_mode6 = math.floor(mode6_active[-1]["wind_speed_mps"] * 2.0) / 2.0
    default_selected = (
        3.0,
        lower_mode6,
        peak6["wind_speed_mps"],
        upper_mode6,
        peak12["wind_speed_mps"],
        16.0,
    )
    selected = tuple(
        float(value)
        for value in (
            default_selected if selected_winds_mps is None else selected_winds_mps
        )
    )
    if not selected or len(set(selected)) != len(selected):
        raise HVAWTResponsePilotError(
            "Static-spectrum prepass selected winds must be nonempty and unique."
        )
    selected_rows = [
        next(
            row
            for row in rows
            if math.isclose(row["wind_speed_mps"], wind, rel_tol=0.0, abs_tol=1.0e-12)
        )
        for wind in selected
    ]
    summary = {
        "schema": "vorlap-hvawt-geometry-blended-airfoil-prepass-v1",
        "velocity_basis": "local_rigid_body_relative_unscaled",
        "facility_velocity_scale_applied": 1.0,
        "wind_scan_min_mps": 2.0,
        "wind_scan_max_mps": 20.0,
        "wind_scan_step_mps": 0.05,
        "azimuth_increment_deg": 30.0,
        "selected_wind_speeds_mps": list(selected),
        "peak_wind_speed_by_mode_mps": {
            "6": peak6["wind_speed_mps"],
            "12": peak12["wind_speed_mps"],
        },
        "predicted_peak_amplitude_target_by_mode_m": {
            "6": peak6["mode6_equilibrium_amplitude_m"],
            "12": peak12["mode12_equilibrium_amplitude_m"],
        },
        "predicted_amplitude_target_by_selected_wind_and_mode_m": [
            {
                "wind_speed_mps": row["wind_speed_mps"],
                "mode6_equilibrium_amplitude_m": row["mode6_equilibrium_amplitude_m"],
                "mode12_equilibrium_amplitude_m": row["mode12_equilibrium_amplitude_m"],
            }
            for row in selected_rows
        ],
        "peak_aerodynamic_to_structural_damping_ratio": {
            "6": peak6["mode6_damping_ratio"],
            "12": peak12["mode12_damping_ratio"],
        },
        "calibration_fingerprint": adapter.transfer.calibration_fingerprint,
        "static_spectrum_sha256": adapter.transfer.static_spectrum_sha256,
        "scan_fingerprint": canonical_fingerprint(rows),
    }
    summary["fingerprint"] = canonical_fingerprint(summary)
    return summary


def _patch_pilot_sim(sim_path: Path, case: Mapping[str, object]) -> None:
    patches: tuple[tuple[str, object], ...] = (
        ("PRESCRIBETYPE", 1),
        ("RPMPRESCRIBED", PILOT_RPM),
        ("INITIAL_AZIMUTH", PILOT_AZIMUTH_DEG),
        ("RELAXSTEPS", 0),
        ("RELAXTIMESTEP", float(case["timestep_s"])),
        ("INCLUDEAERO", 1),
        ("INCLUDEHYDRO", 0),
        ("TIMESTEP", float(case["timestep_s"])),
        ("NUMTIMESTEPS", int(case["total_steps"])),
        ("RAMPUP", 10.0),
        ("ADDDAMP", 5.0),
        ("WNDTYPE", 0),
        ("WNDNAME", ""),
        ("MEANINF", float(case["wind_speed_mps"])),
        ("HORANGLE", 0.0),
        ("VERTANGLE", 0.0),
        ("SHEAREXP", 0.0),
        (
            "STOREFROM",
            float(case["duration_s"]) - float(case["assessment_tail_duration_s"]),
        ),
        ("STOREREPLAY", "false"),
    )
    for keyword, value in patches:
        replace_qblade_keyword_file(sim_path, keyword, value)


def _full_rerun_pending_plan(
    *,
    repo_root: Path,
    campaign_root: Path,
    spec: Mapping[str, Any],
    campaign_fingerprint: str,
) -> dict[str, object]:
    raw_path = Path(str(spec["full_rerun"]["input_manifest"]))
    source = raw_path if raw_path.is_absolute() else repo_root / raw_path
    approved = _load_json(source, label="approved parked-DLC manifest")
    rows = approved.get("rows")
    if (
        approved.get("status") != "approved"
        or not isinstance(rows, list)
        or len(rows) != 34
    ):
        raise HVAWTResponsePilotError(
            "Full rerun source must be the approved 34-row parked-DLC manifest."
        )
    run_rows: list[dict[str, object]] = []
    for source_row in rows:
        if not isinstance(source_row, Mapping) or not isinstance(
            source_row.get("row_id"), str
        ):
            raise HVAWTResponsePilotError(
                "Approved full-rerun rows must have row_id values."
            )
        for scenario in FULL_RERUN_SCENARIOS:
            run_rows.append(
                {
                    "index": len(run_rows),
                    "source_row_id": source_row["row_id"],
                    "scenario": scenario,
                    "wind_speed_mps": source_row.get("wind_speed_mps"),
                    "wind_file": source_row.get("wind_file"),
                    "seed": source_row.get("seed"),
                    "dlc": source_row.get("dlc"),
                    "rpm": source_row.get("rpm"),
                }
            )
    return {
        "schema": FULL_RERUN_SCHEMA,
        "status": "prepared_waiting_for_pilot_pass",
        "execution_authorized": False,
        "launch_implemented": True,
        "execution_entrypoint": spec["full_rerun"]["execution_entrypoint"],
        "analysis_entrypoint": spec["full_rerun"]["analysis_entrypoint"],
        "campaign_fingerprint": campaign_fingerprint,
        "source_manifest": _identity(source),
        "source_row_count": len(rows),
        "scenarios": list(FULL_RERUN_SCENARIOS),
        "expected_runs": len(run_rows),
        "reuse_existing_turbsim_bts": bool(
            spec["full_rerun"]["reuse_existing_turbsim_bts"]
        ),
        "control_policy": spec["full_rerun"]["control_policy"],
        "authorization_gate": "pilot_analysis.status == pilot_passed_complete_hybrid_screen",
        "runs": run_rows,
    }


def generate_campaign(
    *,
    campaign_root: Path,
    rayleigh_analysis_path: Path,
    repo_root: Path | None = None,
    spec_path: Path | None = None,
    runtime_vorlap_root: Path | None = None,
    campaign_kind: str | None = None,
) -> Path:
    """Generate one sealed native-x86 response campaign without launching QBlade."""

    repository = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).expanduser().resolve()
    )
    spec_source = (
        repository / "run_headless/config/hvawt_response_campaign_spec.json"
        if spec_path is None
        else Path(spec_path).expanduser().resolve()
    )
    source_spec = _load_json(spec_source, label="H-VAWT response campaign spec")
    spec = _effective_campaign_spec(source_spec, campaign_kind=campaign_kind)
    validate_campaign_spec(spec)
    turbine_validation_source = _cross_profile_turbine_validation_source(
        repository, spec
    )
    model_source_raw = Path(str(spec["model"]["response_config"]))
    model_source = (
        model_source_raw
        if model_source_raw.is_absolute()
        else repository / model_source_raw
    )
    response_model = _load_json(model_source, label="H-VAWT response model")
    validate_response_model(response_model, spec)
    source_case, rayleigh = _rayleigh_source_case(rayleigh_analysis_path)
    source_sim_relative = str(rayleigh["selected_sim_relative_path"])
    source_sim = source_case / source_sim_relative
    aerodynamic_issues = qblade_aerodynamic_reference_issues(source_sim)
    if aerodynamic_issues:
        raise HVAWTResponsePilotError(
            "Selected Rayleigh case lacks native aerodynamic assets: "
            + "; ".join(aerodynamic_issues)
        )

    root = Path(campaign_root).expanduser().resolve(strict=False)
    if root.exists():
        raise FileExistsError(
            f"Pilot campaign already exists and will not be replaced: {root}"
        )
    root.mkdir(parents=True, exist_ok=False)
    (root / "inputs").mkdir()
    (root / "cases").mkdir()
    (root / "results").mkdir()
    copied_spec = root / "inputs" / spec_source.name
    copied_model = root / "inputs" / model_source.name
    copied_rayleigh = root / "inputs" / Path(rayleigh_analysis_path).name
    copied_turbine_validation = (
        root / "inputs" / "horcas2020_combined_tip_validation_assessment.json"
    )
    atomic_write_json(copied_spec, spec)
    shutil.copy2(model_source, copied_model)
    shutil.copy2(Path(rayleigh_analysis_path).expanduser().resolve(), copied_rayleigh)
    shutil.copy2(turbine_validation_source, copied_turbine_validation)
    result_filter_source = repository / RESULT_FILTER_SOURCE_RELATIVE
    if not result_filter_source.is_file():
        raise FileNotFoundError(result_filter_source)
    if _identity(result_filter_source)["sha256"] != RESULT_FILTER_SHA256:
        raise HVAWTResponsePilotError(
            "Tracked parked-fatigue result filter differs from the frozen identity."
        )
    _result_filter_channels(result_filter_source)
    result_filter = root / "inputs" / "hvawt_complete_fatigue_results.filter"
    shutil.copy2(result_filter_source, result_filter)

    case_rows = build_pilot_cases(root, spec)
    python_root = (
        repository
        if runtime_vorlap_root is None
        else Path(runtime_vorlap_root).expanduser().resolve()
    )
    for case in case_rows:
        case_dir = root / str(case["case_dir"])
        shutil.copytree(source_case, case_dir, symlinks=False)
        case["rayleigh_input_normalization"] = _apply_engineering_rayleigh_inputs(
            case_dir, rayleigh
        )
        sim_path = case_dir / source_sim_relative
        expected_sim = root / str(case["sim"])
        if sim_path.name != expected_sim.name:
            raise HVAWTResponsePilotError(
                f"Selected Rayleigh simulation name {sim_path.name!r} differs from pilot plan."
            )
        case["static_spectrum_staging"] = _stage_calibrated_static_spectrum(
            repository=repository,
            model_source=model_source,
            response_model=response_model,
            sim_path=sim_path,
            case_dir=case_dir,
        )
        _patch_pilot_sim(sim_path, case)
        scenario_model = (
            _zero_gain_observer(response_model)
            if case["scenario"] == SCENARIOS[0]
            else json.loads(json.dumps(response_model, allow_nan=False))
        )
        prepared = prepare_qblade_external_case(
            QBladeCasePrepRequest(
                sim_path=sim_path,
                airfoil_dir=vorlap_airfoil_dir_for_sim(sim_path),
                sample_step=float(case["timestep_s"]),
                node_source="structural",
                include_struts=True,
                include_tower=False,
                n_freq_depth=200,
                force_scale=0.0,
                include_mean_load=False,
                source_bounds_policy="raise",
                # The accepted dt=0.0075 s qualification requires exact
                # Nyquist truncation.  Bind the cap into every generated
                # pilot runtime rather than relying on an ambient environment
                # variable that is absent from the sealed runtime identity.
                source_frequency_cap_hz=(
                    0.5 / float(case["timestep_s"])
                ),
                hvawt_response_model=scenario_model,
                hvawt_response_artifact_dir=model_source.parent,
                inflow_vec=(1.0, 0.0, 0.0),
                include_default_python_path=True,
                default_python_path=python_root,
                absolute_source_parameter_dir=True,
                debug=False,
            )
        )
        storage_filter = case_dir / "hvawt_response_pilot_sil_runtime.filter"
        storage = write_sil_storage_filter(
            result_filter=result_filter,
            storage_filter=storage_filter,
            parameter_file=prepared.parameter_file_path,
        )
        replace_qblade_keyword_file(sim_path, "FILTERFILE", storage_filter.name)
        assets, asset_fingerprint = qblade_runtime_asset_fingerprint(sim_path)
        runtime_config = prepared.parameter_file_path.relative_to(case_dir).as_posix()
        case.update(
            {
                "sim": sim_path.relative_to(root).as_posix(),
                "runtime_config": f"{case['case_dir']}/{runtime_config}",
                "runtime_asset_fingerprint": asset_fingerprint,
                "runtime_asset_count": len(assets),
                "storage_filter": storage_filter.relative_to(root).as_posix(),
                "storage_filter_sha256": storage["sha256"],
                "storage_filter_channel_count": storage["channel_count"],
                "result_filter": result_filter.relative_to(root).as_posix(),
                "external_force_policy": (
                    "exact_zero_complete_hybrid_observer"
                    if case["scenario"] == SCENARIOS[0]
                    else "native_qblade_plus_uncaptured_static_spectral_residual_plus_geometry_blended_modal_work_closure"
                ),
            }
        )

    campaign_matrix = _campaign_matrix(spec)
    campaign_winds = tuple(float(value) for value in campaign_matrix["winds"])
    campaign_timesteps = tuple(
        float(value) for value in campaign_matrix["timesteps"]
    )
    prepass_case = next(
        row
        for row in case_rows
        if row["scenario"] == SCENARIOS[1]
        and math.isclose(
            float(row["timestep_s"]),
            campaign_timesteps[0],
            rel_tol=0.0,
            abs_tol=0.0,
        )
    )
    prepass = run_geometry_blended_airfoil_prepass(
        root / str(prepass_case["runtime_config"]),
        selected_winds_mps=campaign_winds,
    )
    _assert_close_sequence(
        prepass["selected_wind_speeds_mps"],
        campaign_winds,
        label="executed prepass winds",
    )
    for mode, target in spec["pilot"][
        "predicted_peak_amplitude_target_by_mode_m"
    ].items():
        if not math.isclose(
            float(prepass["predicted_peak_amplitude_target_by_mode_m"][mode]),
            float(target),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise HVAWTResponsePilotError(
                f"Executed prepass mode-{mode} amplitude target differs from spec."
            )
    if prepass["predicted_amplitude_target_by_selected_wind_and_mode_m"] != spec[
        "pilot"
    ].get("predicted_amplitude_target_by_selected_wind_and_mode_m"):
        raise HVAWTResponsePilotError(
            "Executed prepass selected-wind amplitude targets differ from spec."
        )
    declared_prepass = spec["pilot"]["static_spectrum_prepass"]
    if prepass["fingerprint"] != declared_prepass.get("fingerprint") or prepass[
        "scan_fingerprint"
    ] != declared_prepass.get("scan_fingerprint"):
        raise HVAWTResponsePilotError(
            "Executed static-spectrum prepass fingerprint differs from spec."
        )

    manifest: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "prepared_not_run",
        "sealed_inputs": True,
        "case_count": len(case_rows),
        "matrix": {
            "campaign_kind": campaign_matrix["kind"],
            "wind_speeds_mps": list(campaign_winds),
            "time_steps_s": list(campaign_timesteps),
            "scenarios": list(campaign_matrix["scenarios"]),
            "duration_s": float(campaign_matrix["duration_s"]),
            "assessment_tail_duration_s": float(
                campaign_matrix["tail_duration_s"]
            ),
            "rpm": PILOT_RPM,
            "initial_azimuth_deg": PILOT_AZIMUTH_DEG,
        },
        "inputs": {
            "campaign_spec": _relative_identity(copied_spec, root),
            "response_model": _relative_identity(copied_model, root),
            "rayleigh_analysis": _relative_identity(copied_rayleigh, root),
            "cross_profile_turbine_validation": _relative_identity(
                copied_turbine_validation, root
            ),
            "result_filter": _relative_identity(result_filter, root),
        },
        "rayleigh_selection": rayleigh,
        "static_spectrum_prepass": prepass,
        "calibration_fingerprint_sha256": spec["calibration_fingerprint_sha256"],
        "static_spectrum_sha256": spec["static_spectrum_sha256"],
        "cross_profile_turbine_validation": {
            "role": spec["cross_profile_turbine_validation"]["evidence_role"],
            "heldout_geometry": "COMB",
            "operating_condition_count": 11,
            "status": "passed",
            "model_fingerprint": TURBINE_VALIDATION_MODEL_FINGERPRINT,
        },
        "conservative_response_envelope": spec["conservative_response_envelope"],
        "native_control_policy": (
            "same geometry-blended airfoil observer, modal basis, and spectra with "
            "response-force and residual-source scales exactly zero"
        ),
        "force_composition": {
            "native_qblade_aerodynamics": "enabled",
            "legacy_vorlap_force_scale": 0.0,
            "single_mode_response_model": "absent",
            "airfoil_spectral_residual": "disabled in native and enabled once in complete scenario",
            "structure_conditioned_modal_capture": "disabled in native and enabled once in complete scenario",
            "native_callback_activity_allowed_in_sil": False,
        },
        "cases": case_rows,
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    manifest_path = root / "campaign_manifest.json"
    atomic_write_json(manifest_path, manifest)
    pending = _full_rerun_pending_plan(
        repo_root=repository,
        campaign_root=root,
        spec=spec,
        campaign_fingerprint=str(manifest["fingerprint"]),
    )
    atomic_write_json(root / "full_rerun_plan_pending.json", pending)
    validate_campaign(manifest_path)
    return manifest_path


def _manifest_path(value: Path) -> Path:
    candidate = Path(value).expanduser().resolve()
    return candidate / "campaign_manifest.json" if candidate.is_dir() else candidate


def validate_campaign(value: Path) -> dict[str, Any]:
    """Recheck the exact matrix, critical input bytes, and force composition."""

    manifest_path = _manifest_path(value)
    root = manifest_path.parent
    manifest = _load_json(manifest_path, label="H-VAWT pilot campaign manifest")
    if (
        manifest.get("schema") != CAMPAIGN_SCHEMA
        or manifest.get("sealed_inputs") is not True
    ):
        raise HVAWTResponsePilotError("Not a sealed H-VAWT response pilot campaign.")
    expected_fingerprint = manifest.get("fingerprint")
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("fingerprint", None)
    if expected_fingerprint != canonical_fingerprint(fingerprint_payload):
        raise HVAWTResponsePilotError("Pilot campaign manifest fingerprint mismatch.")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise HVAWTResponsePilotError("Pilot campaign input identities are missing.")
    for key in (
        "campaign_spec",
        "response_model",
        "rayleigh_analysis",
        "cross_profile_turbine_validation",
        "result_filter",
    ):
        identity = inputs.get(key)
        if not isinstance(identity, Mapping):
            raise HVAWTResponsePilotError(f"Pilot input identity {key!r} is missing.")
        _verify_identity(root, identity, label=f"pilot input {key}")
    spec = _load_json(
        root / str(inputs["campaign_spec"]["path"]), label="copied campaign spec"
    )
    model = _load_json(
        root / str(inputs["response_model"]["path"]), label="copied response model"
    )
    validate_campaign_spec(spec)
    validate_response_model(model, spec)
    copied_result_filter = root / str(inputs["result_filter"]["path"])
    if (
        _identity(copied_result_filter)["sha256"] != RESULT_FILTER_SHA256
        or len(_result_filter_channels(copied_result_filter))
        != RESULT_FILTER_CHANNEL_COUNT
    ):
        raise HVAWTResponsePilotError(
            "Campaign result filter is not the frozen matched fatigue-load export."
        )
    turbine_validation_path = root / str(
        inputs["cross_profile_turbine_validation"]["path"]
    )
    turbine_validation = _load_json(
        turbine_validation_path, label="copied cross-profile turbine validation"
    )
    _validate_cross_profile_turbine_validation(
        spec["cross_profile_turbine_validation"], turbine_validation
    )
    declared_turbine_validation = manifest.get("cross_profile_turbine_validation")
    if declared_turbine_validation != {
        "role": "primary_heldout_cross_profile_validation",
        "heldout_geometry": "COMB",
        "operating_condition_count": 11,
        "status": "passed",
        "model_fingerprint": TURBINE_VALIDATION_MODEL_FINGERPRINT,
    } or manifest.get("conservative_response_envelope") != spec.get(
        "conservative_response_envelope"
    ):
        raise HVAWTResponsePilotError(
            "Pilot manifest does not bind the frozen turbine validation and "
            "conservative response envelope."
        )
    expected_cases = build_pilot_cases(root, spec)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected_cases):
        raise HVAWTResponsePilotError(
            f"Response campaign must contain exactly {len(expected_cases)} cases."
        )
    expected_matrix = [
        (row["scenario"], row["wind_speed_mps"], row["timestep_s"], row["total_steps"])
        for row in expected_cases
    ]
    observed_matrix = [
        (
            row.get("scenario"),
            row.get("wind_speed_mps"),
            row.get("timestep_s"),
            row.get("total_steps"),
        )
        for row in cases
        if isinstance(row, Mapping)
    ]
    if observed_matrix != expected_matrix:
        raise HVAWTResponsePilotError(
            "Response case matrix/order differs from the frozen campaign plan."
        )
    for case in cases:
        assert isinstance(case, Mapping)
        sim_path = root / str(case["sim"])
        runtime_config_path = root / str(case["runtime_config"])
        storage_filter = root / str(case["storage_filter"])
        if (
            not sim_path.is_file()
            or not runtime_config_path.is_file()
            or not storage_filter.is_file()
        ):
            raise FileNotFoundError(f"Pilot case {case['case_id']} is incomplete.")
        expected_keywords = {
            "PRESCRIBETYPE": 1,
            "RPMPRESCRIBED": PILOT_RPM,
            "INITIAL_AZIMUTH": PILOT_AZIMUTH_DEG,
            "INCLUDEAERO": 1,
            "INCLUDEHYDRO": 0,
            "TIMESTEP": float(case["timestep_s"]),
            "NUMTIMESTEPS": int(case["total_steps"]),
            "WNDTYPE": 0,
            "MEANINF": float(case["wind_speed_mps"]),
            "HORANGLE": 0.0,
            "VERTANGLE": 0.0,
            "SHEAREXP": 0.0,
            "STOREFROM": float(case["duration_s"])
            - float(case["assessment_tail_duration_s"]),
        }
        for keyword, expected in expected_keywords.items():
            raw = read_qblade_keyword(sim_path, keyword)
            if raw is None or not math.isclose(
                float(raw), float(expected), rel_tol=0.0, abs_tol=1.0e-10
            ):
                raise HVAWTResponsePilotError(
                    f"Case {case['case_id']} {keyword} differs: observed={raw!r}, expected={expected!r}."
                )
        if read_qblade_keyword(sim_path, "FILTERFILE") != storage_filter.name:
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} does not use its SIL storage filter."
            )
        config = _load_json(
            runtime_config_path, label=f"runtime config for {case['case_id']}"
        )
        staging = case.get("static_spectrum_staging")
        if not isinstance(staging, Mapping) or staging.get("policy") != (
            "replace_archived_naca0018_with_exact_geometry_blended_airfoil_source"
        ):
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} lacks geometry-blended static-spectrum staging provenance."
            )
        staged_destination = staging.get("destination")
        if not isinstance(staged_destination, Mapping):
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} lacks its staged static-spectrum identity."
            )
        staged_path = _verify_identity(
            root / str(case["case_dir"]),
            staged_destination,
            label=f"case {case['case_id']} staged static spectrum",
        )
        if (
            staged_path.name != "NACA0018.h5"
            or staged_destination.get("sha256") != spec["static_spectrum_sha256"]
        ):
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} does not stage the calibrated NACA0018 source."
            )
        source_manifest = config.get("airfoil_source_manifest")
        naca_rows = (
            [
                row
                for row in source_manifest
                if isinstance(row, Mapping) and row.get("airfoil_id") == "NACA0018"
            ]
            if isinstance(source_manifest, list)
            else []
        )
        if (
            len(naca_rows) != 1
            or naca_rows[0].get("sha256") != spec["static_spectrum_sha256"]
        ):
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} runtime manifest lacks the calibrated NACA0018 bytes."
            )
        hvawt = config.get("hvawt_response_model")
        if (
            float(config.get("force_scale", math.nan)) != 0.0
            or config.get("include_mean_load") is not False
            or not math.isclose(
                float(config.get("source_frequency_cap_hz", math.nan)),
                0.5 / float(case["timestep_s"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or "response_model" in config
            or not isinstance(hvawt, Mapping)
        ):
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} violates the no-double-count or Nyquist force contract."
            )
        engineering = hvawt.get("engineering_lockin")
        native_control = case["scenario"] == SCENARIOS[0]
        expected_force_scale = 0.0 if native_control else 1.0
        expected_residual_scale = 0.0 if native_control else 1.0
        if not isinstance(engineering, Mapping) or (
            float(engineering.get("response_force_scale", math.nan))
            != expected_force_scale
            or float(engineering.get("residual_source_scale", math.nan))
            != expected_residual_scale
        ):
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} geometry-blended force composition differs from its scenario."
            )
        observed_storage = _identity(storage_filter)
        if (
            observed_storage["sha256"] != case.get("storage_filter_sha256")
            or int(observed_storage["size_bytes"]) <= 0
        ):
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} storage-filter identity mismatch."
            )
        _assets, fingerprint = qblade_runtime_asset_fingerprint(sim_path)
        if fingerprint != case.get("runtime_asset_fingerprint"):
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} runtime assets changed after generation."
            )
    pending_path = root / "full_rerun_plan_pending.json"
    pending = _load_json(pending_path, label="pending full-rerun plan")
    if (
        pending.get("schema") != FULL_RERUN_SCHEMA
        or pending.get("execution_authorized") is not False
        or pending.get("expected_runs") != 102
        or pending.get("launch_implemented") is not True
        or tuple(pending.get("scenarios", [])) != FULL_RERUN_SCENARIOS
        or pending.get("campaign_fingerprint") != expected_fingerprint
    ):
        raise HVAWTResponsePilotError(
            "Pending full-rerun plan is not the gated 102-run three-path plan."
        )
    return {
        "valid": True,
        "campaign_root": root,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def command_for_case(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
    python_executable: str = sys.executable,
    threads: int = 1,
    checkpoint_every: int = 0,
    progress_every: int = 1000,
    resume: bool = False,
) -> list[str]:
    state = validate_campaign(campaign)
    manifest = state["manifest"]
    root = state["campaign_root"]
    cases = manifest["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise IndexError(f"Pilot case index must be in 0..{len(cases) - 1}.")
    if isinstance(threads, bool) or not 1 <= int(threads) <= 112:
        raise HVAWTResponsePilotError(
            "H-VAWT QBlade threads must be between 1 and 112."
        )
    if int(checkpoint_every) < 0 or int(progress_every) < 0:
        raise HVAWTResponsePilotError(
            "Checkpoint/progress intervals must be nonnegative."
        )
    if resume:
        raise HVAWTResponsePilotError(
            "Actual-aero H-VAWT checkpoint continuation is not yet approved: QBlade "
            "loadProject has not demonstrated time/solver-state preserving resume. "
            "Periodic recovery snapshots may be saved, but a partial case must still "
            "restart from a clean result directory."
        )
    case = cases[case_index]
    sim_path = root / str(case["sim"])
    result_dir = root / str(case["result_dir"])
    result_filter = root / str(case["result_filter"])
    runner = Path(__file__).resolve().parents[1] / "run_headless/run_qblade_sim.py"
    command = [
        str(python_executable),
        str(runner),
        "--sim",
        str(sim_path),
        "--qblade-root",
        str(Path(qblade_root).expanduser().resolve()),
        "--output-dir",
        str(result_dir),
        "--result-name",
        str(case["result_name"]),
        "--result-filter",
        str(result_filter),
        "--threads",
        str(int(threads)),
        "--sil-vorlap",
        "--coupling-mode",
        "sil-vorlap",
        "--sil-vorlap-history-mode",
        "compact",
        "--checkpoint-every",
        str(int(checkpoint_every)),
        "--progress-every",
        str(int(progress_every)),
        "--case-metadata",
        str(result_dir / "case_metadata.json"),
    ]
    return command


def classify_case_execution(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path | None = None,
) -> dict[str, object]:
    """Classify one actual-aero case without modifying its result directory.

    QBlade structural-project checkpoint continuation was demonstrated not to
    preserve the coupled time/solver state. A case is launchable only from an
    empty result directory. A fully audited terminal result is reusable as-is;
    every other nonempty state is blocked for an explicit clean restart.
    """

    state = validate_campaign(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    cases = manifest["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise IndexError(f"Pilot case index must be in 0..{len(cases) - 1}.")
    case = cases[case_index]
    result_dir = root / str(case["result_dir"])
    if not result_dir.exists() or (
        result_dir.is_dir() and not any(result_dir.iterdir())
    ):
        return {
            "case_id": case["case_id"],
            "status": "runnable",
            "reason": "fresh_case_no_prior_artifacts",
        }
    if not result_dir.is_dir():
        return {
            "case_id": case["case_id"],
            "status": "blocked",
            "reason": "result_path_is_not_a_directory",
        }

    inputs = manifest["inputs"]
    spec = _load_json(
        root / str(inputs["campaign_spec"]["path"]), label="campaign spec"
    )
    model = _load_json(
        root / str(inputs["response_model"]["path"]), label="response model"
    )
    expected_runtime_identity = _load_flight_runtime_identity(root)
    if qblade_root is not None:
        observed_runtime_identity = build_flight_runtime_identity(qblade_root)
        if observed_runtime_identity != expected_runtime_identity:
            return {
                "case_id": case["case_id"],
                "status": "blocked",
                "reason": "current_flight_runtime_differs_from_bound_receipt",
            }
    result = _case_result_or_error(
        root=root,
        case=case,
        model=model,
        acceptance=spec["acceptance"],
        campaign_kind=str(_campaign_matrix(spec)["kind"]),
        peak_winds=spec["pilot"]["lock_in_peak_wind_speed_by_mode_mps"],
        amplitude_targets=spec["pilot"]["predicted_peak_amplitude_target_by_mode_m"],
        campaign_fingerprint=str(manifest["fingerprint"]),
        expected_runtime_identity=expected_runtime_identity,
    )
    if (
        result.get("status") == "complete"
        and result.get("execution_receipt_pass") is True
    ):
        return {
            "case_id": case["case_id"],
            "status": "complete",
            "reason": "exact_audited_terminal_result",
            "physics_gates_passed": bool(result.get("passed")),
        }
    return {
        "case_id": case["case_id"],
        "status": "blocked",
        "reason": (
            "partial_or_invalid_actual_aero_artifacts_are_not_restartable:"
            f"{result.get('status')}:{result.get('problem', 'missing terminal summary')}"
        ),
    }


def archive_incomplete_case_attempt(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
) -> Path:
    """Move one blocked partial/invalid result aside without deleting evidence."""

    state = validate_campaign(campaign)
    root = state["campaign_root"]
    case = state["manifest"]["cases"][case_index]
    classification = classify_case_execution(
        campaign, case_index=case_index, qblade_root=qblade_root
    )
    reason = str(classification.get("reason", ""))
    if classification.get("status") != "blocked" or not reason.startswith(
        (
            "partial_or_invalid_actual_aero_artifacts_are_not_restartable:",
            "result_path_is_not_a_directory",
        )
    ):
        raise HVAWTResponsePilotError(
            f"Case {case['case_id']} is not an archivable partial attempt: {reason}"
        )
    result_path = root / str(case["result_dir"])
    if not result_path.exists():
        raise HVAWTResponsePilotError(
            f"Blocked case {case['case_id']} has no result path to archive."
        )
    archive_root = root / "attempt_archive" / str(case["case_id"])
    archive_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.removeprefix("attempt_"))
        for path in archive_root.glob("attempt_[0-9][0-9][0-9]")
        if path.name.removeprefix("attempt_").isdigit()
    ]
    destination = archive_root / f"attempt_{max(existing, default=0) + 1:03d}"
    os.replace(result_path, destination)
    atomic_write_json(
        archive_root / f"{destination.name}_archive_receipt.json",
        {
            "schema": "vorlap-hvawt-partial-attempt-archive-v1",
            "case_id": case["case_id"],
            "campaign_fingerprint": state["manifest"]["fingerprint"],
            "source_result_path": str(case["result_dir"]),
            "archive_path": destination.relative_to(root).as_posix(),
            "classification_reason": reason,
        },
    )
    return destination


def execute_case(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
    python_executable: str = sys.executable,
    threads: int = 1,
    checkpoint_every: int = 0,
    progress_every: int = 1000,
    resume: bool = False,
    archive_partial_and_restart: bool = False,
) -> list[str]:
    """Run one explicit pilot index; this function never expands to the full rerun."""

    bind_flight_runtime_identity(campaign, qblade_root=qblade_root)
    state = validate_campaign(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    case = manifest["cases"][case_index]
    lock_root = root / ".case_locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{case['case_id']}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HVAWTResponsePilotError(
                f"Case {case['case_id']} already has an active execution lock."
            ) from exc
        classification = classify_case_execution(
            campaign, case_index=case_index, qblade_root=qblade_root
        )
        if classification["status"] == "complete":
            return []
        if classification["status"] == "blocked" and archive_partial_and_restart:
            archive_incomplete_case_attempt(
                campaign, case_index=case_index, qblade_root=qblade_root
            )
            classification = classify_case_execution(
                campaign, case_index=case_index, qblade_root=qblade_root
            )
        if classification["status"] != "runnable":
            raise HVAWTResponsePilotError(
                f"Refusing case {case['case_id']}: {classification['reason']}"
            )
        result_dir = root / str(case["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": "vorlap-hvawt-response-pilot-case-metadata-v1",
            "campaign_fingerprint": manifest["fingerprint"],
            "case": case,
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
            resume=resume,
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
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return command


def _finite_csv_float(row: Mapping[str, str], key: str, *, row_index: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise HVAWTResponsePilotError(
            f"History row {row_index} lacks numeric field {key!r}."
        ) from exc
    if not math.isfinite(value):
        raise HVAWTResponsePilotError(
            f"History row {row_index} field {key!r} is non-finite."
        )
    return value


def _dominant_frequency_hz(time_s: np.ndarray, signal: np.ndarray) -> float | None:
    if time_s.size < 16 or signal.size != time_s.size:
        return None
    dt = np.diff(time_s)
    if np.any(dt <= 0.0):
        raise HVAWTResponsePilotError("Pilot history time must be strictly increasing.")
    median_dt = float(np.median(dt))
    if float(np.max(np.abs(dt - median_dt))) > max(1.0e-10, 1.0e-6 * median_dt):
        raise HVAWTResponsePilotError("Pilot history tail is not uniformly sampled.")
    centered = signal - float(np.mean(signal))
    if float(np.max(np.abs(centered))) <= 1.0e-12:
        return None
    windowed = centered * np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(windowed.size, d=median_dt)
    if spectrum.size <= 1:
        return None
    index = 1 + int(np.argmax(spectrum[1:]))
    return float(frequencies[index])


def analyze_history(
    history_path: Path,
    *,
    tail_start_s: float,
    tail_duration_s: float,
) -> dict[str, object]:
    """Reduce one compact SIL history to bounded modal tail metrics."""

    path = Path(history_path).expanduser().resolve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "step_index",
            "time_s",
            "qblade_timestep",
            "response_model_type",
            "generated_max_abs_force_component_n",
            "response_total_maximum_nodal_force_n",
            "response_engineering_maximum_abs_nodal_force_n",
            "response_spectral_source_config_fingerprint",
            "response_spectral_residual_source_scale",
            "response_spectral_applied_residual_maximum_nodal_force_n",
            "response_spectral_applied_residual_force_power_w",
            "response_spectral_source_energy_sum",
            "response_spectral_residual_energy_sum",
            "response_spectral_applied_residual_lift_energy_sum",
            "response_spectral_truncated_residual_lift_energy_sum",
            "response_spectral_truncated_residual_drag_energy_sum",
            "response_spectral_captured_energy_sum",
            "response_spectral_maximum_energy_closure_error",
        }
        for mode in (6, 12):
            required.update(
                {
                    f"response_mode{mode}_raw_complex_projection_real_mps",
                    f"response_mode{mode}_raw_complex_projection_imag_mps",
                    f"response_mode{mode}_filtered_qdot_mps",
                    f"response_mode{mode}_amplitude_estimate_m",
                    f"response_mode{mode}_complex_projection_leakage_ratio",
                    f"response_mode{mode}_engineering_aerodynamic_to_structural_damping_ratio",
                    f"response_mode{mode}_spectral_gain_max",
                }
            )
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise HVAWTResponsePilotError(
                f"Compact SIL history lacks fields: {missing}"
            )
        times: list[float] = []
        qdot = {6: [], 12: []}
        amplitude = {6: [], 12: []}
        raw_complex_real = {6: [], 12: []}
        raw_complex_imag = {6: [], 12: []}
        instantaneous_leakage_max = {6: 0.0, 12: 0.0}
        activation_max = {6: 0.0, 12: 0.0}
        row_count = 0
        history_times: list[float] = []
        max_generated_force = 0.0
        max_response_force = 0.0
        max_total_supplemental_force = 0.0
        model_types: set[str] = set()
        spectral_fingerprints: set[str] = set()
        residual_source_scales: set[float] = set()
        maximum_source_energy = 0.0
        maximum_applied_residual_force = 0.0
        maximum_abs_applied_residual_power = 0.0
        maximum_spectral_energy_closure_error = 0.0
        maximum_applied_spectral_energy_closure_error = 0.0
        maximum_truncated_residual_lift_energy = 0.0
        maximum_truncated_residual_drag_energy = 0.0
        maximum_reported_energy_closure_error = 0.0
        maximum_spectral_gain = {6: 0.0, 12: 0.0}
        for row_index, row in enumerate(reader):
            step_index = int(_finite_csv_float(row, "step_index", row_index=row_index))
            if step_index != row_index:
                raise HVAWTResponsePilotError(
                    "Compact SIL history step indices are not sequential."
                )
            time_value = _finite_csv_float(row, "time_s", row_index=row_index)
            history_times.append(time_value)
            qblade_timestep = _finite_csv_float(
                row, "qblade_timestep", row_index=row_index
            )
            if not math.isclose(
                qblade_timestep,
                float(row_index),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise HVAWTResponsePilotError(
                    "Compact SIL history QBlade timestep counters are not sequential."
                )
            max_generated_force = max(
                max_generated_force,
                abs(
                    _finite_csv_float(
                        row, "generated_max_abs_force_component_n", row_index=row_index
                    )
                ),
            )
            max_response_force = max(
                max_response_force,
                abs(
                    _finite_csv_float(
                        row,
                        "response_engineering_maximum_abs_nodal_force_n",
                        row_index=row_index,
                    )
                ),
            )
            max_total_supplemental_force = max(
                max_total_supplemental_force,
                abs(
                    _finite_csv_float(
                        row,
                        "response_total_maximum_nodal_force_n",
                        row_index=row_index,
                    )
                ),
            )
            model_type = row.get("response_model_type")
            if model_type != "hvawt_airfoil_engineering_lockin_hybrid":
                raise HVAWTResponsePilotError(
                    "Pilot history is not the geometry-blended airfoil hybrid."
                )
            model_types.add(model_type)
            spectral_fingerprint = row.get(
                "response_spectral_source_config_fingerprint"
            )
            if (
                not isinstance(spectral_fingerprint, str)
                or re.fullmatch(r"[0-9a-f]{64}", spectral_fingerprint) is None
            ):
                raise HVAWTResponsePilotError(
                    "Pilot history lacks a valid spectral-source fingerprint."
                )
            spectral_fingerprints.add(spectral_fingerprint)
            residual_source_scales.add(
                _finite_csv_float(
                    row,
                    "response_spectral_residual_source_scale",
                    row_index=row_index,
                )
            )
            maximum_applied_residual_force = max(
                maximum_applied_residual_force,
                abs(
                    _finite_csv_float(
                        row,
                        "response_spectral_applied_residual_maximum_nodal_force_n",
                        row_index=row_index,
                    )
                ),
            )
            maximum_abs_applied_residual_power = max(
                maximum_abs_applied_residual_power,
                abs(
                    _finite_csv_float(
                        row,
                        "response_spectral_applied_residual_force_power_w",
                        row_index=row_index,
                    )
                ),
            )
            source_energy = _finite_csv_float(
                row,
                "response_spectral_source_energy_sum",
                row_index=row_index,
            )
            residual_energy = _finite_csv_float(
                row,
                "response_spectral_residual_energy_sum",
                row_index=row_index,
            )
            captured_energy = _finite_csv_float(
                row,
                "response_spectral_captured_energy_sum",
                row_index=row_index,
            )
            applied_residual_lift_energy = _finite_csv_float(
                row,
                "response_spectral_applied_residual_lift_energy_sum",
                row_index=row_index,
            )
            truncated_residual_lift_energy = _finite_csv_float(
                row,
                "response_spectral_truncated_residual_lift_energy_sum",
                row_index=row_index,
            )
            truncated_residual_drag_energy = _finite_csv_float(
                row,
                "response_spectral_truncated_residual_drag_energy_sum",
                row_index=row_index,
            )
            if (
                min(
                    source_energy,
                    residual_energy,
                    captured_energy,
                    applied_residual_lift_energy,
                    truncated_residual_lift_energy,
                    truncated_residual_drag_energy,
                )
                < 0.0
            ):
                raise HVAWTResponsePilotError(
                    "Pilot spectral energies must be non-negative."
                )
            maximum_source_energy = max(maximum_source_energy, source_energy)
            maximum_spectral_energy_closure_error = max(
                maximum_spectral_energy_closure_error,
                abs(source_energy - residual_energy - captured_energy),
            )
            maximum_applied_spectral_energy_closure_error = max(
                maximum_applied_spectral_energy_closure_error,
                abs(
                    source_energy
                    - captured_energy
                    - applied_residual_lift_energy
                    - truncated_residual_lift_energy
                ),
            )
            maximum_truncated_residual_lift_energy = max(
                maximum_truncated_residual_lift_energy,
                truncated_residual_lift_energy,
            )
            maximum_truncated_residual_drag_energy = max(
                maximum_truncated_residual_drag_energy,
                truncated_residual_drag_energy,
            )
            reported_closure = _finite_csv_float(
                row,
                "response_spectral_maximum_energy_closure_error",
                row_index=row_index,
            )
            if reported_closure < 0.0:
                raise HVAWTResponsePilotError(
                    "Reported spectral-energy closure error must be non-negative."
                )
            maximum_reported_energy_closure_error = max(
                maximum_reported_energy_closure_error,
                reported_closure,
            )
            row_count += 1
            if time_value + 1.0e-12 < tail_start_s:
                continue
            if time_value > tail_start_s + tail_duration_s + 1.0e-8:
                raise HVAWTResponsePilotError(
                    "Compact SIL history extends beyond the declared pilot duration."
                )
            times.append(time_value)
            for mode in (6, 12):
                raw_complex_real[mode].append(
                    _finite_csv_float(
                        row,
                        f"response_mode{mode}_raw_complex_projection_real_mps",
                        row_index=row_index,
                    )
                )
                raw_complex_imag[mode].append(
                    _finite_csv_float(
                        row,
                        f"response_mode{mode}_raw_complex_projection_imag_mps",
                        row_index=row_index,
                    )
                )
                instantaneous_leakage_max[mode] = max(
                    instantaneous_leakage_max[mode],
                    _finite_csv_float(
                        row,
                        f"response_mode{mode}_complex_projection_leakage_ratio",
                        row_index=row_index,
                    ),
                )
                qdot[mode].append(
                    _finite_csv_float(
                        row,
                        f"response_mode{mode}_filtered_qdot_mps",
                        row_index=row_index,
                    )
                )
                amplitude[mode].append(
                    _finite_csv_float(
                        row,
                        f"response_mode{mode}_amplitude_estimate_m",
                        row_index=row_index,
                    )
                )
                activation_max[mode] = max(
                    activation_max[mode],
                    _finite_csv_float(
                        row,
                        f"response_mode{mode}_engineering_aerodynamic_to_structural_damping_ratio",
                        row_index=row_index,
                    ),
                )
                maximum_spectral_gain[mode] = max(
                    maximum_spectral_gain[mode],
                    _finite_csv_float(
                        row,
                        f"response_mode{mode}_spectral_gain_max",
                        row_index=row_index,
                    ),
                )
    if not times:
        raise HVAWTResponsePilotError(
            "Compact SIL history has no samples in the assessment tail."
        )
    history_time_array = np.asarray(history_times, dtype=float)
    if history_time_array.size < 2:
        raise HVAWTResponsePilotError(
            "Compact SIL history cannot establish its physical sample interval."
        )
    reported_timestep_s = float(
        (history_time_array[-1] - history_time_array[0])
        / (history_time_array.size - 1)
    )
    if reported_timestep_s <= 0.0:
        raise HVAWTResponsePilotError(
            "Compact SIL history time must be strictly increasing."
        )
    uniform_time_tolerance = max(1.0e-10, 1.0e-6 * reported_timestep_s)
    expected_history_times = history_time_array[0] + reported_timestep_s * np.arange(
        history_time_array.size,
        dtype=float,
    )
    if (
        abs(float(history_time_array[0])) > uniform_time_tolerance
        or float(np.max(np.abs(history_time_array - expected_history_times)))
        > uniform_time_tolerance
    ):
        raise HVAWTResponsePilotError(
            "Compact SIL history time grid is not uniformly sampled."
        )
    if len(model_types) != 1 or len(spectral_fingerprints) != 1:
        raise HVAWTResponsePilotError(
            "Compact SIL history changes hybrid model or spectral identity."
        )
    if len(residual_source_scales) != 1:
        raise HVAWTResponsePilotError(
            "Compact SIL history changes residual-source scale."
        )
    time_array = np.asarray(times, dtype=float)
    split = tail_start_s + STATIONARITY_TAIL_SPLIT_FRACTION * tail_duration_s
    mode_metrics: dict[str, object] = {}
    for mode in (6, 12):
        amplitude_array = np.asarray(amplitude[mode], dtype=float)
        qdot_array = np.asarray(qdot[mode], dtype=float)
        raw_complex_real_array = np.asarray(raw_complex_real[mode], dtype=float)
        raw_complex_imag_array = np.asarray(raw_complex_imag[mode], dtype=float)
        first = amplitude_array[time_array < split]
        second = amplitude_array[time_array >= split]
        if first.size == 0 or second.size == 0:
            raise HVAWTResponsePilotError(
                "Assessment tail does not cover both stationarity halves."
            )
        first_peak = float(np.max(first))
        second_peak = float(np.max(second))
        peak = float(np.max(amplitude_array))
        real_rms = float(np.sqrt(np.mean(raw_complex_real_array**2)))
        imaginary_rms = float(np.sqrt(np.mean(raw_complex_imag_array**2)))
        tail_l2_leakage = imaginary_rms / max(real_rms, 1.0e-12)
        mode_metrics[str(mode)] = {
            "peak_amplitude_estimate_m": peak,
            "first_half_peak_amplitude_m": first_peak,
            "second_half_peak_amplitude_m": second_peak,
            "tail_half_peak_relative_difference": abs(second_peak - first_peak)
            / max(peak, 1.0e-15),
            "dominant_frequency_hz": _dominant_frequency_hz(time_array, qdot_array),
            "maximum_complex_projection_leakage_ratio": tail_l2_leakage,
            "complex_projection_tail_l2_leakage_ratio": tail_l2_leakage,
            "raw_complex_projection_real_rms_mps": real_rms,
            "raw_complex_projection_imag_rms_mps": imaginary_rms,
            "maximum_instantaneous_complex_projection_leakage_ratio": (
                instantaneous_leakage_max[mode]
            ),
            "maximum_activation": activation_max[mode],
            "maximum_spectral_gain": maximum_spectral_gain[mode],
        }
    energy_tolerance = 2.0e-12 * max(1.0, maximum_source_energy)
    return {
        "rows": row_count,
        "tail_rows": len(times),
        "reported_timestep_s": reported_timestep_s,
        "tail_time_min_s": float(time_array[0]),
        "tail_time_max_s": float(time_array[-1]),
        "maximum_generated_force_component_n": max_generated_force,
        "maximum_response_nodal_force_n": max_response_force,
        "maximum_total_supplemental_nodal_force_n": max_total_supplemental_force,
        "response_model_type": next(iter(model_types)),
        "spectral_source_config_fingerprint": next(iter(spectral_fingerprints)),
        "spectral_residual_source_scale": next(iter(residual_source_scales)),
        "maximum_spectral_source_energy": maximum_source_energy,
        "maximum_applied_residual_force_n": maximum_applied_residual_force,
        "maximum_abs_applied_residual_power_w": maximum_abs_applied_residual_power,
        "maximum_spectral_energy_closure_error": max(
            maximum_spectral_energy_closure_error,
            maximum_applied_spectral_energy_closure_error,
            maximum_reported_energy_closure_error,
        ),
        "maximum_applied_spectral_energy_closure_error": (
            maximum_applied_spectral_energy_closure_error
        ),
        "maximum_truncated_residual_lift_energy": (
            maximum_truncated_residual_lift_energy
        ),
        "maximum_truncated_residual_drag_energy": (
            maximum_truncated_residual_drag_energy
        ),
        "spectral_energy_closure_tolerance": energy_tolerance,
        "modes": mode_metrics,
    }


def _history_path_from_summary(summary: Mapping[str, Any], result_dir: Path) -> Path:
    history = summary.get("sil_vorlap_history")
    if not isinstance(history, Mapping) or history.get("mode") != "compact":
        raise HVAWTResponsePilotError(
            "Completed pilot case lacks compact SIL history metadata."
        )
    raw = history.get("path")
    if not isinstance(raw, str) or not raw:
        raise HVAWTResponsePilotError("Completed pilot case lacks a SIL history path.")
    candidate = Path(raw)
    if not candidate.is_file():
        candidate = result_dir / Path(raw).name
    identity = history.get("file_identity")
    if not isinstance(identity, Mapping):
        raise HVAWTResponsePilotError(
            "Completed pilot history lacks byte identity metadata."
        )
    observed = _identity(candidate)
    if observed["sha256"] != identity.get("sha256") or observed[
        "size_bytes"
    ] != identity.get("size_bytes"):
        raise HVAWTResponsePilotError(
            "Completed pilot SIL history byte identity mismatch."
        )
    return candidate


def _case_result(
    *,
    root: Path,
    case: Mapping[str, Any],
    model: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    campaign_kind: str,
    peak_winds: Mapping[str, Any],
    amplitude_targets: Mapping[str, Any],
    campaign_fingerprint: str,
    expected_runtime_identity: Mapping[str, Any],
) -> dict[str, object]:
    result_dir = root / str(case["result_dir"])
    sim = root / str(case["sim"])
    summary_path = result_dir / f"{sim.stem}_run_summary.json"
    if not summary_path.is_file():
        return {
            "case_id": case["case_id"],
            "status": "missing",
            "passed": False,
            "gates": [],
        }
    summary = _load_json(summary_path, label=f"run summary for {case['case_id']}")
    required_summary = {
        "coupling_mode": "sil-vorlap",
        "qblade_execution_mode": "step-loop",
        "requested_steps": int(case["total_steps"]),
        "completed_steps": int(case["total_steps"]),
        "failed": False,
        "stopped": False,
        "allow_partial": False,
        "exported_results": True,
        "sil_vorlap": True,
        "sil_vorlap_updates": int(case["total_steps"]),
        "sil_vorlap_history_mode": "compact",
    }
    gates: list[dict[str, object]] = []
    for name, expected in required_summary.items():
        observed = summary.get(name)
        gates.append(
            {
                "gate": f"summary_{name}",
                "passed": observed == expected,
                "observed": observed,
                "limit_or_expected": expected,
            }
        )
    result_path = result_dir / f"{case['result_name']}.txt"
    reported_result_identity = summary.get("result_file_identity")
    observed_result_identity = _identity(result_path) if result_path.is_file() else None
    result_identity_pass = (
        isinstance(reported_result_identity, Mapping)
        and isinstance(observed_result_identity, Mapping)
        and reported_result_identity.get("sha256")
        == observed_result_identity.get("sha256")
        and reported_result_identity.get("size_bytes")
        == observed_result_identity.get("size_bytes")
    )
    expected_qblade_sha256 = str(expected_runtime_identity["qblade_library"]["sha256"])
    reproducibility = summary.get("reproducibility")
    gates.append(
        {
            "gate": "result_file_receipt_identity",
            "passed": result_identity_pass,
            "observed": observed_result_identity,
            "limit_or_expected": reported_result_identity,
        }
    )
    result_filter = root / str(case["result_filter"])
    result_export_audit: dict[str, object] | None = None
    result_export_problem: str | None = None
    try:
        result_export_audit = _audit_qblade_fatigue_result(
            result_path,
            result_filter=result_filter,
            timestep_s=float(case["timestep_s"]),
            total_steps=int(case["total_steps"]),
            store_from_s=float(case["duration_s"])
            - float(case["assessment_tail_duration_s"]),
        )
    except (HVAWTResponsePilotError, FileNotFoundError, OSError, ValueError) as exc:
        result_export_problem = str(exc)
    result_export_pass = (
        result_export_audit is not None
        and float(result_export_audit["maximum_abs_structural_load_channel_value"])
        > 0.0
    )
    gates.append(
        {
            "gate": "matched_fatigue_result_history",
            "passed": result_export_pass,
            "observed": (
                result_export_audit
                if result_export_problem is None
                else {"problem": result_export_problem}
            ),
            "limit_or_expected": {
                "channel_count": RESULT_FILTER_CHANNEL_COUNT,
                "retained_tail_duration_s": case["assessment_tail_duration_s"],
                "exact_time_grid": True,
            },
        }
    )
    observed_bridge = (
        reproducibility.get("bridge_library")
        if isinstance(reproducibility, Mapping)
        else None
    )
    expected_bridge = expected_runtime_identity.get("bridge_library")
    bridge_identity_pass = (
        isinstance(observed_bridge, Mapping)
        and isinstance(expected_bridge, Mapping)
        and observed_bridge.get("sha256") == expected_bridge.get("sha256")
        and observed_bridge.get("size_bytes") == expected_bridge.get("size_bytes")
    )
    gates.append(
        {
            "gate": "bridge_library_identity",
            "passed": bridge_identity_pass,
            "observed": observed_bridge,
            "limit_or_expected": expected_bridge,
        }
    )
    observed_python_fingerprint = (
        reproducibility.get("vorlap_python_runtime_fingerprint")
        if isinstance(reproducibility, Mapping)
        else None
    )
    expected_python_fingerprint = expected_runtime_identity.get(
        "vorlap_python_runtime_fingerprint"
    )
    gates.append(
        {
            "gate": "vorlap_python_runtime_fingerprint",
            "passed": observed_python_fingerprint == expected_python_fingerprint,
            "observed": observed_python_fingerprint,
            "limit_or_expected": expected_python_fingerprint,
        }
    )
    metadata_path = result_dir / "case_metadata.json"
    expected_metadata = {
        "schema": "vorlap-hvawt-response-pilot-case-metadata-v1",
        "campaign_fingerprint": campaign_fingerprint,
        "case": case,
    }
    observed_metadata = (
        _load_json(metadata_path, label=f"case metadata for {case['case_id']}")
        if metadata_path.is_file()
        else None
    )
    embedded_metadata = (
        reproducibility.get("case_metadata")
        if isinstance(reproducibility, Mapping)
        else None
    )
    metadata_pass = (
        observed_metadata == expected_metadata
        and embedded_metadata == expected_metadata
    )
    gates.append(
        {
            "gate": "case_metadata_campaign_identity",
            "passed": metadata_pass,
            "observed": {
                "sidecar": observed_metadata,
                "embedded": embedded_metadata,
            },
            "limit_or_expected": expected_metadata,
        }
    )
    qblade = (
        reproducibility.get("qblade") if isinstance(reproducibility, Mapping) else None
    )
    qblade_library = qblade.get("library") if isinstance(qblade, Mapping) else None
    observed_qblade_sha256 = (
        qblade_library.get("sha256") if isinstance(qblade_library, Mapping) else None
    )
    gates.append(
        {
            "gate": "qblade_library_identity",
            "passed": observed_qblade_sha256 == expected_qblade_sha256,
            "observed": observed_qblade_sha256,
            "limit_or_expected": expected_qblade_sha256,
        }
    )
    observed_runtime_fingerprint = (
        reproducibility.get("runtime_asset_fingerprint")
        if isinstance(reproducibility, Mapping)
        else None
    )
    gates.append(
        {
            "gate": "runtime_asset_fingerprint",
            "passed": observed_runtime_fingerprint
            == case.get("runtime_asset_fingerprint"),
            "observed": observed_runtime_fingerprint,
            "limit_or_expected": case.get("runtime_asset_fingerprint"),
        }
    )
    restart = summary.get("restart")
    uninterrupted = (
        isinstance(restart, Mapping)
        and restart.get("resume_requested") is False
        and restart.get("resume_checkpoint") is None
        and int(restart.get("resume_start_step", -1)) == 0
        and int(restart.get("checkpoint_every", -1))
        in {0, FLIGHT_RESTART_SAVE_INTERVAL_STEPS}
    )
    gates.append(
        {
            "gate": "uninterrupted_actual_aero_execution",
            "passed": uninterrupted,
            "observed": restart,
            "limit_or_expected": (
                "resume_requested=false, no resume checkpoint, start step 0, "
                "checkpoint interval 0 or 24000 recovery-save steps"
            ),
        }
    )
    native = summary.get("native_callback_evidence")
    zero_native_callbacks = False
    if isinstance(native, dict):
        try:
            require_zero_native_callback_evidence(
                native, context=f"H-VAWT pilot {case['case_id']}"
            )
        except (RuntimeError, TypeError, ValueError):
            pass
        else:
            zero_native_callbacks = True
    gates.append(
        {
            "gate": "no_native_external_callback_double_count",
            "passed": zero_native_callbacks,
            "observed": native,
            "limit_or_expected": (
                "zero init, log, cache-observed, and maximum callbacks in SIL mode"
            ),
        }
    )
    expected_source_frequency_cap_hz = 0.5 / float(case["timestep_s"])
    observed_source_frequency_cap_hz = summary.get(
        "sil_vorlap_source_frequency_cap_hz"
    )
    source_frequency_cap_pass = (
        isinstance(observed_source_frequency_cap_hz, (int, float))
        and not isinstance(observed_source_frequency_cap_hz, bool)
        and math.isclose(
            float(observed_source_frequency_cap_hz),
            expected_source_frequency_cap_hz,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    )
    gates.append(
        {
            "gate": "exact_nyquist_source_frequency_cap",
            "passed": source_frequency_cap_pass,
            "observed": observed_source_frequency_cap_hz,
            "limit_or_expected": expected_source_frequency_cap_hz,
        }
    )
    coverage = summary.get("sil_vorlap_source_coverage")
    coverage_problem = None
    try:
        validate_complete_source_coverage(
            coverage,
            expected_samples=int(case["total_steps"]),
            component_node_counts=HVAWT_V8_COMPONENT_NODE_COUNTS,
        )
    except (TypeError, ValueError) as exc:
        coverage_problem = str(exc)
    gates.append(
        {
            "gate": "complete_source_coverage",
            "passed": coverage_problem is None,
            "observed": coverage if coverage_problem is None else coverage_problem,
            "limit_or_expected": (
                "one exact schema row per H-VAWT component, every step, zero "
                "hard bounds/alias/nonfinite counts"
            ),
        }
    )
    velocity = summary.get("sil_vorlap_velocity_diagnostics")
    velocity_problem = None
    try:
        validate_complete_velocity_diagnostics(
            velocity,
            expected_samples=int(case["total_steps"]),
            node_count=sum(HVAWT_V8_COMPONENT_NODE_COUNTS.values()),
        )
    except (TypeError, ValueError) as exc:
        velocity_problem = str(exc)
    gates.append(
        {
            "gate": "complete_structural_velocity_channels",
            "passed": velocity_problem is None,
            "observed": velocity if velocity_problem is None else velocity_problem,
            "limit_or_expected": "117 live channels; X/Y/Z sampled every node-step",
        }
    )
    rpm = summary.get("qblade_export_rpm_diagnostics")
    rpm_error = math.inf
    if isinstance(rpm, Mapping) and rpm.get("available") is True:
        rpm_error = float(rpm.get("max_abs_error_rpm", math.inf))
    gates.append(
        {
            "gate": "prescribed_1rpm",
            "passed": rpm_error <= float(acceptance["required_rpm_tolerance"]),
            "observed": rpm_error,
            "limit_or_expected": float(acceptance["required_rpm_tolerance"]),
        }
    )
    history_path = _history_path_from_summary(summary, result_dir)
    metrics = analyze_history(
        history_path,
        tail_start_s=float(case["duration_s"])
        - float(case["assessment_tail_duration_s"]),
        tail_duration_s=float(case["assessment_tail_duration_s"]),
    )
    gates.append(
        {
            "gate": "history_rows_match_completed_updates",
            "passed": metrics["rows"] == int(case["total_steps"]),
            "observed": metrics["rows"],
            "limit_or_expected": int(case["total_steps"]),
        }
    )
    gates.append(
        {
            "gate": "history_timestep",
            "passed": math.isclose(
                float(metrics["reported_timestep_s"]),
                float(case["timestep_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "observed": metrics["reported_timestep_s"],
            "limit_or_expected": case["timestep_s"],
        }
    )
    expected_residual_scale = (
        0.0 if case["scenario"] == SCENARIOS[0] else 1.0
    )
    gates.extend(
        [
            {
                "gate": "airfoil_spectral_source_energy_present",
                "passed": float(metrics["maximum_spectral_source_energy"]) > 0.0,
                "observed": metrics["maximum_spectral_source_energy"],
                "limit_or_expected": "> 0",
            },
            {
                "gate": "spectral_energy_partition_closure",
                "passed": float(metrics["maximum_spectral_energy_closure_error"])
                <= float(metrics["spectral_energy_closure_tolerance"]),
                "observed": metrics["maximum_spectral_energy_closure_error"],
                "limit_or_expected": metrics["spectral_energy_closure_tolerance"],
            },
            {
                "gate": "scenario_residual_source_scale",
                "passed": math.isclose(
                    float(metrics["spectral_residual_source_scale"]),
                    expected_residual_scale,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ),
                "observed": metrics["spectral_residual_source_scale"],
                "limit_or_expected": expected_residual_scale,
            },
            {
                "gate": "scenario_applied_residual_force",
                "passed": (
                    float(metrics["maximum_applied_residual_force_n"]) <= 1.0e-12
                    if case["scenario"] == SCENARIOS[0]
                    else float(metrics["maximum_applied_residual_force_n"]) > 0.0
                ),
                "observed": metrics["maximum_applied_residual_force_n"],
                "limit_or_expected": (
                    "<= 1e-12 N"
                    if case["scenario"] == SCENARIOS[0]
                    else "> 0 N"
                ),
            },
        ]
    )
    maximum_force = float(metrics["maximum_total_supplemental_nodal_force_n"])
    force_limit_raw = model["max_abs_total_nodal_force_n"]
    if force_limit_raw is None:
        gates.append(
            {
                "gate": "total_supplemental_force_finite_and_unclipped",
                "passed": math.isfinite(maximum_force),
                "observed": maximum_force,
                "limit_or_expected": (
                    "finite recorded norm; no arbitrary ceiling or clipping"
                ),
            }
        )
    else:
        force_limit = float(force_limit_raw)
        gates.append(
            {
                "gate": "total_supplemental_force_limit_not_reached",
                "passed": maximum_force < force_limit,
                "observed": maximum_force,
                "limit_or_expected": force_limit,
            }
        )
    if case["scenario"] == SCENARIOS[0]:
        observed_zero_force = max(
            float(metrics["maximum_generated_force_component_n"]),
            abs(float(summary.get("sil_vorlap_max_abs_force", math.inf))),
        )
        gates.append(
            {
                "gate": "native_control_external_force_exact_zero",
                "passed": observed_zero_force <= 1.0e-12,
                "observed": observed_zero_force,
                "limit_or_expected": 1.0e-12,
            }
        )
    else:
        observed_hybrid_force = float(
            metrics["maximum_generated_force_component_n"]
        )
        gates.append(
            {
                "gate": "hybrid_external_force_nonzero",
                "passed": observed_hybrid_force > 0.0,
                "observed": observed_hybrid_force,
                "limit_or_expected": "> 0",
            }
        )
        gates.append(
            {
                "gate": "hybrid_motion_correlated_force_observed",
                "passed": True,
                "diagnostic_only": True,
                "observed": metrics["maximum_response_nodal_force_n"],
                "limit_or_expected": (
                    "nonzero at prepass-active conditions; zero is admissible off band"
                ),
            }
        )
    mode_config = {str(row["tracked_mode"]): row for row in model["modes"]}
    for mode, mode_metrics in metrics["modes"].items():
        leakage = float(mode_metrics["maximum_complex_projection_leakage_ratio"])
        gates.append(
            {
                "gate": f"mode{mode}_complex_projection_leakage",
                "passed": True,
                "diagnostic_only": True,
                "observed": leakage,
                "maximum_instantaneous_ratio": mode_metrics[
                    "maximum_instantaneous_complex_projection_leakage_ratio"
                ],
                "limit_or_expected": (
                    "tail L2 leakage is gated only for active response branches"
                ),
            }
        )
        is_peak_case = case["scenario"] == SCENARIOS[1] and math.isclose(
            float(case["wind_speed_mps"]),
            float(peak_winds[mode]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        if not is_peak_case:
            continue
        spectral_gain = float(mode_metrics["maximum_spectral_gain"])
        gates.append(
            {
                "gate": f"mode{mode}_captured_airfoil_energy_present",
                "passed": spectral_gain > 0.0,
                "observed": spectral_gain,
                "limit_or_expected": "> 0",
            }
        )
        config = mode_config[mode]
        reference_length = float(config["reference_length_m"])
        target = float(amplitude_targets[mode])
        allowance = _amplitude_target_allowance_m(
            target_m=target,
            reference_length_m=reference_length,
            acceptance=acceptance,
        )
        observed_amplitude = float(mode_metrics["peak_amplitude_estimate_m"])
        amplitude_error = abs(observed_amplitude - target)
        gates.append(
            {
                "gate": f"mode{mode}_closure_only_peak_amplitude_reference",
                "passed": True,
                "diagnostic_only": True,
                "observed": observed_amplitude,
                "target": target,
                "absolute_error_m": amplitude_error,
                "reference_allowance_m": allowance,
                "limit_or_expected": "diagnostic only for complete composition",
            }
        )
        stationary = float(mode_metrics["tail_half_peak_relative_difference"])
        stationary_pass = stationary <= float(
            acceptance["maximum_tail_half_peak_amplitude_relative_difference"]
        )
        preflight_diagnostic = campaign_kind == PREFLIGHT_CAMPAIGN_KIND
        gates.append(
            {
                "gate": f"mode{mode}_stationary_tail",
                "passed": stationary_pass or preflight_diagnostic,
                "diagnostic_only": preflight_diagnostic,
                "criterion_passed": stationary_pass,
                "observed": stationary,
                "limit_or_expected": acceptance[
                    "maximum_tail_half_peak_amplitude_relative_difference"
                ],
            }
        )
        frequency = mode_metrics["dominant_frequency_hz"]
        natural = float(DOCUMENTED_MODE_DAMPING[int(mode)]["frequency_hz"])
        frequency_error = (
            math.inf if frequency is None else abs(float(frequency) - natural) / natural
        )
        gates.append(
            {
                "gate": f"mode{mode}_peak_frequency",
                "passed": frequency_error
                <= float(acceptance["maximum_modal_peak_frequency_relative_error"]),
                "observed": frequency,
                "target": natural,
                "relative_error": frequency_error,
                "limit_or_expected": acceptance[
                    "maximum_modal_peak_frequency_relative_error"
                ],
            }
        )
    execution_receipt_gate_names = {
        *(f"summary_{name}" for name in required_summary),
        "result_file_receipt_identity",
        "matched_fatigue_result_history",
        "case_metadata_campaign_identity",
        "qblade_library_identity",
        "bridge_library_identity",
        "vorlap_python_runtime_fingerprint",
        "runtime_asset_fingerprint",
        "uninterrupted_actual_aero_execution",
        "no_native_external_callback_double_count",
        "exact_nyquist_source_frequency_cap",
        "complete_source_coverage",
        "complete_structural_velocity_channels",
        "history_rows_match_completed_updates",
        "history_timestep",
    }
    execution_receipt_pass = all(
        bool(gate["passed"])
        for gate in gates
        if gate["gate"] in execution_receipt_gate_names
    ) and execution_receipt_gate_names.issubset({str(gate["gate"]) for gate in gates})
    return {
        "case_id": case["case_id"],
        "status": "complete",
        "passed": all(bool(row["passed"]) for row in gates),
        "execution_receipt_pass": execution_receipt_pass,
        "scenario": case["scenario"],
        "wind_speed_mps": case["wind_speed_mps"],
        "timestep_s": case["timestep_s"],
        "summary": _identity(summary_path),
        "history": _identity(history_path),
        "metrics": metrics,
        "fatigue_result_export": result_export_audit,
        "gates": gates,
    }


def _case_result_or_error(**kwargs: Any) -> dict[str, object]:
    case = kwargs["case"]
    try:
        return _case_result(**kwargs)
    except (HVAWTResponsePilotError, FileNotFoundError, OSError, ValueError) as exc:
        return {
            "case_id": case["case_id"],
            "status": "invalid_result",
            "passed": False,
            "scenario": case["scenario"],
            "wind_speed_mps": case["wind_speed_mps"],
            "timestep_s": case["timestep_s"],
            "problem": str(exc),
            "gates": [],
        }


def _relative_pair_difference(a: float, b: float, absolute_floor: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), absolute_floor)


def _amplitude_target_allowance_m(
    *,
    target_m: float,
    reference_length_m: float,
    acceptance: Mapping[str, Any],
) -> float:
    return (
        float(acceptance["maximum_peak_amplitude_error_fraction_of_target"]) * target_m
        + float(
            acceptance["peak_amplitude_absolute_allowance_fraction_of_reference_length"]
        )
        * reference_length_m
    )


def analyze_campaign(campaign: Path, *, output_path: Path | None = None) -> Path:
    """Analyze a sealed preflight or full response campaign."""

    repository = Path(__file__).resolve().parents[1]
    state = validate_campaign(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    inputs = manifest["inputs"]
    spec = _load_json(
        root / str(inputs["campaign_spec"]["path"]), label="campaign spec"
    )
    model = _load_json(
        root / str(inputs["response_model"]["path"]), label="response model"
    )
    acceptance = spec["acceptance"]
    matrix = _campaign_matrix(spec)
    campaign_kind = str(matrix["kind"])
    winds = tuple(float(value) for value in matrix["winds"])
    timesteps = tuple(float(value) for value in matrix["timesteps"])
    peak_winds = spec["pilot"]["lock_in_peak_wind_speed_by_mode_mps"]
    amplitude_targets = spec["pilot"]["predicted_peak_amplitude_target_by_mode_m"]
    selected_amplitude_targets = _selected_wind_amplitude_targets(spec["pilot"])
    expected_runtime_identity = _load_flight_runtime_identity(root)
    case_results = [
        _case_result_or_error(
            root=root,
            case=case,
            model=model,
            acceptance=acceptance,
            campaign_kind=campaign_kind,
            peak_winds=peak_winds,
            amplitude_targets=amplitude_targets,
            campaign_fingerprint=str(manifest["fingerprint"]),
            expected_runtime_identity=expected_runtime_identity,
        )
        for case in manifest["cases"]
    ]
    complete = [row for row in case_results if row["status"] == "complete"]
    cross_gates: list[dict[str, object]] = []
    if len(complete) == len(case_results):
        lookup = {
            (
                row["scenario"],
                float(row["wind_speed_mps"]),
                float(row["timestep_s"]),
            ): row
            for row in complete
        }
        reference_length = {
            str(row["tracked_mode"]): float(row["reference_length_m"])
            for row in model["modes"]
        }
        if campaign_kind == TWO_SPEED_TEST_CAMPAIGN_KIND:
            for wind in winds:
                native = lookup[(SCENARIOS[0], wind, timesteps[0])]
                response = lookup[(SCENARIOS[1], wind, timesteps[0])]
                native_force = float(
                    native["metrics"]["maximum_response_nodal_force_n"]
                )
                response_force = float(
                    response["metrics"]["maximum_response_nodal_force_n"]
                )
                cross_gates.extend(
                    [
                        {
                            "gate": f"matched_h0_external_response_force_u{wind:g}",
                            "passed": native_force <= 1.0e-12,
                            "observed": native_force,
                            "limit_or_expected": "<= 1e-12 N",
                        },
                        {
                            "gate": f"matched_complete_response_force_u{wind:g}",
                            "passed": response_force > 0.0,
                            "observed": response_force,
                            "limit_or_expected": "> 0 N at the selected lock-in peak",
                        },
                    ]
                )
        for mode in ("6", "12"):
            for timestep in timesteps:
                if campaign_kind == TWO_SPEED_TEST_CAMPAIGN_KIND:
                    continue
                response_rows = [
                    lookup[(SCENARIOS[1], wind, timestep)] for wind in winds
                ]
                observed_by_wind = {
                    float(row["wind_speed_mps"]): float(
                        row["metrics"]["modes"][mode]["peak_amplitude_estimate_m"]
                    )
                    for row in response_rows
                }
                for wind in winds:
                    response_row = lookup[(SCENARIOS[1], wind, timestep)]
                    target = selected_amplitude_targets[(wind, mode)]
                    observed = observed_by_wind[wind]
                    if target > 0.0:
                        allowance = _amplitude_target_allowance_m(
                            target_m=target,
                            reference_length_m=reference_length[mode],
                            acceptance=acceptance,
                        )
                        error = abs(observed - target)
                        passed_shape = True
                        observed_increment = None
                        response_metrics = response_row["metrics"]["modes"][mode]
                        projection_leakage = float(
                            response_metrics[
                                "complex_projection_tail_l2_leakage_ratio"
                            ]
                        )
                        cross_gates.append(
                            {
                                "gate": (
                                    f"mode{mode}_active_complex_projection_tail_l2_"
                                    f"u{wind:g}_dt{timestep:g}"
                                ),
                                "passed": projection_leakage
                                <= float(
                                    acceptance[
                                        "maximum_complex_modal_projection_leakage_ratio"
                                    ]
                                ),
                                "observed": projection_leakage,
                                "limit_or_expected": acceptance[
                                    "maximum_complex_modal_projection_leakage_ratio"
                                ],
                            }
                        )
                        stationary = float(
                            response_metrics["tail_half_peak_relative_difference"]
                        )
                        stationary_criterion_passed = stationary <= float(
                            acceptance[
                                "maximum_tail_half_peak_amplitude_relative_difference"
                            ]
                        )
                        preflight_diagnostic = (
                            campaign_kind == PREFLIGHT_CAMPAIGN_KIND
                        )
                        cross_gates.append(
                            {
                                "gate": (
                                    f"mode{mode}_active_stationary_tail_"
                                    f"u{wind:g}_dt{timestep:g}"
                                ),
                                "passed": (
                                    stationary_criterion_passed
                                    or preflight_diagnostic
                                ),
                                "diagnostic_only": preflight_diagnostic,
                                "criterion_passed": stationary_criterion_passed,
                                "observed": stationary,
                                "limit_or_expected": acceptance[
                                    "maximum_tail_half_peak_amplitude_relative_difference"
                                ],
                            }
                        )
                        activation = float(response_metrics["maximum_activation"])
                        cross_gates.append(
                            {
                                "gate": (
                                    f"mode{mode}_active_negative_damping_"
                                    f"u{wind:g}_dt{timestep:g}"
                                ),
                                "passed": activation > 1.0,
                                "observed": activation,
                                "limit_or_expected": "> 1.0 aerodynamic/structural damping ratio",
                            }
                        )
                        response_force = float(
                            response_row["metrics"]["maximum_response_nodal_force_n"]
                        )
                        cross_gates.append(
                            {
                                "gate": (
                                    f"mode{mode}_active_motion_correlated_force_"
                                    f"u{wind:g}_dt{timestep:g}"
                                ),
                                "passed": response_force > 0.0,
                                "observed": response_force,
                                "limit_or_expected": "> 0 N",
                            }
                        )
                        frequency = response_metrics["dominant_frequency_hz"]
                        natural = float(
                            DOCUMENTED_MODE_DAMPING[int(mode)]["frequency_hz"]
                        )
                        frequency_error = (
                            math.inf
                            if frequency is None
                            else abs(float(frequency) - natural) / natural
                        )
                        cross_gates.append(
                            {
                                "gate": (
                                    f"mode{mode}_active_frequency_"
                                    f"u{wind:g}_dt{timestep:g}"
                                ),
                                "passed": frequency_error
                                <= float(
                                    acceptance[
                                        "maximum_modal_peak_frequency_relative_error"
                                    ]
                                ),
                                "observed": frequency,
                                "target": natural,
                                "relative_error": frequency_error,
                                "limit_or_expected": acceptance[
                                    "maximum_modal_peak_frequency_relative_error"
                                ],
                            }
                        )
                    else:
                        native = lookup[(SCENARIOS[0], wind, timestep)]
                        native_amplitude = float(
                            native["metrics"]["modes"][mode][
                                "peak_amplitude_estimate_m"
                            ]
                        )
                        observed_increment = max(0.0, observed - native_amplitude)
                        allowance = float(
                            acceptance[
                                "inactive_closure_only_amplitude_reference_m"
                            ]
                        )
                        error = observed_increment
                        passed_shape = True
                        response_metrics = response_row["metrics"]["modes"][mode]
                        first_half = float(
                            response_metrics["first_half_peak_amplitude_m"]
                        )
                        second_half = float(
                            response_metrics["second_half_peak_amplitude_m"]
                        )
                        inactive_stationarity_error = abs(second_half - first_half)
                        inactive_stationarity_limit = max(
                            float(
                                acceptance[
                                    "maximum_tail_half_peak_amplitude_relative_difference"
                                ]
                            )
                            * max(observed, 0.0),
                            float(
                                acceptance[
                                    "maximum_inactive_tail_half_peak_amplitude_absolute_difference_m"
                                ]
                            ),
                        )
                        inactive_stationarity_criterion_passed = (
                            inactive_stationarity_error
                            <= inactive_stationarity_limit
                        )
                        preflight_diagnostic = (
                            campaign_kind == PREFLIGHT_CAMPAIGN_KIND
                        )
                        cross_gates.append(
                            {
                                "gate": (
                                    f"mode{mode}_inactive_stationary_tail_"
                                    f"u{wind:g}_dt{timestep:g}"
                                ),
                                "passed": (
                                    inactive_stationarity_criterion_passed
                                    or preflight_diagnostic
                                ),
                                "diagnostic_only": preflight_diagnostic,
                                "criterion_passed": (
                                    inactive_stationarity_criterion_passed
                                ),
                                "observed": inactive_stationarity_error,
                                "limit_or_expected": inactive_stationarity_limit,
                            }
                        )
                    cross_gates.append(
                        {
                            "gate": (
                                f"mode{mode}_closure_only_amplitude_reference_"
                                f"u{wind:g}_dt{timestep:g}"
                            ),
                            "passed": passed_shape,
                            "diagnostic_only": True,
                            "observed": observed,
                            "observed_response_minus_native_m": observed_increment,
                            "target": target,
                            "absolute_error_m": error,
                            "reference_allowance_m": allowance,
                            "limit_or_expected": "diagnostic only for complete composition",
                        }
                    )
                declared_peak_wind = float(peak_winds[mode])
                if declared_peak_wind in observed_by_wind:
                    declared_peak_amplitude = observed_by_wind[declared_peak_wind]
                    maximum_observed_amplitude = max(observed_by_wind.values())
                    observed_peak_winds = [
                        wind
                        for wind, amplitude in observed_by_wind.items()
                        if math.isclose(
                            amplitude,
                            maximum_observed_amplitude,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    ]
                    cross_gates.append(
                        {
                            "gate": f"mode{mode}_observed_peak_wind_dt{timestep:g}",
                            "passed": True,
                            "diagnostic_only": True,
                            "observed_peak_wind_speeds_mps": observed_peak_winds,
                            "observed_peak_amplitude_m": maximum_observed_amplitude,
                            "declared_peak_wind_speed_mps": declared_peak_wind,
                            "declared_peak_amplitude_m": declared_peak_amplitude,
                            "limit_or_expected": "diagnostic only; residual forcing may shift total-response maximum",
                        }
                    )
            if campaign_kind == FULL_CAMPAIGN_KIND and len(timesteps) == 2:
                peak_wind = float(peak_winds[mode])
                for scenario in SCENARIOS:
                    coarse = lookup[(scenario, peak_wind, timesteps[0])]
                    fine = lookup[(scenario, peak_wind, timesteps[1])]
                    coarse_metrics = coarse["metrics"]["modes"][mode]
                    fine_metrics = fine["metrics"]["modes"][mode]
                    amplitude_difference = _relative_pair_difference(
                        float(coarse_metrics["peak_amplitude_estimate_m"]),
                        float(fine_metrics["peak_amplitude_estimate_m"]),
                        0.01 * reference_length[mode],
                    )
                    cross_gates.append(
                        {
                            "gate": f"mode{mode}_{scenario}_dt_peak_amplitude_repeatability",
                            "passed": amplitude_difference
                            <= float(
                                acceptance[
                                    "maximum_active_case_dt_pair_peak_amplitude_relative_difference"
                                ]
                            ),
                            "observed": amplitude_difference,
                            "limit_or_expected": acceptance[
                                "maximum_active_case_dt_pair_peak_amplitude_relative_difference"
                            ],
                        }
                    )
                    frequencies = (
                        coarse_metrics["dominant_frequency_hz"],
                        fine_metrics["dominant_frequency_hz"],
                    )
                    frequency_difference = (
                        math.inf
                        if None in frequencies
                        else _relative_pair_difference(
                            float(frequencies[0]), float(frequencies[1]), 1.0e-12
                        )
                    )
                    cross_gates.append(
                        {
                            "gate": f"mode{mode}_{scenario}_dt_peak_frequency_repeatability",
                            "passed": frequency_difference
                            <= float(
                                acceptance[
                                    "maximum_active_case_dt_pair_peak_frequency_relative_difference"
                                ]
                            ),
                            "observed": frequency_difference,
                            "limit_or_expected": acceptance[
                                "maximum_active_case_dt_pair_peak_frequency_relative_difference"
                            ],
                        }
                    )
    all_complete = len(complete) == len(case_results)
    execution_complete = all_complete and all(
        bool(row.get("execution_receipt_pass")) for row in complete
    )
    if campaign_kind == TWO_SPEED_TEST_CAMPAIGN_KIND:
        passed = (
            execution_complete
            and all(bool(row["passed"]) for row in complete)
            and all(bool(row["passed"]) for row in cross_gates)
        )
        status = (
            "two_speed_test_passed_terminal_audit"
            if passed
            else "two_speed_test_failed_or_incomplete"
        )
    elif campaign_kind == PREFLIGHT_CAMPAIGN_KIND:
        passed = (
            execution_complete
            and all(bool(row["passed"]) for row in complete)
            and all(bool(row["passed"]) for row in cross_gates)
        )
        status = (
            "preflight_passed_complete_hybrid_screen"
            if passed
            else "preflight_failed_complete_hybrid_screen"
        )
    else:
        passed = (
            all_complete
            and all(bool(row["passed"]) for row in complete)
            and all(bool(row["passed"]) for row in cross_gates)
        )
        status = (
            "pilot_passed_complete_hybrid_screen"
            if passed
            else (
                "pilot_failed_complete_hybrid_screen"
                if all(row["status"] != "missing" for row in case_results)
                else "pilot_incomplete"
            )
        )
    preflight = campaign_kind == PREFLIGHT_CAMPAIGN_KIND
    report: dict[str, object] = {
        "schema": ANALYSIS_SCHEMA,
        "status": status,
        "campaign_fingerprint": manifest["fingerprint"],
        "campaign_kind": campaign_kind,
        "analysis_evaluator": {
            "pilot_model": _relative_identity(Path(__file__), repository),
            "pilot_cli": _relative_identity(
                repository / "run_headless/hvawt_response_pilot.py",
                repository,
            ),
        },
        "complete_case_count": len(complete),
        "expected_case_count": len(case_results),
        "passed_case_count": sum(bool(row["passed"]) for row in complete),
        "geometry_blended_airfoil_gate": (
            (
                "The 120 s preflight gates execution receipts, exact force/source "
                "composition, finite unclipped force application, "
                "active-branch modal frequency, "
                "and tail-L2 projection quality. Tail stationarity is diagnostic until "
                "the 660 s controlled-inflow checkpoint."
            )
            if preflight
            else (
                "Closure-only amplitudes are diagnostic after the uncaptured residual "
                "source is enabled. Acceptance requires exact source composition, "
                "spectral-energy closure, finite unclipped forces, and stationary "
                "frequency-locked active branches."
            )
        ),
        "claim_boundary": (
            (
                "A passed preflight verifies bounded completion and integration of the "
                "H0 and complete force paths. It is not stationary-response evidence, "
                "experimental H-VAWT validation, a turbulent fatigue result, or "
                "certification evidence."
            )
            if preflight
            else (
                "A passed campaign verifies the complete three-path H-VAWT engineering "
                "composition at uniform inflows. It is not experimental H-VAWT "
                "validation, a turbulent fatigue result, or certification evidence."
            )
        ),
        "full_rerun_authorized": (
            passed and campaign_kind == FULL_CAMPAIGN_KIND
        ),
        "cases": case_results,
        "cross_case_gates": cross_gates,
    }
    report["fingerprint"] = canonical_fingerprint(report)
    destination = (
        root
        / (
            (
                "two_speed_test_analysis.json"
                if campaign_kind == TWO_SPEED_TEST_CAMPAIGN_KIND
                else "preflight_analysis.json"
            )
            if campaign_kind
            in {PREFLIGHT_CAMPAIGN_KIND, TWO_SPEED_TEST_CAMPAIGN_KIND}
            else "pilot_analysis.json"
        )
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    atomic_write_json(destination, report)
    return destination


def render_pilot_result_artifacts(
    campaign: Path,
    *,
    analysis_path: Path | None = None,
    output_pdf: Path | None = None,
    output_csv: Path | None = None,
) -> dict[str, object]:
    """Render the completed pilot into a paper-facing table and figure.

    The renderer accepts either a passing or a failing *complete* pilot so that
    an unfavorable engineering result is still preserved and reported.  It
    refuses partial analyses and rechecks the sealed campaign before using the
    analysis payload.
    """

    state = validate_campaign(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    inputs = manifest["inputs"]
    analysis_file = (
        root / "pilot_analysis.json"
        if analysis_path is None
        else Path(analysis_path).expanduser().resolve()
    )
    analysis = _load_json(analysis_file, label="H-VAWT pilot analysis")
    expected_case_count = len(manifest.get("cases", []))
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("campaign_fingerprint") != manifest["fingerprint"]
        or analysis.get("complete_case_count") != expected_case_count
        or analysis.get("expected_case_count") != expected_case_count
        or analysis.get("status")
        not in {
            "pilot_passed_complete_hybrid_screen",
            "pilot_failed_complete_hybrid_screen",
        }
    ):
        raise HVAWTResponsePilotError(
            "Paper artifacts require a complete analysis bound to this full-sweep campaign."
        )
    cases = analysis.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != expected_case_count
        or any(
            not isinstance(case, Mapping) or case.get("status") != "complete"
            for case in cases
        )
    ):
        raise HVAWTResponsePilotError(
            f"Paper artifacts require {expected_case_count} complete case results."
        )

    spec = _load_json(
        root / str(inputs["campaign_spec"]["path"]), label="copied campaign spec"
    )
    model = _load_json(
        root / str(inputs["response_model"]["path"]), label="copied response model"
    )
    validate_campaign_spec(spec)
    validate_response_model(model, spec)
    modes = {str(row["tracked_mode"]): row for row in model["modes"]}
    peak_winds = spec["pilot"]["lock_in_peak_wind_speed_by_mode_mps"]
    amplitude_targets = spec["pilot"]["predicted_peak_amplitude_target_by_mode_m"]
    acceptance = spec["acceptance"]

    rows: list[dict[str, object]] = []
    for case in sorted(cases, key=lambda row: str(row["case_id"])):
        metrics = case.get("metrics")
        if not isinstance(metrics, Mapping) or not isinstance(
            metrics.get("modes"), Mapping
        ):
            raise HVAWTResponsePilotError(
                f"Case {case.get('case_id')} lacks modal metrics."
            )
        for mode in ("6", "12"):
            values = metrics["modes"].get(mode)
            if not isinstance(values, Mapping):
                raise HVAWTResponsePilotError(
                    f"Case {case.get('case_id')} lacks mode-{mode} metrics."
                )
            reference_length = float(modes[mode]["reference_length_m"])
            natural_frequency = float(
                DOCUMENTED_MODE_DAMPING[int(mode)]["frequency_hz"]
            )
            amplitude = float(values["peak_amplitude_estimate_m"])
            frequency_raw = values.get("dominant_frequency_hz")
            frequency = None if frequency_raw is None else float(frequency_raw)
            rows.append(
                {
                    "case_id": str(case["case_id"]),
                    "scenario": str(case["scenario"]),
                    "wind_speed_mps": float(case["wind_speed_mps"]),
                    "timestep_s": float(case["timestep_s"]),
                    "case_passed": bool(case["passed"]),
                    "tracked_mode": int(mode),
                    "reference_length_m": reference_length,
                    "peak_amplitude_estimate_m": amplitude,
                    "peak_amplitude_over_reference_length": amplitude
                    / reference_length,
                    "dominant_frequency_hz": frequency,
                    "dominant_frequency_over_natural_frequency": (
                        None if frequency is None else frequency / natural_frequency
                    ),
                    "activation_max": float(values["maximum_activation"]),
                    "complex_projection_leakage_ratio": float(
                        values["maximum_complex_projection_leakage_ratio"]
                    ),
                    "tail_half_peak_relative_difference": float(
                        values["tail_half_peak_relative_difference"]
                    ),
                }
            )

    artifact_root = root / "paper_artifacts"
    pdf_path = (
        artifact_root / "hvawt_response_pilot_results.pdf"
        if output_pdf is None
        else Path(output_pdf).expanduser().resolve()
    )
    csv_path = (
        artifact_root / "hvawt_response_pilot_results.csv"
        if output_csv is None
        else Path(output_csv).expanduser().resolve()
    )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    csv_tmp = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with csv_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_tmp, csv_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - production dependency gate
        raise HVAWTResponsePilotError(
            "matplotlib is required for the pilot paper figure."
        ) from exc

    colors = {
        SCENARIOS[0]: "#4C78A8",
        SCENARIOS[1]: "#E45756",
    }
    scenario_labels = {
        SCENARIOS[0]: "Native QBlade aero",
        SCENARIOS[1]: "Native + spectral-lock-in hybrid",
    }
    matrix = _campaign_matrix(spec)
    timesteps = tuple(float(value) for value in matrix["timesteps"])
    timestep_styles = {
        timestep: (("-", "o") if index == 0 else ("--", "s"))
        for index, timestep in enumerate(timesteps)
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharex="col")
    for column, mode in enumerate(("6", "12")):
        mode_rows = [row for row in rows if str(row["tracked_mode"]) == mode]
        for scenario in SCENARIOS:
            for timestep in timesteps:
                selected = sorted(
                    (
                        row
                        for row in mode_rows
                        if row["scenario"] == scenario
                        and math.isclose(
                            float(row["timestep_s"]),
                            timestep,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    ),
                    key=lambda row: float(row["wind_speed_mps"]),
                )
                linestyle, marker = timestep_styles[timestep]
                label = (
                    f"{scenario_labels[scenario]}, $\\Delta t={timestep:g}$ s"
                    if column == 0
                    else None
                )
                axes[0, column].plot(
                    [row["wind_speed_mps"] for row in selected],
                    [row["peak_amplitude_over_reference_length"] for row in selected],
                    color=colors[scenario],
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.6,
                    markersize=4.5,
                    label=label,
                )
                frequency_rows = [
                    row
                    for row in selected
                    if row["dominant_frequency_over_natural_frequency"] is not None
                ]
                axes[1, column].plot(
                    [row["wind_speed_mps"] for row in frequency_rows],
                    [
                        row["dominant_frequency_over_natural_frequency"]
                        for row in frequency_rows
                    ],
                    color=colors[scenario],
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.6,
                    markersize=4.5,
                )

        target = float(amplitude_targets[mode]) / float(
            modes[mode]["reference_length_m"]
        )
        allowance = float(
            acceptance["maximum_peak_amplitude_error_fraction_of_target"]
        ) * target + float(
            acceptance["peak_amplitude_absolute_allowance_fraction_of_reference_length"]
        )
        target_wind = float(peak_winds[mode])
        axes[0, column].errorbar(
            [target_wind],
            [target],
            yerr=[[allowance], [allowance]],
            color="black",
            marker="*",
            markersize=9,
            capsize=3,
            linewidth=1.1,
            label="Closure-only equilibrium reference" if column == 0 else None,
            zorder=5,
        )
        axes[0, column].axvline(target_wind, color="0.55", linewidth=0.8, linestyle=":")
        axes[1, column].axhspan(
            1.0 - float(acceptance["maximum_modal_peak_frequency_relative_error"]),
            1.0 + float(acceptance["maximum_modal_peak_frequency_relative_error"]),
            color="#59A14F",
            alpha=0.14,
        )
        axes[1, column].axhline(1.0, color="0.35", linewidth=0.8, linestyle=":")
        axes[0, column].set_title(f"Mode {mode}")
        axes[0, column].set_ylabel("Peak amplitude, $A/D$")
        axes[1, column].set_ylabel("Dominant frequency, $f/f_n$")
        axes[1, column].set_xlabel("Uniform inflow, m s$^{-1}$")
        axes[0, column].grid(True, alpha=0.22)
        axes[1, column].grid(True, alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    status_text = str(analysis["status"]).replace("_", " ")
    fig.suptitle(
        f"H-VAWT matched native/hybrid pilot — {status_text}",
        fontsize=11,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    pdf_tmp = pdf_path.with_name(f".{pdf_path.name}.tmp-{os.getpid()}")
    fig.savefig(
        pdf_tmp,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "H-VAWT matched native and spectral-lock-in hybrid pilot results",
            "Subject": "Engineering screening, not certification validation",
        },
    )
    plt.close(fig)
    os.replace(pdf_tmp, pdf_path)
    return {
        "analysis_status": analysis["status"],
        "row_count": len(rows),
        "pdf": _identity(pdf_path),
        "csv": _identity(csv_path),
    }


def authorize_full_rerun(
    campaign: Path,
    *,
    analysis_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Promote the prepared 102-run manifest after, and only after, a pilot pass."""

    state = validate_campaign(campaign)
    root = state["campaign_root"]
    manifest = state["manifest"]
    analysis_file = (
        root / "pilot_analysis.json"
        if analysis_path is None
        else Path(analysis_path).resolve()
    )
    analysis = _load_json(analysis_file, label="H-VAWT pilot analysis")
    if (
        analysis.get("schema") != ANALYSIS_SCHEMA
        or analysis.get("status") != "pilot_passed_complete_hybrid_screen"
        or analysis.get("full_rerun_authorized") is not True
        or analysis.get("campaign_fingerprint") != manifest["fingerprint"]
    ):
        raise HVAWTResponsePilotError(
            "Full rerun remains blocked until this campaign has a passing pilot analysis."
        )
    pending = _load_json(
        root / "full_rerun_plan_pending.json", label="pending full-rerun plan"
    )
    if (
        pending.get("campaign_fingerprint") != manifest["fingerprint"]
        or pending.get("expected_runs") != 102
        or pending.get("launch_implemented") is not True
        or tuple(pending.get("scenarios", [])) != FULL_RERUN_SCENARIOS
    ):
        raise HVAWTResponsePilotError(
            "Pending full-rerun plan is not bound to this pilot campaign."
        )
    authorized = dict(pending)
    authorized.update(
        {
            "status": "authorized_for_matched_three_path_execution",
            "execution_authorized": True,
            "launch_implemented": True,
            "pilot_analysis": _identity(analysis_file),
            "note": (
                "This authorization permits the separately sealed 102-case raw, "
                "one-way, and complete campaign; this pilot tool does not launch it."
            ),
        }
    )
    authorized["fingerprint"] = canonical_fingerprint(authorized)
    destination = (
        root / "full_rerun_plan_authorized.json"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    atomic_write_json(destination, authorized)
    return destination


__all__ = [
    "ANALYSIS_SCHEMA",
    "CAMPAIGN_SCHEMA",
    "FULL_RERUN_SCHEMA",
    "FULL_RERUN_SCENARIOS",
    "FLIGHT_RUNTIME_IDENTITY_NAME",
    "FLIGHT_RUNTIME_IDENTITY_SCHEMA",
    "FLIGHT_RESTART_SAVE_INTERVAL_SIMULATED_S",
    "FLIGHT_RESTART_SAVE_INTERVAL_STEPS",
    "HVAWTResponsePilotError",
    "FULL_CAMPAIGN_KIND",
    "PILOT_TIMESTEPS_S",
    "PILOT_WINDS_MPS",
    "PREFLIGHT_CAMPAIGN_KIND",
    "PREFLIGHT_PREPASS_FINGERPRINT",
    "SCENARIOS",
    "TWO_SPEED_TEST_CAMPAIGN_KIND",
    "TWO_SPEED_TEST_PREPASS_FINGERPRINT",
    "TWO_SPEED_TEST_SCENARIOS",
    "TWO_SPEED_TEST_WINDS_MPS",
    "analyze_campaign",
    "analyze_history",
    "authorize_full_rerun",
    "archive_incomplete_case_attempt",
    "bind_flight_runtime_identity",
    "build_pilot_cases",
    "build_flight_runtime_identity",
    "classify_case_execution",
    "command_for_case",
    "execute_case",
    "generate_campaign",
    "render_pilot_result_artifacts",
    "validate_campaign",
    "validate_campaign_spec",
    "validate_response_model",
]
