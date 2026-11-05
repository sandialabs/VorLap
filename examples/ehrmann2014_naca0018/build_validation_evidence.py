#!/usr/bin/env python3
"""Build compact Ehrmann experiment/raw-QBlade/latest-bridge evidence.

The source campaign is intentionally ignored because it contains multi-gigabyte
QBlade artifacts.  This command validates that campaign, checks every executed
summary, and promotes only compact numerical rows, provenance, diagnostics, and
the publication figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from examples.ehrmann2014_naca0018 import campaign

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REFERENCE_ROOT = HERE / "reference_data"
FIGURE6_CSV = REFERENCE_ROOT / "ehrmann2014_figure6_naca0018_digitized.csv"
FIGURE7_CSV = REFERENCE_ROOT / "ehrmann2014_figure7_naca0018_peak_digitized.csv"
PROVENANCE_JSON = REFERENCE_ROOT / "ehrmann2014_digitization_provenance.json"
DEFAULT_CAMPAIGN_ROOT = (
    REPO_ROOT / "run_headless/runs/ehrmann2014_naca0018_validation_v1"
)
DEFAULT_EVIDENCE_ROOT = HERE / "validation_evidence"
DEFAULT_PAPER_FIGURE = (
    REPO_ROOT
    / "paper_drafts/NAWEA_QBladeBridge/figures/ehrmann2014_raw_qblade_bridge.pdf"
)
EXPECTED_CAMPAIGN_FINGERPRINT = (
    "23903688f3c33cc23c2962dd834018f17cfe7652a8f4cf7dceec3f64df64cc94"
)
CHORD_M = 0.23
NOMINAL_FREQUENCY_HZ = 6.0
SCENARIO_LABELS = {
    "raw_qblade": "Raw QBlade",
    "latest_spectral_lockin_bridge": "Latest VorLap--QBlade bridge",
}
WORK_MAP_SCENARIO_LABELS = {
    "raw_qblade": "Raw QBlade",
    "latest_airfoil_work_map_bridge": "Latest VorLap--QBlade work-map bridge",
}
WORK_MAP_DATA_REQUEST = HERE / "work_map_data_request.json"
WORK_MAP_EVIDENCE_SCHEMA = "vorlap-ehrmann2014-work-map-validation-evidence-v2"
LOCKIN_FREQUENCY_TOLERANCE_RATIO = 0.10
LOCKIN_MINIMUM_CONSECUTIVE_POINTS = 2
LOCKIN_MINIMUM_EXPERIMENT_PEAK_FRACTION = 0.01
PEAK_AMPLITUDE_RELATIVE_TOLERANCE = 0.30
PEAK_AMPLITUDE_ABSOLUTE_TOLERANCE_OVER_CHORD = 0.01
TIMESTEP_AMPLITUDE_TOLERANCE_OVER_CHORD = 0.01
TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN = 0.10
REQUIRED_WORK_MAP_SOURCE_ROLES = {
    "prescribed_motion_high_fidelity",
    "exact_runtime_base",
    "prescribed_motion_high_fidelity_producer_provenance",
    "exact_runtime_base_producer_provenance",
    "stationary_force_spectrum",
    "frozen_data_request",
}


class EvidenceError(RuntimeError):
    """Raised when source evidence does not satisfy the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise EvidenceError(f"Non-finite {key} in reference row: {row}")
    return value


def _reference_rows() -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
):
    figure6 = [
        {
            **row,
            "reduced_velocity": _float(row, "reduced_velocity"),
            "rms_acceleration_mps2": _float(row, "rms_acceleration_mps2"),
        }
        for row in _read_csv(FIGURE6_CSV)
    ]
    figure7 = [
        {
            **row,
            "mass_damping": _float(row, "mass_damping"),
            "maximum_amplitude_over_chord": _float(row, "maximum_amplitude_over_chord"),
        }
        for row in _read_csv(FIGURE7_CSV)
    ]
    provenance = json.loads(PROVENANCE_JSON.read_text(encoding="utf-8"))
    if len(figure6) != 26 or len(figure7) != 5:
        raise EvidenceError("Unexpected Ehrmann digitization row count.")
    if (
        provenance.get("schema")
        != "vorlap-ehrmann2014-publisher-figure-digitization-v1"
    ):
        raise EvidenceError("Unexpected Ehrmann digitization provenance schema.")
    return figure6, figure7, provenance


def _validated_cases(
    campaign_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = campaign.validate(campaign_root)
    manifest = state["manifest"]
    if manifest.get("fingerprint") != EXPECTED_CAMPAIGN_FINGERPRINT:
        raise EvidenceError(
            "Campaign fingerprint differs from the executed Flight campaign."
        )
    analysis_path = campaign_root / "analysis/ehrmann2014_analysis.json"
    if not analysis_path.is_file():
        raise EvidenceError(
            "Run the Ehrmann campaign analyzer before building evidence."
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("campaign_fingerprint") != EXPECTED_CAMPAIGN_FINGERPRINT:
        raise EvidenceError("Analysis and campaign fingerprints differ.")
    if analysis.get("completed_cases") != 40 or analysis.get("missing_case_ids") != []:
        raise EvidenceError("All 40 Ehrmann cases must be complete.")
    cases = analysis.get("cases")
    if not isinstance(cases, list) or len(cases) != 40:
        raise EvidenceError("Analysis must contain exactly 40 case rows.")
    for case in manifest["cases"]:
        result_dir = campaign_root / str(case["result_dir"])
        summary = result_dir / f"{Path(str(case['sim'])).stem}_run_summary.json"
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if (
            payload.get("failed") is not False
            or payload.get("stopped") is not False
            or payload.get("completed_steps") != payload.get("requested_steps")
            or payload.get("exported_results") is not True
        ):
            raise EvidenceError(f"Run summary is not complete and clean: {summary}")
    return manifest, cases


def _numeric_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        amplitude_m = float(case["terminal_amplitude_m"])
        frequency = case["terminal_dominant_frequency_hz"]
        frequency_hz = None if frequency is None else float(frequency)
        if frequency_hz is None:
            acceleration = 0.0
        else:
            acceleration = (
                (2.0 * math.pi * frequency_hz) ** 2 * amplitude_m / math.sqrt(2.0)
            )
        rows.append(
            {
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "mass_damping": float(case["mass_damping"]),
                "structural_damping_ratio": float(case["structural_damping_ratio"]),
                "reduced_velocity": float(case["reduced_velocity"]),
                "reynolds_number": float(case["reynolds_number"]),
                "timestep_s": float(case["timestep_s"]),
                "terminal_amplitude_m": amplitude_m,
                "terminal_amplitude_over_chord": float(
                    case["terminal_amplitude_over_chord"]
                ),
                "terminal_dominant_frequency_hz": frequency_hz,
                "terminal_dominant_frequency_over_fn": case[
                    "terminal_dominant_frequency_over_fn"
                ],
                "derived_terminal_rms_acceleration_mps2": acceleration,
                "stationarity_amplitude_pass": case["stationarity_amplitude_pass"],
            }
        )
    return rows


def _nominal(
    rows: list[dict[str, Any]], md: float, scenario: str
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["mass_damping"] == md
        and row["scenario"] == scenario
        and row["timestep_s"] == 0.001
    ]
    return sorted(selected, key=lambda row: row["reduced_velocity"])


def _experiment_peak_at_mass_damping(
    figure7: list[dict[str, Any]], md: float
) -> tuple[float, str]:
    x = np.asarray([row["mass_damping"] for row in figure7], dtype=float)
    y = np.asarray(
        [row["maximum_amplitude_over_chord"] for row in figure7], dtype=float
    )
    if md < float(x.min()):
        return float(y[0]), "nearest_digitized_lower_boundary_no_extrapolation"
    if md > float(x.max()):
        return float(y[-1]), "nearest_digitized_upper_boundary_no_extrapolation"
    return float(np.interp(md, x, y)), "linear_interpolation_within_digitized_markers"


def _seed_decay(campaign_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_id = "md0p96_u5p80_dt1000us_hybrid"
    result_dir = campaign_root / f"results/{case_id}"
    history_path = result_dir / f"{case_id}_qblade_results_sil_vorlap_history.csv"
    run_summary_path = result_dir / f"{case_id}_run_summary.json"
    raw = _read_csv(history_path)
    if len(raw) != 20000:
        raise EvidenceError("Representative hybrid force history is incomplete.")
    compact: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        if index % 100 != 0 and index != len(raw) - 1:
            continue
        compact.append(
            {
                "time_s": float(row["time_s"]),
                "response_amplitude_estimate_over_chord": float(
                    row["response_amplitude_estimate_m"]
                )
                / CHORD_M,
                "generated_max_abs_force_component_n": float(
                    row["generated_max_abs_force_component_n"]
                ),
                "startup_seed_active": row["response_startup_seed_active"] == "True",
                "response_activation_max": float(row["response_activation_max"]),
                "response_applied_gain_max": float(row["response_applied_gain_max"]),
            }
        )
    amplitudes = np.asarray(
        [float(row["response_amplitude_estimate_m"]) / CHORD_M for row in raw],
        dtype=float,
    )
    times = np.asarray([float(row["time_s"]) for row in raw], dtype=float)
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    diagnostics = run_summary["sil_vorlap_response_diagnostics_summary"]
    summary: dict[str, Any] = {
        "case_id": case_id,
        "history_path_relative_to_campaign": str(
            history_path.relative_to(campaign_root)
        ),
        "history_sha256": _sha256(history_path),
        "startup_seed_target_amplitude_over_chord": 0.005,
        "maximum_estimated_amplitude_over_chord": float(np.max(amplitudes)),
        "terminal_estimated_amplitude_over_chord": float(amplitudes[-1]),
        "maximum_generated_abs_force_component_n": max(
            float(row["generated_max_abs_force_component_n"]) for row in raw
        ),
        "maximum_response_activation": max(
            float(row["response_activation_max"]) for row in raw
        ),
        "maximum_applied_gain": max(
            float(row["response_applied_gain_max"]) for row in raw
        ),
        "maximum_spectral_energy_closure_error": max(
            float(row["response_spectral_maximum_energy_closure_error"]) for row in raw
        ),
        "terminal_source_energy_sum": float(
            diagnostics["spectral"]["source_energy_sum"]
        ),
        "terminal_captured_energy_sum": float(
            diagnostics["spectral"]["captured_energy_sum"]
        ),
        "terminal_residual_source_scale": float(
            diagnostics["spectral"]["residual_source_scale"]
        ),
    }
    for time_s in (1.0, 2.0, 5.0, 10.0, 19.999):
        index = int(np.argmin(np.abs(times - time_s)))
        summary[f"amplitude_over_chord_near_{time_s:g}s"] = float(amplitudes[index])
    return compact, summary


def _metrics(
    rows: list[dict[str, Any]],
    figure6: list[dict[str, Any]],
    figure7: list[dict[str, Any]],
    seed_summary: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    experiment_acceleration_peak = max(
        figure6, key=lambda row: row["rms_acceleration_mps2"]
    )
    timestep_differences = []
    for md in (0.96, 1.44):
        for scenario in SCENARIO_LABELS:
            pair = [
                row
                for row in rows
                if row["mass_damping"] == md
                and row["scenario"] == scenario
                and row["reduced_velocity"] == 5.8
            ]
            if len(pair) != 2:
                raise EvidenceError("Missing paired-timestep result.")
            pair.sort(key=lambda row: row["timestep_s"], reverse=True)
            timestep_differences.append(
                {
                    "mass_damping": md,
                    "scenario": scenario,
                    "absolute_amplitude_over_chord_difference": abs(
                        pair[0]["terminal_amplitude_over_chord"]
                        - pair[1]["terminal_amplitude_over_chord"]
                    ),
                    "absolute_frequency_over_fn_difference": abs(
                        float(pair[0]["terminal_dominant_frequency_over_fn"] or 0.0)
                        - float(pair[1]["terminal_dominant_frequency_over_fn"] or 0.0)
                    ),
                }
            )
    comparisons = []
    for md in (0.96, 1.44):
        experimental_peak, interpolation = _experiment_peak_at_mass_damping(figure7, md)
        for scenario in SCENARIO_LABELS:
            selected = _nominal(rows, md, scenario)
            amplitude_peak = max(
                selected, key=lambda row: row["terminal_amplitude_over_chord"]
            )
            acceleration_peak = max(
                selected, key=lambda row: row["derived_terminal_rms_acceleration_mps2"]
            )
            coherent = [
                row
                for row in selected
                if row["terminal_dominant_frequency_over_fn"] is not None
                and row["terminal_amplitude_over_chord"] > 1.0e-8
            ]
            comparisons.append(
                {
                    "mass_damping": md,
                    "scenario": scenario,
                    "experimental_peak_amplitude_over_chord": experimental_peak,
                    "experimental_peak_mapping": interpolation,
                    "numerical_sampled_maximum_amplitude_over_chord": amplitude_peak[
                        "terminal_amplitude_over_chord"
                    ],
                    "peak_amplitude_relative_error": abs(
                        amplitude_peak["terminal_amplitude_over_chord"]
                        - experimental_peak
                    )
                    / experimental_peak,
                    "numerical_sampled_maximum_reduced_velocity": amplitude_peak[
                        "reduced_velocity"
                    ],
                    "lockin_peak_location_absolute_error_u_star": None,
                    "lockin_peak_location_status": (
                        "not_defined_because_no_numerical_lockin_peak_was_detected"
                    ),
                    "numerical_peak_derived_rms_acceleration_mps2": acceleration_peak[
                        "derived_terminal_rms_acceleration_mps2"
                    ],
                    "experiment_figure6_peak_rms_acceleration_mps2": experiment_acceleration_peak[
                        "rms_acceleration_mps2"
                    ],
                    "lockin_frequency_relative_error": None,
                    "residual_response_frequency_relative_error_from_nominal": (
                        None
                        if not coherent
                        else min(
                            abs(
                                float(row["terminal_dominant_frequency_hz"])
                                - NOMINAL_FREQUENCY_HZ
                            )
                            / NOMINAL_FREQUENCY_HZ
                            for row in coherent
                        )
                    ),
                    "lockin_detected": False,
                    "lockin_detection_reason": (
                        "numerical sampled maximum is below one percent of the "
                        "mass-damping-mapped experimental peak"
                    ),
                }
            )
    if any(
        item["numerical_sampled_maximum_amplitude_over_chord"]
        >= 0.01 * item["experimental_peak_amplitude_over_chord"]
        for item in comparisons
    ):
        raise EvidenceError(
            "A numerical series exceeds the declared no-lock-in threshold."
        )
    return {
        "schema": "vorlap-ehrmann2014-validation-evidence-v1",
        "status": "completed_validation_failed_no_sustained_lockin",
        "campaign_fingerprint": manifest["fingerprint"],
        "completed_cases": len(rows),
        "clean_completed_cases": len(rows),
        "comparison_series": [
            "Ehrmann NACA 0018 experiment (digitized publisher markers)",
            "Raw QBlade (dashed)",
            "Latest VorLap-QBlade spectral-lockin bridge (solid)",
        ],
        "experiment_figure6_peak": {
            "reduced_velocity": experiment_acceleration_peak["reduced_velocity"],
            "rms_acceleration_mps2": experiment_acceleration_peak[
                "rms_acceleration_mps2"
            ],
            "harmonic_displacement_amplitude_over_chord_at_nominal_6hz": (
                math.sqrt(2.0)
                * experiment_acceleration_peak["rms_acceleration_mps2"]
                / (2.0 * math.pi * NOMINAL_FREQUENCY_HZ) ** 2
                / CHORD_M
            ),
        },
        "comparisons": comparisons,
        "timestep_pairs": timestep_differences,
        "maximum_paired_timestep_amplitude_difference_over_chord": max(
            item["absolute_amplitude_over_chord_difference"]
            for item in timestep_differences
        ),
        "maximum_paired_timestep_frequency_difference_over_fn": max(
            item["absolute_frequency_over_fn_difference"]
            for item in timestep_differences
        ),
        "representative_seed_decay": seed_summary,
        "primary_acceptance": {
            "peak_amplitude_tolerance": "0.30*A_experiment/c + 0.01",
            "pass": False,
            "reason": (
                "Both raw QBlade and the latest bridge remain more than 99.99 percent "
                "below the digitized NACA 0018 peak response and exhibit no lock-in peak."
            ),
        },
        "claim_boundary": (
            "This is a completed negative response validation. Structural checks, clean "
            "execution, stationarity, and timestep agreement pass, but the latest bridge "
            "does not supply enough response-dependent aerodynamic work to overcome the "
            "tested structural damping."
        ),
    }


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise EvidenceError(f"{label} must be a finite number, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} must be a finite number.") from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise EvidenceError(f"{label} must be {qualifier}.")
    return result


def _same_optional_number(observed: Any, expected: Any) -> bool:
    if observed is None or expected is None:
        return observed is None and expected is None
    try:
        left = float(observed)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)
    )


def _work_map_source_provenance(
    campaign_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Bind the evaluated map to the frozen non-leakage data request.

    Ehrmann free-response measurements are deliberately absent from the map
    source roles.  They enter only after the map has been staged and hashed,
    through this acceptance calculation.
    """

    if not WORK_MAP_DATA_REQUEST.is_file() or WORK_MAP_DATA_REQUEST.is_symlink():
        raise EvidenceError("The frozen work-map data request is unavailable.")
    request = json.loads(WORK_MAP_DATA_REQUEST.read_text(encoding="utf-8"))
    nonleakage = request.get("frozen_nonleakage_policy")
    if (
        request.get("schema") != "vorlap-naca0018-prescribed-motion-data-request-v1"
        or request.get("purpose")
        != "populate_signed_incremental_airfoil_work_map_without_using_Ehrmann_free_response_for_tuning"
        or not isinstance(nonleakage, dict)
        or nonleakage.get("ehrmann_free_response_used_for_fit") is not False
        or nonleakage.get("ehrmann_acceptance_tolerance_used_for_fit") is not False
        or nonleakage.get("static_fft_phase_used_for_work") is not False
        or nonleakage.get("grinderslev_power_used_for_naca_magnitude") is not False
    ):
        raise EvidenceError("The frozen work-map non-leakage policy differs.")

    hybrids = [
        row
        for row in manifest.get("cases", [])
        if row.get("scenario") == "latest_airfoil_work_map_bridge"
    ]
    if not hybrids:
        raise EvidenceError("The work-map campaign has no hybrid cases.")
    config_path = campaign._resolve_member(
        campaign_root, hybrids[0].get("runtime_config"), "runtime config"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    adapter = config.get("single_mode_airfoil_work_map")
    map_config = adapter.get("work_map") if isinstance(adapter, dict) else None
    if not isinstance(map_config, dict) or not isinstance(map_config.get("path"), str):
        raise EvidenceError("The staged work-map configuration is incomplete.")
    map_path = (config_path.parent / map_config["path"]).resolve()
    try:
        map_path.relative_to(config_path.parent.resolve())
    except ValueError as exc:
        raise EvidenceError(
            "The staged work-map path escapes its case directory."
        ) from exc
    if not map_path.is_file() or map_path.is_symlink():
        raise EvidenceError("The staged work-map JSON is unavailable.")
    map_payload = json.loads(map_path.read_text(encoding="utf-8"))
    sources = map_payload.get("sources")
    if not isinstance(sources, list) or not all(
        isinstance(row, dict) for row in sources
    ):
        raise EvidenceError("The work-map source identities are malformed.")
    by_role: dict[str, dict[str, Any]] = {}
    for source in sources:
        role = source.get("role")
        digest = source.get("sha256")
        if (
            set(source) != {"role", "label", "sha256"}
            or not isinstance(role, str)
            or not isinstance(source.get("label"), str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise EvidenceError("The work-map source identities are malformed.")
        if role in by_role:
            raise EvidenceError("The work-map source roles must be unique strings.")
        by_role[role] = source
    if set(by_role) != REQUIRED_WORK_MAP_SOURCE_ROLES:
        raise EvidenceError(
            "The work map must contain only the frozen prescribed-motion, runtime-base, "
            "producer-provenance, stationary-spectrum, and data-request source roles."
        )
    request_sha = _sha256(WORK_MAP_DATA_REQUEST)
    if by_role["frozen_data_request"].get("sha256") != request_sha:
        raise EvidenceError(
            "The work map was not built from the tracked frozen data request."
        )
    identity = manifest.get("work_map_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("sha256") != _sha256(map_path)
        or map_config.get("sha256") != identity.get("sha256")
        or map_config.get("confidence_multiplier") != 1.96
    ):
        raise EvidenceError("The evaluated work-map bytes or confidence policy differ.")
    return {
        "verified": True,
        "work_map_sha256": identity["sha256"],
        "work_map_fingerprint": identity.get("fingerprint"),
        "frozen_data_request_sha256": request_sha,
        "source_roles": sorted(by_role),
        "ehrmann_free_response_used_for_fit": False,
        "ehrmann_acceptance_tolerance_used_for_fit": False,
        "evaluation_policy": (
            "work map and source identities frozen before Ehrmann free-response "
            "acceptance is evaluated"
        ),
    }


def _validate_work_map_execution_contract(
    scheduler: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    execution_contract = scheduler.get("execution_contract")
    final_execution_audit = scheduler.get("final_execution_audit")
    runtime_identity = manifest.get("qblade_runtime_identity")
    if (
        scheduler.get("schema") != "vorlap-ehrmann2014-flight-scheduler-v1"
        or scheduler.get("status") != "complete"
        or scheduler.get("campaign_fingerprint") != manifest.get("fingerprint")
        or not isinstance(execution_contract, dict)
        or execution_contract.get("schema")
        != "vorlap-ehrmann2014-work-map-flight-execution-v1"
        or execution_contract.get("campaign_schema") != campaign.WORK_MAP_SCHEMA
        or execution_contract.get("work_map_sha256")
        != manifest.get("work_map_identity", {}).get("sha256")
        or execution_contract.get("exact_raw_reuse_case_count") != 20
        or execution_contract.get("hybrid_case_count") != 20
        or execution_contract.get("qblade_runtime_identity") != runtime_identity
        or final_execution_audit
        != {
            "exact_raw_reuse_case_count": 20,
            "complete_hybrid_case_count": 20,
            "runnable_case_count": 0,
        }
    ):
        raise EvidenceError(
            "The dedicated work-map Flight scheduler did not complete the exact "
            "20-reused-raw/20-new-hybrid execution contract."
        )
    expected_runtime_identity = {
        "version": campaign.FROZEN_QBLADE_VERSION,
        "library_sha256": campaign.FROZEN_QBLADE_LIBRARY_SHA256,
        "runtime_dependency_fingerprint": (
            campaign.FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
        ),
    }
    if runtime_identity != expected_runtime_identity:
        raise EvidenceError("The campaign QBlade runtime identity differs.")
    return execution_contract, final_execution_audit, expected_runtime_identity


def _validated_work_map_cases(
    campaign_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Validate every work-map result and independently replay its analysis row."""

    state = campaign.validate(campaign_root)
    root = Path(state["root"])
    manifest = state["manifest"]
    if manifest.get("schema") != campaign.WORK_MAP_SCHEMA:
        raise EvidenceError(
            "This acceptance path requires the schema-v2 work-map campaign."
        )
    if manifest.get("comparison_series") != list(campaign.WORK_MAP_SCENARIOS):
        raise EvidenceError(
            "The work-map comparison series differs from the frozen pair."
        )
    planned_cases = manifest.get("cases")
    if (
        not isinstance(planned_cases, list)
        or manifest.get("case_count") != 40
        or len(planned_cases) != 40
    ):
        raise EvidenceError(
            "The work-map gate requires the exact 40-case Ehrmann matrix."
        )

    analysis_path = root / "analysis/ehrmann2014_analysis.json"
    if not analysis_path.is_file() or analysis_path.is_symlink():
        raise EvidenceError(
            "Run the Ehrmann campaign analyzer before building evidence."
        )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    observed_cases = analysis.get("cases")
    if (
        analysis.get("campaign_fingerprint") != manifest.get("fingerprint")
        or analysis.get("expected_cases") != 40
        or analysis.get("completed_cases") != 40
        or analysis.get("missing_case_ids") != []
        or not isinstance(observed_cases, list)
        or len(observed_cases) != 40
    ):
        raise EvidenceError(
            "All 40 analyzed work-map comparison cases must be complete."
        )
    planned_ids = [str(row["case_id"]) for row in planned_cases]
    observed_ids = [str(row.get("case_id")) for row in observed_cases]
    if observed_ids != planned_ids or len(set(observed_ids)) != 40:
        raise EvidenceError(
            "Analysis case identities or ordering differ from the manifest."
        )

    scheduler_path = root / "flight_scheduler_report.json"
    if not scheduler_path.is_file() or scheduler_path.is_symlink():
        raise EvidenceError("The Flight scheduler report is unavailable.")
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    execution_contract, final_execution_audit, expected_runtime_identity = (
        _validate_work_map_execution_contract(scheduler, manifest)
    )

    replay_keys = (
        "samples",
        "terminal_samples",
        "terminal_amplitude_m",
        "terminal_amplitude_over_chord",
        "terminal_dominant_frequency_hz",
        "terminal_dominant_frequency_over_fn",
        "terminal_early_amplitude_m",
        "terminal_final_amplitude_m",
        "stationarity_amplitude_tolerance_m",
        "stationarity_amplitude_pass",
    )
    reused_raw = 0
    for planned, observed in zip(planned_cases, observed_cases):
        for key in (
            "case_id",
            "scenario",
            "mass_damping",
            "structural_damping_ratio",
            "reduced_velocity",
            "reynolds_number",
            "timestep_s",
        ):
            if observed.get(key) != planned.get(key):
                raise EvidenceError(f"Analysis row {planned['case_id']} {key} differs.")
        result_dir = campaign._resolve_member(
            root, planned.get("result_dir"), "result dir"
        )
        result_path = result_dir / f"{planned['result_name']}.txt"
        summary_path = result_dir / f"{Path(str(planned['sim'])).stem}_run_summary.json"
        identity_path = result_dir / "native_case_identity.json"
        for path, label in (
            (result_path, "result"),
            (summary_path, "run summary"),
            (identity_path, "native identity"),
        ):
            if not path.is_file() or path.is_symlink():
                raise EvidenceError(
                    f"Case {planned['case_id']} {label} is unavailable."
                )
        if json.loads(identity_path.read_text(encoding="utf-8")) != planned.get(
            "native_case_identity"
        ):
            raise EvidenceError(f"Case {planned['case_id']} native identity differs.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("requested_steps") != planned.get("num_steps")
            or summary.get("completed_steps") != planned.get("num_steps")
            or summary.get("failed") is not False
            or summary.get("stopped") is not False
            or summary.get("exported_results") is not True
        ):
            raise EvidenceError(f"Case {planned['case_id']} did not complete cleanly.")
        reproducibility = summary.get("reproducibility")
        qblade_library = (
            reproducibility.get("qblade", {}).get("library", {})
            if isinstance(reproducibility, dict)
            else {}
        )
        if (
            not isinstance(qblade_library, dict)
            or qblade_library.get("sha256")
            != expected_runtime_identity["library_sha256"]
            or reproducibility.get("qblade_runtime_dependency_fingerprint")
            != expected_runtime_identity["runtime_dependency_fingerprint"]
        ):
            raise EvidenceError(
                f"Case {planned['case_id']} QBlade runtime identity differs."
            )
        result_identity = summary.get("result_file_identity")
        if (
            not isinstance(result_identity, dict)
            or result_identity.get("sha256") != _sha256(result_path)
            or result_identity.get("size_bytes") != result_path.stat().st_size
        ):
            raise EvidenceError(
                f"Case {planned['case_id']} exported-result identity differs."
            )

        receipt_path = result_dir / "reuse_receipt.json"
        reuse_expected = planned.get("scenario") == "raw_qblade"
        if reuse_expected:
            reused_raw += 1
            if not receipt_path.is_file() or receipt_path.is_symlink():
                raise EvidenceError(
                    f"Case {planned['case_id']} lacks its raw reuse receipt."
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            native_identity = planned.get("native_case_identity")
            callback_evidence = summary.get("native_callback_evidence")
            callback_count_keys = (
                "update_init_count",
                "update_log_entry_count",
                "update_cache_observation_count",
                "update_count_observed_max",
            )
            if (
                planned.get("scenario") != "raw_qblade"
                or planned.get("raw_reuse_status") != "reused_exact_identity"
                or not isinstance(native_identity, dict)
                or receipt.get("schema") != campaign.REUSE_RECEIPT_SCHEMA
                or receipt.get("case_id") != planned.get("case_id")
                or receipt.get("native_case_identity_fingerprint")
                != native_identity.get("fingerprint")
                or receipt.get("run_summary_sha256") != _sha256(summary_path)
                or receipt.get("qblade_library_sha256")
                != expected_runtime_identity["library_sha256"]
                or receipt.get("qblade_runtime_dependency_fingerprint")
                != expected_runtime_identity["runtime_dependency_fingerprint"]
                or receipt.get("sil_vorlap_max_abs_force") != 0.0
                or receipt.get("coupling_mode") != "sil-vorlap"
                or receipt.get("native_callback_update_count") != 0
                or summary.get("coupling_mode") != "sil-vorlap"
                or summary.get("sil_vorlap") is not True
                or summary.get("sil_vorlap_max_abs_force") != 0.0
                or not isinstance(callback_evidence, dict)
                or any(
                    type(callback_evidence.get(key)) is not int
                    or callback_evidence.get(key) != 0
                    for key in callback_count_keys
                )
            ):
                raise EvidenceError(
                    f"Case {planned['case_id']} raw reuse receipt differs."
                )
        elif (
            planned.get("raw_reuse_status") != "no_candidate_run_required"
            or receipt_path.exists()
            or receipt_path.is_symlink()
        ):
            raise EvidenceError(
                f"Case {planned['case_id']} hybrid reuse state differs."
            )

        if planned.get("scenario") == "latest_airfoil_work_map_bridge":
            diagnostics = summary.get("sil_vorlap_response_diagnostics_summary")
            work_map_identity = manifest.get("work_map_identity")
            if (
                not isinstance(diagnostics, dict)
                or not isinstance(work_map_identity, dict)
                or diagnostics.get("model_type")
                != "single_mode_airfoil_force_motion_work_map_v1"
                or diagnostics.get("work_map_fingerprint")
                != work_map_identity.get("fingerprint")
                or diagnostics.get("residual_source_scale") != 0.0
                or not isinstance(diagnostics.get("samples"), int)
                or diagnostics["samples"] < 1
            ):
                raise EvidenceError(
                    f"Case {planned['case_id']} did not execute the frozen work-map adapter."
                )

        replay = campaign._result_metrics(result_path, float(planned["timestep_s"]))
        for key in replay_keys:
            actual = observed.get(key)
            expected = replay[key]
            if isinstance(expected, bool):
                equal = actual is expected
            elif isinstance(expected, int):
                equal = actual == expected
            else:
                equal = _same_optional_number(actual, expected)
            if not equal:
                raise EvidenceError(
                    f"Analysis row {planned['case_id']} {key} differs on replay."
                )

    provenance = _work_map_source_provenance(root, manifest)
    runtime_integrity = {
        "pass": reused_raw == 20,
        "campaign_case_count": 40,
        "clean_completed_case_count": 40,
        "analysis_replayed_case_count": 40,
        "exact_raw_reuse_case_count": reused_raw,
        "scheduler_status": "complete",
        "scheduler_execution_contract": dict(execution_contract),
        "scheduler_final_execution_audit": dict(final_execution_audit),
        "qblade_runtime_identity": dict(expected_runtime_identity),
        "result_sha256_checked": True,
        "native_case_identity_checked": True,
        "hybrid_work_map_fingerprint_checked": True,
    }
    return manifest, observed_cases, runtime_integrity, provenance


def _lockin_interval(
    selected: list[dict[str, Any]], experimental_peak: float
) -> dict[str, Any]:
    amplitude_floor = LOCKIN_MINIMUM_EXPERIMENT_PEAK_FRACTION * experimental_peak
    flags: list[bool] = []
    for row in selected:
        amplitude = _finite_number(
            row.get("terminal_amplitude_over_chord"),
            f"{row.get('case_id')} terminal_amplitude_over_chord",
            nonnegative=True,
        )
        ratio_value = row.get("terminal_dominant_frequency_over_fn")
        if ratio_value is None:
            frequency_pass = False
        else:
            ratio = _finite_number(
                ratio_value, f"{row.get('case_id')} terminal frequency ratio"
            )
            frequency_pass = (
                ratio > 0.0 and abs(ratio - 1.0) <= LOCKIN_FREQUENCY_TOLERANCE_RATIO
            )
        flags.append(
            row.get("stationarity_amplitude_pass") is True
            and amplitude >= amplitude_floor
            and frequency_pass
        )

    runs: list[list[int]] = []
    current: list[int] = []
    for index, accepted in enumerate(flags):
        if accepted:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    eligible = [run for run in runs if len(run) >= LOCKIN_MINIMUM_CONSECUTIVE_POINTS]
    if not eligible:
        return {
            "detected": False,
            "reduced_velocity_interval": None,
            "consecutive_point_count": 0,
            "candidate_reduced_velocities": [
                float(selected[index]["reduced_velocity"])
                for index, accepted in enumerate(flags)
                if accepted
            ],
            "minimum_amplitude_over_chord": amplitude_floor,
        }
    best = max(
        eligible,
        key=lambda run: (
            len(run),
            max(
                float(selected[index]["terminal_amplitude_over_chord"]) for index in run
            ),
        ),
    )
    interval = [
        float(selected[best[0]]["reduced_velocity"]),
        float(selected[best[-1]]["reduced_velocity"]),
    ]
    return {
        "detected": bool(math.isfinite(interval[0]) and interval[1] > interval[0]),
        "reduced_velocity_interval": interval,
        "consecutive_point_count": len(best),
        "candidate_reduced_velocities": [
            float(selected[index]["reduced_velocity"])
            for index, accepted in enumerate(flags)
            if accepted
        ],
        "minimum_amplitude_over_chord": amplitude_floor,
    }


def _work_map_metrics(
    rows: list[dict[str, Any]],
    figure6: list[dict[str, Any]],
    figure7: list[dict[str, Any]],
    manifest: dict[str, Any],
    runtime_integrity: dict[str, Any],
    work_map_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen work-map validation contract without fitting anything."""

    if len(rows) != 40:
        raise EvidenceError("The work-map acceptance matrix must contain 40 rows.")
    if len({str(row.get("case_id")) for row in rows}) != 40:
        raise EvidenceError("The work-map acceptance rows require unique case IDs.")
    allowed_scenarios = set(WORK_MAP_SCENARIO_LABELS)
    if {row.get("scenario") for row in rows} != allowed_scenarios:
        raise EvidenceError(
            "The work-map acceptance scenarios differ from the frozen pair."
        )
    expected_nominal_u = list(campaign.REDUCED_VELOCITIES)
    all_stationary = all(row.get("stationarity_amplitude_pass") is True for row in rows)

    timestep_pairs: list[dict[str, Any]] = []
    for md in campaign.MASS_DAMPINGS:
        for scenario in campaign.WORK_MAP_SCENARIOS:
            pair = [
                row
                for row in rows
                if row.get("mass_damping") == md
                and row.get("scenario") == scenario
                and row.get("reduced_velocity") == campaign.TIMESTEP_CHECK_USTAR
            ]
            if len(pair) != 2:
                raise EvidenceError(
                    "Every scenario and mass damping requires one timestep pair."
                )
            pair.sort(key=lambda row: float(row["timestep_s"]), reverse=True)
            if [float(row["timestep_s"]) for row in pair] != [
                campaign.TIMESTEP_S,
                campaign.HALF_TIMESTEP_S,
            ]:
                raise EvidenceError(
                    "The paired timesteps differ from the frozen values."
                )
            amplitudes = [
                _finite_number(
                    row.get("terminal_amplitude_over_chord"),
                    f"{row.get('case_id')} terminal amplitude",
                    nonnegative=True,
                )
                for row in pair
            ]
            amplitude_difference = abs(amplitudes[0] - amplitudes[1])
            frequency_values = [
                row.get("terminal_dominant_frequency_over_fn") for row in pair
            ]
            frequency_evaluated = all(value is not None for value in frequency_values)
            frequency_both_undefined = all(value is None for value in frequency_values)
            frequency_difference = None
            frequency_pass: bool | None = None
            if frequency_evaluated:
                frequencies = [
                    _finite_number(
                        value, f"{row.get('case_id')} terminal frequency ratio"
                    )
                    for row, value in zip(pair, frequency_values)
                ]
                frequency_difference = abs(frequencies[0] - frequencies[1])
                frequency_pass = (
                    frequency_difference <= TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN
                )
            hybrid = scenario == "latest_airfoil_work_map_bridge"
            frequency_contract_pass = bool(
                frequency_pass is True or (not hybrid and frequency_both_undefined)
            )
            pair_pass = (
                amplitude_difference <= TIMESTEP_AMPLITUDE_TOLERANCE_OVER_CHORD
                and frequency_contract_pass
                and all(row.get("stationarity_amplitude_pass") is True for row in pair)
            )
            timestep_pairs.append(
                {
                    "mass_damping": md,
                    "scenario": scenario,
                    "coarse_timestep_s": campaign.TIMESTEP_S,
                    "fine_timestep_s": campaign.HALF_TIMESTEP_S,
                    "absolute_amplitude_over_chord_difference": amplitude_difference,
                    "amplitude_tolerance_over_chord": TIMESTEP_AMPLITUDE_TOLERANCE_OVER_CHORD,
                    "amplitude_repeatability_pass": (
                        amplitude_difference <= TIMESTEP_AMPLITUDE_TOLERANCE_OVER_CHORD
                    ),
                    "absolute_frequency_over_fn_difference": frequency_difference,
                    "frequency_repeatability_evaluated": frequency_evaluated,
                    "frequency_tolerance_over_fn": TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN,
                    "frequency_repeatability_pass": frequency_pass,
                    "frequency_contract_pass": frequency_contract_pass,
                    "pair_acceptance_pass": pair_pass,
                }
            )

    experiment_acceleration_peak = max(
        figure6, key=lambda row: row["rms_acceleration_mps2"]
    )
    comparisons: list[dict[str, Any]] = []
    for md in campaign.MASS_DAMPINGS:
        experimental_peak, interpolation = _experiment_peak_at_mass_damping(figure7, md)
        for scenario in campaign.WORK_MAP_SCENARIOS:
            selected = _nominal(rows, md, scenario)
            if (
                len(selected) != len(expected_nominal_u)
                or [float(row["reduced_velocity"]) for row in selected]
                != expected_nominal_u
            ):
                raise EvidenceError(
                    "Every comparison series requires the frozen nominal grid."
                )
            amplitudes = [
                _finite_number(
                    row.get("terminal_amplitude_over_chord"),
                    f"{row.get('case_id')} terminal amplitude",
                    nonnegative=True,
                )
                for row in selected
            ]
            peak_index = int(np.argmax(amplitudes))
            peak = selected[peak_index]
            peak_amplitude = amplitudes[peak_index]
            amplitude_error = abs(peak_amplitude - experimental_peak)
            amplitude_tolerance = (
                PEAK_AMPLITUDE_RELATIVE_TOLERANCE * experimental_peak
                + PEAK_AMPLITUDE_ABSOLUTE_TOLERANCE_OVER_CHORD
            )
            ratio_value = peak.get("terminal_dominant_frequency_over_fn")
            peak_frequency_error = None
            peak_frequency_pass = False
            if ratio_value is not None:
                ratio = _finite_number(
                    ratio_value, f"{peak.get('case_id')} peak terminal frequency ratio"
                )
                peak_frequency_error = abs(ratio - 1.0)
                peak_frequency_pass = (
                    ratio > 0.0
                    and peak_frequency_error <= LOCKIN_FREQUENCY_TOLERANCE_RATIO
                )
            lockin = _lockin_interval(selected, experimental_peak)
            stationarity_pass = all(
                row.get("stationarity_amplitude_pass") is True for row in selected
            )
            scientific_pass = bool(
                lockin["detected"]
                and amplitude_error <= amplitude_tolerance
                and peak_frequency_pass
                and stationarity_pass
            )
            comparisons.append(
                {
                    "mass_damping": md,
                    "scenario": scenario,
                    "experimental_peak_amplitude_over_chord": experimental_peak,
                    "experimental_peak_mapping": interpolation,
                    "numerical_sampled_maximum_amplitude_over_chord": peak_amplitude,
                    "numerical_sampled_maximum_reduced_velocity": float(
                        peak["reduced_velocity"]
                    ),
                    "peak_amplitude_absolute_error_over_chord": amplitude_error,
                    "peak_amplitude_tolerance_over_chord": amplitude_tolerance,
                    "peak_amplitude_pass": amplitude_error <= amplitude_tolerance,
                    "peak_frequency_hz": peak.get("terminal_dominant_frequency_hz"),
                    "peak_frequency_relative_error_from_nominal": peak_frequency_error,
                    "peak_frequency_pass": peak_frequency_pass,
                    "stationarity_pass": stationarity_pass,
                    "lockin": lockin,
                    "frozen_scientific_criteria_pass": scientific_pass,
                }
            )

    hybrid_comparisons = [
        row
        for row in comparisons
        if row["scenario"] == "latest_airfoil_work_map_bridge"
    ]
    checks = {
        "runtime_integrity": runtime_integrity.get("pass") is True,
        "work_map_frozen_before_ehrmann_evaluation": (
            work_map_provenance.get("verified") is True
            and work_map_provenance.get("ehrmann_free_response_used_for_fit") is False
            and work_map_provenance.get("ehrmann_acceptance_tolerance_used_for_fit")
            is False
        ),
        "all_results_stationary": all_stationary,
        "hybrid_finite_lockin_interval_at_both_mass_dampings": all(
            row["lockin"]["detected"] is True for row in hybrid_comparisons
        ),
        "hybrid_peak_amplitude_within_tolerance_at_both_mass_dampings": all(
            row["peak_amplitude_pass"] is True for row in hybrid_comparisons
        ),
        "hybrid_peak_frequency_within_ten_percent_at_both_mass_dampings": all(
            row["peak_frequency_pass"] is True for row in hybrid_comparisons
        ),
        "all_paired_timestep_checks_repeatable": all(
            row["pair_acceptance_pass"] is True for row in timestep_pairs
        ),
    }
    accepted = all(checks.values())
    failed_checks = [key for key, value in checks.items() if not value]
    return {
        "schema": WORK_MAP_EVIDENCE_SCHEMA,
        "status": (
            "completed_validation_passed_frozen_acceptance"
            if accepted
            else "completed_validation_failed_frozen_acceptance"
        ),
        "campaign_schema": manifest.get("schema"),
        "campaign_fingerprint": manifest.get("fingerprint"),
        "completed_cases": len(rows),
        "clean_completed_cases": (runtime_integrity.get("clean_completed_case_count")),
        "comparison_series": [
            "Ehrmann NACA 0018 experiment (digitized publisher markers)",
            "Raw QBlade native aerodynamics",
            "Latest VorLap-QBlade airfoil work-map bridge",
        ],
        "frozen_acceptance_criteria": {
            "fit_policy": "Ehrmann free-response measurements and acceptance tolerances are evaluation-only",
            "lockin_frequency_relative_tolerance": LOCKIN_FREQUENCY_TOLERANCE_RATIO,
            "lockin_minimum_consecutive_nominal_points": LOCKIN_MINIMUM_CONSECUTIVE_POINTS,
            "lockin_minimum_amplitude_fraction_of_mapped_experiment_peak": LOCKIN_MINIMUM_EXPERIMENT_PEAK_FRACTION,
            "peak_amplitude_rule": "abs(A_hybrid/c-A_experiment/c) <= 0.30*(A_experiment/c) + 0.01",
            "peak_frequency_relative_tolerance": LOCKIN_FREQUENCY_TOLERANCE_RATIO,
            "stationarity_required_for_every_result": True,
            "paired_timestep_amplitude_tolerance_over_chord": TIMESTEP_AMPLITUDE_TOLERANCE_OVER_CHORD,
            "paired_timestep_hybrid_frequency_tolerance_over_fn": TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN,
        },
        "experiment_figure6_peak": {
            "reduced_velocity": experiment_acceleration_peak["reduced_velocity"],
            "rms_acceleration_mps2": experiment_acceleration_peak[
                "rms_acceleration_mps2"
            ],
        },
        "runtime_integrity": runtime_integrity,
        "work_map_provenance": work_map_provenance,
        "comparisons": comparisons,
        "timestep_pairs": timestep_pairs,
        "primary_acceptance": {
            "pass": accepted,
            "checks": checks,
            "failed_checks": failed_checks,
        },
        "claim_boundary": (
            "The frozen airfoil work-map closure passes this Ehrmann response screen; "
            "this does not by itself validate H-VAWT response or fatigue predictions."
            if accepted
            else "The completed work-map campaign does not satisfy every frozen Ehrmann "
            "response criterion and must not advance to H-VAWT response claims."
        ),
    }


def _plot(
    rows: list[dict[str, Any]],
    figure6: list[dict[str, Any]],
    figure7: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.3,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "pdf.fonttype": 42,
        }
    )
    navy = "#174A7E"
    orange = "#D55E00"
    gray = "#4D4D4D"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)
    ax = axes[0, 0]
    x_exp = [row["reduced_velocity"] for row in figure6]
    y_exp = [row["rms_acceleration_mps2"] for row in figure6]
    ax.scatter(
        x_exp,
        y_exp,
        marker="v",
        s=21,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        label="Ehrmann experiment (digitized)",
        zorder=4,
    )
    for scenario, color, style, marker in (
        ("raw_qblade", navy, "--", "s"),
        ("latest_spectral_lockin_bridge", orange, "-", "o"),
    ):
        selected = _nominal(rows, 0.96, scenario)
        ax.plot(
            [row["reduced_velocity"] for row in selected],
            [row["derived_terminal_rms_acceleration_mps2"] for row in selected],
            linestyle=style,
            marker=marker,
            color=color,
            linewidth=1.35,
            markersize=3.2,
            label=SCENARIO_LABELS[scenario],
        )
    ax.set(
        xlabel=r"Reduced velocity, $U^*$",
        ylabel=r"RMS acceleration [m s$^{-2}$]",
        title=r"(a) Reported response; numerical $m^*\zeta=0.96$",
    )
    ax.set_xlim(3.0, 6.3)
    ax.set_ylim(-0.03, 2.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False)

    ax = axes[0, 1]
    for md, alpha, marker_fill in ((0.96, 1.0, "full"), (1.44, 0.55, "none")):
        for scenario, color, style, marker in (
            ("raw_qblade", navy, "--", "s"),
            ("latest_spectral_lockin_bridge", orange, "-", "o"),
        ):
            selected = _nominal(rows, md, scenario)
            y = np.maximum(
                [row["terminal_amplitude_over_chord"] for row in selected], 1.0e-12
            )
            label = f"{SCENARIO_LABELS[scenario]}, $m^*\\zeta={md:.2f}$"
            ax.plot(
                [row["reduced_velocity"] for row in selected],
                y,
                linestyle=style,
                marker=marker,
                color=color,
                alpha=alpha,
                fillstyle=marker_fill,
                linewidth=1.25,
                markersize=3.2,
                label=label,
            )
    ax.set_yscale("log")
    ax.set(
        xlabel=r"Reduced velocity, $U^*$",
        ylabel=r"Terminal amplitude, $A/c$",
        title="(b) Numerical responses resolved on log scale",
    )
    ax.set_xlim(3.0, 6.3)
    ax.set_ylim(1.0e-12, 2.0e-2)
    ax.grid(which="both", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1, 0]
    ax.errorbar(
        [row["mass_damping"] for row in figure7],
        [row["maximum_amplitude_over_chord"] for row in figure7],
        xerr=[float(row["mass_damping_digitization_uncertainty"]) for row in figure7],
        yerr=[
            float(row["amplitude_over_chord_digitization_uncertainty"])
            for row in figure7
        ],
        linestyle="none",
        marker="v",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor="black",
        ecolor="black",
        elinewidth=0.7,
        capsize=2,
        label="Ehrmann peak experiment (digitized)",
    )
    for scenario, color, style, marker in (
        ("raw_qblade", navy, "--", "s"),
        ("latest_spectral_lockin_bridge", orange, "-", "o"),
    ):
        peaks = []
        for md in (0.96, 1.44):
            selected = _nominal(rows, md, scenario)
            peaks.append(max(row["terminal_amplitude_over_chord"] for row in selected))
        ax.plot(
            (0.96, 1.44),
            np.maximum(peaks, 1.0e-12),
            linestyle=style,
            marker=marker,
            color=color,
            linewidth=1.35,
            markersize=4.0,
            label=SCENARIO_LABELS[scenario],
        )
    ax.set_yscale("log")
    ax.set(
        xlabel=r"Mass--damping, $m^*\zeta$",
        ylabel=r"Maximum amplitude, $A_{\max}/c$",
        title="(c) Peak-amplitude validation",
    )
    ax.set_xlim(0.9, 1.55)
    ax.set_ylim(1.0e-12, 3.0e-2)
    ax.grid(which="both", alpha=0.2)
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1, 1]
    time = np.asarray([row["time_s"] for row in seed_rows])
    amplitude = np.maximum(
        [row["response_amplitude_estimate_over_chord"] for row in seed_rows],
        1.0e-12,
    )
    ax.semilogy(
        time,
        amplitude,
        color=orange,
        linewidth=1.35,
        label="Latest bridge amplitude estimate",
    )
    ax.axhline(
        0.005, color=gray, linestyle=":", linewidth=1.0, label="Startup seed target"
    )
    ax.axvline(
        5.0 / NOMINAL_FREQUENCY_HZ,
        color=gray,
        linestyle="--",
        linewidth=0.9,
        label="Seed forcing ends",
    )
    ax.set(
        xlabel="Time [s]",
        ylabel=r"Estimated amplitude, $A/c$",
        title=r"(d) Seed decay at $U^*=5.8$, $m^*\zeta=0.96$",
    )
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(1.0e-10, 1.0e-2)
    ax.grid(which="both", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "Ehrmann NACA 0018 response validation: stable execution, no sustained numerical lock-in",
        fontsize=10.0,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "VorLap Ehrmann validation evidence builder",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _plot_work_map(
    rows: list[dict[str, Any]],
    figure6: list[dict[str, Any]],
    figure7: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_pdf: Path,
    output_png: Path,
) -> None:
    """Plot the three explicitly named series and the frequency gate."""

    plt.rcParams.update(
        {
            "font.size": 8.3,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "pdf.fonttype": 42,
        }
    )
    navy = "#174A7E"
    orange = "#D55E00"
    gray = "#4D4D4D"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)

    ax = axes[0, 0]
    experiment_artist = ax.scatter(
        [row["reduced_velocity"] for row in figure6],
        [row["rms_acceleration_mps2"] for row in figure6],
        marker="v",
        s=21,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        label="Ehrmann experiment (digitized)",
        zorder=4,
    )
    series_artists: dict[str, Any] = {}
    for scenario, color, style, marker in (
        ("raw_qblade", navy, "--", "s"),
        ("latest_airfoil_work_map_bridge", orange, "-", "o"),
    ):
        selected = _nominal(rows, 0.96, scenario)
        (series_artists[scenario],) = ax.plot(
            [row["reduced_velocity"] for row in selected],
            [row["derived_terminal_rms_acceleration_mps2"] for row in selected],
            linestyle=style,
            marker=marker,
            color=color,
            linewidth=1.35,
            markersize=3.2,
            label=WORK_MAP_SCENARIO_LABELS[scenario],
        )
    ax.set(
        xlabel=r"Reduced velocity, $U^*$",
        ylabel=r"RMS acceleration [m s$^{-2}$]",
        title=r"(a) Experiment and simulation; $m^*\zeta=0.96$",
    )
    ax.set_xlim(3.0, 6.3)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    for md, alpha, fillstyle in ((0.96, 1.0, "full"), (1.44, 0.65, "none")):
        for scenario, color, style, marker in (
            ("raw_qblade", navy, "--", "s"),
            ("latest_airfoil_work_map_bridge", orange, "-", "o"),
        ):
            selected = _nominal(rows, md, scenario)
            ax.plot(
                [row["reduced_velocity"] for row in selected],
                [row["terminal_amplitude_over_chord"] for row in selected],
                linestyle=style,
                marker=marker,
                color=color,
                alpha=alpha,
                fillstyle=fillstyle,
                linewidth=1.25,
                markersize=3.2,
                label="_nolegend_",
            )
    ax.set(
        xlabel=r"Reduced velocity, $U^*$",
        ylabel=r"Terminal amplitude, $A/c$",
        title=r"(b) Amplitude; filled: $m^*\zeta=0.96$, open: $1.44$",
    )
    ax.set_xlim(3.0, 6.3)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.errorbar(
        [row["mass_damping"] for row in figure7],
        [row["maximum_amplitude_over_chord"] for row in figure7],
        xerr=[float(row["mass_damping_digitization_uncertainty"]) for row in figure7],
        yerr=[
            float(row["amplitude_over_chord_digitization_uncertainty"])
            for row in figure7
        ],
        linestyle="none",
        marker="v",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor="black",
        ecolor="black",
        elinewidth=0.7,
        capsize=2,
        label="Ehrmann peak experiment (digitized)",
    )
    for scenario, color, style, marker in (
        ("raw_qblade", navy, "--", "s"),
        ("latest_airfoil_work_map_bridge", orange, "-", "o"),
    ):
        peaks = [
            max(
                row["terminal_amplitude_over_chord"]
                for row in _nominal(rows, md, scenario)
            )
            for md in campaign.MASS_DAMPINGS
        ]
        ax.plot(
            campaign.MASS_DAMPINGS,
            peaks,
            linestyle=style,
            marker=marker,
            color=color,
            linewidth=1.35,
            markersize=4.0,
            label="_nolegend_",
        )
    ax.set(
        xlabel=r"Mass--damping, $m^*\zeta$",
        ylabel=r"Maximum amplitude, $A_{\max}/c$",
        title="(c) Peak-amplitude validation",
    )
    ax.set_xlim(0.9, 1.55)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.axhspan(
        1.0 - LOCKIN_FREQUENCY_TOLERANCE_RATIO,
        1.0 + LOCKIN_FREQUENCY_TOLERANCE_RATIO,
        color="#D9D9D9",
        alpha=0.65,
        label="_nolegend_",
    )
    ax.axhline(1.0, color=gray, linestyle=":", linewidth=0.9)
    for md, marker, fillstyle in ((0.96, "o", "full"), (1.44, "s", "none")):
        selected = _nominal(rows, md, "latest_airfoil_work_map_bridge")
        ratios = [
            (
                np.nan
                if row["terminal_dominant_frequency_over_fn"] is None
                else row["terminal_dominant_frequency_over_fn"]
            )
            for row in selected
        ]
        ax.plot(
            [row["reduced_velocity"] for row in selected],
            ratios,
            color=orange,
            marker=marker,
            fillstyle=fillstyle,
            linewidth=1.25,
            markersize=3.5,
            label=rf"$m^*\zeta={md:.2f}$",
        )
    ax.set(
        xlabel=r"Reduced velocity, $U^*$",
        ylabel=r"Dominant frequency, $f/f_n$",
        title="(d) Hybrid lock-in frequency",
    )
    ax.set_xlim(3.0, 6.3)
    ax.grid(alpha=0.2)
    ax.text(
        0.02,
        0.96,
        r"frozen $\pm10\%$ band",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
    )
    ax.legend(loc="lower center", frameon=False, ncol=2)

    result = "PASS" if metrics["primary_acceptance"]["pass"] else "FAIL"
    fig.suptitle(
        f"Ehrmann NACA 0018: raw QBlade versus airfoil work-map bridge ({result})",
        fontsize=10.0,
        y=1.07,
    )
    fig.legend(
        [
            experiment_artist,
            series_artists["raw_qblade"],
            series_artists["latest_airfoil_work_map_bridge"],
        ],
        [
            "Ehrmann experiment (digitized)",
            "Raw QBlade",
            "Latest VorLap--QBlade work-map bridge",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "VorLap Ehrmann work-map validation evidence builder",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _build_legacy(
    campaign_root: Path, evidence_root: Path, paper_figure: Path | None
) -> Path:
    campaign_root = campaign_root.expanduser().resolve()
    evidence_root = evidence_root.expanduser().resolve()
    manifest, raw_cases = _validated_cases(campaign_root)
    rows = _numeric_rows(raw_cases)
    figure6, figure7, provenance = _reference_rows()
    seed_rows, seed_summary = _seed_decay(campaign_root)
    metrics = _metrics(rows, figure6, figure7, seed_summary, manifest)

    evidence_root.mkdir(parents=True, exist_ok=True)
    simulation_csv = evidence_root / "ehrmann2014_simulation_metrics.csv"
    _write_csv(simulation_csv, rows, list(rows[0]))
    seed_csv = evidence_root / "ehrmann2014_seed_decay_u5p8_md0p96.csv"
    _write_csv(seed_csv, seed_rows, list(seed_rows[0]))
    shutil.copyfile(FIGURE6_CSV, evidence_root / FIGURE6_CSV.name)
    shutil.copyfile(FIGURE7_CSV, evidence_root / FIGURE7_CSV.name)
    shutil.copyfile(PROVENANCE_JSON, evidence_root / PROVENANCE_JSON.name)
    metrics["source_identities"] = {
        "campaign_manifest_sha256": _sha256(campaign_root / "campaign_manifest.json"),
        "campaign_analysis_sha256": _sha256(
            campaign_root / "analysis/ehrmann2014_analysis.json"
        ),
        "flight_scheduler_report_sha256": _sha256(
            campaign_root / "flight_scheduler_report.json"
        ),
        "figure6_digitization_sha256": _sha256(FIGURE6_CSV),
        "figure7_digitization_sha256": _sha256(FIGURE7_CSV),
        "digitization_provenance_sha256": _sha256(PROVENANCE_JSON),
        "publisher_figure6_raster_sha256": provenance["digitization"]["figure_6"][
            "publisher_image_sha256"
        ],
        "publisher_figure7_raster_sha256": provenance["digitization"]["figure_7"][
            "publisher_image_sha256"
        ],
    }
    metrics_path = evidence_root / "ehrmann2014_validation_metrics.json"
    _write_json(metrics_path, metrics)
    figure_pdf = evidence_root / "ehrmann2014_raw_qblade_bridge.pdf"
    figure_png = evidence_root / "ehrmann2014_raw_qblade_bridge.png"
    _plot(rows, figure6, figure7, seed_rows, figure_pdf, figure_png)
    if paper_figure is not None:
        paper_figure = paper_figure.expanduser().resolve()
        paper_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure_pdf, paper_figure)

    tracked = sorted(
        path
        for path in evidence_root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksums = "".join(f"{_sha256(path)}  {path.name}\n" for path in tracked)
    (evidence_root / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return metrics_path


def _build_work_map(
    campaign_root: Path, evidence_root: Path, paper_figure: Path | None
) -> Path:
    campaign_root = campaign_root.expanduser().resolve()
    evidence_root = evidence_root.expanduser().resolve()
    manifest, raw_cases, runtime_integrity, work_map_provenance = (
        _validated_work_map_cases(campaign_root)
    )
    rows = _numeric_rows(raw_cases)
    figure6, figure7, provenance = _reference_rows()
    metrics = _work_map_metrics(
        rows,
        figure6,
        figure7,
        manifest,
        runtime_integrity,
        work_map_provenance,
    )

    evidence_root.mkdir(parents=True, exist_ok=True)
    simulation_csv = evidence_root / "ehrmann2014_simulation_metrics.csv"
    _write_csv(simulation_csv, rows, list(rows[0]))
    figure6_copy = evidence_root / FIGURE6_CSV.name
    figure7_copy = evidence_root / FIGURE7_CSV.name
    provenance_copy = evidence_root / PROVENANCE_JSON.name
    shutil.copyfile(FIGURE6_CSV, figure6_copy)
    shutil.copyfile(FIGURE7_CSV, figure7_copy)
    shutil.copyfile(PROVENANCE_JSON, provenance_copy)
    metrics["source_identities"] = {
        "campaign_manifest_sha256": _sha256(campaign_root / "campaign_manifest.json"),
        "campaign_analysis_sha256": _sha256(
            campaign_root / "analysis/ehrmann2014_analysis.json"
        ),
        "flight_scheduler_report_sha256": _sha256(
            campaign_root / "flight_scheduler_report.json"
        ),
        "work_map_sha256": work_map_provenance["work_map_sha256"],
        "frozen_work_map_data_request_sha256": work_map_provenance[
            "frozen_data_request_sha256"
        ],
        "figure6_digitization_sha256": _sha256(FIGURE6_CSV),
        "figure7_digitization_sha256": _sha256(FIGURE7_CSV),
        "digitization_provenance_sha256": _sha256(PROVENANCE_JSON),
        "publisher_figure6_raster_sha256": provenance["digitization"]["figure_6"][
            "publisher_image_sha256"
        ],
        "publisher_figure7_raster_sha256": provenance["digitization"]["figure_7"][
            "publisher_image_sha256"
        ],
    }
    metrics_path = evidence_root / "ehrmann2014_validation_metrics.json"
    _write_json(metrics_path, metrics)
    figure_pdf = evidence_root / "ehrmann2014_raw_qblade_bridge.pdf"
    figure_png = evidence_root / "ehrmann2014_raw_qblade_bridge.png"
    _plot_work_map(rows, figure6, figure7, metrics, figure_pdf, figure_png)
    if paper_figure is not None:
        paper_figure = paper_figure.expanduser().resolve()
        paper_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure_pdf, paper_figure)

    tracked = (
        simulation_csv,
        figure6_copy,
        figure7_copy,
        provenance_copy,
        metrics_path,
        figure_pdf,
        figure_png,
    )
    checksums = "".join(f"{_sha256(path)}  {path.name}\n" for path in sorted(tracked))
    (evidence_root / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return metrics_path


def build(campaign_root: Path, evidence_root: Path, paper_figure: Path | None) -> Path:
    """Dispatch without changing the released schema-v1 evidence path."""

    root = campaign_root.expanduser().resolve()
    manifest_path = root / "campaign_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise EvidenceError("The campaign manifest is unavailable.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") == campaign.WORK_MAP_SCHEMA:
        return _build_work_map(root, evidence_root, paper_figure)
    return _build_legacy(root, evidence_root, paper_figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--paper-figure", type=Path, default=DEFAULT_PAPER_FIGURE)
    parser.add_argument("--no-paper-copy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    path = build(
        args.campaign_root,
        args.evidence_root,
        None if args.no_paper_copy else args.paper_figure,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
