#!/usr/bin/env python3
"""Frozen amplitude-focused engineering screen for the schema-v2 VIV model.

This module deliberately does not replace the stricter hGAST reproduction
assessment in :mod:`response_benchmark`.  It implements the user-declared
engineering question: does the already-frozen response closure reproduce the
experimental lock-in peak amplitudes to within 30 percent, plus the published
digitization allowance?  The selected held-out points, tolerances, and claim
boundary are fixed before any held-out QBlade case is executed.

The functions here are pure assessment helpers.  Campaign dispatch remains a
separate, later integration step so this file cannot accidentally launch a
held-out case while the calibration campaign is still running.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any


SCREEN_SPEC_SCHEMA = "vlastos2026-schema-v2-engineering-holdout-screen-v2"
CALIBRATION_READINESS_SCHEMA = (
    "vlastos2026-schema-v2-engineering-calibration-readiness-v1"
)
HELDOUT_ASSESSMENT_SCHEMA = (
    "vlastos2026-schema-v2-engineering-peak-neighborhood-assessment-v1"
)
LEGACY_STRICT_CALIBRATION_SCHEMA = (
    "vlastos2026-calibration-stage-assessment-v1"
)
BENCHMARK_SUMMARY_SCHEMA = "vlastos2026-response-benchmark-v1"

CAMPAIGN_MANIFEST_SHA256 = (
    "b6b32cce4e20b2458c2f0fd42580966dd54f77322817a614f066f98bcc8296f8"
)
STATIONARY_PROTOCOL_SHA256 = (
    "00fdb882a16c06881db2dd3f54c4985f5c1555b54c3425bceed69d4664f360dc"
)
COEFFICIENT_FINGERPRINT_SHA256 = (
    "330242617d9d04d2cb771767fefbedac6414bea9a04a9b6ea9e813c0aba18c7d"
)
REFERENCE_PEAK_SHA256 = (
    "116e1d63eb6983cede7956280061b523b868926700dd999c5d1446f631ddfa7d"
)
HISTORICAL_SCREEN_V1_SHA256 = (
    "a32c1410bfe9bbb736799d599e9be212bc761332161cfc252809f9635df3c642"
)

# Canonical semantic hashes of the eight entries in the immutable Linux
# campaign manifest above.  Binding the selected payload itself closes the
# gap where a caller could assert the frozen manifest digest while supplying
# a re-written coefficient or response configuration for the same case IDs.
_HELDOUT_CASE_CONTENT_SHA256 = {
    "ambientv2_sc2p94_u6p25_dt": (
        "e2b184afea2ac3112f564399477d1b4864eeedeacec82b366fd10bf9750fcf6d"
    ),
    "ambientv2_sc2p94_u6p25_dt_half": (
        "bbfdfebb94d89ea3dcc64c19d56f40727cb9e5adcede2620d5e161532e8ed5c3"
    ),
    "ambientv2_sc2p94_u6p8_dt": (
        "92eeef04aa322efb074d4666b877d06dc9d284d71fee6a211353883899e541fa"
    ),
    "ambientv2_sc2p94_u6p8_dt_half": (
        "58a87a200562441dab601a4b1f9dc2abed1600dfe7230dbfa4d3ddb81f8b098b"
    ),
    "ambientv2_sc7p02_u6p2_dt": (
        "815de6a3674d4ff9aed1f77b077e12c597f54a4f4948875f5fa34098db3f7141"
    ),
    "ambientv2_sc7p02_u6p2_dt_half": (
        "211f706e5f459990313e92977e63ad5bf16024ce28f43f4f02dce73635e37652"
    ),
    "ambientv2_sc7p02_u6p25_dt": (
        "591efc8d347b6f554a1243f8e8c26bd0ad00dbd08fd31d331aa31d64c9e31d24"
    ),
    "ambientv2_sc7p02_u6p25_dt_half": (
        "bcd1d2796ec088655ef754c2fb65efb82a1f392b5af00ac98e1e749af444c3ce"
    ),
}
_RESPONSE_CONFIG_CONTENT_SHA256_BY_SCRUTON = {
    2.94: "884b0a477e35692b92e91c31c5a957fd545c04434491c7667853ab5fce10b471",
    7.02: "fff6f30581fb88d469df3d11b7e255a55b60a8d15b72b067b66217ec3b510a4d",
}
_RESPONSE_CONFIG_FINGERPRINT_BY_SCRUTON = {
    2.94: "0107d739ebda43482e81e47d0ddc8311a450c5e166088baf926b78bee3223261",
    5.18: "c6895da90eb4ff789fbca2d15303f1650c1a823f04714c9dd75dd5faa719d966",
    7.02: "cf9eaca30e0bf59bf04b94480ce1b156ad726a4ddebcb2011e1d277104186f6d",
}

CALIBRATION_SCRUTON_NUMBER = 5.18
# Exact sampled maximum of the hash-bound Lupi Figure-9 digitization used by
# response_benchmark.py; do not replace it with the rounded 0.235 table value.
CALIBRATION_EXPERIMENTAL_PEAK_AMPLITUDE_OVER_DIAMETER = 0.235286
CALIBRATION_REDUCED_VELOCITIES = (
    4.3,
    4.8,
    5.0,
    5.2,
    5.5,
    5.8,
    6.2,
    6.25,
    6.5,
    6.8,
    7.2,
    8.2,
)
TIMESTEP_LEVELS = ("dt", "dt_half")

PEAK_RELATIVE_TOLERANCE = 0.30
DIGITIZATION_UNCERTAINTY_OVER_DIAMETER = 0.010
TIMESTEP_ABSOLUTE_AMPLITUDE_TOLERANCE = 0.010
TIMESTEP_RELATIVE_AMPLITUDE_TOLERANCE = 0.10
TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN = 0.05
LOCK_IN_FREQUENCY_MINIMUM_OVER_FN = 0.85
LOCK_IN_FREQUENCY_MAXIMUM_OVER_FN = 1.15
ACTIVE_AMPLITUDE_FRACTION_OF_TARGET = 0.10
CALIBRATION_PEAK_LOCATION_TOLERANCE_USTAR = 0.75
OFF_RESONANCE_MAXIMUM_AMPLITUDE_OVER_DIAMETER = 0.05
NUMERICAL_COMPARISON_SLACK = 1.0e-12

_HELDOUT_CASES = {
    2.94: {
        "target_peak_amplitude_over_diameter": 0.350,
        "reduced_velocities": (6.25, 6.8),
    },
    7.02: {
        "target_peak_amplitude_over_diameter": 0.150,
        "reduced_velocities": (6.2, 6.25),
    },
}

_CALIBRATION_REQUIRED_CHECKS = (
    "input_matrix_eligible",
    "all_calibration_results_complete",
    "runtime_artifacts_evaluated",
    "runtime_provenance_pass",
    "runtime_execution_and_model_pass",
    "runtime_limits_pass",
    "startup_seed_evaluated",
    "startup_seed_pass",
    "result_and_estimator_stationarity_evaluated",
    "result_and_estimator_stationarity_pass",
    "runtime_case_audits_pass",
    "timestep_point_checks_evaluated",
    "timestep_amplitude_pass",
    "timestep_frequency_pass",
    "timestep_pair_runtime_stationarity_pass",
    "timestep_curve_properties_evaluated",
    "timestep_curve_properties_pass",
    "live_vs_analytical_evaluated",
    "live_vs_analytical_pass",
    "coordinate_consistency_evaluated",
    "coordinate_consistency_pass",
    "lock_in_mechanism_evaluated",
    "lock_in_mechanism_pass",
    "ambient_v2_equation_reproduction_evaluated",
    "ambient_v2_equation_reproduction_pass",
)
_CALIBRATION_EVALUATION_CHECKS = (
    "all_calibration_results_complete",
    "runtime_artifacts_evaluated",
    "startup_seed_evaluated",
    "result_and_estimator_stationarity_evaluated",
    "timestep_point_checks_evaluated",
    "timestep_curve_properties_evaluated",
    "live_vs_analytical_evaluated",
    "coordinate_consistency_evaluated",
    "lock_in_mechanism_evaluated",
    "ambient_v2_equation_reproduction_evaluated",
    "external_hgast_peak_evaluated",
    "external_hgast_curve_evaluated",
    "external_hgast_band_evaluated",
)
_CALIBRATION_EXTERNAL_PASS_CHECKS = (
    "external_hgast_peak_pass",
    "external_hgast_curve_pass",
    "external_hgast_band_pass",
)

_SOURCE_LIMIT_COUNT_KEYS = {
    "aliased_frequency_count",
    "aoa_out_of_range_count",
    "force_coefficient_out_of_range_count",
    "frequency_truncated_count",
    "nonfinite_count",
    "re_out_of_range_count",
    "source_clip_count",
}
_RESPONSE_LIMIT_COUNT_KEYS = {
    "force_limited_node_samples",
    "force_limited_steps",
    "gain_limited_node_samples",
    "gain_limited_steps",
}

_EVIDENCE_KEYS = {
    "campaign_manifest_sha256",
    "stationary_protocol_sha256",
    "coefficient_fingerprint_sha256",
    "acceptance_summary_content_sha256",
    "response_rows_content_sha256",
    "runtime_audits_content_sha256",
}
_HELDOUT_EVIDENCE_KEYS = {
    "campaign_manifest_sha256",
    "stationary_protocol_sha256",
    "coefficient_fingerprint_sha256",
    "calibration_readiness_content_sha256",
    "selected_cases_content_sha256",
    "response_rows_content_sha256",
    "runtime_audits_content_sha256",
}
_CALIBRATION_READINESS_CHECK_KEYS = (
    set(_CALIBRATION_REQUIRED_CHECKS)
    | set(_CALIBRATION_EVALUATION_CHECKS)
    | {
        "response_row_integrity_pass",
        "runtime_audit_payload_integrity_pass",
        "experimental_peak_amplitude_pass",
        "broad_peak_location_pass",
        "off_resonance_guard_pass",
    }
)
_CALIBRATION_READINESS_KEYS = {
    "schema",
    "status",
    "evaluated",
    "pass",
    "scope",
    "source_evidence",
    "expected_case_ids",
    "checks",
    "experimental_peak_amplitude_over_diameter",
    "response_peak_amplitude_over_diameter",
    "peak_amplitude_absolute_error",
    "peak_amplitude_tolerance",
    "peak_location_absolute_error_u_star",
    "peak_location_tolerance_u_star",
    "off_resonance_half_dt_amplitudes_over_diameter",
    "strict_hgast_stage_status_unchanged",
    "assessment_fingerprint_sha256",
}


def _case_id(scruton_number: float, reduced_velocity: float, level: str) -> str:
    def tag(value: float) -> str:
        return f"{value:.5f}".rstrip("0").rstrip(".").replace(".", "p")

    return f"ambientv2_sc{tag(scruton_number)}_u{tag(reduced_velocity)}_{level}"


def semantic_sha256(value: Any) -> str:
    """Hash one already-parsed evidence payload using canonical JSON bytes."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_fingerprint(value: Any) -> str:
    return semantic_sha256(value)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _is_true(value: Any) -> bool:
    return value is True or value == "True"


def _is_exact_int(value: Any, expected: int) -> bool:
    """Return true only for the requested built-in integer value.

    Python deliberately equates ``True`` with ``1`` and ``24.0`` with ``24``.
    Qualification metadata must not inherit those coercions.
    """

    return type(value) is int and value == expected


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 digest.")
    return value


def _validate_evidence(evidence: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(evidence, Mapping) or set(evidence) != _EVIDENCE_KEYS:
        raise ValueError("Engineering evidence must use the exact frozen key schema.")
    result = {
        key: _require_sha256(evidence[key], key) for key in sorted(_EVIDENCE_KEYS)
    }
    expected = {
        "campaign_manifest_sha256": CAMPAIGN_MANIFEST_SHA256,
        "stationary_protocol_sha256": STATIONARY_PROTOCOL_SHA256,
        "coefficient_fingerprint_sha256": COEFFICIENT_FINGERPRINT_SHA256,
    }
    for key, expected_value in expected.items():
        if result[key] != expected_value:
            raise ValueError(f"Engineering evidence {key} differs from the frozen campaign.")
    return result


def _validate_heldout_evidence(evidence: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(evidence, Mapping) or set(evidence) != _HELDOUT_EVIDENCE_KEYS:
        raise ValueError(
            "Heldout evidence must use the exact frozen key schema."
        )
    result = {
        key: _require_sha256(evidence[key], key)
        for key in sorted(_HELDOUT_EVIDENCE_KEYS)
    }
    expected = {
        "campaign_manifest_sha256": CAMPAIGN_MANIFEST_SHA256,
        "stationary_protocol_sha256": STATIONARY_PROTOCOL_SHA256,
        "coefficient_fingerprint_sha256": COEFFICIENT_FINGERPRINT_SHA256,
    }
    for key, expected_value in expected.items():
        if result[key] != expected_value:
            raise ValueError(f"Heldout evidence {key} differs from the frozen campaign.")
    return result


def engineering_screen_spec() -> dict[str, Any]:
    """Return a defensive copy of the predeclared engineering screen."""

    cases: dict[str, Any] = {}
    exact_case_ids: list[str] = []
    for sc, values in _HELDOUT_CASES.items():
        ids = [
            _case_id(sc, u_star, level)
            for u_star in values["reduced_velocities"]
            for level in TIMESTEP_LEVELS
        ]
        exact_case_ids.extend(ids)
        cases[str(sc)] = {
            "target_peak_amplitude_over_diameter": values[
                "target_peak_amplitude_over_diameter"
            ],
            "digitization_uncertainty_over_diameter": (
                DIGITIZATION_UNCERTAINTY_OVER_DIAMETER
            ),
            "reduced_velocities": list(values["reduced_velocities"]),
            "case_ids": ids,
        }
    spec = {
        "schema": SCREEN_SPEC_SCHEMA,
        "supersedes_unmodified_historical_schema": (
            "vlastos2026-schema-v2-engineering-holdout-screen-v1"
        ),
        "supersedes_unmodified_historical_artifact_sha256": (
            HISTORICAL_SCREEN_V1_SHA256
        ),
        "selection_timing": (
            "Frozen after analytical reference inspection but before any held-out "
            "QBlade execution or completed Sc=5.18 lock-in-peak result."
        ),
        "campaign_manifest_sha256": CAMPAIGN_MANIFEST_SHA256,
        "stationary_protocol_sha256": STATIONARY_PROTOCOL_SHA256,
        "coefficient_fingerprint_sha256": COEFFICIENT_FINGERPRINT_SHA256,
        "reference_peak_file": "reference_data/all_sc_figure10_peak_digitized.csv",
        "reference_peak_file_sha256": REFERENCE_PEAK_SHA256,
        "target_series": "experiment_lupi2021",
        "purpose": (
            "Engineering peak-amplitude adequacy diagnostic; not blind validation, "
            "full-curve validation, fatigue validation, or certification evidence."
        ),
        "exact_case_ids": exact_case_ids,
        "cases": cases,
        "acceptance": {
            "peak_relative_tolerance": PEAK_RELATIVE_TOLERANCE,
            "peak_digitization_allowance_over_diameter": (
                DIGITIZATION_UNCERTAINTY_OVER_DIAMETER
            ),
            "dt_absolute_amplitude_tolerance_over_diameter": (
                TIMESTEP_ABSOLUTE_AMPLITUDE_TOLERANCE
            ),
            "dt_relative_amplitude_tolerance": (
                TIMESTEP_RELATIVE_AMPLITUDE_TOLERANCE
            ),
            "dt_frequency_tolerance_over_fn": (
                TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN
            ),
            "lock_in_frequency_interval_over_fn": [
                LOCK_IN_FREQUENCY_MINIMUM_OVER_FN,
                LOCK_IN_FREQUENCY_MAXIMUM_OVER_FN,
            ],
            "selected_peak_rule": (
                "For each Scruton number, select the maximum amplitude from exactly "
                "the two predeclared stationary half-dt points; no post-result point "
                "addition, deletion, or best-of-menu selection is permitted."
            ),
            "active_point_frequency_rule": (
                "Apply the dt frequency-difference gate only where either timestep "
                "amplitude is at least 10% of that Scruton number's experimental "
                "target peak; the selected peak must always satisfy the lock-in "
                "frequency interval."
            ),
            "reporting_rule": (
                "All eight results must be reported with no clipping, exclusion, "
                "retuning, or best-of-menu selection."
            ),
        },
        "claim_boundary": {
            "allowed": (
                "The frozen response-aware bridge reproduces the two sparse "
                "experimental peak neighborhoods within the user-declared engineering "
                "amplitude tolerance."
            ),
            "forbidden": [
                "blind independent validation",
                "Hartlen-Currie or hGAST equation reproduction",
                "full lock-in curve validation",
                "phase or force-history validation",
                "fatigue conservatism",
                "H-VAWT aerodynamic validation",
                "certification qualification",
            ],
        },
    }
    spec["specification_fingerprint_sha256"] = _canonical_fingerprint(spec)
    return copy.deepcopy(spec)


def select_heldout_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the exact eight immutable manifest entries, or fail closed."""

    expected_ids = engineering_screen_spec()["exact_case_ids"]
    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in cases:
        case_id = entry.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError("Campaign case_id must be a string.")
        if case_id in by_id:
            raise ValueError(f"Campaign contains duplicate case_id {case_id!r}.")
        by_id[case_id] = entry
    if any(case_id not in by_id for case_id in expected_ids):
        missing = [case_id for case_id in expected_ids if case_id not in by_id]
        raise ValueError(f"Campaign is missing engineering held-out cases: {missing}.")

    selected: list[dict[str, Any]] = []
    for sc, values in _HELDOUT_CASES.items():
        for u_star in values["reduced_velocities"]:
            for level in TIMESTEP_LEVELS:
                case_id = _case_id(sc, u_star, level)
                entry = dict(by_id[case_id])
                observed = (
                    _finite_float(entry.get("scruton_number"), "scruton_number"),
                    _finite_float(entry.get("reduced_velocity"), "reduced_velocity"),
                    entry.get("timestep_level"),
                    entry.get("role"),
                )
                if observed != (sc, u_star, level, "held_out"):
                    raise ValueError(
                        f"Engineering case semantics differ for {case_id}: {observed}."
                    )
                expected_parameter = (
                    f"cases/{case_id}/model/Control/vorlap_qblade_external.json"
                )
                expected_sim = f"cases/{case_id}/{case_id}.sim"
                response_config = entry.get("response_model_config")
                if (
                    entry.get("coefficient_fingerprint_sha256")
                    != COEFFICIENT_FINGERPRINT_SHA256
                    or entry.get("parameter_file") != expected_parameter
                    or entry.get("sim_path") != expected_sim
                    or not isinstance(response_config, Mapping)
                    or semantic_sha256(response_config)
                    != _RESPONSE_CONFIG_CONTENT_SHA256_BY_SCRUTON[sc]
                    or semantic_sha256(entry)
                    != _HELDOUT_CASE_CONTENT_SHA256[case_id]
                ):
                    raise ValueError(
                        f"Engineering case manifest/config contract differs for {case_id}."
                    )
                selected.append(entry)
    if [entry["case_id"] for entry in selected] != expected_ids:
        raise RuntimeError("Internal engineering held-out selector order drifted.")
    return selected


def _expected_calibration_case_ids() -> list[str]:
    return [
        _case_id(CALIBRATION_SCRUTON_NUMBER, u_star, level)
        for u_star in CALIBRATION_REDUCED_VELOCITIES
        for level in TIMESTEP_LEVELS
    ]


def assess_calibration_readiness(
    acceptance_summary: Mapping[str, Any],
    response_rows: Sequence[Mapping[str, Any]],
    runtime_audits: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Assess whether the 24-case calibration is ready for the sparse holdout.

    Strict implementation, runtime, stationarity, and timestep gates remain
    conjunctive.  Only the external-data accuracy requirement is relaxed from
    full hGAST curve reproduction to the declared peak-amplitude objective.
    """

    identities = _validate_evidence(evidence)
    expected_content_hashes = {
        "acceptance_summary_content_sha256": semantic_sha256(
            acceptance_summary
        ),
        "response_rows_content_sha256": semantic_sha256(list(response_rows)),
        "runtime_audits_content_sha256": semantic_sha256(list(runtime_audits)),
    }
    for key, expected_value in expected_content_hashes.items():
        if identities[key] != expected_value:
            raise ValueError(
                f"Calibration evidence {key} does not match the assessed payload."
            )
    if acceptance_summary.get("schema") != BENCHMARK_SUMMARY_SCHEMA:
        raise ValueError("Calibration acceptance-summary schema is invalid.")
    if (
        acceptance_summary.get("coefficient_fingerprint_sha256")
        != COEFFICIENT_FINGERPRINT_SHA256
    ):
        raise ValueError("Calibration summary coefficient fingerprint changed.")
    if acceptance_summary.get("qualification_input_eligible") is not True:
        raise ValueError("Calibration campaign is not qualification-input eligible.")

    legacy = acceptance_summary.get("calibration_stage_assessment")
    if not isinstance(legacy, Mapping) or legacy.get("schema") != LEGACY_STRICT_CALIBRATION_SCHEMA:
        raise ValueError("Calibration summary lacks the strict stage assessment.")
    if (
        legacy.get("evaluated") is not True
        or legacy.get("status")
        not in {"calibration_stage_pass", "calibration_stage_fail"}
        or not isinstance(legacy.get("pass"), bool)
        or legacy.get("pass")
        != (legacy.get("status") == "calibration_stage_pass")
        or not _is_exact_int(legacy.get("expected_case_count"), 24)
        or not _is_exact_int(
            legacy.get("manifest_calibration_case_count"), 24
        )
        or not _is_exact_int(
            legacy.get("completed_calibration_case_count"), 24
        )
        or not _is_exact_int(
            legacy.get("runtime_calibration_case_count"), 24
        )
        or legacy.get("expected_scruton_number") != CALIBRATION_SCRUTON_NUMBER
        or tuple(legacy.get("expected_reduced_velocities", ()))
        != CALIBRATION_REDUCED_VELOCITIES
        or tuple(legacy.get("expected_timestep_levels", ())) != TIMESTEP_LEVELS
    ):
        raise ValueError("Calibration-stage matrix is incomplete or changed.")
    legacy_checks = legacy.get("checks")
    if not isinstance(legacy_checks, Mapping):
        raise ValueError("Calibration-stage checks are missing.")
    exact_strict_check_names = (
        set(_CALIBRATION_REQUIRED_CHECKS)
        | set(_CALIBRATION_EVALUATION_CHECKS)
        | set(_CALIBRATION_EXTERNAL_PASS_CHECKS)
    )
    if set(legacy_checks) != exact_strict_check_names:
        raise ValueError("Calibration-stage check schema changed.")
    strict_checks_pass = all(
        legacy_checks[name] is True for name in exact_strict_check_names
    )
    if legacy.get("pass") != strict_checks_pass:
        raise ValueError("Calibration-stage status/pass/checks are inconsistent.")
    check_values = {
        name: legacy_checks.get(name) is True
        for name in set(_CALIBRATION_REQUIRED_CHECKS)
        | set(_CALIBRATION_EVALUATION_CHECKS)
    }

    expected_ids = _expected_calibration_case_ids()
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in response_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in by_id:
            raise ValueError("Calibration response rows have invalid or duplicate case IDs.")
        by_id[case_id] = row
    if set(by_id) != set(expected_ids) or len(by_id) != 24:
        raise ValueError("Calibration response rows do not cover the exact 24-case matrix.")
    runtime_by_id: dict[str, Mapping[str, Any]] = {}
    for row in runtime_audits:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in runtime_by_id:
            raise ValueError(
                "Calibration runtime audits have invalid or duplicate case IDs."
            )
        runtime_by_id[case_id] = row
    if set(runtime_by_id) != set(expected_ids) or len(runtime_by_id) != 24:
        raise ValueError(
            "Calibration runtime audits do not cover the exact 24-case matrix."
        )
    row_integrity = True
    for u_star in CALIBRATION_REDUCED_VELOCITIES:
        for level in TIMESTEP_LEVELS:
            case_id = _case_id(CALIBRATION_SCRUTON_NUMBER, u_star, level)
            row = by_id[case_id]
            row_integrity = row_integrity and (
                _finite_float(row.get("scruton_number"), "scruton_number")
                == CALIBRATION_SCRUTON_NUMBER
                and _finite_float(row.get("reduced_velocity"), "reduced_velocity")
                == u_star
                and row.get("timestep_level") == level
                and row.get("role") == "calibration"
                and _is_true(row.get("runtime_audit_evaluated"))
                and _is_true(row.get("runtime_audit_pass"))
                and _is_true(row.get("stationarity_evaluated"))
                and _is_true(row.get("stationarity_pass"))
                and _case_path_valid(row.get("result_path"), case_id, "result")
                and row.get("result_path")
                == runtime_by_id[case_id]["input_identities"]["result"]["path"]
            )

    lupi = acceptance_summary.get("lupi_experimental_curve_screen")
    if not isinstance(lupi, Mapping) or lupi.get("evaluated") is not True:
        raise ValueError("Calibration Lupi experimental curve screen is not evaluated.")
    predicted_peak = _nonnegative_float(
        lupi.get("response_peak_amplitude_over_diameter"),
        "calibration response peak amplitude",
    )
    target_peak = _nonnegative_float(
        lupi.get("experimental_peak_amplitude_over_diameter"),
        "calibration experimental peak amplitude",
    )
    if not math.isclose(
        target_peak,
        CALIBRATION_EXPERIMENTAL_PEAK_AMPLITUDE_OVER_DIAMETER,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Calibration experimental peak target changed.")
    peak_error = abs(predicted_peak - target_peak)
    peak_tolerance = (
        PEAK_RELATIVE_TOLERANCE * target_peak
        + DIGITIZATION_UNCERTAINTY_OVER_DIAMETER
    )
    peak_amplitude_pass = (
        peak_error <= peak_tolerance + NUMERICAL_COMPARISON_SLACK
    )
    peak_location_error = _nonnegative_float(
        lupi.get("peak_location_absolute_error_u_star"),
        "calibration peak-location error",
    )
    peak_location_pass = (
        peak_location_error
        <= CALIBRATION_PEAK_LOCATION_TOLERANCE_USTAR
        + NUMERICAL_COMPARISON_SLACK
    )

    off_resonance_amplitudes = []
    for u_star in (CALIBRATION_REDUCED_VELOCITIES[0], CALIBRATION_REDUCED_VELOCITIES[-1]):
        row = by_id[_case_id(CALIBRATION_SCRUTON_NUMBER, u_star, "dt_half")]
        off_resonance_amplitudes.append(
            _nonnegative_float(
                row.get("tip_peak_amplitude_over_diameter"),
                "off-resonance amplitude",
            )
        )
    off_resonance_pass = all(
        amplitude
        <= OFF_RESONANCE_MAXIMUM_AMPLITUDE_OVER_DIAMETER
        + NUMERICAL_COMPARISON_SLACK
        for amplitude in off_resonance_amplitudes
    )

    checks = {
        **check_values,
        "response_row_integrity_pass": bool(row_integrity),
        "runtime_audit_payload_integrity_pass": all(
            _runtime_audit_pass(
                runtime_by_id[case_id],
                expected_response_config_fingerprint=(
                    _RESPONSE_CONFIG_FINGERPRINT_BY_SCRUTON[
                        CALIBRATION_SCRUTON_NUMBER
                    ]
                ),
            )
            for case_id in expected_ids
        ),
        "experimental_peak_amplitude_pass": peak_amplitude_pass,
        "broad_peak_location_pass": peak_location_pass,
        "off_resonance_guard_pass": off_resonance_pass,
    }
    evaluated = all(
        check_values[name] for name in _CALIBRATION_EVALUATION_CHECKS
    )
    passed = evaluated and all(checks.values())
    result = {
        "schema": CALIBRATION_READINESS_SCHEMA,
        "status": "pass" if passed else ("fail" if evaluated else "not_evaluated"),
        "evaluated": evaluated,
        "pass": passed,
        "scope": (
            "24-case Sc=5.18 engineering readiness for a sparse amplitude-focused "
            "heldout diagnostic; strict hGAST reproduction remains separate."
        ),
        "source_evidence": identities,
        "expected_case_ids": expected_ids,
        "checks": checks,
        "experimental_peak_amplitude_over_diameter": target_peak,
        "response_peak_amplitude_over_diameter": predicted_peak,
        "peak_amplitude_absolute_error": peak_error,
        "peak_amplitude_tolerance": peak_tolerance,
        "peak_location_absolute_error_u_star": peak_location_error,
        "peak_location_tolerance_u_star": (
            CALIBRATION_PEAK_LOCATION_TOLERANCE_USTAR
        ),
        "off_resonance_half_dt_amplitudes_over_diameter": off_resonance_amplitudes,
        "strict_hgast_stage_status_unchanged": legacy.get("status"),
    }
    result["assessment_fingerprint_sha256"] = _canonical_fingerprint(result)
    return result


def _validate_calibration_readiness_for_heldout(
    calibration_readiness: Mapping[str, Any],
    heldout_identities: Mapping[str, str],
) -> str:
    """Recompute the semantic invariants of a passing readiness artifact."""

    if not isinstance(calibration_readiness, Mapping):
        raise ValueError(
            "A passing engineering calibration-readiness assessment is required."
        )
    readiness_checks = calibration_readiness.get("checks")
    if (
        set(calibration_readiness) != _CALIBRATION_READINESS_KEYS
        or calibration_readiness.get("schema") != CALIBRATION_READINESS_SCHEMA
        or calibration_readiness.get("status") != "pass"
        or calibration_readiness.get("evaluated") is not True
        or calibration_readiness.get("pass") is not True
        or calibration_readiness.get("expected_case_ids")
        != _expected_calibration_case_ids()
        or not isinstance(readiness_checks, Mapping)
        or set(readiness_checks) != _CALIBRATION_READINESS_CHECK_KEYS
        or any(value is not True for value in readiness_checks.values())
    ):
        raise ValueError(
            "A passing engineering calibration-readiness assessment is required."
        )

    fingerprint = calibration_readiness.get("assessment_fingerprint_sha256")
    readiness_without_fingerprint = dict(calibration_readiness)
    readiness_without_fingerprint.pop("assessment_fingerprint_sha256", None)
    if fingerprint != _canonical_fingerprint(readiness_without_fingerprint):
        raise ValueError("Calibration-readiness assessment fingerprint is invalid.")
    fingerprint = _require_sha256(
        fingerprint,
        "calibration-readiness assessment fingerprint",
    )

    expected_scope = (
        "24-case Sc=5.18 engineering readiness for a sparse amplitude-focused "
        "heldout diagnostic; strict hGAST reproduction remains separate."
    )
    if calibration_readiness.get("scope") != expected_scope:
        raise ValueError("Calibration-readiness scope changed.")

    readiness_evidence = calibration_readiness.get("source_evidence")
    try:
        validated_readiness_evidence = _validate_evidence(readiness_evidence)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Calibration-readiness source evidence is invalid."
        ) from exc
    if any(
        validated_readiness_evidence.get(key) != heldout_identities[key]
        for key in (
            "campaign_manifest_sha256",
            "stationary_protocol_sha256",
            "coefficient_fingerprint_sha256",
        )
    ):
        raise ValueError(
            "Calibration-readiness campaign identity does not match heldout evidence."
        )

    target = _nonnegative_float(
        calibration_readiness.get("experimental_peak_amplitude_over_diameter"),
        "calibration-readiness experimental peak",
    )
    predicted = _nonnegative_float(
        calibration_readiness.get("response_peak_amplitude_over_diameter"),
        "calibration-readiness response peak",
    )
    error = _nonnegative_float(
        calibration_readiness.get("peak_amplitude_absolute_error"),
        "calibration-readiness peak error",
    )
    tolerance = _nonnegative_float(
        calibration_readiness.get("peak_amplitude_tolerance"),
        "calibration-readiness peak tolerance",
    )
    expected_tolerance = (
        PEAK_RELATIVE_TOLERANCE
        * CALIBRATION_EXPERIMENTAL_PEAK_AMPLITUDE_OVER_DIAMETER
        + DIGITIZATION_UNCERTAINTY_OVER_DIAMETER
    )
    if (
        not math.isclose(
            target,
            CALIBRATION_EXPERIMENTAL_PEAK_AMPLITUDE_OVER_DIAMETER,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            error,
            abs(predicted - target),
            rel_tol=0.0,
            abs_tol=NUMERICAL_COMPARISON_SLACK,
        )
        or not math.isclose(
            tolerance,
            expected_tolerance,
            rel_tol=0.0,
            abs_tol=NUMERICAL_COMPARISON_SLACK,
        )
        or error > tolerance + NUMERICAL_COMPARISON_SLACK
    ):
        raise ValueError("Calibration-readiness peak fields are inconsistent.")

    location_error = _nonnegative_float(
        calibration_readiness.get("peak_location_absolute_error_u_star"),
        "calibration-readiness peak-location error",
    )
    location_tolerance = _nonnegative_float(
        calibration_readiness.get("peak_location_tolerance_u_star"),
        "calibration-readiness peak-location tolerance",
    )
    if (
        not math.isclose(
            location_tolerance,
            CALIBRATION_PEAK_LOCATION_TOLERANCE_USTAR,
            rel_tol=0.0,
            abs_tol=NUMERICAL_COMPARISON_SLACK,
        )
        or location_error
        > CALIBRATION_PEAK_LOCATION_TOLERANCE_USTAR
        + NUMERICAL_COMPARISON_SLACK
    ):
        raise ValueError(
            "Calibration-readiness peak-location fields are inconsistent."
        )

    off_resonance = calibration_readiness.get(
        "off_resonance_half_dt_amplitudes_over_diameter"
    )
    if not isinstance(off_resonance, list) or len(off_resonance) != 2:
        raise ValueError("Calibration-readiness off-resonance fields are invalid.")
    off_resonance_values = [
        _nonnegative_float(value, "calibration-readiness off-resonance amplitude")
        for value in off_resonance
    ]
    if any(
        value
        > OFF_RESONANCE_MAXIMUM_AMPLITUDE_OVER_DIAMETER
        + NUMERICAL_COMPARISON_SLACK
        for value in off_resonance_values
    ):
        raise ValueError("Calibration-readiness off-resonance guard is inconsistent.")
    if calibration_readiness.get("strict_hgast_stage_status_unchanged") not in {
        "calibration_stage_pass",
        "calibration_stage_fail",
    }:
        raise ValueError("Calibration-readiness strict hGAST status changed.")

    return fingerprint


def _case_path_valid(value: Any, case_id: str, role: str) -> bool:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value.startswith("//")
    ):
        return False
    raw_parts = value.split("/")
    if any(part in {".", ".."} for part in raw_parts):
        return False
    try:
        path = PurePosixPath(value)
    except (TypeError, ValueError):
        return False
    if path.as_posix() != value:
        return False
    parts = path.parts
    if parts.count(case_id) != 1:
        return False
    tail = parts[parts.index(case_id) :]
    exact_tails = {
        "history": (case_id, "result_sil_vorlap_history.csv"),
        "result": (case_id, "result.txt"),
        "sim": (case_id, f"{case_id}.sim"),
        "summary": (case_id, f"{case_id}_run_summary.json"),
    }
    if role in exact_tails:
        return tail == exact_tails[role]
    if role == "vorlap_config":
        return tail == (
            case_id,
            "model",
            "Control",
            "vorlap_qblade_external.json",
        )
    if role == "turbine":
        return (
            len(tail) == 3
            and tail[:2] == (case_id, "model")
            and tail[2].endswith(".trb")
            and tail[2] != ".trb"
        )
    return False


def _file_identity_valid(value: Any, case_id: str, role: str) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        return False
    path = value.get("path")
    size = value.get("size_bytes")
    try:
        _require_sha256(value.get("sha256"), "file identity sha256")
    except ValueError:
        return False
    return bool(
        _case_path_valid(path, case_id, role)
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size > 0
    )


def _runtime_audit_pass(
    row: Mapping[str, Any],
    *,
    expected_response_config_fingerprint: str | None = None,
) -> bool:
    case_id = row.get("case_id")
    if not isinstance(case_id, str):
        return False
    startup = row.get("startup_seed_response")
    source_counts = row.get("source_limit_counts")
    response_counts = row.get("response_limit_counts")
    history = row.get("history")
    expected_config_fingerprint = row.get(
        "expected_response_config_fingerprint_sha256"
    )
    input_identities = row.get("input_identities")
    summary_path = row.get("summary_path")
    history_path = row.get("history_path")
    if expected_response_config_fingerprint is None:
        expected_response_config_fingerprint = expected_config_fingerprint
    return bool(
        row.get("artifacts_present") is True
        and row.get("evaluated") is True
        and row.get("pass") is True
        and row.get("provenance_evaluated") is True
        and row.get("provenance_pass") is True
        and row.get("execution_pass") is True
        and row.get("identity_mismatch") is False
        and row.get("identity_mismatches") == []
        and row.get("missing_identity_fields") == []
        and row.get("problems") == []
        and isinstance(input_identities, Mapping)
        and {
            "history",
            "result",
            "sim",
            "turbine",
            "vorlap_config",
        } == set(input_identities)
        and all(
            _file_identity_valid(value, case_id, role)
            for role, value in input_identities.items()
        )
        and isinstance(expected_config_fingerprint, str)
        and len(expected_config_fingerprint) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_config_fingerprint
        )
        and expected_config_fingerprint
        == expected_response_config_fingerprint
        and PurePosixPath(input_identities["sim"]["path"]).name
        == f"{case_id}.sim"
        and PurePosixPath(input_identities["result"]["path"]).name
        == "result.txt"
        and PurePosixPath(input_identities["history"]["path"]).name
        == "result_sil_vorlap_history.csv"
        and PurePosixPath(input_identities["vorlap_config"]["path"]).name
        == "vorlap_qblade_external.json"
        and row.get("source_coverage_complete") is True
        and row.get("source_limits_pass") is True
        and row.get("response_summary_limits_pass") is True
        and row.get("response_speed_summary_pass") is True
        and row.get("startup_seed_summary_pass") is True
        and isinstance(startup, Mapping)
        and startup.get("evaluated") is True
        and startup.get("pass") is True
        and isinstance(startup.get("export_time_grid_audit"), Mapping)
        and startup["export_time_grid_audit"].get("pass") is True
        and isinstance(source_counts, Mapping)
        and set(source_counts) == _SOURCE_LIMIT_COUNT_KEYS
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value == 0
            for value in source_counts.values()
        )
        and isinstance(response_counts, Mapping)
        and set(response_counts) == _RESPONSE_LIMIT_COUNT_KEYS
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value == 0
            for value in response_counts.values()
        )
        and isinstance(history, Mapping)
        and history.get("history_mode") == "compact"
        and history.get("history_schema") == "vorlap-sil-force-history-compact-v1"
        and isinstance(history.get("response_model_schema_version"), int)
        and not isinstance(history.get("response_model_schema_version"), bool)
        and history.get("response_model_schema_version") == 2
        and history.get("response_model_types") == ["modal_negative_damping"]
        and isinstance(history.get("response_enabled_values"), list)
        and len(history["response_enabled_values"]) == 1
        and history["response_enabled_values"][0] is True
        and history.get("aerodynamic_speed_basis")
        == "declared_ambient_crossflow"
        and history.get("runtime_model_pass") is True
        and history.get("compact_lag_provenance_pass") is True
        and history.get("applied_power_sign_convention_pass") is True
        and history.get("speed_basis_provenance_pass") is True
        and isinstance(history.get("speed_basis_invariant_violations"), int)
        and not isinstance(
            history.get("speed_basis_invariant_violations"), bool
        )
        and history.get("speed_basis_invariant_violations") == 0
        and isinstance(history.get("activation_invariant_violations"), int)
        and not isinstance(history.get("activation_invariant_violations"), bool)
        and history.get("activation_invariant_violations") == 0
        and history.get("limit_headroom_pass") is True
        and history.get("response_estimator_stationarity_evaluated") is True
        and history.get("response_estimator_stationarity_pass") is True
        and history.get("startup_seed_integrity_evaluated") is True
        and history.get("startup_seed_integrity_pass") is True
        and history.get("file_identity") == input_identities.get("history")
        and history.get("path") == history_path
        and _case_path_valid(summary_path, case_id, "summary")
        and _case_path_valid(history_path, case_id, "history")
        and history_path == input_identities["history"]["path"]
        and isinstance(row.get("source_sample_counts"), list)
        and bool(row["source_sample_counts"])
        and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in row["source_sample_counts"]
        )
    )


def assess_heldout_results(
    calibration_readiness: Mapping[str, Any],
    selected_cases: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    runtime_audits: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the exact eight-case peak-neighborhood diagnostic."""

    identities = _validate_heldout_evidence(evidence)
    expected_content_hashes = {
        "calibration_readiness_content_sha256": semantic_sha256(
            calibration_readiness
        ),
        "selected_cases_content_sha256": semantic_sha256(
            list(selected_cases)
        ),
        "response_rows_content_sha256": semantic_sha256(list(response_rows)),
        "runtime_audits_content_sha256": semantic_sha256(
            list(runtime_audits)
        ),
    }
    for key, expected_value in expected_content_hashes.items():
        if identities[key] != expected_value:
            raise ValueError(
                f"Heldout evidence {key} does not match the assessed payload."
            )
    expected_ids = engineering_screen_spec()["exact_case_ids"]
    supplied_case_ids = [
        entry.get("case_id") if isinstance(entry, Mapping) else None
        for entry in selected_cases
    ]
    if (
        len(selected_cases) != len(expected_ids)
        or len(supplied_case_ids) != len(expected_ids)
        or any(not isinstance(case_id, str) for case_id in supplied_case_ids)
        or set(supplied_case_ids) != set(expected_ids)
    ):
        raise ValueError(
            "Heldout selected_cases must contain exactly the eight frozen cases."
        )
    selected = select_heldout_cases(selected_cases)
    fingerprint = _validate_calibration_readiness_for_heldout(
        calibration_readiness,
        identities,
    )
    response_by_id: dict[str, Mapping[str, Any]] = {}
    runtime_by_id: dict[str, Mapping[str, Any]] = {}
    for row in response_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in response_by_id:
            raise ValueError("Heldout response rows have invalid or duplicate case IDs.")
        response_by_id[case_id] = row
    for row in runtime_audits:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in runtime_by_id:
            raise ValueError("Heldout runtime audits have invalid or duplicate case IDs.")
        runtime_by_id[case_id] = row
    expected_set = set(expected_ids)
    if (
        set(response_by_id) != expected_set
        or set(runtime_by_id) != expected_set
    ):
        raise ValueError(
            "Heldout evidence must contain exactly the eight selected cases; "
            "calibration and unknown rows are rejected rather than ignored."
        )

    case_semantics_pass = True
    runtime_pass = True
    stationarity_pass = True
    for entry in selected:
        case_id = str(entry["case_id"])
        response = response_by_id[case_id]
        case_semantics_pass = case_semantics_pass and (
            _finite_float(response.get("scruton_number"), "scruton_number")
            == _finite_float(entry.get("scruton_number"), "scruton_number")
            and _finite_float(response.get("reduced_velocity"), "reduced_velocity")
            == _finite_float(entry.get("reduced_velocity"), "reduced_velocity")
            and response.get("timestep_level") == entry.get("timestep_level")
            and response.get("role") == "held_out"
            and _is_true(response.get("runtime_audit_evaluated"))
            and _is_true(response.get("runtime_audit_pass"))
            and _case_path_valid(response.get("result_path"), case_id, "result")
            and response.get("result_path")
            == runtime_by_id[case_id]["input_identities"]["result"]["path"]
        )
        stationarity_pass = stationarity_pass and (
            _is_true(response.get("stationarity_evaluated"))
            and _is_true(response.get("stationarity_pass"))
        )
        scruton_number = _finite_float(
            entry.get("scruton_number"), "selected scruton_number"
        )
        runtime_pass = runtime_pass and _runtime_audit_pass(
            runtime_by_id[case_id],
            expected_response_config_fingerprint=(
                _RESPONSE_CONFIG_FINGERPRINT_BY_SCRUTON[scruton_number]
            ),
        )

    per_scruton: list[dict[str, Any]] = []
    for sc, values in _HELDOUT_CASES.items():
        target = float(values["target_peak_amplitude_over_diameter"])
        active_amplitude_threshold = ACTIVE_AMPLITUDE_FRACTION_OF_TARGET * target
        timestep_rows: list[dict[str, Any]] = []
        fine_candidates: list[Mapping[str, Any]] = []
        for u_star in values["reduced_velocities"]:
            coarse = response_by_id[_case_id(sc, u_star, "dt")]
            fine = response_by_id[_case_id(sc, u_star, "dt_half")]
            fine_candidates.append(fine)
            coarse_amplitude = _nonnegative_float(
                coarse.get("tip_peak_amplitude_over_diameter"), "coarse amplitude"
            )
            fine_amplitude = _nonnegative_float(
                fine.get("tip_peak_amplitude_over_diameter"), "fine amplitude"
            )
            amplitude_difference = abs(coarse_amplitude - fine_amplitude)
            amplitude_tolerance = max(
                TIMESTEP_ABSOLUTE_AMPLITUDE_TOLERANCE,
                TIMESTEP_RELATIVE_AMPLITUDE_TOLERANCE
                * max(coarse_amplitude, fine_amplitude),
            )
            active = max(coarse_amplitude, fine_amplitude) >= (
                active_amplitude_threshold - NUMERICAL_COMPARISON_SLACK
            )
            if active:
                coarse_frequency = _positive_float(
                    coarse.get("terminal_dominant_frequency_over_fn"),
                    "coarse frequency ratio",
                )
                fine_frequency = _positive_float(
                    fine.get("terminal_dominant_frequency_over_fn"),
                    "fine frequency ratio",
                )
                frequency_difference = abs(coarse_frequency - fine_frequency)
                frequency_pass = (
                    frequency_difference
                    <= TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN
                    + NUMERICAL_COMPARISON_SLACK
                )
            else:
                coarse_frequency = None
                fine_frequency = None
                frequency_difference = None
                frequency_pass = True
            timestep_rows.append(
                {
                    "reduced_velocity": u_star,
                    "active": active,
                    "active_amplitude_threshold_over_diameter": (
                        active_amplitude_threshold
                    ),
                    "coarse_amplitude_over_diameter": coarse_amplitude,
                    "half_dt_amplitude_over_diameter": fine_amplitude,
                    "amplitude_difference_over_diameter": amplitude_difference,
                    "amplitude_tolerance_over_diameter": amplitude_tolerance,
                    "amplitude_pass": (
                        amplitude_difference
                        <= amplitude_tolerance + NUMERICAL_COMPARISON_SLACK
                    ),
                    "coarse_frequency_over_fn": coarse_frequency,
                    "half_dt_frequency_over_fn": fine_frequency,
                    "frequency_difference_over_fn": frequency_difference,
                    "frequency_tolerance_over_fn": (
                        TIMESTEP_FREQUENCY_TOLERANCE_OVER_FN
                    ),
                    "frequency_pass": frequency_pass,
                }
            )
        selected_peak = max(
            fine_candidates,
            key=lambda row: _nonnegative_float(
                row.get("tip_peak_amplitude_over_diameter"), "fine amplitude"
            ),
        )
        selected_amplitude = _nonnegative_float(
            selected_peak.get("tip_peak_amplitude_over_diameter"),
            "selected peak amplitude",
        )
        selected_frequency = _positive_float(
            selected_peak.get("terminal_dominant_frequency_over_fn"),
            "selected peak frequency",
        )
        peak_error = abs(selected_amplitude - target)
        peak_tolerance = (
            PEAK_RELATIVE_TOLERANCE * target
            + DIGITIZATION_UNCERTAINTY_OVER_DIAMETER
        )
        dt_pass = all(
            row["amplitude_pass"] and row["frequency_pass"]
            for row in timestep_rows
        )
        lock_in_pass = (
            LOCK_IN_FREQUENCY_MINIMUM_OVER_FN
            - NUMERICAL_COMPARISON_SLACK
            <= selected_frequency
            <= LOCK_IN_FREQUENCY_MAXIMUM_OVER_FN
            + NUMERICAL_COMPARISON_SLACK
        )
        peak_pass = peak_error <= peak_tolerance + NUMERICAL_COMPARISON_SLACK
        per_scruton.append(
            {
                "scruton_number": sc,
                "timestep_pairs": timestep_rows,
                "timestep_pass": dt_pass,
                "selected_peak_case_id": selected_peak["case_id"],
                "selected_peak_reduced_velocity": _finite_float(
                    selected_peak.get("reduced_velocity"), "selected reduced velocity"
                ),
                "selected_peak_amplitude_over_diameter": selected_amplitude,
                "target_peak_amplitude_over_diameter": target,
                "digitization_uncertainty_over_diameter": (
                    DIGITIZATION_UNCERTAINTY_OVER_DIAMETER
                ),
                "peak_amplitude_absolute_error": peak_error,
                "peak_amplitude_tolerance": peak_tolerance,
                "peak_amplitude_pass": peak_pass,
                "selected_peak_frequency_over_fn": selected_frequency,
                "lock_in_frequency_pass": lock_in_pass,
                "pass": dt_pass and lock_in_pass and peak_pass,
            }
        )

    checks = {
        "exact_case_selector_pass": [entry["case_id"] for entry in selected]
        == expected_ids,
        "case_semantics_pass": bool(case_semantics_pass),
        "runtime_provenance_execution_limits_pass": bool(runtime_pass),
        "stationarity_pass": bool(stationarity_pass),
        "all_timestep_pairs_pass": all(row["timestep_pass"] for row in per_scruton),
        "both_peak_amplitudes_pass": all(
            row["peak_amplitude_pass"] for row in per_scruton
        ),
        "both_selected_peaks_lock_in_frequency_pass": all(
            row["lock_in_frequency_pass"] for row in per_scruton
        ),
        "coefficient_no_retune_pass": (
            all(
                entry.get("coefficient_fingerprint_sha256")
                == COEFFICIENT_FINGERPRINT_SHA256
                and semantic_sha256(entry.get("response_model_config"))
                == _RESPONSE_CONFIG_CONTENT_SHA256_BY_SCRUTON[
                    _finite_float(
                        entry.get("scruton_number"), "selected scruton number"
                    )
                ]
                for entry in selected
            )
        ),
    }
    passed = all(checks.values()) and all(row["pass"] for row in per_scruton)
    result = {
        "schema": HELDOUT_ASSESSMENT_SCHEMA,
        "status": "pass" if passed else "fail",
        "evaluated": True,
        "pass": passed,
        "scope": (
            "Sparse engineering peak-neighborhood diagnostic at two held-out "
            "Scruton numbers; not blind validation, full-curve validation, fatigue "
            "validation, or H-VAWT validation."
        ),
        "source_evidence": identities,
        "calibration_readiness_fingerprint_sha256": fingerprint,
        "screen_specification_fingerprint_sha256": engineering_screen_spec()[
            "specification_fingerprint_sha256"
        ],
        "exact_case_ids": expected_ids,
        "checks": checks,
        "by_scruton_number": per_scruton,
        "strict_hgast_gate_rewritten": False,
        "claim_boundary": engineering_screen_spec()["claim_boundary"],
    }
    result["assessment_fingerprint_sha256"] = _canonical_fingerprint(result)
    return result


__all__ = [
    "CALIBRATION_READINESS_SCHEMA",
    "HELDOUT_ASSESSMENT_SCHEMA",
    "SCREEN_SPEC_SCHEMA",
    "assess_calibration_readiness",
    "assess_heldout_results",
    "engineering_screen_spec",
    "select_heldout_cases",
]
