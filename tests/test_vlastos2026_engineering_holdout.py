from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import PurePosixPath

import pytest

from examples.vlastos2026_cantilever import engineering_holdout as holdout
from examples.vlastos2026_cantilever import response_benchmark as benchmark


_LINUX_CANONICAL_MODE_SHAPE = [
    0.0,
    0.0042941905550188085,
    0.01677349980830435,
    0.03683344359348968,
    0.06387093136402738,
    0.09728580835371185,
    0.1364829368932864,
    0.18087477053912412,
    0.22988437506395787,
    0.2829488509519264,
    0.3395231128653239,
    0.3990839826394248,
    0.46113455374220064,
    0.5252087868318991,
    0.5908762980782943,
    0.6577473043008546,
    0.7254776917340012,
    0.7937741783682278,
    0.8623995433456904,
    0.9311779008172386,
    1.0,
]


def _calibration_evidence(
    summary=None, rows=None, audits=None
) -> dict[str, str]:
    summary = _calibration_summary() if summary is None else summary
    rows = _calibration_rows() if rows is None else rows
    audits = _calibration_runtime_audits() if audits is None else audits
    return {
        "campaign_manifest_sha256": holdout.CAMPAIGN_MANIFEST_SHA256,
        "stationary_protocol_sha256": holdout.STATIONARY_PROTOCOL_SHA256,
        "coefficient_fingerprint_sha256": (
            holdout.COEFFICIENT_FINGERPRINT_SHA256
        ),
        "acceptance_summary_content_sha256": holdout.semantic_sha256(summary),
        "response_rows_content_sha256": holdout.semantic_sha256(rows),
        "runtime_audits_content_sha256": holdout.semantic_sha256(audits),
    }


def _heldout_evidence(
    readiness=None, cases=None, rows=None, audits=None
) -> dict[str, str]:
    readiness = _readiness() if readiness is None else readiness
    cases = _heldout_cases() if cases is None else cases
    rows = _heldout_rows() if rows is None else rows
    audits = _runtime_audits() if audits is None else audits
    return {
        "campaign_manifest_sha256": holdout.CAMPAIGN_MANIFEST_SHA256,
        "stationary_protocol_sha256": holdout.STATIONARY_PROTOCOL_SHA256,
        "coefficient_fingerprint_sha256": (
            holdout.COEFFICIENT_FINGERPRINT_SHA256
        ),
        "calibration_readiness_content_sha256": holdout.semantic_sha256(
            readiness
        ),
        "selected_cases_content_sha256": holdout.semantic_sha256(cases),
        "response_rows_content_sha256": holdout.semantic_sha256(rows),
        "runtime_audits_content_sha256": holdout.semantic_sha256(audits),
    }


def _calibration_summary(
    *,
    predicted_peak: float = (
        holdout.CALIBRATION_EXPERIMENTAL_PEAK_AMPLITUDE_OVER_DIAMETER
    ),
) -> dict:
    checks = {name: True for name in holdout._CALIBRATION_REQUIRED_CHECKS}
    # These strict external-code checks are deliberately not part of the
    # user-declared engineering-amplitude gate.
    checks.update(
        {
            "external_hgast_peak_evaluated": True,
            "external_hgast_peak_pass": False,
            "external_hgast_curve_evaluated": True,
            "external_hgast_curve_pass": False,
            "external_hgast_band_evaluated": True,
            "external_hgast_band_pass": False,
        }
    )
    return {
        "schema": holdout.BENCHMARK_SUMMARY_SCHEMA,
        "coefficient_fingerprint_sha256": (
            holdout.COEFFICIENT_FINGERPRINT_SHA256
        ),
        "qualification_input_eligible": True,
        "calibration_stage_assessment": {
            "schema": holdout.LEGACY_STRICT_CALIBRATION_SCHEMA,
            "status": "calibration_stage_fail",
            "evaluated": True,
            "pass": False,
            "expected_case_count": 24,
            "manifest_calibration_case_count": 24,
            "completed_calibration_case_count": 24,
            "runtime_calibration_case_count": 24,
            "expected_scruton_number": 5.18,
            "expected_reduced_velocities": list(
                holdout.CALIBRATION_REDUCED_VELOCITIES
            ),
            "expected_timestep_levels": list(holdout.TIMESTEP_LEVELS),
            "checks": checks,
        },
        "lupi_experimental_curve_screen": {
            "evaluated": True,
            "pass": False,
            "response_peak_amplitude_over_diameter": predicted_peak,
            "experimental_peak_amplitude_over_diameter": (
                holdout.CALIBRATION_EXPERIMENTAL_PEAK_AMPLITUDE_OVER_DIAMETER
            ),
            "peak_location_absolute_error_u_star": 0.20,
        },
    }


def _calibration_rows() -> list[dict]:
    rows = []
    for u_star in holdout.CALIBRATION_REDUCED_VELOCITIES:
        for level in holdout.TIMESTEP_LEVELS:
            rows.append(
                {
                    "case_id": holdout._case_id(5.18, u_star, level),
                    "scruton_number": 5.18,
                    "role": "calibration",
                    "reduced_velocity": u_star,
                    "timestep_level": level,
                    "runtime_audit_evaluated": True,
                    "runtime_audit_pass": True,
                    "stationarity_evaluated": True,
                    "stationarity_pass": True,
                    "result_path": (
                        f"/evidence/{holdout._case_id(5.18, u_star, level)}/result.txt"
                    ),
                    "tip_peak_amplitude_over_diameter": (
                        0.001 if u_star in (4.3, 8.2) else 0.235
                    ),
                }
            )
    return rows


@lru_cache(maxsize=1)
def _heldout_case_templates() -> tuple[dict, ...]:
    structural_cases = benchmark.load_structural_cases()
    coefficients = benchmark.calibrate_response_coefficients()
    entries = benchmark.build_campaign_spec(
        structural_cases=structural_cases,
        reduced_velocities=benchmark.DEFAULT_REDUCED_VELOCITIES,
        timestep_s=benchmark.DEFAULT_TIMESTEP_S,
        include_dt_halving=True,
        duration_s=None,
        case_id_namespace=benchmark.AMBIENT_V2_CASE_ID_NAMESPACE,
    )
    node_ids = [
        f"BLD_1_{index / (benchmark.N_STATIONS - 1):.6f}"
        for index in range(benchmark.N_STATIONS)
    ]
    expected_ids = set(holdout.engineering_screen_spec()["exact_case_ids"])
    selected = []
    for entry in entries:
        if entry["case_id"] not in expected_ids:
            continue
        case_id = entry["case_id"]
        sc = float(entry["scruton_number"])
        entry["coefficient_fingerprint_sha256"] = (
            holdout.COEFFICIENT_FINGERPRINT_SHA256
        )
        entry["response_model_config"] = benchmark.response_model_config(
            structural_cases[sc],
            node_ids,
            _LINUX_CANONICAL_MODE_SHAPE,
            coefficients,
            response_schema_version=2,
        )
        entry["parameter_file"] = (
            f"cases/{case_id}/model/Control/vorlap_qblade_external.json"
        )
        entry["sim_path"] = f"cases/{case_id}/{case_id}.sim"
        assert holdout.semantic_sha256(entry) == (
            holdout._HELDOUT_CASE_CONTENT_SHA256[case_id]
        )
        selected.append(copy.deepcopy(entry))
    return tuple(selected)


def _heldout_cases() -> list[dict]:
    return copy.deepcopy(list(_heldout_case_templates()))


def _heldout_rows() -> list[dict]:
    amplitudes = {
        (2.94, 6.25, "dt"): 0.345,
        (2.94, 6.25, "dt_half"): 0.350,
        (2.94, 6.8, "dt"): 0.335,
        (2.94, 6.8, "dt_half"): 0.340,
        (7.02, 6.2, "dt"): 0.145,
        (7.02, 6.2, "dt_half"): 0.150,
        (7.02, 6.25, "dt"): 0.140,
        (7.02, 6.25, "dt_half"): 0.145,
    }
    rows = []
    for case in _heldout_cases():
        key = (
            case["scruton_number"],
            case["reduced_velocity"],
            case["timestep_level"],
        )
        rows.append(
            {
                **case,
                "runtime_audit_evaluated": True,
                "runtime_audit_pass": True,
                "stationarity_evaluated": True,
                "stationarity_pass": True,
                "result_path": f"/evidence/{case['case_id']}/result.txt",
                "tip_peak_amplitude_over_diameter": amplitudes[key],
                "terminal_dominant_frequency_over_fn": 1.0,
            }
        )
    return rows


def _runtime_audit_for_case(
    case_id: str, expected_config_fingerprint: str
) -> dict:
    source_counts = {
        "aliased_frequency_count": 0,
        "aoa_out_of_range_count": 0,
        "force_coefficient_out_of_range_count": 0,
        "frequency_truncated_count": 0,
        "nonfinite_count": 0,
        "re_out_of_range_count": 0,
        "source_clip_count": 0,
    }
    response_counts = {
        "force_limited_node_samples": 0,
        "force_limited_steps": 0,
        "gain_limited_node_samples": 0,
        "gain_limited_steps": 0,
    }
    paths = {
        "history": f"/evidence/{case_id}/result_sil_vorlap_history.csv",
        "result": f"/evidence/{case_id}/result.txt",
        "sim": f"/evidence/{case_id}/{case_id}.sim",
        "turbine": f"/evidence/{case_id}/model/cantilever.trb",
        "vorlap_config": (
            f"/evidence/{case_id}/model/Control/vorlap_qblade_external.json"
        ),
    }
    identities = {
        name: {
            "path": path,
            "sha256": "b" * 64,
            "size_bytes": 1,
        }
        for name, path in paths.items()
    }
    history_path = paths["history"]
    return {
            "case_id": case_id,
            "artifacts_present": True,
            "evaluated": True,
            "pass": True,
            "provenance_evaluated": True,
            "provenance_pass": True,
            "execution_pass": True,
            "identity_mismatch": False,
            "identity_mismatches": [],
            "missing_identity_fields": [],
            "problems": [],
            "expected_response_config_fingerprint_sha256": (
                expected_config_fingerprint
            ),
            "input_identities": identities,
            "summary_path": f"/evidence/{case_id}/{case_id}_run_summary.json",
            "history_path": history_path,
            "source_sample_counts": [100],
            "source_coverage_complete": True,
            "source_limits_pass": True,
            "response_summary_limits_pass": True,
            "response_speed_summary_pass": True,
            "startup_seed_summary_pass": True,
            "startup_seed_response": {
                "evaluated": True,
                "pass": True,
                "export_time_grid_audit": {"pass": True},
            },
            "source_limit_counts": dict(source_counts),
            "response_limit_counts": dict(response_counts),
            "history": {
                "history_mode": "compact",
                "history_schema": "vorlap-sil-force-history-compact-v1",
                "response_model_schema_version": 2,
                "response_model_types": ["modal_negative_damping"],
                "response_enabled_values": [True],
                "aerodynamic_speed_basis": "declared_ambient_crossflow",
                "runtime_model_pass": True,
                "compact_lag_provenance_pass": True,
                "applied_power_sign_convention_pass": True,
                "speed_basis_provenance_pass": True,
                "speed_basis_invariant_violations": 0,
                "activation_invariant_violations": 0,
                "limit_headroom_pass": True,
                "response_estimator_stationarity_evaluated": True,
                "response_estimator_stationarity_pass": True,
                "startup_seed_integrity_evaluated": True,
                "startup_seed_integrity_pass": True,
                "path": history_path,
                "file_identity": identities["history"],
            },
        }


def _runtime_audits() -> list[dict]:
    audits = []
    for case in _heldout_cases():
        sc = float(case["scruton_number"])
        audits.append(
            _runtime_audit_for_case(
                case["case_id"],
                holdout._RESPONSE_CONFIG_FINGERPRINT_BY_SCRUTON[sc],
            )
        )
    return audits


def _calibration_runtime_audits() -> list[dict]:
    return [
        _runtime_audit_for_case(
            row["case_id"],
            holdout._RESPONSE_CONFIG_FINGERPRINT_BY_SCRUTON[5.18],
        )
        for row in _calibration_rows()
    ]


def _readiness() -> dict:
    summary = _calibration_summary()
    rows = _calibration_rows()
    audits = _calibration_runtime_audits()
    return holdout.assess_calibration_readiness(
        summary, rows, audits, _calibration_evidence(summary, rows, audits)
    )


def _assess_heldout(
    *, readiness=None, cases=None, rows=None, audits=None, evidence=None
):
    readiness = _readiness() if readiness is None else readiness
    cases = _heldout_cases() if cases is None else cases
    rows = _heldout_rows() if rows is None else rows
    audits = _runtime_audits() if audits is None else audits
    evidence = (
        _heldout_evidence(readiness, cases, rows, audits)
        if evidence is None
        else evidence
    )
    return holdout.assess_heldout_results(
        readiness, cases, rows, audits, evidence
    )


def test_spec_freezes_exact_cases_tolerances_and_claim_boundary():
    spec = holdout.engineering_screen_spec()
    assert spec["schema"] == holdout.SCREEN_SPEC_SCHEMA
    assert len(spec["exact_case_ids"]) == 8
    assert len(set(spec["exact_case_ids"])) == 8
    assert spec["acceptance"]["peak_relative_tolerance"] == 0.30
    assert spec["acceptance"]["peak_digitization_allowance_over_diameter"] == 0.01
    assert "blind independent validation" in spec["claim_boundary"]["forbidden"]
    assert len(spec["specification_fingerprint_sha256"]) == 64

    spec["exact_case_ids"].clear()
    assert len(holdout.engineering_screen_spec()["exact_case_ids"]) == 8


def test_selector_requires_exact_case_semantics_and_is_order_deterministic():
    cases = list(reversed(_heldout_cases()))
    selected = holdout.select_heldout_cases(cases)
    assert [row["case_id"] for row in selected] == holdout.engineering_screen_spec()[
        "exact_case_ids"
    ]

    missing = _heldout_cases()[1:]
    with pytest.raises(ValueError, match="missing"):
        holdout.select_heldout_cases(missing)

    duplicate = _heldout_cases() + [_heldout_cases()[0]]
    with pytest.raises(ValueError, match="duplicate"):
        holdout.select_heldout_cases(duplicate)

    changed = _heldout_cases()
    changed[0] = {**changed[0], "role": "calibration"}
    with pytest.raises(ValueError, match="semantics differ"):
        holdout.select_heldout_cases(changed)


def test_calibration_readiness_passes_without_rewriting_strict_hgast_failure():
    result = _readiness()
    assert result["evaluated"] is True
    assert result["pass"] is True
    assert result["status"] == "pass"
    assert result["strict_hgast_stage_status_unchanged"] == "calibration_stage_fail"
    assert result["checks"]["experimental_peak_amplitude_pass"] is True
    assert len(result["expected_case_ids"]) == 24


def test_calibration_readiness_preserves_and_validates_strict_hgast_pass():
    summary = _calibration_summary()
    strict = summary["calibration_stage_assessment"]
    strict["status"] = "calibration_stage_pass"
    strict["pass"] = True
    for name in holdout._CALIBRATION_EXTERNAL_PASS_CHECKS:
        strict["checks"][name] = True
    rows = _calibration_rows()
    audits = _calibration_runtime_audits()
    readiness = holdout.assess_calibration_readiness(
        summary,
        rows,
        audits,
        _calibration_evidence(summary, rows, audits),
    )
    assert readiness["pass"] is True
    assert (
        readiness["strict_hgast_stage_status_unchanged"]
        == "calibration_stage_pass"
    )
    fingerprint = holdout._validate_calibration_readiness_for_heldout(
        readiness,
        {
            "campaign_manifest_sha256": holdout.CAMPAIGN_MANIFEST_SHA256,
            "stationary_protocol_sha256": holdout.STATIONARY_PROTOCOL_SHA256,
            "coefficient_fingerprint_sha256": (
                holdout.COEFFICIENT_FINGERPRINT_SHA256
            ),
        },
    )
    assert fingerprint == readiness["assessment_fingerprint_sha256"]


def test_calibration_peak_tolerance_boundary_is_inclusive_and_exceedance_fails():
    target = holdout.CALIBRATION_EXPERIMENTAL_PEAK_AMPLITUDE_OVER_DIAMETER
    tolerance = 0.30 * target + 0.01
    rows = _calibration_rows()
    audits = _calibration_runtime_audits()
    summary = _calibration_summary(predicted_peak=target + tolerance)
    boundary = holdout.assess_calibration_readiness(
        summary,
        rows,
        audits,
        _calibration_evidence(summary, rows, audits),
    )
    assert boundary["pass"] is True

    summary = _calibration_summary(predicted_peak=target + tolerance + 1.0e-8)
    failed = holdout.assess_calibration_readiness(
        summary,
        rows,
        audits,
        _calibration_evidence(summary, rows, audits),
    )
    assert failed["evaluated"] is True
    assert failed["pass"] is False
    assert failed["checks"]["experimental_peak_amplitude_pass"] is False


def test_calibration_readiness_rejects_not_evaluated_strict_parent():
    summary = _calibration_summary()
    summary["calibration_stage_assessment"]["evaluated"] = False
    summary["calibration_stage_assessment"]["status"] = "not_evaluated"
    for name in (
        "external_hgast_peak_evaluated",
        "external_hgast_curve_evaluated",
        "external_hgast_band_evaluated",
    ):
        summary["calibration_stage_assessment"]["checks"][name] = False
    rows = _calibration_rows()
    audits = _calibration_runtime_audits()
    with pytest.raises(ValueError, match="matrix is incomplete or changed"):
        holdout.assess_calibration_readiness(
            summary,
            rows,
            audits,
            _calibration_evidence(summary, rows, audits),
        )

def test_calibration_readiness_rejects_negative_peak_location_error():
    summary = _calibration_summary()
    summary["lupi_experimental_curve_screen"][
        "peak_location_absolute_error_u_star"
    ] = -100.0
    rows = _calibration_rows()
    audits = _calibration_runtime_audits()
    with pytest.raises(ValueError, match="must be nonnegative"):
        holdout.assess_calibration_readiness(
            summary,
            rows,
            audits,
            _calibration_evidence(summary, rows, audits),
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("manifest", "differs from the frozen campaign"),
        ("extra_evidence", "exact frozen key schema"),
        ("missing_row", "exact 24-case matrix"),
        ("runtime", ""),
        ("off_resonance", ""),
    ],
)
def test_calibration_readiness_fails_closed(mutation, match):
    summary = _calibration_summary()
    rows = _calibration_rows()
    audits = _calibration_runtime_audits()
    if mutation == "missing_row":
        rows.pop()
    elif mutation == "runtime":
        rows[0]["runtime_audit_pass"] = False
    elif mutation == "off_resonance":
        endpoint = next(
            row
            for row in rows
            if row["reduced_velocity"] == 4.3 and row["timestep_level"] == "dt_half"
        )
        endpoint["tip_peak_amplitude_over_diameter"] = 0.0500001

    evidence = _calibration_evidence(summary, rows, audits)
    if mutation == "manifest":
        evidence["campaign_manifest_sha256"] = "0" * 64
    elif mutation == "extra_evidence":
        evidence["extra"] = "0" * 64

    if match:
        with pytest.raises(ValueError, match=match):
            holdout.assess_calibration_readiness(summary, rows, audits, evidence)
    else:
        result = holdout.assess_calibration_readiness(
            summary, rows, audits, evidence
        )
        assert result["pass"] is False


def test_heldout_assessment_passes_exact_eight_case_screen():
    result = _assess_heldout()
    assert result["evaluated"] is True
    assert result["pass"] is True
    assert result["status"] == "pass"
    assert len(result["exact_case_ids"]) == 8
    assert all(row["pass"] for row in result["by_scruton_number"])
    assert result["strict_hgast_gate_rewritten"] is False


def test_heldout_assessment_rejects_co_resident_calibration_rows():
    rows = _heldout_rows() + _calibration_rows()
    audits = _runtime_audits() + _calibration_runtime_audits()
    with pytest.raises(ValueError, match="exactly the eight selected cases"):
        _assess_heldout(rows=rows, audits=audits)


def test_heldout_assessment_rejects_extra_selected_case_even_when_rehashed():
    cases = _heldout_cases()
    extra = copy.deepcopy(cases[0])
    extra["case_id"] = "ambientv2_sc5p18_u4p3_dt"
    extra["role"] = "calibration"
    cases.append(extra)
    with pytest.raises(ValueError, match="exactly the eight frozen cases"):
        _assess_heldout(cases=cases)


def test_strict_parent_status_and_pass_must_be_consistent():
    summary = _calibration_summary()
    summary["calibration_stage_assessment"]["status"] = "calibration_stage_pass"
    rows = _calibration_rows()
    audits = _calibration_runtime_audits()
    with pytest.raises(ValueError, match="matrix is incomplete or changed"):
        holdout.assess_calibration_readiness(
            summary,
            rows,
            audits,
            _calibration_evidence(summary, rows, audits),
        )

    summary = _calibration_summary()
    summary["calibration_stage_assessment"]["status"] = "calibration_stage_pass"
    summary["calibration_stage_assessment"]["pass"] = True
    with pytest.raises(ValueError, match="status/pass/checks are inconsistent"):
        holdout.assess_calibration_readiness(
            summary,
            rows,
            audits,
            _calibration_evidence(summary, rows, audits),
        )


@pytest.mark.parametrize(
    "field",
    [
        "expected_case_count",
        "manifest_calibration_case_count",
        "completed_calibration_case_count",
        "runtime_calibration_case_count",
    ],
)
def test_strict_parent_case_counts_require_exact_integers(field):
    summary = _calibration_summary()
    summary["calibration_stage_assessment"][field] = 24.0
    rows = _calibration_rows()
    audits = _calibration_runtime_audits()
    with pytest.raises(ValueError, match="matrix is incomplete or changed"):
        holdout.assess_calibration_readiness(
            summary,
            rows,
            audits,
            _calibration_evidence(summary, rows, audits),
        )


@pytest.mark.parametrize("mutation", ["coefficient", "response_config"])
def test_selector_authenticates_frozen_manifest_model_contract(mutation):
    cases = _heldout_cases()
    if mutation == "coefficient":
        cases[0]["coefficient_fingerprint_sha256"] = "0" * 64
    else:
        cases[0]["response_model_config"]["enabled"] = False
    with pytest.raises(ValueError, match="manifest/config contract differs"):
        holdout.select_heldout_cases(cases)


def test_runtime_audit_case_paths_and_config_are_bound_to_selected_case():
    audits = _runtime_audits()
    audits[0]["case_id"], audits[1]["case_id"] = (
        audits[1]["case_id"],
        audits[0]["case_id"],
    )
    result = _assess_heldout(audits=audits)
    assert result["pass"] is False
    assert result["checks"]["runtime_provenance_execution_limits_pass"] is False


def test_runtime_audit_rejects_traversal_and_ambiguous_case_paths():
    audits = _runtime_audits()
    row = audits[0]
    case_id = row["case_id"]
    for identity in row["input_identities"].values():
        name = PurePosixPath(identity["path"]).name
        identity["path"] = f"/evidence/{case_id}/../other/{name}"
    row["summary_path"] = (
        f"/evidence/{case_id}/../other/{case_id}_run_summary.json"
    )
    row["history_path"] = (
        f"/evidence/{case_id}/../other/result_sil_vorlap_history.csv"
    )
    row["history"]["path"] = row["history_path"]
    row["history"]["file_identity"] = row["input_identities"]["history"]
    result = _assess_heldout(audits=audits)
    assert result["pass"] is False
    assert result["checks"]["runtime_provenance_execution_limits_pass"] is False

    audits = _runtime_audits()
    row = audits[0]
    for identity in row["input_identities"].values():
        identity["path"] = "/" + identity["path"]
    row["summary_path"] = "/" + row["summary_path"]
    row["history_path"] = "/" + row["history_path"]
    row["history"]["path"] = row["history_path"]
    row["history"]["file_identity"] = row["input_identities"]["history"]
    result = _assess_heldout(audits=audits)
    assert result["pass"] is False
    assert result["checks"]["runtime_provenance_execution_limits_pass"] is False

    audits = _runtime_audits()
    audits[0]["expected_response_config_fingerprint_sha256"] = "0" * 64
    result = _assess_heldout(audits=audits)
    assert result["pass"] is False
    assert result["checks"]["runtime_provenance_execution_limits_pass"] is False


def test_assessments_accept_exact_csv_boolean_tokens():
    calibration_rows = _calibration_rows()
    heldout_rows = _heldout_rows()
    for row in calibration_rows + heldout_rows:
        for key in (
            "runtime_audit_evaluated",
            "runtime_audit_pass",
            "stationarity_evaluated",
            "stationarity_pass",
        ):
            row[key] = "True"
    summary = _calibration_summary()
    calibration_audits = _calibration_runtime_audits()
    readiness = holdout.assess_calibration_readiness(
        summary,
        calibration_rows,
        calibration_audits,
        _calibration_evidence(summary, calibration_rows, calibration_audits),
    )
    result = _assess_heldout(readiness=readiness, rows=heldout_rows)
    assert result["pass"] is True


def test_heldout_peak_tolerance_and_dt_frequency_boundaries():
    rows = _heldout_rows()
    sc2_fine = next(
        row
        for row in rows
        if row["scruton_number"] == 2.94
        and row["reduced_velocity"] == 6.25
        and row["timestep_level"] == "dt_half"
    )
    sc2_coarse = next(
        row
        for row in rows
        if row["scruton_number"] == 2.94
        and row["reduced_velocity"] == 6.25
        and row["timestep_level"] == "dt"
    )
    tolerance = 0.30 * 0.350 + 0.010
    sc2_fine["tip_peak_amplitude_over_diameter"] = 0.350 + tolerance
    sc2_coarse["tip_peak_amplitude_over_diameter"] = sc2_fine[
        "tip_peak_amplitude_over_diameter"
    ]
    sc2_fine["terminal_dominant_frequency_over_fn"] = 1.05
    sc2_coarse["terminal_dominant_frequency_over_fn"] = 1.00
    boundary = _assess_heldout(rows=rows)
    assert boundary["pass"] is True

    sc2_fine["tip_peak_amplitude_over_diameter"] += 1.0e-8
    sc2_coarse["tip_peak_amplitude_over_diameter"] += 1.0e-8
    failed = _assess_heldout(rows=rows)
    assert failed["pass"] is False
    assert failed["by_scruton_number"][0]["peak_amplitude_pass"] is False


def test_inactive_point_does_not_require_noise_frequency_convergence():
    rows = _heldout_rows()
    for row in rows:
        if row["scruton_number"] == 2.94 and row["reduced_velocity"] == 6.8:
            row["tip_peak_amplitude_over_diameter"] = 0.001
            row["terminal_dominant_frequency_over_fn"] = None
    result = _assess_heldout(rows=rows)
    assert result["pass"] is True
    sc2 = result["by_scruton_number"][0]
    inactive = next(
        row for row in sc2["timestep_pairs"] if row["reduced_velocity"] == 6.8
    )
    assert inactive["active"] is False
    assert inactive["frequency_pass"] is True
    assert inactive["half_dt_frequency_over_fn"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_history",
        "fractional_limit_count",
        "negative_limit_count",
        "problems",
        "missing_identity_fields",
        "wrong_input_hash",
        "runtime_model",
        "lag",
        "sign",
        "speed",
        "headroom",
        "boolean_speed_violation_count",
        "boolean_activation_violation_count",
        "floating_schema_version",
        "result_path_mismatch",
    ],
)
def test_runtime_audit_nested_contract_fails_closed(mutation):
    audits = _runtime_audits()
    row = audits[0]
    if mutation == "missing_history":
        row.pop("history")
    elif mutation == "fractional_limit_count":
        row["source_limit_counts"]["source_clip_count"] = 0.9
    elif mutation == "negative_limit_count":
        row["response_limit_counts"]["force_limited_steps"] = -0.9
    elif mutation == "problems":
        row["problems"] = ["injected"]
    elif mutation == "missing_identity_fields":
        row["missing_identity_fields"] = ["history"]
    elif mutation == "wrong_input_hash":
        row["input_identities"]["history"]["sha256"] = "not-a-hash"
    elif mutation == "runtime_model":
        row["history"]["runtime_model_pass"] = False
    elif mutation == "lag":
        row["history"]["compact_lag_provenance_pass"] = False
    elif mutation == "sign":
        row["history"]["applied_power_sign_convention_pass"] = False
    elif mutation == "speed":
        row["history"]["speed_basis_provenance_pass"] = False
    elif mutation == "headroom":
        row["history"]["limit_headroom_pass"] = False
    elif mutation == "boolean_speed_violation_count":
        row["history"]["speed_basis_invariant_violations"] = False
    elif mutation == "boolean_activation_violation_count":
        row["history"]["activation_invariant_violations"] = False
    elif mutation == "floating_schema_version":
        row["history"]["response_model_schema_version"] = 2.0
    elif mutation == "result_path_mismatch":
        row["input_identities"]["result"]["path"] = (
            f"/evidence/{row['case_id']}/other.txt"
        )
    result = _assess_heldout(audits=audits)
    assert result["pass"] is False
    assert result["checks"]["runtime_provenance_execution_limits_pass"] is False


def test_payload_hashes_and_selected_case_payload_are_bound():
    readiness = _readiness()
    cases = _heldout_cases()
    rows = _heldout_rows()
    audits = _runtime_audits()
    evidence = _heldout_evidence(readiness, cases, rows, audits)
    evidence["calibration_readiness_content_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not match the assessed payload"):
        holdout.assess_heldout_results(
            readiness, cases, rows, audits, evidence
        )

    evidence = _heldout_evidence(readiness, cases, rows, audits)
    cases[0] = {**cases[0], "role": "calibration"}
    with pytest.raises(ValueError, match="selected_cases_content_sha256"):
        holdout.assess_heldout_results(
            readiness, cases, rows, audits, evidence
        )


def test_heldout_recomputes_calibration_readiness_status_and_checks():
    readiness = _readiness()
    readiness["checks"]["runtime_audit_payload_integrity_pass"] = False
    readiness_without_fingerprint = dict(readiness)
    readiness_without_fingerprint.pop("assessment_fingerprint_sha256")
    readiness["assessment_fingerprint_sha256"] = holdout.semantic_sha256(
        readiness_without_fingerprint
    )
    with pytest.raises(ValueError, match="passing engineering"):
        _assess_heldout(readiness=readiness)

    readiness = _readiness()
    readiness["extra"] = True
    readiness_without_fingerprint = dict(readiness)
    readiness_without_fingerprint.pop("assessment_fingerprint_sha256")
    readiness["assessment_fingerprint_sha256"] = holdout.semantic_sha256(
        readiness_without_fingerprint
    )
    with pytest.raises(ValueError, match="passing engineering"):
        _assess_heldout(readiness=readiness)


@pytest.mark.parametrize(
    "field,value,match",
    [
        (
            "strict_hgast_stage_status_unchanged",
            "not_evaluated",
            "strict hGAST status changed",
        ),
        (
            "experimental_peak_amplitude_over_diameter",
            0.5,
            "peak fields are inconsistent",
        ),
        (
            "response_peak_amplitude_over_diameter",
            0.1,
            "peak fields are inconsistent",
        ),
        (
            "peak_amplitude_absolute_error",
            0.01,
            "peak fields are inconsistent",
        ),
        (
            "peak_amplitude_tolerance",
            -1.0,
            "must be nonnegative",
        ),
        (
            "peak_location_tolerance_u_star",
            999.0,
            "peak-location fields are inconsistent",
        ),
        (
            "off_resonance_half_dt_amplitudes_over_diameter",
            [0.0, 0.051],
            "off-resonance guard is inconsistent",
        ),
    ],
)
def test_heldout_recomputes_all_readiness_derived_fields(field, value, match):
    readiness = _readiness()
    readiness[field] = value
    readiness_without_fingerprint = dict(readiness)
    readiness_without_fingerprint.pop("assessment_fingerprint_sha256")
    readiness["assessment_fingerprint_sha256"] = holdout.semantic_sha256(
        readiness_without_fingerprint
    )
    with pytest.raises(ValueError, match=match):
        _assess_heldout(readiness=readiness)


def test_runtime_history_response_enabled_values_requires_exact_bool():
    audits = _runtime_audits()
    audits[0]["history"]["response_enabled_values"] = [1]
    result = _assess_heldout(audits=audits)
    assert result["pass"] is False
    assert result["checks"]["runtime_provenance_execution_limits_pass"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("readiness_pass", "passing engineering"),
        ("readiness_fingerprint", "fingerprint is invalid"),
        ("coefficient", "differs from the frozen campaign"),
        ("missing_response", "exactly the eight"),
        ("extra_runtime", "invalid or duplicate"),
        ("runtime_limit", ""),
        ("stationarity", ""),
    ],
)
def test_heldout_assessment_fails_closed(mutation, match):
    readiness = _readiness()
    cases = _heldout_cases()
    rows = _heldout_rows()
    audits = _runtime_audits()
    if mutation == "readiness_pass":
        readiness["pass"] = False
    elif mutation == "readiness_fingerprint":
        readiness["scope"] = "tampered without updating the public fingerprint"
    elif mutation == "missing_response":
        rows.pop()
    elif mutation == "extra_runtime":
        audits.append(copy.deepcopy(audits[0]))
    elif mutation == "runtime_limit":
        audits[0]["response_limit_counts"]["force_limited_steps"] = 1
    elif mutation == "stationarity":
        rows[0]["stationarity_pass"] = False

    evidence = _heldout_evidence(readiness, cases, rows, audits)
    if mutation == "coefficient":
        evidence["coefficient_fingerprint_sha256"] = "0" * 64

    if match:
        with pytest.raises(ValueError, match=match):
            holdout.assess_heldout_results(
                readiness, cases, rows, audits, evidence
            )
    else:
        result = holdout.assess_heldout_results(
            readiness, cases, rows, audits, evidence
        )
        assert result["pass"] is False
