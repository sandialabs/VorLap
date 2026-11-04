import json
import math
from copy import deepcopy

import pytest

from examples.ehrmann2014_naca0018 import build_validation_evidence as evidence


def test_digitized_ehrmann_series_have_frozen_shape_and_peaks():
    figure6, figure7, provenance = evidence._reference_rows()
    assert len(figure6) == 26
    assert len(figure7) == 5
    peak = max(figure6, key=lambda row: row["rms_acceleration_mps2"])
    assert peak["reduced_velocity"] == pytest.approx(4.586149)
    assert peak["rms_acceleration_mps2"] == pytest.approx(1.851351)
    assert figure7[0]["maximum_amplitude_over_chord"] == pytest.approx(0.01236635)
    assert provenance["digitization"]["figure_6"]["condition_boundary"].startswith(
        "The publisher raster identifies"
    )


def test_peak_mapping_never_extrapolates_beyond_digitized_mass_damping():
    _, figure7, _ = evidence._reference_rows()
    low, low_policy = evidence._experiment_peak_at_mass_damping(figure7, 0.96)
    high, high_policy = evidence._experiment_peak_at_mass_damping(figure7, 1.44)
    assert low == pytest.approx(0.01236635)
    assert low_policy == "nearest_digitized_lower_boundary_no_extrapolation"
    assert high == pytest.approx(0.0068166, rel=2e-5)
    assert high_policy == "linear_interpolation_within_digitized_markers"


def test_reference_provenance_binds_primary_publisher_rasters():
    payload = json.loads(evidence.PROVENANCE_JSON.read_text())
    assert payload["citation"]["doi"] == "10.1016/j.jfluidstructs.2013.10.008"
    assert payload["digitization"]["figure_6"]["publisher_image_sha256"] == (
        "4927964bf9511b4ac37ab159a6b9dd20e102d280778105e2d091c9728d564a89"
    )
    assert payload["digitization"]["figure_7"]["publisher_image_sha256"] == (
        "052e0cee4d7b701e14770c3265c3d9ea4ce4103bafcf60370e484ea2f3daad2a"
    )


def test_promoted_evidence_records_completed_negative_validation():
    path = evidence.DEFAULT_EVIDENCE_ROOT / "ehrmann2014_validation_metrics.json"
    payload = json.loads(path.read_text())
    assert payload["status"] == "completed_validation_failed_no_sustained_lockin"
    assert payload["completed_cases"] == 40
    assert payload["primary_acceptance"]["pass"] is False
    assert payload["maximum_paired_timestep_amplitude_difference_over_chord"] < 1e-9
    assert all(item["lockin_detected"] is False for item in payload["comparisons"])
    assert all(
        item["lockin_peak_location_absolute_error_u_star"] is None
        for item in payload["comparisons"]
    )
    seed = payload["representative_seed_decay"]
    assert seed["maximum_generated_abs_force_component_n"] > 0.1
    assert seed["maximum_spectral_energy_closure_error"] < 1e-12
    assert seed["terminal_estimated_amplitude_over_chord"] < 1e-8


def _work_map_rows() -> list[dict[str, object]]:
    hybrid_amplitudes = {
        0.96: {
            3.0: 0.00001,
            3.5: 0.00005,
            4.0: 0.003,
            4.5: 0.008,
            5.0: 0.012,
            5.5: 0.010,
            5.8: 0.008,
            6.0: 0.001,
            6.3: 0.00005,
        },
        1.44: {
            3.0: 0.00001,
            3.5: 0.00005,
            4.0: 0.002,
            4.5: 0.007,
            5.0: 0.005,
            5.5: 0.002,
            5.8: 0.001,
            6.0: 0.00005,
            6.3: 0.00001,
        },
    }
    hybrid_ratios = {
        3.0: 0.60,
        3.5: 0.70,
        4.0: 0.98,
        4.5: 1.00,
        5.0: 1.02,
        5.5: 1.01,
        5.8: 1.01,
        6.0: 0.75,
        6.3: 0.65,
    }
    rows: list[dict[str, object]] = []
    index = 0
    for md in evidence.campaign.MASS_DAMPINGS:
        for u_star in evidence.campaign.REDUCED_VELOCITIES:
            for scenario in evidence.campaign.WORK_MAP_SCENARIOS:
                hybrid = scenario == "latest_airfoil_work_map_bridge"
                amplitude = hybrid_amplitudes[md][u_star] if hybrid else 1.0e-6
                ratio = hybrid_ratios[u_star] if hybrid else None
                rows.append(
                    {
                        "case_id": f"case_{index:02d}",
                        "scenario": scenario,
                        "mass_damping": md,
                        "structural_damping_ratio": md / 47.7,
                        "reduced_velocity": u_star,
                        "reynolds_number": 20_000.0 * u_star,
                        "timestep_s": 0.001,
                        "terminal_amplitude_m": amplitude * evidence.CHORD_M,
                        "terminal_amplitude_over_chord": amplitude,
                        "terminal_dominant_frequency_hz": (
                            None
                            if ratio is None
                            else ratio * evidence.NOMINAL_FREQUENCY_HZ
                        ),
                        "terminal_dominant_frequency_over_fn": ratio,
                        "derived_terminal_rms_acceleration_mps2": (
                            0.0
                            if ratio is None
                            else (2.0 * math.pi * ratio * evidence.NOMINAL_FREQUENCY_HZ)
                            ** 2
                            * amplitude
                            * evidence.CHORD_M
                            / math.sqrt(2.0)
                        ),
                        "stationarity_amplitude_pass": True,
                    }
                )
                index += 1
        for scenario in evidence.campaign.WORK_MAP_SCENARIOS:
            hybrid = scenario == "latest_airfoil_work_map_bridge"
            nominal = hybrid_amplitudes[md][5.8] if hybrid else 1.0e-6
            ratio = hybrid_ratios[5.8] + 0.005 if hybrid else None
            amplitude = nominal + (0.0002 if hybrid else 0.0)
            rows.append(
                {
                    "case_id": f"case_{index:02d}",
                    "scenario": scenario,
                    "mass_damping": md,
                    "structural_damping_ratio": md / 47.7,
                    "reduced_velocity": 5.8,
                    "reynolds_number": 116_000.0,
                    "timestep_s": 0.0005,
                    "terminal_amplitude_m": amplitude * evidence.CHORD_M,
                    "terminal_amplitude_over_chord": amplitude,
                    "terminal_dominant_frequency_hz": (
                        None if ratio is None else ratio * evidence.NOMINAL_FREQUENCY_HZ
                    ),
                    "terminal_dominant_frequency_over_fn": ratio,
                    "derived_terminal_rms_acceleration_mps2": 0.0,
                    "stationarity_amplitude_pass": True,
                }
            )
            index += 1
    assert len(rows) == 40
    return rows


def _evaluate_work_map(
    rows: list[dict[str, object]],
    *,
    runtime_pass: bool = True,
    provenance_pass: bool = True,
) -> dict[str, object]:
    figure6, figure7, _ = evidence._reference_rows()
    return evidence._work_map_metrics(
        rows,
        figure6,
        figure7,
        {
            "schema": evidence.campaign.WORK_MAP_SCHEMA,
            "fingerprint": "frozen-work-map-campaign",
        },
        {
            "pass": runtime_pass,
            "clean_completed_case_count": 40,
        },
        {
            "verified": provenance_pass,
            "ehrmann_free_response_used_for_fit": False,
            "ehrmann_acceptance_tolerance_used_for_fit": False,
        },
    )


def test_work_map_gate_passes_only_complete_frozen_response_contract():
    metrics = _evaluate_work_map(_work_map_rows())
    assert metrics["schema"] == evidence.WORK_MAP_EVIDENCE_SCHEMA
    assert metrics["status"] == "completed_validation_passed_frozen_acceptance"
    assert metrics["primary_acceptance"]["pass"] is True
    assert metrics["primary_acceptance"]["failed_checks"] == []
    assert metrics["comparison_series"] == [
        "Ehrmann NACA 0018 experiment (digitized publisher markers)",
        "Raw QBlade native aerodynamics",
        "Latest VorLap-QBlade airfoil work-map bridge",
    ]
    hybrid = [
        row
        for row in metrics["comparisons"]
        if row["scenario"] == "latest_airfoil_work_map_bridge"
    ]
    assert len(hybrid) == 2
    assert all(row["peak_amplitude_pass"] for row in hybrid)
    assert all(row["peak_frequency_pass"] for row in hybrid)
    assert all(row["lockin"]["detected"] for row in hybrid)
    assert all(
        row["lockin"]["reduced_velocity_interval"] == [4.0, 5.8] for row in hybrid
    )
    raw_pairs = [
        row for row in metrics["timestep_pairs"] if row["scenario"] == "raw_qblade"
    ]
    assert all(row["frequency_repeatability_evaluated"] is False for row in raw_pairs)
    assert all(row["pair_acceptance_pass"] is True for row in raw_pairs)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            "finite_lockin_interval",
            "hybrid_finite_lockin_interval_at_both_mass_dampings",
        ),
        (
            "peak_amplitude",
            "hybrid_peak_amplitude_within_tolerance_at_both_mass_dampings",
        ),
        (
            "peak_frequency",
            "hybrid_peak_frequency_within_ten_percent_at_both_mass_dampings",
        ),
        ("stationarity", "all_results_stationary"),
        ("timestep_amplitude", "all_paired_timestep_checks_repeatable"),
        ("timestep_frequency", "all_paired_timestep_checks_repeatable"),
    ],
)
def test_work_map_gate_fails_each_frozen_scientific_check(mutation, failed_check):
    rows = deepcopy(_work_map_rows())
    if mutation == "finite_lockin_interval":
        for row in rows:
            if (
                row["mass_damping"] == 0.96
                and row["scenario"] == "latest_airfoil_work_map_bridge"
                and row["timestep_s"] == 0.001
                and row["reduced_velocity"] != 5.0
            ):
                row["terminal_dominant_frequency_over_fn"] = 0.5
                row["terminal_dominant_frequency_hz"] = 3.0
    elif mutation == "peak_amplitude":
        for row in rows:
            if (
                row["mass_damping"] == 0.96
                and row["scenario"] == "latest_airfoil_work_map_bridge"
                and row["timestep_s"] == 0.001
            ):
                row["terminal_amplitude_over_chord"] = 0.05
    elif mutation == "peak_frequency":
        peak = next(
            row
            for row in rows
            if row["mass_damping"] == 0.96
            and row["scenario"] == "latest_airfoil_work_map_bridge"
            and row["reduced_velocity"] == 5.0
            and row["timestep_s"] == 0.001
        )
        peak["terminal_dominant_frequency_over_fn"] = 1.11
        peak["terminal_dominant_frequency_hz"] = 6.66
    elif mutation == "stationarity":
        rows[0]["stationarity_amplitude_pass"] = False
    elif mutation == "timestep_amplitude":
        fine = next(
            row
            for row in rows
            if row["mass_damping"] == 0.96
            and row["scenario"] == "latest_airfoil_work_map_bridge"
            and row["timestep_s"] == 0.0005
        )
        fine["terminal_amplitude_over_chord"] = 0.05
    elif mutation == "timestep_frequency":
        fine = next(
            row
            for row in rows
            if row["mass_damping"] == 0.96
            and row["scenario"] == "latest_airfoil_work_map_bridge"
            and row["timestep_s"] == 0.0005
        )
        fine["terminal_dominant_frequency_over_fn"] = 1.25
        fine["terminal_dominant_frequency_hz"] = 7.5
    metrics = _evaluate_work_map(rows)
    assert metrics["status"] == "completed_validation_failed_frozen_acceptance"
    assert metrics["primary_acceptance"]["pass"] is False
    assert failed_check in metrics["primary_acceptance"]["failed_checks"]


@pytest.mark.parametrize(
    ("runtime_pass", "provenance_pass", "failed_check"),
    [
        (False, True, "runtime_integrity"),
        (True, False, "work_map_frozen_before_ehrmann_evaluation"),
    ],
)
def test_work_map_gate_fails_runtime_or_nonleakage_provenance(
    runtime_pass, provenance_pass, failed_check
):
    metrics = _evaluate_work_map(
        _work_map_rows(),
        runtime_pass=runtime_pass,
        provenance_pass=provenance_pass,
    )
    assert metrics["primary_acceptance"]["pass"] is False
    assert failed_check in metrics["primary_acceptance"]["failed_checks"]


@pytest.mark.parametrize("mutation", ["nineteen_raw", "legacy_scheduler"])
def test_work_map_acceptance_requires_dedicated_twenty_raw_execution_contract(
    mutation,
):
    runtime_identity = {
        "version": evidence.campaign.FROZEN_QBLADE_VERSION,
        "library_sha256": evidence.campaign.FROZEN_QBLADE_LIBRARY_SHA256,
        "runtime_dependency_fingerprint": (
            evidence.campaign.FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
        ),
    }
    manifest = {
        "schema": evidence.campaign.WORK_MAP_SCHEMA,
        "fingerprint": "campaign-fingerprint",
        "work_map_identity": {"sha256": "a" * 64},
        "qblade_runtime_identity": runtime_identity,
    }
    scheduler = {
        "schema": "vorlap-ehrmann2014-flight-scheduler-v1",
        "status": "complete",
        "campaign_fingerprint": manifest["fingerprint"],
        "execution_contract": {
            "schema": "vorlap-ehrmann2014-work-map-flight-execution-v1",
            "campaign_schema": evidence.campaign.WORK_MAP_SCHEMA,
            "work_map_sha256": "a" * 64,
            "exact_raw_reuse_case_count": 20,
            "hybrid_case_count": 20,
            "qblade_runtime_identity": runtime_identity,
        },
        "final_execution_audit": {
            "exact_raw_reuse_case_count": 20,
            "complete_hybrid_case_count": 20,
            "runnable_case_count": 0,
        },
    }
    if mutation == "nineteen_raw":
        scheduler["execution_contract"]["exact_raw_reuse_case_count"] = 19
    else:
        scheduler["execution_contract"]["schema"] = "legacy-flight-contract"
    with pytest.raises(evidence.EvidenceError, match="20-reused-raw"):
        evidence._validate_work_map_execution_contract(scheduler, manifest)


def test_work_map_gate_requires_raw_frequency_repeatability_when_resolved():
    rows = _work_map_rows()
    fine_raw = next(
        row
        for row in rows
        if row["mass_damping"] == 0.96
        and row["scenario"] == "raw_qblade"
        and row["timestep_s"] == 0.0005
    )
    fine_raw["terminal_dominant_frequency_over_fn"] = 1.0
    fine_raw["terminal_dominant_frequency_hz"] = 6.0
    metrics = _evaluate_work_map(rows)
    pair = next(
        row
        for row in metrics["timestep_pairs"]
        if row["mass_damping"] == 0.96 and row["scenario"] == "raw_qblade"
    )
    assert pair["frequency_repeatability_evaluated"] is False
    assert pair["frequency_contract_pass"] is False
    assert pair["pair_acceptance_pass"] is False
    assert metrics["primary_acceptance"]["pass"] is False


def test_work_map_gate_rejects_incomplete_or_nonfinite_matrix():
    with pytest.raises(evidence.EvidenceError, match="40 rows"):
        _evaluate_work_map(_work_map_rows()[:-1])
    rows = _work_map_rows()
    rows[0]["terminal_amplitude_over_chord"] = float("nan")
    with pytest.raises(evidence.EvidenceError, match="finite and non-negative"):
        _evaluate_work_map(rows)


def test_work_map_source_provenance_excludes_ehrmann_fit_inputs(tmp_path):
    case_root = tmp_path / "cases" / "hybrid"
    case_root.mkdir(parents=True)
    request_sha = evidence._sha256(evidence.WORK_MAP_DATA_REQUEST)
    sources = [
        {"role": role, "label": role, "sha256": f"{index + 1:x}" * 64}
        for index, role in enumerate(sorted(evidence.REQUIRED_WORK_MAP_SOURCE_ROLES))
    ]
    request_source = next(
        row for row in sources if row["role"] == "frozen_data_request"
    )
    request_source["sha256"] = request_sha
    map_path = case_root / "work_map.json"
    map_path.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    map_sha = evidence._sha256(map_path)
    (case_root / "runtime.json").write_text(
        json.dumps(
            {
                "single_mode_airfoil_work_map": {
                    "work_map": {
                        "path": "work_map.json",
                        "sha256": map_sha,
                        "confidence_multiplier": 1.96,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "cases": [
            {
                "scenario": "latest_airfoil_work_map_bridge",
                "runtime_config": "cases/hybrid/runtime.json",
            }
        ],
        "work_map_identity": {
            "sha256": map_sha,
            "fingerprint": "frozen-map",
        },
    }
    result = evidence._work_map_source_provenance(tmp_path, manifest)
    assert result["verified"] is True
    assert result["ehrmann_free_response_used_for_fit"] is False
    assert result["source_roles"] == sorted(evidence.REQUIRED_WORK_MAP_SOURCE_ROLES)

    removed = sources.pop(
        next(
            index
            for index, source in enumerate(sources)
            if source["role"] == "prescribed_motion_high_fidelity_producer_provenance"
        )
    )
    map_path.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="must contain only"):
        evidence._work_map_source_provenance(tmp_path, manifest)
    sources.append(removed)

    sources.append(
        {
            "role": "ehrmann_free_response",
            "label": "forbidden evaluation data",
            "sha256": "f" * 64,
        }
    )
    map_path.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    with pytest.raises(evidence.EvidenceError, match="must contain only"):
        evidence._work_map_source_provenance(tmp_path, manifest)
