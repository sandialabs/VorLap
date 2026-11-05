from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "paper_drafts/NAWEA_QBladeBridge/analysis/prepare_hvawt_uniform_pilot_paper_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_hvawt_uniform_pilot_paper_artifacts", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion)


def _synthetic_bundle() -> dict:
    active = {
        (6, 6.5): 0.27,
        (6, 7.25): 0.54,
        (6, 8.25): 0.38,
        (6, 8.5): 0.41,
        (12, 11.2): 0.23,
        (12, 12.2): 0.108,
        (12, 13.2): 0.46,
    }
    targets = []
    rows = []
    cases = []
    for wind in promotion.EXPECTED_WINDS:
        targets.append(
            {
                "wind_speed_mps": wind,
                "mode6_equilibrium_amplitude_m": (
                    0.005 if (6, wind) in active else 0.0
                ),
                "mode12_equilibrium_amplitude_m": (
                    0.007 if (12, wind) in active else 0.0
                ),
            }
        )
        for scenario in promotion.EXPECTED_SCENARIOS:
            case_id = f"U{wind:g}_{'native' if scenario.endswith('aero_only') else 'response'}"
            cases.append(
                {
                    "case_id": case_id,
                    "scenario": scenario,
                    "metrics": {
                        "maximum_total_supplemental_nodal_force_n": (
                            0.0 if scenario.endswith("aero_only") else 10.0 * wind
                        ),
                        "maximum_response_nodal_force_n": (
                            0.0 if scenario.endswith("aero_only") else wind / 2.0
                        ),
                    },
                }
            )
            for mode in promotion.EXPECTED_MODES:
                is_active = (mode, wind) in active
                amplitude = (
                    (0.001 if mode == 6 else 0.002) * (1.0 + wind / 20.0)
                    if scenario == promotion.EXPECTED_SCENARIOS[1]
                    else 1.0e-5
                )
                frequency_ratio = 1.05 if mode == 6 else 0.95
                rows.append(
                    {
                        "case_id": case_id,
                        "scenario": scenario,
                        "wind_speed_mps": wind,
                        "timestep_s": 0.0075,
                        "case_passed": not (
                            scenario == promotion.EXPECTED_SCENARIOS[1]
                            and wind in (8.25, 12.2)
                        ),
                        "tracked_mode": mode,
                        "reference_length_m": 0.5,
                        "peak_amplitude_estimate_m": amplitude,
                        "peak_amplitude_over_reference_length": amplitude / 0.5,
                        "dominant_frequency_hz": frequency_ratio,
                        "dominant_frequency_over_natural_frequency": frequency_ratio,
                        "activation_max": 2.0 if is_active else 0.0,
                        "complex_projection_leakage_ratio": 0.001,
                        "tail_half_peak_relative_difference": (
                            active[(mode, wind)] if is_active else 0.01
                        ),
                    }
                )
    failed_cross = [
        {
            "gate": f"mode{mode}_active_stationary_tail_u{wind:g}_dt0.0075",
            "passed": False,
            "observed": value,
            "limit_or_expected": 0.1,
        }
        for (mode, wind), value in sorted(active.items())
    ]
    periodicity_states = [
        {
            "case_id": f"U{wind:g}_response",
            "wind_speed_mps": wind,
            "mode": mode,
            "late_five_mean_amplitude_trend": {
                "normalized_slope_per_revolution": -0.01,
            },
            "late_five_raw_modal_velocity_rms_trend": {
                "normalized_slope_per_revolution": -0.02,
            },
            "structural_cycle_recurrence": {
                "one_period_autocorrelation": 0.95,
                "dominant_frequency_over_natural_frequency": 1.05,
            },
            "record_peak_revolution": 3,
        }
        for mode, wind in sorted(active)
    ]
    periodicity_summary = {
        "final_revolution_record_peak_count": 0,
        "late_mean_normalized_slope_min_per_revolution": -0.06,
        "late_mean_normalized_slope_max_per_revolution": 0.01,
        "late_peak_normalized_slope_min_per_revolution": -0.12,
        "late_peak_normalized_slope_max_per_revolution": 0.03,
        "late_raw_velocity_rms_normalized_slope_min_per_revolution": -0.07,
        "late_raw_velocity_rms_normalized_slope_max_per_revolution": 0.01,
        "late_mean_longest_consecutive_increase_count_max": 2,
        "late_peak_longest_consecutive_increase_count_max": 2,
        "late_raw_velocity_rms_longest_consecutive_increase_count_max": 2,
        "structural_cycle_autocorrelation_min": 0.92,
        "structural_cycle_autocorrelation_max": 0.99,
        "spectral_energy_within_10_percent_min": 0.84,
        "spectral_energy_within_10_percent_max": 0.99,
        "raw_dominant_frequency_error_min": 0.048,
        "raw_dominant_frequency_error_max": 0.057,
    }
    return {
        "spec": {
            "pilot": {
                "predicted_amplitude_target_by_selected_wind_and_mode_m": targets
            },
            "acceptance": {
                "maximum_tail_half_peak_amplitude_relative_difference": 0.1
            },
        },
        "rows": rows,
        "analysis": {
            "fingerprint": "analysis-fingerprint",
            "schema": promotion.EXPECTED_ANALYSIS_SCHEMA,
            "status": promotion.EXPECTED_STATUS,
            "cross_case_gates": failed_cross,
        },
        "local_analysis": {"fingerprint": "local-analysis-fingerprint"},
        "periodicity": {
            "fingerprint": "periodicity-fingerprint",
            "summary": periodicity_summary,
            "active_states": periodicity_states,
        },
        "manifest": {"fingerprint": "campaign-fingerprint"},
        "cases": cases,
        "source_clip_count": 0,
        "source_nonfinite_count": 0,
        "elapsed_seconds": [100.0 + index for index in range(20)],
    }


def test_derive_summary_preserves_completed_negative_screen() -> None:
    rows, summary = promotion.derive_summary(_synthetic_bundle())

    assert len(rows) == 40
    assert summary["status"] == "complete_v11_flag_with_descriptive_periodicity_followup"
    assert summary["case_count"] == 20
    assert summary["active_state_count"] == 7
    assert summary["failed_active_stationarity_state_count"] == 7
    assert summary["full_rerun_authorized"] is False
    assert summary["active_stationarity_min_fraction"] == pytest.approx(0.108)
    assert summary["active_stationarity_max_fraction"] == pytest.approx(0.54)
    assert summary["maximum_native_control_supplemental_nodal_force_n"] == 0.0
    assert summary["maximum_total_supplemental_nodal_force_n"] == 160.0
    assert summary["periodicity_summary"]["final_revolution_record_peak_count"] == 0


def test_derive_summary_rejects_an_unexpected_passing_active_state() -> None:
    bundle = _synthetic_bundle()
    hybrid = next(
        row
        for row in bundle["rows"]
        if row["scenario"] == promotion.EXPECTED_SCENARIOS[1]
        and row["tracked_mode"] == 6
        and row["wind_speed_mps"] == 6.5
    )
    hybrid["tail_half_peak_relative_difference"] = 0.09

    with pytest.raises(promotion.PromotionError, match="seven failed active"):
        promotion.derive_summary(bundle)


def test_returned_bundle_validates_when_present() -> None:
    campaign = promotion.DEFAULT_CAMPAIGN
    if not (campaign / "pilot_analysis.json").is_file():
        pytest.skip("Compact Flight evidence bundle is not installed in this checkout.")

    bundle = promotion.validate_bundle(campaign)

    assert bundle["analysis"]["status"] == promotion.EXPECTED_STATUS
    assert len(bundle["cases"]) == 20
    assert bundle["source_clip_count"] == 0
    assert bundle["source_nonfinite_count"] == 0
