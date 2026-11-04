import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from examples.ehrmann2014_naca0018 import campaign
from examples.ehrmann2014_naca0018.build_engineering_validation_evidence import (
    EngineeringEvidenceError,
    _adjacent_lockin_points,
    _experimental_peak,
    _validate_execution_report,
    _write_plot_data,
    assess_rows,
)


def _rows():
    figure7_targets = {0.96: 0.0125, 1.44: 0.0064}
    rows = []
    for md in campaign.MASS_DAMPINGS:
        target = figure7_targets[md]
        for ustar in campaign.ENGINEERING_REDUCED_VELOCITIES:
            shape = max(0.05, 1.0 - 0.55 * abs(ustar - 4.7))
            for scenario in campaign.ENGINEERING_LOCKIN_SCENARIOS:
                latest = scenario == campaign.ENGINEERING_LOCKIN_SCENARIOS[1]
                rows.append(
                    {
                        "scenario": scenario,
                        "mass_damping": md,
                        "reduced_velocity": ustar,
                        "timestep_s": campaign.TIMESTEP_S,
                        "terminal_amplitude_over_chord": (
                            target * shape if latest else 1.0e-5
                        ),
                        "terminal_dominant_frequency_over_fn": (
                            1.0 if latest else None
                        ),
                        "stationarity_amplitude_pass": True,
                    }
                )
        for scenario in campaign.ENGINEERING_LOCKIN_SCENARIOS:
            latest = scenario == campaign.ENGINEERING_LOCKIN_SCENARIOS[1]
            rows.append(
                {
                    "scenario": scenario,
                    "mass_damping": md,
                    "reduced_velocity": campaign.ENGINEERING_TIMESTEP_CHECK_USTAR,
                    "timestep_s": campaign.HALF_TIMESTEP_S,
                    "terminal_amplitude_over_chord": target if latest else 1.0e-5,
                    "terminal_dominant_frequency_over_fn": 1.0 if latest else None,
                    "stationarity_amplitude_pass": True,
                }
            )
    figure6 = [{"reduced_velocity": 4.586149, "rms_acceleration_mps2": 1.0}]
    figure7 = [
        {"mass_damping": 0.96, "maximum_amplitude_over_chord": 0.0125},
        {"mass_damping": 1.44, "maximum_amplitude_over_chord": 0.0064},
    ]
    return rows, figure6, figure7


def test_engineering_acceptance_passes_explicit_raw_latest_experiment_contract():
    rows, figure6, figure7 = _rows()
    result = assess_rows(rows, figure6=figure6, figure7=figure7)
    assert result["status"] == "passed"
    assert result["pass"]
    assert all(value["pass"] for value in result["mass_damping_results"])
    assert all(
        len(value["lockin_reduced_velocities"]) >= 2
        for value in result["mass_damping_results"]
    )


def test_resolved_peak_points_are_adjacent_on_the_declared_nonuniform_grid():
    rows = [
        {
            "reduced_velocity": value,
            "terminal_amplitude_over_chord": amplitude,
            "terminal_dominant_frequency_over_fn": 1.0,
        }
        for value, amplitude in ((4.5, 0.0), (4.6, 0.8), (4.7, 1.0), (4.8, 0.0))
    ]
    assert _adjacent_lockin_points(rows) == [4.6, 4.7]


def test_experimental_peak_uses_nearest_marker_outside_digitized_damping_range():
    points = [
        {"mass_damping": 1.0, "maximum_amplitude_over_chord": 0.012},
        {"mass_damping": 1.5, "maximum_amplitude_over_chord": 0.006},
    ]
    assert _experimental_peak(points, 0.96) == 0.012
    assert _experimental_peak(points, 1.6) == 0.006


def test_engineering_acceptance_rejects_31_percent_amplitude_error():
    rows, figure6, figure7 = _rows()
    mutated = copy.deepcopy(rows)
    for row in mutated:
        if (
            row["mass_damping"] == 0.96
            and row["scenario"] == campaign.ENGINEERING_LOCKIN_SCENARIOS[1]
        ):
            row["terminal_amplitude_over_chord"] *= 1.45
    result = assess_rows(mutated, figure6=figure6, figure7=figure7)
    first = result["mass_damping_results"][0]
    assert not first["checks"]["peak_amplitude_within_30_percent_plus_digitization"]
    assert not result["pass"]


def test_engineering_acceptance_rejects_no_finite_lockin_interval():
    rows, figure6, figure7 = _rows()
    mutated = copy.deepcopy(rows)
    for row in mutated:
        if (
            row["scenario"] == campaign.ENGINEERING_LOCKIN_SCENARIOS[1]
            and row["reduced_velocity"] != 4.5
        ):
            row["terminal_amplitude_over_chord"] = 1.0e-8
    result = assess_rows(mutated, figure6=figure6, figure7=figure7)
    assert not result["checks"]["both_mass_damping_cases_pass"]


def test_local_docker_execution_contract_is_distinct_and_capped(tmp_path: Path):
    inspect = tmp_path / "inspect.json"
    inspect.write_text('[{"HostConfig":{"NanoCpus":8000000000}}]\n')
    digest = hashlib.sha256(inspect.read_bytes()).hexdigest()
    manifest = {
        "fingerprint": "a" * 64,
        "engineering_lockin_identity": {"sha256": "b" * 64},
    }
    contract = {
        "schema": "vorlap-ehrmann2014-engineering-local-docker-execution-v2",
        "campaign_schema": campaign.ENGINEERING_LOCKIN_SCHEMA,
        "calibration_sha256": "b" * 64,
        "exact_raw_reuse_case_count": 18,
        "new_raw_case_count": 8,
        "hybrid_case_count": 26,
        "complete_at_end": 52,
        "nalu_required": False,
        "threads_per_case": 1,
        "container": {
            "platform": "docker_linux_amd64",
            "image_id": "sha256:" + "c" * 64,
            "nano_cpus": 8_000_000_000,
            "oom_killed": False,
            "inspect_file": {"path": str(inspect), "sha256": digest},
        },
    }
    report = tmp_path / "execution.json"
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "campaign_fingerprint": "a" * 64,
                "execution_contract": contract,
            }
        )
    )
    assert _validate_execution_report(tmp_path, report, manifest) == contract

    mutated = copy.deepcopy(contract)
    mutated["container"]["nano_cpus"] = 9_000_000_000
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "campaign_fingerprint": "a" * 64,
                "execution_contract": mutated,
            }
        )
    )
    with pytest.raises(EngineeringEvidenceError, match="Docker"):
        _validate_execution_report(tmp_path, report, manifest)


def test_plot_data_explicitly_identifies_experiment_raw_and_latest(tmp_path: Path):
    destination = tmp_path / "plot_data.csv"
    common = {
        "mass_damping": 0.96,
        "reduced_velocity": 4.7,
        "timestep_s": campaign.TIMESTEP_S,
        "terminal_dominant_frequency_hz": 5.97,
        "terminal_amplitude_m": 0.0023,
        "terminal_amplitude_over_chord": 0.01,
        "terminal_dominant_frequency_over_fn": 0.995,
    }
    _write_plot_data(
        destination,
        rows=[
            {**common, "scenario": campaign.ENGINEERING_LOCKIN_SCENARIOS[0]},
            {**common, "scenario": campaign.ENGINEERING_LOCKIN_SCENARIOS[1]},
        ],
        figure6=[{"reduced_velocity": 4.586, "rms_acceleration_mps2": 1.85}],
        figure7=[
            {
                "mass_damping": 1.008,
                "maximum_amplitude_over_chord": 0.0124,
                "mass_damping_digitization_uncertainty": 0.01,
                "amplitude_over_chord_digitization_uncertainty": 0.0006,
            }
        ],
    )
    records = list(csv.DictReader(destination.open(encoding="utf-8")))
    assert {record["series"] for record in records} == {
        "Ehrmann experiment (digitized)",
        "Raw QBlade",
        "Latest VorLap-QBlade engineering bridge",
    }
