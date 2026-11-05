from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import pytest

from vorlap.dlc_manifest import (
    DLCManifestError,
    build_manifest,
    build_wind_record,
    disk_preflight,
    load_manifest,
    selected_row,
    validate_manifest,
    validate_wind_record,
    wind_record_path,
    write_json,
    sha256_file,
    sha256_payload,
)
from vorlap.execution_provenance import (
    build_execution_provenance,
    validate_execution_provenance,
)


def matrix_spec(tmp_path, *, status="approved", weights=True):
    result_filter = tmp_path / "parked_fatigue_results.filter"
    result_filter.write_text("Time [s]\nLSS Rpm [rpm]\n", encoding="utf-8")
    fatigue_weights = {"5": 0.2, "7": 0.3} if weights else {}
    qualification = {
        "schema_version": 1,
        "qualified": True,
        "timestep_s": 0.0075,
        "source_frequency_truncation": "nyquist",
        "modal_frequencies_hz": [1, 2, 3, 4, 5],
    }
    qualification["qualification_sha256"] = sha256_payload(qualification)
    qualification_path = write_json(tmp_path / "modal_timestep_qualification.json", qualification)
    fatigue_review = {
        "schema_version": 1,
        "passed": True,
        "fatigue_weights_by_speed": fatigue_weights,
    }
    fatigue_review["review_sha256"] = sha256_payload(fatigue_review)
    fatigue_review_path = write_json(tmp_path / "fatigue_weighting_review.json", fatigue_review)
    grid_review = {
        "schema_version": 1,
        "passed": True,
        "grid": {
            "num_grid_y": 26,
            "num_grid_z": 38,
            "grid_width_m": 26,
            "grid_height_m": 26,
            "hub_height_m": 18,
            "ref_height_m": 18,
        },
        "qblade_conventions": {
            "horizontal_inflow_angle_deg": 0,
            "vertical_inflow_angle_deg": 0,
        },
    }
    grid_review["review_sha256"] = sha256_payload(grid_review)
    grid_review_path = write_json(tmp_path / "grid_domain_review.json", grid_review)
    modal_overlap_review = {
        "schema_version": 1,
        "passed": True,
        "matrix_spec": {"current_dlc64_wind_speeds_mps": [5.0, 7.0]},
        "current_matrix_coverage": {"passed": True},
    }
    modal_overlap_review["review_sha256"] = sha256_payload(modal_overlap_review)
    modal_overlap_review_path = write_json(
        tmp_path / "modal_overlap_review.json", modal_overlap_review
    )
    naca0018_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "ReferenceHVAWTQBladeModel"
        / "10m_HVAWT_reference_turbine"
        / "VorLapAirfoils"
        / "NACA0018.h5"
    )
    return {
        "status": status,
        "study_id": "test_parked_dlc",
        "basis": {
            "iec_edition": "IEC 61400-1 project edition",
            "wind_class": "IEC IA",
            "fatigue_weighting_source": "test probability table",
            "matrix_approval": "approved for test",
            "matrix_state": status,
            "airfoil_source_sha256": sha256_file(naca0018_path),
            "modal_timestep_qualification_path": str(qualification_path),
            "modal_timestep_qualification_sha256": sha256_file(qualification_path),
            "fatigue_weighting_review_path": str(fatigue_review_path),
            "fatigue_weighting_review_sha256": sha256_file(fatigue_review_path),
            "grid_domain_review_path": str(grid_review_path),
            "grid_domain_review_sha256": sha256_file(grid_review_path),
            "modal_overlap_review_path": str(modal_overlap_review_path),
            "modal_overlap_review_sha256": sha256_file(modal_overlap_review_path),
        },
        "wind_root": str(tmp_path / "winds"),
        "run_root": str(tmp_path / "runs"),
        "duration_s": 660,
        "store_from_s": 60,
        "timestep_s": 0.0075,
        "wind_analysis_time_s": 660,
        "wind_timestep_s": 0.0075,
        "num_grid_y": 26,
        "num_grid_z": 38,
        "grid_width_m": 26,
        "grid_height_m": 26,
        "hub_height_m": 18,
        "ref_height_m": 18,
        "horizontal_inflow_angle_deg": 0,
        "vertical_inflow_angle_deg": 0,
        "shear_exponent": 0.23,
        "air_density_kg_m3": 1.0296,
        "air_kinematic_viscosity_m2_s": 1.647e-5,
        "execution_settings": {
            "source_frequency_truncation": "nyquist",
            "source_bounds_policy": "raise",
            "wake_profile": "parked-reduced-v2",
            "result_filter_path": str(result_filter),
            "result_filter_sha256": sha256_file(result_filter),
            "result_filter_channel_count": 2,
            "checkpoint_every": 4000,
            "store_replay": False,
            "store_project": False,
            "use_opencl": False,
            "case_workers": 1,
            "auto_node_threads": True,
            "run_final_audit": True,
            "require_source_coverage": True,
            "require_postprocess_artifacts": True,
        },
        "min_seeds_by_dlc": {"6.1": 1, "6.4": 6},
        "groups": [
            {
                "dlc": "6.4",
                "wind_speeds_mps": [5, 7],
                "seeds": [4433456, 4433457, 4433458, 4433459, 4433460, 4433461],
                "turbsim_iec_wind_type": "NTM",
                "include_in_fatigue": True,
                "include_in_ultimate": True,
                "fatigue_weights_by_speed": fatigue_weights,
            },
            {
                "dlc": "6.1",
                "wind_speeds_mps": [50],
                "seeds": [5533456],
                "turbsim_iec_wind_type": "1EWM50",
                "include_in_fatigue": False,
                "include_in_ultimate": True,
            },
        ],
    }


def test_build_approved_manifest_expands_one_row_per_dlc_speed_seed(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path))
    validate_manifest(manifest, require_approved=True)

    assert manifest["row_count"] == 13
    assert manifest["retained_duration_s"] == 600.0
    assert manifest["scenarios"] == ["baseline", "vorlap"]
    assert manifest["schema_version"] == 3
    assert manifest["execution_settings"]["source_frequency_truncation"] == "nyquist"
    assert manifest["execution_settings"]["source_bounds_policy"] == "raise"
    assert manifest["execution_settings"]["wake_profile"] == "parked-reduced-v2"
    assert manifest["execution_settings"]["result_filter_sha256"] == sha256_file(
        tmp_path / "parked_fatigue_results.filter"
    )
    assert manifest["execution_settings"]["result_filter_channel_count"] == 2
    assert manifest["execution_settings"]["case_workers"] == 1
    assert len(manifest["manifest_sha256"]) == 64
    first = manifest["rows"][0]
    assert first["row_id"] == "dlc6p4_U05p00ms_s4433456"
    assert first["wind_file"].endswith("/dlc6p4/U05p00ms/dlc6p4_U05p00ms_s4433456.bts")
    assert first["estimated_wind_bytes"] > 500_000_000
    assert manifest["estimated_wind_bytes_total"] == sum(
        row["estimated_wind_bytes"] for row in manifest["rows"]
    )


def test_draft_allows_unfrozen_weights_but_production_validation_rejects_it(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path, status="draft", weights=False))
    validate_manifest(manifest, require_approved=False)

    with pytest.raises(DLCManifestError, match="still draft"):
        validate_manifest(manifest, require_approved=True)

    approved = deepcopy(manifest)
    approved["status"] = "approved"
    approved.pop("manifest_sha256")
    from vorlap.dlc_manifest import sha256_payload

    approved["manifest_sha256"] = sha256_payload(approved)
    with pytest.raises(DLCManifestError, match="missing fatigue_weight"):
        validate_manifest(approved, require_approved=True)


def test_manifest_rejects_duplicate_seed_rows(tmp_path):
    spec = matrix_spec(tmp_path)
    spec["groups"][0]["seeds"][-1] = spec["groups"][0]["seeds"][0]

    with pytest.raises(DLCManifestError, match="Duplicate row_id"):
        build_manifest(spec)


def test_manifest_digest_detects_content_change(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path))
    manifest["basis"]["tampered"] = True

    with pytest.raises(DLCManifestError, match="manifest_sha256"):
        validate_manifest(manifest, require_approved=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_bounds_policy", "warn", "source_bounds_policy='raise'"),
        ("case_workers", 2, "case_workers=1"),
        ("checkpoint_every", 0, "restart checkpointing"),
        ("store_replay", True, "store_replay=false"),
        ("store_project", True, "store_project=false"),
        ("use_opencl", True, "use_opencl=false"),
        ("auto_node_threads", False, "auto_node_threads=true"),
        ("wake_profile", "reference", "wake_profile='parked-reduced-v2'"),
    ],
)
def test_approved_manifest_rejects_nonproduction_execution_policy(tmp_path, field, value, message):
    spec = matrix_spec(tmp_path)
    spec["execution_settings"][field] = value
    manifest = build_manifest(spec)

    with pytest.raises(DLCManifestError, match=message):
        validate_manifest(manifest, require_approved=True)


def test_approved_manifest_requires_qualification_truncation_match(tmp_path):
    spec = matrix_spec(tmp_path)
    qualification_path = Path(spec["basis"]["modal_timestep_qualification_path"])
    qualification = json.loads(qualification_path.read_text())
    qualification["source_frequency_truncation"] = "none"
    qualification.pop("qualification_sha256")
    qualification["qualification_sha256"] = sha256_payload(qualification)
    write_json(qualification_path, qualification)
    spec["basis"]["modal_timestep_qualification_sha256"] = sha256_file(qualification_path)
    manifest = build_manifest(spec)

    with pytest.raises(DLCManifestError, match="truncation does not match"):
        validate_manifest(manifest, require_approved=True)


def test_approved_manifest_rejects_changed_result_filter(tmp_path):
    spec = matrix_spec(tmp_path)
    manifest = build_manifest(spec)
    Path(spec["execution_settings"]["result_filter_path"]).write_text(
        "Time [s]\nchanged channel\n",
        encoding="utf-8",
    )

    with pytest.raises(DLCManifestError, match="result-filter SHA-256"):
        validate_manifest(manifest, require_approved=True)


def test_approved_manifest_requires_matching_grid_review(tmp_path):
    spec = matrix_spec(tmp_path)
    spec["grid_width_m"] = 25
    manifest = build_manifest(spec)

    with pytest.raises(DLCManifestError, match="grid_width_m does not match"):
        validate_manifest(manifest, require_approved=True)


def test_approved_manifest_requires_matching_inflow_angle_review(tmp_path):
    spec = matrix_spec(tmp_path)
    spec["horizontal_inflow_angle_deg"] = 5
    manifest = build_manifest(spec)

    with pytest.raises(DLCManifestError, match="horizontal_inflow_angle_deg does not match"):
        validate_manifest(manifest, require_approved=True)


def test_approved_manifest_requires_matching_fatigue_weighting_review(tmp_path):
    spec = matrix_spec(tmp_path)
    spec["groups"][0]["fatigue_weights_by_speed"]["5"] = 0.25
    manifest = build_manifest(spec)

    with pytest.raises(DLCManifestError, match="weights do not match"):
        validate_manifest(manifest, require_approved=True)


def test_approved_manifest_requires_speeds_matching_modal_overlap_review(tmp_path):
    spec = matrix_spec(tmp_path)
    review_path = Path(spec["basis"]["modal_overlap_review_path"])
    review = json.loads(review_path.read_text())
    review["matrix_spec"]["current_dlc64_wind_speeds_mps"] = [5.0, 8.0]
    review.pop("review_sha256")
    review["review_sha256"] = sha256_payload(review)
    write_json(review_path, review)
    spec["basis"]["modal_overlap_review_sha256"] = sha256_file(review_path)
    manifest = build_manifest(spec)

    with pytest.raises(DLCManifestError, match="do not match the modal-overlap review"):
        validate_manifest(manifest, require_approved=True)


def test_disk_preflight_uses_full_matrix_estimate_and_margin(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path))
    estimated = manifest["estimated_wind_bytes_total"]
    report = disk_preflight(
        manifest,
        required_factor=1.25,
        available_bytes=estimated * 2,
    )
    assert report["passed"] is True
    assert report["required_available_bytes"] == int(estimated * 1.25)

    with pytest.raises(DLCManifestError, match="Insufficient wind-field disk space"):
        disk_preflight(
            manifest,
            required_factor=1.25,
            available_bytes=estimated,
        )


def test_manifest_round_trip_and_row_selection(tmp_path):
    path = write_json(tmp_path / "manifest.json", build_manifest(matrix_spec(tmp_path)))
    manifest = load_manifest(path)

    assert selected_row(manifest, 12)["dlc"] == "6.1"
    with pytest.raises(DLCManifestError, match="outside"):
        selected_row(manifest, 13)


def test_wind_record_fails_closed_on_changed_file(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path))
    row = manifest["rows"][0]
    wind_file = tmp_path / "winds" / "dlc6p4" / "U05p00ms" / f"{row['row_id']}.bts"
    wind_file.parent.mkdir(parents=True)
    wind_file.write_bytes(b"wind-one")
    record = build_wind_record(row, manifest)
    write_json(wind_record_path(row), record)

    assert validate_wind_record(row, manifest)["sha256"] == record["sha256"]
    wind_file.write_bytes(b"wind-two")
    with pytest.raises(DLCManifestError, match="hash changed"):
        validate_wind_record(row, manifest)


def test_cli_build_and_validate(tmp_path):
    spec_path = tmp_path / "spec.json"
    manifest_path = tmp_path / "manifest.json"
    spec_path.write_text(json.dumps(matrix_spec(tmp_path)), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "run_headless" / "build_parked_dlc_manifest.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "build",
            "--spec",
            str(spec_path),
            "--out",
            str(manifest_path),
            "--require-approved",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    completed = subprocess.run(
        [sys.executable, str(script), "count", "--manifest", str(manifest_path)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.strip() == "13"


def test_tracked_three_realization_candidate_expands_fail_closed_production_shape():
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "run_headless" / "config" / "parked_dlc_matrix_draft.json").read_text())
    manifest = build_manifest(spec)

    validate_manifest(manifest, require_approved=False)
    assert manifest["status"] == "draft"
    assert manifest["row_count"] == 34
    assert manifest["execution_settings"]["source_frequency_truncation"] == "nyquist"
    assert manifest["execution_settings"]["source_bounds_policy"] == "raise"
    assert manifest["execution_settings"]["wake_profile"] == "parked-reduced-v2"
    assert manifest["execution_settings"]["result_filter_channel_count"] == 26
    assert manifest["execution_settings"]["result_filter_sha256"] == (
        "e3f051c4e9a347de39ebbd5d10836edc02ec194c7895302b22ef59c8dac29f0b"
    )
    assert manifest["execution_settings"]["auto_node_threads"] is True
    assert manifest["execution_settings"]["use_opencl"] is False
    dlc64_rows = [row for row in manifest["rows"] if row["dlc"] == "6.4"]
    assert sorted({row["seed"] for row in dlc64_rows}) == [4433456, 4433457, 4433458]
    assert sorted({row["wind_speed_mps"] for row in dlc64_rows}) == [
        3.0,
        7.0,
        8.0,
        8.75,
        11.25,
        13.0,
        14.75,
        18.0,
        19.25,
        21.5,
        35.0,
    ]
    expected_weights = {
        0.00026436537041197997,
        0.0002404294826746347,
        0.00027951580403126755,
        0.0005683513708698844,
        0.0006824542890062364,
        0.0007563348336359039,
        0.0008409094520925261,
        0.00102597919253122,
        0.0010489987447227687,
        0.0017800130822315456,
        0.002511985249715154,
    }
    assert {row["fatigue_weight"] for row in dlc64_rows} == expected_weights
    with pytest.raises(DLCManifestError, match="still draft"):
        validate_manifest(manifest, require_approved=True)


def test_tracked_approved_matrix_is_reproducible_and_passes_production_validation():
    root = Path(__file__).resolve().parents[1]
    config = root / "run_headless" / "config"
    spec = json.loads((config / "parked_dlc_matrix_approved_spec.json").read_text())
    tracked_manifest = json.loads((config / "parked_dlc_matrix_approved.json").read_text())

    assert build_manifest(spec) == tracked_manifest
    assert tracked_manifest["status"] == "approved"
    assert tracked_manifest["row_count"] == 34
    assert tracked_manifest["requirements"]["min_seeds_by_dlc"]["6.4"] == 3
    assert tracked_manifest["execution_settings"]["source_bounds_policy"] == "raise"

    local_spec = deepcopy(spec)
    local_spec["basis"].update(
        {
            "modal_timestep_qualification_path": str(config / "modal_timestep_qualification.json"),
            "fatigue_weighting_review_path": str(config / "dlc64_fatigue_weighting_review.json"),
            "modal_overlap_review_path": str(config / "vorlap_modal_overlap_review.json"),
            "grid_domain_review_path": str(config / "turbsim_grid_domain_review.json"),
        }
    )
    local_spec["execution_settings"]["result_filter_path"] = str(
        config / "parked_fatigue_results.filter"
    )
    local_manifest = build_manifest(local_spec)

    validate_manifest(local_manifest, require_approved=True)
    assert local_manifest["execution_settings"]["source_frequency_truncation"] == "nyquist"
    assert local_manifest["execution_settings"]["wake_profile"] == "parked-reduced-v2"


def test_tracked_dlc64_weighting_grid_and_overlap_reviews_are_reproducible(tmp_path):
    root = Path(__file__).resolve().parents[1]
    weighting_script = root / "run_headless" / "build_dlc64_fatigue_weighting.py"
    grid_script = root / "run_headless" / "audit_turbsim_grid_domain.py"
    overlap_script = root / "run_headless" / "screen_vorlap_modal_overlap.py"
    weighting_out = tmp_path / "fatigue_weighting.json"
    grid_out = tmp_path / "grid_domain.json"
    overlap_out = tmp_path / "modal_overlap.json"

    subprocess.run(
        [sys.executable, str(weighting_script), "--out", str(weighting_out)],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, str(grid_script), "--out", str(grid_out)],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, str(overlap_script), "--out", str(overlap_out)],
        check=True,
        text=True,
        capture_output=True,
    )

    tracked_weighting = root / "run_headless" / "config" / "dlc64_fatigue_weighting_review.json"
    tracked_grid = root / "run_headless" / "config" / "turbsim_grid_domain_review.json"
    tracked_overlap = root / "run_headless" / "config" / "vorlap_modal_overlap_review.json"
    assert weighting_out.read_bytes() == tracked_weighting.read_bytes()
    assert grid_out.read_bytes() == tracked_grid.read_bytes()
    if overlap_out.read_bytes() != tracked_overlap.read_bytes():
        # The approved runtime source may be an exact replicated-Re envelope
        # extension of the immutable screening parent. In that one case, the
        # active source hash and low-Re clipping fractions change, while every
        # non-Re dataset and the decision-relevant overlap ranking must remain
        # unchanged.
        active_source = root / (
            "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/"
            "VorLapAirfoils/NACA0018.h5"
        )
        with h5py.File(active_source, "r") as h5:
            processing = json.loads(h5.attrs["processing_config_json"])
        tracked_overlap_payload = json.loads(tracked_overlap.read_text())
        current_overlap_payload = json.loads(overlap_out.read_text())
        tracked_naca0018 = next(
            row
            for row in tracked_overlap_payload["source_basis"]["airfoil_files"]
            if Path(row["path"]).name == "NACA0018.h5"
        )
        current_naca0018 = next(
            row
            for row in current_overlap_payload["source_basis"]["airfoil_files"]
            if Path(row["path"]).name == "NACA0018.h5"
        )
        assert processing["reynolds_envelope_extension_exact"] is True
        assert processing["reynolds_envelope_extension_generator"] == (
            "scripts/extend_replicated_reynolds_grid.py"
        )
        assert processing["replicate_re_parent_sha256"] == tracked_naca0018["sha256"]
        assert current_naca0018["sha256"] == sha256_file(active_source)
        assert current_overlap_payload["required_peak_wind_speeds_mps"] == (
            tracked_overlap_payload["required_peak_wind_speeds_mps"]
        )
        assert current_overlap_payload["recommended_dlc64_wind_speeds_mps"] == (
            tracked_overlap_payload["recommended_dlc64_wind_speeds_mps"]
        )
        for key in ("ranked_force_overlap_peaks", "ranked_downtime_weighted_overlap_peaks"):
            assert [row["wind_speed_mps"] for row in current_overlap_payload[key]] == [
                row["wind_speed_mps"] for row in tracked_overlap_payload[key]
            ]
        for tracked_row, current_row in zip(
            tracked_overlap_payload["sweep"], current_overlap_payload["sweep"], strict=True
        ):
            assert current_row["wind_speed_mps"] == tracked_row["wind_speed_mps"]
            assert current_row["overlap_score_n"] == pytest.approx(
                tracked_row["overlap_score_n"], rel=1.0e-12, abs=1.0e-15
            )
            assert current_row["source_clip_query_fraction"] <= (
                tracked_row["source_clip_query_fraction"] + 1.0e-15
            )
        assert current_overlap_payload["source_domain_checks"]["passed"] is True
    weighting = json.loads(weighting_out.read_text())
    grid = json.loads(grid_out.read_text())
    assert weighting["availability"] == 0.99
    assert weighting["total_non_power_probability_through_dlc64_limit"] == pytest.approx(
        0.009999336871923122
    )
    assert weighting["expected_total_non_power_hours_per_year"] == pytest.approx(87.6)
    assert weighting["expected_total_non_power_days_per_year"] == pytest.approx(3.65)
    assert grid["passed"] is True
    assert grid["margins"] == {
        "above_rotor_m": 8.96,
        "below_rotor_m": 6.5,
        "lateral_each_side_m": 7.23,
        "required_minimum_m": 2.0,
    }


def test_manifest_row_qblade_dry_run_archives_selected_row(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path))
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    script = Path(__file__).resolve().parents[1] / "run_headless" / "run_parked_dlc_manifest_row.py"
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_EXE": sys.executable,
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "RUN_PAPER_PREFLIGHT": "0",
            "QBLADE_HORIZONTAL_INFLOW_ANGLE": "17",
            "QBLADE_VERTICAL_INFLOW_ANGLE": "9",
            "QBLADE_SHEAR_EXPONENT": "0.9",
            "QBLADE_AIR_DENSITY": "9",
            "QBLADE_AIR_KINEMATIC_VISCOSITY": "9",
        }
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--index",
            "12",
            "--phase",
            "qblade",
            "--parent-dry-run",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    run_root = tmp_path / "runs" / "dlc6p1_U50p00ms_s5533456"
    archived = json.loads((run_root / "dlc_manifest_row.json").read_text(encoding="utf-8"))
    settings = json.loads((run_root / "campaign_settings.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_root / "execution_environment_qblade.json").read_text())
    assert archived["row_index"] == 12
    assert archived["row"]["dlc"] == "6.1"
    assert settings["wind_speeds"] == [50.0]
    assert settings["wind_name"].endswith("dlc6p1_U50p00ms_s5533456.bts")
    assert settings["duration_s"] == 660.0
    assert settings["store_from_s"] == 60.0
    assert settings["scenario_names"] == ["baseline", "vorlap"]
    assert settings["source_frequency_truncation"] == "nyquist"
    assert settings["source_bounds_policy"] == "raise"
    assert settings["wake_profile"] == "parked-reduced-v2"
    assert settings["result_filter"] == str(tmp_path / "parked_fatigue_results.filter")
    assert settings["result_filter_sha256"] == sha256_file(
        tmp_path / "parked_fatigue_results.filter"
    )
    assert settings["horizontal_inflow_angle_deg"] == 0.0
    assert settings["vertical_inflow_angle_deg"] == 0.0
    assert settings["shear_exponent"] == 0.23
    assert settings["air_density_kg_m3"] == 1.0296
    assert settings["air_kinematic_viscosity_m2_s"] == 1.647e-5
    assert settings["checkpoint_every"] == 4000
    assert settings["store_replay"] is False
    assert settings["use_opencl"] is False
    assert settings["case_workers"] == 1
    assert provenance["phase"] == "qblade"
    assert provenance["manifest_sha256"] == manifest["manifest_sha256"]
    assert json.loads((run_root / "audits" / "plan_audit.json").read_text())["passed"] is True


def test_manifest_row_wind_phase_builds_hashed_sidecar_without_qblade(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path))
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    script = Path(__file__).resolve().parents[1] / "run_headless" / "run_parked_dlc_manifest_row.py"
    fake_turbsim = tmp_path / "fake_turbsim.sh"
    fake_turbsim.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == "-h" ]]; then
  printf 'TurbSim version test\n'
  exit 0
fi
input="$1"
printf 'fake-bts-data' > "${input%.inp}.bts"
""",
        encoding="utf-8",
    )
    fake_turbsim.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_EXE": sys.executable,
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "TURBSIM_EXE": str(fake_turbsim),
            "OPENFAST_SOURCE_REF": "test-openfast-source",
        }
    )

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--index",
            "0",
            "--phase",
            "wind",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    row = manifest["rows"][0]
    record = validate_wind_record(row, manifest)
    settings = json.loads((Path(row["run_root"]) / "campaign_settings.json").read_text())
    provenance = json.loads(
        (Path(row["run_root"]) / "execution_environment_wind.json").read_text()
    )
    assert record["bytes"] == len(b"fake-bts-data")
    assert len(record["sha256"]) == 64
    assert settings["wind_only"] is True
    assert settings["turbsim_seed_base"] == row["seed"]
    assert settings["turbsim_wind_files"][0]["wind_file_sha256"] == record["sha256"]
    assert provenance["phase"] == "wind"
    assert provenance["turbsim_executable"]["sha256"] == sha256_file(fake_turbsim)
    assert provenance["turbsim_source"]["explicit_ref"] == "test-openfast-source"


def test_manifest_submission_dry_run_creates_wind_then_qblade_arrays(tmp_path):
    manifest_path = write_json(tmp_path / "manifest.json", build_manifest(matrix_spec(tmp_path)))
    script = Path(__file__).resolve().parents[1] / "run_headless" / "submit_parked_dlc_manifest.sj"
    env = os.environ.copy()
    env.update(
        {
            "DLC_MANIFEST": str(manifest_path),
            "SUBMIT_DLC_MANIFEST": "1",
            "DLC_SUBMIT_DRY_RUN": "1",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
    assert len(commands) == 3
    assert "--array=0-12%13" in commands[0]
    assert "--time=02:00:00" in commands[0]
    assert "--partition=short" in commands[0]
    assert "--exclusive" not in commands[0]
    assert "DLC_PHASE=wind" in commands[0]
    assert "--array=0-12%13" in commands[1]
    assert "--exclusive" in commands[1]
    assert "--time=47:59:00" in commands[1]
    assert "--partition=batch" in commands[1]
    assert "afterok:WIND_ARRAY_JOB_ID" in commands[1]
    assert "DLC_PHASE=qblade" in commands[1]
    assert "--time=02:00:00" in commands[2]
    assert "--partition=short" in commands[2]
    assert "afterok:QBLADE_ARRAY_JOB_ID" in commands[2]
    assert "DLC_PHASE=aggregate" in commands[2]


def test_manifest_submission_defaults_to_measured_qblade_allocation(tmp_path):
    manifest_path = write_json(tmp_path / "manifest.json", build_manifest(matrix_spec(tmp_path)))
    script = Path(__file__).resolve().parents[1] / "run_headless" / "submit_parked_dlc_manifest.sj"
    env = os.environ.copy()
    env.update(
        {
            "DLC_MANIFEST": str(manifest_path),
            "SUBMIT_DLC_MANIFEST": "1",
            "DLC_SUBMIT_DRY_RUN": "1",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
    assert "--time=47:59:00" in commands[1]
    assert "--partition=batch" in commands[1]
    assert "--exclusive" in commands[1]


def test_provenance_recovery_dry_run_selects_failed_rows_and_chains_aggregate(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path))
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    for index, row in enumerate(manifest["rows"]):
        audit_path = Path(row["run_root"]) / "audits" / "results_audit.json"
        issues = []
        passed = index != 3
        if not passed:
            issues.append(
                {
                    "severity": "error",
                    "code": "checkpoint_runtime_identity_mismatch",
                    "scenario": "vorlap",
                    "case": row["row_id"],
                }
            )
        write_json(audit_path, {"passed": passed, "issues": issues})

    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    script = (
        Path(__file__).resolve().parents[1]
        / "run_headless"
        / "submit_parked_dlc_provenance_recovery.sj"
    )
    env = os.environ.copy()
    env.update(
        {
            "DLC_MANIFEST": str(manifest_path),
            "DLC_RECOVERY_SUBMIT_DRY_RUN": "1",
            "DLC_RECOVERY_REQUIRE_CLEAN_CHECKOUT": "0",
            "PYTHON_EXE": sys.executable,
            "QBLADE_ROOT": str(qblade_root),
            "VORLAP_ROOT": str(Path(__file__).resolve().parents[1]),
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    commands = [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]
    assert len(commands) == 2
    assert "--array=3%6" in commands[0]
    assert "prov-recover" in commands[0]
    assert "recover" in commands[1]
    assert "afterok:RECOVERY_ARRAY_JOB_ID" in commands[1]
    assert "DLC_PHASE=aggregate" in commands[1]


def test_manifest_production_submission_rejects_dirty_tracked_checkout(tmp_path):
    checkout = tmp_path / "dirty_checkout"
    run_headless = checkout / "run_headless"
    run_headless.mkdir(parents=True)
    marker = run_headless / "build_parked_dlc_manifest.py"
    marker.write_text("# marker\n", encoding="utf-8")
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "add", str(marker)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=VorLap Test",
            "-c",
            "user.email=vorlap-test@example.invalid",
            "commit",
            "-m",
            "test fixture",
        ],
        check=True,
        capture_output=True,
    )
    marker.write_text("# tracked dirty marker\n", encoding="utf-8")
    manifest_path = write_json(tmp_path / "manifest.json", build_manifest(matrix_spec(tmp_path)))
    script = Path(__file__).resolve().parents[1] / "run_headless" / "submit_parked_dlc_manifest.sj"
    env = os.environ.copy()
    env.update(
        {
            "DLC_MANIFEST": str(manifest_path),
            "SUBMIT_DLC_MANIFEST": "1",
            "DLC_QBLADE_TIME": "20:00:00",
            "DLC_QBLADE_PARTITION": "batch",
            "PYTHON_EXE": sys.executable,
            "VORLAP_ROOT": str(checkout),
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "Production submission requires a clean tracked checkout" in completed.stderr


def test_manifest_production_submission_allows_known_cluster_runtime_artifacts(tmp_path):
    checkout = tmp_path / "runtime_dirty_checkout"
    run_headless = checkout / "run_headless"
    marker = run_headless / "build_parked_dlc_manifest.py"
    xfoil = run_headless / "QBladeCE_2.0.9.7" / "Binaries" / "XFoil"
    bridge = (
        run_headless
        / "QBladeCE_2.0.9.7"
        / "ControllerFiles"
        / "libvorlap_qblade_bridge.so"
    )
    for path in (marker, xfoil, bridge):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=VorLap Test",
            "-c",
            "user.email=vorlap-test@example.invalid",
            "commit",
            "-m",
            "test fixture",
        ],
        check=True,
        capture_output=True,
    )
    xfoil.write_text("cluster-built xfoil\n", encoding="utf-8")
    bridge.write_text("cluster-built bridge\n", encoding="utf-8")
    manifest_path = write_json(tmp_path / "manifest.json", build_manifest(matrix_spec(tmp_path)))
    script = Path(__file__).resolve().parents[1] / "run_headless" / "submit_parked_dlc_manifest.sj"
    env = os.environ.copy()
    env.update(
        {
            "DLC_MANIFEST": str(manifest_path),
            "SUBMIT_DLC_MANIFEST": "1",
            "PYTHON_EXE": sys.executable,
            "VORLAP_ROOT": str(checkout),
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Allowing known cluster-built tracked runtime artifacts" in completed.stderr
    assert "Production submission requires a clean tracked checkout" not in completed.stderr


def test_manifest_submission_array_task_routes_selected_row(tmp_path):
    manifest_path = write_json(tmp_path / "manifest.json", build_manifest(matrix_spec(tmp_path)))
    script = Path(__file__).resolve().parents[1] / "run_headless" / "submit_parked_dlc_manifest.sj"
    env = os.environ.copy()
    env.update(
        {
            "DLC_MANIFEST": str(manifest_path),
            "DLC_PHASE": "qblade",
            "DLC_PARENT_DRY_RUN": "1",
            "SLURM_ARRAY_TASK_ID": "12",
            "PYTHON_EXE": sys.executable,
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "RUN_PAPER_PREFLIGHT": "0",
        }
    )

    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    first_line = json.loads(completed.stdout.splitlines()[0])
    assert first_line["row_id"] == "dlc6p1_U50p00ms_s5533456"
    assert first_line["phase"] == "qblade"


def test_cross_row_aggregation_requires_audited_rows_and_applies_seed_weighting(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path))
    manifest_path = write_json(tmp_path / "manifest.json", manifest)
    qblade_root = tmp_path / "qblade"
    qblade_root.mkdir()
    (qblade_root / "libQBlade.so").write_bytes(b"qblade-library")
    controller_dir = qblade_root / "ControllerFiles"
    controller_dir.mkdir()
    (controller_dir / "libvorlap_qblade_bridge.so").write_bytes(b"bridge-library")
    fake_turbsim = tmp_path / "turbsim"
    fake_turbsim.write_bytes(b"turbsim-executable")
    fake_turbsim.chmod(0o755)
    provenance_env = {
        "QBLADE_ROOT": str(qblade_root),
        "TURBSIM_EXE": str(fake_turbsim),
        "PYTHON_EXE": sys.executable,
        "OMP_NUM_THREADS": "16",
        "CASE_WORKERS": "1",
        "OPENFAST_SOURCE_REF": "test-openfast-source",
        "QBLADE_RESULT_FILTER": manifest["execution_settings"]["result_filter_path"],
    }
    repo_root = Path(__file__).resolve().parents[1]
    for index, row in enumerate(manifest["rows"]):
        run_root = Path(row["run_root"])
        wind_file = Path(row["wind_file"])
        wind_file.parent.mkdir(parents=True, exist_ok=True)
        wind_file.write_bytes(f"wind-{index}".encode())
        write_json(wind_record_path(row), build_wind_record(row, manifest))
        write_json(
            run_root / "dlc_manifest_row.json",
            {
                "schema_version": 1,
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "row_index": index,
                "row": row,
            },
        )
        write_json(
            run_root / "audits" / "results_audit.json",
            {"passed": True, "issues": []},
        )
        for phase in ("wind", "qblade"):
            write_json(
                run_root / f"execution_environment_{phase}.json",
                build_execution_provenance(
                    manifest=manifest,
                    row=row,
                    row_index=index,
                    phase=phase,
                    environment=provenance_env,
                    vorlap_root=repo_root,
                ),
            )
        postprocess = run_root / "postprocess"
        postprocess.mkdir(parents=True, exist_ok=True)
        baseline_del = 10.0 + index
        (postprocess / "source_coverage.csv").write_text(
            "re_min,re_lower_bound_warning_count\n10000,0\n",
            encoding="utf-8",
        )
        (postprocess / "signed_del_comparison.csv").write_text(
            "case,wind_speed_mps,azimuth_deg,position,quantity,axis,unit,"
            "channel_baseline,channel_vorlap,signed_del_baseline,signed_del_vorlap,"
            "equivalent_cycles_baseline,equivalent_cycles_vorlap,"
            "smoke_only,postprocess_warnings\n"
            f"case,{row['wind_speed_mps']},0,0.23,moment,X,Nm,base,vorlap,"
            f"{baseline_del},{2.0 * baseline_del},600,600,False,\n",
            encoding="utf-8",
        )
        (postprocess / "extreme_channel_metrics.csv").write_text(
            "scenario,case,wind_speed_mps,azimuth_deg,blade,position,quantity,axis,"
            "unit,channel,min,max,max_abs,smoke_only,postprocess_warnings\n"
            f"vorlap,case,{row['wind_speed_mps']},0,BLD_1,0.23,moment,X,Nm,mx,"
            f"-{100 + index},{100 + index},{100 + index},False,\n",
            encoding="utf-8",
        )

    script = Path(__file__).resolve().parents[1] / "run_headless" / "aggregate_parked_dlc_manifest.py"
    out_dir = tmp_path / "aggregate"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
            "--wohler-exponent",
            "10",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    summary = json.loads((out_dir / "aggregation_summary.json").read_text())
    weighted = list(csv.DictReader((out_dir / "fatigue_weighted_signed_del.csv").open()))
    conditions = list(csv.DictReader((out_dir / "fatigue_seed_aggregates.csv").open()))
    ultimate = list(csv.DictReader((out_dir / "ultimate_envelope.csv").open()))

    assert summary["passed"] is True
    assert summary["row_count"] == 13
    assert float(weighted[0]["ratio_vorlap_to_baseline"]) == pytest.approx(2.0)
    assert float(conditions[0]["paired_ratio_seed_mean"]) == pytest.approx(2.0)
    assert int(conditions[0]["paired_ratio_finite_seed_count"]) == 6
    assert float(conditions[0]["paired_ratio_seed_min"]) == pytest.approx(2.0)
    assert float(conditions[0]["paired_ratio_seed_max"]) == pytest.approx(2.0)
    assert float(conditions[0]["paired_ratio_seed_sample_std"]) == pytest.approx(0.0)
    assert ultimate[0]["critical_row_id"] == manifest["rows"][-1]["row_id"]


def test_wind_import_smoke_profile_selects_one_high_speed_row(tmp_path):
    script = Path(__file__).resolve().parents[1] / "run_headless" / "submit_dlc_wind_import_smoke.sj"
    spool_dir = tmp_path / "slurm-spool"
    spool_dir.mkdir()
    spooled_script = spool_dir / script.name
    spooled_script.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
    spooled_script.chmod(0o755)
    run_root = tmp_path / "wind_smoke"
    wind_root = tmp_path / "winds"
    env = os.environ.copy()
    env.update(
        {
            "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_ARRAY_TASK_MAX": "1",
            "SLURM_CPUS_PER_TASK": "16",
            "SLURM_SUBMIT_DIR": str(script.parents[1]),
            "DRY_RUN_ONLY": "1",
            "RUN_ROOT": str(run_root),
            "WIND_SMOKE_FIELD_ROOT": str(wind_root),
            "PYTHON_EXE": sys.executable,
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
        }
    )

    subprocess.run(
        ["bash", str(spooled_script)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
        cwd=tmp_path,
    )
    settings = json.loads((run_root / "campaign_settings.json").read_text())
    assert settings["wind_speeds"] == [50.0]
    assert settings["duration_s"] == 10.0
    assert settings["qblade_steps_override"] == 20
    assert settings["scenario_names"] == ["baseline"]
    assert settings["generate_turbsim"] is True
    assert settings["turbsim_seed_base"] == 6633457
    assert settings["threads"] == 16
    assert settings["wind_name"].endswith("U50p00ms/wind_import_U50p00ms_s6633457.bts")
    assert json.loads((run_root / "audits" / "plan_audit.json").read_text())["passed"] is True


def test_execution_provenance_hashes_runtime_and_detects_tampering(tmp_path):
    manifest = build_manifest(matrix_spec(tmp_path))
    row = manifest["rows"][0]
    qblade_root = tmp_path / "qblade"
    qblade_root.mkdir()
    (qblade_root / "libQBlade.so").write_bytes(b"qblade")
    controllers = qblade_root / "ControllerFiles"
    controllers.mkdir()
    (controllers / "libvorlap_qblade_bridge.so").write_bytes(b"bridge")
    turbsim = tmp_path / "turbsim"
    turbsim.write_bytes(b"turbsim")
    turbsim.chmod(0o755)
    env = {
        "QBLADE_ROOT": str(qblade_root),
        "TURBSIM_EXE": str(turbsim),
        "PYTHON_EXE": sys.executable,
        "OMP_NUM_THREADS": "16",
        "QBLADE_WAKE_PROFILE": "parked-reduced-v2",
        "QBLADE_RESULT_FILTER": manifest["execution_settings"]["result_filter_path"],
        "LOADEDMODULES": "python:cmake",
    }
    payload = build_execution_provenance(
        manifest=manifest,
        row=row,
        row_index=0,
        phase="qblade",
        environment=env,
        vorlap_root=Path(__file__).resolve().parents[1],
    )
    validate_execution_provenance(
        payload,
        manifest=manifest,
        row=row,
        row_index=0,
        phase="qblade",
    )

    assert payload["modules"]["loaded"] == ["python", "cmake"]
    assert len(payload["qblade"]["library"]["sha256"]) == 64
    assert (
        payload["qblade"]["result_filter"]["sha256"]
        == manifest["execution_settings"]["result_filter_sha256"]
    )
    assert len(payload["bridge_library"]["sha256"]) == 64
    assert (
        payload["source_data"]["manifest_airfoil_source_sha256"]
        == manifest["basis"]["airfoil_source_sha256"]
    )
    assert payload["source_data"]["runtime_airfoil_files"]
    tampered = deepcopy(payload)
    tampered["runtime_environment"]["OMP_NUM_THREADS"] = "8"
    with pytest.raises(DLCManifestError, match="digest is invalid"):
        validate_execution_provenance(
            tampered,
            manifest=manifest,
            row=row,
            row_index=0,
            phase="qblade",
        )
