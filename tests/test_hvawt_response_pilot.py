import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tarfile

import pytest

from vorlap import hvawt_response_pilot as pilot
from vorlap.qblade_source_coverage import (
    HVAWT_V8_COMPONENT_NODE_COUNTS,
    SOURCE_COVERAGE_COLUMNS,
    SOURCE_COVERAGE_INTEGER_COLUMNS,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "run_headless" / "config"


def _spec() -> dict:
    return json.loads(
        (CONFIG / "hvawt_response_campaign_spec.json").read_text(encoding="utf-8")
    )


def _model() -> dict:
    return json.loads(
        (CONFIG / "hvawt_response_model_v8_complete.json").read_text(encoding="utf-8")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_historical_unified_hvawt_receipt_binds_every_named_source_file():
    # This receipt records the August execution, not the evolving current code.
    historical_root = os.environ.get("VORLAP_HISTORICAL_SOURCE_ROOT")
    if not historical_root:
        pytest.skip("Historical source snapshot required; set VORLAP_HISTORICAL_SOURCE_ROOT.")
    source_root = Path(historical_root)
    receipt = json.loads(
        (CONFIG / "unified_viv_gain_hvawt_parity_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    paths = receipt["source_code_paths"]
    hashes = receipt["source_code"]
    assert set(paths) == set(hashes)
    assert paths["campaign_test_sha256"] == "tests/test_hvawt_response_pilot.py"
    for name, relative_path in paths.items():
        assert hashes[name] == _sha256(source_root / relative_path), name


def _runtime_identity() -> dict:
    return {
        "schema": pilot.FLIGHT_RUNTIME_IDENTITY_SCHEMA,
        "qblade_library": {
            "sha256": (
                "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
            ),
            "size_bytes": 18365008,
        },
        "bridge_library": {"sha256": "c" * 64, "size_bytes": 65512},
        "vorlap_python_runtime_fingerprint": "d" * 64,
    }


def _valid_source_and_velocity_receipts(total_steps: int) -> tuple[list[dict], dict]:
    rows = []
    for component, node_count in HVAWT_V8_COMPONENT_NODE_COUNTS.items():
        row = {name: 0.0 for name in SOURCE_COVERAGE_COLUMNS}
        row.update(
            {
                "component": component,
                "airfoil_id": "NACA0018:test",
                "reynolds_model": "replicated_reynolds_independent",
            }
        )
        for name in SOURCE_COVERAGE_INTEGER_COLUMNS:
            row[name] = 0
        row["samples"] = total_steps
        row["node_samples"] = total_steps * node_count
        row["nyquist_hz"] = 10.0
        row["nyquist_ratio"] = 1.0e12
        rows.append(row)
    axis = {
        "samples": total_steps * 39,
        "min": -1.0,
        "max": 1.0,
        "max_abs": 1.0,
        "all_zero": False,
    }
    velocity = {
        "velocity_channel_count": 117,
        "axis_stats": {name: dict(axis) for name in ("X", "Y", "Z")},
        "channel_stats": {
            f"{name}_g Vel. node {index:02d} [m/s]": {
                "samples": total_steps,
                "min": -1.0,
                "max": 1.0,
                "max_abs": 1.0,
                "all_zero": False,
            }
            for name in ("X", "Y", "Z")
            for index in range(39)
        },
        "all_zero_axes": [],
        "all_zero_channels": [],
    }
    return rows, velocity


def test_geometry_blended_airfoil_runtime_view_is_exactly_bound():
    model = _model()
    calibration = CONFIG / "geometry_blended_airfoil_calibration.json"
    parent = ROOT / "examples/viv_geometry_blended/geometry_blended_calibration.json"

    assert model["schema_version"] == 3
    assert model["type"] == "hvawt_airfoil_engineering_lockin_hybrid"
    assert model["engineering_lockin"]["calibration_sha256"] == _sha256(calibration)
    assert _sha256(calibration) == _sha256(
        ROOT / "examples/viv_geometry_blended/airfoil_branch_calibration.json"
    )
    assert _spec()["model"]["parent_calibration_sha256"] == _sha256(parent)
    assert model["kinematics"]["maximum_azimuth_rate_error_radps"] == pytest.approx(
        0.001 * 2.0 * math.pi / 60.0,
        rel=0.0,
        abs=1.0e-18,
    )


def test_frozen_full_and_preflight_matrices_and_native_observer_are_exact(tmp_path):
    spec = _spec()
    pilot.validate_campaign_spec(spec)
    pilot.validate_response_model(_model(), spec)

    cases = pilot.build_pilot_cases(tmp_path, spec)

    assert len(cases) == 20
    assert sorted({row["wind_speed_mps"] for row in cases}) == list(
        pilot.PILOT_WINDS_MPS
    )
    assert sorted({row["timestep_s"] for row in cases}, reverse=True) == list(
        pilot.PILOT_TIMESTEPS_S
    )
    assert {row["scenario"] for row in cases} == set(pilot.SCENARIOS)
    assert {row["total_steps"] for row in cases} == {88000}
    assert all(row["rotor_speed_rpm"] == 1.0 for row in cases)
    assert all(row["duration_s"] == 660.0 for row in cases)
    assert all(row["assessment_tail_duration_s"] == 600.0 for row in cases)

    preflight_spec = pilot._effective_campaign_spec(
        spec, campaign_kind=pilot.PREFLIGHT_CAMPAIGN_KIND
    )
    pilot.validate_campaign_spec(preflight_spec)
    preflight_cases = pilot.build_pilot_cases(tmp_path, preflight_spec)
    assert len(preflight_cases) == 2
    assert {row["scenario"] for row in preflight_cases} == set(pilot.SCENARIOS)
    assert {row["wind_speed_mps"] for row in preflight_cases} == {8.25}
    assert {row["timestep_s"] for row in preflight_cases} == {0.0075}
    assert {row["total_steps"] for row in preflight_cases} == {16000}
    assert all(row["duration_s"] == 120.0 for row in preflight_cases)
    assert all(row["assessment_tail_duration_s"] == 60.0 for row in preflight_cases)
    assert (
        preflight_spec["pilot"]["static_spectrum_prepass"]["fingerprint"]
        == pilot.PREFLIGHT_PREPASS_FINGERPRINT
    )

    two_speed_spec = pilot._effective_campaign_spec(
        spec, campaign_kind=pilot.TWO_SPEED_TEST_CAMPAIGN_KIND
    )
    pilot.validate_campaign_spec(two_speed_spec)
    two_speed_cases = pilot.build_pilot_cases(tmp_path, two_speed_spec)
    assert len(two_speed_cases) == 4
    assert {row["scenario"] for row in two_speed_cases} == set(
        pilot.TWO_SPEED_TEST_SCENARIOS
    )
    assert {row["wind_speed_mps"] for row in two_speed_cases} == {8.25, 12.2}
    assert {row["timestep_s"] for row in two_speed_cases} == {0.0075}
    assert {row["total_steps"] for row in two_speed_cases} == {16000}
    assert (
        two_speed_spec["pilot"]["static_spectrum_prepass"]["fingerprint"]
        == pilot.TWO_SPEED_TEST_PREPASS_FINGERPRINT
    )

    observer = pilot._zero_gain_observer(_model())
    assert observer["force_sources"] == _model()["force_sources"]
    assert observer["engineering_lockin"]["response_force_scale"] == 0.0
    assert observer["engineering_lockin"]["residual_source_scale"] == 0.0
    assert _model()["engineering_lockin"]["response_force_scale"] == 1.0
    assert _model()["engineering_lockin"]["residual_source_scale"] == 1.0


def test_spec_freezes_pragmatic_amplitude_and_no_double_count_gates():
    spec = _spec()
    acceptance = spec["acceptance"]

    assert acceptance["maximum_peak_amplitude_error_fraction_of_target"] == 0.3
    assert (
        acceptance["peak_amplitude_absolute_allowance_fraction_of_reference_length"]
        == 0.0
    )
    assert acceptance["inactive_closure_only_amplitude_reference_m"] == 0.0005
    assert acceptance["stationarity_tail_split_fraction"] == 0.5
    assert acceptance["maximum_modal_peak_frequency_relative_error"] == 0.1
    assert "maximum_response_limit_events" not in acceptance
    assert acceptance["required_legacy_vorlap_force_scale"] == 0.0
    assert acceptance["required_rpm_tolerance"] == 0.001
    assert acceptance["native_qblade_aerodynamics_required"] is True
    assert acceptance["required_external_viv_mode"] == "response_closure"
    assert (
        acceptance["required_hybrid_response_shape_against_all_selected_wind_targets"]
        is False
    )
    assert acceptance["required_observed_modal_peak_wind_agreement"] is False
    assert acceptance["closure_only_amplitude_targets_are_diagnostic"] is True
    assert acceptance["required_hybrid_residual_source_scale"] == 1.0
    model = _model()
    assert model["max_abs_total_nodal_force_n"] is None
    assert model["kinematics"]["maximum_abs_basis_rpm_error_rpm"] == 0.001
    assert model["kinematics"]["maximum_azimuth_rate_error_radps"] == pytest.approx(
        0.001 * 2.0 * math.pi / 60.0
    )
    assert (
        acceptance[
            "maximum_inactive_tail_half_peak_amplitude_absolute_difference_m"
        ]
        == 0.0005
    )
    selected_targets = pilot._selected_wind_amplitude_targets(spec["pilot"])
    assert len(selected_targets) == 20
    assert selected_targets[(3.0, "6")] == 0.0
    assert selected_targets[(3.0, "12")] == 0.0
    assert selected_targets[(8.25, "6")] == pytest.approx(0.006363532058)
    assert selected_targets[(12.2, "12")] == pytest.approx(0.00529808019)
    assert spec["status"] == "ready_for_native_x86_full_submission"
    assert spec["execution"] == {
        "architecture": "native_linux_x86_64",
        "allocated_physical_cores_per_array_task": 112,
        "array_pair_count": 10,
        "maximum_concurrent_cases_per_node": 1,
        "numa_domains": 8,
        "node_physical_cores": 112,
        "qblade_library_sha256": (
            "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
        ),
        "qblade_threads_per_case": 112,
        "result_filter_channel_count": 26,
        "result_filter_role": "matched_26_channel_parked_fatigue_load_export",
        "result_filter_sha256": pilot.RESULT_FILTER_SHA256,
        "submission_entrypoint": "run_headless/submit_hvawt_response_pilot.sj",
    }
    assert spec["full_rerun"]["execution_authorized"] is False
    assert spec["full_rerun"]["execution_status"] == "prepared_waiting_for_pilot_pass"
    assert spec["full_rerun"]["expected_new_runs_if_controls_are_rerun"] == 102
    assert tuple(spec["full_rerun"]["scenarios"]) == pilot.FULL_RERUN_SCENARIOS


def test_spec_binds_primary_multi_condition_turbine_validation_and_envelope():
    spec = _spec()
    declaration = spec["cross_profile_turbine_validation"]
    artifact = ROOT / declaration["artifact"]
    assessment = json.loads(artifact.read_text(encoding="utf-8"))

    assert declaration["evidence_role"] == "primary_heldout_cross_profile_validation"
    assert declaration["source_fidelity"] == (
        "ellipsys3d_hawc2_high_fidelity_wind_turbine_fsi"
    )
    assert declaration["heldout_geometry"] == "COMB"
    assert declaration["operating_condition_count"] == 11
    assert declaration["resolved_half_peak_viv_point_count"] == 2
    assert declaration["artifact_sha256"] == _sha256(artifact)
    assert assessment["matrix"]["all_holdout_rows_reported"] is True
    assert assessment["gates"]["all"] is True
    pilot._validate_cross_profile_turbine_validation(declaration, assessment)

    envelope = spec["conservative_response_envelope"]
    assert envelope["covers_all_11_heldout_conditions"] is True
    assert envelope["multiplicative_factor"] == 1.3
    assert envelope["additive_floor_over_signed_saturation"] == pytest.approx(
        0.10265640699352364
    )
    assert envelope["fatigue_validated"] is False


def test_cross_profile_turbine_validation_rejects_changed_holdout_metric():
    spec = _spec()
    declaration = spec["cross_profile_turbine_validation"]
    artifact = ROOT / declaration["artifact"]
    assessment = json.loads(artifact.read_text(encoding="utf-8"))
    assessment["metrics"]["curve_nrmse_by_target_peak"] = 0.29

    with pytest.raises(pilot.HVAWTResponsePilotError, match="metric"):
        pilot._validate_cross_profile_turbine_validation(declaration, assessment)


def test_spec_rejects_non_full_flight_concurrency():
    spec = _spec()
    spec["execution"]["maximum_concurrent_cases_per_node"] = 2

    with pytest.raises(
        pilot.HVAWTResponsePilotError, match="maximum_concurrent_cases_per_node"
    ):
        pilot.validate_campaign_spec(spec)


def test_native_prep_and_submit_scripts_use_one_high_utilization_flight_node():
    prep = ROOT / "run_headless" / "prepare_hvawt_response_pilot_native.sh"
    submit = ROOT / "run_headless" / "submit_hvawt_response_pilot.sj"
    preflight = ROOT / "run_headless" / "submit_hvawt_response_preflight.sj"
    analysis = ROOT / "run_headless" / "submit_hvawt_response_analysis.sj"
    two_speed = ROOT / "run_headless" / "submit_hvawt_two_speed_120s_test.sj"
    prep_text = prep.read_text(encoding="utf-8")
    submit_text = submit.read_text(encoding="utf-8")
    preflight_text = preflight.read_text(encoding="utf-8")
    analysis_text = analysis.read_text(encoding="utf-8")
    two_speed_text = two_speed.read_text(encoding="utf-8")

    for script in (prep, submit, preflight, analysis, two_speed):
        subprocess.run(["bash", "-n", str(script)], check=True)
    assert "hvawt_pilot_hpc_inputs" in prep_text
    assert "U05p00ms_source_case.tar.gz" in prep_text
    assert "CALIBRATED_STATIC_SPECTRUM" in prep_text
    assert (
        "64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369" in prep_text
    )
    assert (
        "6f6daf91d026087bb5f5acec6b313c7cead1a70faa1deded1819e15c60d7d403" in prep_text
    )
    assert 'cp "${CALIBRATED_STATIC_SPECTRUM}" "${SOURCE_STATIC_SPECTRUM}"' in prep_text
    assert "is_known_cluster_runtime_artifact" in prep_text
    assert "Binaries/XFoil" in prep_text
    assert "ControllerFiles/libvorlap_qblade_bridge.so" in prep_text
    assert "outside the two known cluster-built runtime artifacts" in prep_text
    contract = json.loads(
        (CONFIG / "hvawt_modal_decay_source_contract.json").read_text(encoding="utf-8")
    )
    assert (
        contract["files_sha256"][
            "10m_HVAWT_reference_turbine/VorLapAirfoils/NACA0018.h5"
        ]
        == "64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369"
    )
    sums = (CONFIG / "hvawt_pilot_hpc_inputs" / "SHA256SUMS").read_text(
        encoding="utf-8"
    )
    assert (
        "6f6daf91d026087bb5f5acec6b313c7cead1a70faa1deded1819e15c60d7d403  "
        "U05p00ms_source_case.tar.gz"
    ) in sums
    assert (
        "398bde198b8a3551cdc8469256be2d4b6cd8885ed1e003d93e7464cdf7b025e2" in prep_text
    )
    assert "hvawt_modal_damping_selection" in prep_text
    assert "hvawt_response_pilot.py generate" in prep_text
    assert "hvawt_response_pilot.py run" not in prep_text
    assert prep_text.count("hvawt_response_pilot.py bind-runtime") == 3
    assert '--qblade-root "${QBLADE_ROOT}"' in prep_text
    assert "142c186c9e43ec914219a9d0bdf7e554b35902e3" in prep_text
    assert "merge-base --is-ancestor" in prep_text
    assert "status --porcelain --untracked-files=no" in prep_text
    assert "--campaign-kind two_case_120s_preflight" in prep_text
    assert "--campaign-kind full_660s_resonance_sweep" in prep_text
    assert "#SBATCH --ntasks=1" in submit_text
    assert "#SBATCH --ntasks-per-node=1" in submit_text
    assert "#SBATCH --cpus-per-task" not in submit_text
    assert "#SBATCH --hint" not in submit_text
    assert "#SBATCH --exclusive" in submit_text
    assert "#SBATCH --array=0-19%20" in submit_text
    assert "#SBATCH --time=47:59:00" in submit_text
    assert "#SBATCH --reservation=flight-cldera" in submit_text
    assert 'THREADS="${SLURM_CPUS_ON_NODE:-}"' in submit_text
    assert '[[ "${THREADS}" == "112" ]]' in submit_text
    assert 'case_index="${SLURM_ARRAY_TASK_ID}"' in submit_text
    assert "first_case=" not in submit_text
    assert "module load anaconda3/2025.12-2" in submit_text
    assert "/etc/profile.d/modules.sh" in submit_text
    assert 'PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"' in submit_text
    assert "${QBLADE_ROOT}/Binaries/IPCBridge" in submit_text
    assert "VORLAP_QBLADE_LOADER_ENV=1" in submit_text
    assert "OMP_PLACES" not in submit_text
    assert "OMP_PROC_BIND" not in submit_text
    assert "sched_getaffinity" not in submit_text
    assert "lscpu" not in submit_text
    assert "SLURM_CPUS_ON_NODE" in submit_text
    assert "sha256sum" in submit_text
    assert (
        "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
        in submit_text
    )
    assert 'CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-24000}"' in submit_text
    assert '--checkpoint-every "${CHECKPOINT_EVERY}"' in submit_text
    assert "--resume" not in submit_text
    assert "bind-runtime" in submit_text
    for text in (submit_text, preflight_text, two_speed_text):
        assert "flight_qblade_runtime_identity.json" in text
    assert "--archive-partial-and-restart" in submit_text
    assert "#SBATCH --time=16:00:00" in preflight_text
    assert "#SBATCH --cpus-per-task" not in preflight_text
    assert "#SBATCH --hint" not in preflight_text
    assert "#SBATCH --array" not in preflight_text
    assert "for case_index in 0 1" in preflight_text
    assert "module load anaconda3/2025.12-2" in preflight_text
    assert "/etc/profile.d/modules.sh" in preflight_text
    assert 'THREADS="${SLURM_CPUS_ON_NODE:-}"' in preflight_text
    assert "OMP_PLACES" not in preflight_text
    assert "sched_getaffinity" not in preflight_text
    assert "lscpu" not in preflight_text
    assert "module load anaconda3/2025.12-2" in analysis_text
    assert "/etc/profile.d/modules.sh" in analysis_text
    assert "#SBATCH --reservation=flight-cldera" in analysis_text
    assert "--campaign-kind two_speed_120s_h0_complete_test" in prep_text
    assert "#SBATCH --array=0-3%4" in two_speed_text
    assert "#SBATCH --time=12:00:00" in two_speed_text
    assert '--case-index "${SLURM_ARRAY_TASK_ID}"' in two_speed_text
    assert 'CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-24000}"' in two_speed_text
    assert "Flight recovery saves must use the frozen 24,000-step interval" in two_speed_text
    assert 'VORLAP_HVAWT_PROFILE="${VORLAP_HVAWT_PROFILE:-1}"' in two_speed_text


def test_actual_aero_case_commands_allow_saves_but_forbid_resume(tmp_path, monkeypatch):
    case = {
        "case_id": "case00",
        "sim": "cases/case00/case.sim",
        "result_dir": "results/case00",
        "result_filter": "cases/case00/result.filter",
        "result_name": "case00_qblade_results",
    }
    monkeypatch.setattr(
        pilot,
        "validate_campaign",
        lambda _campaign: {
            "campaign_root": tmp_path,
            "manifest": {"cases": [case]},
        },
    )

    command = pilot.command_for_case(
        tmp_path,
        case_index=0,
        qblade_root=tmp_path / "qblade",
    )
    checkpoint_index = command.index("--checkpoint-every")
    assert command[checkpoint_index + 1] == "0"
    assert "--resume-from-checkpoint" not in command

    saved = pilot.command_for_case(
        tmp_path,
        case_index=0,
        qblade_root=tmp_path / "qblade",
        checkpoint_every=pilot.FLIGHT_RESTART_SAVE_INTERVAL_STEPS,
    )
    assert saved[saved.index("--checkpoint-every") + 1] == "24000"
    with pytest.raises(pilot.HVAWTResponsePilotError, match="continuation is not yet approved"):
        pilot.command_for_case(
            tmp_path,
            case_index=0,
            qblade_root=tmp_path / "qblade",
            resume=True,
        )


def test_actual_aero_case_classification_reuses_terminal_and_blocks_partial(
    tmp_path, monkeypatch
):
    case = {
        "case_id": "case00",
        "sim": "cases/case00/case.sim",
        "result_dir": "results/case00",
    }
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "spec.json").write_text(
        json.dumps(
            {
                "acceptance": {},
                "pilot": {
                    "campaign_kind": pilot.FULL_CAMPAIGN_KIND,
                    "lock_in_peak_wind_speed_by_mode_mps": {},
                    "predicted_peak_amplitude_target_by_mode_m": {},
                },
                "execution": {
                    "qblade_library_sha256": (
                        "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    (inputs / "model.json").write_text("{}\n", encoding="utf-8")
    manifest = {
        "fingerprint": "a" * 64,
        "cases": [case],
        "inputs": {
            "campaign_spec": {"path": "inputs/spec.json"},
            "response_model": {"path": "inputs/model.json"},
        },
    }
    monkeypatch.setattr(
        pilot,
        "validate_campaign",
        lambda _campaign: {"campaign_root": tmp_path, "manifest": manifest},
    )
    monkeypatch.setattr(
        pilot, "_load_flight_runtime_identity", lambda _root: _runtime_identity()
    )

    assert pilot.classify_case_execution(tmp_path, case_index=0)["status"] == "runnable"
    result_dir = tmp_path / "results/case00"
    result_dir.mkdir(parents=True)
    (result_dir / "partial.log").write_text("partial\n", encoding="utf-8")
    monkeypatch.setattr(
        pilot,
        "_case_result_or_error",
        lambda **_kwargs: {"status": "missing", "passed": False},
    )
    blocked = pilot.classify_case_execution(tmp_path, case_index=0)
    assert blocked["status"] == "blocked"
    assert "not_restartable" in blocked["reason"]

    monkeypatch.setattr(
        pilot,
        "_case_result_or_error",
        lambda **_kwargs: {
            "status": "complete",
            "passed": False,
            "execution_receipt_pass": True,
        },
    )
    complete = pilot.classify_case_execution(tmp_path, case_index=0)
    assert complete == {
        "case_id": "case00",
        "status": "complete",
        "reason": "exact_audited_terminal_result",
        "physics_gates_passed": False,
    }


def test_partial_attempt_is_archived_recoverably_before_clean_restart(
    tmp_path, monkeypatch
):
    case = {"case_id": "case00", "result_dir": "results/case00"}
    manifest = {"fingerprint": "a" * 64, "cases": [case]}
    monkeypatch.setattr(
        pilot,
        "validate_campaign",
        lambda _campaign: {"campaign_root": tmp_path, "manifest": manifest},
    )
    monkeypatch.setattr(
        pilot,
        "classify_case_execution",
        lambda *_args, **_kwargs: {
            "case_id": "case00",
            "status": "blocked",
            "reason": (
                "partial_or_invalid_actual_aero_artifacts_are_not_restartable:"
                "invalid_result:interrupted"
            ),
        },
    )
    result_dir = tmp_path / "results/case00"
    result_dir.mkdir(parents=True)
    (result_dir / "partial.log").write_text("preserve me\n", encoding="utf-8")

    archived = pilot.archive_incomplete_case_attempt(
        tmp_path, case_index=0, qblade_root=tmp_path / "qblade"
    )

    assert archived == tmp_path / "attempt_archive/case00/attempt_001"
    assert (archived / "partial.log").read_text(encoding="utf-8") == "preserve me\n"
    assert not result_dir.exists()
    receipt = json.loads(
        (
            tmp_path / "attempt_archive/case00/attempt_001_archive_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["campaign_fingerprint"] == "a" * 64
    assert receipt["archive_path"] == "attempt_archive/case00/attempt_001"


def test_execute_case_uses_one_thread_math_libraries_and_skips_exact_completion(
    tmp_path, monkeypatch
):
    case = {
        "case_id": "case00",
        "result_dir": "results/case00",
    }
    manifest = {"fingerprint": "a" * 64, "cases": [case]}
    monkeypatch.setattr(pilot, "bind_flight_runtime_identity", lambda *_a, **_k: {})
    monkeypatch.setattr(
        pilot,
        "validate_campaign",
        lambda _campaign: {"campaign_root": tmp_path, "manifest": manifest},
    )
    states = iter(
        [
            {
                "case_id": "case00",
                "status": "runnable",
                "reason": "fresh_case_no_prior_artifacts",
            }
        ]
    )
    monkeypatch.setattr(
        pilot, "classify_case_execution", lambda *_a, **_k: next(states)
    )
    monkeypatch.setattr(pilot, "command_for_case", lambda *_a, **_k: ["runner"])
    observed = {}

    def fake_run(command, *, check, env):
        observed.update({"command": command, "check": check, "env": env})

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)
    command = pilot.execute_case(
        tmp_path,
        case_index=0,
        qblade_root=tmp_path / "qblade",
        threads=4,
    )
    assert command == ["runner"]
    assert observed["env"]["OMP_NUM_THREADS"] == "4"
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert observed["env"][name] == "1"

    monkeypatch.setattr(
        pilot,
        "classify_case_execution",
        lambda *_a, **_k: {
            "case_id": "case00",
            "status": "complete",
            "reason": "exact_audited_terminal_result",
            "physics_gates_passed": False,
        },
    )
    monkeypatch.setattr(
        pilot.subprocess,
        "run",
        lambda *_a, **_k: pytest.fail("exact completion must not relaunch QBlade"),
    )
    assert (
        pilot.execute_case(
            tmp_path,
            case_index=0,
            qblade_root=tmp_path / "qblade",
            threads=4,
        )
        == []
    )


def test_case_staging_replaces_archived_naca0018_and_preserves_unused_ffa(tmp_path):
    case_dir = tmp_path / "case"
    turbine_dir = case_dir / "10m_HVAWT_reference_turbine"
    airfoil_dir = turbine_dir / "VorLapAirfoils"
    airfoil_dir.mkdir(parents=True)
    sim = case_dir / "case.sim"
    sim.write_text(
        "10m_HVAWT_reference_turbine/case.trb TURBFILE\n",
        encoding="utf-8",
    )
    (turbine_dir / "case.trb").write_text("test turbine\n", encoding="utf-8")
    archived = airfoil_dir / "NACA0018.h5"
    archived.write_bytes(b"pre-calibration archived spectrum")
    unused_ffa = airfoil_dir / "ffa_w3_211.h5"
    unused_ffa.write_bytes(b"unused ffa spectrum")
    unused_identity = pilot._identity(unused_ffa)
    model_source = CONFIG / "hvawt_response_model_v8_complete.json"

    audit = pilot._stage_calibrated_static_spectrum(
        repository=ROOT,
        model_source=model_source,
        response_model=_model(),
        sim_path=sim,
        case_dir=case_dir,
    )

    assert audit["archived_destination_before_replacement"]["sha256"] != (
        _spec()["static_spectrum_sha256"]
    )
    assert pilot._identity(archived)["sha256"] == _spec()["static_spectrum_sha256"]
    assert pilot._identity(unused_ffa) == unused_identity
    assert audit["other_hdf5_sources_preserved"] == [
        {
            "path": "10m_HVAWT_reference_turbine/VorLapAirfoils/ffa_w3_211.h5",
            "sha256": unused_identity["sha256"],
            "size_bytes": unused_identity["size_bytes"],
        }
    ]


def test_rayleigh_source_requires_completed_interpretable_c20_selection(tmp_path):
    campaign = tmp_path / "decay_campaign"
    case = campaign / "cases" / "m12_c20_baseline"
    case.mkdir(parents=True)
    (case / "selected.sim").write_text("selected C20 case\n", encoding="utf-8")
    manifest = {
        "fingerprint": "b" * 64,
        "cases": [
            {
                "case_id": "m12_c20_baseline",
                "variant": "C20",
                "role": "baseline",
                "sim_relative_path": "selected.sim",
            }
        ],
    }
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    analysis = {
        "schema": pilot.RAYLEIGH_ANALYSIS_SCHEMA,
        "status": "complete_interpretable",
        "campaign_root": str(campaign),
        "campaign_fingerprint": manifest["fingerprint"],
        "classification": {
            "selection_allowed": True,
            "selected_unambiguous_substructure_encoding": "C20",
            "damping_source": "station_vector_dominant",
            "effective_c20_damping_ratio": 0.02,
        },
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")

    source, evidence = pilot._rayleigh_source_case(analysis_path)

    assert source == case.resolve()
    assert evidence["selected_variant"] == "C20"
    analysis["classification"]["selection_allowed"] = False
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    with pytest.raises(pilot.HVAWTResponsePilotError, match="requires a complete"):
        pilot._rayleigh_source_case(analysis_path)


def test_rayleigh_source_accepts_source_defined_corrected_c20_selection(tmp_path):
    campaign = tmp_path / "decay_campaign"
    case = campaign / "cases" / "m12_c20_baseline"
    case.mkdir(parents=True)
    (case / "selected.sim").write_text("selected C20 case\n", encoding="utf-8")
    manifest = {
        "fingerprint": "c" * 64,
        "cases": [
            {
                "case_id": "m12_c20_baseline",
                "variant": "C20",
                "role": "baseline",
                "sim_relative_path": "selected.sim",
            }
        ],
    }
    (campaign / "campaign_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    selection = {
        "schema": pilot.RAYLEIGH_SELECTION_SCHEMA,
        "status": "selected",
        "selection_basis": pilot.RAYLEIGH_DOCUMENTED_SELECTION_BASIS,
        "selected_structural_encoding": "C20",
        "parser_semantics": "fixed_20th_numeric_token_is_damping",
        "damping_source": "distributed_station_beta_with_normalized_scalar_fallback",
        "source_case": {
            "campaign_root": str(campaign),
            "campaign_fingerprint": manifest["fingerprint"],
            "selected_case_id": "m12_c20_baseline",
            "selected_variant": "C20",
            "selected_sim_relative_path": "selected.sim",
        },
        "modes": [
            {
                "tracked_mode": mode,
                **values,
                "qblade_effective_decay_damping_ratio": None,
            }
            for mode, values in pilot.DOCUMENTED_MODE_DAMPING.items()
        ],
    }
    selection["fingerprint"] = pilot.canonical_fingerprint(selection)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    source, evidence = pilot._rayleigh_source_case(selection_path)

    assert source == case.resolve()
    assert evidence["selection_basis"] == pilot.RAYLEIGH_DOCUMENTED_SELECTION_BASIS
    assert evidence["effective_c20_damping_ratio"] is None
    assert [row["tracked_mode"] for row in evidence["modes"]] == [6, 12]

    selection["modes"][1]["rayleigh_energy_projected_damping_ratio"] = 0.5
    selection.pop("fingerprint")
    selection["fingerprint"] = pilot.canonical_fingerprint(selection)
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(pilot.HVAWTResponsePilotError, match="frozen projection"):
        pilot._rayleigh_source_case(selection_path)


def test_engineering_rayleigh_normalization_replaces_only_scalar_fallbacks(tmp_path):
    sub_relative = "model/Structure/substructure.str"
    scalar_relative = "model/Structure/tower.str"
    sub = tmp_path / sub_relative
    scalar = tmp_path / scalar_relative
    sub.parent.mkdir(parents=True)
    sub.write_text(
        "SUBELEMENTS\n"
        "ID RHOA EA EIx EIy GJ GA PITCH KSX KSY RGX RGY XCM YCM XE YE XS YS DIA_[m] DAMP[-]\n"
        "1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1.000 0.02\n"
        "SUBMEMBERS\n",
        encoding="utf-8",
    )
    scalar.write_text("0.02 RAYLEIGHDMP\n", encoding="utf-8")
    evidence = {
        "selection_basis": pilot.RAYLEIGH_DOCUMENTED_SELECTION_BASIS,
        "engineering_scalar_beta_s": 0.0011,
        "selected_substructure_relative_path": sub_relative,
        "normalized_scalar_relative_paths": [scalar_relative],
    }

    audit = pilot._apply_engineering_rayleigh_inputs(tmp_path, evidence)

    assert audit["selected_scalar_beta_s"] == pytest.approx(0.0011)
    assert sub.read_text().splitlines()[2].split()[-1] == "0.0011"
    assert scalar.read_text().split()[0] == "0.0011"


def _write_synthetic_history(
    path: Path, *, dt: float = 0.01, duration: float = 8.0
) -> None:
    fields = [
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
    ]
    for mode in (6, 12):
        fields.extend(
            [
                f"response_mode{mode}_raw_complex_projection_real_mps",
                f"response_mode{mode}_raw_complex_projection_imag_mps",
                f"response_mode{mode}_filtered_qdot_mps",
                f"response_mode{mode}_amplitude_estimate_m",
                f"response_mode{mode}_complex_projection_leakage_ratio",
                f"response_mode{mode}_engineering_aerodynamic_to_structural_damping_ratio",
                f"response_mode{mode}_spectral_gain_max",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        count = int(round(duration / dt)) + 1
        for index in range(count):
            time_s = index * dt
            row = {
                "step_index": index,
                "time_s": time_s,
                "qblade_timestep": index,
                "response_model_type": "hvawt_airfoil_engineering_lockin_hybrid",
                "generated_max_abs_force_component_n": 4.0,
                "response_total_maximum_nodal_force_n": 6.0,
                "response_engineering_maximum_abs_nodal_force_n": 5.0,
                "response_spectral_source_config_fingerprint": "a" * 64,
                "response_spectral_residual_source_scale": 0.0,
                "response_spectral_applied_residual_maximum_nodal_force_n": 0.0,
                "response_spectral_applied_residual_force_power_w": 0.0,
                "response_spectral_source_energy_sum": 1.0,
                "response_spectral_residual_energy_sum": 0.75,
                "response_spectral_applied_residual_lift_energy_sum": 0.75,
                "response_spectral_truncated_residual_lift_energy_sum": 0.0,
                "response_spectral_truncated_residual_drag_energy_sum": 0.0,
                "response_spectral_captured_energy_sum": 0.25,
                "response_spectral_maximum_energy_closure_error": 0.0,
            }
            for mode, frequency in ((6, 2.5), (12, 4.5)):
                modal_velocity = math.sin(2.0 * math.pi * frequency * time_s)
                row.update(
                    {
                        f"response_mode{mode}_raw_complex_projection_real_mps": (
                            modal_velocity
                        ),
                        f"response_mode{mode}_raw_complex_projection_imag_mps": (
                            0.01 * modal_velocity
                        ),
                        f"response_mode{mode}_filtered_qdot_mps": modal_velocity,
                        f"response_mode{mode}_amplitude_estimate_m": 0.148,
                        f"response_mode{mode}_complex_projection_leakage_ratio": 0.01,
                        f"response_mode{mode}_engineering_aerodynamic_to_structural_damping_ratio": 1.0,
                        f"response_mode{mode}_spectral_gain_max": 1.0,
                    }
                )
            writer.writerow(row)


def test_history_analyzer_reports_tail_amplitude_frequency_stationarity_and_limits(
    tmp_path,
):
    history = tmp_path / "history.csv"
    _write_synthetic_history(history)

    metrics = pilot.analyze_history(history, tail_start_s=4.0, tail_duration_s=4.0)

    assert metrics["rows"] == 801
    assert metrics["reported_timestep_s"] == pytest.approx(0.01, abs=1.0e-12)
    assert metrics["maximum_generated_force_component_n"] == 4.0
    assert metrics["maximum_response_nodal_force_n"] == 5.0
    assert metrics["maximum_total_supplemental_nodal_force_n"] == 6.0
    assert metrics["maximum_spectral_energy_closure_error"] == 0.0
    assert metrics["maximum_applied_spectral_energy_closure_error"] == 0.0
    assert metrics["maximum_truncated_residual_lift_energy"] == 0.0
    assert metrics["spectral_residual_source_scale"] == 0.0
    assert metrics["modes"]["6"]["maximum_spectral_gain"] == 1.0
    assert metrics["modes"]["6"][
        "complex_projection_tail_l2_leakage_ratio"
    ] == pytest.approx(0.01)
    assert metrics["modes"]["6"][
        "maximum_instantaneous_complex_projection_leakage_ratio"
    ] == pytest.approx(0.01)
    assert metrics["modes"]["6"]["peak_amplitude_estimate_m"] == pytest.approx(0.148)
    assert metrics["modes"]["6"]["tail_half_peak_relative_difference"] == 0.0
    assert metrics["modes"]["6"]["dominant_frequency_hz"] == pytest.approx(
        2.5, abs=0.02
    )
    assert metrics["modes"]["12"]["dominant_frequency_hz"] == pytest.approx(
        4.5, abs=0.02
    )


def test_history_analyzer_uses_tail_l2_projection_leakage_not_singular_pointwise_max(
    tmp_path,
):
    history = tmp_path / "history.csv"
    _write_synthetic_history(history)
    rows = list(csv.DictReader(history.open(encoding="utf-8")))
    rows[500]["response_mode6_complex_projection_leakage_ratio"] = "1000000"
    with history.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = pilot.analyze_history(history, tail_start_s=4.0, tail_duration_s=4.0)

    mode6 = metrics["modes"]["6"]
    assert mode6["complex_projection_tail_l2_leakage_ratio"] == pytest.approx(
        0.01
    )
    assert mode6["maximum_instantaneous_complex_projection_leakage_ratio"] == (
        1_000_000.0
    )


def test_qblade_fatigue_result_audit_requires_exact_channels_rows_and_time(tmp_path):
    result_filter = CONFIG / "parked_fatigue_results.filter"
    assert pilot._identity(result_filter)["sha256"] == pilot.RESULT_FILTER_SHA256
    header = [
        "~".join(channel.split())
        for channel in pilot._result_filter_channels(result_filter)
    ]
    result = tmp_path / "result.txt"
    rows = []
    for index, time_s in enumerate((0.02, 0.03, 0.04)):
        values = [time_s, 1.0] + [float(index + 1)] * 24
        rows.append("\t".join(f"{value:.8g}" for value in values))
    result.write_text(
        "metadata one\nmetadata two\n"
        + "\t".join(header)
        + "\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )

    audit = pilot._audit_qblade_fatigue_result(
        result,
        result_filter=result_filter,
        timestep_s=0.01,
        total_steps=5,
        store_from_s=0.02,
    )

    assert audit["channel_count"] == 26
    assert audit["row_count"] == 3
    assert audit["first_time_s"] == pytest.approx(0.02)
    assert audit["last_time_s"] == pytest.approx(0.04)
    assert audit["maximum_abs_structural_load_channel_value"] == 3.0

    result.write_text(
        result.read_text(encoding="utf-8").replace("Time~[s]", "Bad"),
        encoding="utf-8",
    )
    with pytest.raises(pilot.HVAWTResponsePilotError, match="header differs"):
        pilot._audit_qblade_fatigue_result(
            result,
            result_filter=result_filter,
            timestep_s=0.01,
            total_steps=5,
            store_from_s=0.02,
        )


def test_prepass_evidence_quantization_removes_platform_blas_roundoff():
    assert pilot._stable_prepass_float(0.0033439420261528133) == (
        pilot._stable_prepass_float(0.003343942026152833)
    )
    assert pilot._stable_prepass_float(0.008447865093605126) == (
        pilot._stable_prepass_float(0.008447865093605128)
    )


def test_history_analyzer_rejects_nonsequential_or_nonfinite_rows(tmp_path):
    history = tmp_path / "history.csv"
    _write_synthetic_history(history)
    rows = list(csv.DictReader(history.open(encoding="utf-8")))
    rows[20]["step_index"] = "21"
    with history.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(pilot.HVAWTResponsePilotError, match="not sequential"):
        pilot.analyze_history(history, tail_start_s=4.0, tail_duration_s=4.0)

    _write_synthetic_history(history)
    rows = list(csv.DictReader(history.open(encoding="utf-8")))
    rows[20]["qblade_timestep"] = "19"
    with history.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(
        pilot.HVAWTResponsePilotError,
        match="counters are not sequential",
    ):
        pilot.analyze_history(history, tail_start_s=4.0, tail_duration_s=4.0)

    _write_synthetic_history(history)
    rows = list(csv.DictReader(history.open(encoding="utf-8")))
    rows[20]["time_s"] = str(float(rows[20]["time_s"]) + 0.001)
    with history.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(pilot.HVAWTResponsePilotError, match="not uniformly sampled"):
        pilot.analyze_history(history, tail_start_s=4.0, tail_duration_s=4.0)


def test_history_analyzer_rejects_predecessor_model_and_bad_energy_accounting(
    tmp_path,
):
    history = tmp_path / "history.csv"
    _write_synthetic_history(history)
    rows = list(csv.DictReader(history.open(encoding="utf-8")))
    rows[0]["response_model_type"] = "hvawt_two_mode_negative_damping"
    with history.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(pilot.HVAWTResponsePilotError, match="not the geometry-blended"):
        pilot.analyze_history(history, tail_start_s=4.0, tail_duration_s=4.0)

    _write_synthetic_history(history)
    rows = list(csv.DictReader(history.open(encoding="utf-8")))
    rows[0]["response_spectral_captured_energy_sum"] = "0.5"
    with history.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metrics = pilot.analyze_history(history, tail_start_s=4.0, tail_duration_s=4.0)
    assert metrics["maximum_spectral_energy_closure_error"] == pytest.approx(0.25)
    assert (
        metrics["maximum_spectral_energy_closure_error"]
        > metrics["spectral_energy_closure_tolerance"]
    )


def test_completed_geometry_blended_peak_case_uses_audited_modal_frequency(
    tmp_path, monkeypatch
):
    spec = _spec()
    model = _model()
    case = next(
        row
        for row in pilot.build_pilot_cases(tmp_path, spec)
        if row["scenario"] == pilot.SCENARIOS[1]
        and row["wind_speed_mps"] == 8.25
        and row["timestep_s"] == pilot.PILOT_TIMESTEPS_S[0]
    )
    case = dict(case)
    case["runtime_asset_fingerprint"] = "b" * 64
    case["result_filter"] = "result.filter"
    sim = tmp_path / case["sim"]
    result_dir = tmp_path / case["result_dir"]
    sim.parent.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    sim.write_text("completed test case\n", encoding="utf-8")
    campaign_fingerprint = "a" * 64
    metadata = {
        "schema": "vorlap-hvawt-response-pilot-case-metadata-v1",
        "campaign_fingerprint": campaign_fingerprint,
        "case": case,
    }
    (result_dir / "case_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    result_path = result_dir / f"{case['result_name']}.txt"
    result_path.write_text("Time [s]\n0.0\n", encoding="utf-8")
    summary_path = result_dir / f"{sim.stem}_run_summary.json"
    summary = {
        "coupling_mode": "sil-vorlap",
        "qblade_execution_mode": "step-loop",
        "requested_steps": case["total_steps"],
        "completed_steps": case["total_steps"],
        "failed": False,
        "stopped": False,
        "allow_partial": False,
        "exported_results": True,
        "sil_vorlap": True,
        "sil_vorlap_updates": case["total_steps"],
        "sil_vorlap_source_frequency_cap_hz": 0.5 / float(case["timestep_s"]),
        "sil_vorlap_history_mode": "compact",
        "native_callback_evidence": {
            "update_init_count": 0,
            "update_count": 0,
            "update_count_observed_max": 0,
        },
        "qblade_export_rpm_diagnostics": {
            "available": True,
            "max_abs_error_rpm": 0.0,
        },
        "sil_vorlap_max_abs_force": 1.0,
        "result_file_identity": pilot._identity(result_path),
        "reproducibility": {
            "case_metadata": metadata,
            "qblade": {
                "library": {"sha256": spec["execution"]["qblade_library_sha256"]}
            },
            "runtime_asset_fingerprint": case["runtime_asset_fingerprint"],
            "bridge_library": _runtime_identity()["bridge_library"],
            "vorlap_python_runtime_fingerprint": _runtime_identity()[
                "vorlap_python_runtime_fingerprint"
            ],
        },
        "restart": {
            "resume_requested": False,
            "resume_checkpoint": None,
            "resume_start_step": 0,
            "checkpoint_every": 0,
        },
    }
    coverage, velocity = _valid_source_and_velocity_receipts(case["total_steps"])
    summary["sil_vorlap_source_coverage"] = coverage
    summary["sil_vorlap_velocity_diagnostics"] = velocity
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    history = result_dir / "history.csv"
    history.write_text("test history\n", encoding="utf-8")
    metrics = {
        "rows": case["total_steps"],
        "reported_timestep_s": case["timestep_s"],
        "maximum_spectral_source_energy": 1.0,
        "maximum_spectral_energy_closure_error": 0.0,
        "spectral_energy_closure_tolerance": 1.0e-12,
        "spectral_residual_source_scale": 1.0,
        "maximum_applied_residual_force_n": 1.0,
        "maximum_response_nodal_force_n": 2.0,
        "maximum_total_supplemental_nodal_force_n": 3.0,
        "maximum_generated_force_component_n": 1.0,
        "modes": {
            "6": {
                "maximum_complex_projection_leakage_ratio": 0.0,
                "maximum_instantaneous_complex_projection_leakage_ratio": 0.0,
                "maximum_spectral_gain": 1.0,
                "peak_amplitude_estimate_m": spec["pilot"][
                    "predicted_peak_amplitude_target_by_mode_m"
                ]["6"],
                "first_half_peak_amplitude_m": spec["pilot"][
                    "predicted_peak_amplitude_target_by_mode_m"
                ]["6"],
                "second_half_peak_amplitude_m": spec["pilot"][
                    "predicted_peak_amplitude_target_by_mode_m"
                ]["6"],
                "tail_half_peak_relative_difference": 0.0,
                "maximum_activation": 1.5,
                "dominant_frequency_hz": pilot.DOCUMENTED_MODE_DAMPING[6][
                    "frequency_hz"
                ],
            },
            "12": {
                "maximum_complex_projection_leakage_ratio": 0.0,
                "maximum_instantaneous_complex_projection_leakage_ratio": 0.0,
                "maximum_spectral_gain": 1.0,
                "peak_amplitude_estimate_m": 0.0,
                "first_half_peak_amplitude_m": 0.0,
                "second_half_peak_amplitude_m": 0.0,
                "tail_half_peak_relative_difference": 0.0,
                "maximum_activation": 0.0,
                "dominant_frequency_hz": pilot.DOCUMENTED_MODE_DAMPING[12][
                    "frequency_hz"
                ],
            },
        },
    }
    monkeypatch.setattr(pilot, "_history_path_from_summary", lambda *_args: history)
    monkeypatch.setattr(pilot, "analyze_history", lambda *_args, **_kwargs: metrics)
    monkeypatch.setattr(
        pilot,
        "_audit_qblade_fatigue_result",
        lambda *_args, **_kwargs: {
            "channel_count": 26,
            "row_count": 80000,
            "expected_row_count": 80000,
            "maximum_abs_structural_load_channel_value": 1.0,
        },
    )

    result = pilot._case_result(
        root=tmp_path,
        case=case,
        model=model,
        acceptance=spec["acceptance"],
        campaign_kind=pilot.FULL_CAMPAIGN_KIND,
        peak_winds=spec["pilot"]["lock_in_peak_wind_speed_by_mode_mps"],
        amplitude_targets=spec["pilot"]["predicted_peak_amplitude_target_by_mode_m"],
        campaign_fingerprint=campaign_fingerprint,
        expected_runtime_identity=_runtime_identity(),
    )

    frequency_gate = next(
        gate for gate in result["gates"] if gate["gate"] == "mode6_peak_frequency"
    )
    force_gate = next(
        gate
        for gate in result["gates"]
        if gate["gate"] == "total_supplemental_force_finite_and_unclipped"
    )
    assert result["status"] == "complete"
    assert result["execution_receipt_pass"] is True
    assert frequency_gate["passed"] is True
    assert force_gate["passed"] is True
    assert force_gate["observed"] == 3.0
    assert "ceiling" in force_gate["limit_or_expected"]
    assert frequency_gate["target"] == pilot.DOCUMENTED_MODE_DAMPING[6]["frequency_hz"]
    assert "natural_frequency_hz" not in model["modes"][0]

    summary["restart"]["resume_start_step"] = 4000
    summary["restart"]["resume_requested"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    resumed = pilot._case_result(
        root=tmp_path,
        case=case,
        model=model,
        acceptance=spec["acceptance"],
        campaign_kind=pilot.FULL_CAMPAIGN_KIND,
        peak_winds=spec["pilot"]["lock_in_peak_wind_speed_by_mode_mps"],
        amplitude_targets=spec["pilot"]["predicted_peak_amplitude_target_by_mode_m"],
        campaign_fingerprint=campaign_fingerprint,
        expected_runtime_identity=_runtime_identity(),
    )
    assert resumed["execution_receipt_pass"] is False
    restart_gate = next(
        gate
        for gate in resumed["gates"]
        if gate["gate"] == "uninterrupted_actual_aero_execution"
    )
    assert restart_gate["passed"] is False

    summary["restart"]["resume_start_step"] = 0
    summary["restart"]["resume_requested"] = False
    summary["native_callback_evidence"]["update_cache_observation_count"] = 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    callback_mutation = pilot._case_result(
        root=tmp_path,
        case=case,
        model=model,
        acceptance=spec["acceptance"],
        campaign_kind=pilot.FULL_CAMPAIGN_KIND,
        peak_winds=spec["pilot"]["lock_in_peak_wind_speed_by_mode_mps"],
        amplitude_targets=spec["pilot"]["predicted_peak_amplitude_target_by_mode_m"],
        campaign_fingerprint=campaign_fingerprint,
        expected_runtime_identity=_runtime_identity(),
    )
    assert callback_mutation["execution_receipt_pass"] is False


def test_completed_analysis_rejects_gross_nonpeak_hybrid_response(
    tmp_path, monkeypatch
):
    spec = _spec()
    model = _model()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (inputs / "model.json").write_text(json.dumps(model), encoding="utf-8")
    cases = pilot.build_pilot_cases(tmp_path, spec)
    manifest = {
        "fingerprint": "b" * 64,
        "inputs": {
            "campaign_spec": {"path": "inputs/spec.json"},
            "response_model": {"path": "inputs/model.json"},
        },
        "cases": cases,
    }
    monkeypatch.setattr(
        pilot,
        "validate_campaign",
        lambda _value: {
            "valid": True,
            "campaign_root": tmp_path,
            "manifest": manifest,
        },
    )
    selected_targets = pilot._selected_wind_amplitude_targets(spec["pilot"])

    def completed_result(**kwargs):
        case = kwargs["case"]
        scenario = case["scenario"]
        wind = float(case["wind_speed_mps"])
        modes = {}
        for mode in ("6", "12"):
            amplitude = (
                0.0
                if scenario == pilot.SCENARIOS[0]
                else selected_targets[(wind, mode)]
            )
            if scenario == pilot.SCENARIOS[1] and wind == 16.0 and mode == "6":
                amplitude = 0.05
            first_half = amplitude
            second_half = amplitude
            if scenario == pilot.SCENARIOS[1] and wind == 16.0 and mode == "6":
                first_half = 0.0
            modes[mode] = {
                "complex_projection_tail_l2_leakage_ratio": 0.01,
                "peak_amplitude_estimate_m": amplitude,
                "first_half_peak_amplitude_m": first_half,
                "second_half_peak_amplitude_m": second_half,
                "tail_half_peak_relative_difference": (
                    0.5
                    if scenario == pilot.SCENARIOS[1]
                    and wind == 7.25
                    and mode == "6"
                    else 0.0
                ),
                "dominant_frequency_hz": pilot.DOCUMENTED_MODE_DAMPING[int(mode)][
                    "frequency_hz"
                ]
                * (
                    1.2
                    if scenario == pilot.SCENARIOS[1]
                    and wind == 7.25
                    and mode == "6"
                    else 1.0
                ),
                "maximum_activation": (
                    1.5
                    if scenario == pilot.SCENARIOS[1]
                    and selected_targets[(wind, mode)] > 0.0
                    else 0.0
                ),
            }
        return {
            "case_id": case["case_id"],
            "status": "complete",
            "passed": True,
            "scenario": scenario,
            "wind_speed_mps": wind,
            "timestep_s": case["timestep_s"],
            "metrics": {
                "maximum_response_nodal_force_n": 1.0 if scenario == pilot.SCENARIOS[1] else 0.0,
                "modes": modes,
            },
            "gates": [],
        }

    monkeypatch.setattr(pilot, "_case_result_or_error", completed_result)
    monkeypatch.setattr(
        pilot, "_load_flight_runtime_identity", lambda _root: _runtime_identity()
    )

    report_path = pilot.analyze_campaign(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    failed = {
        gate["gate"]: gate for gate in report["cross_case_gates"] if not gate["passed"]
    }
    assert report["complete_case_count"] == 20
    assert report["status"] == "pilot_failed_complete_hybrid_screen"
    assert report["full_rerun_authorized"] is False
    assert "mode6_inactive_stationary_tail_u16_dt0.0075" in failed
    assert "mode6_active_stationary_tail_u7.25_dt0.0075" in failed
    assert "mode6_active_frequency_u7.25_dt0.0075" in failed
    amplitude_reference = next(
        gate
        for gate in report["cross_case_gates"]
        if gate["gate"] == "mode6_closure_only_amplitude_reference_u16_dt0.0075"
    )
    assert amplitude_reference["diagnostic_only"] is True
    assert amplitude_reference["target"] == 0.0


def test_preflight_diagnoses_short_tail_stationarity_but_gates_active_projection(
    tmp_path, monkeypatch
):
    spec = pilot._effective_campaign_spec(
        _spec(), campaign_kind=pilot.PREFLIGHT_CAMPAIGN_KIND
    )
    model = _model()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (inputs / "model.json").write_text(json.dumps(model), encoding="utf-8")
    cases = pilot.build_pilot_cases(tmp_path, spec)
    manifest = {
        "fingerprint": "e" * 64,
        "inputs": {
            "campaign_spec": {"path": "inputs/spec.json"},
            "response_model": {"path": "inputs/model.json"},
        },
        "cases": cases,
    }
    monkeypatch.setattr(
        pilot,
        "validate_campaign",
        lambda _value: {
            "valid": True,
            "campaign_root": tmp_path,
            "manifest": manifest,
        },
    )

    active_projection_leakage = {"value": 0.01}

    def completed_result(**kwargs):
        case = kwargs["case"]
        response = case["scenario"] == pilot.SCENARIOS[1]
        return {
            "case_id": case["case_id"],
            "status": "complete",
            "passed": True,
            "execution_receipt_pass": True,
            "scenario": case["scenario"],
            "wind_speed_mps": case["wind_speed_mps"],
            "timestep_s": case["timestep_s"],
            "metrics": {
                "modes": {
                    "6": {
                        "complex_projection_tail_l2_leakage_ratio": (
                            active_projection_leakage["value"]
                        ),
                        "peak_amplitude_estimate_m": 0.001,
                        "first_half_peak_amplitude_m": 0.0002,
                        "second_half_peak_amplitude_m": 0.001,
                        "tail_half_peak_relative_difference": 0.8,
                        "dominant_frequency_hz": pilot.DOCUMENTED_MODE_DAMPING[6][
                            "frequency_hz"
                        ],
                        "maximum_activation": 1.5,
                    },
                    "12": {
                        "complex_projection_tail_l2_leakage_ratio": 0.01,
                        "peak_amplitude_estimate_m": 0.0,
                        "first_half_peak_amplitude_m": 0.0,
                        "second_half_peak_amplitude_m": 0.0,
                        "tail_half_peak_relative_difference": 0.0,
                        "dominant_frequency_hz": pilot.DOCUMENTED_MODE_DAMPING[12][
                            "frequency_hz"
                        ],
                        "maximum_activation": 0.0,
                    },
                },
                "maximum_response_nodal_force_n": 1.0 if response else 0.0,
            },
            "gates": [],
        }

    monkeypatch.setattr(pilot, "_case_result_or_error", completed_result)
    monkeypatch.setattr(
        pilot, "_load_flight_runtime_identity", lambda _root: _runtime_identity()
    )

    report_path = pilot.analyze_campaign(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["complete_case_count"] == 2
    assert report["status"] == "preflight_passed_complete_hybrid_screen"
    assert report["full_rerun_authorized"] is False
    stationarity = next(
        gate
        for gate in report["cross_case_gates"]
        if gate["gate"] == "mode6_active_stationary_tail_u8.25_dt0.0075"
    )
    assert stationarity["passed"] is True
    assert stationarity["diagnostic_only"] is True
    assert stationarity["criterion_passed"] is False
    assert all(
        gate.get("diagnostic_only") is True
        for gate in report["cross_case_gates"]
        if "stationary_tail" in gate["gate"]
    )

    active_projection_leakage["value"] = 0.2
    failed = json.loads(pilot.analyze_campaign(tmp_path).read_text(encoding="utf-8"))
    assert failed["status"] == "preflight_failed_complete_hybrid_screen"
    assert any(
        gate["gate"]
        == "mode6_active_complex_projection_tail_l2_u8.25_dt0.0075"
        and not gate["passed"]
        for gate in failed["cross_case_gates"]
    )


def test_two_speed_demonstration_requires_both_matched_h0_complete_pairs(
    tmp_path, monkeypatch
):
    spec = pilot._effective_campaign_spec(
        _spec(), campaign_kind=pilot.TWO_SPEED_TEST_CAMPAIGN_KIND
    )
    model = _model()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (inputs / "model.json").write_text(json.dumps(model), encoding="utf-8")
    cases = pilot.build_pilot_cases(tmp_path, spec)
    manifest = {
        "fingerprint": "f" * 64,
        "inputs": {
            "campaign_spec": {"path": "inputs/spec.json"},
            "response_model": {"path": "inputs/model.json"},
        },
        "cases": cases,
    }
    monkeypatch.setattr(
        pilot,
        "validate_campaign",
        lambda _value: {
            "valid": True,
            "campaign_root": tmp_path,
            "manifest": manifest,
        },
    )
    monkeypatch.setattr(
        pilot, "_load_flight_runtime_identity", lambda _root: _runtime_identity()
    )

    complete_response_force = {8.25: 2.0, 12.2: 3.0}

    def completed_result(**kwargs):
        case = kwargs["case"]
        response = case["scenario"] == pilot.SCENARIOS[1]
        return {
            "case_id": case["case_id"],
            "status": "complete",
            "passed": True,
            "execution_receipt_pass": True,
            "scenario": case["scenario"],
            "wind_speed_mps": case["wind_speed_mps"],
            "timestep_s": case["timestep_s"],
            "metrics": {
                "maximum_response_nodal_force_n": (
                    complete_response_force[case["wind_speed_mps"]]
                    if response
                    else 0.0
                ),
                "modes": {},
            },
            "gates": [],
        }

    monkeypatch.setattr(pilot, "_case_result_or_error", completed_result)
    report = json.loads(pilot.analyze_campaign(tmp_path).read_text(encoding="utf-8"))

    assert report["status"] == "two_speed_test_passed_terminal_audit"
    assert report["complete_case_count"] == 4
    assert report["passed_case_count"] == 4
    assert report["analysis_evaluator"]["pilot_model"]["path"] == (
        "vorlap/hvawt_response_pilot.py"
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        report["analysis_evaluator"]["pilot_model"]["sha256"],
    )
    assert len(report["cross_case_gates"]) == 4
    assert all(gate["passed"] for gate in report["cross_case_gates"])

    complete_response_force[12.2] = 0.0
    failed = json.loads(pilot.analyze_campaign(tmp_path).read_text(encoding="utf-8"))
    assert failed["status"] == "two_speed_test_failed_or_incomplete"
    assert any(
        gate["gate"] == "matched_complete_response_force_u12.2"
        and not gate["passed"]
        for gate in failed["cross_case_gates"]
    )


def test_complete_pilot_renders_paper_pdf_and_exact_40_row_csv(tmp_path, monkeypatch):
    spec = _spec()
    model = _model()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (inputs / "model.json").write_text(json.dumps(model), encoding="utf-8")
    manifest = {
        "fingerprint": "a" * 64,
        "inputs": {
            "campaign_spec": {"path": "inputs/spec.json"},
            "response_model": {"path": "inputs/model.json"},
        },
        "cases": [],
    }
    monkeypatch.setattr(
        pilot,
        "validate_campaign",
        lambda _value: {
            "valid": True,
            "campaign_root": tmp_path,
            "manifest": manifest,
        },
    )
    natural = {
        str(mode): values["frequency_hz"]
        for mode, values in pilot.DOCUMENTED_MODE_DAMPING.items()
    }
    cases = []
    for scenario in pilot.SCENARIOS:
        for wind in pilot.PILOT_WINDS_MPS:
            for timestep in pilot.PILOT_TIMESTEPS_S:
                metrics = {}
                for mode in ("6", "12"):
                    metrics[mode] = {
                        "peak_amplitude_estimate_m": (
                            0.148 if scenario == pilot.SCENARIOS[1] else 0.01
                        ),
                        "dominant_frequency_hz": natural[mode],
                        "maximum_activation": (
                            1.0 if scenario == pilot.SCENARIOS[1] else 0.0
                        ),
                        "maximum_complex_projection_leakage_ratio": 0.01,
                        "tail_half_peak_relative_difference": 0.02,
                    }
                cases.append(
                    {
                        "case_id": f"case_{len(cases):02d}",
                        "status": "complete",
                        "passed": True,
                        "scenario": scenario,
                        "wind_speed_mps": wind,
                        "timestep_s": timestep,
                        "metrics": {"modes": metrics},
                    }
                )
    manifest["cases"] = cases
    analysis = {
        "schema": pilot.ANALYSIS_SCHEMA,
        "status": "pilot_passed_complete_hybrid_screen",
        "campaign_fingerprint": manifest["fingerprint"],
        "complete_case_count": 20,
        "expected_case_count": 20,
        "cases": cases,
    }
    analysis_path = tmp_path / "pilot_analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")

    artifacts = pilot.render_pilot_result_artifacts(tmp_path)

    assert (
        artifacts["analysis_status"] == "pilot_passed_complete_hybrid_screen"
    )
    assert artifacts["row_count"] == 40
    pdf = Path(artifacts["pdf"]["path"])
    csv_path = Path(artifacts["csv"]["path"])
    assert pdf.read_bytes().startswith(b"%PDF-")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 40
    assert {row["tracked_mode"] for row in rows} == {"6", "12"}
    assert b"\r" not in csv_path.read_bytes()

    analysis["complete_case_count"] = 19
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    with pytest.raises(pilot.HVAWTResponsePilotError, match="complete analysis"):
        pilot.render_pilot_result_artifacts(tmp_path)


def test_full_rerun_authorization_is_fail_closed_and_does_not_launch(
    tmp_path, monkeypatch
):
    manifest = {"fingerprint": "a" * 64, "cases": []}
    monkeypatch.setattr(
        pilot,
        "validate_campaign",
        lambda _value: {
            "valid": True,
            "campaign_root": tmp_path,
            "manifest_path": tmp_path / "campaign_manifest.json",
            "manifest": manifest,
        },
    )
    pending = {
        "schema": pilot.FULL_RERUN_SCHEMA,
        "status": "prepared_waiting_for_pilot_pass",
        "execution_authorized": False,
        "launch_implemented": True,
        "campaign_fingerprint": manifest["fingerprint"],
        "expected_runs": 102,
        "scenarios": list(pilot.FULL_RERUN_SCENARIOS),
        "runs": [{"index": index} for index in range(102)],
    }
    (tmp_path / "full_rerun_plan_pending.json").write_text(
        json.dumps(pending), encoding="utf-8"
    )
    analysis = {
        "schema": pilot.ANALYSIS_SCHEMA,
        "status": "pilot_failed_complete_hybrid_screen",
        "full_rerun_authorized": False,
        "campaign_fingerprint": manifest["fingerprint"],
    }
    analysis_path = tmp_path / "pilot_analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")

    with pytest.raises(pilot.HVAWTResponsePilotError, match="remains blocked"):
        pilot.authorize_full_rerun(tmp_path)

    analysis.update(
        {
            "status": "pilot_passed_complete_hybrid_screen",
            "full_rerun_authorized": True,
        }
    )
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    authorized_path = pilot.authorize_full_rerun(tmp_path)
    authorized = json.loads(authorized_path.read_text(encoding="utf-8"))

    assert authorized["execution_authorized"] is True
    assert authorized["launch_implemented"] is True
    assert authorized["expected_runs"] == 102
    assert len(authorized["runs"]) == 102


def test_cli_requires_explicit_case_index_outside_slurm(monkeypatch):
    from run_headless import hvawt_response_pilot as cli

    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    args = type("Args", (), {"case_index": None})()
    with pytest.raises(ValueError, match="Provide --case-index"):
        cli._case_index(args)


def test_private_paper_source_archive_integrity():
    raw = os.environ.get("VORLAP_PAPER_SOURCE_ARCHIVE")
    if not raw:
        pytest.skip("Exact private paper source case is external; set VORLAP_PAPER_SOURCE_ARCHIVE")
    source_archive = Path(raw)
    assert _sha256(source_archive) == "6f6daf91d026087bb5f5acec6b313c7cead1a70faa1deded1819e15c60d7d403"
    with tarfile.open(source_archive, "r:gz") as archive:
        members = archive.getmembers()
    assert sum(member.isfile() for member in members) == 20
    assert not any(Path(member.name).name.startswith("._") for member in members)
    assert not any(
        key.startswith(("LIBARCHIVE.xattr.", "SCHILY.xattr."))
        for member in members
        for key in member.pax_headers
    )
