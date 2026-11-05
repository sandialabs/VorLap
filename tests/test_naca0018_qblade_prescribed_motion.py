from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from examples.ehrmann2014_naca0018 import nalu_prescribed_motion as nalu
from examples.ehrmann2014_naca0018 import qblade_prescribed_motion as qblade
from run_headless import run_naca0018_work_map_qblade_flight as scheduler
from run_headless import run_naca0018_work_map_qblade_local as local_scheduler
from vorlap.qblade_files import read_qblade_keyword

REPO_ROOT = Path(__file__).resolve().parents[1]
QBLADE_ROOT = REPO_ROOT / "run_headless/QBladeCE_2.0.9.7"


@pytest.fixture
def generated(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "campaign"
    manifest = qblade.generate_campaign(root, QBLADE_ROOT)
    return root, manifest


@pytest.mark.requires_qblade
def test_generates_exact_frozen_matrix_and_matches_nalu_motion_grid(generated) -> None:
    root, manifest = generated
    assert qblade.validate_campaign(root, QBLADE_ROOT)["valid"] is True
    assert manifest["schema"] == qblade.CAMPAIGN_SCHEMA
    assert len(manifest["cases"]) == 48
    assert manifest["runtime"]["library"]["sha256"] == (
        qblade.EXPECTED_QBLADE_LIBRARY_SHA256
    )
    assert manifest["runtime"]["runtime_dependency_fingerprint"] == (
        qblade.EXPECTED_QBLADE_DEPENDENCY_FINGERPRINT
    )
    assert manifest["external_forces"] == {
        "controller_type": 0,
        "external_library_entries": 0,
        "sil_vorlap": False,
        "sil_constant_force": False,
        "expected_native_callback_updates": 0,
    }
    assert manifest["geometry_and_frames"]["qblade_internal_incidence_deg"] == 90.0
    assert (
        manifest["geometry_and_frames"]["vorlap_relative_wind_incidence_deg"] == -90.0
    )
    assert manifest["execution"] == {
        "threads_per_case": 1,
        "total_case_count": 48,
        "preflight_cpu_slots": 1,
        "matrix_cpu_slots": 47,
        "local_docker_max_cpu_slots": 8,
        "default_walltime": "02:00:00",
        "resume_walltime": "08:00:00",
        "checkpoint_every_steps": 1000,
    }
    assert manifest["preflight_gate"]["case_index"] == 35
    assert manifest["preflight_gate"]["case_id"] == "re140000_k0p45_a0p0500"

    for row in manifest["cases"]:
        paired = nalu._motion_values(
            row["reynolds_number"],
            row["reduced_frequency_k"],
            row["amplitude_ratio"],
            manifest["fluid_and_section"],
        )
        assert row["samples_per_cycle"] == paired["samples_per_cycle"]
        assert row["time_step_s"] == paired["time_step_s"]
        assert row["expected_result_rows"] == paired["updates_after_restart"]
        assert row["wrapper_steps"] == paired["updates_after_restart"] + 1
        assert row["relative_history_start_s"] == paired["relative_history_start_s"]
        assert row["relative_history_end_s"] == paired["relative_history_end_s"]
        sim = root / row["sim"]
        turbine = root / row["case_dir"] / "model/naca0018_prescribed_motion.trb"
        assert read_qblade_keyword(sim, "MOTIONFILE") == "motion.dat"
        assert read_qblade_keyword(sim, "FILTERFILE") is None
        assert read_qblade_keyword(sim, "INITIAL_PITCH") == "0"
        assert read_qblade_keyword(sim, "RPMPRESCRIBED") == "0"
        assert read_qblade_keyword(sim, "INCLUDEAERO") == "1"
        assert read_qblade_keyword(turbine, "LIBFILE_1") is None
        assert read_qblade_keyword(turbine, "PARAMETERFILE") is None


@pytest.mark.requires_qblade
def test_motion_file_starts_at_zero_and_covers_wrapper_endpoint(generated) -> None:
    root, manifest = generated
    row = manifest["cases"][0]
    lines = (root / row["motion_file"]).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Time TransX TransY TransZ RotX RotY RotZ"
    assert len(lines) == row["wrapper_steps"] + 2
    first = list(map(float, lines[1].split()))
    last = list(map(float, lines[-1].split()))
    assert first == [0.0] * 7
    assert last[0] == row["wrapper_steps"] * row["time_step_s"]
    expected_last_q = row["amplitude_m"] * math.sin(
        2.0 * math.pi * row["frequency_hz"] * last[0]
    )
    assert last[2] == expected_last_q


@pytest.mark.requires_qblade
def test_campaign_replay_uses_runtime_bytes_not_execution_host_path(generated) -> None:
    root, manifest = generated
    manifest["runtime"]["root"] = "/work/VorLap/run_headless/QBladeCE_2.0.9.7"
    manifest["runtime"]["library"]["path"] = (
        "/work/VorLap/run_headless/QBladeCE_2.0.9.7/"
        + qblade.QBLADE_LIBRARY_NAME
    )
    payload = dict(manifest)
    payload.pop("fingerprint")
    manifest["fingerprint"] = qblade.canonical_fingerprint(payload)
    (root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert qblade.validate_campaign(root)["valid"] is True


@pytest.mark.requires_qblade
def test_case_command_is_one_core_native_and_checkpoint_resumable(generated) -> None:
    root, manifest = generated
    command = qblade.command_for_case(
        root,
        case_index=47,
        qblade_root=QBLADE_ROOT,
        python_executable="python3",
        resume=True,
    )
    assert command[command.index("--steps") + 1] == str(
        manifest["cases"][47]["wrapper_steps"]
    )
    assert command[command.index("--threads") + 1] == "1"
    assert command[command.index("--coupling-mode") + 1] == "none"
    assert "--sil-vorlap" not in command
    assert "--sil-constant-force" not in command
    assert command[-3:] == [
        "--checkpoint-every",
        "1000",
        "--resume-from-checkpoint",
    ]
    assert "--checkpoint-every" in command


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_complete_result(root: Path, manifest: dict, index: int) -> None:
    row = manifest["cases"][index]
    result_dir = root / row["result_dir"]
    result = result_dir / f"{row['result_name']}.txt"
    header = "\t".join(name.replace(" ", "~") for name in qblade.FILTER_CHANNELS)
    count = row["expected_result_rows"]
    dt = row["time_step_s"]
    omega = 2.0 * math.pi * row["frequency_hz"]
    amplitude = row["amplitude_m"]
    with result.open("w", encoding="utf-8") as stream:
        stream.write("Synthetic QBlade result\nSimulation: test\n")
        stream.write(header + "\n")
        for point in range(0, count + 1):
            time = point * dt
            displacement = amplitude * math.sin(omega * time)
            values = (
                time,
                row["wind_speed_mps"],
                0.0,
                0.0,
                0.0,
                displacement,
                displacement,
                10.0 * math.cos(omega * time),
                90.0,
                row["reynolds_number"],
            )
            stream.write("\t".join(f"{value:.17g}" for value in values) + "\n")
    summary = {
        "failed": False,
        "stopped": False,
        "exported_results": True,
        "requested_steps": row["wrapper_steps"],
        "completed_steps": row["wrapper_steps"],
        "threads": 1,
        "coupling_mode": "none",
        "sil_vorlap": False,
        "sil_constant_force": False,
        "native_callback_evidence": {
            "callback_block_present": False,
            "update_init_count": 0,
            "update_count_observed_max": 0,
            "runtime_initialized_message_seen": False,
        },
        "reproducibility": {
            "qblade": {"library": {"sha256": qblade.EXPECTED_QBLADE_LIBRARY_SHA256}}
        },
        "result_file_identity": {
            "sha256": _sha256(result),
            "size_bytes": result.stat().st_size,
        },
    }
    summary_path = result_dir / f"{Path(row['sim']).stem}_run_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def _write_synthetic_preflight_report(root: Path, manifest: dict) -> Path:
    path = root / qblade.PREFLIGHT_SCHEDULER_REPORT
    value = {
        "schema": qblade.PREFLIGHT_SCHEDULER_SCHEMA,
        "stage": "preflight",
        "status": "complete",
        "campaign_fingerprint": manifest["fingerprint"],
        "allocation": {
            "workers": 1,
            "threads_per_case": 1,
            "allocated_physical_cpus": 1,
        },
        "execution_environment": {
            "kind": "flight_slurm",
            "platform": "linux_x86_64",
            "slurm_job_id": "123456",
        },
        "preflight_gate": {
            "status": "complete",
            "case_index": scheduler.PREFLIGHT_CASE_INDEX,
            "case_id": scheduler.PREFLIGHT_CASE_ID,
        },
        "cases": [
            {
                "index": scheduler.PREFLIGHT_CASE_INDEX,
                "case_id": scheduler.PREFLIGHT_CASE_ID,
                "status": "complete",
            }
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.requires_qblade
def test_postprocess_emits_canonical_grid_and_audits_force_motion_channels(
    generated,
) -> None:
    root, manifest = generated
    index = 44  # k=1.05, A/c=0.005: shortest executed test history
    _write_synthetic_complete_result(root, manifest, index)
    audit = qblade.postprocess_case(root, index)
    row = manifest["cases"][index]
    assert audit["status"] == "complete"
    assert audit["runtime"]["coupling_mode"] == "none"
    assert audit["runtime"]["external_callback_updates"] == 0
    assert audit["geometry_frame_audit"]["median_abs_qblade_aoa_deg"] == 90.0
    history = root / audit["canonical_history"]["path"]
    assert audit["canonical_history"]["sha256"] == _sha256(history)
    rows = list(__import__("csv").DictReader(history.open(encoding="utf-8")))
    assert len(rows) == row["expected_result_rows"]
    assert float(rows[0]["time_s"]) == row["time_step_s"]
    assert float(rows[-1]["time_s"]) == row["relative_history_end_s"]
    assert math.isclose(
        float(rows[-1]["time_s"]) - float(rows[0]["time_s"]),
        13.0 * row["period_s"],
        rel_tol=2.0e-13,
    )
    status = qblade.campaign_status(root)
    assert status["complete_cases"] == 1
    assert status["runnable_cases"] == 47


@pytest.mark.requires_qblade
def test_postprocess_rejects_any_native_callback(generated) -> None:
    root, manifest = generated
    index = 44
    _write_synthetic_complete_result(root, manifest, index)
    row = manifest["cases"][index]
    summary_path = (
        root / row["result_dir"] / f"{Path(row['sim']).stem}_run_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["native_callback_evidence"]["callback_block_present"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(qblade.CampaignError, match="external callback"):
        qblade.postprocess_case(root, index)


def test_slurm_wrappers_request_fully_assigned_staged_shared_allocations() -> None:
    matrix = (
        REPO_ROOT / "run_headless/submit_naca0018_work_map_qblade.sj"
    ).read_text(encoding="utf-8")
    preflight = (
        REPO_ROOT / "run_headless/submit_naca0018_work_map_qblade_preflight.sj"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in matrix
    assert "#SBATCH --ntasks=47" in matrix
    assert "#SBATCH --ntasks-per-node=47" in matrix
    assert "#SBATCH --cpus-per-task=1" in matrix
    assert "#SBATCH --hint=nomultithread" in matrix
    assert "#SBATCH --exclusive" not in matrix
    assert "#SBATCH --time=02:00:00" in matrix
    assert 'QBLADE_STAGE="${QBLADE_STAGE:-}"' in matrix
    assert "run_naca0018_work_map_qblade_flight" in matrix
    assert 'expected_slots="$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"' in matrix
    assert '[[ "${expected_slots}" -eq "${WORKERS}" ]]' in matrix
    assert "EXPECTED_NODE_PHYSICAL_CORES=112" in matrix
    assert "EXPECTED_NUMA_DOMAINS=8" in matrix
    assert "lscpu -p=CORE,SOCKET" in matrix
    assert "#SBATCH --ntasks=1" in preflight
    assert "export QBLADE_STAGE=preflight" in preflight


@pytest.mark.requires_qblade
def test_preflight_dry_run_assigns_only_the_frozen_gate_case(generated) -> None:
    root, _manifest = generated
    path = scheduler.run_scheduler(
        stage="preflight",
        campaign_root=root,
        qblade_root=QBLADE_ROOT,
        python_executable="python3",
        dry_run=True,
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "dry_run"
    assert report["stage"] == "preflight"
    assert report["allocation"]["fresh_stage_slot_utilization"] == 1.0
    assert report["allocation"]["initial_resume_slot_utilization"] == 1.0
    assert len(report["cases"]) == 1
    assert report["cases"][0]["index"] == scheduler.PREFLIGHT_CASE_INDEX
    assert report["cases"][0]["case_id"] == scheduler.PREFLIGHT_CASE_ID
    assert report["cases"][0]["status"] == "planned"
    assert report["cases"][0]["command"][:7] == [
        "srun",
        "--exclusive",
        "--exact",
        "-N1",
        "-n1",
        "-c1",
        "--cpu-bind=cores",
    ]


def test_allocation_contract_rejects_underfilled_requested_slots() -> None:
    scheduler._allocation_contract(
        stage="preflight", workers=1, threads=1, allocated_cpus=1
    )
    scheduler._allocation_contract(
        stage="matrix", workers=47, threads=1, allocated_cpus=47
    )
    with pytest.raises(scheduler.SchedulerError, match="47 workers"):
        scheduler._allocation_contract(
            stage="matrix", workers=46, threads=1, allocated_cpus=47
        )
    with pytest.raises(scheduler.SchedulerError, match="one thread"):
        scheduler._allocation_contract(
            stage="matrix", workers=47, threads=2, allocated_cpus=47
        )
    with pytest.raises(scheduler.SchedulerError, match="47 CPU slots"):
        scheduler._allocation_contract(
            stage="matrix", workers=47, threads=1, allocated_cpus=46
        )


@pytest.mark.requires_qblade
def test_local_scheduler_is_explicitly_docker_bound_and_capped_at_eight(
    generated,
) -> None:
    root, _manifest = generated
    image_id = "sha256:" + "a" * 64
    path = local_scheduler.run_scheduler(
        stage="preflight",
        campaign_root=root,
        qblade_root=QBLADE_ROOT,
        python_executable="python3",
        workers=1,
        cpu_limit=8,
        container_image_id=image_id,
        dry_run=True,
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["execution_environment"] == {
        "kind": "local_docker",
        "platform": "linux_amd64",
        "container_image_id": image_id,
        "container_cpu_limit": 8,
    }
    command = report["cases"][0]["command"]
    assert "srun" not in command
    assert command[command.index("--threads") + 1] == "1"
    assert command[-1] == "--resume-from-checkpoint"
    local_scheduler._execution_contract(stage="matrix", workers=8, cpu_limit=8)
    with pytest.raises(local_scheduler.LocalSchedulerError, match=r"\[1, 8\]"):
        local_scheduler._execution_contract(stage="matrix", workers=8, cpu_limit=9)
    with pytest.raises(local_scheduler.LocalSchedulerError, match="one worker"):
        local_scheduler._execution_contract(stage="preflight", workers=2, cpu_limit=8)


def test_local_docker_launcher_has_fixed_cpu_and_provenance_contract() -> None:
    script = (
        REPO_ROOT / "run_headless/docker_run_naca0018_work_map_qblade.sh"
    ).read_text(encoding="utf-8")
    assert 'CPU_LIMIT="${VORLAP_QBLADE_LOCAL_CPUS:-8}"' in script
    assert '--cpus "${CPU_LIMIT}"' in script
    assert "run_naca0018_work_map_qblade_local" in script
    assert "VORLAP_CONTAINER_IMAGE_ID" in script
    assert "srun" not in script


@pytest.mark.requires_qblade
def test_matrix_refuses_to_plan_before_executed_preflight(generated) -> None:
    root, _manifest = generated
    with pytest.raises(scheduler.SchedulerError, match="requires the complete"):
        scheduler.run_scheduler(
            stage="matrix",
            campaign_root=root,
            qblade_root=QBLADE_ROOT,
            python_executable="python3",
            dry_run=True,
        )


@pytest.mark.requires_qblade
def test_matrix_excludes_completed_preflight_and_assigns_47_slots(generated) -> None:
    root, manifest = generated
    _write_synthetic_complete_result(root, manifest, scheduler.PREFLIGHT_CASE_INDEX)
    qblade.postprocess_case(root, scheduler.PREFLIGHT_CASE_INDEX)
    _write_synthetic_preflight_report(root, manifest)
    path = scheduler.run_scheduler(
        stage="matrix",
        campaign_root=root,
        qblade_root=QBLADE_ROOT,
        python_executable="python3",
        dry_run=True,
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["stage"] == "matrix"
    assert report["execution_contract"]["preflight_complete_at_start"] is True
    assert len(report["cases"]) == 47
    assert scheduler.PREFLIGHT_CASE_INDEX not in {
        row["index"] for row in report["cases"]
    }
    assert report["allocation"]["initial_resume_slot_utilization"] == 1.0
