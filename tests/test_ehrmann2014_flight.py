from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_headless import run_ehrmann2014_flight as scheduler


REPO_ROOT = Path(__file__).resolve().parents[1]


def _row(index: int, steps: int) -> dict[str, object]:
    return {
        "index": index,
        "case_id": f"case_{index:02d}",
        "num_steps": steps,
        "sim": f"cases/case_{index:02d}/case_{index:02d}.sim",
        "result_dir": f"results/case_{index:02d}",
        "result_name": f"result_{index:02d}",
        "native_case_identity": {"fingerprint": f"identity-{index}"},
    }


def test_lpt_launch_order_puts_half_timestep_cases_first(tmp_path):
    rows = [_row(index, 40_000 if index in {3, 9, 17, 31} else 20_000) for index in range(40)]
    plan = scheduler.ordered_case_plan(tmp_path, {"cases": rows})
    assert [row["index"] for row in plan["runnable"][:4]] == [3, 9, 17, 31]
    assert [row["num_steps"] for row in plan["runnable"]] == sorted(
        [row["num_steps"] for row in plan["runnable"]], reverse=True
    )


def test_srun_command_is_exclusive_exact_bound_one_core_step(tmp_path):
    command = scheduler.build_srun_command(
        case_index=7,
        campaign_root=tmp_path / "campaign",
        qblade_root=tmp_path / "qblade",
        python_executable="python3",
        srun_executable="srun",
        threads=1,
    )
    assert command[:7] == [
        "srun",
        "--exclusive",
        "--exact",
        "-N1",
        "-n1",
        "-c1",
        "--cpu-bind=cores",
    ]
    assert "--mem-bind=local" not in command
    assert command[-1] == "--resume"


def test_allocation_contract_refuses_oversubscription_or_thread_change():
    scheduler._allocation_contract(workers=40, threads=1, allocated_cpus=40)
    with pytest.raises(scheduler.SchedulerError, match="benchmarked 1 thread"):
        scheduler._allocation_contract(workers=10, threads=4, allocated_cpus=40)
    with pytest.raises(scheduler.SchedulerError, match="exactly 40"):
        scheduler._allocation_contract(workers=40, threads=1, allocated_cpus=39)
    with pytest.raises(scheduler.SchedulerError, match="requires 40 workers"):
        scheduler._allocation_contract(workers=39, threads=1, allocated_cpus=40)


def test_slurm_cpu_slots_use_task_contract_not_node_visible_count():
    environment = {
        "SLURM_NTASKS": "40",
        "SLURM_CPUS_PER_TASK": "1",
        "SLURM_CPUS_ON_NODE": "112",
        "SLURM_JOB_CPUS_PER_NODE": "112",
    }
    assert scheduler._requested_slurm_cpu_slots(environment) == 40
    assert scheduler._requested_slurm_cpu_slots({}) == 40
    assert scheduler._requested_slurm_cpu_slots(
        {"SLURM_NTASKS": "invalid", "SLURM_CPUS_PER_TASK": "1"}
    ) == 40


def test_thread_scaling_benchmark_is_schema_and_hash_bound():
    evidence = scheduler._validate_thread_scaling_benchmark()
    assert evidence["schema"] == "vorlap-ehrmann2014-thread-scaling-benchmark-v1"
    assert evidence["scope"] == "operational_allocation_smoke_not_physical_validation"
    assert [row["threads"] for row in evidence["runs"]] == [1, 4]
    assert evidence["comparison"]["four_thread_elapsed_reduction_percent"] == pytest.approx(
        3.011764613593723
    )
    assert evidence["comparison"]["numeric_result_rows_identical"] is True
    assert evidence["comparison"]["decision"] == "one_core_per_case"


def test_complete_skip_requires_identity_and_result_digest(tmp_path):
    row = _row(0, 20_000)
    result_dir = tmp_path / str(row["result_dir"])
    result_dir.mkdir(parents=True)
    result = result_dir / f"{row['result_name']}.txt"
    result.write_text("verified result\n", encoding="utf-8")
    (result_dir / "native_case_identity.json").write_text(
        json.dumps(row["native_case_identity"]), encoding="utf-8"
    )
    summary = {
        "requested_steps": 20_000,
        "completed_steps": 20_000,
        "failed": False,
        "stopped": False,
        "exported_results": True,
        "threads": 1,
        "coupling_mode": "sil-vorlap",
        "sil_vorlap": True,
        "result_file_identity": {
            "sha256": scheduler.campaign.sha256_file(result),
            "size_bytes": result.stat().st_size,
        },
    }
    summary_path = result_dir / "case_00_run_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert scheduler.classify_case(tmp_path, row) == (
        "complete",
        "exact_identity_complete_result",
    )

    result.write_text("mutated result\n", encoding="utf-8")
    assert scheduler.classify_case(tmp_path, row)[0] == "blocked"


def test_complete_summary_without_native_identity_is_blocked(tmp_path):
    row = _row(0, 20_000)
    result_dir = tmp_path / str(row["result_dir"])
    result_dir.mkdir(parents=True)
    (result_dir / "case_00_run_summary.json").write_text(
        json.dumps(
            {
                "requested_steps": 20_000,
                "completed_steps": 20_000,
                "failed": False,
                "stopped": False,
                "exported_results": True,
            }
        ),
        encoding="utf-8",
    )
    assert scheduler.classify_case(tmp_path, row) == (
        "blocked",
        "complete_summary_missing_native_case_identity",
    )


def test_exact_campaign_raw_reuse_receipt_can_skip_prior_native_run(tmp_path):
    row = {**_row(0, 20_000), "scenario": "raw_qblade"}
    result_dir = tmp_path / str(row["result_dir"])
    result_dir.mkdir(parents=True)
    result = result_dir / f"{row['result_name']}.txt"
    result.write_text("prior exact native result\n", encoding="utf-8")
    identity = result_dir / "native_case_identity.json"
    identity.write_text(json.dumps(row["native_case_identity"]), encoding="utf-8")
    summary = {
        "requested_steps": 20_000,
        "completed_steps": 20_000,
        "failed": False,
        "stopped": False,
        "exported_results": True,
        "threads": 8,
        "coupling_mode": "qblade-native",
        "sil_vorlap": False,
        "result_file_identity": {
            "sha256": scheduler.campaign.sha256_file(result),
            "size_bytes": result.stat().st_size,
        },
    }
    summary_path = result_dir / "case_00_run_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    receipt = {
        "schema": scheduler.campaign.REUSE_RECEIPT_SCHEMA,
        "case_id": row["case_id"],
        "native_case_identity_fingerprint": row["native_case_identity"]["fingerprint"],
        "run_summary_sha256": scheduler.campaign.sha256_file(summary_path),
    }
    (result_dir / "reuse_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert scheduler.classify_case(tmp_path, row) == (
        "complete",
        "exact_identity_complete_result",
    )


def test_dynamic_queue_backfills_after_first_case_finishes(tmp_path, monkeypatch):
    rows = [_row(index, 20_000) for index in range(42)]
    manifest = {"fingerprint": "campaign-fingerprint", "cases": rows}
    monkeypatch.setattr(
        scheduler.campaign,
        "validate",
        lambda _path: {"root": tmp_path, "manifest": manifest},
    )
    completed: set[int] = set()
    launch_snapshots: list[tuple[int, set[int]]] = []

    def fake_classify(_root, row):
        index = int(row["index"])
        return (
            ("complete", "exact_identity_complete_result")
            if index in completed
            else ("runnable", "no_run_summary")
        )

    monkeypatch.setattr(scheduler, "classify_case", fake_classify)

    class FakeProcess:
        def __init__(self, index: int):
            self.index = index
            self.polls = 0
            self.returncode = None

        def poll(self):
            self.polls += 1
            threshold = 1 if self.index == 0 or self.index >= 40 else 3
            if self.polls >= threshold:
                completed.add(self.index)
                self.returncode = 0
                return 0
            return None

    def fake_popen(command, **_kwargs):
        index = int(command[command.index("--case-index") + 1])
        launch_snapshots.append((index, set(completed)))
        return FakeProcess(index)

    report_path = scheduler.run_scheduler(
        campaign_root=tmp_path,
        qblade_root=tmp_path / "qblade",
        python_executable="python3",
        workers=40,
        threads=1,
        allocated_cpus=40,
        poll_seconds=0.0,
        popen_factory=fake_popen,
    )
    assert launch_snapshots[40] == (40, {0})
    report = json.loads(report_path.read_text())
    assert report["status"] == "complete"
    assert len(report["cases"]) == 42


def test_flight_slurm_wrapper_is_one_node_40_by_1_shared_and_two_hours():
    text = (REPO_ROOT / "run_headless/submit_ehrmann2014_flight.sj").read_text()
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=40" in text
    assert "#SBATCH --ntasks-per-node=40" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=80G" in text
    assert "#SBATCH --hint=nomultithread" in text
    assert "#SBATCH --exclusive" not in text
    assert "#SBATCH --time=02:00:00" in text
    assert "#SBATCH --array" not in text
    assert "EXPECTED_ALLOCATED_CPUS=40" in text
    assert "EXPECTED_NODE_PHYSICAL_CORES=112" in text
    assert "EXPECTED_NUMA_DOMAINS=8" in text
    assert 'allocated_cpus="$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"' in text
    assert 'node_visible_cpus="${SLURM_CPUS_ON_NODE:-unset}"' in text
    assert 'job_cpus_per_node="${SLURM_JOB_CPUS_PER_NODE:-unset}"' in text
    assert 'allocated_cpus="${SLURM_CPUS_ON_NODE:-}"' not in text
    assert 'VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"' in text
    assert "/pscratch/kevmoor/vorlap/.venv/bin/python" in text
    assert 'resolved_python="$(command -v "${candidate}"' in text
    assert "import h5py, numpy, scipy" in text
    assert 'EHRMANN_FLIGHT_DRY_RUN="${EHRMANN_FLIGHT_DRY_RUN:-0}"' in text
    assert 'if [[ ! -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]]' in text
    assert '--output "${staging_root}"' in text
    assert '--runtime-vorlap-root "${VORLAP_ROOT}"' in text
    assert 'mv -T -- "${staging_root}" "${CAMPAIGN_ROOT}"' in text
    assert 'scheduler+=(--dry-run)' in text
    assert 'assert len(manifest["cases"]) == 40' in text
    assert 'assert report["status"] == "dry_run"' in text


def test_flight_wrapper_auto_generation_is_fail_closed_and_atomic():
    text = (REPO_ROOT / "run_headless/submit_ehrmann2014_flight.sj").read_text()
    assert '[[ ! -e "${CAMPAIGN_ROOT}" ]]' in text
    assert "Campaign root exists without a manifest; refusing to replace it" in text
    assert '[[ ! -e "${staging_root}" ]]' in text
    assert (
        '"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign validate \\\n'
        '    --campaign "${staging_root}"'
    ) in text
    assert text.index('--campaign "${staging_root}"') < text.index(
        'mv -T -- "${staging_root}" "${CAMPAIGN_ROOT}"'
    )


def test_all_ehrmann_shell_entrypoints_embed_frozen_python_fallback():
    for relative in (
        "run_headless/submit_ehrmann2014_flight.sj",
        "run_headless/submit_ehrmann2014_validation.sj",
        "run_headless/prepare_ehrmann2014_validation_native.sh",
    ):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "ehrmann2014_python_env.sh" not in text
        assert "/pscratch/kevmoor/vorlap/.venv/bin/python" in text
        assert "PYTHON_EXE_FALLBACK:-python3" in text
        assert "import h5py, numpy, scipy" in text


def test_prepare_prints_primary_two_hour_and_resume_eight_hour_commands():
    text = (REPO_ROOT / "run_headless/prepare_ehrmann2014_validation_native.sh").read_text()
    assert "submit_ehrmann2014_flight.sj" in text
    assert "sbatch --time=08:00:00" in text


def test_benchmark_spec_names_flight_primary_and_old_array_fallback():
    spec = json.loads(
        (REPO_ROOT / "examples/ehrmann2014_naca0018/benchmark_spec.json").read_text()
    )
    executable = spec["executable_campaign"]
    assert executable["slurm_entrypoint"] == "run_headless/submit_ehrmann2014_flight.sj"
    assert executable["slurm_fallback_entrypoint"] == (
        "run_headless/submit_ehrmann2014_validation.sj"
    )
