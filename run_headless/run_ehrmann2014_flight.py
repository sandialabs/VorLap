#!/usr/bin/env python3
"""Run the Ehrmann campaign as dynamically scheduled one-core Slurm steps.

The Flight allocation contract is 40 physical cores on one shared 112-core
node.  One core per QBlade case follows the measured thread-scaling smoke and
gives materially better allocation throughput than reserving four cores per
case. Longest-processing-time ordering remains deterministic for resume
submissions, although all 40 cases launch together on a fresh campaign.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.ehrmann2014_naca0018 import campaign
from vorlap.qblade_run_support import atomic_write_json


SCHEMA = "vorlap-ehrmann2014-flight-scheduler-v1"
BENCHMARKED_THREADS_PER_CASE = 1
FLIGHT_NODE_PHYSICAL_CORES = 112
FLIGHT_NUMA_DOMAINS = 8
FLIGHT_ALLOCATED_CORES = 40
DEFAULT_WORKERS = 40
THREAD_SCALING_BENCHMARK = (
    REPO_ROOT / "examples/ehrmann2014_naca0018/thread_scaling_benchmark.json"
)
THREAD_SCALING_BENCHMARK_SHA256 = (
    "45e481daf7a25e67eaaa35b9cae07ed5db254f3d45e35c2a3a2ba917f8aefac1"
)


class SchedulerError(RuntimeError):
    """Raised when campaign state or the node allocation violates the contract."""


def _requested_slurm_cpu_slots(environ: Mapping[str, str] | None = None) -> int:
    """Return task CPU slots, never the node-visible CPU count.

    Flight exposes all 112 node CPUs through ``SLURM_CPUS_ON_NODE`` even for
    this 40-task campaign.  The scheduler contract is instead the product of
    the requested task count and CPUs per task.
    """

    values = os.environ if environ is None else environ
    try:
        ntasks = int(values["SLURM_NTASKS"])
        cpus_per_task = int(values["SLURM_CPUS_PER_TASK"])
    except (KeyError, TypeError, ValueError):
        return FLIGHT_ALLOCATED_CORES
    if ntasks < 1 or cpus_per_task < 1:
        return FLIGHT_ALLOCATED_CORES
    return ntasks * cpus_per_task


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchedulerError(f"Cannot read required JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchedulerError(f"Required JSON object is invalid: {path}")
    return value


def _case_paths(root: Path, row: Mapping[str, Any]) -> dict[str, Path]:
    result_dir = root / str(row["result_dir"])
    return {
        "result_dir": result_dir,
        "result": result_dir / f"{row['result_name']}.txt",
        "summary": result_dir / f"{Path(str(row['sim'])).stem}_run_summary.json",
        "identity": result_dir / "native_case_identity.json",
        "reuse_receipt": result_dir / "reuse_receipt.json",
    }


def _integer(value: object, default: int = -1) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _validate_thread_scaling_benchmark() -> dict[str, Any]:
    path = THREAD_SCALING_BENCHMARK
    if not path.is_file() or campaign.sha256_file(path) != THREAD_SCALING_BENCHMARK_SHA256:
        raise SchedulerError("The tracked thread-scaling benchmark identity differs.")
    evidence = _read_json(path)
    runs = evidence.get("runs")
    comparison = evidence.get("comparison")
    if (
        evidence.get("schema") != "vorlap-ehrmann2014-thread-scaling-benchmark-v1"
        or evidence.get("status") != "passed"
        or evidence.get("scope") != "operational_allocation_smoke_not_physical_validation"
        or not isinstance(runs, list)
        or len(runs) != 2
        or not isinstance(comparison, dict)
    ):
        raise SchedulerError("The tracked thread-scaling benchmark schema/status differs.")
    by_threads = {_integer(row.get("threads")): row for row in runs if isinstance(row, dict)}
    if set(by_threads) != {1, 4}:
        raise SchedulerError("The tracked thread-scaling benchmark thread pair differs.")
    for threads, row in by_threads.items():
        if (
            _integer(row.get("requested_steps")) != 500
            or _integer(row.get("completed_steps")) != 500
            or row.get("failed") is not False
            or row.get("stopped") is not False
            or row.get("exported_results") is not True
            or float(row.get("elapsed_seconds", 0.0)) <= 0.0
            or len(str(row.get("run_summary_sha256", ""))) != 64
            or len(str(row.get("result_file_sha256", ""))) != 64
        ):
            raise SchedulerError(f"The tracked {threads}-thread benchmark run is incomplete.")
    one = float(by_threads[1]["elapsed_seconds"])
    four = float(by_threads[4]["elapsed_seconds"])
    measured_gain = 100.0 * (one - four) / one
    if (
        abs(float(comparison.get("four_thread_elapsed_reduction_percent", -1.0)) - measured_gain)
        > 1.0e-12
        or comparison.get("numeric_result_rows_identical") is not True
        or by_threads[1].get("numeric_result_row_sha256")
        != by_threads[4].get("numeric_result_row_sha256")
        or _integer(comparison.get("recommended_threads_per_case"))
        != BENCHMARKED_THREADS_PER_CASE
        or comparison.get("decision") != "one_core_per_case"
    ):
        raise SchedulerError("The tracked thread-scaling benchmark decision differs.")
    return evidence


def classify_case(root: Path, row: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``complete``, ``runnable``, or ``blocked`` with an audit reason.

    Completed results are skipped only after checking the exact native-case
    identity and exported-result digest.  A complete-looking but unverifiable
    result is blocked rather than overwritten or silently reused.
    """

    paths = _case_paths(root, row)
    identity_exists = paths["identity"].is_file()
    summary_exists = paths["summary"].is_file()
    if identity_exists:
        observed = _read_json(paths["identity"])
        if observed != row.get("native_case_identity"):
            return "blocked", "native_case_identity_mismatch"
    if not summary_exists:
        return "runnable", "no_run_summary"

    summary = _read_json(paths["summary"])
    complete_steps = (
        _integer(summary.get("requested_steps")) == int(row["num_steps"])
        and _integer(summary.get("completed_steps")) == int(row["num_steps"])
    )
    clean_finish = (
        summary.get("failed") is False
        and summary.get("stopped") is False
        and summary.get("exported_results") is True
    )
    if not (complete_steps and clean_finish):
        return "runnable", "incomplete_or_failed_summary_resume_from_checkpoint"
    if not identity_exists:
        return "blocked", "complete_summary_missing_native_case_identity"
    if paths["reuse_receipt"].is_file():
        receipt = _read_json(paths["reuse_receipt"])
        expected_identity = row.get("native_case_identity")
        if (
            row.get("scenario") != campaign.SCENARIOS[0]
            or receipt.get("schema") != campaign.REUSE_RECEIPT_SCHEMA
            or receipt.get("case_id") != row.get("case_id")
            or not isinstance(expected_identity, Mapping)
            or receipt.get("native_case_identity_fingerprint")
            != expected_identity.get("fingerprint")
            or receipt.get("run_summary_sha256") != campaign.sha256_file(paths["summary"])
        ):
            return "blocked", "raw_reuse_receipt_identity_mismatch"
    else:
        if _integer(summary.get("threads")) != BENCHMARKED_THREADS_PER_CASE:
            return "blocked", "complete_summary_wrong_thread_count"
        if summary.get("coupling_mode") != "sil-vorlap" or summary.get("sil_vorlap") is not True:
            return "blocked", "complete_summary_wrong_coupling_mode"
    if not paths["result"].is_file():
        return "blocked", "complete_summary_missing_result_file"
    result_identity = summary.get("result_file_identity")
    if not isinstance(result_identity, dict):
        return "blocked", "complete_summary_missing_result_identity"
    actual_size = paths["result"].stat().st_size
    if int(result_identity.get("size_bytes", -1)) != actual_size:
        return "blocked", "complete_summary_result_size_mismatch"
    if str(result_identity.get("sha256", "")) != campaign.sha256_file(paths["result"]):
        return "blocked", "complete_summary_result_sha256_mismatch"
    return "complete", "exact_identity_complete_result"


def ordered_case_plan(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    complete: list[dict[str, Any]] = []
    runnable: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for source in manifest["cases"]:
        row = dict(source)
        status, reason = classify_case(root, row)
        record = {
            "index": int(row["index"]),
            "case_id": str(row["case_id"]),
            "num_steps": int(row["num_steps"]),
            "status": status,
            "reason": reason,
        }
        if status == "complete":
            complete.append(record)
        elif status == "blocked":
            blocked.append(record)
        else:
            runnable.append(record)
    runnable.sort(key=lambda row: (-int(row["num_steps"]), int(row["index"])))
    complete.sort(key=lambda row: int(row["index"]))
    blocked.sort(key=lambda row: int(row["index"]))
    return {"complete": complete, "runnable": runnable, "blocked": blocked}


def build_srun_command(
    *,
    case_index: int,
    campaign_root: Path,
    qblade_root: Path,
    python_executable: str,
    srun_executable: str,
    threads: int,
) -> list[str]:
    return [
        srun_executable,
        "--exclusive",
        "--exact",
        "-N1",
        "-n1",
        f"-c{threads}",
        "--cpu-bind=cores",
        f"--job-name=ehrmann-{case_index:02d}",
        python_executable,
        "-m",
        "examples.ehrmann2014_naca0018.campaign",
        "run",
        "--campaign",
        str(campaign_root),
        "--qblade-root",
        str(qblade_root),
        "--case-index",
        str(case_index),
        "--threads",
        str(threads),
        "--python",
        python_executable,
        "--resume",
    ]


def _allocation_contract(
    *,
    workers: int,
    threads: int,
    allocated_cpus: int,
    expected_workers: int = DEFAULT_WORKERS,
    expected_allocated_cpus: int = FLIGHT_ALLOCATED_CORES,
) -> None:
    _validate_thread_scaling_benchmark()
    if threads != BENCHMARKED_THREADS_PER_CASE:
        raise SchedulerError(
            f"Each QBlade case must use the benchmarked {BENCHMARKED_THREADS_PER_CASE} thread; "
            f"got {threads}."
        )
    if workers < 1:
        raise SchedulerError("workers must be positive.")
    if allocated_cpus != expected_allocated_cpus:
        raise SchedulerError(
            f"Flight launcher requires exactly {expected_allocated_cpus} allocated physical CPUs; "
            f"got {allocated_cpus}."
        )
    if workers * threads > allocated_cpus:
        raise SchedulerError(
            f"{workers} workers x {threads} threads oversubscribes {allocated_cpus} CPUs."
        )
    if workers != expected_workers:
        raise SchedulerError(
            f"Flight launcher requires {expected_workers} workers to occupy every allocated core; "
            f"got {workers}."
        )


def run_scheduler(
    *,
    campaign_root: Path,
    qblade_root: Path,
    python_executable: str,
    srun_executable: str = "srun",
    workers: int = DEFAULT_WORKERS,
    threads: int = BENCHMARKED_THREADS_PER_CASE,
    allocated_cpus: int = FLIGHT_ALLOCATED_CORES,
    expected_workers: int = DEFAULT_WORKERS,
    expected_allocated_cpus: int = FLIGHT_ALLOCATED_CORES,
    poll_seconds: float = 0.5,
    dry_run: bool = False,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> Path:
    _allocation_contract(
        workers=workers,
        threads=threads,
        allocated_cpus=allocated_cpus,
        expected_workers=expected_workers,
        expected_allocated_cpus=expected_allocated_cpus,
    )
    campaign_root = Path(campaign_root).expanduser().resolve()
    qblade_root = Path(qblade_root).expanduser().resolve()
    state = campaign.validate(campaign_root)
    manifest = state["manifest"]
    root = Path(state["root"])
    plan = ordered_case_plan(root, manifest)
    if plan["blocked"]:
        details = ", ".join(f"{row['case_id']}:{row['reason']}" for row in plan["blocked"])
        raise SchedulerError(f"Refusing unverifiable existing result state: {details}")

    log_root = root / "scheduler_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    report_path = root / "flight_scheduler_report.json"
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "dry_run" if dry_run else "running",
        "campaign_fingerprint": manifest["fingerprint"],
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "elapsed_seconds": None,
        "allocation": {
            "nodes": 1,
            "allocated_physical_cpus": allocated_cpus,
            "node_physical_cpus": FLIGHT_NODE_PHYSICAL_CORES,
            "expected_numa_domains": FLIGHT_NUMA_DOMAINS,
            "node_exclusive": False,
            "workers": workers,
            "threads_per_case": threads,
            "cpu_slots": workers * threads,
            "oversubscribed": False,
        },
        "scheduling_policy": "dynamic_longest_num_steps_first",
        "resume_policy": "exact_complete_results_skip_otherwise_qblade_checkpoint_resume",
        "complete_at_start": plan["complete"],
        "launch_order": [int(row["index"]) for row in plan["runnable"]],
        "cases": [],
    }
    atomic_write_json(report_path, report)

    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "OMP_PLACES": "cores",
            "OMP_PROC_BIND": "close",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    pending = deque(plan["runnable"])
    if dry_run:
        for row in pending:
            report["cases"].append(
                {
                    **row,
                    "command": build_srun_command(
                        case_index=int(row["index"]),
                        campaign_root=root,
                        qblade_root=qblade_root,
                        python_executable=python_executable,
                        srun_executable=srun_executable,
                        threads=threads,
                    ),
                }
            )
        report["finished_at_utc"] = _utc_now()
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_write_json(report_path, report)
        return report_path

    running: dict[subprocess.Popen[Any], tuple[dict[str, Any], Any]] = {}
    failed = False
    while pending or running:
        while pending and len(running) < workers and not failed:
            row = pending.popleft()
            command = build_srun_command(
                case_index=int(row["index"]),
                campaign_root=root,
                qblade_root=qblade_root,
                python_executable=python_executable,
                srun_executable=srun_executable,
                threads=threads,
            )
            log_path = log_root / f"case_{int(row['index']):02d}_{row['case_id']}.log"
            handle = log_path.open("a", encoding="utf-8")
            launched = {
                **row,
                "status": "running",
                "log": str(log_path),
                "command": command,
                "started_at_utc": _utc_now(),
                "finished_at_utc": None,
                "returncode": None,
            }
            report["cases"].append(launched)
            atomic_write_json(report_path, report)
            try:
                process = popen_factory(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                handle.close()
                launched["status"] = "failed_to_launch"
                launched["finished_at_utc"] = _utc_now()
                launched["launch_error"] = str(exc)
                failed = True
                atomic_write_json(report_path, report)
                break
            running[process] = (launched, handle)

        if not running:
            break
        while True:
            finished = [process for process in running if process.poll() is not None]
            if finished:
                break
            time.sleep(poll_seconds)
        for process in finished:
            launched, handle = running.pop(process)
            handle.close()
            launched["returncode"] = int(process.returncode)
            launched["finished_at_utc"] = _utc_now()
            if process.returncode != 0:
                launched["status"] = "failed"
                failed = True
            else:
                source_row = manifest["cases"][int(launched["index"])]
                final_status, reason = classify_case(root, source_row)
                launched["completion_reason"] = reason
                if final_status != "complete":
                    launched["status"] = "failed_post_run_identity_audit"
                    failed = True
                else:
                    launched["status"] = "complete"
            atomic_write_json(report_path, report)

    if pending:
        report["not_launched_after_failure"] = [dict(row) for row in pending]
    report["status"] = "failed" if failed else "complete"
    report["finished_at_utc"] = _utc_now()
    report["elapsed_seconds"] = time.monotonic() - started
    atomic_write_json(report_path, report)
    if failed:
        raise SchedulerError(
            f"At least one case failed; no new cases were launched after failure. See {report_path}."
        )
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--qblade-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--srun", default="srun")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--threads", type=int, default=BENCHMARKED_THREADS_PER_CASE)
    parser.add_argument(
        "--allocated-cpus",
        type=int,
        default=_requested_slurm_cpu_slots(),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report_path = run_scheduler(
            campaign_root=args.campaign,
            qblade_root=args.qblade_root,
            python_executable=args.python,
            srun_executable=args.srun,
            workers=args.workers,
            threads=args.threads,
            allocated_cpus=args.allocated_cpus,
            poll_seconds=args.poll_seconds,
            dry_run=args.dry_run,
        )
    except (campaign.CampaignError, SchedulerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    report = _read_json(report_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "launch_count": len(report["cases"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
