#!/usr/bin/env python3
"""Dynamically schedule the 48 native-QBlade prescribed-motion cases."""

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

from examples.ehrmann2014_naca0018 import qblade_prescribed_motion as campaign

SCHEMA = campaign.PREFLIGHT_SCHEDULER_SCHEMA
PREFLIGHT_CASE_INDEX = campaign.PREFLIGHT_CASE_INDEX
PREFLIGHT_CASE_ID = campaign.PREFLIGHT_CASE_ID
PREFLIGHT_WORKERS = 1
MATRIX_WORKERS = 47
THREADS_PER_CASE = 1


class SchedulerError(RuntimeError):
    """Raised when the shared-node allocation or completed evidence differs."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _allocation_contract(
    *, stage: str, workers: int, threads: int, allocated_cpus: int
) -> None:
    expected = PREFLIGHT_WORKERS if stage == "preflight" else MATRIX_WORKERS
    if stage not in {"preflight", "matrix"}:
        raise SchedulerError("stage must be exactly preflight or matrix.")
    if workers != expected:
        raise SchedulerError(
            f"The Flight {stage} stage requires exactly {expected} workers."
        )
    if threads != THREADS_PER_CASE:
        raise SchedulerError("Each native-QBlade case requires exactly one thread.")
    if allocated_cpus != expected:
        raise SchedulerError(
            f"The {stage} shared allocation must contain exactly {expected} CPU slots."
        )
    if workers * threads != allocated_cpus:
        raise SchedulerError("Every requested shared-node CPU slot must be assigned.")


def build_srun_command(
    *,
    case_index: int,
    campaign_root: Path,
    qblade_root: Path,
    python_executable: str,
    srun_executable: str = "srun",
) -> list[str]:
    return [
        srun_executable,
        "--exclusive",
        "--exact",
        "-N1",
        "-n1",
        "-c1",
        "--cpu-bind=cores",
        f"--job-name=naca-qb-{case_index:02d}",
        python_executable,
        "-m",
        "examples.ehrmann2014_naca0018.qblade_prescribed_motion",
        "run-case",
        "--campaign",
        str(campaign_root),
        "--case-index",
        str(case_index),
        "--qblade-root",
        str(qblade_root),
        "--python-executable",
        python_executable,
        "--resume",
    ]


def _summary_path(root: Path, row: Mapping[str, Any]) -> Path:
    return (
        root / str(row["result_dir"]) / f"{Path(str(row['sim'])).stem}_run_summary.json"
    )


def _summary_is_clean_complete(path: Path, row: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("failed") is False
        and value.get("stopped") is False
        and value.get("exported_results") is True
        and int(value.get("requested_steps", -1)) == int(row["wrapper_steps"])
        and int(value.get("completed_steps", -1)) == int(row["wrapper_steps"])
    )


def execution_plan(
    root: Path, manifest: Mapping[str, Any], *, stage: str
) -> dict[str, Any]:
    """Promote clean exported results, then return complete/runnable indices."""

    status = campaign.campaign_status(root)
    complete = {
        int(row["index"]) for row in status["cases"] if row["status"] == "complete"
    }
    promoted: list[int] = []
    for row in manifest["cases"]:
        index = int(row["index"])
        if index in complete:
            continue
        summary = _summary_path(root, row)
        if _summary_is_clean_complete(summary, row):
            # A complete-looking native run must pass the channel, callback,
            # frame, and history audit.  Failure is blocking; it is never
            # silently replaced by a rerun.
            campaign.postprocess_case(root, index)
            promoted.append(index)
            complete.add(index)
    if manifest["cases"][PREFLIGHT_CASE_INDEX]["case_id"] != PREFLIGHT_CASE_ID:
        raise SchedulerError("The frozen QBlade preflight case identity differs.")
    if stage == "matrix" and PREFLIGHT_CASE_INDEX not in complete:
        raise SchedulerError(
            "Matrix execution requires the complete prescribed-motion audit for "
            f"preflight case {PREFLIGHT_CASE_ID}."
        )
    if stage == "matrix":
        try:
            campaign.validate_preflight_evidence(root, manifest)
        except campaign.CampaignError as exc:
            raise SchedulerError(
                "Matrix execution requires the provenance-bound executed preflight "
                f"report and case audit: {exc}"
            ) from exc
    eligible = (
        {PREFLIGHT_CASE_INDEX}
        if stage == "preflight"
        else set(range(len(manifest["cases"]))) - {PREFLIGHT_CASE_INDEX}
    )
    runnable = [
        {
            "index": int(row["index"]),
            "case_id": str(row["case_id"]),
            "wrapper_steps": int(row["wrapper_steps"]),
        }
        for row in manifest["cases"]
        if int(row["index"]) in eligible and int(row["index"]) not in complete
    ]
    runnable.sort(key=lambda row: (-row["wrapper_steps"], row["index"]))
    return {
        "complete": sorted(complete),
        "promoted_without_rerun": sorted(promoted),
        "runnable": runnable,
        "preflight_complete": PREFLIGHT_CASE_INDEX in complete,
    }


def run_scheduler(
    *,
    stage: str,
    campaign_root: Path,
    qblade_root: Path,
    python_executable: str,
    srun_executable: str = "srun",
    workers: int | None = None,
    threads: int = THREADS_PER_CASE,
    allocated_cpus: int | None = None,
    poll_seconds: float = 0.5,
    dry_run: bool = False,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> Path:
    expected_workers = PREFLIGHT_WORKERS if stage == "preflight" else MATRIX_WORKERS
    workers = expected_workers if workers is None else workers
    allocated_cpus = expected_workers if allocated_cpus is None else allocated_cpus
    _allocation_contract(
        stage=stage,
        workers=workers,
        threads=threads,
        allocated_cpus=allocated_cpus,
    )
    state = campaign.validate_campaign(campaign_root, qblade_root)
    root = Path(state["root"])
    manifest = state["manifest"]
    plan = execution_plan(root, manifest, stage=stage)
    pending = deque(plan["runnable"])
    report_path = (
        root / campaign.PREFLIGHT_SCHEDULER_REPORT
        if stage == "preflight"
        else root / "qblade_matrix_flight_scheduler_report.json"
    )
    log_root = root / "scheduler_logs"
    log_root.mkdir(exist_ok=True)
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "dry_run" if dry_run else "running",
        "campaign_fingerprint": manifest["fingerprint"],
        "stage": stage,
        "started_at_utc": _utc_now(),
        "finished_at_utc": None,
        "elapsed_seconds": None,
        "allocation": {
            "nodes": 1,
            "node_exclusive": False,
            "allocated_physical_cpus": allocated_cpus,
            "workers": workers,
            "threads_per_case": threads,
            "fresh_stage_slot_utilization": 1.0,
            "initial_resume_slot_utilization": min(len(pending), workers) / workers,
        },
        "execution_environment": {
            "kind": "flight_slurm",
            "platform": "linux_x86_64",
            "slurm_job_id": str(os.environ.get("SLURM_JOB_ID", "unavailable")),
        },
        "execution_contract": {
            "native_qblade_aerodynamics": True,
            "external_callback_updates": 0,
            "required_cases": 48,
            "complete_at_start": len(plan["complete"]),
            "promoted_without_rerun": plan["promoted_without_rerun"],
            "runnable_at_start": len(pending),
            "checkpoint_resume": True,
            "preflight_case_index": PREFLIGHT_CASE_INDEX,
            "preflight_case_id": PREFLIGHT_CASE_ID,
            "preflight_complete_at_start": plan["preflight_complete"],
        },
        "scheduling_policy": "longest_step_count_first_one_case_per_cpu_slot",
        "cases": [],
    }
    if stage == "preflight" and plan["preflight_complete"] and not pending:
        report["cases"].append(
            {
                "index": PREFLIGHT_CASE_INDEX,
                "case_id": PREFLIGHT_CASE_ID,
                "status": "complete",
                "completion_source": "preexisting_exact_case_audit",
                "command": None,
            }
        )
    _write_json(report_path, report)

    if dry_run:
        for row in pending:
            report["cases"].append(
                {
                    **row,
                    "status": "planned",
                    "command": build_srun_command(
                        case_index=row["index"],
                        campaign_root=root,
                        qblade_root=qblade_root.resolve(),
                        python_executable=python_executable,
                        srun_executable=srun_executable,
                    ),
                }
            )
        report["finished_at_utc"] = _utc_now()
        report["elapsed_seconds"] = time.monotonic() - started
        _write_json(report_path, report)
        return report_path

    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_PLACES": "cores",
            "OMP_PROC_BIND": "close",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    running: dict[subprocess.Popen[Any], tuple[dict[str, Any], Any]] = {}
    failed = False
    while pending or running:
        while pending and len(running) < workers and not failed:
            row = pending.popleft()
            command = build_srun_command(
                case_index=row["index"],
                campaign_root=root,
                qblade_root=qblade_root.resolve(),
                python_executable=python_executable,
                srun_executable=srun_executable,
            )
            log = log_root / f"case_{row['index']:02d}_{row['case_id']}.log"
            handle = log.open("a", encoding="utf-8")
            record = {
                **row,
                "status": "running",
                "command": command,
                "log": str(log),
                "started_at_utc": _utc_now(),
                "finished_at_utc": None,
                "returncode": None,
            }
            report["cases"].append(record)
            _write_json(report_path, report)
            try:
                process = popen_factory(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                handle.close()
                record["status"] = "failed_to_launch"
                record["launch_error"] = str(exc)
                record["finished_at_utc"] = _utc_now()
                failed = True
                _write_json(report_path, report)
                break
            running[process] = (record, handle)

        if not running:
            break
        while True:
            finished = [process for process in running if process.poll() is not None]
            if finished:
                break
            time.sleep(poll_seconds)
        for process in finished:
            record, handle = running.pop(process)
            handle.close()
            record["returncode"] = int(process.returncode)
            record["finished_at_utc"] = _utc_now()
            if process.returncode != 0:
                record["status"] = "failed"
                failed = True
            else:
                try:
                    audit = campaign.postprocess_case(root, int(record["index"]))
                except (campaign.CampaignError, OSError, ValueError) as exc:
                    record["status"] = "failed_postprocess_audit"
                    record["audit_error"] = str(exc)
                    failed = True
                else:
                    record["status"] = "complete"
                    record["canonical_history_sha256"] = audit["canonical_history"][
                        "sha256"
                    ]
            _write_json(report_path, report)

    if pending:
        report["not_launched_after_failure"] = list(pending)
    if failed:
        report["status"] = "failed"
    else:
        final = campaign.campaign_status(root)
        expected_complete = 1 if stage == "preflight" else 48
        if (
            (stage == "preflight" and final["cases"][PREFLIGHT_CASE_INDEX]["status"] != "complete")
            or (stage == "matrix" and final["complete_cases"] != expected_complete)
        ):
            report["status"] = "failed"
            report["final_audit_error"] = (
                f"{stage}_gate_incomplete_with_{final['complete_cases']}_total_cases"
            )
            failed = True
        else:
            report["status"] = "complete"
            if stage == "matrix":
                results = campaign.finalize_results(root)
                report["results_manifest"] = {
                    "path": str(results),
                    "sha256": campaign._sha256(results),
                }
            else:
                report["preflight_gate"] = {
                    "status": "complete",
                    "case_index": PREFLIGHT_CASE_INDEX,
                    "case_id": PREFLIGHT_CASE_ID,
                }
    report["finished_at_utc"] = _utc_now()
    report["elapsed_seconds"] = time.monotonic() - started
    _write_json(report_path, report)
    if failed:
        raise SchedulerError(
            "At least one native-QBlade prescribed-motion case failed; see "
            f"{report_path}."
        )
    return report_path


def _requested_slots(environ: Mapping[str, str] | None = None) -> int | None:
    values = os.environ if environ is None else environ
    try:
        tasks = int(values["SLURM_NTASKS"])
        cpus = int(values["SLURM_CPUS_PER_TASK"])
    except (KeyError, TypeError, ValueError):
        return None
    return tasks * cpus if tasks > 0 and cpus > 0 else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "matrix"), required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--qblade-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--srun", default="srun")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--threads", type=int, default=THREADS_PER_CASE)
    parser.add_argument("--allocated-cpus", type=int, default=_requested_slots())
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        path = run_scheduler(
            campaign_root=args.campaign,
            stage=args.stage,
            qblade_root=args.qblade_root,
            python_executable=args.python,
            srun_executable=args.srun,
            workers=args.workers,
            threads=args.threads,
            allocated_cpus=args.allocated_cpus,
            poll_seconds=args.poll_seconds,
            dry_run=args.dry_run,
        )
    except (campaign.CampaignError, SchedulerError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "report": str(path),
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
