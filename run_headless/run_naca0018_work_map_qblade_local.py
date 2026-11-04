#!/usr/bin/env python3
"""Run the native-QBlade prescribed-motion producer locally inside Docker.

The host launcher caps the container at eight CPUs.  This scheduler runs the
explicit one-case preflight first and then at most eight independent one-thread
QBlade cases concurrently.  It writes the same provenance-bound scheduler-v2
contract consumed by the producer and map assembler without pretending that a
local Docker execution was a Flight allocation.
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

from examples.ehrmann2014_naca0018 import qblade_prescribed_motion as campaign
from run_headless.run_naca0018_work_map_qblade_flight import execution_plan


SCHEMA = campaign.PREFLIGHT_SCHEDULER_SCHEMA
MAX_LOCAL_CPUS = 8


class LocalSchedulerError(RuntimeError):
    """Raised when the local Docker execution contract differs."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _image_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise LocalSchedulerError(
            "VORLAP_CONTAINER_IMAGE_ID must be an exact sha256:<64 hex> image ID."
        )
    return value


def _execution_contract(*, stage: str, workers: int, cpu_limit: int) -> None:
    if stage not in {"preflight", "matrix"}:
        raise LocalSchedulerError("stage must be exactly preflight or matrix.")
    if type(cpu_limit) is not int or not 1 <= cpu_limit <= MAX_LOCAL_CPUS:
        raise LocalSchedulerError("Local Docker CPU limit must be in [1, 8].")
    if stage == "preflight" and workers != 1:
        raise LocalSchedulerError("The local preflight requires exactly one worker.")
    if stage == "matrix" and (
        type(workers) is not int or not 1 <= workers <= cpu_limit
    ):
        raise LocalSchedulerError(
            "Local matrix workers must be positive and no greater than the Docker CPU limit."
        )


def build_command(
    *,
    case_index: int,
    campaign_root: Path,
    qblade_root: Path,
    python_executable: str,
) -> list[str]:
    return campaign.command_for_case(
        campaign_root,
        case_index=case_index,
        qblade_root=qblade_root,
        python_executable=python_executable,
        resume=True,
    )


def run_scheduler(
    *,
    stage: str,
    campaign_root: Path,
    qblade_root: Path,
    python_executable: str,
    workers: int,
    cpu_limit: int,
    container_image_id: str,
    poll_seconds: float = 0.5,
    dry_run: bool = False,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> Path:
    _execution_contract(stage=stage, workers=workers, cpu_limit=cpu_limit)
    image_id = _image_id(container_image_id)
    state = campaign.validate_campaign(campaign_root, qblade_root)
    root = Path(state["root"])
    manifest = state["manifest"]
    plan = execution_plan(root, manifest, stage=stage)
    pending = deque(plan["runnable"])
    report_path = (
        root / campaign.PREFLIGHT_SCHEDULER_REPORT
        if stage == "preflight"
        else root / "qblade_matrix_local_scheduler_report.json"
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
            "allocated_physical_cpus": workers,
            "workers": workers,
            "threads_per_case": 1,
            "fresh_stage_slot_utilization": min(len(pending), workers) / workers,
            "initial_resume_slot_utilization": min(len(pending), workers) / workers,
        },
        "execution_environment": {
            "kind": "local_docker",
            "platform": "linux_amd64",
            "container_image_id": image_id,
            "container_cpu_limit": cpu_limit,
        },
        "execution_contract": {
            "native_qblade_aerodynamics": True,
            "external_callback_updates": 0,
            "required_cases": 48,
            "complete_at_start": len(plan["complete"]),
            "promoted_without_rerun": plan["promoted_without_rerun"],
            "runnable_at_start": len(pending),
            "checkpoint_resume": True,
            "preflight_case_index": campaign.PREFLIGHT_CASE_INDEX,
            "preflight_case_id": campaign.PREFLIGHT_CASE_ID,
            "preflight_complete_at_start": plan["preflight_complete"],
        },
        "scheduling_policy": "longest_step_count_first_one_thread_per_local_cpu",
        "cases": [],
    }
    if stage == "preflight" and plan["preflight_complete"] and not pending:
        report["cases"].append(
            {
                "index": campaign.PREFLIGHT_CASE_INDEX,
                "case_id": campaign.PREFLIGHT_CASE_ID,
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
                    "command": build_command(
                        case_index=int(row["index"]),
                        campaign_root=root,
                        qblade_root=qblade_root.resolve(),
                        python_executable=python_executable,
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
            command = build_command(
                case_index=int(row["index"]),
                campaign_root=root,
                qblade_root=qblade_root.resolve(),
                python_executable=python_executable,
            )
            log = log_root / f"case_{int(row['index']):02d}_{row['case_id']}.log"
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
                    record["canonical_history_sha256"] = audit[
                        "canonical_history"
                    ]["sha256"]
            _write_json(report_path, report)

    if pending:
        report["not_launched_after_failure"] = list(pending)
    if failed:
        report["status"] = "failed"
    else:
        final = campaign.campaign_status(root)
        expected_complete = 1 if stage == "preflight" else 48
        if (
            stage == "preflight"
            and final["cases"][campaign.PREFLIGHT_CASE_INDEX]["status"] != "complete"
        ) or (stage == "matrix" and final["complete_cases"] != expected_complete):
            report["status"] = "failed"
            report["final_audit_error"] = (
                f"{stage}_gate_incomplete_with_{final['complete_cases']}_total_cases"
            )
            failed = True
        else:
            report["status"] = "complete"
            if stage == "preflight":
                report["preflight_gate"] = {
                    "status": "complete",
                    "case_index": campaign.PREFLIGHT_CASE_INDEX,
                    "case_id": campaign.PREFLIGHT_CASE_ID,
                }
            else:
                results = campaign.finalize_results(root)
                report["results_manifest"] = {
                    "path": str(results),
                    "sha256": campaign._sha256(results),
                }
    report["finished_at_utc"] = _utc_now()
    report["elapsed_seconds"] = time.monotonic() - started
    _write_json(report_path, report)
    if failed:
        raise LocalSchedulerError(
            f"At least one local prescribed-motion case failed; see {report_path}."
        )
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "matrix"), required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--qblade-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--cpu-limit", type=int, required=True)
    parser.add_argument(
        "--container-image-id",
        default=os.environ.get("VORLAP_CONTAINER_IMAGE_ID", ""),
    )
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        path = run_scheduler(
            stage=args.stage,
            campaign_root=args.campaign,
            qblade_root=args.qblade_root,
            python_executable=args.python,
            workers=args.workers,
            cpu_limit=args.cpu_limit,
            container_image_id=args.container_image_id,
            poll_seconds=args.poll_seconds,
            dry_run=args.dry_run,
        )
    except (campaign.CampaignError, LocalSchedulerError, OSError, ValueError) as exc:
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
