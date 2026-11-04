#!/usr/bin/env python3
"""Run only the 20 new Ehrmann work-map hybrids on Flight.

The schema-v2 campaign must contain 20 exact-identity raw-QBlade results copied
from the explicitly supplied frozen campaign and 20 work-map hybrid cases.  A
shared-node allocation therefore reserves exactly 20 physical cores and runs
one single-threaded QBlade process per new hybrid.  Complete hybrid results are
identity checked and skipped on a resubmission; incomplete hybrids retain the
ordinary 5,000-step QBlade checkpoint/resume path.

The scheduler report intentionally retains the released
``vorlap-ehrmann2014-flight-scheduler-v1`` schema consumed by the independent
validation-evidence gate.  ``execution_contract`` distinguishes this dedicated
20-case work-map path from the legacy 40-case launcher.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
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
from run_headless import run_ehrmann2014_flight as legacy
from vorlap.qblade_run_support import (
    atomic_write_json,
    find_qblade_library,
    qblade_runtime_dependency_manifest,
)

SCHEMA = legacy.SCHEMA
EXECUTION_CONTRACT_SCHEMA = "vorlap-ehrmann2014-work-map-flight-execution-v1"
WORKERS = 20
THREADS_PER_CASE = 1
ALLOCATED_PHYSICAL_CPUS = 20
FLIGHT_NODE_PHYSICAL_CORES = 112
FLIGHT_NUMA_DOMAINS = 8


class WorkMapSchedulerError(RuntimeError):
    """Raised when schema-v2 inputs or result state violate the run contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkMapSchedulerError(f"Cannot read required JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkMapSchedulerError(f"Required JSON object is invalid: {path}")
    return value


def _regular_file(path: Path, label: str) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_symlink() or not expanded.is_file():
        raise WorkMapSchedulerError(
            f"{label} must be a regular non-symlink file: {expanded}"
        )
    return expanded.resolve()


def _regular_directory(path: Path, label: str) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_symlink() or not expanded.is_dir():
        raise WorkMapSchedulerError(
            f"{label} must be a real non-symlink directory: {expanded}"
        )
    return expanded.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qblade_runtime_identity(qblade_root: Path) -> dict[str, str]:
    root = _regular_directory(qblade_root, "QBlade runtime")
    library = find_qblade_library(root)
    _dependencies, dependency_fingerprint = qblade_runtime_dependency_manifest(root)
    observed = {
        "version": campaign.FROZEN_QBLADE_VERSION,
        "library_sha256": _sha256(library),
        "runtime_dependency_fingerprint": dependency_fingerprint,
    }
    expected = {
        "version": campaign.FROZEN_QBLADE_VERSION,
        "library_sha256": campaign.FROZEN_QBLADE_LIBRARY_SHA256,
        "runtime_dependency_fingerprint": (
            campaign.FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
        ),
    }
    if observed != expected or root.name != f"QBladeCE_{campaign.FROZEN_QBLADE_VERSION}":
        raise WorkMapSchedulerError(
            "QBlade runtime differs from the exact runtime used by the reused raw cases."
        )
    return observed


def _allocation_contract(*, workers: int, threads: int, allocated_cpus: int) -> None:
    legacy._validate_thread_scaling_benchmark()
    if threads != THREADS_PER_CASE:
        raise WorkMapSchedulerError(
            f"Each QBlade work-map case must use one thread; got {threads}."
        )
    if workers != WORKERS:
        raise WorkMapSchedulerError(
            f"The work-map launcher requires exactly {WORKERS} workers; got {workers}."
        )
    if allocated_cpus != ALLOCATED_PHYSICAL_CPUS:
        raise WorkMapSchedulerError(
            "The work-map launcher requires exactly "
            f"{ALLOCATED_PHYSICAL_CPUS} allocated physical CPUs; got {allocated_cpus}."
        )
    if workers * threads != allocated_cpus:
        raise WorkMapSchedulerError(
            "Every allocated campaign CPU must belong to one QBlade worker."
        )


def validate_execution_plan(
    campaign_root: Path,
    *,
    work_map: Path,
    raw_reuse_root: Path,
    qblade_root: Path,
) -> dict[str, Any]:
    """Bind the campaign to one map and exactly 20 reused raw results.

    A resume may contain completed hybrid cases, but raw cases must always be
    complete exact-identity copies from ``raw_reuse_root`` and may never enter
    the runnable queue.
    """

    work_map_path = _regular_file(work_map, "WORK_MAP")
    reuse_root = _regular_directory(raw_reuse_root, "RAW_REUSE_ROOT")
    state = campaign.validate(Path(campaign_root))
    root = Path(state["root"])
    manifest = state["manifest"]
    runtime_identity = _qblade_runtime_identity(qblade_root)
    identity = manifest.get("work_map_identity")
    if (
        manifest.get("schema") != campaign.WORK_MAP_SCHEMA
        or manifest.get("comparison_series") != list(campaign.WORK_MAP_SCENARIOS)
        or not isinstance(identity, Mapping)
        or identity.get("schema") != "vorlap-airfoil-force-motion-work-map-v1"
        or identity.get("sha256") != _sha256(work_map_path)
        or identity.get("confidence_multiplier") != 1.96
        or manifest.get("qblade_runtime_identity") != runtime_identity
    ):
        raise WorkMapSchedulerError(
            "Campaign is not the exact schema-v2 campaign for the supplied WORK_MAP."
        )

    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 40:
        raise WorkMapSchedulerError(
            "The work-map campaign must contain exactly 40 cases."
        )
    raw = [
        row for row in cases if row.get("scenario") == campaign.WORK_MAP_SCENARIOS[0]
    ]
    hybrid = [
        row for row in cases if row.get("scenario") == campaign.WORK_MAP_SCENARIOS[1]
    ]
    if len(raw) != 20 or len(hybrid) != 20:
        raise WorkMapSchedulerError(
            "The work-map campaign must contain exactly 20 raw and 20 hybrid cases."
        )

    plan = legacy.ordered_case_plan(root, manifest)
    if plan["blocked"]:
        details = ", ".join(
            f"{row['case_id']}:{row['reason']}" for row in plan["blocked"]
        )
        raise WorkMapSchedulerError(
            f"Refusing unverifiable existing result state: {details}"
        )
    status_by_id = {
        str(row["case_id"]): str(row["status"])
        for group in (plan["complete"], plan["runnable"])
        for row in group
    }

    for row in raw:
        case_id = str(row["case_id"])
        if row.get("raw_reuse_status") != "reused_exact_identity":
            raise WorkMapSchedulerError(
                f"Raw case {case_id} was not generated as exact-identity reuse."
            )
        if status_by_id.get(case_id) != "complete":
            raise WorkMapSchedulerError(
                f"Raw case {case_id} is not an exact complete reused result."
            )
        result_dir = root / str(row["result_dir"])
        receipt_path = result_dir / "reuse_receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise WorkMapSchedulerError(
                f"Raw case {case_id} lacks a regular reuse receipt."
            )
        receipt = _read_json(receipt_path)
        expected_source_entry = reuse_root / case_id
        expected_source = expected_source_entry.resolve()
        observed_source = (
            Path(str(receipt.get("source_result", ""))).expanduser().resolve()
        )
        if (
            expected_source_entry.is_symlink()
            or not expected_source_entry.is_dir()
            or receipt.get("schema") != campaign.REUSE_RECEIPT_SCHEMA
            or observed_source != expected_source
            or receipt.get("copied_result") != str(row["result_dir"])
            or receipt.get("qblade_library_sha256")
            != runtime_identity["library_sha256"]
            or receipt.get("qblade_runtime_dependency_fingerprint")
            != runtime_identity["runtime_dependency_fingerprint"]
            or receipt.get("sil_vorlap_max_abs_force") != 0.0
            or receipt.get("coupling_mode") != "sil-vorlap"
            or receipt.get("native_callback_update_count") != 0
        ):
            raise WorkMapSchedulerError(
                f"Raw case {case_id} is not bound to the supplied RAW_REUSE_ROOT."
            )

    for row in hybrid:
        case_id = str(row["case_id"])
        if row.get("raw_reuse_status") != "no_candidate_run_required":
            raise WorkMapSchedulerError(
                f"Hybrid case {case_id} has an invalid reuse status."
            )
        receipt = root / str(row["result_dir"]) / "reuse_receipt.json"
        if receipt.exists() or receipt.is_symlink():
            raise WorkMapSchedulerError(
                f"Hybrid case {case_id} may not use a raw-result reuse receipt."
            )
        if status_by_id.get(case_id) not in {"complete", "runnable"}:
            raise WorkMapSchedulerError(f"Hybrid case {case_id} is not resumable.")

    runnable = [
        row
        for row in plan["runnable"]
        if status_by_id.get(str(row["case_id"])) == "runnable"
    ]
    hybrid_ids = {str(row["case_id"]) for row in hybrid}
    if any(str(row["case_id"]) not in hybrid_ids for row in runnable):
        raise WorkMapSchedulerError("A raw case entered the work-map runnable queue.")
    hybrid_complete = [
        row for row in plan["complete"] if str(row["case_id"]) in hybrid_ids
    ]
    if len(runnable) + len(hybrid_complete) != 20:
        raise WorkMapSchedulerError("The 20 hybrid cases are not fully accounted for.")

    return {
        "root": root,
        "manifest": manifest,
        "plan": plan,
        "runnable_hybrids": runnable,
        "hybrid_complete": hybrid_complete,
        "work_map": work_map_path,
        "raw_reuse_root": reuse_root,
        "qblade_runtime_identity": runtime_identity,
    }


def run_scheduler(
    *,
    campaign_root: Path,
    qblade_root: Path,
    work_map: Path,
    raw_reuse_root: Path,
    python_executable: str,
    srun_executable: str = "srun",
    workers: int = WORKERS,
    threads: int = THREADS_PER_CASE,
    allocated_cpus: int = ALLOCATED_PHYSICAL_CPUS,
    poll_seconds: float = 0.5,
    dry_run: bool = False,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> Path:
    _allocation_contract(
        workers=workers, threads=threads, allocated_cpus=allocated_cpus
    )
    state = validate_execution_plan(
        campaign_root,
        work_map=work_map,
        raw_reuse_root=raw_reuse_root,
        qblade_root=qblade_root,
    )
    root = Path(state["root"])
    manifest = state["manifest"]
    plan = state["plan"]
    runnable = list(state["runnable_hybrids"])

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
        "execution_contract": {
            "schema": EXECUTION_CONTRACT_SCHEMA,
            "campaign_schema": campaign.WORK_MAP_SCHEMA,
            "work_map_sha256": manifest["work_map_identity"]["sha256"],
            "qblade_runtime_identity": state["qblade_runtime_identity"],
            "raw_reuse_root": str(state["raw_reuse_root"]),
            "exact_raw_reuse_case_count": 20,
            "hybrid_case_count": 20,
            "hybrid_complete_at_start": len(state["hybrid_complete"]),
            "hybrid_runnable_at_start": len(runnable),
        },
        "allocation": {
            "nodes": 1,
            "allocated_physical_cpus": allocated_cpus,
            "node_physical_cpus": FLIGHT_NODE_PHYSICAL_CORES,
            "expected_numa_domains": FLIGHT_NUMA_DOMAINS,
            "node_exclusive": False,
            "workers": workers,
            "threads_per_case": threads,
            "cpu_slots": workers * threads,
            "initial_slot_utilization": (
                min(len(runnable), workers) / workers if workers else 0.0
            ),
            "oversubscribed": False,
        },
        "scheduling_policy": "dynamic_longest_num_steps_first_hybrids_only",
        "resume_policy": "exact_complete_results_skip_otherwise_qblade_checkpoint_resume",
        "complete_at_start": plan["complete"],
        "launch_order": [int(row["index"]) for row in runnable],
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
    pending = deque(runnable)
    if dry_run:
        for row in pending:
            report["cases"].append(
                {
                    **row,
                    "command": legacy.build_srun_command(
                        case_index=int(row["index"]),
                        campaign_root=root,
                        qblade_root=Path(qblade_root).expanduser().resolve(),
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
            command = legacy.build_srun_command(
                case_index=int(row["index"]),
                campaign_root=root,
                qblade_root=Path(qblade_root).expanduser().resolve(),
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
                final_status, reason = legacy.classify_case(root, source_row)
                launched["completion_reason"] = reason
                if final_status != "complete":
                    launched["status"] = "failed_post_run_identity_audit"
                    failed = True
                else:
                    launched["status"] = "complete"
            atomic_write_json(report_path, report)

    if pending:
        report["not_launched_after_failure"] = [dict(row) for row in pending]
    if failed:
        report["status"] = "failed"
    else:
        final_state = validate_execution_plan(
            root,
            work_map=state["work_map"],
            raw_reuse_root=state["raw_reuse_root"],
            qblade_root=qblade_root,
        )
        if final_state["runnable_hybrids"] or len(final_state["hybrid_complete"]) != 20:
            report["status"] = "failed"
            report["final_audit_error"] = "not_all_20_hybrid_cases_complete"
            failed = True
        else:
            report["status"] = "complete"
            report["final_execution_audit"] = {
                "exact_raw_reuse_case_count": 20,
                "complete_hybrid_case_count": 20,
                "runnable_case_count": 0,
            }
    report["finished_at_utc"] = _utc_now()
    report["elapsed_seconds"] = time.monotonic() - started
    atomic_write_json(report_path, report)
    if failed:
        raise WorkMapSchedulerError(
            "At least one work-map hybrid failed; no new cases were launched after failure. "
            f"See {report_path}."
        )
    return report_path


def _requested_slurm_cpu_slots(environ: Mapping[str, str] | None = None) -> int:
    values = os.environ if environ is None else environ
    try:
        tasks = int(values["SLURM_NTASKS"])
        cpus = int(values["SLURM_CPUS_PER_TASK"])
    except (KeyError, TypeError, ValueError):
        return ALLOCATED_PHYSICAL_CPUS
    return tasks * cpus if tasks > 0 and cpus > 0 else ALLOCATED_PHYSICAL_CPUS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--qblade-root", type=Path, required=True)
    parser.add_argument("--work-map", type=Path, required=True)
    parser.add_argument("--raw-reuse-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--srun", default="srun")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--threads", type=int, default=THREADS_PER_CASE)
    parser.add_argument(
        "--allocated-cpus", type=int, default=_requested_slurm_cpu_slots()
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
            work_map=args.work_map,
            raw_reuse_root=args.raw_reuse_root,
            python_executable=args.python,
            srun_executable=args.srun,
            workers=args.workers,
            threads=args.threads,
            allocated_cpus=args.allocated_cpus,
            poll_seconds=args.poll_seconds,
            dry_run=args.dry_run,
        )
    except (campaign.CampaignError, WorkMapSchedulerError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    report = _read_json(report_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": report["status"],
                "elapsed_seconds": report["elapsed_seconds"],
                "exact_raw_reuse_case_count": report["execution_contract"][
                    "exact_raw_reuse_case_count"
                ],
                "hybrid_complete_at_start": report["execution_contract"][
                    "hybrid_complete_at_start"
                ],
                "launch_count": len(report["cases"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
