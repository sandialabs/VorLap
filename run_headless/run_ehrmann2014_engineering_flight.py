#!/usr/bin/env python3
"""Run the 34 new cases in the Ehrmann engineering lock-in campaign.

Eighteen nominal raw-QBlade results are exact-identity copies from the prior
40-case campaign.  Eight new raw cases and all twenty-six latest
engineering-bridge cases are launched on 34 shared physical cores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.ehrmann2014_naca0018 import campaign
from run_headless import run_ehrmann2014_flight as legacy
from run_headless.run_ehrmann2014_work_map_flight import _qblade_runtime_identity
from vorlap.qblade_run_support import atomic_write_json

EXECUTION_SCHEMA = "vorlap-ehrmann2014-engineering-flight-execution-v2"
WORKERS = 34
THREADS = 1
ALLOCATED_CPUS = 34


class EngineeringSchedulerError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_execution_plan(
    campaign_root: Path,
    *,
    calibration: Path,
    raw_reuse_root: Path,
    qblade_root: Path,
) -> dict[str, Any]:
    calibration_path = Path(calibration).expanduser().resolve()
    reuse_root = Path(raw_reuse_root).expanduser().resolve()
    if calibration_path.is_symlink() or not calibration_path.is_file():
        raise EngineeringSchedulerError("CALIBRATION must be a regular file.")
    if reuse_root.is_symlink() or not reuse_root.is_dir():
        raise EngineeringSchedulerError("RAW_REUSE_ROOT must be a real directory.")
    state = campaign.validate(campaign_root)
    root = Path(state["root"])
    manifest = state["manifest"]
    identity = manifest.get("engineering_lockin_identity")
    runtime_identity = _qblade_runtime_identity(Path(qblade_root))
    if (
        manifest.get("schema") != campaign.ENGINEERING_LOCKIN_SCHEMA
        or manifest.get("comparison_series")
        != list(campaign.ENGINEERING_LOCKIN_SCENARIOS)
        or not isinstance(identity, Mapping)
        or identity.get("sha256") != _sha256(calibration_path)
        or identity.get("nalu_required") is not False
        or manifest.get("qblade_runtime_identity") != runtime_identity
    ):
        raise EngineeringSchedulerError(
            "Campaign does not match the supplied engineering calibration/runtime."
        )
    plan = legacy.ordered_case_plan(root, manifest)
    if plan["blocked"]:
        raise EngineeringSchedulerError(
            "Existing result state is unverifiable: "
            + ", ".join(f"{row['case_id']}:{row['reason']}" for row in plan["blocked"])
        )
    status_by_id = {
        str(row["case_id"]): str(row["status"])
        for group in (plan["complete"], plan["runnable"])
        for row in group
    }
    raw = [
        row
        for row in manifest["cases"]
        if row["scenario"] == campaign.ENGINEERING_LOCKIN_SCENARIOS[0]
    ]
    hybrid = [
        row
        for row in manifest["cases"]
        if row["scenario"] == campaign.ENGINEERING_LOCKIN_SCENARIOS[1]
    ]
    if len(raw) != 26 or len(hybrid) != 26:
        raise EngineeringSchedulerError("Expected exactly 26 raw and 26 hybrid cases.")
    reused = 0
    new_raw = 0
    for row in raw:
        case_id = str(row["case_id"])
        if row.get("raw_reuse_status") == "no_candidate_run_required":
            if status_by_id[case_id] not in {"runnable", "complete"}:
                raise EngineeringSchedulerError(
                    f"New raw case {case_id} is not resumable."
                )
            new_raw += 1
            continue
        if row.get("raw_reuse_status") != "reused_exact_identity":
            raise EngineeringSchedulerError(
                f"Nominal raw case {case_id} was not exact-identity reused."
            )
        if status_by_id[case_id] != "complete":
            raise EngineeringSchedulerError(
                f"Reused raw case {case_id} is not complete."
            )
        receipt_path = root / str(row["result_dir"]) / "reuse_receipt.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise EngineeringSchedulerError(f"Reused raw case {case_id} lacks receipt.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        source = reuse_root / case_id
        if (
            source.is_symlink()
            or not source.is_dir()
            or Path(str(receipt.get("source_result", ""))).resolve() != source.resolve()
            or receipt.get("qblade_library_sha256")
            != runtime_identity["library_sha256"]
            or receipt.get("qblade_runtime_dependency_fingerprint")
            != runtime_identity["runtime_dependency_fingerprint"]
            or receipt.get("sil_vorlap_max_abs_force") != 0.0
            or receipt.get("native_callback_update_count") != 0
        ):
            raise EngineeringSchedulerError(
                f"Reused raw case {case_id} is not bound to RAW_REUSE_ROOT."
            )
        reused += 1
    if reused != 18 or new_raw != 8:
        raise EngineeringSchedulerError("Expected 18 reused and 8 new raw cases.")
    if any(
        status_by_id[str(row["case_id"])] not in {"runnable", "complete"}
        for row in hybrid
    ):
        raise EngineeringSchedulerError("At least one hybrid is not resumable.")
    runnable = list(plan["runnable"])
    if len(runnable) > WORKERS:
        raise EngineeringSchedulerError(
            f"At most {WORKERS} new cases may be runnable; found {len(runnable)}."
        )
    return {
        "root": root,
        "manifest": manifest,
        "plan": plan,
        "runnable": runnable,
        "calibration": calibration_path,
        "reuse_root": reuse_root,
        "runtime_identity": runtime_identity,
        "reused_raw": reused,
        "new_raw": new_raw,
    }


def run_scheduler(
    *,
    campaign_root: Path,
    qblade_root: Path,
    calibration: Path,
    raw_reuse_root: Path,
    python_executable: str,
    srun_executable: str = "srun",
    workers: int = WORKERS,
    threads: int = THREADS,
    allocated_cpus: int = ALLOCATED_CPUS,
    dry_run: bool = False,
) -> Path:
    if (workers, threads, allocated_cpus) != (WORKERS, THREADS, ALLOCATED_CPUS):
        raise EngineeringSchedulerError(
            "Engineering allocation must be exactly 34x1 cores."
        )
    state = validate_execution_plan(
        campaign_root,
        calibration=calibration,
        raw_reuse_root=raw_reuse_root,
        qblade_root=qblade_root,
    )
    report_path = legacy.run_scheduler(
        campaign_root=state["root"],
        qblade_root=qblade_root,
        python_executable=python_executable,
        srun_executable=srun_executable,
        workers=workers,
        threads=threads,
        allocated_cpus=allocated_cpus,
        expected_workers=WORKERS,
        expected_allocated_cpus=ALLOCATED_CPUS,
        dry_run=dry_run,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    final = (
        state
        if dry_run
        else validate_execution_plan(
            campaign_root,
            calibration=calibration,
            raw_reuse_root=raw_reuse_root,
            qblade_root=qblade_root,
        )
    )
    if not dry_run and final["plan"]["runnable"]:
        raise EngineeringSchedulerError(
            "Final campaign audit still has runnable cases."
        )
    report["execution_contract"] = {
        "schema": EXECUTION_SCHEMA,
        "campaign_schema": campaign.ENGINEERING_LOCKIN_SCHEMA,
        "calibration_sha256": state["manifest"]["engineering_lockin_identity"][
            "sha256"
        ],
        "raw_reuse_root": str(state["reuse_root"]),
        "exact_raw_reuse_case_count": 18,
        "new_raw_case_count": 8,
        "hybrid_case_count": 26,
        "runnable_at_start": len(state["runnable"]),
        "complete_at_end": (len(final["plan"]["complete"]) if not dry_run else None),
        "nalu_required": False,
    }
    atomic_write_json(report_path, report)
    return report_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--qblade-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--raw-reuse-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--srun", default="srun")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--allocated-cpus", type=int, default=ALLOCATED_CPUS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report_path = run_scheduler(
            campaign_root=args.campaign,
            qblade_root=args.qblade_root,
            calibration=args.calibration,
            raw_reuse_root=args.raw_reuse_root,
            python_executable=args.python,
            srun_executable=args.srun,
            workers=args.workers,
            threads=args.threads,
            allocated_cpus=args.allocated_cpus,
            dry_run=args.dry_run,
        )
    except (
        campaign.CampaignError,
        legacy.SchedulerError,
        EngineeringSchedulerError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": report["status"],
                "launch_count": len(report["cases"]),
                "execution_contract": report["execution_contract"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
