#!/usr/bin/env python3
"""Recover completed DLC exports whose checkout changed during execution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
VORLAP_ROOT = SCRIPT_DIR.parent
if str(VORLAP_ROOT) not in sys.path:
    sys.path.insert(0, str(VORLAP_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_parked_dlc_manifest_row import row_environment  # noqa: E402
from vorlap.dlc_manifest import (  # noqa: E402
    DLCManifestError,
    load_manifest,
    selected_row,
    sha256_file,
    validate_wind_record,
    write_json,
)


RECOVERABLE_ERROR_CODE = "checkpoint_runtime_identity_mismatch"


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DLCManifestError(f"Required recovery artifact is missing: {path}") from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise DLCManifestError(f"Could not read recovery artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DLCManifestError(f"Recovery artifact must contain a JSON object: {path}")
    return payload


def results_audit_path(row: Mapping[str, Any]) -> Path:
    return Path(str(row["run_root"])) / "audits" / "results_audit.json"


def recoverable_audit_errors(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = results_audit_path(row)
    audit = read_json(path)
    if audit.get("passed") is True:
        return []
    issues = audit.get("issues")
    if not isinstance(issues, list):
        raise DLCManifestError(f"Failed results audit has no issue list: {path}")
    errors = [
        issue
        for issue in issues
        if isinstance(issue, dict) and str(issue.get("severity", "")) == "error"
    ]
    if not errors:
        raise DLCManifestError(f"Failed results audit has no explicit errors: {path}")
    unexpected = [
        issue
        for issue in errors
        if issue.get("code") != RECOVERABLE_ERROR_CODE
        or issue.get("scenario") != "vorlap"
    ]
    if unexpected:
        codes = ", ".join(
            f"{issue.get('scenario', '?')}:{issue.get('code', '?')}"
            for issue in unexpected
        )
        raise DLCManifestError(
            f"Row {row['row_id']} has non-recoverable results-audit errors: {codes}"
        )
    return errors


def recoverable_indices(manifest: Mapping[str, Any]) -> list[int]:
    indices: list[int] = []
    for index, row in enumerate(manifest["rows"]):
        if recoverable_audit_errors(row):
            indices.append(index)
    return indices


def slurm_array_spec(indices: list[int]) -> str:
    if not indices:
        return ""
    ordered = sorted(set(indices))
    ranges: list[str] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _case_for_recovery(row: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    scenario_root = Path(str(row["run_root"])) / "vorlap"
    manifest = read_json(scenario_root / "cases.json")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise DLCManifestError(
            f"Provenance recovery requires exactly one archived VorLap case: {scenario_root}"
        )
    return scenario_root, cases[0]


def _summary_path(case: Mapping[str, Any]) -> Path:
    sim = Path(str(case["sim"]))
    return Path(str(case["result_dir"])) / f"{sim.stem}_run_summary.json"


def _preserve_original_artifacts(
    *,
    row: Mapping[str, Any],
    case: Mapping[str, Any],
    audit_path: Path,
) -> dict[Path, Path]:
    result_dir = Path(str(case["result_dir"]))
    sim = Path(str(case["sim"]))
    sources = (
        audit_path,
        _summary_path(case),
        result_dir / f"{case['result_name']}.txt",
        result_dir / "case_metadata.json",
        result_dir / f"{sim.stem}_qblade.log",
    )
    backup_dir = (
        Path(str(row["run_root"]))
        / "audits"
        / "provenance_recovery_original"
    )
    backup_dir.mkdir(parents=True, exist_ok=True)
    preserved: dict[Path, Path] = {}
    for source in sources:
        if not source.is_file():
            continue
        target = backup_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
        preserved[source] = target
    return preserved


def _restore_original_artifacts(preserved: Mapping[Path, Path]) -> None:
    for destination, backup in preserved.items():
        if backup.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, destination)


def _validate_export_only_summary(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    restart = summary.get("restart")
    if not isinstance(restart, dict) or not bool(
        restart.get("completed_checkpoint_export_only", False)
    ):
        raise DLCManifestError(
            f"Checkpoint recovery did not record its export-only identity policy: {summary_path}"
        )
    requested = int(summary.get("requested_steps", 0) or 0)
    completed = int(summary.get("completed_steps", 0) or 0)
    resume_start = int(restart.get("resume_start_step", -1) or 0)
    if requested <= 0 or completed != requested or resume_start != requested:
        raise DLCManifestError(
            "Checkpoint recovery must load a completed project and advance zero timesteps; "
            f"observed resume/completed/requested={resume_start}/{completed}/{requested}."
        )
    if bool(summary.get("failed", False)) or not bool(summary.get("exported_results", False)):
        raise DLCManifestError(f"Checkpoint recovery did not produce a successful export: {summary_path}")
    return summary


def _recovery_command(
    *,
    scenario_root: Path,
    qblade_root: Path,
    vorlap_root: Path,
    python_exe: Path,
    result_filter: Path,
    checkpoint_every: int,
    threads: int,
) -> list[str]:
    return [
        str(python_exe),
        str(vorlap_root / "run_headless" / "vorlap_qblade_batch.py"),
        "recover-export",
        "--run-root",
        str(scenario_root),
        "--qblade-root",
        str(qblade_root),
        "--vorlap-root",
        str(vorlap_root),
        "--python",
        str(python_exe),
        "--result-filter",
        str(result_filter),
        "--case-index",
        "0",
        "--threads",
        str(threads),
        "--progress-every",
        "10",
        "--checkpoint-every",
        str(checkpoint_every),
        "--skip-vorlap-install",
        "--skip-bridge-build",
        "--coupling",
        "sil-vorlap",
    ]


def recover_row(args: argparse.Namespace) -> dict[str, Any]:
    vorlap_root = args.vorlap_root.expanduser().resolve()
    python_exe = args.python_exe.expanduser().resolve()
    parent_submit = args.parent_submit.expanduser().resolve()
    qblade_root = args.qblade_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for required in (python_exe, parent_submit):
        if not required.is_file():
            raise FileNotFoundError(f"Required recovery executable/script is missing: {required}")

    manifest = load_manifest(manifest_path, require_approved=True)
    row = selected_row(manifest, args.index)
    errors = recoverable_audit_errors(row)
    if not errors:
        return {
            "index": args.index,
            "row_id": row["row_id"],
            "status": "already-passed",
        }
    validate_wind_record(row, manifest)
    execution = manifest["execution_settings"]
    result_filter = Path(str(execution["result_filter_path"])).expanduser().resolve()
    if not result_filter.is_file() or sha256_file(result_filter) != execution["result_filter_sha256"]:
        raise DLCManifestError(
            f"Approved result filter is missing or changed: {result_filter}"
        )

    scenario_root, case = _case_for_recovery(row)
    summary_path = _summary_path(case)
    original_audit_path = results_audit_path(row)
    original_audit_sha256 = sha256_file(original_audit_path)
    preserved = _preserve_original_artifacts(
        row=row,
        case=case,
        audit_path=original_audit_path,
    )
    checkpoint_every = int(execution["checkpoint_every"])
    if checkpoint_every <= 0:
        raise DLCManifestError("Completed-checkpoint recovery requires checkpointing to be enabled.")

    recovery_command = _recovery_command(
        scenario_root=scenario_root,
        qblade_root=qblade_root,
        vorlap_root=vorlap_root,
        python_exe=python_exe,
        result_filter=result_filter,
        checkpoint_every=checkpoint_every,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "action": "completed-checkpoint-export-only",
                "index": args.index,
                "row_id": row["row_id"],
                "scenario_root": str(scenario_root),
                "threads": args.threads,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        subprocess.run(recovery_command, check=True)
        recovered_summary = _validate_export_only_summary(summary_path)
    except (DLCManifestError, OSError, subprocess.CalledProcessError):
        _restore_original_artifacts(preserved)
        raise

    postprocess_env = row_environment(
        manifest,
        row,
        phase="qblade",
        parent_submit=parent_submit,
        parent_dry_run=False,
    )
    postprocess_env.update(
        {
            "ONLY_POSTPROCESS": "1",
            "PRESERVE_COMPLETED_SCENARIOS": "1",
            "VORLAP_ROOT": str(vorlap_root),
            "PYTHON_EXE": str(python_exe),
            "QBLADE_ROOT": str(qblade_root),
        }
    )
    subprocess.run(["bash", str(parent_submit)], env=postprocess_env, check=True)

    final_audit = read_json(original_audit_path)
    if final_audit.get("passed") is not True:
        raise DLCManifestError(
            f"Provenance recovery completed but the final results audit still failed: {original_audit_path}"
        )
    recovery_record = {
        "schema_version": 1,
        "created_unix": time.time(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "row_index": args.index,
        "row_id": row["row_id"],
        "recovered_error_codes": sorted({str(issue["code"]) for issue in errors}),
        "original_results_audit_sha256": original_audit_sha256,
        "preserved_original_artifacts": {
            str(source): {
                "backup": str(backup),
                "sha256": sha256_file(backup),
            }
            for source, backup in sorted(preserved.items(), key=lambda item: str(item[0]))
        },
        "policy": recovered_summary["restart"]["identity_policy"],
        "resume_start_step": recovered_summary["restart"]["resume_start_step"],
        "completed_steps": recovered_summary["completed_steps"],
        "requested_steps": recovered_summary["requested_steps"],
        "advanced_timesteps": 0,
        "run_summary": str(summary_path),
        "run_summary_sha256": sha256_file(summary_path),
        "final_results_audit": str(original_audit_path),
        "final_results_audit_sha256": sha256_file(original_audit_path),
        "final_results_audit_passed": True,
    }
    record_path = Path(str(row["run_root"])) / "audits" / "provenance_recovery.json"
    write_json(record_path, recovery_record)
    return {
        "index": args.index,
        "row_id": row["row_id"],
        "status": "recovered",
        "recovery_record": str(record_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="List rows failed only by recoverable provenance drift.")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--format", choices=("json", "array"), default="json")

    run = subparsers.add_parser("run", help="Recover and re-audit one manifest row.")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--vorlap-root", type=Path, default=VORLAP_ROOT)
    run.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    run.add_argument(
        "--qblade-root",
        type=Path,
        default=VORLAP_ROOT / "run_headless" / "QBladeCE_2.0.9.7",
    )
    run.add_argument(
        "--parent-submit",
        type=Path,
        default=SCRIPT_DIR / "submit_parked_demo_campaign.sj",
    )
    run.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()
    if getattr(args, "threads", 1) <= 0:
        parser.error("--threads must be positive.")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.command == "plan":
            manifest = load_manifest(args.manifest.expanduser().resolve(), require_approved=True)
            indices = recoverable_indices(manifest)
            if args.format == "array":
                print(slurm_array_spec(indices))
            else:
                print(
                    json.dumps(
                        {
                            "array_spec": slurm_array_spec(indices),
                            "recoverable_count": len(indices),
                            "recoverable_indices": indices,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            return 0
        print(json.dumps(recover_row(args), indent=2, sort_keys=True))
        return 0
    except (DLCManifestError, FileNotFoundError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
