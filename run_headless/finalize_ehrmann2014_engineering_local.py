#!/usr/bin/env python3
"""Publish provenance for a completed local Ehrmann engineering campaign.

The production Flight scheduler writes its own execution report.  This helper
provides the equivalent, explicitly local contract for a campaign executed in
the QBlade Docker image.  It never launches or modifies a simulation; it only
accepts a complete, replay-valid campaign and an exact ``docker inspect``
snapshot proving the requested CPU cap.
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
from run_headless import run_ehrmann2014_engineering_flight as engineering
from vorlap.qblade_run_support import atomic_write_json

SCHEMA = "vorlap-ehrmann2014-engineering-local-docker-execution-v2"
EXPECTED_NANO_CPUS = 8_000_000_000


class LocalExecutionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_inspect(path: Path) -> Mapping[str, Any]:
    inspect_path = Path(path).expanduser().resolve()
    if inspect_path.is_symlink() or not inspect_path.is_file():
        raise LocalExecutionError("Docker inspect evidence must be a regular file.")
    try:
        payload = json.loads(inspect_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalExecutionError("Docker inspect evidence is not valid JSON.") from exc
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], Mapping)
    ):
        raise LocalExecutionError("Docker inspect evidence must contain one container.")
    return payload[0]


def _container_contract(path: Path) -> dict[str, Any]:
    value = _read_inspect(path)
    host = value.get("HostConfig")
    config = value.get("Config")
    state = value.get("State")
    if not all(isinstance(item, Mapping) for item in (host, config, state)):
        raise LocalExecutionError("Docker inspect evidence lacks runtime sections.")
    if host.get("NanoCpus") != EXPECTED_NANO_CPUS:
        raise LocalExecutionError("Docker execution must be capped at exactly 8 CPUs.")
    if value.get("Platform") != "linux":
        raise LocalExecutionError(
            "QBlade Docker execution must use the Linux platform."
        )
    image_id = value.get("Image")
    image_ref = config.get("Image")
    if (
        not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or len(image_id) != 71
        or not isinstance(image_ref, str)
        or not image_ref
    ):
        raise LocalExecutionError("Docker image identity is malformed.")
    if state.get("OOMKilled") is not False:
        raise LocalExecutionError("Docker execution was OOM-killed or is ambiguous.")
    inspect_path = Path(path).expanduser().resolve()
    return {
        "platform": "docker_linux_amd64",
        "image_reference": image_ref,
        "image_id": image_id,
        "nano_cpus": EXPECTED_NANO_CPUS,
        "oom_killed": False,
        "inspect_file": {
            "path": str(inspect_path),
            "sha256": _sha256(inspect_path),
        },
    }


def build_report(
    *,
    campaign_root: Path,
    calibration: Path,
    raw_reuse_root: Path,
    qblade_root: Path,
    docker_inspect: Path,
    output: Path | None = None,
) -> Path:
    state = engineering.validate_execution_plan(
        campaign_root,
        calibration=calibration,
        raw_reuse_root=raw_reuse_root,
        qblade_root=qblade_root,
    )
    plan = state["plan"]
    if plan["blocked"] or plan["runnable"] or len(plan["complete"]) != 52:
        raise LocalExecutionError(
            "Local execution report requires exactly 52 replay-valid complete cases."
        )
    root = Path(state["root"])
    manifest = state["manifest"]
    container = _container_contract(docker_inspect)
    inspect_path = Path(docker_inspect).expanduser().resolve()
    try:
        container["inspect_file"]["path"] = inspect_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise LocalExecutionError(
            "Docker inspect evidence must be stored under the campaign root."
        ) from exc
    case_results: list[dict[str, str]] = []
    for row in manifest["cases"]:
        case_id = str(row["case_id"])
        if row.get("raw_reuse_status") == "reused_exact_identity":
            artifact = root / str(row["result_dir"]) / "reuse_receipt.json"
        else:
            artifact = (
                root
                / str(row["result_dir"])
                / f"{Path(str(row['sim'])).stem}_run_summary.json"
            )
        if artifact.is_symlink() or not artifact.is_file():
            raise LocalExecutionError(f"Missing terminal artifact for {case_id}.")
        case_results.append(
            {
                "case_id": case_id,
                "artifact": artifact.relative_to(root).as_posix(),
                "sha256": _sha256(artifact),
            }
        )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "campaign_fingerprint": manifest["fingerprint"],
        "execution_contract": {
            "schema": SCHEMA,
            "campaign_schema": campaign.ENGINEERING_LOCKIN_SCHEMA,
            "calibration_sha256": manifest["engineering_lockin_identity"]["sha256"],
            "raw_reuse_root": str(state["reuse_root"]),
            "exact_raw_reuse_case_count": 18,
            "new_raw_case_count": 8,
            "hybrid_case_count": 26,
            "complete_at_end": 52,
            "nalu_required": False,
            "threads_per_case": 1,
            "container": container,
        },
        "case_results": case_results,
    }
    destination = (
        root / "local_docker_execution_report.json"
        if output is None
        else Path(output).expanduser().resolve()
    )
    atomic_write_json(destination, report)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--raw-reuse-root", type=Path, required=True)
    parser.add_argument("--qblade-root", type=Path, required=True)
    parser.add_argument("--docker-inspect", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        path = build_report(
            campaign_root=args.campaign,
            calibration=args.calibration,
            raw_reuse_root=args.raw_reuse_root,
            qblade_root=args.qblade_root,
            docker_inspect=args.docker_inspect,
            output=args.output,
        )
    except (
        campaign.CampaignError,
        engineering.EngineeringSchedulerError,
        LocalExecutionError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
