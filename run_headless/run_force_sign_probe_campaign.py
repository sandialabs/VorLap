#!/usr/bin/env python3
"""Run and summarize SIL constant-force sign probes for a parked campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import vorlap_qblade_batch as batch


AXES = ("X", "Y", "Z")
SIGNS = ("positive", "negative")
REPORT_COLUMNS = (
    "scenario",
    "case",
    "node",
    "axis",
    "sign",
    "force_n",
    "local",
    "output_dir",
    "summary_file",
    "result_file",
    "requested_steps",
    "completed_steps",
    "failed",
    "exported_results",
    "observation_status",
    "observed_position_channel",
    "delta_position_m",
    "paired_response_delta_m",
    "paired_passed",
    "paired_issue",
    "passed",
    "issue",
)

PAIRED_RESPONSE_TOLERANCE_M = 1.0e-9


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_headless_root() -> Path:
    return Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_axes(spec: str) -> list[str]:
    axes = [part.upper() for part in re.split(r"[\s,]+", str(spec).strip()) if part]
    if not axes:
        raise ValueError("At least one sign-probe axis is required.")
    bad = [axis for axis in axes if axis not in AXES]
    if bad:
        raise ValueError(f"Unsupported sign-probe axis/axes: {', '.join(bad)}.")
    return axes


def safe_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text).strip())
    return label.strip("_") or "probe"


def geometry_audit_path(case: dict[str, Any]) -> Path | None:
    sim_path = Path(str(case.get("sim", "")))
    for pattern in ("*/Control/vorlap_qblade_geometry_audit.csv", "**/vorlap_qblade_geometry_audit.csv"):
        matches = sorted(sim_path.parent.glob(pattern))
        if matches:
            return matches[0]
    return None


def geometry_nodes(case: dict[str, Any]) -> list[str]:
    path = geometry_audit_path(case)
    if path is None or not path.is_file():
        return []
    nodes: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            node = str(row.get("requested_node_id", "")).strip()
            if node and node not in nodes:
                nodes.append(node)
    return nodes


def default_probe_node(case: dict[str, Any]) -> str:
    nodes = geometry_nodes(case)
    for node in nodes:
        if node.startswith("BLD_1_"):
            return node
    for node in nodes:
        if node.startswith("BLD_"):
            return node
    if nodes:
        return nodes[0]
    raise ValueError(
        f"Could not choose a default force-sign probe node for case {case.get('label', '?')!r}; "
        "pass --node explicitly or run setup so the geometry audit exists."
    )


def load_case(run_root: Path, scenario: str, *, case_label: str | None, case_index: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = run_root / scenario / "cases.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing generated case manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    cases = list(manifest.get("cases", []))
    if not cases:
        raise ValueError(f"Case manifest has no cases: {manifest_path}")
    if case_label is not None:
        for case in cases:
            if str(case.get("label", "")) == case_label:
                return manifest, case
        raise ValueError(f"Case label {case_label!r} was not found in {manifest_path}.")
    index = 0 if case_index is None else int(case_index)
    if index < 0 or index >= len(cases):
        raise IndexError(f"Case index {index} is outside 0..{len(cases) - 1}.")
    return manifest, cases[index]


def force_value(magnitude: float, sign: str) -> float:
    if sign == "positive":
        return magnitude
    if sign == "negative":
        return -magnitude
    raise ValueError(f"Unsupported sign {sign!r}.")


def build_probe_plan(args: argparse.Namespace, manifest: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    axes = parse_axes(args.axes)
    nodes = list(args.node) if args.node else [default_probe_node(case)]
    if args.force <= 0.0 or not math.isfinite(args.force):
        raise ValueError("--force must be a positive finite force magnitude in N.")
    probes: list[dict[str, Any]] = []
    for node in nodes:
        for axis in axes:
            for sign in SIGNS:
                value = force_value(float(args.force), sign)
                local_suffix = ":local" if args.local else ":global"
                force_spec = f"{node}:{axis}:{value:.9g}{local_suffix}"
                label = safe_label(f"{case['label']}_{node}_{axis}_{sign}")
                output_dir = args.output_root / label
                result_name = f"{label}_qblade_results"
                metadata_file = output_dir / "case_metadata.json"
                summary_file = output_dir / f"{Path(case['sim']).stem}_run_summary.json"
                result_file = output_dir / f"{result_name}.txt"
                probes.append(
                    {
                        "scenario": args.scenario,
                        "case": case["label"],
                        "node": node,
                        "axis": axis,
                        "sign": sign,
                        "force_n": value,
                        "local": bool(args.local),
                        "force_spec": force_spec,
                        "sim": case["sim"],
                        "output_dir": str(output_dir),
                        "result_name": result_name,
                        "metadata_file": str(metadata_file),
                        "summary_file": str(summary_file),
                        "result_file": str(result_file),
                    }
                )
    return {
        "created_unix": time.time(),
        "run_root": str(args.run_root),
        "scenario": args.scenario,
        "case": case,
        "manifest_settings": manifest.get("settings", {}),
        "qblade_root": str(args.qblade_root),
        "vorlap_root": str(args.vorlap_root),
        "python": args.python,
        "steps": args.steps,
        "force_magnitude_n": args.force,
        "axes": axes,
        "nodes": nodes,
        "local": bool(args.local),
        "probes": probes,
    }


def command_for_probe(args: argparse.Namespace, probe: dict[str, Any]) -> list[str]:
    command = [
        args.python,
        str(run_headless_root() / "run_qblade_sim.py"),
        "--qblade-root",
        str(args.qblade_root),
        "--sim",
        str(probe["sim"]),
        "--output-dir",
        str(probe["output_dir"]),
        "--result-name",
        str(probe["result_name"]),
        "--steps",
        str(args.steps),
        "--threads",
        str(args.threads),
        "--coupling-mode",
        "sil-constant-force",
        "--sil-constant-force",
        str(probe["force_spec"]),
        "--case-metadata",
        str(probe["metadata_file"]),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.allow_partial:
        command.append("--allow-partial")
    return command


def write_probe_metadata(plan: dict[str, Any], probe: dict[str, Any]) -> None:
    metadata = {
        "case": plan["case"],
        "manifest_settings": plan.get("manifest_settings", {}),
        "manifest_created_unix": None,
        "force_sign_probe": {
            key: probe[key]
            for key in ("scenario", "case", "node", "axis", "sign", "force_n", "local", "force_spec")
        },
    }
    write_json(Path(str(probe["metadata_file"])), metadata)


def run_checked(command: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def probe_summary_path(probe: dict[str, Any]) -> Path:
    return Path(str(probe["summary_file"]))


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def summarize_probe(probe: dict[str, Any]) -> dict[str, Any]:
    row = {column: "" for column in REPORT_COLUMNS}
    row.update(
        {
            "scenario": probe["scenario"],
            "case": probe["case"],
            "node": probe["node"],
            "axis": probe["axis"],
            "sign": probe["sign"],
            "force_n": probe["force_n"],
            "local": _bool_text(probe["local"]),
            "output_dir": probe["output_dir"],
            "summary_file": probe["summary_file"],
            "result_file": probe["result_file"],
            "passed": "false",
        }
    )
    summary_path = probe_summary_path(probe)
    if not summary_path.is_file():
        row["issue"] = "missing_run_summary"
        return row
    summary = read_json(summary_path)
    row["requested_steps"] = int(summary.get("requested_steps", 0) or 0)
    row["completed_steps"] = int(summary.get("completed_steps", 0) or 0)
    row["failed"] = _bool_text(summary.get("failed", False))
    row["exported_results"] = _bool_text(summary.get("exported_results", False))
    row["result_file"] = str(summary.get("result_file") or probe["result_file"])

    observations = summary.get("force_sign_probe_observations")
    if not isinstance(observations, list) or not observations:
        row["issue"] = "missing_force_sign_observation"
        return row
    observation = observations[0]
    row["observation_status"] = str(observation.get("status", ""))
    row["observed_position_channel"] = str(observation.get("observed_position_channel", ""))
    row["delta_position_m"] = observation.get("delta_position_m", "")
    if summary.get("coupling_mode") != "sil-constant-force":
        row["issue"] = "wrong_coupling_mode"
    elif bool(summary.get("failed", False)):
        row["issue"] = "failed_run"
    elif not bool(summary.get("exported_results", False)):
        row["issue"] = "results_not_exported"
    elif int(summary.get("completed_steps", 0) or 0) < 1:
        row["issue"] = "no_completed_steps"
    elif not row["observed_position_channel"]:
        row["issue"] = "missing_position_channel"
    elif row["delta_position_m"] == "":
        row["issue"] = "missing_delta_position"
    elif row["observation_status"] in {"missing_position_channel", "insufficient_samples", "zero_or_unresolved_response"}:
        row["issue"] = f"observation_{row['observation_status'] or 'missing'}"
    else:
        row["passed"] = "true"
    return row


def _append_issue(row: dict[str, Any], issue: str) -> None:
    existing = str(row.get("issue", "")).strip()
    row["issue"] = issue if not existing else f"{existing};{issue}"


def _row_float(row: dict[str, Any], name: str) -> float:
    try:
        return float(row.get(name, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def apply_paired_sign_checks(rows: list[dict[str, Any]]) -> None:
    """Validate sign evidence from paired +F/-F response, not absolute drift."""
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("scenario", "")).strip(),
            str(row.get("case", "")).strip(),
            str(row.get("node", "")).strip(),
            str(row.get("axis", "")).strip().upper(),
            str(row.get("local", "")).strip().lower(),
        )
        sign = str(row.get("sign", "")).strip().lower()
        grouped.setdefault(key, {})[sign] = row

    for sign_rows in grouped.values():
        positive = sign_rows.get("positive")
        negative = sign_rows.get("negative")
        if positive is None or negative is None:
            for row in sign_rows.values():
                row["paired_passed"] = "false"
                row["paired_issue"] = "missing_positive_or_negative_pair"
                row["passed"] = "false"
                _append_issue(row, row["paired_issue"])
            continue

        pos_force = _row_float(positive, "force_n")
        neg_force = _row_float(negative, "force_n")
        pos_delta = _row_float(positive, "delta_position_m")
        neg_delta = _row_float(negative, "delta_position_m")
        response_delta = pos_delta - neg_delta
        force_delta = pos_force - neg_force
        paired_ok = (
            math.isfinite(response_delta)
            and math.isfinite(force_delta)
            and force_delta > 0.0
            and response_delta > PAIRED_RESPONSE_TOLERANCE_M
        )
        issue = "" if paired_ok else "paired_response_not_positive"
        for row in (positive, negative):
            row["paired_response_delta_m"] = response_delta
            row["paired_passed"] = _bool_text(paired_ok)
            row["paired_issue"] = issue
            if not paired_ok:
                row["passed"] = "false"
                _append_issue(row, issue)


def write_report(plan: dict[str, Any], output_root: Path) -> dict[str, Any]:
    rows = [summarize_probe(probe) for probe in plan["probes"]]
    apply_paired_sign_checks(rows)
    csv_path = output_root / "force_sign_probe_report.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if str(row.get("passed", "")).lower() != "true"]
    report = {
        "passed": not failures,
        "report_csv": str(csv_path),
        "rows": len(rows),
        "failures": len(failures),
        "probes": rows,
    }
    write_json(output_root / "force_sign_probe_report.json", report)
    return report


def resolve_qblade_root(args: argparse.Namespace, settings: dict[str, Any]) -> Path:
    if args.qblade_root is not None:
        return args.qblade_root.resolve()
    if settings.get("qblade_root"):
        return Path(str(settings["qblade_root"])).resolve()
    return batch.default_qblade_root(repo_root()).resolve()


def resolve_vorlap_root(args: argparse.Namespace, settings: dict[str, Any]) -> Path:
    if args.vorlap_root is not None:
        return args.vorlap_root.resolve()
    if settings.get("vorlap_root"):
        return Path(str(settings["vorlap_root"])).resolve()
    return Path(os.environ.get("VORLAP_ROOT", repo_root())).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=run_headless_root() / "runs" / "vorlap_parked_demo")
    parser.add_argument("--scenario", default="identity", help="Generated scenario manifest to probe. Defaults to the zero-force identity case.")
    parser.add_argument("--case-label", default=None, help="Case label to probe. Defaults to --case-index.")
    parser.add_argument("--case-index", type=int, default=0, help="Case index to probe when --case-label is omitted.")
    parser.add_argument("--node", action="append", default=[], help="QBlade node id to probe. Repeat for multiple nodes. Defaults to the first BLD_1 row in the geometry audit.")
    parser.add_argument("--axes", default="X Y Z", help="Axes to probe, e.g. 'X Y Z'.")
    parser.add_argument("--force", type=float, default=1000.0, help="Positive force magnitude in N. Both positive and negative signs are run.")
    parser.add_argument("--local", action="store_true", help="Probe local QBlade axes instead of global axes.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "4")))
    parser.add_argument("--qblade-root", type=Path, default=None)
    parser.add_argument("--vorlap-root", type=Path, default=None)
    parser.add_argument("--python", default=os.environ.get("PYTHON_EXE", batch.default_python(repo_root())))
    parser.add_argument("--output-root", type=Path, default=None, help="Defaults to RUN_ROOT/sign_probes.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write and print the probe plan without launching QBlade.")
    parser.add_argument("--only-report", action="store_true", help="Do not run probes; rebuild the report from existing summaries.")
    parser.add_argument("--no-fail", action="store_true", help="Exit 0 even if the summarized probe evidence fails.")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    campaign_settings_path = args.run_root / "campaign_settings.json"
    settings = read_json(campaign_settings_path) if campaign_settings_path.is_file() else {}
    args.qblade_root = resolve_qblade_root(args, settings)
    args.vorlap_root = resolve_vorlap_root(args, settings)
    args.output_root = (args.output_root or (args.run_root / "sign_probes")).resolve()
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    return args


def main() -> None:
    args = parse_args()
    manifest, case = load_case(args.run_root, args.scenario, case_label=args.case_label, case_index=args.case_index)
    plan = build_probe_plan(args, manifest, case)
    args.output_root.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_root / "force_sign_probe_plan.json"
    write_json(plan_path, plan)

    if args.dry_run:
        dry_plan = dict(plan)
        dry_plan["commands"] = [command_for_probe(args, probe) for probe in plan["probes"]]
        print(json.dumps(dry_plan, indent=2, sort_keys=True))
        return

    env_args = argparse.Namespace(qblade_root=args.qblade_root, vorlap_root=args.vorlap_root, threads=args.threads)
    env = batch.qblade_env(env_args)
    if not args.only_report:
        for probe in plan["probes"]:
            Path(str(probe["output_dir"])).mkdir(parents=True, exist_ok=True)
            write_probe_metadata(plan, probe)
            run_checked(command_for_probe(args, probe), env)

    report = write_report(plan, args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"] and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
