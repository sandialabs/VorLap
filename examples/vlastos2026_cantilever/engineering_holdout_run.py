#!/usr/bin/env python3
"""Run and assess the frozen eight-case schema-v2 engineering holdout.

The full immutable campaign retains 48 held-out cases for diagnostic curve
work.  This wrapper executes only the eight peak-neighborhood cases fixed in
``engineering_holdout.py``.  It deliberately reuses the schema-v2 campaign
loader, runtime identities, fresh-output checks, and post-dispatch audits from
``response_benchmark.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # Package import in tests.
    from examples.vlastos2026_cantilever import engineering_holdout
    from examples.vlastos2026_cantilever import response_benchmark
except ImportError:  # Direct execution from this example directory.
    import engineering_holdout  # type: ignore[no-redef]
    import response_benchmark  # type: ignore[no-redef]


PLAN_SCHEMA = "vlastos2026-schema-v2-engineering-holdout-run-plan-v1"
EXECUTION_INTENT = "diagnostic_only_after_frozen_hgast_gate_failure"

_FLOAT_COLUMNS = {
    "scruton_number",
    "reduced_velocity",
    "tip_peak_amplitude_over_diameter",
    "terminal_dominant_frequency_over_fn",
}
_BOOLEAN_COLUMNS = {
    "runtime_audit_evaluated",
    "runtime_audit_pass",
    "stationarity_evaluated",
    "stationarity_pass",
}


def _json_file(path: Path) -> Any:
    source = Path(path).expanduser().resolve()
    return json.loads(source.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _file_identity(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "sha256": response_benchmark.sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _load_response_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in _FLOAT_COLUMNS:
                if key in row and row[key] not in (None, ""):
                    row[key] = float(row[key])
            for key in _BOOLEAN_COLUMNS:
                if key in row:
                    if row[key] not in {"True", "False"}:
                        raise ValueError(
                            f"Analysis CSV {key} must be True or False; observed {row[key]!r}."
                        )
                    row[key] = row[key] == "True"
            rows.append(row)
    return rows


def _calibration_payloads(
    analysis_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(analysis_dir).expanduser().resolve()
    summary = _json_file(root / "acceptance_summary.json")
    rows = _load_response_rows(root / "response_by_case.csv")
    audits = _json_file(root / "runtime_case_audits.json")
    if not isinstance(summary, dict) or not isinstance(audits, list):
        raise ValueError("Calibration analysis artifacts have invalid JSON types.")
    return summary, rows, audits


def build_calibration_readiness(analysis_dir: Path) -> dict[str, Any]:
    summary, rows, audits = _calibration_payloads(analysis_dir)
    evidence = {
        "campaign_manifest_sha256": engineering_holdout.CAMPAIGN_MANIFEST_SHA256,
        "stationary_protocol_sha256": engineering_holdout.STATIONARY_PROTOCOL_SHA256,
        "coefficient_fingerprint_sha256": (
            engineering_holdout.COEFFICIENT_FINGERPRINT_SHA256
        ),
        "acceptance_summary_content_sha256": engineering_holdout.semantic_sha256(
            summary
        ),
        "response_rows_content_sha256": engineering_holdout.semantic_sha256(rows),
        "runtime_audits_content_sha256": engineering_holdout.semantic_sha256(audits),
    }
    return engineering_holdout.assess_calibration_readiness(
        summary,
        rows,
        audits,
        evidence,
    )


def _load_passing_readiness(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    value = _json_file(source)
    if not isinstance(value, dict):
        raise ValueError("Calibration readiness must be a JSON object.")
    source_evidence = value.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise ValueError("Calibration readiness lacks source evidence.")
    engineering_holdout._validate_calibration_readiness_for_heldout(
        value,
        source_evidence,
    )
    return value, _file_identity(source)


def _full_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    return engineering_holdout.semantic_sha256(dict(plan))


def build_exact_eight_plan(
    manifest_path: Path,
    calibration_readiness_path: Path,
    results_root: Path,
    *,
    qblade_root: Path,
    checkpoint_every: int = 0,
    threads: int = 4,
) -> dict[str, Any]:
    readiness, readiness_identity = _load_passing_readiness(
        calibration_readiness_path
    )
    manifest = Path(manifest_path).expanduser().resolve()
    full_plan = response_benchmark.build_run_plan(
        manifest,
        Path(results_root),
        stage="held_out",
        qblade_root=Path(qblade_root),
        checkpoint_every=checkpoint_every,
        threads=threads,
        diagnostic_only=True,
    )
    expected_ids = engineering_holdout.engineering_screen_spec()["exact_case_ids"]
    command_by_id = {row["case_id"]: row for row in full_plan["commands"]}
    if set(expected_ids).difference(command_by_id):
        raise ValueError("The immutable campaign lacks a selected held-out command.")
    selected_commands = [command_by_id[case_id] for case_id in expected_ids]
    plan = {
        "schema": PLAN_SCHEMA,
        "execution_intent": EXECUTION_INTENT,
        "campaign_manifest": _file_identity(manifest),
        "calibration_readiness": readiness_identity,
        "calibration_readiness_fingerprint_sha256": readiness[
            "assessment_fingerprint_sha256"
        ],
        "screen_specification_fingerprint_sha256": (
            engineering_holdout.engineering_screen_spec()[
                "specification_fingerprint_sha256"
            ]
        ),
        "full_heldout_plan_fingerprint_sha256": _full_plan_fingerprint(full_plan),
        "results_root": full_plan["results_root"],
        "qblade_root": full_plan["qblade_root"],
        "checkpoint_every": full_plan["checkpoint_every"],
        "threads": full_plan["threads"],
        "exact_case_ids": expected_ids,
        "commands": selected_commands,
        "strict_hgast_gate_status": full_plan[
            "postfreeze_analytical_gate_status"
        ],
        "strict_hgast_gate_rewritten": False,
    }
    plan["plan_fingerprint_sha256"] = engineering_holdout.semantic_sha256(plan)
    return plan


def _canonical_plan_from_supplied(
    supplied: Mapping[str, Any],
    manifest_path: Path,
    readiness_path: Path,
) -> dict[str, Any]:
    return build_exact_eight_plan(
        manifest_path,
        readiness_path,
        Path(str(supplied.get("results_root", ""))),
        qblade_root=Path(str(supplied.get("qblade_root", ""))),
        checkpoint_every=supplied.get("checkpoint_every"),
        threads=supplied.get("threads"),
    )


def execute_exact_eight_plan(
    plan: Mapping[str, Any],
    manifest_path: Path,
    calibration_readiness_path: Path,
    *,
    acknowledge_failed_strict_hgast_gate: bool,
) -> None:
    if acknowledge_failed_strict_hgast_gate is not True:
        raise RuntimeError(
            "Exact-eight execution requires explicit acknowledgement that the "
            "strict hGAST gate failed and this is an engineering diagnostic."
        )
    if not isinstance(plan, dict):
        raise TypeError("Engineering holdout plan must be a JSON object.")
    expected = _canonical_plan_from_supplied(
        plan,
        Path(manifest_path).expanduser().resolve(),
        Path(calibration_readiness_path).expanduser().resolve(),
    )
    if plan != expected:
        raise RuntimeError("Engineering holdout plan differs from the canonical plan.")

    manifest = Path(manifest_path).expanduser().resolve()
    campaign_root, campaign = response_benchmark._load_campaign(manifest)
    full_plan = response_benchmark.build_run_plan(
        manifest,
        Path(plan["results_root"]),
        stage="held_out",
        qblade_root=Path(plan["qblade_root"]),
        checkpoint_every=plan["checkpoint_every"],
        threads=plan["threads"],
        diagnostic_only=True,
    )
    response_benchmark._validate_ambient_v2_run_plan_shape(full_plan, campaign)
    selected_commands = list(plan["commands"])
    response_benchmark._assert_v2_case_outputs_fresh(full_plan, selected_commands)
    baseline_tree = response_benchmark._campaign_tree_snapshot(campaign_root)
    entries = {entry["case_id"]: entry for entry in campaign["cases"]}
    first = entries[selected_commands[0]["case_id"]]
    response_benchmark._preflight_response_runtime(
        campaign_root / first["parameter_file"]
    )

    for selected_index, row in enumerate(selected_commands):
        current = _canonical_plan_from_supplied(plan, manifest, calibration_readiness_path)
        if current != plan or current["commands"][selected_index] != row:
            raise RuntimeError("Engineering holdout inputs changed after planning.")
        response_benchmark._assert_v2_case_outputs_fresh(full_plan, [row])
        subprocess.run(row["argv"], check=True)
        response_benchmark._assert_v2_post_dispatch(
            campaign_root,
            manifest,
            baseline_tree,
            row,
        )


def _require_exact_result_directories(results_root: Path) -> None:
    root = Path(results_root).expanduser().resolve()
    expected = set(engineering_holdout.engineering_screen_spec()["exact_case_ids"])
    present = {
        child.name
        for child in root.iterdir()
        if child.is_dir() and any(child.iterdir())
    }
    if present != expected:
        raise ValueError(
            "Engineering heldout result root must contain exactly the eight "
            f"selected nonempty case directories; missing={sorted(expected - present)}, "
            f"extra={sorted(present - expected)}."
        )


def _plot_assessment(assessment: Mapping[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = assessment["by_scruton_number"]
    labels = [f"Sc={float(row['scruton_number']):.2f}" for row in rows]
    targets = np.asarray(
        [row["target_peak_amplitude_over_diameter"] for row in rows], dtype=float
    )
    predictions = np.asarray(
        [row["selected_peak_amplitude_over_diameter"] for row in rows], dtype=float
    )
    tolerances = np.asarray([row["peak_amplitude_tolerance"] for row in rows])
    x = np.arange(len(rows), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.bar(x - width / 2, targets, width, label="Lupi experiment", color="#4C78A8")
    bars = ax.bar(
        x + width / 2,
        predictions,
        width,
        label="Frozen schema-v2 QBlade",
        color="#F58518",
    )
    ax.errorbar(
        x - width / 2,
        targets,
        yerr=tolerances,
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.2,
        label="engineering tolerance",
    )
    for bar, row in zip(bars, rows):
        target = float(row["target_peak_amplitude_over_diameter"])
        prediction = float(row["selected_peak_amplitude_over_diameter"])
        relative = 100.0 * (prediction - target) / target
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{relative:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Peak amplitude, $A/D$")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Sparse held-out lock-in amplitude screen")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def analyze_exact_eight(
    manifest_path: Path,
    calibration_readiness_path: Path,
    results_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    readiness, _ = _load_passing_readiness(calibration_readiness_path)
    _require_exact_result_directories(results_root)
    output = Path(output_dir).expanduser().resolve()
    response_benchmark.analyze_campaign(
        Path(manifest_path),
        Path(results_root),
        output,
    )
    _, campaign = response_benchmark._load_campaign(Path(manifest_path))
    selected_ids = engineering_holdout.engineering_screen_spec()["exact_case_ids"]
    by_id = {entry["case_id"]: entry for entry in campaign["cases"]}
    selected_cases = [by_id[case_id] for case_id in selected_ids]
    rows = _load_response_rows(output / "response_by_case.csv")
    audits = _json_file(output / "runtime_case_audits.json")
    evidence = {
        "campaign_manifest_sha256": engineering_holdout.CAMPAIGN_MANIFEST_SHA256,
        "stationary_protocol_sha256": engineering_holdout.STATIONARY_PROTOCOL_SHA256,
        "coefficient_fingerprint_sha256": (
            engineering_holdout.COEFFICIENT_FINGERPRINT_SHA256
        ),
        "calibration_readiness_content_sha256": engineering_holdout.semantic_sha256(
            readiness
        ),
        "selected_cases_content_sha256": engineering_holdout.semantic_sha256(
            selected_cases
        ),
        "response_rows_content_sha256": engineering_holdout.semantic_sha256(rows),
        "runtime_audits_content_sha256": engineering_holdout.semantic_sha256(audits),
    }
    assessment = engineering_holdout.assess_heldout_results(
        readiness,
        selected_cases,
        rows,
        audits,
        evidence,
    )
    (output / "engineering_holdout_screen_spec.json").write_text(
        json.dumps(
            engineering_holdout.engineering_screen_spec(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "engineering_holdout_assessment.json").write_text(
        json.dumps(assessment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_assessment(
        assessment,
        output / "engineering_holdout_peak_amplitude.pdf",
    )
    return assessment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--analysis", type=Path, required=True)
    readiness.add_argument("--output", type=Path, required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--calibration-readiness", type=Path, required=True)
    plan.add_argument("--results", type=Path, required=True)
    plan.add_argument("--qblade-root", type=Path, required=True)
    plan.add_argument("--checkpoint-every", type=int, default=0)
    plan.add_argument("--threads", type=int, default=4)
    plan.add_argument("--output", type=Path, required=True)

    execute = subparsers.add_parser("execute")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--calibration-readiness", type=Path, required=True)
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument(
        "--acknowledge-failed-strict-hgast-gate",
        action="store_true",
    )

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--calibration-readiness", type=Path, required=True)
    analyze.add_argument("--results", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "readiness":
        _write_json(args.output, build_calibration_readiness(args.analysis))
    elif args.command == "plan":
        _write_json(
            args.output,
            build_exact_eight_plan(
                args.manifest,
                args.calibration_readiness,
                args.results,
                qblade_root=args.qblade_root,
                checkpoint_every=args.checkpoint_every,
                threads=args.threads,
            ),
        )
    elif args.command == "execute":
        execute_exact_eight_plan(
            _json_file(args.plan),
            args.manifest,
            args.calibration_readiness,
            acknowledge_failed_strict_hgast_gate=(
                args.acknowledge_failed_strict_hgast_gate
            ),
        )
    elif args.command == "analyze":
        assessment = analyze_exact_eight(
            args.manifest,
            args.calibration_readiness,
            args.results,
            args.output,
        )
        print(json.dumps(assessment, indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
