#!/usr/bin/env python3
"""Draft a structural-model qualification manifest for a parked QBlade run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


STRUCTURAL_QUALIFICATION_DEFAULT = "structural_model_qualification.json"
STRUCTURAL_MODEL_CHOICES = (
    "generated_patched_demonstration",
    "repaired_original_export",
    "source_exported_model",
)
SOLVER_KEYS = ("TIMESTEP", "STRITERATIONS", "RAMPUP", "ADDDAMP")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_root(run_root: Path, scenario: str | None) -> Path:
    if scenario:
        path = run_root / scenario
        if not path.is_dir():
            raise FileNotFoundError(f"Scenario directory does not exist: {path}")
        return path
    preferred = run_root / "vorlap"
    if preferred.is_dir():
        return preferred
    candidates = sorted(path.parent for path in run_root.glob("*/cases.json") if path.parent.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No scenario cases.json files found under {run_root}")
    return candidates[0]


def discover_structural_files(root: Path) -> list[Path]:
    files = sorted(root.glob("cases/*/*/Structure/*.str"))
    if not files:
        raise FileNotFoundError(f"No QBlade structural .str files found under {root}")
    return files


def discover_patch_manifests(root: Path) -> list[Path]:
    return sorted(root.glob("cases/*/*/Control/vorlap_stable_support_patch.json"))


def first_run_summary(root: Path) -> Path | None:
    summaries = sorted(root.glob("results/*/*_run_summary.json"))
    return summaries[0] if summaries else None


def solver_settings(run_root: Path, root: Path) -> dict[str, float]:
    solver: dict[str, float] = {}
    summary_path = first_run_summary(root)
    if summary_path is not None:
        payload = read_json(summary_path)
        qblade_settings = (
            (payload.get("reproducibility") or {}).get("qblade_case_settings")
            if isinstance(payload.get("reproducibility"), dict)
            else {}
        )
        if isinstance(qblade_settings, dict):
            for key in SOLVER_KEYS:
                raw = qblade_settings.get(key)
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    solver[key] = value
    settings_path = run_root / "campaign_settings.json"
    if settings_path.is_file():
        settings = read_json(settings_path)
        fallback = {
            "TIMESTEP": settings.get("timestep_s"),
            "RAMPUP": settings.get("rampup_s", 0.0),
            "ADDDAMP": settings.get("add_damp_s", 0.0),
            "STRITERATIONS": settings.get("str_iterations"),
        }
        for key, raw in fallback.items():
            if key in solver or raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                solver[key] = value
    return solver


def damping_summary(paths: list[Path], root: Path) -> dict[str, Any]:
    rayleigh: dict[str, str] = {}
    for path in paths:
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "RAYLEIGHDMP" in line.upper():
                    rayleigh[_relative_or_absolute(root, path)] = line.strip()
                    break
        except OSError:
            continue
    return {
        "source": "QBlade structural files",
        "assumption": "Damping entries are read from structural RAYLEIGHDMP rows and must be reviewed before final claims.",
        "rayleigh_damping_rows": rayleigh,
    }


def file_entries(paths: list[Path], root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": _relative_or_absolute(root, path),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def baseline_identity_artifact(run_root: Path) -> Path:
    return run_root / "postprocess" / "baseline_identity_comparison.csv"


def build_manifest(
    run_root: Path,
    *,
    scenario: str | None = None,
    model_id: str | None = None,
    model_choice: str = "generated_patched_demonstration",
    modal_frequencies_hz: list[float],
    qualified: bool = False,
    baseline_identity_passed: bool = False,
) -> dict[str, Any]:
    root = run_root.resolve()
    if model_choice not in STRUCTURAL_MODEL_CHOICES:
        raise ValueError(f"model_choice must be one of {', '.join(STRUCTURAL_MODEL_CHOICES)}")
    if not modal_frequencies_hz:
        raise ValueError("At least one modal frequency must be supplied.")
    for frequency in modal_frequencies_hz:
        if not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
            raise ValueError("Modal frequencies must be finite positive Hz values.")

    selected_root = scenario_root(root, scenario)
    structural_files = discover_structural_files(selected_root)
    patch_manifests = discover_patch_manifests(selected_root)
    baseline = baseline_identity_artifact(root)
    if not baseline.is_file():
        raise FileNotFoundError(f"Baseline identity artifact does not exist: {baseline}")

    return {
        "model_id": model_id or f"{root.name}-{selected_root.name}-structural-model",
        "model_choice": model_choice,
        "qualified": bool(qualified),
        "review_status": "draft_generated_from_run_archive",
        "scenario": selected_root.name,
        "modal_frequencies_hz": [float(frequency) for frequency in modal_frequencies_hz],
        "damping": damping_summary(structural_files, root),
        "solver_settings": solver_settings(root, selected_root),
        "structural_files": file_entries(structural_files, root),
        "patch_manifests": file_entries(patch_manifests, root),
        "baseline_identity_evidence": {
            "status": "passed" if baseline_identity_passed else "pending_review",
            "artifact": _relative_or_absolute(root, baseline),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Campaign run root.")
    parser.add_argument("--scenario", default=None, help="Scenario directory to qualify. Defaults to vorlap when present.")
    parser.add_argument("--model-id", default=None, help="Model identifier. Defaults to '<run-root>-<scenario>-structural-model'.")
    parser.add_argument("--model-choice", choices=STRUCTURAL_MODEL_CHOICES, default="generated_patched_demonstration")
    parser.add_argument("--modal-frequency-hz", type=float, action="append", default=[], help="Reviewed modal frequency in Hz. Repeat for each mode.")
    parser.add_argument("--qualified", action="store_true", help="Set qualified=true after review.")
    parser.add_argument("--baseline-identity-passed", action="store_true", help="Mark baseline_identity_evidence.status as passed after review.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output manifest path. Defaults to RUN_ROOT/{STRUCTURAL_QUALIFICATION_DEFAULT}.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out_path = (args.out or (run_root / STRUCTURAL_QUALIFICATION_DEFAULT)).resolve()
    if out_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing manifest without --force: {out_path}")
    manifest = build_manifest(
        run_root,
        scenario=args.scenario,
        model_id=args.model_id,
        model_choice=args.model_choice,
        modal_frequencies_hz=args.modal_frequency_hz,
        qualified=args.qualified,
        baseline_identity_passed=args.baseline_identity_passed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
