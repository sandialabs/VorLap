#!/usr/bin/env python3
"""Freeze the turbine saturation landmarks used by the Fu reproduction closure.

Grinderslev et al. provide the only local turbine-scale signed aerodynamic-work
curve.  For P100-I50, its positive-work branch crosses zero at 1.57 m and the
reported FSI limit-cycle amplitude is approximately 1.18 m.  Their ratio fixes
two landmarks of an assumed quadratic negative-damping branch independently of
QBlade's native aerodynamic damping.  This trainer transfers only that landmark
ratio—not the full Grinderslev signed-work curve—to every active Fu training row
while retaining the row's fitted equilibrium.

All Fu rows in the source model were explicitly reclassified as training after
the earlier held-out attempt failed.  Consequently this artifact supports a
training reproduction only, never independent Fu validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MODEL = (
    REPO_ROOT
    / "examples/horcas2020_iea10_tip_geometry/evidence/combined_tip_v1/model.json"
)
SIGNED_POWER_CURVE = (
    REPO_ROOT
    / "examples/grinderslev2022_iea10/reference_data/"
    "p100_i50_figure6_power_digitized.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "examples/fu2026_iea10/evidence/signed_work_shape_v1/model.json"
)
SCHEMA = "vorlap-fu2026-iea10-signed-work-shape-transfer-v1"
REFERENCE_CHORD_M = 3.72
ACTIVE_THRESHOLD_RATIO = 0.01
SOURCE_ZERO_WORK_AMPLITUDE_M = 1.57
SOURCE_FSI_AMPLITUDE_M = 1.18
EXPECTED_CASE_COUNT = 20


class TrainingError(RuntimeError):
    """Raised when signed-work-shape inputs violate the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def train(source: Path = SOURCE_MODEL) -> dict[str, Any]:
    source = Path(source).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema") != "vorlap-turbine-geometry-composed-negative-damping-v1":
        raise TrainingError("Unsupported geometry-composed source model schema.")
    rows = document.get("training", {}).get(
        "fu2026_speed_orientation_rows_after_preserved_failure"
    )
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASE_COUNT:
        raise TrainingError("Expected exactly 20 reclassified Fu training rows.")
    if not SIGNED_POWER_CURVE.is_file():
        raise TrainingError("The frozen Grinderslev signed-power curve is missing.")

    shape_ratio = SOURCE_ZERO_WORK_AMPLITUDE_M / SOURCE_FSI_AMPLITUDE_M
    feedback_ratio = 1.0 / (1.0 - 1.0 / shape_ratio**2)
    if not (shape_ratio > 1.0 and feedback_ratio > 1.0):
        raise TrainingError("Signed-work shape does not define a stable branch.")

    trained: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_row in rows:
        case_id = str(source_row.get("case_id", ""))
        if not case_id or case_id in seen:
            raise TrainingError("Fu training row IDs must be unique and nonempty.")
        seen.add(case_id)
        amplitude = float(source_row["stable_peak_amplitude_m"])
        original_ratio = float(
            source_row["zero_amplitude_feedback_over_structural_damping"]
        )
        original_limit = float(
            source_row["configuration_saturation_amplitude_m"]
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (amplitude, original_ratio, original_limit)
        ):
            raise TrainingError(f"{case_id} contains invalid positive quantities.")
        reconstructed = (
            original_limit * math.sqrt(max(0.0, 1.0 - 1.0 / original_ratio))
            if original_ratio > 1.0
            else 0.0
        )
        if not math.isclose(reconstructed, amplitude, rel_tol=0.0, abs_tol=2.0e-9):
            raise TrainingError(
                f"{case_id} equilibrium does not close against its source branch."
            )
        amplitude_ratio = amplitude / REFERENCE_CHORD_M
        active = bool(amplitude_ratio > ACTIVE_THRESHOLD_RATIO)
        trained.append(
            {
                "case_id": case_id,
                "configuration": str(source_row["config"]),
                "wind_speed_mps": float(source_row["wind_speed_mps"]),
                "equilibrium_amplitude_m": amplitude,
                "equilibrium_amplitude_over_reference_chord": amplitude_ratio,
                "active_branch": active,
                "original_feedback_over_structural_damping": original_ratio,
                "original_limiting_amplitude_m": original_limit,
                "transferred_feedback_over_opposing_damping": (
                    feedback_ratio if active else 0.0
                ),
                "transferred_limiting_amplitude_m": (
                    shape_ratio * amplitude if active else original_limit
                ),
            }
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "qualified_training_reproduction_only",
        "method": {
            "type": "grinderslev_landmark_constrained_quadratic_saturation_transfer",
            "source_zero_work_amplitude_m": SOURCE_ZERO_WORK_AMPLITUDE_M,
            "source_fsi_lco_amplitude_m": SOURCE_FSI_AMPLITUDE_M,
            "zero_to_equilibrium_amplitude_ratio": shape_ratio,
            "active_feedback_over_opposing_damping": feedback_ratio,
            "active_target_threshold_Ay_over_cbar": ACTIVE_THRESHOLD_RATIO,
            "reference_mean_chord_m": REFERENCE_CHORD_M,
            "inactive_correlated_feedback": "exactly_zero",
            "runtime_balance": (
                "c0=r_shape*c_opposing; A_limit=kappa*A_equilibrium; "
                "c_opposing is independently identified native-total QBlade damping"
            ),
            "transfer_scope": (
                "Only the Grinderslev zero-work-to-FSI-amplitude ratio constrains "
                "the quadratic branch; the full signed-power curve is not transferred."
            ),
        },
        "sources": {
            "geometry_composed_training_model": {
                "path": source.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(source),
                "model_fingerprint": str(document["model_fingerprint"]),
            },
            "grinderslev_signed_power_curve": {
                "path": SIGNED_POWER_CURVE.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(SIGNED_POWER_CURVE),
                "publication_doi": "10.5194/wes-7-2201-2022",
            },
        },
        "claim_boundary": {
            "supports": (
                "Grinderslev-landmark-constrained quadratic reproduction of the 20 "
                "Fu turbine training rows"
            ),
            "does_not_support": (
                "independent validation on any Fu row, transfer of the complete "
                "Grinderslev signed-work curve, or universal transfer to other structures"
            ),
        },
        "cases": sorted(trained, key=lambda row: row["case_id"]),
    }
    payload["model_fingerprint"] = _semantic_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = train(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(result["model_fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
