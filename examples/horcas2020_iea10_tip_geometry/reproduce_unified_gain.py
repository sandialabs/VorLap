#!/usr/bin/env python3
"""Replay the frozen Horcas predictions through the unified gain evaluator."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from . import assess_geometry_composition as frozen_assessment
from vorlap.unified_viv_gain import UnifiedVIVGainModel


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVIDENCE = HERE / "evidence/combined_tip_v1"
UNIFIED_MODEL_PATH = REPO / "run_headless/config/unified_viv_gain_model_v1.json"
UNIFIED_EVALUATOR_PATH = REPO / "vorlap/unified_viv_gain.py"
RECEIPT_PATH = EVIDENCE / "unified_gain_reproduction.json"
INCLINATIONS = (40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 65.0, 70.0)


class UnifiedReproductionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnifiedReproductionError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unified_predictions() -> list[dict[str, Any]]:
    """Return all 11 predictions without reading COMB target values."""

    evaluator = UnifiedVIVGainModel.from_json(
        UNIFIED_MODEL_PATH,
        repository_root=REPO,
    )
    rows: list[dict[str, Any]] = []
    for inclination in INCLINATIONS:
        components = {
            geometry: evaluator.evaluate_turbine_geometry(
                profile_family="FFA-W3_family",
                geometry_family=geometry,
                inclination_deg=inclination,
            )
            for geometry in ("EXT", "SWEEP", "DIHE")
        }
        combined = evaluator.evaluate_turbine_geometry(
            profile_family="FFA-W3_family",
            geometry_family="COMB",
            inclination_deg=inclination,
        )
        rows.append(
            {
                "case_id": f"COMB_I{inclination:05.1f}".replace(".", "p"),
                "model_fingerprint": combined.source_fingerprint,
                "inclination_deg": inclination,
                "predicted_peak_edge_amplitude_m": (
                    combined.stable_equilibrium_amplitude_m()
                ),
                "predicted_response_frequency_over_fn": (
                    combined.response_frequency_ratio
                ),
                "combined_normalized_positive_work_margin": combined.work_margin,
                "zero_amplitude_feedback_over_structural_damping": (
                    combined.zero_amplitude_gain
                ),
                "ext_work_margin": components["EXT"].work_margin,
                "sweep_work_margin": components["SWEEP"].work_margin,
                "dihe_work_margin": components["DIHE"].work_margin,
                "unified_model_fingerprint": evaluator.fingerprint,
            }
        )
    return rows


def reproduce() -> dict[str, Any]:
    """Require exact prediction parity, then inherit the frozen assessment."""

    stored = _rows(frozen_assessment.PREDICTIONS_PATH)
    unified = unified_predictions()
    _require(len(stored) == len(unified) == 11, "Unified prediction count changed.")
    numeric_fields = (
        "inclination_deg",
        "predicted_peak_edge_amplitude_m",
        "predicted_response_frequency_over_fn",
        "combined_normalized_positive_work_margin",
        "zero_amplitude_feedback_over_structural_damping",
        "ext_work_margin",
        "sweep_work_margin",
        "dihe_work_margin",
    )
    for stored_row, unified_row in zip(stored, unified):
        _require(
            stored_row["case_id"] == unified_row["case_id"]
            and stored_row["model_fingerprint"] == unified_row["model_fingerprint"],
            "Unified prediction identity differs from the frozen row.",
        )
        for field in numeric_fields:
            _require(
                float(stored_row[field]) == float(unified_row[field]),
                f"Unified prediction field {field} differs from the frozen row.",
            )
    frozen = frozen_assessment.assess()
    return {
        "schema": "horcas2020-unified-gain-reproduction-v1",
        "status": "passed" if frozen["status"] == "passed" else "failed",
        "prediction_parity": {
            "comparison": "exact_binary64_after_decimal_parse",
            "all_11_rows_exact": True,
            "row_count": len(unified),
        },
        "unified_model": {
            "path": str(UNIFIED_MODEL_PATH.relative_to(REPO)),
            "sha256": _sha(UNIFIED_MODEL_PATH),
            "model_fingerprint": unified[0]["unified_model_fingerprint"],
            "evaluator_path": str(UNIFIED_EVALUATOR_PATH.relative_to(REPO)),
            "evaluator_sha256": _sha(UNIFIED_EVALUATOR_PATH),
        },
        "frozen_evidence": {
            "model_sha256": frozen_assessment.MODEL_SHA,
            "predictions_sha256": frozen_assessment.PREDICTIONS_SHA,
            "assessment_sha256": _sha(frozen_assessment.ASSESSMENT_PATH),
            "assessment_code_sha256": _sha(HERE / "assess_geometry_composition.py"),
            "model_fingerprint": frozen_assessment.EXPECTED_MODEL_FINGERPRINT,
        },
        "reproduction_code_sha256": _sha(Path(__file__).resolve()),
        "metrics": frozen["metrics"],
        "gates": frozen["gates"],
        "claim": frozen["claim"],
        "claim_boundary": frozen["claim_boundary"],
        "conservative_engineering_envelope": frozen[
            "conservative_engineering_envelope"
        ],
    }


def main() -> None:
    result = reproduce()
    RECEIPT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"receipt": str(RECEIPT_PATH), "status": result["status"]}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
