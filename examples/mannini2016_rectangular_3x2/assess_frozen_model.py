#!/usr/bin/env python3
"""Assess the frozen Ehrmann-trained saturation bound against Mannini Sc=87.7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vorlap.qblade_engineering_lockin import EngineeringLockInCalibration


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = HERE / "reference_data/mannini2015_figure4_sc87p7_digitized.csv"
PROVENANCE = HERE / "reference_data/digitization_provenance.json"
CALIBRATION = (
    REPO / "examples/ehrmann2014_naca0018/engineering_lockin_calibration.json"
)
EXPECTED_CALIBRATION_SHA256 = (
    "d96c0b73a708b918360a15e79224f6a3a3f4e80702edf721bc606501a46fa02d"
)
SCHEMA = "vorlap-mannini2016-frozen-engineering-screen-v1"
MAX_RELATIVE_ERROR = 0.30


class AssessmentError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows() -> list[dict[str, Any]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 10 or {row["series"] for row in rows} != {"Sc_87p7"}:
        raise AssessmentError("Digitized Sc=87.7 response series is incomplete.")
    parsed = []
    for row in rows:
        if row["role"] != "independent_validation":
            raise AssessmentError("Response target was not frozen as validation data.")
        parsed.append(
            {
                **row,
                "u_over_ur": float(row["u_over_ur"]),
                "amplitude": float(row["amplitude_sqrt2_rms_over_depth"]),
                "uncertainty": float(row["amplitude_uncertainty"]),
            }
        )
    return parsed


def assess() -> dict[str, Any]:
    if _sha256(CALIBRATION) != EXPECTED_CALIBRATION_SHA256:
        raise AssessmentError("Frozen Ehrmann calibration bytes differ.")
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if (
        provenance.get("status") != "digitized_before_model_evaluation"
        or provenance["claim_boundary"]["first_use"]
        != "independent_no_retuning_validation"
        or provenance["experiment"]["scruton_number"] != 87.7
    ):
        raise AssessmentError("Digitization provenance or evidence role differs.")
    rows = _rows()
    peak = max(rows, key=lambda row: row["amplitude"])
    calibration = EngineeringLockInCalibration.from_json(CALIBRATION)
    limiting_ratio = calibration.limiting_amplitude_ratio
    # The experimental response and Strouhal number use the cross-flow depth D.
    # For c_fb=c_fb0*(1-(A/(lambda*D))^2), the supplemental aerodynamic
    # damping is restoring at A>lambda*D. In the high-Sc pure-VIV case, where
    # the experiment places quasi-steady galloping onset at 5.82 Ur, lambda is
    # therefore a hard upper bound on the closure-supported resonance branch.
    experiment_lower = max(0.0, peak["amplitude"] - peak["uncertainty"])
    minimum_relative_error = (
        (experiment_lower - limiting_ratio) / experiment_lower
        if experiment_lower > limiting_ratio
        else 0.0
    )
    passed = bool(minimum_relative_error <= MAX_RELATIVE_ERROR)
    return {
        "schema": SCHEMA,
        "status": (
            "passed_frozen_model_bound"
            if passed
            else "failed_frozen_model_saturation_bound"
        ),
        "pass": passed,
        "model": {
            "type": calibration.raw["model_type"],
            "calibration_role": "Ehrmann training and same-study qualification",
            "calibration_sha256": EXPECTED_CALIBRATION_SHA256,
            "calibration_fingerprint": calibration.fingerprint,
            "retuned_for_this_study": False,
            "limiting_amplitude_over_crossflow_depth": limiting_ratio,
            "bound_basis": "signed negative-damping saturation changes sign at A/(lambda*D)=1",
        },
        "experiment": {
            "study": "Mannini et al. 3:2 rectangular cylinder",
            "series": "Sc=87.7 separated pure-VIV resonance",
            "reynolds_at_resonance": 33600,
            "peak_u_over_ur": peak["u_over_ur"],
            "peak_amplitude_over_depth": peak["amplitude"],
            "amplitude_digitization_uncertainty": peak["uncertainty"],
            "uncertainty_favorable_lower_amplitude": experiment_lower,
        },
        "assessment": {
            "maximum_allowed_relative_error": MAX_RELATIVE_ERROR,
            "best_possible_absolute_error_over_depth": max(
                0.0, experiment_lower - limiting_ratio
            ),
            "best_possible_relative_error": minimum_relative_error,
            "pointwise_simulation_required_to_establish_this_failure": False,
            "reason": (
                "The experimental lower uncertainty bound exceeds the model's "
                "largest closure-supported amplitude by more than 30%."
            ),
        },
        "sources": {
            "digitized_response": {
                "path": str(DATA.relative_to(REPO)),
                "sha256": _sha256(DATA),
            },
            "digitization_provenance": {
                "path": str(PROVENANCE.relative_to(REPO)),
                "sha256": _sha256(PROVENANCE),
            },
        },
        "claim_boundary": {
            "independent_experimental_assessment": True,
            "failure_must_be_preserved_before_training": True,
            "does_not_validate_hvawt_or_fatigue": True,
            "next_action": "reclassify_3x2_response_as_training_then_freeze_one_physics_revision_and_test_an_untouched_study",
        },
    }


def _plot(report: dict[str, Any], output: Path) -> None:
    rows = _rows()
    x = [row["u_over_ur"] for row in rows]
    y = [row["amplitude"] for row in rows]
    uncertainty = report["experiment"]["amplitude_digitization_uncertainty"]
    limit = report["model"]["limiting_amplitude_over_crossflow_depth"]
    fig, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    axis.plot(x, y, "+", color="#b21f2d", ms=6, mew=1.2, label="Experiment, Sc=87.7")
    axis.fill_between(
        x,
        [max(0.0, value - uncertainty) for value in y],
        [value + uncertainty for value in y],
        color="#b21f2d",
        alpha=0.12,
        linewidth=0,
        label="Digitization allowance",
    )
    axis.axhline(
        limit,
        color="#2468a2",
        lw=1.8,
        label="Frozen closure saturation bound",
    )
    axis.set(xlabel=r"$U/U_r$", ylabel=r"$\sqrt{2}\,y'/D$", xlim=(0.0, 2.0), ylim=(0.0, 0.06))
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        0.98,
        0.05,
        f"Best-case error = {100*report['assessment']['best_possible_relative_error']:.1f}%\n30% gate: FAIL",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
    )
    fig.savefig(output)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "evidence")
    args = parser.parse_args(argv)
    report = assess()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "frozen_model_screen.json"
    plot_path = args.output_dir / "frozen_model_screen.pdf"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _plot(report, plot_path)
    sums = {
        path.name: _sha256(path)
        for path in (report_path, plot_path)
    }
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
