from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from examples.marra2015_rectangular_4x1.assess_scaled_saturation import assess

REPO = Path(__file__).resolve().parents[1]


def test_marra_rows_are_exact_training_evidence() -> None:
    root = REPO / "examples/marra2015_rectangular_4x1"
    rows = list(
        csv.DictReader((root / "reference_data/gao2025_table5_training.csv").open())
    )
    assert [float(row["scruton_number"]) for row in rows] == [1.9, 21.7, 78.1]
    assert [float(row["peak_amplitude_over_depth"]) for row in rows] == [
        0.06551,
        0.0292,
        0.01106,
    ]
    assert {row["evidence_role"] for row in rows} == {"training"}
    provenance = json.loads(
        (root / "reference_data/training_provenance.json").read_text()
    )
    assert provenance["status"] == "frozen_as_training_after_prior_model_failure"
    assert provenance["claim_boundary"][
        "reclassified_as_training_after_that_model_failed"
    ]


def test_liu_target_and_nguyen_static_conversion_were_frozen_as_independent() -> None:
    root = REPO / "examples/liu2024_rectangular_5x1"
    provenance = json.loads(
        (root / "reference_data/validation_provenance.json").read_text()
    )
    assert provenance["status"] == "frozen_before_fitting_4x1_training_closure"
    response = provenance["response_experiment"]
    assert response["source_role"] == "untouched_independent_response_validation"
    assert response["target"]["maximum_allowed_model_relative_error"] == 0.30
    assert response["target"]["nominal_peak_hrms_over_depth"] == 0.02095808383233533

    rows = list(
        csv.DictReader(
            (root / "reference_data/nguyen2018_fixed_body_spectrum.csv").open()
        )
    )
    high_re = rows[-1]
    assert float(high_re["strouhal_crossflow_depth"]) == 0.1202
    assert float(high_re["rms_lift_coefficient_crossflow_depth"]) == 0.575
    assert math.isclose(
        float(high_re["sinusoidal_equivalent_lift_coefficient_crossflow_depth"]),
        math.sqrt(2.0) * 0.575,
        rel_tol=1e-15,
    )


def test_spectrum_scaled_rayleigh_revision_fails_untouched_4x1() -> None:
    report = assess()
    assert report["status"] == "failed_untouched_4x1"
    assert report["pass"] is False
    assert report["revision"]["not_fitted_to_4x1"] is True
    assert report["revision"]["q"] == pytest.approx(0.3271098634035249)
    rows = report["untouched_4x1_results"]
    assert rows[0]["absolute_relative_error"] < 0.30
    assert rows[1]["absolute_relative_error"] > 0.90
    assert rows[2]["absolute_relative_error"] > 3.6
    assert report["claim_boundary"]["4x1_may_now_be_used_for_training"] is True
