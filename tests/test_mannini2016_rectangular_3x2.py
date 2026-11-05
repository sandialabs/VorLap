from __future__ import annotations

import math

from examples.mannini2016_rectangular_3x2.assess_frozen_model import assess


def test_untuned_ehrmann_closure_fails_independent_3x2_saturation_bound() -> None:
    report = assess()
    assert report["status"] == "failed_frozen_model_saturation_bound"
    assert report["pass"] is False
    assert report["model"]["retuned_for_this_study"] is False
    assert math.isclose(
        report["experiment"]["peak_amplitude_over_depth"],
        0.04921875,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert report["assessment"]["best_possible_relative_error"] > 0.50
    assert report["assessment"]["maximum_allowed_relative_error"] == 0.30
    assert report["claim_boundary"]["does_not_validate_hvawt_or_fatigue"] is True
