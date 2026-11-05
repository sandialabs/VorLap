import hashlib
import json
import math
from pathlib import Path

import pytest

from examples.fu2026_iea10 import fit_signed_work_shape_transfer as fit


def _fingerprint(document):
    unsigned = dict(document)
    stored = unsigned.pop("model_fingerprint")
    actual = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return stored, actual


def test_tracked_signed_work_shape_is_exact_reproduction(tmp_path: Path) -> None:
    expected = fit.train()
    tracked = json.loads(fit.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert tracked == expected
    stored, actual = _fingerprint(tracked)
    assert stored == actual
    assert len(tracked["cases"]) == 20


def test_shape_transfer_has_one_source_ratio_and_exact_active_roots() -> None:
    model = fit.train()
    method = model["method"]
    kappa = fit.SOURCE_ZERO_WORK_AMPLITUDE_M / fit.SOURCE_FSI_AMPLITUDE_M
    feedback = 1.0 / (1.0 - 1.0 / kappa**2)
    assert method["zero_to_equilibrium_amplitude_ratio"] == pytest.approx(kappa)
    assert method["active_feedback_over_opposing_damping"] == pytest.approx(
        feedback
    )
    assert method["type"] == (
        "grinderslev_landmark_constrained_quadratic_saturation_transfer"
    )
    assert "full signed-power curve is not transferred" in method["transfer_scope"]
    for row in model["cases"]:
        if row["active_branch"]:
            assert row["transferred_feedback_over_opposing_damping"] == pytest.approx(
                feedback
            )
            root = row["transferred_limiting_amplitude_m"] * math.sqrt(
                1.0 - 1.0 / feedback
            )
            assert root == pytest.approx(row["equilibrium_amplitude_m"], abs=2e-12)
        else:
            assert row["transferred_feedback_over_opposing_damping"] == 0.0


def test_shape_transfer_is_explicitly_training_only() -> None:
    model = fit.train()
    assert model["status"] == "qualified_training_reproduction_only"
    assert "independent validation" in model["claim_boundary"]["does_not_support"]
    assert (
        model["sources"]["geometry_composed_training_model"]["sha256"]
        == fit._sha256(fit.SOURCE_MODEL)
    )
