import copy
import json
import math

import numpy as np
import pytest

from vorlap.qblade_airfoil_work_map import (
    AirfoilForceMotionWorkMap,
    SCHEMA,
    frozen_conventions,
)


def _payload():
    axes = {
        "incidence_deg": [80.0, 100.0],
        "reynolds_number": [1.0e5, 4.0e5],
        "reduced_frequency_k": [0.1, 0.3],
        "amplitude_ratio": [0.0, 0.1, 0.2],
        "direction_deg_mod_180": [0.0, 90.0],
    }
    shape = tuple(len(axes[name]) for name in axes)
    entrainment = np.zeros(shape)
    power = np.zeros(shape)
    uncertainty = np.zeros(shape)
    reference_available = np.ones(shape)
    for ia in range(shape[0]):
        for ir in range(shape[1]):
            for ik in range(shape[2]):
                for iam in range(1, shape[3]):
                    for ib in range(shape[4]):
                        vertex = 0.10 + ia + 2 * ir + 4 * ik + 8 * (iam - 1) + 16 * ib
                        entrainment[ia, ir, ik, iam, ib] = min(1.0, 0.1 * vertex)
                        power[ia, ir, ik, iam, ib] = vertex
                        uncertainty[ia, ir, ik, iam, ib] = 0.05 * vertex
    return {
        "schema": SCHEMA,
        "status": "calibrated",
        "profile": "synthetic-test-airfoil",
        "motion": "prescribed_crossflow_translation",
        "axes": axes,
        "fields": {
            "entrainment_fraction": entrainment.tolist(),
            "reference_available_force_coefficient": reference_available.tolist(),
            "signed_power_coefficient": power.tolist(),
            "power_coefficient_standard_uncertainty": uncertainty.tolist(),
        },
        "conventions": frozen_conventions(),
        "sources": [
            {
                "role": "prescribed_motion_high_fidelity",
                "label": "synthetic high fidelity",
                "sha256": "1" * 64,
            },
            {
                "role": "exact_runtime_base",
                "label": "synthetic runtime base",
                "sha256": "2" * 64,
            },
        ],
    }


def test_exact_knots_log_reynolds_and_convex_multilinear_interpolation():
    model = AirfoilForceMotionWorkMap(_payload())
    knot = model.query(
        incidence_deg=80.0,
        reynolds_number=1.0e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=0.0,
        runtime_available_force_coefficient=1.0,
    )
    assert knot.signed_power_coefficient == pytest.approx(0.10)

    midpoint = model.query(
        incidence_deg=90.0,
        reynolds_number=2.0e5,
        reduced_frequency_k=0.2,
        amplitude_ratio=0.15,
        direction_deg=45.0,
        runtime_available_force_coefficient=1.0,
    )
    corners = []
    for incidence in (80.0, 100.0):
        for reynolds in (1.0e5, 4.0e5):
            for k in (0.1, 0.3):
                for amplitude in (0.1, 0.2):
                    for direction in (0.0, 90.0):
                        corners.append(
                            model.query(
                                incidence_deg=incidence,
                                reynolds_number=reynolds,
                                reduced_frequency_k=k,
                                amplitude_ratio=amplitude,
                                direction_deg=direction,
                                runtime_available_force_coefficient=1.0,
                            ).signed_power_coefficient
                        )
    assert midpoint.signed_power_coefficient == pytest.approx(np.mean(corners))
    assert min(corners) <= midpoint.signed_power_coefficient <= max(corners)


def test_quadratic_origin_has_finite_equivalent_gain_and_exact_zero():
    model = AirfoilForceMotionWorkMap(_payload())
    zero = model.query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0,
        direction_deg=0,
        runtime_available_force_coefficient=1.0,
    )
    half = model.query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.05,
        direction_deg=0,
        runtime_available_force_coefficient=1.0,
    )
    full = model.query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=0,
        runtime_available_force_coefficient=1.0,
    )
    assert zero.signed_power_coefficient == 0.0
    assert half.signed_power_coefficient == pytest.approx(0.25 * full.signed_power_coefficient)
    assert half.signed_power_coefficient / 0.05**2 == pytest.approx(
        full.signed_power_coefficient / 0.1**2
    )


def test_direction_is_modulo_pi_but_not_silently_extrapolated():
    model = AirfoilForceMotionWorkMap(_payload())
    forward = model.query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=45,
        runtime_available_force_coefficient=1.0,
    )
    reversed_mode = model.query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=225,
        runtime_available_force_coefficient=1.0,
    )
    assert forward == reversed_mode
    across_seam = model.query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=135,
        runtime_available_force_coefficient=1.0,
    )
    endpoint_mean = 0.5 * (
        model.query(
            incidence_deg=80,
            reynolds_number=1e5,
            reduced_frequency_k=0.1,
            amplitude_ratio=0.1,
            direction_deg=90,
            runtime_available_force_coefficient=1.0,
        ).signed_power_coefficient
        + model.query(
            incidence_deg=80,
            reynolds_number=1e5,
            reduced_frequency_k=0.1,
            amplitude_ratio=0.1,
            direction_deg=0,
            runtime_available_force_coefficient=1.0,
        ).signed_power_coefficient
    )
    assert across_seam.signed_power_coefficient == pytest.approx(endpoint_mean)
    with pytest.raises(ValueError, match="incidence_deg"):
        model.query(
            incidence_deg=79,
            reynolds_number=1e5,
            reduced_frequency_k=0.1,
            amplitude_ratio=0.1,
            direction_deg=0,
            runtime_available_force_coefficient=1.0,
        )
    with pytest.raises(ValueError, match="amplitude_ratio"):
        model.query(
            incidence_deg=80,
            reynolds_number=1e5,
            reduced_frequency_k=0.1,
            amplitude_ratio=0.21,
            direction_deg=0,
            runtime_available_force_coefficient=1.0,
        )


def test_uncertainty_lower_bound_never_injects_negative_work():
    payload = _payload()
    payload["fields"]["signed_power_coefficient"][0][0][0][1][0] = -0.1
    model = AirfoilForceMotionWorkMap(payload)
    negative = model.query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=0,
        runtime_available_force_coefficient=1.0,
    )
    assert negative.signed_power_coefficient == -0.1
    assert negative.positive_injection_power_coefficient == 0.0

    positive = AirfoilForceMotionWorkMap(_payload()).query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=0,
        runtime_available_force_coefficient=1.0,
        confidence_multiplier=3.0,
    )
    assert positive.lower_confidence_power_coefficient == pytest.approx(0.085)
    assert positive.positive_injection_power_coefficient == pytest.approx(0.085)


def test_saturation_domain_requires_positive_work_to_end_nonpositive():
    unbounded = AirfoilForceMotionWorkMap(_payload())
    assert unbounded.positive_work_domain_status() == {
        "has_positive_work": True,
        "highest_amplitude_plane_nonpositive": False,
    }
    payload = _payload()
    terminal = np.asarray(payload["fields"]["signed_power_coefficient"])
    terminal[:, :, :, -1, :] = -0.01
    payload["fields"]["signed_power_coefficient"] = terminal.tolist()
    bounded = AirfoilForceMotionWorkMap(payload)
    assert bounded.positive_work_domain_status() == {
        "has_positive_work": True,
        "highest_amplitude_plane_nonpositive": True,
    }


def test_static_spectrum_scales_available_work_amplitude_but_not_sign():
    model = AirfoilForceMotionWorkMap(_payload())
    query = model.query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=0,
        runtime_available_force_coefficient=0.4,
    )
    assert query.reference_available_force_coefficient == 1.0
    assert query.available_force_amplitude_scale == pytest.approx(0.4)
    assert query.scaled_positive_injection_power_coefficient == pytest.approx(
        0.4 * query.positive_injection_power_coefficient
    )

    no_source = model.query(
        incidence_deg=80,
        reynolds_number=1e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=0,
        runtime_available_force_coefficient=0.0,
    )
    assert no_source.scaled_positive_injection_power_coefficient == 0.0


def test_dimensionalization_is_power_conjugate_and_has_si_units():
    coefficient = 0.2
    power = AirfoilForceMotionWorkMap.dimensional_power_w(
        coefficient,
        fluid_density_kgpm3=1.2,
        speed_mps=10.0,
        chord_m=0.25,
        span_m=2.0,
    )
    assert power == pytest.approx(60.0)
    force_amplitude = 100.0
    velocity_amplitude = 2.0
    phase = np.linspace(0.0, 2.0 * math.pi, 20001)
    force = force_amplitude * np.cos(phase)
    velocity = velocity_amplitude * np.cos(phase)
    assert np.trapezoid(force * velocity, phase) / (2.0 * math.pi) == pytest.approx(
        0.5 * force_amplitude * velocity_amplitude,
        rel=1e-8,
    )


def test_json_bytes_are_hash_bound(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(_payload(), sort_keys=True) + "\n", encoding="utf-8")
    first = AirfoilForceMotionWorkMap.from_json(path)
    mutated = _payload()
    mutated["profile"] = "synthetic-test-airfoil-v2"
    path.write_text(json.dumps(mutated, sort_keys=True) + "\n", encoding="utf-8")
    second = AirfoilForceMotionWorkMap.from_json(path)
    assert first.source_sha256 != second.source_sha256
    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda value: value["sources"].pop(), "paired high-fidelity"),
        (
            lambda value: value["fields"]["signed_power_coefficient"][0][0][0][0].__setitem__(0, 1.0),
            "exactly zero",
        ),
        (
            lambda value: value["axes"]["reynolds_number"].__setitem__(0, 0.0),
            "must be positive",
        ),
        (
            lambda value: value["conventions"].__setitem__("power_sign", "opposite"),
            "frozen policy",
        ),
    ],
)
def test_invalid_or_unpaired_map_evidence_fails_closed(mutator, match):
    payload = copy.deepcopy(_payload())
    mutator(payload)
    with pytest.raises(ValueError, match=match):
        AirfoilForceMotionWorkMap(payload)
