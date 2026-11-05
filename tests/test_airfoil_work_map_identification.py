import math
import copy

import numpy as np
import pytest

from vorlap.airfoil_work_map_identification import (
    assemble_airfoil_work_map_payload,
    identify_prescribed_motion_work,
)
from vorlap.qblade_airfoil_work_map import AirfoilForceMotionWorkMap


def _histories(*, force_phase=0.0, cycle_offsets=(), cycles=13, samples_per_cycle=400):
    frequency = 2.0
    omega = 2.0 * math.pi * frequency
    time = np.linspace(0.0, cycles / frequency, cycles * samples_per_cycle + 1)
    amplitude = 0.03
    displacement = amplitude * np.sin(omega * time)
    velocity = amplitude * omega * np.cos(omega * time)
    base = 10.0 * np.sin(omega * time + 0.2)
    incremental = 20.0 * np.cos(omega * time + force_phase)
    for cycle, offset in enumerate(cycle_offsets):
        mask = (time >= cycle / frequency) & (time <= (cycle + 1) / frequency)
        incremental[mask] *= 1.0 + offset
    return time, displacement, velocity, base + incremental, base, frequency


def test_exact_runtime_base_subtraction_recovers_positive_cycle_work_and_phase():
    time, displacement, velocity, high, base, frequency = _histories()
    result = identify_prescribed_motion_work(
        time_s=time,
        displacement_m=displacement,
        velocity_mps=velocity,
        high_fidelity_force_n=high,
        runtime_base_force_n=base,
        motion_frequency_hz=frequency,
        fluid_density_kgpm3=1.2,
        speed_mps=10.0,
        chord_m=0.25,
        span_m=2.0,
        discard_cycles=5,
    )
    expected_power = 0.5 * 20.0 * 0.03 * 2.0 * math.pi * frequency
    assert result.mean_incremental_power_w == pytest.approx(expected_power, rel=1e-5)
    assert result.motion_amplitude_m == pytest.approx(0.03, rel=1e-8)
    assert result.phase_force_relative_to_velocity_rad == pytest.approx(0.0, abs=1e-8)
    assert result.incremental_force_velocity_harmonic_n == pytest.approx(20.0, rel=1e-8)
    assert result.incremental_force_displacement_harmonic_n == pytest.approx(0.0, abs=1e-8)
    assert result.signed_power_coefficient == pytest.approx(expected_power / 300.0)
    assert result.included_cycle_count == 8
    assert result.minimum_samples_per_included_cycle >= 400


def test_quadrature_force_has_zero_work_and_opposite_force_removes_energy():
    time, displacement, velocity, high, base, frequency = _histories(force_phase=math.pi / 2)
    quadrature = identify_prescribed_motion_work(
        time_s=time,
        displacement_m=displacement,
        velocity_mps=velocity,
        high_fidelity_force_n=high,
        runtime_base_force_n=base,
        motion_frequency_hz=frequency,
        fluid_density_kgpm3=1.2,
        speed_mps=10.0,
        chord_m=0.25,
        span_m=2.0,
    )
    assert quadrature.mean_incremental_power_w == pytest.approx(0.0, abs=1e-12)

    time, displacement, velocity, high, base, frequency = _histories(force_phase=math.pi)
    damping = identify_prescribed_motion_work(
        time_s=time,
        displacement_m=displacement,
        velocity_mps=velocity,
        high_fidelity_force_n=high,
        runtime_base_force_n=base,
        motion_frequency_hz=frequency,
        fluid_density_kgpm3=1.2,
        speed_mps=10.0,
        chord_m=0.25,
        span_m=2.0,
    )
    assert damping.mean_incremental_power_w < 0.0
    assert abs(abs(damping.phase_force_relative_to_velocity_rad) - math.pi) < 1e-8


def test_cycle_uncertainty_is_reported_instead_of_hidden():
    histories = _histories(
        cycle_offsets=(0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1)
    )
    result = identify_prescribed_motion_work(
        time_s=histories[0],
        displacement_m=histories[1],
        velocity_mps=histories[2],
        high_fidelity_force_n=histories[3],
        runtime_base_force_n=histories[4],
        motion_frequency_hz=histories[5],
        fluid_density_kgpm3=1.2,
        speed_mps=10.0,
        chord_m=0.25,
        span_m=2.0,
    )
    assert result.included_cycle_count == 8
    assert result.standard_uncertainty_power_w > 0.0
    assert result.power_coefficient_standard_uncertainty > 0.0


def test_rejects_partial_or_nonmonotone_histories():
    time, displacement, velocity, high, base, frequency = _histories()
    with pytest.raises(ValueError, match="at least 8 complete cycles"):
        identify_prescribed_motion_work(
            time_s=time[:200],
            displacement_m=displacement[:200],
            velocity_mps=velocity[:200],
            high_fidelity_force_n=high[:200],
            runtime_base_force_n=base[:200],
            motion_frequency_hz=frequency,
            fluid_density_kgpm3=1.2,
            speed_mps=10.0,
            chord_m=0.25,
            span_m=2.0,
        )
    bad_time = time.copy()
    bad_time[10] = bad_time[9]
    with pytest.raises(ValueError, match="strictly increasing"):
        identify_prescribed_motion_work(
            time_s=bad_time,
            displacement_m=displacement,
            velocity_mps=velocity,
            high_fidelity_force_n=high,
            runtime_base_force_n=base,
            motion_frequency_hz=frequency,
            fluid_density_kgpm3=1.2,
            speed_mps=10.0,
            chord_m=0.25,
            span_m=2.0,
        )


def test_rejects_quality_below_discard_cycle_and_sampling_floors():
    histories = _histories()
    with pytest.raises(ValueError, match="discard_cycles must be an integer of at least 5"):
        identify_prescribed_motion_work(
            time_s=histories[0],
            displacement_m=histories[1],
            velocity_mps=histories[2],
            high_fidelity_force_n=histories[3],
            runtime_base_force_n=histories[4],
            motion_frequency_hz=histories[5],
            fluid_density_kgpm3=1.2,
            speed_mps=10.0,
            chord_m=0.25,
            span_m=2.0,
            discard_cycles=4,
        )

    short = _histories(cycles=12)
    with pytest.raises(ValueError, match="at least 8 complete cycles"):
        identify_prescribed_motion_work(
            time_s=short[0],
            displacement_m=short[1],
            velocity_mps=short[2],
            high_fidelity_force_n=short[3],
            runtime_base_force_n=short[4],
            motion_frequency_hz=short[5],
            fluid_density_kgpm3=1.2,
            speed_mps=10.0,
            chord_m=0.25,
            span_m=2.0,
        )

    under_sampled = _histories(samples_per_cycle=99)
    with pytest.raises(ValueError, match="at least 100 samples"):
        identify_prescribed_motion_work(
            time_s=under_sampled[0],
            displacement_m=under_sampled[1],
            velocity_mps=under_sampled[2],
            high_fidelity_force_n=under_sampled[3],
            runtime_base_force_n=under_sampled[4],
            motion_frequency_hz=under_sampled[5],
            fluid_density_kgpm3=1.2,
            speed_mps=10.0,
            chord_m=0.25,
            span_m=2.0,
        )


def test_complete_identified_records_assemble_without_axis_transposition():
    records = []
    for incidence in (80.0, 100.0):
        for reynolds in (1.0e5, 2.0e5):
            for k in (0.1,):
                for amplitude in (0.0, 0.1):
                    for direction in (0.0, 90.0):
                        records.append(
                            {
                                "incidence_deg": incidence,
                                "reynolds_number": reynolds,
                                "reduced_frequency_k": k,
                                "amplitude_ratio": amplitude,
                                "direction_deg_mod_180": direction,
                                "entrainment_fraction": 0.0 if amplitude == 0.0 else 0.5,
                                "reference_available_force_coefficient": 0.2,
                                "signed_power_coefficient": (
                                    0.0
                                    if amplitude == 0.0
                                    else 1.0e-3
                                    * (incidence + reynolds / 1.0e5 + direction)
                                ),
                                "power_coefficient_standard_uncertainty": 0.0,
                            }
                        )
    payload = assemble_airfoil_work_map_payload(
        records=records,
        profile="NACA0018",
        sources=(
            {
                "role": "prescribed_motion_high_fidelity",
                "label": "paired CFD",
                "sha256": "1" * 64,
            },
            {
                "role": "exact_runtime_base",
                "label": "paired QBlade",
                "sha256": "2" * 64,
            },
        ),
    )
    model = AirfoilForceMotionWorkMap(payload)
    query = model.query(
        incidence_deg=100.0,
        reynolds_number=2.0e5,
        reduced_frequency_k=0.1,
        amplitude_ratio=0.1,
        direction_deg=90.0,
        runtime_available_force_coefficient=0.2,
    )
    assert query.signed_power_coefficient == pytest.approx(0.192)

    missing = copy.deepcopy(records)
    missing.pop()
    with pytest.raises(ValueError, match="complete rectangular"):
        assemble_airfoil_work_map_payload(
            records=missing,
            profile="NACA0018",
            sources=payload["sources"],
        )
