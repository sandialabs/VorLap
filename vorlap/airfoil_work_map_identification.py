"""Identify signed incremental airfoil work from prescribed-motion histories.

The input force and velocity must be synchronous.  The high-fidelity force is
paired with the exact runtime-base force at the same motion; the difference is
what a supplemental bridge is allowed to reproduce.  Static FFT phase is not
an accepted substitute for either history.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .qblade_airfoil_work_map import SCHEMA, frozen_conventions


_AXES = (
    "incidence_deg",
    "reynolds_number",
    "reduced_frequency_k",
    "amplitude_ratio",
    "direction_deg_mod_180",
)
_FIELDS = (
    "entrainment_fraction",
    "reference_available_force_coefficient",
    "signed_power_coefficient",
    "power_coefficient_standard_uncertainty",
)

# These are evidence-quality floors, not tuning knobs.  They are duplicated in
# the tracked NACA0018 data request and are enforced here as a second line of
# defense for direct callers that bypass the manifest builder.
MINIMUM_DISCARD_CYCLES = 5
MINIMUM_COMPLETE_CYCLES_AFTER_DISCARD = 8
MINIMUM_SAMPLES_PER_CYCLE = 100


def _finite_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size < 4 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional history.")
    return np.array(array, dtype=float, copy=True)


@dataclass(frozen=True)
class PrescribedMotionWorkIdentification:
    motion_frequency_hz: float
    motion_amplitude_m: float
    included_cycle_count: int
    minimum_samples_per_included_cycle: int
    mean_incremental_power_w: float
    standard_uncertainty_power_w: float
    signed_power_coefficient: float
    power_coefficient_standard_uncertainty: float
    incremental_force_velocity_harmonic_n: float
    incremental_force_displacement_harmonic_n: float
    phase_force_relative_to_velocity_rad: float
    cycle_mean_power_w: tuple[float, ...]


def assemble_airfoil_work_map_payload(
    *,
    records: Sequence[Mapping[str, Any]],
    profile: str,
    sources: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Assemble a complete rectangular work-map payload from identified rows.

    The caller must supply every tensor vertex explicitly, including the exact
    zero-amplitude plane.  This intentionally refuses gridding, smoothing,
    extrapolation, and best-of-menu selection inside the evidence assembler.
    """

    if not str(profile).strip():
        raise ValueError("profile must not be empty.")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of mappings.")
    if not records:
        raise ValueError("At least one work-map record is required.")
    normalized: list[dict[str, float]] = []
    expected_keys = set(_AXES) | set(_FIELDS)
    for row in records:
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise ValueError("Every work-map record must have exact axis and field keys.")
        converted = {}
        for key in (*_AXES, *_FIELDS):
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"Work-map record {key} must be finite.")
            converted[key] = value
        normalized.append(converted)

    axes = {
        name: sorted({float(row[name]) for row in normalized}) for name in _AXES
    }
    shape = tuple(len(axes[name]) for name in _AXES)
    if len(normalized) != math.prod(shape):
        raise ValueError("Work-map records do not form one complete rectangular tensor.")
    index = {
        name: {value: position for position, value in enumerate(axes[name])}
        for name in _AXES
    }
    tensors = {name: np.full(shape, np.nan, dtype=float) for name in _FIELDS}
    seen = set()
    for row in normalized:
        location = tuple(index[name][row[name]] for name in _AXES)
        if location in seen:
            raise ValueError("Work-map records contain a duplicate tensor vertex.")
        seen.add(location)
        for name in _FIELDS:
            tensors[name][location] = row[name]
    if len(seen) != math.prod(shape) or any(
        np.any(~np.isfinite(value)) for value in tensors.values()
    ):
        raise ValueError("Work-map records omit at least one tensor vertex.")

    # Iterating the product here is a deliberate final association audit.  It
    # catches future changes that might otherwise silently transpose an axis.
    for location in itertools.product(*(range(length) for length in shape)):
        expected_axis_values = tuple(axes[name][location[i]] for i, name in enumerate(_AXES))
        if not any(
            tuple(row[name] for name in _AXES) == expected_axis_values
            for row in normalized
        ):
            raise ValueError("Work-map axis/record association is incomplete.")

    return {
        "schema": SCHEMA,
        "status": "calibrated",
        "profile": str(profile).strip(),
        "motion": "prescribed_crossflow_translation",
        "axes": axes,
        "fields": {name: tensors[name].tolist() for name in _FIELDS},
        "conventions": frozen_conventions(),
        "sources": [dict(value) for value in sources],
    }


def identify_prescribed_motion_work(
    *,
    time_s: Any,
    displacement_m: Any,
    velocity_mps: Any,
    high_fidelity_force_n: Any,
    runtime_base_force_n: Any,
    motion_frequency_hz: float,
    fluid_density_kgpm3: float,
    speed_mps: float,
    chord_m: float,
    span_m: float,
    discard_cycles: int = MINIMUM_DISCARD_CYCLES,
    minimum_complete_cycles_after_discard: int = MINIMUM_COMPLETE_CYCLES_AFTER_DISCARD,
    minimum_samples_per_cycle: int = MINIMUM_SAMPLES_PER_CYCLE,
) -> PrescribedMotionWorkIdentification:
    """Return paired incremental work metrics over complete motion cycles.

    Cycle bins are defined only by the supplied frequency and time.  No signal
    peak or validation response is used to choose a window.
    """

    time = _finite_vector(time_s, "time_s")
    displacement = _finite_vector(displacement_m, "displacement_m")
    velocity = _finite_vector(velocity_mps, "velocity_mps")
    high_fidelity = _finite_vector(high_fidelity_force_n, "high_fidelity_force_n")
    runtime_base = _finite_vector(runtime_base_force_n, "runtime_base_force_n")
    if not all(
        array.size == time.size
        for array in (displacement, velocity, high_fidelity, runtime_base)
    ):
        raise ValueError("All prescribed-motion histories must have equal length.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be strictly increasing.")
    raw_physical = (
        motion_frequency_hz,
        fluid_density_kgpm3,
        speed_mps,
        chord_m,
        span_m,
    )
    if any(isinstance(value, bool) for value in raw_physical):
        raise TypeError("Physical scalars must be real numbers, not booleans.")
    frequency, density, speed, chord, span = (float(value) for value in raw_physical)
    if not all(math.isfinite(value) for value in (frequency, density, speed, chord, span)):
        raise ValueError("Physical scalars must be finite.")
    if frequency <= 0.0 or density <= 0.0 or speed <= 0.0 or chord <= 0.0 or span <= 0.0:
        raise ValueError("Frequency, density, speed, chord, and span must be positive.")
    if type(discard_cycles) is not int or discard_cycles < MINIMUM_DISCARD_CYCLES:
        raise ValueError(
            f"discard_cycles must be an integer of at least {MINIMUM_DISCARD_CYCLES}."
        )
    if (
        type(minimum_complete_cycles_after_discard) is not int
        or minimum_complete_cycles_after_discard
        < MINIMUM_COMPLETE_CYCLES_AFTER_DISCARD
    ):
        raise ValueError(
            "minimum_complete_cycles_after_discard must be an integer of at least "
            f"{MINIMUM_COMPLETE_CYCLES_AFTER_DISCARD}."
        )
    if (
        type(minimum_samples_per_cycle) is not int
        or minimum_samples_per_cycle < MINIMUM_SAMPLES_PER_CYCLE
    ):
        raise ValueError(
            "minimum_samples_per_cycle must be an integer of at least "
            f"{MINIMUM_SAMPLES_PER_CYCLE}."
        )

    period = 1.0 / frequency
    maximum_complete = int(math.floor((time[-1] - time[0]) / period + 1.0e-10))
    if maximum_complete - discard_cycles < minimum_complete_cycles_after_discard:
        raise ValueError(
            "Prescribed-motion history must contain at least "
            f"{minimum_complete_cycles_after_discard} complete cycles after discarding "
            f"the first {discard_cycles}."
        )
    cycles = []
    cycle_sample_counts = []
    incremental_force = high_fidelity - runtime_base
    for index in range(discard_cycles, maximum_complete):
        start = time[0] + index * period
        stop = start + period
        selected = (time >= start - 1.0e-12) & (time <= stop + 1.0e-12)
        selected_time = time[selected]
        if selected_time.size < 2:
            raise ValueError("Prescribed-motion history omits an included cycle.")
        if selected_time[0] > start + 1.0e-9 or selected_time[-1] < stop - 1.0e-9:
            raise ValueError("Prescribed-motion history does not resolve complete cycles.")
        # Count intervals rather than points because adjacent complete cycles
        # share their boundary sample.  A 100-sample-per-cycle export therefore
        # contains 101 inclusive points in each integration bin.
        sample_count = int(selected_time.size - 1)
        if sample_count < minimum_samples_per_cycle:
            raise ValueError(
                "Each included cycle must contain at least "
                f"{minimum_samples_per_cycle} samples."
            )
        maximum_allowed_step = period / minimum_samples_per_cycle
        if float(np.max(np.diff(selected_time))) > maximum_allowed_step * (1.0 + 1.0e-9):
            raise ValueError(
                "Each included cycle must maintain the requested temporal resolution "
                "throughout the complete cycle."
            )
        cycle_sample_counts.append(sample_count)
        power = np.trapezoid(
            incremental_force[selected] * velocity[selected], selected_time
        ) / period
        cycles.append(float(power))
    if len(cycles) < minimum_complete_cycles_after_discard:
        raise ValueError(
            "At least "
            f"{minimum_complete_cycles_after_discard} complete post-discard cycles are required."
        )

    cycle_power = np.asarray(cycles, dtype=float)
    mean_power = float(np.mean(cycle_power))
    standard_uncertainty = float(np.std(cycle_power, ddof=1) / math.sqrt(cycle_power.size))
    normalization = 0.5 * density * speed**3 * chord * span

    # Fit the declared motion frequency on the same complete-cycle interval.
    start = time[0] + discard_cycles * period
    stop = time[0] + maximum_complete * period
    selected = (time >= start - 1.0e-12) & (time <= stop + 1.0e-12)
    fit_time = time[selected]
    omega = 2.0 * math.pi * frequency
    design = np.column_stack(
        (
            np.ones(fit_time.size),
            fit_time - fit_time[0],
            np.cos(omega * fit_time),
            np.sin(omega * fit_time),
        )
    )
    force_coefficients, *_ = np.linalg.lstsq(
        design, incremental_force[selected], rcond=None
    )
    displacement_coefficients, *_ = np.linalg.lstsq(
        design, displacement[selected], rcond=None
    )
    velocity_coefficients, *_ = np.linalg.lstsq(design, velocity[selected], rcond=None)
    force_velocity_harmonic = float(
        force_coefficients[2] * velocity_coefficients[2]
        + force_coefficients[3] * velocity_coefficients[3]
    ) / max(
        math.hypot(velocity_coefficients[2], velocity_coefficients[3]),
        np.finfo(float).tiny,
    )
    force_displacement_harmonic = float(
        force_coefficients[2] * displacement_coefficients[2]
        + force_coefficients[3] * displacement_coefficients[3]
    ) / max(
        math.hypot(displacement_coefficients[2], displacement_coefficients[3]),
        np.finfo(float).tiny,
    )
    force_phase = math.atan2(force_coefficients[3], force_coefficients[2])
    velocity_phase = math.atan2(velocity_coefficients[3], velocity_coefficients[2])
    relative_phase = math.atan2(
        math.sin(force_phase - velocity_phase),
        math.cos(force_phase - velocity_phase),
    )
    motion_amplitude = float(
        math.hypot(displacement_coefficients[2], displacement_coefficients[3])
    )
    if motion_amplitude <= 0.0:
        raise ValueError("Prescribed displacement has zero fitted motion amplitude.")

    return PrescribedMotionWorkIdentification(
        motion_frequency_hz=frequency,
        motion_amplitude_m=motion_amplitude,
        included_cycle_count=int(cycle_power.size),
        minimum_samples_per_included_cycle=min(cycle_sample_counts),
        mean_incremental_power_w=mean_power,
        standard_uncertainty_power_w=standard_uncertainty,
        signed_power_coefficient=mean_power / normalization,
        power_coefficient_standard_uncertainty=standard_uncertainty / normalization,
        incremental_force_velocity_harmonic_n=force_velocity_harmonic,
        incremental_force_displacement_harmonic_n=force_displacement_harmonic,
        phase_force_relative_to_velocity_rad=relative_phase,
        cycle_mean_power_w=tuple(float(value) for value in cycle_power),
    )
