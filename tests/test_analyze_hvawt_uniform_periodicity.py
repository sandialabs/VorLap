from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_headless/analyze_hvawt_uniform_periodicity.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_hvawt_uniform_periodicity", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
periodicity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(periodicity)


def test_normalized_trend_reports_direction_without_posthoc_significance_gate() -> None:
    increasing = periodicity._normalized_trend([1.0, 1.1, 1.2, 1.3, 1.4])
    modulated = periodicity._normalized_trend([1.0, 1.2, 0.9, 1.1, 1.0])

    assert increasing["normalized_slope_per_revolution"] == pytest.approx(1.0 / 12.0)
    assert increasing["normalized_theil_sen_slope_per_revolution"] == pytest.approx(
        1.0 / 12.0
    )
    assert increasing["longest_consecutive_increase_count"] == 4
    assert modulated["longest_consecutive_increase_count"] == 1
    assert "p_value" not in " ".join(increasing)
    assert all("passed" not in name for name in increasing)


def test_cycle_recurrence_uses_unfiltered_signal_and_frozen_natural_frequency() -> None:
    timestep_s = 0.01
    natural_frequency_hz = 2.5
    time_s = np.arange(0.0, 120.0, timestep_s)
    velocity = np.sin(2.0 * np.pi * natural_frequency_hz * time_s)

    result = periodicity._cycle_recurrence(
        velocity,
        timestep_s=timestep_s,
        natural_frequency_hz=natural_frequency_hz,
    )

    assert result["dominant_frequency_hz"] == pytest.approx(
        natural_frequency_hz, abs=1.0 / 120.0
    )
    assert result["dominant_frequency_relative_error"] < 1.0e-12
    assert result["one_period_autocorrelation"] > 0.999
    assert result["spectral_energy_within_10_percent_of_natural_frequency"] > 0.999


def test_revolution_metrics_preserve_exact_equal_windows() -> None:
    timestep_s = 0.5
    time_s = np.arange(60.0, 660.0, timestep_s)
    amplitude = 0.001 * (1.0 + 0.1 * np.sin(2.0 * np.pi * time_s / 60.0))
    velocity = np.cos(2.0 * np.pi * 2.5 * time_s)

    rows = periodicity._revolution_metrics(time_s, amplitude, velocity)

    assert len(rows) == 10
    assert {row["sample_count"] for row in rows} == {120}
    assert rows[0]["start_time_s"] == 60.0
    assert rows[-1]["end_time_s"] == 660.0
    assert all(row["amplitude_mean_m"] == pytest.approx(0.001) for row in rows)


def test_revolution_metrics_reject_negative_amplitude() -> None:
    time_s = np.arange(60.0, 660.0, 1.0)
    with pytest.raises(periodicity.PeriodicityError, match="negative"):
        periodicity._revolution_metrics(
            time_s,
            -np.ones_like(time_s),
            np.ones_like(time_s),
        )
