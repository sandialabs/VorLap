#!/usr/bin/env python3
"""Diagnose periodicity and boundedness in the completed uniform H-VAWT pilot.

The original pilot compared the maximum response-amplitude estimate in two
300-second halves.  That useful record-sensitivity diagnostic cannot distinguish
secular growth from bounded beating or multi-frequency modulation.  This script
therefore performs a descriptive, revolution-resolved follow-up on the seven
predeclared active complete-hybrid states.  It does not retroactively replace the
sealed v11 acceptance policy or claim stationary/turbulent response validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import signal, stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGN = REPO_ROOT / "run_headless/runs/hvawt_unified_uniform_20case_v5"
SCHEMA = "vorlap-hvawt-uniform-periodicity-boundedness-v1"
EXPECTED_ANALYSIS_SCHEMA = "vorlap-hvawt-response-pilot-analysis-v11"
EXPECTED_CAMPAIGN_KIND = "full_660s_resonance_sweep"
COMPLETE_SCENARIO = "native_qblade_aero_plus_geometry_blended_airfoil_hybrid"
MODES = (6, 12)
REVOLUTION_DURATION_S = 60.0
ASSESSMENT_START_S = 60.0
ASSESSMENT_END_S = 660.0
REVOLUTION_COUNT = 10
LATE_REVOLUTION_COUNT = 5


class PeriodicityError(RuntimeError):
    """Raised when the evidence cannot support the requested diagnosis."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    resolved = path.resolve()
    shown = resolved
    if relative_to is not None:
        try:
            shown = resolved.relative_to(relative_to.resolve())
        except ValueError:
            pass
    return {
        "path": str(shown),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PeriodicityError(f"Cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise PeriodicityError(f"{label} must be a JSON object: {path}")
    return payload


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _verify_fingerprint(payload: Mapping[str, Any], *, label: str) -> None:
    recorded = payload.get("fingerprint")
    body = dict(payload)
    body.pop("fingerprint", None)
    observed = _fingerprint(body)
    if recorded != observed:
        raise PeriodicityError(
            f"{label} fingerprint mismatch: recorded {recorded}, computed {observed}."
        )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _normalized_trend(values: Sequence[float]) -> dict[str, float | int]:
    """Return descriptive scale-normalized trends for a short sequence."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 3 or not np.all(np.isfinite(array)):
        raise PeriodicityError("Trend input must contain at least three finite values.")
    scale = float(np.mean(np.abs(array)))
    if scale <= 0.0:
        raise PeriodicityError("Trend input has zero scale.")
    fit = stats.linregress(np.arange(array.size, dtype=float), array)
    theil = stats.theilslopes(array, np.arange(array.size, dtype=float))
    normalized_slope = float(fit.slope / scale)
    consecutive = np.diff(array) > 0.0
    longest = 0
    current = 0
    for rising in consecutive:
        current = current + 1 if bool(rising) else 0
        longest = max(longest, current)
    return {
        "normalized_slope_per_revolution": normalized_slope,
        "normalized_theil_sen_slope_per_revolution": float(theil.slope / scale),
        "longest_consecutive_increase_count": int(longest),
    }


def _cycle_recurrence(
    velocity: np.ndarray,
    *,
    timestep_s: float,
    natural_frequency_hz: float,
) -> dict[str, float | int]:
    """Measure recurrence at one observed structural-response period."""

    values = np.asarray(velocity, dtype=float)
    if values.ndim != 1 or values.size < 100 or not np.all(np.isfinite(values)):
        raise PeriodicityError("Velocity history is too short or nonfinite.")
    if timestep_s <= 0.0 or natural_frequency_hz <= 0.0:
        raise PeriodicityError("Timestep and natural frequency must be positive.")
    values = values - float(np.mean(values))
    frequencies, density = signal.periodogram(
        values, fs=1.0 / timestep_s, window="hann", detrend="constant"
    )
    total_energy = float(np.sum(density))
    if not math.isfinite(total_energy) or total_energy <= 0.0:
        raise PeriodicityError("Velocity periodogram has no finite positive energy.")
    dominant_index = int(np.argmax(density[1:])) + 1
    dominant_frequency_hz = float(frequencies[dominant_index])
    if dominant_frequency_hz <= 0.0:
        raise PeriodicityError("Raw modal velocity has no positive dominant frequency.")
    lag_samples = int(round(1.0 / (dominant_frequency_hz * timestep_s)))
    if lag_samples < 1 or lag_samples >= values.size:
        raise PeriodicityError("Observed response period is outside the history.")
    correlation = float(
        np.corrcoef(values[:-lag_samples], values[lag_samples:])[0, 1]
    )

    def fraction(relative_band: float) -> float:
        mask = (
            frequencies >= (1.0 - relative_band) * dominant_frequency_hz
        ) & (frequencies <= (1.0 + relative_band) * dominant_frequency_hz)
        return float(np.sum(density[mask]) / total_energy)

    structural_band = (
        frequencies >= 0.9 * natural_frequency_hz
    ) & (frequencies <= 1.1 * natural_frequency_hz)

    return {
        "dominant_frequency_hz": float(dominant_frequency_hz),
        "natural_frequency_hz": float(natural_frequency_hz),
        "dominant_frequency_over_natural_frequency": float(
            dominant_frequency_hz / natural_frequency_hz
        ),
        "dominant_frequency_relative_error": float(
            abs(dominant_frequency_hz / natural_frequency_hz - 1.0)
        ),
        "one_period_lag_samples": lag_samples,
        "one_period_lag_s": float(lag_samples * timestep_s),
        "one_period_autocorrelation": correlation,
        "spectral_energy_within_2_percent": fraction(0.02),
        "spectral_energy_within_10_percent": fraction(0.10),
        "spectral_energy_within_10_percent_of_natural_frequency": float(
            np.sum(density[structural_band]) / total_energy
        ),
    }


def _revolution_metrics(
    time_s: np.ndarray,
    amplitude_m: np.ndarray,
    velocity_mps: np.ndarray,
) -> list[dict[str, float | int]]:
    if not (
        time_s.shape == amplitude_m.shape == velocity_mps.shape
        and time_s.ndim == 1
        and np.all(np.isfinite(time_s))
        and np.all(np.isfinite(amplitude_m))
        and np.all(np.isfinite(velocity_mps))
    ):
        raise PeriodicityError("History arrays are mismatched or nonfinite.")
    if np.any(amplitude_m < 0.0):
        raise PeriodicityError("Amplitude estimate contains negative values.")
    indices = np.floor(
        (time_s - ASSESSMENT_START_S + 1.0e-9) / REVOLUTION_DURATION_S
    ).astype(int)
    rows: list[dict[str, float | int]] = []
    expected_samples: int | None = None
    for revolution in range(REVOLUTION_COUNT):
        selected = indices == revolution
        sample_count = int(np.count_nonzero(selected))
        if sample_count == 0:
            raise PeriodicityError(f"Revolution {revolution + 1} has no samples.")
        if expected_samples is None:
            expected_samples = sample_count
        elif sample_count != expected_samples:
            raise PeriodicityError("Retained revolutions do not have equal sample counts.")
        amplitude = amplitude_m[selected]
        velocity = velocity_mps[selected]
        rows.append(
            {
                "revolution": revolution + 1,
                "start_time_s": ASSESSMENT_START_S
                + revolution * REVOLUTION_DURATION_S,
                "end_time_s": ASSESSMENT_START_S
                + (revolution + 1) * REVOLUTION_DURATION_S,
                "sample_count": sample_count,
                "amplitude_mean_m": float(np.mean(amplitude)),
                "amplitude_rms_m": float(np.sqrt(np.mean(amplitude**2))),
                "amplitude_peak_m": float(np.max(amplitude)),
                "raw_modal_velocity_rms_mps": float(np.sqrt(np.mean(velocity**2))),
            }
        )
    return rows


def _active_states(spec: Mapping[str, Any]) -> set[tuple[float, int]]:
    pilot = spec.get("pilot")
    targets = (
        pilot.get("predicted_amplitude_target_by_selected_wind_and_mode_m")
        if isinstance(pilot, Mapping)
        else None
    )
    if not isinstance(targets, list):
        raise PeriodicityError("Campaign specification lacks selected-wind targets.")
    active: set[tuple[float, int]] = set()
    for row in targets:
        if not isinstance(row, Mapping):
            raise PeriodicityError("Selected-wind target is not an object.")
        wind = float(row["wind_speed_mps"])
        for mode in MODES:
            if float(row[f"mode{mode}_equilibrium_amplitude_m"]) > 0.0:
                active.add((wind, mode))
    if len(active) != 7:
        raise PeriodicityError(f"Expected seven predeclared active states, found {len(active)}.")
    return active


def analyze(
    campaign_root: Path,
    *,
    analysis_path: Path | None = None,
    output_path: Path | None = None,
    revolution_csv_path: Path | None = None,
) -> dict[str, Any]:
    root = campaign_root.expanduser().resolve()
    manifest_path = root / "campaign_manifest.json"
    if analysis_path is None:
        local_replay = root / "pilot_analysis_local_replay.json"
        analysis_path = local_replay if local_replay.is_file() else root / "pilot_analysis.json"
    analysis_path = analysis_path.expanduser().resolve()
    output_path = (
        root / "periodicity_analysis.json"
        if output_path is None
        else output_path.expanduser().resolve()
    )
    revolution_csv_path = (
        root / "periodicity_revolutions.csv"
        if revolution_csv_path is None
        else revolution_csv_path.expanduser().resolve()
    )

    manifest = _load_json(manifest_path, label="campaign manifest")
    analysis = _load_json(analysis_path, label="pilot analysis")
    _verify_fingerprint(manifest, label="Campaign manifest")
    _verify_fingerprint(analysis, label="Pilot analysis")
    matrix = manifest.get("matrix")
    if (
        not isinstance(matrix, Mapping)
        or manifest.get("case_count") != 20
        or matrix.get("campaign_kind") != EXPECTED_CAMPAIGN_KIND
        or float(matrix.get("rpm", 0.0)) != 1.0
        or float(matrix.get("duration_s", 0.0)) != ASSESSMENT_END_S
        or float(matrix.get("assessment_tail_duration_s", 0.0))
        != ASSESSMENT_END_S - ASSESSMENT_START_S
        or analysis.get("schema") != EXPECTED_ANALYSIS_SCHEMA
        or analysis.get("status") != "pilot_failed_complete_hybrid_screen"
        or analysis.get("campaign_fingerprint") != manifest.get("fingerprint")
        or analysis.get("expected_case_count") != 20
        or analysis.get("complete_case_count") != 20
        or analysis.get("passed_case_count") != 18
        or analysis.get("full_rerun_authorized") is not False
    ):
        raise PeriodicityError("Evidence is not the completed 20-case uniform campaign.")

    spec_identity = manifest.get("inputs", {}).get("campaign_spec")
    if not isinstance(spec_identity, Mapping):
        raise PeriodicityError("Campaign manifest lacks its specification identity.")
    spec_path = root / str(spec_identity["path"])
    if _identity(spec_path)["sha256"] != spec_identity.get("sha256"):
        raise PeriodicityError("Campaign specification identity changed.")
    spec = _load_json(spec_path, label="campaign specification")
    active = _active_states(spec)
    rayleigh_identity = manifest.get("inputs", {}).get("rayleigh_analysis")
    if not isinstance(rayleigh_identity, Mapping):
        raise PeriodicityError("Campaign manifest lacks its Rayleigh-analysis identity.")
    rayleigh_path = root / str(rayleigh_identity["path"])
    if _identity(rayleigh_path)["sha256"] != rayleigh_identity.get("sha256"):
        raise PeriodicityError("Rayleigh-analysis identity changed.")
    rayleigh = _load_json(rayleigh_path, label="Rayleigh analysis")
    natural_frequencies = {
        int(row["tracked_mode"]): float(row["frequency_hz"])
        for row in rayleigh.get("modes", [])
        if isinstance(row, Mapping)
    }
    if set(natural_frequencies) != set(MODES):
        raise PeriodicityError("Rayleigh analysis lacks the two tracked modes.")

    cases = analysis.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 20
        or any(
            not isinstance(case, Mapping)
            or case.get("status") != "complete"
            or case.get("execution_receipt_pass") is not True
            for case in cases
        )
    ):
        raise PeriodicityError("Pilot analysis lacks 20 case records.")
    case_keys = [
        (float(case["wind_speed_mps"]), str(case["scenario"])) for case in cases
    ]
    if len(set(case_keys)) != len(case_keys):
        raise PeriodicityError("Pilot analysis contains duplicate wind/scenario cases.")
    complete_cases = {
        (float(case["wind_speed_mps"]), str(case["scenario"])): case
        for case in cases
        if isinstance(case, Mapping)
    }
    states: list[dict[str, Any]] = []
    revolution_rows: list[dict[str, object]] = []
    for wind, mode in sorted(active):
        case = complete_cases.get((wind, COMPLETE_SCENARIO))
        if not isinstance(case, Mapping) or case.get("status") != "complete":
            raise PeriodicityError(f"Active state U={wind:g}, mode {mode} is incomplete.")
        history_identity = case.get("history")
        if not isinstance(history_identity, Mapping):
            raise PeriodicityError(f"Active state U={wind:g}, mode {mode} lacks history identity.")
        history_path = root / "results" / str(case["case_id"]) / Path(
            str(history_identity["path"])
        ).name
        observed_history = _identity(history_path, relative_to=REPO_ROOT)
        if (
            observed_history["sha256"] != history_identity.get("sha256")
            or observed_history["size_bytes"] != history_identity.get("size_bytes")
        ):
            raise PeriodicityError(f"History identity changed for {case['case_id']}.")

        amplitude_column = f"response_mode{mode}_amplitude_estimate_m"
        # Use the real modal-coordinate extraction before the explicit bandpass.
        # A periodicity check on filtered_qdot would partly assume its conclusion.
        velocity_column = f"response_mode{mode}_corrected_raw_qdot_mps"
        frame = pd.read_csv(
            history_path,
            usecols=["time_s", amplitude_column, velocity_column],
        )
        time_s = frame["time_s"].to_numpy(dtype=float)
        amplitude = frame[amplitude_column].to_numpy(dtype=float)
        velocity = frame[velocity_column].to_numpy(dtype=float)
        timestep_s = float(case["timestep_s"])
        if (
            time_s.size != 88_000
            or not np.allclose(np.diff(time_s), timestep_s, rtol=0.0, atol=1.0e-9)
            or not math.isclose(time_s[0], 0.0, rel_tol=0.0, abs_tol=1.0e-9)
        ):
            raise PeriodicityError(f"History time grid changed for {case['case_id']}.")
        retained = (time_s >= ASSESSMENT_START_S - 1.0e-9) & (
            time_s < ASSESSMENT_END_S - 1.0e-9
        )
        retained_time = time_s[retained]
        retained_amplitude = amplitude[retained]
        retained_velocity = velocity[retained]
        revolutions = _revolution_metrics(
            retained_time, retained_amplitude, retained_velocity
        )
        late = revolutions[-LATE_REVOLUTION_COUNT:]
        late_samples = retained_time >= (
            ASSESSMENT_END_S - LATE_REVOLUTION_COUNT * REVOLUTION_DURATION_S - 1.0e-9
        )
        recurrence = _cycle_recurrence(
            retained_velocity[late_samples],
            timestep_s=timestep_s,
            natural_frequency_hz=natural_frequencies[mode],
        )
        mean_values = [float(row["amplitude_mean_m"]) for row in revolutions]
        peak_values = [float(row["amplitude_peak_m"]) for row in revolutions]
        mean_trend = _normalized_trend(
            [float(row["amplitude_mean_m"]) for row in late]
        )
        peak_trend = _normalized_trend(
            [float(row["amplitude_peak_m"]) for row in late]
        )
        velocity_rms_trend = _normalized_trend(
            [float(row["raw_modal_velocity_rms_mps"]) for row in late]
        )
        state = {
            "case_id": case["case_id"],
            "wind_speed_mps": wind,
            "mode": mode,
            "history": observed_history,
            "revolutions": revolutions,
            "late_five_mean_amplitude_trend": mean_trend,
            "late_five_peak_amplitude_trend": peak_trend,
            "late_five_raw_modal_velocity_rms_trend": velocity_rms_trend,
            "structural_cycle_recurrence": recurrence,
            "final_to_first_revolution_mean_amplitude_ratio": float(
                mean_values[-1] / mean_values[0]
            ),
            "final_to_record_peak_amplitude_ratio": float(
                peak_values[-1] / max(peak_values)
            ),
            "record_peak_revolution": int(np.argmax(peak_values)) + 1,
        }
        states.append(state)
        for row in revolutions:
            revolution_rows.append(
                {
                    "case_id": case["case_id"],
                    "wind_speed_mps": wind,
                    "mode": mode,
                    **row,
                }
            )

    final_record_peaks = [
        state for state in states if state["record_peak_revolution"] == REVOLUTION_COUNT
    ]
    autocorrelations = [
        float(state["structural_cycle_recurrence"]["one_period_autocorrelation"])
        for state in states
    ]
    narrow_10 = [
        float(
            state["structural_cycle_recurrence"][
                "spectral_energy_within_10_percent_of_natural_frequency"
            ]
        )
        for state in states
    ]
    frequency_errors = [
        float(
            state["structural_cycle_recurrence"][
                "dominant_frequency_relative_error"
            ]
        )
        for state in states
    ]
    mean_slopes = [
        float(
            state["late_five_mean_amplitude_trend"][
                "normalized_slope_per_revolution"
            ]
        )
        for state in states
    ]
    peak_slopes = [
        float(
            state["late_five_peak_amplitude_trend"][
                "normalized_slope_per_revolution"
            ]
        )
        for state in states
    ]
    velocity_rms_slopes = [
        float(
            state["late_five_raw_modal_velocity_rms_trend"][
                "normalized_slope_per_revolution"
            ]
        )
        for state in states
    ]
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=list(revolution_rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(revolution_rows)
    _atomic_text(revolution_csv_path, stream.getvalue())

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed_descriptive_revolution_diagnostics",
        "campaign_fingerprint": manifest["fingerprint"],
        "campaign_manifest": _identity(manifest_path, relative_to=REPO_ROOT),
        "pilot_analysis": _identity(analysis_path, relative_to=REPO_ROOT),
        "campaign_specification": _identity(spec_path, relative_to=REPO_ROOT),
        "rayleigh_analysis": _identity(rayleigh_path, relative_to=REPO_ROOT),
        "evaluator": _identity(Path(__file__), relative_to=REPO_ROOT),
        "revolution_csv": _identity(revolution_csv_path, relative_to=REPO_ROOT),
        "method": {
            "inflow": "uniform mean inflow; not TurbSim turbulence",
            "rotor_speed_rpm": 1.0,
            "assessment_interval_s": [ASSESSMENT_START_S, ASSESSMENT_END_S],
            "revolution_duration_s": REVOLUTION_DURATION_S,
            "revolution_count": REVOLUTION_COUNT,
            "late_trend_revolution_count": LATE_REVOLUTION_COUNT,
            "trend_interpretation": "OLS and Theil-Sen slopes are descriptive post-hoc diagnostics over five serially correlated revolutions; they are not hypothesis tests or replacement acceptance gates",
            "cycle_recurrence_signal": "corrected raw modal velocity before explicit bandpass, over the final five rotor revolutions",
        },
        "active_state_count": len(states),
        "active_states": states,
        "summary": {
            "final_revolution_record_peak_count": len(final_record_peaks),
            "late_mean_normalized_slope_min_per_revolution": min(mean_slopes),
            "late_mean_normalized_slope_max_per_revolution": max(mean_slopes),
            "late_peak_normalized_slope_min_per_revolution": min(peak_slopes),
            "late_peak_normalized_slope_max_per_revolution": max(peak_slopes),
            "late_mean_longest_consecutive_increase_count_max": max(
                int(
                    state["late_five_mean_amplitude_trend"][
                        "longest_consecutive_increase_count"
                    ]
                )
                for state in states
            ),
            "late_peak_longest_consecutive_increase_count_max": max(
                int(
                    state["late_five_peak_amplitude_trend"][
                        "longest_consecutive_increase_count"
                    ]
                )
                for state in states
            ),
            "late_raw_velocity_rms_normalized_slope_min_per_revolution": min(
                velocity_rms_slopes
            ),
            "late_raw_velocity_rms_normalized_slope_max_per_revolution": max(
                velocity_rms_slopes
            ),
            "late_raw_velocity_rms_longest_consecutive_increase_count_max": max(
                int(
                    state["late_five_raw_modal_velocity_rms_trend"][
                        "longest_consecutive_increase_count"
                    ]
                )
                for state in states
            ),
            "structural_cycle_autocorrelation_min": min(autocorrelations),
            "structural_cycle_autocorrelation_max": max(autocorrelations),
            "spectral_energy_within_10_percent_min": min(narrow_10),
            "spectral_energy_within_10_percent_max": max(narrow_10),
            "raw_dominant_frequency_error_min": min(frequency_errors),
            "raw_dominant_frequency_error_max": max(frequency_errors),
        },
        "interpretation": (
            "The seven active uniform-inflow responses are narrow-band and recur at "
            "their dominant raw modal-velocity period. Their revolution envelopes are "
            "modulated rather than exactly rotor-periodic. Observed late-five-revolution "
            "mean-amplitude, peak-amplitude, and raw-velocity-RMS slopes are reported "
            "without a post-hoc pass/fail threshold; no record peak occurs in the final "
            "revolution."
        ),
        "claim_boundary": (
            "This completed post-hoc numerical diagnostic can inform an operator decision "
            "to attempt the turbulent checkpoint. It does not establish stationary "
            "equilibrium, turbulent-load convergence, physical H-VAWT validation, fatigue "
            "life, or certification loads."
        ),
    }
    report["fingerprint"] = _fingerprint(report)
    _atomic_json(output_path, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--analysis", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--revolution-csv", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze(
        args.campaign_root,
        analysis_path=args.analysis,
        output_path=args.output,
        revolution_csv_path=args.revolution_csv,
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "fingerprint": report["fingerprint"],
                **report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
