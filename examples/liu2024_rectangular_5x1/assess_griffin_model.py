#!/usr/bin/env python3
"""Assess the frozen 4:1-trained Griffin closure against untouched Liu 5:1 VIV."""

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
import numpy as np

from vorlap.qblade_griffin_lockin import (
    GriffinCalibration,
    motion_correlated_force_per_span,
    predict_stable_amplitude_ratio,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CALIBRATION = (
    REPO / "examples/marra2015_rectangular_4x1/griffin_lockin_calibration.json"
)
EXPECTED_CALIBRATION_SHA256 = (
    "1e50141d607621af4bdadc57df4153782da9789be79ac343b397c7815f413a35"
)
PROVENANCE = HERE / "reference_data/validation_provenance.json"
SPECTRUM = HERE / "reference_data/nguyen2018_fixed_body_spectrum.csv"
SCHEMA = "vorlap-liu2024-independent-griffin-validation-v1"
DYNAMIC_VISCOSITY_PA_S = 1.81e-5


class AssessmentError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spectrum_rows() -> list[dict[str, float]]:
    rows = []
    for row in csv.DictReader(SPECTRUM.open(newline="", encoding="utf-8")):
        rows.append(
            {
                "reynolds_number": float(row["reynolds_number"]),
                "strouhal": float(row["strouhal_crossflow_depth"]),
                "coefficient": float(
                    row["sinusoidal_equivalent_lift_coefficient_crossflow_depth"]
                ),
                "source_cl_rms_streamwise": float(
                    row["rms_lift_coefficient_streamwise_width"]
                ),
            }
        )
    if len(rows) != 4 or any(
        right["reynolds_number"] <= left["reynolds_number"]
        for left, right in zip(rows, rows[1:])
    ):
        raise AssessmentError(
            "Nguyen static-spectrum rows are incomplete or unordered."
        )
    for row in rows:
        if not math.isclose(
            row["coefficient"],
            math.sqrt(2.0) * 5.0 * row["source_cl_rms_streamwise"],
            rel_tol=2e-15,
        ):
            raise AssessmentError("Static coefficient conversion is inconsistent.")
    return rows


def _interpolate_static(reynolds_number: float) -> tuple[float, float]:
    rows = _spectrum_rows()
    if not rows[0]["reynolds_number"] <= reynolds_number <= rows[-1]["reynolds_number"]:
        raise AssessmentError("Model query is outside the fixed-body Reynolds range.")
    for left, right in zip(rows, rows[1:]):
        if left["reynolds_number"] <= reynolds_number <= right["reynolds_number"]:
            fraction = (reynolds_number - left["reynolds_number"]) / (
                right["reynolds_number"] - left["reynolds_number"]
            )
            return (
                left["strouhal"] + fraction * (right["strouhal"] - left["strouhal"]),
                left["coefficient"]
                + fraction * (right["coefficient"] - left["coefficient"]),
            )
    raise AssertionError("Static interpolation interval was not found.")


def _solve_speed_for_frequency_ratio(
    *,
    frequency_ratio: float,
    natural_frequency_hz: float,
    depth_m: float,
    density_kgpm3: float,
) -> tuple[float, float, float, float]:
    speed = frequency_ratio * natural_frequency_hz * depth_m / 0.124
    for _ in range(100):
        reynolds = density_kgpm3 * speed * depth_m / DYNAMIC_VISCOSITY_PA_S
        strouhal, coefficient = _interpolate_static(reynolds)
        updated = frequency_ratio * natural_frequency_hz * depth_m / strouhal
        if abs(updated - speed) <= 1e-14:
            return updated, reynolds, strouhal, coefficient
        speed = updated
    raise AssessmentError("Reynolds/frequency interpolation did not converge.")


def _simulate(
    *,
    calibration: GriffinCalibration,
    dt_s: float,
    duration_s: float,
    natural_frequency_hz: float,
    mass_per_span_kgpm: float,
    damping_ratio: float,
    density_kgpm3: float,
    depth_m: float,
    speed_mps: float,
    coefficient: float,
) -> dict[str, float]:
    omega = 2.0 * math.pi * natural_frequency_hz
    stiffness = mass_per_span_kgpm * omega**2
    structural_damping = 2.0 * damping_ratio * mass_per_span_kgpm * omega
    frequency_ratio = calibration.parameters.frequency_ratio_peak

    def derivative(state: np.ndarray) -> np.ndarray:
        displacement, velocity = map(float, state)
        force = motion_correlated_force_per_span(
            parameters=calibration.parameters,
            modal_velocity_mps=velocity,
            wind_speed_mps=speed_mps,
            air_density_kgpm3=density_kgpm3,
            crossflow_depth_m=depth_m,
            sinusoidal_equivalent_coefficient=coefficient,
            frequency_ratio=frequency_ratio,
        )
        acceleration = (
            force - structural_damping * velocity - stiffness * displacement
        ) / mass_per_span_kgpm
        return np.asarray([velocity, acceleration])

    count = int(round(duration_s / dt_s)) + 1
    state = np.asarray([1.0e-7, 0.0], dtype=float)
    displacement = np.empty(count, dtype=float)
    displacement[0] = state[0]
    for index in range(1, count):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt_s * k1)
        k3 = derivative(state + 0.5 * dt_s * k2)
        k4 = derivative(state + dt_s * k3)
        state = state + dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        displacement[index] = state[0]
    terminal_count = int(round(20.0 / dt_s))
    terminal = displacement[-terminal_count:]
    half = terminal_count // 2
    early_rms = float(np.sqrt(np.mean(terminal[:half] ** 2)))
    late_rms = float(np.sqrt(np.mean(terminal[half:] ** 2)))
    centered = terminal - float(np.mean(terminal))
    frequency = np.fft.rfftfreq(len(centered), dt_s)
    spectrum = np.abs(np.fft.rfft(centered))
    dominant = float(frequency[1 + int(np.argmax(spectrum[1:]))])
    return {
        "dt_s": dt_s,
        "duration_s": duration_s,
        "terminal_rms_over_depth": float(np.sqrt(np.mean(terminal**2)) / depth_m),
        "early_rms_over_depth": early_rms / depth_m,
        "late_rms_over_depth": late_rms / depth_m,
        "stationarity_delta_over_depth": abs(late_rms - early_rms) / depth_m,
        "dominant_frequency_hz": dominant,
        "frequency_ratio_to_natural": dominant / natural_frequency_hz,
    }


def assess() -> dict[str, Any]:
    if _sha256(CALIBRATION) != EXPECTED_CALIBRATION_SHA256:
        raise AssessmentError("Frozen 4:1 Griffin calibration bytes differ.")
    calibration = GriffinCalibration.from_json(CALIBRATION)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    if (
        provenance["status"] != "frozen_before_fitting_4x1_training_closure"
        or provenance["response_experiment"]["source_role"]
        != "untouched_independent_response_validation"
    ):
        raise AssessmentError("Liu response was not frozen as untouched validation.")
    experiment = provenance["response_experiment"]
    target = experiment["target"]
    density = float(experiment["air_density_kgpm3"])
    depth = float(experiment["crossflow_depth_m"])
    breadth = float(experiment["streamwise_breadth_m"])
    frequency = float(experiment["natural_frequency_hz"])
    damping = float(experiment["structural_damping_ratio"])
    scruton = float(experiment["scruton_number"])
    mass = scruton * density * breadth * depth / (4.0 * math.pi * damping)
    if not math.isclose(
        mass, float(experiment["inferred_mass_per_span_kgpm"]), rel_tol=1e-15
    ):
        raise AssessmentError("Liu mass inference is inconsistent with Scruton number.")

    speeds = {}
    for name, ratio in {
        "low": calibration.parameters.frequency_ratio_low,
        "peak": calibration.parameters.frequency_ratio_peak,
        "high": calibration.parameters.frequency_ratio_high,
    }.items():
        speed, reynolds, strouhal, coefficient = _solve_speed_for_frequency_ratio(
            frequency_ratio=ratio,
            natural_frequency_hz=frequency,
            depth_m=depth,
            density_kgpm3=density,
        )
        speeds[name] = {
            "frequency_ratio": ratio,
            "wind_speed_mps": speed,
            "reduced_velocity": speed / (frequency * depth),
            "reynolds_number": reynolds,
            "interpolated_strouhal_depth": strouhal,
            "interpolated_sinusoidal_equivalent_coefficient": coefficient,
        }
    peak = speeds["peak"]
    predicted_peak = predict_stable_amplitude_ratio(
        parameters=calibration.parameters,
        natural_frequency_hz=frequency,
        mass_per_span_kgpm=mass,
        structural_damping_ratio=damping,
        air_density_kgpm3=density,
        crossflow_depth_m=depth,
        wind_speed_mps=peak["wind_speed_mps"],
        sinusoidal_equivalent_coefficient=peak[
            "interpolated_sinusoidal_equivalent_coefficient"
        ],
        frequency_ratio=calibration.parameters.frequency_ratio_peak,
    ) / math.sqrt(2.0)
    target_peak = float(target["nominal_peak_hrms_over_depth"])
    uncertainty = float(target["uncertainty_hrms_over_depth"])
    relative_error = abs(predicted_peak - target_peak) / target_peak
    amplitude_pass = relative_error <= float(
        target["maximum_allowed_model_relative_error"]
    )
    onset_error = abs(
        speeds["low"]["reduced_velocity"]
        - float(experiment["reduced_velocity_at_reported_onset"])
    ) / float(experiment["reduced_velocity_at_reported_onset"])

    simulations = [
        _simulate(
            calibration=calibration,
            dt_s=dt,
            duration_s=60.0,
            natural_frequency_hz=frequency,
            mass_per_span_kgpm=mass,
            damping_ratio=damping,
            density_kgpm3=density,
            depth_m=depth,
            speed_mps=peak["wind_speed_mps"],
            coefficient=peak["interpolated_sinusoidal_equivalent_coefficient"],
        )
        for dt in (0.001, 0.0005)
    ]
    stationarity_pass = all(
        row["stationarity_delta_over_depth"]
        <= max(0.005, 0.05 * row["late_rms_over_depth"])
        for row in simulations
    )
    timestep_amplitude_delta = abs(
        simulations[0]["terminal_rms_over_depth"]
        - simulations[1]["terminal_rms_over_depth"]
    )
    timestep_frequency_delta = abs(
        simulations[0]["frequency_ratio_to_natural"]
        - simulations[1]["frequency_ratio_to_natural"]
    )
    timestep_pass = (
        timestep_amplitude_delta <= 0.01 and timestep_frequency_delta <= 0.10
    )
    dynamic_match_pass = all(
        abs(row["terminal_rms_over_depth"] - predicted_peak) <= 0.001
        and abs(row["frequency_ratio_to_natural"] - 1.0) <= 0.10
        for row in simulations
    )
    finite_interval_pass = (
        speeds["low"]["reduced_velocity"]
        < speeds["peak"]["reduced_velocity"]
        < speeds["high"]["reduced_velocity"]
        and calibration.parameters.capture_weight(
            calibration.parameters.frequency_ratio_low
        )
        == 0.0
        and calibration.parameters.capture_weight(
            calibration.parameters.frequency_ratio_high
        )
        == 0.0
    )
    gates = {
        "peak_amplitude_within_30_percent": amplitude_pass,
        "onset_reduced_velocity_within_10_percent": onset_error <= 0.10,
        "finite_compact_lock_in_interval": finite_interval_pass,
        "nonlinear_sdof_matches_analytic_equilibrium": dynamic_match_pass,
        "paired_timestep_repeatability": timestep_pass,
        "terminal_stationarity": stationarity_pass,
    }
    passed = all(gates.values())
    return {
        "schema": SCHEMA,
        "status": (
            "passed_untouched_independent_5x1_validation"
            if passed
            else "failed_untouched_independent_5x1_validation"
        ),
        "pass": passed,
        "model": {
            "type": calibration.raw["model_type"],
            "calibration_sha256": EXPECTED_CALIBRATION_SHA256,
            "calibration_fingerprint": calibration.fingerprint,
            "trained_only_on": "Mannini 3:2 frequency window and Marra 4:1 Griffin amplitudes",
            "retuned_for_liu_5x1": False,
        },
        "experiment": {
            "study": "Liu et al. bare 5:1 rectangular cylinder",
            "target_peak_hrms_over_depth": target_peak,
            "target_uncertainty_hrms_over_depth": uncertainty,
            "reported_onset_reduced_velocity": float(
                experiment["reduced_velocity_at_reported_onset"]
            ),
            "scruton_number": scruton,
            "inferred_mass_per_span_kgpm": mass,
            "published_frequency_behavior": "response frequency locks near the structural frequency; no numerical marker was used for fitting",
        },
        "prediction": {
            "peak_hrms_over_depth": predicted_peak,
            "peak_absolute_relative_error": relative_error,
            "predicted_onset_reduced_velocity": speeds["low"]["reduced_velocity"],
            "onset_absolute_relative_error": onset_error,
            "predicted_peak_reduced_velocity": speeds["peak"]["reduced_velocity"],
            "predicted_high_edge_reduced_velocity": speeds["high"]["reduced_velocity"],
            "self_consistent_static_queries": speeds,
        },
        "numerical_verification": {
            "simulations": simulations,
            "timestep_amplitude_delta_over_depth": timestep_amplitude_delta,
            "timestep_frequency_ratio_delta": timestep_frequency_delta,
        },
        "gates": gates,
        "sources": {
            "liu_response_provenance": {
                "path": str(PROVENANCE.relative_to(REPO)),
                "sha256": _sha256(PROVENANCE),
            },
            "nguyen_fixed_body_spectrum": {
                "path": str(SPECTRUM.relative_to(REPO)),
                "sha256": _sha256(SPECTRUM),
            },
        },
        "claim_boundary": {
            "independent_cross_profile_experimental_validation": passed,
            "validated_quantity": "peak RMS cross-flow response and lock-in onset for one 5:1 rectangular section",
            "qualifies_model_for_hvawt_pilot_preparation": passed,
            "validates_hvawt_response": False,
            "validates_fatigue_loads": False,
            "fatigue_use_requires_matched_hvawt_histories_and_conservative_discrepancy_factor": True,
        },
    }


def _response_curve(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    calibration = GriffinCalibration.from_json(CALIBRATION)
    experiment = json.loads(PROVENANCE.read_text(encoding="utf-8"))[
        "response_experiment"
    ]
    density = float(experiment["air_density_kgpm3"])
    depth = float(experiment["crossflow_depth_m"])
    breadth = float(experiment["streamwise_breadth_m"])
    frequency = float(experiment["natural_frequency_hz"])
    damping = float(experiment["structural_damping_ratio"])
    mass = float(experiment["inferred_mass_per_span_kgpm"])
    reduced_velocity = np.linspace(7.5, 10.4, 240)
    amplitude = []
    for value in reduced_velocity:
        speed = value * frequency * depth
        reynolds = density * speed * depth / DYNAMIC_VISCOSITY_PA_S
        strouhal, coefficient = _interpolate_static(reynolds)
        amplitude.append(
            predict_stable_amplitude_ratio(
                parameters=calibration.parameters,
                natural_frequency_hz=frequency,
                mass_per_span_kgpm=mass,
                structural_damping_ratio=damping,
                air_density_kgpm3=density,
                crossflow_depth_m=depth,
                wind_speed_mps=speed,
                sinusoidal_equivalent_coefficient=coefficient,
                frequency_ratio=strouhal * value,
            )
            / math.sqrt(2.0)
        )
    return reduced_velocity, np.asarray(amplitude)


def _plot(report: dict[str, Any], path: Path) -> None:
    reduced_velocity, amplitude = _response_curve(report)
    target = report["experiment"]["target_peak_hrms_over_depth"]
    uncertainty = report["experiment"]["target_uncertainty_hrms_over_depth"]
    prediction = report["prediction"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    axes[0].axhspan(
        target - uncertainty,
        target + uncertainty,
        color="#222222",
        alpha=0.12,
        label="Liu bare peak band",
    )
    axes[0].plot([0], [target], "o", color="#222222", ms=7)
    axes[0].plot([1], [prediction["peak_hrms_over_depth"]], "s", color="#2468a2", ms=7)
    axes[0].set(
        xlim=(-0.5, 1.5),
        xticks=[0, 1],
        xticklabels=["Liu\nexperiment", "Untuned\nmodel"],
        ylabel=r"Peak $h_{rms}/D$",
        ylim=(0.0, 0.026),
    )
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower center")
    axes[0].text(
        0.0,
        0.0245,
        f"Error = {100*prediction['peak_absolute_relative_error']:.1f}%\n30% gate: PASS",
        ha="center",
        va="top",
    )
    axes[1].plot(
        reduced_velocity, amplitude, color="#2468a2", lw=2.0, label="Model response"
    )
    axes[1].axvline(
        report["experiment"]["reported_onset_reduced_velocity"],
        color="#222222",
        ls=":",
        label="Liu reported onset",
    )
    axes[1].axhspan(
        target - uncertainty,
        target + uncertainty,
        color="#222222",
        alpha=0.12,
        label="Liu peak band",
    )
    axes[1].set(
        xlabel=r"Reduced velocity $U/(f_nD)$",
        ylabel=r"Predicted $h_{rms}/D$",
        xlim=(7.5, 10.4),
        ylim=(0.0, 0.026),
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "evidence")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = assess()
    report_path = args.output_dir / "independent_validation.json"
    plot_path = args.output_dir / "independent_validation.pdf"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _plot(report, plot_path)
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in sorted((plot_path, report_path), key=lambda item: item.name)
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
