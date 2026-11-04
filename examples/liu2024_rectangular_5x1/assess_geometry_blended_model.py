#!/usr/bin/env python3
"""Assess the frozen geometry-blended closure on protected Liu 5:1 response."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.liu2024_rectangular_5x1.assess_griffin_model import (
    _interpolate_static,
    _solve_speed_for_frequency_ratio,
)
from vorlap.qblade_geometry_blended_lockin import (
    GeometryBlendedCalibration,
    motion_correlated_force_per_span,
    predict_stable_amplitude_ratio,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CALIBRATION = REPO / "examples/viv_geometry_blended/geometry_blended_calibration.json"
EXPECTED_CALIBRATION_SHA256 = "f2dee270b43683b99b08bd3fcab10d816920be770ebe9355cd238f1d9ceadd0c"
PROVENANCE = HERE / "reference_data/validation_provenance.json"
SPECTRUM = HERE / "reference_data/nguyen2018_fixed_body_spectrum.csv"
SCHEMA = "vorlap-liu2024-protected-geometry-blended-validation-v1"
DYNAMIC_VISCOSITY_PA_S = 1.81e-5


class AssessmentError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _simulate(
    *,
    calibration: GeometryBlendedCalibration,
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
    damping = 2.0 * damping_ratio * mass_per_span_kgpm * omega
    _, peak_ratio, _ = calibration.parameters.frequency_bounds(5.0)

    def derivative(state: np.ndarray) -> np.ndarray:
        displacement, velocity = map(float, state)
        force = motion_correlated_force_per_span(
            parameters=calibration.parameters,
            modal_velocity_mps=velocity,
            amplitude_ratio=abs(displacement) / depth_m,
            wind_speed_mps=speed_mps,
            air_density_kgpm3=density_kgpm3,
            crossflow_depth_m=depth_m,
            sinusoidal_equivalent_coefficient=coefficient,
            frequency_ratio=peak_ratio,
            breadth_depth_ratio=5.0,
        )
        return np.asarray(
            [velocity, (force - damping * velocity - stiffness * displacement) / mass_per_span_kgpm]
        )

    count = int(round(duration_s / dt_s)) + 1
    state = np.asarray([1.0e-7, 0.0])
    displacement = np.empty(count)
    displacement[0] = state[0]
    for index in range(1, count):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * dt_s * k1)
        k3 = derivative(state + 0.5 * dt_s * k2)
        k4 = derivative(state + dt_s * k3)
        state += dt_s * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        displacement[index] = state[0]
    terminal_count = int(round(20.0 / dt_s))
    terminal = displacement[-terminal_count:]
    half = terminal_count // 2
    early = float(np.sqrt(np.mean(terminal[:half] ** 2))) / depth_m
    late = float(np.sqrt(np.mean(terminal[half:] ** 2))) / depth_m
    centered = terminal - np.mean(terminal)
    frequency = np.fft.rfftfreq(len(centered), dt_s)
    spectrum = np.abs(np.fft.rfft(centered))
    dominant = float(frequency[1 + int(np.argmax(spectrum[1:]))])
    return {
        "dt_s": dt_s,
        "terminal_rms_over_depth": float(np.sqrt(np.mean(terminal**2))) / depth_m,
        "early_rms_over_depth": early,
        "late_rms_over_depth": late,
        "stationarity_delta_over_depth": abs(late - early),
        "dominant_frequency_hz": dominant,
        "frequency_ratio_to_natural": dominant / natural_frequency_hz,
    }


def assess() -> dict[str, Any]:
    if _sha256(CALIBRATION) != EXPECTED_CALIBRATION_SHA256:
        raise AssessmentError("Frozen geometry-blended calibration bytes differ.")
    calibration = GeometryBlendedCalibration.from_json(CALIBRATION)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    experiment = provenance["response_experiment"]
    if (
        provenance["status"] != "frozen_before_fitting_4x1_training_closure"
        or experiment["source_role"] != "untouched_independent_response_validation"
    ):
        raise AssessmentError("Liu protected-validation role differs.")
    target = experiment["target"]
    density = float(experiment["air_density_kgpm3"])
    depth = float(experiment["crossflow_depth_m"])
    breadth = float(experiment["streamwise_breadth_m"])
    frequency = float(experiment["natural_frequency_hz"])
    damping = float(experiment["structural_damping_ratio"])
    scruton = float(experiment["scruton_number"])
    mass = scruton * density * breadth * depth / (4.0 * math.pi * damping)
    low_ratio, peak_ratio, high_ratio = calibration.parameters.frequency_bounds(5.0)
    speeds = {}
    for name, ratio in {"low": low_ratio, "peak": peak_ratio, "high": high_ratio}.items():
        speed, reynolds, st, coefficient = _solve_speed_for_frequency_ratio(
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
            "strouhal": st,
            "sinusoidal_equivalent_coefficient": coefficient,
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
        sinusoidal_equivalent_coefficient=peak["sinusoidal_equivalent_coefficient"],
        frequency_ratio=peak_ratio,
        breadth_depth_ratio=5.0,
    ) / math.sqrt(2.0)
    target_peak = float(target["nominal_peak_hrms_over_depth"])
    peak_error = abs(predicted_peak - target_peak) / target_peak
    onset_target = float(experiment["reduced_velocity_at_reported_onset"])
    onset_error = abs(speeds["low"]["reduced_velocity"] - onset_target) / onset_target
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
            coefficient=peak["sinusoidal_equivalent_coefficient"],
        )
        for dt in (0.001, 0.0005)
    ]
    gates = {
        "peak_amplitude_within_30_percent": peak_error <= 0.30,
        "onset_reduced_velocity_within_10_percent": onset_error <= 0.10,
        "finite_compact_lock_in_interval": speeds["low"]["reduced_velocity"] < speeds["peak"]["reduced_velocity"] < speeds["high"]["reduced_velocity"],
        "paired_timestep_amplitude_within_0p01D": abs(simulations[0]["terminal_rms_over_depth"] - simulations[1]["terminal_rms_over_depth"]) <= 0.01,
        "paired_timestep_frequency_within_10_percent": abs(simulations[0]["frequency_ratio_to_natural"] - simulations[1]["frequency_ratio_to_natural"]) <= 0.10,
        "terminal_stationarity": all(row["stationarity_delta_over_depth"] <= max(0.005, 0.05 * row["late_rms_over_depth"]) for row in simulations),
        "dynamic_sdof_within_0p001D_of_analytic": all(abs(row["terminal_rms_over_depth"] - predicted_peak) <= 0.001 for row in simulations),
    }
    passed = all(gates.values())
    return {
        "schema": SCHEMA,
        "status": "passed_protected_liu_5x1_validation" if passed else "failed_protected_liu_5x1_validation",
        "pass": passed,
        "model": {
            "model_type": calibration.raw["model_type"],
            "calibration_sha256": EXPECTED_CALIBRATION_SHA256,
            "calibration_fingerprint": calibration.fingerprint,
            "trained_on": calibration.raw["training"]["studies"],
            "liu_response_used_for_fitting": False,
        },
        "experiment": {
            "study": "Liu et al. bare 5:1 rectangular cylinder",
            "target_peak_hrms_over_depth": target_peak,
            "reported_onset_reduced_velocity": onset_target,
            "scruton_number": scruton,
            "breadth_depth_ratio": breadth / depth,
        },
        "prediction": {
            "peak_hrms_over_depth": predicted_peak,
            "peak_absolute_relative_error": peak_error,
            "predicted_onset_reduced_velocity": speeds["low"]["reduced_velocity"],
            "onset_absolute_relative_error": onset_error,
            "predicted_peak_reduced_velocity": speeds["peak"]["reduced_velocity"],
            "predicted_high_edge_reduced_velocity": speeds["high"]["reduced_velocity"],
            "static_queries": speeds,
        },
        "numerical_verification": {"simulations": simulations},
        "gates": gates,
        "sources": {
            "liu_response_provenance": {"path": str(PROVENANCE.relative_to(REPO)), "sha256": _sha256(PROVENANCE)},
            "nguyen_fixed_body_spectrum": {"path": str(SPECTRUM.relative_to(REPO)), "sha256": _sha256(SPECTRUM)},
        },
        "claim_boundary": {
            "protected_cross_geometry_experimental_validation": passed,
            "validated_quantity": "one 5:1 rectangular peak RMS response and onset",
            "supports_engineering_hvawt_pilot": passed,
            "validates_hvawt_or_hawt_response": False,
            "validates_fatigue": False,
            "fatigue_use_requires_matched_native_hybrid_histories_and_conservative_discrepancy_factor": True,
        },
    }


def _curve(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    calibration = GeometryBlendedCalibration.from_json(CALIBRATION)
    experiment = json.loads(PROVENANCE.read_text(encoding="utf-8"))["response_experiment"]
    density = float(experiment["air_density_kgpm3"])
    depth = float(experiment["crossflow_depth_m"])
    frequency = float(experiment["natural_frequency_hz"])
    damping = float(experiment["structural_damping_ratio"])
    mass = float(experiment["inferred_mass_per_span_kgpm"])
    reduced_velocity = np.linspace(7.5, 10.4, 240)
    amplitude = []
    for value in reduced_velocity:
        speed = value * frequency * depth
        reynolds = density * speed * depth / DYNAMIC_VISCOSITY_PA_S
        st, coefficient = _interpolate_static(reynolds)
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
                frequency_ratio=st * value,
                breadth_depth_ratio=5.0,
            ) / math.sqrt(2.0)
        )
    return reduced_velocity, np.asarray(amplitude)


def _plot(report: dict[str, Any], path: Path) -> None:
    u, amplitude = _curve(report)
    target = report["experiment"]["target_peak_hrms_over_depth"]
    prediction = report["prediction"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    axes[0].plot([0], [target], "o", color="#222222", ms=7, label="Liu experiment")
    axes[0].plot([1], [prediction["peak_hrms_over_depth"]], "s", color="#2468a2", ms=7, label="Protected prediction")
    axes[0].set(xlim=(-0.5, 1.5), xticks=[0, 1], xticklabels=["Experiment", "Frozen\nclosure"], ylabel=r"Peak $h_{rms}/D$", ylim=(0, 0.026))
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].text(0.04, 0.92, f"Error = {100*prediction['peak_absolute_relative_error']:.1f}%\n30% gate: PASS", transform=axes[0].transAxes, va="top")
    axes[1].plot(u, amplitude, color="#2468a2", lw=2, label="Model response")
    axes[1].axvline(report["experiment"]["reported_onset_reduced_velocity"], color="#222222", ls=":", label="Reported onset")
    axes[1].axhline(target, color="#222222", ls="--", label="Experimental peak")
    axes[1].set(xlabel=r"Reduced velocity $U/(f_nD)$", ylabel=r"Predicted $h_{rms}/D$", xlim=(7.5, 10.4), ylim=(0, 0.026))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "evidence/geometry_blended_validation")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = assess()
    report_path = args.output_dir / "validation.json"
    plot_path = args.output_dir / "validation.pdf"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(report, plot_path)
    (args.output_dir / "SHA256SUMS").write_text(f"{_sha256(report_path)}  {report_path.name}\n{_sha256(plot_path)}  {plot_path.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["prediction"]}, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
