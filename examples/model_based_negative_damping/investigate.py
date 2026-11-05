"""Reproduce an offline CFD-work/structural-model feasibility comparison.

Run from the repository root with .venv/bin/python. This reads existing
figure digitizations and structural identification; it does not run CFD,
retune a production gain, or use free-response targets to identify a law.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vorlap-model-based-mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.integrate import solve_ivp

from vorlap.model_based_negative_damping import (
    StructuralMode, fit_quadratic_work_law, project_sectional_work,
)
from vorlap.qblade_modal_power_model import TabulatedAerodynamicPowerCurve


REFERENCE = REPOSITORY / "examples/grinderslev2022_iea10/reference_data"
POWER_PATH = REFERENCE / "p100_i50_figure6_power_digitized.csv"
STRUCTURE_PATH = REFERENCE / "qblade_first_edge_mode_identification.json"
MODE_PATH = REFERENCE / "qblade_first_edge_mode.csv"


def identity(path: Path) -> dict:
    return {"path": str(path.relative_to(REPOSITORY)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prediction(mode: StructuralMode, curve: TabulatedAerodynamicPowerCurve) -> dict:
    fit = fit_quadratic_work_law(curve.amplitude_m, curve.power_w,
                               omega_radps=mode.omega_radps)
    table_roots = curve.structural_equilibria_m(mode.power_coefficient_wpm2)
    if len(table_roots) != 1:
        raise ValueError("This benchmark expects one resolved nonzero equilibrium.")
    # The reused table root finder suffices for this descending branch. It is
    # not an exhaustive root/stability solver for arbitrary multibranch maps.
    aeq = table_roots[0]
    derivative_step = 1.0e-5
    net_slope = ((curve.evaluate(aeq + derivative_step) - curve.evaluate(aeq - derivative_step))
                 / (2 * derivative_step) - 2 * mode.power_coefficient_wpm2 * aeq)
    if net_slope >= 0.0:
        raise ValueError("The selected table equilibrium must be attracting.")
    return {"fit": fit, "table_equilibrium_m": aeq,
            "quadratic_equilibrium_m": fit.law.equilibrium_amplitude_m(mode.damping_kgps),
            "table_net_power_slope_wpm": net_slope}


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    curve = TabulatedAerodynamicPowerCurve.from_csv(POWER_PATH)
    structure = json.loads(STRUCTURE_PATH.read_text())
    shape_data = np.genfromtxt(MODE_PATH, delimiter=",", names=True)
    if identity(MODE_PATH)["sha256"] != structure["mode_csv_sha256"]:
        raise ValueError("Structural mode CSV identity differs from its identification.")
    phi = shape_data["qblade_mode_tip_normalized"]
    mass = float(np.sum(shape_data["modal_mass_weight_kg"] * phi**2))
    if not math.isclose(mass, structure["qblade_modal_mass_kg"], rel_tol=1.0e-12):
        raise ValueError("Structural mass projection differs from archived identity.")
    omega = 2 * math.pi * structure["natural_frequency_hz"]
    # These are the equivalent single-mode coefficients, not imported full
    # Chrono matrices. Frequency comes from the archived structural probe;
    # beta is the documented structural damping assumption, not an aero fit.
    stiffness = mass * omega**2
    damping = structure["rayleigh_stiffness_coefficient_s"] * stiffness
    mode = StructuralMode(mass, stiffness, damping, 0.0)
    if not math.isclose(mode.power_coefficient_wpm2,
                        structure["structural_power_coefficient_wpm2"], rel_tol=1.0e-12):
        raise ValueError("Derived structural dissipation differs from archived identity.")
    nominal = prediction(mode, curve)
    fit = nominal.pop("fit")
    a = np.asarray(curve.amplitude_m)
    power = np.asarray(curve.power_w)

    # Fixed, reproducible model-reduction diagnostics; no target response in
    # any fit. A removed table knot is a numerical interpolation check, not
    # a new physical validation case. Keep the origin and final support knot.
    left_out = []
    for index in range(1, a.size - 1):
        reduced = TabulatedAerodynamicPowerCurve(tuple(np.delete(a, index)),
                                                tuple(np.delete(power, index)))
        reduced_fit = fit_quadratic_work_law(reduced.amplitude_m, reduced.power_w,
                                           omega_radps=omega)
        left_out.append({"omitted_amplitude_m": float(a[index]),
                         "table_power_error_w": reduced.evaluate(a[index]) - power[index],
                         "quadratic_power_error_w": reduced_fit.law.power_w(a[index]) - power[index]})

    # Deterministic digitization perturbations. These are not confidence
    # intervals or CFD uncertainty; all nonzero points receive the same offset.
    raw = np.genfromtxt(POWER_PATH, delimiter=",", names=True, dtype=None, encoding="utf-8")
    perturbations = []
    for sign_a in (-1, 1):
        for sign_p in (-1, 1):
            shifted_a = a + sign_a * raw["amplitude_uncertainty_m"]
            shifted_p = power + sign_p * raw["power_uncertainty_w"]
            shifted_a[0], shifted_p[0] = 0.0, 0.0
            shifted_curve = TabulatedAerodynamicPowerCurve(tuple(shifted_a), tuple(shifted_p))
            row = prediction(mode, shifted_curve)
            row.pop("fit")
            perturbations.append({"amplitude_shift_sign": sign_a, "power_shift_sign": sign_p, **row})

    damping_sweep = []
    for scale in np.linspace(0.0, 2.0, 41):
        coefficient = mode.power_coefficient_wpm2 * scale
        roots = curve.structural_equilibria_m(coefficient)
        if len(roots) != 1:
            raise ValueError("Damping sweep left the single stable branch of this benchmark.")
        damping_sweep.append({"damping_multiplier": float(scale),
                              "table_equilibrium_m": roots[0],
                              "quadratic_equilibrium_m": fit.law.equilibrium_amplitude_m(mode.damping_kgps * scale)})

    # Reuse corrected energy equation dA/dt=(P_a-P_s)/(M omega² A).
    # Rewrite using finite velocity gain at A=0. Zero initial motion remains
    # zero; a declared 0.01 m numerical seed chooses the growing branch.
    def envelope_rhs(law):
        def rhs(_time, state):
            amplitude = float(state[0])
            gain = (curve.equivalent_velocity_gain(amplitude, omega) if law == "table"
                    else fit.law.velocity_gain(amplitude))
            return [(gain - mode.damping_kgps) * amplitude / (2.0 * mass)]
        return rhs

    histories, integration = {}, {}
    time = np.linspace(0, 600, 1201)
    for name in ("table", "quadratic"):
        solutions = []
        for tolerance in (1.0e-7, 1.0e-9):
            solution = solve_ivp(envelope_rhs(name), (0, time[-1]), [0.01], t_eval=time,
                                 rtol=tolerance, atol=tolerance * 0.01, max_step=0.5)
            if not solution.success:
                raise RuntimeError(solution.message)
            solutions.append(solution.y[0])
        histories[name] = solutions[-1]
        equilibrium = nominal[f"{name}_equilibrium_m"]
        integration[name] = {
            "terminal_amplitude_m": float(solutions[-1][-1]),
            "absolute_terminal_equilibrium_error_m": abs(float(solutions[-1][-1]) - equilibrium),
            "maximum_tolerance_refinement_difference_m": float(np.max(np.abs(solutions[1] - solutions[0]))),
        }
        if integration[name]["absolute_terminal_equilibrium_error_m"] > 1.0e-6:
            raise RuntimeError("Envelope solve did not settle to the calculated equilibrium.")
        if integration[name]["maximum_tolerance_refinement_difference_m"] > 1.0e-4:
            raise RuntimeError("Envelope solve did not pass its tolerance-refinement check.")

    # Analytic illustration: uniform local work, phi=(s/L)^n. This isolates
    # structural projection; neither the local b0 nor saturation is predicted.
    span = np.linspace(0, 1, 2001)
    weights = np.full(span.size, 1 / (span.size - 1))
    weights[[0, -1]] *= 0.5
    projection = []
    for exponent in (0, 1, 2, 3):
        shape = span**exponent
        projected = project_sectional_work(
            zero_gain_kgps=10 * weights, saturation_coefficient_kgpspm2=1000 * weights,
            mode_shape=shape, omega_radps=omega, maximum_local_amplitude_m=np.ones(span.size))
        factor = projected.limiting_amplitude_m / 0.1
        expected = math.sqrt((4 * exponent + 1) / (2 * exponent + 1))
        if abs(factor - expected) > 2.0e-6:
            raise RuntimeError("Sectional work quadrature disagrees with analytic moments.")
        projection.append({"admissible_shape_exponent": exponent,
                           "modal_to_local_saturation_ratio": factor,
                           "analytic_ratio": expected})

    # The reference FSI response enters ONLY here, after every coefficient,
    # sensitivity, and prediction is fixed. It is a published numerical target,
    # already known during design of this study, not newly blinded data.
    fsi_damped, fsi_undamped = 1.18, 1.63
    comparison = [
        {"model": "Full CFD work table + converted QBlade structure",
         "damped_m": nominal["table_equilibrium_m"],
         "undamped_m": curve.positive_zero_crossings_m()[0]},
        {"model": "Quadratic CFD work fit + converted QBlade structure",
         "damped_m": nominal["quadratic_equilibrium_m"],
         "undamped_m": fit.law.limiting_amplitude_m},
    ]
    for row in comparison:
        row["damped_relative_difference_percent"] = 100 * (row["damped_m"] / fsi_damped - 1)
        row["undamped_relative_difference_percent"] = 100 * (row["undamped_m"] / fsi_undamped - 1)

    result = {
        "schema": "vorlap-model-based-negative-damping-investigation-v1",
        "evidence": "offline published-CFD-work reduction and numerical comparison; no experimental validation",
        "production_model_changed": False,
        "sources": [identity(p) for p in (POWER_PATH, STRUCTURE_PATH, MODE_PATH)],
        "code": [identity(Path(__file__)), identity(REPOSITORY / "vorlap/model_based_negative_damping.py"),
                 identity(REPOSITORY / "vorlap/qblade_modal_power_model.py")],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "structural_mode": asdict(mode), "structural_power_coefficient_wpm2": mode.power_coefficient_wpm2,
        "structural_scope": "Mass projected from archived nodal weights; frequency from archived structural probe; beta stiffness damping assumption; not full exported Chrono M/K/C",
        "motion_frequency_approximation": {"published_forced_frequency_hz": 0.67,
            "reduction_frequency_hz": omega / (2 * math.pi),
            "relative_difference_percent": 100 * (omega / (2 * math.pi * 0.67) - 1),
            "scope": "Treat the 0.0224 percent difference as negligible, as in the existing benchmark; no independently verified frequency transfer"},
        "quadratic_fit": asdict(fit), "nominal": nominal,
        "comparisons": comparison, "published_fsi_damped_m": fsi_damped,
        "published_fsi_undamped_m": fsi_undamped,
        "paper_fitted_damping_540_wpm2_context_root_m": curve.structural_equilibria_m(540)[0],
        "paper_damping_context_scope": "540 W/m² was fitted by the paper to FSI dissipation; excluded from structural-model prediction",
        "leave_one_amplitude_out": left_out,
        "leave_one_amplitude_out_rmse_w": {
            name: float(np.sqrt(np.mean([row[f"{name}_power_error_w"]**2 for row in left_out])))
            for name in ("table", "quadratic")},
        "digitization_sensitivity_not_confidence_intervals": perturbations,
        "damping_sweep": damping_sweep, "envelope_integration": integration,
        "analytic_modal_projection_illustration": projection,
        "missing_for_experimental_parity": ["Target-matched prescribed-motion NACA0018 aerodynamic data",
            "Exact retained native-base work at identical prescribed motion", "Frozen matched experimental validation split"],
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0), layout="constrained")
    grid = np.linspace(0, a[-1], 500)
    axes[0].plot(grid, [curve.evaluate(x) for x in grid], color="black", lw=1.6, label="CFD work table")
    axes[0].plot(a, power, "o", color="black", ms=3.5)
    axes[0].plot(grid, [fit.law.power_w(x) for x in grid], color="#0067a5", lw=1.8,
                 label="Quadratic reduction")
    axes[0].plot(grid, mode.power_coefficient_wpm2 * grid**2, color="#c65f00", ls="--", lw=1.8,
                 label="Structural dissipation")
    axes[0].axhline(0, color="0.7", lw=0.7)
    axes[0].set(xlabel="Peak tip amplitude [m]", ylabel="Cycle-mean power [W]")
    axes[0].legend(fontsize=8.5, loc="upper left")
    scales = [row["damping_multiplier"] for row in damping_sweep]
    for name, color in (("table", "black"), ("quadratic", "#0067a5")):
        axes[1].plot(scales, [row[f"{name}_equilibrium_m"] for row in damping_sweep],
                     color=color, lw=1.8, label="Full work table" if name == "table" else "Quadratic reduction")
    axes[1].set(xlabel="Converted-model damping multiplier [–]", ylabel="Equilibrium peak tip amplitude [m]")
    axes[1].axvline(1, color="0.7", lw=0.7, ls=":")
    axes[1].legend(fontsize=8.5)
    for name, color in (("table", "black"), ("quadratic", "#0067a5")):
        axes[2].plot(time, histories[name], color=color, lw=1.8,
                     label="Full work table" if name == "table" else "Quadratic reduction")
    axes[2].set(xlabel="Time [s]", ylabel="Peak tip amplitude [m]", xlim=(0, 160))
    axes[2].legend(fontsize=8.5, loc="lower right")
    for i, ax in enumerate(axes):
        ax.text(0.01, 1.035, f"({chr(97+i)})", transform=ax.transAxes)
        ax.grid(alpha=0.15)
    fig.savefig(output / "work_and_response.pdf")
    fig.savefig(output / "work_and_response.png", dpi=170)
    plt.close(fig)

    rows = ["# Model-based negative damping: reproduced results", "",
            "Offline published-CFD-work reduction; no new CFD or experimental validation.", "",
            "| Model | Damped [m] | Difference from 1.18 m FSI | Undamped [m] |", "|---|---:|---:|---:|"]
    for row in comparison:
        rows.append(f"| {row['model']} | {row['damped_m']:.6f} | {row['damped_relative_difference_percent']:+.2f}% | {row['undamped_m']:.6f} |")
    rows += ["", f"Quadratic power RMSE: {fit.power_rmse_w:.3f} W; relative L2 error: {100*fit.power_relative_l2_error:.2f}%.",
             "", f"Projected mass: {mass:.6f} kg; assumed modal damping: {mode.damping_ratio:.9f}; structural dissipation: {mode.power_coefficient_wpm2:.6f} A² W.",
             "", "The 540 W/m² damping fit in the paper is context only; it does not enter these model-derived predictions.",
             "", "Figure: (a) digitized CFD work and quadratic reduction, (b) damping sensitivity, (c) averaged-energy envelope with a 0.01 m seed. Panel (c) is a reduced-model integration, not a fresh FSI simulation.",
             "", "![Work and response](work_and_response.png)",
             "", "Full inputs, hashes, sensitivity results and convergence evidence: [results.json](results.json)."]
    (output / "results.md").write_text("\n".join(rows) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPOSITORY / "run_headless/runs/model_based_negative_damping")
    args = parser.parse_args()
    result = run(args.output_dir.resolve())
    print(json.dumps({"output_dir": str(args.output_dir), "comparisons": result["comparisons"],
                      "power_rmse_w": result["quadratic_fit"]["power_rmse_w"],
                      "envelope_integration": result["envelope_integration"]}, indent=2))
