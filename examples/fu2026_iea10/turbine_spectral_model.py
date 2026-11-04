#!/usr/bin/env python3
"""Freeze and evaluate the first IEA-10MW static-spectrum VIV closure.

This is an intentionally compact engineering model.  The local FFA-W3 fixed-
body spectrum supplies the speed-dependent near-resonant force scale.  The
IEA-10MW first-edge modal mass, frequency, damping, and shape supply the
structural damping balance.  Horcas et al. (2022) Figure 4 amplitudes at
18 m/s identify one motion-correlated gain per declared inflow configuration,
while the Grinderslev et al. (2022) P100-I50 signed-power zero crossing fixes a
common saturation amplitude.  Fu non-18-m/s response values are not read by
the fitting or prediction path.

The frozen model is evaluated by ``assess.py`` in a separate command.  This
separation leaves a hash-bound model artifact before the held-out responses
are scored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.grinderslev2022_iea10.response_aware_benchmark import (
    derive_blade_quadrature,
)
from vorlap.airfoil_io import load_airfoil_fft
from vorlap.geometry_frames import prepare_component_frames
from vorlap.interpolation import interpolate_fft_spectrum_batch
from vorlap.qblade_conversion import convert_qblade_to_vorlap_inputs
from vorlap.qblade_engineering_lockin import capture_weight


MODEL_SCHEMA = "vorlap-iea10-turbine-spectral-negative-damping-v1"
PREDICTION_COLUMNS = (
    "case_id",
    "model_fingerprint",
    "Ay_pred_over_cbar",
    "fy_pred_over_fn2",
)
TRAINING = HERE / "horcas2022_figure4_training.csv"
TRAINING_PROVENANCE = HERE / "horcas2022_training_provenance.json"
POWER_CURVE = (
    REPO_ROOT
    / "examples/grinderslev2022_iea10/reference_data/"
    "p100_i50_figure6_power_digitized.csv"
)
SPECTRUM = REPO_ROOT / "data/airfoils/ffa_w3_211.h5"
BASE_CASE = REPO_ROOT / "examples/grinderslev2022_iea10/generated_case"
SIMULATION = BASE_CASE / "grinderslev2022_p100_i50_iea10_single_blade.sim"
MEAN_CHORD_M = 3.72
AIR_DENSITY_KGPM3 = 1.225
AIR_KINEMATIC_VISCOSITY_M2PS = 1.48e-5
CAPTURE_LOW = 0.8
CAPTURE_PEAK = 1.0
CAPTURE_HIGH = 1.2


class TurbineModelError(ValueError):
    """Raised when the frozen turbine-model contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TurbineModelError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TurbineModelError(f"{name} must be a finite real number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TurbineModelError(f"{name} must be a finite real number.") from exc
    _require(math.isfinite(result), f"{name} must be finite.")
    return result


def _read_training() -> list[dict[str, Any]]:
    expected = (
        "config",
        "dtu_pitch_deg",
        "dtu_inclination_deg",
        "wind_speed_mps",
        "edgewise_peak_amplitude_m",
        "amplitude_digitization_uncertainty_m",
        "source_role",
    )
    with TRAINING.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == expected, "Training columns changed.")
        rows = list(reader)
    _require(len(rows) == 4, "Expected exactly four Horcas training rows.")
    _require(len({row["config"] for row in rows}) == 4, "Training configs repeat.")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed_row = {
            "config": row["config"],
            "dtu_pitch_deg": _finite(row["dtu_pitch_deg"], "pitch"),
            "dtu_inclination_deg": _finite(
                row["dtu_inclination_deg"], "inclination"
            ),
            "wind_speed_mps": _finite(row["wind_speed_mps"], "wind speed"),
            "edgewise_peak_amplitude_m": _finite(
                row["edgewise_peak_amplitude_m"], "training amplitude"
            ),
            "amplitude_digitization_uncertainty_m": _finite(
                row["amplitude_digitization_uncertainty_m"],
                "training digitization uncertainty",
            ),
            "source_role": row["source_role"],
        }
        _require(
            parsed_row["wind_speed_mps"] == 18.0,
            "Turbine fitting is restricted to U=18 m/s.",
        )
        _require(
            parsed_row["edgewise_peak_amplitude_m"] >= 0.0,
            "Training amplitudes must be non-negative.",
        )
        _require(row["source_role"] == "training", "Unexpected training role.")
        parsed.append(parsed_row)
    return parsed


def _common_saturation_amplitude_m() -> float:
    with POWER_CURVE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    zeros = [
        _finite(row["tip_amplitude_m"], "power-curve amplitude")
        for row in rows
        if _finite(row["aerodynamic_power_w"], "power") == 0.0
        and _finite(row["tip_amplitude_m"], "power-curve amplitude") > 0.0
    ]
    _require(zeros == [1.57], "Expected the declared 1.57 m power zero crossing.")
    return zeros[0]


def _blade_state() -> dict[str, Any]:
    components, _params, _node_ids = convert_qblade_to_vorlap_inputs(
        str(SIMULATION), include_struts=False, include_tower=False
    )
    _require(len(components) == 1, "Expected exactly one IEA-10MW blade.")
    prepare_component_frames(components)
    component = components[0]
    quadrature = derive_blade_quadrature(BASE_CASE)
    chord = np.asarray(component.chord_vector, dtype=float)
    normal = np.asarray(component.normal_vector, dtype=float)
    span = np.cross(chord, normal)
    span /= np.linalg.norm(span, axis=1)[:, None]
    _require(chord.shape == (30, 3), "Unexpected blade-frame shape.")
    _require(len(quadrature.chord_m) == 30, "Unexpected blade quadrature size.")
    return {
        "quadrature": quadrature,
        "chord": chord,
        "normal": normal,
        "span": span,
        "airfoil": load_airfoil_fft(str(SPECTRUM)),
    }


def _inflow_unit_vector(
    blade: dict[str, Any], pitch_deg: float, inclination_deg: float
) -> np.ndarray:
    chord = blade["chord"][0]
    normal = blade["normal"][0]
    span = blade["span"][0]
    pitch = math.radians(pitch_deg)
    inclination = math.radians(inclination_deg)
    vector = (
        math.cos(inclination)
        * (math.cos(pitch) * (-chord) + math.sin(pitch) * normal)
        - math.sin(inclination) * span
    )
    vector /= np.linalg.norm(vector)
    return vector


def unit_gain_feedback_coefficient_kgps(
    *,
    wind_speed_mps: float,
    pitch_deg: float,
    inclination_deg: float,
    blade: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the modal negative-damping coefficient for unit wake-work gain."""

    speed = _finite(wind_speed_mps, "wind_speed_mps")
    _require(speed > 0.0, "wind_speed_mps must be positive.")
    state = _blade_state() if blade is None else blade
    quadrature = state["quadrature"]
    inflow = _inflow_unit_vector(state, pitch_deg, inclination_deg)
    chordwise = speed * (state["chord"] @ inflow)
    normal = speed * (state["normal"] @ inflow)
    local_speed = np.hypot(chordwise, normal)
    aoa_deg = np.degrees(np.arctan2(normal, chordwise))
    reynolds = (
        local_speed
        * quadrature.chord_m
        / AIR_KINEMATIC_VISCOSITY_M2PS
    )
    airfoil = state["airfoil"]
    strouhal, amplitude, _phase = interpolate_fft_spectrum_batch(
        airfoil,
        reynolds,
        aoa_deg,
        "CL",
        bounds_policy="clip",
    )
    projected_width = quadrature.chord_m * np.maximum(
        np.abs(np.sin(np.radians(aoa_deg))), float(airfoil.Thickness)
    )
    source_frequency = (
        strouhal * local_speed[:, None] / projected_width[:, None]
    )
    weights = capture_weight(
        source_frequency / quadrature.natural_frequency_hz,
        low=CAPTURE_LOW,
        peak=CAPTURE_PEAK,
        high=CAPTURE_HIGH,
    )
    weights[:, 0] = 0.0
    lift_to_edge_projection = np.abs(np.sin(np.radians(aoa_deg)))
    captured_lift = (
        np.sqrt(np.sum((amplitude * weights) ** 2, axis=1))
        * lift_to_edge_projection
    )
    nodal = (
        0.5
        * AIR_DENSITY_KGPM3
        * local_speed
        * projected_width
        * quadrature.tributary_span_m
        * captured_lift
        * quadrature.response_qblade_mode_shape**2
    )
    coefficient = float(np.sum(nodal))
    _require(
        math.isfinite(coefficient) and coefficient >= 0.0,
        "Spectral feedback coefficient is invalid.",
    )
    return {
        "coefficient_kgps_per_unit_gain": coefficient,
        "minimum_local_aoa_deg": float(np.min(aoa_deg)),
        "maximum_local_aoa_deg": float(np.max(aoa_deg)),
        "maximum_captured_lift_coefficient": float(np.max(captured_lift)),
    }


def fit_model() -> dict[str, Any]:
    training = _read_training()
    blade = _blade_state()
    quadrature = blade["quadrature"]
    omega = 2.0 * math.pi * quadrature.natural_frequency_hz
    structural_damping_kgps = (
        2.0
        * quadrature.structural_damping_ratio
        * quadrature.modal_mass_kg
        * omega
    )
    saturation_m = _common_saturation_amplitude_m()
    fitted: list[dict[str, Any]] = []
    for row in training:
        spectral = unit_gain_feedback_coefficient_kgps(
            wind_speed_mps=row["wind_speed_mps"],
            pitch_deg=row["dtu_pitch_deg"],
            inclination_deg=row["dtu_inclination_deg"],
            blade=blade,
        )
        amplitude = row["edgewise_peak_amplitude_m"]
        _require(amplitude < saturation_m, "Training amplitude exceeds saturation.")
        saturation_factor = 1.0 - (amplitude / saturation_m) ** 2
        unit_coefficient = spectral["coefficient_kgps_per_unit_gain"]
        _require(unit_coefficient > 0.0, "Training spectrum has no captured energy.")
        gain = structural_damping_kgps / (unit_coefficient * saturation_factor)
        fitted.append(
            {
                **row,
                **spectral,
                "saturation_factor": saturation_factor,
                "motion_correlated_gain": gain,
                "reconstructed_amplitude_m": saturation_m
                * math.sqrt(
                    max(
                        0.0,
                        1.0 - structural_damping_kgps / (gain * unit_coefficient),
                    )
                ),
            }
        )
    model: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "status": "frozen_before_fu_non18_evaluation",
        "model_type": "iea10_static_spectrum_modal_negative_damping_v1",
        "equations": {
            "zero_amplitude_feedback": "c_a0=H*sum_i(0.5*rho*U_i*D_i*ds_i*Ccap_i*phi_i^2)",
            "signed_feedback": "c_a(A)=c_a0*(1-(A/A_sat)^2)",
            "equilibrium": "A=A_sat*sqrt(max(0,1-c_s/c_a0))",
            "structural_damping": "c_s=2*zeta*m_modal*omega_n",
            "frequency": "f_response/f_n=1 when c_a0>c_s; otherwise 0",
        },
        "parameters": {
            "air_density_kgpm3": AIR_DENSITY_KGPM3,
            "air_kinematic_viscosity_m2ps": AIR_KINEMATIC_VISCOSITY_M2PS,
            "mean_chord_m": MEAN_CHORD_M,
            "capture_frequency_ratio": [CAPTURE_LOW, CAPTURE_PEAK, CAPTURE_HIGH],
            "saturation_tip_amplitude_m": saturation_m,
        },
        "structure": {
            "natural_frequency_hz": quadrature.natural_frequency_hz,
            "modal_mass_kg": quadrature.modal_mass_kg,
            "structural_damping_ratio": quadrature.structural_damping_ratio,
            "structural_damping_coefficient_kgps": structural_damping_kgps,
        },
        "sources": {
            "training_csv": {
                "path": TRAINING.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(TRAINING),
            },
            "training_provenance": {
                "path": TRAINING_PROVENANCE.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(TRAINING_PROVENANCE),
            },
            "stationary_spectrum": {
                "path": SPECTRUM.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(SPECTRUM),
                "role": "fixed-body shedding frequency and available fluctuating-lift scale only",
                "provenance_boundary": "hash-bound legacy HDF5; full source histories are not tracked",
            },
            "grinderslev_power_curve": {
                "path": POWER_CURVE.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(POWER_CURVE),
                "role": "common signed-power zero crossing only",
            },
            "blade_simulation": {
                "path": SIMULATION.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(SIMULATION),
            },
            "model_code": {
                "path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "training": fitted,
        "protected_validation": {
            "protocol": "examples/fu2026_iea10/validation_protocol.json",
            "case_count": 16,
            "fu_non18_response_values_used_for_fit": False,
            "failure_policy": "preserve failed assessment before any retuning",
        },
        "claim_boundary": {
            "training": "Horcas U=18 orientation map plus Grinderslev signed-power saturation",
            "not_yet_validated": True,
            "fatigue_validated": False,
        },
    }
    model["model_fingerprint"] = semantic_sha256(model)
    return model


def _load_model(path: Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(model.get("schema") == MODEL_SCHEMA, "Unexpected model schema.")
    fingerprint = model.pop("model_fingerprint", None)
    _require(fingerprint == semantic_sha256(model), "Model fingerprint differs.")
    model["model_fingerprint"] = fingerprint
    _require(
        model["sources"]["training_csv"]["sha256"] == sha256_file(TRAINING),
        "Training CSV bytes differ from the frozen model.",
    )
    _require(
        model["sources"]["training_provenance"]["sha256"]
        == sha256_file(TRAINING_PROVENANCE),
        "Training provenance bytes differ from the frozen model.",
    )
    _require(
        model["sources"]["stationary_spectrum"]["sha256"] == sha256_file(SPECTRUM),
        "Stationary spectrum bytes differ from the frozen model.",
    )
    _require(
        model["sources"]["model_code"]["sha256"]
        == sha256_file(Path(__file__).resolve()),
        "Model implementation differs from the frozen fit.",
    )
    return model


def predict_model(model_path: Path) -> list[dict[str, Any]]:
    model = _load_model(model_path)
    blade = _blade_state()
    damping = float(model["structure"]["structural_damping_coefficient_kgps"])
    saturation = float(model["parameters"]["saturation_tip_amplitude_m"])
    fitted = {row["config"]: row for row in model["training"]}
    protocol = json.loads((HERE / "validation_protocol.json").read_text(encoding="utf-8"))
    case_ids = protocol["primary_scored_case_ids"]
    _require(len(case_ids) == 16, "Expected 16 protected case IDs.")
    predictions: list[dict[str, Any]] = []
    for case_id in case_ids:
        config, speed_token = case_id.rsplit("_U", 1)
        _require(config in fitted, f"No frozen training configuration for {config}.")
        speed = float(int(speed_token))
        row = fitted[config]
        spectral = unit_gain_feedback_coefficient_kgps(
            wind_speed_mps=speed,
            pitch_deg=float(row["dtu_pitch_deg"]),
            inclination_deg=float(row["dtu_inclination_deg"]),
            blade=blade,
        )
        feedback = (
            float(row["motion_correlated_gain"])
            * spectral["coefficient_kgps_per_unit_gain"]
        )
        active = feedback > damping
        amplitude_m = (
            saturation * math.sqrt(max(0.0, 1.0 - damping / feedback))
            if active
            else 0.0
        )
        predictions.append(
            {
                "case_id": case_id,
                "model_fingerprint": model["model_fingerprint"],
                "Ay_pred_over_cbar": amplitude_m / MEAN_CHORD_M,
                "fy_pred_over_fn2": 1.0 if active else 0.0,
            }
        )
    return predictions


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in predictions:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "model_fingerprint": row["model_fingerprint"],
                    "Ay_pred_over_cbar": format(row["Ay_pred_over_cbar"], ".17g"),
                    "fy_pred_over_fn2": format(row["fy_pred_over_fn2"], ".17g"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", help="Freeze the training-only model.")
    fit.add_argument("--output", type=Path, required=True)
    predict = subparsers.add_parser("predict", help="Predict protected case IDs.")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        model = fit_model()
        _write_json(args.output, model)
        print(json.dumps({"path": str(args.output), "model_fingerprint": model["model_fingerprint"]}, indent=2))
    else:
        predictions = predict_model(args.model)
        _write_predictions(args.output, predictions)
        print(json.dumps({"path": str(args.output), "case_count": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()
