#!/usr/bin/env python3
"""Generate, run, and analyze the response-aware Vlastos 2026 benchmark.

The benchmark uses the three published cantilever-cylinder structural cases.
Only the Sc=5.18 hGAST curve is used to calibrate the compact response closure.
The resulting dimensionless coefficients are fingerprinted and applied without
change to the Sc=2.94 and Sc=7.02 held-out cases.  Experimental Lupi peaks are
reported separately and are never calibration inputs.

Generation and run planning are non-destructive.  Existing campaign roots are
not replaced, and ``run`` is a dry plan unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import shutil
import struct
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_case as fixed_wake_case
import stationary_campaign_protocol as stationary_protocol

from vorlap.geometry_frames import prepare_component_frames
from vorlap.qblade_case_prep import QBladeCasePrepRequest, prepare_qblade_external_case
from vorlap.qblade_conversion import convert_qblade_to_vorlap_inputs
from vorlap.qblade_response_model import QBladeResponseModel
from vorlap.qblade_runtime import _response_model_from_config
from vorlap.qblade_run_support import (
    file_sha256 as runtime_file_identity,
    find_bridge_library,
    find_qblade_library,
    qblade_runtime_dependency_manifest,
    vorlap_python_runtime_manifest,
)
from vorlap.qblade_sil import normalize_qblade_header, read_qblade_result_numeric_columns


REFERENCE_DOI = "10.5194/wes-11-2585-2026"
EXPERIMENT_DOI = "10.1016/j.jweia.2020.104438"
HEIGHT_M = 0.775
DIAMETER_M = 0.050
CALIBRATION_SC = 5.18
HELD_OUT_SC = (2.94, 7.02)
DEFAULT_REDUCED_VELOCITIES = (
    4.30,
    4.80,
    5.00,
    5.20,
    5.50,
    5.80,
    6.20,
    6.25,
    6.50,
    6.80,
    7.20,
    8.20,
)
DEFAULT_TIMESTEP_S = 5.0e-4
DEFAULT_AMBIENT_V2_CAMPAIGN_ROOT = (
    HERE / "generated_response_campaign_ambient_v2"
)
# Retained only for explicit short diagnostic campaigns.  The production
# generator now defaults to the case-specific cycle caps in
# ``stationary_campaign_protocol.py``.
DEFAULT_DURATION_S = 8.0
MINIMUM_GLOBAL_STEPS_PER_CYCLE = 50.0
N_STATIONS = 21
CALIBRATION_SERIES = "hgast_varying_parameters"
CALIBRATION_THRESHOLD_FRACTION = 0.10
RESULT_CHANNEL = "X_b Tip Trl.Def. BLD_1 [m]"
SCHEMA = "vlastos2026-response-benchmark-v1"
CALIBRATION_STAGE_SCHEMA = "vlastos2026-calibration-stage-assessment-v1"
LUPI_EXPERIMENT_SCREEN_SCHEMA = "vlastos2026-lupi-experimental-curve-screen-v1"

# Enhanced qualification thresholds were frozen before the live response-aware
# campaign was inspected.  They supplement, but do not replace or relax, the
# original external-code thresholds in ``analyze_campaign``.
STATIONARITY_WINDOW_CYCLES = 20.0
STATIONARITY_ABSOLUTE_AMPLITUDE_RATIO = 0.005
STATIONARITY_RELATIVE_AMPLITUDE = 0.05
STATIONARITY_FREQUENCY_RATIO = 0.01
ACTIVE_AMPLITUDE_RATIO = 0.05
LOCK_IN_FREQUENCY_TOLERANCE_RATIO = 0.05
LOCK_IN_MINIMUM_MODE_BAND_POWER_FRACTION = 0.80
LOCK_IN_MINIMUM_CONSECUTIVE_POINTS = 3
LOCK_IN_REFERENCE_PEAK_NEIGHBORHOOD_USTAR = 0.30
LOCK_IN_ENDPOINT_TOLERANCE_USTAR = 0.30
TIMESTEP_FREQUENCY_TOLERANCE_RATIO = 0.01
TIMESTEP_PEAK_LOCATION_TOLERANCE_USTAR = 0.25
TIMESTEP_BAND_ENDPOINT_TOLERANCE_USTAR = 0.15
LIMIT_HEADROOM_FRACTION = 0.90
LIVE_ANALYTICAL_RMSE_TOLERANCE = 0.03
LIVE_ANALYTICAL_MAX_ABS_TOLERANCE = 0.05
LIVE_ANALYTICAL_PEAK_LOCATION_TOLERANCE_USTAR = 0.25
COORDINATE_ABSOLUTE_AMPLITUDE_RATIO_TOLERANCE = 0.01
COORDINATE_RELATIVE_AMPLITUDE_TOLERANCE = 0.05
MATERIAL_IMPROVEMENT_MINIMUM_SKILL = 0.50
FATIGUE_PROXY_SLOPE = 10.0
STARTUP_SEED_OBSERVATION_CYCLES = 1.0
STARTUP_SEED_REALIZED_MINIMUM_AMPLITUDE_RATIO = 0.001
STARTUP_SEED_REALIZED_MAXIMUM_AMPLITUDE_RATIO = 0.010
LUPI_EXPERIMENT_CURVE_RMSE_TOLERANCE = 0.05
LUPI_EXPERIMENT_PEAK_LOCATION_TOLERANCE_USTAR = 0.30
LUPI_EXPERIMENT_PEAK_ABSOLUTE_TOLERANCE = 0.03
LUPI_EXPERIMENT_PEAK_RELATIVE_TOLERANCE = 0.15
AMBIENT_V2_EQUATION_GATE_SCHEMA = (
    "vlastos2026-ambient-response-equation-reproduction-v1"
)
AMBIENT_V2_AERODYNAMIC_SPEED_BASIS = "declared_ambient_crossflow"
AMBIENT_V2_CASE_ID_NAMESPACE = stationary_protocol.AMBIENT_V2_CASE_ID_NAMESPACE
AMBIENT_V2_CHECKPOINT_REUSE_POLICY = (
    stationary_protocol.AMBIENT_V2_CHECKPOINT_REUSE_POLICY
)
AMBIENT_V2_CALIBRATION_QBLADE_SCRUTON_NUMBER = 4.953017025691645
AMBIENT_V2_FINE_ESTIMATOR_RMSE_TOLERANCE = 0.005
AMBIENT_V2_FINE_ESTIMATOR_MAX_ABS_TOLERANCE = 0.010
AMBIENT_V2_COARSE_FINE_ESTIMATOR_TOLERANCE = 0.005
AMBIENT_V2_ACTIVE_AMPLITUDE_RATIO = 0.05
AMBIENT_V2_EXPECTED_ACTIVE_REDUCED_VELOCITIES = (5.8, 6.2, 6.25, 6.5)
AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256 = (
    "330242617d9d04d2cb771767fefbedac6414bea9a04a9b6ea9e813c0aba18c7d"
)
AMBIENT_V2_RUN_PLAN_SCHEMA = stationary_protocol.AMBIENT_V2_RUN_PLAN_SCHEMA
QBLADE_RESULT_TIME_FORMAT = "%.5E"
QBLADE_RESULT_TIME_FORMAT_DECIMALS = 5
QBLADE_RESULT_TIME_GRID_SCHEMA = "qblade-result-time-grid-reconstruction-v1"


@dataclass(frozen=True)
class StructuralCase:
    scruton_number: float
    natural_frequency_hz: float
    equivalent_mass_kg_per_m: float
    damping_ratio: float
    g_prime_max: float
    h_prime_max: float
    u_star_max_figure_digitized: float


PUBLISHED_STRUCTURAL_CASES = {
    2.94: (20.02, 0.2311, 0.0030),
    5.18: (15.38, 0.4311, 0.0028),
    7.02: (13.18, 0.6027, 0.0027),
}


def derived_scruton_number(case: StructuralCase) -> float:
    """Scruton number implied by the generated QBlade mass/damping/fluid data."""

    return (
        4.0
        * math.pi
        * case.equivalent_mass_kg_per_m
        * case.damping_ratio
        / (fixed_wake_case.AIR_DENSITY_KGPM3 * DIAMETER_M**2)
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_tag(value: float, *, prefix: str = "") -> str:
    return prefix + f"{float(value):.5f}".rstrip("0").rstrip(".").replace(".", "p")


def _reference_path(name: str) -> Path:
    return HERE / "reference_data" / name


def load_structural_cases(path: Path | None = None) -> dict[float, StructuralCase]:
    """Load and strictly check the three Vlastos Table 1 structures."""

    source = _reference_path("vlastos2026_model_parameters.csv") if path is None else Path(path)
    rows: dict[float, StructuralCase] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            case = StructuralCase(
                scruton_number=float(raw["scruton_number"]),
                natural_frequency_hz=float(raw["natural_frequency_hz"]),
                equivalent_mass_kg_per_m=float(raw["equivalent_mass_kg_per_m"]),
                damping_ratio=float(raw["damping_ratio"]),
                g_prime_max=float(raw["g_prime_max"]),
                h_prime_max=float(raw["h_prime_max"]),
                u_star_max_figure_digitized=float(raw["u_star_max_figure_digitized"]),
            )
            rows[case.scruton_number] = case
    for sc, (frequency, mass, damping) in PUBLISHED_STRUCTURAL_CASES.items():
        if sc not in rows:
            continue
        case = rows[sc]
        observed = (case.natural_frequency_hz, case.equivalent_mass_kg_per_m, case.damping_ratio)
        if observed != (frequency, mass, damping):
            raise ValueError(f"Structural case Sc={sc:.2f} does not match published Table 1: {observed!r}.")
    unexpected = set(rows).difference(PUBLISHED_STRUCTURAL_CASES)
    if unexpected:
        raise ValueError(f"Unexpected Scruton cases in {source}: {sorted(unexpected)}")
    return rows


def _read_figure9_series(path: Path, series: str) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["series"] == series:
                points.append((float(row["reduced_velocity"]), float(row["tip_amplitude_over_diameter"])))
    if len(points) < 3:
        raise ValueError(f"Calibration series {series!r} has fewer than three points in {path}.")
    points.sort()
    values = np.asarray(points, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(np.diff(values[:, 0]) <= 0.0) or np.any(values[:, 1] < 0.0):
        raise ValueError(f"Calibration series {series!r} is not finite, ordered, and non-negative.")
    return values[:, 0], values[:, 1]


def digitized_series_peak(path: Path, series: str) -> dict[str, float]:
    """Return the sampled maximum of one provenance-bound Figure 9 series."""

    reduced_velocity, amplitude = _read_figure9_series(Path(path), series)
    peak_index = int(np.argmax(amplitude))
    return {
        "reduced_velocity": float(reduced_velocity[peak_index]),
        "tip_amplitude_over_diameter": float(amplitude[peak_index]),
    }


def _threshold_crossing(
    reduced_velocity: np.ndarray,
    amplitude: np.ndarray,
    threshold: float,
    *,
    rising: bool,
) -> float:
    peak_index = int(np.argmax(amplitude))
    candidate_indices = range(peak_index) if rising else range(peak_index, len(amplitude) - 1)
    for index in candidate_indices:
        y0 = float(amplitude[index])
        y1 = float(amplitude[index + 1])
        crossed = y0 <= threshold <= y1 if rising else y0 >= threshold >= y1
        if crossed:
            if y1 == y0:
                return float(reduced_velocity[index])
            fraction = (threshold - y0) / (y1 - y0)
            return float(reduced_velocity[index] + fraction * (reduced_velocity[index + 1] - reduced_velocity[index]))
    side = "rising" if rising else "falling"
    raise ValueError(f"Could not determine the {side} calibration threshold crossing.")


def _smooth_activation(
    reduced_velocity: np.ndarray | Sequence[float],
    *,
    low: float,
    peak: float,
    high: float,
) -> np.ndarray:
    values = np.asarray(reduced_velocity, dtype=float)
    activation = np.zeros_like(values)
    rising = (values > low) & (values < peak)
    falling = (values >= peak) & (values < high)
    rise_coordinate = (values[rising] - low) / (peak - low)
    fall_coordinate = (high - values[falling]) / (high - peak)
    activation[rising] = rise_coordinate**2 * (3.0 - 2.0 * rise_coordinate)
    activation[falling] = fall_coordinate**2 * (3.0 - 2.0 * fall_coordinate)
    activation[values == peak] = 1.0
    return activation


def analytical_equilibrium_amplitude_ratio(
    scruton_number: float,
    reduced_velocity: np.ndarray | Sequence[float],
    *,
    aerodynamic_damping_gain_h1: float,
    saturation_amplitude_ratio: float,
    reduced_velocity_low: float,
    reduced_velocity_peak: float,
    reduced_velocity_high: float,
) -> np.ndarray:
    """Return the simple closure's steady energy-balance amplitude ``A/D``.

    With the tip-normalized mode used here, the uniform modal-mass and
    aerodynamic tributary integrals cancel.  Equating structural damping to
    the response-model force gives

    ``A/D = a_sat sqrt(max(0, 1 - 2 Sc/(U* H1 activation)))``.

    This is a deliberately cheap pre-QBlade gate.  It does not include the
    fixed-wake seed force or transient effects and cannot establish a passing
    coupled result, but a failure identifies a structural model-form problem
    before an expensive campaign.
    """

    u_star = np.asarray(reduced_velocity, dtype=float)
    activation = _smooth_activation(
        u_star,
        low=reduced_velocity_low,
        peak=reduced_velocity_peak,
        high=reduced_velocity_high,
    )
    denominator = u_star * float(aerodynamic_damping_gain_h1) * activation
    radicand = np.full_like(u_star, -np.inf)
    active = denominator > 0.0
    radicand[active] = 1.0 - 2.0 * float(scruton_number) / denominator[active]
    return float(saturation_amplitude_ratio) * np.sqrt(np.maximum(0.0, radicand))


def calibrate_response_coefficients(
    *,
    figure9_path: Path | None = None,
    model_parameters_path: Path | None = None,
) -> dict[str, Any]:
    """Calibrate solely from the Sc=5.18 hGAST curve and parameter row.

    The held-out Figure 10 file is intentionally not accepted by this API.
    This makes accidental use of Sc=2.94 or Sc=7.02 response targets during
    tuning structurally difficult and testable.
    """

    figure9 = _reference_path("sc5p18_figure9_digitized.csv") if figure9_path is None else Path(figure9_path)
    model_parameters = (
        _reference_path("vlastos2026_model_parameters.csv")
        if model_parameters_path is None
        else Path(model_parameters_path)
    )
    cases = load_structural_cases(model_parameters)
    if CALIBRATION_SC not in cases:
        raise ValueError(f"Calibration parameter file has no Sc={CALIBRATION_SC:.2f} row.")
    calibration_case = cases[CALIBRATION_SC]
    u_star, amplitude = _read_figure9_series(figure9, CALIBRATION_SERIES)
    peak_index = int(np.argmax(amplitude))
    peak_amplitude = float(amplitude[peak_index])
    threshold = CALIBRATION_THRESHOLD_FRACTION * peak_amplitude
    low = _threshold_crossing(u_star, amplitude, threshold, rising=True)
    high = _threshold_crossing(u_star, amplitude, threshold, rising=False)
    peak = float(u_star[peak_index])
    if not low < peak < high:
        raise ValueError("Derived response activation must satisfy low < peak < high.")

    # Fit H1 and a_sat to the Sc=5.18 curve only.  For each trial H1 the
    # least-squares a_sat is analytic because the equilibrium response is
    # linear in a_sat.  The deterministic logarithmic grid avoids an optional
    # SciPy dependency and is more than adequate for this pre-registered gate.
    activation_parameters = {
        "reduced_velocity_low": low,
        "reduced_velocity_peak": peak,
        "reduced_velocity_high": high,
    }
    best: tuple[float, float, float, np.ndarray] | None = None
    for h1 in np.geomspace(0.1, 100.0, 20001):
        unit_response = analytical_equilibrium_amplitude_ratio(
            derived_scruton_number(calibration_case),
            u_star,
            aerodynamic_damping_gain_h1=float(h1),
            saturation_amplitude_ratio=1.0,
            **activation_parameters,
        )
        denominator = float(np.dot(unit_response, unit_response))
        if denominator == 0.0:
            continue
        saturation = float(np.dot(amplitude, unit_response) / denominator)
        prediction = saturation * unit_response
        rmse = float(np.sqrt(np.mean((prediction - amplitude) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, float(h1), saturation, prediction)
    if best is None:
        raise ValueError("Sc=5.18 analytical calibration has no active response solution.")
    calibration_rmse, fitted_h1, fitted_saturation, calibration_prediction = best
    frozen = {
        **activation_parameters,
        "aerodynamic_damping_gain_h1": fitted_h1,
        "saturation_amplitude_ratio": fitted_saturation,
        # Dimensionless cycle count is frozen; each case converts it using its measured fn.
        "rms_time_constant_cycles": 2.0,
        "bandpass_damping_ratio": 0.1,
        "max_abs_gain": 10.0,
        "max_abs_nodal_force_n": 25.0,
    }
    provenance = {
        "schema": SCHEMA,
        "calibration_scruton_number": CALIBRATION_SC,
        "calibration_series": CALIBRATION_SERIES,
        "calibration_target": "published figure-digitized hGAST varying-parameter response, not experiment",
        "threshold_fraction_of_peak": CALIBRATION_THRESHOLD_FRACTION,
        "figure9_path": figure9.resolve().as_posix(),
        "figure9_sha256": sha256_file(figure9),
        "model_parameters_path": model_parameters.resolve().as_posix(),
        "model_parameters_sha256": sha256_file(model_parameters),
        "calibration_points": int(u_star.size),
        "held_out_scruton_numbers": list(HELD_OUT_SC),
        "held_out_targets_accessed_during_calibration": False,
        "analytical_calibration_fit": {
            "equation": "A/D=a_sat*sqrt(max(0,1-2*Sc/(Ustar*H1*activation)))",
            "objective": "unweighted RMSE over every digitized Sc=5.18 hGAST point",
            "rmse_tip_amplitude_over_diameter": calibration_rmse,
            "maximum_absolute_error_tip_amplitude_over_diameter": float(
                np.max(np.abs(calibration_prediction - amplitude))
            ),
            "published_h_prime_max_recorded_but_not_assumed_equal_to_fitted_h1": calibration_case.h_prime_max,
            "published_scruton_number_label": CALIBRATION_SC,
            "qblade_implied_scruton_number_used_in_energy_balance": derived_scruton_number(calibration_case),
        },
        "frozen_parameters": frozen,
    }
    provenance["coefficient_fingerprint_sha256"] = _json_fingerprint(provenance)
    return provenance


def portable_ambient_v2_coefficients(
    coefficients: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the path-independent frozen coefficient object used by v2.

    The legacy schema-v1 fingerprint includes absolute source paths and is
    intentionally left untouched.  A v2 campaign may be generated in a Git
    worktree, the main checkout, or a container, so its otherwise identical
    coefficients must not acquire a different identity from the checkout
    prefix alone.  Content hashes remain authoritative; only the two source
    path labels are canonicalized before the v2 fingerprint is computed.
    """

    try:
        portable = json.loads(
            json.dumps(dict(coefficients), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Response coefficients are not finite portable JSON."
        ) from exc
    expected_sources = {
        "figure9": (
            "sc5p18_figure9_digitized.csv",
            "figure9_path",
            "figure9_sha256",
        ),
        "model_parameters": (
            "vlastos2026_model_parameters.csv",
            "model_parameters_path",
            "model_parameters_sha256",
        ),
    }
    for label, (filename, path_field, hash_field) in expected_sources.items():
        source = _reference_path(filename)
        expected_identity = stationary_protocol.AMBIENT_V2_REFERENCE_IDENTITIES[
            filename
        ]
        expected_hash = str(expected_identity["sha256"])
        expected_bytes = int(expected_identity["bytes"])
        if (
            portable.get(hash_field) != expected_hash
            or sha256_file(source) != expected_hash
            or source.stat().st_size != expected_bytes
        ):
            raise ValueError(
                f"Ambient-v2 {label} source hash is not the frozen reference."
            )
        portable[path_field] = expected_identity["path"]
        portable[f"{label}_bytes"] = expected_bytes
    portable.pop("coefficient_fingerprint_sha256", None)
    fingerprint = _json_fingerprint(portable)
    if fingerprint != AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256:
        raise ValueError(
            "Ambient-v2 portable response coefficient fingerprint drifted."
        )
    portable["coefficient_fingerprint_sha256"] = fingerprint
    return portable


def evaluate_postfreeze_analytical_gate(
    coefficients: Mapping[str, Any],
    *,
    figure10_path: Path | None = None,
    model_parameters_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate held-outs after coefficient freeze, without any retuning."""

    figure10 = (
        _reference_path("all_sc_figure10_peak_digitized.csv")
        if figure10_path is None
        else Path(figure10_path)
    )
    model_parameters = (
        _reference_path("vlastos2026_model_parameters.csv")
        if model_parameters_path is None
        else Path(model_parameters_path)
    )
    peaks = _load_figure10_peaks(figure10)
    cases = load_structural_cases(model_parameters)
    frozen = dict(coefficients["frozen_parameters"])
    rows = []
    for sc in HELD_OUT_SC:
        target, uncertainty = peaks[(sc, "hgast_hartlen_currie")]
        u_star = cases[sc].u_star_max_figure_digitized
        qblade_implied_sc = derived_scruton_number(cases[sc])
        prediction = float(
            analytical_equilibrium_amplitude_ratio(
                qblade_implied_sc,
                [u_star],
                aerodynamic_damping_gain_h1=float(frozen["aerodynamic_damping_gain_h1"]),
                saturation_amplitude_ratio=float(frozen["saturation_amplitude_ratio"]),
                reduced_velocity_low=float(frozen["reduced_velocity_low"]),
                reduced_velocity_peak=float(frozen["reduced_velocity_peak"]),
                reduced_velocity_high=float(frozen["reduced_velocity_high"]),
            )[0]
        )
        tolerance = max(0.03, 0.15 * target)
        error = abs(prediction - target)
        rows.append(
            {
                "scruton_number": sc,
                "qblade_implied_scruton_number": qblade_implied_sc,
                "reference_peak_u_star": u_star,
                "postfreeze_prediction_peak_over_diameter": prediction,
                "hgast_reference_peak_over_diameter": target,
                "hgast_digitization_uncertainty": uncertainty,
                "absolute_error": error,
                "acceptance_tolerance": tolerance,
                "pass": error <= tolerance,
            }
        )
    passed = all(bool(row["pass"]) for row in rows)
    return {
        "schema": SCHEMA,
        "gate": "post-freeze analytical held-out screen",
        "coefficient_fingerprint_sha256": coefficients["coefficient_fingerprint_sha256"],
        "coefficients_changed_after_held_out_access": False,
        "figure10_path": figure10.resolve().as_posix(),
        "figure10_sha256": sha256_file(figure10),
        "status": "pass" if passed else "fail",
        "pass": passed,
        "comparisons": rows,
        "decision": (
            "simple response closure passed the pre-QBlade held-out gate"
            if passed
            else "simple response closure failed without retuning; a phase-continuous wake oscillator is the next model rung"
        ),
    }


def portable_ambient_v2_analytical_gate(
    gate: Mapping[str, Any],
    *,
    figure10_path: Path | None = None,
) -> dict[str, Any]:
    """Canonicalize only the v2 Figure 10 path while retaining its identity."""

    portable = json.loads(json.dumps(dict(gate), allow_nan=False))
    source = (
        _reference_path("all_sc_figure10_peak_digitized.csv")
        if figure10_path is None
        else Path(figure10_path)
    )
    expected = stationary_protocol.AMBIENT_V2_REFERENCE_IDENTITIES[
        "all_sc_figure10_peak_digitized.csv"
    ]
    if (
        portable.get("figure10_sha256") != expected["sha256"]
        or sha256_file(source) != expected["sha256"]
        or source.stat().st_size != expected["bytes"]
    ):
        raise ValueError("Ambient-v2 Figure 10 identity is not frozen.")
    portable["figure10_path"] = expected["path"]
    portable["figure10_bytes"] = expected["bytes"]
    return portable


def write_frozen_coefficients(path: Path, coefficients: Mapping[str, Any]) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != dict(coefficients):
            raise FileExistsError(f"Refusing to replace different frozen coefficients: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(coefficients), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def target_bending_stiffness(case: StructuralCase) -> float:
    omega = 2.0 * math.pi * case.natural_frequency_hz
    beta = fixed_wake_case.EULER_BERNOULLI_BETA1
    return case.equivalent_mass_kg_per_m * (omega * HEIGHT_M**2 / beta**2) ** 2


def write_beam_structure(path: Path, span_fraction: np.ndarray, case: StructuralCase) -> None:
    ei = target_bending_stiffness(case)
    rayleigh_beta = case.damping_ratio / (math.pi * case.natural_frequency_hz)
    header = [
        f"{rayleigh_beta:.12e} RAYLEIGHDMP - beta=zeta/(pi fn) [s]",
        "1.0 STIFFTUNER",
        "1.0 MASSTUNER",
        "0 INTPTYPE - linear interpolation",
        "1 BEAMTYPE - Timoshenko",
        "2 DISCTYPE - structural stations",
        f"{len(span_fraction)} DISC",
        "LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] EA_[N] GJ_[N.m^2] GA_[N] "
        "STRPIT_[deg] KSX_[-] KSY_[-] RGX_[-] RGY_[-] XCM_[-] YCM_[-] XCE_[-] YCE_[-] XCS_[-] YCS_[-]",
    ]
    rows = []
    for station in span_fraction:
        values = (
            station,
            case.equivalent_mass_kg_per_m,
            ei,
            ei,
            2.0e7,
            90.0,
            7.0e6,
            0.0,
            0.85,
            0.85,
            0.25,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        rows.append(" ".join(f"{value:.12e}" for value in values))
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def write_simulation(
    path: Path,
    template: Path,
    *,
    case_id: str,
    natural_frequency_hz: float,
    reduced_velocity: float,
    timestep_s: float,
    num_steps: int,
) -> None:
    speed = reduced_velocity * natural_frequency_hz * DIAMETER_M
    text = fixed_wake_case.patch_keywords(
        template.read_text(encoding="utf-8"),
        {
            "OBJECTNAME": case_id,
            "TURBFILE": f"model/{fixed_wake_case.CASE_NAME}.trb",
            "TURBNAME": case_id,
            "INITIAL_YAW": 0.0,
            "INITIAL_PITCH": 0.0,
            "INITIAL_AZIMUTH": 0.0,
            "STRSUBSTEP": 2,
            "RELAXSTEPS": 0,
            "PRESCRIBETYPE": 1,
            "RPMPRESCRIBED": 0.0,
            "INCLUDEAERO": 0,
            "TIMESTEP": timestep_s,
            "NUMTIMESTEPS": num_steps,
            "RAMPUP": 0.0,
            "ADDDAMP": 0.0,
            "WNDTYPE": 0,
            "WNDNAME": "",
            "MEANINF": speed,
            "HORANGLE": 90.0,
            "VERTANGLE": 0.0,
            "PROFILETYPE": 0,
            "SHEAREXP": 0.0,
            "DENSITYAIR": fixed_wake_case.AIR_DENSITY_KGPM3,
            "VISCOSITYAIR": fixed_wake_case.AIR_KINEMATIC_VISCOSITY_M2PS,
            "STOREFROM": 0.0,
            "STOREREPLAY": "false",
            "STOREAERO": "false",
            "STOREBLADE": "false",
            "STORESTRUCT": "true",
            "STORESIM": "true",
            "STOREHYDRO": "false",
            "STORECONTROLLER": "false",
            "CALCMODAL": "false",
        },
    )
    path.write_text(text, encoding="utf-8")


def response_model_config(
    case: StructuralCase,
    node_ids: Sequence[str],
    mode_shape: Sequence[float],
    coefficients: Mapping[str, Any],
    *,
    response_schema_version: int = 1,
) -> dict[str, Any]:
    if (
        isinstance(response_schema_version, bool)
        or response_schema_version not in {1, 2}
    ):
        raise ValueError("response_schema_version must be integer 1 or 2.")
    frozen = coefficients["frozen_parameters"]
    if len(node_ids) < 2:
        raise ValueError("The cantilever response model requires at least two structural nodes.")
    tributary_length = np.full(len(node_ids), HEIGHT_M / (len(node_ids) - 1), dtype=float)
    tributary_length[[0, -1]] *= 0.5
    generalized_mass_weights = case.equivalent_mass_kg_per_m * tributary_length
    config = {
        "type": "modal_negative_damping",
        "schema_version": response_schema_version,
        "node_ids": [str(value) for value in node_ids],
        "mode_shape": [float(value) for value in mode_shape],
        "natural_frequency_hz": case.natural_frequency_hz,
        "reference_length_m": DIAMETER_M,
        "reduced_velocity_low": float(frozen["reduced_velocity_low"]),
        "reduced_velocity_peak": float(frozen["reduced_velocity_peak"]),
        "reduced_velocity_high": float(frozen["reduced_velocity_high"]),
        "aerodynamic_damping_gain_h1": float(frozen["aerodynamic_damping_gain_h1"]),
        "saturation_amplitude_ratio": float(frozen["saturation_amplitude_ratio"]),
        "rms_time_constant_s": float(frozen["rms_time_constant_cycles"]) / case.natural_frequency_hz,
        "bandpass_damping_ratio": float(frozen["bandpass_damping_ratio"]),
        "projection_weights": generalized_mass_weights.tolist(),
        "max_abs_gain": float(frozen["max_abs_gain"]),
        "max_abs_nodal_force_n": float(frozen["max_abs_nodal_force_n"]),
        "startup_seed": {
            "schema_version": 1,
            "enabled": True,
            "duration_cycles": 5.0,
            "target_amplitude_ratio": 0.005,
            "structural_damping_ratio": case.damping_ratio,
            "taper": "hann_sine_squared",
            "carrier": "natural_frequency",
            "nodal_weighting": "projection_mass_times_mode",
            "nodal_force_limit_source": (
                "response_model_max_abs_nodal_force_n"
            ),
        },
        # VorLap normal is -X_g in this zero-pitch/zero-azimuth HAWT mapping;
        # the sign of the normalized modal basis is invariant, so this excites
        # the intended QBlade X_b/global-X transverse response.
        "force_direction": "normal",
        "enabled": True,
    }
    if response_schema_version == 2:
        config["aerodynamic_speed_basis"] = (
            AMBIENT_V2_AERODYNAMIC_SPEED_BASIS
        )
    return config


def ambient_v2_equation_reproduction_spec(
    calibration_case: StructuralCase,
    coefficients: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable analytical target and estimator gates for v2.

    The values are derived from the generated QBlade mass/damping case and the
    already-frozen coefficient object.  Any drift fails generation instead of
    silently recalibrating the response closure.
    """

    qblade_scruton = derived_scruton_number(calibration_case)
    if qblade_scruton != AMBIENT_V2_CALIBRATION_QBLADE_SCRUTON_NUMBER:
        raise ValueError(
            "Generated Sc=5.18 calibration case no longer reproduces the "
            "frozen QBlade Scruton number."
        )
    fingerprint = str(coefficients.get("coefficient_fingerprint_sha256", ""))
    if fingerprint != AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256:
        raise ValueError(
            "Ambient-v2 generation requires the frozen response coefficient fingerprint."
        )
    frozen = dict(coefficients["frozen_parameters"])
    analytical = analytical_equilibrium_amplitude_ratio(
        qblade_scruton,
        DEFAULT_REDUCED_VELOCITIES,
        aerodynamic_damping_gain_h1=float(
            frozen["aerodynamic_damping_gain_h1"]
        ),
        saturation_amplitude_ratio=float(frozen["saturation_amplitude_ratio"]),
        reduced_velocity_low=float(frozen["reduced_velocity_low"]),
        reduced_velocity_peak=float(frozen["reduced_velocity_peak"]),
        reduced_velocity_high=float(frozen["reduced_velocity_high"]),
    )
    active_grid = tuple(
        float(u_star)
        for u_star, amplitude in zip(DEFAULT_REDUCED_VELOCITIES, analytical)
        if float(amplitude) >= AMBIENT_V2_ACTIVE_AMPLITUDE_RATIO
    )
    if active_grid != AMBIENT_V2_EXPECTED_ACTIVE_REDUCED_VELOCITIES:
        raise ValueError(
            "Frozen analytical response no longer has the declared active U* grid."
        )
    table = [
        {
            "reduced_velocity": float(u_star),
            "analytical_equilibrium_amplitude_over_diameter": float(amplitude),
        }
        for u_star, amplitude in zip(DEFAULT_REDUCED_VELOCITIES, analytical)
    ]
    return {
        "schema": AMBIENT_V2_EQUATION_GATE_SCHEMA,
        "response_model_schema_version": 2,
        "aerodynamic_speed_basis": AMBIENT_V2_AERODYNAMIC_SPEED_BASIS,
        "calibration_published_scruton_label": CALIBRATION_SC,
        "calibration_qblade_implied_scruton_number": qblade_scruton,
        "scruton_source": (
            "generated case equivalent mass, damping ratio, campaign air "
            "density, and diameter; no Scruton retuning"
        ),
        "equation": (
            "A/D=a_sat*sqrt(max(0,1-2*Sc_QBlade/"
            "(Ustar*H1*activation)))"
        ),
        "coefficient_fingerprint_sha256": fingerprint,
        "frozen_parameters": frozen,
        "startup_seed": {
            "schema_version": 1,
            "duration_cycles": 5.0,
            "target_amplitude_ratio": 0.005,
            "taper": "hann_sine_squared",
            "carrier": "natural_frequency",
        },
        "limit_caps": {
            "max_abs_gain": float(frozen["max_abs_gain"]),
            "max_abs_nodal_force_n": float(frozen["max_abs_nodal_force_n"]),
        },
        "stationary_cycle_caps_sha256": _json_fingerprint(
            {
                str(sc): {
                    str(u_star): cycles
                    for u_star, cycles in sorted(rows.items())
                }
                for sc, rows in sorted(
                    stationary_protocol.FROZEN_MAXIMUM_CYCLES.items()
                )
            }
        ),
        "active_threshold_amplitude_over_diameter": (
            AMBIENT_V2_ACTIVE_AMPLITUDE_RATIO
        ),
        "expected_active_reduced_velocities": list(
            AMBIENT_V2_EXPECTED_ACTIVE_REDUCED_VELOCITIES
        ),
        "fine_estimator_rmse_tolerance_over_diameter": (
            AMBIENT_V2_FINE_ESTIMATOR_RMSE_TOLERANCE
        ),
        "fine_estimator_maximum_absolute_error_tolerance_over_diameter": (
            AMBIENT_V2_FINE_ESTIMATOR_MAX_ABS_TOLERANCE
        ),
        "coarse_fine_estimator_maximum_absolute_difference_tolerance_over_diameter": (
            AMBIENT_V2_COARSE_FINE_ESTIMATOR_TOLERANCE
        ),
        "analytical_table": table,
    }


def build_campaign_spec(
    *,
    structural_cases: Mapping[float, StructuralCase],
    reduced_velocities: Sequence[float],
    timestep_s: float,
    include_dt_halving: bool,
    duration_s: float | None,
    case_id_namespace: str = "",
) -> list[dict[str, Any]]:
    if not math.isfinite(timestep_s) or timestep_s <= 0.0:
        raise ValueError("timestep_s must be finite and positive.")
    if duration_s is not None and (
        not math.isfinite(duration_s) or duration_s <= 0.0
    ):
        raise ValueError("duration_s must be finite and positive.")
    velocities = tuple(float(value) for value in reduced_velocities)
    if not velocities or any(not math.isfinite(value) or value <= 0.0 for value in velocities):
        raise ValueError("Reduced velocities must be finite and positive.")
    if not isinstance(case_id_namespace, str):
        raise TypeError("case_id_namespace must be a string.")
    if case_id_namespace and (
        not case_id_namespace.replace("_", "").isalnum()
        or case_id_namespace.lower() != case_id_namespace
    ):
        raise ValueError(
            "case_id_namespace must contain only lowercase letters, digits, and underscores."
        )
    case_id_prefix = f"{case_id_namespace}_" if case_id_namespace else ""
    levels = (("dt", timestep_s), ("dt_half", 0.5 * timestep_s)) if include_dt_halving else (("dt", timestep_s),)
    entries: list[dict[str, Any]] = []
    for sc in sorted(structural_cases):
        case = structural_cases[sc]
        for u_star in velocities:
            for timestep_level, dt in levels:
                steps_per_cycle = 1.0 / (case.natural_frequency_hz * dt)
                if steps_per_cycle + 1.0e-12 < MINIMUM_GLOBAL_STEPS_PER_CYCLE:
                    raise ValueError(
                        f"Sc={sc:.2f}, {timestep_level} has only {steps_per_cycle:.3f} global steps/cycle; "
                        f"at least {MINIMUM_GLOBAL_STEPS_PER_CYCLE:.0f} are required."
                    )
                case_id = (
                    f"{case_id_prefix}sc{_float_tag(sc)}_"
                    f"u{_float_tag(u_star)}_{timestep_level}"
                )
                if duration_s is None:
                    if not math.isclose(
                        timestep_s,
                        stationary_protocol.COARSE_TIMESTEP_S,
                        rel_tol=0.0,
                        abs_tol=1.0e-15,
                    ):
                        raise ValueError(
                            "Stationary-duration generation requires the frozen "
                            f"nominal timestep {stationary_protocol.COARSE_TIMESTEP_S:g} s. "
                            "Supply an explicit duration for a diagnostic campaign."
                        )
                    try:
                        duration_bound = stationary_protocol.derive_duration_bound(
                            sc, u_star
                        )
                    except stationary_protocol.ProtocolError as exc:
                        raise ValueError(
                            "Stationary-duration generation is defined only for the "
                            "frozen 3x12 Vlastos grid; supply an explicit duration for "
                            "a diagnostic campaign."
                        ) from exc
                    endpoint = duration_bound["maximum_endpoint"]
                    num_steps = int(endpoint[f"{timestep_level}_steps"])
                    actual_duration_s = float(endpoint["endpoint_time_s"])
                    maximum_cycles = int(duration_bound["maximum_cycles"])
                    duration_policy = "frozen_stationary_cycle_cap"
                else:
                    num_steps = int(math.ceil(duration_s / dt))
                    actual_duration_s = num_steps * dt
                    maximum_cycles = None
                    duration_policy = "explicit_fixed_diagnostic"
                entries.append(
                    {
                        "case_id": case_id,
                        "scruton_number": sc,
                        "qblade_implied_scruton_number": derived_scruton_number(case),
                        "scruton_relative_difference_percent": 100.0
                        * (derived_scruton_number(case) - sc)
                        / sc,
                        "role": "calibration" if sc == CALIBRATION_SC else "held_out",
                        "reduced_velocity": u_star,
                        "mean_inflow_mps": u_star * case.natural_frequency_hz * DIAMETER_M,
                        "reynolds_number": u_star
                        * case.natural_frequency_hz
                        * DIAMETER_M**2
                        / fixed_wake_case.AIR_KINEMATIC_VISCOSITY_M2PS,
                        "timestep_level": timestep_level,
                        "timestep_s": dt,
                        "steps_per_natural_cycle": steps_per_cycle,
                        "num_steps": num_steps,
                        "duration_s": actual_duration_s,
                        "duration_policy": duration_policy,
                        "stationary_maximum_cycles": maximum_cycles,
                        "structure": asdict(case),
                    }
                )
    return entries


def _manifest_file_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "campaign_manifest.json" or path.name.endswith(".vorlap.bak"):
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def generate_campaign(
    output: Path,
    *,
    reduced_velocities: Sequence[float] = DEFAULT_REDUCED_VELOCITIES,
    timestep_s: float = DEFAULT_TIMESTEP_S,
    include_dt_halving: bool = True,
    duration_s: float | None = None,
    selected_scruton_numbers: Sequence[float] = tuple(PUBLISHED_STRUCTURAL_CASES),
    response_schema_version: int = 1,
) -> Path:
    if (
        isinstance(response_schema_version, bool)
        or response_schema_version not in {1, 2}
    ):
        raise ValueError("response_schema_version must be integer 1 or 2.")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing campaign directory: {output}")
    all_cases = load_structural_cases()
    selected = {float(sc): all_cases[float(sc)] for sc in selected_scruton_numbers}
    coefficients = calibrate_response_coefficients()
    if response_schema_version == 2:
        coefficients = portable_ambient_v2_coefficients(coefficients)
    equation_reproduction_spec = None
    if response_schema_version == 2:
        if CALIBRATION_SC not in selected:
            raise ValueError(
                "Ambient-v2 campaign generation requires the Sc=5.18 calibration case."
            )
        equation_reproduction_spec = ambient_v2_equation_reproduction_spec(
            selected[CALIBRATION_SC],
            coefficients,
        )
    entries = build_campaign_spec(
        structural_cases=selected,
        reduced_velocities=reduced_velocities,
        timestep_s=timestep_s,
        include_dt_halving=include_dt_halving,
        duration_s=duration_s,
        case_id_namespace=(
            AMBIENT_V2_CASE_ID_NAMESPACE
            if response_schema_version == 2
            else ""
        ),
    )
    stationary_duration_mode = duration_s is None
    full_stationary_grid = (
        stationary_duration_mode
        and include_dt_halving
        and math.isclose(
            timestep_s,
            stationary_protocol.COARSE_TIMESTEP_S,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and tuple(sorted(selected)) == tuple(sorted(stationary_protocol.STRUCTURES))
        and tuple(float(value) for value in reduced_velocities)
        == stationary_protocol.REDUCED_VELOCITIES
    )

    output.mkdir(parents=True)
    if response_schema_version == 2:
        reference_output = output / "reference_data"
        reference_output.mkdir()
        for filename, identity in (
            stationary_protocol.AMBIENT_V2_REFERENCE_IDENTITIES.items()
        ):
            source = _reference_path(filename)
            if (
                sha256_file(source) != identity["sha256"]
                or source.stat().st_size != identity["bytes"]
            ):
                raise ValueError(
                    f"Ambient-v2 reference source identity changed: {filename}."
                )
            shutil.copyfile(source, reference_output / filename)
        result_filter_source = HERE / "cantilever_results.filter"
        frozen_filter = stationary_protocol.AMBIENT_V2_RESULT_FILTER_IDENTITY
        if (
            sha256_file(result_filter_source) != frozen_filter["sha256"]
            or result_filter_source.stat().st_size != frozen_filter["bytes"]
        ):
            raise ValueError("Ambient-v2 source result filter is not frozen.")
        shutil.copyfile(
            result_filter_source,
            output / "cantilever_results.filter",
        )
    coefficient_path = write_frozen_coefficients(output / "frozen_response_coefficients.json", coefficients)
    # Only after the coefficient file is frozen and hashed may held-out
    # Figure 10 targets be read.  A failure is retained as evidence and never
    # triggers an automatic retune.
    analytical_gate = evaluate_postfreeze_analytical_gate(coefficients)
    if response_schema_version == 2:
        analytical_gate = portable_ambient_v2_analytical_gate(analytical_gate)
        if analytical_gate["status"] != "fail" or analytical_gate["pass"] is not False:
            raise ValueError(
                "Ambient-v2 generation requires the frozen post-freeze analytical failure."
            )
    analytical_gate_path = output / "postfreeze_analytical_gate.json"
    analytical_gate_path.write_text(
        json.dumps(analytical_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    span = np.linspace(0.0, HEIGHT_M, N_STATIONS)
    span_fraction = span / HEIGHT_M
    mode_shape = fixed_wake_case.first_mode(span_fraction)
    # Runtime component selection canonicalizes QBlade action positions to six
    # decimals even though the structural file retains ten.  The response
    # model is controller-side and must use that canonical order exactly.
    node_ids = [f"BLD_1_{value:.6f}" for value in span_fraction]
    sim_template = REPO_ROOT / "examples" / "ReferenceHVAWTQBladeModel" / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
    trb_template = REPO_ROOT / "examples" / "ReferenceHVAWTQBladeModel" / "10m_HVAWT_reference_turbine" / "10m_HVAWT_reference_turbine.trb"

    for entry in entries:
        case = selected[float(entry["scruton_number"])]
        case_root = output / "cases" / str(entry["case_id"])
        model_root = case_root / "model"
        aero_dir = model_root / "Aero"
        structure_dir = model_root / "Structure"
        source_dir = model_root / "VorLapAirfoils"
        for directory in (aero_dir / "Polars", aero_dir / "Airfoils", structure_dir, source_dir):
            directory.mkdir(parents=True, exist_ok=True)

        fixed_wake_case.write_blade(aero_dir / f"{fixed_wake_case.CASE_NAME}.bld", span)
        fixed_wake_case.write_airfoil(aero_dir / "Airfoils" / "CYLINDER.afl")
        fixed_wake_case.write_polar(aero_dir / "Polars" / "CYLINDER.plr")
        write_beam_structure(structure_dir / "cantilever_beam.str", span_fraction, case)
        fixed_wake_case.write_rigid_tower(structure_dir / "rigid_tower.str")
        fixed_wake_case.write_main_structure(structure_dir / "cantilever_main.str", span_fraction)
        fixed_wake_case.write_source(source_dir / "CYLINDER.h5")
        with h5py.File(source_dir / "CYLINDER.h5", "a") as source:
            source.attrs["response_closure"] = (
                "frozen-coefficient response-aware single-mode aerodynamic damping"
            )
            source.attrs["coefficient_fingerprint_sha256"] = coefficients["coefficient_fingerprint_sha256"]
        fixed_wake_case.write_turbine(
            model_root / f"{fixed_wake_case.CASE_NAME}.trb",
            trb_template,
            object_name=str(entry["case_id"]),
        )
        sim_path = case_root / f"{entry['case_id']}.sim"
        write_simulation(
            sim_path,
            sim_template,
            case_id=str(entry["case_id"]),
            natural_frequency_hz=case.natural_frequency_hz,
            reduced_velocity=float(entry["reduced_velocity"]),
            timestep_s=float(entry["timestep_s"]),
            num_steps=int(entry["num_steps"]),
        )
        response_config = response_model_config(
            case,
            node_ids,
            mode_shape,
            coefficients,
            response_schema_version=response_schema_version,
        )
        prepared = prepare_qblade_external_case(
            QBladeCasePrepRequest(
                sim_path=sim_path,
                airfoil_dir=source_dir,
                include_struts=False,
                include_tower=False,
                node_source="structural",
                n_freq_depth=2,
                # The persistent CL0=0.3 fixed-wake tone belongs to the
                # separate legacy frozen-phase bridge comparator. This
                # campaign uses only the response closure plus its finite
                # five-cycle modal seed.
                force_scale=0.0,
                include_mean_load=False,
                source_bounds_policy="raise",
                source_frequency_cap_hz=0.5 / float(entry["timestep_s"]),
                response_model=response_config,
                write_geometry_audit=True,
                include_default_python_path=False,
                python_path_entries=("/work/VorLap",),
            )
        )
        components, params, converted_node_ids = convert_qblade_to_vorlap_inputs(
            str(sim_path), include_struts=False, include_tower=False
        )
        prepare_component_frames(components)
        converted_fractions = np.asarray(
            [float(str(node_id).rsplit("_", 1)[1]) for node_id in converted_node_ids],
            dtype=float,
        )
        if (
            len(components) != 1
            or len(converted_node_ids) != N_STATIONS
            or not np.allclose(converted_fractions, span_fraction, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"Case {entry['case_id']} did not retain the intended 21-node cantilever.")
        np.testing.assert_allclose(components[0].chord_vector[0], [0.0, 1.0, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(components[0].normal_vector[0], [-1.0, 0.0, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(params.inflow_vec, [0.0, 1.0, 0.0], atol=1.0e-12)
        runtime_config = json.loads(prepared.parameter_file_path.read_text(encoding="utf-8"))
        if runtime_config.get("response_model") != response_config:
            raise ValueError(f"Case {entry['case_id']} lost its response-model configuration during preparation.")
        entry["sim_path"] = sim_path.relative_to(output).as_posix()
        entry["parameter_file"] = prepared.parameter_file_path.relative_to(output).as_posix()
        entry["response_model_config"] = response_config
        entry["coefficient_fingerprint_sha256"] = coefficients["coefficient_fingerprint_sha256"]

    geometry_path = output / "cantilever_geometry_and_mode.csv"
    np.savetxt(
        geometry_path,
        np.column_stack((span, span_fraction, mode_shape)),
        delimiter=",",
        header="height_m,span_fraction,euler_bernoulli_first_mode_tip_normalized",
        comments="",
    )
    if response_schema_version == 2:
        reference_files = {
            name: dict(identity)
            for name, identity in (
                stationary_protocol.AMBIENT_V2_REFERENCE_IDENTITIES.items()
            )
        }
    else:
        reference_files = {
            name: {
                "path": _reference_path(name).resolve().as_posix(),
                "sha256": sha256_file(_reference_path(name)),
            }
            for name in (
                "sc5p18_figure9_digitized.csv",
                "all_sc_figure10_peak_digitized.csv",
                "vlastos2026_model_parameters.csv",
            )
        }
    manifest = {
        "schema": SCHEMA,
        "scope": (
            "external hGAST response comparison and model-form assessment; not equation-level code "
            "verification, a resolved-wake model, or certification validation"
        ),
        "reference": {
            "doi": REFERENCE_DOI,
            "experiment_doi": EXPERIMENT_DOI,
            "height_m": HEIGHT_M,
            "diameter_m": DIAMETER_M,
            "reference_files": reference_files,
            "figure9_publisher_sha256": "e20f72c24c431b064615883965ef0f20d0f69b91bdefddf1c1ec52ef9cf9ee91",
            "table1_publisher_sha256": "d24a65f63a33783ee6d69cc239915affa0c2cb6a43eb180cd1ee9db2ee7c4a22",
        },
        "calibration_policy": {
            "calibration_scruton_number": CALIBRATION_SC,
            "held_out_scruton_numbers": list(HELD_OUT_SC),
            "coefficient_file": coefficient_path.relative_to(output).as_posix(),
            "coefficient_file_sha256": sha256_file(coefficient_path),
            "coefficient_fingerprint_sha256": coefficients["coefficient_fingerprint_sha256"],
            "held_out_targets_accessed_during_calibration": False,
            "postfreeze_analytical_gate_file": analytical_gate_path.relative_to(output).as_posix(),
            "postfreeze_analytical_gate_sha256": sha256_file(analytical_gate_path),
            "postfreeze_analytical_gate_status": analytical_gate["status"],
            "coefficients_changed_after_held_out_access": False,
        },
        "numerics": {
            "minimum_global_steps_per_cycle": MINIMUM_GLOBAL_STEPS_PER_CYCLE,
            "dt_halving_enabled": include_dt_halving,
            "duration_requested_s": duration_s,
            "duration_policy": (
                "frozen_stationary_cycle_caps"
                if stationary_duration_mode
                else "explicit_fixed_diagnostic_duration"
            ),
            "stationary_protocol_schema": (
                stationary_protocol.PROTOCOL_SCHEMA
                if stationary_duration_mode
                else None
            ),
            "stationary_protocol_compatible_full_grid": full_stationary_grid,
            "qualification_input_eligible": full_stationary_grid,
            "required_sil_vorlap_history_mode": "compact",
            "response_window_cycles": 40.0,
        },
        "qblade_mapping": {
            "representation": "one isolated HAWT blade beam used as the horizontal cantilever cylinder",
            "inflow_direction": "+Y_g/+Y_b chord",
            "response_direction": "global +X; QBlade X_b tip deflection; VorLap normal is -X",
            "native_aerodynamics": False,
            "persistent_fixed_wake_source": "disabled with force_scale=0",
            "startup_seed": {
                "duration_cycles": 5.0,
                "target_amplitude_ratio": 0.005,
                "waveform": "hann_sine_squared at the natural frequency",
                "nodal_weighting": "projection_mass_times_mode",
                "exactly_zero_after_seed": True,
            },
            "response_closure": (
                "frozen-coefficient response-aware single-mode aerodynamic damping"
            ),
            "scruton_mapping": {
                "definition": "Sc=4*pi*m_equivalent*zeta/(rho*D^2)",
                "decision": (
                    "match the published Table 1 equivalent mass, damping, and natural frequency exactly; "
                    "retain the existing case air density; report the resulting Sc mismatch explicitly"
                ),
                "air_density_kg_per_m3": fixed_wake_case.AIR_DENSITY_KGPM3,
                "not_an_exact_scruton_match": True,
                "reason_no_density_retune": (
                    "No single inferred density reconciles all three rounded Table 1 mass, damping, and Sc values, "
                    "so changing density would trade one undocumented mismatch for another."
                ),
                "published_sc_used_for_published_regressions": True,
                "qblade_implied_sc_used_for_response_energy_balance": True,
            },
        },
        "cases": entries,
        "files": _manifest_file_rows(output),
    }
    if response_schema_version == 2:
        if equation_reproduction_spec is None:  # pragma: no cover - branch invariant
            raise RuntimeError("Ambient-v2 equation gate was not constructed.")
        manifest["response_model_semantics"] = (
            stationary_protocol.ambient_v2_response_semantics()
        )
        manifest["equation_reproduction_gate"] = equation_reproduction_spec
        manifest["calibration_policy"].update(
            {
                "held_out_targets_used_for_calibration": False,
                "coefficients_changed_after_holdout_evaluation": False,
            }
        )
        manifest["runtime_contract"] = (
            stationary_protocol.ambient_v2_runtime_contract(output)
        )
    manifest_path = output / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if full_stationary_grid:
        # This is a read-only, fail-closed compatibility check.  The protocol
        # evidence itself is deliberately written outside the immutable input
        # tree by stationary_campaign_protocol.py.
        stationary_protocol.build_protocol(manifest_path)
    return manifest_path


def _validate_campaign_case_response_semantics(
    root: Path,
    manifest: Mapping[str, Any],
    response_semantics: Mapping[str, Any],
) -> None:
    """Bind every case ID, config, speed, width, and U* to one schema."""

    response_schema_version = int(response_semantics["response_schema_version"])
    ambient_v2 = bool(response_semantics["v2"])
    reserved_prefix = f"{AMBIENT_V2_CASE_ID_NAMESPACE}_"
    coefficient_fingerprint = str(
        manifest["calibration_policy"]["coefficient_fingerprint_sha256"]
    )
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Campaign manifest has no cases.")
    seen_ids: set[str] = set()
    calibration_count = 0
    held_out_count = 0
    for index, entry in enumerate(raw_cases):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Campaign case {index} is not an object.")
        case_id = str(entry.get("case_id", ""))
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Campaign case_id is missing or duplicated: {case_id!r}.")
        seen_ids.add(case_id)
        if ambient_v2:
            expected_case_id = stationary_protocol.ambient_v2_case_id(
                float(entry["scruton_number"]),
                float(entry["reduced_velocity"]),
                str(entry["timestep_level"]),
            )
            if case_id != expected_case_id:
                raise ValueError(
                    f"Schema-v2 case {case_id!r} does not match its canonical Sc/U*/timestep ID."
                )
            if not case_id.startswith(reserved_prefix):
                raise ValueError(
                    f"Schema-v2 case {case_id!r} is outside the reserved "
                    f"{AMBIENT_V2_CASE_ID_NAMESPACE!r} result/checkpoint namespace."
                )
        elif case_id.startswith(reserved_prefix):
            raise ValueError(
                f"Schema-v1 case {case_id!r} uses the reserved schema-v2 namespace."
            )
        role = str(entry.get("role", ""))
        if role == "calibration":
            calibration_count += 1
        elif role == "held_out":
            held_out_count += 1
        else:
            raise ValueError(f"Campaign case {case_id} has unsupported role {role!r}.")
        if entry.get("coefficient_fingerprint_sha256") != coefficient_fingerprint:
            raise ValueError(
                f"Campaign case {case_id} coefficient fingerprint is not frozen."
            )
        response_config = entry.get("response_model_config")
        if not isinstance(response_config, Mapping):
            raise ValueError(f"Campaign case {case_id} has no response-model object.")
        if response_config.get("schema_version") != response_schema_version:
            raise ValueError(
                f"Campaign case {case_id} response schema does not match its campaign."
            )
        if ambient_v2:
            if response_config.get("aerodynamic_speed_basis") != (
                AMBIENT_V2_AERODYNAMIC_SPEED_BASIS
            ):
                raise ValueError(
                    f"Campaign case {case_id} does not use declared ambient crossflow."
                )
            if "response_section_width_m" in response_config:
                raise ValueError(
                    f"Campaign case {case_id} must derive response width from "
                    "declared ambient crossflow."
                )
            structure = entry.get("structure")
            if not isinstance(structure, Mapping):
                raise ValueError(f"Campaign case {case_id} has no structure object.")
            expected_speed = (
                float(entry["reduced_velocity"])
                * float(structure["natural_frequency_hz"])
                * float(response_config["reference_length_m"])
            )
            if not math.isclose(
                float(entry["mean_inflow_mps"]),
                expected_speed,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    f"Campaign case {case_id} ambient speed does not reproduce its U*."
                )
        elif "aerodynamic_speed_basis" in response_config:
            raise ValueError(
                f"Campaign case {case_id} legacy response model gained a v2 speed basis."
            )
        parameter_relative = PurePosixPath(str(entry.get("parameter_file", "")))
        if (
            not str(parameter_relative)
            or parameter_relative.is_absolute()
            or ".." in parameter_relative.parts
        ):
            raise ValueError(f"Campaign case {case_id} parameter path is unsafe.")
        parameter_path = root.joinpath(*parameter_relative.parts)
        runtime_config = json.loads(parameter_path.read_text(encoding="utf-8"))
        if runtime_config.get("response_model") != response_config:
            raise ValueError(
                f"Campaign case {case_id} manifest/runtime response models differ."
            )
        if ambient_v2:
            sim_relative = PurePosixPath(str(entry.get("sim_path", "")))
            if (
                not str(sim_relative)
                or sim_relative.is_absolute()
                or ".." in sim_relative.parts
            ):
                raise ValueError(f"Campaign case {case_id} simulation path is unsafe.")
            sim_path = root.joinpath(*sim_relative.parts)
            protocol_structure = {
                "natural_frequency_hz": float(structure["natural_frequency_hz"]),
                "equivalent_mass_kg_per_m": float(
                    structure["equivalent_mass_kg_per_m"]
                ),
                "structural_damping_ratio": float(structure["damping_ratio"]),
            }
            stationary_protocol._validate_runtime_mapping_config(
                runtime_config,
                config_path=parameter_path,
                sim_path=sim_path,
                timestep_s=float(entry["timestep_s"]),
                case_id=case_id,
            )
            stationary_protocol._validate_response_model(
                response_config,
                protocol_structure,
                case_id,
                response_schema_version=2,
            )
            stationary_protocol.validate_ambient_v2_airfoil_source(
                root=root,
                file_index=manifest["_verified_manifest_files"],
                case_id=case_id,
                parameter_file=str(entry["parameter_file"]),
                config=runtime_config,
                coefficient_fingerprint=coefficient_fingerprint,
            )
    if ambient_v2 and len(raw_cases) == 72:
        if calibration_count != 24 or held_out_count != 48:
            raise ValueError(
                "Full schema-v2 campaign must partition into 24 calibration "
                "and 48 held-out cases."
            )


def _load_campaign(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported campaign schema in {path}.")
    response_semantics = stationary_protocol.validate_response_semantics(manifest)
    if bool(response_semantics["v2"]):
        stationary_protocol.validate_ambient_v2_manifest_contract(manifest)
    root = path.parent
    verified_files: dict[str, dict[str, Any]] = {}
    for row in manifest.get("files", []):
        if not isinstance(row, Mapping):
            raise ValueError("Campaign manifest contains a non-object file identity.")
        relative_text = str(row.get("path", ""))
        relative = PurePosixPath(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in verified_files
        ):
            raise ValueError(
                f"Campaign manifest contains an unsafe or duplicate file path: {relative_text!r}."
            )
        actual = root.joinpath(*relative.parts)
        if not actual.is_file():
            raise ValueError(f"Campaign manifest file is unavailable: {relative_text}")
        expected_size = int(row.get("bytes", -1))
        actual_size = actual.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Campaign manifest file size changed for {relative_text}: "
                f"expected {expected_size}, observed {actual_size}."
            )
        expected_sha256 = str(row.get("sha256", ""))
        actual_sha256 = sha256_file(actual)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Campaign manifest file hash changed for {relative_text}: "
                f"expected {expected_sha256}, observed {actual_sha256}."
            )
        verified_files[relative_text] = {
            "path": relative_text,
            "sha256": expected_sha256,
            "size_bytes": expected_size,
            "bytes": expected_size,
        }
    if not verified_files:
        raise ValueError("Campaign manifest has no immutable file identities.")
    manifest["_verified_manifest_files"] = verified_files
    if bool(response_semantics["v2"]):
        stationary_protocol.validate_ambient_v2_campaign_tree(
            root, verified_files
        )
    coefficient_relative = str(
        manifest["calibration_policy"]["coefficient_file"]
    )
    if coefficient_relative not in verified_files:
        raise ValueError(
            "Frozen response-coefficient file is absent from the immutable file table."
        )
    coefficient_path, _ = stationary_protocol._safe_manifest_path(
        root, coefficient_relative, "Frozen response-coefficient file"
    )
    if sha256_file(coefficient_path) != manifest["calibration_policy"]["coefficient_file_sha256"]:
        raise ValueError("Frozen response-coefficient file hash does not match the campaign manifest.")
    coefficients = json.loads(coefficient_path.read_text(encoding="utf-8"))
    if coefficients["coefficient_fingerprint_sha256"] != manifest["calibration_policy"]["coefficient_fingerprint_sha256"]:
        raise ValueError("Frozen response-coefficient fingerprint does not match the campaign manifest.")
    if bool(response_semantics["v2"]):
        portable_coefficients = portable_ambient_v2_coefficients(coefficients)
        if portable_coefficients != coefficients:
            raise ValueError(
                "Ambient-v2 coefficient file is not the exact portable frozen object."
            )
    _validate_campaign_case_response_semantics(root, manifest, response_semantics)
    gate_relative = str(
        manifest["calibration_policy"]["postfreeze_analytical_gate_file"]
    )
    if gate_relative not in verified_files:
        raise ValueError(
            "Post-freeze analytical gate is absent from the immutable file table."
        )
    gate_path, _ = stationary_protocol._safe_manifest_path(
        root, gate_relative, "Post-freeze analytical gate"
    )
    if sha256_file(gate_path) != manifest["calibration_policy"]["postfreeze_analytical_gate_sha256"]:
        raise ValueError("Post-freeze analytical gate hash does not match the campaign manifest.")
    if bool(response_semantics["v2"]):
        reference_paths = stationary_protocol.validate_ambient_v2_reference_files(
            root, manifest, verified_files
        )
        stationary_protocol.validate_ambient_v2_runtime_contract(
            root, manifest, verified_files
        )
        stationary_protocol.validate_ambient_v2_postfreeze_gate(
            root=root,
            manifest=manifest,
            file_index=verified_files,
            coefficients=coefficients,
            reference_paths=reference_paths,
        )
        for name, reference in reference_paths.items():
            manifest["reference"]["reference_files"][name][
                "_resolved_path"
            ] = reference.as_posix()
    else:
        for name, identity in manifest["reference"]["reference_files"].items():
            recorded = Path(identity["path"])
            candidates = (recorded, _reference_path(recorded.name))
            reference = next(
                (
                    candidate.resolve()
                    for candidate in candidates
                    if candidate.is_file()
                    and sha256_file(candidate) == identity["sha256"]
                ),
                None,
            )
            if reference is None:
                raise ValueError(f"Reference identity changed or is unavailable: {name}")
            # Legacy absolute paths remain exact schema-v1 provenance.
            identity["_resolved_path"] = reference.as_posix()
    return root, manifest


def _resolved_reference_path(identity: Mapping[str, Any]) -> Path:
    return Path(str(identity.get("_resolved_path", identity["path"])))


_AMBIENT_V2_RUN_PLAN_KEYS = {
    "schema",
    "stage",
    "dry_plan",
    "results_root",
    "qblade_root",
    "checkpoint_every",
    "threads",
    "threads_per_qblade_run",
    "coefficient_fingerprint_sha256",
    "postfreeze_analytical_gate_status",
    "sil_vorlap_history_mode",
    "warning",
    "execution_intent",
    "commands",
    "response_model_schema_version",
    "aerodynamic_speed_basis",
    "case_id_namespace",
    "checkpoint_reuse_policy",
    "campaign_manifest_sha256",
    "response_semantics_sha256",
    "stationary_protocol_sha256",
    "runtime_identities",
}
_RUN_PLAN_COMMAND_KEYS = {"case_id", "role", "argv"}


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        Path(path).relative_to(Path(parent))
    except ValueError:
        return False
    return True


def _paths_overlap(first: Path, second: Path) -> bool:
    first = Path(first).resolve()
    second = Path(second).resolve()
    return _path_is_within(first, second) or _path_is_within(second, first)


def _source_workspace_roots() -> tuple[Path, ...]:
    """Return the source and enclosing workspace roots for write exclusion."""

    roots = {REPO_ROOT.resolve()}
    dot_git = REPO_ROOT / ".git"
    if dot_git.is_file():
        try:
            prefix, raw_path = dot_git.read_text(encoding="utf-8").strip().split(
                ":", 1
            )
            if prefix == "gitdir":
                git_dir = Path(raw_path.strip())
                if not git_dir.is_absolute():
                    git_dir = (REPO_ROOT / git_dir).resolve()
                if git_dir.parent.name == "worktrees":
                    main_checkout = git_dir.parent.parent.parent.resolve()
                    roots.update((main_checkout, main_checkout.parent))
        except (OSError, ValueError):
            pass
    elif dot_git.is_dir():
        roots.add(REPO_ROOT.parent.resolve())
    return tuple(sorted(roots, key=lambda value: value.as_posix()))


def _validate_v2_results_root(
    results_root: Path,
    *,
    campaign_root: Path,
    qblade_root: Path,
) -> None:
    resolved = Path(results_root).resolve()
    anchor = Path(resolved.anchor).resolve()
    if resolved == anchor:
        raise ValueError("Schema-v2 results_root cannot be a filesystem anchor.")
    forbidden_roots = [
        ("campaign", Path(campaign_root).resolve()),
        ("QBlade runtime", Path(qblade_root).resolve()),
        ("user home", Path.home().resolve()),
        *(
            ("VorLap source/workspace", root)
            for root in _source_workspace_roots()
        ),
    ]
    for label, forbidden in forbidden_roots:
        if _paths_overlap(resolved, forbidden):
            raise ValueError(
                "Schema-v2 results_root must neither be inside nor contain "
                f"the {label} root."
            )


def _campaign_tree_snapshot(root: Path) -> tuple[tuple[Any, ...], ...]:
    root = Path(root).resolve()
    rows: list[tuple[Any, ...]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            rows.append((relative, "symlink", str(path.readlink())))
        elif path.is_dir():
            rows.append((relative, "directory"))
        elif path.is_file():
            rows.append(
                (relative, "file", path.stat().st_size, sha256_file(path))
            )
        else:
            rows.append((relative, "other"))
    return tuple(rows)


def _portable_runtime_identity(
    path: Path, *, relative_to: Path, label: str
) -> dict[str, Any]:
    source = Path(path).resolve()
    identity = runtime_file_identity(source)
    if identity is None:
        raise FileNotFoundError(f"Pinned {label} is unavailable: {source}")
    try:
        relative = source.relative_to(Path(relative_to).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Pinned {label} escapes its declared runtime root.") from exc
    return {
        "path": relative,
        "sha256": identity["sha256"],
        "bytes": identity["size_bytes"],
    }


def _ambient_v2_plan_runtime_identities(
    campaign_root: Path,
    manifest: Mapping[str, Any],
    qblade_root: Path,
) -> dict[str, Any]:
    contract = manifest["runtime_contract"]
    qblade_library = _portable_runtime_identity(
        find_qblade_library(qblade_root),
        relative_to=qblade_root,
        label="QBlade library",
    )
    bridge_path = find_bridge_library(qblade_root, "libvorlap_qblade_bridge")
    if bridge_path is None:
        raise FileNotFoundError(
            f"Pinned VorLap/QBlade bridge library is unavailable in {qblade_root}."
        )
    bridge_library = _portable_runtime_identity(
        bridge_path,
        relative_to=qblade_root,
        label="VorLap/QBlade bridge library",
    )
    if qblade_library != stationary_protocol.AMBIENT_V2_QBLADE_LIBRARY_IDENTITY:
        raise ValueError("QBlade library differs from the frozen v2 runtime identity.")
    if bridge_library != stationary_protocol.AMBIENT_V2_BRIDGE_LIBRARY_IDENTITY:
        raise ValueError(
            "VorLap/QBlade bridge differs from the frozen v2 runtime identity."
        )
    qblade_dependencies, qblade_dependency_fingerprint = (
        qblade_runtime_dependency_manifest(qblade_root)
    )
    if (
        qblade_dependencies != contract["qblade_runtime_dependencies"]
        or qblade_dependency_fingerprint
        != contract["qblade_runtime_dependency_fingerprint"]
    ):
        raise ValueError(
            "QBlade dependency tree differs from the campaign runtime contract."
        )
    # Recompute rather than trusting the manifest's self-description.
    python_assets, python_fingerprint = vorlap_python_runtime_manifest()
    if (
        python_assets != contract["vorlap_python_runtime_assets"]
        or python_fingerprint != contract["vorlap_python_runtime_fingerprint"]
    ):
        raise ValueError("VorLap Python runtime differs from the campaign contract.")
    runner = REPO_ROOT / str(contract["runner"]["path"])
    if (
        sha256_file(runner) != contract["runner"]["sha256"]
        or runner.stat().st_size != contract["runner"]["bytes"]
    ):
        raise ValueError("QBlade runner differs from the campaign runtime contract.")
    result_filter = campaign_root / str(contract["result_filter"]["path"])
    if (
        sha256_file(result_filter) != contract["result_filter"]["sha256"]
        or result_filter.stat().st_size != contract["result_filter"]["bytes"]
    ):
        raise ValueError("Campaign-local result filter changed.")
    return {
        "runner": dict(contract["runner"]),
        "result_filter": dict(contract["result_filter"]),
        "vorlap_python_runtime_assets": list(python_assets),
        "vorlap_python_runtime_fingerprint": python_fingerprint,
        "qblade_library": qblade_library,
        "bridge_library": bridge_library,
        "qblade_sil_wrapper": dict(contract["qblade_sil_wrapper"]),
        "qblade_runtime_dependencies": list(qblade_dependencies),
        "qblade_runtime_dependency_fingerprint": (
            qblade_dependency_fingerprint
        ),
        "numerical_runtime": dict(contract["numerical_runtime"]),
    }


def _v2_stationary_protocol_sha256(
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> str | None:
    if manifest.get("numerics", {}).get("qualification_input_eligible") is not True:
        return None
    protocol = stationary_protocol.build_protocol(manifest_path)
    return str(protocol["protocol_sha256"])


def build_run_plan(
    manifest_path: Path,
    results_root: Path,
    *,
    stage: str,
    qblade_root: Path,
    checkpoint_every: int = 0,
    threads: int = 4,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    campaign_root, manifest = _load_campaign(manifest_path)
    if stage not in {"calibration", "held_out", "all"}:
        raise ValueError("stage must be calibration, held_out, or all.")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads < 8:
        raise ValueError("threads must be an integer from 1 through 7.")
    if (
        isinstance(checkpoint_every, bool)
        or not isinstance(checkpoint_every, int)
        or checkpoint_every < 0
    ):
        raise ValueError("checkpoint_every must be an integer greater than or equal to zero.")
    if not isinstance(diagnostic_only, bool):
        raise TypeError("diagnostic_only must be a boolean.")
    response_semantics = stationary_protocol.validate_response_semantics(manifest)
    raw_results_root = Path(results_root).expanduser()
    results_root = raw_results_root.resolve()
    runner = REPO_ROOT / "run_headless" / "run_qblade_sim.py"
    result_filter = HERE / "cantilever_results.filter"
    resolved_qblade_root = Path(qblade_root).expanduser().resolve()
    if bool(response_semantics["v2"]):
        if raw_results_root.is_symlink():
            raise ValueError("Schema-v2 results_root cannot be a symlink.")
        if manifest.get("numerics", {}).get("qualification_input_eligible") is not True:
            raise ValueError(
                "Schema-v2 execution requires the immutable full 72-case qualification input grid."
            )
        _validate_v2_results_root(
            results_root,
            campaign_root=campaign_root,
            qblade_root=resolved_qblade_root,
        )
        if stage in {"held_out", "all"} and not diagnostic_only:
            raise ValueError(
                "Schema-v2 held-out/all execution is diagnostic-only after the frozen "
                "analytical gate failure; set diagnostic_only=True explicitly."
            )
        runner = REPO_ROOT / str(manifest["runtime_contract"]["runner"]["path"])
        result_filter = campaign_root / str(
            manifest["runtime_contract"]["result_filter"]["path"]
        )
        runtime_identities = _ambient_v2_plan_runtime_identities(
            campaign_root, manifest, resolved_qblade_root
        )
    else:
        runtime_identities = None
    commands = []
    for entry in manifest["cases"]:
        if stage != "all" and entry["role"] != stage:
            continue
        case_output = results_root / entry["case_id"]
        command = [
            sys.executable,
            str(runner),
            "--sim",
            str(campaign_root / entry["sim_path"]),
            "--qblade-root",
            str(resolved_qblade_root),
            "--output-dir",
            str(case_output),
            "--result-name",
            "result",
            "--result-filter",
            str(result_filter),
            "--threads",
            str(threads),
            "--sil-vorlap",
            "--coupling-mode",
            "sil-vorlap",
            "--sil-vorlap-history-mode",
            "compact",
        ]
        if checkpoint_every > 0:
            command.extend(("--checkpoint-every", str(checkpoint_every), "--checkpoint-dir", str(case_output / "checkpoints")))
        commands.append({"case_id": entry["case_id"], "role": entry["role"], "argv": command})
    plan = {
        "schema": SCHEMA,
        "stage": stage,
        "dry_plan": True,
        "coefficient_fingerprint_sha256": manifest["calibration_policy"]["coefficient_fingerprint_sha256"],
        "postfreeze_analytical_gate_status": manifest["calibration_policy"]["postfreeze_analytical_gate_status"],
        "threads_per_qblade_run": threads,
        "sil_vorlap_history_mode": "compact",
        "warning": (
            "The simple response closure failed its post-freeze held-out analytical gate; "
            "commands are retained for diagnostic investigation and must not be presented as an accepted model."
            if manifest["calibration_policy"]["postfreeze_analytical_gate_status"] != "pass"
            else None
        ),
        "commands": commands,
    }
    if bool(response_semantics["v2"]):
        plan = {
            **plan,
            "schema": AMBIENT_V2_RUN_PLAN_SCHEMA,
            "results_root": str(results_root),
            "qblade_root": str(resolved_qblade_root),
            "checkpoint_every": checkpoint_every,
            "threads": threads,
            "execution_intent": (
                "diagnostic_only_after_frozen_gate_failure"
                if stage in {"held_out", "all"}
                else "calibration_stage"
            ),
            "response_model_schema_version": 2,
            "aerodynamic_speed_basis": AMBIENT_V2_AERODYNAMIC_SPEED_BASIS,
            "case_id_namespace": AMBIENT_V2_CASE_ID_NAMESPACE,
            "checkpoint_reuse_policy": AMBIENT_V2_CHECKPOINT_REUSE_POLICY,
            "campaign_manifest_sha256": sha256_file(
                Path(manifest_path).expanduser().resolve()
            ),
            "response_semantics_sha256": response_semantics[
                "response_semantics_sha256"
            ],
            "stationary_protocol_sha256": _v2_stationary_protocol_sha256(
                Path(manifest_path).expanduser().resolve(), manifest
            ),
            "runtime_identities": runtime_identities,
        }
    return plan


def _preflight_response_runtime(parameter_file: Path) -> None:
    from vorlap.qblade_runtime import VorLapQBladeRuntime

    runtime = VorLapQBladeRuntime.from_qblade_config(str(parameter_file))
    model = getattr(runtime.controller, "response_model", None)
    enabled = bool(getattr(runtime.controller, "response_model_enabled", False))
    if model is None or not enabled or not bool(model.config.enabled):
        raise RuntimeError(
            "The installed VorLap runtime does not consume response_model from the prepared QBlade config; "
            "refusing to mislabel a legacy frozen-phase bridge run as response-aware."
        )


def _validate_ambient_v2_run_plan_shape(
    plan: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(plan, dict) or set(plan) != _AMBIENT_V2_RUN_PLAN_KEYS:
        unexpected = sorted(set(plan).difference(_AMBIENT_V2_RUN_PLAN_KEYS))
        missing = sorted(_AMBIENT_V2_RUN_PLAN_KEYS.difference(plan))
        raise RuntimeError(
            "Schema-v2 run plan key schema differs; "
            f"unexpected={unexpected}, missing={missing}."
        )
    supplied = dict(plan)
    if supplied.get("schema") != AMBIENT_V2_RUN_PLAN_SCHEMA:
        raise RuntimeError("Unsupported schema-v2 run-plan schema.")
    if supplied.get("dry_plan") is not True:
        raise RuntimeError("Schema-v2 run plan must retain dry_plan=true provenance.")
    if supplied.get("stage") not in {"calibration", "held_out", "all"}:
        raise RuntimeError("Schema-v2 run-plan stage is invalid.")
    for field in ("checkpoint_every", "threads", "threads_per_qblade_run"):
        value = supplied.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"Schema-v2 run-plan {field} must be an integer.")
    if supplied["checkpoint_every"] < 0:
        raise RuntimeError("Schema-v2 checkpoint_every must be nonnegative.")
    if not 1 <= supplied["threads"] < 8:
        raise RuntimeError("Schema-v2 threads must be from 1 through 7.")
    if supplied["threads"] != supplied["threads_per_qblade_run"]:
        raise RuntimeError("Schema-v2 run-plan thread fields differ.")
    for name in ("results_root", "qblade_root"):
        value = supplied.get(name)
        if not isinstance(value, str) or str(Path(value).resolve()) != value:
            raise RuntimeError(
                f"Schema-v2 run-plan {name} must be one resolved absolute path."
            )
    commands = supplied.get("commands")
    if not isinstance(commands, list):
        raise RuntimeError("Schema-v2 run-plan commands must be a list.")
    by_id = {str(entry["case_id"]): entry for entry in manifest["cases"]}
    seen: set[str] = set()
    for index, row in enumerate(commands):
        if not isinstance(row, dict) or set(row) != _RUN_PLAN_COMMAND_KEYS:
            raise RuntimeError(
                f"Schema-v2 run-plan command {index} key schema differs."
            )
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in by_id:
            raise RuntimeError(
                f"Schema-v2 run plan contains unknown case command {case_id!r}."
            )
        if case_id in seen:
            raise RuntimeError(
                f"Schema-v2 run plan contains duplicate case command {case_id!r}."
            )
        seen.add(case_id)
        if row.get("role") != by_id[case_id]["role"]:
            raise RuntimeError(
                f"Schema-v2 run-plan role differs for case {case_id}."
            )
        argv = row.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) for value in argv)
        ):
            raise RuntimeError(
                f"Schema-v2 run-plan argv is invalid for case {case_id}."
            )
    return supplied


def _assert_v2_case_outputs_fresh(
    plan: Mapping[str, Any], commands: Sequence[Mapping[str, Any]]
) -> None:
    results_root = Path(str(plan["results_root"]))
    resolved_root = results_root.resolve()
    for row in commands:
        case_output = results_root / str(row["case_id"])
        if case_output.is_symlink():
            raise FileExistsError(
                f"Refusing symlinked schema-v2 case output: {case_output}"
            )
        resolved_output = case_output.resolve(strict=False)
        if not _path_is_within(resolved_output, resolved_root):
            raise FileExistsError(
                f"Schema-v2 case output escapes results_root: {case_output}"
            )
        if case_output.exists():
            if not case_output.is_dir() or any(case_output.iterdir()):
                raise FileExistsError(
                    "Refusing existing nonempty schema-v2 per-case output; "
                    f"fresh execution has no resume semantics: {case_output}"
                )


def _assert_v2_post_dispatch(
    campaign_root: Path,
    manifest_path: Path,
    baseline_tree: tuple[tuple[Any, ...], ...],
    row: Mapping[str, Any],
) -> None:
    """Fail if a SIL case mutated inputs or reported native callbacks."""

    if _campaign_tree_snapshot(campaign_root) != baseline_tree:
        raise RuntimeError(
            "Schema-v2 subprocess mutated the immutable campaign tree."
        )
    # Reparse and rehash every bound input after the subprocess, too.  This is
    # deliberately redundant with the snapshot so a restored-but-substituted
    # file cannot evade the immutable schema checks.
    _load_campaign(manifest_path)
    case_output = Path(str(row["argv"][row["argv"].index("--output-dir") + 1]))
    summaries = sorted(case_output.glob("*_run_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(
            "Schema-v2 subprocess must publish exactly one run summary; "
            f"observed {len(summaries)} for {row['case_id']}."
        )
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    evidence = summary.get("native_callback_evidence")
    if not isinstance(evidence, Mapping):
        raise RuntimeError(
            "Schema-v2 run summary lacks native-callback evidence."
        )
    activity = (
        bool(evidence.get("log_file_exists"))
        or int(evidence.get("update_init_count", 0) or 0) != 0
        or int(evidence.get("update_count", 0) or 0) != 0
        or int(evidence.get("update_count_observed_max", 0) or 0) != 0
    )
    if activity:
        raise RuntimeError(
            "Unexpected native callback activity in schema-v2 SIL execution."
        )


def _canonical_v2_plan_from_supplied(
    supplied: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    intent = supplied["execution_intent"]
    diagnostic_only = intent == "diagnostic_only_after_frozen_gate_failure"
    return build_run_plan(
        manifest_path,
        Path(str(supplied["results_root"])),
        stage=str(supplied["stage"]),
        qblade_root=Path(str(supplied["qblade_root"])),
        checkpoint_every=int(supplied["checkpoint_every"]),
        threads=int(supplied["threads"]),
        diagnostic_only=diagnostic_only,
    )


def execute_run_plan(
    plan: Mapping[str, Any],
    manifest_path: Path,
    *,
    allow_failed_analytical_gate: bool = False,
) -> None:
    campaign_root, manifest = _load_campaign(manifest_path)
    response_semantics = stationary_protocol.validate_response_semantics(manifest)
    if bool(response_semantics["v2"]):
        supplied = _validate_ambient_v2_run_plan_shape(plan, manifest)
        expected_plan = _canonical_v2_plan_from_supplied(
            supplied, Path(manifest_path).expanduser().resolve()
        )
        if supplied != expected_plan:
            raise RuntimeError(
                "Schema-v2 run plan differs from the canonical immutable plan."
            )
        if (
            supplied["stage"] in {"held_out", "all"}
            and (
                supplied["execution_intent"]
                != "diagnostic_only_after_frozen_gate_failure"
                or not allow_failed_analytical_gate
            )
        ):
            raise RuntimeError(
                "Schema-v2 held-out/all dispatch requires the provenance-bound "
                "diagnostic-only plan and explicit failed-gate override."
            )
        _assert_v2_case_outputs_fresh(supplied, supplied["commands"])
        baseline_tree = _campaign_tree_snapshot(campaign_root)
    if (
        manifest["calibration_policy"]["postfreeze_analytical_gate_status"] != "pass"
        and plan.get("stage") in {"held_out", "all"}
        and not allow_failed_analytical_gate
    ):
        raise RuntimeError(
            "Post-freeze analytical held-out gate failed. Refusing campaign execution unless "
            "--allow-failed-analytical-gate is supplied for an explicitly diagnostic run."
        )
    by_id = {entry["case_id"]: entry for entry in manifest["cases"]}
    commands = list(plan["commands"])
    if commands:
        first = by_id[commands[0]["case_id"]]
        _preflight_response_runtime(campaign_root / first["parameter_file"])
    for index, row in enumerate(commands):
        if bool(response_semantics["v2"]):
            # Sequential campaigns may run for days.  Rebuild from current
            # immutable inputs immediately before every dispatch so a changed
            # manifest, case bundle, filter, runner, Python runtime, QBlade
            # library/dependency, or command is rejected before it can run.
            current_plan = _canonical_v2_plan_from_supplied(
                supplied, Path(manifest_path).expanduser().resolve()
            )
            if current_plan != supplied or current_plan["commands"][index] != row:
                raise RuntimeError(
                    "Schema-v2 campaign inputs or dispatch command changed after planning."
                )
            _assert_v2_case_outputs_fresh(supplied, [row])
        subprocess.run(row["argv"], check=True)
        if bool(response_semantics["v2"]):
            _assert_v2_post_dispatch(
                campaign_root,
                Path(manifest_path).expanduser().resolve(),
                baseline_tree,
                row,
            )


def _column(columns: Mapping[str, Sequence[float]], requested: str) -> np.ndarray:
    lookup = {normalize_qblade_header(name): name for name in columns}
    normalized = normalize_qblade_header(requested)
    if normalized not in lookup:
        raise KeyError(f"QBlade result is missing required channel {requested!r}.")
    return np.asarray(columns[lookup[normalized]], dtype=float)


def _half_range(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size < 3 or np.any(~np.isfinite(array)):
        raise ValueError("A finite history with at least three samples is required.")
    return 0.5 * (float(np.max(array)) - float(np.min(array)))


def _dominant_frequency_metrics(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    natural_frequency_hz: float,
) -> dict[str, float | None]:
    """Return the raw-signal dominant frequency and power near ``f_n``.

    The signal is deliberately not band-pass filtered: a fixed-source response
    that follows ``St U*`` must not be made to look locked-in by preprocessing.
    A Hann window limits leakage, and a three-bin log-power interpolation gives
    a frequency estimate finer than the finite record's FFT-bin spacing.
    """

    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if time.size != signal.size or time.size < 8:
        raise ValueError("Frequency analysis requires aligned histories with at least eight samples.")
    dt_values = np.diff(time)
    dt = float(np.median(dt_values))
    if (
        not math.isfinite(dt)
        or dt <= 0.0
        or np.any(dt_values <= 0.0)
        or not np.allclose(dt_values, dt, rtol=1.0e-5, atol=1.0e-12)
    ):
        raise ValueError("Frequency analysis requires a positive uniform timestep.")
    centered = signal - float(np.mean(signal))
    windowed = centered * np.hanning(centered.size)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    frequency = np.fft.rfftfreq(centered.size, dt)
    if power.size <= 1:
        return {"dominant_frequency_hz": None, "mode_band_power_fraction": None}
    power[0] = 0.0
    total_power = float(np.sum(power))
    peak_index = int(np.argmax(power))
    if not math.isfinite(total_power) or total_power <= 0.0 or power[peak_index] <= 0.0:
        return {"dominant_frequency_hz": None, "mode_band_power_fraction": None}

    dominant_frequency = float(frequency[peak_index])
    if 0 < peak_index < power.size - 1:
        neighboring_log_power = np.log(np.maximum(power[peak_index - 1 : peak_index + 2], 1.0e-300))
        denominator = float(
            neighboring_log_power[0]
            - 2.0 * neighboring_log_power[1]
            + neighboring_log_power[2]
        )
        if denominator != 0.0:
            offset = 0.5 * float(
                neighboring_log_power[0] - neighboring_log_power[2]
            ) / denominator
            if abs(offset) <= 1.0:
                dominant_frequency += offset * float(frequency[1] - frequency[0])

    lower = (1.0 - LOCK_IN_FREQUENCY_TOLERANCE_RATIO) * natural_frequency_hz
    upper = (1.0 + LOCK_IN_FREQUENCY_TOLERANCE_RATIO) * natural_frequency_hz
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, upper)
    mode_band = (frequency >= lower - tolerance) & (frequency <= upper + tolerance)
    return {
        "dominant_frequency_hz": dominant_frequency,
        "mode_band_power_fraction": float(np.sum(power[mode_band]) / total_power),
    }


def _window_response_metrics(
    time: np.ndarray,
    displacement: np.ndarray,
    *,
    natural_frequency_hz: float,
) -> dict[str, float | None]:
    amplitude_m = _half_range(displacement)
    frequency = _dominant_frequency_metrics(
        time,
        displacement,
        natural_frequency_hz=natural_frequency_hz,
    )
    return {"amplitude_m": amplitude_m, **frequency}


def _validated_qblade_result_time_grid(
    exported_time: np.ndarray,
    *,
    expected_timestep_s: float,
    expected_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate rounded QBlade times and return the exact configured grid.

    QBlade exports result values with ``%.5E`` formatting.  At fine time
    steps, successive parsed differences therefore alternate and eventually
    repeat even when the internal grid is exact.  Each parsed timestamp is
    bound by sample index to the deterministic rendering of the configured
    grid.  The reconstructed exact grid is used only after sample count,
    nondecreasing order, and every rendered-value comparison pass.
    """

    time = np.asarray(exported_time, dtype=float)
    if (
        time.ndim != 1
        or isinstance(expected_samples, bool)
        or not isinstance(expected_samples, int)
        or expected_samples < 3
    ):
        raise ValueError("QBlade time-grid validation requires at least three expected samples.")
    if not math.isfinite(expected_timestep_s) or expected_timestep_s <= 0.0:
        raise ValueError("The configured QBlade timestep must be finite and positive.")
    if time.size != expected_samples:
        raise ValueError(
            "QBlade result sample count does not match the configured grid: "
            f"expected {expected_samples}, observed {time.size}."
        )
    if np.any(~np.isfinite(time)):
        raise ValueError("QBlade result time contains non-finite values.")
    if np.any(np.diff(time) < 0.0):
        raise ValueError(
            "QBlade exported time must remain nondecreasing before grid reconstruction."
        )

    exact_time = np.arange(expected_samples, dtype=float) * float(
        expected_timestep_s
    )
    # QBlade advances time by repeated binary64 additions.  The classical
    # nonnegative summation bound gamma_n * sum(abs(dt)) brackets that result;
    # one additional unit-roundoff term accounts for the rounded i*dt value
    # used here as the interval center.  This interval is many orders smaller
    # than an export bin, but can straddle a half-bin boundary.
    unit_roundoff = 0.5 * np.finfo(float).eps
    addition_index = np.arange(expected_samples, dtype=float)
    n_unit_roundoff = addition_index * unit_roundoff
    if np.any(n_unit_roundoff >= 1.0):
        raise ValueError(
            "Configured QBlade grid is too long for the binary64 accumulation bound."
        )
    gamma_n = n_unit_roundoff / (1.0 - n_unit_roundoff)
    accumulation_bound = np.nextafter(
        (gamma_n + unit_roundoff) * np.abs(exact_time),
        np.inf,
    )
    accumulation_bound[0] = 0.0
    rendered_lower_time = np.fromiter(
        (
            float(QBLADE_RESULT_TIME_FORMAT % value)
            for value in exact_time - accumulation_bound
        ),
        dtype=float,
        count=expected_samples,
    )
    rendered_upper_time = np.fromiter(
        (
            float(QBLADE_RESULT_TIME_FORMAT % value)
            for value in exact_time + accumulation_bound
        ),
        dtype=float,
        count=expected_samples,
    )
    # Zero has no scientific exponent.  The configured dt supplies the local
    # scale for its formatting interval; the expected first value remains zero.
    format_scale = np.maximum(np.abs(exact_time), float(expected_timestep_s))
    scientific_exponent = np.floor(np.log10(format_scale))
    printed_unit = np.power(
        10.0,
        scientific_exponent - QBLADE_RESULT_TIME_FORMAT_DECIMALS,
    )
    parse_slack = 16.0 * np.finfo(float).eps * np.maximum.reduce(
        (
            np.ones_like(time),
            np.abs(time),
            np.abs(rendered_lower_time),
            np.abs(rendered_upper_time),
        )
    )
    rendered_span = np.abs(rendered_upper_time - rendered_lower_time)
    lower_rendered_scale = np.maximum(
        np.abs(rendered_lower_time), float(expected_timestep_s)
    )
    upper_rendered_scale = np.maximum(
        np.abs(rendered_upper_time), float(expected_timestep_s)
    )
    minimum_candidate_printed_unit = np.minimum(
        np.power(
            10.0,
            np.floor(np.log10(lower_rendered_scale))
            - QBLADE_RESULT_TIME_FORMAT_DECIMALS,
        ),
        np.power(
            10.0,
            np.floor(np.log10(upper_rendered_scale))
            - QBLADE_RESULT_TIME_FORMAT_DECIMALS,
        ),
    )
    if np.any(rendered_span > minimum_candidate_printed_unit + 2.0 * parse_slack):
        raise ValueError(
            "Binary64 accumulation interval spans more than two QBlade export "
            "tokens; refusing ambiguous time-grid reconstruction."
        )
    rounding_tolerance = 0.5 * printed_unit + accumulation_bound + parse_slack
    absolute_error = np.abs(time - exact_time)
    lower_renderer_mismatch = np.abs(time - rendered_lower_time)
    upper_renderer_mismatch = np.abs(time - rendered_upper_time)
    renderer_mismatch = np.minimum(
        lower_renderer_mismatch,
        upper_renderer_mismatch,
    )
    invalid = renderer_mismatch > parse_slack
    if np.any(invalid):
        first_invalid = int(np.flatnonzero(invalid)[0])
        raise ValueError(
            "QBlade exported time is inconsistent with the configured grid and "
            f"{QBLADE_RESULT_TIME_FORMAT} rounding at sample {first_invalid}: "
            f"expected {exact_time[first_invalid]:.17g} s, observed "
            f"{time[first_invalid]:.17g} s, admissible rendered token(s) "
            f"{rendered_lower_time[first_invalid]:.17g} and "
            f"{rendered_upper_time[first_invalid]:.17g} s, renderer mismatch "
            f"{renderer_mismatch[first_invalid]:.6g} s, allowed parse slack "
            f"{parse_slack[first_invalid]:.6g} s."
        )

    error_ratio = absolute_error / rounding_tolerance
    adjacent_difference = np.diff(time)
    return exact_time, {
        "schema": QBLADE_RESULT_TIME_GRID_SCHEMA,
        "source": (
            "manifest timestep_s multiplied by zero-based sequential sample "
            "index, bounded for IEEE-754 binary64 repeated-addition error, "
            "then deterministically rendered with the declared export format"
        ),
        "export_format": QBLADE_RESULT_TIME_FORMAT,
        "rounding_bound": (
            "half of one unit in the final printed decimal at each expected "
            "scientific exponent, plus the binary64 repeated-addition bound "
            "and floating-point parse slack"
        ),
        "acceptance_rule": (
            "each parsed timestamp must match one of the at-most-two rendered "
            "tokens at its configured-grid repeated-addition bounds within "
            "floating-point parse slack"
        ),
        "floating_accumulation_model": (
            "nonnegative sequential IEEE-754 binary64 additions from zero; "
            "Higham gamma_n summation bound plus one unit-roundoff centering term"
        ),
        "binary64_unit_roundoff": float(unit_roundoff),
        "expected_samples": expected_samples,
        "configured_timestep_s": float(expected_timestep_s),
        "expected_first_time_s": float(exact_time[0]),
        "expected_last_time_s": float(exact_time[-1]),
        "exported_first_time_s": float(time[0]),
        "exported_last_time_s": float(time[-1]),
        "nondecreasing_exported_time": True,
        "duplicate_exported_time_count": int(
            np.count_nonzero(adjacent_difference == 0.0)
        ),
        "two_token_rounding_boundary_sample_count": int(
            np.count_nonzero(rendered_lower_time != rendered_upper_time)
        ),
        "maximum_repeated_addition_time_bound_s": float(
            np.max(accumulation_bound)
        ),
        "exported_adjacent_time_difference_min_s": float(
            np.min(adjacent_difference)
        ),
        "exported_adjacent_time_difference_median_s": float(
            np.median(adjacent_difference)
        ),
        "exported_adjacent_time_difference_max_s": float(
            np.max(adjacent_difference)
        ),
        "maximum_export_time_absolute_error_s": float(np.max(absolute_error)),
        "maximum_export_time_rounding_tolerance_s": float(
            np.max(rounding_tolerance)
        ),
        "maximum_export_time_error_to_tolerance_ratio": float(
            np.max(error_ratio)
        ),
        "maximum_export_time_renderer_mismatch_s": float(
            np.max(renderer_mismatch)
        ),
        "maximum_export_time_renderer_match_tolerance_s": float(
            np.max(parse_slack)
        ),
        "pass": True,
    }


def response_amplitude_from_result(
    result_path: Path,
    *,
    natural_frequency_hz: float,
    response_window_cycles: float,
    expected_timestep_s: float,
    expected_samples: int,
) -> dict[str, Any]:
    columns = read_qblade_result_numeric_columns(Path(result_path))
    if not columns:
        raise ValueError(f"No parseable QBlade result columns in {result_path}.")
    time = _column(columns, "Time [s]")
    displacement = _column(columns, RESULT_CHANNEL)
    if time.size != displacement.size or time.size < 3 or np.any(~np.isfinite(displacement)):
        raise ValueError(f"Invalid time/displacement history in {result_path}.")
    analysis_time, time_grid_audit = _validated_qblade_result_time_grid(
        time,
        expected_timestep_s=expected_timestep_s,
        expected_samples=expected_samples,
    )
    window_s = response_window_cycles / natural_frequency_hz
    retain = analysis_time >= analysis_time[-1] - window_s - 1.0e-12
    retained_time = analysis_time[retain]
    retained_displacement = displacement[retain]
    if retained_time.size < 3 or retained_time[-1] - retained_time[0] < 0.9 * window_s:
        raise ValueError(
            f"Result {result_path} does not contain the requested {response_window_cycles:g}-cycle response window."
        )
    amplitude_m = _half_range(retained_displacement)
    retained_frequency = _dominant_frequency_metrics(
        retained_time,
        retained_displacement,
        natural_frequency_hz=natural_frequency_hz,
    )

    stationarity_window_s = STATIONARITY_WINDOW_CYCLES / natural_frequency_hz
    split_time = analysis_time[-1] - stationarity_window_s
    early_start = split_time - stationarity_window_s
    early_mask = (analysis_time >= early_start - 1.0e-12) & (
        analysis_time < split_time - 1.0e-12
    )
    final_mask = analysis_time >= split_time - 1.0e-12
    stationarity_evaluated = bool(
        np.count_nonzero(early_mask) >= 8 and np.count_nonzero(final_mask) >= 8
    )
    if stationarity_evaluated:
        early = _window_response_metrics(
            analysis_time[early_mask],
            displacement[early_mask],
            natural_frequency_hz=natural_frequency_hz,
        )
        final = _window_response_metrics(
            analysis_time[final_mask],
            displacement[final_mask],
            natural_frequency_hz=natural_frequency_hz,
        )
        early_ratio = float(early["amplitude_m"]) / DIAMETER_M
        final_ratio = float(final["amplitude_m"]) / DIAMETER_M
        amplitude_stationarity_tolerance = max(
            STATIONARITY_ABSOLUTE_AMPLITUDE_RATIO,
            STATIONARITY_RELATIVE_AMPLITUDE * abs(final_ratio),
        )
        amplitude_stationarity_difference = abs(final_ratio - early_ratio)
        amplitude_stationarity_pass = (
            amplitude_stationarity_difference <= amplitude_stationarity_tolerance
        )
        frequency_stationarity_evaluated = final_ratio >= ACTIVE_AMPLITUDE_RATIO
        early_frequency = early["dominant_frequency_hz"]
        final_frequency = final["dominant_frequency_hz"]
        if frequency_stationarity_evaluated:
            frequency_stationarity_difference = (
                None
                if early_frequency is None or final_frequency is None
                else abs(float(final_frequency) - float(early_frequency))
                / natural_frequency_hz
            )
            frequency_stationarity_pass = (
                frequency_stationarity_difference is not None
                and frequency_stationarity_difference
                <= STATIONARITY_FREQUENCY_RATIO
            )
        else:
            frequency_stationarity_difference = None
            frequency_stationarity_pass = None
        stationarity_pass = amplitude_stationarity_pass and (
            not frequency_stationarity_evaluated
            or bool(frequency_stationarity_pass)
        )
    else:
        early_ratio = None
        final_ratio = None
        amplitude_stationarity_tolerance = None
        amplitude_stationarity_difference = None
        amplitude_stationarity_pass = False
        frequency_stationarity_evaluated = False
        early_frequency = None
        final_frequency = None
        frequency_stationarity_difference = None
        frequency_stationarity_pass = None
        stationarity_pass = False
    return {
        "result_path": Path(result_path).resolve().as_posix(),
        "samples": int(time.size),
        "retained_samples": int(retained_time.size),
        "retained_cycles": (retained_time[-1] - retained_time[0]) * natural_frequency_hz,
        "exported_adjacent_time_difference_min_s": time_grid_audit[
            "exported_adjacent_time_difference_min_s"
        ],
        "exported_adjacent_time_difference_median_s": time_grid_audit[
            "exported_adjacent_time_difference_median_s"
        ],
        "exported_adjacent_time_difference_max_s": time_grid_audit[
            "exported_adjacent_time_difference_max_s"
        ],
        "configured_analysis_timestep_s": float(expected_timestep_s),
        "export_time_grid_schema": time_grid_audit["schema"],
        "export_time_grid_source": time_grid_audit["source"],
        "export_time_grid_acceptance_rule": time_grid_audit["acceptance_rule"],
        "exported_duplicate_time_count": time_grid_audit[
            "duplicate_exported_time_count"
        ],
        "two_token_rounding_boundary_sample_count": time_grid_audit[
            "two_token_rounding_boundary_sample_count"
        ],
        "maximum_repeated_addition_time_bound_s": time_grid_audit[
            "maximum_repeated_addition_time_bound_s"
        ],
        "maximum_export_time_absolute_error_s": time_grid_audit[
            "maximum_export_time_absolute_error_s"
        ],
        "maximum_export_time_rounding_tolerance_s": time_grid_audit[
            "maximum_export_time_rounding_tolerance_s"
        ],
        "maximum_export_time_renderer_mismatch_s": time_grid_audit[
            "maximum_export_time_renderer_mismatch_s"
        ],
        "maximum_export_time_renderer_match_tolerance_s": time_grid_audit[
            "maximum_export_time_renderer_match_tolerance_s"
        ],
        "maximum_export_time_error_to_tolerance_ratio": time_grid_audit[
            "maximum_export_time_error_to_tolerance_ratio"
        ],
        "export_time_grid_audit": time_grid_audit,
        "tip_peak_amplitude_m": amplitude_m,
        "tip_peak_amplitude_over_diameter": amplitude_m / DIAMETER_M,
        "terminal_dominant_frequency_hz": retained_frequency["dominant_frequency_hz"],
        "terminal_dominant_frequency_over_fn": (
            None
            if retained_frequency["dominant_frequency_hz"] is None
            else float(retained_frequency["dominant_frequency_hz"])
            / natural_frequency_hz
        ),
        "terminal_mode_band_power_fraction": retained_frequency[
            "mode_band_power_fraction"
        ],
        "stationarity_evaluated": stationarity_evaluated,
        "stationarity_early_amplitude_over_diameter": early_ratio,
        "stationarity_final_amplitude_over_diameter": final_ratio,
        "stationarity_amplitude_absolute_difference": amplitude_stationarity_difference,
        "stationarity_amplitude_tolerance": amplitude_stationarity_tolerance,
        "stationarity_amplitude_pass": amplitude_stationarity_pass,
        "stationarity_frequency_evaluated": frequency_stationarity_evaluated,
        "stationarity_early_frequency_hz": early_frequency,
        "stationarity_final_frequency_hz": final_frequency,
        "stationarity_frequency_difference_over_fn": frequency_stationarity_difference,
        "stationarity_frequency_tolerance_over_fn": STATIONARITY_FREQUENCY_RATIO,
        "stationarity_frequency_pass": frequency_stationarity_pass,
        "stationarity_pass": stationarity_pass,
    }


def startup_seed_response_audit(
    result_path: Path,
    *,
    natural_frequency_hz: float,
    seed_duration_cycles: float,
    expected_timestep_s: float,
    expected_samples: int,
) -> dict[str, Any]:
    """Check the realized free response during the first cycle after the seed."""

    if not math.isfinite(natural_frequency_hz) or natural_frequency_hz <= 0.0:
        raise ValueError("natural_frequency_hz must be finite and positive.")
    if not math.isfinite(seed_duration_cycles) or seed_duration_cycles <= 0.0:
        raise ValueError("seed_duration_cycles must be finite and positive.")
    columns = read_qblade_result_numeric_columns(Path(result_path))
    if not columns:
        raise ValueError(f"No parseable QBlade result columns in {result_path}.")
    exported_time = _column(columns, "Time [s]")
    displacement = _column(columns, RESULT_CHANNEL)
    if (
        exported_time.size != displacement.size
        or exported_time.size < 8
        or np.any(~np.isfinite(displacement))
    ):
        raise ValueError(f"Invalid seed-response history in {result_path}.")
    analysis_time, time_grid_audit = _validated_qblade_result_time_grid(
        exported_time,
        expected_timestep_s=expected_timestep_s,
        expected_samples=expected_samples,
    )
    seed_end_s = seed_duration_cycles / natural_frequency_hz
    observation_end_s = (
        seed_duration_cycles + STARTUP_SEED_OBSERVATION_CYCLES
    ) / natural_frequency_hz
    tolerance_s = 0.51 * float(expected_timestep_s)
    retain = (analysis_time >= seed_end_s - tolerance_s) & (
        analysis_time <= observation_end_s + tolerance_s
    )
    retained_time = analysis_time[retain]
    retained_displacement = displacement[retain]
    required_span_s = STARTUP_SEED_OBSERVATION_CYCLES / natural_frequency_hz
    if (
        retained_time.size < 8
        or retained_time[-1] - retained_time[0] < 0.95 * required_span_s
    ):
        raise ValueError(
            f"Result {result_path} does not contain one complete post-seed cycle."
        )
    amplitude_m = _half_range(retained_displacement)
    amplitude_ratio = amplitude_m / DIAMETER_M
    passed = (
        STARTUP_SEED_REALIZED_MINIMUM_AMPLITUDE_RATIO
        <= amplitude_ratio
        <= STARTUP_SEED_REALIZED_MAXIMUM_AMPLITUDE_RATIO
    )
    return {
        "evaluated": True,
        "seed_end_s": seed_end_s,
        "observation_cycles": STARTUP_SEED_OBSERVATION_CYCLES,
        "observation_samples": int(retained_time.size),
        "realized_amplitude_m": amplitude_m,
        "realized_amplitude_ratio": amplitude_ratio,
        "minimum_amplitude_ratio": STARTUP_SEED_REALIZED_MINIMUM_AMPLITUDE_RATIO,
        "maximum_amplitude_ratio": STARTUP_SEED_REALIZED_MAXIMUM_AMPLITUDE_RATIO,
        "export_time_grid_audit": time_grid_audit,
        "pass": passed,
    }


def _load_figure10_peaks(path: Path) -> dict[tuple[float, str], tuple[float, float]]:
    values: dict[tuple[float, str], tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[(float(row["scruton_number"]), row["series"])] = (
                float(row["peak_tip_amplitude_over_diameter"]),
                float(row["digitization_uncertainty"]),
            )
    return values


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _unique_matching_file(directory: Path, pattern: str) -> tuple[Path | None, str | None]:
    matches = sorted(directory.glob(pattern))
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, f"missing {pattern}"
    return None, f"multiple files match {pattern}: {[path.name for path in matches]}"


def _history_float(value_text: str, name: str) -> float:
    if value_text == "":
        raise ValueError(f"SIL response history is missing {name!r}.")
    value = float(value_text)
    if not math.isfinite(value):
        raise ValueError(f"SIL response history contains non-finite {name!r}.")
    return value


def _history_bool(value_text: str, name: str) -> bool:
    value = str(value_text).strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"SIL response history contains invalid Boolean {name!r}={value!r}.")


def _history_count(value_text: str, name: str) -> int:
    value = _history_float(value_text, name)
    count = int(value)
    if value != count or count < 0:
        raise ValueError(
            f"SIL response history contains invalid nonnegative count {name!r}={value!r}."
        )
    return count


def _history_optional_count(value_text: str, name: str) -> int | None:
    if str(value_text).strip() == "":
        return None
    return _history_count(value_text, name)


def _history_optional_float(value_text: str, name: str) -> float | None:
    if str(value_text).strip() == "":
        return None
    return _history_float(value_text, name)


def _history_sha256(value_text: str, name: str) -> str:
    value = str(value_text).strip()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"SIL response history contains invalid SHA-256 {name!r}.")
    return value


def _zero_vector_sha256(component_count: int) -> str:
    if isinstance(component_count, bool) or component_count <= 0:
        raise ValueError("force-vector component count must be positive.")
    payload = b"vorlap-force-vector-sha256-v1\0"
    payload += struct.pack("!Q", component_count)
    payload += struct.pack(f"!{component_count}d", *([0.0] * component_count))
    return hashlib.sha256(payload).hexdigest()


def response_history_audit(
    path: Path,
    *,
    natural_frequency_hz: float,
    max_abs_gain: float,
    max_abs_nodal_force_n: float,
    history_mode: str = "full",
    expected_force_vector_component_count: int | None = None,
    startup_seed_config: Mapping[str, Any] | None = None,
    response_model_config: Mapping[str, Any] | None = None,
    expected_reduced_velocity: float | None = None,
    expected_ambient_response_speed_mps: float | None = None,
) -> dict[str, Any]:
    """Audit response state with two streaming passes and O(1) row memory.

    SIL histories contain hundreds of per-node columns and can be hundreds of
    megabytes.  The first pass validates time/order, compact lag provenance,
    startup-seed shutdown, and global extrema/counts from named columns only.
    Once the terminal time is
    known, the second pass accumulates the two bounded stationarity-window
    means without retaining the full table or any per-node columns.
    """

    history_path = Path(path)
    if not math.isfinite(natural_frequency_hz) or natural_frequency_hz <= 0.0:
        raise ValueError("natural_frequency_hz must be finite and positive.")
    if not math.isfinite(max_abs_gain) or max_abs_gain <= 0.0:
        raise ValueError("max_abs_gain must be finite and positive.")
    if not math.isfinite(max_abs_nodal_force_n) or max_abs_nodal_force_n <= 0.0:
        raise ValueError("max_abs_nodal_force_n must be finite and positive.")
    if history_mode not in {"full", "compact"}:
        raise ValueError("history_mode must be 'full' or 'compact'.")

    required = (
        "time_s",
        "response_model_type",
        "response_enabled",
        "response_amplitude_ratio",
        "response_raw_gain_min",
        "response_raw_gain_max",
        "response_gain_limited_nodes",
        "response_force_limited_nodes",
        "response_max_abs_nodal_force_n",
    )
    mode_required = ("force_history_schema", "step_index")
    if history_mode == "compact":
        mode_required += (
            "force_vector_digest_schema",
            "force_vector_component_count",
            "generated_step_index",
            "generated_time_s",
            "applied_source_step_index",
            "applied_source_time_s",
            "applied_startup_zero",
            "applied_force_lag_steps",
            "generated_force_vector_sha256",
            "applied_force_vector_sha256",
            "structural_velocity_vector_sha256",
            "applied_force_dot_structural_velocity_w",
            "applied_power_sign_convention",
        )
    seed_required: tuple[str, ...] = ()
    if startup_seed_config is not None:
        seed_required = (
            "response_startup_seed_enabled",
            "response_startup_seed_active",
            "response_startup_seed_elapsed_cycles",
            "response_startup_seed_duration_cycles",
            "response_startup_seed_duration_s",
            "response_startup_seed_taper",
            "response_startup_seed_target_amplitude_ratio",
            "response_startup_seed_target_amplitude_m",
            "response_startup_seed_ideal_terminal_amplitude_m",
            "response_startup_seed_modal_acceleration_scale_mps2",
            "response_startup_seed_modal_acceleration_mps2",
            "response_startup_seed_raw_generalized_force_n",
            "response_startup_seed_raw_max_abs_nodal_force_n",
            "response_startup_seed_force_limited_nodes",
            "response_startup_seed_modal_power_w",
        )
    response_schema_version = (
        None
        if response_model_config is None
        else response_model_config.get("schema_version")
    )
    ambient_v2 = response_schema_version == 2
    v2_required: tuple[str, ...] = ()
    if ambient_v2:
        if response_model_config.get("aerodynamic_speed_basis") != (
            AMBIENT_V2_AERODYNAMIC_SPEED_BASIS
        ):
            raise ValueError("Schema-v2 history audit requires the frozen speed basis.")
        if expected_reduced_velocity is None or expected_ambient_response_speed_mps is None:
            raise ValueError(
                "Schema-v2 history audit requires expected U* and ambient response speed."
            )
        v2_required = (
            "response_response_model_schema_version",
            "response_aerodynamic_speed_basis",
            "response_response_declared_ambient_crossflow_speed_min_mps",
            "response_response_declared_ambient_crossflow_speed_max_mps",
            "response_response_relative_crossflow_speed_min_mps",
            "response_response_relative_crossflow_speed_max_mps",
            "response_response_section_width_basis",
            "response_response_section_width_min_m",
            "response_response_section_width_max_m",
            "response_configured_response_reduced_velocity_min",
            "response_configured_response_reduced_velocity_max",
            "response_realized_response_reduced_velocity_min",
            "response_realized_response_reduced_velocity_max",
            "response_speed_basis_invariant_violations",
            "response_activation_invariant_violations",
        )
    required += mode_required + seed_required + v2_required

    def open_required_rows():
        handle = history_path.open(newline="", encoding="utf-8")
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            handle.close()
            raise ValueError(f"SIL response history {path} is empty.")
        if len(header) != len(set(header)):
            handle.close()
            raise ValueError(f"SIL response history {path} has duplicate column names.")
        missing = set(required).difference(header)
        if missing:
            handle.close()
            raise ValueError(
                f"SIL response history {path} lacks required column(s): {sorted(missing)}"
            )
        indices = {name: header.index(name) for name in required}
        maximum_index = max(indices.values())

        def selected_rows():
            try:
                for row_number, row in enumerate(reader, start=2):
                    if len(row) <= maximum_index:
                        raise ValueError(
                            f"SIL response history {path} row {row_number} is truncated."
                        )
                    yield {name: row[index] for name, index in indices.items()}
            finally:
                handle.close()

        return header, selected_rows()

    header, rows = open_required_rows()
    samples = 0
    previous_time: float | None = None
    last_time: float | None = None
    model_types: set[str] = set()
    enabled_values: set[bool] = set()
    maximum_raw_gain = 0.0
    maximum_nodal_force = 0.0
    gain_limited_node_samples = 0
    force_limited_node_samples = 0
    expected_history_schema = (
        "vorlap-sil-force-history-compact-v1"
        if history_mode == "compact"
        else "vorlap-sil-force-history-v2"
    )
    previous_generated_digest: str | None = None
    previous_generated_time: float | None = None
    compact_component_count: int | None = None
    compact_lag_provenance_pass = history_mode != "compact"
    seed_active_samples = 0
    seed_inactive_samples = 0
    seed_acceleration_scale_mps2: float | None = None
    seed_integrity_pass = startup_seed_config is None
    v2_provenance_pass = ambient_v2
    ambient_speed_min = math.inf
    ambient_speed_max = -math.inf
    relative_speed_min = math.inf
    relative_speed_max = -math.inf
    response_width_min = math.inf
    response_width_max = -math.inf
    if startup_seed_config is not None:
        seed_duration_cycles = float(startup_seed_config["duration_cycles"])
        seed_duration_s = seed_duration_cycles / natural_frequency_hz
        seed_target_ratio = float(startup_seed_config["target_amplitude_ratio"])
        seed_integrity_pass = True
    else:
        seed_duration_cycles = None
        seed_duration_s = None
        seed_target_ratio = None
    for row in rows:
        time_value = _history_float(row["time_s"], "time_s")
        if previous_time is not None and time_value <= previous_time:
            raise ValueError(
                f"SIL response history time is not strictly increasing in {path}."
            )
        previous_time = time_value
        last_time = time_value
        step_index = _history_count(row["step_index"], "step_index")
        if step_index != samples:
            raise ValueError(
                f"SIL response history step_index is not sequential at row {samples}."
            )
        if str(row["force_history_schema"]).strip() != expected_history_schema:
            raise ValueError(
                "SIL response history row schema does not match the declared mode."
            )
        if history_mode == "compact":
            if str(row["force_vector_digest_schema"]).strip() != (
                "vorlap-force-vector-sha256-v1"
            ):
                raise ValueError("Compact SIL history vector-digest schema is invalid.")
            component_count = _history_count(
                row["force_vector_component_count"],
                "force_vector_component_count",
            )
            if component_count <= 0:
                raise ValueError("Compact SIL history vector size must be positive.")
            if compact_component_count is None:
                compact_component_count = component_count
            elif component_count != compact_component_count:
                raise ValueError("Compact SIL history vector size changed between rows.")
            if (
                expected_force_vector_component_count is not None
                and component_count != expected_force_vector_component_count
            ):
                raise ValueError(
                    "Compact SIL history vector size does not match configured nodes."
                )
            generated_step = _history_count(
                row["generated_step_index"], "generated_step_index"
            )
            generated_time = _history_float(
                row["generated_time_s"], "generated_time_s"
            )
            if generated_step != step_index or not math.isclose(
                generated_time, time_value, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    "Compact SIL history generated step/time is inconsistent."
                )
            lag_steps = _history_count(
                row["applied_force_lag_steps"], "applied_force_lag_steps"
            )
            if lag_steps != 1:
                raise ValueError(
                    "Vlastos qualification requires the declared one-step SIL force lag."
                )
            generated_digest = _history_sha256(
                row["generated_force_vector_sha256"],
                "generated_force_vector_sha256",
            )
            applied_digest = _history_sha256(
                row["applied_force_vector_sha256"],
                "applied_force_vector_sha256",
            )
            _history_sha256(
                row["structural_velocity_vector_sha256"],
                "structural_velocity_vector_sha256",
            )
            _history_float(
                row["applied_force_dot_structural_velocity_w"],
                "applied_force_dot_structural_velocity_w",
            )
            if str(row["applied_power_sign_convention"]).strip() != (
                "positive_when_applied_force_does_work_on_current_structural_velocity"
            ):
                raise ValueError(
                    "Compact SIL history applied-power sign convention is invalid."
                )
            startup_zero = _history_bool(
                row["applied_startup_zero"], "applied_startup_zero"
            )
            source_step = _history_optional_count(
                row["applied_source_step_index"], "applied_source_step_index"
            )
            source_time = _history_optional_float(
                row["applied_source_time_s"], "applied_source_time_s"
            )
            if step_index == 0:
                if (
                    not startup_zero
                    or source_step is not None
                    or source_time is not None
                    or applied_digest != _zero_vector_sha256(component_count)
                ):
                    raise ValueError(
                        "Compact SIL history startup-zero provenance is invalid."
                    )
            elif (
                startup_zero
                or source_step != step_index - 1
                or source_time is None
                or previous_generated_time is None
                or not math.isclose(
                    source_time,
                    previous_generated_time,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or applied_digest != previous_generated_digest
            ):
                raise ValueError(
                    "Compact SIL history applied digest does not match the prior "
                    "generated packet."
                )
            previous_generated_digest = generated_digest
            previous_generated_time = generated_time
            compact_lag_provenance_pass = True

        if ambient_v2:
            assert expected_reduced_velocity is not None
            assert expected_ambient_response_speed_mps is not None
            schema_value = _history_count(
                row["response_response_model_schema_version"],
                "response_response_model_schema_version",
            )
            speed_basis = str(row["response_aerodynamic_speed_basis"]).strip()
            width_basis = str(
                row["response_response_section_width_basis"]
            ).strip()
            expected_width_basis = (
                "explicit_response_section_width_m"
                if response_model_config.get("response_section_width_m") is not None
                else "declared_ambient_crossflow_angle"
            )
            ambient_min_row = _history_float(
                row["response_response_declared_ambient_crossflow_speed_min_mps"],
                "response_response_declared_ambient_crossflow_speed_min_mps",
            )
            ambient_max_row = _history_float(
                row["response_response_declared_ambient_crossflow_speed_max_mps"],
                "response_response_declared_ambient_crossflow_speed_max_mps",
            )
            relative_min_row = _history_float(
                row["response_response_relative_crossflow_speed_min_mps"],
                "response_response_relative_crossflow_speed_min_mps",
            )
            relative_max_row = _history_float(
                row["response_response_relative_crossflow_speed_max_mps"],
                "response_response_relative_crossflow_speed_max_mps",
            )
            width_min_row = _history_float(
                row["response_response_section_width_min_m"],
                "response_response_section_width_min_m",
            )
            width_max_row = _history_float(
                row["response_response_section_width_max_m"],
                "response_response_section_width_max_m",
            )
            configured_min = _history_float(
                row["response_configured_response_reduced_velocity_min"],
                "response_configured_response_reduced_velocity_min",
            )
            configured_max = _history_float(
                row["response_configured_response_reduced_velocity_max"],
                "response_configured_response_reduced_velocity_max",
            )
            realized_min = _history_float(
                row["response_realized_response_reduced_velocity_min"],
                "response_realized_response_reduced_velocity_min",
            )
            realized_max = _history_float(
                row["response_realized_response_reduced_velocity_max"],
                "response_realized_response_reduced_velocity_max",
            )
            speed_violations = _history_count(
                row["response_speed_basis_invariant_violations"],
                "response_speed_basis_invariant_violations",
            )
            activation_violations = _history_count(
                row["response_activation_invariant_violations"],
                "response_activation_invariant_violations",
            )
            explicit_width = response_model_config.get("response_section_width_m")
            if explicit_width is None:
                expected_width_min = DIAMETER_M
                expected_width_max = DIAMETER_M
            else:
                width_values = np.asarray(explicit_width, dtype=float)
                if width_values.ndim == 0:
                    width_values = np.asarray([float(width_values)])
                expected_width_min = float(np.min(width_values))
                expected_width_max = float(np.max(width_values))
            v2_provenance_pass = v2_provenance_pass and all(
                (
                    schema_value == 2,
                    speed_basis == AMBIENT_V2_AERODYNAMIC_SPEED_BASIS,
                    width_basis == expected_width_basis,
                    math.isclose(
                        ambient_min_row,
                        expected_ambient_response_speed_mps,
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    ),
                    math.isclose(
                        ambient_max_row,
                        expected_ambient_response_speed_mps,
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    ),
                    0.0 <= relative_min_row <= relative_max_row,
                    math.isclose(
                        width_min_row,
                        expected_width_min,
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    ),
                    math.isclose(
                        width_max_row,
                        expected_width_max,
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    ),
                    math.isclose(
                        configured_min,
                        expected_reduced_velocity,
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    ),
                    math.isclose(
                        configured_max,
                        expected_reduced_velocity,
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    ),
                    math.isclose(
                        realized_min,
                        expected_reduced_velocity,
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    ),
                    math.isclose(
                        realized_max,
                        expected_reduced_velocity,
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-12,
                    ),
                    speed_violations == 0,
                    activation_violations == 0,
                )
            )
            ambient_speed_min = min(ambient_speed_min, ambient_min_row)
            ambient_speed_max = max(ambient_speed_max, ambient_max_row)
            relative_speed_min = min(relative_speed_min, relative_min_row)
            relative_speed_max = max(relative_speed_max, relative_max_row)
            response_width_min = min(response_width_min, width_min_row)
            response_width_max = max(response_width_max, width_max_row)

        if startup_seed_config is not None:
            assert seed_duration_s is not None
            assert seed_duration_cycles is not None
            assert seed_target_ratio is not None
            seed_enabled = _history_bool(
                row["response_startup_seed_enabled"],
                "response_startup_seed_enabled",
            )
            seed_active = _history_bool(
                row["response_startup_seed_active"],
                "response_startup_seed_active",
            )
            expected_active = time_value < seed_duration_s
            seed_active_samples += int(seed_active)
            seed_inactive_samples += int(not seed_active)
            seed_integrity_pass = seed_integrity_pass and seed_enabled
            seed_integrity_pass = seed_integrity_pass and seed_active == expected_active
            elapsed_cycles = _history_float(
                row["response_startup_seed_elapsed_cycles"],
                "response_startup_seed_elapsed_cycles",
            )
            seed_integrity_pass = seed_integrity_pass and math.isclose(
                elapsed_cycles,
                min(max(time_value, 0.0) * natural_frequency_hz, seed_duration_cycles),
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            )
            seed_integrity_pass = seed_integrity_pass and math.isclose(
                _history_float(
                    row["response_startup_seed_duration_cycles"],
                    "response_startup_seed_duration_cycles",
                ),
                seed_duration_cycles,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            target_amplitude_m = seed_target_ratio * DIAMETER_M
            for field in (
                "response_startup_seed_target_amplitude_m",
                "response_startup_seed_ideal_terminal_amplitude_m",
            ):
                seed_integrity_pass = seed_integrity_pass and math.isclose(
                    _history_float(row[field], field),
                    target_amplitude_m,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-12,
                )
            reported_scale = _history_float(
                row["response_startup_seed_modal_acceleration_scale_mps2"],
                "response_startup_seed_modal_acceleration_scale_mps2",
            )
            if seed_acceleration_scale_mps2 is None:
                seed_acceleration_scale_mps2 = reported_scale
            seed_integrity_pass = seed_integrity_pass and reported_scale > 0.0
            seed_integrity_pass = seed_integrity_pass and math.isclose(
                reported_scale,
                seed_acceleration_scale_mps2,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            expected_taper = (
                math.sin(math.pi * time_value / seed_duration_s) ** 2
                if expected_active and time_value >= 0.0
                else 0.0
            )
            taper = _history_float(
                row["response_startup_seed_taper"],
                "response_startup_seed_taper",
            )
            seed_integrity_pass = seed_integrity_pass and math.isclose(
                taper,
                expected_taper,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            )
            modal_acceleration = _history_float(
                row["response_startup_seed_modal_acceleration_mps2"],
                "response_startup_seed_modal_acceleration_mps2",
            )
            expected_modal_acceleration = (
                reported_scale
                * expected_taper
                * math.sin(2.0 * math.pi * natural_frequency_hz * time_value)
                if expected_active and time_value >= 0.0
                else 0.0
            )
            seed_integrity_pass = seed_integrity_pass and math.isclose(
                modal_acceleration,
                expected_modal_acceleration,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
            seed_integrity_pass = seed_integrity_pass and math.isclose(
                _history_float(
                    row["response_startup_seed_duration_s"],
                    "response_startup_seed_duration_s",
                ),
                seed_duration_s,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            seed_integrity_pass = seed_integrity_pass and math.isclose(
                _history_float(
                    row["response_startup_seed_target_amplitude_ratio"],
                    "response_startup_seed_target_amplitude_ratio",
                ),
                seed_target_ratio,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            seed_limited = _history_count(
                row["response_startup_seed_force_limited_nodes"],
                "response_startup_seed_force_limited_nodes",
            )
            seed_integrity_pass = seed_integrity_pass and seed_limited == 0
            if not expected_active:
                for field in (
                    "response_startup_seed_taper",
                    "response_startup_seed_modal_acceleration_mps2",
                    "response_startup_seed_raw_generalized_force_n",
                    "response_startup_seed_raw_max_abs_nodal_force_n",
                    "response_startup_seed_modal_power_w",
                ):
                    seed_integrity_pass = seed_integrity_pass and (
                        _history_float(row[field], field) == 0.0
                    )
        model_types.add(str(row["response_model_type"]).strip())
        enabled_values.add(_history_bool(row["response_enabled"], "response_enabled"))
        _history_float(row["response_amplitude_ratio"], "response_amplitude_ratio")
        maximum_raw_gain = max(
            maximum_raw_gain,
            abs(_history_float(row["response_raw_gain_min"], "response_raw_gain_min")),
            abs(_history_float(row["response_raw_gain_max"], "response_raw_gain_max")),
        )
        maximum_nodal_force = max(
            maximum_nodal_force,
            abs(
                _history_float(
                    row["response_max_abs_nodal_force_n"],
                    "response_max_abs_nodal_force_n",
                )
            ),
        )
        gain_limited_node_samples += _history_count(
            row["response_gain_limited_nodes"], "response_gain_limited_nodes"
        )
        force_limited_node_samples += _history_count(
            row["response_force_limited_nodes"], "response_force_limited_nodes"
        )
        samples += 1
    if samples < 3 or last_time is None:
        raise ValueError(f"SIL response history {path} has fewer than three samples.")

    window_s = STATIONARITY_WINDOW_CYCLES / natural_frequency_hz
    split_time = last_time - window_s
    early_start = split_time - window_s
    early_count = 0
    early_sum = 0.0
    final_count = 0
    final_sum = 0.0
    second_pass_samples = 0
    second_header, terminal_rows = open_required_rows()
    if second_header != header:
        raise ValueError(f"SIL response history header changed while auditing {path}.")
    for row in terminal_rows:
        time_value = _history_float(row["time_s"], "time_s")
        amplitude_value = _history_float(
            row["response_amplitude_ratio"], "response_amplitude_ratio"
        )
        if early_start - 1.0e-12 <= time_value < split_time - 1.0e-12:
            early_count += 1
            early_sum += amplitude_value
        if time_value >= split_time - 1.0e-12:
            final_count += 1
            final_sum += amplitude_value
        second_pass_samples += 1
    if second_pass_samples != samples:
        raise ValueError(f"SIL response history row count changed while auditing {path}.")

    if early_count < 3 or final_count < 3:
        estimator_stationarity_evaluated = False
        early_mean = None
        final_mean = None
        estimator_difference = None
        estimator_tolerance = None
        estimator_stationarity_pass = False
    else:
        estimator_stationarity_evaluated = True
        early_mean = early_sum / early_count
        final_mean = final_sum / final_count
        estimator_difference = abs(final_mean - early_mean)
        estimator_tolerance = max(
            STATIONARITY_ABSOLUTE_AMPLITUDE_RATIO,
            STATIONARITY_RELATIVE_AMPLITUDE * abs(final_mean),
        )
        estimator_stationarity_pass = estimator_difference <= estimator_tolerance

    gain_headroom_limit = LIMIT_HEADROOM_FRACTION * float(max_abs_gain)
    force_headroom_limit = LIMIT_HEADROOM_FRACTION * float(max_abs_nodal_force_n)
    limit_pass = (
        gain_limited_node_samples == 0
        and force_limited_node_samples == 0
        and maximum_raw_gain <= gain_headroom_limit
        and maximum_nodal_force <= force_headroom_limit
    )
    result = {
        "path": history_path.resolve().as_posix(),
        "file_identity": {
            "path": history_path.resolve().as_posix(),
            "sha256": sha256_file(history_path),
            "size_bytes": history_path.stat().st_size,
        },
        "samples": samples,
        "columns": len(header),
        "streaming_passes": 2,
        "rows_retained_in_memory": 1,
        "terminal_window_samples_processed": early_count + final_count,
        "response_model_types": sorted(model_types),
        "response_enabled_values": sorted(enabled_values),
        "runtime_model_pass": model_types == {"modal_negative_damping"}
        and enabled_values == {True},
        "response_estimator_stationarity_evaluated": estimator_stationarity_evaluated,
        "response_estimator_early_mean_amplitude_ratio": early_mean,
        "response_estimator_final_mean_amplitude_ratio": final_mean,
        "response_estimator_stationarity_absolute_difference": estimator_difference,
        "response_estimator_stationarity_tolerance": estimator_tolerance,
        "response_estimator_stationarity_pass": estimator_stationarity_pass,
        "gain_limited_node_samples": gain_limited_node_samples,
        "force_limited_node_samples": force_limited_node_samples,
        "maximum_absolute_raw_gain": maximum_raw_gain,
        "gain_headroom_limit": gain_headroom_limit,
        "maximum_absolute_response_nodal_force_n": maximum_nodal_force,
        "force_headroom_limit_n": force_headroom_limit,
        "limit_headroom_pass": limit_pass,
        "history_mode": history_mode,
        "history_schema": expected_history_schema,
        "force_vector_component_count": compact_component_count,
        "compact_lag_provenance_pass": compact_lag_provenance_pass,
        "applied_power_sign_convention_pass": True,
        "startup_seed_integrity_evaluated": startup_seed_config is not None,
        "startup_seed_integrity_pass": (
            seed_integrity_pass
            and seed_active_samples > 0
            and seed_inactive_samples > 0
            if startup_seed_config is not None
            else None
        ),
        "startup_seed_active_samples": seed_active_samples,
        "startup_seed_inactive_samples": seed_inactive_samples,
    }
    if ambient_v2:
        result.update(
            {
                "response_model_schema_version": 2,
                "aerodynamic_speed_basis": AMBIENT_V2_AERODYNAMIC_SPEED_BASIS,
                "speed_basis_provenance_pass": v2_provenance_pass,
                "ambient_response_speed_min_mps": ambient_speed_min,
                "ambient_response_speed_max_mps": ambient_speed_max,
                "relative_crossflow_speed_min_mps": relative_speed_min,
                "relative_crossflow_speed_max_mps": relative_speed_max,
                "response_section_width_min_m": response_width_min,
                "response_section_width_max_m": response_width_max,
                "configured_response_reduced_velocity": (
                    expected_reduced_velocity
                ),
                "realized_response_reduced_velocity": (
                    expected_reduced_velocity
                ),
                "speed_basis_invariant_violations": 0,
                "activation_invariant_violations": 0,
            }
        )
    return result


_SOURCE_ZERO_FIELDS = (
    "aliased_frequency_count",
    "frequency_truncated_count",
    "source_clip_count",
    "force_coefficient_out_of_range_count",
    "nonfinite_count",
    "aoa_out_of_range_count",
    "re_out_of_range_count",
)


def _portable_path_has_suffix(path_value: Any, expected_relative: str) -> bool:
    """Compare path components without depending on a host/container prefix."""

    observed = PurePosixPath(str(path_value).replace("\\", "/"))
    expected = PurePosixPath(str(expected_relative).replace("\\", "/"))
    return (
        bool(observed.parts)
        and bool(expected.parts)
        and len(observed.parts) >= len(expected.parts)
        and observed.parts[-len(expected.parts) :] == expected.parts
    )


def _actual_file_identity(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": path.resolve().as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _compare_reported_file_identity(
    label: str,
    reported: Any,
    actual: Mapping[str, Any],
    *,
    expected_manifest: Mapping[str, Any] | None,
    expected_path_suffix: str,
    missing: list[str],
    mismatches: list[str],
) -> None:
    """Compare hash/size identities while treating absolute paths as metadata."""

    if not isinstance(reported, Mapping):
        missing.append(f"{label} file identity")
        return
    required = ("path", "sha256", "size_bytes")
    absent = [field for field in required if reported.get(field) is None]
    if absent:
        missing.append(f"{label} identity field(s) {absent}")
        return
    if not _portable_path_has_suffix(reported["path"], expected_path_suffix):
        mismatches.append(
            f"{label} reported path does not end with {expected_path_suffix!r}"
        )
    if str(reported["sha256"]) != str(actual["sha256"]):
        mismatches.append(f"{label} SHA-256 does not match the analyzed file")
    if int(reported["size_bytes"]) != int(actual["size_bytes"]):
        mismatches.append(f"{label} size does not match the analyzed file")
    if expected_manifest is not None:
        if str(reported["sha256"]) != str(expected_manifest["sha256"]):
            mismatches.append(f"{label} SHA-256 does not match the campaign manifest")
        if int(reported["size_bytes"]) != int(expected_manifest["size_bytes"]):
            mismatches.append(f"{label} size does not match the campaign manifest")


def _manifest_case_identity(
    manifest_files: Mapping[str, Mapping[str, Any]], relative_path: str
) -> Mapping[str, Any]:
    identity = manifest_files.get(PurePosixPath(relative_path).as_posix())
    if not isinstance(identity, Mapping):
        raise ValueError(
            f"Campaign manifest has no identity for required case file {relative_path!r}."
        )
    return identity


def _response_config_fingerprint(
    response_config: Mapping[str, Any],
) -> str:
    node_ids = response_config.get("node_ids")
    if not isinstance(node_ids, list) or not node_ids:
        raise ValueError("Response configuration has no ordered node_ids.")
    model = _response_model_from_config(response_config, node_ids)
    if not isinstance(model, QBladeResponseModel):
        raise ValueError(
            "Response configuration does not construct the required negative-damping model."
        )
    return model.config_fingerprint


def _validated_response_config_fingerprint(
    response_config: Mapping[str, Any], reported_fingerprint: Any
) -> str:
    expected = _response_config_fingerprint(response_config)
    if reported_fingerprint != expected:
        raise ValueError(
            "Run-summary response-model fingerprint does not reproduce the bound configuration."
        )
    return expected


def _runtime_identity_expected_manifest(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sha256": identity.get("sha256"),
        "size_bytes": identity.get("bytes"),
    }


def _audit_ambient_v2_runtime_contract(
    reproducibility: Mapping[str, Any],
    *,
    campaign_root: Path,
    runtime_contract: Mapping[str, Any],
    missing: list[str],
    mismatches: list[str],
) -> None:
    """Compare rich run-summary provenance to the frozen launch contract."""

    command_line = reproducibility.get("command_line")
    if (
        not isinstance(command_line, list)
        or len(command_line) < 2
        or not _portable_path_has_suffix(
            command_line[1], str(runtime_contract["runner"]["path"])
        )
    ):
        mismatches.append("runtime command line does not identify the pinned runner")
    runner_path = REPO_ROOT / str(runtime_contract["runner"]["path"])
    _compare_reported_file_identity(
        "QBlade runner",
        reproducibility.get("runner"),
        _actual_file_identity(runner_path),
        expected_manifest=_runtime_identity_expected_manifest(
            runtime_contract["runner"]
        ),
        expected_path_suffix=str(runtime_contract["runner"]["path"]),
        missing=missing,
        mismatches=mismatches,
    )

    input_files = reproducibility.get("input_files")
    if not isinstance(input_files, Mapping):
        missing.append("reproducibility.input_files object")
        input_files = {}
    filter_path = campaign_root / str(runtime_contract["result_filter"]["path"])
    _compare_reported_file_identity(
        "QBlade result filter",
        input_files.get("result_filter"),
        _actual_file_identity(filter_path),
        expected_manifest=_runtime_identity_expected_manifest(
            runtime_contract["result_filter"]
        ),
        expected_path_suffix=str(runtime_contract["result_filter"]["path"]),
        missing=missing,
        mismatches=mismatches,
    )

    qblade = reproducibility.get("qblade")
    if not isinstance(qblade, Mapping):
        missing.append("reproducibility.qblade object")
        qblade = {}
    reported_qblade = qblade.get("library")
    expected_qblade = runtime_contract["qblade_library"]
    if not isinstance(reported_qblade, Mapping):
        missing.append("QBlade library identity")
    else:
        if not _portable_path_has_suffix(
            reported_qblade.get("path", ""), str(expected_qblade["path"])
        ):
            mismatches.append("QBlade library path differs from the runtime contract")
        if (
            reported_qblade.get("sha256") != expected_qblade["sha256"]
            or reported_qblade.get("size_bytes") != expected_qblade["bytes"]
        ):
            mismatches.append("QBlade library identity differs from the runtime contract")

    reported_bridge = reproducibility.get("bridge_library")
    expected_bridge = runtime_contract["bridge_library"]
    if not isinstance(reported_bridge, Mapping):
        missing.append("VorLap/QBlade bridge identity")
    else:
        if not _portable_path_has_suffix(
            reported_bridge.get("path", ""), str(expected_bridge["path"])
        ):
            mismatches.append("Bridge library path differs from the runtime contract")
        if (
            reported_bridge.get("sha256") != expected_bridge["sha256"]
            or reported_bridge.get("size_bytes") != expected_bridge["bytes"]
        ):
            mismatches.append("Bridge library identity differs from the runtime contract")

    for field in (
        "vorlap_python_runtime_assets",
        "vorlap_python_runtime_fingerprint",
        "qblade_runtime_dependencies",
        "qblade_runtime_dependency_fingerprint",
    ):
        if field not in reproducibility:
            missing.append(f"reproducibility.{field}")
        elif reproducibility[field] != runtime_contract[field]:
            mismatches.append(f"reproducibility.{field} differs from the runtime contract")
    dependencies = reproducibility.get("qblade_runtime_dependencies")
    wrapper = runtime_contract["qblade_sil_wrapper"]
    if isinstance(dependencies, list):
        wrapper_rows = [
            row
            for row in dependencies
            if isinstance(row, Mapping) and row.get("path") == wrapper["path"]
        ]
        if len(wrapper_rows) != 1 or dict(wrapper_rows[0]) != dict(wrapper):
            mismatches.append("QBlade SIL wrapper identity differs from the runtime contract")

    numerical = runtime_contract["numerical_runtime"]
    python_metadata = reproducibility.get("python")
    packages = reproducibility.get("packages")
    if not isinstance(python_metadata, Mapping):
        missing.append("reproducibility.python object")
    else:
        if python_metadata.get("implementation") != numerical["python_implementation"]:
            mismatches.append("Python implementation differs from the runtime contract")
        reported_version = str(python_metadata.get("version", "")).split()[0]
        if reported_version != numerical["python_version"]:
            mismatches.append("Python version differs from the runtime contract")
    if not isinstance(packages, Mapping):
        missing.append("reproducibility.packages object")
    else:
        for name, expected in numerical["packages"].items():
            if packages.get(name) != expected:
                mismatches.append(
                    f"Python package {name} differs from the runtime contract"
                )


def _audit_ambient_v2_checkpoint_fingerprint(
    summary: Mapping[str, Any],
    *,
    result_directory: Path,
    expected_fingerprint: str,
    runtime_contract: Mapping[str, Any] | None,
    missing: list[str],
    mismatches: list[str],
) -> None:
    restart = summary.get("restart")
    if not isinstance(restart, Mapping):
        missing.append("run-summary restart object")
        return
    raw_path = restart.get("latest_checkpoint")
    if raw_path in (None, ""):
        return
    checkpoint_path = Path(str(raw_path)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = result_directory / checkpoint_path
    try:
        checkpoint_path = checkpoint_path.resolve(strict=True)
        checkpoint_path.relative_to(Path(result_directory).resolve())
    except (OSError, ValueError):
        mismatches.append("latest checkpoint escapes or is absent from the case output")
        return
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        mismatches.append("latest checkpoint is not a regular case-output file")
        return
    try:
        metadata = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        controller = metadata["sil_vorlap_state"]["runtime"]["controller"]
        model_state = controller["response_model_state"]
        diagnostics = controller["response_diagnostics_summary"]
    except (KeyError, TypeError, json.JSONDecodeError, OSError):
        missing.append("checkpoint response-model provenance")
        return
    if model_state.get("config_fingerprint") != expected_fingerprint:
        mismatches.append("checkpoint response-model fingerprint differs")
    if (
        diagnostics.get("response_config_fingerprint_sha256")
        != expected_fingerprint
    ):
        mismatches.append("checkpoint response-diagnostics fingerprint differs")
    if runtime_contract is not None:
        checkpoint_identity = metadata.get("checkpoint_identity")
        if not isinstance(checkpoint_identity, Mapping):
            missing.append("checkpoint runtime identity")
        else:
            for field in (
                "qblade_runtime_dependencies",
                "qblade_runtime_dependency_fingerprint",
            ):
                if checkpoint_identity.get(field) != runtime_contract.get(field):
                    mismatches.append(
                        f"checkpoint {field} differs from the runtime contract"
                    )


def runtime_case_audit(
    result_directory: Path,
    *,
    entry: Mapping[str, Any],
    campaign_root: Path,
    manifest_files: Mapping[str, Mapping[str, Any]],
    result_path: Path,
    expected_history_mode: str | None = None,
    runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit a runtime and bind every artifact to its immutable case inputs."""

    result_directory = Path(result_directory)
    summary_path, summary_problem = _unique_matching_file(
        result_directory, "*_run_summary.json"
    )
    history_path, history_problem = _unique_matching_file(
        result_directory, "*_sil_vorlap_history.csv"
    )
    artifacts_present = summary_path is not None or history_path is not None
    problems = [value for value in (summary_problem, history_problem) if value]
    if problems:
        return {
            "evaluated": False,
            "pass": False,
            "artifacts_present": artifacts_present,
            "provenance_evaluated": False,
            "provenance_pass": False,
            "identity_mismatch": False,
            "problems": problems,
            "summary_path": None if summary_path is None else summary_path.resolve().as_posix(),
            "history_path": None if history_path is None else history_path.resolve().as_posix(),
        }

    assert summary_path is not None and history_path is not None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        missing_identities: list[str] = []
        identity_mismatches: list[str] = []
        campaign_root = Path(campaign_root).resolve()
        result_path = Path(result_path)
        sim_relative = PurePosixPath(str(entry["sim_path"])).as_posix()
        config_relative = PurePosixPath(str(entry["parameter_file"])).as_posix()
        sim_path = campaign_root.joinpath(*PurePosixPath(sim_relative).parts)
        config_path = campaign_root.joinpath(*PurePosixPath(config_relative).parts)
        sim_manifest = _manifest_case_identity(manifest_files, sim_relative)
        config_manifest = _manifest_case_identity(manifest_files, config_relative)
        sim_actual = _actual_file_identity(sim_path)
        config_actual = _actual_file_identity(config_path)

        reproducibility = summary.get("reproducibility")
        if not isinstance(reproducibility, Mapping):
            missing_identities.append("reproducibility object")
            reproducibility = {}
        input_files = reproducibility.get("input_files")
        if not isinstance(input_files, Mapping):
            missing_identities.append("reproducibility.input_files object")
            input_files = {}
        _compare_reported_file_identity(
            "simulation",
            input_files.get("sim"),
            sim_actual,
            expected_manifest=sim_manifest,
            expected_path_suffix=sim_relative,
            missing=missing_identities,
            mismatches=identity_mismatches,
        )
        _compare_reported_file_identity(
            "VorLap configuration",
            input_files.get("vorlap_config"),
            config_actual,
            expected_manifest=config_manifest,
            expected_path_suffix=config_relative,
            missing=missing_identities,
            mismatches=identity_mismatches,
        )
        if not _portable_path_has_suffix(summary.get("sim", ""), sim_relative):
            identity_mismatches.append("run-summary sim path does not identify this case")
        if not _portable_path_has_suffix(
            summary.get("sil_vorlap_parameter_file", ""), config_relative
        ):
            identity_mismatches.append(
                "run-summary SIL parameter path does not identify this case"
            )

        qblade_settings = reproducibility.get("qblade_case_settings")
        if not isinstance(qblade_settings, Mapping):
            missing_identities.append("reproducibility.qblade_case_settings object")
            qblade_settings = {}

        def require_float_setting(name: str, expected: float) -> None:
            try:
                observed = float(qblade_settings[name])
            except (KeyError, TypeError, ValueError):
                missing_identities.append(f"QBlade setting {name}")
                return
            if not math.isclose(
                observed,
                float(expected),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                identity_mismatches.append(
                    f"QBlade setting {name}={observed!r} does not match {expected!r}"
                )

        require_float_setting("MEANINF", float(entry["mean_inflow_mps"]))
        require_float_setting("TIMESTEP", float(entry["timestep_s"]))
        require_float_setting("NUMTIMESTEPS", float(entry["num_steps"]))

        turbine_setting = qblade_settings.get("TURBFILE")
        if not turbine_setting:
            missing_identities.append("QBlade setting TURBFILE")
            turbine_relative = ""
        else:
            turbine_candidate = (
                campaign_root
                / Path(sim_relative).parent
                / str(turbine_setting)
            ).resolve()
            try:
                turbine_relative = turbine_candidate.relative_to(campaign_root).as_posix()
            except ValueError:
                turbine_relative = ""
                identity_mismatches.append("QBlade TURBFILE escapes the campaign root")
        if turbine_relative:
            turbine_manifest = _manifest_case_identity(
                manifest_files, turbine_relative
            )
            turbine_actual = _actual_file_identity(
                campaign_root.joinpath(*PurePosixPath(turbine_relative).parts)
            )
            _compare_reported_file_identity(
                "turbine",
                input_files.get("turbine"),
                turbine_actual,
                expected_manifest=turbine_manifest,
                expected_path_suffix=turbine_relative,
                missing=missing_identities,
                mismatches=identity_mismatches,
            )

        result_actual = _actual_file_identity(result_path)
        _compare_reported_file_identity(
            "result",
            summary.get("result_file_identity"),
            result_actual,
            expected_manifest=None,
            expected_path_suffix=result_path.name,
            missing=missing_identities,
            mismatches=identity_mismatches,
        )
        if not _portable_path_has_suffix(
            summary.get("result_file", ""), result_path.name
        ):
            identity_mismatches.append("run-summary result path does not identify the analyzed result")

        response_summary = summary.get("sil_vorlap_response_diagnostics_summary")
        if not isinstance(response_summary, Mapping):
            raise ValueError("run summary has no SIL response diagnostics summary")
        source_coverage = summary.get("sil_vorlap_source_coverage")
        if not isinstance(source_coverage, list) or not source_coverage:
            raise ValueError("run summary has no SIL source-coverage rows")
        source_counts: dict[str, int] = {}
        for field in _SOURCE_ZERO_FIELDS:
            values = [row.get(field) for row in source_coverage if isinstance(row, Mapping)]
            if len(values) != len(source_coverage) or any(value is None for value in values):
                raise ValueError(f"source coverage is missing {field}")
            source_counts[field] = int(sum(int(value) for value in values))
        source_limits_pass = all(value == 0 for value in source_counts.values())

        expected_steps = int(entry["num_steps"])
        source_sample_counts = [
            int(row.get("samples", -1))
            for row in source_coverage
            if isinstance(row, Mapping)
        ]
        source_coverage_complete = (
            len(source_sample_counts) == len(source_coverage)
            and all(value == expected_steps for value in source_sample_counts)
        )
        execution_pass = (
            not bool(summary.get("failed"))
            and not bool(summary.get("stopped"))
            and int(summary.get("completed_steps", -1)) == expected_steps
            and int(summary.get("requested_steps", -1)) == expected_steps
            and int(summary.get("sil_vorlap_updates", -1)) == expected_steps
            and summary.get("coupling_mode") == "sil-vorlap"
            and response_summary.get("model_type") == "modal_negative_damping"
            and int(response_summary.get("samples", -1)) == expected_steps
        )
        response_limit_count_fields = (
            "gain_limited_steps",
            "force_limited_steps",
            "gain_limited_node_samples",
            "force_limited_node_samples",
        )
        response_limit_counts = {
            field: int(response_summary.get(field, -1))
            for field in response_limit_count_fields
        }
        response_summary_limits_pass = all(
            value == 0 for value in response_limit_counts.values()
        )
        response_config = entry["response_model_config"]
        ambient_v2 = response_config.get("schema_version") == 2
        expected_response_config_fingerprint = None
        response_speed_summary_pass = True
        if ambient_v2:
            expected_response_config_fingerprint = _response_config_fingerprint(
                response_config
            )
            expected_u_star = float(entry["reduced_velocity"])
            expected_ambient_speed = float(entry["mean_inflow_mps"])
            expected_width = (
                float(response_config["response_section_width_m"])
                if isinstance(
                    response_config.get("response_section_width_m"),
                    (int, float),
                )
                else DIAMETER_M
            )
            try:
                fingerprint = str(
                    response_summary["response_config_fingerprint_sha256"]
                )
                expected_response_config_fingerprint = (
                    _validated_response_config_fingerprint(
                        response_config, fingerprint
                    )
                )
                response_speed_summary_pass = all(
                    (
                        int(response_summary["response_model_schema_version"]) == 2,
                        response_summary["aerodynamic_speed_basis"]
                        == AMBIENT_V2_AERODYNAMIC_SPEED_BASIS,
                        response_summary["response_section_width_basis"]
                        == (
                            "explicit_response_section_width_m"
                            if response_config.get("response_section_width_m")
                            is not None
                            else "declared_ambient_crossflow_angle"
                        ),
                        fingerprint == expected_response_config_fingerprint,
                        math.isclose(
                            float(
                                response_summary[
                                    "response_declared_ambient_crossflow_speed_min_mps"
                                ]
                            ),
                            expected_ambient_speed,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-12,
                        ),
                        math.isclose(
                            float(
                                response_summary[
                                    "response_declared_ambient_crossflow_speed_max_mps"
                                ]
                            ),
                            expected_ambient_speed,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-12,
                        ),
                        0.0
                        <= float(
                            response_summary[
                                "response_relative_crossflow_speed_min_mps"
                            ]
                        )
                        <= float(
                            response_summary[
                                "response_relative_crossflow_speed_max_mps"
                            ]
                        ),
                        math.isclose(
                            float(response_summary["response_section_width_min_m"]),
                            expected_width,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-12,
                        ),
                        math.isclose(
                            float(response_summary["response_section_width_max_m"]),
                            expected_width,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-12,
                        ),
                        math.isclose(
                            float(
                                response_summary[
                                    "configured_response_reduced_velocity_min"
                                ]
                            ),
                            expected_u_star,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-12,
                        ),
                        math.isclose(
                            float(
                                response_summary[
                                    "configured_response_reduced_velocity_max"
                                ]
                            ),
                            expected_u_star,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-12,
                        ),
                        math.isclose(
                            float(
                                response_summary[
                                    "realized_response_reduced_velocity_min"
                                ]
                            ),
                            expected_u_star,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-12,
                        ),
                        math.isclose(
                            float(
                                response_summary[
                                    "realized_response_reduced_velocity_max"
                                ]
                            ),
                            expected_u_star,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-12,
                        ),
                        int(response_summary["speed_basis_invariant_violations"])
                        == 0,
                        int(response_summary["activation_invariant_violations"])
                        == 0,
                        int(
                            response_summary[
                                "speed_basis_invariant_violation_steps"
                            ]
                        )
                        == 0,
                        int(
                            response_summary[
                                "activation_invariant_violation_steps"
                            ]
                        )
                        == 0,
                    )
                )
            except (KeyError, TypeError, ValueError):
                response_speed_summary_pass = False
            if runtime_contract is None:
                missing_identities.append("ambient-v2 runtime contract")
            else:
                _audit_ambient_v2_runtime_contract(
                    reproducibility,
                    campaign_root=campaign_root,
                    runtime_contract=runtime_contract,
                    missing=missing_identities,
                    mismatches=identity_mismatches,
                )
            native_evidence = summary.get("native_callback_evidence")
            if not isinstance(native_evidence, Mapping):
                missing_identities.append("native_callback_evidence object")
            elif (
                bool(native_evidence.get("log_file_exists"))
                or int(native_evidence.get("update_init_count", 0) or 0) != 0
                or int(native_evidence.get("update_count", 0) or 0) != 0
                or int(
                    native_evidence.get("update_count_observed_max", 0) or 0
                )
                != 0
            ):
                identity_mismatches.append(
                    "unexpected native callback activity in SIL-VorLap run"
                )
            _audit_ambient_v2_checkpoint_fingerprint(
                summary,
                result_directory=result_directory,
                expected_fingerprint=expected_response_config_fingerprint,
                runtime_contract=runtime_contract,
                missing=missing_identities,
                mismatches=identity_mismatches,
            )
        history_metadata = summary.get("sil_vorlap_history")
        if not isinstance(history_metadata, Mapping):
            missing_identities.append("sil_vorlap_history metadata")
            history_metadata = {}
        reported_top_history_mode = summary.get("sil_vorlap_history_mode")
        reported_nested_history_mode = history_metadata.get("mode")
        if reported_top_history_mode is None:
            missing_identities.append("sil_vorlap_history_mode")
        if reported_nested_history_mode is None:
            missing_identities.append("sil_vorlap_history mode")
        if expected_history_mode is None:
            history_mode = str(reported_nested_history_mode or "full")
        else:
            history_mode = str(expected_history_mode)
        if history_mode not in {"full", "compact"}:
            raise ValueError("Required SIL-history mode must be 'full' or 'compact'.")
        if (
            reported_top_history_mode is not None
            and str(reported_top_history_mode) != history_mode
        ):
            identity_mismatches.append(
                "run-summary SIL-history mode does not match the required mode"
            )
        if (
            reported_nested_history_mode is not None
            and str(reported_nested_history_mode) != history_mode
        ):
            identity_mismatches.append(
                "nested SIL-history mode does not match the required mode"
            )
        startup_seed_config = response_config.get("startup_seed")
        if not isinstance(startup_seed_config, Mapping) or not bool(
            startup_seed_config.get("enabled", False)
        ):
            startup_seed_config = None
        configured_node_ids = response_config.get("node_ids")
        expected_force_components = (
            3 * len(configured_node_ids)
            if isinstance(configured_node_ids, list) and configured_node_ids
            else None
        )
        history = response_history_audit(
            history_path,
            natural_frequency_hz=float(entry["structure"]["natural_frequency_hz"]),
            max_abs_gain=float(response_config["max_abs_gain"]),
            max_abs_nodal_force_n=float(
                response_config["max_abs_nodal_force_n"]
            ),
            history_mode=history_mode,
            expected_force_vector_component_count=expected_force_components,
            startup_seed_config=startup_seed_config,
            response_model_config=response_config,
            expected_reduced_velocity=(
                float(entry["reduced_velocity"]) if ambient_v2 else None
            ),
            expected_ambient_response_speed_mps=(
                float(entry["mean_inflow_mps"]) if ambient_v2 else None
            ),
        )
        _compare_reported_file_identity(
            "SIL history",
            history_metadata.get("file_identity"),
            history["file_identity"],
            expected_manifest=None,
            expected_path_suffix=history_path.name,
            missing=missing_identities,
            mismatches=identity_mismatches,
        )
        if not _portable_path_has_suffix(
            history_metadata.get("path", ""), history_path.name
        ):
            identity_mismatches.append(
                "run-summary SIL-history path does not identify the analyzed history"
            )
        expected_history_schema = str(history["history_schema"])
        for field, observed, expected in (
            ("rows", history_metadata.get("rows"), int(history["samples"])),
            ("columns", history_metadata.get("columns"), int(history["columns"])),
            ("schema", history_metadata.get("schema"), expected_history_schema),
            ("mode", history_metadata.get("mode"), history_mode),
        ):
            if observed is None:
                missing_identities.append(f"SIL-history metadata {field}")
            elif str(observed) != str(expected):
                identity_mismatches.append(
                    f"SIL-history metadata {field} does not match the analyzed history"
                )
        if history_mode == "compact":
            digest_schema = history_metadata.get("vector_digest_schema")
            if digest_schema is None:
                missing_identities.append("SIL-history metadata vector_digest_schema")
            elif str(digest_schema) != "vorlap-force-vector-sha256-v1":
                identity_mismatches.append(
                    "SIL-history vector-digest schema does not match the analyzed history"
                )
            if history_metadata.get("streaming") is not True:
                missing_identities.append("SIL-history streaming=true metadata")
            if int(history_metadata.get("retained_rows", -1)) != 0:
                identity_mismatches.append(
                    "Compact SIL history retained in-memory rows"
                )
            if int(
                history_metadata.get("discarded_uncheckpointed_tail_bytes", -1)
            ) != 0:
                identity_mismatches.append(
                    "Compact SIL history discarded an uncheckpointed tail"
                )
            if (
                expected_force_components is None
                or int(history_metadata.get("node_count", -1)) * 3
                != expected_force_components
            ):
                identity_mismatches.append(
                    "Compact SIL-history node count does not match the controller"
                )

        if startup_seed_config is None:
            seed_summary_pass = True
            seed_response = None
        else:
            seed_summary_integer_fields = (
                "startup_seed_samples",
                "startup_seed_active_steps",
                "startup_seed_force_limited_steps",
                "startup_seed_force_limited_node_samples",
                "max_startup_seed_force_limited_nodes",
            )
            try:
                seed_summary_counts = {
                    field: int(response_summary[field])
                    for field in seed_summary_integer_fields
                }
                seed_summary_target_ratio = float(
                    response_summary["startup_seed_target_amplitude_ratio"]
                )
                seed_summary_target_m = float(
                    response_summary["startup_seed_target_amplitude_m"]
                )
                seed_summary_max_nodal_force = float(
                    response_summary["max_startup_seed_raw_nodal_force_n"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "run summary lacks complete startup-seed diagnostics"
                ) from exc
            seed_summary_pass = (
                seed_summary_counts["startup_seed_samples"] == expected_steps
                and seed_summary_counts["startup_seed_active_steps"]
                == int(history["startup_seed_active_samples"])
                and seed_summary_counts["startup_seed_active_steps"] > 0
                and seed_summary_counts["startup_seed_active_steps"] < expected_steps
                and seed_summary_counts["startup_seed_force_limited_steps"] == 0
                and seed_summary_counts[
                    "startup_seed_force_limited_node_samples"
                ]
                == 0
                and seed_summary_counts["max_startup_seed_force_limited_nodes"] == 0
                and math.isclose(
                    seed_summary_target_ratio,
                    float(startup_seed_config["target_amplitude_ratio"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and math.isclose(
                    seed_summary_target_m,
                    float(startup_seed_config["target_amplitude_ratio"])
                    * DIAMETER_M,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and math.isfinite(seed_summary_max_nodal_force)
                and seed_summary_max_nodal_force
                <= LIMIT_HEADROOM_FRACTION
                * float(response_config["max_abs_nodal_force_n"])
            )
            seed_response = startup_seed_response_audit(
                result_path,
                natural_frequency_hz=float(
                    entry["structure"]["natural_frequency_hz"]
                ),
                seed_duration_cycles=float(startup_seed_config["duration_cycles"]),
                expected_timestep_s=float(entry["timestep_s"]),
                expected_samples=expected_steps,
            )

        time_diagnostics = summary.get("qblade_export_time_diagnostics")
        if not isinstance(time_diagnostics, Mapping):
            missing_identities.append("qblade_export_time_diagnostics object")
            time_diagnostics = {}
        if int(time_diagnostics.get("samples", -1)) != expected_steps:
            identity_mismatches.append(
                "QBlade result sample count does not match the manifest case"
            )
        provenance_evaluated = not missing_identities
        provenance_pass = provenance_evaluated and not identity_mismatches
        pass_value = (
            provenance_pass
            and execution_pass
            and source_coverage_complete
            and source_limits_pass
            and response_summary_limits_pass
            and response_speed_summary_pass
            and int(history["samples"]) == expected_steps
            and bool(history["runtime_model_pass"])
            and bool(history["response_estimator_stationarity_pass"])
            and bool(history["limit_headroom_pass"])
            and bool(history["compact_lag_provenance_pass"])
            and (
                not ambient_v2
                or bool(history["speed_basis_provenance_pass"])
            )
            and (
                startup_seed_config is None
                or (
                    bool(history["startup_seed_integrity_pass"])
                    and seed_summary_pass
                    and bool(seed_response and seed_response["pass"])
                )
            )
        )
        return {
            "evaluated": True,
            "pass": pass_value,
            "artifacts_present": True,
            "provenance_evaluated": provenance_evaluated,
            "provenance_pass": provenance_pass,
            "identity_mismatch": bool(identity_mismatches),
            "missing_identity_fields": missing_identities,
            "identity_mismatches": identity_mismatches,
            "problems": [*missing_identities, *identity_mismatches],
            "summary_path": summary_path.resolve().as_posix(),
            "history_path": history_path.resolve().as_posix(),
            "input_identities": {
                "sim": sim_actual,
                "vorlap_config": config_actual,
                "turbine": turbine_actual if turbine_relative else None,
                "result": result_actual,
                "history": history["file_identity"],
            },
            "execution_pass": execution_pass,
            "source_sample_counts": source_sample_counts,
            "source_coverage_complete": source_coverage_complete,
            "source_limit_counts": source_counts,
            "source_limits_pass": source_limits_pass,
            "response_limit_counts": response_limit_counts,
            "response_summary_limits_pass": response_summary_limits_pass,
            "response_speed_summary_pass": response_speed_summary_pass,
            "expected_response_config_fingerprint_sha256": (
                expected_response_config_fingerprint
            ),
            "startup_seed_summary_pass": seed_summary_pass,
            "startup_seed_response": seed_response,
            "history": history,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        return {
            "evaluated": False,
            "pass": False,
            "artifacts_present": True,
            "provenance_evaluated": False,
            "provenance_pass": False,
            "identity_mismatch": False,
            "problems": [str(exc)],
            "summary_path": summary_path.resolve().as_posix(),
            "history_path": history_path.resolve().as_posix(),
        }


def _band_endpoints(
    reduced_velocity: Sequence[float],
    amplitude: Sequence[float],
    threshold: float,
) -> tuple[float | None, float | None]:
    """Linearly interpolate threshold crossings around the largest peak."""

    u_star = np.asarray(reduced_velocity, dtype=float)
    response = np.asarray(amplitude, dtype=float)
    if (
        u_star.size < 3
        or response.shape != u_star.shape
        or np.any(~np.isfinite(u_star))
        or np.any(~np.isfinite(response))
        or np.any(np.diff(u_star) <= 0.0)
        or not math.isfinite(threshold)
        or threshold <= 0.0
    ):
        return None, None
    peak_index = int(np.argmax(response))
    if response[peak_index] < threshold:
        return None, None

    low = None
    for index in range(peak_index - 1, -1, -1):
        if response[index] < threshold <= response[index + 1]:
            fraction = (threshold - response[index]) / (
                response[index + 1] - response[index]
            )
            low = float(u_star[index] + fraction * (u_star[index + 1] - u_star[index]))
            break
    high = None
    for index in range(peak_index, response.size - 1):
        if response[index] >= threshold > response[index + 1]:
            fraction = (threshold - response[index]) / (
                response[index + 1] - response[index]
            )
            high = float(u_star[index] + fraction * (u_star[index + 1] - u_star[index]))
            break
    return low, high


def _interpolated_value(
    reduced_velocity: Sequence[float],
    amplitude: Sequence[float],
    target_reduced_velocity: float,
) -> float | None:
    u_star = np.asarray(reduced_velocity, dtype=float)
    response = np.asarray(amplitude, dtype=float)
    if (
        u_star.size < 2
        or response.shape != u_star.shape
        or target_reduced_velocity < u_star[0]
        or target_reduced_velocity > u_star[-1]
    ):
        return None
    return float(np.interp(target_reduced_velocity, u_star, response))


def material_improvement_metrics(
    *,
    reference_u: Sequence[float],
    reference_amplitude: Sequence[float],
    response_u: Sequence[float],
    response_amplitude: Sequence[float],
    frozen_u: Sequence[float],
    frozen_amplitude: Sequence[float],
) -> dict[str, Any]:
    """Quantify improvement over zero-VIV and the live frozen-source curve."""

    reference_u = np.asarray(reference_u, dtype=float)
    reference_amplitude = np.asarray(reference_amplitude, dtype=float)
    response_u = np.asarray(response_u, dtype=float)
    response_amplitude = np.asarray(response_amplitude, dtype=float)
    frozen_u = np.asarray(frozen_u, dtype=float)
    frozen_amplitude = np.asarray(frozen_amplitude, dtype=float)
    inputs = (
        (reference_u, reference_amplitude),
        (response_u, response_amplitude),
        (frozen_u, frozen_amplitude),
    )
    if any(
        u.size < 3
        or a.shape != u.shape
        or np.any(~np.isfinite(u))
        or np.any(~np.isfinite(a))
        or np.any(np.diff(u) <= 0.0)
        for u, a in inputs
    ):
        return {"evaluated": False, "pass": False, "reason": "invalid curve input"}
    if (
        response_u[0] > reference_u[0]
        or response_u[-1] < reference_u[-1]
        or frozen_u[0] > reference_u[0]
        or frozen_u[-1] < reference_u[-1]
    ):
        return {
            "evaluated": False,
            "pass": False,
            "reason": "response and frozen curves must span every reference point",
        }
    response_at_reference = np.interp(reference_u, response_u, response_amplitude)
    frozen_at_reference = np.interp(reference_u, frozen_u, frozen_amplitude)
    no_viv_at_reference = np.zeros_like(reference_amplitude)
    response_rmse = float(
        np.sqrt(np.mean((response_at_reference - reference_amplitude) ** 2))
    )
    frozen_rmse = float(
        np.sqrt(np.mean((frozen_at_reference - reference_amplitude) ** 2))
    )
    no_viv_rmse = float(np.sqrt(np.mean(reference_amplitude**2)))
    skill_over_frozen = (
        None if frozen_rmse == 0.0 else 1.0 - response_rmse / frozen_rmse
    )
    skill_over_no_viv = (
        None if no_viv_rmse == 0.0 else 1.0 - response_rmse / no_viv_rmse
    )

    # Use one paired reference grid for every skill and peak metric.  Mixing
    # raw-grid peak locations with interpolated-grid RMSE values can reward a
    # baseline merely because it was sampled more or less densely.
    reference_peak_index = int(np.argmax(reference_amplitude))
    response_peak_index = int(np.argmax(response_at_reference))
    frozen_peak_index = int(np.argmax(frozen_at_reference))
    reference_peak_amplitude = float(reference_amplitude[reference_peak_index])
    reference_peak_u = float(reference_u[reference_peak_index])
    response_peak_amplitude_error = abs(
        float(response_at_reference[response_peak_index]) - reference_peak_amplitude
    )
    frozen_peak_amplitude_error = abs(
        float(frozen_at_reference[frozen_peak_index]) - reference_peak_amplitude
    )
    response_peak_location_error = abs(
        float(reference_u[response_peak_index]) - reference_peak_u
    )
    frozen_peak_location_error = abs(
        float(reference_u[frozen_peak_index]) - reference_peak_u
    )
    peak_amplitude_improvement_pass = (
        response_peak_amplitude_error
        <= 0.5 * frozen_peak_amplitude_error
    )
    peak_location_improvement_pass = (
        response_peak_location_error <= 0.5 * frozen_peak_location_error
    )
    passed = (
        skill_over_frozen is not None
        and skill_over_no_viv is not None
        and skill_over_frozen >= MATERIAL_IMPROVEMENT_MINIMUM_SKILL
        and skill_over_no_viv >= MATERIAL_IMPROVEMENT_MINIMUM_SKILL
        and peak_amplitude_improvement_pass
        and peak_location_improvement_pass
    )
    return {
        "evaluated": True,
        "pass": passed,
        "reason": None,
        "reference_points": int(reference_u.size),
        "baseline_pairing": "response, frozen, and zero-VIV evaluated on the shared reference U* grid",
        "response_rmse_tip_amplitude_over_diameter": response_rmse,
        "frozen_rmse_tip_amplitude_over_diameter": frozen_rmse,
        "no_viv_rmse_tip_amplitude_over_diameter": no_viv_rmse,
        "skill_over_frozen": skill_over_frozen,
        "skill_over_no_viv": skill_over_no_viv,
        "minimum_skill_required": MATERIAL_IMPROVEMENT_MINIMUM_SKILL,
        "response_peak_amplitude_error": response_peak_amplitude_error,
        "frozen_peak_amplitude_error": frozen_peak_amplitude_error,
        "peak_amplitude_improvement_pass": peak_amplitude_improvement_pass,
        "response_peak_location_error_u_star": response_peak_location_error,
        "frozen_peak_location_error_u_star": frozen_peak_location_error,
        "peak_location_improvement_pass": peak_location_improvement_pass,
        "scope": (
            "material improvement on the Sc=5.18 calibration target; not an "
            "independent accuracy, validation, or fatigue-conservatism claim"
        ),
    }


def _plot_response_curves(
    path: Path,
    curve_rows: Sequence[Mapping[str, Any]],
    figure9_path: Path,
    peak_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    colors = {2.94: "#0072B2", 5.18: "#D55E00", 7.02: "#009E73"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for sc in sorted({float(row["scruton_number"]) for row in curve_rows}):
        selected = sorted(
            (row for row in curve_rows if float(row["scruton_number"]) == sc),
            key=lambda row: float(row["reduced_velocity"]),
        )
        ax.plot(
            [float(row["reduced_velocity"]) for row in selected],
            [float(row["tip_peak_amplitude_over_diameter"]) for row in selected],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=colors[sc],
            label="Response-aware VorLap–QBlade (this work)",
        )
    with figure9_path.open(newline="", encoding="utf-8") as handle:
        figure9_rows = list(csv.DictReader(handle))
    exp = [row for row in figure9_rows if row["series"] == "experiment_lupi2021"]
    hgast = [row for row in figure9_rows if row["series"] == CALIBRATION_SERIES]
    ax.plot(
        [float(row["reduced_velocity"]) for row in hgast],
        [float(row["tip_amplitude_over_diameter"]) for row in hgast],
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="hGAST numerical training curve (digitized)",
        zorder=3,
    )
    ax.scatter(
        [float(row["reduced_velocity"]) for row in exp],
        [float(row["tip_amplitude_over_diameter"]) for row in exp],
        s=14,
        facecolors="none",
        edgecolors="#222222",
        linewidths=0.7,
        label="Lupi et al. experiment (digitized)",
        zorder=4,
    )
    for row in peak_rows:
        if row["role"] != "held_out":
            continue
        sc = float(row["scruton_number"])
        ax.scatter(
            [float(row["reference_peak_u_star"])],
            [float(row["hgast_reference_peak_over_diameter"])],
            marker="x",
            s=45,
            linewidths=1.4,
            color=colors[sc],
            zorder=5,
        )
    ax.set_xlabel(r"Reduced velocity, $U^*=U/(f_nD)$")
    ax.set_ylabel(r"Tip peak amplitude, $y_{\max}/D$")
    ax.set_xlim(4.0, 8.5)
    ax.set_ylim(bottom=0.0)
    ax.set_title(r"Calibration evidence at $Sc=5.18$", fontsize=10)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(
        frameon=False,
        fontsize=7.4,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        columnspacing=1.5,
        handlelength=3.0,
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.14, top=0.72)
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def calibration_curve_comparison(
    curve_rows: Sequence[Mapping[str, Any]],
    figure9_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Interpolate the Sc=5.18 QBlade curve onto every hGAST digitization point."""

    predicted = sorted(
        (row for row in curve_rows if float(row["scruton_number"]) == CALIBRATION_SC),
        key=lambda row: float(row["reduced_velocity"]),
    )
    reference_u, reference_amplitude = _read_figure9_series(figure9_path, CALIBRATION_SERIES)
    if len(predicted) < 3:
        return [], {
            "evaluated": False,
            "pass": False,
            "reason": "at least three Sc=5.18 reduced-velocity results are required",
            "rmse_tip_amplitude_over_diameter": None,
            "acceptance_tolerance": 0.05,
        }
    predicted_u = np.asarray([float(row["reduced_velocity"]) for row in predicted])
    predicted_amplitude = np.asarray(
        [float(row["tip_peak_amplitude_over_diameter"]) for row in predicted]
    )
    if predicted_u[0] > reference_u[0] or predicted_u[-1] < reference_u[-1]:
        return [], {
            "evaluated": False,
            "pass": False,
            "reason": (
                f"Sc=5.18 prediction range [{predicted_u[0]:.4g}, {predicted_u[-1]:.4g}] "
                f"does not span the hGAST range [{reference_u[0]:.4g}, {reference_u[-1]:.4g}]"
            ),
            "rmse_tip_amplitude_over_diameter": None,
            "acceptance_tolerance": 0.05,
        }
    interpolated = np.interp(reference_u, predicted_u, predicted_amplitude)
    errors = interpolated - reference_amplitude
    rows = [
        {
            "scruton_number": CALIBRATION_SC,
            "reduced_velocity": float(u_star),
            "qblade_interpolated_peak_over_diameter": float(prediction),
            "hgast_reference_peak_over_diameter": float(reference),
            "signed_error": float(error),
        }
        for u_star, prediction, reference, error in zip(
            reference_u, interpolated, reference_amplitude, errors
        )
    ]
    rmse = float(np.sqrt(np.mean(errors**2)))
    tolerance = 0.05
    return rows, {
        "evaluated": True,
        "pass": rmse <= tolerance,
        "reason": None,
        "points": len(rows),
        "rmse_tip_amplitude_over_diameter": rmse,
        "maximum_absolute_error_tip_amplitude_over_diameter": float(np.max(np.abs(errors))),
        "acceptance_tolerance": tolerance,
    }


def ambient_v2_equation_reproduction_gate(
    result_rows: Sequence[Mapping[str, Any]],
    gate_spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the frozen closure against coarse/fine modal estimators."""

    stationary_protocol.validate_response_semantics(
        {
            "response_model_semantics": (
                stationary_protocol.ambient_v2_response_semantics()
            ),
            "equation_reproduction_gate": dict(gate_spec),
        }
    )
    table = gate_spec.get("analytical_table")
    if not isinstance(table, list) or not table:
        raise ValueError("Ambient-v2 equation gate has no analytical table.")
    expected_u = [float(row["reduced_velocity"]) for row in table]
    analytical_by_u = {
        float(row["reduced_velocity"]): float(
            row["analytical_equilibrium_amplitude_over_diameter"]
        )
        for row in table
    }
    grouped: dict[float, dict[str, Mapping[str, Any]]] = {}
    for row in result_rows:
        if float(row.get("scruton_number", math.nan)) != CALIBRATION_SC:
            continue
        grouped.setdefault(float(row["reduced_velocity"]), {})[
            str(row["timestep_level"])
        ] = row
    complete = (
        sorted(grouped) == expected_u
        and all(set(grouped[u_star]) == {"dt", "dt_half"} for u_star in expected_u)
    )
    if not complete:
        return [], {
            "schema": AMBIENT_V2_EQUATION_GATE_SCHEMA,
            "evaluated": False,
            "pass": False,
            "reason": "calibration estimator grid is incomplete",
            "expected_reduced_velocities": expected_u,
            "observed_reduced_velocities": sorted(grouped),
        }

    comparison_rows: list[dict[str, Any]] = []
    try:
        for u_star in expected_u:
            coarse = float(
                grouped[u_star]["dt"][
                    "response_estimator_final_mean_amplitude_ratio"
                ]
            )
            fine = float(
                grouped[u_star]["dt_half"][
                    "response_estimator_final_mean_amplitude_ratio"
                ]
            )
            analytical = analytical_by_u[u_star]
            comparison_rows.append(
                {
                    "reduced_velocity": u_star,
                    "analytical_equilibrium_amplitude_over_diameter": analytical,
                    "coarse_response_estimator_amplitude_over_diameter": coarse,
                    "fine_response_estimator_amplitude_over_diameter": fine,
                    "fine_minus_analytical": fine - analytical,
                    "fine_absolute_error_over_diameter": abs(fine - analytical),
                    "coarse_fine_absolute_difference_over_diameter": abs(
                        fine - coarse
                    ),
                    "fine_active_at_threshold": fine
                    >= float(
                        gate_spec[
                            "active_threshold_amplitude_over_diameter"
                        ]
                    ),
                }
            )
    except (KeyError, TypeError, ValueError) as exc:
        return [], {
            "schema": AMBIENT_V2_EQUATION_GATE_SCHEMA,
            "evaluated": False,
            "pass": False,
            "reason": f"calibration estimator is missing or nonnumeric: {exc}",
            "expected_reduced_velocities": expected_u,
            "observed_reduced_velocities": sorted(grouped),
        }

    fine_errors = np.asarray(
        [row["fine_minus_analytical"] for row in comparison_rows], dtype=float
    )
    coarse_fine = np.asarray(
        [
            row["coarse_fine_absolute_difference_over_diameter"]
            for row in comparison_rows
        ],
        dtype=float,
    )
    fine_rmse = float(np.sqrt(np.mean(fine_errors**2)))
    fine_max_abs = float(np.max(np.abs(fine_errors)))
    coarse_fine_max_abs = float(np.max(coarse_fine))
    active_grid = [
        float(row["reduced_velocity"])
        for row in comparison_rows
        if bool(row["fine_active_at_threshold"])
    ]
    expected_active = [
        float(value)
        for value in gate_spec["expected_active_reduced_velocities"]
    ]
    rmse_tolerance = float(
        gate_spec["fine_estimator_rmse_tolerance_over_diameter"]
    )
    max_abs_tolerance = float(
        gate_spec[
            "fine_estimator_maximum_absolute_error_tolerance_over_diameter"
        ]
    )
    coarse_fine_tolerance = float(
        gate_spec[
            "coarse_fine_estimator_maximum_absolute_difference_tolerance_over_diameter"
        ]
    )
    active_grid_pass = active_grid == expected_active
    passed = (
        fine_rmse <= rmse_tolerance
        and fine_max_abs <= max_abs_tolerance
        and active_grid_pass
        and coarse_fine_max_abs <= coarse_fine_tolerance
    )
    return comparison_rows, {
        "schema": AMBIENT_V2_EQUATION_GATE_SCHEMA,
        "evaluated": True,
        "pass": passed,
        "calibration_qblade_implied_scruton_number": float(
            gate_spec["calibration_qblade_implied_scruton_number"]
        ),
        "fine_estimator_rmse_over_diameter": fine_rmse,
        "fine_estimator_rmse_tolerance_over_diameter": rmse_tolerance,
        "fine_estimator_maximum_absolute_error_over_diameter": fine_max_abs,
        "fine_estimator_maximum_absolute_error_tolerance_over_diameter": (
            max_abs_tolerance
        ),
        "coarse_fine_estimator_maximum_absolute_difference_over_diameter": (
            coarse_fine_max_abs
        ),
        "coarse_fine_estimator_maximum_absolute_difference_tolerance_over_diameter": (
            coarse_fine_tolerance
        ),
        "active_threshold_amplitude_over_diameter": float(
            gate_spec["active_threshold_amplitude_over_diameter"]
        ),
        "observed_active_reduced_velocities": active_grid,
        "expected_active_reduced_velocities": expected_active,
        "active_grid_exact_match_pass": active_grid_pass,
        "point_count": len(comparison_rows),
        "scope": (
            "response-estimator reproduction of the frozen modal energy-balance "
            "equation; separate from tip-channel and external-code validation"
        ),
    }


def lupi_experimental_curve_screen(
    curve_rows: Sequence[Mapping[str, Any]],
    figure9_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Screen the complete fine Sc=5.18 curve against Lupi markers.

    The experiment is not a calibration target.  Response values are linearly
    interpolated only to experimental markers inside the frozen 12-point
    response grid.  Curve RMSE, peak amplitude, and peak location retain the
    predeclared external-comparison tolerance concepts as separate components.
    """

    figure9 = Path(figure9_path).expanduser().resolve()
    predicted = sorted(
        (
            row
            for row in curve_rows
            if float(row.get("scruton_number", math.nan)) == CALIBRATION_SC
        ),
        key=lambda row: float(row["reduced_velocity"]),
    )
    predicted_u = np.asarray(
        [float(row["reduced_velocity"]) for row in predicted], dtype=float
    )
    complete_fine_curve = (
        predicted_u.tolist() == list(DEFAULT_REDUCED_VELOCITIES)
        and all(row.get("selected_timestep_level") == "dt_half" for row in predicted)
    )
    base_summary = {
        "schema": LUPI_EXPERIMENT_SCREEN_SCHEMA,
        "scope": (
            "experimental validation screen only; not a calibration input, "
            "analytical-reproduction requirement, or fatigue-conservatism claim"
        ),
        "series": "experiment_lupi2021",
        "figure9_path": figure9.as_posix(),
        "figure9_sha256": sha256_file(figure9),
        "complete_12_point_fine_response_curve": complete_fine_curve,
        "curve_rmse_tolerance": LUPI_EXPERIMENT_CURVE_RMSE_TOLERANCE,
        "peak_location_tolerance_u_star": (
            LUPI_EXPERIMENT_PEAK_LOCATION_TOLERANCE_USTAR
        ),
        "peak_amplitude_tolerance_definition": (
            "max(0.03D, 15% of digitized experimental peak amplitude)"
        ),
    }

    def unevaluated_summary(
        reason: str,
        *,
        experimental_marker_count: int | None = None,
        in_range_marker_count: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        out_of_range_marker_count = (
            None
            if experimental_marker_count is None or in_range_marker_count is None
            else experimental_marker_count - in_range_marker_count
        )
        return [], {
            **base_summary,
            "evaluated": False,
            "pass": False,
            "reason": reason,
            "experimental_marker_count": experimental_marker_count,
            "in_range_marker_count": in_range_marker_count,
            "out_of_range_marker_count": out_of_range_marker_count,
            "curve_rmse_tip_amplitude_over_diameter": None,
            "curve_maximum_absolute_error_tip_amplitude_over_diameter": None,
            "curve_rmse_pass": False,
            "response_peak_u_star": None,
            "experimental_peak_u_star": None,
            "peak_location_absolute_error_u_star": None,
            "peak_location_pass": False,
            "response_peak_amplitude_over_diameter": None,
            "experimental_peak_amplitude_over_diameter": None,
            "peak_amplitude_absolute_error": None,
            "peak_amplitude_tolerance": None,
            "peak_amplitude_pass": False,
        }

    if not complete_fine_curve:
        return unevaluated_summary(
            "the exact 12-point Sc=5.18 half-dt response curve is required"
        )

    predicted_amplitude = np.asarray(
        [float(row["tip_peak_amplitude_over_diameter"]) for row in predicted],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(predicted_u))
        or np.any(~np.isfinite(predicted_amplitude))
        or np.any(np.diff(predicted_u) <= 0.0)
        or np.any(predicted_amplitude < 0.0)
    ):
        return unevaluated_summary(
            "the complete response curve contains invalid values"
        )

    experimental_u, experimental_amplitude = _read_figure9_series(
        figure9, "experiment_lupi2021"
    )
    in_range = (experimental_u >= predicted_u[0]) & (
        experimental_u <= predicted_u[-1]
    )
    retained_u = experimental_u[in_range]
    retained_amplitude = experimental_amplitude[in_range]
    if retained_u.size < 3:
        return unevaluated_summary(
            "fewer than three experimental markers lie inside the response grid",
            experimental_marker_count=int(experimental_u.size),
            in_range_marker_count=int(retained_u.size),
        )

    interpolated = np.interp(retained_u, predicted_u, predicted_amplitude)
    errors = interpolated - retained_amplitude
    curve_rmse = float(np.sqrt(np.mean(errors**2)))
    curve_maximum_error = float(np.max(np.abs(errors)))
    curve_rmse_pass = curve_rmse <= LUPI_EXPERIMENT_CURVE_RMSE_TOLERANCE

    response_peak_index = int(np.argmax(predicted_amplitude))
    experimental_peak_index = int(np.argmax(experimental_amplitude))
    response_peak_u = float(predicted_u[response_peak_index])
    response_peak_amplitude = float(predicted_amplitude[response_peak_index])
    experimental_peak_u = float(experimental_u[experimental_peak_index])
    experimental_peak_amplitude = float(
        experimental_amplitude[experimental_peak_index]
    )
    peak_location_error = abs(response_peak_u - experimental_peak_u)
    peak_location_pass = (
        peak_location_error <= LUPI_EXPERIMENT_PEAK_LOCATION_TOLERANCE_USTAR
    )
    peak_amplitude_error = abs(response_peak_amplitude - experimental_peak_amplitude)
    peak_amplitude_tolerance = max(
        LUPI_EXPERIMENT_PEAK_ABSOLUTE_TOLERANCE,
        LUPI_EXPERIMENT_PEAK_RELATIVE_TOLERANCE * experimental_peak_amplitude,
    )
    peak_amplitude_pass = peak_amplitude_error <= peak_amplitude_tolerance
    rows = [
        {
            "scruton_number": CALIBRATION_SC,
            "experimental_reduced_velocity": float(u_star),
            "qblade_interpolated_peak_over_diameter": float(prediction),
            "lupi_experimental_peak_over_diameter": float(reference),
            "signed_error": float(error),
        }
        for u_star, prediction, reference, error in zip(
            retained_u, interpolated, retained_amplitude, errors
        )
    ]
    passed = curve_rmse_pass and peak_location_pass and peak_amplitude_pass
    return rows, {
        **base_summary,
        "evaluated": True,
        "pass": passed,
        "reason": None,
        "experimental_marker_count": int(experimental_u.size),
        "in_range_marker_count": int(retained_u.size),
        "out_of_range_marker_count": int(experimental_u.size - retained_u.size),
        "curve_rmse_tip_amplitude_over_diameter": curve_rmse,
        "curve_maximum_absolute_error_tip_amplitude_over_diameter": (
            curve_maximum_error
        ),
        "curve_rmse_pass": curve_rmse_pass,
        "response_peak_u_star": response_peak_u,
        "experimental_peak_u_star": experimental_peak_u,
        "peak_location_absolute_error_u_star": peak_location_error,
        "peak_location_pass": peak_location_pass,
        "response_peak_amplitude_over_diameter": response_peak_amplitude,
        "experimental_peak_amplitude_over_diameter": experimental_peak_amplitude,
        "peak_amplitude_absolute_error": peak_amplitude_error,
        "peak_amplitude_tolerance": peak_amplitude_tolerance,
        "peak_amplitude_pass": peak_amplitude_pass,
    }


def calibration_stage_assessment(
    *,
    manifest: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
    runtime_audits: Sequence[Mapping[str, Any]],
    dt_rows: Sequence[Mapping[str, Any]],
    dt_curve_rows: Sequence[Mapping[str, Any]],
    mechanism_rows: Sequence[Mapping[str, Any]],
    live_analytical_summary: Mapping[str, Any],
    calibration_curve_summary: Mapping[str, Any],
    calibration_band_summary: Mapping[str, Any],
    peak_rows: Sequence[Mapping[str, Any]],
    equation_reproduction_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a fail-closed assessment of the 24-case calibration stage.

    This gate intentionally ignores missing held-out results.  It does not
    replace the all-structure assessment or qualification gates: it answers
    only whether the complete stationary Sc=5.18 nominal/half-step campaign
    reproduces its frozen analytical closure and passes the already declared
    external-curve, mechanism, runtime, and numerical checks.
    """

    expected_pairs = {
        (float(u_star), timestep_level)
        for u_star in DEFAULT_REDUCED_VELOCITIES
        for timestep_level in ("dt", "dt_half")
    }
    calibration_entries = [
        entry
        for entry in manifest.get("cases", [])
        if float(entry.get("scruton_number", math.nan)) == CALIBRATION_SC
        and entry.get("role") == "calibration"
    ]
    expected_case_ids = [str(entry.get("case_id", "")) for entry in calibration_entries]
    manifest_pairs = {
        (float(entry.get("reduced_velocity", math.nan)), str(entry.get("timestep_level", "")))
        for entry in calibration_entries
    }
    input_matrix_eligible = (
        bool(manifest.get("numerics", {}).get("qualification_input_eligible", False))
        and len(calibration_entries) == len(expected_pairs)
        and len(expected_case_ids) == len(set(expected_case_ids))
        and all(expected_case_ids)
        and manifest_pairs == expected_pairs
    )
    expected_case_id_set = set(expected_case_ids)

    calibration_results = [
        row
        for row in result_rows
        if float(row.get("scruton_number", math.nan)) == CALIBRATION_SC
        and row.get("role") == "calibration"
    ]
    completed_case_ids = [str(row.get("case_id", "")) for row in calibration_results]
    results_complete = (
        input_matrix_eligible
        and len(calibration_results) == len(expected_pairs)
        and len(completed_case_ids) == len(set(completed_case_ids))
        and set(completed_case_ids) == expected_case_id_set
    )

    calibration_runtime = [
        row
        for row in runtime_audits
        if str(row.get("case_id", "")) in expected_case_id_set
    ]
    runtime_case_ids = [str(row.get("case_id", "")) for row in calibration_runtime]
    runtime_artifacts_evaluated = (
        input_matrix_eligible
        and len(calibration_runtime) == len(expected_pairs)
        and len(runtime_case_ids) == len(set(runtime_case_ids))
        and set(runtime_case_ids) == expected_case_id_set
        and all(bool(row.get("evaluated")) for row in calibration_runtime)
        and all(bool(row.get("provenance_evaluated")) for row in calibration_runtime)
    )
    runtime_provenance_pass = runtime_artifacts_evaluated and all(
        bool(row.get("provenance_pass"))
        and not bool(row.get("identity_mismatch"))
        for row in calibration_runtime
    )
    runtime_execution_pass = runtime_artifacts_evaluated and all(
        bool(row.get("execution_pass"))
        and bool(row.get("source_coverage_complete"))
        and bool(row.get("history", {}).get("runtime_model_pass"))
        and bool(row.get("history", {}).get("compact_lag_provenance_pass"))
        and bool(row.get("history", {}).get("applied_power_sign_convention_pass"))
        for row in calibration_runtime
    )
    runtime_limits_pass = runtime_artifacts_evaluated and all(
        bool(row.get("source_limits_pass"))
        and bool(row.get("response_summary_limits_pass"))
        and bool(row.get("history", {}).get("limit_headroom_pass"))
        for row in calibration_runtime
    )
    startup_seed_evaluated = runtime_artifacts_evaluated and all(
        bool(row.get("history", {}).get("startup_seed_integrity_evaluated"))
        and bool((row.get("startup_seed_response") or {}).get("evaluated"))
        for row in calibration_runtime
    )
    startup_seed_pass = startup_seed_evaluated and all(
        bool(row.get("startup_seed_summary_pass"))
        and bool(row.get("history", {}).get("startup_seed_integrity_pass"))
        and bool((row.get("startup_seed_response") or {}).get("pass"))
        for row in calibration_runtime
    )
    stationarity_evaluated = results_complete and runtime_artifacts_evaluated and all(
        bool(row.get("stationarity_evaluated")) for row in calibration_results
    ) and all(
        bool(row.get("history", {}).get("response_estimator_stationarity_evaluated"))
        for row in calibration_runtime
    )
    stationarity_pass = stationarity_evaluated and all(
        bool(row.get("stationarity_pass")) for row in calibration_results
    ) and all(
        bool(row.get("history", {}).get("response_estimator_stationarity_pass"))
        for row in calibration_runtime
    )
    runtime_case_audits_pass = runtime_artifacts_evaluated and all(
        bool(row.get("pass")) for row in calibration_runtime
    )

    calibration_dt_rows = sorted(
        [row for row in dt_rows if float(row.get("scruton_number", math.nan)) == CALIBRATION_SC],
        key=lambda row: float(row.get("reduced_velocity", math.nan)),
    )
    timestep_points_evaluated = (
        len(calibration_dt_rows) == len(DEFAULT_REDUCED_VELOCITIES)
        and [float(row.get("reduced_velocity", math.nan)) for row in calibration_dt_rows]
        == list(DEFAULT_REDUCED_VELOCITIES)
        and all(bool(row.get("frequency_evaluated")) for row in calibration_dt_rows)
    )
    timestep_amplitude_pass = timestep_points_evaluated and all(
        bool(row.get("pass")) for row in calibration_dt_rows
    )
    timestep_frequency_pass = timestep_points_evaluated and all(
        bool(row.get("frequency_pass")) for row in calibration_dt_rows
    )
    timestep_pair_runtime_stationarity_pass = timestep_points_evaluated and all(
        bool(row.get("both_stationary"))
        and bool(row.get("both_runtime_audits_pass"))
        for row in calibration_dt_rows
    )
    calibration_dt_curve = [
        row
        for row in dt_curve_rows
        if float(row.get("scruton_number", math.nan)) == CALIBRATION_SC
    ]
    timestep_curve_evaluated = (
        len(calibration_dt_curve) == 1
        and bool(calibration_dt_curve[0].get("evaluated"))
    )
    timestep_curve_pass = timestep_curve_evaluated and bool(
        calibration_dt_curve[0].get("pass")
    )

    calibration_mechanism = [
        row
        for row in mechanism_rows
        if float(row.get("scruton_number", math.nan)) == CALIBRATION_SC
    ]
    lock_in_mechanism_evaluated = (
        len(calibration_mechanism) == 1
        and bool(calibration_mechanism[0].get("evaluated"))
    )
    lock_in_mechanism_pass = lock_in_mechanism_evaluated and bool(
        calibration_mechanism[0].get("pass")
    )

    calibration_peaks = [
        row
        for row in peak_rows
        if float(row.get("scruton_number", math.nan)) == CALIBRATION_SC
        and row.get("role") == "calibration"
    ]
    external_peak_evaluated = len(calibration_peaks) == 1 and bool(
        calibration_peaks[0].get("enhanced_curve_complete")
    )
    external_peak_pass = external_peak_evaluated and bool(
        calibration_peaks[0].get("external_code_comparison_pass")
    )
    live_analytical_evaluated = bool(live_analytical_summary.get("evaluated"))
    coordinate_consistency_evaluated = bool(
        live_analytical_summary.get("coordinate_consistency_evaluated")
    )
    external_curve_evaluated = bool(calibration_curve_summary.get("evaluated"))
    external_band_evaluated = bool(calibration_band_summary.get("evaluated"))
    equation_reproduction_required = isinstance(
        manifest.get("equation_reproduction_gate"), Mapping
    )
    equation_reproduction_evaluated = bool(
        equation_reproduction_summary
        and equation_reproduction_summary.get("evaluated")
    )
    equation_reproduction_pass = bool(
        equation_reproduction_summary
        and equation_reproduction_summary.get("pass")
    )

    fully_evaluated = (
        input_matrix_eligible
        and results_complete
        and runtime_artifacts_evaluated
        and startup_seed_evaluated
        and stationarity_evaluated
        and timestep_points_evaluated
        and timestep_curve_evaluated
        and live_analytical_evaluated
        and coordinate_consistency_evaluated
        and lock_in_mechanism_evaluated
        and external_peak_evaluated
        and external_curve_evaluated
        and external_band_evaluated
        and (
            not equation_reproduction_required
            or equation_reproduction_evaluated
        )
    )
    checks = {
        "input_matrix_eligible": input_matrix_eligible,
        "all_calibration_results_complete": results_complete,
        "runtime_artifacts_evaluated": runtime_artifacts_evaluated,
        "runtime_provenance_pass": runtime_provenance_pass,
        "runtime_execution_and_model_pass": runtime_execution_pass,
        "runtime_limits_pass": runtime_limits_pass,
        "startup_seed_evaluated": startup_seed_evaluated,
        "startup_seed_pass": startup_seed_pass,
        "result_and_estimator_stationarity_evaluated": stationarity_evaluated,
        "result_and_estimator_stationarity_pass": stationarity_pass,
        "runtime_case_audits_pass": runtime_case_audits_pass,
        "timestep_point_checks_evaluated": timestep_points_evaluated,
        "timestep_amplitude_pass": timestep_amplitude_pass,
        "timestep_frequency_pass": timestep_frequency_pass,
        "timestep_pair_runtime_stationarity_pass": timestep_pair_runtime_stationarity_pass,
        "timestep_curve_properties_evaluated": timestep_curve_evaluated,
        "timestep_curve_properties_pass": timestep_curve_pass,
        "live_vs_analytical_evaluated": live_analytical_evaluated,
        "live_vs_analytical_pass": bool(live_analytical_summary.get("pass")),
        "coordinate_consistency_evaluated": coordinate_consistency_evaluated,
        "coordinate_consistency_pass": bool(
            live_analytical_summary.get("coordinate_consistency_pass")
        ),
        "lock_in_mechanism_evaluated": lock_in_mechanism_evaluated,
        "lock_in_mechanism_pass": lock_in_mechanism_pass,
        "external_hgast_peak_evaluated": external_peak_evaluated,
        "external_hgast_peak_pass": external_peak_pass,
        "external_hgast_curve_evaluated": external_curve_evaluated,
        "external_hgast_curve_pass": bool(calibration_curve_summary.get("pass")),
        "external_hgast_band_evaluated": external_band_evaluated,
        "external_hgast_band_pass": bool(calibration_band_summary.get("pass")),
    }
    if equation_reproduction_required:
        checks.update(
            {
                "ambient_v2_equation_reproduction_evaluated": (
                    equation_reproduction_evaluated
                ),
                "ambient_v2_equation_reproduction_pass": (
                    equation_reproduction_pass
                ),
            }
        )
    pass_fields = (
        "runtime_provenance_pass",
        "runtime_execution_and_model_pass",
        "runtime_limits_pass",
        "startup_seed_pass",
        "result_and_estimator_stationarity_pass",
        "runtime_case_audits_pass",
        "timestep_amplitude_pass",
        "timestep_frequency_pass",
        "timestep_pair_runtime_stationarity_pass",
        "timestep_curve_properties_pass",
        "live_vs_analytical_pass",
        "coordinate_consistency_pass",
        "lock_in_mechanism_pass",
        "external_hgast_peak_pass",
        "external_hgast_curve_pass",
        "external_hgast_band_pass",
    )
    if equation_reproduction_required:
        pass_fields += ("ambient_v2_equation_reproduction_pass",)
    passed = fully_evaluated and all(bool(checks[field]) for field in pass_fields)
    return {
        "schema": CALIBRATION_STAGE_SCHEMA,
        "status": (
            "calibration_stage_pass"
            if passed
            else ("calibration_stage_fail" if fully_evaluated else "not_evaluated")
        ),
        "evaluated": fully_evaluated,
        "pass": passed,
        "scope": (
            "stationary Sc=5.18 live-QBlade calibration-stage assessment; independent "
            "of held-out result availability and separate from frozen-source improvement"
        ),
        "expected_scruton_number": CALIBRATION_SC,
        "expected_case_count": len(expected_pairs),
        "expected_reduced_velocities": list(DEFAULT_REDUCED_VELOCITIES),
        "expected_timestep_levels": ["dt", "dt_half"],
        "manifest_calibration_case_count": len(calibration_entries),
        "completed_calibration_case_count": len(calibration_results),
        "runtime_calibration_case_count": len(calibration_runtime),
        "checks": checks,
    }


def analyze_campaign(
    manifest_path: Path,
    results_root: Path,
    output: Path,
    *,
    response_window_cycles: float = 40.0,
) -> Path:
    results_root = Path(results_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(
                f"Analysis output exists and is not a directory: {output}"
            )
        if any(output.iterdir()):
            raise FileExistsError(
                f"Refusing nonempty analysis output directory {output}; use a new or empty directory."
            )
    campaign_root, manifest = _load_campaign(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    runtime_audits: list[dict[str, Any]] = []
    missing: list[str] = []
    result_errors: list[dict[str, str]] = []
    for entry in manifest["cases"]:
        case_directory = results_root / entry["case_id"]
        path = case_directory / "result.txt"
        if not path.is_file():
            missing.append(entry["case_id"])
            continue
        runtime = runtime_case_audit(
            case_directory,
            entry=entry,
            campaign_root=campaign_root,
            manifest_files=manifest["_verified_manifest_files"],
            result_path=path,
            expected_history_mode=manifest.get("numerics", {}).get(
                "required_sil_vorlap_history_mode"
            ),
            runtime_contract=(
                manifest.get("runtime_contract")
                if int(
                    entry.get("response_model_config", {}).get(
                        "schema_version", 1
                    )
                )
                == 2
                else None
            ),
        )
        runtime_audits.append({"case_id": entry["case_id"], **runtime})
        if bool(runtime.get("artifacts_present")) and (
            not bool(runtime.get("evaluated"))
            or bool(runtime.get("identity_mismatch"))
        ):
            result_errors.append(
                {
                    "case_id": entry["case_id"],
                    "error": "runtime artifact identity rejection: "
                    + "; ".join(str(value) for value in runtime.get("problems", [])),
                }
            )
            continue
        try:
            response = response_amplitude_from_result(
                path,
                natural_frequency_hz=float(
                    entry["structure"]["natural_frequency_hz"]
                ),
                response_window_cycles=response_window_cycles,
                expected_timestep_s=float(entry["timestep_s"]),
                expected_samples=int(entry["num_steps"]),
            )
        except (KeyError, TypeError, ValueError, OSError) as exc:
            result_errors.append({"case_id": entry["case_id"], "error": str(exc)})
            continue
        history = runtime.get("history", {}) if runtime.get("evaluated") else {}
        result_rows.append(
            {
                "case_id": entry["case_id"],
                "scruton_number": entry["scruton_number"],
                "role": entry["role"],
                "reduced_velocity": entry["reduced_velocity"],
                "timestep_level": entry["timestep_level"],
                "specified_timestep_s": entry["timestep_s"],
                "steps_per_natural_cycle": entry["steps_per_natural_cycle"],
                "runtime_audit_evaluated": bool(runtime.get("evaluated")),
                "runtime_audit_pass": bool(runtime.get("pass")),
                "response_estimator_final_mean_amplitude_ratio": history.get(
                    "response_estimator_final_mean_amplitude_ratio"
                ),
                "response_estimator_stationarity_pass": history.get(
                    "response_estimator_stationarity_pass"
                ),
                "response_limit_headroom_pass": history.get(
                    "limit_headroom_pass"
                ),
                **response,
            }
        )
    _write_csv(output / "response_by_case.csv", result_rows)
    (output / "runtime_case_audits.json").write_text(
        json.dumps(runtime_audits, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    grouped: dict[tuple[float, float], dict[str, dict[str, Any]]] = {}
    for row in result_rows:
        grouped.setdefault((float(row["scruton_number"]), float(row["reduced_velocity"])), {})[
            str(row["timestep_level"])
        ] = row
    curve_rows: list[dict[str, Any]] = []
    dt_rows: list[dict[str, Any]] = []
    coefficients = json.loads(
        (campaign_root / manifest["calibration_policy"]["coefficient_file"]).read_text(
            encoding="utf-8"
        )
    )
    frozen_coefficients = coefficients["frozen_parameters"]
    structures = load_structural_cases(
        _resolved_reference_path(
            manifest["reference"]["reference_files"][
                "vlastos2026_model_parameters.csv"
            ]
        )
    )
    for (sc, u_star), levels in sorted(grouped.items()):
        primary = levels.get("dt_half", levels.get("dt"))
        if primary is None:
            continue
        analytical_amplitude = float(
            analytical_equilibrium_amplitude_ratio(
                derived_scruton_number(structures[sc]),
                [u_star],
                aerodynamic_damping_gain_h1=float(
                    frozen_coefficients["aerodynamic_damping_gain_h1"]
                ),
                saturation_amplitude_ratio=float(
                    frozen_coefficients["saturation_amplitude_ratio"]
                ),
                reduced_velocity_low=float(
                    frozen_coefficients["reduced_velocity_low"]
                ),
                reduced_velocity_peak=float(
                    frozen_coefficients["reduced_velocity_peak"]
                ),
                reduced_velocity_high=float(
                    frozen_coefficients["reduced_velocity_high"]
                ),
            )[0]
        )
        curve_rows.append(
            {
                "scruton_number": sc,
                "role": primary["role"],
                "reduced_velocity": u_star,
                "tip_peak_amplitude_over_diameter": primary["tip_peak_amplitude_over_diameter"],
                "selected_timestep_level": primary["timestep_level"],
                "analytical_equilibrium_amplitude_over_diameter": analytical_amplitude,
                "terminal_dominant_frequency_over_fn": primary[
                    "terminal_dominant_frequency_over_fn"
                ],
                "terminal_mode_band_power_fraction": primary[
                    "terminal_mode_band_power_fraction"
                ],
                "stationarity_pass": primary["stationarity_pass"],
                "runtime_audit_evaluated": primary["runtime_audit_evaluated"],
                "runtime_audit_pass": primary["runtime_audit_pass"],
                "response_estimator_final_mean_amplitude_ratio": primary[
                    "response_estimator_final_mean_amplitude_ratio"
                ],
            }
        )
        if "dt" in levels and "dt_half" in levels:
            coarse = float(levels["dt"]["tip_peak_amplitude_over_diameter"])
            fine = float(levels["dt_half"]["tip_peak_amplitude_over_diameter"])
            absolute = abs(fine - coarse)
            tolerance = max(0.01, 0.05 * abs(fine))
            coarse_frequency = levels["dt"]["terminal_dominant_frequency_over_fn"]
            fine_frequency = levels["dt_half"][
                "terminal_dominant_frequency_over_fn"
            ]
            frequency_evaluated = (
                coarse_frequency is not None and fine_frequency is not None
            )
            frequency_difference = (
                None
                if not frequency_evaluated
                else abs(float(fine_frequency) - float(coarse_frequency))
            )
            frequency_pass = (
                frequency_evaluated
                and float(frequency_difference)
                <= TIMESTEP_FREQUENCY_TOLERANCE_RATIO
            )
            dt_rows.append(
                {
                    "scruton_number": sc,
                    "reduced_velocity": u_star,
                    "coarse_peak_over_diameter": coarse,
                    "half_dt_peak_over_diameter": fine,
                    "absolute_difference": absolute,
                    "acceptance_tolerance": tolerance,
                    "pass": absolute <= tolerance,
                    "coarse_dominant_frequency_over_fn": coarse_frequency,
                    "half_dt_dominant_frequency_over_fn": fine_frequency,
                    "frequency_difference_over_fn": frequency_difference,
                    "frequency_tolerance_over_fn": TIMESTEP_FREQUENCY_TOLERANCE_RATIO,
                    "frequency_evaluated": frequency_evaluated,
                    "frequency_pass": frequency_pass,
                    "both_stationary": bool(levels["dt"]["stationarity_pass"])
                    and bool(levels["dt_half"]["stationarity_pass"]),
                    "both_runtime_audits_pass": bool(
                        levels["dt"]["runtime_audit_pass"]
                    )
                    and bool(levels["dt_half"]["runtime_audit_pass"]),
                }
            )
    _write_csv(output / "response_peak_vs_reduced_velocity.csv", curve_rows)
    _write_csv(output / "dt_halving_comparison.csv", dt_rows)
    equation_gate_spec = manifest.get("equation_reproduction_gate")
    equation_gate_required = isinstance(equation_gate_spec, Mapping)
    if equation_gate_required:
        equation_gate_rows, equation_gate_summary = (
            ambient_v2_equation_reproduction_gate(
                result_rows,
                equation_gate_spec,
            )
        )
        _write_csv(
            output / "ambient_v2_equation_reproduction.csv",
            equation_gate_rows,
        )
    else:
        equation_gate_rows = []
        equation_gate_summary = None

    figure10 = _resolved_reference_path(
        manifest["reference"]["reference_files"][
            "all_sc_figure10_peak_digitized.csv"
        ]
    )
    figure9 = _resolved_reference_path(
        manifest["reference"]["reference_files"][
            "sc5p18_figure9_digitized.csv"
        ]
    )
    model_parameters = _resolved_reference_path(
        manifest["reference"]["reference_files"][
            "vlastos2026_model_parameters.csv"
        ]
    )
    peaks = _load_figure10_peaks(figure10)
    calibration_hgast_peak = digitized_series_peak(figure9, CALIBRATION_SERIES)
    calibration_lupi_peak = digitized_series_peak(figure9, "experiment_lupi2021")
    expected_velocities_by_sc = {
        sc: sorted(
            {
                float(entry["reduced_velocity"])
                for entry in manifest["cases"]
                if float(entry["scruton_number"]) == sc
            }
        )
        for sc in PUBLISHED_STRUCTURAL_CASES
        if any(float(entry["scruton_number"]) == sc for entry in manifest["cases"])
    }
    qualification_velocities = list(DEFAULT_REDUCED_VELOCITIES)
    calibration_curve_rows, calibration_curve_summary = calibration_curve_comparison(
        curve_rows,
        figure9,
    )
    _write_csv(output / "calibration_curve_comparison.csv", calibration_curve_rows)
    peak_rows = []
    for sc in sorted({float(row["scruton_number"]) for row in curve_rows}):
        candidates = sorted(
            [row for row in curve_rows if float(row["scruton_number"]) == sc],
            key=lambda row: float(row["reduced_velocity"]),
        )
        predicted = max(candidates, key=lambda row: float(row["tip_peak_amplitude_over_diameter"]))
        hgast_value, hgast_uncertainty = peaks[(sc, "hgast_hartlen_currie")]
        experiment_value, experiment_uncertainty = peaks[(sc, "experiment_lupi2021")]
        code_tolerance = max(0.03, 0.15 * hgast_value)
        candidate_u = [float(row["reduced_velocity"]) for row in candidates]
        candidate_amplitude = [
            float(row["tip_peak_amplitude_over_diameter"]) for row in candidates
        ]
        enhanced_curve_complete = (
            candidate_u == qualification_velocities
            and expected_velocities_by_sc.get(sc) == qualification_velocities
        )
        if sc == CALIBRATION_SC:
            reference_peak_u_star = calibration_hgast_peak["reduced_velocity"]
            experiment_peak_u_star = calibration_lupi_peak["reduced_velocity"]
            peak_u_star_source = "sampled maximum of digitized Figure 9 series"
        else:
            # Full Figure 9 held-out curves are not provenance-bound locally;
            # retain their existing model-parameter peak-velocity metadata.
            reference_peak_u_star = structures[sc].u_star_max_figure_digitized
            experiment_peak_u_star = reference_peak_u_star
            peak_u_star_source = "existing held-out model-parameter metadata"
        colocated_prediction = _interpolated_value(
            candidate_u,
            candidate_amplitude,
            reference_peak_u_star,
        )
        colocated_error = (
            None
            if colocated_prediction is None
            else abs(colocated_prediction - hgast_value)
        )
        peak_location_error = abs(
            float(predicted["reduced_velocity"]) - reference_peak_u_star
        )
        if colocated_prediction is None or hgast_value <= 0.0:
            fatigue_proxy_fraction = None
            fatigue_proxy_correction = None
            fatigue_proxy_correction_unbounded = False
        elif colocated_prediction <= 0.0:
            fatigue_proxy_fraction = 0.0
            fatigue_proxy_correction = None
            fatigue_proxy_correction_unbounded = True
        else:
            fatigue_proxy_fraction = (
                colocated_prediction / hgast_value
            ) ** FATIGUE_PROXY_SLOPE
            fatigue_proxy_correction = (
                hgast_value / colocated_prediction
            ) ** FATIGUE_PROXY_SLOPE
            fatigue_proxy_correction_unbounded = False
        peak_rows.append(
            {
                "scruton_number": sc,
                "role": "calibration" if sc == CALIBRATION_SC else "held_out",
                "predicted_peak_over_diameter": float(predicted["tip_peak_amplitude_over_diameter"]),
                "predicted_peak_u_star": float(predicted["reduced_velocity"]),
                "hgast_reference_peak_over_diameter": hgast_value,
                "hgast_digitization_uncertainty": hgast_uncertainty,
                "external_code_absolute_error": abs(float(predicted["tip_peak_amplitude_over_diameter"]) - hgast_value),
                "external_code_comparison_tolerance": code_tolerance,
                "external_code_comparison_pass": abs(float(predicted["tip_peak_amplitude_over_diameter"]) - hgast_value) <= code_tolerance,
                "experiment_reference_peak_over_diameter": experiment_value,
                "experiment_digitization_uncertainty": experiment_uncertainty,
                "experiment_absolute_error_screen_only": abs(float(predicted["tip_peak_amplitude_over_diameter"]) - experiment_value),
                "reference_peak_u_star": reference_peak_u_star,
                "hgast_reference_peak_u_star": reference_peak_u_star,
                "experiment_reference_peak_u_star": experiment_peak_u_star,
                "reference_peak_u_star_source": peak_u_star_source,
                "experiment_peak_location_absolute_error_u_star_screen_only": abs(
                    float(predicted["reduced_velocity"])
                    - experiment_peak_u_star
                ),
                "enhanced_curve_complete": enhanced_curve_complete,
                "peak_location_evaluated": enhanced_curve_complete,
                "peak_location_absolute_error_u_star": peak_location_error,
                "peak_location_tolerance_u_star": LOCK_IN_REFERENCE_PEAK_NEIGHBORHOOD_USTAR,
                "peak_location_pass": enhanced_curve_complete
                and peak_location_error <= LOCK_IN_REFERENCE_PEAK_NEIGHBORHOOD_USTAR,
                "predicted_at_reference_peak_u_star_over_diameter": colocated_prediction,
                "colocated_external_code_absolute_error": colocated_error,
                "colocated_external_code_comparison_tolerance": code_tolerance,
                "colocated_external_code_comparison_pass": colocated_error
                is not None
                and colocated_error <= code_tolerance,
                "material_skill_over_no_viv_at_reference_peak": (
                    None
                    if colocated_error is None or hgast_value == 0.0
                    else 1.0 - colocated_error / hgast_value
                ),
                "displacement_amplitude_pseudo_damage_slope": FATIGUE_PROXY_SLOPE,
                "response_to_reference_pseudo_damage_fraction_screen_only": fatigue_proxy_fraction,
                "required_pseudo_damage_correction_factor_screen_only": fatigue_proxy_correction,
                "required_pseudo_damage_correction_unbounded": fatigue_proxy_correction_unbounded,
            }
        )
    _write_csv(output / "peak_acceptance_by_scruton.csv", peak_rows)

    stationarity_complete = len(result_rows) == len(manifest["cases"]) and all(
        bool(row["stationarity_pass"]) for row in result_rows
    )
    runtime_complete = (
        len(runtime_audits) == len(manifest["cases"])
        and all(
            bool(row.get("evaluated"))
            and bool(row.get("provenance_evaluated"))
            for row in runtime_audits
        )
    )
    runtime_pass = runtime_complete and all(
        bool(row.get("pass")) for row in runtime_audits
    )

    dt_curve_rows: list[dict[str, Any]] = []
    for sc, expected_velocities in expected_velocities_by_sc.items():
        curves_by_level: dict[str, list[dict[str, Any]]] = {}
        for level in ("dt", "dt_half"):
            curves_by_level[level] = sorted(
                [
                    row
                    for row in result_rows
                    if float(row["scruton_number"]) == sc
                    and row["timestep_level"] == level
                ],
                key=lambda row: float(row["reduced_velocity"]),
            )
        complete_levels = all(
            [float(row["reduced_velocity"]) for row in rows]
            == expected_velocities
            for rows in curves_by_level.values()
        )
        hgast_peak = peaks[(sc, "hgast_hartlen_currie")][0]
        threshold = CALIBRATION_THRESHOLD_FRACTION * hgast_peak
        level_metrics: dict[str, Any] = {}
        for level, rows in curves_by_level.items():
            if not rows:
                level_metrics[level] = {
                    "peak_u_star": None,
                    "band_low_u_star": None,
                    "band_high_u_star": None,
                }
                continue
            u_values = [float(row["reduced_velocity"]) for row in rows]
            amplitudes = [
                float(row["tip_peak_amplitude_over_diameter"]) for row in rows
            ]
            low, high = _band_endpoints(u_values, amplitudes, threshold)
            peak = rows[int(np.argmax(amplitudes))]
            level_metrics[level] = {
                "peak_u_star": float(peak["reduced_velocity"]),
                "band_low_u_star": low,
                "band_high_u_star": high,
            }
        coarse_metrics = level_metrics["dt"]
        fine_metrics = level_metrics["dt_half"]
        peak_difference = (
            None
            if coarse_metrics["peak_u_star"] is None
            or fine_metrics["peak_u_star"] is None
            else abs(
                fine_metrics["peak_u_star"] - coarse_metrics["peak_u_star"]
            )
        )
        low_difference = (
            None
            if coarse_metrics["band_low_u_star"] is None
            or fine_metrics["band_low_u_star"] is None
            else abs(
                fine_metrics["band_low_u_star"]
                - coarse_metrics["band_low_u_star"]
            )
        )
        high_difference = (
            None
            if coarse_metrics["band_high_u_star"] is None
            or fine_metrics["band_high_u_star"] is None
            else abs(
                fine_metrics["band_high_u_star"]
                - coarse_metrics["band_high_u_star"]
            )
        )
        evaluated = (
            complete_levels
            and expected_velocities == qualification_velocities
            and peak_difference is not None
            and low_difference is not None
            and high_difference is not None
        )
        pass_value = (
            evaluated
            and peak_difference <= TIMESTEP_PEAK_LOCATION_TOLERANCE_USTAR
            and low_difference <= TIMESTEP_BAND_ENDPOINT_TOLERANCE_USTAR
            and high_difference <= TIMESTEP_BAND_ENDPOINT_TOLERANCE_USTAR
        )
        dt_curve_rows.append(
            {
                "scruton_number": sc,
                "evaluated": evaluated,
                "pass": pass_value,
                "threshold_amplitude_over_diameter": threshold,
                "coarse_peak_u_star": coarse_metrics["peak_u_star"],
                "half_dt_peak_u_star": fine_metrics["peak_u_star"],
                "peak_location_difference_u_star": peak_difference,
                "peak_location_tolerance_u_star": TIMESTEP_PEAK_LOCATION_TOLERANCE_USTAR,
                "coarse_band_low_u_star": coarse_metrics["band_low_u_star"],
                "half_dt_band_low_u_star": fine_metrics["band_low_u_star"],
                "band_low_difference_u_star": low_difference,
                "coarse_band_high_u_star": coarse_metrics["band_high_u_star"],
                "half_dt_band_high_u_star": fine_metrics["band_high_u_star"],
                "band_high_difference_u_star": high_difference,
                "band_endpoint_tolerance_u_star": TIMESTEP_BAND_ENDPOINT_TOLERANCE_USTAR,
            }
        )
    _write_csv(output / "dt_curve_qualification.csv", dt_curve_rows)

    mechanism_rows: list[dict[str, Any]] = []
    for sc, expected_velocities in expected_velocities_by_sc.items():
        rows = sorted(
            [row for row in curve_rows if float(row["scruton_number"]) == sc],
            key=lambda row: float(row["reduced_velocity"]),
        )
        reference_peak_amplitude = peaks[(sc, "hgast_hartlen_currie")][0]
        threshold = CALIBRATION_THRESHOLD_FRACTION * reference_peak_amplitude
        active = [
            float(row["tip_peak_amplitude_over_diameter"]) >= threshold
            for row in rows
        ]
        longest_run = 0
        current_run = 0
        for value in active:
            current_run = current_run + 1 if value else 0
            longest_run = max(longest_run, current_run)
        active_rows = [row for row, is_active in zip(rows, active) if is_active]
        frequency_data_complete = bool(active_rows) and all(
            row["terminal_dominant_frequency_over_fn"] is not None
            and row["terminal_mode_band_power_fraction"] is not None
            for row in active_rows
        )
        frequency_pass = frequency_data_complete and all(
            abs(float(row["terminal_dominant_frequency_over_fn"]) - 1.0)
            <= LOCK_IN_FREQUENCY_TOLERANCE_RATIO
            and float(row["terminal_mode_band_power_fraction"])
            >= LOCK_IN_MINIMUM_MODE_BAND_POWER_FRACTION
            for row in active_rows
        )
        reference_peak_u = (
            calibration_hgast_peak["reduced_velocity"]
            if sc == CALIBRATION_SC
            else structures[sc].u_star_max_figure_digitized
        )
        peak_neighborhood_present = any(
            abs(float(row["reduced_velocity"]) - reference_peak_u)
            <= LOCK_IN_REFERENCE_PEAK_NEIGHBORHOOD_USTAR
            for row in active_rows
        )
        complete_curve = (
            [float(row["reduced_velocity"]) for row in rows]
            == qualification_velocities
            and expected_velocities == qualification_velocities
        )
        evaluated = complete_curve and frequency_data_complete
        pass_value = (
            evaluated
            and frequency_pass
            and longest_run >= LOCK_IN_MINIMUM_CONSECUTIVE_POINTS
            and peak_neighborhood_present
        )
        mechanism_rows.append(
            {
                "scruton_number": sc,
                "evaluated": evaluated,
                "pass": pass_value,
                "active_threshold_amplitude_over_diameter": threshold,
                "active_point_count": len(active_rows),
                "longest_consecutive_active_point_count": longest_run,
                "minimum_consecutive_active_points": LOCK_IN_MINIMUM_CONSECUTIVE_POINTS,
                "frequency_pass": frequency_pass,
                "frequency_tolerance_over_fn": LOCK_IN_FREQUENCY_TOLERANCE_RATIO,
                "minimum_mode_band_power_fraction": LOCK_IN_MINIMUM_MODE_BAND_POWER_FRACTION,
                "reference_peak_neighborhood_present": peak_neighborhood_present,
                "reference_peak_neighborhood_tolerance_u_star": LOCK_IN_REFERENCE_PEAK_NEIGHBORHOOD_USTAR,
            }
        )
    _write_csv(output / "lock_in_mechanism_qualification.csv", mechanism_rows)

    calibration_fine = sorted(
        [
            row
            for row in result_rows
            if float(row["scruton_number"]) == CALIBRATION_SC
            and row["timestep_level"] == "dt_half"
        ],
        key=lambda row: float(row["reduced_velocity"]),
    )
    expected_calibration_velocities = expected_velocities_by_sc.get(CALIBRATION_SC, [])
    live_analytical_evaluated = (
        [float(row["reduced_velocity"]) for row in calibration_fine]
        == qualification_velocities
        and expected_calibration_velocities == qualification_velocities
        and bool(calibration_fine)
    )
    if live_analytical_evaluated:
        live_values = np.asarray(
            [float(row["tip_peak_amplitude_over_diameter"]) for row in calibration_fine]
        )
        analytical_values = np.asarray(
            [
                next(
                    float(curve["analytical_equilibrium_amplitude_over_diameter"])
                    for curve in curve_rows
                    if float(curve["scruton_number"]) == CALIBRATION_SC
                    and float(curve["reduced_velocity"])
                    == float(row["reduced_velocity"])
                )
                for row in calibration_fine
            ]
        )
        errors = live_values - analytical_values
        live_analytical_rmse = float(np.sqrt(np.mean(errors**2)))
        live_analytical_max_abs = float(np.max(np.abs(errors)))
        live_peak_index = int(np.argmax(live_values))
        analytical_peak_index = int(np.argmax(analytical_values))
        live_peak_error = abs(
            float(live_values[live_peak_index])
            - float(analytical_values[analytical_peak_index])
        )
        live_peak_tolerance = max(
            0.03, 0.15 * abs(float(analytical_values[analytical_peak_index]))
        )
        live_peak_location_error = abs(
            float(calibration_fine[live_peak_index]["reduced_velocity"])
            - float(calibration_fine[analytical_peak_index]["reduced_velocity"])
        )
        active_coordinate_rows = [
            row
            for row in calibration_fine
            if float(row["tip_peak_amplitude_over_diameter"])
            >= ACTIVE_AMPLITUDE_RATIO
        ]
        coordinate_complete = bool(active_coordinate_rows) and all(
            row["response_estimator_final_mean_amplitude_ratio"] is not None
            for row in active_coordinate_rows
        )
        coordinate_pass = coordinate_complete and all(
            abs(
                float(row["tip_peak_amplitude_over_diameter"])
                - float(row["response_estimator_final_mean_amplitude_ratio"])
            )
            <= max(
                COORDINATE_ABSOLUTE_AMPLITUDE_RATIO_TOLERANCE,
                COORDINATE_RELATIVE_AMPLITUDE_TOLERANCE
                * abs(float(row["tip_peak_amplitude_over_diameter"])),
            )
            for row in active_coordinate_rows
        )
        live_analytical_pass = (
            live_analytical_rmse <= LIVE_ANALYTICAL_RMSE_TOLERANCE
            and live_analytical_max_abs <= LIVE_ANALYTICAL_MAX_ABS_TOLERANCE
            and live_peak_error <= live_peak_tolerance
            and live_peak_location_error
            <= LIVE_ANALYTICAL_PEAK_LOCATION_TOLERANCE_USTAR
            and coordinate_pass
        )
    else:
        live_analytical_rmse = None
        live_analytical_max_abs = None
        live_peak_error = None
        live_peak_tolerance = None
        live_peak_location_error = None
        coordinate_complete = False
        coordinate_pass = False
        live_analytical_pass = False
    live_analytical_summary = {
        "evaluated": live_analytical_evaluated,
        "pass": live_analytical_pass,
        "rmse_tip_amplitude_over_diameter": live_analytical_rmse,
        "rmse_tolerance": LIVE_ANALYTICAL_RMSE_TOLERANCE,
        "maximum_absolute_error_tip_amplitude_over_diameter": live_analytical_max_abs,
        "maximum_absolute_error_tolerance": LIVE_ANALYTICAL_MAX_ABS_TOLERANCE,
        "peak_amplitude_absolute_error": live_peak_error,
        "peak_amplitude_tolerance": live_peak_tolerance,
        "peak_location_absolute_error_u_star": live_peak_location_error,
        "peak_location_tolerance_u_star": LIVE_ANALYTICAL_PEAK_LOCATION_TOLERANCE_USTAR,
        "coordinate_consistency_evaluated": coordinate_complete,
        "coordinate_consistency_pass": coordinate_pass,
        "scope": (
            "live QBlade reproduction of the frozen analytical closure; not an "
            "independent hGAST or experimental comparison"
        ),
    }

    reference_u, reference_amplitude = _read_figure9_series(
        figure9, CALIBRATION_SERIES
    )
    calibration_curve = sorted(
        [row for row in curve_rows if float(row["scruton_number"]) == CALIBRATION_SC],
        key=lambda row: float(row["reduced_velocity"]),
    )
    calibration_enhanced_curve_complete = (
        [float(row["reduced_velocity"]) for row in calibration_curve]
        == qualification_velocities
        and all(row["selected_timestep_level"] == "dt_half" for row in calibration_curve)
        and expected_calibration_velocities == qualification_velocities
    )
    lupi_curve_rows, lupi_curve_summary = lupi_experimental_curve_screen(
        calibration_curve,
        figure9,
    )
    _write_csv(output / "lupi_experimental_curve_comparison.csv", lupi_curve_rows)
    frozen_curve_path = (
        REPO_ROOT
        / "paper_drafts"
        / "NAWEA_QBladeBridge"
        / "plot_data"
        / "cantilever_bridge_sweep.csv"
    )
    if calibration_enhanced_curve_complete and frozen_curve_path.is_file():
        with frozen_curve_path.open(newline="", encoding="utf-8") as handle:
            frozen_rows = list(csv.DictReader(handle))
        material_improvement = material_improvement_metrics(
            reference_u=reference_u,
            reference_amplitude=reference_amplitude,
            response_u=[float(row["reduced_velocity"]) for row in calibration_curve],
            response_amplitude=[
                float(row["tip_peak_amplitude_over_diameter"])
                for row in calibration_curve
            ],
            frozen_u=[float(row["reduced_velocity"]) for row in frozen_rows],
            frozen_amplitude=[
                float(row["terminal_tip_amplitude_over_diameter"])
                for row in frozen_rows
            ],
        )
        material_improvement["frozen_curve_path"] = frozen_curve_path.resolve().as_posix()
        material_improvement["frozen_curve_sha256"] = sha256_file(frozen_curve_path)
    else:
        material_improvement = {
            "evaluated": False,
            "pass": False,
            "reason": (
                "the complete frozen 12-point half-dt calibration curve and "
                "frozen-source curve are required"
            ),
        }

    heldout_material_rows = []
    for row in [value for value in peak_rows if value["role"] == "held_out"]:
        skill = row["material_skill_over_no_viv_at_reference_peak"]
        no_viv_evaluated = bool(row["enhanced_curve_complete"]) and skill is not None
        no_viv_pass = (
            no_viv_evaluated and skill >= MATERIAL_IMPROVEMENT_MINIMUM_SKILL
        )
        heldout_material_rows.append(
            {
                "scruton_number": row["scruton_number"],
                "reference_peak_u_star": row["reference_peak_u_star"],
                "reference_peak_over_diameter": row[
                    "hgast_reference_peak_over_diameter"
                ],
                "response_at_reference_peak_over_diameter": row[
                    "predicted_at_reference_peak_u_star_over_diameter"
                ],
                "skill_over_no_viv": skill,
                "minimum_skill_required": MATERIAL_IMPROVEMENT_MINIMUM_SKILL,
                "no_viv_evaluated": no_viv_evaluated,
                "no_viv_pass": no_viv_pass,
                "matched_frozen_evaluated": False,
                "matched_frozen_pass": False,
                "assessment_pass": no_viv_pass,
                "qualification_pass": False,
                "matched_frozen_reason": (
                    "matched held-out frozen-source sweeps are not in this campaign"
                ),
            }
        )
    _write_csv(output / "heldout_material_improvement.csv", heldout_material_rows)
    heldout_material_no_viv_evaluated = (
        len(heldout_material_rows) == len(HELD_OUT_SC)
        and all(bool(row["no_viv_evaluated"]) for row in heldout_material_rows)
    )
    heldout_material_no_viv_pass = heldout_material_no_viv_evaluated and all(
        bool(row["no_viv_pass"]) for row in heldout_material_rows
    )
    # A qualification claim requires matched legacy frozen-phase bridge sweeps
    # for both held-out Scruton cases. No such data are supplied by this
    # campaign, and the legacy Sc=5.18 manuscript sweep is neither held out nor
    # matched.
    heldout_matched_frozen_evaluated = False
    heldout_matched_frozen_pass = False

    reference_low, reference_high = _band_endpoints(
        reference_u,
        reference_amplitude,
        CALIBRATION_THRESHOLD_FRACTION * float(np.max(reference_amplitude)),
    )
    if calibration_enhanced_curve_complete:
        predicted_u = [float(row["reduced_velocity"]) for row in calibration_curve]
        predicted_amplitude = [
            float(row["tip_peak_amplitude_over_diameter"])
            for row in calibration_curve
        ]
        predicted_low, predicted_high = _band_endpoints(
            predicted_u,
            predicted_amplitude,
            CALIBRATION_THRESHOLD_FRACTION * float(np.max(reference_amplitude)),
        )
    else:
        predicted_low, predicted_high = None, None
    band_evaluated = all(
        value is not None
        for value in (reference_low, reference_high, predicted_low, predicted_high)
    )
    band_low_error = (
        None if not band_evaluated else abs(float(predicted_low) - float(reference_low))
    )
    band_high_error = (
        None if not band_evaluated else abs(float(predicted_high) - float(reference_high))
    )
    calibration_band_summary = {
        "evaluated": band_evaluated,
        "pass": band_evaluated
        and band_low_error <= LOCK_IN_ENDPOINT_TOLERANCE_USTAR
        and band_high_error <= LOCK_IN_ENDPOINT_TOLERANCE_USTAR,
        "threshold_fraction_of_reference_peak": CALIBRATION_THRESHOLD_FRACTION,
        "reference_low_u_star": reference_low,
        "reference_high_u_star": reference_high,
        "predicted_low_u_star": predicted_low,
        "predicted_high_u_star": predicted_high,
        "low_endpoint_absolute_error_u_star": band_low_error,
        "high_endpoint_absolute_error_u_star": band_high_error,
        "endpoint_tolerance_u_star": LOCK_IN_ENDPOINT_TOLERANCE_USTAR,
    }

    calibration_stage = calibration_stage_assessment(
        manifest=manifest,
        result_rows=result_rows,
        runtime_audits=runtime_audits,
        dt_rows=dt_rows,
        dt_curve_rows=dt_curve_rows,
        mechanism_rows=mechanism_rows,
        live_analytical_summary=live_analytical_summary,
        calibration_curve_summary=calibration_curve_summary,
        calibration_band_summary=calibration_band_summary,
        peak_rows=peak_rows,
        equation_reproduction_summary=equation_gate_summary,
    )

    complete = (
        not missing
        and not result_errors
        and len(result_rows) == len(manifest["cases"])
    )
    dt_complete = bool(dt_rows) and len(dt_rows) * 2 == len(result_rows)
    dt_pass = dt_complete and all(bool(row["pass"]) for row in dt_rows)
    calibration_rows = [row for row in peak_rows if row["role"] == "calibration"]
    heldout_rows = [row for row in peak_rows if row["role"] == "held_out"]
    calibration_peak_pass = len(calibration_rows) == 1 and bool(
        calibration_rows[0]["external_code_comparison_pass"]
    )
    calibration_curve_pass = bool(calibration_curve_summary["pass"])
    calibration_pass = calibration_peak_pass and calibration_curve_pass
    heldout_pass = len(heldout_rows) == 2 and all(
        bool(row["external_code_comparison_pass"]) for row in heldout_rows
    )
    colocated_heldout_pass = len(heldout_rows) == 2 and all(
        bool(row["colocated_external_code_comparison_pass"])
        for row in heldout_rows
    )
    peak_location_pass = len(peak_rows) == len(expected_velocities_by_sc) and all(
        bool(row["peak_location_pass"]) for row in peak_rows
    )
    enhanced_dt_evaluated = (
        dt_complete
        and len(dt_curve_rows) == len(expected_velocities_by_sc)
        and all(bool(row["evaluated"]) for row in dt_curve_rows)
        and all(bool(row["frequency_evaluated"]) for row in dt_rows)
    )
    enhanced_dt_pass = (
        enhanced_dt_evaluated
        and dt_pass
        and all(bool(row["frequency_pass"]) for row in dt_rows)
        and all(bool(row["both_stationary"]) for row in dt_rows)
        and all(bool(row["pass"]) for row in dt_curve_rows)
    )
    mechanism_evaluated = (
        len(mechanism_rows) == len(expected_velocities_by_sc)
        and all(bool(row["evaluated"]) for row in mechanism_rows)
    )
    mechanism_pass = mechanism_evaluated and all(
        bool(row["pass"]) for row in mechanism_rows
    )
    fully_evaluated = complete and bool(calibration_curve_summary["evaluated"])
    status = (
        "comparison_pass"
        if fully_evaluated and dt_pass and calibration_pass and heldout_pass
        else ("not_evaluated" if not fully_evaluated else "comparison_fail")
    )
    assessment_fully_evaluated = (
        fully_evaluated
        and runtime_complete
        and enhanced_dt_evaluated
        and mechanism_evaluated
        and bool(live_analytical_summary["evaluated"])
        and bool(calibration_band_summary["evaluated"])
        and bool(material_improvement["evaluated"])
        and heldout_material_no_viv_evaluated
        and (
            not equation_gate_required
            or bool(equation_gate_summary and equation_gate_summary["evaluated"])
        )
    )
    assessment_pass = (
        assessment_fully_evaluated
        and status == "comparison_pass"
        and stationarity_complete
        and runtime_pass
        and enhanced_dt_pass
        and mechanism_pass
        and bool(live_analytical_summary["pass"])
        and bool(calibration_band_summary["pass"])
        and peak_location_pass
        and colocated_heldout_pass
        and bool(material_improvement["pass"])
        and heldout_material_no_viv_pass
        and (
            not equation_gate_required
            or bool(equation_gate_summary and equation_gate_summary["pass"])
        )
    )
    assessment_status = (
        "assessment_pass"
        if assessment_pass
        else (
            "not_evaluated"
            if not assessment_fully_evaluated
            else "assessment_fail"
        )
    )
    qualification_input_eligible = bool(
        manifest.get("numerics", {}).get("qualification_input_eligible", False)
    )
    qualification_fully_evaluated = (
        assessment_fully_evaluated
        and qualification_input_eligible
        and heldout_matched_frozen_evaluated
    )
    qualification_pass = (
        qualification_fully_evaluated
        and assessment_pass
        and heldout_matched_frozen_pass
    )
    qualification_status = (
        "qualification_pass"
        if qualification_pass
        else (
            "not_evaluated"
            if not qualification_fully_evaluated
            else "qualification_fail"
        )
    )
    summary = {
        "schema": SCHEMA,
        "status": status,
        "calibration_stage_status": calibration_stage["status"],
        "calibration_stage_evaluated": calibration_stage["evaluated"],
        "calibration_stage_pass": calibration_stage["pass"],
        "lupi_experimental_curve_screen_evaluated": lupi_curve_summary[
            "evaluated"
        ],
        "lupi_experimental_curve_screen_pass": lupi_curve_summary["pass"],
        "enhanced_assessment_status": assessment_status,
        "enhanced_qualification_status": qualification_status,
        "assessment_evaluated": assessment_fully_evaluated,
        "assessment_pass": assessment_pass,
        "qualification_evaluated": qualification_fully_evaluated,
        "qualification_pass": qualification_pass,
        "qualification_input_eligible": qualification_input_eligible,
        "scope": (
            "external code response comparison/model-form assessment against hGAST; the simple closure does not "
            "solve the Hartlen-Currie equations, so this is not equation-level verification; experiment columns "
            "are a separate validation screen"
        ),
        "runtime_history_storage_scope": (
            "analysis streams only required columns with constant row memory; qualification accepts the "
            "hash-bound compact SIL schema when it retains response, startup-seed, applied-work, limit, "
            "and one-step-lag digest evidence, while legacy full histories remain supported"
        ),
        "coefficient_fingerprint_sha256": manifest["calibration_policy"]["coefficient_fingerprint_sha256"],
        "calibration_scruton_number": CALIBRATION_SC,
        "calibration_digitized_peak_locations": {
            "hgast_reduced_velocity": calibration_hgast_peak["reduced_velocity"],
            "hgast_tip_amplitude_over_diameter": calibration_hgast_peak[
                "tip_amplitude_over_diameter"
            ],
            "lupi_reduced_velocity": calibration_lupi_peak["reduced_velocity"],
            "lupi_tip_amplitude_over_diameter": calibration_lupi_peak[
                "tip_amplitude_over_diameter"
            ],
            "source": "sampled maxima of hash-bound digitized Figure 9 series",
        },
        "held_out_scruton_numbers": list(HELD_OUT_SC),
        "held_out_targets_accessed_during_calibration": False,
        "expected_case_count": len(manifest["cases"]),
        "completed_case_count": len(result_rows),
        "missing_case_ids": missing,
        "result_errors": result_errors,
        "minimum_global_steps_per_cycle": min(
            float(entry["steps_per_natural_cycle"]) for entry in manifest["cases"]
        ),
        "minimum_global_steps_per_cycle_required": MINIMUM_GLOBAL_STEPS_PER_CYCLE,
        "checks": {
            "all_results_complete": complete,
            "dt_halving_complete": dt_complete,
            "dt_halving_pass": dt_pass,
            "calibration_external_code_peak_comparison_pass": calibration_peak_pass,
            "calibration_external_code_curve_comparison_pass": calibration_curve_pass,
            "held_out_external_code_peak_comparison_pass": heldout_pass,
        },
        "enhanced_checks": {
            "all_result_windows_stationary": stationarity_complete,
            "runtime_artifacts_complete": runtime_complete,
            "runtime_execution_stationarity_and_limits_pass": runtime_pass,
            "enhanced_timestep_evaluated": enhanced_dt_evaluated,
            "enhanced_timestep_pass": enhanced_dt_pass,
            "live_vs_analytical_evaluated": live_analytical_summary["evaluated"],
            "live_vs_analytical_pass": live_analytical_summary["pass"],
            "lock_in_mechanism_evaluated": mechanism_evaluated,
            "lock_in_mechanism_pass": mechanism_pass,
            "calibration_band_endpoint_pass": calibration_band_summary["pass"],
            "peak_location_pass": peak_location_pass,
            "heldout_colocated_peak_amplitude_pass": colocated_heldout_pass,
            "calibration_material_improvement_pass": material_improvement["pass"],
            "lupi_experimental_curve_screen_evaluated": lupi_curve_summary[
                "evaluated"
            ],
            "lupi_experimental_curve_screen_pass": lupi_curve_summary["pass"],
            "heldout_material_improvement_over_no_viv_pass": heldout_material_no_viv_pass,
            "heldout_material_improvement_over_matched_frozen_evaluated": heldout_matched_frozen_evaluated,
            "heldout_material_improvement_over_matched_frozen_pass": heldout_matched_frozen_pass,
        },
        "calibration_curve_comparison": calibration_curve_summary,
        "calibration_stage_assessment": calibration_stage,
        "lupi_experimental_curve_screen": lupi_curve_summary,
        "live_vs_analytical_comparison": live_analytical_summary,
        "calibration_lock_in_band_comparison": calibration_band_summary,
        "material_improvement_over_no_and_frozen": material_improvement,
        "heldout_material_improvement_over_no_viv": heldout_material_rows,
        "timestep_curve_qualification": dt_curve_rows,
        "lock_in_mechanism_qualification": mechanism_rows,
        "acceptance_definition": {
            "dt_halving": "absolute y_max/D difference <= max(0.01, 5% of half-dt value)",
            "external_code_peak_comparison": "absolute y_max/D error <= max(0.03, 15% of hGAST Figure 10 peak)",
            "calibration_curve": "Sc=5.18 RMSE in y_max/D <= 0.05 over every digitized hGAST Figure 9 point",
            "experiment": (
                "separate Sc=5.18 validation screen: in-range-marker RMSE<=0.05D, "
                "peak U* error<=0.30, and peak amplitude error<=max(0.03D,15%); "
                "not an analytical-reproduction or calibration-stage requirement"
            ),
        },
        "enhanced_qualification_definition": {
            "claim_boundary": (
                "additional closure-reproduction, stationarity, mechanism, and material-improvement gates; "
                "the original external-code acceptance thresholds above are unchanged"
            ),
            "stationarity": (
                "last 40 natural cycles split 20+20; amplitude difference/D <= max(0.005, 5% final); "
                "active-case frequency difference/fn <= 0.01"
            ),
            "lock_in_frequency": (
                "at amplitudes >=10% of the reference peak, |f_response/fn-1| <= 0.05, "
                "mode-band power fraction >=0.80, at least three consecutive points, and the "
                "reference-peak neighborhood is active"
            ),
            "limits": (
                "zero source/response limit counts and <=90% of configured gain and nodal-force safeguards"
            ),
            "live_vs_analytical": (
                "RMSE<=0.03D, maximum absolute error<=0.05D, peak amplitude error<=max(0.03D,15%), "
                "peak U* error<=0.25, and active tip/controller coordinate difference<=max(0.01D,5%)"
            ),
            "material_improvement": (
                "Sc=5.18 RMSE skill >=0.50 versus both zero-VIV and the frozen-source curve, with peak "
                "magnitude and location errors no more than half the frozen errors on one shared U* grid; "
                "each heldout assessment must have co-located peak skill >=0.50 versus zero-VIV, while "
                "qualification additionally requires matched heldout legacy frozen-phase bridge sweeps"
            ),
            "fatigue": (
                "amplitude improvement is not fatigue conservatism; matched local-load histories and "
                "independent direct pseudo-damage holdouts are required"
            ),
        },
        "peak_comparisons": peak_rows,
    }
    if equation_gate_required:
        summary["ambient_v2_equation_reproduction"] = equation_gate_summary
        summary["enhanced_checks"][
            "ambient_v2_equation_reproduction_evaluated"
        ] = bool(equation_gate_summary and equation_gate_summary["evaluated"])
        summary["enhanced_checks"][
            "ambient_v2_equation_reproduction_pass"
        ] = bool(equation_gate_summary and equation_gate_summary["pass"])
        summary["enhanced_qualification_definition"][
            "ambient_v2_equation_reproduction"
        ] = (
            "fine response-estimator RMSE<=0.005D, max error<=0.010D, "
            "active U* grid exactly {5.8,6.2,6.25,6.5} at 0.05D, and "
            "pointwise coarse/fine estimator difference<=0.005D"
        )
    summary_path = output / "acceptance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if curve_rows:
        _plot_response_curves(output / "response_peak_vs_reduced_velocity", curve_rows, figure9, peak_rows)
    return summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration = subparsers.add_parser("calibrate", help="Write the Sc=5.18-only frozen coefficient file.")
    calibration.add_argument("--output", type=Path, required=True)

    for command, help_text, default_output, response_schema_version in (
        (
            "generate",
            "Generate the legacy schema-v1 QBlade campaign and immutable manifest.",
            HERE / "generated_response_campaign",
            1,
        ),
        (
            "generate-v2",
            "Generate the separate declared-ambient schema-v2 campaign root.",
            DEFAULT_AMBIENT_V2_CAMPAIGN_ROOT,
            2,
        ),
    ):
        generation = subparsers.add_parser(command, help=help_text)
        generation.set_defaults(response_schema_version=response_schema_version)
        generation.add_argument("--output", type=Path, default=default_output)
        generation.add_argument(
            "--duration",
            type=float,
            default=None,
            help=(
                "Explicit fixed duration for a diagnostic campaign. If omitted, use the "
                "frozen case-specific stationarity caps; the legacy 8 s duration is not "
                "qualification eligible."
            ),
        )
        generation.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP_S)
        generation.add_argument("--no-dt-halving", action="store_true")
        generation.add_argument(
            "--reduced-velocities",
            type=float,
            nargs="+",
            default=DEFAULT_REDUCED_VELOCITIES,
        )
        generation.add_argument(
            "--scruton",
            type=float,
            nargs="+",
            default=tuple(PUBLISHED_STRUCTURAL_CASES),
        )

    run = subparsers.add_parser("run", help="Write a run plan, or explicitly execute it.")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--results", type=Path, required=True)
    run.add_argument("--stage", choices=("calibration", "held_out", "all"), required=True)
    run.add_argument("--qblade-root", type=Path, default=REPO_ROOT / "run_headless" / "QBladeCE_2.0.9.7")
    run.add_argument("--checkpoint-every", type=int, default=0)
    run.add_argument(
        "--threads",
        type=int,
        default=4,
        help="QBlade threads per sequential case (must remain below eight).",
    )
    run.add_argument("--plan-output", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument(
        "--allow-failed-analytical-gate",
        action="store_true",
        help="Permit an explicitly diagnostic run after the recorded simple-model gate failure.",
    )

    analysis = subparsers.add_parser("analyze", help="Extract peak response, dt convergence, and acceptance.")
    analysis.add_argument("--campaign", type=Path, required=True)
    analysis.add_argument("--results", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--response-window-cycles", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "calibrate":
        result = write_frozen_coefficients(args.output, calibrate_response_coefficients())
    elif args.command in {"generate", "generate-v2"}:
        result = generate_campaign(
            args.output,
            reduced_velocities=args.reduced_velocities,
            timestep_s=args.timestep,
            include_dt_halving=not args.no_dt_halving,
            duration_s=args.duration,
            selected_scruton_numbers=args.scruton,
            response_schema_version=args.response_schema_version,
        )
    elif args.command == "run":
        plan = build_run_plan(
            args.campaign,
            args.results,
            stage=args.stage,
            qblade_root=args.qblade_root,
            checkpoint_every=args.checkpoint_every,
            threads=args.threads,
            diagnostic_only=args.allow_failed_analytical_gate,
        )
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for row in plan["commands"]:
            print(shlex.join(row["argv"]))
        if args.execute:
            execute_run_plan(
                plan,
                args.campaign,
                allow_failed_analytical_gate=args.allow_failed_analytical_gate,
            )
        result = args.plan_output.resolve()
    else:
        result = analyze_campaign(
            args.campaign,
            args.results,
            args.output,
            response_window_cycles=args.response_window_cycles,
        )
    print(result)


if __name__ == "__main__":
    main()
