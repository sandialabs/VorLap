#!/usr/bin/env python3
"""Prepare and assess actual-aero IEA-10-MW native/complete bridge cases.

This is a numerical integration campaign at four conditions already present in
the frozen turbine training table.  It is not a new validation dataset.  Each
matched pair shares public IEA geometry/polars and one TurbSim field.  Native
QBlade supplies turbulent aerodynamic loading; the complete member adds the
uncaptured Bidadi fixed-body spectrum and the trained motion-correlated work
closure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import generate_case as base
from .response_aware_benchmark import derive_blade_quadrature
from vorlap.qblade_case_prep import QBladeCasePrepRequest, prepare_qblade_external_case
from vorlap.qblade_files import replace_qblade_keyword_text
from vorlap.qblade_sil import (
    normalize_qblade_header,
    read_qblade_result_numeric_columns,
)
from vorlap.turbsim import generate_turbsim_wind_file

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "vorlap-iea10-actual-aero-complete-campaign-v1"
# U=6 m/s would put the thinnest tip below the frozen Re=5e4 source bound.
# The user explicitly excluded very-low-Re cases, so the matrix starts at U=10.
SPEEDS_MPS = (10.0, 14.0, 18.0, 22.0)
DT_S = 0.006
DEFAULT_DURATION_S = 12.0
INITIAL_YAW_DEG = 178.71455998
SHAFT_TILT_DEG = 50.30132183
TOWER_HEIGHT_M = 2.5
TRAINING_MODEL = (
    REPO_ROOT
    / "examples/horcas2020_iea10_tip_geometry/evidence/combined_tip_v1/model.json"
)
GEOMETRY_CALIBRATION = (
    REPO_ROOT / "examples/viv_geometry_blended/geometry_blended_calibration.json"
)
SPECTRUM_DIR = REPO_ROOT / "data/airfoils/bidadi2024_ffa_w3_spectral"
PROFILES = (211, 241, 270, 301, 330, 360, 500)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _patch(path: Path, values: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text, _old, count = replace_qblade_keyword_text(text, key, value)
        if count != 1:
            raise ValueError(f"Could not patch exactly one {key} in {path}.")
    path.write_text(text, encoding="utf-8")


def _numeric_rows_after_numalf(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = next(i for i, line in enumerate(lines) if "NumAlf" in line)
    rows = []
    for line in lines[marker + 1 :]:
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            rows.append([float(value) for value in fields[:4]])
        except ValueError:
            continue
    data = np.asarray(rows, dtype=float)
    if data.shape != (200, 4) or np.any(~np.isfinite(data)):
        raise ValueError(f"Expected exactly 200 finite polar rows in {path}.")
    if np.any(np.diff(data[:, 0]) <= 0.0) or np.any(data[:, 2] < 0.0):
        raise ValueError(f"Invalid alpha/Cd data in {path}.")
    return data


def _coordinates(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            rows.append([float(fields[0]), float(fields[1])])
        except ValueError:
            continue
    # The first numeric row is the OpenFAST airfoil-reference point.
    data = np.asarray(rows[1:], dtype=float)
    if data.shape != (200, 2) or np.any(~np.isfinite(data)):
        raise ValueError(f"Expected exactly 200 contour points in {path}.")
    # AeroDyn stores lower TE -> LE -> upper TE; QBlade's imported-airfoil
    # convention is upper TE -> LE -> lower TE.
    return data[::-1].copy()


def _thickness_ratio(coordinates: np.ndarray) -> float:
    upper = coordinates[coordinates[:, 1] >= 0.0]
    lower = coordinates[coordinates[:, 1] <= 0.0]
    grid = np.linspace(0.0, 1.0, 2001)
    yu = np.interp(grid, np.sort(upper[:, 0]), upper[np.argsort(upper[:, 0]), 1])
    yl = np.interp(grid, np.sort(lower[:, 0]), lower[np.argsort(lower[:, 0]), 1])
    return float(np.max(yu - yl))


def _profile_for_thickness(thickness: float) -> int:
    return min(PROFILES, key=lambda value: abs(value / 1000.0 - thickness))


def _write_public_aero(case_root: Path, iea_root: Path) -> list[dict[str, Any]]:
    model = case_root / base.CASE_NAME
    aero = model / "Aero"
    foil_dir = aero / "Airfoils"
    polar_dir = aero / "Polars"
    for path in foil_dir.glob("*"):
        path.unlink()
    for path in polar_dir.glob("*"):
        path.unlink()
    mappings = []
    replacements = {}
    for index in range(30):
        source_foil = (
            iea_root
            / "openfast/Airfoils"
            / f"IEA-10.0-198-RWT_AF{index:02d}_Coords.txt"
        )
        source_polar = (
            iea_root
            / "openfast/Airfoils"
            / f"IEA-10.0-198-RWT_AeroDyn15_Polar_{index:02d}.dat"
        )
        coordinates = _coordinates(source_foil)
        polar = _numeric_rows_after_numalf(source_polar)
        thickness = _thickness_ratio(coordinates)
        profile = _profile_for_thickness(thickness)
        airfoil_id = f"FFA_W3_{profile}"
        foil_name = f"IEA10_AF{index:02d}.afl"
        polar_name = f"{airfoil_id}_STATION_{index:02d}.plr"
        (foil_dir / foil_name).write_text(
            f"IEA-10-MW public AF{index:02d}; source commit {base.EXPECTED_SOURCE_COMMIT}\n"
            + "\n".join(f"{x:.15g} {y:.15g}" for x, y in coordinates)
            + "\n",
            encoding="utf-8",
        )
        lines = [
            "----------------------------------------QBlade Multi RE Polar File--------------------------------------------------",
            "Public IEA-10-MW OpenFAST polar converted without coefficient fitting",
            "----------------------------------------Object Names----------------------------------------------------------------",
            f"IEA10_AF{index:02d}_RE1E7 POLARNAME",
            f"../Airfoils/{foil_name} FOILNAME",
            "----------------------------------------Parameters------------------------------------------------------------------",
            f"{100.0 * thickness:.10f} THICKNESS",
            "false ISDECOMPOSED",
            "REYNOLDS 1.0000E+07",
            "----------------------------------------Polar Data------------------------------------------------------------------",
            "AOA CL CD CM",
        ]
        lines.extend(" ".join(f"{value:.15g}" for value in row) for row in polar)
        (polar_dir / polar_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        replacements[f"Polars/IEA10_STATION_{index:02d}.plr"] = f"Polars/{polar_name}"
        mappings.append(
            {
                "station": index,
                "public_airfoil": f"AF{index:02d}",
                "thickness_ratio": thickness,
                "spectral_proxy": airfoil_id,
                "foil_sha256": sha256_file(source_foil),
                "polar_sha256": sha256_file(source_polar),
            }
        )
    blade_path = aero / f"{base.CASE_NAME}.bld"
    text = blade_path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"Expected one blade polar reference {old}.")
        text = text.replace(old, new)
    blade_path.write_text(text, encoding="utf-8")
    source_dir = model / "VorLapAirfoils"
    for path in source_dir.glob("*.h5"):
        path.unlink()
    for profile in PROFILES:
        source = SPECTRUM_DIR / f"ffa_w3_{profile}.h5"
        if not source.is_file():
            raise FileNotFoundError(f"Missing frozen FFA spectrum: {source}")
        shutil.copy2(source, source_dir / source.name)
    return mappings


def _training_rows() -> dict[float, dict[str, Any]]:
    model = json.loads(TRAINING_MODEL.read_text(encoding="utf-8"))
    rows = model["training"]["fu2026_speed_orientation_rows_after_preserved_failure"]
    result = {
        float(row["wind_speed_mps"]): row
        for row in rows
        if row["config"] == "P80A310" and float(row["wind_speed_mps"]) in SPEEDS_MPS
    }
    if tuple(sorted(result)) != SPEEDS_MPS:
        raise ValueError("Frozen P80A310 training rows differ from the four-case grid.")
    return result


def _response_model(quadrature: Any, enabled: bool) -> dict[str, Any]:
    return {
        "type": "modal_negative_damping",
        "schema_version": 2,
        "aerodynamic_speed_basis": "declared_ambient_crossflow",
        "node_ids": list(quadrature.controller_node_ids),
        "mode_shape": quadrature.response_qblade_mode_shape.tolist(),
        "natural_frequency_hz": quadrature.natural_frequency_hz,
        "reference_length_m": float(quadrature.chord_m[-1]),
        "reduced_velocity_low": 0.1,
        "reduced_velocity_peak": 1.0,
        "reduced_velocity_high": 1000.0,
        "aerodynamic_damping_gain_h1": 0.0,
        "saturation_amplitude_ratio": 1.0,
        "rms_time_constant_s": 1.0 / quadrature.natural_frequency_hz,
        "max_abs_gain": 1.0e6,
        "max_abs_nodal_force_n": 1.0e7,
        "force_direction": "chord",
        "projection_weights": quadrature.modal_mass_weights_kg.tolist(),
        "bandpass_damping_ratio": 0.1,
        "enabled": True,
        "startup_seed": {
            "schema_version": 1,
            "enabled": bool(enabled),
            "duration_cycles": 3.0,
            "target_amplitude_ratio": 0.001,
            "structural_damping_ratio": quadrature.structural_damping_ratio,
            "taper": "hann_sine_squared",
            "carrier": "natural_frequency",
            "nodal_weighting": "projection_mass_times_mode",
            "nodal_force_limit_source": "response_model_max_abs_nodal_force_n",
        },
    }


def _adapter_config(
    config_dir: Path, row: dict[str, Any], quadrature: Any, enabled: bool
) -> dict[str, Any]:
    calibration = config_dir / GEOMETRY_CALIBRATION.name
    trained = config_dir / "combined_tip_v1_model.json"
    shutil.copy2(GEOMETRY_CALIBRATION, calibration)
    shutil.copy2(TRAINING_MODEL, trained)
    source_hashes = {
        f"FFA_W3_{profile}": sha256_file(SPECTRUM_DIR / f"ffa_w3_{profile}.h5")
        for profile in PROFILES
    }
    return {
        "schema_version": 3,
        "type": "single_mode_airfoil_geometry_blended_complete_transfer_v3",
        "profile": "FFA-W3_family",
        "kinematics": "fixed_nonrotating",
        "calibration": {"path": calibration.name, "sha256": sha256_file(calibration)},
        "trained_condition": {
            "path": trained.name,
            "sha256": sha256_file(trained),
            "case_id": row["case_id"],
            "wind_speed_mps": row["wind_speed_mps"],
            "zero_amplitude_feedback_over_structural_damping": row[
                "zero_amplitude_feedback_over_structural_damping"
            ],
            "limiting_amplitude_m": row["configuration_saturation_amplitude_m"],
        },
        "static_spectrum": {
            "channel": "CL",
            "bounds_policy": "raise",
            "source_sha256_by_airfoil": source_hashes,
            "n_frequency_entries": 200,
            "maximum_spectral_gain": 10.0,
            "minimum_modal_participation": 1.0e-12,
            "retain_uncaptured_lift_spectrum": True,
            "retain_fluctuating_drag_spectrum": True,
            "include_mean_load": False,
        },
        "modal_power": {
            "rms_time_constant_s": 1.0 / quadrature.natural_frequency_hz,
            "bandpass_damping_ratio": 0.1,
            "max_abs_target_power_w": 1.0e8,
            "max_abs_generalized_force_n": 1.0e8,
            "max_abs_nodal_force_n": 1.0e7,
            "limit_action": "error",
            "enabled": bool(enabled),
        },
        "structural_damping_ratio": quadrature.structural_damping_ratio,
        "residual_source_scale": 1.0 if enabled else 0.0,
    }


def _wind_settings(speed: float, duration: float) -> dict[str, Any]:
    return {
        "RandSeed1": 4433456 + int(speed),
        "NumGrid_Z": 39,
        "NumGrid_Y": 9,
        "TimeStep": DT_S,
        # Reserve additional advection time for QBlade's wind auto-shift.
        "AnalysisTime": duration + 10.0,
        "UsableTime": "ALL",
        "HubHt": 37.5,
        "GridHeight": 74.0,
        "GridWidth": 12.0,
        "VFlowAng": 0.0,
        "HFlowAng": 0.0,
        "TurbModel": "IECKAI",
        "IECstandard": 1,
        "IECturbc": "A",
        "IEC_WindType": "NTM",
        "RefHt": 37.5,
        "URef": speed,
    }


def generate(
    output: Path, iea_root: Path, duration: float = DEFAULT_DURATION_S
) -> Path:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace campaign {output}.")
    output.mkdir(parents=True)
    winds = output / "winds"
    winds.mkdir()
    training = _training_rows()
    turbsim = REPO_ROOT / "run_headless/openfast/install/bin/turbsim"
    wind_records = {}
    for speed in SPEEDS_MPS:
        wind = winds / f"U{int(speed):02d}.bts"
        record = generate_turbsim_wind_file(
            wind_file=wind,
            settings=_wind_settings(speed, duration),
            executable=turbsim,
            vorlap_root=REPO_ROOT,
        )
        wind_records[str(speed)] = record.as_dict()

    template = output / "_template"
    base.generate(
        SimpleNamespace(
            iea_root=Path(iea_root),
            output=template,
            duration=duration,
            allow_source_commit_mismatch=False,
        )
    )
    mappings = _write_public_aero(template, Path(iea_root))
    cases = []
    for speed in SPEEDS_MPS:
        for role in ("native", "complete"):
            case_id = f"U{int(speed):02d}_{role}"
            case_root = output / "cases" / case_id
            shutil.copytree(template, case_root)
            sim = case_root / f"{base.CASE_NAME}.sim"
            relative_wind = Path("../../winds") / f"U{int(speed):02d}.bts"
            _patch(
                sim,
                {
                    "OBJECTNAME": case_id,
                    "INITIAL_YAW": INITIAL_YAW_DEG,
                    "INITIAL_AZIMUTH": 0.0,
                    "INCLUDEAERO": 1,
                    "WNDTYPE": 1,
                    "WNDNAME": relative_wind.as_posix(),
                    "MEANINF": speed,
                    "HORANGLE": 0.0,
                    "VERTANGLE": 0.0,
                    "TIMESTEP": DT_S,
                    "NUMTIMESTEPS": int(round(duration / DT_S)),
                    "RAMPUP": 5.0,
                    "STOREAERO": "true",
                    "STOREBLADE": "true",
                    "STORESTRUCT": "true",
                    "STOREREPLAY": "true",
                },
            )
            turbine = case_root / base.CASE_NAME / f"{base.CASE_NAME}.trb"
            structure = (
                case_root / base.CASE_NAME / "Structure/iea10_single_blade_main.str"
            )
            _patch(
                turbine,
                {"SHAFTTILT": SHAFT_TILT_DEG, "TOWERHEIGHT": TOWER_HEIGHT_M},
            )
            _patch(
                structure,
                {"SHFTTILT": SHAFT_TILT_DEG, "TWRHEIGHT": TOWER_HEIGHT_M},
            )
            quadrature = derive_blade_quadrature(case_root)
            source_dir = case_root / base.CASE_NAME / "VorLapAirfoils"
            prep = prepare_qblade_external_case(
                QBladeCasePrepRequest(
                    sim_path=sim,
                    airfoil_dir=source_dir,
                    include_struts=False,
                    include_tower=False,
                    node_source="structural",
                    n_freq_depth=200,
                    force_scale=0.0,
                    include_mean_load=False,
                    source_bounds_policy="raise",
                    source_frequency_cap_hz=0.5 / DT_S,
                    response_model=_response_model(quadrature, role == "complete"),
                    include_default_python_path=True,
                    default_python_path=REPO_ROOT,
                    absolute_source_parameter_dir=True,
                )
            )
            config = json.loads(prep.parameter_file_path.read_text(encoding="utf-8"))
            config["single_mode_engineering_lockin"] = _adapter_config(
                prep.parameter_file_path.parent,
                training[speed],
                quadrature,
                role == "complete",
            )
            prep.parameter_file_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            cases.append(
                {
                    "index": len(cases),
                    "case_id": case_id,
                    "speed_mps": speed,
                    "role": role,
                    "sim": sim.relative_to(output).as_posix(),
                    "config": prep.parameter_file_path.relative_to(output).as_posix(),
                    "result_dir": f"results/{case_id}",
                    "training_case_id": training[speed]["case_id"],
                    "target_amplitude_m": training[speed]["stable_peak_amplitude_m"],
                    "requested_steps": int(round(duration / DT_S)),
                }
            )
    shutil.rmtree(template)
    manifest = {
        "schema": SCHEMA,
        "status": "prepared",
        "scope": "training-space numerical integration; not independent validation",
        "composition": [
            "native_qblade_turbsim_aerodynamics",
            "uncaptured_zero_mean_bidadi_fixed_body_spectrum",
            "trained_motion_correlated_negative_damping",
        ],
        "public_iea_commit": base.EXPECTED_SOURCE_COMMIT,
        "training_model": {
            "path": str(TRAINING_MODEL.relative_to(REPO_ROOT)),
            "sha256": sha256_file(TRAINING_MODEL),
        },
        "spectrum_bundle": {
            "path": str((SPECTRUM_DIR / "bundle_manifest.json").relative_to(REPO_ROOT)),
            "sha256": sha256_file(SPECTRUM_DIR / "bundle_manifest.json"),
        },
        "airfoil_mapping": mappings,
        "wind_records": wind_records,
        "cases": cases,
    }
    manifest["fingerprint"] = semantic_sha256(manifest)
    path = output / "campaign_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def analyze(campaign: Path, output: Path | None = None) -> Path:
    root = Path(campaign).resolve()
    manifest = json.loads((root / "campaign_manifest.json").read_text())
    rows = []
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in manifest["cases"]:
        result_dir = root / case["result_dir"]
        outputs = sorted(result_dir.glob("*.txt"))
        if len(outputs) != 1:
            rows.append({**case, "status": "missing"})
            continue
        summaries = sorted(result_dir.glob("*_run_summary.json"))
        if len(summaries) != 1:
            raise ValueError(
                f"Expected exactly one run summary for {case['case_id']}; "
                f"found {len(summaries)}."
            )
        summary_path = summaries[0]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("failed")
            or summary.get("stopped")
            or summary.get("completed_steps") != case["requested_steps"]
            or summary.get("requested_steps") != case["requested_steps"]
            or not summary.get("exported_results")
            or summary.get("sil_vorlap_updates") != case["requested_steps"]
        ):
            raise ValueError(f"Incomplete or failed run summary for {case['case_id']}.")
        columns = read_qblade_result_numeric_columns(outputs[0])
        normalized = {normalize_qblade_header(name): name for name in columns}
        time_name = normalized.get("Time [s]")
        tip_name = normalized.get("Y_b Tip Trl.Def. BLD_1 [m]")
        if time_name is None:
            raise ValueError(f"No time channel in {outputs[0]}.")
        if tip_name is None:
            raise ValueError(
                f"No blade-local edgewise tip-deflection channel in {outputs[0]}."
            )
        time = np.asarray(columns[time_name], dtype=float)
        tail = time >= max(float(time[-1]) - min(6.0, 0.5 * float(time[-1])), 0.0)
        tip = np.asarray(columns[tip_name], dtype=float)
        amplitude = 0.5 * (float(np.max(tip[tail])) - float(np.min(tip[tail])))
        series[case["case_id"]] = (time, tip)
        history_path = result_dir / f"{case['case_id']}_sil_vorlap_history.csv"
        if not history_path.is_file():
            raise ValueError(f"Missing SIL history for {case['case_id']}.")
        response = summary["sil_vorlap_response_diagnostics_summary"]
        expected_residual = 0.0 if case["role"] == "native" else 1.0
        if (
            response.get("model_type")
            != "single_mode_airfoil_geometry_blended_complete_transfer_v3"
            or response.get("residual_source_scale") != expected_residual
        ):
            raise ValueError(f"Unexpected response contract for {case['case_id']}.")
        maximum_force = float(summary["sil_vorlap_max_abs_force"])
        if (case["role"] == "native" and maximum_force != 0.0) or (
            case["role"] == "complete" and maximum_force <= 0.0
        ):
            raise ValueError(
                f"Supplemental-force contract failed for {case['case_id']}."
            )
        callback = summary["native_callback_evidence"]
        callback_counts = {
            name: int(callback[name])
            for name in (
                "update_init_count",
                "update_log_entry_count",
                "update_cache_observation_count",
                "update_count_observed_max",
            )
        }
        if any(callback_counts.values()):
            raise ValueError(
                f"Unexpected native callback activity for {case['case_id']}."
            )
        spectral_closure = float(response["spectral"]["maximum_energy_closure_error"])
        if not np.isfinite(spectral_closure) or spectral_closure > 1.0e-12:
            raise ValueError(f"Spectral energy closure failed for {case['case_id']}.")
        rows.append(
            {
                **case,
                "status": "complete",
                "terminal_peak_amplitude_m": amplitude,
                "result": str(outputs[0].relative_to(root)),
                "result_sha256": sha256_file(outputs[0]),
                "run_summary": str(summary_path.relative_to(root)),
                "run_summary_sha256": sha256_file(summary_path),
                "sil_history": str(history_path.relative_to(root)),
                "sil_history_sha256": sha256_file(history_path),
                "model_type": response["model_type"],
                "residual_source_scale": response["residual_source_scale"],
                "maximum_supplemental_force_n": maximum_force,
                "spectral_stretch_gain": float(
                    response["last"]["spectral_stretch_gain"]
                ),
                "maximum_spectral_energy_closure_error": spectral_closure,
                "native_callback_counts": callback_counts,
                "qblade_library_sha256": summary["reproducibility"]["qblade"][
                    "library"
                ]["sha256"],
                "vorlap_runtime_fingerprint": summary["reproducibility"][
                    "vorlap_python_runtime_fingerprint"
                ],
            }
        )
    pairs = []
    for speed in SPEEDS_MPS:
        native = next(
            row for row in rows if row["speed_mps"] == speed and row["role"] == "native"
        )
        complete = next(
            row
            for row in rows
            if row["speed_mps"] == speed and row["role"] == "complete"
        )
        pair = {
            "speed_mps": speed,
            "target_amplitude_m": complete["target_amplitude_m"],
        }
        if native["status"] == complete["status"] == "complete":
            native_time, native_tip = series[native["case_id"]]
            complete_time, complete_tip = series[complete["case_id"]]
            if not np.array_equal(native_time, complete_time):
                raise ValueError(f"Paired time grids differ at U={speed:g} m/s.")
            tail = native_time >= max(
                float(native_time[-1]) - min(6.0, 0.5 * float(native_time[-1])),
                0.0,
            )
            paired_tip = complete_tip - native_tip
            paired_amplitude = 0.5 * (
                float(np.max(paired_tip[tail])) - float(np.min(paired_tip[tail]))
            )
            pair.update(
                native_amplitude_m=native["terminal_peak_amplitude_m"],
                complete_amplitude_m=complete["terminal_peak_amplitude_m"],
                difference_of_amplitudes_m=complete["terminal_peak_amplitude_m"]
                - native["terminal_peak_amplitude_m"],
                paired_increment_amplitude_m=paired_amplitude,
                relative_target_error=(
                    complete["terminal_peak_amplitude_m"]
                    - complete["target_amplitude_m"]
                )
                / max(complete["target_amplitude_m"], 1.0e-12),
            )
        pairs.append(pair)
    report = {
        "schema": "vorlap-iea10-actual-aero-complete-analysis-v1",
        "status": (
            "complete_integration_screen"
            if len(series) == len(manifest["cases"])
            else "incomplete"
        ),
        "campaign_fingerprint": manifest["fingerprint"],
        "campaign_manifest_sha256": sha256_file(root / "campaign_manifest.json"),
        "completed_cases": sum(row["status"] == "complete" for row in rows),
        "cases": rows,
        "pairs": pairs,
        "claim_boundary": (
            "Four 12 s numerical integrations at frozen training-table rows verify "
            "native TurbSim aero plus residual shedding plus signed lock-in feedback. "
            "They are not independent physical validation, stationary LCO estimates, "
            "or fatigue qualification."
        ),
    }
    path = root / "analysis.json" if output is None else Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def promote(campaign: Path, output: Path) -> Path:
    """Write the compact, tracked evidence needed to reproduce the paper figure."""
    root = Path(campaign).resolve()
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    analysis_path = analyze(root, destination / "analysis.json")
    manifest_source = root / "campaign_manifest.json"
    shutil.copy2(manifest_source, destination / "campaign_manifest.json")

    selected = "U14_complete"
    case_root = root / "cases" / selected
    model_root = case_root / base.CASE_NAME
    geometry_source = case_root / "iea10_blade_geometry.csv"
    audit_source = model_root / "Control/vorlap_qblade_geometry_audit.csv"
    shutil.copy2(geometry_source, destination / "iea10_blade_geometry.csv")
    shutil.copy2(audit_source, destination / "qblade_geometry_audit.csv")
    aero_source = model_root / "Aero"
    aero_destination = destination / "qblade_aero"
    if aero_destination.exists():
        shutil.rmtree(aero_destination)
    shutil.copytree(aero_source, aero_destination)
    if (
        len(list((aero_destination / "Airfoils").glob("*.afl"))) != 30
        or len(list((aero_destination / "Polars").glob("*.plr"))) != 30
        or len(list(aero_destination.glob("*.bld"))) != 1
    ):
        raise ValueError("Promoted QBlade aero assets are incomplete.")

    paired: dict[str, np.ndarray] = {}
    for role in ("native", "complete"):
        result = root / "results" / f"U14_{role}" / f"U14_{role}.txt"
        columns = read_qblade_result_numeric_columns(result)
        normalized = {normalize_qblade_header(name): name for name in columns}
        paired[f"{role}_time"] = np.asarray(columns[normalized["Time [s]"]])
        paired[f"{role}_tip"] = np.asarray(
            columns[normalized["Y_b Tip Trl.Def. BLD_1 [m]"]]
        )
    if not np.array_equal(paired["native_time"], paired["complete_time"]):
        raise ValueError("Promoted U14 paired time grids differ.")
    with (destination / "u14_paired_tip_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "time_s",
                "native_tip_edgewise_m",
                "complete_tip_edgewise_m",
                "paired_increment_m",
            ]
        )
        for time_s, native, complete in zip(
            paired["native_time"], paired["native_tip"], paired["complete_tip"]
        ):
            writer.writerow(
                [
                    f"{float(time_s):.17g}",
                    f"{float(native):.17g}",
                    f"{float(complete):.17g}",
                    f"{float(complete - native):.17g}",
                ]
            )

    history_source = root / "results/U14_complete/U14_complete_sil_vorlap_history.csv"
    selected_fields = (
        "step_index",
        "time_s",
        "applied_force_dot_structural_velocity_w",
        "response_amplitude_estimate_m",
        "response_target_cycle_mean_power_w",
        "response_engineering_spectral_stretch_gain",
        "response_engineering_current_saturation_factor",
        "response_spectral_maximum_residual_nodal_force_n",
        "response_spectral_maximum_energy_closure_error",
        "generated_max_abs_force_component_n",
    )
    with history_source.open(newline="", encoding="utf-8") as source_stream:
        reader = csv.DictReader(source_stream)
        missing = set(selected_fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"U14 compact history lacks fields: {sorted(missing)}")
        with (destination / "u14_complete_bridge_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as output_stream:
            writer = csv.DictWriter(
                output_stream, fieldnames=selected_fields, lineterminator="\n"
            )
            writer.writeheader()
            for row in reader:
                writer.writerow({field: row[field] for field in selected_fields})

    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "evidence_manifest.json":
            files[path.relative_to(destination).as_posix()] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    evidence = {
        "schema": "vorlap-iea10-actual-aero-complete-evidence-v1",
        "status": "complete_integration_screen",
        "campaign_fingerprint": json.loads(analysis_path.read_text())[
            "campaign_fingerprint"
        ],
        "selected_visual_case": selected,
        "files": files,
        "claim_boundary": (
            "Compact evidence for an executed 12 s training-space integration screen; "
            "not an independent validation, stationary LCO, or fatigue result."
        ),
    }
    path = destination / "evidence_manifest.json"
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--output", required=True, type=Path)
    gen.add_argument("--iea-root", required=True, type=Path)
    gen.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    ana = sub.add_parser("analyze")
    ana.add_argument("--campaign", required=True, type=Path)
    ana.add_argument("--output", type=Path)
    pro = sub.add_parser("promote")
    pro.add_argument("--campaign", required=True, type=Path)
    pro.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        path = generate(args.output, args.iea_root, args.duration)
    elif args.command == "analyze":
        path = analyze(args.campaign, args.output)
    else:
        path = promote(args.campaign, args.output)
    print(path)


if __name__ == "__main__":
    main()
