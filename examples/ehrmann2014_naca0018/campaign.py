#!/usr/bin/env python3
"""Generate, validate, run, and analyze the Ehrmann NACA0018 VIV campaign.

The structural surrogate preserves the published nondimensional quantities
needed by the present lock-in screen: NACA0018 geometry at 90 degrees,
``m*=47.7``, ``m*zeta=0.96--1.44``, nominal ``fn=6 Hz``, chord Reynolds
number, and the reported ``U*=U/(fd*c)`` definition.  It does not claim to reproduce unreported apparatus
details.  Raw QBlade and hybrid cases share byte-identical structural/native
aerodynamic inputs; the raw controller emits exactly zero external force.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.qblade_case_prep import QBladeCasePrepRequest, prepare_qblade_external_case
from vorlap.qblade_airfoil_work_map import AirfoilForceMotionWorkMap
from vorlap.qblade_engineering_lockin import (
    EngineeringLockInCalibration,
    MODEL_TYPE as ENGINEERING_LOCKIN_MODEL_TYPE,
)
from vorlap.qblade_files import read_qblade_keyword, replace_qblade_keyword_text
from vorlap.qblade_run_support import (
    atomic_write_json,
    canonical_fingerprint,
    require_zero_native_callback_evidence,
)
from vorlap.qblade_sil import normalize_qblade_header, read_qblade_result_numeric_columns


SCHEMA = "vorlap-ehrmann2014-naca0018-campaign-v1"
WORK_MAP_SCHEMA = "vorlap-ehrmann2014-naca0018-work-map-campaign-v2"
ENGINEERING_LOCKIN_SCHEMA = (
    "vorlap-ehrmann2014-naca0018-engineering-lockin-campaign-v5"
)
CASE_IDENTITY_SCHEMA = "vorlap-ehrmann2014-native-case-identity-v1"
REUSE_RECEIPT_SCHEMA = "vorlap-ehrmann2014-raw-reuse-receipt-v1"
SCENARIOS = ("raw_qblade", "latest_spectral_lockin_bridge")
WORK_MAP_SCENARIOS = ("raw_qblade", "latest_airfoil_work_map_bridge")
ENGINEERING_LOCKIN_SCENARIOS = (
    "raw_qblade",
    "latest_engineering_lockin_bridge",
)
REDUCED_VELOCITIES = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.8, 6.0, 6.3)
ENGINEERING_REDUCED_VELOCITIES = (
    3.0,
    3.5,
    4.0,
    4.5,
    4.6,
    4.7,
    4.8,
    5.0,
    5.5,
    5.8,
    6.0,
    6.3,
)
TIMESTEP_S = 0.001
HALF_TIMESTEP_S = 0.0005
TIMESTEP_CHECK_USTAR = 5.8
ENGINEERING_TIMESTEP_CHECK_USTAR = 4.7
NATURAL_CYCLES = 120
TAIL_CYCLES = 40
NATURAL_FREQUENCY_HZ = 6.0
CHORD_M = 0.230
SPAN_M = 0.9144
N_STATIONS = 21
AIR_DENSITY_KGPM3 = 1.225
AIR_KINEMATIC_VISCOSITY_M2PS = 1.48e-5
MASS_RATIO = 47.7
MASS_DAMPINGS = (0.96, 1.44)
PUBLISHED_EFFECTIVE_MASS_KGPM = (
    MASS_RATIO * math.pi * AIR_DENSITY_KGPM3 * CHORD_M**2 / 4.0
)
EULER_BERNOULLI_BETA1 = 1.875104068711961
ORTHOGONAL_BENDING_STIFFNESS_RATIO = 100.0
RESULT_CHANNEL = "Y_b Tip Trl.Def. BLD_1 [m]"
CASE_NAME = "ehrmann2014_naca0018"
STRUCTURAL_VERIFICATION_REPORT = HERE / "structural_verification_report.json"
STRUCTURAL_VERIFICATION_REPORT_SHA256 = (
    "adfc2ccd0731387b97eccf1bfe0a6fcc353c5d4fd86e79c32da21ada7feaf83d"
)
TRACKED_STATIONARY_SPECTRUM_SHA256 = (
    "64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369"
)
FROZEN_QBLADE_VERSION = "2.0.9.7"
FROZEN_QBLADE_LIBRARY_SHA256 = (
    "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
)
FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT = (
    "c50c6909317c754caec4130ccbc8e2ed3f705f810f07da11e7ce3ad003499089"
)


class CampaignError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structural_qualification_identity() -> dict[str, object]:
    path = Path(STRUCTURAL_VERIFICATION_REPORT)
    if not path.is_file() or sha256_file(path) != STRUCTURAL_VERIFICATION_REPORT_SHA256:
        raise CampaignError("The tracked structural-verification report identity differs.")
    report = json.loads(path.read_text(encoding="utf-8"))
    expected_design = {
        "EIx_over_EIy": ORTHOGONAL_BENDING_STIFFNESS_RATIO,
        "chord_m": CHORD_M,
        "crossflow_bending_axis": "EIy_global_Y",
        "mass_damping_bracket": list(MASS_DAMPINGS),
        "mass_ratio": MASS_RATIO,
        "natural_frequency_hz": NATURAL_FREQUENCY_HZ,
        "orthogonal_bending_axis": "EIx_global_X",
        "result_channel": RESULT_CHANNEL,
        "span_m": SPAN_M,
        "station_count": N_STATIONS,
    }
    if (
        report.get("schema") != "vorlap-ehrmann2014-structural-verification-v1"
        or report.get("status") != "passed"
        or report.get("qualified_design") != expected_design
    ):
        raise CampaignError("The structural-verification report does not qualify this design.")
    axis = report.get("axis_channel_probe", {})
    if (
        axis.get("passed") is not True
        or axis.get("native_aerodynamics") is not False
        or axis.get("verified_body_tip_channel") != RESULT_CHANNEL
        or float(axis.get("global_y_paired_rms_m", 0.0)) <= 1.0e-8
    ):
        raise CampaignError("The structural-verification axis/channel evidence differs.")
    decays = report.get("free_decay")
    if not isinstance(decays, list) or len(decays) != len(MASS_DAMPINGS):
        raise CampaignError("The structural-verification damping bracket is incomplete.")
    for row, mass_damping in zip(decays, MASS_DAMPINGS):
        target = mass_damping / MASS_RATIO
        if (
            float(row.get("mass_damping", math.nan)) != mass_damping
            or row.get("frequency_pass") is not True
            or row.get("damping_pass") is not True
            or abs(float(row.get("dominant_frequency_hz", math.nan)) - NATURAL_FREQUENCY_HZ)
            / NATURAL_FREQUENCY_HZ > 0.05
            or abs(float(row.get("estimated_damping_ratio", math.nan)) - target) > 0.01
        ):
            raise CampaignError("The structural-verification frequency/damping evidence differs.")
    mass = report.get("generalized_mass_check", {})
    target_mass = PUBLISHED_EFFECTIVE_MASS_KGPM * SPAN_M
    if (
        abs(float(mass.get("generalized_mass_kg", math.nan)) - target_mass) > 1.0e-12
        or abs(float(mass.get("target_generalized_mass_kg", math.nan)) - target_mass) > 1.0e-12
        or float(mass.get("implied_mass_ratio", math.nan)) != MASS_RATIO
    ):
        raise CampaignError("The structural-verification generalized-mass evidence differs.")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": STRUCTURAL_VERIFICATION_REPORT_SHA256,
        "status": "passed",
        "verified_result_channel": RESULT_CHANNEL,
    }


def _identity(path: Path, root: Path) -> dict[str, object]:
    path = Path(path).resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _patch(text: str, values: Mapping[str, object]) -> str:
    for keyword, value in values.items():
        text, _old, count = replace_qblade_keyword_text(text, keyword, value)
        if count != 1:
            raise CampaignError(f"Could not patch required QBlade keyword {keyword}.")
    return text


def _tag(value: float) -> str:
    return f"u{value:.2f}".replace(".", "p")


def _first_mode(span_fraction: np.ndarray) -> np.ndarray:
    beta = EULER_BERNOULLI_BETA1
    sigma = (math.cosh(beta) + math.cos(beta)) / (math.sinh(beta) + math.sin(beta))
    mode = (
        np.cosh(beta * span_fraction)
        - np.cos(beta * span_fraction)
        - sigma * (np.sinh(beta * span_fraction) - np.sin(beta * span_fraction))
    )
    return mode / mode[-1]


def _write_blade(path: Path, span: np.ndarray) -> None:
    lines = [
        "----------------------------------------QBlade Blade Definition File------------------------------------------------",
        "Generated by examples/ehrmann2014_naca0018/campaign.py",
        f"{CASE_NAME}_beam OBJECTNAME",
        "HAWT ROTORTYPE - isolated stationary one-blade section surrogate",
        "1 NUMBLADES",
        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE",
    ]
    lines.extend(
        f"{height:.10e} {CHORD_M:.10e} 0.0 0.0 0.0 0.5 Polars/NACA_0018_RFoil_MultiRePolar.plr"
        for height in span
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _surrogate_distributed_mass_kgpm(fractions: np.ndarray) -> float:
    mode = _first_mode(fractions)
    span = fractions * SPAN_M
    modal_shape_integral_m = float(np.trapezoid(mode**2, span))
    if not math.isfinite(modal_shape_integral_m) or modal_shape_integral_m <= 0.0:
        raise CampaignError("Cantilever modal-shape integral is invalid.")
    target_generalized_mass_kg = PUBLISHED_EFFECTIVE_MASS_KGPM * SPAN_M
    return target_generalized_mass_kg / modal_shape_integral_m


def _target_ei(distributed_mass_kgpm: float) -> float:
    omega = 2.0 * math.pi * NATURAL_FREQUENCY_HZ
    return distributed_mass_kgpm * (
        omega * SPAN_M**2 / EULER_BERNOULLI_BETA1**2
    ) ** 2


def _write_beam(path: Path, fractions: np.ndarray, damping_ratio: float) -> None:
    beta = damping_ratio / (math.pi * NATURAL_FREQUENCY_HZ)
    distributed_mass = _surrogate_distributed_mass_kgpm(fractions)
    header = [
        f"{beta:.12e} RAYLEIGHDMP - beta=zeta/(pi fn) [s]",
        "1.0 STIFFTUNER",
        "1.0 MASSTUNER",
        "0 INTPTYPE",
        "1 BEAMTYPE",
        "2 DISCTYPE",
        f"{len(fractions)} DISC",
        "LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] EA_[N] GJ_[N.m^2] GA_[N] STRPIT_[deg] KSX_[-] KSY_[-] RGX_[-] RGY_[-] XCM_[-] YCM_[-] XCE_[-] YCE_[-] XCS_[-] YCS_[-]",
    ]
    rows = []
    for value in fractions:
        data = (
            value,
            distributed_mass,
            ORTHOGONAL_BENDING_STIFFNESS_RATIO * _target_ei(distributed_mass),
            _target_ei(distributed_mass),
            2.0e8,
            9.0e4,
            7.0e7,
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
        rows.append(" ".join(f"{item:.12e}" for item in data))
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def _write_rigid_tower(path: Path) -> None:
    section = "1.0 1.0e15 1.0e15 1.0e15 1.0e15 1.0e15 0.0 0.85 0.85 0.25 0.25 0.0 0.0 0.0 0.0 0.0 0.0 1.0"
    lines = [
        "0.0 RAYLEIGHDMP", "1.0 STIFFTUNER", "1.0 MASSTUNER", "0 INTPTYPE",
        "1 BEAMTYPE", "0 DISCTYPE", "2 DISC",
        "LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] EA_[N] GJ_[N.m^2] GA_[N] STRPIT_[deg] KSX_[-] KSY_[-] RGX_[-] RGY_[-] XCM_[-] YCM_[-] XCE_[-] YCE_[-] XCS_[-] YCS_[-] Diameter_[m]",
        f"0.0 {section}", f"1.0 {section}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_main_structure(path: Path, fractions: np.ndarray) -> None:
    lines = [
        "---------------------- QBLADE STRUCTURAL MODEL INPUT FILE -----------------",
        "0.0005 GLBGEOEPS", "0.0 PRECONE", "0.0 SHFTTILT", "0.0 OVERHANG", "0.0 TWR2SHFT",
        "0.0 YAWBRMASS", "0.0 NACMASS", "0.0 NACCMX", "0.0 NACCMY", "0.0 NACCMZ",
        "0 0 0 0 0 0 NACINER", "0.0 NACYINER", "0.0 HUBMASS", "0.0 HUBCM", "1.0 HUBINER",
        "1.0 GBRATIO", "1.0 GBOXEFF", "1.0 GENEFF", "false DRTRDOF", "1.0 GENINER",
        "1.0e12 DTTORSPR", "1.0e9 DTTORDMP", "1 NUMBLD", "ehrmann_beam.str BLDFILE_1",
        "1.0 TWRHEIGHT", "rigid_tower.str TWRFILE",
        "true FOR_OUT", "true ROT_OUT", "true MOM_OUT", "true DEF_OUT", "true POS_OUT",
        "true VEL_OUT", "true ACC_OUT", "true LVE_OUT", "true LAC_OUT",
    ]
    lines.extend(f"BLD_1_{value:.10f}" for value in fractions)
    lines.extend(("TWR_0.0", "TWR_1.0"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_turbine(path: Path, template: Path, object_name: str) -> None:
    text = _patch(
        template.read_text(encoding="utf-8"),
        {
            "OBJECTNAME": object_name,
            "BLADEFILE": f"Aero/{CASE_NAME}.bld",
            "TURBTYPE": 0,
            "NUMBLADES": 1,
            "ROTORCONFIG": 0,
            "OVERHANG": 0.0,
            "SHAFTTILT": 0.0,
            "ROTORCONE": 0.0,
            "TOWERHEIGHT": 1.0,
            "TOWERTOPRAD": 0.5,
            "TOWERBOTRAD": 0.5,
            "WAKETYPE": 1,
            "TOWERSHADOW": "false",
            "STRUCTURALFILE": "Structure/ehrmann_main.str",
            "CONTROLLERTYPE": 0,
        },
    )
    path.write_text(text, encoding="utf-8")


def _write_sim(path: Path, template: Path, case: Mapping[str, object]) -> None:
    text = _patch(
        template.read_text(encoding="utf-8"),
        {
            "OBJECTNAME": case["case_id"],
            "TURBFILE": f"model/{CASE_NAME}.trb",
            "TURBNAME": case["case_id"],
            "INITIAL_YAW": 0.0,
            "INITIAL_PITCH": 0.0,
            "INITIAL_AZIMUTH": 0.0,
            "STRSUBSTEP": 2,
            "RELAXSTEPS": 0,
            "PRESCRIBETYPE": 1,
            "RPMPRESCRIBED": 0.0,
            "INCLUDEAERO": 1,
            "INCLUDEHYDRO": 0,
            "TIMESTEP": case["timestep_s"],
            "NUMTIMESTEPS": case["num_steps"],
            "RAMPUP": 0.0,
            "ADDDAMP": 0.0,
            "WNDTYPE": 0,
            "WNDNAME": "",
            "MEANINF": case["wind_speed_mps"],
            "HORANGLE": 0.0,
            "VERTANGLE": 0.0,
            "PROFILETYPE": 0,
            "SHEAREXP": 0.0,
            "DENSITYAIR": AIR_DENSITY_KGPM3,
            "VISCOSITYAIR": AIR_KINEMATIC_VISCOSITY_M2PS,
            "STOREFROM": (NATURAL_CYCLES - TAIL_CYCLES) / NATURAL_FREQUENCY_HZ,
            "STOREREPLAY": "false",
            "STOREAERO": "true",
            "STOREBLADE": "true",
            "STORESTRUCT": "true",
            "STORESIM": "true",
            "STOREHYDRO": "false",
            "STORECONTROLLER": "false",
            "CALCMODAL": "false",
        },
    )
    path.write_text(text, encoding="utf-8")


def build_cases(
    scenarios: Sequence[str] = SCENARIOS,
) -> list[dict[str, object]]:
    scenario_values = tuple(str(value) for value in scenarios)
    if scenario_values not in {
        SCENARIOS,
        WORK_MAP_SCENARIOS,
        ENGINEERING_LOCKIN_SCENARIOS,
    }:
        raise CampaignError("Unsupported Ehrmann comparison-series contract.")
    timestep_check_ustar = (
        ENGINEERING_TIMESTEP_CHECK_USTAR
        if scenario_values == ENGINEERING_LOCKIN_SCENARIOS
        else TIMESTEP_CHECK_USTAR
    )
    reduced_velocities = (
        ENGINEERING_REDUCED_VELOCITIES
        if scenario_values == ENGINEERING_LOCKIN_SCENARIOS
        else REDUCED_VELOCITIES
    )
    matrix = [
        (mass_damping, u_star, TIMESTEP_S)
        for mass_damping in MASS_DAMPINGS
        for u_star in reduced_velocities
    ]
    matrix.extend(
        (mass_damping, timestep_check_ustar, HALF_TIMESTEP_S)
        for mass_damping in MASS_DAMPINGS
    )
    rows: list[dict[str, object]] = []
    duration = NATURAL_CYCLES / NATURAL_FREQUENCY_HZ
    for mass_damping, u_star, timestep in matrix:
        damping_ratio = mass_damping / MASS_RATIO
        damped_frequency = NATURAL_FREQUENCY_HZ * math.sqrt(1.0 - damping_ratio**2)
        speed = u_star * damped_frequency * CHORD_M
        reynolds = speed * CHORD_M / AIR_KINEMATIC_VISCOSITY_M2PS
        physical_id = (
            f"md{mass_damping:.2f}_{_tag(u_star)}_dt{int(round(timestep*1e6)):04d}us"
        ).replace(".", "p")
        for scenario in scenario_values:
            case_id = f"{physical_id}_{'raw' if scenario == SCENARIOS[0] else 'hybrid'}"
            rows.append(
                {
                    "index": len(rows),
                    "case_id": case_id,
                    "scenario": scenario,
                    "native_physical_id": physical_id,
                    "mass_damping": mass_damping,
                    "structural_damping_ratio": damping_ratio,
                    "damped_natural_frequency_hz": damped_frequency,
                    "reduced_velocity": u_star,
                    "wind_speed_mps": speed,
                    "reynolds_number": reynolds,
                    "timestep_s": timestep,
                    "duration_s": duration,
                    "num_steps": int(round(duration / timestep)),
                    "case_dir": f"cases/{case_id}",
                    "sim": f"cases/{case_id}/{case_id}.sim",
                    "result_dir": f"results/{case_id}",
                    "result_name": f"{case_id}_qblade_results",
                }
            )
    return rows


def _response_config(
    *,
    node_ids: Sequence[str],
    mode_shape: np.ndarray,
    weights: np.ndarray,
    damping_ratio: float,
    hybrid: bool,
    work_map_path: bool = False,
    engineering_lockin_path: bool = False,
) -> dict[str, object]:
    return {
        "type": "modal_negative_damping",
        "schema_version": 2,
        "aerodynamic_speed_basis": "declared_ambient_crossflow",
        "node_ids": list(node_ids),
        "mode_shape": [float(value) for value in mode_shape],
        "natural_frequency_hz": NATURAL_FREQUENCY_HZ,
        "reference_length_m": CHORD_M,
        "reduced_velocity_low": 2.0,
        "reduced_velocity_peak": 5.8,
        "reduced_velocity_high": 6.8,
        "aerodynamic_damping_gain_h1": (
            0.0
            if (work_map_path or engineering_lockin_path)
            else (3.920125762951649 if hybrid else 0.0)
        ),
        "saturation_amplitude_ratio": 0.2968375879119047,
        "rms_time_constant_s": 2.0 / NATURAL_FREQUENCY_HZ,
        "max_abs_gain": 10.0,
        "max_abs_nodal_force_n": 500.0,
        "force_direction": "chord",
        "projection_weights": [float(value) for value in weights],
        "bandpass_damping_ratio": 0.1,
        "enabled": True,
        "startup_seed": {
            "schema_version": 1,
            "enabled": hybrid,
            "duration_cycles": 5.0,
            "target_amplitude_ratio": 0.005,
            "structural_damping_ratio": damping_ratio,
            "taper": "hann_sine_squared",
            "carrier": "natural_frequency",
            "nodal_weighting": "projection_mass_times_mode",
            "nodal_force_limit_source": "response_model_max_abs_nodal_force_n",
        },
    }


def _spectral_adapter_config(damping_ratio: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "type": "single_mode_airfoil_spectral_lockin_hybrid",
        "structural_damping_ratio": damping_ratio,
        "spectral_lockin": {
            "schema_version": 1,
            "source_channel": "CL",
            "reference_strouhal": 0.2,
            "reference_lift_amplitude": 0.3,
            "frequency_ratio_low": 1.07250616424331,
            "frequency_ratio_peak": 1.249165,
            "frequency_ratio_high": 1.3965603702125565,
            "maximum_spectral_gain": 10.0,
            "minimum_modal_participation": 1e-12,
            "retain_uncaptured_lift_spectrum": True,
            "retain_fluctuating_drag_spectrum": True,
            "include_mean_load": False,
            "residual_source_scale": 0.0,
        },
    }


def _work_map_adapter_config(
    damping_ratio: float,
    *,
    map_name: str,
    map_sha256: str,
    stationary_spectrum_sha256: str,
    enabled: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "type": "single_mode_airfoil_force_motion_work_map_v1",
        "profile": "NACA0018",
        "kinematics": "fixed_nonrotating",
        "work_map": {
            "path": map_name,
            "sha256": map_sha256,
            "confidence_multiplier": 1.96,
            "require_nonpositive_high_amplitude_work": True,
        },
        "static_spectrum": {
            "channel": "CL",
            "force_axis": "wind_axis_lift",
            "amplitude_measure": "rss_nonzero_frequency_harmonics",
            "source_frequency_measure": "energy_centroid",
            "bounds_policy": "raise",
            "source_sha256": stationary_spectrum_sha256,
        },
        "modal_power": {
            "rms_time_constant_s": 2.0 / NATURAL_FREQUENCY_HZ,
            "bandpass_damping_ratio": 0.1,
            "max_abs_target_power_w": 1000.0,
            "max_abs_generalized_force_n": 1000.0,
            "max_abs_nodal_force_n": 500.0,
            "limit_action": "error",
            "enabled": bool(enabled),
        },
        "structural_damping_ratio": damping_ratio,
        "residual_source_scale": 0.0,
    }


def _engineering_lockin_adapter_config(
    damping_ratio: float,
    *,
    calibration_name: str,
    calibration_sha256: str,
    enabled: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "type": ENGINEERING_LOCKIN_MODEL_TYPE,
        "profile": "NACA0018",
        "kinematics": "fixed_nonrotating",
        "calibration": {
            "path": calibration_name,
            "sha256": calibration_sha256,
        },
        "static_spectrum": {
            "channel": "CL",
            "bounds_policy": "raise",
            "source_sha256": TRACKED_STATIONARY_SPECTRUM_SHA256,
            "n_frequency_entries": 200,
        },
        "modal_power": {
            "rms_time_constant_s": 2.0 / NATURAL_FREQUENCY_HZ,
            "bandpass_damping_ratio": 0.1,
            "max_abs_target_power_w": 1000.0,
            "max_abs_generalized_force_n": 1000.0,
            "max_abs_nodal_force_n": 500.0,
            "limit_action": "error",
            "enabled": bool(enabled),
        },
        "structural_damping_ratio": damping_ratio,
        "residual_source_scale": 0.0,
    }


def _native_identity(root: Path, case: Mapping[str, object]) -> dict[str, object]:
    case_root = root / str(case["case_dir"])
    paths = [
        case_root / str(case["case_id"] + ".sim"),
        case_root / "model" / f"{CASE_NAME}.trb",
        case_root / "model" / "Aero" / f"{CASE_NAME}.bld",
        case_root / "model" / "Aero" / "Airfoils" / "NACA_0018_RFoil.afl",
        case_root / "model" / "Aero" / "Polars" / "NACA_0018_RFoil_MultiRePolar.plr",
        case_root / "model" / "Structure" / "ehrmann_beam.str",
        case_root / "model" / "Structure" / "ehrmann_main.str",
        case_root / "model" / "Structure" / "rigid_tower.str",
    ]
    files = []
    for path in paths:
        if path.suffix == ".sim":
            # Controller table paths are intentionally excluded; the QBlade
            # native identity is instead bound by the frozen physical fields.
            continue
        files.append(_identity(path, case_root))
    payload = {
        "schema": CASE_IDENTITY_SCHEMA,
        "geometry_structure_native_aero_files": files,
        "physical_inputs": {
            "profile": "NACA0018",
            "incidence_deg": 90.0,
            "natural_frequency_hz": NATURAL_FREQUENCY_HZ,
            "damped_natural_frequency_hz": case["damped_natural_frequency_hz"],
            "mass_ratio": MASS_RATIO,
            "structural_damping_ratio": case["structural_damping_ratio"],
            "reduced_velocity": case["reduced_velocity"],
            "reduced_velocity_definition": "Ustar=U/(fd*c)",
            "wind_speed_mps": case["wind_speed_mps"],
            "reynolds_number": case["reynolds_number"],
            "timestep_s": case["timestep_s"],
            "duration_s": case["duration_s"],
            "num_steps": case["num_steps"],
            "native_qblade_aerodynamics": True,
        },
    }
    payload["fingerprint"] = canonical_fingerprint(payload)
    return payload


def _reuse_raw_result(
    *, campaign_root: Path, case: Mapping[str, object], reuse_root: Path
) -> dict[str, object] | None:
    if case["scenario"] != SCENARIOS[0]:
        return None
    candidate = Path(reuse_root) / str(case["case_id"])
    identity_path = candidate / "native_case_identity.json"
    summary_name = f"{Path(str(case['sim'])).stem}_run_summary.json"
    summary_path = candidate / summary_name
    if not identity_path.is_file() and not summary_path.is_file():
        return None
    if not identity_path.is_file() or not summary_path.is_file():
        raise CampaignError(f"Incomplete raw reuse evidence: {candidate}")
    expected = _native_identity(campaign_root, case)
    observed = json.loads(identity_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise CampaignError(
            f"Raw result identity mismatch for {case['case_id']}; refusing reuse from {candidate}."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        bool(summary.get("failed", True))
        or bool(summary.get("stopped", True))
        or int(summary.get("completed_steps", -1)) != int(case["num_steps"])
        or int(summary.get("requested_steps", -1)) != int(case["num_steps"])
    ):
        raise CampaignError(
            f"Raw result is not a complete production run for {case['case_id']}; "
            f"refusing reuse from {candidate}."
        )
    reproducibility = summary.get("reproducibility")
    qblade_identity = (
        reproducibility.get("qblade", {}).get("library", {})
        if isinstance(reproducibility, Mapping)
        else {}
    )
    dependency_fingerprint = (
        reproducibility.get("qblade_runtime_dependency_fingerprint")
        if isinstance(reproducibility, Mapping)
        else None
    )
    external_force = summary.get("sil_vorlap_max_abs_force")
    raw_callback = summary.get("native_callback_evidence")
    callback_count_keys = (
        "update_init_count",
        "update_log_entry_count",
        "update_cache_observation_count",
        "update_count_observed_max",
    )
    if (
        summary.get("coupling_mode") != "sil-vorlap"
        or summary.get("sil_vorlap") is not True
        or type(external_force) not in (int, float)
        or not math.isfinite(float(external_force))
        or float(external_force) != 0.0
        or not isinstance(raw_callback, Mapping)
        or any(type(raw_callback.get(key)) is not int for key in callback_count_keys)
    ):
        raise CampaignError(
            f"Raw result is not a zero-supplemental-force SIL observer run for "
            f"{case['case_id']}; refusing reuse from {candidate}."
        )
    try:
        require_zero_native_callback_evidence(
            dict(raw_callback),
            context=f"Raw reuse {case['case_id']}",
        )
    except RuntimeError as exc:
        raise CampaignError(
            f"Raw result has native callback activity for {case['case_id']}; "
            f"refusing reuse from {candidate}."
        ) from exc
    if (
        not isinstance(qblade_identity, Mapping)
        or qblade_identity.get("sha256") != FROZEN_QBLADE_LIBRARY_SHA256
        or dependency_fingerprint
        != FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
    ):
        raise CampaignError(
            f"Raw result QBlade runtime identity differs for {case['case_id']}; "
            f"refusing reuse from {candidate}."
        )
    destination = campaign_root / str(case["result_dir"])
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(candidate, destination)
    receipt = {
        "schema": REUSE_RECEIPT_SCHEMA,
        "case_id": case["case_id"],
        "native_case_identity_fingerprint": expected["fingerprint"],
        "source_result": str(candidate.resolve()),
        "copied_result": str(destination.relative_to(campaign_root)),
        "run_summary_sha256": sha256_file(destination / summary_name),
        "qblade_library_sha256": qblade_identity["sha256"],
        "qblade_runtime_dependency_fingerprint": dependency_fingerprint,
        "sil_vorlap_max_abs_force": float(external_force),
        "coupling_mode": "sil-vorlap",
        "native_callback_update_count": 0,
    }
    atomic_write_json(destination / "reuse_receipt.json", receipt)
    return receipt


def generate(
    output: Path,
    *,
    runtime_vorlap_root: Path,
    raw_reuse_root: Path | None = None,
    work_map: Path | None = None,
    engineering_calibration: Path | None = None,
) -> Path:
    structural_qualification = _structural_qualification_identity()
    if work_map is not None and engineering_calibration is not None:
        raise CampaignError("work_map and engineering_calibration are mutually exclusive.")
    work_map_path = (
        None if work_map is None else Path(work_map).expanduser().resolve()
    )
    work_map_model = None
    work_map_bytes = None
    work_map_sha256 = None
    work_map_stationary_spectrum_sha256 = None
    engineering_path = (
        None
        if engineering_calibration is None
        else Path(engineering_calibration).expanduser().resolve()
    )
    engineering_model = None
    engineering_bytes = None
    engineering_sha256 = None
    scenarios = SCENARIOS
    schema = SCHEMA
    if work_map_path is not None:
        if work_map_path.is_symlink() or not work_map_path.is_file():
            raise CampaignError("work_map must be a regular non-symlink JSON file.")
        work_map_bytes = work_map_path.read_bytes()
        work_map_sha256 = hashlib.sha256(work_map_bytes).hexdigest()
        work_map_model = AirfoilForceMotionWorkMap.from_json(work_map_path)
        if work_map_model.profile != "NACA0018":
            raise CampaignError("The Ehrmann work map must identify NACA0018.")
        stationary_sources = tuple(
            source
            for source in work_map_model.sources
            if source["role"] == "stationary_force_spectrum"
        )
        if len(stationary_sources) != 1:
            raise CampaignError(
                "The Ehrmann work map must bind exactly one stationary-force spectrum."
            )
        work_map_stationary_spectrum_sha256 = stationary_sources[0]["sha256"]
        if (
            work_map_stationary_spectrum_sha256
            != TRACKED_STATIONARY_SPECTRUM_SHA256
        ):
            raise CampaignError(
                "The Ehrmann work map does not bind the tracked NACA0018.h5 source."
            )
        domain = work_map_model.positive_work_domain_status(
            confidence_multiplier=1.96
        )
        if domain != {
            "has_positive_work": True,
            "highest_amplitude_plane_nonpositive": True,
        }:
            raise CampaignError(
                "The Ehrmann work map must contain a bounded positive lower-confidence branch."
            )
        scenarios = WORK_MAP_SCENARIOS
        schema = WORK_MAP_SCHEMA
    elif engineering_path is not None:
        if engineering_path.is_symlink() or not engineering_path.is_file():
            raise CampaignError(
                "engineering_calibration must be a regular non-symlink JSON file."
            )
        engineering_bytes = engineering_path.read_bytes()
        engineering_sha256 = hashlib.sha256(engineering_bytes).hexdigest()
        engineering_model = EngineeringLockInCalibration.from_json(engineering_path)
        if engineering_model.raw["profile"] != "NACA0018":
            raise CampaignError("The engineering calibration must identify NACA0018.")
        if (
            engineering_model.raw["sources"]["stationary_spectrum"]["sha256"]
            != TRACKED_STATIONARY_SPECTRUM_SHA256
        ):
            raise CampaignError(
                "The engineering calibration does not bind the tracked NACA0018 spectrum."
            )
        scenarios = ENGINEERING_LOCKIN_SCENARIOS
        schema = ENGINEERING_LOCKIN_SCHEMA
    source_h5 = (
        REPO_ROOT
        / "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/VorLapAirfoils/NACA0018.h5"
    )
    if (work_map_path is not None or engineering_path is not None) and sha256_file(
        source_h5
    ) != (
        work_map_stationary_spectrum_sha256
        if work_map_path is not None
        else TRACKED_STATIONARY_SPECTRUM_SHA256
    ):
        raise CampaignError(
            "The staged runtime NACA0018 spectrum source differs from the work map."
        )
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing campaign: {output}")
    (output / "cases").mkdir(parents=True)
    (output / "results").mkdir()
    span = np.linspace(0.0, SPAN_M, N_STATIONS)
    fractions = span / SPAN_M
    mode = _first_mode(fractions)
    tributary = np.full(N_STATIONS, SPAN_M / (N_STATIONS - 1))
    tributary[[0, -1]] *= 0.5
    distributed_mass = _surrogate_distributed_mass_kgpm(fractions)
    weights = distributed_mass * tributary
    node_ids = tuple(f"BLD_1_{value:.6f}" for value in fractions)
    sim_template = REPO_ROOT / "examples/ReferenceHVAWTQBladeModel/baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
    trb_template = REPO_ROOT / "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/10m_HVAWT_reference_turbine.trb"
    reference = REPO_ROOT / "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine"
    afl = reference / "Aero/Airfoils/NACA_0018_RFoil.afl"
    polar = reference / "Aero/Polars/NACA_0018_RFoil_MultiRePolar.plr"
    rows = build_cases(scenarios)
    for case in rows:
        case_root = output / str(case["case_dir"])
        model_root = case_root / "model"
        aero = model_root / "Aero"
        structure = model_root / "Structure"
        source = model_root / "VorLapAirfoils"
        for directory in (aero / "Airfoils", aero / "Polars", structure, source):
            directory.mkdir(parents=True, exist_ok=True)
        _write_blade(aero / f"{CASE_NAME}.bld", span)
        shutil.copy2(afl, aero / "Airfoils" / afl.name)
        shutil.copy2(polar, aero / "Polars" / polar.name)
        _write_beam(
            structure / "ehrmann_beam.str",
            fractions,
            float(case["structural_damping_ratio"]),
        )
        _write_rigid_tower(structure / "rigid_tower.str")
        _write_main_structure(structure / "ehrmann_main.str", fractions)
        shutil.copy2(source_h5, source / "NACA0018.h5")
        _write_turbine(
            model_root / f"{CASE_NAME}.trb",
            trb_template,
            str(case["native_physical_id"]),
        )
        sim = case_root / f"{case['case_id']}.sim"
        _write_sim(sim, sim_template, case)
        hybrid = case["scenario"] == scenarios[1]
        prepared = prepare_qblade_external_case(
            QBladeCasePrepRequest(
                sim_path=sim,
                airfoil_dir=source,
                include_struts=False,
                include_tower=False,
                node_source="structural",
                n_freq_depth=200,
                force_scale=0.0,
                include_mean_load=False,
                source_bounds_policy="raise",
                source_frequency_cap_hz=0.5 / float(case["timestep_s"]),
                response_model=_response_config(
                    node_ids=node_ids,
                    mode_shape=mode,
                    weights=weights,
                    damping_ratio=float(case["structural_damping_ratio"]),
                    hybrid=hybrid,
                    work_map_path=work_map_path is not None,
                    engineering_lockin_path=engineering_path is not None,
                ),
                inflow_vec=(1.0, 0.0, 0.0),
                include_default_python_path=True,
                default_python_path=Path(runtime_vorlap_root).expanduser().resolve(),
                absolute_source_parameter_dir=True,
            )
        )
        config = json.loads(prepared.parameter_file_path.read_text(encoding="utf-8"))
        if work_map_path is None and engineering_path is None:
            config["single_mode_spectral_lockin"] = _spectral_adapter_config(
                float(case["structural_damping_ratio"])
            )
        elif work_map_path is not None:
            if work_map_bytes is None or work_map_sha256 is None:
                raise RuntimeError("Validated work-map bytes are unavailable.")
            staged_map = prepared.parameter_file_path.parent / "naca0018_work_map.json"
            staged_map.write_bytes(work_map_bytes)
            config["single_mode_airfoil_work_map"] = _work_map_adapter_config(
                float(case["structural_damping_ratio"]),
                map_name=staged_map.name,
                map_sha256=work_map_sha256,
                stationary_spectrum_sha256=work_map_stationary_spectrum_sha256,
                enabled=hybrid,
            )
        else:
            if engineering_bytes is None or engineering_sha256 is None:
                raise RuntimeError("Validated engineering calibration bytes are unavailable.")
            staged_calibration = (
                prepared.parameter_file_path.parent
                / "engineering_lockin_calibration.json"
            )
            staged_calibration.write_bytes(engineering_bytes)
            config["single_mode_engineering_lockin"] = (
                _engineering_lockin_adapter_config(
                    float(case["structural_damping_ratio"]),
                    calibration_name=staged_calibration.name,
                    calibration_sha256=engineering_sha256,
                    enabled=hybrid,
                )
            )
        atomic_write_json(prepared.parameter_file_path, config)
        case["runtime_config"] = prepared.parameter_file_path.relative_to(output).as_posix()
        case["native_case_identity"] = _native_identity(output, case)
        case["hybrid_adapter_status"] = (
            "generic_one_mode_spectral_adapter_enabled"
            if work_map_path is None and engineering_path is None
            else (
                "single_mode_airfoil_force_motion_work_map_enabled"
                if work_map_path is not None
                else "single_mode_airfoil_engineering_lockin_enabled"
            )
        )
        if work_map_path is None and engineering_path is None:
            case["external_force_policy"] = (
                "exact_zero_h1_no_seed_no_residual"
                if not hybrid
                else "captured_CL_energy_stretched_to_single_mode_no_residual"
            )
        elif work_map_path is not None:
            case["external_force_policy"] = (
                "exact_zero_modal_power_no_seed_no_residual"
                if not hybrid
                else "positive_lower_confidence_incremental_work_map_no_residual"
            )
        else:
            case["external_force_policy"] = (
                "exact_zero_engineering_modal_power_no_seed_no_residual"
                if not hybrid
                else "static_spectrum_conditioned_negative_damping_with_modal_saturation"
            )
        if raw_reuse_root is not None:
            receipt = _reuse_raw_result(
                campaign_root=output, case=case, reuse_root=raw_reuse_root
            )
            case["raw_reuse_status"] = "reused_exact_identity" if receipt else "no_candidate_run_required"
        elif not hybrid:
            case["raw_reuse_status"] = "not_requested_run_required"

    manifest: dict[str, object] = {
        "schema": schema,
        "status": "ready_for_native_linux_x86_submission",
        "case_count": len(rows),
        "comparison_series": list(scenarios),
        "hybrid_adapter_status": (
            "generic_one_mode_spectral_adapter_enabled"
            if work_map_path is None and engineering_path is None
            else (
                "single_mode_airfoil_force_motion_work_map_enabled"
                if work_map_path is not None
                else "single_mode_airfoil_engineering_lockin_enabled"
            )
        ),
        "structural_qualification": structural_qualification,
        "dynamic_similarity": {
            "profile": "NACA0018",
            "incidence_deg": 90.0,
            "natural_frequency_hz": NATURAL_FREQUENCY_HZ,
            "chord_m": CHORD_M,
            "span_m": SPAN_M,
            "mass_ratio": MASS_RATIO,
            "published_effective_mass_per_span_kgpm": PUBLISHED_EFFECTIVE_MASS_KGPM,
            "target_generalized_modal_mass_kg": PUBLISHED_EFFECTIVE_MASS_KGPM * SPAN_M,
            "surrogate_distributed_mass_kgpm": distributed_mass,
            "surrogate_mass_policy": "distributed_mass_selected_so_tip_normalized_first_mode_generalized_mass_matches_published_effective_mass_per_span_times_span",
            "bending_axis_policy": "executed_QBlade_global_Y_probe_maps_crossflow_to_EIy;_EIx_is_100x_stiff_global_X_orthogonal_axis",
            "orthogonal_bending_stiffness_ratio": ORTHOGONAL_BENDING_STIFFNESS_RATIO,
            "mass_damping_bracket": list(MASS_DAMPINGS),
            "structural_damping_ratio_bracket": [
                value / MASS_RATIO for value in MASS_DAMPINGS
            ],
            "pairing_policy": "bracket_reported_range_no_inferred_NACA_specific_endpoint",
            "reduced_velocity_definition": "Ustar=U/(fd*c)",
            "damped_frequency_definition": "fd=fn*sqrt(1-zeta^2)",
            "reynolds_definition": "Re=U*c/nu",
            "mounting_surrogate": "first-bending cantilever matched in fn,mstar,zeta; not exact rigid-translation apparatus",
        },
        "matrix": {
            "reduced_velocities": list(
                ENGINEERING_REDUCED_VELOCITIES
                if engineering_path is not None
                else REDUCED_VELOCITIES
            ),
            "mass_dampings": list(MASS_DAMPINGS),
            "paired_timestep_reduced_velocity": (
                ENGINEERING_TIMESTEP_CHECK_USTAR
                if engineering_path is not None
                else TIMESTEP_CHECK_USTAR
            ),
            "timesteps_s": [TIMESTEP_S, HALF_TIMESTEP_S],
            "natural_cycles": NATURAL_CYCLES,
            "tail_cycles": TAIL_CYCLES,
            "case_count": len(rows),
        },
        "force_contract": {
            "native_qblade_aerodynamics": "enabled_both_scenarios",
            "legacy_vorlap_force_scale": 0.0,
            "raw_external_force": "identically_zero",
            "hybrid_source": (
                "latest_airfoil_spectral_energy_capture_and_frequency_pull"
                if work_map_path is None and engineering_path is None
                else (
                    "stationary_airfoil_spectral_energy_scaled_positive_incremental_prescribed_motion_work"
                    if work_map_path is not None
                    else "full_static_CL_spectrum_conditioned_velocity_correlated_negative_damping_with_modal_saturation"
                )
            ),
            "hybrid_residual_source_scale": 0.0,
            "residual_justification": "unqualified wind-axis-to-section-axis semantics at 90deg; native QBlade supplies baseline aerodynamics",
        },
        "raw_reuse_policy": "reuse_only_if_native_case_identity_json_matches_exactly_otherwise_refuse",
        "cases": rows,
    }
    if work_map_model is not None:
        manifest["work_map_identity"] = {
            "schema": "vorlap-airfoil-force-motion-work-map-v1",
            "sha256": work_map_sha256,
            "fingerprint": work_map_model.fingerprint,
            "stationary_force_spectrum_sha256": work_map_stationary_spectrum_sha256,
            "confidence_multiplier": 1.96,
        }
        manifest["qblade_runtime_identity"] = {
            "version": FROZEN_QBLADE_VERSION,
            "library_sha256": FROZEN_QBLADE_LIBRARY_SHA256,
            "runtime_dependency_fingerprint": (
                FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
            ),
        }
    elif engineering_model is not None:
        manifest["engineering_lockin_identity"] = {
            "schema": engineering_model.raw["schema"],
            "sha256": engineering_sha256,
            "fingerprint": engineering_model.fingerprint,
            "stationary_force_spectrum_sha256": (
                TRACKED_STATIONARY_SPECTRUM_SHA256
            ),
            "holdout_status": engineering_model.raw["holdout_assessment"][
                "status"
            ],
            "maximum_holdout_absolute_relative_error": engineering_model.raw[
                "holdout_assessment"
            ]["maximum_observed_absolute_relative_error"],
            "nalu_required": False,
        }
        manifest["qblade_runtime_identity"] = {
            "version": FROZEN_QBLADE_VERSION,
            "library_sha256": FROZEN_QBLADE_LIBRARY_SHA256,
            "runtime_dependency_fingerprint": (
                FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
            ),
        }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    path = output / "campaign_manifest.json"
    atomic_write_json(path, manifest)
    validate(path)
    return path


def _resolve_member(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise CampaignError(f"{label} must be a relative path.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignError(f"Unsafe {label}: {value!r}")
    path = root.joinpath(*relative.parts).resolve()
    path.relative_to(root.resolve())
    return path


def validate(value: Path) -> dict[str, Any]:
    path = Path(value).expanduser().resolve()
    path = path / "campaign_manifest.json" if path.is_dir() else path
    root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema not in {
        SCHEMA,
        WORK_MAP_SCHEMA,
        ENGINEERING_LOCKIN_SCHEMA,
    } or manifest.get("status") != "ready_for_native_linux_x86_submission":
        raise CampaignError("Not a released Ehrmann campaign.")
    payload = dict(manifest)
    fingerprint = payload.pop("fingerprint", None)
    if fingerprint != canonical_fingerprint(payload):
        raise CampaignError("Campaign fingerprint mismatch.")
    if manifest.get("structural_qualification") != _structural_qualification_identity():
        raise CampaignError("Campaign structural qualification identity differs.")
    work_map_campaign = schema == WORK_MAP_SCHEMA
    engineering_campaign = schema == ENGINEERING_LOCKIN_SCHEMA
    scenarios = (
        WORK_MAP_SCENARIOS
        if work_map_campaign
        else (
            ENGINEERING_LOCKIN_SCENARIOS
            if engineering_campaign
            else SCENARIOS
        )
    )
    if manifest.get("comparison_series") != list(scenarios):
        raise CampaignError("Campaign comparison series differs from its schema.")
    expected = build_cases(scenarios)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != len(expected):
        raise CampaignError("Campaign case count differs.")
    for observed, planned in zip(cases, expected):
        if not isinstance(observed, Mapping):
            raise CampaignError("Campaign case row is malformed.")
        for key in (
            "index", "case_id", "scenario", "native_physical_id",
            "mass_damping", "structural_damping_ratio", "damped_natural_frequency_hz", "reduced_velocity",
            "timestep_s", "num_steps",
        ):
            if observed.get(key) != planned[key]:
                raise CampaignError(f"Case {planned['case_id']} {key} differs.")
        sim = _resolve_member(root, observed["sim"], "sim")
        config_path = _resolve_member(root, observed["runtime_config"], "runtime config")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if read_qblade_keyword(sim, "INCLUDEAERO") != "1":
            raise CampaignError(f"Case {observed['case_id']} lacks native aerodynamics.")
        if float(config.get("force_scale", math.nan)) != 0.0 or config.get("include_mean_load") is not False:
            raise CampaignError(f"Case {observed['case_id']} violates the no-double-count force contract.")
        model = config.get("response_model")
        adapter = config.get(
            "single_mode_airfoil_work_map"
            if work_map_campaign
            else (
                "single_mode_engineering_lockin"
                if engineering_campaign
                else "single_mode_spectral_lockin"
            )
        )
        if not isinstance(model, Mapping) or not isinstance(adapter, Mapping):
            raise CampaignError(f"Case {observed['case_id']} lacks its declared adapter.")
        if work_map_campaign:
            identity = manifest.get("work_map_identity")
            qblade_runtime_identity = manifest.get("qblade_runtime_identity")
            if not isinstance(identity, Mapping):
                raise CampaignError("Work-map campaign identity is missing.")
            if qblade_runtime_identity != {
                "version": FROZEN_QBLADE_VERSION,
                "library_sha256": FROZEN_QBLADE_LIBRARY_SHA256,
                "runtime_dependency_fingerprint": (
                    FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
                ),
            }:
                raise CampaignError("Work-map QBlade runtime identity differs.")
            map_config = adapter.get("work_map")
            spectrum_config = adapter.get("static_spectrum")
            if (
                adapter.get("type")
                != "single_mode_airfoil_force_motion_work_map_v1"
                or adapter.get("residual_source_scale") != 0.0
                or not isinstance(map_config, Mapping)
                or map_config.get("sha256") != identity.get("sha256")
                or map_config.get("confidence_multiplier") != 1.96
                or map_config.get("require_nonpositive_high_amplitude_work") is not True
                or not isinstance(spectrum_config, Mapping)
                or spectrum_config.get("source_sha256")
                != identity.get("stationary_force_spectrum_sha256")
                or identity.get("stationary_force_spectrum_sha256")
                != TRACKED_STATIONARY_SPECTRUM_SHA256
            ):
                raise CampaignError(f"Case {observed['case_id']} work-map adapter differs.")
            staged_map = config_path.parent / str(map_config.get("path", ""))
            if (
                staged_map.is_symlink()
                or not staged_map.is_file()
                or sha256_file(staged_map) != identity.get("sha256")
            ):
                raise CampaignError(f"Case {observed['case_id']} work-map bytes differ.")
            parsed_map = AirfoilForceMotionWorkMap.from_json(staged_map)
            if parsed_map.fingerprint != identity.get("fingerprint"):
                raise CampaignError(f"Case {observed['case_id']} work-map fingerprint differs.")
            map_spectrum_sources = tuple(
                source
                for source in parsed_map.sources
                if source["role"] == "stationary_force_spectrum"
            )
            staged_spectrum = (
                root
                / str(observed["case_dir"])
                / "model"
                / "VorLapAirfoils"
                / "NACA0018.h5"
            )
            if (
                len(map_spectrum_sources) != 1
                or map_spectrum_sources[0]["sha256"]
                != identity.get("stationary_force_spectrum_sha256")
                or staged_spectrum.is_symlink()
                or not staged_spectrum.is_file()
                or sha256_file(staged_spectrum)
                != identity.get("stationary_force_spectrum_sha256")
            ):
                raise CampaignError(
                    f"Case {observed['case_id']} stationary-spectrum identity differs."
                )
        elif engineering_campaign:
            identity = manifest.get("engineering_lockin_identity")
            qblade_runtime_identity = manifest.get("qblade_runtime_identity")
            calibration_config = adapter.get("calibration")
            spectrum_config = adapter.get("static_spectrum")
            if (
                not isinstance(identity, Mapping)
                or qblade_runtime_identity
                != {
                    "version": FROZEN_QBLADE_VERSION,
                    "library_sha256": FROZEN_QBLADE_LIBRARY_SHA256,
                    "runtime_dependency_fingerprint": (
                        FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
                    ),
                }
                or adapter.get("type") != ENGINEERING_LOCKIN_MODEL_TYPE
                or adapter.get("residual_source_scale") != 0.0
                or not isinstance(calibration_config, Mapping)
                or calibration_config.get("sha256") != identity.get("sha256")
                or not isinstance(spectrum_config, Mapping)
                or spectrum_config.get("source_sha256")
                != TRACKED_STATIONARY_SPECTRUM_SHA256
            ):
                raise CampaignError(
                    f"Case {observed['case_id']} engineering adapter differs."
                )
            staged_calibration = config_path.parent / str(
                calibration_config.get("path", "")
            )
            if (
                staged_calibration.is_symlink()
                or not staged_calibration.is_file()
                or sha256_file(staged_calibration) != identity.get("sha256")
            ):
                raise CampaignError(
                    f"Case {observed['case_id']} engineering calibration differs."
                )
            parsed_calibration = EngineeringLockInCalibration.from_json(
                staged_calibration
            )
            if parsed_calibration.fingerprint != identity.get("fingerprint"):
                raise CampaignError(
                    f"Case {observed['case_id']} engineering fingerprint differs."
                )
            staged_spectrum = (
                root
                / str(observed["case_dir"])
                / "model"
                / "VorLapAirfoils"
                / "NACA0018.h5"
            )
            if (
                staged_spectrum.is_symlink()
                or not staged_spectrum.is_file()
                or sha256_file(staged_spectrum)
                != TRACKED_STATIONARY_SPECTRUM_SHA256
            ):
                raise CampaignError(
                    f"Case {observed['case_id']} engineering spectrum differs."
                )
        elif (
            adapter.get("type") != "single_mode_airfoil_spectral_lockin_hybrid"
            or adapter["spectral_lockin"].get("residual_source_scale") != 0.0
        ):
            raise CampaignError(f"Case {observed['case_id']} spectral adapter differs.")
        hybrid = observed["scenario"] == scenarios[1]
        expected_h1 = (
            0.0
            if (work_map_campaign or engineering_campaign)
            else (3.920125762951649 if hybrid else 0.0)
        )
        if float(model.get("aerodynamic_damping_gain_h1", math.nan)) != expected_h1:
            raise CampaignError(f"Case {observed['case_id']} H1 differs.")
        if bool(model["startup_seed"]["enabled"]) != hybrid:
            raise CampaignError(f"Case {observed['case_id']} seed policy differs.")
        if (work_map_campaign or engineering_campaign) and bool(
            adapter["modal_power"]["enabled"]
        ) != hybrid:
            raise CampaignError(f"Case {observed['case_id']} modal-power enablement differs.")
        current_identity = _native_identity(root, observed)
        if current_identity != observed.get("native_case_identity"):
            raise CampaignError(f"Case {observed['case_id']} native identity changed.")
    return {"valid": True, "root": root, "manifest": manifest}


def command_for_case(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
    python_executable: str,
    threads: int,
    resume: bool,
    qblade_steps: int | None = None,
) -> list[str]:
    state = validate(campaign)
    cases = state["manifest"]["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise CampaignError(f"case_index must be in [0,{len(cases)-1}].")
    case = cases[case_index]
    result = state["root"] / str(case["result_dir"])
    reuse_receipt = result / "reuse_receipt.json"
    if reuse_receipt.is_file():
        return []
    command = [
        str(python_executable),
        str(REPO_ROOT / "run_headless/run_qblade_sim.py"),
        "--sim", str(state["root"] / str(case["sim"])),
        "--qblade-root", str(Path(qblade_root).expanduser().resolve()),
        "--output-dir", str(result),
        "--result-name", str(case["result_name"]),
        "--result-filter", str(REPO_ROOT / "examples/vlastos2026_cantilever/cantilever_results.filter"),
        "--threads", str(threads),
        "--sil-vorlap",
        "--sil-vorlap-history-mode", "compact",
    ]
    if resume:
        command.extend(("--checkpoint-every", "5000", "--resume-from-checkpoint"))
    if qblade_steps is not None:
        command.extend(("--steps", str(qblade_steps)))
    return command


def run_case(**kwargs: Any) -> None:
    command = command_for_case(**kwargs)
    if not command:
        print("Exact-identity raw result already present; skipping execution.")
        return
    env = dict(**__import__("os").environ)
    env.update({"OMP_NUM_THREADS": str(kwargs["threads"]), "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)
    state = validate(kwargs["campaign"])
    case = state["manifest"]["cases"][int(kwargs["case_index"])]
    result = state["root"] / str(case["result_dir"])
    atomic_write_json(result / "native_case_identity.json", case["native_case_identity"])


def _result_metrics(result_path: Path, timestep_s: float) -> dict[str, object]:
    columns = read_qblade_result_numeric_columns(result_path)
    normalized = {normalize_qblade_header(name): name for name in columns}
    time_name = normalized.get("Time [s]")
    displacement_name = normalized.get(RESULT_CHANNEL)
    if time_name is None or displacement_name is None:
        raise CampaignError(
            f"Result lacks Time and {RESULT_CHANNEL} channels: {result_path}"
        )
    time = np.asarray(columns[time_name], dtype=float)
    displacement = np.asarray(columns[displacement_name], dtype=float)
    if (
        time.size < 16
        or displacement.size != time.size
        or np.any(~np.isfinite(time))
        or np.any(~np.isfinite(displacement))
        or np.any(np.diff(time) <= 0.0)
    ):
        raise CampaignError(f"Result history is incomplete or invalid: {result_path}")
    tail_start = max(time[0], time[-1] - TAIL_CYCLES / NATURAL_FREQUENCY_HZ)
    retain = time >= tail_start - 0.51 * timestep_s
    retained_time = time[retain]
    retained = displacement[retain]
    if retained.size < 16:
        raise CampaignError(f"Result lacks the declared terminal window: {result_path}")
    centered = retained - float(np.mean(retained))
    amplitude = 0.5 * (float(np.max(retained)) - float(np.min(retained)))
    frequency = np.fft.rfftfreq(retained.size, d=float(np.median(np.diff(retained_time))))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size <= 1 or float(np.sum(power[1:])) <= 0.0:
        dominant = None
    else:
        dominant = float(frequency[1 + int(np.argmax(power[1:]))])
    half = retained.size // 2
    early_amplitude = 0.5 * (
        float(np.max(retained[:half])) - float(np.min(retained[:half]))
    )
    final_amplitude = 0.5 * (
        float(np.max(retained[half:])) - float(np.min(retained[half:]))
    )
    tolerance = max(0.005 * CHORD_M, 0.05 * max(amplitude, 1.0e-12))
    return {
        "result_file": str(result_path.resolve()),
        "samples": int(time.size),
        "terminal_samples": int(retained.size),
        "terminal_amplitude_m": amplitude,
        "terminal_amplitude_over_chord": amplitude / CHORD_M,
        "terminal_dominant_frequency_hz": dominant,
        "terminal_dominant_frequency_over_fn": (
            None if dominant is None else dominant / NATURAL_FREQUENCY_HZ
        ),
        "terminal_early_amplitude_m": early_amplitude,
        "terminal_final_amplitude_m": final_amplitude,
        "stationarity_amplitude_tolerance_m": tolerance,
        "stationarity_amplitude_pass": abs(final_amplitude - early_amplitude) <= tolerance,
    }


def analyze(campaign_root: Path, *, output: Path | None = None, csv_output: Path | None = None) -> Path:
    state = validate(campaign_root)
    root = state["root"]
    rows = []
    missing = []
    for case in state["manifest"]["cases"]:
        result_dir = root / str(case["result_dir"])
        result_path = result_dir / f"{case['result_name']}.txt"
        summary_path = result_dir / f"{Path(str(case['sim'])).stem}_run_summary.json"
        if not result_path.is_file() or not summary_path.is_file():
            missing.append(str(case["case_id"]))
            continue
        metrics = _result_metrics(result_path, float(case["timestep_s"]))
        rows.append(
            {
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "mass_damping": case["mass_damping"],
                "structural_damping_ratio": case["structural_damping_ratio"],
                "reduced_velocity": case["reduced_velocity"],
                "reynolds_number": case["reynolds_number"],
                "timestep_s": case["timestep_s"],
                **metrics,
            }
        )
    status = "complete_simulation_comparison_reference_overlay_tracked_separately"
    if missing:
        status = "incomplete_missing_results"
    dt_pairs = []
    comparison_series = tuple(str(value) for value in state["manifest"]["comparison_series"])
    for mass_damping in MASS_DAMPINGS:
        for scenario in comparison_series:
            selected = [
                row for row in rows
                if row["mass_damping"] == mass_damping
                and row["scenario"] == scenario
                and row["reduced_velocity"] == TIMESTEP_CHECK_USTAR
            ]
            if len(selected) == 2:
                selected.sort(key=lambda row: row["timestep_s"], reverse=True)
                coarse, fine = selected
                dt_pairs.append(
                    {
                        "mass_damping": mass_damping,
                        "scenario": scenario,
                        "reduced_velocity": TIMESTEP_CHECK_USTAR,
                        "coarse_timestep_s": coarse["timestep_s"],
                        "fine_timestep_s": fine["timestep_s"],
                        "absolute_amplitude_over_chord_difference": abs(
                            float(coarse["terminal_amplitude_over_chord"])
                            - float(fine["terminal_amplitude_over_chord"])
                        ),
                    }
                )
    report = {
        "schema": "vorlap-ehrmann2014-naca0018-analysis-v1",
        "status": status,
        "campaign_fingerprint": state["manifest"]["fingerprint"],
        "expected_cases": len(state["manifest"]["cases"]),
        "completed_cases": len(rows),
        "missing_case_ids": missing,
        "experimental_overlay_status": (
            "available_in_examples/ehrmann2014_naca0018/reference_data_and_"
            "validation_evidence"
        ),
        "comparison_contract": [
            "Ehrmann et al. experiment",
            "Raw QBlade native aerodynamics",
            (
                "Raw QBlade plus latest VorLap airfoil work-map bridge"
                if state["manifest"]["schema"] == WORK_MAP_SCHEMA
                else (
                    "Raw QBlade plus latest VorLap engineering spectral negative-damping bridge"
                    if state["manifest"]["schema"]
                    == ENGINEERING_LOCKIN_SCHEMA
                    else "Raw QBlade plus latest VorLap spectral-lockin bridge"
                )
            ),
        ],
        "timestep_pairs": dt_pairs,
        "cases": rows,
    }
    output_path = root / "analysis/ehrmann2014_analysis.json" if output is None else Path(output).expanduser().resolve()
    csv_path = root / "analysis/ehrmann2014_case_metrics.csv" if csv_output is None else Path(csv_output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, report)
    fieldnames = [
        "case_id", "scenario", "mass_damping", "structural_damping_ratio",
        "reduced_velocity", "reynolds_number", "timestep_s",
        "terminal_amplitude_m", "terminal_amplitude_over_chord",
        "terminal_dominant_frequency_hz", "terminal_dominant_frequency_over_fn",
        "stationarity_amplitude_pass",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--runtime-vorlap-root", type=Path, default=REPO_ROOT)
    generate_parser.add_argument("--raw-reuse-root", type=Path, default=None)
    generate_parser.add_argument(
        "--work-map",
        type=Path,
        default=None,
        help="Generate the v2 work-map campaign from calibrated map JSON.",
    )
    generate_parser.add_argument(
        "--engineering-calibration",
        type=Path,
        default=None,
        help="Generate the v3 engineering spectral negative-damping campaign.",
    )
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--campaign", type=Path, required=True)
    list_parser = commands.add_parser("list")
    list_parser.add_argument("--campaign", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--campaign", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, default=None)
    analyze_parser.add_argument("--csv-output", type=Path, default=None)
    for name in ("show-command", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--campaign", type=Path, required=True)
        sub.add_argument("--qblade-root", type=Path, required=True)
        sub.add_argument("--case-index", type=int, required=True)
        sub.add_argument("--python", default=sys.executable)
        sub.add_argument("--threads", type=int, default=4)
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--qblade-steps", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        print(
            generate(
                args.output,
                runtime_vorlap_root=args.runtime_vorlap_root,
                raw_reuse_root=args.raw_reuse_root,
                work_map=args.work_map,
                engineering_calibration=args.engineering_calibration,
            )
        )
    elif args.command == "validate":
        state = validate(args.campaign)
        print(json.dumps({"valid": True, "case_count": len(state["manifest"]["cases"]), "fingerprint": state["manifest"]["fingerprint"]}, indent=2))
    elif args.command == "list":
        for case in validate(args.campaign)["manifest"]["cases"]:
            print(f"{case['index']:2d} {case['case_id']} U*={case['reduced_velocity']:g} Re={case['reynolds_number']:.0f} {case['scenario']}")
    elif args.command == "analyze":
        path = analyze(args.campaign, output=args.output, csv_output=args.csv_output)
        report = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({"path": str(path), "status": report["status"], "completed_cases": report["completed_cases"]}, indent=2))
        return 0 if report["status"] != "incomplete_missing_results" else 2
    else:
        kwargs = dict(campaign=args.campaign, case_index=args.case_index, qblade_root=args.qblade_root, python_executable=args.python, threads=args.threads, resume=args.resume, qblade_steps=args.qblade_steps)
        if args.command == "show-command":
            import shlex
            print(shlex.join(command_for_case(**kwargs)))
        else:
            run_case(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
