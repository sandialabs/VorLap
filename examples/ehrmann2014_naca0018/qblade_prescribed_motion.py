#!/usr/bin/env python3
"""Generate and audit native-QBlade prescribed-motion NACA0018 histories.

This is the exact-runtime-base half of the signed incremental work-map
identification.  It owns no VorLap callback: QBlade's native aerodynamic model
is advanced with a prescribed global-Y platform translation, and the total
native aerodynamic force in hub/global Y is exported on the same analytic
motion grid used by the paired Nalu campaign.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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

from vorlap.qblade_files import (
    read_qblade_keyword,
    replace_qblade_keyword_text,
)
from vorlap.qblade_run_support import (
    canonical_fingerprint,
    qblade_runtime_dependency_manifest,
)
from vorlap.qblade_sil import (
    normalize_qblade_header,
    read_qblade_result_numeric_columns,
)

REQUEST_PATH = HERE / "work_map_data_request.json"
CAMPAIGN_SCHEMA = "vorlap-naca0018-qblade-prescribed-motion-campaign-v1"
RESULTS_SCHEMA = "vorlap-naca0018-qblade-prescribed-motion-results-v1"
HISTORY_COLUMNS = (
    "time_s",
    "displacement_m",
    "velocity_mps",
    "crossflow_force_n",
)

EXPECTED_REYNOLDS = (64_000.0, 100_000.0, 140_000.0)
EXPECTED_K = (0.45, 0.65, 0.8, 1.05)
EXPECTED_NONZERO_AMPLITUDE_RATIOS = (0.005, 0.0125, 0.025, 0.05)
EXPECTED_VORLAP_INCIDENCE_DEG = -90.0
EXPECTED_QBLADE_INCIDENCE_DEG = 90.0
EXPECTED_DIRECTION_DEG = 0.0
TARGET_ST_MAX = 0.4
TARGET_SAMPLES_AT_ST_MAX = 120
DISCARD_CYCLES = 5
RETAINED_CYCLES = 8
TOTAL_CYCLES = DISCARD_CYCLES + RETAINED_CYCLES
N_STATIONS = 21
PREFLIGHT_CASE_INDEX = 35
PREFLIGHT_CASE_ID = "re140000_k0p45_a0p0500"
PREFLIGHT_SCHEDULER_SCHEMA = (
    "vorlap-naca0018-qblade-prescribed-motion-scheduler-v2"
)
PREFLIGHT_SCHEDULER_REPORT = "qblade_preflight_scheduler_report.json"

EXPECTED_QBLADE_VERSION = "2.0.9.7"
EXPECTED_QBLADE_LIBRARY_SHA256 = (
    "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
)
EXPECTED_QBLADE_DEPENDENCY_FINGERPRINT = (
    "c50c6909317c754caec4130ccbc8e2ed3f705f810f07da11e7ce3ad003499089"
)
QBLADE_LIBRARY_NAME = "libQBladeCE_2.0.9.7.so.1.0.0"

TIME_CHANNEL = "Time [s]"
PLATFORM_Y_CHANNEL = "Y_g Prescr. Platform Trans. [m]"
BLADE_TIP_Y_CHANNEL = "Y_g Pos. BLD_1 pos 1.000 [m]"
FORCE_CHANNEL = "Aerodynamic Force in Hub Y_h Direction [N]"
AOA_CHANNEL = "Angle of Attack at 0.25c BLD_1 PAN_10 [deg]"
REYNOLDS_CHANNEL = "Reynolds Number BLD_1 PAN_10 [-]"
FILTER_CHANNELS = (
    TIME_CHANNEL,
    "Abs Inflow Vel. at Hub [m/s]",
    "LSS Rpm [rpm]",
    "True Pitch Angle BLD_1 [deg]",
    "True Yaw Angle [deg]",
    PLATFORM_Y_CHANNEL,
    BLADE_TIP_Y_CHANNEL,
    FORCE_CHANNEL,
    AOA_CHANNEL,
    REYNOLDS_CHANNEL,
)


class CampaignError(RuntimeError):
    """Raised when the native-QBlade runtime-base contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _identity(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _relative_member(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise CampaignError(f"{label} must be a relative path string.")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise CampaignError(f"{label} is not a canonical relative path.")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2.0e-13, abs_tol=1.0e-14):
        raise CampaignError(
            f"{label} changed: expected {expected!r}, found {actual!r}."
        )


def _load_request() -> tuple[dict[str, Any], str]:
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    if request.get("schema") != "vorlap-naca0018-prescribed-motion-data-request-v1":
        raise CampaignError("The prescribed-motion request schema is not frozen v1.")
    axes = request["frozen_axes"]
    if tuple(map(float, axes["reynolds_number"])) != EXPECTED_REYNOLDS:
        raise CampaignError("The requested Reynolds grid differs.")
    if tuple(map(float, axes["reduced_frequency_k"])) != EXPECTED_K:
        raise CampaignError("The requested reduced-frequency grid differs.")
    if tuple(map(float, axes["incidence_deg"])) != (EXPECTED_VORLAP_INCIDENCE_DEG,):
        raise CampaignError("The requested VorLap incidence differs.")
    if tuple(map(float, axes["direction_deg_mod_180"])) != (EXPECTED_DIRECTION_DEG,):
        raise CampaignError("The requested cross-flow direction differs.")
    if (
        tuple(map(float, axes["amplitude_ratio"]))
        != (0.0,) + EXPECTED_NONZERO_AMPLITUDE_RATIOS
    ):
        raise CampaignError("The requested amplitude grid differs.")
    minimum = request["minimum_accepted_history"]
    if (
        int(minimum["discard_cycles"]) != DISCARD_CYCLES
        or int(minimum["complete_cycles_after_discard"]) != RETAINED_CYCLES
        or int(minimum["samples_per_cycle"]) > TARGET_SAMPLES_AT_ST_MAX
    ):
        raise CampaignError("The requested history minimum differs.")
    physical = request["frozen_fluid_and_section"]
    _assert_close(float(physical["reference_chord_m"]), 0.23, "reference chord")
    _assert_close(float(physical["section_span_m"]), 4.0, "section span")
    return request, _sha256(REQUEST_PATH)


def _physics(request: Mapping[str, Any]) -> dict[str, float]:
    source = request["frozen_fluid_and_section"]
    return {
        "fluid_density_kgpm3": float(source["fluid_density_kgpm3"]),
        "fluid_dynamic_viscosity_pas": float(source["fluid_dynamic_viscosity_pas"]),
        "reference_chord_m": float(source["reference_chord_m"]),
        "section_span_m": float(source["section_span_m"]),
    }


def _tag(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace("-", "m").replace(".", "p")


def _case_id(reynolds: float, reduced_frequency: float, amplitude_ratio: float) -> str:
    return (
        f"re{int(round(reynolds)):06d}_k{_tag(reduced_frequency, 2)}_"
        f"a{_tag(amplitude_ratio, 4)}"
    )


def _motion_values(
    reynolds: float,
    reduced_frequency: float,
    amplitude_ratio: float,
    physics: Mapping[str, float],
) -> dict[str, float | int]:
    rho = physics["fluid_density_kgpm3"]
    mu = physics["fluid_dynamic_viscosity_pas"]
    chord = physics["reference_chord_m"]
    speed = reynolds * mu / (rho * chord)
    frequency = reduced_frequency * speed / (math.pi * chord)
    period = 1.0 / frequency
    samples = math.ceil(
        max(
            100.0,
            TARGET_SAMPLES_AT_ST_MAX * TARGET_ST_MAX * math.pi / reduced_frequency,
        )
    )
    timestep = period / samples
    expected_rows = TOTAL_CYCLES * samples + 1
    wrapper_steps = expected_rows + 1
    return {
        "wind_speed_mps": speed,
        "frequency_hz": frequency,
        "period_s": period,
        "amplitude_m": amplitude_ratio * chord,
        "samples_per_cycle": samples,
        "time_step_s": timestep,
        "expected_result_rows": expected_rows,
        "wrapper_steps": wrapper_steps,
        "relative_history_start_s": timestep,
        "relative_history_end_s": expected_rows * timestep,
    }


def _patch(text: str, values: Mapping[str, object]) -> str:
    for keyword, value in values.items():
        text, _old, count = replace_qblade_keyword_text(text, keyword, value)
        if count != 1:
            raise CampaignError(f"Could not patch required QBlade keyword {keyword}.")
    return text


def _write_blade(path: Path, chord: float, span: float) -> None:
    stations = np.linspace(0.0, span, N_STATIONS)
    lines = [
        "----------------------------------------QBlade Blade Definition File------------------------------------------------",
        "Generated by examples/ehrmann2014_naca0018/qblade_prescribed_motion.py",
        "naca0018_prescribed_motion_rigid OBJECTNAME",
        "HAWT ROTORTYPE",
        "1 NUMBLADES",
        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE",
    ]
    lines.extend(
        f"{station:.17g} {chord:.17g} 0 0 0 0.5 Polars/NACA_0018_RFoil_MultiRePolar.plr"
        for station in stations
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rigid_beam(path: Path) -> None:
    header = [
        "0 RAYLEIGHDMP",
        "1 STIFFTUNER",
        "1 MASSTUNER",
        "0 INTPTYPE",
        "1 BEAMTYPE",
        "2 DISCTYPE",
        f"{N_STATIONS} DISC",
        "LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] EA_[N] GJ_[N.m^2] GA_[N] STRPIT_[deg] KSX_[-] KSY_[-] RGX_[-] RGY_[-] XCM_[-] YCM_[-] XCE_[-] YCE_[-] XCS_[-] YCS_[-]",
    ]
    row_tail = "1 1e15 1e15 1e15 1e15 1e15 0 0.85 0.85 0.25 0.25 0 0 0 0 0 0"
    rows = [
        f"{fraction:.17g} {row_tail}" for fraction in np.linspace(0.0, 1.0, N_STATIONS)
    ]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def _write_rigid_tower(path: Path) -> None:
    header = [
        "0 RAYLEIGHDMP",
        "1 STIFFTUNER",
        "1 MASSTUNER",
        "0 INTPTYPE",
        "1 BEAMTYPE",
        "0 DISCTYPE",
        "2 DISC",
        "LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] EA_[N] GJ_[N.m^2] GA_[N] STRPIT_[deg] KSX_[-] KSY_[-] RGX_[-] RGY_[-] XCM_[-] YCM_[-] XCE_[-] YCE_[-] XCS_[-] YCS_[-] Diameter_[m]",
    ]
    tail = "1 1e15 1e15 1e15 1e15 1e15 0 0.85 0.85 0.25 0.25 0 0 0 0 0 0 1"
    path.write_text(
        "\n".join(header + [f"0 {tail}", f"1 {tail}"]) + "\n", encoding="utf-8"
    )


def _write_main_structure(path: Path) -> None:
    lines = [
        "---------------------- QBLADE STRUCTURAL MODEL INPUT FILE -----------------",
        "0.0005 GLBGEOEPS",
        "0 PRECONE",
        "0 SHFTTILT",
        "0 OVERHANG",
        "0 TWR2SHFT",
        "0 YAWBRMASS",
        "0 NACMASS",
        "0 NACCMX",
        "0 NACCMY",
        "0 NACCMZ",
        "0 0 0 0 0 0 NACINER",
        "0 NACYINER",
        "0 HUBMASS",
        "0 HUBCM",
        "1 HUBINER",
        "1 GBRATIO",
        "1 GBOXEFF",
        "1 GENEFF",
        "false DRTRDOF",
        "1 GENINER",
        "1e15 DTTORSPR",
        "1e12 DTTORDMP",
        "1 NUMBLD",
        "rigid_blade.str BLDFILE_1",
        "1 TWRHEIGHT",
        "rigid_tower.str TWRFILE",
        "true FOR_OUT",
        "true ROT_OUT",
        "true MOM_OUT",
        "true DEF_OUT",
        "true POS_OUT",
        "true VEL_OUT",
        "true ACC_OUT",
        "true LVE_OUT",
        "true LAC_OUT",
    ]
    lines.extend(
        f"BLD_1_{fraction:.10f}" for fraction in np.linspace(0.0, 1.0, N_STATIONS)
    )
    lines.extend(("TWR_0.0", "TWR_1.0"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_motion(path: Path, case: Mapping[str, Any]) -> None:
    amplitude = float(case["amplitude_m"])
    omega = 2.0 * math.pi * float(case["frequency_hz"])
    timestep = float(case["time_step_s"])
    steps = int(case["wrapper_steps"])
    lines = ["Time TransX TransY TransZ RotX RotY RotZ"]
    for index in range(steps + 1):
        time = index * timestep
        displacement = amplitude * math.sin(omega * time)
        lines.append(f"{time:.17g} 0 {displacement:.17g} 0 0 0 0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _runtime_contract(qblade_root: Path) -> dict[str, Any]:
    root = qblade_root.expanduser().resolve()
    if root.name != f"QBladeCE_{EXPECTED_QBLADE_VERSION}":
        raise CampaignError(
            f"QBlade root must be the frozen QBladeCE_{EXPECTED_QBLADE_VERSION} runtime."
        )
    library = root / QBLADE_LIBRARY_NAME
    if library.is_symlink() or not library.is_file():
        raise CampaignError(f"Frozen QBlade library is unavailable: {library}")
    library_identity = _file_identity(library)
    if library_identity["sha256"] != EXPECTED_QBLADE_LIBRARY_SHA256:
        raise CampaignError("The QBlade 2.0.9.7 library SHA-256 differs.")
    dependencies, fingerprint = qblade_runtime_dependency_manifest(root)
    if fingerprint != EXPECTED_QBLADE_DEPENDENCY_FINGERPRINT:
        raise CampaignError("The QBlade runtime dependency fingerprint differs.")
    return {
        "root": str(root),
        "version": EXPECTED_QBLADE_VERSION,
        "library": library_identity,
        "runtime_dependencies": dependencies,
        "runtime_dependency_fingerprint": fingerprint,
    }


def _case_assets(case_root: Path) -> list[dict[str, object]]:
    paths = [
        case_root / f"{case_root.name}.sim",
        case_root / "motion.dat",
        case_root / "results.filter",
        case_root / "case_metadata.json",
        case_root / "model" / "naca0018_prescribed_motion.trb",
        case_root / "model" / "Aero" / "naca0018_prescribed_motion.bld",
        case_root / "model" / "Aero" / "Airfoils" / "NACA_0018_RFoil.afl",
        case_root / "model" / "Aero" / "Polars" / "NACA_0018_RFoil_MultiRePolar.plr",
        case_root / "model" / "Structure" / "rigid_blade.str",
        case_root / "model" / "Structure" / "rigid_tower.str",
        case_root / "model" / "Structure" / "rigid_main.str",
    ]
    return [_identity(path, case_root) for path in paths]


def generate_campaign(campaign: Path, qblade_root: Path) -> dict[str, Any]:
    request, request_sha = _load_request()
    physical = _physics(request)
    runtime = _runtime_contract(qblade_root)
    campaign = campaign.expanduser().resolve()
    if campaign.exists():
        raise CampaignError(f"Refusing to replace an existing campaign: {campaign}")
    campaign.mkdir(parents=True)

    reference = (
        REPO_ROOT / "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine"
    )
    sim_template = (
        REPO_ROOT
        / "examples/ReferenceHVAWTQBladeModel/baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
    )
    trb_template = reference / "10m_HVAWT_reference_turbine.trb"
    airfoil = reference / "Aero/Airfoils/NACA_0018_RFoil.afl"
    polar = reference / "Aero/Polars/NACA_0018_RFoil_MultiRePolar.plr"
    for source in (sim_template, trb_template, airfoil, polar):
        if not source.is_file():
            raise CampaignError(
                f"Required QBlade source asset is unavailable: {source}"
            )

    rows: list[dict[str, Any]] = []
    for reynolds in EXPECTED_REYNOLDS:
        for reduced_frequency in EXPECTED_K:
            for amplitude_ratio in EXPECTED_NONZERO_AMPLITUDE_RATIOS:
                values = _motion_values(
                    reynolds, reduced_frequency, amplitude_ratio, physical
                )
                case_id = _case_id(reynolds, reduced_frequency, amplitude_ratio)
                case_root = campaign / "cases" / case_id
                aero = case_root / "model" / "Aero"
                structure = case_root / "model" / "Structure"
                (aero / "Airfoils").mkdir(parents=True)
                (aero / "Polars").mkdir()
                structure.mkdir()
                result_dir = campaign / "results" / case_id
                result_dir.mkdir(parents=True)

                shutil.copy2(airfoil, aero / "Airfoils" / airfoil.name)
                shutil.copy2(polar, aero / "Polars" / polar.name)
                _write_blade(
                    aero / "naca0018_prescribed_motion.bld",
                    physical["reference_chord_m"],
                    physical["section_span_m"],
                )
                _write_rigid_beam(structure / "rigid_blade.str")
                _write_rigid_tower(structure / "rigid_tower.str")
                _write_main_structure(structure / "rigid_main.str")

                turbine_text = _patch(
                    trb_template.read_text(encoding="utf-8"),
                    {
                        "OBJECTNAME": case_id,
                        "BLADEFILE": "Aero/naca0018_prescribed_motion.bld",
                        "TURBTYPE": 0,
                        "NUMBLADES": 1,
                        "ROTORCONFIG": 0,
                        "ROTATIONALDIR": 0,
                        "DISCTYPE": 0,
                        "INTPTYPE": 0,
                        "OVERHANG": 0,
                        "SHAFTTILT": 0,
                        "ROTORCONE": 0,
                        "TOWERSHADOW": "false",
                        "WAKETYPE": 1,
                        "STRUCTURALFILE": "Structure/rigid_main.str",
                        "GEOMSTIFFNESS": "false",
                        "AEROPANELLOADS": "false",
                        "CONTROLLERTYPE": 0,
                        "CONTROLLERFILE": "",
                        "PARAMETERFILE": "",
                        "LIBFILE_1": "",
                        "LIBFUNCTION_1": "",
                        "LIBARRAYSIZE_1": "",
                        "LIBPARAMETERFILE_1": "",
                    },
                )
                turbine = case_root / "model" / "naca0018_prescribed_motion.trb"
                turbine.write_text(turbine_text, encoding="utf-8")

                row: dict[str, Any] = {
                    "index": len(rows),
                    "case_id": case_id,
                    "incidence_deg": EXPECTED_VORLAP_INCIDENCE_DEG,
                    "qblade_internal_incidence_deg": EXPECTED_QBLADE_INCIDENCE_DEG,
                    "reynolds_number": reynolds,
                    "reduced_frequency_k": reduced_frequency,
                    "amplitude_ratio": amplitude_ratio,
                    "direction_deg_mod_180": EXPECTED_DIRECTION_DEG,
                    **values,
                    "discard_cycles": DISCARD_CYCLES,
                    "retained_cycles": RETAINED_CYCLES,
                    "case_dir": f"cases/{case_id}",
                    "sim": f"cases/{case_id}/{case_id}.sim",
                    "motion_file": f"cases/{case_id}/motion.dat",
                    "result_filter": f"cases/{case_id}/results.filter",
                    "result_dir": f"results/{case_id}",
                    "result_name": f"{case_id}_qblade_native",
                    "canonical_history": {
                        "path": f"histories/{case_id}_qblade.csv",
                        "status": "pending",
                        "sha256": None,
                        "rows_expected": int(values["expected_result_rows"]),
                        "columns": list(HISTORY_COLUMNS),
                    },
                }
                _write_motion(case_root / "motion.dat", row)
                (case_root / "results.filter").write_text(
                    "\n".join(FILTER_CHANNELS) + "\n", encoding="utf-8"
                )
                case_metadata = {
                    "schema": "vorlap-naca0018-qblade-prescribed-motion-case-v1",
                    "case_id": case_id,
                    "vertex": {
                        key: row[key]
                        for key in (
                            "incidence_deg",
                            "reynolds_number",
                            "reduced_frequency_k",
                            "amplitude_ratio",
                            "direction_deg_mod_180",
                        )
                    },
                    "external_force_policy": "none_native_qblade_aerodynamics_only",
                }
                _write_json(case_root / "case_metadata.json", case_metadata)

                sim_text = _patch(
                    sim_template.read_text(encoding="utf-8"),
                    {
                        "OBJECTNAME": case_id,
                        "TURBFILE": "model/naca0018_prescribed_motion.trb",
                        "TURBNAME": case_id,
                        "INITIAL_YAW": 0,
                        "INITIAL_PITCH": 0,
                        "INITIAL_AZIMUTH": 0,
                        "STRSUBSTEP": 1,
                        "RELAXSTEPS": 0,
                        "PRESCRIBETYPE": 1,
                        "RPMPRESCRIBED": 0,
                        "INCLUDEAERO": 1,
                        "INCLUDEHYDRO": 0,
                        "GLOBPOS_X": 0,
                        "GLOBPOS_Y": 0,
                        "GLOBPOS_Z": 0,
                        "GLOBROT_X": 0,
                        "GLOBROT_Y": 0,
                        "GLOBROT_Z": 0,
                        "EVENTFILE": "",
                        "LOADINGFILE": "",
                        "SIMFILE": "",
                        "MOTIONFILE": "motion.dat",
                        "TIMESTEP": f"{float(row['time_step_s']):.17g}",
                        "NUMTIMESTEPS": row["wrapper_steps"],
                        "RAMPUP": 0,
                        "ADDDAMP": 0,
                        "WNDTYPE": 0,
                        "WNDNAME": "",
                        "MEANINF": f"{float(row['wind_speed_mps']):.17g}",
                        "HORANGLE": 0,
                        "VERTANGLE": 0,
                        "PROFILETYPE": 0,
                        "SHEAREXP": 0,
                        "DENSITYAIR": f"{physical['fluid_density_kgpm3']:.17g}",
                        "VISCOSITYAIR": f"{physical['fluid_dynamic_viscosity_pas'] / physical['fluid_density_kgpm3']:.17g}",
                        "STOREFROM": 0,
                        "STOREREPLAY": "false",
                        "STOREAERO": "true",
                        "STOREBLADE": "true",
                        "STORESTRUCT": "true",
                        "STORESIM": "true",
                        "STOREHYDRO": "false",
                        "STORECONTROLLER": "false",
                        "FILTERFILE": "",
                        "CALCMODAL": "false",
                    },
                )
                (case_root / f"{case_id}.sim").write_text(sim_text, encoding="utf-8")
                row["assets"] = _case_assets(case_root)
                row["asset_fingerprint"] = canonical_fingerprint(row["assets"])
                rows.append(row)

    manifest: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "generated_awaiting_qblade_results",
        "request": {
            "path": str(REQUEST_PATH.resolve()),
            "sha256": request_sha,
            "schema": request["schema"],
        },
        "runtime": runtime,
        "fluid_and_section": physical,
        "geometry_and_frames": {
            "profile": "NACA0018",
            "blade_count": 1,
            "station_count": N_STATIONS,
            "chord_axis_global": [0.0, 1.0, 0.0],
            "normal_axis_global": [-1.0, 0.0, 0.0],
            "ambient_wind_global": [1.0, 0.0, 0.0],
            "qblade_internal_incidence_deg": EXPECTED_QBLADE_INCIDENCE_DEG,
            "vorlap_relative_wind_incidence_deg": EXPECTED_VORLAP_INCIDENCE_DEG,
            "mapping": (
                "QBlade +90 uses the opposite relative-wind sign convention from "
                "VorLap atan2(U.normal,U.chord)=-90; pitch and twist remain zero"
            ),
            "hub_y_equals_global_y": "zero_yaw_zero_shaft_tilt_zero_global_rotation",
        },
        "native_aero": {
            "enabled": True,
            "dynamic_stall_type": 1,
            "unsteady_noncirculatory_aero": False,
            "two_point_lift_drag_correction": True,
            "wake_type": "unsteady_BEM",
            "bem_tip_loss": False,
            "tower_shadow": False,
            "rotation_rpm": 0.0,
            "internal_storage_filter": None,
            "export_filter": "results.filter_exact_10_channels",
            "filter_rationale": (
                "QBlade 2.0.9.7 silently omits the requested panel AoA/Re "
                "diagnostics when FILTERFILE is active during simulation; retain "
                "native channels internally and filter only at exportResults"
            ),
        },
        "external_forces": {
            "controller_type": 0,
            "external_library_entries": 0,
            "sil_vorlap": False,
            "sil_constant_force": False,
            "expected_native_callback_updates": 0,
        },
        "history_contract": {
            "first_row": "dt",
            "last_row": "(13*samples_per_cycle+1)*dt",
            "wrapper_steps_rule": "13*samples_per_cycle+2",
            "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
            "columns": list(HISTORY_COLUMNS),
            "float_encoding": ".17g",
            "force_channel": FORCE_CHANNEL,
            "force_frame": "hub_Y_equals_global_crossflow_Y",
            "force_sign": "positive_fluid_on_body_in_positive_global_Y",
            "force_units": "N_total_over_4m_span",
        },
        "execution": {
            "threads_per_case": 1,
            "total_case_count": 48,
            "preflight_cpu_slots": 1,
            "matrix_cpu_slots": 47,
            "local_docker_max_cpu_slots": 8,
            "default_walltime": "02:00:00",
            "resume_walltime": "08:00:00",
            "checkpoint_every_steps": 1000,
        },
        "preflight_gate": {
            "case_index": PREFLIGHT_CASE_INDEX,
            "case_id": PREFLIGHT_CASE_ID,
            "selection_basis": (
                "maximum samples-per-cycle, maximum A/c, and maximum Reynolds "
                "number: longest-grid, largest-motion, high-Re representative"
            ),
            "required_before_matrix": True,
            "required_scheduler_schema": PREFLIGHT_SCHEDULER_SCHEMA,
            "required_scheduler_stage": "preflight",
            "required_scheduler_status": "complete",
            "required_case_audit_schema": (
                "vorlap-naca0018-qblade-prescribed-motion-case-audit-v1"
            ),
        },
        "cases": rows,
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    _write_json(campaign / "campaign_manifest.json", manifest)
    return manifest


def _read_manifest(campaign: Path) -> tuple[Path, dict[str, Any]]:
    root = campaign.expanduser().resolve()
    path = root / "campaign_manifest.json"
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"Campaign manifest is unavailable: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError("Campaign manifest must contain a JSON object.")
    return root, value


def _validate_no_callback(turbine: Path) -> None:
    for keyword in (
        "LIBFILE_1",
        "LIBFUNCTION_1",
        "LIBARRAYSIZE_1",
        "LIBPARAMETERFILE_1",
        "CONTROLLERFILE",
        "PARAMETERFILE",
    ):
        if read_qblade_keyword(turbine, keyword) is not None:
            raise CampaignError(f"Native runtime-base turbine enables {keyword}.")


def validate_campaign(
    campaign: Path, qblade_root: Path | None = None
) -> dict[str, Any]:
    root, manifest = _read_manifest(campaign)
    if manifest.get("schema") != CAMPAIGN_SCHEMA:
        raise CampaignError("Campaign schema differs.")
    fingerprint = manifest.get("fingerprint")
    payload = dict(manifest)
    payload.pop("fingerprint", None)
    if fingerprint != canonical_fingerprint(payload):
        raise CampaignError("Campaign fingerprint differs.")
    if manifest.get("request", {}).get("sha256") != _sha256(REQUEST_PATH):
        raise CampaignError("Tracked request identity differs.")
    # ``runtime.root`` records where the producer actually executed, but is not
    # a portable replay path.  Resolve the verifier's local frozen runtime when
    # no explicit root is supplied, and compare byte identities below.
    runtime_root = (
        REPO_ROOT / f"run_headless/QBladeCE_{EXPECTED_QBLADE_VERSION}"
        if qblade_root is None
        else qblade_root
    )
    runtime = _runtime_contract(runtime_root)
    recorded_runtime = manifest["runtime"]
    recorded_library = recorded_runtime.get("library")
    if not isinstance(recorded_library, Mapping):
        raise CampaignError("QBlade runtime library differs.")
    if (
        recorded_runtime.get("version") != runtime["version"]
        or recorded_runtime.get("runtime_dependencies")
        != runtime["runtime_dependencies"]
        or recorded_runtime.get("runtime_dependency_fingerprint")
        != runtime["runtime_dependency_fingerprint"]
        or recorded_library.get("sha256") != runtime["library"]["sha256"]
        or recorded_library.get("bytes") != runtime["library"]["bytes"]
        or Path(str(recorded_library.get("path", ""))).name
        != QBLADE_LIBRARY_NAME
    ):
        raise CampaignError("QBlade runtime byte identity differs.")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 48:
        raise CampaignError("Campaign must contain exactly 48 prescribed-motion cases.")
    preflight = manifest.get("preflight_gate")
    if not isinstance(preflight, Mapping) or (
        preflight.get("case_index"),
        preflight.get("case_id"),
        preflight.get("required_before_matrix"),
        preflight.get("required_scheduler_schema"),
        preflight.get("required_scheduler_stage"),
        preflight.get("required_scheduler_status"),
    ) != (
        PREFLIGHT_CASE_INDEX,
        PREFLIGHT_CASE_ID,
        True,
        PREFLIGHT_SCHEDULER_SCHEMA,
        "preflight",
        "complete",
    ):
        raise CampaignError("The explicit QBlade preflight-gate contract differs.")
    if cases[PREFLIGHT_CASE_INDEX].get("case_id") != PREFLIGHT_CASE_ID:
        raise CampaignError("The designated QBlade preflight case differs.")
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping) or (
        execution.get("threads_per_case"),
        execution.get("total_case_count"),
        execution.get("preflight_cpu_slots"),
        execution.get("matrix_cpu_slots"),
        execution.get("local_docker_max_cpu_slots"),
    ) != (1, 48, 1, 47, 8):
        raise CampaignError("The staged QBlade execution contract differs.")
    expected_vertices = {
        (reynolds, reduced_frequency, amplitude)
        for reynolds in EXPECTED_REYNOLDS
        for reduced_frequency in EXPECTED_K
        for amplitude in EXPECTED_NONZERO_AMPLITUDE_RATIOS
    }
    observed_vertices: set[tuple[float, float, float]] = set()
    for index, case in enumerate(cases):
        if int(case.get("index", -1)) != index:
            raise CampaignError("Case indices are not canonical.")
        observed_vertices.add(
            (
                float(case["reynolds_number"]),
                float(case["reduced_frequency_k"]),
                float(case["amplitude_ratio"]),
            )
        )
        values = _motion_values(
            float(case["reynolds_number"]),
            float(case["reduced_frequency_k"]),
            float(case["amplitude_ratio"]),
            manifest["fluid_and_section"],
        )
        for key, expected in values.items():
            if case.get(key) != expected:
                raise CampaignError(f"Case {case['case_id']} {key} differs.")
        case_root = root / str(case["case_dir"])
        assets = _case_assets(case_root)
        if assets != case.get("assets") or canonical_fingerprint(assets) != case.get(
            "asset_fingerprint"
        ):
            raise CampaignError(f"Case {case['case_id']} asset identity differs.")
        sim = _relative_member(root, case["sim"], "QBlade simulation")
        turbine = case_root / "model" / "naca0018_prescribed_motion.trb"
        _validate_no_callback(turbine)
        expected_keywords = {
            "MOTIONFILE": "motion.dat",
            "FILTERFILE": None,
            "INCLUDEAERO": "1",
            "INCLUDEHYDRO": "0",
            "INITIAL_YAW": "0",
            "INITIAL_PITCH": "0",
            "INITIAL_AZIMUTH": "0",
            "RPMPRESCRIBED": "0",
            "HORANGLE": "0",
            "VERTANGLE": "0",
            "STORECONTROLLER": "false",
        }
        for keyword, expected in expected_keywords.items():
            if read_qblade_keyword(sim, keyword) != expected:
                raise CampaignError(
                    f"Case {case['case_id']} QBlade keyword {keyword} differs."
                )
        if int(read_qblade_keyword(sim, "NUMTIMESTEPS") or -1) != int(
            case["wrapper_steps"]
        ):
            raise CampaignError(f"Case {case['case_id']} wrapper step count differs.")
    if observed_vertices != expected_vertices:
        raise CampaignError("The prescribed-motion vertex matrix differs.")
    return {"valid": True, "root": root, "manifest": manifest}


def command_for_case(
    campaign: Path,
    *,
    case_index: int,
    qblade_root: Path,
    python_executable: str,
    resume: bool,
) -> list[str]:
    state = validate_campaign(campaign, qblade_root)
    cases = state["manifest"]["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise CampaignError(f"case_index must be in [0,{len(cases) - 1}].")
    case = cases[case_index]
    root = state["root"]
    case_root = root / str(case["case_dir"])
    command = [
        str(python_executable),
        str(REPO_ROOT / "run_headless/run_qblade_sim.py"),
        "--sim",
        str(root / str(case["sim"])),
        "--qblade-root",
        str(qblade_root.expanduser().resolve()),
        "--output-dir",
        str(root / str(case["result_dir"])),
        "--result-name",
        str(case["result_name"]),
        "--result-filter",
        str(root / str(case["result_filter"])),
        "--steps",
        str(case["wrapper_steps"]),
        "--threads",
        "1",
        "--coupling-mode",
        "none",
        "--case-metadata",
        str(case_root / "case_metadata.json"),
        "--progress-every",
        "500",
    ]
    if resume:
        command.extend(
            (
                "--checkpoint-every",
                "1000",
                "--resume-from-checkpoint",
            )
        )
    return command


def run_case(**kwargs: Any) -> None:
    command = command_for_case(**kwargs)
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=environment)


def _normalized_columns(path: Path) -> dict[str, np.ndarray]:
    raw = read_qblade_result_numeric_columns(path)
    normalized: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        canonical = normalize_qblade_header(name)
        if canonical in normalized:
            raise CampaignError(f"Duplicate normalized QBlade channel: {canonical}")
        normalized[canonical] = np.asarray(values, dtype=float)
    return normalized


def _clean_summary(path: Path, case: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CampaignError(f"QBlade run summary is unavailable: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("failed") is not False
        or summary.get("stopped") is not False
        or summary.get("exported_results") is not True
        or int(summary.get("requested_steps", -1)) != int(case["wrapper_steps"])
        or int(summary.get("completed_steps", -1)) != int(case["wrapper_steps"])
        or int(summary.get("threads", -1)) != 1
        or summary.get("coupling_mode") != "none"
        or summary.get("sil_vorlap") is not False
        or summary.get("sil_constant_force") is not False
    ):
        raise CampaignError(
            f"QBlade case {case['case_id']} did not finish natively and cleanly."
        )
    callback = summary.get("native_callback_evidence")
    if not isinstance(callback, Mapping):
        raise CampaignError("QBlade run summary lacks native callback evidence.")
    if (
        callback.get("callback_block_present") is not False
        or int(callback.get("update_init_count", 0)) != 0
        or int(callback.get("update_count_observed_max", 0)) != 0
        or callback.get("runtime_initialized_message_seen") is not False
    ):
        raise CampaignError(f"Case {case['case_id']} executed an external callback.")
    reproducibility = summary.get("reproducibility")
    library = (
        reproducibility.get("qblade", {}).get("library", {})
        if isinstance(reproducibility, Mapping)
        else {}
    )
    if library.get("sha256") != EXPECTED_QBLADE_LIBRARY_SHA256:
        raise CampaignError("Completed case used a different QBlade library.")
    return summary


def _write_canonical_history(
    path: Path,
    case: Mapping[str, Any],
    force: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestep = float(case["time_step_s"])
    amplitude = float(case["amplitude_m"])
    omega = 2.0 * math.pi * float(case["frequency_hz"])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HISTORY_COLUMNS)
        for index, force_value in enumerate(force, start=1):
            time = index * timestep
            displacement = amplitude * math.sin(omega * time)
            velocity = amplitude * omega * math.cos(omega * time)
            writer.writerow(
                (
                    f"{time:.17g}",
                    f"{displacement:.17g}",
                    f"{velocity:.17g}",
                    f"{float(force_value):.17g}",
                )
            )


def postprocess_case(campaign: Path, case_index: int) -> dict[str, Any]:
    state = validate_campaign(campaign)
    cases = state["manifest"]["cases"]
    if case_index < 0 or case_index >= len(cases):
        raise CampaignError(f"case_index must be in [0,{len(cases) - 1}].")
    case = cases[case_index]
    root = state["root"]
    result_dir = root / str(case["result_dir"])
    result = result_dir / f"{case['result_name']}.txt"
    summary_path = result_dir / f"{Path(str(case['sim'])).stem}_run_summary.json"
    summary = _clean_summary(summary_path, case)
    if not result.is_file():
        raise CampaignError(f"QBlade result is unavailable: {result}")
    identity = summary.get("result_file_identity", {})
    if (
        identity.get("sha256") != _sha256(result)
        or int(identity.get("size_bytes", -1)) != result.stat().st_size
    ):
        raise CampaignError("QBlade result identity differs from its run summary.")

    columns = _normalized_columns(result)
    missing = [name for name in FILTER_CHANNELS if name not in columns]
    if missing:
        raise CampaignError(
            "QBlade result lacks required channel(s): " + ", ".join(missing)
        )
    count = int(case["expected_result_rows"])
    raw_count = int(case["wrapper_steps"])
    if any(len(columns[name]) != raw_count for name in FILTER_CHANNELS):
        observed = {name: len(columns[name]) for name in FILTER_CHANNELS}
        raise CampaignError(
            f"QBlade result row count differs for {case['case_id']}: {observed}"
        )
    if not all(np.all(np.isfinite(columns[name])) for name in FILTER_CHANNELS):
        raise CampaignError("QBlade result contains non-finite required values.")

    timestep = float(case["time_step_s"])
    if abs(float(columns[TIME_CHANNEL][0])) > 2.0e-11:
        raise CampaignError("QBlade result does not begin with the wrapper t=0 row.")
    columns = {name: values[1:] for name, values in columns.items()}
    if any(len(columns[name]) != count for name in FILTER_CHANNELS):
        raise CampaignError("Discarding the wrapper t=0 row did not yield j=1..13N+1.")
    expected_time = np.arange(1, count + 1, dtype=float) * timestep
    raw_time = columns[TIME_CHANNEL]
    time_tolerance = np.maximum(2.0e-11, 6.0e-6 * np.maximum(expected_time, timestep))
    if np.any(np.abs(raw_time - expected_time) > time_tolerance):
        raise CampaignError("QBlade exported timestamps do not follow j*dt.")
    amplitude = float(case["amplitude_m"])
    omega = 2.0 * math.pi * float(case["frequency_hz"])
    expected_q = amplitude * np.sin(omega * expected_time)
    prescribed_q = columns[PLATFORM_Y_CHANNEL]
    motion_error = float(np.max(np.abs(prescribed_q - expected_q)))
    if motion_error > max(2.0e-9, 2.0e-5 * amplitude):
        raise CampaignError(
            "QBlade prescribed platform motion differs from the analytic motion."
        )
    tip_y = columns[BLADE_TIP_Y_CHANNEL]
    tip_motion_error = float(
        np.max(np.abs((tip_y - tip_y[0]) - (prescribed_q - prescribed_q[0])))
    )
    if tip_motion_error > 2.0e-6:
        raise CampaignError(
            "The effectively rigid blade does not follow platform global Y."
        )
    if float(np.max(np.abs(columns["LSS Rpm [rpm]"]))) > 1.0e-8:
        raise CampaignError("The prescribed-motion section rotated.")
    if float(np.max(np.abs(columns["True Pitch Angle BLD_1 [deg]"]))) > 1.0e-8:
        raise CampaignError("The prescribed-motion section pitch changed.")
    if float(np.max(np.abs(columns["True Yaw Angle [deg]"]))) > 1.0e-8:
        raise CampaignError("The prescribed-motion section yaw changed.")

    aoa = columns[AOA_CHANNEL]
    median_abs_aoa = float(np.median(np.abs(aoa)))
    max_abs_aoa_deviation = float(np.max(np.abs(np.abs(aoa) - 90.0)))
    if abs(median_abs_aoa - 90.0) > 2.0 or max_abs_aoa_deviation > 12.0:
        raise CampaignError("QBlade AoA does not support the audited +90/-90 mapping.")
    reynolds = columns[REYNOLDS_CHANNEL]
    median_reynolds = float(np.median(reynolds))
    if (
        abs(median_reynolds - float(case["reynolds_number"]))
        / float(case["reynolds_number"])
        > 0.03
    ):
        raise CampaignError(
            "QBlade panel Reynolds number differs from the frozen vertex."
        )
    force = columns[FORCE_CHANNEL]
    canonical = root / str(case["canonical_history"]["path"])
    _write_canonical_history(canonical, case, force)
    history_identity = {
        **case["canonical_history"],
        "status": "complete",
        "sha256": _sha256(canonical),
        "rows": count,
    }
    audit = {
        "schema": "vorlap-naca0018-qblade-prescribed-motion-case-audit-v1",
        "status": "complete",
        "case_id": case["case_id"],
        "campaign_fingerprint": state["manifest"]["fingerprint"],
        "run_summary": _identity(summary_path, root),
        "raw_result": _identity(result, root),
        "canonical_history": history_identity,
        "runtime": {
            "qblade_library_sha256": EXPECTED_QBLADE_LIBRARY_SHA256,
            "qblade_runtime_dependency_fingerprint": EXPECTED_QBLADE_DEPENDENCY_FINGERPRINT,
            "threads": 1,
            "coupling_mode": "none",
            "external_callback_updates": 0,
        },
        "channel_contract": {
            "raw_export_rows": raw_count,
            "discarded_wrapper_t0_rows": 1,
            "prescribed_motion": PLATFORM_Y_CHANNEL,
            "rigid_blade_position": BLADE_TIP_Y_CHANNEL,
            "crossflow_force": FORCE_CHANNEL,
            "crossflow_force_units": "N_total_over_4m_span",
            "crossflow_force_sign": "positive_fluid_on_body_positive_global_Y",
            "qblade_aoa": AOA_CHANNEL,
            "qblade_reynolds": REYNOLDS_CHANNEL,
        },
        "geometry_frame_audit": {
            "qblade_internal_incidence_target_deg": EXPECTED_QBLADE_INCIDENCE_DEG,
            "vorlap_incidence_deg": EXPECTED_VORLAP_INCIDENCE_DEG,
            "median_abs_qblade_aoa_deg": median_abs_aoa,
            "max_abs_qblade_aoa_deviation_from_90_deg": max_abs_aoa_deviation,
            "median_qblade_panel_reynolds": median_reynolds,
            "analytic_platform_max_abs_error_m": motion_error,
            "tip_follows_platform_max_abs_error_m": tip_motion_error,
        },
        "force_statistics_n": {
            "minimum": float(np.min(force)),
            "maximum": float(np.max(force)),
            "mean": float(np.mean(force)),
            "rms": float(np.sqrt(np.mean(force**2))),
        },
    }
    _write_json(result_dir / "prescribed_motion_audit.json", audit)
    return audit


def _completed_audit(
    root: Path, manifest: Mapping[str, Any], case: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = root / str(case["result_dir"]) / "prescribed_motion_audit.json"
    if not path.is_file():
        return None
    audit = json.loads(path.read_text(encoding="utf-8"))
    history = audit.get("canonical_history", {})
    history_path = root / str(history.get("path", ""))
    if (
        audit.get("schema") != "vorlap-naca0018-qblade-prescribed-motion-case-audit-v1"
        or audit.get("status") != "complete"
        or audit.get("case_id") != case["case_id"]
        or audit.get("campaign_fingerprint") != manifest["fingerprint"]
        or history.get("status") != "complete"
        or int(history.get("rows", -1)) != int(case["expected_result_rows"])
        or int(history.get("rows_expected", -1)) != int(case["expected_result_rows"])
        or history.get("columns") != list(HISTORY_COLUMNS)
        or not history_path.is_file()
        or history.get("sha256") != _sha256(history_path)
    ):
        raise CampaignError(f"Completed audit differs for {case['case_id']}.")
    return audit


def validate_preflight_evidence(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the executed one-case gate before matrix scheduling/finalization."""

    case = manifest["cases"][PREFLIGHT_CASE_INDEX]
    if case["case_id"] != PREFLIGHT_CASE_ID:
        raise CampaignError("The designated QBlade preflight case differs.")
    audit = _completed_audit(root, manifest, case)
    if audit is None:
        raise CampaignError(
            f"Preflight case {PREFLIGHT_CASE_ID} lacks a complete case audit."
        )
    audit_path = root / str(case["result_dir"]) / "prescribed_motion_audit.json"
    report_path = root / PREFLIGHT_SCHEDULER_REPORT
    if report_path.is_symlink() or not report_path.is_file():
        raise CampaignError("The executed preflight scheduler report is unavailable.")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError("The preflight scheduler report is not valid JSON.") from exc
    gate = report.get("preflight_gate")
    allocation = report.get("allocation")
    environment = report.get("execution_environment")
    environment_valid = False
    if isinstance(environment, Mapping):
        if environment.get("kind") == "flight_slurm":
            environment_valid = (
                environment.get("platform") == "linux_x86_64"
                and isinstance(environment.get("slurm_job_id"), str)
                and environment.get("slurm_job_id").isdigit()
            )
        elif environment.get("kind") == "local_docker":
            image_id = environment.get("container_image_id")
            cpu_limit = environment.get("container_cpu_limit")
            environment_valid = (
                environment.get("platform") == "linux_amd64"
                and isinstance(image_id, str)
                and len(image_id) == 71
                and image_id.startswith("sha256:")
                and all(ch in "0123456789abcdef" for ch in image_id[7:])
                and type(cpu_limit) is int
                and 1 <= cpu_limit <= 8
            )
    if (
        report.get("schema") != PREFLIGHT_SCHEDULER_SCHEMA
        or report.get("stage") != "preflight"
        or report.get("status") != "complete"
        or report.get("campaign_fingerprint") != manifest["fingerprint"]
        or not isinstance(gate, Mapping)
        or gate.get("status") != "complete"
        or gate.get("case_index") != PREFLIGHT_CASE_INDEX
        or gate.get("case_id") != PREFLIGHT_CASE_ID
        or not isinstance(allocation, Mapping)
        or allocation.get("workers") != 1
        or allocation.get("threads_per_case") != 1
        or allocation.get("allocated_physical_cpus") != 1
        or not environment_valid
    ):
        raise CampaignError(
            "The preflight scheduler report does not prove the explicit one-case gate."
        )
    completed_rows = [
        row
        for row in report.get("cases", [])
        if isinstance(row, Mapping)
        and row.get("index") == PREFLIGHT_CASE_INDEX
        and row.get("case_id") == PREFLIGHT_CASE_ID
        and row.get("status") == "complete"
    ]
    if len(completed_rows) != 1:
        raise CampaignError(
            "The preflight scheduler report lacks the unique completed gate case."
        )
    return {
        "status": "complete",
        "case_index": PREFLIGHT_CASE_INDEX,
        "case_id": PREFLIGHT_CASE_ID,
        "case_audit": _identity(audit_path, root),
        "scheduler_report": _identity(report_path, root),
        "scheduler_schema": report["schema"],
        "scheduler_stage": report["stage"],
        "scheduler_status": report["status"],
        "execution_environment": dict(environment),
    }


def campaign_status(campaign: Path) -> dict[str, Any]:
    state = validate_campaign(campaign)
    root = state["root"]
    manifest = state["manifest"]
    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        audit = _completed_audit(root, manifest, case)
        result_dir = root / str(case["result_dir"])
        summary = result_dir / f"{Path(str(case['sim'])).stem}_run_summary.json"
        rows.append(
            {
                "index": case["index"],
                "case_id": case["case_id"],
                "status": "complete" if audit is not None else "runnable",
                "run_summary_present": summary.is_file(),
                "canonical_history": (
                    case["canonical_history"]
                    if audit is None
                    else audit["canonical_history"]
                ),
            }
        )
    complete = sum(row["status"] == "complete" for row in rows)
    return {
        "schema": "vorlap-naca0018-qblade-prescribed-motion-status-v1",
        "campaign_fingerprint": manifest["fingerprint"],
        "complete_cases": complete,
        "required_cases": len(rows),
        "runnable_cases": len(rows) - complete,
        "cases": rows,
    }


def finalize_results(campaign: Path) -> Path:
    state = validate_campaign(campaign)
    root = state["root"]
    manifest = state["manifest"]
    status = campaign_status(root)
    if status["complete_cases"] != 48:
        raise CampaignError(
            f"Cannot finalize: {status['complete_cases']}/48 QBlade cases are complete."
        )
    preflight_evidence = validate_preflight_evidence(root, manifest)
    rows: list[dict[str, Any]] = []
    for case, status_row in zip(manifest["cases"], status["cases"]):
        audit_path = root / str(case["result_dir"]) / "prescribed_motion_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "case_id": case["case_id"],
                "incidence_deg": case["incidence_deg"],
                "reynolds_number": case["reynolds_number"],
                "reduced_frequency_k": case["reduced_frequency_k"],
                "amplitude_ratio": case["amplitude_ratio"],
                "direction_deg_mod_180": case["direction_deg_mod_180"],
                "discard_cycles": case["discard_cycles"],
                "canonical_history": status_row["canonical_history"],
                "case_audit": _identity(audit_path, root),
            }
        )
    result = {
        "schema": RESULTS_SCHEMA,
        "status": "complete",
        "campaign_manifest": "campaign_manifest.json",
        "campaign_fingerprint": manifest["fingerprint"],
        "request": manifest["request"],
        "fluid_and_section": manifest["fluid_and_section"],
        "runtime": manifest["runtime"],
        "geometry_and_frames": manifest["geometry_and_frames"],
        "native_aero": manifest["native_aero"],
        "external_forces": manifest["external_forces"],
        "history_contract": manifest["history_contract"],
        "preflight_gate": {
            "campaign_contract": manifest["preflight_gate"],
            **preflight_evidence,
        },
        "summary": {
            "required_cases": 48,
            "complete_cases": 48,
            "failed_cases": 0,
        },
        "cases": rows,
    }
    path = root / "qblade_results.json"
    _write_json(path, result)
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--campaign", type=Path, required=True)
    generate.add_argument("--qblade-root", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--campaign", type=Path, required=True)
    validate.add_argument("--qblade-root", type=Path, default=None)

    command = subparsers.add_parser("command")
    command.add_argument("--campaign", type=Path, required=True)
    command.add_argument("--case-index", type=int, required=True)
    command.add_argument("--qblade-root", type=Path, required=True)
    command.add_argument("--python-executable", default=sys.executable)
    command.add_argument("--resume", action="store_true")

    run = subparsers.add_parser("run-case")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument("--case-index", type=int, required=True)
    run.add_argument("--qblade-root", type=Path, required=True)
    run.add_argument("--python-executable", default=sys.executable)
    run.add_argument("--resume", action="store_true")

    post = subparsers.add_parser("postprocess-case")
    post.add_argument("--campaign", type=Path, required=True)
    post.add_argument("--case-index", type=int, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--campaign", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--campaign", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        value = generate_campaign(args.campaign, args.qblade_root)
        print(
            json.dumps(
                {"path": str(args.campaign), "fingerprint": value["fingerprint"]},
                indent=2,
            )
        )
    elif args.command == "validate":
        value = validate_campaign(args.campaign, args.qblade_root)
        print(
            json.dumps(
                {
                    "valid": value["valid"],
                    "case_count": len(value["manifest"]["cases"]),
                    "fingerprint": value["manifest"]["fingerprint"],
                },
                indent=2,
            )
        )
    elif args.command == "command":
        print(
            " ".join(
                command_for_case(
                    args.campaign,
                    case_index=args.case_index,
                    qblade_root=args.qblade_root,
                    python_executable=args.python_executable,
                    resume=args.resume,
                )
            )
        )
    elif args.command == "run-case":
        run_case(
            campaign=args.campaign,
            case_index=args.case_index,
            qblade_root=args.qblade_root,
            python_executable=args.python_executable,
            resume=args.resume,
        )
    elif args.command == "postprocess-case":
        print(json.dumps(postprocess_case(args.campaign, args.case_index), indent=2))
    elif args.command == "status":
        print(json.dumps(campaign_status(args.campaign), indent=2))
    elif args.command == "finalize":
        print(finalize_results(args.campaign))
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
