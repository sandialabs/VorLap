#!/usr/bin/env python3
"""Generate the Grinderslev et al. (2022) IEA-10-MW single-blade bridge case.

The public IEA-10-MW WindIO model supplies blade geometry and the official
HAWC2 set-1 table supplies the 19 Timoshenko structural sections used to build
18 beam elements.  The generated VorLap sources are an explicitly synthetic,
spatially coherent first-edgewise-mode forcing probe at 0.67 Hz.  They are not
the EllipSys3D loads and cannot recreate wake lock-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.qblade_case_prep import QBladeCasePrepRequest, prepare_qblade_external_case
from vorlap.qblade_conversion import convert_qblade_to_vorlap_inputs
from vorlap.qblade_files import replace_qblade_keyword_text
from vorlap.geometry_frames import prepare_component_frames


EXPECTED_SOURCE_COMMIT = "350cf06f5db9cec12bcd73c2b174a5d9642958d8"
SOURCE_REPOSITORY = "https://github.com/IEAWindSystems/IEA-10.0-198-RWT"
REFERENCE_DOI = "10.5194/wes-7-2201-2022"
CORRIGENDUM_DOI = "10.5194/wes-7-2201-2022-corrigendum"
CASE_NAME = "grinderslev2022_p100_i50_iea10_single_blade"
FREQUENCY_HZ = 0.67
INFLOW_SPEED_MPS = 18.0
PITCH_ANGLE_DEG = 100.0
INCLINATION_ANGLE_DEG = 50.0
TIMESTEP_S = 0.006
DEFAULT_DURATION_S = 12.0
PROBE_CHORDWISE_AMPLITUDE = 0.08
AIR_DENSITY_KGPM3 = 1.225
AIR_KINEMATIC_VISCOSITY_M2PS = 1.48e-5
EDGE_MODE_COEFFICIENTS = np.array(
    [
        0.26823590819889026,
        1.7185652699151985,
        -3.7976587747017136,
        5.201943784912145,
        -2.3910861883245196,
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iea-root",
        required=True,
        type=Path,
        help="Pinned checkout of IEAWindSystems/IEA-10.0-198-RWT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_case",
        help="Generated case directory.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help="Simulation duration in seconds (default: 12).",
    )
    parser.add_argument(
        "--allow-source-commit-mismatch",
        action="store_true",
        help="Allow generation from a different IEA model commit and record it in the manifest.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def load_blade_model(source_yaml: Path) -> dict[str, Any]:
    # The legacy monolithic WindIO file reuses anchors after the blade block in
    # a way that recent PyYAML rejects.  The blade mapping is self-contained up
    # to the hub entry, so load only that auditable prefix.
    text = source_yaml.read_text(encoding="utf-8")
    blade_prefix = text.split("    hub:", 1)[0]
    model = yaml.safe_load(blade_prefix)
    return model["components"]["blade"]


def edge_mode(normalized_span: np.ndarray) -> np.ndarray:
    x = np.asarray(normalized_span, dtype=float)
    mode = sum(coef * x ** power for power, coef in enumerate(EDGE_MODE_COEFFICIENTS, start=2))
    tip = float(mode[-1])
    if not math.isfinite(tip) or abs(tip) < 1.0e-12:
        raise ValueError("The public OpenFAST first-edgewise mode is not tip-normalizable.")
    return mode / tip


def load_openfast_edge_damping_ratio(path: Path) -> float:
    """Read the public first-edgewise modal damping percentage as a ratio."""
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "BldEdDmp1" not in raw:
            continue
        value_percent = float(raw.split()[0])
        ratio = value_percent / 100.0
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(f"Invalid BldEdDmp1 value in {path}: {value_percent}")
        return ratio
    raise ValueError(f"BldEdDmp1 is missing from {path}.")


def published_inflow_in_qblade_frame(component: Any) -> tuple[np.ndarray, float, float, dict[str, float]]:
    """Map the published root-frame P/I definition to QBlade global angles."""
    prepare_component_frames([component])
    chord = np.asarray(component.chord_vector[0], dtype=float)
    normal = np.asarray(component.normal_vector[0], dtype=float)
    span = np.cross(chord, normal)
    span /= np.linalg.norm(span)

    pitch_rad = math.radians(PITCH_ANGLE_DEG)
    inclination_rad = math.radians(INCLINATION_ANGLE_DEG)
    projected = math.cos(pitch_rad) * (-chord) + math.sin(pitch_rad) * normal
    inflow = math.cos(inclination_rad) * projected - math.sin(inclination_rad) * span
    inflow /= np.linalg.norm(inflow)

    horizontal_deg = math.degrees(math.atan2(float(inflow[1]), float(inflow[0])))
    vertical_deg = math.degrees(math.asin(float(inflow[2])))
    in_plane = inflow - float(np.dot(inflow, span)) * span
    in_plane /= np.linalg.norm(in_plane)
    recovered_pitch_deg = math.degrees(
        math.atan2(float(np.dot(in_plane, normal)), float(np.dot(in_plane, -chord)))
    )
    recovered_inclination_deg = math.degrees(math.asin(float(-np.dot(inflow, span))))
    audit = {
        "target_pitch_deg": PITCH_ANGLE_DEG,
        "target_inclination_deg": INCLINATION_ANGLE_DEG,
        "recovered_pitch_deg": recovered_pitch_deg,
        "recovered_inclination_deg": recovered_inclination_deg,
        "qblade_horizontal_angle_deg": horizontal_deg,
        "qblade_vertical_angle_deg": vertical_deg,
    }
    if not math.isclose(recovered_pitch_deg, PITCH_ANGLE_DEG, abs_tol=1.0e-10):
        raise ValueError("Root-frame pitch-angle mapping failed.")
    if not math.isclose(recovered_inclination_deg, INCLINATION_ANGLE_DEG, abs_tol=1.0e-10):
        raise ValueError("Root-frame inclination-angle mapping failed.")
    return inflow, horizontal_deg, vertical_deg, audit


def patch_keywords(text: str, values: dict[str, object]) -> str:
    for keyword, value in values.items():
        text, _old, count = replace_qblade_keyword_text(text, keyword, value)
        if count != 1:
            raise ValueError(f"Could not patch required QBlade keyword {keyword}.")
    return text


def write_blade_definition(
    path: Path,
    *,
    span_m: np.ndarray,
    chord_m: np.ndarray,
    twist_deg: np.ndarray,
    pitch_axis: np.ndarray,
    prebend_m: np.ndarray,
    sweep_m: np.ndarray,
    hub_radius_m: float,
) -> list[str]:
    airfoil_ids = [f"IEA10_STATION_{index:02d}" for index in range(len(span_m))]
    lines = [
        "----------------------------------------QBlade Blade Definition File------------------------------------------------",
        "Generated by examples/grinderslev2022_iea10/generate_case.py",
        "----------------------------------------Object Name-----------------------------------------------------------------",
        f"{CASE_NAME}_blade OBJECTNAME",
        "----------------------------------------Parameters------------------------------------------------------------------",
        "HAWT ROTORTYPE - rotor type",
        "1 NUMBLADES - isolated single blade",
        "----------------------------------------Blade Data------------------------------------------------------------------",
        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] OFFSET_Y_[m] TAXIS_[-] POLAR_FILE",
    ]
    for index, values in enumerate(zip(span_m, chord_m, twist_deg, sweep_m, prebend_m, pitch_axis)):
        span, chord, twist, sweep, prebend, axis = values
        lines.append(
            f"{hub_radius_m + span:.10e} {chord:.10e} {twist:.10e} "
            f"{sweep:.10e} {prebend:.10e} {axis:.10e} Polars/{airfoil_ids[index]}.plr"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return airfoil_ids


def load_hawc2_structure(path: Path, *, set_number: int = 1) -> np.ndarray:
    """Load one numbered 19-column HAWC2 Timoshenko property set."""
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = f"${set_number} "
    for index, line in enumerate(lines):
        if line.strip().startswith(marker):
            fields = line.split()
            count = int(fields[1])
            rows = []
            for candidate in lines[index + 1 :]:
                stripped = candidate.strip()
                if not stripped or stripped.startswith("="):
                    continue
                if stripped.startswith("$"):
                    break
                values = [float(value) for value in stripped.split()]
                if len(values) != 19:
                    break
                rows.append(values)
                if len(rows) == count:
                    break
            data = np.asarray(rows, dtype=float)
            if data.shape != (count, 19):
                raise ValueError(f"Could not read HAWC2 structural set {set_number} from {path}.")
            return data
    raise ValueError(f"HAWC2 structural set {set_number} is missing from {path}.")


def write_hawc2_timoshenko_blade(
    path: Path,
    *,
    hawc2_structure: np.ndarray,
    aero_radius_m: np.ndarray,
    aero_chord_m: np.ndarray,
    rayleigh_beta_s: float,
) -> None:
    if hawc2_structure.ndim != 2 or hawc2_structure.shape[1] != 19:
        raise ValueError("Expected a HAWC2 structural table with 19 columns.")
    radius = hawc2_structure[:, 0]
    if np.any(np.diff(radius) <= 0.0) or radius[0] != 0.0:
        raise ValueError("HAWC2 structural radii must start at zero and increase strictly.")
    chord = np.interp(radius, aero_radius_m, aero_chord_m)
    if np.any(chord <= 0.0):
        raise ValueError("Interpolated HAWC2 structural chord must remain positive.")

    mass = hawc2_structure[:, 1]
    x_cg, y_cg = hawc2_structure[:, 2], hawc2_structure[:, 3]
    ri_x, ri_y = hawc2_structure[:, 4], hawc2_structure[:, 5]
    x_sh, y_sh = hawc2_structure[:, 6], hawc2_structure[:, 7]
    youngs, shear = hawc2_structure[:, 8], hawc2_structure[:, 9]
    i_x, i_y, polar_i = hawc2_structure[:, 10], hawc2_structure[:, 11], hawc2_structure[:, 12]
    k_x, k_y, area = hawc2_structure[:, 13], hawc2_structure[:, 14], hawc2_structure[:, 15]
    structural_pitch = hawc2_structure[:, 16]
    x_elastic, y_elastic = hawc2_structure[:, 17], hawc2_structure[:, 18]
    properties = np.column_stack(
        [
            radius / radius[-1],
            mass,
            # Both tables define EI about the named section axis.  QBlade's
            # +Y local translation is chordwise, so its bending is governed
            # by EIx (bending about local X).  Preserve the HAWC2 Ix/Iy order;
            # swapping these terms incorrectly moved the 0.67 Hz family to
            # QBlade's normal rather than chordwise response channel.
            youngs * i_x,
            youngs * i_y,
            youngs * area,
            shear * polar_i,
            shear * area,
            # QBlade's section frame is already rotated by the WindIO blade
            # twist.  HAWC2's principal-axis angle uses a different cross-
            # section convention and cannot be transferred verbatim (doing so
            # swaps the chordwise and normal modal families).  Keep the QBlade
            # section axes on the converted blade frame and record the
            # reduction in the manifest.
            np.zeros_like(structural_pitch),
            k_x,
            k_y,
            ri_x / chord,
            ri_y / chord,
            x_cg / chord,
            y_cg / chord,
            # Keep the QBlade beam reference on the public WindIO blade axis
            # used by the bridge geometry.  Applying HAWC2's elastic-center
            # offsets here moves QBlade sensor/action locations away from the
            # aerodynamic line by up to O(0.5 m), which breaks node-level load
            # transfer.  The omission is recorded as a structural reduction;
            # mass and shear-center offsets remain explicit below.
            np.zeros_like(x_elastic),
            np.zeros_like(y_elastic),
            x_sh / chord,
            y_sh / chord,
        ]
    )
    if not np.all(np.isfinite(properties)) or np.any(properties[:, 1:7] <= 0.0):
        raise ValueError("Converted HAWC2 Timoshenko properties must be finite and positive.")
    header = [
        f"{rayleigh_beta_s:.12g} RAYLEIGHDMP - stiffness-proportional beta [s]",
        "1.0 STIFFTUNER",
        "1.0 MASSTUNER",
        "0 INTPTYPE - linear interpolation",
        "1 BEAMTYPE - Timoshenko",
        "2 DISCTYPE - structural stations",
        f"{len(radius)} DISC",
        "LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] EA_[N] GJ_[N.m^2] GA_[N] "
        "STRPIT_[deg] KSX_[-] KSY_[-] RGX_[-] RGY_[-] XCM_[-] YCM_[-] XCE_[-] YCE_[-] XCS_[-] YCS_[-]",
    ]
    rows = [" ".join(f"{value:.12e}" for value in row) for row in properties]
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def write_rigid_tower(path: Path) -> None:
    rows = [
        "0.0 RAYLEIGHDMP",
        "1.0 STIFFTUNER",
        "1.0 MASSTUNER",
        "0 INTPTYPE",
        "1 BEAMTYPE - Timoshenko",
        "0 DISCTYPE - linear",
        "2 DISC",
        "LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] EA_[N] GJ_[N.m^2] GA_[N] STRPIT_[deg] KSX_[-] KSY_[-] RGX_[-] RGY_[-] XCM_[-] YCM_[-] XCE_[-] YCE_[-] XCS_[-] YCS_[-] Diameter_[m]",
    ]
    section = "1.0 1.0e15 1.0e15 1.0e15 1.0e15 1.0e15 0.0 0.85 0.85 0.25 0.25 0.0 0.0 0.0 0.0 0.0 0.0 1.0"
    rows.extend((f"0.0 {section}", f"1.0 {section}"))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_main_structure(path: Path, normalized_span: np.ndarray) -> None:
    sensor_positions = sorted(set(float(value) for value in normalized_span))
    lines = [
        "---------------------- QBLADE STRUCTURAL MODEL INPUT FILE -----------------",
        "0.2 GLBGEOEPS",
        "------------------------------- HAWT TURBINE CONFIGURATION ----------------",
        "0.0 PRECONE",
        "0.0 SHFTTILT",
        "0.0 OVERHANG",
        "0.0 TWR2SHFT",
        "------------------------------- MASS AND INERTIA --------------------------",
        "0.0 YAWBRMASS",
        "0.0 NACMASS",
        "0.0 NACCMX",
        "0.0 NACCMY",
        "0.0 NACCMZ",
        "0 0 0 0 0 0 NACINER",
        "0.0 NACYINER",
        "0.0 HUBMASS",
        "0.0 HUBCM",
        "1.0 HUBINER",
        "------------------------------- DRIVETRAIN MODEL --------------------------",
        "1.0 GBRATIO",
        "1.0 GBOXEFF",
        "1.0 GENEFF",
        "false DRTRDOF",
        "1.0 GENINER",
        "1.0e12 DTTORSPR",
        "1.0e9 DTTORDMP",
        "------------------------------- BLADES ------------------------------------",
        "1 NUMBLD",
        "iea10_blade_hawc2.str BLDFILE_1",
        "------------------------------- TOWER -------------------------------------",
        "1.0 TWRHEIGHT",
        "rigid_tower.str TWRFILE",
        "------------------------------- DATA OUTPUT TYPES -------------------------",
        "true FOR_OUT",
        "true ROT_OUT",
        "true MOM_OUT",
        "true DEF_OUT",
        "true POS_OUT",
        "true VEL_OUT",
        "true ACC_OUT",
        "true LVE_OUT",
        "true LAC_OUT",
        "------------------------------- DATA OUTPUT LOCATIONS ---------------------",
    ]
    lines.extend(f"BLD_1_{position:.10f}" for position in sensor_positions)
    lines.extend(("TWR_0.0", "TWR_1.0"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_placeholder_airfoil(path: Path) -> None:
    coords = [
        (1.0, 0.0019), (0.8, 0.0394), (0.6, 0.0685), (0.4, 0.0871), (0.2, 0.0861),
        (0.0, 0.0), (0.2, -0.0861), (0.4, -0.0871), (0.6, -0.0685), (0.8, -0.0394), (1.0, -0.0019),
    ]
    lines = ["NACA 0018 import placeholder; QBlade native aerodynamics disabled"]
    lines.extend(f"{x:.5f} {y:.5f}" for x, y in coords)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_placeholder_polar(path: Path, airfoil_id: str) -> None:
    text = f"""----------------------------------------QBlade Multi RE Polar File--------------------------------------------------
Generated by examples/grinderslev2022_iea10/generate_case.py
----------------------------------------Object Names----------------------------------------------------------------
{airfoil_id}_import_placeholder POLARNAME
../Airfoils/NACA0018_placeholder.afl FOILNAME
----------------------------------------Parameters------------------------------------------------------------------
18.0 THICKNESS
false ISDECOMPOSED
REYNOLDS 2.0000E+06
----------------------------------------Polar Data------------------------------------------------------------------
AOA CL CD CM
-180.0 0.0 1.0 0.0
-90.0 0.0 1.8 0.0
0.0 0.0 0.01 0.0
90.0 0.0 1.8 0.0
180.0 0.0 1.0 0.0
"""
    path.write_text(text, encoding="utf-8")


def write_probe_source(
    path: Path,
    *,
    airfoil_id: str,
    mode_amplitude: float,
    strouhal: float,
) -> float:
    shape = (2, 2, 2)
    zeros = np.zeros(shape, dtype=float)
    st = zeros.copy()
    st[:, :, 1] = strouhal
    chordwise_amp = zeros.copy()
    chordwise_amp[:, :, 1] = PROBE_CHORDWISE_AMPLITUDE * abs(mode_amplitude)
    with h5py.File(path, "w") as h5:
        h5.attrs["reynolds_model"] = "replicated_reynolds_independent"
        h5.attrs["source_scope"] = "synthetic_first_edgewise_mode_transfer_probe_not_cfd"
        h5.attrs["target_frequency_hz"] = FREQUENCY_HZ
        h5.create_dataset("Airfoilname", data=airfoil_id.encode("utf-8"))
        h5.create_dataset("Re", data=np.array([1.0e4, 1.0e8]))
        h5.create_dataset("Thickness", data=np.array([0.18]))
        h5.create_dataset("AOA", data=np.array([-180.0, 180.0]))
        for field in ("CL", "CD", "CM", "CF"):
            h5.create_dataset(f"{field}_ST", data=st if field == "CD" else zeros)
            h5.create_dataset(f"{field}_Amp", data=chordwise_amp if field == "CD" else zeros)
            h5.create_dataset(f"{field}_Pha", data=zeros)
    return strouhal


def write_simulation(
    path: Path,
    template: Path,
    turbine_dir_name: str,
    num_steps: int,
    *,
    horizontal_angle_deg: float,
    vertical_angle_deg: float,
) -> None:
    text = template.read_text(encoding="utf-8")
    text = patch_keywords(
        text,
        {
            "OBJECTNAME": CASE_NAME,
            "TURBFILE": f"{turbine_dir_name}/{CASE_NAME}.trb",
            "TURBNAME": CASE_NAME,
            "INITIAL_YAW": 0.0,
            "INITIAL_PITCH": 0.0,
            "INITIAL_AZIMUTH": 0.0,
            "STRSUBSTEP": 2,
            "RELAXSTEPS": 0,
            "PRESCRIBETYPE": 1,
            "RPMPRESCRIBED": 0.0,
            "INCLUDEAERO": 0,
            "TIMESTEP": TIMESTEP_S,
            "NUMTIMESTEPS": num_steps,
            "RAMPUP": 0.0,
            "ADDDAMP": 0.0,
            "WNDTYPE": 0,
            "WNDNAME": "",
            "MEANINF": INFLOW_SPEED_MPS,
            "HORANGLE": horizontal_angle_deg,
            "VERTANGLE": vertical_angle_deg,
            "PROFILETYPE": 0,
            "SHEAREXP": 0.0,
            "DENSITYAIR": AIR_DENSITY_KGPM3,
            "VISCOSITYAIR": AIR_KINEMATIC_VISCOSITY_M2PS,
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


def write_turbine(path: Path, template: Path) -> None:
    text = template.read_text(encoding="utf-8")
    text = patch_keywords(
        text,
        {
            "OBJECTNAME": CASE_NAME,
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
            "STRUCTURALFILE": "Structure/iea10_single_blade_main.str",
            "CONTROLLERTYPE": 0,
        },
    )
    path.write_text(text, encoding="utf-8")


def generate(args: argparse.Namespace) -> Path:
    source_root = args.iea_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_yaml = source_root / "yaml" / "IEA-10-198-RWT.yaml"
    source_mode = source_root / "OpenFAST" / "IEA-10.0-198-RWT_ElastoDyn_blade.dat"
    source_hawc2_structure = source_root / "hawc2" / "data" / "iea10mw_rc1_hawc2_blade_st.dat"
    source_license = source_root / "LICENSE"
    if (
        not source_yaml.is_file()
        or not source_mode.is_file()
        or not source_hawc2_structure.is_file()
        or not source_license.is_file()
    ):
        raise FileNotFoundError(
            "The IEA root does not contain the expected WindIO, HAWC2, OpenFAST, and license files."
        )
    source_commit = git_commit(source_root)
    if source_commit != EXPECTED_SOURCE_COMMIT and not args.allow_source_commit_mismatch:
        raise ValueError(
            f"IEA source commit {source_commit} does not match pinned {EXPECTED_SOURCE_COMMIT}; "
            "use --allow-source-commit-mismatch only for an explicitly reviewed update."
        )
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        raise ValueError("--duration must be finite and positive.")

    if output.exists():
        manifest = output / "benchmark_manifest.json"
        if output.name != "generated_case" and not manifest.is_file():
            raise ValueError(
                f"Refusing to replace unrecognized output directory {output}; "
                "use a new path or a prior generated benchmark directory."
            )
        shutil.rmtree(output)
    turbine_dir = output / CASE_NAME
    aero_dir = turbine_dir / "Aero"
    polar_dir = aero_dir / "Polars"
    foil_dir = aero_dir / "Airfoils"
    structure_dir = turbine_dir / "Structure"
    source_dir = turbine_dir / "VorLapAirfoils"
    for directory in (polar_dir, foil_dir, structure_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_license, output / "IEA_PUBLIC_MODEL_LICENSE.txt")

    blade = load_blade_model(source_yaml)
    outer = blade["outer_shape_bem"]
    normalized_span = np.asarray(outer["chord"]["grid"], dtype=float)
    chord_m = np.asarray(outer["chord"]["values"], dtype=float)
    twist_deg = np.rad2deg(np.asarray(outer["twist"]["values"], dtype=float))
    pitch_axis = np.asarray(outer["pitch_axis"]["values"], dtype=float)
    reference_axis = outer["reference_axis"]
    span_m = np.asarray(reference_axis["z"]["values"], dtype=float)
    prebend_m = np.asarray(reference_axis["x"]["values"], dtype=float)
    sweep_m = np.asarray(reference_axis["y"]["values"], dtype=float)
    hawc2_structure = load_hawc2_structure(source_hawc2_structure, set_number=1)
    edge_damping_ratio = load_openfast_edge_damping_ratio(source_mode)
    rayleigh_beta_s = edge_damping_ratio / (math.pi * FREQUENCY_HZ)
    arrays = (normalized_span, chord_m, twist_deg, pitch_axis, span_m, prebend_m, sweep_m)
    if any(len(array) != 30 for array in arrays):
        raise ValueError("The pinned public blade is expected to contain 30 aligned stations.")
    if np.any(np.diff(span_m) <= 0.0) or np.any(chord_m <= 0.0):
        raise ValueError("Blade span must increase strictly and chord must remain positive.")

    rotor_radius_m = 0.5 * 199.38249593364753
    hub_radius_m = rotor_radius_m - float(span_m[-1])
    mode_shape = edge_mode(normalized_span)

    blade_path = aero_dir / f"{CASE_NAME}.bld"
    airfoil_ids = write_blade_definition(
        blade_path,
        span_m=span_m,
        chord_m=chord_m,
        twist_deg=twist_deg,
        pitch_axis=pitch_axis,
        prebend_m=prebend_m,
        sweep_m=sweep_m,
        hub_radius_m=hub_radius_m,
    )
    write_placeholder_airfoil(foil_dir / "NACA0018_placeholder.afl")
    for index, airfoil_id in enumerate(airfoil_ids):
        write_placeholder_polar(polar_dir / f"{airfoil_id}.plr", airfoil_id)
    write_hawc2_timoshenko_blade(
        structure_dir / "iea10_blade_hawc2.str",
        hawc2_structure=hawc2_structure,
        aero_radius_m=span_m,
        aero_chord_m=chord_m,
        rayleigh_beta_s=rayleigh_beta_s,
    )
    write_rigid_tower(structure_dir / "rigid_tower.str")
    write_main_structure(structure_dir / "iea10_single_blade_main.str", normalized_span)

    sim_template = REPO_ROOT / "examples" / "ReferenceHVAWTQBladeModel" / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
    trb_template = REPO_ROOT / "examples" / "ReferenceHVAWTQBladeModel" / "10m_HVAWT_reference_turbine" / "10m_HVAWT_reference_turbine.trb"
    if not sim_template.is_file() or not trb_template.is_file():
        raise FileNotFoundError("The repository QBlade text templates are missing.")
    num_steps = int(round(args.duration / TIMESTEP_S))
    sim_path = output / f"{CASE_NAME}.sim"
    trb_path = turbine_dir / f"{CASE_NAME}.trb"
    # Write a temporary simulation so the canonical conversion can construct
    # the root section basis.  Then map the paper's pitch/inclination
    # definition into QBlade's global horizontal/vertical angles.
    write_simulation(
        sim_path,
        sim_template,
        turbine_dir.name,
        num_steps,
        horizontal_angle_deg=0.0,
        vertical_angle_deg=0.0,
    )
    write_turbine(trb_path, trb_template)

    preliminary_components, _preliminary_params, _preliminary_ids = convert_qblade_to_vorlap_inputs(
        str(sim_path), include_struts=False, include_tower=False
    )
    expected_inflow, horizontal_angle_deg, vertical_angle_deg, root_angle_audit = (
        published_inflow_in_qblade_frame(preliminary_components[0])
    )
    write_simulation(
        sim_path,
        sim_template,
        turbine_dir.name,
        num_steps,
        horizontal_angle_deg=horizontal_angle_deg,
        vertical_angle_deg=vertical_angle_deg,
    )

    components, params, converted_ids = convert_qblade_to_vorlap_inputs(
        str(sim_path), include_struts=False, include_tower=False
    )
    prepare_component_frames(components)
    strouhal_by_station = []
    for index, airfoil_id in enumerate(airfoil_ids):
        chord_vector = np.asarray(components[0].chord_vector[index], dtype=float)
        normal_vector = np.asarray(components[0].normal_vector[index], dtype=float)
        chordwise_speed = INFLOW_SPEED_MPS * float(np.dot(expected_inflow, chord_vector))
        normal_speed = INFLOW_SPEED_MPS * float(np.dot(expected_inflow, normal_vector))
        effective_speed = math.hypot(chordwise_speed, normal_speed)
        local_aoa_deg = math.degrees(math.atan2(normal_speed, chordwise_speed))
        reference_length_m = float(chord_m[index]) * max(
            abs(math.sin(math.radians(local_aoa_deg))), 0.18
        )
        strouhal = FREQUENCY_HZ * reference_length_m / effective_speed
        strouhal_by_station.append(
            write_probe_source(
                source_dir / f"{airfoil_id}.h5",
                airfoil_id=airfoil_id,
                mode_amplitude=float(mode_shape[index]),
                strouhal=strouhal,
            )
        )

    prep = prepare_qblade_external_case(
        QBladeCasePrepRequest(
            sim_path=sim_path,
            airfoil_dir=source_dir,
            include_struts=False,
            include_tower=False,
            node_source="structural",
            n_freq_depth=2,
            force_scale=1.0,
            include_mean_load=False,
            source_bounds_policy="raise",
            source_frequency_cap_hz=0.5 / TIMESTEP_S,
            write_geometry_audit=True,
        )
    )

    if len(components) != 1 or components[0].id != "BLD_1":
        raise ValueError("Generated case did not convert to exactly one blade component.")
    if len(converted_ids) != 30 or len(prep.node_ids) < 30:
        raise ValueError("Generated case did not retain all 30 blade stations/action locations.")
    if not np.allclose(params.rotation_axis, [1.0, 0.0, 0.0]):
        raise ValueError("Generated HAWT did not retain the hub-X rotation axis.")
    if not np.allclose(params.inflow_vec, expected_inflow, atol=1.0e-12):
        raise ValueError("Generated P/I inflow vector does not match the documented coordinate mapping.")

    geometry_table = output / "iea10_blade_geometry.csv"
    table = np.column_stack(
        (
            normalized_span,
            span_m,
            chord_m,
            twist_deg,
            pitch_axis,
            prebend_m,
            sweep_m,
            mode_shape,
            np.asarray(strouhal_by_station),
        )
    )
    np.savetxt(
        geometry_table,
        table,
        delimiter=",",
        header="span_fraction,span_m,chord_m,twist_deg,pitch_axis,prebend_m,sweep_m,edge_mode_tip_normalized,probe_strouhal",
        comments="",
    )

    manifest_files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if (
            path.name.endswith(".vorlap.bak")
            or path.name == "benchmark_manifest.json"
            or path.name == "vorlap_qblade_geometry_audit.csv"
        ):
            continue
        manifest_files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "case_name": CASE_NAME,
        "scope": "single-blade structural and imposed-load transfer benchmark; not physical VIV validation",
        "reference": {
            "doi": REFERENCE_DOI,
            "corrigendum_doi": CORRIGENDUM_DOI,
            "case": "P100-I50",
            "inflow_speed_mps": INFLOW_SPEED_MPS,
            "first_edgewise_frequency_hz": FREQUENCY_HZ,
            "published_fsi_lco_amplitude_m": 1.18,
            "published_fsi_zero_damping_lco_amplitude_m": 1.63,
            "published_forced_motion_damped_intersection_m": 1.24,
        },
        "public_model": {
            "repository": SOURCE_REPOSITORY,
            "commit": source_commit,
            "expected_commit": EXPECTED_SOURCE_COMMIT,
            "windio_file": source_yaml.relative_to(source_root).as_posix(),
            "windio_sha256": sha256_file(source_yaml),
            "openfast_mode_file": source_mode.relative_to(source_root).as_posix(),
            "openfast_mode_file_sha256": sha256_file(source_mode),
            "hawc2_structure_file": source_hawc2_structure.relative_to(source_root).as_posix(),
            "hawc2_structure_file_sha256": sha256_file(source_hawc2_structure),
            "hawc2_structure_set": 1,
            "license": "Apache-2.0",
            "license_file": source_license.relative_to(source_root).as_posix(),
            "license_sha256": sha256_file(source_license),
        },
        "qblade_mapping": {
            "station_count": len(normalized_span),
            "structural_section_count": len(hawc2_structure),
            "structural_element_count": len(hawc2_structure) - 1,
            "blade_span_m": float(span_m[-1] - span_m[0]),
            "hub_radius_m": hub_radius_m,
            "beam_type": "Timoshenko",
            "structural_source": "official HAWC2 set 1 converted to QBlade sectional properties",
            "elastic_center_mapping": "QBlade beam reference held on WindIO aerodynamic axis; HAWC2 elastic-center offsets omitted",
            "structural_pitch_mapping": "QBlade section axes held on WindIO blade frame; HAWC2 principal-axis pitch omitted",
            "bending_axis_mapping": "HAWC2 E*Ix to QBlade EIx and HAWC2 E*Iy to QBlade EIy",
            "mass_and_shear_center_offsets_retained": True,
            "damping_mapping": {
                "source": "OpenFAST/IEA-10.0-198-RWT_ElastoDyn_blade.dat BldEdDmp1",
                "source_edge_damping_ratio": edge_damping_ratio,
                "single_mode_stiffness_proportional_beta_s": rayleigh_beta_s,
                "relation": "beta = damping_ratio / (pi * first_edgewise_frequency_hz)",
                "scope": "selected first-edgewise single-mode approximation; not the paper's HAWC2 Rayleigh matrix",
            },
            "timestep_s": TIMESTEP_S,
            "duration_s": num_steps * TIMESTEP_S,
            "inflow_unit_vector": expected_inflow.tolist(),
            "root_frame_angle_audit": root_angle_audit,
            "rotation_axis": params.rotation_axis.tolist(),
        },
        "probe_source": {
            "kind": "synthetic coherent first-edgewise modal load",
            "target_frequency_hz": FREQUENCY_HZ,
            "coefficient_channel": "CD mapped to the section-chord direction",
            "peak_chordwise_amplitude": PROBE_CHORDWISE_AMPLITUDE,
            "phase_rad": 0.0,
            "wake_response_feedback": False,
            "calibrated_to_published_amplitude": False,
        },
        "preparation": {
            "swap_size": prep.swap_size,
            "node_count": len(prep.node_ids),
            "structural_node_count": len(prep.structural_node_ids),
            "converted_node_count": len(prep.converted_node_ids),
            "runtime_mutable_files_excluded_from_static_hashes": [
                f"{CASE_NAME}/Control/vorlap_qblade_geometry_audit.csv"
            ],
        },
        "files": manifest_files,
    }
    manifest_path = output / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    manifest_path = generate(args)
    print(manifest_path)


if __name__ == "__main__":
    main()
