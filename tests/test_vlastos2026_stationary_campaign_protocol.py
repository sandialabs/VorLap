import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import h5py
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "vlastos2026_cantilever"
    / "stationary_campaign_protocol.py"
)
SPEC = importlib.util.spec_from_file_location(
    "vlastos2026_stationary_campaign_protocol", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
protocol_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol_module
SPEC.loader.exec_module(protocol_module)


def _response_benchmark_module():
    name = "vlastos2026_response_benchmark_for_protocol_tests"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    module_path = MODULE_PATH.with_name("response_benchmark.py")
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ambient_v2_protocol_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("ambient_v2_protocol") / "campaign"
    return _response_benchmark_module().generate_campaign(
        root,
        response_schema_version=2,
    )


def _copy_v2_campaign(manifest_path, destination):
    shutil.copytree(manifest_path.parent, destination)
    return destination / "campaign_manifest.json"


def _write_manifest(path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rehash_manifest_file(manifest, root, relative):
    path = root / relative
    row = next(item for item in manifest["files"] if item["path"] == relative)
    row["sha256"] = protocol_module.sha256_file(path)
    row["bytes"] = path.stat().st_size


def _tag(value):
    return f"{value:g}".replace(".", "p")


def _response_model(structure, *, response_schema_version=1):
    node_ids = [
        f"BLD_1_{index / (protocol_module.N_STATIONS - 1):.6f}"
        for index in range(protocol_module.N_STATIONS)
    ]
    response_model = {
        "type": "modal_negative_damping",
        "schema_version": response_schema_version,
        "enabled": True,
        "node_ids": node_ids,
        "mode_shape": protocol_module._expected_mode_shape(),
        "natural_frequency_hz": structure["natural_frequency_hz"],
        "reference_length_m": protocol_module.DIAMETER_M,
        "aerodynamic_damping_gain_h1": (
            protocol_module.AERODYNAMIC_DAMPING_GAIN_H1
        ),
        "saturation_amplitude_ratio": protocol_module.SATURATION_AMPLITUDE_RATIO,
        "reduced_velocity_low": protocol_module.REDUCED_VELOCITY_LOW,
        "reduced_velocity_peak": protocol_module.REDUCED_VELOCITY_PEAK,
        "reduced_velocity_high": protocol_module.REDUCED_VELOCITY_HIGH,
        "rms_time_constant_s": (
            protocol_module.RMS_TIME_CONSTANT_CYCLES
            / structure["natural_frequency_hz"]
        ),
        "bandpass_damping_ratio": protocol_module.BANDPASS_DAMPING_RATIO,
        "projection_weights": protocol_module._expected_projection_weights(
            structure["equivalent_mass_kg_per_m"]
        ),
        "max_abs_gain": protocol_module.MAX_ABS_GAIN,
        "max_abs_nodal_force_n": protocol_module.MAX_ABS_NODAL_FORCE_N,
        "force_direction": "normal",
        "startup_seed": {
            "schema_version": 1,
            "enabled": True,
            "duration_cycles": protocol_module.STARTUP_SEED_DURATION_CYCLES,
            "target_amplitude_ratio": protocol_module.REQUESTED_SEED_TARGET_RATIO,
            "structural_damping_ratio": structure["structural_damping_ratio"],
            "taper": "hann_sine_squared",
            "carrier": "natural_frequency",
            "nodal_weighting": "projection_mass_times_mode",
            "nodal_force_limit_source": "response_model_max_abs_nodal_force_n",
        },
    }
    if response_schema_version == 2:
        response_model["aerodynamic_speed_basis"] = (
            protocol_module.AMBIENT_V2_AERODYNAMIC_SPEED_BASIS
        )
    return response_model


def _ambient_v2_equation_gate():
    table = []
    for u_star in protocol_module.REDUCED_VELOCITIES:
        activation = protocol_module.smooth_activation(u_star)
        denominator = (
            u_star
            * protocol_module.AERODYNAMIC_DAMPING_GAIN_H1
            * activation
        )
        amplitude = (
            0.0
            if denominator <= 0.0
            else protocol_module.SATURATION_AMPLITUDE_RATIO
            * math.sqrt(
                max(
                    0.0,
                    1.0
                    - 2.0
                    * protocol_module.AMBIENT_V2_CALIBRATION_QBLADE_SCRUTON_NUMBER
                    / denominator,
                )
            )
        )
        table.append(
            {
                "reduced_velocity": u_star,
                "analytical_equilibrium_amplitude_over_diameter": amplitude,
            }
        )
    caps_hash = protocol_module.sha256_payload(
        {
            str(sc): {
                str(u_star): cycles
                for u_star, cycles in sorted(rows.items())
            }
            for sc, rows in sorted(
                protocol_module.FROZEN_MAXIMUM_CYCLES.items()
            )
        }
    )
    return {
        "schema": protocol_module.AMBIENT_V2_EQUATION_GATE_SCHEMA,
        "response_model_schema_version": 2,
        "aerodynamic_speed_basis": (
            protocol_module.AMBIENT_V2_AERODYNAMIC_SPEED_BASIS
        ),
        "calibration_published_scruton_label": 5.18,
        "calibration_qblade_implied_scruton_number": (
            protocol_module.AMBIENT_V2_CALIBRATION_QBLADE_SCRUTON_NUMBER
        ),
        "scruton_source": (
            "generated case equivalent mass, damping ratio, campaign air "
            "density, and diameter; no Scruton retuning"
        ),
        "equation": (
            "A/D=a_sat*sqrt(max(0,1-2*Sc_QBlade/"
            "(Ustar*H1*activation)))"
        ),
        "coefficient_fingerprint_sha256": (
            protocol_module.AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
        ),
        "frozen_parameters": {
            "reduced_velocity_low": protocol_module.REDUCED_VELOCITY_LOW,
            "reduced_velocity_peak": protocol_module.REDUCED_VELOCITY_PEAK,
            "reduced_velocity_high": protocol_module.REDUCED_VELOCITY_HIGH,
            "aerodynamic_damping_gain_h1": (
                protocol_module.AERODYNAMIC_DAMPING_GAIN_H1
            ),
            "saturation_amplitude_ratio": (
                protocol_module.SATURATION_AMPLITUDE_RATIO
            ),
            "rms_time_constant_cycles": (
                protocol_module.RMS_TIME_CONSTANT_CYCLES
            ),
            "bandpass_damping_ratio": protocol_module.BANDPASS_DAMPING_RATIO,
            "max_abs_gain": protocol_module.MAX_ABS_GAIN,
            "max_abs_nodal_force_n": protocol_module.MAX_ABS_NODAL_FORCE_N,
        },
        "startup_seed": {
            "schema_version": 1,
            "duration_cycles": 5.0,
            "target_amplitude_ratio": 0.005,
            "taper": "hann_sine_squared",
            "carrier": "natural_frequency",
        },
        "limit_caps": {
            "max_abs_gain": protocol_module.MAX_ABS_GAIN,
            "max_abs_nodal_force_n": protocol_module.MAX_ABS_NODAL_FORCE_N,
        },
        "stationary_cycle_caps_sha256": caps_hash,
        "active_threshold_amplitude_over_diameter": (
            protocol_module.AMBIENT_V2_ACTIVE_AMPLITUDE_RATIO
        ),
        "expected_active_reduced_velocities": list(
            protocol_module.AMBIENT_V2_EXPECTED_ACTIVE_REDUCED_VELOCITIES
        ),
        "fine_estimator_rmse_tolerance_over_diameter": (
            protocol_module.AMBIENT_V2_FINE_ESTIMATOR_RMSE_TOLERANCE
        ),
        "fine_estimator_maximum_absolute_error_tolerance_over_diameter": (
            protocol_module.AMBIENT_V2_FINE_ESTIMATOR_MAX_ABS_TOLERANCE
        ),
        "coarse_fine_estimator_maximum_absolute_difference_tolerance_over_diameter": (
            protocol_module.AMBIENT_V2_COARSE_FINE_ESTIMATOR_TOLERANCE
        ),
        "analytical_table": table,
    }


def _ambient_v2_coefficients():
    coefficients = {
        "schema": protocol_module.CAMPAIGN_SCHEMA,
        "calibration_scruton_number": 5.18,
        "calibration_series": "hgast_varying_parameters",
        "calibration_target": (
            "published figure-digitized hGAST varying-parameter response, "
            "not experiment"
        ),
        "threshold_fraction_of_peak": 0.1,
        "figure9_path": "reference_data/sc5p18_figure9_digitized.csv",
        "figure9_sha256": (
            "674d8bb8f3b932f97bd2a177717e3042a1a0ba0b7c81cc915d51086f2ab550a1"
        ),
        "figure9_bytes": 1766,
        "model_parameters_path": (
            "reference_data/vlastos2026_model_parameters.csv"
        ),
        "model_parameters_sha256": (
            "b9bb5d03ca43e8d1dfa75059da54c0f11e3e68217d64dcad79f04b22cbdfd488"
        ),
        "model_parameters_bytes": 269,
        "calibration_points": 16,
        "held_out_scruton_numbers": [2.94, 7.02],
        "held_out_targets_accessed_during_calibration": False,
        "analytical_calibration_fit": {
            "equation": (
                "A/D=a_sat*sqrt(max(0,1-2*Sc/(Ustar*H1*activation)))"
            ),
            "objective": (
                "unweighted RMSE over every digitized Sc=5.18 hGAST point"
            ),
            "rmse_tip_amplitude_over_diameter": 0.02292400303734684,
            "maximum_absolute_error_tip_amplitude_over_diameter": 0.041349,
            "published_h_prime_max_recorded_but_not_assumed_equal_to_fitted_h1": (
                1.797174
            ),
            "published_scruton_number_label": 5.18,
            "qblade_implied_scruton_number_used_in_energy_balance": (
                protocol_module.AMBIENT_V2_CALIBRATION_QBLADE_SCRUTON_NUMBER
            ),
        },
        "frozen_parameters": dict(
            _ambient_v2_equation_gate()["frozen_parameters"]
        ),
    }
    fingerprint = protocol_module.sha256_payload(coefficients)
    assert fingerprint == (
        protocol_module.AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
    )
    coefficients["coefficient_fingerprint_sha256"] = fingerprint
    return coefficients


def _write_campaign(tmp_path, *, response_schema_version=1):
    root = tmp_path / "campaign"
    if response_schema_version == 2:
        return _response_benchmark_module().generate_campaign(
            root,
            response_schema_version=2,
        )
    root.mkdir(parents=True)
    coefficient_fingerprint = (
        protocol_module.AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
        if response_schema_version == 2
        else "a" * 64
    )
    coefficient_path = root / "frozen_response_coefficients.json"
    coefficient_object = (
        _ambient_v2_coefficients()
        if response_schema_version == 2
        else {"coefficient_fingerprint_sha256": coefficient_fingerprint}
    )
    coefficient_path.write_text(
        json.dumps(coefficient_object, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cases = []
    for sc in sorted(protocol_module.STRUCTURES):
        structure = protocol_module.STRUCTURES[sc]
        for u_star in protocol_module.REDUCED_VELOCITIES:
            maximum = protocol_module.derive_duration_bound(sc, u_star)[
                "maximum_endpoint"
            ]
            for level, timestep_s in (
                ("dt", protocol_module.COARSE_TIMESTEP_S),
                ("dt_half", protocol_module.FINE_TIMESTEP_S),
            ):
                namespace = (
                    f"{protocol_module.AMBIENT_V2_CASE_ID_NAMESPACE}_"
                    if response_schema_version == 2
                    else ""
                )
                case_id = (
                    f"{namespace}sc{_tag(sc)}_u{_tag(u_star)}_{level}"
                )
                case_root = root / "cases" / case_id
                case_root.mkdir(parents=True)
                sim_path = case_root / f"{case_id}.sim"
                turbine_path = case_root / "model" / "cantilever.trb"
                config_path = (
                    case_root
                    / "model"
                    / "Control"
                    / "vorlap_qblade_external.json"
                )
                main_path = case_root / "model" / "Structure" / "cantilever_main.str"
                beam_path = case_root / "model" / "Structure" / "cantilever_beam.str"
                tower_path = case_root / "model" / "Structure" / "rigid_tower.str"
                blade_path = case_root / "model" / "Aero" / "cantilever.bld"
                for path in (
                    turbine_path,
                    config_path,
                    main_path,
                    beam_path,
                    tower_path,
                    blade_path,
                ):
                    path.parent.mkdir(parents=True, exist_ok=True)
                turbine_path.write_text(
                    "Aero/cantilever.bld BLADEFILE\n"
                    "0 TURBTYPE\n1 NUMBLADES\n0 ROTORCONFIG\n0 ROTATIONALDIR\n"
                    "0 DISCTYPE\n0 INTPTYPE\n20 NUMPANELS\n"
                    "0 OVERHANG\n0 SHAFTTILT\n0 ROTORCONE\n"
                    "1 TOWERHEIGHT\n0.5 TOWERTOPRAD\n0.5 TOWERBOTRAD\n"
                    "false UNSTEADYAERO\ntrue 2PLIFTDRAG\nfalse HIMMELSKAMP\n"
                    "false TOWERSHADOW\n1 WAKETYPE\n"
                    "Structure/cantilever_main.str STRUCTURALFILE\n"
                    "false GEOMSTIFFNESS\nfalse AEROPANELLOADS\n0 CONTROLLERTYPE\n"
                    "libvorlap_qblade_bridge LIBFILE_1\nupdate LIBFUNCTION_1\n"
                    "129 LIBARRAYSIZE_1\n"
                    "Control/vorlap_qblade_external.json LIBPARAMETERFILE_1\n",
                    encoding="utf-8",
                )
                fraction = [
                    index / (protocol_module.N_STATIONS - 1)
                    for index in range(protocol_module.N_STATIONS)
                ]
                structural_input_lines = [
                    '0 "Time [s]"',
                    '1 "Timestep [-]"',
                    '2 "LSS Azimuthal Pos. [deg]"',
                ]
                structural_output_lines = []
                input_index = 3
                output_index = 3 + 3 * protocol_module.N_STATIONS
                for station in range(protocol_module.N_STATIONS):
                    position = station / (protocol_module.N_STATIONS - 1)
                    for direction in ("X", "Y", "Z"):
                        structural_input_lines.append(
                            f'{input_index} "{direction}_g Vel. BLD_1 pos '
                            f'{position:.3f} [m/s]"'
                        )
                        structural_output_lines.append(
                            f"{output_index} ADDFORCE BLD_1 {position:.6f} "
                            f"{direction} false"
                        )
                        input_index += 1
                        output_index += 1
                main_path.write_text(
                    "0.0005 GLBGEOEPS\n0 PRECONE\n0 SHFTTILT\n0 OVERHANG\n"
                    "0 TWR2SHFT\n0 YAWBRMASS\n0 NACMASS\n0 NACCMX\n0 NACCMY\n"
                    "0 NACCMZ\n0 0 0 0 0 0 NACINER\n0 NACYINER\n0 HUBMASS\n"
                    "0 HUBCM\n1 HUBINER\n1 GBRATIO\n1 GBOXEFF\n1 GENEFF\n"
                    "false DRTRDOF\n1 GENINER\n1e12 DTTORSPR\n1e9 DTTORDMP\n"
                    "1 NUMBLD\ncantilever_beam.str BLDFILE_1\n"
                    "1 TWRHEIGHT\nrigid_tower.str TWRFILE\n"
                    "true FOR_OUT\ntrue ROT_OUT\ntrue MOM_OUT\ntrue DEF_OUT\n"
                    "true POS_OUT\ntrue VEL_OUT\ntrue ACC_OUT\ntrue LVE_OUT\n"
                    "true LAC_OUT\n"
                    + "\n".join(f"BLD_1_{value:.10f}" for value in fraction)
                    + "\nTWR_0.0\nTWR_1.0\n\nEXTERNAL_1_IN\nSWAP DATA\n"
                    + "\n".join(structural_input_lines)
                    + "\n\nEXTERNAL_1_OUT\nSWAP ACTION ID POS DIR LOCAL\n"
                    + "\n".join(structural_output_lines)
                    + "\n",
                    encoding="utf-8",
                )
                mass_per_length = structure["equivalent_mass_kg_per_m"]
                frequency_hz = structure["natural_frequency_hz"]
                target_ei = mass_per_length * (
                    (2.0 * math.pi * frequency_hz)
                    * protocol_module.HEIGHT_M**2
                    / protocol_module.EULER_BERNOULLI_BETA1**2
                ) ** 2
                beam_lines = [
                    f"{structure['structural_damping_ratio'] / (math.pi * frequency_hz):.12e} RAYLEIGHDMP",
                    "1 STIFFTUNER",
                    "1 MASSTUNER",
                    "0 INTPTYPE",
                    "1 BEAMTYPE",
                    "2 DISCTYPE",
                    f"{protocol_module.N_STATIONS} DISC",
                    (
                        "LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] "
                        "EA_[N] GJ_[N.m^2] GA_[N] STRPIT_[deg] KSX_[-] KSY_[-] "
                        "RGX_[-] RGY_[-] XCM_[-] YCM_[-] XCE_[-] YCE_[-] XCS_[-] YCS_[-]"
                    ),
                ]
                for value in fraction:
                    beam_lines.append(
                        " ".join(
                            f"{item:.12e}"
                            for item in (
                                value,
                                mass_per_length,
                                target_ei,
                                target_ei,
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
                        )
                    )
                beam_path.write_text("\n".join(beam_lines) + "\n", encoding="utf-8")
                tower_section = (
                    "1 1e15 1e15 1e15 1e15 1e15 0 0.85 0.85 0.25 0.25 "
                    "0 0 0 0 0 0 1"
                )
                tower_path.write_text(
                    "0 RAYLEIGHDMP\n1 STIFFTUNER\n1 MASSTUNER\n"
                    "0 INTPTYPE\n1 BEAMTYPE\n0 DISCTYPE\n2 DISC\n"
                    "LENFRACT_[-] MASSD_[kg/m] EIx_[N.m^2] EIy_[N.m^2] "
                    "EA_[N] GJ_[N.m^2] GA_[N] STRPIT_[deg] KSX_[-] KSY_[-] "
                    "RGX_[-] RGY_[-] XCM_[-] YCM_[-] XCE_[-] YCE_[-] "
                    "XCS_[-] YCS_[-] Diameter_[m]\n"
                    f"0 {tower_section}\n1 {tower_section}\n",
                    encoding="utf-8",
                )
                blade_lines = [
                    "HAWT ROTORTYPE",
                    "1 NUMBLADES",
                    (
                        "POS_[m] CHORD_[m] TWIST_[deg] OFFSET_X_[m] "
                        "OFFSET_Y_[m] TAXIS_[-] POLAR_FILE"
                    ),
                ]
                blade_lines.extend(
                    f"{protocol_module.HEIGHT_M * value:.12e} "
                    f"{protocol_module.DIAMETER_M:.12e} 0 0 0 0.5 polar.plr"
                    for value in fraction
                )
                blade_path.write_text("\n".join(blade_lines) + "\n", encoding="utf-8")
                mean_inflow = (
                    u_star
                    * structure["natural_frequency_hz"]
                    * protocol_module.DIAMETER_M
                )
                sim_path.write_text(
                    "0 ISOFFSHORE\n0 INITIAL_YAW\n0 INITIAL_PITCH\n"
                    "0 INITIAL_AZIMUTH\n2 STRSUBSTEP\n0 RELAXSTEPS\n"
                    "0 RELAXTIMESTEP\n1 PRESCRIBETYPE\n0 RPMPRESCRIBED\n"
                    "5 STRITERATIONS\n0 MODNEWTONITER\n"
                    f"{maximum[level + '_steps']} NUMTIMESTEPS\n"
                    f"{timestep_s:.8f} TIMESTEP\n"
                    f"{mean_inflow:.12g} MEANINF\n"
                    "model/cantilever.trb TURBFILE\n"
                    f"{protocol_module.AIR_DENSITY_KGPM3} DENSITYAIR\n"
                    f"{protocol_module.AIR_KINEMATIC_VISCOSITY_M2PS} VISCOSITYAIR\n"
                    "9.80665 GRAVITY\n0 INCLUDEAERO\n0 INCLUDEHYDRO\n"
                    "0 RAMPUP\n0 ADDDAMP\n0 WNDTYPE\n90 HORANGLE\n"
                    "0 VERTANGLE\n0 PROFILETYPE\n0 SHEAREXP\nfalse GROUNDEFFECT\n"
                    "0 GLOBPOS_X\n0 GLOBPOS_Y\n0 GLOBPOS_Z\n"
                    "0 GLOBROT_X\n0 GLOBROT_Y\n0 GLOBROT_Z\n0 STOREFROM\n"
                    "EVENTFILE\nLOADINGFILE\nSIMFILE\nMOTIONFILE\n"
                    "false STOREREPLAY\nfalse STOREAERO\nfalse STOREBLADE\n"
                    "true STORESTRUCT\ntrue STORESIM\nfalse STOREHYDRO\n"
                    "false STORECONTROLLER\nfalse CALCMODAL\n",
                    encoding="utf-8",
                )
                response_model = _response_model(
                    structure,
                    response_schema_version=response_schema_version,
                )
                node_ids = [
                    f"BLD_1_{index / (protocol_module.N_STATIONS - 1):.10f}"
                    for index in range(protocol_module.N_STATIONS)
                ]
                external_input_rows = [
                    {"index": 0, "variable": "Time [s]"},
                    {"index": 1, "variable": "Timestep [-]"},
                    {"index": 2, "variable": "LSS Azimuthal Pos. [deg]"},
                ]
                external_output_rows = []
                input_index = 3
                output_index = 3 + 3 * protocol_module.N_STATIONS
                for station in range(protocol_module.N_STATIONS):
                    position = station / (protocol_module.N_STATIONS - 1)
                    for direction in ("X", "Y", "Z"):
                        external_input_rows.append(
                            {
                                "index": input_index,
                                "variable": (
                                    f"{direction}_g Vel. BLD_1 pos "
                                    f"{position:.3f} [m/s]"
                                ),
                            }
                        )
                        external_output_rows.append(
                            {
                                "action": "ADDFORCE",
                                "direction": direction,
                                "id": "BLD_1",
                                "index": output_index,
                                "local": False,
                                "position": position,
                            }
                        )
                        input_index += 1
                        output_index += 1
                config_path.write_text(
                    json.dumps(
                        {
                            "airfoil_dir": "../VorLapAirfoils",
                            "airfoil_fallbacks": [],
                            "airfoil_source_manifest": [],
                            "available_airfoil_ids": ["CYLINDER", "default"],
                            "debug": False,
                            "default_airfoil_id": "default",
                            "default_airfoil_source_id": "CYLINDER",
                            "external_input_rows": external_input_rows,
                            "external_output_rows": external_output_rows,
                            "force_scale": 0.0,
                            "function_name": "update",
                            "geometry_audit_file": "vorlap_qblade_geometry_audit.csv",
                            "include_mean_load": False,
                            "include_struts": False,
                            "include_tower": False,
                            "library_stem": "libvorlap_qblade_bridge",
                            "n_freq_depth": 2,
                            "node_ids": node_ids,
                            "parameter_file": "Control/vorlap_qblade_external.json",
                            "python_path_entries": ["/work/VorLap"],
                            "response_model": response_model,
                            "sim_path": f"../../{case_id}.sim",
                            "source_bounds_policy": "raise",
                            "source_frequency_cap_hz": 0.5 / timestep_s,
                            "source_parameter_dir": ".",
                            "swap_layout": [
                                {"name": "time", "offset": 0, "shape": []},
                                {"name": "timestep", "offset": 1, "shape": []},
                                {"name": "azimuth_deg", "offset": 2, "shape": []},
                                {
                                    "name": "velocity",
                                    "offset": 3,
                                    "shape": [protocol_module.N_STATIONS, 3],
                                },
                                {
                                    "name": "force",
                                    "offset": 3 + 3 * protocol_module.N_STATIONS,
                                    "shape": [protocol_module.N_STATIONS, 3],
                                },
                            ],
                            "swap_size": 3 + 6 * protocol_module.N_STATIONS,
                            "tower_airfoil_id": "cylinder",
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                cases.append(
                    {
                        "case_id": case_id,
                        "scruton_number": sc,
                        "qblade_implied_scruton_number": structure[
                            "qblade_implied_scruton_number"
                        ],
                        "role": structure["role"],
                        "reduced_velocity": u_star,
                        "mean_inflow_mps": mean_inflow,
                        "timestep_level": level,
                        "timestep_s": timestep_s,
                        "num_steps": maximum[level + "_steps"],
                        "duration_s": maximum["endpoint_time_s"],
                        "structure": {
                            "natural_frequency_hz": structure[
                                "natural_frequency_hz"
                            ],
                            "damping_ratio": structure[
                                "structural_damping_ratio"
                            ],
                            "equivalent_mass_kg_per_m": structure[
                                "equivalent_mass_kg_per_m"
                            ],
                        },
                        "sim_path": sim_path.relative_to(root).as_posix(),
                        "parameter_file": config_path.relative_to(root).as_posix(),
                        "response_model_config": response_model,
                        "coefficient_fingerprint_sha256": coefficient_fingerprint,
                    }
                )

    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": protocol_module.sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    manifest = {
        "schema": protocol_module.CAMPAIGN_SCHEMA,
        "calibration_policy": {
            "coefficient_file": coefficient_path.relative_to(root).as_posix(),
            "coefficient_file_sha256": protocol_module.sha256_file(
                coefficient_path
            ),
            "coefficient_fingerprint_sha256": coefficient_fingerprint,
        },
        "numerics": {
            "duration_policy": "frozen_stationary_cycle_caps",
            "stationary_protocol_compatible_full_grid": True,
            "qualification_input_eligible": True,
            "dt_halving_enabled": True,
            "required_sil_vorlap_history_mode": "compact",
        },
        "cases": cases,
        "files": files,
    }
    if response_schema_version == 2:
        manifest["response_model_semantics"] = (
            protocol_module.ambient_v2_response_semantics()
        )
        manifest["equation_reproduction_gate"] = (
            _ambient_v2_equation_gate()
        )
    manifest_path = root / "campaign_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _input_snapshot(root):
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def test_frozen_caps_reproduce_sc5_table_and_derive_distinct_heldout_caps():
    rows = {
        (row["scruton_number"], row["reduced_velocity"]): row
        for row in protocol_module.frozen_duration_table()
    }

    expected_sc5 = {
        4.30: (240, 31_210, 62_420, 15.6050),
        4.80: (240, 31_210, 62_420, 15.6050),
        5.00: (240, 31_210, 62_420, 15.6050),
        5.20: (240, 31_210, 62_420, 15.6050),
        5.50: (240, 31_210, 62_420, 15.6050),
        5.80: (2520, 327_699, 655_398, 163.8495),
        6.20: (320, 41_613, 83_226, 20.8065),
        6.25: (320, 41_613, 83_226, 20.8065),
        6.50: (480, 62_419, 124_838, 31.2095),
        6.80: (360, 46_815, 93_630, 23.4075),
        7.20: (240, 31_210, 62_420, 15.6050),
        8.20: (240, 31_210, 62_420, 15.6050),
    }
    for u_star, expected in expected_sc5.items():
        row = rows[(5.18, u_star)]
        endpoint = row["maximum_endpoint"]
        assert (
            row["maximum_cycles"],
            endpoint["dt_steps"],
            endpoint["dt_half_steps"],
            endpoint["endpoint_time_s"],
        ) == expected
        assert endpoint["dt_half_steps"] == 2 * endpoint["dt_steps"]

    assert rows[(2.94, 5.80)]["maximum_cycles"] == 400
    assert rows[(5.18, 5.80)]["maximum_cycles"] == 2520
    assert rows[(7.02, 5.80)]["maximum_cycles"] == 1080
    assert rows[(2.94, 6.80)]["maximum_cycles"] == 600
    assert rows[(5.18, 6.80)]["maximum_cycles"] == 360
    assert rows[(7.02, 6.80)]["maximum_cycles"] == 320
    assert rows[(5.18, 5.80)]["branch"] == "growth_to_lco"
    assert rows[(5.18, 6.80)]["branch"] == "decay_to_terminal_tolerance"


def test_protocol_binds_all_inputs_without_mutating_campaign(tmp_path):
    manifest_path = _write_campaign(tmp_path)
    before = _input_snapshot(manifest_path.parent)

    protocol = protocol_module.build_protocol(manifest_path)
    validation = protocol_module.validate_protocol(protocol, manifest_path)

    assert validation["valid"] is True
    assert validation["case_count"] == 72
    assert validation["case_pair_count"] == 36
    assert len(protocol["case_pairs"]) == 36
    assert all(
        endpoint["dt_half_steps"] == 2 * endpoint["dt_steps"]
        for pair in protocol["case_pairs"]
        for endpoint in pair["evaluation_endpoints"]
    )
    assert _input_snapshot(manifest_path.parent) == before


@pytest.mark.requires_qblade
def test_ambient_v2_protocol_binds_full_partition_and_response_semantics(
    tmp_path,
):
    manifest_path = _write_campaign(
        tmp_path,
        response_schema_version=2,
    )
    before = _input_snapshot(manifest_path.parent)

    protocol = protocol_module.build_protocol(manifest_path)
    validation = protocol_module.validate_protocol(protocol, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert validation["valid"] is True
    assert validation["case_count"] == 72
    assert validation["case_pair_count"] == 36
    assert "response_semantics_sha256" in protocol["campaign_binding"]
    assert manifest["response_model_semantics"] == (
        protocol_module.ambient_v2_response_semantics()
    )
    assert sum(row["role"] == "calibration" for row in manifest["cases"]) == 24
    assert sum(row["role"] == "held_out" for row in manifest["cases"]) == 48
    assert all(
        row["case_id"].startswith(
            f"{protocol_module.AMBIENT_V2_CASE_ID_NAMESPACE}_"
        )
        for row in manifest["cases"]
    )
    assert {
        row["response_model_config"]["schema_version"]
        for row in manifest["cases"]
    } == {2}
    assert {
        row["response_model_config"]["aerodynamic_speed_basis"]
        for row in manifest["cases"]
    } == {protocol_module.AMBIENT_V2_AERODYNAMIC_SPEED_BASIS}
    assert _input_snapshot(manifest_path.parent) == before


def test_ambient_v2_semantics_and_equation_gate_reject_any_frozen_drift():
    base = {
        "response_model_semantics": (
            protocol_module.ambient_v2_response_semantics()
        ),
        "equation_reproduction_gate": _ambient_v2_equation_gate(),
    }
    validated = protocol_module.validate_response_semantics(base)
    assert validated["response_schema_version"] == 2
    assert validated["v2"] is True

    tampered_documents = []
    changed_semantics = copy.deepcopy(base)
    changed_semantics["response_model_semantics"][
        "checkpoint_reuse_policy"
    ] = "allow"
    tampered_documents.append(changed_semantics)

    changed_equation = copy.deepcopy(base)
    changed_equation["equation_reproduction_gate"]["equation"] = "different"
    tampered_documents.append(changed_equation)

    changed_scruton = copy.deepcopy(base)
    changed_scruton["equation_reproduction_gate"][
        "calibration_qblade_implied_scruton_number"
    ] = 5.18
    tampered_documents.append(changed_scruton)

    changed_tolerance = copy.deepcopy(base)
    changed_tolerance["equation_reproduction_gate"][
        "fine_estimator_rmse_tolerance_over_diameter"
    ] = 1.0
    tampered_documents.append(changed_tolerance)

    changed_table = copy.deepcopy(base)
    changed_table["equation_reproduction_gate"]["analytical_table"][5][
        "analytical_equilibrium_amplitude_over_diameter"
    ] += 1.0e-3
    tampered_documents.append(changed_table)

    extra_field = copy.deepcopy(base)
    extra_field["equation_reproduction_gate"]["unfrozen_override"] = True
    tampered_documents.append(extra_field)

    for document in tampered_documents:
        with pytest.raises(protocol_module.ProtocolError):
            protocol_module.validate_response_semantics(document)


@pytest.mark.requires_qblade
def test_ambient_v2_protocol_rejects_namespace_speed_and_case_schema_drift(
    tmp_path,
):
    manifest_path = _write_campaign(
        tmp_path,
        response_schema_version=2,
    )
    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    namespace_drift = copy.deepcopy(original)
    namespace_drift["cases"][0]["case_id"] = namespace_drift["cases"][0][
        "case_id"
    ].removeprefix(f"{protocol_module.AMBIENT_V2_CASE_ID_NAMESPACE}_")
    manifest_path.write_text(
        json.dumps(namespace_drift, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(protocol_module.ProtocolError, match="case_id"):
        protocol_module.build_protocol(manifest_path)

    speed_drift = copy.deepcopy(original)
    speed_drift["cases"][0]["mean_inflow_mps"] += 0.01
    manifest_path.write_text(
        json.dumps(speed_drift, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(protocol_module.ProtocolError, match="mean_inflow_mps"):
        protocol_module.build_protocol(manifest_path)

    schema_drift = copy.deepcopy(original)
    schema_drift["cases"][0]["response_model_config"]["schema_version"] = 1
    manifest_path.write_text(
        json.dumps(schema_drift, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        protocol_module.ProtocolError,
        match="response_model|response-model|objects differ",
    ):
        protocol_module.build_protocol(manifest_path)


@pytest.mark.requires_qblade
def test_ambient_v2_protocol_rejects_rehashed_h5_thickness_and_coefficients(
    ambient_v2_protocol_campaign,
    tmp_path,
):
    manifest_path = _copy_v2_campaign(
        ambient_v2_protocol_campaign, tmp_path / "h5_campaign"
    )
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = original_manifest["cases"][0]
    config_relative = entry["parameter_file"]
    config_path = manifest_path.parent / config_relative
    original_config_bytes = config_path.read_bytes()
    source_relative = (
        Path(config_relative).parent.parent / "VorLapAirfoils" / "CYLINDER.h5"
    ).as_posix()
    source_path = manifest_path.parent / source_relative
    original_source_bytes = source_path.read_bytes()

    mutations = (
        lambda h5: h5["Thickness"].__setitem__(slice(None), [0.5]),
        lambda h5: h5["CL_Amp"].__setitem__((0, 0, 1), 0.4),
    )
    for mutate in mutations:
        source_path.write_bytes(original_source_bytes)
        config_path.write_bytes(original_config_bytes)
        manifest = copy.deepcopy(original_manifest)
        with h5py.File(source_path, "a") as h5:
            mutate(h5)
        _rehash_manifest_file(manifest, manifest_path.parent, source_relative)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source_row = config["airfoil_source_manifest"][0]
        source_row["sha256"] = protocol_module.sha256_file(source_path)
        source_row["bytes"] = source_path.stat().st_size
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _rehash_manifest_file(manifest, manifest_path.parent, config_relative)
        _write_manifest(manifest_path, manifest)
        with pytest.raises(protocol_module.ProtocolError, match="CYLINDER"):
            protocol_module.build_protocol(manifest_path)


@pytest.mark.requires_qblade
def test_ambient_v2_protocol_rejects_rehashed_gate_reference_filter_and_log(
    ambient_v2_protocol_campaign,
    tmp_path,
):
    cases = []

    gate_manifest = _copy_v2_campaign(
        ambient_v2_protocol_campaign, tmp_path / "gate_campaign"
    )
    manifest = json.loads(gate_manifest.read_text(encoding="utf-8"))
    gate_relative = manifest["calibration_policy"][
        "postfreeze_analytical_gate_file"
    ]
    gate_path = gate_manifest.parent / gate_relative
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["status"] = "pass"
    gate["pass"] = True
    gate["decision"] = "tampered pass"
    gate_path.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_manifest_file(manifest, gate_manifest.parent, gate_relative)
    manifest["calibration_policy"]["postfreeze_analytical_gate_sha256"] = (
        protocol_module.sha256_file(gate_path)
    )
    manifest["calibration_policy"]["postfreeze_analytical_gate_status"] = "pass"
    _write_manifest(gate_manifest, manifest)
    cases.append((gate_manifest, "gate|calibration_policy"))

    reference_manifest = _copy_v2_campaign(
        ambient_v2_protocol_campaign, tmp_path / "reference_campaign"
    )
    manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
    reference_relative = "reference_data/all_sc_figure10_peak_digitized.csv"
    reference_path = reference_manifest.parent / reference_relative
    reference_path.write_text(
        reference_path.read_text(encoding="utf-8") + "# substituted\n",
        encoding="utf-8",
    )
    _rehash_manifest_file(manifest, reference_manifest.parent, reference_relative)
    identity = manifest["reference"]["reference_files"][
        "all_sc_figure10_peak_digitized.csv"
    ]
    identity["sha256"] = protocol_module.sha256_file(reference_path)
    identity["bytes"] = reference_path.stat().st_size
    _write_manifest(reference_manifest, manifest)
    cases.append((reference_manifest, "reference"))

    filter_manifest = _copy_v2_campaign(
        ambient_v2_protocol_campaign, tmp_path / "filter_campaign"
    )
    manifest = json.loads(filter_manifest.read_text(encoding="utf-8"))
    filter_relative = "cantilever_results.filter"
    filter_path = filter_manifest.parent / filter_relative
    filter_path.write_text(
        filter_path.read_text(encoding="utf-8") + "extra channel\n",
        encoding="utf-8",
    )
    _rehash_manifest_file(manifest, filter_manifest.parent, filter_relative)
    manifest["runtime_contract"]["result_filter"].update(
        {
            "sha256": protocol_module.sha256_file(filter_path),
            "bytes": filter_path.stat().st_size,
        }
    )
    _write_manifest(filter_manifest, manifest)
    cases.append((filter_manifest, "filter"))

    log_manifest = _copy_v2_campaign(
        ambient_v2_protocol_campaign, tmp_path / "log_campaign"
    )
    manifest = json.loads(log_manifest.read_text(encoding="utf-8"))
    log_path = (
        log_manifest.parent / manifest["cases"][0]["parameter_file"]
    ).parent / "vorlap_qblade_bridge_cpp.log"
    log_path.write_text("update entered; count=1\n", encoding="utf-8")
    cases.append((log_manifest, "campaign tree|native bridge"))

    for path, match in cases:
        with pytest.raises(protocol_module.ProtocolError, match=match):
            protocol_module.build_protocol(path)


@pytest.mark.requires_qblade
def test_ambient_v2_protocol_rejects_unknown_response_and_seed_fields(
    ambient_v2_protocol_campaign,
    tmp_path,
):
    manifest_path = _copy_v2_campaign(
        ambient_v2_protocol_campaign, tmp_path / "response_campaign"
    )
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = original_manifest["cases"][0]
    config_relative = entry["parameter_file"]
    config_path = manifest_path.parent / config_relative
    original_config_bytes = config_path.read_bytes()

    mutations = (
        lambda model: model.__setitem__("unfrozen_override", True),
        lambda model: model.__setitem__(
            "reduced_velocity_activation", [4.0, 6.25, 8.0]
        ),
        lambda model: model["startup_seed"].__setitem__(
            "unfrozen_seed_override", True
        ),
    )
    for mutate in mutations:
        manifest = copy.deepcopy(original_manifest)
        config_path.write_bytes(original_config_bytes)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        mutate(config["response_model"])
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["cases"][0]["response_model_config"] = config[
            "response_model"
        ]
        _rehash_manifest_file(manifest, manifest_path.parent, config_relative)
        _write_manifest(manifest_path, manifest)
        with pytest.raises(protocol_module.ProtocolError, match="key schema"):
            protocol_module.build_protocol(manifest_path)


@pytest.mark.requires_qblade
def test_ambient_v2_protocol_rejects_policy_and_manifest_semantic_drift(
    ambient_v2_protocol_campaign,
    tmp_path,
):
    manifest_path = _copy_v2_campaign(
        ambient_v2_protocol_campaign, tmp_path / "semantic_campaign"
    )
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutations = (
        lambda value: value["calibration_policy"].__setitem__(
            "calibration_scruton_number", 5.0
        ),
        lambda value: value["calibration_policy"].__setitem__(
            "held_out_scruton_numbers", [2.94, 5.18]
        ),
        lambda value: value["calibration_policy"].__setitem__(
            "held_out_targets_accessed_during_calibration", True
        ),
        lambda value: value["calibration_policy"].__setitem__(
            "coefficients_changed_after_holdout_evaluation", True
        ),
        lambda value: value["calibration_policy"].__setitem__(
            "coefficient_file", "../frozen_response_coefficients.json"
        ),
        lambda value: value["cases"][0].__setitem__(
            "role", "calibration"
        ),
        lambda value: value["cases"][0].__setitem__(
            "reynolds_number", value["cases"][0]["reynolds_number"] + 1.0
        ),
        lambda value: value["cases"][0]["structure"].__setitem__(
            "h_prime_max", 99.0
        ),
        lambda value: value["numerics"].__setitem__(
            "minimum_global_steps_per_cycle", 1.0
        ),
        lambda value: value["reference"].__setitem__("diameter_m", 0.1),
        lambda value: value["qblade_mapping"].__setitem__(
            "native_aerodynamics", True
        ),
        lambda value: value["cases"].__setitem__(
            slice(0, 2), [value["cases"][1], value["cases"][0]]
        ),
        lambda value: value.__setitem__("unfrozen_override", True),
        lambda value: value["cases"][0].__setitem__(
            "unfrozen_override", True
        ),
        lambda value: value["cases"][0]["structure"].__setitem__(
            "unfrozen_override", True
        ),
        lambda value: value["files"][0].__setitem__(
            "unfrozen_override", True
        ),
    )
    for mutate in mutations:
        manifest = copy.deepcopy(original)
        mutate(manifest)
        _write_manifest(manifest_path, manifest)
        with pytest.raises(protocol_module.ProtocolError):
            protocol_module.build_protocol(manifest_path)


def test_validation_fails_closed_for_config_and_case_hash_mismatches(tmp_path):
    manifest_path = _write_campaign(tmp_path)
    protocol = protocol_module.build_protocol(manifest_path)
    first_pair = protocol["case_pairs"][0]
    config_relative = next(
        case["parameter_file"]
        for case in json.loads(manifest_path.read_text(encoding="utf-8"))["cases"]
        if case["case_id"] == first_pair["cases"]["dt"]["case_id"]
    )
    config_path = manifest_path.parent / config_relative
    config_path.write_text(config_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(protocol_module.ProtocolError, match="parameter config hash mismatch"):
        protocol_module.validate_protocol(protocol, manifest_path)

    second_manifest = _write_campaign(tmp_path / "second")
    second_protocol = protocol_module.build_protocol(second_manifest)
    tampered = copy.deepcopy(second_protocol)
    tampered["case_pairs"][0]["cases"]["dt"]["case_binding_sha256"] = "b" * 64
    payload = dict(tampered)
    del payload["protocol_sha256"]
    tampered["protocol_sha256"] = protocol_module.sha256_payload(payload)
    with pytest.raises(protocol_module.ProtocolError, match="does not match"):
        protocol_module.validate_protocol(tampered, second_manifest)


def test_protocol_parses_simulation_and_response_physics_not_only_hashes(tmp_path):
    manifest_path = _write_campaign(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["cases"][0]
    sim_relative = entry["sim_path"]
    sim_path = manifest_path.parent / sim_relative
    text = sim_path.read_text(encoding="utf-8")
    text = text.replace(
        f"{entry['num_steps']} NUMTIMESTEPS", "1 NUMTIMESTEPS"
    )
    sim_path.write_text(text, encoding="utf-8")
    sim_row = next(row for row in manifest["files"] if row["path"] == sim_relative)
    sim_row["sha256"] = protocol_module.sha256_file(sim_path)
    sim_row["bytes"] = sim_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(protocol_module.ProtocolError, match="NUMTIMESTEPS"):
        protocol_module.build_protocol(manifest_path)

    second_manifest_path = _write_campaign(tmp_path / "response")
    second_manifest = json.loads(
        second_manifest_path.read_text(encoding="utf-8")
    )
    second_entry = second_manifest["cases"][0]
    config_relative = second_entry["parameter_file"]
    config_path = second_manifest_path.parent / config_relative
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["response_model"]["max_abs_gain"] = 999.0
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    second_entry["response_model_config"] = config["response_model"]
    config_row = next(
        row for row in second_manifest["files"] if row["path"] == config_relative
    )
    config_row["sha256"] = protocol_module.sha256_file(config_path)
    config_row["bytes"] = config_path.stat().st_size
    second_manifest_path.write_text(
        json.dumps(second_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(protocol_module.ProtocolError, match="max_abs_gain"):
        protocol_module.build_protocol(second_manifest_path)

    structural_manifest_path = _write_campaign(tmp_path / "structure")
    structural_manifest = json.loads(
        structural_manifest_path.read_text(encoding="utf-8")
    )
    structural_entry = structural_manifest["cases"][0]
    beam_relative = (
        Path(structural_entry["sim_path"]).parent
        / "model"
        / "Structure"
        / "cantilever_beam.str"
    ).as_posix()
    beam_path = structural_manifest_path.parent / beam_relative
    beam_text = beam_path.read_text(encoding="utf-8")
    beam_lines = beam_text.splitlines()
    beam_lines[0] = "0.999999 RAYLEIGHDMP"
    beam_path.write_text("\n".join(beam_lines) + "\n", encoding="utf-8")
    beam_row = next(
        row for row in structural_manifest["files"] if row["path"] == beam_relative
    )
    beam_row["sha256"] = protocol_module.sha256_file(beam_path)
    beam_row["bytes"] = beam_path.stat().st_size
    structural_manifest_path.write_text(
        json.dumps(structural_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(protocol_module.ProtocolError, match="RAYLEIGHDMP"):
        protocol_module.build_protocol(structural_manifest_path)


def test_protocol_rejects_rehashed_material_dynamics_and_mapping_tampering(tmp_path):
    manifest_path = _write_campaign(tmp_path)
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = original_manifest["cases"][0]
    case_root = manifest_path.parent / Path(entry["sim_path"]).parent
    cases = (
        (Path(entry["sim_path"]), "0 RPMPRESCRIBED", "10 RPMPRESCRIBED"),
        (Path(entry["sim_path"]), "90 HORANGLE", "0 HORANGLE"),
        (Path(entry["sim_path"]), "LOADINGFILE", "loads.dat LOADINGFILE"),
        (
            case_root.relative_to(manifest_path.parent)
            / "model"
            / "cantilever.trb",
            "0 TURBTYPE",
            "1 TURBTYPE",
        ),
        (
            case_root.relative_to(manifest_path.parent)
            / "model"
            / "cantilever.trb",
            "libvorlap_qblade_bridge LIBFILE_1",
            "wrong_bridge LIBFILE_1",
        ),
        (
            case_root.relative_to(manifest_path.parent)
            / "model"
            / "cantilever.trb",
            "Control/vorlap_qblade_external.json LIBPARAMETERFILE_1",
            "Control/vorlap_qblade_external.json LIBPARAMETERFILE_1\n"
            "wrong_bridge LIBFILE_2",
        ),
        (
            case_root.relative_to(manifest_path.parent)
            / "model"
            / "Structure"
            / "cantilever_main.str",
            "rigid_tower.str TWRFILE",
            "cantilever_beam.str TWRFILE",
        ),
        (
            case_root.relative_to(manifest_path.parent)
            / "model"
            / "Structure"
            / "cantilever_main.str",
            '4 "Y_g Vel. BLD_1 pos 0.000 [m/s]"',
            '4 "X_g Vel. BLD_1 pos 0.000 [m/s]"',
        ),
        (
            case_root.relative_to(manifest_path.parent)
            / "model"
            / "Structure"
            / "cantilever_main.str",
            "67 ADDFORCE BLD_1 0.000000 Y false",
            "67 ADDFORCE BLD_1 0.000000 X false",
        ),
        (
            case_root.relative_to(manifest_path.parent)
            / "model"
            / "Structure"
            / "cantilever_beam.str",
            "1 BEAMTYPE",
            "0 BEAMTYPE",
        ),
        (
            case_root.relative_to(manifest_path.parent)
            / "model"
            / "Structure"
            / "cantilever_beam.str",
            "7.000000000000e+06",
            "7.000000000000e+00",
        ),
        (
            case_root.relative_to(manifest_path.parent)
            / "model"
            / "Structure"
            / "rigid_tower.str",
            "1e15",
            "1e1",
        ),
        (
            Path(entry["parameter_file"]),
            '"local": false',
            '"local": true',
        ),
        (
            Path(entry["parameter_file"]),
            '"tower_airfoil_id": "cylinder"',
            '"fluid_density": 100.0,\n  "tower_airfoil_id": "cylinder"',
        ),
        (
            Path(entry["parameter_file"]),
            '"/work/VorLap"',
            '"/tmp/alternate_vorlap"',
        ),
    )
    original_files = {
        relative.as_posix(): (manifest_path.parent / relative).read_bytes()
        for relative, _old, _new in cases
    }

    for relative, old, new in cases:
        manifest = json.loads(json.dumps(original_manifest))
        path = manifest_path.parent / relative
        original_text = original_files[relative.as_posix()].decode("utf-8")
        assert old in original_text
        path.write_text(original_text.replace(old, new, 1), encoding="utf-8")
        row = next(item for item in manifest["files"] if item["path"] == relative.as_posix())
        row["sha256"] = protocol_module.sha256_file(path)
        row["bytes"] = path.stat().st_size
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(protocol_module.ProtocolError):
            protocol_module.build_protocol(manifest_path)
        path.write_bytes(original_files[relative.as_posix()])
        manifest_path.write_text(
            json.dumps(original_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_adaptive_decision_and_publication_prototypes_are_disabled(tmp_path):
    manifest_path = _write_campaign(tmp_path)
    protocol = protocol_module.build_protocol(manifest_path)
    with pytest.raises(protocol_module.ProtocolError, match="disabled"):
        protocol_module.build_decision_record(
            protocol,
            [],
            scruton_number=5.18,
            reduced_velocity=6.25,
            evaluation_cycles=200,
            gate_results={"anything": True},
        )
    with pytest.raises(protocol_module.ProtocolError, match="disabled"):
        protocol_module.derive_publication_plan(protocol, [])


def test_cli_generates_and_validates_only_outside_campaign(tmp_path, capsys):
    manifest_path = _write_campaign(tmp_path)
    output = tmp_path / "evidence" / "stationary_protocol.json"

    assert (
        protocol_module.main(
            [
                "generate",
                "--campaign-manifest",
                str(manifest_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert (
        protocol_module.main(
            [
                "validate",
                "--campaign-manifest",
                str(manifest_path),
                "--protocol",
                str(output),
            ]
        )
        == 0
    )
    assert '"valid": true' in capsys.readouterr().out

    with pytest.raises(protocol_module.ProtocolError, match="outside"):
        protocol_module.main(
            [
                "generate",
                "--campaign-manifest",
                str(manifest_path),
                "--output",
                str(manifest_path.parent / "protocol.json"),
            ]
        )
