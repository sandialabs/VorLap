import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from examples.ehrmann2014_naca0018.build_airfoil_work_map import (
    SCHEMA,
    build_work_map_from_manifest,
)
from vorlap.qblade_airfoil_work_map import AirfoilForceMotionWorkMap

_TRACKED_REQUEST = (
    Path(__file__).parents[1]
    / "examples"
    / "ehrmann2014_naca0018"
    / "work_map_data_request.json"
)
_TRACKED_SPECTRUM = (
    Path(__file__).parents[1]
    / "examples"
    / "ReferenceHVAWTQBladeModel"
    / "10m_HVAWT_reference_turbine"
    / "VorLapAirfoils"
    / "NACA0018.h5"
)
_TRACKED_SPECTRUM_SHA256 = (
    "64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _request() -> dict:
    return json.loads(_TRACKED_REQUEST.read_text(encoding="utf-8"))


def _write_spectrum(path: Path, request: dict) -> None:
    del request
    assert _sha(_TRACKED_SPECTRUM) == _TRACKED_SPECTRUM_SHA256
    shutil.copy2(_TRACKED_SPECTRUM, path)


def _write_history(
    path: Path,
    amplitude: float,
    gain: float,
    *,
    frequency: float,
    cycles: int = 13,
    samples_per_cycle: int = 100,
    time_offset: float = 0.0,
) -> None:
    omega = 2.0 * math.pi * frequency
    time = np.linspace(
        time_offset,
        time_offset + cycles / frequency,
        cycles * samples_per_cycle + 1,
    )
    displacement = amplitude * np.sin(omega * time)
    velocity = amplitude * omega * np.cos(omega * time)
    base_force = 2.0 * np.sin(omega * time + 0.3)
    force = base_force + gain * velocity
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            ("time_s", "displacement_m", "velocity_mps", "crossflow_force_n")
        )
        for values in zip(time, displacement, velocity, force, strict=True):
            writer.writerow((format(value, ".17g") for value in values))


def _semantic_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _producer_provenance(
    tmp_path: Path, request_path: Path, request: dict, cases: list[dict]
) -> dict:
    provenance_root = tmp_path / "provenance"
    nalu_root = provenance_root / "nalu"
    qblade_root = provenance_root / "qblade"
    nalu_root.mkdir(parents=True)
    (qblade_root / "case_audits").mkdir(parents=True)
    nalu_executable = {"path": "/frozen/naluX", "sha256": "d" * 64}
    request_identity = {
        "path": str(request_path),
        "sha256": _sha(request_path),
        "schema": request["schema"],
    }
    physical = {
        key: request["frozen_fluid_and_section"][key]
        for key in (
            "fluid_density_kgpm3",
            "fluid_dynamic_viscosity_pas",
            "reference_chord_m",
            "section_span_m",
        )
    }
    rho = physical["fluid_density_kgpm3"]
    mu = physical["fluid_dynamic_viscosity_pas"]
    chord = physical["reference_chord_m"]

    nalu_history_contract = {
        "time_origin": "precursor_restart",
        "first_row": "dt",
        "last_row": "(13*samples_per_cycle+1)*dt",
        "complete_cycle_intervals": 13,
        "raw_absolute_time_rule": "t0+j*dt for j=1..13*N+1",
        "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
        "columns": [
            "time_s",
            "displacement_m",
            "velocity_mps",
            "crossflow_force_n",
        ],
        "float_encoding": ".17g",
        "pairing_policy": "exact_array_equal_time_displacement_velocity_no_interpolation",
        "force_units": "N_total_over_4m_span",
    }
    qblade_history_contract = {
        "first_row": "dt",
        "last_row": "(13*samples_per_cycle+1)*dt",
        "wrapper_steps_rule": "13*samples_per_cycle+2",
        "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
        "columns": nalu_history_contract["columns"],
        "float_encoding": ".17g",
        "force_channel": "Aerodynamic Force in Hub Y_h Direction [N]",
        "force_frame": "hub_Y_equals_global_crossflow_Y",
        "force_sign": "positive_fluid_on_body_in_positive_global_Y",
        "force_units": "N_total_over_4m_span",
    }

    def derived(case: dict) -> dict:
        speed = case["reynolds_number"] * mu / (rho * chord)
        frequency = case["reduced_frequency_k"] * speed / (math.pi * chord)
        period = 1.0 / frequency
        samples = math.ceil(
            max(100.0, 120.0 * 0.4 * math.pi / case["reduced_frequency_k"])
        )
        rows = 13 * samples + 1
        precursor_restart_time = 20.0 * math.pi * chord / speed
        precursor_maximum_dt = chord / (120.0 * 0.4 * speed)
        precursor_steps = math.ceil(precursor_restart_time / precursor_maximum_dt)
        return {
            "wind_speed_mps": speed,
            "frequency_hz": frequency,
            "period_s": period,
            "amplitude_m": case["amplitude_ratio"] * chord,
            "samples_per_cycle": samples,
            "time_step_s": period / samples,
            "rows": rows,
            "relative_history_start_s": period / samples,
            "relative_history_end_s": rows * period / samples,
            "precursor_restart_time_s": precursor_restart_time,
            "precursor_steps": precursor_steps,
            "phase_cycles_at_restart": 20.0 * case["reduced_frequency_k"],
        }

    sorted_cases = sorted(cases, key=lambda row: row["case_id"])
    nalu_campaign_cases = []
    nalu_result_cases = []
    qblade_campaign_cases = []
    qblade_result_cases = []
    for index, case in enumerate(sorted_cases):
        values = derived(case)
        common = {
            "index": index,
            "case_id": case["case_id"],
            **{
                key: case[key]
                for key in (
                    "incidence_deg",
                    "reynolds_number",
                    "reduced_frequency_k",
                    "amplitude_ratio",
                    "direction_deg_mod_180",
                    "discard_cycles",
                )
            },
            **{
                key: values[key]
                for key in (
                    "wind_speed_mps",
                    "frequency_hz",
                    "period_s",
                    "amplitude_m",
                    "samples_per_cycle",
                    "time_step_s",
                    "relative_history_start_s",
                    "relative_history_end_s",
                )
            },
        }
        nalu_pending = {
            "path": f"histories/{case['case_id']}_nalu.csv",
            "status": "pending",
            "sha256": None,
            "rows_expected": values["rows"],
            "columns": nalu_history_contract["columns"],
        }
        nalu_campaign_cases.append(
            {
                **common,
                "retained_cycles": 8,
                "updates_after_restart": values["rows"],
                "precursor_restart_step": values["precursor_steps"],
                "precursor_restart_time_s": values["precursor_restart_time_s"],
                "phase_cycles_at_restart": values["phase_cycles_at_restart"],
                "canonical_history": nalu_pending,
            }
        )
        nalu_result_cases.append(
            {
                **common,
                "solver_status": "complete",
                "failure_marker": None,
                "selected_force_rows": values["rows"],
                "precursor_restart_step": values["precursor_steps"],
                "runtime_identity": {
                    "status": "verified",
                    "nalu_executable": nalu_executable,
                    "observed_solver_version": "v2.2.1",
                    "observed_solver_git_commit": (
                        "ffa9de729df2a11b5241fdeb7628e7fab9f48f9b"
                    ),
                    "terminal_step": values["precursor_steps"] + values["rows"],
                    "log": {
                        "path": f"motion_cases/{case['case_id']}/log.out",
                        "sha256": hashlib.sha256(
                            f"nalu-motion-{case['case_id']}".encode()
                        ).hexdigest(),
                    },
                },
                "canonical_history": {
                    **nalu_pending,
                    "status": "complete",
                    "sha256": case["high_fidelity_history"]["sha256"],
                    "rows": values["rows"],
                },
            }
        )
        qblade_pending = {
            "path": f"histories/{case['case_id']}_qblade.csv",
            "status": "pending",
            "sha256": None,
            "rows_expected": values["rows"],
            "columns": qblade_history_contract["columns"],
        }
        qblade_campaign_cases.append(
            {
                **common,
                "qblade_internal_incidence_deg": 90.0,
                "retained_cycles": 8,
                "expected_result_rows": values["rows"],
                "wrapper_steps": values["rows"] + 1,
                "canonical_history": qblade_pending,
            }
        )
        qblade_result_cases.append(
            {
                **{
                    key: common[key]
                    for key in (
                        "case_id",
                        "incidence_deg",
                        "reynolds_number",
                        "reduced_frequency_k",
                        "amplitude_ratio",
                        "direction_deg_mod_180",
                        "discard_cycles",
                    )
                },
                "canonical_history": {
                    **qblade_pending,
                    "status": "complete",
                    "sha256": case["exact_runtime_base_history"]["sha256"],
                    "rows": values["rows"],
                },
            }
        )

    nalu_solver = {
        "name": "Nalu-Wind",
        "version": "v2.2.1",
        "expected_git_commit": "ffa9de729df2a11b5241fdeb7628e7fab9f48f9b",
        "turbulence_model": "sst_iddes",
        "transition_model": "none",
    }
    nalu_mesh = {
        "path": "/frozen/NACA0018-90.exo",
        "sha256": "a" * 64,
        "source_chord_m": 1.0,
        "source_span_m": 4.0,
        "transformation_xyz": [0.23, 0.23, 1.0],
        "transformation_stage": "precursor_only",
        "required_parts": [
            "fluid-hex",
            "wing",
            "wing-pp",
            "inlet",
            "outlet",
            "front",
            "back",
        ],
    }
    nalu_campaign = {
        "schema": "vorlap-naca0018-nalu-prescribed-motion-campaign-v1",
        "request": request_identity,
        "solver": nalu_solver,
        "mesh": nalu_mesh,
        "fluid_and_section": physical,
        "numerics": {
            "discard_cycles": 5,
            "retained_cycles": 8,
            "updates_rule": "13*samples_per_cycle+1",
            "force_surface": ["wing", "wing-pp"],
            "crossflow_force_expression": "Fpy+Fvy",
        },
        "history_contract": nalu_history_contract,
        "precursors": [
            {
                "precursor_id": f"precursor_re{int(reynolds):06d}",
                "reynolds_number": reynolds,
                "precursor_steps": 3016,
            }
            for reynolds in (64000.0, 100000.0, 140000.0)
        ],
        "cases": nalu_campaign_cases,
    }
    nalu_campaign["fingerprint"] = _semantic_sha(nalu_campaign)
    nalu_campaign_path = nalu_root / "campaign_manifest.json"
    _write_json(nalu_campaign_path, nalu_campaign)
    mesh_audit = {
        "schema": "vorlap-naca0018-exodus-geometry-audit-v1",
        "status": "pass",
        "campaign_fingerprint": nalu_campaign["fingerprint"],
        "mesh": {"path": nalu_mesh["path"], "sha256": nalu_mesh["sha256"]},
        "combined_force_surface": ["wing", "wing-pp"],
        "element_block_names": ["fluid-hex"],
        "side_set_names": ["wing", "wing-pp", "inlet", "outlet", "front", "back"],
        "chord_axis": "global_Y",
        "incidence_orientation_deg": -90.0,
        "source_chord_m": 1.0,
        "source_span_m": 4.0,
    }
    mesh_audit_path = nalu_root / "mesh_audit.json"
    _write_json(mesh_audit_path, mesh_audit)
    scaling_identities = []
    target_pass = {}
    maximum_pass = {}
    for nodes, hours in ((2, 10.0), (4, 6.0), (8, 4.0)):
        passes_target = hours <= 2.0
        passes_maximum = hours <= 8.0
        scaling = {
            "schema": "vorlap-naca0018-nalu-scaling-report-v1",
            "status": "pass" if passes_maximum else "fail",
            "campaign_fingerprint": nalu_campaign["fingerprint"],
            "nodes": nodes,
            "ranks": nodes * 112,
            "reynolds_number": 64000.0,
            "probe_updates": 64,
            "observed_solver_version": "v2.2.1",
            "observed_solver_git_commit": nalu_solver["expected_git_commit"],
            "terminal_step": 64,
            "nalu_executable": nalu_executable,
            "projected_precursor_hours": hours,
            "projected_longest_motion_hours": hours,
            "target_walltime_hours": 2.0,
            "target_walltime_pass": passes_target,
            "maximum_walltime_hours": 8.0,
            "maximum_walltime_pass": passes_maximum,
        }
        path = nalu_root / f"scaling_nodes{nodes}.json"
        _write_json(path, scaling)
        scaling_identities.append(
            {
                "nodes": nodes,
                "file": {
                    "path": path.relative_to(tmp_path).as_posix(),
                    "sha256": _sha(path),
                },
            }
        )
        target_pass[nodes] = passes_target
        maximum_pass[nodes] = passes_maximum
    target_nodes = [node for node, passed in target_pass.items() if passed]
    eligible_nodes = target_nodes or [
        node for node, passed in maximum_pass.items() if passed
    ]
    recommended = min(eligible_nodes)
    selection_tier = "target_2h" if target_nodes else "fallback_8h"
    preflight_case = nalu_result_cases[35]
    nalu_results = {
        "schema": "vorlap-naca0018-nalu-prescribed-motion-results-v1",
        "status": "complete",
        "campaign_fingerprint": nalu_campaign["fingerprint"],
        "campaign_manifest": "campaign_manifest.json",
        "request": request_identity,
        "mesh": nalu_mesh,
        "solver": nalu_solver,
        "fluid_and_section": physical,
        "force_contract": {
            "surface": ["wing", "wing-pp"],
            "crossflow_expression": "Fpy+Fvy",
            "sign": "positive_fluid_on_body_global_positive_Y",
            "units": "N_total_over_declared_4m_span",
        },
        "history_contract": nalu_history_contract,
        "mesh_audit": {
            "status": "pass",
            "path": "mesh_audit.json",
            "sha256": _sha(mesh_audit_path),
        },
        "scaling_gate": {
            "status": "pass",
            "required_probe_count": 3,
            "complete_probe_count": 3,
            "recommended_nodes": recommended,
            "recommended_ranks": recommended * 112,
            "selection_tier": selection_tier,
            "recommended_walltime": (
                "02:00:00" if selection_tier == "target_2h" else "08:00:00"
            ),
            "target_walltime_hours": 2.0,
            "maximum_walltime_hours": 8.0,
            "by_nodes": {
                str(nodes): {
                    "complete_reynolds_sweep": True,
                    "target_walltime_pass": target_pass[nodes],
                    "maximum_walltime_pass": maximum_pass[nodes],
                    "pass": maximum_pass[nodes],
                    "reports": 1,
                }
                for nodes in (2, 4, 8)
            },
            "runtime_identity": {
                "status": "verified",
                "observed_versions": ["v2.2.1"],
                "observed_git_commits": [nalu_solver["expected_git_commit"]],
                "nalu_executable": nalu_executable,
            },
        },
        "precursors": [
            {
                "precursor_id": f"precursor_re{int(reynolds):06d}",
                "reynolds_number": reynolds,
                "status": "complete",
                "required_step": 3016,
                "maximum_logged_step": 3016,
                "runtime_identity": {
                    "status": "verified",
                    "nalu_executable": nalu_executable,
                    "observed_solver_version": "v2.2.1",
                    "observed_solver_git_commit": nalu_solver["expected_git_commit"],
                    "terminal_step": 3016,
                    "log": {
                        "path": f"precursors/re{int(reynolds):06d}/log.out",
                        "sha256": hashlib.sha256(
                            f"nalu-precursor-{reynolds}".encode()
                        ).hexdigest(),
                    },
                },
            }
            for reynolds in (64000.0, 100000.0, 140000.0)
        ],
        "motion_preflight": {
            "case_id": preflight_case["case_id"],
            "case_index": 35,
            "status": "complete",
            "history_sha256": preflight_case["canonical_history"]["sha256"],
        },
        "summary": {
            "complete_precursors": 3,
            "required_precursors": 3,
            "complete_motion_cases": 48,
            "failed_motion_cases": 0,
            "failed_precursors": 0,
            "required_motion_cases": 48,
        },
        "cases": nalu_result_cases,
    }
    nalu_results_path = nalu_root / "nalu_results_manifest.json"
    _write_json(nalu_results_path, nalu_results)

    qblade_runtime = {
        "root": "/frozen/QBladeCE_2.0.9.7",
        "version": "2.0.9.7",
        "library": {
            "path": "/frozen/libQBladeCE.so",
            "sha256": "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d",
        },
        "runtime_dependencies": [],
        "runtime_dependency_fingerprint": "c50c6909317c754caec4130ccbc8e2ed3f705f810f07da11e7ce3ad003499089",
    }
    geometry = {
        "profile": "NACA0018",
        "blade_count": 1,
        "station_count": 21,
        "chord_axis_global": [0.0, 1.0, 0.0],
        "normal_axis_global": [-1.0, 0.0, 0.0],
        "ambient_wind_global": [1.0, 0.0, 0.0],
        "qblade_internal_incidence_deg": 90.0,
        "vorlap_relative_wind_incidence_deg": -90.0,
        "mapping": "QBlade +90 uses the opposite relative-wind sign convention from VorLap atan2(U.normal,U.chord)=-90; pitch and twist remain zero",
        "hub_y_equals_global_y": "zero_yaw_zero_shaft_tilt_zero_global_rotation",
    }
    native = {
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
    }
    external = {
        "controller_type": 0,
        "external_library_entries": 0,
        "sil_vorlap": False,
        "sil_constant_force": False,
        "expected_native_callback_updates": 0,
    }
    execution = {
        "threads_per_case": 1,
        "total_case_count": 48,
        "preflight_cpu_slots": 1,
        "matrix_cpu_slots": 47,
        "local_docker_max_cpu_slots": 8,
    }
    qblade_preflight_contract = {
        "case_index": 35,
        "case_id": "re140000_k0p45_a0p0500",
        "required_before_matrix": True,
    }
    qblade_campaign = {
        "schema": "vorlap-naca0018-qblade-prescribed-motion-campaign-v1",
        "request": request_identity,
        "runtime": qblade_runtime,
        "fluid_and_section": physical,
        "geometry_and_frames": geometry,
        "native_aero": native,
        "external_forces": external,
        "history_contract": qblade_history_contract,
        "execution": execution,
        "preflight_gate": qblade_preflight_contract,
        "cases": qblade_campaign_cases,
    }
    qblade_campaign["fingerprint"] = _semantic_sha(qblade_campaign)
    qblade_campaign_path = qblade_root / "campaign_manifest.json"
    _write_json(qblade_campaign_path, qblade_campaign)
    scheduler = {
        "schema": "vorlap-naca0018-qblade-prescribed-motion-scheduler-v2",
        "stage": "preflight",
        "status": "complete",
        "campaign_fingerprint": qblade_campaign["fingerprint"],
        "preflight_gate": {
            "status": "complete",
            "case_index": 35,
            "case_id": "re140000_k0p45_a0p0500",
        },
        "allocation": {
            "workers": 1,
            "threads_per_case": 1,
            "allocated_physical_cpus": 1,
        },
        "execution_environment": {
            "kind": "flight_slurm",
            "platform": "linux_x86_64",
            "slurm_job_id": "123456",
        },
        "cases": [
            {"index": 35, "case_id": "re140000_k0p45_a0p0500", "status": "complete"}
        ],
    }
    scheduler_path = qblade_root / "qblade_preflight_scheduler_report.json"
    _write_json(scheduler_path, scheduler)
    audit_identities = []
    audit_sha_by_id = {}
    for result_case in qblade_result_cases:
        case_id = result_case["case_id"]
        values = derived(result_case)
        audit = {
            "schema": "vorlap-naca0018-qblade-prescribed-motion-case-audit-v1",
            "status": "complete",
            "case_id": case_id,
            "campaign_fingerprint": qblade_campaign["fingerprint"],
            "canonical_history": result_case["canonical_history"],
            "runtime": {
                "qblade_library_sha256": qblade_runtime["library"]["sha256"],
                "qblade_runtime_dependency_fingerprint": qblade_runtime[
                    "runtime_dependency_fingerprint"
                ],
                "threads": 1,
                "coupling_mode": "none",
                "external_callback_updates": 0,
            },
            "channel_contract": {
                "raw_export_rows": values["rows"] + 1,
                "discarded_wrapper_t0_rows": 1,
                "prescribed_motion": "Y_g Prescr. Platform Trans. [m]",
                "rigid_blade_position": "Y_g Pos. BLD_1 pos 1.000 [m]",
                "crossflow_force": "Aerodynamic Force in Hub Y_h Direction [N]",
                "crossflow_force_units": "N_total_over_4m_span",
                "crossflow_force_sign": "positive_fluid_on_body_positive_global_Y",
                "qblade_aoa": "Angle of Attack at 0.25c BLD_1 PAN_10 [deg]",
                "qblade_reynolds": "Reynolds Number BLD_1 PAN_10 [-]",
            },
            "geometry_frame_audit": {
                "qblade_internal_incidence_target_deg": 90.0,
                "vorlap_incidence_deg": -90.0,
                "median_abs_qblade_aoa_deg": 90.0,
                "max_abs_qblade_aoa_deviation_from_90_deg": 1.0,
                "median_qblade_panel_reynolds": result_case["reynolds_number"],
                "analytic_platform_max_abs_error_m": min(
                    1e-10, 1e-6 * values["amplitude_m"]
                ),
                "tip_follows_platform_max_abs_error_m": 1e-9,
            },
            "force_statistics_n": {
                "minimum": -2.0,
                "maximum": 2.0,
                "mean": 0.0,
                "rms": 1.0,
            },
        }
        path = qblade_root / "case_audits" / f"{case_id}.json"
        _write_json(path, audit)
        digest = _sha(path)
        result_case["case_audit"] = {
            "path": f"results/{case_id}/audit.json",
            "sha256": digest,
        }
        audit_identities.append(
            {
                "case_id": case_id,
                "file": {
                    "path": path.relative_to(tmp_path).as_posix(),
                    "sha256": digest,
                },
            }
        )
        audit_sha_by_id[case_id] = digest
    preflight_id = "re140000_k0p45_a0p0500"
    qblade_results = {
        "schema": "vorlap-naca0018-qblade-prescribed-motion-results-v1",
        "status": "complete",
        "campaign_manifest": "campaign_manifest.json",
        "campaign_fingerprint": qblade_campaign["fingerprint"],
        "request": request_identity,
        "fluid_and_section": physical,
        "runtime": qblade_runtime,
        "geometry_and_frames": geometry,
        "native_aero": native,
        "external_forces": external,
        "history_contract": qblade_history_contract,
        "preflight_gate": {
            "campaign_contract": qblade_preflight_contract,
            "status": "complete",
            "case_index": 35,
            "case_id": preflight_id,
            "case_audit": {
                "path": f"results/{preflight_id}/audit.json",
                "sha256": audit_sha_by_id[preflight_id],
            },
            "scheduler_report": {
                "path": "qblade_preflight_scheduler_report.json",
                "sha256": _sha(scheduler_path),
            },
            "scheduler_schema": scheduler["schema"],
            "scheduler_stage": "preflight",
            "scheduler_status": "complete",
            "execution_environment": scheduler["execution_environment"],
        },
        "summary": {"required_cases": 48, "complete_cases": 48, "failed_cases": 0},
        "cases": qblade_result_cases,
    }
    qblade_results_path = qblade_root / "qblade_results.json"
    _write_json(qblade_results_path, qblade_results)
    return {
        "high_fidelity": {
            "kind": "nalu_wind_prescribed_motion",
            "results_manifest": {
                "path": nalu_results_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha(nalu_results_path),
            },
            "campaign_manifest": {
                "path": nalu_campaign_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha(nalu_campaign_path),
            },
            "mesh_audit": {
                "path": mesh_audit_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha(mesh_audit_path),
            },
            "scaling_reports": scaling_identities,
        },
        "exact_runtime_base": {
            "kind": "native_qblade_prescribed_motion",
            "results_manifest": {
                "path": qblade_results_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha(qblade_results_path),
            },
            "campaign_manifest": {
                "path": qblade_campaign_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha(qblade_campaign_path),
            },
            "preflight_scheduler_report": {
                "path": scheduler_path.relative_to(tmp_path).as_posix(),
                "sha256": _sha(scheduler_path),
            },
            "case_audits": audit_identities,
        },
    }


def _bundle(tmp_path: Path) -> Path:
    request = _request()
    request_path = tmp_path / "request.json"
    request_path.write_bytes(_TRACKED_REQUEST.read_bytes())
    spectrum_path = tmp_path / "NACA0018.h5"
    _write_spectrum(spectrum_path, request)
    physical = request["frozen_fluid_and_section"]
    density = physical["fluid_density_kgpm3"]
    viscosity = physical["fluid_dynamic_viscosity_pas"]
    chord = physical["reference_chord_m"]
    cases = []
    axes = request["frozen_axes"]
    nonzero_amplitudes = [value for value in axes["amplitude_ratio"] if value > 0.0]
    vertices = itertools.product(
        axes["incidence_deg"],
        axes["reynolds_number"],
        axes["reduced_frequency_k"],
        nonzero_amplitudes,
        axes["direction_deg_mod_180"],
    )
    for index, (
        incidence,
        reynolds,
        reduced_frequency,
        amplitude_ratio,
        direction,
    ) in enumerate(vertices):
        del index
        label = (
            (
                f"re{int(round(reynolds)):06d}_"
                f"k{reduced_frequency:.2f}_a{amplitude_ratio:.4f}"
            )
            .replace("-", "m")
            .replace(".", "p")
        )
        speed = reynolds * viscosity / (density * chord)
        frequency = reduced_frequency * speed / (math.pi * chord)
        amplitude = amplitude_ratio * chord
        gain = -10.0 if amplitude_ratio == max(nonzero_amplitudes) else 10.0
        high_path = tmp_path / f"{label}_high.csv"
        base_path = tmp_path / f"{label}_base.csv"
        _write_history(high_path, amplitude, gain, frequency=frequency)
        _write_history(base_path, amplitude, 0.0, frequency=frequency)
        cases.append(
            {
                "case_id": label,
                "incidence_deg": incidence,
                "reynolds_number": reynolds,
                "reduced_frequency_k": reduced_frequency,
                "amplitude_ratio": amplitude_ratio,
                "direction_deg_mod_180": direction,
                "discard_cycles": request["minimum_accepted_history"]["discard_cycles"],
                "high_fidelity_history": {
                    "path": high_path.name,
                    "sha256": _sha(high_path),
                },
                "exact_runtime_base_history": {
                    "path": base_path.name,
                    "sha256": _sha(base_path),
                },
            }
        )
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "profile": "NACA0018",
        "request": {"path": request_path.name, "sha256": _sha(request_path)},
        "static_spectrum": {"path": spectrum_path.name, "sha256": _sha(spectrum_path)},
        "producer_provenance": _producer_provenance(
            tmp_path, request_path, request, cases
        ),
        "fluid_and_section": {
            "fluid_density_kgpm3": physical["fluid_density_kgpm3"],
            "fluid_dynamic_viscosity_pas": physical["fluid_dynamic_viscosity_pas"],
            "reference_chord_m": physical["reference_chord_m"],
            "section_span_m": physical["section_span_m"],
            "static_spectrum_frequency_depth": 3,
        },
        "cases": cases,
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _rebind_qblade_case_history(
    tmp_path: Path, manifest: dict, case_id: str, history_sha256: str
) -> None:
    provenance = manifest["producer_provenance"]["exact_runtime_base"]
    results_path = tmp_path / provenance["results_manifest"]["path"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    result_case = next(row for row in results["cases"] if row["case_id"] == case_id)
    result_case["canonical_history"]["sha256"] = history_sha256
    audit_identity = next(
        row for row in provenance["case_audits"] if row["case_id"] == case_id
    )["file"]
    audit_path = tmp_path / audit_identity["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["canonical_history"]["sha256"] = history_sha256
    _write_json(audit_path, audit)
    audit_identity["sha256"] = _sha(audit_path)
    result_case["case_audit"]["sha256"] = audit_identity["sha256"]
    _write_json(results_path, results)
    provenance["results_manifest"]["sha256"] = _sha(results_path)


def test_builds_qualified_map_from_paired_histories_and_exact_origin(tmp_path):
    manifest = _bundle(tmp_path)
    map_path = tmp_path / "work_map.json"
    report_path = tmp_path / "report.json"
    report = build_work_map_from_manifest(
        manifest,
        output_map_path=map_path,
        output_report_path=report_path,
    )
    assert report["status"] == "qualified"
    assert report["static_spectrum_sha256"] == _TRACKED_SPECTRUM_SHA256
    assert report["case_count"] == 48
    assert report["origin_vertex_count"] == 12
    assert report["quality_contract"] == {
        "minimum_discard_cycles": 5,
        "minimum_complete_cycles_after_discard": 8,
        "minimum_samples_per_cycle": 100,
        "fluid_density_kgpm3": 1.225,
        "fluid_dynamic_viscosity_pas": 1.813e-5,
        "reference_chord_m": 0.23,
        "section_span_m": 4.0,
    }
    assert all(case["included_cycle_count"] >= 8 for case in report["cases"])
    assert all(
        case["minimum_samples_per_included_cycle"] >= 100 for case in report["cases"]
    )
    assert report["claim_boundary"] == {
        "static_fft_phase_used": False,
        "ehrmann_free_response_used_for_fit": False,
        "grinderslev_power_used_for_naca_magnitude": False,
        "supplemental_work_is_high_fidelity_minus_exact_runtime_base": True,
    }
    model = AirfoilForceMotionWorkMap.from_json(map_path)
    origin = model.query(
        incidence_deg=-90.0,
        reynolds_number=64000.0,
        reduced_frequency_k=0.49,
        amplitude_ratio=0.0,
        direction_deg=0.0,
        runtime_available_force_coefficient=math.sqrt(0.05),
    )
    positive = model.query(
        incidence_deg=-90.0,
        reynolds_number=64000.0,
        reduced_frequency_k=0.49,
        amplitude_ratio=0.005,
        direction_deg=0.0,
        runtime_available_force_coefficient=math.sqrt(0.05),
    )
    high = model.query(
        incidence_deg=-90.0,
        reynolds_number=64000.0,
        reduced_frequency_k=0.49,
        amplitude_ratio=0.05,
        direction_deg=0.0,
        runtime_available_force_coefficient=math.sqrt(0.05),
    )
    assert origin.signed_power_coefficient == 0.0
    assert positive.signed_power_coefficient > 0.0
    assert high.signed_power_coefficient < 0.0
    assert {source["role"] for source in model.sources} == {
        "prescribed_motion_high_fidelity",
        "exact_runtime_base",
        "prescribed_motion_high_fidelity_producer_provenance",
        "exact_runtime_base_producer_provenance",
        "stationary_force_spectrum",
        "frozen_data_request",
    }
    assert set(report["producer_provenance_digests"]) == {
        "high_fidelity",
        "exact_runtime_base",
    }


def test_rejects_paired_motion_grid_mismatch_before_identification(tmp_path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    request = _request()
    physical = request["frozen_fluid_and_section"]
    case = manifest["cases"][0]
    speed = (
        case["reynolds_number"]
        * physical["fluid_dynamic_viscosity_pas"]
        / (physical["fluid_density_kgpm3"] * physical["reference_chord_m"])
    )
    frequency = (
        case["reduced_frequency_k"] * speed / (math.pi * physical["reference_chord_m"])
    )
    amplitude = case["amplitude_ratio"] * physical["reference_chord_m"]
    base = tmp_path / manifest["cases"][0]["exact_runtime_base_history"]["path"]
    _write_history(base, amplitude, 0.0, frequency=frequency, time_offset=1.0e-4)
    manifest["cases"][0]["exact_runtime_base_history"]["sha256"] = _sha(base)
    _rebind_qblade_case_history(
        tmp_path,
        manifest,
        manifest["cases"][0]["case_id"],
        _sha(base),
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="time_s histories are not exactly identical"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )


def test_rejects_missing_frozen_nonzero_vertex(tmp_path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"].pop()
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="do not cover"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )


def test_rejects_self_consistent_but_untracked_request_bytes(tmp_path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    request_path = tmp_path / manifest["request"]["path"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["frozen_nonleakage_policy"]["ehrmann_free_response_used_for_fit"] = True
    _write_json(request_path, request)
    manifest["request"]["sha256"] = _sha(request_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="do not exactly match tracked"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )


def test_rejects_self_consistent_but_untracked_stationary_spectrum(tmp_path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spectrum = tmp_path / manifest["static_spectrum"]["path"]
    spectrum.write_bytes(spectrum.read_bytes() + b"changed")
    manifest["static_spectrum"]["sha256"] = _sha(spectrum)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="not the tracked NACA0018.h5"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )


def test_rejects_manifest_physics_or_discard_below_tracked_contract(tmp_path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fluid_and_section"]["section_span_m"] = 1.0
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="must exactly match"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )

    manifest["fluid_and_section"]["section_span_m"] = 4.0
    manifest["cases"][0]["discard_cycles"] = 4
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="not cross-linked|tracked minimum"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )


def test_rejects_producer_snapshot_byte_tampering(tmp_path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["producer_provenance"]["exact_runtime_base"][
        "preflight_scheduler_report"
    ]
    scheduler = tmp_path / identity["path"]
    scheduler.write_bytes(scheduler.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="preflight scheduler snapshot SHA-256"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )


def test_rejects_self_consistent_producer_semantic_tamper(tmp_path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["producer_provenance"]["exact_runtime_base"]["results_manifest"]
    results_path = tmp_path / identity["path"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["native_aero"]["enabled"] = False
    _write_json(results_path, results)
    identity["sha256"] = _sha(results_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="native-aerodynamics contract"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )


def test_rejects_self_consistent_nalu_completed_run_identity_tamper(tmp_path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["producer_provenance"]["high_fidelity"]["results_manifest"]
    results_path = tmp_path / identity["path"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["cases"][0]["runtime_identity"]["nalu_executable"]["sha256"] = "e" * 64
    _write_json(results_path, results)
    identity["sha256"] = _sha(results_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="does not prove the frozen Nalu runtime"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )


def test_rejects_producer_history_sha_not_crosslinked(tmp_path):
    manifest_path = _bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest["producer_provenance"]["high_fidelity"]["results_manifest"]
    results_path = tmp_path / identity["path"]
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["cases"][0]["canonical_history"]["sha256"] = "b" * 64
    _write_json(results_path, results)
    identity["sha256"] = _sha(results_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="canonical history is not cross-linked"):
        build_work_map_from_manifest(
            manifest_path,
            output_map_path=tmp_path / "map.json",
            output_report_path=tmp_path / "report.json",
        )
