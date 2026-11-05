import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from examples.ehrmann2014_naca0018 import (
    assemble_airfoil_work_map_identification as assembly,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_history(path: Path, force_offset: float = 0.0) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(assembly.HISTORY_COLUMNS)
        for index in range(1, 5):
            time = index * 0.01
            writer.writerow(
                (
                    format(time, ".17g"),
                    format(0.001 * math.sin(index), ".17g"),
                    format(0.1 * math.cos(index), ".17g"),
                    format(force_offset + index, ".17g"),
                )
            )
    return {
        "path": path.name,
        "status": "complete",
        "sha256": _sha(path),
        "rows": 4,
        "rows_expected": 4,
        "columns": list(assembly.HISTORY_COLUMNS),
    }


def _results(root: Path, *, qblade: bool) -> Path:
    root.mkdir()
    cases = []
    index = 0
    for reynolds in (64_000.0, 100_000.0, 140_000.0):
        for reduced_frequency in (0.45, 0.65, 0.8, 1.05):
            for amplitude in (0.005, 0.0125, 0.025, 0.05):
                case_id = f"case_{index:02d}"
                history_path = root / f"{case_id}.csv"
                history = _write_history(
                    history_path, force_offset=0.5 if qblade else 1.0
                )
                cases.append(
                    {
                        "case_id": case_id,
                        "incidence_deg": -90.0,
                        "reynolds_number": reynolds,
                        "reduced_frequency_k": reduced_frequency,
                        "amplitude_ratio": amplitude,
                        "direction_deg_mod_180": 0.0,
                        "discard_cycles": 5,
                        "canonical_history": history,
                    }
                )
                index += 1
    payload = {
        "schema": (
            assembly.QBLADE_RESULTS_SCHEMA if qblade else assembly.NALU_RESULTS_SCHEMA
        ),
        "status": "complete",
        "request": {
            "sha256": _sha(assembly.REQUEST_PATH),
        },
        "fluid_and_section": {
            "fluid_density_kgpm3": 1.225,
            "fluid_dynamic_viscosity_pas": 1.813e-5,
            "reference_chord_m": 0.23,
            "section_span_m": 4.0,
        },
        "summary": {
            "complete_motion_cases": 48,
            "failed_motion_cases": 0,
            "required_motion_cases": 48,
        },
        "cases": cases,
    }
    path = root / (
        "qblade_results_manifest.json" if qblade else "nalu_results_manifest.json"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stub_producer_provenance(
    monkeypatch: pytest.MonkeyPatch, nalu_results: Path, qblade_results: Path
) -> None:
    nalu_root = nalu_results.parent
    qblade_root = qblade_results.parent
    nalu_campaign = nalu_root / "campaign_manifest.json"
    nalu_mesh_audit = nalu_root / "mesh_audit.json"
    nalu_campaign.write_text("{}\n", encoding="utf-8")
    nalu_mesh_audit.write_text("{}\n", encoding="utf-8")
    scaling = []
    for nodes in (2, 4, 8):
        path = nalu_root / f"scaling_nodes{nodes}.json"
        path.write_text("{}\n", encoding="utf-8")
        scaling.append((nodes, path))
    qblade_campaign = qblade_root / "campaign_manifest.json"
    preflight = qblade_root / "qblade_preflight_scheduler_report.json"
    qblade_campaign.write_text("{}\n", encoding="utf-8")
    preflight.write_text("{}\n", encoding="utf-8")
    audits = []
    for row in json.loads(qblade_results.read_text(encoding="utf-8"))["cases"]:
        path = qblade_root / f"{row['case_id']}_audit.json"
        path.write_text("{}\n", encoding="utf-8")
        audits.append((row["case_id"], path))
    monkeypatch.setattr(
        assembly,
        "_validate_nalu_producer",
        lambda *_args: {
            "campaign": nalu_campaign,
            "mesh_audit": nalu_mesh_audit,
            "scaling_reports": scaling,
        },
    )
    monkeypatch.setattr(
        assembly,
        "_validate_qblade_producer",
        lambda *_args: {
            "campaign": qblade_campaign,
            "preflight_scheduler_report": preflight,
            "case_audits": audits,
        },
    )


def _physical() -> dict[str, float]:
    return {
        "fluid_density_kgpm3": 1.225,
        "fluid_dynamic_viscosity_pas": 1.813e-5,
        "reference_chord_m": 0.23,
        "section_span_m": 4.0,
    }


def _canonical_producer_rows(kind: str) -> tuple[list[dict], list[dict]]:
    campaign_rows = []
    result_rows = []
    for reynolds in assembly.EXPECTED_REYNOLDS:
        for reduced_frequency in assembly.EXPECTED_K:
            for amplitude in assembly.EXPECTED_AMPLITUDES:
                values = assembly._motion_values(
                    reynolds, reduced_frequency, amplitude, _physical()
                )
                case_id = assembly._case_id(reynolds, reduced_frequency, amplitude)
                count = int(values["expected_result_rows"])
                suffix = "nalu" if kind == "nalu" else "qblade"
                history = {
                    "path": f"histories/{case_id}_{suffix}.csv",
                    "status": "pending",
                    "sha256": None,
                    "rows_expected": count,
                    "columns": list(assembly.HISTORY_COLUMNS),
                }
                common = {
                    "index": len(campaign_rows),
                    "case_id": case_id,
                    "incidence_deg": -90.0,
                    "reynolds_number": reynolds,
                    "reduced_frequency_k": reduced_frequency,
                    "amplitude_ratio": amplitude,
                    "direction_deg_mod_180": 0.0,
                    "wind_speed_mps": values["wind_speed_mps"],
                    "frequency_hz": values["frequency_hz"],
                    "period_s": values["period_s"],
                    "amplitude_m": values["amplitude_m"],
                    "samples_per_cycle": values["samples_per_cycle"],
                    "time_step_s": values["time_step_s"],
                    "relative_history_start_s": values["relative_history_start_s"],
                    "relative_history_end_s": values["relative_history_end_s"],
                    "discard_cycles": 5,
                    "retained_cycles": 8,
                    "canonical_history": history,
                }
                complete_history = {
                    **history,
                    "status": "complete",
                    "sha256": hashlib.sha256(case_id.encode()).hexdigest(),
                    "rows": count,
                }
                if kind == "nalu":
                    common.update(
                        {
                            "updates_after_restart": count,
                            "precursor_restart_step": values["precursor_steps"],
                            "precursor_restart_time_s": values[
                                "precursor_restart_time_s"
                            ],
                        }
                    )
                    result = {
                        **common,
                        "solver_status": "complete",
                        "failure_marker": None,
                        "selected_force_rows": count,
                        "canonical_history": complete_history,
                    }
                else:
                    common.update(
                        {
                            "qblade_internal_incidence_deg": 90.0,
                            "expected_result_rows": count,
                            "wrapper_steps": values["wrapper_steps"],
                        }
                    )
                    result = {
                        key: common[key]
                        for key in (
                            "case_id",
                            *assembly.VERTEX_KEYS,
                            "discard_cycles",
                        )
                    }
                    result["canonical_history"] = complete_history
                campaign_rows.append(common)
                result_rows.append(result)
    return campaign_rows, result_rows


def _nalu_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "nalu_contract"
    root.mkdir()
    fingerprint = "1" * 64
    physical = _physical()
    campaign_rows, result_rows = _canonical_producer_rows("nalu")
    nalu_executable = {"path": "/frozen/naluX", "sha256": "d" * 64}

    def runtime_identity(name: str, terminal_step: int) -> dict[str, object]:
        path = root / "runtime_logs" / f"{name}.log"
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            "Nalu-Wind Version: v2.2.1\n"
            f"Nalu-Wind GIT Commit SHA: {assembly.EXPECTED_NALU_COMMIT}\n"
            f"Time Step Count: {terminal_step} Current Time: 1\n"
            f"Simulation Shall Complete: time/timestep: 1/{terminal_step}\n",
            encoding="utf-8",
        )
        return {
            "status": "verified",
            "nalu_executable": nalu_executable,
            "observed_solver_version": "v2.2.1",
            "observed_solver_git_commit": assembly.EXPECTED_NALU_COMMIT,
            "terminal_step": terminal_step,
            "log": {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha(path),
            },
        }

    for source, row in zip(campaign_rows, result_rows, strict=True):
        row["runtime_identity"] = runtime_identity(
            row["case_id"],
            int(source["precursor_restart_step"])
            + int(source["updates_after_restart"]),
        )
    solver = {
        "name": "Nalu-Wind",
        "version": "v2.2.1",
        "expected_git_commit": assembly.EXPECTED_NALU_COMMIT,
        "turbulence_model": "sst_iddes",
        "transition_model": "none",
    }
    mesh = {
        "path": "/bound/NACA0018-90.exo",
        "sha256": "2" * 64,
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
    request = {
        "path": str(assembly.REQUEST_PATH),
        "sha256": _sha(assembly.REQUEST_PATH),
    }
    scaling_cases = []
    by_nodes = {}
    for nodes, projected in ((2, 9.0), (4, 4.0), (8, 2.0)):
        target_pass = projected <= 2.0
        maximum_pass = projected <= 8.0
        log = root / f"scaling_{nodes}.log"
        log.write_text(
            "Nalu-Wind Version: v2.2.1\n"
            f"Nalu-Wind GIT Commit SHA: {assembly.EXPECTED_NALU_COMMIT}\n"
            "Time Step Count: 64 Current Time: 1\n"
            "Simulation Shall Complete: time/timestep: 1/64\n",
            encoding="utf-8",
        )
        report_path = root / f"scaling_{nodes}.json"
        report = {
            "schema": assembly.nalu_producer.SCALING_SCHEMA,
            "status": "pass" if maximum_pass else "fail",
            "campaign_fingerprint": fingerprint,
            "nodes": nodes,
            "ranks": nodes * 112,
            "reynolds_number": 64_000.0,
            "probe_updates": 64,
            "observed_solver_version": "v2.2.1",
            "observed_solver_git_commit": assembly.EXPECTED_NALU_COMMIT,
            "terminal_step": 64,
            "nalu_executable": nalu_executable,
            "projected_precursor_hours": projected,
            "projected_longest_motion_hours": projected,
            "target_walltime_hours": 2.0,
            "target_walltime_pass": target_pass,
            "maximum_walltime_hours": 8.0,
            "maximum_walltime_pass": maximum_pass,
            "log": log.name,
            "log_sha256": _sha(log),
        }
        report_path.write_text(json.dumps(report), encoding="utf-8")
        scaling_cases.append({"nodes": nodes, "report": report_path.name})
        by_nodes[str(nodes)] = {
            "complete_reynolds_sweep": True,
            "target_walltime_pass": target_pass,
            "maximum_walltime_pass": maximum_pass,
            "pass": maximum_pass,
            "reports": 1,
            "worst_projected_hours": projected,
        }
    preflight_case = campaign_rows[-1]
    campaign = {
        "schema": assembly.nalu_producer.CAMPAIGN_SCHEMA,
        "fingerprint": fingerprint,
        "request": request,
        "solver": solver,
        "mesh": mesh,
        "fluid_and_section": physical,
        "preflight": {
            "scaling_cases": scaling_cases,
            "motion_case_id": preflight_case["case_id"],
            "motion_case_index": preflight_case["index"],
        },
        "precursors": [
            {
                "precursor_id": f"precursor_re{int(reynolds):06d}",
                "reynolds_number": reynolds,
                "precursor_steps": 3016,
            }
            for reynolds in assembly.EXPECTED_REYNOLDS
        ],
        "cases": campaign_rows,
    }
    (root / "campaign_manifest.json").write_text(json.dumps(campaign), encoding="utf-8")
    mesh_audit_path = root / "mesh_audit.json"
    mesh_audit = {
        "schema": assembly.nalu_producer.MESH_AUDIT_SCHEMA,
        "status": "pass",
        "campaign_fingerprint": fingerprint,
        "mesh": {"sha256": mesh["sha256"]},
        "combined_force_surface": ["wing", "wing-pp"],
        "chord_axis": "global_Y",
        "incidence_orientation_deg": -90.0,
        "source_chord_m": 1.0,
        "source_span_m": 4.0,
    }
    mesh_audit_path.write_text(json.dumps(mesh_audit), encoding="utf-8")
    history_contract = {
        "time_origin": "precursor_restart",
        "first_row": "dt",
        "last_row": "(13*samples_per_cycle+1)*dt",
        "complete_cycle_intervals": 13,
        "raw_absolute_time_rule": "t0+j*dt for j=1..13*N+1",
        "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
        "columns": list(assembly.HISTORY_COLUMNS),
        "float_encoding": ".17g",
        "pairing_policy": "exact_array_equal_time_displacement_velocity_no_interpolation",
        "force_units": "N_total_over_4m_span",
    }
    campaign["history_contract"] = history_contract
    (root / "campaign_manifest.json").write_text(json.dumps(campaign), encoding="utf-8")
    results = {
        "campaign_manifest": "campaign_manifest.json",
        "campaign_fingerprint": fingerprint,
        "request": request,
        "solver": solver,
        "mesh": mesh,
        "fluid_and_section": physical,
        "force_contract": {
            "surface": ["wing", "wing-pp"],
            "crossflow_expression": "Fpy+Fvy",
            "sign": "positive_fluid_on_body_global_positive_Y",
            "units": "N_total_over_declared_4m_span",
        },
        "history_contract": history_contract,
        "mesh_audit": {
            "status": "pass",
            "path": mesh_audit_path.name,
            "sha256": _sha(mesh_audit_path),
        },
        "scaling_gate": {
            "status": "pass",
            "required_probe_count": 3,
            "complete_probe_count": 3,
            "recommended_nodes": 8,
            "recommended_ranks": 896,
            "selection_tier": "target_2h",
            "recommended_walltime": "02:00:00",
            "target_walltime_hours": 2.0,
            "maximum_walltime_hours": 8.0,
            "by_nodes": by_nodes,
            "runtime_identity": {
                "status": "verified",
                "observed_versions": ["v2.2.1"],
                "observed_git_commits": [assembly.EXPECTED_NALU_COMMIT],
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
                "runtime_identity": runtime_identity(
                    f"precursor_re{int(reynolds):06d}", 3016
                ),
            }
            for reynolds in assembly.EXPECTED_REYNOLDS
        ],
        "motion_preflight": {
            "status": "complete",
            "case_id": preflight_case["case_id"],
            "case_index": preflight_case["index"],
            "history_sha256": result_rows[-1]["canonical_history"]["sha256"],
        },
        "summary": {
            "complete_precursors": 3,
            "required_precursors": 3,
            "complete_motion_cases": 48,
            "failed_motion_cases": 0,
            "failed_precursors": 0,
            "required_motion_cases": 48,
        },
        "cases": result_rows,
    }
    monkeypatch.setattr(assembly.nalu_producer, "validate_campaign", lambda _root: None)
    return root, campaign, results


def _artifact_identity(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }


def _qblade_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "qblade_contract"
    root.mkdir()
    fingerprint = "3" * 64
    physical = _physical()
    campaign_rows, result_rows = _canonical_producer_rows("qblade")
    request = {
        "path": str(assembly.REQUEST_PATH),
        "sha256": _sha(assembly.REQUEST_PATH),
    }
    runtime = {
        "root": "/frozen/QBladeCE_2.0.9.7",
        "version": "2.0.9.7",
        "library": {
            "path": "/frozen/QBladeCE_2.0.9.7/libQBladeCE_2.0.9.7.so.1.0.0",
            "sha256": assembly.EXPECTED_QBLADE_LIBRARY_SHA256,
            "bytes": 1,
        },
        "runtime_dependencies": [],
        "runtime_dependency_fingerprint": assembly.EXPECTED_QBLADE_DEPENDENCY_FINGERPRINT,
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
        "mapping": (
            "QBlade +90 uses the opposite relative-wind sign convention from "
            "VorLap atan2(U.normal,U.chord)=-90; pitch and twist remain zero"
        ),
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
    history_contract = {
        "first_row": "dt",
        "last_row": "(13*samples_per_cycle+1)*dt",
        "wrapper_steps_rule": "13*samples_per_cycle+2",
        "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
        "columns": list(assembly.HISTORY_COLUMNS),
        "float_encoding": ".17g",
        "force_channel": assembly.qblade_producer.FORCE_CHANNEL,
        "force_frame": "hub_Y_equals_global_crossflow_Y",
        "force_sign": "positive_fluid_on_body_in_positive_global_Y",
        "force_units": "N_total_over_4m_span",
    }
    preflight_contract = {
        "case_index": assembly.qblade_producer.PREFLIGHT_CASE_INDEX,
        "case_id": assembly.qblade_producer.PREFLIGHT_CASE_ID,
        "selection_basis": (
            "maximum samples-per-cycle, maximum A/c, and maximum Reynolds "
            "number: longest-grid, largest-motion, high-Re representative"
        ),
        "required_before_matrix": True,
        "required_scheduler_schema": assembly.qblade_producer.PREFLIGHT_SCHEDULER_SCHEMA,
        "required_scheduler_stage": "preflight",
        "required_scheduler_status": "complete",
        "required_case_audit_schema": (
            "vorlap-naca0018-qblade-prescribed-motion-case-audit-v1"
        ),
    }
    campaign = {
        "schema": assembly.qblade_producer.CAMPAIGN_SCHEMA,
        "fingerprint": fingerprint,
        "request": request,
        "runtime": runtime,
        "fluid_and_section": physical,
        "geometry_and_frames": geometry,
        "native_aero": native,
        "external_forces": external,
        "history_contract": history_contract,
        "execution": {
            "threads_per_case": 1,
            "total_case_count": 48,
            "preflight_cpu_slots": 1,
            "matrix_cpu_slots": 47,
            "local_docker_max_cpu_slots": 8,
        },
        "preflight_gate": preflight_contract,
        "cases": campaign_rows,
    }
    (root / "campaign_manifest.json").write_text(json.dumps(campaign), encoding="utf-8")

    runtime_audit = {
        "qblade_library_sha256": assembly.EXPECTED_QBLADE_LIBRARY_SHA256,
        "qblade_runtime_dependency_fingerprint": assembly.EXPECTED_QBLADE_DEPENDENCY_FINGERPRINT,
        "threads": 1,
        "coupling_mode": "none",
        "external_callback_updates": 0,
    }
    for source, result in zip(campaign_rows, result_rows, strict=True):
        case_id = source["case_id"]
        case_root = root / "results" / case_id
        case_root.mkdir(parents=True)
        run_summary = case_root / "run_summary.json"
        raw_result = case_root / "raw.txt"
        run_summary.write_text("{}\n", encoding="utf-8")
        raw_result.write_text("0\n", encoding="utf-8")
        audit = {
            "schema": "vorlap-naca0018-qblade-prescribed-motion-case-audit-v1",
            "status": "complete",
            "case_id": case_id,
            "campaign_fingerprint": fingerprint,
            "run_summary": _artifact_identity(root, run_summary),
            "raw_result": _artifact_identity(root, raw_result),
            "canonical_history": result["canonical_history"],
            "runtime": runtime_audit,
            "channel_contract": {
                "raw_export_rows": source["wrapper_steps"],
                "discarded_wrapper_t0_rows": 1,
                "prescribed_motion": assembly.qblade_producer.PLATFORM_Y_CHANNEL,
                "rigid_blade_position": assembly.qblade_producer.BLADE_TIP_Y_CHANNEL,
                "crossflow_force": assembly.qblade_producer.FORCE_CHANNEL,
                "crossflow_force_units": "N_total_over_4m_span",
                "crossflow_force_sign": "positive_fluid_on_body_positive_global_Y",
                "qblade_aoa": assembly.qblade_producer.AOA_CHANNEL,
                "qblade_reynolds": assembly.qblade_producer.REYNOLDS_CHANNEL,
            },
            "geometry_frame_audit": {
                "qblade_internal_incidence_target_deg": 90.0,
                "vorlap_incidence_deg": -90.0,
                "median_abs_qblade_aoa_deg": 90.0,
                "max_abs_qblade_aoa_deviation_from_90_deg": 0.0,
                "median_qblade_panel_reynolds": source["reynolds_number"],
                "analytic_platform_max_abs_error_m": 0.0,
                "tip_follows_platform_max_abs_error_m": 0.0,
            },
            "force_statistics_n": {
                "minimum": -1.0,
                "maximum": 1.0,
                "mean": 0.0,
                "rms": 1.0,
            },
        }
        audit_path = case_root / "prescribed_motion_audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        result["case_audit"] = _artifact_identity(root, audit_path)

    preflight_report = root / "qblade_preflight_scheduler_report.json"
    preflight_environment = {
        "kind": "flight_slurm",
        "platform": "linux_x86_64",
        "slurm_job_id": "123456",
    }
    preflight_report.write_text(
        json.dumps(
            {
                "schema": assembly.qblade_producer.PREFLIGHT_SCHEDULER_SCHEMA,
                "stage": "preflight",
                "status": "complete",
                "campaign_fingerprint": fingerprint,
                "preflight_gate": {
                    "status": "complete",
                    "case_index": assembly.qblade_producer.PREFLIGHT_CASE_INDEX,
                    "case_id": assembly.qblade_producer.PREFLIGHT_CASE_ID,
                },
                "allocation": {
                    "workers": 1,
                    "threads_per_case": 1,
                    "allocated_physical_cpus": 1,
                },
                "execution_environment": preflight_environment,
                "cases": [
                    {
                        "index": assembly.qblade_producer.PREFLIGHT_CASE_INDEX,
                        "case_id": assembly.qblade_producer.PREFLIGHT_CASE_ID,
                        "status": "complete",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    preflight_result = result_rows[assembly.qblade_producer.PREFLIGHT_CASE_INDEX]
    preflight_evidence = {
        "status": "complete",
        "case_index": assembly.qblade_producer.PREFLIGHT_CASE_INDEX,
        "case_id": assembly.qblade_producer.PREFLIGHT_CASE_ID,
        "case_audit": preflight_result["case_audit"],
        "scheduler_report": _artifact_identity(root, preflight_report),
        "scheduler_schema": assembly.qblade_producer.PREFLIGHT_SCHEDULER_SCHEMA,
        "scheduler_stage": "preflight",
        "scheduler_status": "complete",
        "execution_environment": preflight_environment,
    }
    results = {
        "campaign_manifest": "campaign_manifest.json",
        "campaign_fingerprint": fingerprint,
        "request": request,
        "fluid_and_section": physical,
        "runtime": runtime,
        "geometry_and_frames": geometry,
        "native_aero": native,
        "external_forces": external,
        "history_contract": history_contract,
        "preflight_gate": {
            "campaign_contract": preflight_contract,
            **preflight_evidence,
        },
        "summary": {
            "required_cases": 48,
            "complete_cases": 48,
            "failed_cases": 0,
        },
        "cases": result_rows,
    }
    monkeypatch.setattr(
        assembly.qblade_producer, "validate_campaign", lambda _root: None
    )
    monkeypatch.setattr(
        assembly.qblade_producer,
        "validate_preflight_evidence",
        lambda _root, _manifest: preflight_evidence,
    )
    return root, campaign, results


def test_assemble_snapshots_exact_paired_histories(tmp_path, monkeypatch):
    nalu = _results(tmp_path / "nalu", qblade=False)
    qblade = _results(tmp_path / "qblade", qblade=True)
    _stub_producer_provenance(monkeypatch, nalu, qblade)
    spectrum = tmp_path / "NACA0018.h5"
    spectrum.write_bytes(b"stationary-spectrum-test")
    output = tmp_path / "bundle"

    manifest_path = assembly.assemble(
        nalu_results=nalu,
        qblade_results=qblade,
        static_spectrum=spectrum,
        output=output,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads((output / "bundle_report.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == assembly.IDENTIFICATION_SCHEMA
    assert manifest["status"] == "complete"
    assert len(manifest["cases"]) == 48
    assert report["case_count"] == 48
    provenance = manifest["producer_provenance"]
    assert set(provenance) == {"high_fidelity", "exact_runtime_base"}
    assert provenance["high_fidelity"]["kind"] == "nalu_wind_prescribed_motion"
    assert len(provenance["high_fidelity"]["scaling_reports"]) == 3
    assert provenance["exact_runtime_base"]["kind"] == "native_qblade_prescribed_motion"
    assert len(provenance["exact_runtime_base"]["case_audits"]) == 48
    assert "nalu_results_manifest" not in report
    assert "qblade_results_manifest" not in report
    for producer in provenance.values():
        for identity in (producer["results_manifest"], producer["campaign_manifest"]):
            assert not Path(identity["path"]).is_absolute()
            assert (output / identity["path"]).is_file()
    assert report["claim_boundary"] == {
        "ehrmann_free_response_used": False,
        "incremental_force": "high_fidelity_minus_exact_runtime_base",
        "static_fft_phase_used_for_work": False,
    }
    first = manifest["cases"][0]
    assert (
        first["high_fidelity_history"]["sha256"]
        != first["exact_runtime_base_history"]["sha256"]
    )
    assert (output / first["high_fidelity_history"]["path"]).is_file()
    assert (output / first["exact_runtime_base_history"]["path"]).is_file()


def test_assemble_rejects_motion_grid_mismatch_and_removes_partial_bundle(
    tmp_path, monkeypatch
):
    nalu = _results(tmp_path / "nalu", qblade=False)
    qblade = _results(tmp_path / "qblade", qblade=True)
    _stub_producer_provenance(monkeypatch, nalu, qblade)
    payload = json.loads(qblade.read_text(encoding="utf-8"))
    row = payload["cases"][0]
    history_path = qblade.parent / row["canonical_history"]["path"]
    text = history_path.read_text(encoding="utf-8").replace("0.01,", "0.011,", 1)
    history_path.write_text(text, encoding="utf-8")
    row["canonical_history"]["sha256"] = _sha(history_path)
    qblade.write_text(json.dumps(payload), encoding="utf-8")
    spectrum = tmp_path / "NACA0018.h5"
    spectrum.write_bytes(b"spectrum")
    output = tmp_path / "bundle"
    with pytest.raises(assembly.AssemblyError, match="time/displacement/velocity"):
        assembly.assemble(
            nalu_results=nalu,
            qblade_results=qblade,
            static_spectrum=spectrum,
            output=output,
        )
    assert not output.exists()


def test_assemble_rejects_incomplete_result_manifest(tmp_path):
    nalu = _results(tmp_path / "nalu", qblade=False)
    qblade = _results(tmp_path / "qblade", qblade=True)
    payload = json.loads(nalu.read_text(encoding="utf-8"))
    payload["status"] = "incomplete"
    nalu.write_text(json.dumps(payload), encoding="utf-8")
    spectrum = tmp_path / "NACA0018.h5"
    spectrum.write_bytes(b"spectrum")
    with pytest.raises(assembly.AssemblyError, match="not a complete"):
        assembly.assemble(
            nalu_results=nalu,
            qblade_results=qblade,
            static_spectrum=spectrum,
            output=tmp_path / "bundle",
        )


def test_nalu_producer_contract_accepts_exact_evidence_and_rejects_solver_mutation(
    tmp_path, monkeypatch
):
    root, _campaign, results = _nalu_contract(tmp_path, monkeypatch)
    cases = assembly._case_index(results, "Nalu test results")
    provenance = assembly._validate_nalu_producer(root, results, cases)
    assert provenance["campaign"] == root / "campaign_manifest.json"
    assert [nodes for nodes, _path in provenance["scaling_reports"]] == [2, 4, 8]

    runtime = results["cases"][0]["runtime_identity"]
    runtime_log = root / runtime["log"]["path"]
    runtime_log.write_text(
        "Nalu-Wind Version: v2.2.1\n"
        "Nalu-Wind GIT Commit SHA: wrong\n"
        f"Simulation Shall Complete: time/timestep: 1/{runtime['terminal_step']}\n",
        encoding="utf-8",
    )
    runtime["log"]["sha256"] = _sha(runtime_log)
    with pytest.raises(assembly.AssemblyError, match="does not prove the exact"):
        assembly._validate_nalu_producer(root, results, cases)
    runtime_log.write_text(
        "Nalu-Wind Version: v2.2.1\n"
        f"Nalu-Wind GIT Commit SHA: {assembly.EXPECTED_NALU_COMMIT}\n"
        f"Simulation Shall Complete: time/timestep: 1/{runtime['terminal_step']}\n",
        encoding="utf-8",
    )
    runtime["log"]["sha256"] = _sha(runtime_log)

    results["solver"] = {**results["solver"], "transition_model": "gamma_retheta"}
    with pytest.raises(assembly.AssemblyError, match="solver/model identity"):
        assembly._validate_nalu_producer(root, results, cases)


def test_qblade_producer_contract_accepts_exact_evidence_and_rejects_runtime_mutation(
    tmp_path, monkeypatch
):
    root, campaign, results = _qblade_contract(tmp_path, monkeypatch)
    cases = assembly._case_index(results, "QBlade test results")
    provenance = assembly._validate_qblade_producer(root, results, cases)
    assert provenance["campaign"] == root / "campaign_manifest.json"
    assert provenance["preflight_scheduler_report"] == (
        root / "qblade_preflight_scheduler_report.json"
    )
    assert len(provenance["case_audits"]) == 48

    mutated_runtime = {
        **results["runtime"],
        "library": {**results["runtime"]["library"], "sha256": "0" * 64},
    }
    results["runtime"] = mutated_runtime
    campaign["runtime"] = mutated_runtime
    (root / "campaign_manifest.json").write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(assembly.AssemblyError, match="runtime/library identity"):
        assembly._validate_qblade_producer(root, results, cases)


def test_rejects_noncanonical_case_identity_and_self_consistent_wrong_physics(
    tmp_path, monkeypatch
):
    row = {
        "case_id": "arbitrary",
        "incidence_deg": -90.0,
        "reynolds_number": 64_000.0,
        "reduced_frequency_k": 0.45,
        "amplitude_ratio": 0.005,
        "direction_deg_mod_180": 0.0,
    }
    with pytest.raises(assembly.AssemblyError, match="case_id is not canonical"):
        assembly._validate_vertex_and_grid(row, _physical(), "mutated")

    nalu = _results(tmp_path / "nalu_wrong_physics", qblade=False)
    qblade = _results(tmp_path / "qblade_wrong_physics", qblade=True)
    _stub_producer_provenance(monkeypatch, nalu, qblade)
    for path in (nalu, qblade):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["fluid_and_section"]["section_span_m"] = 1.0
        path.write_text(json.dumps(payload), encoding="utf-8")
    spectrum = tmp_path / "spectrum.h5"
    spectrum.write_bytes(b"spectrum")
    with pytest.raises(assembly.AssemblyError, match="tracked request"):
        assembly.assemble(
            nalu_results=nalu,
            qblade_results=qblade,
            static_spectrum=spectrum,
            output=tmp_path / "wrong_physics_bundle",
        )
