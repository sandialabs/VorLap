import json
from pathlib import Path

import pytest

from examples.grinderslev2022_iea10 import complete_hybrid_campaign as campaign


def _write_result(path: Path, tip: list[float]) -> None:
    rows = [f"{time:.1f}\t{value:.9g}" for time, value in zip(range(len(tip)), tip)]
    path.write_text(
        "QBlade result\nsynthetic test\n"
        "Time~[s]\tY_b~Tip~Trl.Def.~BLD_1~[m]\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_analyze_binds_complete_paired_results_and_increment(tmp_path: Path):
    root = tmp_path / "campaign"
    root.mkdir()
    cases = []
    for speed in campaign.SPEEDS_MPS:
        for role in ("native", "complete"):
            case_id = f"U{int(speed):02d}_{role}"
            result_dir = root / "results" / case_id
            result_dir.mkdir(parents=True)
            tip = [0.0, 0.0, -1.0, 0.0, 1.0]
            if role == "complete":
                tip = [0.0, 0.0, -1.2, 0.0, 1.2]
            _write_result(result_dir / f"{case_id}.txt", tip)
            summary = {
                "failed": False,
                "stopped": False,
                "completed_steps": 5,
                "requested_steps": 5,
                "exported_results": True,
                "sil_vorlap_updates": 5,
                "sil_vorlap_max_abs_force": 0.0 if role == "native" else 2.0,
                "sil_vorlap_response_diagnostics_summary": {
                    "model_type": "single_mode_airfoil_geometry_blended_complete_transfer_v3",
                    "residual_source_scale": 0.0 if role == "native" else 1.0,
                    "last": {"spectral_stretch_gain": 12.0},
                    "spectral": {"maximum_energy_closure_error": 1.0e-16},
                },
                "native_callback_evidence": {
                    "update_init_count": 0,
                    "update_log_entry_count": 0,
                    "update_cache_observation_count": 0,
                    "update_count_observed_max": 0,
                },
                "reproducibility": {
                    "qblade": {"library": {"sha256": "a" * 64}},
                    "vorlap_python_runtime_fingerprint": "b" * 64,
                },
            }
            # QBlade names the summary from the simulation model, not the
            # campaign case identifier.
            (
                result_dir
                / "grinderslev2022_p100_i50_iea10_single_blade_run_summary.json"
            ).write_text(json.dumps(summary), encoding="utf-8")
            (result_dir / f"{case_id}_sil_vorlap_history.csv").write_text(
                "time_s\n0\n", encoding="utf-8"
            )
            cases.append(
                {
                    "case_id": case_id,
                    "speed_mps": speed,
                    "role": role,
                    "result_dir": f"results/{case_id}",
                    "target_amplitude_m": 1.0,
                    "requested_steps": 5,
                }
            )
    manifest = {"fingerprint": "c" * 64, "cases": cases}
    (root / "campaign_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    u14_case = root / "cases/U14_complete"
    (u14_case / campaign.base.CASE_NAME / "Control").mkdir(parents=True)
    (u14_case / campaign.base.CASE_NAME / "Aero/Airfoils").mkdir(parents=True)
    (u14_case / campaign.base.CASE_NAME / "Aero/Polars").mkdir(parents=True)
    (u14_case / "iea10_blade_geometry.csv").write_text(
        "span_fraction,span_m,chord_m\n0,0,1\n", encoding="utf-8"
    )
    (
        u14_case / campaign.base.CASE_NAME / "Control/vorlap_qblade_geometry_audit.csv"
    ).write_text("requested_node_id,vorlap_x_m\nBLD_1_0.0,0\n", encoding="utf-8")
    for index in range(30):
        (
            u14_case
            / campaign.base.CASE_NAME
            / f"Aero/Airfoils/IEA10_AF{index:02d}.afl"
        ).write_text("test foil\n1 0\n0 0\n", encoding="utf-8")
        (
            u14_case
            / campaign.base.CASE_NAME
            / f"Aero/Polars/IEA10_AF{index:02d}.plr"
        ).write_text("test polar\n", encoding="utf-8")
    (
        u14_case
        / campaign.base.CASE_NAME
        / "Aero/grinderslev2022_p100_i50_iea10_single_blade.bld"
    ).write_text("test blade\n", encoding="utf-8")
    history_fields = (
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
    (root / "results/U14_complete/U14_complete_sil_vorlap_history.csv").write_text(
        ",".join(history_fields) + "\n" + ",".join("0" for _ in history_fields) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "evidence" / "analysis.json"
    assert campaign.analyze(root, output) == output.resolve()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "complete_integration_screen"
    assert report["completed_cases"] == 8
    assert len(report["pairs"]) == 4
    assert report["pairs"][0]["native_amplitude_m"] == 1.0
    assert report["pairs"][0]["complete_amplitude_m"] == 1.2
    assert report["pairs"][0]["paired_increment_amplitude_m"] == pytest.approx(0.2)
    assert report["cases"][0]["result_sha256"]
    assert report["cases"][1]["residual_source_scale"] == 1.0
    assert report["cases"][1]["spectral_stretch_gain"] == 12.0
    assert not any(report["cases"][1]["native_callback_counts"].values())

    promoted = tmp_path / "promoted"
    evidence_manifest = campaign.promote(root, promoted)
    evidence = json.loads(evidence_manifest.read_text(encoding="utf-8"))
    assert evidence["status"] == "complete_integration_screen"
    assert evidence["selected_visual_case"] == "U14_complete"
    primary_files = {
        "analysis.json",
        "campaign_manifest.json",
        "iea10_blade_geometry.csv",
        "qblade_geometry_audit.csv",
        "u14_complete_bridge_history.csv",
        "u14_paired_tip_history.csv",
    }
    assert primary_files <= set(evidence["files"])
    assert len(
        [name for name in evidence["files"] if name.startswith("qblade_aero/")]
    ) == 61


def test_low_re_point_is_excluded_and_root_uses_declared_proxy():
    assert campaign.SPEEDS_MPS == (10.0, 14.0, 18.0, 22.0)
    assert campaign._profile_for_thickness(0.95) == 500
    assert campaign._profile_for_thickness(0.215) == 211
