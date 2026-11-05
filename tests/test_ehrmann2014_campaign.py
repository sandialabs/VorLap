import json
from pathlib import Path

import numpy as np
import pytest

from examples.ehrmann2014_naca0018 import campaign
from examples.ehrmann2014_naca0018 import structural_verification
from examples.ehrmann2014_naca0018 import work_map_smoke
from vorlap.qblade_sil import SilVorLapCoupler


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_work_map_smoke_binds_the_runtime_stationary_spectrum():
    payload = work_map_smoke.synthetic_map_payload()
    stationary = [
        source
        for source in payload["sources"]
        if source["role"] == "stationary_force_spectrum"
    ]
    assert stationary == [
        {
            "role": "stationary_force_spectrum",
            "label": "tracked NACA0018 stationary spectrum",
            "sha256": campaign.TRACKED_STATIONARY_SPECTRUM_SHA256,
        }
    ]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("ehrmann2014") / "campaign"
    campaign.generate(root, runtime_vorlap_root=REPO_ROOT)
    return root


@pytest.fixture(scope="module")
def engineering_generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("ehrmann2014-engineering") / "campaign"
    campaign.generate(
        root,
        runtime_vorlap_root=REPO_ROOT,
        engineering_calibration=(
            REPO_ROOT
            / "examples/ehrmann2014_naca0018/engineering_lockin_calibration.json"
        ),
    )
    return root


def test_campaign_brackets_reported_mass_damping_and_lockin_reynolds(generated):
    state = campaign.validate(generated)
    manifest = state["manifest"]
    assert manifest["case_count"] == 40
    assert manifest["matrix"]["mass_dampings"] == [0.96, 1.44]
    assert manifest["matrix"]["reduced_velocities"] == [
        3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.8, 6.0, 6.3
    ]
    assert manifest["matrix"]["paired_timestep_reduced_velocity"] == 5.8
    assert manifest["structural_qualification"] == {
        "path": "examples/ehrmann2014_naca0018/structural_verification_report.json",
        "sha256": campaign.STRUCTURAL_VERIFICATION_REPORT_SHA256,
        "status": "passed",
        "verified_result_channel": campaign.RESULT_CHANNEL,
    }
    near_lockin = [
        row for row in manifest["cases"]
        if row["reduced_velocity"] == 5.8 and row["timestep_s"] == 0.001
    ]
    assert len(near_lockin) == 4
    assert all(abs(row["reynolds_number"] - 125000.0) / 125000.0 < 0.01 for row in near_lockin)
    fractions = np.linspace(0.0, 1.0, campaign.N_STATIONS)
    mode = campaign._first_mode(fractions)
    span = fractions * campaign.SPAN_M
    distributed_mass = campaign._surrogate_distributed_mass_kgpm(fractions)
    generalized_mass = distributed_mass * np.trapezoid(mode**2, span)
    implied_effective_mass_per_span = generalized_mass / campaign.SPAN_M
    implied_mass_ratio = (
        4.0
        * implied_effective_mass_per_span
        / (
            np.pi
            * campaign.AIR_DENSITY_KGPM3
            * campaign.CHORD_M**2
        )
    )
    assert implied_mass_ratio == pytest.approx(campaign.MASS_RATIO, rel=0.0, abs=1e-12)
    beam = next(generated.glob("cases/*_raw/model/Structure/ehrmann_beam.str"))
    numeric = [
        line.split() for line in beam.read_text().splitlines()
        if len(line.split()) == 18 and line.split()[0][0].isdigit()
    ]
    assert numeric
    for row in numeric:
        eix = float(row[2])
        eiy = float(row[3])
        assert eix / eiy == pytest.approx(campaign.ORTHOGONAL_BENDING_STIFFNESS_RATIO)


def test_engineering_campaign_is_exact_52_case_raw_latest_comparison(
    engineering_generated,
):
    manifest = campaign.validate(engineering_generated)["manifest"]
    assert manifest["schema"] == campaign.ENGINEERING_LOCKIN_SCHEMA
    assert manifest["comparison_series"] == list(
        campaign.ENGINEERING_LOCKIN_SCENARIOS
    )
    assert manifest["matrix"]["paired_timestep_reduced_velocity"] == 4.7
    assert manifest["matrix"]["reduced_velocities"] == list(
        campaign.ENGINEERING_REDUCED_VELOCITIES
    )
    assert manifest["case_count"] == 52
    assert manifest["engineering_lockin_identity"]["nalu_required"] is False
    assert (
        manifest["engineering_lockin_identity"][
            "maximum_holdout_absolute_relative_error"
        ]
        < 0.15
    )
    pairs = {}
    for row in manifest["cases"]:
        pairs.setdefault(row["native_physical_id"], {})[row["scenario"]] = row
    assert len(pairs) == 26
    for rows in pairs.values():
        raw = rows["raw_qblade"]
        hybrid = rows["latest_engineering_lockin_bridge"]
        assert raw["native_case_identity"] == hybrid["native_case_identity"]
        raw_config = json.loads(
            (engineering_generated / raw["runtime_config"]).read_text()
        )
        hybrid_config = json.loads(
            (engineering_generated / hybrid["runtime_config"]).read_text()
        )
        assert raw_config["response_model"]["aerodynamic_damping_gain_h1"] == 0.0
        assert hybrid_config["response_model"]["aerodynamic_damping_gain_h1"] == 0.0
        assert raw_config["response_model"]["startup_seed"]["enabled"] is False
        assert hybrid_config["response_model"]["startup_seed"]["enabled"] is True
        assert raw_config["single_mode_engineering_lockin"]["modal_power"]["enabled"] is False
        assert hybrid_config["single_mode_engineering_lockin"]["modal_power"]["enabled"] is True
        assert raw_config["single_mode_engineering_lockin"]["residual_source_scale"] == 0.0


def test_raw_and_hybrid_physical_inputs_are_identical_and_force_contract_is_exact(generated):
    manifest = campaign.validate(generated)["manifest"]
    pairs = {}
    for row in manifest["cases"]:
        pairs.setdefault(row["native_physical_id"], {})[row["scenario"]] = row
    assert len(pairs) == 20
    for rows in pairs.values():
        raw = rows["raw_qblade"]
        hybrid = rows["latest_spectral_lockin_bridge"]
        assert raw["native_case_identity"] == hybrid["native_case_identity"]
        raw_config = json.loads((generated / raw["runtime_config"]).read_text())
        hybrid_config = json.loads((generated / hybrid["runtime_config"]).read_text())
        assert raw_config["force_scale"] == hybrid_config["force_scale"] == 0.0
        assert raw_config["include_mean_load"] is hybrid_config["include_mean_load"] is False
        assert raw_config["response_model"]["aerodynamic_damping_gain_h1"] == 0.0
        assert hybrid_config["response_model"]["aerodynamic_damping_gain_h1"] == pytest.approx(
            3.920125762951649
        )
        assert raw_config["response_model"]["startup_seed"]["enabled"] is False
        assert hybrid_config["response_model"]["startup_seed"]["enabled"] is True
        assert raw_config["single_mode_spectral_lockin"]["spectral_lockin"]["residual_source_scale"] == 0.0
        assert hybrid_config["single_mode_spectral_lockin"]["spectral_lockin"]["residual_source_scale"] == 0.0


def test_generic_adapter_raw_is_zero_and_hybrid_uses_spectral_activation(generated):
    manifest = campaign.validate(generated)["manifest"]
    selected = [manifest["cases"][0], manifest["cases"][1]]
    maxima = []
    for row in selected:
        coupler = SilVorLapCoupler(generated / row["sim"], history_mode="off")
        adapter = coupler.hvawt_response_adapter
        assert type(adapter).__name__ == "QBladeSingleModeSpectralSilAdapter"
        layout = coupler.runtime.layout
        coupler.swap[layout["time"].offset] = 0.0
        coupler.swap[layout["azimuth_deg"].offset] = 0.0
        adapter.step(coupler.swap, response_time_s=0.0)
        velocity = layout["velocity"]
        coupler.swap[velocity.offset : velocity.offset + velocity.size] = np.tile(
            [0.0, 0.05, 0.0], len(adapter.node_ids)
        )
        forces = adapter.step(coupler.swap, response_time_s=0.05)
        assert adapter.last_spectral_result.partition.maximum_energy_closure_error < 1e-12
        summary = adapter.diagnostics_summary()
        assert summary["samples"] == 1
        assert summary["spectral"]["residual_source_scale"] == 0.0
        maxima.append(float(np.max(np.abs(forces))))
    assert maxima[0] == 0.0
    assert maxima[1] > 0.0


def test_raw_reuse_refuses_mismatched_identity(generated, tmp_path):
    manifest = campaign.validate(generated)["manifest"]
    raw = next(row for row in manifest["cases"] if row["scenario"] == "raw_qblade")
    candidate = tmp_path / raw["case_id"]
    candidate.mkdir(parents=True)
    (candidate / "native_case_identity.json").write_text("{}\n")
    summary_name = f"{Path(str(raw['sim'])).stem}_run_summary.json"
    (candidate / summary_name).write_text("{}\n")
    with pytest.raises(campaign.CampaignError, match="identity mismatch"):
        campaign._reuse_raw_result(
            campaign_root=generated,
            case=raw,
            reuse_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonzero_external_force", "zero-supplemental-force"),
        ("native_callback_activity", "native callback activity"),
        ("qblade_library_mismatch", "runtime identity differs"),
        ("qblade_dependencies_mismatch", "runtime identity differs"),
    ],
)
def test_raw_reuse_requires_zero_force_and_exact_qblade_runtime(
    generated, tmp_path, mutation, message
):
    manifest = campaign.validate(generated)["manifest"]
    raw = next(row for row in manifest["cases"] if row["scenario"] == "raw_qblade")
    candidate = tmp_path / raw["case_id"]
    candidate.mkdir(parents=True)
    (candidate / "native_case_identity.json").write_text(
        json.dumps(raw["native_case_identity"]) + "\n", encoding="utf-8"
    )
    summary_name = f"{Path(str(raw['sim'])).stem}_run_summary.json"
    summary = {
        "failed": False,
        "stopped": False,
        "completed_steps": raw["num_steps"],
        "requested_steps": raw["num_steps"],
        "coupling_mode": "sil-vorlap",
        "sil_vorlap": True,
        "sil_vorlap_max_abs_force": 0.0,
        "native_callback_evidence": {
            "update_log_entry_count": 0,
            "update_init_count": 0,
            "update_cache_observation_count": 0,
            "update_count_observed_max": 0,
        },
        "reproducibility": {
            "qblade": {
                "library": {"sha256": campaign.FROZEN_QBLADE_LIBRARY_SHA256}
            },
            "qblade_runtime_dependency_fingerprint": (
                campaign.FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
            ),
        },
    }
    if mutation == "nonzero_external_force":
        summary["sil_vorlap_max_abs_force"] = 1.0e-9
    elif mutation == "native_callback_activity":
        summary["native_callback_evidence"]["update_cache_observation_count"] = 1
        summary["native_callback_evidence"]["update_count_observed_max"] = 7
    elif mutation == "qblade_library_mismatch":
        summary["reproducibility"]["qblade"]["library"]["sha256"] = "0" * 64
    else:
        summary["reproducibility"][
            "qblade_runtime_dependency_fingerprint"
        ] = "0" * 64
    (candidate / summary_name).write_text(
        json.dumps(summary) + "\n", encoding="utf-8"
    )
    with pytest.raises(campaign.CampaignError, match=message):
        campaign._reuse_raw_result(
            campaign_root=generated,
            case=raw,
            reuse_root=tmp_path,
        )


def test_slurm_is_four_threads_two_way_concurrency_and_exact_array():
    text = (REPO_ROOT / "run_headless/submit_ehrmann2014_validation.sj").read_text()
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --array=0-39%2" in text
    assert "--threads \"${THREADS}\"" in text


def test_runner_command_uses_native_run_qblade_cli(generated, tmp_path):
    command = campaign.command_for_case(
        generated,
        case_index=1,
        qblade_root=tmp_path,
        python_executable="python3",
        threads=4,
        resume=True,
        qblade_steps=2,
    )
    assert command[2] == "--sim"
    assert "--result-name" in command
    assert "--output-name" not in command
    assert "--steps" in command
    assert "--checkpoint-every" in command
    assert "--resume-from-checkpoint" in command
    assert "--execute" not in command


def test_analyze_is_fail_closed_until_every_result_exists(generated):
    path = campaign.analyze(generated)
    report = json.loads(path.read_text())
    assert report["status"] == "incomplete_missing_results"
    assert report["expected_cases"] == 40
    assert report["completed_cases"] == 0
    assert len(report["missing_case_ids"]) == 40
    assert report["experimental_overlay_status"] == (
        "available_in_examples/ehrmann2014_naca0018/reference_data_and_"
        "validation_evidence"
    )
    csv_path = generated / "analysis/ehrmann2014_case_metrics.csv"
    assert csv_path.is_file()


def test_structural_verification_prepares_aero_off_axis_and_decay_cases(generated, tmp_path):
    root = tmp_path / "structural_verification"
    manifest_path = structural_verification.prepare(generated, root)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["native_aerodynamics"] is False
    assert manifest["orthogonal_bending_stiffness_ratio_EIx_over_EIy"] == 100.0
    assert len(manifest["cases"]) == 4
    for row in manifest["cases"]:
        sim = (root / row["sim"]).read_text()
        assert campaign.read_qblade_keyword(root / row["sim"], "INCLUDEAERO") == "0"
        assert campaign.read_qblade_keyword(root / row["sim"], "MEANINF") == "0"
        if row["kind"] == "free_decay":
            assert row["loading_file"] in {
                str(path.relative_to(root)) for path in root.glob("cases/*/global_y_burst_loading.txt")
            }
    probe = next(row for row in manifest["cases"] if row["kind"] == "axis_probe")
    command = structural_verification._command(root, probe, tmp_path, 4)
    assert "sil-constant-force" in command
    assert any("BLD_1_1.000000:Y:" in value for value in command)


def test_campaign_refuses_mutated_structural_qualification(generated, tmp_path, monkeypatch):
    mutated = tmp_path / "structural_verification_report.json"
    payload = json.loads(campaign.STRUCTURAL_VERIFICATION_REPORT.read_text())
    payload["axis_channel_probe"]["verified_body_tip_channel"] = (
        "X_b Tip Trl.Def. BLD_1 [m]"
    )
    mutated.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setattr(campaign, "STRUCTURAL_VERIFICATION_REPORT", mutated)
    with pytest.raises(campaign.CampaignError, match="identity differs"):
        campaign.validate(generated)
