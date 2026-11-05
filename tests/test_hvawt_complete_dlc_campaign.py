from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from vorlap import hvawt_complete_dlc_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]


def _approved_manifest() -> dict:
    return json.loads(
        (ROOT / campaign.APPROVED_MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )


def _fingerprinted(payload: dict) -> dict:
    result = dict(payload)
    result["fingerprint"] = campaign.canonical_fingerprint(result)
    return result


def test_case_plan_is_exact_34_by_three_matched_composition(tmp_path):
    rows = campaign.build_case_plan(tmp_path, _approved_manifest())

    assert len(rows) == 102
    assert [row["index"] for row in rows] == list(range(102))
    assert tuple(row["scenario"] for row in rows[:3]) == campaign.SCENARIOS
    assert len({row["row_id"] for row in rows}) == 34
    for row_index in range(34):
        trio = rows[3 * row_index : 3 * row_index + 3]
        assert len({row["row_id"] for row in trio}) == 1
        assert tuple(row["scenario"] for row in trio) == campaign.SCENARIOS
        assert {row["timestep_s"] for row in trio} == {0.0075}
        assert {row["duration_s"] for row in trio} == {660.0}
        assert {row["total_steps"] for row in trio} == {88_000}

    assert campaign.SCENARIO_FORCE_CONTRACT["raw_qblade"] == {
        "coupling_mode": "none",
        "legacy_vorlap_force_scale": 0.0,
        "residual_source_scale": 0.0,
        "response_force_scale": 0.0,
    }
    assert campaign.SCENARIO_FORCE_CONTRACT["one_way_spectral_vorlap"] == {
        "coupling_mode": "sil-vorlap",
        "legacy_vorlap_force_scale": 1.0,
        "residual_source_scale": 0.0,
        "response_force_scale": 0.0,
    }
    assert campaign.SCENARIO_FORCE_CONTRACT["complete_hybrid_v8"] == {
        "coupling_mode": "sil-vorlap",
        "legacy_vorlap_force_scale": 0.0,
        "residual_source_scale": 1.0,
        "response_force_scale": 1.0,
    }


def test_common_physics_fingerprint_excludes_only_scenario_coupling_plumbing(
    tmp_path,
):
    source_case = ROOT / "examples/ReferenceHVAWTQBladeModel"
    model_source = ROOT / campaign.RESPONSE_MODEL_RELATIVE
    response_model = json.loads(model_source.read_text(encoding="utf-8"))
    result_filter = ROOT / campaign.RESULT_FILTER_SOURCE_RELATIVE
    contracts = {}
    case_dirs = {}
    wake_audits = {}
    source_row = {"row_id": "matched_test_row"}
    wind_record = {
        "record_sha256": "record",
        "sha256": "wind",
        "bytes": 1,
    }
    normalization = {"status": "identical_test_normalization"}

    for scenario in campaign.SCENARIOS:
        case_dir = (tmp_path / scenario).resolve()
        shutil.copytree(source_case, case_dir)
        sim_path = case_dir / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
        wake_audit = campaign._apply_wake_profile(sim_path)
        case = {"scenario": scenario}
        campaign._prepare_scenario(
            runtime_repository=ROOT,
            case=case,
            case_dir=case_dir,
            sim_path=sim_path,
            response_model=response_model,
            model_source=model_source,
            result_filter=result_filter,
        )
        contracts[scenario] = campaign._common_physics_contract(
            case_dir=case_dir,
            sim_path=sim_path,
            source_row=source_row,
            wind_record=wind_record,
            rayleigh_normalization=normalization,
            wake_audit=wake_audit,
        )
        case_dirs[scenario] = case_dir
        wake_audits[scenario] = wake_audit

    assert len({row["fingerprint"] for row in contracts.values()}) == 1
    assert len(
        {
            next(
                asset["sha256"]
                for asset in row["physical_assets"]
                if asset["path"].endswith("10m_HVAWT_reference_main.str")
            )
            for row in contracts.values()
        }
    ) == 1

    changed_case_dir = case_dirs[campaign.SCENARIOS[2]]
    aero_file = next(
        path
        for path in (
            changed_case_dir / "10m_HVAWT_reference_turbine/Aero"
        ).rglob("*")
        if path.is_file()
    )
    aero_file.write_bytes(aero_file.read_bytes() + b"\n")
    changed_contract = campaign._common_physics_contract(
        case_dir=changed_case_dir,
        sim_path=changed_case_dir
        / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim",
        source_row=source_row,
        wind_record=wind_record,
        rayleigh_normalization=normalization,
        wake_audit=wake_audits[campaign.SCENARIOS[2]],
    )
    assert changed_contract["fingerprint"] != contracts[campaign.SCENARIOS[2]][
        "fingerprint"
    ]


def _checkpoint_manifest(tmp_path: Path) -> tuple[dict, dict]:
    rows = campaign.build_case_plan(tmp_path, _approved_manifest())
    for row in rows:
        row["common_physics_fingerprint"] = f"row-{row['row_index']:02d}"
    manifest = {"fingerprint": "f" * 64, "cases": rows}
    policy = json.loads(
        (ROOT / campaign.TURBULENT_CHECKPOINT_POLICY_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    return manifest, policy


def test_turbulent_checkpoint_is_exact_twenty_reusable_raw_complete_cases(tmp_path):
    manifest, policy = _checkpoint_manifest(tmp_path)

    selected = campaign.validate_turbulent_checkpoint_selection(manifest, policy)

    assert tuple(case["index"] for case in selected) == campaign.CHECKPOINT_CASE_INDICES
    assert len(selected) == 20
    assert {case["scenario"] for case in selected} == set(
        campaign.CHECKPOINT_SCENARIOS
    )
    assert "one_way_spectral_vorlap" not in {case["scenario"] for case in selected}
    assert {case["seed"] for case in selected} == {campaign.CHECKPOINT_SEED}
    assert tuple(selected[index]["wind_speed_mps"] for index in range(0, 20, 2)) == (
        campaign.CHECKPOINT_WIND_SPEEDS_MPS
    )
    for index in range(0, 20, 2):
        assert selected[index]["row_id"] == selected[index + 1]["row_id"]
        assert (
            selected[index]["common_physics_fingerprint"]
            == selected[index + 1]["common_physics_fingerprint"]
        )


@pytest.mark.paper
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir(),
    reason="Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
)
def test_checkpoint_authorization_binds_reviewed_evidence_and_detects_tamper(
    tmp_path, monkeypatch
):
    root = tmp_path / "dlc"
    root.mkdir()
    manifest, _policy = _checkpoint_manifest(root)
    monkeypatch.setattr(
        campaign,
        "validate_campaign",
        lambda *_args, **_kwargs: {
            "campaign_root": root,
            "manifest": manifest,
        },
    )
    evidence = ROOT / campaign.UNIFORM_EVIDENCE_RECEIPT_RELATIVE

    receipt = campaign.bind_checkpoint_authorization(
        root, evidence_path=evidence
    )

    assert tuple(receipt["authorized_case_indices"]) == campaign.CHECKPOINT_CASE_INDICES
    assert tuple(receipt["scenarios"]) == campaign.CHECKPOINT_SCENARIOS
    assert receipt["u50_smoke_required"] is False
    assert campaign._load_checkpoint_authorization(root, manifest) == receipt

    snapshot = root / receipt["snapshots"]["checkpoint_policy"]["path"]
    snapshot.write_text("{}", encoding="utf-8")
    with pytest.raises(
        (campaign.HVAWTCompleteDLCError, campaign.HVAWTResponsePilotError),
        match="identity",
    ):
        campaign._load_checkpoint_authorization(root, manifest)


def test_checkpoint_execution_scope_rejects_unselected_and_does_not_require_u50(
    tmp_path, monkeypatch
):
    root = tmp_path / "dlc"
    root.mkdir()
    manifest, _policy = _checkpoint_manifest(root)
    state = {"campaign_root": root, "manifest": manifest}
    monkeypatch.setattr(campaign, "validate_campaign", lambda *_a, **_k: state)
    monkeypatch.setattr(campaign, "_load_runtime", lambda *_a, **_k: {"runtime": 1})
    monkeypatch.setattr(
        campaign, "build_runtime_identity", lambda *_a, **_k: {"runtime": 1}
    )
    monkeypatch.setattr(
        campaign, "_load_checkpoint_authorization", lambda *_a, **_k: {"ok": True}
    )
    monkeypatch.setattr(
        campaign,
        "_load_u50_smoke_receipt",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("U50 not allowed")),
    )
    monkeypatch.setattr(
        campaign,
        "_classify_case_state",
        lambda *_a, **_k: {"status": "complete", "case_id": "reused"},
    )

    reused = campaign.execute_case(
        root,
        case_index=campaign.CHECKPOINT_CASE_INDICES[0],
        qblade_root=tmp_path,
        python_executable="python",
        threads=1,
        authorization_scope="turbulent_20case_checkpoint",
    )
    assert reused["status"] == "complete"

    with pytest.raises(campaign.HVAWTCompleteDLCError, match="outside"):
        campaign.execute_case(
            root,
            case_index=1,
            qblade_root=tmp_path,
            python_executable="python",
            threads=1,
            authorization_scope="turbulent_20case_checkpoint",
        )


def test_flight_scripts_reuse_simple_successful_direct_process_policy():
    prep = ROOT / "run_headless/prepare_hvawt_complete_dlc_native.sh"
    submit = ROOT / "run_headless/submit_hvawt_complete_dlc_flight.sj"
    analysis = ROOT / "run_headless/submit_hvawt_complete_dlc_analysis.sj"
    smoke = ROOT / "run_headless/submit_hvawt_complete_dlc_u50_smoke.sj"
    checkpoint = ROOT / "run_headless/submit_hvawt_turbulent_20case_checkpoint.sj"
    checkpoint_analysis = (
        ROOT / "run_headless/submit_hvawt_turbulent_20case_checkpoint_analysis.sj"
    )
    for path in (prep, submit, analysis, smoke, checkpoint, checkpoint_analysis):
        subprocess.run(["bash", "-n", str(path)], check=True)

    prep_text = prep.read_text(encoding="utf-8")
    assert "is_known_cluster_runtime_artifact" in prep_text
    assert "Binaries/XFoil" in prep_text
    assert "ControllerFiles/libvorlap_qblade_bridge.so" in prep_text
    assert "outside the two known cluster-built runtime artifacts" in prep_text

    text = submit.read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --ntasks-per-node=1" in text
    assert "#SBATCH --exclusive" in text
    assert "#SBATCH --array=0-101%34" in text
    assert hashlib.sha256(submit.read_bytes()).hexdigest() == (
        "8e6c4e129b680631c6f4a4005ec7c5bfbb96e875e8f2277a178ef93dde4e32bb"
    )
    assert "#SBATCH --time=47:59:00" in text
    assert "anaconda3/2025.12-2" in text
    assert "VORLAP_ROOT:-/pscratch/kevmoor/vorlap" in text
    assert "${VORLAP_ROOT}/.venv/bin/python" in text
    assert "SLURM_CPUS_ON_NODE" in text
    assert 'OMP_NUM_THREADS="${THREADS}"' in text
    assert " srun " not in text
    assert "taskset" not in text
    assert "--cpu-bind" not in text
    assert "--hint=" not in text
    assert 'CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-24000}"' in text
    assert '--checkpoint-every "${CHECKPOINT_EVERY}"' in text
    assert "full_rerun_execution_authorization.json" in text
    assert "u50_complete_four_step_smoke.json" in text

    smoke_text = smoke.read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in smoke_text
    assert "#SBATCH --exclusive" in smoke_text
    assert "anaconda3/2025.12-2" in smoke_text
    assert "smoke-u50" in smoke_text
    assert " srun " not in smoke_text
    assert "taskset" not in smoke_text

    checkpoint_text = checkpoint.read_text(encoding="utf-8")
    assert (
        "#SBATCH --array=0,2,9,11,18,20,27,29,36,38,45,47,54,56,63,65,81,83,90,92%20"
        in checkpoint_text
    )
    assert "run-checkpoint" in checkpoint_text
    assert "turbulent_20case_execution_authorization.json" in checkpoint_text
    assert "u50_complete_four_step_smoke.json" not in checkpoint_text
    assert " srun " not in checkpoint_text

    checkpoint_analysis_text = checkpoint_analysis.read_text(encoding="utf-8")
    assert "analyze-checkpoint" in checkpoint_analysis_text
    assert "turbulent_20case_execution_authorization.json" in checkpoint_analysis_text

    analysis_text = analysis.read_text(encoding="utf-8")
    assert "prepare_unified_hvawt_paper_results.py" in analysis_text
    assert '--output-dir "${CAMPAIGN_ROOT}/paper_export"' in analysis_text
    assert 'MPLCONFIGDIR="${MPLCONFIGDIR:-${CAMPAIGN_ROOT}/.matplotlib}"' in analysis_text

    prep_text = prep.read_text(encoding="utf-8")
    assert "bind-runtime" in prep_text
    assert "bind-authorization" in prep_text
    assert "bind-checkpoint-authorization" in prep_text
    assert "CHECKPOINT_EVIDENCE" in prep_text
    assert "PILOT_CAMPAIGN_ROOT" in prep_text
    assert "142c186c9e43ec914219a9d0bdf7e554b35902e3" in prep_text
    assert "merge-base --is-ancestor" in prep_text
    assert "status --porcelain --untracked-files=no" in prep_text


def test_authorization_binds_passed_full_pilot_and_detects_snapshot_tamper(
    tmp_path, monkeypatch
):
    root = tmp_path / "dlc"
    pilot_root = tmp_path / "pilot"
    root.mkdir()
    pilot_root.mkdir()
    spec_sha = "1" * 64
    model_sha = "2" * 64
    full_manifest = {
        "fingerprint": "3" * 64,
        "inputs": {
            "campaign_spec": {"sha256": spec_sha},
            "response_model": {"sha256": model_sha},
        },
    }
    pilot_manifest = {
        "fingerprint": "4" * 64,
        "case_count": 20,
        "matrix": {"campaign_kind": campaign.FULL_CAMPAIGN_KIND},
        "inputs": {
            "campaign_spec": {"sha256": spec_sha},
            "response_model": {"sha256": model_sha},
        },
    }
    pilot_manifest_path = pilot_root / "campaign_manifest.json"
    pilot_manifest_path.write_text(json.dumps(pilot_manifest), encoding="utf-8")
    analysis = _fingerprinted(
        {
            "schema": campaign.PILOT_ANALYSIS_SCHEMA,
            "status": "pilot_passed_complete_hybrid_screen",
            "campaign_kind": campaign.FULL_CAMPAIGN_KIND,
            "campaign_fingerprint": pilot_manifest["fingerprint"],
            "complete_case_count": 20,
            "expected_case_count": 20,
            "passed_case_count": 20,
            "full_rerun_authorized": True,
            "cases": [
                {"passed": True, "execution_receipt_pass": True} for _ in range(20)
            ],
            "cross_case_gates": [{"passed": True}],
        }
    )
    analysis_path = pilot_root / "pilot_analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    authorization = _fingerprinted(
        {
            "schema": campaign.FULL_RERUN_SCHEMA,
            "status": "authorized_for_matched_three_path_execution",
            "execution_authorized": True,
            "launch_implemented": True,
            "campaign_fingerprint": pilot_manifest["fingerprint"],
            "expected_runs": 102,
            "scenarios": list(campaign.FULL_RERUN_SCENARIOS),
            "pilot_analysis": campaign._identity(analysis_path),
            "source_manifest": {"sha256": campaign.APPROVED_MANIFEST_FILE_SHA256},
        }
    )
    (pilot_root / "full_rerun_plan_authorized.json").write_text(
        json.dumps(authorization), encoding="utf-8"
    )
    monkeypatch.setattr(
        campaign,
        "validate_campaign",
        lambda *_args, **_kwargs: {
            "campaign_root": root,
            "manifest": full_manifest,
        },
    )
    monkeypatch.setattr(
        campaign,
        "validate_pilot_campaign",
        lambda *_args, **_kwargs: {
            "campaign_root": pilot_root,
            "manifest_path": pilot_manifest_path,
            "manifest": pilot_manifest,
        },
    )

    receipt = campaign.bind_execution_authorization(root, pilot_campaign=pilot_root)
    assert receipt["expected_case_count"] == 102
    assert tuple(receipt["scenarios"]) == campaign.SCENARIOS
    assert campaign._load_execution_authorization(root, full_manifest) == receipt

    snapshot = root / receipt["snapshots"]["pilot_analysis"]["path"]
    snapshot.write_text("{}", encoding="utf-8")
    with pytest.raises(
        (campaign.HVAWTCompleteDLCError, campaign.HVAWTResponsePilotError),
        match="identity",
    ):
        campaign._load_execution_authorization(root, full_manifest)


def test_authorization_rejects_failed_pilot(tmp_path, monkeypatch):
    root = tmp_path / "dlc"
    pilot_root = tmp_path / "pilot"
    root.mkdir()
    pilot_root.mkdir()
    manifest = {
        "fingerprint": "a" * 64,
        "inputs": {
            "campaign_spec": {"sha256": "b" * 64},
            "response_model": {"sha256": "c" * 64},
        },
    }
    pilot_manifest = {
        "fingerprint": "d" * 64,
        "case_count": 20,
        "matrix": {"campaign_kind": campaign.FULL_CAMPAIGN_KIND},
        "inputs": manifest["inputs"],
    }
    pilot_manifest_path = pilot_root / "campaign_manifest.json"
    pilot_manifest_path.write_text(json.dumps(pilot_manifest), encoding="utf-8")
    failed = _fingerprinted(
        {
            "schema": campaign.PILOT_ANALYSIS_SCHEMA,
            "status": "pilot_failed_complete_hybrid_screen",
            "campaign_kind": campaign.FULL_CAMPAIGN_KIND,
            "campaign_fingerprint": pilot_manifest["fingerprint"],
            "complete_case_count": 20,
            "expected_case_count": 20,
            "passed_case_count": 19,
            "full_rerun_authorized": False,
            "cases": [],
            "cross_case_gates": [],
        }
    )
    (pilot_root / "pilot_analysis.json").write_text(
        json.dumps(failed), encoding="utf-8"
    )
    failed_path = pilot_root / "pilot_analysis.json"
    authorization = _fingerprinted(
        {
            "schema": campaign.FULL_RERUN_SCHEMA,
            "status": "authorized_for_matched_three_path_execution",
            "execution_authorized": True,
            "launch_implemented": True,
            "campaign_fingerprint": pilot_manifest["fingerprint"],
            "expected_runs": 102,
            "scenarios": list(campaign.FULL_RERUN_SCENARIOS),
            "pilot_analysis": campaign._identity(failed_path),
            "source_manifest": {"sha256": campaign.APPROVED_MANIFEST_FILE_SHA256},
        }
    )
    (pilot_root / "full_rerun_plan_authorized.json").write_text(
        json.dumps(authorization), encoding="utf-8"
    )
    monkeypatch.setattr(
        campaign,
        "validate_campaign",
        lambda *_args, **_kwargs: {"campaign_root": root, "manifest": manifest},
    )
    monkeypatch.setattr(
        campaign,
        "validate_pilot_campaign",
        lambda *_args, **_kwargs: {
            "campaign_root": pilot_root,
            "manifest_path": pilot_manifest_path,
            "manifest": pilot_manifest,
        },
    )

    with pytest.raises(campaign.HVAWTCompleteDLCError, match="has not passed"):
        campaign.bind_execution_authorization(root, pilot_campaign=pilot_root)


def test_u50_smoke_receipt_is_hash_bound_and_requires_finite_force(tmp_path):
    summary = tmp_path / "smoke_results/case/summary.json"
    history = tmp_path / "smoke_results/case/history.csv"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")
    history.write_text("time_s\n0\n", encoding="utf-8")
    cases = [{} for _ in range(102)]
    cases[101] = {
        "case_id": "dlc6p1_U50p00ms_s5533456__complete_hybrid_v8",
        "scenario": "complete_hybrid_v8",
        "wind_speed_mps": 50.0,
    }
    manifest = {"fingerprint": "a" * 64, "cases": cases}
    receipt = _fingerprinted(
        {
            "schema": "vorlap-hvawt-complete-dlc-four-step-smoke-v2",
            "status": "passed_input_runtime_finite_force_smoke_only",
            "campaign_fingerprint": manifest["fingerprint"],
            "case_index": 101,
            "case_id": cases[101]["case_id"],
            "completed_steps": 4,
            "maximum_total_supplemental_nodal_force_n": 42.0,
            "maximum_applied_residual_force_n": 21.0,
            "summary": campaign._relative_identity(summary, tmp_path),
            "history": campaign._relative_identity(history, tmp_path),
        }
    )
    (tmp_path / campaign.U50_SMOKE_RECEIPT_NAME).write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    assert campaign._load_u50_smoke_receipt(tmp_path, manifest) == receipt
    receipt["maximum_total_supplemental_nodal_force_n"] = 0.0
    invalid = dict(receipt)
    invalid.pop("fingerprint")
    receipt["fingerprint"] = campaign.canonical_fingerprint(invalid)
    (tmp_path / campaign.U50_SMOKE_RECEIPT_NAME).write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(campaign.HVAWTCompleteDLCError, match="invalid"):
        campaign._load_u50_smoke_receipt(tmp_path, manifest)
