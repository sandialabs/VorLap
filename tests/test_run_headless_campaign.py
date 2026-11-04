from __future__ import annotations

import importlib.util
import json
import math
import shlex
import sys
import csv
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import vorlap.qblade_batch as qblade_batch_service


RUN_HEADLESS = Path(__file__).resolve().parents[1] / "run_headless"
if str(RUN_HEADLESS) not in sys.path:
    sys.path.insert(0, str(RUN_HEADLESS))


def load_script(name: str):
    module_path = RUN_HEADLESS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


batch = load_script("vorlap_qblade_batch")
campaign = load_script("run_parked_demo_campaign")
qblade_runner = load_script("run_qblade_sim")
audit = load_script("audit_parked_campaign")
archive_builder = load_script("build_production_archive_manifest")
signed_selection_builder = load_script("build_signed_channel_selection")
ultimate_basis_builder = load_script("build_ultimate_input_basis")
structural_qualification_builder = load_script("build_structural_model_qualification")
spectral_psd_builder = load_script("build_spectral_psd_comparison")
postprocess = load_script("postprocess_parked_demo")
postprocess_compare = load_script("compare_parked_postprocess")
dlc_aggregator = load_script("aggregate_parked_dlc_manifest")
sign_probe = load_script("run_force_sign_probe_campaign")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def first_json_object(text: str) -> dict:
    payload, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    return payload


def test_reference_selection_requires_an_explicit_nondefault_model(tmp_path, monkeypatch):
    example = tmp_path / "examples" / "custom_model"
    example.mkdir(parents=True)
    (example / "custom.sim").write_text("model.trb TURBFILE\n", encoding="utf-8")
    monkeypatch.delenv("VORLAP_QBLADE_REFERENCE_CASE", raising=False)
    monkeypatch.delenv("VORLAP_QBLADE_REFERENCE_SIM", raising=False)

    with pytest.raises(FileNotFoundError, match="reference QBlade model not found"):
        qblade_batch_service.reference_case_dir_for_root(tmp_path)
    monkeypatch.setenv("VORLAP_QBLADE_REFERENCE_CASE", "custom_model")
    assert qblade_batch_service.reference_case_dir_for_root(tmp_path) == example
    assert qblade_batch_service.reference_sim_name_for_root(tmp_path) == "custom.sim"


def test_qblade_runner_cli_parses_explicit_compact_sil_history(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qblade_sim.py",
            "--sim",
            "case.sim",
            "--sil-vorlap",
            "--sil-vorlap-history-mode",
            "compact",
        ],
    )

    args = qblade_runner.parse_args()

    assert args.record_sil_vorlap_history is False
    assert args.sil_vorlap_history_mode == "compact"


def test_del_uses_one_fixed_equivalent_cycle_count_for_paired_histories():
    exponent = 10.0
    equivalent_cycles = 600.0
    one_cycle = np.asarray([-1.0, 1.0, -1.0])
    two_cycles = np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0])

    one_del, one_count = postprocess.equivalent_load_with_cycles(
        one_cycle,
        exponent,
        equivalent_cycles,
    )
    two_del, two_count = postprocess.equivalent_load_with_cycles(
        two_cycles,
        exponent,
        equivalent_cycles,
    )

    assert one_count == 1.0
    assert two_count == 2.0
    assert np.isclose(two_del / one_del, 2.0 ** (1.0 / exponent))
    assert not np.isclose(two_del / one_del, 1.0)

    validation = postprocess.rainflow_validation_table(
        exponent,
        equivalent_cycles,
    )
    assert validation["passed"].all()
    assert set(validation["equivalent_cycles"]) == {equivalent_cycles}


def test_raw_postprocess_manifest_hashes_inputs_outputs_and_aggregator_accepts_it(tmp_path):
    cases = []
    for scenario in ("baseline", "vorlap"):
        raw_path = tmp_path / scenario / "case_qblade_results.txt"
        raw_path.parent.mkdir()
        raw_path.write_text(f"raw-{scenario}\n", encoding="utf-8")
        cases.append(
            postprocess.CaseResult(
                scenario=scenario,
                label="case",
                wind_speed=35.0,
                azimuth_deg=0.0,
                path=raw_path,
                data=pd.DataFrame({"Time [s]": [0.0]}),
            )
        )
    out_dir = tmp_path / "postprocess_raw_neq600"
    out_dir.mkdir()
    outputs = []
    for name in (
        "signed_del_comparison.csv",
        "extreme_channel_metrics.csv",
        "source_coverage.csv",
    ):
        path = out_dir / name
        path.write_text("value\n1\n", encoding="utf-8")
        outputs.append(path)
    manifest_path = out_dir / "postprocess_manifest.json"
    payload = postprocess.write_raw_postprocess_manifest(
        manifest_path,
        cases,
        ["baseline", "vorlap"],
        "baseline",
        "vorlap",
        10.0,
        600.0,
        postprocess.FatigueQualityThresholds(1000, 600.0, 10.0),
        outputs,
    )

    assert payload["passed"] is True
    assert payload["source_mode"] == "raw_qblade_exports"
    assert len(payload["inputs"]) == 2
    assert postprocess_compare.validate_raw_manifest(manifest_path, 600.0, 10.0)[0] == 2
    assert dlc_aggregator.validate_derived_postprocess(
        out_dir,
        tuple(outputs),
        600.0,
        10.0,
    ) == "raw_qblade_exports"


def test_postprocess_comparison_is_keyed_and_ignores_local_paths(tmp_path):
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        {
            "case": ["b", "a"],
            "value": [2.0, 1.0],
            "result_file": ["/remote/b", "/remote/a"],
        }
    ).to_csv(reference, index=False)
    pd.DataFrame(
        {
            "case": ["a", "b"],
            "value": [1.0 + 1.0e-14, 2.0],
            "result_file": ["/local/a", "/local/b"],
        }
    ).to_csv(candidate, index=False)

    rows = postprocess_compare.compare_artifact(
        reference,
        candidate,
        ["case"],
        {"result_file"},
        set(),
        row_id="row",
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    assert len(rows) == 1
    assert rows[0]["passed"] is True
    assert rows[0]["max_absolute_error"] == pytest.approx(1.0e-14)


def test_campaign_settings_json_is_published_atomically(tmp_path):
    path = tmp_path / "campaign_settings.json"
    payload = {"campaign_settings_version": 2, "wind_speeds": [5.0, 50.0]}

    campaign.atomic_write_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(".campaign_settings.json.tmp-*"))


def test_audit_accepts_only_the_approved_naca0018_reynolds_envelope_extension():
    parent_sha256 = "a" * 64
    source_row = {
        "airfoil_id": "NACA0018",
        "generator": "scripts/process_naluout_to_h5.py",
        "reynolds_model": "replicated_reynolds_independent",
        "reynolds": [10_000.0, 500_000.0, 2_000_000.0],
        "processing_config": {
            "replicate_re_parent_sha256": parent_sha256,
            "reynolds_model": "replicated_reynolds_independent",
            "source_reynolds": [500_000.0],
            "output_reynolds": [10_000.0, 500_000.0, 2_000_000.0],
            "reynolds_envelope_extension_exact": True,
            "reynolds_envelope_extension_generator": "scripts/extend_replicated_reynolds_grid.py",
            "reynolds_envelope_parent_grid": [30_000.0, 500_000.0, 2_000_000.0],
        },
    }

    assert audit.approved_reynolds_envelope_extension_matches(source_row, parent_sha256)
    source_row["processing_config"]["replicate_re_parent_sha256"] = "b" * 64
    assert not audit.approved_reynolds_envelope_extension_matches(source_row, parent_sha256)
    source_row["processing_config"]["replicate_re_parent_sha256"] = parent_sha256
    source_row["reynolds"][0] = 20_000.0
    assert not audit.approved_reynolds_envelope_extension_matches(source_row, parent_sha256)
    source_row["reynolds"][0] = 10_000.0
    source_row["processing_config"]["reynolds_envelope_extension_exact"] = False
    assert not audit.approved_reynolds_envelope_extension_matches(source_row, parent_sha256)


def test_10s_sample_wrappers_postprocess_baseline_vs_sil_before_figures():
    docker_text = (RUN_HEADLESS / "run_qblade_vorlap_10s_sample.sh").read_text(encoding="utf-8")
    common_text = (RUN_HEADLESS / "run_qblade_vorlap_10s_sample_linux_common.sh").read_text(encoding="utf-8")
    for text in (docker_text, common_text):
        assert "run_sample_postprocess || return 0" in text
        assert "postprocess_parked_demo.py" in text
        assert '--comparison-baseline baseline' in text
        assert "POSTPROCESS_INCLUDE_SMOKE_IN_DEL" in text
    assert '--scenarios "baseline sil_vorlap"' in docker_text
    assert '--comparison-coupled sil_vorlap' in docker_text
    assert '--scenarios "baseline ${COUPLED_SCENARIO}"' in common_text
    assert '--comparison-coupled "${COUPLED_SCENARIO}"' in common_text

    sil_cpu = (RUN_HEADLESS / "run_qblade_vorlap_sil_10s_sample_linux_cpu.sh").read_text(encoding="utf-8")
    sil_gpu = (RUN_HEADLESS / "run_qblade_vorlap_sil_10s_sample_linux_gpu.sh").read_text(encoding="utf-8")
    for text in (sil_cpu, sil_gpu):
        assert "COUPLED_SCENARIO=\"${COUPLED_SCENARIO:-sil_vorlap}\"" in text
        assert "COUPLED_COUPLING=\"${COUPLED_COUPLING:-sil-vorlap}\"" in text
        assert "NATIVE_RUN_FULL_SIMULATION=\"${NATIVE_RUN_FULL_SIMULATION:-0}\"" in text


def test_10s_qblade_driven_wrappers_select_native_external_callback_path():
    cpu_text = (RUN_HEADLESS / "run_qblade_vorlap_qblade_driven_10s_sample_linux_cpu.sh").read_text(
        encoding="utf-8"
    )
    gpu_text = (RUN_HEADLESS / "run_qblade_vorlap_qblade_driven_10s_sample_linux_gpu.sh").read_text(
        encoding="utf-8"
    )
    for text in (cpu_text, gpu_text):
        assert "COUPLED_SCENARIO=\"${COUPLED_SCENARIO:-qblade_driven_vorlap}\"" in text
        assert "COUPLED_COUPLING=\"${COUPLED_COUPLING:-qblade-external}\"" in text
        assert "NATIVE_RUN_FULL_SIMULATION=\"${NATIVE_RUN_FULL_SIMULATION:-1}\"" in text
        assert "CHECKPOINT_EVERY=\"${CHECKPOINT_EVERY:-0}\"" in text
        assert "SKIP_BRIDGE_BUILD=\"${SKIP_BRIDGE_BUILD:-0}\"" in text
        assert "FORCE_BRIDGE_BUILD=\"${FORCE_BRIDGE_BUILD:-1}\"" in text


def test_10s_common_runner_applies_native_full_simulation_only_to_qblade_external():
    text = (RUN_HEADLESS / "run_qblade_vorlap_10s_sample_linux_common.sh").read_text(encoding="utf-8")

    assert 'if [[ "${NATIVE_RUN_FULL_SIMULATION}" == "1" && "${coupling}" == "qblade-external" ]]' in text
    assert 'if [[ "${use_native_full_simulation}" == "1" ]]; then' in text
    assert 'command+=(--native-run-full-simulation)' in text


def test_prune_restart_checkpoints_keeps_latest_two_complete_pairs(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    stems = []
    for step in (200, 400, 600):
        stem = f"case_step_{step:07d}"
        stems.append(stem)
        project = checkpoint_dir / f"{stem}.qpr"
        project.write_text(f"project {step}\n", encoding="utf-8")
        write_json(
            checkpoint_dir / f"{stem}.json",
            {
                "completed_steps": step,
                "qblade_project": str(project),
            },
        )
    write_json(checkpoint_dir / "latest_checkpoint.json", {"completed_steps": 600})

    deleted = qblade_runner.prune_restart_checkpoints(checkpoint_dir, keep=2)

    assert any(stems[0] in path for path in deleted)
    assert not (checkpoint_dir / f"{stems[0]}.json").exists()
    assert not (checkpoint_dir / f"{stems[0]}.qpr").exists()
    for stem in stems[1:]:
        assert (checkpoint_dir / f"{stem}.json").exists()
        assert (checkpoint_dir / f"{stem}.qpr").exists()
    assert (checkpoint_dir / "latest_checkpoint.json").exists()


def test_headless_runner_persists_qblade_reported_geometry_diagnostics(tmp_path, monkeypatch):
    summary_path = tmp_path / "case_run_summary.json"
    result = SimpleNamespace(
        summary={"completed_steps": 4},
        summary_path=summary_path,
        result_path=tmp_path / "case_results.txt",
    )
    expected = {
        "geometry_audit_reported_updated": True,
        "geometry_audit_reported_available_count": 39,
        "geometry_audit_reported_frame_fit_status": "within_tolerance",
    }
    observed_call = {}

    def record_geometry_call(observed, *, output_path):
        observed_call.update(observed=observed, output_path=output_path)
        return expected if observed is result else {}

    monkeypatch.setattr(
        qblade_runner,
        "write_run_geometry_audit_reported_positions",
        record_geometry_call,
    )

    diagnostics = qblade_runner.finalize_run_geometry_audit(result)

    assert diagnostics == expected
    assert observed_call == {
        "observed": result,
        "output_path": tmp_path / "case_results_geometry_audit_reported.csv",
    }
    assert result.summary["geometry_audit_reported_diagnostics"] == expected
    assert json.loads(summary_path.read_text(encoding="utf-8"))["geometry_audit_reported_diagnostics"] == expected


def write_geometry_audit(case_dir: Path, node_id: str = "BLD_1_0.000000") -> Path:
    path = case_dir / "reference_model" / "Control" / "vorlap_qblade_geometry_audit.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "swap_index": 0,
                "controller_index": 0,
                "requested_node_id": node_id,
                "canonical_node_id": node_id,
                "controller_node_id": node_id,
                "action_id": "BLD_1",
                "action_position": 0.0,
                "force_swap_x": 6,
                "force_swap_y": 7,
                "force_swap_z": 8,
                "velocity_swap_x": 3,
                "velocity_swap_y": 4,
                "velocity_swap_z": 5,
                "vorlap_x_m": 0.0,
                "vorlap_y_m": 0.0,
                "vorlap_z_m": 0.0,
                "chord_norm": 1.0,
                "normal_norm": 1.0,
                "span_norm": 1.0,
                "chord_dot_normal": 0.0,
                "chord_dot_span": 0.0,
                "normal_dot_span": 0.0,
                "basis_determinant": 1.0,
                "basis_status": "ok",
            }
        ]
    ).to_csv(path, index=False)
    return path


def write_stable_support_patch(case_dir: Path, *, timestep: float = 0.0075, rampup: float = 0.0, add_damp: float = 0.0) -> Path:
    path = case_dir / "reference_model" / "Control" / "vorlap_stable_support_patch.json"
    structural = [
        {
            "file": "reference_model/Structure/reference_main.str",
            "keyword": "SUBFILE",
            "old_value": "legacy.str",
            "new_value": "",
        }
    ]
    write_json(
        path,
        {
            "applied": True,
            "scope": "generated_case_only",
            "structural_patches": structural,
            "sim_patches": [
                {"file": "baseline_reference.sim", "keyword": "RELAXSTEPS", "new_value": "0"},
                {"file": "baseline_reference.sim", "keyword": "RELAXTIMESTEP", "new_value": f"{timestep:g}"},
                {"file": "baseline_reference.sim", "keyword": "RAMPUP", "new_value": f"{rampup:g}"},
                {"file": "baseline_reference.sim", "keyword": "ADDDAMP", "new_value": f"{add_damp:g}"},
            ],
        },
    )
    return path


def test_repo_local_batch_manifest_expands_fixed_azimuth_matrix(tmp_path):
    args = SimpleNamespace(
        run_root=tmp_path / "run",
        qblade_root=tmp_path / "QBladeCE_2.0.9.7",
        vorlap_root=Path(__file__).resolve().parents[1],
        python=sys.executable,
        wind_speeds="5 10",
        azimuths="range:0:120:60",
        rpm=0.0,
        timesteps=80000,
        timestep=0.0075,
        store_from=60.0,
        store_replay=False,
        n_freq_depth=20,
        source_bounds_policy="warn",
        force_scale=1.0,
        diagnostic_force_scale=False,
        wind_type=0,
        wind_name="",
        skip_stable_support_patch=False,
        rampup=0.0,
        add_damp=0.0,
        str_iterations=None,
        allow_partial=False,
        coupling="sil-vorlap",
    )

    manifest = batch.build_case_manifest(args)

    assert len(manifest["cases"]) == 6
    assert manifest["settings"]["rpm"] == 0.0
    assert manifest["settings"]["force_scale"] == 1.0
    assert manifest["settings"]["azimuths_deg"] == [0.0, 60.0, 120.0]
    assert manifest["cases"][0]["label"] == "U05p00ms_A000p00deg"
    assert manifest["cases"][2]["azimuth_deg"] == 120.0


def test_batch_setup_regenerates_when_cached_manifest_differs(tmp_path, monkeypatch):
    args = SimpleNamespace(
        run_root=tmp_path / "run",
        qblade_root=tmp_path / "QBladeCE_2.0.9.7",
        vorlap_root=Path(__file__).resolve().parents[1],
        python=sys.executable,
        wind_speeds="5",
        azimuths="0",
        rpm=0.0,
        timesteps=80000,
        timestep=0.0075,
        store_from=60.0,
        store_replay=False,
        n_freq_depth=20,
        source_bounds_policy="warn",
        force_scale=1.0,
        diagnostic_force_scale=False,
        wind_type=0,
        wind_name="",
        skip_stable_support_patch=False,
        rampup=0.0,
        add_damp=0.0,
        str_iterations=None,
        allow_partial=False,
        coupling="sil-vorlap",
        force_setup=False,
    )
    args.run_root.mkdir(parents=True)
    old_manifest = batch.build_case_manifest(args)
    write_json(args.run_root / "cases.json", old_manifest)
    write_json(args.run_root / "setup_complete.json", {"setup_complete_unix": 1.0, "cases": len(old_manifest["cases"])})

    args.wind_speeds = "10"
    prepared: list[tuple[bool, dict]] = []
    monkeypatch.setattr(qblade_batch_service, "validate_qblade_root", lambda _root: None)
    monkeypatch.setattr(qblade_batch_service, "install_vorlap", lambda _args: None)
    monkeypatch.setattr(qblade_batch_service, "build_bridge", lambda _args: None)

    def capture_prepare(_args, manifest):
        prepared.append((bool(_args.force_setup), manifest))

    monkeypatch.setattr(qblade_batch_service, "prepare_cases", capture_prepare)

    manifest = batch.setup(args)

    assert prepared
    assert prepared[0][0] is True
    assert manifest["settings"]["wind_speeds_mps"] == [10.0]
    assert json.loads((args.run_root / "cases.json").read_text(encoding="utf-8")) == manifest


def test_batch_setup_reuses_matching_cached_manifest_despite_creation_time(tmp_path, monkeypatch):
    args = SimpleNamespace(
        run_root=tmp_path / "run",
        qblade_root=tmp_path / "QBladeCE_2.0.9.7",
        vorlap_root=Path(__file__).resolve().parents[1],
        python=sys.executable,
        wind_speeds="5",
        azimuths="0",
        rpm=0.0,
        timesteps=80000,
        timestep=0.0075,
        store_from=60.0,
        store_replay=False,
        n_freq_depth=20,
        source_bounds_policy="warn",
        force_scale=1.0,
        diagnostic_force_scale=False,
        wind_type=0,
        wind_name="",
        skip_stable_support_patch=False,
        rampup=0.0,
        add_damp=0.0,
        str_iterations=None,
        allow_partial=False,
        coupling="sil-vorlap",
        force_setup=False,
    )
    args.run_root.mkdir(parents=True)
    cached_manifest = batch.build_case_manifest(args)
    cached_manifest["created_unix"] = 1.0
    write_json(args.run_root / "cases.json", cached_manifest)
    write_json(args.run_root / "setup_complete.json", {"setup_complete_unix": 1.0, "cases": len(cached_manifest["cases"])})
    cached_sim = Path(cached_manifest["cases"][0]["sim"])
    cached_sim.parent.mkdir(parents=True)
    cached_sim.write_text(
        "5 MEANINF - mean wind\n"
        "0 INITIAL_AZIMUTH - azimuth\n"
        "1 PRESCRIBETYPE - rpm prescribe mode\n"
        "0 RPMPRESCRIBED - rpm\n"
        "80000 NUMTIMESTEPS - steps\n"
        "0.0075 TIMESTEP - timestep\n"
        "60 STOREFROM - store from\n"
        "false STOREREPLAY - replay\n"
        "0 WNDTYPE - wind type\n"
        "0 HORANGLE - horizontal inflow angle\n"
        "0 VERTANGLE - vertical inflow angle\n"
        "0.23 SHEAREXP - shear exponent\n"
        "18 REFHEIGHT - reference height\n"
        "1.0296 DENSITYAIR - density\n"
        "0.00001647 VISCOSITYAIR - viscosity\n"
        " WNDNAME - unused wind file\n",
        encoding="utf-8",
    )
    assets, fingerprint = qblade_batch_service.qblade_runtime_asset_fingerprint(cached_sim)
    cached_manifest["cases"][0]["runtime_assets"] = assets
    cached_manifest["cases"][0]["runtime_asset_fingerprint"] = fingerprint
    write_json(args.run_root / "cases.json", cached_manifest)

    monkeypatch.setattr(qblade_batch_service, "validate_qblade_root", lambda _root: None)

    def fail_setup_step(*_args, **_kwargs):
        raise AssertionError("matching cached setup should be reused")

    monkeypatch.setattr(qblade_batch_service, "install_vorlap", fail_setup_step)
    monkeypatch.setattr(qblade_batch_service, "build_bridge", fail_setup_step)
    monkeypatch.setattr(qblade_batch_service, "prepare_cases", fail_setup_step)

    assert batch.setup(args) == cached_manifest


def test_batch_setup_rebuilds_cached_case_with_malformed_long_wind_name(
    tmp_path,
    monkeypatch,
    capsys,
):
    args = SimpleNamespace(
        run_root=tmp_path / "run",
        qblade_root=tmp_path / "QBladeCE_2.0.9.7",
        vorlap_root=Path(__file__).resolve().parents[1],
        python=sys.executable,
        wind_speeds="5",
        azimuths="0",
        rpm=1.0,
        timesteps=88000,
        timestep=0.0075,
        store_from=60.0,
        store_replay=False,
        n_freq_depth=20,
        source_bounds_policy="warn",
        force_scale=0.0,
        diagnostic_force_scale=False,
        wind_type=1,
        wind_name=(
            "/pscratch/kevmoor/vorlap_wind_fields/changed_geometry_trial/"
            "{speed_label}/dlc6p4_{speed_label}.bts"
        ),
        skip_stable_support_patch=False,
        rampup=0.0,
        add_damp=0.0,
        str_iterations=None,
        allow_partial=False,
        coupling="none",
        force_setup=False,
    )
    cached_manifest = batch.build_case_manifest(args)
    write_json(args.run_root / "cases.json", cached_manifest)
    write_json(
        args.run_root / "setup_complete.json",
        {"setup_complete_unix": 1.0, "cases": len(cached_manifest["cases"])},
    )
    case = cached_manifest["cases"][0]
    sim_path = Path(case["sim"])
    sim_path.parent.mkdir(parents=True)
    sim_path.write_text(
        "5 MEANINF - mean wind\n"
        "0 INITIAL_AZIMUTH - azimuth\n"
        "1 PRESCRIBETYPE - rpm prescribe mode\n"
        "1 RPMPRESCRIBED - rpm\n"
        "88000 NUMTIMESTEPS - steps\n"
        "0.0075 TIMESTEP - timestep\n"
        "60 STOREFROM - store from\n"
        "false STOREREPLAY - replay\n"
        "1 WNDTYPE - wind type\n"
        " WNDNAME - malformed missing wind path\n",
        encoding="utf-8",
    )

    prepared: list[bool] = []
    monkeypatch.setattr(qblade_batch_service, "validate_qblade_root", lambda _root: None)
    monkeypatch.setattr(qblade_batch_service, "install_vorlap", lambda _args: None)
    monkeypatch.setattr(qblade_batch_service, "build_bridge", lambda _args: None)
    monkeypatch.setattr(
        qblade_batch_service,
        "prepare_cases",
        lambda setup_args, _manifest: prepared.append(bool(setup_args.force_setup)),
    )

    batch.setup(args)

    assert prepared == [True]
    captured = capsys.readouterr()
    assert "Existing setup artifacts do not match the cached manifest" in captured.err
    assert "WNDNAME is None" in captured.err


def test_stable_support_patch_writes_machine_readable_manifest(tmp_path, monkeypatch):
    case_dir = tmp_path / "case"
    structure = case_dir / "reference_model" / "Structure" / "reference_main.str"
    turbine = case_dir / "reference_model" / "test.trb"
    sim = case_dir / "baseline_reference.sim"
    structure.parent.mkdir(parents=True)
    sim.parent.mkdir(parents=True, exist_ok=True)
    structure.write_text(
        "\n".join(
            [
                "10.0 TWRHEIGHT - tower height",
                "2.0 TRQTBCLEAR - torque tube clearance",
                "3.0 HUBPOS - hub position",
                "4.0 TRQTBCONN - torque tube connection",
                "5.0 RTRCLEAR - rotor clearance",
                "legacy_sub.str SUBFILE - substructure",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    sim.write_text(
        "\n".join(
            [
                "reference_model/test.trb TURBFILE - turbine definition",
                "100 RELAXSTEPS - static relaxation steps",
                "0.1 RELAXTIMESTEP - relaxation timestep",
                "5.0 RAMPUP - rampup",
                "0.2 ADDDAMP - damping",
                "8 STRITERATIONS - structural iterations",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    turbine.write_text(
        "Structure/reference_main.str STRUCTURALFILE - structural model\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("VORLAP_STABLE_PATCH_REMOVE_SUBFILE", raising=False)
    batch.apply_stable_support_patch(case_dir, sim, timestep=0.0075, rampup=0.0, add_damp=0.0, str_iterations=2)
    assert "legacy_sub.str" in structure.read_text(encoding="utf-8")
    monkeypatch.setenv("VORLAP_STABLE_PATCH_REMOVE_SUBFILE", "1")
    batch.apply_stable_support_patch(case_dir, sim, timestep=0.0075, rampup=0.0, add_damp=0.0, str_iterations=2)

    manifest = json.loads(
        (case_dir / "reference_model" / "Control" / "vorlap_stable_support_patch.json").read_text(encoding="utf-8")
    )
    structural = {row["keyword"]: row for row in manifest["structural_patches"]}
    sim_rows = {row["keyword"]: row for row in manifest["sim_patches"]}

    assert manifest["applied"] is True
    assert manifest["scope"] == "generated_case_only"
    assert "TWRHEIGHT" not in structural
    assert "TRQTBCLEAR" not in structural
    assert "HUBPOS" not in structural
    assert "TRQTBCONN" not in structural
    assert "RTRCLEAR" not in structural
    assert structural["SUBFILE"]["new_value"] == ""
    assert "10.0 TWRHEIGHT" in structure.read_text(encoding="utf-8")
    assert "2.0 TRQTBCLEAR" in structure.read_text(encoding="utf-8")
    assert "3.0 HUBPOS" in structure.read_text(encoding="utf-8")
    assert "4.0 TRQTBCONN" in structure.read_text(encoding="utf-8")
    assert "5.0 RTRCLEAR" in structure.read_text(encoding="utf-8")
    assert sim_rows["RELAXSTEPS"]["new_value"] == "0"
    assert sim_rows["RELAXTIMESTEP"]["new_value"] == "0.0075"
    assert sim_rows["STRITERATIONS"]["new_value"] == "2"


def test_stable_support_audit_rejects_collapsed_rotor_geometry_patch(tmp_path):
    case_dir = tmp_path / "case"
    path = write_stable_support_patch(case_dir)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["structural_patches"].append(
        {
            "file": "reference_model/Structure/reference_main.str",
            "keyword": "HUBPOS",
            "old_value": "1.3",
            "new_value": "0.001",
        }
    )
    write_json(path, manifest)

    issues: list[audit.AuditIssue] = []
    audit.audit_stable_support_patch_file(
        path,
        {"timestep_s": 0.0075, "rampup_s": 0.0, "add_damp_s": 0.0, "str_iterations": None},
        issues,
        scenario="vorlap",
        case_label="U10p58ms",
    )

    assert "stable_support_patch_rotor_geometry_modified" in {item.code for item in issues}


def test_parked_campaign_postprocess_command_passes_ultimate_allowables(tmp_path):
    allowables = tmp_path / "allowables.csv"
    args = SimpleNamespace(
        run_root=tmp_path / "run",
        python=sys.executable,
        baseline_force_scale=0.0,
        identity_force_scale=0.0,
        vorlap_force_scale=1.0,
        identity_coupling="sil-vorlap",
        coupling="sil-vorlap",
        postprocess_locations="0.230 0.500",
        postprocess_blade="BLD_1",
        wohler_exponent=10.0,
        ultimate_allowables=allowables,
        include_smoke_in_del=True,
    )

    command = campaign.postprocess_command(args)

    assert "--ultimate-allowables" in command
    assert str(allowables) in command
    assert "--include-smoke-in-del" in command


def test_parked_campaign_batch_command_passes_opencl_flags(tmp_path):
    result_filter = tmp_path / "fatigue.filter"
    result_filter.write_text("Time [s]\nLSS Rpm [rpm]\n", encoding="utf-8")
    args = SimpleNamespace(
        python=sys.executable,
        run_root=tmp_path / "run",
        qblade_root=tmp_path / "QBladeCE",
        vorlap_root=tmp_path / "vorlap",
        wind_speeds="5",
        azimuths="0",
        timestep=0.0075,
        store_from=0.0,
        store_replay=True,
        result_filter=result_filter,
        rpm=0.0,
        rampup=0.0,
        add_damp=0.0,
        source_bounds_policy="warn",
        wake_profile="parked-reduced-v1",
        threads=8,
        progress_every=100,
        checkpoint_every=4000,
        coupling="sil-vorlap",
        use_opencl=True,
        opencl_device=2,
        opencl_group_size=64,
        resume_initialize=False,
        str_iterations=None,
        wind_name="",
        qblade_steps=None,
        native_run_full_simulation=False,
        store_project=True,
        diagnostic_force_scale=False,
        allow_partial=False,
        record_sil_vorlap_history=False,
        reuse_setup=True,
        force_bridge_build=False,
        skip_vorlap_install=True,
        skip_bridge_build=True,
        skip_stable_support_patch=False,
        allow_coarse_timestep=False,
    )

    command = campaign.batch_command(args, "vorlap", 1.0, 80000)

    assert "--use-opencl" in command
    assert command[command.index("--opencl-device") + 1] == "2"
    assert command[command.index("--opencl-group-size") + 1] == "64"
    assert command[command.index("--source-bounds-policy") + 1] == "warn"
    assert command[command.index("--wake-profile") + 1] == "parked-reduced-v1"
    assert command[command.index("--result-filter") + 1] == str(result_filter)
    assert "--store-project" in command


def test_parked_campaign_postprocess_defaults_can_reuse_archived_settings(tmp_path):
    allowables = tmp_path / "allowables.csv"
    args = SimpleNamespace(
        postprocess_locations=None,
        postprocess_blade=None,
        wohler_exponent=None,
        equivalent_cycles=None,
        ultimate_allowables=None,
    )

    campaign.apply_postprocess_defaults(
        args,
        {
            "postprocess_locations": "0.111 0.222",
            "postprocess_blade": "BLD_2",
            "wohler_exponent": 8.0,
            "equivalent_cycles": 900.0,
            "ultimate_allowables": str(allowables),
        },
    )

    assert args.postprocess_locations == "0.111 0.222"
    assert args.postprocess_blade == "BLD_2"
    assert args.wohler_exponent == 8.0
    assert args.equivalent_cycles == 900.0
    assert args.ultimate_allowables == allowables.resolve()


def test_postprocess_uses_sampled_record_duration():
    duration = postprocess.sampled_duration_seconds(np.asarray([0.0, 0.0075, 0.015], dtype=float))

    assert duration == 0.0225


def test_postprocess_duration_gate_allows_one_hundredth_second_shortfall():
    thresholds = postprocess.FatigueQualityThresholds(
        min_samples=1000,
        min_duration_s=600.0,
        min_rainflow_cycles=10.0,
    )

    assert postprocess.meets_minimum_duration(599.99, 600.0)
    assert postprocess.meets_minimum_duration(599.992, 600.0)
    assert not postprocess.meets_minimum_duration(599.989, 600.0)
    assert "completed_time_s<600" not in postprocess.completion_warnings(
        pd.Series({"completed_steps": 80000, "completed_time_s": 599.992, "failed": False}),
        thresholds,
    )
    assert not postprocess.fatigue_metric_warnings(
        samples=80000,
        duration_s=599.992,
        force_cycles=10.0,
        moment_cycles=10.0,
        thresholds=thresholds,
    )
    assert not postprocess.signed_metric_warnings(
        samples=80000,
        duration_s=599.992,
        cycles=10.0,
        thresholds=thresholds,
    )


def test_postprocess_completion_summary_infers_duration_from_qblade_export(tmp_path):
    run_root = tmp_path / "run"
    result_dir = run_root / "baseline" / "results" / "U10p58ms"
    write_json(
        run_root / "baseline" / "cases.json",
        {
            "cases": [
                {
                    "label": "U10p58ms",
                    "wind_speed": 10.58,
                    "azimuth_deg": 270.0,
                    "result_dir": str(result_dir),
                    "result_name": "U10p58ms_qblade_results",
                }
            ]
        },
    )
    write_json(
        result_dir / "baseline_reference_run_summary.json",
        {
            "requested_steps": 1334,
            "completed_steps": 1334,
            "failed": False,
            "exported_results": True,
            "qblade_export_time_diagnostics": {
                "samples": 1334,
                "first_time_s": 0.0,
                "last_time_s": 9.9975,
            },
        },
    )

    table = postprocess.load_completion_summary(run_root, ["baseline"], math.nan)

    assert table.loc[0, "completed_time_s"] == 10.005


def test_parked_campaign_dry_run_records_postprocess_reproducibility(tmp_path):
    run_root = tmp_path / "dryrun"
    allowables = tmp_path / "allowables.csv"

    subprocess.run(
        [
            sys.executable,
            str(RUN_HEADLESS / "run_parked_demo_campaign.py"),
            "--run-root",
            str(run_root),
            "--qblade-root",
            str(tmp_path / "missing_qblade"),
            "--wind-speeds",
            "5",
            "--azimuths",
            "0",
            "--duration",
            "600",
            "--timestep",
            "0.0075",
            "--qblade-steps",
            "20",
            "--postprocess-locations",
            "0.230 0.770",
            "--postprocess-blade",
            "BLD_2",
            "--wohler-exponent",
            "8",
            "--ultimate-allowables",
            str(allowables),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    settings = json.loads((run_root / "campaign_settings.json").read_text(encoding="utf-8"))

    assert settings["qblade_steps_override"] == 20
    assert settings["native_run_full_simulation"] is False
    assert settings["postprocess_locations"] == "0.230 0.770"
    assert settings["postprocess_blade"] == "BLD_2"
    assert settings["wohler_exponent"] == 8.0
    assert settings["ultimate_allowables"] == str(allowables.resolve())
    assert settings["reuse_setup"] is False


def test_parked_campaign_case_index_dry_run_lists_one_case_commands(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_HEADLESS / "run_parked_demo_campaign.py"),
            "--run-root",
            str(tmp_path / "case_index_dryrun"),
            "--qblade-root",
            str(tmp_path / "missing_qblade"),
            "--wind-speeds",
            "5 10 30 50",
            "--azimuths",
            "0",
            "--duration",
            "600",
            "--timestep",
            "0.0075",
            "--rpm",
            "1",
            "--case-index",
            "2",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["case_index"] == 2
    assert len(payload["cases_per_scenario"]) == 4
    assert payload["scenario_case_commands"] is not None
    assert len(payload["scenario_case_commands"]) == 2
    for command in payload["scenario_case_commands"]:
        assert command[2] == "run"
        assert command[command.index("--case-index") + 1] == "2"
        assert "--force-setup" not in command


def test_parked_campaign_can_select_baseline_vorlap_only(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_HEADLESS / "run_parked_demo_campaign.py"),
            "--run-root",
            str(tmp_path / "baseline_vorlap_dryrun"),
            "--qblade-root",
            str(tmp_path / "missing_qblade"),
            "--wind-speeds",
            "5 10",
            "--azimuths",
            "0",
            "--duration-minutes",
            "1",
            "--timestep",
            "0.0075",
            "--rpm",
            "1",
            "--scenarios",
            "baseline vorlap",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    settings = json.loads((tmp_path / "baseline_vorlap_dryrun" / "campaign_settings.json").read_text(encoding="utf-8"))
    assert payload["duration_s"] == 60.0
    assert payload["timesteps_per_case"] == 8000
    assert [scenario["name"] for scenario in payload["scenarios"]] == ["baseline", "vorlap"]
    assert settings["scenario_names"] == ["baseline", "vorlap"]
    assert settings["planned_case_count_total"] == 4
    postprocess = payload["postprocess_command"]
    assert postprocess[postprocess.index("--scenarios") + 1] == "baseline vorlap"
    assert postprocess[postprocess.index("--comparison-baseline") + 1] == "baseline"
    assert postprocess[postprocess.index("--comparison-coupled") + 1] == "vorlap"


def test_parked_campaign_only_postprocess_preserves_existing_settings(tmp_path):
    run_root = tmp_path / "existing_campaign"
    original_settings = {
        "wind_speeds": [5.0, 10.0],
        "azimuths_deg": [0.0, 60.0],
        "planned_cases": [{"wind_speed_mps": 5.0, "parked_azimuth_deg": 0.0}],
        "planned_case_count_per_scenario": 4,
        "planned_case_count_total": 12,
        "scenario_names": ["baseline", "identity", "vorlap"],
        "scenario_force_scales": {"baseline": 0.0, "identity": 0.0, "vorlap": 1.0},
        "scenario_couplings": {"baseline": "none", "identity": "sil-vorlap", "vorlap": "sil-vorlap"},
        "duration_s": 600.0,
        "timestep_s": 0.0075,
        "timesteps": 80000,
        "rpm": 0.0,
        "postprocess_locations": "0.230",
    }
    write_json(run_root / "campaign_settings.json", original_settings)
    before = (run_root / "campaign_settings.json").read_text(encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(RUN_HEADLESS / "run_parked_demo_campaign.py"),
            "--run-root",
            str(run_root),
            "--qblade-root",
            str(tmp_path / "missing_qblade"),
            "--wind-speeds",
            "99",
            "--azimuths",
            "30",
            "--duration",
            "120",
            "--timestep",
            "0.0075",
            "--only-postprocess",
            "--skip-postprocess",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (run_root / "campaign_settings.json").read_text(encoding="utf-8") == before


def test_parked_campaign_only_postprocess_requires_existing_settings(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_HEADLESS / "run_parked_demo_campaign.py"),
            "--run-root",
            str(tmp_path / "missing_campaign"),
            "--only-postprocess",
            "--skip-postprocess",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "requires an existing campaign settings file" in completed.stderr


def test_production_submission_script_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")], check=True)


def test_all_slurm_submission_scripts_notify_completion_and_failure():
    scripts = sorted(RUN_HEADLESS.glob("*.sj"))
    assert scripts
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        if "#SBATCH" not in text:
            continue
        # Historical launchers are byte-bound to campaign receipts. FAIL,END
        # provides the required notifications without changing their identity.
        assert any(
            directive in text
            for directive in ("#SBATCH --mail-type=ALL", "#SBATCH --mail-type=FAIL,END")
        ), script
        assert "#SBATCH --mail-user=kevmoor@sandia.gov" in text, script


def test_all_slurm_submission_scripts_defer_filesystem_licenses_to_scheduler():
    scripts = sorted(RUN_HEADLESS.glob("*.sj"))
    assert scripts
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "#SBATCH --licenses" not in text, script


def test_production_submission_script_is_array_oriented_one_node_per_case():
    script = (RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")

    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --time=47:59:00" in script
    assert "#SBATCH --account=FY260158" in script
    assert "#SBATCH --partition=batch" in script
    assert "#SBATCH --qos=normal" in script
    assert "#SBATCH --reservation=flight-cldera" in script
    assert "#SBATCH --licenses" not in script
    assert "#SBATCH --array" not in script
    assert "SUBMIT_WIND_JOBS" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "--case-index" in script
    assert "SLURM_CPUS_ON_NODE SLURM_JOB_CPUS_PER_NODE SLURM_CPUS_PER_TASK" in script


def test_nyquist_ten_minute_profile_is_self_contained(tmp_path):
    script_path = RUN_HEADLESS / "submit_nyquist_10min_u05_u50.sj"
    subprocess.run(["bash", "-n", str(script_path)], check=True)
    script = script_path.read_text(encoding="utf-8")

    assert "#SBATCH --exclusive" in script
    assert "#SBATCH --time=47:59:00" in script
    assert "#SBATCH --array=0-1%2" in script
    assert 'export QBLADE_DURATION="660"' in script
    assert 'export QBLADE_STORE_FROM="60"' in script
    assert 'export QBLADE_TIMESTEP="0.0075"' in script
    assert 'export CAMPAIGN_SCENARIOS="baseline vorlap"' in script
    assert 'export VORLAP_SOURCE_FREQUENCY_TRUNCATION="nyquist"' in script
    assert 'export QBLADE_WAKE_PROFILE="parked-reduced-v2"' in script
    assert 'export QBLADE_RESULT_FILTER="${QBLADE_RESULT_FILTER:-${VORLAP_ROOT}/run_headless/config/parked_fatigue_results.filter}"' in script
    assert "reduced_wake_v3_filtered_20260803" in script
    assert 'export QBLADE_STORE_REPLAY="false"' in script
    assert 'export CASE_WORKERS="1"' in script
    assert 'export VORLAP_AUTO_NODE_THREADS="1"' in script
    assert 'export GENERATE_TURBSIM="1"' in script
    assert 'export ADOPT_EXISTING_TURBSIM="1"' in script
    assert 'export FORCE_GENERATE_TURBSIM="0"' in script
    assert 'export REUSE_SETUP="1"' in script
    assert 'export RUN_FINAL_AUDIT="1"' in script
    assert "unset QBLADE_STEPS" in script
    assert "unset SLURM_ARRAY_TASK_ID" in script

    env = os.environ.copy()
    env.update(
        {
            "VORLAP_ROOT": str(RUN_HEADLESS.parent),
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "TURBSIM_WIND_ROOT": str(tmp_path / "winds"),
            "RUN_ROOT": str(tmp_path / "must_not_be_inherited"),
            "VORLAP_PROFILE_RUN_ROOT": str(tmp_path / "profile"),
            "PYTHON_EXE": sys.executable,
            "DRY_RUN_ONLY": "1",
            "SLURM_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_CPUS_ON_NODE": "20",
            "SLURM_CPUS_PER_TASK": "1",
            "OMP_NUM_THREADS": "3",
        }
    )
    completed = subprocess.run(
        ["bash", str(script_path)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    dry_plan = first_json_object(completed.stdout)
    settings = json.loads((tmp_path / "profile" / "U50p00ms" / "campaign_settings.json").read_text())
    assert "speed=50m/s" in completed.stdout
    assert "must_not_be_inherited" not in completed.stdout
    assert "QBlade CPU policy: OMP_NUM_THREADS=20; CASE_WORKERS=1." in completed.stdout
    assert settings["wind_speeds"] == [50.0]
    assert settings["scenario_names"] == ["baseline", "vorlap"]
    assert settings["duration_s"] == 660.0
    assert settings["store_from_s"] == 60.0
    assert settings["min_fatigue_duration_s"] == 600.0
    assert settings["timesteps"] == 88_000
    assert settings["threads"] == 20
    assert settings["case_workers"] == 1
    assert settings["checkpoint_every"] == 4000
    assert settings["store_replay"] is False
    assert settings["result_filter"] == str(RUN_HEADLESS / "config" / "parked_fatigue_results.filter")
    assert settings["result_filter_sha256"]
    assert settings["source_frequency_truncation"] == "nyquist"
    assert settings["wake_profile"] == "parked-reduced-v2"
    assert settings["wake_settings"]["MAXWAKESIZE"] == 120000
    assert settings["wake_settings"]["MAXWAKEDIST"] == 10
    assert settings["wake_settings"]["DYNSTALLTYPE"] == 1
    assert settings["wake_settings"]["TF_OYE"] == 8
    assert settings["wake_settings"]["BOUNDCORERADIUS"] == 1.0
    assert settings["wake_settings"]["WAKECORERADIUS"] == 0.05
    assert settings["wake_settings"]["VORTEXVISCOSITY"] == 80.0
    assert settings["turbsim_seed_base"] == 4433457
    assert settings["turbsim_wind_files"][0]["settings"]["RandSeed1"] == 4433457
    assert settings["generate_turbsim"] is True
    assert settings["adopt_existing_turbsim"] is True
    assert settings["force_generate_turbsim"] is False
    assert settings["reuse_setup"] is True
    assert all("--force-setup" not in command for command in dry_plan["scenario_setup_commands"])


def test_nyquist_old_run_recovery_profile_is_restart_safe_and_filtered():
    script_path = RUN_HEADLESS / "submit_nyquist_10min_u05_u50_recover_old.sj"
    subprocess.run(["bash", "-n", str(script_path)], check=True)
    script = script_path.read_text(encoding="utf-8")

    assert "#SBATCH --time=47:59:00" in script
    assert "#SBATCH --array=0-1%2" in script
    assert "reduced_wake_v2_20260801" in script
    assert "recover-export" in script
    assert "--result-filter" in script
    assert "--source-frequency-truncation nyquist" in script
    assert "--wake-profile parked-reduced-v2" in script
    assert "--only-postprocess" in script
    assert "--require-postprocess-artifacts" in script
    assert "--force-setup" not in script
    assert "GENERATE_TURBSIM" not in script


def test_nyquist_reference_wake_old_run_recovery_profile_is_restart_safe_and_filtered():
    script_path = RUN_HEADLESS / "submit_nyquist_10min_u05_u10_recover_reference_wake.sj"
    subprocess.run(["bash", "-n", str(script_path)], check=True)
    script = script_path.read_text(encoding="utf-8")

    assert "#SBATCH --time=47:59:00" in script
    assert "#SBATCH --array=0-1%2" in script
    assert "vorlap_nyquist_10min_U05_U10_20260801" in script
    assert 'recovery_wind_speed="10"' in script
    assert 'recovery_wind_label="U10p00ms"' in script
    assert "recover-export" in script
    assert "--result-filter" in script
    assert "--source-frequency-truncation nyquist" in script
    assert script.count("--wake-profile reference") == 2
    assert "parked-reduced" not in script
    assert "--only-postprocess" in script
    assert "--require-postprocess-artifacts" in script
    assert "--force-setup" not in script
    assert "GENERATE_TURBSIM" not in script


def test_completed_filtered_export_ready_requires_matching_result_and_filter_hashes(tmp_path):
    result_dir = tmp_path / "results" / "U05p00ms"
    result_dir.mkdir(parents=True)
    sim = tmp_path / "cases" / "U05p00ms" / "case.sim"
    sim.parent.mkdir(parents=True)
    sim.write_text("case\n", encoding="utf-8")
    result_filter = tmp_path / "fatigue.filter"
    result_filter.write_text("Time [s]\n", encoding="utf-8")
    result_path = result_dir / "U05p00ms_qblade_results.txt"
    result_path.write_text("filtered results\n", encoding="utf-8")
    case = {
        "sim": str(sim),
        "result_dir": str(result_dir),
        "result_name": "U05p00ms_qblade_results",
    }
    summary = {
        "requested_steps": 88000,
        "completed_steps": 88000,
        "failed": False,
        "exported_results": True,
        "restart": {"completed_checkpoint_export_only": False},
        "result_filter_identity": qblade_batch_service.file_sha256(result_filter),
        "result_file_identity": qblade_batch_service.file_sha256(result_path),
    }
    write_json(result_dir / "case_run_summary.json", summary)

    assert batch.completed_filtered_export_ready(case, result_filter) is True
    assert (
        batch.completed_filtered_export_ready(
            case,
            result_filter,
            require_completed_checkpoint_export_only=True,
        )
        is False
    )
    summary["restart"]["completed_checkpoint_export_only"] = True
    write_json(result_dir / "case_run_summary.json", summary)
    assert (
        batch.completed_filtered_export_ready(
            case,
            result_filter,
            require_completed_checkpoint_export_only=True,
        )
        is True
    )
    result_path.write_text("modified results\n", encoding="utf-8")
    assert batch.completed_filtered_export_ready(case, result_filter) is False


def test_campaign_preserves_only_complete_hash_verified_scenario(tmp_path):
    result_filter = tmp_path / "fatigue.filter"
    result_filter.write_text("Time [s]\n", encoding="utf-8")
    scenario_root = tmp_path / "baseline"
    sim = scenario_root / "cases" / "U05p00ms" / "case.sim"
    result_dir = scenario_root / "results" / "U05p00ms"
    sim.parent.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    sim.write_text("case\n", encoding="utf-8")
    result_path = result_dir / "U05p00ms_qblade_results.txt"
    result_path.write_text("filtered results\n", encoding="utf-8")
    case = {
        "label": "U05p00ms",
        "sim": str(sim),
        "result_dir": str(result_dir),
        "result_name": "U05p00ms_qblade_results",
    }
    write_json(scenario_root / "cases.json", {"cases": [case]})
    write_json(
        result_dir / "case_run_summary.json",
        {
            "requested_steps": 88_000,
            "completed_steps": 88_000,
            "failed": False,
            "exported_results": True,
            "result_filter_identity": qblade_batch_service.file_sha256(result_filter),
            "result_file_identity": qblade_batch_service.file_sha256(result_path),
        },
    )
    args = SimpleNamespace(
        run_root=tmp_path,
        preserve_completed_scenarios=True,
        result_filter=result_filter,
        wind_speeds="5",
        azimuths="0",
    )

    assert campaign.completed_scenario_can_be_preserved(args, "baseline") is True
    result_path.write_text("changed\n", encoding="utf-8")
    assert campaign.completed_scenario_can_be_preserved(args, "baseline") is False


def test_parked_dlc6x_wrapper_scripts_have_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(RUN_HEADLESS / "python_env.sh")], check=True)
    subprocess.run(["bash", "-n", str(RUN_HEADLESS / "run_parked_dlc6x_linux_gpu.sh")], check=True)
    subprocess.run(["bash", "-n", str(RUN_HEADLESS / "run_parked_dlc6x_linux_cpu.sh")], check=True)


def test_u03_recovery_submission_is_qblade_only_and_fail_closed_on_baseline_reuse():
    path = RUN_HEADLESS / "submit_parked_dlc_u03_recovery.sj"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text(encoding="utf-8")

    assert "#SBATCH --array=0-2%3" in text
    assert "#SBATCH --time=47:59:00" in text
    assert "verify_u03_recovery.py" in text
    assert "export DLC_PHASE=qblade" in text
    assert "export PRESERVE_COMPLETED_SCENARIOS=1" in text
    assert "DLC_PHASE=wind" not in text
    parent_text = (RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")
    assert "--preserve-completed-scenarios" in parent_text
    subprocess.run(["bash", "-n", str(RUN_HEADLESS / "run_qblade_vorlap_10s_sample.sh")], check=True)
    subprocess.run(["bash", "-n", str(RUN_HEADLESS / "run_qblade_vorlap_10s_sample_linux_common.sh")], check=True)
    subprocess.run(["bash", "-n", str(RUN_HEADLESS / "run_qblade_vorlap_sil_10s_sample_linux_cpu.sh")], check=True)
    subprocess.run(["bash", "-n", str(RUN_HEADLESS / "run_qblade_vorlap_sil_10s_sample_linux_gpu.sh")], check=True)
    subprocess.run(
        ["bash", "-n", str(RUN_HEADLESS / "run_qblade_vorlap_qblade_driven_10s_sample_linux_cpu.sh")], check=True
    )
    subprocess.run(
        ["bash", "-n", str(RUN_HEADLESS / "run_qblade_vorlap_qblade_driven_10s_sample_linux_gpu.sh")], check=True
    )


def test_python_env_helper_validates_current_environment():
    env = os.environ.copy()
    env.update(
        {
            "AUTO_SETUP_PYTHON": "0",
            "PYTHON_TEST": sys.executable,
            "VORLAP_DIR": str(RUN_HEADLESS.parent),
        }
    )
    helper = shlex.quote(str(RUN_HEADLESS / "python_env.sh"))
    command = f"""
set -e
log() {{ printf '%s\\n' "$*"; }}
die() {{ printf 'ERROR: %s\\n' "$*" >&2; exit 1; }}
source {helper}
ensure_vorlap_python_environment PYTHON_TEST "test runtime" "" 1 numpy scipy pandas h5py plotly matplotlib
printf 'python=%s\\n' "$PYTHON_TEST"
"""

    completed = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True, check=True)

    assert "Python environment ready for test runtime" in completed.stdout
    assert f"python={sys.executable}" in completed.stdout


def test_python_env_helper_repairs_nonstarting_turbsim_and_keeps_installer_output_visible(tmp_path):
    vorlap_root = tmp_path / "vorlap"
    installer = vorlap_root / "scripts" / "install_openfast_turbsim.sh"
    installer.parent.mkdir(parents=True)
    installer.write_text(
        """#!/usr/bin/env bash
set -e
printf '#!/bin/sh\\nprintf "Running TurbSim version 5.0.0\\\\n"\\n' > "${TURBSIM_EXE}"
chmod 755 "${TURBSIM_EXE}"
printf 'automatic installer output is visible\\n'
""",
        encoding="utf-8",
    )
    installer.chmod(0o755)
    broken = tmp_path / "broken-turbsim"
    broken.write_text("#!/bin/sh\necho broken >&2\nexit 127\n", encoding="utf-8")
    broken.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "AUTO_SETUP_TURBSIM": "1",
            "GENERATE_TURBSIM": "1",
            "TURBSIM_EXE": str(broken),
            "VORLAP_DIR": str(vorlap_root),
        }
    )
    helper = shlex.quote(str(RUN_HEADLESS / "python_env.sh"))
    command = f"""
set -e
log() {{ printf '%s\\n' "$*"; }}
die() {{ printf 'ERROR: %s\\n' "$*" >&2; exit 1; }}
source {helper}
ensure_vorlap_turbsim
"""

    completed = subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Configured TurbSim executable is missing or failed its startup check" in completed.stdout
    assert "automatic installer output is visible" in completed.stdout
    assert "Automatic TurbSim setup/repair succeeded" in completed.stdout


def test_python_env_helper_force_rebuilds_usable_turbsim(tmp_path):
    vorlap_root = tmp_path / "vorlap"
    installer = vorlap_root / "scripts" / "install_openfast_turbsim.sh"
    installer.parent.mkdir(parents=True)
    installer.write_text(
        """#!/usr/bin/env bash
set -e
printf 'forced installer ran\\n'
printf '#!/bin/sh\\nprintf "Running TurbSim version 5.0.1\\\\n"\\n' > "${TURBSIM_EXE}"
chmod 755 "${TURBSIM_EXE}"
""",
        encoding="utf-8",
    )
    installer.chmod(0o755)
    turbsim = tmp_path / "turbsim"
    turbsim.write_text(
        '#!/bin/sh\nprintf "Running TurbSim version 5.0.0\\n"\n',
        encoding="utf-8",
    )
    turbsim.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "AUTO_SETUP_TURBSIM": "1",
            "FORCE_OPENFAST_TURBSIM_BUILD": "1",
            "GENERATE_TURBSIM": "1",
            "TURBSIM_EXE": str(turbsim),
            "VORLAP_DIR": str(vorlap_root),
        }
    )
    helper = shlex.quote(str(RUN_HEADLESS / "python_env.sh"))
    command = f"""
set -e
log() {{ printf '%s\\n' "$*"; }}
die() {{ printf 'ERROR: %s\\n' "$*" >&2; exit 1; }}
source {helper}
ensure_vorlap_turbsim
"""

    completed = subprocess.run(
        ["bash", "-c", command],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Forced TurbSim reinstall/rebuild requested." in completed.stdout
    assert "forced installer ran" in completed.stdout
    assert "version 5.0.1" in turbsim.read_text(encoding="utf-8")


def test_python_env_helper_reports_manual_setup_when_auto_setup_disabled(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "AUTO_SETUP_PYTHON": "0",
            "PYTHON_TEST": str(tmp_path / "missing-python"),
            "VORLAP_DIR": str(RUN_HEADLESS.parent),
        }
    )
    helper = shlex.quote(str(RUN_HEADLESS / "python_env.sh"))
    command = f"""
set -e
log() {{ printf '%s\\n' "$*"; }}
die() {{ printf 'ERROR: %s\\n' "$*" >&2; exit 1; }}
source {helper}
ensure_vorlap_python_environment PYTHON_TEST "test runtime" "skip hint" 0 numpy
"""

    completed = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True)

    assert completed.returncode != 0
    assert "Automatic setup is disabled by AUTO_SETUP_PYTHON=0." in completed.stdout
    assert "skip hint" in completed.stdout
    assert "Python environment is not ready for test runtime." in completed.stderr


def test_python_env_helper_passes_trusted_hosts_to_pip(tmp_path):
    fake_python = tmp_path / "fake-python"
    pip_log = tmp_path / "pip.log"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -e
if [[ "$1" == "-m" && "$2" == "pip" && "$3" == "--version" ]]; then
  printf 'pip 99.0\\n'
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "pip" ]]; then
  {
    printf 'PIP_TRUSTED_HOST=%s\\n' "${PIP_TRUSTED_HOST:-}"
    printf 'ARGS=%s\\n' "$*"
  } >> "${VORLAP_FAKE_PIP_LOG}"
  exit 0
fi
printf 'unexpected fake-python command: %s\\n' "$*" >&2
exit 1
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "AUTO_SETUP_PYTHON": "1",
            "PYTHON_TEST": str(fake_python),
            "VORLAP_DIR": str(RUN_HEADLESS.parent),
            "VORLAP_FAKE_PIP_LOG": str(pip_log),
            "VORLAP_PIP_TRUSTED_HOSTS": "pypi.org files.pythonhosted.org pypi.python.org mirror.example.edu",
        }
    )
    helper = shlex.quote(str(RUN_HEADLESS / "python_env.sh"))
    command = f"""
set -e
log() {{ printf '%s\\n' "$*"; }}
die() {{ printf 'ERROR: %s\\n' "$*" >&2; exit 1; }}
source {helper}
vorlap_setup_python_environment PYTHON_TEST
printf 'python=%s\\n' "$PYTHON_TEST"
"""

    completed = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True, check=True)
    pip_text = pip_log.read_text(encoding="utf-8")

    trusted_hosts = "pypi.org files.pythonhosted.org pypi.python.org mirror.example.edu"
    assert f"Using pip trusted host(s): {trusted_hosts}" in completed.stdout
    assert f"PIP_TRUSTED_HOST={trusted_hosts}" in pip_text
    assert pip_text.count("--trusted-host pypi.org") == 2
    assert pip_text.count("--trusted-host files.pythonhosted.org") == 2
    assert pip_text.count("--trusted-host pypi.python.org") == 2
    assert pip_text.count("--trusted-host mirror.example.edu") == 2


def test_batch_install_vorlap_uses_cluster_safe_pip_options(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_checked(command, *, cwd=None, env=None):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env

    monkeypatch.setattr(qblade_batch_service, "run_checked", fake_run_checked)
    monkeypatch.setenv("VORLAP_PIP_TRUSTED_HOSTS", "pypi.org files.pythonhosted.org mirror.example.edu")
    monkeypatch.delenv("VORLAP_PIP_NO_BUILD_ISOLATION", raising=False)
    args = SimpleNamespace(skip_vorlap_install=False, python="python3", vorlap_root=tmp_path / "vorlap")

    qblade_batch_service.install_vorlap(args)

    command = captured["command"]
    env = captured["env"]
    assert isinstance(command, list)
    assert isinstance(env, dict)
    assert command[:5] == ["python3", "-m", "pip", "install", "--no-deps"]
    assert "--no-build-isolation" in command
    assert command.count("--trusted-host") == 3
    assert "mirror.example.edu" in command
    assert command[-2:] == ["-e", str(args.vorlap_root)]
    assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert env["PIP_TRUSTED_HOST"] == "pypi.org files.pythonhosted.org mirror.example.edu"


def test_batch_build_bridge_rebuilds_stale_python_linked_library(tmp_path, monkeypatch, capsys):
    qblade_root = tmp_path / "QBladeCE_2.0.9.7"
    controller_files = qblade_root / "ControllerFiles"
    controller_files.mkdir(parents=True)
    existing = controller_files / "libvorlap_qblade_bridge.so"
    existing.write_text("stale bridge\n", encoding="utf-8")
    run_root = tmp_path / "run"
    vorlap_root = tmp_path / "vorlap"
    (vorlap_root / "scripts").mkdir(parents=True)
    args = SimpleNamespace(
        skip_bridge_build=False,
        qblade_root=qblade_root,
        run_root=run_root,
        vorlap_root=vorlap_root,
        python="python3",
        force_bridge_build=False,
    )
    validate_calls: list[Path] = []
    build_calls: list[tuple[list[str], Path | None, dict | None]] = []

    def fake_validate(path: Path) -> None:
        validate_calls.append(path)
        if len(validate_calls) == 1:
            raise OSError("libpython3.10.so.1.0: cannot open shared object file")

    def fake_run_checked(command, *, cwd=None, env=None):
        build_calls.append((command, cwd, env))
        existing.write_text("rebuilt bridge\n", encoding="utf-8")

    monkeypatch.setattr(qblade_batch_service, "validate_bridge_library_loadable", fake_validate)
    monkeypatch.setattr(qblade_batch_service, "run_checked", fake_run_checked)

    qblade_batch_service.build_bridge(args)

    assert len(validate_calls) == 2
    assert len(build_calls) == 1
    command, cwd, env = build_calls[0]
    assert command[0] == str(vorlap_root / "scripts" / "build_qblade_external_linux.sh")
    assert command[1] == str(run_root / "build" / "qblade_external_linux")
    assert command[2] == str(controller_files)
    assert cwd == vorlap_root
    assert env is not None
    assert env["PYTHON_EXE"] == "python3"
    assert "not loadable with python3" in capsys.readouterr().err


def test_production_submission_script_dry_run_reaches_plan_audit(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_dryrun"),
            "DRY_RUN_ONLY": "1",
            "FORCE_SETUP": "1",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "QBLADE_DURATION": "600",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert '"stage": "plan"' in completed.stdout
    assert '"passed": true' in completed.stdout
    assert (tmp_path / "submit_dryrun" / "campaign_settings.json").is_file()
    plan_audit = json.loads((tmp_path / "submit_dryrun" / "audits" / "plan_audit.json").read_text(encoding="utf-8"))
    assert plan_audit["stage"] == "plan"
    assert plan_audit["passed"] is True


def test_production_submission_script_default_dry_run_is_idling_wind_speed_matrix(tmp_path):
    env = os.environ.copy()
    env.pop("QBLADE_STORE_REPLAY", None)
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_production_dryrun"),
            "DRY_RUN_ONLY": "1",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    settings = json.loads((tmp_path / "submit_production_dryrun" / "campaign_settings.json").read_text(encoding="utf-8"))
    plan_audit = json.loads((tmp_path / "submit_production_dryrun" / "audits" / "plan_audit.json").read_text(encoding="utf-8"))

    assert completed.stderr == ""
    assert settings["wind_speeds"] == [5.0, 10.0, 30.0, 50.0]
    assert settings["azimuths_deg"] == [0.0]
    assert settings["rpm"] == 1.0
    assert settings["rpm_prescribe_type"] == 1
    assert settings["duration_s"] == 660.0
    assert settings["store_from_s"] == 60.0
    assert settings["store_replay"] is False
    assert settings["min_fatigue_duration_s"] == 600.0
    assert settings["scenario_names"] == ["baseline", "vorlap"]
    assert settings["planned_case_count_per_scenario"] == 4
    assert settings["planned_case_count_total"] == 8
    assert settings["airfoil_source_manifest_error"] == ""
    assert settings["airfoil_source_manifest"]
    naca0018 = next(
        row for row in settings["airfoil_source_manifest"]
        if row["airfoil_id"] == "NACA0018"
    )
    assert naca0018["reynolds"] == [10_000.0, 500_000.0, 2_000_000.0]
    assert abs(naca0018["thickness_ratio"] - 0.18) < 1.0e-12
    assert naca0018["reynolds_model"] == "replicated_reynolds_independent"
    assert len(naca0018["sha256"]) == 64
    assert plan_audit["passed"] is True


def test_production_submission_script_rejects_less_than_600_retained_seconds(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_short_retained_record"),
            "DRY_RUN_ONLY": "1",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "QBLADE_DURATION": "600",
            "QBLADE_STORE_FROM": "60",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        env=env,
        text=True,
        capture_output=True,
    )
    plan_audit = json.loads(
        (tmp_path / "submit_short_retained_record" / "audits" / "plan_audit.json").read_text(
            encoding="utf-8"
        )
    )
    codes = {issue["code"] for issue in plan_audit["issues"]}

    assert completed.returncode != 0
    assert plan_audit["passed"] is False
    assert "analyzed_duration_too_short" in codes


def test_production_submission_script_accepts_sim_minutes(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_one_minute_dryrun"),
            "DRY_RUN_ONLY": "1",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "SIM_MINUTES": "1",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    settings = json.loads((tmp_path / "submit_one_minute_dryrun" / "campaign_settings.json").read_text(encoding="utf-8"))
    plan_audit = json.loads((tmp_path / "submit_one_minute_dryrun" / "audits" / "plan_audit.json").read_text(encoding="utf-8"))

    assert settings["duration_s"] == 60.0
    assert settings["timesteps"] == 8000
    assert plan_audit["passed"] is True


def test_production_submission_script_array_case_dry_run_uses_node_thread_count(tmp_path):
    env = os.environ.copy()
    env.pop("OMP_NUM_THREADS", None)
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_array_case_dryrun"),
            "ARRAY_CASE_DRY_RUN": "1",
            "SLURM_ARRAY_TASK_ID": "2",
            "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_CPUS_ON_NODE": "20",
            "SLURM_CPUS_PER_TASK": "1",
            "SLURM_JOB_NODELIST": "node-test",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5 10 30 50",
            "QBLADE_AZIMUTHS": "0",
            "QBLADE_DURATION": "600",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    payload = first_json_object(completed.stdout)

    assert "Running Slurm array case index 2 on node-test; OMP_NUM_THREADS=20." in completed.stdout
    assert payload["case_index"] == 2
    assert len(payload["scenario_case_commands"]) == 2
    assert [scenario["name"] for scenario in payload["scenarios"]] == ["baseline", "vorlap"]
    for command in payload["scenario_case_commands"]:
        assert command[2] == "run"
        assert command[command.index("--case-index") + 1] == "2"
        assert command[command.index("--threads") + 1] == "20"
        assert "--force-setup" not in command
    assert (tmp_path / "submit_array_case_dryrun" / "campaign_settings.json").is_file()
    assert not (tmp_path / "submit_array_case_dryrun" / "audits").exists()


def test_production_submission_script_child_replaces_inherited_thread_count(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_child_thread_dryrun"),
            "DRY_RUN_ONLY": "1",
            "SLURM_JOB_ID": "12345",
            "SLURM_CPUS_ON_NODE": "20",
            "SLURM_CPUS_PER_TASK": "1",
            "OMP_NUM_THREADS": "3",
            "VORLAP_AUTO_NODE_THREADS": "1",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "SIM_MINUTES": "1",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    settings = json.loads(
        (tmp_path / "submit_child_thread_dryrun" / "campaign_settings.json").read_text(encoding="utf-8")
    )
    assert "QBlade CPU policy: OMP_NUM_THREADS=20; CASE_WORKERS=1." in completed.stdout
    assert settings["threads"] == 20
    assert settings["case_workers"] == 1


def test_changed_geometry_trial_submission_profile_defaults_to_two_scenarios(tmp_path):
    script = RUN_HEADLESS / "submit_changed_geometry_trial.sj"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --time=02:00:00" in text
    assert "#SBATCH --partition=short" in text
    assert "#SBATCH --array=0-1" in text
    assert 'export GENERATE_TURBSIM="${GENERATE_TURBSIM:-1}"' in text
    assert 'export AUTO_SETUP_FIRST_RUN="${AUTO_SETUP_FIRST_RUN:-1}"' in text
    assert 'export AUTO_SETUP_PYTHON="${AUTO_SETUP_PYTHON:-1}"' in text
    assert 'export AUTO_SETUP_TURBSIM="${AUTO_SETUP_TURBSIM:-1}"' in text
    assert 'export QBLADE_STORE_REPLAY="${QBLADE_STORE_REPLAY:-true}"' in text
    assert 'export QBLADE_STORE_PROJECT="${QBLADE_STORE_PROJECT:-true}"' in text
    assert 'export QBLADE_CHECKPOINT_EVERY="${QBLADE_CHECKPOINT_EVERY:-250}"' in text
    assert 'export FORCE_REINSTALL_SETUP="${FORCE_REINSTALL_SETUP:-0}"' in text
    assert 'export FORCE_CASE_SETUP="${FORCE_CASE_SETUP:-1}"' in text
    assert "export FORCE_OPENFAST_TURBSIM_BUILD=1" in text
    assert "export FORCE_BRIDGE_BUILD=1" in text
    assert "export FORCE_CASE_SETUP=1" in text
    assert 'export VORLAP_SERIALIZE_FORCE_SETUP="${VORLAP_SERIALIZE_FORCE_SETUP:-1}"' in text
    assert "export FORCE_GENERATE_TURBSIM=1" not in text
    assert 'export SKIP_BRIDGE_BUILD="${SKIP_BRIDGE_BUILD:-0}"' in text
    spool_dir = tmp_path / "slurm-spool"
    spool_dir.mkdir()
    spooled_script = spool_dir / script.name
    spooled_script.write_text(text, encoding="utf-8")
    spooled_script.chmod(0o755)

    env = os.environ.copy()
    for name in (
        "WIND_SPEEDS",
        "PRODUCTION_WIND_SPEEDS",
        "QBLADE_AZIMUTHS",
        "QBLADE_RPM",
        "QBLADE_DURATION",
        "QBLADE_STORE_FROM",
        "QBLADE_STORE_REPLAY",
        "QBLADE_STORE_PROJECT",
        "QBLADE_CHECKPOINT_EVERY",
        "QBLADE_TIMESTEP",
        "QBLADE_STEPS",
        "CAMPAIGN_SCENARIOS",
        "CASE_WORKERS",
        "OMP_NUM_THREADS",
    ):
        env.pop(name, None)
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "changed_geometry_trial"),
            "ARRAY_CASE_DRY_RUN": "1",
            "SLURM_ARRAY_TASK_ID": "1",
            "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_CPUS_PER_TASK": "16",
            "SLURM_JOB_NODELIST": "trial-node",
            "SLURM_SUBMIT_DIR": str(RUN_HEADLESS.parent),
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(spooled_script)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
        cwd=tmp_path,
    )
    payload = first_json_object(completed.stdout)
    settings = json.loads(
        (tmp_path / "changed_geometry_trial" / "campaign_settings.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["case_index"] == 1
    assert "Changed-geometry trial:" in completed.stdout
    assert "Wind-import smoke:" not in completed.stdout
    assert payload["duration_s"] == 660.0
    assert payload["rpm"] == 1.0
    assert settings["rpm_prescribe_type"] == 1
    assert [case["wind_speed_mps"] for case in payload["cases_per_scenario"]] == [5.0, 50.0]
    assert [scenario["name"] for scenario in payload["scenarios"]] == ["baseline", "vorlap"]
    assert len(payload["scenario_case_commands"]) == 2
    assert settings["qblade_steps_override"] == 500
    assert settings["generate_turbsim"] is True
    assert settings["store_from_s"] == 0.0
    assert settings["store_replay"] is True
    assert settings["store_project"] is True
    assert settings["checkpoint_every"] == 250
    assert settings["case_workers"] == 1
    assert settings["threads"] == 16
    for command in payload["scenario_case_commands"]:
        assert command[command.index("--case-index") + 1] == "1"
        assert command[command.index("--qblade-steps") + 1] == "500"
        assert command[command.index("--threads") + 1] == "16"
        assert "--store-project" in command


def test_changed_geometry_trial_submission_queues_dependent_postprocess(tmp_path):
    wrapper = RUN_HEADLESS / "submit_changed_geometry_trial_with_postprocess.sh"
    postprocess = RUN_HEADLESS / "postprocess_changed_geometry_trial.sj"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    subprocess.run(["bash", "-n", str(postprocess)], check=True)

    fake_sbatch = tmp_path / "sbatch"
    call_log = tmp_path / "sbatch_calls.log"
    fake_sbatch.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "${SBATCH_CALL_LOG}"
if [[ "$*" == *"postprocess_changed_geometry_trial.sj"* ]]; then
  printf '9002\n'
else
  printf '9001\n'
fi
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update({"SBATCH_EXE": str(fake_sbatch), "SBATCH_CALL_LOG": str(call_log)})

    completed = subprocess.run(
        ["bash", str(wrapper)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    calls = call_log.read_text(encoding="utf-8").splitlines()

    assert calls[0].endswith("submit_changed_geometry_trial.sj")
    assert "--dependency=afterok:9001" in calls[1]
    assert calls[1].endswith("postprocess_changed_geometry_trial.sj")
    assert "Submitted changed-geometry trial array: 9001" in completed.stdout
    assert "Submitted postprocess job 9002 with dependency afterok:9001" in completed.stdout
    postprocess_text = postprocess.read_text(encoding="utf-8")
    assert "#SBATCH --mail-type=ALL" in postprocess_text
    assert "#SBATCH --mail-user=kevmoor@sandia.gov" in postprocess_text
    assert "export ONLY_POSTPROCESS=1" in postprocess_text
    assert 'export POSTPROCESS_INCLUDE_SMOKE_IN_DEL="${POSTPROCESS_INCLUDE_SMOKE_IN_DEL:-1}"' in postprocess_text
    assert "--stage runtime" in postprocess_text
    assert "--require-fresh-run" in postprocess_text
    assert "--allow-diagnostic-postprocess-artifacts" in postprocess_text
    assert 'identity_audit_flags+=(--require-baseline-identity-proof)' in postprocess_text
    assert "changed_geometry_trial_acceptance.json" in postprocess_text
    assert "trial_diagnostic_results_audit.json" in postprocess_text
    assert "--require-fresh-run" in postprocess_text
    assert "changed_geometry_trial_acceptance.json" in postprocess_text


def test_full_parked_dlc_wrappers_default_replay_off():
    expected = 'QBLADE_STORE_REPLAY="${QBLADE_STORE_REPLAY:-false}"'
    for name in (
        "run_parked_dlc6x_linux_cpu.sh",
        "run_parked_dlc6x_linux_gpu.sh",
    ):
        wrapper = (RUN_HEADLESS / name).read_text(encoding="utf-8")
        assert expected in wrapper
        assert '--wake-profile "${QBLADE_WAKE_PROFILE}"' in wrapper
        assert 'RUN_ARGS+=(--result-filter "${QBLADE_RESULT_FILTER}")' in wrapper

    staged = (RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")
    assert '--store-replay "${QBLADE_STORE_REPLAY:-false}"' in staged
    assert '--wake-profile "${QBLADE_WAKE_PROFILE}"' in staged
    assert 'campaign_cmd+=(--result-filter "${QBLADE_RESULT_FILTER}")' in staged


def test_next_step_campaign_entry_points_default_identity_off():
    submitter = (RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")
    trial = (RUN_HEADLESS / "submit_changed_geometry_trial.sj").read_text(encoding="utf-8")
    trial_postprocess = (RUN_HEADLESS / "postprocess_changed_geometry_trial.sj").read_text(
        encoding="utf-8"
    )
    campaign = (RUN_HEADLESS / "run_parked_demo_campaign.py").read_text(encoding="utf-8")
    audit_script = (RUN_HEADLESS / "audit_parked_campaign.py").read_text(encoding="utf-8")
    postprocess = (RUN_HEADLESS / "postprocess_parked_demo.py").read_text(encoding="utf-8")

    assert 'CAMPAIGN_SCENARIOS="${CAMPAIGN_SCENARIOS:-baseline vorlap}"' in submitter
    assert 'CAMPAIGN_SCENARIOS="${CAMPAIGN_SCENARIOS:-baseline vorlap}"' in trial
    assert 'CAMPAIGN_SCENARIOS="${CAMPAIGN_SCENARIOS:-baseline vorlap}"' in trial_postprocess
    assert 'os.environ.get("CAMPAIGN_SCENARIOS", "baseline vorlap")' in campaign
    assert 'parser.add_argument("--scenarios", default="baseline vorlap")' in audit_script
    assert 'parser.add_argument("--scenarios", default="baseline vorlap")' in postprocess
    assert 'identity_audit_flags+=(--require-baseline-identity-proof)' in trial_postprocess


def test_forced_array_setup_uses_one_leader_and_batch_marker():
    staged = (RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")

    assert 'SERIALIZED_FORCE_SETUP_ROLE="leader"' in staged
    assert 'SERIALIZED_FORCE_SETUP_ROLE="follower"' in staged
    assert "serialized-batch" in staged
    assert "SLURM_ARRAY_TASK_MIN" in staged
    assert "VORLAP_SERIALIZED_SETUP_WAIT_SECONDS" in staged
    assert '"${setup_cmd[@]}" --setup-only' in staged
    assert "disable_forced_setup_flags" in staged
    assert "One-time forced setup completed; releasing the remaining array tasks" in staged
    assert "with all force flags disabled" in staged


def test_forced_array_setup_executes_once_before_followers_run(tmp_path):
    fake_python = tmp_path / "fake-python"
    event_log = tmp_path / "events.log"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$*" == *"--setup-only"* ]]; then
  printf 'setup-start task=%s\\n' "${SLURM_ARRAY_TASK_ID}" >> "${SERIAL_SETUP_EVENT_LOG}"
  printf 'setup-args task=%s args=%s\\n' "${SLURM_ARRAY_TASK_ID}" "$*" >> "${SERIAL_SETUP_EVENT_LOG}"
  sleep 1
  printf 'setup-done task=%s\\n' "${SLURM_ARRAY_TASK_ID}" >> "${SERIAL_SETUP_EVENT_LOG}"
  exit 0
fi
if [[ "$*" == *"--case-index"* ]]; then
  printf 'case task=%s args=%s\\n' "${SLURM_ARRAY_TASK_ID}" "$*" >> "${SERIAL_SETUP_EVENT_LOG}"
  exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_vorlap_root = tmp_path / "vorlap"
    fake_qblade_root = tmp_path / "qblade"
    fake_vorlap_root.mkdir()
    fake_qblade_root.mkdir()
    common_env = os.environ.copy()
    common_env.update(
        {
            "ARRAY_CASE_DRY_RUN": "0",
            "AUTO_SETUP_FIRST_RUN": "0",
            "DRY_RUN_ONLY": "0",
            "FORCE_BRIDGE_BUILD": "1",
            "FORCE_CASE_SETUP": "1",
            "FORCE_GENERATE_TURBSIM": "1",
            "FORCE_OPENFAST_TURBSIM_BUILD": "0",
            "FORCE_PYTHON_SETUP": "0",
            "GENERATE_TURBSIM": "0",
            "PYTHON_EXE": str(fake_python),
            "QBLADE_DURATION": "660",
            "QBLADE_ROOT": str(fake_qblade_root),
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "RUN_ROOT": str(tmp_path / "run"),
            "SERIAL_SETUP_EVENT_LOG": str(event_log),
            "SKIP_QBLADE_ROOT_PREFLIGHT": "1",
            "SKIP_VORLAP_MODULE_LOAD": "1",
            "SLURM_ARRAY_JOB_ID": "batch-once-test",
            "SLURM_ARRAY_TASK_MIN": "0",
            "SLURM_JOB_ID": "9000",
            "VORLAP_FORCE_SETUP_TOKEN": "batch-once-test",
            "VORLAP_ROOT": str(fake_vorlap_root),
            "VORLAP_SERIALIZED_SETUP_WAIT_SECONDS": "20",
            "VORLAP_SERIALIZE_FORCE_SETUP": "1",
            "VORLAP_SUBMIT_SCRIPT": str(RUN_HEADLESS / "submit_parked_demo_campaign.sj"),
        }
    )

    follower_env = common_env.copy()
    follower_env["SLURM_ARRAY_TASK_ID"] = "1"
    leader_env = common_env.copy()
    leader_env["SLURM_ARRAY_TASK_ID"] = "0"
    follower = subprocess.Popen(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        env=follower_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    leader = subprocess.Popen(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        env=leader_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    leader_stdout, leader_stderr = leader.communicate(timeout=20)
    follower_stdout, follower_stderr = follower.communicate(timeout=20)

    assert leader.returncode == 0, leader_stderr
    assert follower.returncode == 0, follower_stderr
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert events.count("setup-start task=0") == 1
    assert events.count("setup-done task=0") == 1
    setup_args = [line for line in events if line.startswith("setup-args task=0")]
    assert len(setup_args) == 1
    assert "--reuse-setup" not in setup_args[0]
    assert "--force-generate-turbsim" in setup_args[0]
    assert not any(line.startswith("setup-") and "task=1" in line for line in events)
    case_args = [line for line in events if line.startswith("case task=")]
    assert len(case_args) == 2
    assert all("--force-generate-turbsim" not in line for line in case_args)
    assert all("--force-bridge-build" not in line for line in case_args)
    assert events.index("setup-done task=0") < next(
        index for index, line in enumerate(events) if line.startswith("case task=0 ")
    )
    assert events.index("setup-done task=0") < next(
        index for index, line in enumerate(events) if line.startswith("case task=1 ")
    )
    assert "one-time forced-setup leader" in leader_stdout
    assert "Forced generated-case refresh requested" in leader_stdout
    assert "waiting for one-time forced setup" in follower_stdout
    assert "will reuse it" in follower_stdout


def test_production_submission_script_submits_one_job_per_wind_speed(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "split_winds"),
            "SUBMIT_WIND_JOBS": "1",
            "SUBMIT_WIND_JOBS_DRY_RUN": "1",
            "SUBMIT_WIND_JOB_TIME": "47:59:00",
            "WIND_SPEEDS": "5 10",
            "VORLAP_SOURCE_FREQUENCY_TRUNCATION": "nyquist",
            "SIM_MINUTES": "1",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Submitting wind speed 5 m/s as U05p00ms" in completed.stdout
    assert "Submitting wind speed 10 m/s as U10p00ms" in completed.stdout
    assert completed.stdout.count("sbatch --job-name") == 2
    assert completed.stdout.count("--time 47:59:00") == 2
    assert "WIND_SPEEDS=5" in completed.stdout
    assert "WIND_SPEEDS=10" in completed.stdout
    assert "QBLADE_DURATION=60" in completed.stdout
    assert "VORLAP_SUBMIT_SCRIPT=" in completed.stdout
    assert "SKIP_VORLAP_INSTALL=1" in completed.stdout
    assert "VORLAP_SOURCE_BOUNDS_POLICY=warn" in completed.stdout
    assert "VORLAP_SOURCE_FREQUENCY_TRUNCATION=nyquist" in completed.stdout
    assert "VORLAP_AUTO_NODE_THREADS=1" in completed.stdout
    assert str(RUN_HEADLESS / "submit_parked_demo_campaign.sj") in completed.stdout
    assert str(tmp_path / "split_winds" / "U05p00ms") in completed.stdout
    assert str(tmp_path / "split_winds" / "U10p00ms") in completed.stdout


def test_production_submission_script_loads_configured_slurm_module_for_split_jobs(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    module_log = tmp_path / "module.log"
    module_exe = bin_dir / "module"
    module_exe.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"${VORLAP_MODULE_LOG}\"\n",
        encoding="utf-8",
    )
    module_exe.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
            "RUN_ROOT": str(tmp_path / "split_winds_modules"),
            "SUBMIT_WIND_JOBS": "1",
            "SUBMIT_WIND_JOBS_DRY_RUN": "1",
            "VORLAP_MODULE_LOG": str(module_log),
            "VORLAP_SLURM_MODULES": "anaconda3/2025.12-2",
            "WIND_SPEEDS": "5",
            "SIM_MINUTES": "1",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Loading module anaconda3/2025.12-2." in completed.stdout
    assert module_log.read_text(encoding="utf-8").strip() == "load anaconda3/2025.12-2"
    assert "sbatch --job-name" in completed.stdout


def test_production_submission_script_skips_vorlap_install_by_default(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_skip_install_default"),
            "DRY_RUN_ONLY": "1",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "QBLADE_DURATION": "60",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    payload = first_json_object(completed.stdout)

    for command in payload["scenario_setup_commands"]:
        assert "--skip-vorlap-install" in command
        assert command[command.index("--source-bounds-policy") + 1] == "warn"


def test_production_submission_script_can_force_vorlap_install(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_force_install"),
            "DRY_RUN_ONLY": "1",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "SKIP_VORLAP_INSTALL": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "QBLADE_DURATION": "60",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    payload = first_json_object(completed.stdout)

    for command in payload["scenario_setup_commands"]:
        assert "--skip-vorlap-install" not in command
        assert command[command.index("--source-bounds-policy") + 1] == "warn"


def test_production_submission_script_preflights_missing_qblade_before_sbatch(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "split_winds"),
            "SUBMIT_WIND_JOBS": "1",
            "WIND_SPEEDS": "50",
            "SIM_MINUTES": "1",
            "PYTHON_EXE": sys.executable,
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "QBlade root is incomplete" in completed.stderr
    assert "must contain the complete QBlade CE runtime" in completed.stderr
    assert "rsync -a /path/to/working/QBladeCE_2.0.9.7/" in completed.stderr
    assert "sbatch --job-name" not in completed.stdout


def test_production_submission_script_resolves_real_path_from_slurm_spool_copy(tmp_path):
    spool_dir = tmp_path / "spool" / "job123"
    spool_dir.mkdir(parents=True)
    spool_script = spool_dir / "slurm_script"
    spool_script.write_text((RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8"), encoding="utf-8")
    spool_script.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "spool_submit_dryrun"),
            "DRY_RUN_ONLY": "1",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "QBLADE_DURATION": "60",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
            "SLURM_SUBMIT_DIR": str(RUN_HEADLESS.parent),
        }
    )

    completed = subprocess.run(
        ["bash", str(spool_script)],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.stderr == ""
    assert str(spool_dir) not in completed.stdout
    settings = json.loads((tmp_path / "spool_submit_dryrun" / "campaign_settings.json").read_text(encoding="utf-8"))
    assert settings["vorlap_root"] == str(RUN_HEADLESS.parent)
    assert settings["planned_case_count_total"] == 2


def test_production_submission_script_rejects_step_override_when_production_required(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_step_limited_production"),
            "DRY_RUN_ONLY": "1",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "QBLADE_DURATION": "600",
            "QBLADE_TIMESTEP": "0.0075",
            "QBLADE_STEPS": "2",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        env=env,
        text=True,
        capture_output=True,
    )
    plan_audit = json.loads(
        (tmp_path / "submit_step_limited_production" / "audits" / "plan_audit.json").read_text(encoding="utf-8")
    )
    codes = {issue["code"] for issue in plan_audit["issues"]}

    assert completed.returncode != 0
    assert plan_audit["passed"] is False
    assert "qblade_steps_override_in_production_plan" in codes


def test_production_submission_script_allows_step_override_for_debug_matrix(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_step_limited_debug"),
            "DRY_RUN_ONLY": "1",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "QBLADE_DURATION": "600",
            "QBLADE_TIMESTEP": "0.0075",
            "QBLADE_STEPS": "2",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    settings = json.loads((tmp_path / "submit_step_limited_debug" / "campaign_settings.json").read_text(encoding="utf-8"))
    plan_audit = json.loads((tmp_path / "submit_step_limited_debug" / "audits" / "plan_audit.json").read_text(encoding="utf-8"))

    assert completed.stderr == ""
    assert settings["qblade_steps_override"] == 2
    assert plan_audit["passed"] is True


def test_production_submission_script_rejects_narrow_matrix_when_production_required(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_narrow_matrix"),
            "DRY_RUN_ONLY": "1",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "QBLADE_DURATION": "600",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        env=env,
        text=True,
        capture_output=True,
    )
    plan_audit = json.loads((tmp_path / "submit_narrow_matrix" / "audits" / "plan_audit.json").read_text(encoding="utf-8"))
    codes = {issue["code"] for issue in plan_audit["issues"]}

    assert completed.returncode != 0
    assert plan_audit["passed"] is False
    assert "required_wind_speed_grid_mismatch" in codes
    assert "required_azimuth_grid_mismatch" not in codes


def test_production_submission_script_requires_allowables_for_required_ultimate_margins(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_missing_allowables"),
            "DRY_RUN_ONLY": "1",
            "REQUIRE_ULTIMATE_MARGINS": "1",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "QBLADE_DURATION": "600",
            "QBLADE_TIMESTEP": "0.0075",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert "REQUIRE_ULTIMATE_MARGINS=1 requires ULTIMATE_ALLOWABLES" in completed.stderr
    assert not (tmp_path / "submit_missing_allowables" / "campaign_settings.json").exists()


def test_production_submission_script_accepts_archived_allowables_for_postprocess_rerun(tmp_path):
    run_root = tmp_path / "submit_archived_allowables"
    allowables = run_root / "postprocess" / "ultimate_allowables.csv"
    allowables.parent.mkdir(parents=True, exist_ok=True)
    allowables.write_text(
        "blade,position,quantity,axis,unit,allowable_abs,allowable_label,load_factor,resistance_factor,notes\n"
        "BLD_1,0.5,Force,X,N,1000,test,1.0,1.0,synthetic\n",
        encoding="utf-8",
    )
    write_json(run_root / "campaign_settings.json", {"ultimate_allowables": str(allowables)})
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(run_root),
            "ONLY_POSTPROCESS": "1",
            "SKIP_POSTPROCESS": "1",
            "RUN_FINAL_AUDIT": "0",
            "REQUIRE_ULTIMATE_MARGINS": "1",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.stderr == ""


def test_production_submission_script_exposes_ultimate_input_basis_audit_flag():
    script = (RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")

    assert "REQUIRE_ULTIMATE_INPUT_BASIS" in script
    assert "--require-ultimate-input-basis" in script


def test_production_submission_script_exposes_spectral_psd_comparison_audit_flag():
    script = (RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")

    assert "REQUIRE_SPECTRAL_PSD_COMPARISON" in script
    assert "--require-spectral-psd-comparison" in script
    assert "GENERATE_SPECTRAL_PSD_COMPARISON" in script
    assert "build_spectral_psd_comparison.py" in script


def test_production_submission_script_exposes_final_evidence_audit_flags():
    script = (RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")

    expected = {
        "REQUIRE_GEOMETRY_POSITION_PROOF": "--require-geometry-position-proof",
        "REQUIRE_NATIVE_CALLBACK_PROOF": "--require-native-callback-proof",
        "REQUIRE_PRODUCTION_ARCHIVE_MANIFEST": "--require-production-archive-manifest",
        "REQUIRE_SPECTRAL_FORCE_CONTENT": "--require-spectral-force-content",
        "REQUIRE_STRUCTURAL_MODEL_QUALIFICATION": "--require-structural-model-qualification",
        "REQUIRE_SIGNED_CHANNEL_SELECTION": "--require-signed-channel-selection",
    }
    for env_var, flag in expected.items():
        assert env_var in script
        assert flag in script
    assert "GENERATE_PRODUCTION_ARCHIVE_MANIFEST" in script
    assert "build_production_archive_manifest.py" in script
    assert "GENERATE_SIGNED_CHANNEL_SELECTION" in script
    assert "build_signed_channel_selection.py" in script
    assert "GENERATE_ULTIMATE_INPUT_BASIS" in script
    assert "build_ultimate_input_basis.py" in script
    assert "GENERATE_STRUCTURAL_MODEL_QUALIFICATION" in script
    assert "build_structural_model_qualification.py" in script
    assert "GENERATE_SPECTRAL_PSD_COMPARISON" in script
    assert "build_spectral_psd_comparison.py" in script


def test_production_submission_script_exposes_paper_submission_gates():
    script = (RUN_HEADLESS / "submit_parked_demo_campaign.sj").read_text(encoding="utf-8")

    assert "PAPER_CHECKS_ONLY" in script
    assert "RUN_PAPER_PREFLIGHT" in script
    assert "REQUIRE_PAPER_EVIDENCE_MATRIX_READY" in script
    assert "REQUIRE_PLACEHOLDER_FREE_SUBMISSION" in script
    assert "check_evidence_matrix.py" in script
    assert "check_submission_ready.py" in script
    assert "--require-submission-ready" in script


@pytest.mark.paper
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir(),
    reason="Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
)
def test_production_submission_script_paper_checks_only_runs_matrix_preflight():
    env = os.environ.copy()
    env.update(
        {
            "PAPER_CHECKS_ONLY": "1",
            "RUN_PAPER_PREFLIGHT": "1",
            "REQUIRE_PAPER_EVIDENCE_MATRIX_READY": "0",
            "REQUIRE_PLACEHOLDER_FREE_SUBMISSION": "0",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Evidence matrix check passed" in completed.stdout
    assert completed.stderr == ""


@pytest.mark.paper
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir(),
    reason="Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
)
def test_production_submission_script_rejects_external_release_blocker():
    env = os.environ.copy()
    env.update(
        {
            "PAPER_CHECKS_ONLY": "1",
            "RUN_PAPER_PREFLIGHT": "0",
            "REQUIRE_PAPER_EVIDENCE_MATRIX_READY": "1",
            "REQUIRE_PLACEHOLDER_FREE_SUBMISSION": "0",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "Evidence matrix check failed" in completed.stderr
    assert "archival_and_human_release" in completed.stderr
    assert "submission_blocked" in completed.stderr


@pytest.mark.paper
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir(),
    reason="Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
)
def test_production_submission_script_passes_placeholder_free_submission_gate():
    env = os.environ.copy()
    env.update(
        {
            "PAPER_CHECKS_ONLY": "1",
            "RUN_PAPER_PREFLIGHT": "0",
            "REQUIRE_PAPER_EVIDENCE_MATRIX_READY": "0",
            "REQUIRE_PLACEHOLDER_FREE_SUBMISSION": "1",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "Submission placeholder check passed" in completed.stdout


def test_production_submission_script_dry_run_can_reuse_setup(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(tmp_path / "submit_dryrun_reuse"),
            "DRY_RUN_ONLY": "1",
            "REUSE_SETUP": "1",
            "REQUIRE_PRODUCTION_MATRIX": "0",
            "QBLADE_ROOT": str(tmp_path / "missing_qblade"),
            "WIND_SPEEDS": "5",
            "QBLADE_AZIMUTHS": "0",
            "QBLADE_DURATION": "600",
            "QBLADE_TIMESTEP": "0.0075",
            "POSTPROCESS_LOCATIONS": "0.111 0.222",
            "POSTPROCESS_BLADE": "BLD_2",
            "WOHLER_EXPONENT": "8",
            "PYTHON_EXE": sys.executable,
        }
    )

    completed = subprocess.run(
        ["bash", str(RUN_HEADLESS / "submit_parked_demo_campaign.sj")],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert '"stage": "plan"' in completed.stdout
    assert '"passed": true' in completed.stdout
    assert "--force-setup" not in completed.stdout
    settings = json.loads((tmp_path / "submit_dryrun_reuse" / "campaign_settings.json").read_text(encoding="utf-8"))
    assert settings["reuse_setup"] is True
    assert settings["postprocess_locations"] == "0.111 0.222"
    assert settings["postprocess_blade"] == "BLD_2"
    assert settings["wohler_exponent"] == 8.0
    assert (tmp_path / "submit_dryrun_reuse" / "audits" / "plan_audit.json").is_file()


def test_native_callback_evidence_parses_cpp_bridge_log(tmp_path):
    sim = tmp_path / "case" / "demo.sim"
    trb = tmp_path / "case" / "demo.trb"
    config = tmp_path / "case" / "Control" / "vorlap_qblade_external.json"
    log = config.parent / "vorlap_qblade_bridge_cpp.log"
    config.parent.mkdir(parents=True)
    sim.write_text("demo.trb TURBFILE\n", encoding="utf-8")
    trb.write_text(
        "libvorlap_qblade_bridge LIBFILE_1\n"
        "update LIBFUNCTION_1\n"
        "15 LIBARRAYSIZE_1\n"
        "Control/vorlap_qblade_external.json LIBPARAMETERFILE_1\n",
        encoding="utf-8",
    )
    config.write_text("{}\n", encoding="utf-8")
    log.write_text(
        "\n".join(
            [
                "update_init entered; paramFile=Control/vorlap_qblade_external.json",
                "python_env_before_import="
                + json.dumps(
                    {
                        "python_path_entries": [str(tmp_path)],
                        "python_path_entries_added": [str(tmp_path)],
                        "python_version": "3.test",
                        "sys_executable": sys.executable,
                        "sys_path": [str(tmp_path)],
                    },
                    sort_keys=True,
                ),
                "python_env_after_import="
                + json.dumps(
                    {
                        "numpy_file": "/tmp/site-packages/numpy/__init__.py",
                        "numpy_version": "2.0.test",
                        "python_version": "3.test",
                        "sys_executable": sys.executable,
                        "sys_path": [str(tmp_path)],
                        "vorlap_file": "/tmp/vorlap/__init__.py",
                        "vorlap_qblade_runtime_file": "/tmp/vorlap/qblade_runtime.py",
                    },
                    sort_keys=True,
                ),
                "MESSAGE: VorLap QBlade runtime initialized.",
                "update entered; count=1",
                "MESSAGE: VorLap dbg t=0.015s az=270deg max|F|=1N cache=1/2",
                "update entered; count=2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = qblade_runner.native_callback_evidence(sim)

    assert evidence["callback_block_present"] is True
    assert evidence["turbine_file_resolved"] == str(trb)
    assert evidence["library_stem"] == "libvorlap_qblade_bridge"
    assert evidence["function_name"] == "update"
    assert evidence["array_size"] == "15"
    assert evidence["parameter_file_from_turbine"] == "Control/vorlap_qblade_external.json"
    assert evidence["parameter_file_resolved"] == str(config)
    assert evidence["parameter_file_exists"] is True
    assert evidence["log_file_exists"] is True
    assert evidence["update_init_seen"] is True
    assert evidence["update_init_count"] == 1
    assert evidence["update_log_entry_count"] == 2
    assert evidence["update_cache_observation_count"] == 1
    assert evidence["update_count_observed_max"] == 3
    assert evidence["runtime_initialized_message_seen"] is True
    assert evidence["python_env_before_import_seen"] is True
    assert evidence["python_env_after_import_seen"] is True
    assert evidence["python_path_entries_added"] == [str(tmp_path)]
    assert evidence["numpy_version"] == "2.0.test"
    assert evidence["numpy_file"] == "/tmp/site-packages/numpy/__init__.py"
    assert evidence["vorlap_file"] == "/tmp/vorlap/__init__.py"
    assert evidence["vorlap_qblade_runtime_file"] == "/tmp/vorlap/qblade_runtime.py"
    assert evidence["python_env_parse_errors"] == []
    assert evidence["log_file_hash"]["sha256"]


def test_native_update_log_lower_bound_matches_cpp_logging_cadence():
    expected_by_completed_steps = {
        0: 0,
        1: 1,
        10: 10,
        11: 10,
        42: 10,
        99: 10,
        100: 100,
        101: 100,
        80000: 80000,
        80005: 80000,
    }

    observed = {
        completed_steps: audit.native_update_log_lower_bound(completed_steps)
        for completed_steps in expected_by_completed_steps
    }

    assert observed == expected_by_completed_steps


def test_native_callback_audit_requires_latest_sampled_update_count():
    evidence = {
        "update_init_seen": True,
        "update_count_observed_max": 10,
        "runtime_initialized_message_seen": True,
        "python_env_parse_errors": [],
        "python_env_before_import_seen": True,
        "python_env_after_import_seen": True,
        "numpy_version": "2.0.test",
        "numpy_file": "/tmp/site-packages/numpy/__init__.py",
        "vorlap_qblade_runtime_file": "/tmp/vorlap/qblade_runtime.py",
    }
    issues = []

    audit.audit_native_callback_evidence(evidence, issues, scenario="vorlap", case_label="case", completed_steps=80000)

    errors = {issue.code for issue in issues if issue.severity == "error"}
    assert errors == {"native_update_count_missing"}

    evidence["update_count_observed_max"] = 80000
    issues = []
    audit.audit_native_callback_evidence(evidence, issues, scenario="vorlap", case_label="case", completed_steps=80000)

    assert not [issue for issue in issues if issue.severity == "error"]


def test_native_callback_audit_reports_missing_callback_block():
    evidence = {
        "callback_block_present": False,
        "update_init_seen": True,
        "update_count_observed_max": 1,
        "runtime_initialized_message_seen": True,
        "python_env_parse_errors": [],
        "python_env_before_import_seen": True,
        "python_env_after_import_seen": True,
        "numpy_version": "2.0.test",
        "numpy_file": "/tmp/site-packages/numpy/__init__.py",
        "vorlap_qblade_runtime_file": "/tmp/vorlap/qblade_runtime.py",
    }
    issues = []

    audit.audit_native_callback_evidence(evidence, issues, scenario="vorlap", case_label="case", completed_steps=1)

    errors = {issue.code for issue in issues if issue.severity == "error"}
    assert "native_callback_block_missing" in errors


def test_native_callback_audit_reports_missing_configured_library():
    evidence = {
        "callback_block_present": True,
        "library_stem": "libvorlap_qblade_bridge",
        "library_file_exists": False,
        "library_candidate_names": ["libvorlap_qblade_bridge.so"],
        "library_search_roots": ["/tmp/QBladeCE/ControllerFiles"],
        "update_init_seen": True,
        "update_count_observed_max": 1,
        "runtime_initialized_message_seen": True,
        "python_env_parse_errors": [],
        "python_env_before_import_seen": True,
        "python_env_after_import_seen": True,
        "numpy_version": "2.0.test",
        "numpy_file": "/tmp/site-packages/numpy/__init__.py",
        "vorlap_qblade_runtime_file": "/tmp/vorlap/qblade_runtime.py",
    }
    issues = []

    audit.audit_native_callback_evidence(evidence, issues, scenario="vorlap", case_label="case", completed_steps=1)

    errors = {issue.code for issue in issues if issue.severity == "error"}
    assert errors == {"native_library_missing"}


def test_native_callback_audit_reports_missing_required_library_symbols():
    evidence = {
        "callback_block_present": True,
        "library_stem": "libvorlap_qblade_bridge",
        "library_file_exists": True,
        "function_name": "update",
        "library_symbol_check_available": True,
        "library_missing_required_symbols": ["update_message"],
        "update_init_seen": True,
        "update_count_observed_max": 1,
        "runtime_initialized_message_seen": True,
        "python_env_parse_errors": [],
        "python_env_before_import_seen": True,
        "python_env_after_import_seen": True,
        "numpy_version": "2.0.test",
        "numpy_file": "/tmp/site-packages/numpy/__init__.py",
        "vorlap_qblade_runtime_file": "/tmp/vorlap/qblade_runtime.py",
    }
    issues = []

    audit.audit_native_callback_evidence(evidence, issues, scenario="vorlap", case_label="case", completed_steps=1)

    errors = {issue.code for issue in issues if issue.severity == "error"}
    assert errors == {"native_library_missing_required_symbols"}


def complete_native_callback_evidence(completed_steps: int = 25) -> dict:
    return {
        "callback_block_present": True,
        "library_stem": "libvorlap_qblade_bridge",
        "library_file_exists": True,
        "function_name": "update",
        "library_symbol_check_available": True,
        "library_missing_required_symbols": [],
        "update_init_seen": True,
        "update_count_observed_max": completed_steps,
        "runtime_initialized_message_seen": True,
        "python_env_parse_errors": [],
        "python_env_before_import_seen": True,
        "python_env_after_import_seen": True,
        "numpy_version": "2.0.test",
        "numpy_file": "/tmp/site-packages/numpy/__init__.py",
        "vorlap_qblade_runtime_file": "/tmp/vorlap/qblade_runtime.py",
    }


def test_native_callback_proof_gate_requires_qblade_external_case(tmp_path):
    settings, manifests, args, _source_path = source_coverage_audit_fixture(tmp_path)
    result_dir = Path(manifests["vorlap"]["cases"][0]["result_dir"])
    summary_path = result_dir / "baseline_reference_run_summary.json"
    args.require_native_callback_proof = True
    args.native_callback_proof_scenarios = "vorlap"
    write_json(
        summary_path,
        {
            "completed_steps": 25,
            "coupling_mode": "sil-vorlap",
        },
    )
    issues: list[audit.AuditIssue] = []

    report = audit.audit_native_callback_proof(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["checked_cases"] == 1
    assert report["qblade_external_cases"] == 0
    assert report["passing_cases"] == 0
    assert "native_callback_proof_requires_qblade_external" in errors
    assert "native_callback_proof_no_qblade_external_cases" in errors
    assert "native_callback_proof_no_passing_cases" in errors

    write_json(
        summary_path,
        {
            "completed_steps": 25,
            "coupling_mode": "qblade-external",
            "native_callback_evidence": complete_native_callback_evidence(25),
        },
    )
    issues = []
    report = audit.audit_native_callback_proof(settings, manifests, args, issues)

    assert report["checked_cases"] == 1
    assert report["qblade_external_cases"] == 1
    assert report["passing_cases"] == 1
    assert report["cases"][0]["passed"] is True
    assert not [issue for issue in issues if issue.severity == "error"]


def write_run_summary_for_structural_qualification(result_dir: Path, *, timestep: float = 0.0075) -> Path:
    path = result_dir / "case_run_summary.json"
    write_json(
        path,
        {
            "reproducibility": {
                "qblade_case_settings": {
                    "TIMESTEP": f"{timestep:g}",
                    "RAMPUP": "0",
                    "ADDDAMP": "0",
                }
            }
        },
    )
    return path


def test_structural_model_qualification_accepts_complete_manifest(tmp_path):
    run_root = tmp_path / "run"
    case_dir = run_root / "vorlap" / "cases" / "U05p00ms"
    result_dir = run_root / "vorlap" / "results" / "U05p00ms"
    structural_file = case_dir / "reference_model" / "Structure" / "reference_main.str"
    baseline_identity = run_root / "postprocess" / "baseline_identity_comparison.csv"
    structural_file.parent.mkdir(parents=True)
    baseline_identity.parent.mkdir(parents=True)
    structural_file.write_text("0.0200 RAYLEIGHDMP\n", encoding="utf-8")
    baseline_identity.write_text("case,force_delta_pct\nU05p00ms,0\n", encoding="utf-8")
    patch_manifest = write_stable_support_patch(case_dir)
    write_run_summary_for_structural_qualification(result_dir)
    qualification = run_root / "structural_model_qualification.json"
    write_json(
        qualification,
        {
            "model_id": "generic-hvawt-demo",
            "model_choice": "generated_patched_demonstration",
            "qualified": True,
            "modal_frequencies_hz": [0.42, {"mode": 2, "frequency_hz": 1.7}],
            "damping": {"source": "QBlade structural files", "assumption": "documented Rayleigh damping rows"},
            "solver_settings": {"TIMESTEP": 0.0075, "RAMPUP": 0.0, "ADDDAMP": 0.0},
            "structural_files": [
                {
                    "path": str(structural_file.relative_to(run_root)),
                    "sha256": audit.sha256_file(structural_file),
                }
            ],
            "patch_manifests": [{"path": str(patch_manifest.relative_to(run_root))}],
            "baseline_identity_evidence": {
                "status": "passed",
                "artifact": str(baseline_identity.relative_to(run_root)),
            },
        },
    )
    args = SimpleNamespace(
        run_root=run_root,
        structural_model_qualification=None,
        require_structural_model_qualification=True,
    )
    settings = {"timestep_s": 0.0075, "rampup_s": 0.0, "add_damp_s": 0.0}
    manifests = {"vorlap": {"cases": [{"label": "U05p00ms", "result_dir": str(result_dir)}]}}
    issues: list = []

    report = audit.audit_structural_model_qualification(settings, manifests, args, issues)

    assert report == {
        "path": str(qualification),
        "available": True,
        "model_id": "generic-hvawt-demo",
        "model_choice": "generated_patched_demonstration",
        "modal_frequency_count": 2,
        "structural_file_count": 1,
        "patch_manifest_count": 1,
        "baseline_identity_artifact": str(baseline_identity),
    }
    assert not [issue for issue in issues if issue.severity == "error"]


def test_structural_model_qualification_builder_drafts_reviewed_manifest(tmp_path):
    run_root = tmp_path / "run"
    case_dir = run_root / "vorlap" / "cases" / "U05p00ms"
    result_dir = run_root / "vorlap" / "results" / "U05p00ms"
    structural_file = case_dir / "reference_model" / "Structure" / "reference_main.str"
    baseline_identity = run_root / "postprocess" / "baseline_identity_comparison.csv"
    structural_file.parent.mkdir(parents=True)
    baseline_identity.parent.mkdir(parents=True)
    structural_file.write_text("0.0200 RAYLEIGHDMP\n", encoding="utf-8")
    baseline_identity.write_text("case,force_delta_pct\nU05p00ms,0\n", encoding="utf-8")
    write_stable_support_patch(case_dir)
    write_run_summary_for_structural_qualification(result_dir)
    write_json(run_root / "campaign_settings.json", {"timestep_s": 0.0075, "rampup_s": 0.0, "add_damp_s": 0.0})

    manifest = structural_qualification_builder.build_manifest(
        run_root,
        scenario="vorlap",
        model_id="generated-structural-model",
        modal_frequencies_hz=[0.42, 1.7],
        qualified=True,
        baseline_identity_passed=True,
    )
    write_json(run_root / "structural_model_qualification.json", manifest)
    args = SimpleNamespace(
        run_root=run_root,
        structural_model_qualification=None,
        require_structural_model_qualification=True,
    )
    settings = {"timestep_s": 0.0075, "rampup_s": 0.0, "add_damp_s": 0.0}
    manifests = {"vorlap": {"cases": [{"label": "U05p00ms", "result_dir": str(result_dir)}]}}
    issues: list[audit.AuditIssue] = []

    report = audit.audit_structural_model_qualification(settings, manifests, args, issues)

    assert manifest["model_id"] == "generated-structural-model"
    assert manifest["qualified"] is True
    assert manifest["review_status"] == "draft_generated_from_run_archive"
    assert manifest["baseline_identity_evidence"] == {
        "status": "passed",
        "artifact": "postprocess/baseline_identity_comparison.csv",
    }
    assert manifest["solver_settings"]["TIMESTEP"] == 0.0075
    assert manifest["structural_files"][0]["path"] == "vorlap/cases/U05p00ms/reference_model/Structure/reference_main.str"
    assert manifest["patch_manifests"][0]["path"] == "vorlap/cases/U05p00ms/reference_model/Control/vorlap_stable_support_patch.json"
    assert report["model_id"] == "generated-structural-model"
    assert report["modal_frequency_count"] == 2
    assert report["structural_file_count"] == 1
    assert report["patch_manifest_count"] == 1
    assert not [issue for issue in issues if issue.severity == "error"]


def test_structural_model_qualification_rejects_empty_baseline_identity_artifact(tmp_path):
    run_root = tmp_path / "run"
    case_dir = run_root / "vorlap" / "cases" / "U05p00ms"
    result_dir = run_root / "vorlap" / "results" / "U05p00ms"
    structural_file = case_dir / "reference_model" / "Structure" / "reference_main.str"
    baseline_identity = run_root / "postprocess" / "baseline_identity_comparison.csv"
    structural_file.parent.mkdir(parents=True)
    baseline_identity.parent.mkdir(parents=True)
    structural_file.write_text("0.0200 RAYLEIGHDMP\n", encoding="utf-8")
    with baseline_identity.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=("case", "force_identity_delta_pct")).writeheader()
    patch_manifest = write_stable_support_patch(case_dir)
    write_run_summary_for_structural_qualification(result_dir)
    write_json(
        run_root / "structural_model_qualification.json",
        {
            "model_id": "generated-structural-model",
            "model_choice": "generated_patched_demonstration",
            "qualified": True,
            "modal_frequencies_hz": [0.42, 1.7],
            "damping": {"assumption": "reviewed Rayleigh damping rows"},
            "solver_settings": {"TIMESTEP": 0.0075, "RAMPUP": 0.0, "ADDDAMP": 0.0},
            "structural_files": [
                {
                    "path": str(structural_file.relative_to(run_root)),
                    "sha256": audit.sha256_file(structural_file),
                }
            ],
            "patch_manifests": [{"path": str(patch_manifest.relative_to(run_root))}],
            "baseline_identity_evidence": {
                "status": "passed",
                "artifact": str(baseline_identity.relative_to(run_root)),
            },
        },
    )
    args = SimpleNamespace(
        run_root=run_root,
        structural_model_qualification=None,
        require_structural_model_qualification=True,
    )
    settings = {"timestep_s": 0.0075, "rampup_s": 0.0, "add_damp_s": 0.0}
    manifests = {"vorlap": {"cases": [{"label": "U05p00ms", "result_dir": str(result_dir)}]}}
    issues: list[audit.AuditIssue] = []

    report = audit.audit_structural_model_qualification(settings, manifests, args, issues)
    errors = [issue.code for issue in issues if issue.severity == "error"]

    assert report["baseline_identity_artifact"] == str(baseline_identity)
    assert errors == ["structural_qualification_empty_baseline_identity_artifact"]


def test_structural_model_qualification_rejects_missing_and_inconsistent_evidence(tmp_path):
    run_root = tmp_path / "run"
    result_dir = run_root / "vorlap" / "results" / "U05p00ms"
    write_run_summary_for_structural_qualification(result_dir, timestep=0.015)
    qualification = run_root / "structural_model_qualification.json"
    write_json(
        qualification,
        {
            "model_id": "",
            "model_choice": "generated_patched_demonstration",
            "qualified": False,
            "modal_frequencies_hz": [-1.0],
            "damping": {},
            "solver_settings": {"TIMESTEP": 0.0075},
            "structural_files": [{"path": "missing/reference_main.str", "sha256": "not-real"}],
            "patch_manifests": [],
            "baseline_identity_evidence": {"status": "pending", "artifact": "postprocess/missing.csv"},
        },
    )
    args = SimpleNamespace(
        run_root=run_root,
        structural_model_qualification=None,
        require_structural_model_qualification=True,
    )
    settings = {"timestep_s": 0.0075, "rampup_s": 0.0, "add_damp_s": 0.0}
    manifests = {"vorlap": {"cases": [{"label": "U05p00ms", "result_dir": str(result_dir)}]}}
    issues: list = []

    audit.audit_structural_model_qualification(settings, manifests, args, issues)

    errors = {issue.code for issue in issues if issue.severity == "error"}
    assert {
        "structural_qualification_missing_model_id",
        "structural_qualification_not_marked_qualified",
        "structural_qualification_invalid_modal_frequency",
        "structural_qualification_missing_damping",
        "missing_structural_qualification_structural_file",
        "missing_structural_qualification_patch_manifest",
        "structural_qualification_baseline_identity_not_passed",
        "structural_qualification_missing_baseline_identity_artifact",
        "structural_qualification_case_setting_mismatch",
    } <= errors


def test_required_structural_model_qualification_reports_missing_manifest(tmp_path):
    args = SimpleNamespace(
        run_root=tmp_path,
        structural_model_qualification=None,
        require_structural_model_qualification=True,
    )
    issues: list = []

    report = audit.audit_structural_model_qualification({}, {}, args, issues)

    assert report == {"path": str(tmp_path / "structural_model_qualification.json"), "available": False}
    assert [issue.code for issue in issues if issue.severity == "error"] == [
        "missing_structural_model_qualification"
    ]


def write_signed_selection_artifacts(run_root: Path) -> tuple[Path, Path]:
    post_dir = run_root / "postprocess"
    post_dir.mkdir(parents=True, exist_ok=True)
    signed_path = post_dir / "signed_del_comparison.csv"
    critical_path = post_dir / "critical_case_summary.csv"
    signed_row = {
        "case": "U05p00ms",
        "wind_speed_mps": "5",
        "azimuth_deg": "270",
        "position": "0.769545",
        "quantity": "moment",
        "axis": "X",
        "unit": "Nm",
        "channel_baseline": "X_l Mom. BLD_1 pos 0.770 [Nm]",
        "channel_vorlap": "X_l Mom. BLD_1 pos 0.770 [Nm]",
        "signed_del_baseline": "10",
        "signed_del_vorlap": "15",
        "signed_delta_pct": "50",
        "rainflow_cycles_baseline": "4",
        "rainflow_cycles_vorlap": "5",
        "smoke_only": "false",
        "postprocess_warnings": "",
    }
    with signed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_del_comparison.csv"])
        writer.writeheader()
        writer.writerow(signed_row)

    critical_row = {
        **{column: signed_row.get(column, "") for column in audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/critical_case_summary.csv"]},
        "signed_del_ratio": "1.5",
        "selection_reason": "max_signed_del_ratio",
        "min_nyquist_ratio": "2",
        "max_force_coefficient_ratio": "0.8",
        "force_power_max_abs_w": "10",
        "source_coverage_violation_count": "0",
    }
    with critical_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/critical_case_summary.csv"])
        writer.writeheader()
        writer.writerow(critical_row)
    return signed_path, critical_path


def test_signed_channel_selection_accepts_complete_manifest(tmp_path):
    signed_path, critical_path = write_signed_selection_artifacts(tmp_path)
    selection_path = tmp_path / "signed_channel_selection.json"
    write_json(
        selection_path,
        {
            "selection_id": "final-signed-del-v1",
            "normalization": {
                "metric": "signed_del_ratio",
                "baseline": "no_sil_baseline",
                "comparison": "vorlap_over_baseline",
                "wohler_exponent": 10.0,
                "reference_cycle_convention": "rainflow cycles from retained 600 s record",
                "transient_removal": "none after STOREFROM=60 s",
                "mean_stress_assumption": "not applied to load-channel DEL ratios",
            },
            "artifacts": {
                "signed_del_comparison": str(signed_path.relative_to(tmp_path)),
                "critical_case_summary": str(critical_path.relative_to(tmp_path)),
            },
            "selected_channels": [
                {
                    "case": "U05p00ms",
                    "position": 0.769545,
                    "quantity": "moment",
                    "axis": "X",
                    "unit": "Nm",
                    "channel_vorlap": "X_l Mom. BLD_1 pos 0.770 [Nm]",
                    "critical": True,
                }
            ],
        },
    )
    args = SimpleNamespace(run_root=tmp_path, signed_channel_selection=None, require_signed_channel_selection=True)
    issues: list = []

    report = audit.audit_signed_channel_selection({"wohler_exponent": 10.0}, args, issues)

    assert report == {
        "path": str(selection_path),
        "available": True,
        "selection_id": "final-signed-del-v1",
        "selected_channels": 1,
        "matched_channels": 1,
        "critical_selected": 1,
        "signed_del_comparison": str(signed_path),
        "critical_case_summary": str(critical_path),
    }
    assert not [issue for issue in issues if issue.severity == "error"]


def test_signed_channel_selection_builder_selects_critical_channel(tmp_path):
    signed_path, critical_path = write_signed_selection_artifacts(tmp_path)
    write_json(tmp_path / "campaign_settings.json", {"wohler_exponent": 10.0})

    manifest = signed_selection_builder.build_manifest(tmp_path, selection_id="generated-selection")
    write_json(tmp_path / "signed_channel_selection.json", manifest)
    args = SimpleNamespace(run_root=tmp_path, signed_channel_selection=None, require_signed_channel_selection=True)
    issues: list[audit.AuditIssue] = []

    report = audit.audit_signed_channel_selection({"wohler_exponent": 10.0}, args, issues)

    assert manifest["selection_id"] == "generated-selection"
    assert manifest["review_status"] == "draft_generated_from_postprocess"
    assert manifest["artifacts"] == {
        "signed_del_comparison": "postprocess/signed_del_comparison.csv",
        "critical_case_summary": "postprocess/critical_case_summary.csv",
    }
    assert manifest["reports_stress_or_reserve_factors"] is False
    assert manifest["selected_channels"] == [
        {
            "case": "U05p00ms",
            "position": 0.769545,
            "quantity": "moment",
            "axis": "X",
            "unit": "Nm",
            "channel_vorlap": "X_l Mom. BLD_1 pos 0.770 [Nm]",
            "channel_baseline": "X_l Mom. BLD_1 pos 0.770 [Nm]",
            "critical": True,
            "signed_del_ratio": 1.5,
            "signed_delta_pct": 50.0,
        }
    ]
    assert report["selection_id"] == "generated-selection"
    assert report["matched_channels"] == 1
    assert report["critical_selected"] == 1
    assert not [issue for issue in issues if issue.severity == "error"]
    assert signed_path.is_file()
    assert critical_path.is_file()


def write_complete_signed_selection_manifest(
    run_root: Path,
    signed_path: Path,
    critical_path: Path,
    *,
    selection_id: str = "final-signed-del-v1",
) -> Path:
    selection_path = run_root / "signed_channel_selection.json"
    write_json(
        selection_path,
        {
            "selection_id": selection_id,
            "normalization": {
                "metric": "signed_del_ratio",
                "baseline": "no_sil_baseline",
                "comparison": "vorlap_over_baseline",
                "wohler_exponent": 10.0,
                "reference_cycle_convention": "rainflow cycles from retained 600 s record",
                "transient_removal": "none after STOREFROM=60 s",
                "mean_stress_assumption": "not applied to load-channel DEL ratios",
            },
            "artifacts": {
                "signed_del_comparison": str(signed_path.relative_to(run_root)),
                "critical_case_summary": str(critical_path.relative_to(run_root)),
            },
            "selected_channels": [
                {
                    "case": "U05p00ms",
                    "position": 0.769545,
                    "quantity": "moment",
                    "axis": "X",
                    "unit": "Nm",
                    "channel_vorlap": "X_l Mom. BLD_1 pos 0.770 [Nm]",
                    "critical": True,
                }
            ],
        },
    )
    return selection_path


def test_signed_channel_selection_rejects_empty_signed_artifact(tmp_path):
    signed_path, critical_path = write_signed_selection_artifacts(tmp_path)
    with signed_path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(
            handle,
            fieldnames=audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_del_comparison.csv"],
        ).writeheader()
    write_complete_signed_selection_manifest(tmp_path, signed_path, critical_path)
    args = SimpleNamespace(run_root=tmp_path, signed_channel_selection=None, require_signed_channel_selection=True)
    issues: list[audit.AuditIssue] = []

    report = audit.audit_signed_channel_selection({"wohler_exponent": 10.0}, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["matched_channels"] == 0
    assert "signed_channel_selection_empty_artifact" in errors
    assert "signed_channel_selection_channel_not_found" in errors


def test_signed_channel_selection_rejects_ambiguous_critical_summary(tmp_path):
    signed_path, critical_path = write_signed_selection_artifacts(tmp_path)
    with critical_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    with critical_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([rows[0], rows[0]])
    write_complete_signed_selection_manifest(tmp_path, signed_path, critical_path)
    args = SimpleNamespace(run_root=tmp_path, signed_channel_selection=None, require_signed_channel_selection=True)
    issues: list[audit.AuditIssue] = []

    report = audit.audit_signed_channel_selection({"wohler_exponent": 10.0}, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["matched_channels"] == 1
    assert report["critical_selected"] == 1
    assert "signed_channel_selection_critical_artifact_row_count" in errors


def test_signed_channel_selection_rejects_incomplete_or_stale_manifest(tmp_path):
    write_signed_selection_artifacts(tmp_path)
    selection_path = tmp_path / "signed_channel_selection.json"
    write_json(
        selection_path,
        {
            "selection_id": "",
            "normalization": {
                "metric": "",
                "baseline": "no_sil_baseline",
                "comparison": "vorlap_over_baseline",
                "wohler_exponent": 8.0,
                "reference_cycle_convention": "",
                "transient_removal": "",
                "mean_stress_assumption": "",
            },
            "selected_channels": [
                {
                    "case": "U99p00ms",
                    "position": 0.1,
                    "quantity": "force",
                    "axis": "Z",
                    "unit": "N",
                    "channel": "Z_l Force. Missing [N]",
                }
            ],
            "reports_stress_or_reserve_factors": True,
        },
    )
    args = SimpleNamespace(run_root=tmp_path, signed_channel_selection=None, require_signed_channel_selection=True)
    issues: list = []

    audit.audit_signed_channel_selection({"wohler_exponent": 10.0}, args, issues)

    errors = {issue.code for issue in issues if issue.severity == "error"}
    assert {
        "signed_channel_selection_missing_id",
        "signed_channel_selection_incomplete_normalization",
        "signed_channel_selection_wohler_mismatch",
        "signed_channel_selection_channel_not_found",
        "signed_channel_selection_critical_count",
        "signed_channel_selection_missing_stress_basis",
    } <= errors


def test_required_signed_channel_selection_reports_missing_manifest(tmp_path):
    args = SimpleNamespace(run_root=tmp_path, signed_channel_selection=None, require_signed_channel_selection=True)
    issues: list = []

    report = audit.audit_signed_channel_selection({}, args, issues)

    assert report == {"path": str(tmp_path / "signed_channel_selection.json"), "available": False}
    assert [issue.code for issue in issues if issue.severity == "error"] == ["missing_signed_channel_selection"]


def test_constant_force_probe_parses_action_and_observes_position_delta(tmp_path):
    action = qblade_runner.parse_constant_force_action("BLD_1_0.230455:Y:-250:global")

    assert action == {
        "source_node_id": "BLD_1_0.230455",
        "action": "ADDFORCE",
        "id": "BLD_1",
        "position": 0.230455,
        "direction": "Y",
        "value": -250.0,
        "local": False,
    }

    result_file = tmp_path / "probe_results.txt"
    result_file.write_text(
        "QBlade synthetic export\n"
        "generated by test\n"
        "Time~[s]\tY_g~Pos.~BLD_1~pos~0.230~[m]\n"
        "0.0\t0.0\n"
        "1.0\t-0.002\n",
        encoding="utf-8",
    )

    observations = qblade_runner.force_sign_probe_observations(result_file, [action])

    assert observations[0]["observed_position_channel"] == "Y_g~Pos.~BLD_1~pos~0.230~[m]"
    assert observations[0]["delta_position_m"] == -0.002
    assert observations[0]["status"] == "same_sign"


def test_qblade_result_time_diagnostics_parses_exported_time_column(tmp_path):
    result_file = tmp_path / "qblade_results.txt"
    result_file.write_text(
        "QBlade synthetic export\n"
        "Time~[s]\tX_g~Pos.~BLD_1~pos~0.000~[m]\n"
        "0.00\t0.0\n"
        "0.25\t0.1\n"
        "0.50\t0.2\n",
        encoding="utf-8",
    )

    diagnostics = qblade_runner.qblade_result_time_diagnostics(result_file)

    assert diagnostics["available"] is True
    assert diagnostics["column"] == "Time~[s]"
    assert diagnostics["samples"] == 3
    assert diagnostics["first_time_s"] == 0.0
    assert diagnostics["last_time_s"] == 0.5
    assert diagnostics["min_step_s"] == 0.25
    assert diagnostics["max_step_s"] == 0.25


def test_qblade_wind_input_preflight_rejects_missing_windfield(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text(
        "1 WNDTYPE - use windfield\n"
        "full6_u=10.58.bts WNDNAME - wind file\n",
        encoding="utf-8",
    )

    try:
        qblade_runner.preflight_qblade_wind_input(sim)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing WNDNAME file to fail preflight.")

    assert "full6_u=10.58.bts" in message
    assert str(tmp_path / "full6_u=10.58.bts") in message


def test_qblade_wind_input_preflight_accepts_existing_windfield(tmp_path):
    wind = tmp_path / "full6_u=10.58.bts"
    wind.write_text("synthetic placeholder\n", encoding="utf-8")
    sim = tmp_path / "case.sim"
    sim.write_text(
        "1 WNDTYPE - use windfield\n"
        "full6_u=10.58.bts WNDNAME - wind file\n",
        encoding="utf-8",
    )

    diagnostics = qblade_runner.preflight_qblade_wind_input(sim)

    assert diagnostics["requires_file"] is True
    assert diagnostics["resolved_wind_file"] == str(wind.resolve())
    assert diagnostics["exists"] is True


def test_sil_timing_diagnostics_records_pre_advance_time_level():
    diagnostics = qblade_runner.SilTimingDiagnostics(sample_limit=2)

    diagnostics.record(
        step_index=0,
        wrapper_time_s=0.0,
        wrapper_timestep=0.0,
        qblade_time_pre_advance_s=0.0,
        azimuth_deg=10.0,
        applied_force_time_s=0.0,
    )
    diagnostics.record(
        step_index=1,
        wrapper_time_s=0.25,
        wrapper_timestep=1.0,
        qblade_time_pre_advance_s=0.20,
        azimuth_deg=10.5,
        applied_force_time_s=0.25,
    )
    diagnostics.record(
        step_index=2,
        wrapper_time_s=0.50,
        wrapper_timestep=2.0,
        qblade_time_pre_advance_s=None,
        azimuth_deg=11.0,
        applied_force_time_s=0.50,
    )

    report = diagnostics.as_dict()

    assert report["time_level"] == "pre_advance_explicit"
    assert report["samples"] == 3
    assert report["qblade_time_available_count"] == 2
    assert report["qblade_time_missing_count"] == 1
    assert report["time_delta_qblade_minus_wrapper_min_s"] == -0.04999999999999999
    assert report["time_delta_qblade_minus_wrapper_max_abs_s"] == 0.04999999999999999
    assert len(report["first_samples"]) == 2
    assert report["last_sample"]["step_index"] == 2
    assert report["last_sample"]["force_application_order"] == "before_advance"


def test_results_audit_accepts_valid_sil_timing_diagnostics():
    diagnostics = {
        "time_level": "pre_advance_explicit",
        "samples": 2,
        "qblade_time_available_count": 2,
        "qblade_time_missing_count": 0,
        "wrapper_time_min_s": 0.0,
        "wrapper_time_max_s": 0.25,
        "applied_force_time_min_s": 0.0,
        "applied_force_time_max_s": 0.25,
        "time_delta_qblade_minus_wrapper_max_abs_s": 0.0,
        "first_samples": [
            {
                "step_index": 0,
                "wrapper_time_s": 0.0,
                "wrapper_timestep": 0.0,
                "qblade_time_pre_advance_s": 0.0,
                "azimuth_deg": 0.0,
                "applied_force_time_s": 0.0,
                "force_application_order": "before_advance",
            }
        ],
        "last_sample": {
            "step_index": 1,
            "wrapper_time_s": 0.25,
            "wrapper_timestep": 1.0,
            "qblade_time_pre_advance_s": 0.25,
            "azimuth_deg": 0.0,
            "applied_force_time_s": 0.25,
            "force_application_order": "before_advance",
        },
    }
    summary = {
        "sil_vorlap_timing_diagnostics": diagnostics,
        "qblade_export_time_diagnostics": {
            "available": True,
            "column": "Time~[s]",
            "samples": 2,
            "first_time_s": 0.0,
            "last_time_s": 0.25,
        },
    }
    issues = []

    audit.audit_timing_diagnostics(summary, issues, scenario="vorlap", case_label="case", completed_steps=2)

    assert not [issue for issue in issues if issue.severity == "error"]
    assert not [issue for issue in issues if issue.severity == "warning"]


def test_results_audit_rejects_missing_or_inconsistent_sil_timing_diagnostics():
    issues = []
    audit.audit_timing_diagnostics({}, issues, scenario="vorlap", case_label="case", completed_steps=2)
    assert "missing_timing_diagnostics" in {issue.code for issue in issues}

    issues = []
    summary = {
        "sil_vorlap_timing_diagnostics": {
            "time_level": "post_advance",
            "samples": 1,
            "qblade_time_available_count": 1,
            "qblade_time_missing_count": 1,
            "first_samples": [],
            "last_sample": None,
        },
        "qblade_export_time_diagnostics": {"available": False, "samples": 0},
    }

    audit.audit_timing_diagnostics(summary, issues, scenario="vorlap", case_label="case", completed_steps=2)
    errors = {issue.code for issue in issues if issue.severity == "error"}
    warnings = {issue.code for issue in issues if issue.severity == "warning"}

    assert "invalid_timing_time_level" in errors
    assert "timing_sample_count_mismatch" in errors
    assert "missing_timing_last_sample" in errors
    assert "missing_timing_first_samples" in errors
    assert "timing_qblade_time_count_mismatch" in errors
    assert "missing_qblade_export_time_diagnostics" in warnings


def test_force_sign_probe_plan_uses_generated_case_geometry(tmp_path):
    case_dir = tmp_path / "identity" / "cases" / "U05p00ms_A000p00deg"
    sim = case_dir / "baseline_reference.sim"
    sim.parent.mkdir(parents=True)
    sim.write_text("synthetic\n", encoding="utf-8")
    write_geometry_audit(case_dir, node_id="BLD_1_0.230455")
    case = {
        "label": "U05p00ms_A000p00deg",
        "sim": str(sim),
        "wind_speed": 5.0,
        "azimuth_deg": 0.0,
    }
    args = SimpleNamespace(
        run_root=tmp_path,
        scenario="identity",
        node=[],
        axes="X Z",
        force=750.0,
        local=False,
        output_root=tmp_path / "sign_probes",
        qblade_root=tmp_path / "QBladeCE_2.0.9.7",
        vorlap_root=Path(__file__).resolve().parents[1],
        python=sys.executable,
        steps=25,
        threads=1,
        progress_every=0,
        allow_partial=False,
    )

    plan = sign_probe.build_probe_plan(args, {"settings": {"rpm": 0.0}}, case)
    commands = [sign_probe.command_for_probe(args, probe) for probe in plan["probes"]]

    assert plan["nodes"] == ["BLD_1_0.230455"]
    assert len(plan["probes"]) == 4
    assert plan["probes"][0]["force_spec"] == "BLD_1_0.230455:X:750:global"
    assert plan["probes"][1]["force_spec"] == "BLD_1_0.230455:X:-750:global"
    assert all("--coupling-mode" in command and "sil-constant-force" in command for command in commands)
    assert all("--sil-constant-force" in command for command in commands)


def test_sign_probe_report_accepts_positive_paired_response_with_drift():
    rows = [
        {
            "scenario": "identity",
            "case": "U10p58ms",
            "node": "BLD_1_0.047438",
            "axis": "X",
            "sign": "positive",
            "force_n": 1000.0,
            "local": "false",
            "delta_position_m": -0.03086,
            "passed": "true",
        },
        {
            "scenario": "identity",
            "case": "U10p58ms",
            "node": "BLD_1_0.047438",
            "axis": "X",
            "sign": "negative",
            "force_n": -1000.0,
            "local": "false",
            "delta_position_m": -0.45831,
            "passed": "true",
        },
    ]

    sign_probe.apply_paired_sign_checks(rows)

    assert all(row["passed"] == "true" for row in rows)
    assert all(row["paired_passed"] == "true" for row in rows)
    assert rows[0]["paired_response_delta_m"] > 0.0


def write_force_sign_probe_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.FORCE_SIGN_PROBE_REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in audit.FORCE_SIGN_PROBE_REQUIRED_COLUMNS})


def force_sign_probe_row(
    tmp_path: Path,
    *,
    axis: str,
    sign: str,
    status: str | None = None,
    passed: bool = True,
    delta_position_m: float | None = None,
    paired_response_delta_m: float = 0.002,
    paired_passed: bool = True,
    case: str = "U05p00ms_A000p00deg",
    scenario: str = "identity",
    node: str = "BLD_1_0.230455",
    sim: Path | str = "",
    summary_overrides: dict | None = None,
) -> dict:
    force = 1000.0 if sign == "positive" else -1000.0
    delta = delta_position_m if delta_position_m is not None else (0.001 if force > 0.0 else -0.001)
    observation_status = status or ("same_sign" if force * delta > 0.0 else "opposite_sign")
    stem = f"{case}_{axis}_{sign}"
    summary = tmp_path / "summaries" / f"{stem}_run_summary.json"
    result = tmp_path / "summaries" / f"{stem}.txt"
    summary.parent.mkdir(parents=True, exist_ok=True)
    result.write_text("Time~[s]\n0\n", encoding="utf-8")
    action = {
        "source_node_id": node,
        "action": "ADDFORCE",
        "id": "BLD_1",
        "position": 0.230455,
        "direction": axis,
        "value": force,
        "local": False,
    }
    summary_data = {
        "sim": str(sim),
        "coupling_mode": "sil-constant-force",
        "requested_steps": 25,
        "completed_steps": 25,
        "failed": False,
        "exported_results": True,
        "result_file": str(result),
        "sil_constant_force": True,
        "sil_constant_force_updates": 25,
        "sil_constant_force_actions": [action],
        "sil_constant_force_max_abs_force": abs(force),
        "force_sign_probe_observations": [
            {
                "source_node_id": node,
                "action_id": "BLD_1",
                "position": 0.230455,
                "direction": axis,
                "force_n": force,
                "local": False,
                "observed_position_channel": f"{axis}_g~Pos.~BLD_1~pos~0.230~[m]",
                "delta_position_m": delta,
                "status": observation_status,
            }
        ],
        "reproducibility": {
            "case_metadata": {
                "case": {"label": case, "sim": str(sim)},
                "force_sign_probe": {
                    "scenario": scenario,
                    "case": case,
                    "node": node,
                    "axis": axis,
                    "sign": sign,
                    "force_n": force,
                    "local": False,
                    "force_spec": f"{node}:{axis}:{force:g}:global",
                },
            }
        },
    }
    if summary_overrides:
        summary_data.update(summary_overrides)
    write_json(summary, summary_data)
    return {
        "scenario": scenario,
        "case": case,
        "node": node,
        "axis": axis,
        "sign": sign,
        "force_n": force,
        "output_dir": str(summary.parent),
        "summary_file": str(summary),
        "result_file": str(result),
        "completed_steps": 25,
        "failed": "false",
        "exported_results": "true",
        "observation_status": observation_status,
        "delta_position_m": delta,
        "paired_response_delta_m": paired_response_delta_m,
        "paired_passed": "true" if paired_passed else "false",
        "passed": "true" if passed else "false",
    }


def test_force_sign_probe_audit_accepts_complete_axis_sign_set(tmp_path):
    rows = [
        force_sign_probe_row(tmp_path, axis=axis, sign=sign)
        for axis in ("X", "Y", "Z")
        for sign in ("positive", "negative")
    ]
    report_path = tmp_path / "sign_probes" / "force_sign_probe_report.csv"
    write_force_sign_probe_report(report_path, rows)
    issues = []
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="identity",
        force_sign_probe_report=report_path,
        require_force_sign_probes=True,
        force_sign_required_axes="X Y Z",
        force_sign_required_node=[],
    )

    report = audit.audit_force_sign_probes({"scenario_names": ["identity"]}, args, issues)

    assert report["complete_nodes"] == ["BLD_1_0.230455"]
    assert not [issue for issue in issues if issue.severity == "error"]


def test_force_sign_probe_audit_uses_paired_response_not_absolute_drift(tmp_path):
    deltas = {
        ("X", "positive"): -0.03086,
        ("X", "negative"): -0.45831,
        ("Y", "positive"): 0.049041,
        ("Y", "negative"): -0.090828,
        ("Z", "positive"): -0.11843,
        ("Z", "negative"): -0.36983,
    }
    rows = []
    for axis in ("X", "Y", "Z"):
        paired_delta = deltas[(axis, "positive")] - deltas[(axis, "negative")]
        for sign in ("positive", "negative"):
            rows.append(
                force_sign_probe_row(
                    tmp_path,
                    axis=axis,
                    sign=sign,
                    delta_position_m=deltas[(axis, sign)],
                    paired_response_delta_m=paired_delta,
                )
            )
    report_path = tmp_path / "sign_probes" / "force_sign_probe_report.csv"
    write_force_sign_probe_report(report_path, rows)
    issues = []
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="identity",
        force_sign_probe_report=report_path,
        require_force_sign_probes=True,
        force_sign_required_axes="X Y Z",
        force_sign_required_node=[],
    )

    report = audit.audit_force_sign_probes({"scenario_names": ["identity"]}, args, issues)

    assert report["complete_nodes"] == ["BLD_1_0.230455"]
    assert not [issue for issue in issues if issue.severity == "error"]


def test_force_sign_probe_audit_rejects_complete_rows_from_unexpected_scenario(tmp_path):
    rows = [
        force_sign_probe_row(tmp_path, axis=axis, sign=sign)
        for axis in ("X", "Y", "Z")
        for sign in ("positive", "negative")
    ]
    for row in rows:
        row["scenario"] = "unexpected"
    report_path = tmp_path / "sign_probes" / "force_sign_probe_report.csv"
    write_force_sign_probe_report(report_path, rows)
    issues = []
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="identity",
        force_sign_probe_report=report_path,
        require_force_sign_probes=True,
        force_sign_required_axes="X Y Z",
        force_sign_required_node=[],
    )

    report = audit.audit_force_sign_probes({"scenario_names": ["identity"]}, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}
    warnings = {issue.code for issue in issues if issue.severity == "warning"}

    assert report["complete_nodes"] == []
    assert "missing_complete_force_sign_probe_node" in errors
    assert "unexpected_force_sign_probe_scenario" in warnings


def test_force_sign_probe_audit_rejects_rows_from_unexpected_or_split_cases(tmp_path):
    rows = [
        force_sign_probe_row(tmp_path, axis=axis, sign=sign, case="stale_case")
        for axis in ("X", "Y", "Z")
        for sign in ("positive", "negative")
    ]
    report_path = tmp_path / "sign_probes" / "force_sign_probe_report.csv"
    write_force_sign_probe_report(report_path, rows)
    issues = []
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="identity",
        force_sign_probe_report=report_path,
        require_force_sign_probes=True,
        force_sign_required_axes="X Y Z",
        force_sign_required_node=[],
    )
    manifests = {"identity": {"cases": [{"label": "U05p00ms_A000p00deg"}]}}

    report = audit.audit_force_sign_probes({"scenario_names": ["identity"]}, args, issues, manifests)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["complete_nodes"] == []
    assert "unexpected_force_sign_probe_case" in errors
    assert "missing_complete_force_sign_probe_node" in errors

    split_rows = [
        force_sign_probe_row(
            tmp_path,
            axis=axis,
            sign=sign,
            case="U05p00ms_A000p00deg" if axis in {"X", "Y"} else "U10p00ms_A000p00deg",
        )
        for axis in ("X", "Y", "Z")
        for sign in ("positive", "negative")
    ]
    write_force_sign_probe_report(report_path, split_rows)
    issues = []
    manifests = {
        "identity": {
            "cases": [
                {"label": "U05p00ms_A000p00deg"},
                {"label": "U10p00ms_A000p00deg"},
            ]
        }
    }

    report = audit.audit_force_sign_probes({"scenario_names": ["identity"]}, args, issues, manifests)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["complete_nodes"] == []
    assert report["complete_node_cases"] == []
    assert "missing_complete_force_sign_probe_node" in errors


def test_force_sign_probe_audit_rejects_summary_that_contradicts_csv(tmp_path):
    rows = [
        force_sign_probe_row(tmp_path, axis=axis, sign=sign)
        for axis in ("X", "Y", "Z")
        for sign in ("positive", "negative")
    ]
    bad_summary_path = Path(rows[0]["summary_file"])
    bad_summary = json.loads(bad_summary_path.read_text(encoding="utf-8"))
    bad_summary["sil_constant_force_actions"][0]["direction"] = "Y"
    bad_summary["force_sign_probe_observations"][0]["direction"] = "Y"
    bad_summary["reproducibility"]["case_metadata"]["force_sign_probe"]["axis"] = "Y"
    write_json(bad_summary_path, bad_summary)
    report_path = tmp_path / "sign_probes" / "force_sign_probe_report.csv"
    write_force_sign_probe_report(report_path, rows)
    issues = []
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="identity",
        force_sign_probe_report=report_path,
        require_force_sign_probes=True,
        force_sign_required_axes="X Y Z",
        force_sign_required_node=[],
    )

    report = audit.audit_force_sign_probes({"scenario_names": ["identity"]}, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["complete_nodes"] == []
    assert "force_sign_probe_summary_action_mismatch" in errors
    assert "force_sign_probe_summary_observation_mismatch" in errors
    assert "force_sign_probe_summary_metadata_mismatch" in errors
    assert "missing_complete_force_sign_probe_node" in errors


def test_force_sign_probe_audit_rejects_summary_from_different_manifest_case(tmp_path):
    sim = tmp_path / "identity" / "cases" / "U05p00ms_A000p00deg" / "baseline_reference.sim"
    sim.parent.mkdir(parents=True)
    sim.write_text("synthetic\n", encoding="utf-8")
    stale_sim = tmp_path / "stale" / "baseline_reference.sim"
    rows = [
        force_sign_probe_row(tmp_path, axis=axis, sign=sign, sim=stale_sim)
        for axis in ("X", "Y", "Z")
        for sign in ("positive", "negative")
    ]
    report_path = tmp_path / "sign_probes" / "force_sign_probe_report.csv"
    write_force_sign_probe_report(report_path, rows)
    issues = []
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="identity",
        force_sign_probe_report=report_path,
        require_force_sign_probes=True,
        force_sign_required_axes="X Y Z",
        force_sign_required_node=[],
    )
    manifests = {"identity": {"cases": [{"label": "U05p00ms_A000p00deg", "sim": str(sim)}]}}

    report = audit.audit_force_sign_probes({"scenario_names": ["identity"]}, args, issues, manifests)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["complete_nodes"] == []
    assert "force_sign_probe_manifest_sim_mismatch" in errors
    assert "missing_complete_force_sign_probe_node" in errors


def test_force_sign_probe_audit_rejects_missing_or_failed_evidence(tmp_path):
    issues = []
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="identity",
        force_sign_probe_report=None,
        require_force_sign_probes=True,
        force_sign_required_axes="X Y Z",
        force_sign_required_node=[],
    )

    missing_report = audit.audit_force_sign_probes({"scenario_names": ["identity"]}, args, issues)
    codes = {issue.code for issue in issues}

    assert missing_report is None
    assert "missing_force_sign_probe_report" in codes

    rows = [
        force_sign_probe_row(
            tmp_path,
            axis=axis,
            sign=sign,
            status="missing_position_channel" if axis == "Y" and sign == "negative" else None,
            passed=not (axis == "Y" and sign == "negative"),
        )
        for axis in ("X", "Y", "Z")
        for sign in ("positive", "negative")
    ]
    report_path = tmp_path / "sign_probes" / "force_sign_probe_report.csv"
    write_force_sign_probe_report(report_path, rows)
    issues = []
    args.force_sign_probe_report = report_path

    report = audit.audit_force_sign_probes({"scenario_names": ["identity"]}, args, issues)
    codes = {issue.code for issue in issues}

    assert report["complete_nodes"] == []
    assert "force_sign_probe_observation_failed" in codes
    assert "missing_complete_force_sign_probe_node" in codes


def test_sign_probe_audit_stage_accepts_setup_with_complete_report(tmp_path):
    settings = {
        "wind_speeds": [5.0],
        "azimuths_deg": [0.0],
        "planned_cases": [{"wind_speed_mps": 5.0, "parked_azimuth_deg": 0.0}],
        "planned_case_count_per_scenario": 1,
        "planned_case_count_total": 1,
        "scenario_names": ["identity"],
        "scenario_force_scales": {"identity": 0.0},
        "scenario_couplings": {"identity": "sil-vorlap"},
        "duration_s": 600.0,
        "timestep_s": 0.0075,
        "timesteps": 80000,
        "rpm": 0.0,
    }
    write_json(tmp_path / "campaign_settings.json", settings)
    case_dir = tmp_path / "identity" / "cases" / "U05p00ms_A000p00deg"
    sim = case_dir / "baseline_reference.sim"
    sim.parent.mkdir(parents=True)
    sim.write_text("synthetic\n", encoding="utf-8")
    write_geometry_audit(case_dir, node_id="BLD_1_0.230455")
    case = {
        "index": 0,
        "label": "U05p00ms_A000p00deg",
        "wind_speed": 5.0,
        "azimuth_deg": 0.0,
        "sim": str(sim),
        "result_dir": str(tmp_path / "identity" / "results" / "U05p00ms_A000p00deg"),
    }
    write_json(
        tmp_path / "identity" / "cases.json",
        {"settings": {"rpm": 0.0, "timesteps": 80000, "timestep": 0.0075, "force_scale": 0.0, "coupling": "sil-vorlap"}, "cases": [case]},
    )
    rows = [
        force_sign_probe_row(tmp_path, axis=axis, sign=sign, sim=sim)
        for axis in ("X", "Y", "Z")
        for sign in ("positive", "negative")
    ]
    report_path = tmp_path / "sign_probes" / "force_sign_probe_report.csv"
    write_force_sign_probe_report(report_path, rows)
    args = SimpleNamespace(
        run_root=tmp_path,
        stage="sign-probes",
        scenarios="identity",
        min_duration=600.0,
        max_timestep=0.01,
        expected_rpm=0.0,
        require_wind_name=False,
        allow_quasi_static=False,
        force_sign_probe_report=report_path,
        require_force_sign_probes=False,
        force_sign_required_axes="X Y Z",
        force_sign_required_node=[],
    )

    report = audit.audit_campaign(args)

    assert report["passed"]
    assert report["force_sign_probes"]["complete_nodes"] == ["BLD_1_0.230455"]


def test_plan_audit_reports_malformed_version_two_environment_without_crashing(tmp_path):
    settings = {
        "campaign_settings_version": 2,
        "wind_speeds": [5.0],
        "azimuths_deg": [0.0],
        "planned_cases": [{"wind_speed_mps": 5.0, "parked_azimuth_deg": 0.0}],
        "planned_case_count_per_scenario": 1,
        "planned_case_count_total": 1,
        "scenario_names": ["baseline"],
        "scenario_force_scales": {"baseline": 0.0},
        "scenario_couplings": {"baseline": "none"},
        "duration_s": 1.0,
        "store_from_s": 0.0,
        "timestep_s": 0.1,
        "timesteps": 10,
        "rpm": 1.0,
        "rpm_prescribe_type": 1,
        "horizontal_inflow_angle_deg": 0.0,
        "vertical_inflow_angle_deg": 0.0,
        "shear_exponent": 0.23,
        "reference_height_m": None,
        "turbsim_ref_height_m": None,
        "air_density_kg_m3": 1.0296,
        "air_kinematic_viscosity_m2_s": 1.647e-5,
    }
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="baseline",
        min_duration=1.0,
        max_timestep=0.1,
        expected_rpm=1.0,
        allow_quasi_static=False,
        require_wind_name=False,
    )
    issues = []

    audit.audit_plan(settings, args, issues)

    assert "invalid_explicit_environment_setting" in {issue.code for issue in issues}


def write_source_coverage_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys([
        *audit.SOURCE_COVERAGE_REQUIRED_COLUMNS,
        *audit.SOURCE_COVERAGE_OPTIONAL_COLUMNS,
        *audit.SOURCE_COVERAGE_COUNT_COLUMNS,
    ]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def source_coverage_row(*, component: str, samples: int = 25, node_samples: int | None = None) -> dict:
    row = {
        "scenario": "vorlap",
        "case": "U05p00ms_A000p00deg",
        "component": component,
        "airfoil_id": "NACA0018",
        "reynolds_model": "replicated_reynolds_independent",
        "wind_speed_mps": 5.0,
        "azimuth_deg": 0.0,
        "aoa_min_deg": -5.0,
        "aoa_max_deg": 5.0,
        "source_aoa_min_deg": -30.0,
        "source_aoa_max_deg": 30.0,
        "re_min": 1000.0,
        "re_max": 2000.0,
        "source_re_min": 500.0,
        "source_re_max": 5000.0,
        "freq_min_hz": 0.1,
        "freq_max_hz": 2.0,
        "nyquist_hz": 50.0,
        "nyquist_ratio": 25.0,
        "max_nodal_force_n": 1.0,
        "force_coefficient_max_abs": 0.1,
        "source_force_coefficient_max_abs": 1.0,
        "max_structural_velocity_mps": 0.1,
        "force_power_mean_w": 0.0,
        "force_power_max_abs_w": 0.1,
        "samples": samples,
        "node_samples": samples if node_samples is None else node_samples,
    }
    row.update({column: 0 for column in audit.SOURCE_COVERAGE_COUNT_COLUMNS})
    row["re_lower_bound_warning_count"] = 0
    return row


def source_coverage_audit_fixture(tmp_path: Path) -> tuple[dict, dict, SimpleNamespace, Path]:
    case_dir = tmp_path / "vorlap" / "cases" / "U05p00ms_A000p00deg"
    result_dir = tmp_path / "vorlap" / "results" / "U05p00ms_A000p00deg"
    sim = case_dir / "baseline_reference.sim"
    config = case_dir / "reference_model" / "Control" / "vorlap_qblade_external.json"
    sim.parent.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    sim.write_text("synthetic\n", encoding="utf-8")
    write_json(result_dir / "baseline_reference_run_summary.json", {"completed_steps": 25})
    write_json(
        config,
        {
            "node_ids": [
                "BLD_1_0.100000",
                "BLD_2_0.200000",
                "STR_1_1_0.300000",
            ],
        },
    )
    case = {
        "label": "U05p00ms_A000p00deg",
        "wind_speed": 5.0,
        "azimuth_deg": 0.0,
        "sim": str(sim),
        "result_dir": str(result_dir),
    }
    settings = {
        "scenario_names": ["vorlap"],
        "airfoil_source_manifest": [
            {
                "airfoil_id": "NACA0018",
                "reynolds_model": "replicated_reynolds_independent",
            }
        ],
    }
    manifests = {"vorlap": {"cases": [case]}}
    source_path = tmp_path / "postprocess" / "source_coverage.csv"
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="vorlap",
        source_coverage=source_path,
        require_source_coverage=True,
        source_coverage_scenarios="vorlap",
        min_nyquist_ratio=1.0,
        allow_source_coverage_violations=0,
        allow_source_range_excursions=False,
        source_range_tolerance=1.0e-9,
    )
    return settings, manifests, args, source_path


def test_source_coverage_audit_requires_rows_for_all_runtime_config_components(tmp_path):
    settings, manifests, args, source_path = source_coverage_audit_fixture(tmp_path)
    write_source_coverage_report(source_path, [source_coverage_row(component="BLD_1")])
    issues = []

    report = audit.audit_source_coverage(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}
    missing = {issue.message.rsplit(" ", 1)[-1].rstrip(".") for issue in issues if issue.code == "missing_source_coverage_component"}

    assert report["expected_case_components"] == 3
    assert report["covered_case_components"] == 1
    assert "missing_source_coverage_component" in errors
    assert missing == {"BLD_2", "STR_1_1"}

    write_source_coverage_report(
        source_path,
        [
            source_coverage_row(component="BLD_1"),
            source_coverage_row(component="BLD_2"),
            source_coverage_row(component="STR_1_1"),
        ],
    )
    issues = []
    report = audit.audit_source_coverage(settings, manifests, args, issues)

    assert report["expected_case_components"] == 3
    assert report["covered_case_components"] == 3
    assert not [issue for issue in issues if issue.severity == "error"]


def test_source_coverage_audit_rejects_incomplete_sample_counts(tmp_path):
    settings, manifests, args, source_path = source_coverage_audit_fixture(tmp_path)
    write_source_coverage_report(
        source_path,
        [
            source_coverage_row(component="BLD_1", samples=24, node_samples=24),
            source_coverage_row(component="BLD_2", samples=25, node_samples=2),
            source_coverage_row(component="STR_1_1", samples=25, node_samples=25),
        ],
    )
    issues = []

    audit.audit_source_coverage(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert "source_coverage_sample_count_mismatch" in errors
    assert "source_coverage_node_sample_count_mismatch" in errors


def test_source_coverage_audit_rejects_production_limit_violations(tmp_path):
    settings, manifests, args, source_path = source_coverage_audit_fixture(tmp_path)
    bad = source_coverage_row(component="BLD_1")
    bad.update(
        {
            "aoa_min_deg": -40.0,
            "re_max": 6000.0,
            "freq_max_hz": 60.0,
            "nyquist_hz": 50.0,
            "nyquist_ratio": 0.75,
            "force_coefficient_max_abs": 1.5,
            "re_out_of_range_count": 1,
            "aliased_frequency_count": 2,
        }
    )
    write_source_coverage_report(
        source_path,
        [
            bad,
            source_coverage_row(component="BLD_2"),
            source_coverage_row(component="STR_1_1"),
        ],
    )
    issues = []

    audit.audit_source_coverage(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert "source_aoa_out_of_range" in errors
    assert "source_re_out_of_range" in errors
    assert "source_frequency_above_nyquist" in errors
    assert "source_nyquist_margin_too_low" in errors
    assert "source_force_coefficient_out_of_range" in errors
    assert "source_coverage_violation_count" in errors


def test_source_coverage_audit_warns_for_verified_independent_lower_re_only(tmp_path):
    settings, manifests, args, source_path = source_coverage_audit_fixture(tmp_path)
    low_re = source_coverage_row(component="BLD_1")
    low_re.update({"re_min": 400.0, "re_lower_bound_warning_count": 3})
    write_source_coverage_report(
        source_path,
        [
            low_re,
            source_coverage_row(component="BLD_2"),
            source_coverage_row(component="STR_1_1"),
        ],
    )
    issues = []

    report = audit.audit_source_coverage(settings, manifests, args, issues)

    assert report["re_lower_bound_warning_count"] == 3
    assert not [item for item in issues if item.severity == "error"]
    assert "source_re_lower_bound_warning" in {
        item.code for item in issues if item.severity == "warning"
    }


def test_source_coverage_audit_rejects_unverified_lower_re_warning(tmp_path):
    settings, manifests, args, source_path = source_coverage_audit_fixture(tmp_path)
    low_re = source_coverage_row(component="BLD_1")
    low_re.update(
        {
            "re_min": 400.0,
            "reynolds_model": "",
            "re_lower_bound_warning_count": 3,
        }
    )
    write_source_coverage_report(
        source_path,
        [
            low_re,
            source_coverage_row(component="BLD_2"),
            source_coverage_row(component="STR_1_1"),
        ],
    )
    issues = []

    audit.audit_source_coverage(settings, manifests, args, issues)
    errors = {item.code for item in issues if item.severity == "error"}

    assert "source_re_out_of_range" in errors
    assert "invalid_re_lower_bound_warning" in errors


def test_source_coverage_audit_can_warn_for_explicitly_allowed_source_range_excursions(tmp_path):
    settings, manifests, args, source_path = source_coverage_audit_fixture(tmp_path)
    args.allow_source_range_excursions = True
    bad = source_coverage_row(component="BLD_1")
    bad.update(
        {
            "aoa_min_deg": -40.0,
            "re_max": 6000.0,
            "force_coefficient_max_abs": 1.5,
            "re_out_of_range_count": 1,
            "source_clip_count": 1,
            "force_coefficient_out_of_range_count": 1,
        }
    )
    write_source_coverage_report(
        source_path,
        [
            bad,
            source_coverage_row(component="BLD_2"),
            source_coverage_row(component="STR_1_1"),
        ],
    )
    issues = []

    audit.audit_source_coverage(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}
    warnings = {issue.code for issue in issues if issue.severity == "warning"}

    assert "source_aoa_out_of_range" in warnings
    assert "source_re_out_of_range" in warnings
    assert "source_force_coefficient_out_of_range" in warnings
    assert "source_coverage_violation_count" in warnings
    assert not errors


def geometry_position_proof_row(
    node_id: str,
    *,
    index: int,
    reported: bool = True,
    fit: bool = True,
    residual_m: float = 0.02,
) -> dict:
    action_position = 0.1 * (index + 1)
    row = {
        "swap_index": index,
        "controller_index": index,
        "requested_node_id": node_id,
        "canonical_node_id": node_id,
        "controller_node_id": node_id,
        "action_id": "BLD_1",
        "action_position": action_position,
        "force_swap_x": 6 + 3 * index,
        "force_swap_y": 7 + 3 * index,
        "force_swap_z": 8 + 3 * index,
        "velocity_swap_x": 3 + 3 * index,
        "velocity_swap_y": 4 + 3 * index,
        "velocity_swap_z": 5 + 3 * index,
        "vorlap_x_m": float(index),
        "vorlap_y_m": 0.0,
        "vorlap_z_m": 0.0,
        "chord_norm": 1.0,
        "normal_norm": 1.0,
        "span_norm": 1.0,
        "chord_dot_normal": 0.0,
        "chord_dot_span": 0.0,
        "normal_dot_span": 0.0,
        "basis_determinant": 1.0,
        "basis_status": "ok",
        "qblade_reported_available": str(reported).lower(),
        "qblade_reported_x_m": float(index),
        "qblade_reported_y_m": 0.0,
        "qblade_reported_z_m": -3.0,
        "qblade_frame_fit_available": str(fit).lower(),
        "qblade_frame_fit_x_m": float(index),
        "qblade_frame_fit_y_m": 0.0,
        "qblade_frame_fit_z_m": -3.0,
        "delta_reported_frame_fit_m": residual_m,
    }
    return row


def write_geometry_position_proof(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(
        dict.fromkeys(
            [
                *audit.GEOMETRY_AUDIT_REQUIRED_COLUMNS,
                "qblade_reported_available",
                "qblade_reported_x_m",
                "qblade_reported_y_m",
                "qblade_reported_z_m",
                "qblade_frame_fit_available",
                "qblade_frame_fit_x_m",
                "qblade_frame_fit_y_m",
                "qblade_frame_fit_z_m",
                "delta_reported_frame_fit_m",
            ]
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def geometry_position_proof_fixture(tmp_path: Path) -> tuple[dict, dict, SimpleNamespace, Path]:
    case_dir = tmp_path / "vorlap" / "cases" / "U05p00ms_A000p00deg"
    result_dir = tmp_path / "vorlap" / "results" / "U05p00ms_A000p00deg"
    sim = case_dir / "baseline_reference.sim"
    audit_path = case_dir / "reference_model" / "Control" / "vorlap_qblade_geometry_audit.csv"
    sim.parent.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    sim.write_text("synthetic\n", encoding="utf-8")
    write_json(
        result_dir / "baseline_reference_run_summary.json",
        {
            "completed_steps": 200,
            "coupling_mode": "sil-vorlap",
            "sil_vorlap_geometry_audit_file": str(audit_path),
        },
    )
    case = {
        "label": "U05p00ms_A000p00deg",
        "wind_speed": 5.0,
        "azimuth_deg": 0.0,
        "sim": str(sim),
        "result_dir": str(result_dir),
    }
    settings = {"scenario_names": ["vorlap"]}
    manifests = {"vorlap": {"cases": [case]}}
    args = SimpleNamespace(
        run_root=tmp_path,
        scenarios="vorlap",
        source_coverage_scenarios="vorlap",
        require_geometry_position_proof=True,
        geometry_proof_scenarios="vorlap",
        max_geometry_frame_fit_residual_m=0.1,
        min_geometry_frame_fit_points=3,
        allow_missing_geometry_reported_nodes=0,
    )
    return settings, manifests, args, audit_path


def test_geometry_position_proof_audit_requires_loaded_manifest(tmp_path):
    settings, _manifests, args, _audit_path = geometry_position_proof_fixture(tmp_path)
    issues: list[audit.AuditIssue] = []

    report = audit.audit_geometry_position_proof(settings, {}, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["checked_cases"] == 0
    assert report["passing_cases"] == 0
    assert "geometry_position_proof_missing_manifest" in errors
    assert "geometry_position_proof_no_cases" in errors
    assert "geometry_position_proof_no_passing_cases" in errors


def test_geometry_position_proof_audit_rejects_missing_and_out_of_tolerance_rows(tmp_path):
    settings, manifests, args, audit_path = geometry_position_proof_fixture(tmp_path)
    write_geometry_position_proof(
        audit_path,
        [
            geometry_position_proof_row("BLD_1_0.100000", index=0, residual_m=0.02),
            geometry_position_proof_row("BLD_1_0.200000", index=1, residual_m=0.25),
            geometry_position_proof_row("BLD_1_0.300000", index=2, reported=False, fit=False),
        ],
    )
    issues: list[audit.AuditIssue] = []

    report = audit.audit_geometry_position_proof(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["checked_cases"] == 1
    assert report["passing_cases"] == 0
    assert report["checked_nodes"] == 3
    assert report["cases"][0]["passed"] is False
    assert "geometry_position_fit_residual_exceeds_tolerance" in errors
    assert "geometry_position_missing_reported_nodes" in errors
    assert "geometry_position_insufficient_frame_fit_points" in errors
    assert "geometry_position_proof_no_passing_cases" in errors

    write_geometry_position_proof(
        audit_path,
        [
            geometry_position_proof_row("BLD_1_0.100000", index=0, residual_m=0.02),
            geometry_position_proof_row("BLD_1_0.200000", index=1, residual_m=0.03),
            geometry_position_proof_row("BLD_1_0.300000", index=2, residual_m=0.04),
        ],
    )
    issues = []
    report = audit.audit_geometry_position_proof(settings, manifests, args, issues)

    assert report["passing_cases"] == 1
    assert report["reported_nodes"] == 3
    assert report["frame_fit_nodes"] == 3
    assert report["cases"][0]["passed"] is True
    assert not [issue for issue in issues if issue.severity == "error"]


def write_sil_force_history(path: Path, time: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step_index", "time_s", *columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, t in enumerate(time):
            row = {"step_index": i, "time_s": float(t)}
            row.update({column: float(values[i]) for column, values in columns.items()})
            writer.writerow(row)


def spectral_force_audit_fixture(tmp_path: Path) -> tuple[dict, dict, SimpleNamespace, Path]:
    settings, manifests, args, _source_path = source_coverage_audit_fixture(tmp_path)
    args.require_spectral_force_content = True
    args.spectral_force_scenarios = "vorlap"
    args.min_spectral_force_samples = 32
    args.min_spectral_force_dynamic_power = 1.0e-10
    args.min_spectral_force_band_power_fraction = 0.95

    case = manifests["vorlap"]["cases"][0]
    result_dir = Path(case["result_dir"])
    history_path = result_dir / "U05p00ms_A000p00deg_qblade_results_sil_vorlap_history.csv"
    write_json(
        result_dir / "baseline_reference_run_summary.json",
        {
            "completed_steps": 200,
            "coupling_mode": "sil-vorlap",
            "sil_vorlap_history": {"path": str(history_path), "rows": 200},
            "sil_vorlap_source_coverage": [
                source_coverage_row(component="BLD_1", samples=200, node_samples=200),
                source_coverage_row(component="BLD_2", samples=200, node_samples=200),
                source_coverage_row(component="STR_1_1", samples=200, node_samples=200),
            ],
        },
    )
    return settings, manifests, args, history_path


def test_spectral_force_content_audit_requires_loaded_manifest(tmp_path):
    settings, _manifests, args, _history_path = spectral_force_audit_fixture(tmp_path)
    issues: list[audit.AuditIssue] = []

    report = audit.audit_spectral_force_content(settings, {}, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["checked_cases"] == 0
    assert report["passing_cases"] == 0
    assert report["checked_components"] == 0
    assert report["passing_components"] == 0
    assert report["checked_force_channels"] == 0
    assert "spectral_force_content_missing_manifest" in errors
    assert "spectral_force_content_no_cases" in errors
    assert "spectral_force_content_no_components" in errors
    assert "spectral_force_content_no_force_channels" in errors
    assert "spectral_force_content_no_passing_cases" in errors


def test_spectral_force_content_audit_rejects_out_of_band_force_peak(tmp_path):
    settings, manifests, args, history_path = spectral_force_audit_fixture(tmp_path)
    time = np.arange(0.0, 4.0, 0.02)
    in_band = np.cos(2.0 * np.pi * 1.5 * time)
    out_of_band = np.cos(2.0 * np.pi * 4.0 * time)
    write_sil_force_history(
        history_path,
        time,
        {
            "force_BLD_1_0_100000_x_n": out_of_band,
            "force_BLD_2_0_200000_x_n": in_band,
            "force_STR_1_1_0_300000_x_n": in_band,
        },
    )
    issues: list[audit.AuditIssue] = []

    report = audit.audit_spectral_force_content(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["checked_cases"] == 1
    assert report["passing_cases"] == 0
    assert report["checked_components"] == 3
    assert report["passing_components"] == 2
    assert report["checked_force_channels"] == 3
    assert report["cases"][0]["passed"] is False
    assert "spectral_force_content_out_of_source_band" in errors
    assert "spectral_force_content_no_passing_cases" in errors

    write_sil_force_history(
        history_path,
        time,
        {
            "force_BLD_1_0_100000_x_n": in_band,
            "force_BLD_2_0_200000_x_n": in_band,
            "force_STR_1_1_0_300000_x_n": in_band,
        },
    )
    issues = []
    report = audit.audit_spectral_force_content(settings, manifests, args, issues)

    assert report["passing_cases"] == 1
    assert report["passing_components"] == 3
    assert report["checked_components"] == 3
    assert report["cases"][0]["passed"] is True
    assert all(component["passed"] for component in report["components"])
    assert not [issue for issue in issues if issue.severity == "error"]


def write_spectral_psd_comparison(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.SPECTRAL_PSD_COMPARISON_REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in audit.SPECTRAL_PSD_COMPARISON_REQUIRED_COLUMNS})


def spectral_psd_comparison_row(*, source: str, peak_frequency_hz: float = 1.5, **updates) -> dict:
    expected_frequency_hz = float(updates.pop("expected_frequency_hz", 1.5))
    row = {
        "scenario": "vorlap",
        "case": "U05p00ms_A000p00deg",
        "component": "BLD_1",
        "channel": "force_BLD_1_0_100000_x_n",
        "source": source,
        "expected_frequency_hz": expected_frequency_hz,
        "peak_frequency_hz": peak_frequency_hz,
        "frequency_error_hz": abs(float(peak_frequency_hz) - expected_frequency_hz),
        "band_power_fraction": 0.98,
        "peak_power": 1.0,
        "passed": "true",
        "artifact": f"postprocess/{source}_psd.csv",
    }
    row.update(updates)
    return row


def spectral_psd_comparison_audit_fixture(tmp_path: Path) -> tuple[dict, dict, SimpleNamespace, Path]:
    settings, manifests, args, _source_path = source_coverage_audit_fixture(tmp_path)
    psd_path = tmp_path / "postprocess" / "spectral_psd_comparison.csv"
    args.require_spectral_psd_comparison = True
    args.spectral_psd_comparison = psd_path
    args.spectral_psd_comparison_scenarios = "vorlap"
    args.max_spectral_psd_frequency_error_hz = 0.05
    args.min_spectral_psd_band_power_fraction = 0.8
    return settings, manifests, args, psd_path


def test_spectral_psd_comparison_audit_requires_loaded_manifest(tmp_path):
    settings, _manifests, args, psd_path = spectral_psd_comparison_audit_fixture(tmp_path)
    write_spectral_psd_comparison(
        psd_path,
        [spectral_psd_comparison_row(source=source) for source in audit.SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES],
    )
    issues: list[audit.AuditIssue] = []

    report = audit.audit_spectral_psd_comparison(settings, {}, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["missing_manifests"] == ["vorlap"]
    assert report["expected_cases"] == 0
    assert report["checked_groups"] == 1
    assert report["complete_cases"] == 0
    assert "spectral_psd_comparison_missing_manifest" in errors
    assert "spectral_psd_comparison_no_expected_cases" in errors
    assert "spectral_psd_comparison_no_complete_cases" in errors
    assert "unknown_spectral_psd_comparison_case" in errors


def test_spectral_psd_comparison_audit_rejects_empty_group_bundle(tmp_path):
    settings, manifests, args, psd_path = spectral_psd_comparison_audit_fixture(tmp_path)
    write_spectral_psd_comparison(psd_path, [])
    issues: list[audit.AuditIssue] = []

    report = audit.audit_spectral_psd_comparison(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["expected_cases"] == 1
    assert report["checked_groups"] == 0
    assert report["complete_cases"] == 0
    assert "missing_spectral_psd_comparison_case" in errors
    assert "missing_complete_spectral_psd_comparison_group" in errors
    assert "spectral_psd_comparison_no_groups" in errors
    assert "spectral_psd_comparison_no_complete_cases" in errors


def test_spectral_psd_comparison_audit_requires_complete_source_group(tmp_path):
    settings, manifests, args, psd_path = spectral_psd_comparison_audit_fixture(tmp_path)
    write_spectral_psd_comparison(
        psd_path,
        [
            spectral_psd_comparison_row(source="analytic"),
            spectral_psd_comparison_row(source="offline"),
            spectral_psd_comparison_row(source="sil"),
        ],
    )
    issues: list[audit.AuditIssue] = []

    report = audit.audit_spectral_psd_comparison(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["checked_groups"] == 1
    assert report["complete_cases"] == 0
    assert "spectral_psd_comparison_missing_source" in errors
    assert "missing_complete_spectral_psd_comparison_group" in errors
    assert "spectral_psd_comparison_no_complete_cases" in errors

    write_spectral_psd_comparison(
        psd_path,
        [spectral_psd_comparison_row(source=source) for source in audit.SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES],
    )
    issues = []
    report = audit.audit_spectral_psd_comparison(settings, manifests, args, issues)

    assert report["valid_rows"] == 4
    assert report["complete_cases"] == 1
    assert not [issue for issue in issues if issue.severity == "error"]


def test_spectral_psd_comparison_audit_rejects_bad_rows(tmp_path):
    settings, manifests, args, psd_path = spectral_psd_comparison_audit_fixture(tmp_path)
    rows = [spectral_psd_comparison_row(source=source) for source in audit.SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES]
    rows[0].update({"peak_frequency_hz": 1.6, "frequency_error_hz": 0.1})
    rows[1].update({"peak_frequency_hz": 1.7, "frequency_error_hz": 0.01})
    rows[2].update({"band_power_fraction": 0.5})
    rows[3].update({"passed": "false", "artifact": ""})
    write_spectral_psd_comparison(psd_path, rows)
    issues: list[audit.AuditIssue] = []

    audit.audit_spectral_psd_comparison(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert {
        "spectral_psd_comparison_error_mismatch",
        "spectral_psd_comparison_frequency_error_exceeds_tolerance",
        "spectral_psd_comparison_band_power_too_low",
        "spectral_psd_comparison_not_passed",
        "spectral_psd_comparison_missing_artifact",
    } <= errors


def test_spectral_psd_comparison_builder_writes_audit_compatible_rows(tmp_path):
    settings, manifests, args, psd_path = spectral_psd_comparison_audit_fixture(tmp_path)
    spec_path = tmp_path / "postprocess" / "spectral_psd_sources.csv"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(0.0, 16.0, 0.1)
    expected_hz = 1.5
    with spec_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spectral_psd_builder.SPECTRAL_PSD_SOURCE_SPEC_REQUIRED_COLUMNS)
        writer.writeheader()
        for index, source in enumerate(audit.SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES):
            artifact = tmp_path / "postprocess" / f"{source}_source_history.csv"
            values = (index + 1) * np.cos(2.0 * np.pi * expected_hz * time + 0.1 * index)
            write_sil_force_history(artifact, time, {"value": values})
            writer.writerow(
                {
                    "scenario": "vorlap",
                    "case": "U05p00ms_A000p00deg",
                    "component": "BLD_1",
                    "channel": "force_BLD_1_0_100000_x_n",
                    "source": source,
                    "expected_frequency_hz": expected_hz,
                    "artifact": artifact.relative_to(tmp_path).as_posix(),
                    "time_column": "time_s",
                    "value_column": "value",
                }
            )

    rows = spectral_psd_builder.build_comparison_rows(tmp_path, spec_path)
    spectral_psd_builder.write_comparison_csv(psd_path, rows)
    issues: list[audit.AuditIssue] = []
    report = audit.audit_spectral_psd_comparison(settings, manifests, args, issues)

    assert len(rows) == 4
    assert {row["source"] for row in rows} == set(audit.SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES)
    assert all(row["passed"] == "true" for row in rows)
    assert all(math.isclose(float(row["peak_frequency_hz"]), expected_hz) for row in rows)
    assert report["complete_cases"] == 1
    assert not [issue for issue in issues if issue.severity == "error"]


def write_ultimate_margin_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.ULTIMATE_MARGIN_REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in audit.ULTIMATE_MARGIN_REQUIRED_COLUMNS})


def write_extreme_channel_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit.ULTIMATE_EXTREME_REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in audit.ULTIMATE_EXTREME_REQUIRED_COLUMNS})


def extreme_channel_row(**overrides) -> dict:
    row = {
        "scenario": "vorlap",
        "case": "U05p00ms_A000p00deg",
        "wind_speed_mps": 5.0,
        "azimuth_deg": 0.0,
        "blade": "BLD_1",
        "position": 0.23,
        "quantity": "moment",
        "axis": "Y",
        "unit": "Nm",
        "channel": "Y_l~Mom.~BLD_1~pos~0.230~[Nm]",
        "max_abs": 10.0,
    }
    row.update(overrides)
    return row


def ultimate_margin_row(*, status: str = "ok", utilization: float = 0.5, allowable_abs: float = 20.0) -> dict:
    max_abs = 10.0
    load_factor = 1.0
    resistance_factor = 1.0
    factored_demand = max_abs * load_factor
    factored_allowable = allowable_abs * resistance_factor
    reserve_factor = factored_allowable / factored_demand if factored_demand else float("inf")
    return {
        "scenario": "vorlap",
        "case": "U05p00ms_A000p00deg",
        "wind_speed_mps": 5.0,
        "azimuth_deg": 0.0,
        "blade": "BLD_1",
        "position": 0.23,
        "quantity": "moment",
        "axis": "Y",
        "unit": "Nm",
        "channel": "Y_l~Mom.~BLD_1~pos~0.230~[Nm]",
        "max_abs": max_abs,
        "allowable_abs": allowable_abs,
        "load_factor": load_factor,
        "resistance_factor": resistance_factor,
        "factored_demand": factored_demand,
        "factored_allowable": factored_allowable,
        "utilization": utilization,
        "reserve_factor": reserve_factor,
        "ultimate_status": status,
    }


def ultimate_audit_args(tmp_path: Path, report_path: Path, *, fail_on_exceedance: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=tmp_path,
        scenarios="vorlap",
        ultimate_margins=report_path,
        require_ultimate_margins=True,
        ultimate_margin_scenarios="vorlap",
        ultimate_margin_tolerance=1.0e-9,
        fail_on_ultimate_exceedance=fail_on_exceedance,
    )


def ultimate_audit_settings_and_manifest(tmp_path: Path) -> tuple[dict, dict]:
    settings = {
        "wind_speeds": [5.0],
        "azimuths_deg": [0.0],
        "planned_cases": [{"wind_speed_mps": 5.0, "parked_azimuth_deg": 0.0}],
        "scenario_names": ["vorlap"],
    }
    manifest = {
        "cases": [
            {
                "label": "U05p00ms_A000p00deg",
                "wind_speed": 5.0,
                "azimuth_deg": 0.0,
            }
        ]
    }
    return settings, {"vorlap": manifest}


def write_ultimate_allowables(path: Path, *, allowable_abs: str = "20", load_factor: str = "1.5", resistance_factor: str = "0.9") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "blade",
                "position",
                "quantity",
                "axis",
                "unit",
                "allowable_abs",
                "allowable_label",
                "load_factor",
                "resistance_factor",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "blade": "BLD_1",
                "position": "0.23",
                "quantity": "moment",
                "axis": "Y",
                "unit": "Nm",
                "allowable_abs": allowable_abs,
                "allowable_label": "reviewed blade-root allowable",
                "load_factor": load_factor,
                "resistance_factor": resistance_factor,
                "notes": "unit-test basis",
            }
        )


def test_ultimate_input_basis_accepts_complete_manifest(tmp_path):
    allowables = tmp_path / "postprocess" / "ultimate_allowables.csv"
    write_ultimate_allowables(allowables)
    basis_path = tmp_path / "ultimate_input_basis.json"
    write_json(
        basis_path,
        {
            "basis_id": "ultimate-inputs-v1",
            "review_status": "reviewed",
            "allowables_artifact": str(allowables.relative_to(tmp_path)),
            "allowables_sha256": audit.sha256_file(allowables),
            "allowable_type": "local signed QBlade load allowable",
            "channel_mapping_basis": "selected critical local blade-root moment channel",
            "stress_recovery_or_local_load_basis": "screening local-load allowable, no material stress recovery",
            "load_factor_basis": "constant 1.5 factor from design review",
            "resistance_factor_basis": "constant 0.9 resistance factor from design review",
            "units_convention": "QBlade exported local load units",
        },
    )
    args = SimpleNamespace(run_root=tmp_path, ultimate_input_basis=None, require_ultimate_input_basis=True)
    issues: list = []

    report = audit.audit_ultimate_input_basis(args, issues)

    assert report == {
        "path": str(basis_path),
        "available": True,
        "basis_id": "ultimate-inputs-v1",
        "review_status": "reviewed",
        "allowables": {
            "path": str(allowables),
            "available": True,
            "rows": 1,
            "valid_rows": 1,
        },
    }
    assert not [issue for issue in issues if issue.severity == "error"]


def test_ultimate_input_basis_builder_uses_completed_allowables(tmp_path):
    allowables = tmp_path / "postprocess" / "ultimate_allowables.csv"
    write_ultimate_allowables(allowables)

    manifest = ultimate_basis_builder.build_manifest(
        tmp_path,
        allowables=allowables,
        basis_id="generated-ultimate-basis",
        allowable_type="reviewed local load allowable",
        channel_mapping_basis="allowables keyed by blade, position, quantity, axis, and unit",
        stress_recovery_or_local_load_basis="local load basis reviewed outside generator",
        load_factor_basis="allowables CSV load_factor column",
        resistance_factor_basis="allowables CSV resistance_factor column",
        units_convention="QBlade exported local load units",
    )
    write_json(tmp_path / "ultimate_input_basis.json", manifest)
    args = SimpleNamespace(run_root=tmp_path, ultimate_input_basis=None, require_ultimate_input_basis=True)
    issues: list[audit.AuditIssue] = []

    report = audit.audit_ultimate_input_basis(args, issues)

    assert manifest["basis_id"] == "generated-ultimate-basis"
    assert manifest["review_status"] == "draft_generated_from_allowables"
    assert manifest["allowables_artifact"] == "postprocess/ultimate_allowables.csv"
    assert manifest["allowables_sha256"] == audit.sha256_file(allowables)
    assert manifest["allowables_summary"] == {"rows": 1, "valid_rows": 1}
    assert report["review_status"] == "draft_generated_from_allowables"
    assert report["allowables"]["valid_rows"] == 1
    assert [issue.code for issue in issues if issue.severity == "error"] == ["ultimate_input_basis_not_reviewed"]


def test_ultimate_input_basis_rejects_draft_text_and_stale_allowables_hash(tmp_path):
    allowables = tmp_path / "postprocess" / "ultimate_allowables.csv"
    write_ultimate_allowables(allowables)
    write_json(
        tmp_path / "ultimate_input_basis.json",
        {
            "basis_id": "ultimate-inputs-v1",
            "review_status": "reviewed",
            "allowables_artifact": str(allowables.relative_to(tmp_path)),
            "allowables_sha256": "0" * 64,
            "allowable_type": "DRAFT: reviewed local QBlade load allowable",
            "channel_mapping_basis": "selected critical local blade-root moment channel",
            "stress_recovery_or_local_load_basis": "screening local-load allowable, no material stress recovery",
            "load_factor_basis": "constant 1.5 factor from design review",
            "resistance_factor_basis": "constant 0.9 resistance factor from design review",
            "units_convention": "QBlade exported local load units",
        },
    )
    args = SimpleNamespace(run_root=tmp_path, ultimate_input_basis=None, require_ultimate_input_basis=True)
    issues: list[audit.AuditIssue] = []

    audit.audit_ultimate_input_basis(args, issues)

    errors = {issue.code for issue in issues if issue.severity == "error"}
    assert {
        "ultimate_input_basis_draft_text",
        "ultimate_input_basis_allowables_hash_mismatch",
    } <= errors


def test_ultimate_input_basis_rejects_incomplete_manifest_and_allowables(tmp_path):
    allowables = tmp_path / "postprocess" / "ultimate_allowables.csv"
    write_ultimate_allowables(allowables, allowable_abs="0", load_factor="-1", resistance_factor="nan")
    write_json(
        tmp_path / "ultimate_input_basis.json",
        {
            "basis_id": "",
            "allowables_artifact": str(allowables.relative_to(tmp_path)),
            "allowable_type": "",
            "channel_mapping_basis": "",
            "stress_recovery_or_local_load_basis": "",
            "load_factor_basis": "",
            "resistance_factor_basis": "",
            "units_convention": "",
        },
    )
    args = SimpleNamespace(run_root=tmp_path, ultimate_input_basis=None, require_ultimate_input_basis=True)
    issues: list = []

    audit.audit_ultimate_input_basis(args, issues)

    errors = {issue.code for issue in issues if issue.severity == "error"}
    assert {
        "ultimate_input_basis_missing_id",
        "ultimate_input_basis_not_reviewed",
        "ultimate_input_basis_incomplete",
        "ultimate_input_basis_missing_allowables_hash",
        "ultimate_input_basis_nonfinite_allowable_value",
        "ultimate_input_basis_no_valid_allowables",
    } <= errors


def test_required_ultimate_input_basis_reports_missing_manifest(tmp_path):
    args = SimpleNamespace(run_root=tmp_path, ultimate_input_basis=None, require_ultimate_input_basis=True)
    issues: list = []

    report = audit.audit_ultimate_input_basis(args, issues)

    assert report == {"path": str(tmp_path / "ultimate_input_basis.json"), "available": False}
    assert [issue.code for issue in issues if issue.severity == "error"] == ["missing_ultimate_input_basis"]


def test_ultimate_margin_audit_accepts_valid_allowable_rows(tmp_path):
    report_path = tmp_path / "postprocess" / "ultimate_margin_metrics.csv"
    write_ultimate_margin_report(report_path, [ultimate_margin_row()])
    write_extreme_channel_report(tmp_path / "postprocess" / "extreme_channel_metrics.csv", [extreme_channel_row()])
    settings, manifests = ultimate_audit_settings_and_manifest(tmp_path)
    issues = []

    report = audit.audit_ultimate_margins(settings, manifests, ultimate_audit_args(tmp_path, report_path), issues)

    assert report["covered_cases"] == 1
    assert report["covered_channels"] == 1
    assert report["expected_channels"] == 1
    assert report["status_counts"] == {"ok": 1}
    assert not [issue for issue in issues if issue.severity == "error"]


def test_ultimate_margin_audit_rejects_missing_allowable_and_warns_on_exceedance(tmp_path):
    settings, manifests = ultimate_audit_settings_and_manifest(tmp_path)
    missing_path = tmp_path / "postprocess" / "ultimate_margin_missing.csv"
    write_ultimate_margin_report(
        missing_path,
        [ultimate_margin_row(status="missing_allowable", utilization=0.0, allowable_abs=0.0)],
    )
    write_extreme_channel_report(tmp_path / "postprocess" / "extreme_channel_metrics.csv", [extreme_channel_row()])
    issues = []

    audit.audit_ultimate_margins(settings, manifests, ultimate_audit_args(tmp_path, missing_path), issues)
    codes = {issue.code for issue in issues}

    assert "missing_ultimate_allowable" in codes
    assert "invalid_ultimate_margin_status" in codes

    exceed_path = tmp_path / "postprocess" / "ultimate_margin_exceed.csv"
    write_ultimate_margin_report(
        exceed_path,
        [ultimate_margin_row(status="exceeds_allowable", utilization=1.25, allowable_abs=8.0)],
    )
    issues = []

    report = audit.audit_ultimate_margins(settings, manifests, ultimate_audit_args(tmp_path, exceed_path), issues)

    assert report["status_counts"] == {"exceeds_allowable": 1}
    assert "ultimate_allowable_exceeded" in {issue.code for issue in issues if issue.severity == "warning"}
    assert not [issue for issue in issues if issue.severity == "error"]

    issues = []
    audit.audit_ultimate_margins(
        settings,
        manifests,
        ultimate_audit_args(tmp_path, exceed_path, fail_on_exceedance=True),
        issues,
    )

    assert "ultimate_allowable_exceeded" in {issue.code for issue in issues if issue.severity == "error"}


def test_ultimate_margin_audit_rejects_inconsistent_calculations(tmp_path):
    settings, manifests = ultimate_audit_settings_and_manifest(tmp_path)
    report_path = tmp_path / "postprocess" / "ultimate_margin_bad_math.csv"
    row = ultimate_margin_row()
    row["factored_demand"] = 999.0
    row["utilization"] = 0.5
    write_ultimate_margin_report(report_path, [row])
    write_extreme_channel_report(tmp_path / "postprocess" / "extreme_channel_metrics.csv", [extreme_channel_row()])
    issues = []

    audit.audit_ultimate_margins(settings, manifests, ultimate_audit_args(tmp_path, report_path), issues)
    codes = {issue.code for issue in issues if issue.severity == "error"}

    assert "ultimate_margin_calculation_inconsistent" in codes


def test_ultimate_margin_audit_rejects_partial_or_stale_extreme_channel_coverage(tmp_path):
    settings, manifests = ultimate_audit_settings_and_manifest(tmp_path)
    report_path = tmp_path / "postprocess" / "ultimate_margin_partial.csv"
    second_extreme = extreme_channel_row(
        quantity="force",
        axis="X",
        unit="N",
        channel="X_l~Force~BLD_1~pos~0.230~[N]",
        max_abs=12.0,
    )
    write_extreme_channel_report(
        tmp_path / "postprocess" / "extreme_channel_metrics.csv",
        [extreme_channel_row(), second_extreme],
    )
    write_ultimate_margin_report(report_path, [ultimate_margin_row()])
    issues = []

    report = audit.audit_ultimate_margins(settings, manifests, ultimate_audit_args(tmp_path, report_path), issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert report["expected_channels"] == 2
    assert report["covered_channels"] == 1
    assert "missing_ultimate_margin_channel" in errors

    stale_path = tmp_path / "postprocess" / "ultimate_margin_stale.csv"
    stale_row = ultimate_margin_row()
    stale_row["max_abs"] = 9.0
    stale_row["factored_demand"] = 9.0
    stale_row["utilization"] = 9.0 / stale_row["allowable_abs"]
    stale_row["reserve_factor"] = stale_row["allowable_abs"] / 9.0
    write_extreme_channel_report(tmp_path / "postprocess" / "extreme_channel_metrics.csv", [extreme_channel_row()])
    write_ultimate_margin_report(stale_path, [stale_row])
    issues = []

    audit.audit_ultimate_margins(settings, manifests, ultimate_audit_args(tmp_path, stale_path), issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert "ultimate_margin_extreme_mismatch" in errors


def write_postprocess_csv(path: Path, columns: tuple[str, ...], *, smoke_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {column: 1.0 for column in columns}
    for column in columns:
        if column in {"case", "scenario", "component", "blade", "quantity", "axis", "unit", "summary_file", "result_file"}:
            row[column] = "synthetic"
        elif column == "selection_reason":
            row[column] = "max_signed_del_ratio"
        elif column.startswith("channel_") or column == "postprocess_warnings":
            row[column] = ""
        elif column == "ultimate_status":
            row[column] = "ok"
        elif column == "smoke_only":
            row[column] = "true" if smoke_only else "false"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def write_postprocess_csv_with_updates(path: Path, columns: tuple[str, ...], updates: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {column: 1.0 for column in columns}
    for column in columns:
        if column in {"case", "scenario", "component", "blade", "quantity", "axis", "unit", "summary_file", "result_file"}:
            row[column] = "synthetic"
        elif column == "selection_reason":
            row[column] = "max_signed_del_ratio"
        elif column.startswith("channel_") or column == "postprocess_warnings":
            row[column] = ""
        elif column == "ultimate_status":
            row[column] = "ok"
        elif column == "smoke_only":
            row[column] = "false"
    row.update(updates)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def write_minimal_postprocess_artifacts(run_root: Path) -> None:
    for relative, kind in audit.POSTPROCESS_REQUIRED_ARTIFACTS:
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "rainflow_validation":
            path.write_text("case,passed\nconstant,True\n", encoding="utf-8")
        elif relative in audit.POSTPROCESS_PAPER_CSV_SCHEMAS:
            write_postprocess_csv(path, audit.POSTPROCESS_PAPER_CSV_SCHEMAS[relative])
        elif kind == "csv":
            path.write_text("column\nvalue\n", encoding="utf-8")
        elif path.suffix == ".png":
            path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic\n")
        else:
            path.write_text("synthetic artifact\n", encoding="utf-8")


def write_archive_artifact(run_root: Path, relative: str, category: str, content: bytes | str = "artifact\n") -> dict:
    path = run_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return {
        "category": category,
        "path": relative,
        "sha256": audit.sha256_file(path),
        "bytes": path.stat().st_size,
    }


def test_production_archive_manifest_builder_hashes_required_artifacts(tmp_path):
    settings, manifests, _args, source_path = source_coverage_audit_fixture(tmp_path)
    case_dir = Path(manifests["vorlap"]["cases"][0]["sim"]).parent
    result_dir = Path(manifests["vorlap"]["cases"][0]["result_dir"])
    (tmp_path / "audits").mkdir()
    write_json(tmp_path / "audits" / "results_audit.json", {"passed": True})
    (case_dir / "reference_model" / "Control" / "vorlap_qblade_debug.log").write_text("bridge log\n", encoding="utf-8")
    (tmp_path / "postprocess").mkdir()
    (tmp_path / "postprocess" / "figure_provenance.json").write_text('{"figures": []}\n', encoding="utf-8")
    source_path.write_text("scenario,case\nvorlap,U05p00ms_A000p00deg\n", encoding="utf-8")
    (result_dir / "baseline_reference_qblade.log").write_text("qblade log\n", encoding="utf-8")
    write_json(tmp_path / "dlc_manifest_row.json", {"row_id": "synthetic-row"})
    write_json(tmp_path / "wind_record.json", {"sha256": "synthetic-wind-hash"})
    write_json(tmp_path / "execution_environment_wind.json", {"phase": "wind"})
    write_json(tmp_path / "execution_environment_qblade.json", {"phase": "qblade"})
    manifest = archive_builder.build_manifest(tmp_path, archive_id="synthetic-production")
    archive_manifest = tmp_path / "production_archive_manifest.json"
    write_json(archive_manifest, manifest)
    args = SimpleNamespace(
        run_root=tmp_path,
        require_production_archive_manifest=True,
        production_archive_manifest=archive_manifest,
    )
    issues: list[audit.AuditIssue] = []

    report = audit.audit_production_archive_manifest(settings, manifests, args, issues)

    assert manifest["archive_id"] == "synthetic-production"
    assert manifest["missing_required_categories"] == []
    assert all(not str(item["path"]).startswith("/") for item in manifest["artifacts"])
    assert report["category_counts"]["audit_json"] == 1
    assert report["category_counts"]["bridge_log"] == 1
    assert report["category_counts"]["figure_provenance"] == 1
    assert report["category_counts"]["postprocess_csv"] == 1
    assert report["category_counts"]["qblade_log"] == 1
    assert report["category_counts"]["run_summary"] == 1
    assert manifest["category_counts"]["dlc_manifest_row"] == 1
    assert manifest["category_counts"]["execution_environment"] == 2
    assert manifest["category_counts"]["wind_record"] == 1
    assert not [issue for issue in issues if issue.severity == "error"]


def test_production_archive_manifest_accepts_complete_hashed_artifacts(tmp_path):
    settings, manifests, _args, _source_path = source_coverage_audit_fixture(tmp_path)
    archive_manifest = tmp_path / "production_archive_manifest.json"
    artifacts = [
        write_archive_artifact(tmp_path, "audits/results_audit.json", "audit_json"),
        write_archive_artifact(tmp_path, "vorlap/results/U05p00ms_A000p00deg/bridge.log", "bridge_log"),
        write_archive_artifact(tmp_path, "postprocess/figure_provenance.json", "figure_provenance"),
        write_archive_artifact(tmp_path, "postprocess/source_coverage.csv", "postprocess_csv", "scenario,case\nvorlap,U05\n"),
        write_archive_artifact(tmp_path, "vorlap/results/U05p00ms_A000p00deg/qblade.log", "qblade_log"),
        write_archive_artifact(
            tmp_path,
            "vorlap/results/U05p00ms_A000p00deg/baseline_reference_run_summary.json",
            "run_summary",
        ),
    ]
    write_json(archive_manifest, {"archive_id": "synthetic-production-archive", "artifacts": artifacts})
    args = SimpleNamespace(
        run_root=tmp_path,
        require_production_archive_manifest=True,
        production_archive_manifest=archive_manifest,
    )
    issues: list[audit.AuditIssue] = []

    report = audit.audit_production_archive_manifest(settings, manifests, args, issues)

    assert report["artifacts"] == 6
    assert report["expected_cases"] == 1
    assert report["expected_run_summaries"] == 1
    assert report["category_counts"]["run_summary"] == 1
    assert not [issue for issue in issues if issue.severity == "error"]


def test_production_archive_manifest_requires_expected_run_summary_paths(tmp_path):
    settings, manifests, _args, _source_path = source_coverage_audit_fixture(tmp_path)
    archive_manifest = tmp_path / "production_archive_manifest.json"
    artifacts = [
        write_archive_artifact(tmp_path, "audits/results_audit.json", "audit_json"),
        write_archive_artifact(tmp_path, "vorlap/results/U05p00ms_A000p00deg/bridge.log", "bridge_log"),
        write_archive_artifact(tmp_path, "postprocess/figure_provenance.json", "figure_provenance"),
        write_archive_artifact(tmp_path, "postprocess/source_coverage.csv", "postprocess_csv", "scenario,case\nvorlap,U05\n"),
        write_archive_artifact(tmp_path, "vorlap/results/U05p00ms_A000p00deg/qblade.log", "qblade_log"),
        write_archive_artifact(tmp_path, "summaries/unrelated_run_summary.json", "run_summary"),
    ]
    write_json(archive_manifest, {"archive_id": "wrong-summary-archive", "artifacts": artifacts})
    args = SimpleNamespace(
        run_root=tmp_path,
        require_production_archive_manifest=True,
        production_archive_manifest=archive_manifest,
    )
    issues: list[audit.AuditIssue] = []

    report = audit.audit_production_archive_manifest(settings, manifests, args, issues)

    assert report["expected_cases"] == 1
    assert report["expected_run_summaries"] == 1
    assert report["category_counts"]["run_summary"] == 1
    assert [issue.code for issue in issues if issue.severity == "error"] == [
        "production_archive_manifest_missing_expected_run_summary"
    ]


def test_production_archive_manifest_rejects_missing_categories_and_bad_hashes(tmp_path):
    settings, manifests, _args, _source_path = source_coverage_audit_fixture(tmp_path)
    archive_manifest = tmp_path / "production_archive_manifest.json"
    artifact = write_archive_artifact(tmp_path, "summaries/case_run_summary.json", "run_summary")
    artifact["sha256"] = "0" * 64
    artifact["bytes"] = artifact["bytes"] + 1
    write_json(archive_manifest, {"archive_id": "bad-production-archive", "artifacts": [artifact]})
    args = SimpleNamespace(
        run_root=tmp_path,
        require_production_archive_manifest=True,
        production_archive_manifest=archive_manifest,
    )
    issues: list[audit.AuditIssue] = []

    audit.audit_production_archive_manifest(settings, manifests, args, issues)
    errors = {issue.code for issue in issues if issue.severity == "error"}

    assert {
        "production_archive_artifact_sha256_mismatch",
        "production_archive_artifact_size_mismatch",
        "production_archive_manifest_missing_expected_run_summary",
        "production_archive_manifest_missing_category",
    } <= errors


def test_results_audit_can_require_postprocess_artifact_bundle(tmp_path):
    args = SimpleNamespace(run_root=tmp_path, require_postprocess_artifacts=True)
    issues = []

    missing = audit.audit_postprocess_artifacts(args, issues)

    assert missing["checked"] == len(audit.POSTPROCESS_REQUIRED_ARTIFACTS)
    assert "missing_postprocess_artifact" in {issue.code for issue in issues}

    write_minimal_postprocess_artifacts(tmp_path)
    issues = []
    report = audit.audit_postprocess_artifacts(args, issues)

    assert report["checked"] == len(audit.POSTPROCESS_REQUIRED_ARTIFACTS)
    assert not [issue for issue in issues if issue.severity == "error"]
    assert report["row_counts"]["postprocess/rainflow_validation.csv"] == 1


def test_results_audit_rejects_paper_fatigue_placeholder_or_smoke_rows(tmp_path):
    write_minimal_postprocess_artifacts(tmp_path)
    (tmp_path / "postprocess" / "signed_del_comparison.csv").write_text("column\nvalue\n", encoding="utf-8")
    args = SimpleNamespace(run_root=tmp_path, require_postprocess_artifacts=True)
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "postprocess_artifact_schema_mismatch" in {issue.code for issue in issues if issue.severity == "error"}

    write_minimal_postprocess_artifacts(tmp_path)
    write_postprocess_csv(
        tmp_path / "postprocess" / "signed_del_comparison.csv",
        audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_del_comparison.csv"],
        smoke_only=True,
    )
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "postprocess_paper_artifact_contains_smoke_only" in {issue.code for issue in issues if issue.severity == "error"}

    args.allow_diagnostic_postprocess_artifacts = True
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "postprocess_paper_artifact_contains_smoke_only" not in {
        issue.code for issue in issues if issue.severity == "error"
    }


def test_results_audit_rejects_warnings_or_nonfinite_values_in_paper_artifacts(tmp_path):
    args = SimpleNamespace(run_root=tmp_path, require_postprocess_artifacts=True)
    write_minimal_postprocess_artifacts(tmp_path)
    write_postprocess_csv_with_updates(
        tmp_path / "postprocess" / "signed_del_comparison.csv",
        audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_del_comparison.csv"],
        {"postprocess_warnings": "cycles<10"},
    )
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "postprocess_paper_artifact_contains_warnings" in {issue.code for issue in issues if issue.severity == "error"}

    args.allow_diagnostic_postprocess_artifacts = True
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "postprocess_paper_artifact_contains_warnings" not in {
        issue.code for issue in issues if issue.severity == "error"
    }

    write_minimal_postprocess_artifacts(tmp_path)
    write_postprocess_csv_with_updates(
        tmp_path / "postprocess" / "critical_case_summary.csv",
        audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/critical_case_summary.csv"],
        {"signed_del_vorlap": "nan"},
    )
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "postprocess_paper_artifact_nonfinite_value" in {issue.code for issue in issues if issue.severity == "error"}


def test_results_audit_rejects_stale_critical_case_selection(tmp_path):
    write_minimal_postprocess_artifacts(tmp_path)
    signed_columns = audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_del_comparison.csv"]
    critical_columns = audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/critical_case_summary.csv"]
    lower_ratio_row = {
        "case": "case-low",
        "wind_speed_mps": 8.0,
        "azimuth_deg": 0.0,
        "position": 0.5,
        "quantity": "Force",
        "axis": "X",
        "unit": "N",
        "channel_baseline": "base-low",
        "channel_vorlap": "vorlap-low",
        "signed_del_baseline": 10.0,
        "signed_del_vorlap": 12.0,
        "signed_delta_pct": 20.0,
        "rainflow_cycles_baseline": 100.0,
        "rainflow_cycles_vorlap": 100.0,
        "smoke_only": "false",
        "postprocess_warnings": "",
    }
    higher_ratio_row = {
        "case": "case-high",
        "wind_speed_mps": 10.0,
        "azimuth_deg": 90.0,
        "position": 0.5,
        "quantity": "Force",
        "axis": "Y",
        "unit": "N",
        "channel_baseline": "base-high",
        "channel_vorlap": "vorlap-high",
        "signed_del_baseline": 8.0,
        "signed_del_vorlap": 16.0,
        "signed_delta_pct": 100.0,
        "rainflow_cycles_baseline": 120.0,
        "rainflow_cycles_vorlap": 120.0,
        "smoke_only": "false",
        "postprocess_warnings": "",
    }
    with (tmp_path / "postprocess" / "signed_del_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=signed_columns)
        writer.writeheader()
        writer.writerow(lower_ratio_row)
        writer.writerow(higher_ratio_row)
    stale_critical = {
        **{column: lower_ratio_row.get(column, "") for column in critical_columns},
        "signed_del_ratio": 1.2,
        "selection_reason": "max_signed_del_ratio",
        "min_nyquist_ratio": 1.0,
        "max_force_coefficient_ratio": 1.0,
        "force_power_max_abs_w": 1.0,
        "source_coverage_violation_count": 0,
    }
    with (tmp_path / "postprocess" / "critical_case_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=critical_columns)
        writer.writeheader()
        writer.writerow(stale_critical)
    args = SimpleNamespace(run_root=tmp_path, require_postprocess_artifacts=True)
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "postprocess_critical_case_selection_mismatch" in {issue.code for issue in issues if issue.severity == "error"}


def test_results_audit_requires_signed_table_subset_of_diagnostics(tmp_path):
    write_minimal_postprocess_artifacts(tmp_path)
    signed_columns = audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_del_comparison.csv"]
    paper_row = {
        "case": "case-paper",
        "wind_speed_mps": 8.0,
        "azimuth_deg": 0.0,
        "position": 0.5,
        "quantity": "Force",
        "axis": "X",
        "unit": "N",
        "channel_baseline": "base-paper",
        "channel_vorlap": "vorlap-paper",
        "signed_del_baseline": 10.0,
        "signed_del_vorlap": 12.0,
        "signed_delta_pct": 20.0,
        "rainflow_cycles_baseline": 100.0,
        "rainflow_cycles_vorlap": 100.0,
        "smoke_only": "false",
        "postprocess_warnings": "",
    }
    diagnostic_row = {**paper_row, "case": "case-diagnostic-only"}
    with (tmp_path / "postprocess" / "signed_del_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=signed_columns)
        writer.writeheader()
        writer.writerow(paper_row)
    with (tmp_path / "postprocess" / "signed_del_comparison_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=signed_columns)
        writer.writeheader()
        writer.writerow(diagnostic_row)
    args = SimpleNamespace(run_root=tmp_path, require_postprocess_artifacts=True)
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "postprocess_signed_table_not_in_diagnostics" in {issue.code for issue in issues if issue.severity == "error"}


def test_results_audit_rejects_baseline_identity_drift_above_threshold(tmp_path):
    write_minimal_postprocess_artifacts(tmp_path)
    write_postprocess_csv_with_updates(
        tmp_path / "postprocess" / "baseline_identity_comparison.csv",
        audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/baseline_identity_comparison.csv"],
        {
            "force_identity_delta_pct": 7.5,
            "moment_identity_delta_pct": 1.0,
        },
    )
    write_postprocess_csv_with_updates(
        tmp_path / "postprocess" / "signed_baseline_identity_comparison.csv",
        audit.POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_baseline_identity_comparison.csv"],
        {
            "signed_identity_delta_pct": -6.25,
        },
    )
    args = SimpleNamespace(
        run_root=tmp_path,
        require_postprocess_artifacts=True,
        max_baseline_identity_delta_pct=5.0,
        max_signed_baseline_identity_delta_pct=5.0,
    )
    issues = []

    report = audit.audit_postprocess_artifacts(args, issues)

    errors = {issue.code for issue in issues if issue.severity == "error"}
    assert "baseline_identity_delta_exceeds_tolerance" in errors
    assert "signed_baseline_identity_delta_exceeds_tolerance" in errors
    identity_report = report["baseline_identity_drift"]
    assert math.isclose(identity_report["postprocess/baseline_identity_comparison.csv"]["max_abs_delta_pct"], 7.5)
    assert math.isclose(identity_report["postprocess/signed_baseline_identity_comparison.csv"]["max_abs_delta_pct"], 6.25)


def test_results_audit_can_require_standalone_baseline_identity_proof(tmp_path):
    args = SimpleNamespace(
        run_root=tmp_path,
        max_baseline_identity_delta_pct=5.0,
        max_signed_baseline_identity_delta_pct=5.0,
    )
    issues = []

    missing = audit.audit_baseline_identity_drift(tmp_path, args, issues, require_artifacts=True)

    assert missing is None
    assert [issue.code for issue in issues if issue.severity == "error"] == [
        "missing_baseline_identity_artifact",
        "missing_baseline_identity_artifact",
    ]

    write_minimal_postprocess_artifacts(tmp_path)
    issues = []
    report = audit.audit_baseline_identity_drift(tmp_path, args, issues, require_artifacts=True)

    assert report is not None
    assert set(report) == {
        "postprocess/baseline_identity_comparison.csv",
        "postprocess/signed_baseline_identity_comparison.csv",
    }
    assert not [issue for issue in issues if issue.severity == "error"]


def checkpoint_artifact_fixture(tmp_path):
    result_dir = tmp_path / "baseline" / "results" / "U05p00ms_A000p00deg"
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    identity = {
        "fingerprint": "checkpoint-identity",
        "runtime_asset_fingerprint": "runtime-assets",
        "vorlap_python_runtime_fingerprint": "python-runtime",
    }
    latest_metadata = None
    for step in (250, 500):
        project = checkpoint_dir / f"case_step_{step:07d}.qpr"
        project.write_text(f"checkpoint {step}\n", encoding="utf-8")
        project_sha256 = audit.sha256_file(project)
        metadata = {
            "version": 3,
            "requested_steps": 500,
            "completed_steps": step,
            "coupling_mode": "none",
            "qblade_project": str(project),
            "qblade_project_sha256": project_sha256,
            "qblade_project_size_bytes": project.stat().st_size,
            "checkpoint_identity": identity,
            "checkpoint_identity_fingerprint": identity["fingerprint"],
        }
        write_json(checkpoint_dir / f"case_step_{step:07d}.json", metadata)
        latest_metadata = metadata
    assert latest_metadata is not None
    write_json(
        checkpoint_dir / "latest_checkpoint.json",
        {
            **latest_metadata,
            "_metadata_path": str(checkpoint_dir / "case_step_0000500.json"),
        },
    )
    completed_project = result_dir / "case_completed.qpr"
    completed_project.write_text("completed\n", encoding="utf-8")
    summary = {
        "requested_steps": 500,
        "completed_steps": 500,
        "coupling_mode": "none",
        "stored_project": str(completed_project),
        "restart": {
            "checkpoint_every": 250,
            "checkpoint_dir": str(checkpoint_dir),
            "resume_requested": True,
            "resume_checkpoint": None,
            "resume_start_step": 0,
            "resume_rejection": None,
            "latest_checkpoint": str(checkpoint_dir / "case_step_0000500.json"),
            "latest_checkpoint_completed_steps": 500,
            "latest_checkpoint_project": str(checkpoint_dir / "case_step_0000500.qpr"),
            "checkpoint_identity_fingerprint": identity["fingerprint"],
        },
        "reproducibility": {
            "runtime_asset_fingerprint": identity["runtime_asset_fingerprint"],
            "vorlap_python_runtime_fingerprint": identity["vorlap_python_runtime_fingerprint"],
        },
    }
    settings = {"checkpoint_every": 250, "store_project": True}
    return result_dir, checkpoint_dir, summary, settings


def test_results_audit_accepts_complete_checkpoint_and_completed_project_evidence(tmp_path):
    result_dir, _checkpoint_dir, summary, settings = checkpoint_artifact_fixture(tmp_path)
    issues = []

    audit.audit_checkpoint_and_project_artifacts(
        summary,
        settings,
        SimpleNamespace(require_fresh_run=True),
        issues,
        result_dir=result_dir,
        scenario="baseline",
        case_label="U05p00ms_A000p00deg",
    )

    assert not [item for item in issues if item.severity == "error"]


def test_results_audit_rejects_missing_or_incompatible_checkpoint_evidence(tmp_path):
    result_dir, checkpoint_dir, summary, settings = checkpoint_artifact_fixture(tmp_path)
    Path(summary["stored_project"]).unlink()
    (checkpoint_dir / "case_step_0000250.qpr").unlink()
    latest_metadata_path = checkpoint_dir / "case_step_0000500.json"
    latest_metadata = json.loads(latest_metadata_path.read_text(encoding="utf-8"))
    latest_metadata["completed_steps"] = "not-a-step"
    latest_metadata["checkpoint_identity_fingerprint"] = "stale-identity"
    write_json(latest_metadata_path, latest_metadata)
    issues = []

    audit.audit_checkpoint_and_project_artifacts(
        summary,
        settings,
        SimpleNamespace(require_fresh_run=True),
        issues,
        result_dir=result_dir,
        scenario="baseline",
        case_label="U05p00ms_A000p00deg",
    )

    error_codes = {item.code for item in issues if item.severity == "error"}
    assert "missing_completed_qblade_project" in error_codes
    assert "checkpoint_completed_steps_mismatch" in error_codes
    assert "checkpoint_identity_mismatch" in error_codes
    assert "checkpoint_retention_mismatch" in error_codes


def test_results_audit_rejects_modified_checkpoint_project(tmp_path):
    result_dir, checkpoint_dir, summary, settings = checkpoint_artifact_fixture(tmp_path)
    project = checkpoint_dir / "case_step_0000500.qpr"
    project.write_text("tampered project\n", encoding="utf-8")
    issues = []

    audit.audit_checkpoint_and_project_artifacts(
        summary,
        settings,
        SimpleNamespace(require_fresh_run=True),
        issues,
        result_dir=result_dir,
        scenario="baseline",
        case_label="U05p00ms_A000p00deg",
    )

    error_codes = {item.code for item in issues if item.severity == "error"}
    assert "checkpoint_project_binding_mismatch" in error_codes
    assert "retained_checkpoint_project_binding_mismatch" in error_codes
    assert "checkpoint_retention_mismatch" in error_codes


def test_results_audit_rejects_resume_state_in_fresh_trial(tmp_path):
    result_dir, _checkpoint_dir, summary, settings = checkpoint_artifact_fixture(tmp_path)
    summary["restart"]["resume_requested"] = True
    summary["restart"]["resume_start_step"] = 250
    issues = []

    audit.audit_checkpoint_and_project_artifacts(
        summary,
        settings,
        SimpleNamespace(require_fresh_run=True),
        issues,
        result_dir=result_dir,
        scenario="baseline",
        case_label="U05p00ms_A000p00deg",
    )

    assert "run_not_fresh" in {item.code for item in issues if item.severity == "error"}


def test_results_audit_rejects_empty_standalone_baseline_identity_artifacts(tmp_path):
    for relative in (
        "postprocess/baseline_identity_comparison.csv",
        "postprocess/signed_baseline_identity_comparison.csv",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=audit.POSTPROCESS_PAPER_CSV_SCHEMAS[relative]).writeheader()
    args = SimpleNamespace(
        run_root=tmp_path,
        max_baseline_identity_delta_pct=5.0,
        max_signed_baseline_identity_delta_pct=5.0,
    )
    issues: list[audit.AuditIssue] = []

    report = audit.audit_baseline_identity_drift(tmp_path, args, issues, require_artifacts=True)
    errors = [issue.code for issue in issues if issue.severity == "error"]

    assert report is not None
    assert errors == [
        "baseline_identity_artifact_empty",
        "baseline_identity_artifact_empty",
    ]
    assert report["postprocess/baseline_identity_comparison.csv"]["rows"] == 0
    assert report["postprocess/signed_baseline_identity_comparison.csv"]["rows"] == 0


def test_results_audit_reports_standalone_baseline_identity_proof(tmp_path):
    for relative in (
        "postprocess/baseline_identity_comparison.csv",
        "postprocess/signed_baseline_identity_comparison.csv",
    ):
        write_postprocess_csv(tmp_path / relative, audit.POSTPROCESS_PAPER_CSV_SCHEMAS[relative])
    cases: dict[str, dict[str, object]] = {}
    for scenario in ("baseline",):
        case_dir = tmp_path / scenario / "cases" / "U05p00ms_A000p00deg"
        result_dir = tmp_path / scenario / "results" / "U05p00ms_A000p00deg"
        sim = case_dir / "baseline_reference.sim"
        result_file = result_dir / "U05p00ms_A000p00deg_qblade_results.txt"
        sim.parent.mkdir(parents=True)
        result_dir.mkdir(parents=True)
        sim.write_text("synthetic\n", encoding="utf-8")
        result_file.write_text("Time~[s]\n0\n600\n", encoding="utf-8")
        write_geometry_audit(case_dir)
        cases[scenario] = {
            "index": 0,
            "label": "U05p00ms_A000p00deg",
            "wind_speed": 5.0,
            "azimuth_deg": 0.0,
            "sim": str(sim),
            "result_dir": str(result_dir),
            "result_name": "U05p00ms_A000p00deg_qblade_results",
        }
        write_json(
            result_dir / "baseline_reference_run_summary.json",
            {
                "requested_steps": 80000,
                "completed_steps": 80000,
                "failed": False,
                "exported_results": True,
                "result_file": str(result_file),
                "coupling_mode": "none",
                "reproducibility": {
                    "qblade_case_settings": {
                        "NUMTIMESTEPS": "80000",
                        "TIMESTEP": "0.0075",
                        "PRESCRIBETYPE": "1",
                        "RPMPRESCRIBED": "0",
                        "MEANINF": "5",
                            "INITIAL_AZIMUTH": "0",
                            "WNDTYPE": "0",
                            "HORANGLE": "0",
                            "VERTANGLE": "0",
                            "SHEAREXP": "0.23",
                            "REFHEIGHT": "18",
                            "DENSITYAIR": "1.0296",
                            "VISCOSITYAIR": "0.00001647",
                        },
                        "case_metadata": {"case": {"label": "U05p00ms_A000p00deg"}},
                        "runtime_assets": [{"path": str(sim), "sha256": "synthetic"}],
                        "runtime_asset_fingerprint": "synthetic-runtime-assets",
                    },
                    "qblade_export_rpm_diagnostics": {
                        "available": True,
                        "samples": 2,
                        "start_time_s": 0.1,
                        "expected_rpm": 0.0,
                        "max_abs_error_rpm": 0.0,
                    },
                    "native_callback_evidence": {"update_init_count": 0, "update_count": 0},
                },
        )
        write_json(
            tmp_path / scenario / "cases.json",
            {
                "settings": {
            "rpm": 0.0,
            "rpm_prescribe_type": 1,
            "rpm_audit_tolerance_rpm": 0.01,
            "rpm_audit_startup_allowance_s": 0.1,
            "horizontal_inflow_angle_deg": 0.0,
            "vertical_inflow_angle_deg": 0.0,
            "shear_exponent": 0.23,
            "reference_height_m": 18.0,
            "air_density_kg_m3": 1.0296,
            "air_kinematic_viscosity_m2_s": 1.647e-5,
                    "timesteps": 80000,
                    "timestep": 0.0075,
                    "force_scale": 0.0,
                    "coupling": "none",
                    "stable_support_patch": False,
                },
                "cases": [cases[scenario]],
            },
        )
    settings = {
        "wind_speeds": [5.0],
        "azimuths_deg": [0.0],
        "planned_cases": [{"wind_speed_mps": 5.0, "parked_azimuth_deg": 0.0}],
        "planned_case_count_per_scenario": 1,
        "planned_case_count_total": 1,
        "scenario_names": ["baseline"],
        "scenario_force_scales": {"baseline": 0.0},
        "scenario_couplings": {"baseline": "none"},
        "duration_s": 600.0,
        "timestep_s": 0.0075,
        "timesteps": 80000,
        "rpm": 0.0,
    }
    write_json(tmp_path / "campaign_settings.json", settings)
    args = SimpleNamespace(
        run_root=tmp_path,
        stage="results",
        scenarios="baseline",
        min_duration=600.0,
        max_timestep=0.01,
        expected_rpm=0.0,
        require_wind_name=False,
        allow_quasi_static=False,
        require_baseline_identity_proof=True,
        max_baseline_identity_delta_pct=5.0,
        max_signed_baseline_identity_delta_pct=5.0,
        source_coverage=None,
        require_source_coverage=False,
        source_coverage_scenarios="baseline",
        min_nyquist_ratio=1.0,
        allow_source_coverage_violations=0,
        source_range_tolerance=1.0e-9,
    )

    report = audit.audit_campaign(args)

    assert report["passed"] is True
    assert "baseline_identity_proof" in report
    assert report["baseline_identity_proof"]["postprocess/baseline_identity_comparison.csv"]["rows"] == 1

    args.stage = "runtime"
    runtime_report = audit.audit_campaign(args)
    assert runtime_report["passed"] is True
    assert runtime_report["stage"] == "runtime"
    args.stage = "results"

    summary_path = (
        tmp_path
        / "baseline"
        / "results"
        / "U05p00ms_A000p00deg"
        / "baseline_reference_run_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["native_callback_evidence"]["update_init_count"] = 1
    write_json(summary_path, summary)
    native_report = audit.audit_campaign(args)
    assert "unexpected_native_callback_activity" in {
        row["code"] for row in native_report["issues"] if row["severity"] == "error"
    }

    summary["native_callback_evidence"]["update_init_count"] = 0
    summary["qblade_export_rpm_diagnostics"]["max_abs_error_rpm"] = 0.5
    write_json(summary_path, summary)
    rpm_report = audit.audit_campaign(args)
    assert "exported_rpm_not_prescribed" in {
        row["code"] for row in rpm_report["issues"] if row["severity"] == "error"
    }

    manifest_path = tmp_path / "baseline" / "cases.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["settings"]["rpm_prescribe_type"] = 0
    write_json(manifest_path, manifest)
    settings["rpm_prescribe_type"] = 0
    write_json(tmp_path / "campaign_settings.json", settings)
    summary.pop("qblade_export_rpm_diagnostics")
    write_json(summary_path, summary)
    non_fixed_speed_report = audit.audit_campaign(args)
    assert "missing_exported_rpm_diagnostics" not in {
        row["code"] for row in non_fixed_speed_report["issues"] if row["severity"] == "error"
    }

    settings["qblade_steps_override"] = 400
    write_json(tmp_path / "campaign_settings.json", settings)
    summary["requested_steps"] = 400
    summary["completed_steps"] = 400
    write_json(summary_path, summary)
    step_limited_report = audit.audit_campaign(args)
    assert "requested_steps_mismatch" not in {
        row["code"] for row in step_limited_report["issues"] if row["severity"] == "error"
    }


def test_results_audit_rejects_critical_rainflow_cycle_mismatch(tmp_path):
    write_minimal_postprocess_artifacts(tmp_path)
    (tmp_path / "postprocess" / "critical_case_rainflow_ranges.csv").write_text("range,count\n1.0,0.5\n", encoding="utf-8")
    args = SimpleNamespace(run_root=tmp_path, require_postprocess_artifacts=True)
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "postprocess_critical_rainflow_cycle_mismatch" in {issue.code for issue in issues if issue.severity == "error"}


def test_results_audit_rejects_placeholder_raw_fatigue_or_ultimate_artifacts(tmp_path):
    args = SimpleNamespace(run_root=tmp_path, require_postprocess_artifacts=True)
    for relative in ("postprocess/signed_channel_metrics.csv", "postprocess/extreme_channel_metrics.csv"):
        write_minimal_postprocess_artifacts(tmp_path)
        (tmp_path / relative).write_text("column\nvalue\n", encoding="utf-8")
        issues = []

        audit.audit_postprocess_artifacts(args, issues)

        assert "postprocess_artifact_schema_mismatch" in {issue.code for issue in issues if issue.severity == "error"}


def test_results_audit_rejects_failed_rainflow_validation_artifact(tmp_path):
    write_minimal_postprocess_artifacts(tmp_path)
    (tmp_path / "postprocess" / "rainflow_validation.csv").write_text("case,passed\nclosed_cycle,False\n", encoding="utf-8")
    args = SimpleNamespace(run_root=tmp_path, require_postprocess_artifacts=True)
    issues = []

    audit.audit_postprocess_artifacts(args, issues)

    assert "rainflow_validation_failed" in {issue.code for issue in issues if issue.severity == "error"}


def test_results_audit_rejects_qblade_external_without_callback_evidence(tmp_path):
    settings = {
        "wind_speeds": [5.0],
        "azimuths_deg": [0.0],
        "planned_cases": [{"wind_speed_mps": 5.0, "parked_azimuth_deg": 0.0}],
        "planned_case_count_per_scenario": 1,
        "planned_case_count_total": 1,
        "scenario_names": ["vorlap"],
        "scenario_force_scales": {"vorlap": 1.0},
        "scenario_couplings": {"vorlap": "qblade-external"},
        "duration_s": 600.0,
        "timestep_s": 0.0075,
        "timesteps": 80000,
        "rpm": 0.0,
    }
    write_json(tmp_path / "campaign_settings.json", settings)
    case_dir = tmp_path / "vorlap" / "cases" / "U05p00ms_A000p00deg"
    result_dir = tmp_path / "vorlap" / "results" / "U05p00ms_A000p00deg"
    sim = case_dir / "baseline_reference.sim"
    result_file = result_dir / "U05p00ms_A000p00deg_qblade_results.txt"
    sim.parent.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    sim.write_text("synthetic\n", encoding="utf-8")
    write_geometry_audit(case_dir)
    result_file.write_text("Time~[s]\n0\n", encoding="utf-8")
    case = {
        "index": 0,
        "label": "U05p00ms_A000p00deg",
        "wind_speed": 5.0,
        "azimuth_deg": 0.0,
        "sim": str(sim),
        "result_dir": str(result_dir),
        "result_name": "U05p00ms_A000p00deg_qblade_results",
    }
    write_json(tmp_path / "vorlap" / "cases.json", {"settings": {"rpm": 0.0, "timesteps": 80000, "timestep": 0.0075, "force_scale": 1.0, "coupling": "qblade-external"}, "cases": [case]})
    write_json(
        result_dir / "baseline_reference_run_summary.json",
        {
            "requested_steps": 80000,
            "completed_steps": 80000,
            "failed": False,
            "exported_results": True,
            "result_file": str(result_file),
            "coupling_mode": "qblade-external",
            "native_callback_evidence": {"update_init_seen": False, "update_count_observed_max": 0},
            "reproducibility": {
                "qblade_case_settings": {
                    "NUMTIMESTEPS": "80000",
                    "TIMESTEP": "0.0075",
                    "PRESCRIBETYPE": "1",
                    "RPMPRESCRIBED": "0",
                    "MEANINF": "5",
                    "INITIAL_AZIMUTH": "0",
                },
                "case_metadata": {"case": {"label": "U05p00ms_A000p00deg"}},
            },
        },
    )
    args = SimpleNamespace(
        run_root=tmp_path,
        stage="results",
        scenarios="vorlap",
        min_duration=600.0,
        max_timestep=0.01,
        expected_rpm=0.0,
        require_wind_name=False,
        allow_quasi_static=False,
        source_coverage=None,
        require_source_coverage=False,
        source_coverage_scenarios="vorlap",
        min_nyquist_ratio=1.0,
        allow_source_coverage_violations=0,
        source_range_tolerance=1.0e-9,
    )

    report = audit.audit_campaign(args)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["passed"]
    assert "native_update_init_missing" in codes
    assert "native_update_count_missing" in codes
    assert "native_runtime_init_missing" in codes


def test_results_audit_rejects_qblade_external_without_python_import_diagnostics(tmp_path):
    settings = {
        "wind_speeds": [5.0],
        "azimuths_deg": [0.0],
        "planned_cases": [{"wind_speed_mps": 5.0, "parked_azimuth_deg": 0.0}],
        "planned_case_count_per_scenario": 1,
        "planned_case_count_total": 1,
        "scenario_names": ["vorlap"],
        "scenario_force_scales": {"vorlap": 1.0},
        "scenario_couplings": {"vorlap": "qblade-external"},
        "duration_s": 600.0,
        "timestep_s": 0.0075,
        "timesteps": 80000,
        "rpm": 0.0,
    }
    write_json(tmp_path / "campaign_settings.json", settings)
    case_dir = tmp_path / "vorlap" / "cases" / "U05p00ms_A000p00deg"
    result_dir = tmp_path / "vorlap" / "results" / "U05p00ms_A000p00deg"
    sim = case_dir / "baseline_reference.sim"
    result_file = result_dir / "U05p00ms_A000p00deg_qblade_results.txt"
    sim.parent.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    sim.write_text("synthetic\n", encoding="utf-8")
    write_geometry_audit(case_dir)
    result_file.write_text("Time~[s]\n0\n", encoding="utf-8")
    case = {
        "index": 0,
        "label": "U05p00ms_A000p00deg",
        "wind_speed": 5.0,
        "azimuth_deg": 0.0,
        "sim": str(sim),
        "result_dir": str(result_dir),
        "result_name": "U05p00ms_A000p00deg_qblade_results",
    }
    write_json(
        tmp_path / "vorlap" / "cases.json",
        {
            "settings": {"rpm": 0.0, "timesteps": 80000, "timestep": 0.0075, "force_scale": 1.0, "coupling": "qblade-external"},
            "cases": [case],
        },
    )
    write_json(
        result_dir / "baseline_reference_run_summary.json",
        {
            "requested_steps": 80000,
            "completed_steps": 80000,
            "failed": False,
            "exported_results": True,
            "result_file": str(result_file),
            "coupling_mode": "qblade-external",
            "native_callback_evidence": {
                "update_init_seen": True,
                "update_count_observed_max": 80000,
                "runtime_initialized_message_seen": True,
            },
            "reproducibility": {
                "qblade_case_settings": {
                    "NUMTIMESTEPS": "80000",
                    "TIMESTEP": "0.0075",
                    "PRESCRIBETYPE": "1",
                    "RPMPRESCRIBED": "0",
                    "MEANINF": "5",
                    "INITIAL_AZIMUTH": "0",
                },
                "case_metadata": {"case": {"label": "U05p00ms_A000p00deg"}},
            },
        },
    )
    args = SimpleNamespace(
        run_root=tmp_path,
        stage="results",
        scenarios="vorlap",
        min_duration=600.0,
        max_timestep=0.01,
        expected_rpm=0.0,
        require_wind_name=False,
        allow_quasi_static=False,
        source_coverage=None,
        require_source_coverage=False,
        source_coverage_scenarios="vorlap",
        min_nyquist_ratio=1.0,
        allow_source_coverage_violations=0,
        source_range_tolerance=1.0e-9,
    )

    report = audit.audit_campaign(args)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["passed"]
    assert "native_update_init_missing" not in codes
    assert "native_update_count_missing" not in codes
    assert "native_runtime_init_missing" not in codes
    assert "native_python_env_before_import_missing" in codes
    assert "native_python_env_after_import_missing" in codes
    assert "native_numpy_diagnostics_missing" in codes
    assert "native_runtime_path_missing" in codes


def test_setup_audit_rejects_missing_geometry_audit(tmp_path):
    settings = {
        "wind_speeds": [5.0],
        "azimuths_deg": [0.0],
        "planned_cases": [{"wind_speed_mps": 5.0, "parked_azimuth_deg": 0.0}],
        "planned_case_count_per_scenario": 1,
        "planned_case_count_total": 1,
        "scenario_names": ["vorlap"],
        "scenario_force_scales": {"vorlap": 1.0},
        "scenario_couplings": {"vorlap": "sil-vorlap"},
        "duration_s": 600.0,
        "timestep_s": 0.0075,
        "timesteps": 80000,
        "rpm": 0.0,
    }
    write_json(tmp_path / "campaign_settings.json", settings)
    case_dir = tmp_path / "vorlap" / "cases" / "U05p00ms_A000p00deg"
    sim = case_dir / "baseline_reference.sim"
    sim.parent.mkdir(parents=True)
    sim.write_text("synthetic\n", encoding="utf-8")
    case = {
        "index": 0,
        "label": "U05p00ms_A000p00deg",
        "wind_speed": 5.0,
        "azimuth_deg": 0.0,
        "sim": str(sim),
        "result_dir": str(tmp_path / "vorlap" / "results" / "U05p00ms_A000p00deg"),
    }
    write_json(tmp_path / "vorlap" / "cases.json", {"settings": {"rpm": 0.0, "timesteps": 80000, "timestep": 0.0075, "force_scale": 1.0, "coupling": "sil-vorlap"}, "cases": [case]})
    args = SimpleNamespace(
        run_root=tmp_path,
        stage="setup",
        scenarios="vorlap",
        min_duration=600.0,
        max_timestep=0.01,
        expected_rpm=0.0,
        require_wind_name=False,
        allow_quasi_static=False,
    )

    report = audit.audit_campaign(args)
    codes = {issue["code"] for issue in report["issues"]}

    assert not report["passed"]
    assert "missing_geometry_audit" in codes


def test_setup_audit_requires_stable_support_patch_manifest_when_enabled(tmp_path):
    settings = {
        "wind_speeds": [5.0],
        "azimuths_deg": [0.0],
        "planned_cases": [{"wind_speed_mps": 5.0, "parked_azimuth_deg": 0.0}],
        "planned_case_count_per_scenario": 1,
        "planned_case_count_total": 1,
        "scenario_names": ["vorlap"],
        "scenario_force_scales": {"vorlap": 1.0},
        "scenario_couplings": {"vorlap": "sil-vorlap"},
        "duration_s": 600.0,
        "timestep_s": 0.0075,
        "timesteps": 80000,
        "rpm": 0.0,
        "stable_support_patch": True,
        "rampup_s": 0.0,
        "add_damp_s": 0.0,
        "str_iterations": None,
    }
    write_json(tmp_path / "campaign_settings.json", settings)
    case_dir = tmp_path / "vorlap" / "cases" / "U05p00ms_A000p00deg"
    sim = case_dir / "baseline_reference.sim"
    sim.parent.mkdir(parents=True)
    sim.write_text("synthetic\n", encoding="utf-8")
    write_geometry_audit(case_dir)
    case = {
        "index": 0,
        "label": "U05p00ms_A000p00deg",
        "wind_speed": 5.0,
        "azimuth_deg": 0.0,
        "sim": str(sim),
        "result_dir": str(tmp_path / "vorlap" / "results" / "U05p00ms_A000p00deg"),
    }
    write_json(
        tmp_path / "vorlap" / "cases.json",
        {
            "settings": {
                "rpm": 0.0,
                "timesteps": 80000,
                "timestep": 0.0075,
                "force_scale": 1.0,
                "coupling": "sil-vorlap",
                "stable_support_patch": True,
            },
            "cases": [case],
        },
    )
    args = SimpleNamespace(
        run_root=tmp_path,
        stage="setup",
        scenarios="vorlap",
        min_duration=600.0,
        max_timestep=0.01,
        expected_rpm=0.0,
        require_wind_name=False,
        allow_quasi_static=False,
    )

    invalid_report = audit.audit_campaign(args)
    assert not invalid_report["passed"]
    assert "invalid_stable_support_patch_model" in {issue["code"] for issue in invalid_report["issues"]}

    sim.write_text("reference_model/reference.trb TURBFILE - model\n", encoding="utf-8")
    missing_report = audit.audit_campaign(args)
    missing_codes = {issue["code"] for issue in missing_report["issues"]}

    assert not missing_report["passed"]
    assert "missing_stable_support_patch_manifest" in missing_codes

    write_stable_support_patch(case_dir)

    passing_report = audit.audit_campaign(args)

    assert passing_report["passed"]


def test_postprocess_writes_campaign_review_plot_artifacts(tmp_path):
    signed_comparison = pd.DataFrame(
        [
            {
                "case": "U05p00ms_A000p00deg",
                "wind_speed_mps": 5.0,
                "azimuth_deg": 0.0,
                "position": 0.23,
                "quantity": "moment",
                "axis": "Y",
                "unit": "Nm",
                "signed_del_baseline": 10.0,
                "signed_del_vorlap": 12.0,
            },
            {
                "case": "U05p00ms_A060p00deg",
                "wind_speed_mps": 5.0,
                "azimuth_deg": 60.0,
                "position": 0.23,
                "quantity": "moment",
                "axis": "Y",
                "unit": "Nm",
                "signed_del_baseline": 8.0,
                "signed_del_vorlap": 7.0,
            },
        ]
    )
    completion = pd.DataFrame(
        [
            {"scenario": "baseline", "wind_speed_mps": 5.0, "azimuth_deg": 0.0, "requested_steps": 100, "completed_steps": 100},
            {"scenario": "vorlap", "wind_speed_mps": 5.0, "azimuth_deg": 0.0, "requested_steps": 100, "completed_steps": 90},
        ]
    )
    source_coverage = pd.DataFrame(
        [
            {
                "scenario": "vorlap",
                "case": "U05p00ms_A000p00deg",
                "component": "BLD_1",
                "wind_speed_mps": 5.0,
                "azimuth_deg": 0.0,
                "aoa_min_deg": -10.0,
                "aoa_max_deg": 12.0,
                "source_aoa_min_deg": -30.0,
                "source_aoa_max_deg": 30.0,
                "re_min": 1.0e5,
                "re_max": 2.0e5,
                "source_re_min": 5.0e4,
                "source_re_max": 3.0e5,
                "nyquist_ratio": 5.0,
                "max_nodal_force_n": 100.0,
                "force_coefficient_max_abs": 1.0,
                "source_force_coefficient_max_abs": 2.0,
                "max_structural_velocity_mps": 0.2,
                "force_power_max_abs_w": 4.0,
                "re_out_of_range_count": 0,
                "aoa_out_of_range_count": 0,
                "aliased_frequency_count": 0,
                "source_clip_count": 0,
                "force_coefficient_out_of_range_count": 0,
                "nonfinite_count": 0,
            }
        ]
    )

    source_summary = postprocess.source_coverage_case_summary(source_coverage)
    assert source_summary["min_nyquist_ratio"].iloc[0] == 5.0
    assert source_summary["max_force_coefficient_ratio"].iloc[0] == 0.5
    assert source_summary["violation_count"].iloc[0] == 0

    paths = [
        postprocess.plot_signed_del_heatmap(signed_comparison, tmp_path),
        postprocess.plot_completion_heatmap(completion, tmp_path),
        postprocess.plot_source_coverage_sanity(source_summary, tmp_path),
    ]

    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0


def test_postprocess_writes_critical_case_diagnostic_artifacts(tmp_path):
    channel = "Y_l~Mom.~BLD_1~pos~0.230~[Nm]"
    signed_comparison = pd.DataFrame(
        [
            {
                "case": "U05p00ms_A000p00deg",
                "wind_speed_mps": 5.0,
                "azimuth_deg": 0.0,
                "position": 0.23,
                "quantity": "moment",
                "axis": "Y",
                "unit": "Nm",
                "channel_baseline": channel,
                "channel_vorlap": channel,
                "signed_del_baseline": 10.0,
                "signed_del_vorlap": 15.0,
                "signed_delta_pct": 50.0,
                "rainflow_cycles_baseline": 2.0,
                "rainflow_cycles_vorlap": 2.0,
                "postprocess_warnings": "",
            }
        ]
    )
    source_summary = pd.DataFrame(
        [
            {
                "scenario": "vorlap",
                "case": "U05p00ms_A000p00deg",
                "wind_speed_mps": 5.0,
                "azimuth_deg": 0.0,
                "min_nyquist_ratio": 4.0,
                "max_force_coefficient_ratio": 0.75,
                "force_power_max_abs_w": 12.0,
                "violation_count": 0,
            }
        ]
    )
    time = [0.0, 1.0, 2.0, 3.0, 4.0]
    cases = [
        postprocess.CaseResult(
            scenario="identity",
            label="U05p00ms_A000p00deg",
            wind_speed=5.0,
            azimuth_deg=0.0,
            path=tmp_path / "identity.txt",
            data=pd.DataFrame({"Time~[s]": time, channel: [0.0, 1.0, 0.0, -1.0, 0.0]}),
        ),
        postprocess.CaseResult(
            scenario="vorlap",
            label="U05p00ms_A000p00deg",
            wind_speed=5.0,
            azimuth_deg=0.0,
            path=tmp_path / "vorlap.txt",
            data=pd.DataFrame({"Time~[s]": time, channel: [0.0, 2.0, 0.0, -2.0, 0.0]}),
        ),
    ]

    summary = postprocess.critical_case_summary(signed_comparison, source_summary)
    assert summary["case"].iloc[0] == "U05p00ms_A000p00deg"
    assert summary["selection_reason"].iloc[0] == "max_signed_del_ratio"
    assert summary["signed_del_ratio"].iloc[0] == 1.5
    assert summary["force_power_max_abs_w"].iloc[0] == 12.0

    plot_path, rainflow_path = postprocess.plot_critical_case_diagnostic(
        cases,
        summary,
        tmp_path,
        comparison_baseline="identity",
        comparison_coupled="vorlap",
        wohler_exponent=10.0,
    )

    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
    rainflow = pd.read_csv(rainflow_path)
    assert not rainflow.empty
    assert {"range", "count"}.issubset(rainflow.columns)


def test_postprocess_computes_ultimate_margin_from_signed_extrema():
    signed_metrics = pd.DataFrame(
        [
            {
                "scenario": "vorlap",
                "case": "U05p00ms_A000p00deg",
                "wind_speed_mps": 5.0,
                "azimuth_deg": 0.0,
                "blade": "BLD_1",
                "position": 0.23,
                "quantity": "moment",
                "axis": "Y",
                "unit": "Nm",
                "channel": "Y_l~Mom.~BLD_1~pos~0.230~[Nm]",
                "min": -6.0,
                "max": 10.0,
                "samples": 2000,
                "duration_s": 20.0,
                "smoke_only": False,
                "postprocess_warnings": "",
                "result_file": "result.txt",
            }
        ]
    )

    extreme = postprocess.extreme_channel_table(signed_metrics)
    template = postprocess.ultimate_allowables_template(extreme)

    assert template[["blade", "position", "quantity", "axis", "unit"]].iloc[0].to_dict() == {
        "blade": "BLD_1",
        "position": 0.23,
        "quantity": "moment",
        "axis": "Y",
        "unit": "Nm",
    }

    missing = postprocess.ultimate_margin_table(extreme, pd.DataFrame(columns=postprocess.ULTIMATE_ALLOWABLE_TEMPLATE_COLUMNS))
    assert missing["ultimate_status"].iloc[0] == "missing_allowable"
    assert "missing positive allowable_abs" in missing["postprocess_warnings"].iloc[0]

    allowables = template.copy()
    allowables["allowable_abs"] = 20.0
    allowables["allowable_label"] = "blade-root test allowable"
    allowables["load_factor"] = 1.5
    allowables["resistance_factor"] = 0.9
    margins = postprocess.ultimate_margin_table(extreme, allowables)

    assert margins["ultimate_status"].iloc[0] == "ok"
    assert margins["allowable_label"].iloc[0] == "blade-root test allowable"
    assert margins["factored_demand"].iloc[0] == 15.0
    assert margins["factored_allowable"].iloc[0] == 18.0
    assert margins["utilization"].iloc[0] == 15.0 / 18.0
    assert margins["reserve_factor"].iloc[0] == 18.0 / 15.0

    minimal_allowables = template[["blade", "position", "quantity", "axis", "unit"]].copy()
    minimal_allowables["allowable_abs"] = 12.5
    minimal_extreme = extreme.drop(columns=["postprocess_warnings"]).set_index(pd.Index([42]))
    minimal_margins = postprocess.ultimate_margin_table(minimal_extreme, minimal_allowables)
    assert minimal_margins["ultimate_status"].iloc[0] == "ok"
    assert minimal_margins["allowable_label"].iloc[0] == ""
    assert minimal_margins["load_factor"].iloc[0] == 1.0
    assert minimal_margins["resistance_factor"].iloc[0] == 1.0
