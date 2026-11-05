from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vorlap import qblade_batch
from vorlap.qblade_files import read_qblade_keyword, render_qblade_value


def batch_args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
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
        result_filter=None,
        n_freq_depth=20,
        source_bounds_policy="warn",
        source_frequency_truncation="none",
        force_scale=1.0,
        diagnostic_force_scale=False,
        wind_type=0,
        wind_name="",
        horizontal_inflow_angle=0.0,
        vertical_inflow_angle=0.0,
        shear_exponent=0.23,
        reference_height=18.0,
        air_density=1.0296,
        air_kinematic_viscosity=1.647e-5,
        skip_stable_support_patch=False,
        rampup=0.0,
        add_damp=0.0,
        str_iterations=None,
        allow_partial=False,
        coupling="sil-vorlap",
        threads=8,
        progress_every=100,
        use_opencl=False,
        opencl_device=0,
        opencl_group_size=32,
        qblade_steps=None,
        checkpoint_every=0,
        resume_initialize=False,
        record_sil_vorlap_history=False,
        store_project=False,
        native_run_full_simulation=False,
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_qblade_float_rendering_preserves_small_physical_inputs():
    assert render_qblade_value(1.647e-5) == "1.647e-05"
    assert float(render_qblade_value(1.647e-5)) == 1.647e-5


def write_minimal_reference_case(vorlap_root: Path) -> None:
    reference = vorlap_root / "examples" / "ReferenceHVAWTQBladeModel"
    turbine = reference / "10m_HVAWT_reference_turbine"
    structure = turbine / "Structure" / "10m_HVAWT_reference_main.str"
    airfoils = turbine / "VorLapAirfoils"
    structure.parent.mkdir(parents=True)
    airfoils.mkdir(parents=True)
    (reference / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim").write_text(
        "\n".join(
            [
                "10m_HVAWT_reference_turbine/10m_HVAWT_reference_turbine.trb TURBFILE - turbine",
                "10m_HVAWT_reference_turbine TURBNAME - turbine name",
                "6.0 MEANINF - mean inflow",
                "270.0 INITIAL_AZIMUTH - initial azimuth",
                "0 PRESCRIBETYPE - rpm prescribe mode",
                "0.0 RPMPRESCRIBED - rpm",
                "2000 NUMTIMESTEPS - steps",
                "0.0075 TIMESTEP - timestep",
                "0.0 STOREFROM - store from",
                "false STOREREPLAY - store replay",
                " FILTERFILE - results filter",
                "0 WNDTYPE - wind type",
                " WNDNAME - wind file",
                "0 HORANGLE - horizontal inflow angle",
                "0 VERTANGLE - vertical inflow angle",
                "0.23 SHEAREXP - shear exponent",
                "18 REFHEIGHT - reference height",
                "1.0296 DENSITYAIR - air density",
                "0.00001647 VISCOSITYAIR - air kinematic viscosity",
                "100 RELAXSTEPS - static relaxation steps",
                "0.1 RELAXTIMESTEP - relaxation timestep",
                "5.0 RAMPUP - rampup",
                "0.2 ADDDAMP - damping",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (turbine / "10m_HVAWT_reference_turbine.trb").write_text(
        "\n".join(
            [
                "10m H-VAWT Reference Turbine OBJECTNAME - turbine object",
                "Structure/10m_HVAWT_reference_main.str STRUCTURALFILE - structure",
                "1 DYNSTALLTYPE - dynamic stall model",
                "6 TF_OYE - Oye time constant",
                "0 WAKELENGTHTYPE - wake age unit",
                "200000 MAXWAKESIZE - maximum wake elements",
                "100 MAXWAKEDIST - normalized wake distance",
                "0.005 WAKEREDUCTION - circulation cutoff",
                "0.1 NEARWAKELENGTH - near wake",
                "1.3 ZONE1LENGTH - zone 1",
                "2.7 ZONE2LENGTH - zone 2",
                "13.3 ZONE3LENGTH - zone 3",
                "0.50 BOUNDCORERADIUS - bound-vortex core radius",
                "0.50 WAKECORERADIUS - wake-vortex core radius",
                "110.00 VORTEXVISCOSITY - turbulent vortex viscosity",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
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


def test_validate_qblade_root_reports_all_missing_runtime_pieces(tmp_path):
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        qblade_batch.validate_qblade_root(qblade_root)

    message = str(excinfo.value)
    assert "Incomplete QBlade root" in message
    assert str(qblade_root / "SIL_Interface") in message
    assert str(qblade_root / "ControllerFiles") in message
    assert f"{qblade_root}/*QBlade*.so*" in message
    assert "missing QBlade SIL_Interface directory" in message
    assert "missing QBlade ControllerFiles directory" in message
    assert "missing QBlade shared library" in message


def test_build_bridge_validates_freshly_installed_library(tmp_path, monkeypatch):
    args = batch_args(tmp_path, force_bridge_build=True, skip_bridge_build=False)
    install_dir = args.qblade_root / "ControllerFiles"
    install_dir.mkdir(parents=True)
    library = install_dir / "libvorlap_qblade_bridge.so"
    calls: list[Path] = []

    def fake_run_checked(command, *, cwd=None, env=None):
        library.write_text("placeholder\n", encoding="utf-8")

    monkeypatch.setattr(qblade_batch, "run_checked", fake_run_checked)
    monkeypatch.setattr(qblade_batch, "validate_bridge_library_loadable", lambda path: calls.append(Path(path)))

    qblade_batch.build_bridge(args)

    assert calls == [library]


def test_build_bridge_validates_existing_library_before_reuse(tmp_path, monkeypatch):
    args = batch_args(tmp_path, force_bridge_build=False, skip_bridge_build=False)
    install_dir = args.qblade_root / "ControllerFiles"
    install_dir.mkdir(parents=True)
    existing = install_dir / "libvorlap_qblade_bridge.so"
    existing.write_text("placeholder\n", encoding="utf-8")
    calls: list[Path] = []

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("existing bridge should skip rebuild")

    monkeypatch.setattr(qblade_batch, "run_checked", fail_if_called)
    monkeypatch.setattr(qblade_batch, "validate_bridge_library_loadable", lambda path: calls.append(Path(path)))

    qblade_batch.build_bridge(args)

    assert calls == [existing]


def test_forced_bridge_build_runs_once_per_setup_token(tmp_path, monkeypatch):
    args = batch_args(tmp_path, force_bridge_build=True, skip_bridge_build=False)
    install_dir = args.qblade_root / "ControllerFiles"
    install_dir.mkdir(parents=True)
    library = install_dir / "libvorlap_qblade_bridge.so"
    build_calls: list[list[str]] = []
    validate_calls: list[Path] = []

    def fake_run_checked(command, *, cwd=None, env=None):
        build_calls.append(command)
        library.write_text("rebuilt\n", encoding="utf-8")

    monkeypatch.setenv("VORLAP_FORCE_SETUP_TOKEN", "slurm-array-42")
    monkeypatch.setattr(qblade_batch, "run_checked", fake_run_checked)
    monkeypatch.setattr(
        qblade_batch,
        "validate_bridge_library_loadable",
        lambda path: validate_calls.append(Path(path)),
    )

    qblade_batch.build_bridge(args)
    qblade_batch.build_bridge(args)

    assert len(build_calls) == 1
    assert validate_calls == [library, library]


def test_matching_cached_setup_still_validates_required_bridge(tmp_path, monkeypatch):
    args = batch_args(
        tmp_path,
        wind_speeds="5",
        azimuths="0",
        skip_vorlap_install=True,
        skip_bridge_build=False,
        force_setup=False,
    )
    args.run_root.mkdir(parents=True)
    manifest = qblade_batch.build_case_manifest(args)
    manifest_path = args.run_root / "cases.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (args.run_root / "setup_complete.json").write_text("{}", encoding="utf-8")
    sim = Path(manifest["cases"][0]["sim"])
    sim.parent.mkdir(parents=True)
    sim.write_text(
        "5 MEANINF\n"
        "0 INITIAL_AZIMUTH\n"
        "1 PRESCRIBETYPE\n"
        "0 RPMPRESCRIBED\n"
        "80000 NUMTIMESTEPS\n"
        "0.0075 TIMESTEP\n"
        "60 STOREFROM\n"
        "false STOREREPLAY\n"
        "0 WNDTYPE\n"
        "0 HORANGLE\n"
        "0 VERTANGLE\n"
        "0.23 SHEAREXP\n"
        "18 REFHEIGHT\n"
        "1.0296 DENSITYAIR\n"
        "0.00001647 VISCOSITYAIR\n",
        encoding="utf-8",
    )
    assets, fingerprint = qblade_batch.qblade_runtime_asset_fingerprint(sim)
    manifest["cases"][0]["runtime_assets"] = assets
    manifest["cases"][0]["runtime_asset_fingerprint"] = fingerprint
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(qblade_batch, "validate_qblade_root", lambda _root: None)
    monkeypatch.setattr(qblade_batch, "build_bridge", lambda value: calls.append(value))

    assert qblade_batch.setup(args) == manifest
    assert calls == [args]


def test_setup_artifact_check_rejects_rampup_only_rpm_prescription(tmp_path):
    args = batch_args(tmp_path, wind_speeds="5", azimuths="0", rpm=1.0)
    manifest = qblade_batch.build_case_manifest(args)
    sim = Path(manifest["cases"][0]["sim"])
    sim.parent.mkdir(parents=True)
    sim.write_text(
        "5 MEANINF\n"
        "0 INITIAL_AZIMUTH\n"
        "0 PRESCRIBETYPE\n"
        "1 RPMPRESCRIBED\n"
        "80000 NUMTIMESTEPS\n"
        "0.0075 TIMESTEP\n"
        "60 STOREFROM\n"
        "false STOREREPLAY\n"
        "0 WNDTYPE\n",
        encoding="utf-8",
    )

    issues = qblade_batch.setup_artifact_issues(manifest)

    assert any("PRESCRIBETYPE is '0'" in item and "expected '1'" in item for item in issues)


def test_typed_batch_manifest_expands_wind_azimuth_matrix(tmp_path):
    args = batch_args(tmp_path)

    manifest = qblade_batch.QBladeBatchManifest.from_args(args, created_unix=123.0).as_dict()

    assert manifest["created_unix"] == 123.0
    assert manifest["qblade_root"] == str(args.qblade_root)
    assert manifest["vorlap_root"] == str(args.vorlap_root)
    assert manifest["python"] == sys.executable
    assert manifest["settings"]["wind_speeds_mps"] == [5.0, 10.0]
    assert manifest["settings"]["rpm_prescribe_type"] == 1
    assert manifest["settings"]["azimuths_deg"] == [0.0, 60.0, 120.0]
    assert manifest["settings"]["stable_support_patch"] is True
    assert manifest["settings"]["coupling"] == "sil-vorlap"
    assert manifest["settings"]["wake_profile"] == "reference"
    assert manifest["settings"]["wake_settings"] == {}
    assert len(manifest["cases"]) == 6

    first_case = manifest["cases"][0]
    assert first_case["index"] == 0
    assert first_case["label"] == "U05p00ms_A000p00deg"
    assert first_case["speed_label"] == "U05p00ms"
    assert first_case["azimuth_label"] == "A000p00deg"
    assert first_case["wind_speed"] == 5.0
    assert first_case["azimuth_deg"] == 0.0
    assert first_case["case_dir"] == str(args.run_root / "cases" / "U05p00ms_A000p00deg")
    assert first_case["sim"] == str(
        args.run_root / "cases" / "U05p00ms_A000p00deg" / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"
    )
    assert first_case["result_dir"] == str(args.run_root / "results" / "U05p00ms_A000p00deg")
    assert first_case["result_name"] == "U05p00ms_A000p00deg_qblade_results"

    assert manifest["cases"][-1]["label"] == "U10p00ms_A120p00deg"


def test_parked_reduced_wake_profile_is_explicit_and_returns_a_copy():
    expected = {
        "WAKELENGTHTYPE": 1,
        "MAXWAKESIZE": 120000,
        "MAXWAKEDIST": 30,
        "WAKEREDUCTION": 0.005,
        "NEARWAKELENGTH": 200,
        "ZONE1LENGTH": 800,
        "ZONE2LENGTH": 2400,
        "ZONE3LENGTH": 6400,
    }

    settings = qblade_batch.wake_profile_settings("parked-reduced-v1")
    assert settings == expected
    settings["MAXWAKESIZE"] = 1
    assert qblade_batch.wake_profile_settings("parked-reduced-v1") == expected


def test_parked_reduced_v2_selects_accepted_unh_parameters_without_changing_other_cost_controls():
    expected = {
        "DYNSTALLTYPE": 1,
        "TF_OYE": 8,
        "WAKELENGTHTYPE": 1,
        "MAXWAKESIZE": 120000,
        "MAXWAKEDIST": 10,
        "WAKEREDUCTION": 0.005,
        "NEARWAKELENGTH": 200,
        "ZONE1LENGTH": 800,
        "ZONE2LENGTH": 2400,
        "ZONE3LENGTH": 6400,
        "BOUNDCORERADIUS": 1.0,
        "WAKECORERADIUS": 0.05,
        "VORTEXVISCOSITY": 80.0,
    }

    settings = qblade_batch.wake_profile_settings("parked-reduced-v2")
    assert settings == expected
    settings["MAXWAKEDIST"] = 100
    assert qblade_batch.wake_profile_settings("parked-reduced-v2") == expected


def test_prepare_cases_applies_reduced_wake_only_to_generated_turbine(tmp_path, monkeypatch):
    vorlap_root = tmp_path / "vorlap"
    write_minimal_reference_case(vorlap_root)
    source_trb = (
        vorlap_root
        / "examples"
        / "ReferenceHVAWTQBladeModel"
        / "10m_HVAWT_reference_turbine"
        / "10m_HVAWT_reference_turbine.trb"
    )
    original_source = source_trb.read_text(encoding="utf-8")
    args = batch_args(
        tmp_path,
        vorlap_root=vorlap_root,
        wind_speeds="5",
        azimuths="0",
        rpm=1.0,
        wake_profile="parked-reduced-v2",
    )
    manifest = qblade_batch.build_case_manifest(args)
    monkeypatch.setattr(qblade_batch, "run_checked", lambda *_args, **_kwargs: None)

    qblade_batch.prepare_cases(args, manifest)

    generated_sim = Path(str(manifest["cases"][0]["sim"]))
    generated_trb = qblade_batch.turbine_definition_path_for_sim(generated_sim)
    for keyword, value in qblade_batch.wake_profile_settings("parked-reduced-v2").items():
        assert qblade_batch.qblade_setting_matches(
            read_qblade_keyword(generated_trb, keyword),
            render_qblade_value(value),
        )
    wake_manifest = generated_trb.parent / "Control" / "vorlap_wake_profile.json"
    payload = json.loads(wake_manifest.read_text(encoding="utf-8"))
    assert payload["profile"] == "parked-reduced-v2"
    assert len(payload["patches"]) == 13
    assert source_trb.read_text(encoding="utf-8") == original_source
    assert qblade_batch.setup_artifact_issues(manifest) == []

    qblade_batch.replace_qblade_keyword(generated_trb, "MAXWAKESIZE", 200000)
    assert any(
        "MAXWAKESIZE is '200000', expected '120000'" in issue
        for issue in qblade_batch.setup_artifact_issues(manifest)
    )


@pytest.mark.parametrize("profile", ["parked-reduced-v1", "parked-reduced-v2"])
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rpm": 0.0}, "requires prescribed rpm=1"),
        ({"rpm": 1.0, "timestep": 0.01}, "requires timestep=0.0075 s"),
    ],
)
def test_parked_reduced_wake_profile_rejects_unqualified_time_basis(
    tmp_path, profile, overrides, message
):
    args = batch_args(
        tmp_path,
        wake_profile=profile,
        **overrides,
    )

    with pytest.raises(ValueError, match=message):
        qblade_batch.build_case_manifest(args)


def test_manifest_setup_comparison_ignores_only_creation_time(tmp_path):
    args = batch_args(tmp_path)
    planned = qblade_batch.build_case_manifest(args)
    existing = deepcopy(planned)
    existing["created_unix"] = 1.0

    assert qblade_batch.manifest_matches_setup_inputs(existing, planned)

    changed = deepcopy(existing)
    changed["settings"]["force_scale"] = 2.0

    assert not qblade_batch.manifest_matches_setup_inputs(changed, planned)


@pytest.mark.parametrize("azimuth_spec", ["-1", "360", "range:0:360:180"])
def test_parse_azimuths_rejects_values_outside_single_turn(azimuth_spec):
    with pytest.raises(ValueError, match="Azimuths must be finite values in \\[0, 360\\) deg"):
        qblade_batch.parse_azimuths(azimuth_spec)


def test_single_azimuth_labels_cases_by_speed_only(tmp_path):
    args = batch_args(tmp_path, wind_speeds="linspace:2:4:3", azimuths="270")

    manifest = qblade_batch.QBladeBatchManifest.from_args(args, created_unix=123.0).as_dict()

    assert manifest["settings"]["wind_speeds_mps"] == [2.0, 3.0, 4.0]
    assert manifest["settings"]["azimuths_deg"] == [270.0]
    assert [case["label"] for case in manifest["cases"]] == ["U02p00ms", "U03p00ms", "U04p00ms"]


def test_batch_run_options_build_runner_command_from_typed_fields(tmp_path):
    result_filter = tmp_path / "fatigue.filter"
    result_filter.write_text("Time [s]\nLSS Rpm [rpm]\n", encoding="utf-8")
    args = batch_args(
        tmp_path,
        use_opencl=True,
        opencl_device=2,
        opencl_group_size=64,
        qblade_steps=25,
        allow_partial=True,
        checkpoint_every=5,
        resume_initialize=True,
        record_sil_vorlap_history=True,
        store_project=True,
        result_filter=result_filter,
    )
    manifest = qblade_batch.build_case_manifest(args)
    options = qblade_batch.QBladeBatchRunOptions.from_args(args)
    metadata = tmp_path / "metadata.json"
    runner = tmp_path / "run_qblade_sim.py"

    command = options.command_for_case(manifest["cases"][0], runner_path=runner, case_metadata_path=metadata)

    assert command[:2] == [sys.executable, str(runner)]
    assert command[command.index("--qblade-root") + 1] == str(args.qblade_root)
    assert command[command.index("--sim") + 1] == manifest["cases"][0]["sim"]
    assert command[command.index("--output-dir") + 1] == manifest["cases"][0]["result_dir"]
    assert command[command.index("--threads") + 1] == "8"
    assert command[command.index("--result-name") + 1] == manifest["cases"][0]["result_name"]
    assert command[command.index("--coupling-mode") + 1] == "sil-vorlap"
    assert command[command.index("--case-metadata") + 1] == str(metadata)
    assert command[command.index("--progress-every") + 1] == "100"
    assert command[command.index("--device") + 1] == "2"
    assert command[command.index("--group-size") + 1] == "64"
    assert command[command.index("--steps") + 1] == "25"
    assert command[command.index("--checkpoint-every") + 1] == "5"
    assert command[command.index("--result-filter") + 1] == str(result_filter.resolve())
    assert "--resume-from-checkpoint" in command
    assert "--resume-initialize" in command
    assert "--sil-vorlap" in command
    assert "--record-sil-vorlap-history" in command
    assert "--allow-partial" in command
    assert "--store-project" in command
    assert "--native-run-full-simulation" not in command


def test_batch_run_options_passes_explicit_compact_sil_history_mode(tmp_path):
    args = batch_args(
        tmp_path,
        sil_vorlap_history_mode="compact",
    )
    manifest = qblade_batch.build_case_manifest(args)
    options = qblade_batch.QBladeBatchRunOptions.from_args(args)
    command = options.command_for_case(
        manifest["cases"][0],
        runner_path=tmp_path / "run_qblade_sim.py",
        case_metadata_path=tmp_path / "metadata.json",
    )

    assert "--record-sil-vorlap-history" not in command
    assert command[command.index("--sil-vorlap-history-mode") + 1] == "compact"


def test_batch_run_options_rejects_compact_mode_with_legacy_full_flag(tmp_path):
    args = batch_args(
        tmp_path,
        record_sil_vorlap_history=True,
        sil_vorlap_history_mode="compact",
    )

    with pytest.raises(ValueError, match="legacy record_history flag selects full history"):
        qblade_batch.QBladeBatchRunOptions.from_args(args)


def test_batch_run_options_direct_construction_rejects_conflicting_history_modes(tmp_path):
    with pytest.raises(ValueError, match="legacy record_history flag selects full history"):
        qblade_batch.QBladeBatchRunOptions(
            python=sys.executable,
            qblade_root=tmp_path / "QBladeCE",
            vorlap_root=tmp_path,
            threads=1,
            coupling="sil-vorlap",
            progress_every=1,
            record_sil_vorlap_history=True,
            sil_vorlap_history_mode="compact",
        )


def test_batch_run_options_rejects_history_for_non_sil_coupling(tmp_path):
    args = batch_args(
        tmp_path,
        coupling="qblade-external",
        sil_vorlap_history_mode="compact",
    )

    with pytest.raises(ValueError, match="history options require coupling='sil-vorlap'"):
        qblade_batch.QBladeBatchRunOptions.from_args(args)


def test_batch_run_options_passes_native_full_run_diagnostic_flag(tmp_path):
    args = batch_args(
        tmp_path,
        coupling="qblade-external",
        native_run_full_simulation=True,
    )
    manifest = qblade_batch.build_case_manifest(args)
    options = qblade_batch.QBladeBatchRunOptions.from_args(args)
    metadata = tmp_path / "metadata.json"
    runner = tmp_path / "run_qblade_sim.py"

    command = options.command_for_case(manifest["cases"][0], runner_path=runner, case_metadata_path=metadata)

    assert command[command.index("--coupling-mode") + 1] == "qblade-external"
    assert "--native-run-full-simulation" in command
    assert "--sil-vorlap" not in command


def test_run_checked_waits_on_isolated_process_and_raises_nonzero(monkeypatch, tmp_path):
    launched: list[dict[str, object]] = []

    class Process:
        def wait(self):
            return 7

    def launch(command, *, cwd=None, env=None):
        launched.append({"command": command, "cwd": cwd, "env": env})
        return Process()

    env = {"VORLAP_TEST": "1"}
    monkeypatch.setattr(qblade_batch, "launch_isolated_process", launch)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        qblade_batch.run_checked(["qblade", "--case", "demo.sim"], cwd=tmp_path, env=env)

    assert excinfo.value.returncode == 7
    assert launched == [{"command": ["qblade", "--case", "demo.sim"], "cwd": tmp_path, "env": env}]


def test_run_checked_terminates_isolated_process_on_keyboard_interrupt(monkeypatch):
    terminated: list[object] = []

    class Process:
        def wait(self):
            raise KeyboardInterrupt

    process = Process()
    monkeypatch.setattr(qblade_batch, "launch_isolated_process", lambda command, *, cwd=None, env=None: process)
    monkeypatch.setattr(qblade_batch, "terminate_process_tree", lambda value: terminated.append(value))

    with pytest.raises(KeyboardInterrupt):
        qblade_batch.run_checked(["qblade", "--case", "demo.sim"])

    assert terminated == [process]


def test_prepare_cases_copies_patches_and_invokes_case_prep(tmp_path, monkeypatch):
    vorlap_root = tmp_path / "vorlap"
    write_minimal_reference_case(vorlap_root)
    result_filter = tmp_path / "parked_fatigue_results.filter"
    result_filter.write_text("Time [s]\nLSS Rpm [rpm]\n", encoding="utf-8")
    args = batch_args(
        tmp_path,
        vorlap_root=vorlap_root,
        wind_speeds="7",
        azimuths="30",
        timesteps=1234,
        timestep=0.01,
        source_frequency_truncation="nyquist",
        store_from=3.0,
        store_replay=True,
        n_freq_depth=12,
        wind_type=1,
        wind_name="wind_{speed_label}_{azimuth_label}.bts",
        rampup=0.0,
        add_damp=0.0,
        result_filter=result_filter,
    )
    manifest = qblade_batch.build_case_manifest(args)
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def capture_run_checked(command: list[str], *, cwd=None, env=None) -> None:
        assert cwd is None
        calls.append((command, env))
        prepared_sim = Path(command[command.index("--sim") + 1])
        turbine = prepared_sim.parent / "10m_HVAWT_reference_turbine"
        control = turbine / "Control"
        control.mkdir(parents=True, exist_ok=True)
        parameter_file = control / "vorlap_qblade_external.json"
        parameter_file.write_text(
            json.dumps(
                {
                    "external_input_rows": [
                        {"index": 0, "variable": "Time [s]"},
                        {"index": 1, "variable": "Timestep [-]"},
                        {"index": 2, "variable": "LSS Azimuthal Pos. [deg]"},
                        {"index": 3, "variable": "X_g Vel. BLD_1 pos 0.500 [m/s]"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        turbine_file = turbine / "10m_HVAWT_reference_turbine.trb"
        turbine_file.write_text(
            turbine_file.read_text(encoding="utf-8")
            + "Control/vorlap_qblade_external.json LIBPARAMETERFILE_1\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(qblade_batch, "run_checked", capture_run_checked)

    qblade_batch.prepare_cases(args, manifest)

    case = manifest["cases"][0]
    sim_path = Path(str(case["sim"]))
    assert sim_path.is_file()
    assert read_qblade_keyword(sim_path, "MEANINF") == "7"
    assert read_qblade_keyword(sim_path, "INITIAL_AZIMUTH") == "30"
    assert read_qblade_keyword(sim_path, "PRESCRIBETYPE") == "1"
    assert read_qblade_keyword(sim_path, "RPMPRESCRIBED") == "0"
    assert read_qblade_keyword(sim_path, "NUMTIMESTEPS") == "1234"
    assert read_qblade_keyword(sim_path, "TIMESTEP") == "0.01"
    assert read_qblade_keyword(sim_path, "STOREFROM") == "3"
    assert read_qblade_keyword(sim_path, "STOREREPLAY") == "true"
    storage_filter = Path(str(case["storage_filter"]))
    assert read_qblade_keyword(sim_path, "FILTERFILE") == storage_filter.name
    copied_filter = Path(str(case["result_filter"]))
    assert copied_filter.read_text(encoding="utf-8") == result_filter.read_text(encoding="utf-8")
    assert qblade_batch.qblade_filter_channels(storage_filter) == [
        "Time [s]",
        "LSS Rpm [rpm]",
        "LSS Azimuthal Pos. [deg]",
        "X_g Vel. BLD_1 pos 0.500 [m/s]",
    ]
    assert case["storage_filter_channel_count"] == 4
    assert case["storage_filter_sha256"]
    assert manifest["settings"]["result_filter_channel_count"] == 2
    assert manifest["settings"]["result_filter_sha256"]
    assert read_qblade_keyword(sim_path, "WNDTYPE") == "1"
    assert read_qblade_keyword(sim_path, "WNDNAME") == "wind_U07p00ms_A030p00deg.bts"
    assert read_qblade_keyword(sim_path, "HORANGLE") == "0"
    assert read_qblade_keyword(sim_path, "VERTANGLE") == "0"
    assert read_qblade_keyword(sim_path, "SHEAREXP") == "0.23"
    assert read_qblade_keyword(sim_path, "REFHEIGHT") == "18"
    assert read_qblade_keyword(sim_path, "DENSITYAIR") == "1.0296"
    assert read_qblade_keyword(sim_path, "VISCOSITYAIR") == "1.647e-05"
    assert read_qblade_keyword(sim_path, "RELAXSTEPS") == "0"
    assert (
        sim_path.parent
        / "10m_HVAWT_reference_turbine"
        / "Control"
        / "vorlap_stable_support_patch.json"
    ).is_file()

    assert len(calls) == 1
    command, env = calls[0]
    assert command[:2] == [sys.executable, str(vorlap_root / "scripts" / "prepare_qblade_external_case.py")]
    assert command[command.index("--sim") + 1] == str(sim_path)
    assert command[command.index("--airfoils") + 1] == str(
        sim_path.parent / "10m_HVAWT_reference_turbine" / "VorLapAirfoils"
    )
    assert command[command.index("--node-source") + 1] == "structural"
    assert command[command.index("--n-freq-depth") + 1] == "12"
    assert command[command.index("--source-bounds-policy") + 1] == "warn"
    assert command[command.index("--source-frequency-cap-hz") + 1] == "50"
    assert command[command.index("--force-scale") + 1] == "1.0"
    assert "--debug" in command
    assert "--absolute-source-parameter-dir" in command
    assert env is not None
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(vorlap_root)


def test_setup_artifact_check_detects_mutated_copied_airfoil_h5(tmp_path, monkeypatch):
    vorlap_root = tmp_path / "vorlap"
    write_minimal_reference_case(vorlap_root)
    source_h5 = (
        vorlap_root
        / "examples"
        / "ReferenceHVAWTQBladeModel"
        / "10m_HVAWT_reference_turbine"
        / "VorLapAirfoils"
        / "NACA0018.h5"
    )
    source_h5.write_bytes(b"source airfoil")
    args = batch_args(
        tmp_path,
        vorlap_root=vorlap_root,
        wind_speeds="7",
        azimuths="30",
        wind_type=0,
        wind_name="",
    )
    manifest = qblade_batch.build_case_manifest(args)
    monkeypatch.setattr(qblade_batch, "run_checked", lambda *_args, **_kwargs: None)
    qblade_batch.prepare_cases(args, manifest)

    assert qblade_batch.setup_artifact_issues(manifest) == []
    copied_h5 = (
        Path(str(manifest["cases"][0]["case_dir"]))
        / "10m_HVAWT_reference_turbine"
        / "VorLapAirfoils"
        / "NACA0018.h5"
    )
    copied_h5.write_bytes(b"mutated airfoil")

    issues = qblade_batch.setup_artifact_issues(manifest)
    assert any("runtime-asset fingerprint" in item for item in issues)


def test_qblade_aerodynamic_reference_issues_reports_missing_airfoil_geometry(tmp_path):
    reference = tmp_path / "reference"
    turbine = reference / "turbine"
    aero = turbine / "Aero"
    polars = aero / "Polars"
    polars.mkdir(parents=True)
    sim = reference / "case.sim"
    trb = turbine / "case.trb"
    blade = aero / "rotor.bld"
    polar = polars / "NACA_0018_RFoil_MultiRePolar.plr"
    sim.write_text("turbine/case.trb TURBFILE - turbine\n", encoding="utf-8")
    trb.write_text("Aero/rotor.bld BLADEFILE - blade\n", encoding="utf-8")
    blade.write_text(
        "Polars/NACA_0018_RFoil_MultiRePolar.plr POLAR_FILE\n",
        encoding="utf-8",
    )
    polar.write_text(
        "../Airfoils/NACA_0018_RFoil.afl FOILNAME - airfoil geometry\n",
        encoding="utf-8",
    )

    issues = qblade_batch.qblade_aerodynamic_reference_issues(sim)

    assert issues == [
        f"missing QBlade airfoil geometry file: "
        f"{aero / 'Airfoils' / 'NACA_0018_RFoil.afl'}"
    ]


def test_qblade_aerodynamic_reference_issues_accepts_complete_model(tmp_path):
    reference = tmp_path / "reference"
    turbine = reference / "turbine"
    aero = turbine / "Aero"
    airfoils = aero / "Airfoils"
    polars = aero / "Polars"
    airfoils.mkdir(parents=True)
    polars.mkdir(parents=True)
    sim = reference / "case.sim"
    trb = turbine / "case.trb"
    blade = aero / "rotor.bld"
    polar = polars / "NACA_0018_RFoil_MultiRePolar.plr"
    foil = airfoils / "NACA_0018_RFoil.afl"
    sim.write_text("turbine/case.trb TURBFILE - turbine\n", encoding="utf-8")
    trb.write_text("Aero/rotor.bld BLADEFILE - blade\n", encoding="utf-8")
    blade.write_text(
        "Polars/NACA_0018_RFoil_MultiRePolar.plr POLAR_FILE\n",
        encoding="utf-8",
    )
    polar.write_text(
        "../Airfoils/NACA_0018_RFoil.afl FOILNAME - airfoil geometry\n",
        encoding="utf-8",
    )
    foil.write_text("NACA 0018\n1.0 0.0\n", encoding="utf-8")

    assert qblade_batch.qblade_aerodynamic_reference_issues(sim) == []
