import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import pytest

import vorlap.qblade_run as qblade_run_module
from vorlap.qblade_run import QBladeRunRequest, QBladeRunSummary, run_qblade_case
from vorlap.qblade_modal_decay import ModalBurstPlan
from vorlap.qblade_run_support import (
    build_modal_native_bridge_runtime_identity,
    file_sha256,
)


class FakeQBlade:
    def __init__(self) -> None:
        self.log_file: str | None = None
        self.created = False
        self.threads: int | None = None
        self.use_opencl: bool | None = None
        self.loaded_sim: str | None = None
        self.initialized = False
        self.advance_calls = 0
        self.run_full_calls = 0
        self.exported: Path | None = None
        self.export_filter: str | None = None
        self.stored_projects: list[Path] = []
        self.unloaded = False

    def setLogFile(self, path: bytes) -> None:
        self.log_file = path.decode("utf-8")

    def createInstance(self, _device: int, _group_size: int) -> None:
        self.created = True

    def setOmpNumThreads(self, threads: int) -> None:
        self.threads = int(threads)

    def setUseOpenCl(self, use_opencl: bool) -> None:
        self.use_opencl = bool(use_opencl)

    def loadSimDefinition(self, name: bytes) -> None:
        self.loaded_sim = name.decode("utf-8")

    def loadProject(self, name: bytes) -> None:
        self.loaded_sim = name.decode("utf-8")

    def initializeSimulation(self) -> None:
        self.initialized = True

    def advanceTurbineSimulation(self) -> bool:
        self.advance_calls += 1
        return True

    def runFullSimulation(self) -> bool:
        self.run_full_calls += 1
        return True

    def exportResults(self, _start: int, output_dir: bytes, result_name: bytes, suffix: bytes) -> None:
        result_path = Path(output_dir.decode("utf-8")) / f"{result_name.decode('utf-8')}.txt"
        result_path.write_text(
            "QBlade synthetic export\n"
            "Synthetic units\n"
            "Time~[s]\tLSS Rpm~[rpm]\n"
            "0.0\t0.0\n"
            "0.5\t0.0\n",
            encoding="utf-8",
        )
        self.exported = result_path
        self.export_filter = suffix.decode("utf-8")

    def storeProject(self, path: bytes) -> None:
        project_path = Path(path.decode("utf-8"))
        project_path.write_text("QBlade synthetic project\n", encoding="utf-8")
        self.stored_projects.append(project_path)

    def setExternalAction(self, *_args) -> None:
        pass

    def unload(self) -> None:
        self.unloaded = True


class FakeModalQBlade(FakeQBlade):
    def __init__(self, timestep_s: float, rpm: float) -> None:
        super().__init__()
        self.timestep_s = float(timestep_s)
        self.rpm = float(rpm)
        self.time_s = 0.0
        self.azimuth_deg = 0.0
        self.external_action_calls = 0

    def getCustomData_at_num(self, variable: bytes, _position: float, _index: int) -> float:
        if variable == b"Time [s]":
            return self.time_s
        if variable == b"LSS Azimuthal Pos. [deg]":
            return self.azimuth_deg
        return float("nan")

    def setExternalAction(self, *_args) -> None:
        self.external_action_calls += 1

    def advanceTurbineSimulation(self) -> bool:
        result = super().advanceTurbineSimulation()
        physical_index = max(0, self.advance_calls - 1)
        self.time_s = physical_index * self.timestep_s
        self.azimuth_deg = self.time_s * self.rpm * 6.0
        return result

    def exportResults(self, _start: int, output_dir: bytes, result_name: bytes, suffix: bytes) -> None:
        result_path = Path(output_dir.decode("utf-8")) / f"{result_name.decode('utf-8')}.txt"
        result_path.write_text(
            "QBlade synthetic modal export\n"
            "Synthetic units\n"
            "Time~[s]\tLSS Rpm~[rpm]\n"
            f"0.0\t{self.rpm}\n"
            f"{self.time_s}\t{self.rpm}\n",
            encoding="utf-8",
        )
        self.exported = result_path
        self.export_filter = suffix.decode("utf-8")


class ResumingFakeModalQBlade(FakeModalQBlade):
    def loadProject(self, name: bytes) -> None:
        super().loadProject(name)
        match = __import__("re").search(r"_step_(\d+)\.qpr$", self.loaded_sim or "")
        if match is None:
            raise AssertionError(f"Synthetic checkpoint name has no step: {self.loaded_sim}")
        self.advance_calls = int(match.group(1))
        physical_index = max(0, self.advance_calls - 1)
        self.time_s = physical_index * self.timestep_s
        self.azimuth_deg = self.time_s * self.rpm * 6.0


class ContaminatingFakeModalQBlade(FakeModalQBlade):
    def __init__(self, timestep_s: float, rpm: float, log_path: Path) -> None:
        super().__init__(timestep_s, rpm)
        self.native_log_path = log_path

    def unload(self) -> None:
        super().unload()
        self.native_log_path.write_text(
            "update_init entered\nupdate entered; count=1\n",
            encoding="utf-8",
        )


class RuntimeMutatingFakeModalQBlade(FakeModalQBlade):
    def __init__(self, timestep_s: float, rpm: float, target: Path) -> None:
        super().__init__(timestep_s, rpm)
        self.runtime_target = target
        self.mutated = False

    def storeProject(self, path: bytes) -> None:
        super().storeProject(path)
        if not self.mutated:
            self.runtime_target.write_bytes(
                self.runtime_target.read_bytes() + b"runtime mutation\n"
            )
            self.mutated = True


def _mutate_modal_runtime_input(
    kind: str,
    *,
    sim: Path,
    plan_path: Path,
    qblade_root: Path,
) -> None:
    if kind == "config":
        config_path = sim.parent / "Control" / "legacy.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["force_scale"] = 1.0
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if kind == "plan":
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["generalized_force_amplitude_n"] = 2.0
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if kind == "qblade_library":
        library = qblade_root / "libQBladeCE_test.so"
        library.write_bytes(library.read_bytes() + b"mutated main QBlade runtime\n")
        return
    raise AssertionError(f"Unknown modal runtime mutation: {kind}")


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rehash_modal_external_config(plan_path: Path, config_path: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["artifacts"]["external_config"]["sha256"] = _sha256_text(config_path)
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _compile_linux_callback_elf(path: Path, source: str) -> None:
    clang = shutil.which("clang")
    linker = shutil.which("ld.lld")
    if clang is None or linker is None:
        pytest.skip("An ELF-capable clang/lld toolchain is required for this regression.")
    completed = subprocess.run(
        [
            clang,
            "--target=x86_64-unknown-linux-gnu",
            "-fuse-ld=lld",
            "-shared",
            "-nostdlib",
            "-x",
            "c",
            "-",
            "-o",
            str(path),
        ],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"Could not build ELF adversary: {completed.stderr}")


def _write_modal_runner_case(tmp_path: Path, *, force_scale: float = 0.0):
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    external = (
        root
        / "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/Control"
        / "vorlap_10m_HVAWT_reference_external.json"
    )
    shapes = root / "run_headless/config/hvawt_response_mode_shapes.csv"
    duals = root / "run_headless/config/hvawt_response_mode_force_duals.csv"
    timestep_s = 0.01
    frequency_hz = 2.5115545017693828
    plan = ModalBurstPlan(
        timestep_s=timestep_s,
        total_wrapper_updates=202,
        bootstrap_zero=True,
        rotor_speed_rpm=1.0,
        target_mode=6,
        frequency_hz=frequency_hz,
        generalized_force_amplitude_n=1.0,
        burst_start_time_s=0.0,
        burst_cycles=5,
        azimuth_reference_deg=0.0,
        rotation_sign=1,
    )
    control = tmp_path / "Control"
    control.mkdir()
    legacy_config = control / "legacy.json"
    config = json.loads(external.read_text(encoding="utf-8"))
    config.update(
        {
            "force_scale": force_scale,
            "include_mean_load": False,
            "response_model": None,
            "source_parameter_dir": str(control.resolve()),
            "python_path_entries": [],
            "parameter_file": "Control/legacy.json",
        }
    )
    source_bridge = (
        root
        / "run_headless/QBladeCE_2.0.9.7/ControllerFiles/libvorlap_qblade_bridge.so"
    )
    config["native_bridge_runtime_identity"] = (
        build_modal_native_bridge_runtime_identity(source_bridge)
    )
    legacy_config.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    turbine = tmp_path / "case.trb"
    turbine.write_text(
        "libvorlap_qblade_bridge LIBFILE_1\n"
        "update LIBFUNCTION_1\n"
        "237 LIBARRAYSIZE_1\n"
        "Control/legacy.json LIBPARAMETERFILE_1\n",
        encoding="utf-8",
    )
    sim = tmp_path / "case.sim"
    sim.write_text(
        "\n".join(
            (
                "case.trb TURBFILE",
                "202 NUMTIMESTEPS",
                "0.01 TIMESTEP",
                "0 STOREFROM",
                "false STOREREPLAY",
                "0 ISOFFSHORE",
                "0 INITIAL_AZIMUTH",
                "0 RELAXSTEPS",
                "0.01 RELAXTIMESTEP",
                "1 PRESCRIBETYPE",
                "1 RPMPRESCRIBED",
                "0 INCLUDEAERO",
                "0 INCLUDEHYDRO",
                "0 RAMPUP",
                "0 ADDDAMP",
                "0 WAKEINTERACTION",
                "0 WNDTYPE",
                "0 MEANINF",
                "0 SURF_CURR_U",
                "0 SUB_CURR_U",
                "0 SHORE_CURR_U",
                " EVENTFILE",
                " LOADINGFILE",
                " SIMFILE",
                " MOTIONFILE",
                " WAVEFILE",
                " MOORINGSYSTEM",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    plan_path = tmp_path / "modal_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                **plan.as_dict(),
                "artifacts": {
                    "external_config": {
                        "path": str(legacy_config),
                        "sha256": _sha256_text(legacy_config),
                    },
                    "action_shapes": {"path": str(shapes), "sha256": _sha256_text(shapes)},
                    "force_duals": {"path": str(duals), "sha256": _sha256_text(duals)},
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return sim, plan_path, plan


def _write_modal_qblade_root(tmp_path: Path) -> tuple[Path, Path]:
    qblade_root = tmp_path / "QBladeCE"
    controller = qblade_root / "ControllerFiles"
    controller.mkdir(parents=True)
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    source_bridge = (
        Path(__file__).resolve().parents[1]
        / "run_headless/QBladeCE_2.0.9.7/ControllerFiles/libvorlap_qblade_bridge.so"
    )
    shutil.copy2(source_bridge, controller / source_bridge.name)
    libraries = qblade_root / "Libraries"
    libraries.mkdir()
    (libraries / "libChronoEngine.so").write_text(
        "chrono runtime placeholder\n",
        encoding="utf-8",
    )
    sil_interface = qblade_root / "SIL_Interface"
    sil_interface.mkdir()
    (sil_interface / "QBladeLibrary.py").write_text(
        "# synthetic SIL wrapper\n",
        encoding="utf-8",
    )
    return qblade_root, qblade_library


def _write_partial_modal_run(tmp_path: Path, *, stop_step: int = 50):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)
    fake = FakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm)
    output = tmp_path / "split"
    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=output,
            sil_modal_burst_plan=plan_path,
            checkpoint_every=stop_step,
            progress_every=0,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
            stop_requested=lambda: fake.advance_calls >= stop_step,
        )
    )
    checkpoint = Path(result.summary["restart"]["latest_checkpoint"])
    return sim, plan_path, plan, qblade_root, qblade_library, output, checkpoint


def test_run_qblade_case_service_runs_fake_qblade_and_writes_summary(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("3 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()
    messages: list[str] = []

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            threads=2,
            progress_every=1,
            command_line=["python", "run_qblade_sim.py", "--sim", str(sim)],
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
            message_callback=messages.append,
        )
    )

    assert fake.created is True
    assert fake.threads == 2
    assert fake.use_opencl is False
    assert fake.loaded_sim == sim.name
    assert fake.initialized is True
    assert fake.advance_calls == 3
    assert fake.unloaded is True
    assert fake.exported == result.result_path

    assert result.summary_path.is_file()
    assert result.result_path.is_file()
    assert json.loads(result.summary_path.read_text(encoding="utf-8")) == result.summary
    assert set(result.summary) == set(QBladeRunSummary.__dataclass_fields__)
    assert result.summary["requested_steps"] == 3
    assert result.summary["completed_steps"] == 3
    assert result.summary["qblade_execution_mode"] == "step-loop"
    assert result.summary["exported_results"] is True
    assert result.summary["sil_vorlap_history_mode"] == "off"
    assert result.summary["result_file_identity"]["sha256"] == file_sha256(result.result_path)["sha256"]
    assert result.summary["result_file_identity"]["size_bytes"] == result.result_path.stat().st_size
    assert result.summary["failed"] is False
    assert result.summary["qblade_export_time_diagnostics"]["available"] is True
    assert result.summary["qblade_export_rpm_diagnostics"]["available"] is True
    assert result.summary["qblade_export_rpm_diagnostics"]["max_abs_error_rpm"] == 0.0
    assert result.summary["reproducibility"]["command_line"] == [
        "python",
        "run_qblade_sim.py",
        "--sim",
        str(sim),
    ]
    reproducibility = result.summary["reproducibility"]
    assert reproducibility["qblade"]["library"]["sha256"]
    assert reproducibility["runtime_asset_fingerprint"]
    assert {Path(row["path"]).name for row in reproducibility["runtime_assets"]} == {"case.sim"}
    assert reproducibility["runtime_assets"][0]["sha256"]
    assert reproducibility["vorlap_python_runtime_fingerprint"]
    assert reproducibility["vorlap_python_runtime_assets"]
    assert all(row["sha256"] for row in reproducibility["vorlap_python_runtime_assets"])
    assert messages == [
        "Completed 1/3 timesteps",
        "Completed 2/3 timesteps",
        "Completed 3/3 timesteps",
    ]


@pytest.mark.requires_qblade
def test_modal_burst_runner_binds_contract_timing_actions_and_checkpoint(tmp_path):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)
    fake = FakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm)

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            sil_modal_burst_plan=plan_path,
            checkpoint_every=100,
            progress_every=0,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
        )
    )

    assert result.summary["coupling_mode"] == "sil-modal-burst"
    assert fake.external_action_calls == plan.total_wrapper_updates * 117
    modal = result.summary["sil_modal_burst"]
    assert modal["enabled"] is True
    assert modal["case_contract"]["passed"] is True
    assert modal["runtime_contract_final"]["passed"] is True
    assert modal["runtime_identity_audit_final"]["passed"] is True
    assert modal["runtime_dependency_fingerprint"] == result.summary[
        "reproducibility"
    ]["qblade_runtime_dependency_fingerprint"]
    assert modal["finalized"] is True
    assert modal["final_summary"]["ended_with_explicit_zero"] is True
    assert "does not claim" in modal["physical_consumption_claim"]
    witnesses = modal["timing_witnesses"]
    assert [row["wrapper_step_index"] for row in witnesses] == [0, 1, 201]
    assert witnesses[0]["bootstrap_zero"] is True
    assert witnesses[0]["qblade_time_pre_advance_s"] == 0.0
    assert witnesses[1]["physical_sample_index"] == 0
    assert witnesses[1]["qblade_time_pre_advance_s"] == 0.0
    assert witnesses[-1]["physical_sample_index"] == 200
    assert witnesses[-1]["qblade_time_pre_advance_s"] == pytest.approx(2.0)
    assert all(row["applied_action_count"] == 117 for row in witnesses)
    native = result.summary["native_callback_evidence"]
    assert int(native.get("update_init_count", 0) or 0) == 0
    assert int(native.get("update_count", 0) or 0) == 0

    checkpoint_path = Path(result.summary["restart"]["latest_checkpoint"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["version"] == 4
    assert checkpoint["sil_modal_burst_state"]["wrapper_updates"] == 202
    assert checkpoint["sil_modal_burst_resume_position"][
        "expected_post_advance_physical_sample_index"
    ] == 201
    assert checkpoint["sil_modal_burst_resume_position"][
        "qblade_time_post_advance_s"
    ] == pytest.approx(2.01)
    assert checkpoint["qblade_runtime_dependency_audit"]["fingerprint"] == modal[
        "runtime_dependency_fingerprint"
    ]
    assert checkpoint["modal_runtime_identity_audit"] == modal[
        "runtime_identity_audit_final"
    ]
    assert checkpoint["modal_runtime_identity_audit"][
        "checkpoint_identity_fingerprint"
    ] == checkpoint["checkpoint_identity_fingerprint"]
    assert [
        row["wrapper_step_index"]
        for row in checkpoint["sil_modal_burst_timing_witnesses"]
    ] == [0, 1, 201]
    assert (
        checkpoint["checkpoint_identity"]["modal_burst_identity"]["fingerprint"]
        == modal["identity"]["fingerprint"]
    )


@pytest.mark.requires_qblade
def test_modal_burst_split_resume_matches_uninterrupted_force_and_timing_summary(tmp_path):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)
    full_fake = FakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm)
    full = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "full",
            sil_modal_burst_plan=plan_path,
            checkpoint_every=0,
            progress_every=0,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (full_fake, qblade_library),
        )
    )

    first_fake = FakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm)
    split_output = tmp_path / "split"
    first = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=split_output,
            sil_modal_burst_plan=plan_path,
            checkpoint_every=50,
            progress_every=0,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (first_fake, qblade_library),
            stop_requested=lambda: first_fake.advance_calls >= 100,
        )
    )
    assert first.summary["stopped"] is True
    assert first.summary["completed_steps"] == 100
    assert first.summary["sil_modal_burst"]["finalized"] is False

    second_fake = ResumingFakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm)
    resumed = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=split_output,
            sil_modal_burst_plan=plan_path,
            checkpoint_every=50,
            resume_from_checkpoint=True,
            progress_every=0,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (second_fake, qblade_library),
        )
    )

    assert resumed.summary["restart"]["resume_start_step"] == 100
    assert resumed.summary["sil_modal_burst"]["loaded_resume_position_audit"][
        "expected_post_advance_physical_sample_index"
    ] == 99
    assert first_fake.external_action_calls + second_fake.external_action_calls == (
        plan.total_wrapper_updates * 117
    )
    assert resumed.summary["sil_modal_burst"]["identity"] == full.summary[
        "sil_modal_burst"
    ]["identity"]
    assert resumed.summary["sil_modal_burst"]["timing_witnesses"] == full.summary[
        "sil_modal_burst"
    ]["timing_witnesses"]
    assert resumed.summary["sil_modal_burst"]["final_summary"] == full.summary[
        "sil_modal_burst"
    ]["final_summary"]


@pytest.mark.parametrize(
    ("fake_factory", "passes"),
    (
        (lambda plan: ResumingFakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm), True),
        (lambda plan: FakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm), False),
    ),
)
@pytest.mark.requires_qblade
def test_modal_burst_full_step_resume_validates_loaded_qpr_position(
    tmp_path,
    fake_factory,
    passes,
):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)
    output = tmp_path / "out"
    first_fake = FakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm)
    first = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=output,
            sil_modal_burst_plan=plan_path,
            checkpoint_every=100,
            progress_every=0,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (first_fake, qblade_library),
        )
    )
    checkpoint = Path(first.summary["restart"]["latest_checkpoint"])
    second_fake = fake_factory(plan)
    request = QBladeRunRequest(
        sim_path=sim,
        qblade_root=qblade_root,
        output_dir=output,
        sil_modal_burst_plan=plan_path,
        resume_checkpoint=checkpoint,
        checkpoint_every=100,
        progress_every=0,
        manage_loader_env=False,
        qblade_api_loader=lambda _root: (second_fake, qblade_library),
    )

    if passes:
        resumed = run_qblade_case(request)
        assert resumed.summary["completed_steps"] == plan.total_wrapper_updates
        assert second_fake.advance_calls == plan.total_wrapper_updates
        assert second_fake.external_action_calls == 0
        assert resumed.summary["sil_modal_burst"][
            "loaded_resume_position_audit"
        ]["qblade_time_post_advance_s"] == pytest.approx(2.01)
    else:
        with pytest.raises(RuntimeError, match="pre-advance time does not match"):
            run_qblade_case(request)
        assert second_fake.advance_calls == 0
        assert second_fake.external_action_calls == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("extra_top_level", "exact top-level schema"),
        ("duplicate_witness", "duplicate timing witness"),
        ("forged_witness", "not deterministic"),
        ("missing_zero_evidence", "exact top-level schema"),
        ("missing_runtime_identity_audit", "exact top-level schema"),
        ("forged_runtime_identity_audit", "does not match this plan/runtime"),
        ("state_step_mismatch", "does not match wrapper_updates"),
        ("extra_resume_position_key", "exact locked schema keys"),
        ("forged_resume_position", "time-grid audit is inconsistent"),
    ),
)
@pytest.mark.requires_qblade
def test_modal_burst_v4_resume_rejects_forged_schema_and_witnesses(
    tmp_path,
    mutation,
    message,
):
    (
        sim,
        plan_path,
        plan,
        qblade_root,
        qblade_library,
        output,
        checkpoint_path,
    ) = _write_partial_modal_run(tmp_path)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if mutation == "extra_top_level":
        checkpoint["unexpected"] = True
    elif mutation == "duplicate_witness":
        checkpoint["sil_modal_burst_timing_witnesses"].append(
            dict(checkpoint["sil_modal_burst_timing_witnesses"][0])
        )
        from vorlap.qblade_run_support import canonical_fingerprint

        checkpoint["sil_modal_burst_timing_witnesses_fingerprint"] = canonical_fingerprint(
            checkpoint["sil_modal_burst_timing_witnesses"]
        )
    elif mutation == "forged_witness":
        checkpoint["sil_modal_burst_timing_witnesses"][0]["global_force_sha256"] = "0" * 64
        from vorlap.qblade_run_support import canonical_fingerprint

        checkpoint["sil_modal_burst_timing_witnesses_fingerprint"] = canonical_fingerprint(
            checkpoint["sil_modal_burst_timing_witnesses"]
        )
    elif mutation == "missing_zero_evidence":
        checkpoint.pop("zero_native_callback_evidence")
    elif mutation == "missing_runtime_identity_audit":
        checkpoint.pop("modal_runtime_identity_audit")
    elif mutation == "forged_runtime_identity_audit":
        checkpoint["modal_runtime_identity_audit"]["plan_file"] = "/forged/plan.json"
        from vorlap.qblade_run_support import canonical_fingerprint

        checkpoint["modal_runtime_identity_audit_fingerprint"] = canonical_fingerprint(
            checkpoint["modal_runtime_identity_audit"]
        )
    elif mutation == "state_step_mismatch":
        checkpoint["sil_modal_burst_state"]["wrapper_updates"] = 49
    elif mutation == "extra_resume_position_key":
        checkpoint["sil_modal_burst_resume_position"]["unexpected"] = True
        from vorlap.qblade_run_support import canonical_fingerprint

        checkpoint["sil_modal_burst_resume_position_fingerprint"] = canonical_fingerprint(
            checkpoint["sil_modal_burst_resume_position"]
        )
    elif mutation == "forged_resume_position":
        checkpoint["sil_modal_burst_resume_position"][
            "qblade_time_post_advance_s"
        ] = 0.0
        from vorlap.qblade_run_support import canonical_fingerprint

        checkpoint["sil_modal_burst_resume_position_fingerprint"] = canonical_fingerprint(
            checkpoint["sil_modal_burst_resume_position"]
        )
    forged = tmp_path / f"{mutation}.json"
    forged.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                sil_modal_burst_plan=plan_path,
                resume_checkpoint=forged,
                checkpoint_every=50,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (
                    ResumingFakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm),
                    qblade_library,
                ),
            )
        )


@pytest.mark.requires_qblade
def test_modal_burst_final_callback_contamination_discards_final_checkpoint(tmp_path):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)
    log_path = sim.parent / "Control" / "vorlap_qblade_bridge_cpp.log"
    fake = ContaminatingFakeModalQBlade(
        plan.timestep_s,
        plan.rotor_speed_rpm,
        log_path,
    )
    output = tmp_path / "out"

    with pytest.raises(RuntimeError, match="native bridge log already exists"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                sil_modal_burst_plan=plan_path,
                checkpoint_every=100,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )

    summary = json.loads((output / "case_run_summary.json").read_text(encoding="utf-8"))
    assert summary["failed"] is True
    assert summary["restart"]["latest_checkpoint"] is None
    removed = summary["restart"]["discarded_contaminated_checkpoint_files"]
    assert any(path.endswith("_step_0000202.json") for path in removed)
    assert any(path.endswith("_step_0000202.qpr") for path in removed)
    assert not (output / "checkpoints" / "latest_checkpoint.json").exists()
    assert not (output / "checkpoints" / "case_step_0000202.json").exists()
    assert not (output / "checkpoints" / "case_step_0000202.qpr").exists()


@pytest.mark.parametrize(
    ("runtime_target", "message"),
    (
        ("bridge", "approved hash-bound runtime identity"),
        ("dependency", "dependency fingerprint changed"),
    ),
)
@pytest.mark.requires_qblade
def test_modal_checkpoint_reaudits_bridge_and_dependencies_before_publication(
    tmp_path,
    runtime_target,
    message,
):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)
    target = (
        qblade_root / "ControllerFiles" / "libvorlap_qblade_bridge.so"
        if runtime_target == "bridge"
        else qblade_root / "Libraries" / "libChronoEngine.so"
    )
    fake = RuntimeMutatingFakeModalQBlade(
        plan.timestep_s,
        plan.rotor_speed_rpm,
        target,
    )
    output = tmp_path / "out"

    with pytest.raises((ValueError, RuntimeError), match=message):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                sil_modal_burst_plan=plan_path,
                checkpoint_every=50,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )

    checkpoint_dir = output / "checkpoints"
    assert not (checkpoint_dir / "case_step_0000050.qpr").exists()
    assert not (checkpoint_dir / "case_step_0000050.json").exists()
    assert not (checkpoint_dir / "latest_checkpoint.json").exists()


@pytest.mark.parametrize("mutable_root", ("output", "checkpoint"))
@pytest.mark.requires_qblade
def test_modal_burst_prohibits_mutable_run_roots_inside_staged_case(
    tmp_path,
    mutable_root,
):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)
    model_dir = sim.parent / "model"
    model_dir.mkdir()
    turbine = sim.parent / "case.trb"
    nested_turbine = model_dir / turbine.name
    turbine.rename(nested_turbine)
    nested_turbine.write_text(
        nested_turbine.read_text(encoding="utf-8").replace(
            "Control/legacy.json LIBPARAMETERFILE_1",
            "../Control/legacy.json LIBPARAMETERFILE_1",
        ),
        encoding="utf-8",
    )
    sim.write_text(
        sim.read_text(encoding="utf-8").replace(
            "case.trb TURBFILE",
            "model/case.trb TURBFILE",
        ),
        encoding="utf-8",
    )
    output = model_dir if mutable_root == "output" else tmp_path / "out"
    checkpoint_dir = model_dir if mutable_root == "checkpoint" else None
    fake = FakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm)

    with pytest.raises(ValueError, match="outside the immutable staged case root"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                checkpoint_dir=checkpoint_dir,
                sil_modal_burst_plan=plan_path,
                checkpoint_every=50,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )

    assert fake.stored_projects == []


@pytest.mark.requires_qblade
def test_modal_burst_containment_preflight_preserves_staged_files_and_paths(tmp_path):
    sim, plan_path, _plan = _write_modal_runner_case(tmp_path / "inputs")
    sentinel = sim.parent / "victim.txt"
    sentinel.write_text("immutable staged sentinel\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the immutable staged case root"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=sim.parent,
                result_name="victim",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )

    assert sentinel.read_text(encoding="utf-8") == "immutable staged sentinel\n"

    nested_output = sim.parent / "not-created-output"
    with pytest.raises(ValueError, match="outside the immutable staged case root"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=nested_output,
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )

    assert not nested_output.exists()


@pytest.mark.parametrize(
    ("runtime_target", "message"),
    (
        ("config", "artifact SHA-256 mismatch"),
        ("plan", "plan or plan-bound artifact identity changed"),
        ("qblade_library", "complete checkpoint runtime identity changed"),
    ),
)
@pytest.mark.requires_qblade
def test_modal_checkpoint_reaudits_plan_config_and_main_library_before_publication(
    tmp_path,
    runtime_target,
    message,
):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)

    class FullRuntimeMutatingFakeModalQBlade(FakeModalQBlade):
        def storeProject(self, path: bytes) -> None:
            super().storeProject(path)
            if not getattr(self, "runtime_mutated", False):
                _mutate_modal_runtime_input(
                    runtime_target,
                    sim=sim,
                    plan_path=plan_path,
                    qblade_root=qblade_root,
                )
                self.runtime_mutated = True

    output = tmp_path / "out"
    fake = FullRuntimeMutatingFakeModalQBlade(
        plan.timestep_s,
        plan.rotor_speed_rpm,
    )

    with pytest.raises((ValueError, RuntimeError), match=message):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                sil_modal_burst_plan=plan_path,
                checkpoint_every=50,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )

    checkpoint_dir = output / "checkpoints"
    assert not (checkpoint_dir / "case_step_0000050.qpr").exists()
    assert not (checkpoint_dir / "case_step_0000050.json").exists()
    assert not (checkpoint_dir / "latest_checkpoint.json").exists()


@pytest.mark.parametrize(
    ("runtime_target", "message"),
    (
        ("config", "artifact SHA-256 mismatch"),
        ("plan", "plan or plan-bound artifact identity changed"),
        ("qblade_library", "complete checkpoint runtime identity changed"),
    ),
)
@pytest.mark.requires_qblade
def test_modal_final_audit_rebinds_plan_config_and_main_library(
    tmp_path,
    runtime_target,
    message,
):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)

    class FinalRuntimeMutatingFakeModalQBlade(FakeModalQBlade):
        def unload(self) -> None:
            super().unload()
            _mutate_modal_runtime_input(
                runtime_target,
                sim=sim,
                plan_path=plan_path,
                qblade_root=qblade_root,
            )

    output = tmp_path / "out"
    fake = FinalRuntimeMutatingFakeModalQBlade(
        plan.timestep_s,
        plan.rotor_speed_rpm,
    )

    with pytest.raises(RuntimeError, match=message):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                sil_modal_burst_plan=plan_path,
                checkpoint_every=100,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )

    summary = json.loads((output / "case_run_summary.json").read_text(encoding="utf-8"))
    assert summary["failed"] is True
    assert summary["restart"]["latest_checkpoint"] is None
    assert not (output / "checkpoints" / "latest_checkpoint.json").exists()


@pytest.mark.requires_qblade
def test_modal_final_audit_removes_completed_project_created_before_runtime_mutation(
    tmp_path,
):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)

    class StoredProjectMutatingFakeModalQBlade(FakeModalQBlade):
        def storeProject(self, path: bytes) -> None:
            super().storeProject(path)
            _mutate_modal_runtime_input(
                "qblade_library",
                sim=sim,
                plan_path=plan_path,
                qblade_root=qblade_root,
            )

    output = tmp_path / "out"
    fake = StoredProjectMutatingFakeModalQBlade(
        plan.timestep_s,
        plan.rotor_speed_rpm,
    )

    with pytest.raises(RuntimeError, match="complete checkpoint runtime identity changed"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                sil_modal_burst_plan=plan_path,
                store_project=True,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )

    summary = json.loads((output / "case_run_summary.json").read_text(encoding="utf-8"))
    project = output / "case_completed.qpr"
    assert summary["failed"] is True
    assert summary["stored_project"] is None
    assert summary["restart"]["discarded_contaminated_stored_project"] == str(project)
    assert not project.exists()


@pytest.mark.requires_qblade
def test_modal_checkpoint_rejects_nonregular_log_created_after_preflight(tmp_path):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)
    log_path = sim.parent / "Control" / "vorlap_qblade_bridge_cpp.log"

    class LogTargetMutatingFakeModalQBlade(FakeModalQBlade):
        def storeProject(self, path: bytes) -> None:
            super().storeProject(path)
            log_path.mkdir()

    output = tmp_path / "out"
    fake = LogTargetMutatingFakeModalQBlade(
        plan.timestep_s,
        plan.rotor_speed_rpm,
    )

    with pytest.raises(RuntimeError, match="native bridge log already exists"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                sil_modal_burst_plan=plan_path,
                checkpoint_every=50,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )

    checkpoint_dir = output / "checkpoints"
    assert not (checkpoint_dir / "case_step_0000050.qpr").exists()
    assert not (checkpoint_dir / "case_step_0000050.json").exists()
    assert not (checkpoint_dir / "latest_checkpoint.json").exists()


@pytest.mark.requires_qblade
def test_modal_burst_completed_checkpoint_cannot_be_laundered_after_callback_log(tmp_path):
    sim, plan_path, plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, qblade_library = _write_modal_qblade_root(tmp_path)
    fake = FakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm)
    output = tmp_path / "out"
    run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=output,
            sil_modal_burst_plan=plan_path,
            checkpoint_every=100,
            progress_every=0,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
        )
    )
    (sim.parent / "Control" / "vorlap_qblade_bridge_cpp.log").write_text(
        "update_init entered\nupdate entered; count=1\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="already exists"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                sil_modal_burst_plan=plan_path,
                resume_from_checkpoint=True,
                checkpoint_every=100,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (
                    ResumingFakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm),
                    qblade_library,
                ),
            )
        )


@pytest.mark.requires_qblade
def test_modal_burst_runner_fails_closed_on_active_legacy_source(tmp_path):
    sim, plan_path, _plan = _write_modal_runner_case(
        tmp_path / "inputs",
        force_scale=1.0,
    )

    with pytest.raises(ValueError, match="force_scale exactly 0.0"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=tmp_path / "out",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )


@pytest.mark.requires_qblade
def test_modal_burst_runner_requires_plan_external_config_to_be_live_staged_config(tmp_path):
    sim, plan_path, _plan = _write_modal_runner_case(tmp_path / "inputs")
    original_config = sim.parent / "Control" / "legacy.json"
    alternate = sim.parent / "Control" / "alternate.json"
    alternate.write_bytes(original_config.read_bytes())
    turbine = sim.parent / "case.trb"
    turbine.write_text(
        turbine.read_text(encoding="utf-8").replace(
            "Control/legacy.json LIBPARAMETERFILE_1",
            "Control/alternate.json LIBPARAMETERFILE_1",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact staged LIBPARAMETERFILE_1"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=tmp_path / "out",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )


@pytest.mark.requires_qblade
def test_modal_burst_runner_rejects_alternate_loading_and_log_env(tmp_path, monkeypatch):
    sim, plan_path, _plan = _write_modal_runner_case(tmp_path / "inputs")
    sim.write_text(
        sim.read_text(encoding="utf-8").replace(" LOADINGFILE", "loads.lod LOADINGFILE"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blank LOADINGFILE"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=tmp_path / "out",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )

    sim, plan_path, _plan = _write_modal_runner_case(tmp_path / "second-inputs")
    monkeypatch.setenv("VORLAP_QBLADE_BRIDGE_LOG_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="prohibit.*VORLAP_QBLADE_BRIDGE_LOG_DIR"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=tmp_path / "out2",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )


@pytest.mark.requires_qblade
def test_modal_burst_runner_rejects_raw_whitespace_log_override(tmp_path, monkeypatch):
    sim, plan_path, _plan = _write_modal_runner_case(tmp_path / "inputs")
    monkeypatch.setenv("VORLAP_QBLADE_BRIDGE_LOG_DIR", "   ")

    with pytest.raises(ValueError, match="raw nonempty"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=tmp_path / "out",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )


@pytest.mark.parametrize("mutation", ("duplicate", "unicode_escape"))
@pytest.mark.requires_qblade
def test_modal_burst_runner_rejects_python_cpp_source_directory_divergence(
    tmp_path,
    mutation,
):
    sim, plan_path, _plan = _write_modal_runner_case(tmp_path / "inputs")
    config_path = sim.parent / "Control" / "legacy.json"
    text = config_path.read_text(encoding="utf-8")
    source_dir = str(config_path.parent.resolve())
    member = f'  "source_parameter_dir": "{source_dir}",'
    if mutation == "duplicate":
        replacement = (
            '  "source_parameter_dir": "/tmp/native-first-wrong",\n' + member
        )
    else:
        escaped = "\\u002f" + source_dir.lstrip("/")
        replacement = f'  "source_parameter_dir": "{escaped}",'
    config_path.write_text(text.replace(member, replacement), encoding="utf-8")
    _rehash_modal_external_config(plan_path, config_path)

    with pytest.raises(ValueError, match="strict JSON|canonical sorted JSON"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=tmp_path / "out",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )


@pytest.mark.parametrize(
    "entry_kind",
    ("directory", "dangling_symlink", "file_symlink", "fifo"),
)
@pytest.mark.requires_qblade
def test_modal_burst_runner_rejects_any_preexisting_callback_log_entry(
    tmp_path,
    entry_kind,
):
    sim, plan_path, _plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, _qblade_library = _write_modal_qblade_root(tmp_path)
    log_path = sim.parent / "Control" / "vorlap_qblade_bridge_cpp.log"
    if entry_kind == "directory":
        log_path.mkdir()
    elif entry_kind == "dangling_symlink":
        log_path.symlink_to(log_path.with_name("missing-target"))
    elif entry_kind == "file_symlink":
        target = log_path.with_name("callback-target.log")
        target.write_text("", encoding="utf-8")
        log_path.symlink_to(target)
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("mkfifo is unavailable on this platform")
        os.mkfifo(log_path)

    with pytest.raises(RuntimeError, match="native bridge log already exists"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=tmp_path / "out",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )


@pytest.mark.requires_qblade
def test_modal_burst_runner_rejects_noop_callback_elf_not_approved_bridge(tmp_path):
    sim, plan_path, _plan = _write_modal_runner_case(tmp_path / "inputs")
    qblade_root, _qblade_library = _write_modal_qblade_root(tmp_path)
    bridge = qblade_root / "ControllerFiles" / "libvorlap_qblade_bridge.so"
    _compile_linux_callback_elf(
        bridge,
        "void update(float *x) {(void)x;}\n"
        "void update_init(float *x) {(void)x;}\n"
        "void update_message(char *x) {(void)x;}\n",
    )

    with pytest.raises(ValueError, match="do not match the approved"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=tmp_path / "out",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )


@pytest.mark.requires_qblade
def test_modal_burst_resume_binds_qblade_runtime_dependencies(tmp_path):
    (
        sim,
        plan_path,
        plan,
        qblade_root,
        qblade_library,
        output,
        checkpoint_path,
    ) = _write_partial_modal_run(tmp_path)
    dependency = qblade_root / "Libraries" / "libChronoEngine.so"
    dependency.write_text("mutated chrono runtime\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=output,
                sil_modal_burst_plan=plan_path,
                resume_checkpoint=checkpoint_path,
                checkpoint_every=50,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (
                    ResumingFakeModalQBlade(plan.timestep_s, plan.rotor_speed_rpm),
                    qblade_library,
                ),
            )
        )

@pytest.mark.parametrize("empty_plan", ("", "   "))
def test_modal_burst_runner_rejects_supplied_empty_plan(tmp_path, empty_plan):
    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS\n", encoding="utf-8")
    with pytest.raises(ValueError, match="supplied but is empty or whitespace"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=tmp_path / "out",
                sil_modal_burst_plan=empty_plan,
                manage_loader_env=False,
            )
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ("202.5 NUMTIMESTEPS", "finite integral token"),
        ("NaN NUMTIMESTEPS", "finite integral token"),
        ("Inf NUMTIMESTEPS", "finite integral token"),
        (
            "202 NUMTIMESTEPS\n202 NUMTIMESTEPS",
            "exactly one unambiguous declaration",
        ),
    ),
)
@pytest.mark.requires_qblade
def test_modal_burst_rejects_fractional_nonfinite_or_duplicate_declared_steps(
    tmp_path,
    replacement,
    message,
):
    sim, plan_path, _plan = _write_modal_runner_case(tmp_path / "inputs")
    sim.write_text(
        sim.read_text(encoding="utf-8").replace("202 NUMTIMESTEPS", replacement),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=tmp_path / "out",
                sil_modal_burst_plan=plan_path,
                manage_loader_env=False,
            )
        )


def test_legacy_runner_identity_and_summary_do_not_gain_modal_keys(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            coupling_mode="none",
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (FakeQBlade(), qblade_library),
        )
    )

    assert set(result.summary) == set(QBladeRunSummary.__dataclass_fields__)
    assert "sil_modal_burst" not in result.summary
    assert "modal_burst_identity" not in result.summary["restart"]


def test_run_qblade_case_rejects_compact_history_without_sil_vorlap(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS - steps\n", encoding="utf-8")

    with pytest.raises(ValueError, match="history recording requires sil_vorlap=True"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "unused-qblade",
                output_dir=tmp_path / "out",
                sil_vorlap_history_mode="compact",
                manage_loader_env=False,
            )
        )


def test_run_qblade_case_passes_explicit_result_filter_to_export(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS - steps\n", encoding="utf-8")
    result_filter = tmp_path / "fatigue.filter"
    result_filter.write_text("Time [s]\nLSS Rpm [rpm]\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            result_filter=result_filter,
            coupling_mode="none",
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
        )
    )

    assert fake.export_filter == str(result_filter.resolve())
    assert result.summary["result_filter"] == str(result_filter.resolve())
    assert result.summary["result_filter_identity"]["sha256"] == file_sha256(result_filter)["sha256"]
    assert result.summary["reproducibility"]["input_files"]["result_filter"]["sha256"]


def test_sil_resume_requires_and_restores_causal_checkpoint_state():
    class Coupler:
        def __init__(self) -> None:
            self.restored: dict[str, object] | None = None

        def restore_checkpoint_state(self, state: dict[str, object]) -> None:
            self.restored = state

    coupler = Coupler()
    with pytest.raises(ValueError, match="missing a valid sil_vorlap_state"):
        qblade_run_module._restore_sil_vorlap_resume_state(
            coupler,
            {"completed_steps": 3},
            completed_checkpoint_export_only=False,
        )

    qblade_run_module._restore_sil_vorlap_resume_state(
        coupler,
        {"completed_steps": 3, "sil_vorlap_state": {"updates": 3}},
        completed_checkpoint_export_only=False,
    )
    assert coupler.restored == {"updates": 3}

    export_only_coupler = Coupler()
    qblade_run_module._restore_sil_vorlap_resume_state(
        export_only_coupler,
        {"completed_steps": 3},
        completed_checkpoint_export_only=True,
    )
    assert export_only_coupler.restored is None


def test_completed_checkpoint_export_only_skips_identity_match_and_timesteps(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("3 NUMTIMESTEPS - steps\n", encoding="utf-8")
    result_filter = tmp_path / "fatigue.filter"
    result_filter.write_text("Time [s]\nLSS Rpm [rpm]\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    project = checkpoint_dir / "case_step_0000003.qpr"
    project.write_text("completed synthetic QBlade project\n", encoding="utf-8")
    project_identity = file_sha256(project)
    metadata = {
        "version": 3,
        "sim": str(sim.resolve()),
        "qblade_root": str(qblade_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "result_name": "case_results",
        "coupling_mode": "none",
        "requested_steps": 3,
        "completed_steps": 3,
        "qblade_project": str(project),
        "qblade_project_sha256": project_identity["sha256"],
        "qblade_project_size_bytes": project_identity["size_bytes"],
        "checkpoint_identity_fingerprint": "old-python-tree-fingerprint",
        "checkpoint_identity": {"fingerprint": "old-python-tree-fingerprint"},
    }
    metadata_path = checkpoint_dir / "case_step_0000003.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (checkpoint_dir / "latest_checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")
    fake = FakeQBlade()
    messages: list[str] = []

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=output_dir,
            result_filter=result_filter,
            checkpoint_every=1,
            resume_from_checkpoint=True,
            completed_checkpoint_export_only=True,
            coupling_mode="none",
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
            message_callback=messages.append,
        )
    )

    assert fake.loaded_sim == str(project)
    assert fake.initialized is False
    assert fake.advance_calls == 0
    assert fake.export_filter == str(result_filter.resolve())
    assert result.summary["completed_steps"] == 3
    assert result.summary["restart"]["completed_checkpoint_export_only"] is True
    assert result.summary["restart"]["checkpoint_identity_fingerprint"] == "old-python-tree-fingerprint"
    assert result.summary["restart"]["checkpoint_current_identity_fingerprint"] != "old-python-tree-fingerprint"
    assert any("without initialization or timestep advancement" in message for message in messages)


def test_run_qblade_case_can_delegate_qblade_external_case_to_run_full_simulation(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("2 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()
    messages: list[str] = []

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            coupling_mode="qblade-external",
            qblade_execution_mode="run-full-simulation",
            progress_every=1,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
            message_callback=messages.append,
        )
    )

    assert fake.initialized is True
    assert fake.advance_calls == 0
    assert fake.run_full_calls == 1
    assert fake.exported == result.result_path
    assert result.summary["qblade_execution_mode"] == "run-full-simulation"
    assert result.summary["coupling_mode"] == "qblade-external"
    assert result.summary["requested_steps"] == 2
    assert result.summary["completed_steps"] == 2
    assert result.summary["exported_results"] is True
    assert messages == [
        "Starting QBlade runFullSimulation() for 2 declared timesteps.",
        "Completed 2/2 timesteps via runFullSimulation().",
    ]


def test_run_qblade_case_run_full_requires_qblade_external_coupling(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("2 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()

    try:
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=tmp_path / "out",
                coupling_mode="sil-vorlap",
                sil_vorlap=True,
                qblade_execution_mode="run-full-simulation",
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )
    except ValueError as exc:
        assert "--native-run-full-simulation requires --coupling-mode qblade-external" in str(exc)
    else:
        raise AssertionError("run-full-simulation should reject non-native coupling")


def test_run_qblade_case_run_full_rejects_steps_mismatch(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("2 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()

    try:
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=tmp_path / "out",
                steps=3,
                coupling_mode="qblade-external",
                qblade_execution_mode="run-full-simulation",
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )
    except ValueError as exc:
        assert "requested 3 steps but the .sim declares 2" in str(exc)
    else:
        raise AssertionError("run-full-simulation should reject steps mismatches")


def test_run_qblade_case_summary_reports_latest_checkpoint_project(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()
    checkpoint_dir = tmp_path / "checkpoints"

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            checkpoint_every=1,
            checkpoint_dir=checkpoint_dir,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
        )
    )

    restart = result.summary["restart"]
    assert restart["latest_checkpoint"].endswith("case_step_0000001.json")
    assert restart["latest_checkpoint_completed_steps"] == 1
    assert restart["latest_checkpoint_project"].endswith("case_step_0000001.qpr")
    assert Path(restart["latest_checkpoint"]).is_file()
    assert Path(restart["latest_checkpoint_project"]).is_file()
    assert fake.stored_projects == [checkpoint_dir / "case_step_0000001.tmp.qpr"]


def test_run_qblade_case_records_completed_project_when_requested(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("2 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            store_project=True,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
        )
    )

    stored_project = result.summary["stored_project"]
    assert stored_project.endswith("case_completed.qpr")
    assert Path(stored_project).is_file()
    assert fake.stored_projects == [Path(stored_project)]


def test_run_rejects_empty_completed_project(tmp_path):
    class EmptyProjectQBlade(FakeQBlade):
        def storeProject(self, path: bytes) -> None:
            Path(path.decode("utf-8")).write_bytes(b"")

    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="nonempty completed project"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=tmp_path / "out",
                coupling_mode="none",
                store_project=True,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (EmptyProjectQBlade(), qblade_library),
            )
        )

    assert not (tmp_path / "out" / "case_completed.qpr").exists()


def test_run_qblade_case_emits_live_sil_vorlap_diagnostics_in_progress_messages(tmp_path, monkeypatch):
    sim = tmp_path / "case.sim"
    sim.write_text("2 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()
    messages: list[str] = []
    coupler_lifecycle: dict[str, object] = {}

    class FakeSilVorLapCoupler:
        def __init__(
            self,
            sim_path,
            *,
            record_history=False,
            history_mode=None,
            history_path=None,
            resume_history=False,
        ):
            self.parameter_file = Path(sim_path)
            self.config = {"geometry_audit_file": "Control/vorlap_qblade_geometry_audit.csv"}
            self.record_history = bool(record_history)
            self.history_mode = history_mode or ("full" if record_history else "off")
            self.history_path = None if history_path is None else Path(history_path)
            self.resume_history = bool(resume_history)
            coupler_lifecycle["history_path"] = self.history_path
            coupler_lifecycle["resume_history"] = self.resume_history
            self.runtime = SimpleNamespace(
                controller=SimpleNamespace(cache_stats={"hits": 0, "misses": 0}, force_scale=100.0),
                response_diagnostics_summary={
                    "model_type": "modal_aerodynamic_power",
                    "samples": 0,
                    "target_power_limited_steps": 0,
                },
                source_bounds_policy="raise",
                _ambient_velocity_global=[3.0, 0.0, 0.0],
                _last_controller_structural_velocity=[],
            )
            self.max_abs_force = 0.0
            self.updates = 0
            self.last_message = ""

        def validate_channels(self, _qblade) -> None:
            return None

        def step(self, _qblade, step_index: int) -> str:
            self.updates = step_index + 1
            self.max_abs_force = 10.0 * self.updates
            self.runtime.controller.cache_stats = {"hits": self.updates, "misses": 2}
            self.runtime.response_diagnostics_summary["samples"] = self.updates
            self.runtime._last_controller_structural_velocity = [[0.5 * self.updates, 0.0, 0.0]]
            self.last_message = f"fake SIL step {self.updates}"
            return self.last_message

        def velocity_diagnostics(self) -> dict[str, object]:
            return {
                "velocity_channel_count": 3,
                "axis_stats": {
                    "X": {"samples": self.updates, "max_abs": 1.25 * self.updates},
                    "Y": {"samples": self.updates, "max_abs": 0.0},
                    "Z": {"samples": self.updates, "max_abs": 0.0},
                },
                "all_zero_axes": ["Z"],
                "all_zero_channels": ["Z_g Position BLD_1_0.000000 [m/s]"],
            }

        def timing_diagnostics(self) -> dict[str, object]:
            wrapper_time = 0.25 * self.updates
            return {
                "samples": self.updates,
                "last_sample": {
                    "step_index": self.updates - 1,
                    "wrapper_time_s": wrapper_time,
                    "wrapper_timestep": float(self.updates - 1),
                    "qblade_time_pre_advance_s": wrapper_time + 0.01,
                    "azimuth_deg": 15.0 * self.updates,
                    "applied_force_time_s": wrapper_time,
                    "time_delta_qblade_minus_wrapper_s": 0.01,
                    "force_application_order": "before_advance",
                },
            }

        def source_coverage(self) -> list[dict[str, object]]:
            return [
                {
                    "component": "BLD_1",
                    "samples": self.updates,
                    "source_clip_count": self.updates,
                }
            ]

        def write_history(self, path):
            path.write_text("step_index,time_s\n0,0.0\n", encoding="utf-8")
            return {
                "path": str(path),
                "rows": 1,
                "columns": 2,
                "node_count": 1,
                "mode": self.history_mode,
                "schema": (
                    "vorlap-sil-force-history-compact-v1"
                    if self.history_mode == "compact"
                    else "vorlap-sil-force-history-v2"
                ),
            }

        def close_history(self, *, durable=False):
            coupler_lifecycle["closed_durable"] = bool(durable)

    monkeypatch.setattr("vorlap.qblade_run.SilVorLapCoupler", FakeSilVorLapCoupler)

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            progress_every=1,
            sil_vorlap=True,
            sil_vorlap_history_mode="compact",
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
            message_callback=messages.append,
        )
    )

    assert messages == [
        "Completed 1/2 timesteps; VorLap diagnostics: max|F|=10 N, "
        "max|Vstruct|=1.25 m/s, max|Vrel|=2.5 m/s, force scale=100, "
        "t=0.25 s, qblade step=0, az=15 deg, qblade dt=0.01 s, cache=1/2, "
        "source rows=1, source samples=1, source clips=1, velocity warnings=1 channel/1 axis",
        "Completed 2/2 timesteps; VorLap diagnostics: max|F|=20 N, "
        "max|Vstruct|=2.5 m/s, max|Vrel|=2 m/s, force scale=100, "
        "t=0.5 s, qblade step=1, az=30 deg, qblade dt=0.01 s, cache=2/2, "
        "source rows=1, source samples=2, source clips=2, velocity warnings=1 channel/1 axis",
    ]
    assert result.summary["sil_vorlap_max_abs_force"] == 20.0
    assert result.summary["sil_vorlap_message"] == "fake SIL step 2"
    assert result.summary["sil_vorlap_geometry_audit_file"].endswith(
        "Control/vorlap_qblade_geometry_audit.csv"
    )
    assert result.summary["sil_vorlap_cache_stats"] == {"hits": 2, "misses": 2}
    assert result.summary["sil_vorlap_response_diagnostics_summary"] == {
        "model_type": "modal_aerodynamic_power",
        "samples": 2,
        "target_power_limited_steps": 0,
    }
    assert result.summary["sil_vorlap_source_bounds_policy"] == "raise"
    history = result.summary["sil_vorlap_history"]
    assert result.summary["sil_vorlap_history_mode"] == "compact"
    assert history["mode"] == "compact"
    assert history["schema"] == "vorlap-sil-force-history-compact-v1"
    assert history["columns"] == 2
    assert Path(history["path"]) == tmp_path / "out" / "case_results_sil_vorlap_history.csv"
    assert coupler_lifecycle == {
        "history_path": tmp_path / "out" / "case_results_sil_vorlap_history.csv",
        "resume_history": False,
        "closed_durable": True,
    }
    assert history["rows"] == 1
    assert history["file_identity"]["path"] == history["path"]
    assert history["file_identity"]["size_bytes"] > 0
    assert len(history["file_identity"]["sha256"]) == 64


def test_hvawt_response_diagnostics_do_not_replace_legacy_cache_stats():
    class FakeHVAWTAdapter:
        @staticmethod
        def diagnostics_summary():
            return {
                "model_type": "hvawt_two_mode_negative_damping",
                "samples": 1,
            }

    coupler = SimpleNamespace(
        hvawt_response_adapter=FakeHVAWTAdapter(),
        runtime=SimpleNamespace(
            controller=SimpleNamespace(cache_stats={"hits": 3, "misses": 4})
        ),
    )

    assert qblade_run_module._sil_vorlap_cache_stats(coupler) == {
        "hits": 3,
        "misses": 4,
    }
    assert qblade_run_module._sil_vorlap_response_diagnostics_summary(coupler) == {
        "model_type": "hvawt_two_mode_negative_damping",
        "samples": 1,
    }


def test_sil_allow_partial_failure_refuses_causally_invalid_checkpoint(tmp_path, monkeypatch):
    class FailSecondAdvanceQBlade(FakeQBlade):
        def advanceTurbineSimulation(self) -> bool:
            self.advance_calls += 1
            return self.advance_calls < 2

    class FakeSilVorLapCoupler:
        def __init__(self, sim_path, **_kwargs):
            self.parameter_file = Path(sim_path)
            self.config = {}
            self.runtime = SimpleNamespace()
            self.updates = 0
            self.closed_durable = False

        def validate_channels(self, _qblade) -> None:
            return None

        def step(self, _qblade, _step_index: int) -> str:
            self.updates += 1
            return "advanced fake SIL state"

        def checkpoint_state(self) -> dict[str, object]:
            return {"updates": self.updates}

        def close_history(self, *, durable: bool = False) -> None:
            self.closed_durable = bool(durable)

    checkpoint_steps: list[int] = []

    def record_checkpoint(**kwargs):
        completed_steps = int(kwargs["completed_steps"])
        checkpoint_steps.append(completed_steps)
        return {
            "completed_steps": completed_steps,
            "_metadata_path": str(tmp_path / f"checkpoint_{completed_steps}.json"),
            "qblade_project": str(tmp_path / f"checkpoint_{completed_steps}.qpr"),
        }

    monkeypatch.setattr(qblade_run_module, "SilVorLapCoupler", FakeSilVorLapCoupler)
    monkeypatch.setattr(qblade_run_module, "write_restart_checkpoint", record_checkpoint)

    sim = tmp_path / "case.sim"
    sim.write_text("3 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FailSecondAdvanceQBlade()

    with pytest.raises(RuntimeError, match="would not be causally consistent"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=tmp_path / "out",
                coupling_mode="sil-vorlap",
                sil_vorlap=True,
                allow_partial=True,
                checkpoint_every=1,
                progress_every=0,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )

    assert fake.advance_calls == 2
    assert fake.exported is None
    assert fake.unloaded is True
    assert checkpoint_steps == [1]


def test_run_qblade_case_stop_request_exports_completed_partial_result(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("5 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()
    messages: list[str] = []
    stop_checks = {"count": 0}

    def stop_requested() -> bool:
        stop_checks["count"] += 1
        return stop_checks["count"] > 2

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            progress_every=0,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
            message_callback=messages.append,
            stop_requested=stop_requested,
        )
    )

    assert fake.advance_calls == 2
    assert fake.unloaded is True
    assert fake.exported == result.result_path
    assert result.result_path.is_file()
    assert result.summary["requested_steps"] == 5
    assert result.summary["completed_steps"] == 2
    assert result.summary["stopped"] is True
    assert result.summary["stop_message"] == "QBlade simulation stopped by user at 2/5 completed timesteps."
    assert result.summary["failed"] is False
    assert result.summary["failure_message"] is None
    assert result.summary["exported_results"] is True
    assert messages == ["QBlade simulation stopped by user at 2/5 completed timesteps."]


def test_run_qblade_case_stop_before_first_step_skips_empty_export(tmp_path):
    sim = tmp_path / "case.sim"
    sim.write_text("5 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()
    messages: list[str] = []

    result = run_qblade_case(
        QBladeRunRequest(
            sim_path=sim,
            qblade_root=qblade_root,
            output_dir=tmp_path / "out",
            progress_every=0,
            manage_loader_env=False,
            qblade_api_loader=lambda _root: (fake, qblade_library),
            message_callback=messages.append,
            stop_requested=lambda: True,
        )
    )

    assert fake.advance_calls == 0
    assert fake.unloaded is True
    assert fake.exported is None
    assert result.result_path.is_file() is False
    assert result.summary["completed_steps"] == 0
    assert result.summary["stopped"] is True
    assert result.summary["failed"] is False
    assert result.summary["exported_results"] is False
    assert result.summary["result_file_identity"] is None
    assert result.summary["qblade_export_time_diagnostics"]["available"] is False
    assert messages == [
        "QBlade simulation stopped by user at 0/5 completed timesteps.",
        "Stop requested before any timestep completed; no QBlade results were exported.",
    ]


@pytest.mark.parametrize(
    ("coupling_mode", "force_specs"),
    [
        ("none", ()),
        ("sil-constant-force", ("BLD_1_0.000000:X:1.0",)),
    ],
)
def test_non_native_run_fails_if_native_callback_was_also_active(
    tmp_path,
    monkeypatch,
    coupling_mode,
    force_specs,
):
    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS - steps\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade library placeholder\n", encoding="utf-8")
    fake = FakeQBlade()
    monkeypatch.setattr(
        qblade_run_module,
        "native_callback_evidence",
        lambda *_args, **_kwargs: {"update_init_count": 1, "update_count": 1},
    )

    with pytest.raises(RuntimeError, match="Unexpected native callback activity"):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=qblade_root,
                output_dir=tmp_path / "out",
                coupling_mode=coupling_mode,
                sil_constant_force_specs=force_specs,
                manage_loader_env=False,
                qblade_api_loader=lambda _root: (fake, qblade_library),
            )
        )

    summary_path = tmp_path / "out" / "case_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["failed"] is True
    assert summary["native_callback_evidence"]["update_count"] == 1


@pytest.mark.parametrize(
    ("coupling_mode", "sil_vorlap", "force_specs", "message"),
    [
        ("qblade-external", True, (), "requires --coupling-mode sil-vorlap"),
        ("sil-vorlap", False, (), "requires --sil-vorlap"),
        ("sil-constant-force", False, (), "requires at least one --sil-constant-force"),
        ("none", False, ("BLD_1_0.000000:X:1.0",), "requires --coupling-mode sil-constant-force"),
    ],
)
def test_run_rejects_mismatched_coupling_label_and_active_mechanism(
    tmp_path,
    coupling_mode,
    sil_vorlap,
    force_specs,
    message,
):
    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS - steps\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        run_qblade_case(
            QBladeRunRequest(
                sim_path=sim,
                qblade_root=tmp_path / "QBladeCE",
                output_dir=tmp_path / "out",
                coupling_mode=coupling_mode,
                sil_vorlap=sil_vorlap,
                sil_constant_force_specs=force_specs,
                manage_loader_env=False,
            )
        )
