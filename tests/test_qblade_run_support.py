from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import shutil
import subprocess
import sys
import pytest

import vorlap.qblade_run_support as run_support

from vorlap.qblade_run_support import (
    atomic_write_json,
    build_reproducibility_metadata,
    build_checkpoint_identity,
    file_sha256,
    find_bridge_library,
    find_qblade_library,
    ensure_qblade_loader_env,
    load_resume_checkpoint,
    native_callback_evidence,
    preflight_qblade_wind_input,
    prune_restart_checkpoints,
    qblade_library_dirs,
    qblade_runtime_asset_fingerprint,
    qblade_external_library_candidate_names,
    qblade_external_library_symbol_diagnostics,
    qblade_external_required_symbols,
    resolve_qblade_external_library,
    write_restart_checkpoint,
)


def test_loader_reexec_preserves_module_invocation(tmp_path, monkeypatch):
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    observed: dict[str, object] = {}

    def fake_execvpe(executable, argv, env):
        observed.update(executable=executable, argv=argv, env=env)
        raise RuntimeError("reexec intercepted")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        sys,
        "argv",
        ["/work/VorLap/vorlap/hvawt_modal_decay_campaign.py", "run-case"],
    )
    monkeypatch.setattr(
        sys,
        "orig_argv",
        [
            sys.executable,
            "-m",
            "vorlap.hvawt_modal_decay_campaign",
            "run-case",
        ],
    )
    monkeypatch.delenv("VORLAP_QBLADE_LOADER_ENV", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(os, "execvpe", fake_execvpe)

    with pytest.raises(RuntimeError, match="reexec intercepted"):
        ensure_qblade_loader_env(qblade_root)

    assert observed["executable"] == sys.executable
    assert observed["argv"] == [
        sys.executable,
        "-m",
        "vorlap.hvawt_modal_decay_campaign",
        "run-case",
    ]
    assert observed["env"]["VORLAP_QBLADE_LOADER_ENV"] == "1"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class FakeQBlade:
    def storeProject(self, path: bytes) -> None:
        Path(path.decode("utf-8")).write_text("project\n", encoding="utf-8")


def test_atomic_write_json_replaces_existing_file_and_removes_temp(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    atomic_write_json(path, {"new": True, "value": 3})

    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True, "value": 3}
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_atomic_write_json_supports_concurrent_same_destination_writers(tmp_path):
    path = tmp_path / "flight_qblade_runtime_identity.json"
    payload = {"schema": "runtime-v1", "values": list(range(20_000))}

    with ThreadPoolExecutor(max_workers=16) as executor:
        completed = list(
            executor.map(
                lambda _index: atomic_write_json(path, payload), range(64)
            )
        )

    assert completed == [None] * 64
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_file_sha256_cache_detects_same_size_rewrite_with_restored_mtime(tmp_path):
    path = tmp_path / "runtime.h5"
    path.write_bytes(b"first-runtime")
    original_stat = path.stat()
    first = file_sha256(path)
    path.write_bytes(b"other-runtime")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    second = file_sha256(path)

    assert first is not None and second is not None
    assert first["size_bytes"] == second["size_bytes"]
    assert first["sha256"] != second["sha256"]


def test_file_sha256_retries_one_changed_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "qblade-result.txt"
    path.write_bytes(b"first snapshot")
    final = b"settled qblade export"
    real_sha256 = hashlib.sha256
    changed = False

    class MutatingDigest:
        def __init__(self):
            self.inner = real_sha256()

        def update(self, block):
            nonlocal changed
            self.inner.update(block)
            if not changed:
                changed = True
                path.write_bytes(final)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(run_support.hashlib, "sha256", MutatingDigest)
    monkeypatch.setattr(run_support.time, "sleep", lambda _seconds: None)
    identity = file_sha256(path)

    assert changed
    assert identity is not None
    assert identity["sha256"] == real_sha256(final).hexdigest()
    assert identity["size_bytes"] == len(final)


def test_runtime_asset_manifest_hashes_complete_generated_case_and_external_wind(
    tmp_path,
):
    case = tmp_path / "case"
    case.mkdir()
    wind = tmp_path / "wind.bts"
    wind.write_bytes(b"wind")
    paths = {
        "case.sim": b"1 WNDTYPE\n../wind.bts WNDNAME\n",
        "model.trb": b"turbine",
        "Structure/model.str": b"structure",
        "Aero/blade.bld": b"blade",
        "Aero/Polars/polar.plr": b"polar",
        "Aero/Airfoils/foil.afl": b"airfoil",
        "VorLapAirfoils/NACA0018.h5": b"source",
        "Control/vorlap_qblade_external.json": b"{}\n",
    }
    for relative, content in paths.items():
        target = case / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    assets, fingerprint = qblade_runtime_asset_fingerprint(case / "case.sim")

    case_assets = {
        str(row["relative_path"]): row for row in assets if row["role"] == "case_asset"
    }
    assert set(case_assets) == set(paths)
    assert all(row["sha256"] and row["size_bytes"] > 0 for row in case_assets.values())
    wind_assets = [row for row in assets if row["role"] == "wind_input"]
    assert len(wind_assets) == 1
    assert wind_assets[0]["path"] == str(wind.resolve())
    assert wind_assets[0]["sha256"]
    assert len(fingerprint) == 64


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

    deleted = prune_restart_checkpoints(checkpoint_dir, keep=2)

    assert any(stems[0] in path for path in deleted)
    assert not (checkpoint_dir / f"{stems[0]}.json").exists()
    assert not (checkpoint_dir / f"{stems[0]}.qpr").exists()
    for stem in stems[1:]:
        assert (checkpoint_dir / f"{stem}.json").exists()
        assert (checkpoint_dir / f"{stem}.qpr").exists()


def test_write_and_load_restart_checkpoint_round_trip(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS\n", encoding="utf-8")

    latest = write_restart_checkpoint(
        qblade=FakeQBlade(),
        checkpoint_dir=checkpoint_dir,
        sim_path=sim,
        qblade_root=tmp_path / "QBladeCE",
        output_dir=tmp_path / "out",
        result_name="case_results",
        coupling_mode="sil-vorlap",
        requested_steps=10,
        completed_steps=4,
        threads=2,
        use_opencl=False,
        checkpoint_identity={"fingerprint": "identity-a", "runtime_assets": []},
        resume_checkpoint=None,
        sil_vorlap=None,
        reason="test",
    )
    args = SimpleNamespace(
        resume_checkpoint=None,
        resume_from_checkpoint=True,
    )

    loaded = load_resume_checkpoint(
        args,
        checkpoint_dir,
        expected_identity={"fingerprint": "identity-a"},
    )

    assert latest["completed_steps"] == 4
    assert Path(str(latest["qblade_project"])).is_file()
    assert loaded is not None
    assert loaded["_metadata_path"] == str(checkpoint_dir / "latest_checkpoint.json")
    assert loaded["qblade_project"] == latest["qblade_project"]
    assert loaded["checkpoint_identity_fingerprint"] == "identity-a"
    assert loaded["version"] == 3
    assert set(latest) == {
        "version",
        "created_unix",
        "reason",
        "sim",
        "qblade_root",
        "output_dir",
        "result_name",
        "coupling_mode",
        "requested_steps",
        "completed_steps",
        "qblade_project",
        "qblade_project_sha256",
        "qblade_project_size_bytes",
        "threads",
        "use_opencl",
        "checkpoint_identity",
        "checkpoint_identity_fingerprint",
        "qblade_resume_mode",
        "qblade_resume_note",
        "previous_checkpoint",
        "sil_vorlap_state",
        "_metadata_path",
    }
    assert not any("modal" in key for key in latest)
    loaded_identity = file_sha256(Path(str(loaded["qblade_project"])))
    assert loaded_identity is not None
    assert loaded["qblade_project_sha256"] == loaded_identity["sha256"]
    assert loaded["qblade_project_size_bytes"] == loaded_identity["size_bytes"]


def test_automatic_resume_quarantines_incompatible_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    project = checkpoint_dir / "case_step_0000004.qpr"
    project.write_text("project\n", encoding="utf-8")
    stale_project = checkpoint_dir / "case_step_0084000.qpr"
    stale_project.write_text("stale project\n", encoding="utf-8")
    write_json(
        checkpoint_dir / "case_step_0084000.json",
        {
            "completed_steps": 84000,
            "qblade_project": str(stale_project),
            "checkpoint_identity_fingerprint": "old-identity",
        },
    )
    compatible_project = checkpoint_dir / "case_step_0000002.qpr"
    compatible_project.write_text("compatible project\n", encoding="utf-8")
    compatible_identity = file_sha256(compatible_project)
    assert compatible_identity is not None
    write_json(
        checkpoint_dir / "case_step_0000002.json",
        {
            "completed_steps": 2,
            "qblade_project": str(compatible_project),
            "checkpoint_identity_fingerprint": "new-identity",
            "qblade_project_sha256": compatible_identity["sha256"],
            "qblade_project_size_bytes": compatible_identity["size_bytes"],
        },
    )
    write_json(
        checkpoint_dir / "latest_checkpoint.json",
        {
            "completed_steps": 4,
            "qblade_project": str(project),
            "checkpoint_identity_fingerprint": "old-identity",
        },
    )
    args = SimpleNamespace(resume_checkpoint=None, resume_from_checkpoint=True)

    loaded = load_resume_checkpoint(
        args,
        checkpoint_dir,
        expected_identity={"fingerprint": "new-identity"},
    )

    assert loaded is None
    assert not (checkpoint_dir / "latest_checkpoint.json").exists()
    assert list(checkpoint_dir.glob("latest_checkpoint.incompatible-*.json"))
    assert args._checkpoint_rejection["observed_fingerprint"] == "old-identity"
    quarantine_dir = Path(args._checkpoint_rejection["quarantined_checkpoint_dir"])
    assert (quarantine_dir / project.name).is_file()
    assert (quarantine_dir / stale_project.name).is_file()
    assert (quarantine_dir / "case_step_0084000.json").is_file()
    assert compatible_project.is_file()
    assert (checkpoint_dir / "case_step_0000002.json").is_file()


def test_explicit_resume_rejects_incompatible_checkpoint_without_moving_it(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    project = checkpoint_dir / "case.qpr"
    project.write_text("project\n", encoding="utf-8")
    metadata = checkpoint_dir / "explicit.json"
    write_json(
        metadata,
        {
            "completed_steps": 4,
            "qblade_project": str(project),
            "checkpoint_identity_fingerprint": "old-identity",
        },
    )
    args = SimpleNamespace(resume_checkpoint=metadata, resume_from_checkpoint=False)

    with pytest.raises(ValueError, match="incompatible"):
        load_resume_checkpoint(
            args,
            checkpoint_dir,
            expected_identity={"fingerprint": "new-identity"},
        )

    assert metadata.is_file()


@pytest.mark.parametrize("failure_mode", ["missing", "empty", "modified"])
def test_automatic_resume_quarantines_stale_matching_qblade_project(
    tmp_path, failure_mode
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    project = checkpoint_dir / "case_step_0000004.qpr"
    project.write_text("complete project\n", encoding="utf-8")
    project_identity = file_sha256(project)
    assert project_identity is not None
    metadata = {
        "version": 3,
        "completed_steps": 4,
        "qblade_project": str(project),
        "qblade_project_sha256": project_identity["sha256"],
        "qblade_project_size_bytes": project_identity["size_bytes"],
        "checkpoint_identity_fingerprint": "matching-identity",
    }
    write_json(checkpoint_dir / "case_step_0000004.json", metadata)
    write_json(checkpoint_dir / "latest_checkpoint.json", metadata)
    if failure_mode == "missing":
        project.unlink()
    elif failure_mode == "empty":
        project.write_bytes(b"")
    else:
        project.write_text("modified project\n", encoding="utf-8")
    args = SimpleNamespace(resume_checkpoint=None, resume_from_checkpoint=True)

    loaded = load_resume_checkpoint(
        args,
        checkpoint_dir,
        expected_identity={"fingerprint": "matching-identity"},
    )

    assert loaded is None
    assert not (checkpoint_dir / "latest_checkpoint.json").exists()
    assert list(checkpoint_dir.glob("latest_checkpoint.incompatible-*.json"))
    quarantine_dir = Path(args._checkpoint_rejection["quarantined_checkpoint_dir"])
    assert (quarantine_dir / "case_step_0000004.json").is_file()
    if failure_mode != "missing":
        assert (quarantine_dir / project.name).is_file()


def test_explicit_resume_rejects_modified_matching_project_without_moving_it(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    project = checkpoint_dir / "case.qpr"
    project.write_text("complete project\n", encoding="utf-8")
    project_identity = file_sha256(project)
    assert project_identity is not None
    metadata = checkpoint_dir / "explicit.json"
    write_json(
        metadata,
        {
            "version": 3,
            "completed_steps": 4,
            "qblade_project": str(project),
            "qblade_project_sha256": project_identity["sha256"],
            "qblade_project_size_bytes": project_identity["size_bytes"],
            "checkpoint_identity_fingerprint": "matching-identity",
        },
    )
    project.write_text("modified project\n", encoding="utf-8")
    args = SimpleNamespace(resume_checkpoint=metadata, resume_from_checkpoint=False)

    with pytest.raises(ValueError, match="stale or incomplete"):
        load_resume_checkpoint(
            args,
            checkpoint_dir,
            expected_identity={"fingerprint": "matching-identity"},
        )

    assert metadata.is_file()
    assert project.is_file()


def test_checkpoint_identity_changes_with_runtime_airfoil_or_coupling(tmp_path):
    case = tmp_path / "case"
    airfoils = case / "model" / "VorLapAirfoils"
    airfoils.mkdir(parents=True)
    sim = case / "case.sim"
    sim.write_text("0 WNDTYPE\n3 NUMTIMESTEPS\n", encoding="utf-8")
    h5 = airfoils / "NACA0018.h5"
    h5.write_bytes(b"first airfoil")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    (qblade_root / "libQBladeCE_test.so").write_bytes(b"qblade")

    first = build_checkpoint_identity(
        sim_path=sim,
        qblade_root=qblade_root,
        coupling_mode="sil-vorlap",
        requested_steps=3,
        threads=2,
        use_opencl=False,
        device=0,
        group_size=32,
    )
    h5.write_bytes(b"second airfoil")
    changed_asset = build_checkpoint_identity(
        sim_path=sim,
        qblade_root=qblade_root,
        coupling_mode="sil-vorlap",
        requested_steps=3,
        threads=2,
        use_opencl=False,
        device=0,
        group_size=32,
    )
    changed_coupling = build_checkpoint_identity(
        sim_path=sim,
        qblade_root=qblade_root,
        coupling_mode="none",
        requested_steps=3,
        threads=2,
        use_opencl=False,
        device=0,
        group_size=32,
    )
    changed_resume_behavior = build_checkpoint_identity(
        sim_path=sim,
        qblade_root=qblade_root,
        coupling_mode="none",
        requested_steps=3,
        threads=2,
        use_opencl=False,
        device=0,
        group_size=32,
        resume_initialize=True,
    )
    explicit_no_modal = build_checkpoint_identity(
        sim_path=sim,
        qblade_root=qblade_root,
        coupling_mode="none",
        requested_steps=3,
        threads=2,
        use_opencl=False,
        device=0,
        group_size=32,
        modal_burst_identity=None,
    )

    assert first["fingerprint"] != changed_asset["fingerprint"]
    assert changed_asset["fingerprint"] != changed_coupling["fingerprint"]
    assert changed_coupling["fingerprint"] != changed_resume_behavior["fingerprint"]
    assert explicit_no_modal == changed_coupling
    assert "modal_burst_identity" not in explicit_no_modal


def test_explicit_false_modal_flag_preserves_legacy_reproducibility_payload(
    tmp_path,
    monkeypatch,
):
    sim = tmp_path / "case.sim"
    sim.write_text("1 NUMTIMESTEPS\n", encoding="utf-8")
    qblade_root = tmp_path / "QBladeCE"
    qblade_root.mkdir()
    qblade_library = qblade_root / "libQBladeCE_test.so"
    qblade_library.write_text("qblade\n", encoding="utf-8")
    monkeypatch.setattr("vorlap.qblade_run_support.time.time", lambda: 123.0)
    kwargs = {
        "command_line": ["runner"],
        "sim_path": sim,
        "output_dir": tmp_path / "out",
        "qblade_root": qblade_root,
        "qblade_library": qblade_library,
        "coupling_mode": "none",
        "requested_steps": 1,
        "completed_steps": 1,
        "threads": 1,
        "use_opencl": False,
    }

    omitted = build_reproducibility_metadata(**kwargs)
    explicit_false = build_reproducibility_metadata(**kwargs, modal_burst=False)

    assert explicit_false == omitted
    assert "qblade_runtime_dependencies" not in explicit_false
    assert "qblade_runtime_dependency_fingerprint" not in explicit_false


def test_native_callback_evidence_parses_cpp_bridge_log(tmp_path):
    sim = tmp_path / "case" / "demo.sim"
    trb = tmp_path / "case" / "demo.trb"
    config = tmp_path / "case" / "Control" / "vorlap_qblade_external.json"
    qblade_root = tmp_path / "QBladeCE"
    bridge_library = qblade_root / "ControllerFiles" / "libvorlap_qblade_bridge.so"
    log = config.parent / "vorlap_qblade_bridge_cpp.log"
    config.parent.mkdir(parents=True)
    bridge_library.parent.mkdir(parents=True)
    bridge_library.write_text("bridge\n", encoding="utf-8")
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
        "update_init entered; paramFile=Control/vorlap_qblade_external.json\n"
        'python_env_before_import={"python_path_entries_added": ["/tmp/example"]}\n'
        'python_env_after_import={"numpy_version": "2.0.test", "vorlap_file": "/tmp/vorlap/__init__.py"}\n'
        "MESSAGE: VorLap QBlade runtime initialized.\n"
        "update entered; count=1\n"
        "MESSAGE: VorLap dbg t=0.015s az=270deg max|F|=1N cache=1/2\n"
        "update entered; count=2\n",
        encoding="utf-8",
    )

    evidence = native_callback_evidence(sim, qblade_root=qblade_root)

    assert evidence["callback_block_present"] is True
    assert evidence["turbine_file_resolved"] == str(trb)
    assert evidence["library_stem"] == "libvorlap_qblade_bridge"
    assert evidence["library_candidate_names"] == [
        "libvorlap_qblade_bridge.so",
        "libvorlap_qblade_bridge.dylib",
        "libvorlap_qblade_bridge.dll",
    ]
    assert evidence["library_search_roots"] == [
        str(qblade_root.resolve() / "ControllerFiles"),
        str(qblade_root.resolve()),
    ]
    assert evidence["library_file_resolved"] == str(bridge_library.resolve())
    assert evidence["library_file_exists"] is True
    assert evidence["library_file_hash"]["sha256"]
    assert evidence["function_name"] == "update"
    assert evidence["array_size"] == "15"
    assert (
        evidence["parameter_file_from_turbine"] == "Control/vorlap_qblade_external.json"
    )
    assert evidence["parameter_file_resolved"] == str(config)
    assert evidence["parameter_file_exists"] is True
    assert evidence["log_file_exists"] is True
    assert evidence["update_init_seen"] is True
    assert evidence["update_init_count"] == 1
    assert evidence["update_log_entry_count"] == 2
    assert evidence["update_cache_observation_count"] == 1
    assert evidence["update_count_observed_max"] == 3
    assert evidence["runtime_initialized_message_seen"] is True
    assert evidence["python_path_entries_added"] == ["/tmp/example"]
    assert evidence["numpy_version"] == "2.0.test"
    assert evidence["vorlap_file"] == "/tmp/vorlap/__init__.py"
    assert evidence["log_file_hash"]["sha256"]


def test_qblade_wind_input_preflight_accepts_existing_windfield(tmp_path):
    wind = tmp_path / "full6_u=10.58.bts"
    wind.write_text("synthetic placeholder\n", encoding="utf-8")
    sim = tmp_path / "case.sim"
    sim.write_text(
        "1 WNDTYPE - use windfield\n" "full6_u=10.58.bts WNDNAME - wind file\n",
        encoding="utf-8",
    )

    diagnostics = preflight_qblade_wind_input(sim)

    assert diagnostics["requires_file"] is True
    assert diagnostics["resolved_wind_file"] == str(wind.resolve())
    assert diagnostics["exists"] is True


def test_qblade_library_discovery_and_loader_dirs(tmp_path, monkeypatch):
    qblade_root = tmp_path / "QBladeCE"
    libraries = qblade_root / "Libraries"
    controller_files = qblade_root / "ControllerFiles"
    extra = tmp_path / "extra_libs"
    for path in (libraries, controller_files, extra):
        path.mkdir(parents=True)
    qblade_lib = qblade_root / "libQBladeCE_test.so"
    qblade_lib.write_text("shared lib placeholder\n", encoding="utf-8")
    bridge_lib = controller_files / "libvorlap_qblade_bridge.so"
    bridge_lib.write_text("bridge lib placeholder\n", encoding="utf-8")
    monkeypatch.setenv("QBLADE_EXTRA_LIBRARY_PATHS", str(extra))

    dirs = qblade_library_dirs(qblade_root)

    assert find_qblade_library(qblade_root) == qblade_lib
    assert find_bridge_library(qblade_root) == bridge_lib.resolve()
    assert extra.resolve() in dirs
    assert qblade_root.resolve() in dirs
    assert libraries.resolve() in dirs


def test_qblade_external_library_resolution_reports_missing_configured_stem(tmp_path):
    qblade_root = tmp_path / "QBladeCE"
    (qblade_root / "ControllerFiles").mkdir(parents=True)

    resolved = resolve_qblade_external_library(qblade_root, "vorlap_qblade_bridge")

    assert qblade_external_library_candidate_names("vorlap_qblade_bridge") == [
        "vorlap_qblade_bridge.so",
        "vorlap_qblade_bridge.dylib",
        "vorlap_qblade_bridge.dll",
    ]
    assert resolved["library_file_exists"] is False
    assert resolved["library_file_resolved"] is None
    assert resolved["library_search_roots"] == [
        str(qblade_root.resolve() / "ControllerFiles"),
        str(qblade_root.resolve()),
    ]


def test_qblade_external_library_resolution_checks_required_symbols(
    tmp_path, monkeypatch
):
    qblade_root = tmp_path / "QBladeCE"
    controller_files = qblade_root / "ControllerFiles"
    controller_files.mkdir(parents=True)
    library = controller_files / "libvorlap_qblade_bridge.so"
    library.write_text("synthetic shared library placeholder\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nm = fake_bin / "nm"
    fake_nm.write_text(
        "#!/bin/sh\n" "printf '00000000 T update\\n00000010 T update_init\\n'\n",
        encoding="utf-8",
    )
    fake_nm.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    resolved = resolve_qblade_external_library(
        qblade_root, "libvorlap_qblade_bridge", "update"
    )

    assert qblade_external_required_symbols("update") == [
        "update",
        "update_init",
        "update_message",
    ]
    assert resolved["library_file_resolved"] == str(library.resolve())
    assert resolved["library_symbol_check_available"] is True
    assert resolved["library_exported_required_symbols"] == ["update", "update_init"]
    assert resolved["library_missing_required_symbols"] == ["update_message"]


def test_qblade_symbol_audit_rejects_real_elf_with_only_undefined_callbacks(tmp_path):
    clang = shutil.which("clang")
    linker = shutil.which("ld.lld")
    if clang is None or linker is None:
        pytest.skip(
            "An ELF-capable clang/lld toolchain is required for this regression."
        )
    library = tmp_path / "libundefined_callbacks.so"
    source = (
        "extern void update(void);\n"
        "extern void update_init(void);\n"
        "extern void update_message(void);\n"
        "void call_callbacks(void) { update(); update_init(); update_message(); }\n"
    )
    completed = subprocess.run(
        [
            clang,
            "--target=x86_64-unknown-linux-gnu",
            "-fuse-ld=lld",
            "-shared",
            "-nostdlib",
            "-Wl,--allow-shlib-undefined",
            "-x",
            "c",
            "-",
            "-o",
            str(library),
        ],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"Could not build ELF adversary: {completed.stderr}")

    diagnostics = qblade_external_library_symbol_diagnostics(library, "update")

    assert diagnostics["library_symbol_check_tool"] == "nm -D --defined-only"
    assert diagnostics["library_symbol_check_available"] is True
    assert diagnostics["library_exported_required_symbols"] == []
    assert diagnostics["library_missing_required_symbols"] == [
        "update",
        "update_init",
        "update_message",
    ]
