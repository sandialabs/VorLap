import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import netcdf_file

from examples.ehrmann2014_naca0018 import nalu_prescribed_motion as nalu


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _name_array(names: list[str], width: int) -> np.ndarray:
    value = np.full((len(names), width), b"\x00", dtype="S1")
    for index, name in enumerate(names):
        encoded = np.frombuffer(name.encode("ascii"), dtype="S1")
        value[index, : len(encoded)] = encoded
    return value


def _write_synthetic_frozen_mesh(path: Path) -> None:
    """Write the minimal HEX8 Exodus metadata needed by the geometry audit."""

    block_names = ["fluid-hex"]
    side_names = ["wing", "wing-pp", "inlet", "outlet", "front", "back"]
    coordinates = np.array(
        [
            [-0.09, 0.0, 0.0],
            [0.09, 0.0, 0.0],
            [0.09, -1.0, 0.0],
            [-0.09, -1.0, 0.0],
            [-0.09, 0.0, 4.0],
            [0.09, 0.0, 4.0],
            [0.09, -1.0, 4.0],
            [-0.09, -1.0, 4.0],
        ],
        dtype=np.float64,
    )
    with netcdf_file(path, "w") as database:
        database.createDimension("len_name", 33)
        database.createDimension("num_el_blk", len(block_names))
        database.createDimension("num_side_sets", len(side_names))
        database.createDimension("num_nodes", len(coordinates))
        database.createDimension("num_el_in_blk1", 1)
        database.createDimension("num_nod_per_el1", 8)
        database.createVariable("eb_names", "c", ("num_el_blk", "len_name"))[:] = (
            _name_array(block_names, 33)
        )
        database.createVariable("ss_names", "c", ("num_side_sets", "len_name"))[:] = (
            _name_array(side_names, 33)
        )
        for axis, name in enumerate(("coordx", "coordy", "coordz")):
            database.createVariable(name, "d", ("num_nodes",))[:] = coordinates[:, axis]
        database.createVariable("connect1", "i", ("num_el_in_blk1", "num_nod_per_el1"))[
            :
        ] = np.arange(1, 9, dtype=np.int32).reshape(1, 8)
        for index, side_name in enumerate(side_names, start=1):
            dimension = f"num_side_ss{index}"
            database.createDimension(dimension, 1)
            database.createVariable(f"elem_ss{index}", "i", (dimension,))[:] = [1]
            side = 1 if side_name == "wing" else 3 if side_name == "wing-pp" else index
            database.createVariable(f"side_ss{index}", "i", (dimension,))[:] = [side]


@pytest.fixture
def generated_campaign(tmp_path: Path) -> tuple[Path, Path, dict]:
    mesh = tmp_path / "NACA0018-90.exo"
    _write_synthetic_frozen_mesh(mesh)
    campaign = tmp_path / "campaign"
    manifest = nalu.generate_campaign(campaign, mesh, _sha256(mesh))
    return campaign, mesh, manifest


def test_generates_frozen_precursors_motion_matrix_and_native_ale_decks(
    generated_campaign,
) -> None:
    campaign, mesh, manifest = generated_campaign
    assert nalu.validate_campaign(campaign)["valid"] is True
    assert len(manifest["precursors"]) == 3
    assert len(manifest["cases"]) == 48
    assert len(manifest["preflight"]["scaling_cases"]) == 3
    assert {row["nodes"] for row in manifest["preflight"]["scaling_cases"]} == {2, 4, 8}
    assert {
        row["reynolds_number"] for row in manifest["preflight"]["scaling_cases"]
    } == {64_000.0}
    assert manifest["solver"]["turbulence_model"] == "sst_iddes"
    assert manifest["solver"]["transition_model"] == "none"
    assert manifest["mesh"]["sha256"] == _sha256(mesh)

    for precursor in manifest["precursors"]:
        assert precursor["precursor_steps"] == 3016
        deck = yaml.safe_load(
            (campaign / precursor["yaml"]).read_text(encoding="utf-8")
        )
        realm = deck["realms"][0]
        assert realm["mesh_transformation"][0]["motion"][0]["factor"] == [
            0.23,
            0.23,
            1.0,
        ]
        motion = realm["mesh_motion"][0]["motion"][0]
        assert motion["type"] == "oscillation"
        assert motion["amplitude"] == 0.0
        assert motion["direction"] == [0.0, 1.0, 0.0]

    expected_phases = {0.45: 9.0, 0.65: 13.0, 0.8: 16.0, 1.05: 21.0}
    for case in manifest["cases"]:
        samples = case["samples_per_cycle"]
        assert samples >= 100
        assert case["updates_after_restart"] == 13 * samples + 1
        assert math.isclose(
            case["relative_history_end_s"] - case["relative_history_start_s"],
            13.0 * case["period_s"],
            rel_tol=2.0e-13,
        )
        assert math.isclose(
            case["phase_cycles_at_restart"],
            expected_phases[case["reduced_frequency_k"]],
            abs_tol=2.0e-13,
        )
        deck = yaml.safe_load((campaign / case["yaml"]).read_text(encoding="utf-8"))
        realm = deck["realms"][0]
        assert "mesh_transformation" not in realm
        assert realm["mesh_motion"][0]["mesh_parts"] == ["all_blocks"]
        assert realm["post_processing"][0]["target_name"] == ["wing", "wing-pp"]
        assert realm["restart"]["restart_time"] == case["precursor_restart_time_s"]
        integrator = deck["Time_Integrators"][0]["StandardTimeIntegrator"]
        assert integrator["termination_step_count"] == (
            case["precursor_restart_step"] + case["updates_after_restart"]
        )


def test_mesh_audit_checks_parts_span_and_minus_90_chord_orientation(
    generated_campaign,
) -> None:
    campaign, _, _ = generated_campaign
    report = nalu.audit_mesh(campaign)
    assert report["status"] == "pass"
    assert report["chord_axis"] == "global_Y"
    assert report["incidence_orientation_deg"] == -90.0
    assert report["combined_force_surface"] == ["wing", "wing-pp"]
    assert report["source_surface_extents_xyz_m"] == pytest.approx([0.18, 1.0, 4.0])
    status = nalu.campaign_status(campaign)
    assert status["mesh_audit"]["status"] == "pass"


def _write_scaling_log(path: Path, seconds_per_step: float) -> None:
    rows = [
        "Nalu-Wind Version: v2.2.1",
        "Nalu-Wind GIT Commit SHA: ffa9de729df2a11b5241fdeb7628e7fab9f48f9b",
    ]
    for step in range(1, nalu.SCALING_PROBE_UPDATES + 1):
        rows.append(f"Time Step Count: {step} Current Time: {step * 0.001:.17g}")
        rows.append(
            f"WallClockTime: {step} Pre: 0 NLI: {seconds_per_step:.9g} "
            f"Post: 0 Total: {seconds_per_step:.9g}"
        )
    rows.append(
        "Simulation Shall Complete: time/timestep: "
        f"{nalu.SCALING_PROBE_UPDATES * 0.001:.17g}/"
        f"{nalu.SCALING_PROBE_UPDATES}"
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fake_nalu_executable(root: Path, name: str = "naluX") -> Path:
    path = root / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _record_all_scaling(
    campaign: Path,
    manifest: dict,
    executable: Path,
    seconds: dict[int, float],
) -> dict:
    for nodes in (2, 4, 8):
        probe = next(
            row
            for row in manifest["preflight"]["scaling_cases"]
            if row["nodes"] == nodes
        )
        log = campaign / probe["log"]
        _write_scaling_log(log, seconds[nodes])
        nalu.record_scaling(
            campaign,
            nodes=nodes,
            reynolds=64_000.0,
            log=log,
            elapsed_seconds=nalu.SCALING_PROBE_UPDATES * seconds[nodes],
            nalu_executable=executable,
        )
    return nalu.campaign_status(campaign, nalu_executable=executable)


def test_scaling_requires_all_three_representative_probes_and_selects_smallest_pass(
    generated_campaign,
) -> None:
    campaign, _, manifest = generated_campaign
    executable = _fake_nalu_executable(campaign.parent)
    seconds = {2: 10.0, 4: 4.0, 8: 1.0}
    for nodes in (2, 4):
        probe = next(
            row
            for row in manifest["preflight"]["scaling_cases"]
            if row["nodes"] == nodes
        )
        log = campaign / probe["log"]
        _write_scaling_log(log, seconds[nodes])
        nalu.record_scaling(
            campaign,
            nodes=nodes,
            reynolds=64_000.0,
            log=log,
            elapsed_seconds=nalu.SCALING_PROBE_UPDATES * seconds[nodes],
            nalu_executable=executable,
        )
    incomplete = nalu.campaign_status(campaign)
    assert incomplete["scaling_gate"]["status"] == "awaiting_passing_sweep"
    assert incomplete["scaling_gate"]["complete_probe_count"] == 2

    probe = next(
        row for row in manifest["preflight"]["scaling_cases"] if row["nodes"] == 8
    )
    log = campaign / probe["log"]
    _write_scaling_log(log, seconds[8])
    nalu.record_scaling(
        campaign,
        nodes=8,
        reynolds=64_000.0,
        log=log,
        elapsed_seconds=nalu.SCALING_PROBE_UPDATES * seconds[8],
        nalu_executable=executable,
    )
    complete = nalu.campaign_status(campaign)
    assert complete["scaling_gate"]["status"] == "pass"
    assert complete["scaling_gate"]["recommended_nodes"] == 8
    assert complete["scaling_gate"]["selection_tier"] == "target_2h"
    assert complete["scaling_gate"]["recommended_walltime"] == "02:00:00"
    assert complete["scaling_gate"]["by_nodes"]["2"]["pass"] is False
    assert complete["scaling_gate"]["by_nodes"]["4"]["pass"] is True
    assert complete["scaling_gate"]["by_nodes"]["4"]["target_walltime_pass"] is False
    assert complete["scaling_gate"]["by_nodes"]["8"]["target_walltime_pass"] is True
    assert complete["scaling_gate"]["runtime_identity"]["nalu_executable"] == {
        "path": str(executable.resolve()),
        "sha256": _sha256(executable),
    }


def test_scaling_falls_back_to_smallest_eight_hour_size_and_rejects_binary_change(
    generated_campaign,
) -> None:
    campaign, _, manifest = generated_campaign
    executable = _fake_nalu_executable(campaign.parent)
    complete = _record_all_scaling(
        campaign, manifest, executable, {2: 10.0, 4: 4.0, 8: 2.0}
    )
    assert complete["scaling_gate"]["recommended_nodes"] == 4
    assert complete["scaling_gate"]["selection_tier"] == "fallback_8h"
    assert complete["scaling_gate"]["recommended_walltime"] == "08:00:00"

    changed = _fake_nalu_executable(campaign.parent, "naluX_changed")
    changed.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    changed.chmod(0o755)
    with pytest.raises(nalu.CampaignError, match="differs from the scaling identity"):
        nalu.campaign_status(campaign, nalu_executable=changed)


def test_scaling_rejects_last_step_without_normal_completion_banner(
    generated_campaign,
) -> None:
    campaign, _, manifest = generated_campaign
    executable = _fake_nalu_executable(campaign.parent)
    probe = next(
        row for row in manifest["preflight"]["scaling_cases"] if row["nodes"] == 2
    )
    log = campaign / probe["log"]
    _write_scaling_log(log, 1.0)
    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(nalu.CampaignError, match="normal-completion terminal banner"):
        nalu.record_scaling(
            campaign,
            nodes=2,
            reynolds=64_000.0,
            log=log,
            elapsed_seconds=nalu.SCALING_PROBE_UPDATES,
            nalu_executable=executable,
        )


def test_case_path_skip_check_is_read_only(generated_campaign, monkeypatch) -> None:
    campaign, _, manifest = generated_campaign
    observed_writes = []

    def fake_status(_campaign, *, write=True, nalu_executable=None):
        del nalu_executable
        observed_writes.append(write)
        return {
            "precursors": [{"status": "pending"} for _ in manifest["precursors"]],
            "cases": [{"solver_status": "pending"} for _ in manifest["cases"]],
        }

    monkeypatch.setattr(nalu, "campaign_status", fake_status)
    path = nalu._case_path(
        campaign,
        stage="precursor",
        index=0,
        nodes=None,
        reynolds=None,
        skip_complete=True,
    )
    assert path == campaign / manifest["precursors"][0]["yaml"]
    assert observed_writes == [False]


def test_completed_force_history_is_rebased_and_uses_analytic_motion(
    generated_campaign,
) -> None:
    campaign, _, manifest = generated_campaign
    executable = _fake_nalu_executable(campaign.parent)
    _record_all_scaling(campaign, manifest, executable, {2: 1.0, 4: 0.8, 8: 0.6})
    case = manifest["cases"][-1]
    force_path = campaign / case["force_output"]
    force_path.parent.mkdir(parents=True, exist_ok=True)
    with force_path.open("w", encoding="utf-8") as stream:
        stream.write("Time Fpx Fpy Fpz Fvx Fvy Fvz\n")
        for index in range(1, case["updates_after_restart"] + 1):
            absolute_time = (
                case["precursor_restart_time_s"] + index * case["time_step_s"]
            )
            stream.write(
                f"{absolute_time:.6g} 0 {2.0 * index:.17g} 0 0 {-0.5 * index:.17g} 0\n"
            )
    log_path = campaign / case["log"]
    log_path.write_text(
        "Nalu-Wind Version: v2.2.1\n"
        "Nalu-Wind GIT Commit SHA: "
        "ffa9de729df2a11b5241fdeb7628e7fab9f48f9b\n"
        f"Simulation Shall Complete: time/timestep: 1/"
        f"{case['precursor_restart_step'] + case['updates_after_restart']}\n",
        encoding="utf-8",
    )
    history_path = campaign / case["canonical_history"]["path"]
    assert not history_path.exists()
    dry_status = nalu.campaign_status(campaign, write=False, nalu_executable=executable)
    assert dry_status["cases"][-1]["solver_status"] == "complete"
    assert not history_path.exists()
    status = nalu.campaign_status(campaign, nalu_executable=executable)
    result = status["cases"][-1]
    history = result["canonical_history"]
    assert result["solver_status"] == "complete"
    assert result["runtime_identity"]["status"] == "verified"
    assert result["runtime_identity"]["nalu_executable"]["sha256"] == _sha256(
        executable
    )
    assert history["status"] == "complete"
    assert history["rows"] == case["updates_after_restart"]
    history_path = campaign / history["path"]
    assert history["sha256"] == _sha256(history_path)
    with history_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    first = {key: float(value) for key, value in rows[0].items()}
    last = {key: float(value) for key, value in rows[-1].items()}
    omega = 2.0 * math.pi * case["frequency_hz"]
    assert first["time_s"] == case["time_step_s"]
    assert last["time_s"] == case["relative_history_end_s"]
    assert first["displacement_m"] == pytest.approx(
        case["amplitude_m"] * math.sin(omega * first["time_s"]), rel=1.0e-15
    )
    assert first["velocity_mps"] == pytest.approx(
        case["amplitude_m"] * omega * math.cos(omega * first["time_s"]), rel=1.0e-15
    )
    assert first["crossflow_force_n"] == 1.5
    assert last["crossflow_force_n"] == 1.5 * case["updates_after_restart"]
    assert math.isclose(
        last["time_s"] - first["time_s"], 13.0 * case["period_s"], rel_tol=2.0e-13
    )
    log_path.write_text(
        "Nalu-Wind Version: v2.2.1\n"
        "Nalu-Wind GIT Commit SHA: "
        "ffa9de729df2a11b5241fdeb7628e7fab9f48f9b\n",
        encoding="utf-8",
    )
    incomplete = nalu.campaign_status(campaign, nalu_executable=executable)
    assert incomplete["cases"][-1]["solver_status"] == "running_or_incomplete"
    log_path.write_text(
        "Nalu-Wind Version: v2.2.1\nNalu-Wind GIT Commit SHA: wrong\n"
        f"Simulation Shall Complete: time/timestep: 1/"
        f"{case['precursor_restart_step'] + case['updates_after_restart']}\n",
        encoding="utf-8",
    )
    with pytest.raises(nalu.CampaignError, match="frozen Nalu-Wind commit"):
        nalu.campaign_status(campaign, nalu_executable=executable)


def test_mesh_identity_change_is_rejected(generated_campaign) -> None:
    campaign, mesh, _ = generated_campaign
    with mesh.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(nalu.CampaignError, match="mesh SHA-256 mismatch"):
        nalu.validate_campaign(campaign)


def test_flight_scripts_reserve_every_physical_core_and_keep_eight_hour_gate() -> None:
    root = Path(__file__).parents[1]
    for name in (
        "submit_naca0018_work_map_nalu_preflight.sj",
        "submit_naca0018_work_map_nalu.sj",
    ):
        text = (root / "run_headless" / name).read_text(encoding="utf-8")
        assert "#SBATCH --ntasks-per-node=112" in text
        assert "#SBATCH --hint=nomultithread" in text
        assert "#SBATCH --exclusive" in text
        assert "#SBATCH --time=08:00:00" in text
        assert 'expected_tasks="$((SLURM_JOB_NUM_NODES * CORES_PER_NODE))"' in text
        assert "--cpu-bind=cores" in text
        assert '--nalu-executable "${NALU_EXECUTABLE}"' in text
        assert '"${NALU_EXECUTABLE}" -i "${yaml_path}"' in text
        assert "/pscratch/ndeveld" not in text
        assert 'NALU_FLIGHT_ENV="${NALU_FLIGHT_ENV:-}"' in text
        assert 'NALU_EXECUTABLE="${NALU_EXECUTABLE:-}"' in text
        assert "command -v naluX" in text
        assert "No Nalu runtime found" in text
    production = (root / "run_headless" / "submit_naca0018_work_map_nalu.sj").read_text(
        encoding="utf-8"
    )
    preflight = (
        root / "run_headless" / "submit_naca0018_work_map_nalu_preflight.sj"
    ).read_text(encoding="utf-8")
    assert production.count("--no-write") == 2
    assert 'exec 8>"${CAMPAIGN_ROOT}.status.lock"' in production
    assert 'flock 8\n"${PY[@]}" status --campaign "${CAMPAIGN_ROOT}"' in production
    assert preflight.count("--no-write") == 2
    assert preflight.count('exec 8>"${CAMPAIGN_ROOT}.status.lock"') == 2
    assert (
        preflight.count('flock 8\n  "${PY[@]}" status --campaign "${CAMPAIGN_ROOT}"')
        == 1
    )
    assert (
        preflight.count('flock 8\n"${PY[@]}" status --campaign "${CAMPAIGN_ROOT}"') == 1
    )


def test_flight_input_audit_is_read_only_and_binds_official_release() -> None:
    root = Path(__file__).parents[1]
    text = (
        root / "run_headless" / "audit_naca0018_work_map_flight_inputs.sh"
    ).read_text(encoding="utf-8")
    assert "ffa9de729df2a11b5241fdeb7628e7fab9f48f9b" in text
    assert "-DIRTY" not in text
    assert "spack location -i nalu-wind@2.2.1" in text
    assert "NACA0018-90.exo" in text
    assert "AUDIT_STATUS=ready_for_scaling_preflight" in text
    assert "KNOWN_FLIGHT_OPENMPI_ROOT" in text
    assert "mpicc -show" in text
    assert "ompi_info --parsable" in text
    assert "srun --mpi=list" in text
    assert "sbatch" not in text
