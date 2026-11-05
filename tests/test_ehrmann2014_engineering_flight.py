import hashlib
import json
from pathlib import Path

import pytest

from run_headless import run_ehrmann2014_engineering_flight as scheduler
from run_headless import finalize_ehrmann2014_engineering_local as local_finalizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def _state(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"calibrated":true}\n')
    reuse_root = tmp_path / "old_results"
    reuse_root.mkdir()
    qblade_root = tmp_path / "QBladeCE_2.0.9.7"
    qblade_root.mkdir()
    runtime = {
        "version": scheduler.campaign.FROZEN_QBLADE_VERSION,
        "library_sha256": scheduler.campaign.FROZEN_QBLADE_LIBRARY_SHA256,
        "runtime_dependency_fingerprint": scheduler.campaign.FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT,
    }
    monkeypatch.setattr(scheduler, "_qblade_runtime_identity", lambda _path: runtime)
    root = tmp_path / "campaign"
    root.mkdir()
    rows = []
    complete = []
    runnable = []
    for physical in range(26):
        new_raw = physical >= 18
        half = physical >= 24
        for hybrid in (False, True):
            index = 2 * physical + int(hybrid)
            suffix = "hybrid" if hybrid else "raw"
            case_id = f"physical_{physical:02d}_{suffix}"
            row = {
                "index": index,
                "case_id": case_id,
                "scenario": scheduler.campaign.ENGINEERING_LOCKIN_SCENARIOS[
                    int(hybrid)
                ],
                "timestep_s": (
                    scheduler.campaign.HALF_TIMESTEP_S
                    if half
                    else scheduler.campaign.TIMESTEP_S
                ),
                "num_steps": 40_000 if half else 20_000,
                "sim": f"cases/{case_id}/{case_id}.sim",
                "result_dir": f"results/{case_id}",
                "raw_reuse_status": (
                    "no_candidate_run_required"
                    if hybrid or new_raw
                    else "reused_exact_identity"
                ),
            }
            rows.append(row)
            record = {
                "index": index,
                "case_id": case_id,
                "num_steps": row["num_steps"],
                "status": "runnable",
                "reason": "no_run_summary",
            }
            if not hybrid and not new_raw:
                source = reuse_root / case_id
                source.mkdir()
                result = root / row["result_dir"]
                result.mkdir(parents=True)
                (result / "reuse_receipt.json").write_text(
                    json.dumps(
                        {
                            "source_result": str(source.resolve()),
                            "qblade_library_sha256": runtime["library_sha256"],
                            "qblade_runtime_dependency_fingerprint": runtime[
                                "runtime_dependency_fingerprint"
                            ],
                            "sil_vorlap_max_abs_force": 0.0,
                            "native_callback_update_count": 0,
                        }
                    )
                )
                record["status"] = "complete"
                record["reason"] = "exact_identity_complete_result"
                complete.append(record)
            else:
                runnable.append(record)
    runnable.sort(key=lambda row: (-row["num_steps"], row["index"]))
    manifest = {
        "schema": scheduler.campaign.ENGINEERING_LOCKIN_SCHEMA,
        "comparison_series": list(scheduler.campaign.ENGINEERING_LOCKIN_SCENARIOS),
        "fingerprint": "engineering-campaign",
        "engineering_lockin_identity": {
            "sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
            "nalu_required": False,
        },
        "qblade_runtime_identity": runtime,
        "cases": rows,
    }
    plan = {"complete": complete, "runnable": runnable, "blocked": []}
    monkeypatch.setattr(
        scheduler.campaign,
        "validate",
        lambda _path: {"root": root, "manifest": manifest},
    )
    monkeypatch.setattr(scheduler.legacy, "ordered_case_plan", lambda *_args: plan)
    return root, calibration, reuse_root, qblade_root, manifest, plan


def test_execution_plan_is_18_reused_raw_plus_34_new(tmp_path, monkeypatch):
    root, calibration, reuse, qblade, _manifest, _plan = _state(tmp_path, monkeypatch)
    state = scheduler.validate_execution_plan(
        root,
        calibration=calibration,
        raw_reuse_root=reuse,
        qblade_root=qblade,
    )
    assert state["reused_raw"] == 18
    assert state["new_raw"] == 8
    assert len(state["runnable"]) == 34


def test_execution_plan_rejects_other_calibration(tmp_path, monkeypatch):
    root, calibration, reuse, qblade, _manifest, _plan = _state(tmp_path, monkeypatch)
    calibration.write_text('{"calibrated":false}\n')
    with pytest.raises(scheduler.EngineeringSchedulerError, match="does not match"):
        scheduler.validate_execution_plan(
            root,
            calibration=calibration,
            raw_reuse_root=reuse,
            qblade_root=qblade,
        )


def test_shared_allocation_and_wrapper_use_34_one_core_slots():
    text = (
        REPO_ROOT / "run_headless/submit_ehrmann2014_engineering_flight.sj"
    ).read_text()
    assert "#SBATCH --ntasks=34" in text
    assert "#SBATCH --ntasks-per-node=34" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --mem=68G" in text
    assert "#SBATCH --exclusive" not in text
    assert "--engineering-calibration" in text
    assert "RAW_REUSE_ROOT" in text
    assert "nalu" not in text.lower()


def test_legacy_scheduler_variable_shared_allocation_preserves_default_contract():
    scheduler.legacy._allocation_contract(
        workers=34,
        threads=1,
        allocated_cpus=34,
        expected_workers=34,
        expected_allocated_cpus=34,
    )
    with pytest.raises(scheduler.legacy.SchedulerError, match="40 workers"):
        scheduler.legacy._allocation_contract(workers=34, threads=1, allocated_cpus=40)


def test_local_finalizer_requires_exact_eight_cpu_docker_cap(tmp_path):
    inspect = tmp_path / "inspect.json"
    payload = [
        {
            "Platform": "linux",
            "Image": "sha256:" + "a" * 64,
            "Config": {"Image": "vorlap-qblade-headless:amd64"},
            "HostConfig": {"NanoCpus": 8_000_000_000},
            "State": {"OOMKilled": False},
        }
    ]
    inspect.write_text(json.dumps(payload) + "\n")
    contract = local_finalizer._container_contract(inspect)
    assert contract["nano_cpus"] == 8_000_000_000
    assert contract["oom_killed"] is False
    assert contract["image_id"] == "sha256:" + "a" * 64

    payload[0]["HostConfig"]["NanoCpus"] = 9_000_000_000
    inspect.write_text(json.dumps(payload) + "\n")
    with pytest.raises(local_finalizer.LocalExecutionError, match="8 CPUs"):
        local_finalizer._container_contract(inspect)


def test_local_finalizer_publishes_only_after_all_52_cases_complete(
    tmp_path, monkeypatch
):
    root, calibration, reuse, qblade, manifest, plan = _state(tmp_path, monkeypatch)
    for record in list(plan["runnable"]):
        record["status"] = "complete"
        record["reason"] = "exact_identity_complete_result"
        plan["complete"].append(record)
    plan["runnable"].clear()
    for row in manifest["cases"]:
        if row.get("raw_reuse_status") == "reused_exact_identity":
            continue
        result = root / row["result_dir"]
        result.mkdir(parents=True, exist_ok=True)
        (result / f"{Path(row['sim']).stem}_run_summary.json").write_text(
            json.dumps({"failed": False}) + "\n"
        )
    inspect = root / "docker_inspect.json"
    inspect.write_text(
        json.dumps(
            [
                {
                    "Platform": "linux",
                    "Image": "sha256:" + "a" * 64,
                    "Config": {"Image": "vorlap-qblade-headless:amd64"},
                    "HostConfig": {"NanoCpus": 8_000_000_000},
                    "State": {"OOMKilled": False},
                }
            ]
        )
        + "\n"
    )
    path = local_finalizer.build_report(
        campaign_root=root,
        calibration=calibration,
        raw_reuse_root=reuse,
        qblade_root=qblade,
        docker_inspect=inspect,
    )
    report = json.loads(path.read_text())
    contract = report["execution_contract"]
    assert report["status"] == "complete"
    assert len(report["case_results"]) == 52
    assert contract["exact_raw_reuse_case_count"] == 18
    assert contract["new_raw_case_count"] == 8
    assert contract["hybrid_case_count"] == 26
    assert contract["container"]["inspect_file"]["path"] == "docker_inspect.json"
