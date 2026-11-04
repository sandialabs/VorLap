from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from run_headless import run_ehrmann2014_work_map_flight as scheduler

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(tmp_path: Path, monkeypatch, *, hybrid_complete: int = 0):
    work_map = tmp_path / "NACA0018_work_map.json"
    work_map.write_text('{"frozen": true}\n', encoding="utf-8")
    reuse_root = tmp_path / "frozen_raw_results"
    reuse_root.mkdir()
    qblade_root = tmp_path / f"QBladeCE_{scheduler.campaign.FROZEN_QBLADE_VERSION}"
    qblade_root.mkdir()
    runtime_identity = {
        "version": scheduler.campaign.FROZEN_QBLADE_VERSION,
        "library_sha256": scheduler.campaign.FROZEN_QBLADE_LIBRARY_SHA256,
        "runtime_dependency_fingerprint": (
            scheduler.campaign.FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
        ),
    }
    monkeypatch.setattr(
        scheduler, "_qblade_runtime_identity", lambda _root: dict(runtime_identity)
    )
    root = tmp_path / "campaign"
    root.mkdir()
    rows = []
    complete = []
    runnable = []
    for physical in range(20):
        for scenario, suffix in zip(
            scheduler.campaign.WORK_MAP_SCENARIOS, ("raw", "hybrid")
        ):
            index = 2 * physical + (suffix == "hybrid")
            case_id = f"physical_{physical:02d}_{suffix}"
            row = {
                "index": int(index),
                "case_id": case_id,
                "scenario": scenario,
                "num_steps": 40_000 if physical == 19 else 20_000,
                "result_dir": f"results/{case_id}",
                "raw_reuse_status": (
                    "reused_exact_identity"
                    if suffix == "raw"
                    else "no_candidate_run_required"
                ),
            }
            rows.append(row)
            record = {
                "index": int(index),
                "case_id": case_id,
                "num_steps": row["num_steps"],
                "status": "complete" if suffix == "raw" else "runnable",
                "reason": (
                    "exact_identity_complete_result"
                    if suffix == "raw"
                    else "no_run_summary"
                ),
            }
            if suffix == "raw":
                source = reuse_root / case_id
                source.mkdir()
                result = root / row["result_dir"]
                result.mkdir(parents=True)
                (result / "reuse_receipt.json").write_text(
                    json.dumps(
                        {
                            "schema": scheduler.campaign.REUSE_RECEIPT_SCHEMA,
                            "source_result": str(source.resolve()),
                            "copied_result": row["result_dir"],
                            "qblade_library_sha256": runtime_identity[
                                "library_sha256"
                            ],
                            "qblade_runtime_dependency_fingerprint": runtime_identity[
                                "runtime_dependency_fingerprint"
                            ],
                            "sil_vorlap_max_abs_force": 0.0,
                            "coupling_mode": "sil-vorlap",
                            "native_callback_update_count": 0,
                        }
                    ),
                    encoding="utf-8",
                )
                complete.append(record)
            elif physical < hybrid_complete:
                record["status"] = "complete"
                record["reason"] = "exact_identity_complete_result"
                complete.append(record)
            else:
                runnable.append(record)
    runnable.sort(key=lambda row: (-int(row["num_steps"]), int(row["index"])))
    manifest = {
        "schema": scheduler.campaign.WORK_MAP_SCHEMA,
        "comparison_series": list(scheduler.campaign.WORK_MAP_SCENARIOS),
        "fingerprint": "work-map-campaign-fingerprint",
        "work_map_identity": {
            "schema": "vorlap-airfoil-force-motion-work-map-v1",
            "sha256": _sha256(work_map),
            "fingerprint": "work-map-fingerprint",
            "confidence_multiplier": 1.96,
        },
        "qblade_runtime_identity": runtime_identity,
        "cases": rows,
    }
    plan = {"complete": complete, "runnable": runnable, "blocked": []}
    monkeypatch.setattr(
        scheduler.campaign,
        "validate",
        lambda _path: {"root": root, "manifest": manifest},
    )
    monkeypatch.setattr(scheduler.legacy, "ordered_case_plan", lambda *_args: plan)
    return root, work_map, reuse_root, qblade_root, manifest, plan


def test_execution_plan_requires_exact_20_raw_reuse_and_20_hybrids(
    tmp_path, monkeypatch
):
    root, work_map, reuse_root, qblade_root, _manifest, _plan = _state(
        tmp_path, monkeypatch
    )
    state = scheduler.validate_execution_plan(
        root,
        work_map=work_map,
        raw_reuse_root=reuse_root,
        qblade_root=qblade_root,
    )
    assert len(state["plan"]["complete"]) == 20
    assert len(state["runnable_hybrids"]) == 20
    assert state["hybrid_complete"] == []
    assert all(row["case_id"].endswith("_hybrid") for row in state["runnable_hybrids"])


def test_execution_plan_accepts_checkpoint_resume_accounting(tmp_path, monkeypatch):
    root, work_map, reuse_root, qblade_root, _manifest, _plan = _state(
        tmp_path, monkeypatch, hybrid_complete=7
    )
    state = scheduler.validate_execution_plan(
        root,
        work_map=work_map,
        raw_reuse_root=reuse_root,
        qblade_root=qblade_root,
    )
    assert len(state["hybrid_complete"]) == 7
    assert len(state["runnable_hybrids"]) == 13


def test_execution_plan_rejects_different_map_bytes(tmp_path, monkeypatch):
    root, work_map, reuse_root, qblade_root, _manifest, _plan = _state(
        tmp_path, monkeypatch
    )
    work_map.write_text('{"frozen": false}\n', encoding="utf-8")
    with pytest.raises(scheduler.WorkMapSchedulerError, match="exact schema-v2"):
        scheduler.validate_execution_plan(
            root,
            work_map=work_map,
            raw_reuse_root=reuse_root,
            qblade_root=qblade_root,
        )


def test_execution_plan_rejects_raw_receipt_from_other_root(tmp_path, monkeypatch):
    root, work_map, reuse_root, qblade_root, manifest, _plan = _state(
        tmp_path, monkeypatch
    )
    first_raw = next(
        row for row in manifest["cases"] if row["scenario"] == "raw_qblade"
    )
    receipt_path = root / first_raw["result_dir"] / "reuse_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_result"] = str(
        (tmp_path / "different" / first_raw["case_id"]).resolve()
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(scheduler.WorkMapSchedulerError, match="RAW_REUSE_ROOT"):
        scheduler.validate_execution_plan(
            root,
            work_map=work_map,
            raw_reuse_root=reuse_root,
            qblade_root=qblade_root,
        )


def test_execution_plan_rejects_missing_raw_receipt(tmp_path, monkeypatch):
    root, work_map, reuse_root, qblade_root, manifest, _plan = _state(
        tmp_path, monkeypatch
    )
    first_raw = next(
        row for row in manifest["cases"] if row["scenario"] == "raw_qblade"
    )
    (root / first_raw["result_dir"] / "reuse_receipt.json").unlink()
    with pytest.raises(scheduler.WorkMapSchedulerError, match="reuse receipt"):
        scheduler.validate_execution_plan(
            root,
            work_map=work_map,
            raw_reuse_root=reuse_root,
            qblade_root=qblade_root,
        )


def test_execution_plan_rejects_qblade_runtime_mismatch(tmp_path, monkeypatch):
    root, work_map, reuse_root, qblade_root, _manifest, _plan = _state(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        scheduler,
        "_qblade_runtime_identity",
        lambda _root: {
            "version": scheduler.campaign.FROZEN_QBLADE_VERSION,
            "library_sha256": "0" * 64,
            "runtime_dependency_fingerprint": (
                scheduler.campaign.FROZEN_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
            ),
        },
    )
    with pytest.raises(scheduler.WorkMapSchedulerError, match="exact schema-v2"):
        scheduler.validate_execution_plan(
            root,
            work_map=work_map,
            raw_reuse_root=reuse_root,
            qblade_root=qblade_root,
        )


def test_20_by_1_allocation_is_fully_assigned_and_fixed():
    scheduler._allocation_contract(workers=20, threads=1, allocated_cpus=20)
    with pytest.raises(scheduler.WorkMapSchedulerError, match="exactly 20 workers"):
        scheduler._allocation_contract(workers=19, threads=1, allocated_cpus=20)
    with pytest.raises(scheduler.WorkMapSchedulerError, match="exactly 20 allocated"):
        scheduler._allocation_contract(workers=20, threads=1, allocated_cpus=19)
    with pytest.raises(scheduler.WorkMapSchedulerError, match="one thread"):
        scheduler._allocation_contract(workers=20, threads=2, allocated_cpus=20)


def test_dry_run_launches_only_20_hybrids_and_writes_acceptance_schema(
    tmp_path, monkeypatch
):
    root, work_map, reuse_root, qblade_root, manifest, plan = _state(
        tmp_path, monkeypatch
    )
    validated = {
        "root": root,
        "manifest": manifest,
        "plan": plan,
        "runnable_hybrids": plan["runnable"],
        "hybrid_complete": [],
        "work_map": work_map,
        "raw_reuse_root": reuse_root,
        "qblade_runtime_identity": manifest["qblade_runtime_identity"],
    }
    monkeypatch.setattr(
        scheduler, "validate_execution_plan", lambda *_args, **_kwargs: validated
    )
    report_path = scheduler.run_scheduler(
        campaign_root=root,
        qblade_root=qblade_root,
        work_map=work_map,
        raw_reuse_root=reuse_root,
        python_executable="python3",
        dry_run=True,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "vorlap-ehrmann2014-flight-scheduler-v1"
    assert report["status"] == "dry_run"
    assert report["execution_contract"]["schema"] == (
        "vorlap-ehrmann2014-work-map-flight-execution-v1"
    )
    assert report["execution_contract"]["exact_raw_reuse_case_count"] == 20
    assert report["execution_contract"]["qblade_runtime_identity"] == manifest[
        "qblade_runtime_identity"
    ]
    assert report["allocation"]["initial_slot_utilization"] == 1.0
    assert len(report["cases"]) == 20
    assert all(row["case_id"].endswith("_hybrid") for row in report["cases"])
    assert all(row["command"][-1] == "--resume" for row in report["cases"])


def test_work_map_slurm_wrapper_is_dedicated_20_way_schema_v2_path():
    text = (REPO_ROOT / "run_headless/submit_ehrmann2014_work_map_flight.sj").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --nodes=1" in text
    assert "#SBATCH --ntasks=20" in text
    assert "#SBATCH --ntasks-per-node=20" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --hint=nomultithread" in text
    assert "#SBATCH --exclusive" not in text
    assert "#SBATCH --time=02:00:00" in text
    assert 'WORK_MAP="${WORK_MAP:-}"' in text
    assert 'RAW_REUSE_ROOT="${RAW_REUSE_ROOT:-}"' in text
    assert 'fail "WORK_MAP is required' in text
    assert 'fail "RAW_REUSE_ROOT is required' in text
    assert '--raw-reuse-root "${RAW_REUSE_ROOT}"' in text
    assert '--work-map "${WORK_MAP}"' in text
    assert "run_ehrmann2014_work_map_flight" in text
    assert "build_validation_evidence" in text
    assert "--no-paper-copy" in text
    assert 'if metrics["primary_acceptance"]["pass"] is not True:' in text
    assert "raise SystemExit(2)" in text
    assert "submit_ehrmann2014_flight.sj" not in text


def test_legacy_flight_wrapper_is_unchanged_by_dedicated_path():
    text = (REPO_ROOT / "run_headless/submit_ehrmann2014_flight.sj").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --ntasks=40" in text
    assert "WORK_MAP" not in text
    assert "run_ehrmann2014_work_map_flight" not in text
