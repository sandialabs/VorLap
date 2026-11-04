from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from vorlap.dlc_manifest import DLCManifestError


RUN_HEADLESS = Path(__file__).resolve().parents[1] / "run_headless"


def load_recovery_module():
    path = RUN_HEADLESS / "recover_parked_dlc_provenance.py"
    spec = importlib.util.spec_from_file_location("recover_parked_dlc_provenance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


recovery = load_recovery_module()


def write_audit(
    root: Path,
    *,
    passed: bool,
    code: str = "checkpoint_runtime_identity_mismatch",
    scenario: str = "vorlap",
) -> dict:
    path = root / "audits" / "results_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    issues = []
    if not passed:
        issues.append(
            {
                "severity": "error",
                "code": code,
                "scenario": scenario,
                "case": root.name,
            }
        )
    path.write_text(
        json.dumps({"passed": passed, "issues": issues}) + "\n",
        encoding="utf-8",
    )
    return {"row_id": root.name, "run_root": str(root)}


def test_recovery_plan_selects_only_failed_checkpoint_python_identity_rows(tmp_path):
    rows = [
        write_audit(tmp_path / "row0", passed=True),
        write_audit(tmp_path / "row1", passed=False),
        write_audit(tmp_path / "row2", passed=False),
        write_audit(tmp_path / "row3", passed=True),
    ]

    assert recovery.recoverable_indices({"rows": rows}) == [1, 2]
    assert recovery.slurm_array_spec([1, 2, 4, 7, 8, 9]) == "1-2,4,7-9"


def test_recovery_plan_fails_closed_on_any_other_results_error(tmp_path):
    row = write_audit(
        tmp_path / "row0",
        passed=False,
        code="checkpoint_project_binding_mismatch",
    )

    with pytest.raises(DLCManifestError, match="non-recoverable"):
        recovery.recoverable_indices({"rows": [row]})


def test_recovery_plan_rejects_checkpoint_mismatch_outside_vorlap(tmp_path):
    row = write_audit(
        tmp_path / "row0",
        passed=False,
        scenario="baseline",
    )

    with pytest.raises(DLCManifestError, match="non-recoverable"):
        recovery.recoverable_indices({"rows": [row]})


def test_provenance_recovery_submitter_is_zero_step_and_chains_aggregate():
    submitter = RUN_HEADLESS / "submit_parked_dlc_provenance_recovery.sj"
    subprocess.run(["bash", "-n", str(submitter)], check=True)
    text = submitter.read_text(encoding="utf-8")
    helper = (RUN_HEADLESS / "recover_parked_dlc_provenance.py").read_text(
        encoding="utf-8"
    )

    assert "recover_parked_dlc_provenance.py" in text
    assert "--dependency=${dependency}" in text
    assert "DLC_PHASE=aggregate" in text
    assert "--array=" in text
    assert "DLC_RECOVERY_CONCURRENCY:-6" in text
    assert "recover-export" in helper
    assert '"advanced_timesteps": 0' in helper
    assert '"ONLY_POSTPROCESS": "1"' in helper
    assert "GENERATE_TURBSIM" not in text
    assert "DLC_PHASE=qblade" not in text
