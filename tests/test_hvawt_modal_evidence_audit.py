from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "run_headless" / "audit_hvawt_modal_evidence.py"
CONFIG_DIR = ROOT / "run_headless" / "config"


def load_audit():
    spec = importlib.util.spec_from_file_location("audit_hvawt_modal_evidence", AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["audit_hvawt_modal_evidence"] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_marker_hash(config_dir: Path, relative_label: str) -> None:
    marker_path = config_dir / "hvawt_modal_evidence_review_marker.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["tracked_evidence"][relative_label] = sha256(config_dir / Path(relative_label).name)
    marker_path.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


TRACKED_MODAL_CSVS = (
    "hvawt_campbell_mode_properties.csv",
    "hvawt_campbell_mode_tracking.csv",
    "hvawt_response_mode_force_duals.csv",
    "hvawt_response_mode_force_duals_repeat.csv",
    "hvawt_response_mode_shapes.csv",
    "hvawt_response_mode_shapes_repeat.csv",
)


def test_tracked_hvawt_modal_csvs_are_lf_only():
    for name in TRACKED_MODAL_CSVS:
        raw = CONFIG_DIR.joinpath(name).read_bytes()
        assert b"\r" not in raw, name
        assert raw.endswith(b"\n"), name


def test_tracked_hvawt_modal_evidence_passes_clean_checkout_audit():
    audit = load_audit()
    result = audit.audit(repo_root=ROOT, config_dir=CONFIG_DIR)

    assert result["status"] == "pass"
    assert result["tracked_file_count"] == 9
    assert result["owens_commit"] == audit.EXPECTED_OWENS_COMMIT
    assert result["selected_damping_candidate_id"] is None
    assert result["repeatability_summary"] == {
        "minimum_selected_action_shape_mac": pytest.approx(0.9999999999984774),
        "maximum_phase_aligned_selected_action_shape_relative_difference": pytest.approx(
            1.316140482223967e-6
        ),
        "minimum_selected_force_dual_mac": pytest.approx(0.9999999999993137),
        "maximum_phase_aligned_selected_force_dual_relative_difference": pytest.approx(
            9.424626873050301e-7
        ),
        "maximum_cross_basis_target_generalized_force_error": pytest.approx(
            4.696878486828982e-7
        ),
        "maximum_cross_basis_off_target_generalized_force": pytest.approx(
            1.358646100936869e-7
        ),
    }


def test_modal_evidence_audit_runs_from_minimal_clean_checkout(tmp_path):
    checkout = tmp_path / "checkout"
    shutil.copytree(CONFIG_DIR, checkout / "run_headless" / "config")
    for name in (
        "qblade_to_owens.py",
        "run_owens_campbell.jl",
        "identify_hvawt_response_modes.py",
        "audit_hvawt_modal_evidence.py",
    ):
        destination = checkout / "run_headless" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "run_headless" / name, destination)
    source_relative = Path(
        "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/Structure/10m_HVAWT_reference_substructure.str"
    )
    source_destination = checkout / source_relative
    source_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / source_relative, source_destination)

    completed = subprocess.run(
        [sys.executable, str(checkout / "run_headless" / "audit_hvawt_modal_evidence.py")],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(completed.stdout)["status"] == "pass"


def test_modal_evidence_audit_rejects_dirty_owens_request(tmp_path):
    audit = load_audit()
    copied = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, copied)
    request_path = copied / "hvawt_owens_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["execution_provenance"]["owens_git_dirty"] = True
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    update_marker_hash(copied, "run_headless/config/hvawt_owens_request.json")

    with pytest.raises(ValueError, match="owens_request.execution_provenance.owens_git_dirty"):
        audit.audit(repo_root=ROOT, config_dir=copied)


def test_modal_evidence_audit_rejects_absolute_host_path(tmp_path):
    audit = load_audit()
    copied = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, copied)
    metadata_path = copied / "hvawt_owens_run_metadata.txt"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + "Host path: /private/tmp/leak\n",
        encoding="utf-8",
    )
    identification_path = copied / "hvawt_response_mode_identification.json"
    identification = json.loads(identification_path.read_text(encoding="utf-8"))
    identity = identification["provenance"]["tracked_clean_checkout_evidence"][
        "hvawt_owens_run_metadata.txt"
    ]
    identity["sha256"] = sha256(metadata_path)
    identity["size_bytes"] = metadata_path.stat().st_size
    identification_path.write_text(
        json.dumps(identification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    update_marker_hash(copied, "run_headless/config/hvawt_owens_run_metadata.txt")
    update_marker_hash(copied, "run_headless/config/hvawt_response_mode_identification.json")

    with pytest.raises(ValueError, match="Absolute host path leaked"):
        audit.audit(repo_root=ROOT, config_dir=copied)


def test_modal_evidence_audit_rejects_crlf_csv(tmp_path):
    audit = load_audit()
    copied = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, copied)
    name = "hvawt_response_mode_shapes.csv"
    path = copied / name
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    update_marker_hash(copied, f"run_headless/config/{name}")

    with pytest.raises(ValueError, match="not LF-only"):
        audit.audit(repo_root=ROOT, config_dir=copied)
