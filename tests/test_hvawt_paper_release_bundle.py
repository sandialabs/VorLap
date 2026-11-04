from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_drafts/NAWEA_QBladeBridge"
MODULE_PATH = PAPER / "analysis/reproducibility/build_release_bundle.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_hvawt_paper_release_bundle", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_files(module):
    return {
        "README.md": module.GitFile("README.md", b"test readme\n", 0o100644),
        "replay.sh": module.GitFile(
            "replay.sh",
            b"#!/bin/bash\nexit 0\n",
            0o100755,
        ),
    }


def test_release_bundle_is_deterministic_and_self_verifying(tmp_path):
    module = _load_module()
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    kwargs = {
        "source_ref": "synthetic",
        "source_tree": "a" * 40,
        "source_commit": "b" * 40,
        "source_date_epoch": module.DEFAULT_SOURCE_DATE_EPOCH,
    }

    receipt = module.write_bundle(first, _synthetic_files(module), **kwargs)
    module.write_bundle(second, _synthetic_files(module), **kwargs)

    assert first.read_bytes() == second.read_bytes()
    assert receipt["status"] == "verified_prepared_release_bundle"
    assert receipt["file_count"] == 2
    assert receipt["persistent_identifier"] is None
    assert first.with_suffix(".zip.sha256").is_file()
    assert first.stat().st_mode & 0o777 == 0o644

    with zipfile.ZipFile(first) as archive:
        manifest_name = next(
            name for name in archive.namelist() if name.endswith("RELEASE_MANIFEST.json")
        )
        manifest = json.loads(archive.read(manifest_name))
    assert manifest["status"] == "prepared_not_publicly_deposited"
    assert manifest["release_boundary"]["human_authorship_review"] == (
        "required_before_submission"
    )


def test_release_bundle_rejects_manifest_content_mutation(tmp_path):
    module = _load_module()
    bundle = tmp_path / "bundle.zip"
    module.write_bundle(
        bundle,
        _synthetic_files(module),
        source_ref="synthetic",
        source_tree="a" * 40,
        source_commit=None,
    )
    altered = tmp_path / "altered.zip"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(altered, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename.endswith("README.md"):
                data += b"changed\n"
            target.writestr(item, data)

    with pytest.raises(module.PaperReleaseBundleError, match="identity changed"):
        module.verify_bundle(altered)


def test_current_index_release_inventory_contains_claim_chain():
    module = _load_module()
    source_tree = subprocess.run(
        ["git", "write-tree"], cwd=PAPER, check=True, capture_output=True, text=True
    ).stdout.strip()

    files, observed_tree, source_commit = module.collect_git_files(PAPER, source_tree)

    assert observed_tree == source_tree
    assert source_commit is None
    assert {
        "README.md",
        "NAWEA2026_QBladeBridge.tex",
        "NAWEA2026_QBladeBridge.pdf",
        "figures/hvawt_qblade_hybrid_replay.gif",
        "figures/hvawt_qblade_hybrid_replay.mp4",
        "figures/hvawt_qblade_hybrid_replay.png",
        "figures/hvawt_qblade_hybrid_replay_u11p25.gif",
        "figures/hvawt_qblade_hybrid_replay_u11p25.mp4",
        "figures/hvawt_qblade_hybrid_replay_u35_default_wake.gif",
        "figures/hvawt_qblade_hybrid_replay_u35_default_wake.mp4",
        "figures/hvawt_three_path_wind_bin_contribution.pdf",
        "plot_data/hvawt_qblade_hybrid_replay_provenance.json",
        "plot_data/hvawt_qblade_hybrid_replay_u11p25_provenance.json",
        "plot_data/hvawt_qblade_hybrid_replay_u35_default_wake_provenance.json",
        "generated/hvawt_sensitivity_selection_v1.json",
        "analysis/reproducibility/run_all.sh",
        "analysis/reproducibility/build_release_bundle.py",
        "analysis/verify_reproducibility_package.py",
        "figure_generation/generate_hvawt_qblade_replay_visual.py",
    } <= set(files)
