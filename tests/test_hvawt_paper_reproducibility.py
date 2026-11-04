from __future__ import annotations

import importlib.util
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_drafts/NAWEA_QBladeBridge"
MODULE_PATH = PAPER / "analysis/verify_reproducibility_package.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "verify_hvawt_paper_reproducibility", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_compact_package_verifies_and_replays_sensitivity_byte_for_byte():
    module = _load_module()
    result = module.verify_package(PAPER)

    assert result["status"] == "verified_compact_claim_bounded_paper_package"
    assert result["matched_three_path"] == {
        "fingerprint": "0ff5f6b1bf1ad165cb3e928467df8004ad7736acf614ff18eed0285e4216dd1a",
        "case_count": 102,
        "matched_physical_row_count": 34,
        "output_count": 10,
    }
    assert result["matched_contribution_derivative"] == {
        "fingerprint": json.loads((PAPER / "generated/hvawt_matched_three_path_paper_derivative_receipt.json").read_text())["fingerprint"],
        "normal_case_count": 99,
        "u50_case_count": 3,
        "regenerated_byte_for_byte": True,
        "claim_boundary": (
            "Accepted 102-case DEL and monitored moment extrema; common-material Miner "
            "damage and longitudinal strength screening have a separate complete-history gate."
        ),
    }
    current = result["current_hourly_and_cfd"]
    assert current["status"] == "verified_current_hourly_exposure_histories_and_cfd"
    assert current["hourly_wind_range_mps"] == [3, 18]
    assert "generated/presentation_alignment_receipt.json" in current["receipts"]
    assert "analysis/lifetime_exposure_20260921/results/validation.json" in current["receipts"]
    assert result["matched_u35_unsteady_visual"] == {
        "status": "verified_matched_three_path_u35_visual",
        "source_count": 3,
        "data_row_count": 4000,
        "data_sha256": "b9891da6a5f9a01cf96ce4b2654f80ecce2c1b351456d3a879aa40991afe3ee9",
        "figure_sha256": "f41ca356e97737b389b765db351940dae354a080e53e405e9a8162f8de7dfd10",
        "one_way_source_sha256": (
            "774be0b30c5316aa88ab14856a555543a079d95949391e04c6934c28844dfded"
        ),
    }
    assert result["qblade_replay_visual"]["status"] == (
        "verified_presentation_only_replay_derivatives"
    )
    assert result["qblade_replay_visual"]["source_qpr_sha256"] == (
        "98856347e5917b3c063cfe1f94ee8a20e1c32fcc83be1710821702a89e68450d"
    )
    assert result["qblade_replay_visual"]["trace_row_count"] == 534
    assert result["qblade_replay_visual"][
        "source_archive_required_for_visual_regeneration"
    ] is True
    assert result["qblade_replay_visual_u11p25"]["status"] == (
        "verified_presentation_only_u11p25_replay_derivatives"
    )
    assert result["qblade_replay_visual_u11p25"]["source_qpr_sha256"] == (
        "b01ed052e52e902d0ec1a287376a0c72c933a559e52bf8c23c565826827c3d5f"
    )
    assert result["qblade_replay_visual_u11p25"]["trace_row_count"] == 534
    assert result["qblade_replay_visual_u11p25"][
        "source_archive_required_for_visual_regeneration"
    ] is True
    assert result["qblade_replay_visual_u35_default_wake"]["status"] == (
        "verified_presentation_only_u35_default_wake_derivatives"
    )
    assert result["qblade_replay_visual_u35_default_wake"][
        "source_qpr_sha256"
    ] == "09257d8a1f0d5784014411fa11583f75bc869ad3d67d8b11dcd3d24c1e01d9e3"
    assert result["qblade_replay_visual_u35_default_wake"][
        "trace_row_count"
    ] == 534
    assert result["iea10_training_visual"]["status"] == (
        "verified_stationary_training_visual"
    )
    assert result["iea10_training_visual"]["figure_sha256"] == (
        "aa7fe4b19d1f69b15d1d9bbac06d686a7fa6aedd059e757b7056924f76009fed"
    )
    assert result["iea10_training_visual"]["dependency_count"] == 39
    assert result["iea10_training_visual"]["regenerated_byte_for_byte"] is True
    assert result["targeted_sensitivity"]["fingerprint"] == (
        "c588df7222569a110742d628fab059d49b72939cad2017bba2df5cfb3c25cad9"
    )
    assert result["targeted_sensitivity"]["regenerated_byte_for_byte"] is True
    assert result["preregistered_sensitivity_selection"] == {
        "fingerprint": "55a0622ee828175667a63f8379115b4ace317f36585834d9c8be4cdf97252c1e",
        "selected_row_count": 2,
        "selected_row_ids": [
            "dlc6p4_U35p00ms_s4433458",
            "dlc6p4_U35p00ms_s4433456",
        ],
        "case_generation_performed": False,
        "simulation_execution_performed": False,
    }


def test_matched_output_byte_change_fails_closed(tmp_path):
    module = _load_module()
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = paper / "plot_data/hvawt_three_path_weighted_del.csv"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(module.PaperReproducibilityError, match="byte count changed"):
        module._verify_matched_package(paper)


def test_sensitivity_output_byte_change_fails_closed(tmp_path):
    module = _load_module()
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = (
        paper
        / "generated/hvawt_sensitivity_paper_recovered_v2"
        / "hvawt_response_attribution_v1.csv"
    )
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(module.PaperReproducibilityError, match="byte count changed"):
        module._verify_sensitivity_package(paper)


def test_matched_contribution_figure_byte_change_fails_closed(tmp_path):
    module = _load_module()
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = paper / "figures/hvawt_three_path_11min_results_presentation.pdf"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(
        module.PaperReproducibilityError,
        match="does not reproduce byte-for-byte",
    ):
        module._verify_matched_contribution_derivative(paper)


def test_current_material_cycle_and_factor_conventions():
    """Check current amplitude/range, count and asymmetric mean-stress arithmetic."""
    import numpy as np
    path = PAPER / "analysis/dlc11_operating/archived_curve_20260918/reduce_archived_curve.py"
    spec = importlib.util.spec_from_file_location("current_cycle_reduction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    material = json.loads((PAPER / "analysis/dlc11_operating/fatigue_reference/alignment_config.json").read_text())["material"]
    m = math.log(100) / math.log(275 / 200)
    scenario = dict(id="synthetic", curve="power_law", exponent=m,
                    load_factor=1.35, material_retention=.5, cyclic_factor=1.)
    for mean, strength in [(5., 200.), (-5., 100.)]:
        cycles = np.array([[20., mean, 1.]])
        result = module.cycle_result(cycles, (mean-10, mean+10), material, scenario)
        corrected_amplitude = 1.35 * 10 / (1-1.35*abs(mean)/strength)
        expected = (corrected_amplitude / 100)**m / 1e6
        assert math.isclose(result["damage_record"], expected, rel_tol=1e-12)
        halves = np.array([[20., mean, .5], [20., mean, .5]])
        assert math.isclose(module.cycle_result(halves, (mean-10, mean+10), material, scenario)["damage_record"], expected, rel_tol=1e-12)
        assert np.array_equal(cycles, np.array([[20., mean, 1.]]))
    zero = module.cycle_result(np.array([[0., 0., 100.]]), (0., 0.), material, scenario)
    assert zero["damage_record"] == 0 and zero["zero_node_floor_record"] == 0
    with pytest.raises(ValueError, match="Static-strength domain exceeded"):
        module.cycle_result(np.array([[20., 0., 1.]]), (-100., 0.), material, scenario)


def test_current_hourly_input_change_fails_closed(tmp_path):
    module = _load_module()
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = paper / "plot_data/operating_comparison/hourly_by_wind.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(module.PaperReproducibilityError, match="Changed compact input"):
        module._verify_current_paper_additions(paper)


def test_qblade_replay_artifact_byte_change_fails_closed(tmp_path):
    module = _load_module()
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = paper / "figures/hvawt_qblade_hybrid_replay.gif"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(module.PaperReproducibilityError, match="byte count changed"):
        module._verify_replay_visual(paper)


def test_qblade_u11p25_replay_artifact_byte_change_fails_closed(tmp_path):
    module = _load_module()
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = paper / "figures/hvawt_qblade_hybrid_replay_u11p25.gif"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(module.PaperReproducibilityError, match="byte count changed"):
        module._verify_replay_visual_u11p25(paper)


def test_qblade_u35_default_wake_artifact_byte_change_fails_closed(tmp_path):
    module = _load_module()
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = paper / "figures/hvawt_qblade_hybrid_replay_u35_default_wake.gif"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(module.PaperReproducibilityError, match="byte count changed"):
        module._verify_replay_visual_u35_default_wake(paper)


def test_iea10_training_figure_byte_change_fails_closed(tmp_path):
    module = _load_module()
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = paper / "figures/iea10_hybrid_qblade_visual.pdf"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(module.PaperReproducibilityError, match="byte count changed"):
        module._verify_iea10_training_visual(paper)


def _copy_manuscript_inputs(tmp_path):
    paper = tmp_path / "paper"
    for relative in [
        "NAWEA2026_QBladeBridge.tex",
        "analysis/reproducibility/historical_tex_artifacts.json",
        "generated/fitted_del_values.json",
    ]:
        target = paper / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER / relative, target)
    return paper


def test_manuscript_rejects_external_tex_input(tmp_path):
    module = _load_module()
    paper = _copy_manuscript_inputs(tmp_path)
    manuscript = paper / "NAWEA2026_QBladeBridge.tex"
    manuscript.write_text(
        manuscript.read_text(encoding="utf-8") + "\n\\input{generated/obsolete.tex}\n",
        encoding="utf-8",
    )
    with pytest.raises(module.PaperReproducibilityError, match="self-contained"):
        module._verify_manuscript_source(paper)


def test_manuscript_accepts_author_prose_edits_without_regeneration(tmp_path):
    module = _load_module()
    paper = _copy_manuscript_inputs(tmp_path)
    manuscript = paper / "NAWEA2026_QBladeBridge.tex"
    manuscript.write_text(manuscript.read_text().replace(
        "% BEGIN PAPER BLOCK: hourly-damage\n",
        "% BEGIN PAPER BLOCK: hourly-damage\nAuthor-owned wording may change without replotting.\n",
    ))
    before = manuscript.read_bytes()
    assert module._verify_manuscript_source(paper)["status"] == "single_author_owned_tex_source"
    assert manuscript.read_bytes() == before


def test_manuscript_rejects_changed_fitted_value(tmp_path):
    module = _load_module()
    paper = _copy_manuscript_inputs(tmp_path)
    manuscript = paper / "NAWEA2026_QBladeBridge.tex"
    source = manuscript.read_text()
    assert r"\newcommand{\FittedStressRatio}{1.109}" in source
    manuscript.write_text(source.replace(
        r"\newcommand{\FittedStressRatio}{1.109}", r"\newcommand{\FittedStressRatio}{2.000}"
    ))
    with pytest.raises(module.PaperReproducibilityError, match="fitted DEL macros"):
        module._verify_manuscript_source(paper)


def test_historical_tex_change_fails_receipt_binding(tmp_path):
    module = _load_module()
    paper = _copy_manuscript_inputs(tmp_path)
    relative = Path("generated/hvawt_three_path_results_macros.tex")
    archive = paper / module.HISTORICAL_ARCHIVE
    payload = json.loads(archive.read_text())
    record = payload["artifacts"][relative.as_posix()]
    original_identity = {"path": str(relative), "bytes": record["size_bytes"], "sha256": record["sha256"]}
    import base64
    changed = base64.b64decode(record["content_base64"]).replace(b"310.9", b"999.9")
    record.update(content_base64=base64.b64encode(changed).decode(), sha256=hashlib.sha256(changed).hexdigest())
    archive.write_text(json.dumps(payload))
    with pytest.raises(module.PaperReproducibilityError, match="SHA-256 changed"):
        module._require_historical_identity(paper, relative, original_identity, label="historical macro", size_key="bytes")


def test_manuscript_block_reader_rejects_duplicate_markers(tmp_path):
    module = _load_module()
    paper = _copy_manuscript_inputs(tmp_path)
    manuscript = paper / "NAWEA2026_QBladeBridge.tex"
    marker = "% BEGIN PAPER BLOCK: hourly-damage\n"
    manuscript.write_text(manuscript.read_text().replace(marker, marker + marker))
    with pytest.raises(module.ManuscriptSourceError, match="duplicate"):
        module.read_block(paper, "hourly-damage")


def test_manuscript_block_reader_preserves_body_whitespace(tmp_path):
    module = _load_module()
    manuscript = tmp_path / "NAWEA2026_QBladeBridge.tex"
    body = "  Author wording.\r\n\r\n"
    manuscript.write_bytes((
        "% BEGIN PAPER BLOCK: example\r\n" + body + "% END PAPER BLOCK: example\r\n"
    ).encode())
    assert module.read_block(tmp_path, "example") == body



def test_preregistered_selection_byte_change_fails_closed(tmp_path):
    module = _load_module()
    paper = tmp_path / "paper"
    shutil.copytree(PAPER, paper, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = paper / "generated/hvawt_sensitivity_selection_v1.json"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(
        module.PaperReproducibilityError,
        match="fingerprint changed|Preregistered sensitivity selection contract changed",
    ):
        module._verify_sensitivity_selection(paper)


def test_compact_reproducibility_package_is_git_tracked():
    module = _load_module()
    matched = [PAPER / path for path in module.MATCHED_OUTPUTS.values() if path.suffix != ".tex"]
    sensitivity_package = (
        PAPER / "generated/hvawt_sensitivity_paper_recovered_v2"
    )
    required = [
        MODULE_PATH,
        PAPER / "analysis/prepare_presentation_aligned_paper.py",
        PAPER / "generated/presentation_alignment_receipt.json",
        PAPER / "analysis/manuscript_source.py",
        PAPER / "analysis/reproducibility/historical_tex_artifacts.json",
        PAPER / "generated/fitted_del_values.json",
        PAPER / "figures/hvawt_main_talk_histories.pdf",
        PAPER / "analysis/lifetime_exposure_20260921/calculate.py",
        PAPER / "analysis/lifetime_exposure_20260921/inputs/manifest.json",
        PAPER / "analysis/lifetime_exposure_20260921/inputs/wind_rates_all_points.csv",
        PAPER / "analysis/lifetime_exposure_20260921/inputs/hourly_by_wind.csv",
        PAPER / "analysis/lifetime_exposure_20260921/inputs/summary.json",
        PAPER / "presentation/data/full_load_histories/receipt.json",
        PAPER / "presentation/data/full_load_histories/moments_full_600s_U07p00_samples.npz",
        PAPER / "presentation/data/full_load_histories/moments_full_600s_U11p25_samples.npz",
        PAPER / "analysis/reproducibility/README.md",
        PAPER / "analysis/reproducibility/run_all.sh",
        PAPER / "generated/hvawt_three_path_promotion_receipt.json",
        PAPER / "generated/hvawt_sensitivity_selection_v1.json",
        PAPER / "generated/hvawt_matched_three_path_paper_derivative_receipt.json",
        PAPER / "figures/hvawt_three_path_11min_results_presentation.pdf",
        PAPER / "figures/hvawt_three_path_u50_moment_extremes.pdf",
        PAPER / "figures/hvawt_three_path_wind_bin_contribution.pdf",
        PAPER / "figures/hvawt_u35_unsteady_inboard_moments.pdf",
        PAPER / "plot_data/hvawt_u35_unsteady_inboard_moments.csv",
        PAPER / "plot_data/hvawt_u35_unsteady_inboard_moments_source.json",
        PAPER / "figure_generation/generate_hvawt_unsteady_load_visual.py",
        PAPER / "figures/hvawt_qblade_hybrid_replay.png",
        PAPER / "figures/hvawt_qblade_hybrid_replay.gif",
        PAPER / "figures/hvawt_qblade_hybrid_replay.mp4",
        PAPER / "plot_data/hvawt_qblade_hybrid_replay_trace.csv",
        PAPER / "plot_data/hvawt_qblade_hybrid_replay_provenance.json",
        PAPER / "figure_generation/generate_hvawt_qblade_replay_visual.py",
        PAPER / "figures/hvawt_qblade_hybrid_replay_u11p25.gif",
        PAPER / "figures/hvawt_qblade_hybrid_replay_u11p25.mp4",
        PAPER / "plot_data/hvawt_qblade_hybrid_replay_u11p25_trace.csv",
        PAPER / "plot_data/hvawt_qblade_hybrid_replay_u11p25_provenance.json",
        PAPER / "figure_generation/generate_hvawt_qblade_replay_visual_u11p25.py",
        PAPER / "figures/hvawt_qblade_hybrid_replay_u35_default_wake.gif",
        PAPER / "figures/hvawt_qblade_hybrid_replay_u35_default_wake.mp4",
        PAPER / "plot_data/hvawt_qblade_hybrid_replay_u35_default_wake_trace.csv",
        PAPER / "plot_data/hvawt_qblade_hybrid_replay_u35_default_wake_provenance.json",
        PAPER / "figure_generation/generate_hvawt_qblade_replay_visual_u35_default_wake.py",
        PAPER / "figure_generation/generate_iea10_p80_v6_visual.py",
        PAPER / "plot_data/iea10_hybrid_qblade_visual_provenance.json",
        PAPER / "analysis/prepare_hvawt_matched_paper_derivatives.py",
        *matched,
        *sorted(path for path in sensitivity_package.iterdir() if path.is_file()),
    ]
    relative = [path.relative_to(PAPER).as_posix() for path in required]

    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative],
        cwd=PAPER,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
