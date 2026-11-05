from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "paper_drafts/NAWEA_QBladeBridge/figure_generation/generate_hvawt_qblade_replay_visual.py"
)
SPEC = importlib.util.spec_from_file_location("generate_hvawt_qblade_replay_visual", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

VARIANT_SCRIPT = (
    ROOT
    / "paper_drafts/NAWEA_QBladeBridge/figure_generation/"
    "generate_hvawt_qblade_replay_visual_u11p25.py"
)
VARIANT_SPEC = importlib.util.spec_from_file_location(
    "generate_hvawt_qblade_replay_visual_u11p25", VARIANT_SCRIPT
)
assert VARIANT_SPEC is not None and VARIANT_SPEC.loader is not None
variant = importlib.util.module_from_spec(VARIANT_SPEC)
VARIANT_SPEC.loader.exec_module(variant)

U35_SCRIPT = (
    ROOT
    / "paper_drafts/NAWEA_QBladeBridge/figure_generation/"
    "generate_hvawt_qblade_replay_visual_u35_default_wake.py"
)
U35_SPEC = importlib.util.spec_from_file_location(
    "generate_hvawt_qblade_replay_visual_u35_default_wake", U35_SCRIPT
)
assert U35_SPEC is not None and U35_SPEC.loader is not None
u35 = importlib.util.module_from_spec(U35_SPEC)
U35_SPEC.loader.exec_module(u35)


def test_load_inputs_accepts_only_complete_exact_state_replay(
    tmp_path: Path, monkeypatch
) -> None:
    frame_root = tmp_path / "frames"
    frame_root.mkdir()
    last_step = 533
    frame_count = 60
    steps = [
        (frame_index * last_step + (frame_count - 1) // 2) // (frame_count - 1)
        for frame_index in range(frame_count)
    ]
    pd.DataFrame(
        {"frame_index": range(frame_count), "stored_step_index": steps}
    ).to_csv(frame_root / "sampled_frame_map.tsv", sep="\t", index=False)
    for frame_index in range(frame_count):
        Image.new("RGB", (4, 4), (frame_index, 0, 0)).save(
            frame_root / f"frame_{frame_index:04d}.png"
        )

    history_path = tmp_path / "history.csv"
    history = pd.DataFrame(
        {
            "time_s": np.arange(534) * 0.0075,
            "response_mode6_filtered_qdot_mps": np.zeros(534),
            "response_mode6_amplitude_estimate_m": np.zeros(534),
            "response_mode6_engineering_generalized_force_n": np.zeros(534),
            "response_spectral_applied_residual_maximum_nodal_force_n": np.zeros(534),
            "response_engineering_maximum_abs_nodal_force_n": np.zeros(534),
            "response_total_maximum_nodal_force_n": np.zeros(534),
            "response_maximum_flexible_speed_mps": np.zeros(534),
        }
    )
    history.to_csv(history_path, index=False)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "failed": False,
                "stopped": False,
                "completed_steps": 534,
                "requested_steps": 534,
            }
        )
    )
    monkeypatch.setattr(replay, "HISTORY", history_path)
    monkeypatch.setattr(replay, "SUMMARY", summary_path)

    loaded_map, loaded_history, loaded_summary, frame_paths = replay.load_inputs(
        frame_root
    )

    assert loaded_map["stored_step_index"].tolist() == steps
    assert len(loaded_history) == 534
    assert loaded_summary["completed_steps"] == 534
    assert len(frame_paths) == 60
    assert frame_paths[0].name == "frame_0000.png"
    assert frame_paths[-1].name == "frame_0059.png"


def test_manuscript_keeps_replay_outside_quantitative_evidence() -> None:
    manuscript = (
        ROOT / "paper_drafts/NAWEA_QBladeBridge/NAWEA2026_QBladeBridge.tex"
    ).read_text()

    assert "hvawt_qblade_hybrid_replay.png" in manuscript
    assert "Hybrid VorLap SIL replay" in manuscript
    assert "initial-transient view illustrates the coupled model" in manuscript
    assert "not a lock-in or load assessment" in manuscript
    assert "quantitative comparisons use the longer records defined below" in manuscript


def test_u11p25_variant_changes_only_the_uniform_inflow_record() -> None:
    base_text = "header\n7.25                                              MEANINF - wind\nfooter\n"
    expected = "header\n11.25                                             MEANINF - wind\nfooter\n"

    assert variant.expected_variant_sim_text(base_text) == expected
    with pytest.raises(RuntimeError, match="one 7.25-m/s MEANINF"):
        variant.expected_variant_sim_text("header without a wind record\n")


def test_u11p25_variant_requires_the_exact_completed_runtime() -> None:
    settings = {
        "MEANINF": "11.25",
        "RPMPRESCRIBED": "1",
        "TIMESTEP": "0.0075",
        "NUMTIMESTEPS": "534",
        "RAMPUP": "10",
        "ADDDAMP": "0",
        "STOREFROM": "0",
        "STOREREPLAY": "true",
        "WNDTYPE": "0",
    }
    summary = {
        "reproducibility": {"qblade_case_settings": settings},
        "coupling_mode": "sil-vorlap",
        "qblade_execution_mode": "step-loop",
        "threads": 6,
        "use_opencl": False,
        "failed": False,
        "stopped": False,
        "completed_steps": 534,
        "requested_steps": 534,
    }
    history = pd.DataFrame({"time_s": np.arange(534) * 0.0075})

    variant.validate_runtime(summary, history)

    wrong_wind = json.loads(json.dumps(summary))
    wrong_wind["reproducibility"]["qblade_case_settings"]["MEANINF"] = "7.25"
    with pytest.raises(RuntimeError, match="MEANINF=11.25"):
        variant.validate_runtime(wrong_wind, history)


def test_u35_variant_changes_only_the_uniform_inflow_record() -> None:
    base_text = "header\n7.25                                              MEANINF - wind\nfooter\n"
    expected = "header\n35.00                                             MEANINF - wind\nfooter\n"

    assert u35.expected_variant_sim_text(base_text) == expected
    with pytest.raises(RuntimeError, match="one 7.25-m/s MEANINF"):
        u35.expected_variant_sim_text("header without a wind record\n")


def test_u35_variant_requires_native_default_wake_capture_runtime() -> None:
    settings = {
        "MEANINF": "35.00",
        "RPMPRESCRIBED": "1",
        "TIMESTEP": "0.0075",
        "NUMTIMESTEPS": "534",
        "RAMPUP": "10",
        "ADDDAMP": "0",
        "STOREFROM": "0",
        "STOREREPLAY": "true",
        "WNDTYPE": "0",
    }
    summary = {
        "reproducibility": {"qblade_case_settings": settings},
        "coupling_mode": "sil-vorlap",
        "qblade_execution_mode": "step-loop",
        "threads": 6,
        "use_opencl": False,
        "failed": False,
        "stopped": False,
        "completed_steps": 534,
        "requested_steps": 534,
    }
    history = pd.DataFrame({"time_s": np.arange(534) * 0.0075})

    u35.validate_runtime(summary, history)
    assert u35.DEFAULT_WAKE_FRAME_CROP == (452, 5, 1850, 937)

    wrong_wind = json.loads(json.dumps(summary))
    wrong_wind["reproducibility"]["qblade_case_settings"]["MEANINF"] = "11.25"
    with pytest.raises(RuntimeError, match="MEANINF=35"):
        u35.validate_runtime(wrong_wind, history)
