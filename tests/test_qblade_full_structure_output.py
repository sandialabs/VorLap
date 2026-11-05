from __future__ import annotations

import json
from pathlib import Path
import shutil

from vorlap.qblade_files import read_qblade_keyword, resolve_qblade_path
from vorlap.qblade_full_structure_output import (
    OUTPUT_MODE,
    build_monitor_plan,
    configure_full_structure_outputs,
    filter_channels,
    write_full_structure_filter,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CASE = REPO_ROOT / "examples" / "ReferenceHVAWTQBladeModel"
REFERENCE_SIM = REFERENCE_CASE / "baseline_10m_HVAWT_reference_turbine_Wwnd.sim"


def structural_file_for_sim(sim_path: Path) -> Path:
    turbine = resolve_qblade_path(sim_path, str(read_qblade_keyword(sim_path, "TURBFILE")))
    return resolve_qblade_path(
        turbine, str(read_qblade_keyword(turbine, "STRUCTURALFILE"))
    )


def test_full_structure_plan_matches_converted_physical_elements() -> None:
    # The public uniform blade stations replace private reinforcement stations.
    # Counts bind this explicit example mesh, not the historical paper model.
    monitors, metadata = build_monitor_plan(REFERENCE_SIM)
    assert metadata["output_mode"] == OUTPUT_MODE
    assert metadata["monitor_count"] == 186
    assert metadata["component_monitor_counts"] == {
        "BLD_1": 8,
        "BLD_2": 8,
        "BLD_3": 8,
        "STR_1_1": 19,
        "STR_2_1": 19,
        "STR_1_2": 19,
        "STR_2_2": 19,
        "STR_1_3": 19,
        "STR_2_3": 19,
        "TWR": 12,
        "TRQ": 19,
        "SUB_1": 10,
        "SUB_2": 7,
    }
    channels = filter_channels(monitors)
    assert len(channels) == 560
    assert len(channels) == len(set(channels))
    assert channels[:5] == [
        "Time [s]",
        "LSS Rpm [rpm]",
        "Z_l For. BLD_1 pos 0.025 [N]",
        "X_l Mom. BLD_1 pos 0.025 [Nm]",
        "Y_l Mom. BLD_1 pos 0.025 [Nm]",
    ]
    assert any("STR_2_3" in channel for channel in channels)
    assert any("TWR" in channel for channel in channels)
    assert any("TRQ" in channel for channel in channels)
    assert any("SUB_Bottom" in channel for channel in channels)
    assert any("SUB_Upper" in channel for channel in channels)
    assert not any("X_l For." in channel for channel in channels)
    assert not any("Z_l Mom." in channel for channel in channels)


def test_generated_case_patch_preserves_external_swap_block(tmp_path: Path) -> None:
    case = tmp_path / "case"
    shutil.copytree(REFERENCE_CASE, case)
    sim_path = case / REFERENCE_SIM.name
    result_filter = tmp_path / "full_structure_strain_results.filter"
    filter_metadata = tmp_path / "full_structure_strain_results.json"
    written = write_full_structure_filter(
        sim_path,
        result_filter,
        metadata_path=filter_metadata,
    )
    assert written["result_filter_channel_count"] == 560
    assert json.loads(filter_metadata.read_text(encoding="utf-8"))["monitor_count"] == 186

    structural_file = structural_file_for_sim(sim_path)
    original = structural_file.read_text(encoding="utf-8")
    original_external = original[original.index("EXTERNAL_1_IN") :]
    patch_metadata = tmp_path / "case_patch.json"
    configured = configure_full_structure_outputs(
        sim_path,
        result_filter,
        metadata_path=patch_metadata,
    )
    updated = structural_file.read_text(encoding="utf-8")
    assert updated[updated.index("EXTERNAL_1_IN") :] == original_external
    selected = updated[
        updated.index("LAC_OUT") + len("LAC_OUT") : updated.index("EXTERNAL_1_IN")
    ]
    selected_lines = [line.strip() for line in selected.splitlines()[1:] if line.strip()]
    assert len(selected_lines) == 169
    assert selected_lines[0].startswith("BLD_1_")
    assert selected_lines[-1].startswith("TRQ_")
    substructure_file = case / "10m_HVAWT_reference_turbine" / "Structure" / "10m_HVAWT_reference_substructure.str"
    substructure = substructure_file.read_text(encoding="utf-8")
    sub_tokens = [
        line.strip()
        for line in substructure.splitlines()
        if line.strip().startswith("SUB_")
    ]
    assert len(sub_tokens) == 17
    assert sub_tokens[0].startswith("SUB_1_")
    assert sub_tokens[-1].startswith("SUB_2_")
    assert configured["external_swap_block_modified"] is False
    assert json.loads(patch_metadata.read_text(encoding="utf-8"))["monitor_count"] == 186
