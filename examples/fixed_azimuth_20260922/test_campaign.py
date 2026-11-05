"""Physical acceptance checks for the fixed rotor, including startup."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

spec = importlib.util.spec_from_file_location("fixed_campaign", Path(__file__).with_name("campaign.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def columns():
    return {"Time [s]": [i * campaign.DT for i in range(160)],
            "LSS Rpm [rpm]": [0.0] * 160,
            "LSS Azimuthal Pos. [deg]": [147.0] * 160}


def test_true_fixed_rotor_passes():
    result = campaign.audit_columns(columns(), azimuth=147, min_last_time=1.1924)
    assert result["samples"] == 160
    assert result["maximum_abs_rpm"] == 0
    assert result["maximum_azimuth_error_deg"] == 0


@pytest.mark.parametrize("channel,value", [
    ("LSS Rpm [rpm]", 1.0), ("LSS Azimuthal Pos. [deg]", 187.0),
    ("LSS Rpm [rpm]", float("nan")),
])
def test_startup_rotation_offset_or_nan_cannot_be_hidden(channel, value):
    data = columns()
    data[channel][1] = value
    with pytest.raises(ValueError):
        campaign.audit_columns(data, azimuth=147, min_last_time=1.1924)


def test_missing_sample_fails_even_when_azimuth_constant():
    data = columns()
    for values in data.values():
        values.pop(34)
    with pytest.raises(ValueError, match="timesteps"):
        campaign.audit_columns(data, azimuth=147, min_last_time=1.1924)


def test_short_result_cannot_pass_as_660s():
    with pytest.raises(ValueError, match="cover"):
        campaign.audit_columns(columns(), azimuth=147, min_last_time=659.99)


def test_millisecond_rounded_time_exports_retain_full_grid():
    data = columns()
    data["Time [s]"] = [round(v, 3) for v in data["Time [s]"]]
    result = campaign.audit_columns(data, azimuth=147, min_last_time=1.1925,
                                    expected_samples=160)
    assert result["samples"] == 160
    with pytest.raises(ValueError, match="every requested"):
        campaign.audit_columns(data, azimuth=147, min_last_time=1.1925,
                                expected_samples=161)


def test_missing_flapwise_channel_cannot_pass_as_finite_export():
    required = ["Time [s]", "Y_l Mom. BLD_1 pos 0.230 [Nm]"]
    data = columns()
    with pytest.raises(ValueError, match="Missing requested"):
        campaign.audit_required_channels(data, required, 160)
    data[required[1]] = [0.0] * 159
    with pytest.raises(ValueError, match="Incomplete result channel"):
        campaign.audit_required_channels(data, required, 160)
    data[required[1]].append(0.0)
    assert campaign.audit_required_channels(data, required, 160)["all_required_channels_finite_and_complete"]
    data[required[1]][3] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        campaign.audit_required_channels(data, required, 160)


def test_modal_transfer_is_quantified_against_recorded_zero_rpm_solution():
    receipt = campaign.basis_transfer()
    assert receipt["basis_rpm"] == 1.0
    assert receipt["simulation_rpm"] == 0.0
    assert {r["tracked_mode"] for r in receipt["mode_comparisons"]} == {6, 12}
    for row in receipt["mode_comparisons"]:
        assert abs(row["frequency_shift_fraction"]) < 0.001
        assert row["action_translation_mac"] > 0.999


def test_clean_rebuild_does_not_copy_previous_runtime_state(tmp_path):
    reference = campaign.ROOT / "run_headless/runs/hvawt_complete_dlc_three_path_v3"
    if not (reference / "campaign_manifest.json").exists():
        pytest.skip("Archived full-campaign metadata is not downloaded in this checkout")
    manifest = json.loads((reference / "campaign_manifest.json").read_text())
    original = next(c for c in manifest["cases"]
                    if c["row_id"] == campaign.ROW and c["scenario"] == "raw_qblade")
    source = tmp_path / "old_campaign"
    (source / "inputs").mkdir(parents=True)
    shutil.copy2(reference / "inputs/hvawt_documented_rayleigh_selection.json", source / "inputs")
    stale = source / original["case_dir"]
    stale.mkdir(parents=True)
    (stale / "old_checkpoint.qpr").write_text("must not be copied")
    (stale / "vorlap_bridge_debug.log").write_text("mutable old state")
    rebuilt = tmp_path / "fresh_case"
    campaign.restore_source(rebuilt, original, source)
    assert not (rebuilt / "old_checkpoint.qpr").exists()
    assert not (rebuilt / "vorlap_bridge_debug.log").exists()
    sim = rebuilt / Path(original["sim"]).name
    assert campaign.previous._physics_assets(rebuilt, sim) == original["common_physics_contract"]["physical_assets"]
