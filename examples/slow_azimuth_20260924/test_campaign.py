"""Physical and export acceptance for the prescribed 120-degree sweep."""
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("slow_campaign", Path(__file__).with_name("campaign.py"))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def columns(count=160, phase=148.33358):
    # The arbitrary measured startup phase intentionally differs from 147 deg.
    return {"Time [s]": [i * campaign.DT for i in range(count)],
            "LSS Rpm [rpm]": [1 / 30] * count,
            "LSS Azimuthal Pos. [deg]": [(phase + 0.2 * i * campaign.DT) % 360
                                        for i in range(count)]}


def test_full_600s_sector_and_660s_trajectory():
    result = campaign.audit_columns(columns(88000), min_last_time=659.9925,
                                    expected_samples=88000)
    assert result["actual_time_zero_azimuth_deg"] == pytest.approx(148.33358)
    assert result["actual_analysis_start_azimuth_deg"] == pytest.approx(160.33358)
    assert result["actual_unwrapped_sweep_deg"] == pytest.approx(131.9985)
    assert result["actual_retained_endpoint_sweep_deg"] == pytest.approx(119.9985)
    assert result["nominal_600s_sweep_deg"] == pytest.approx(120)
    assert result["measured_mean_angular_speed_deg_s"] == pytest.approx(0.2)


def test_wrapped_azimuth_maintains_positive_slope():
    result = campaign.audit_columns(columns(88000, phase=359.9), min_last_time=659.9925)
    assert result["actual_last_azimuth_deg"] == pytest.approx(131.8985)
    assert result["actual_unwrapped_sweep_deg"] == pytest.approx(131.9985)


@pytest.mark.parametrize("rpm", [0, 1, -1 / 30, float("nan")])
def test_wrong_rpm_cannot_pass(rpm):
    data = columns()
    data["LSS Rpm [rpm]"][12] = rpm
    with pytest.raises(ValueError):
        campaign.audit_columns(data, min_last_time=1.1925)


@pytest.mark.parametrize("deg_s", [0.0, -0.2, 0.1, 0.3])
def test_rpm_header_alone_cannot_hide_wrong_measured_rotation(deg_s):
    data = columns()
    data["LSS Azimuthal Pos. [deg]"] = [147 + deg_s * i * campaign.DT for i in range(160)]
    with pytest.raises(ValueError, match="prescribed slow sweep"):
        campaign.audit_columns(data, min_last_time=1.1925)


def test_missing_or_short_time_history_fails():
    with pytest.raises(ValueError, match="cover"):
        campaign.audit_columns(columns(), min_last_time=659.9925)
    data = columns()
    for values in data.values():
        values.pop(34)
    with pytest.raises(ValueError, match="timesteps"):
        campaign.audit_columns(data, min_last_time=1.1925)


def test_millisecond_export_grid_and_count():
    data = columns()
    data["Time [s]"] = [round(v, 3) for v in data["Time [s]"]]
    assert campaign.audit_columns(data, min_last_time=1.1925,
                                  expected_samples=160)["samples"] == 160
    with pytest.raises(ValueError, match="every requested"):
        campaign.audit_columns(data, min_last_time=1.1925, expected_samples=161)


def test_all_three_blades_exported_and_each_load_required():
    base = (campaign.ROOT / campaign.previous.RESULT_FILTER_SOURCE_RELATIVE).read_text().splitlines()
    requested = campaign.result_channels(base)
    assert len(requested) == len(set(requested)) == 75
    for blade in range(1, 4):
        assert len([x for x in requested if f"BLD_{blade}" in x]) == 24
    data = {key: [0.0] * 160 for key in requested}
    assert campaign.audit_required_channels(data, requested, 160)["all_required_channels_finite_and_complete"]
    key = "Y_l Mom. BLD_3 pos 0.230 [Nm]"
    del data[key]
    with pytest.raises(ValueError, match="Missing requested"):
        campaign.audit_required_channels(data, requested, 160)
    data[key] = [0.0] * 159
    with pytest.raises(ValueError, match="Incomplete"):
        campaign.audit_required_channels(data, requested, 160)
    data[key].append(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        campaign.audit_required_channels(data, requested, 160)


def test_modal_transfer_is_supported_by_archived_comparison():
    receipt = campaign.basis_transfer()
    assert receipt["basis_rpm"] == 1
    assert receipt["simulation_rpm"] == 1 / 30
    assert {r["tracked_mode"] for r in receipt["mode_comparisons"]} == {6, 12}
    for row in receipt["mode_comparisons"]:
        assert abs(row["frequency_shift_fraction"]) < 0.001
        assert row["action_translation_mac"] > 0.999


def test_fresh_source_rebuild_retains_physics_and_avoids_old_state(tmp_path):
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
    rebuilt = tmp_path / "fresh_case"
    campaign.restore_source(rebuilt, original, source)
    assert not (rebuilt / "old_checkpoint.qpr").exists()
    sim = rebuilt / Path(original["sim"]).name
    assert campaign.previous._physics_assets(rebuilt, sim) == original["common_physics_contract"]["physical_assets"]


def test_prepared_trio_retains_all_blades_and_prescribed_motion(tmp_path):
    reference = campaign.ROOT / "run_headless/runs/hvawt_complete_dlc_three_path_v3"
    if not (reference / "campaign_manifest.json").exists():
        pytest.skip("Archived full-campaign metadata is not downloaded in this checkout")
    root = tmp_path / "prepared"
    args = SimpleNamespace(campaign_root=root, source_campaign=reference,
                           azimuth_deg=147.0, smoke_uniform=True)
    manifest = campaign.prepare(args)
    requested = set((root / manifest["result_filter"]).read_text().splitlines())
    assert len(requested) == 75
    assert manifest["rotor_speed_rpm"] == pytest.approx(1 / 30)
    for case in manifest["cases"]:
        sim = root / case["sim"]
        assert float(campaign.read_qblade_keyword(sim, "RPMPRESCRIBED")) == pytest.approx(1 / 30)
        assert int(campaign.read_qblade_keyword(sim, "NUMTIMESTEPS")) == 88000
        storage = set((root / case["case_dir"] / case["storage_filter"]).read_text().splitlines())
        assert requested <= storage
    with pytest.raises(FileExistsError, match="fresh"):
        campaign.prepare(args)
    with pytest.raises(ValueError, match="uniform smoke"):
        campaign.run(SimpleNamespace(campaign_root=root, smoke_steps=0))
    sim = root / manifest["cases"][0]["sim"]
    campaign.replace_qblade_keyword_file(sim, "RPMPRESCRIBED", 1.0)
    with pytest.raises(ValueError, match="Prepared input changed"):
        campaign.validate(root, check_wind=False)
