import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "paper_drafts"
    / "NAWEA_QBladeBridge"
    / "analysis"
    / "prepare_wes_paper_artifacts.py"
)
METHODS_SCRIPT = (
    REPO
    / "paper_drafts"
    / "NAWEA_QBladeBridge"
    / "figure_generation"
    / "generate_methods_visuals.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_wes_paper_artifacts",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_methods_module():
    spec = importlib.util.spec_from_file_location(
        "generate_methods_visuals",
        METHODS_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tracked_pilot_evidence_and_manifest_are_internally_consistent():
    module = load_module()
    summaries, component_rows, qualification, manifest, config = (
        module.load_current_evidence(
            pilot_dir=module.DEFAULT_PILOT_DIR,
            qualification_path=module.DEFAULT_QUALIFICATION,
            manifest_path=module.DEFAULT_MANIFEST,
            config_path=module.DEFAULT_CONFIG,
        )
    )

    assert [row["wind_speed_mps"] for row in summaries] == [5.0, 50.0]
    assert len(component_rows) == 18
    assert all(row["violation_count"] == 0 for row in component_rows)
    assert qualification["source_samples_per_cycle_min"] == pytest.approx(
        2.000000245185284
    )
    assert manifest["row_count"] == 34
    assert config["production"]["expected_dlc64_seed_count"] == 3


def test_source_atlas_and_campaign_map_use_current_evidence(tmp_path):
    module = load_module()
    source_rows, experimental_rows = module.source_spectral_atlas(
        module.DEFAULT_SOURCE_H5,
        module.DEFAULT_EXPERIMENTAL_DIR,
        tmp_path / "source_spectral_atlas.pdf",
    )
    manifest = module.load_json(module.DEFAULT_MANIFEST)
    campaign_rows = module.campaign_design_figure(
        manifest,
        tmp_path / "campaign_design_map.pdf",
    )

    assert len(source_rows) == 90 * 8
    assert sum(row["is_strongest"] for row in source_rows) == 90
    assert {
        row["source_plane"] for row in source_rows
    } == {"simulated", "periodic_copy"}
    assert not any(row["aoa_deg"] == 108.0 for row in source_rows)
    assert {row["component"] for row in experimental_rows} == {"lift", "drag"}
    assert all(
        row["comparison_scope"] == "qualitative_frequency_only"
        for row in experimental_rows
    )
    assert len(campaign_rows) == 12
    assert sum(row["realization_count"] for row in campaign_rows) == 34
    fatigue_rows = [row for row in campaign_rows if row["dlc"] == "6.4"]
    assert fatigue_rows[0]["bin_lower_edge_mps"] == 0.0
    assert fatigue_rows[-1]["bin_lower_edge_mps"] == 21.5
    assert fatigue_rows[-1]["bin_upper_edge_mps"] == 35.0
    assert sum(row["annual_fatigue_weight"] for row in campaign_rows) == pytest.approx(
        0.01,
        abs=1.0e-6,
    )
    assert (tmp_path / "source_spectral_atlas.pdf").is_file()
    assert (tmp_path / "campaign_design_map.pdf").is_file()


def test_modal_overlap_uses_tracked_evidence(tmp_path):
    module = load_methods_module()
    modal_rows = module.modal_overlap_selection(
        output_path=tmp_path / "modal_overlap_selection.pdf",
        plot_data_path=tmp_path / "modal_overlap_wind_selection.csv",
    )
    assert len(modal_rows) == 129
    assert sum(row["selected_matrix_speed"] for row in modal_rows) == 11
    assert sum(row["ranked_peak"] for row in modal_rows) >= 10
    assert (tmp_path / "modal_overlap_selection.pdf").is_file()


def test_legacy_turbine_map_replays_archived_weighted_dels(tmp_path):
    """This historical map is not an input to the current manuscript."""
    module = load_methods_module()
    configured = os.environ.get("VORLAP_ARCHIVE_RUNS")
    root = Path(configured).expanduser() if configured else REPO / "run_headless/runs"
    source = root / module.WEIGHTED_DEL.relative_to(REPO / "run_headless/runs")
    if not source.is_file():
        message = (f"Legacy turbine-map integration requires {source}. Set "
                   "VORLAP_ARCHIVE_RUNS to the retained run archive; current paper recreation is independent.")
        if configured:
            pytest.fail(message)
        pytest.skip(message)
    configured_paper = os.environ.get("VORLAP_ARCHIVE_PAPER")
    paper = Path(configured_paper).expanduser() if configured_paper else module.PAPER_DIR
    render = paper / "figures/turbine_del_map_paraview.png"
    metadata = paper / "plot_data/turbine_del_paraview_metadata.json"
    field = paper / "plot_data/turbine_del_field.vtu"
    missing = [str(path) for path in (render, metadata, field) if not path.is_file()]
    if missing:
        message = ("Legacy turbine-map integration requires archived rendering inputs: "
                   + ", ".join(missing)
                   + ". Set VORLAP_ARCHIVE_PAPER to the historical paper directory.")
        if configured_paper:
            pytest.fail(message)
        pytest.skip(message)
    turbine_rows = module.turbine_del_map(
        weighted_del_path=source,
        output_path=tmp_path / "turbine_del_map.pdf",
        plot_data_path=tmp_path / "turbine_del_map.csv",
        paraview_render_path=render,
        paraview_metadata_path=metadata,
        paraview_field_path=field,
    )
    assert len(turbine_rows) == 4
    assert max(row["ratio"] for row in turbine_rows) == pytest.approx(
        2.4830185015
    )
    assert (tmp_path / "turbine_del_map.pdf").is_file()


def test_turbine_map_rejects_stale_paraview_render_metadata(tmp_path):
    module = load_methods_module()
    # Exercise the real hash guard with generic files, without a historical
    # campaign dependency or weakened source-identity checks.
    weighted = tmp_path / "weighted.csv"
    weighted.write_text("quantity,axis,position,signed_del_baseline_weighted,signed_del_vorlap_weighted,ratio_vorlap_to_baseline\n"
                        "moment,X,0.5,10,20,2\n")
    reference = tmp_path / "synthetic.sim"
    reference.write_text("Synthetic hash-verification input; not a turbine model.\n")
    field = tmp_path / "synthetic.vtu"
    field.write_text("Synthetic hash-verification input; not a structural mesh.\n")
    rendering = tmp_path / "synthetic.png"
    rendering.write_bytes(b"synthetic render bytes; rejected before image decoding")
    metadata = {
        "schema": "vorlap-nawea-paraview-del-map-v1",
        "sources": {"weighted_del": {"sha256": module._sha256(weighted)},
                    "qblade_sim": {"sha256": module._sha256(reference)}},
        "outputs": {"paraview_png": {"sha256": "0" * 64},
                    "fielded_vtu": {"sha256": module._sha256(field)}},
    }
    stale_metadata = tmp_path / "stale_paraview_metadata.json"
    stale_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ParaView rendering hash"):
        module.turbine_del_map(
            weighted_del_path=weighted,
            reference_sim=reference,
            output_path=tmp_path / "turbine_del_map.pdf",
            paraview_render_path=rendering,
            paraview_metadata_path=stale_metadata,
            paraview_field_path=field,
        )


def test_full_structure_turbine_map_pairs_ratio_image_with_absolute_scale(
    tmp_path,
    monkeypatch,
):
    module = load_methods_module()
    rows = []
    for family_index, family in enumerate(
        ("blade", "strut", "torque_tube", "tower", "substructure"),
        start=1,
    ):
        for sample in range(10):
            baseline = 1.0e-6 * family_index * (1.0 + 0.05 * sample)
            vorlap = baseline * (1.0 + 0.02 * family_index)
            rows.append(
                {
                    "family": family,
                    "component": family,
                    "monitor_index": family_index,
                    "perimeter_index": sample,
                    "baseline": baseline,
                    "vorlap": vorlap,
                    "ratio": vorlap / baseline,
                }
            )
    monkeypatch.setattr(
        module,
        "_validated_surface_paraview_render",
        lambda *args: (np.ones((1000, 1200, 3), dtype=float), rows),
    )
    output = tmp_path / "full_structure_turbine_map.pdf"
    plot_data = tmp_path / "full_structure_turbine_map.csv"
    summary = module._surface_turbine_del_map(
        output,
        plot_data,
        tmp_path / "render.png",
        tmp_path / "metadata.json",
        tmp_path / "field.vtu",
        tmp_path / "surface.csv",
        tmp_path / "surface.json",
    )
    assert [row["family"] for row in summary] == [
        "blade",
        "strut",
        "torque_tube",
        "tower",
        "substructure",
    ]
    assert all(row["baseline_p05"] < row["baseline_p95"] for row in summary)
    assert all(row["vorlap_median"] > row["baseline_median"] for row in summary)
    assert output.is_file()
    assert plot_data.is_file()


def test_production_validation_fails_closed_when_files_are_missing(tmp_path):
    module = load_module()
    manifest = module.load_json(module.DEFAULT_MANIFEST)
    config = module.load_json(module.DEFAULT_CONFIG)

    with pytest.raises(module.ArtifactError, match="aggregate is incomplete"):
        module.validate_production_aggregate(tmp_path, manifest, config)


def test_require_production_rejects_absent_aggregate():
    module = load_module()

    assert module.main(["--require-production"]) == 2


def test_valid_synthetic_aggregate_generates_final_fatigue_products(tmp_path):
    module = load_module()
    manifest = module.load_json(module.DEFAULT_MANIFEST)
    config = module.load_json(module.DEFAULT_CONFIG)
    axes = config["fatigue"]["primary_axes"]
    signed_rows = []
    condition_rows = []
    weighted_rows = []

    for manifest_row in manifest["rows"]:
        speed = float(manifest_row["wind_speed_mps"])
        for axis_index, axis in enumerate(axes, start=1):
            aggregate_axis = axis.removesuffix("_l")
            baseline = 10.0 * axis_index + 0.1 * speed
            ratio = 1.0 + 0.002 * speed + 0.01 * axis_index
            signed_rows.append(
                {
                    "row_id": manifest_row["row_id"],
                    "dlc": manifest_row["dlc"],
                    "seed": manifest_row["seed"],
                    "fatigue_weight": manifest_row.get("fatigue_weight", ""),
                    "postprocess_warnings": "",
                    "smoke_only": "false",
                    "position": "0.23",
                    "quantity": "Mom.",
                    "axis": aggregate_axis,
                    "unit": "Nm",
                        "channel_baseline": (
                            f"{axis}~Mom.~BLD_1~pos~0.230~[Nm]"
                        ),
                        "channel_vorlap": (
                            f"{axis}~Mom.~BLD_1~pos~0.230~[Nm]"
                        ),
                    "wind_speed_mps": speed,
                    "signed_del_baseline": baseline,
                    "signed_del_vorlap": baseline * ratio,
                    "equivalent_cycles_baseline": config["fatigue"][
                        "equivalent_cycles"
                    ],
                    "equivalent_cycles_vorlap": config["fatigue"][
                        "equivalent_cycles"
                    ],
                }
            )

    dlc64_speeds = config["production"]["expected_dlc64_wind_speeds_mps"]
    for axis_index, axis in enumerate(axes, start=1):
        aggregate_axis = axis.removesuffix("_l")
        for speed in dlc64_speeds:
            baseline = 10.0 * axis_index + 0.1 * speed
            ratio = 1.0 + 0.002 * speed + 0.01 * axis_index
            condition_rows.append(
                {
                    "position": "0.23",
                    "quantity": "Mom.",
                    "axis": aggregate_axis,
                    "unit": "Nm",
                        "channel_baseline": (
                            f"{axis}~Mom.~BLD_1~pos~0.230~[Nm]"
                        ),
                        "channel_vorlap": (
                            f"{axis}~Mom.~BLD_1~pos~0.230~[Nm]"
                        ),
                    "wind_speed_mps": speed,
                    "seed_count": 3,
                    "fatigue_weight": next(
                        row["fatigue_weight"]
                        for row in manifest["rows"]
                        if row["dlc"] == "6.4"
                        and float(row["wind_speed_mps"]) == float(speed)
                    ),
                    "equivalent_cycles": config["fatigue"][
                        "equivalent_cycles"
                    ],
                    "signed_del_baseline_seed_aggregate": baseline,
                    "signed_del_vorlap_seed_aggregate": baseline * ratio,
                    "signed_del_baseline_seed_min": baseline * 0.98,
                    "signed_del_baseline_seed_max": baseline * 1.02,
                    "signed_del_vorlap_seed_min": baseline * ratio * 0.98,
                    "signed_del_vorlap_seed_max": baseline * ratio * 1.02,
                    "paired_ratio_seed_mean": ratio,
                    "paired_ratio_finite_seed_count": 3,
                    "paired_ratio_seed_min": ratio * 0.99,
                    "paired_ratio_seed_max": ratio * 1.01,
                    "paired_ratio_seed_sample_std": ratio * 0.01,
                    "ratio_vorlap_to_baseline": ratio,
                }
            )
        weighted_ratio = 1.05 + 0.01 * axis_index
        weighted_rows.append(
            {
                "position": "0.23",
                "quantity": "Mom.",
                "axis": aggregate_axis,
                "unit": "Nm",
                    "channel_baseline": (
                        f"{axis}~Mom.~BLD_1~pos~0.230~[Nm]"
                    ),
                    "channel_vorlap": (
                        f"{axis}~Mom.~BLD_1~pos~0.230~[Nm]"
                    ),
                "wind_speed_bin_count": 11,
                "weight_sum": 0.01,
                "wohler_exponent": 10.0,
                "equivalent_cycles": config["fatigue"]["equivalent_cycles"],
                "signed_del_baseline_weighted": 20.0 * axis_index,
                "signed_del_vorlap_weighted": 20.0
                * axis_index
                * weighted_ratio,
                "ratio_vorlap_to_baseline": weighted_ratio,
            }
        )

    extreme_rows = [
        {
            "row_id": manifest["rows"][-1]["row_id"],
            "dlc": "6.1",
            "seed": manifest["rows"][-1]["seed"],
            "scenario": "baseline",
            "wind_speed_mps": 50.0,
            "max_abs": 1.0,
        }
    ]
    ultimate_rows = []
    for quantity, axis, unit, values in (
        ("Mom.", "X_l", "Nm", (1.0, 1.2)),
        ("Force", "Y_l", "N", (2.0, 2.2)),
    ):
        for scenario, max_abs in zip(("baseline", "vorlap"), values):
            ultimate_rows.append(
                {
                    "scenario": scenario,
                    "blade": "BLD_1",
                    "position": "0.23",
                    "quantity": quantity,
                    "axis": axis,
                    "unit": unit,
                    "channel": f"{axis}-{scenario}",
                    "max_abs": max_abs,
                    "critical_row_id": manifest["rows"][-1]["row_id"],
                }
            )
    audit_rows = [
        {
            "row_id": row["row_id"],
            "dlc": row["dlc"],
            "seed": row["seed"],
            "wind_speed_mps": row["wind_speed_mps"],
            "result_audit_passed": True,
            "result_audit_sha256": "a" * 64,
            "warning_count": 0,
            "warning_codes": "",
            "warning_messages": "",
            "source_re_min": 10_000.0,
            "source_re_lower_bound_warning_samples": 0,
            "provenance_recovery_record_count": 0,
            "provenance_recovery_records": "",
        }
        for row in manifest["rows"]
    ]
    tables = {
        "signed_del_rows.csv": signed_rows,
        "fatigue_seed_aggregates.csv": condition_rows,
        "fatigue_weighted_signed_del.csv": weighted_rows,
        "extreme_channel_rows.csv": extreme_rows,
        "ultimate_envelope.csv": ultimate_rows,
        "campaign_audit_inventory.csv": audit_rows,
    }
    for name, rows in tables.items():
        module.write_csv(tmp_path / name, rows)
    summary = {
        "schema_version": 1,
        "passed": True,
        "manifest_sha256": manifest["manifest_sha256"],
        "row_count": manifest["row_count"],
        "equivalent_cycles": config["fatigue"]["equivalent_cycles"],
        "audit_warning_row_count": 0,
        "source_re_lower_bound_warning_samples": 0,
        "provenance_recovery_row_count": 0,
        "artifacts": {
            name: {"sha256": module.sha256_file(tmp_path / name)}
            for name in tables
        },
    }
    (tmp_path / "aggregation_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    aggregate = module.validate_production_aggregate(
        tmp_path,
        manifest,
        config,
    )
    figures = tmp_path / "figures"
    plot_data = tmp_path / "plot_data"
    result = module.production_fatigue_figures(
        aggregate,
        config,
        figures,
        plot_data,
    )
    result.update(
        module.production_ultimate_products(
            aggregate,
            tmp_path / "generated",
            plot_data,
        )
    )
    module.write_production_table(
        tmp_path / "generated" / "production_primary_fatigue_table.tex",
        aggregate,
        config,
    )

    assert result["primary_channel_count"] == 3
    assert result["u50_governing_ratio"] == pytest.approx(1.2)
    assert (figures / "fatigue_results.pdf").is_file()
    assert (figures / "fatigue_weighted_contribution.pdf").is_file()
    assert (plot_data / "fatigue_primary_channels.csv").is_file()
    assert (plot_data / "ultimate_baseline_vorlap.csv").is_file()
    assert (
        tmp_path / "generated" / "production_u50_extreme_table.tex"
    ).is_file()
    u50_table = (
        tmp_path / "generated" / "production_u50_extreme_table.tex"
    ).read_text()
    assert "Selection criterion" in u50_table
    assert "Largest absolute force" in u50_table
    assert "Largest moment ratio" in u50_table
    fatigue_table = (
        tmp_path / "generated" / "production_primary_fatigue_table.tex"
    ).read_text()
    assert "Baseline DEL [N\\,m]" in fatigue_table
    assert "VorLap DEL [N\\,m]" in fatigue_table
