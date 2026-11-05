from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.paper
if not (Path(__file__).resolve().parents[1] / "paper_drafts/NAWEA_QBladeBridge").is_dir():
    pytest.skip(
        "Separate SIL manuscript checkout is absent; see docs/reproducibility.md",
        allow_module_level=True,
    )

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "paper_drafts/NAWEA_QBladeBridge/analysis/prepare_unified_hvawt_paper_results.py"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vorlap-unified-paper-test-mpl")
SPEC = importlib.util.spec_from_file_location(
    "prepare_unified_hvawt_paper_results", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
promotion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promotion)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _case_value(wind: float, position: float, quantity: str, axis: str) -> float:
    axis_index = {"X": 1.0, "Y": 2.0, "Z": 3.0}[axis]
    quantity_scale = 10.0 if quantity == "moment" else 1.0
    return quantity_scale * (20.0 + wind) * (1.0 + position) * axis_index


def _write_fixture(root: Path) -> None:
    row_specs: list[tuple[str, float, int, bool, bool, float]] = []
    for speed in promotion.EXPECTED_SPEEDS:
        for seed in (4433456, 4433457, 4433458):
            speed_label = f"{speed:05.2f}".replace(".", "p")
            row_specs.append(
                (
                    f"dlc6p4_U{speed_label}ms_s{seed}",
                    speed,
                    seed,
                    True,
                    True,
                    0.001 + speed / 10_000.0,
                )
            )
    row_specs.append(("dlc6p1_U50p00ms_EWM50", 50.0, 0, False, True, 0.0))

    cases: list[dict[str, object]] = []
    row_metadata: dict[str, tuple[float, int, bool, bool, float]] = {}
    for row_id, wind, seed, fatigue, ultimate, weight in row_specs:
        row_metadata[row_id] = (wind, seed, fatigue, ultimate, weight)
        physics = hashlib.sha256(row_id.encode()).hexdigest()
        for scenario in promotion.SCENARIOS:
            cases.append(
                {
                    "case_id": f"{row_id}__{scenario}",
                    "row_id": row_id,
                    "scenario": scenario,
                    "wind_speed_mps": wind,
                    "seed": seed,
                    "dlc": "6.4" if fatigue else "6.1",
                    "fatigue_weight": weight,
                    "include_in_fatigue": fatigue,
                    "include_in_ultimate": ultimate,
                    "common_physics_fingerprint": physics,
                }
            )
    manifest: dict[str, object] = {
        "schema": promotion.CAMPAIGN_SCHEMA,
        "status": "prepared_not_run",
        "sealed_inputs": True,
        "case_count": 102,
        "row_count": 34,
        "scenarios": list(promotion.SCENARIOS),
        "matrix": {
            "duration_s": 660.0,
            "store_from_s": 60.0,
            "timestep_s": 0.0075,
            "total_steps": 88_000,
            "retained_rows": 80_000,
            "rpm": 1.0,
            "initial_azimuth_deg": 0.0,
            "wake_profile": "parked-reduced-v2",
        },
        "approved_manifest_sha256": promotion.APPROVED_MANIFEST_SHA256,
        "cases": cases,
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    (root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    analysis = {
        "schema": promotion.ANALYSIS_SCHEMA,
        "status": "complete_ready_for_load_postprocess",
        "campaign_fingerprint": manifest["fingerprint"],
        "expected_case_count": 102,
        "complete_case_count": 102,
        "all_three_scenarios_complete": True,
    }
    (root / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    full_extrema: list[dict[str, object]] = []
    signed_values: dict[tuple[str, float, float, str, str], dict[str, float]] = {}
    for row_id, (wind, _seed, _fatigue, _ultimate, _weight) in row_metadata.items():
        for position in promotion.EXPECTED_POSITIONS:
            for quantity, unit in (("force", "N"), ("moment", "Nm")):
                for axis in ("X", "Y", "Z"):
                    raw = _case_value(wind, position, quantity, axis)
                    axis_offset = {"X": 0.0, "Y": 0.01, "Z": 0.02}[axis]
                    values = {
                        promotion.SCENARIOS[0]: raw,
                        promotion.SCENARIOS[1]: raw * (1.10 + axis_offset),
                        promotion.SCENARIOS[2]: raw * (1.35 + 2.0 * axis_offset),
                    }
                    key = (row_id, wind, position, quantity, axis)
                    signed_values[key] = values
                    for scenario, value in values.items():
                        full_extrema.append(
                            {
                                "scenario": scenario,
                                "case": row_id,
                                "wind_speed_mps": wind,
                                "azimuth_deg": 0.0,
                                "blade": "BLD_1",
                                "position": position,
                                "quantity": quantity,
                                "axis": axis,
                                "unit": unit,
                                "channel": f"{quantity}_{axis}_{position}",
                                "min": -2.0 * value,
                                "max": 1.5 * value,
                                "max_abs": 2.0 * value,
                                "samples": 80_000,
                                "duration_s": 600.0,
                                "smoke_only": False,
                                "postprocess_warnings": "",
                                "result_file": f"/{scenario}/{row_id}.outq",
                            }
                        )

    for name, (baseline, coupled) in promotion.PAIR_SPECS.items():
        directory = root / f"postprocess_{name}"
        signed_rows: list[dict[str, object]] = []
        for (row_id, wind, position, quantity, axis), values in signed_values.items():
            base = values[baseline]
            comp = values[coupled]
            signed_rows.append(
                {
                    "case": row_id,
                    "wind_speed_mps": wind,
                    "azimuth_deg": 0.0,
                    "position": position,
                    "quantity": quantity,
                    "axis": axis,
                    "unit": "Nm" if quantity == "moment" else "N",
                    "channel_baseline": f"{quantity}_{axis}_{position}",
                    "channel_vorlap": f"{quantity}_{axis}_{position}",
                    "signed_del_baseline": base,
                    "signed_del_vorlap": comp,
                    "signed_delta_pct": 100.0 * (comp - base) / base,
                    "rainflow_cycles_baseline": 100.0,
                    "rainflow_cycles_vorlap": 100.0,
                    "equivalent_cycles_baseline": 600.0,
                    "equivalent_cycles_vorlap": 600.0,
                    "smoke_only": False,
                    "postprocess_warnings": "",
                }
            )
        signed_path = directory / "signed_del_comparison.csv"
        extreme_path = directory / "extreme_channel_metrics.csv"
        _write_csv(signed_path, signed_rows)
        _write_csv(extreme_path, full_extrema)
        inputs = {
            f"{scenario}:{row_id}": {
                "scenario": scenario,
                "case": row_id,
                "path": f"/{scenario}/{row_id}.outq",
                "bytes": 1,
                "sha256": "0" * 64,
            }
            for row_id in row_metadata
            for scenario in promotion.SCENARIOS
        }
        postprocess = {
            "schema_version": 1,
            "passed": True,
            "source_mode": "raw_qblade_exports",
            "scenarios": list(promotion.SCENARIOS),
            "comparison_baseline": baseline,
            "comparison_coupled": coupled,
            "wohler_exponent": 10.0,
            "equivalent_cycles": 600.0,
            "quality_thresholds": {
                "min_samples": 80_000,
                "min_duration_s": 600.0,
                "min_rainflow_cycles": 10.0,
            },
            "inputs": inputs,
            "outputs": {
                signed_path.name: _identity(signed_path),
                extreme_path.name: _identity(extreme_path),
            },
        }
        (directory / "postprocess_manifest.json").write_text(
            json.dumps(postprocess, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_promotes_only_complete_matched_three_path_evidence(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_fixture(campaign)
    output = tmp_path / "paper_export"

    result = promotion.promote(campaign, output)

    assert result["status"] == "promoted_from_102_complete_matched_cases"
    primary = list(
        csv.DictReader((output / "hvawt_three_path_primary_channels.csv").open())
    )
    weighted = list(
        csv.DictReader((output / "hvawt_three_path_weighted_del.csv").open())
    )
    u50 = list(csv.DictReader((output / "hvawt_three_path_u50_channels.csv").open()))
    assert len(primary) == 33
    assert len(weighted) == 3
    assert len(u50) == 12
    assert {row["axis"] for row in weighted} == {"X", "Y", "Z"}
    assert all(
        float(row["one_way_to_raw_ratio"])
        == pytest.approx(1.10 + 0.01 * {"X": 0, "Y": 1, "Z": 2}[row["axis"]])
        for row in weighted
    )
    assert all(
        float(row["complete_to_raw_ratio"])
        == pytest.approx(1.35 + 0.02 * {"X": 0, "Y": 1, "Z": 2}[row["axis"]])
        for row in weighted
    )
    assert (output / "hvawt_three_path_11min_results.pdf").stat().st_size > 10_000
    assert (
        "ThreePathXCompleteRawRatio"
        in (output / "hvawt_three_path_results_macros.tex").read_text()
    )
    table = (output / "hvawt_three_path_summary_table.tex").read_text()
    assert "tab:matchedthreepath" in table
    assert "ThreePathUltimateFiftyLargestComplete" in table
    assert r"\begin{tabularx}{\textwidth}" in table
    fragment = (output / "hvawt_matched_three_path_results.tex").read_text()
    assert "Matched three-path results" in fragment
    assert "ThreePathUltimateFiftyLargestComplete" in fragment
    assert "generated/hvawt_three_path_summary_table.tex" in fragment
    assert "not an H-VAWT experiment" in fragment
    assert fragment.index(r"\subsection{Matched three-path results}") < fragment.index(
        r"\begin{figure}[H]"
    )
    assert fragment.index(r"\begin{figure}[H]") < fragment.index(
        r"\input{generated/hvawt_three_path_summary_table.tex}"
    )

    receipt = json.loads(
        (output / "hvawt_three_path_promotion_receipt.json").read_text()
    )
    assert "latex_results_fragment" in receipt["outputs"]
    for record in receipt["outputs"].values():
        path = Path(record["path"])
        assert path.stat().st_size == record["bytes"]
        assert _sha(path) == record["sha256"]

    first_hashes = {
        path.name: _sha(path)
        for path in output.iterdir()
        if path.name != "hvawt_three_path_promotion_receipt.json"
    }
    promotion.promote(campaign, output)
    second_hashes = {
        path.name: _sha(path)
        for path in output.iterdir()
        if path.name != "hvawt_three_path_promotion_receipt.json"
    }
    assert first_hashes == second_hashes


def test_rejects_incomplete_campaign_without_writing_outputs(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_fixture(campaign)
    analysis_path = campaign / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["complete_case_count"] = 101
    analysis["all_three_scenarios_complete"] = False
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
    output = tmp_path / "paper_export"

    with pytest.raises(promotion.PromotionError, match="All 102"):
        promotion.promote(campaign, output)
    assert not output.exists()


def test_rejects_cross_postprocessor_numeric_mismatch(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    _write_fixture(campaign)
    directory = campaign / "postprocess_raw_vs_complete"
    signed_path = directory / "signed_del_comparison.csv"
    rows = list(csv.DictReader(signed_path.open()))
    rows[0]["signed_del_baseline"] = str(float(rows[0]["signed_del_baseline"]) + 1.0)
    _write_csv(signed_path, rows)
    manifest_path = directory / "postprocess_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][signed_path.name] = _identity(signed_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(promotion.PromotionError, match="Cross-postprocessor mismatch"):
        promotion.promote(campaign, tmp_path / "paper_export")
