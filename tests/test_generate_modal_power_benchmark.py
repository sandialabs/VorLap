from __future__ import annotations

import csv
import importlib.util
import json
import math
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


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "paper_drafts"
    / "NAWEA_QBladeBridge"
    / "figure_generation"
    / "generate_modal_power_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("generate_modal_power_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)

SYNTHETIC_FREQUENCY_HZ = 0.71325
SYNTHETIC_POWER_COEFFICIENT_WPM2 = 431.987654321
SYNTHETIC_MODEL_TO_PAIRED_AMPLITUDE = 0.991


def test_current_analyzer_schema_versions_are_required() -> None:
    assert generator.EXPECTED_COMPARISON_SCHEMA == (
        "grinderslev2022-modal-power-comparison-v3"
    )
    assert generator.EXPECTED_ANALYSIS_SCHEMA == (
        "grinderslev2022-modal-power-analysis-v3"
    )
    assert generator.EXPECTED_SETTLING_SCHEMA == (
        "grinderslev2022-modal-power-settling-qualification-v2"
    )
    assert generator.GATE_LABEL_FONTSIZE_PT * 0.98 >= 6.5
    assert generator.PANEL_B_TITLE == "(b) Calibration-input mean-power balance"


def _history(timestep_s: float, amplitude_m: float) -> dict[str, np.ndarray]:
    samples = int(round(12.0 / timestep_s)) + 1
    time = 48.0 + np.arange(samples, dtype=float) * timestep_s
    omega = 2.0 * math.pi * SYNTHETIC_FREQUENCY_HZ
    harmonic = np.sin(omega * time + 0.23)
    unloaded_tip = 0.02 * np.exp(-(time - time[0]) / 5.0)
    paired_tip = amplitude_m * harmonic
    response_tip = unloaded_tip + paired_tip
    unloaded_root = 2.0e4 * np.cos(0.2 * time)
    paired_root = 6.0e6 * harmonic
    response_root = unloaded_root + paired_root
    model_amplitude = SYNTHETIC_MODEL_TO_PAIRED_AMPLITUDE * amplitude_m
    target = np.full(
        samples,
        SYNTHETIC_POWER_COEFFICIENT_WPM2 * model_amplitude**2,
    )
    generated = target + 42.0 * np.cos(omega * time)
    applied = target + 38.0 * np.cos(omega * (time - timestep_s))
    mass_coordinate_work = 1.0004 * applied
    return {
        "time_s": time,
        "response_tip_edgewise_m": response_tip,
        "unloaded_tip_edgewise_m": unloaded_tip,
        "paired_tip_edgewise_m": paired_tip,
        "response_tip_envelope_m": np.full(samples, amplitude_m),
        "unloaded_tip_envelope_m": np.abs(unloaded_tip),
        "paired_tip_envelope_m": np.full(samples, amplitude_m),
        "response_root_edgewise_moment_nm": response_root,
        "unloaded_root_edgewise_moment_nm": unloaded_root,
        "paired_root_edgewise_moment_nm": paired_root,
        "target_cycle_mean_power_w": target,
        "response_model_amplitude_estimate_m": np.full(samples, model_amplitude),
        "generated_response_work_w": generated,
        "actually_applied_response_work_w": applied,
        "mass_coordinate_lagged_work_w": mass_coordinate_work,
        "stored_total_applied_work_w": applied,
        "inferred_base_applied_work_w": np.zeros(samples),
    }


def _write_history(path: Path, columns: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(generator.HISTORY_COLUMNS)
        for row in zip(*(columns[name] for name in generator.HISTORY_COLUMNS)):
            writer.writerow([f"{float(value):.17g}" for value in row])


def _settling_qualification(
    *,
    amplitude_m: float,
    model_amplitude_m: float,
    fail_envelope: bool,
) -> dict[str, object]:
    tolerances = {
        "paired_tip_amplitude_half_to_half": 0.01,
        "paired_tip_envelope_projected_change_over_10_cycles": 0.01,
        "paired_root_del_half_to_half": 0.01,
        "paired_root_pseudo_damage_A_m_half_to_half": 0.05,
        "lagged_applied_work_vs_mapped_structural_damping": 0.05,
        "lagged_applied_work_vs_target_work": 0.05,
        "model_amplitude_estimate_vs_paired_tip_amplitude": 0.05,
    }
    gates: dict[str, dict[str, object]] = {}
    for gate_name, _label, measure in generator.SETTLING_GATES:
        tolerance = tolerances[gate_name]
        difference = 0.25 * tolerance
        if fail_envelope and gate_name == (
            "paired_tip_envelope_projected_change_over_10_cycles"
        ):
            difference = 1.2 * tolerance
        passed = difference <= tolerance
        gates[gate_name] = {
            measure: difference,
            "tolerance": tolerance,
            "passed": passed,
        }
    damping_gate = gates["lagged_applied_work_vs_mapped_structural_damping"]
    damping_gate.update(
        {
            "paired_tip_fitted_amplitude_m": amplitude_m,
            "structural_power_coefficient_wpm2": (
                SYNTHETIC_POWER_COEFFICIENT_WPM2
            ),
            "mapped_structural_damping_power_w": (
                SYNTHETIC_POWER_COEFFICIENT_WPM2 * amplitude_m**2
            ),
        }
    )
    coordinate_gate = gates["model_amplitude_estimate_vs_paired_tip_amplitude"]
    coordinate_gate.update(
        {
            "mean_response_model_amplitude_estimate_m": model_amplitude_m,
            "paired_tip_fitted_amplitude_m": amplitude_m,
        }
    )
    return {
        "schema": generator.EXPECTED_SETTLING_SCHEMA,
        "frequency_hz": SYNTHETIC_FREQUENCY_HZ,
        "gates": gates,
        "passed": all(bool(row["passed"]) for row in gates.values()),
    }


def _analysis(
    timestep_s: float,
    amplitude_m: float,
    columns: dict[str, np.ndarray],
    *,
    fail_envelope: bool,
) -> dict[str, object]:
    mean_applied_work = float(np.mean(columns["actually_applied_response_work_w"]))
    mean_mass_coordinate_work = float(
        np.mean(columns["mass_coordinate_lagged_work_w"])
    )
    work_coordinate_difference = abs(
        mean_applied_work - mean_mass_coordinate_work
    ) / abs(mean_mass_coordinate_work)
    return {
        "schema": generator.EXPECTED_ANALYSIS_SCHEMA,
        "scope": "synthetic analyzer fixture",
        "case": {
            "timestep_s": timestep_s,
            "duration_s": 60.0,
            "steps": int(round(60.0 / timestep_s)),
            "steady_window_s": 12.0,
            "steady_samples": int(columns["time_s"].size),
            "natural_frequency_hz": SYNTHETIC_FREQUENCY_HZ,
            "force_scale": 0.0,
            "response_model_type": "modal_aerodynamic_power",
            "power_curve_sha256": generator.EXPECTED_POWER_CURVE_SHA256,
        },
        "response": {
            "tip_edgewise": {"fitted_amplitude": amplitude_m},
            "paired_tip_edgewise": {"fitted_amplitude": amplitude_m},
            "response_model_mean_amplitude_estimate_m": float(
                np.mean(columns["response_model_amplitude_estimate_m"])
            ),
        },
        "power": {
            "mean_target_cycle_power_w": float(
                np.mean(columns["target_cycle_mean_power_w"])
            ),
            "mean_generated_response_work_w": float(
                np.mean(columns["generated_response_work_w"])
            ),
            "mean_actually_applied_response_work_w": mean_applied_work,
            "mean_mass_coordinate_lagged_work_w": mean_mass_coordinate_work,
            "area_lift_vs_mass_coordinate_work_diagnostic": {
                "area_lift_mean_lagged_work_w": mean_applied_work,
                "mass_coordinate_mean_lagged_work_w": mean_mass_coordinate_work,
                "absolute_relative_difference": work_coordinate_difference,
                "tolerance": 0.001,
                "passed": True,
                "same_record_definition": (
                    "compare sum_i(F_i[n-1] dot v_i[n]) with "
                    "Q[n-1] qdot_raw[n] from the same replayed SIL history"
                ),
                "qualification_role": (
                    "cause-isolation diagnostic only; not a structural-damping "
                    "or response-amplitude acceptance gate"
                ),
            },
            "mean_structural_damping_power_at_measured_amplitude_w": (
                SYNTHETIC_POWER_COEFFICIENT_WPM2 * amplitude_m**2
            ),
            "positive_sign_convention": "force does work on current structural velocity",
        },
        "published_comparisons": {
            "published_fsi_amplitude_m": 1.18,
            "published_fsi_power_w": 775.0,
            "published_forced_motion_intersection_m": 1.24,
            "published_forced_motion_intersection_power_w": 880.0,
        },
        "limit_diagnostics": {
            "target_power_limited_steps": 0,
            "generalized_force_limited_steps": 0,
            "nodal_force_limited_steps": 0,
            "nodal_force_limited_node_samples": 0,
            "max_nodal_force_limited_nodes": 0,
        },
        "settling_qualification": _settling_qualification(
            amplitude_m=amplitude_m,
            model_amplitude_m=float(
                np.mean(columns["response_model_amplitude_estimate_m"])
            ),
            fail_envelope=fail_envelope,
        ),
        "provenance": {
            "response_run_summary_sha256": "1" * 64,
            "unloaded_run_summary_sha256": "2" * 64,
            "response_result_sha256": "3" * 64,
            "unloaded_result_sha256": "4" * 64,
            "sil_history_sha256": "5" * 64,
        },
    }


def _write_fixture(root: Path, *, failing_fine_gate: bool = True) -> Path:
    root.mkdir()
    coarse_history = _history(0.006, 1.250)
    fine_history = _history(0.003, 1.247)
    _write_history(root / "coarse_steady_histories.csv", coarse_history)
    _write_history(root / "fine_steady_histories.csv", fine_history)
    coarse = _analysis(
        0.006,
        1.250,
        coarse_history,
        fail_envelope=False,
    )
    fine = _analysis(
        0.003,
        1.247,
        fine_history,
        fail_envelope=failing_fine_gate,
    )
    timestep_gates: dict[str, dict[str, object]] = {}
    for gate_name, _label in generator.TIMESTEP_GATES:
        tolerance = 0.05 if "range_power" in gate_name else 0.01
        difference = 0.4 * tolerance
        timestep_gates[gate_name] = {
            "coarse": 1.0 + difference,
            "fine": 1.0,
            "tolerance": tolerance,
            "absolute_relative_difference": difference,
            "passed": True,
        }
    layers = {
        "coarse_run_settling": True,
        "fine_run_settling": not failing_fine_gate,
        "coarse_fine_timestep": True,
    }
    comparison = {
        "schema": generator.EXPECTED_COMPARISON_SCHEMA,
        "coarse": coarse,
        "fine": fine,
        "timestep_gate": {
            "schema": "grinderslev2022-modal-power-timestep-gate-v1",
            "gates": timestep_gates,
            "passed": True,
        },
        "overall_qualification": {
            "schema": "grinderslev2022-modal-power-overall-qualification-v1",
            "layers": layers,
            "passed": all(layers.values()),
        },
    }
    (root / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_generates_vector_figure_and_hash_bound_compact_plot_data(tmp_path: Path) -> None:
    analysis_dir = _write_fixture(tmp_path / "analysis")
    plot_data = tmp_path / "plot_data"
    figure = tmp_path / "figures" / "comparison.pdf"

    result = generator.generate(
        analysis_dir,
        plot_data_dir=plot_data,
        figure_path=figure,
        maximum_history_points=180,
    )

    assert result["overall_passed"] is False
    assert result["failed_gates"] == [
        "fine:paired_tip_envelope_projected_change_over_10_cycles"
    ]
    pdf = figure.read_bytes()
    assert pdf.startswith(b"%PDF")
    assert b"/Subtype /Image" not in pdf

    provenance_path = plot_data / "modal_power_benchmark_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema"] == "grinderslev2022-modal-power-paper-plot-v1"
    assert "not independent CFD" in provenance["scope"]
    assert provenance["source_inputs"]["comparison.json"]["sha256"] == (
        generator.sha256_file(analysis_dir / "comparison.json")
    )
    assert provenance["outputs"][figure.name]["sha256"] == generator.sha256_file(
        figure
    )
    assert provenance["qualification"]["gate_count"] == 18
    assert provenance["plot_settings"]["natural_frequency_hz"] == pytest.approx(
        SYNTHETIC_FREQUENCY_HZ
    )
    assert provenance["plot_settings"][
        "structural_power_coefficient_wpm2"
    ] == pytest.approx(SYNTHETIC_POWER_COEFFICIENT_WPM2)
    assert provenance["plot_settings"]["modal_property_schema_keys"][
        "structural_power_coefficient_wpm2"
    ].endswith(".structural_power_coefficient_wpm2")
    assert provenance["plot_settings"]["panel_b_bridge_amplitude_coordinate"] == (
        "mean response-model amplitude estimate A_hat"
    )
    assert provenance["plot_settings"]["panel_c_mapped_damping_definition"] == (
        "C * A_hat(t)^2 before smoothing"
    )
    assert "instead uses fitted paired-tip amplitude" in provenance[
        "caption_metadata"
    ]["panel_c"]
    cause_isolation = provenance["cause_isolation_diagnostic"]
    assert cause_isolation["steady_history_column"] == (
        "mass_coordinate_lagged_work_w"
    )
    assert cause_isolation["coarse"]["passed"] is True
    assert cause_isolation["fine"]["passed"] is True
    assert cause_isolation["maximum_absolute_relative_difference"] <= 0.001
    assert "cause isolation" in provenance["caption_metadata"]["panel_c"]

    terminal_path = plot_data / "modal_power_terminal_histories.csv"
    with terminal_path.open(encoding="utf-8", newline="") as handle:
        terminal_rows = list(csv.DictReader(handle))
    assert 300 <= len(terminal_rows) <= 360
    assert {row["resolution"] for row in terminal_rows} == {"coarse", "fine"}
    assert all(
        math.isfinite(float(row["mapped_structural_damping_power_w"]))
        for row in terminal_rows
    )
    assert all(
        math.isfinite(float(row["mass_coordinate_cycle_smoothed_work_w"]))
        for row in terminal_rows
    )
    assert all(
        float(row["mapped_structural_damping_power_w"])
        == pytest.approx(
            SYNTHETIC_POWER_COEFFICIENT_WPM2
            * float(row["response_model_amplitude_estimate_m"]) ** 2
        )
        for row in terminal_rows
    )

    gate_path = plot_data / "modal_power_qualification_gates.csv"
    with gate_path.open(encoding="utf-8", newline="") as handle:
        gate_rows = list(csv.DictReader(handle))
    assert len(gate_rows) == 18
    failed = [row for row in gate_rows if row["passed"] == "False"]
    assert len(failed) == 1
    assert float(failed[0]["normalized_ratio"]) == pytest.approx(1.2)

    power_path = plot_data / "modal_power_curve_and_markers.csv"
    with power_path.open(encoding="utf-8", newline="") as handle:
        power_rows = list(csv.DictReader(handle))
    series = {row["series"] for row in power_rows}
    assert {
        "digitized_forced_motion_calibration",
        "mapped_structural_damping",
        "published_fsi",
        "published_forced_motion_intersection",
        "coarse_qblade_measured",
        "fine_qblade_measured",
    }.issubset(series)
    unit_amplitude_damping = next(
        row
        for row in power_rows
        if row["series"] == "mapped_structural_damping"
        and float(row["amplitude_m"]) == pytest.approx(1.0)
    )
    assert float(unit_amplitude_damping["power_w"]) == pytest.approx(
        SYNTHETIC_POWER_COEFFICIENT_WPM2
    )
    coarse_bridge = next(
        row for row in power_rows if row["series"] == "coarse_qblade_measured"
    )
    assert float(coarse_bridge["amplitude_m"]) == pytest.approx(
        1.250 * SYNTHETIC_MODEL_TO_PAIRED_AMPLITUDE
    )
    assert float(coarse_bridge["amplitude_m"]) != pytest.approx(1.250)
    assert coarse_bridge["amplitude_coordinate"] == (
        "response-model mean amplitude A_hat"
    )

    repeat_plot_data = tmp_path / "repeat_plot_data"
    repeat_figure = tmp_path / "repeat_figures" / figure.name
    generator.generate(
        analysis_dir,
        plot_data_dir=repeat_plot_data,
        figure_path=repeat_figure,
        maximum_history_points=180,
    )
    assert repeat_figure.read_bytes() == figure.read_bytes()
    assert (
        repeat_plot_data / terminal_path.name
    ).read_bytes() == terminal_path.read_bytes()


def test_rejects_comparison_with_different_calibration_curve(tmp_path: Path) -> None:
    analysis_dir = _write_fixture(tmp_path / "analysis", failing_fine_gate=False)
    comparison_path = analysis_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["fine"]["case"]["power_curve_sha256"] = "0" * 64
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="does not use the committed power curve"):
        generator.generate(
            analysis_dir,
            plot_data_dir=tmp_path / "plot_data",
            figure_path=tmp_path / "figure.pdf",
        )


def test_rejects_gate_pass_flag_inconsistent_with_frozen_tolerance(tmp_path: Path) -> None:
    analysis_dir = _write_fixture(tmp_path / "analysis", failing_fine_gate=False)
    comparison_path = analysis_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["coarse"]["settling_qualification"]["gates"][
        "lagged_applied_work_vs_target_work"
    ]["passed"] = False
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="Stored pass/fail is inconsistent"):
        generator.generate(
            analysis_dir,
            plot_data_dir=tmp_path / "plot_data",
            figure_path=tmp_path / "figure.pdf",
        )


def test_rejects_coarse_fine_structural_power_coefficient_disagreement(
    tmp_path: Path,
) -> None:
    analysis_dir = _write_fixture(tmp_path / "analysis", failing_fine_gate=False)
    comparison_path = analysis_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    replacement = 447.25
    fine = comparison["fine"]
    damping_gate = fine["settling_qualification"]["gates"][
        "lagged_applied_work_vs_mapped_structural_damping"
    ]
    damping_gate["structural_power_coefficient_wpm2"] = replacement
    damping_gate["mapped_structural_damping_power_w"] = (
        replacement * damping_gate["paired_tip_fitted_amplitude_m"] ** 2
    )
    fine_amplitude = fine["response"]["tip_edgewise"]["fitted_amplitude"]
    fine["power"]["mean_structural_damping_power_at_measured_amplitude_w"] = (
        replacement * fine_amplitude**2
    )
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="coarse/fine structural power coefficient"):
        generator.generate(
            analysis_dir,
            plot_data_dir=tmp_path / "plot_data",
            figure_path=tmp_path / "figure.pdf",
        )


def test_rejects_coarse_fine_natural_frequency_disagreement(tmp_path: Path) -> None:
    analysis_dir = _write_fixture(tmp_path / "analysis", failing_fine_gate=False)
    comparison_path = analysis_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["fine"]["case"]["natural_frequency_hz"] = 0.72
    comparison["fine"]["settling_qualification"]["frequency_hz"] = 0.72
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="coarse/fine natural frequency"):
        generator.generate(
            analysis_dir,
            plot_data_dir=tmp_path / "plot_data",
            figure_path=tmp_path / "figure.pdf",
        )


def test_rejects_model_amplitude_gate_inconsistent_with_analyzer_summary(
    tmp_path: Path,
) -> None:
    analysis_dir = _write_fixture(tmp_path / "analysis", failing_fine_gate=False)
    comparison_path = analysis_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["coarse"]["settling_qualification"]["gates"][
        "model_amplitude_estimate_vs_paired_tip_amplitude"
    ]["mean_response_model_amplitude_estimate_m"] += 0.01
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="gated response-model amplitude"):
        generator.generate(
            analysis_dir,
            plot_data_dir=tmp_path / "plot_data",
            figure_path=tmp_path / "figure.pdf",
        )


def test_rejects_failed_mass_coordinate_cause_isolation_diagnostic(
    tmp_path: Path,
) -> None:
    analysis_dir = _write_fixture(tmp_path / "analysis", failing_fine_gate=False)
    comparison_path = analysis_dir / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    diagnostic = comparison["fine"]["power"][
        "area_lift_vs_mass_coordinate_work_diagnostic"
    ]
    diagnostic["passed"] = False
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="must pass at <=0.1%"):
        generator.generate(
            analysis_dir,
            plot_data_dir=tmp_path / "plot_data",
            figure_path=tmp_path / "figure.pdf",
        )
