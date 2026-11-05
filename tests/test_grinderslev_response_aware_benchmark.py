from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "examples"
    / "grinderslev2022_iea10"
    / "response_aware_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("grinderslev_response_benchmark", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_qblade_identified_quadrature_and_untuned_equilibrium_are_frozen() -> None:
    quadrature = benchmark.derive_blade_quadrature()

    assert len(quadrature.node_ids) == 30
    assert quadrature.node_ids[0] == "BLD_1_0.0000000000"
    assert quadrature.node_ids[-1] == "BLD_1_1.0000000000"
    assert quadrature.controller_node_ids[0] == "BLD_1_0.000000"
    assert quadrature.controller_node_ids[1] == "BLD_1_0.034483"
    assert quadrature.controller_node_ids[-1] == "BLD_1_1.000000"
    assert len(set(quadrature.controller_node_ids)) == 30
    assert np.all(quadrature.modal_mass_weights_kg > 0.0)
    assert np.all(quadrature.aerodynamic_area_weights_m2 > 0.0)
    assert np.allclose(
        quadrature.aerodynamic_area_weights_m2,
        quadrature.chord_m * quadrature.tributary_span_m,
        rtol=0.0,
        atol=0.0,
    )
    assert quadrature.public_openfast_modal_mass_kg == pytest.approx(
        1886.8146280823894, rel=1.0e-12
    )
    assert quadrature.modal_mass_kg == pytest.approx(1679.2069804136597, rel=1.0e-12)
    assert quadrature.natural_frequency_hz == pytest.approx(0.66985, abs=1.0e-15)
    assert quadrature.structural_damping_ratio == pytest.approx(
        0.0036131908955160446, abs=1.0e-14
    )
    assert quadrature.structural_power_coefficient_wpm2 == pytest.approx(
        452.342535828994,
        rel=1.0e-12,
    )
    assert quadrature.mode_identification.public_projection_overlap == pytest.approx(
        0.9430217402406977, rel=1.0e-13
    )
    assert not np.array_equal(
        quadrature.public_openfast_mode_shape,
        quadrature.response_qblade_mode_shape,
    )
    equilibrium = benchmark.structural_equilibrium_amplitude_m(
        quadrature.structural_power_coefficient_wpm2
    )
    assert equilibrium == pytest.approx(1.2720227178686943, rel=1.0e-12)


def test_mass_quadrature_accepts_roundoff_and_retains_pinned_coordinate(monkeypatch) -> None:
    original = benchmark.np.interp

    def roundoff(*args, **kwargs):
        density = original(*args, **kwargs)
        density[24] = np.nextafter(density[24], np.inf)
        return density

    monkeypatch.setattr(benchmark.np, "interp", roundoff)
    quadrature = benchmark.derive_blade_quadrature()
    assert np.array_equal(
        quadrature.modal_mass_weights_kg,
        quadrature.mode_identification.modal_mass_weights_kg,
    )


def test_mass_quadrature_still_rejects_changed_physical_input(monkeypatch) -> None:
    original = benchmark.np.interp

    def changed_mass(*args, **kwargs):
        density = original(*args, **kwargs)
        density[24] *= 1.0 + 1.0e-7
        return density

    monkeypatch.setattr(benchmark.np, "interp", changed_mass)
    with pytest.raises(ValueError, match="mass quadrature"):
        benchmark.derive_blade_quadrature()


def test_qblade_mode_identification_is_hash_bound_and_window_stable() -> None:
    identification = benchmark.load_qblade_mode_identification()
    metadata = identification.metadata

    assert benchmark.sha256_file(benchmark.QBLADE_MODE_CSV) == (
        benchmark.EXPECTED_QBLADE_MODE_CSV_SHA256
    )
    assert benchmark.sha256_file(benchmark.QBLADE_MODE_IDENTIFICATION) == (
        benchmark.EXPECTED_QBLADE_MODE_IDENTIFICATION_SHA256
    )
    assert metadata["scope"].endswith(
        "independent of the Grinderslev aerodynamic-power curve and 120 s response result"
    )
    windows = metadata["mode_window_sensitivity"]
    assert [row["start_s"] for row in windows] == [0.0, 2.0, 4.0, 6.0]
    overlaps = np.asarray([row["public_projection_overlap"] for row in windows])
    masses = np.asarray([row["modal_mass_kg"] for row in windows])
    assert float(np.ptp(overlaps)) < 0.005
    assert float(np.ptp(masses) / masses[0]) < 0.01
    assert max(row["max_abs_phase_offset_rad"] for row in windows) < 0.03
    assert metadata["frequency_window_sensitivity"][0]["frequency_hz"] == pytest.approx(
        0.66985, abs=1.0e-15
    )


def test_response_config_is_json_portable_and_fail_closed() -> None:
    quadrature = benchmark.derive_blade_quadrature()
    config = benchmark.build_response_model_config(quadrature)

    assert config["type"] == "modal_aerodynamic_power"
    assert config["schema_version"] == 1
    assert config["enabled"] is True
    assert config["node_ids"] == list(quadrature.controller_node_ids)
    assert config["force_direction"] == "chord"
    assert config["natural_frequency_hz"] == pytest.approx(0.66985)
    assert config["rms_time_constant_s"] == pytest.approx(1.0 / 0.66985)
    assert config["mode_shape"] == pytest.approx(
        quadrature.response_qblade_mode_shape.tolist()
    )
    assert config["aerodynamic_tributary_weights"] == pytest.approx(
        quadrature.aerodynamic_area_weights_m2.tolist(), rel=0.0, abs=0.0
    )
    assert not np.allclose(
        config["aerodynamic_tributary_weights"],
        config["modal_mass_weights_kg"],
    )
    assert config["limit_action"] == "error"
    assert config["max_abs_target_power_w"] == 2000.0
    assert config["max_abs_generalized_force_n"] == 10000.0
    assert config["max_abs_nodal_force_n"] == 2000.0
    assert config["power_curve"]["source_sha256"] == benchmark.EXPECTED_POWER_CURVE_SHA256
    assert config["power_curve"]["source_label"] == "p100_i50_forced_motion"
    json.dumps(config, allow_nan=False)


def test_mass_coordinate_lagged_work_uses_same_record_causality() -> None:
    generalized_force = np.asarray([2.0, -3.0, 5.0, 7.0])
    raw_qdot = np.asarray([11.0, 13.0, -17.0, 19.0])
    work = benchmark.lagged_mass_coordinate_work(generalized_force, raw_qdot)

    assert np.array_equal(work, np.asarray([0.0, 26.0, 51.0, 95.0]))
    with pytest.raises(ValueError, match="equal one-dimensional"):
        benchmark.lagged_mass_coordinate_work(
            generalized_force,
            raw_qdot[:-1],
        )
    nonfinite = raw_qdot.copy()
    nonfinite[2] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        benchmark.lagged_mass_coordinate_work(generalized_force, nonfinite)


def test_coarse_same_record_work_coordinate_reference_is_not_the_six_percent_gap() -> None:
    area_lift_work_w = 768.4665208672756
    mass_coordinate_work_w = 768.655849873873
    difference = benchmark._relative_difference(
        area_lift_work_w,
        mass_coordinate_work_w,
        "coarse same-record work coordinate",
    )

    assert difference == pytest.approx(0.00024631180082539, rel=1.0e-13)
    assert difference <= benchmark.WORK_COORDINATE_DIAGNOSTIC_TOLERANCE
    assert difference < 0.004 * 0.06389714816805808


def test_prepare_creates_independent_coarse_fine_response_and_baseline_cases(
    tmp_path: Path,
) -> None:
    quadrature = benchmark.derive_blade_quadrature()
    base_config = benchmark.BASE_CASE_ROOT / benchmark.CONFIG_RELATIVE
    base_sim = benchmark.BASE_CASE_ROOT / benchmark.BASE_SIM_NAME
    before_config = benchmark.sha256_file(base_config)
    before_sim = benchmark.sha256_file(base_sim)

    output = tmp_path / "prepared"
    plan_path = benchmark.prepare_response_cases(output, duration_s=0.018)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert benchmark.sha256_file(base_config) == before_config
    assert benchmark.sha256_file(base_sim) == before_sim
    assert [row["case_root"] for row in plan["cases"]] == [
        "dt_0p006/response",
        "dt_0p006/unloaded",
        "dt_0p003/response",
        "dt_0p003/unloaded",
    ]
    assert [row["requested_steps"] for row in plan["cases"]] == [3, 3, 6, 6]
    assert plan["predicted_forced_motion_curve_intersection_m"] == pytest.approx(
        1.2720227178686943
    )
    assert plan["natural_frequency_hz"] == pytest.approx(0.66985)
    assert plan["modal_mass_kg"] == pytest.approx(1679.2069804136597)
    assert plan["public_openfast_modal_mass_kg"] == pytest.approx(1886.8146280823894)

    for row in plan["cases"]:
        sim_path = output / row["sim"]
        config_path = output / row["config"]
        manifest_path = output / row["manifest"]
        assert sim_path.is_file() and config_path.is_file() and manifest_path.is_file()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert config["force_scale"] == 0.0
        assert config["source_frequency_cap_hz"] == pytest.approx(
            0.5 / row["timestep_s"]
        )
        assert benchmark._qblade_keyword_float(sim_path, "TIMESTEP") == row["timestep_s"]
        assert benchmark._qblade_keyword_float(sim_path, "NUMTIMESTEPS") == row[
            "requested_steps"
        ]
        assert manifest["response_aware_variant"]["role"] == row["role"]
        if row["role"] == "response":
            assert config["response_model"]["type"] == "modal_aerodynamic_power"
            assert config["response_model"]["mode_shape"] == pytest.approx(
                quadrature.response_qblade_mode_shape.tolist()
            )
            identity = config["response_model_node_identity"]
            assert identity["exact_swap_node_ids"] == config["node_ids"]
            assert identity["canonical_controller_node_ids"] == config[
                "response_model"
            ]["node_ids"]
            assert len(identity["exact_to_canonical"]) == 30
            assert identity["exact_to_canonical"][1] == {
                "exact_swap_node_id": "BLD_1_0.0344827586",
                "canonical_controller_node_id": "BLD_1_0.034483",
            }
            assert "--record-sil-vorlap-history" in row[
                "runner_args_relative_to_output_root"
            ]
        else:
            assert "response_model" not in config
            assert row["runner_args_relative_to_output_root"][-2:] == [
                "--coupling-mode",
                "none",
            ]

    from vorlap.qblade_runtime import VorLapQBladeRuntime

    response_row = plan["cases"][0]
    runtime = VorLapQBladeRuntime.from_qblade_config(
        str(output / response_row["config"])
    )
    assert runtime.controller.response_model_type == "modal_aerodynamic_power"
    assert runtime.controller.response_model_enabled is True
    assert runtime.controller.force_scale == 0.0
    assert runtime.controller.node_ids[1] == "BLD_1_0.034483"

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        benchmark.prepare_response_cases(output, duration_s=0.018)


def _write_result(path: Path, data: np.ndarray) -> None:
    lines = [
        "Results Output File created by test",
        "Simulation: test; Turbine: test",
        "\t".join(benchmark.REQUIRED_RESULT_COLUMNS),
    ]
    lines.extend("\t".join(f"{value:.17g}" for value in row) for row in data)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_strict_qblade_result_reader_rejects_malformed_numeric_history(
    tmp_path: Path,
) -> None:
    data = np.column_stack(
        (
            np.arange(4, dtype=float) * 0.006,
            np.asarray([0.0, 1.0, 0.0, -1.0]),
            np.asarray([0.0, 0.1, 0.0, -0.1]),
        )
    )
    path = tmp_path / "result.txt"
    _write_result(path, data)
    loaded = benchmark.read_qblade_result_strict(
        path,
        expected_steps=4,
        expected_timestep_s=0.006,
    )
    assert loaded.row_count == 4
    assert np.array_equal(loaded.time_s, data[:, 0])

    malformed = data.copy()
    malformed[2, 1] = np.nan
    _write_result(path, malformed)
    with pytest.raises(ValueError, match="must be finite"):
        benchmark.read_qblade_result_strict(
            path,
            expected_steps=4,
            expected_timestep_s=0.006,
        )


def test_time_grid_alignment_uses_numpy_rtol_keyword_and_rejects_offset() -> None:
    time = np.arange(8, dtype=float) * 0.006
    benchmark._validate_aligned_time_grids(
        time,
        time + 5.0e-13,
        label="synthetic",
        atol_s=1.0e-12,
    )
    with pytest.raises(ValueError, match="not aligned"):
        benchmark._validate_aligned_time_grids(
            time,
            time + 2.0e-12,
            label="synthetic",
            atol_s=1.0e-12,
        )


def _sil_rows(node_id: str) -> tuple[list[str], list[dict[str, object]]]:
    token = benchmark._safe_node_token(node_id)
    fixed = [
        "force_history_schema",
        "legacy_force_columns_semantics",
        "step_index",
        "time_s",
        "generated_step_index",
        "generated_time_s",
        "applied_source_step_index",
        "applied_source_time_s",
        "applied_startup_zero",
        "applied_force_lag_steps",
        "qblade_timestep",
        "azimuth_deg",
        "applied_force_dot_structural_velocity_w",
        "applied_power_sign_convention",
    ]
    vector_fields = []
    for axis in "xyz":
        vector_fields.extend(
            (
                f"vel_{token}_{axis}_mps",
                f"generated_force_{token}_{axis}_n",
                f"applied_force_{token}_{axis}_n",
                f"force_{token}_{axis}_n",
            )
        )
    rows = []
    generated_values = ((1.0, 2.0, 3.0), (2.0, 3.0, 4.0))
    velocity_values = ((4.0, 5.0, 6.0), (0.5, 0.25, 0.1))
    for step in range(2):
        applied_values = (0.0, 0.0, 0.0) if step == 0 else generated_values[0]
        row: dict[str, object] = {
            "force_history_schema": "vorlap-sil-force-history-v2",
            "legacy_force_columns_semantics": "generated_force_alias",
            "step_index": step,
            "time_s": step * 0.006,
            "generated_step_index": step,
            "generated_time_s": step * 0.006,
            "applied_source_step_index": "" if step == 0 else 0,
            "applied_source_time_s": "" if step == 0 else 0.0,
            "applied_startup_zero": step == 0,
            "applied_force_lag_steps": 1,
            "qblade_timestep": step,
            "azimuth_deg": 0.0,
            "applied_force_dot_structural_velocity_w": float(
                np.dot(applied_values, velocity_values[step])
            ),
            "applied_power_sign_convention": (
                "positive_when_applied_force_does_work_on_current_structural_velocity"
            ),
        }
        for axis_index, axis in enumerate("xyz"):
            row[f"vel_{token}_{axis}_mps"] = velocity_values[step][axis_index]
            row[f"generated_force_{token}_{axis}_n"] = generated_values[step][axis_index]
            row[f"applied_force_{token}_{axis}_n"] = applied_values[axis_index]
            row[f"force_{token}_{axis}_n"] = generated_values[step][axis_index]
        rows.append(row)
    return fixed + vector_fields, rows


def test_strict_sil_reader_checks_generated_applied_force_causality(tmp_path: Path) -> None:
    node_id = "BLD_1_0.0000000000"
    fields, rows = _sil_rows(node_id)
    path = tmp_path / "history.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    history = benchmark.read_sil_history_strict(
        path,
        node_ids=[node_id],
        expected_steps=2,
        expected_timestep_s=0.006,
    )
    assert history.applied_power_w.tolist() == pytest.approx([0.0, 1.3])
    assert np.array_equal(history.applied_force_n[1], history.generated_force_n[0])

    rows[1][f"applied_force_{benchmark._safe_node_token(node_id)}_x_n"] = 9.0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="one-step provenance"):
        benchmark.read_sil_history_strict(
            path,
            node_ids=[node_id],
            expected_steps=2,
            expected_timestep_s=0.006,
        )


def test_signal_and_load_metrics_recover_known_harmonic() -> None:
    timestep = 0.006
    time = np.arange(4000, dtype=float) * timestep
    amplitude = 1.25
    signal = 2.0 + 0.01 * time + amplitude * np.sin(
        2.0 * math.pi * benchmark.REFERENCE_FREQUENCY_HZ * time + 0.31
    )
    metrics, envelope = benchmark.sinusoidal_signal_metrics(time, signal)

    assert metrics["fitted_amplitude"] == pytest.approx(amplitude, rel=2.0e-12)
    assert metrics["fitted_linear_drift_per_s"] == pytest.approx(0.01, rel=1.0e-10)
    assert metrics["median_hilbert_envelope"] == pytest.approx(amplitude, rel=5.0e-3)
    assert envelope.shape == signal.shape
    fatigue = benchmark.load_channel_fatigue_metrics(
        1.0e6 * signal,
        timestep_s=timestep,
        wohler_exponent=10.0,
    )
    assert fatigue["rainflow_cycles"] > 10.0
    assert fatigue["damage_equivalent_range"] > 0.0
    assert fatigue["rainflow_range_power_sum"] > 0.0


def _stationary_settling_inputs() -> dict[str, object]:
    frequency = benchmark.REFERENCE_FREQUENCY_HZ
    samples_per_cycle = 250
    timestep = 1.0 / (frequency * samples_per_cycle)
    cycles = 20
    time = np.arange(cycles * samples_per_cycle, dtype=float) * timestep
    amplitude = 1.2
    phase = 0.31
    harmonic = np.sin(2.0 * math.pi * frequency * time + phase)
    coefficient = 500.0
    mapped_power = coefficient * amplitude**2
    return {
        "time": time,
        "tip": amplitude * harmonic,
        "root": 6.0e6 * harmonic,
        "model_amplitude": np.full(time.size, amplitude),
        "target_power": np.full(time.size, mapped_power),
        "applied_power": np.full(time.size, mapped_power),
        "timestep": timestep,
        "coefficient": coefficient,
        "exponent": 10.0,
        "amplitude": amplitude,
    }


def _evaluate_synthetic_settling(data: dict[str, object]) -> dict[str, object]:
    return benchmark.evaluate_settling_qualification(
        data["time"],
        data["tip"],
        data["root"],
        data["model_amplitude"],
        data["target_power"],
        data["applied_power"],
        timestep_s=data["timestep"],
        structural_power_coefficient_wpm2=data["coefficient"],
        wohler_exponent=data["exponent"],
    )


def test_frozen_settling_gates_accept_stationary_energy_closed_harmonic() -> None:
    data = _stationary_settling_inputs()
    result = _evaluate_synthetic_settling(data)

    assert benchmark.SETTLING_TIP_HALF_WINDOW_TOLERANCE == 0.01
    assert benchmark.SETTLING_ENVELOPE_10_CYCLE_TOLERANCE == 0.01
    assert benchmark.SETTLING_DEL_HALF_WINDOW_TOLERANCE == 0.01
    assert benchmark.SETTLING_RANGE_POWER_HALF_WINDOW_TOLERANCE == 0.05
    assert benchmark.SETTLING_POWER_CLOSURE_TOLERANCE == 0.05
    assert benchmark.SETTLING_AMPLITUDE_COORDINATE_TOLERANCE == 0.05
    assert benchmark.SETTLING_CYCLES_PER_HALF == 6
    assert benchmark.SETTLING_PHASE_GRID_INTERVALS_PER_CYCLE == 500
    assert result["passed"] is True
    assert all(row["passed"] for row in result["gates"].values())
    mapped_damping = result["gates"][
        "lagged_applied_work_vs_mapped_structural_damping"
    ]
    assert mapped_damping["closure_claim"] is False
    assert mapped_damping["direct_qblade_structural_dissipation_available"] is False
    assert mapped_damping["diagnostic_scope"].startswith(
        "nominal single-coordinate structural-model"
    )
    windows = result["cycle_synchronous_windows"]
    assert windows["total_target_cycles"] == 12
    assert windows["target_cycles_per_half"] == 6
    assert windows["phase_grid_samples_per_half"] == 3001
    assert windows["boundary_selection"] == (
        "time/frequency only; no signal extrema or rainflow cycles"
    )
    damage_gate = result["gates"]["paired_root_pseudo_damage_A_m_half_to_half"]
    assert damage_gate["first_half_A_m_per_target_cycle"] == pytest.approx(
        damage_gate["second_half_A_m_per_target_cycle"], rel=1.0e-10
    )


def test_cycle_synchronous_damage_gate_removes_arbitrary_endpoint_count_bias() -> None:
    timestep = 0.006
    frequency = 0.67
    time = np.arange(int(math.floor(20.0 / timestep)), dtype=float) * timestep
    phase = 0.3 * math.pi
    signal = 6.0e6 * np.sin(2.0 * math.pi * frequency * time + phase)
    half = time.size // 2
    raw_first = benchmark.load_channel_fatigue_metrics(
        signal[:half],
        timestep_s=timestep,
        wohler_exponent=10.0,
        frequency_hz=frequency,
    )
    raw_second = benchmark.load_channel_fatigue_metrics(
        signal[-half:],
        timestep_s=timestep,
        wohler_exponent=10.0,
        frequency_hz=frequency,
    )
    raw_difference = abs(
        raw_first["rainflow_range_power_sum"]
        - raw_second["rainflow_range_power_sum"]
    ) / raw_second["rainflow_range_power_sum"]
    assert raw_first["rainflow_cycles"] != raw_second["rainflow_cycles"]
    assert raw_difference > benchmark.SETTLING_RANGE_POWER_HALF_WINDOW_TOLERANCE

    coefficient = 500.0
    amplitude = 1.2
    harmonic = np.sin(2.0 * math.pi * frequency * time + phase)
    result = benchmark.evaluate_settling_qualification(
        time,
        amplitude * harmonic,
        signal,
        np.full(time.size, amplitude),
        np.full(time.size, coefficient * amplitude**2),
        np.full(time.size, coefficient * amplitude**2),
        timestep_s=timestep,
        structural_power_coefficient_wpm2=coefficient,
        wohler_exponent=10.0,
        frequency_hz=frequency,
    )
    gate = result["gates"]["paired_root_pseudo_damage_A_m_half_to_half"]
    assert gate["absolute_relative_difference"] < 1.0e-4
    assert gate["passed"] is True


def test_frozen_settling_gates_reject_growing_envelope_and_loads() -> None:
    data = _stationary_settling_inputs()
    time = np.asarray(data["time"])
    phase = 0.31
    growth = np.linspace(1.0, 1.3, time.size)
    harmonic = np.sin(
        2.0 * math.pi * benchmark.REFERENCE_FREQUENCY_HZ * time + phase
    )
    data["tip"] = growth * harmonic
    data["root"] = 5.0e6 * growth * harmonic
    data["model_amplitude"] = growth
    mean_amplitude = float(np.mean(growth))
    matched_power = float(data["coefficient"]) * mean_amplitude**2
    data["target_power"] = np.full(time.size, matched_power)
    data["applied_power"] = np.full(time.size, matched_power)

    result = _evaluate_synthetic_settling(data)
    gates = result["gates"]
    assert result["passed"] is False
    assert gates["paired_tip_amplitude_half_to_half"]["passed"] is False
    assert gates[
        "paired_tip_envelope_projected_change_over_10_cycles"
    ]["passed"] is False
    assert gates["paired_root_del_half_to_half"]["passed"] is False
    assert gates["paired_root_pseudo_damage_A_m_half_to_half"]["passed"] is False


def test_frozen_settling_gates_reject_power_and_amplitude_coordinate_mismatch() -> None:
    data = _stationary_settling_inputs()
    data["model_amplitude"] = 1.1 * np.asarray(data["model_amplitude"])
    data["applied_power"] = 1.1 * np.asarray(data["target_power"])

    result = _evaluate_synthetic_settling(data)
    gates = result["gates"]
    assert result["passed"] is False
    assert gates["lagged_applied_work_vs_mapped_structural_damping"]["passed"] is False
    assert gates["lagged_applied_work_vs_target_work"]["passed"] is False
    assert gates[
        "model_amplitude_estimate_vs_paired_tip_amplitude"
    ]["passed"] is False


def _minimal_analysis(amplitude: float, load_del: float, damage: float, power: float) -> dict:
    return {
        "case": {"timestep_s": 0.006, "duration_s": 60.0},
        "response": {"tip_edgewise": {"fitted_amplitude": amplitude}},
        "root_moment_fatigue": {
            "paired_response_minus_unloaded": {
                "damage_equivalent_range": load_del,
                "rainflow_range_power_sum": damage,
            }
        },
        "power": {"mean_actually_applied_response_work_w": power},
    }


def test_timestep_gate_applies_frozen_amplitude_del_damage_and_power_limits() -> None:
    coarse = _minimal_analysis(1.254, 2.01e6, 1.04e63, 803.0)
    fine = _minimal_analysis(1.25, 2.0e6, 1.0e63, 800.0)
    fine["case"]["timestep_s"] = 0.003
    gate = benchmark.compare_timestep_analyses(coarse, fine)
    assert gate["passed"] is True

    coarse["root_moment_fatigue"]["paired_response_minus_unloaded"][
        "rainflow_range_power_sum"
    ] = 1.051e63
    gate = benchmark.compare_timestep_analyses(coarse, fine)
    assert gate["passed"] is False
    assert gate["gates"]["paired_root_moment_range_power_sum"]["passed"] is False


def test_overall_qualification_requires_both_settled_runs_and_timestep_gate() -> None:
    coarse = {"settling_qualification": {"passed": True}}
    fine = {"settling_qualification": {"passed": True}}
    timestep_gate = {"passed": True}
    result = benchmark.combine_qualification_layers(coarse, fine, timestep_gate)
    assert result["passed"] is True
    assert all(result["layers"].values())

    fine["settling_qualification"]["passed"] = False
    result = benchmark.combine_qualification_layers(coarse, fine, timestep_gate)
    assert result["passed"] is False
    assert result["layers"] == {
        "coarse_run_settling": True,
        "fine_run_settling": False,
        "coarse_fine_timestep": True,
    }

    fine["settling_qualification"]["passed"] = True
    timestep_gate["passed"] = False
    result = benchmark.combine_qualification_layers(coarse, fine, timestep_gate)
    assert result["passed"] is False
    assert result["layers"]["coarse_fine_timestep"] is False


def test_steady_window_excludes_first_hundred_seconds_for_default_duration() -> None:
    for timestep in (0.006, 0.003):
        steps = int(round(120.0 / timestep))
        time = np.arange(steps, dtype=float) * timestep
        mask = benchmark._steady_mask(
            time,
            timestep_s=timestep,
            duration_s=120.0,
            steady_window_s=20.0,
        )
        assert time[mask][0] == pytest.approx(100.002 if timestep == 0.006 else 100.002)
        assert time[mask][-1] == pytest.approx(120.0 - timestep)
        assert not np.any(mask[time < 100.0])
    assert benchmark.DEFAULT_DURATION_S == 120.0
