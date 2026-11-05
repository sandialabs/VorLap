import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from vorlap.airfoil_io import load_airfoil_fft
from vorlap.qblade_controller import SwapBlockSpec
from vorlap.qblade_hvawt_response import (
    decompose_fixed_axis_velocity,
    load_tracked_hvawt_modal_basis,
    separate_hvawt_flows,
)
from vorlap.qblade_hvawt_sil import (
    HVAWT_SIL_ENGINEERING_MODEL_TYPE,
    HVAWT_SIL_ENGINEERING_SCHEMA_VERSION,
    HVAWT_SIL_GRIFFIN_MODEL_TYPE,
    HVAWT_SIL_GRIFFIN_SCHEMA_VERSION,
    HVAWT_SIL_HYBRID_MODEL_TYPE,
    HVAWT_SIL_RESPONSE_MODEL_TYPE,
    HVAWTSilResponseAdapter,
    calculate_hvawt_engineering_force,
    calculate_hvawt_griffin_force,
)
from vorlap.qblade_source_coverage import (
    HVAWT_V8_COMPONENT_NODE_COUNTS,
    SOURCE_COVERAGE_COLUMNS,
)
from vorlap.unified_viv_gain import UnifiedVIVGainModel

REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = REPO_ROOT / "run_headless" / "config"
LEGACY_MODEL_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "legacy_hvawt_models"
COEFFICIENT_FINGERPRINT = (
    "330242617d9d04d2cb771767fefbedac6414bea9a04a9b6ea9e813c0aba18c7d"
)


class FakeRuntime:
    time_block = "time"
    timestep_block = "timestep"
    azimuth_block = "azimuth_deg"
    velocity_block = "velocity"
    force_block = "force"

    def __init__(self, node_ids, *, hybrid=False, engineering=False):
        self.layout = {
            "time": SwapBlockSpec(0, ()),
            "timestep": SwapBlockSpec(1, ()),
            "azimuth_deg": SwapBlockSpec(2, ()),
            "velocity": SwapBlockSpec(3, (39, 3)),
            "force": SwapBlockSpec(120, (39, 3)),
            "lss_rpm": SwapBlockSpec(237, ()),
            "position_global_m": SwapBlockSpec(238, (39, 3)),
        }
        self.controller = SimpleNamespace(
            node_ids=tuple(node_ids),
            force_scale=0.0,
            include_mean_load=False,
            response_model_enabled=False,
            _base_chord=np.tile([1.0, 0.0, 0.0], (39, 1)),
            _base_normal=np.tile([0.0, 1.0, 0.0], (39, 1)),
            _chord=np.full(39, 0.5),
            _thickness=np.full(39, 0.18),
            _local_length=np.ones(39),
            fluid_density=1.225,
        )
        if hybrid or engineering:
            source_dir = (
                REPO_ROOT
                / "examples"
                / "ReferenceHVAWTQBladeModel"
                / "10m_HVAWT_reference_turbine"
                / "VorLapAirfoils"
                if engineering
                else REPO_ROOT / "data" / "airfoils"
            )
            if engineering:
                airfoil = load_airfoil_fft(str(source_dir / "NACA0018.h5"))
            else:
                with pytest.warns(RuntimeWarning):
                    airfoil = load_airfoil_fft(str(source_dir / "NACA0018.h5"))
            self.controller.airfoils = {
                "NACA0018": airfoil,
                "default": airfoil,
            }
            self.controller._node_airfoil_ids = ["NACA0018"] * 39
            self.controller._node_component_ids = [
                "_".join(node_id.split("_")[:3])
                if node_id.startswith("STR_")
                else "_".join(node_id.split("_")[:2])
                for node_id in node_ids
            ]
            self.controller.n_freq_depth = 200
            self.controller.fluid_dynamicviscosity = 1.81e-5
            self.controller.source_bounds_policy = "raise"
            self.controller.source_frequency_cap_hz = 50.0
            self.resolved_airfoil_dir = str(source_dir)
        self._ambient_velocity_global = np.array([7.0, 0.0, 0.0])


def _stage_evidence(tmp_path):
    control = tmp_path / "Control"
    control.mkdir()
    names = {
        "identification": "hvawt_response_mode_identification.json",
        "shapes": "hvawt_response_mode_shapes.csv",
        "duals": "hvawt_response_mode_force_duals.csv",
        "review_marker": "hvawt_modal_evidence_review_marker.json",
    }
    for name in names.values():
        shutil.copy2(EVIDENCE / name, control / name)
    shutil.copy2(
        EVIDENCE / "engineering_lockin_calibration.json",
        control / "engineering_lockin_calibration.json",
    )
    shutil.copy2(
        EVIDENCE / "griffin_lockin_calibration.json",
        control / "griffin_lockin_calibration.json",
    )
    shutil.copy2(
        EVIDENCE / "geometry_blended_airfoil_calibration.json",
        control / "geometry_blended_airfoil_calibration.json",
    )
    shutil.copy2(
        EVIDENCE / "unified_viv_gain_model_v1.json",
        control / "unified_viv_gain_model_v1.json",
    )
    parameter = control / "vorlap_qblade_external.json"
    parameter.write_text("{}\n", encoding="utf-8")
    return control, parameter, names


def _basis(control, names):
    return load_tracked_hvawt_modal_basis(
        identification_path=control / names["identification"],
        shapes_path=control / names["shapes"],
        duals_path=control / names["duals"],
        review_marker_path=control / names["review_marker"],
    )


def _raw_config(names, basis):
    modes = []
    for index, tracked_mode in enumerate((6, 12)):
        modes.append(
            {
                "tracked_mode": tracked_mode,
                "reference_length_m": 0.5,
                "reduced_velocity_low": 5.3625308212165494,
                "reduced_velocity_peak": 6.245825,
                "reduced_velocity_high": 6.982801851062782,
                "aerodynamic_damping_gain_h1": 3.920125762951649,
                "saturation_amplitude_ratio": 0.2968375879119047,
                "rms_time_constant_s": 2.0 / basis.natural_frequency_hz[index],
                "bandpass_damping_ratio": 0.1,
                "max_abs_gain": 10.0,
                "coefficient_source_sha256": COEFFICIENT_FINGERPRINT,
            }
        )
    return {
        "schema_version": 1,
        "type": HVAWT_SIL_RESPONSE_MODEL_TYPE,
        "enabled": True,
        "rpm_block": "lss_rpm",
        "position_block": "position_global_m",
        "modal_artifacts": names,
        "kinematics": {
            "axis_unit_global": [0.0, 0.0, 1.0],
            "axis_origin_global_m": [0.0, 0.0, 0.0],
            "body_to_global_at_reference": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "reference_azimuth_deg": 0.0,
            "maximum_abs_basis_rpm_error_rpm": 0.01,
            "maximum_azimuth_rate_error_radps": 1.0e-5,
            "rpm_sign": 1,
            "azimuth_sign": 1,
        },
        "modes": modes,
        "max_abs_total_nodal_force_n": 250.0,
        "force_sources": {
            "native_qblade_aerodynamics": "required_enabled",
            "external_viv_mode": "response_closure",
            "include_mean_load": False,
            "coefficient_role": "incremental_viv_screening",
        },
    }


def _raw_hybrid_config(names, basis, *, residual_source_scale=1.0, zero_gain=False):
    config = _raw_config(names, basis)
    config["schema_version"] = 2
    config["type"] = HVAWT_SIL_HYBRID_MODEL_TYPE
    damping = {6: 0.007448479057063753, 12: 0.009503978382703563}
    for row in config["modes"]:
        row["structural_damping_ratio"] = damping[row["tracked_mode"]]
        if zero_gain:
            row["aerodynamic_damping_gain_h1"] = 0.0
    config["spectral_lockin"] = {
        "schema_version": 1,
        "source_channel": "CL",
        "reference_strouhal": 0.2,
        "reference_lift_amplitude": 0.3,
        "frequency_ratio_low": 1.07250616424331,
        "frequency_ratio_peak": 1.249165,
        "frequency_ratio_high": 1.3965603702125565,
        "maximum_spectral_gain": 10.0,
        "minimum_modal_participation": 1.0e-12,
        "retain_uncaptured_lift_spectrum": True,
        "retain_fluctuating_drag_spectrum": True,
        "include_mean_load": False,
        "residual_source_scale": residual_source_scale,
    }
    return config


def _raw_engineering_config(names):
    config = json.loads(
        (LEGACY_MODEL_FIXTURES / "hvawt_response_model_v3.json").read_text(
            encoding="utf-8"
        )
    )
    config["modal_artifacts"] = names
    return config


def _raw_griffin_config(names):
    config = json.loads(
        (LEGACY_MODEL_FIXTURES / "hvawt_response_model_v4.json").read_text(
            encoding="utf-8"
        )
    )
    config["modal_artifacts"] = names
    return config


def _body_positions():
    angles = np.linspace(0.0, 2.0 * math.pi, 39, endpoint=False)
    return np.column_stack(
        (5.0 * np.cos(angles), 5.0 * np.sin(angles), np.linspace(0.0, 10.0, 39))
    )


def _rotation(angle_deg):
    angle = math.radians(angle_deg)
    return np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _swap_for_step(basis, *, time_s, qdot_mode6=0.0, qdot_mode12=0.0):
    swap = np.zeros(355)
    azimuth = 6.0 * time_s
    rotation = _rotation(azimuth)
    positions = _body_positions() @ rotation.T
    rigid = np.cross(
        np.tile([0.0, 0.0, 2.0 * math.pi / 60.0], (39, 1)),
        positions,
    )
    mode6_global = np.einsum(
        "ab,ib->ia",
        rotation,
        basis.real_shape_body_per_m[:, :, 0],
    )
    mode12_global = np.einsum(
        "ab,ib->ia",
        rotation,
        basis.real_shape_body_per_m[:, :, 1],
    )
    velocity = rigid + qdot_mode6 * mode6_global + qdot_mode12 * mode12_global
    swap[0] = time_s
    swap[1] = round(time_s / 0.01)
    swap[2] = azimuth
    swap[3:120] = velocity.reshape(-1)
    swap[237] = 1.0
    swap[238:355] = positions.reshape(-1)
    return swap


def test_hvawt_sil_adapter_rigid_spin_is_force_free_and_checkpointable(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_config(names, basis),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids),
    )

    force = adapter.step(_swap_for_step(basis, time_s=0.0))
    assert force.shape == (39, 3)
    assert np.max(np.abs(force)) == pytest.approx(0.0, abs=1.0e-12)
    checkpoint = adapter.checkpoint_state()
    restored = HVAWTSilResponseAdapter(
        raw_config=_raw_config(names, basis),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids),
    )
    restored.restore_checkpoint_state(json.loads(json.dumps(checkpoint)))
    assert restored.config_fingerprint == adapter.config_fingerprint


def test_hvawt_sil_adapter_accepts_explicit_qblade_startup_hold_time(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_config(names, basis),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids),
    )

    adapter.step(
        _swap_for_step(basis, time_s=0.01),
        response_time_s=0.0,
    )

    assert adapter.history_fields()["response_time_s"] == 0.0


def test_hvawt_sil_adapter_mode6_motion_produces_bounded_finite_force(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_config(names, basis),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids),
    )
    adapter.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.2))
    force = adapter.step(_swap_for_step(basis, time_s=0.01, qdot_mode6=0.25))

    assert np.all(np.isfinite(force))
    assert 0.0 < np.max(np.linalg.norm(force, axis=1)) < 250.0
    fields = adapter.history_fields()
    assert fields["response_force_source_mode"] == "response_closure"
    assert fields["response_mode6_activation_max"] > 0.0


def test_hvawt_sil_adapter_rejects_legacy_source_double_count(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    runtime = FakeRuntime(basis.node_ids)
    runtime.controller.force_scale = 1.0
    with pytest.raises(ValueError, match="force_scale=0"):
        HVAWTSilResponseAdapter(
            raw_config=_raw_config(names, basis),
            parameter_file=parameter,
            runtime=runtime,
        )


def test_hybrid_adapter_uses_real_airfoil_spectrum_and_reports_energy_partition(
    tmp_path,
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_hybrid_config(names, basis),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, hybrid=True),
    )

    first = adapter.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.1))
    second = adapter.step(_swap_for_step(basis, time_s=0.01, qdot_mode6=0.12))
    assert np.all(np.isfinite(first)) and np.all(np.isfinite(second))
    assert np.max(np.linalg.norm(second, axis=1)) > 0.0
    fields = adapter.history_fields()
    source = fields["response_spectral_source_energy_sum"]
    residual = fields["response_spectral_residual_energy_sum"]
    captured = fields["response_spectral_captured_energy_sum"]
    assert source == pytest.approx(residual + captured, rel=0.0, abs=2.0e-12)
    assert fields["response_mode6_spectral_gain_max"] >= 0.0
    assert fields["response_mode12_spectral_gain_max"] >= 0.0
    assert fields["response_mode6_pulled_frequency_hz"] == pytest.approx(
        basis.natural_frequency_hz[0]
    )
    assert fields["response_mode6_structural_damping_coefficient_nspm"] > 0.0


def test_hybrid_adapter_commits_exact_per_component_source_coverage(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_hybrid_config(names, basis),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, hybrid=True),
        sample_period_s=0.01,
    )
    adapter.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.1))
    adapter.commit_source_coverage(step_index=0)
    # QBlade wrapper updates 0 and 1 share response time zero.  They are still
    # distinct accepted samples in the terminal receipt.
    adapter.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.1))
    adapter.commit_source_coverage(step_index=1)

    rows = adapter.source_coverage()
    assert {row["component"] for row in rows} == set(
        HVAWT_V8_COMPONENT_NODE_COUNTS
    )
    for row in rows:
        assert tuple(row) == SOURCE_COVERAGE_COLUMNS
        assert row["samples"] == 2
        assert row["node_samples"] == (
            2 * HVAWT_V8_COMPONENT_NODE_COUNTS[row["component"]]
        )
    checkpoint = json.loads(json.dumps(adapter.checkpoint_state()))
    restored = HVAWTSilResponseAdapter(
        raw_config=_raw_hybrid_config(names, basis),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, hybrid=True),
        sample_period_s=0.01,
    )
    restored.restore_checkpoint_state(checkpoint)
    assert restored.source_coverage() == rows


def test_hybrid_native_control_is_exact_zero_with_same_spectral_observer(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_hybrid_config(
            names,
            basis,
            residual_source_scale=0.0,
            zero_gain=True,
        ),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, hybrid=True),
    )
    adapter.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.1))
    force = adapter.step(_swap_for_step(basis, time_s=0.01, qdot_mode6=0.12))
    np.testing.assert_allclose(force, 0.0, rtol=0.0, atol=0.0)
    fields = adapter.history_fields()
    assert fields["response_spectral_source_energy_sum"] > 0.0
    assert fields["response_spectral_residual_source_scale"] == 0.0


def test_hybrid_checkpoint_split_matches_uninterrupted_force(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    config = _raw_hybrid_config(names, basis)
    uninterrupted = HVAWTSilResponseAdapter(
        raw_config=config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, hybrid=True),
    )
    uninterrupted.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.1))
    checkpoint = uninterrupted.checkpoint_state()
    expected = uninterrupted.step(_swap_for_step(basis, time_s=0.01, qdot_mode6=0.12))

    restarted = HVAWTSilResponseAdapter(
        raw_config=config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, hybrid=True),
    )
    restarted.restore_checkpoint_state(json.loads(json.dumps(checkpoint)))
    observed = restarted.step(_swap_for_step(basis, time_s=0.01, qdot_mode6=0.12))
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_hybrid_damping_diagnostic_failure_rolls_back_source_and_core(
    tmp_path,
    monkeypatch,
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_hybrid_config(names, basis),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, hybrid=True),
    )
    before = json.dumps(adapter.checkpoint_state(), sort_keys=True)

    def fail_diagnostic(**_kwargs):
        raise RuntimeError("synthetic damping-balance failure")

    monkeypatch.setattr(
        "vorlap.qblade_hvawt_sil.evaluate_modal_damping_balance",
        fail_diagnostic,
    )
    with pytest.raises(RuntimeError, match="synthetic damping-balance failure"):
        adapter.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.1))

    assert json.dumps(adapter.checkpoint_state(), sort_keys=True) == before
    assert adapter.last_spectral_result is None
    assert adapter.last_result is None


def test_hybrid_contract_rejects_missing_damping_and_fractional_residual_scale(
    tmp_path,
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    config = _raw_hybrid_config(names, basis)
    del config["modes"][0]["structural_damping_ratio"]
    with pytest.raises(ValueError, match="keys differ"):
        HVAWTSilResponseAdapter(
            raw_config=config,
            parameter_file=parameter,
            runtime=FakeRuntime(basis.node_ids, hybrid=True),
        )

    config = _raw_hybrid_config(names, basis, residual_source_scale=0.5)
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        HVAWTSilResponseAdapter(
            raw_config=config,
            parameter_file=parameter,
            runtime=FakeRuntime(basis.node_ids, hybrid=True),
        )


def test_tracked_hvawt_response_and_campaign_specs_are_runnable_and_consistent(
    tmp_path,
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    response_config = json.loads(
        (EVIDENCE / "hvawt_response_model_v8_complete.json").read_text(encoding="utf-8")
    )
    adapter = HVAWTSilResponseAdapter(
        raw_config=response_config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    campaign = json.loads(
        (EVIDENCE / "hvawt_response_campaign_spec.json").read_text(encoding="utf-8")
    )

    assert adapter.config_fingerprint
    assert adapter.model_type == campaign["model"]["response_model_type"]
    assert campaign["model"]["response_config"].endswith(
        "hvawt_response_model_v8_complete.json"
    )
    assert campaign["pilot"]["expected_runs"] == (
        len(campaign["pilot"]["wind_speeds_mps"])
        * len(campaign["pilot"]["time_steps_s"])
        * len(campaign["pilot"]["scenarios"])
    )
    assert campaign["full_rerun"]["expected_new_runs_if_controls_are_rerun"] == (
        campaign["full_rerun"]["row_count"] * len(campaign["full_rerun"]["scenarios"])
    )
    assert (
        campaign["calibration_fingerprint_sha256"]
        == adapter.transfer.calibration_fingerprint
    )


@pytest.mark.parametrize("wind_speed_mps", [8.25, 12.2])
def test_unified_gain_path_is_exact_shadow_parity_at_planned_flight_winds(
    tmp_path, wind_speed_mps
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    unified_config = json.loads(
        (EVIDENCE / "hvawt_response_model_v8_complete.json").read_text(
            encoding="utf-8"
        )
    )
    unified_config["modal_artifacts"] = names
    legacy_config = json.loads(json.dumps(unified_config))
    legacy_config["engineering_lockin"]["schema_version"] = 1
    for key in (
        "unified_gain_model_path",
        "unified_gain_model_sha256",
        "unified_gain_branch",
    ):
        legacy_config["engineering_lockin"].pop(key)

    unified_runtime = FakeRuntime(basis.node_ids, engineering=True)
    legacy_runtime = FakeRuntime(basis.node_ids, engineering=True)
    unified_runtime._ambient_velocity_global = np.array([wind_speed_mps, 0.0, 0.0])
    legacy_runtime._ambient_velocity_global = np.array([wind_speed_mps, 0.0, 0.0])
    unified = HVAWTSilResponseAdapter(
        raw_config=unified_config,
        parameter_file=parameter,
        runtime=unified_runtime,
        sample_period_s=0.0075,
    )
    legacy = HVAWTSilResponseAdapter(
        raw_config=legacy_config,
        parameter_file=parameter,
        runtime=legacy_runtime,
        sample_period_s=0.0075,
    )
    assert unified.unified_gain_state is not None
    assert legacy.unified_gain_state is None

    for index in range(6):
        time_s = 0.0075 * index
        swap = _swap_for_step(
            basis,
            time_s=time_s,
            qdot_mode6=0.015 + 0.001 * index,
            qdot_mode12=-0.008 + 0.0005 * index,
        )
        unified_force = unified.step(swap)
        legacy_force = legacy.step(swap)
        np.testing.assert_array_equal(unified_force, legacy_force)
        assert unified.history_fields() == legacy.history_fields()

    unified_checkpoint = unified.checkpoint_state()
    legacy_checkpoint = legacy.checkpoint_state()
    unified_checkpoint.pop("config_fingerprint")
    legacy_checkpoint.pop("config_fingerprint")
    assert unified_checkpoint == legacy_checkpoint
    assert unified.last_total_maximum_nodal_force_n == (
        legacy.last_total_maximum_nodal_force_n
    )


@pytest.mark.parametrize("wind_speed_mps", [8.25, 12.2])
def test_quasi_static_fixed_azimuth_matches_one_rpm_instantaneous_state(
    tmp_path, wind_speed_mps
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    config = json.loads(
        (EVIDENCE / "hvawt_response_model_v8_complete.json").read_text(
            encoding="utf-8"
        )
    )
    config["modal_artifacts"] = names
    runtime = FakeRuntime(basis.node_ids, engineering=True)
    adapter = HVAWTSilResponseAdapter(
        raw_config=config,
        parameter_file=parameter,
        runtime=runtime,
        sample_period_s=0.0075,
    )
    fixed_gain = UnifiedVIVGainModel.from_json(
        control / "unified_viv_gain_model_v1.json"
    ).evaluate_airfoil(
        profile="NACA0018",
        kinematic_family="quasi_static_azimuth",
    )
    assert fixed_gain == adapter.unified_gain_state

    positions_body = _body_positions()
    positions_body[:, :2] *= 5.27 / 5.0
    omega = np.array([0.0, 0.0, 2.0 * math.pi / 60.0])
    ambient = np.array([wind_speed_mps, 0.0, 0.0])
    maximum_sensitivity_ratio = 0.0
    for azimuth_deg in np.arange(0.0, 360.0, 30.0):
        rotation = _rotation(float(azimuth_deg))
        positions = positions_body @ rotation.T
        prescribed_rigid = np.cross(
            np.broadcast_to(omega, positions.shape),
            positions,
        )
        instantaneous = decompose_fixed_axis_velocity(
            adapter.core.config.kinematics,
            positions_global_m=positions,
            total_structural_velocity_global_mps=prescribed_rigid,
            lss_rpm=1.0,
            azimuth_deg=float(azimuth_deg),
        )
        np.testing.assert_array_equal(
            instantaneous.rigid_body_velocity_global_mps,
            prescribed_rigid,
        )
        np.testing.assert_allclose(
            instantaneous.rotation_body_to_global,
            rotation,
            rtol=0.0,
            atol=2.0e-15,
        )
        chord = runtime.controller._base_chord @ rotation.T
        normal = runtime.controller._base_normal @ rotation.T
        flow = separate_hvawt_flows(
            ambient_velocity_global_mps=ambient,
            total_structural_velocity_global_mps=prescribed_rigid,
            rigid_body_velocity_global_mps=(
                instantaneous.rigid_body_velocity_global_mps
            ),
            chord_unit_global=chord,
            normal_unit_global=normal,
            chord_m=runtime.controller._chord,
            thickness_ratio=runtime.controller._thickness,
        )
        # This is the corresponding fixed-azimuth direct evaluation: geometry
        # is held at the sampled azimuth while the instantaneous 1-rpm rigid
        # velocity is prescribed explicitly.
        fixed_azimuth_response_velocity = ambient - prescribed_rigid
        np.testing.assert_array_equal(
            flow.response_relative_velocity_global_mps,
            fixed_azimuth_response_velocity,
        )
        sensitivity_ratio = (
            np.max(np.linalg.norm(prescribed_rigid, axis=1)) / wind_speed_mps
        )
        maximum_sensitivity_ratio = max(
            maximum_sensitivity_ratio,
            float(sensitivity_ratio),
        )
    # Preregistered compact quasi-static bound: omega*R/U < 0.07 at both
    # planned winds. This is a kinematic applicability check, not induction or
    # dynamic-stall validation.
    assert maximum_sensitivity_ratio < 0.07


def test_complete_model_uses_nodal_wind_for_spectrum_and_lockin(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    config = json.loads((EVIDENCE / "hvawt_response_model_v8_complete.json").read_text())
    config["modal_artifacts"] = names
    adapters = [HVAWTSilResponseAdapter(
        raw_config=config, parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    ) for _ in range(2)]
    mean, local = adapters
    uniform = np.tile([7.0, 0.0, 0.0], (39, 1))
    swap = _swap_for_step(basis, time_s=0.0, qdot_mode6=0.02)
    np.testing.assert_allclose(
        local.step(swap, ambient_velocity_global_mps=uniform), mean.step(swap),
        rtol=0.0, atol=0.0,
    )
    swap = _swap_for_step(basis, time_s=0.01, qdot_mode6=0.03)
    wind = uniform.copy()
    wind[:, 0] += np.linspace(-0.6, 0.6, 39)
    wind[:, 1] = np.linspace(0.1, 0.4, 39)
    wind[:, 2] = 0.2
    force = local.step(swap, ambient_velocity_global_mps=wind)
    mean.step(swap)

    positions = swap[238:355].reshape(39, 3)
    rigid = np.cross([0.0, 0.0, 2.0 * math.pi / 60.0], positions)
    expected_query_velocity = wind - rigid
    rotation = _rotation(swap[2])
    chord = local.runtime.controller._base_chord @ rotation.T
    normal = local.runtime.controller._base_normal @ rotation.T
    u_chord = np.sum(expected_query_velocity * chord, axis=1)
    u_normal = np.sum(expected_query_velocity * normal, axis=1)
    expected_speed = np.hypot(u_chord, u_normal)
    spectral = local.last_spectral_result
    np.testing.assert_allclose(local.last_result.response_relative_velocity_global_mps,
                               expected_query_velocity, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(spectral.base_effective_crossflow_speed_mps, expected_speed)
    np.testing.assert_allclose(spectral.base_aoa_deg, np.degrees(np.arctan2(u_normal, u_chord)))
    np.testing.assert_allclose(
        spectral.base_reynolds,
        1.225 * expected_speed * 0.5 / local.spectral_source.fluid_dynamic_viscosity_pas,
    )
    assert np.isfinite(force).all()
    assert not np.allclose(
        [row.zero_amplitude_feedback_coefficient_nspm for row in local.last_engineering_diagnostics.mode_diagnostics],
        [row.zero_amplitude_feedback_coefficient_nspm for row in mean.last_engineering_diagnostics.mode_diagnostics],
    )
    assert abs(local.last_engineering_diagnostics.power_closure_error_w) < 1e-12


def test_complete_composition_adds_only_partitioned_residual_to_v5_lockin_force(
    tmp_path,
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    isolation_config = json.loads(
        (LEGACY_MODEL_FIXTURES / "hvawt_response_model_v5.json").read_text(
            encoding="utf-8"
        )
    )
    complete_config = json.loads(
        (EVIDENCE / "hvawt_response_model_v8_complete.json").read_text(encoding="utf-8")
    )
    isolation_config["modal_artifacts"] = names
    complete_config["modal_artifacts"] = names
    isolation = HVAWTSilResponseAdapter(
        raw_config=isolation_config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    complete = HVAWTSilResponseAdapter(
        raw_config=complete_config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )

    for time_s, qdot6, qdot12 in ((0.0, 0.08, -0.04), (0.01, 0.11, -0.06)):
        swap = _swap_for_step(
            basis,
            time_s=time_s,
            qdot_mode6=qdot6,
            qdot_mode12=qdot12,
        )
        isolation_force = isolation.step(swap)
        complete_force = complete.step(swap)

    residual = np.asarray(
        complete.last_spectral_result.residual_force_global_n,
        dtype=float,
    )
    np.testing.assert_allclose(
        complete_force - isolation_force,
        residual,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    fields = complete.history_fields()
    assert fields["response_spectral_residual_source_scale"] == 1.0
    assert fields["response_spectral_applied_residual_maximum_nodal_force_n"] > 0.0
    assert fields["response_total_maximum_nodal_force_n"] == pytest.approx(
        np.max(np.linalg.norm(complete_force, axis=1))
    )
    assert fields["response_spectral_residual_energy_sum"] > 0.0
    assert fields["response_spectral_source_energy_sum"] == pytest.approx(
        fields["response_spectral_residual_energy_sum"]
        + fields["response_spectral_captured_energy_sum"],
        rel=0.0,
        abs=2.0e-12,
    )


def test_total_force_limit_includes_residual_when_engineering_response_is_zero(
    tmp_path,
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    config = json.loads(
        (EVIDENCE / "hvawt_response_model_v8_complete.json").read_text(
            encoding="utf-8"
        )
    )
    config["modal_artifacts"] = names
    config["engineering_lockin"]["response_force_scale"] = 0.0
    assert config["max_abs_total_nodal_force_n"] is None
    unrestricted = HVAWTSilResponseAdapter(
        raw_config=config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    swaps = [
        _swap_for_step(
            basis,
            time_s=time_s,
            qdot_mode6=0.0,
            qdot_mode12=0.0,
        )
        for time_s in (0.0, 0.01)
    ]
    forces = [unrestricted.step(swap) for swap in swaps]
    maximum_total = max(float(np.max(np.linalg.norm(force, axis=1))) for force in forces)
    assert maximum_total > 0.0
    assert unrestricted.history_fields()[
        "response_engineering_maximum_abs_nodal_force_n"
    ] == pytest.approx(0.0)

    limited_config = json.loads(json.dumps(config))
    limited_config["max_abs_total_nodal_force_n"] = 0.5 * maximum_total
    limited = HVAWTSilResponseAdapter(
        raw_config=limited_config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    with pytest.raises(ValueError, match="response-plus-companion nodal force"):
        for swap in swaps:
            limited.step(swap)


def test_no_force_ceiling_is_exactly_equivalent_below_a_finite_guard(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    no_ceiling_config = json.loads(
        (EVIDENCE / "hvawt_response_model_v8_complete.json").read_text(
            encoding="utf-8"
        )
    )
    no_ceiling_config["modal_artifacts"] = names
    assert no_ceiling_config["max_abs_total_nodal_force_n"] is None
    guarded_config = json.loads(json.dumps(no_ceiling_config))
    guarded_config["max_abs_total_nodal_force_n"] = 1.0e9
    no_ceiling = HVAWTSilResponseAdapter(
        raw_config=no_ceiling_config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    guarded = HVAWTSilResponseAdapter(
        raw_config=guarded_config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )

    for time_s, qdot6, qdot12 in (
        (0.0, 0.0, 0.0),
        (0.01, 0.02, -0.01),
        (0.02, -0.01, 0.015),
    ):
        swap = _swap_for_step(
            basis,
            time_s=time_s,
            qdot_mode6=qdot6,
            qdot_mode12=qdot12,
        )
        np.testing.assert_array_equal(no_ceiling.step(swap), guarded.step(swap))

    no_ceiling_history = no_ceiling.history_fields()
    guarded_history = guarded.history_fields()
    assert no_ceiling_history.keys() == guarded_history.keys()
    for key in no_ceiling_history:
        if key == "response_config_fingerprint":
            assert no_ceiling_history[key] != guarded_history[key]
            continue
        left = no_ceiling_history[key]
        right = guarded_history[key]
        if isinstance(left, float):
            assert left == right, key
        else:
            assert left == right, key


def test_tracked_hybrid_model_loads_real_naca_spectra_and_preserves_v1(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    hybrid_config = json.loads(
        (LEGACY_MODEL_FIXTURES / "hvawt_response_model_v2.json").read_text(
            encoding="utf-8"
        )
    )
    hybrid = HVAWTSilResponseAdapter(
        raw_config=hybrid_config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, hybrid=True),
    )
    legacy_config = json.loads(
        (LEGACY_MODEL_FIXTURES / "hvawt_response_model_v1.json").read_text(
            encoding="utf-8"
        )
    )
    legacy = HVAWTSilResponseAdapter(
        raw_config=legacy_config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids),
    )

    assert hybrid.model_type == HVAWT_SIL_HYBRID_MODEL_TYPE
    assert hybrid.schema_version == 2
    assert hybrid.spectral_source is not None
    assert hybrid.config_fingerprint != legacy.config_fingerprint
    assert legacy.model_type == HVAWT_SIL_RESPONSE_MODEL_TYPE
    assert legacy.schema_version == 1
    assert legacy.spectral_source is None


def test_engineering_adapter_uses_transfer_calibration_and_exact_two_mode_work(
    tmp_path,
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_engineering_config(names),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )

    adapter.step(
        _swap_for_step(
            basis,
            time_s=0.0,
            qdot_mode6=0.08,
            qdot_mode12=-0.05,
        )
    )
    force = adapter.step(
        _swap_for_step(
            basis,
            time_s=0.01,
            qdot_mode6=0.12,
            qdot_mode12=-0.07,
        )
    )

    assert adapter.model_type == HVAWT_SIL_ENGINEERING_MODEL_TYPE
    assert adapter.schema_version == HVAWT_SIL_ENGINEERING_SCHEMA_VERSION
    assert np.all(np.isfinite(force))
    assert np.max(np.linalg.norm(force, axis=1)) > 0.0
    fields = adapter.history_fields()
    assert fields["response_spectral_residual_source_scale"] == 0.0
    assert fields["response_engineering_facility_velocity_scale_applied"] == 1.0
    assert (
        fields["response_engineering_velocity_basis"]
        == "local_rigid_body_relative_unscaled"
    )
    assert fields["response_engineering_power_closure_error_w"] < 1.0e-10
    assert fields[
        "response_mode6_engineering_structural_damping_coefficient_nspm"
    ] == pytest.approx(
        79.4222945240581,
        rel=2.0e-9,
    )
    assert fields[
        "response_mode12_engineering_structural_damping_coefficient_nspm"
    ] == pytest.approx(
        67.65400695103965,
        rel=2.0e-9,
    )


def test_engineering_checkpoint_split_matches_simultaneous_two_mode_force(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    config = _raw_engineering_config(names)
    uninterrupted = HVAWTSilResponseAdapter(
        raw_config=config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    uninterrupted.step(
        _swap_for_step(basis, time_s=0.0, qdot_mode6=0.08, qdot_mode12=-0.04)
    )
    checkpoint = uninterrupted.checkpoint_state()
    expected = uninterrupted.step(
        _swap_for_step(basis, time_s=0.01, qdot_mode6=0.11, qdot_mode12=-0.06)
    )

    restarted = HVAWTSilResponseAdapter(
        raw_config=config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    restarted.restore_checkpoint_state(json.loads(json.dumps(checkpoint)))
    observed = restarted.step(
        _swap_for_step(basis, time_s=0.01, qdot_mode6=0.11, qdot_mode12=-0.06)
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    assert (
        restarted.history_fields()["response_engineering_power_closure_error_w"]
        < 1.0e-10
    )


def test_engineering_native_control_keeps_observer_but_returns_exact_zero(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    config = _raw_engineering_config(names)
    config["engineering_lockin"]["response_force_scale"] = 0.0
    adapter = HVAWTSilResponseAdapter(
        raw_config=config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    adapter.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.08, qdot_mode12=-0.04))
    force = adapter.step(
        _swap_for_step(basis, time_s=0.01, qdot_mode6=0.11, qdot_mode12=-0.06)
    )

    np.testing.assert_allclose(force, 0.0, rtol=0.0, atol=0.0)
    fields = adapter.history_fields()
    assert fields["response_spectral_source_energy_sum"] > 0.0
    assert (
        fields["response_mode6_engineering_zero_amplitude_feedback_coefficient_nspm"]
        > 0.0
    )
    assert fields["response_mode6_engineering_generalized_force_n"] == 0.0
    assert fields["response_engineering_maximum_abs_nodal_force_n"] == 0.0


def test_reciprocal_griffin_force_has_exact_two_mode_work_and_restoring_branch(
    tmp_path,
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_griffin_config(names),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    count = len(basis.node_ids)
    captured = np.full((count, 2), 0.2)
    spectral = SimpleNamespace(
        partition=SimpleNamespace(captured_amplitude_by_node_mode=captured)
    )
    response_velocity = np.tile([7.0, 0.0, 0.0], (count, 1))
    chord = np.tile([1.0, 0.0, 0.0], (count, 1))
    normal = np.tile([0.0, 1.0, 0.0], (count, 1))
    raw = np.array([0.02, -0.015])
    flexible_velocity = np.einsum("iam,m->ia", basis.real_shape_body_per_m, raw)

    force, diagnostics = calculate_hvawt_griffin_force(
        modal_basis=basis,
        parameters=adapter.griffin_parameters,
        calibration_fingerprint=adapter.griffin_calibration_fingerprint,
        static_spectrum_sha256=adapter.griffin_static_spectrum_sha256,
        spectral_result=spectral,
        mode_configs=adapter.core.config.modes,
        structural_damping_ratios=adapter.structural_damping_ratios,
        rotation_body_to_global=np.eye(3),
        response_relative_velocity_global_mps=response_velocity,
        response_effective_crossflow_speed_mps=np.full(count, 7.0),
        chord_unit_global=chord,
        normal_unit_global=normal,
        chord_m=np.full(count, 0.5),
        local_length_m=np.ones(count),
        fluid_density_kgpm3=1.225,
        flexible_velocity_global_mps=flexible_velocity,
        raw_qdot_mps=raw,
        filtered_qdot_mps=raw,
        maximum_power_closure_relative_error=1.0e-10,
    )

    assert diagnostics.power_closure_error_w < 1.0e-12
    assert float(np.sum(force * flexible_velocity)) == pytest.approx(
        diagnostics.generalized_power_w, abs=1.0e-12
    )
    assert all(
        row.predicted_physical_peak_amplitude_m > 0.0
        for row in diagnostics.mode_diagnostics
    )

    fast = np.array([10.0, -10.0])
    _force, restoring = calculate_hvawt_griffin_force(
        modal_basis=basis,
        parameters=adapter.griffin_parameters,
        calibration_fingerprint=adapter.griffin_calibration_fingerprint,
        static_spectrum_sha256=adapter.griffin_static_spectrum_sha256,
        spectral_result=spectral,
        mode_configs=adapter.core.config.modes,
        structural_damping_ratios=adapter.structural_damping_ratios,
        rotation_body_to_global=np.eye(3),
        response_relative_velocity_global_mps=response_velocity,
        response_effective_crossflow_speed_mps=np.full(count, 7.0),
        chord_unit_global=chord,
        normal_unit_global=normal,
        chord_m=np.full(count, 0.5),
        local_length_m=np.ones(count),
        fluid_density_kgpm3=1.225,
        flexible_velocity_global_mps=np.einsum(
            "iam,m->ia", basis.real_shape_body_per_m, fast
        ),
        raw_qdot_mps=fast,
        filtered_qdot_mps=fast,
        maximum_power_closure_relative_error=1.0e-10,
    )
    assert restoring.generalized_power_w < 0.0


def test_reciprocal_griffin_adapter_is_checkpointable_and_native_control_is_zero(
    tmp_path,
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    config = _raw_griffin_config(names)
    active = HVAWTSilResponseAdapter(
        raw_config=config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    active.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.02))
    checkpoint = active.checkpoint_state()
    expected = active.step(
        _swap_for_step(basis, time_s=0.01, qdot_mode6=0.03, qdot_mode12=-0.02)
    )
    restarted = HVAWTSilResponseAdapter(
        raw_config=config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    restarted.restore_checkpoint_state(json.loads(json.dumps(checkpoint)))
    observed = restarted.step(
        _swap_for_step(basis, time_s=0.01, qdot_mode6=0.03, qdot_mode12=-0.02)
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    assert active.model_type == HVAWT_SIL_GRIFFIN_MODEL_TYPE
    assert active.schema_version == HVAWT_SIL_GRIFFIN_SCHEMA_VERSION
    fields = active.history_fields()
    assert fields["response_griffin_power_closure_error_w"] < 1.0e-10
    assert fields["response_griffin_closure_name"].endswith(
        "reciprocal-Griffin work closure"
    )

    control_config = _raw_griffin_config(names)
    control_config["griffin_lockin"]["response_force_scale"] = 0.0
    control_adapter = HVAWTSilResponseAdapter(
        raw_config=control_config,
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    control_adapter.step(_swap_for_step(basis, time_s=0.0, qdot_mode6=0.02))
    force = control_adapter.step(_swap_for_step(basis, time_s=0.01, qdot_mode6=0.03))
    np.testing.assert_allclose(force, 0.0, rtol=0.0, atol=0.0)
    assert (
        control_adapter.history_fields()["response_mode6_griffin_generalized_force_n"]
        == 0.0
    )


def test_engineering_force_failure_rolls_back_core_and_spectral_state(
    tmp_path, monkeypatch
):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    adapter = HVAWTSilResponseAdapter(
        raw_config=_raw_engineering_config(names),
        parameter_file=parameter,
        runtime=FakeRuntime(basis.node_ids, engineering=True),
    )
    before = json.dumps(adapter.checkpoint_state(), sort_keys=True)

    def fail_engineering(**_kwargs):
        raise RuntimeError("synthetic engineering-force failure")

    monkeypatch.setattr(
        "vorlap.qblade_hvawt_sil.calculate_hvawt_engineering_force",
        fail_engineering,
    )
    with pytest.raises(RuntimeError, match="synthetic engineering-force failure"):
        adapter.step(
            _swap_for_step(basis, time_s=0.0, qdot_mode6=0.1, qdot_mode12=0.02)
        )

    assert json.dumps(adapter.checkpoint_state(), sort_keys=True) == before
    assert adapter.last_result is None
    assert adapter.last_spectral_result is None
    assert adapter.last_engineering_diagnostics is None


def test_engineering_adapter_rejects_noncalibrated_runtime_hdf5(tmp_path):
    control, parameter, names = _stage_evidence(tmp_path)
    basis = _basis(control, names)
    with pytest.raises(ValueError, match="HDF5 source differs"):
        HVAWTSilResponseAdapter(
            raw_config=_raw_engineering_config(names),
            parameter_file=parameter,
            runtime=FakeRuntime(basis.node_ids, hybrid=True),
        )


def _toy_engineering_force(*, filtered_qdot, raw_qdot, qdot_rms):
    shape = np.zeros((2, 3, 2))
    shape[0, 1, 0] = 1.0
    shape[1, 1, 0] = 2.0
    shape[:, 2, 1] = 1.0

    class ToyBasis:
        node_ids = ("n0", "n1")
        real_shape_body_per_m = shape

        @staticmethod
        def energy_conjugate_force_body(generalized_force):
            force = np.zeros((2, 3))
            force[0, 1] = generalized_force[0]
            force[0, 2] = generalized_force[1]
            return force

    transfer = SimpleNamespace(
        feedback_gain=3.0,
        limiting_amplitude_ratio=0.1,
        calibration_fingerprint="a" * 64,
        static_spectrum_sha256="b" * 64,
    )
    spectral = SimpleNamespace(
        partition=SimpleNamespace(
            captured_amplitude_by_node_mode=np.array([[1.0, 0.0], [1.0, 0.0]])
        )
    )
    modes = (
        SimpleNamespace(
            tracked_mode=6,
            generalized_modal_mass_kg=1.0,
            generalized_modal_stiffness_npm=1.0,
            reference_length_m=1.0,
        ),
        SimpleNamespace(
            tracked_mode=12,
            generalized_modal_mass_kg=1.0,
            generalized_modal_stiffness_npm=4.0,
            reference_length_m=1.0,
        ),
    )
    flexible = np.zeros((2, 3))
    flexible[0, 1] = raw_qdot[0]
    flexible[0, 2] = raw_qdot[1]
    return calculate_hvawt_engineering_force(
        modal_basis=ToyBasis(),
        gain_state=transfer,
        spectral_result=spectral,
        mode_configs=modes,
        structural_damping_ratios=(0.01, 0.01),
        rotation_body_to_global=np.eye(3),
        response_relative_velocity_global_mps=np.tile([1.0, 0.0, 0.0], (2, 1)),
        response_effective_crossflow_speed_mps=np.ones(2),
        chord_unit_global=np.tile([1.0, 0.0, 0.0], (2, 1)),
        normal_unit_global=np.tile([0.0, 1.0, 0.0], (2, 1)),
        chord_m=np.ones(2),
        local_length_m=np.ones(2),
        fluid_density_kgpm3=2.0,
        flexible_velocity_global_mps=flexible,
        raw_qdot_mps=raw_qdot,
        filtered_qdot_mps=filtered_qdot,
        qdot_rms_mps=qdot_rms,
        maximum_power_closure_relative_error=1.0e-12,
    )


def test_engineering_feedback_coefficient_uses_squared_nonuniform_mode_shape():
    _force, diagnostics = _toy_engineering_force(
        filtered_qdot=[0.0, 0.0],
        raw_qdot=[0.0, 0.0],
        qdot_rms=[0.0, 0.0],
    )

    # H * sum(q_dyn * CL_capture * phi^2) = 3 * (1^2 + 2^2).
    assert diagnostics.mode_diagnostics[
        0
    ].zero_amplitude_feedback_coefficient_nspm == pytest.approx(15.0)


def test_engineering_saturation_is_signed_and_removes_work_above_lambda():
    force, diagnostics = _toy_engineering_force(
        filtered_qdot=[1.0, 0.0],
        raw_qdot=[1.0, 0.0],
        qdot_rms=[0.1, 0.0],
    )

    mode = diagnostics.mode_diagnostics[0]
    assert mode.physical_amplitude_ratio > 0.1
    assert mode.saturation_factor < 0.0
    assert mode.generalized_force_n < 0.0
    assert diagnostics.generalized_power_w < 0.0
    assert diagnostics.nodal_flexible_power_w < 0.0
    assert force[0, 1] < 0.0
