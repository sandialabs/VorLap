import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples.ehrmann2014_naca0018 import campaign
from vorlap.qblade_sil import SilVorLapCoupler
from vorlap.qblade_files import replace_qblade_keyword_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = (
    REPO_ROOT
    / "examples"
    / "ehrmann2014_naca0018"
    / "engineering_lockin_calibration.json"
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("engineering-lockin") / "campaign"
    campaign.generate(root, runtime_vorlap_root=REPO_ROOT)
    return root


def _prepare(
    generated,
    *,
    enabled=True,
    transfer=False,
    geometry_transfer=False,
    fu_training_reproduction=False,
    native_corrected=False,
    signed_work_shape=False,
    trained_case_id=None,
    residual_scale=0.0,
):
    manifest = json.loads((generated / "campaign_manifest.json").read_text())
    row = next(
        value
        for value in manifest["cases"]
        if value["scenario"] == "latest_spectral_lockin_bridge"
        and value["mass_damping"] == 0.96
        and value["reduced_velocity"] == 4.5
        and value["timestep_s"] == campaign.TIMESTEP_S
    )
    config_path = generated / row["runtime_config"]
    baseline = config_path.with_name("engineering_baseline.json")
    if not baseline.exists():
        baseline.write_bytes(config_path.read_bytes())
    config = json.loads(baseline.read_text())
    config["response_model"]["aerodynamic_damping_gain_h1"] = 0.0
    config["response_model"]["startup_seed"]["enabled"] = False
    config.pop("single_mode_spectral_lockin", None)
    staged = config_path.parent / CALIBRATION.name
    staged.write_bytes(CALIBRATION.read_bytes())
    native_corrected = bool(native_corrected or signed_work_shape)
    fu_training_reproduction = bool(fu_training_reproduction or native_corrected)
    geometry_transfer = bool(geometry_transfer or fu_training_reproduction)
    transfer = bool(transfer or geometry_transfer)
    config["single_mode_engineering_lockin"] = {
        "schema_version": (
            6
            if signed_work_shape
            else (
                5
                if native_corrected
                else (
                    4
                    if fu_training_reproduction
                    else (3 if geometry_transfer else (2 if transfer else 1))
                )
            )
        ),
        "type": (
            "single_mode_airfoil_geometry_blended_fu_signed_work_shape_reproduction_v6"
            if signed_work_shape
            else (
                "single_mode_airfoil_geometry_blended_fu_native_corrected_reproduction_v5"
                if native_corrected
                else (
                    "single_mode_airfoil_geometry_blended_fu_training_reproduction_v4"
                    if fu_training_reproduction
                    else (
                        "single_mode_airfoil_geometry_blended_complete_transfer_v3"
                        if geometry_transfer
                        else (
                            "single_mode_airfoil_engineering_lockin_transfer_v2"
                            if transfer
                            else "single_mode_airfoil_engineering_lockin_v1"
                        )
                    )
                )
            )
        ),
        "profile": "FFA-W3_family" if geometry_transfer else "NACA0018",
        "kinematics": "fixed_nonrotating",
        "calibration": {
            "path": staged.name,
            "sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
        },
        "static_spectrum": {
            "channel": "CL",
            "bounds_policy": "raise",
            "source_sha256": campaign.TRACKED_STATIONARY_SPECTRUM_SHA256,
            "n_frequency_entries": 200,
        },
        "modal_power": {
            "rms_time_constant_s": 0.1,
            "bandpass_damping_ratio": 0.1,
            "max_abs_target_power_w": 1000.0,
            "max_abs_generalized_force_n": 1000.0,
            "max_abs_nodal_force_n": 1000.0,
            "limit_action": "error",
            "enabled": bool(enabled),
        },
        "structural_damping_ratio": row["structural_damping_ratio"],
        "residual_source_scale": float(residual_scale),
    }
    if geometry_transfer:
        parent = (
            REPO_ROOT
            / "examples/viv_geometry_blended/geometry_blended_calibration.json"
        )
        staged = config_path.parent / parent.name
        staged.write_bytes(parent.read_bytes())
        config["single_mode_engineering_lockin"]["calibration"] = {
            "path": staged.name,
            "sha256": hashlib.sha256(staged.read_bytes()).hexdigest(),
        }
        trained_parent = (
            REPO_ROOT
            / "examples/horcas2020_iea10_tip_geometry/evidence/combined_tip_v1/model.json"
        )
        trained_staged = config_path.parent / "combined_tip_v1_model.json"
        trained_staged.write_bytes(trained_parent.read_bytes())
        trained_case_id = trained_case_id or (
            "P80A310_U06" if fu_training_reproduction else "P80A310_U14"
        )
        trained_row = next(
            value
            for value in json.loads(trained_staged.read_text())["training"][
                "fu2026_speed_orientation_rows_after_preserved_failure"
            ]
            if value["case_id"] == trained_case_id
        )
        config["single_mode_engineering_lockin"]["trained_condition"] = {
            "path": trained_staged.name,
            "sha256": hashlib.sha256(trained_staged.read_bytes()).hexdigest(),
            "case_id": trained_row["case_id"],
            "wind_speed_mps": trained_row["wind_speed_mps"],
            "zero_amplitude_feedback_over_structural_damping": trained_row[
                "zero_amplitude_feedback_over_structural_damping"
            ],
            "limiting_amplitude_m": trained_row["configuration_saturation_amplitude_m"],
        }
        if fu_training_reproduction:
            config["single_mode_engineering_lockin"]["static_spectrum"][
                "bounds_policy"
            ] = "clip"
            config["single_mode_engineering_lockin"]["published_tip_trigger"] = {
                "schema_version": 1,
                "enabled": True,
                "node_id": config["response_model"]["node_ids"][-1],
                "peak_force_n": 8000.0,
                "ramp_up_end_s": 10.0,
                "ramp_down_end_s": 11.0,
                "direction": "selected_edgewise_mode_direction",
                "source": "Fu2026_Method4_Table5",
            }
        if native_corrected:
            native_document = {
                "schema": "vorlap-fu2026-qblade-native-damping-calibration-v1",
                "status": "qualified",
                "cases": [
                    {
                        "case_id": trained_row["case_id"],
                        "native_total_to_structural_damping_ratio": 2.5,
                    }
                ],
            }
            native_document["fingerprint"] = hashlib.sha256(
                json.dumps(
                    native_document, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            native_path = config_path.parent / "native_damping_calibration.json"
            native_path.write_text(json.dumps(native_document, indent=2) + "\n")
            config["single_mode_engineering_lockin"]["native_damping_calibration"] = {
                "path": native_path.name,
                "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
                "case_id": trained_row["case_id"],
                "native_total_to_structural_damping_ratio": 2.5,
            }
        if signed_work_shape:
            shape_parent = (
                REPO_ROOT
                / "examples/fu2026_iea10/evidence/signed_work_shape_v1/model.json"
            )
            shape_path = config_path.parent / "signed_work_shape_transfer.json"
            shape_path.write_bytes(shape_parent.read_bytes())
            shape_document = json.loads(shape_path.read_text())
            shape_row = next(
                value
                for value in shape_document["cases"]
                if value["case_id"] == trained_row["case_id"]
            )
            config["single_mode_engineering_lockin"]["signed_work_shape_transfer"] = {
                "path": shape_path.name,
                "sha256": hashlib.sha256(shape_path.read_bytes()).hexdigest(),
                "case_id": shape_row["case_id"],
                "active_branch": shape_row["active_branch"],
                "equilibrium_amplitude_m": shape_row["equilibrium_amplitude_m"],
                "feedback_over_opposing_damping": shape_row[
                    "transferred_feedback_over_opposing_damping"
                ],
                "limiting_amplitude_m": shape_row[
                    "transferred_limiting_amplitude_m"
                ],
            }
    if transfer:
        config["single_mode_engineering_lockin"]["static_spectrum"].update(
            {
                "maximum_spectral_gain": 10.0,
                "minimum_modal_participation": 1.0e-12,
                "retain_uncaptured_lift_spectrum": True,
                "retain_fluctuating_drag_spectrum": True,
                "include_mean_load": False,
            }
        )
    if geometry_transfer:
        config["single_mode_engineering_lockin"]["static_spectrum"].pop("source_sha256")
        config["single_mode_engineering_lockin"]["static_spectrum"][
            "source_sha256_by_airfoil"
        ] = {"NACA0018": campaign.TRACKED_STATIONARY_SPECTRUM_SHA256}
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if geometry_transfer:
        replace_qblade_keyword_file(
            generated / row["sim"], "MEANINF", trained_row["wind_speed_mps"]
        )
    return generated / row["sim"]


def _velocity(adapter, scale):
    directions = np.asarray(adapter.runtime.controller._base_chord, dtype=float)
    mode = np.asarray(adapter._mode_shape, dtype=float)
    return directions * (scale * mode)[:, None]


def test_adapter_uses_full_static_spectrum_and_has_negative_damping(generated):
    coupler = SilVorLapCoupler(_prepare(generated), history_mode="off")
    adapter = coupler.hvawt_response_adapter
    assert type(adapter).__name__ == "QBladeSingleModeEngineeringLockInSilAdapter"
    adapter.step(coupler.swap, response_time_s=0.0)
    velocity_spec = coupler.runtime.layout["velocity"]
    velocity = _velocity(adapter, 0.02)
    coupler.swap[velocity_spec.offset : velocity_spec.offset + velocity_spec.size] = (
        velocity.reshape(-1)
    )
    force = adapter.step(coupler.swap, response_time_s=0.05)
    diagnostics = adapter.last_diagnostics
    assert diagnostics.captured_active_harmonics_min >= 1
    assert diagnostics.captured_lift_coefficient_min > 0.0
    assert diagnostics.spectral_stretch_gain == pytest.approx(adapter.feedback_gain)
    assert diagnostics.zero_amplitude_feedback_coefficient_kgps > 0.0
    assert diagnostics.negative_to_structural_damping_ratio > 1.0
    assert diagnostics.linear_lockin_available
    assert diagnostics.predicted_equilibrium_amplitude_ratio > 0.0
    assert float(np.sum(force * velocity)) > 0.0
    assert adapter.seed_model.config.aerodynamic_damping_gain_h1 == 0.0


def test_modal_force_is_energy_conjugate_for_arbitrary_nonmodal_velocity(generated):
    adapter = SilVorLapCoupler(
        _prepare(generated), history_mode="off"
    ).hvawt_response_adapter
    model = adapter.power_model
    directions = np.asarray(adapter.runtime.controller._base_chord, dtype=float)
    zeros = np.zeros_like(directions)

    def target(amplitude):
        from vorlap.qblade_modal_power_model import ExternalModalPowerTarget

        omega = 2.0 * np.pi * model.config.natural_frequency_hz
        return ExternalModalPowerTarget(
            0.5 * 2.0 * omega**2 * amplitude**2,
            2.0,
            "a" * 64,
        )

    model.step(
        0.0,
        node_ids=adapter.node_ids,
        direction_basis=directions,
        structural_velocity_mps=zeros,
        target_power_evaluator=target,
    )
    velocity = directions * np.linspace(1.0, -0.4, len(adapter.node_ids))[:, None]
    result = model.step(
        0.05,
        node_ids=adapter.node_ids,
        direction_basis=directions,
        structural_velocity_mps=velocity,
        target_power_evaluator=target,
    )
    conjugate = (
        result.diagnostics.applied_generalized_force_n * result.diagnostics.raw_qdot_mps
    )
    assert result.diagnostics.instantaneous_structural_power_w == pytest.approx(
        conjugate, rel=2e-14, abs=1e-14
    )


@pytest.mark.parametrize(
    "prepare_kwargs",
    [
        {"fu_training_reproduction": True},
        {"native_corrected": True},
        {"signed_work_shape": True},
    ],
    ids=["v4", "v5", "v6"],
)
def test_fu_reproduction_uses_spectrum_weighted_conjugate_dual(
    generated, prepare_kwargs
):
    coupler = SilVorLapCoupler(
        _prepare(generated, **prepare_kwargs), history_mode="off"
    )
    adapter = coupler.hvawt_response_adapter
    adapter.step(coupler.swap, response_time_s=0.0)
    directions = np.asarray(adapter.runtime.controller._base_chord, dtype=float)
    directional_velocity = np.linspace(0.8, -0.3, len(adapter.node_ids))
    velocity = directions * directional_velocity[:, None]
    velocity_spec = coupler.runtime.layout["velocity"]
    coupler.swap[velocity_spec.offset : velocity_spec.offset + velocity_spec.size] = (
        velocity.reshape(-1)
    )
    adapter.step(coupler.swap, response_time_s=0.05)

    result = adapter.last_power_result
    diagnostics = result.diagnostics
    assert diagnostics.dynamic_conjugate_weight_override is True
    assert diagnostics.dynamic_conjugate_weight_max > (
        diagnostics.dynamic_conjugate_weight_min
    )
    assert diagnostics.instantaneous_structural_power_w == pytest.approx(
        float(np.sum(result.additive_force_n * velocity)),
        rel=2.0e-14,
        abs=1.0e-14,
    )
    assert diagnostics.instantaneous_structural_power_w == pytest.approx(
        diagnostics.applied_generalized_force_n * diagnostics.raw_qdot_mps,
        rel=2.0e-14,
        abs=1.0e-14,
    )
    assert adapter.last_diagnostics.spectrum_reynolds_clip_low_nodes == 0
    assert adapter.last_diagnostics.spectrum_reynolds_clip_high_nodes == 0
    assert adapter.last_diagnostics.spectrum_incidence_clip_low_nodes == 0
    assert adapter.last_diagnostics.spectrum_incidence_clip_high_nodes == 0


def test_native_corrected_reproduction_scales_feedback_against_total_damping(
    generated,
):
    coupler = SilVorLapCoupler(
        _prepare(generated, native_corrected=True), history_mode="off"
    )
    adapter = coupler.hvawt_response_adapter
    adapter.step(coupler.swap, response_time_s=0.0)
    diagnostics = adapter.last_diagnostics
    trained_ratio = float(adapter.trained_condition["feedback_ratio"])
    assert adapter.model_type.endswith("fu_native_corrected_reproduction_v5")
    assert diagnostics.native_total_to_structural_damping_ratio == pytest.approx(2.5)
    assert diagnostics.native_total_damping_coefficient_kgps == pytest.approx(
        2.5 * diagnostics.structural_damping_coefficient_kgps
    )
    assert diagnostics.negative_to_structural_damping_ratio == pytest.approx(
        2.5 * trained_ratio
    )
    assert diagnostics.negative_to_native_total_damping_ratio == pytest.approx(
        trained_ratio
    )
    expected = adapter._limiting_amplitude_m * np.sqrt(1.0 - 1.0 / trained_ratio)
    assert diagnostics.predicted_equilibrium_amplitude_m == pytest.approx(expected)


def test_signed_work_shape_transfer_preserves_active_training_equilibrium(generated):
    coupler = SilVorLapCoupler(
        _prepare(generated, signed_work_shape=True), history_mode="off"
    )
    adapter = coupler.hvawt_response_adapter
    adapter.step(coupler.swap, response_time_s=0.0)
    diagnostics = adapter.last_diagnostics
    shape = adapter.signed_work_shape_transfer
    assert adapter.model_type.endswith("fu_signed_work_shape_reproduction_v6")
    assert shape["active_branch"] is True
    assert diagnostics.negative_to_native_total_damping_ratio == pytest.approx(
        shape["feedback_over_opposing_damping"]
    )
    assert diagnostics.limiting_amplitude_m == pytest.approx(
        shape["limiting_amplitude_m"]
    )
    assert diagnostics.predicted_equilibrium_amplitude_m == pytest.approx(
        shape["equilibrium_amplitude_m"], abs=2.0e-12
    )


def test_signed_work_shape_transfer_disables_inactive_correlated_feedback(generated):
    coupler = SilVorLapCoupler(
        _prepare(
            generated,
            signed_work_shape=True,
            trained_case_id="P80A310_U22",
        ),
        history_mode="off",
    )
    adapter = coupler.hvawt_response_adapter
    adapter.step(coupler.swap, response_time_s=0.0)
    diagnostics = adapter.last_diagnostics
    assert adapter.signed_work_shape_transfer["active_branch"] is False
    assert diagnostics.zero_amplitude_feedback_coefficient_kgps == 0.0
    assert diagnostics.negative_to_native_total_damping_ratio == 0.0
    assert diagnostics.linear_lockin_available is False
    assert diagnostics.predicted_equilibrium_amplitude_m == 0.0


def test_saturation_reverses_work_and_checkpoint_replays(generated):
    adapter = SilVorLapCoupler(
        _prepare(generated), history_mode="off"
    ).hvawt_response_adapter
    evaluator = adapter._target_evaluator(
        zero_amplitude_feedback_kgps=3.0,
        coefficients=np.ones(len(adapter.node_ids)),
        captures=[],
        effective_speed_mps=np.ones(len(adapter.node_ids)),
    )
    below = evaluator(0.5 * adapter._limiting_amplitude_m)
    at = evaluator(adapter._limiting_amplitude_m)
    above = evaluator(1.5 * adapter._limiting_amplitude_m)
    assert below.target_cycle_mean_power_w > 0.0
    assert at.target_cycle_mean_power_w == pytest.approx(0.0, abs=1e-15)
    assert above.target_cycle_mean_power_w < 0.0

    state = adapter.checkpoint_state()
    adapter.restore_checkpoint_state(state)
    assert adapter.checkpoint_state() == state


def test_disabled_adapter_returns_exact_zero_without_seed(generated):
    adapter = SilVorLapCoupler(
        _prepare(generated, enabled=False), history_mode="off"
    ).hvawt_response_adapter
    assert not adapter.seed_model.config.startup_seed.enabled
    force = adapter.step(np.zeros(adapter.runtime.swap_size), response_time_s=0.0)
    np.testing.assert_array_equal(force, 0.0)


def test_fixed_adapter_accepts_only_roundoff_scale_azimuth_drift(generated):
    coupler = SilVorLapCoupler(
        _prepare(generated, geometry_transfer=True), history_mode="off"
    )
    adapter = coupler.hvawt_response_adapter
    azimuth = coupler.runtime.layout[coupler.runtime.azimuth_block]
    coupler.swap[azimuth.offset] = 5.0e-7
    force = adapter.step(coupler.swap, response_time_s=0.0)
    assert np.all(np.isfinite(force))

    coupler.swap[azimuth.offset] = 1.0e-4
    with pytest.raises(ValueError, match="observed remainder"):
        adapter.step(coupler.swap, response_time_s=0.05)


def test_fu_method4_trigger_is_exact_and_applies_only_at_tip(generated):
    coupler = SilVorLapCoupler(
        _prepare(generated, fu_training_reproduction=True), history_mode="off"
    )
    adapter = coupler.hvawt_response_adapter
    assert adapter.model_type.endswith("fu_training_reproduction_v4")
    expected = {
        0.0: 0.0,
        5.0: 4000.0,
        10.0: 8000.0,
        10.5: 4000.0,
        11.0: 0.0,
        12.0: 0.0,
    }
    for time_s, force_n in expected.items():
        assert adapter._prescribed_tip_trigger_force_n(time_s) == pytest.approx(
            force_n, abs=1e-12
        )

    force = adapter.step(coupler.swap, response_time_s=5.0)
    tip = int(adapter.tip_trigger["node_index"])
    nonzero_rows = np.flatnonzero(np.linalg.norm(force, axis=1) > 0.0)
    np.testing.assert_array_equal(nonzero_rows, [tip])
    assert np.linalg.norm(force[tip]) == pytest.approx(4000.0, rel=1e-13)
    fields = adapter.history_fields()
    assert fields["response_engineering_prescribed_tip_trigger_active"] is True
    assert fields[
        "response_engineering_prescribed_tip_trigger_force_n"
    ] == pytest.approx(4000.0)


def test_fu_trigger_contract_fails_closed_on_wrong_node(generated):
    sim = _prepare(generated, fu_training_reproduction=True)
    manifest = json.loads((generated / "campaign_manifest.json").read_text())
    row = next(value for value in manifest["cases"] if generated / value["sim"] == sim)
    config_path = generated / row["runtime_config"]
    config = json.loads(config_path.read_text())
    config["single_mode_engineering_lockin"]["published_tip_trigger"][
        "node_id"
    ] = "missing-tip"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    with pytest.raises(ValueError, match="exactly one runtime node"):
        SilVorLapCoupler(sim, history_mode="off")


def test_complete_v2_adds_only_partitioned_zero_mean_residual(generated):
    isolation_coupler = SilVorLapCoupler(
        _prepare(generated, transfer=True, residual_scale=0.0), history_mode="off"
    )
    complete_coupler = SilVorLapCoupler(
        _prepare(generated, transfer=True, residual_scale=1.0), history_mode="off"
    )
    isolation = isolation_coupler.hvawt_response_adapter
    complete = complete_coupler.hvawt_response_adapter
    for time_s, scale in ((0.0, 0.0), (0.05, 0.02)):
        for adapter, coupler in (
            (isolation, isolation_coupler),
            (complete, complete_coupler),
        ):
            velocity_spec = adapter.runtime.layout["velocity"]
            coupler.swap[
                velocity_spec.offset : velocity_spec.offset + velocity_spec.size
            ] = _velocity(adapter, scale).reshape(-1)
        isolation_force = isolation.step(isolation_coupler.swap, response_time_s=time_s)
        complete_force = complete.step(complete_coupler.swap, response_time_s=time_s)

    residual = np.asarray(
        complete.last_spectral_result.residual_force_global_n, dtype=float
    )
    np.testing.assert_allclose(
        complete_force - isolation_force,
        residual,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    fields = complete.history_fields()
    assert fields["response_spectral_residual_source_scale"] == 1.0
    assert fields["response_spectral_applied_residual_force_power_w"] == pytest.approx(
        fields["response_spectral_raw_residual_force_power_w"]
    )
    assert fields["response_spectral_residual_energy_sum"] > 0.0
    assert fields["response_spectral_source_energy_sum"] == pytest.approx(
        fields["response_spectral_residual_energy_sum"]
        + fields["response_spectral_captured_energy_sum"],
        rel=0.0,
        abs=2.0e-12,
    )
    assert complete.checkpoint_state()["schema_version"] == 2


def test_geometry_blended_v3_complete_path_is_three_component_and_bound(generated):
    coupler = SilVorLapCoupler(
        _prepare(generated, geometry_transfer=True, residual_scale=1.0),
        history_mode="off",
    )
    adapter = coupler.hvawt_response_adapter
    assert adapter.model_type == (
        "single_mode_airfoil_geometry_blended_complete_transfer_v3"
    )
    adapter.step(coupler.swap, response_time_s=0.0)
    velocity_spec = adapter.runtime.layout["velocity"]
    velocity = _velocity(adapter, 0.02)
    coupler.swap[velocity_spec.offset : velocity_spec.offset + velocity_spec.size] = (
        velocity.reshape(-1)
    )
    force = adapter.step(coupler.swap, response_time_s=0.05)
    assert np.all(np.isfinite(force))
    assert adapter.last_spectral_result is not None
    assert adapter.last_diagnostics.zero_amplitude_feedback_coefficient_kgps >= 0.0
    assert adapter.last_diagnostics.spectral_stretch_gain >= 0.0
    assert adapter.trained_condition["feedback_ratio"] == pytest.approx(
        10.756097560975602
    )
    assert adapter.checkpoint_state()["schema_version"] == 3
