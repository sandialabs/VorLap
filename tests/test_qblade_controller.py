import copy
import json

import numpy as np
import pytest

import vorlap
from conftest import make_component, make_constant_airfoil_fft, make_viv_params
from vorlap import ControllerSourceState, QBladeController, SwapArrayAdapter, SwapBlockSpec
from vorlap.qblade_response_model import (
    QBladeResponseModel,
    QBladeResponseModelConfig,
)
from vorlap.qblade_modal_power_model import (
    QBladeModalPowerDiagnostics,
    QBladeModalPowerLimitError,
    QBladeModalPowerModel,
    QBladeModalPowerModelConfig,
    TabulatedAerodynamicPowerCurve,
)


def _build_controller(
    node_ids=None,
    force_scale=1.0,
    afft=None,
    include_mean_load=False,
    source_bounds_policy="warn",
    source_frequency_cap_hz=None,
    n_freq_depth=2,
    response_model=None,
):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    if afft is None:
        afft = make_constant_airfoil_fft(n_freq=2)
    viv_params = make_viv_params()
    viv_params.output_time = np.array([0.0, 0.25], dtype=float)
    if node_ids is None:
        node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    return QBladeController(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=node_ids,
        n_freq_depth=n_freq_depth,
        force_scale=force_scale,
        include_mean_load=include_mean_load,
        source_bounds_policy=source_bounds_policy,
        source_frequency_cap_hz=source_frequency_cap_hz,
        response_model=response_model,
    )


def _response_model(
    *,
    node_ids=("BLD_1_0.000000", "BLD_1_1.000000"),
    mode_shape=(0.5, 1.0),
    enabled=True,
    aerodynamic_damping_gain_h1=2.0,
    force_direction="normal",
    projection_weights=(1.0, 1.0),
    bandpass_damping_ratio=0.2,
    max_abs_gain=20.0,
    max_abs_nodal_force_n=1.0e5,
    schema_version=1,
    aerodynamic_speed_basis="relative_crossflow",
    response_section_width_m=None,
):
    return QBladeResponseModel(
        QBladeResponseModelConfig(
            node_ids=tuple(node_ids),
            mode_shape=tuple(mode_shape),
            natural_frequency_hz=1.0,
            reference_length_m=1.0,
            reduced_velocity_low=4.0,
            reduced_velocity_peak=5.0,
            reduced_velocity_high=6.0,
            aerodynamic_damping_gain_h1=aerodynamic_damping_gain_h1,
            saturation_amplitude_ratio=0.2,
            rms_time_constant_s=0.5,
            max_abs_gain=max_abs_gain,
            max_abs_nodal_force_n=max_abs_nodal_force_n,
            force_direction=force_direction,
            projection_weights=tuple(projection_weights),
            bandpass_damping_ratio=bandpass_damping_ratio,
            enabled=enabled,
            schema_version=schema_version,
            aerodynamic_speed_basis=aerodynamic_speed_basis,
            response_section_width_m=response_section_width_m,
        )
    )


def _modal_power_model(
    *,
    node_ids=("BLD_1_0.000000", "BLD_1_1.000000"),
    mode_shape=(0.5, 1.0),
    enabled=True,
    force_direction="chord",
    limit_action="error",
    max_abs_target_power_w=1.0e4,
    max_abs_generalized_force_n=1.0e4,
    max_abs_nodal_force_n=1.0e4,
):
    curve = TabulatedAerodynamicPowerCurve(
        amplitude_m=(0.0, 0.1, 0.5, 1.0),
        power_w=(0.0, 5.0, 50.0, -10.0),
        source_label="synthetic controller integration curve",
    )
    return QBladeModalPowerModel(
        QBladeModalPowerModelConfig(
            node_ids=tuple(node_ids),
            mode_shape=tuple(mode_shape),
            modal_mass_weights_kg=(2.0, 1.0),
            aerodynamic_tributary_weights=(1.0, 2.0),
            natural_frequency_hz=1.0,
            power_curve=curve,
            rms_time_constant_s=0.5,
            max_abs_target_power_w=max_abs_target_power_w,
            max_abs_generalized_force_n=max_abs_generalized_force_n,
            max_abs_nodal_force_n=max_abs_nodal_force_n,
            bandpass_damping_ratio=0.2,
            force_direction=force_direction,
            limit_action=limit_action,
            enabled=enabled,
        )
    )


def test_qblade_controller_is_reexported_from_package_root():
    assert vorlap.QBladeController is QBladeController
    assert vorlap.QBladeResponseModel is QBladeResponseModel
    assert vorlap.QBladeResponseModelConfig is QBladeResponseModelConfig
    assert vorlap.QBladeResponseDiagnostics.__name__ == "QBladeResponseDiagnostics"
    assert vorlap.QBladeResponseResult.__name__ == "QBladeResponseResult"
    assert vorlap.QBladeModalPowerModel is QBladeModalPowerModel
    assert vorlap.QBladeModalPowerModelConfig is QBladeModalPowerModelConfig
    assert vorlap.QBladeModalPowerDiagnostics is QBladeModalPowerDiagnostics
    assert vorlap.TabulatedAerodynamicPowerCurve is TabulatedAerodynamicPowerCurve


def test_step_returns_per_node_forces():
    controller = _build_controller()
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    forces = controller.step(
        {
            "velocity": velocities,
            "time": 0.0,
            "azimuth_deg": 0.0,
        }
    )

    assert forces.shape == (2, 3)
    assert np.isfinite(forces).all()
    assert forces.flags.writeable


def test_step_returns_stable_copy_by_default():
    controller = _build_controller()
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    first = controller.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})
    first_snapshot = first.copy()
    second = controller.step({"velocity": velocities, "time": 0.1, "azimuth_deg": 0.0})

    assert second is not first
    np.testing.assert_allclose(first, first_snapshot, atol=1e-12)


def test_force_scale_scales_output_forces():
    afft = make_constant_airfoil_fft(n_freq=2, cl_dc=0.0, cd_dc=0.0, cl_h1_amp=1.0, cd_h1_amp=0.5)
    base = _build_controller(force_scale=1.0, afft=afft)
    scaled = _build_controller(force_scale=100.0, afft=afft)
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    base_forces = base.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})
    scaled_forces = scaled.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})

    np.testing.assert_allclose(scaled_forces, 100.0 * base_forces, rtol=1e-12, atol=1e-12)


def test_disabled_response_model_preserves_legacy_force_path():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=1.0,
        cd_h1_amp=0.5,
        st_h1=0.4,
    )
    legacy = _build_controller(afft=afft)
    disabled = _build_controller(
        afft=afft,
        response_model=_response_model(enabled=False),
    )

    for time_s, speed_mps in ((0.0, 2.0), (0.17, 3.0), (0.31, 1.5)):
        relative_velocity = np.tile([speed_mps, 0.0, 0.0], (2, 1))
        node_data = {
            "velocity": relative_velocity,
            "structural_velocity": np.tile([0.0, 0.1, 0.0], (2, 1)),
            "time": time_s,
            "azimuth_deg": 0.0,
        }
        np.testing.assert_array_equal(legacy.step(node_data), disabled.step(node_data))

    assert disabled.response_model_enabled is False
    assert disabled.last_response_diagnostics is None


def test_disabled_modal_power_model_preserves_legacy_force_path_bit_for_bit():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=1.0,
        cd_h1_amp=0.5,
        st_h1=0.4,
    )
    legacy = _build_controller(afft=afft)
    disabled = _build_controller(
        afft=afft,
        response_model=_modal_power_model(enabled=False),
    )

    for time_s, speed_mps in ((0.0, 2.0), (0.17, 3.0), (0.31, 1.5)):
        relative_velocity = np.tile([speed_mps, 0.0, 0.0], (2, 1))
        node_data = {
            "velocity": relative_velocity,
            "structural_velocity": np.tile([0.1, 0.0, 0.0], (2, 1)),
            "time": time_s,
            "azimuth_deg": 0.0,
        }
        np.testing.assert_array_equal(legacy.step(node_data), disabled.step(node_data))

    assert disabled.response_model_enabled is False
    assert disabled.response_model_type == "modal_aerodynamic_power"
    assert disabled.last_response_diagnostics is None


def test_modal_power_force_and_generalized_summary_use_rotating_chord():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    controller = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_modal_power_model(),
    )
    relative_velocity = np.tile([5.0, 0.0, 0.0], (2, 1))
    forces = None
    for index in range(413):
        time_s = index * 0.02
        qdot_mps = 2.0 * np.pi * 0.3 * np.sin(2.0 * np.pi * time_s)
        forces = controller.step(
            {
                "velocity": relative_velocity,
                "structural_velocity": np.array(
                    [[0.5 * qdot_mps, 0.0, 0.0], [qdot_mps, 0.0, 0.0]],
                    dtype=float,
                ),
                "time": time_s,
                "azimuth_deg": 0.0,
            }
        )
    assert forces is not None
    diagnostics = controller.last_response_diagnostics
    assert isinstance(diagnostics, QBladeModalPowerDiagnostics)
    assert diagnostics.target_cycle_mean_power_w > 0.0
    assert diagnostics.applied_target_cycle_mean_power_w == pytest.approx(
        diagnostics.target_cycle_mean_power_w
    )
    assert diagnostics.instantaneous_modal_power_w > 0.0
    np.testing.assert_allclose(forces[:, 1:], 0.0, atol=1.0e-12)
    assert np.max(np.abs(forces[:, 0])) > 0.0
    assert diagnostics.applied_generalized_force_n == pytest.approx(
        np.dot(np.array([0.5, 1.0]), forces[:, 0])
    )

    summary = controller.response_diagnostics_summary
    assert controller.response_model_type == "modal_aerodynamic_power"
    assert summary["model_type"] == "modal_aerodynamic_power"
    assert summary["samples"] == 413
    assert summary["max_amplitude_estimate_m"] > 0.0
    assert summary["max_abs_target_cycle_mean_power_w"] > 0.0
    assert summary["max_abs_applied_target_cycle_mean_power_w"] > 0.0
    assert summary["max_abs_instantaneous_modal_power_w"] > 0.0
    assert summary["max_abs_instantaneous_structural_power_w"] > 0.0
    assert summary["max_abs_realized_modal_power_w"] > 0.0
    assert summary["max_abs_realized_structural_power_w"] > 0.0
    assert summary["target_power_limited_steps"] == 0
    assert summary["generalized_force_limited_steps"] == 0
    assert summary["nodal_force_limited_steps"] == 0

    source = controller.last_source_diagnostics
    assert source is not None
    np.testing.assert_array_equal(source.force_norm_n, 0.0)
    np.testing.assert_allclose(source.response_force_norm_n, np.linalg.norm(forces, axis=1))


def test_modal_power_limit_counts_accumulate_and_survive_checkpoint():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    model_kwargs = dict(
        limit_action="clip",
        max_abs_target_power_w=1.0,
        max_abs_generalized_force_n=0.1,
        max_abs_nodal_force_n=0.01,
    )
    controller = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_modal_power_model(**model_kwargs),
    )
    for index in range(213):
        time_s = index * 0.02
        qdot_mps = 2.0 * np.pi * 0.3 * np.sin(2.0 * np.pi * time_s)
        controller.step(
            {
                "velocity": np.tile([5.0, 0.0, 0.0], (2, 1)),
                "structural_velocity": np.array(
                    [[0.5 * qdot_mps, 0.0, 0.0], [qdot_mps, 0.0, 0.0]],
                    dtype=float,
                ),
                "time": time_s,
                "azimuth_deg": 0.0,
            }
        )

    summary = controller.response_diagnostics_summary
    assert summary["target_power_limited_steps"] > 0
    assert summary["generalized_force_limited_steps"] > 0
    assert summary["nodal_force_limited_steps"] > 0
    assert summary["nodal_force_limited_node_samples"] > 0
    assert summary["max_nodal_force_limited_nodes"] > 0
    assert summary["force_limited_node_samples"] == summary[
        "nodal_force_limited_node_samples"
    ]

    restored = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_modal_power_model(**model_kwargs),
    )
    restored.restore_checkpoint_state(
        json.loads(json.dumps(controller.checkpoint_state()))
    )
    assert restored.response_diagnostics_summary == summary
    assert restored.last_response_diagnostics == controller.last_response_diagnostics


def test_rejected_modal_power_step_preserves_last_accepted_controller_cache():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    controller = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_modal_power_model(
            max_abs_target_power_w=1.0,
        ),
    )
    last_inputs = None
    last_force = None
    last_checkpoint = None
    rejected = False
    for index in range(1, 401):
        time_s = index * 0.02
        qdot_mps = 2.0 * np.pi * 0.3 * np.sin(2.0 * np.pi * time_s)
        inputs = {
            "velocity": np.tile([5.0, 0.0, 0.0], (2, 1)),
            "structural_velocity": np.array(
                [[0.5 * qdot_mps, 0.0, 0.0], [qdot_mps, 0.0, 0.0]],
                dtype=float,
            ),
            "time": time_s,
            "azimuth_deg": 0.0,
        }
        before = controller.checkpoint_state()
        try:
            force = controller.step(inputs)
        except QBladeModalPowerLimitError as exc:
            assert "target power" in str(exc)
            assert controller.checkpoint_state() == before
            rejected = True
            break
        last_inputs = inputs
        last_force = force
        last_checkpoint = controller.checkpoint_state()

    assert rejected
    assert last_inputs is not None and last_force is not None and last_checkpoint is not None
    retry = controller.step(last_inputs)
    np.testing.assert_array_equal(retry, last_force)
    retry_checkpoint = controller.checkpoint_state()
    assert retry_checkpoint["cached_step"] == last_checkpoint["cached_step"]
    assert retry_checkpoint["response_model_state"] == last_checkpoint["response_model_state"]
    assert retry_checkpoint["response_diagnostics_summary"] == last_checkpoint[
        "response_diagnostics_summary"
    ]


def test_response_force_uses_rotating_normal_and_is_added_after_base_scale():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    controller = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_response_model(),
    )
    # At +90 deg about +z, the base +y normal is global -x.  Excite the
    # configured mode at f_n long enough to clear the bandpass transient.
    relative_velocity = np.tile([0.0, 5.0, 0.0], (2, 1))
    forces = None
    for index in range(413):
        time_s = index * 0.02
        qdot_mps = 0.2 * np.sin(2.0 * np.pi * time_s)
        structural_velocity = np.array(
            [[-0.5 * qdot_mps, 0.0, 0.0], [-qdot_mps, 0.0, 0.0]],
            dtype=float,
        )
        forces = controller.step(
            {
                "velocity": relative_velocity,
                "structural_velocity": structural_velocity,
                "time": time_s,
                "azimuth_deg": 90.0,
            }
        )
    assert forces is not None

    # D is the thickness-limited Strouhal length 0.12 m and each endpoint has
    # a 1 m tributary length.  The filtered modal velocity, not rigid drift,
    # enters F=0.5*rho*U*D*l*gain*phi*qdot_filtered.
    diagnostics = controller.last_response_diagnostics
    assert diagnostics is not None
    expected_scalar = (
        0.5
        * 1.0
        * 5.0
        * 0.12
        * 1.0
        * diagnostics.applied_gain_max
        * np.array([0.5, 1.0])
        * diagnostics.filtered_qdot_mps
    )
    np.testing.assert_allclose(forces[:, 0], -expected_scalar, atol=1.0e-12)
    np.testing.assert_allclose(forces[:, 1:], 0.0, atol=1.0e-12)
    assert diagnostics.activation_max == pytest.approx(1.0)
    assert diagnostics.modal_power_w > 0.0
    source = controller.last_source_diagnostics
    assert source is not None
    np.testing.assert_allclose(source.force_norm_n, 0.0, atol=0.0)
    np.testing.assert_allclose(source.force_coefficient_abs, 0.0, atol=0.0)
    np.testing.assert_allclose(source.response_force_norm_n, expected_scalar)
    np.testing.assert_allclose(source.total_force_norm_n, expected_scalar)


def test_response_force_can_use_configured_rotating_chord_direction():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    controller = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_response_model(force_direction="chord"),
    )
    # At +90 deg about +z, the base +x chord direction is global +y.
    relative_velocity = np.tile([0.0, 5.0, 0.0], (2, 1))
    forces = None
    for index in range(413):
        time_s = index * 0.02
        qdot_mps = 0.2 * np.sin(2.0 * np.pi * time_s)
        structural_velocity = np.array(
            [[0.0, 0.5 * qdot_mps, 0.0], [0.0, qdot_mps, 0.0]],
            dtype=float,
        )
        forces = controller.step(
            {
                "velocity": relative_velocity,
                "structural_velocity": structural_velocity,
                "time": time_s,
                "azimuth_deg": 90.0,
            }
        )
    assert forces is not None
    diagnostics = controller.last_response_diagnostics
    assert diagnostics is not None
    expected_scalar = (
        0.5
        * 1.0
        * 5.0
        * 0.12
        * 1.0
        * diagnostics.applied_gain_max
        * np.array([0.5, 1.0])
        * diagnostics.filtered_qdot_mps
    )
    np.testing.assert_allclose(forces[:, 0], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(forces[:, 1], expected_scalar, atol=1.0e-12)
    np.testing.assert_allclose(forces[:, 2], 0.0, atol=1.0e-12)


def test_controller_rejects_response_node_order_mismatch():
    response_model = _response_model(
        node_ids=("BLD_1_1.000000", "BLD_1_0.000000"),
        mode_shape=(1.0, 0.5),
    )
    with pytest.raises(ValueError, match="node_ids.*order"):
        _build_controller(response_model=response_model)


def test_response_clipping_counts_accumulate_and_survive_checkpoint():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    controller = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_response_model(
            aerodynamic_damping_gain_h1=10.0,
            max_abs_gain=0.1,
            max_abs_nodal_force_n=0.001,
        ),
    )
    for index in range(213):
        time_s = index * 0.02
        qdot_mps = np.sin(2.0 * np.pi * time_s)
        controller.step(
            {
                "velocity": np.tile([5.0, 0.0, 0.0], (2, 1)),
                "structural_velocity": np.array(
                    [
                        [0.0, 0.5 * qdot_mps, 0.0],
                        [0.0, qdot_mps, 0.0],
                    ],
                    dtype=float,
                ),
                "time": time_s,
                "azimuth_deg": 0.0,
            }
        )

    summary = controller.response_diagnostics_summary
    assert summary["samples"] == 213
    assert summary["gain_limited_node_samples"] > 0
    assert summary["force_limited_node_samples"] > 0
    diagnostics = controller.last_response_diagnostics
    assert diagnostics is not None
    assert diagnostics.gain_limited_nodes == 2
    assert diagnostics.force_limited_nodes == 2
    restored = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_response_model(
            aerodynamic_damping_gain_h1=10.0,
            max_abs_gain=0.1,
            max_abs_nodal_force_n=0.001,
        ),
    )
    restored.restore_checkpoint_state(
        json.loads(json.dumps(controller.checkpoint_state()))
    )
    assert restored.response_diagnostics_summary == summary


def test_schema_v2_response_uses_ambient_crossflow_and_width_while_base_stays_relative():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    controller = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_response_model(
            schema_version=2,
            aerodynamic_speed_basis="declared_ambient_crossflow",
        ),
    )
    # The 12 m/s spanwise component must not enter the response speed.  In the
    # section plane, Uinf=(3,4) gives exactly 5 m/s and an ambient width of
    # c*|sin(alpha)|=0.8 m, well above the 0.12 thickness floor.
    ambient_velocity = np.tile([3.0, 4.0, 12.0], (2, 1))
    for index in range(25):
        time_s = index * 0.02
        qdot_mps = 0.5 * np.sin(2.0 * np.pi * time_s)
        structural_velocity = np.array(
            [[0.0, 0.5 * qdot_mps, 0.0], [0.0, qdot_mps, 0.0]],
            dtype=float,
        )
        relative_velocity = ambient_velocity - structural_velocity
        controller.step(
            {
                "velocity": relative_velocity,
                "structural_velocity": structural_velocity,
                "declared_ambient_velocity": ambient_velocity,
                "time": time_s,
                "azimuth_deg": 0.0,
            }
        )

    diagnostics = controller.last_response_diagnostics
    source = controller.last_source_diagnostics
    assert diagnostics is not None and source is not None
    assert diagnostics.response_model_schema_version == 2
    assert diagnostics.aerodynamic_speed_basis == "declared_ambient_crossflow"
    assert diagnostics.response_declared_ambient_crossflow_speed_min_mps == 5.0
    assert diagnostics.response_declared_ambient_crossflow_speed_max_mps == 5.0
    assert diagnostics.realized_response_reduced_velocity_min == 5.0
    assert diagnostics.realized_response_reduced_velocity_max == 5.0
    assert diagnostics.response_relative_crossflow_speed_max_mps < 5.0
    assert diagnostics.response_section_width_basis == (
        "declared_ambient_crossflow_angle"
    )
    assert diagnostics.response_section_width_min_m == pytest.approx(0.8)
    assert diagnostics.response_section_width_max_m == pytest.approx(0.8)
    assert diagnostics.speed_basis_invariant_violations == 0
    assert diagnostics.activation_invariant_violations == 0

    # Base VorLap AOA is still evaluated from Uinf - Vstruct, not from Uinf.
    ambient_aoa_deg = np.degrees(np.arctan2(4.0, 3.0))
    assert np.max(np.abs(source.aoa_deg - ambient_aoa_deg)) > 0.0
    np.testing.assert_allclose(
        source.aoa_deg,
        np.degrees(np.arctan2(relative_velocity[:, 1], relative_velocity[:, 0])),
    )
    expected_relative_speed = np.hypot(
        relative_velocity[:, 0],
        relative_velocity[:, 1],
    )
    np.testing.assert_allclose(
        diagnostics.response_relative_crossflow_speed_min_mps,
        np.min(expected_relative_speed),
    )

    summary = controller.response_diagnostics_summary
    assert summary["response_model_schema_version"] == 2
    assert summary["aerodynamic_speed_basis"] == "declared_ambient_crossflow"
    assert summary["response_section_width_basis"] == (
        "declared_ambient_crossflow_angle"
    )
    assert summary["response_declared_ambient_crossflow_speed_min_mps"] == 5.0
    assert summary["response_declared_ambient_crossflow_speed_max_mps"] == 5.0
    assert summary["response_relative_crossflow_speed_min_mps"] < 5.0
    assert summary["response_relative_crossflow_speed_max_mps"] == 5.0
    assert summary["speed_basis_invariant_violations"] == 0
    assert summary["activation_invariant_violations"] == 0

    restored = _build_controller(
        afft=afft,
        force_scale=0.0,
        response_model=_response_model(
            schema_version=2,
            aerodynamic_speed_basis="declared_ambient_crossflow",
        ),
    )
    restored.restore_checkpoint_state(
        json.loads(json.dumps(controller.checkpoint_state()))
    )
    assert restored.response_diagnostics_summary == summary


def test_schema_v1_controller_summary_cache_and_checkpoint_layout_are_frozen():
    legacy = _build_controller(response_model=_response_model())
    compatibility = _build_controller(response_model=_response_model())
    relative_velocity = np.tile([5.0, 0.0, 0.0], (2, 1))
    structural_velocity = np.zeros((2, 3), dtype=float)
    base_inputs = {
        "velocity": relative_velocity,
        "structural_velocity": structural_velocity,
        "time": 0.0,
        "azimuth_deg": 0.0,
    }
    legacy_force = legacy.step(base_inputs)
    compatibility_force = compatibility.step(
        {
            **base_inputs,
            "declared_ambient_velocity": np.tile([99.0, 1.0, 0.0], (2, 1)),
        }
    )
    np.testing.assert_array_equal(compatibility_force, legacy_force)

    expected_summary_fields = (
        "model_type",
        "samples",
        "gain_limited_steps",
        "force_limited_steps",
        "gain_limited_node_samples",
        "force_limited_node_samples",
        "max_gain_limited_nodes",
        "max_force_limited_nodes",
        "max_amplitude_ratio",
        "max_abs_modal_power_w",
        "max_response_nodal_force_n",
        "target_power_limited_steps",
        "generalized_force_limited_steps",
        "nodal_force_limited_steps",
        "nodal_force_limited_node_samples",
        "max_nodal_force_limited_nodes",
        "max_amplitude_estimate_m",
        "max_abs_target_cycle_mean_power_w",
        "max_abs_applied_target_cycle_mean_power_w",
        "max_abs_instantaneous_modal_power_w",
        "max_abs_instantaneous_structural_power_w",
        "max_abs_realized_modal_power_w",
        "max_abs_realized_structural_power_w",
    )
    assert tuple(legacy.response_diagnostics_summary) == expected_summary_fields
    assert legacy.response_diagnostics_summary["model_type"] == (
        "modal_negative_damping"
    )
    assert compatibility.checkpoint_state() == legacy.checkpoint_state()

    checkpoint = legacy.checkpoint_state()
    assert tuple(checkpoint["cached_step"]) == (
        "time_s",
        "azimuth_deg",
        "relative_velocity_mps",
        "structural_velocity_mps",
        "force_n",
    )
    assert tuple(checkpoint["response_model_state"]) == (
        "schema",
        "config_fingerprint",
        "last_time_s",
        "bandpass_filter",
        "qdot_mean_square_m2ps2",
        "last_input_signature",
        "last_result",
    )
    restored = _build_controller(response_model=_response_model())
    restored.restore_checkpoint_state(copy.deepcopy(checkpoint))
    assert restored.checkpoint_state() == checkpoint


def test_schema_v2_controller_restore_rejections_are_transactional():
    controller = _build_controller(
        response_model=_response_model(
            schema_version=2,
            aerodynamic_speed_basis="declared_ambient_crossflow",
        )
    )
    ambient = np.tile([3.0, 4.0, 12.0], (2, 1))
    structural = np.array([[0.0, 0.1, 0.0], [0.0, 0.2, 0.0]])
    controller.step(
        {
            "velocity": ambient - structural,
            "structural_velocity": structural,
            "declared_ambient_velocity": ambient,
            "time": 0.0,
            "azimuth_deg": 0.0,
        }
    )
    before = copy.deepcopy(controller.checkpoint_state())

    corruptions = []
    bad_schema = copy.deepcopy(before)
    bad_schema["response_model_state"]["response_model_schema_version"] = 1
    corruptions.append((bad_schema, "schema-version mismatch"))
    bad_basis = copy.deepcopy(before)
    bad_basis["response_model_state"]["aerodynamic_speed_basis"] = (
        "relative_crossflow"
    )
    corruptions.append((bad_basis, "speed-basis mismatch"))
    bad_summary_basis = copy.deepcopy(before)
    bad_summary_basis["response_diagnostics_summary"][
        "aerodynamic_speed_basis"
    ] = "relative_crossflow"
    corruptions.append((bad_summary_basis, "aerodynamic_speed_basis mismatch"))
    missing_ambient_cache = copy.deepcopy(before)
    missing_ambient_cache["cached_step"].pop("declared_ambient_velocity_mps")
    corruptions.append((missing_ambient_cache, "requires cached ambient velocity"))

    for corrupted, message in corruptions:
        with pytest.raises(ValueError, match=message):
            controller.restore_checkpoint_state(corrupted)
        assert controller.checkpoint_state() == before


def test_schema_v2_controller_requires_declared_ambient_velocity():
    controller = _build_controller(
        response_model=_response_model(
            schema_version=2,
            aerodynamic_speed_basis="declared_ambient_crossflow",
        )
    )
    with pytest.raises(KeyError, match="declared_ambient_velocity"):
        controller.step(
            {
                "velocity": np.tile([5.0, 0.0, 0.0], (2, 1)),
                "structural_velocity": np.zeros((2, 3)),
                "time": 0.0,
                "azimuth_deg": 0.0,
            }
        )


def test_step_suppresses_dc_mean_load_by_default():
    afft = make_constant_airfoil_fft(n_freq=1, cl_dc=1.0, cd_dc=2.0)
    controller = _build_controller(afft=afft)
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    forces = controller.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})

    np.testing.assert_allclose(forces, 0.0, atol=1e-12)


def test_step_keeps_oscillatory_cl_cd_loads_when_mean_is_suppressed():
    afft = make_constant_airfoil_fft(n_freq=2, cl_dc=10.0, cd_dc=20.0, cl_h1_amp=1.0, cd_h1_amp=0.5)
    controller = _build_controller(afft=afft)
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    forces = controller.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})

    assert np.max(np.abs(forces)) > 0.0


def test_step_can_include_dc_mean_load_explicitly():
    afft = make_constant_airfoil_fft(n_freq=1, cl_dc=1.0, cd_dc=2.0)
    no_mean = _build_controller(afft=afft, include_mean_load=False)
    with_mean = _build_controller(afft=afft, include_mean_load=True)
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    no_mean_forces = no_mean.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})
    with_mean_forces = with_mean.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})

    np.testing.assert_allclose(no_mean_forces, 0.0, atol=1e-12)
    assert np.max(np.abs(with_mean_forces)) > 0.0


def test_step_can_raise_on_source_bounds_violation():
    controller = _build_controller(source_bounds_policy="raise")
    velocities = np.array([[2.0e8, 0.0, 0.0], [2.0e8, 0.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="outside source grid"):
        controller.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})


def test_step_records_source_coverage_state():
    controller = _build_controller()
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    controller.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})
    diagnostics = controller.last_source_diagnostics
    state = controller.last_source_state

    assert isinstance(diagnostics, ControllerSourceState)
    assert diagnostics.node_count == 2
    assert diagnostics.node_ids.tolist() == ["BLD_1_0.000000", "BLD_1_1.000000"]
    assert diagnostics.component_ids.tolist() == ["blade", "blade"]
    assert diagnostics.source_airfoil_ids.tolist() == ["default", "default"]
    assert diagnostics.reynolds_models.tolist() == ["", ""]
    np.testing.assert_allclose(diagnostics.aoa_deg, 0.0)
    assert np.all(diagnostics.reynolds > 0.0)
    assert np.all(diagnostics.freq_max_hz > 0.0)
    assert np.all(diagnostics.source_aoa_min_deg <= diagnostics.aoa_deg)
    assert np.all(diagnostics.source_aoa_max_deg >= diagnostics.aoa_deg)
    assert np.all(diagnostics.source_force_coefficient_max_abs >= diagnostics.force_coefficient_abs)

    assert state is not None
    assert state["node_ids"].tolist() == ["BLD_1_0.000000", "BLD_1_1.000000"]
    assert state["component_ids"].tolist() == ["blade", "blade"]
    assert state["source_airfoil_ids"].tolist() == ["default", "default"]
    assert state["reynolds_models"].tolist() == ["", ""]
    np.testing.assert_allclose(state["aoa_deg"], 0.0)
    assert np.all(state["reynolds"] > 0.0)
    assert np.all(state["freq_max_hz"] > 0.0)
    assert np.all(state["source_aoa_min_deg"] <= state["aoa_deg"])
    assert np.all(state["source_aoa_max_deg"] >= state["aoa_deg"])
    assert np.all(state["source_force_coefficient_max_abs"] >= state["force_coefficient_abs"])

    diagnostics.aoa_deg[:] = 99.0
    state["aoa_deg"][:] = 88.0
    fresh = controller.last_source_diagnostics
    assert fresh is not None
    np.testing.assert_allclose(fresh.aoa_deg, 0.0)


def test_step_discards_oscillatory_tones_above_fixed_frequency_cap():
    afft = make_constant_airfoil_fft(
        n_freq=3,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=1.0,
        cd_h1_amp=0.5,
        st_h1=0.5,
    )
    afft.CL_ST[:, :, 2] = 2.0
    afft.CD_ST[:, :, 2] = 2.0
    afft.CL_Amp[:, :, 2] = 3.0
    afft.CD_Amp[:, :, 2] = 1.5
    velocities = np.full((2, 3), [2.0, 0.0, 0.0], dtype=float)

    unfiltered = _build_controller(afft=afft, n_freq_depth=3)
    unfiltered_forces = unfiltered.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})
    unfiltered_state = unfiltered.last_source_diagnostics
    assert unfiltered_state is not None
    cutoff_hz = 0.625 * float(np.max(unfiltered_state.freq_max_hz))

    filtered = _build_controller(
        afft=afft,
        n_freq_depth=3,
        source_frequency_cap_hz=cutoff_hz,
    )
    filtered_forces = filtered.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})
    state = filtered.last_source_diagnostics

    assert state is not None
    assert np.all(state.freq_max_hz <= cutoff_hz)
    np.testing.assert_allclose(state.unfiltered_freq_max_hz, unfiltered_state.freq_max_hz)
    np.testing.assert_array_equal(state.frequency_truncated_count, [2, 2])
    assert np.max(np.abs(filtered_forces)) < np.max(np.abs(unfiltered_forces))


@pytest.mark.parametrize("cap", [0.0, -1.0, np.inf, np.nan])
def test_controller_rejects_invalid_source_frequency_cap(cap):
    with pytest.raises(ValueError, match="source_frequency_cap_hz"):
        _build_controller(source_frequency_cap_hz=cap)


def test_controller_setup_does_not_build_unused_airfoil_interpolator_cache(monkeypatch):
    afft = make_constant_airfoil_fft(n_freq=2)

    def fail_unused_cache_build():
        raise AssertionError("controller setup should use the vectorized bilinear path directly")

    monkeypatch.setattr(afft, "_cache_interpolators", fail_unused_cache_build, raising=False)

    controller = _build_controller(afft=afft)
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    forces = controller.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})

    assert forces.shape == (2, 3)
    assert np.isfinite(forces).all()


def test_exact_input_cache_skips_recompute(monkeypatch):
    controller = _build_controller()
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    import vorlap.interpolation as interpolation

    calls = {"count": 0}
    original = interpolation.interpolate_fft_spectrum_batch

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(interpolation, "interpolate_fft_spectrum_batch", wrapped)

    first = controller.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0})
    first_calls = calls["count"]
    assert first_calls > 0
    assert controller.cache_stats == {"hits": 0, "misses": 1}

    second = controller.step({"velocity": velocities.copy(), "time": 0.0, "azimuth_deg": 0.0})
    assert calls["count"] == first_calls
    assert second is not first
    np.testing.assert_allclose(second, first, atol=1e-12)
    assert controller.cache_stats == {"hits": 1, "misses": 1}

    controller.step({"velocity": velocities, "time": 0.1, "azimuth_deg": 0.0})
    assert calls["count"] > first_calls
    assert controller.cache_stats == {"hits": 1, "misses": 2}


def test_phase_continuous_path_matches_legacy_at_constant_frequency():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=1.0,
        cd_h1_amp=0.5,
        st_h1=0.4,
    )
    legacy = _build_controller(afft=afft)
    continuous = _build_controller(
        afft=afft,
        response_model=_response_model(aerodynamic_damping_gain_h1=0.0),
    )
    relative_velocity = np.tile([2.0, 0.0, 0.0], (2, 1))
    structural_velocity = np.zeros((2, 3), dtype=float)

    for time_s in (0.0, 0.07, 0.19, 0.41):
        legacy_force = legacy.step(
            {
                "velocity": relative_velocity,
                "time": time_s,
                "azimuth_deg": 0.0,
            }
        )
        continuous_force = continuous.step(
            {
                "velocity": relative_velocity,
                "structural_velocity": structural_velocity,
                "time": time_s,
                "azimuth_deg": 0.0,
            }
        )
        np.testing.assert_allclose(
            continuous_force,
            legacy_force,
            rtol=0.0,
            atol=2.0e-14,
        )


def test_phase_continuous_path_trapezoidally_integrates_changing_frequency():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=1.0,
        cd_h1_amp=0.0,
        st_h1=0.2,
    )
    continuous = _build_controller(
        afft=afft,
        response_model=_response_model(aerodynamic_damping_gain_h1=0.0),
    )
    legacy = _build_controller(afft=afft)
    structural_velocity = np.zeros((2, 3), dtype=float)
    first_velocity = np.tile([1.0, 0.0, 0.0], (2, 1))
    second_velocity = np.tile([2.0, 0.0, 0.0], (2, 1))
    continuous.step(
        {
            "velocity": first_velocity,
            "structural_velocity": structural_velocity,
            "time": 0.0,
            "azimuth_deg": 0.0,
        }
    )
    legacy.step({"velocity": first_velocity, "time": 0.0, "azimuth_deg": 0.0})

    continuous_force = continuous.step(
        {
            "velocity": second_velocity,
            "structural_velocity": structural_velocity,
            "time": 0.1,
            "azimuth_deg": 0.0,
        }
    )
    legacy_force = legacy.step(
        {"velocity": second_velocity, "time": 0.1, "azimuth_deg": 0.0}
    )

    diameter_m = 0.12
    first_frequency_hz = 0.2 * 1.0 / diameter_m
    second_frequency_hz = 0.2 * 2.0 / diameter_m
    expected_angle_rad = np.pi * (first_frequency_hz + second_frequency_hz) * 0.1
    expected_dynamic_force_n = 0.5 * 2.0**2
    expected_normal_force_n = expected_dynamic_force_n * np.cos(expected_angle_rad)
    np.testing.assert_allclose(
        continuous_force,
        np.tile([0.0, expected_normal_force_n, 0.0], (2, 1)),
        rtol=0.0,
        atol=2.0e-14,
    )
    assert not np.allclose(continuous_force, legacy_force, rtol=0.0, atol=1.0e-8)
    phase_state = continuous.checkpoint_state()["phase_state"]
    np.testing.assert_allclose(
        np.asarray(phase_state["angle_cl_rad"])[:, 1],
        expected_angle_rad,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_response_and_phase_checkpoint_restart_matches_uninterrupted_run():
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=0.7,
        cd_h1_amp=0.2,
        st_h1=0.3,
    )
    uninterrupted = _build_controller(
        afft=afft,
        response_model=_response_model(),
    )
    first = {
        "velocity": np.tile([5.0, 0.0, 0.0], (2, 1)),
        "structural_velocity": np.array(
            [[0.0, 0.05, 0.0], [0.0, 0.1, 0.0]], dtype=float
        ),
        "time": 0.0,
        "azimuth_deg": 0.0,
    }
    second = {
        "velocity": np.tile([5.2, 0.0, 0.0], (2, 1)),
        "structural_velocity": np.array(
            [[0.0, 0.06, 0.0], [0.0, 0.12, 0.0]], dtype=float
        ),
        "time": 0.1,
        "azimuth_deg": 0.0,
    }
    future = {
        "velocity": np.tile([4.8, 0.0, 0.0], (2, 1)),
        "structural_velocity": np.array(
            [[0.0, 0.04, 0.0], [0.0, 0.08, 0.0]], dtype=float
        ),
        "time": 0.2,
        "azimuth_deg": 0.0,
    }
    uninterrupted.step(first)
    accepted = uninterrupted.step(second)
    checkpoint = json.loads(json.dumps(uninterrupted.checkpoint_state()))

    restarted = _build_controller(
        afft=afft,
        response_model=_response_model(),
    )
    restarted.restore_checkpoint_state(checkpoint)
    restored_source = restarted.last_source_diagnostics
    original_source = uninterrupted.last_source_diagnostics
    assert restored_source is not None
    assert original_source is not None
    np.testing.assert_allclose(
        restored_source.total_force_norm_n,
        original_source.total_force_norm_n,
    )
    assert restarted.last_response_diagnostics == uninterrupted.last_response_diagnostics
    duplicate = restarted.step(second)
    np.testing.assert_allclose(duplicate, accepted, rtol=0.0, atol=1.0e-14)

    uninterrupted_future = uninterrupted.step(future)
    restarted_future = restarted.step(future)
    np.testing.assert_allclose(
        restarted_future,
        uninterrupted_future,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert restarted.last_response_diagnostics == uninterrupted.last_response_diagnostics
    assert (
        restarted.checkpoint_state()["phase_state"]
        == uninterrupted.checkpoint_state()["phase_state"]
    )


def test_copy_false_returns_read_only_internal_view_and_reuses_identity():
    controller = _build_controller()
    velocities = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)

    first = controller.step({"velocity": velocities, "time": 0.0, "azimuth_deg": 0.0}, copy=False)
    second = controller.step({"velocity": velocities.copy(), "time": 0.0, "azimuth_deg": 0.0}, copy=False)

    assert second is first
    assert not first.flags.writeable
    assert controller.cache_hits == 1
    assert controller.cache_misses == 1

    controller.reset_cache_stats()
    assert controller.cache_stats == {"hits": 0, "misses": 0}


def test_constructor_validates_node_ids_length():
    with pytest.raises(ValueError, match="node_ids length must match"):
        _build_controller(node_ids=["BLD_1_0.000000"])


def test_constructor_validates_force_scale():
    with pytest.raises(ValueError, match="force_scale must be a finite, non-negative scalar"):
        _build_controller(force_scale=-1.0)


def test_step_validates_bad_velocity_input():
    controller = _build_controller()

    with pytest.raises(ValueError, match="Velocity vector must have length"):
        controller.step({"velocity": np.array([1.0, 2.0]), "time": 0.0, "azimuth_deg": 0.0})

    with pytest.raises(ValueError, match="contains non-finite"):
        controller.step(
            {
                "velocity": np.array([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
                "time": 0.0,
                "azimuth_deg": 0.0,
            }
        )


def test_swap_array_adapter_round_trip_with_controller():
    controller = _build_controller()
    swap = np.zeros(14, dtype=float)
    layout = {
        "velocity": SwapBlockSpec(offset=0, shape=(2, 3)),
        "time": SwapBlockSpec(offset=6, shape=()),
        "azimuth_deg": SwapBlockSpec(offset=7, shape=()),
        "force": SwapBlockSpec(offset=8, shape=(2, 3)),
    }
    adapter = SwapArrayAdapter.from_layout(swap, layout)

    velocity = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    adapter.write("velocity", velocity)
    adapter.write("time", 0.0)
    adapter.write("azimuth_deg", 0.0)

    direct = controller.step({"velocity": velocity, "time": 0.0, "azimuth_deg": 0.0})
    via_swap = controller.step_from_swap(adapter)

    np.testing.assert_allclose(via_swap, direct, atol=1e-12)
    np.testing.assert_allclose(adapter.read("force"), direct, atol=1e-12)


def test_swap_array_adapter_validates_overlap_and_bounds():
    with pytest.raises(ValueError, match="overlaps"):
        SwapArrayAdapter(
            np.zeros(10, dtype=float),
            {
                "a": SwapBlockSpec(offset=0, shape=(6,)),
                "b": SwapBlockSpec(offset=5, shape=(3,)),
            },
        )

    with pytest.raises(ValueError, match="exceeds avr_swap length"):
        SwapArrayAdapter(
            np.zeros(10, dtype=float),
            {
                "a": SwapBlockSpec(offset=9, shape=(2,)),
            },
        )
