import copy
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import make_component, make_constant_airfoil_fft, make_viv_params
from vorlap import (
    QBladeController,
    QBladeModalPowerDiagnostics,
    QBladeModalPowerModel,
)
from vorlap.computations import rotate_vector
from vorlap.interpolation import AirfoilRangeWarning
from vorlap import qblade_runtime
from vorlap.qblade_external_config import (
    build_qblade_external_config as config_build_qblade_external_config,
    patch_qblade_structural_definition as config_patch_qblade_structural_definition,
    patch_qblade_turbine_definition as config_patch_qblade_turbine_definition,
    write_qblade_external_config as config_write_qblade_external_config,
)
from vorlap.qblade_external_table import (
    build_external_library_table_spec as table_build_external_library_table_spec,
    build_hvawt_response_table_spec,
)
from vorlap.qblade_geometry_audit import (
    GEOMETRY_AUDIT_COLUMNS as audit_GEOMETRY_AUDIT_COLUMNS,
    build_qblade_geometry_audit_rows as audit_build_qblade_geometry_audit_rows,
    write_qblade_geometry_audit as audit_write_qblade_geometry_audit,
)
from vorlap.qblade_source_coverage import (
    SOURCE_COVERAGE_COLUMNS as coverage_SOURCE_COVERAGE_COLUMNS,
    SourceCoverageRow as CoverageSourceCoverageRow,
)
from vorlap.qblade_runtime import (
    GEOMETRY_AUDIT_COLUMNS,
    SOURCE_COVERAGE_COLUMNS,
    SourceCoverageRow,
    VorLapQBladeRuntime,
    build_external_library_table_spec,
    build_qblade_geometry_audit_rows,
    build_qblade_external_config,
    patch_qblade_structural_definition,
    patch_qblade_turbine_definition,
    write_qblade_geometry_audit,
    write_qblade_external_config,
)


def _unit(vector):
    arr = np.asarray(vector, dtype=float)
    return arr / np.linalg.norm(arr)


def _component_with_distinct_3d_nodes():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.id = "BLD_1"
    component.chord[:] = np.array([1.0, 1.5], dtype=float)
    component.chord_vector[0] = _unit([1.0, 0.0, 0.25])
    component.normal_vector[0] = _unit([0.25, 0.0, -1.0])
    component.chord_vector[1] = _unit([0.4, 0.8, 0.2])
    component.normal_vector[1] = _unit([0.09759000729485331, 0.19518001458970663, -0.9759000729485331])
    return component


def _expected_dc_force(component, node_index, relative_velocity, azimuth_deg, *, cl_dc, cd_dc, density=1.0):
    axis = np.array([0.0, 0.0, 1.0], dtype=float)
    chord_unit = rotate_vector(component.chord_vector[node_index], axis, azimuth_deg)
    normal_unit = rotate_vector(component.normal_vector[node_index], axis, azimuth_deg)
    v_chord = float(np.dot(relative_velocity, chord_unit))
    v_normal = float(np.dot(relative_velocity, normal_unit))
    v_eff = float(np.hypot(v_chord, v_normal))
    local_length = 1.0
    q = 0.5 * density * v_eff**2 * float(component.chord[node_index]) * local_length
    return q * (cd_dc * chord_unit + cl_dc * normal_unit)


def _write_config(path, config):
    write_qblade_external_config(str(path), config)
    return path


def _patch_config_runtime_io(monkeypatch, component, afft):
    def fake_convert(sim_arg, **kwargs):
        viv_params = make_viv_params()
        viv_params.inflow_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        viv_params.inflow_speeds = np.array([3.0], dtype=float)
        return [component], viv_params, ["BLD_1_0.000000", "BLD_1_1.000000"]

    def fake_load_airfoils(airfoil_arg):
        return {"default": afft}

    monkeypatch.setattr(qblade_runtime, "convert_qblade_to_vorlap_inputs", fake_convert)
    monkeypatch.setattr(qblade_runtime, "load_airfoil_fft_directory", fake_load_airfoils)


def _response_config(
    node_ids,
    *,
    enabled=True,
    force_direction="normal",
    schema_version=1,
    aerodynamic_speed_basis=None,
):
    config = {
        "type": "modal_negative_damping",
        "schema_version": schema_version,
        "enabled": enabled,
        "node_ids": list(node_ids),
        "mode_shape": [0.5, 1.0],
        "natural_frequency_hz": 1.0,
        "reference_length_m": 1.0,
        "reduced_velocity_activation": [2.0, 3.0, 4.0],
        "aerodynamic_damping_gain_h1": 2.0,
        "saturation_amplitude_ratio": 0.2,
        "rms_time_constant_s": 0.5,
        "max_abs_gain": 20.0,
        "max_abs_nodal_force_n": 1.0e5,
        "force_direction": force_direction,
        "projection_weights": [1.0, 2.0],
        "bandpass_damping_ratio": 0.2,
    }
    if aerodynamic_speed_basis is not None:
        config["aerodynamic_speed_basis"] = aerodynamic_speed_basis
    return config


def _modal_power_config(
    node_ids,
    *,
    enabled=True,
    limit_action="error",
    max_abs_target_power_w=1.0e4,
    max_abs_generalized_force_n=1.0e4,
    max_abs_nodal_force_n=1.0e4,
):
    return {
        "type": "modal_aerodynamic_power",
        "schema_version": 1,
        "enabled": enabled,
        "node_ids": list(node_ids),
        "mode_shape": [0.5, 1.0],
        "modal_mass_weights_kg": [2.0, 1.0],
        "aerodynamic_tributary_weights": [1.0, 2.0],
        "natural_frequency_hz": 1.0,
        "power_curve": {
            "amplitude_m": [0.0, 0.1, 0.5, 1.0],
            "power_w": [0.0, 5.0, 50.0, -10.0],
            "source_label": "synthetic runtime integration curve",
            "source_sha256": "0" * 64,
        },
        "rms_time_constant_s": 0.5,
        "max_abs_target_power_w": max_abs_target_power_w,
        "max_abs_generalized_force_n": max_abs_generalized_force_n,
        "max_abs_nodal_force_n": max_abs_nodal_force_n,
        "bandpass_damping_ratio": 0.2,
        "force_direction": "chord",
        "limit_action": limit_action,
    }


def test_modal_negative_damping_schema_v2_config_is_explicit_and_fail_closed():
    node_ids = ("root", "tip")
    legacy_raw = _response_config(node_ids)
    legacy_raw.pop("schema_version")
    legacy = qblade_runtime._response_model_from_config(legacy_raw, node_ids)
    assert legacy.schema_version == 1
    assert legacy.aerodynamic_speed_basis == "relative_crossflow"

    valid_v2 = _response_config(
        node_ids,
        schema_version=2,
        aerodynamic_speed_basis="declared_ambient_crossflow",
    )
    valid_v2["response_section_width_m"] = 0.05
    ambient = qblade_runtime._response_model_from_config(valid_v2, node_ids)
    assert ambient.schema_version == 2
    assert ambient.aerodynamic_speed_basis == "declared_ambient_crossflow"
    np.testing.assert_array_equal(ambient.response_section_width_m, [0.05, 0.05])

    missing_basis = dict(valid_v2)
    missing_basis.pop("aerodynamic_speed_basis")
    with pytest.raises(ValueError, match="requires.*aerodynamic_speed_basis"):
        qblade_runtime._response_model_from_config(missing_basis, node_ids)

    noninteger_schema = dict(valid_v2, schema_version=2.0)
    with pytest.raises(ValueError, match="explicit integer 2"):
        qblade_runtime._response_model_from_config(noninteger_schema, node_ids)

    null_schema = dict(valid_v2, schema_version=None)
    with pytest.raises(ValueError, match="schema_version"):
        qblade_runtime._response_model_from_config(null_schema, node_ids)

    unsupported = dict(valid_v2, aerodynamic_speed_bais="typo")
    with pytest.raises(ValueError, match="unsupported field.*aerodynamic_speed_bais"):
        qblade_runtime._response_model_from_config(unsupported, node_ids)


def test_runtime_reexports_qblade_table_and_geometry_audit_helpers():
    assert build_external_library_table_spec is table_build_external_library_table_spec
    assert GEOMETRY_AUDIT_COLUMNS is audit_GEOMETRY_AUDIT_COLUMNS
    assert build_qblade_geometry_audit_rows is audit_build_qblade_geometry_audit_rows
    assert write_qblade_geometry_audit is audit_write_qblade_geometry_audit
    assert SOURCE_COVERAGE_COLUMNS is coverage_SOURCE_COVERAGE_COLUMNS
    assert SourceCoverageRow is CoverageSourceCoverageRow
    assert build_qblade_external_config is config_build_qblade_external_config
    assert patch_qblade_structural_definition is config_patch_qblade_structural_definition
    assert patch_qblade_turbine_definition is config_patch_qblade_turbine_definition
    assert write_qblade_external_config is config_write_qblade_external_config


def test_build_external_library_table_spec_matches_node_count():
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    spec = build_external_library_table_spec(node_ids)

    assert spec.swap_size == 15
    assert spec.swap_layout["time"].offset == 0
    assert spec.swap_layout["velocity"].shape == (2, 3)
    assert spec.swap_layout["force"].offset == 9
    assert spec.input_rows[0] == (0, "Time [s]")
    assert spec.input_rows[2] == (2, "LSS Azimuthal Pos. [deg]")
    assert spec.output_rows[0] == (9, "ADDFORCE", "BLD_1", 0.0, "X", False)


def test_build_external_library_table_spec_keeps_action_positions_per_node():
    spec = build_external_library_table_spec(["BLD_2_0.230455", "STR_1_3_0.172168"])
    assert spec.output_rows[0] == (9, "ADDFORCE", "BLD_2", 0.230455, "X", False)
    assert spec.output_rows[3] == (12, "ADDFORCE", "STR_1_3", 0.172168, "X", False)


def test_build_hvawt_response_table_spec_appends_rpm_and_positions_without_moving_forces():
    node_ids = [f"BLD_1_{index / 38.0:.6f}" for index in range(39)]
    spec = build_hvawt_response_table_spec(node_ids)

    assert spec.swap_size == 355
    assert spec.swap_layout["force"].offset == 120
    assert spec.swap_layout["lss_rpm"].offset == 237
    assert spec.swap_layout["position_global_m"].offset == 238
    assert spec.swap_layout["position_global_m"].shape == (39, 3)
    assert spec.input_rows[120] == (237, "LSS Rpm [rpm]")
    assert spec.input_rows[121] == (
        238,
        "X_g Pos. BLD_1 pos 0.000 [m]",
    )
    assert spec.output_rows[0] == (120, "ADDFORCE", "BLD_1", 0.0, "X", False)


def test_build_external_library_table_spec_rejects_duplicate_node_ids():
    with pytest.raises(ValueError, match="Duplicate external-library table node id"):
        build_external_library_table_spec(["BLD_1_0.000000", "BLD_1_0.000000"])


def test_build_external_library_table_spec_rejects_out_of_range_node_positions():
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        build_external_library_table_spec(["BLD_1_1.500000"])
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        build_external_library_table_spec(["STR_1_2_-0.100000"])
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        build_external_library_table_spec(["TWR_1_1.250000"])


def test_geometry_audit_rows_record_swap_and_controller_mapping(tmp_path):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.id = "BLD_1"
    requested = ["BLD_1_1.000000", "BLD_1_0.000000"]

    rows = build_qblade_geometry_audit_rows([component], requested)

    assert [row["requested_node_id"] for row in rows] == requested
    assert tuple(rows[0]) == GEOMETRY_AUDIT_COLUMNS
    assert [row["controller_index"] for row in rows] == [1, 0]
    assert rows[0]["force_swap_x"] == 9
    assert rows[0]["velocity_swap_x"] == 3
    assert rows[0]["action_id"] == "BLD_1"
    assert rows[0]["action_position"] == 1.0
    assert rows[0]["vorlap_z_m"] == 2.0
    assert rows[0]["basis_status"] == "ok"
    assert rows[0]["qblade_reported_available"] is False
    np.testing.assert_allclose(
        [rows[0]["chord_dot_normal"], rows[0]["chord_dot_span"], rows[0]["normal_dot_span"]],
        np.zeros(3),
        atol=1.0e-12,
    )
    assert rows[0]["basis_determinant"] == pytest.approx(1.0)

    audit_path = tmp_path / "vorlap_qblade_geometry_audit.csv"
    write_qblade_geometry_audit(str(audit_path), [component], requested)
    text = audit_path.read_text(encoding="utf-8")
    assert "requested_node_id" in text
    assert "BLD_1_1.000000" in text


def test_patch_qblade_turbine_definition_inserts_external_library_section():
    text = """----------------------------------------Turbine Controller-----------------------------------------------------------\n1                                                  CONTROLLERTYPE     - the type of turbine controller 0 = none, 1 = BLADED, 2 = DTU, 3 = TUB\nlibexample_controller                            CONTROLLERFILE     - the controller file name, WITHOUT file ending (.dll or .so ) - leave blank if unused\nControl/example_controller.in                      PARAMETERFILE      - the controller parameter file name (leave blank if unused)\n"""
    patched = patch_qblade_turbine_definition(
        text,
        library_stem="libvorlap_qblade_bridge",
        function_name="update",
        swap_size=15,
        parameter_file="Control/vorlap_qblade_external.json",
    )

    assert "CONTROLLERTYPE" in patched
    assert "1                                                  CONTROLLERTYPE" not in patched
    assert "LIBFILE_1" in patched
    assert "libvorlap_qblade_bridge" in patched
    assert "LIBARRAYSIZE_1" in patched
    assert "15" in patched
    assert "LIBPARAMETERFILE_1" in patched


def test_patch_qblade_structural_definition_appends_external_tables():
    text = "TWR_1.0\n"
    spec = build_external_library_table_spec(["BLD_1_0.000000"])
    patched = patch_qblade_structural_definition(text, table_spec=spec)

    assert "EXTERNAL_1_IN" in patched
    assert "EXTERNAL_1_OUT" in patched
    assert '"Time [s]"' in patched
    assert "ADDFORCE" in patched


def test_runtime_updates_forces_in_place_and_reuses_cache(monkeypatch):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    viv_params = make_viv_params()
    viv_params.output_time = np.array([0.0, 0.25], dtype=float)
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["timestep"].offset] = 0.25
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 6] = np.array(
        [2.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32
    )

    import vorlap.interpolation as interpolation

    calls = {"count": 0}
    original = interpolation.interpolate_fft_spectrum_batch

    def wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(interpolation, "interpolate_fft_spectrum_batch", wrapped)

    first = runtime.update(swap)
    assert first.shape == (2, 3)
    assert np.isfinite(first).all()
    assert calls["count"] > 0

    first_count = calls["count"]
    second = runtime.update(swap.copy())
    assert calls["count"] == first_count
    assert controller.cache_hits == 1
    assert controller.cache_misses == 1
    assert "cache=1/1" in runtime.update_message()
    np.testing.assert_allclose(second, first)
    np.testing.assert_allclose(
        swap[spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 6],
        first.reshape(-1),
    )


def test_runtime_returns_swap_order_forces_when_controller_order_differs():
    class FakeController:
        def __init__(self):
            self.viv_params = make_viv_params()
            self.viv_params.inflow_speeds = np.array([0.0], dtype=float)
            self.node_ids = ("controller_0", "controller_1", "controller_2")
            self.force_scale = 1.0
            self.cache_stats = {"hits": 0, "misses": 1}
            self.seen_velocity = None
            self.seen_structural_velocity = None

        def step(self, node_data, *, copy=False):
            self.seen_velocity = np.array(node_data["velocity"], copy=True)
            self.seen_structural_velocity = np.array(
                node_data["structural_velocity"], copy=True
            )
            return np.array(
                [
                    [1.0, 10.0, 100.0],
                    [2.0, 20.0, 200.0],
                    [3.0, 30.0, 300.0],
                ],
                dtype=float,
            )

    layout = {
        "time": 0,
        "timestep": 1,
        "azimuth_deg": 2,
        "velocity": (3, (3, 3)),
        "force": (12, (3, 3)),
    }
    controller = FakeController()
    runtime = VorLapQBladeRuntime(
        controller,
        layout,
        swap_to_controller_idx=[1, 2, 0],
        controller_to_swap_idx=[2, 0, 1],
    )
    swap = np.zeros(21, dtype=float)
    swap[3:12] = np.array(
        [
            [100.0, 101.0, 102.0],
            [200.0, 201.0, 202.0],
            [300.0, 301.0, 302.0],
        ],
        dtype=float,
    ).reshape(-1)

    returned = runtime.update(swap)

    expected_controller_velocity = -np.array(
        [
            [300.0, 301.0, 302.0],
            [100.0, 101.0, 102.0],
            [200.0, 201.0, 202.0],
        ],
        dtype=float,
    )
    expected_swap_forces = np.array(
        [
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [1.0, 10.0, 100.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(controller.seen_velocity, expected_controller_velocity)
    np.testing.assert_allclose(
        controller.seen_structural_velocity,
        -expected_controller_velocity,
    )
    np.testing.assert_allclose(swap[12:21].reshape(3, 3), expected_swap_forces)
    np.testing.assert_allclose(returned, expected_swap_forces)

    # Legacy/custom disabled controllers need not implement the new causal
    # response checkpoint API.
    checkpoint = runtime.checkpoint_state()
    assert checkpoint["controller"] is None
    runtime.restore_checkpoint_state(
        {"source_coverage": {}, "message": "legacy checkpoint"}
    )
    assert runtime.message == "legacy checkpoint"


def test_runtime_replay_matches_hand_computed_3d_dc_force():
    component = _component_with_distinct_3d_nodes()
    component.shape_xyz = component.shape_xyz[:1].copy()
    component.shape_xyz_global = component.shape_xyz_global[:1].copy()
    component.chord = component.chord[:1].copy()
    component.twist = component.twist[:1].copy()
    component.thickness = component.thickness[:1].copy()
    component.offset = component.offset[:1].copy()
    component.airfoil_ids = component.airfoil_ids[:1]
    component.chord_vector = component.chord_vector[:1].copy()
    component.normal_vector = component.normal_vector[:1].copy()
    cl_dc = 1.25
    cd_dc = 0.75
    afft = make_constant_airfoil_fft(n_freq=1, cl_dc=cl_dc, cd_dc=cd_dc)
    viv_params = make_viv_params()
    viv_params.inflow_vec = np.array([1.0, 0.0, 0.0], dtype=float)
    viv_params.inflow_speeds = np.array([3.0], dtype=float)
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000"],
        n_freq_depth=1,
        include_mean_load=True,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    structural_velocity = np.array([0.2, -0.4, 0.1], dtype=np.float32)
    azimuth = np.float32(30.0)
    swap[spec.swap_layout["time"].offset] = np.float32(0.125)
    swap[spec.swap_layout["timestep"].offset] = np.float32(0.25)
    swap[spec.swap_layout["azimuth_deg"].offset] = azimuth
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 3] = structural_velocity

    forces = runtime.update(swap)

    relative_velocity = np.array([3.0, 0.0, 0.0], dtype=float) - structural_velocity.astype(float)
    expected = _expected_dc_force(
        component,
        0,
        relative_velocity,
        float(azimuth),
        cl_dc=cl_dc,
        cd_dc=cd_dc,
    )
    np.testing.assert_allclose(forces[0], expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        swap[spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 3],
        expected.astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_runtime_uses_ambient_inflow_minus_structural_velocity():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2, cl_dc=0.0, cd_dc=0.0, cl_h1_amp=1.0, cd_h1_amp=0.5)
    viv_params = make_viv_params()
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["timestep"].offset] = 0.25
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0
    # Structural velocity is zero, so any non-zero force must come from ambient inflow.
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 6] = 0.0

    forces = runtime.update(swap)

    assert np.isfinite(forces).all()
    assert np.max(np.linalg.norm(forces, axis=1)) > 0.0


def test_runtime_accumulates_source_coverage_summary():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2, cl_dc=0.0, cd_dc=0.0, cl_h1_amp=1.0, cd_h1_amp=0.5)
    viv_params = make_viv_params()
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0

    runtime.update(swap)
    runtime.record_source_coverage_sample(0.01)
    rows = runtime.source_coverage_summary(0.01)

    assert len(rows) == 1
    row = rows[0]
    assert row["component"] == "blade"
    assert row["samples"] == 1
    assert row["node_samples"] == 2
    assert row["aoa_min_deg"] <= row["aoa_max_deg"]
    assert row["re_min"] <= row["re_max"]
    assert row["freq_max_hz"] > 0.0
    assert row["nyquist_hz"] == 50.0
    assert row["nyquist_ratio"] > 1.0
    typed = SourceCoverageRow.from_mapping(row)
    assert tuple(typed.as_dict()) == SOURCE_COVERAGE_COLUMNS
    assert typed.as_dict() == row
    assert row["max_nodal_force_n"] > 0.0
    assert row["re_out_of_range_count"] == 0
    assert row["aoa_out_of_range_count"] == 0
    assert row["force_coefficient_out_of_range_count"] == 0
    assert row["nonfinite_count"] == 0


def test_runtime_records_verified_lower_re_as_warning_not_violation():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(
        n_freq=2,
        re_grid=np.array([1.0e3, 1.0e4]),
        reynolds_model="replicated_reynolds_independent",
    )
    viv_params = make_viv_params()
    viv_params.inflow_speeds = np.array([1.0e-3])
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
        source_bounds_policy="raise",
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.zeros(spec.swap_size, dtype=np.float32)

    with pytest.warns(AirfoilRangeWarning, match="warning-only"):
        runtime.update(swap)
    runtime.record_source_coverage_sample(0.01)
    row = runtime.source_coverage_summary(0.01)[0]

    assert row["airfoil_id"] == "default"
    assert row["reynolds_model"] == "replicated_reynolds_independent"
    assert row["re_min"] < row["source_re_min"]
    assert row["re_lower_bound_warning_count"] == 2
    assert row["re_out_of_range_count"] == 0
    assert row["source_clip_count"] == 0


def test_runtime_coverage_distinguishes_retained_and_truncated_frequency_ranges():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2, cl_dc=0.0, cd_dc=0.0, cl_h1_amp=1.0, cd_h1_amp=0.5)
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=make_viv_params(),
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
        source_frequency_cap_hz=0.1,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.zeros(spec.swap_size, dtype=np.float32)

    runtime.update(swap)
    runtime.record_source_coverage_sample(0.01)
    row = runtime.source_coverage_summary(0.01)[0]

    assert row["unfiltered_freq_max_hz"] > row["freq_max_hz"]
    assert row["frequency_truncated_count"] == 4
    assert row["aliased_frequency_count"] == 0


def test_runtime_normalizes_ambient_inflow_direction():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=1)
    viv_params = make_viv_params()
    viv_params.inflow_vec = np.array([10.0, 0.0, 0.0], dtype=float)
    viv_params.inflow_speeds = np.array([2.0], dtype=float)
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=1,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["timestep"].offset] = 0.25
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0

    forces = runtime.update(swap)
    expected = controller.step(
        {
            "velocity": np.tile(np.array([2.0, 0.0, 0.0], dtype=float), (2, 1)),
            "time": 0.0,
            "azimuth_deg": 0.0,
        }
    )

    np.testing.assert_allclose(forces, expected, atol=1e-12)


def test_runtime_zeroes_forces_for_nonfinite_relaxation_inputs():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    viv_params = make_viv_params()
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.ones(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0
    swap[spec.swap_layout["velocity"].offset] = np.nan

    forces = runtime.update(swap)

    assert np.isfinite(forces).all()
    np.testing.assert_allclose(forces, 0.0)
    np.testing.assert_allclose(
        swap[spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 6],
        0.0,
    )
    assert "non-finite structural velocity" in runtime.update_message()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("time", -1.0, "implausible QBlade swap time"),
        ("time", 1.0e12, "implausible QBlade swap time"),
        ("timestep", -1.0, "implausible QBlade swap timestep"),
        ("timestep", 1.0e12, "implausible QBlade swap timestep"),
        ("azimuth_deg", 1.0e12, "implausible QBlade swap azimuth"),
        ("velocity", 2.0e3, "implausible structural velocity"),
    ],
)
def test_runtime_zeroes_stale_force_block_for_implausible_swap_values(field, value, message):
    component = make_component(n_nodes=1, airfoil_id="default")
    component.id = "BLD_1"
    afft = make_constant_airfoil_fft(n_freq=1)
    viv_params = make_viv_params()
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000"],
        n_freq_depth=1,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.ones(spec.swap_size, dtype=np.float32) * 55.0
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["timestep"].offset] = 0.0
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 3] = 0.0
    if field == "velocity":
        swap[spec.swap_layout["velocity"].offset] = value
    else:
        swap[spec.swap_layout[field].offset] = value

    forces = runtime.update(swap)

    np.testing.assert_allclose(forces, 0.0)
    np.testing.assert_allclose(
        swap[spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 3],
        0.0,
    )
    assert message in runtime.update_message()


def test_runtime_zeroes_stale_force_block_for_nonfinite_time():
    component = make_component(n_nodes=1, airfoil_id="default")
    component.id = "BLD_1"
    afft = make_constant_airfoil_fft(n_freq=1)
    viv_params = make_viv_params()
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000"],
        n_freq_depth=1,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.ones(spec.swap_size, dtype=np.float32) * 99.0
    swap[spec.swap_layout["time"].offset] = np.nan
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0

    forces = runtime.update(swap)

    np.testing.assert_allclose(forces, 0.0)
    np.testing.assert_allclose(
        swap[spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 3],
        0.0,
    )
    assert "non-finite time/timestep/azimuth" in runtime.update_message()


def test_runtime_clear_force_output_zeroes_stale_force_block():
    component = make_component(n_nodes=1, airfoil_id="default")
    component.id = "BLD_1"
    afft = make_constant_airfoil_fft(n_freq=1)
    viv_params = make_viv_params()
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000"],
        n_freq_depth=1,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(controller, spec.swap_layout)
    swap = np.ones(spec.swap_size, dtype=np.float32) * 77.0

    cleared = runtime.clear_force_output(swap, "forced clear from test")

    np.testing.assert_allclose(cleared, 0.0)
    np.testing.assert_allclose(
        swap[spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 3],
        0.0,
    )
    assert runtime.update_message() == "forced clear from test"


def test_build_qblade_external_config_and_write(tmp_path):
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
        sample_step=0.1,
    )
    assert config["swap_size"] == 9
    assert config["force_scale"] == 1.0
    assert config["include_mean_load"] is False
    assert config["source_bounds_policy"] == "warn"

    out_path = tmp_path / "vorlap_qblade_external.json"
    written = write_qblade_external_config(str(out_path), config)
    assert out_path.exists()
    assert written == str(out_path.resolve())


def test_build_qblade_external_config_preserves_relative_airfoil_dir():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="../VorLapAirfoils",
        node_ids=["BLD_1_0.000000"],
    )
    assert config["airfoil_dir"] == "../VorLapAirfoils"


def test_build_qblade_external_config_preserves_relative_sim_path():
    config = build_qblade_external_config(
        sim_path="../../baseline_reference.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
    )
    assert config["sim_path"] == "../../baseline_reference.sim"


def test_build_qblade_external_config_accepts_force_scale():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
        force_scale=100.0,
    )
    assert config["force_scale"] == 100.0


def test_build_qblade_external_config_accepts_include_mean_load():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
        include_mean_load=True,
        source_bounds_policy="raise",
    )
    assert config["include_mean_load"] is True
    assert config["source_bounds_policy"] == "raise"


def test_build_qblade_external_config_accepts_viv_parameter_overrides():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
        fluid_density=1.3,
        fluid_dynamicviscosity=1.7e-5,
        rotation_axis=(0.0, 0.0, 1.0),
        rotation_axis_offset=(1.0, 2.0, 3.0),
        inflow_vec=(1.0, 1.0, 0.0),
    )

    assert config["fluid_density"] == 1.3
    assert config["fluid_dynamicviscosity"] == 1.7e-5
    assert config["rotation_axis"] == [0.0, 0.0, 1.0]
    assert config["rotation_axis_offset"] == [1.0, 2.0, 3.0]
    assert config["inflow_vec"] == [1.0, 1.0, 0.0]


def test_build_qblade_external_config_accepts_debug():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
        debug=True,
    )
    assert config["debug"] is True


def test_build_qblade_external_config_accepts_log_file():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000"],
        log_file="/tmp/vorlap_qblade_debug.log",
    )
    assert config["log_file"] == "/tmp/vorlap_qblade_debug.log"


def test_build_qblade_external_config_accepts_source_parameter_dir():
    config = build_qblade_external_config(
        sim_path="../../baseline_reference.sim",
        airfoil_dir="../VorLapAirfoils",
        node_ids=["BLD_1_0.000000"],
        source_parameter_dir=".",
    )
    assert config["source_parameter_dir"] == "."


def test_build_qblade_external_config_accepts_portable_response_model():
    response_model = {
        "type": "modal_negative_damping",
        "schema_version": 1,
        "natural_frequency_hz": 0.67,
        "mode_shape": [0.0, 1.0],
    }
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        response_model=response_model,
    )

    assert config["response_model"] == response_model
    assert config["response_model"] is not response_model


def test_build_qblade_external_config_accepts_hvawt_response_model_and_355_table():
    node_ids = [f"BLD_1_{index / 38.0:.6f}" for index in range(39)]
    model = {"schema_version": 1, "type": "hvawt_two_mode_negative_damping"}
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=node_ids,
        force_scale=0.0,
        hvawt_response_model=model,
    )

    assert config["swap_size"] == 355
    assert config["force_scale"] == 0.0
    assert config["hvawt_response_model"] == model
    assert any(row["name"] == "lss_rpm" for row in config["swap_layout"])
    assert any(row["name"] == "position_global_m" for row in config["swap_layout"])


def test_build_qblade_external_config_rejects_two_response_models():
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_qblade_external_config(
            sim_path="/tmp/case.sim",
            airfoil_dir="/tmp/airfoils",
            node_ids=["BLD_1_0.000000"],
            response_model={"type": "one"},
            hvawt_response_model={"type": "two"},
        )


@pytest.mark.parametrize(
    "response_model",
    [[], {"bad": np.nan}, {"bad": object()}],
)
def test_build_qblade_external_config_rejects_nonportable_response_model(response_model):
    expected_error = TypeError if isinstance(response_model, list) else ValueError
    with pytest.raises(expected_error, match="response_model"):
        build_qblade_external_config(
            sim_path="/tmp/case.sim",
            airfoil_dir="/tmp/airfoils",
            node_ids=["BLD_1_0.000000"],
            response_model=response_model,
        )


def test_build_qblade_external_config_accepts_python_path_entries():
    config = build_qblade_external_config(
        sim_path="../../baseline_reference.sim",
        airfoil_dir="../VorLapAirfoils",
        node_ids=["BLD_1_0.000000"],
        python_path_entries=["/opt/vorlap", "../local_vorlap"],
    )
    assert config["python_path_entries"] == ["/opt/vorlap", "../local_vorlap"]


def test_checked_reference_config_uses_nominal_scale_and_relative_paths():
    repo = Path(__file__).resolve().parents[1]
    config_path = (
        repo
        / "examples"
        / "ReferenceHVAWTQBladeModel"
        / "10m_HVAWT_reference_turbine"
        / "Control"
        / "vorlap_10m_HVAWT_reference_external.json"
    )

    cfg = qblade_runtime._load_config_json(str(config_path))

    assert cfg["force_scale"] == 1.0
    assert cfg["source_parameter_dir"] == "."
    assert cfg["log_file"] == "vorlap_qblade_debug.log"
    for key in ("sim_path", "airfoil_dir", "log_file", "source_parameter_dir"):
        assert not Path(str(cfg[key])).is_absolute()


def test_runtime_resolves_relative_paths_from_original_parameter_dir_when_copied_to_temp(tmp_path, monkeypatch):
    original_control = tmp_path / "qblade_example" / "reference_model" / "Control"
    original_control.mkdir(parents=True)
    temp_control = tmp_path / "QBladeCE" / "TEMP" / "run1"
    temp_control.mkdir(parents=True)

    sim_path = tmp_path / "qblade_example" / "baseline.sim"
    sim_path.write_text("sim", encoding="utf-8")
    airfoil_dir = tmp_path / "qblade_example" / "reference_model" / "VorLapAirfoils"
    airfoil_dir.mkdir(parents=True)
    (airfoil_dir / "default.h5").write_text("placeholder", encoding="utf-8")

    config = build_qblade_external_config(
        sim_path="../../baseline.sim",
        airfoil_dir="../VorLapAirfoils",
        node_ids=["BLD_1_0.000000"],
        source_parameter_dir=str(original_control),
        source_bounds_policy="raise",
    )
    cfg_path = temp_control / "vorlap_qblade_external.json"
    write_qblade_external_config(str(cfg_path), config)

    calls = {}

    def fake_convert(sim_arg, **kwargs):
        calls["sim_path"] = sim_arg
        component = make_component(n_nodes=1, airfoil_id="default")
        component.id = "BLD_1"
        viv_params = make_viv_params()
        return [component], viv_params, ["BLD_1_0.000000"]

    def fake_load_airfoils(airfoil_arg):
        calls["airfoil_dir"] = airfoil_arg
        return {"default": make_constant_airfoil_fft()}

    monkeypatch.setattr(qblade_runtime, "convert_qblade_to_vorlap_inputs", fake_convert)
    monkeypatch.setattr(qblade_runtime, "load_airfoil_fft_directory", fake_load_airfoils)

    runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))

    assert runtime is not None
    assert runtime.source_bounds_policy == "raise"
    assert runtime.controller.source_bounds_policy == "raise"
    assert calls["sim_path"] == str(sim_path)
    assert calls["airfoil_dir"] == str(airfoil_dir)


def test_runtime_from_config_replays_shuffled_node_ids_in_swap_order(tmp_path, monkeypatch):
    component = _component_with_distinct_3d_nodes()
    cl_dc = 0.9
    cd_dc = 1.6
    afft = make_constant_airfoil_fft(n_freq=1, cl_dc=cl_dc, cd_dc=cd_dc)
    _patch_config_runtime_io(monkeypatch, component, afft)
    node_ids = ["BLD_1_1.000000", "BLD_1_0.000000"]
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=node_ids,
        n_freq_depth=1,
        include_mean_load=True,
    )
    cfg_path = _write_config(tmp_path / "vorlap_qblade_external.json", cfg)
    runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    spec = build_external_library_table_spec(node_ids)
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    azimuth = np.float32(20.0)
    structural_velocity_swap = np.array(
        [
            [0.0, 0.2, 0.1],
            [0.5, -0.1, 0.0],
        ],
        dtype=np.float32,
    )
    swap[spec.swap_layout["time"].offset] = np.float32(0.0)
    swap[spec.swap_layout["timestep"].offset] = np.float32(0.25)
    swap[spec.swap_layout["azimuth_deg"].offset] = azimuth
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 6] = structural_velocity_swap.reshape(-1)

    runtime.update(swap)

    ambient = np.array([3.0, 0.0, 0.0], dtype=float)
    expected_swap = np.vstack(
        [
            _expected_dc_force(
                component,
                1,
                ambient - structural_velocity_swap[0].astype(float),
                float(azimuth),
                cl_dc=cl_dc,
                cd_dc=cd_dc,
            ),
            _expected_dc_force(
                component,
                0,
                ambient - structural_velocity_swap[1].astype(float),
                float(azimuth),
                cl_dc=cl_dc,
                cd_dc=cd_dc,
            ),
        ]
    )
    np.testing.assert_allclose(
        swap[spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 6].reshape(2, 3),
        expected_swap.astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("reverse_nodes", [False, True])
def test_nodal_inflow_matches_analytic_force_in_swap_order(tmp_path, monkeypatch, reverse_nodes):
    component = _component_with_distinct_3d_nodes()
    cl_dc, cd_dc = 0.9, 1.6
    _patch_config_runtime_io(
        monkeypatch, component,
        make_constant_airfoil_fft(n_freq=1, cl_dc=cl_dc, cd_dc=cd_dc),
    )
    indices = [1, 0] if reverse_nodes else [0, 1]
    node_ids = [f"BLD_1_{index:.6f}" for index in indices]
    cfg = build_qblade_external_config(
        sim_path="case.sim", airfoil_dir="airfoils", node_ids=node_ids,
        n_freq_depth=1, include_mean_load=True, use_qblade_local_inflow=True,
    )
    runtime = VorLapQBladeRuntime.from_qblade_config(
        str(_write_config(tmp_path / "nodal_inflow.json", cfg))
    )
    spec = build_hvawt_response_table_spec(node_ids)
    swap = np.zeros(spec.swap_size)
    swap[2] = 20.0
    structural = np.array([[0.0, 0.2, 0.1], [0.5, -0.1, 0.0]])
    wind = np.array([[3.0, 4.0, 1.0], [6.0, -2.0, 3.0]])
    offset = spec.swap_layout["velocity"].offset
    swap[offset:offset + 6] = structural.reshape(-1)

    forces = runtime.update(swap, ambient_velocity_global_mps=wind)

    # q_i = rho |U_i - v_i|_section^2 c_i ds_i / 2, resolved on the
    # rotated chord/normal. Distinct XYZ winds and a shuffled table expose
    # an incorrect node permutation, frame, or structural-velocity sign.
    expected = np.vstack([
        _expected_dc_force(component, inode, wind[i] - structural[i], 20.0,
                           cl_dc=cl_dc, cd_dc=cd_dc)
        for i, inode in enumerate(indices)
    ])
    np.testing.assert_allclose(forces, expected, rtol=1e-12, atol=1e-12)


def test_uniform_nodal_inflow_reproduces_mean_inflow(tmp_path, monkeypatch):
    component = _component_with_distinct_3d_nodes()
    _patch_config_runtime_io(
        monkeypatch, component,
        make_constant_airfoil_fft(cl_dc=0.5, cd_dc=0.2, cl_h1_amp=0.3),
    )
    cfg = build_qblade_external_config(
        sim_path="case.sim", airfoil_dir="airfoils",
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2, include_mean_load=True,
    )
    path = _write_config(tmp_path / "mean.json", cfg)
    mean = VorLapQBladeRuntime.from_qblade_config(str(path))
    cfg["use_qblade_local_inflow"] = True
    local = VorLapQBladeRuntime.from_qblade_config(
        str(_write_config(tmp_path / "local.json", cfg))
    )
    swap = np.zeros(mean.swap_size)
    swap[2] = 20.0
    swap[3:9] = [0.0, 0.2, 0.1, 0.5, -0.1, 0.0]
    for time_s in (0.0, 0.25, 0.5):
        swap[0] = time_s
        np.testing.assert_allclose(
            local.update(swap.copy(), ambient_velocity_global_mps=np.tile([3.0, 0.0, 0.0], (2, 1))),
            mean.update(swap.copy()), rtol=1e-12, atol=1e-12,
        )
    with pytest.raises(ValueError, match="requires nodal wind samples"):
        local.update(swap)
    for invalid in (np.ones(3), np.full((2, 3), np.nan)):
        with pytest.raises(ValueError, match="Nodal ambient velocity"):
            local.update(swap, ambient_velocity_global_mps=invalid)


def test_runtime_from_config_applies_viv_parameter_overrides(tmp_path, monkeypatch):
    component = _component_with_distinct_3d_nodes()
    afft = make_constant_airfoil_fft(n_freq=1)
    _patch_config_runtime_io(monkeypatch, component, afft)
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=node_ids,
        fluid_density=1.3,
        fluid_dynamicviscosity=1.7e-5,
        rotation_axis=(0.0, 1.0, 0.0),
        rotation_axis_offset=(1.0, 2.0, 3.0),
        inflow_vec=(1.0, 1.0, 0.0),
    )
    cfg_path = _write_config(tmp_path / "vorlap_qblade_external.json", cfg)

    runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    viv = runtime.controller.viv_params

    assert viv.fluid_density == pytest.approx(1.3)
    assert viv.fluid_dynamicviscosity == pytest.approx(1.7e-5)
    np.testing.assert_allclose(viv.rotation_axis, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(viv.rotation_axis_offset, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(viv.inflow_vec, np.array([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(
        runtime._ambient_velocity_global,
        np.array([3.0 / np.sqrt(2.0), 3.0 / np.sqrt(2.0), 0.0]),
    )


def test_runtime_from_config_suppresses_mean_load_by_default(tmp_path, monkeypatch):
    component = _component_with_distinct_3d_nodes()
    afft = make_constant_airfoil_fft(n_freq=1, cl_dc=0.9, cd_dc=1.6)
    _patch_config_runtime_io(monkeypatch, component, afft)
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=node_ids,
        n_freq_depth=1,
    )
    cfg_path = _write_config(tmp_path / "vorlap_qblade_external.json", cfg)
    runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    spec = build_external_library_table_spec(node_ids)
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = np.float32(0.0)
    swap[spec.swap_layout["timestep"].offset] = np.float32(0.25)
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 6] = np.array(
        [[0.0, 0.2, 0.1], [0.5, -0.1, 0.0]],
        dtype=np.float32,
    ).reshape(-1)

    runtime.update(swap)

    np.testing.assert_allclose(
        swap[spec.swap_layout["force"].offset : spec.swap_layout["force"].offset + 6],
        0.0,
        atol=1e-12,
    )


def test_runtime_from_config_enables_response_diagnostics_and_restart(tmp_path, monkeypatch):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.id = "BLD_1"
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    _patch_config_runtime_io(monkeypatch, component, afft)
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=node_ids,
        force_scale=0.0,
        sample_step=0.1,
        response_model=_response_config(node_ids),
    )
    cfg_path = _write_config(tmp_path / "response_model.json", cfg)
    uninterrupted = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    spec = build_external_library_table_spec(node_ids)

    def response_swap(time_s, structural_velocity):
        swap = np.zeros(spec.swap_size, dtype=np.float32)
        swap[spec.swap_layout["time"].offset] = np.float32(time_s)
        swap[spec.swap_layout["timestep"].offset] = np.float32(time_s / 0.1)
        swap[spec.swap_layout["azimuth_deg"].offset] = np.float32(0.0)
        swap[
            spec.swap_layout["velocity"].offset :
            spec.swap_layout["velocity"].offset + 6
        ] = np.asarray(structural_velocity, dtype=np.float32).reshape(-1)
        return swap

    first_velocity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    first_force = uninterrupted.update(response_swap(0.0, first_velocity))
    last_force = first_force
    for index in range(1, 23):
        time_s = 0.1 * index
        qdot_mps = 0.1 * np.sin(2.0 * np.pi * time_s)
        modal_velocity = [
            [0.0, 0.5 * qdot_mps, 0.0],
            [0.0, qdot_mps, 0.0],
        ]
        last_force = uninterrupted.update(response_swap(time_s, modal_velocity))

    assert uninterrupted.controller.response_model_enabled is True
    assert uninterrupted.response_model_type == "modal_negative_damping"
    np.testing.assert_array_equal(first_force, 0.0)
    assert np.max(np.abs(last_force)) > 0.0
    diagnostics = uninterrupted.last_response_diagnostics
    assert diagnostics is not None
    assert diagnostics.enabled is True
    assert diagnostics.projection_weighting == "configured"
    assert diagnostics.projection_weight_min == 1.0
    assert diagnostics.projection_weight_max == 2.0
    assert diagnostics.modal_power_w > 0.0
    response_summary = uninterrupted.response_diagnostics_summary
    assert response_summary["model_type"] == "modal_negative_damping"
    assert response_summary["samples"] == 23
    assert response_summary["max_abs_instantaneous_modal_power_w"] > 0.0
    assert response_summary["max_abs_instantaneous_structural_power_w"] > 0.0
    assert response_summary["max_abs_realized_modal_power_w"] > 0.0
    assert response_summary["max_abs_realized_structural_power_w"] > 0.0
    assert "gain_clipped_nodes=0" in uninterrupted.update_message()
    assert "force_clipped_nodes=0" in uninterrupted.update_message()
    assert "gain_clipped_node_samples=0" in uninterrupted.update_message()
    assert "force_clipped_node_samples=0" in uninterrupted.update_message()
    coverage = uninterrupted.source_coverage_summary(2.2)
    assert len(coverage) == 1
    assert coverage[0]["max_nodal_force_n"] == pytest.approx(0.0)
    assert coverage[0]["force_coefficient_max_abs"] == pytest.approx(0.0)
    assert coverage[0]["force_coefficient_out_of_range_count"] == 0
    assert coverage[0]["max_response_nodal_force_n"] > 0.0
    assert coverage[0]["max_total_nodal_force_n"] == pytest.approx(
        coverage[0]["max_response_nodal_force_n"]
    )

    checkpoint = json.loads(json.dumps(uninterrupted.checkpoint_state()))
    assert checkpoint["controller"]["phase_state"] is not None
    assert checkpoint["controller"]["response_model_state"] is not None
    missing_controller_state = dict(checkpoint)
    missing_controller_state.pop("controller")
    missing_state_runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    with pytest.raises(ValueError, match="requires valid controller checkpoint"):
        missing_state_runtime.restore_checkpoint_state(missing_controller_state)
    malformed_controller_state = dict(checkpoint)
    malformed_controller_state["controller"] = []
    malformed_state_runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    with pytest.raises(ValueError, match="requires valid controller checkpoint"):
        malformed_state_runtime.restore_checkpoint_state(malformed_controller_state)

    restarted = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    restarted.restore_checkpoint_state(checkpoint)
    assert restarted.controller.last_source_diagnostics is not None
    assert restarted.last_response_diagnostics == uninterrupted.last_response_diagnostics

    future_time = 2.3
    future_qdot_mps = 0.1 * np.sin(2.0 * np.pi * future_time)
    future_velocity = [
        [0.0, 0.5 * future_qdot_mps, 0.0],
        [0.0, future_qdot_mps, 0.0],
    ]
    expected = uninterrupted.update(response_swap(future_time, future_velocity))
    actual = restarted.update(response_swap(future_time, future_velocity))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-7)
    assert restarted.last_response_diagnostics == uninterrupted.last_response_diagnostics


def test_runtime_passes_declared_ambient_explicitly_to_schema_v2(tmp_path, monkeypatch):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.id = "BLD_1"
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    _patch_config_runtime_io(monkeypatch, component, afft)
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    response_config = _response_config(
        node_ids,
        schema_version=2,
        aerodynamic_speed_basis="declared_ambient_crossflow",
    )
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=node_ids,
        force_scale=0.0,
        response_model=response_config,
    )
    cfg_path = _write_config(tmp_path / "response_model_v2.json", cfg)
    runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    spec = build_external_library_table_spec(node_ids)

    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = np.float32(0.0)
    swap[spec.swap_layout["timestep"].offset] = np.float32(0.0)
    swap[spec.swap_layout["azimuth_deg"].offset] = np.float32(0.0)
    structural_velocity = np.array(
        [[0.0, 0.25, 0.0], [0.0, 0.5, 0.0]],
        dtype=np.float32,
    )
    swap[
        spec.swap_layout["velocity"].offset :
        spec.swap_layout["velocity"].offset + 6
    ] = structural_velocity.reshape(-1)
    runtime.update(swap)

    diagnostics = runtime.last_response_diagnostics
    assert diagnostics is not None
    assert diagnostics.aerodynamic_speed_basis == "declared_ambient_crossflow"
    assert diagnostics.response_declared_ambient_crossflow_speed_min_mps == 3.0
    assert diagnostics.response_declared_ambient_crossflow_speed_max_mps == 3.0
    assert diagnostics.response_relative_crossflow_speed_min_mps > 3.0
    assert diagnostics.response_relative_crossflow_speed_max_mps > 3.0
    assert diagnostics.realized_response_reduced_velocity_min == 3.0
    assert diagnostics.realized_response_reduced_velocity_max == 3.0
    assert diagnostics.speed_basis_invariant_violations == 0
    assert diagnostics.activation_invariant_violations == 0

    before = copy.deepcopy(runtime.checkpoint_state())
    bad_basis = copy.deepcopy(before)
    bad_basis["source_coverage"] = {"must_not_commit": {"samples": 99}}
    bad_basis["message"] = "must not commit"
    bad_basis["controller"]["response_model_state"][
        "aerodynamic_speed_basis"
    ] = "relative_crossflow"
    with pytest.raises(ValueError, match="speed-basis mismatch"):
        runtime.restore_checkpoint_state(bad_basis)
    assert runtime.checkpoint_state() == before

    missing_ambient = copy.deepcopy(before)
    missing_ambient["source_coverage"] = {
        "must_not_commit": {"samples": 100}
    }
    missing_ambient["controller"]["cached_step"].pop(
        "declared_ambient_velocity_mps"
    )
    with pytest.raises(ValueError, match="requires cached ambient velocity"):
        runtime.restore_checkpoint_state(missing_ambient)
    assert runtime.checkpoint_state() == before


def test_runtime_from_portable_modal_power_config_reports_power_limits_and_restart(
    tmp_path,
    monkeypatch,
):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.id = "BLD_1"
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_dc=0.0,
        cd_dc=0.0,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    _patch_config_runtime_io(monkeypatch, component, afft)
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    response_config = _modal_power_config(node_ids)
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=node_ids,
        force_scale=0.0,
        sample_step=0.1,
        response_model=response_config,
    )
    cfg_path = _write_config(tmp_path / "modal_power_model.json", cfg)
    uninterrupted = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    spec = build_external_library_table_spec(node_ids)

    assert uninterrupted.response_model_type == "modal_aerodynamic_power"
    assert isinstance(uninterrupted.controller.response_model, QBladeModalPowerModel)
    assert uninterrupted.controller.response_model.config.force_direction == "chord"
    assert uninterrupted.controller.response_model.config.modal_mass_weights_kg == (
        2.0,
        1.0,
    )
    assert (
        uninterrupted.controller.response_model.config.power_curve.source_sha256
        == "0" * 64
    )

    def modal_power_swap(time_s):
        qdot_mps = 2.0 * np.pi * 0.3 * np.sin(2.0 * np.pi * time_s)
        structural_velocity = np.array(
            [[0.5 * qdot_mps, 0.0, 0.0], [qdot_mps, 0.0, 0.0]],
            dtype=np.float32,
        )
        swap = np.zeros(spec.swap_size, dtype=np.float32)
        swap[spec.swap_layout["time"].offset] = np.float32(time_s)
        swap[spec.swap_layout["timestep"].offset] = np.float32(time_s / 0.1)
        swap[spec.swap_layout["azimuth_deg"].offset] = np.float32(0.0)
        swap[
            spec.swap_layout["velocity"].offset :
            spec.swap_layout["velocity"].offset + 6
        ] = structural_velocity.reshape(-1)
        return swap

    last_force = None
    for index in range(41):
        last_force = uninterrupted.update(modal_power_swap(0.1 * index))
    assert last_force is not None
    assert np.max(np.abs(last_force)) > 0.0
    diagnostics = uninterrupted.last_response_diagnostics
    assert isinstance(diagnostics, QBladeModalPowerDiagnostics)
    assert diagnostics.enabled is True
    assert diagnostics.amplitude_estimate_m > 0.0
    assert diagnostics.target_cycle_mean_power_w > 0.0
    assert diagnostics.applied_target_cycle_mean_power_w == pytest.approx(
        diagnostics.target_cycle_mean_power_w
    )
    assert diagnostics.realized_modal_power_ema_w > 0.0
    assert diagnostics.realized_structural_power_ema_w > 0.0

    summary = uninterrupted.response_diagnostics_summary
    assert summary["model_type"] == uninterrupted.response_model_type
    assert summary["samples"] == 41
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
    message = uninterrupted.update_message()
    assert "response_model=modal_aerodynamic_power" in message
    assert "response_A=" in message
    assert "target_cycle_power=" in message
    assert "applied_target_cycle_power=" in message
    assert "modal_power_inst=" in message
    assert "structural_power_inst=" in message
    assert "modal_power_ema=" in message
    assert "structural_power_ema=" in message
    assert "target_power_limited_steps=0" in message
    assert "generalized_force_limited_steps=0" in message
    assert "nodal_force_limited_node_samples=0" in message

    checkpoint = json.loads(json.dumps(uninterrupted.checkpoint_state()))
    assert checkpoint["controller"]["response_diagnostics_summary"] == summary
    restarted = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    restarted.restore_checkpoint_state(checkpoint)
    assert restarted.response_model_type == uninterrupted.response_model_type
    assert restarted.response_diagnostics_summary == summary
    assert restarted.last_response_diagnostics == diagnostics

    future_time = 4.1
    expected = uninterrupted.update(modal_power_swap(future_time))
    actual = restarted.update(modal_power_swap(future_time))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-7)
    assert restarted.last_response_diagnostics == uninterrupted.last_response_diagnostics
    assert restarted.response_diagnostics_summary == uninterrupted.response_diagnostics_summary


def test_runtime_modal_power_clip_summary_reports_every_limit(tmp_path, monkeypatch):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.id = "BLD_1"
    afft = make_constant_airfoil_fft(
        n_freq=2,
        cl_h1_amp=0.0,
        cd_h1_amp=0.0,
    )
    _patch_config_runtime_io(monkeypatch, component, afft)
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=node_ids,
        force_scale=0.0,
        sample_step=0.1,
        response_model=_modal_power_config(
            node_ids,
            limit_action="clip",
            max_abs_target_power_w=1.0,
            max_abs_generalized_force_n=0.1,
            max_abs_nodal_force_n=0.01,
        ),
    )
    cfg_path = _write_config(tmp_path / "modal_power_clipped.json", cfg)
    runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))
    spec = build_external_library_table_spec(node_ids)
    for index in range(41):
        time_s = 0.1 * index
        qdot_mps = 2.0 * np.pi * 0.3 * np.sin(2.0 * np.pi * time_s)
        swap = np.zeros(spec.swap_size, dtype=np.float32)
        swap[spec.swap_layout["time"].offset] = time_s
        swap[spec.swap_layout["timestep"].offset] = index
        swap[
            spec.swap_layout["velocity"].offset :
            spec.swap_layout["velocity"].offset + 6
        ] = np.array(
            [0.5 * qdot_mps, 0.0, 0.0, qdot_mps, 0.0, 0.0],
            dtype=np.float32,
        )
        runtime.update(swap)

    summary = runtime.response_diagnostics_summary
    assert summary["target_power_limited_steps"] > 0
    assert summary["generalized_force_limited_steps"] > 0
    assert summary["nodal_force_limited_steps"] > 0
    assert summary["nodal_force_limited_node_samples"] > 0
    assert summary["max_nodal_force_limited_nodes"] > 0


def test_runtime_from_config_rejects_response_node_order_mismatch(tmp_path, monkeypatch):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.id = "BLD_1"
    afft = make_constant_airfoil_fft(n_freq=2)
    _patch_config_runtime_io(monkeypatch, component, afft)
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    response_model = _response_config(node_ids)
    response_model["node_ids"] = list(reversed(node_ids))
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=node_ids,
        response_model=response_model,
    )
    cfg_path = _write_config(tmp_path / "bad_response_order.json", cfg)

    with pytest.raises(ValueError, match="response_model.node_ids.*order"):
        VorLapQBladeRuntime.from_qblade_config(str(cfg_path))


@pytest.mark.parametrize("force_direction", ["normal", "chord"])
def test_runtime_maps_validated_response_force_direction(
    tmp_path,
    monkeypatch,
    force_direction,
):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    component.id = "BLD_1"
    _patch_config_runtime_io(monkeypatch, component, make_constant_airfoil_fft())
    node_ids = ["BLD_1_0.000000", "BLD_1_1.000000"]
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=node_ids,
        response_model=_response_config(
            node_ids,
            force_direction=force_direction,
        ),
    )
    cfg_path = _write_config(
        tmp_path / f"response_{force_direction}.json",
        cfg,
    )

    runtime = VorLapQBladeRuntime.from_qblade_config(str(cfg_path))

    assert runtime.controller.response_model.config.force_direction == force_direction
    assert runtime.controller.response_model.config.projection_weights == (1.0, 2.0)
    assert runtime.controller.response_model.config.bandpass_damping_ratio == pytest.approx(
        0.2
    )


def test_runtime_from_config_rejects_duplicate_requested_node_ids(tmp_path):
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=["BLD_1_0.000000"],
    )
    cfg["node_ids"] = ["BLD_1_0.000000", "BLD_1_0.000000"]
    cfg_path = _write_config(tmp_path / "duplicate_nodes.json", cfg)

    with pytest.raises(ValueError, match="Duplicate requested config node id"):
        VorLapQBladeRuntime.from_qblade_config(str(cfg_path))


def test_runtime_from_config_rejects_missing_requested_node_component(tmp_path, monkeypatch):
    component = _component_with_distinct_3d_nodes()
    afft = make_constant_airfoil_fft(n_freq=1)
    _patch_config_runtime_io(monkeypatch, component, afft)
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=["STR_9_9_0.500000"],
    )
    cfg_path = _write_config(tmp_path / "missing_node.json", cfg)

    with pytest.raises(KeyError, match="unknown components"):
        VorLapQBladeRuntime.from_qblade_config(str(cfg_path))


def test_runtime_from_config_rejects_swap_size_mismatch(tmp_path, monkeypatch):
    component = _component_with_distinct_3d_nodes()
    afft = make_constant_airfoil_fft(n_freq=1)
    _patch_config_runtime_io(monkeypatch, component, afft)
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
    )
    cfg["swap_size"] += 1
    cfg_path = _write_config(tmp_path / "bad_swap_size.json", cfg)

    with pytest.raises(ValueError, match="swap_size .* does not match"):
        VorLapQBladeRuntime.from_qblade_config(str(cfg_path))


def test_runtime_from_config_rejects_velocity_shape_mismatch(tmp_path, monkeypatch):
    component = _component_with_distinct_3d_nodes()
    afft = make_constant_airfoil_fft(n_freq=1)
    _patch_config_runtime_io(monkeypatch, component, afft)
    cfg = build_qblade_external_config(
        sim_path="case.sim",
        airfoil_dir="airfoils",
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
    )
    for entry in cfg["swap_layout"]:
        if entry["name"] == "velocity":
            entry["shape"] = [1, 3]
            break
    cfg_path = _write_config(tmp_path / "bad_velocity_shape.json", cfg)

    with pytest.raises(ValueError, match="velocity block shape"):
        VorLapQBladeRuntime.from_qblade_config(str(cfg_path))


def test_runtime_debug_message_reports_max_force():
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    viv_params = make_viv_params()
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
        force_scale=100.0,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    runtime = VorLapQBladeRuntime(
        controller,
        spec.swap_layout,
        debug=True,
        resolved_sim_path="/tmp/example/baseline.sim",
        resolved_airfoil_dir="/tmp/example/VorLapAirfoils",
    )
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["timestep"].offset] = 0.25
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 6] = np.array(
        [2.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32
    )

    runtime.update(swap)

    assert "max|F|=" in runtime.update_message()
    assert "node=BLD_1_0.000000" in runtime.update_message()
    assert "scale=100" in runtime.update_message()
    assert "max|Vstruct|=" in runtime.update_message()
    assert "max|Vrel|=" in runtime.update_message()
    assert "cache=0/1" in runtime.update_message()


def test_runtime_writes_log_file(tmp_path):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    viv_params = make_viv_params()
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
        force_scale=100.0,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    log_path = tmp_path / "vorlap_qblade_debug.log"
    runtime = VorLapQBladeRuntime(
        controller,
        spec.swap_layout,
        debug=True,
        resolved_sim_path="/tmp/example/baseline.sim",
        resolved_airfoil_dir="/tmp/example/VorLapAirfoils",
        log_file=str(log_path),
    )
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["timestep"].offset] = 0.25
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 6] = np.array(
        [2.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32
    )

    runtime.update(swap)

    text = log_path.read_text(encoding="utf-8")
    assert "VorLap init ok;" in text
    assert "VorLap dbg" in text
    assert "max|F|=" in text


def test_runtime_log_write_failure_is_nonfatal(tmp_path):
    component = make_component(n_nodes=2, span=2.0, airfoil_id="default")
    afft = make_constant_airfoil_fft(n_freq=2)
    viv_params = make_viv_params()
    controller = QBladeController.from_components(
        components=[component],
        airfoils={"default": afft},
        viv_params=viv_params,
        node_ids=["BLD_1_0.000000", "BLD_1_1.000000"],
        n_freq_depth=2,
        force_scale=100.0,
    )
    spec = build_external_library_table_spec(controller.node_ids)
    bad_parent = tmp_path / "not_a_directory"
    bad_parent.write_text("blocking file", encoding="utf-8")
    runtime = VorLapQBladeRuntime(
        controller,
        spec.swap_layout,
        debug=True,
        resolved_sim_path="/tmp/example/baseline.sim",
        resolved_airfoil_dir="/tmp/example/VorLapAirfoils",
        log_file=str(bad_parent / "vorlap_qblade_debug.log"),
    )
    swap = np.zeros(spec.swap_size, dtype=np.float32)
    swap[spec.swap_layout["time"].offset] = 0.0
    swap[spec.swap_layout["timestep"].offset] = 0.25
    swap[spec.swap_layout["azimuth_deg"].offset] = 0.0
    swap[spec.swap_layout["velocity"].offset : spec.swap_layout["velocity"].offset + 6] = np.array(
        [2.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float32
    )

    runtime.update(swap)

    assert runtime._log_write_failed is True
    assert runtime._log_write_error is not None
    assert "VorLap dbg" in runtime.update_message()


def test_build_qblade_external_config_infers_required_geometry_flags():
    config = build_qblade_external_config(
        sim_path="/tmp/case.sim",
        airfoil_dir="/tmp/airfoils",
        node_ids=["BLD_1_0.000000", "STR_1_1_1.000000", "TWR_1_0.000000"],
        include_struts=False,
        include_tower=False,
    )

    assert config["include_struts"] is True
    assert config["include_tower"] is True
