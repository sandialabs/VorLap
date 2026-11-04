from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

import vorlap.qblade_hvawt_response as hvawt_response_module
from vorlap.qblade_hvawt_response import (
    EXTERNAL_VIV_LEGACY,
    EXTERNAL_VIV_NONE,
    EXTERNAL_VIV_RESPONSE,
    HVAWTForceSourceContract,
    HVAWTModalBasis,
    HVAWTModeResponseConfig,
    HVAWTResponseCore,
    HVAWTResponseCoreConfig,
    FixedAxisKinematicsConfig,
    INCREMENTAL_VIV_SCREENING,
    NOT_APPLICABLE,
    calculate_hvawt_response_force,
    decompose_fixed_axis_velocity,
    load_tracked_hvawt_modal_basis,
    separate_hvawt_flows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = REPO_ROOT / "run_headless" / "config"
EVIDENCE_FILENAMES = (
    "hvawt_response_mode_identification.json",
    "hvawt_response_mode_shapes.csv",
    "hvawt_response_mode_force_duals.csv",
    "hvawt_modal_evidence_review_marker.json",
)
ZERO_SHA256 = "0" * 64
RPM_TO_RADPS = 2.0 * math.pi / 60.0


def _node_ids():
    return tuple(f"node_{index:02d}" for index in range(39))


def _synthetic_basis():
    shape = np.zeros((39, 3, 2), dtype=complex)
    shape[0, 0, 0] = 1.0
    shape[1, 1, 1] = 1.0
    dual = np.array(shape, copy=True)
    frequencies = (1.0, 2.0)
    masses = (10.0, 5.0)
    stiffnesses = tuple(
        mass * (2.0 * math.pi * frequency) ** 2
        for mass, frequency in zip(masses, frequencies)
    )
    return HVAWTModalBasis(
        node_ids=_node_ids(),
        tracked_modes=(6, 12),
        basis_rpm=1.0,
        complex_shape_body_per_m=shape,
        complex_dual_body=dual,
        natural_frequency_hz=frequencies,
        generalized_modal_mass_kg=masses,
        generalized_modal_stiffness_npm=stiffnesses,
        identification_sha256=ZERO_SHA256,
        shapes_sha256=ZERO_SHA256,
        duals_sha256=ZERO_SHA256,
        review_marker_sha256=ZERO_SHA256,
    )


def _kinematics(**overrides):
    values = dict(
        node_ids=_node_ids(),
        axis_unit_global=(0.0, 0.0, 1.0),
        axis_origin_global_m=(0.0, 0.0, 0.0),
        body_to_global_at_reference=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        reference_azimuth_deg=0.0,
        basis_rpm=1.0,
        maximum_abs_basis_rpm_error_rpm=0.01,
        maximum_azimuth_rate_error_radps=1.0e-8,
    )
    values.update(overrides)
    return FixedAxisKinematicsConfig(**values)


def _mode_configs(basis=None, **overrides):
    basis = _synthetic_basis() if basis is None else basis
    result = []
    for index, mode in enumerate((6, 12)):
        values = dict(
            tracked_mode=mode,
            natural_frequency_hz=basis.natural_frequency_hz[index],
            generalized_modal_mass_kg=basis.generalized_modal_mass_kg[index],
            generalized_modal_stiffness_npm=(
                basis.generalized_modal_stiffness_npm[index]
            ),
            reference_length_m=1.0 if mode == 6 else 0.5,
            reduced_velocity_low=4.0,
            reduced_velocity_peak=5.0,
            reduced_velocity_high=8.0,
            aerodynamic_damping_gain_h1=2.0,
            saturation_amplitude_ratio=0.2,
            rms_time_constant_s=0.3,
            bandpass_damping_ratio=0.2,
            max_abs_gain=100.0,
            coefficient_source_sha256="1" * 64,
        )
        values.update(overrides)
        result.append(HVAWTModeResponseConfig(**values))
    return tuple(result)


def _core_config(**overrides):
    basis = overrides.pop("modal_basis", _synthetic_basis())
    values = dict(
        kinematics=_kinematics(node_ids=basis.node_ids, basis_rpm=basis.basis_rpm),
        modal_basis=basis,
        modes=_mode_configs(basis),
        force_sources=HVAWTForceSourceContract(),
        max_abs_total_nodal_force_n=1.0e8,
    )
    values.update(overrides)
    return HVAWTResponseCoreConfig(**values)


def _body_positions():
    angles = np.linspace(0.0, 2.0 * math.pi, 39, endpoint=False)
    radii = 5.0 + 0.2 * (np.arange(39) % 3)
    z = 10.0 + 0.25 * np.arange(39)
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles), z))


def _rotation_z(angle_deg):
    angle = math.radians(angle_deg)
    return np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _axis_angle(axis, angle_deg):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ]
    )


def _rehash_checkpoint(state):
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    state["state_sha256"] = hashlib.sha256(encoded).hexdigest()


def _step_inputs(core, time_s, *, qdot=(0.0, 0.0), order=None):
    azimuth_deg = 6.0 * time_s
    rotation = _rotation_z(azimuth_deg)
    body_positions = _body_positions()
    positions = body_positions @ rotation.T
    omega = np.array([0.0, 0.0, RPM_TO_RADPS])
    rigid = np.cross(np.broadcast_to(omega, positions.shape), positions)
    shape_global = np.einsum(
        "ab,ibm->iam",
        rotation,
        core.config.modal_basis.real_shape_body_per_m,
    )
    flexible = np.einsum("iam,m->ia", shape_global, np.asarray(qdot, dtype=float))
    total = rigid + flexible
    ambient = rigid + np.array([5.0, 0.0, 0.0])[None, :]
    values = dict(
        node_ids=core.config.modal_basis.node_ids,
        positions_global_m=positions,
        total_structural_velocity_global_mps=total,
        lss_rpm=1.0,
        azimuth_deg=azimuth_deg,
        ambient_velocity_global_mps=ambient,
        chord_unit_global=np.tile([1.0, 0.0, 0.0], (39, 1)),
        normal_unit_global=np.tile([0.0, 1.0, 0.0], (39, 1)),
        chord_m=np.ones(39),
        thickness_ratio=np.full(39, 0.2),
        local_length_m=np.ones(39),
        fluid_density_kgpm3=1.2,
    )
    if order is not None:
        order = np.asarray(order, dtype=int)
        values["node_ids"] = tuple(values["node_ids"][index] for index in order)
        for name in (
            "positions_global_m",
            "total_structural_velocity_global_mps",
            "ambient_velocity_global_mps",
            "chord_unit_global",
            "normal_unit_global",
            "chord_m",
            "thickness_ratio",
            "local_length_m",
        ):
            values[name] = np.asarray(values[name])[order]
    return values


def _tracked_basis():
    return load_tracked_hvawt_modal_basis(
        identification_path=EVIDENCE_DIR / "hvawt_response_mode_identification.json",
        shapes_path=EVIDENCE_DIR / "hvawt_response_mode_shapes.csv",
        duals_path=EVIDENCE_DIR / "hvawt_response_mode_force_duals.csv",
        review_marker_path=EVIDENCE_DIR / "hvawt_modal_evidence_review_marker.json",
    )


def _copy_evidence(destination):
    for name in EVIDENCE_FILENAMES:
        shutil.copy2(EVIDENCE_DIR / name, destination / name)


def test_force_source_contract_is_exclusive_and_requires_native_aero():
    response = HVAWTForceSourceContract()
    response.validate_runtime_flags(
        native_aero_enabled=True,
        legacy_source_enabled=False,
        response_closure_enabled=True,
        include_mean_load=False,
    )
    legacy = HVAWTForceSourceContract(
        external_viv_mode=EXTERNAL_VIV_LEGACY,
        coefficient_role=NOT_APPLICABLE,
    )
    none = HVAWTForceSourceContract(
        external_viv_mode=EXTERNAL_VIV_NONE,
        coefficient_role=NOT_APPLICABLE,
    )
    assert legacy.legacy_source_enabled and not legacy.response_closure_enabled
    assert not none.legacy_source_enabled and not none.response_closure_enabled

    with pytest.raises(ValueError, match="mutually exclusive"):
        response.validate_runtime_flags(
            native_aero_enabled=True,
            legacy_source_enabled=True,
            response_closure_enabled=True,
            include_mean_load=False,
        )
    with pytest.raises(ValueError, match="Native QBlade"):
        response.validate_runtime_flags(
            native_aero_enabled=False,
            legacy_source_enabled=False,
            response_closure_enabled=True,
            include_mean_load=False,
        )
    with pytest.raises(ValueError, match="coefficient_role"):
        HVAWTForceSourceContract(coefficient_role=NOT_APPLICABLE)
    with pytest.raises(ValueError, match="include_mean_load"):
        HVAWTForceSourceContract(include_mean_load=True)


def test_fixed_axis_decomposition_uses_current_positions_and_exact_units():
    config = _kinematics()
    positions = _body_positions()
    omega = np.array([0.0, 0.0, RPM_TO_RADPS])
    rigid = np.cross(np.broadcast_to(omega, positions.shape), positions)
    flexible = np.linspace(-0.2, 0.2, 39)[:, None] * np.array([1.0, -0.5, 0.25])
    state = decompose_fixed_axis_velocity(
        config,
        positions_global_m=positions,
        total_structural_velocity_global_mps=rigid + flexible,
        lss_rpm=1.0,
        azimuth_deg=0.0,
    )
    np.testing.assert_allclose(state.angular_velocity_global_radps, omega, atol=1.0e-15)
    np.testing.assert_allclose(state.rigid_body_velocity_global_mps, rigid, atol=1.0e-15)
    np.testing.assert_allclose(state.flexible_velocity_global_mps, flexible, atol=1.0e-15)
    np.testing.assert_allclose(
        state.rigid_body_velocity_global_mps + state.flexible_velocity_global_mps,
        rigid + flexible,
        atol=1.0e-15,
    )


def test_tilted_axis_nonidentity_reference_rotation_recovers_coordinates():
    basis = _synthetic_basis()
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    origin = np.array([0.4, -0.3, 1.2])
    reference = _axis_angle([0.3, -0.4, 0.5], 23.0)
    reference_azimuth = 13.0
    azimuth = 50.0
    kinematics = _kinematics(
        node_ids=basis.node_ids,
        axis_unit_global=tuple(axis),
        axis_origin_global_m=tuple(origin),
        body_to_global_at_reference=tuple(tuple(row) for row in reference),
        reference_azimuth_deg=reference_azimuth,
    )
    core = HVAWTResponseCore(
        _core_config(modal_basis=basis, kinematics=kinematics)
    )
    rotation = _axis_angle(axis, azimuth - reference_azimuth) @ reference
    positions = _body_positions() @ rotation.T + origin
    omega = RPM_TO_RADPS * axis
    rigid = np.cross(np.broadcast_to(omega, positions.shape), positions - origin)
    requested = np.array([0.31, -0.22])
    flexible = np.einsum(
        "ab,ibm,m->ia",
        rotation,
        basis.real_shape_body_per_m,
        requested,
    )
    result = core.step(
        0.0,
        node_ids=basis.node_ids,
        positions_global_m=positions,
        total_structural_velocity_global_mps=rigid + flexible,
        lss_rpm=1.0,
        azimuth_deg=azimuth,
        ambient_velocity_global_mps=rigid + 5.0 * rotation[:, 0],
        chord_unit_global=np.tile(rotation[:, 0], (39, 1)),
        normal_unit_global=np.tile(rotation[:, 1], (39, 1)),
        chord_m=1.0,
        thickness_ratio=0.2,
        local_length_m=1.0,
        fluid_density_kgpm3=1.2,
    )

    np.testing.assert_allclose(
        result.rigid_body_velocity_global_mps,
        rigid,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert tuple(
        row.corrected_raw_qdot_mps for row in result.diagnostics.mode_diagnostics
    ) == pytest.approx(requested, rel=0.0, abs=2.0e-15)


def test_negative_raw_rpm_and_azimuth_wrap_follow_declared_signs():
    basis = _synthetic_basis()
    kinematics = _kinematics(
        node_ids=basis.node_ids,
        rpm_sign=-1,
        azimuth_sign=-1,
    )
    core = HVAWTResponseCore(
        _core_config(modal_basis=basis, kinematics=kinematics)
    )

    def inputs(azimuth):
        rotation = kinematics.rotation_body_to_global(azimuth)
        positions = _body_positions() @ rotation.T
        omega = np.array([0.0, 0.0, RPM_TO_RADPS])
        rigid = np.cross(np.broadcast_to(omega, positions.shape), positions)
        return dict(
            node_ids=basis.node_ids,
            positions_global_m=positions,
            total_structural_velocity_global_mps=rigid,
            lss_rpm=-1.0,
            azimuth_deg=azimuth,
            ambient_velocity_global_mps=rigid + np.array([5.0, 0.0, 0.0]),
            chord_unit_global=np.tile([1.0, 0.0, 0.0], (39, 1)),
            normal_unit_global=np.tile([0.0, 1.0, 0.0], (39, 1)),
            chord_m=1.0,
            thickness_ratio=0.2,
            local_length_m=1.0,
            fluid_density_kgpm3=1.2,
        )

    core.step(0.0, **inputs(0.03))
    wrapped = core.step(0.01, **inputs(359.97))
    np.testing.assert_allclose(wrapped.flexible_velocity_global_mps, 0.0, atol=2.0e-15)
    assert wrapped.diagnostics.lss_rpm == -1.0
    assert wrapped.diagnostics.angular_speed_radps == pytest.approx(RPM_TO_RADPS)


def test_rigid_rotation_has_zero_flexible_coordinate_and_response_force():
    core = HVAWTResponseCore(_core_config())
    inputs = _step_inputs(core, 0.0)
    result = core.step(0.0, **inputs)

    np.testing.assert_allclose(result.flexible_velocity_global_mps, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(result.additive_force_global_n, 0.0, atol=0.0)
    assert all(
        mode.corrected_raw_qdot_mps == pytest.approx(0.0, abs=1.0e-15)
        for mode in result.diagnostics.mode_diagnostics
    )


def test_flow_separation_excludes_flex_only_from_response_flow():
    positions = _body_positions()
    omega = np.array([0.0, 0.0, RPM_TO_RADPS])
    rigid = np.cross(np.broadcast_to(omega, positions.shape), positions)
    flex_a = np.zeros((39, 3))
    flex_b = np.tile([0.4, -0.2, 0.0], (39, 1))
    ambient = rigid + np.array([5.0, 0.0, 0.0])
    common = dict(
        ambient_velocity_global_mps=ambient,
        rigid_body_velocity_global_mps=rigid,
        chord_unit_global=np.tile([1.0, 0.0, 0.0], (39, 1)),
        normal_unit_global=np.tile([0.0, 1.0, 0.0], (39, 1)),
        chord_m=1.0,
        thickness_ratio=0.2,
    )
    flow_a = separate_hvawt_flows(
        total_structural_velocity_global_mps=rigid + flex_a,
        **common,
    )
    flow_b = separate_hvawt_flows(
        total_structural_velocity_global_mps=rigid + flex_b,
        **common,
    )
    assert not np.array_equal(
        flow_a.base_relative_velocity_global_mps,
        flow_b.base_relative_velocity_global_mps,
    )
    np.testing.assert_array_equal(
        flow_a.response_relative_velocity_global_mps,
        flow_b.response_relative_velocity_global_mps,
    )
    np.testing.assert_array_equal(
        flow_a.response_effective_crossflow_speed_mps,
        flow_b.response_effective_crossflow_speed_mps,
    )


def test_flow_separation_rejects_nonunit_or_nonorthogonal_section_frames():
    common = dict(
        ambient_velocity_global_mps=np.tile([5.0, 0.0, 0.0], (39, 1)),
        total_structural_velocity_global_mps=np.zeros((39, 3)),
        rigid_body_velocity_global_mps=np.zeros((39, 3)),
        chord_m=1.0,
        thickness_ratio=0.2,
    )
    with pytest.raises(ValueError, match="unit length"):
        separate_hvawt_flows(
            chord_unit_global=np.tile([2.0, 0.0, 0.0], (39, 1)),
            normal_unit_global=np.tile([0.0, 1.0, 0.0], (39, 1)),
            **common,
        )
    with pytest.raises(ValueError, match="orthogonal"):
        separate_hvawt_flows(
            chord_unit_global=np.tile([1.0, 0.0, 0.0], (39, 1)),
            normal_unit_global=np.tile([1.0, 0.0, 0.0], (39, 1)),
            **common,
        )


def test_synthetic_two_mode_extraction_is_rotation_covariant_and_has_si_units():
    basis = _synthetic_basis()
    qdot = np.array([0.32, -0.17])
    for angle in (0.0, 37.0, 143.0):
        rotation = _rotation_z(angle)
        velocity_global = np.einsum(
            "ab,ibm,m->ia",
            rotation,
            basis.real_shape_body_per_m,
            qdot,
        )
        velocity_body = velocity_global @ rotation
        recovered, raw_complex, leakage = basis.extract_modal_velocity(velocity_body)
        np.testing.assert_allclose(recovered, qdot, rtol=0.0, atol=2.0e-15)
        np.testing.assert_allclose(raw_complex.real, qdot, rtol=0.0, atol=2.0e-15)
        np.testing.assert_allclose(raw_complex.imag, 0.0, atol=0.0)
        np.testing.assert_allclose(leakage, 0.0, atol=0.0)


def test_real_tracked_modes_and_duals_load_with_frozen_numeric_identity():
    basis = _tracked_basis()

    assert basis.fingerprint == "7b0ba1f6fe00d539b16c03c4e169a8f424e2ee475c660e45952effd094b6917c"
    assert basis.review_marker_sha256 == (
        "de237c327b55a6a9ccb4dc127ccd1a34cac258cc1b4b7a44de4e19c85cb4a843"
    )
    assert basis.tracked_modes == (6, 12)
    assert basis.basis_rpm == 1.0
    assert basis.natural_frequency_hz == pytest.approx(
        (2.5115545017693828, 4.572593676110874),
        rel=0.0,
        abs=1.0e-14,
    )
    assert basis.generalized_modal_mass_kg == pytest.approx(
        (337.9307091909933, 123.99012430136153),
        rel=0.0,
        abs=1.0e-12,
    )
    np.testing.assert_allclose(
        basis.coordinate_correction_matrix,
        np.array(
            [
                [0.999976321, -0.00000712776509],
                [-0.00000324439699, 0.999811985],
            ]
        ),
        rtol=0.0,
        atol=5.0e-10,
    )
    assert basis.real_shape_maximum_site_norm == pytest.approx(
        (0.9999785948619707, 0.9997562060525813),
        rel=0.0,
        abs=1.0e-15,
    )


def test_real_artifact_two_mode_coordinate_recovery_uses_complex_dual_correction():
    basis = _tracked_basis()
    requested = np.array([0.21, -0.08])
    velocity_body = np.einsum(
        "iam,m->ia",
        basis.real_shape_body_per_m,
        requested,
    )
    recovered, raw_complex, leakage = basis.extract_modal_velocity(velocity_body)

    np.testing.assert_allclose(recovered, requested, rtol=0.0, atol=2.0e-15)
    assert np.max(np.abs(raw_complex.real - requested)) > 1.0e-6
    assert np.max(leakage) < 0.001


def test_real_force_adjoint_has_exact_multimode_virtual_work_for_arbitrary_velocity():
    basis = _tracked_basis()
    rng = np.random.default_rng(7813)
    velocity_body = rng.normal(size=(39, 3))
    generalized_force = np.array([17.3, -8.2])

    qdot, _, _ = basis.extract_modal_velocity(velocity_body)
    force_body = basis.energy_conjugate_force_body(generalized_force)

    assert float(np.sum(velocity_body * force_body)) == pytest.approx(
        float(np.dot(qdot, generalized_force)), rel=0.0, abs=2.0e-14
    )
    shape = basis.real_shape_body_per_m.reshape(117, 2)
    adjoint = basis.real_force_adjoint_body.reshape(117, 2)
    np.testing.assert_allclose(shape.T @ adjoint, np.eye(2), atol=4.0e-16)
    assert not np.allclose(shape, adjoint)


def test_artifact_hash_mutation_fails_before_numeric_use(tmp_path):
    _copy_evidence(tmp_path)
    shape_path = tmp_path / "hvawt_response_mode_shapes.csv"
    payload = bytearray(shape_path.read_bytes())
    payload[-2] = ord("1") if payload[-2] != ord("1") else ord("2")
    shape_path.write_bytes(bytes(payload))

    with pytest.raises(ValueError, match="SHA-256"):
        load_tracked_hvawt_modal_basis(
            identification_path=tmp_path / "hvawt_response_mode_identification.json",
            shapes_path=shape_path,
            duals_path=tmp_path / "hvawt_response_mode_force_duals.csv",
            review_marker_path=tmp_path / "hvawt_modal_evidence_review_marker.json",
        )


def test_review_marker_mutation_cannot_rebind_tracked_artifacts(tmp_path):
    _copy_evidence(tmp_path)
    marker_path = tmp_path / "hvawt_modal_evidence_review_marker.json"
    marker = json.loads(marker_path.read_text())
    marker["damping_candidate_selected"] = True
    marker_path.write_text(json.dumps(marker))

    with pytest.raises(ValueError, match="modal review marker SHA-256"):
        load_tracked_hvawt_modal_basis(
            identification_path=tmp_path / "hvawt_response_mode_identification.json",
            shapes_path=tmp_path / "hvawt_response_mode_shapes.csv",
            duals_path=tmp_path / "hvawt_response_mode_force_duals.csv",
            review_marker_path=marker_path,
        )


@pytest.mark.parametrize("linked_name", EVIDENCE_FILENAMES)
def test_tracked_loader_rejects_each_symlinked_artifact(tmp_path, linked_name):
    for name in EVIDENCE_FILENAMES:
        destination = tmp_path / name
        if name == linked_name:
            destination.symlink_to(EVIDENCE_DIR / name)
        else:
            shutil.copy2(EVIDENCE_DIR / name, destination)

    with pytest.raises(ValueError, match="symbolic link"):
        load_tracked_hvawt_modal_basis(
            identification_path=tmp_path / "hvawt_response_mode_identification.json",
            shapes_path=tmp_path / "hvawt_response_mode_shapes.csv",
            duals_path=tmp_path / "hvawt_response_mode_force_duals.csv",
            review_marker_path=tmp_path / "hvawt_modal_evidence_review_marker.json",
        )


def test_tracked_loader_rejects_nonregular_artifact(tmp_path):
    for name in EVIDENCE_FILENAMES:
        destination = tmp_path / name
        if name == "hvawt_response_mode_shapes.csv":
            destination.mkdir()
        else:
            shutil.copy2(EVIDENCE_DIR / name, destination)

    with pytest.raises(ValueError, match="regular file"):
        load_tracked_hvawt_modal_basis(
            identification_path=tmp_path / "hvawt_response_mode_identification.json",
            shapes_path=tmp_path / "hvawt_response_mode_shapes.csv",
            duals_path=tmp_path / "hvawt_response_mode_force_duals.csv",
            review_marker_path=tmp_path / "hvawt_modal_evidence_review_marker.json",
        )


def test_tracked_loader_rejects_symlinked_parent_root(tmp_path):
    regular_root = tmp_path / "regular_root"
    regular_root.mkdir()
    _copy_evidence(regular_root)
    linked_root = tmp_path / "linked_root"
    linked_root.symlink_to(regular_root, target_is_directory=True)

    with pytest.raises(ValueError, match="path component.*symbolic link"):
        load_tracked_hvawt_modal_basis(
            identification_path=linked_root / "hvawt_response_mode_identification.json",
            shapes_path=linked_root / "hvawt_response_mode_shapes.csv",
            duals_path=linked_root / "hvawt_response_mode_force_duals.csv",
            review_marker_path=linked_root / "hvawt_modal_evidence_review_marker.json",
        )


def test_tracked_loader_reads_each_artifact_payload_exactly_once(monkeypatch):
    real_fdopen = hvawt_response_module.os.fdopen
    read_count = 0

    class CountingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, *args, **kwargs):
            nonlocal read_count
            read_count += 1
            return self.stream.read(*args, **kwargs)

        def fileno(self):
            return self.stream.fileno()

    def counting_fdopen(*args, **kwargs):
        return CountingStream(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(hvawt_response_module.os, "fdopen", counting_fdopen)
    basis = _tracked_basis()
    assert read_count == len(EVIDENCE_FILENAMES)
    assert basis.fingerprint == (
        "7b0ba1f6fe00d539b16c03c4e169a8f424e2ee475c660e45952effd094b6917c"
    )


def test_tracked_loader_is_portable_across_two_regular_copied_roots(tmp_path):
    loaded = []
    for root_name in ("source_a", "source_b"):
        root = tmp_path / root_name
        root.mkdir()
        _copy_evidence(root)
        loaded.append(
            load_tracked_hvawt_modal_basis(
                identification_path=root / "hvawt_response_mode_identification.json",
                shapes_path=root / "hvawt_response_mode_shapes.csv",
                duals_path=root / "hvawt_response_mode_force_duals.csv",
                review_marker_path=root / "hvawt_modal_evidence_review_marker.json",
            )
        )

    assert loaded[0].fingerprint == loaded[1].fingerprint
    np.testing.assert_array_equal(
        loaded[0].complex_shape_body_per_m,
        loaded[1].complex_shape_body_per_m,
    )
    np.testing.assert_array_equal(
        loaded[0].complex_dual_body,
        loaded[1].complex_dual_body,
    )


def test_stable_reader_rejects_mutation_during_the_single_read(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b'{"stable":true}\n')
    real_fdopen = hvawt_response_module.os.fdopen
    mutated = False

    class MutatingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, *args, **kwargs):
            nonlocal mutated
            payload = self.stream.read(*args, **kwargs)
            if not mutated:
                artifact.write_bytes(payload + b" ")
                mutated = True
            return payload

        def fileno(self):
            return self.stream.fileno()

    monkeypatch.setattr(
        hvawt_response_module.os,
        "fdopen",
        lambda *args, **kwargs: MutatingStream(real_fdopen(*args, **kwargs)),
    )
    with pytest.raises(ValueError, match="changed while"):
        hvawt_response_module._read_stable_regular_file(
            artifact,
            label="test artifact",
        )


def test_swap_after_verified_read_cannot_change_bytes_parsed(tmp_path, monkeypatch):
    _copy_evidence(tmp_path)
    shape_path = tmp_path / "hvawt_response_mode_shapes.csv"
    real_reader = hvawt_response_module._read_stable_regular_file
    swapped = False

    def swapping_reader(path, *, label):
        nonlocal swapped
        snapshot = real_reader(path, label=label)
        if Path(path) == shape_path and not swapped:
            replacement = tmp_path / "attacker-shapes.csv"
            replacement.write_bytes(b"attacker-controlled replacement\n")
            replacement.replace(shape_path)
            swapped = True
        return snapshot

    monkeypatch.setattr(
        hvawt_response_module,
        "_read_stable_regular_file",
        swapping_reader,
    )
    basis = load_tracked_hvawt_modal_basis(
        identification_path=tmp_path / "hvawt_response_mode_identification.json",
        shapes_path=shape_path,
        duals_path=tmp_path / "hvawt_response_mode_force_duals.csv",
        review_marker_path=tmp_path / "hvawt_modal_evidence_review_marker.json",
    )

    assert swapped
    assert shape_path.read_bytes() == b"attacker-controlled replacement\n"
    assert basis.shapes_sha256 == (
        "d147983ee1f818c491a2c4d1ae30f70753dfe53acd60324273cc9dff3f7b6dd1"
    )
    assert basis.fingerprint == (
        "7b0ba1f6fe00d539b16c03c4e169a8f424e2ee475c660e45952effd094b6917c"
    )


def test_real_three_dimensional_force_scales_dimensionally_and_rotates():
    basis = _synthetic_basis()
    modes = _mode_configs(basis)
    flexible_body = np.einsum(
        "iam,m->ia",
        basis.real_shape_body_per_m,
        np.array([0.1, -0.04]),
    )
    speed = np.full(39, 5.0)
    width = np.full(39, 0.2)
    length = np.ones(39)
    kwargs = dict(
        modal_basis=basis,
        mode_configs=modes,
        filtered_qdot_mps=(0.1, -0.04),
        qdot_rms_mps=(0.01, 0.01),
        response_effective_crossflow_speed_mps=speed,
        response_section_width_m=width,
        local_length_m=length,
        flexible_velocity_global_mps=flexible_body,
        rigid_body_velocity_global_mps=np.zeros((39, 3)),
    )
    force_1, diagnostics_1 = calculate_hvawt_response_force(
        rotation_body_to_global=np.eye(3),
        fluid_density_kgpm3=1.2,
        **kwargs,
    )
    force_2, _ = calculate_hvawt_response_force(
        rotation_body_to_global=np.eye(3),
        fluid_density_kgpm3=2.4,
        **kwargs,
    )
    rotation = _rotation_z(61.0)
    force_rotated, _ = calculate_hvawt_response_force(
        rotation_body_to_global=rotation,
        fluid_density_kgpm3=1.2,
        **{
            **kwargs,
            "flexible_velocity_global_mps": flexible_body @ rotation.T,
        },
    )

    np.testing.assert_allclose(force_2, 2.0 * force_1, rtol=1.0e-14, atol=1.0e-15)
    np.testing.assert_allclose(force_rotated, force_1 @ rotation.T, rtol=1.0e-14, atol=1.0e-15)
    assert sum(row["flexible_power_w"] for row in diagnostics_1) > 0.0


def test_force_equation_has_exact_rho_u_d_length_scaling_and_units():
    basis = _synthetic_basis()
    modes = _mode_configs(
        basis,
        reduced_velocity_low=0.0,
        reduced_velocity_peak=5.0,
        reduced_velocity_high=10.0,
    )
    common = dict(
        modal_basis=basis,
        mode_configs=modes,
        rotation_body_to_global=np.eye(3),
        filtered_qdot_mps=(0.1, 0.0),
        qdot_rms_mps=(0.01, 0.0),
        flexible_velocity_global_mps=np.zeros((39, 3)),
        rigid_body_velocity_global_mps=np.zeros((39, 3)),
    )

    def force(*, density=1.2, speed=2.5, width=0.2, length=1.0):
        return calculate_hvawt_response_force(
            response_effective_crossflow_speed_mps=np.full(39, speed),
            response_section_width_m=np.full(39, width),
            local_length_m=np.full(39, length),
            fluid_density_kgpm3=density,
            **common,
        )[0]

    baseline = force()
    np.testing.assert_allclose(force(density=2.4), 2.0 * baseline, rtol=1.0e-14)
    np.testing.assert_allclose(force(width=0.4), 2.0 * baseline, rtol=1.0e-14)
    np.testing.assert_allclose(force(length=2.0), 2.0 * baseline, rtol=1.0e-14)
    np.testing.assert_allclose(force(speed=7.5), 3.0 * baseline, rtol=1.0e-14)
    amplitude = math.sqrt(2.0) * 0.01 / (2.0 * math.pi)
    saturation = 1.0 - (amplitude / modes[0].saturation_amplitude_ratio) ** 2
    expected_node_force_n = 0.5 * 1.2 * 2.5 * 0.2 * 1.0 * 2.0 * 0.5 * saturation * 0.1
    assert baseline[0, 0] == pytest.approx(expected_node_force_n, rel=1.0e-14)


def test_spectral_gain_replaces_fixed_activation_without_double_windowing():
    basis = _synthetic_basis()
    modes = _mode_configs(basis)
    spectral_gain = np.zeros((39, 2))
    spectral_gain[0, 0] = 0.75
    common = dict(
        modal_basis=basis,
        mode_configs=modes,
        rotation_body_to_global=np.eye(3),
        filtered_qdot_mps=(0.1, 0.0),
        qdot_rms_mps=(0.0, 0.0),
        # U*=1 for mode 6 is outside the fixed low=4 activation window.
        response_effective_crossflow_speed_mps=np.ones(39),
        response_section_width_m=np.ones(39),
        local_length_m=np.ones(39),
        fluid_density_kgpm3=1.2,
        flexible_velocity_global_mps=np.zeros((39, 3)),
        rigid_body_velocity_global_mps=np.zeros((39, 3)),
    )
    fixed_force, fixed_diagnostics = calculate_hvawt_response_force(**common)
    hybrid_force, hybrid_diagnostics = calculate_hvawt_response_force(
        spectral_gain_by_node_mode=spectral_gain,
        **common,
    )

    assert fixed_force[0, 0] == 0.0
    assert hybrid_force[0, 0] == pytest.approx(
        0.5 * 1.2 * 1.0 * 1.0 * 1.0 * 2.0 * 0.75 * 0.1
    )
    assert fixed_diagnostics[0]["activation_basis"] == "fixed_reduced_velocity"
    assert hybrid_diagnostics[0]["activation_basis"] == (
        "captured_airfoil_lift_spectrum"
    )


def test_core_spectral_gain_is_node_order_invariant_and_part_of_duplicate_identity():
    core = HVAWTResponseCore(_core_config())
    spectral_gain = np.zeros((39, 2))
    spectral_gain[0, 0] = 0.5
    inputs = _step_inputs(core, 0.0, qdot=(0.1, 0.0))
    first = core.step(
        0.0,
        spectral_gain_by_node_mode=spectral_gain,
        **inputs,
    )
    canonical_duplicate = core.step(
        0.0,
        spectral_gain_by_node_mode=spectral_gain,
        **inputs,
    )
    assert canonical_duplicate is first
    assert not canonical_duplicate.additive_force_global_n.flags.writeable
    order = np.arange(38, -1, -1)
    reordered_inputs = _step_inputs(core, 0.0, qdot=(0.1, 0.0), order=order)
    reordered = core.step(
        0.0,
        spectral_gain_by_node_mode=spectral_gain[order],
        **reordered_inputs,
    )
    np.testing.assert_allclose(
        reordered.additive_force_global_n,
        first.additive_force_global_n[order],
        rtol=0.0,
        atol=0.0,
    )

    changed = np.array(spectral_gain, copy=True)
    changed[0, 0] = 0.6
    with pytest.raises(ValueError, match="changed at the same time"):
        core.step(
            0.0,
            spectral_gain_by_node_mode=changed,
            **inputs,
        )


def test_real_tracked_basis_drives_both_real_three_dimensional_mode_forces():
    basis = _tracked_basis()
    modes = _mode_configs(
        basis,
        reduced_velocity_low=0.5,
        reduced_velocity_peak=2.0,
        reduced_velocity_high=3.0,
    )
    core = HVAWTResponseCore(_core_config(modal_basis=basis, modes=modes))
    core.step(0.0, **_step_inputs(core, 0.0, qdot=(0.0, 0.0)))
    result = core.step(
        0.01,
        **_step_inputs(core, 0.01, qdot=(0.12, -0.07)),
    )

    assert tuple(
        row.corrected_raw_qdot_mps for row in result.diagnostics.mode_diagnostics
    ) == pytest.approx((0.12, -0.07), rel=0.0, abs=3.0e-15)
    assert all(
        row.activation_max > 0.0 and row.maximum_mode_nodal_force_n > 0.0
        for row in result.diagnostics.mode_diagnostics
    )
    assert np.isrealobj(result.additive_force_global_n)
    assert np.max(np.abs(result.additive_force_global_n[:, 2])) > 0.0


def test_core_recovers_two_raw_coordinates_and_is_node_order_invariant():
    core = HVAWTResponseCore(_core_config())
    qdot = (0.25, -0.13)
    first = core.step(0.0, **_step_inputs(core, 0.0, qdot=qdot))
    assert tuple(
        mode.corrected_raw_qdot_mps for mode in first.diagnostics.mode_diagnostics
    ) == pytest.approx(qdot, rel=0.0, abs=2.0e-15)
    np.testing.assert_array_equal(first.additive_force_global_n, 0.0)

    order = np.arange(38, -1, -1)
    duplicate = core.step(
        0.0,
        **_step_inputs(core, 0.0, qdot=qdot, order=order),
    )
    np.testing.assert_array_equal(
        duplicate.flexible_velocity_global_mps,
        first.flexible_velocity_global_mps[order],
    )
    np.testing.assert_array_equal(
        duplicate.additive_force_global_n,
        first.additive_force_global_n[order],
    )


def test_core_precomputed_canonical_kinematics_and_flow_are_exact():
    reference = HVAWTResponseCore(_core_config())
    candidate = HVAWTResponseCore(_core_config())
    inputs = _step_inputs(reference, 0.0, qdot=(0.25, -0.13))
    kinematics = decompose_fixed_axis_velocity(
        candidate.config.kinematics,
        positions_global_m=inputs["positions_global_m"],
        total_structural_velocity_global_mps=inputs[
            "total_structural_velocity_global_mps"
        ],
        lss_rpm=inputs["lss_rpm"],
        azimuth_deg=inputs["azimuth_deg"],
    )
    flows = separate_hvawt_flows(
        ambient_velocity_global_mps=inputs["ambient_velocity_global_mps"],
        total_structural_velocity_global_mps=inputs[
            "total_structural_velocity_global_mps"
        ],
        rigid_body_velocity_global_mps=kinematics.rigid_body_velocity_global_mps,
        chord_unit_global=inputs["chord_unit_global"],
        normal_unit_global=inputs["normal_unit_global"],
        chord_m=inputs["chord_m"],
        thickness_ratio=inputs["thickness_ratio"],
    )
    expected = reference.step(0.0, **inputs)
    bound = candidate._bind_precomputed_step_state(
        node_ids=inputs["node_ids"],
        positions_global_m=inputs["positions_global_m"],
        total_structural_velocity_global_mps=inputs[
            "total_structural_velocity_global_mps"
        ],
        lss_rpm=inputs["lss_rpm"],
        azimuth_deg=inputs["azimuth_deg"],
        ambient_velocity_global_mps=inputs["ambient_velocity_global_mps"],
        chord_unit_global=inputs["chord_unit_global"],
        normal_unit_global=inputs["normal_unit_global"],
        chord_m=inputs["chord_m"],
        thickness_ratio=inputs["thickness_ratio"],
        kinematics=kinematics,
        flows=flows,
    )
    observed = candidate.step(
        0.0,
        **inputs,
        _precomputed_state=bound,
    )
    np.testing.assert_array_equal(
        observed.additive_force_global_n,
        expected.additive_force_global_n,
    )
    assert candidate.checkpoint_state() == reference.checkpoint_state()

    stale_inputs = dict(inputs)
    stale_inputs["positions_global_m"] = np.array(
        inputs["positions_global_m"], copy=True
    )
    with pytest.raises(TypeError, match="not bound"):
        HVAWTResponseCore(_core_config()).step(
            0.0,
            **stale_inputs,
            _precomputed_state=bound,
        )


def test_core_injects_flexible_energy_below_saturation_without_changing_response_flow():
    core = HVAWTResponseCore(_core_config())
    dt = 0.01
    results = []
    for index in range(801):
        time_s = index * dt
        qdot = 0.1 * math.sin(2.0 * math.pi * time_s)
        results.append(
            core.step(time_s, **_step_inputs(core, time_s, qdot=(qdot, 0.0)))
        )
    settled = results[-100:]

    assert np.mean([row.diagnostics.flexible_power_w for row in settled]) > 0.0
    assert min(
        row.diagnostics.mode_diagnostics[0].saturation_factor for row in settled
    ) > 0.0
    assert all(
        row.diagnostics.response_crossflow_speed_min_mps == pytest.approx(5.0)
        and row.diagnostics.response_crossflow_speed_max_mps == pytest.approx(5.0)
        for row in settled
    )
    assert max(
        row.diagnostics.base_crossflow_speed_max_mps for row in settled
    ) > 5.0


def test_core_removes_flexible_energy_above_saturation():
    modes = _mode_configs(saturation_amplitude_ratio=0.01)
    core = HVAWTResponseCore(_core_config(modes=modes))
    dt = 0.01
    results = []
    for index in range(601):
        time_s = index * dt
        qdot = 0.2 * math.sin(2.0 * math.pi * time_s)
        results.append(
            core.step(time_s, **_step_inputs(core, time_s, qdot=(qdot, 0.0)))
        )
    settled = results[-100:]
    assert max(
        row.diagnostics.mode_diagnostics[0].saturation_factor for row in settled
    ) < 0.0
    assert np.mean([row.diagnostics.flexible_power_w for row in settled]) < 0.0


def test_azimuth_rpm_sign_and_basis_rpm_mutations_fail_closed():
    core = HVAWTResponseCore(_core_config())
    core.step(0.0, **_step_inputs(core, 0.0))
    changed_azimuth = _step_inputs(core, 0.01)
    changed_azimuth["azimuth_deg"] += 0.5
    with pytest.raises(ValueError, match="Azimuth/LSS-RPM"):
        core.step(0.01, **changed_azimuth)

    fresh = HVAWTResponseCore(_core_config())
    outside = _step_inputs(fresh, 0.0)
    outside["lss_rpm"] = 1.02
    with pytest.raises(ValueError, match="outside the audited"):
        fresh.step(0.0, **outside)


def test_azimuth_rate_accepts_point_one_percent_but_rejects_more():
    rate_tolerance = 0.001 * RPM_TO_RADPS
    kinematics = _kinematics(maximum_azimuth_rate_error_radps=rate_tolerance)
    core = HVAWTResponseCore(_core_config(kinematics=kinematics))
    core.step(0.0, **_step_inputs(core, 0.0))
    inside = _step_inputs(core, 0.01)
    inside["azimuth_deg"] += math.degrees(0.9 * rate_tolerance * 0.01)
    core.step(0.01, **inside)

    fresh = HVAWTResponseCore(_core_config(kinematics=kinematics))
    fresh.step(0.0, **_step_inputs(fresh, 0.0))
    outside = _step_inputs(fresh, 0.01)
    outside["azimuth_deg"] += math.degrees(1.1 * rate_tolerance * 0.01)
    with pytest.raises(ValueError, match="Azimuth/LSS-RPM"):
        fresh.step(0.01, **outside)


def test_changed_same_time_and_force_limit_fail_transactionally():
    core = HVAWTResponseCore(_core_config())
    core.step(0.0, **_step_inputs(core, 0.0, qdot=(0.1, 0.0)))
    before = json.dumps(core.checkpoint_state(), sort_keys=True)
    changed = _step_inputs(core, 0.0, qdot=(0.2, 0.0))
    with pytest.raises(ValueError, match="same time"):
        core.step(0.0, **changed)
    assert json.dumps(core.checkpoint_state(), sort_keys=True) == before

    limited = HVAWTResponseCore(
        _core_config(max_abs_total_nodal_force_n=1.0e-12)
    )
    limited.step(0.0, **_step_inputs(limited, 0.0, qdot=(0.1, 0.0)))
    state_before_limit = json.dumps(limited.checkpoint_state(), sort_keys=True)
    with pytest.raises(ValueError, match="exceeds"):
        limited.step(
            0.25,
            **_step_inputs(limited, 0.25, qdot=(-0.1, 0.0)),
        )
    assert json.dumps(limited.checkpoint_state(), sort_keys=True) == state_before_limit


def test_checkpoint_round_trip_replays_identically_and_rejects_forgery():
    original = HVAWTResponseCore(_core_config())
    original.step(0.0, **_step_inputs(original, 0.0, qdot=(0.1, -0.02)))
    original.step(0.01, **_step_inputs(original, 0.01, qdot=(0.11, -0.01)))
    checkpoint = json.loads(json.dumps(original.checkpoint_state()))

    restored = HVAWTResponseCore(_core_config())
    restored.restore_checkpoint_state(checkpoint)
    expected = original.step(
        0.02,
        **_step_inputs(original, 0.02, qdot=(0.12, 0.0)),
    )
    actual = restored.step(
        0.02,
        **_step_inputs(restored, 0.02, qdot=(0.12, 0.0)),
    )
    np.testing.assert_array_equal(actual.additive_force_global_n, expected.additive_force_global_n)
    assert actual.diagnostics == expected.diagnostics

    forged = json.loads(json.dumps(checkpoint))
    forged["config_fingerprint"] = "f" * 64
    before = restored.checkpoint_state()
    with pytest.raises(ValueError, match="configuration mismatch"):
        restored.restore_checkpoint_state(forged)
    assert restored.checkpoint_state() == before


def test_checkpoint_rejects_rehashed_force_source_and_filter_state_corruption():
    source = HVAWTResponseCore(_core_config())
    source.step(0.0, **_step_inputs(source, 0.0, qdot=(0.0, 0.0)))
    source.step(0.01, **_step_inputs(source, 0.01, qdot=(0.1, -0.03)))
    checkpoint = json.loads(json.dumps(source.checkpoint_state()))

    mutations = []
    changed_force = json.loads(json.dumps(checkpoint))
    changed_force["last_result"]["additive_force_global_n"][0][0] = 9.0e99
    mutations.append(changed_force)
    changed_source = json.loads(json.dumps(checkpoint))
    changed_source["last_result"]["diagnostics"][
        "force_source_mode"
    ] = EXTERNAL_VIV_LEGACY
    mutations.append(changed_source)
    changed_filter = json.loads(json.dumps(checkpoint))
    changed_filter["bandpass_states"][0]["previous_input_mps"] += 0.01
    mutations.append(changed_filter)

    for forged in mutations:
        _rehash_checkpoint(forged)
        target = HVAWTResponseCore(_core_config())
        before = target.checkpoint_state()
        with pytest.raises(ValueError):
            target.restore_checkpoint_state(forged)
        assert target.checkpoint_state() == before


def test_checkpoint_rejects_rehashed_nonpristine_state_without_history():
    pristine = HVAWTResponseCore(_core_config()).checkpoint_state()
    pristine["bandpass_states"][0]["position_state"] = 0.25
    _rehash_checkpoint(pristine)
    target = HVAWTResponseCore(_core_config())
    with pytest.raises(ValueError, match="non-pristine"):
        target.restore_checkpoint_state(pristine)


def test_checkpoint_rejects_unknown_fields_and_unrehashed_mutation():
    core = HVAWTResponseCore(_core_config())
    checkpoint = core.checkpoint_state()
    extra = json.loads(json.dumps(checkpoint))
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="keys differ"):
        core.restore_checkpoint_state(extra)
    changed = json.loads(json.dumps(checkpoint))
    changed["qdot_mean_square_m2ps2"][0] = 1.0
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        core.restore_checkpoint_state(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        {"rpm_sign": 0},
        {"azimuth_sign": 0},
        {"axis_motion_model": "moving"},
        {"position_basis": "reference_positions"},
        {"body_to_global_at_reference": ((1.0, 0.0, 0.0),) * 3},
    ],
)
def test_kinematic_contract_rejects_ambiguous_or_unsupported_semantics(mutation):
    with pytest.raises(ValueError):
        _kinematics(**mutation)


def test_modal_basis_rejects_naive_real_reduction_when_complex_gate_fails():
    basis = _synthetic_basis()
    shape = np.array(basis.complex_shape_body_per_m, copy=True)
    shape[0, 1, 0] = 0.03j
    maximum = np.max(np.linalg.norm(shape[:, :, 0], axis=1))
    shape[:, :, 0] /= maximum
    dual_flat = np.linalg.pinv(shape.reshape(117, 2)).conj().T
    dual = dual_flat.reshape(39, 3, 2)
    with pytest.raises(ValueError, match="imaginary/real"):
        HVAWTModalBasis(
            node_ids=basis.node_ids,
            tracked_modes=(6, 12),
            basis_rpm=1.0,
            complex_shape_body_per_m=shape,
            complex_dual_body=dual,
            natural_frequency_hz=basis.natural_frequency_hz,
            generalized_modal_mass_kg=basis.generalized_modal_mass_kg,
            generalized_modal_stiffness_npm=basis.generalized_modal_stiffness_npm,
            identification_sha256=ZERO_SHA256,
            shapes_sha256=ZERO_SHA256,
            duals_sha256=ZERO_SHA256,
            review_marker_sha256=ZERO_SHA256,
        )


def test_config_rejects_total_damping_label_and_property_drift():
    basis = _synthetic_basis()
    with pytest.raises(ValueError, match="incremental_viv_screening"):
        replace(_mode_configs(basis)[0], coefficient_role="total_aerodynamic_damping")
    drifted = replace(
        _mode_configs(basis)[0],
        natural_frequency_hz=1.01,
        generalized_modal_stiffness_npm=10.0 * (2.0 * math.pi * 1.01) ** 2,
    )
    with pytest.raises(ValueError, match="differ from the modal artifact"):
        _core_config(modal_basis=basis, modes=(drifted, _mode_configs(basis)[1]))
    with pytest.raises(TypeError, match="real numeric scalar"):
        replace(_mode_configs(basis)[0], natural_frequency_hz="1.0")
