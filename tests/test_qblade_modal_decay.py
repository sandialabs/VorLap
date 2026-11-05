from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vorlap.qblade_modal_decay import (
    AZIMUTH_CHANNEL,
    EXPECTED_ACTION_COUNT,
    MODAL_BURST_CHECKPOINT_SCHEMA,
    MODAL_BURST_PLAN_SCHEMA,
    TIME_CHANNEL,
    ModalBurstPlan,
    SilModalBurstCoupler,
    _validate_qblade_pre_time,
)


SHAPE_FIELDS = [
    "rpm",
    "tracked_mode",
    "frequency_hz",
    "action_node_id",
    "qblade_component",
    "qblade_fraction",
    "qblade_force_output_index_x",
    "qblade_force_output_index_y",
    "qblade_force_output_index_z",
    "ux_per_m_real",
    "ux_per_m_imag",
    "uy_per_m_real",
    "uy_per_m_imag",
    "uz_per_m_real",
    "uz_per_m_imag",
]

DUAL_FIELDS = [
    "rpm",
    "target_mode",
    "target_basis_column_zero_based",
    "action_site_row_zero_based",
    "action_node_id",
    "qblade_component",
    "qblade_fraction",
    "qblade_force_output_index_x",
    "qblade_force_output_index_y",
    "qblade_force_output_index_z",
    "force_x_n_per_generalized_force_real",
    "force_x_n_per_generalized_force_imag",
    "force_y_n_per_generalized_force_real",
    "force_y_n_per_generalized_force_imag",
    "force_z_n_per_generalized_force_real",
    "force_z_n_per_generalized_force_imag",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_synthetic_artifacts(root: Path) -> dict[str, object]:
    target_mode = 6
    target_frequency_hz = 1.0
    target_phase_rad = 0.47
    node_ids = [f"NODE_{index:02d}_0.500000" for index in range(39)]
    external_inputs: list[dict[str, object]] = [
        {"index": 0, "variable": TIME_CHANNEL},
        {"index": 1, "variable": "Timestep [-]"},
        {"index": 2, "variable": AZIMUTH_CHANNEL},
    ]
    for site_index, node_id in enumerate(node_ids):
        for axis_offset, axis in enumerate("XYZ"):
            external_inputs.append(
                {
                    "index": 3 + 3 * site_index + axis_offset,
                    "variable": f"{axis}_g Vel. {node_id} [m/s]",
                }
            )
    external_outputs: list[dict[str, object]] = []
    for site_index in range(39):
        for axis_offset, axis in enumerate("XYZ"):
            external_outputs.append(
                {
                    "action": "ADDFORCE",
                    "direction": axis,
                    "id": f"NODE_{site_index:02d}",
                    "index": 120 + 3 * site_index + axis_offset,
                    "local": False,
                    "position": 0.5,
                }
            )
    external = root / "external.json"
    external.write_text(
        json.dumps(
            {
                "swap_size": 237,
                "node_ids": node_ids,
                "external_input_rows": external_inputs,
                "external_output_rows": external_outputs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    phi = np.zeros((117, 12), dtype=complex)
    for mode in range(1, 13):
        phase = target_phase_rad if mode == target_mode else 0.03 * mode
        phi[3 * (mode - 1), mode - 1] = complex(math.cos(phase), math.sin(phase))
    shape_rows: list[dict[str, object]] = []
    for mode in range(1, 13):
        frequency = 0.4 + 0.1 * mode
        for site_index, node_id in enumerate(node_ids):
            values = phi[3 * site_index : 3 * site_index + 3, mode - 1]
            shape_rows.append(
                {
                    "rpm": 1.0,
                    "tracked_mode": mode,
                    "frequency_hz": frequency,
                    "action_node_id": node_id,
                    "qblade_component": f"NODE_{site_index:02d}",
                    "qblade_fraction": 0.5,
                    "qblade_force_output_index_x": 120 + 3 * site_index,
                    "qblade_force_output_index_y": 121 + 3 * site_index,
                    "qblade_force_output_index_z": 122 + 3 * site_index,
                    "ux_per_m_real": values[0].real,
                    "ux_per_m_imag": values[0].imag,
                    "uy_per_m_real": values[1].real,
                    "uy_per_m_imag": values[1].imag,
                    "uz_per_m_real": values[2].real,
                    "uz_per_m_imag": values[2].imag,
                }
            )
    shapes = root / "shapes.csv"
    write_csv(shapes, SHAPE_FIELDS, shape_rows)

    # For this phase-only orthonormal basis, D_target == Phi_target and
    # therefore Phi^H D_target == e_target exactly.
    dual = np.array(phi[:, target_mode - 1], copy=True)
    dual_rows: list[dict[str, object]] = []
    for site_index, node_id in enumerate(node_ids):
        values = dual[3 * site_index : 3 * site_index + 3]
        dual_rows.append(
            {
                "rpm": 1.0,
                "target_mode": target_mode,
                "target_basis_column_zero_based": target_mode - 1,
                "action_site_row_zero_based": site_index,
                "action_node_id": node_id,
                "qblade_component": f"NODE_{site_index:02d}",
                "qblade_fraction": 0.5,
                "qblade_force_output_index_x": 120 + 3 * site_index,
                "qblade_force_output_index_y": 121 + 3 * site_index,
                "qblade_force_output_index_z": 122 + 3 * site_index,
                "force_x_n_per_generalized_force_real": values[0].real,
                "force_x_n_per_generalized_force_imag": values[0].imag,
                "force_y_n_per_generalized_force_real": values[1].real,
                "force_y_n_per_generalized_force_imag": values[1].imag,
                "force_z_n_per_generalized_force_real": values[2].real,
                "force_z_n_per_generalized_force_imag": values[2].imag,
            }
        )
    duals = root / "duals.csv"
    write_csv(duals, DUAL_FIELDS, dual_rows)
    return {
        "external": external,
        "shapes": shapes,
        "duals": duals,
        "phi": phi,
        "dual": dual,
        "target_mode": target_mode,
        "target_frequency_hz": target_frequency_hz,
        "target_phase_rad": target_phase_rad,
    }


def make_plan(**updates: object) -> ModalBurstPlan:
    values: dict[str, object] = {
        "timestep_s": 0.05,
        "total_wrapper_updates": 105,
        "bootstrap_zero": True,
        "rotor_speed_rpm": 1.0,
        "target_mode": 6,
        "frequency_hz": 1.0,
        "generalized_force_amplitude_n": 20.0,
        "burst_start_time_s": 0.1,
        "burst_cycles": 5,
        "azimuth_reference_deg": 0.0,
        "rotation_sign": 1,
    }
    values.update(updates)
    return ModalBurstPlan(**values)  # type: ignore[arg-type]


def make_coupler(artifacts: dict[str, object], **plan_updates: object) -> SilModalBurstCoupler:
    return SilModalBurstCoupler.from_artifacts(
        plan=make_plan(**plan_updates),
        external_config_path=Path(artifacts["external"]),
        action_shapes_path=Path(artifacts["shapes"]),
        force_duals_path=Path(artifacts["duals"]),
    )


@pytest.fixture
def synthetic_artifacts(tmp_path: Path) -> dict[str, object]:
    return write_synthetic_artifacts(tmp_path)


class RecordingQBlade:
    def __init__(self) -> None:
        self.time_s = 0.0
        self.azimuth_deg = 0.0
        self.current_step = 0
        self.actions: list[tuple[int, bytes, bytes, float, float, bytes, bool, int]] = []

    def getCustomData_at_num(self, variable: bytes, _position: float, _index: int) -> float:
        if variable == TIME_CHANNEL.encode("utf-8"):
            return self.time_s
        if variable == AZIMUTH_CHANNEL.encode("utf-8"):
            return self.azimuth_deg
        return float("nan")

    def setExternalAction(
        self,
        action: bytes,
        component: bytes,
        value: float,
        position: float,
        direction: bytes,
        local: bool,
        index: int,
    ) -> None:
        self.actions.append(
            (
                self.current_step,
                action,
                component,
                value,
                position,
                direction,
                local,
                index,
            )
        )


def advance_fake(qblade: RecordingQBlade, coupler: SilModalBurstCoupler, step: int) -> str:
    qblade.current_step = step
    physical_sample = None if step == 0 else step - 1
    qblade.time_s = (
        0.0 if physical_sample is None else physical_sample * coupler.plan.timestep_s
    )
    qblade.azimuth_deg = 0.0 if physical_sample is None else 3.0 * physical_sample
    return coupler.step(qblade, step)


def frequency_map(coupler: SilModalBurstCoupler) -> dict[int, float]:
    return dict(zip(coupler.basis_modes, coupler.basis_frequencies_hz))


def direct_clone(
    coupler: SilModalBurstCoupler,
    *,
    plan: ModalBurstPlan | None = None,
    actions=None,
    action_basis: np.ndarray | None = None,
    force_dual: np.ndarray | None = None,
    basis_frequencies_hz: dict[int, float] | None = None,
    artifact_identities=None,
) -> SilModalBurstCoupler:
    return SilModalBurstCoupler(
        plan=coupler.plan if plan is None else plan,
        actions=coupler.actions if actions is None else actions,
        action_basis=coupler.action_basis if action_basis is None else action_basis,
        force_dual=coupler.force_dual if force_dual is None else force_dual,
        basis_frequencies_hz=(
            frequency_map(coupler)
            if basis_frequencies_hz is None
            else basis_frequencies_hz
        ),
        artifact_identities=(
            coupler.artifact_identities
            if artifact_identities is None
            else artifact_identities
        ),
    )


def test_plan_requires_bootstrap_rotation_sign_grid_alignment_and_postburst_zero():
    with pytest.raises(ValueError, match="bootstrap_zero"):
        make_plan(bootstrap_zero=False)
    with pytest.raises(ValueError, match="explicitly set"):
        make_plan(rotation_sign=0)
    with pytest.raises(ValueError, match="timestep grid"):
        make_plan(burst_start_time_s=0.125)
    with pytest.raises(ValueError, match="after the burst"):
        make_plan(total_wrapper_updates=104)


def test_plan_round_trips_with_canonical_types_and_verified_derived_fields():
    plan = make_plan()
    payload = json.loads(json.dumps(plan.as_dict(), sort_keys=True))
    restored = ModalBurstPlan.from_mapping(payload)

    assert restored == plan
    assert type(restored.timestep_s) is float
    assert type(restored.total_wrapper_updates) is int
    assert type(restored.bootstrap_zero) is bool
    assert type(restored.target_mode) is int
    assert type(restored.burst_cycles) is int
    assert type(restored.minimum_samples_per_cycle) is int
    assert type(restored.maximum_realized_burst_cycle_error) is float
    assert type(payload["total_physical_samples"]) is int

    inconsistent = dict(payload)
    inconsistent["burst_end_sample"] += 1
    with pytest.raises(ValueError, match="derived field"):
        ModalBurstPlan.from_mapping(inconsistent)
    noncanonical = dict(payload)
    noncanonical["total_physical_samples"] = float(payload["total_physical_samples"])
    with pytest.raises(ValueError, match="derived field"):
        ModalBurstPlan.from_mapping(noncanonical)
    with pytest.raises(ValueError, match="unsupported field.*artifacts"):
        ModalBurstPlan.from_mapping({**payload, "artifacts": 42})


def test_plan_enforces_nyquist_resolution_and_realized_cycle_error_gates():
    with pytest.raises(ValueError, match="Nyquist"):
        make_plan(
            timestep_s=1.0,
            frequency_hz=0.5,
            burst_start_time_s=0.0,
            total_wrapper_updates=20,
        )
    with pytest.raises(ValueError, match="under-resolved"):
        make_plan(
            timestep_s=0.0251,
            frequency_hz=2.0,
            burst_start_time_s=0.0,
            total_wrapper_updates=200,
        )
    with pytest.raises(ValueError, match="realized-cycle-error"):
        make_plan(
            timestep_s=1.0 / 20.3,
            frequency_hz=1.0,
            burst_start_time_s=0.0,
            total_wrapper_updates=200,
        )
    with pytest.raises(ValueError, match="at least 20"):
        make_plan(minimum_samples_per_cycle=19)
    with pytest.raises(ValueError, match="cannot weaken"):
        make_plan(maximum_realized_burst_cycle_error=0.02)


def test_hann_burst_is_exactly_zero_before_at_endpoints_and_after(synthetic_artifacts):
    coupler = make_coupler(synthetic_artifacts)
    plan = coupler.plan
    samples = [
        coupler.force_sample(0, azimuth_deg=0.0),
        coupler.force_sample(plan.burst_start_sample, azimuth_deg=0.0),
        coupler.force_sample(plan.burst_end_sample, azimuth_deg=0.0),
        coupler.force_sample(plan.burst_end_sample + 1, azimuth_deg=0.0),
    ]

    assert all(sample.scheduled_zero for sample in samples)
    assert all(sample.hann_weight == 0.0 for sample in samples)
    assert all(all(value == 0.0 for value in sample.body_force_n) for sample in samples)
    assert all(all(value == 0.0 for value in sample.global_force_n) for sample in samples)
    assert len({sample.global_force_sha256 for sample in samples}) == 1


def test_zero_amplitude_baseline_explicitly_stays_zero_inside_nominal_burst(
    synthetic_artifacts,
):
    coupler = make_coupler(synthetic_artifacts, generalized_force_amplitude_n=0.0)
    interior = coupler.force_sample(coupler.plan.burst_start_sample + 9, azimuth_deg=41.0)

    assert interior.scheduled_zero is True
    assert interior.hann_weight == 0.0
    assert all(value == 0.0 for value in interior.body_force_n)
    assert all(value == 0.0 for value in interior.global_force_n)
    assert coupler.generalized_force_audit["passed"] is True


def test_hann_interior_uses_real_and_imaginary_force_dual_quadratures(
    synthetic_artifacts,
):
    coupler = make_coupler(synthetic_artifacts)
    k = 1
    sample_index = coupler.plan.burst_start_sample + k
    sample = coupler.force_sample(sample_index, azimuth_deg=0.0)
    phase = 2.0 * math.pi * coupler.plan.frequency_hz * k * coupler.plan.timestep_s
    weight = 0.5 * (
        1.0 - math.cos(2.0 * math.pi * k / (coupler.plan.burst_sample_count - 1))
    )
    dual_phase = float(synthetic_artifacts["target_phase_rad"])
    expected = (
        coupler.plan.generalized_force_amplitude_n
        * weight
        * (
            math.cos(dual_phase) * math.cos(phase)
            - math.sin(dual_phase) * math.sin(phase)
        )
    )
    target_component = 3 * (coupler.plan.target_mode - 1)

    assert sample.scheduled_zero is False
    assert sample.hann_weight == pytest.approx(weight)
    assert sample.body_force_n[target_component] == pytest.approx(expected)
    assert np.count_nonzero(sample.body_force_n) == 1
    real_only_value = (
        coupler.plan.generalized_force_amplitude_n
        * weight
        * math.cos(dual_phase)
        * math.cos(phase)
    )
    assert sample.body_force_n[target_component] != pytest.approx(real_only_value)


def test_pure_force_record_is_a_positive_frequency_design_not_an_application(
    synthetic_artifacts,
):
    coupler = make_coupler(synthetic_artifacts)
    sample = coupler.force_sample(coupler.plan.burst_start_sample + 1, azimuth_deg=8.0)
    record = sample.design_record()

    assert record["record_kind"] == "designed_force_sample"
    assert "positive-frequency body-frame" in record["complex_phasor_convention"]
    assert "application_wrapper_step_index" not in record
    assert "action_application_completed" not in record
    assert "applied_action_count" not in record


def test_body_to_global_rotation_uses_the_explicit_sign_and_preserves_z(
    synthetic_artifacts,
):
    positive = make_coupler(synthetic_artifacts, rotation_sign=1)
    negative = direct_clone(positive, plan=replace(positive.plan, rotation_sign=-1))
    sample_index = positive.plan.burst_start_sample + 1
    plus_sample = positive.force_sample(sample_index, azimuth_deg=90.0)
    minus_sample = negative.force_sample(sample_index, azimuth_deg=90.0)
    site = positive.plan.target_mode - 1
    body_x = plus_sample.body_force_n[3 * site]

    assert plus_sample.global_force_n[3 * site] == pytest.approx(0.0, abs=1.0e-15)
    assert plus_sample.global_force_n[3 * site + 1] == pytest.approx(body_x)
    assert minus_sample.global_force_n[3 * site] == pytest.approx(0.0, abs=1.0e-15)
    assert minus_sample.global_force_n[3 * site + 1] == pytest.approx(-body_x)
    assert plus_sample.global_force_n[3 * site + 2] == plus_sample.body_force_n[3 * site + 2]
    assert (
        positive.checkpoint_identity()["fingerprint"]
        != negative.checkpoint_identity()["fingerprint"]
    )


def test_complex_phihf_audit_is_applied_to_every_nonzero_schedule_sample(
    synthetic_artifacts,
):
    coupler = make_coupler(synthetic_artifacts)
    sample = coupler.force_sample(coupler.plan.burst_start_sample + 7, azimuth_deg=37.0)

    assert sample.target_complex_generalized_force_phasor_error_n < 1.0e-12
    assert sample.maximum_off_target_complex_generalized_force_phasor_n < 1.0e-12
    assert coupler.generalized_force_audit["passed"] is True


def test_constructor_rejects_target_and_offtarget_force_dual_failures(
    synthetic_artifacts,
):
    good = make_coupler(synthetic_artifacts)
    target_bad = np.array(good.force_dual, copy=True)
    target_bad *= 0.9
    with pytest.raises(ValueError, match="target gate"):
        direct_clone(good, force_dual=target_bad)

    off_target_bad = np.array(good.force_dual, copy=True)
    off_target_bad[0] = 0.01 * good.action_basis[0, 0]
    with pytest.raises(ValueError, match="off-target gate"):
        direct_clone(good, force_dual=off_target_bad)


def test_nontarget_basis_permutation_preserves_force_audit(synthetic_artifacts):
    original = make_coupler(synthetic_artifacts)
    swapped_basis = np.array(original.action_basis, copy=True)
    swapped_basis[:, [0, 1]] = swapped_basis[:, [1, 0]]
    swapped = direct_clone(original, action_basis=swapped_basis)

    original_audit = dict(original.generalized_force_audit)
    swapped_audit = dict(swapped.generalized_force_audit)
    # This phase-only orthonormal basis has condition number one. LAPACK
    # may round its singular values differently after a column permutation.
    for audit in (original_audit, swapped_audit):
        assert audit.pop("condition_number") == pytest.approx(
            1.0, rel=8 * np.finfo(float).eps, abs=0.0
        )
    assert swapped_audit == original_audit

    sample = original.plan.burst_start_sample + 7
    original_force = original.force_sample(sample, azimuth_deg=37.0)
    swapped_force = swapped.force_sample(sample, azimuth_deg=37.0)
    np.testing.assert_array_equal(swapped_force.body_force_n, original_force.body_force_n)
    np.testing.assert_array_equal(swapped_force.global_force_n, original_force.global_force_n)


def test_identity_binds_complete_complex_basis_and_basis_frequency_order(
    synthetic_artifacts,
):
    loaded = make_coupler(synthetic_artifacts)
    # Exact coordinate vectors and Gaussian unit phases keep the audit
    # identical, so fingerprint changes must bind the basis, not SVD roundoff.
    exact_basis = np.zeros_like(loaded.action_basis)
    columns = np.arange(exact_basis.shape[1])
    exact_basis[3 * columns, columns] = np.array([1, 1j, -1, -1j])[columns % 4]
    original = direct_clone(
        loaded,
        action_basis=exact_basis,
        force_dual=exact_basis[:, loaded.target_basis_column],
    )
    swapped_basis = np.array(original.action_basis, copy=True)
    swapped_basis[:, [0, 1]] = swapped_basis[:, [1, 0]]
    swapped = direct_clone(original, action_basis=swapped_basis)
    changed_frequencies = frequency_map(original)
    changed_frequencies[1] += 0.001
    frequency_changed = direct_clone(
        original,
        basis_frequencies_hz=changed_frequencies,
    )

    assert swapped.generalized_force_audit == original.generalized_force_audit
    assert (
        swapped.checkpoint_identity()["action_basis_sha256"]
        != original.checkpoint_identity()["action_basis_sha256"]
    )
    assert (
        swapped.checkpoint_identity()["fingerprint"]
        != original.checkpoint_identity()["fingerprint"]
    )
    assert (
        frequency_changed.checkpoint_identity()["basis_frequencies_sha256"]
        != original.checkpoint_identity()["basis_frequencies_sha256"]
    )
    assert (
        frequency_changed.checkpoint_identity()["fingerprint"]
        != original.checkpoint_identity()["fingerprint"]
    )
    qblade = RecordingQBlade()
    advance_fake(qblade, original, 0)
    with pytest.raises(ValueError, match="identity"):
        swapped.restore_checkpoint_state(original.checkpoint_state())


def test_constructor_rejects_noncanonical_artifact_identity_and_action_position(
    synthetic_artifacts,
):
    good = make_coupler(synthetic_artifacts)
    malformed_identity = dict(good.artifact_identities[0])
    malformed_identity.pop("size_bytes")
    with pytest.raises(ValueError, match="exactly role"):
        direct_clone(good, artifact_identities=(malformed_identity,))
    with pytest.raises(ValueError, match="duplicated"):
        direct_clone(
            good,
            artifact_identities=(
                good.artifact_identities[0],
                good.artifact_identities[0],
            ),
        )

    actions = list(good.actions)
    actions[0] = replace(actions[0], position=1.01)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        direct_clone(good, actions=actions)


def test_constructor_requires_unique_site_ids_and_physical_targets(synthetic_artifacts):
    good = make_coupler(synthetic_artifacts)
    duplicate_node_ids = list(good.actions)
    for axis in range(3):
        duplicate_node_ids[3 + axis] = replace(
            duplicate_node_ids[3 + axis],
            node_id=good.actions[axis].node_id,
        )
    with pytest.raises(ValueError, match="39 unique site node_ids"):
        direct_clone(good, actions=duplicate_node_ids)

    duplicate_targets = list(good.actions)
    for axis in range(3):
        duplicate_targets[3 + axis] = replace(
            duplicate_targets[3 + axis],
            component_id=good.actions[axis].component_id,
            position=good.actions[axis].position,
        )
    with pytest.raises(ValueError, match=r"39 unique \(component_id, position\)"):
        direct_clone(good, actions=duplicate_targets)


def test_bootstrap_and_physical_counters_follow_observed_qblade_pre_time(
    synthetic_artifacts,
):
    coupler = make_coupler(synthetic_artifacts)
    qblade = RecordingQBlade()

    bootstrap_message = advance_fake(qblade, coupler, 0)
    assert qblade.time_s == 0.0
    assert coupler.last_record["bootstrap_zero"] is True
    assert coupler.last_record["physical_sample_index"] is None
    assert all(row[3] == 0.0 for row in qblade.actions[-EXPECTED_ACTION_COUNT:])
    assert "bootstrap-zero" in bootstrap_message

    first_physical_message = advance_fake(qblade, coupler, 1)
    assert qblade.time_s == 0.0
    assert coupler.last_record["bootstrap_zero"] is False
    assert coupler.last_record["physical_sample_index"] == 0
    assert coupler.last_record["wrapper_step_index"] == 1
    assert "physical-zero" in first_physical_message
    assert coupler.wrapper_updates == 2
    assert coupler.bootstrap_updates == 1
    assert coupler.physical_samples_applied == 1


def test_float32_qblade_pre_times_pass_step_restore_and_long_time_grid_audit(
    synthetic_artifacts,
):
    coupler = make_coupler(
        synthetic_artifacts,
        timestep_s=0.0005,
        total_wrapper_updates=10203,
    )
    qblade = RecordingQBlade()
    advance_fake(qblade, coupler, 0)
    advance_fake(qblade, coupler, 1)
    qblade.current_step = 2
    qblade.time_s = 0.0005000000237487257
    qblade.azimuth_deg = 3.0
    coupler.step(qblade, 2)

    audit = coupler.last_record["qblade_pre_time_grid_audit"]
    assert audit["expected_physical_sample_index"] == 1
    assert audit["nearest_observed_grid_index"] == 1
    assert audit["absolute_residual_s"] == pytest.approx(2.374872564228636e-11)
    assert audit["passed"] is True
    restored = make_coupler(
        synthetic_artifacts,
        timestep_s=0.0005,
        total_wrapper_updates=10203,
    )
    restored.restore_checkpoint_state(coupler.checkpoint_state())
    assert restored.last_record == coupler.last_record

    long_audit = _validate_qblade_pre_time(
        119.99400329589844,
        timestep_s=0.0005,
        expected_physical_sample_index=239988,
    )
    assert long_audit["expected_time_s"] == pytest.approx(119.994)
    assert long_audit["nearest_observed_grid_index"] == 239988
    assert long_audit["absolute_residual_s"] == pytest.approx(3.2958984437223553e-6)
    assert long_audit["absolute_residual_s"] < long_audit["float32_residual_limit_s"]


def test_float32_grid_mapping_accepts_quantized_sequence_and_rejects_aliases():
    timestep_s = 0.0005
    for sample_index in (0, 1, 2, 199, 200, 10_000, 239_988):
        observed = float(np.float32(sample_index * timestep_s))
        audit = _validate_qblade_pre_time(
            observed,
            timestep_s=timestep_s,
            expected_physical_sample_index=sample_index,
        )
        assert audit["nearest_observed_grid_index"] == sample_index
        assert audit["passed"] is True

    sample_index = 239_988
    expected = sample_index * timestep_s
    with pytest.raises(RuntimeError, match="nearest_observed_index=239989"):
        _validate_qblade_pre_time(
            expected + timestep_s,
            timestep_s=timestep_s,
            expected_physical_sample_index=sample_index,
        )
    ulp32 = abs(float(np.spacing(np.float32(expected))))
    with pytest.raises(RuntimeError, match="float32_limit"):
        _validate_qblade_pre_time(
            expected + 5.0 * ulp32,
            timestep_s=timestep_s,
            expected_physical_sample_index=sample_index,
        )


def test_step_applies_current_designed_force_in_same_call_with_no_software_queue(
    synthetic_artifacts,
):
    coupler = make_coupler(synthetic_artifacts)
    qblade = RecordingQBlade()
    physical_sample = coupler.plan.burst_start_sample + 1
    wrapper_step = physical_sample + 1
    for step in range(wrapper_step + 1):
        message = advance_fake(qblade, coupler, step)

    expected = coupler.force_sample(
        physical_sample,
        azimuth_deg=3.0 * physical_sample,
    )
    applied = qblade.actions[-EXPECTED_ACTION_COUNT:]
    assert [row[0] for row in applied] == [wrapper_step] * EXPECTED_ACTION_COUNT
    assert [row[3] for row in applied] == pytest.approx(expected.global_force_n)
    assert any(row[3] != 0.0 for row in applied)
    assert coupler.last_record["generation_wrapper_step_index"] == wrapper_step
    assert coupler.last_record["application_wrapper_step_index"] == wrapper_step
    assert coupler.last_record["software_force_queue_depth"] == 0
    assert coupler.last_record["generation_and_application_same_wrapper_call"] is True
    assert coupler.last_record["action_application_completed"] is True
    assert coupler.last_record["applied_action_count"] == EXPECTED_ACTION_COUNT
    assert "no software queue" in message
    assert "pending_force_packets" not in coupler.__dict__


def test_step_records_application_only_after_all_117_calls_succeed(synthetic_artifacts):
    class FailingQBlade(RecordingQBlade):
        def setExternalAction(self, *args) -> None:
            if len(self.actions) == 10:
                raise RuntimeError("synthetic action failure")
            super().setExternalAction(*args)

    coupler = make_coupler(synthetic_artifacts)
    qblade = FailingQBlade()
    with pytest.raises(RuntimeError, match="synthetic action failure"):
        coupler.step(qblade, 0)

    assert len(qblade.actions) == 10
    assert coupler.wrapper_updates == 0
    assert coupler.bootstrap_updates == 0
    assert coupler.physical_samples_applied == 0
    assert coupler.last_record is None
    assert coupler.checkpoint_state()["last_record"] is None


def test_step_rejects_old_step_times_and_noncontiguous_sequence_before_applying(
    synthetic_artifacts,
):
    wrong_bootstrap = make_coupler(synthetic_artifacts)
    qblade = RecordingQBlade()
    qblade.time_s = 0.1
    with pytest.raises(RuntimeError, match="pre-advance time"):
        wrong_bootstrap.step(qblade, 0)
    assert wrong_bootstrap.wrapper_updates == 0
    assert qblade.actions == []

    noncontiguous = make_coupler(synthetic_artifacts)
    with pytest.raises(RuntimeError, match="contiguous"):
        noncontiguous.step(qblade, 1)
    assert noncontiguous.wrapper_updates == 0

    old_assumption = make_coupler(synthetic_artifacts)
    qblade_old = RecordingQBlade()
    advance_fake(qblade_old, old_assumption, 0)
    qblade_old.time_s = old_assumption.plan.timestep_s
    with pytest.raises(RuntimeError, match="expected_sample=0"):
        old_assumption.step(qblade_old, 1)
    assert old_assumption.wrapper_updates == 1


def test_checkpoint_restore_continues_identically_without_pending_force_packet(
    synthetic_artifacts,
):
    uninterrupted = make_coupler(synthetic_artifacts)
    qblade_full = RecordingQBlade()
    for step in range(uninterrupted.plan.total_wrapper_updates):
        advance_fake(qblade_full, uninterrupted, step)
    uninterrupted_summary = uninterrupted.finalize()

    first = make_coupler(synthetic_artifacts)
    qblade_first = RecordingQBlade()
    split_wrapper_step = first.plan.burst_start_sample + 9
    for step in range(split_wrapper_step):
        advance_fake(qblade_first, first, step)
    state = first.checkpoint_state()

    assert state["schema"] == MODAL_BURST_CHECKPOINT_SCHEMA
    assert state["software_force_queue_depth"] == 0
    assert state["pending_force_packets"] == []
    resumed = make_coupler(synthetic_artifacts)
    resumed.restore_checkpoint_state(state)
    assert resumed.lifecycle == "restored"
    qblade_resumed = RecordingQBlade()
    for step in range(split_wrapper_step, resumed.plan.total_wrapper_updates):
        advance_fake(qblade_resumed, resumed, step)
    resumed_summary = resumed.finalize()

    assert resumed_summary == uninterrupted_summary
    assert resumed.last_record == uninterrupted.last_record
    assert (
        resumed.scheduled_nonzero_physical_samples
        == uninterrupted.scheduled_nonzero_physical_samples
    )
    assert resumed.exact_zero_physical_samples == uninterrupted.exact_zero_physical_samples


def test_checkpoint_restore_is_atomic_and_rejects_changed_identity(synthetic_artifacts):
    source = make_coupler(synthetic_artifacts)
    qblade = RecordingQBlade()
    for step in range(4):
        advance_fake(qblade, source, step)
    state = source.checkpoint_state()

    changed_plan = make_coupler(synthetic_artifacts, rotation_sign=-1)
    with pytest.raises(ValueError, match="identity"):
        changed_plan.restore_checkpoint_state(state)
    assert changed_plan.checkpoint_state()["wrapper_updates"] == 0

    corrupt = json.loads(json.dumps(state))
    corrupt["last_record"]["global_force_sha256"] = "0" * 64
    fresh = make_coupler(synthetic_artifacts)
    before = fresh.checkpoint_state()
    with pytest.raises(ValueError, match="global_force_sha256"):
        fresh.restore_checkpoint_state(corrupt)
    assert fresh.checkpoint_state() == before


def test_checkpoint_restore_rejects_all_unregistered_top_record_and_audit_keys(
    synthetic_artifacts,
):
    source = make_coupler(synthetic_artifacts)
    qblade = RecordingQBlade()
    for step in range(4):
        advance_fake(qblade, source, step)
    state = source.checkpoint_state()
    fresh = make_coupler(synthetic_artifacts)
    before = fresh.checkpoint_state()

    top_extra = json.loads(json.dumps(state))
    top_extra["qblade_force_consumption_verified"] = True
    with pytest.raises(ValueError, match="exact locked keys"):
        fresh.restore_checkpoint_state(top_extra)
    assert fresh.checkpoint_state() == before

    record_extra = json.loads(json.dumps(state))
    record_extra["last_record"]["qblade_force_consumption_verified"] = True
    with pytest.raises(ValueError, match="last_record.*exact locked keys"):
        fresh.restore_checkpoint_state(record_extra)
    assert fresh.checkpoint_state() == before

    audit_extra = json.loads(json.dumps(state))
    audit_extra["last_record"]["qblade_pre_time_grid_audit"][
        "qblade_force_consumption_verified"
    ] = True
    with pytest.raises(ValueError, match="time-grid audit.*exact locked keys"):
        fresh.restore_checkpoint_state(audit_extra)
    assert fresh.checkpoint_state() == before

    missing = json.loads(json.dumps(state))
    missing["last_record"].pop("applied_action_count")
    with pytest.raises(ValueError, match="last_record.*exact locked keys"):
        fresh.restore_checkpoint_state(missing)
    assert fresh.checkpoint_state() == before


def test_finalize_requires_complete_schedule_and_prohibits_further_steps(
    synthetic_artifacts,
):
    coupler = make_coupler(synthetic_artifacts)
    qblade = RecordingQBlade()
    advance_fake(qblade, coupler, 0)
    with pytest.raises(RuntimeError, match="incomplete"):
        coupler.finalize()

    for step in range(1, coupler.plan.total_wrapper_updates):
        advance_fake(qblade, coupler, step)
    summary = coupler.finalize()
    expected_nonzero = coupler.plan.burst_sample_count - 2
    assert summary["ended_with_explicit_zero"] is True
    assert summary["wrapper_updates"] == coupler.plan.total_wrapper_updates
    assert summary["physical_samples_applied"] == coupler.plan.total_physical_samples
    assert summary["bootstrap_updates"] == 1
    assert summary["scheduled_nonzero_physical_samples"] == expected_nonzero
    assert summary["exact_zero_physical_samples"] == (
        coupler.plan.total_physical_samples - expected_nonzero
    )
    assert summary["software_force_queue_depth"] == 0
    assert summary["generation_and_application_same_wrapper_call"] is True
    assert coupler.lifecycle == "finalized"
    with pytest.raises(RuntimeError, match="finalized"):
        coupler.step(qblade, coupler.plan.total_wrapper_updates)


def test_expected_artifact_hashes_and_plan_file_are_bound_to_identity(
    synthetic_artifacts,
    tmp_path: Path,
):
    paths = {
        "external_config": Path(synthetic_artifacts["external"]),
        "action_shapes": Path(synthetic_artifacts["shapes"]),
        "force_duals": Path(synthetic_artifacts["duals"]),
    }
    expected = {role: file_sha256(path) for role, path in paths.items()}
    direct = SilModalBurstCoupler.from_artifacts(
        plan=make_plan(),
        external_config_path=paths["external_config"],
        action_shapes_path=paths["action_shapes"],
        force_duals_path=paths["force_duals"],
        expected_sha256=expected,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        SilModalBurstCoupler.from_artifacts(
            plan=make_plan(),
            external_config_path=paths["external_config"],
            action_shapes_path=paths["action_shapes"],
            force_duals_path=paths["force_duals"],
            expected_sha256={**expected, "force_duals": "0" * 64},
        )

    plan_payload = {
        **make_plan().as_dict(),
        "artifacts": {
            role: {"path": path.name, "sha256": expected[role]}
            for role, path in paths.items()
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    from_plan = SilModalBurstCoupler.from_plan_file(plan_path)

    assert from_plan.generalized_force_audit == direct.generalized_force_audit
    assert (
        from_plan.checkpoint_identity()["fingerprint"]
        != direct.checkpoint_identity()["fingerprint"]
    )
    assert {row["role"] for row in from_plan.checkpoint_identity()["artifacts"]} == {
        "plan",
        "external_config",
        "action_shapes",
        "force_duals",
    }


def test_plan_and_artifact_symlinks_are_rejected_before_resolution(
    synthetic_artifacts,
    tmp_path: Path,
):
    external_path = Path(synthetic_artifacts["external"])
    symlink = tmp_path / "external-link.json"
    try:
        symlink.symlink_to(external_path)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="must not be a symlink"):
        SilModalBurstCoupler.from_artifacts(
            plan=make_plan(),
            external_config_path=symlink,
            action_shapes_path=Path(synthetic_artifacts["shapes"]),
            force_duals_path=Path(synthetic_artifacts["duals"]),
        )

    raw_plan = tmp_path / "raw-plan.json"
    raw_plan.write_text("{}\n", encoding="utf-8")
    linked_plan = tmp_path / "plan-link.json"
    linked_plan.symlink_to(raw_plan)
    with pytest.raises(ValueError, match="must not be a symlink"):
        SilModalBurstCoupler.from_plan_file(linked_plan)


def test_external_layout_must_be_global_complete_and_exactly_117(synthetic_artifacts):
    external_path = Path(synthetic_artifacts["external"])
    raw = json.loads(external_path.read_text(encoding="utf-8"))
    raw["external_output_rows"][7]["local"] = True
    external_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicitly global"):
        make_coupler(synthetic_artifacts)


def test_external_loader_rejects_duplicate_component_position_targets(
    synthetic_artifacts,
):
    external_path = Path(synthetic_artifacts["external"])
    raw = json.loads(external_path.read_text(encoding="utf-8"))
    first_target = raw["external_output_rows"][0]
    for axis in range(3):
        raw["external_output_rows"][3 + axis]["id"] = first_target["id"]
        raw["external_output_rows"][3 + axis]["position"] = first_target["position"]
    external_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"39 unique \(component_id, position\)"):
        make_coupler(synthetic_artifacts)


def test_artifact_frequency_must_match_the_locked_plan(synthetic_artifacts):
    with pytest.raises(ValueError, match="frequency does not match"):
        make_coupler(synthetic_artifacts, frequency_hz=0.999)


@pytest.mark.parametrize(
    ("rpm", "target_mode", "frequency_hz", "condition_number"),
    (
        (0.0, 6, 2.511557314896635, 21.6651),
        (0.0, 12, 4.572512385909129, 21.6651),
        (1.0, 6, 2.5115545017693828, 2.43123),
        (1.0, 12, 4.572593676110874, 2.43123),
    ),
)
def test_real_hvawt_modal_artifacts_pass_full_constructor_audit(
    rpm: float,
    target_mode: int,
    frequency_hz: float,
    condition_number: float,
):
    root = Path(__file__).resolve().parents[1]
    external = (
        root
        / "examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/Control"
        / "vorlap_10m_HVAWT_reference_external.json"
    )
    shapes = root / "run_headless/config/hvawt_response_mode_shapes.csv"
    duals = root / "run_headless/config/hvawt_response_mode_force_duals.csv"
    assert all(path.is_file() for path in (external, shapes, duals)), (
        "The reviewed H-VAWT modal evidence must be tracked in this checkout."
    )

    timestep_s = 0.001
    start_sample = 10
    burst_sample_count = int(round(5 / (frequency_hz * timestep_s))) + 1
    burst_end_sample = start_sample + burst_sample_count - 1
    plan = ModalBurstPlan(
        timestep_s=timestep_s,
        total_wrapper_updates=burst_end_sample + 3,
        bootstrap_zero=True,
        rotor_speed_rpm=rpm,
        target_mode=target_mode,
        frequency_hz=frequency_hz,
        generalized_force_amplitude_n=1.0,
        burst_start_time_s=start_sample * timestep_s,
        burst_cycles=5,
        azimuth_reference_deg=0.0,
        rotation_sign=1,
    )
    coupler = SilModalBurstCoupler.from_artifacts(
        plan=plan,
        external_config_path=external,
        action_shapes_path=shapes,
        force_duals_path=duals,
    )

    assert len(coupler.actions) == EXPECTED_ACTION_COUNT
    assert all(0.0 <= action.position <= 1.0 for action in coupler.actions)
    assert coupler.generalized_force_audit["passed"] is True
    assert coupler.generalized_force_audit["target_error"] < 1.0e-12
    assert coupler.generalized_force_audit["maximum_off_target"] < 1.0e-12
    assert coupler.generalized_force_audit["condition_number"] == pytest.approx(
        condition_number,
        rel=1.0e-5,
    )
