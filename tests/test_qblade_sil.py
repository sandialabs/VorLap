import csv
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vorlap.qblade_sil import (
    SIL_COMPACT_FORCE_HISTORY_SCHEMA,
    SIL_FORCE_HISTORY_SCHEMA,
    SIL_FORCE_PACKET_SCHEMA,
    SIL_FORCE_VECTOR_DIGEST_SCHEMA,
    SilConstantForceCoupler,
    SilForcePacket,
    SilTimingDiagnostics,
    SilVorLapCoupler,
    force_sign_probe_observations,
    parse_constant_force_action,
    qblade_result_time_diagnostics,
    resolve_sil_history_mode,
)


class RecordingQBlade:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str, float, float, str, bool, int]] = []
        self.current_time_s = 0.0

    def getCustomData_at_num(self, variable: bytes, _position: float, _turbine: int) -> float:
        if variable == b"Time [s]":
            return self.current_time_s
        return 0.0

    def setExternalAction(
        self,
        action: bytes,
        target: bytes,
        value: float,
        position: float,
        direction: bytes,
        local: bool,
        index: int,
    ) -> None:
        self.actions.append(
            (
                action.decode("utf-8"),
                target.decode("utf-8"),
                float(value),
                float(position),
                direction.decode("utf-8"),
                bool(local),
                int(index),
            )
        )


class DeterministicRuntime:
    def __init__(self, *, force_offset: int = 6, force_size: int = 3) -> None:
        self.message = "deterministic runtime"
        self.sample_step = None
        self.coverage_samples = 0
        self.restored_state: dict[str, object] | None = None
        self.force_offset = int(force_offset)
        self.force_size = int(force_size)

    def update(self, swap):
        step = int(swap[1])
        vector = np.array([10.0 + step, -20.0 - step, 30.0 + 2.0 * step])
        swap[self.force_offset : self.force_offset + self.force_size] = np.tile(
            vector,
            self.force_size // 3,
        )
        return swap[self.force_offset : self.force_offset + self.force_size].reshape((-1, 3))

    def record_source_coverage_sample(self, _timestep: float) -> None:
        self.coverage_samples += 1

    def source_coverage_summary(self, _timestep: float):
        return []

    def checkpoint_state(self) -> dict[str, object]:
        return {"coverage_samples": self.coverage_samples}

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        self.restored_state = dict(state)
        self.coverage_samples = int(state["coverage_samples"])


@dataclass(frozen=True)
class DeterministicResponseDiagnostics:
    enabled: bool
    amplitude_estimate_m: float
    target_cycle_mean_power_w: float
    projection_weighting: str


class DeterministicResponseRuntime(DeterministicRuntime):
    response_model_type = "modal_aerodynamic_power"

    def __init__(self) -> None:
        super().__init__()
        self.last_response_diagnostics = None

    def update(self, swap):
        result = super().update(swap)
        step = int(swap[1])
        self.last_response_diagnostics = DeterministicResponseDiagnostics(
            enabled=True,
            amplitude_estimate_m=0.25 + 0.01 * step,
            target_cycle_mean_power_w=100.0 + step,
            projection_weighting="configured",
        )
        return result


@dataclass(frozen=True)
class DeterministicSeededResponseDiagnostics(DeterministicResponseDiagnostics):
    startup_seed_enabled: bool
    startup_seed_active: bool
    startup_seed_taper: float
    startup_seed_nodal_weighting: str


class DeterministicSeededResponseRuntime(DeterministicRuntime):
    response_model_type = "modal_negative_damping"

    def __init__(self, *, force_offset: int = 6, force_size: int = 3) -> None:
        super().__init__(force_offset=force_offset, force_size=force_size)
        self.last_response_diagnostics = None

    def update(self, swap):
        result = super().update(swap)
        step = int(swap[1])
        self.last_response_diagnostics = DeterministicSeededResponseDiagnostics(
            enabled=True,
            amplitude_estimate_m=0.05 + 0.001 * step,
            target_cycle_mean_power_w=25.0,
            projection_weighting="configured",
            startup_seed_enabled=True,
            startup_seed_active=step < 2,
            startup_seed_taper=1.0 - 0.25 * step,
            startup_seed_nodal_weighting="modal_mass",
        )
        return result


def build_deterministic_sil_coupler(
    *,
    record_history: bool = True,
    history_mode: str | None = None,
    node_count: int = 1,
    force_application_lag_steps: int = 1,
    history_path: Path | None = None,
    resume_history: bool = False,
) -> SilVorLapCoupler:
    coupler = SilVorLapCoupler.__new__(SilVorLapCoupler)
    coupler.np = np
    coupler.force_application_lag_steps = int(force_application_lag_steps)
    coupler.timestep = 0.25
    velocity_size = 3 * node_count
    force_offset = 3 + velocity_size
    force_size = 3 * node_count
    coupler.swap = np.zeros(force_offset + force_size, dtype=float)
    coupler.input_rows = [
        {"index": 0, "variable": "Time [s]"},
        {"index": 1, "variable": "Timestep [-]"},
    ]
    coupler.output_rows = []
    coupler.node_ids = []
    for inode in range(node_count):
        position = (inode + 1) / (node_count + 1)
        node_id = f"BLD_1_{position:.6f}"
        coupler.node_ids.append(node_id)
        for axis_index, axis in enumerate(("X", "Y", "Z")):
            coupler.output_rows.append(
                {
                    "index": force_offset + 3 * inode + axis_index,
                    "action": "ADDFORCE",
                    "id": "BLD_1",
                    "position": position,
                    "direction": axis,
                    "local": False,
                }
            )
    coupler._timing = SilTimingDiagnostics()
    coupler._time_offset = 0
    coupler._timestep_offset = 1
    coupler._azimuth_offset = 2
    coupler._velocity_spec = SimpleNamespace(offset=3, size=velocity_size)
    coupler._force_spec = SimpleNamespace(offset=force_offset, size=force_size)
    coupler._force_application_queue = [
        coupler._startup_zero_force_packet()
        for _ in range(coupler.force_application_lag_steps)
    ]
    coupler.runtime = DeterministicRuntime(
        force_offset=force_offset,
        force_size=force_size,
    )
    coupler.updates = 0
    coupler.max_abs_force = 0.0
    coupler.last_message = coupler.runtime.message
    coupler.history_mode = resolve_sil_history_mode(
        record_history=record_history,
        history_mode=history_mode,
    )
    coupler.record_history = coupler.history_mode != "off"
    coupler.history_rows = []
    coupler._initialize_compact_history_state()
    if history_path is not None:
        coupler.configure_history_output(history_path, resume=resume_history)
    coupler.qblade_input_variables = []
    coupler.velocity_axis_by_variable = {}
    coupler.velocity_axis_stats = {
        axis: coupler._new_velocity_stat() for axis in ("X", "Y", "Z")
    }
    coupler.velocity_channel_stats = {}
    return coupler


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_parse_constant_force_action_and_constant_force_coupler_apply_global_force():
    action = parse_constant_force_action("BLD_1_0.230455:Y:-250:global")
    qblade = RecordingQBlade()
    coupler = SilConstantForceCoupler([action])

    message = coupler.step(qblade, 3)

    assert action == {
        "source_node_id": "BLD_1_0.230455",
        "action": "ADDFORCE",
        "id": "BLD_1",
        "position": 0.230455,
        "direction": "Y",
        "value": -250.0,
        "local": False,
    }
    assert qblade.actions == [("ADDFORCE", "BLD_1", -250.0, 0.230455, "Y", False, 0)]
    assert coupler.updates == 1
    assert coupler.max_abs_force == pytest.approx(250.0)
    assert message == "Applied 1 constant SIL force action(s) at step 3."


def test_qblade_result_time_diagnostics_parses_normalized_time_column(tmp_path: Path):
    result_file = tmp_path / "qblade_results.txt"
    result_file.write_text(
        "QBlade synthetic export\n"
        "Time~[s]\tX_g~Pos.~BLD_1~pos~0.000~[m]\n"
        "0.00\t0.0\n"
        "0.25\t0.1\n"
        "0.50\t0.2\n",
        encoding="utf-8",
    )

    diagnostics = qblade_result_time_diagnostics(result_file)

    assert diagnostics["available"] is True
    assert diagnostics["column"] == "Time~[s]"
    assert diagnostics["duration_s"] == pytest.approx(0.5)
    assert diagnostics["min_step_s"] == pytest.approx(0.25)
    assert diagnostics["max_step_s"] == pytest.approx(0.25)


def test_force_sign_probe_observations_compare_expected_channel_with_normalized_header(tmp_path: Path):
    action = parse_constant_force_action("BLD_1_0.230455:Y:-250:global")
    result_file = tmp_path / "probe_results.txt"
    result_file.write_text(
        "QBlade synthetic export\n"
        "Time~[s]\tY_g~Pos.~BLD_1~pos~0.230~[m]\n"
        "0.0\t0.0\n"
        "1.0\t-0.002\n",
        encoding="utf-8",
    )

    observations = force_sign_probe_observations(result_file, [action])

    assert observations[0]["observed_position_channel"] == "Y_g~Pos.~BLD_1~pos~0.230~[m]"
    assert observations[0]["delta_position_m"] == pytest.approx(-0.002)
    assert observations[0]["status"] == "same_sign"


def test_sil_timing_diagnostics_records_pre_advance_samples_and_restores_state():
    diagnostics = SilTimingDiagnostics(sample_limit=1)
    diagnostics.record(
        step_index=0,
        wrapper_time_s=0.0,
        wrapper_timestep=0.0,
        qblade_time_pre_advance_s=0.0,
        azimuth_deg=10.0,
        applied_force_time_s=0.0,
        applied_force_lag_steps=1,
    )
    diagnostics.record(
        step_index=1,
        wrapper_time_s=0.25,
        wrapper_timestep=1.0,
        qblade_time_pre_advance_s=0.50,
        azimuth_deg=20.0,
        applied_force_time_s=0.25,
    )

    state = diagnostics.as_dict()
    restored = SilTimingDiagnostics()
    restored.restore(state)

    assert state["time_level"] == "pre_advance_explicit"
    assert state["samples"] == 2
    assert state["first_samples"][0]["force_application_order"] == "before_advance"
    assert state["first_samples"][0]["applied_force_lag_steps"] == 1
    assert state["last_sample"]["step_index"] == 1
    assert restored.as_dict() == state


def test_sil_vorlap_force_application_queue_delays_current_forces_by_one_step():
    coupler = build_deterministic_sil_coupler()

    first_generated, first_applied = coupler._force_packets_for_step(
        np.array([1.0, 2.0, 3.0]),
        generation_step=0,
        generation_time_s=0.0,
    )
    second_generated, second_applied = coupler._force_packets_for_step(
        np.array([4.0, 5.0, 6.0]),
        generation_step=1,
        generation_time_s=0.25,
    )

    assert first_applied == SilForcePacket(None, None, (0.0, 0.0, 0.0), True)
    assert first_generated == second_applied
    assert second_generated.values == (4.0, 5.0, 6.0)
    assert coupler._force_application_queue == [second_generated]


def test_sil_step_records_generated_and_actual_applied_forces_with_startup_zero():
    coupler = build_deterministic_sil_coupler()
    qblade = RecordingQBlade()
    coupler.swap[3:6] = [0.5, -1.0, 2.0]

    coupler.step(qblade, 0)
    qblade.current_time_s = 0.25
    coupler.step(qblade, 1)

    applied_values = [action[2] for action in qblade.actions]
    assert applied_values == [0.0, 0.0, 0.0, 10.0, -20.0, 30.0]
    first, second = coupler.history_rows
    assert first["force_history_schema"] == SIL_FORCE_HISTORY_SCHEMA
    assert first["applied_startup_zero"] is True
    assert first["applied_source_step_index"] is None
    assert first["applied_source_time_s"] is None
    assert first["applied_force_BLD_1_0_500000_x_n"] == 0.0
    assert second["applied_startup_zero"] is False
    assert second["applied_source_step_index"] == 0
    assert second["applied_source_time_s"] == pytest.approx(0.0)
    assert second["generated_force_BLD_1_0_500000_x_n"] == pytest.approx(11.0)
    assert second["force_BLD_1_0_500000_x_n"] == pytest.approx(11.0)
    assert second["legacy_force_columns_semantics"] == "generated_force_alias"
    assert second["applied_force_BLD_1_0_500000_x_n"] == pytest.approx(10.0)
    assert second["applied_force_dot_structural_velocity_w"] == pytest.approx(85.0)
    assert coupler.timing_diagnostics()["first_samples"][0]["applied_force_time_s"] is None
    assert coupler.timing_diagnostics()["first_samples"][0]["applied_force_startup_zero"] is True


def test_sil_history_persists_typed_response_diagnostics_per_generated_step():
    coupler = build_deterministic_sil_coupler()
    coupler.runtime = DeterministicResponseRuntime()
    qblade = RecordingQBlade()
    coupler.swap[3:6] = [0.5, -1.0, 2.0]

    coupler.step(qblade, 0)
    qblade.current_time_s = 0.25
    coupler.step(qblade, 1)

    first, second = coupler.history_rows
    assert first["response_model_type"] == "modal_aerodynamic_power"
    assert first["response_enabled"] is True
    assert first["response_amplitude_estimate_m"] == pytest.approx(0.25)
    assert first["response_target_cycle_mean_power_w"] == pytest.approx(100.0)
    assert first["response_projection_weighting"] == "configured"
    assert second["response_amplitude_estimate_m"] == pytest.approx(0.26)
    assert second["response_target_cycle_mean_power_w"] == pytest.approx(101.0)
    coupler._validate_history_rows(coupler.history_rows)


def test_compact_history_retains_seed_diagnostics_and_exact_lag_digests(tmp_path: Path):
    coupler = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=tmp_path / "compact.csv",
    )
    coupler.runtime = DeterministicSeededResponseRuntime()
    qblade = RecordingQBlade()
    for step in range(3):
        coupler.swap[3:6] = [0.5, -1.0, 2.0]
        qblade.current_time_s = step * coupler.timestep
        coupler.step(qblade, step)

    assert coupler.history_rows == []
    metadata = coupler.write_history(tmp_path / "compact.csv")
    columns, rows = read_csv_rows(tmp_path / "compact.csv")
    first, second, third = rows
    digest_prefix = b"vorlap-force-vector-sha256-v1\0" + struct.pack("!Q", 3)
    expected_zero_digest = hashlib.sha256(
        digest_prefix + struct.pack("!3d", 0.0, 0.0, 0.0)
    ).hexdigest()
    expected_first_generated_digest = hashlib.sha256(
        digest_prefix + struct.pack("!3d", 10.0, -20.0, 30.0)
    ).hexdigest()
    assert first["force_history_schema"] == SIL_COMPACT_FORCE_HISTORY_SCHEMA
    assert first["force_vector_digest_schema"] == SIL_FORCE_VECTOR_DIGEST_SCHEMA
    assert first["applied_startup_zero"] == "True"
    assert first["applied_force_vector_sha256"] == expected_zero_digest
    assert first["generated_force_vector_sha256"] == expected_first_generated_digest
    assert second["applied_force_vector_sha256"] == first["generated_force_vector_sha256"]
    assert third["applied_force_vector_sha256"] == second["generated_force_vector_sha256"]
    assert second["applied_source_step_index"] == "0"
    assert float(second["applied_source_time_s"]) == pytest.approx(0.0)
    assert float(second["applied_force_dot_structural_velocity_w"]) == pytest.approx(85.0)
    assert first["response_startup_seed_enabled"] == "True"
    assert first["response_startup_seed_active"] == "True"
    assert float(first["response_startup_seed_taper"]) == pytest.approx(1.0)
    assert first["response_startup_seed_nodal_weighting"] == "modal_mass"
    assert not any(key.startswith("vel_BLD") for key in first)
    assert not any(key.startswith("generated_force_BLD") for key in first)
    assert not any(key.startswith("applied_force_BLD") for key in first)

    assert metadata is not None
    assert metadata["mode"] == "compact"
    assert metadata["schema"] == SIL_COMPACT_FORCE_HISTORY_SCHEMA
    assert metadata["rows"] == 3
    assert metadata["columns"] == len(columns) == len(first)
    assert metadata["streaming"] is True
    assert metadata["retained_rows"] == 0


def test_compact_history_is_fixed_width_and_materially_smaller_than_full(tmp_path: Path):
    full = build_deterministic_sil_coupler(node_count=21)
    compact = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        node_count=21,
        history_path=tmp_path / "compact_width.csv",
    )
    for coupler in (full, compact):
        qblade = RecordingQBlade()
        velocity_stop = coupler._velocity_spec.offset + coupler._velocity_spec.size
        coupler.swap[coupler._velocity_spec.offset : velocity_stop] = np.linspace(
            -1.0,
            1.0,
            coupler._velocity_spec.size,
        )
        coupler.step(qblade, 0)

    full_columns = len(full.history_rows[0])
    compact.write_history(tmp_path / "compact_width.csv")
    compact_columns = len(read_csv_rows(tmp_path / "compact_width.csv")[0])
    assert full_columns == 32 + 12 * 21
    assert compact_columns < full_columns / 3
    assert compact_columns < 50


def test_compact_history_rejects_post_lag_digest_tampering(tmp_path: Path):
    coupler = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=tmp_path / "tampered_compact.csv",
    )
    qblade = RecordingQBlade()
    for step in range(2):
        coupler.swap[3:6] = [0.5, -1.0, 2.0]
        qblade.current_time_s = step * coupler.timestep
        coupler.step(qblade, step)
    coupler.close_history(durable=True)
    columns, rows = read_csv_rows(tmp_path / "tampered_compact.csv")
    rows[1]["applied_force_vector_sha256"] = "0" * 64
    with (tmp_path / "tampered_compact.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="applied force digest does not equal"):
        coupler.write_history(tmp_path / "tampered_compact.csv")

    assert (tmp_path / "tampered_compact.csv").exists()


@pytest.mark.parametrize("lag_steps", [0, 2])
def test_compact_digest_lineage_handles_configured_lag_edges(
    lag_steps: int,
    tmp_path: Path,
):
    history_path = tmp_path / f"lag_{lag_steps}.csv"
    coupler = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        force_application_lag_steps=lag_steps,
        history_path=history_path,
    )
    qblade = RecordingQBlade()
    for step in range(4):
        coupler.swap[3:6] = [0.5, -1.0, 2.0]
        qblade.current_time_s = step * coupler.timestep
        coupler.step(qblade, step)

    coupler.write_history(history_path)
    _columns, rows = read_csv_rows(history_path)
    assert coupler.history_rows == []
    for step, row in enumerate(rows):
        source_step = step - lag_steps
        if source_step < 0:
            assert row["applied_startup_zero"] == "True"
            assert row["applied_source_step_index"] == ""
        else:
            assert row["applied_startup_zero"] == "False"
            assert int(row["applied_source_step_index"]) == source_step
            assert row["applied_force_vector_sha256"] == rows[source_step][
                "generated_force_vector_sha256"
            ]


def test_legacy_flag_and_explicit_full_mode_write_bit_identical_csv(tmp_path: Path):
    legacy = build_deterministic_sil_coupler(record_history=True)
    explicit = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="full",
    )
    for coupler in (legacy, explicit):
        qblade = RecordingQBlade()
        for step in range(3):
            coupler.swap[3:6] = [0.5, -1.0, 2.0]
            qblade.current_time_s = step * coupler.timestep
            coupler.step(qblade, step)

    legacy_path = tmp_path / "legacy.csv"
    explicit_path = tmp_path / "explicit.csv"
    legacy_metadata = legacy.write_history(legacy_path)
    explicit_metadata = explicit.write_history(explicit_path)

    assert legacy.history_rows == explicit.history_rows
    assert legacy_path.read_bytes() == explicit_path.read_bytes()
    assert legacy_metadata is not None and explicit_metadata is not None
    assert legacy_metadata["mode"] == explicit_metadata["mode"] == "full"
    assert legacy_metadata["schema"] == explicit_metadata["schema"] == SIL_FORCE_HISTORY_SCHEMA


def test_sil_checkpoint_restart_preserves_exact_next_applied_force_and_history():
    uninterrupted = build_deterministic_sil_coupler()
    uninterrupted_qblade = RecordingQBlade()
    for step in range(3):
        uninterrupted.swap[3:6] = [0.5, -1.0, 2.0]
        uninterrupted_qblade.current_time_s = step * uninterrupted.timestep
        uninterrupted.step(uninterrupted_qblade, step)

    pre_restart = build_deterministic_sil_coupler()
    pre_restart_qblade = RecordingQBlade()
    for step in range(2):
        pre_restart.swap[3:6] = [0.5, -1.0, 2.0]
        pre_restart_qblade.current_time_s = step * pre_restart.timestep
        pre_restart.step(pre_restart_qblade, step)
    checkpoint = json.loads(json.dumps(pre_restart.checkpoint_state()))
    assert checkpoint["force_packet_schema"] == SIL_FORCE_PACKET_SCHEMA

    resumed = build_deterministic_sil_coupler()
    resumed.restore_checkpoint_state(checkpoint)
    resumed_qblade = RecordingQBlade()
    resumed.swap[3:6] = [0.5, -1.0, 2.0]
    resumed_qblade.current_time_s = 0.5
    resumed.step(resumed_qblade, 2)

    assert resumed_qblade.actions == uninterrupted_qblade.actions[-3:]
    assert resumed.history_rows == uninterrupted.history_rows
    assert resumed.checkpoint_state()["force_application_queue"] == uninterrupted.checkpoint_state()[
        "force_application_queue"
    ]


def test_compact_history_checkpoint_restart_preserves_exact_history_and_queue(tmp_path: Path):
    uninterrupted_path = tmp_path / "uninterrupted.csv"
    restarted_path = tmp_path / "restarted.csv"

    def compact_coupler(path: Path, *, resume: bool = False) -> SilVorLapCoupler:
        return build_deterministic_sil_coupler(
            record_history=False,
            history_mode="compact",
            history_path=path,
            resume_history=resume,
        )

    uninterrupted = compact_coupler(uninterrupted_path)
    uninterrupted_qblade = RecordingQBlade()
    for step in range(4):
        uninterrupted.swap[3:6] = [0.5, -1.0, 2.0]
        uninterrupted_qblade.current_time_s = step * uninterrupted.timestep
        uninterrupted.step(uninterrupted_qblade, step)

    pre_restart = compact_coupler(restarted_path)
    pre_restart_qblade = RecordingQBlade()
    for step in range(2):
        pre_restart.swap[3:6] = [0.5, -1.0, 2.0]
        pre_restart_qblade.current_time_s = step * pre_restart.timestep
        pre_restart.step(pre_restart_qblade, step)
    checkpoint = json.loads(json.dumps(pre_restart.checkpoint_state()))
    assert checkpoint["history_mode"] == "compact"
    assert checkpoint["history_rows"] == []
    assert checkpoint["compact_history"]["rows"] == 2

    resumed = compact_coupler(restarted_path, resume=True)
    resumed.restore_checkpoint_state(checkpoint)
    resumed_qblade = RecordingQBlade()
    for step in range(2, 4):
        resumed.swap[3:6] = [0.5, -1.0, 2.0]
        resumed_qblade.current_time_s = step * resumed.timestep
        resumed.step(resumed_qblade, step)

    assert resumed_qblade.actions == uninterrupted_qblade.actions[-6:]
    uninterrupted.write_history(uninterrupted_path)
    resumed_metadata = resumed.write_history(restarted_path)
    assert resumed.history_rows == uninterrupted.history_rows == []
    assert restarted_path.read_bytes() == uninterrupted_path.read_bytes()
    assert resumed_metadata is not None
    assert resumed_metadata["discarded_uncheckpointed_tail_bytes"] == 0
    assert resumed.checkpoint_state()["force_application_queue"] == uninterrupted.checkpoint_state()[
        "force_application_queue"
    ]


def test_compact_restart_verifies_prefix_and_discards_uncheckpointed_tail(tmp_path: Path):
    uninterrupted_path = tmp_path / "tail_uninterrupted.csv"
    restarted_path = tmp_path / "tail_restarted.csv"
    uninterrupted = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=uninterrupted_path,
    )
    for step in range(4):
        uninterrupted.swap[3:6] = [0.5, -1.0, 2.0]
        uninterrupted.step(RecordingQBlade(), step)

    source = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=restarted_path,
    )
    for step in range(2):
        source.swap[3:6] = [0.5, -1.0, 2.0]
        source.step(RecordingQBlade(), step)
    checkpoint = json.loads(json.dumps(source.checkpoint_state()))
    checkpoint_size = checkpoint["compact_history"]["size_bytes"]
    source.swap[3:6] = [0.5, -1.0, 2.0]
    source.step(RecordingQBlade(), 2)
    source.close_history(durable=True)
    uncheckpointed_tail_bytes = restarted_path.stat().st_size - checkpoint_size
    assert uncheckpointed_tail_bytes > 0

    resumed = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=restarted_path,
        resume_history=True,
    )
    resumed.restore_checkpoint_state(checkpoint)
    assert restarted_path.stat().st_size == checkpoint_size
    for step in range(2, 4):
        resumed.swap[3:6] = [0.5, -1.0, 2.0]
        resumed.step(RecordingQBlade(), step)

    uninterrupted.write_history(uninterrupted_path)
    metadata = resumed.write_history(restarted_path)
    assert restarted_path.read_bytes() == uninterrupted_path.read_bytes()
    assert metadata is not None
    assert metadata["discarded_uncheckpointed_tail_bytes"] == uncheckpointed_tail_bytes


def test_compact_restart_from_zero_row_checkpoint_appends_to_verified_empty_file(tmp_path: Path):
    uninterrupted_path = tmp_path / "zero_uninterrupted.csv"
    restarted_path = tmp_path / "zero_restarted.csv"
    source = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=restarted_path,
    )
    checkpoint = json.loads(json.dumps(source.checkpoint_state()))
    assert checkpoint["compact_history"]["rows"] == 0
    assert checkpoint["compact_history"]["size_bytes"] == 0
    restarted_path.write_bytes(b"")

    resumed = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=restarted_path,
        resume_history=True,
    )
    resumed.restore_checkpoint_state(checkpoint)
    resumed.swap[3:6] = [0.5, -1.0, 2.0]
    resumed.step(RecordingQBlade(), 0)
    resumed.write_history(restarted_path)

    uninterrupted = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=uninterrupted_path,
    )
    uninterrupted.swap[3:6] = [0.5, -1.0, 2.0]
    uninterrupted.step(RecordingQBlade(), 0)
    uninterrupted.write_history(uninterrupted_path)

    assert restarted_path.read_bytes() == uninterrupted_path.read_bytes()


def test_compact_restart_rejects_deleted_nonempty_prefix_before_append(tmp_path: Path):
    history_path = tmp_path / "deleted_prefix.csv"
    source = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=history_path,
    )
    source.swap[3:6] = [0.5, -1.0, 2.0]
    source.step(RecordingQBlade(), 0)
    checkpoint = json.loads(json.dumps(source.checkpoint_state()))
    source.close_history(durable=True)

    resumed = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=history_path,
        resume_history=True,
    )
    resumed.restore_checkpoint_state(checkpoint)
    history_path.unlink()
    resumed.swap[3:6] = [0.5, -1.0, 2.0]

    with pytest.raises(RuntimeError, match="disappeared after restart validation"):
        resumed.step(RecordingQBlade(), 1)

    assert not history_path.exists()


def test_compact_restart_rejects_tampered_checkpoint_prefix(tmp_path: Path):
    history_path = tmp_path / "tampered_prefix.csv"
    source = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=history_path,
    )
    for step in range(2):
        source.swap[3:6] = [0.5, -1.0, 2.0]
        source.step(RecordingQBlade(), step)
    checkpoint = json.loads(json.dumps(source.checkpoint_state()))
    source.close_history(durable=True)
    original_digest = checkpoint["compact_history"]["generated_digest_window"][0][
        "generated_force_vector_sha256"
    ]
    data = history_path.read_bytes()
    assert original_digest.encode("ascii") in data
    history_path.write_bytes(data.replace(original_digest.encode("ascii"), b"0" * 64, 1))

    resumed = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=history_path,
        resume_history=True,
    )
    with pytest.raises(ValueError, match="Corrupt compact SIL force history|SHA-256"):
        resumed.restore_checkpoint_state(checkpoint)

    assert resumed.updates == 0


def test_compact_restart_runtime_rejection_preserves_local_state_and_file(tmp_path: Path):
    history_path = tmp_path / "runtime_rejection.csv"
    source = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=history_path,
    )
    for step in range(2):
        source.swap[3:6] = [0.5, -1.0, 2.0]
        source.step(RecordingQBlade(), step)
    checkpoint = json.loads(json.dumps(source.checkpoint_state()))
    source.swap[3:6] = [0.5, -1.0, 2.0]
    source.step(RecordingQBlade(), 2)
    source.close_history(durable=True)
    bytes_before_restore = history_path.read_bytes()

    resumed = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=history_path,
        resume_history=True,
    )

    def reject_runtime_restore(_state: dict[str, object]) -> None:
        raise RuntimeError("injected runtime restore rejection")

    resumed.runtime.restore_checkpoint_state = reject_runtime_restore
    with pytest.raises(RuntimeError, match="injected runtime restore rejection"):
        resumed.restore_checkpoint_state(checkpoint)

    assert resumed.updates == 0
    assert resumed.max_abs_force == 0.0
    assert resumed.timing_diagnostics()["samples"] == 0
    assert resumed._force_application_queue[0].startup_zero is True
    assert resumed._compact_history_rows_written == 0
    assert resumed._compact_history_handle is None
    assert history_path.read_bytes() == bytes_before_restore


def test_checkpoint_adapter_rejection_restores_runtime_and_adapter_atomically():
    class Adapter:
        def __init__(self, value: int, *, reject: int | None = None) -> None:
            self.value = value
            self.reject = reject

        def checkpoint_state(self) -> dict[str, object]:
            return {"value": self.value}

        def restore_checkpoint_state(self, state: dict[str, object]) -> None:
            value = int(state["value"])
            if value == self.reject:
                raise RuntimeError("injected adapter restore rejection")
            self.value = value

    source = build_deterministic_sil_coupler(record_history=False)
    source.runtime.coverage_samples = 5
    source.hvawt_response_adapter = Adapter(5)
    checkpoint = json.loads(json.dumps(source.checkpoint_state()))

    restored = build_deterministic_sil_coupler(record_history=False)
    restored.hvawt_response_adapter = Adapter(0, reject=5)
    runtime_before = restored.runtime.checkpoint_state()
    adapter_before = restored.hvawt_response_adapter.checkpoint_state()
    with pytest.raises(RuntimeError, match="injected adapter restore rejection"):
        restored.restore_checkpoint_state(checkpoint)

    assert restored.runtime.checkpoint_state() == runtime_before
    assert restored.hvawt_response_adapter.checkpoint_state() == adapter_before
    assert restored.updates == 0


def test_checkpoint_timing_rejection_occurs_before_runtime_restore():
    source = build_deterministic_sil_coupler(record_history=False)
    source.runtime.coverage_samples = 5
    checkpoint = json.loads(json.dumps(source.checkpoint_state()))
    checkpoint["timing_diagnostics"]["samples"] = "bad"

    restored = build_deterministic_sil_coupler(record_history=False)
    runtime_before = restored.runtime.checkpoint_state()
    with pytest.raises(ValueError, match="timing samples"):
        restored.restore_checkpoint_state(checkpoint)

    assert restored.runtime.checkpoint_state() == runtime_before
    assert restored.updates == 0


def test_step_rejects_nodal_wind_without_opt_in_before_state_update(monkeypatch):
    coupler = build_deterministic_sil_coupler(record_history=False)
    coupler.use_qblade_local_inflow = False
    qblade = RecordingQBlade()
    swap_before = coupler.swap.copy()

    def unexpected_input_read(*_args):
        pytest.fail("Rejected nodal wind must not read or update QBlade inputs.")

    monkeypatch.setattr(coupler, "_fill_inputs", unexpected_input_read)
    with pytest.raises(ValueError, match="require use_qblade_local_inflow=true"):
        coupler.step(qblade, 0, ambient_velocity_global_mps=np.array([[8.0, 1.0, 0.0]]))

    np.testing.assert_array_equal(coupler.swap, swap_before)
    assert coupler.updates == 0
    assert coupler.runtime.coverage_samples == 0
    assert qblade.actions == []


def test_checkpoint_rejects_changed_local_inflow_mode_before_restore():
    source = build_deterministic_sil_coupler(record_history=False)
    source.runtime.coverage_samples = 5
    state = source.checkpoint_state()
    target = build_deterministic_sil_coupler(record_history=False)
    target.use_qblade_local_inflow = True
    with pytest.raises(ValueError, match="local-inflow mode differs"):
        target.restore_checkpoint_state(state)
    assert target.runtime.restored_state is None
    assert target.updates == 0


def test_legacy_unversioned_checkpoint_is_explicitly_incompatible():
    coupler = build_deterministic_sil_coupler(record_history=False)
    checkpoint = coupler.checkpoint_state()
    checkpoint.pop("schema")

    with pytest.raises(ValueError, match="legacy.*schema/timestep"):
        coupler.restore_checkpoint_state(checkpoint)


def test_compact_history_retains_constant_memory_state_over_many_rows(tmp_path: Path):
    history_path = tmp_path / "many_rows.csv"
    coupler = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=history_path,
    )
    qblade = RecordingQBlade()
    small_state_size = None
    for step in range(5000):
        coupler.swap[3:6] = [0.5, -1.0, 2.0]
        coupler.step(qblade, step)
        qblade.actions.clear()
        if step == 99:
            small_state_size = len(json.dumps(coupler.checkpoint_state()))

    large_state = coupler.checkpoint_state()
    assert small_state_size is not None
    assert coupler.history_rows == []
    assert len(coupler._compact_generated_digest_window) == 1
    assert large_state["history_rows"] == []
    assert large_state["compact_history"]["rows"] == 5000
    assert len(json.dumps(large_state)) <= small_state_size + 128
    assert history_path.stat().st_size > 1_000_000
    coupler.write_history(history_path)


def test_compact_history_refuses_stale_output_on_direct_configuration(tmp_path: Path):
    history_path = tmp_path / "stale.csv"
    history_path.write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_deterministic_sil_coupler(
            record_history=False,
            history_mode="compact",
            history_path=history_path,
        )


def test_compact_history_checkpoint_rejects_queue_vector_not_bound_to_digest(tmp_path: Path):
    history_path = tmp_path / "queue_digest.csv"
    source = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=history_path,
    )
    qblade = RecordingQBlade()
    source.step(qblade, 0)
    state = json.loads(json.dumps(source.checkpoint_state()))
    state["force_application_queue"][0]["values"][0] = 999.0

    restored = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=history_path,
        resume_history=True,
    )
    with pytest.raises(ValueError, match="does not match its compact generated-history digest"):
        restored.restore_checkpoint_state(state)

    assert restored.updates == 0


def test_checkpoint_restart_rejects_full_compact_history_mode_mismatch(tmp_path: Path):
    source = build_deterministic_sil_coupler(
        record_history=False,
        history_mode="compact",
        history_path=tmp_path / "mode_mismatch.csv",
    )
    source.step(RecordingQBlade(), 0)
    state = json.loads(json.dumps(source.checkpoint_state()))

    restored = build_deterministic_sil_coupler(record_history=True)
    with pytest.raises(ValueError, match="checkpoint=compact, configured=full"):
        restored.restore_checkpoint_state(state)

    assert restored.updates == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "legacy_bare_vector",
        "nonfinite_vector",
        "wrong_generation_step",
        "wrong_generation_time",
        "wrong_queue_length",
    ],
)
def test_sil_checkpoint_restore_rejects_corrupt_force_queue(corruption: str):
    source = build_deterministic_sil_coupler(record_history=False)
    qblade = RecordingQBlade()
    source.step(qblade, 0)
    state = json.loads(json.dumps(source.checkpoint_state()))
    if corruption == "legacy_bare_vector":
        state["force_application_queue"][0] = [10.0, -20.0, 30.0]
    elif corruption == "nonfinite_vector":
        state["force_application_queue"][0]["values"][1] = float("nan")
    elif corruption == "wrong_generation_step":
        state["force_application_queue"][0]["generation_step"] = 4
    elif corruption == "wrong_generation_time":
        state["force_application_queue"][0]["generation_time_s"] = 0.25
    elif corruption == "wrong_queue_length":
        state["force_application_queue"] = []

    restored = build_deterministic_sil_coupler(record_history=False)
    with pytest.raises(ValueError, match="Corrupt|corrupt|legacy"):
        restored.restore_checkpoint_state(state)

    assert restored.updates == 0
    assert restored._force_application_queue[0].startup_zero is True


def test_sil_history_writer_rejects_tampered_applied_force(tmp_path: Path):
    coupler = build_deterministic_sil_coupler()
    qblade = RecordingQBlade()
    for step in range(2):
        coupler.swap[3:6] = [0.5, -1.0, 2.0]
        qblade.current_time_s = step * coupler.timestep
        coupler.step(qblade, step)
    coupler.history_rows[1]["applied_force_BLD_1_0_500000_x_n"] = 999.0

    with pytest.raises(ValueError, match="applied force does not equal its generated source"):
        coupler.write_history(tmp_path / "tampered.csv")

    assert not (tmp_path / "tampered.csv").exists()


def test_sil_qblade_value_uses_finite_documented_position_argument():
    calls = []

    class QBlade:
        def getCustomData_at_num(self, variable, position, turbine):
            calls.append((variable, position, turbine))
            return 12.5

    value = SilVorLapCoupler._qblade_value(QBlade(), "LSS Azimuthal Pos. [deg]")

    assert value == pytest.approx(12.5)
    assert calls == [(b"LSS Azimuthal Pos. [deg]", 0.0, 0)]


def test_sil_runtime_filter_must_retain_qblade_input_channels(tmp_path: Path):
    sim = tmp_path / "case.sim"
    runtime_filter = tmp_path / "fatigue.filter"
    sim.write_text("fatigue.filter FILTERFILE\n", encoding="utf-8")
    runtime_filter.write_text("Time [s]\n", encoding="utf-8")
    coupler = SilVorLapCoupler.__new__(SilVorLapCoupler)
    coupler.qblade_input_variables = [
        "LSS Azimuthal Pos. [deg]",
        "X_g Vel. BLD_1 pos 0.500 [m/s]",
    ]

    with pytest.raises(RuntimeError, match="FILTERFILE omits SIL-VorLap runtime input"):
        coupler._validate_runtime_input_storage(sim)


def test_sil_channel_validation_rejects_qblade_missing_data_sentinel():
    class QBlade:
        def getCustomData_at_num(self, _variable, _position, _turbine):
            return -1.0

    coupler = SilVorLapCoupler.__new__(SilVorLapCoupler)
    coupler.qblade_input_variables = [
        "LSS Azimuthal Pos. [deg]",
        "X_g Vel. BLD_1 pos 0.500 [m/s]",
    ]

    with pytest.raises(RuntimeError, match="missing-custom-data sentinel -1"):
        coupler.validate_channels(QBlade())
