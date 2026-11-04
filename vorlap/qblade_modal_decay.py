"""Scheduled complex modal-burst forcing for QBlade SIL decay probes.

This module deliberately contains no QBlade runner integration.  It owns the
small, auditable core needed by that integration: loading the fixed 39-site
global-force layout, checking the saved complex force dual against the full
12-mode action basis, generating a five-cycle Hann burst, and applying the
force generated for a timestep before that timestep is advanced.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


MODAL_BURST_PLAN_SCHEMA = "vorlap-qblade-modal-burst-plan-v1"
MODAL_BURST_IDENTITY_SCHEMA = "vorlap-qblade-modal-burst-identity-v1"
MODAL_BURST_CHECKPOINT_SCHEMA = "vorlap-qblade-modal-burst-checkpoint-v1"
MODAL_BURST_STEP_SCHEMA = "vorlap-qblade-modal-burst-step-v1"
MODAL_BURST_FORCE_DIGEST_SCHEMA = "vorlap-qblade-modal-burst-force-sha256-v1"
QBLADE_PRE_TIME_GRID_AUDIT_SCHEMA = "vorlap-qblade-pre-time-grid-audit-v1"

EXPECTED_ACTION_COUNT = 117
EXPECTED_SITE_COUNT = 39
EXPECTED_SWAP_SIZE = 237
EXPECTED_BASIS_MODES = tuple(range(1, 13))
EXPECTED_DIRECTIONS = ("X", "Y", "Z")
MINIMUM_SAMPLES_PER_CYCLE = 20
MAXIMUM_REALIZED_BURST_CYCLE_ERROR = 0.01
QBLADE_PRE_TIME_FLOAT32_ULPS = 4.0
QBLADE_PRE_TIME_ABSOLUTE_FLOOR_S = 1.0e-12
QBLADE_PRE_TIME_MAX_TIMESTEP_FRACTION = 0.25
TIME_CHANNEL = "Time [s]"
AZIMUTH_CHANNEL = "LSS Azimuthal Pos. [deg]"

_DESIGN_RECORD_KEYS = frozenset(
    {
        "schema",
        "record_kind",
        "physical_sample_index",
        "physical_time_s",
        "azimuth_deg",
        "bootstrap_zero",
        "burst_sample_index",
        "hann_weight",
        "phase_rad",
        "scheduled_zero",
        "designed_force_frame",
        "complex_phasor_convention",
        "force_component_count",
        "global_force_sha256",
        "maximum_absolute_force_component_n",
        "generalized_force_audit_convention",
        "target_complex_generalized_force_phasor_real_n",
        "target_complex_generalized_force_phasor_imag_n",
        "target_complex_generalized_force_phasor_error_n",
        "maximum_off_target_complex_generalized_force_phasor_n",
    }
)
_APPLIED_RECORD_EXTRA_KEYS = frozenset(
    {
        "wrapper_step_index",
        "generation_wrapper_step_index",
        "application_wrapper_step_index",
        "qblade_time_pre_advance_s",
        "qblade_pre_time_grid_audit",
        "software_force_queue_depth",
        "generation_and_application_same_wrapper_call",
        "action_application_order",
        "action_application_completed",
        "applied_action_count",
    }
)
_APPLIED_RECORD_KEYS = _DESIGN_RECORD_KEYS | _APPLIED_RECORD_EXTRA_KEYS
_CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "identity_fingerprint",
        "lifecycle",
        "wrapper_updates",
        "next_wrapper_step_index",
        "physical_samples_applied",
        "bootstrap_updates",
        "software_force_queue_depth",
        "pending_force_packets",
        "max_abs_force",
        "scheduled_nonzero_physical_samples",
        "exact_zero_physical_samples",
        "last_record",
    }
)
_QBLADE_PRE_TIME_GRID_AUDIT_KEYS = frozenset(
    {
        "schema",
        "schedule_kind",
        "expected_physical_sample_index",
        "expected_grid_index",
        "nearest_observed_grid_index",
        "nearest_index_required",
        "nearest_index_matches",
        "expected_time_s",
        "observed_time_s",
        "signed_residual_s",
        "absolute_residual_s",
        "float32_expected_time_s",
        "float32_ulp_s",
        "float32_ulp_multiplier",
        "absolute_floor_s",
        "float32_residual_limit_s",
        "quarter_timestep_limit_s",
        "passed",
    }
)


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number; got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite; got {value!r}.")
    return result


def _strict_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer; got {value!r}.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}; got {result}.")
    return result


def _require_exact_keys(
    value: Mapping[object, object],
    expected: frozenset[str],
    label: str,
) -> None:
    observed = set(value)
    if observed == set(expected):
        return
    missing = sorted(str(key) for key in set(expected) - observed)
    unexpected = sorted(str(key) for key in observed - set(expected))
    raise ValueError(
        f"{label} must contain the exact locked keys; "
        f"missing={missing}, unexpected={unexpected}."
    )


def _qblade_pre_time_policy() -> dict[str, object]:
    return {
        "schema": QBLADE_PRE_TIME_GRID_AUDIT_SCHEMA,
        "nearest_index_rule": "int(round(observed_time_s / timestep_s))",
        "float32_ulp_rule": "max(absolute_floor_s, float32_ulp_multiplier * ulp32)",
        "float32_ulp_multiplier": QBLADE_PRE_TIME_FLOAT32_ULPS,
        "absolute_floor_s": QBLADE_PRE_TIME_ABSOLUTE_FLOOR_S,
        "strict_max_timestep_fraction": QBLADE_PRE_TIME_MAX_TIMESTEP_FRACTION,
    }


def _validate_qblade_pre_time(
    observed_time_s: object,
    *,
    timestep_s: float,
    expected_physical_sample_index: int | None,
) -> dict[str, object]:
    """Validate QBlade's float32-like time against one locked schedule sample."""
    observed = _finite_float(observed_time_s, "QBlade pre-advance time")
    timestep = _finite_float(timestep_s, "QBlade pre-time timestep")
    if timestep <= 0.0:
        raise ValueError("QBlade pre-time timestep must be positive.")
    if expected_physical_sample_index is None:
        expected_index = 0
        schedule_kind = "bootstrap_zero"
        nearest_index_required = False
    else:
        expected_index = _strict_int(
            expected_physical_sample_index,
            "expected physical sample index",
            minimum=0,
        )
        schedule_kind = "physical_sample"
        nearest_index_required = True
    expected = expected_index * timestep
    if not math.isfinite(expected):
        raise ValueError("Expected QBlade pre-advance grid time must be finite.")
    nearest_index = int(round(observed / timestep))
    signed_residual = observed - expected
    absolute_residual = abs(signed_residual)
    with np.errstate(over="ignore", invalid="ignore"):
        expected_float32_value = np.float32(expected)
        float32_ulp = abs(float(np.spacing(expected_float32_value)))
    expected_float32 = float(expected_float32_value)
    if not math.isfinite(expected_float32) or not math.isfinite(float32_ulp):
        raise ValueError("Expected QBlade pre-advance time is outside float32 range.")
    float32_limit = max(
        QBLADE_PRE_TIME_ABSOLUTE_FLOOR_S,
        QBLADE_PRE_TIME_FLOAT32_ULPS * float32_ulp,
    )
    quarter_timestep_limit = QBLADE_PRE_TIME_MAX_TIMESTEP_FRACTION * timestep
    nearest_matches = nearest_index == expected_index
    residual_passes = (
        absolute_residual <= float32_limit
        and absolute_residual < quarter_timestep_limit
    )
    if (nearest_index_required and not nearest_matches) or not residual_passes:
        raise RuntimeError(
            "QBlade pre-advance time does not match the bootstrap/physical float32 grid: "
            f"schedule={schedule_kind}, expected_sample={expected_physical_sample_index}, "
            f"expected_grid_index={expected_index}, nearest_observed_index={nearest_index}, "
            f"expected={expected:.17g}, observed={observed:.17g}, "
            f"absolute_residual={absolute_residual:.9g}, "
            f"float32_limit={float32_limit:.9g}, "
            f"strict_quarter_dt_limit={quarter_timestep_limit:.9g}."
        )
    audit: dict[str, object] = {
        "schema": QBLADE_PRE_TIME_GRID_AUDIT_SCHEMA,
        "schedule_kind": schedule_kind,
        "expected_physical_sample_index": expected_physical_sample_index,
        "expected_grid_index": expected_index,
        "nearest_observed_grid_index": nearest_index,
        "nearest_index_required": nearest_index_required,
        "nearest_index_matches": nearest_matches,
        "expected_time_s": expected,
        "observed_time_s": observed,
        "signed_residual_s": signed_residual,
        "absolute_residual_s": absolute_residual,
        "float32_expected_time_s": expected_float32,
        "float32_ulp_s": float32_ulp,
        "float32_ulp_multiplier": QBLADE_PRE_TIME_FLOAT32_ULPS,
        "absolute_floor_s": QBLADE_PRE_TIME_ABSOLUTE_FLOOR_S,
        "float32_residual_limit_s": float32_limit,
        "quarter_timestep_limit_s": quarter_timestep_limit,
        "passed": True,
    }
    _require_exact_keys(audit, _QBLADE_PRE_TIME_GRID_AUDIT_KEYS, "QBlade time-grid audit")
    return audit


def _canonical_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_artifact_bytes(
    role: str,
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Read and identify exactly the immutable bytes that will be parsed."""
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"Modal-burst artifact must not be a symlink: {candidate}")
    path = candidate.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    with path.open("rb") as stream:
        data = stream.read()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
    ):
        raise RuntimeError(f"Modal-burst artifact changed while hashing: {path}")
    if len(data) != before.st_size:
        raise RuntimeError(f"Modal-burst artifact byte count changed while reading: {path}")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Expected SHA-256 for {role} is malformed: {expected_sha256!r}.")
        if digest != expected:
            raise ValueError(
                f"Modal-burst artifact SHA-256 mismatch for {role}: "
                f"expected {expected}, observed {digest}."
            )
    identity = {
        "role": str(role),
        "path": str(path),
        "name": path.name,
        "sha256": digest,
        "size_bytes": int(len(data)),
    }
    return data, identity


def _stable_artifact_identity(identity: Mapping[str, object]) -> dict[str, object]:
    return {
        "role": str(identity["role"]),
        "name": str(identity["name"]),
        "sha256": str(identity["sha256"]),
        "size_bytes": int(identity["size_bytes"]),
    }


def _validate_artifact_identities(
    identities: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    required = {"role", "path", "name", "sha256", "size_bytes"}
    normalized: list[dict[str, object]] = []
    roles: set[str] = set()
    for index, identity in enumerate(identities):
        if not isinstance(identity, Mapping) or set(identity) != required:
            raise ValueError(
                "Each modal-burst artifact identity must contain exactly role, path, name, "
                f"sha256, and size_bytes; invalid entry {index}."
            )
        role = identity["role"]
        path_text = identity["path"]
        name = identity["name"]
        digest = identity["sha256"]
        size = identity["size_bytes"]
        if not isinstance(role, str) or not role.strip() or role in roles:
            raise ValueError(f"Artifact identity role is empty or duplicated at entry {index}.")
        if not isinstance(path_text, str) or not Path(path_text).is_absolute():
            raise ValueError(f"Artifact identity path must be absolute at entry {index}.")
        if not isinstance(name, str) or name != Path(path_text).name:
            raise ValueError(f"Artifact identity name/path mismatch at entry {index}.")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Artifact identity SHA-256 is malformed at entry {index}.")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"Artifact identity size_bytes is invalid at entry {index}.")
        roles.add(role)
        normalized.append(
            {
                "role": role,
                "path": path_text,
                "name": name,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return tuple(normalized)


def _force_vector_sha256(values: Sequence[float]) -> str:
    numeric = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("Cannot digest a non-finite modal-burst force vector.")
    payload = MODAL_BURST_FORCE_DIGEST_SCHEMA.encode("ascii") + b"\0"
    payload += struct.pack("!Q", len(numeric))
    if numeric:
        payload += struct.pack(f"!{len(numeric)}d", *numeric)
    return hashlib.sha256(payload).hexdigest()


def _complex_array_sha256(values: np.ndarray, *, schema: str) -> str:
    array = np.asarray(values, dtype=complex)
    if not np.all(np.isfinite(array)):
        raise ValueError("Cannot digest a non-finite complex modal array.")
    payload = schema.encode("ascii") + b"\0"
    payload += struct.pack("!Q", array.ndim)
    payload += struct.pack(f"!{array.ndim}Q", *array.shape)
    flattened = array.reshape(-1, order="C")
    if flattened.size:
        numeric: list[float] = []
        for value in flattened:
            numeric.extend((float(value.real), float(value.imag)))
        payload += struct.pack(f"!{len(numeric)}d", *numeric)
    return hashlib.sha256(payload).hexdigest()


def _float_vector_sha256(values: Sequence[float], *, schema: str) -> str:
    numeric = tuple(_finite_float(value, "frequency digest value") for value in values)
    payload = schema.encode("ascii") + b"\0" + struct.pack("!Q", len(numeric))
    if numeric:
        payload += struct.pack(f"!{len(numeric)}d", *numeric)
    return hashlib.sha256(payload).hexdigest()


def _read_csv_rows(data: bytes, source: Path) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV artifact is not valid UTF-8: {source}") from exc
    with io.StringIO(text, newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV artifact has no header: {source}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV artifact has no data rows: {source}")
    return rows


def _require_columns(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    path: Path,
) -> None:
    available = set(rows[0]) if rows else set()
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")


def _rpm_matches(value: object, target_rpm: float) -> bool:
    observed = _finite_float(value, "artifact RPM")
    return math.isclose(observed, target_rpm, rel_tol=0.0, abs_tol=1.0e-9)


@dataclass(frozen=True)
class ModalBurstPlan:
    """Immutable physical and numerical definition of one modal burst."""

    timestep_s: float
    total_wrapper_updates: int
    bootstrap_zero: bool
    rotor_speed_rpm: float
    target_mode: int
    frequency_hz: float
    generalized_force_amplitude_n: float
    burst_start_time_s: float
    burst_cycles: int
    azimuth_reference_deg: float
    rotation_sign: int
    target_generalized_force_tolerance: float = 1.0e-5
    off_target_generalized_force_tolerance: float = 1.0e-4
    minimum_samples_per_cycle: int = MINIMUM_SAMPLES_PER_CYCLE
    maximum_realized_burst_cycle_error: float = MAXIMUM_REALIZED_BURST_CYCLE_ERROR

    def __post_init__(self) -> None:
        timestep = _finite_float(self.timestep_s, "timestep_s")
        if timestep <= 0.0:
            raise ValueError("timestep_s must be positive.")
        total_updates = _strict_int(
            self.total_wrapper_updates,
            "total_wrapper_updates",
            minimum=3,
        )
        if not isinstance(self.bootstrap_zero, bool) or self.bootstrap_zero is not True:
            raise ValueError("bootstrap_zero must be explicitly set to true.")
        target_mode = _strict_int(self.target_mode, "target_mode", minimum=1)
        if target_mode not in EXPECTED_BASIS_MODES:
            raise ValueError(
                f"target_mode must be in the saved 12-mode basis; got {target_mode}."
            )
        frequency = _finite_float(self.frequency_hz, "frequency_hz")
        if frequency <= 0.0:
            raise ValueError("frequency_hz must be positive.")
        amplitude = _finite_float(
            self.generalized_force_amplitude_n,
            "generalized_force_amplitude_n",
        )
        if amplitude < 0.0:
            raise ValueError("generalized_force_amplitude_n must be nonnegative.")
        burst_start = _finite_float(self.burst_start_time_s, "burst_start_time_s")
        if burst_start < 0.0:
            raise ValueError("burst_start_time_s must be nonnegative.")
        burst_cycles = _strict_int(self.burst_cycles, "burst_cycles", minimum=1)
        rotor_speed = _finite_float(self.rotor_speed_rpm, "rotor_speed_rpm")
        azimuth_reference = _finite_float(
            self.azimuth_reference_deg,
            "azimuth_reference_deg",
        )
        rotation_sign = _strict_int(self.rotation_sign, "rotation_sign")
        if rotation_sign not in (-1, 1):
            raise ValueError("rotation_sign must be explicitly set to +1 or -1.")
        target_tolerance = _finite_float(
            self.target_generalized_force_tolerance,
            "target_generalized_force_tolerance",
        )
        off_target_tolerance = _finite_float(
            self.off_target_generalized_force_tolerance,
            "off_target_generalized_force_tolerance",
        )
        if target_tolerance <= 0.0 or off_target_tolerance <= 0.0:
            raise ValueError("Generalized-force tolerances must be positive.")
        minimum_samples = _strict_int(
            self.minimum_samples_per_cycle,
            "minimum_samples_per_cycle",
            minimum=MINIMUM_SAMPLES_PER_CYCLE,
        )
        maximum_cycle_error = _finite_float(
            self.maximum_realized_burst_cycle_error,
            "maximum_realized_burst_cycle_error",
        )
        if maximum_cycle_error <= 0.0:
            raise ValueError("maximum_realized_burst_cycle_error must be positive.")
        if maximum_cycle_error > MAXIMUM_REALIZED_BURST_CYCLE_ERROR:
            raise ValueError(
                "maximum_realized_burst_cycle_error cannot weaken the locked gate above "
                f"{MAXIMUM_REALIZED_BURST_CYCLE_ERROR:g}."
            )

        nyquist_frequency = 0.5 / timestep
        if frequency >= nyquist_frequency:
            raise ValueError(
                "frequency_hz must be strictly below the Nyquist frequency: "
                f"frequency={frequency}, Nyquist={nyquist_frequency}."
            )
        samples_per_cycle = 1.0 / (frequency * timestep)
        if samples_per_cycle < minimum_samples:
            raise ValueError(
                "Modal burst is under-resolved: "
                f"{samples_per_cycle:.9g} samples/cycle, required >= {minimum_samples:.9g}."
            )

        raw_start_step = burst_start / timestep
        rounded_start_step = int(round(raw_start_step))
        if not math.isclose(
            raw_start_step,
            rounded_start_step,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "burst_start_time_s must lie exactly on the timestep grid; "
                f"got {burst_start} s for dt={timestep} s."
            )
        burst_samples = int(round(burst_cycles / (frequency * timestep))) + 1
        if burst_samples < 3:
            raise ValueError("The Hann burst must contain at least three samples.")
        realized_cycles = frequency * (burst_samples - 1) * timestep
        realized_cycle_error = abs(realized_cycles - burst_cycles)
        if realized_cycle_error > maximum_cycle_error:
            raise ValueError(
                "Rounded Hann schedule exceeds the realized-cycle-error gate: "
                f"requested={burst_cycles}, realized={realized_cycles:.12g}, "
                f"error={realized_cycle_error:.12g}, limit={maximum_cycle_error:.12g}."
            )
        burst_end_sample = rounded_start_step + burst_samples - 1
        total_physical_samples = total_updates - 1
        last_physical_sample = total_physical_samples - 1
        if burst_end_sample >= last_physical_sample:
            raise ValueError(
                "total_wrapper_updates must retain bootstrap plus at least one explicitly "
                "zero physical sample after the burst; "
                f"burst ends at physical sample {burst_end_sample}, "
                f"last physical sample={last_physical_sample}."
            )

        # Normalize every serialized field to a canonical built-in type.
        canonical = {
            "timestep_s": timestep,
            "total_wrapper_updates": total_updates,
            "bootstrap_zero": True,
            "rotor_speed_rpm": rotor_speed,
            "target_mode": target_mode,
            "frequency_hz": frequency,
            "generalized_force_amplitude_n": amplitude,
            "burst_start_time_s": burst_start,
            "burst_cycles": burst_cycles,
            "azimuth_reference_deg": azimuth_reference,
            "rotation_sign": rotation_sign,
            "target_generalized_force_tolerance": target_tolerance,
            "off_target_generalized_force_tolerance": off_target_tolerance,
            "minimum_samples_per_cycle": minimum_samples,
            "maximum_realized_burst_cycle_error": maximum_cycle_error,
        }
        for name, value in canonical.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "ModalBurstPlan":
        if raw.get("schema") != MODAL_BURST_PLAN_SCHEMA:
            raise ValueError(
                f"Modal-burst plan schema must be {MODAL_BURST_PLAN_SCHEMA!r}."
            )
        field_names = {
            "timestep_s",
            "total_wrapper_updates",
            "bootstrap_zero",
            "rotor_speed_rpm",
            "target_mode",
            "frequency_hz",
            "generalized_force_amplitude_n",
            "burst_start_time_s",
            "burst_cycles",
            "azimuth_reference_deg",
            "rotation_sign",
            "target_generalized_force_tolerance",
            "off_target_generalized_force_tolerance",
            "minimum_samples_per_cycle",
            "maximum_realized_burst_cycle_error",
        }
        derived_names = {
            "total_physical_samples",
            "last_physical_sample_index",
            "burst_start_sample",
            "burst_sample_count",
            "burst_end_sample",
            "samples_per_cycle",
            "realized_burst_cycles",
            "realized_burst_cycle_error",
        }
        missing = sorted(field_names - set(raw))
        if missing:
            raise ValueError(f"Modal-burst plan is missing field(s): {', '.join(missing)}")
        allowed = field_names | derived_names | {"schema"}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Modal-burst plan has unsupported field(s): {', '.join(unknown)}")
        values = {name: raw[name] for name in field_names}
        plan = cls(**values)  # type: ignore[arg-type]
        serialized = plan.as_dict()
        for name in sorted(derived_names & set(raw)):
            observed = raw[name]
            expected = serialized[name]
            if isinstance(expected, float):
                try:
                    matches = math.isclose(
                        _finite_float(observed, f"plan {name}"),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-14,
                    )
                except ValueError:
                    matches = False
            else:
                matches = type(observed) is int and observed == expected
            if not matches:
                raise ValueError(
                    f"Modal-burst derived field {name!r} is inconsistent: "
                    f"expected {expected!r}, observed {observed!r}."
                )
        return plan

    @property
    def total_physical_samples(self) -> int:
        return self.total_wrapper_updates - 1

    @property
    def last_physical_sample_index(self) -> int:
        return self.total_physical_samples - 1

    @property
    def burst_start_sample(self) -> int:
        return int(round(self.burst_start_time_s / self.timestep_s))

    @property
    def burst_sample_count(self) -> int:
        return int(round(self.burst_cycles / (self.frequency_hz * self.timestep_s))) + 1

    @property
    def burst_end_sample(self) -> int:
        return self.burst_start_sample + self.burst_sample_count - 1

    @property
    def samples_per_cycle(self) -> float:
        return 1.0 / (self.frequency_hz * self.timestep_s)

    @property
    def realized_burst_cycles(self) -> float:
        return self.frequency_hz * (self.burst_sample_count - 1) * self.timestep_s

    @property
    def realized_burst_cycle_error(self) -> float:
        return abs(self.realized_burst_cycles - self.burst_cycles)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "schema": MODAL_BURST_PLAN_SCHEMA,
                "total_physical_samples": self.total_physical_samples,
                "last_physical_sample_index": self.last_physical_sample_index,
                "burst_start_sample": self.burst_start_sample,
                "burst_sample_count": self.burst_sample_count,
                "burst_end_sample": self.burst_end_sample,
                "samples_per_cycle": self.samples_per_cycle,
                "realized_burst_cycles": self.realized_burst_cycles,
                "realized_burst_cycle_error": self.realized_burst_cycle_error,
            }
        )
        return result


@dataclass(frozen=True)
class QBladeModalForceAction:
    """One ordered, global QBlade ``ADDFORCE`` action."""

    index: int
    node_id: str
    action: str
    component_id: str
    position: float
    direction: str
    local: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModalBurstForceSample:
    """Pure force design for one physical sample; no QBlade action is implied."""

    physical_sample_index: int | None
    physical_time_s: float
    azimuth_deg: float
    bootstrap_zero: bool
    burst_sample_index: int | None
    hann_weight: float
    phase_rad: float | None
    scheduled_zero: bool
    body_force_n: tuple[float, ...]
    global_force_n: tuple[float, ...]
    global_force_sha256: str
    target_complex_generalized_force_phasor_real_n: float
    target_complex_generalized_force_phasor_imag_n: float
    target_complex_generalized_force_phasor_error_n: float
    maximum_off_target_complex_generalized_force_phasor_n: float

    def design_record(self) -> dict[str, object]:
        maximum = max((abs(value) for value in self.global_force_n), default=0.0)
        record: dict[str, object] = {
            "schema": MODAL_BURST_STEP_SCHEMA,
            "record_kind": "designed_force_sample",
            "physical_sample_index": self.physical_sample_index,
            "physical_time_s": self.physical_time_s,
            "azimuth_deg": self.azimuth_deg,
            "bootstrap_zero": self.bootstrap_zero,
            "burst_sample_index": self.burst_sample_index,
            "hann_weight": self.hann_weight,
            "phase_rad": self.phase_rad,
            "scheduled_zero": self.scheduled_zero,
            "designed_force_frame": "body-fixed then rotated to QBlade global XYZ",
            "complex_phasor_convention": (
                "positive-frequency body-frame D*exp(+i*omega*t); the designed physical "
                "force is its real part"
            ),
            "force_component_count": len(self.global_force_n),
            "global_force_sha256": self.global_force_sha256,
            "maximum_absolute_force_component_n": maximum,
            "generalized_force_audit_convention": "Phi^H of the complex force phasor",
            "target_complex_generalized_force_phasor_real_n": (
                self.target_complex_generalized_force_phasor_real_n
            ),
            "target_complex_generalized_force_phasor_imag_n": (
                self.target_complex_generalized_force_phasor_imag_n
            ),
            "target_complex_generalized_force_phasor_error_n": (
                self.target_complex_generalized_force_phasor_error_n
            ),
            "maximum_off_target_complex_generalized_force_phasor_n": (
                self.maximum_off_target_complex_generalized_force_phasor_n
            ),
        }
        _require_exact_keys(record, _DESIGN_RECORD_KEYS, "Modal-burst design record")
        return record


def _load_force_actions(
    data: bytes,
    source: Path,
) -> tuple[tuple[QBladeModalForceAction, ...], dict[str, object]]:
    try:
        config = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid QBlade external configuration JSON: {source}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"QBlade external configuration must be an object: {source}")
    swap_size = _strict_int(config.get("swap_size"), "external swap_size", minimum=1)
    if swap_size != EXPECTED_SWAP_SIZE:
        raise ValueError(
            f"Modal-burst external swap_size must be {EXPECTED_SWAP_SIZE}; got {swap_size}."
        )

    input_rows = config.get("external_input_rows")
    if not isinstance(input_rows, list) or len(input_rows) != 120:
        raise ValueError("Modal-burst external table must define exactly 120 input rows.")
    try:
        sorted_inputs = sorted(input_rows, key=lambda row: int(row["index"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Malformed external_input_rows in QBlade external configuration.") from exc
    if [int(row["index"]) for row in sorted_inputs] != list(range(120)):
        raise ValueError("Modal-burst external input indices must be exactly 0:119.")
    expected_prefix = (TIME_CHANNEL, "Timestep [-]", AZIMUTH_CHANNEL)
    if tuple(str(row.get("variable")) for row in sorted_inputs[:3]) != expected_prefix:
        raise ValueError(
            "Modal-burst external input rows 0:2 must be time, timestep, and LSS azimuth."
        )

    raw_node_ids = config.get("node_ids")
    if (
        not isinstance(raw_node_ids, list)
        or len(raw_node_ids) != EXPECTED_SITE_COUNT
        or any(not isinstance(node_id, str) or not node_id for node_id in raw_node_ids)
        or len(set(raw_node_ids)) != EXPECTED_SITE_COUNT
    ):
        raise ValueError("Modal-burst external configuration must contain 39 unique node_ids.")
    node_ids = tuple(str(node_id) for node_id in raw_node_ids)

    output_rows = config.get("external_output_rows")
    if not isinstance(output_rows, list) or len(output_rows) != EXPECTED_ACTION_COUNT:
        raise ValueError("Modal-burst external table must define exactly 117 output actions.")
    try:
        sorted_outputs = sorted(output_rows, key=lambda row: int(row["index"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Malformed external_output_rows in QBlade external configuration."
        ) from exc
    expected_indices = list(range(120, 120 + EXPECTED_ACTION_COUNT))
    if [int(row["index"]) for row in sorted_outputs] != expected_indices:
        raise ValueError("Modal-burst force action indices must be exactly 120:236.")

    actions: list[QBladeModalForceAction] = []
    for site_index in range(EXPECTED_SITE_COUNT):
        group = sorted_outputs[3 * site_index : 3 * site_index + 3]
        component_ids = {str(row.get("id", "")) for row in group}
        positions = [_finite_float(row.get("position"), "force action position") for row in group]
        if any(position < 0.0 or position > 1.0 for position in positions):
            raise ValueError(
                f"Force action site {site_index} positions must lie in normalized [0, 1]."
            )
        directions = tuple(str(row.get("direction", "")).upper() for row in group)
        if len(component_ids) != 1 or "" in component_ids:
            raise ValueError(f"Force action site {site_index} has inconsistent component IDs.")
        if max(positions) - min(positions) > 1.0e-12:
            raise ValueError(f"Force action site {site_index} has inconsistent positions.")
        if directions != EXPECTED_DIRECTIONS:
            raise ValueError(
                f"Force action site {site_index} directions must be X, Y, Z; got {directions}."
            )
        for row, direction, position in zip(group, directions, positions):
            if str(row.get("action", "")).upper() != "ADDFORCE":
                raise ValueError("Every modal-burst output row must use ADDFORCE.")
            local = row.get("local")
            if not isinstance(local, bool) or local:
                raise ValueError(
                    "Every modal-burst output row must be explicitly global (local=false)."
                )
            actions.append(
                QBladeModalForceAction(
                    index=int(row["index"]),
                    node_id=node_ids[site_index],
                    action="ADDFORCE",
                    component_id=next(iter(component_ids)),
                    position=position,
                    direction=direction,
                    local=False,
                )
            )
    action_site_node_ids = [actions[3 * site].node_id for site in range(EXPECTED_SITE_COUNT)]
    if len(set(action_site_node_ids)) != EXPECTED_SITE_COUNT:
        raise ValueError("Modal-burst force actions must retain 39 unique site node_ids.")
    physical_targets = {
        (actions[3 * site].component_id, float(actions[3 * site].position))
        for site in range(EXPECTED_SITE_COUNT)
    }
    if len(physical_targets) != EXPECTED_SITE_COUNT:
        raise ValueError(
            "Modal-burst force actions must address 39 unique (component_id, position) "
            "physical targets."
        )
    return tuple(actions), config


def _layout_key_from_csv_row(
    row: Mapping[str, object],
) -> tuple[str, str, float, tuple[int, int, int]]:
    return (
        str(row["action_node_id"]),
        str(row["qblade_component"]),
        _finite_float(row["qblade_fraction"], "qblade_fraction"),
        (
            int(row["qblade_force_output_index_x"]),
            int(row["qblade_force_output_index_y"]),
            int(row["qblade_force_output_index_z"]),
        ),
    )


def _validate_artifact_site_layout(
    rows: Sequence[Mapping[str, object]],
    actions: Sequence[QBladeModalForceAction],
    *,
    artifact_label: str,
) -> list[Mapping[str, object]]:
    try:
        ordered = sorted(rows, key=lambda row: int(row["qblade_force_output_index_x"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{artifact_label} has malformed force output indices.") from exc
    if len(ordered) != EXPECTED_SITE_COUNT:
        raise ValueError(f"{artifact_label} must contain exactly 39 action-site rows.")
    for site_index, row in enumerate(ordered):
        node_id, component_id, position, indices = _layout_key_from_csv_row(row)
        group = actions[3 * site_index : 3 * site_index + 3]
        expected_indices = tuple(action.index for action in group)
        if indices != expected_indices:
            raise ValueError(
                f"{artifact_label} site {site_index} force indices {indices} do not match "
                f"external actions {expected_indices}."
            )
        if node_id != group[0].node_id or component_id != group[0].component_id:
            raise ValueError(
                f"{artifact_label} site {site_index} does not match the external node layout."
            )
        if not math.isclose(position, group[0].position, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"{artifact_label} site {site_index} position does not match the external layout."
            )
    return ordered


def _load_action_basis(
    data: bytes,
    source: Path,
    *,
    rpm: float,
    actions: Sequence[QBladeModalForceAction],
) -> tuple[np.ndarray, dict[int, float], tuple[int, ...]]:
    rows = _read_csv_rows(data, source)
    _require_columns(
        rows,
        (
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
        ),
        source,
    )
    selected = [row for row in rows if _rpm_matches(row["rpm"], rpm)]
    grouped: dict[int, list[Mapping[str, object]]] = {}
    for row in selected:
        try:
            mode = int(row["tracked_mode"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid tracked_mode in {source}.") from exc
        grouped.setdefault(mode, []).append(row)
    if tuple(sorted(grouped)) != EXPECTED_BASIS_MODES:
        raise ValueError(
            f"{source} must contain modes 1:12 at {rpm} RPM; got {sorted(grouped)}."
        )

    frequencies: dict[int, float] = {}
    columns: list[np.ndarray] = []
    for mode in EXPECTED_BASIS_MODES:
        ordered = _validate_artifact_site_layout(
            grouped[mode],
            actions,
            artifact_label=f"action-shape RPM {rpm:g}, mode {mode}",
        )
        frequency_values = [
            _finite_float(row["frequency_hz"], f"mode {mode} frequency") for row in ordered
        ]
        if any(
            not math.isclose(value, frequency_values[0], rel_tol=1.0e-12, abs_tol=1.0e-12)
            for value in frequency_values[1:]
        ):
            raise ValueError(f"Mode {mode} has inconsistent frequencies across action sites.")
        if frequency_values[0] <= 0.0:
            raise ValueError(f"Mode {mode} frequency must be positive.")
        frequencies[mode] = frequency_values[0]
        values: list[complex] = []
        for row in ordered:
            for axis in ("ux", "uy", "uz"):
                values.append(
                    complex(
                        _finite_float(row[f"{axis}_per_m_real"], f"{axis} real"),
                        _finite_float(row[f"{axis}_per_m_imag"], f"{axis} imag"),
                    )
                )
        column = np.asarray(values, dtype=complex)
        if column.shape != (EXPECTED_ACTION_COUNT,) or not np.all(np.isfinite(column)):
            raise ValueError(f"Mode {mode} action shape is incomplete or non-finite.")
        columns.append(column)
    phi = np.column_stack(columns)
    singular_values = np.linalg.svd(phi, compute_uv=False)
    tolerance = max(phi.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank != len(EXPECTED_BASIS_MODES):
        raise ValueError(
            f"The saved action basis is rank deficient: rank {rank}, expected 12."
        )
    condition = float(singular_values[0] / singular_values[-1])
    if condition > 1.0e8:
        raise ValueError(
            f"The saved action basis condition number {condition:.6g} exceeds 1e8."
        )
    phi.setflags(write=False)
    return phi, frequencies, tuple(int(mode) for mode in EXPECTED_BASIS_MODES)


def _load_force_dual(
    data: bytes,
    source: Path,
    *,
    rpm: float,
    target_mode: int,
    actions: Sequence[QBladeModalForceAction],
) -> tuple[np.ndarray, int]:
    rows = _read_csv_rows(data, source)
    _require_columns(
        rows,
        (
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
        ),
        source,
    )
    selected = [
        row
        for row in rows
        if _rpm_matches(row["rpm"], rpm) and int(row["target_mode"]) == target_mode
    ]
    ordered = _validate_artifact_site_layout(
        selected,
        actions,
        artifact_label=f"force-dual RPM {rpm:g}, mode {target_mode}",
    )
    site_rows = [int(row["action_site_row_zero_based"]) for row in ordered]
    # The modal generator's interpolation order is not the external swap-index
    # order for the struts.  Both basis and dual are deliberately permuted to
    # swap order above; only the one-to-one site-index set must be retained.
    if sorted(site_rows) != list(range(EXPECTED_SITE_COUNT)):
        raise ValueError("Force-dual action_site_row_zero_based must retain each row 0:38 once.")
    target_columns = {int(row["target_basis_column_zero_based"]) for row in ordered}
    if target_columns != {target_mode - 1}:
        raise ValueError(
            "Force-dual target basis column does not match the tracked target mode."
        )
    values: list[complex] = []
    for row in ordered:
        for axis in ("x", "y", "z"):
            values.append(
                complex(
                    _finite_float(
                        row[f"force_{axis}_n_per_generalized_force_real"],
                        f"force dual {axis} real",
                    ),
                    _finite_float(
                        row[f"force_{axis}_n_per_generalized_force_imag"],
                        f"force dual {axis} imag",
                    ),
                )
            )
    dual = np.asarray(values, dtype=complex)
    if dual.shape != (EXPECTED_ACTION_COUNT,) or not np.all(np.isfinite(dual)):
        raise ValueError("Force-dual vector is incomplete or non-finite.")
    if float(np.linalg.norm(dual)) <= 0.0:
        raise ValueError("Force-dual vector must not be zero.")
    dual.setflags(write=False)
    return dual, target_mode - 1


class SilModalBurstCoupler:
    """Apply an audited modal burst in the same wrapper call, with no software queue."""

    def __init__(
        self,
        *,
        plan: ModalBurstPlan,
        actions: Sequence[QBladeModalForceAction],
        action_basis: np.ndarray,
        force_dual: np.ndarray,
        basis_frequencies_hz: Mapping[int, float],
        basis_modes: Sequence[int] = EXPECTED_BASIS_MODES,
        artifact_identities: Sequence[Mapping[str, object]] = (),
    ) -> None:
        self.plan = plan
        self.actions = tuple(actions)
        self.basis_modes = tuple(
            _strict_int(mode, "basis mode", minimum=1) for mode in basis_modes
        )
        if len(self.actions) != EXPECTED_ACTION_COUNT:
            raise ValueError(
                f"Modal-burst coupler requires exactly {EXPECTED_ACTION_COUNT} actions."
            )
        if tuple(action.index for action in self.actions) != tuple(range(120, 237)):
            raise ValueError("Modal-burst actions must be ordered at external indices 120:236.")
        if any(
            not isinstance(action.local, bool)
            or action.local
            or action.action != "ADDFORCE"
            for action in self.actions
        ):
            raise ValueError("Modal-burst actions must all be global ADDFORCE actions.")
        action_site_node_ids: list[str] = []
        physical_targets: list[tuple[str, float]] = []
        for site_index in range(EXPECTED_SITE_COUNT):
            group = self.actions[3 * site_index : 3 * site_index + 3]
            if any(
                not isinstance(action.node_id, str)
                or not action.node_id
                or not isinstance(action.component_id, str)
                or not action.component_id
                or not math.isfinite(float(action.position))
                or float(action.position) < 0.0
                or float(action.position) > 1.0
                for action in group
            ):
                raise ValueError(
                    f"Modal-burst action site {site_index} has invalid IDs or position "
                    "outside [0, 1]."
                )
            if tuple(action.direction for action in group) != EXPECTED_DIRECTIONS:
                raise ValueError(
                    f"Modal-burst action site {site_index} must be ordered global X, Y, Z."
                )
            if len({action.node_id for action in group}) != 1:
                raise ValueError(f"Modal-burst action site {site_index} has inconsistent node IDs.")
            if len({action.component_id for action in group}) != 1:
                raise ValueError(
                    f"Modal-burst action site {site_index} has inconsistent component IDs."
                )
            if max(action.position for action in group) - min(
                action.position for action in group
            ) > 1.0e-12:
                raise ValueError(
                    f"Modal-burst action site {site_index} has inconsistent positions."
                )
            action_site_node_ids.append(group[0].node_id)
            physical_targets.append((group[0].component_id, float(group[0].position)))
        if len(set(action_site_node_ids)) != EXPECTED_SITE_COUNT:
            raise ValueError("Modal-burst actions must contain 39 unique site node_ids.")
        if len(set(physical_targets)) != EXPECTED_SITE_COUNT:
            raise ValueError(
                "Modal-burst actions must address 39 unique (component_id, position) "
                "physical targets."
            )
        if self.basis_modes != EXPECTED_BASIS_MODES:
            raise ValueError("Modal-burst action basis must use tracked modes 1:12 in order.")
        if (
            any(type(mode) is not int for mode in basis_frequencies_hz)
            or set(basis_frequencies_hz) != set(self.basis_modes)
        ):
            raise ValueError("basis_frequencies_hz must contain exactly tracked modes 1:12.")
        self.basis_frequencies_hz = tuple(
            _finite_float(basis_frequencies_hz[mode], f"mode {mode} basis frequency")
            for mode in self.basis_modes
        )
        if any(frequency <= 0.0 for frequency in self.basis_frequencies_hz):
            raise ValueError("Every modal-basis frequency must be positive.")
        if not math.isclose(
            self.basis_frequencies_hz[self.basis_modes.index(plan.target_mode)],
            plan.frequency_hz,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Target basis frequency does not match the locked modal-burst plan.")

        phi = np.array(action_basis, dtype=complex, copy=True)
        dual = np.array(force_dual, dtype=complex, copy=True).reshape(-1)
        if phi.shape != (EXPECTED_ACTION_COUNT, len(EXPECTED_BASIS_MODES)):
            raise ValueError(
                f"action_basis must have shape (117, 12); got {phi.shape}."
            )
        if dual.shape != (EXPECTED_ACTION_COUNT,):
            raise ValueError(f"force_dual must have shape (117,); got {dual.shape}.")
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(dual)):
            raise ValueError("Modal action basis and force dual must be finite.")
        singular_values = np.linalg.svd(phi, compute_uv=False)
        tolerance = max(phi.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.count_nonzero(singular_values > tolerance))
        condition = (
            float("inf")
            if rank != len(EXPECTED_BASIS_MODES)
            else float(singular_values[0] / singular_values[-1])
        )
        if rank != len(EXPECTED_BASIS_MODES) or condition > 1.0e8:
            raise ValueError(
                "Modal action basis failed the rank/conditioning gate: "
                f"rank={rank}, condition={condition}."
            )

        target_column = self.basis_modes.index(plan.target_mode)
        generalized = phi.conj().T @ dual
        target_error = float(abs(generalized[target_column] - 1.0))
        off_target = float(
            max(
                (
                    abs(value)
                    for index, value in enumerate(generalized)
                    if index != target_column
                ),
                default=0.0,
            )
        )
        if target_error > plan.target_generalized_force_tolerance:
            raise ValueError(
                "Force dual failed Phi^H F target gate: "
                f"error={target_error:.6g}, limit={plan.target_generalized_force_tolerance:.6g}."
            )
        if off_target > plan.off_target_generalized_force_tolerance:
            raise ValueError(
                "Force dual failed Phi^H F off-target gate: "
                f"maximum={off_target:.6g}, "
                f"limit={plan.off_target_generalized_force_tolerance:.6g}."
            )

        phi.setflags(write=False)
        dual.setflags(write=False)
        self.action_basis = phi
        self.force_dual = dual
        self.target_basis_column = target_column
        self.generalized_force_audit = {
            "convention": "Phi^H D = e_target",
            "basis_mode_order": list(self.basis_modes),
            "target_mode": int(plan.target_mode),
            "target_basis_column_zero_based": int(target_column),
            "target_real": float(generalized[target_column].real),
            "target_imag": float(generalized[target_column].imag),
            "target_error": target_error,
            "maximum_off_target": off_target,
            "rank": rank,
            "condition_number": condition,
            "passed": True,
        }
        self.artifact_identities = _validate_artifact_identities(artifact_identities)
        self._identity = self._build_identity()
        self.wrapper_updates = 0
        self.physical_samples_applied = 0
        self.bootstrap_updates = 0
        self.max_abs_force = 0.0
        self.scheduled_nonzero_physical_samples = 0
        self.exact_zero_physical_samples = 0
        self.last_record: dict[str, object] | None = None
        self._lifecycle = "ready"
        self._final_summary: dict[str, object] | None = None

    @classmethod
    def from_artifacts(
        cls,
        *,
        plan: ModalBurstPlan,
        external_config_path: Path,
        action_shapes_path: Path,
        force_duals_path: Path,
        expected_sha256: Mapping[str, str] | None = None,
        additional_artifact_identities: Sequence[Mapping[str, object]] = (),
    ) -> "SilModalBurstCoupler":
        expected = dict(expected_sha256 or {})
        unknown_expected_roles = sorted(
            set(expected) - {"external_config", "action_shapes", "force_duals"}
        )
        if unknown_expected_roles:
            raise ValueError(
                "Unsupported expected modal-burst artifact role(s): "
                + ", ".join(unknown_expected_roles)
            )
        paths = {
            "external_config": Path(external_config_path),
            "action_shapes": Path(action_shapes_path),
            "force_duals": Path(force_duals_path),
        }
        contents: dict[str, bytes] = {}
        identities: list[dict[str, object]] = []
        for role, artifact_path in paths.items():
            data, identity = _read_artifact_bytes(
                role,
                artifact_path,
                expected_sha256=expected.get(role),
            )
            contents[role] = data
            identities.append(identity)
        actions, _ = _load_force_actions(
            contents["external_config"],
            paths["external_config"],
        )
        phi, frequencies, basis_modes = _load_action_basis(
            contents["action_shapes"],
            paths["action_shapes"],
            rpm=plan.rotor_speed_rpm,
            actions=actions,
        )
        artifact_frequency = frequencies[plan.target_mode]
        if not math.isclose(
            artifact_frequency,
            plan.frequency_hz,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Modal-burst plan frequency does not match the selected action-shape artifact: "
                f"plan={plan.frequency_hz}, artifact={artifact_frequency}."
        )
        dual, target_column = _load_force_dual(
            contents["force_duals"],
            paths["force_duals"],
            rpm=plan.rotor_speed_rpm,
            target_mode=plan.target_mode,
            actions=actions,
        )
        if target_column != basis_modes.index(plan.target_mode):
            raise ValueError("Force-dual target column does not match the action-basis mode order.")
        identities.extend(dict(identity) for identity in additional_artifact_identities)
        return cls(
            plan=plan,
            actions=actions,
            action_basis=phi,
            force_dual=dual,
            basis_frequencies_hz=frequencies,
            basis_modes=basis_modes,
            artifact_identities=identities,
        )

    @classmethod
    def from_plan_file(cls, path: Path) -> "SilModalBurstCoupler":
        plan_data, plan_identity = _read_artifact_bytes("plan", Path(path))
        plan_path = Path(str(plan_identity["path"]))
        try:
            raw = json.loads(plan_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid modal-burst plan JSON: {plan_path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Modal-burst plan JSON must contain an object.")
        plan_mapping = dict(raw)
        raw_artifacts = plan_mapping.pop("artifacts", None)
        if not isinstance(raw_artifacts, dict):
            raise ValueError("Modal-burst plan must contain an artifacts object.")
        roles = ("external_config", "action_shapes", "force_duals")
        if set(raw_artifacts) != set(roles):
            raise ValueError(
                "Modal-burst plan artifacts must contain exactly external_config, "
                "action_shapes, and force_duals."
            )
        resolved_paths: dict[str, Path] = {}
        expected: dict[str, str] = {}
        for role in roles:
            spec = raw_artifacts[role]
            if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
                raise ValueError(
                    f"Modal-burst artifact {role} must contain exactly path and sha256."
                )
            raw_path = spec["path"]
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"Modal-burst artifact {role} path must be a nonempty string.")
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = plan_path.parent / candidate
            resolved_paths[role] = candidate
            if not isinstance(spec["sha256"], str):
                raise ValueError(f"Modal-burst artifact {role} sha256 must be a string.")
            expected[role] = spec["sha256"]
        plan = ModalBurstPlan.from_mapping(plan_mapping)
        return cls.from_artifacts(
            plan=plan,
            external_config_path=resolved_paths["external_config"],
            action_shapes_path=resolved_paths["action_shapes"],
            force_duals_path=resolved_paths["force_duals"],
            expected_sha256=expected,
            additional_artifact_identities=(plan_identity,),
        )

    def _build_identity(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": MODAL_BURST_IDENTITY_SCHEMA,
            "version": 1,
            "plan": self.plan.as_dict(),
            "bootstrap_contract": (
                "wrapper update 0 writes all-zero actions; wrapper update j>=1 designs and "
                "applies physical sample j-1 at observed QBlade pre-advance time"
            ),
            "software_force_queue_depth": 0,
            "generation_and_application_same_wrapper_call": True,
            "action_application_order": "before_advance",
            "qblade_pre_time_grid_policy": _qblade_pre_time_policy(),
            "action_layout": [action.as_dict() for action in self.actions],
            "artifacts": [
                _stable_artifact_identity(identity) for identity in self.artifact_identities
            ],
            "generalized_force_audit": dict(self.generalized_force_audit),
            "action_basis_shape": list(self.action_basis.shape),
            "action_basis_sha256": _complex_array_sha256(
                self.action_basis,
                schema="vorlap-modal-action-basis-complex-v1",
            ),
            "basis_frequencies_hz": [float(value) for value in self.basis_frequencies_hz],
            "basis_frequencies_sha256": _float_vector_sha256(
                self.basis_frequencies_hz,
                schema="vorlap-modal-basis-frequencies-v1",
            ),
            "force_dual_sha256": _complex_array_sha256(
                self.force_dual,
                schema="vorlap-modal-force-dual-complex-v1",
            ),
            "force_dual_real": [float(value.real) for value in self.force_dual],
            "force_dual_imag": [float(value.imag) for value in self.force_dual],
        }
        payload["fingerprint"] = _canonical_fingerprint(payload)
        return payload

    def checkpoint_identity(self) -> dict[str, object]:
        """Return a JSON-safe identity binding schedule, basis, dual, and artifacts."""
        return json.loads(json.dumps(self._identity, sort_keys=True))

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @staticmethod
    def _rotate_body_to_global(
        body_force: np.ndarray,
        *,
        azimuth_deg: float,
        azimuth_reference_deg: float,
        rotation_sign: int,
    ) -> np.ndarray:
        values = np.asarray(body_force, dtype=float).reshape((-1, 3))
        delta_deg = math.remainder(float(azimuth_deg) - float(azimuth_reference_deg), 360.0)
        signed_angle = float(rotation_sign) * math.radians(delta_deg)
        if signed_angle == 0.0:
            result = np.array(values, dtype=float, copy=True)
        else:
            cosine = math.cos(signed_angle)
            sine = math.sin(signed_angle)
            result = np.empty_like(values)
            result[:, 0] = cosine * values[:, 0] - sine * values[:, 1]
            result[:, 1] = sine * values[:, 0] + cosine * values[:, 1]
            result[:, 2] = values[:, 2]
        return result.reshape(-1)

    def _zero_force_sample(
        self,
        *,
        physical_sample_index: int | None,
        physical_time_s: float,
        azimuth_deg: float,
        bootstrap_zero: bool,
        burst_sample_index: int | None = None,
        phase_rad: float | None = None,
    ) -> ModalBurstForceSample:
        zeros = np.zeros(EXPECTED_ACTION_COUNT, dtype=float)
        return ModalBurstForceSample(
            physical_sample_index=physical_sample_index,
            physical_time_s=float(physical_time_s),
            azimuth_deg=float(azimuth_deg),
            bootstrap_zero=bool(bootstrap_zero),
            burst_sample_index=burst_sample_index,
            hann_weight=0.0,
            phase_rad=phase_rad,
            scheduled_zero=True,
            body_force_n=tuple(float(value) for value in zeros),
            global_force_n=tuple(float(value) for value in zeros),
            global_force_sha256=_force_vector_sha256(zeros),
            target_complex_generalized_force_phasor_real_n=0.0,
            target_complex_generalized_force_phasor_imag_n=0.0,
            target_complex_generalized_force_phasor_error_n=0.0,
            maximum_off_target_complex_generalized_force_phasor_n=0.0,
        )

    def force_sample(
        self,
        physical_sample_index: int,
        *,
        azimuth_deg: float,
    ) -> ModalBurstForceSample:
        """Design one positive-frequency physical sample without applying it."""
        sample_index = _strict_int(
            physical_sample_index,
            "physical_sample_index",
            minimum=0,
        )
        if sample_index >= self.plan.total_physical_samples:
            raise ValueError(
                f"physical_sample_index {sample_index} is outside the planned "
                f"0:{self.plan.last_physical_sample_index} range."
            )
        azimuth = _finite_float(azimuth_deg, "azimuth_deg")
        time_s = sample_index * self.plan.timestep_s
        burst_index = sample_index - self.plan.burst_start_sample
        inside = 0 <= burst_index < self.plan.burst_sample_count
        endpoint = inside and burst_index in (0, self.plan.burst_sample_count - 1)
        scheduled_zero = (
            not inside
            or endpoint
            or self.plan.generalized_force_amplitude_n == 0.0
        )

        if scheduled_zero:
            burst_sample_index = burst_index if inside else None
            phase_rad = (
                2.0 * math.pi * self.plan.frequency_hz * burst_index * self.plan.timestep_s
                if inside
                else None
            )
            return self._zero_force_sample(
                physical_sample_index=sample_index,
                physical_time_s=time_s,
                azimuth_deg=azimuth,
                bootstrap_zero=False,
                burst_sample_index=burst_sample_index,
                phase_rad=None if phase_rad is None else float(phase_rad),
            )
        else:
            burst_sample_index = burst_index
            hann_weight = 0.5 * (
                1.0
                - math.cos(
                    2.0 * math.pi * burst_index / (self.plan.burst_sample_count - 1)
                )
            )
            phase_rad = (
                2.0 * math.pi * self.plan.frequency_hz * burst_index * self.plan.timestep_s
            )
            expected_target = (
                self.plan.generalized_force_amplitude_n
                * hann_weight
                * complex(math.cos(phase_rad), math.sin(phase_rad))
            )
            complex_force = self.force_dual * expected_target
            body = np.asarray(complex_force.real, dtype=float)
            generalized = self.action_basis.conj().T @ complex_force
            global_force = self._rotate_body_to_global(
                body,
                azimuth_deg=azimuth,
                azimuth_reference_deg=self.plan.azimuth_reference_deg,
                rotation_sign=self.plan.rotation_sign,
            )

        target_error = float(abs(generalized[self.target_basis_column] - expected_target))
        off_target = float(
            max(
                (
                    abs(value)
                    for index, value in enumerate(generalized)
                    if index != self.target_basis_column
                ),
                default=0.0,
            )
        )
        scale = abs(expected_target)
        if target_error > max(1.0e-13, self.plan.target_generalized_force_tolerance * scale):
            raise RuntimeError(
                "Scheduled complex force failed its target generalized-force audit: "
                f"error={target_error:.6g}."
            )
        if off_target > max(1.0e-13, self.plan.off_target_generalized_force_tolerance * scale):
            raise RuntimeError(
                "Scheduled complex force failed its off-target generalized-force audit: "
                f"maximum={off_target:.6g}."
            )
        return ModalBurstForceSample(
            physical_sample_index=sample_index,
            physical_time_s=float(time_s),
            azimuth_deg=azimuth,
            bootstrap_zero=False,
            burst_sample_index=burst_sample_index,
            hann_weight=float(hann_weight),
            phase_rad=None if phase_rad is None else float(phase_rad),
            scheduled_zero=bool(scheduled_zero),
            body_force_n=tuple(float(value) for value in body),
            global_force_n=tuple(float(value) for value in global_force),
            global_force_sha256=_force_vector_sha256(global_force),
            target_complex_generalized_force_phasor_real_n=float(
                generalized[self.target_basis_column].real
            ),
            target_complex_generalized_force_phasor_imag_n=float(
                generalized[self.target_basis_column].imag
            ),
            target_complex_generalized_force_phasor_error_n=target_error,
            maximum_off_target_complex_generalized_force_phasor_n=off_target,
        )

    @staticmethod
    def _qblade_value(qblade, channel: str) -> float:
        value = float(qblade.getCustomData_at_num(channel.encode("utf-8"), 0.0, 0))
        if not math.isfinite(value):
            raise RuntimeError(f"QBlade returned missing or non-finite {channel!r}: {value}.")
        return value

    def validate_channels(self, qblade) -> dict[str, float]:
        """Require finite pre-advance time and azimuth channels before stepping."""
        return {
            TIME_CHANNEL: self._qblade_value(qblade, TIME_CHANNEL),
            AZIMUTH_CHANNEL: self._qblade_value(qblade, AZIMUTH_CHANNEL),
        }

    def step(self, qblade, step_index: int) -> str:
        """Design and apply actions in one pre-advance wrapper call, without a queue."""
        if self._lifecycle == "finalized":
            raise RuntimeError("Cannot step a finalized modal-burst coupler.")
        wrapper_step = _strict_int(step_index, "wrapper_step_index", minimum=0)
        if wrapper_step != self.wrapper_updates:
            raise RuntimeError(
                "Modal-burst wrapper steps must be contiguous and pre-advance: "
                f"expected {self.wrapper_updates}, received {wrapper_step}."
            )
        if wrapper_step >= self.plan.total_wrapper_updates:
            raise RuntimeError("Modal-burst wrapper updates exceed the locked plan.")
        physical_sample_index = None if wrapper_step == 0 else wrapper_step - 1
        qblade_time = self._qblade_value(qblade, TIME_CHANNEL)
        time_grid_audit = _validate_qblade_pre_time(
            qblade_time,
            timestep_s=self.plan.timestep_s,
            expected_physical_sample_index=physical_sample_index,
        )
        expected_time = float(time_grid_audit["expected_time_s"])
        azimuth = self._qblade_value(qblade, AZIMUTH_CHANNEL)
        if physical_sample_index is None:
            sample = self._zero_force_sample(
                physical_sample_index=None,
                physical_time_s=0.0,
                azimuth_deg=azimuth,
                bootstrap_zero=True,
            )
        else:
            sample = self.force_sample(physical_sample_index, azimuth_deg=azimuth)
        record = sample.design_record()
        for action, value in zip(self.actions, sample.global_force_n):
            qblade.setExternalAction(
                action.action.encode("utf-8"),
                action.component_id.encode("utf-8"),
                float(value),
                float(action.position),
                action.direction.encode("utf-8"),
                False,
                0,
            )

        # Application provenance is added only after every setExternalAction
        # call above has completed successfully.
        record.update(
            {
                "record_kind": "applied_force_sample",
                "wrapper_step_index": wrapper_step,
                "generation_wrapper_step_index": wrapper_step,
                "application_wrapper_step_index": wrapper_step,
                "qblade_time_pre_advance_s": qblade_time,
                "qblade_pre_time_grid_audit": time_grid_audit,
                "software_force_queue_depth": 0,
                "generation_and_application_same_wrapper_call": True,
                "action_application_order": "before_advance",
                "action_application_completed": True,
                "applied_action_count": EXPECTED_ACTION_COUNT,
            }
        )
        _require_exact_keys(record, _APPLIED_RECORD_KEYS, "Modal-burst applied record")
        self.wrapper_updates += 1
        if sample.bootstrap_zero:
            self.bootstrap_updates += 1
        else:
            self.physical_samples_applied += 1
            if sample.scheduled_zero:
                self.exact_zero_physical_samples += 1
            else:
                self.scheduled_nonzero_physical_samples += 1
        self.max_abs_force = max(
            self.max_abs_force,
            max((abs(value) for value in sample.global_force_n), default=0.0),
        )
        self.last_record = record
        self._lifecycle = "active"
        state = (
            "bootstrap-zero"
            if sample.bootstrap_zero
            else ("physical-zero" if sample.scheduled_zero else "physical-burst")
        )
        return (
            f"Applied {len(self.actions)} modal SIL actions in wrapper call {wrapper_step} "
            f"with no software queue ({state}, QBlade pre-time={expected_time:.9g} s)."
        )

    def _expected_lifecycle_counts(self, wrapper_updates: int) -> tuple[int, int, int, int]:
        bootstrap_updates = 1 if wrapper_updates > 0 else 0
        physical_samples = max(0, wrapper_updates - bootstrap_updates)
        if self.plan.generalized_force_amplitude_n == 0.0:
            nonzero = 0
        else:
            first = self.plan.burst_start_sample + 1
            stop = self.plan.burst_end_sample
            nonzero = max(
                0,
                min(physical_samples, stop) - min(physical_samples, first),
            )
        return bootstrap_updates, physical_samples, nonzero, physical_samples - nonzero

    def checkpoint_state(self) -> dict[str, object]:
        """Serialize restart state; no software force packet is pending."""
        state: dict[str, object] = {
            "schema": MODAL_BURST_CHECKPOINT_SCHEMA,
            "identity_fingerprint": self._identity["fingerprint"],
            "lifecycle": self._lifecycle,
            "wrapper_updates": int(self.wrapper_updates),
            "next_wrapper_step_index": int(self.wrapper_updates),
            "physical_samples_applied": int(self.physical_samples_applied),
            "bootstrap_updates": int(self.bootstrap_updates),
            "software_force_queue_depth": 0,
            "pending_force_packets": [],
            "max_abs_force": float(self.max_abs_force),
            "scheduled_nonzero_physical_samples": int(
                self.scheduled_nonzero_physical_samples
            ),
            "exact_zero_physical_samples": int(self.exact_zero_physical_samples),
            "last_record": (
                None
                if self.last_record is None
                else json.loads(json.dumps(self.last_record, allow_nan=False, sort_keys=True))
            ),
        }
        _require_exact_keys(state, _CHECKPOINT_KEYS, "Modal-burst checkpoint")
        return state

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None:
        """Atomically restore a checkpoint only when its complete identity matches."""
        if self.wrapper_updates != 0 or self.last_record is not None or self._lifecycle != "ready":
            raise RuntimeError("Modal-burst state can only be restored into a fresh coupler.")
        if not isinstance(state, Mapping):
            raise ValueError("Modal-burst checkpoint must be a mapping.")
        _require_exact_keys(state, _CHECKPOINT_KEYS, "Modal-burst checkpoint")
        if state.get("schema") != MODAL_BURST_CHECKPOINT_SCHEMA:
            raise ValueError("Modal-burst checkpoint schema is missing or incompatible.")
        if state.get("identity_fingerprint") != self._identity["fingerprint"]:
            raise ValueError("Modal-burst checkpoint identity does not match this coupler.")
        if state.get("software_force_queue_depth") != 0:
            raise ValueError("Modal-burst checkpoints must retain an empty software force queue.")
        if state.get("pending_force_packets") != []:
            raise ValueError(
                "Modal-burst checkpoints cannot contain pending software force packets."
            )
        updates = _strict_int(
            state.get("wrapper_updates"),
            "checkpoint wrapper_updates",
            minimum=0,
        )
        if updates > self.plan.total_wrapper_updates:
            raise ValueError("Modal-burst checkpoint updates exceed total_wrapper_updates.")
        next_wrapper_step = _strict_int(
            state.get("next_wrapper_step_index"),
            "checkpoint next_wrapper_step_index",
            minimum=0,
        )
        if next_wrapper_step != updates:
            raise ValueError("Checkpoint next_wrapper_step_index does not match wrapper_updates.")
        physical_samples = _strict_int(
            state.get("physical_samples_applied"),
            "checkpoint physical_samples_applied",
            minimum=0,
        )
        bootstrap_updates = _strict_int(
            state.get("bootstrap_updates"),
            "checkpoint bootstrap_updates",
            minimum=0,
        )
        nonzero = _strict_int(
            state.get("scheduled_nonzero_physical_samples"),
            "checkpoint scheduled_nonzero_physical_samples",
            minimum=0,
        )
        exact_zero = _strict_int(
            state.get("exact_zero_physical_samples"),
            "checkpoint exact_zero_physical_samples",
            minimum=0,
        )
        expected_counts = self._expected_lifecycle_counts(updates)
        if (bootstrap_updates, physical_samples, nonzero, exact_zero) != expected_counts:
            raise ValueError("Modal-burst checkpoint wrapper/physical counts are inconsistent.")
        max_abs_force = _finite_float(state.get("max_abs_force"), "checkpoint max_abs_force")
        if max_abs_force < 0.0:
            raise ValueError("Modal-burst checkpoint max_abs_force must be nonnegative.")
        maximum_dual_node_norm = float(
            np.max(np.linalg.norm(self.force_dual.reshape((-1, 3)), axis=1))
        )
        maximum_possible_force = (
            self.plan.generalized_force_amplitude_n * maximum_dual_node_norm
        )
        if max_abs_force > maximum_possible_force * (1.0 + 1.0e-12) + 1.0e-14:
            raise ValueError("Modal-burst checkpoint max_abs_force exceeds the planned bound.")
        if nonzero == 0 and max_abs_force != 0.0:
            raise ValueError("A zero-only modal-burst checkpoint must have zero max_abs_force.")
        raw_last = state.get("last_record")
        restored_last: dict[str, object] | None
        if updates == 0:
            if raw_last is not None or max_abs_force != 0.0:
                raise ValueError("An empty modal-burst checkpoint must not contain force history.")
            restored_last = None
        else:
            if not isinstance(raw_last, dict):
                raise ValueError("Modal-burst checkpoint is missing its last step record.")
            restored_last = dict(raw_last)
            _require_exact_keys(
                restored_last,
                _APPLIED_RECORD_KEYS,
                "Modal-burst checkpoint last_record",
            )
            if restored_last.get("schema") != MODAL_BURST_STEP_SCHEMA:
                raise ValueError("Modal-burst checkpoint last_record has an invalid schema.")
            last_wrapper_step = updates - 1
            if restored_last.get("wrapper_step_index") != last_wrapper_step:
                raise ValueError("Modal-burst checkpoint last wrapper step is inconsistent.")
            application_fields = {
                "record_kind": "applied_force_sample",
                "generation_wrapper_step_index": last_wrapper_step,
                "application_wrapper_step_index": last_wrapper_step,
                "software_force_queue_depth": 0,
                "generation_and_application_same_wrapper_call": True,
                "action_application_order": "before_advance",
                "action_application_completed": True,
                "applied_action_count": EXPECTED_ACTION_COUNT,
            }
            for key, expected_value in application_fields.items():
                if restored_last.get(key) != expected_value:
                    raise ValueError(
                        f"Modal-burst checkpoint application field {key!r} is inconsistent."
                    )
            azimuth = _finite_float(
                restored_last.get("azimuth_deg"),
                "checkpoint last_record azimuth_deg",
            )
            if last_wrapper_step == 0:
                expected_sample = self._zero_force_sample(
                    physical_sample_index=None,
                    physical_time_s=0.0,
                    azimuth_deg=azimuth,
                    bootstrap_zero=True,
                )
            else:
                expected_sample = self.force_sample(
                    last_wrapper_step - 1,
                    azimuth_deg=azimuth,
                )
            expected_record = expected_sample.design_record()
            for key, expected_value in expected_record.items():
                if key == "record_kind":
                    continue
                if restored_last.get(key) != expected_value:
                    raise ValueError(
                        f"Modal-burst checkpoint last_record field {key!r} is inconsistent."
                    )
            qblade_time = _finite_float(
                restored_last.get("qblade_time_pre_advance_s"),
                "checkpoint last_record qblade_time_pre_advance_s",
            )
            expected_physical_sample_index = (
                None if last_wrapper_step == 0 else last_wrapper_step - 1
            )
            try:
                expected_time_audit = _validate_qblade_pre_time(
                    qblade_time,
                    timestep_s=self.plan.timestep_s,
                    expected_physical_sample_index=expected_physical_sample_index,
                )
            except RuntimeError as exc:
                raise ValueError(
                    "Modal-burst checkpoint last QBlade time is inconsistent."
                ) from exc
            raw_time_audit = restored_last.get("qblade_pre_time_grid_audit")
            if not isinstance(raw_time_audit, dict):
                raise ValueError(
                    "Modal-burst checkpoint QBlade time-grid audit must be an object."
                )
            _require_exact_keys(
                raw_time_audit,
                _QBLADE_PRE_TIME_GRID_AUDIT_KEYS,
                "Modal-burst checkpoint QBlade time-grid audit",
            )
            if raw_time_audit != expected_time_audit:
                raise ValueError(
                    "Modal-burst checkpoint QBlade time-grid audit is inconsistent."
                )
            last_maximum = _finite_float(
                restored_last.get("maximum_absolute_force_component_n"),
                "checkpoint last_record maximum force",
            )
            if last_maximum > max_abs_force + 1.0e-14:
                raise ValueError("Modal-burst checkpoint accumulated max is below its last step.")
        lifecycle = state.get("lifecycle")
        if lifecycle not in {"ready", "active", "restored"}:
            raise ValueError("Modal-burst checkpoint lifecycle is invalid for resume.")
        if updates == 0 and lifecycle != "ready":
            raise ValueError("An empty modal-burst checkpoint must have ready lifecycle.")
        if updates > 0 and lifecycle == "ready":
            raise ValueError("A nonempty modal-burst checkpoint cannot have ready lifecycle.")

        self.wrapper_updates = updates
        self.physical_samples_applied = physical_samples
        self.bootstrap_updates = bootstrap_updates
        self.max_abs_force = max_abs_force
        self.scheduled_nonzero_physical_samples = nonzero
        self.exact_zero_physical_samples = exact_zero
        self.last_record = (
            None
            if restored_last is None
            else json.loads(json.dumps(restored_last, allow_nan=False, sort_keys=True))
        )
        self._lifecycle = "restored" if updates > 0 else "ready"

    def finalize(self) -> dict[str, object]:
        """Close a complete schedule only after an explicit post-burst zero step."""
        if self._final_summary is not None:
            return json.loads(json.dumps(self._final_summary, sort_keys=True))
        if self.wrapper_updates != self.plan.total_wrapper_updates:
            raise RuntimeError(
                "Cannot finalize an incomplete modal-burst schedule: "
                f"completed {self.wrapper_updates}/{self.plan.total_wrapper_updates} "
                "wrapper updates."
            )
        if self.last_record is None or self.last_record.get("scheduled_zero") is not True:
            raise RuntimeError(
                "Completed modal-burst schedule did not end with explicit zero force."
            )
        expected_counts = self._expected_lifecycle_counts(self.wrapper_updates)
        observed_counts = (
            self.bootstrap_updates,
            self.physical_samples_applied,
            self.scheduled_nonzero_physical_samples,
            self.exact_zero_physical_samples,
        )
        if observed_counts != expected_counts:
            raise RuntimeError(
                "Modal-burst lifecycle counts do not match the planned Hann schedule."
            )
        summary: dict[str, object] = {
            "schema": "vorlap-qblade-modal-burst-summary-v1",
            "identity_fingerprint": self._identity["fingerprint"],
            "wrapper_updates": int(self.wrapper_updates),
            "physical_samples_applied": int(self.physical_samples_applied),
            "bootstrap_updates": int(self.bootstrap_updates),
            "software_force_queue_depth": 0,
            "generation_and_application_same_wrapper_call": True,
            "action_application_order": "before_advance",
            "scheduled_nonzero_physical_samples": int(
                self.scheduled_nonzero_physical_samples
            ),
            "exact_zero_physical_samples": int(self.exact_zero_physical_samples),
            "max_abs_force": float(self.max_abs_force),
            "ended_with_explicit_zero": True,
            "plan": self.plan.as_dict(),
            "generalized_force_audit": dict(self.generalized_force_audit),
            "last_record": dict(self.last_record),
        }
        self._final_summary = summary
        self._lifecycle = "finalized"
        return json.loads(json.dumps(summary, sort_keys=True))


__all__ = [
    "AZIMUTH_CHANNEL",
    "EXPECTED_ACTION_COUNT",
    "EXPECTED_BASIS_MODES",
    "EXPECTED_SITE_COUNT",
    "MAXIMUM_REALIZED_BURST_CYCLE_ERROR",
    "MINIMUM_SAMPLES_PER_CYCLE",
    "MODAL_BURST_CHECKPOINT_SCHEMA",
    "MODAL_BURST_FORCE_DIGEST_SCHEMA",
    "MODAL_BURST_IDENTITY_SCHEMA",
    "MODAL_BURST_PLAN_SCHEMA",
    "MODAL_BURST_STEP_SCHEMA",
    "QBLADE_PRE_TIME_ABSOLUTE_FLOOR_S",
    "QBLADE_PRE_TIME_FLOAT32_ULPS",
    "QBLADE_PRE_TIME_GRID_AUDIT_SCHEMA",
    "QBLADE_PRE_TIME_MAX_TIMESTEP_FRACTION",
    "ModalBurstForceSample",
    "ModalBurstPlan",
    "QBladeModalForceAction",
    "SilModalBurstCoupler",
    "TIME_CHANNEL",
]
