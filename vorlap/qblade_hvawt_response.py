"""Fail-closed response core for the fixed-axis 1 RPM H-VAWT.

This module deliberately does not know how QBlade swap tables, runners, or
campaigns are built.  It accepts already sampled SI-valued kinematics at the
39 audited action locations and returns an additive, real, global force.  The
legacy VorLap/QBlade source path is not modified.

The rigid/flexible split is co-rotational.  For current global position
``x_i``, total structural velocity ``v_i``, fixed shaft origin ``o`` and shaft
angular velocity ``omega``,

``v_rb_i = omega x (x_i - o)`` and ``v_flex_i = v_i - v_rb_i``.

Base VorLap flow, when selected by a future integration layer, is therefore
``U_inf - v_i``.  The response closure instead uses ``U_inf - v_rb_i`` for
activation and aerodynamic prefactors and uses only ``v_flex`` to estimate
modal velocity.  This prevents elastic velocity from changing the declared
response-flow speed while retaining its feedback through modal power.

The audited body-fixed complex mode matrix is ``Phi`` [m/m] and its
minimum-norm dual is ``F=(Phi^+)^H`` [N/N], so ``Phi^H F=I``.  Modes 6 and 12
pass the stored phase-fixed real-shape gate.  With ``psi=Re(Phi[:, (6,12)])``
and selected dual columns ``F_s``, the real two-mode coordinate extraction is

``b = Re(F_s^H v_flex_body)`` and ``qdot = C^-1 b``,
``C = Re(F_s^H psi)``.

The complex dual is used for coordinate isolation and diagnostics only.  It
is not silently reduced to a real aerodynamic force dual.  The real three-
dimensional force distribution uses ``psi`` and the existing Scanlan-style
velocity-proportional equation.  Cylinder-derived coefficients remain an
incremental VIV screening closure, not total native-QBlade aerodynamic
damping or certification evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import csv
import hashlib
import io
import json
import math
import numbers
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from .force_model import strouhal_reference_length


HVAWT_RESPONSE_CORE_SCHEMA_VERSION = 1
HVAWT_RESPONSE_CHECKPOINT_SCHEMA = "vorlap-qblade-hvawt-response-core-v1"
HVAWT_MODAL_BASIS_SCHEMA = "vorlap-hvawt-response-mode-identification-v2"
HVAWT_MODAL_REVIEW_SCHEMA = "vorlap-hvawt-modal-evidence-review-marker-v2"
HVAWT_MODAL_REVIEW_MARKER_SHA256 = (
    "de237c327b55a6a9ccb4dc127ccd1a34cac258cc1b4b7a44de4e19c85cb4a843"
)
HVAWT_MODAL_REVIEW_MARKER_SIZE_BYTES = 4048
HVAWT_MODAL_IDENTIFICATION_SIZE_BYTES = 184137

NATIVE_AERO_REQUIRED = "required_enabled"
EXTERNAL_VIV_NONE = "none"
EXTERNAL_VIV_LEGACY = "legacy_vorlap_source"
EXTERNAL_VIV_RESPONSE = "response_closure"
INCREMENTAL_VIV_SCREENING = "incremental_viv_screening"
NOT_APPLICABLE = "not_applicable"

FIXED_AXIS_MOTION_MODEL = "fixed_global_axis"
CURRENT_POSITION_BASIS = "current_global_deformed_positions"
TOTAL_VELOCITY_BASIS = "total_global_structural_velocity"
BASE_FLOW_BASIS = "ambient_minus_total_structural"
RESPONSE_FLOW_BASIS = "ambient_minus_rigid_body"
FEEDBACK_VELOCITY_BASIS = "total_minus_rigid_body"
REAL_REDUCTION_POLICY = "phase_anchored_real_part_no_rescale"
COORDINATE_EXTRACTOR = "real_part_complex_dual_two_mode_corrected"

_TARGET_MODES = (6, 12)
_SITE_COUNT = 39
_ACTION_DOF_COUNT = 3 * _SITE_COUNT
_RPM_TO_RADPS = 2.0 * math.pi / 60.0
_TIME_TOLERANCE_S = 1.0e-12


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _strict_finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real numeric scalar, not {type(value).__name__}.")
    return _finite_scalar(value, name)


def _finite_vector3(value: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly three finite values.")
    return np.array(array, dtype=float, copy=True)


def _finite_node_vectors(value: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (count, 3) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must have shape ({count}, 3) with finite values.")
    return np.array(array, dtype=float, copy=True)


def _finite_node_scalars(value: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(count, float(array), dtype=float)
    if array.shape != (count,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be scalar or a finite ({count},) array.")
    return np.array(array, dtype=float, copy=True)


def _finite_exact_scalars(value: Any, count: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (count,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite ({count},) array.")
    return np.array(array, dtype=float, copy=True)


def _hex_digest(value: Any, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 digest.")
    return digest


@dataclass(frozen=True)
class _StableArtifactBytes:
    path: Path
    payload: bytes
    sha256: str
    size_bytes: int


def _artifact_stat_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        stat.S_IFMT(file_stat.st_mode),
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            component_stat = current.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(current) from None
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValueError(
                f"{label} path component {current} must not be a symbolic link."
            )


def _read_stable_regular_file(path: Path, *, label: str) -> _StableArtifactBytes:
    """Read one immutable snapshot of a stable, non-symlink regular file."""

    artifact_path = Path(path)
    _assert_no_symlink_components(artifact_path, label=label)
    try:
        path_before = artifact_path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(artifact_path) from None
    if stat.S_ISLNK(path_before.st_mode):
        raise ValueError(f"{label} must not be a symbolic link.")
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"{label} must be a regular file.")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(artifact_path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} opened object is not a regular file.")
        if _artifact_stat_identity(opened) != _artifact_stat_identity(path_before):
            raise ValueError(f"{label} changed between path validation and open.")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = None
            payload = stream.read()
            descriptor_after = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError(f"{label} could not be opened as a stable regular file: {error}.") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)

    try:
        path_after = artifact_path.lstat()
    except FileNotFoundError:
        raise ValueError(f"{label} disappeared while it was being read.") from None
    if stat.S_ISLNK(path_after.st_mode) or not stat.S_ISREG(path_after.st_mode):
        raise ValueError(f"{label} ceased to be a non-symlink regular file while read.")
    _assert_no_symlink_components(artifact_path, label=label)
    identities = (
        _artifact_stat_identity(path_before),
        _artifact_stat_identity(opened),
        _artifact_stat_identity(descriptor_after),
        _artifact_stat_identity(path_after),
    )
    if len(set(identities)) != 1 or len(payload) != int(opened.st_size):
        raise ValueError(f"{label} changed while it was being read.")
    immutable_payload = bytes(payload)
    return _StableArtifactBytes(
        path=artifact_path,
        payload=immutable_payload,
        sha256=hashlib.sha256(immutable_payload).hexdigest(),
        size_bytes=len(immutable_payload),
    )


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        raise ValueError(f"{label} keys differ: missing={missing}, extra={extra}.")


def _canonical_mapping_sha256(mapping: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_array_bytes(array: Any, *, complex_values: bool = False) -> bytes:
    dtype = "<c16" if complex_values else "<f8"
    canonical = np.ascontiguousarray(array, dtype=dtype)
    shape = np.asarray(canonical.shape, dtype="<i8").tobytes()
    return shape + canonical.tobytes()


def _copy_readonly(array: Any, *, dtype: Any = float) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _axis_angle_rotation(axis_unit: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = np.asarray(axis_unit, dtype=float)
    cosine = math.cos(float(angle_rad))
    sine = math.sin(float(angle_rad))
    one_minus = 1.0 - cosine
    return np.array(
        [
            [cosine + x * x * one_minus, x * y * one_minus - z * sine, x * z * one_minus + y * sine],
            [y * x * one_minus + z * sine, cosine + y * y * one_minus, y * z * one_minus - x * sine],
            [z * x * one_minus - y * sine, z * y * one_minus + x * sine, cosine + z * z * one_minus],
        ],
        dtype=float,
    )


def _smooth_activation(value: np.ndarray, low: float, peak: float, high: float) -> np.ndarray:
    reduced_velocity = np.asarray(value, dtype=float)
    activation = np.zeros_like(reduced_velocity)
    rising = (reduced_velocity > low) & (reduced_velocity < peak)
    falling = (reduced_velocity >= peak) & (reduced_velocity < high)
    rise_coordinate = (reduced_velocity[rising] - low) / (peak - low)
    fall_coordinate = (high - reduced_velocity[falling]) / (high - peak)
    activation[rising] = rise_coordinate**2 * (3.0 - 2.0 * rise_coordinate)
    activation[falling] = fall_coordinate**2 * (3.0 - 2.0 * fall_coordinate)
    activation[reduced_velocity == peak] = 1.0
    return activation


@dataclass(frozen=True)
class HVAWTForceSourceContract:
    """Exclusive external-force policy for native-aero H-VAWT scenarios."""

    native_qblade_aerodynamics: str = NATIVE_AERO_REQUIRED
    external_viv_mode: str = EXTERNAL_VIV_RESPONSE
    include_mean_load: bool = False
    coefficient_role: str = INCREMENTAL_VIV_SCREENING
    schema_version: int = HVAWT_RESPONSE_CORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, (bool, np.bool_))
            or not isinstance(self.schema_version, numbers.Integral)
            or int(self.schema_version) != HVAWT_RESPONSE_CORE_SCHEMA_VERSION
        ):
            raise ValueError("force-source schema_version must be 1.")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if self.native_qblade_aerodynamics != NATIVE_AERO_REQUIRED:
            raise ValueError("H-VAWT response scenarios require native QBlade aerodynamics enabled.")
        if self.external_viv_mode not in {
            EXTERNAL_VIV_NONE,
            EXTERNAL_VIV_LEGACY,
            EXTERNAL_VIV_RESPONSE,
        }:
            raise ValueError("Unsupported external_viv_mode.")
        if not isinstance(self.include_mean_load, bool) or self.include_mean_load:
            raise ValueError("H-VAWT external VIV paths require include_mean_load=False.")
        expected_role = (
            INCREMENTAL_VIV_SCREENING
            if self.external_viv_mode == EXTERNAL_VIV_RESPONSE
            else NOT_APPLICABLE
        )
        if self.coefficient_role != expected_role:
            raise ValueError(
                f"coefficient_role must be {expected_role!r} for external_viv_mode "
                f"{self.external_viv_mode!r}."
            )

    @property
    def legacy_source_enabled(self) -> bool:
        return self.external_viv_mode == EXTERNAL_VIV_LEGACY

    @property
    def response_closure_enabled(self) -> bool:
        return self.external_viv_mode == EXTERNAL_VIV_RESPONSE

    def validate_runtime_flags(
        self,
        *,
        native_aero_enabled: bool,
        legacy_source_enabled: bool,
        response_closure_enabled: bool,
        include_mean_load: bool,
    ) -> None:
        """Reject any runtime composition that differs from the frozen enum."""

        values = (
            native_aero_enabled,
            legacy_source_enabled,
            response_closure_enabled,
            include_mean_load,
        )
        if any(not isinstance(value, bool) for value in values):
            raise TypeError("Runtime force-source flags must be booleans.")
        if not native_aero_enabled:
            raise ValueError("Native QBlade aerodynamics are required by this contract.")
        if legacy_source_enabled and response_closure_enabled:
            raise ValueError("Legacy VorLap source and response closure are mutually exclusive.")
        expected = (self.legacy_source_enabled, self.response_closure_enabled, False)
        actual = (legacy_source_enabled, response_closure_enabled, include_mean_load)
        if actual != expected:
            raise ValueError(
                "Runtime force-source flags do not match the frozen external_viv_mode."
            )


@dataclass(frozen=True)
class FixedAxisKinematicsConfig:
    """Hashable fixed-axis frame and sign contract in QBlade global units."""

    node_ids: Tuple[str, ...]
    axis_unit_global: Tuple[float, float, float]
    axis_origin_global_m: Tuple[float, float, float]
    body_to_global_at_reference: Tuple[Tuple[float, float, float], ...]
    reference_azimuth_deg: float
    basis_rpm: float = 1.0
    maximum_abs_basis_rpm_error_rpm: float = 0.01
    maximum_azimuth_rate_error_radps: float = 1.0e-5
    rpm_sign: int = 1
    azimuth_sign: int = 1
    axis_motion_model: str = FIXED_AXIS_MOTION_MODEL
    position_basis: str = CURRENT_POSITION_BASIS
    velocity_basis: str = TOTAL_VELOCITY_BASIS

    def __post_init__(self) -> None:
        if isinstance(self.node_ids, (str, bytes)) or any(
            not isinstance(node_id, str) or not node_id for node_id in self.node_ids
        ):
            raise TypeError("node_ids must contain nonempty strings.")
        node_ids = tuple(self.node_ids)
        object.__setattr__(self, "node_ids", node_ids)
        if len(node_ids) != _SITE_COUNT or len(set(node_ids)) != _SITE_COUNT:
            raise ValueError("Fixed-axis H-VAWT kinematics require 39 unique node_ids.")
        axis = _finite_vector3(self.axis_unit_global, "axis_unit_global")
        norm = float(np.linalg.norm(axis))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("axis_unit_global must be unit length.")
        origin = _finite_vector3(self.axis_origin_global_m, "axis_origin_global_m")
        rotation = np.asarray(self.body_to_global_at_reference, dtype=float)
        if rotation.shape != (3, 3) or np.any(~np.isfinite(rotation)):
            raise ValueError("body_to_global_at_reference must be a finite 3x3 matrix.")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1.0e-12):
            raise ValueError("body_to_global_at_reference must be orthogonal.")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("body_to_global_at_reference must be a proper rotation.")
        reference_azimuth = _strict_finite_scalar(
            self.reference_azimuth_deg,
            "reference_azimuth_deg",
        )
        basis_rpm = _strict_finite_scalar(self.basis_rpm, "basis_rpm")
        rpm_error = _strict_finite_scalar(
            self.maximum_abs_basis_rpm_error_rpm,
            "maximum_abs_basis_rpm_error_rpm",
        )
        rate_error = _strict_finite_scalar(
            self.maximum_azimuth_rate_error_radps,
            "maximum_azimuth_rate_error_radps",
        )
        if basis_rpm <= 0.0 or rpm_error <= 0.0 or rate_error <= 0.0:
            raise ValueError("Basis RPM and kinematic tolerances must be positive.")
        if (
            isinstance(self.rpm_sign, (bool, np.bool_))
            or not isinstance(self.rpm_sign, numbers.Integral)
            or isinstance(self.azimuth_sign, (bool, np.bool_))
            or not isinstance(self.azimuth_sign, numbers.Integral)
            or self.rpm_sign not in (-1, 1)
            or self.azimuth_sign not in (-1, 1)
        ):
            raise ValueError("rpm_sign and azimuth_sign must be +1 or -1.")
        if self.axis_motion_model != FIXED_AXIS_MOTION_MODEL:
            raise ValueError("Only a fixed global rotor axis is supported.")
        if self.position_basis != CURRENT_POSITION_BASIS:
            raise ValueError("Rigid velocity requires current global deformed positions.")
        if self.velocity_basis != TOTAL_VELOCITY_BASIS:
            raise ValueError("Kinematic decomposition requires total global structural velocity.")
        object.__setattr__(self, "axis_unit_global", tuple(float(value) for value in axis))
        object.__setattr__(self, "axis_origin_global_m", tuple(float(value) for value in origin))
        object.__setattr__(
            self,
            "body_to_global_at_reference",
            tuple(tuple(float(value) for value in row) for row in rotation),
        )
        object.__setattr__(self, "reference_azimuth_deg", reference_azimuth)
        object.__setattr__(self, "basis_rpm", basis_rpm)
        object.__setattr__(self, "maximum_abs_basis_rpm_error_rpm", rpm_error)
        object.__setattr__(self, "maximum_azimuth_rate_error_radps", rate_error)
        object.__setattr__(self, "rpm_sign", int(self.rpm_sign))
        object.__setattr__(self, "azimuth_sign", int(self.azimuth_sign))

    def rotation_body_to_global(self, azimuth_deg: float) -> np.ndarray:
        azimuth = _strict_finite_scalar(azimuth_deg, "azimuth_deg")
        delta_rad = math.radians(
            self.azimuth_sign * (azimuth - float(self.reference_azimuth_deg))
        )
        axis = _finite_vector3(self.axis_unit_global, "axis_unit_global")
        reference = np.asarray(self.body_to_global_at_reference, dtype=float)
        rotation = _axis_angle_rotation(axis, delta_rad) @ reference
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=2.0e-12):
            raise FloatingPointError("Body/global rotation lost orthogonality.")
        return rotation


@dataclass(frozen=True)
class FixedAxisKinematicState:
    rotation_body_to_global: np.ndarray
    angular_velocity_global_radps: np.ndarray
    rigid_body_velocity_global_mps: np.ndarray
    flexible_velocity_global_mps: np.ndarray
    maximum_total_speed_mps: float
    maximum_rigid_body_speed_mps: float
    maximum_flexible_speed_mps: float


def decompose_fixed_axis_velocity(
    config: FixedAxisKinematicsConfig,
    *,
    positions_global_m: Any,
    total_structural_velocity_global_mps: Any,
    lss_rpm: float,
    azimuth_deg: float,
) -> FixedAxisKinematicState:
    """Return ``v_global=v_rb+v_flex`` using current lever arms."""

    if not isinstance(config, FixedAxisKinematicsConfig):
        raise TypeError("config must be FixedAxisKinematicsConfig.")
    count = len(config.node_ids)
    positions = _finite_node_vectors(positions_global_m, count, "positions_global_m")
    total_velocity = _finite_node_vectors(
        total_structural_velocity_global_mps,
        count,
        "total_structural_velocity_global_mps",
    )
    rpm = _strict_finite_scalar(lss_rpm, "lss_rpm")
    signed_rpm = config.rpm_sign * rpm
    if abs(signed_rpm - config.basis_rpm) > config.maximum_abs_basis_rpm_error_rpm:
        raise ValueError(
            f"LSS RPM {signed_rpm:.17g} is outside the audited {config.basis_rpm:.17g} "
            f"RPM basis tolerance {config.maximum_abs_basis_rpm_error_rpm:.17g}."
        )
    axis = _finite_vector3(config.axis_unit_global, "axis_unit_global")
    omega = signed_rpm * _RPM_TO_RADPS * axis
    origin = _finite_vector3(config.axis_origin_global_m, "axis_origin_global_m")
    lever_arm = positions - origin[None, :]
    rigid = np.cross(np.broadcast_to(omega, lever_arm.shape), lever_arm)
    flexible = total_velocity - rigid
    if not np.array_equal(total_velocity, rigid + flexible):
        if not np.allclose(total_velocity, rigid + flexible, rtol=0.0, atol=2.0e-15):
            raise FloatingPointError("Rigid/flexible velocity closure failed.")
    rotation = config.rotation_body_to_global(azimuth_deg)
    return FixedAxisKinematicState(
        rotation_body_to_global=_copy_readonly(rotation),
        angular_velocity_global_radps=_copy_readonly(omega),
        rigid_body_velocity_global_mps=_copy_readonly(rigid),
        flexible_velocity_global_mps=_copy_readonly(flexible),
        maximum_total_speed_mps=float(np.max(np.linalg.norm(total_velocity, axis=1))),
        maximum_rigid_body_speed_mps=float(np.max(np.linalg.norm(rigid, axis=1))),
        maximum_flexible_speed_mps=float(np.max(np.linalg.norm(flexible, axis=1))),
    )


@dataclass(frozen=True)
class HVAWTSeparatedFlowState:
    base_relative_velocity_global_mps: np.ndarray
    response_relative_velocity_global_mps: np.ndarray
    base_effective_crossflow_speed_mps: np.ndarray
    response_effective_crossflow_speed_mps: np.ndarray
    response_aoa_deg: np.ndarray
    response_section_width_m: np.ndarray


def separate_hvawt_flows(
    *,
    ambient_velocity_global_mps: Any,
    total_structural_velocity_global_mps: Any,
    rigid_body_velocity_global_mps: Any,
    chord_unit_global: Any,
    normal_unit_global: Any,
    chord_m: Any,
    thickness_ratio: Any,
) -> HVAWTSeparatedFlowState:
    """Compute base ``U-v_global`` and response ``U-v_rb`` cross-flow."""

    total = np.asarray(total_structural_velocity_global_mps, dtype=float)
    if total.ndim != 2 or total.shape[1:] != (3,):
        raise ValueError("total_structural_velocity_global_mps must have shape (n, 3).")
    count = int(total.shape[0])
    total = _finite_node_vectors(total, count, "total_structural_velocity_global_mps")
    rigid = _finite_node_vectors(
        rigid_body_velocity_global_mps,
        count,
        "rigid_body_velocity_global_mps",
    )
    ambient_array = np.asarray(ambient_velocity_global_mps, dtype=float)
    if ambient_array.shape == (3,):
        ambient_array = np.broadcast_to(ambient_array, (count, 3))
    ambient = _finite_node_vectors(ambient_array, count, "ambient_velocity_global_mps")
    chord = _finite_node_vectors(chord_unit_global, count, "chord_unit_global")
    normal = _finite_node_vectors(normal_unit_global, count, "normal_unit_global")
    chord_norm = np.linalg.norm(chord, axis=1)
    normal_norm = np.linalg.norm(normal, axis=1)
    if np.any(chord_norm <= 0.0) or np.any(normal_norm <= 0.0):
        raise ValueError("Chord and normal basis vectors must be nonzero.")
    if (
        np.max(np.abs(chord_norm - 1.0)) > 1.0e-8
        or np.max(np.abs(normal_norm - 1.0)) > 1.0e-8
    ):
        raise ValueError("Chord and normal unit vectors must have unit length.")
    chord /= chord_norm[:, None]
    normal /= normal_norm[:, None]
    if np.max(np.abs(np.sum(chord * normal, axis=1))) > 1.0e-8:
        raise ValueError("Chord and normal basis vectors must be orthogonal.")
    chord_length = _finite_node_scalars(chord_m, count, "chord_m")
    thickness = _finite_node_scalars(thickness_ratio, count, "thickness_ratio")
    if np.any(chord_length <= 0.0) or np.any(thickness < 0.0):
        raise ValueError("Chord must be positive and thickness_ratio non-negative.")

    base = ambient - total
    response = ambient - rigid
    base_chord = np.sum(base * chord, axis=1)
    base_normal = np.sum(base * normal, axis=1)
    response_chord = np.sum(response * chord, axis=1)
    response_normal = np.sum(response * normal, axis=1)
    response_aoa = np.degrees(np.arctan2(response_normal, response_chord))
    response_width = strouhal_reference_length(
        chord_length,
        response_aoa,
        thickness,
    )
    return HVAWTSeparatedFlowState(
        base_relative_velocity_global_mps=_copy_readonly(base),
        response_relative_velocity_global_mps=_copy_readonly(response),
        base_effective_crossflow_speed_mps=_copy_readonly(
            np.hypot(base_chord, base_normal)
        ),
        response_effective_crossflow_speed_mps=_copy_readonly(
            np.hypot(response_chord, response_normal)
        ),
        response_aoa_deg=_copy_readonly(response_aoa),
        response_section_width_m=_copy_readonly(response_width),
    )


@dataclass(frozen=True)
class HVAWTModalBasis:
    """Audited selected complex shapes and duals in the body frame.

    Array layout is ``(39 sites, XYZ, 2 selected modes)``.  Shapes are
    dimensionless displacement per metre of generalized coordinate.  Duals
    are dimensionless nodal force per generalized force and also provide the
    kinematic left inverse used here.
    """

    node_ids: Tuple[str, ...]
    tracked_modes: Tuple[int, int]
    basis_rpm: float
    complex_shape_body_per_m: np.ndarray
    complex_dual_body: np.ndarray
    natural_frequency_hz: Tuple[float, float]
    generalized_modal_mass_kg: Tuple[float, float]
    generalized_modal_stiffness_npm: Tuple[float, float]
    identification_sha256: str
    shapes_sha256: str
    duals_sha256: str
    review_marker_sha256: str
    real_reduction_policy: str = REAL_REDUCTION_POLICY
    coordinate_extractor: str = COORDINATE_EXTRACTOR

    def __post_init__(self) -> None:
        if isinstance(self.node_ids, (str, bytes)) or any(
            not isinstance(node_id, str) or not node_id for node_id in self.node_ids
        ):
            raise TypeError("node_ids must contain nonempty strings.")
        if any(
            isinstance(mode, (bool, np.bool_))
            or not isinstance(mode, numbers.Integral)
            for mode in self.tracked_modes
        ):
            raise TypeError("tracked_modes must contain integer mode numbers.")
        node_ids = tuple(self.node_ids)
        modes = tuple(int(mode) for mode in self.tracked_modes)
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "tracked_modes", modes)
        if len(node_ids) != _SITE_COUNT or len(set(node_ids)) != _SITE_COUNT:
            raise ValueError("The H-VAWT modal basis requires 39 unique action node IDs.")
        if modes != _TARGET_MODES:
            raise ValueError("The first H-VAWT response core is restricted to tracked modes 6 and 12.")
        basis_rpm = _strict_finite_scalar(self.basis_rpm, "basis_rpm")
        if not math.isclose(basis_rpm, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("The tracked response basis must be the audited 1 RPM basis.")
        object.__setattr__(self, "basis_rpm", basis_rpm)
        shape = np.asarray(self.complex_shape_body_per_m, dtype=complex)
        dual = np.asarray(self.complex_dual_body, dtype=complex)
        expected_shape = (_SITE_COUNT, 3, len(_TARGET_MODES))
        if shape.shape != expected_shape or dual.shape != expected_shape:
            raise ValueError(f"Modal shapes and duals must have shape {expected_shape}.")
        if np.any(~np.isfinite(shape.real)) or np.any(~np.isfinite(shape.imag)):
            raise ValueError("Modal shapes must be finite.")
        if np.any(~np.isfinite(dual.real)) or np.any(~np.isfinite(dual.imag)):
            raise ValueError("Modal duals must be finite.")
        for mode_index, mode in enumerate(modes):
            complex_norm = float(np.linalg.norm(shape[:, :, mode_index]))
            real_norm = float(np.linalg.norm(shape[:, :, mode_index].real))
            imaginary_norm = float(np.linalg.norm(shape[:, :, mode_index].imag))
            maximum_site = float(
                np.max(np.linalg.norm(shape[:, :, mode_index], axis=1))
            )
            if complex_norm <= 0.0 or real_norm <= 0.0:
                raise ValueError(f"Mode {mode} has a degenerate action shape.")
            if not math.isclose(maximum_site, 1.0, rel_tol=0.0, abs_tol=2.0e-12):
                raise ValueError(f"Mode {mode} is not normalized to unit complex action amplitude.")
            if imaginary_norm / real_norm > 0.02 + 1.0e-15:
                raise ValueError(f"Mode {mode} fails the 0.02 imaginary/real action-shape gate.")

        phi = shape.reshape(_ACTION_DOF_COUNT, len(modes))
        force_dual = dual.reshape(_ACTION_DOF_COUNT, len(modes))
        selected_biorthogonality = phi.conj().T @ force_dual
        if float(np.max(np.abs(selected_biorthogonality - np.eye(2)))) > 1.0e-8:
            raise ValueError("Selected complex shapes and duals fail Phi^H F=I.")
        real_shape = phi.real
        correction = np.real(force_dual.conj().T @ real_shape)
        condition = float(np.linalg.cond(correction))
        if not math.isfinite(condition) or condition > 1.0e4:
            raise ValueError("The selected real coordinate-correction matrix is ill-conditioned.")
        if float(np.max(np.abs(correction - np.eye(2)))) > 0.01:
            raise ValueError("The selected real coordinate correction differs from identity by more than 0.01.")

        frequencies = tuple(
            _strict_finite_scalar(value, "natural_frequency_hz")
            for value in self.natural_frequency_hz
        )
        masses = tuple(
            _strict_finite_scalar(value, "generalized_modal_mass_kg")
            for value in self.generalized_modal_mass_kg
        )
        stiffnesses = tuple(
            _strict_finite_scalar(value, "generalized_modal_stiffness_npm")
            for value in self.generalized_modal_stiffness_npm
        )
        if len(frequencies) != 2 or len(masses) != 2 or len(stiffnesses) != 2:
            raise ValueError("Modal frequencies, masses, and stiffnesses require two values.")
        for mode, frequency, mass, stiffness in zip(modes, frequencies, masses, stiffnesses):
            if frequency <= 0.0 or mass <= 0.0 or stiffness <= 0.0:
                raise ValueError(f"Mode {mode} properties must be positive.")
            expected_stiffness = mass * (2.0 * math.pi * frequency) ** 2
            if not math.isclose(stiffness, expected_stiffness, rel_tol=2.0e-3, abs_tol=0.0):
                raise ValueError(f"Mode {mode} mass/stiffness/frequency identity failed.")
        object.__setattr__(self, "natural_frequency_hz", frequencies)
        object.__setattr__(self, "generalized_modal_mass_kg", masses)
        object.__setattr__(self, "generalized_modal_stiffness_npm", stiffnesses)
        object.__setattr__(self, "identification_sha256", _hex_digest(self.identification_sha256, "identification_sha256"))
        object.__setattr__(self, "shapes_sha256", _hex_digest(self.shapes_sha256, "shapes_sha256"))
        object.__setattr__(self, "duals_sha256", _hex_digest(self.duals_sha256, "duals_sha256"))
        object.__setattr__(
            self,
            "review_marker_sha256",
            _hex_digest(self.review_marker_sha256, "review_marker_sha256"),
        )
        if self.real_reduction_policy != REAL_REDUCTION_POLICY:
            raise ValueError("Unsupported real_reduction_policy.")
        if self.coordinate_extractor != COORDINATE_EXTRACTOR:
            raise ValueError("Unsupported coordinate_extractor.")
        object.__setattr__(self, "complex_shape_body_per_m", _copy_readonly(shape, dtype=complex))
        object.__setattr__(self, "complex_dual_body", _copy_readonly(dual, dtype=complex))

    @property
    def real_shape_body_per_m(self) -> np.ndarray:
        return _copy_readonly(self.complex_shape_body_per_m.real)

    @property
    def real_shape_maximum_site_norm(self) -> Tuple[float, float]:
        values = np.max(np.linalg.norm(self.real_shape_body_per_m, axis=1), axis=0)
        return tuple(float(value) for value in values)

    @property
    def coordinate_correction_matrix(self) -> np.ndarray:
        shape = self.real_shape_body_per_m.reshape(_ACTION_DOF_COUNT, 2)
        dual = self.complex_dual_body.reshape(_ACTION_DOF_COUNT, 2)
        return _copy_readonly(np.real(dual.conj().T @ shape))

    @property
    def real_coordinate_extractor_body(self) -> np.ndarray:
        """Return ``D`` [2,117] such that ``qdot=D*v_body``.

        The correction is required because the phase-fixed real shapes are a
        controlled reduction of the complex eigenvectors rather than an
        independently normalized real modal basis.
        """

        dual = self.complex_dual_body.reshape(_ACTION_DOF_COUNT, 2)
        extractor = np.linalg.solve(
            self.coordinate_correction_matrix,
            np.real(dual.conj().T),
        )
        return _copy_readonly(extractor)

    @property
    def real_force_adjoint_body(self) -> np.ndarray:
        """Return the energy-conjugate nodal force map ``D.T`` [39,3,2]."""

        return _copy_readonly(
            self.real_coordinate_extractor_body.T.reshape(_SITE_COUNT, 3, 2)
        )

    def energy_conjugate_force_body(self, generalized_force_n: Any) -> np.ndarray:
        """Distribute two generalized forces with exact real virtual work."""

        generalized = _finite_exact_scalars(
            generalized_force_n,
            2,
            "generalized_force_n",
        )
        force = np.einsum("iam,m->ia", self.real_force_adjoint_body, generalized)
        if np.any(~np.isfinite(force)):
            raise FloatingPointError("Energy-conjugate H-VAWT force became non-finite.")
        return _copy_readonly(force)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"vorlap-hvawt-modal-basis-v1\0")
        for node_id in self.node_ids:
            encoded = node_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(np.asarray(self.tracked_modes, dtype="<i8").tobytes())
        digest.update(np.asarray([self.basis_rpm], dtype="<f8").tobytes())
        digest.update(_canonical_array_bytes(self.complex_shape_body_per_m, complex_values=True))
        digest.update(_canonical_array_bytes(self.complex_dual_body, complex_values=True))
        digest.update(np.asarray(self.natural_frequency_hz, dtype="<f8").tobytes())
        digest.update(np.asarray(self.generalized_modal_mass_kg, dtype="<f8").tobytes())
        digest.update(np.asarray(self.generalized_modal_stiffness_npm, dtype="<f8").tobytes())
        for value in (
            self.identification_sha256,
            self.shapes_sha256,
            self.duals_sha256,
            self.review_marker_sha256,
            self.real_reduction_policy,
            self.coordinate_extractor,
        ):
            digest.update(str(value).encode("utf-8") + b"\0")
        return digest.hexdigest()

    def extract_modal_velocity(
        self,
        flexible_velocity_body_mps: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return corrected real qdot, raw complex projections, and leakage."""

        velocity = _finite_node_vectors(
            flexible_velocity_body_mps,
            _SITE_COUNT,
            "flexible_velocity_body_mps",
        ).reshape(_ACTION_DOF_COUNT)
        dual = self.complex_dual_body.reshape(_ACTION_DOF_COUNT, 2)
        raw_complex = dual.conj().T @ velocity
        corrected = self.real_coordinate_extractor_body @ velocity
        leakage = np.abs(raw_complex.imag) / np.maximum(
            np.abs(raw_complex.real),
            1.0e-12,
        )
        return (
            _copy_readonly(corrected),
            _copy_readonly(raw_complex, dtype=complex),
            _copy_readonly(leakage),
        )


def _require_snapshot_identity(
    snapshot: _StableArtifactBytes,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
    label: str,
) -> None:
    if snapshot.sha256 != _hex_digest(expected_sha256, f"{label} sha256"):
        raise ValueError(f"{label} SHA-256 differs from the tracked identity.")
    if expected_size is not None and snapshot.size_bytes != int(expected_size):
        raise ValueError(f"{label} size differs from the tracked identity.")


def load_tracked_hvawt_modal_basis(
    *,
    identification_path: Any,
    shapes_path: Any,
    duals_path: Any,
    review_marker_path: Any,
) -> HVAWTModalBasis:
    """Load modes 6/12 only after all tracked identities and gates pass."""

    identification_file = Path(identification_path)
    shapes_file = Path(shapes_path)
    duals_file = Path(duals_path)
    marker_file = Path(review_marker_path)
    marker_snapshot = _read_stable_regular_file(
        marker_file,
        label="modal review marker",
    )
    _require_snapshot_identity(
        marker_snapshot,
        expected_sha256=HVAWT_MODAL_REVIEW_MARKER_SHA256,
        expected_size=HVAWT_MODAL_REVIEW_MARKER_SIZE_BYTES,
        label="modal review marker",
    )
    marker = json.loads(marker_snapshot.payload)
    if marker.get("schema") != HVAWT_MODAL_REVIEW_SCHEMA or marker.get("status") != "ready_for_clean_checkout_read_only_review":
        raise ValueError("The H-VAWT modal review marker is not the accepted v2 review state.")
    if marker.get("qblade_executed") is not False or marker.get("damping_candidate_selected") is not False:
        raise ValueError("The tracked marker changed its QBlade/damping evidence boundary.")
    tracked = marker.get("tracked_evidence")
    if not isinstance(tracked, Mapping):
        raise ValueError("The modal review marker has no tracked_evidence map.")

    def marker_digest(path: Path) -> str:
        candidates = [path.as_posix()]
        parts = path.parts
        if "run_headless" in parts:
            candidates.append(Path(*parts[parts.index("run_headless") :]).as_posix())
        candidates.append(path.name)
        for candidate in candidates:
            if candidate in tracked:
                return str(tracked[candidate])
        basename_matches = [
            str(digest)
            for tracked_path, digest in tracked.items()
            if Path(str(tracked_path)).name == path.name
        ]
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            raise ValueError(
                f"The review marker ambiguously binds multiple artifacts named {path.name}."
            )
        raise ValueError(f"The review marker does not bind {path.name}.")

    identification_snapshot = _read_stable_regular_file(
        identification_file,
        label="mode identification",
    )
    shapes_snapshot = _read_stable_regular_file(
        shapes_file,
        label="mode shapes",
    )
    duals_snapshot = _read_stable_regular_file(
        duals_file,
        label="mode duals",
    )
    _require_snapshot_identity(
        identification_snapshot,
        expected_sha256=marker_digest(identification_file),
        expected_size=HVAWT_MODAL_IDENTIFICATION_SIZE_BYTES,
        label="mode identification",
    )
    _require_snapshot_identity(
        shapes_snapshot,
        expected_sha256=marker_digest(shapes_file),
        label="mode shapes",
    )
    _require_snapshot_identity(
        duals_snapshot,
        expected_sha256=marker_digest(duals_file),
        label="mode duals",
    )

    identification = json.loads(identification_snapshot.payload)
    if identification.get("schema") != HVAWT_MODAL_BASIS_SCHEMA:
        raise ValueError("Unsupported H-VAWT modal identification schema.")
    if identification.get("status") != "undamped_modal_evidence_identified_damping_candidates_unresolved":
        raise ValueError("Unexpected H-VAWT modal evidence status.")
    if identification.get("selected_modes") != list(_TARGET_MODES):
        raise ValueError("The identification does not select exactly modes 6 and 12.")
    selected_rpm = _finite_scalar(identification.get("selected_rpm"), "selected_rpm")
    if not math.isclose(selected_rpm, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The identification must select the 1 RPM basis.")
    shape_identity = identification.get("external_action_mapping", {}).get("action_shape_artifact", {})
    dual_identity = identification.get("minimum_norm_force_duals", {}).get("artifact", {})
    _require_snapshot_identity(
        shapes_snapshot,
        expected_sha256=str(shape_identity.get("sha256", "")),
        expected_size=int(shape_identity.get("size_bytes", 0)),
        label="embedded mode shapes",
    )
    _require_snapshot_identity(
        duals_snapshot,
        expected_sha256=str(dual_identity.get("sha256", "")),
        expected_size=int(dual_identity.get("size_bytes", 0)),
        label="embedded mode duals",
    )
    if identification.get("external_action_mapping", {}).get("site_count") != _SITE_COUNT:
        raise ValueError("The modal identification does not cover 39 action sites.")

    svd_rows = identification.get("minimum_norm_force_duals", {}).get("svd_evidence_by_rpm", [])
    selected_svd = [
        row for row in svd_rows if math.isclose(float(row.get("rpm", math.nan)), selected_rpm, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if len(selected_svd) != 1:
        raise ValueError("The identification must contain one 1 RPM SVD gate row.")
    svd = selected_svd[0]
    if (
        svd.get("passed") is not True
        or int(svd.get("rank", 0)) != 12
        or float(svd.get("condition_number", math.inf)) > 1.0e8
        or float(svd.get("maximum_biorthogonality_residual", math.inf)) > 1.0e-8
        or float(svd.get("maximum_target_off_mode_generalized_force", math.inf)) > 0.01
    ):
        raise ValueError("The 1 RPM complex dual fails its frozen SVD gates.")
    real_gate_rows = {
        int(row["tracked_mode"]): row
        for row in svd.get("real_reduction_gate_by_mode", [])
    }
    for mode in _TARGET_MODES:
        gate = real_gate_rows.get(mode, {})
        if (
            gate.get("passed") is not True
            or float(gate.get("maximum_imaginary_over_real_l2", math.nan)) != 0.02
            or float(gate.get("maximum_weighted_phase_rms_rad", math.nan)) != 0.02
            or float(gate.get("maximum_material_component_phase_residual_rad", math.nan)) != 0.05
        ):
            raise ValueError(f"Mode {mode} does not retain the frozen real-shape gate.")

    with io.StringIO(shapes_snapshot.payload.decode("utf-8"), newline="") as stream:
        shape_rows = list(csv.DictReader(stream))
    selected_shape_rows: dict[int, list[dict[str, str]]] = {}
    for mode in _TARGET_MODES:
        rows = [
            row
            for row in shape_rows
            if int(row["tracked_mode"]) == mode
            and math.isclose(float(row["rpm"]), selected_rpm, rel_tol=0.0, abs_tol=1.0e-12)
        ]
        if len(rows) != _SITE_COUNT:
            raise ValueError(f"Mode {mode} does not contain 39 shape rows at 1 RPM.")
        selected_shape_rows[mode] = rows
    node_ids = tuple(row["action_node_id"] for row in selected_shape_rows[6])
    if len(set(node_ids)) != _SITE_COUNT:
        raise ValueError("The tracked modal shapes contain duplicate action node IDs.")
    for mode in _TARGET_MODES:
        rows = selected_shape_rows[mode]
        if tuple(row["action_node_id"] for row in rows) != node_ids:
            raise ValueError("Tracked shape node order differs between modes.")
        output_indices = sorted(
            int(row[f"qblade_force_output_index_{axis}"])
            for row in rows
            for axis in ("x", "y", "z")
        )
        if output_indices != list(range(120, 237)):
            raise ValueError("Tracked shape force indices are not exactly 120:236.")

    shape = np.empty((_SITE_COUNT, 3, 2), dtype=complex)
    for mode_index, mode in enumerate(_TARGET_MODES):
        for site_index, row in enumerate(selected_shape_rows[mode]):
            for axis_index, axis in enumerate("xyz"):
                shape[site_index, axis_index, mode_index] = complex(
                    _finite_scalar(row[f"u{axis}_per_m_real"], f"mode {mode} u{axis} real"),
                    _finite_scalar(row[f"u{axis}_per_m_imag"], f"mode {mode} u{axis} imag"),
                )

    with io.StringIO(duals_snapshot.payload.decode("utf-8"), newline="") as stream:
        dual_rows = list(csv.DictReader(stream))
    dual = np.empty_like(shape)
    for mode_index, mode in enumerate(_TARGET_MODES):
        rows = [
            row
            for row in dual_rows
            if int(row["target_mode"]) == mode
            and math.isclose(float(row["rpm"]), selected_rpm, rel_tol=0.0, abs_tol=1.0e-12)
        ]
        if len(rows) != _SITE_COUNT or tuple(row["action_node_id"] for row in rows) != node_ids:
            raise ValueError(f"Mode {mode} dual rows do not match the tracked shape layout.")
        for site_index, row in enumerate(rows):
            for axis_index, axis in enumerate("xyz"):
                dual[site_index, axis_index, mode_index] = complex(
                    _finite_scalar(
                        row[f"force_{axis}_n_per_generalized_force_real"],
                        f"mode {mode} force {axis} real",
                    ),
                    _finite_scalar(
                        row[f"force_{axis}_n_per_generalized_force_imag"],
                        f"mode {mode} force {axis} imag",
                    ),
                )

    mode_summaries = {int(row["tracked_mode"]): row for row in identification.get("modes", [])}
    frequencies = tuple(
        _finite_scalar(mode_summaries[mode]["frequency_hz"], f"mode {mode} frequency")
        for mode in _TARGET_MODES
    )
    masses = tuple(
        _finite_scalar(
            mode_summaries[mode]["generalized_modal_mass_kg"],
            f"mode {mode} modal mass",
        )
        for mode in _TARGET_MODES
    )
    stiffnesses = tuple(
        _finite_scalar(
            mode_summaries[mode]["generalized_modal_stiffness_n_per_m"],
            f"mode {mode} modal stiffness",
        )
        for mode in _TARGET_MODES
    )
    return HVAWTModalBasis(
        node_ids=node_ids,
        tracked_modes=_TARGET_MODES,
        basis_rpm=selected_rpm,
        complex_shape_body_per_m=shape,
        complex_dual_body=dual,
        natural_frequency_hz=frequencies,
        generalized_modal_mass_kg=masses,
        generalized_modal_stiffness_npm=stiffnesses,
        identification_sha256=identification_snapshot.sha256,
        shapes_sha256=shapes_snapshot.sha256,
        duals_sha256=duals_snapshot.sha256,
        review_marker_sha256=marker_snapshot.sha256,
    )


@dataclass(frozen=True)
class HVAWTModeResponseConfig:
    """One tracked mode's bounded incremental negative-damping parameters."""

    tracked_mode: int
    natural_frequency_hz: float
    generalized_modal_mass_kg: float
    generalized_modal_stiffness_npm: float
    reference_length_m: float
    reduced_velocity_low: float
    reduced_velocity_peak: float
    reduced_velocity_high: float
    aerodynamic_damping_gain_h1: float
    saturation_amplitude_ratio: float
    rms_time_constant_s: float
    bandpass_damping_ratio: float
    max_abs_gain: float
    coefficient_source_sha256: str
    coefficient_role: str = INCREMENTAL_VIV_SCREENING

    def __post_init__(self) -> None:
        if (
            isinstance(self.tracked_mode, (bool, np.bool_))
            or not isinstance(self.tracked_mode, numbers.Integral)
            or int(self.tracked_mode) not in _TARGET_MODES
        ):
            raise ValueError("tracked_mode must be 6 or 12.")
        object.__setattr__(self, "tracked_mode", int(self.tracked_mode))
        positive = {
            "natural_frequency_hz": self.natural_frequency_hz,
            "generalized_modal_mass_kg": self.generalized_modal_mass_kg,
            "generalized_modal_stiffness_npm": self.generalized_modal_stiffness_npm,
            "reference_length_m": self.reference_length_m,
            "saturation_amplitude_ratio": self.saturation_amplitude_ratio,
            "rms_time_constant_s": self.rms_time_constant_s,
            "bandpass_damping_ratio": self.bandpass_damping_ratio,
            "max_abs_gain": self.max_abs_gain,
        }
        normalized: dict[str, float] = {}
        for name, raw_value in positive.items():
            value = _strict_finite_scalar(raw_value, name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
            normalized[name] = value
        if not 0.0 < float(self.bandpass_damping_ratio) < 1.0:
            raise ValueError("bandpass_damping_ratio must be in (0, 1).")
        h1 = _strict_finite_scalar(
            self.aerodynamic_damping_gain_h1,
            "aerodynamic_damping_gain_h1",
        )
        if h1 < 0.0:
            raise ValueError("aerodynamic_damping_gain_h1 must be non-negative.")
        low = _strict_finite_scalar(self.reduced_velocity_low, "reduced_velocity_low")
        peak = _strict_finite_scalar(self.reduced_velocity_peak, "reduced_velocity_peak")
        high = _strict_finite_scalar(self.reduced_velocity_high, "reduced_velocity_high")
        if low < 0.0 or not low < peak < high:
            raise ValueError("Reduced-velocity bounds must satisfy 0 <= low < peak < high.")
        expected_stiffness = float(self.generalized_modal_mass_kg) * (
            2.0 * math.pi * float(self.natural_frequency_hz)
        ) ** 2
        if not math.isclose(
            float(self.generalized_modal_stiffness_npm),
            expected_stiffness,
            rel_tol=2.0e-3,
            abs_tol=0.0,
        ):
            raise ValueError("Mode mass/stiffness/frequency identity failed.")
        if self.coefficient_role != INCREMENTAL_VIV_SCREENING:
            raise ValueError("H-VAWT cylinder coefficients are permitted only as incremental_viv_screening.")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "aerodynamic_damping_gain_h1", h1)
        object.__setattr__(self, "reduced_velocity_low", low)
        object.__setattr__(self, "reduced_velocity_peak", peak)
        object.__setattr__(self, "reduced_velocity_high", high)
        object.__setattr__(
            self,
            "coefficient_source_sha256",
            _hex_digest(self.coefficient_source_sha256, "coefficient_source_sha256"),
        )


@dataclass(frozen=True)
class HVAWTModeForceDiagnostics:
    tracked_mode: int
    raw_complex_projection_real_mps: float
    raw_complex_projection_imag_mps: float
    complex_projection_leakage_ratio: float
    corrected_raw_qdot_mps: float
    filtered_qdot_mps: float
    qdot_rms_mps: float
    amplitude_estimate_m: float
    amplitude_ratio: float
    reduced_velocity_min: float
    reduced_velocity_max: float
    activation_min: float
    activation_max: float
    saturation_factor: float
    raw_gain_min: float
    raw_gain_max: float
    applied_gain_min: float
    applied_gain_max: float
    maximum_mode_nodal_force_n: float
    flexible_power_w: float
    rigid_body_power_w: float


@dataclass(frozen=True)
class HVAWTResponseDiagnostics:
    schema_version: int
    config_fingerprint: str
    modal_basis_fingerprint: str
    force_source_mode: str
    coefficient_role: str
    base_flow_basis: str
    response_flow_basis: str
    feedback_velocity_basis: str
    time_s: float
    dt_s: float
    lss_rpm: float
    angular_speed_radps: float
    azimuth_deg: float
    maximum_total_speed_mps: float
    maximum_rigid_body_speed_mps: float
    maximum_flexible_speed_mps: float
    base_crossflow_speed_min_mps: float
    base_crossflow_speed_max_mps: float
    response_crossflow_speed_min_mps: float
    response_crossflow_speed_max_mps: float
    response_section_width_min_m: float
    response_section_width_max_m: float
    mode_diagnostics: Tuple[HVAWTModeForceDiagnostics, ...]
    flexible_power_w: float
    rigid_body_power_w: float
    total_structural_power_w: float
    maximum_abs_nodal_force_n: float


@dataclass(frozen=True)
class HVAWTResponseResult:
    additive_force_global_n: np.ndarray
    rigid_body_velocity_global_mps: np.ndarray
    flexible_velocity_global_mps: np.ndarray
    base_relative_velocity_global_mps: np.ndarray
    response_relative_velocity_global_mps: np.ndarray
    diagnostics: HVAWTResponseDiagnostics


@dataclass(frozen=True)
class HVAWTResponseCoreConfig:
    """Complete immutable response-core configuration."""

    kinematics: FixedAxisKinematicsConfig
    modal_basis: HVAWTModalBasis
    modes: Tuple[HVAWTModeResponseConfig, ...]
    force_sources: HVAWTForceSourceContract
    max_abs_total_nodal_force_n: float | None
    limit_action: str = "error"
    schema_version: int = HVAWT_RESPONSE_CORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, (bool, np.bool_))
            or not isinstance(self.schema_version, numbers.Integral)
            or int(self.schema_version) != HVAWT_RESPONSE_CORE_SCHEMA_VERSION
        ):
            raise ValueError("HVAWT response-core schema_version must be 1.")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if not isinstance(self.kinematics, FixedAxisKinematicsConfig):
            raise TypeError("kinematics must be FixedAxisKinematicsConfig.")
        if not isinstance(self.modal_basis, HVAWTModalBasis):
            raise TypeError("modal_basis must be HVAWTModalBasis.")
        if not isinstance(self.force_sources, HVAWTForceSourceContract):
            raise TypeError("force_sources must be HVAWTForceSourceContract.")
        if not self.force_sources.response_closure_enabled:
            raise ValueError("HVAWTResponseCore requires external_viv_mode='response_closure'.")
        if self.kinematics.node_ids != self.modal_basis.node_ids:
            raise ValueError("Kinematic and modal-basis node IDs/order must match exactly.")
        if not math.isclose(
            self.kinematics.basis_rpm,
            self.modal_basis.basis_rpm,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Kinematic and modal basis RPM differ.")
        modes = tuple(self.modes)
        object.__setattr__(self, "modes", modes)
        if any(not isinstance(mode, HVAWTModeResponseConfig) for mode in modes):
            raise TypeError("modes must contain HVAWTModeResponseConfig objects.")
        if tuple(mode.tracked_mode for mode in modes) != _TARGET_MODES:
            raise ValueError("modes must contain tracked mode 6 then mode 12 exactly once.")
        for index, mode in enumerate(modes):
            expected = (
                self.modal_basis.natural_frequency_hz[index],
                self.modal_basis.generalized_modal_mass_kg[index],
                self.modal_basis.generalized_modal_stiffness_npm[index],
            )
            actual = (
                mode.natural_frequency_hz,
                mode.generalized_modal_mass_kg,
                mode.generalized_modal_stiffness_npm,
            )
            if any(
                not math.isclose(float(a), float(e), rel_tol=0.0, abs_tol=1.0e-12)
                for a, e in zip(actual, expected)
            ):
                raise ValueError(f"Mode {mode.tracked_mode} properties differ from the modal artifact.")
        if self.max_abs_total_nodal_force_n is None:
            if self.limit_action not in {"error", "none"}:
                raise ValueError(
                    "A response core without a force ceiling requires "
                    "limit_action='error' or 'none'."
                )
            object.__setattr__(self, "limit_action", "none")
        else:
            limit = _strict_finite_scalar(
                self.max_abs_total_nodal_force_n,
                "max_abs_total_nodal_force_n",
            )
            if limit <= 0.0:
                raise ValueError("max_abs_total_nodal_force_n must be positive.")
            if self.limit_action != "error":
                raise ValueError(
                    "A finite response limit requires limit_action='error'."
                )
            object.__setattr__(self, "max_abs_total_nodal_force_n", limit)

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "kinematics": asdict(self.kinematics),
            "modal_basis_fingerprint": self.modal_basis.fingerprint,
            "modes": [asdict(mode) for mode in self.modes],
            "force_sources": asdict(self.force_sources),
            "max_abs_total_nodal_force_n": (
                None
                if self.max_abs_total_nodal_force_n is None
                else float(self.max_abs_total_nodal_force_n)
            ),
            "limit_action": self.limit_action,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _BandpassState:
    position_state: float = 0.0
    rate_state: float = 0.0
    previous_input_mps: float | None = None


def _advance_bandpass(
    state: _BandpassState,
    *,
    raw_qdot_mps: float,
    dt_s: float,
    natural_frequency_hz: float,
    damping_ratio: float,
    transition_cache: dict[tuple[float, float, float], np.ndarray],
) -> tuple[_BandpassState, float]:
    omega = 2.0 * math.pi * float(natural_frequency_hz)
    damping = float(damping_ratio)
    raw = float(raw_qdot_mps)
    if state.previous_input_mps is None:
        return _BandpassState(raw / omega**2, 0.0, raw), 0.0
    if dt_s <= 0.0:
        raise RuntimeError("Bandpass advancement requires a positive accepted time step.")
    key = (float(dt_s), omega, damping)
    transition = transition_cache.get(key)
    if transition is None:
        dynamics = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-omega**2, -2.0 * damping * omega, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        transition = expm(dynamics * dt_s)
        if len(transition_cache) >= 32:
            transition_cache.pop(next(iter(transition_cache)))
        transition_cache[key] = transition
    previous = float(state.previous_input_mps)
    slope = (raw - previous) / dt_s
    advanced = transition @ np.array(
        [state.position_state, state.rate_state, previous, slope],
        dtype=float,
    )
    new_state = _BandpassState(float(advanced[0]), float(advanced[1]), raw)
    return new_state, 2.0 * damping * omega * new_state.rate_state


def calculate_hvawt_response_force(
    *,
    modal_basis: HVAWTModalBasis,
    mode_configs: Sequence[HVAWTModeResponseConfig],
    rotation_body_to_global: Any,
    filtered_qdot_mps: Any,
    qdot_rms_mps: Any,
    response_effective_crossflow_speed_mps: Any,
    response_section_width_m: Any,
    local_length_m: Any,
    fluid_density_kgpm3: float,
    flexible_velocity_global_mps: Any,
    rigid_body_velocity_global_mps: Any,
    spectral_gain_by_node_mode: Any | None = None,
) -> tuple[np.ndarray, Tuple[dict[str, Any], ...]]:
    """Return the real three-dimensional two-mode force and raw diagnostics.

    ``spectral_gain_by_node_mode`` activates the hybrid airfoil-data path.  Its
    entries are captured lift-spectrum amplitudes normalized by the frozen
    cylinder lift amplitude.  When omitted, the schema-v1 reduced-velocity
    activation is retained byte-for-byte.  The two activation mechanisms are
    mutually exclusive rather than multiplicative: the spectral partition
    already embeds the cylinder frequency window and the measured airfoil
    energy.
    """

    if not isinstance(modal_basis, HVAWTModalBasis):
        raise TypeError("modal_basis must be HVAWTModalBasis.")
    modes = tuple(mode_configs)
    if any(not isinstance(mode, HVAWTModeResponseConfig) for mode in modes):
        raise TypeError("mode_configs must contain HVAWTModeResponseConfig objects.")
    if tuple(mode.tracked_mode for mode in modes) != _TARGET_MODES:
        raise ValueError("mode_configs must contain mode 6 then mode 12.")
    for index, mode in enumerate(modes):
        expected = (
            modal_basis.natural_frequency_hz[index],
            modal_basis.generalized_modal_mass_kg[index],
            modal_basis.generalized_modal_stiffness_npm[index],
        )
        actual = (
            mode.natural_frequency_hz,
            mode.generalized_modal_mass_kg,
            mode.generalized_modal_stiffness_npm,
        )
        if any(
            not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1.0e-12)
            for actual_value, expected_value in zip(actual, expected)
        ):
            raise ValueError(
                f"Mode {mode.tracked_mode} properties differ from the modal artifact."
            )
    rotation = np.asarray(rotation_body_to_global, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=2.0e-12,
    ):
        raise ValueError("rotation_body_to_global must be a proper orthogonal 3x3 matrix.")
    if float(np.linalg.det(rotation)) <= 0.0:
        raise ValueError("rotation_body_to_global must not reflect the modal basis.")
    filtered = _finite_exact_scalars(filtered_qdot_mps, 2, "filtered_qdot_mps")
    rms = _finite_exact_scalars(qdot_rms_mps, 2, "qdot_rms_mps")
    if np.any(rms < 0.0):
        raise ValueError("qdot_rms_mps must be non-negative.")
    speed = _finite_node_scalars(
        response_effective_crossflow_speed_mps,
        _SITE_COUNT,
        "response_effective_crossflow_speed_mps",
    )
    width = _finite_node_scalars(
        response_section_width_m,
        _SITE_COUNT,
        "response_section_width_m",
    )
    length = _finite_node_scalars(local_length_m, _SITE_COUNT, "local_length_m")
    flexible = _finite_node_vectors(
        flexible_velocity_global_mps,
        _SITE_COUNT,
        "flexible_velocity_global_mps",
    )
    rigid = _finite_node_vectors(
        rigid_body_velocity_global_mps,
        _SITE_COUNT,
        "rigid_body_velocity_global_mps",
    )
    density = _strict_finite_scalar(fluid_density_kgpm3, "fluid_density_kgpm3")
    if density <= 0.0 or np.any(speed < 0.0) or np.any(width <= 0.0) or np.any(length < 0.0):
        raise ValueError("Density/width must be positive and speed/length non-negative.")
    spectral_gain = None
    if spectral_gain_by_node_mode is not None:
        spectral_gain = np.asarray(spectral_gain_by_node_mode, dtype=float)
        if spectral_gain.shape != (_SITE_COUNT, len(_TARGET_MODES)):
            raise ValueError(
                "spectral_gain_by_node_mode must have shape (39, 2)."
            )
        if np.any(~np.isfinite(spectral_gain)) or np.any(spectral_gain < 0.0):
            raise ValueError(
                "spectral_gain_by_node_mode must be finite and non-negative."
            )

    real_shape_body = modal_basis.real_shape_body_per_m
    real_shape_global = np.einsum("ab,ibm->iam", rotation, real_shape_body)
    alpha = modal_basis.real_shape_maximum_site_norm
    force_total = np.zeros((_SITE_COUNT, 3), dtype=float)
    diagnostics: list[dict[str, Any]] = []
    prefactor = 0.5 * density * speed * width * length
    for mode_index, mode in enumerate(modes):
        omega = 2.0 * math.pi * mode.natural_frequency_hz
        amplitude = float(alpha[mode_index]) * math.sqrt(2.0) * float(rms[mode_index]) / omega
        amplitude_ratio = amplitude / mode.reference_length_m
        saturation = 1.0 - (
            amplitude_ratio / mode.saturation_amplitude_ratio
        ) ** 2
        reduced_velocity = speed / (
            mode.natural_frequency_hz * mode.reference_length_m
        )
        if spectral_gain is None:
            activation = _smooth_activation(
                reduced_velocity,
                mode.reduced_velocity_low,
                mode.reduced_velocity_peak,
                mode.reduced_velocity_high,
            )
            activation_basis = "fixed_reduced_velocity"
        else:
            activation = spectral_gain[:, mode_index]
            activation_basis = "captured_airfoil_lift_spectrum"
        raw_gain = mode.aerodynamic_damping_gain_h1 * activation * saturation
        if np.any(np.abs(raw_gain) > mode.max_abs_gain):
            raise ValueError(
                f"Mode {mode.tracked_mode} response gain exceeds max_abs_gain."
            )
        applied_gain = np.array(raw_gain, copy=True)
        modal_velocity = real_shape_global[:, :, mode_index] * filtered[mode_index]
        mode_force = (
            prefactor * applied_gain
        )[:, None] * modal_velocity
        if np.any(~np.isfinite(mode_force)):
            raise FloatingPointError(f"Mode {mode.tracked_mode} response force became non-finite.")
        force_total += mode_force
        diagnostics.append(
            {
                "tracked_mode": mode.tracked_mode,
                "amplitude_estimate_m": amplitude,
                "amplitude_ratio": amplitude_ratio,
                "reduced_velocity_min": float(np.min(reduced_velocity)),
                "reduced_velocity_max": float(np.max(reduced_velocity)),
                "activation_min": float(np.min(activation)),
                "activation_max": float(np.max(activation)),
                "activation_basis": activation_basis,
                "saturation_factor": saturation,
                "raw_gain_min": float(np.min(raw_gain)),
                "raw_gain_max": float(np.max(raw_gain)),
                "applied_gain_min": float(np.min(applied_gain)),
                "applied_gain_max": float(np.max(applied_gain)),
                "maximum_mode_nodal_force_n": float(
                    np.max(np.linalg.norm(mode_force, axis=1))
                ),
                "flexible_power_w": float(np.sum(mode_force * flexible)),
                "rigid_body_power_w": float(np.sum(mode_force * rigid)),
            }
        )
    return _copy_readonly(force_total), tuple(diagnostics)


def _response_result_copy(result: HVAWTResponseResult) -> HVAWTResponseResult:
    return HVAWTResponseResult(
        additive_force_global_n=_copy_readonly(result.additive_force_global_n),
        rigid_body_velocity_global_mps=_copy_readonly(
            result.rigid_body_velocity_global_mps
        ),
        flexible_velocity_global_mps=_copy_readonly(
            result.flexible_velocity_global_mps
        ),
        base_relative_velocity_global_mps=_copy_readonly(
            result.base_relative_velocity_global_mps
        ),
        response_relative_velocity_global_mps=_copy_readonly(
            result.response_relative_velocity_global_mps
        ),
        diagnostics=result.diagnostics,
    )


@dataclass(frozen=True)
class _BoundPrecomputedCoreState:
    """Core-owned binding between typed fast-path state and raw input objects."""

    owner: object
    node_ids: tuple[str, ...]
    raw_inputs: tuple[object, ...]
    lss_rpm: float
    azimuth_deg: float
    kinematics: FixedAxisKinematicState
    flows: HVAWTSeparatedFlowState


class HVAWTResponseCore:
    """Stateful two-mode fixed-axis response closure.

    Calls are transactional and idempotent at an exact accepted time.  A
    changed input at the same time, a decreasing time, a force limit, or a
    kinematic inconsistency leaves every accepted filter/RMS/cache state
    unchanged.
    """

    def __init__(self, config: HVAWTResponseCoreConfig):
        if not isinstance(config, HVAWTResponseCoreConfig):
            raise TypeError("config must be HVAWTResponseCoreConfig.")
        self.config = config
        self._filter_states = (_BandpassState(), _BandpassState())
        self._qdot_mean_square_m2ps2 = np.zeros(2, dtype=float)
        self._transition_cache: dict[tuple[float, float, float], np.ndarray] = {}
        self._last_time_s: float | None = None
        self._last_azimuth_deg: float | None = None
        self._last_signed_rpm: float | None = None
        self._last_input_signature: str | None = None
        self._last_result_canonical: HVAWTResponseResult | None = None
        self._precomputed_state_owner = object()

    @property
    def last_diagnostics(self) -> HVAWTResponseDiagnostics | None:
        if self._last_result_canonical is None:
            return None
        return self._last_result_canonical.diagnostics

    def _bind_precomputed_step_state(
        self,
        *,
        node_ids: Sequence[str],
        positions_global_m: Any,
        total_structural_velocity_global_mps: Any,
        lss_rpm: float,
        azimuth_deg: float,
        ambient_velocity_global_mps: Any,
        chord_unit_global: Any,
        normal_unit_global: Any,
        chord_m: Any,
        thickness_ratio: Any,
        kinematics: FixedAxisKinematicState,
        flows: HVAWTSeparatedFlowState,
    ) -> _BoundPrecomputedCoreState:
        """Bind adapter-prepared state to this core and these exact raw objects.

        This private fast path is intentionally object-bound: a state prepared
        for a prior wrapper sample or another core cannot be replayed against
        merely same-shaped inputs.  The ordinary step signature still hashes
        current values for duplicate-time identity.
        """

        if tuple(node_ids) != self.config.modal_basis.node_ids:
            raise ValueError("Precomputed H-VAWT state requires canonical node order.")
        if not isinstance(kinematics, FixedAxisKinematicState) or not isinstance(
            flows, HVAWTSeparatedFlowState
        ):
            raise TypeError("Precomputed H-VAWT state must be typed.")
        return _BoundPrecomputedCoreState(
            owner=self._precomputed_state_owner,
            node_ids=tuple(node_ids),
            raw_inputs=(
                positions_global_m,
                total_structural_velocity_global_mps,
                ambient_velocity_global_mps,
                chord_unit_global,
                normal_unit_global,
                chord_m,
                thickness_ratio,
            ),
            lss_rpm=float(lss_rpm),
            azimuth_deg=float(azimuth_deg),
            kinematics=kinematics,
            flows=flows,
        )

    @staticmethod
    def _canonical_input_signature(
        *,
        time_s: float,
        lss_rpm: float,
        azimuth_deg: float,
        positions_global_m: np.ndarray,
        total_velocity_global_mps: np.ndarray,
        ambient_velocity_global_mps: np.ndarray,
        chord_unit_global: np.ndarray,
        normal_unit_global: np.ndarray,
        chord_m: np.ndarray,
        thickness_ratio: np.ndarray,
        local_length_m: np.ndarray,
        fluid_density_kgpm3: float,
        spectral_gain_by_node_mode: np.ndarray | None,
        limit_companion_force_global_n: np.ndarray | None,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"vorlap-hvawt-response-input-v1\0")
        digest.update(
            np.asarray(
                [time_s, lss_rpm, azimuth_deg, fluid_density_kgpm3],
                dtype="<f8",
            ).tobytes()
        )
        for array in (
            positions_global_m,
            total_velocity_global_mps,
            ambient_velocity_global_mps,
            chord_unit_global,
            normal_unit_global,
            chord_m,
            thickness_ratio,
            local_length_m,
        ):
            digest.update(_canonical_array_bytes(array))
        if spectral_gain_by_node_mode is None:
            digest.update(b"fixed-reduced-velocity-activation\0")
        else:
            digest.update(b"airfoil-spectral-lockin-activation\0")
            digest.update(_canonical_array_bytes(spectral_gain_by_node_mode))
        if limit_companion_force_global_n is None:
            digest.update(b"no-limit-companion-force\0")
        else:
            digest.update(b"limit-companion-force\0")
            digest.update(_canonical_array_bytes(limit_companion_force_global_n))
        return digest.hexdigest()

    def _check_azimuth_rate(
        self,
        *,
        time_s: float,
        azimuth_deg: float,
        signed_rpm: float,
    ) -> None:
        if self._last_time_s is None:
            return
        dt_s = time_s - self._last_time_s
        if dt_s <= 0.0:
            raise RuntimeError("Azimuth-rate validation requires positive dt.")
        if self._last_azimuth_deg is None or self._last_signed_rpm is None:
            raise RuntimeError("Accepted kinematic history is incomplete.")
        raw_delta = math.radians(
            self.config.kinematics.azimuth_sign
            * (azimuth_deg - self._last_azimuth_deg)
        )
        observed_delta = math.remainder(raw_delta, 2.0 * math.pi)
        expected_delta = (
            0.5
            * (signed_rpm + self._last_signed_rpm)
            * _RPM_TO_RADPS
            * dt_s
        )
        if abs(expected_delta) >= math.pi:
            raise ValueError("Time step is too large to unwrap shaft azimuth uniquely.")
        rate_error = abs(observed_delta - expected_delta) / dt_s
        if rate_error > self.config.kinematics.maximum_azimuth_rate_error_radps:
            raise ValueError(
                f"Azimuth/LSS-RPM rate mismatch {rate_error:.17g} rad/s exceeds "
                f"{self.config.kinematics.maximum_azimuth_rate_error_radps:.17g} rad/s."
            )

    def _canonicalize_step_inputs(
        self,
        *,
        node_ids: Sequence[str],
        positions_global_m: Any,
        total_structural_velocity_global_mps: Any,
        ambient_velocity_global_mps: Any,
        chord_unit_global: Any,
        normal_unit_global: Any,
        chord_m: Any,
        thickness_ratio: Any,
        local_length_m: Any,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
            raise TypeError("node_ids must contain nonempty strings.")
        step_ids = tuple(node_ids)
        expected = self.config.modal_basis.node_ids
        if len(step_ids) != _SITE_COUNT or len(set(step_ids)) != _SITE_COUNT or set(step_ids) != set(expected):
            raise ValueError("node_ids must contain each configured H-VAWT action site exactly once.")
        canonical_order = step_ids == expected
        if canonical_order:
            canonical_from_step = np.arange(_SITE_COUNT, dtype=int)
            step_from_canonical = canonical_from_step
        else:
            step_index = {node_id: index for index, node_id in enumerate(step_ids)}
            canonical_from_step = np.asarray(
                [step_index[node_id] for node_id in expected], dtype=int
            )
            step_from_canonical = np.empty(_SITE_COUNT, dtype=int)
            step_from_canonical[canonical_from_step] = np.arange(
                _SITE_COUNT, dtype=int
            )

        def vectors(value: Any, name: str) -> np.ndarray:
            if not canonical_order:
                return _finite_node_vectors(value, _SITE_COUNT, name)[
                    canonical_from_step
                ]
            array = np.asarray(value, dtype=float)
            if array.shape != (_SITE_COUNT, 3) or np.any(~np.isfinite(array)):
                raise ValueError(
                    f"{name} must have shape ({_SITE_COUNT}, 3) with finite values."
                )
            return array

        def scalars(value: Any, name: str) -> np.ndarray:
            if not canonical_order:
                return _finite_node_scalars(value, _SITE_COUNT, name)[
                    canonical_from_step
                ]
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                array = np.full(_SITE_COUNT, float(array), dtype=float)
            if array.shape != (_SITE_COUNT,) or np.any(~np.isfinite(array)):
                raise ValueError(
                    f"{name} must be scalar or a finite ({_SITE_COUNT},) array."
                )
            return array

        ambient_raw = np.asarray(ambient_velocity_global_mps, dtype=float)
        if ambient_raw.shape == (3,):
            ambient_raw = np.broadcast_to(ambient_raw, (_SITE_COUNT, 3))
        canonical = {
            "positions": vectors(positions_global_m, "positions_global_m"),
            "total_velocity": vectors(
                total_structural_velocity_global_mps,
                "total_structural_velocity_global_mps",
            ),
            "ambient": vectors(ambient_raw, "ambient_velocity_global_mps"),
            "chord_unit": vectors(chord_unit_global, "chord_unit_global"),
            "normal_unit": vectors(normal_unit_global, "normal_unit_global"),
            "chord_m": scalars(chord_m, "chord_m"),
            "thickness_ratio": scalars(thickness_ratio, "thickness_ratio"),
            "local_length_m": scalars(local_length_m, "local_length_m"),
        }
        return canonical, step_from_canonical

    def step(
        self,
        time_s: float,
        *,
        node_ids: Sequence[str],
        positions_global_m: Any,
        total_structural_velocity_global_mps: Any,
        lss_rpm: float,
        azimuth_deg: float,
        ambient_velocity_global_mps: Any,
        chord_unit_global: Any,
        normal_unit_global: Any,
        chord_m: Any,
        thickness_ratio: Any,
        local_length_m: Any,
        fluid_density_kgpm3: float,
        spectral_gain_by_node_mode: Any | None = None,
        limit_companion_force_global_n: Any | None = None,
        _precomputed_state: _BoundPrecomputedCoreState | None = None,
    ) -> HVAWTResponseResult:
        current_time = _strict_finite_scalar(time_s, "time_s")
        rpm = _strict_finite_scalar(lss_rpm, "lss_rpm")
        azimuth = _strict_finite_scalar(azimuth_deg, "azimuth_deg")
        density = _strict_finite_scalar(fluid_density_kgpm3, "fluid_density_kgpm3")
        if density <= 0.0:
            raise ValueError("fluid_density_kgpm3 must be positive.")
        canonical_order = tuple(node_ids) == self.config.modal_basis.node_ids
        current_raw_inputs = (
            positions_global_m,
            total_structural_velocity_global_mps,
            ambient_velocity_global_mps,
            chord_unit_global,
            normal_unit_global,
            chord_m,
            thickness_ratio,
        )
        if _precomputed_state is not None and (
            not isinstance(_precomputed_state, _BoundPrecomputedCoreState)
            or _precomputed_state.owner is not self._precomputed_state_owner
            or not canonical_order
            or _precomputed_state.node_ids != tuple(node_ids)
            or _precomputed_state.lss_rpm != rpm
            or _precomputed_state.azimuth_deg != azimuth
            or any(
                prepared is not current
                for prepared, current in zip(
                    _precomputed_state.raw_inputs,
                    current_raw_inputs,
                )
            )
        ):
            raise TypeError(
                "Precomputed H-VAWT state is not bound to this core/input sample."
            )
        canonical, step_from_canonical = self._canonicalize_step_inputs(
            node_ids=node_ids,
            positions_global_m=positions_global_m,
            total_structural_velocity_global_mps=total_structural_velocity_global_mps,
            ambient_velocity_global_mps=ambient_velocity_global_mps,
            chord_unit_global=chord_unit_global,
            normal_unit_global=normal_unit_global,
            chord_m=chord_m,
            thickness_ratio=thickness_ratio,
            local_length_m=local_length_m,
        )
        canonical_spectral_gain = None
        if spectral_gain_by_node_mode is not None:
            step_spectral_gain = np.asarray(
                spectral_gain_by_node_mode,
                dtype=float,
            )
            if step_spectral_gain.shape != (_SITE_COUNT, len(_TARGET_MODES)):
                raise ValueError(
                    "spectral_gain_by_node_mode must have shape (39, 2)."
                )
            if np.any(~np.isfinite(step_spectral_gain)) or np.any(
                step_spectral_gain < 0.0
            ):
                raise ValueError(
                    "spectral_gain_by_node_mode must be finite and non-negative."
                )
            canonical_spectral_gain = (
                step_spectral_gain
                if canonical_order
                else step_spectral_gain[np.argsort(step_from_canonical)]
            )
        canonical_companion_force = None
        if limit_companion_force_global_n is not None:
            step_companion_force = _finite_node_vectors(
                limit_companion_force_global_n,
                _SITE_COUNT,
                "limit_companion_force_global_n",
            )
            canonical_companion_force = (
                step_companion_force
                if canonical_order
                else step_companion_force[np.argsort(step_from_canonical)]
            )
        signature = self._canonical_input_signature(
            time_s=current_time,
            lss_rpm=rpm,
            azimuth_deg=azimuth,
            positions_global_m=canonical["positions"],
            total_velocity_global_mps=canonical["total_velocity"],
            ambient_velocity_global_mps=canonical["ambient"],
            chord_unit_global=canonical["chord_unit"],
            normal_unit_global=canonical["normal_unit"],
            chord_m=canonical["chord_m"],
            thickness_ratio=canonical["thickness_ratio"],
            local_length_m=canonical["local_length_m"],
            fluid_density_kgpm3=density,
            spectral_gain_by_node_mode=canonical_spectral_gain,
            limit_companion_force_global_n=canonical_companion_force,
        )
        if self._last_time_s is not None:
            if current_time < self._last_time_s - _TIME_TOLERANCE_S:
                raise ValueError(
                    f"Response-core time decreased from {self._last_time_s:.17g} "
                    f"to {current_time:.17g} s."
                )
            if math.isclose(
                current_time,
                self._last_time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            ):
                if signature != self._last_input_signature:
                    raise ValueError("Response-core inputs changed at the same time level.")
                if self._last_result_canonical is None:
                    raise RuntimeError("Response-core duplicate cache is incomplete.")
                return (
                    self._last_result_canonical
                    if canonical_order
                    else self._result_in_step_order(
                        self._last_result_canonical,
                        step_from_canonical,
                    )
                )

        dt_s = 0.0 if self._last_time_s is None else current_time - self._last_time_s
        signed_rpm = self.config.kinematics.rpm_sign * rpm
        self._check_azimuth_rate(
            time_s=current_time,
            azimuth_deg=azimuth,
            signed_rpm=signed_rpm,
        )
        kinematics = (
            _precomputed_state.kinematics
            if _precomputed_state is not None
            else decompose_fixed_axis_velocity(
                self.config.kinematics,
                positions_global_m=canonical["positions"],
                total_structural_velocity_global_mps=canonical["total_velocity"],
                lss_rpm=rpm,
                azimuth_deg=azimuth,
            )
        )
        rotation = np.asarray(kinematics.rotation_body_to_global, dtype=float)
        flexible_body = np.asarray(
            kinematics.flexible_velocity_global_mps,
            dtype=float,
        ) @ rotation
        raw_qdot, raw_complex, complex_leakage = (
            self.config.modal_basis.extract_modal_velocity(flexible_body)
        )

        candidate_filter_states: list[_BandpassState] = []
        filtered_qdot = np.empty(2, dtype=float)
        candidate_transition_cache = dict(self._transition_cache)
        candidate_mean_square = np.array(
            self._qdot_mean_square_m2ps2,
            dtype=float,
            copy=True,
        )
        for index, mode in enumerate(self.config.modes):
            state, filtered = _advance_bandpass(
                self._filter_states[index],
                raw_qdot_mps=float(raw_qdot[index]),
                dt_s=dt_s,
                natural_frequency_hz=mode.natural_frequency_hz,
                damping_ratio=mode.bandpass_damping_ratio,
                transition_cache=candidate_transition_cache,
            )
            candidate_filter_states.append(state)
            filtered_qdot[index] = filtered
            if dt_s > 0.0:
                decay = math.exp(-dt_s / mode.rms_time_constant_s)
                candidate_mean_square[index] = (
                    decay * candidate_mean_square[index]
                    + (1.0 - decay) * filtered**2
                )
        qdot_rms = np.sqrt(np.maximum(candidate_mean_square, 0.0))

        flows = (
            _precomputed_state.flows
            if _precomputed_state is not None
            else separate_hvawt_flows(
                ambient_velocity_global_mps=canonical["ambient"],
                total_structural_velocity_global_mps=canonical["total_velocity"],
                rigid_body_velocity_global_mps=(
                    kinematics.rigid_body_velocity_global_mps
                ),
                chord_unit_global=canonical["chord_unit"],
                normal_unit_global=canonical["normal_unit"],
                chord_m=canonical["chord_m"],
                thickness_ratio=canonical["thickness_ratio"],
            )
        )
        force, raw_mode_diagnostics = calculate_hvawt_response_force(
            modal_basis=self.config.modal_basis,
            mode_configs=self.config.modes,
            rotation_body_to_global=rotation,
            filtered_qdot_mps=filtered_qdot,
            qdot_rms_mps=qdot_rms,
            response_effective_crossflow_speed_mps=(
                flows.response_effective_crossflow_speed_mps
            ),
            response_section_width_m=flows.response_section_width_m,
            local_length_m=canonical["local_length_m"],
            fluid_density_kgpm3=density,
            flexible_velocity_global_mps=kinematics.flexible_velocity_global_mps,
            rigid_body_velocity_global_mps=kinematics.rigid_body_velocity_global_mps,
            spectral_gain_by_node_mode=canonical_spectral_gain,
        )
        force_for_limit = (
            force
            if canonical_companion_force is None
            else force + canonical_companion_force
        )
        force_norm = np.linalg.norm(force_for_limit, axis=1)
        maximum_force_for_limit = float(np.max(force_norm))
        maximum_response_force = float(np.max(np.linalg.norm(force, axis=1)))
        if not (
            math.isfinite(maximum_force_for_limit)
            and math.isfinite(maximum_response_force)
        ):
            raise FloatingPointError(
                "H-VAWT response-plus-companion nodal forces must remain finite."
            )
        if (
            self.config.max_abs_total_nodal_force_n is not None
            and maximum_force_for_limit
            > self.config.max_abs_total_nodal_force_n
        ):
            raise ValueError(
                "H-VAWT response-plus-companion nodal force "
                f"{maximum_force_for_limit:.17g} N exceeds "
                f"{self.config.max_abs_total_nodal_force_n:.17g} N."
            )

        mode_diagnostics = tuple(
            HVAWTModeForceDiagnostics(
                tracked_mode=int(raw["tracked_mode"]),
                raw_complex_projection_real_mps=float(raw_complex[index].real),
                raw_complex_projection_imag_mps=float(raw_complex[index].imag),
                complex_projection_leakage_ratio=float(complex_leakage[index]),
                corrected_raw_qdot_mps=float(raw_qdot[index]),
                filtered_qdot_mps=float(filtered_qdot[index]),
                qdot_rms_mps=float(qdot_rms[index]),
                amplitude_estimate_m=float(raw["amplitude_estimate_m"]),
                amplitude_ratio=float(raw["amplitude_ratio"]),
                reduced_velocity_min=float(raw["reduced_velocity_min"]),
                reduced_velocity_max=float(raw["reduced_velocity_max"]),
                activation_min=float(raw["activation_min"]),
                activation_max=float(raw["activation_max"]),
                saturation_factor=float(raw["saturation_factor"]),
                raw_gain_min=float(raw["raw_gain_min"]),
                raw_gain_max=float(raw["raw_gain_max"]),
                applied_gain_min=float(raw["applied_gain_min"]),
                applied_gain_max=float(raw["applied_gain_max"]),
                maximum_mode_nodal_force_n=float(raw["maximum_mode_nodal_force_n"]),
                flexible_power_w=float(raw["flexible_power_w"]),
                rigid_body_power_w=float(raw["rigid_body_power_w"]),
            )
            for index, raw in enumerate(raw_mode_diagnostics)
        )
        flexible_power = float(
            np.sum(force * kinematics.flexible_velocity_global_mps)
        )
        rigid_power = float(
            np.sum(force * kinematics.rigid_body_velocity_global_mps)
        )
        diagnostics = HVAWTResponseDiagnostics(
            schema_version=HVAWT_RESPONSE_CORE_SCHEMA_VERSION,
            config_fingerprint=self.config.fingerprint,
            modal_basis_fingerprint=self.config.modal_basis.fingerprint,
            force_source_mode=self.config.force_sources.external_viv_mode,
            coefficient_role=self.config.force_sources.coefficient_role,
            base_flow_basis=BASE_FLOW_BASIS,
            response_flow_basis=RESPONSE_FLOW_BASIS,
            feedback_velocity_basis=FEEDBACK_VELOCITY_BASIS,
            time_s=current_time,
            dt_s=dt_s,
            lss_rpm=rpm,
            angular_speed_radps=float(np.linalg.norm(kinematics.angular_velocity_global_radps)),
            azimuth_deg=azimuth,
            maximum_total_speed_mps=kinematics.maximum_total_speed_mps,
            maximum_rigid_body_speed_mps=kinematics.maximum_rigid_body_speed_mps,
            maximum_flexible_speed_mps=kinematics.maximum_flexible_speed_mps,
            base_crossflow_speed_min_mps=float(
                np.min(flows.base_effective_crossflow_speed_mps)
            ),
            base_crossflow_speed_max_mps=float(
                np.max(flows.base_effective_crossflow_speed_mps)
            ),
            response_crossflow_speed_min_mps=float(
                np.min(flows.response_effective_crossflow_speed_mps)
            ),
            response_crossflow_speed_max_mps=float(
                np.max(flows.response_effective_crossflow_speed_mps)
            ),
            response_section_width_min_m=float(
                np.min(flows.response_section_width_m)
            ),
            response_section_width_max_m=float(
                np.max(flows.response_section_width_m)
            ),
            mode_diagnostics=mode_diagnostics,
            flexible_power_w=flexible_power,
            rigid_body_power_w=rigid_power,
            total_structural_power_w=flexible_power + rigid_power,
            maximum_abs_nodal_force_n=maximum_response_force,
        )
        canonical_result = HVAWTResponseResult(
            additive_force_global_n=_copy_readonly(force),
            rigid_body_velocity_global_mps=_copy_readonly(
                kinematics.rigid_body_velocity_global_mps
            ),
            flexible_velocity_global_mps=_copy_readonly(
                kinematics.flexible_velocity_global_mps
            ),
            base_relative_velocity_global_mps=_copy_readonly(
                flows.base_relative_velocity_global_mps
            ),
            response_relative_velocity_global_mps=_copy_readonly(
                flows.response_relative_velocity_global_mps
            ),
            diagnostics=diagnostics,
        )

        self._filter_states = tuple(candidate_filter_states)
        self._qdot_mean_square_m2ps2 = candidate_mean_square
        self._transition_cache = candidate_transition_cache
        self._last_time_s = current_time
        self._last_azimuth_deg = azimuth
        self._last_signed_rpm = signed_rpm
        self._last_input_signature = signature
        # Every canonical-result array is copied into a read-only buffer above.
        # The adapter always uses canonical node order, so its caller and the
        # duplicate-time cache can safely share this immutable result.
        self._last_result_canonical = canonical_result
        return (
            canonical_result
            if canonical_order
            else self._result_in_step_order(canonical_result, step_from_canonical)
        )

    @staticmethod
    def _result_in_step_order(
        canonical: HVAWTResponseResult,
        step_from_canonical: np.ndarray,
    ) -> HVAWTResponseResult:
        indices = np.asarray(step_from_canonical, dtype=int)
        return HVAWTResponseResult(
            additive_force_global_n=_copy_readonly(
                canonical.additive_force_global_n[indices]
            ),
            rigid_body_velocity_global_mps=_copy_readonly(
                canonical.rigid_body_velocity_global_mps[indices]
            ),
            flexible_velocity_global_mps=_copy_readonly(
                canonical.flexible_velocity_global_mps[indices]
            ),
            base_relative_velocity_global_mps=_copy_readonly(
                canonical.base_relative_velocity_global_mps[indices]
            ),
            response_relative_velocity_global_mps=_copy_readonly(
                canonical.response_relative_velocity_global_mps[indices]
            ),
            diagnostics=canonical.diagnostics,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        """Return JSON-serializable state with exact config/basis identity."""

        result = None
        if self._last_result_canonical is not None:
            result = {
                "additive_force_global_n": self._last_result_canonical.additive_force_global_n.tolist(),
                "rigid_body_velocity_global_mps": self._last_result_canonical.rigid_body_velocity_global_mps.tolist(),
                "flexible_velocity_global_mps": self._last_result_canonical.flexible_velocity_global_mps.tolist(),
                "base_relative_velocity_global_mps": self._last_result_canonical.base_relative_velocity_global_mps.tolist(),
                "response_relative_velocity_global_mps": self._last_result_canonical.response_relative_velocity_global_mps.tolist(),
                "diagnostics": asdict(self._last_result_canonical.diagnostics),
            }
        payload = {
            "schema": HVAWT_RESPONSE_CHECKPOINT_SCHEMA,
            "config_fingerprint": self.config.fingerprint,
            "modal_basis_fingerprint": self.config.modal_basis.fingerprint,
            "last_time_s": self._last_time_s,
            "last_azimuth_deg": self._last_azimuth_deg,
            "last_signed_rpm": self._last_signed_rpm,
            "last_input_signature": self._last_input_signature,
            "bandpass_states": [asdict(state) for state in self._filter_states],
            "qdot_mean_square_m2ps2": self._qdot_mean_square_m2ps2.tolist(),
            "last_result": result,
        }
        return {**payload, "state_sha256": _canonical_mapping_sha256(payload)}

    def transaction_snapshot(self) -> tuple[object, ...]:
        """Capture the accepted in-process state without serializing it.

        ``step`` constructs candidate arrays, dictionaries, and result objects
        and swaps them into the core only after every acceptance check passes.
        The previously accepted objects are therefore immutable for the
        duration of a coupled transaction and can be retained by reference.
        Persistent restart checkpoints continue to use ``checkpoint_state``.
        """

        return (
            self._filter_states,
            self._qdot_mean_square_m2ps2,
            self._transition_cache,
            self._last_time_s,
            self._last_azimuth_deg,
            self._last_signed_rpm,
            self._last_input_signature,
            self._last_result_canonical,
        )

    def restore_transaction_snapshot(self, snapshot: tuple[object, ...]) -> None:
        """Restore an opaque snapshot returned by ``transaction_snapshot``."""

        if not isinstance(snapshot, tuple) or len(snapshot) != 8:
            raise TypeError("H-VAWT response transaction snapshot is invalid.")
        (
            filter_states,
            mean_square,
            transition_cache,
            last_time,
            last_azimuth,
            last_rpm,
            last_signature,
            last_result,
        ) = snapshot
        self._filter_states = filter_states  # type: ignore[assignment]
        self._qdot_mean_square_m2ps2 = mean_square  # type: ignore[assignment]
        self._transition_cache = transition_cache  # type: ignore[assignment]
        self._last_time_s = last_time  # type: ignore[assignment]
        self._last_azimuth_deg = last_azimuth  # type: ignore[assignment]
        self._last_signed_rpm = last_rpm  # type: ignore[assignment]
        self._last_input_signature = last_signature  # type: ignore[assignment]
        self._last_result_canonical = last_result  # type: ignore[assignment]

    def _diagnostics_from_mapping(
        self,
        raw: Mapping[str, Any],
        *,
        last_time_s: float,
        last_azimuth_deg: float,
        last_signed_rpm: float,
    ) -> HVAWTResponseDiagnostics:
        diagnostic_keys = {field.name for field in fields(HVAWTResponseDiagnostics)}
        _require_exact_keys(raw, diagnostic_keys, "checkpoint diagnostics")
        values = dict(raw)
        mode_rows = values.get("mode_diagnostics")
        if (
            not isinstance(mode_rows, Sequence)
            or isinstance(mode_rows, (str, bytes))
            or len(mode_rows) != 2
        ):
            raise ValueError("Checkpoint mode_diagnostics must contain exactly two rows.")
        mode_keys = {field.name for field in fields(HVAWTModeForceDiagnostics)}
        parsed_modes: list[HVAWTModeForceDiagnostics] = []
        for index, row in enumerate(mode_rows):
            if not isinstance(row, Mapping):
                raise ValueError("Checkpoint mode diagnostic rows must be mappings.")
            _require_exact_keys(row, mode_keys, "checkpoint mode diagnostics")
            mode_values = dict(row)
            tracked_mode = mode_values.get("tracked_mode")
            if (
                isinstance(tracked_mode, (bool, np.bool_))
                or not isinstance(tracked_mode, numbers.Integral)
                or int(tracked_mode) != _TARGET_MODES[index]
            ):
                raise ValueError("Checkpoint mode diagnostics are not ordered modes 6 and 12.")
            mode_values["tracked_mode"] = int(tracked_mode)
            for name in mode_keys - {"tracked_mode"}:
                mode_values[name] = _strict_finite_scalar(
                    mode_values[name],
                    f"checkpoint mode diagnostic {name}",
                )
            parsed_modes.append(HVAWTModeForceDiagnostics(**mode_values))

        schema_version = values.get("schema_version")
        if (
            isinstance(schema_version, (bool, np.bool_))
            or not isinstance(schema_version, numbers.Integral)
            or int(schema_version) != HVAWT_RESPONSE_CORE_SCHEMA_VERSION
        ):
            raise ValueError("Checkpoint diagnostic schema_version differs.")
        values["schema_version"] = int(schema_version)
        string_values = {
            "config_fingerprint": self.config.fingerprint,
            "modal_basis_fingerprint": self.config.modal_basis.fingerprint,
            "force_source_mode": self.config.force_sources.external_viv_mode,
            "coefficient_role": self.config.force_sources.coefficient_role,
            "base_flow_basis": BASE_FLOW_BASIS,
            "response_flow_basis": RESPONSE_FLOW_BASIS,
            "feedback_velocity_basis": FEEDBACK_VELOCITY_BASIS,
        }
        for name, expected in string_values.items():
            if values.get(name) != expected:
                raise ValueError(f"Checkpoint diagnostic {name} identity differs.")
        scalar_names = diagnostic_keys - set(string_values) - {
            "schema_version",
            "mode_diagnostics",
        }
        for name in scalar_names:
            values[name] = _strict_finite_scalar(
                values[name],
                f"checkpoint diagnostic {name}",
            )
        values["mode_diagnostics"] = tuple(parsed_modes)
        diagnostics = HVAWTResponseDiagnostics(**values)

        if diagnostics.time_s != last_time_s or diagnostics.azimuth_deg != last_azimuth_deg:
            raise ValueError("Checkpoint diagnostics accepted time/azimuth differs.")
        expected_lss_rpm = self.config.kinematics.rpm_sign * last_signed_rpm
        if diagnostics.lss_rpm != expected_lss_rpm:
            raise ValueError("Checkpoint diagnostics LSS RPM differs from history.")
        expected_omega = abs(last_signed_rpm) * _RPM_TO_RADPS
        if not math.isclose(
            diagnostics.angular_speed_radps,
            expected_omega,
            rel_tol=0.0,
            abs_tol=2.0e-15,
        ):
            raise ValueError("Checkpoint diagnostic angular speed differs from LSS RPM.")
        if diagnostics.dt_s < 0.0:
            raise ValueError("Checkpoint diagnostic dt_s must be non-negative.")
        nonnegative = (
            diagnostics.maximum_total_speed_mps,
            diagnostics.maximum_rigid_body_speed_mps,
            diagnostics.maximum_flexible_speed_mps,
            diagnostics.base_crossflow_speed_min_mps,
            diagnostics.base_crossflow_speed_max_mps,
            diagnostics.response_crossflow_speed_min_mps,
            diagnostics.response_crossflow_speed_max_mps,
            diagnostics.response_section_width_min_m,
            diagnostics.response_section_width_max_m,
            diagnostics.maximum_abs_nodal_force_n,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("Checkpoint diagnostics contain a negative magnitude.")
        ordered_ranges = (
            (
                diagnostics.base_crossflow_speed_min_mps,
                diagnostics.base_crossflow_speed_max_mps,
            ),
            (
                diagnostics.response_crossflow_speed_min_mps,
                diagnostics.response_crossflow_speed_max_mps,
            ),
            (
                diagnostics.response_section_width_min_m,
                diagnostics.response_section_width_max_m,
            ),
        )
        if any(low > high for low, high in ordered_ranges):
            raise ValueError("Checkpoint diagnostic min/max ordering failed.")
        if diagnostics.response_section_width_min_m <= 0.0:
            raise ValueError("Checkpoint response-section width must be positive.")
        for index, (mode_row, mode_config) in enumerate(
            zip(diagnostics.mode_diagnostics, self.config.modes)
        ):
            if (
                mode_row.complex_projection_leakage_ratio < 0.0
                or mode_row.qdot_rms_mps < 0.0
                or mode_row.amplitude_estimate_m < 0.0
                or mode_row.amplitude_ratio < 0.0
                or mode_row.reduced_velocity_min < 0.0
                or mode_row.reduced_velocity_min > mode_row.reduced_velocity_max
                or mode_row.activation_min < 0.0
                or mode_row.activation_max > 1.0
                or mode_row.activation_min > mode_row.activation_max
                or mode_row.maximum_mode_nodal_force_n < 0.0
            ):
                raise ValueError(
                    f"Checkpoint mode {mode_row.tracked_mode} diagnostics violate bounds."
                )
            expected_amplitude = (
                self.config.modal_basis.real_shape_maximum_site_norm[index]
                * math.sqrt(2.0)
                * mode_row.qdot_rms_mps
                / (2.0 * math.pi * mode_config.natural_frequency_hz)
            )
            expected_ratio = expected_amplitude / mode_config.reference_length_m
            expected_saturation = 1.0 - (
                expected_ratio / mode_config.saturation_amplitude_ratio
            ) ** 2
            if (
                not math.isclose(
                    mode_row.amplitude_estimate_m,
                    expected_amplitude,
                    rel_tol=1.0e-13,
                    abs_tol=2.0e-15,
                )
                or not math.isclose(
                    mode_row.amplitude_ratio,
                    expected_ratio,
                    rel_tol=1.0e-13,
                    abs_tol=2.0e-15,
                )
                or not math.isclose(
                    mode_row.saturation_factor,
                    expected_saturation,
                    rel_tol=1.0e-13,
                    abs_tol=2.0e-15,
                )
            ):
                raise ValueError(
                    f"Checkpoint mode {mode_row.tracked_mode} amplitude identity failed."
                )
            if (
                abs(mode_row.raw_gain_min) > mode_config.max_abs_gain
                or abs(mode_row.raw_gain_max) > mode_config.max_abs_gain
                or mode_row.raw_gain_min != mode_row.applied_gain_min
                or mode_row.raw_gain_max != mode_row.applied_gain_max
            ):
                raise ValueError(
                    f"Checkpoint mode {mode_row.tracked_mode} gain identity failed."
                )
        return diagnostics

    def _validate_restored_result(self, result: HVAWTResponseResult) -> None:
        force = np.asarray(result.additive_force_global_n)
        rigid = np.asarray(result.rigid_body_velocity_global_mps)
        flexible = np.asarray(result.flexible_velocity_global_mps)
        base = np.asarray(result.base_relative_velocity_global_mps)
        response = np.asarray(result.response_relative_velocity_global_mps)
        diagnostics = result.diagnostics
        if not np.allclose(response - base, flexible, rtol=0.0, atol=2.0e-14):
            raise ValueError("Checkpoint base/response/flexible flow identity failed.")
        total = rigid + flexible
        maximum_force = float(np.max(np.linalg.norm(force, axis=1)))
        scalar_identities = (
            (
                diagnostics.maximum_total_speed_mps,
                float(np.max(np.linalg.norm(total, axis=1))),
                "maximum total speed",
            ),
            (
                diagnostics.maximum_rigid_body_speed_mps,
                float(np.max(np.linalg.norm(rigid, axis=1))),
                "maximum rigid speed",
            ),
            (
                diagnostics.maximum_flexible_speed_mps,
                float(np.max(np.linalg.norm(flexible, axis=1))),
                "maximum flexible speed",
            ),
            (diagnostics.maximum_abs_nodal_force_n, maximum_force, "maximum force"),
            (
                diagnostics.flexible_power_w,
                float(np.sum(force * flexible)),
                "flexible power",
            ),
            (
                diagnostics.rigid_body_power_w,
                float(np.sum(force * rigid)),
                "rigid-body power",
            ),
        )
        for actual, expected, label in scalar_identities:
            if not math.isclose(actual, expected, rel_tol=2.0e-13, abs_tol=2.0e-12):
                raise ValueError(f"Checkpoint {label} identity failed.")
        if not math.isclose(
            diagnostics.total_structural_power_w,
            diagnostics.flexible_power_w + diagnostics.rigid_body_power_w,
            rel_tol=2.0e-13,
            abs_tol=2.0e-12,
        ):
            raise ValueError("Checkpoint total structural power identity failed.")
        if not math.isclose(
            diagnostics.flexible_power_w,
            sum(mode.flexible_power_w for mode in diagnostics.mode_diagnostics),
            rel_tol=2.0e-13,
            abs_tol=2.0e-12,
        ) or not math.isclose(
            diagnostics.rigid_body_power_w,
            sum(mode.rigid_body_power_w for mode in diagnostics.mode_diagnostics),
            rel_tol=2.0e-13,
            abs_tol=2.0e-12,
        ):
            raise ValueError("Checkpoint per-mode power identity failed.")
        if not math.isfinite(maximum_force):
            raise ValueError("Checkpoint response force must be finite.")
        if (
            self.config.max_abs_total_nodal_force_n is not None
            and maximum_force > self.config.max_abs_total_nodal_force_n
        ):
            raise ValueError("Checkpoint response force exceeds the configured force limit.")

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Transactionally restore an exact response-core checkpoint."""

        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping.")
        expected_state_keys = {
            "schema",
            "config_fingerprint",
            "modal_basis_fingerprint",
            "last_time_s",
            "last_azimuth_deg",
            "last_signed_rpm",
            "last_input_signature",
            "bandpass_states",
            "qdot_mean_square_m2ps2",
            "last_result",
            "state_sha256",
        }
        _require_exact_keys(state, expected_state_keys, "response checkpoint")
        if state.get("schema") != HVAWT_RESPONSE_CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported H-VAWT response checkpoint schema.")
        if state.get("config_fingerprint") != self.config.fingerprint:
            raise ValueError("H-VAWT response checkpoint configuration mismatch.")
        if state.get("modal_basis_fingerprint") != self.config.modal_basis.fingerprint:
            raise ValueError("H-VAWT response checkpoint modal-basis mismatch.")
        checksum = _hex_digest(state.get("state_sha256"), "state_sha256")
        payload = dict(state)
        del payload["state_sha256"]
        if checksum != _canonical_mapping_sha256(payload):
            raise ValueError("H-VAWT response checkpoint state SHA-256 mismatch.")
        state_rows = state.get("bandpass_states")
        if not isinstance(state_rows, Sequence) or isinstance(state_rows, (str, bytes)) or len(state_rows) != 2:
            raise ValueError("Checkpoint must contain two bandpass states.")
        candidate_filters: list[_BandpassState] = []
        for row in state_rows:
            if not isinstance(row, Mapping):
                raise ValueError("Checkpoint bandpass state must be a mapping.")
            _require_exact_keys(
                row,
                {"position_state", "rate_state", "previous_input_mps"},
                "checkpoint bandpass state",
            )
            position = _strict_finite_scalar(row["position_state"], "bandpass position_state")
            rate = _strict_finite_scalar(row["rate_state"], "bandpass rate_state")
            previous = row["previous_input_mps"]
            if previous is not None:
                previous = _strict_finite_scalar(previous, "bandpass previous_input_mps")
            filter_state = _BandpassState(position, rate, previous)
            candidate_filters.append(filter_state)
        mean_square_raw = state.get("qdot_mean_square_m2ps2")
        if (
            not isinstance(mean_square_raw, Sequence)
            or isinstance(mean_square_raw, (str, bytes))
            or len(mean_square_raw) != 2
        ):
            raise ValueError("Checkpoint qdot mean-square state is invalid.")
        mean_square = np.asarray(
            [
                _strict_finite_scalar(value, "qdot_mean_square_m2ps2")
                for value in mean_square_raw
            ],
            dtype=float,
        )
        if np.any(mean_square < 0.0):
            raise ValueError("Checkpoint qdot mean-square state is invalid.")
        last_time = state.get("last_time_s")
        last_azimuth = state.get("last_azimuth_deg")
        last_rpm = state.get("last_signed_rpm")
        last_signature = state.get("last_input_signature")
        history_values = (last_time, last_azimuth, last_rpm, last_signature)
        if any(value is None for value in history_values) and not all(
            value is None for value in history_values
        ):
            raise ValueError("Checkpoint accepted-time history is incomplete.")
        if last_time is not None:
            last_time = _strict_finite_scalar(last_time, "last_time_s")
            last_azimuth = _strict_finite_scalar(last_azimuth, "last_azimuth_deg")
            last_rpm = _strict_finite_scalar(last_rpm, "last_signed_rpm")
            last_signature = _hex_digest(last_signature, "last_input_signature")
            if (
                abs(last_rpm - self.config.kinematics.basis_rpm)
                > self.config.kinematics.maximum_abs_basis_rpm_error_rpm
            ):
                raise ValueError("Checkpoint LSS RPM is outside the audited modal basis.")
        raw_result = state.get("last_result")
        candidate_result: HVAWTResponseResult | None = None
        if raw_result is not None:
            if last_time is None or not isinstance(raw_result, Mapping):
                raise ValueError("Checkpoint last_result/history consistency failed.")
            _require_exact_keys(
                raw_result,
                {
                    "additive_force_global_n",
                    "rigid_body_velocity_global_mps",
                    "flexible_velocity_global_mps",
                    "base_relative_velocity_global_mps",
                    "response_relative_velocity_global_mps",
                    "diagnostics",
                },
                "checkpoint last_result",
            )
            diagnostics_raw = raw_result.get("diagnostics")
            if not isinstance(diagnostics_raw, Mapping):
                raise ValueError("Checkpoint response diagnostics are missing.")
            diagnostics = self._diagnostics_from_mapping(
                diagnostics_raw,
                last_time_s=last_time,
                last_azimuth_deg=last_azimuth,
                last_signed_rpm=last_rpm,
            )
            candidate_result = HVAWTResponseResult(
                additive_force_global_n=_copy_readonly(
                    _finite_node_vectors(
                        raw_result.get("additive_force_global_n"),
                        _SITE_COUNT,
                        "checkpoint force",
                    )
                ),
                rigid_body_velocity_global_mps=_copy_readonly(
                    _finite_node_vectors(
                        raw_result.get("rigid_body_velocity_global_mps"),
                        _SITE_COUNT,
                        "checkpoint rigid velocity",
                    )
                ),
                flexible_velocity_global_mps=_copy_readonly(
                    _finite_node_vectors(
                        raw_result.get("flexible_velocity_global_mps"),
                        _SITE_COUNT,
                        "checkpoint flexible velocity",
                    )
                ),
                base_relative_velocity_global_mps=_copy_readonly(
                    _finite_node_vectors(
                        raw_result.get("base_relative_velocity_global_mps"),
                        _SITE_COUNT,
                        "checkpoint base flow",
                    )
                ),
                response_relative_velocity_global_mps=_copy_readonly(
                    _finite_node_vectors(
                        raw_result.get("response_relative_velocity_global_mps"),
                        _SITE_COUNT,
                        "checkpoint response flow",
                    )
                ),
                diagnostics=diagnostics,
            )
            self._validate_restored_result(candidate_result)
            for index, (filter_state, mode_diagnostics, mode_config) in enumerate(
                zip(
                    candidate_filters,
                    diagnostics.mode_diagnostics,
                    self.config.modes,
                )
            ):
                expected_filtered = (
                    2.0
                    * mode_config.bandpass_damping_ratio
                    * 2.0
                    * math.pi
                    * mode_config.natural_frequency_hz
                    * filter_state.rate_state
                )
                if (
                    filter_state.previous_input_mps is None
                    or not math.isclose(
                        filter_state.previous_input_mps,
                        mode_diagnostics.corrected_raw_qdot_mps,
                        rel_tol=1.0e-13,
                        abs_tol=2.0e-15,
                    )
                    or not math.isclose(
                        expected_filtered,
                        mode_diagnostics.filtered_qdot_mps,
                        rel_tol=1.0e-13,
                        abs_tol=2.0e-15,
                    )
                    or not math.isclose(
                        float(mean_square[index]),
                        mode_diagnostics.qdot_rms_mps**2,
                        rel_tol=1.0e-13,
                        abs_tol=2.0e-15,
                    )
                ):
                    raise ValueError(
                        f"Checkpoint mode {mode_config.tracked_mode} filter/diagnostic state identity failed."
                    )
        elif last_time is not None:
            raise ValueError("Accepted checkpoint history requires last_result.")

        if last_time is None:
            if raw_result is not None:
                raise ValueError("Pristine checkpoint must not contain last_result.")
            if any(state != _BandpassState() for state in candidate_filters) or np.any(
                mean_square != 0.0
            ):
                raise ValueError("Pristine checkpoint contains non-pristine response state.")
        elif any(state.previous_input_mps is None for state in candidate_filters):
            raise ValueError("Accepted checkpoint has uninitialized bandpass state.")

        self._filter_states = tuple(candidate_filters)
        self._qdot_mean_square_m2ps2 = np.array(mean_square, copy=True)
        self._last_time_s = last_time
        self._last_azimuth_deg = last_azimuth
        self._last_signed_rpm = last_rpm
        self._last_input_signature = last_signature
        self._last_result_canonical = (
            None if candidate_result is None else _response_result_copy(candidate_result)
        )


__all__ = [
    "BASE_FLOW_BASIS",
    "COORDINATE_EXTRACTOR",
    "CURRENT_POSITION_BASIS",
    "EXTERNAL_VIV_LEGACY",
    "EXTERNAL_VIV_NONE",
    "EXTERNAL_VIV_RESPONSE",
    "FEEDBACK_VELOCITY_BASIS",
    "FIXED_AXIS_MOTION_MODEL",
    "FixedAxisKinematicState",
    "FixedAxisKinematicsConfig",
    "HVAWTForceSourceContract",
    "HVAWTModalBasis",
    "HVAWTModeForceDiagnostics",
    "HVAWTModeResponseConfig",
    "HVAWTResponseCore",
    "HVAWTResponseCoreConfig",
    "HVAWTResponseDiagnostics",
    "HVAWTResponseResult",
    "HVAWTSeparatedFlowState",
    "INCREMENTAL_VIV_SCREENING",
    "NATIVE_AERO_REQUIRED",
    "NOT_APPLICABLE",
    "REAL_REDUCTION_POLICY",
    "RESPONSE_FLOW_BASIS",
    "TOTAL_VELOCITY_BASIS",
    "calculate_hvawt_response_force",
    "decompose_fixed_axis_velocity",
    "load_tracked_hvawt_modal_basis",
    "separate_hvawt_flows",
]
