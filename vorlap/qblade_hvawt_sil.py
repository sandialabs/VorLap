"""SIL adapter for the audited two-mode H-VAWT response core.

The adapter deliberately reuses the geometry prepared by the existing
``VorLapQBladeRuntime`` but never evaluates its legacy aerodynamic source.
Native QBlade aerodynamics remain active in QBlade.  Schema v3 uses the core
as a zero-gain modal observer and applies only the calibrated static-spectrum
engineering lock-in force through an exact real-coordinate force adjoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numbers
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .qblade_hvawt_response import (
    EXTERNAL_VIV_RESPONSE,
    HVAWTForceSourceContract,
    HVAWTModeResponseConfig,
    HVAWTResponseCore,
    HVAWTResponseCoreConfig,
    FixedAxisKinematicsConfig,
    INCREMENTAL_VIV_SCREENING,
    NATIVE_AERO_REQUIRED,
    decompose_fixed_axis_velocity,
    load_tracked_hvawt_modal_basis,
    separate_hvawt_flows,
)
from .force_model import resolve_airfoil
from .qblade_engineering_lockin import (
    EngineeringLockInCalibration,
    EngineeringLockInTransferParameters,
)
from .unified_viv_gain import UnifiedVIVGainModel, UnifiedVIVGainState
from .qblade_griffin_lockin import GriffinCalibration, GriffinLockInParameters
from .qblade_single_mode_work_map_sil import (
    _lift_unit,
    _runtime_airfoil_source_identities,
    _sha256_file,
)
from .qblade_spectral_lockin import (
    HVAWTSpectralLockInSource,
    SpectralLockInCaptureConfig,
    SpectralLockInModeConfig,
    evaluate_modal_damping_balance,
)

HVAWT_SIL_RESPONSE_SCHEMA_VERSION = 1
HVAWT_SIL_RESPONSE_MODEL_TYPE = "hvawt_two_mode_negative_damping"
HVAWT_SIL_HYBRID_SCHEMA_VERSION = 2
HVAWT_SIL_HYBRID_MODEL_TYPE = "hvawt_airfoil_spectral_lockin_hybrid"
HVAWT_SIL_ENGINEERING_SCHEMA_VERSION = 3
HVAWT_SIL_ENGINEERING_MODEL_TYPE = "hvawt_airfoil_engineering_lockin_hybrid"
HVAWT_SIL_GRIFFIN_SCHEMA_VERSION = 4
HVAWT_SIL_GRIFFIN_MODEL_TYPE = "hvawt_airfoil_reciprocal_griffin_hybrid"
_TARGET_MODES = (6, 12)


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(mapping)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}.")


def _strict_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be a real numeric value.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _optional_positive_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    result = _strict_float(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive or null.")
    return result


def _case_member(control_dir: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} must be a nonempty case-relative path.")
    candidate = (control_dir / value).resolve()
    try:
        candidate.relative_to(control_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the staged Control directory.") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist: {candidate}")
    return candidate


@dataclass(frozen=True)
class HVAWTEngineeringModeDiagnostics:
    tracked_mode: int
    captured_lift_coefficient_rss: float
    zero_amplitude_feedback_coefficient_nspm: float
    structural_damping_coefficient_nspm: float
    aerodynamic_to_structural_damping_ratio: float
    linear_lockin_available: bool
    physical_amplitude_estimate_m: float
    physical_amplitude_ratio: float
    saturation_factor: float
    generalized_force_n: float
    generalized_power_w: float
    predicted_equilibrium_amplitude_m: float


@dataclass(frozen=True)
class HVAWTEngineeringStepDiagnostics:
    calibration_fingerprint: str
    static_spectrum_sha256: str
    velocity_basis: str
    facility_velocity_scale_applied: float
    mode_diagnostics: tuple[HVAWTEngineeringModeDiagnostics, ...]
    generalized_power_w: float
    nodal_flexible_power_w: float
    power_closure_error_w: float
    maximum_abs_nodal_force_n: float


@dataclass(frozen=True)
class HVAWTGriffinModeDiagnostics:
    tracked_mode: int
    captured_lift_coefficient_rss: float
    ideal_coulomb_generalized_force_scale_n: float
    regularized_coulomb_generalized_force_n: float
    velocity_damping_coefficient_nspm: float
    structural_damping_coefficient_nspm: float
    raw_modal_velocity_mps: float
    filtered_modal_velocity_mps: float
    generalized_force_n: float
    generalized_power_w: float
    predicted_coordinate_peak_amplitude_m: float
    predicted_physical_peak_amplitude_m: float


@dataclass(frozen=True)
class HVAWTGriffinStepDiagnostics:
    calibration_fingerprint: str
    static_spectrum_sha256: str
    closure_name: str
    velocity_basis: str
    mode_diagnostics: tuple[HVAWTGriffinModeDiagnostics, ...]
    generalized_power_w: float
    nodal_flexible_power_w: float
    power_closure_error_w: float
    maximum_abs_nodal_force_n: float


def calculate_hvawt_engineering_force(
    *,
    modal_basis: Any,
    gain_state: EngineeringLockInTransferParameters | UnifiedVIVGainState,
    spectral_result: Any,
    mode_configs: Sequence[HVAWTModeResponseConfig],
    structural_damping_ratios: Sequence[float],
    rotation_body_to_global: Any,
    response_relative_velocity_global_mps: Any,
    response_effective_crossflow_speed_mps: Any,
    chord_unit_global: Any,
    normal_unit_global: Any,
    chord_m: Any,
    local_length_m: Any,
    fluid_density_kgpm3: float,
    flexible_velocity_global_mps: Any,
    raw_qdot_mps: Any,
    filtered_qdot_mps: Any,
    qdot_rms_mps: Any,
    maximum_power_closure_relative_error: float,
    response_force_scale: float = 1.0,
) -> tuple[np.ndarray, HVAWTEngineeringStepDiagnostics]:
    """Evaluate the two-mode facility-independent engineering closure.

    The fixed-body HDF5 spectrum sets the available lift coefficient and the
    structural modal velocity sets its phase.  Generalized forces are mapped
    back with the exact real-coordinate adjoint, so nodal and modal work are
    identical for arbitrary simultaneous mode motion.
    """

    response_velocity = np.asarray(response_relative_velocity_global_mps, dtype=float)
    speed = np.asarray(response_effective_crossflow_speed_mps, dtype=float)
    chord_unit = np.asarray(chord_unit_global, dtype=float)
    normal_unit = np.asarray(normal_unit_global, dtype=float)
    chord_length = np.asarray(chord_m, dtype=float)
    section_length = np.asarray(local_length_m, dtype=float)
    flexible_velocity = np.asarray(flexible_velocity_global_mps, dtype=float)
    rotation = np.asarray(rotation_body_to_global, dtype=float)
    raw_qdot = np.asarray(raw_qdot_mps, dtype=float)
    filtered_qdot = np.asarray(filtered_qdot_mps, dtype=float)
    qdot_rms = np.asarray(qdot_rms_mps, dtype=float)
    captured = np.asarray(
        spectral_result.partition.captured_amplitude_by_node_mode,
        dtype=float,
    )
    count = len(modal_basis.node_ids)
    if any(
        array.shape != expected
        for array, expected in (
            (response_velocity, (count, 3)),
            (speed, (count,)),
            (chord_unit, (count, 3)),
            (normal_unit, (count, 3)),
            (chord_length, (count,)),
            (section_length, (count,)),
            (flexible_velocity, (count, 3)),
            (rotation, (3, 3)),
            (raw_qdot, (2,)),
            (filtered_qdot, (2,)),
            (qdot_rms, (2,)),
            (captured, (count, 2)),
        )
    ):
        raise ValueError(
            "Engineering H-VAWT closure received an array with wrong shape."
        )
    if any(
        np.any(~np.isfinite(value))
        for value in (
            response_velocity,
            speed,
            chord_unit,
            normal_unit,
            chord_length,
            section_length,
            flexible_velocity,
            rotation,
            raw_qdot,
            filtered_qdot,
            qdot_rms,
            captured,
        )
    ):
        raise ValueError("Engineering H-VAWT closure inputs must be finite.")
    if (
        np.any(speed <= 0.0)
        or np.any(chord_length <= 0.0)
        or np.any(section_length < 0.0)
    ):
        raise ValueError("Engineering H-VAWT section speed/chord/length is invalid.")
    density = _strict_float(fluid_density_kgpm3, "fluid_density_kgpm3")
    if density <= 0.0:
        raise ValueError("fluid_density_kgpm3 must be positive.")
    if len(mode_configs) != 2 or len(structural_damping_ratios) != 2:
        raise ValueError("Engineering H-VAWT closure requires exactly two modes.")
    force_scale = _strict_float(response_force_scale, "response_force_scale")
    if force_scale not in (0.0, 1.0):
        raise ValueError("response_force_scale must be exactly 0 or 1.")

    velocity_chord = np.sum(response_velocity * chord_unit, axis=1)
    velocity_normal = np.sum(response_velocity * normal_unit, axis=1)
    lift_unit = _lift_unit(
        chord_unit,
        normal_unit,
        velocity_chord,
        velocity_normal,
        speed,
    )
    mode_shape_global = np.einsum(
        "ab,ibm->iam",
        rotation,
        modal_basis.real_shape_body_per_m,
    )
    section_dynamic_force = 0.5 * density * speed * chord_length * section_length
    generalized_force = np.zeros(2, dtype=float)
    diagnostics: list[HVAWTEngineeringModeDiagnostics] = []
    for index, (mode, damping_ratio) in enumerate(
        zip(mode_configs, structural_damping_ratios)
    ):
        mode_shape = mode_shape_global[:, :, index]
        participation_norm = np.linalg.norm(mode_shape, axis=1)
        normalized_shape = np.divide(
            mode_shape,
            participation_norm[:, None],
            out=np.zeros_like(mode_shape),
            where=participation_norm[:, None] > 0.0,
        )
        lift_projection = np.abs(np.sum(lift_unit * normalized_shape, axis=1))
        spectral_basis = float(
            np.sum(
                section_dynamic_force
                * captured[:, index]
                * lift_projection
                * participation_norm**2
            )
        )
        c_a0 = (
            gain_state.zero_amplitude_feedback(spectral_basis=spectral_basis)
            if isinstance(gain_state, UnifiedVIVGainState)
            else gain_state.feedback_gain * spectral_basis
        )
        omega = math.sqrt(
            mode.generalized_modal_stiffness_npm / mode.generalized_modal_mass_kg
        )
        c_structural = (
            2.0 * float(damping_ratio) * mode.generalized_modal_mass_kg * omega
        )
        coordinate_amplitude = math.sqrt(2.0) * qdot_rms[index] / omega
        physical_amplitude = float(np.max(participation_norm)) * coordinate_amplitude
        physical_ratio = physical_amplitude / mode.reference_length_m
        saturation = (
            gain_state.saturation_factor_from_ratio(physical_ratio)
            if isinstance(gain_state, UnifiedVIVGainState)
            else 1.0
            - (physical_ratio / gain_state.limiting_amplitude_ratio) ** 2
        )
        generalized_force[index] = (
            force_scale * c_a0 * saturation * filtered_qdot[index]
        )
        ratio = c_a0 / c_structural if c_structural > 0.0 else math.inf
        equilibrium = (
            gain_state.limiting_amplitude_ratio
            * mode.reference_length_m
            * math.sqrt(max(0.0, 1.0 - c_structural / c_a0))
            if c_a0 > c_structural
            else 0.0
        )
        diagnostics.append(
            HVAWTEngineeringModeDiagnostics(
                tracked_mode=mode.tracked_mode,
                captured_lift_coefficient_rss=float(
                    np.sqrt(np.sum(captured[:, index] ** 2))
                ),
                zero_amplitude_feedback_coefficient_nspm=c_a0,
                structural_damping_coefficient_nspm=c_structural,
                aerodynamic_to_structural_damping_ratio=ratio,
                linear_lockin_available=bool(c_a0 > c_structural),
                physical_amplitude_estimate_m=physical_amplitude,
                physical_amplitude_ratio=physical_ratio,
                saturation_factor=saturation,
                generalized_force_n=float(generalized_force[index]),
                generalized_power_w=float(generalized_force[index] * raw_qdot[index]),
                predicted_equilibrium_amplitude_m=equilibrium,
            )
        )

    force_body = modal_basis.energy_conjugate_force_body(generalized_force)
    force_global = force_body @ rotation.T
    nodal_power = float(np.sum(force_global * flexible_velocity))
    generalized_power = float(np.dot(generalized_force, raw_qdot))
    closure_error = abs(nodal_power - generalized_power)
    relative_tolerance = _strict_float(
        maximum_power_closure_relative_error,
        "maximum_power_closure_relative_error",
    )
    if relative_tolerance <= 0.0:
        raise ValueError("maximum_power_closure_relative_error must be positive.")
    tolerance = relative_tolerance * max(
        1.0,
        abs(nodal_power),
        abs(generalized_power),
    )
    if closure_error > tolerance:
        raise FloatingPointError(
            f"Engineering H-VAWT virtual-work error {closure_error:.17g} W "
            f"exceeds {tolerance:.17g} W."
        )
    return force_global, HVAWTEngineeringStepDiagnostics(
        calibration_fingerprint=gain_state.calibration_fingerprint,
        static_spectrum_sha256=gain_state.static_spectrum_sha256,
        velocity_basis="local_rigid_body_relative_unscaled",
        facility_velocity_scale_applied=1.0,
        mode_diagnostics=tuple(diagnostics),
        generalized_power_w=generalized_power,
        nodal_flexible_power_w=nodal_power,
        power_closure_error_w=closure_error,
        maximum_abs_nodal_force_n=float(np.max(np.linalg.norm(force_global, axis=1))),
    )


def calculate_hvawt_griffin_force(
    *,
    modal_basis: Any,
    parameters: GriffinLockInParameters,
    calibration_fingerprint: str,
    static_spectrum_sha256: str,
    spectral_result: Any,
    mode_configs: Sequence[HVAWTModeResponseConfig],
    structural_damping_ratios: Sequence[float],
    rotation_body_to_global: Any,
    response_relative_velocity_global_mps: Any,
    response_effective_crossflow_speed_mps: Any,
    chord_unit_global: Any,
    normal_unit_global: Any,
    chord_m: Any,
    local_length_m: Any,
    fluid_density_kgpm3: float,
    flexible_velocity_global_mps: Any,
    raw_qdot_mps: Any,
    filtered_qdot_mps: Any,
    maximum_power_closure_relative_error: float,
    response_force_scale: float = 1.0,
) -> tuple[np.ndarray, HVAWTGriffinStepDiagnostics]:
    """Evaluate the frozen reciprocal-Griffin closure on the rotating H-VAWT.

    Static spectra determine the captured fluctuating lift coefficient at each
    section.  The calibrated Coulomb-like term supplies low-amplitude positive
    work and the velocity term supplies stabilizing aerodynamic damping.  The
    two generalized forces are returned through the audited real-coordinate
    adjoint, so ``sum(F_i dot v_i) == sum(Q_m*qdot_m)`` for simultaneous modes.
    """

    response_velocity = np.asarray(response_relative_velocity_global_mps, dtype=float)
    speed = np.asarray(response_effective_crossflow_speed_mps, dtype=float)
    chord_unit = np.asarray(chord_unit_global, dtype=float)
    normal_unit = np.asarray(normal_unit_global, dtype=float)
    chord_length = np.asarray(chord_m, dtype=float)
    section_length = np.asarray(local_length_m, dtype=float)
    flexible_velocity = np.asarray(flexible_velocity_global_mps, dtype=float)
    rotation = np.asarray(rotation_body_to_global, dtype=float)
    raw_qdot = np.asarray(raw_qdot_mps, dtype=float)
    filtered_qdot = np.asarray(filtered_qdot_mps, dtype=float)
    captured = np.asarray(
        spectral_result.partition.captured_amplitude_by_node_mode,
        dtype=float,
    )
    count = len(modal_basis.node_ids)
    expected_arrays = (
        (response_velocity, (count, 3)),
        (speed, (count,)),
        (chord_unit, (count, 3)),
        (normal_unit, (count, 3)),
        (chord_length, (count,)),
        (section_length, (count,)),
        (flexible_velocity, (count, 3)),
        (rotation, (3, 3)),
        (raw_qdot, (2,)),
        (filtered_qdot, (2,)),
        (captured, (count, 2)),
    )
    if any(array.shape != shape for array, shape in expected_arrays):
        raise ValueError(
            "Reciprocal-Griffin H-VAWT closure received a wrong array shape."
        )
    if any(np.any(~np.isfinite(array)) for array, _shape in expected_arrays):
        raise ValueError("Reciprocal-Griffin H-VAWT closure inputs must be finite.")
    if (
        np.any(speed <= 0.0)
        or np.any(chord_length <= 0.0)
        or np.any(section_length < 0.0)
        or np.any(captured < 0.0)
    ):
        raise ValueError(
            "Reciprocal-Griffin section speed/chord/length/capture is invalid."
        )
    density = _strict_float(fluid_density_kgpm3, "fluid_density_kgpm3")
    if density <= 0.0:
        raise ValueError("fluid_density_kgpm3 must be positive.")
    if len(mode_configs) != 2 or len(structural_damping_ratios) != 2:
        raise ValueError(
            "Reciprocal-Griffin H-VAWT closure requires exactly two modes."
        )
    force_scale = _strict_float(response_force_scale, "response_force_scale")
    if force_scale not in (0.0, 1.0):
        raise ValueError("response_force_scale must be exactly 0 or 1.")
    if (
        not isinstance(calibration_fingerprint, str)
        or len(calibration_fingerprint) != 64
    ):
        raise ValueError("calibration_fingerprint must be one SHA-256 digest.")
    if not isinstance(static_spectrum_sha256, str) or len(static_spectrum_sha256) != 64:
        raise ValueError("static_spectrum_sha256 must be one SHA-256 digest.")

    velocity_chord = np.sum(response_velocity * chord_unit, axis=1)
    velocity_normal = np.sum(response_velocity * normal_unit, axis=1)
    lift_unit = _lift_unit(
        chord_unit,
        normal_unit,
        velocity_chord,
        velocity_normal,
        speed,
    )
    mode_shape_global = np.einsum(
        "ab,ibm->iam",
        rotation,
        modal_basis.real_shape_body_per_m,
    )
    generalized_force = np.zeros(2, dtype=float)
    diagnostics: list[HVAWTGriffinModeDiagnostics] = []
    for index, (mode, damping_ratio) in enumerate(
        zip(mode_configs, structural_damping_ratios, strict=True)
    ):
        projection = np.sum(lift_unit * mode_shape_global[:, :, index], axis=1)
        local_force_scale = (
            0.5
            * density
            * speed**2
            * chord_length
            * section_length
            * captured[:, index]
        )
        ideal_coulomb_scale = float(np.sum(local_force_scale * np.abs(projection)))
        local_modal_velocity = projection * filtered_qdot[index]
        epsilon = parameters.regularization_velocity_over_wind * speed
        regularized_direction = np.divide(
            local_modal_velocity,
            np.sqrt(local_modal_velocity**2 + epsilon**2),
            out=np.zeros_like(local_modal_velocity),
            where=(local_modal_velocity != 0.0),
        )
        regularized_coulomb = parameters.coulomb_work_gain * float(
            np.sum(local_force_scale * projection * regularized_direction)
        )
        velocity_damping = parameters.velocity_damping_gain * float(
            np.sum(local_force_scale * projection**2 / speed)
        )
        generalized_force[index] = force_scale * (
            regularized_coulomb - velocity_damping * filtered_qdot[index]
        )
        omega = math.sqrt(
            mode.generalized_modal_stiffness_npm / mode.generalized_modal_mass_kg
        )
        structural_damping = (
            2.0 * float(damping_ratio) * mode.generalized_modal_mass_kg * omega
        )
        ideal_peak_velocity = (
            4.0
            * parameters.coulomb_work_gain
            * ideal_coulomb_scale
            / (math.pi * (structural_damping + velocity_damping))
            if ideal_coulomb_scale > 0.0
            else 0.0
        )
        coordinate_peak = ideal_peak_velocity / omega
        physical_peak = (
            float(np.max(np.linalg.norm(mode_shape_global[:, :, index], axis=1)))
            * coordinate_peak
        )
        diagnostics.append(
            HVAWTGriffinModeDiagnostics(
                tracked_mode=mode.tracked_mode,
                captured_lift_coefficient_rss=float(
                    np.sqrt(np.sum(captured[:, index] ** 2))
                ),
                ideal_coulomb_generalized_force_scale_n=ideal_coulomb_scale,
                regularized_coulomb_generalized_force_n=regularized_coulomb,
                velocity_damping_coefficient_nspm=velocity_damping,
                structural_damping_coefficient_nspm=structural_damping,
                raw_modal_velocity_mps=float(raw_qdot[index]),
                filtered_modal_velocity_mps=float(filtered_qdot[index]),
                generalized_force_n=float(generalized_force[index]),
                generalized_power_w=float(generalized_force[index] * raw_qdot[index]),
                predicted_coordinate_peak_amplitude_m=coordinate_peak,
                predicted_physical_peak_amplitude_m=physical_peak,
            )
        )

    force_body = modal_basis.energy_conjugate_force_body(generalized_force)
    force_global = force_body @ rotation.T
    nodal_power = float(np.sum(force_global * flexible_velocity))
    generalized_power = float(np.dot(generalized_force, raw_qdot))
    closure_error = abs(nodal_power - generalized_power)
    relative_tolerance = _strict_float(
        maximum_power_closure_relative_error,
        "maximum_power_closure_relative_error",
    )
    if relative_tolerance <= 0.0:
        raise ValueError("maximum_power_closure_relative_error must be positive.")
    tolerance = relative_tolerance * max(1.0, abs(nodal_power), abs(generalized_power))
    if closure_error > tolerance:
        raise FloatingPointError(
            f"Reciprocal-Griffin H-VAWT virtual-work error {closure_error:.17g} W "
            f"exceeds {tolerance:.17g} W."
        )
    return force_global, HVAWTGriffinStepDiagnostics(
        calibration_fingerprint=calibration_fingerprint,
        static_spectrum_sha256=static_spectrum_sha256,
        closure_name="static-spectrum-scaled Gao--Scanlan reciprocal-Griffin work closure",
        velocity_basis="local_rigid_body_relative_unscaled",
        mode_diagnostics=tuple(diagnostics),
        generalized_power_w=generalized_power,
        nodal_flexible_power_w=nodal_power,
        power_closure_error_w=closure_error,
        maximum_abs_nodal_force_n=float(np.max(np.linalg.norm(force_global, axis=1))),
    )


class HVAWTSilResponseAdapter:
    """Translate one 355-channel SIL snapshot into 39 global nodal forces."""

    def __init__(
        self,
        *,
        raw_config: Mapping[str, Any],
        parameter_file: Path,
        runtime: Any,
        sample_period_s: float | None = None,
    ) -> None:
        if not isinstance(raw_config, Mapping):
            raise TypeError("hvawt_response_model must be a mapping.")
        resolved_sample_period_s = (
            None
            if sample_period_s is None
            else _strict_float(sample_period_s, "sample_period_s")
        )
        if resolved_sample_period_s is not None and resolved_sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive.")
        schema_version = raw_config.get("schema_version")
        model_type = raw_config.get("type")
        legacy_model = (
            schema_version == HVAWT_SIL_RESPONSE_SCHEMA_VERSION
            and model_type == HVAWT_SIL_RESPONSE_MODEL_TYPE
        )
        hybrid_model = (
            schema_version == HVAWT_SIL_HYBRID_SCHEMA_VERSION
            and model_type == HVAWT_SIL_HYBRID_MODEL_TYPE
        )
        engineering_model = (
            schema_version == HVAWT_SIL_ENGINEERING_SCHEMA_VERSION
            and model_type == HVAWT_SIL_ENGINEERING_MODEL_TYPE
        )
        griffin_model = (
            schema_version == HVAWT_SIL_GRIFFIN_SCHEMA_VERSION
            and model_type == HVAWT_SIL_GRIFFIN_MODEL_TYPE
        )
        if (
            not legacy_model
            and not hybrid_model
            and not engineering_model
            and not griffin_model
        ):
            raise ValueError(
                "Unsupported H-VAWT response schema/type; expected the legacy "
                "schema-v1 closure, airfoil-spectral schema-v2 hybrid, or "
                "engineering-lock-in schema-v3 or reciprocal-Griffin schema-v4 hybrid."
            )
        expected_top = {
            "schema_version",
            "type",
            "enabled",
            "rpm_block",
            "position_block",
            "modal_artifacts",
            "kinematics",
            "modes",
            "max_abs_total_nodal_force_n",
            "force_sources",
        }
        if hybrid_model:
            expected_top.add("spectral_lockin")
        if engineering_model:
            expected_top.add("engineering_lockin")
        if griffin_model:
            expected_top.add("griffin_lockin")
        _exact_keys(raw_config, expected_top, "hvawt_response_model")
        if raw_config["enabled"] is not True:
            raise ValueError("H-VAWT response model must be explicitly enabled.")

        controller = runtime.controller
        if float(controller.force_scale) != 0.0:
            raise ValueError(
                "H-VAWT response mode requires legacy VorLap force_scale=0."
            )
        if bool(controller.include_mean_load):
            raise ValueError("H-VAWT response mode requires include_mean_load=false.")
        if bool(getattr(controller, "response_model_enabled", False)):
            raise ValueError(
                "The single-mode controller response model must be disabled."
            )

        layout = runtime.layout
        self.rpm_block = str(raw_config["rpm_block"])
        self.position_block = str(raw_config["position_block"])
        if self.rpm_block not in layout or layout[self.rpm_block].shape != ():
            raise ValueError("H-VAWT RPM swap block must be scalar.")
        node_ids = tuple(str(value) for value in controller.node_ids)
        component_ids_by_node = tuple(
            str(value)
            for value in getattr(controller, "_node_component_ids", node_ids)
        )
        if self.position_block not in layout or layout[self.position_block].shape != (
            len(node_ids),
            3,
        ):
            raise ValueError("H-VAWT position swap block must have shape (39, 3).")
        self._rpm_spec = layout[self.rpm_block]
        self._position_spec = layout[self.position_block]

        artifacts = raw_config["modal_artifacts"]
        if not isinstance(artifacts, Mapping):
            raise TypeError("modal_artifacts must be a mapping.")
        _exact_keys(
            artifacts,
            {"identification", "shapes", "duals", "review_marker"},
            "modal_artifacts",
        )
        control_dir = parameter_file.resolve().parent
        modal_basis = load_tracked_hvawt_modal_basis(
            identification_path=_case_member(
                control_dir, artifacts["identification"], "modal identification"
            ),
            shapes_path=_case_member(control_dir, artifacts["shapes"], "modal shapes"),
            duals_path=_case_member(control_dir, artifacts["duals"], "modal duals"),
            review_marker_path=_case_member(
                control_dir, artifacts["review_marker"], "modal review marker"
            ),
        )
        if modal_basis.node_ids != node_ids:
            raise ValueError("Runtime node order differs from the audited modal basis.")

        kinematics = raw_config["kinematics"]
        if not isinstance(kinematics, Mapping):
            raise TypeError("kinematics must be a mapping.")
        _exact_keys(
            kinematics,
            {
                "axis_unit_global",
                "axis_origin_global_m",
                "body_to_global_at_reference",
                "reference_azimuth_deg",
                "maximum_abs_basis_rpm_error_rpm",
                "maximum_azimuth_rate_error_radps",
                "rpm_sign",
                "azimuth_sign",
            },
            "kinematics",
        )
        kinematic_config = FixedAxisKinematicsConfig(
            node_ids=node_ids,
            basis_rpm=modal_basis.basis_rpm,
            **dict(kinematics),
        )

        force_sources = raw_config["force_sources"]
        if not isinstance(force_sources, Mapping):
            raise TypeError("force_sources must be a mapping.")
        _exact_keys(
            force_sources,
            {
                "native_qblade_aerodynamics",
                "external_viv_mode",
                "include_mean_load",
                "coefficient_role",
            },
            "force_sources",
        )
        force_contract = HVAWTForceSourceContract(**dict(force_sources))
        force_contract.validate_runtime_flags(
            native_aero_enabled=True,
            legacy_source_enabled=False,
            response_closure_enabled=True,
            include_mean_load=False,
        )
        if (
            force_contract.native_qblade_aerodynamics != NATIVE_AERO_REQUIRED
            or force_contract.external_viv_mode != EXTERNAL_VIV_RESPONSE
            or force_contract.coefficient_role != INCREMENTAL_VIV_SCREENING
        ):
            raise ValueError(
                "H-VAWT force-source contract is not the frozen screening composition."
            )

        engineering = None
        transfer = None
        unified_gain_state = None
        unified_gain_model_sha256 = None
        calibration = None
        calibration_sha256 = None
        if engineering_model:
            engineering = raw_config["engineering_lockin"]
            if not isinstance(engineering, Mapping):
                raise TypeError("engineering_lockin must be a mapping.")
            engineering_keys = {
                "schema_version",
                "calibration_path",
                "calibration_sha256",
                "source_channel",
                "static_spectrum_sha256",
                "source_bounds_policy",
                "n_frequency_entries",
                "maximum_spectral_gain",
                "minimum_modal_participation",
                "residual_source_scale",
                "velocity_basis",
                "maximum_power_closure_relative_error",
                "response_force_scale",
            }
            engineering_schema = engineering.get("schema_version")
            if engineering_schema == 2:
                engineering_keys.update(
                    {
                        "unified_gain_model_path",
                        "unified_gain_model_sha256",
                        "unified_gain_branch",
                    }
                )
            _exact_keys(
                engineering,
                engineering_keys,
                "engineering_lockin",
            )
            if (
                engineering_schema not in (1, 2)
                or engineering["source_channel"] != "CL"
                or engineering["residual_source_scale"] not in (0.0, 1.0)
                or engineering["velocity_basis"] != "local_rigid_body_relative_unscaled"
                or engineering["response_force_scale"] not in (0.0, 1.0)
            ):
                raise ValueError(
                    "Engineering H-VAWT contract requires CL capture, residual "
                    "scale exactly 0 or 1, and unscaled local rigid-body-relative "
                    "velocity."
                )
            calibration_path = _case_member(
                control_dir,
                engineering["calibration_path"],
                "engineering calibration",
            )
            calibration_sha256 = _sha256_file(calibration_path)
            if calibration_sha256 != str(engineering["calibration_sha256"]).lower():
                raise ValueError("Engineering calibration SHA-256 differs from config.")
            calibration = EngineeringLockInCalibration.from_json(calibration_path)
            transfer = calibration.transferable_parameters
            if engineering_schema == 2:
                unified_path = _case_member(
                    control_dir,
                    engineering["unified_gain_model_path"],
                    "unified gain model",
                )
                unified_gain_model_sha256 = _sha256_file(unified_path)
                if unified_gain_model_sha256 != str(
                    engineering["unified_gain_model_sha256"]
                ).lower():
                    raise ValueError(
                        "Unified gain model SHA-256 differs from config."
                    )
                unified_model = UnifiedVIVGainModel.from_json(unified_path)
                if engineering["unified_gain_branch"] != "naca0018_quasi_static_azimuth":
                    raise ValueError("Unsupported H-VAWT unified gain branch.")
                unified_gain_state = unified_model.evaluate_airfoil(
                    profile="NACA0018",
                    kinematic_family="quasi_static_azimuth",
                )
                parity = {
                    "feedback_gain": (
                        unified_gain_state.feedback_gain,
                        transfer.feedback_gain,
                    ),
                    "limiting_amplitude_ratio": (
                        unified_gain_state.limiting_amplitude_ratio,
                        transfer.limiting_amplitude_ratio,
                    ),
                    "capture_window": (
                        unified_gain_state.capture_window,
                        transfer.capture_window,
                    ),
                    "frequency_ratio_low": (
                        unified_gain_state.frequency_ratio_low,
                        transfer.frequency_ratio_low,
                    ),
                    "frequency_ratio_peak": (
                        unified_gain_state.frequency_ratio_peak,
                        transfer.frequency_ratio_peak,
                    ),
                    "frequency_ratio_high": (
                        unified_gain_state.frequency_ratio_high,
                        transfer.frequency_ratio_high,
                    ),
                    "static_spectrum_sha256": (
                        unified_gain_state.static_spectrum_sha256,
                        transfer.static_spectrum_sha256,
                    ),
                }
                mismatches = [
                    key for key, (unified, legacy) in parity.items() if unified != legacy
                ]
                if mismatches:
                    raise ValueError(
                        "Unified gain state differs from the frozen H-VAWT "
                        f"calibration: {mismatches}."
                    )
            if (
                transfer.static_spectrum_sha256
                != str(engineering["static_spectrum_sha256"]).lower()
            ):
                raise ValueError(
                    "Engineering static-spectrum identity differs from calibration."
                )
            if (
                str(controller.source_bounds_policy)
                != engineering["source_bounds_policy"]
            ):
                raise ValueError(
                    "Engineering source-bounds policy differs from runtime."
                )
            if int(controller.n_freq_depth) != engineering["n_frequency_entries"]:
                raise ValueError("Engineering frequency depth differs from runtime.")

        griffin = None
        griffin_calibration = None
        griffin_calibration_sha256 = None
        if griffin_model:
            griffin = raw_config["griffin_lockin"]
            if not isinstance(griffin, Mapping):
                raise TypeError("griffin_lockin must be a mapping.")
            _exact_keys(
                griffin,
                {
                    "schema_version",
                    "calibration_path",
                    "calibration_sha256",
                    "static_spectrum_path",
                    "static_spectrum_sha256",
                    "source_channel",
                    "source_bounds_policy",
                    "n_frequency_entries",
                    "maximum_spectral_gain",
                    "minimum_modal_participation",
                    "residual_source_scale",
                    "velocity_basis",
                    "maximum_power_closure_relative_error",
                    "response_force_scale",
                },
                "griffin_lockin",
            )
            if (
                griffin["schema_version"] != 1
                or griffin["source_channel"] != "CL"
                or griffin["residual_source_scale"] != 0.0
                or griffin["velocity_basis"] != "local_rigid_body_relative_unscaled"
                or griffin["response_force_scale"] not in (0.0, 1.0)
            ):
                raise ValueError(
                    "Reciprocal-Griffin H-VAWT contract requires CL capture, "
                    "residual=0, and unscaled local rigid-body-relative velocity."
                )
            calibration_path = _case_member(
                control_dir,
                griffin["calibration_path"],
                "reciprocal-Griffin calibration",
            )
            griffin_calibration_sha256 = _sha256_file(calibration_path)
            if griffin_calibration_sha256 != str(griffin["calibration_sha256"]).lower():
                raise ValueError(
                    "Reciprocal-Griffin calibration SHA-256 differs from config."
                )
            griffin_calibration = GriffinCalibration.from_json(calibration_path)
            if str(controller.source_bounds_policy) != griffin["source_bounds_policy"]:
                raise ValueError(
                    "Reciprocal-Griffin source-bounds policy differs from runtime."
                )
            if int(controller.n_freq_depth) != griffin["n_frequency_entries"]:
                raise ValueError(
                    "Reciprocal-Griffin frequency depth differs from runtime."
                )

        mode_rows = raw_config["modes"]
        if not isinstance(mode_rows, list) or len(mode_rows) != 2:
            raise ValueError("modes must contain exactly mode 6 and mode 12.")
        expected_mode_keys = {
            "tracked_mode",
            "reference_length_m",
            "reduced_velocity_low",
            "reduced_velocity_peak",
            "reduced_velocity_high",
            "aerodynamic_damping_gain_h1",
            "saturation_amplitude_ratio",
            "rms_time_constant_s",
            "bandpass_damping_ratio",
            "max_abs_gain",
            "coefficient_source_sha256",
        }
        if hybrid_model:
            expected_mode_keys.add("structural_damping_ratio")
        if engineering_model or griffin_model:
            expected_mode_keys = {
                "tracked_mode",
                "reference_length_m",
                "structural_damping_ratio",
                "rms_time_constant_s",
                "bandpass_damping_ratio",
            }
        modes = []
        spectral_modes = []
        structural_damping_ratios = []
        for index, row in enumerate(mode_rows):
            if not isinstance(row, Mapping):
                raise TypeError("Each mode row must be a mapping.")
            _exact_keys(row, expected_mode_keys, f"modes[{index}]")
            if row["tracked_mode"] != _TARGET_MODES[index]:
                raise ValueError("Mode rows must be ordered as mode 6 then mode 12.")
            response_row = dict(row)
            structural_damping_ratio = response_row.pop(
                "structural_damping_ratio",
                None,
            )
            if hybrid_model or engineering_model or griffin_model:
                structural_damping_ratios.append(
                    _strict_float(
                        structural_damping_ratio,
                        f"modes[{index}].structural_damping_ratio",
                    )
                )
            if engineering_model or griffin_model:
                response_row.update(
                    {
                        "reduced_velocity_low": 0.0,
                        "reduced_velocity_peak": 1.0,
                        "reduced_velocity_high": 2.0,
                        "aerodynamic_damping_gain_h1": 0.0,
                        "saturation_amplitude_ratio": (
                            transfer.limiting_amplitude_ratio
                            if engineering_model
                            else 1.0
                        ),
                        "max_abs_gain": 1.0,
                        "coefficient_source_sha256": (
                            transfer.calibration_fingerprint
                            if engineering_model
                            else griffin_calibration.fingerprint
                        ),
                    }
                )
            modes.append(
                HVAWTModeResponseConfig(
                    natural_frequency_hz=modal_basis.natural_frequency_hz[index],
                    generalized_modal_mass_kg=modal_basis.generalized_modal_mass_kg[
                        index
                    ],
                    generalized_modal_stiffness_npm=modal_basis.generalized_modal_stiffness_npm[
                        index
                    ],
                    **response_row,
                )
            )
            if hybrid_model or engineering_model or griffin_model:
                spectral_modes.append(
                    SpectralLockInModeConfig(
                        tracked_mode=int(row["tracked_mode"]),
                        natural_frequency_hz=modal_basis.natural_frequency_hz[index],
                        generalized_modal_mass_kg=(
                            modal_basis.generalized_modal_mass_kg[index]
                        ),
                        structural_damping_ratio=structural_damping_ratio,
                        reference_length_m=row["reference_length_m"],
                        aerodynamic_damping_gain_h1=(
                            row["aerodynamic_damping_gain_h1"] if hybrid_model else 0.0
                        ),
                        saturation_amplitude_ratio=(
                            row["saturation_amplitude_ratio"]
                            if hybrid_model
                            else (
                                transfer.limiting_amplitude_ratio
                                if engineering_model
                                else 1.0
                            )
                        ),
                    )
                )

        self.core = HVAWTResponseCore(
            HVAWTResponseCoreConfig(
                kinematics=kinematic_config,
                modal_basis=modal_basis,
                modes=tuple(modes),
                force_sources=force_contract,
                max_abs_total_nodal_force_n=_optional_positive_float(
                    raw_config["max_abs_total_nodal_force_n"],
                    "max_abs_total_nodal_force_n",
                ),
            )
        )
        self.runtime = runtime
        self.node_ids = node_ids
        self.model_type = str(model_type)
        self.schema_version = int(schema_version)
        self.max_abs_total_nodal_force_n = _optional_positive_float(
            raw_config["max_abs_total_nodal_force_n"],
            "max_abs_total_nodal_force_n",
        )
        self.spectral_source = None
        self.spectral_modes = tuple(spectral_modes)
        self.residual_source_scale = 0.0
        self._hybrid_config_fingerprint = None
        self.last_spectral_result = None
        self.last_damping_balance = ()
        self.transfer = transfer
        self.unified_gain_state = unified_gain_state
        self.unified_gain_model_sha256 = unified_gain_model_sha256
        self.calibration_sha256 = calibration_sha256
        self.structural_damping_ratios = tuple(structural_damping_ratios)
        self.maximum_power_closure_relative_error = None
        self.engineering_force_scale = 0.0
        self.last_engineering_diagnostics = None
        self.griffin_parameters = (
            None if griffin_calibration is None else griffin_calibration.parameters
        )
        self.griffin_calibration_fingerprint = (
            None if griffin_calibration is None else griffin_calibration.fingerprint
        )
        self.griffin_calibration_sha256 = griffin_calibration_sha256
        self.griffin_static_spectrum_sha256 = None
        self.griffin_force_scale = 0.0
        self.last_griffin_diagnostics = None
        self.last_total_maximum_nodal_force_n = 0.0
        self.performance_timing = None
        if hybrid_model:
            spectral = raw_config["spectral_lockin"]
            if not isinstance(spectral, Mapping):
                raise TypeError("spectral_lockin must be a mapping.")
            _exact_keys(
                spectral,
                {
                    "schema_version",
                    "source_channel",
                    "reference_strouhal",
                    "reference_lift_amplitude",
                    "frequency_ratio_low",
                    "frequency_ratio_peak",
                    "frequency_ratio_high",
                    "maximum_spectral_gain",
                    "minimum_modal_participation",
                    "retain_uncaptured_lift_spectrum",
                    "retain_fluctuating_drag_spectrum",
                    "include_mean_load",
                    "residual_source_scale",
                },
                "spectral_lockin",
            )
            if (
                spectral["schema_version"] != 1
                or spectral["source_channel"] != "CL"
                or spectral["retain_uncaptured_lift_spectrum"] is not True
                or spectral["retain_fluctuating_drag_spectrum"] is not True
                or spectral["include_mean_load"] is not False
            ):
                raise ValueError(
                    "Hybrid spectral contract requires CL capture, retained CL/CD "
                    "fluctuations, and no external mean load."
                )
            residual_scale = _strict_float(
                spectral["residual_source_scale"],
                "spectral_lockin.residual_source_scale",
            )
            if residual_scale not in (0.0, 1.0):
                raise ValueError("residual_source_scale must be exactly 0 or 1.")
            self.residual_source_scale = residual_scale
            capture = SpectralLockInCaptureConfig(
                schema_version=spectral["schema_version"],
                reference_strouhal=spectral["reference_strouhal"],
                reference_lift_amplitude=spectral["reference_lift_amplitude"],
                frequency_ratio_low=spectral["frequency_ratio_low"],
                frequency_ratio_peak=spectral["frequency_ratio_peak"],
                frequency_ratio_high=spectral["frequency_ratio_high"],
                maximum_spectral_gain=spectral["maximum_spectral_gain"],
                minimum_modal_participation=spectral["minimum_modal_participation"],
            )
            airfoils_by_node = tuple(
                resolve_airfoil(controller.airfoils, airfoil_id)
                for airfoil_id in controller._node_airfoil_ids
            )
            modal_participation = np.linalg.norm(
                modal_basis.real_shape_body_per_m,
                axis=1,
            )
            self.spectral_source = HVAWTSpectralLockInSource(
                node_ids=node_ids,
                airfoils_by_node=airfoils_by_node,
                component_ids_by_node=component_ids_by_node,
                modes=self.spectral_modes,
                modal_participation=modal_participation,
                n_frequency_entries=int(controller.n_freq_depth),
                fluid_density_kgpm3=float(controller.fluid_density),
                fluid_dynamic_viscosity_pas=float(controller.fluid_dynamicviscosity),
                source_bounds_policy=str(controller.source_bounds_policy),
                source_frequency_cap_hz=controller.source_frequency_cap_hz,
                sample_period_s=resolved_sample_period_s,
                capture=capture,
            )
            self._hybrid_config_fingerprint = _semantic_sha256(
                {
                    "raw_config": raw_config,
                    "modal_basis_fingerprint": modal_basis.fingerprint,
                    "spectral_source_fingerprint": (
                        self.spectral_source.config_fingerprint
                    ),
                }
            )
        if engineering_model:
            active_airfoils = tuple(
                resolve_airfoil(controller.airfoils, airfoil_id)
                for airfoil_id in controller._node_airfoil_ids
            )
            source_identities = _runtime_airfoil_source_identities(
                runtime,
                active_airfoils,
            )
            mismatches = sorted(
                {
                    row["raw_sha256"]
                    for row in source_identities
                    if row["raw_sha256"] != transfer.static_spectrum_sha256
                }
            )
            if mismatches:
                raise ValueError(
                    "Active H-VAWT airfoil HDF5 source differs from the "
                    "engineering calibration source identity."
                )
            capture = SpectralLockInCaptureConfig(
                schema_version=2,
                capture_window=transfer.capture_window,
                reference_strouhal=1.0,
                reference_lift_amplitude=1.0,
                frequency_ratio_low=transfer.frequency_ratio_low,
                frequency_ratio_peak=transfer.frequency_ratio_peak,
                frequency_ratio_high=transfer.frequency_ratio_high,
                maximum_spectral_gain=engineering["maximum_spectral_gain"],
                minimum_modal_participation=engineering["minimum_modal_participation"],
            )
            modal_participation = np.linalg.norm(
                modal_basis.real_shape_body_per_m,
                axis=1,
            )
            self.spectral_source = HVAWTSpectralLockInSource(
                node_ids=node_ids,
                airfoils_by_node=active_airfoils,
                component_ids_by_node=component_ids_by_node,
                modes=self.spectral_modes,
                modal_participation=modal_participation,
                n_frequency_entries=int(controller.n_freq_depth),
                fluid_density_kgpm3=float(controller.fluid_density),
                fluid_dynamic_viscosity_pas=float(controller.fluid_dynamicviscosity),
                source_bounds_policy=str(controller.source_bounds_policy),
                source_frequency_cap_hz=controller.source_frequency_cap_hz,
                sample_period_s=resolved_sample_period_s,
                capture=capture,
            )
            self.residual_source_scale = float(engineering["residual_source_scale"])
            self.maximum_power_closure_relative_error = _strict_float(
                engineering["maximum_power_closure_relative_error"],
                "engineering_lockin.maximum_power_closure_relative_error",
            )
            if self.maximum_power_closure_relative_error <= 0.0:
                raise ValueError(
                    "maximum_power_closure_relative_error must be positive."
                )
            self.engineering_force_scale = float(engineering["response_force_scale"])
            self._hybrid_config_fingerprint = _semantic_sha256(
                {
                    "raw_config": raw_config,
                    "modal_basis_fingerprint": modal_basis.fingerprint,
                    "calibration_fingerprint": transfer.calibration_fingerprint,
                    "calibration_sha256": calibration_sha256,
                    "unified_gain_model_sha256": unified_gain_model_sha256,
                    "unified_gain_model_fingerprint": (
                        None
                        if unified_gain_state is None
                        else unified_gain_state.unified_model_fingerprint
                    ),
                    "active_airfoil_sources": source_identities,
                    "spectral_source_fingerprint": (
                        self.spectral_source.config_fingerprint
                    ),
                }
            )
        if griffin_model:
            active_airfoils = tuple(
                resolve_airfoil(controller.airfoils, airfoil_id)
                for airfoil_id in controller._node_airfoil_ids
            )
            source_identities = _runtime_airfoil_source_identities(
                runtime,
                active_airfoils,
            )
            expected_static_sha = str(griffin["static_spectrum_sha256"]).lower()
            mismatches = sorted(
                {
                    row["raw_sha256"]
                    for row in source_identities
                    if row["raw_sha256"] != expected_static_sha
                }
            )
            if mismatches:
                raise ValueError(
                    "Active H-VAWT airfoil HDF5 source differs from the "
                    "reciprocal-Griffin static-spectrum identity."
                )
            capture = SpectralLockInCaptureConfig(
                schema_version=2,
                capture_window="compact_sine_squared",
                reference_strouhal=1.0,
                reference_lift_amplitude=1.0,
                frequency_ratio_low=self.griffin_parameters.frequency_ratio_low,
                frequency_ratio_peak=self.griffin_parameters.frequency_ratio_peak,
                frequency_ratio_high=self.griffin_parameters.frequency_ratio_high,
                maximum_spectral_gain=griffin["maximum_spectral_gain"],
                minimum_modal_participation=griffin["minimum_modal_participation"],
            )
            modal_participation = np.linalg.norm(
                modal_basis.real_shape_body_per_m,
                axis=1,
            )
            self.spectral_source = HVAWTSpectralLockInSource(
                node_ids=node_ids,
                airfoils_by_node=active_airfoils,
                component_ids_by_node=component_ids_by_node,
                modes=self.spectral_modes,
                modal_participation=modal_participation,
                n_frequency_entries=int(controller.n_freq_depth),
                fluid_density_kgpm3=float(controller.fluid_density),
                fluid_dynamic_viscosity_pas=float(controller.fluid_dynamicviscosity),
                source_bounds_policy=str(controller.source_bounds_policy),
                source_frequency_cap_hz=controller.source_frequency_cap_hz,
                sample_period_s=resolved_sample_period_s,
                capture=capture,
            )
            self.residual_source_scale = 0.0
            self.maximum_power_closure_relative_error = _strict_float(
                griffin["maximum_power_closure_relative_error"],
                "griffin_lockin.maximum_power_closure_relative_error",
            )
            if self.maximum_power_closure_relative_error <= 0.0:
                raise ValueError(
                    "maximum_power_closure_relative_error must be positive."
                )
            self.griffin_static_spectrum_sha256 = expected_static_sha
            self.griffin_force_scale = float(griffin["response_force_scale"])
            self._hybrid_config_fingerprint = _semantic_sha256(
                {
                    "raw_config": raw_config,
                    "modal_basis_fingerprint": modal_basis.fingerprint,
                    "griffin_calibration_fingerprint": (
                        self.griffin_calibration_fingerprint
                    ),
                    "griffin_calibration_sha256": griffin_calibration_sha256,
                    "active_airfoil_sources": source_identities,
                    "spectral_source_fingerprint": (
                        self.spectral_source.config_fingerprint
                    ),
                }
            )
        self.last_result = None
        self._pending_source_coverage: tuple[np.ndarray, ...] | None = None

    @property
    def config_fingerprint(self) -> str:
        if self._hybrid_config_fingerprint is not None:
            return self._hybrid_config_fingerprint
        return self.core.config.fingerprint

    def step(
        self,
        swap: np.ndarray,
        *,
        response_time_s: float | None = None,
        ambient_velocity_global_mps: Any | None = None,
    ) -> np.ndarray:
        timing = self.performance_timing
        started_ns = timing.now_ns() if timing is not None else 0
        layout = self.runtime.layout
        time_value = (
            float(swap[layout[self.runtime.time_block].offset])
            if response_time_s is None
            else float(response_time_s)
        )
        azimuth = float(swap[layout[self.runtime.azimuth_block].offset])
        rpm = float(swap[self._rpm_spec.offset])
        velocity_spec = layout[self.runtime.velocity_block]
        velocity = np.asarray(
            swap[velocity_spec.offset : velocity_spec.offset + velocity_spec.size],
            dtype=float,
        ).reshape((len(self.node_ids), 3))
        position = np.asarray(
            swap[
                self._position_spec.offset : self._position_spec.offset
                + self._position_spec.size
            ],
            dtype=float,
        ).reshape((len(self.node_ids), 3))
        rotation = self.core.config.kinematics.rotation_body_to_global(azimuth)
        controller = self.runtime.controller
        chord_unit_global = np.asarray(controller._base_chord, dtype=float) @ rotation.T
        normal_unit_global = (
            np.asarray(controller._base_normal, dtype=float) @ rotation.T
        )
        chord_m = np.asarray(controller._chord, dtype=float)
        thickness_ratio = np.asarray(controller._thickness, dtype=float)
        local_length_m = np.asarray(controller._local_length, dtype=float)
        ambient_velocity = np.asarray(
            self.runtime._ambient_velocity_global
            if ambient_velocity_global_mps is None
            else ambient_velocity_global_mps,
            dtype=float,
        )
        spectral_gain = None
        residual_force = np.zeros((len(self.node_ids), 3), dtype=float)
        engineering_force = np.zeros((len(self.node_ids), 3), dtype=float)
        griffin_force = np.zeros((len(self.node_ids), 3), dtype=float)
        kinematics = None
        flows = None
        spectral_transaction = None
        previous_result = self.last_result
        previous_spectral_result = self.last_spectral_result
        previous_damping_balance = self.last_damping_balance
        previous_engineering_diagnostics = self.last_engineering_diagnostics
        previous_griffin_diagnostics = self.last_griffin_diagnostics
        previous_total_maximum_nodal_force_n = (
            self.last_total_maximum_nodal_force_n
        )
        if timing is not None:
            timing.record_since(
                "adapter_state_reads_and_geometry_conversion",
                started_ns,
            )
        started_ns = timing.now_ns() if timing is not None else 0
        core_transaction = self.core.transaction_snapshot()
        if timing is not None:
            timing.record_since("response_core_checkpoint_preparation", started_ns)
            started_ns = timing.now_ns()
        try:
            if self.spectral_source is not None:
                spectral_transaction = self.spectral_source.transaction_snapshot()
                if timing is not None:
                    timing.record_since(
                        "spectral_transaction_snapshot",
                        started_ns,
                    )
                    started_ns = timing.now_ns()
                kinematics = decompose_fixed_axis_velocity(
                    self.core.config.kinematics,
                    positions_global_m=position,
                    total_structural_velocity_global_mps=velocity,
                    lss_rpm=rpm,
                    azimuth_deg=azimuth,
                )
                flows = separate_hvawt_flows(
                    ambient_velocity_global_mps=ambient_velocity,
                    total_structural_velocity_global_mps=velocity,
                    rigid_body_velocity_global_mps=(
                        kinematics.rigid_body_velocity_global_mps
                    ),
                    chord_unit_global=chord_unit_global,
                    normal_unit_global=normal_unit_global,
                    chord_m=chord_m,
                    thickness_ratio=thickness_ratio,
                )
                if timing is not None:
                    timing.record_since("fixed_axis_kinematics_and_local_flow", started_ns)
                    self.spectral_source.performance_timing = timing
                spectral_result = self.spectral_source.step(
                    time_value,
                    base_relative_velocity_global_mps=(
                        flows.response_relative_velocity_global_mps
                        if (
                            self.transfer is not None
                            or self.griffin_parameters is not None
                        )
                        else flows.base_relative_velocity_global_mps
                    ),
                    response_effective_crossflow_speed_mps=(
                        flows.response_effective_crossflow_speed_mps
                    ),
                    response_section_width_m=flows.response_section_width_m,
                    chord_unit_global=chord_unit_global,
                    normal_unit_global=normal_unit_global,
                    chord_m=chord_m,
                    thickness_ratio=thickness_ratio,
                    local_length_m=local_length_m,
                    structural_velocity_global_mps=velocity,
                )
                spectral_gain = spectral_result.partition.spectral_gain_by_node_mode
                residual_force = self.residual_source_scale * np.asarray(
                    spectral_result.residual_force_global_n,
                    dtype=float,
                )
                response_prefactor = (
                    0.5
                    * float(controller.fluid_density)
                    * np.asarray(flows.response_effective_crossflow_speed_mps)
                    * np.asarray(flows.response_section_width_m)
                    * local_length_m
                )
                if self.transfer is None:
                    self.last_damping_balance = evaluate_modal_damping_balance(
                        partition=spectral_result.partition,
                        modes=self.spectral_modes,
                        real_mode_shape=(
                            self.core.config.modal_basis.real_shape_body_per_m
                        ),
                        section_dynamic_prefactor_nspm=response_prefactor,
                    )
                self.last_spectral_result = spectral_result
            started_ns = timing.now_ns() if timing is not None else 0
            precomputed_core_state = (
                None
                if kinematics is None or flows is None
                else self.core._bind_precomputed_step_state(
                    node_ids=self.node_ids,
                    positions_global_m=position,
                    total_structural_velocity_global_mps=velocity,
                    lss_rpm=rpm,
                    azimuth_deg=azimuth,
                    ambient_velocity_global_mps=ambient_velocity,
                    chord_unit_global=chord_unit_global,
                    normal_unit_global=normal_unit_global,
                    chord_m=chord_m,
                    thickness_ratio=thickness_ratio,
                    kinematics=kinematics,
                    flows=flows,
                )
            )
            result = self.core.step(
                time_value,
                node_ids=self.node_ids,
                positions_global_m=position,
                total_structural_velocity_global_mps=velocity,
                lss_rpm=rpm,
                azimuth_deg=azimuth,
                ambient_velocity_global_mps=ambient_velocity,
                chord_unit_global=chord_unit_global,
                normal_unit_global=normal_unit_global,
                chord_m=chord_m,
                thickness_ratio=thickness_ratio,
                local_length_m=local_length_m,
                fluid_density_kgpm3=float(controller.fluid_density),
                spectral_gain_by_node_mode=spectral_gain,
                limit_companion_force_global_n=(
                    residual_force if self.spectral_source is not None else None
                ),
                _precomputed_state=precomputed_core_state,
            )
            if timing is not None:
                timing.record_since(
                    "modal_filtering_rms_negative_damping_and_saturation",
                    started_ns,
                )
            if self.transfer is not None:
                started_ns = timing.now_ns() if timing is not None else 0
                mode_diagnostics = result.diagnostics.mode_diagnostics
                engineering_force, engineering_diagnostics = (
                    calculate_hvawt_engineering_force(
                        modal_basis=self.core.config.modal_basis,
                        gain_state=(
                            self.unified_gain_state
                            if self.unified_gain_state is not None
                            else self.transfer
                        ),
                        spectral_result=spectral_result,
                        mode_configs=self.core.config.modes,
                        structural_damping_ratios=self.structural_damping_ratios,
                        rotation_body_to_global=rotation,
                        response_relative_velocity_global_mps=(
                            result.response_relative_velocity_global_mps
                        ),
                        response_effective_crossflow_speed_mps=(
                            flows.response_effective_crossflow_speed_mps
                        ),
                        chord_unit_global=chord_unit_global,
                        normal_unit_global=normal_unit_global,
                        chord_m=chord_m,
                        local_length_m=local_length_m,
                        fluid_density_kgpm3=float(controller.fluid_density),
                        flexible_velocity_global_mps=(
                            result.flexible_velocity_global_mps
                        ),
                        raw_qdot_mps=[
                            row.corrected_raw_qdot_mps for row in mode_diagnostics
                        ],
                        filtered_qdot_mps=[
                            row.filtered_qdot_mps for row in mode_diagnostics
                        ],
                        qdot_rms_mps=[row.qdot_rms_mps for row in mode_diagnostics],
                        maximum_power_closure_relative_error=(
                            self.maximum_power_closure_relative_error
                        ),
                        response_force_scale=self.engineering_force_scale,
                    )
                )
                self.last_engineering_diagnostics = engineering_diagnostics
                if timing is not None:
                    timing.record_since(
                        "engineering_generalized_force_and_force_adjoint",
                        started_ns,
                    )
            if self.griffin_parameters is not None:
                started_ns = timing.now_ns() if timing is not None else 0
                mode_diagnostics = result.diagnostics.mode_diagnostics
                griffin_force, griffin_diagnostics = calculate_hvawt_griffin_force(
                    modal_basis=self.core.config.modal_basis,
                    parameters=self.griffin_parameters,
                    calibration_fingerprint=self.griffin_calibration_fingerprint,
                    static_spectrum_sha256=self.griffin_static_spectrum_sha256,
                    spectral_result=spectral_result,
                    mode_configs=self.core.config.modes,
                    structural_damping_ratios=self.structural_damping_ratios,
                    rotation_body_to_global=rotation,
                    response_relative_velocity_global_mps=(
                        result.response_relative_velocity_global_mps
                    ),
                    response_effective_crossflow_speed_mps=(
                        flows.response_effective_crossflow_speed_mps
                    ),
                    chord_unit_global=chord_unit_global,
                    normal_unit_global=normal_unit_global,
                    chord_m=chord_m,
                    local_length_m=local_length_m,
                    fluid_density_kgpm3=float(controller.fluid_density),
                    flexible_velocity_global_mps=(result.flexible_velocity_global_mps),
                    raw_qdot_mps=[
                        row.corrected_raw_qdot_mps for row in mode_diagnostics
                    ],
                    filtered_qdot_mps=[
                        row.filtered_qdot_mps for row in mode_diagnostics
                    ],
                    maximum_power_closure_relative_error=(
                        self.maximum_power_closure_relative_error
                    ),
                    response_force_scale=self.griffin_force_scale,
                )
                self.last_griffin_diagnostics = griffin_diagnostics
                if timing is not None:
                    timing.record_since(
                        "engineering_generalized_force_and_force_adjoint",
                        started_ns,
                    )
            started_ns = timing.now_ns() if timing is not None else 0
            total_force = (
                np.asarray(result.additive_force_global_n, dtype=float)
                + residual_force
                + engineering_force
                + griffin_force
            )
            maximum_force = float(np.max(np.linalg.norm(total_force, axis=1)))
            if not math.isfinite(maximum_force):
                raise FloatingPointError(
                    "H-VAWT total nodal force must remain finite."
                )
            if (
                self.max_abs_total_nodal_force_n is not None
                and maximum_force > self.max_abs_total_nodal_force_n
            ):
                raise ValueError(
                    f"H-VAWT total nodal force {maximum_force:.17g} N exceeds "
                    f"{self.max_abs_total_nodal_force_n:.17g} N."
                )
            if timing is not None:
                timing.record_since(
                    "total_force_limits_virtual_work_and_energy_closure",
                    started_ns,
                )
        except Exception:
            self.core.restore_transaction_snapshot(core_transaction)
            self.last_result = previous_result
            self.last_engineering_diagnostics = previous_engineering_diagnostics
            self.last_griffin_diagnostics = previous_griffin_diagnostics
            self.last_total_maximum_nodal_force_n = (
                previous_total_maximum_nodal_force_n
            )
            if self.spectral_source is not None and spectral_transaction is not None:
                self.spectral_source.restore_transaction_snapshot(spectral_transaction)
                self.last_spectral_result = previous_spectral_result
                self.last_damping_balance = previous_damping_balance
            raise
        self.last_result = result
        self.last_total_maximum_nodal_force_n = maximum_force
        if self.spectral_source is not None:
            response_force = (
                np.asarray(result.additive_force_global_n, dtype=float)
                + engineering_force
                + griffin_force
            )
            self._pending_source_coverage = (
                np.array(residual_force, copy=True),
                np.array(response_force, copy=True),
                np.array(total_force, copy=True),
                # Coverage and applied-force power use the same total
                # structural velocity supplied by QBlade and by the spectral
                # source.  The flexible-only velocity omits rigid 1-RPM
                # motion and is not the established source-coverage meaning.
                np.array(velocity, copy=True),
            )
        else:
            self._pending_source_coverage = None
        return total_force

    def commit_source_coverage(self, *, step_index: int) -> None:
        """Commit receipt diagnostics after the wrapper step fully succeeds."""

        if self.spectral_source is None or self._pending_source_coverage is None:
            return
        residual, response, total, velocity = self._pending_source_coverage
        self.spectral_source.commit_source_coverage(
            sample_index=step_index,
            applied_residual_force_global_n=residual,
            response_force_global_n=response,
            total_force_global_n=total,
            structural_velocity_global_mps=velocity,
        )

    def source_coverage(self) -> list[dict[str, Any]]:
        if self.spectral_source is None:
            return []
        return self.spectral_source.source_coverage_summary()

    def checkpoint_state(self) -> dict[str, Any]:
        state = {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "config_fingerprint": self.config_fingerprint,
            "core": self.core.checkpoint_state(),
        }
        if self.spectral_source is not None:
            state["spectral_source"] = self.spectral_source.checkpoint_state()
        return state

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("H-VAWT SIL response checkpoint must be a mapping.")
        expected = {"schema_version", "model_type", "config_fingerprint", "core"}
        if self.spectral_source is not None:
            expected.add("spectral_source")
        _exact_keys(state, expected, "H-VAWT SIL response checkpoint")
        if (
            state["schema_version"] != self.schema_version
            or state["model_type"] != self.model_type
            or state["config_fingerprint"] != self.config_fingerprint
        ):
            raise ValueError("H-VAWT SIL response checkpoint identity mismatch.")
        spectral_transaction = (
            None
            if self.spectral_source is None
            else self.spectral_source.transaction_snapshot()
        )
        if self.spectral_source is not None:
            self.spectral_source.restore_checkpoint_state(state["spectral_source"])
        try:
            self.core.restore_checkpoint_state(state["core"])
        except Exception:
            if self.spectral_source is not None and spectral_transaction is not None:
                self.spectral_source.restore_transaction_snapshot(spectral_transaction)
            raise
        self.last_result = None
        self.last_spectral_result = None
        self._pending_source_coverage = None
        self.last_damping_balance = ()
        self.last_engineering_diagnostics = None
        self.last_griffin_diagnostics = None
        self.last_total_maximum_nodal_force_n = 0.0

    def history_fields(self) -> dict[str, object]:
        diagnostics = self.core.last_diagnostics
        if diagnostics is None:
            return {}
        values = asdict(diagnostics)
        mode_rows = values.pop("mode_diagnostics")
        fields: dict[str, object] = {
            "response_model_type": self.model_type,
            "response_total_maximum_nodal_force_n": (
                self.last_total_maximum_nodal_force_n
            ),
        }
        for name, value in values.items():
            fields[f"response_{name}"] = value
        for mode_row in mode_rows:
            mode = int(mode_row.pop("tracked_mode"))
            for name, value in mode_row.items():
                fields[f"response_mode{mode}_{name}"] = value
        spectral = self.last_spectral_result
        if spectral is not None:
            partition = spectral.partition
            fields.update(
                {
                    "response_spectral_source_config_fingerprint": (
                        self.spectral_source.config_fingerprint
                    ),
                    "response_spectral_residual_source_scale": (
                        self.residual_source_scale
                    ),
                    "response_spectral_maximum_residual_nodal_force_n": (
                        spectral.maximum_residual_nodal_force_n
                    ),
                    "response_spectral_residual_force_power_w": (
                        spectral.residual_force_power_w
                    ),
                    "response_spectral_applied_residual_maximum_nodal_force_n": (
                        self.residual_source_scale
                        * spectral.maximum_residual_nodal_force_n
                    ),
                    "response_spectral_applied_residual_force_power_w": (
                        self.residual_source_scale * spectral.residual_force_power_w
                    ),
                    "response_spectral_source_energy_sum": float(
                        np.sum(partition.total_source_energy_by_node)
                    ),
                    "response_spectral_residual_energy_sum": float(
                        np.sum(partition.total_residual_energy_by_node)
                    ),
                    "response_spectral_applied_residual_lift_energy_sum": float(
                        np.sum(spectral.applied_residual_lift_energy_by_node)
                    ),
                    "response_spectral_truncated_residual_lift_energy_sum": float(
                        np.sum(spectral.truncated_residual_lift_energy_by_node)
                    ),
                    "response_spectral_truncated_residual_drag_energy_sum": float(
                        np.sum(spectral.truncated_residual_drag_energy_by_node)
                    ),
                    "response_spectral_captured_energy_sum": float(
                        np.sum(partition.total_captured_energy_by_node)
                    ),
                    "response_spectral_maximum_energy_closure_error": (
                        partition.maximum_energy_closure_error
                    ),
                }
            )
            for mode_index, mode in enumerate(self.spectral_modes):
                gain = partition.spectral_gain_by_node_mode[:, mode_index]
                centroid = partition.source_centroid_frequency_hz_by_node_mode[
                    :, mode_index
                ]
                active = gain > 0.0
                fields[f"response_mode{mode.tracked_mode}_spectral_gain_min"] = float(
                    np.min(gain)
                )
                fields[f"response_mode{mode.tracked_mode}_spectral_gain_max"] = float(
                    np.max(gain)
                )
                fields[
                    f"response_mode{mode.tracked_mode}_captured_source_frequency_min_hz"
                ] = (float(np.min(centroid[active])) if np.any(active) else 0.0)
                fields[
                    f"response_mode{mode.tracked_mode}_captured_source_frequency_max_hz"
                ] = (float(np.max(centroid[active])) if np.any(active) else 0.0)
                fields[f"response_mode{mode.tracked_mode}_pulled_frequency_hz"] = (
                    mode.natural_frequency_hz
                )
            for balance in self.last_damping_balance:
                for name, value in asdict(balance).items():
                    if name != "tracked_mode":
                        fields[f"response_mode{balance.tracked_mode}_{name}"] = value
        engineering = self.last_engineering_diagnostics
        if engineering is not None:
            values = asdict(engineering)
            mode_rows = values.pop("mode_diagnostics")
            for name, value in values.items():
                fields[f"response_engineering_{name}"] = value
            for mode_row in mode_rows:
                mode = int(mode_row.pop("tracked_mode"))
                for name, value in mode_row.items():
                    fields[f"response_mode{mode}_engineering_{name}"] = value
        griffin = self.last_griffin_diagnostics
        if griffin is not None:
            values = asdict(griffin)
            mode_rows = values.pop("mode_diagnostics")
            for name, value in values.items():
                fields[f"response_griffin_{name}"] = value
            for mode_row in mode_rows:
                mode = int(mode_row.pop("tracked_mode"))
                for name, value in mode_row.items():
                    fields[f"response_mode{mode}_griffin_{name}"] = value
        return fields

    def diagnostics_summary(self) -> dict[str, object]:
        diagnostics = self.core.last_diagnostics
        if diagnostics is None:
            return {
                "model_type": self.model_type,
                "config_fingerprint": self.config_fingerprint,
                "samples": 0,
            }
        result = {
            "model_type": self.model_type,
            "config_fingerprint": self.config_fingerprint,
            "samples": 1,
            "last": asdict(diagnostics),
            "total_maximum_nodal_force_n": self.last_total_maximum_nodal_force_n,
        }
        if self.last_spectral_result is not None:
            result["spectral"] = {
                "source_config_fingerprint": self.spectral_source.config_fingerprint,
                "residual_source_scale": self.residual_source_scale,
                "maximum_residual_nodal_force_n": (
                    self.last_spectral_result.maximum_residual_nodal_force_n
                ),
                "residual_force_power_w": (
                    self.last_spectral_result.residual_force_power_w
                ),
                "applied_residual_maximum_nodal_force_n": (
                    self.residual_source_scale
                    * self.last_spectral_result.maximum_residual_nodal_force_n
                ),
                "applied_residual_force_power_w": (
                    self.residual_source_scale
                    * self.last_spectral_result.residual_force_power_w
                ),
                "source_energy_sum": float(
                    np.sum(
                        self.last_spectral_result.partition.total_source_energy_by_node
                    )
                ),
                "residual_energy_sum": float(
                    np.sum(
                        self.last_spectral_result.partition.total_residual_energy_by_node
                    )
                ),
                "applied_residual_lift_energy_sum": float(
                    np.sum(
                        self.last_spectral_result.applied_residual_lift_energy_by_node
                    )
                ),
                "truncated_residual_lift_energy_sum": float(
                    np.sum(
                        self.last_spectral_result.truncated_residual_lift_energy_by_node
                    )
                ),
                "truncated_residual_drag_energy_sum": float(
                    np.sum(
                        self.last_spectral_result.truncated_residual_drag_energy_by_node
                    )
                ),
                "captured_energy_sum": float(
                    np.sum(
                        self.last_spectral_result.partition.total_captured_energy_by_node
                    )
                ),
                "damping_balance": [
                    asdict(value) for value in self.last_damping_balance
                ],
            }
        if self.last_engineering_diagnostics is not None:
            result["engineering_lockin"] = asdict(self.last_engineering_diagnostics)
        if self.last_griffin_diagnostics is not None:
            result["reciprocal_griffin_lockin"] = asdict(self.last_griffin_diagnostics)
        return result


__all__ = [
    "HVAWT_SIL_RESPONSE_MODEL_TYPE",
    "HVAWT_SIL_RESPONSE_SCHEMA_VERSION",
    "HVAWT_SIL_HYBRID_MODEL_TYPE",
    "HVAWT_SIL_HYBRID_SCHEMA_VERSION",
    "HVAWT_SIL_ENGINEERING_MODEL_TYPE",
    "HVAWT_SIL_ENGINEERING_SCHEMA_VERSION",
    "HVAWT_SIL_GRIFFIN_MODEL_TYPE",
    "HVAWT_SIL_GRIFFIN_SCHEMA_VERSION",
    "HVAWTEngineeringModeDiagnostics",
    "HVAWTEngineeringStepDiagnostics",
    "HVAWTGriffinModeDiagnostics",
    "HVAWTGriffinStepDiagnostics",
    "HVAWTSilResponseAdapter",
    "calculate_hvawt_engineering_force",
    "calculate_hvawt_griffin_force",
]
