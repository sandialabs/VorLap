"""Generic one-mode SIL adapter for the airfoil-spectral lock-in bridge.

This adapter is the stationary-section counterpart of the audited H-VAWT
adapter.  It uses the same interpolated airfoil spectra, energy-conserving
modal capture, frequency pull, residual lift/drag reconstruction, and generic
schema-v2 modal response state.  Native QBlade aerodynamics remains enabled;
all forces returned here are incremental fluctuating VIV loads.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .force_model import resolve_airfoil, section_flow_state, strouhal_reference_length
from .qblade_response_model import (
    AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW,
    QBladeResponseModel,
    RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW,
)
from .qblade_spectral_lockin import (
    HVAWTSpectralLockInSource,
    SpectralLockInCaptureConfig,
    SpectralLockInModeConfig,
)


MODEL_TYPE = "single_mode_airfoil_spectral_lockin_hybrid"


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("rotation axis must be finite and nonzero.")
    x, y, z = axis / norm
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    d = 1.0 - c
    return np.array(
        [
            [c + x * x * d, x * y * d - z * s, x * z * d + y * s],
            [y * x * d + z * s, c + y * y * d, y * z * d - x * s],
            [z * x * d - y * s, z * y * d + x * s, c + z * z * d],
        ],
        dtype=float,
    )


class QBladeSingleModeSpectralSilAdapter:
    """Apply the latest spectral bridge to an arbitrary one-mode QBlade beam."""

    def __init__(
        self,
        *,
        raw_config: Mapping[str, Any],
        parameter_file: Path,
        runtime: Any,
    ) -> None:
        del parameter_file  # retained for the common SIL-adapter construction API
        if not isinstance(raw_config, Mapping):
            raise TypeError("single_mode_spectral_lockin must be a mapping.")
        expected = {"schema_version", "type", "structural_damping_ratio", "spectral_lockin"}
        if set(raw_config) != expected:
            raise ValueError("single_mode_spectral_lockin keys differ from the schema.")
        if raw_config["schema_version"] != 1 or raw_config["type"] != MODEL_TYPE:
            raise ValueError("Unsupported generic single-mode spectral adapter schema/type.")

        controller = runtime.controller
        response_model = controller.response_model
        if not isinstance(response_model, QBladeResponseModel):
            raise ValueError("The generic spectral adapter requires QBladeResponseModel.")
        if (
            response_model.schema_version
            != RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
            or response_model.aerodynamic_speed_basis
            != AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW
            or not response_model.config.enabled
        ):
            raise ValueError("The generic spectral adapter requires enabled response schema v2.")
        if float(controller.force_scale) != 0.0 or bool(controller.include_mean_load):
            raise ValueError("The generic spectral adapter requires force_scale=0 and no mean load.")

        spectral = raw_config["spectral_lockin"]
        expected_spectral = {
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
        }
        if not isinstance(spectral, Mapping) or set(spectral) != expected_spectral:
            raise ValueError("single-mode spectral_lockin keys differ from the schema.")
        if (
            spectral["schema_version"] != 1
            or spectral["source_channel"] != "CL"
            or spectral["retain_uncaptured_lift_spectrum"] is not True
            or spectral["retain_fluctuating_drag_spectrum"] is not True
            or spectral["include_mean_load"] is not False
        ):
            raise ValueError("The single-mode bridge requires CL capture, CL/CD residuals, and no mean load.")
        residual_scale = float(spectral["residual_source_scale"])
        if residual_scale != 0.0:
            raise ValueError(
                "The generic stationary-airfoil adapter requires residual_source_scale=0; "
                "captured CL energy is stretched into the measured mode while native QBlade "
                "supplies baseline aerodynamics. Flow-axis residual-force semantics are not "
                "yet qualified for this orientation."
            )
        damping = float(raw_config["structural_damping_ratio"])
        if not math.isfinite(damping) or not 0.0 <= damping < 1.0:
            raise ValueError("structural_damping_ratio must be finite in [0, 1).")

        self.runtime = runtime
        self.response_model = response_model
        self.node_ids = tuple(str(value) for value in controller.node_ids)
        if response_model.config.projection_weights is None:
            raise ValueError(
                "The generic spectral adapter requires projection_weights as nodal mass quadrature [kg]."
            )
        mode_shape = np.asarray(response_model._mode_shape, dtype=float)
        weights = np.asarray(response_model.config.projection_weights, dtype=float)
        generalized_mass = float(np.dot(weights * mode_shape, mode_shape))
        mode = SpectralLockInModeConfig(
            tracked_mode=1,
            natural_frequency_hz=float(response_model.config.natural_frequency_hz),
            generalized_modal_mass_kg=generalized_mass,
            structural_damping_ratio=damping,
            reference_length_m=float(response_model.config.reference_length_m),
            aerodynamic_damping_gain_h1=float(
                response_model.config.aerodynamic_damping_gain_h1
            ),
            saturation_amplitude_ratio=float(
                response_model.config.saturation_amplitude_ratio
            ),
        )
        capture = SpectralLockInCaptureConfig(
            schema_version=int(spectral["schema_version"]),
            reference_strouhal=float(spectral["reference_strouhal"]),
            reference_lift_amplitude=float(spectral["reference_lift_amplitude"]),
            frequency_ratio_low=float(spectral["frequency_ratio_low"]),
            frequency_ratio_peak=float(spectral["frequency_ratio_peak"]),
            frequency_ratio_high=float(spectral["frequency_ratio_high"]),
            maximum_spectral_gain=float(spectral["maximum_spectral_gain"]),
            minimum_modal_participation=float(spectral["minimum_modal_participation"]),
        )
        airfoils = tuple(
            resolve_airfoil(controller.airfoils, airfoil_id)
            for airfoil_id in controller._node_airfoil_ids
        )
        self.spectral_source = HVAWTSpectralLockInSource(
            node_ids=self.node_ids,
            airfoils_by_node=airfoils,
            modes=(mode,),
            modal_participation=np.abs(mode_shape)[:, None],
            n_frequency_entries=int(controller.n_freq_depth),
            fluid_density_kgpm3=float(controller.fluid_density),
            fluid_dynamic_viscosity_pas=float(controller.fluid_dynamicviscosity),
            source_bounds_policy=str(controller.source_bounds_policy),
            source_frequency_cap_hz=controller.source_frequency_cap_hz,
            capture=capture,
        )
        self.residual_source_scale = residual_scale
        self.model_type = MODEL_TYPE
        self.schema_version = 1
        self.config_fingerprint = _semantic_sha256(
            {
                "raw_config": raw_config,
                "response_model_fingerprint": response_model.config_fingerprint,
                "spectral_source_fingerprint": self.spectral_source.config_fingerprint,
            }
        )
        self.last_spectral_result = None
        self.last_result = None

    def step(
        self, swap: np.ndarray, *, response_time_s: float | None = None,
        ambient_velocity_global_mps: Any | None = None,
    ) -> np.ndarray:
        layout = self.runtime.layout
        time_value = (
            float(swap[layout[self.runtime.time_block].offset])
            if response_time_s is None
            else float(response_time_s)
        )
        azimuth = float(swap[layout[self.runtime.azimuth_block].offset])
        velocity_spec = layout[self.runtime.velocity_block]
        velocity = np.asarray(
            swap[velocity_spec.offset : velocity_spec.offset + velocity_spec.size],
            dtype=float,
        ).reshape((len(self.node_ids), 3))
        controller = self.runtime.controller
        rotation = _axis_angle(controller.rotation_axis, math.radians(azimuth))
        chord = np.asarray(controller._base_chord, dtype=float) @ rotation.T
        normal = np.asarray(controller._base_normal, dtype=float) @ rotation.T
        chord_m = np.asarray(controller._chord, dtype=float)
        thickness = np.asarray(controller._thickness, dtype=float)
        local_length = np.asarray(controller._local_length, dtype=float)
        ambient = np.broadcast_to(
            np.asarray(
                self.runtime._ambient_velocity_global
                if ambient_velocity_global_mps is None
                else ambient_velocity_global_mps,
                dtype=float,
            ),
            velocity.shape,
        )
        relative = ambient - velocity
        ambient_flow = section_flow_state(
            ambient,
            chord,
            normal,
            chord_m,
            local_length,
            float(controller.fluid_density),
            float(controller.fluid_dynamicviscosity),
        )
        relative_flow = section_flow_state(
            relative,
            chord,
            normal,
            chord_m,
            local_length,
            float(controller.fluid_density),
            float(controller.fluid_dynamicviscosity),
        )
        response_width = strouhal_reference_length(
            chord_m, ambient_flow.aoa_deg, thickness
        )
        spectral_snapshot = self.spectral_source.transaction_snapshot()
        response_snapshot = self.response_model.checkpoint_state()
        previous_spectral = self.last_spectral_result
        previous_result = self.last_result
        try:
            spectral_result = self.spectral_source.step(
                time_value,
                base_relative_velocity_global_mps=relative,
                response_effective_crossflow_speed_mps=(
                    ambient_flow.effective_velocity_mps
                ),
                response_section_width_m=response_width,
                chord_unit_global=chord,
                normal_unit_global=normal,
                chord_m=chord_m,
                thickness_ratio=thickness,
                local_length_m=local_length,
                structural_velocity_global_mps=velocity,
            )
            direction = (
                normal
                if self.response_model.config.force_direction == "normal"
                else chord
            )
            response_result = self.response_model.step(
                time_value,
                node_ids=self.node_ids,
                direction_basis=direction,
                structural_velocity_mps=velocity,
                effective_velocity_mps=ambient_flow.effective_velocity_mps,
                section_width_m=response_width,
                local_length_m=local_length,
                fluid_density_kgpm3=float(controller.fluid_density),
                declared_ambient_crossflow_speed_mps=(
                    ambient_flow.effective_velocity_mps
                ),
                relative_crossflow_speed_mps=relative_flow.effective_velocity_mps,
                response_section_width_basis="declared_ambient_crossflow_angle",
                activation_override=(
                    spectral_result.partition.spectral_gain_by_node_mode[:, 0]
                ),
            )
        except Exception:
            self.spectral_source.restore_transaction_snapshot(spectral_snapshot)
            self.response_model.restore_checkpoint_state(response_snapshot)
            self.last_spectral_result = previous_spectral
            self.last_result = previous_result
            raise
        self.last_spectral_result = spectral_result
        self.last_result = response_result
        return (
            np.asarray(response_result.additive_force_n, dtype=float)
            + self.residual_source_scale
            * np.asarray(spectral_result.residual_force_global_n, dtype=float)
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_type": self.model_type,
            "config_fingerprint": self.config_fingerprint,
            "spectral_source": self.spectral_source.checkpoint_state(),
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        expected = {"schema_version", "model_type", "config_fingerprint", "spectral_source"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Single-mode spectral checkpoint keys differ.")
        if (
            state["schema_version"] != 1
            or state["model_type"] != self.model_type
            or state["config_fingerprint"] != self.config_fingerprint
        ):
            raise ValueError("Single-mode spectral checkpoint identity mismatch.")
        self.spectral_source.restore_checkpoint_state(state["spectral_source"])
        self.last_spectral_result = None
        self.last_result = None

    def history_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {"response_model_type": self.model_type}
        if self.last_result is not None:
            for name, value in asdict(self.last_result.diagnostics).items():
                fields[f"response_{name}"] = value
        if self.last_spectral_result is not None:
            partition = self.last_spectral_result.partition
            gain = partition.spectral_gain_by_node_mode[:, 0]
            fields.update(
                {
                    "response_spectral_source_config_fingerprint": (
                        self.spectral_source.config_fingerprint
                    ),
                    "response_spectral_residual_source_scale": self.residual_source_scale,
                    "response_spectral_maximum_residual_nodal_force_n": (
                        self.last_spectral_result.maximum_residual_nodal_force_n
                    ),
                    "response_mode1_spectral_gain_min": float(np.min(gain)),
                    "response_mode1_spectral_gain_max": float(np.max(gain)),
                    "response_spectral_maximum_energy_closure_error": (
                        partition.maximum_energy_closure_error
                    ),
                }
            )
        return fields

    def diagnostics_summary(self) -> dict[str, object]:
        """Return the immutable run-summary block expected by the SIL runner."""

        if self.last_result is None:
            return {
                "model_type": self.model_type,
                "config_fingerprint": self.config_fingerprint,
                "samples": 0,
            }
        result: dict[str, object] = {
            "model_type": self.model_type,
            "config_fingerprint": self.config_fingerprint,
            "samples": 1,
            "last": asdict(self.last_result.diagnostics),
        }
        spectral = self.last_spectral_result
        if spectral is not None:
            partition = spectral.partition
            result["spectral"] = {
                "source_config_fingerprint": self.spectral_source.config_fingerprint,
                "residual_source_scale": self.residual_source_scale,
                "maximum_residual_nodal_force_n": (
                    spectral.maximum_residual_nodal_force_n
                ),
                "residual_force_power_w": spectral.residual_force_power_w,
                "source_energy_sum": float(
                    np.sum(partition.total_source_energy_by_node)
                ),
                "residual_energy_sum": float(
                    np.sum(partition.total_residual_energy_by_node)
                ),
                "captured_energy_sum": float(
                    np.sum(partition.total_captured_energy_by_node)
                ),
                "maximum_energy_closure_error": (
                    partition.maximum_energy_closure_error
                ),
            }
        return result


__all__ = ["MODEL_TYPE", "QBladeSingleModeSpectralSilAdapter"]
