"""One-mode QBlade SIL adapter for engineering spectral negative damping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .force_model import resolve_airfoil, section_flow_state, strouhal_reference_length
from .interpolation import interpolate_fft_spectrum
from .qblade_engineering_lockin import (
    EngineeringLockInCalibration,
    MODEL_TYPE,
    captured_spectral_coefficient,
    semantic_sha256,
)
from .qblade_geometry_blended_lockin import (
    GeometryBlendedCalibration,
    MODEL_TYPE as GEOMETRY_BLENDED_MODEL_TYPE,
)
from .qblade_modal_power_model import (
    ExternalModalPowerTarget,
    QBladeModalPowerModel,
    QBladeModalPowerModelConfig,
)
from .qblade_response_model import (
    AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW,
    QBladeResponseModel,
    RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW,
)
from .qblade_single_mode_work_map_sil import (
    _lift_unit,
    _mode_shape_in_node_order,
    _resolve_relative_file,
    _runtime_airfoil_source_identities,
    _sha256_file,
)
from .qblade_spectral_lockin import (
    HVAWTSpectralLockInSource,
    SpectralLockInCaptureConfig,
    SpectralLockInModeConfig,
)

TRANSFER_MODEL_TYPE = "single_mode_airfoil_engineering_lockin_transfer_v2"
GEOMETRY_TRANSFER_MODEL_TYPE = (
    "single_mode_airfoil_geometry_blended_complete_transfer_v3"
)
FU_TRAINING_REPRODUCTION_MODEL_TYPE = (
    "single_mode_airfoil_geometry_blended_fu_training_reproduction_v4"
)
FU_NATIVE_CORRECTED_REPRODUCTION_MODEL_TYPE = (
    "single_mode_airfoil_geometry_blended_fu_native_corrected_reproduction_v5"
)
FU_SIGNED_WORK_SHAPE_REPRODUCTION_MODEL_TYPE = (
    "single_mode_airfoil_geometry_blended_fu_signed_work_shape_reproduction_v6"
)


@dataclass(frozen=True)
class EngineeringLockInDiagnostics:
    calibration_fingerprint: str
    static_spectrum_fingerprint: str
    incidence_min_deg: float
    incidence_max_deg: float
    reynolds_min: float
    reynolds_max: float
    spectrum_reynolds_clip_low_nodes: int
    spectrum_reynolds_clip_high_nodes: int
    spectrum_incidence_clip_low_nodes: int
    spectrum_incidence_clip_high_nodes: int
    captured_lift_coefficient_min: float
    captured_lift_coefficient_max: float
    captured_lift_coefficient_weighted_mean: float
    captured_active_harmonics_min: int
    captured_active_harmonics_max: int
    captured_centroid_frequency_ratio_min: float
    captured_centroid_frequency_ratio_max: float
    spectral_stretch_gain: float
    zero_amplitude_feedback_coefficient_kgps: float
    structural_damping_coefficient_kgps: float
    negative_to_structural_damping_ratio: float
    native_total_to_structural_damping_ratio: float
    native_total_damping_coefficient_kgps: float
    negative_to_native_total_damping_ratio: float
    linear_lockin_available: bool
    limiting_amplitude_m: float
    predicted_equilibrium_amplitude_m: float
    predicted_equilibrium_amplitude_ratio: float
    current_saturation_factor: float
    target_aerodynamic_power_w: float
    realized_structural_power_w: float


def _strict_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


class QBladeSingleModeEngineeringLockInSilAdapter:
    """Apply static-spectrum-conditioned negative damping to one fixed mode."""

    def __init__(
        self,
        *,
        raw_config: Mapping[str, Any],
        parameter_file: Path,
        runtime: Any,
    ) -> None:
        if not isinstance(raw_config, Mapping):
            raise TypeError("single_mode_engineering_lockin must be a mapping.")
        schema_version = int(raw_config.get("schema_version", -1))
        model_type = str(raw_config.get("type", ""))
        expected = {
            "schema_version",
            "type",
            "profile",
            "kinematics",
            "calibration",
            "static_spectrum",
            "modal_power",
            "structural_damping_ratio",
            "residual_source_scale",
        }
        geometry_v3 = schema_version == 3 and model_type == GEOMETRY_TRANSFER_MODEL_TYPE
        fu_reproduction_v4 = (
            schema_version == 4 and model_type == FU_TRAINING_REPRODUCTION_MODEL_TYPE
        )
        fu_native_corrected_v5 = (
            schema_version == 5
            and model_type == FU_NATIVE_CORRECTED_REPRODUCTION_MODEL_TYPE
        )
        fu_signed_work_shape_v6 = (
            schema_version == 6
            and model_type == FU_SIGNED_WORK_SHAPE_REPRODUCTION_MODEL_TYPE
        )
        fu_reproduction_contract = (
            fu_reproduction_v4
            or fu_native_corrected_v5
            or fu_signed_work_shape_v6
        )
        if geometry_v3 or fu_reproduction_contract:
            expected.add("trained_condition")
        if fu_reproduction_contract:
            expected.add("published_tip_trigger")
        if fu_native_corrected_v5 or fu_signed_work_shape_v6:
            expected.add("native_damping_calibration")
        if fu_signed_work_shape_v6:
            expected.add("signed_work_shape_transfer")
        if set(raw_config) != expected:
            raise ValueError(
                "single_mode_engineering_lockin keys differ from the schema."
            )
        schema_version = int(raw_config["schema_version"])
        model_type = str(raw_config["type"])
        legacy_contract = schema_version == 1 and model_type == MODEL_TYPE
        transfer_contract = schema_version == 2 and model_type == TRANSFER_MODEL_TYPE
        geometry_contract = geometry_v3 or fu_reproduction_contract
        complete_contract = transfer_contract or geometry_contract
        if not legacy_contract and not complete_contract:
            raise ValueError("Unsupported engineering lock-in schema/type.")
        if raw_config["kinematics"] != "fixed_nonrotating":
            raise ValueError(
                "Engineering lock-in v1 supports fixed non-rotating sections."
            )
        residual_source_scale = float(raw_config["residual_source_scale"])
        if residual_source_scale not in ({0.0} if legacy_contract else {0.0, 1.0}):
            raise ValueError(
                "Engineering lock-in v1 requires residual_source_scale=0; "
                "transfer v2 requires residual_source_scale exactly 0 or 1."
            )

        controller = runtime.controller
        seed_model = controller.response_model
        if not isinstance(seed_model, QBladeResponseModel):
            raise ValueError("Engineering lock-in requires the schema-v2 seed model.")
        if (
            seed_model.schema_version
            != RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
            or seed_model.aerodynamic_speed_basis
            != AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW
            or not seed_model.config.enabled
        ):
            raise ValueError("Engineering lock-in requires enabled response schema v2.")
        if float(seed_model.config.aerodynamic_damping_gain_h1) != 0.0:
            raise ValueError(
                "Cylinder H1 must be exactly zero on engineering lock-in cases."
            )
        if seed_model.config.projection_weights is None:
            raise ValueError(
                "Engineering lock-in requires modal mass projection weights."
            )
        if str(seed_model.config.force_direction) not in {"chord", "normal"}:
            raise ValueError(
                "Engineering lock-in mode direction must be chord or normal."
            )
        if float(controller.force_scale) != 0.0 or bool(controller.include_mean_load):
            raise ValueError(
                "Engineering lock-in requires force_scale=0 and no mean source load."
            )

        calibration_raw = raw_config["calibration"]
        if not isinstance(calibration_raw, Mapping) or set(calibration_raw) != {
            "path",
            "sha256",
        }:
            raise ValueError("calibration keys differ from the schema.")
        calibration_path = _resolve_relative_file(
            Path(parameter_file).resolve().parent,
            calibration_raw["path"],
            "calibration.path",
        )
        if _sha256_file(calibration_path) != str(calibration_raw["sha256"]).lower():
            raise ValueError("Calibration SHA-256 differs from staged bytes.")
        if geometry_contract:
            calibration = GeometryBlendedCalibration.from_json(calibration_path)
            if raw_config["profile"] != "FFA-W3_family":
                raise ValueError("Geometry transfer requires profile=FFA-W3_family.")
            parameters = calibration.parameters
            capture_bounds = parameters.frequency_bounds(
                parameters.airfoil_breadth_depth_ratio
            )
            feedback_gain = parameters.airfoil_velocity_feedback_gain
            limiting_amplitude_ratio = parameters.airfoil_limiting_amplitude_ratio
        else:
            calibration = EngineeringLockInCalibration.from_json(calibration_path)
            if calibration.raw["profile"] != raw_config["profile"]:
                raise ValueError("Configured profile differs from calibration profile.")
            capture_bounds = calibration.capture_bounds
            feedback_gain = calibration.feedback_gain
            limiting_amplitude_ratio = calibration.limiting_amplitude_ratio

        trained_condition = None
        if geometry_contract:
            trained_raw = raw_config["trained_condition"]
            trained_keys = {
                "path",
                "sha256",
                "case_id",
                "wind_speed_mps",
                "zero_amplitude_feedback_over_structural_damping",
                "limiting_amplitude_m",
            }
            if not isinstance(trained_raw, Mapping) or set(trained_raw) != trained_keys:
                raise ValueError("trained_condition keys differ from the schema.")
            trained_path = _resolve_relative_file(
                Path(parameter_file).resolve().parent,
                trained_raw["path"],
                "trained_condition.path",
            )
            trained_sha = _sha256_file(trained_path)
            if trained_sha != str(trained_raw["sha256"]).lower():
                raise ValueError(
                    "Trained-condition source SHA-256 differs from staged bytes."
                )
            trained_model = json.loads(trained_path.read_text(encoding="utf-8"))
            if (
                trained_model.get("schema")
                != "vorlap-turbine-geometry-composed-negative-damping-v1"
            ):
                raise ValueError("Unsupported trained-condition source schema.")
            rows = trained_model.get("training", {}).get(
                "fu2026_speed_orientation_rows_after_preserved_failure", []
            )
            matches = [
                row for row in rows if row.get("case_id") == trained_raw["case_id"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    "trained_condition.case_id is not unique in the frozen source."
                )
            row = matches[0]
            required = {
                "wind_speed_mps": _strict_positive(
                    trained_raw["wind_speed_mps"], "trained_condition.wind_speed_mps"
                ),
                "zero_amplitude_feedback_over_structural_damping": _strict_positive(
                    trained_raw["zero_amplitude_feedback_over_structural_damping"],
                    "trained_condition.zero_amplitude_feedback_over_structural_damping",
                ),
                "configuration_saturation_amplitude_m": _strict_positive(
                    trained_raw["limiting_amplitude_m"],
                    "trained_condition.limiting_amplitude_m",
                ),
            }
            for key, value in required.items():
                if not math.isclose(
                    float(row[key]), value, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    raise ValueError(
                        f"trained_condition.{key} differs from the frozen row."
                    )
            trained_condition = {
                "source_sha256": trained_sha,
                "source_model_fingerprint": str(trained_model["model_fingerprint"]),
                "case_id": str(trained_raw["case_id"]),
                "wind_speed_mps": required["wind_speed_mps"],
                "feedback_ratio": required[
                    "zero_amplitude_feedback_over_structural_damping"
                ],
                "limiting_amplitude_m": required[
                    "configuration_saturation_amplitude_m"
                ],
            }

        native_total_to_structural_damping_ratio = 1.0
        native_damping_calibration = None
        if fu_native_corrected_v5 or fu_signed_work_shape_v6:
            native_raw = raw_config["native_damping_calibration"]
            native_keys = {
                "path",
                "sha256",
                "case_id",
                "native_total_to_structural_damping_ratio",
            }
            if not isinstance(native_raw, Mapping) or set(native_raw) != native_keys:
                raise ValueError("native_damping_calibration keys differ from schema.")
            native_path = _resolve_relative_file(
                Path(parameter_file).resolve().parent,
                native_raw["path"],
                "native_damping_calibration.path",
            )
            native_sha = _sha256_file(native_path)
            if native_sha != str(native_raw["sha256"]).lower():
                raise ValueError(
                    "Native-damping calibration SHA-256 differs from staged bytes."
                )
            native_document = json.loads(native_path.read_text(encoding="utf-8"))
            if (
                native_document.get("schema")
                != "vorlap-fu2026-qblade-native-damping-calibration-v1"
                or native_document.get("status") != "qualified"
            ):
                raise ValueError("Native-damping calibration is not qualified.")
            unsigned_native = dict(native_document)
            stored_native_fingerprint = unsigned_native.pop("fingerprint", None)
            if stored_native_fingerprint != semantic_sha256(unsigned_native):
                raise ValueError("Native-damping calibration fingerprint changed.")
            native_case_id = str(native_raw["case_id"])
            native_rows = native_document.get("cases")
            if not isinstance(native_rows, list):
                raise ValueError("Native-damping calibration cases must be a list.")
            native_matches = [
                row for row in native_rows if str(row.get("case_id")) == native_case_id
            ]
            if len(native_matches) != 1:
                raise ValueError(
                    "native_damping_calibration.case_id is not unique in source."
                )
            native_row = native_matches[0]
            native_total_to_structural_damping_ratio = _strict_positive(
                native_raw["native_total_to_structural_damping_ratio"],
                "native_damping_calibration.native_total_to_structural_damping_ratio",
            )
            if not math.isclose(
                float(native_row["native_total_to_structural_damping_ratio"]),
                native_total_to_structural_damping_ratio,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "Configured native-damping ratio differs from calibrated row."
                )
            if (
                trained_condition is None
                or native_case_id != trained_condition["case_id"]
            ):
                raise ValueError(
                    "Native-damping calibration and trained condition case IDs differ."
                )
            native_damping_calibration = {
                "source_sha256": native_sha,
                "source_fingerprint": str(stored_native_fingerprint),
                "case_id": native_case_id,
                "native_total_to_structural_damping_ratio": (
                    native_total_to_structural_damping_ratio
                ),
            }

        signed_work_shape_transfer = None
        if fu_signed_work_shape_v6:
            shape_raw = raw_config["signed_work_shape_transfer"]
            shape_keys = {
                "path",
                "sha256",
                "case_id",
                "active_branch",
                "equilibrium_amplitude_m",
                "feedback_over_opposing_damping",
                "limiting_amplitude_m",
            }
            if not isinstance(shape_raw, Mapping) or set(shape_raw) != shape_keys:
                raise ValueError("signed_work_shape_transfer keys differ from schema.")
            shape_path = _resolve_relative_file(
                Path(parameter_file).resolve().parent,
                shape_raw["path"],
                "signed_work_shape_transfer.path",
            )
            shape_sha = _sha256_file(shape_path)
            if shape_sha != str(shape_raw["sha256"]).lower():
                raise ValueError("Signed-work-shape source SHA-256 differs from staged bytes.")
            shape_document = json.loads(shape_path.read_text(encoding="utf-8"))
            if (
                shape_document.get("schema")
                != "vorlap-fu2026-iea10-signed-work-shape-transfer-v1"
                or shape_document.get("status")
                != "qualified_training_reproduction_only"
            ):
                raise ValueError("Signed-work-shape transfer is not qualified.")
            unsigned_shape = dict(shape_document)
            stored_shape_fingerprint = unsigned_shape.pop("model_fingerprint", None)
            if stored_shape_fingerprint != semantic_sha256(unsigned_shape):
                raise ValueError("Signed-work-shape transfer fingerprint changed.")
            shape_case_id = str(shape_raw["case_id"])
            shape_rows = shape_document.get("cases")
            if not isinstance(shape_rows, list):
                raise ValueError("Signed-work-shape transfer cases must be a list.")
            shape_matches = [
                row for row in shape_rows if str(row.get("case_id")) == shape_case_id
            ]
            if len(shape_matches) != 1:
                raise ValueError(
                    "signed_work_shape_transfer.case_id is not unique in source."
                )
            shape_row = shape_matches[0]
            if (
                trained_condition is None
                or native_damping_calibration is None
                or shape_case_id != trained_condition["case_id"]
                or shape_case_id != native_damping_calibration["case_id"]
            ):
                raise ValueError(
                    "Signed-work shape, native damping, and trained row IDs differ."
                )
            active_branch = bool(shape_raw["active_branch"])
            if active_branch is not bool(shape_row["active_branch"]):
                raise ValueError("Configured active-branch flag differs from trained row.")
            equilibrium_amplitude = _strict_positive(
                shape_raw["equilibrium_amplitude_m"],
                "signed_work_shape_transfer.equilibrium_amplitude_m",
            )
            limiting_amplitude = _strict_positive(
                shape_raw["limiting_amplitude_m"],
                "signed_work_shape_transfer.limiting_amplitude_m",
            )
            feedback_over_opposing = float(
                shape_raw["feedback_over_opposing_damping"]
            )
            if (
                not math.isfinite(feedback_over_opposing)
                or feedback_over_opposing < 0.0
                or (active_branch and feedback_over_opposing <= 1.0)
                or (not active_branch and feedback_over_opposing != 0.0)
            ):
                raise ValueError("Signed-work-shape feedback ratio is invalid.")
            expected_shape_values = {
                "equilibrium_amplitude_m": equilibrium_amplitude,
                "transferred_feedback_over_opposing_damping": feedback_over_opposing,
                "transferred_limiting_amplitude_m": limiting_amplitude,
            }
            for key, value in expected_shape_values.items():
                if not math.isclose(
                    float(shape_row[key]), value, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    raise ValueError(
                        f"signed_work_shape_transfer.{key} differs from trained row."
                    )
            if not math.isclose(
                equilibrium_amplitude,
                float(shape_row["equilibrium_amplitude_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("Signed-work-shape equilibrium amplitude changed.")
            if (
                shape_document.get("sources", {})
                .get("geometry_composed_training_model", {})
                .get("sha256")
                != trained_condition["source_sha256"]
            ):
                raise ValueError(
                    "Signed-work shape is not bound to the executed trained model."
                )
            signed_work_shape_transfer = {
                "source_sha256": shape_sha,
                "model_fingerprint": str(stored_shape_fingerprint),
                "case_id": shape_case_id,
                "active_branch": active_branch,
                "equilibrium_amplitude_m": equilibrium_amplitude,
                "feedback_over_opposing_damping": feedback_over_opposing,
                "limiting_amplitude_m": limiting_amplitude,
            }

        tip_trigger = None
        if fu_reproduction_contract:
            trigger_raw = raw_config["published_tip_trigger"]
            trigger_keys = {
                "schema_version",
                "enabled",
                "node_id",
                "peak_force_n",
                "ramp_up_end_s",
                "ramp_down_end_s",
                "direction",
                "source",
            }
            if not isinstance(trigger_raw, Mapping) or set(trigger_raw) != trigger_keys:
                raise ValueError("published_tip_trigger keys differ from the schema.")
            if int(trigger_raw["schema_version"]) != 1:
                raise ValueError("published_tip_trigger.schema_version must be 1.")
            if not isinstance(trigger_raw["enabled"], bool):
                raise TypeError("published_tip_trigger.enabled must be boolean.")
            if trigger_raw["enabled"] is not True:
                raise ValueError(
                    "Fu training reproduction requires the trigger enabled."
                )
            peak_force_n = _strict_positive(
                trigger_raw["peak_force_n"], "published_tip_trigger.peak_force_n"
            )
            ramp_up_end_s = _strict_positive(
                trigger_raw["ramp_up_end_s"],
                "published_tip_trigger.ramp_up_end_s",
            )
            ramp_down_end_s = _strict_positive(
                trigger_raw["ramp_down_end_s"],
                "published_tip_trigger.ramp_down_end_s",
            )
            if ramp_down_end_s <= ramp_up_end_s:
                raise ValueError(
                    "published_tip_trigger.ramp_down_end_s must exceed ramp_up_end_s."
                )
            if (
                peak_force_n != 8000.0
                or ramp_up_end_s != 10.0
                or ramp_down_end_s != 11.0
            ):
                raise ValueError(
                    "Fu Method-4 requires exactly 8000 N, 10 s ramp-up, and "
                    "11 s ramp-down end."
                )
            if trigger_raw["direction"] != "selected_edgewise_mode_direction":
                raise ValueError("published_tip_trigger.direction is unsupported.")
            if trigger_raw["source"] != "Fu2026_Method4_Table5":
                raise ValueError("published_tip_trigger.source is unsupported.")
            tip_trigger = {
                "enabled": bool(trigger_raw["enabled"]),
                "node_id": str(trigger_raw["node_id"]),
                "peak_force_n": peak_force_n,
                "ramp_up_end_s": ramp_up_end_s,
                "ramp_down_end_s": ramp_down_end_s,
            }

        spectrum = raw_config["static_spectrum"]
        expected_spectrum = {
            "channel",
            "bounds_policy",
            "source_sha256",
            "n_frequency_entries",
        }
        if complete_contract:
            expected_spectrum |= {
                "maximum_spectral_gain",
                "minimum_modal_participation",
                "retain_uncaptured_lift_spectrum",
                "retain_fluctuating_drag_spectrum",
                "include_mean_load",
            }
        if geometry_contract:
            expected_spectrum.remove("source_sha256")
            expected_spectrum.add("source_sha256_by_airfoil")
        if not isinstance(spectrum, Mapping) or set(spectrum) != expected_spectrum:
            raise ValueError("static_spectrum keys differ from the schema.")
        spectrum_bounds_policy = "clip" if fu_reproduction_contract else "raise"
        if (
            spectrum["channel"] != "CL"
            or spectrum["bounds_policy"] != spectrum_bounds_policy
        ):
            raise ValueError(
                "Engineering lock-in spectrum channel/bounds policy differs "
                "from the active schema."
            )
        if complete_contract and (
            spectrum["retain_uncaptured_lift_spectrum"] is not True
            or spectrum["retain_fluctuating_drag_spectrum"] is not True
            or spectrum["include_mean_load"] is not False
        ):
            raise ValueError(
                "Complete composition requires retained CL/CD fluctuations and "
                "no supplemental mean load."
            )
        n_frequency_entries = int(spectrum["n_frequency_entries"])
        if n_frequency_entries < 2:
            raise ValueError("At least two spectral entries are required.")
        configured_spectrum_sha = None
        configured_spectrum_by_airfoil = None
        if geometry_contract:
            raw_sources = spectrum["source_sha256_by_airfoil"]
            if not isinstance(raw_sources, Mapping) or not raw_sources:
                raise ValueError("source_sha256_by_airfoil must be a nonempty mapping.")
            configured_spectrum_by_airfoil = {
                str(name): str(value).lower() for name, value in raw_sources.items()
            }
            if any(
                len(value) != 64 for value in configured_spectrum_by_airfoil.values()
            ):
                raise ValueError(
                    "Every configured spectrum SHA-256 must have 64 hex characters."
                )
        else:
            configured_spectrum_sha = str(spectrum["source_sha256"]).lower()
            calibration_spectrum_sha = str(
                calibration.raw["sources"]["stationary_spectrum"]["sha256"]
            ).lower()
            if configured_spectrum_sha != calibration_spectrum_sha:
                raise ValueError("Static spectrum differs from calibration source.")

        damping = float(raw_config["structural_damping_ratio"])
        if not math.isfinite(damping) or not 0.0 <= damping < 1.0:
            raise ValueError("structural_damping_ratio must be finite in [0,1).")
        modal = raw_config["modal_power"]
        expected_modal = {
            "rms_time_constant_s",
            "bandpass_damping_ratio",
            "max_abs_target_power_w",
            "max_abs_generalized_force_n",
            "max_abs_nodal_force_n",
            "limit_action",
            "enabled",
        }
        if not isinstance(modal, Mapping) or set(modal) != expected_modal:
            raise ValueError("modal_power keys differ from the schema.")
        if modal["limit_action"] != "error" or not isinstance(modal["enabled"], bool):
            raise ValueError(
                "Engineering qualification requires boolean enabled and error limits."
            )

        node_ids = tuple(str(value) for value in controller.node_ids)
        if tip_trigger is not None:
            if node_ids.count(tip_trigger["node_id"]) != 1:
                raise ValueError(
                    "published_tip_trigger.node_id must identify exactly one runtime node."
                )
            tip_trigger["node_index"] = node_ids.index(tip_trigger["node_id"])
        airfoils = tuple(
            resolve_airfoil(controller.airfoils, airfoil_id)
            for airfoil_id in controller._node_airfoil_ids
        )
        source_identities = _runtime_airfoil_source_identities(runtime, airfoils)
        if geometry_contract:
            active_names = {str(source["airfoil_name"]) for source in source_identities}
            if set(configured_spectrum_by_airfoil) != active_names or any(
                source["raw_sha256"]
                != configured_spectrum_by_airfoil[str(source["airfoil_name"])]
                for source in source_identities
            ):
                raise ValueError(
                    "Runtime FFA source set differs from configured spectra."
                )
        elif any(
            source["raw_sha256"] != configured_spectrum_sha
            for source in source_identities
        ):
            raise ValueError(
                "Runtime airfoil source differs from the calibrated spectrum."
            )
        static_fingerprint = semantic_sha256(
            {
                "source_identities": list(source_identities),
                "source_sha256": configured_spectrum_sha,
                "source_sha256_by_airfoil": configured_spectrum_by_airfoil,
                "n_frequency_entries": n_frequency_entries,
                "bounds_policy": spectrum_bounds_policy,
            }
        )
        source_fingerprint = semantic_sha256(
            {
                "model_type": MODEL_TYPE,
                "calibration_fingerprint": calibration.fingerprint,
                "static_spectrum_fingerprint": static_fingerprint,
            }
        )
        mode_shape = tuple(float(value) for value in seed_model.config.mode_shape)
        mass_weights = tuple(
            float(value) for value in seed_model.config.projection_weights
        )
        power_model = QBladeModalPowerModel(
            QBladeModalPowerModelConfig(
                node_ids=node_ids,
                mode_shape=mode_shape,
                modal_mass_weights_kg=mass_weights,
                aerodynamic_tributary_weights=mass_weights,
                natural_frequency_hz=float(seed_model.config.natural_frequency_hz),
                power_curve=None,
                rms_time_constant_s=_strict_positive(
                    modal["rms_time_constant_s"], "modal_power.rms_time_constant_s"
                ),
                max_abs_target_power_w=_strict_positive(
                    modal["max_abs_target_power_w"],
                    "modal_power.max_abs_target_power_w",
                ),
                max_abs_generalized_force_n=_strict_positive(
                    modal["max_abs_generalized_force_n"],
                    "modal_power.max_abs_generalized_force_n",
                ),
                max_abs_nodal_force_n=_strict_positive(
                    modal["max_abs_nodal_force_n"], "modal_power.max_abs_nodal_force_n"
                ),
                bandpass_damping_ratio=_strict_positive(
                    modal["bandpass_damping_ratio"],
                    "modal_power.bandpass_damping_ratio",
                ),
                force_direction=str(seed_model.config.force_direction),
                limit_action="error",
                enabled=modal["enabled"],
                external_target_source_fingerprint=source_fingerprint,
            )
        )

        self.runtime = runtime
        self.seed_model = seed_model
        self.power_model = power_model
        self.calibration = calibration
        self.node_ids = node_ids
        self.airfoils = airfoils
        self.n_frequency_entries = n_frequency_entries
        self.static_spectrum_source_sha256 = configured_spectrum_sha
        self.static_spectrum_source_sha256_by_airfoil = configured_spectrum_by_airfoil
        self.static_spectrum_fingerprint = static_fingerprint
        self.spectrum_bounds_policy = spectrum_bounds_policy
        self.structural_damping_ratio = damping
        self.velocity_scale = 1.0 if complete_contract else calibration.velocity_scale
        self.model_type = model_type
        self.schema_version = schema_version
        self.residual_source_scale = residual_source_scale
        self._mode_shape = _mode_shape_in_node_order(power_model, node_ids)
        self._generalized_mass_kg = float(power_model._projection_denominator)
        self._reference_length_m = float(seed_model.config.reference_length_m)
        self._limiting_amplitude_m = limiting_amplitude_ratio * self._reference_length_m
        if trained_condition is not None:
            self._limiting_amplitude_m = float(
                trained_condition["limiting_amplitude_m"]
            )
        if signed_work_shape_transfer is not None:
            self._limiting_amplitude_m = float(
                signed_work_shape_transfer["limiting_amplitude_m"]
            )
        self.capture_bounds = tuple(float(value) for value in capture_bounds)
        self.feedback_gain = float(feedback_gain)
        self.limiting_amplitude_ratio = float(limiting_amplitude_ratio)
        self.trained_condition = trained_condition
        self.native_damping_calibration = native_damping_calibration
        self.signed_work_shape_transfer = signed_work_shape_transfer
        self.native_total_to_structural_damping_ratio = (
            native_total_to_structural_damping_ratio
        )
        self.tip_trigger = tip_trigger
        self.spectral_source = None
        if complete_contract:
            if geometry_contract:
                low, peak, high = self.capture_bounds
                capture_window = "compact_sine_squared"
            else:
                transfer = calibration.transferable_parameters
                low = transfer.frequency_ratio_low
                peak = transfer.frequency_ratio_peak
                high = transfer.frequency_ratio_high
                capture_window = transfer.capture_window
            spectral_mode = SpectralLockInModeConfig(
                tracked_mode=1,
                natural_frequency_hz=float(seed_model.config.natural_frequency_hz),
                generalized_modal_mass_kg=self._generalized_mass_kg,
                structural_damping_ratio=damping,
                reference_length_m=self._reference_length_m,
                aerodynamic_damping_gain_h1=0.0,
                saturation_amplitude_ratio=self.limiting_amplitude_ratio,
            )
            spectral_capture = SpectralLockInCaptureConfig(
                schema_version=2,
                capture_window=capture_window,
                reference_strouhal=1.0,
                reference_lift_amplitude=1.0,
                frequency_ratio_low=low,
                frequency_ratio_peak=peak,
                frequency_ratio_high=high,
                maximum_spectral_gain=_strict_positive(
                    spectrum["maximum_spectral_gain"],
                    "static_spectrum.maximum_spectral_gain",
                ),
                minimum_modal_participation=_strict_positive(
                    spectrum["minimum_modal_participation"],
                    "static_spectrum.minimum_modal_participation",
                ),
            )
            self.spectral_source = HVAWTSpectralLockInSource(
                node_ids=node_ids,
                airfoils_by_node=airfoils,
                modes=(spectral_mode,),
                modal_participation=np.abs(self._mode_shape)[:, None],
                n_frequency_entries=n_frequency_entries,
                fluid_density_kgpm3=float(controller.fluid_density),
                fluid_dynamic_viscosity_pas=float(controller.fluid_dynamicviscosity),
                source_bounds_policy=spectrum_bounds_policy,
                source_frequency_cap_hz=controller.source_frequency_cap_hz,
                capture=spectral_capture,
            )
        fingerprint_payload = {
            "raw_config": raw_config,
            "calibration_fingerprint": calibration.fingerprint,
            "static_spectrum_fingerprint": static_fingerprint,
            "seed_model_fingerprint": seed_model.config_fingerprint,
            "modal_power_fingerprint": power_model.checkpoint_state()[
                "config_fingerprint"
            ],
        }
        if self.spectral_source is not None:
            fingerprint_payload["spectral_source_fingerprint"] = (
                self.spectral_source.config_fingerprint
            )
        self.config_fingerprint = semantic_sha256(fingerprint_payload)
        self.last_seed_result = None
        self.last_power_result = None
        self.last_spectral_result = None
        self.last_diagnostics: EngineeringLockInDiagnostics | None = None
        self.last_tip_trigger_active = False
        self.last_tip_trigger_force_n = 0.0
        self.last_tip_trigger_power_w = 0.0

    def _prescribed_tip_trigger_force_n(self, time_s: float) -> float:
        """Return the published Fu Method-4 pull-release force at ``time_s``."""

        trigger = self.tip_trigger
        if trigger is None or not trigger["enabled"] or time_s < 0.0:
            return 0.0
        ramp_up_end_s = float(trigger["ramp_up_end_s"])
        ramp_down_end_s = float(trigger["ramp_down_end_s"])
        peak_force_n = float(trigger["peak_force_n"])
        if time_s <= ramp_up_end_s:
            return peak_force_n * time_s / ramp_up_end_s
        if time_s < ramp_down_end_s:
            return (
                peak_force_n
                * (ramp_down_end_s - time_s)
                / (ramp_down_end_s - ramp_up_end_s)
            )
        return 0.0

    def _captured_coefficients(
        self,
        *,
        flow: Any,
        response_width_m: np.ndarray,
        mode_direction: np.ndarray,
        lift_direction: np.ndarray,
    ) -> tuple[np.ndarray, list[Any], dict[str, int]]:
        low, peak, high = self.capture_bounds
        velocity_scale = self.velocity_scale
        coefficients = np.zeros(len(self.node_ids), dtype=float)
        captures = []
        clip_counts = {
            "reynolds_low": 0,
            "reynolds_high": 0,
            "incidence_low": 0,
            "incidence_high": 0,
        }
        for index, airfoil in enumerate(self.airfoils):
            reynolds = float(flow.reynolds[index])
            incidence = float(flow.aoa_deg[index])
            clip_counts["reynolds_low"] += int(reynolds < float(np.min(airfoil.Re)))
            clip_counts["reynolds_high"] += int(reynolds > float(np.max(airfoil.Re)))
            clip_counts["incidence_low"] += int(incidence < float(np.min(airfoil.AOA)))
            clip_counts["incidence_high"] += int(incidence > float(np.max(airfoil.AOA)))
            st, amplitude, _ = interpolate_fft_spectrum(
                airfoil,
                reynolds,
                incidence,
                "CL",
                n_freq_depth=self.n_frequency_entries,
                bounds_policy=self.spectrum_bounds_policy,
            )
            capture = captured_spectral_coefficient(
                st,
                amplitude,
                effective_speed_mps=(
                    velocity_scale * float(flow.effective_velocity_mps[index])
                ),
                strouhal_reference_length_m=float(response_width_m[index]),
                natural_frequency_hz=float(
                    self.power_model.config.natural_frequency_hz
                ),
                low=low,
                peak=peak,
                high=high,
            )
            projection = abs(
                float(np.dot(mode_direction[index], lift_direction[index]))
            )
            coefficients[index] = projection * capture.coefficient_rss
            captures.append(capture)
        return coefficients, captures, clip_counts

    def _target_evaluator(
        self,
        *,
        zero_amplitude_feedback_kgps: float,
        coefficients: np.ndarray,
        captures: list[Any],
        effective_speed_mps: np.ndarray,
    ):
        omega = 2.0 * math.pi * float(self.power_model.config.natural_frequency_hz)
        limiting = self._limiting_amplitude_m

        def evaluator(amplitude_m: float) -> ExternalModalPowerTarget:
            saturation = 1.0 - (float(amplitude_m) / limiting) ** 2
            velocity_gain = zero_amplitude_feedback_kgps * saturation
            target = 0.5 * velocity_gain * omega**2 * float(amplitude_m) ** 2
            context = semantic_sha256(
                {
                    "calibration_fingerprint": self.calibration.fingerprint,
                    "static_spectrum_fingerprint": self.static_spectrum_fingerprint,
                    "amplitude_m": float(amplitude_m),
                    "limiting_amplitude_m": limiting,
                    "saturation_factor": saturation,
                    "zero_amplitude_feedback_kgps": zero_amplitude_feedback_kgps,
                    "captured_lift_coefficient": [
                        float(value) for value in coefficients
                    ],
                    "captured_spectrum": [asdict(value) for value in captures],
                    "effective_speed_mps": [
                        float(value) for value in effective_speed_mps
                    ],
                }
            )
            evaluator.last_saturation = saturation
            return ExternalModalPowerTarget(
                target, zero_amplitude_feedback_kgps, context
            )

        evaluator.last_saturation = 1.0
        return evaluator

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
        azimuth_remainder = math.remainder(azimuth, 360.0)
        if not math.isclose(azimuth_remainder, 0.0, abs_tol=1.0e-6):
            raise ValueError(
                "Fixed engineering lock-in requires zero azimuth; "
                f"observed remainder={azimuth_remainder:.17g} deg."
            )
        velocity_spec = layout[self.runtime.velocity_block]
        velocity = np.asarray(
            swap[velocity_spec.offset : velocity_spec.offset + velocity_spec.size],
            dtype=float,
        ).reshape((len(self.node_ids), 3))
        controller = self.runtime.controller
        chord = np.asarray(controller._base_chord, dtype=float)
        normal = np.asarray(controller._base_normal, dtype=float)
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
        flow = section_flow_state(
            ambient,
            chord,
            normal,
            chord_m,
            local_length,
            float(controller.fluid_density),
            float(controller.fluid_dynamicviscosity),
        )
        relative_flow = section_flow_state(
            ambient - velocity,
            chord,
            normal,
            chord_m,
            local_length,
            float(controller.fluid_density),
            float(controller.fluid_dynamicviscosity),
        )
        response_width = strouhal_reference_length(chord_m, flow.aoa_deg, thickness)
        lift_direction = _lift_unit(
            chord,
            normal,
            flow.velocity_chord_mps,
            flow.velocity_normal_mps,
            flow.effective_velocity_mps,
        )
        mode_direction = (
            chord if self.power_model.config.force_direction == "chord" else normal
        )
        coefficients, captures, clip_counts = self._captured_coefficients(
            flow=flow,
            response_width_m=response_width,
            mode_direction=mode_direction,
            lift_direction=lift_direction,
        )
        effective_speed = self.velocity_scale * flow.effective_velocity_mps
        spectral_conjugate_weights = (
            0.5
            * float(controller.fluid_density)
            * effective_speed
            * chord_m
            * local_length
            * coefficients
        )
        sectional_spectral_basis = spectral_conjugate_weights * self._mode_shape**2
        spectral_basis = float(np.sum(sectional_spectral_basis))
        spectral_stretch_gain = self.feedback_gain
        zero_feedback = spectral_stretch_gain * spectral_basis
        omega = 2.0 * math.pi * float(self.power_model.config.natural_frequency_hz)
        structural_damping = (
            2.0 * self.structural_damping_ratio * self._generalized_mass_kg * omega
        )
        if self.trained_condition is not None:
            ambient_speed = float(np.linalg.norm(self.runtime._ambient_velocity_global))
            if not math.isclose(
                ambient_speed,
                float(self.trained_condition["wind_speed_mps"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "Runtime ambient speed differs from the frozen trained condition."
                )
            # The trained turbine row supplies the required global work margin,
            # while the fixed-body spectra supply the dimensional sectional
            # capture basis.  Their ratio is the explicit spectrum-stretch gain;
            # a zero captured basis cannot activate the motion-correlated term.
            feedback_ratio = float(self.trained_condition["feedback_ratio"])
            if self.signed_work_shape_transfer is not None:
                feedback_ratio = float(
                    self.signed_work_shape_transfer[
                        "feedback_over_opposing_damping"
                    ]
                )
            desired_feedback = (
                feedback_ratio
                * structural_damping
                * self.native_total_to_structural_damping_ratio
            )
            spectral_stretch_gain = (
                desired_feedback / spectral_basis if spectral_basis > 0.0 else 0.0
            )
            zero_feedback = spectral_stretch_gain * spectral_basis
        evaluator = self._target_evaluator(
            zero_amplitude_feedback_kgps=zero_feedback,
            coefficients=coefficients,
            captures=captures,
            effective_speed_mps=effective_speed,
        )

        seed_snapshot = self.seed_model.checkpoint_state()
        power_snapshot = self.power_model.checkpoint_state()
        spectral_snapshot = (
            None
            if self.spectral_source is None
            else self.spectral_source.transaction_snapshot()
        )
        previous = (
            self.last_seed_result,
            self.last_power_result,
            self.last_spectral_result,
            self.last_diagnostics,
            self.last_tip_trigger_active,
            self.last_tip_trigger_force_n,
            self.last_tip_trigger_power_w,
        )
        try:
            spectral_result = None
            if self.spectral_source is not None:
                spectral_result = self.spectral_source.step(
                    time_value,
                    # The fixed-body spectrum remains tied to the declared
                    # ambient flow. Flexible velocity may exchange work with
                    # the lock-in closure, but it must not retune the residual
                    # source spectrum a second time.
                    base_relative_velocity_global_mps=ambient,
                    response_effective_crossflow_speed_mps=(
                        self.velocity_scale * flow.effective_velocity_mps
                    ),
                    response_section_width_m=response_width,
                    chord_unit_global=chord,
                    normal_unit_global=normal,
                    chord_m=chord_m,
                    thickness_ratio=thickness,
                    local_length_m=local_length,
                    structural_velocity_global_mps=velocity,
                )
                partition_coefficient = (
                    spectral_result.partition.captured_amplitude_by_node_mode[:, 0]
                )
                projection = np.abs(np.sum(mode_direction * lift_direction, axis=1))
                active = np.abs(self._mode_shape) > float(
                    self.spectral_source.capture.minimum_modal_participation
                )
                if not np.allclose(
                    coefficients[active],
                    (projection * partition_coefficient)[active],
                    rtol=2.0e-12,
                    atol=2.0e-13,
                ):
                    raise RuntimeError(
                        "Residual partition and engineering capture differ."
                    )
            seed_result = self.seed_model.step(
                time_value,
                node_ids=self.node_ids,
                direction_basis=mode_direction,
                structural_velocity_mps=velocity,
                effective_velocity_mps=flow.effective_velocity_mps,
                section_width_m=response_width,
                local_length_m=local_length,
                fluid_density_kgpm3=float(controller.fluid_density),
                declared_ambient_crossflow_speed_mps=flow.effective_velocity_mps,
                relative_crossflow_speed_mps=relative_flow.effective_velocity_mps,
                response_section_width_basis="declared_ambient_crossflow_angle",
                activation_override=np.zeros(len(self.node_ids), dtype=float),
            )
            power_result = self.power_model.step(
                time_value,
                node_ids=self.node_ids,
                direction_basis=mode_direction,
                structural_velocity_mps=velocity,
                energy_conjugate_weights=(
                    spectral_conjugate_weights
                    if self.model_type
                    in {
                        FU_TRAINING_REPRODUCTION_MODEL_TYPE,
                        FU_NATIVE_CORRECTED_REPRODUCTION_MODEL_TYPE,
                        FU_SIGNED_WORK_SHAPE_REPRODUCTION_MODEL_TYPE,
                    }
                    and spectral_basis > 0.0
                    else None
                ),
                target_power_evaluator=evaluator,
            )
            power_diag = power_result.diagnostics
            trigger_force_n = self._prescribed_tip_trigger_force_n(time_value)
            trigger_power_w = 0.0
            if self.tip_trigger is not None:
                trigger_index = int(self.tip_trigger["node_index"])
                trigger_power_w = trigger_force_n * float(
                    np.dot(mode_direction[trigger_index], velocity[trigger_index])
                )
            ratio = (
                zero_feedback / structural_damping
                if structural_damping > 0.0
                else math.inf
            )
            native_total_damping = (
                structural_damping * self.native_total_to_structural_damping_ratio
            )
            ratio_to_native_total = (
                zero_feedback / native_total_damping
                if native_total_damping > 0.0
                else math.inf
            )
            equilibrium = (
                self._limiting_amplitude_m
                * math.sqrt(max(0.0, 1.0 - 1.0 / ratio_to_native_total))
                if ratio_to_native_total > 1.0
                else 0.0
            )
            finite_centroids = [
                value.centroid_frequency_ratio
                for value in captures
                if math.isfinite(value.centroid_frequency_ratio)
            ]
            weighted_mean = float(
                np.average(
                    coefficients,
                    weights=np.maximum(local_length * self._mode_shape**2, 1e-30),
                )
            )
            self.last_diagnostics = EngineeringLockInDiagnostics(
                calibration_fingerprint=self.calibration.fingerprint,
                static_spectrum_fingerprint=self.static_spectrum_fingerprint,
                incidence_min_deg=float(np.min(flow.aoa_deg)),
                incidence_max_deg=float(np.max(flow.aoa_deg)),
                reynolds_min=float(np.min(flow.reynolds)),
                reynolds_max=float(np.max(flow.reynolds)),
                spectrum_reynolds_clip_low_nodes=clip_counts["reynolds_low"],
                spectrum_reynolds_clip_high_nodes=clip_counts["reynolds_high"],
                spectrum_incidence_clip_low_nodes=clip_counts["incidence_low"],
                spectrum_incidence_clip_high_nodes=clip_counts["incidence_high"],
                captured_lift_coefficient_min=float(np.min(coefficients)),
                captured_lift_coefficient_max=float(np.max(coefficients)),
                captured_lift_coefficient_weighted_mean=weighted_mean,
                captured_active_harmonics_min=min(
                    value.active_harmonics for value in captures
                ),
                captured_active_harmonics_max=max(
                    value.active_harmonics for value in captures
                ),
                captured_centroid_frequency_ratio_min=(
                    min(finite_centroids) if finite_centroids else math.nan
                ),
                captured_centroid_frequency_ratio_max=(
                    max(finite_centroids) if finite_centroids else math.nan
                ),
                spectral_stretch_gain=spectral_stretch_gain,
                zero_amplitude_feedback_coefficient_kgps=zero_feedback,
                structural_damping_coefficient_kgps=structural_damping,
                negative_to_structural_damping_ratio=ratio,
                native_total_to_structural_damping_ratio=(
                    self.native_total_to_structural_damping_ratio
                ),
                native_total_damping_coefficient_kgps=native_total_damping,
                negative_to_native_total_damping_ratio=ratio_to_native_total,
                linear_lockin_available=bool(ratio_to_native_total > 1.0),
                limiting_amplitude_m=self._limiting_amplitude_m,
                predicted_equilibrium_amplitude_m=equilibrium,
                predicted_equilibrium_amplitude_ratio=equilibrium
                / self._reference_length_m,
                current_saturation_factor=float(evaluator.last_saturation),
                target_aerodynamic_power_w=power_diag.target_cycle_mean_power_w,
                realized_structural_power_w=power_diag.instantaneous_structural_power_w,
            )
        except Exception:
            self.seed_model.restore_checkpoint_state(seed_snapshot)
            self.power_model.restore_checkpoint_state(power_snapshot)
            if self.spectral_source is not None and spectral_snapshot is not None:
                self.spectral_source.restore_transaction_snapshot(spectral_snapshot)
            (
                self.last_seed_result,
                self.last_power_result,
                self.last_spectral_result,
                self.last_diagnostics,
                self.last_tip_trigger_active,
                self.last_tip_trigger_force_n,
                self.last_tip_trigger_power_w,
            ) = previous
            raise
        self.last_seed_result = seed_result
        self.last_power_result = power_result
        self.last_spectral_result = spectral_result
        self.last_tip_trigger_active = bool(trigger_force_n != 0.0)
        self.last_tip_trigger_force_n = trigger_force_n
        self.last_tip_trigger_power_w = trigger_power_w
        residual_force = (
            np.zeros_like(power_result.additive_force_n)
            if spectral_result is None
            else self.residual_source_scale
            * np.asarray(spectral_result.residual_force_global_n, dtype=float)
        )
        trigger_force = np.zeros_like(power_result.additive_force_n)
        if self.tip_trigger is not None:
            trigger_index = int(self.tip_trigger["node_index"])
            trigger_force[trigger_index] = (
                self._prescribed_tip_trigger_force_n(time_value)
                * mode_direction[trigger_index]
            )
        return (
            np.asarray(seed_result.additive_force_n)
            + np.asarray(power_result.additive_force_n)
            + residual_force
            + trigger_force
        )

    def checkpoint_state(self) -> dict[str, Any]:
        state = {
            "schema_version": 1,
            "model_type": self.model_type,
            "config_fingerprint": self.config_fingerprint,
            "seed_model": self.seed_model.checkpoint_state(),
            "modal_power": self.power_model.checkpoint_state(),
        }
        if self.schema_version >= 2:
            state["schema_version"] = self.schema_version
            state["spectral_source"] = self.spectral_source.checkpoint_state()
        return state

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version",
            "model_type",
            "config_fingerprint",
            "seed_model",
            "modal_power",
        }
        if self.schema_version >= 2:
            expected.add("spectral_source")
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Engineering lock-in checkpoint keys differ.")
        if (
            state["schema_version"] != self.schema_version
            or state["model_type"] != self.model_type
            or state["config_fingerprint"] != self.config_fingerprint
        ):
            raise ValueError("Engineering lock-in checkpoint identity mismatch.")
        seed_snapshot = self.seed_model.checkpoint_state()
        power_snapshot = self.power_model.checkpoint_state()
        spectral_snapshot = (
            None
            if self.spectral_source is None
            else self.spectral_source.checkpoint_state()
        )
        try:
            self.seed_model.restore_checkpoint_state(state["seed_model"])
            self.power_model.restore_checkpoint_state(state["modal_power"])
            if self.spectral_source is not None:
                self.spectral_source.restore_checkpoint_state(state["spectral_source"])
        except Exception:
            self.seed_model.restore_checkpoint_state(seed_snapshot)
            self.power_model.restore_checkpoint_state(power_snapshot)
            if self.spectral_source is not None and spectral_snapshot is not None:
                self.spectral_source.restore_checkpoint_state(spectral_snapshot)
            raise
        self.last_seed_result = None
        self.last_power_result = None
        self.last_spectral_result = None
        self.last_diagnostics = None
        self.last_tip_trigger_active = False
        self.last_tip_trigger_force_n = 0.0
        self.last_tip_trigger_power_w = 0.0

    def history_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {"response_model_type": self.model_type}
        if self.last_power_result is not None:
            fields.update(
                {
                    f"response_{key}": value
                    for key, value in asdict(self.last_power_result.diagnostics).items()
                }
            )
        if self.last_diagnostics is not None:
            fields.update(
                {
                    f"response_engineering_{key}": value
                    for key, value in asdict(self.last_diagnostics).items()
                }
            )
        if self.tip_trigger is not None:
            fields.update(
                {
                    "response_engineering_prescribed_tip_trigger_active": (
                        self.last_tip_trigger_active
                    ),
                    "response_engineering_prescribed_tip_trigger_force_n": (
                        self.last_tip_trigger_force_n
                    ),
                    "response_engineering_prescribed_tip_trigger_power_w": (
                        self.last_tip_trigger_power_w
                    ),
                }
            )
        if self.last_spectral_result is not None:
            partition = self.last_spectral_result.partition
            fields.update(
                {
                    "response_spectral_source_config_fingerprint": (
                        self.spectral_source.config_fingerprint
                    ),
                    "response_spectral_residual_source_scale": (
                        self.residual_source_scale
                    ),
                    "response_spectral_maximum_residual_nodal_force_n": (
                        self.last_spectral_result.maximum_residual_nodal_force_n
                    ),
                    "response_spectral_raw_residual_force_power_w": (
                        self.last_spectral_result.residual_force_power_w
                    ),
                    "response_spectral_applied_residual_force_power_w": (
                        self.residual_source_scale
                        * self.last_spectral_result.residual_force_power_w
                    ),
                    "response_spectral_source_energy_sum": float(
                        np.sum(partition.total_source_energy_by_node)
                    ),
                    "response_spectral_residual_energy_sum": float(
                        np.sum(partition.total_residual_energy_by_node)
                    ),
                    "response_spectral_captured_energy_sum": float(
                        np.sum(partition.total_captured_energy_by_node)
                    ),
                    "response_spectral_maximum_energy_closure_error": (
                        partition.maximum_energy_closure_error
                    ),
                }
            )
        return fields

    def diagnostics_summary(self) -> dict[str, object]:
        result: dict[str, object] = {
            "model_type": self.model_type,
            "config_fingerprint": self.config_fingerprint,
            "calibration_fingerprint": self.calibration.fingerprint,
            "static_spectrum_source_sha256": self.static_spectrum_source_sha256,
            "residual_source_scale": self.residual_source_scale,
            "samples": int(self.last_power_result is not None),
        }
        if self.tip_trigger is not None:
            result["published_tip_trigger"] = {
                key: value
                for key, value in self.tip_trigger.items()
                if key != "node_index"
            }
        if self.last_spectral_result is not None:
            partition = self.last_spectral_result.partition
            result["spectral"] = {
                "source_config_fingerprint": (self.spectral_source.config_fingerprint),
                "residual_source_scale": self.residual_source_scale,
                "maximum_residual_nodal_force_n": (
                    self.last_spectral_result.maximum_residual_nodal_force_n
                ),
                "residual_force_power_w": (
                    self.last_spectral_result.residual_force_power_w
                ),
                "applied_residual_force_power_w": (
                    self.residual_source_scale
                    * self.last_spectral_result.residual_force_power_w
                ),
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
        if self.last_diagnostics is not None:
            result["last"] = asdict(self.last_diagnostics)
        return result


__all__ = [
    "TRANSFER_MODEL_TYPE",
    "FU_TRAINING_REPRODUCTION_MODEL_TYPE",
    "EngineeringLockInDiagnostics",
    "QBladeSingleModeEngineeringLockInSilAdapter",
]
