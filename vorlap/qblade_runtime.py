"""Runtime helpers for QBlade external-library integration."""

from __future__ import annotations

import glob
import json
import math
import numbers
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .airfoil_io import load_airfoil_fft
from .qblade_conversion import convert_qblade_to_vorlap_inputs
from .qblade_component_selection import (
    build_node_permutations as _build_node_permutations,
    build_selected_components as _build_selected_components,
    component_span_coordinate as _component_span_coordinate,
    component_span_vectors_global as _component_span_vectors_global,
    interp_along_span as _interp_along_span,
    resample_component as _resample_component,
)
from .qblade_controller import QBladeController, SwapArrayAdapter, SwapBlockSpec
from .qblade_external_config import (
    build_qblade_external_config,
    patch_qblade_structural_definition,
    patch_qblade_turbine_definition,
    write_qblade_external_config,
)
from .qblade_external_table import (
    ExternalInputRow,
    ExternalLibraryTableSpec,
    ExternalOutputRow,
    NodeID,
    SwapLayout,
    build_external_library_table_spec,
    normalize_shape as _normalize_shape,
    validate_unique_node_ids as _validate_unique_node_ids,
)
from .qblade_files import resolve_qblade_path_text
from .qblade_geometry_audit import (
    GEOMETRY_AUDIT_COLUMNS,
    GeometryAuditRow,
    build_qblade_geometry_audit_rows,
    write_qblade_geometry_audit,
)
from .qblade_nodes import (
    infer_geometry_inclusions,
)
from .qblade_response_model import (
    AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW,
    AERODYNAMIC_SPEED_BASIS_RELATIVE_CROSSFLOW,
    QBladeResponseDiagnostics,
    QBladeResponseModel,
    QBladeResponseModelConfig,
    QBladeSeededResponseDiagnostics,
    QBladeSeededResponseDiagnosticsV2,
    QBladeStartupSeedConfig,
    RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW,
    RESPONSE_MODEL_SCHEMA_VERSION_RELATIVE_CROSSFLOW,
)
from .qblade_modal_power_model import (
    QBladeModalPowerDiagnostics,
    QBladeModalPowerModel,
    QBladeModalPowerModelConfig,
    TabulatedAerodynamicPowerCurve,
)
from .qblade_source_coverage import SOURCE_COVERAGE_COLUMNS, SourceCoverageRow
from .structs import AirfoilFFT, Component


_EPS = 1.0e-12
_MAX_QBLADE_TIME_S = 1.0e9
_MAX_QBLADE_TIMESTEP_INDEX = 1.0e9
_MAX_QBLADE_AZIMUTH_DEG = 1.0e9
_MAX_STRUCTURAL_VELOCITY_MPS = 1.0e3
ResponseModel = Union[QBladeResponseModel, QBladeModalPowerModel]
ResponseDiagnostics = Union[
    QBladeResponseDiagnostics,
    QBladeModalPowerDiagnostics,
]


def _resolve_path(base_file: str, candidate: str, extra_bases: Optional[Sequence[str]] = None) -> str:
    """Resolve a path relative to the parameter file directory or fallback bases."""
    return resolve_qblade_path_text(base_file, candidate, extra_bases=extra_bases)


def _resolve_source_parameter_dir(param_path: str, source_parameter_dir: Optional[str]) -> Optional[str]:
    """Resolve an optional source parameter directory from a config file."""
    if source_parameter_dir in (None, ""):
        return None
    return resolve_qblade_path_text(param_path, str(source_parameter_dir))


def load_airfoil_fft_directory(airfoil_dir: str) -> Dict[str, AirfoilFFT]:
    """Load all HDF5 airfoil FFT files in a directory."""
    if not os.path.isdir(airfoil_dir):
        raise FileNotFoundError(f"Airfoil directory does not exist: {airfoil_dir}")
    affts: Dict[str, AirfoilFFT] = {}
    for file in sorted(glob.glob(os.path.join(airfoil_dir, "*.h5"))):
        afft = load_airfoil_fft(file)
        affts[afft.name] = afft
    if not affts:
        raise FileNotFoundError(f"No airfoil FFT .h5 files were found in: {airfoil_dir}")
    if "default" not in affts:
        affts["default"] = next(iter(affts.values()))
    return affts


def _coerce_layout(layout: Mapping[str, Union[SwapBlockSpec, Sequence[int], Tuple[int, Sequence[int]], int]]) -> Dict[str, SwapBlockSpec]:
    coerced: Dict[str, SwapBlockSpec] = {}
    for name, spec in layout.items():
        if isinstance(spec, SwapBlockSpec):
            coerced[str(name)] = spec
        elif isinstance(spec, int):
            coerced[str(name)] = SwapBlockSpec(offset=spec)
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)) and len(spec) == 2:
            offset = int(spec[0])
            shape = _normalize_shape(spec[1])
            coerced[str(name)] = SwapBlockSpec(offset=offset, shape=shape)
        else:
            raise TypeError(f"Unsupported swap layout entry for '{name}'.")
    return coerced


def _load_config_json(param_file: str) -> Dict[str, Any]:
    with open(param_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _config_float_override(cfg: Mapping[str, Any], key: str) -> Optional[float]:
    if key not in cfg or cfg[key] in (None, ""):
        return None
    value = float(cfg[key])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be positive and finite when provided.")
    return value


def _config_vector_override(cfg: Mapping[str, Any], key: str, *, require_nonzero: bool) -> Optional[np.ndarray]:
    if key not in cfg or cfg[key] in (None, ""):
        return None
    raw = cfg[key]
    if isinstance(raw, str):
        parts = [part for part in raw.replace(",", " ").split() if part]
    else:
        parts = list(raw)
    if len(parts) != 3:
        raise ValueError(f"{key} must contain exactly 3 numeric components.")
    vector = np.asarray([float(component) for component in parts], dtype=float)
    if np.any(~np.isfinite(vector)):
        raise ValueError(f"{key} components must be finite.")
    if require_nonzero and float(np.linalg.norm(vector)) <= _EPS:
        raise ValueError(f"{key} must have non-zero magnitude when provided.")
    return vector


def _apply_viv_param_overrides(viv_params: Any, cfg: Mapping[str, Any]) -> None:
    fluid_density = _config_float_override(cfg, "fluid_density")
    if fluid_density is not None:
        viv_params.fluid_density = fluid_density
    fluid_dynamicviscosity = _config_float_override(cfg, "fluid_dynamicviscosity")
    if fluid_dynamicviscosity is not None:
        viv_params.fluid_dynamicviscosity = fluid_dynamicviscosity
    rotation_axis = _config_vector_override(cfg, "rotation_axis", require_nonzero=True)
    if rotation_axis is not None:
        viv_params.rotation_axis = rotation_axis
    rotation_axis_offset = _config_vector_override(cfg, "rotation_axis_offset", require_nonzero=False)
    if rotation_axis_offset is not None:
        viv_params.rotation_axis_offset = rotation_axis_offset
    inflow_vec = _config_vector_override(cfg, "inflow_vec", require_nonzero=True)
    if inflow_vec is not None:
        viv_params.inflow_vec = inflow_vec


def _response_model_from_config(
    raw_config: Any,
    controller_node_ids: Sequence[str],
) -> Optional[ResponseModel]:
    """Build and semantically validate the optional portable response model."""
    if raw_config is None:
        return None
    if not isinstance(raw_config, Mapping):
        raise TypeError("response_model must be a mapping when provided.")
    model_type = str(raw_config.get("type", ""))
    if model_type not in {
        "modal_negative_damping",
        "modal_aerodynamic_power",
    }:
        raise ValueError(
            "response_model.type must be 'modal_negative_damping' or "
            "'modal_aerodynamic_power'."
        )
    schema_version_raw = raw_config.get("schema_version")
    if model_type == "modal_negative_damping":
        # An omitted schema is the legacy response closure.  Schema v2 is
        # deliberately explicit and fail-closed so a typo cannot silently
        # reintroduce motion-relative aerodynamic feedback.
        if "schema_version" not in raw_config:
            schema_version = RESPONSE_MODEL_SCHEMA_VERSION_RELATIVE_CROSSFLOW
        elif (
            not isinstance(schema_version_raw, bool)
            and isinstance(schema_version_raw, numbers.Integral)
            and int(schema_version_raw)
            == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        ):
            schema_version = RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        else:
            try:
                legacy_schema = (
                    not isinstance(schema_version_raw, bool)
                    and int(schema_version_raw)
                    == RESPONSE_MODEL_SCHEMA_VERSION_RELATIVE_CROSSFLOW
                )
            except (TypeError, ValueError):
                legacy_schema = False
            if not legacy_schema:
                raise ValueError(
                    "response_model.schema_version must be 1 or explicit integer 2."
                )
            schema_version = RESPONSE_MODEL_SCHEMA_VERSION_RELATIVE_CROSSFLOW

        speed_basis_raw = raw_config.get("aerodynamic_speed_basis")
        if (
            schema_version
            == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
        ):
            if "aerodynamic_speed_basis" not in raw_config:
                raise ValueError(
                    "response_model schema_version 2 requires "
                    "aerodynamic_speed_basis='declared_ambient_crossflow'."
                )
            aerodynamic_speed_basis = speed_basis_raw
            if (
                not isinstance(aerodynamic_speed_basis, str)
                or aerodynamic_speed_basis
                != AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW
            ):
                raise ValueError(
                    "response_model.aerodynamic_speed_basis must be "
                    "'declared_ambient_crossflow' for schema_version 2."
                )
        else:
            aerodynamic_speed_basis = (
                AERODYNAMIC_SPEED_BASIS_RELATIVE_CROSSFLOW
                if speed_basis_raw is None
                else speed_basis_raw
            )
            if (
                not isinstance(aerodynamic_speed_basis, str)
                or aerodynamic_speed_basis
                != AERODYNAMIC_SPEED_BASIS_RELATIVE_CROSSFLOW
            ):
                raise ValueError(
                    "Legacy response_model schema_version 1 uses only "
                    "aerodynamic_speed_basis='relative_crossflow'."
                )
    else:
        try:
            valid_schema = (
                not isinstance(schema_version_raw, bool)
                and int(schema_version_raw) == 1
            )
        except (TypeError, ValueError):
            valid_schema = False
        if not valid_schema:
            raise ValueError("response_model.schema_version must be 1.")
        schema_version = 1
        if "aerodynamic_speed_basis" in raw_config:
            raise ValueError(
                "response_model.aerodynamic_speed_basis is unsupported for "
                "modal_aerodynamic_power."
            )
    enabled_raw = raw_config.get("enabled", False)
    if not isinstance(enabled_raw, bool):
        raise TypeError("response_model.enabled must be a boolean.")

    if "node_ids" not in raw_config:
        raise ValueError("response_model.node_ids is required.")
    if isinstance(raw_config["node_ids"], (str, bytes)):
        raise TypeError("response_model.node_ids must be a sequence of node IDs.")
    model_node_ids = tuple(str(node_id) for node_id in raw_config["node_ids"])
    expected_node_ids = tuple(str(node_id) for node_id in controller_node_ids)
    if model_node_ids != expected_node_ids:
        raise ValueError(
            "response_model.node_ids must exactly match controller node_ids and order."
        )

    if model_type == "modal_aerodynamic_power":
        required = (
            "mode_shape",
            "modal_mass_weights_kg",
            "aerodynamic_tributary_weights",
            "natural_frequency_hz",
            "power_curve",
            "rms_time_constant_s",
            "max_abs_target_power_w",
            "max_abs_generalized_force_n",
            "max_abs_nodal_force_n",
        )
        missing = [name for name in required if name not in raw_config]
        if missing:
            raise ValueError(
                "response_model is missing required field(s): "
                + ", ".join(missing)
            )
        for sequence_name in (
            "mode_shape",
            "modal_mass_weights_kg",
            "aerodynamic_tributary_weights",
        ):
            if isinstance(raw_config[sequence_name], (str, bytes)):
                raise TypeError(
                    f"response_model.{sequence_name} must be a numeric sequence."
                )
        power_curve_raw = raw_config["power_curve"]
        if not isinstance(power_curve_raw, Mapping):
            raise TypeError("response_model.power_curve must be a mapping.")
        curve_missing = [
            name
            for name in ("amplitude_m", "power_w")
            if name not in power_curve_raw
        ]
        if curve_missing:
            raise ValueError(
                "response_model.power_curve is missing required field(s): "
                + ", ".join(curve_missing)
            )
        if isinstance(power_curve_raw["amplitude_m"], (str, bytes)):
            raise TypeError(
                "response_model.power_curve.amplitude_m must be a numeric sequence."
            )
        if isinstance(power_curve_raw["power_w"], (str, bytes)):
            raise TypeError(
                "response_model.power_curve.power_w must be a numeric sequence."
            )
        power_curve = TabulatedAerodynamicPowerCurve(
            amplitude_m=tuple(
                float(value) for value in power_curve_raw["amplitude_m"]
            ),
            power_w=tuple(float(value) for value in power_curve_raw["power_w"]),
            source_label=str(
                power_curve_raw.get(
                    "source_label",
                    "portable inline aerodynamic-power curve",
                )
            ),
            source_sha256=str(power_curve_raw.get("source_sha256", "")),
        )
        config = QBladeModalPowerModelConfig(
            node_ids=model_node_ids,
            mode_shape=tuple(float(value) for value in raw_config["mode_shape"]),
            modal_mass_weights_kg=tuple(
                float(value) for value in raw_config["modal_mass_weights_kg"]
            ),
            aerodynamic_tributary_weights=tuple(
                float(value)
                for value in raw_config["aerodynamic_tributary_weights"]
            ),
            natural_frequency_hz=float(raw_config["natural_frequency_hz"]),
            power_curve=power_curve,
            rms_time_constant_s=float(raw_config["rms_time_constant_s"]),
            max_abs_target_power_w=float(raw_config["max_abs_target_power_w"]),
            max_abs_generalized_force_n=float(
                raw_config["max_abs_generalized_force_n"]
            ),
            max_abs_nodal_force_n=float(raw_config["max_abs_nodal_force_n"]),
            bandpass_damping_ratio=float(
                raw_config.get("bandpass_damping_ratio", 0.1)
            ),
            force_direction=str(raw_config.get("force_direction", "chord")),
            limit_action=str(raw_config.get("limit_action", "error")),
            enabled=enabled_raw,
        )
        return QBladeModalPowerModel(config)

    if (
        schema_version
        == RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
    ):
        allowed_fields = {
            "type",
            "schema_version",
            "aerodynamic_speed_basis",
            "node_ids",
            "mode_shape",
            "natural_frequency_hz",
            "reference_length_m",
            "reduced_velocity_activation",
            "reduced_velocity_low",
            "reduced_velocity_peak",
            "reduced_velocity_high",
            "aerodynamic_damping_gain_h1",
            "saturation_amplitude_ratio",
            "rms_time_constant_s",
            "max_abs_gain",
            "max_abs_nodal_force_n",
            "force_direction",
            "projection_weights",
            "bandpass_damping_ratio",
            "enabled",
            "startup_seed",
            "response_section_width_m",
        }
        unsupported = sorted(set(raw_config) - allowed_fields)
        if unsupported:
            raise ValueError(
                "response_model schema_version 2 contains unsupported field(s): "
                + ", ".join(str(name) for name in unsupported)
            )
        if "reduced_velocity_activation" in raw_config and any(
            name in raw_config
            for name in (
                "reduced_velocity_low",
                "reduced_velocity_peak",
                "reduced_velocity_high",
            )
        ):
            raise ValueError(
                "response_model schema_version 2 must use either "
                "reduced_velocity_activation or the three explicit bounds, not both."
            )

    activation = raw_config.get("reduced_velocity_activation")
    if activation is not None:
        try:
            valid_activation = (
                not isinstance(activation, (str, bytes)) and len(activation) == 3
            )
        except TypeError:
            valid_activation = False
        if not valid_activation:
            raise ValueError(
                "response_model.reduced_velocity_activation must contain [low, peak, high]."
            )
        reduced_velocity_low, reduced_velocity_peak, reduced_velocity_high = activation
    else:
        required_activation = (
            "reduced_velocity_low",
            "reduced_velocity_peak",
            "reduced_velocity_high",
        )
        missing_activation = [
            name for name in required_activation if name not in raw_config
        ]
        if missing_activation:
            raise ValueError(
                "response_model is missing activation field(s): "
                + ", ".join(missing_activation)
            )
        reduced_velocity_low = raw_config["reduced_velocity_low"]
        reduced_velocity_peak = raw_config["reduced_velocity_peak"]
        reduced_velocity_high = raw_config["reduced_velocity_high"]

    required = (
        "mode_shape",
        "natural_frequency_hz",
        "reference_length_m",
        "aerodynamic_damping_gain_h1",
        "saturation_amplitude_ratio",
        "rms_time_constant_s",
        "max_abs_gain",
        "max_abs_nodal_force_n",
    )
    missing = [name for name in required if name not in raw_config]
    if missing:
        raise ValueError(
            "response_model is missing required field(s): " + ", ".join(missing)
        )
    if isinstance(raw_config["mode_shape"], (str, bytes)):
        raise TypeError("response_model.mode_shape must be a numeric sequence.")
    projection_weights_raw = raw_config.get("projection_weights")
    if isinstance(projection_weights_raw, (str, bytes)):
        raise TypeError(
            "response_model.projection_weights must be a numeric sequence."
        )
    projection_weights = (
        None
        if projection_weights_raw is None
        else tuple(float(value) for value in projection_weights_raw)
    )
    response_section_width_raw = raw_config.get("response_section_width_m")
    if isinstance(response_section_width_raw, bool) or isinstance(
        response_section_width_raw,
        (str, bytes),
    ):
        raise TypeError(
            "response_model.response_section_width_m must be a real scalar or numeric sequence."
        )
    if response_section_width_raw is None:
        response_section_width_m = None
    elif isinstance(response_section_width_raw, numbers.Real):
        response_section_width_m = float(response_section_width_raw)
    else:
        try:
            response_section_width_values = tuple(response_section_width_raw)
        except TypeError as exc:
            raise TypeError(
                "response_model.response_section_width_m must be a real scalar or numeric sequence."
            ) from exc
        if any(isinstance(value, (bool, np.bool_)) for value in response_section_width_values):
            raise TypeError(
                "response_model.response_section_width_m values must be real numeric values, not boolean."
            )
        response_section_width_m = tuple(
            float(value) for value in response_section_width_values
        )
    startup_seed_raw = raw_config.get("startup_seed")
    startup_seed = None
    if startup_seed_raw is not None:
        if not isinstance(startup_seed_raw, Mapping):
            raise TypeError("response_model.startup_seed must be a mapping.")
        seed_required = (
            "schema_version",
            "enabled",
            "duration_cycles",
            "target_amplitude_ratio",
            "structural_damping_ratio",
            "taper",
            "carrier",
            "nodal_weighting",
            "nodal_force_limit_source",
        )
        seed_missing = [
            name for name in seed_required if name not in startup_seed_raw
        ]
        if seed_missing:
            raise ValueError(
                "response_model.startup_seed is missing required field(s): "
                + ", ".join(seed_missing)
            )
        seed_unknown = sorted(set(startup_seed_raw) - set(seed_required))
        if seed_unknown:
            raise ValueError(
                "response_model.startup_seed contains unsupported field(s): "
                + ", ".join(str(name) for name in seed_unknown)
            )
        seed_schema = startup_seed_raw["schema_version"]
        if (
            isinstance(seed_schema, bool)
            or not isinstance(seed_schema, numbers.Integral)
            or seed_schema != 1
        ):
            raise ValueError(
                "response_model.startup_seed.schema_version must be 1."
            )
        seed_enabled = startup_seed_raw["enabled"]
        if not isinstance(seed_enabled, bool):
            raise TypeError(
                "response_model.startup_seed.enabled must be a boolean."
            )
        for numeric_name in (
            "duration_cycles",
            "target_amplitude_ratio",
            "structural_damping_ratio",
        ):
            if isinstance(startup_seed_raw[numeric_name], bool) or not isinstance(
                startup_seed_raw[numeric_name],
                numbers.Real,
            ):
                raise TypeError(
                    f"response_model.startup_seed.{numeric_name} must be a real numeric value, not boolean."
                )
        startup_seed = QBladeStartupSeedConfig(
            enabled=seed_enabled,
            duration_cycles=float(startup_seed_raw["duration_cycles"]),
            target_amplitude_ratio=float(
                startup_seed_raw["target_amplitude_ratio"]
            ),
            structural_damping_ratio=float(
                startup_seed_raw["structural_damping_ratio"]
            ),
            schema_version=seed_schema,
            taper=startup_seed_raw["taper"],
            carrier=startup_seed_raw["carrier"],
            nodal_weighting=startup_seed_raw["nodal_weighting"],
            nodal_force_limit_source=startup_seed_raw[
                "nodal_force_limit_source"
            ],
        )

    config = QBladeResponseModelConfig(
        node_ids=model_node_ids,
        mode_shape=tuple(float(value) for value in raw_config["mode_shape"]),
        natural_frequency_hz=float(raw_config["natural_frequency_hz"]),
        reference_length_m=float(raw_config["reference_length_m"]),
        reduced_velocity_low=float(reduced_velocity_low),
        reduced_velocity_peak=float(reduced_velocity_peak),
        reduced_velocity_high=float(reduced_velocity_high),
        aerodynamic_damping_gain_h1=float(
            raw_config["aerodynamic_damping_gain_h1"]
        ),
        saturation_amplitude_ratio=float(
            raw_config["saturation_amplitude_ratio"]
        ),
        rms_time_constant_s=float(raw_config["rms_time_constant_s"]),
        max_abs_gain=float(raw_config["max_abs_gain"]),
        max_abs_nodal_force_n=float(raw_config["max_abs_nodal_force_n"]),
        force_direction=str(raw_config.get("force_direction", "normal")),
        projection_weights=projection_weights,
        bandpass_damping_ratio=float(
            raw_config.get("bandpass_damping_ratio", 0.1)
        ),
        enabled=enabled_raw,
        startup_seed=startup_seed,
        schema_version=schema_version,
        aerodynamic_speed_basis=aerodynamic_speed_basis,
        response_section_width_m=response_section_width_m,
    )
    return QBladeResponseModel(config)


def _layout_extent(layout: Mapping[str, SwapBlockSpec]) -> int:
    return max(spec.offset + spec.size for spec in layout.values()) if layout else 0


def _validate_config_swap_layout(
    cfg: Mapping[str, Any],
    swap_layout: Mapping[str, SwapBlockSpec],
    requested_node_ids: Sequence[str],
) -> None:
    """Fail fast when a JSON config layout cannot match the requested nodes."""
    required_blocks = {
        "time": str(cfg.get("time_block", "time")),
        "timestep": str(cfg.get("timestep_block", "timestep")),
        "azimuth": str(cfg.get("azimuth_block", "azimuth_deg")),
        "velocity": str(cfg.get("velocity_block", "velocity")),
        "force": str(cfg.get("force_block", "force")),
    }
    missing = [block for block in required_blocks.values() if block not in swap_layout]
    if missing:
        raise ValueError(f"Config swap_layout is missing required block(s): {', '.join(missing)}")

    for label in ("time", "timestep", "azimuth"):
        spec = swap_layout[required_blocks[label]]
        if spec.shape != ():
            raise ValueError(f"Config swap block '{required_blocks[label]}' must be scalar.")

    node_count = len(tuple(requested_node_ids))
    expected_shape = (node_count, 3)
    velocity_shape = swap_layout[required_blocks["velocity"]].shape
    force_shape = swap_layout[required_blocks["force"]].shape
    if velocity_shape != expected_shape:
        raise ValueError(
            f"Config velocity block shape must be {expected_shape} for {node_count} node(s); got {velocity_shape}."
        )
    if force_shape != expected_shape:
        raise ValueError(
            f"Config force block shape must be {expected_shape} for {node_count} node(s); got {force_shape}."
        )

    if "swap_size" in cfg:
        declared = int(cfg["swap_size"])
        actual = _layout_extent(swap_layout)
        if declared != actual:
            raise ValueError(f"Config swap_size {declared} does not match swap_layout extent {actual}.")


def _runtime_input_sanity_message(
    *,
    time_s: float,
    timestep_index: float,
    azimuth_deg: float,
    velocity_swap: Optional[np.ndarray] = None,
) -> Optional[str]:
    """Return a failure message when QBlade swap inputs are finite but implausible."""
    scalar_values = {
        "time": float(time_s),
        "timestep": float(timestep_index),
        "azimuth": float(azimuth_deg),
    }
    if not np.isfinite(list(scalar_values.values())).all():
        return (
            "VorLap skipped update: non-finite time/timestep/azimuth from QBlade relaxation "
            f"(t={time_s}, step={timestep_index}, az={azimuth_deg})."
        )
    if time_s < -1.0e-9 or abs(time_s) > _MAX_QBLADE_TIME_S:
        return f"VorLap skipped update: implausible QBlade swap time value t={time_s:.6g}s."
    if timestep_index < -1.0e-9 or abs(timestep_index) > _MAX_QBLADE_TIMESTEP_INDEX:
        return f"VorLap skipped update: implausible QBlade swap timestep value step={timestep_index:.6g}."
    if abs(azimuth_deg) > _MAX_QBLADE_AZIMUTH_DEG:
        return f"VorLap skipped update: implausible QBlade swap azimuth value az={azimuth_deg:.6g}deg."
    if velocity_swap is not None:
        velocity = np.asarray(velocity_swap, dtype=float)
        if np.any(~np.isfinite(velocity)):
            return (
                "VorLap skipped update: non-finite structural velocity from QBlade relaxation "
                f"(t={time_s:.6g}s, step={timestep_index:.6g}, az={azimuth_deg:.6g}deg)."
            )
        max_abs_velocity = float(np.max(np.abs(velocity))) if velocity.size else 0.0
        if max_abs_velocity > _MAX_STRUCTURAL_VELOCITY_MPS:
            return (
                "VorLap skipped update: implausible structural velocity from QBlade swap "
                f"(max|Vstruct|={max_abs_velocity:.6g}m/s, "
                f"limit={_MAX_STRUCTURAL_VELOCITY_MPS:.6g}m/s)."
            )
    return None


class VorLapQBladeRuntime:
    """
    Stateful runtime object used by the embedded QBlade bridge.

    The runtime keeps the VorLap controller alive across timesteps, caches the
    swap-array layout, and exposes a small diagnostic message buffer for the
    optional QBlade `update_message` callback.
    """

    def __init__(
        self,
        controller: QBladeController,
        swap_layout: Mapping[str, Union[SwapBlockSpec, Sequence[int], Tuple[int, Sequence[int]], int]],
        *,
        time_block: str = "time",
        timestep_block: str = "timestep",
        azimuth_block: str = "azimuth_deg",
        velocity_block: str = "velocity",
        force_block: str = "force",
        sample_step: Optional[float] = None,
        parameter_file: Optional[str] = None,
        swap_to_controller_idx: Optional[Sequence[int]] = None,
        controller_to_swap_idx: Optional[Sequence[int]] = None,
        debug: bool = False,
        debug_failure_inject_update_exception: bool = False,
        source_bounds_policy: str = "warn",
        resolved_sim_path: Optional[str] = None,
        resolved_airfoil_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        use_qblade_local_inflow: bool = False,
    ) -> None:
        if controller is None:
            raise ValueError("controller must not be None.")
        if not isinstance(use_qblade_local_inflow, bool):
            raise ValueError("use_qblade_local_inflow must be a JSON boolean.")
        self.use_qblade_local_inflow = use_qblade_local_inflow
        self.controller = controller
        self.time_block = str(time_block)
        self.timestep_block = str(timestep_block)
        self.azimuth_block = str(azimuth_block)
        self.velocity_block = str(velocity_block)
        self.force_block = str(force_block)
        self.sample_step = None if sample_step is None else float(sample_step)
        self.parameter_file = parameter_file
        self.debug = bool(debug)
        self.source_bounds_policy = str(source_bounds_policy)
        self.source_frequency_cap_hz = getattr(controller, "source_frequency_cap_hz", None)
        self._debug_failure_inject_update_exception = bool(debug_failure_inject_update_exception)
        if self._debug_failure_inject_update_exception and not self.debug:
            raise ValueError("debug_failure_inject_update_exception requires debug=True.")
        self.resolved_sim_path = None if resolved_sim_path is None else str(resolved_sim_path)
        self.resolved_airfoil_dir = None if resolved_airfoil_dir is None else str(resolved_airfoil_dir)
        self.log_file = None if log_file is None else str(log_file)
        self._log_write_failed = False
        self._log_write_error: Optional[str] = None
        self._layout = _coerce_layout(swap_layout)
        self._adapter: Optional[SwapArrayAdapter] = None
        if (swap_to_controller_idx is None) != (controller_to_swap_idx is None):
            raise ValueError("swap_to_controller_idx and controller_to_swap_idx must be provided together.")
        self._swap_to_controller_idx = (
            None if swap_to_controller_idx is None else np.asarray(swap_to_controller_idx, dtype=int).reshape(-1)
        )
        self._controller_to_swap_idx = (
            None if controller_to_swap_idx is None else np.asarray(controller_to_swap_idx, dtype=int).reshape(-1)
        )
        self._controller_velocity_buffer: Optional[np.ndarray] = None
        self._controller_structural_velocity_buffer: Optional[np.ndarray] = None
        self._swap_force_buffer: Optional[np.ndarray] = None
        self._last_controller_structural_velocity: Optional[np.ndarray] = None
        self._last_controller_forces: Optional[np.ndarray] = None
        self._source_coverage: Dict[str, Dict[str, Any]] = {}
        inflow_dir = np.asarray(self.controller.viv_params.inflow_vec, dtype=float).reshape(3)
        inflow_norm = float(np.linalg.norm(inflow_dir))
        if not np.isfinite(inflow_norm) or inflow_norm <= 1.0e-12:
            raise ValueError("inflow_vec must have non-zero finite magnitude.")
        inflow_dir = inflow_dir / inflow_norm
        inflow_speed = float(np.asarray(self.controller.viv_params.inflow_speeds, dtype=float).reshape(-1)[0])
        self._ambient_velocity_global = inflow_dir * inflow_speed
        if self.debug:
            sim_name = os.path.basename(self.resolved_sim_path) if self.resolved_sim_path else "?"
            airfoil_name = self.resolved_airfoil_dir if self.resolved_airfoil_dir else "?"
            self._last_message = (
                f"VorLap init ok; sim={sim_name}; airfoils={airfoil_name}; "
                f"nodes={self.controller.node_ids[0] if self.controller.node_ids else '?'}.. "
                f"count={len(self.controller.node_ids) if self.controller.node_ids else 0}"
            )
        else:
            self._last_message = "VorLap QBlade runtime initialized."
        self._append_log_line(self._last_message)

    def _append_log_line(self, message: str) -> None:
        if not self.log_file or self._log_write_failed:
            return
        log_path = os.path.abspath(self.log_file)
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{message}\n")
        except OSError as exc:
            self._log_write_failed = True
            self._log_write_error = f"{type(exc).__name__}: {exc}"

    @classmethod
    def from_qblade_config(cls, param_file: str) -> "VorLapQBladeRuntime":
        """Build a runtime directly from the JSON config written by the case-prep script."""
        param_path = os.path.abspath(param_file)
        cfg = _load_config_json(param_path)

        # Resolve model-relative paths from the generated control file; copied
        # working cases can add source_parameter_dir as a fallback search root.
        source_parameter_dir = _resolve_source_parameter_dir(param_path, cfg.get("source_parameter_dir"))
        extra_bases = [] if source_parameter_dir in (None, "") else [str(source_parameter_dir)]

        sim_path = _resolve_path(param_path, cfg["sim_path"], extra_bases=extra_bases)
        airfoil_dir = _resolve_path(param_path, cfg["airfoil_dir"], extra_bases=extra_bases)
        log_file = cfg.get("log_file")
        log_file = None if log_file in (None, "") else _resolve_path(param_path, str(log_file), extra_bases=extra_bases)

        default_airfoil_id = str(cfg.get("default_airfoil_id", "default"))
        tower_airfoil_id = str(cfg.get("tower_airfoil_id", "cylinder"))
        n_freq_depth = cfg.get("n_freq_depth", None)
        n_freq_depth = None if n_freq_depth is None else int(n_freq_depth)
        force_scale = float(cfg.get("force_scale", 1.0))
        include_mean_load = bool(cfg.get("include_mean_load", False))
        source_bounds_policy = str(cfg.get("source_bounds_policy", "warn"))
        source_frequency_cap_hz_raw = cfg.get("source_frequency_cap_hz")
        source_frequency_cap_hz = (
            None if source_frequency_cap_hz_raw is None else float(source_frequency_cap_hz_raw)
        )
        debug = bool(cfg.get("debug", False))
        debug_failure_inject_update_exception = bool(cfg.get("debug_failure_inject_update_exception", False))

        requested_node_ids_raw = cfg.get("node_ids")
        if requested_node_ids_raw is None:
            requested_node_ids: Tuple[str, ...] = ()
        else:
            requested_node_ids = tuple(str(node_id) for node_id in requested_node_ids_raw)
        _validate_unique_node_ids(requested_node_ids, "requested config")
        include_struts, include_tower = infer_geometry_inclusions(
            requested_node_ids,
            bool(cfg.get("include_struts", True)),
            bool(cfg.get("include_tower", True)),
        )

        components, viv_params, node_ids = convert_qblade_to_vorlap_inputs(
            sim_path,
            default_airfoil_id=default_airfoil_id,
            include_struts=include_struts,
            include_tower=include_tower,
            tower_airfoil_id=tower_airfoil_id,
        )
        _apply_viv_param_overrides(viv_params, cfg)
        affts = load_airfoil_fft_directory(airfoil_dir)
        viv_params.airfoil_folder = airfoil_dir

        if not requested_node_ids:
            requested_node_ids = tuple(str(node_id) for node_id in node_ids)
        if not requested_node_ids:
            raise ValueError("Config file does not define any node_ids.")

        swap_layout_cfg = cfg.get("swap_layout")
        if not isinstance(swap_layout_cfg, list) or not swap_layout_cfg:
            raise ValueError("Config file does not define a valid swap_layout.")
        swap_layout: Dict[str, Union[SwapBlockSpec, Sequence[int], Tuple[int, Sequence[int]], int]] = {}
        for entry in swap_layout_cfg:
            if not isinstance(entry, Mapping):
                raise TypeError("swap_layout entries must be mapping objects.")
            name = str(entry["name"])
            offset = int(entry["offset"])
            shape = _normalize_shape(entry.get("shape", ()))
            swap_layout[name] = SwapBlockSpec(offset=offset, shape=shape)
        _validate_config_swap_layout(cfg, swap_layout, requested_node_ids)

        selected_components, controller_node_ids, swap_node_ids = _build_selected_components(
            components, requested_node_ids
        )
        response_model = _response_model_from_config(
            cfg.get("response_model"),
            controller_node_ids,
        )
        controller = QBladeController.from_components(
            selected_components,
            affts,
            viv_params=viv_params,
            n_freq_depth=n_freq_depth,
            node_ids=controller_node_ids,
            force_scale=force_scale,
            include_mean_load=include_mean_load,
            source_bounds_policy=source_bounds_policy,
            source_frequency_cap_hz=source_frequency_cap_hz,
            response_model=response_model,
        )
        swap_to_controller_idx, controller_to_swap_idx = _build_node_permutations(
            swap_node_ids, controller.node_ids
        )

        return cls(
            controller=controller,
            swap_layout=swap_layout,
            time_block=str(cfg.get("time_block", "time")),
            timestep_block=str(cfg.get("timestep_block", "timestep")),
            azimuth_block=str(cfg.get("azimuth_block", "azimuth_deg")),
            velocity_block=str(cfg.get("velocity_block", "velocity")),
            force_block=str(cfg.get("force_block", "force")),
            sample_step=cfg.get("sample_step"),
            parameter_file=param_path,
            swap_to_controller_idx=swap_to_controller_idx,
            controller_to_swap_idx=controller_to_swap_idx,
            debug=debug,
            debug_failure_inject_update_exception=debug_failure_inject_update_exception,
            source_bounds_policy=source_bounds_policy,
            resolved_sim_path=sim_path,
            resolved_airfoil_dir=airfoil_dir,
            log_file=log_file,
            use_qblade_local_inflow=cfg.get("use_qblade_local_inflow", False),
        )

    @classmethod
    def from_qblade_sim(
        cls,
        sim_path: str,
        airfoil_dir: str,
        *,
        default_airfoil_id: str = "default",
        include_struts: bool = True,
        include_tower: bool = True,
        tower_airfoil_id: str = "cylinder",
        n_freq_depth: Optional[int] = None,
        force_scale: float = 1.0,
        include_mean_load: bool = False,
        source_bounds_policy: str = "warn",
        source_frequency_cap_hz: Optional[float] = None,
        response_model: Optional[ResponseModel] = None,
    ) -> "VorLapQBladeRuntime":
        """Construct a runtime from a QBlade simulation definition and an airfoil FFT directory."""
        components, viv_params, node_ids = convert_qblade_to_vorlap_inputs(
            sim_path,
            default_airfoil_id=default_airfoil_id,
            include_struts=include_struts,
            include_tower=include_tower,
            tower_airfoil_id=tower_airfoil_id,
        )
        affts = load_airfoil_fft_directory(airfoil_dir)
        viv_params.airfoil_folder = airfoil_dir
        controller = QBladeController.from_components(
            components,
            affts,
            viv_params=viv_params,
            n_freq_depth=n_freq_depth,
            node_ids=node_ids,
            force_scale=force_scale,
            include_mean_load=include_mean_load,
            source_bounds_policy=source_bounds_policy,
            source_frequency_cap_hz=source_frequency_cap_hz,
            response_model=response_model,
        )
        table_spec = build_external_library_table_spec(node_ids)
        return cls(
            controller=controller,
            swap_layout=table_spec.swap_layout,
            sample_step=None,
            debug=False,
            source_bounds_policy=source_bounds_policy,
            resolved_sim_path=sim_path,
            resolved_airfoil_dir=airfoil_dir,
            log_file=None,
        )

    @property
    def message(self) -> str:
        return self._last_message

    @property
    def response_model_type(self) -> str:
        """Stable portable type name for the configured response closure."""
        return str(getattr(self.controller, "response_model_type", "none"))

    @property
    def last_response_diagnostics(self) -> Optional[ResponseDiagnostics]:
        """Return the latest typed response-model diagnostics, when enabled."""
        return getattr(self.controller, "last_response_diagnostics", None)

    @property
    def response_diagnostics_summary(self) -> Dict[str, Any]:
        """Return cumulative response extrema and clipping counts."""
        return dict(getattr(self.controller, "response_diagnostics_summary", {}))

    @property
    def swap_size(self) -> int:
        return max(spec.offset + spec.size for spec in self._layout.values())

    @property
    def layout(self) -> Dict[str, SwapBlockSpec]:
        return dict(self._layout)

    def _adapter_for(self, avr_swap: np.ndarray) -> SwapArrayAdapter:
        if self._adapter is None:
            self._adapter = SwapArrayAdapter(avr_swap, self._layout)
            return self._adapter
        self._adapter.rebind(avr_swap)
        return self._adapter

    def _zero_force_output(self, adapter: SwapArrayAdapter, message: str) -> np.ndarray:
        force_view = adapter.block(self.force_block)
        force_view[...] = 0.0
        self._last_controller_structural_velocity = None
        self._last_controller_forces = None
        self._last_message = message
        self._append_log_line(self._last_message)
        return np.asarray(force_view, dtype=float)

    def clear_force_output(self, avr_swap: np.ndarray, message: str = "VorLap force output cleared.") -> np.ndarray:
        """Clear the configured force block after a bridge-side failure."""
        swap = np.asarray(avr_swap)
        if swap.ndim != 1:
            raise ValueError("avr_swap must be a one-dimensional buffer.")
        if swap.size < self.swap_size:
            raise ValueError(f"avr_swap is too small: expected at least {self.swap_size}, got {swap.size}.")
        adapter = self._adapter_for(swap)
        return self._zero_force_output(adapter, str(message))

    @staticmethod
    def _coverage_entry(component: str) -> Dict[str, Any]:
        return {
            "component": str(component),
            "airfoil_id": "",
            "reynolds_model": "",
            "samples": 0,
            "node_samples": 0,
            "aoa_min_deg": np.inf,
            "aoa_max_deg": -np.inf,
            "source_aoa_min_deg": np.inf,
            "source_aoa_max_deg": -np.inf,
            "re_min": np.inf,
            "re_max": -np.inf,
            "source_re_min": np.inf,
            "source_re_max": -np.inf,
            "freq_min_hz": np.inf,
            "freq_max_hz": 0.0,
            "unfiltered_freq_max_hz": 0.0,
            "frequency_truncated_count": 0,
            "nyquist_hz": np.inf,
            "max_nodal_force_n": 0.0,
            "force_coefficient_max_abs": 0.0,
            "max_response_nodal_force_n": 0.0,
            "response_force_coefficient_max_abs": 0.0,
            "max_total_nodal_force_n": 0.0,
            "total_force_coefficient_max_abs": 0.0,
            "source_force_coefficient_max_abs": 0.0,
            "max_structural_velocity_mps": 0.0,
            "force_power_sum_w": 0.0,
            "force_power_count": 0,
            "force_power_max_abs_w": 0.0,
            "re_lower_bound_warning_count": 0,
            "re_out_of_range_count": 0,
            "aoa_out_of_range_count": 0,
            "aliased_frequency_count": 0,
            "source_clip_count": 0,
            "force_coefficient_out_of_range_count": 0,
            "nonfinite_count": 0,
        }

    @staticmethod
    def _finite_values(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]

    def record_source_coverage_sample(self, sample_period_s: Optional[float] = None) -> None:
        """Accumulate per-component aerodynamic/source coverage for the last update."""
        state = self.controller.last_source_diagnostics
        structural_velocity = self._last_controller_structural_velocity
        forces = self._last_controller_forces
        if state is None or structural_velocity is None or forces is None:
            return

        component_ids = np.asarray(state.component_ids, dtype=object)
        airfoil_ids = np.asarray(state.source_airfoil_ids, dtype=object)
        reynolds_models = np.asarray(state.reynolds_models, dtype=object)
        unique_components = sorted({str(component) for component in component_ids})
        structural_speed = np.linalg.norm(np.asarray(structural_velocity, dtype=float), axis=1)
        force_power = np.einsum("ij,ij->i", np.asarray(forces, dtype=float), np.asarray(structural_velocity, dtype=float))

        nyquist_hz: Optional[float]
        if sample_period_s is None:
            nyquist_hz = None
        else:
            sample_period = float(sample_period_s)
            nyquist_hz = 0.5 / sample_period if np.isfinite(sample_period) and sample_period > 0.0 else None

        numeric_names = (
            "aoa_deg",
            "reynolds",
            "freq_min_hz",
            "freq_max_hz",
            "unfiltered_freq_max_hz",
            "force_norm_n",
            "force_coefficient_abs",
            "response_force_norm_n",
            "response_force_coefficient_abs",
            "total_force_norm_n",
            "total_force_coefficient_abs",
            "source_aoa_min_deg",
            "source_aoa_max_deg",
            "source_re_min",
            "source_re_max",
            "source_force_coefficient_max_abs",
        )
        numeric = {name: np.asarray(getattr(state, name), dtype=float) for name in numeric_names}

        for component in unique_components:
            mask = component_ids == component
            if not np.any(mask):
                continue
            entry = self._source_coverage.setdefault(component, self._coverage_entry(component))
            component_airfoils = sorted({str(value) for value in airfoil_ids[mask]})
            component_models = sorted({str(value) for value in reynolds_models[mask]})
            airfoil_label = ",".join(component_airfoils)
            model_label = ",".join(component_models)
            if entry["airfoil_id"] not in ("", airfoil_label):
                airfoil_label = ",".join(sorted(set(str(entry["airfoil_id"]).split(",")) | set(component_airfoils)))
            if entry["reynolds_model"] not in ("", model_label):
                model_label = ",".join(sorted(set(str(entry["reynolds_model"]).split(",")) | set(component_models)))
            entry["airfoil_id"] = airfoil_label
            entry["reynolds_model"] = model_label
            entry["samples"] += 1
            node_count = int(np.count_nonzero(mask))
            entry["node_samples"] += node_count

            finite_bundle = [self._finite_values(numeric[name][mask]) for name in numeric_names]
            finite_bundle.extend([self._finite_values(structural_speed[mask]), self._finite_values(force_power[mask])])
            entry["nonfinite_count"] += sum(node_count - int(values.size) for values in finite_bundle)

            aoa = numeric["aoa_deg"][mask]
            source_aoa_min = numeric["source_aoa_min_deg"][mask]
            source_aoa_max = numeric["source_aoa_max_deg"][mask]
            reynolds = numeric["reynolds"][mask]
            source_re_min = numeric["source_re_min"][mask]
            source_re_max = numeric["source_re_max"][mask]
            freq_min = numeric["freq_min_hz"][mask]
            freq_max = numeric["freq_max_hz"][mask]
            unfiltered_freq_max = numeric["unfiltered_freq_max_hz"][mask]
            force_norm = numeric["force_norm_n"][mask]
            force_coeff = numeric["force_coefficient_abs"][mask]
            response_force_norm = numeric["response_force_norm_n"][mask]
            response_force_coeff = numeric["response_force_coefficient_abs"][mask]
            total_force_norm = numeric["total_force_norm_n"][mask]
            total_force_coeff = numeric["total_force_coefficient_abs"][mask]
            source_force_coeff = numeric["source_force_coefficient_max_abs"][mask]

            entry["aoa_min_deg"] = min(float(entry["aoa_min_deg"]), float(np.nanmin(aoa)))
            entry["aoa_max_deg"] = max(float(entry["aoa_max_deg"]), float(np.nanmax(aoa)))
            entry["source_aoa_min_deg"] = min(float(entry["source_aoa_min_deg"]), float(np.nanmin(source_aoa_min)))
            entry["source_aoa_max_deg"] = max(float(entry["source_aoa_max_deg"]), float(np.nanmax(source_aoa_max)))
            entry["re_min"] = min(float(entry["re_min"]), float(np.nanmin(reynolds)))
            entry["re_max"] = max(float(entry["re_max"]), float(np.nanmax(reynolds)))
            entry["source_re_min"] = min(float(entry["source_re_min"]), float(np.nanmin(source_re_min)))
            entry["source_re_max"] = max(float(entry["source_re_max"]), float(np.nanmax(source_re_max)))
            entry["freq_min_hz"] = min(float(entry["freq_min_hz"]), float(np.nanmin(freq_min)))
            entry["freq_max_hz"] = max(float(entry["freq_max_hz"]), float(np.nanmax(freq_max)))
            entry["unfiltered_freq_max_hz"] = max(
                float(entry["unfiltered_freq_max_hz"]),
                float(np.nanmax(unfiltered_freq_max)),
            )
            entry["frequency_truncated_count"] += int(
                np.sum(np.asarray(state.frequency_truncated_count, dtype=int)[mask])
            )
            entry["max_nodal_force_n"] = max(float(entry["max_nodal_force_n"]), float(np.nanmax(force_norm)))
            entry["force_coefficient_max_abs"] = max(float(entry["force_coefficient_max_abs"]), float(np.nanmax(force_coeff)))
            entry["max_response_nodal_force_n"] = max(
                float(entry["max_response_nodal_force_n"]),
                float(np.nanmax(response_force_norm)),
            )
            entry["response_force_coefficient_max_abs"] = max(
                float(entry["response_force_coefficient_max_abs"]),
                float(np.nanmax(response_force_coeff)),
            )
            entry["max_total_nodal_force_n"] = max(
                float(entry["max_total_nodal_force_n"]),
                float(np.nanmax(total_force_norm)),
            )
            entry["total_force_coefficient_max_abs"] = max(
                float(entry["total_force_coefficient_max_abs"]),
                float(np.nanmax(total_force_coeff)),
            )
            entry["source_force_coefficient_max_abs"] = max(
                float(entry["source_force_coefficient_max_abs"]),
                float(np.nanmax(source_force_coeff)),
            )
            entry["max_structural_velocity_mps"] = max(
                float(entry["max_structural_velocity_mps"]),
                float(np.nanmax(structural_speed[mask])),
            )
            entry["force_power_sum_w"] += float(np.nansum(force_power[mask]))
            entry["force_power_count"] += node_count
            entry["force_power_max_abs_w"] = max(
                float(entry["force_power_max_abs_w"]),
                float(np.nanmax(np.abs(force_power[mask]))),
            )

            re_below = reynolds < source_re_min
            re_above = reynolds > source_re_max
            reynolds_independent = (
                reynolds_models[mask] == "replicated_reynolds_independent"
            )
            re_lower_warning = re_below & reynolds_independent
            re_out = re_above | (re_below & ~reynolds_independent)
            aoa_out = (aoa < source_aoa_min) | (aoa > source_aoa_max)
            # A reconstructed single-channel coefficient can round a few ulps
            # above its source amplitude after force normalization.  Preserve
            # the audit for material excursions without counting that floating-
            # point noise as source-range exceedance.
            coeff_tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, source_force_coeff)
            coeff_out = force_coeff > source_force_coeff + coeff_tolerance
            entry["re_lower_bound_warning_count"] += int(np.count_nonzero(re_lower_warning))
            entry["re_out_of_range_count"] += int(np.count_nonzero(re_out))
            entry["aoa_out_of_range_count"] += int(np.count_nonzero(aoa_out))
            entry["source_clip_count"] += int(np.count_nonzero(re_out | aoa_out))
            entry["force_coefficient_out_of_range_count"] += int(np.count_nonzero(coeff_out))
            if nyquist_hz is not None:
                entry["nyquist_hz"] = min(float(entry["nyquist_hz"]), nyquist_hz)
                entry["aliased_frequency_count"] += int(np.count_nonzero(freq_max > nyquist_hz))

    def source_coverage_summary(self, sample_period_s: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return accumulated source-coverage rows suitable for run summaries."""
        rows: List[Dict[str, Any]] = []
        fallback_nyquist = None
        if sample_period_s is not None:
            sample_period = float(sample_period_s)
            if np.isfinite(sample_period) and sample_period > 0.0:
                fallback_nyquist = 0.5 / sample_period
        for component, entry in sorted(self._source_coverage.items()):
            nyquist_hz = float(entry["nyquist_hz"])
            if not np.isfinite(nyquist_hz):
                nyquist_hz = float(fallback_nyquist) if fallback_nyquist is not None else 0.0
            freq_max = float(entry["freq_max_hz"])
            nyquist_ratio = nyquist_hz / freq_max if freq_max > 1.0e-12 else 1.0e12
            force_power_count = int(entry["force_power_count"])
            force_power_mean = float(entry["force_power_sum_w"]) / force_power_count if force_power_count else 0.0
            row = SourceCoverageRow(
                    component=component,
                    airfoil_id=str(entry["airfoil_id"]),
                    reynolds_model=str(entry["reynolds_model"]),
                    samples=int(entry["samples"]),
                    node_samples=int(entry["node_samples"]),
                    aoa_min_deg=float(entry["aoa_min_deg"]),
                    aoa_max_deg=float(entry["aoa_max_deg"]),
                    source_aoa_min_deg=float(entry["source_aoa_min_deg"]),
                    source_aoa_max_deg=float(entry["source_aoa_max_deg"]),
                    re_min=float(entry["re_min"]),
                    re_max=float(entry["re_max"]),
                    source_re_min=float(entry["source_re_min"]),
                    source_re_max=float(entry["source_re_max"]),
                    freq_min_hz=float(entry["freq_min_hz"]),
                    freq_max_hz=freq_max,
                    unfiltered_freq_max_hz=float(entry["unfiltered_freq_max_hz"]),
                    frequency_truncated_count=int(entry["frequency_truncated_count"]),
                    nyquist_hz=nyquist_hz,
                    nyquist_ratio=float(nyquist_ratio),
                    max_nodal_force_n=float(entry["max_nodal_force_n"]),
                    force_coefficient_max_abs=float(entry["force_coefficient_max_abs"]),
                    max_response_nodal_force_n=float(
                        entry["max_response_nodal_force_n"]
                    ),
                    response_force_coefficient_max_abs=float(
                        entry["response_force_coefficient_max_abs"]
                    ),
                    max_total_nodal_force_n=float(
                        entry["max_total_nodal_force_n"]
                    ),
                    total_force_coefficient_max_abs=float(
                        entry["total_force_coefficient_max_abs"]
                    ),
                    source_force_coefficient_max_abs=float(entry["source_force_coefficient_max_abs"]),
                    max_structural_velocity_mps=float(entry["max_structural_velocity_mps"]),
                    force_power_mean_w=force_power_mean,
                    force_power_max_abs_w=float(entry["force_power_max_abs_w"]),
                    re_lower_bound_warning_count=int(entry["re_lower_bound_warning_count"]),
                    re_out_of_range_count=int(entry["re_out_of_range_count"]),
                    aoa_out_of_range_count=int(entry["aoa_out_of_range_count"]),
                    aliased_frequency_count=int(entry["aliased_frequency_count"]),
                    source_clip_count=int(entry["source_clip_count"]),
                    force_coefficient_out_of_range_count=int(entry["force_coefficient_out_of_range_count"]),
                    nonfinite_count=int(entry["nonfinite_count"]),
                ).as_dict()
            rows.append(row)
        return rows

    def checkpoint_state(self) -> Dict[str, Any]:
        """Return restartable diagnostic state for wrapper-level checkpoints.

        VorLap force evaluation is reconstructed from the current QBlade
        structural state at each update. The accumulated source-coverage
        diagnostics are stateful, so preserve them separately from the QBlade
        project checkpoint.
        """
        checkpoint_controller = getattr(self.controller, "checkpoint_state", None)
        response_enabled = bool(
            getattr(self.controller, "response_model_enabled", False)
        )
        if callable(checkpoint_controller):
            controller_state = checkpoint_controller()
        elif response_enabled:
            raise ValueError(
                "Enabled response model requires controller checkpoint support."
            )
        else:
            controller_state = None
        return {
            "source_coverage": {
                str(component): dict(entry)
                for component, entry in sorted(self._source_coverage.items())
            },
            "message": self._last_message,
            "controller": controller_state,
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        """Restore diagnostic state written by :meth:`checkpoint_state`."""
        controller_state = state.get("controller")
        response_enabled = bool(
            getattr(self.controller, "response_model_enabled", False)
        )
        if response_enabled and not isinstance(
            controller_state,
            Mapping,
        ):
            raise ValueError(
                "Enabled response model requires valid controller checkpoint state."
            )
        coverage = state.get("source_coverage", {})
        restored: Dict[str, Dict[str, Any]] = {}
        if isinstance(coverage, Mapping):
            for component, entry in coverage.items():
                if not isinstance(entry, Mapping):
                    continue
                merged = self._coverage_entry(str(component))
                merged.update(dict(entry))
                restored[str(component)] = merged
        message = state.get("message")
        restored_message = (
            message
            if isinstance(message, str) and message
            else self._last_message
        )
        if isinstance(controller_state, Mapping):
            restore_controller = getattr(
                self.controller,
                "restore_checkpoint_state",
                None,
            )
            if not callable(restore_controller):
                if response_enabled:
                    raise ValueError(
                        "Enabled response model requires controller restore support."
                    )
            else:
                restore_controller(controller_state)
        # Controller restore is transactional.  Commit wrapper state only
        # after it accepts all schema/basis/cache provenance.
        self._source_coverage = restored
        self._last_message = restored_message

    def update(
        self,
        avr_swap: np.ndarray,
        *,
        ambient_velocity_global_mps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate VorLap for a single QBlade timestep.

        The passed array must contain the input blocks defined in the runtime
        config and will be modified in place with the returned force vectors.
        """
        swap = np.asarray(avr_swap)
        if swap.ndim != 1:
            raise ValueError("avr_swap must be a one-dimensional buffer.")
        if swap.size < self.swap_size:
            raise ValueError(f"avr_swap is too small: expected at least {self.swap_size}, got {swap.size}.")
        if self.use_qblade_local_inflow and ambient_velocity_global_mps is None:
            raise ValueError(
                "use_qblade_local_inflow requires nodal wind samples from the Python SIL loop."
            )
        ambient = self._ambient_velocity_global
        if ambient_velocity_global_mps is not None:
            ambient = np.asarray(ambient_velocity_global_mps, dtype=float)
            expected_shape = self.layout[self.velocity_block].shape
            if ambient.shape != expected_shape or np.any(~np.isfinite(ambient)):
                raise ValueError(
                    f"Nodal ambient velocity must be finite with shape {expected_shape}."
                )
        adapter = self._adapter_for(swap)
        if self._debug_failure_inject_update_exception:
            raise RuntimeError("debug failure injection requested before force block clearing.")
        force_view = adapter.block(self.force_block)
        force_view[...] = 0.0

        time = adapter.scalar(self.time_block)
        timestep_index = adapter.scalar(self.timestep_block)
        azimuth = adapter.scalar(self.azimuth_block)
        sanity_message = _runtime_input_sanity_message(
            time_s=time,
            timestep_index=timestep_index,
            azimuth_deg=azimuth,
        )
        if sanity_message is not None:
            return self._zero_force_output(adapter, sanity_message)

        velocity_swap = np.asarray(adapter.read(self.velocity_block), dtype=float)
        sanity_message = _runtime_input_sanity_message(
            time_s=time,
            timestep_index=timestep_index,
            azimuth_deg=azimuth,
            velocity_swap=velocity_swap,
        )
        if sanity_message is not None:
            return self._zero_force_output(adapter, sanity_message)

        if self._swap_to_controller_idx is None:
            structural_velocity = velocity_swap
            controller_structural_velocity = structural_velocity
            controller_ambient = ambient
            relative_velocity = controller_ambient - structural_velocity
            forces = self.controller.step(
                {
                    "velocity": relative_velocity,
                    "relative_velocity": relative_velocity,
                    "structural_velocity": controller_structural_velocity,
                    "declared_ambient_velocity": controller_ambient,
                    "time": time,
                    "azimuth_deg": azimuth,
                },
                copy=False,
            )
            adapter.write(self.force_block, forces)
        else:
            n_controller = int(self._controller_to_swap_idx.size)
            n_swap = int(self._swap_to_controller_idx.size)
            if self._controller_velocity_buffer is None or self._controller_velocity_buffer.shape != (n_controller, 3):
                self._controller_velocity_buffer = np.empty((n_controller, 3), dtype=float)
            if self._controller_structural_velocity_buffer is None or self._controller_structural_velocity_buffer.shape != (n_controller, 3):
                self._controller_structural_velocity_buffer = np.empty((n_controller, 3), dtype=float)
            if self._swap_force_buffer is None or self._swap_force_buffer.shape != (n_swap, 3):
                self._swap_force_buffer = np.empty((n_swap, 3), dtype=float)

            np.take(
                velocity_swap,
                self._controller_to_swap_idx,
                axis=0,
                out=self._controller_structural_velocity_buffer,
            )
            controller_structural_velocity = self._controller_structural_velocity_buffer
            controller_ambient = (
                ambient
                if ambient.ndim == 1
                else np.take(ambient, self._controller_to_swap_idx, axis=0)
            )
            np.subtract(
                controller_ambient,
                controller_structural_velocity,
                out=self._controller_velocity_buffer,
            )
            forces = self.controller.step(
                {
                    "velocity": self._controller_velocity_buffer,
                    "relative_velocity": self._controller_velocity_buffer,
                    "structural_velocity": controller_structural_velocity,
                    "declared_ambient_velocity": controller_ambient,
                    "time": time,
                    "azimuth_deg": azimuth,
                },
                copy=False,
            )
            np.take(
                forces,
                self._swap_to_controller_idx,
                axis=0,
                out=self._swap_force_buffer,
            )
            adapter.write(self.force_block, self._swap_force_buffer)

        self._last_controller_structural_velocity = np.array(controller_structural_velocity, copy=True)
        self._last_controller_forces = np.array(forces, copy=True)
        if self.sample_step is not None:
            self.record_source_coverage_sample(self.sample_step)

        if self.debug:
            cache_stats = self.controller.cache_stats
            force_norm = np.linalg.norm(forces, axis=1)
            max_idx = int(np.argmax(force_norm)) if force_norm.size else 0
            max_force = float(force_norm[max_idx]) if force_norm.size else 0.0
            max_node = self.controller.node_ids[max_idx] if force_norm.size else "?"
            max_structural_speed = float(np.max(np.linalg.norm(controller_structural_velocity, axis=1))) if controller_structural_velocity.size else 0.0
            max_relative_speed = float(np.max(np.linalg.norm(controller_ambient - controller_structural_velocity, axis=1))) if controller_structural_velocity.size else 0.0
            self._last_message = (
                f"VorLap dbg t={time:.6g}s az={azimuth:.6g}deg "
                f"max|F|={max_force:.6g}N node={max_node} "
                f"scale={self.controller.force_scale:.6g} "
                f"max|Vstruct|={max_structural_speed:.6g}m/s "
                f"max|Vrel|={max_relative_speed:.6g}m/s "
                f"cache={cache_stats['hits']}/{cache_stats['misses']}"
            )
        else:
            cache_stats = self.controller.cache_stats
            self._last_message = (
                f"VorLap update complete: t={time:.6g} s, azimuth={azimuth:.6g} deg, "
                f"nodes={forces.shape[0]}, cache={cache_stats['hits']}/{cache_stats['misses']}"
            )
        response = getattr(self.controller, "last_response_diagnostics", None)
        if isinstance(response, QBladeResponseDiagnostics):
            response_summary = self.controller.response_diagnostics_summary
            self._last_message += (
                f" response_model={self.response_model_type} "
                f" response_A/D={response.amplitude_ratio:.6g} "
                f"activation={response.activation_max:.6g} "
                f"response_power={response.modal_power_w:.6g}W "
                f"response_structural_power={response.structural_power_w:.6g}W "
                f"gain_clipped_nodes={response.gain_limited_nodes} "
                f"force_clipped_nodes={response.force_limited_nodes} "
                f"gain_clipped_node_samples="
                f"{response_summary['gain_limited_node_samples']} "
                f"force_clipped_node_samples="
                f"{response_summary['force_limited_node_samples']}"
            )
            if isinstance(
                response,
                (
                    QBladeSeededResponseDiagnostics,
                    QBladeSeededResponseDiagnosticsV2,
                ),
            ):
                self._last_message += (
                    f" seed_active={int(response.startup_seed_active)} "
                    f"seed_taper={response.startup_seed_taper:.6g} "
                    f"seed_acceleration="
                    f"{response.startup_seed_modal_acceleration_mps2:.6g}m/s2 "
                    f"seed_force={response.startup_seed_raw_generalized_force_n:.6g}N "
                    f"seed_target_A/D="
                    f"{response.startup_seed_target_amplitude_ratio:.6g} "
                    f"seed_force_clipped_nodes="
                    f"{response.startup_seed_force_limited_nodes} "
                    f"seed_force_clipped_node_samples="
                    f"{response_summary['startup_seed_force_limited_node_samples']}"
                )
        elif isinstance(response, QBladeModalPowerDiagnostics):
            response_summary = self.controller.response_diagnostics_summary
            self._last_message += (
                f" response_model={self.response_model_type} "
                f"response_A={response.amplitude_estimate_m:.6g}m "
                f"target_cycle_power={response.target_cycle_mean_power_w:.6g}W "
                f"applied_target_cycle_power="
                f"{response.applied_target_cycle_mean_power_w:.6g}W "
                f"modal_power_inst={response.instantaneous_modal_power_w:.6g}W "
                f"structural_power_inst="
                f"{response.instantaneous_structural_power_w:.6g}W "
                f"modal_power_ema={response.realized_modal_power_ema_w:.6g}W "
                f"structural_power_ema="
                f"{response.realized_structural_power_ema_w:.6g}W "
                f"target_power_limited={int(response.target_power_limited)} "
                f"generalized_force_limited="
                f"{int(response.generalized_force_limited)} "
                f"nodal_force_limited_nodes="
                f"{response.nodal_force_limited_nodes} "
                f"target_power_limited_steps="
                f"{response_summary['target_power_limited_steps']} "
                f"generalized_force_limited_steps="
                f"{response_summary['generalized_force_limited_steps']} "
                f"nodal_force_limited_node_samples="
                f"{response_summary['nodal_force_limited_node_samples']}"
            )
        self._append_log_line(self._last_message)
        return adapter.read(self.force_block, copy=True)

    def update_message(self) -> str:
        """Return the last human-readable status string."""
        return self._last_message


def create_runtime(param_file: str) -> VorLapQBladeRuntime:
    """Public factory used by the C++ bridge."""
    return VorLapQBladeRuntime.from_qblade_config(param_file)


__all__ = [
    "ExternalInputRow",
    "ExternalLibraryTableSpec",
    "ExternalOutputRow",
    "GEOMETRY_AUDIT_COLUMNS",
    "GeometryAuditRow",
    "NodeID",
    "SOURCE_COVERAGE_COLUMNS",
    "SourceCoverageRow",
    "SwapLayout",
    "VorLapQBladeRuntime",
    "build_external_library_table_spec",
    "build_qblade_geometry_audit_rows",
    "build_qblade_external_config",
    "create_runtime",
    "load_airfoil_fft_directory",
    "patch_qblade_structural_definition",
    "patch_qblade_turbine_definition",
    "write_qblade_external_config",
    "write_qblade_geometry_audit",
]
