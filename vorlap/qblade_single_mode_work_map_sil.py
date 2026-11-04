"""Single-mode SIL adapter for prescribed-motion airfoil work maps.

This path is deliberately separate from the legacy spectral lock-in adapter.
Stationary airfoil spectra supply only the available fluctuating lift scale
and source-frequency diagnostic.  A provenance-bound prescribed-motion map
supplies signed incremental cycle-mean work.  Positive lower-confidence work
is applied through the energy-conjugate modal-power actuator; QBlade retains
native aerodynamics and structural Rayleigh damping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .airfoil_io import load_airfoil_fft
from .force_model import resolve_airfoil, section_flow_state, strouhal_reference_length
from .interpolation import interpolate_fft_spectrum
from .qblade_airfoil_work_map import AirfoilForceMotionWorkMap
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
from .qblade_spectral_lockin import airfoil_fft_content_sha256


MODEL_TYPE = "single_mode_airfoil_force_motion_work_map_v1"
_EPS = 1.0e-12


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive.")
    return result


def _strict_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _resolve_relative_file(parent: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty relative path.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
        raise ValueError(f"{label} must be a canonical relative path.")
    path = parent.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must resolve to a regular non-symlink file.")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_airfoil_source_identities(
    runtime: Any, active_airfoils: tuple[Any, ...]
) -> tuple[dict[str, str], ...]:
    """Resolve the raw HDF5 source used for every active parsed airfoil.

    The directory is replayed in the same sorted/name-keyed order as the
    runtime loader.  This prevents an unused byte-identical HDF5 file from
    masking a different file that actually supplied an active airfoil object.
    """

    raw_directory = getattr(runtime, "resolved_airfoil_dir", None)
    if not isinstance(raw_directory, str) or not raw_directory:
        raise ValueError("The work-map runtime lacks a resolved airfoil source directory.")
    directory = Path(raw_directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(
            "The work-map runtime airfoil source must be a regular non-symlink directory."
        )
    sources_by_name: dict[str, dict[str, str]] = {}
    for path in sorted(directory.glob("*.h5")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "The work-map runtime airfoil sources must be regular non-symlink files."
            )
        parsed = load_airfoil_fft(str(path))
        sources_by_name[parsed.name] = {
            "airfoil_name": parsed.name,
            "file": path.name,
            "raw_sha256": _sha256_file(path),
            "content_sha256": airfoil_fft_content_sha256(parsed),
        }
    if not sources_by_name:
        raise ValueError("The work-map runtime airfoil source directory is empty.")

    identities = []
    for airfoil in active_airfoils:
        source = sources_by_name.get(airfoil.name)
        if source is None:
            raise ValueError(
                f"No runtime HDF5 source identity is available for airfoil {airfoil.name!r}."
            )
        active_content = airfoil_fft_content_sha256(airfoil)
        if source["content_sha256"] != active_content:
            raise ValueError(
                "Runtime-loaded airfoil content differs from its resolved HDF5 source."
            )
        identities.append(dict(source))
    return tuple(identities)


def _mode_shape_in_node_order(
    power_model: QBladeModalPowerModel, node_ids: tuple[str, ...]
) -> np.ndarray:
    """Return the canonical modal basis associated by ID with ``node_ids``."""

    canonical_ids = tuple(str(value) for value in power_model._canonical_node_ids)
    requested_ids = tuple(str(value) for value in node_ids)
    if len(requested_ids) != len(canonical_ids) or set(requested_ids) != set(
        canonical_ids
    ):
        raise ValueError("Controller node IDs differ from the modal-power basis.")
    canonical_mode_by_node = dict(zip(canonical_ids, power_model._mode_shape))
    return np.asarray(
        [canonical_mode_by_node[node_id] for node_id in requested_ids], dtype=float
    )


def _lift_unit(
    chord: np.ndarray,
    normal: np.ndarray,
    velocity_chord_mps: np.ndarray,
    velocity_normal_mps: np.ndarray,
    speed_mps: np.ndarray,
) -> np.ndarray:
    """Return the wind-axis lift unit in the local chord/normal plane.

    The convention gives ``+normal`` for flow along ``+chord``.  Reversing
    this unit cannot change injected work because the motion direction is an
    unoriented modal line and the available-force projection uses magnitude.
    """

    speed = np.asarray(speed_mps, dtype=float)
    if np.any(speed <= _EPS):
        raise ValueError("Work-map flow speed must remain positive at every node.")
    return (
        -np.asarray(velocity_normal_mps)[:, None] * chord
        + np.asarray(velocity_chord_mps)[:, None] * normal
    ) / speed[:, None]


@dataclass(frozen=True)
class WorkMapStepDiagnostics:
    map_fingerprint: str
    static_spectrum_fingerprint: str
    source_frequency_ratio_min: float
    source_frequency_ratio_max: float
    incidence_min_deg: float
    incidence_max_deg: float
    reynolds_min: float
    reynolds_max: float
    reduced_frequency_min: float
    reduced_frequency_max: float
    local_amplitude_ratio_min: float
    local_amplitude_ratio_max: float
    available_lift_coefficient_min: float
    available_lift_coefficient_max: float
    entrainment_fraction_min: float
    entrainment_fraction_max: float
    signed_power_coefficient_min: float
    signed_power_coefficient_max: float
    scaled_positive_power_coefficient_min: float
    scaled_positive_power_coefficient_max: float
    target_aerodynamic_power_w: float
    structural_rayleigh_dissipation_w: float
    power_balance_w: float


class QBladeSingleModeWorkMapSilAdapter:
    """Apply a prescribed-motion work map to one fixed, non-rotating mode."""

    def __init__(
        self,
        *,
        raw_config: Mapping[str, Any],
        parameter_file: Path,
        runtime: Any,
    ) -> None:
        if not isinstance(raw_config, Mapping):
            raise TypeError("single_mode_airfoil_work_map must be a mapping.")
        expected = {
            "schema_version",
            "type",
            "profile",
            "kinematics",
            "work_map",
            "static_spectrum",
            "modal_power",
            "structural_damping_ratio",
            "residual_source_scale",
        }
        if set(raw_config) != expected:
            raise ValueError("single_mode_airfoil_work_map keys differ from the schema.")
        if raw_config["schema_version"] != 1 or raw_config["type"] != MODEL_TYPE:
            raise ValueError("Unsupported single-mode airfoil work-map schema/type.")
        if raw_config["kinematics"] != "fixed_nonrotating":
            raise ValueError("The v1 work-map adapter supports fixed non-rotating sections only.")
        if float(raw_config["residual_source_scale"]) != 0.0:
            raise ValueError("The work-map adapter requires residual_source_scale=0.")

        controller = runtime.controller
        seed_model = controller.response_model
        if not isinstance(seed_model, QBladeResponseModel):
            raise ValueError("The work-map adapter requires the schema-v2 modal seed model.")
        if (
            seed_model.schema_version
            != RESPONSE_MODEL_SCHEMA_VERSION_DECLARED_AMBIENT_CROSSFLOW
            or seed_model.aerodynamic_speed_basis
            != AERODYNAMIC_SPEED_BASIS_DECLARED_AMBIENT_CROSSFLOW
            or not seed_model.config.enabled
        ):
            raise ValueError("The work-map adapter requires enabled response schema v2.")
        if float(seed_model.config.aerodynamic_damping_gain_h1) != 0.0:
            raise ValueError("Cylinder-derived H1 must be exactly zero on the work-map path.")
        if float(controller.force_scale) != 0.0 or bool(controller.include_mean_load):
            raise ValueError("The work-map adapter requires force_scale=0 and no mean source load.")
        if seed_model.config.projection_weights is None:
            raise ValueError("The work-map adapter requires nodal mass projection_weights.")
        if str(seed_model.config.force_direction) not in {"chord", "normal"}:
            raise ValueError("The work-map modal direction must be chord or normal.")

        map_raw = raw_config["work_map"]
        if not isinstance(map_raw, Mapping) or set(map_raw) != {
            "path",
            "sha256",
            "confidence_multiplier",
            "require_nonpositive_high_amplitude_work",
        }:
            raise ValueError("work_map keys differ from the schema.")
        map_path = _resolve_relative_file(
            Path(parameter_file).resolve().parent,
            map_raw["path"],
            "work_map.path",
        )
        expected_map_sha = str(map_raw["sha256"]).lower()
        actual_map_sha = hashlib.sha256(map_path.read_bytes()).hexdigest()
        if expected_map_sha != actual_map_sha:
            raise ValueError("work_map SHA-256 differs from the staged bytes.")
        work_map = AirfoilForceMotionWorkMap.from_json(map_path)
        if work_map.profile != str(raw_config["profile"]):
            raise ValueError("Configured profile differs from the work-map profile.")
        confidence_multiplier = _strict_nonnegative(
            map_raw["confidence_multiplier"], "work_map.confidence_multiplier"
        )
        if map_raw["require_nonpositive_high_amplitude_work"] is not True:
            raise ValueError(
                "Qualification work maps must require non-positive work on the "
                "highest-amplitude plane."
            )
        domain_status = work_map.positive_work_domain_status(
            confidence_multiplier=confidence_multiplier
        )
        if not domain_status["has_positive_work"]:
            raise ValueError("The work map contains no positive injection branch.")
        if not domain_status["highest_amplitude_plane_nonpositive"]:
            raise ValueError(
                "The work map does not bound positive work at its highest amplitude."
            )

        spectrum = raw_config["static_spectrum"]
        expected_spectrum = {
            "channel",
            "force_axis",
            "amplitude_measure",
            "source_frequency_measure",
            "bounds_policy",
            "source_sha256",
        }
        if not isinstance(spectrum, Mapping) or set(spectrum) != expected_spectrum:
            raise ValueError("static_spectrum keys differ from the schema.")
        required_spectrum = {
            "channel": "CL",
            "force_axis": "wind_axis_lift",
            "amplitude_measure": "rss_nonzero_frequency_harmonics",
            "source_frequency_measure": "energy_centroid",
            "bounds_policy": "raise",
        }
        if {key: spectrum[key] for key in required_spectrum} != required_spectrum:
            raise ValueError("static_spectrum policy differs from the frozen contract.")
        configured_spectrum_sha256 = str(spectrum["source_sha256"]).lower()
        if len(configured_spectrum_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in configured_spectrum_sha256
        ):
            raise ValueError("static_spectrum.source_sha256 must be a SHA-256 digest.")
        map_spectrum_sources = tuple(
            source
            for source in work_map.sources
            if source["role"] == "stationary_force_spectrum"
        )
        if len(map_spectrum_sources) != 1:
            raise ValueError(
                "The work map must contain exactly one stationary_force_spectrum source."
            )
        map_spectrum_sha256 = map_spectrum_sources[0]["sha256"]
        if configured_spectrum_sha256 != map_spectrum_sha256:
            raise ValueError(
                "Configured static-spectrum source differs from the work-map source."
            )

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
        if not isinstance(modal["enabled"], bool):
            raise TypeError("modal_power.enabled must be boolean.")
        if modal["limit_action"] != "error":
            raise ValueError("Qualification work-map cases require limit_action='error'.")

        damping = _strict_nonnegative(
            raw_config["structural_damping_ratio"],
            "structural_damping_ratio",
        )
        if damping >= 1.0:
            raise ValueError("structural_damping_ratio must be below one.")

        node_ids = tuple(str(value) for value in controller.node_ids)
        airfoils = tuple(
            resolve_airfoil(controller.airfoils, airfoil_id)
            for airfoil_id in controller._node_airfoil_ids
        )
        airfoil_hashes = tuple(airfoil_fft_content_sha256(value) for value in airfoils)
        runtime_source_identities = _runtime_airfoil_source_identities(runtime, airfoils)
        if any(
            source["raw_sha256"] != map_spectrum_sha256
            for source in runtime_source_identities
        ):
            raise ValueError(
                "Runtime-loaded airfoil source differs from the work map's "
                "stationary_force_spectrum source."
            )
        static_fingerprint = _semantic_sha256(
            {
                "policy": required_spectrum,
                "stationary_force_spectrum_sha256": map_spectrum_sha256,
                "n_frequency_entries": int(controller.n_freq_depth),
                "node_airfoil_content_sha256": list(airfoil_hashes),
                "node_airfoil_source_identity": list(runtime_source_identities),
            }
        )
        external_source_fingerprint = _semantic_sha256(
            {
                "model_type": MODEL_TYPE,
                "work_map_fingerprint": work_map.fingerprint,
                "static_spectrum_fingerprint": static_fingerprint,
                "confidence_multiplier": confidence_multiplier,
            }
        )
        mode_shape = tuple(float(value) for value in seed_model.config.mode_shape)
        mass_weights = tuple(
            float(value) for value in seed_model.config.projection_weights
        )
        local_length = tuple(float(value) for value in controller._local_length)
        power_model = QBladeModalPowerModel(
            QBladeModalPowerModelConfig(
                node_ids=node_ids,
                mode_shape=mode_shape,
                modal_mass_weights_kg=mass_weights,
                # This supplemental force is an energy-conjugate generalized
                # modal actuator, not a sectional aerodynamic load model.  Use
                # the same mass-weighted dual for modal velocity and force so
                # sum_i(F_i dot v_i) == Q*qdot for arbitrary nodal motion, not
                # only for a velocity field proportional to the mode shape.
                # Local aerodynamic/span weights remain confined to the
                # sectional target-power integration below.
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
                    modal["max_abs_nodal_force_n"],
                    "modal_power.max_abs_nodal_force_n",
                ),
                bandpass_damping_ratio=_strict_positive(
                    modal["bandpass_damping_ratio"],
                    "modal_power.bandpass_damping_ratio",
                ),
                force_direction=str(seed_model.config.force_direction),
                limit_action="error",
                enabled=modal["enabled"],
                external_target_source_fingerprint=external_source_fingerprint,
            )
        )

        self.runtime = runtime
        self.seed_model = seed_model
        self.power_model = power_model
        self.work_map = work_map
        self.confidence_multiplier = confidence_multiplier
        self.node_ids = node_ids
        self.airfoils = airfoils
        self.static_spectrum_source_sha256 = map_spectrum_sha256
        self.static_spectrum_fingerprint = static_fingerprint
        self.external_source_fingerprint = external_source_fingerprint
        self.structural_damping_ratio = damping
        self.residual_source_scale = 0.0
        self.model_type = MODEL_TYPE
        self.schema_version = 1
        self.config_fingerprint = _semantic_sha256(
            {
                "raw_config": raw_config,
                "seed_model_fingerprint": seed_model.config_fingerprint,
                "power_model_fingerprint": power_model.checkpoint_state()[
                    "config_fingerprint"
                ],
                "work_map_fingerprint": work_map.fingerprint,
                "static_spectrum_fingerprint": static_fingerprint,
            }
        )
        # QBladeModalPowerModel canonicalizes node IDs internally.  The flow,
        # chord, and span arrays used by this adapter remain in controller
        # order, so explicitly map the mode back by node ID before computing
        # local A/c.  Positional reuse of the canonical array silently
        # associates the wrong amplitude with a section when controller nodes
        # are permuted.
        self._mode_shape = _mode_shape_in_node_order(power_model, node_ids)
        self._generalized_mass_kg = float(power_model._projection_denominator)
        self._amplitude_origin_axis = float(work_map.axes[3][1])
        self.last_power_result = None
        self.last_seed_result = None
        self.last_diagnostics: WorkMapStepDiagnostics | None = None

    def _static_lift_scale(
        self,
        *,
        reynolds: np.ndarray,
        incidence_deg: np.ndarray,
        speed_mps: np.ndarray,
        response_width_m: np.ndarray,
        mode_direction: np.ndarray,
        lift_direction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        available = np.zeros(len(self.node_ids), dtype=float)
        centroid_frequency = np.zeros(len(self.node_ids), dtype=float)
        for index, airfoil in enumerate(self.airfoils):
            st, amplitude, _ = interpolate_fft_spectrum(
                airfoil,
                float(reynolds[index]),
                float(incidence_deg[index]),
                "CL",
                n_freq_depth=int(self.runtime.controller.n_freq_depth),
                bounds_policy="raise",
            )
            st = np.asarray(st, dtype=float)
            amplitude = np.asarray(amplitude, dtype=float)
            active = (st > 0.0) & (amplitude > 0.0)
            projection = abs(
                float(np.dot(mode_direction[index], lift_direction[index]))
            )
            energy = float(np.sum(amplitude[active] ** 2))
            available[index] = projection * math.sqrt(max(0.0, energy))
            if energy > 0.0:
                centroid_st = float(
                    np.sum(st[active] * amplitude[active] ** 2) / energy
                )
                centroid_frequency[index] = (
                    centroid_st * speed_mps[index] / response_width_m[index]
                )
        return available, centroid_frequency

    def _target_evaluator(
        self,
        *,
        flow: Any,
        chord_m: np.ndarray,
        local_length_m: np.ndarray,
        direction_deg: np.ndarray,
        runtime_available: np.ndarray,
        source_frequency_hz: np.ndarray,
    ):
        omega = 2.0 * math.pi * float(self.power_model.config.natural_frequency_hz)
        nonzero = np.abs(self._mode_shape) > _EPS
        origin_candidates = (
            self._amplitude_origin_axis
            * chord_m[nonzero]
            / np.abs(self._mode_shape[nonzero])
        )
        origin_reference_amplitude = 0.5 * float(np.min(origin_candidates))
        if not math.isfinite(origin_reference_amplitude) or origin_reference_amplitude <= 0.0:
            raise ValueError("The work-map quadratic-origin reference is invalid.")

        def evaluate_total(amplitude_m: float):
            queries = []
            contributions = []
            for index in range(len(self.node_ids)):
                local_ratio = abs(self._mode_shape[index]) * amplitude_m / chord_m[index]
                query = self.work_map.query(
                    incidence_deg=float(flow.aoa_deg[index]),
                    reynolds_number=float(flow.reynolds[index]),
                    reduced_frequency_k=float(
                        omega * chord_m[index] / (2.0 * flow.effective_velocity_mps[index])
                    ),
                    amplitude_ratio=float(local_ratio),
                    direction_deg=float(direction_deg[index]),
                    runtime_available_force_coefficient=float(runtime_available[index]),
                    confidence_multiplier=self.confidence_multiplier,
                )
                queries.append(query)
                contributions.append(
                    self.work_map.dimensional_power_w(
                        query.scaled_positive_injection_power_coefficient,
                        fluid_density_kgpm3=float(
                            self.runtime.controller.fluid_density
                        ),
                        speed_mps=float(flow.effective_velocity_mps[index]),
                        chord_m=float(chord_m[index]),
                        span_m=float(local_length_m[index]),
                    )
                )
            return float(math.fsum(contributions)), tuple(queries), tuple(contributions)

        origin_power, _, _ = evaluate_total(origin_reference_amplitude)
        origin_gain = 2.0 * origin_power / (
            omega**2 * origin_reference_amplitude**2
        )

        def evaluator(amplitude_m: float) -> ExternalModalPowerTarget:
            target, queries, contributions = evaluate_total(amplitude_m)
            context = _semantic_sha256(
                {
                    "map_fingerprint": self.work_map.fingerprint,
                    "static_spectrum_fingerprint": self.static_spectrum_fingerprint,
                    "amplitude_m": float(amplitude_m),
                    "incidence_deg": [float(v) for v in flow.aoa_deg],
                    "reynolds_number": [float(v) for v in flow.reynolds],
                    "speed_mps": [float(v) for v in flow.effective_velocity_mps],
                    "direction_deg": [float(v) for v in direction_deg],
                    "runtime_available_force_coefficient": [
                        float(v) for v in runtime_available
                    ],
                    "source_frequency_hz": [float(v) for v in source_frequency_hz],
                    "node_power_w": [float(v) for v in contributions],
                    "query": [asdict(value) for value in queries],
                    "target_power_w": target,
                    "origin_gain_kgps": origin_gain,
                }
            )
            evaluator.last_queries = queries
            return ExternalModalPowerTarget(target, origin_gain, context)

        evaluator.last_queries = ()
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
        wrapped_azimuth = math.remainder(azimuth, 360.0)
        if not math.isclose(
            wrapped_azimuth, 0.0, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError("The fixed-nonrotating work-map adapter requires zero azimuth.")
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
        response_width = strouhal_reference_length(
            chord_m, flow.aoa_deg, thickness
        )
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
        direction_deg = np.degrees(
            np.arctan2(
                np.sum(mode_direction * normal, axis=1),
                np.sum(mode_direction * chord, axis=1),
            )
        ) % 180.0
        runtime_available, source_frequency = self._static_lift_scale(
            reynolds=flow.reynolds,
            incidence_deg=flow.aoa_deg,
            speed_mps=flow.effective_velocity_mps,
            response_width_m=response_width,
            mode_direction=mode_direction,
            lift_direction=lift_direction,
        )
        evaluator = self._target_evaluator(
            flow=flow,
            chord_m=chord_m,
            local_length_m=local_length,
            direction_deg=direction_deg,
            runtime_available=runtime_available,
            source_frequency_hz=source_frequency,
        )

        seed_snapshot = self.seed_model.checkpoint_state()
        power_snapshot = self.power_model.checkpoint_state()
        previous = (self.last_seed_result, self.last_power_result, self.last_diagnostics)
        try:
            zero_activation = np.zeros(len(self.node_ids), dtype=float)
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
                relative_crossflow_speed_mps=section_flow_state(
                    ambient - velocity,
                    chord,
                    normal,
                    chord_m,
                    local_length,
                    float(controller.fluid_density),
                    float(controller.fluid_dynamicviscosity),
                ).effective_velocity_mps,
                response_section_width_basis="declared_ambient_crossflow_angle",
                activation_override=zero_activation,
            )
            power_result = self.power_model.step(
                time_value,
                node_ids=self.node_ids,
                direction_basis=mode_direction,
                structural_velocity_mps=velocity,
                target_power_evaluator=evaluator,
            )
            queries = evaluator.last_queries
            if len(queries) != len(self.node_ids):
                raise RuntimeError("Work-map query diagnostics are incomplete.")
            power_diag = power_result.diagnostics
            structural_power = (
                2.0
                * self.structural_damping_ratio
                * (2.0 * math.pi * float(self.power_model.config.natural_frequency_hz))
                * self._generalized_mass_kg
                * power_diag.qdot_rms_mps**2
            )
            source_ratio = source_frequency / float(
                self.power_model.config.natural_frequency_hz
            )
            self.last_diagnostics = WorkMapStepDiagnostics(
                map_fingerprint=self.work_map.fingerprint,
                static_spectrum_fingerprint=self.static_spectrum_fingerprint,
                source_frequency_ratio_min=float(np.min(source_ratio)),
                source_frequency_ratio_max=float(np.max(source_ratio)),
                incidence_min_deg=float(np.min(flow.aoa_deg)),
                incidence_max_deg=float(np.max(flow.aoa_deg)),
                reynolds_min=float(np.min(flow.reynolds)),
                reynolds_max=float(np.max(flow.reynolds)),
                reduced_frequency_min=float(
                    np.min(
                        2.0
                        * math.pi
                        * float(self.power_model.config.natural_frequency_hz)
                        * chord_m
                        / (2.0 * flow.effective_velocity_mps)
                    )
                ),
                reduced_frequency_max=float(
                    np.max(
                        2.0
                        * math.pi
                        * float(self.power_model.config.natural_frequency_hz)
                        * chord_m
                        / (2.0 * flow.effective_velocity_mps)
                    )
                ),
                local_amplitude_ratio_min=float(
                    np.min(
                        np.abs(self._mode_shape)
                        * power_diag.amplitude_estimate_m
                        / chord_m
                    )
                ),
                local_amplitude_ratio_max=float(
                    np.max(
                        np.abs(self._mode_shape)
                        * power_diag.amplitude_estimate_m
                        / chord_m
                    )
                ),
                available_lift_coefficient_min=float(np.min(runtime_available)),
                available_lift_coefficient_max=float(np.max(runtime_available)),
                entrainment_fraction_min=min(q.entrainment_fraction for q in queries),
                entrainment_fraction_max=max(q.entrainment_fraction for q in queries),
                signed_power_coefficient_min=min(q.signed_power_coefficient for q in queries),
                signed_power_coefficient_max=max(q.signed_power_coefficient for q in queries),
                scaled_positive_power_coefficient_min=min(
                    q.scaled_positive_injection_power_coefficient for q in queries
                ),
                scaled_positive_power_coefficient_max=max(
                    q.scaled_positive_injection_power_coefficient for q in queries
                ),
                target_aerodynamic_power_w=power_diag.target_cycle_mean_power_w,
                structural_rayleigh_dissipation_w=structural_power,
                power_balance_w=(
                    power_diag.target_cycle_mean_power_w - structural_power
                ),
            )
        except Exception:
            self.seed_model.restore_checkpoint_state(seed_snapshot)
            self.power_model.restore_checkpoint_state(power_snapshot)
            self.last_seed_result, self.last_power_result, self.last_diagnostics = previous
            raise
        self.last_seed_result = seed_result
        self.last_power_result = power_result
        return np.asarray(seed_result.additive_force_n) + np.asarray(
            power_result.additive_force_n
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_type": self.model_type,
            "config_fingerprint": self.config_fingerprint,
            "seed_model": self.seed_model.checkpoint_state(),
            "modal_power": self.power_model.checkpoint_state(),
        }

    def restore_checkpoint_state(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version",
            "model_type",
            "config_fingerprint",
            "seed_model",
            "modal_power",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Single-mode work-map checkpoint keys differ.")
        if (
            state["schema_version"] != 1
            or state["model_type"] != self.model_type
            or state["config_fingerprint"] != self.config_fingerprint
        ):
            raise ValueError("Single-mode work-map checkpoint identity mismatch.")
        seed_snapshot = self.seed_model.checkpoint_state()
        power_snapshot = self.power_model.checkpoint_state()
        try:
            self.seed_model.restore_checkpoint_state(state["seed_model"])
            self.power_model.restore_checkpoint_state(state["modal_power"])
        except Exception:
            self.seed_model.restore_checkpoint_state(seed_snapshot)
            self.power_model.restore_checkpoint_state(power_snapshot)
            raise
        self.last_seed_result = None
        self.last_power_result = None
        self.last_diagnostics = None

    def history_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {
            "response_model_type": self.model_type,
            "response_work_map_fingerprint": self.work_map.fingerprint,
            "response_static_spectrum_fingerprint": self.static_spectrum_fingerprint,
            "response_static_spectrum_source_sha256": self.static_spectrum_source_sha256,
            "response_residual_source_scale": 0.0,
        }
        if self.last_power_result is not None:
            for name, value in asdict(self.last_power_result.diagnostics).items():
                fields[f"response_power_{name}"] = value
        if self.last_seed_result is not None:
            for name, value in asdict(self.last_seed_result.diagnostics).items():
                fields[f"response_seed_{name}"] = value
        if self.last_diagnostics is not None:
            for name, value in asdict(self.last_diagnostics).items():
                fields[f"response_work_map_{name}"] = value
        return fields

    def diagnostics_summary(self) -> dict[str, object]:
        result: dict[str, object] = {
            "model_type": self.model_type,
            "config_fingerprint": self.config_fingerprint,
            "work_map_fingerprint": self.work_map.fingerprint,
            "work_map_source_sha256": self.work_map.source_sha256,
            "static_spectrum_fingerprint": self.static_spectrum_fingerprint,
            "static_spectrum_source_sha256": self.static_spectrum_source_sha256,
            "residual_source_scale": 0.0,
            "samples": 0 if self.last_power_result is None else 1,
        }
        if self.last_power_result is not None:
            result["modal_power"] = asdict(self.last_power_result.diagnostics)
        if self.last_seed_result is not None:
            result["startup_seed"] = asdict(self.last_seed_result.diagnostics)
        if self.last_diagnostics is not None:
            result["work_map"] = asdict(self.last_diagnostics)
        return result


__all__ = [
    "MODEL_TYPE",
    "QBladeSingleModeWorkMapSilAdapter",
    "WorkMapStepDiagnostics",
]
