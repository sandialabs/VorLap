"""Provenance-bound force--motion work maps for airfoil VIV closures.

The stationary airfoil FFT database can identify shedding frequencies and
available fluctuating-force energy.  It cannot identify aerodynamic work,
because its phase origin is not synchronized with structural motion.  This
module therefore keeps the missing quantity explicit: a prescribed-motion
dataset supplies a signed, dimensionless cycle-mean power coefficient

``C_P = <F'_e ydot> / (0.5 rho U**3 c span)``.

Positive ``C_P`` means that the *incremental* aerodynamic force does work on
the structural coordinate.  Qualification maps are expected to contain the
difference between a high-fidelity prescribed-motion result and the exact
runtime base model (normally native QBlade for the Ehrmann case).  This
prevents the supplemental bridge force from double counting native
aerodynamic work.

The map axes are incidence, Reynolds number, reduced frequency
``k = omega*c/(2*U)``, local amplitude ratio ``A/c``, and the in-plane motion
direction modulo 180 degrees.  Interpolation is convex multilinear (linear in
``log(Re)``) and never extrapolates.  The first amplitude interval uses a
quadratic origin continuation, so target power is ``O(A**2)`` and its
equivalent velocity gain remains finite.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "vorlap-airfoil-force-motion-work-map-v1"
_AXIS_NAMES = (
    "incidence_deg",
    "reynolds_number",
    "reduced_frequency_k",
    "amplitude_ratio",
    "direction_deg_mod_180",
)
_FIELD_NAMES = (
    "entrainment_fraction",
    "reference_available_force_coefficient",
    "signed_power_coefficient",
    "power_coefficient_standard_uncertainty",
)
_FLOAT_TOL = 1.0e-12


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _strict_axis(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional axis.")
    if array.size > 1 and np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return np.array(array, dtype=float, copy=True)


@dataclass(frozen=True)
class AirfoilWorkMapQuery:
    """One interpolated map query and its conservative injection value."""

    incidence_deg: float
    reynolds_number: float
    reduced_frequency_k: float
    amplitude_ratio: float
    direction_deg_mod_180: float
    entrainment_fraction: float
    reference_available_force_coefficient: float
    runtime_available_force_coefficient: float
    available_force_amplitude_scale: float
    signed_power_coefficient: float
    power_coefficient_standard_uncertainty: float
    confidence_multiplier: float
    lower_confidence_power_coefficient: float
    positive_injection_power_coefficient: float
    scaled_positive_injection_power_coefficient: float


class AirfoilForceMotionWorkMap:
    """Strict immutable five-dimensional prescribed-motion work map."""

    def __init__(self, payload: Mapping[str, Any], *, source_sha256: str = ""):
        if not isinstance(payload, Mapping):
            raise TypeError("Airfoil work-map payload must be a mapping.")
        expected = {
            "schema",
            "status",
            "profile",
            "motion",
            "axes",
            "fields",
            "conventions",
            "sources",
        }
        if set(payload) != expected:
            raise ValueError("Airfoil work-map top-level keys differ from the schema.")
        if payload["schema"] != SCHEMA or payload["status"] != "calibrated":
            raise ValueError("Airfoil work map is not a calibrated supported schema.")
        if not str(payload["profile"]).strip():
            raise ValueError("Airfoil work-map profile must not be empty.")
        if payload["motion"] != "prescribed_crossflow_translation":
            raise ValueError("Only prescribed cross-flow translation is supported.")

        axes_raw = payload["axes"]
        if not isinstance(axes_raw, Mapping) or set(axes_raw) != set(_AXIS_NAMES):
            raise ValueError("Airfoil work-map axes differ from the schema.")
        axes = tuple(_strict_axis(axes_raw[name], name) for name in _AXIS_NAMES)
        incidence, reynolds, reduced_frequency, amplitude, direction = axes
        if np.any(reynolds <= 0.0):
            raise ValueError("reynolds_number must be positive.")
        if np.any(reduced_frequency <= 0.0):
            raise ValueError("reduced_frequency_k must be positive.")
        if (
            amplitude.size < 2
            or amplitude[0] != 0.0
            or amplitude[1] <= 0.0
            or np.any(amplitude < 0.0)
        ):
            raise ValueError(
                "amplitude_ratio must start at zero and include a positive plane."
            )
        if direction[0] < 0.0 or direction[-1] >= 180.0:
            raise ValueError("direction_deg_mod_180 must lie in [0, 180).")

        fields_raw = payload["fields"]
        if not isinstance(fields_raw, Mapping) or set(fields_raw) != set(_FIELD_NAMES):
            raise ValueError("Airfoil work-map fields differ from the schema.")
        shape = tuple(axis.size for axis in axes)
        fields = {}
        for name in _FIELD_NAMES:
            array = np.asarray(fields_raw[name], dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must have finite shape {shape}.")
            fields[name] = np.array(array, dtype=float, copy=True)
        if np.any(fields["entrainment_fraction"] < 0.0) or np.any(
            fields["entrainment_fraction"] > 1.0
        ):
            raise ValueError("entrainment_fraction must lie in [0, 1].")
        if np.any(fields["reference_available_force_coefficient"] <= 0.0):
            raise ValueError(
                "reference_available_force_coefficient must remain positive."
            )
        if np.any(fields["power_coefficient_standard_uncertainty"] < 0.0):
            raise ValueError("Power-coefficient uncertainty must be non-negative.")
        origin = (slice(None), slice(None), slice(None), 0, slice(None))
        for name in (
            "entrainment_fraction",
            "signed_power_coefficient",
            "power_coefficient_standard_uncertainty",
        ):
            if np.any(fields[name][origin] != 0.0):
                raise ValueError(f"{name} must be exactly zero at zero amplitude.")

        conventions = payload["conventions"]
        expected_conventions = {
            "power_coefficient",
            "power_sign",
            "reduced_frequency",
            "reynolds_interpolation",
            "amplitude_origin_interpolation",
            "out_of_domain",
            "base_model_subtraction",
        }
        if not isinstance(conventions, Mapping) or set(conventions) != expected_conventions:
            raise ValueError("Airfoil work-map conventions differ from the schema.")
        required_values = {
            "power_coefficient": "mean_incremental_force_dot_velocity/(0.5*rho*U^3*c*span)",
            "power_sign": "positive_flow_to_structure",
            "reduced_frequency": "omega*c/(2*U)",
            "reynolds_interpolation": "linear_in_log_reynolds",
            "amplitude_origin_interpolation": "quadratic_first_interval",
            "out_of_domain": "error",
            "base_model_subtraction": "high_fidelity_minus_exact_runtime_base",
        }
        if dict(conventions) != required_values:
            raise ValueError("Airfoil work-map conventions are not the frozen policy.")

        sources = payload["sources"]
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
            raise ValueError("Airfoil work map requires at least one source identity.")
        normalized_sources = []
        for source in sources:
            if not isinstance(source, Mapping) or set(source) != {
                "role",
                "label",
                "sha256",
            }:
                raise ValueError("Every airfoil work-map source has exact role/label/sha256 keys.")
            digest = str(source["sha256"]).lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("Every airfoil work-map source requires a SHA-256 digest.")
            role = str(source["role"]).strip()
            label = str(source["label"]).strip()
            if not role or not label:
                raise ValueError("Airfoil work-map source role/label must not be empty.")
            normalized_sources.append({"role": role, "label": label, "sha256": digest})
        roles = {source["role"] for source in normalized_sources}
        if "prescribed_motion_high_fidelity" not in roles or "exact_runtime_base" not in roles:
            raise ValueError(
                "Work map must bind paired high-fidelity and exact-runtime-base sources."
            )

        self.profile = str(payload["profile"]).strip()
        self.axes = axes
        self._fields = fields
        self.sources = tuple(normalized_sources)
        self.source_sha256 = str(source_sha256).lower()
        if self.source_sha256 and (
            len(self.source_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a SHA-256 digest.")
        self.payload_fingerprint = _semantic_sha256(payload)

    @classmethod
    def from_json(cls, path: Any) -> "AirfoilForceMotionWorkMap":
        map_path = Path(path)
        raw = map_path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Airfoil work-map JSON is invalid.") from exc
        return cls(payload, source_sha256=_sha256_bytes(raw))

    @property
    def fingerprint(self) -> str:
        return _semantic_sha256(
            {
                "schema": SCHEMA,
                "payload_fingerprint": self.payload_fingerprint,
                "source_sha256": self.source_sha256,
            }
        )

    def positive_work_domain_status(
        self, *, confidence_multiplier: float = 0.0
    ) -> dict[str, bool]:
        """Report whether mapped positive work is bounded at high amplitude.

        Qualification runs require the lower-confidence work to become
        non-positive on the largest amplitude plane.  Otherwise a predicted
        limit cycle could lie outside the calibrated domain and the adapter
        would only discover that after an expensive run.
        """

        multiplier = _finite_float(confidence_multiplier, "confidence_multiplier")
        if multiplier < 0.0:
            raise ValueError("confidence_multiplier must be non-negative.")
        lower = self._fields["signed_power_coefficient"] - multiplier * self._fields[
            "power_coefficient_standard_uncertainty"
        ]
        positive_nonzero = np.take(lower, range(1, self.axes[3].size), axis=3)
        high = np.take(lower, self.axes[3].size - 1, axis=3)
        return {
            "has_positive_work": bool(np.any(positive_nonzero > 0.0)),
            "highest_amplitude_plane_nonpositive": bool(np.all(high <= 0.0)),
        }

    @staticmethod
    def dimensional_power_w(
        power_coefficient: float,
        *,
        fluid_density_kgpm3: float,
        speed_mps: float,
        chord_m: float,
        span_m: float,
    ) -> float:
        coefficient = _finite_float(power_coefficient, "power_coefficient")
        density = _finite_float(fluid_density_kgpm3, "fluid_density_kgpm3")
        speed = _finite_float(speed_mps, "speed_mps")
        chord = _finite_float(chord_m, "chord_m")
        span = _finite_float(span_m, "span_m")
        if density <= 0.0 or speed < 0.0 or chord <= 0.0 or span <= 0.0:
            raise ValueError("Density/chord/span must be positive and speed non-negative.")
        return 0.5 * density * speed**3 * chord * span * coefficient

    def query(
        self,
        *,
        incidence_deg: float,
        reynolds_number: float,
        reduced_frequency_k: float,
        amplitude_ratio: float,
        direction_deg: float,
        runtime_available_force_coefficient: float,
        confidence_multiplier: float = 0.0,
    ) -> AirfoilWorkMapQuery:
        values = [
            _finite_float(incidence_deg, "incidence_deg"),
            _finite_float(reynolds_number, "reynolds_number"),
            _finite_float(reduced_frequency_k, "reduced_frequency_k"),
            _finite_float(amplitude_ratio, "amplitude_ratio"),
            _finite_float(direction_deg, "direction_deg") % 180.0,
        ]
        multiplier = _finite_float(confidence_multiplier, "confidence_multiplier")
        runtime_available = _finite_float(
            runtime_available_force_coefficient,
            "runtime_available_force_coefficient",
        )
        if (
            values[1] <= 0.0
            or values[2] <= 0.0
            or values[3] < 0.0
            or runtime_available < 0.0
            or multiplier < 0.0
        ):
            raise ValueError("Re/k must be positive; amplitude/confidence must be non-negative.")

        transformed_axes = list(self.axes)
        transformed_axes[1] = np.log(self.axes[1])
        transformed_values = list(values)
        transformed_values[1] = math.log(values[1])
        brackets = []
        origin_scale = 1.0
        for axis_index, (axis, value) in enumerate(zip(transformed_axes, transformed_values)):
            if axis_index == 4 and axis.size > 1 and self.axes[4][0] == 0.0:
                # Direction is an unoriented line: interpolate across the
                # 180/0-degree seam without duplicating a map plane.
                if value > axis[-1] + _FLOAT_TOL:
                    fraction = (value - axis[-1]) / (180.0 - axis[-1])
                    brackets.append((axis.size - 1, 0, float(fraction)))
                    continue
            if value < axis[0] - _FLOAT_TOL or value > axis[-1] + _FLOAT_TOL:
                raise ValueError(
                    f"{_AXIS_NAMES[axis_index]}={values[axis_index]:.17g} is outside "
                    f"the calibrated domain [{self.axes[axis_index][0]:.17g}, "
                    f"{self.axes[axis_index][-1]:.17g}]."
                )
            value = min(max(value, axis[0]), axis[-1])
            if axis_index == 3 and axis.size > 1 and value < axis[1]:
                origin_scale = (value / axis[1]) ** 2
                brackets.append((1, 1, 0.0))
                continue
            if axis.size == 1:
                brackets.append((0, 0, 0.0))
                continue
            right = int(np.searchsorted(axis, value, side="right"))
            right = min(max(right, 1), axis.size - 1)
            left = right - 1
            if math.isclose(value, axis[right], rel_tol=0.0, abs_tol=_FLOAT_TOL):
                left = right
                weight = 0.0
            else:
                weight = (value - axis[left]) / (axis[right] - axis[left])
            brackets.append((left, right, float(weight)))

        def interpolate(array: np.ndarray, *, apply_origin_scale: bool = True) -> float:
            total = 0.0
            choices = []
            for left, right, fraction in brackets:
                if left == right:
                    choices.append(((left, 1.0),))
                else:
                    choices.append(((left, 1.0 - fraction), (right, fraction)))
            for corner in itertools.product(*choices):
                index = tuple(item[0] for item in corner)
                weight = math.prod(item[1] for item in corner)
                total += weight * float(array[index])
            return total * (origin_scale if apply_origin_scale else 1.0)

        entrainment = interpolate(self._fields["entrainment_fraction"])
        reference_available = interpolate(
            self._fields["reference_available_force_coefficient"],
            apply_origin_scale=False,
        )
        if reference_available <= 0.0:
            raise FloatingPointError(
                "Interpolated reference available-force coefficient is not positive."
            )
        available_scale = runtime_available / reference_available
        signed = interpolate(self._fields["signed_power_coefficient"])
        uncertainty = interpolate(
            self._fields["power_coefficient_standard_uncertainty"]
        )
        lower = signed - multiplier * uncertainty
        positive = max(0.0, lower)
        return AirfoilWorkMapQuery(
            incidence_deg=values[0],
            reynolds_number=values[1],
            reduced_frequency_k=values[2],
            amplitude_ratio=values[3],
            direction_deg_mod_180=values[4],
            entrainment_fraction=entrainment,
            reference_available_force_coefficient=reference_available,
            runtime_available_force_coefficient=runtime_available,
            available_force_amplitude_scale=available_scale,
            signed_power_coefficient=signed,
            power_coefficient_standard_uncertainty=uncertainty,
            confidence_multiplier=multiplier,
            lower_confidence_power_coefficient=lower,
            positive_injection_power_coefficient=positive,
            scaled_positive_injection_power_coefficient=(
                positive * available_scale
            ),
        )


def frozen_conventions() -> dict[str, str]:
    """Return the exact convention block required by schema v1."""

    return {
        "power_coefficient": "mean_incremental_force_dot_velocity/(0.5*rho*U^3*c*span)",
        "power_sign": "positive_flow_to_structure",
        "reduced_frequency": "omega*c/(2*U)",
        "reynolds_interpolation": "linear_in_log_reynolds",
        "amplitude_origin_interpolation": "quadratic_first_interval",
        "out_of_domain": "error",
        "base_model_subtraction": "high_fidelity_minus_exact_runtime_base",
    }
