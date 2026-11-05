"""Frozen common gain state for the turbine and H-VAWT VIV closures.

The two qualified legacy branches use the same signed quadratic work law but
store their zero-amplitude gain in different coordinates.  This module keeps
that distinction explicit: the H-VAWT branch multiplies an existing spectral
and modal basis, while the Horcas turbine branch multiplies opposing damping.
No fitting is performed here and requests outside the frozen anchor domains
fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import numbers
from pathlib import Path
from typing import Any, Mapping

from .qblade_engineering_lockin import EngineeringLockInCalibration, semantic_sha256


SCHEMA = "vorlap-unified-viv-gain-model-v1"
STATUS = "frozen_before_unified_parity_and_hvawt_application"
AIRFOIL_BRANCH = "naca0018_quasi_static_azimuth"
TURBINE_BRANCH = "ffa_w3_iea10_tip_geometry"
SPECTRAL_BASIS = "spectral_basis"
OPPOSING_DAMPING = "opposing_damping"


class UnifiedVIVGainError(ValueError):
    """Raised when the frozen unified gain contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnifiedVIVGainError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise UnifiedVIVGainError(f"{label} must be finite.")
    return result


@dataclass(frozen=True)
class UnifiedVIVGainState:
    """One evaluated zero-amplitude gain and signed-saturation state."""

    branch: str
    gain_basis: str
    zero_amplitude_gain: float
    limiting_amplitude_ratio: float
    reference_length_m: float | None
    work_margin: float | None
    active: bool | None
    profile_family: str
    kinematic_family: str
    capture_window: str | None
    frequency_ratio_low: float | None
    frequency_ratio_peak: float | None
    frequency_ratio_high: float | None
    response_frequency_ratio: float | None
    source_fingerprint: str
    unified_model_fingerprint: str
    static_spectrum_sha256: str | None = None
    static_spectrum_reynolds_model: str | None = None

    @property
    def feedback_gain(self) -> float:
        """Compatibility name for the spectral-basis branch."""

        if self.gain_basis != SPECTRAL_BASIS:
            raise UnifiedVIVGainError(
                "feedback_gain is defined only for the spectral-basis branch."
            )
        return self.zero_amplitude_gain

    @property
    def calibration_fingerprint(self) -> str:
        """Compatibility identity for existing engineering diagnostics."""

        return self.source_fingerprint

    def zero_amplitude_feedback(
        self,
        *,
        spectral_basis: float | None = None,
        opposing_damping: float | None = None,
    ) -> float:
        """Convert the tagged gain to the common ``c_a0`` coefficient."""

        if self.gain_basis == SPECTRAL_BASIS:
            _require(opposing_damping is None, "Spectral gain rejects opposing damping.")
            _require(spectral_basis is not None, "Spectral gain requires B_spectral.")
            basis = _finite(spectral_basis, "spectral_basis")
        elif self.gain_basis == OPPOSING_DAMPING:
            _require(spectral_basis is None, "Damping-ratio gain rejects B_spectral.")
            _require(opposing_damping is not None, "Damping-ratio gain requires c_s.")
            basis = _finite(opposing_damping, "opposing_damping")
        else:  # pragma: no cover - construction is validated by the loader.
            raise UnifiedVIVGainError(f"Unsupported gain basis {self.gain_basis!r}.")
        _require(basis >= 0.0, "The zero-amplitude feedback basis must be nonnegative.")
        return self.zero_amplitude_gain * basis

    def limiting_amplitude_m(self, reference_length_m: float | None = None) -> float:
        reference = self.reference_length_m if reference_length_m is None else reference_length_m
        _require(reference is not None, "A reference length is required for this branch.")
        length = _finite(reference, "reference_length_m")
        _require(length > 0.0, "reference_length_m must be positive.")
        return self.limiting_amplitude_ratio * length

    def saturation_factor_from_ratio(self, amplitude_ratio: float) -> float:
        ratio = _finite(amplitude_ratio, "amplitude_ratio")
        _require(ratio >= 0.0, "amplitude_ratio must be nonnegative.")
        return 1.0 - (ratio / self.limiting_amplitude_ratio) ** 2

    def stable_equilibrium_amplitude_m(self) -> float:
        _require(
            self.gain_basis == OPPOSING_DAMPING and self.work_margin is not None,
            "A stored work margin is required for turbine equilibrium.",
        )
        return self.limiting_amplitude_m() * math.sqrt(self.work_margin)


@dataclass(frozen=True)
class UnifiedVIVGainModel:
    """Schema-validated, hash-bound resolver for the two frozen branches."""

    raw: Mapping[str, Any]
    path: Path
    repository_root: Path
    fingerprint: str

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        repository_root: str | Path | None = None,
    ) -> "UnifiedVIVGainModel":
        resolved = Path(path).expanduser().resolve()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("Unified VIV gain model must be a JSON object.")
        expected = {
            "schema",
            "status",
            "common_force_law",
            "branches",
            "training_roles",
            "claim_boundary",
            "model_fingerprint",
        }
        _require(set(raw) == expected, "Unified VIV gain model keys differ from schema.")
        _require(raw["schema"] == SCHEMA, "Unsupported unified VIV gain schema.")
        _require(raw["status"] == STATUS, "Unified VIV gain model is not frozen.")
        branches = raw["branches"]
        _require(
            isinstance(branches, Mapping)
            and set(branches) == {AIRFOIL_BRANCH, TURBINE_BRANCH},
            "Unified VIV gain branch set changed.",
        )
        airfoil = branches[AIRFOIL_BRANCH]
        turbine = branches[TURBINE_BRANCH]
        _require(
            isinstance(airfoil, Mapping)
            and set(airfoil)
            == {"domain", "gain_representation", "gain_state", "source"},
            "Unified airfoil branch keys differ from schema.",
        )
        _require(
            isinstance(turbine, Mapping)
            and set(turbine)
            == {"domain", "gain_representation", "parameters", "sources"},
            "Unified turbine branch keys differ from schema.",
        )
        _require(
            airfoil["gain_representation"] == "spectral_basis_stretch"
            and turbine["gain_representation"]
            == "zero_amplitude_feedback_over_opposing_damping",
            "Unified gain representations changed.",
        )
        _require(
            isinstance(airfoil["domain"], Mapping)
            and set(airfoil["domain"])
            == {"profile", "kinematic_family", "reynolds_policy"},
            "Unified airfoil domain keys differ from schema.",
        )
        _require(
            isinstance(airfoil["gain_state"], Mapping)
            and set(airfoil["gain_state"])
            == {
                "zero_amplitude_gain",
                "limiting_amplitude_ratio",
                "capture_window",
                "frequency_ratio_low",
                "frequency_ratio_peak",
                "frequency_ratio_high",
                "static_spectrum_sha256",
                "static_spectrum_reynolds_model",
            },
            "Unified airfoil gain-state keys differ from schema.",
        )
        _require(
            isinstance(turbine["domain"], Mapping)
            and set(turbine["domain"])
            == {
                "profile_family",
                "geometry_families",
                "inclination_anchors_deg",
                "interpolation_policy",
                "fu_configurations",
                "fu_wind_speed_anchors_mps",
            },
            "Unified turbine domain keys differ from schema.",
        )
        _require(
            isinstance(turbine["parameters"], Mapping)
            and set(turbine["parameters"])
            == {
                "composition_order",
                "geometry_margin_weights",
                "limiting_amplitude_m",
                "normalization",
                "response_frequency_ratio",
            },
            "Unified turbine parameter keys differ from schema.",
        )
        sources = {
            AIRFOIL_BRANCH: {"airfoil_calibration": airfoil["source"]},
            TURBINE_BRANCH: turbine["sources"],
        }
        _require(
            isinstance(sources[TURBINE_BRANCH], Mapping)
            and set(sources[TURBINE_BRANCH])
            == {"geometry_composition", "signed_work_shape"},
            "Unified turbine source roles differ from schema.",
        )
        for branch_name, branch_sources in sources.items():
            for source in branch_sources.values():
                _require(
                    isinstance(source, Mapping)
                    and set(source)
                    == {"path", "path_basis", "semantic_fingerprint", "sha256"},
                    f"Unified {branch_name} source keys differ from schema.",
                )
        fingerprint_payload = dict(raw)
        stored_fingerprint = fingerprint_payload.pop("model_fingerprint")
        expected_fingerprint = semantic_sha256(fingerprint_payload)
        _require(
            stored_fingerprint == expected_fingerprint,
            "Unified VIV gain model fingerprint changed.",
        )
        repo = (
            Path(__file__).resolve().parents[1]
            if repository_root is None
            else Path(repository_root).expanduser().resolve()
        )
        return cls(
            raw=dict(raw),
            path=resolved,
            repository_root=repo,
            fingerprint=str(stored_fingerprint),
        )

    def _branch(self, name: str) -> Mapping[str, Any]:
        branch = self.raw["branches"][name]
        if not isinstance(branch, Mapping):
            raise UnifiedVIVGainError(f"Unified branch {name!r} must be an object.")
        return branch

    def _source_path(self, source: Mapping[str, Any]) -> Path:
        path = source.get("path")
        basis = source.get("path_basis")
        _require(isinstance(path, str) and path and not Path(path).is_absolute(), "Source path must be relative.")
        if basis == "artifact_directory":
            root = self.path.parent
        elif basis == "repository_root":
            root = self.repository_root
        else:
            raise UnifiedVIVGainError("Unsupported unified source path basis.")
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise UnifiedVIVGainError("Unified source path escapes its declared root.") from exc
        _require(resolved.is_file(), f"Unified source is unavailable: {resolved}")
        _require(
            _sha256_file(resolved) == str(source.get("sha256", "")).lower(),
            f"Unified source SHA-256 changed: {resolved}",
        )
        return resolved

    def evaluate_airfoil(
        self,
        *,
        profile: str,
        kinematic_family: str,
    ) -> UnifiedVIVGainState:
        """Return the exact NACA0018 anchor; no H-VAWT retuning is allowed."""

        branch = self._branch(AIRFOIL_BRANCH)
        domain = branch.get("domain")
        declared = branch.get("gain_state")
        source = branch.get("source")
        _require(isinstance(domain, Mapping) and isinstance(declared, Mapping) and isinstance(source, Mapping), "Airfoil branch is malformed.")
        _require(profile == domain.get("profile"), "Profile is outside the frozen airfoil branch.")
        _require(
            kinematic_family == domain.get("kinematic_family"),
            "Kinematic family is outside the frozen airfoil branch.",
        )
        calibration = EngineeringLockInCalibration.from_json(self._source_path(source))
        transfer = calibration.transferable_parameters
        _require(
            calibration.fingerprint == source.get("semantic_fingerprint"),
            "Airfoil calibration semantic fingerprint changed.",
        )
        exact = {
            "zero_amplitude_gain": transfer.feedback_gain,
            "limiting_amplitude_ratio": transfer.limiting_amplitude_ratio,
            "capture_window": transfer.capture_window,
            "frequency_ratio_low": transfer.frequency_ratio_low,
            "frequency_ratio_peak": transfer.frequency_ratio_peak,
            "frequency_ratio_high": transfer.frequency_ratio_high,
            "static_spectrum_sha256": transfer.static_spectrum_sha256,
            "static_spectrum_reynolds_model": transfer.static_spectrum_reynolds_model,
        }
        for key, value in exact.items():
            _require(declared.get(key) == value, f"Airfoil unified {key} differs from its frozen source.")
        return UnifiedVIVGainState(
            branch=AIRFOIL_BRANCH,
            gain_basis=SPECTRAL_BASIS,
            zero_amplitude_gain=transfer.feedback_gain,
            limiting_amplitude_ratio=transfer.limiting_amplitude_ratio,
            reference_length_m=None,
            work_margin=None,
            active=None,
            profile_family=str(profile),
            kinematic_family=str(kinematic_family),
            capture_window=transfer.capture_window,
            frequency_ratio_low=transfer.frequency_ratio_low,
            frequency_ratio_peak=transfer.frequency_ratio_peak,
            frequency_ratio_high=transfer.frequency_ratio_high,
            response_frequency_ratio=None,
            source_fingerprint=calibration.fingerprint,
            unified_model_fingerprint=self.fingerprint,
            static_spectrum_sha256=transfer.static_spectrum_sha256,
            static_spectrum_reynolds_model=transfer.static_spectrum_reynolds_model,
        )

    def _load_semantic_model(self, source: Mapping[str, Any], label: str) -> Mapping[str, Any]:
        model = json.loads(self._source_path(source).read_text(encoding="utf-8"))
        _require(isinstance(model, Mapping), f"Frozen {label} model must be an object.")
        _require(
            model.get("model_fingerprint") == source.get("semantic_fingerprint"),
            f"Frozen {label} model fingerprint changed.",
        )
        unsigned = dict(model)
        stored = unsigned.pop("model_fingerprint", None)
        _require(stored == semantic_sha256(unsigned), f"Frozen {label} semantic content changed.")
        return model

    def _turbine_model(self) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        branch = self._branch(TURBINE_BRANCH)
        sources = branch.get("sources")
        _require(isinstance(sources, Mapping), "Turbine sources are malformed.")
        model = self._load_semantic_model(
            sources["geometry_composition"], "turbine geometry"
        )
        return branch, model

    def _fu_anchor(
        self,
        *,
        configuration: str,
        wind_speed_mps: float,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        branch, geometry_model = self._turbine_model()
        domain = branch["domain"]
        _require(
            configuration in domain["fu_configurations"],
            "Configuration is outside the frozen Fu training domain.",
        )
        wind = _finite(wind_speed_mps, "wind_speed_mps")
        _require(
            wind in domain["fu_wind_speed_anchors_mps"],
            "Wind speed is outside the frozen Fu anchor grid.",
        )
        rows = geometry_model.get("training", {}).get(
            "fu2026_speed_orientation_rows_after_preserved_failure"
        )
        _require(isinstance(rows, list), "Frozen Fu training anchors are unavailable.")
        matches = [
            row
            for row in rows
            if row.get("config") == configuration
            and row.get("wind_speed_mps") == wind
        ]
        _require(len(matches) == 1, "Frozen Fu training anchor is not unique.")
        return branch, geometry_model, matches[0]

    def evaluate_turbine_training_condition(
        self,
        *,
        profile_family: str,
        configuration: str,
        wind_speed_mps: float,
    ) -> UnifiedVIVGainState:
        """Return one original Fu training-row work-margin state."""

        branch, model, row = self._fu_anchor(
            configuration=configuration,
            wind_speed_mps=wind_speed_mps,
        )
        _require(
            profile_family == branch["domain"]["profile_family"],
            "Profile family is outside the turbine branch.",
        )
        margin = _finite(row["normalized_positive_work_margin"], "work_margin")
        ratio = _finite(
            row["zero_amplitude_feedback_over_structural_damping"],
            "feedback_ratio",
        )
        limiting = _finite(
            row["configuration_saturation_amplitude_m"],
            "limiting_amplitude_m",
        )
        _require(ratio == 1.0 / (1.0 - margin), "Frozen Fu gain conversion changed.")
        return UnifiedVIVGainState(
            branch=TURBINE_BRANCH,
            gain_basis=OPPOSING_DAMPING,
            zero_amplitude_gain=ratio,
            limiting_amplitude_ratio=1.0,
            reference_length_m=limiting,
            work_margin=margin,
            active=bool(margin > 0.0),
            profile_family=profile_family,
            kinematic_family=f"fu_speed_orientation_{configuration}",
            capture_window=None,
            frequency_ratio_low=None,
            frequency_ratio_peak=None,
            frequency_ratio_high=None,
            response_frequency_ratio=None,
            source_fingerprint=str(model["model_fingerprint"]),
            unified_model_fingerprint=self.fingerprint,
        )

    def evaluate_turbine_signed_work_condition(
        self,
        *,
        profile_family: str,
        configuration: str,
        wind_speed_mps: float,
    ) -> UnifiedVIVGainState:
        """Return the frozen active/inactive signed-work transfer state."""

        branch, _geometry_model, original = self._fu_anchor(
            configuration=configuration,
            wind_speed_mps=wind_speed_mps,
        )
        _require(
            profile_family == branch["domain"]["profile_family"],
            "Profile family is outside the turbine branch.",
        )
        shape_model = self._load_semantic_model(
            branch["sources"]["signed_work_shape"], "signed-work shape"
        )
        matches = [
            row
            for row in shape_model.get("cases", [])
            if row.get("configuration") == configuration
            and row.get("wind_speed_mps") == float(wind_speed_mps)
        ]
        _require(len(matches) == 1, "Frozen signed-work anchor is not unique.")
        row = matches[0]
        _require(
            row["original_feedback_over_structural_damping"]
            == original["zero_amplitude_feedback_over_structural_damping"],
            "Signed-work source differs from the original Fu anchor.",
        )
        active = row["active_branch"]
        _require(isinstance(active, bool), "Signed-work active flag must be Boolean.")
        ratio = _finite(
            row["transferred_feedback_over_opposing_damping"],
            "transferred_feedback_ratio",
        )
        limiting = _finite(
            row["transferred_limiting_amplitude_m"],
            "transferred_limiting_amplitude_m",
        )
        _require(
            (active and ratio > 1.0) or (not active and ratio == 0.0),
            "Signed-work active/inactive feedback rule changed.",
        )
        margin = 1.0 - 1.0 / ratio if active else None
        return UnifiedVIVGainState(
            branch=TURBINE_BRANCH,
            gain_basis=OPPOSING_DAMPING,
            zero_amplitude_gain=ratio,
            limiting_amplitude_ratio=1.0,
            reference_length_m=limiting,
            work_margin=margin,
            active=active,
            profile_family=profile_family,
            kinematic_family=f"fu_signed_work_{configuration}",
            capture_window=None,
            frequency_ratio_low=None,
            frequency_ratio_peak=None,
            frequency_ratio_high=None,
            response_frequency_ratio=None,
            source_fingerprint=str(shape_model["model_fingerprint"]),
            unified_model_fingerprint=self.fingerprint,
        )

    def evaluate_turbine_geometry(
        self,
        *,
        profile_family: str,
        geometry_family: str,
        inclination_deg: float,
    ) -> UnifiedVIVGainState:
        """Evaluate one exact Horcas anchor or the frozen COMB composition."""

        branch, model = self._turbine_model()
        domain = branch.get("domain")
        parameters = branch.get("parameters")
        _require(isinstance(domain, Mapping) and isinstance(parameters, Mapping), "Turbine branch is malformed.")
        _require(profile_family == domain.get("profile_family"), "Profile family is outside the turbine branch.")
        _require(geometry_family in domain.get("geometry_families", []), "Geometry family is outside the turbine branch.")
        inclination = _finite(inclination_deg, "inclination_deg")
        _require(inclination in domain.get("inclination_anchors_deg", []), "Inclination is outside the frozen anchor grid.")

        source_parameters = model.get("parameters", {})
        saturation = float(source_parameters.get("grinderslev_signed_power_saturation_tip_amplitude_m"))
        frequency_ratio = float(source_parameters.get("resolved_viv_frequency_ratio"))
        _require(saturation == parameters.get("limiting_amplitude_m"), "Turbine saturation changed.")
        _require(frequency_ratio == parameters.get("response_frequency_ratio"), "Turbine frequency ratio changed.")
        source_weights = source_parameters.get("geometry_margin_weights")
        _require(source_weights == parameters.get("geometry_margin_weights"), "Turbine composition weights changed.")

        training = model.get("training", {}).get("horcas2020_primitive_geometry_rows")
        _require(isinstance(training, Mapping), "Frozen turbine anchors are unavailable.")

        def primitive_margin(geometry: str) -> float:
            rows = training.get(geometry)
            _require(isinstance(rows, list), f"Frozen {geometry} anchors are unavailable.")
            matches = [row for row in rows if row.get("inclination_deg") == inclination]
            _require(len(matches) == 1, f"Frozen {geometry} inclination anchor is not unique.")
            margin = _finite(matches[0].get("normalized_positive_work_margin"), "work_margin")
            _require(0.0 <= margin < 1.0, "Frozen work margin is outside [0,1).")
            return margin

        if geometry_family == "COMB":
            # Preserve the frozen order and operation ordering exactly.
            component_order = tuple(parameters.get("composition_order", []))
            _require(component_order == ("EXT", "SWEEP", "DIHE"), "COMB composition order changed.")
            margins = [primitive_margin(geometry) for geometry in component_order]
            margin = sum(margins) / len(margins)
        else:
            margin = primitive_margin(geometry_family)
        feedback_ratio = 1.0 / (1.0 - margin)
        return UnifiedVIVGainState(
            branch=TURBINE_BRANCH,
            gain_basis=OPPOSING_DAMPING,
            zero_amplitude_gain=feedback_ratio,
            limiting_amplitude_ratio=1.0,
            reference_length_m=saturation,
            work_margin=margin,
            active=bool(margin > 0.0),
            profile_family=str(profile_family),
            kinematic_family="inclined_fixed_tip_geometry",
            capture_window=None,
            frequency_ratio_low=None,
            frequency_ratio_peak=None,
            frequency_ratio_high=None,
            response_frequency_ratio=frequency_ratio,
            source_fingerprint=str(model["model_fingerprint"]),
            unified_model_fingerprint=self.fingerprint,
        )


__all__ = [
    "AIRFOIL_BRANCH",
    "OPPOSING_DAMPING",
    "SCHEMA",
    "SPECTRAL_BASIS",
    "STATUS",
    "TURBINE_BRANCH",
    "UnifiedVIVGainError",
    "UnifiedVIVGainModel",
    "UnifiedVIVGainState",
]
