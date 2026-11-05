"""Validated manifest support for sampled parked DLC 6.1/6.4 campaigns."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

from .turbsim import estimate_full_field_bytes


SCHEMA_VERSION = 3
REQUIRED_SCENARIOS = ("baseline", "vorlap")
SUPPORTED_DLCS = ("6.1", "6.4")
SOURCE_FREQUENCY_TRUNCATION_POLICIES = ("none", "nyquist")
SOURCE_BOUNDS_POLICIES = ("warn", "raise", "clip")
WAKE_PROFILES = ("reference", "parked-reduced-v1", "parked-reduced-v2")
_ROW_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PLACEHOLDER_RE = re.compile(r"(?:\bTBD\b|REPLACE|PLACEHOLDER|UNVERIFIED)", re.IGNORECASE)


class DLCManifestError(ValueError):
    """Raised when a parked-DLC manifest is incomplete or inconsistent."""


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def speed_label(speed: float) -> str:
    return (f"U{float(speed):05.2f}ms").replace(".", "p").replace("-", "m")


def dlc_label(dlc: str) -> str:
    return f"dlc{str(dlc).replace('.', 'p')}"


def row_id_for(*, dlc: str, wind_speed_mps: float, seed: int) -> str:
    return f"{dlc_label(dlc)}_{speed_label(wind_speed_mps)}_s{int(seed)}"


def _finite_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DLCManifestError(f"{name} must be numeric.") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise DLCManifestError(f"{name} must be finite and positive.")
    return number


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise DLCManifestError(f"{name} must be an integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise DLCManifestError(f"{name} must be an integer.") from exc
    if str(value).strip() not in {str(number), f"+{number}"} and not isinstance(value, int):
        raise DLCManifestError(f"{name} must be an integer.")
    return number


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DLCManifestError(f"{name} must be a JSON boolean.")
    return value


def _result_filter_channels(path: Path) -> list[str]:
    try:
        channels = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise DLCManifestError(f"Could not read QBlade result filter {path}: {exc}") from exc
    if not channels:
        raise DLCManifestError(f"QBlade result filter has no variable names: {path}")
    if len(channels) != len(set(channels)):
        raise DLCManifestError(f"QBlade result filter contains duplicate variable names: {path}")
    return channels


def _validate_result_filter(
    execution: dict[str, Any],
    *,
    require_file: bool,
) -> None:
    path = Path(str(execution.get("result_filter_path", "") or "")).expanduser()
    if not path.is_absolute():
        raise DLCManifestError("execution_settings.result_filter_path must be absolute.")
    _validate_sha256(
        execution.get("result_filter_sha256"),
        "execution_settings.result_filter_sha256",
    )
    channel_count = _integer(
        execution.get("result_filter_channel_count"),
        "execution_settings.result_filter_channel_count",
    )
    if channel_count <= 0:
        raise DLCManifestError("execution_settings.result_filter_channel_count must be positive.")
    if not require_file:
        return
    if not path.is_file():
        raise DLCManifestError(
            f"Approved manifest QBlade result filter does not exist: {path}"
        )
    if sha256_file(path) != execution["result_filter_sha256"]:
        raise DLCManifestError("Approved manifest QBlade result-filter SHA-256 does not match.")
    observed_count = len(_result_filter_channels(path))
    if observed_count != channel_count:
        raise DLCManifestError(
            "Approved manifest QBlade result-filter channel count does not match: "
            f"expected {channel_count}, observed {observed_count}."
        )


def _weight_for_speed(mapping: Any, speed: float) -> float | None:
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        raise DLCManifestError("fatigue_weights_by_speed must be an object.")
    candidates = (str(speed), f"{speed:g}", f"{speed:.1f}", f"{speed:.2f}")
    for key in candidates:
        if key in mapping:
            value = mapping[key]
            if value is None:
                return None
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise DLCManifestError(f"Fatigue weight for {speed:g} m/s must be finite and nonnegative.")
            return number
    return None


def build_manifest(spec: dict[str, Any]) -> dict[str, Any]:
    """Expand a compact group specification into one immutable row per seed."""
    if not isinstance(spec, dict):
        raise DLCManifestError("DLC matrix specification must be a JSON object.")
    groups = spec.get("groups")
    if not isinstance(groups, list) or not groups:
        raise DLCManifestError("DLC matrix specification must contain a nonempty groups list.")

    duration = _finite_positive(spec.get("duration_s", 660.0), "duration_s")
    store_from = float(spec.get("store_from_s", 60.0))
    timestep = _finite_positive(spec.get("timestep_s", 0.0075), "timestep_s")
    if not math.isfinite(store_from) or store_from < 0.0 or store_from >= duration:
        raise DLCManifestError("store_from_s must be finite, nonnegative, and below duration_s.")
    wind_root = Path(str(spec.get("wind_root", "") or "")).expanduser()
    run_root = Path(str(spec.get("run_root", "") or "")).expanduser()
    if not wind_root.is_absolute() or not run_root.is_absolute():
        raise DLCManifestError("wind_root and run_root must be absolute paths.")

    num_grid_y = _integer(spec.get("num_grid_y", 26), "num_grid_y")
    num_grid_z = _integer(spec.get("num_grid_z", 38), "num_grid_z")
    if num_grid_y < 2 or num_grid_z < 2:
        raise DLCManifestError("TurbSim grid dimensions must both be at least 2.")
    analysis_time = _finite_positive(spec.get("wind_analysis_time_s", duration), "wind_analysis_time_s")
    wind_timestep = _finite_positive(spec.get("wind_timestep_s", timestep), "wind_timestep_s")
    estimated_bytes = estimate_full_field_bytes(
        analysis_time_s=analysis_time,
        time_step_s=wind_timestep,
        num_grid_y=num_grid_y,
        num_grid_z=num_grid_z,
    )

    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise DLCManifestError(f"groups[{group_index}] must be an object.")
        dlc = str(group.get("dlc", "")).strip()
        if dlc not in SUPPORTED_DLCS:
            raise DLCManifestError(f"groups[{group_index}].dlc must be one of {SUPPORTED_DLCS}.")
        speeds = group.get("wind_speeds_mps")
        seeds = group.get("seeds")
        if not isinstance(speeds, list) or not speeds:
            raise DLCManifestError(f"groups[{group_index}].wind_speeds_mps must be a nonempty list.")
        if not isinstance(seeds, list) or not seeds:
            raise DLCManifestError(f"groups[{group_index}].seeds must be a nonempty list.")
        wind_type = str(group.get("turbsim_iec_wind_type", "") or "").strip()
        if not wind_type:
            raise DLCManifestError(f"groups[{group_index}].turbsim_iec_wind_type is required.")
        include_in_fatigue = bool(group.get("include_in_fatigue", dlc == "6.4"))
        include_in_ultimate = bool(group.get("include_in_ultimate", dlc == "6.1"))
        for speed_value in speeds:
            speed = _finite_positive(speed_value, f"groups[{group_index}] wind speed")
            fatigue_weight = _weight_for_speed(group.get("fatigue_weights_by_speed"), speed)
            for seed_value in seeds:
                seed = _integer(seed_value, f"groups[{group_index}] seed")
                row_id = row_id_for(dlc=dlc, wind_speed_mps=speed, seed=seed)
                wind_file = wind_root / dlc_label(dlc) / speed_label(speed) / f"{row_id}.bts"
                rows.append(
                    {
                        "row_id": row_id,
                        "dlc": dlc,
                        "wind_speed_mps": speed,
                        "seed": seed,
                        "parked_azimuth_deg": float(spec.get("parked_azimuth_deg", 0.0)),
                        "rpm": float(spec.get("rpm", 1.0)),
                        "turbsim_iec_wind_type": wind_type,
                        "turbsim_turb_model": str(group.get("turbsim_turb_model", "IECKAI")),
                        "turbsim_iec_standard": str(group.get("turbsim_iec_standard", "1")),
                        "turbsim_iec_turbc": str(group.get("turbsim_iec_turbc", "A")),
                        "fatigue_weight": fatigue_weight,
                        "include_in_fatigue": include_in_fatigue,
                        "include_in_ultimate": include_in_ultimate,
                        "wind_file": str(wind_file),
                        "run_root": str(run_root / row_id),
                        "estimated_wind_bytes": estimated_bytes,
                    }
                )

    execution_input = spec.get("execution_settings", {})
    if not isinstance(execution_input, dict):
        raise DLCManifestError("execution_settings must be an object.")
    source_frequency_truncation = str(
        execution_input.get("source_frequency_truncation", "none")
    ).strip().lower()
    if source_frequency_truncation not in SOURCE_FREQUENCY_TRUNCATION_POLICIES:
        raise DLCManifestError(
            "execution_settings.source_frequency_truncation must be one of "
            f"{SOURCE_FREQUENCY_TRUNCATION_POLICIES}."
        )
    source_bounds_policy = str(execution_input.get("source_bounds_policy", "raise")).strip().lower()
    if source_bounds_policy not in SOURCE_BOUNDS_POLICIES:
        raise DLCManifestError(
            f"execution_settings.source_bounds_policy must be one of {SOURCE_BOUNDS_POLICIES}."
        )
    checkpoint_every = _integer(
        execution_input.get("checkpoint_every", 4000),
        "execution_settings.checkpoint_every",
    )
    if checkpoint_every < 0:
        raise DLCManifestError("execution_settings.checkpoint_every must be nonnegative.")
    case_workers = _integer(execution_input.get("case_workers", 1), "execution_settings.case_workers")
    if case_workers < 1:
        raise DLCManifestError("execution_settings.case_workers must be positive.")
    wake_profile = str(execution_input.get("wake_profile", "reference")).strip().lower()
    if wake_profile not in WAKE_PROFILES:
        raise DLCManifestError(
            f"execution_settings.wake_profile must be one of {WAKE_PROFILES}."
        )
    result_filter_path = str(execution_input.get("result_filter_path", "") or "").strip()
    result_filter_sha256 = str(
        execution_input.get("result_filter_sha256", "") or ""
    ).strip().lower()
    result_filter_channel_count = _integer(
        execution_input.get("result_filter_channel_count", 0),
        "execution_settings.result_filter_channel_count",
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": str(spec.get("status", "draft")).strip().lower(),
        "study_id": str(spec.get("study_id", "parked_dlc_6p1_6p4")).strip(),
        "basis": deepcopy(spec.get("basis", {})),
        "wind_root": str(wind_root),
        "run_root": str(run_root),
        "duration_s": duration,
        "store_from_s": store_from,
        "retained_duration_s": duration - store_from,
        "timestep_s": timestep,
        "scenarios": list(spec.get("scenarios", REQUIRED_SCENARIOS)),
        "wind_settings": {
            "analysis_time_s": analysis_time,
            "time_step_s": wind_timestep,
            "num_grid_y": num_grid_y,
            "num_grid_z": num_grid_z,
            "grid_width_m": _finite_positive(spec.get("grid_width_m", 26.0), "grid_width_m"),
            "grid_height_m": _finite_positive(spec.get("grid_height_m", 26.0), "grid_height_m"),
            "hub_height_m": _finite_positive(spec.get("hub_height_m", 18.0), "hub_height_m"),
            "ref_height_m": _finite_positive(spec.get("ref_height_m", 18.0), "ref_height_m"),
            "horizontal_inflow_angle_deg": float(spec.get("horizontal_inflow_angle_deg", 0.0)),
            "vertical_inflow_angle_deg": float(spec.get("vertical_inflow_angle_deg", 0.0)),
            "shear_exponent": float(spec.get("shear_exponent", 0.23)),
            "air_density_kg_m3": _finite_positive(
                spec.get("air_density_kg_m3", 1.0296), "air_density_kg_m3"
            ),
            "air_kinematic_viscosity_m2_s": _finite_positive(
                spec.get("air_kinematic_viscosity_m2_s", 1.647e-5),
                "air_kinematic_viscosity_m2_s",
            ),
        },
        "requirements": {
            "min_seeds_by_dlc": deepcopy(spec.get("min_seeds_by_dlc", {"6.1": 1, "6.4": 6})),
            "require_fatigue_weights": bool(spec.get("require_fatigue_weights", True)),
        },
        "execution_settings": {
            "source_frequency_truncation": source_frequency_truncation,
            "source_bounds_policy": source_bounds_policy,
            "wake_profile": wake_profile,
            "result_filter_path": result_filter_path,
            "result_filter_sha256": result_filter_sha256,
            "result_filter_channel_count": result_filter_channel_count,
            "checkpoint_every": checkpoint_every,
            "store_replay": _boolean(
                execution_input.get("store_replay", False),
                "execution_settings.store_replay",
            ),
            "store_project": _boolean(
                execution_input.get("store_project", False),
                "execution_settings.store_project",
            ),
            "use_opencl": _boolean(
                execution_input.get("use_opencl", False),
                "execution_settings.use_opencl",
            ),
            "case_workers": case_workers,
            "auto_node_threads": _boolean(
                execution_input.get("auto_node_threads", True),
                "execution_settings.auto_node_threads",
            ),
            "run_final_audit": _boolean(
                execution_input.get("run_final_audit", True),
                "execution_settings.run_final_audit",
            ),
            "require_source_coverage": _boolean(
                execution_input.get("require_source_coverage", True),
                "execution_settings.require_source_coverage",
            ),
            "require_postprocess_artifacts": _boolean(
                execution_input.get("require_postprocess_artifacts", True),
                "execution_settings.require_postprocess_artifacts",
            ),
        },
        "rows": rows,
        "row_count": len(rows),
        "estimated_wind_bytes_total": sum(int(row["estimated_wind_bytes"]) for row in rows),
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    validate_manifest(manifest, require_approved=False)
    return manifest


def _validate_sha256(value: Any, name: str) -> None:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise DLCManifestError(f"{name} must be a 64-character lowercase hexadecimal SHA-256 digest.")


def _load_review_artifact(basis: dict[str, Any], prefix: str) -> dict[str, Any]:
    path_name = f"{prefix}_review_path"
    hash_name = f"{prefix}_review_sha256"
    path = Path(str(basis.get(path_name, "") or "")).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise DLCManifestError(f"Approved manifest basis.{path_name} must be an existing absolute file.")
    _validate_sha256(basis.get(hash_name), f"basis.{hash_name}")
    if sha256_file(path) != basis[hash_name]:
        raise DLCManifestError(f"{prefix.replace('_', ' ').title()} review file hash does not match the approved basis.")
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DLCManifestError(f"Could not read {prefix.replace('_', ' ')} review: {exc}") from exc
    if review.get("passed") is not True:
        raise DLCManifestError(f"{prefix.replace('_', ' ').title()} review is not marked passed.")
    archived_digest = str(review.get("review_sha256", "") or "")
    without_digest = dict(review)
    without_digest.pop("review_sha256", None)
    if archived_digest != sha256_payload(without_digest):
        raise DLCManifestError(f"{prefix.replace('_', ' ').title()} review internal digest is invalid.")
    return review


def validate_manifest(manifest: dict[str, Any], *, require_approved: bool = True) -> dict[str, Any]:
    """Validate row uniqueness, durations, seeds, weights, paths, and digest."""
    if not isinstance(manifest, dict):
        raise DLCManifestError("DLC manifest must be a JSON object.")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DLCManifestError(f"schema_version must equal {SCHEMA_VERSION}.")
    status = str(manifest.get("status", "")).strip().lower()
    if status not in {"draft", "approved"}:
        raise DLCManifestError("status must be 'draft' or 'approved'.")
    if require_approved and status != "approved":
        raise DLCManifestError("Full DLC submission requires an approved manifest; this manifest is still draft.")
    basis = manifest.get("basis")
    if not isinstance(basis, dict):
        raise DLCManifestError("basis must be an object.")
    fatigue_weighting_review: dict[str, Any] | None = None
    grid_domain_review: dict[str, Any] | None = None
    modal_overlap_review: dict[str, Any] | None = None
    if require_approved:
        for name in ("iec_edition", "wind_class", "fatigue_weighting_source", "matrix_approval"):
            value = str(basis.get(name, "") or "").strip()
            if not value or _PLACEHOLDER_RE.search(value):
                raise DLCManifestError(f"Approved manifest basis.{name} must be explicit and placeholder-free.")
        _validate_sha256(basis.get("airfoil_source_sha256"), "basis.airfoil_source_sha256")
        qualification_path = Path(str(basis.get("modal_timestep_qualification_path", "") or "")).expanduser()
        if not qualification_path.is_absolute() or not qualification_path.is_file():
            raise DLCManifestError(
                "Approved manifest basis.modal_timestep_qualification_path must be an existing absolute file."
            )
        _validate_sha256(
            basis.get("modal_timestep_qualification_sha256"),
            "basis.modal_timestep_qualification_sha256",
        )
        if sha256_file(qualification_path) != basis["modal_timestep_qualification_sha256"]:
            raise DLCManifestError("Modal/timestep qualification file hash does not match the approved basis.")
        try:
            qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DLCManifestError(f"Could not read modal/timestep qualification: {exc}") from exc
        if qualification.get("qualified") is not True:
            raise DLCManifestError("Modal/timestep qualification is not marked qualified.")
        qualification_digest = str(qualification.get("qualification_sha256", "") or "")
        qualification_without_digest = dict(qualification)
        qualification_without_digest.pop("qualification_sha256", None)
        if qualification_digest != sha256_payload(qualification_without_digest):
            raise DLCManifestError("Modal/timestep qualification internal digest is invalid.")
        fatigue_weighting_review = _load_review_artifact(basis, "fatigue_weighting")
        grid_domain_review = _load_review_artifact(basis, "grid_domain")
        modal_overlap_review = _load_review_artifact(basis, "modal_overlap")
    if tuple(manifest.get("scenarios", [])) != REQUIRED_SCENARIOS:
        raise DLCManifestError(f"scenarios must be exactly {list(REQUIRED_SCENARIOS)}.")
    execution = manifest.get("execution_settings")
    if not isinstance(execution, dict):
        raise DLCManifestError("execution_settings must be an object.")
    source_frequency_truncation = str(execution.get("source_frequency_truncation", "")).strip().lower()
    if source_frequency_truncation not in SOURCE_FREQUENCY_TRUNCATION_POLICIES:
        raise DLCManifestError(
            "execution_settings.source_frequency_truncation must be one of "
            f"{SOURCE_FREQUENCY_TRUNCATION_POLICIES}."
        )
    source_bounds_policy = str(execution.get("source_bounds_policy", "")).strip().lower()
    if source_bounds_policy not in SOURCE_BOUNDS_POLICIES:
        raise DLCManifestError(
            f"execution_settings.source_bounds_policy must be one of {SOURCE_BOUNDS_POLICIES}."
        )
    checkpoint_every = _integer(execution.get("checkpoint_every"), "execution_settings.checkpoint_every")
    if checkpoint_every < 0:
        raise DLCManifestError("execution_settings.checkpoint_every must be nonnegative.")
    case_workers = _integer(execution.get("case_workers"), "execution_settings.case_workers")
    if case_workers < 1:
        raise DLCManifestError("execution_settings.case_workers must be positive.")
    wake_profile = str(execution.get("wake_profile", "")).strip().lower()
    if wake_profile not in WAKE_PROFILES:
        raise DLCManifestError(
            f"execution_settings.wake_profile must be one of {WAKE_PROFILES}."
        )
    _validate_result_filter(execution, require_file=require_approved)
    for name in (
        "store_replay",
        "store_project",
        "use_opencl",
        "auto_node_threads",
        "run_final_audit",
        "require_source_coverage",
        "require_postprocess_artifacts",
    ):
        _boolean(execution.get(name), f"execution_settings.{name}")
    if require_approved:
        if source_bounds_policy != "raise":
            raise DLCManifestError("Approved production manifests require source_bounds_policy='raise'.")
        if wake_profile != "parked-reduced-v2":
            raise DLCManifestError(
                "Approved production manifests require wake_profile='parked-reduced-v2'."
            )
        if case_workers != 1:
            raise DLCManifestError("Approved production manifests require case_workers=1.")
        if checkpoint_every <= 0:
            raise DLCManifestError("Approved production manifests require restart checkpointing.")
        if execution["store_replay"] is not False:
            raise DLCManifestError("Approved production manifests require store_replay=false.")
        if execution["store_project"] is not False:
            raise DLCManifestError("Approved production manifests require store_project=false.")
        if execution["use_opencl"] is not False:
            raise DLCManifestError("Approved production manifests require CPU/OpenMP with use_opencl=false.")
        if execution["auto_node_threads"] is not True:
            raise DLCManifestError("Approved production manifests require auto_node_threads=true.")
        for name in ("run_final_audit", "require_source_coverage", "require_postprocess_artifacts"):
            if execution[name] is not True:
                raise DLCManifestError(f"Approved production manifests require {name}=true.")
        qualification_policy = str(qualification.get("source_frequency_truncation", "none")).strip().lower()
        if qualification_policy != source_frequency_truncation:
            raise DLCManifestError(
                "Manifest source-frequency truncation does not match the modal/timestep qualification."
            )
    for root_name in ("wind_root", "run_root"):
        root_path = Path(str(manifest.get(root_name, "") or "")).expanduser()
        if not root_path.is_absolute():
            raise DLCManifestError(f"{root_name} must be absolute.")
    duration = _finite_positive(manifest.get("duration_s"), "duration_s")
    store_from = float(manifest.get("store_from_s", math.nan))
    retained = float(manifest.get("retained_duration_s", math.nan))
    if not math.isfinite(store_from) or store_from < 0.0 or store_from >= duration:
        raise DLCManifestError("store_from_s must be finite, nonnegative, and below duration_s.")
    if not math.isclose(retained, duration - store_from, rel_tol=0.0, abs_tol=1.0e-9):
        raise DLCManifestError("retained_duration_s does not equal duration_s - store_from_s.")
    if retained < 600.0:
        raise DLCManifestError("A production DLC manifest must retain at least 600 seconds.")
    _finite_positive(manifest.get("timestep_s"), "timestep_s")
    if require_approved and not math.isclose(
        float(qualification.get("timestep_s", math.nan)),
        float(manifest["timestep_s"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise DLCManifestError("Manifest timestep does not match the modal/timestep qualification.")

    wind_settings = manifest.get("wind_settings")
    if not isinstance(wind_settings, dict):
        raise DLCManifestError("wind_settings must be an object.")
    _finite_positive(wind_settings.get("analysis_time_s"), "wind_settings.analysis_time_s")
    _finite_positive(wind_settings.get("time_step_s"), "wind_settings.time_step_s")
    for name in ("num_grid_y", "num_grid_z"):
        if _integer(wind_settings.get(name), f"wind_settings.{name}") < 2:
            raise DLCManifestError(f"wind_settings.{name} must be at least two.")
    for name in ("grid_width_m", "grid_height_m", "hub_height_m", "ref_height_m"):
        _finite_positive(wind_settings.get(name), f"wind_settings.{name}")
    for name in ("horizontal_inflow_angle_deg", "vertical_inflow_angle_deg", "shear_exponent"):
        try:
            value = float(wind_settings.get(name))
        except (TypeError, ValueError) as exc:
            raise DLCManifestError(f"wind_settings.{name} must be numeric.") from exc
        if not math.isfinite(value):
            raise DLCManifestError(f"wind_settings.{name} must be finite.")
    if float(wind_settings["shear_exponent"]) < 0.0:
        raise DLCManifestError("wind_settings.shear_exponent must be nonnegative.")
    for name in ("air_density_kg_m3", "air_kinematic_viscosity_m2_s"):
        _finite_positive(wind_settings.get(name), f"wind_settings.{name}")
    if require_approved:
        assert grid_domain_review is not None
        reviewed_grid = grid_domain_review.get("grid")
        if not isinstance(reviewed_grid, dict):
            raise DLCManifestError("Grid/domain review has no grid object.")
        for name in ("num_grid_y", "num_grid_z"):
            if _integer(reviewed_grid.get(name), f"grid/domain review grid.{name}") != int(
                wind_settings[name]
            ):
                raise DLCManifestError(f"Manifest wind setting {name} does not match the grid/domain review.")
        for name in ("grid_width_m", "grid_height_m", "hub_height_m", "ref_height_m"):
            if not math.isclose(
                float(reviewed_grid.get(name, math.nan)),
                float(wind_settings[name]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise DLCManifestError(f"Manifest wind setting {name} does not match the grid/domain review.")
        reviewed_conventions = grid_domain_review.get("qblade_conventions")
        if not isinstance(reviewed_conventions, dict):
            raise DLCManifestError("Grid/domain review has no qblade_conventions object.")
        for manifest_name, review_name in (
            ("horizontal_inflow_angle_deg", "horizontal_inflow_angle_deg"),
            ("vertical_inflow_angle_deg", "vertical_inflow_angle_deg"),
        ):
            if not math.isclose(
                float(reviewed_conventions.get(review_name, math.nan)),
                float(wind_settings[manifest_name]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise DLCManifestError(
                    f"Manifest wind setting {manifest_name} does not match the grid/domain review."
                )

    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise DLCManifestError("rows must be a nonempty list.")
    if manifest.get("row_count") != len(rows):
        raise DLCManifestError("row_count does not match the number of rows.")
    ids: set[str] = set()
    keys: set[tuple[str, float, int]] = set()
    seeds_by_condition: dict[tuple[str, float], set[int]] = defaultdict(set)
    fatigue_weights: dict[tuple[str, float], float] = {}
    estimated_total = 0
    dlcs: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DLCManifestError(f"rows[{index}] must be an object.")
        row_id = str(row.get("row_id", "")).strip()
        if not _ROW_ID_RE.fullmatch(row_id):
            raise DLCManifestError(f"rows[{index}].row_id is empty or unsafe.")
        if row_id in ids:
            raise DLCManifestError(f"Duplicate row_id: {row_id}")
        ids.add(row_id)
        dlc = str(row.get("dlc", "")).strip()
        if dlc not in SUPPORTED_DLCS:
            raise DLCManifestError(f"rows[{index}].dlc must be one of {SUPPORTED_DLCS}.")
        dlcs.add(dlc)
        speed = _finite_positive(row.get("wind_speed_mps"), f"rows[{index}].wind_speed_mps")
        seed = _integer(row.get("seed"), f"rows[{index}].seed")
        key = (dlc, speed, seed)
        if key in keys:
            raise DLCManifestError(f"Duplicate DLC/wind-speed/seed row: {key}")
        keys.add(key)
        seeds_by_condition[(dlc, speed)].add(seed)
        for path_name in ("wind_file", "run_root"):
            path = Path(str(row.get(path_name, "") or "")).expanduser()
            if not path.is_absolute():
                raise DLCManifestError(f"rows[{index}].{path_name} must be absolute.")
        if Path(str(row["wind_file"])).suffix.lower() != ".bts":
            raise DLCManifestError(f"rows[{index}].wind_file must end in .bts.")
        if not str(row.get("turbsim_iec_wind_type", "") or "").strip():
            raise DLCManifestError(f"rows[{index}].turbsim_iec_wind_type is required.")
        if require_approved and _PLACEHOLDER_RE.search(str(row["turbsim_iec_wind_type"])):
            raise DLCManifestError(
                f"rows[{index}].turbsim_iec_wind_type contains an unresolved placeholder."
            )
        estimated = _integer(row.get("estimated_wind_bytes"), f"rows[{index}].estimated_wind_bytes")
        if estimated <= 0:
            raise DLCManifestError(f"rows[{index}].estimated_wind_bytes must be positive.")
        estimated_total += estimated
        if bool(row.get("include_in_fatigue", False)):
            weight = row.get("fatigue_weight")
            if weight is not None:
                weight_value = float(weight)
                if not math.isfinite(weight_value) or weight_value < 0.0:
                    raise DLCManifestError(f"rows[{index}].fatigue_weight must be finite and nonnegative.")
                condition = (dlc, speed)
                previous = fatigue_weights.setdefault(condition, weight_value)
                if not math.isclose(previous, weight_value, rel_tol=0.0, abs_tol=1.0e-15):
                    raise DLCManifestError(f"Fatigue weight differs between seeds for DLC {dlc}, U={speed:g}.")
            elif require_approved and bool(manifest.get("requirements", {}).get("require_fatigue_weights", True)):
                raise DLCManifestError(f"Approved fatigue row {row_id} is missing fatigue_weight.")

    if set(SUPPORTED_DLCS) - dlcs:
        raise DLCManifestError("The sampled production manifest must include both DLC 6.1 and DLC 6.4.")
    if int(manifest.get("estimated_wind_bytes_total", -1)) != estimated_total:
        raise DLCManifestError("estimated_wind_bytes_total does not match the row sum.")
    requirements = manifest.get("requirements", {})
    minimums = requirements.get("min_seeds_by_dlc", {}) if isinstance(requirements, dict) else {}
    if require_approved:
        for (dlc, speed), seeds in seeds_by_condition.items():
            minimum = _integer(minimums.get(dlc, 1), f"minimum seed count for DLC {dlc}")
            if len(seeds) < minimum:
                raise DLCManifestError(
                    f"DLC {dlc}, U={speed:g} has {len(seeds)} unique seed(s); approved manifest requires {minimum}."
                )
        assert fatigue_weighting_review is not None
        reviewed_weights = fatigue_weighting_review.get("fatigue_weights_by_speed")
        if not isinstance(reviewed_weights, dict):
            raise DLCManifestError("Fatigue-weighting review has no fatigue_weights_by_speed object.")
        reviewed_conditions: dict[tuple[str, float], float] = {}
        for speed_text, weight_value in reviewed_weights.items():
            speed = _finite_positive(speed_text, "fatigue-weighting review speed")
            weight = float(weight_value)
            if not math.isfinite(weight) or weight < 0.0:
                raise DLCManifestError("Fatigue-weighting review weights must be finite and nonnegative.")
            reviewed_conditions[("6.4", speed)] = weight
        if set(reviewed_conditions) != set(fatigue_weights):
            raise DLCManifestError(
                "Approved DLC 6.4 fatigue conditions do not match the fatigue-weighting review."
            )
        for condition, reviewed_weight in reviewed_conditions.items():
            if not math.isclose(
                fatigue_weights[condition], reviewed_weight, rel_tol=0.0, abs_tol=1.0e-15
            ):
                raise DLCManifestError(
                    "Approved DLC 6.4 fatigue weights do not match the fatigue-weighting review."
                )
        assert modal_overlap_review is not None
        reviewed_matrix = modal_overlap_review.get("matrix_spec")
        if not isinstance(reviewed_matrix, dict):
            raise DLCManifestError("Modal-overlap review has no matrix_spec object.")
        reviewed_speeds_raw = reviewed_matrix.get("current_dlc64_wind_speeds_mps")
        if not isinstance(reviewed_speeds_raw, list) or not reviewed_speeds_raw:
            raise DLCManifestError("Modal-overlap review has no current DLC 6.4 speed list.")
        reviewed_speeds = sorted(
            _finite_positive(value, "modal-overlap review speed") for value in reviewed_speeds_raw
        )
        manifest_dlc64_speeds = sorted(
            speed for dlc, speed in seeds_by_condition if dlc == "6.4"
        )
        if reviewed_speeds != manifest_dlc64_speeds:
            raise DLCManifestError(
                "Approved DLC 6.4 wind speeds do not match the modal-overlap review."
            )
        coverage = modal_overlap_review.get("current_matrix_coverage")
        if not isinstance(coverage, dict) or coverage.get("passed") is not True:
            raise DLCManifestError("Modal-overlap review does not pass current-matrix peak coverage.")
    archived_digest = str(manifest.get("manifest_sha256", "") or "").strip().lower()
    _validate_sha256(archived_digest, "manifest_sha256")
    without_digest = dict(manifest)
    without_digest.pop("manifest_sha256", None)
    expected_digest = sha256_payload(without_digest)
    if archived_digest != expected_digest:
        raise DLCManifestError("manifest_sha256 does not match the manifest content.")
    return manifest


def load_manifest(path: str | Path, *, require_approved: bool = True) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_manifest(manifest, require_approved=require_approved)


def manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    rows = manifest["rows"]
    counts: dict[str, int] = defaultdict(int)
    conditions: dict[str, set[float]] = defaultdict(set)
    seeds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        dlc = str(row["dlc"])
        counts[dlc] += 1
        conditions[dlc].add(float(row["wind_speed_mps"]))
        seeds[dlc].add(int(row["seed"]))
    return {
        "status": manifest["status"],
        "manifest_sha256": manifest["manifest_sha256"],
        "row_count": len(rows),
        "rows_by_dlc": dict(sorted(counts.items())),
        "wind_speeds_by_dlc": {key: sorted(value) for key, value in sorted(conditions.items())},
        "unique_seeds_by_dlc": {key: sorted(value) for key, value in sorted(seeds.items())},
        "estimated_wind_bytes_total": int(manifest["estimated_wind_bytes_total"]),
        "execution_settings": deepcopy(manifest["execution_settings"]),
    }


def disk_preflight(
    manifest: dict[str, Any],
    *,
    required_factor: float = 1.25,
    available_bytes: int | None = None,
) -> dict[str, Any]:
    """Check free space against the conservatively estimated full wind matrix."""
    if not math.isfinite(required_factor) or required_factor < 1.0:
        raise DLCManifestError("Disk required_factor must be finite and at least 1.0.")
    wind_files = [Path(str(row["wind_file"])) for row in manifest["rows"]]
    common = Path(str(Path(wind_files[0]).parent))
    for path in wind_files[1:]:
        common = Path(str(Path(_common_path(common, path.parent))))
    existing = common
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.exists():
        raise DLCManifestError(f"Could not find an existing ancestor for wind path: {common}")
    free = int(available_bytes) if available_bytes is not None else int(shutil.disk_usage(existing).free)
    estimated = int(manifest["estimated_wind_bytes_total"])
    required = int(math.ceil(estimated * required_factor))
    report = {
        "wind_common_path": str(common),
        "checked_filesystem_path": str(existing),
        "estimated_wind_bytes_total": estimated,
        "required_factor": required_factor,
        "required_available_bytes": required,
        "available_bytes": free,
        "margin_bytes": free - required,
        "passed": free >= required,
    }
    if not report["passed"]:
        raise DLCManifestError(
            f"Insufficient wind-field disk space: require {required} bytes including margin, "
            f"but only {free} bytes are available at {existing}."
        )
    return report


def _common_path(left: Path, right: Path) -> Path:
    import os

    return Path(os.path.commonpath([str(left), str(right)]))


def wind_record_path(row: dict[str, Any]) -> Path:
    return Path(str(row["wind_file"])).with_suffix(".wind_record.json")


def build_wind_record(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    wind_file = Path(str(row["wind_file"]))
    if not wind_file.is_file():
        raise DLCManifestError(f"Wind file is missing: {wind_file}")
    record = {
        "schema_version": 1,
        "row_id": row["row_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "wind_file": str(wind_file),
        "bytes": wind_file.stat().st_size,
        "sha256": sha256_file(wind_file),
        "seed": row["seed"],
        "dlc": row["dlc"],
        "wind_speed_mps": row["wind_speed_mps"],
    }
    record["record_sha256"] = sha256_payload(record)
    return record


def validate_wind_record(row: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    path = wind_record_path(row)
    if not path.is_file():
        raise DLCManifestError(f"Wind record is missing: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("row_id") != row["row_id"] or record.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise DLCManifestError(f"Wind record does not match manifest row {row['row_id']}.")
    archived_record_digest = str(record.get("record_sha256", "") or "")
    without_digest = dict(record)
    without_digest.pop("record_sha256", None)
    if archived_record_digest != sha256_payload(without_digest):
        raise DLCManifestError(f"Wind record digest is invalid for row {row['row_id']}.")
    wind_file = Path(str(row["wind_file"]))
    if not wind_file.is_file():
        raise DLCManifestError(f"Wind file is missing: {wind_file}")
    if int(record.get("bytes", -1)) != wind_file.stat().st_size:
        raise DLCManifestError(f"Wind file size changed for row {row['row_id']}.")
    if str(record.get("sha256", "")) != sha256_file(wind_file):
        raise DLCManifestError(f"Wind file hash changed for row {row['row_id']}.")
    return record


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def selected_row(manifest: dict[str, Any], index: int) -> dict[str, Any]:
    rows = manifest["rows"]
    if index < 0 or index >= len(rows):
        raise DLCManifestError(f"Manifest row index {index} is outside 0..{len(rows) - 1}.")
    return rows[index]


def unique_conditions(rows: Iterable[dict[str, Any]]) -> set[tuple[str, float]]:
    return {(str(row["dlc"]), float(row["wind_speed_mps"])) for row in rows}
