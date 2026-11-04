"""Build the NACA0018 force--motion work map from paired histories.

The builder accepts only a complete frozen tensor of nonzero prescribed-motion
cases.  It subtracts the exact runtime-base force history from the paired
high-fidelity history, computes signed cycle-mean work, and adds the exact
zero-amplitude plane analytically.  The stationary NACA0018 spectrum supplies
only the reference fluctuating-lift amplitude; its arbitrary FFT phase is
never used for work identification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from vorlap.airfoil_io import load_airfoil_fft
from vorlap.airfoil_work_map_identification import (
    MINIMUM_COMPLETE_CYCLES_AFTER_DISCARD,
    MINIMUM_DISCARD_CYCLES,
    MINIMUM_SAMPLES_PER_CYCLE,
    assemble_airfoil_work_map_payload,
    identify_prescribed_motion_work,
)
from vorlap.interpolation import interpolate_fft_spectrum
from vorlap.qblade_airfoil_work_map import AirfoilForceMotionWorkMap

SCHEMA = "vorlap-airfoil-work-map-identification-manifest-v2"
REPORT_SCHEMA = "vorlap-airfoil-work-map-identification-report-v1"
TRACKED_STATIONARY_SPECTRUM_SHA256 = (
    "64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369"
)
_NALU_RESULTS_SCHEMA = "vorlap-naca0018-nalu-prescribed-motion-results-v1"
_NALU_CAMPAIGN_SCHEMA = "vorlap-naca0018-nalu-prescribed-motion-campaign-v1"
_NALU_MESH_AUDIT_SCHEMA = "vorlap-naca0018-exodus-geometry-audit-v1"
_NALU_SCALING_SCHEMA = "vorlap-naca0018-nalu-scaling-report-v1"
_NALU_VERSION = "v2.2.1"
_NALU_GIT_COMMIT = "ffa9de729df2a11b5241fdeb7628e7fab9f48f9b"
_QBLADE_RESULTS_SCHEMA = "vorlap-naca0018-qblade-prescribed-motion-results-v1"
_QBLADE_CAMPAIGN_SCHEMA = "vorlap-naca0018-qblade-prescribed-motion-campaign-v1"
_QBLADE_CASE_AUDIT_SCHEMA = "vorlap-naca0018-qblade-prescribed-motion-case-audit-v1"
_QBLADE_PREFLIGHT_SCHEMA = (
    "vorlap-naca0018-qblade-prescribed-motion-scheduler-v2"
)
_QBLADE_VERSION = "2.0.9.7"
_QBLADE_LIBRARY_SHA256 = (
    "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
)
_QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT = (
    "c50c6909317c754caec4130ccbc8e2ed3f705f810f07da11e7ce3ad003499089"
)
_CSV_COLUMNS = (
    "time_s",
    "displacement_m",
    "velocity_mps",
    "crossflow_force_n",
)
_CASE_KEY_ORDER = (
    "case_id",
    "incidence_deg",
    "reynolds_number",
    "reduced_frequency_k",
    "amplitude_ratio",
    "direction_deg_mod_180",
    "discard_cycles",
    "high_fidelity_history",
    "exact_runtime_base_history",
)
_CASE_KEYS = set(_CASE_KEY_ORDER)
_TRACKED_REQUEST_PATH = Path(__file__).with_name("work_map_data_request.json")
_REQUEST_SCHEMA = "vorlap-naca0018-prescribed-motion-data-request-v1"
_REQUEST_STATUS = "awaiting_high_fidelity_and_exact_runtime_base_histories"
_REQUEST_KEYS = {
    "schema",
    "status",
    "purpose",
    "profile",
    "motion",
    "force_motion_convention",
    "frozen_axes",
    "nonzero_motion_record_count_per_fidelity",
    "analytical_zero_amplitude_vertex_count",
    "zero_amplitude_policy",
    "frozen_fluid_and_section",
    "minimum_accepted_history",
    "pairing",
    "high_fidelity_initialization",
    "identification_manifest_format",
    "frozen_nonleakage_policy",
    "flight_parallel_plan",
}
_FROZEN_AXES = {
    "incidence_deg": [-90.0],
    "reynolds_number": [64000.0, 100000.0, 140000.0],
    "reduced_frequency_k": [0.45, 0.65, 0.8, 1.05],
    "amplitude_ratio": [0.0, 0.005, 0.0125, 0.025, 0.05],
    "direction_deg_mod_180": [0.0],
}
_FROZEN_PHYSICAL = {
    "fluid_density_kgpm3": 1.225,
    "fluid_dynamic_viscosity_pas": 1.813e-5,
    "reference_chord_m": 0.23,
    "section_span_m": 4.0,
    "speed_definition": "U=Re*mu/(rho*c)",
    "frequency_definition": "f=k*U/(pi*c)",
}
_FROZEN_FORCE_MOTION_CONVENTION = {
    "displacement": "y=A*sin(2*pi*f*t)",
    "velocity": "ydot=2*pi*f*A*cos(2*pi*f*t)",
    "positive_force": "same_global_crossflow_direction_as_positive_y",
    "positive_power": "flow_to_structure",
    "power_coefficient": "mean((F_high_fidelity-F_exact_runtime_base)*ydot)/(0.5*rho*U^3*c*span)",
    "reduced_frequency": "k=pi*f*c/U",
    "amplitude_ratio": "A/c",
}
_FROZEN_MINIMUM_HISTORY = {
    "discard_cycles": MINIMUM_DISCARD_CYCLES,
    "complete_cycles_after_discard": MINIMUM_COMPLETE_CYCLES_AFTER_DISCARD,
    "samples_per_cycle": MINIMUM_SAMPLES_PER_CYCLE,
    "required_synchronized_columns": list(_CSV_COLUMNS),
    "section_span_m": "must_be_declared_positive",
    "fluid_density_kgpm3": "must_be_declared_positive",
    "fluid_dynamic_viscosity_pas": "must_be_declared_positive",
    "reference_chord_m": 0.23,
    "force_history_units": "total force in N over declared section span",
}
_FROZEN_NONLEAKAGE = {
    "ehrmann_free_response_used_for_fit": False,
    "ehrmann_acceptance_tolerance_used_for_fit": False,
    "static_fft_phase_used_for_work": False,
    "grinderslev_power_used_for_naca_magnitude": False,
    "static_naca_spectrum_role": "source_frequency_and_available_lift_energy_only",
    "minimum_map_condition_for_free_response": "positive lower-confidence work exists and highest-amplitude plane is nonpositive",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _json_no_duplicates(raw: bytes, label: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}.")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON.") from exc


def _digest(value: Any, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return digest


def _relative_file(root: Path, value: Any, label: str) -> Path:
    text = str(value)
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative path.")
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must identify a regular non-symlink file.")
    return path


def _verified_file(root: Path, identity: Any, label: str) -> tuple[Path, bytes, str]:
    if not isinstance(identity, Mapping) or set(identity) != {"path", "sha256"}:
        raise ValueError(f"{label} must have exact path/sha256 keys.")
    path = _relative_file(root, identity["path"], f"{label}.path")
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    expected = _digest(identity["sha256"], f"{label}.sha256")
    if actual != expected:
        raise ValueError(f"{label} SHA-256 does not match its bytes.")
    return path, raw, actual


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive.")
    return result


def _strict_equal(value: Any, expected: Any) -> bool:
    """Return equality without Python's bool/int or int/float coercions."""

    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _strict_equal(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _strict_equal(item, reference)
            for item, reference in zip(value, expected, strict=True)
        )
    return bool(value == expected)


def _finite_axis_value(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _validate_frozen_request(request: Any) -> None:
    """Validate the tracked request as policy, not merely as parseable JSON."""

    if not isinstance(request, Mapping) or set(request) != _REQUEST_KEYS:
        raise ValueError(
            "The data request top-level keys differ from the frozen schema."
        )
    required_scalars = {
        "schema": _REQUEST_SCHEMA,
        "status": _REQUEST_STATUS,
        "purpose": "populate_signed_incremental_airfoil_work_map_without_using_Ehrmann_free_response_for_tuning",
        "profile": "NACA0018",
        "motion": "prescribed_crossflow_translation",
        "nonzero_motion_record_count_per_fidelity": 48,
        "analytical_zero_amplitude_vertex_count": 12,
        "zero_amplitude_policy": "set entrainment, signed power, and uncertainty exactly to zero without pretending that a zero-motion history identifies phase",
    }
    for key, expected in required_scalars.items():
        if not _strict_equal(request[key], expected):
            raise ValueError(
                f"The data request {key} differs from the frozen contract."
            )
    required_mappings = {
        "force_motion_convention": _FROZEN_FORCE_MOTION_CONVENTION,
        "frozen_axes": _FROZEN_AXES,
        "frozen_fluid_and_section": _FROZEN_PHYSICAL,
        "minimum_accepted_history": _FROZEN_MINIMUM_HISTORY,
        "frozen_nonleakage_policy": _FROZEN_NONLEAKAGE,
    }
    for key, expected in required_mappings.items():
        value = request[key]
        if not isinstance(value, Mapping) or not _strict_equal(dict(value), expected):
            raise ValueError(
                f"The data request {key} differs from the frozen contract."
            )

    pairing = request["pairing"]
    expected_pairing = {
        "high_fidelity": "prescribed-motion NACA0018 CFD or synchronized experiment",
        "exact_runtime_base": "native QBlade prescribed-motion force at the identical axis point and motion history",
        "required_identity_match": [
            "incidence_deg",
            "reynolds_number",
            "reduced_frequency_k",
            "amplitude_ratio",
            "direction_deg_mod_180",
            "time_grid",
            "displacement_history",
            "velocity_history",
            "chord_m",
            "span_m",
            "fluid_properties",
        ],
    }
    if not isinstance(pairing, Mapping) or not _strict_equal(
        dict(pairing), expected_pairing
    ):
        raise ValueError(
            "The data request pairing identity differs from the frozen contract."
        )

    initialization = request["high_fidelity_initialization"]
    expected_initialization = {
        "solver": "official Nalu-Wind v2.2.1 release at git commit ffa9de729df2a11b5241fdeb7628e7fab9f48f9b",
        "mesh": "the original periodic NACA0018 span-4m IDDES mesh with exact file identity bound at campaign generation",
        "mesh_transformation": "apply the one-time [0.23,0.23,1.0] x/y/z scaling only in the precursor initialized from the original chord-1m mesh; motion cases restart from the already-scaled precursor mesh without applying the scaling again",
        "precursor_count": 3,
        "precursor_policy": "one stationary developed-wake precursor per Reynolds number using the same all-block mesh_motion schema with zero oscillation amplitude, branched to all k and A/c cases at that Reynolds number",
        "precursor_restart_time_policy": "the selected precursor restart timestamp must equal 20*pi*c/U on an exact precursor step endpoint computed from the same unrounded U and c used by every child case",
        "motion": "native all-block ALE oscillation with start_time equal to the precursor restart time",
        "phase_policy": "the precursor time is 20*pi*c/U, making f*t0=20*k exactly 9,13,16,21 cycles for the frozen k grid; displacement starts at zero and velocity at positive maximum",
        "wall_policy": "Nalu copies mesh_velocity to the no-slip wall when mesh motion is active",
        "force_surface": "the full [wing,wing-pp] surface over the original 4m periodic span; wing-pp alone is forbidden",
        "force_channel": "crossflow_force_n=Fpy+Fvy from surface_force_and_moment",
        "force_sign": "positive fluid-on-body force in the positive prescribed-motion direction",
        "runtime_base_geometry": "native QBlade uses the same 0.23m chord, 4m span, incidence, fluid properties, and total-force normalization",
        "runtime_base_force_channel": "total native aerodynamic force in global crossflow Y over the complete 4m section",
    }
    if not isinstance(initialization, Mapping) or not _strict_equal(
        dict(initialization), expected_initialization
    ):
        raise ValueError(
            "The data request high-fidelity initialization differs from the frozen contract."
        )

    manifest_format = request["identification_manifest_format"]
    if not isinstance(manifest_format, Mapping) or set(manifest_format) != {
        "schema",
        "status_required_for_build",
        "top_level_keys",
        "file_identity",
        "case_keys",
        "builder",
    }:
        raise ValueError(
            "The data request manifest format differs from the frozen contract."
        )
    if (
        manifest_format["schema"] != SCHEMA
        or manifest_format["status_required_for_build"] != "complete"
        or not _strict_equal(
            manifest_format["top_level_keys"],
            [
                "schema",
                "status",
                "profile",
                "request",
                "static_spectrum",
                "producer_provenance",
                "fluid_and_section",
                "cases",
            ],
        )
        or not _strict_equal(
            manifest_format["file_identity"],
            {
                "path": "canonical path relative to the manifest directory",
                "sha256": "SHA-256 of the exact bytes parsed",
            },
        )
        or not _strict_equal(manifest_format["case_keys"], list(_CASE_KEY_ORDER))
        or manifest_format["builder"]
        != "python -m examples.ehrmann2014_naca0018.build_airfoil_work_map --manifest MANIFEST.json --output-map NACA0018_work_map.json --output-report NACA0018_work_map_report.json"
    ):
        raise ValueError(
            "The data request manifest format is internally contradictory."
        )

    parallel = request["flight_parallel_plan"]
    if not isinstance(parallel, Mapping) or set(parallel) != {
        "high_fidelity_cases",
        "runtime_base_cases",
        "preferred_execution",
        "preferred_target_walltime",
        "maximum_requested_walltime",
        "node_selection_policy",
        "submission_status",
        "scope_boundary",
    }:
        raise ValueError(
            "The data request Flight plan differs from the frozen contract."
        )
    expected_parallel = {
        "high_fidelity_cases": 48,
        "runtime_base_cases": 48,
        "preferred_execution": "one_independent_case_per_physical_core_or_solver_scaling_equivalent",
        "preferred_target_walltime": "02:00:00",
        "maximum_requested_walltime": "08:00:00",
        "node_selection_policy": "smallest_2h_passing_size_among_2_4_8_else_smallest_8h_passing_size",
        "submission_status": "requires_executed_one_case_nalu_and_qblade_motion_force_preflight_before_full_matrix",
        "scope_boundary": "this_minus_90_degree_slice can qualify Ehrmann only; H-VAWT requires a separately frozen incidence/Re extension after Ehrmann passes",
    }
    if not _strict_equal(dict(parallel), expected_parallel):
        raise ValueError(
            "The data request Flight plan differs from the frozen contract."
        )

    axes = request["frozen_axes"]
    nonzero_count = math.prod(
        len(axes[name])
        for name in (
            "incidence_deg",
            "reynolds_number",
            "reduced_frequency_k",
            "direction_deg_mod_180",
        )
    ) * sum(float(value) > 0.0 for value in axes["amplitude_ratio"])
    origin_count = math.prod(
        len(axes[name])
        for name in (
            "incidence_deg",
            "reynolds_number",
            "reduced_frequency_k",
            "direction_deg_mod_180",
        )
    )
    if nonzero_count != 48 or origin_count != 12:
        raise ValueError(
            "The frozen axes must form exactly 48 nonzero and 12 origin vertices."
        )


def _load_verified_tracked_request(request_raw: bytes) -> Mapping[str, Any]:
    tracked_raw = _TRACKED_REQUEST_PATH.read_bytes()
    tracked = _json_no_duplicates(tracked_raw, "tracked data request")
    _validate_frozen_request(tracked)
    if request_raw != tracked_raw:
        raise ValueError(
            "The manifest data request bytes do not exactly match tracked "
            "work_map_data_request.json."
        )
    request = _json_no_duplicates(request_raw, "data request")
    _validate_frozen_request(request)
    return request


def _read_history(raw: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8 CSV.") from exc
    if tuple(reader.fieldnames or ()) != _CSV_COLUMNS:
        raise ValueError(f"{label} columns must be exactly {_CSV_COLUMNS}.")
    values = {name: [] for name in _CSV_COLUMNS}
    for row_index, row in enumerate(reader, start=2):
        if None in row or set(row) != set(_CSV_COLUMNS):
            raise ValueError(f"{label} row {row_index} has the wrong field count.")
        for name in _CSV_COLUMNS:
            try:
                number = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} row {row_index} {name} is not numeric."
                ) from exc
            if not math.isfinite(number):
                raise ValueError(f"{label} row {row_index} {name} is not finite.")
            values[name].append(number)
    arrays = {name: np.asarray(items, dtype=float) for name, items in values.items()}
    if arrays["time_s"].size < 4:
        raise ValueError(f"{label} must contain at least four data rows.")
    return arrays


def _motion_identity_check(
    history: Mapping[str, np.ndarray],
    *,
    frequency_hz: float,
    amplitude_m: float,
) -> None:
    time = history["time_s"]
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Prescribed-motion time must be strictly increasing.")
    omega = 2.0 * math.pi * frequency_hz
    expected_displacement = amplitude_m * np.sin(omega * time)
    expected_velocity = omega * amplitude_m * np.cos(omega * time)
    displacement_tolerance = max(1.0e-8 * amplitude_m, 1.0e-12)
    velocity_tolerance = max(1.0e-8 * omega * amplitude_m, 1.0e-12)
    if not np.allclose(
        history["displacement_m"],
        expected_displacement,
        rtol=0.0,
        atol=displacement_tolerance,
    ):
        raise ValueError(
            "Prescribed displacement does not follow the frozen y=A*sin(2*pi*f*t) convention."
        )
    if not np.allclose(
        history["velocity_mps"],
        expected_velocity,
        rtol=0.0,
        atol=velocity_tolerance,
    ):
        raise ValueError(
            "Prescribed velocity does not follow the frozen ydot=2*pi*f*A*cos(2*pi*f*t) convention."
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _selected_semantics(
    value: Any, expected: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    for key, reference in expected.items():
        if key not in mapping or not _strict_equal(mapping[key], reference):
            raise ValueError(
                f"{label}.{key} differs from the frozen producer contract."
            )
    return mapping


def _verified_json_file(
    root: Path, identity: Any, label: str
) -> tuple[Mapping[str, Any], str]:
    _path, raw, digest = _verified_file(root, identity, label)
    payload = _json_no_duplicates(raw, label)
    return _mapping(payload, label), digest


def _recorded_nalu_executable(value: Any, label: str) -> dict[str, str]:
    identity = _mapping(value, label)
    path = str(identity.get("path"))
    pure = PurePosixPath(path)
    if (
        set(identity) != {"path", "sha256"}
        or not isinstance(identity.get("path"), str)
        or not pure.is_absolute()
        or str(pure) != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{label} must contain a canonical absolute path and SHA-256.")
    return {
        "path": path,
        "sha256": _digest(identity["sha256"], f"{label}.sha256"),
    }


def _validate_nalu_run_runtime(
    value: Any,
    expected_executable: Mapping[str, str],
    expected_terminal_step: int,
    label: str,
) -> None:
    runtime = _mapping(value, label)
    if set(runtime) != {
        "status",
        "nalu_executable",
        "observed_solver_version",
        "observed_solver_git_commit",
        "terminal_step",
        "log",
    }:
        raise ValueError(f"{label} keys differ from the completed-run contract.")
    log = _mapping(runtime.get("log"), f"{label}.log")
    log_path = PurePosixPath(str(log.get("path")))
    if (
        runtime.get("status") != "verified"
        or _recorded_nalu_executable(runtime.get("nalu_executable"), label)
        != dict(expected_executable)
        or runtime.get("observed_solver_version") != _NALU_VERSION
        or runtime.get("observed_solver_git_commit") != _NALU_GIT_COMMIT
        or runtime.get("terminal_step") != expected_terminal_step
        or set(log) != {"path", "sha256"}
        or log_path.is_absolute()
        or not log_path.parts
        or any(part in {"", ".", ".."} for part in log_path.parts)
    ):
        raise ValueError(f"{label} does not prove the frozen Nalu runtime.")
    _digest(log.get("sha256"), f"{label}.log.sha256")


def _campaign_fingerprint(payload: Mapping[str, Any], label: str) -> str:
    fingerprint = _digest(payload.get("fingerprint"), f"{label}.fingerprint")
    normalized = dict(payload)
    normalized.pop("fingerprint", None)
    if fingerprint != _semantic_sha256(normalized):
        raise ValueError(f"{label} fingerprint differs from its semantic content.")
    return fingerprint


def _case_index(payload: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    rows = payload.get("cases")
    if not isinstance(rows, list) or len(rows) != 48:
        raise ValueError(f"{label} must contain exactly 48 cases.")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item = _mapping(row, f"{label} case")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in result:
            raise ValueError(f"{label} case IDs must be unique non-empty strings.")
        result[case_id] = item
    return result


def _canonical_case_id(reynolds: float, k: float, amplitude_ratio: float) -> str:
    def tag(value: float, digits: int) -> str:
        return f"{value:.{digits}f}".replace("-", "m").replace(".", "p")

    return f"re{int(round(reynolds)):06d}_k{tag(k, 2)}_" f"a{tag(amplitude_ratio, 4)}"


def _expected_motion_values(
    *, reynolds: float, k: float, amplitude_ratio: float
) -> dict[str, float | int]:
    density = _FROZEN_PHYSICAL["fluid_density_kgpm3"]
    viscosity = _FROZEN_PHYSICAL["fluid_dynamic_viscosity_pas"]
    chord = _FROZEN_PHYSICAL["reference_chord_m"]
    speed = reynolds * viscosity / (density * chord)
    frequency = k * speed / (math.pi * chord)
    period = 1.0 / frequency
    samples = math.ceil(max(100.0, 120.0 * 0.4 * math.pi / k))
    rows = 13 * samples + 1
    precursor_restart_time = 20.0 * math.pi * chord / speed
    precursor_maximum_dt = chord / (120.0 * 0.4 * speed)
    precursor_steps = math.ceil(precursor_restart_time / precursor_maximum_dt)
    return {
        "wind_speed_mps": speed,
        "frequency_hz": frequency,
        "period_s": period,
        "amplitude_m": amplitude_ratio * chord,
        "samples_per_cycle": samples,
        "time_step_s": period / samples,
        "rows": rows,
        "relative_history_start_s": period / samples,
        "relative_history_end_s": rows * period / samples,
        "precursor_restart_time_s": precursor_restart_time,
        "precursor_steps": precursor_steps,
        "phase_cycles_at_restart": 20.0 * k,
    }


def _same_number(actual: Any, expected: float, label: str) -> float:
    value = _finite_axis_value(actual, label)
    if not math.isclose(value, expected, rel_tol=2.0e-13, abs_tol=1.0e-14):
        raise ValueError(f"{label} differs from the frozen derived value.")
    return value


def _validate_case_crosslinks(
    *,
    identification_cases: Mapping[str, Mapping[str, Any]],
    producer_results: Mapping[str, Mapping[str, Any]],
    producer_campaign: Mapping[str, Mapping[str, Any]],
    history_role: str,
    label: str,
) -> None:
    if set(identification_cases) != set(producer_results) or set(
        identification_cases
    ) != set(producer_campaign):
        raise ValueError(f"{label} case IDs differ from the identification manifest.")
    for index, case_id in enumerate(sorted(identification_cases)):
        identified = identification_cases[case_id]
        result = producer_results[case_id]
        campaign = producer_campaign[case_id]
        for key in (
            "incidence_deg",
            "reynolds_number",
            "reduced_frequency_k",
            "amplitude_ratio",
            "direction_deg_mod_180",
            "discard_cycles",
        ):
            if not _strict_equal(
                result.get(key), identified.get(key)
            ) or not _strict_equal(campaign.get(key), identified.get(key)):
                raise ValueError(f"{label} {case_id}.{key} is not cross-linked.")
        reynolds = float(identified["reynolds_number"])
        k = float(identified["reduced_frequency_k"])
        amplitude_ratio = float(identified["amplitude_ratio"])
        if case_id != _canonical_case_id(reynolds, k, amplitude_ratio):
            raise ValueError(
                f"{label} case ID is not canonical for its vertex: {case_id}."
            )
        expected = _expected_motion_values(
            reynolds=reynolds, k=k, amplitude_ratio=amplitude_ratio
        )
        if campaign.get("index") != index:
            raise ValueError(f"{label} {case_id} index is not canonical.")
        for key in (
            "wind_speed_mps",
            "frequency_hz",
            "period_s",
            "amplitude_m",
            "time_step_s",
        ):
            _same_number(
                campaign.get(key), float(expected[key]), f"{label} {case_id}.{key}"
            )
        if campaign.get("samples_per_cycle") != expected["samples_per_cycle"]:
            raise ValueError(f"{label} {case_id} samples_per_cycle differs.")
        result_history = _mapping(
            result.get("canonical_history"), f"{label} {case_id} result history"
        )
        campaign_history = _mapping(
            campaign.get("canonical_history"), f"{label} {case_id} campaign history"
        )
        identified_history = _mapping(
            identified.get(history_role), f"{label} {case_id} identified history"
        )
        if (
            result_history.get("status") != "complete"
            or result_history.get("columns") != list(_CSV_COLUMNS)
            or result_history.get("rows") != expected["rows"]
            or result_history.get("rows_expected") != expected["rows"]
            or campaign_history.get("rows_expected") != expected["rows"]
            or result_history.get("sha256") != identified_history.get("sha256")
        ):
            raise ValueError(
                f"{label} {case_id} canonical history is not cross-linked."
            )


def _validate_nalu_provenance(
    root: Path,
    provenance: Mapping[str, Any],
    *,
    request_sha: str,
    identification_cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        set(provenance)
        != {
            "kind",
            "results_manifest",
            "campaign_manifest",
            "mesh_audit",
            "scaling_reports",
        }
        or provenance.get("kind") != "nalu_wind_prescribed_motion"
    ):
        raise ValueError("High-fidelity producer provenance keys or kind differ.")
    results, results_sha = _verified_json_file(
        root, provenance["results_manifest"], "Nalu results snapshot"
    )
    campaign, campaign_sha = _verified_json_file(
        root, provenance["campaign_manifest"], "Nalu campaign snapshot"
    )
    mesh_audit, mesh_audit_sha = _verified_json_file(
        root, provenance["mesh_audit"], "Nalu mesh-audit snapshot"
    )
    _selected_semantics(
        results,
        {"schema": _NALU_RESULTS_SCHEMA, "status": "complete"},
        "Nalu results snapshot",
    )
    _selected_semantics(
        campaign,
        {"schema": _NALU_CAMPAIGN_SCHEMA},
        "Nalu campaign snapshot",
    )
    fingerprint = _campaign_fingerprint(campaign, "Nalu campaign snapshot")
    if (
        results.get("campaign_fingerprint") != fingerprint
        or results.get("campaign_manifest") != "campaign_manifest.json"
    ):
        raise ValueError("Nalu result/campaign provenance is not cross-linked.")
    for payload, label in ((results, "Nalu results"), (campaign, "Nalu campaign")):
        request = _mapping(payload.get("request"), f"{label} request")
        if request.get("sha256") != request_sha:
            raise ValueError(f"{label} does not use the frozen request bytes.")

    expected_solver = {
        "name": "Nalu-Wind",
        "version": _NALU_VERSION,
        "expected_git_commit": _NALU_GIT_COMMIT,
        "turbulence_model": "sst_iddes",
        "transition_model": "none",
    }
    solver = _selected_semantics(results.get("solver"), expected_solver, "Nalu solver")
    if not _strict_equal(results.get("solver"), campaign.get("solver")):
        raise ValueError("Nalu result and campaign solver identities differ.")
    del solver
    expected_physical = {
        key: _FROZEN_PHYSICAL[key]
        for key in (
            "fluid_density_kgpm3",
            "fluid_dynamic_viscosity_pas",
            "reference_chord_m",
            "section_span_m",
        )
    }
    if not _strict_equal(
        results.get("fluid_and_section"), expected_physical
    ) or not _strict_equal(campaign.get("fluid_and_section"), expected_physical):
        raise ValueError(
            "Nalu producer fluid/section values differ from the frozen request."
        )
    mesh = _selected_semantics(
        campaign.get("mesh"),
        {
            "source_chord_m": 1.0,
            "source_span_m": 4.0,
            "transformation_xyz": [0.23, 0.23, 1.0],
            "transformation_stage": "precursor_only",
            "required_parts": [
                "fluid-hex",
                "wing",
                "wing-pp",
                "inlet",
                "outlet",
                "front",
                "back",
            ],
        },
        "Nalu mesh contract",
    )
    if not _strict_equal(results.get("mesh"), campaign.get("mesh")):
        raise ValueError("Nalu result and campaign mesh identities differ.")
    _selected_semantics(
        results.get("force_contract"),
        {
            "surface": ["wing", "wing-pp"],
            "crossflow_expression": "Fpy+Fvy",
            "sign": "positive_fluid_on_body_global_positive_Y",
            "units": "N_total_over_declared_4m_span",
        },
        "Nalu force contract",
    )
    history_contract = {
        "time_origin": "precursor_restart",
        "first_row": "dt",
        "last_row": "(13*samples_per_cycle+1)*dt",
        "complete_cycle_intervals": 13,
        "raw_absolute_time_rule": "t0+j*dt for j=1..13*N+1",
        "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
        "columns": list(_CSV_COLUMNS),
        "float_encoding": ".17g",
        "pairing_policy": "exact_array_equal_time_displacement_velocity_no_interpolation",
        "force_units": "N_total_over_4m_span",
    }
    if not _strict_equal(
        results.get("history_contract"), history_contract
    ) or not _strict_equal(campaign.get("history_contract"), history_contract):
        raise ValueError("Nalu canonical-history contract differs.")
    numerics = _selected_semantics(
        campaign.get("numerics"),
        {
            "discard_cycles": 5,
            "retained_cycles": 8,
            "updates_rule": "13*samples_per_cycle+1",
            "force_surface": ["wing", "wing-pp"],
            "crossflow_force_expression": "Fpy+Fvy",
        },
        "Nalu numerics",
    )
    del numerics

    _selected_semantics(
        mesh_audit,
        {
            "schema": _NALU_MESH_AUDIT_SCHEMA,
            "status": "pass",
            "campaign_fingerprint": fingerprint,
            "combined_force_surface": ["wing", "wing-pp"],
            "chord_axis": "global_Y",
            "incidence_orientation_deg": -90.0,
        },
        "Nalu mesh audit",
    )
    audit_mesh = _mapping(mesh_audit.get("mesh"), "Nalu mesh audit identity")
    result_mesh_audit = _mapping(results.get("mesh_audit"), "Nalu result mesh audit")
    if (
        audit_mesh.get("sha256") != mesh.get("sha256")
        or result_mesh_audit.get("status") != "pass"
        or result_mesh_audit.get("sha256") != mesh_audit_sha
    ):
        raise ValueError(
            "Nalu mesh audit is not identity-bound to the result and mesh."
        )
    _same_number(mesh_audit.get("source_span_m"), 4.0, "Nalu audited span")
    source_chord = _finite_positive(
        mesh_audit.get("source_chord_m"), "Nalu audited chord"
    )
    if not 0.95 <= source_chord <= 1.05:
        raise ValueError("Nalu audited source chord is outside the frozen mesh bounds.")
    observed_parts: set[str] = set()
    for key in ("element_block_names", "side_set_names"):
        names = mesh_audit.get(key)
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError(
                "Nalu mesh audit does not contain observed Exodus part names."
            )
        observed_parts.update(names)
    if not set(mesh["required_parts"]).issubset(observed_parts):
        raise ValueError("Nalu mesh audit does not prove every frozen Exodus part.")

    scaling = provenance.get("scaling_reports")
    if not isinstance(scaling, list) or len(scaling) != 3:
        raise ValueError("Nalu provenance must snapshot the 2/4/8-node scaling sweep.")
    scaling_shas: dict[str, str] = {}
    target_pass: dict[int, bool] = {}
    maximum_pass: dict[int, bool] = {}
    scaling_executable: dict[str, str] | None = None
    for row in scaling:
        if not isinstance(row, Mapping) or set(row) != {"nodes", "file"}:
            raise ValueError(
                "Nalu scaling provenance rows must have exact nodes/file keys."
            )
        nodes = row["nodes"]
        if (
            type(nodes) is not int
            or nodes not in {2, 4, 8}
            or str(nodes) in scaling_shas
        ):
            raise ValueError(
                "Nalu scaling provenance must contain unique 2/4/8-node rows."
            )
        report, digest = _verified_json_file(
            root, row["file"], f"Nalu {nodes}-node scaling snapshot"
        )
        _selected_semantics(
            report,
            {
                "schema": _NALU_SCALING_SCHEMA,
                "campaign_fingerprint": fingerprint,
                "nodes": nodes,
                "ranks": nodes * 112,
                "reynolds_number": 64000.0,
                "probe_updates": 64,
                "observed_solver_version": _NALU_VERSION,
                "observed_solver_git_commit": _NALU_GIT_COMMIT,
                "terminal_step": 64,
                "target_walltime_hours": 2.0,
                "maximum_walltime_hours": 8.0,
            },
            f"Nalu {nodes}-node scaling report",
        )
        report_executable = _recorded_nalu_executable(
            report.get("nalu_executable"),
            f"Nalu {nodes}-node scaling executable",
        )
        if scaling_executable is None:
            scaling_executable = report_executable
        elif report_executable != scaling_executable:
            raise ValueError(
                "Nalu 2/4/8-node scaling reports used different executables."
            )
        projected = [
            _finite_positive(report.get(key), f"Nalu scaling report {key}")
            for key in ("projected_precursor_hours", "projected_longest_motion_hours")
        ]
        projected_maximum = max(projected)
        expected_target_pass = projected_maximum <= 2.0
        expected_maximum_pass = projected_maximum <= 8.0
        expected_status = "pass" if expected_maximum_pass else "fail"
        if (
            report.get("status") != expected_status
            or report.get("target_walltime_pass") is not expected_target_pass
            or report.get("maximum_walltime_pass") is not expected_maximum_pass
        ):
            raise ValueError(
                "Nalu scaling-report verdict is inconsistent with its projection."
            )
        scaling_shas[str(nodes)] = digest
        target_pass[nodes] = expected_target_pass
        maximum_pass[nodes] = expected_maximum_pass
    if set(scaling_shas) != {"2", "4", "8"}:
        raise ValueError("Nalu scaling provenance does not cover 2/4/8 nodes.")
    if scaling_executable is None:
        raise ValueError("Nalu scaling provenance lacks an executable identity.")
    gate = _mapping(results.get("scaling_gate"), "Nalu scaling gate")
    target_nodes = [nodes for nodes in (2, 4, 8) if target_pass[nodes]]
    fallback_nodes = [nodes for nodes in (2, 4, 8) if maximum_pass[nodes]]
    selected_nodes = target_nodes if target_nodes else fallback_nodes
    recommended = min(selected_nodes) if selected_nodes else None
    expected_tier = "target_2h" if target_nodes else "fallback_8h"
    expected_walltime = "02:00:00" if target_nodes else "08:00:00"
    runtime_identity = _mapping(
        gate.get("runtime_identity"), "Nalu scaling runtime identity"
    )
    gate_executable = _recorded_nalu_executable(
        runtime_identity.get("nalu_executable"),
        "Nalu scaling runtime executable",
    )
    if (
        recommended is None
        or gate.get("status") != "pass"
        or gate.get("required_probe_count") != 3
        or gate.get("complete_probe_count") != 3
        or gate.get("recommended_nodes") != recommended
        or gate.get("recommended_ranks") != recommended * 112
        or gate.get("selection_tier") != expected_tier
        or gate.get("recommended_walltime") != expected_walltime
        or gate.get("target_walltime_hours") != 2.0
        or gate.get("maximum_walltime_hours") != 8.0
        or set(runtime_identity)
        != {
            "status",
            "observed_versions",
            "observed_git_commits",
            "nalu_executable",
        }
        or runtime_identity.get("status") != "verified"
        or runtime_identity.get("observed_versions") != [_NALU_VERSION]
        or runtime_identity.get("observed_git_commits") != [_NALU_GIT_COMMIT]
        or gate_executable != scaling_executable
    ):
        raise ValueError("Nalu scaling gate is incomplete or inconsistent.")
    by_nodes = _mapping(gate.get("by_nodes"), "Nalu scaling gate by_nodes")
    if set(by_nodes) != {"2", "4", "8"}:
        raise ValueError("Nalu scaling gate by_nodes coverage differs.")
    for nodes in (2, 4, 8):
        row = _mapping(by_nodes[str(nodes)], f"Nalu scaling gate {nodes} nodes")
        if (
            row.get("complete_reynolds_sweep") is not True
            or row.get("reports") != 1
            or row.get("target_walltime_pass") is not target_pass[nodes]
            or row.get("maximum_walltime_pass") is not maximum_pass[nodes]
            or row.get("pass") is not maximum_pass[nodes]
        ):
            raise ValueError("Nalu scaling gate does not match its snapshots.")

    summary = _mapping(results.get("summary"), "Nalu result summary")
    if (
        summary.get("complete_precursors") != 3
        or summary.get("required_precursors") != 3
        or summary.get("complete_motion_cases") != 48
        or summary.get("failed_motion_cases") != 0
        or summary.get("failed_precursors") != 0
        or summary.get("required_motion_cases") != 48
    ):
        raise ValueError(
            "Nalu result summary is not a clean 3-precursor/48-case result."
        )
    precursor_rows = results.get("precursors")
    if (
        not isinstance(precursor_rows, list)
        or len(precursor_rows) != 3
        or {
            float(row.get("reynolds_number"))
            for row in precursor_rows
            if isinstance(row, Mapping) and row.get("status") == "complete"
        }
        != {64000.0, 100000.0, 140000.0}
    ):
        raise ValueError("Nalu developed-wake precursor evidence is incomplete.")
    campaign_precursor_rows = campaign.get("precursors")
    if (
        not isinstance(campaign_precursor_rows, list)
        or len(campaign_precursor_rows) != 3
    ):
        raise ValueError("Nalu campaign precursor evidence is incomplete.")
    campaign_precursors = {
        row.get("precursor_id"): row
        for row in campaign_precursor_rows
        if isinstance(row, Mapping) and isinstance(row.get("precursor_id"), str)
    }
    if len(campaign_precursors) != 3:
        raise ValueError("Nalu campaign precursor IDs are incomplete.")
    for row in precursor_rows:
        source = campaign_precursors.get(row.get("precursor_id"))
        if (
            source is None
            or row.get("required_step") != source.get("precursor_steps")
            or row.get("maximum_logged_step") != source.get("precursor_steps")
        ):
            raise ValueError(
                "Nalu precursor terminal-step evidence is not cross-linked."
            )
        _validate_nalu_run_runtime(
            row.get("runtime_identity"),
            scaling_executable,
            int(source["precursor_steps"]),
            f"Nalu precursor {row.get('precursor_id')} runtime",
        )

    result_cases = _case_index(results, "Nalu results")
    campaign_cases = _case_index(campaign, "Nalu campaign")
    _validate_case_crosslinks(
        identification_cases=identification_cases,
        producer_results=result_cases,
        producer_campaign=campaign_cases,
        history_role="high_fidelity_history",
        label="Nalu",
    )
    for case_id, row in result_cases.items():
        expected = _expected_motion_values(
            reynolds=float(row["reynolds_number"]),
            k=float(row["reduced_frequency_k"]),
            amplitude_ratio=float(row["amplitude_ratio"]),
        )
        campaign_row = campaign_cases[case_id]
        if (
            row.get("solver_status") != "complete"
            or row.get("failure_marker") is not None
            or row.get("selected_force_rows") != expected["rows"]
            or campaign_row.get("retained_cycles") != 8
            or campaign_row.get("updates_after_restart") != expected["rows"]
            or campaign_row.get("precursor_restart_step") != expected["precursor_steps"]
            or row.get("precursor_restart_step") != expected["precursor_steps"]
        ):
            raise ValueError(f"Nalu {case_id} is not a clean completed motion solve.")
        _validate_nalu_run_runtime(
            row.get("runtime_identity"),
            scaling_executable,
            int(campaign_row["precursor_restart_step"])
            + int(campaign_row["updates_after_restart"]),
            f"Nalu {case_id} runtime",
        )
        for key in (
            "relative_history_start_s",
            "relative_history_end_s",
            "precursor_restart_time_s",
            "phase_cycles_at_restart",
        ):
            _same_number(
                campaign_row.get(key), float(expected[key]), f"Nalu {case_id}.{key}"
            )
    preflight = _mapping(results.get("motion_preflight"), "Nalu motion preflight")
    preflight_case = result_cases.get(preflight.get("case_id"))
    if (
        preflight.get("status") != "complete"
        or preflight_case is None
        or preflight.get("history_sha256")
        != _mapping(
            preflight_case.get("canonical_history"), "Nalu preflight history"
        ).get("sha256")
    ):
        raise ValueError("Nalu complete moving-mesh preflight is not cross-linked.")
    return {
        "semantic_sha256": _semantic_sha256(provenance),
        "results_manifest_sha256": results_sha,
        "campaign_manifest_sha256": campaign_sha,
        "mesh_audit_sha256": mesh_audit_sha,
        "scaling_report_sha256_by_nodes": scaling_shas,
    }


def _validate_qblade_provenance(
    root: Path,
    provenance: Mapping[str, Any],
    *,
    request_sha: str,
    identification_cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        set(provenance)
        != {
            "kind",
            "results_manifest",
            "campaign_manifest",
            "preflight_scheduler_report",
            "case_audits",
        }
        or provenance.get("kind") != "native_qblade_prescribed_motion"
    ):
        raise ValueError("Runtime-base producer provenance keys or kind differ.")
    results, results_sha = _verified_json_file(
        root, provenance["results_manifest"], "QBlade results snapshot"
    )
    campaign, campaign_sha = _verified_json_file(
        root, provenance["campaign_manifest"], "QBlade campaign snapshot"
    )
    scheduler, scheduler_sha = _verified_json_file(
        root,
        provenance["preflight_scheduler_report"],
        "QBlade preflight scheduler snapshot",
    )
    _selected_semantics(
        results,
        {"schema": _QBLADE_RESULTS_SCHEMA, "status": "complete"},
        "QBlade results snapshot",
    )
    _selected_semantics(
        campaign,
        {"schema": _QBLADE_CAMPAIGN_SCHEMA},
        "QBlade campaign snapshot",
    )
    fingerprint = _campaign_fingerprint(campaign, "QBlade campaign snapshot")
    if (
        results.get("campaign_fingerprint") != fingerprint
        or results.get("campaign_manifest") != "campaign_manifest.json"
    ):
        raise ValueError("QBlade result/campaign provenance is not cross-linked.")
    for payload, label in ((results, "QBlade results"), (campaign, "QBlade campaign")):
        request = _mapping(payload.get("request"), f"{label} request")
        if request.get("sha256") != request_sha:
            raise ValueError(f"{label} does not use the frozen request bytes.")
    expected_physical = {
        key: _FROZEN_PHYSICAL[key]
        for key in (
            "fluid_density_kgpm3",
            "fluid_dynamic_viscosity_pas",
            "reference_chord_m",
            "section_span_m",
        )
    }
    if not _strict_equal(
        results.get("fluid_and_section"), expected_physical
    ) or not _strict_equal(campaign.get("fluid_and_section"), expected_physical):
        raise ValueError(
            "QBlade producer fluid/section values differ from the frozen request."
        )
    runtime = _mapping(results.get("runtime"), "QBlade runtime")
    if not _strict_equal(runtime, campaign.get("runtime")):
        raise ValueError("QBlade result and campaign runtime identities differ.")
    library = _mapping(runtime.get("library"), "QBlade runtime library")
    if (
        runtime.get("version") != _QBLADE_VERSION
        or library.get("sha256") != _QBLADE_LIBRARY_SHA256
        or runtime.get("runtime_dependency_fingerprint")
        != _QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT
    ):
        raise ValueError("QBlade 2.0.9.7 runtime identity differs.")
    _selected_semantics(
        results.get("geometry_and_frames"),
        {
            "profile": "NACA0018",
            "blade_count": 1,
            "station_count": 21,
            "chord_axis_global": [0.0, 1.0, 0.0],
            "normal_axis_global": [-1.0, 0.0, 0.0],
            "ambient_wind_global": [1.0, 0.0, 0.0],
            "qblade_internal_incidence_deg": 90.0,
            "vorlap_relative_wind_incidence_deg": -90.0,
            "mapping": (
                "QBlade +90 uses the opposite relative-wind sign convention from "
                "VorLap atan2(U.normal,U.chord)=-90; pitch and twist remain zero"
            ),
            "hub_y_equals_global_y": "zero_yaw_zero_shaft_tilt_zero_global_rotation",
        },
        "QBlade geometry/frame contract",
    )
    if not _strict_equal(
        results.get("geometry_and_frames"), campaign.get("geometry_and_frames")
    ):
        raise ValueError("QBlade result and campaign geometry/frame contracts differ.")
    native_aero = {
        "enabled": True,
        "dynamic_stall_type": 1,
        "unsteady_noncirculatory_aero": False,
        "two_point_lift_drag_correction": True,
        "wake_type": "unsteady_BEM",
        "bem_tip_loss": False,
        "tower_shadow": False,
        "rotation_rpm": 0.0,
        "internal_storage_filter": None,
        "export_filter": "results.filter_exact_10_channels",
        "filter_rationale": (
            "QBlade 2.0.9.7 silently omits the requested panel AoA/Re "
            "diagnostics when FILTERFILE is active during simulation; retain "
            "native channels internally and filter only at exportResults"
        ),
    }
    external = {
        "controller_type": 0,
        "external_library_entries": 0,
        "sil_vorlap": False,
        "sil_constant_force": False,
        "expected_native_callback_updates": 0,
    }
    if not _strict_equal(results.get("native_aero"), native_aero) or not _strict_equal(
        campaign.get("native_aero"), native_aero
    ):
        raise ValueError("QBlade native-aerodynamics contract differs.")
    if not _strict_equal(results.get("external_forces"), external) or not _strict_equal(
        campaign.get("external_forces"), external
    ):
        raise ValueError("QBlade external-force exclusion differs.")
    qblade_history = {
        "first_row": "dt",
        "last_row": "(13*samples_per_cycle+1)*dt",
        "wrapper_steps_rule": "13*samples_per_cycle+2",
        "analytic_motion_rule": "q=A*sin(2*pi*f*t); qdot=2*pi*f*A*cos(2*pi*f*t)",
        "columns": list(_CSV_COLUMNS),
        "float_encoding": ".17g",
        "force_channel": "Aerodynamic Force in Hub Y_h Direction [N]",
        "force_frame": "hub_Y_equals_global_crossflow_Y",
        "force_sign": "positive_fluid_on_body_in_positive_global_Y",
        "force_units": "N_total_over_4m_span",
    }
    if not _strict_equal(
        results.get("history_contract"), qblade_history
    ) or not _strict_equal(campaign.get("history_contract"), qblade_history):
        raise ValueError("QBlade canonical-history/force contract differs.")
    _selected_semantics(
        campaign.get("execution"),
        {
            "threads_per_case": 1,
            "total_case_count": 48,
            "preflight_cpu_slots": 1,
            "matrix_cpu_slots": 47,
            "local_docker_max_cpu_slots": 8,
        },
        "QBlade staged execution contract",
    )
    preflight = _mapping(results.get("preflight_gate"), "QBlade preflight gate")
    campaign_preflight = _mapping(
        campaign.get("preflight_gate"), "QBlade campaign preflight gate"
    )
    if (
        preflight.get("campaign_contract") != campaign_preflight
        or preflight.get("status") != "complete"
        or preflight.get("case_index") != 35
        or preflight.get("case_id") != "re140000_k0p45_a0p0500"
        or preflight.get("scheduler_schema") != _QBLADE_PREFLIGHT_SCHEMA
        or preflight.get("scheduler_stage") != "preflight"
        or preflight.get("scheduler_status") != "complete"
        or campaign_preflight.get("required_before_matrix") is not True
    ):
        raise ValueError("QBlade executed one-case preflight gate differs.")
    scheduler_gate = _mapping(
        scheduler.get("preflight_gate"), "QBlade scheduler preflight gate"
    )
    allocation = _mapping(scheduler.get("allocation"), "QBlade preflight allocation")
    execution_environment = _mapping(
        scheduler.get("execution_environment"),
        "QBlade preflight execution environment",
    )
    environment_kind = execution_environment.get("kind")
    if environment_kind == "flight_slurm":
        job_id = execution_environment.get("slurm_job_id")
        environment_valid = (
            execution_environment.get("platform") == "linux_x86_64"
            and isinstance(job_id, str)
            and job_id.isdigit()
        )
    elif environment_kind == "local_docker":
        image_id = execution_environment.get("container_image_id")
        cpu_limit = execution_environment.get("container_cpu_limit")
        environment_valid = (
            execution_environment.get("platform") == "linux_amd64"
            and isinstance(image_id, str)
            and len(image_id) == 71
            and image_id.startswith("sha256:")
            and all(character in "0123456789abcdef" for character in image_id[7:])
            and type(cpu_limit) is int
            and 1 <= cpu_limit <= 8
        )
    else:
        environment_valid = False
    scheduler_cases = scheduler.get("cases")
    completed_preflight = (
        [
            row
            for row in scheduler_cases
            if isinstance(row, Mapping)
            and row.get("index") == 35
            and row.get("case_id") == "re140000_k0p45_a0p0500"
            and row.get("status") == "complete"
        ]
        if isinstance(scheduler_cases, list)
        else []
    )
    scheduler_identity = _mapping(
        preflight.get("scheduler_report"), "QBlade result scheduler identity"
    )
    if (
        scheduler.get("schema") != _QBLADE_PREFLIGHT_SCHEMA
        or scheduler.get("stage") != "preflight"
        or scheduler.get("status") != "complete"
        or scheduler.get("campaign_fingerprint") != fingerprint
        or scheduler_gate.get("status") != "complete"
        or scheduler_gate.get("case_index") != 35
        or scheduler_gate.get("case_id") != "re140000_k0p45_a0p0500"
        or allocation.get("workers") != 1
        or allocation.get("threads_per_case") != 1
        or allocation.get("allocated_physical_cpus") != 1
        or preflight.get("execution_environment") != execution_environment
        or not environment_valid
        or len(completed_preflight) != 1
        or scheduler_identity.get("sha256") != scheduler_sha
    ):
        raise ValueError(
            "QBlade preflight scheduler evidence is incomplete or not cross-linked."
        )
    summary = _mapping(results.get("summary"), "QBlade result summary")
    if not _strict_equal(
        dict(summary), {"required_cases": 48, "complete_cases": 48, "failed_cases": 0}
    ):
        raise ValueError("QBlade result summary is not a clean 48-case result.")

    result_cases = _case_index(results, "QBlade results")
    campaign_cases = _case_index(campaign, "QBlade campaign")
    _validate_case_crosslinks(
        identification_cases=identification_cases,
        producer_results=result_cases,
        producer_campaign=campaign_cases,
        history_role="exact_runtime_base_history",
        label="QBlade",
    )
    for case_id, row in campaign_cases.items():
        expected = _expected_motion_values(
            reynolds=float(row["reynolds_number"]),
            k=float(row["reduced_frequency_k"]),
            amplitude_ratio=float(row["amplitude_ratio"]),
        )
        if (
            row.get("qblade_internal_incidence_deg") != 90.0
            or row.get("retained_cycles") != 8
            or row.get("expected_result_rows") != expected["rows"]
            or row.get("wrapper_steps") != expected["rows"] + 1
        ):
            raise ValueError(f"QBlade {case_id} derived motion grid differs.")
        for key in ("relative_history_start_s", "relative_history_end_s"):
            _same_number(row.get(key), float(expected[key]), f"QBlade {case_id}.{key}")
    audit_rows = provenance.get("case_audits")
    if not isinstance(audit_rows, list) or len(audit_rows) != 48:
        raise ValueError("QBlade provenance must snapshot all 48 case audits.")
    audit_shas: dict[str, str] = {}
    for row in audit_rows:
        if not isinstance(row, Mapping) or set(row) != {"case_id", "file"}:
            raise ValueError(
                "QBlade audit provenance rows must have exact case_id/file keys."
            )
        case_id = row.get("case_id")
        if (
            not isinstance(case_id, str)
            or case_id not in result_cases
            or case_id in audit_shas
        ):
            raise ValueError("QBlade audit provenance case IDs differ.")
        audit, digest = _verified_json_file(
            root, row["file"], f"QBlade {case_id} case-audit snapshot"
        )
        _selected_semantics(
            audit,
            {
                "schema": _QBLADE_CASE_AUDIT_SCHEMA,
                "status": "complete",
                "case_id": case_id,
                "campaign_fingerprint": fingerprint,
            },
            f"QBlade {case_id} case audit",
        )
        result_case = result_cases[case_id]
        expected = _expected_motion_values(
            reynolds=float(result_case["reynolds_number"]),
            k=float(result_case["reduced_frequency_k"]),
            amplitude_ratio=float(result_case["amplitude_ratio"]),
        )
        audit_history = _mapping(audit.get("canonical_history"), "QBlade audit history")
        result_history = _mapping(
            result_case.get("canonical_history"), "QBlade result history"
        )
        result_audit_identity = _mapping(
            result_case.get("case_audit"), "QBlade result case-audit identity"
        )
        if (
            not _strict_equal(dict(audit_history), dict(result_history))
            or result_audit_identity.get("sha256") != digest
        ):
            raise ValueError(
                f"QBlade {case_id} audit/history identities are not cross-linked."
            )
        _selected_semantics(
            audit.get("runtime"),
            {
                "qblade_library_sha256": _QBLADE_LIBRARY_SHA256,
                "qblade_runtime_dependency_fingerprint": _QBLADE_RUNTIME_DEPENDENCY_FINGERPRINT,
                "threads": 1,
                "coupling_mode": "none",
                "external_callback_updates": 0,
            },
            f"QBlade {case_id} audit runtime",
        )
        _selected_semantics(
            audit.get("channel_contract"),
            {
                "raw_export_rows": int(expected["rows"]) + 1,
                "discarded_wrapper_t0_rows": 1,
                "prescribed_motion": "Y_g Prescr. Platform Trans. [m]",
                "rigid_blade_position": "Y_g Pos. BLD_1 pos 1.000 [m]",
                "crossflow_force": "Aerodynamic Force in Hub Y_h Direction [N]",
                "crossflow_force_units": "N_total_over_4m_span",
                "crossflow_force_sign": "positive_fluid_on_body_positive_global_Y",
                "qblade_aoa": "Angle of Attack at 0.25c BLD_1 PAN_10 [deg]",
                "qblade_reynolds": "Reynolds Number BLD_1 PAN_10 [-]",
            },
            f"QBlade {case_id} channel contract",
        )
        frame = _selected_semantics(
            audit.get("geometry_frame_audit"),
            {
                "qblade_internal_incidence_target_deg": 90.0,
                "vorlap_incidence_deg": -90.0,
            },
            f"QBlade {case_id} frame audit",
        )
        median_aoa = _finite_axis_value(
            frame.get("median_abs_qblade_aoa_deg"), "QBlade median absolute AoA"
        )
        max_aoa_error = _finite_axis_value(
            frame.get("max_abs_qblade_aoa_deviation_from_90_deg"),
            "QBlade maximum AoA deviation",
        )
        median_re = _finite_positive(
            frame.get("median_qblade_panel_reynolds"), "QBlade median panel Reynolds"
        )
        motion_error = _finite_axis_value(
            frame.get("analytic_platform_max_abs_error_m"), "QBlade motion error"
        )
        tip_error = _finite_axis_value(
            frame.get("tip_follows_platform_max_abs_error_m"), "QBlade rigid-tip error"
        )
        if (
            abs(median_aoa - 90.0) > 2.0
            or max_aoa_error < 0.0
            or max_aoa_error > 12.0
            or abs(median_re - float(result_case["reynolds_number"]))
            / float(result_case["reynolds_number"])
            > 0.03
            or motion_error < 0.0
            or motion_error > max(2.0e-9, 2.0e-5 * float(expected["amplitude_m"]))
            or tip_error < 0.0
            or tip_error > 2.0e-6
        ):
            raise ValueError(f"QBlade {case_id} motion/incidence/Re audit failed.")
        statistics = _mapping(
            audit.get("force_statistics_n"), f"QBlade {case_id} force statistics"
        )
        if set(statistics) != {"minimum", "maximum", "mean", "rms"}:
            raise ValueError(f"QBlade {case_id} force-statistics keys differ.")
        numeric_statistics = {
            key: _finite_axis_value(value, f"QBlade {case_id} force {key}")
            for key, value in statistics.items()
        }
        if (
            numeric_statistics["minimum"] > numeric_statistics["maximum"]
            or numeric_statistics["rms"] < 0.0
        ):
            raise ValueError(f"QBlade {case_id} force statistics are inconsistent.")
        audit_shas[case_id] = digest
    if set(audit_shas) != set(result_cases):
        raise ValueError("QBlade case-audit provenance does not cover the full matrix.")
    preflight_audit_identity = _mapping(
        preflight.get("case_audit"), "QBlade result preflight case-audit identity"
    )
    if preflight_audit_identity.get("sha256") != audit_shas.get(
        "re140000_k0p45_a0p0500"
    ):
        raise ValueError(
            "QBlade preflight result is not bound to its case-audit snapshot."
        )
    return {
        "semantic_sha256": _semantic_sha256(provenance),
        "results_manifest_sha256": results_sha,
        "campaign_manifest_sha256": campaign_sha,
        "preflight_scheduler_report_sha256": scheduler_sha,
        "case_audit_bundle_sha256": _semantic_sha256(audit_shas),
    }


def _validate_producer_provenance(
    root: Path,
    value: Any,
    *,
    request_sha: str,
    cases: Sequence[Any],
) -> dict[str, Any]:
    provenance = _mapping(value, "producer_provenance")
    if set(provenance) != {"high_fidelity", "exact_runtime_base"}:
        raise ValueError("producer_provenance must have exact high/base keys.")
    identification_cases: dict[str, Mapping[str, Any]] = {}
    for row in cases:
        case = _mapping(row, "identification case")
        case_id = case.get("case_id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in identification_cases
        ):
            raise ValueError("Identification case IDs must be unique strings.")
        identification_cases[case_id] = case
    if len(identification_cases) != 48:
        raise ValueError(
            "Producer provenance requires exactly 48 identification cases."
        )
    high = _mapping(provenance["high_fidelity"], "high_fidelity provenance")
    base = _mapping(provenance["exact_runtime_base"], "exact_runtime_base provenance")
    return {
        "high_fidelity": _validate_nalu_provenance(
            root,
            high,
            request_sha=request_sha,
            identification_cases=identification_cases,
        ),
        "exact_runtime_base": _validate_qblade_provenance(
            root,
            base,
            request_sha=request_sha,
            identification_cases=identification_cases,
        ),
    }


def build_work_map_from_manifest(
    manifest_path: Path,
    *,
    output_map_path: Path,
    output_report_path: Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    raw_manifest = manifest_path.read_bytes()
    manifest = _json_no_duplicates(raw_manifest, "identification manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema",
        "status",
        "profile",
        "request",
        "static_spectrum",
        "producer_provenance",
        "fluid_and_section",
        "cases",
    }:
        raise ValueError(
            "Identification manifest top-level keys differ from the schema."
        )
    if manifest["schema"] != SCHEMA or manifest["status"] != "complete":
        raise ValueError("Identification manifest must be complete schema v2.")
    if manifest["profile"] != "NACA0018":
        raise ValueError("The first work-map campaign is frozen to NACA0018.")
    root = manifest_path.parent
    request_path, request_raw, request_sha = _verified_file(
        root, manifest["request"], "request"
    )
    request = _load_verified_tracked_request(request_raw)
    spectrum_path, _, spectrum_sha = _verified_file(
        root, manifest["static_spectrum"], "static_spectrum"
    )
    if spectrum_sha != TRACKED_STATIONARY_SPECTRUM_SHA256:
        raise ValueError(
            "Static spectrum is not the tracked NACA0018.h5 source required by "
            "the frozen work-map campaign."
        )

    physical = manifest["fluid_and_section"]
    if not isinstance(physical, Mapping) or set(physical) != {
        "fluid_density_kgpm3",
        "fluid_dynamic_viscosity_pas",
        "reference_chord_m",
        "section_span_m",
        "static_spectrum_frequency_depth",
    }:
        raise ValueError("fluid_and_section keys differ from the schema.")
    density = _finite_positive(physical["fluid_density_kgpm3"], "fluid density")
    viscosity = _finite_positive(
        physical["fluid_dynamic_viscosity_pas"], "dynamic viscosity"
    )
    chord = _finite_positive(physical["reference_chord_m"], "reference chord")
    span = _finite_positive(physical["section_span_m"], "section span")
    frozen_physical = request["frozen_fluid_and_section"]
    declared_physical = {
        "fluid_density_kgpm3": density,
        "fluid_dynamic_viscosity_pas": viscosity,
        "reference_chord_m": chord,
        "section_span_m": span,
    }
    expected_physical = {
        name: float(frozen_physical[name]) for name in declared_physical
    }
    if declared_physical != expected_physical:
        raise ValueError(
            "Manifest fluid/section values must exactly match the tracked frozen request."
        )
    depth = physical["static_spectrum_frequency_depth"]
    if type(depth) is not int or depth <= 1:
        raise ValueError(
            "static_spectrum_frequency_depth must be an integer above one."
        )

    frozen_axes = request["frozen_axes"]
    if not isinstance(frozen_axes, Mapping):
        raise ValueError("The data request does not contain frozen_axes.")
    axes = {
        name: tuple(float(value) for value in frozen_axes[name])
        for name in (
            "incidence_deg",
            "reynolds_number",
            "reduced_frequency_k",
            "amplitude_ratio",
            "direction_deg_mod_180",
        )
    }
    nonzero_amplitudes = tuple(
        value for value in axes["amplitude_ratio"] if value > 0.0
    )
    expected_vertices = set(
        itertools.product(
            axes["incidence_deg"],
            axes["reynolds_number"],
            axes["reduced_frequency_k"],
            nonzero_amplitudes,
            axes["direction_deg_mod_180"],
        )
    )
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != len(expected_vertices):
        raise ValueError(
            "Manifest cases do not cover the frozen nonzero-amplitude tensor."
        )
    producer_digests = _validate_producer_provenance(
        root,
        manifest["producer_provenance"],
        request_sha=request_sha,
        cases=cases,
    )

    airfoil = load_airfoil_fft(str(spectrum_path))
    records: list[dict[str, float]] = []
    report_cases = []
    observed_vertices = set()
    observed_case_ids = set()
    normalized_cases = []
    reference_by_incidence_re: dict[tuple[float, float], float] = {}
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != _CASE_KEYS:
            raise ValueError("Every identification case must have exact schema keys.")
        if (
            type(case["case_id"]) is not str
            or case["case_id"] != case["case_id"].strip()
        ):
            raise ValueError(
                "case_id must be a canonical string without surrounding whitespace."
            )
        case_id = case["case_id"]
        if not case_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in case_id
        ):
            raise ValueError(
                "case_id must use lowercase ASCII letters, digits, '_' or '-'."
            )
        if case_id in observed_case_ids:
            raise ValueError("Every identification case_id must be unique.")
        observed_case_ids.add(case_id)
        vertex = tuple(
            _finite_axis_value(case[name], f"{case_id}.{name}")
            for name in (
                "incidence_deg",
                "reynolds_number",
                "reduced_frequency_k",
                "amplitude_ratio",
                "direction_deg_mod_180",
            )
        )
        if vertex not in expected_vertices or vertex in observed_vertices:
            raise ValueError(
                "Identification case vertex is duplicate or outside the frozen grid."
            )
        observed_vertices.add(vertex)
        incidence, reynolds, reduced_frequency, amplitude_ratio, direction = vertex
        speed = reynolds * viscosity / (density * chord)
        frequency = reduced_frequency * speed / (math.pi * chord)
        amplitude = amplitude_ratio * chord
        discard = case["discard_cycles"]
        minimum_history = request["minimum_accepted_history"]
        if type(discard) is not int or discard < minimum_history["discard_cycles"]:
            raise ValueError(
                "discard_cycles must be an integer at or above the tracked minimum."
            )
        hf_path, hf_raw, hf_sha = _verified_file(
            root, case["high_fidelity_history"], f"{case_id}.high_fidelity_history"
        )
        base_path, base_raw, base_sha = _verified_file(
            root,
            case["exact_runtime_base_history"],
            f"{case_id}.exact_runtime_base_history",
        )
        high = _read_history(hf_raw, f"{case_id} high-fidelity history")
        base = _read_history(base_raw, f"{case_id} runtime-base history")
        for name in ("time_s", "displacement_m", "velocity_mps"):
            if not np.array_equal(high[name], base[name]):
                raise ValueError(
                    f"{case_id} paired {name} histories are not exactly identical."
                )
        _motion_identity_check(high, frequency_hz=frequency, amplitude_m=amplitude)
        key = (incidence, reynolds)
        if key not in reference_by_incidence_re:
            st, spectrum_amplitude, _ = interpolate_fft_spectrum(
                airfoil,
                reynolds,
                incidence,
                "CL",
                n_freq_depth=depth,
                bounds_policy="raise",
            )
            st = np.asarray(st, dtype=float)
            spectrum_amplitude = np.asarray(spectrum_amplitude, dtype=float)
            active = (st > 0.0) & (spectrum_amplitude > 0.0)
            reference = float(math.sqrt(float(np.sum(spectrum_amplitude[active] ** 2))))
            if not math.isfinite(reference) or reference <= 0.0:
                raise ValueError(
                    "Static spectrum has no positive oscillatory CL energy."
                )
            reference_by_incidence_re[key] = reference
        reference = reference_by_incidence_re[key]
        identified = identify_prescribed_motion_work(
            time_s=high["time_s"],
            displacement_m=high["displacement_m"],
            velocity_mps=high["velocity_mps"],
            high_fidelity_force_n=high["crossflow_force_n"],
            runtime_base_force_n=base["crossflow_force_n"],
            motion_frequency_hz=frequency,
            fluid_density_kgpm3=density,
            speed_mps=speed,
            chord_m=chord,
            span_m=span,
            discard_cycles=discard,
            minimum_complete_cycles_after_discard=minimum_history[
                "complete_cycles_after_discard"
            ],
            minimum_samples_per_cycle=minimum_history["samples_per_cycle"],
        )
        incremental_harmonic = math.hypot(
            identified.incremental_force_velocity_harmonic_n,
            identified.incremental_force_displacement_harmonic_n,
        )
        force_reference = 0.5 * density * speed**2 * chord * span * reference
        entrainment = min(1.0, incremental_harmonic / force_reference)
        records.append(
            {
                "incidence_deg": incidence,
                "reynolds_number": reynolds,
                "reduced_frequency_k": reduced_frequency,
                "amplitude_ratio": amplitude_ratio,
                "direction_deg_mod_180": direction,
                "entrainment_fraction": entrainment,
                "reference_available_force_coefficient": reference,
                "signed_power_coefficient": identified.signed_power_coefficient,
                "power_coefficient_standard_uncertainty": identified.power_coefficient_standard_uncertainty,
            }
        )
        normalized_cases.append(
            {
                "case_id": case_id,
                "high_fidelity_history": {
                    "path": str(case["high_fidelity_history"]["path"]),
                    "sha256": hf_sha,
                },
                "exact_runtime_base_history": {
                    "path": str(case["exact_runtime_base_history"]["path"]),
                    "sha256": base_sha,
                },
            }
        )
        report_cases.append(
            {
                "case_id": case_id,
                "vertex": list(vertex),
                "speed_mps": speed,
                "motion_frequency_hz": frequency,
                "motion_amplitude_m": identified.motion_amplitude_m,
                "included_cycle_count": identified.included_cycle_count,
                "minimum_samples_per_included_cycle": identified.minimum_samples_per_included_cycle,
                "mean_incremental_power_w": identified.mean_incremental_power_w,
                "standard_uncertainty_power_w": identified.standard_uncertainty_power_w,
                "signed_power_coefficient": identified.signed_power_coefficient,
                "power_coefficient_standard_uncertainty": identified.power_coefficient_standard_uncertainty,
                "phase_force_relative_to_velocity_rad": identified.phase_force_relative_to_velocity_rad,
                "entrainment_fraction": entrainment,
                "reference_available_force_coefficient": reference,
                "high_fidelity_sha256": hf_sha,
                "exact_runtime_base_sha256": base_sha,
            }
        )
    if observed_vertices != expected_vertices:
        raise ValueError(
            "Manifest cases do not exactly match the frozen tensor vertices."
        )

    for incidence, reynolds, reduced_frequency, direction in itertools.product(
        axes["incidence_deg"],
        axes["reynolds_number"],
        axes["reduced_frequency_k"],
        axes["direction_deg_mod_180"],
    ):
        records.append(
            {
                "incidence_deg": incidence,
                "reynolds_number": reynolds,
                "reduced_frequency_k": reduced_frequency,
                "amplitude_ratio": 0.0,
                "direction_deg_mod_180": direction,
                "entrainment_fraction": 0.0,
                "reference_available_force_coefficient": reference_by_incidence_re[
                    (incidence, reynolds)
                ],
                "signed_power_coefficient": 0.0,
                "power_coefficient_standard_uncertainty": 0.0,
            }
        )

    normalized_cases.sort(key=lambda item: item["case_id"])
    hf_bundle_sha = _semantic_sha256(
        [
            {"case_id": item["case_id"], **item["high_fidelity_history"]}
            for item in normalized_cases
        ]
    )
    base_bundle_sha = _semantic_sha256(
        [
            {"case_id": item["case_id"], **item["exact_runtime_base_history"]}
            for item in normalized_cases
        ]
    )
    sources = (
        {
            "role": "prescribed_motion_high_fidelity",
            "label": "paired NACA0018 prescribed-motion history bundle",
            "sha256": hf_bundle_sha,
        },
        {
            "role": "exact_runtime_base",
            "label": "paired native-QBlade prescribed-motion history bundle",
            "sha256": base_bundle_sha,
        },
        {
            "role": "prescribed_motion_high_fidelity_producer_provenance",
            "label": "Nalu-Wind prescribed-motion producer evidence",
            "sha256": producer_digests["high_fidelity"]["semantic_sha256"],
        },
        {
            "role": "exact_runtime_base_producer_provenance",
            "label": "native-QBlade prescribed-motion producer evidence",
            "sha256": producer_digests["exact_runtime_base"]["semantic_sha256"],
        },
        {
            "role": "stationary_force_spectrum",
            "label": str(manifest["static_spectrum"]["path"]),
            "sha256": spectrum_sha,
        },
        {
            "role": "frozen_data_request",
            "label": str(request_path.name),
            "sha256": request_sha,
        },
    )
    payload = assemble_airfoil_work_map_payload(
        records=records,
        profile="NACA0018",
        sources=sources,
    )
    model = AirfoilForceMotionWorkMap(payload)
    domain = model.positive_work_domain_status(confidence_multiplier=1.96)
    domain["qualified"] = bool(
        domain["has_positive_work"] and domain["highest_amplitude_plane_nonpositive"]
    )
    output_map_path = Path(output_map_path)
    output_report_path = Path(output_report_path)
    output_map_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    map_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    output_map_path.write_bytes(map_bytes)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "qualified" if domain["qualified"] else "identified_but_unbounded",
        "manifest_sha256": _sha256_bytes(raw_manifest),
        "request_sha256": request_sha,
        "static_spectrum_sha256": spectrum_sha,
        "high_fidelity_bundle_sha256": hf_bundle_sha,
        "exact_runtime_base_bundle_sha256": base_bundle_sha,
        "producer_provenance_digests": producer_digests,
        "work_map_sha256": _sha256_bytes(map_bytes),
        "work_map_payload_fingerprint": model.payload_fingerprint,
        "case_count": len(report_cases),
        "origin_vertex_count": len(records) - len(report_cases),
        "quality_contract": {
            "minimum_discard_cycles": request["minimum_accepted_history"][
                "discard_cycles"
            ],
            "minimum_complete_cycles_after_discard": request[
                "minimum_accepted_history"
            ]["complete_cycles_after_discard"],
            "minimum_samples_per_cycle": request["minimum_accepted_history"][
                "samples_per_cycle"
            ],
            "fluid_density_kgpm3": density,
            "fluid_dynamic_viscosity_pas": viscosity,
            "reference_chord_m": chord,
            "section_span_m": span,
        },
        "positive_work_domain": domain,
        "cases": sorted(report_cases, key=lambda item: item["case_id"]),
        "claim_boundary": {
            "static_fft_phase_used": False,
            "ehrmann_free_response_used_for_fit": False,
            "grinderslev_power_used_for_naca_magnitude": False,
            "supplemental_work_is_high_fidelity_minus_exact_runtime_base": True,
        },
    }
    output_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_work_map_from_manifest(
        args.manifest,
        output_map_path=args.output_map,
        output_report_path=args.output_report,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "case_count", "work_map_sha256")},
            indent=2,
        )
    )
    return 0 if report["status"] == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
