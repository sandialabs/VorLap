#!/usr/bin/env python3
"""Prepare and assess the Grinderslev P100--I50 modal-power benchmark.

The preparation path copies the immutable public-model case into independent
working directories at 0.006 s and 0.003 s.  For each time step it creates a
response-aware SIL case and a no-coupling structural baseline.  The response
case uses the digitized forced-motion aerodynamic-power curve as a calibrated
one-mode closure; it does not predict the wake.

The analysis path is deliberately strict.  It requires complete QBlade result
and version-2 SIL histories, replays the configured response model from the
recorded structural velocities, checks every generated force, evaluates the
actually applied one-step-lag work, and applies frozen time-step gates.  A
calibration-curve match is a coupling/energy-bookkeeping comparison, not an
independent CFD verification or experimental validation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import hilbert


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLE_ROOT = Path(__file__).resolve().parent
BASE_CASE_ROOT = EXAMPLE_ROOT / "generated_case"
CASE_NAME = "grinderslev2022_p100_i50_iea10_single_blade"
BASE_SIM_NAME = f"{CASE_NAME}.sim"
CONFIG_RELATIVE = Path(CASE_NAME) / "Control" / "vorlap_qblade_external.json"
STRUCTURE_RELATIVE = Path(CASE_NAME) / "Structure" / "iea10_blade_hawc2.str"
GEOMETRY_RELATIVE = Path("iea10_blade_geometry.csv")
MANIFEST_RELATIVE = Path("benchmark_manifest.json")
POWER_CURVE = EXAMPLE_ROOT / "reference_data" / "p100_i50_figure6_power_digitized.csv"
QBLADE_MODE_CSV = EXAMPLE_ROOT / "reference_data" / "qblade_first_edge_mode.csv"
QBLADE_MODE_IDENTIFICATION = (
    EXAMPLE_ROOT / "reference_data" / "qblade_first_edge_mode_identification.json"
)
RESULT_FILTER = EXAMPLE_ROOT / "response_benchmark_results.filter"

REFERENCE_FREQUENCY_HZ = 0.67
REFERENCE_FSI_AMPLITUDE_M = 1.18
REFERENCE_FSI_POWER_W = 775.0
REFERENCE_FORCED_MOTION_INTERSECTION_M = 1.24
REFERENCE_FORCED_MOTION_INTERSECTION_POWER_W = 880.0
REFERENCE_ZERO_DAMPING_AMPLITUDE_M = 1.63
RESPONSE_MODEL_TYPE = "modal_aerodynamic_power"
RESPONSE_MODEL_SCHEMA_VERSION = 1
POWER_CURVE_SERIES = "p100_i50_forced_motion"
EXPECTED_POWER_CURVE_SHA256 = (
    "016ce5265b77dc0c9e577d7548e453d09f67ef732dda817618f79909546f323d"
)
EXPECTED_QBLADE_MODE_CSV_SHA256 = (
    "0e08e5c98fe423b8c171ad5d3ea503fc37ab2f54447f57e2f7586cabdadcec18"
)
EXPECTED_QBLADE_MODE_IDENTIFICATION_SHA256 = (
    "7a9ef98a21dd83e3e7faa0c8698c3311a475c09077db7fb147607cb8e6f5cf86"
)
DEFAULT_TIMESTEPS_S = (0.006, 0.003)
DEFAULT_DURATION_S = 120.0
DEFAULT_STEADY_WINDOW_S = 20.0
DEFAULT_WOHLER_EXPONENT = 10.0

AMPLITUDE_TIMESTEP_TOLERANCE = 0.01
DEL_TIMESTEP_TOLERANCE = 0.01
RANGE_POWER_TIMESTEP_TOLERANCE = 0.05
POWER_TIMESTEP_TOLERANCE = 0.01
FORCE_REPLAY_RTOL = 2.0e-11
FORCE_REPLAY_ATOL_N = 2.0e-8
WORK_COORDINATE_DIAGNOSTIC_TOLERANCE = 1.0e-3

# Frozen before inspecting an extended-duration result.  These are physical
# stationarity/nominal structural-model gates for each run, not coarse/fine discretization
# gates.  Keep the two qualification layers separate in the output schema.
SETTLING_TIP_HALF_WINDOW_TOLERANCE = 0.01
SETTLING_ENVELOPE_10_CYCLE_TOLERANCE = 0.01
SETTLING_DEL_HALF_WINDOW_TOLERANCE = 0.01
SETTLING_RANGE_POWER_HALF_WINDOW_TOLERANCE = 0.05
SETTLING_POWER_CLOSURE_TOLERANCE = 0.05
SETTLING_AMPLITUDE_COORDINATE_TOLERANCE = 0.05
SETTLING_CYCLES_PER_HALF = 6
SETTLING_PHASE_GRID_INTERVALS_PER_CYCLE = 500

REQUIRED_RESULT_COLUMNS = (
    "Time~[s]",
    "X_b~RootBend.~Mom.~BLD_1~[Nm]",
    "Y_b~Tip~Trl.Def.~BLD_1~[m]",
)
LIMIT_COUNTERS = (
    "target_power_limited_steps",
    "generalized_force_limited_steps",
    "nodal_force_limited_steps",
    "nodal_force_limited_node_samples",
    "max_nodal_force_limited_nodes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _strict_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _strict_bool_text(value: Any, name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{name} must be exactly 'True' or 'False'.")


def _safe_node_token(node_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in node_id).strip("_")


def _string_sequence_sha256(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def node_identity_metadata(quadrature: "BladeQuadrature") -> dict[str, Any]:
    exact = list(quadrature.node_ids)
    canonical = list(quadrature.controller_node_ids)
    mapping = [
        {"exact_swap_node_id": source, "canonical_controller_node_id": target}
        for source, target in zip(exact, canonical)
    ]

    return {
        "schema": "qblade-exact-to-canonical-node-map-v1",
        "node_count": len(exact),
        "canonicalization": (
            "vorlap.qblade_nodes.parse_qblade_node_id(...).canonical_node_id; "
            "normalized position rendered to six decimals"
        ),
        "exact_swap_node_ids": exact,
        "canonical_controller_node_ids": canonical,
        "exact_swap_node_ids_sha256": _string_sequence_sha256(exact),
        "canonical_controller_node_ids_sha256": _string_sequence_sha256(canonical),
        "exact_to_canonical": mapping,
    }


@dataclass(frozen=True)
class QBladeModeIdentification:
    span_fraction: np.ndarray
    public_openfast_mode_shape: np.ndarray
    qblade_mode_shape: np.ndarray
    phase_offset_rad: np.ndarray
    modal_mass_weights_kg: np.ndarray
    natural_frequency_hz: float
    public_openfast_modal_mass_kg: float
    qblade_modal_mass_kg: float
    public_projection_overlap: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BladeQuadrature:
    node_ids: tuple[str, ...]
    controller_node_ids: tuple[str, ...]
    span_fraction: np.ndarray
    span_m: np.ndarray
    tributary_span_m: np.ndarray
    chord_m: np.ndarray
    public_openfast_mode_shape: np.ndarray
    response_qblade_mode_shape: np.ndarray
    mass_density_kgpm: np.ndarray
    modal_mass_weights_kg: np.ndarray
    aerodynamic_area_weights_m2: np.ndarray
    public_openfast_modal_mass_kg: float
    modal_mass_kg: float
    natural_frequency_hz: float
    structural_damping_ratio: float
    structural_power_coefficient_wpm2: float
    mode_identification: QBladeModeIdentification


def load_qblade_mode_identification() -> QBladeModeIdentification:
    if sha256_file(QBLADE_MODE_CSV) != EXPECTED_QBLADE_MODE_CSV_SHA256:
        raise ValueError("Pinned QBlade mode-shape CSV SHA-256 changed.")
    if (
        sha256_file(QBLADE_MODE_IDENTIFICATION)
        != EXPECTED_QBLADE_MODE_IDENTIFICATION_SHA256
    ):
        raise ValueError("Pinned QBlade mode-identification JSON SHA-256 changed.")
    metadata = load_json(QBLADE_MODE_IDENTIFICATION)
    if metadata.get("schema") != "grinderslev2022-qblade-structural-coordinate-v1":
        raise ValueError("Unsupported QBlade mode-identification schema.")
    if metadata.get("mode_csv_sha256") != EXPECTED_QBLADE_MODE_CSV_SHA256:
        raise ValueError("QBlade mode-identification JSON does not bind the mode CSV.")
    required = (
        "span_fraction",
        "public_openfast_mode_tip_normalized",
        "qblade_mode_tip_normalized",
        "qblade_phase_offset_rad",
        "modal_mass_weight_kg",
    )
    columns = {name: [] for name in required}
    with QBLADE_MODE_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != required:
            raise ValueError("QBlade mode CSV header changed.")
        for row_number, row in enumerate(reader, start=2):
            for name in required:
                columns[name].append(_finite_float(row[name], f"mode row {row_number} {name}"))
    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    span = arrays["span_fraction"]
    public_mode = arrays["public_openfast_mode_tip_normalized"]
    qblade_mode = arrays["qblade_mode_tip_normalized"]
    phase = arrays["qblade_phase_offset_rad"]
    mass = arrays["modal_mass_weight_kg"]
    if (
        span.size != 30
        or np.any(np.diff(span) <= 0.0)
        or span[0] != 0.0
        or span[-1] != 1.0
        or public_mode[0] != 0.0
        or public_mode[-1] != 1.0
        or qblade_mode[0] != 0.0
        or qblade_mode[-1] != 1.0
        or np.any(qblade_mode < 0.0)
        or np.any(mass <= 0.0)
        or float(np.max(np.abs(phase))) > 0.03
    ):
        raise ValueError("Pinned QBlade mode table violates its structural invariants.")
    natural_frequency = _finite_float(
        metadata.get("natural_frequency_hz"), "identified natural frequency"
    )
    public_mass = float(np.sum(mass * public_mode**2))
    qblade_mass = float(np.sum(mass * qblade_mode**2))
    overlap = float(np.sum(mass * public_mode * qblade_mode) / public_mass)
    expected_values = {
        "public_openfast_modal_mass_kg": public_mass,
        "qblade_modal_mass_kg": qblade_mass,
        "public_projection_overlap": overlap,
    }
    for name, observed in expected_values.items():
        expected = _finite_float(metadata.get(name), name)
        if not math.isclose(observed, expected, rel_tol=2.0e-14, abs_tol=1.0e-12):
            raise ValueError(f"QBlade mode metadata is inconsistent for {name}.")
    return QBladeModeIdentification(
        span_fraction=span,
        public_openfast_mode_shape=public_mode,
        qblade_mode_shape=qblade_mode,
        phase_offset_rad=phase,
        modal_mass_weights_kg=mass,
        natural_frequency_hz=natural_frequency,
        public_openfast_modal_mass_kg=public_mass,
        qblade_modal_mass_kg=qblade_mass,
        public_projection_overlap=overlap,
        metadata=metadata,
    )


def _read_geometry(path: Path) -> dict[str, np.ndarray]:
    required = (
        "span_fraction",
        "span_m",
        "chord_m",
        "edge_mode_tip_normalized",
    )
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Geometry CSV has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Geometry CSV contains duplicate columns: {path}")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"Geometry CSV is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if len(rows) != 30:
        raise ValueError(f"Expected exactly 30 blade stations, observed {len(rows)}.")
    result: dict[str, np.ndarray] = {}
    for name in required:
        result[name] = np.asarray(
            [_finite_float(row[name], f"geometry row {index + 2} {name}") for index, row in enumerate(rows)],
            dtype=float,
        )
    if np.any(np.diff(result["span_fraction"]) <= 0.0):
        raise ValueError("Geometry span fractions must increase strictly.")
    if np.any(np.diff(result["span_m"]) <= 0.0):
        raise ValueError("Geometry physical span must increase strictly.")
    if not math.isclose(float(result["span_fraction"][0]), 0.0, abs_tol=1.0e-14):
        raise ValueError("Geometry span fractions must start at zero.")
    if not math.isclose(float(result["span_fraction"][-1]), 1.0, abs_tol=1.0e-14):
        raise ValueError("Geometry span fractions must end at one.")
    if np.any(result["chord_m"] <= 0.0):
        raise ValueError("Geometry chord must remain positive.")
    mode = result["edge_mode_tip_normalized"]
    if not math.isclose(float(mode[-1]), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The edgewise mode must be normalized to one at the tip.")
    if not math.isclose(float(mode[0]), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("The clamped edgewise mode must be zero at the root.")
    return result


def _read_qblade_structure(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"QBlade structure file is empty: {path}")
    first = lines[0].split()
    if len(first) < 2 or first[1] != "RAYLEIGHDMP":
        raise ValueError(f"QBlade structure file lacks a leading RAYLEIGHDMP value: {path}")
    rayleigh_beta_s = _finite_float(first[0], "RAYLEIGHDMP")
    if rayleigh_beta_s <= 0.0:
        raise ValueError("RAYLEIGHDMP must be positive for this benchmark.")

    disc_count = None
    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[1] == "DISC":
            disc_count = _strict_integer(fields[0], "structural DISC")
            break
    if disc_count is None or disc_count < 2:
        raise ValueError(f"QBlade structure file has no valid DISC count: {path}")
    header_index = next(
        (index for index, line in enumerate(lines) if line.split()[:1] == ["LENFRACT_[-]"]),
        None,
    )
    if header_index is None:
        raise ValueError(f"QBlade structure file has no sectional header: {path}")
    header = lines[header_index].split()
    if "MASSD_[kg/m]" not in header:
        raise ValueError(f"QBlade structure file has no MASSD_[kg/m] column: {path}")
    mass_index = header.index("MASSD_[kg/m]")
    data_lines = lines[header_index + 1 : header_index + 1 + disc_count]
    if len(data_lines) != disc_count:
        raise ValueError("QBlade structure file ends before all declared sections.")
    rows = []
    for row_index, line in enumerate(data_lines, start=1):
        fields = line.split()
        if len(fields) != len(header):
            raise ValueError(
                f"Structural row {row_index} has {len(fields)} fields; expected {len(header)}."
            )
        rows.append([_finite_float(value, f"structural row {row_index}") for value in fields])
    table = np.asarray(rows, dtype=float)
    fraction = table[:, 0]
    mass_density = table[:, mass_index]
    if np.any(np.diff(fraction) <= 0.0) or fraction[0] != 0.0 or fraction[-1] != 1.0:
        raise ValueError("Structural fractions must run strictly from zero to one.")
    if np.any(mass_density <= 0.0):
        raise ValueError("Structural mass density must remain positive.")
    return fraction, mass_density, rayleigh_beta_s


def _tributary_lengths(span_m: np.ndarray) -> np.ndarray:
    span = np.asarray(span_m, dtype=float)
    if span.ndim != 1 or span.size < 2 or np.any(~np.isfinite(span)):
        raise ValueError("span_m must be a finite one-dimensional grid.")
    if np.any(np.diff(span) <= 0.0):
        raise ValueError("span_m must increase strictly.")
    weights = np.empty_like(span)
    weights[0] = 0.5 * (span[1] - span[0])
    weights[-1] = 0.5 * (span[-1] - span[-2])
    weights[1:-1] = 0.5 * (span[2:] - span[:-2])
    if np.any(weights <= 0.0):  # pragma: no cover - increasing span guarantees this
        raise ValueError("Physical tributary lengths must be positive.")
    return weights


def derive_blade_quadrature(case_root: Path = BASE_CASE_ROOT) -> BladeQuadrature:
    from vorlap.qblade_nodes import parse_qblade_node_id

    root = Path(case_root)
    geometry = _read_geometry(root / GEOMETRY_RELATIVE)
    structural_fraction, structural_mass, rayleigh_beta_s = _read_qblade_structure(
        root / STRUCTURE_RELATIVE
    )
    span_fraction = geometry["span_fraction"]
    span_m = geometry["span_m"]
    chord_m = geometry["chord_m"]
    public_mode_shape = geometry["edge_mode_tip_normalized"]
    mode_identification = load_qblade_mode_identification()
    tributary = _tributary_lengths(span_m)
    mass_density = np.interp(span_fraction, structural_fraction, structural_mass)
    modal_mass_weights = mass_density * tributary
    aerodynamic_weights = chord_m * tributary
    if (
        not np.array_equal(span_fraction, mode_identification.span_fraction)
        or not np.array_equal(
            public_mode_shape, mode_identification.public_openfast_mode_shape
        )
        or not np.allclose(
            modal_mass_weights, mode_identification.modal_mass_weights_kg,
            rtol=8.0 * np.finfo(float).eps, atol=0.0
        )
    ):
        raise ValueError(
            "Pinned QBlade structural coordinate no longer matches the public geometry "
            "and converted-structure mass quadrature."
        )
    # Interpolation can differ by one ULP between NumPy/compiler platforms.
    # Once the independent quadrature agrees to roundoff, retain the hash-bound
    # identified coordinate for reproducible response coefficients and models.
    modal_mass_weights = mode_identification.modal_mass_weights_kg.copy()
    response_mode_shape = mode_identification.qblade_mode_shape
    public_modal_mass = float(np.sum(modal_mass_weights * public_mode_shape**2))
    modal_mass = float(np.sum(modal_mass_weights * response_mode_shape**2))
    natural_frequency = mode_identification.natural_frequency_hz
    damping_ratio = rayleigh_beta_s * math.pi * natural_frequency
    omega = 2.0 * math.pi * natural_frequency
    structural_power_coefficient = damping_ratio * modal_mass * omega**3
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (
            public_modal_mass,
            modal_mass,
            natural_frequency,
            damping_ratio,
            structural_power_coefficient,
        )
    ):
        raise ValueError("Derived modal mass and damping quantities must be finite and positive.")

    node_ids = tuple(f"BLD_1_{value:.10f}" for value in span_fraction)
    base_config = load_json(root / CONFIG_RELATIVE)
    configured_ids = tuple(str(value) for value in base_config.get("node_ids", ()))
    if configured_ids != node_ids:
        raise ValueError("The 30 generated node IDs do not exactly match the geometry grid.")
    controller_node_ids = tuple(
        parse_qblade_node_id(node_id).canonical_node_id for node_id in node_ids
    )
    if len(set(controller_node_ids)) != len(controller_node_ids):
        raise ValueError(
            "Exact QBlade node IDs collide after canonical six-decimal controller rendering."
        )
    return BladeQuadrature(
        node_ids=node_ids,
        controller_node_ids=controller_node_ids,
        span_fraction=span_fraction,
        span_m=span_m,
        tributary_span_m=tributary,
        chord_m=chord_m,
        public_openfast_mode_shape=public_mode_shape,
        response_qblade_mode_shape=response_mode_shape,
        mass_density_kgpm=mass_density,
        modal_mass_weights_kg=modal_mass_weights,
        aerodynamic_area_weights_m2=aerodynamic_weights,
        public_openfast_modal_mass_kg=public_modal_mass,
        modal_mass_kg=modal_mass,
        natural_frequency_hz=natural_frequency,
        structural_damping_ratio=damping_ratio,
        structural_power_coefficient_wpm2=structural_power_coefficient,
        mode_identification=mode_identification,
    )


def load_power_curve() -> tuple[list[float], list[float], str]:
    from vorlap.qblade_modal_power_model import TabulatedAerodynamicPowerCurve

    observed_hash = sha256_file(POWER_CURVE)
    if observed_hash != EXPECTED_POWER_CURVE_SHA256:
        raise ValueError(
            "The Grinderslev digitized power curve hash changed; review the dataset before use."
        )
    curve = TabulatedAerodynamicPowerCurve.from_csv(
        POWER_CURVE,
        expected_series=POWER_CURVE_SERIES,
    )
    if curve.source_sha256 != EXPECTED_POWER_CURVE_SHA256:
        raise ValueError("Power-curve loader and direct SHA-256 do not agree.")
    return list(curve.amplitude_m), list(curve.power_w), observed_hash


def structural_equilibrium_amplitude_m(
    structural_power_coefficient_wpm2: float,
) -> float:
    from vorlap.qblade_modal_power_model import TabulatedAerodynamicPowerCurve

    curve = TabulatedAerodynamicPowerCurve.from_csv(
        POWER_CURVE,
        expected_series=POWER_CURVE_SERIES,
    )
    roots = curve.structural_equilibria_m(structural_power_coefficient_wpm2)
    if len(roots) != 1:
        raise ValueError(
            "The selected power curve and structural damping must define exactly one nonzero equilibrium."
        )
    return float(roots[0])


def build_response_model_config(quadrature: BladeQuadrature) -> dict[str, Any]:
    amplitude, power, source_hash = load_power_curve()
    return {
        "type": RESPONSE_MODEL_TYPE,
        "schema_version": RESPONSE_MODEL_SCHEMA_VERSION,
        "enabled": True,
        "node_ids": list(quadrature.controller_node_ids),
        "mode_shape": quadrature.response_qblade_mode_shape.tolist(),
        "modal_mass_weights_kg": quadrature.modal_mass_weights_kg.tolist(),
        "aerodynamic_tributary_weights": (
            quadrature.aerodynamic_area_weights_m2.tolist()
        ),
        "natural_frequency_hz": quadrature.natural_frequency_hz,
        "power_curve": {
            "amplitude_m": amplitude,
            "power_w": power,
            "source_label": POWER_CURVE_SERIES,
            "source_sha256": source_hash,
        },
        "rms_time_constant_s": 1.0 / quadrature.natural_frequency_hz,
        "bandpass_damping_ratio": 0.1,
        "force_direction": "chord",
        "max_abs_target_power_w": 2000.0,
        "max_abs_generalized_force_n": 10000.0,
        "max_abs_nodal_force_n": 2000.0,
        "limit_action": "error",
    }


def _patch_qblade_keyword(text: str, keyword: str, value: Any) -> str:
    from vorlap.qblade_files import replace_qblade_keyword_text

    patched, _old, count = replace_qblade_keyword_text(text, keyword, value)
    if count != 1:
        raise ValueError(f"Could not patch exactly one QBlade {keyword} keyword.")
    return patched


def _variant_token(timestep_s: float) -> str:
    formatted = f"{timestep_s:.9f}".rstrip("0").rstrip(".")
    return f"dt_{formatted}".replace(".", "p")


def _duration_token(duration_s: float) -> str:
    formatted = f"{_finite_float(duration_s, 'duration_s'):.9f}".rstrip("0").rstrip(".")
    return f"{formatted}s".replace(".", "p")


def _number_of_steps(duration_s: float, timestep_s: float) -> int:
    duration = _finite_float(duration_s, "duration_s")
    timestep = _finite_float(timestep_s, "timestep_s")
    if duration <= 0.0 or timestep <= 0.0:
        raise ValueError("duration_s and timestep_s must be positive.")
    raw = duration / timestep
    steps = int(round(raw))
    if steps < 2 or not math.isclose(raw, steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("duration_s must be an integer number of timesteps.")
    return steps


def _rebuild_variant_manifest(
    variant_root: Path,
    *,
    base_manifest_sha256: str,
    timestep_s: float,
    duration_s: float,
    role: str,
    quadrature: BladeQuadrature,
    equilibrium_amplitude_m: float,
) -> None:
    manifest_path = variant_root / MANIFEST_RELATIVE
    manifest = load_json(manifest_path)
    manifest["scope"] = (
        "calibration-based one-mode aerodynamic-power coupling demonstration; "
        "not independent wake prediction, CFD verification, physical validation, or certification evidence"
        if role == "response"
        else "matched no-coupling structural baseline for the modal-power demonstration"
    )
    mapping = dict(manifest.get("qblade_mapping", {}))
    mapping["timestep_s"] = float(timestep_s)
    mapping["duration_s"] = float(duration_s)
    manifest["qblade_mapping"] = mapping
    probe = dict(manifest.get("probe_source", {}))
    probe["runtime_force_scale"] = 0.0
    probe["contribution_to_generated_force"] = False
    probe["structural_seed"] = (
        "unforced QBlade initialization transient; no synthetic VorLap probe force is applied"
    )
    manifest["probe_source"] = probe
    manifest["response_aware_variant"] = {
        "schema": "grinderslev2022-modal-power-case-v2",
        "role": role,
        "base_manifest_sha256": base_manifest_sha256,
        "response_model_type": RESPONSE_MODEL_TYPE if role == "response" else "none",
        "calibration_basis": (
            "Grinderslev et al. 2022 Figure 6 P100-I50 forced-motion P(A) digitization"
        ),
        "power_curve_sha256": EXPECTED_POWER_CURVE_SHA256,
        "power_curve_series": POWER_CURVE_SERIES,
        "node_count": len(quadrature.node_ids),
        "node_identity": node_identity_metadata(quadrature),
        "modal_mass_quadrature": (
            "MASSD interpolated from iea10_blade_hawc2.str to each action node and "
            "multiplied by physical nodal tributary span"
        ),
        "response_mode_shape": (
            "unit-tip QBlade first-edgewise shape identified from the hash-pinned "
            "12 s axis-corrected-v2 structural probe; not fitted to P(A) or the "
            "120 s response result"
        ),
        "public_openfast_mode_shape_scope": (
            "retained for public-model provenance and visualization only; not used "
            "by the response-model projection or force distribution"
        ),
        "mode_shape_csv_sha256": EXPECTED_QBLADE_MODE_CSV_SHA256,
        "mode_identification_json_sha256": (
            EXPECTED_QBLADE_MODE_IDENTIFICATION_SHA256
        ),
        "public_projection_overlap": (
            quadrature.mode_identification.public_projection_overlap
        ),
        "aerodynamic_quadrature": "local chord multiplied by physical nodal tributary span",
        "public_openfast_modal_mass_kg": quadrature.public_openfast_modal_mass_kg,
        "modal_mass_kg": quadrature.modal_mass_kg,
        "natural_frequency_hz": quadrature.natural_frequency_hz,
        "structural_damping_ratio": quadrature.structural_damping_ratio,
        "structural_power_relation": "P_struct(A) = zeta * M_modal * omega_n^3 * A^2",
        "structural_power_coefficient_wpm2": (
            quadrature.structural_power_coefficient_wpm2
        ),
        "predicted_forced_motion_curve_intersection_m": equilibrium_amplitude_m,
        "published_fsi_amplitude_m": REFERENCE_FSI_AMPLITUDE_M,
        "published_forced_motion_intersection_m": (
            REFERENCE_FORCED_MOTION_INTERSECTION_M
        ),
        "published_zero_damping_amplitude_m": REFERENCE_ZERO_DAMPING_AMPLITUDE_M,
        "published_comparison_boundary": (
            "The 1.18 m FSI result is separate from the approximately 1.24 m "
            "forced-motion/damping-curve intersection used by this calibration closure."
        ),
    }

    files = []
    for path in sorted(item for item in variant_root.rglob("*") if item.is_file()):
        if (
            path == manifest_path
            or path.name.endswith(".vorlap.bak")
            or path.name == "vorlap_qblade_geometry_audit.csv"
            or path.name == "vorlap_qblade_bridge_cpp.log"
        ):
            continue
        files.append(
            {
                "path": path.relative_to(variant_root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest["files"] = files
    write_json(manifest_path, manifest)


def _prepare_variant(
    target: Path,
    *,
    timestep_s: float,
    duration_s: float,
    role: str,
    quadrature: BladeQuadrature,
    response_config: Mapping[str, Any],
    base_manifest_sha256: str,
    equilibrium_amplitude_m: float,
) -> dict[str, Any]:
    if role not in {"response", "unloaded"}:
        raise ValueError("Variant role must be 'response' or 'unloaded'.")
    shutil.copytree(BASE_CASE_ROOT, target)
    steps = _number_of_steps(duration_s, timestep_s)
    sim_path = target / BASE_SIM_NAME
    sim_text = sim_path.read_text(encoding="utf-8")
    sim_text = _patch_qblade_keyword(sim_text, "TIMESTEP", timestep_s)
    sim_text = _patch_qblade_keyword(sim_text, "NUMTIMESTEPS", steps)
    sim_path.write_text(sim_text, encoding="utf-8")

    config_path = target / CONFIG_RELATIVE
    config = load_json(config_path)
    if tuple(str(value) for value in config.get("node_ids", ())) != quadrature.node_ids:
        raise ValueError("Copied case node IDs changed before response configuration.")
    config["force_scale"] = 0.0
    config["source_frequency_cap_hz"] = 0.5 / timestep_s
    if role == "response":
        config["response_model"] = dict(response_config)
        config["response_model_node_identity"] = node_identity_metadata(quadrature)
    else:
        config.pop("response_model", None)
        config.pop("response_model_node_identity", None)
    write_json(config_path, config)
    _rebuild_variant_manifest(
        target,
        base_manifest_sha256=base_manifest_sha256,
        timestep_s=timestep_s,
        duration_s=duration_s,
        role=role,
        quadrature=quadrature,
        equilibrium_amplitude_m=equilibrium_amplitude_m,
    )
    return {
        "role": role,
        "timestep_s": timestep_s,
        "duration_s": duration_s,
        "requested_steps": steps,
        "sim": sim_path,
        "config": config_path,
        "manifest": target / MANIFEST_RELATIVE,
    }


def prepare_response_cases(
    output_root: Path,
    *,
    duration_s: float = DEFAULT_DURATION_S,
    timesteps_s: Sequence[float] = DEFAULT_TIMESTEPS_S,
) -> Path:
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to replace existing response-benchmark output: {output}"
        )
    if not BASE_CASE_ROOT.is_dir():
        raise FileNotFoundError(BASE_CASE_ROOT)
    if not RESULT_FILTER.is_file():
        raise FileNotFoundError(RESULT_FILTER)
    timestep_values = tuple(_finite_float(value, "timestep_s") for value in timesteps_s)
    if not timestep_values or len(set(timestep_values)) != len(timestep_values):
        raise ValueError("timesteps_s must contain unique values.")
    for timestep in timestep_values:
        _number_of_steps(duration_s, timestep)

    quadrature = derive_blade_quadrature(BASE_CASE_ROOT)
    response_config = build_response_model_config(quadrature)
    equilibrium = structural_equilibrium_amplitude_m(
        quadrature.structural_power_coefficient_wpm2
    )
    base_manifest_sha = sha256_file(BASE_CASE_ROOT / MANIFEST_RELATIVE)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        records: list[dict[str, Any]] = []
        for timestep in timestep_values:
            token = _variant_token(timestep)
            for role in ("response", "unloaded"):
                target = staging / token / role
                record = _prepare_variant(
                    target,
                    timestep_s=timestep,
                    duration_s=duration_s,
                    role=role,
                    quadrature=quadrature,
                    response_config=response_config,
                    base_manifest_sha256=base_manifest_sha,
                    equilibrium_amplitude_m=equilibrium,
                )
                relative_sim = record["sim"].relative_to(staging).as_posix()
                relative_config = record["config"].relative_to(staging).as_posix()
                output_name = (
                    "grinderslev2022_modal_power_qblade_mode_"
                    f"{_duration_token(duration_s)}_{token}_{role}"
                )
                runner_args = [
                    "--sim",
                    relative_sim,
                    "--qblade-root",
                    "run_headless/QBladeCE_2.0.9.7",
                    "--output-dir",
                    f"run_headless/runs/{output_name}",
                    "--result-name",
                    output_name,
                    "--result-filter",
                    str(RESULT_FILTER.relative_to(REPO_ROOT)),
                    "--threads",
                    "2",
                    "--no-use-opencl",
                    "--progress-every",
                    "500",
                ]
                if role == "response":
                    runner_args.extend(
                        [
                            "--sil-vorlap",
                            "--coupling-mode",
                            "sil-vorlap",
                            "--record-sil-vorlap-history",
                        ]
                    )
                else:
                    runner_args.extend(["--coupling-mode", "none"])
                records.append(
                    {
                        **{key: value for key, value in record.items() if key not in {"sim", "config", "manifest"}},
                        "case_root": f"{token}/{role}",
                        "sim": relative_sim,
                        "config": relative_config,
                        "manifest": record["manifest"].relative_to(staging).as_posix(),
                        "suggested_output_dir": f"run_headless/runs/{output_name}",
                        "runner_args_relative_to_output_root": runner_args,
                    }
                )
        run_plan = {
            "schema": "grinderslev2022-modal-power-run-plan-v2",
            "created_from_base_manifest_sha256": base_manifest_sha,
            "duration_s": float(duration_s),
            "timesteps_s": list(timestep_values),
            "result_filter": str(RESULT_FILTER.relative_to(REPO_ROOT)),
            "model_scope": (
                "calibration-based one-mode forced-motion power closure; no wake prediction"
            ),
            "modal_mass_kg": quadrature.modal_mass_kg,
            "public_openfast_modal_mass_kg": quadrature.public_openfast_modal_mass_kg,
            "natural_frequency_hz": quadrature.natural_frequency_hz,
            "mode_shape_csv_sha256": EXPECTED_QBLADE_MODE_CSV_SHA256,
            "mode_identification_json_sha256": (
                EXPECTED_QBLADE_MODE_IDENTIFICATION_SHA256
            ),
            "structural_damping_ratio": quadrature.structural_damping_ratio,
            "structural_power_coefficient_wpm2": (
                quadrature.structural_power_coefficient_wpm2
            ),
            "predicted_forced_motion_curve_intersection_m": equilibrium,
            "published_targets": {
                "fsi_amplitude_m": REFERENCE_FSI_AMPLITUDE_M,
                "fsi_aerodynamic_power_w": REFERENCE_FSI_POWER_W,
                "forced_motion_damping_intersection_m": (
                    REFERENCE_FORCED_MOTION_INTERSECTION_M
                ),
                "forced_motion_damping_intersection_power_w": (
                    REFERENCE_FORCED_MOTION_INTERSECTION_POWER_W
                ),
                "zero_damping_amplitude_m": REFERENCE_ZERO_DAMPING_AMPLITUDE_M,
            },
            "cases": records,
        }
        write_json(staging / "run_plan.json", run_plan)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "run_plan.json"


@dataclass(frozen=True)
class QBladeResultHistory:
    columns: dict[str, np.ndarray]
    row_count: int

    @property
    def time_s(self) -> np.ndarray:
        return self.columns["Time~[s]"]


@dataclass(frozen=True)
class SilHistory:
    time_s: np.ndarray
    azimuth_deg: np.ndarray
    velocity_mps: np.ndarray
    generated_force_n: np.ndarray
    applied_force_n: np.ndarray
    applied_power_w: np.ndarray


@dataclass(frozen=True)
class ReplayHistory:
    generated_response_force_n: np.ndarray
    target_cycle_mean_power_w: np.ndarray
    applied_target_cycle_mean_power_w: np.ndarray
    amplitude_estimate_m: np.ndarray
    raw_generalized_velocity_mps: np.ndarray
    generated_generalized_force_n: np.ndarray
    generated_response_power_w: np.ndarray
    applied_response_power_w: np.ndarray
    mass_coordinate_lagged_work_w: np.ndarray
    inferred_base_generated_power_w: np.ndarray
    inferred_base_applied_power_w: np.ndarray
    maximum_force_replay_error_n: float


def lagged_mass_coordinate_work(
    generated_generalized_force_n: np.ndarray,
    raw_generalized_velocity_mps: np.ndarray,
) -> np.ndarray:
    """Return same-record ``Q[n-1] qdot_raw[n]`` work with startup zero.

    This is the power that the replayed generalized force would deliver if its
    nodal lift-back were exactly work-conjugate to the mass-projected response
    coordinate.  It is a diagnostic countercheck, not a replacement force law.
    """

    force = np.asarray(generated_generalized_force_n, dtype=float)
    velocity = np.asarray(raw_generalized_velocity_mps, dtype=float)
    if force.ndim != 1 or force.shape != velocity.shape or force.size < 2:
        raise ValueError(
            "Mass-coordinate work requires equal one-dimensional force and velocity histories."
        )
    if np.any(~np.isfinite(force)) or np.any(~np.isfinite(velocity)):
        raise ValueError("Mass-coordinate force and velocity histories must be finite.")
    work = np.zeros_like(force)
    work[1:] = force[:-1] * velocity[1:]
    return work


def _qblade_keyword_float(path: Path, keyword: str) -> float:
    matches = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) >= 2 and fields[1] == keyword:
            matches.append(fields[0])
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one QBlade {keyword} keyword in {path}.")
    return _finite_float(matches[0], f"QBlade {keyword}")


def read_qblade_result_strict(
    path: Path,
    *,
    expected_steps: int,
    expected_timestep_s: float,
) -> QBladeResultHistory:
    """Read a complete finite QBlade export without skipping malformed rows."""

    result_path = Path(path)
    lines = result_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4:
        raise ValueError(f"QBlade result is incomplete: {result_path}")
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Time~[s]" in line.split("\t")
        ),
        None,
    )
    if header_index != 2:
        raise ValueError(
            f"QBlade result must have the standard two-line preamble: {result_path}"
        )
    header = lines[header_index].split("\t")
    if any(not name for name in header) or len(header) != len(set(header)):
        raise ValueError(f"QBlade result header is blank or duplicated: {result_path}")
    missing = [name for name in REQUIRED_RESULT_COLUMNS if name not in header]
    if missing:
        raise ValueError(
            f"QBlade result is missing required channel(s): {', '.join(missing)}"
        )
    data_lines = lines[header_index + 1 :]
    if len(data_lines) != expected_steps:
        raise ValueError(
            f"QBlade result has {len(data_lines)} rows; expected exactly {expected_steps}."
        )
    selected = {name: np.empty(expected_steps, dtype=float) for name in REQUIRED_RESULT_COLUMNS}
    selected_indices = {name: header.index(name) for name in REQUIRED_RESULT_COLUMNS}
    for row_index, line in enumerate(data_lines):
        cells = line.split("\t")
        if len(cells) != len(header):
            raise ValueError(
                f"QBlade result row {row_index} has {len(cells)} cells; expected {len(header)}."
            )
        for cell_index, value in enumerate(cells):
            _finite_float(value, f"QBlade result row {row_index} column {header[cell_index]}")
        for name, column_index in selected_indices.items():
            selected[name][row_index] = float(cells[column_index])

    timestep = _finite_float(expected_timestep_s, "expected_timestep_s")
    expected_time = np.arange(expected_steps, dtype=float) * timestep
    time_tolerance = max(1.0e-11, timestep * 1.0e-8)
    time_error = np.abs(selected["Time~[s]"] - expected_time)
    if float(np.max(time_error)) > time_tolerance:
        worst = int(np.argmax(time_error))
        raise ValueError(
            "QBlade result time grid mismatch at row "
            f"{worst}: observed={selected['Time~[s]'][worst]:.17g}, "
            f"expected={expected_time[worst]:.17g}."
        )
    return QBladeResultHistory(columns=selected, row_count=expected_steps)


def read_sil_history_strict(
    path: Path,
    *,
    node_ids: Sequence[str],
    expected_steps: int,
    expected_timestep_s: float,
) -> SilHistory:
    """Read and cross-check the immutable generated/applied SIL force provenance."""

    history_path = Path(path)
    ids = tuple(str(node_id) for node_id in node_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("node_ids must contain unique values.")
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"SIL history has no header: {history_path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"SIL history contains duplicate columns: {history_path}")
        required = {
            "force_history_schema",
            "legacy_force_columns_semantics",
            "step_index",
            "time_s",
            "generated_step_index",
            "generated_time_s",
            "applied_source_step_index",
            "applied_source_time_s",
            "applied_startup_zero",
            "applied_force_lag_steps",
            "qblade_timestep",
            "azimuth_deg",
            "applied_force_dot_structural_velocity_w",
            "applied_power_sign_convention",
        }
        for node_id in ids:
            token = _safe_node_token(node_id)
            for axis in "xyz":
                required.update(
                    {
                        f"vel_{token}_{axis}_mps",
                        f"generated_force_{token}_{axis}_n",
                        f"applied_force_{token}_{axis}_n",
                        f"force_{token}_{axis}_n",
                    }
                )
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(
                "SIL history is missing required column(s): " + ", ".join(missing[:12])
            )
        rows = list(reader)
    if len(rows) != expected_steps:
        raise ValueError(
            f"SIL history has {len(rows)} rows; expected exactly {expected_steps}."
        )

    timestep = _finite_float(expected_timestep_s, "expected_timestep_s")
    time = np.empty(expected_steps, dtype=float)
    azimuth = np.empty(expected_steps, dtype=float)
    velocity = np.empty((expected_steps, len(ids), 3), dtype=float)
    generated = np.empty_like(velocity)
    applied = np.empty_like(velocity)
    applied_power = np.empty(expected_steps, dtype=float)
    for row_index, row in enumerate(rows):
        if None in row:
            raise ValueError(f"SIL history row {row_index} has extra cells.")
        if row["force_history_schema"] != "vorlap-sil-force-history-v2":
            raise ValueError(f"SIL history row {row_index} does not use schema v2.")
        if row["legacy_force_columns_semantics"] != "generated_force_alias":
            raise ValueError(f"SIL history row {row_index} has invalid legacy semantics.")
        if row["applied_power_sign_convention"] != (
            "positive_when_applied_force_does_work_on_current_structural_velocity"
        ):
            raise ValueError(f"SIL history row {row_index} has an unknown power convention.")
        if _strict_integer(row["step_index"], "step_index") != row_index:
            raise ValueError(f"SIL history step is nonsequential at row {row_index}.")
        if _strict_integer(row["generated_step_index"], "generated_step_index") != row_index:
            raise ValueError(f"SIL generated step is nonsequential at row {row_index}.")
        if _strict_integer(row["applied_force_lag_steps"], "applied_force_lag_steps") != 1:
            raise ValueError("This benchmark requires the recorded one-step SIL force lag.")
        if not math.isclose(
            _finite_float(row["qblade_timestep"], "qblade_timestep"),
            float(row_index),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"SIL QBlade timestep mismatch at row {row_index}.")
        expected_time = row_index * timestep
        wrapper_time = _finite_float(row["time_s"], "time_s")
        generated_time = _finite_float(row["generated_time_s"], "generated_time_s")
        if not math.isclose(wrapper_time, expected_time, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"SIL wrapper time mismatch at row {row_index}.")
        if not math.isclose(generated_time, wrapper_time, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"SIL generated time mismatch at row {row_index}.")
        time[row_index] = wrapper_time
        azimuth[row_index] = _finite_float(row["azimuth_deg"], "azimuth_deg")
        startup = _strict_bool_text(row["applied_startup_zero"], "applied_startup_zero")
        if row_index == 0:
            if not startup or row["applied_source_step_index"] not in {"", None}:
                raise ValueError("The first SIL row must apply the explicit startup-zero packet.")
            if row["applied_source_time_s"] not in {"", None}:
                raise ValueError("The startup-zero packet must not name a source time.")
        else:
            if startup:
                raise ValueError(f"Unexpected SIL startup-zero packet at row {row_index}.")
            if _strict_integer(
                row["applied_source_step_index"], "applied_source_step_index"
            ) != row_index - 1:
                raise ValueError(f"SIL applied source step mismatch at row {row_index}.")
            expected_source_time = (row_index - 1) * timestep
            if not math.isclose(
                _finite_float(row["applied_source_time_s"], "applied_source_time_s"),
                expected_source_time,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(f"SIL applied source time mismatch at row {row_index}.")

        for node_index, node_id in enumerate(ids):
            token = _safe_node_token(node_id)
            for axis_index, axis in enumerate("xyz"):
                velocity[row_index, node_index, axis_index] = _finite_float(
                    row[f"vel_{token}_{axis}_mps"],
                    f"row {row_index} {node_id} velocity {axis}",
                )
                generated_value = _finite_float(
                    row[f"generated_force_{token}_{axis}_n"],
                    f"row {row_index} {node_id} generated force {axis}",
                )
                legacy_value = _finite_float(
                    row[f"force_{token}_{axis}_n"],
                    f"row {row_index} {node_id} legacy force {axis}",
                )
                if legacy_value != generated_value:
                    raise ValueError(
                        f"SIL legacy/generated force mismatch at row {row_index}, {node_id}, {axis}."
                    )
                generated[row_index, node_index, axis_index] = generated_value
                applied[row_index, node_index, axis_index] = _finite_float(
                    row[f"applied_force_{token}_{axis}_n"],
                    f"row {row_index} {node_id} applied force {axis}",
                )
        expected_applied = np.zeros_like(applied[row_index]) if row_index == 0 else generated[row_index - 1]
        if not np.array_equal(applied[row_index], expected_applied):
            raise ValueError(f"Stored applied force violates one-step provenance at row {row_index}.")
        computed_power = float(np.sum(applied[row_index] * velocity[row_index]))
        stored_power = _finite_float(
            row["applied_force_dot_structural_velocity_w"],
            "applied_force_dot_structural_velocity_w",
        )
        if not math.isclose(computed_power, stored_power, rel_tol=2.0e-13, abs_tol=2.0e-9):
            raise ValueError(f"Stored SIL applied work is inconsistent at row {row_index}.")
        applied_power[row_index] = stored_power
    return SilHistory(
        time_s=time,
        azimuth_deg=azimuth,
        velocity_mps=velocity,
        generated_force_n=generated,
        applied_force_n=applied,
        applied_power_w=applied_power,
    )


def _resolve_summary_file(run_dir: Path) -> Path:
    matches = sorted(Path(run_dir).glob("*_run_summary.json"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one QBlade run summary in {run_dir}, observed {len(matches)}."
        )
    return matches[0]


def _resolve_run_artifact(run_dir: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Run summary has no {label} path.")
    candidate = Path(run_dir) / Path(raw_path).name
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _validate_input_hash(
    summary: Mapping[str, Any],
    key: str,
    current_path: Path,
) -> None:
    try:
        expected = str(summary["reproducibility"]["input_files"][key]["sha256"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Run summary lacks reproducibility hash for {key}.") from exc
    observed = sha256_file(current_path)
    if observed != expected:
        raise ValueError(
            f"Executed {key} hash {expected} does not match current input {observed}."
        )


def validate_run_summary(
    summary: Mapping[str, Any],
    *,
    expected_steps: int,
    expected_timestep_s: float,
    coupled: bool,
    config_path: Path | None = None,
) -> None:
    if summary.get("failed") is not False or summary.get("stopped") is not False:
        raise ValueError("QBlade run summary records a failure or stop.")
    if _strict_integer(summary.get("requested_steps"), "requested_steps") != expected_steps:
        raise ValueError("QBlade run requested-step count does not match the case.")
    if _strict_integer(summary.get("completed_steps"), "completed_steps") != expected_steps:
        raise ValueError("QBlade run did not complete every requested timestep.")
    diagnostics = summary.get("qblade_export_time_diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("available") is not True:
        raise ValueError("QBlade export time diagnostics are unavailable.")
    if _strict_integer(diagnostics.get("samples"), "export samples") != expected_steps:
        raise ValueError("QBlade export sample count does not match the case.")
    for name in ("min_step_s", "max_step_s"):
        if not math.isclose(
            _finite_float(diagnostics.get(name), name),
            expected_timestep_s,
            rel_tol=1.0e-7,
            abs_tol=1.0e-10,
        ):
            raise ValueError(f"QBlade export {name} differs from the case timestep.")

    if coupled:
        if summary.get("coupling_mode") != "sil-vorlap" or summary.get("sil_vorlap") is not True:
            raise ValueError("Response benchmark did not use SIL-VorLap coupling.")
        if _strict_integer(summary.get("sil_vorlap_updates"), "sil_vorlap_updates") != expected_steps:
            raise ValueError("Response benchmark has an incomplete SIL update count.")
        timing = summary.get("sil_vorlap_timing_diagnostics")
        if not isinstance(timing, Mapping) or _strict_integer(
            timing.get("samples"), "SIL timing samples"
        ) != expected_steps:
            raise ValueError("Response benchmark lacks complete SIL timing diagnostics.")
        last_sample = timing.get("last_sample")
        if not isinstance(last_sample, Mapping) or _strict_integer(
            last_sample.get("applied_force_lag_steps"), "applied_force_lag_steps"
        ) != 1:
            raise ValueError("Response benchmark does not record the required one-step force lag.")
        response_summary = summary.get("sil_vorlap_response_diagnostics_summary")
        if not isinstance(response_summary, Mapping):
            raise ValueError("Run summary lacks sil_vorlap_response_diagnostics_summary.")
        if response_summary.get("model_type") != RESPONSE_MODEL_TYPE:
            raise ValueError("Run summary response model type is not modal_aerodynamic_power.")
        if _strict_integer(response_summary.get("samples"), "response samples") != expected_steps:
            raise ValueError("Response diagnostic sample count is incomplete.")
        for counter in LIMIT_COUNTERS:
            if _strict_integer(response_summary.get(counter, 0), counter) != 0:
                raise ValueError(f"Response-model qualification limit was active: {counter}.")
        coverage = summary.get("sil_vorlap_source_coverage")
        if not isinstance(coverage, list) or len(coverage) != 1:
            raise ValueError("Expected one complete source-coverage block for the single blade.")
        for counter in (
            "aliased_frequency_count",
            "frequency_truncated_count",
            "source_clip_count",
            "force_coefficient_out_of_range_count",
            "nonfinite_count",
            "aoa_out_of_range_count",
            "re_out_of_range_count",
        ):
            if _strict_integer(coverage[0].get(counter, 0), counter) != 0:
                raise ValueError(f"Source-coverage gate failed: {counter}.")
        if config_path is None:
            raise ValueError("Coupled-run validation requires the executed config path.")
        _validate_input_hash(summary, "vorlap_config", Path(config_path))
    else:
        if summary.get("coupling_mode") != "none":
            raise ValueError("Matched structural baseline must use coupling_mode='none'.")


def replay_response_model(
    config_path: Path,
    history: SilHistory,
) -> ReplayHistory:
    """Replay the response closure and match all generated force components."""

    from vorlap.qblade_modal_power_model import QBladeModalPowerDiagnostics
    from vorlap.qblade_nodes import parse_qblade_node_id
    from vorlap.qblade_runtime import VorLapQBladeRuntime

    config = load_json(config_path)
    if _finite_float(config.get("force_scale"), "force_scale") != 0.0:
        raise ValueError(
            "The Grinderslev response benchmark requires force_scale=0 so all SIL force "
            "comes from the response closure."
        )
    response_config = config.get("response_model")
    if not isinstance(response_config, Mapping) or response_config.get("type") != RESPONSE_MODEL_TYPE:
        raise ValueError("Config does not enable the modal aerodynamic-power response model.")
    if response_config.get("limit_action") != "error":
        raise ValueError("Qualification replay requires fail-closed limit_action='error'.")
    exact_node_ids = tuple(str(value) for value in config.get("node_ids", ()))
    canonical_node_ids = tuple(
        parse_qblade_node_id(node_id).canonical_node_id for node_id in exact_node_ids
    )
    if len(canonical_node_ids) != len(set(canonical_node_ids)):
        raise ValueError("Exact swap node IDs collide after runtime canonicalization.")
    if tuple(str(value) for value in response_config.get("node_ids", ())) != canonical_node_ids:
        raise ValueError(
            "response_model.node_ids do not exactly match canonicalized top-level swap IDs."
        )
    identity = config.get("response_model_node_identity")
    expected_mapping = [
        {"exact_swap_node_id": source, "canonical_controller_node_id": target}
        for source, target in zip(exact_node_ids, canonical_node_ids)
    ]
    if not isinstance(identity, Mapping) or (
        identity.get("exact_swap_node_ids") != list(exact_node_ids)
        or identity.get("canonical_controller_node_ids") != list(canonical_node_ids)
        or identity.get("exact_to_canonical") != expected_mapping
        or identity.get("exact_swap_node_ids_sha256")
        != _string_sequence_sha256(exact_node_ids)
        or identity.get("canonical_controller_node_ids_sha256")
        != _string_sequence_sha256(canonical_node_ids)
    ):
        raise ValueError("Config exact-to-canonical response node identity is missing or stale.")

    runtime = VorLapQBladeRuntime.from_qblade_config(str(Path(config_path).resolve()))
    if runtime.controller.response_model_type != RESPONSE_MODEL_TYPE:
        raise ValueError("Runtime did not construct the requested modal aerodynamic-power model.")
    if not runtime.controller.response_model_enabled:
        raise ValueError("Runtime response model is disabled.")
    layout = runtime.layout
    scalar_blocks = {
        runtime.time_block: history.time_s,
        runtime.timestep_block: np.arange(history.time_s.size, dtype=float),
        runtime.azimuth_block: history.azimuth_deg,
    }
    velocity_spec = layout[runtime.velocity_block]
    force_spec = layout[runtime.force_block]
    replay_force = np.empty_like(history.generated_force_n)
    target_power = np.empty(history.time_s.size, dtype=float)
    applied_target_power = np.empty_like(target_power)
    amplitude = np.empty_like(target_power)
    raw_generalized_velocity = np.empty_like(target_power)
    generated_generalized_force = np.empty_like(target_power)
    generated_power = np.empty_like(target_power)
    max_error = 0.0
    swap = np.zeros(runtime.swap_size, dtype=float)
    for step in range(history.time_s.size):
        for block_name, values in scalar_blocks.items():
            spec = layout[block_name]
            swap[spec.offset] = values[step]
        swap[
            velocity_spec.offset : velocity_spec.offset + velocity_spec.size
        ] = history.velocity_mps[step].reshape(-1)
        runtime.update(swap)
        force = np.array(
            swap[force_spec.offset : force_spec.offset + force_spec.size],
            dtype=float,
            copy=True,
        ).reshape(history.generated_force_n.shape[1:])
        replay_force[step] = force
        error = float(np.max(np.abs(force - history.generated_force_n[step])))
        max_error = max(max_error, error)
        if not np.allclose(
            force,
            history.generated_force_n[step],
            rtol=FORCE_REPLAY_RTOL,
            atol=FORCE_REPLAY_ATOL_N,
        ):
            location = np.unravel_index(
                int(np.argmax(np.abs(force - history.generated_force_n[step]))),
                force.shape,
            )
            raise ValueError(
                "Offline response replay differs from stored generated force at "
                f"step {step}, node {location[0]}, axis {location[1]} by {error:.6g} N."
            )
        diagnostics = runtime.last_response_diagnostics
        if not isinstance(diagnostics, QBladeModalPowerDiagnostics):
            raise ValueError("Runtime did not expose modal-power diagnostics during replay.")
        if (
            diagnostics.target_power_limited
            or diagnostics.generalized_force_limited
            or diagnostics.nodal_force_limited_nodes != 0
        ):
            raise ValueError(f"Response-model limit was active at replay step {step}.")
        target_power[step] = diagnostics.target_cycle_mean_power_w
        applied_target_power[step] = diagnostics.applied_target_cycle_mean_power_w
        amplitude[step] = diagnostics.amplitude_estimate_m
        raw_generalized_velocity[step] = diagnostics.raw_qdot_mps
        generated_generalized_force[step] = diagnostics.applied_generalized_force_n
        generated_power[step] = float(np.sum(force * history.velocity_mps[step]))
        if not math.isclose(
            generated_power[step],
            diagnostics.instantaneous_structural_power_w,
            rel_tol=2.0e-12,
            abs_tol=2.0e-8,
        ):
            raise ValueError(f"Replayed structural power mismatch at step {step}.")

    applied_response_force = np.zeros_like(replay_force)
    applied_response_force[1:] = replay_force[:-1]
    applied_response_power = np.einsum(
        "tij,tij->t", applied_response_force, history.velocity_mps
    )
    mass_coordinate_lagged_work = lagged_mass_coordinate_work(
        generated_generalized_force,
        raw_generalized_velocity,
    )
    inferred_base_generated = np.einsum(
        "tij,tij->t",
        history.generated_force_n - replay_force,
        history.velocity_mps,
    )
    inferred_base_applied = history.applied_power_w - applied_response_power
    if float(np.max(np.abs(inferred_base_generated))) > 5.0e-8:
        raise ValueError("A nonzero generated base-force contribution remains despite force_scale=0.")
    if float(np.max(np.abs(inferred_base_applied))) > 5.0e-8:
        raise ValueError("A nonzero applied base-force contribution remains despite force_scale=0.")
    return ReplayHistory(
        generated_response_force_n=replay_force,
        target_cycle_mean_power_w=target_power,
        applied_target_cycle_mean_power_w=applied_target_power,
        amplitude_estimate_m=amplitude,
        raw_generalized_velocity_mps=raw_generalized_velocity,
        generated_generalized_force_n=generated_generalized_force,
        generated_response_power_w=generated_power,
        applied_response_power_w=applied_response_power,
        mass_coordinate_lagged_work_w=mass_coordinate_lagged_work,
        inferred_base_generated_power_w=inferred_base_generated,
        inferred_base_applied_power_w=inferred_base_applied,
        maximum_force_replay_error_n=max_error,
    )


def _steady_mask(
    time_s: np.ndarray,
    *,
    timestep_s: float,
    duration_s: float,
    steady_window_s: float,
) -> np.ndarray:
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or time.size < 2 or np.any(~np.isfinite(time)):
        raise ValueError("time_s must be a finite one-dimensional history.")
    window = _finite_float(steady_window_s, "steady_window_s")
    minimum_window = 2.0 * SETTLING_CYCLES_PER_HALF / REFERENCE_FREQUENCY_HZ
    if window < minimum_window:
        raise ValueError(
            "steady_window_s must contain at least "
            f"{2 * SETTLING_CYCLES_PER_HALF} target-frequency cycles "
            f"({minimum_window:.6g} s at the reference frequency)."
        )
    if window >= duration_s:
        raise ValueError("steady_window_s must be shorter than the full simulation duration.")
    expected_count = int(math.floor(window / timestep_s + 1.0e-12))
    if expected_count < 2 or expected_count >= time.size:
        raise ValueError("Steady-window sample count is invalid for the result history.")
    mask = np.zeros(time.size, dtype=bool)
    mask[-expected_count:] = True
    start = duration_s - window
    if time[mask][0] < start - max(1.0e-12, timestep_s * 1.0e-9):
        raise ValueError("Steady window unexpectedly includes the excluded startup interval.")
    return mask


def sinusoidal_signal_metrics(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    frequency_hz: float = REFERENCE_FREQUENCY_HZ,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Fit a target-frequency amplitude and return a detrended Hilbert envelope."""

    time = np.asarray(time_s, dtype=float)
    signal = np.asarray(values, dtype=float)
    if (
        time.ndim != 1
        or signal.shape != time.shape
        or time.size < 16
        or np.any(~np.isfinite(time))
        or np.any(~np.isfinite(signal))
    ):
        raise ValueError("Signal metrics require aligned finite one-dimensional histories.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Signal-metric timestamps must increase strictly.")
    frequency = _finite_float(frequency_hz, "frequency_hz")
    if frequency <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    centered_time = time - float(np.mean(time))
    omega_time = 2.0 * math.pi * frequency * time
    design = np.column_stack(
        (
            np.ones(time.size),
            centered_time,
            np.sin(omega_time),
            np.cos(omega_time),
        )
    )
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(
        design, signal, rcond=None
    )
    if rank != 4 or np.any(~np.isfinite(coefficients)) or np.any(~np.isfinite(singular_values)):
        raise ValueError("Target-frequency signal fit is rank deficient or non-finite.")
    trend = coefficients[0] + coefficients[1] * centered_time
    detrended = signal - trend
    envelope = np.abs(hilbert(detrended))
    if np.any(~np.isfinite(envelope)):
        raise ValueError("Hilbert amplitude envelope became non-finite.")
    dt = float(np.median(np.diff(time)))
    trim = int(math.ceil((1.0 / frequency) / dt))
    interior = envelope[trim:-trim] if 2 * trim < envelope.size else envelope
    if interior.size < 8:
        raise ValueError("Steady history is too short after envelope edge trimming.")

    window = np.hanning(detrended.size)
    spectrum = np.abs(np.fft.rfft(detrended * window)) ** 2
    frequencies = np.fft.rfftfreq(detrended.size, dt)
    search = (frequencies >= 0.2) & (frequencies <= 1.5)
    if not np.any(search) or float(np.max(spectrum[search])) <= 0.0:
        raise ValueError("Cannot identify a finite nonzero response spectrum.")
    peak_frequency = float(frequencies[search][np.argmax(spectrum[search])])
    fit_amplitude = float(math.hypot(coefficients[2], coefficients[3]))
    metrics: dict[str, float | int] = {
        "samples": int(time.size),
        "duration_s": float(time.size * dt),
        "target_frequency_hz": frequency,
        "fitted_amplitude": fit_amplitude,
        "median_hilbert_envelope": float(np.median(interior)),
        "p95_hilbert_envelope": float(np.quantile(interior, 0.95)),
        "peak_abs_detrended": float(np.max(np.abs(detrended))),
        "fitted_linear_drift_per_s": float(coefficients[1]),
        "spectral_peak_hz": peak_frequency,
        "spectral_bin_width_hz": float(frequencies[1] - frequencies[0]),
        "fit_rms_residual": float(
            math.sqrt(float(residuals[0]) / time.size) if residuals.size else 0.0
        ),
    }
    return metrics, envelope


def load_channel_fatigue_metrics(
    values: np.ndarray,
    *,
    timestep_s: float,
    wohler_exponent: float,
    frequency_hz: float = REFERENCE_FREQUENCY_HZ,
) -> dict[str, float]:
    """Return a fixed-cycle DEL and the unscaled rainflow range-power sum.

    The range-power sum is suitable for a relative time-step gate.  It is not
    Miner damage because no S--N intercept, stress recovery, mean-stress
    correction, material factor, or lifetime occurrence count is applied.
    """

    from vorlap.qblade_surface_strain_del import equivalent_load_with_cycles_fast

    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or signal.size < 3 or np.any(~np.isfinite(signal)):
        raise ValueError("Fatigue metric requires a finite one-dimensional history.")
    exponent = _finite_float(wohler_exponent, "wohler_exponent")
    if exponent <= 0.0:
        raise ValueError("wohler_exponent must be positive.")
    equivalent_cycles = signal.size * timestep_s * frequency_hz
    damage_equivalent_range, counted_cycles = equivalent_load_with_cycles_fast(
        signal,
        exponent,
        equivalent_cycles,
    )
    unit_cycle_range, independently_counted_cycles = equivalent_load_with_cycles_fast(
        signal,
        exponent,
        1.0,
    )
    if not math.isclose(
        counted_cycles,
        independently_counted_cycles,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Rainflow cycle count changed with equivalent-cycle normalization.")
    range_power_sum = float(unit_cycle_range**exponent)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (damage_equivalent_range, counted_cycles, range_power_sum)
    ):
        raise ValueError("Load-channel fatigue metric became invalid.")
    return {
        "wohler_exponent": exponent,
        "equivalent_cycles": equivalent_cycles,
        "rainflow_cycles": float(counted_cycles),
        "damage_equivalent_range": float(damage_equivalent_range),
        "rainflow_range_power_sum": range_power_sum,
    }


def _relative_difference(coarse: float, fine: float, name: str) -> float:
    coarse_value = _finite_float(coarse, f"coarse {name}")
    fine_value = _finite_float(fine, f"fine {name}")
    if fine_value == 0.0:
        if coarse_value == 0.0:
            return 0.0
        raise ValueError(f"Cannot normalize nonzero coarse {name} by zero fine value.")
    return abs(coarse_value - fine_value) / abs(fine_value)


def _validate_aligned_time_grids(
    left_time_s: np.ndarray,
    right_time_s: np.ndarray,
    *,
    label: str,
    atol_s: float = 1.0e-12,
) -> None:
    """Require two finite time grids to be aligned sample for sample."""

    left = np.asarray(left_time_s, dtype=float)
    right = np.asarray(right_time_s, dtype=float)
    tolerance = _finite_float(atol_s, "atol_s")
    if tolerance < 0.0:
        raise ValueError("atol_s must be non-negative.")
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.shape != right.shape
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
        or not np.allclose(left, right, rtol=0.0, atol=tolerance)
    ):
        raise ValueError(f"{label} time grids are not aligned.")


def _centered_cycle_synchronous_pair(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return centered, phase-aligned six-cycle halves on one fixed phase grid."""

    time = np.asarray(time_s, dtype=float)
    signal = np.asarray(values, dtype=float)
    frequency = _finite_float(frequency_hz, "frequency_hz")
    if (
        time.ndim != 1
        or signal.shape != time.shape
        or time.size < 2
        or np.any(~np.isfinite(time))
        or np.any(~np.isfinite(signal))
        or np.any(np.diff(time) <= 0.0)
        or frequency <= 0.0
    ):
        raise ValueError("Cycle-synchronous windows require aligned finite histories.")
    half_duration = SETTLING_CYCLES_PER_HALF / frequency
    center = 0.5 * (float(time[0]) + float(time[-1]))
    start = center - half_duration
    end = center + half_duration
    if start < time[0] or end > time[-1]:
        raise ValueError(
            "Steady history is too short for the centered 12-cycle settling record."
        )
    intervals = (
        SETTLING_CYCLES_PER_HALF * SETTLING_PHASE_GRID_INTERVALS_PER_CYCLE
    )
    first_time = np.linspace(start, center, intervals + 1)
    second_time = np.linspace(center, end, intervals + 1)
    first_values = np.interp(first_time, time, signal)
    second_values = np.interp(second_time, time, signal)
    grid_timestep = half_duration / intervals
    provenance = {
        "schema": "grinderslev2022-centered-cycle-windows-v1",
        "construction": (
            "center the fixed 12-target-cycle record in the declared steady history; "
            "split into phase-aligned 6+6 cycle halves; linearly interpolate both "
            "halves onto the same fixed phase grid"
        ),
        "frequency_hz": frequency,
        "total_target_cycles": 2 * SETTLING_CYCLES_PER_HALF,
        "target_cycles_per_half": SETTLING_CYCLES_PER_HALF,
        "phase_grid_intervals_per_cycle": (
            SETTLING_PHASE_GRID_INTERVALS_PER_CYCLE
        ),
        "phase_grid_samples_per_half": intervals + 1,
        "phase_grid_timestep_s": grid_timestep,
        "window_start_s": start,
        "window_split_s": center,
        "window_end_s": end,
        "boundary_selection": "time/frequency only; no signal extrema or rainflow cycles",
    }
    return first_time, first_values, second_time, second_values, provenance


def evaluate_settling_qualification(
    time_s: np.ndarray,
    paired_tip_edgewise_m: np.ndarray,
    paired_root_edgewise_moment_nm: np.ndarray,
    response_model_amplitude_estimate_m: np.ndarray,
    target_cycle_mean_power_w: np.ndarray,
    actually_applied_response_work_w: np.ndarray,
    *,
    timestep_s: float,
    structural_power_coefficient_wpm2: float,
    wohler_exponent: float,
    frequency_hz: float = REFERENCE_FREQUENCY_HZ,
) -> dict[str, Any]:
    """Apply frozen within-run stationarity and structural-model gates.

    Half-to-half metrics use a centered, fixed 12-target-cycle record split
    into phase-aligned 6+6 cycle windows on a common 500-interval-per-cycle
    phase grid.  Window boundaries depend only on time and the pinned target
    frequency, not signal extrema or rainflow counts.  The envelope gate
    regresses the detrended Hilbert amplitude after trimming one target period
    from each edge, then projects that slope over ten cycles.  This is an
    amplitude-growth metric; the DC trend returned by
    :func:`sinusoidal_signal_metrics` is not used as a settling surrogate.

    ``rainflow_range_power_sum`` is denoted ``A_m`` only as an unscaled
    pseudo-damage sum for a numerical comparison.  It is not Miner damage.
    """

    timestep = _finite_float(timestep_s, "timestep_s")
    coefficient = _finite_float(
        structural_power_coefficient_wpm2,
        "structural_power_coefficient_wpm2",
    )
    exponent = _finite_float(wohler_exponent, "wohler_exponent")
    frequency = _finite_float(frequency_hz, "frequency_hz")
    if timestep <= 0.0 or coefficient <= 0.0 or exponent <= 0.0 or frequency <= 0.0:
        raise ValueError(
            "Settling qualification requires positive timestep, damping coefficient, "
            "Wohler exponent, and frequency."
        )

    names_and_values = {
        "time_s": time_s,
        "paired_tip_edgewise_m": paired_tip_edgewise_m,
        "paired_root_edgewise_moment_nm": paired_root_edgewise_moment_nm,
        "response_model_amplitude_estimate_m": response_model_amplitude_estimate_m,
        "target_cycle_mean_power_w": target_cycle_mean_power_w,
        "actually_applied_response_work_w": actually_applied_response_work_w,
    }
    arrays = {name: np.asarray(values, dtype=float) for name, values in names_and_values.items()}
    time = arrays["time_s"]
    if time.ndim != 1 or time.size < 64:
        raise ValueError("Settling qualification requires a one-dimensional history.")
    if any(array.shape != time.shape for array in arrays.values()):
        raise ValueError("Settling qualification histories must have equal one-dimensional shape.")
    if any(np.any(~np.isfinite(array)) for array in arrays.values()):
        raise ValueError("Settling qualification histories must be finite.")
    if np.any(np.diff(time) <= 0.0) or not np.allclose(
        np.diff(time),
        timestep,
        rtol=1.0e-8,
        atol=max(1.0e-12, timestep * 1.0e-10),
    ):
        raise ValueError("Settling qualification time grid is not uniform at timestep_s.")
    if time.size * timestep < 2.0 * SETTLING_CYCLES_PER_HALF / frequency:
        raise ValueError(
            "Settling qualification requires at least 12 target-frequency cycles."
        )

    paired_tip = arrays["paired_tip_edgewise_m"]
    paired_root = arrays["paired_root_edgewise_moment_nm"]
    model_amplitude = arrays["response_model_amplitude_estimate_m"]
    target_power = arrays["target_cycle_mean_power_w"]
    applied_power = arrays["actually_applied_response_work_w"]

    first_tip_time, first_tip, second_tip_time, second_tip, window_provenance = (
        _centered_cycle_synchronous_pair(
            time, paired_tip, frequency_hz=frequency
        )
    )
    _first_root_time, first_root, _second_root_time, second_root, root_provenance = (
        _centered_cycle_synchronous_pair(
            time, paired_root, frequency_hz=frequency
        )
    )
    if window_provenance != root_provenance:
        raise ValueError("Tip and root cycle-window provenance changed unexpectedly.")
    first_tip_metrics, _ = sinusoidal_signal_metrics(
        first_tip_time, first_tip, frequency_hz=frequency
    )
    second_tip_metrics, _ = sinusoidal_signal_metrics(
        second_tip_time, second_tip, frequency_hz=frequency
    )
    full_tip_metrics, tip_envelope = sinusoidal_signal_metrics(
        time, paired_tip, frequency_hz=frequency
    )

    first_tip_amplitude = float(first_tip_metrics["fitted_amplitude"])
    second_tip_amplitude = float(second_tip_metrics["fitted_amplitude"])
    full_tip_amplitude = float(full_tip_metrics["fitted_amplitude"])
    tip_half_difference = _relative_difference(
        first_tip_amplitude,
        second_tip_amplitude,
        "settling paired tip amplitude",
    )

    edge_trim_samples = int(math.ceil((1.0 / frequency) / timestep))
    envelope_time = time[edge_trim_samples:-edge_trim_samples]
    envelope_interior = tip_envelope[edge_trim_samples:-edge_trim_samples]
    if envelope_time.size < 8:
        raise ValueError("Settling envelope is too short after edge trimming.")
    centered_envelope_time = envelope_time - float(np.mean(envelope_time))
    envelope_design = np.column_stack(
        (np.ones(envelope_time.size), centered_envelope_time)
    )
    envelope_coefficients, _, envelope_rank, _ = np.linalg.lstsq(
        envelope_design,
        envelope_interior,
        rcond=None,
    )
    if envelope_rank != 2 or np.any(~np.isfinite(envelope_coefficients)):
        raise ValueError("Settling envelope regression is rank deficient or non-finite.")
    envelope_reference = float(np.median(envelope_interior))
    if envelope_reference <= 0.0:
        raise ValueError("Settling envelope reference amplitude must be positive.")
    envelope_slope = float(envelope_coefficients[1])
    projected_duration = 10.0 / frequency
    projected_envelope_change = abs(envelope_slope) * projected_duration
    projected_envelope_relative_change = projected_envelope_change / envelope_reference

    phase_timestep = float(window_provenance["phase_grid_timestep_s"])
    first_root_fatigue = load_channel_fatigue_metrics(
        first_root,
        timestep_s=phase_timestep,
        wohler_exponent=exponent,
        frequency_hz=frequency,
    )
    second_root_fatigue = load_channel_fatigue_metrics(
        second_root,
        timestep_s=phase_timestep,
        wohler_exponent=exponent,
        frequency_hz=frequency,
    )
    root_del_difference = _relative_difference(
        first_root_fatigue["damage_equivalent_range"],
        second_root_fatigue["damage_equivalent_range"],
        "settling paired root DEL",
    )
    first_A_m_per_cycle = (
        first_root_fatigue["rainflow_range_power_sum"] / SETTLING_CYCLES_PER_HALF
    )
    second_A_m_per_cycle = (
        second_root_fatigue["rainflow_range_power_sum"] / SETTLING_CYCLES_PER_HALF
    )
    root_range_power_difference = _relative_difference(
        first_A_m_per_cycle,
        second_A_m_per_cycle,
        "settling paired root pseudo-damage A_m",
    )

    mean_applied_power = float(np.mean(applied_power))
    mean_target_power = float(np.mean(target_power))
    mean_model_amplitude = float(np.mean(model_amplitude))
    mapped_structural_damping_power = coefficient * full_tip_amplitude**2
    applied_damping_difference = _relative_difference(
        mean_applied_power,
        mapped_structural_damping_power,
        "settling applied work versus mapped damping",
    )
    applied_target_difference = _relative_difference(
        mean_applied_power,
        mean_target_power,
        "settling applied work versus target work",
    )
    amplitude_coordinate_difference = _relative_difference(
        mean_model_amplitude,
        full_tip_amplitude,
        "settling model versus paired-tip amplitude",
    )

    gates: dict[str, dict[str, Any]] = {
        "paired_tip_amplitude_half_to_half": {
            "first_half_fitted_amplitude_m": first_tip_amplitude,
            "second_half_fitted_amplitude_m": second_tip_amplitude,
            "absolute_relative_difference": tip_half_difference,
            "tolerance": SETTLING_TIP_HALF_WINDOW_TOLERANCE,
            "passed": bool(tip_half_difference <= SETTLING_TIP_HALF_WINDOW_TOLERANCE),
        },
        "paired_tip_envelope_projected_change_over_10_cycles": {
            "envelope_slope_mps": envelope_slope,
            "projection_duration_s": projected_duration,
            "projected_absolute_change_m": projected_envelope_change,
            "reference_median_envelope_m": envelope_reference,
            "absolute_relative_change": projected_envelope_relative_change,
            "tolerance": SETTLING_ENVELOPE_10_CYCLE_TOLERANCE,
            "passed": bool(
                projected_envelope_relative_change
                <= SETTLING_ENVELOPE_10_CYCLE_TOLERANCE
            ),
        },
        "paired_root_del_half_to_half": {
            "first_half_damage_equivalent_range_nm": first_root_fatigue[
                "damage_equivalent_range"
            ],
            "second_half_damage_equivalent_range_nm": second_root_fatigue[
                "damage_equivalent_range"
            ],
            "absolute_relative_difference": root_del_difference,
            "tolerance": SETTLING_DEL_HALF_WINDOW_TOLERANCE,
            "passed": bool(root_del_difference <= SETTLING_DEL_HALF_WINDOW_TOLERANCE),
        },
        "paired_root_pseudo_damage_A_m_half_to_half": {
            "wohler_exponent": exponent,
            "first_half_A_m": first_root_fatigue["rainflow_range_power_sum"],
            "second_half_A_m": second_root_fatigue["rainflow_range_power_sum"],
            "first_half_A_m_per_target_cycle": first_A_m_per_cycle,
            "second_half_A_m_per_target_cycle": second_A_m_per_cycle,
            "target_cycles_per_half": SETTLING_CYCLES_PER_HALF,
            "absolute_relative_difference": root_range_power_difference,
            "tolerance": SETTLING_RANGE_POWER_HALF_WINDOW_TOLERANCE,
            "passed": bool(
                root_range_power_difference
                <= SETTLING_RANGE_POWER_HALF_WINDOW_TOLERANCE
            ),
        },
        "lagged_applied_work_vs_mapped_structural_damping": {
            "mean_lagged_applied_response_work_w": mean_applied_power,
            "paired_tip_fitted_amplitude_m": full_tip_amplitude,
            "structural_power_coefficient_wpm2": coefficient,
            "mapped_structural_damping_power_w": mapped_structural_damping_power,
            "absolute_relative_difference": applied_damping_difference,
            "tolerance": SETTLING_POWER_CLOSURE_TOLERANCE,
            "diagnostic_scope": (
                "nominal single-coordinate structural-model adequacy diagnostic; "
                "C*A^2 is not a direct QBlade internal-dissipation measurement"
            ),
            "direct_qblade_structural_dissipation_available": False,
            "closure_claim": False,
            "passed": bool(
                applied_damping_difference <= SETTLING_POWER_CLOSURE_TOLERANCE
            ),
        },
        "lagged_applied_work_vs_target_work": {
            "mean_lagged_applied_response_work_w": mean_applied_power,
            "mean_target_cycle_power_w": mean_target_power,
            "absolute_relative_difference": applied_target_difference,
            "tolerance": SETTLING_POWER_CLOSURE_TOLERANCE,
            "passed": bool(
                applied_target_difference <= SETTLING_POWER_CLOSURE_TOLERANCE
            ),
        },
        "model_amplitude_estimate_vs_paired_tip_amplitude": {
            "mean_response_model_amplitude_estimate_m": mean_model_amplitude,
            "paired_tip_fitted_amplitude_m": full_tip_amplitude,
            "absolute_relative_difference": amplitude_coordinate_difference,
            "tolerance": SETTLING_AMPLITUDE_COORDINATE_TOLERANCE,
            "passed": bool(
                amplitude_coordinate_difference
                <= SETTLING_AMPLITUDE_COORDINATE_TOLERANCE
            ),
        },
    }
    return {
        "schema": "grinderslev2022-modal-power-settling-qualification-v2",
        "steady_samples": int(time.size),
        "cycle_synchronous_windows": window_provenance,
        "frequency_hz": frequency,
        "gates": gates,
        "passed": all(bool(row["passed"]) for row in gates.values()),
        "fatigue_boundary": (
            "A_m is the unscaled rainflow sum of range**m for a stationarity gate; "
            "it is not Miner damage or fatigue life."
        ),
    }


def _case_variant_manifest(config_path: Path, expected_role: str) -> tuple[Path, dict[str, Any]]:
    case_root = Path(config_path).resolve().parents[2]
    manifest_path = case_root / MANIFEST_RELATIVE
    manifest = load_json(manifest_path)
    variant = manifest.get("response_aware_variant")
    if not isinstance(variant, Mapping) or variant.get("role") != expected_role:
        raise ValueError(
            f"Case manifest does not identify the expected {expected_role!r} response variant."
        )
    file_rows = {
        str(row["path"]): row
        for row in manifest.get("files", [])
        if isinstance(row, Mapping) and "path" in row
    }
    relative_config = CONFIG_RELATIVE.as_posix()
    if relative_config not in file_rows:
        raise ValueError("Variant manifest does not hash the external config.")
    if file_rows[relative_config].get("sha256") != sha256_file(config_path):
        raise ValueError("Variant external config differs from its manifest hash.")
    return case_root, manifest


def analyze_run_pair(
    response_run_dir: Path,
    unloaded_run_dir: Path,
    *,
    response_config_path: Path,
    unloaded_config_path: Path,
    steady_window_s: float = DEFAULT_STEADY_WINDOW_S,
    wohler_exponent: float = DEFAULT_WOHLER_EXPONENT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    response_config_path = Path(response_config_path).resolve()
    unloaded_config_path = Path(unloaded_config_path).resolve()
    response_case_root, manifest = _case_variant_manifest(response_config_path, "response")
    unloaded_case_root, unloaded_manifest = _case_variant_manifest(
        unloaded_config_path, "unloaded"
    )
    timestep = _finite_float(manifest["qblade_mapping"]["timestep_s"], "timestep_s")
    duration = _finite_float(manifest["qblade_mapping"]["duration_s"], "duration_s")
    if not math.isclose(
        _finite_float(unloaded_manifest["qblade_mapping"]["timestep_s"], "baseline timestep"),
        timestep,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ) or not math.isclose(
        _finite_float(unloaded_manifest["qblade_mapping"]["duration_s"], "baseline duration"),
        duration,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Response and unloaded variants do not have identical time settings.")
    steps = _number_of_steps(duration, timestep)

    response_summary_path = _resolve_summary_file(Path(response_run_dir))
    unloaded_summary_path = _resolve_summary_file(Path(unloaded_run_dir))
    response_summary = load_json(response_summary_path)
    unloaded_summary = load_json(unloaded_summary_path)
    validate_run_summary(
        response_summary,
        expected_steps=steps,
        expected_timestep_s=timestep,
        coupled=True,
        config_path=response_config_path,
    )
    validate_run_summary(
        unloaded_summary,
        expected_steps=steps,
        expected_timestep_s=timestep,
        coupled=False,
    )
    response_sim = response_case_root / BASE_SIM_NAME
    unloaded_sim = unloaded_case_root / BASE_SIM_NAME
    _validate_input_hash(response_summary, "sim", response_sim)
    _validate_input_hash(unloaded_summary, "sim", unloaded_sim)
    if not math.isclose(_qblade_keyword_float(response_sim, "TIMESTEP"), timestep, abs_tol=1.0e-15):
        raise ValueError("Response .sim timestep differs from its variant manifest.")
    if _strict_integer(_qblade_keyword_float(response_sim, "NUMTIMESTEPS"), "NUMTIMESTEPS") != steps:
        raise ValueError("Response .sim step count differs from its variant manifest.")

    response_result_path = _resolve_run_artifact(
        Path(response_run_dir), response_summary.get("result_file"), "result_file"
    )
    unloaded_result_path = _resolve_run_artifact(
        Path(unloaded_run_dir), unloaded_summary.get("result_file"), "result_file"
    )
    history_block = response_summary.get("sil_vorlap_history")
    if not isinstance(history_block, Mapping):
        raise ValueError("Response run summary lacks SIL history provenance.")
    history_path = _resolve_run_artifact(
        Path(response_run_dir), history_block.get("path"), "SIL history"
    )
    response_result = read_qblade_result_strict(
        response_result_path,
        expected_steps=steps,
        expected_timestep_s=timestep,
    )
    unloaded_result = read_qblade_result_strict(
        unloaded_result_path,
        expected_steps=steps,
        expected_timestep_s=timestep,
    )
    if not np.array_equal(response_result.time_s, unloaded_result.time_s):
        raise ValueError("Response and unloaded QBlade result time grids differ.")
    config = load_json(response_config_path)
    node_ids = tuple(str(value) for value in config.get("node_ids", ()))
    sil = read_sil_history_strict(
        history_path,
        node_ids=node_ids,
        expected_steps=steps,
        expected_timestep_s=timestep,
    )
    _validate_aligned_time_grids(
        sil.time_s,
        response_result.time_s,
        label="SIL history and QBlade result",
        atol_s=1.0e-12,
    )
    replay = replay_response_model(response_config_path, sil)

    time = response_result.time_s
    response_tip = response_result.columns["Y_b~Tip~Trl.Def.~BLD_1~[m]"]
    unloaded_tip = unloaded_result.columns["Y_b~Tip~Trl.Def.~BLD_1~[m]"]
    paired_tip = response_tip - unloaded_tip
    response_root = response_result.columns["X_b~RootBend.~Mom.~BLD_1~[Nm]"]
    unloaded_root = unloaded_result.columns["X_b~RootBend.~Mom.~BLD_1~[Nm]"]
    paired_root = response_root - unloaded_root
    natural_frequency = _finite_float(
        manifest["response_aware_variant"]["natural_frequency_hz"],
        "identified natural frequency",
    )
    steady = _steady_mask(
        time,
        timestep_s=timestep,
        duration_s=duration,
        steady_window_s=steady_window_s,
    )
    response_tip_metrics, response_envelope = sinusoidal_signal_metrics(
        time[steady], response_tip[steady], frequency_hz=natural_frequency
    )
    unloaded_tip_metrics, unloaded_envelope = sinusoidal_signal_metrics(
        time[steady], unloaded_tip[steady], frequency_hz=natural_frequency
    )
    paired_tip_metrics, paired_envelope = sinusoidal_signal_metrics(
        time[steady], paired_tip[steady], frequency_hz=natural_frequency
    )
    paired_root_signal_metrics, _paired_root_envelope = sinusoidal_signal_metrics(
        time[steady], paired_root[steady], frequency_hz=natural_frequency
    )
    total_root_fatigue = load_channel_fatigue_metrics(
        response_root[steady],
        timestep_s=timestep,
        wohler_exponent=wohler_exponent,
        frequency_hz=natural_frequency,
    )
    paired_root_fatigue = load_channel_fatigue_metrics(
        paired_root[steady],
        timestep_s=timestep,
        wohler_exponent=wohler_exponent,
        frequency_hz=natural_frequency,
    )
    response_summary_block = response_summary["sil_vorlap_response_diagnostics_summary"]
    measured_amplitude = float(response_tip_metrics["fitted_amplitude"])
    expected_equilibrium = _finite_float(
        manifest["response_aware_variant"][
            "predicted_forced_motion_curve_intersection_m"
        ],
        "predicted equilibrium",
    )
    structural_power_coefficient = _finite_float(
        manifest["response_aware_variant"]["structural_power_coefficient_wpm2"],
        "structural power coefficient",
    )
    mean_target_power = float(np.mean(replay.target_cycle_mean_power_w[steady]))
    mean_generated_power = float(np.mean(replay.generated_response_power_w[steady]))
    mean_applied_power = float(np.mean(replay.applied_response_power_w[steady]))
    mean_mass_coordinate_lagged_work = float(
        np.mean(replay.mass_coordinate_lagged_work_w[steady])
    )
    work_coordinate_difference = _relative_difference(
        mean_applied_power,
        mean_mass_coordinate_lagged_work,
        "area-lift versus mass-coordinate lagged work",
    )
    mean_structural_power_at_measured_amplitude = (
        structural_power_coefficient * measured_amplitude**2
    )
    model_amplitude = float(np.mean(replay.amplitude_estimate_m[steady]))
    settling_qualification = evaluate_settling_qualification(
        time[steady],
        paired_tip[steady],
        paired_root[steady],
        replay.amplitude_estimate_m[steady],
        replay.target_cycle_mean_power_w[steady],
        replay.applied_response_power_w[steady],
        timestep_s=timestep,
        structural_power_coefficient_wpm2=structural_power_coefficient,
        wohler_exponent=wohler_exponent,
        frequency_hz=natural_frequency,
    )
    output_summary: dict[str, Any] = {
        "schema": "grinderslev2022-modal-power-analysis-v3",
        "scope": (
            "calibration-based structural coupling and energy-bookkeeping comparison; "
            "not independent wake prediction, validation, or certification evidence"
        ),
        "case": {
            "timestep_s": timestep,
            "duration_s": duration,
            "steps": steps,
            "steady_window_s": steady_window_s,
            "steady_samples": int(np.count_nonzero(steady)),
            "natural_frequency_hz": natural_frequency,
            "force_scale": 0.0,
            "response_model_type": RESPONSE_MODEL_TYPE,
            "power_curve_sha256": EXPECTED_POWER_CURVE_SHA256,
        },
        "response": {
            "tip_edgewise": response_tip_metrics,
            "unloaded_tip_edgewise": unloaded_tip_metrics,
            "paired_tip_edgewise": paired_tip_metrics,
            "paired_root_edgewise_moment": paired_root_signal_metrics,
            "response_model_mean_amplitude_estimate_m": model_amplitude,
            "maximum_force_replay_error_n": replay.maximum_force_replay_error_n,
            "maximum_inferred_base_generated_power_abs_w": float(
                np.max(np.abs(replay.inferred_base_generated_power_w))
            ),
            "maximum_inferred_base_applied_power_abs_w": float(
                np.max(np.abs(replay.inferred_base_applied_power_w))
            ),
        },
        "power": {
            "mean_target_cycle_power_w": mean_target_power,
            "mean_generated_response_work_w": mean_generated_power,
            "mean_actually_applied_response_work_w": mean_applied_power,
            "mean_mass_coordinate_lagged_work_w": mean_mass_coordinate_lagged_work,
            "area_lift_vs_mass_coordinate_work_diagnostic": {
                "area_lift_mean_lagged_work_w": mean_applied_power,
                "mass_coordinate_mean_lagged_work_w": (
                    mean_mass_coordinate_lagged_work
                ),
                "absolute_relative_difference": work_coordinate_difference,
                "tolerance": WORK_COORDINATE_DIAGNOSTIC_TOLERANCE,
                "passed": bool(
                    work_coordinate_difference
                    <= WORK_COORDINATE_DIAGNOSTIC_TOLERANCE
                ),
                "same_record_definition": (
                    "compare sum_i(F_i[n-1] dot v_i[n]) with "
                    "Q[n-1] qdot_raw[n] from the same replayed SIL history"
                ),
                "qualification_role": (
                    "cause-isolation diagnostic only; not a structural-damping "
                    "or response-amplitude acceptance gate"
                ),
            },
            "mean_structural_damping_power_at_measured_amplitude_w": (
                mean_structural_power_at_measured_amplitude
            ),
            "target_minus_applied_mean_w": mean_target_power - mean_applied_power,
            "positive_sign_convention": "force does work on current structural velocity",
        },
        "root_moment_fatigue": {
            "quantity": "edgewise blade-root bending-moment range",
            "units": "N m",
            "total": total_root_fatigue,
            "paired_response_minus_unloaded": paired_root_fatigue,
            "range_power_scope": (
                "relative numerical gate only; not Miner damage or fatigue life"
            ),
        },
        "published_comparisons": {
            "calibrated_curve_structural_intersection_m": expected_equilibrium,
            "measured_minus_calibrated_intersection_m": (
                measured_amplitude - expected_equilibrium
            ),
            "measured_relative_to_calibrated_intersection": (
                measured_amplitude / expected_equilibrium
            ),
            "published_forced_motion_intersection_m": (
                REFERENCE_FORCED_MOTION_INTERSECTION_M
            ),
            "measured_minus_published_forced_motion_intersection_m": (
                measured_amplitude - REFERENCE_FORCED_MOTION_INTERSECTION_M
            ),
            "published_fsi_amplitude_m": REFERENCE_FSI_AMPLITUDE_M,
            "measured_minus_separate_fsi_amplitude_m": (
                measured_amplitude - REFERENCE_FSI_AMPLITUDE_M
            ),
            "published_zero_damping_amplitude_m": REFERENCE_ZERO_DAMPING_AMPLITUDE_M,
            "published_fsi_power_w": REFERENCE_FSI_POWER_W,
            "published_forced_motion_intersection_power_w": (
                REFERENCE_FORCED_MOTION_INTERSECTION_POWER_W
            ),
            "comparison_boundary": (
                "The 1.18 m FSI result is not the calibration target; the tabulated "
                "forced-motion curve intersects the mapped structural damping near 1.25 m."
            ),
        },
        "limit_diagnostics": {
            counter: int(response_summary_block.get(counter, 0))
            for counter in LIMIT_COUNTERS
        },
        "settling_qualification": settling_qualification,
        "provenance": {
            "response_run_summary_sha256": sha256_file(response_summary_path),
            "unloaded_run_summary_sha256": sha256_file(unloaded_summary_path),
            "response_result_sha256": sha256_file(response_result_path),
            "unloaded_result_sha256": sha256_file(unloaded_result_path),
            "sil_history_sha256": sha256_file(history_path),
            "response_config_sha256": sha256_file(response_config_path),
            "response_manifest_sha256": sha256_file(response_case_root / MANIFEST_RELATIVE),
        },
    }
    table = {
        "time_s": time[steady],
        "response_tip_edgewise_m": response_tip[steady],
        "unloaded_tip_edgewise_m": unloaded_tip[steady],
        "paired_tip_edgewise_m": paired_tip[steady],
        "response_tip_envelope_m": response_envelope,
        "unloaded_tip_envelope_m": unloaded_envelope,
        "paired_tip_envelope_m": paired_envelope,
        "response_root_edgewise_moment_nm": response_root[steady],
        "unloaded_root_edgewise_moment_nm": unloaded_root[steady],
        "paired_root_edgewise_moment_nm": paired_root[steady],
        "target_cycle_mean_power_w": replay.target_cycle_mean_power_w[steady],
        "response_model_amplitude_estimate_m": replay.amplitude_estimate_m[steady],
        "generated_response_work_w": replay.generated_response_power_w[steady],
        "actually_applied_response_work_w": replay.applied_response_power_w[steady],
        "mass_coordinate_lagged_work_w": replay.mass_coordinate_lagged_work_w[steady],
        "stored_total_applied_work_w": sil.applied_power_w[steady],
        "inferred_base_applied_work_w": replay.inferred_base_applied_power_w[steady],
    }
    return output_summary, table


def compare_timestep_analyses(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
) -> dict[str, Any]:
    coarse_dt = _finite_float(coarse["case"]["timestep_s"], "coarse timestep")
    fine_dt = _finite_float(fine["case"]["timestep_s"], "fine timestep")
    if coarse_dt <= fine_dt:
        raise ValueError("Time-step gate requires the first analysis to be the coarser case.")
    if not math.isclose(
        _finite_float(coarse["case"]["duration_s"], "coarse duration"),
        _finite_float(fine["case"]["duration_s"], "fine duration"),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Coarse and fine analyses must have the same duration.")
    metrics = {
        "tip_amplitude": {
            "coarse": coarse["response"]["tip_edgewise"]["fitted_amplitude"],
            "fine": fine["response"]["tip_edgewise"]["fitted_amplitude"],
            "tolerance": AMPLITUDE_TIMESTEP_TOLERANCE,
        },
        "paired_root_moment_del": {
            "coarse": coarse["root_moment_fatigue"]["paired_response_minus_unloaded"][
                "damage_equivalent_range"
            ],
            "fine": fine["root_moment_fatigue"]["paired_response_minus_unloaded"][
                "damage_equivalent_range"
            ],
            "tolerance": DEL_TIMESTEP_TOLERANCE,
        },
        "paired_root_moment_range_power_sum": {
            "coarse": coarse["root_moment_fatigue"]["paired_response_minus_unloaded"][
                "rainflow_range_power_sum"
            ],
            "fine": fine["root_moment_fatigue"]["paired_response_minus_unloaded"][
                "rainflow_range_power_sum"
            ],
            "tolerance": RANGE_POWER_TIMESTEP_TOLERANCE,
        },
        "mean_actually_applied_response_work": {
            "coarse": coarse["power"]["mean_actually_applied_response_work_w"],
            "fine": fine["power"]["mean_actually_applied_response_work_w"],
            "tolerance": POWER_TIMESTEP_TOLERANCE,
        },
    }
    gate_rows = {}
    for name, row in metrics.items():
        difference = _relative_difference(row["coarse"], row["fine"], name)
        gate_rows[name] = {
            **row,
            "absolute_relative_difference": difference,
            "passed": bool(difference <= row["tolerance"]),
        }
    return {
        "schema": "grinderslev2022-modal-power-timestep-gate-v1",
        "coarse_timestep_s": coarse_dt,
        "fine_timestep_s": fine_dt,
        "gates": gate_rows,
        "passed": all(bool(row["passed"]) for row in gate_rows.values()),
        "fatigue_boundary": (
            "The range-power gate compares numerical rainflow accumulation only; "
            "it is not Miner damage, material qualification, or fatigue life."
        ),
    }


def combine_qualification_layers(
    coarse: Mapping[str, Any],
    fine: Mapping[str, Any],
    timestep_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Require both per-run settling gates and the separate time-step gate."""

    try:
        coarse_settling = coarse["settling_qualification"]
        fine_settling = fine["settling_qualification"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Per-run analyses lack settling qualification.") from exc
    if not isinstance(coarse_settling, Mapping) or not isinstance(fine_settling, Mapping):
        raise ValueError("Per-run settling qualification must be a mapping.")
    if not isinstance(timestep_gate, Mapping):
        raise ValueError("Time-step qualification must be a mapping.")

    layers = {
        "coarse_run_settling": bool(coarse_settling.get("passed") is True),
        "fine_run_settling": bool(fine_settling.get("passed") is True),
        "coarse_fine_timestep": bool(timestep_gate.get("passed") is True),
    }
    return {
        "schema": "grinderslev2022-modal-power-overall-qualification-v1",
        "layers": layers,
        "passed": all(layers.values()),
        "boundary": (
            "Per-run physical stationarity/closure and coarse/fine numerical "
            "convergence are independent required layers."
        ),
    }


def _write_numeric_table(path: Path, columns: Mapping[str, np.ndarray]) -> None:
    names = tuple(columns)
    arrays = [np.asarray(columns[name], dtype=float) for name in names]
    if not names or any(array.ndim != 1 for array in arrays):
        raise ValueError("Output table columns must be nonempty one-dimensional arrays.")
    row_count = arrays[0].size
    if any(array.size != row_count for array in arrays):
        raise ValueError("Output table columns must have equal length.")
    if any(np.any(~np.isfinite(array)) for array in arrays):
        raise ValueError("Output table columns must be finite.")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for row_index in range(row_count):
            writer.writerow([f"{array[row_index]:.17g}" for array in arrays])


def analyze_timestep_pair(
    *,
    coarse_response_run: Path,
    coarse_unloaded_run: Path,
    coarse_response_config: Path,
    coarse_unloaded_config: Path,
    fine_response_run: Path,
    fine_unloaded_run: Path,
    fine_response_config: Path,
    fine_unloaded_config: Path,
    output_dir: Path,
    steady_window_s: float = DEFAULT_STEADY_WINDOW_S,
    wohler_exponent: float = DEFAULT_WOHLER_EXPONENT,
) -> Path:
    coarse, coarse_table = analyze_run_pair(
        coarse_response_run,
        coarse_unloaded_run,
        response_config_path=coarse_response_config,
        unloaded_config_path=coarse_unloaded_config,
        steady_window_s=steady_window_s,
        wohler_exponent=wohler_exponent,
    )
    fine, fine_table = analyze_run_pair(
        fine_response_run,
        fine_unloaded_run,
        response_config_path=fine_response_config,
        unloaded_config_path=fine_unloaded_config,
        steady_window_s=steady_window_s,
        wohler_exponent=wohler_exponent,
    )
    timestep_gate = compare_timestep_analyses(coarse, fine)
    overall_qualification = combine_qualification_layers(coarse, fine, timestep_gate)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "coarse_analysis.json", coarse)
    write_json(output / "fine_analysis.json", fine)
    write_json(output / "timestep_gate.json", timestep_gate)
    write_json(output / "overall_qualification.json", overall_qualification)
    _write_numeric_table(output / "coarse_steady_histories.csv", coarse_table)
    _write_numeric_table(output / "fine_steady_histories.csv", fine_table)
    combined = {
        "schema": "grinderslev2022-modal-power-comparison-v3",
        "coarse": coarse,
        "fine": fine,
        "timestep_gate": timestep_gate,
        "overall_qualification": overall_qualification,
    }
    combined_path = output / "comparison.json"
    write_json(combined_path, combined)
    if not overall_qualification["passed"]:
        failed = []
        for prefix, analysis in (("coarse_settling", coarse), ("fine_settling", fine)):
            settling = analysis["settling_qualification"]
            failed.extend(
                f"{prefix}:{name}"
                for name, row in settling["gates"].items()
                if not row["passed"]
            )
        failed.extend(
            f"timestep:{name}"
            for name, row in timestep_gate["gates"].items()
            if not row["passed"]
        )
        raise RuntimeError(
            "Response benchmark overall qualification failed: " + ", ".join(failed)
        )
    return combined_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Create independent coarse/fine case copies.")
    prepare.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "run_headless"
            / "runs"
            / "grinderslev2022_modal_power_inputs_qblade_mode_120s_v2"
        ),
    )
    prepare.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)

    analyze = subparsers.add_parser("analyze", help="Analyze completed paired coarse/fine runs.")
    for prefix in ("coarse", "fine"):
        analyze.add_argument(f"--{prefix}-response-run", type=Path, required=True)
        analyze.add_argument(f"--{prefix}-unloaded-run", type=Path, required=True)
        analyze.add_argument(f"--{prefix}-response-config", type=Path, required=True)
        analyze.add_argument(f"--{prefix}-unloaded-config", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--steady-window-s", type=float, default=DEFAULT_STEADY_WINDOW_S)
    analyze.add_argument("--wohler-exponent", type=float, default=DEFAULT_WOHLER_EXPONENT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        result = prepare_response_cases(args.output_root, duration_s=args.duration_s)
    elif args.command == "analyze":
        result = analyze_timestep_pair(
            coarse_response_run=args.coarse_response_run,
            coarse_unloaded_run=args.coarse_unloaded_run,
            coarse_response_config=args.coarse_response_config,
            coarse_unloaded_config=args.coarse_unloaded_config,
            fine_response_run=args.fine_response_run,
            fine_unloaded_run=args.fine_unloaded_run,
            fine_response_config=args.fine_response_config,
            fine_unloaded_config=args.fine_unloaded_config,
            output_dir=args.output_dir,
            steady_window_s=args.steady_window_s,
            wohler_exponent=args.wohler_exponent,
        )
    else:  # pragma: no cover - argparse restricts commands
        raise RuntimeError(f"Unsupported command: {args.command}")
    print(result)


if __name__ == "__main__":
    main()
