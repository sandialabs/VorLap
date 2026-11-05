#!/usr/bin/env python3
"""Immutable stationarity protocol for the Vlastos response-aware campaign.

This module is deliberately independent of ``response_benchmark.py``. It binds
a generated 72-case campaign (three structures, twelve reduced velocities, and
two time steps) to frozen, conservative fixed-duration caps without editing any
generated input. Each case is run to its cap and then subjected to terminal
stationarity and runtime-evidence checks by ``response_benchmark.py``. This is
not an adaptive stop/continuation driver.

The duration cap for each structure and reduced velocity is derived from the
frozen analytical modal envelope.  With ``a=A/D``, saturation amplitude
``a_s``, structural damping ``zeta``, and smooth activation ``p(U*)``, define

``R = U* H1 p(U*) / (2 Sc_QB)`` and
``g = 2 pi zeta (R - 1)`` per natural cycle.

For ``R > 1``, the squared-amplitude logistic solution is

``a(N) = a_eq / sqrt(1 + ((a_eq/a0)^2 - 1) exp(-2 g N))``,
``a_eq = a_s sqrt(1 - 1/R)``.

Growth is planned from the conservative realized seed floor ``a0=0.001`` to
within ``max(0.005, 0.05*a_eq)`` of equilibrium.  For ``R <= 1``, the stable
branch is instead planned to decay from ``a=0.1`` to ``a=0.005``.  The stable
initial bound is ten times the requested seed upper sensitivity and protects
against startup/off-branch contamination.  Forty final observation cycles are
then appended, the result is rounded upward to a 40-cycle boundary, and a
240-cycle floor is applied.  These are planning bounds, not proof of
stationarity: a failed gate at the cap is ``incomplete_nonstationary``.

Nominal and half-step cases end at the same integer-nanosecond endpoint. The
fine case therefore uses exactly twice the nominal step count. The optional
decision-ledger/publication-plan prototypes remain disabled until an analyzer
can derive a fixed gate schema directly from hash-bound result artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import sys
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np


CAMPAIGN_SCHEMA = "vlastos2026-response-benchmark-v1"
PROTOCOL_SCHEMA = "vlastos2026-stationary-campaign-protocol-v1"
LEDGER_RECORD_SCHEMA = "vlastos2026-stationarity-decision-v1"
PUBLICATION_PLAN_SCHEMA = "vlastos2026-fixed-duration-publication-plan-v1"
ADAPTIVE_DECISION_SCHEMA_ENABLED = False

COARSE_TIMESTEP_NS = 500_000
FINE_TIMESTEP_NS = 250_000
COARSE_TIMESTEP_S = COARSE_TIMESTEP_NS / 1.0e9
FINE_TIMESTEP_S = FINE_TIMESTEP_NS / 1.0e9
FIRST_EVALUATION_CYCLES = 200
EXTENSION_CYCLES = 40
CONSECUTIVE_PASSING_ENDPOINTS = 2
EARLIEST_ACCEPTANCE_CYCLES = 240
FINAL_OBSERVATION_CYCLES = 40
ACTIVE_SEED_LOWER_BOUND_RATIO = 0.001
REQUESTED_SEED_TARGET_RATIO = 0.005
REQUESTED_SEED_UPPER_SENSITIVITY_RATIO = 0.010
STABLE_INITIAL_AMPLITUDE_RATIO = 0.100
TERMINAL_ABSOLUTE_TOLERANCE_RATIO = 0.005
TERMINAL_RELATIVE_TOLERANCE = 0.05
STARTUP_SEED_DURATION_CYCLES = 5.0
HEIGHT_M = 0.775
DIAMETER_M = 0.050
AIR_DENSITY_KGPM3 = 1.225
AIR_KINEMATIC_VISCOSITY_M2PS = 1.48e-5
N_STATIONS = 21
EULER_BERNOULLI_BETA1 = 1.875104068711961
RMS_TIME_CONSTANT_CYCLES = 2.0
BANDPASS_DAMPING_RATIO = 0.1
MAX_ABS_GAIN = 10.0
MAX_ABS_NODAL_FORCE_N = 25.0
AMBIENT_V2_EQUATION_GATE_SCHEMA = (
    "vlastos2026-ambient-response-equation-reproduction-v1"
)
AMBIENT_V2_AERODYNAMIC_SPEED_BASIS = "declared_ambient_crossflow"
AMBIENT_V2_CASE_ID_NAMESPACE = "ambientv2"
AMBIENT_V2_CHECKPOINT_REUSE_POLICY = (
    "response_schema_and_config_fingerprint_bound_no_cross_schema_reuse"
)
AMBIENT_V2_CONFIGURED_REDUCED_VELOCITY_EQUATION = (
    "Ustar=Uresponse/(natural_frequency_hz*reference_length_m)"
)
AMBIENT_V2_CALIBRATION_QBLADE_SCRUTON_NUMBER = 4.953017025691645
AMBIENT_V2_FINE_ESTIMATOR_RMSE_TOLERANCE = 0.005
AMBIENT_V2_FINE_ESTIMATOR_MAX_ABS_TOLERANCE = 0.010
AMBIENT_V2_COARSE_FINE_ESTIMATOR_TOLERANCE = 0.005
AMBIENT_V2_ACTIVE_AMPLITUDE_RATIO = 0.05
AMBIENT_V2_EXPECTED_ACTIVE_REDUCED_VELOCITIES = (5.8, 6.2, 6.25, 6.5)
AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256 = (
    "330242617d9d04d2cb771767fefbedac6414bea9a04a9b6ea9e813c0aba18c7d"
)
AMBIENT_V2_RUN_PLAN_SCHEMA = "vlastos2026-ambient-response-run-plan-v1"
AMBIENT_V2_RUNTIME_CONTRACT_SCHEMA = (
    "vlastos2026-ambient-response-runtime-contract-v1"
)
AMBIENT_V2_RESULT_FILTER_IDENTITY = {
    "path": "cantilever_results.filter",
    "sha256": "04959598a0858889fcd1dab926d535e66bbae6d2c782bc1102822a866a595733",
    "bytes": 239,
}
AMBIENT_V2_REFERENCE_IDENTITIES = {
    "sc5p18_figure9_digitized.csv": {
        "path": "reference_data/sc5p18_figure9_digitized.csv",
        "sha256": "674d8bb8f3b932f97bd2a177717e3042a1a0ba0b7c81cc915d51086f2ab550a1",
        "bytes": 1766,
    },
    "all_sc_figure10_peak_digitized.csv": {
        "path": "reference_data/all_sc_figure10_peak_digitized.csv",
        "sha256": "116e1d63eb6983cede7956280061b523b868926700dd999c5d1446f631ddfa7d",
        "bytes": 506,
    },
    "vlastos2026_model_parameters.csv": {
        "path": "reference_data/vlastos2026_model_parameters.csv",
        "sha256": "b9bb5d03ca43e8d1dfa75059da54c0f11e3e68217d64dcad79f04b22cbdfd488",
        "bytes": 269,
    },
}
AMBIENT_V2_QBLADE_LIBRARY_IDENTITY = {
    "path": "libQBladeCE_2.0.9.7.so.1.0.0",
    "sha256": "7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d",
    "bytes": 18_365_008,
}
AMBIENT_V2_BRIDGE_LIBRARY_IDENTITY = {
    "path": "ControllerFiles/libvorlap_qblade_bridge.so",
    "sha256": "63401df29cf74bc41f5395be0d0ded40f5d371c228a6ffe1b77ad550220769c8",
    "bytes": 65_512,
}
AMBIENT_V2_QBLADE_SIL_WRAPPER_IDENTITY = {
    "path": "SIL_Interface/QBladeLibrary.py",
    "sha256": "109ef49481381edd715a62033c56163ac7d87661a02f34d4226c71d98d6928bb",
    "bytes": 5_628,
}
AMBIENT_V2_CHRONO_ENGINE_IDENTITY = {
    "path": "Libraries/libChronoEngine.so",
    "sha256": "c4f0a5a7a8b24c5efabe27fb1dfddffc87ebb591679989660d60bba6dcb2bf0b",
    "bytes": 15_201_504,
}

AERODYNAMIC_DAMPING_GAIN_H1 = 3.920125762951649
SATURATION_AMPLITUDE_RATIO = 0.2968375879119047
REDUCED_VELOCITY_LOW = 5.3625308212165494
REDUCED_VELOCITY_PEAK = 6.245825
REDUCED_VELOCITY_HIGH = 6.982801851062782

REDUCED_VELOCITIES = (
    4.30,
    4.80,
    5.00,
    5.20,
    5.50,
    5.80,
    6.20,
    6.25,
    6.50,
    6.80,
    7.20,
    8.20,
)

# These values are the actual QBlade energy-balance properties, not merely the
# rounded published Sc labels.  The natural-frequency numerator/denominator
# pairs make shared endpoint step counts exact and independent of float roundoff.
STRUCTURES = {
    2.94: {
        "qblade_implied_scruton_number": 2.8448211418563956,
        "structural_damping_ratio": 0.0030,
        "natural_frequency_hz": 20.02,
        "natural_frequency_fraction": (1001, 50),
        "equivalent_mass_kg_per_m": 0.2311,
        "role": "held_out",
    },
    5.18: {
        "qblade_implied_scruton_number": 4.953017025691645,
        "structural_damping_ratio": 0.0028,
        "natural_frequency_hz": 15.38,
        "natural_frequency_fraction": (769, 50),
        "equivalent_mass_kg_per_m": 0.4311,
        "role": "calibration",
    },
    7.02: {
        "qblade_implied_scruton_number": 6.677266689645888,
        "structural_damping_ratio": 0.0027,
        "natural_frequency_hz": 13.18,
        "natural_frequency_fraction": (659, 50),
        "equivalent_mass_kg_per_m": 0.6027,
        "role": "held_out",
    },
}

# Frozen cycle caps.  Sc=5.18 reproduces the campaign-duration-design table;
# the held-out caps use the same equations and frozen closure without retuning.
FROZEN_MAXIMUM_CYCLES = {
    2.94: {
        4.30: 240,
        4.80: 240,
        5.00: 240,
        5.20: 240,
        5.50: 280,
        5.80: 400,
        6.20: 240,
        6.25: 240,
        6.50: 240,
        6.80: 600,
        7.20: 240,
        8.20: 240,
    },
    5.18: {
        4.30: 240,
        4.80: 240,
        5.00: 240,
        5.20: 240,
        5.50: 240,
        5.80: 2520,
        6.20: 320,
        6.25: 320,
        6.50: 480,
        6.80: 360,
        7.20: 240,
        8.20: 240,
    },
    7.02: {
        4.30: 240,
        4.80: 240,
        5.00: 240,
        5.20: 240,
        5.50: 240,
        5.80: 1080,
        6.20: 520,
        6.25: 520,
        6.50: 1000,
        6.80: 320,
        7.20: 240,
        8.20: 240,
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GENESIS_SHA256 = "0" * 64
_FLOAT_ABS_TOL = 1.0e-12
_QBLADE_KEYVALUE_RE = re.compile(
    r"^(?P<value>.*?)\s+(?P<keyword>[A-Z0-9][A-Z0-9_]+)\s*(?P<rest>-.*)?$"
)


class ProtocolError(ValueError):
    """Raised when immutable protocol evidence is missing or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique byte representation used for every payload hash."""

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Payload is not finite canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if _SHA256_RE.fullmatch(text) is None:
        raise ProtocolError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def _require_close(actual: Any, expected: float, label: str) -> float:
    if isinstance(actual, bool):
        raise ProtocolError(f"{label} must be numeric, not boolean.")
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} must be numeric.") from exc
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL
    ):
        raise ProtocolError(f"{label}={value!r} does not match frozen value {expected!r}.")
    return value


def _require_rendered_close(
    actual: Any,
    expected: float,
    label: str,
    *,
    relative_tolerance: float,
) -> float:
    """Require agreement at the precision used by a rendered QBlade input."""

    if isinstance(actual, bool):
        raise ProtocolError(f"{label} must be numeric, not boolean.")
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} must be numeric.") from exc
    if not math.isfinite(value) or not math.isclose(
        value,
        expected,
        rel_tol=relative_tolerance,
        abs_tol=_FLOAT_ABS_TOL,
    ):
        raise ProtocolError(f"{label}={value!r} does not match frozen value {expected!r}.")
    return value


def _match_frozen_value(raw: Any, values: Sequence[float], label: str) -> float:
    if isinstance(raw, bool):
        raise ProtocolError(f"{label} must be numeric, not boolean.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} must be numeric.") from exc
    matches = [
        expected
        for expected in values
        if math.isclose(value, expected, rel_tol=0.0, abs_tol=_FLOAT_ABS_TOL)
    ]
    if len(matches) != 1:
        raise ProtocolError(f"{label}={value!r} is not one frozen protocol value.")
    return matches[0]


def _expected_mode_shape() -> List[float]:
    beta = EULER_BERNOULLI_BETA1
    sigma = (math.cosh(beta) + math.cos(beta)) / (
        math.sinh(beta) + math.sin(beta)
    )
    values = []
    for index in range(N_STATIONS):
        x = index / (N_STATIONS - 1)
        values.append(
            math.cosh(beta * x)
            - math.cos(beta * x)
            - sigma * (math.sinh(beta * x) - math.sin(beta * x))
        )
    return [value / values[-1] for value in values]


def _expected_projection_weights(equivalent_mass_kg_per_m: float) -> List[float]:
    tributary_length = HEIGHT_M / (N_STATIONS - 1)
    return [
        equivalent_mass_kg_per_m
        * tributary_length
        * (0.5 if index in {0, N_STATIONS - 1} else 1.0)
        for index in range(N_STATIONS)
    ]


def _require_numeric_sequence(
    raw: Any, expected: Sequence[float], label: str
) -> List[float]:
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise ProtocolError(f"{label} must contain exactly {len(expected)} values.")
    values = []
    for index, (actual, target) in enumerate(zip(raw, expected)):
        values.append(_require_close(actual, float(target), f"{label}[{index}]"))
    return values


def _parse_required_qblade_settings(
    path: Path, keywords: Sequence[str], label: str
) -> Dict[str, str]:
    parsed: Dict[str, List[str]] = {}
    try:
        with Path(path).open("r", encoding="utf-8", errors="strict") as handle:
            for raw_line in handle:
                text = raw_line.strip()
                if not text or re.match(r"^-{3,}", text):
                    continue
                match = _QBLADE_KEYVALUE_RE.match(text)
                if match is None:
                    continue
                parsed.setdefault(match.group("keyword"), []).append(
                    match.group("value").strip()
                )
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"Could not parse {label} {path}: {exc}") from exc
    result = {}
    for keyword in keywords:
        values = parsed.get(keyword, [])
        if len(values) != 1 or not values[0]:
            raise ProtocolError(
                f"{label} must contain exactly one nonempty {keyword} setting."
            )
        result[keyword] = values[0]
    return result


def _require_blank_qblade_settings(
    path: Path, keywords: Sequence[str], label: str
) -> None:
    values: Dict[str, List[str]] = {keyword: [] for keyword in keywords}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"Could not parse {label} {path}: {exc}") from exc
    keyword_set = set(keywords)
    for raw in lines:
        body = raw.split(" - ", 1)[0].strip()
        fields = body.split()
        if fields and fields[-1] in keyword_set:
            values[fields[-1]].append(" ".join(fields[:-1]))
    for keyword in keywords:
        if values[keyword] != [""]:
            raise ProtocolError(
                f"{label} must contain exactly one blank {keyword} setting."
            )


def _validate_external_library_slots(path: Path, case_id: str) -> None:
    pattern = re.compile(
        r"\b(?P<keyword>LIB(?:FILE|FUNCTION|ARRAYSIZE|PARAMETERFILE)_(?P<slot>\d+))\b"
    )
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(
            f"Could not read Case {case_id} external-library slots: {exc}"
        ) from exc
    observed = [match.group("keyword") for match in pattern.finditer(text)]
    expected = [
        "LIBFILE_1",
        "LIBFUNCTION_1",
        "LIBARRAYSIZE_1",
        "LIBPARAMETERFILE_1",
    ]
    if sorted(observed) != sorted(expected):
        raise ProtocolError(
            f"Case {case_id} must contain exactly one complete external-library slot 1."
        )


def _require_integer_text(raw: Any, expected: int, label: str) -> int:
    try:
        numeric = float(str(raw).strip())
        value = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(f"{label} must be an integer.") from exc
    if numeric != value or value != expected:
        raise ProtocolError(f"{label}={raw!r} does not match {expected}.")
    return value


def _require_text(raw: Any, expected: str, label: str) -> str:
    value = str(raw).strip()
    if value != expected:
        raise ProtocolError(f"{label}={value!r} does not match {expected!r}.")
    return value


def _require_boolean_text(raw: Any, expected: bool, label: str) -> bool:
    value = str(raw).strip().lower()
    expected_text = "true" if expected else "false"
    if value != expected_text:
        raise ProtocolError(f"{label}={value!r} does not match {expected_text!r}.")
    return expected


def _require_numeric_text_sequence(
    raw: Any, expected: Sequence[float], label: str
) -> List[float]:
    fields = str(raw).split()
    if len(fields) != len(expected):
        raise ProtocolError(f"{label} must contain exactly {len(expected)} values.")
    values: List[float] = []
    for index, (field, target) in enumerate(zip(fields, expected)):
        values.append(_require_close(field, float(target), f"{label}[{index}]"))
    return values


def _validate_single_value_settings(
    settings: Mapping[str, str],
    *,
    numeric: Mapping[str, float] | None = None,
    integer: Mapping[str, int] | None = None,
    boolean: Mapping[str, bool] | None = None,
    text: Mapping[str, str] | None = None,
    label: str,
) -> None:
    for keyword, expected in (numeric or {}).items():
        _require_close(settings[keyword], expected, f"{label} {keyword}")
    for keyword, expected in (integer or {}).items():
        _require_integer_text(settings[keyword], expected, f"{label} {keyword}")
    for keyword, expected in (boolean or {}).items():
        _require_boolean_text(settings[keyword], expected, f"{label} {keyword}")
    for keyword, expected in (text or {}).items():
        _require_text(settings[keyword], expected, f"{label} {keyword}")


def _resolve_linked_manifest_file(
    root: Path,
    file_index: Mapping[str, Mapping[str, Any]],
    *,
    base_relative: str,
    raw_link: Any,
    case_prefix: str,
    label: str,
) -> Tuple[Path, str, Dict[str, Any]]:
    link = Path(str(raw_link).strip())
    if not str(link) or link.is_absolute() or ".." in link.parts:
        raise ProtocolError(f"{label} is an unsafe linked path.")
    relative = (Path(base_relative).parent / link).as_posix()
    if not relative.startswith(case_prefix):
        raise ProtocolError(f"{label} escapes its case directory.")
    identity = _verify_file_identity(root, file_index, relative, label)
    path, normalized = _safe_manifest_path(root, relative, label)
    return path, normalized, identity


def _numeric_table_after_header(
    path: Path,
    *,
    header_prefix: str,
    expected_rows: int,
    expected_columns: int,
    label: str,
) -> List[List[float]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(f"Could not read {label} {path}: {exc}") from exc
    indices = [
        index
        for index, line in enumerate(lines)
        if line.strip().startswith(header_prefix)
    ]
    if len(indices) != 1:
        raise ProtocolError(f"{label} must contain one {header_prefix!r} table header.")
    raw_rows = [line.strip() for line in lines[indices[0] + 1 :] if line.strip()]
    if len(raw_rows) != expected_rows:
        raise ProtocolError(
            f"{label} must contain exactly {expected_rows} table rows."
        )
    rows: List[List[float]] = []
    for row_index, raw_row in enumerate(raw_rows):
        fields = raw_row.split()
        if len(fields) < expected_columns:
            raise ProtocolError(
                f"{label} row {row_index} has fewer than {expected_columns} columns."
            )
        try:
            numeric = [float(value) for value in fields[:expected_columns]]
        except ValueError as exc:
            raise ProtocolError(
                f"{label} row {row_index} contains a nonnumeric value."
            ) from exc
        if any(not math.isfinite(value) for value in numeric):
            raise ProtocolError(f"{label} row {row_index} is non-finite.")
        rows.append(numeric)
    return rows


def _validate_structural_output_nodes(path: Path, case_id: str) -> None:
    try:
        lines = [
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(
            f"Could not read Case {case_id} structural output nodes: {exc}"
        ) from exc
    observed_blade = [line for line in lines if line.startswith("BLD_1_")]
    expected_blade = [
        f"BLD_1_{index / (N_STATIONS - 1):.10f}" for index in range(N_STATIONS)
    ]
    if observed_blade != expected_blade:
        raise ProtocolError(
            f"Case {case_id} structural blade output nodes do not match the "
            "21-station controller mapping."
        )
    observed_tower = [line for line in lines if line.startswith("TWR_")]
    if observed_tower != ["TWR_0.0", "TWR_1.0"]:
        raise ProtocolError(
            f"Case {case_id} structural tower output nodes are not frozen."
        )


def _expected_external_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    input_rows: List[Dict[str, Any]] = [
        {"index": 0, "variable": "Time [s]"},
        {"index": 1, "variable": "Timestep [-]"},
        {"index": 2, "variable": "LSS Azimuthal Pos. [deg]"},
    ]
    output_rows: List[Dict[str, Any]] = []
    input_index = 3
    output_index = 3 + 3 * N_STATIONS
    for station in range(N_STATIONS):
        position = station / (N_STATIONS - 1)
        for direction in ("X", "Y", "Z"):
            input_rows.append(
                {
                    "index": input_index,
                    "variable": f"{direction}_g Vel. BLD_1 pos {position:.3f} [m/s]",
                }
            )
            output_rows.append(
                {
                    "action": "ADDFORCE",
                    "direction": direction,
                    "id": "BLD_1",
                    "index": output_index,
                    "local": False,
                    "position": position,
                }
            )
            input_index += 1
            output_index += 1
    return input_rows, output_rows


def _validate_structural_external_tables(path: Path, case_id: str) -> None:
    try:
        lines = [
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        raise ProtocolError(
            f"Could not read Case {case_id} structural external tables: {exc}"
        ) from exc
    input_markers = [index for index, line in enumerate(lines) if line == "EXTERNAL_1_IN"]
    output_markers = [
        index for index, line in enumerate(lines) if line == "EXTERNAL_1_OUT"
    ]
    if (
        len(input_markers) != 1
        or len(output_markers) != 1
        or input_markers[0] >= output_markers[0]
    ):
        raise ProtocolError(
            f"Case {case_id} structural external-table markers are invalid."
        )
    observed_inputs: List[Dict[str, Any]] = []
    for raw in lines[input_markers[0] + 1 : output_markers[0]]:
        if raw == "SWAP DATA":
            continue
        fields = shlex.split(raw)
        if len(fields) != 2:
            raise ProtocolError(
                f"Case {case_id} structural external-input row is malformed."
            )
        observed_inputs.append(
            {
                "index": _require_integer_text(
                    fields[0], len(observed_inputs), f"Case {case_id} input SWAP index"
                ),
                "variable": fields[1],
            }
        )
    observed_outputs: List[Dict[str, Any]] = []
    expected_inputs, expected_outputs = _expected_external_rows()
    for raw in lines[output_markers[0] + 1 :]:
        fields = shlex.split(raw)
        if fields == ["SWAP", "ACTION", "ID", "POS", "DIR", "LOCAL"]:
            continue
        if len(fields) != 6:
            raise ProtocolError(
                f"Case {case_id} structural external-output row is malformed."
            )
        observed_outputs.append(
            {
                "index": _require_integer_text(
                    fields[0],
                    3 + 3 * N_STATIONS + len(observed_outputs),
                    f"Case {case_id} output SWAP index",
                ),
                "action": fields[1],
                "id": fields[2],
                "position": _require_rendered_close(
                    fields[3],
                    expected_outputs[len(observed_outputs)]["position"]
                    if len(observed_outputs) < len(expected_outputs)
                    else math.nan,
                    f"Case {case_id} output position",
                    relative_tolerance=5.0e-10,
                ),
                "direction": fields[4],
                "local": _require_boolean_text(
                    fields[5], False, f"Case {case_id} output LOCAL"
                ),
            }
        )
    if observed_inputs != expected_inputs:
        raise ProtocolError(
            f"Case {case_id} structural EXTERNAL_1_IN table is not frozen."
        )
    if observed_outputs != expected_outputs:
        raise ProtocolError(
            f"Case {case_id} structural EXTERNAL_1_OUT table is not frozen."
        )


def _validate_structural_and_blade_files(
    *,
    root: Path,
    file_index: Mapping[str, Mapping[str, Any]],
    turbine_path: Path,
    turbine_relative: str,
    parameter_file: str,
    case_prefix: str,
    structure: Mapping[str, Any],
    case_id: str,
) -> Dict[str, str]:
    turbine = _parse_required_qblade_settings(
        turbine_path,
        (
            "BLADEFILE",
            "TURBTYPE",
            "NUMBLADES",
            "ROTORCONFIG",
            "ROTATIONALDIR",
            "DISCTYPE",
            "INTPTYPE",
            "NUMPANELS",
            "OVERHANG",
            "SHAFTTILT",
            "ROTORCONE",
            "TOWERHEIGHT",
            "TOWERTOPRAD",
            "TOWERBOTRAD",
            "UNSTEADYAERO",
            "2PLIFTDRAG",
            "HIMMELSKAMP",
            "TOWERSHADOW",
            "WAKETYPE",
            "STRUCTURALFILE",
            "GEOMSTIFFNESS",
            "AEROPANELLOADS",
            "CONTROLLERTYPE",
            "LIBFILE_1",
            "LIBFUNCTION_1",
            "LIBARRAYSIZE_1",
            "LIBPARAMETERFILE_1",
        ),
        f"Case {case_id} turbine input",
    )
    _validate_external_library_slots(turbine_path, case_id)
    _validate_single_value_settings(
        turbine,
        numeric={
            "OVERHANG": 0.0,
            "SHAFTTILT": 0.0,
            "ROTORCONE": 0.0,
            "TOWERHEIGHT": 1.0,
            "TOWERTOPRAD": 0.5,
            "TOWERBOTRAD": 0.5,
        },
        integer={
            "TURBTYPE": 0,
            "NUMBLADES": 1,
            "ROTORCONFIG": 0,
            "ROTATIONALDIR": 0,
            "DISCTYPE": 0,
            "INTPTYPE": 0,
            "NUMPANELS": N_STATIONS - 1,
            "WAKETYPE": 1,
            "CONTROLLERTYPE": 0,
            "LIBARRAYSIZE_1": 3 + 6 * N_STATIONS,
        },
        boolean={
            "UNSTEADYAERO": False,
            "2PLIFTDRAG": True,
            "HIMMELSKAMP": False,
            "TOWERSHADOW": False,
            "GEOMSTIFFNESS": False,
            "AEROPANELLOADS": False,
        },
        text={
            "LIBFILE_1": "libvorlap_qblade_bridge",
            "LIBFUNCTION_1": "update",
        },
        label=f"Case {case_id} turbine",
    )
    config_link, config_relative, config_identity = _resolve_linked_manifest_file(
        root,
        file_index,
        base_relative=turbine_relative,
        raw_link=turbine["LIBPARAMETERFILE_1"],
        case_prefix=case_prefix,
        label=f"Case {case_id} turbine LIBPARAMETERFILE_1",
    )
    del config_link
    if config_relative != parameter_file:
        raise ProtocolError(
            f"Case {case_id} turbine library parameter file does not match the "
            "validated VorLap config."
        )

    main_path, main_relative, main_identity = _resolve_linked_manifest_file(
        root,
        file_index,
        base_relative=turbine_relative,
        raw_link=turbine["STRUCTURALFILE"],
        case_prefix=case_prefix,
        label=f"Case {case_id} turbine STRUCTURALFILE",
    )
    main = _parse_required_qblade_settings(
        main_path,
        (
            "GLBGEOEPS",
            "PRECONE",
            "SHFTTILT",
            "OVERHANG",
            "TWR2SHFT",
            "YAWBRMASS",
            "NACMASS",
            "NACCMX",
            "NACCMY",
            "NACCMZ",
            "NACINER",
            "NACYINER",
            "HUBMASS",
            "HUBCM",
            "HUBINER",
            "GBRATIO",
            "GBOXEFF",
            "GENEFF",
            "DRTRDOF",
            "GENINER",
            "DTTORSPR",
            "DTTORDMP",
            "NUMBLD",
            "BLDFILE_1",
            "TWRHEIGHT",
            "TWRFILE",
            "FOR_OUT",
            "ROT_OUT",
            "MOM_OUT",
            "DEF_OUT",
            "POS_OUT",
            "VEL_OUT",
            "ACC_OUT",
            "LVE_OUT",
            "LAC_OUT",
        ),
        f"Case {case_id} main structure",
    )
    _validate_single_value_settings(
        main,
        numeric={
            "GLBGEOEPS": 5.0e-4,
            "PRECONE": 0.0,
            "SHFTTILT": 0.0,
            "OVERHANG": 0.0,
            "TWR2SHFT": 0.0,
            "YAWBRMASS": 0.0,
            "NACMASS": 0.0,
            "NACCMX": 0.0,
            "NACCMY": 0.0,
            "NACCMZ": 0.0,
            "NACYINER": 0.0,
            "HUBMASS": 0.0,
            "HUBCM": 0.0,
            "HUBINER": 1.0,
            "GBRATIO": 1.0,
            "GBOXEFF": 1.0,
            "GENEFF": 1.0,
            "GENINER": 1.0,
            "DTTORSPR": 1.0e12,
            "DTTORDMP": 1.0e9,
            "TWRHEIGHT": 1.0,
        },
        integer={"NUMBLD": 1},
        boolean={
            "DRTRDOF": False,
            "FOR_OUT": True,
            "ROT_OUT": True,
            "MOM_OUT": True,
            "DEF_OUT": True,
            "POS_OUT": True,
            "VEL_OUT": True,
            "ACC_OUT": True,
            "LVE_OUT": True,
            "LAC_OUT": True,
        },
        label=f"Case {case_id} main structure",
    )
    _require_numeric_text_sequence(
        main["NACINER"], [0.0] * 6, f"Case {case_id} main structure NACINER"
    )
    _validate_structural_output_nodes(main_path, case_id)
    _validate_structural_external_tables(main_path, case_id)
    beam_path, beam_relative, beam_identity = _resolve_linked_manifest_file(
        root,
        file_index,
        base_relative=main_relative,
        raw_link=main["BLDFILE_1"],
        case_prefix=case_prefix,
        label=f"Case {case_id} main BLDFILE_1",
    )
    beam_settings = _parse_required_qblade_settings(
        beam_path,
        (
            "RAYLEIGHDMP",
            "DISC",
            "STIFFTUNER",
            "MASSTUNER",
            "INTPTYPE",
            "BEAMTYPE",
            "DISCTYPE",
        ),
        f"Case {case_id} beam structure",
    )
    _validate_single_value_settings(
        beam_settings,
        numeric={"STIFFTUNER": 1.0, "MASSTUNER": 1.0},
        integer={
            "DISC": N_STATIONS,
            "INTPTYPE": 0,
            "BEAMTYPE": 1,
            "DISCTYPE": 2,
        },
        label=f"Case {case_id} beam structure",
    )
    natural_frequency_hz = float(structure["natural_frequency_hz"])
    damping_ratio = float(structure["structural_damping_ratio"])
    mass_per_length = float(structure["equivalent_mass_kg_per_m"])
    _require_rendered_close(
        beam_settings["RAYLEIGHDMP"],
        damping_ratio / (math.pi * natural_frequency_hz),
        f"Case {case_id} beam RAYLEIGHDMP",
        relative_tolerance=5.0e-12,
    )
    beam_rows = _numeric_table_after_header(
        beam_path,
        header_prefix="LENFRACT_[-]",
        expected_rows=N_STATIONS,
        expected_columns=18,
        label=f"Case {case_id} beam structure",
    )
    target_ei = mass_per_length * (
        (2.0 * math.pi * natural_frequency_hz)
        * HEIGHT_M**2
        / EULER_BERNOULLI_BETA1**2
    ) ** 2
    expected_fraction = [index / (N_STATIONS - 1) for index in range(N_STATIONS)]
    for row_index, row in enumerate(beam_rows):
        expected_row = (
            expected_fraction[row_index],
            mass_per_length,
            target_ei,
            target_ei,
            2.0e7,
            90.0,
            7.0e6,
            0.0,
            0.85,
            0.85,
            0.25,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        for column, expected in enumerate(expected_row):
            _require_rendered_close(
                row[column],
                expected,
                f"Case {case_id} beam row {row_index} column {column}",
                relative_tolerance=5.0e-12,
            )

    tower_path, tower_relative, tower_identity = _resolve_linked_manifest_file(
        root,
        file_index,
        base_relative=main_relative,
        raw_link=main["TWRFILE"],
        case_prefix=case_prefix,
        label=f"Case {case_id} main TWRFILE",
    )
    tower_settings = _parse_required_qblade_settings(
        tower_path,
        (
            "RAYLEIGHDMP",
            "STIFFTUNER",
            "MASSTUNER",
            "INTPTYPE",
            "BEAMTYPE",
            "DISCTYPE",
            "DISC",
        ),
        f"Case {case_id} rigid tower",
    )
    _validate_single_value_settings(
        tower_settings,
        numeric={
            "RAYLEIGHDMP": 0.0,
            "STIFFTUNER": 1.0,
            "MASSTUNER": 1.0,
        },
        integer={"INTPTYPE": 0, "BEAMTYPE": 1, "DISCTYPE": 0, "DISC": 2},
        label=f"Case {case_id} rigid tower",
    )
    tower_rows = _numeric_table_after_header(
        tower_path,
        header_prefix="LENFRACT_[-]",
        expected_rows=2,
        expected_columns=19,
        label=f"Case {case_id} rigid tower",
    )
    tower_section = (
        1.0,
        1.0e15,
        1.0e15,
        1.0e15,
        1.0e15,
        1.0e15,
        0.0,
        0.85,
        0.85,
        0.25,
        0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    for row_index, row in enumerate(tower_rows):
        expected_row = (float(row_index), *tower_section)
        for column, expected in enumerate(expected_row):
            _require_rendered_close(
                row[column],
                expected,
                f"Case {case_id} tower row {row_index} column {column}",
                relative_tolerance=5.0e-12,
            )

    blade_path, blade_relative, blade_identity = _resolve_linked_manifest_file(
        root,
        file_index,
        base_relative=turbine_relative,
        raw_link=turbine["BLADEFILE"],
        case_prefix=case_prefix,
        label=f"Case {case_id} turbine BLADEFILE",
    )
    blade_settings = _parse_required_qblade_settings(
        blade_path,
        ("ROTORTYPE", "NUMBLADES"),
        f"Case {case_id} blade geometry",
    )
    _require_text(
        blade_settings["ROTORTYPE"], "HAWT", f"Case {case_id} blade ROTORTYPE"
    )
    _require_integer_text(
        blade_settings["NUMBLADES"], 1, f"Case {case_id} blade NUMBLADES"
    )
    blade_rows = _numeric_table_after_header(
        blade_path,
        header_prefix="POS_[m]",
        expected_rows=N_STATIONS,
        expected_columns=6,
        label=f"Case {case_id} blade geometry",
    )
    for row_index, row in enumerate(blade_rows):
        _require_rendered_close(
            row[0],
            HEIGHT_M * expected_fraction[row_index],
            f"Case {case_id} blade span",
            relative_tolerance=5.0e-10,
        )
        _require_rendered_close(
            row[1],
            DIAMETER_M,
            f"Case {case_id} blade chord",
            relative_tolerance=5.0e-10,
        )
        for column, name in ((2, "twist"), (3, "offset_x"), (4, "offset_y")):
            _require_rendered_close(
                row[column],
                0.0,
                f"Case {case_id} blade {name}",
                relative_tolerance=5.0e-10,
            )
        _require_rendered_close(
            row[5],
            0.5,
            f"Case {case_id} blade pitch axis",
            relative_tolerance=5.0e-10,
        )
    return {
        "config_sha256": config_identity["sha256"],
        "main_structure_sha256": main_identity["sha256"],
        "beam_structure_sha256": beam_identity["sha256"],
        "tower_structure_sha256": tower_identity["sha256"],
        "blade_geometry_sha256": blade_identity["sha256"],
        "beam_relative": beam_relative,
        "tower_relative": tower_relative,
        "blade_relative": blade_relative,
    }


def smooth_activation(reduced_velocity: float) -> float:
    """Frozen cubic-smoothstep activation used by the response closure."""

    u_star = _match_frozen_value(
        reduced_velocity, REDUCED_VELOCITIES, "reduced_velocity"
    )
    if REDUCED_VELOCITY_LOW < u_star < REDUCED_VELOCITY_PEAK:
        coordinate = (u_star - REDUCED_VELOCITY_LOW) / (
            REDUCED_VELOCITY_PEAK - REDUCED_VELOCITY_LOW
        )
        return coordinate**2 * (3.0 - 2.0 * coordinate)
    if REDUCED_VELOCITY_PEAK <= u_star < REDUCED_VELOCITY_HIGH:
        coordinate = (REDUCED_VELOCITY_HIGH - u_star) / (
            REDUCED_VELOCITY_HIGH - REDUCED_VELOCITY_PEAK
        )
        return coordinate**2 * (3.0 - 2.0 * coordinate)
    return 0.0


def derive_duration_bound(scruton_number: float, reduced_velocity: float) -> Dict[str, Any]:
    """Derive one frozen cap from the analytical logistic envelope."""

    sc = _match_frozen_value(scruton_number, tuple(STRUCTURES), "scruton_number")
    u_star = _match_frozen_value(
        reduced_velocity, REDUCED_VELOCITIES, "reduced_velocity"
    )
    structure = STRUCTURES[sc]
    qblade_sc = float(structure["qblade_implied_scruton_number"])
    damping = float(structure["structural_damping_ratio"])
    activation = smooth_activation(u_star)
    response_ratio = (
        u_star * AERODYNAMIC_DAMPING_GAIN_H1 * activation / (2.0 * qblade_sc)
    )
    rate_scale = 2.0 * math.pi * damping

    if response_ratio > 1.0:
        branch = "growth_to_lco"
        exponent_per_cycle = rate_scale * (response_ratio - 1.0)
        equilibrium = SATURATION_AMPLITUDE_RATIO * math.sqrt(
            1.0 - 1.0 / response_ratio
        )
        tolerance = max(
            TERMINAL_ABSOLUTE_TOLERANCE_RATIO,
            TERMINAL_RELATIVE_TOLERANCE * equilibrium,
        )
        target = max(ACTIVE_SEED_LOWER_BOUND_RATIO, equilibrium - tolerance)
        if target <= ACTIVE_SEED_LOWER_BOUND_RATIO:
            settling_cycles = 0.0
        else:
            initial_factor = (
                (equilibrium / ACTIVE_SEED_LOWER_BOUND_RATIO) ** 2 - 1.0
            )
            target_factor = (equilibrium / target) ** 2 - 1.0
            settling_cycles = math.log(initial_factor / target_factor) / (
                2.0 * exponent_per_cycle
            )
    else:
        branch = "decay_to_terminal_tolerance"
        exponent_per_cycle = rate_scale * (response_ratio - 1.0)
        equilibrium = 0.0
        tolerance = TERMINAL_ABSOLUTE_TOLERANCE_RATIO
        decay_rate = rate_scale * (1.0 - response_ratio)
        nonlinear_ratio = (
            0.0
            if response_ratio == 0.0
            else response_ratio
            / ((1.0 - response_ratio) * SATURATION_AMPLITUDE_RATIO**2)
        )
        settling_cycles = math.log(
            (
                1.0 / TERMINAL_ABSOLUTE_TOLERANCE_RATIO**2
                + nonlinear_ratio
            )
            /
            (
                1.0 / STABLE_INITIAL_AMPLITUDE_RATIO**2
                + nonlinear_ratio
            )
        ) / (2.0 * decay_rate)

    unrounded_bound = settling_cycles + FINAL_OBSERVATION_CYCLES
    rounded_bound = int(
        EXTENSION_CYCLES
        * math.ceil((unrounded_bound - 1.0e-12) / EXTENSION_CYCLES)
    )
    maximum_cycles = max(EARLIEST_ACCEPTANCE_CYCLES, rounded_bound)
    frozen_cycles = FROZEN_MAXIMUM_CYCLES[sc][u_star]
    if maximum_cycles != frozen_cycles:
        raise ProtocolError(
            f"Frozen cap drift for Sc={sc:.2f}, U*={u_star:.2f}: "
            f"derived {maximum_cycles}, table {frozen_cycles}."
        )
    endpoint = endpoint_for_cycles(sc, maximum_cycles)
    return {
        "scruton_number": sc,
        "qblade_implied_scruton_number": qblade_sc,
        "reduced_velocity": u_star,
        "activation": activation,
        "response_ratio_R": response_ratio,
        "linear_envelope_exponent_per_cycle": exponent_per_cycle,
        "branch": branch,
        "equilibrium_amplitude_ratio": equilibrium,
        "terminal_tolerance_ratio": tolerance,
        "settling_cycles_unrounded": settling_cycles,
        "final_observation_cycles": FINAL_OBSERVATION_CYCLES,
        "bound_before_rounding_cycles": unrounded_bound,
        "maximum_cycles": maximum_cycles,
        "maximum_endpoint": endpoint,
    }


def endpoint_for_cycles(scruton_number: float, cycles: int) -> Dict[str, Any]:
    """Return the exact nominal/fine endpoint shared in physical time."""

    sc = _match_frozen_value(scruton_number, tuple(STRUCTURES), "scruton_number")
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise ProtocolError("cycles must be a positive integer.")
    frequency_numerator, frequency_denominator = STRUCTURES[sc][
        "natural_frequency_fraction"
    ]
    # cycles / (fn_num/fn_den) / (1/2000)
    step_numerator = cycles * int(frequency_denominator) * 2000
    coarse_steps = (step_numerator + int(frequency_numerator) - 1) // int(
        frequency_numerator
    )
    fine_steps = 2 * coarse_steps
    endpoint_time_ns = coarse_steps * COARSE_TIMESTEP_NS
    if fine_steps * FINE_TIMESTEP_NS != endpoint_time_ns:
        raise ProtocolError("Internal endpoint alignment error.")
    return {
        "evaluation_cycles": cycles,
        "endpoint_time_ns": endpoint_time_ns,
        "endpoint_time_s": endpoint_time_ns / 1.0e9,
        "dt_steps": coarse_steps,
        "dt_half_steps": fine_steps,
    }


def evaluation_schedule(scruton_number: float, reduced_velocity: float) -> List[Dict[str, Any]]:
    sc = _match_frozen_value(scruton_number, tuple(STRUCTURES), "scruton_number")
    u_star = _match_frozen_value(
        reduced_velocity, REDUCED_VELOCITIES, "reduced_velocity"
    )
    maximum = FROZEN_MAXIMUM_CYCLES[sc][u_star]
    return [
        endpoint_for_cycles(sc, cycles)
        for cycles in range(
            FIRST_EVALUATION_CYCLES, maximum + 1, EXTENSION_CYCLES
        )
    ]


def frozen_duration_table() -> List[Dict[str, Any]]:
    """Return all 36 frozen cap rows, including derivation diagnostics."""

    return [
        derive_duration_bound(sc, u_star)
        for sc in sorted(STRUCTURES)
        for u_star in REDUCED_VELOCITIES
    ]


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Could not read {label} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must contain one JSON object.")
    return value


def _safe_manifest_path(root: Path, raw_path: Any, label: str) -> Tuple[Path, str]:
    relative = Path(str(raw_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError(f"{label} must be a campaign-relative path.")
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(relative)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"{label} does not exist: {relative.as_posix()}") from exc
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes the campaign root.") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise ProtocolError(f"{label} must be a regular, non-symlink file.")
    return resolved, relative.as_posix()


def _manifest_file_index(root: Path, manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_rows = manifest.get("files")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ProtocolError("Campaign manifest has no immutable files table.")
    index: Dict[str, Dict[str, Any]] = {}
    for row_index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise ProtocolError(f"Campaign files[{row_index}] is not an object.")
        _path, relative = _safe_manifest_path(
            root, raw_row.get("path"), f"campaign files[{row_index}].path"
        )
        if relative in index:
            raise ProtocolError(f"Duplicate campaign file identity: {relative}")
        expected_sha = _require_sha256(
            raw_row.get("sha256"), f"campaign files[{row_index}].sha256"
        )
        size = raw_row.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ProtocolError(f"campaign files[{row_index}].bytes is invalid.")
        index[relative] = {
            "path": relative,
            "sha256": expected_sha,
            "bytes": size,
        }
    return index


def validate_ambient_v2_campaign_tree(
    root: Path,
    file_index: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require the v2 input tree to contain exactly its immutable files.

    ``campaign_manifest.json`` authenticates the table and is intentionally
    omitted from that table to avoid a self-hash.  Every other regular file is
    declared, every declared file is rehashed, and symlinks are forbidden.
    This makes a native-bridge log (or any other case-local residue) a launch
    blocker instead of something the runner may silently delete.
    """

    root = Path(root).resolve()
    actual_files: set[str] = set()
    symlinks: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
        elif path.is_file() and relative != "campaign_manifest.json":
            actual_files.add(relative)
    if symlinks:
        raise ProtocolError(
            "Ambient-v2 campaign tree contains symlink(s): "
            + ", ".join(symlinks)
        )
    declared_files = set(file_index)
    if actual_files != declared_files:
        unexpected = sorted(actual_files.difference(declared_files))
        missing = sorted(declared_files.difference(actual_files))
        raise ProtocolError(
            "Ambient-v2 campaign tree differs from its immutable file table; "
            f"unexpected={unexpected}, missing={missing}."
        )
    native_logs = sorted(
        relative
        for relative in actual_files
        if PurePosixPath(relative).name == "vorlap_qblade_bridge_cpp.log"
    )
    if native_logs:
        raise ProtocolError(
            "Ambient-v2 campaign contains preexisting native bridge log(s): "
            + ", ".join(native_logs)
        )
    for relative in sorted(declared_files):
        _verify_file_identity(
            root,
            file_index,
            relative,
            f"Ambient-v2 immutable file {relative}",
        )


def _verify_file_identity(
    root: Path,
    file_index: Mapping[str, Mapping[str, Any]],
    relative: str,
    label: str,
) -> Dict[str, Any]:
    if relative not in file_index:
        raise ProtocolError(f"{label} is absent from the campaign files table: {relative}")
    path, normalized = _safe_manifest_path(root, relative, label)
    expected = file_index[normalized]
    observed_size = path.stat().st_size
    observed_sha = sha256_file(path)
    if observed_sha != expected["sha256"]:
        raise ProtocolError(
            f"{label} hash mismatch for {normalized}: expected "
            f"{expected['sha256']}, observed {observed_sha}."
        )
    if observed_size != expected["bytes"]:
        raise ProtocolError(
            f"{label} byte-count mismatch for {normalized}: expected "
            f"{expected['bytes']}, observed {observed_size}."
        )
    return dict(expected)


_AMBIENT_V2_AIRFOIL_MANIFEST_KEYS = {
    "airfoil_id",
    "file",
    "sha256",
    "bytes",
    "reynolds",
    "reynolds_min",
    "reynolds_max",
    "aoa_min_deg",
    "aoa_max_deg",
    "thickness_ratio",
    "generator",
    "source_data_sha256",
    "reynolds_model",
    "reynolds_replication_justification",
    "processing_config",
}


def _h5_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def validate_ambient_v2_airfoil_source(
    *,
    root: Path,
    file_index: Mapping[str, Mapping[str, Any]],
    case_id: str,
    parameter_file: str,
    config: Mapping[str, Any],
    coefficient_fingerprint: str,
) -> Dict[str, Any]:
    """Validate the single source file that controls v2 width and forcing.

    Manifest hashes alone are not sufficient because a modified HDF5 file and
    its enclosing JSON/manifest hashes can be changed together.  This parser
    freezes every dataset and attribute consumed by the runtime for the
    deliberately synthetic CYLINDER source.
    """

    if config.get("airfoil_dir") != "../VorLapAirfoils":
        raise ProtocolError(
            f"Case {case_id} airfoil_dir must be the case-local VorLapAirfoils directory."
        )
    config_path, _ = _safe_manifest_path(
        root, parameter_file, f"Case {case_id} parameter config"
    )
    expected_case_root = (Path(root).resolve() / "cases" / case_id).resolve()
    expected_airfoil_dir = expected_case_root / "model" / "VorLapAirfoils"
    candidate_dir = config_path.parent / str(config["airfoil_dir"])
    try:
        resolved_dir = candidate_dir.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"Case {case_id} airfoil_dir is unavailable.") from exc
    if candidate_dir.is_symlink() or resolved_dir != expected_airfoil_dir:
        raise ProtocolError(
            f"Case {case_id} airfoil_dir does not resolve inside its frozen case path."
        )
    try:
        resolved_dir.relative_to(expected_case_root)
    except ValueError as exc:  # pragma: no cover - guarded by exact equality
        raise ProtocolError(f"Case {case_id} airfoil_dir escapes its case.") from exc
    source_files = sorted(resolved_dir.glob("*.h5"))
    if (
        len(source_files) != 1
        or source_files[0].name != "CYLINDER.h5"
        or source_files[0].is_symlink()
    ):
        raise ProtocolError(
            f"Case {case_id} must contain one regular CYLINDER.h5 source."
        )
    source_path = source_files[0]
    source_relative = source_path.relative_to(Path(root).resolve()).as_posix()
    source_identity = _verify_file_identity(
        Path(root),
        file_index,
        source_relative,
        f"Case {case_id} CYLINDER source",
    )

    expected_dataset_names = {
        "Airfoilname",
        "Re",
        "Thickness",
        "AOA",
        *(f"{field}_{suffix}" for field in ("CL", "CD", "CM", "CF") for suffix in ("ST", "Amp", "Pha")),
    }
    expected_attributes = {
        "source_scope": "fixed_wake_harmonic_screen_not_hartlen_currie",
        "reference_doi": "10.5194/wes-11-2585-2026",
        "coefficient_convention": (
            "published CL0=0.3 inserted as peak harmonic coefficient"
        ),
        "response_closure": (
            "frozen-coefficient response-aware single-mode aerodynamic damping"
        ),
        "coefficient_fingerprint_sha256": coefficient_fingerprint,
    }
    try:
        with h5py.File(source_path, "r") as h5:
            if set(h5.keys()) != expected_dataset_names:
                raise ProtocolError(
                    f"Case {case_id} CYLINDER dataset schema is not frozen."
                )
            if set(h5.attrs) != set(expected_attributes):
                raise ProtocolError(
                    f"Case {case_id} CYLINDER attribute schema is not frozen."
                )
            for name, expected in expected_attributes.items():
                if _h5_text(h5.attrs[name]) != expected:
                    raise ProtocolError(
                        f"Case {case_id} CYLINDER attribute {name} is not frozen."
                    )
            if _h5_text(h5["Airfoilname"][()]) != "CYLINDER":
                raise ProtocolError(f"Case {case_id} airfoil ID is not CYLINDER.")
            reynolds = np.asarray(h5["Re"][()], dtype=float)
            aoa = np.asarray(h5["AOA"][()], dtype=float)
            thickness = np.asarray(h5["Thickness"][()], dtype=float)
            if not np.array_equal(reynolds, np.array([1.0e3, 1.0e6])):
                raise ProtocolError(f"Case {case_id} CYLINDER Reynolds grid changed.")
            if not np.array_equal(aoa, np.array([-180.0, 180.0])):
                raise ProtocolError(f"Case {case_id} CYLINDER angle grid changed.")
            if not np.array_equal(thickness, np.array([1.0])):
                raise ProtocolError(f"Case {case_id} CYLINDER Thickness must equal 1.0.")
            shape = (2, 2, 2)
            zeros = np.zeros(shape, dtype=float)
            expected_st = zeros.copy()
            expected_amp = zeros.copy()
            expected_st[:, :, 1] = 0.2
            expected_amp[:, :, 1] = 0.3
            for field in ("CL", "CD", "CM", "CF"):
                for suffix in ("ST", "Amp", "Pha"):
                    actual = np.asarray(h5[f"{field}_{suffix}"][()], dtype=float)
                    expected = zeros
                    if field == "CL" and suffix == "ST":
                        expected = expected_st
                    elif field == "CL" and suffix == "Amp":
                        expected = expected_amp
                    if actual.shape != shape or not np.array_equal(actual, expected):
                        raise ProtocolError(
                            f"Case {case_id} CYLINDER {field}_{suffix} changed."
                        )
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(
            f"Could not parse Case {case_id} CYLINDER source: {exc}"
        ) from exc

    raw_manifest = config.get("airfoil_source_manifest")
    if not isinstance(raw_manifest, list) or len(raw_manifest) != 1:
        raise ProtocolError(
            f"Case {case_id} requires one airfoil_source_manifest row."
        )
    row = raw_manifest[0]
    if not isinstance(row, dict) or set(row) != _AMBIENT_V2_AIRFOIL_MANIFEST_KEYS:
        raise ProtocolError(
            f"Case {case_id} airfoil_source_manifest key schema is not frozen."
        )
    expected_row = {
        "airfoil_id": "CYLINDER",
        "file": "CYLINDER.h5",
        "sha256": source_identity["sha256"],
        "bytes": source_identity["bytes"],
        "reynolds": [1000.0, 1000000.0],
        "reynolds_min": 1000.0,
        "reynolds_max": 1000000.0,
        "aoa_min_deg": -180.0,
        "aoa_max_deg": 180.0,
        "thickness_ratio": 1.0,
        "generator": "",
        "source_data_sha256": "",
        "reynolds_model": "",
        "reynolds_replication_justification": "",
        "processing_config": None,
    }
    if row != expected_row:
        raise ProtocolError(
            f"Case {case_id} airfoil_source_manifest differs from CYLINDER.h5."
        )
    return {
        "path": source_relative,
        "sha256": source_identity["sha256"],
        "bytes": source_identity["bytes"],
        "manifest_sha256": sha256_payload(row),
    }


def _validate_startup_seed(
    response_model: Mapping[str, Any], structure: Mapping[str, Any], case_id: str
) -> None:
    seed = response_model.get("startup_seed")
    if not isinstance(seed, dict):
        raise ProtocolError(f"Case {case_id} has no hash-bound startup_seed object.")
    expected_seed_keys = {
        "schema_version",
        "enabled",
        "duration_cycles",
        "target_amplitude_ratio",
        "structural_damping_ratio",
        "taper",
        "carrier",
        "nodal_weighting",
        "nodal_force_limit_source",
    }
    if set(seed) != expected_seed_keys:
        unexpected = sorted(set(seed).difference(expected_seed_keys))
        missing = sorted(expected_seed_keys.difference(seed))
        raise ProtocolError(
            f"Case {case_id} startup_seed key schema differs; "
            f"unexpected={unexpected}, missing={missing}."
        )
    if seed.get("enabled") is not True:
        raise ProtocolError(f"Case {case_id} startup_seed must be enabled.")
    if seed.get("schema_version") != 1:
        raise ProtocolError(f"Case {case_id} startup_seed schema_version must be 1.")
    _require_close(
        seed.get("duration_cycles"),
        STARTUP_SEED_DURATION_CYCLES,
        f"Case {case_id} startup_seed.duration_cycles",
    )
    _require_close(
        seed.get("target_amplitude_ratio"),
        REQUESTED_SEED_TARGET_RATIO,
        f"Case {case_id} startup_seed.target_amplitude_ratio",
    )
    _require_close(
        seed.get("structural_damping_ratio"),
        float(structure["structural_damping_ratio"]),
        f"Case {case_id} startup_seed.structural_damping_ratio",
    )
    expected_strings = {
        "taper": "hann_sine_squared",
        "carrier": "natural_frequency",
        "nodal_weighting": "projection_mass_times_mode",
        "nodal_force_limit_source": "response_model_max_abs_nodal_force_n",
    }
    for name, expected in expected_strings.items():
        if seed.get(name) != expected:
            raise ProtocolError(
                f"Case {case_id} startup_seed.{name} does not match {expected!r}."
            )


def _validate_response_model(
    response_model: Any,
    structure: Mapping[str, Any],
    case_id: str,
    *,
    response_schema_version: int,
) -> Dict[str, Any]:
    if not isinstance(response_model, dict):
        raise ProtocolError(f"Case {case_id} response_model must be an object.")
    if response_model.get("type") != "modal_negative_damping":
        raise ProtocolError(f"Case {case_id} response model type is not frozen.")
    if response_model.get("schema_version") != response_schema_version:
        raise ProtocolError(
            f"Case {case_id} response model schema_version must be "
            f"{response_schema_version}."
        )
    if response_schema_version == 2:
        expected_response_keys = {
            "type",
            "schema_version",
            "node_ids",
            "mode_shape",
            "natural_frequency_hz",
            "reference_length_m",
            "projection_weights",
            "aerodynamic_damping_gain_h1",
            "saturation_amplitude_ratio",
            "reduced_velocity_low",
            "reduced_velocity_peak",
            "reduced_velocity_high",
            "rms_time_constant_s",
            "bandpass_damping_ratio",
            "max_abs_gain",
            "max_abs_nodal_force_n",
            "startup_seed",
            "force_direction",
            "enabled",
            "aerodynamic_speed_basis",
        }
        if set(response_model) != expected_response_keys:
            unexpected = sorted(set(response_model).difference(expected_response_keys))
            missing = sorted(expected_response_keys.difference(response_model))
            raise ProtocolError(
                f"Case {case_id} schema-v2 response_model key schema differs; "
                f"unexpected={unexpected}, missing={missing}."
            )
        if response_model.get("aerodynamic_speed_basis") != (
            AMBIENT_V2_AERODYNAMIC_SPEED_BASIS
        ):
            raise ProtocolError(
                f"Case {case_id} response model aerodynamic_speed_basis is not frozen."
            )
        if "response_section_width_m" in response_model:
            raise ProtocolError(
                f"Case {case_id} must derive response width from declared ambient flow."
            )
    elif "aerodynamic_speed_basis" in response_model:
        raise ProtocolError(
            f"Case {case_id} legacy response model gained a v2 speed basis."
        )
    if response_model.get("enabled") is not True:
        raise ProtocolError(f"Case {case_id} response model is not enabled.")
    expected_values = {
        "natural_frequency_hz": float(structure["natural_frequency_hz"]),
        "reference_length_m": DIAMETER_M,
        "aerodynamic_damping_gain_h1": AERODYNAMIC_DAMPING_GAIN_H1,
        "saturation_amplitude_ratio": SATURATION_AMPLITUDE_RATIO,
        "reduced_velocity_low": REDUCED_VELOCITY_LOW,
        "reduced_velocity_peak": REDUCED_VELOCITY_PEAK,
        "reduced_velocity_high": REDUCED_VELOCITY_HIGH,
        "rms_time_constant_s": (
            RMS_TIME_CONSTANT_CYCLES / float(structure["natural_frequency_hz"])
        ),
        "bandpass_damping_ratio": BANDPASS_DAMPING_RATIO,
        "max_abs_gain": MAX_ABS_GAIN,
        "max_abs_nodal_force_n": MAX_ABS_NODAL_FORCE_N,
    }
    for name, expected in expected_values.items():
        _require_close(
            response_model.get(name), expected, f"Case {case_id} response_model.{name}"
        )
    _validate_startup_seed(response_model, structure, case_id)
    if response_model.get("force_direction") != "normal":
        raise ProtocolError(
            f"Case {case_id} response_model.force_direction must be 'normal'."
        )
    expected_node_ids = [
        f"BLD_1_{index / (N_STATIONS - 1):.6f}"
        for index in range(N_STATIONS)
    ]
    if response_model.get("node_ids") != expected_node_ids:
        raise ProtocolError(
            f"Case {case_id} response_model.node_ids do not match the frozen grid."
        )
    _require_numeric_sequence(
        response_model.get("mode_shape"),
        _expected_mode_shape(),
        f"Case {case_id} response_model.mode_shape",
    )
    _require_numeric_sequence(
        response_model.get("projection_weights"),
        _expected_projection_weights(
            float(structure["equivalent_mass_kg_per_m"])
        ),
        f"Case {case_id} response_model.projection_weights",
    )
    return dict(response_model)


def _validate_runtime_mapping_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    sim_path: Path,
    timestep_s: float,
    case_id: str,
) -> None:
    expected_keys = {
        "airfoil_dir",
        "airfoil_fallbacks",
        "airfoil_source_manifest",
        "available_airfoil_ids",
        "debug",
        "default_airfoil_id",
        "default_airfoil_source_id",
        "external_input_rows",
        "external_output_rows",
        "force_scale",
        "function_name",
        "geometry_audit_file",
        "include_mean_load",
        "include_struts",
        "include_tower",
        "library_stem",
        "n_freq_depth",
        "node_ids",
        "parameter_file",
        "python_path_entries",
        "response_model",
        "sim_path",
        "source_bounds_policy",
        "source_frequency_cap_hz",
        "source_parameter_dir",
        "swap_layout",
        "swap_size",
        "tower_airfoil_id",
    }
    if set(config) != expected_keys:
        unexpected = sorted(set(config).difference(expected_keys))
        missing = sorted(expected_keys.difference(config))
        raise ProtocolError(
            f"Case {case_id} runtime config key schema differs; "
            f"unexpected={unexpected}, missing={missing}."
        )
    expected_node_ids = [
        f"BLD_1_{index / (N_STATIONS - 1):.10f}"
        for index in range(N_STATIONS)
    ]
    if config.get("node_ids") != expected_node_ids:
        raise ProtocolError(
            f"Case {case_id} runtime node_ids do not match the structural output grid."
        )
    expected_scalars = {
        "airfoil_dir": "../VorLapAirfoils",
        "default_airfoil_id": "default",
        "default_airfoil_source_id": "CYLINDER",
        "function_name": "update",
        "geometry_audit_file": "vorlap_qblade_geometry_audit.csv",
        "library_stem": "libvorlap_qblade_bridge",
        "parameter_file": "Control/vorlap_qblade_external.json",
        "source_bounds_policy": "raise",
        "source_parameter_dir": ".",
        "tower_airfoil_id": "cylinder",
    }
    for name, expected in expected_scalars.items():
        if config.get(name) != expected:
            raise ProtocolError(
                f"Case {case_id} runtime {name} does not match {expected!r}."
            )
    for name, expected in {
        "debug": False,
        "include_mean_load": False,
        "include_struts": False,
        "include_tower": False,
    }.items():
        if config.get(name) is not expected:
            raise ProtocolError(
                f"Case {case_id} runtime {name} must be {str(expected).lower()}."
            )
    if config.get("airfoil_fallbacks") != []:
        raise ProtocolError(f"Case {case_id} runtime airfoil_fallbacks must be empty.")
    if config.get("available_airfoil_ids") != ["CYLINDER", "default"]:
        raise ProtocolError(
            f"Case {case_id} runtime available_airfoil_ids are not frozen."
        )
    if config.get("python_path_entries") != ["/work/VorLap"]:
        raise ProtocolError(
            f"Case {case_id} runtime python_path_entries must identify the "
            "fixed Docker qualification checkout."
        )
    _require_close(
        config.get("source_frequency_cap_hz"),
        0.5 / timestep_s,
        f"Case {case_id} source frequency cap",
    )
    _require_integer_text(config.get("n_freq_depth"), 2, f"Case {case_id} n_freq_depth")
    _require_integer_text(
        config.get("swap_size"), 3 + 6 * N_STATIONS, f"Case {case_id} swap_size"
    )

    raw_sim_link = Path(str(config.get("sim_path", "")))
    if not str(raw_sim_link) or raw_sim_link.is_absolute() or ".." not in raw_sim_link.parts:
        raise ProtocolError(f"Case {case_id} runtime sim_path is not the expected relative link.")
    resolved_sim = (config_path.parent / raw_sim_link).resolve()
    if resolved_sim != sim_path.resolve():
        raise ProtocolError(
            f"Case {case_id} runtime sim_path does not resolve to the bound simulation."
        )

    expected_layout = [
        {"name": "time", "offset": 0, "shape": []},
        {"name": "timestep", "offset": 1, "shape": []},
        {"name": "azimuth_deg", "offset": 2, "shape": []},
        {"name": "velocity", "offset": 3, "shape": [N_STATIONS, 3]},
        {
            "name": "force",
            "offset": 3 + 3 * N_STATIONS,
            "shape": [N_STATIONS, 3],
        },
    ]
    if config.get("swap_layout") != expected_layout:
        raise ProtocolError(f"Case {case_id} runtime swap_layout is not frozen.")

    expected_input_rows, expected_output_rows = _expected_external_rows()
    if config.get("external_input_rows") != expected_input_rows:
        raise ProtocolError(
            f"Case {case_id} runtime external_input_rows are not frozen."
        )
    if config.get("external_output_rows") != expected_output_rows:
        raise ProtocolError(
            f"Case {case_id} runtime external_output_rows are not frozen."
        )


def _case_bundle_rows(
    case_id: str, file_index: Mapping[str, Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    prefix = f"cases/{case_id}/"
    rows = [dict(row) for path, row in file_index.items() if path.startswith(prefix)]
    if not rows:
        raise ProtocolError(f"Case {case_id} has no hash-bound input files.")
    return sorted(rows, key=lambda row: row["path"])


def ambient_v2_response_semantics() -> Dict[str, Any]:
    """Return the exact response-only speed and namespace contract for v2."""

    return {
        "schema_version": 2,
        "aerodynamic_speed_basis": AMBIENT_V2_AERODYNAMIC_SPEED_BASIS,
        "response_speed_equation": "hypot(Uinf_dot_chord,Uinf_dot_normal)",
        "activation_and_scanlan_prefactor_share_response_speed": True,
        "base_vorlap_speed_basis": "relative_crossflow_Uinf_minus_Vstruct",
        "response_width_basis": (
            "declared_ambient_crossflow_angle_or_explicit_response_width"
        ),
        "configured_reduced_velocity_equation": (
            AMBIENT_V2_CONFIGURED_REDUCED_VELOCITY_EQUATION
        ),
        "case_id_namespace": AMBIENT_V2_CASE_ID_NAMESPACE,
        "checkpoint_reuse_policy": AMBIENT_V2_CHECKPOINT_REUSE_POLICY,
    }


def validate_response_semantics(
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate optional v2 semantics while leaving legacy manifests unchanged."""

    raw = manifest.get("response_model_semantics")
    gate = manifest.get("equation_reproduction_gate")
    if raw is None:
        if gate is not None:
            raise ProtocolError(
                "Legacy campaign cannot contain an ambient-v2 equation gate."
            )
        return {"response_schema_version": 1, "v2": False}
    expected_semantics = ambient_v2_response_semantics()
    if raw != expected_semantics:
        raise ProtocolError("Campaign ambient-v2 response semantics are not frozen.")
    if not isinstance(gate, dict):
        raise ProtocolError("Ambient-v2 campaign has no equation-reproduction gate.")
    expected_gate_fields = {
        "schema",
        "response_model_schema_version",
        "aerodynamic_speed_basis",
        "calibration_published_scruton_label",
        "calibration_qblade_implied_scruton_number",
        "scruton_source",
        "equation",
        "coefficient_fingerprint_sha256",
        "frozen_parameters",
        "startup_seed",
        "limit_caps",
        "stationary_cycle_caps_sha256",
        "active_threshold_amplitude_over_diameter",
        "expected_active_reduced_velocities",
        "fine_estimator_rmse_tolerance_over_diameter",
        "fine_estimator_maximum_absolute_error_tolerance_over_diameter",
        "coarse_fine_estimator_maximum_absolute_difference_tolerance_over_diameter",
        "analytical_table",
    }
    if set(gate) != expected_gate_fields:
        unexpected = sorted(set(gate).difference(expected_gate_fields))
        missing = sorted(expected_gate_fields.difference(gate))
        raise ProtocolError(
            "Ambient-v2 equation gate field schema changed; "
            f"unexpected={unexpected}, missing={missing}."
        )
    expected_gate_scalars = {
        "schema": AMBIENT_V2_EQUATION_GATE_SCHEMA,
        "response_model_schema_version": 2,
        "aerodynamic_speed_basis": AMBIENT_V2_AERODYNAMIC_SPEED_BASIS,
        "calibration_published_scruton_label": 5.18,
        "calibration_qblade_implied_scruton_number": (
            AMBIENT_V2_CALIBRATION_QBLADE_SCRUTON_NUMBER
        ),
        "coefficient_fingerprint_sha256": (
            AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
        ),
        "active_threshold_amplitude_over_diameter": (
            AMBIENT_V2_ACTIVE_AMPLITUDE_RATIO
        ),
        "fine_estimator_rmse_tolerance_over_diameter": (
            AMBIENT_V2_FINE_ESTIMATOR_RMSE_TOLERANCE
        ),
        "fine_estimator_maximum_absolute_error_tolerance_over_diameter": (
            AMBIENT_V2_FINE_ESTIMATOR_MAX_ABS_TOLERANCE
        ),
        "coarse_fine_estimator_maximum_absolute_difference_tolerance_over_diameter": (
            AMBIENT_V2_COARSE_FINE_ESTIMATOR_TOLERANCE
        ),
    }
    for name, expected in expected_gate_scalars.items():
        actual = gate.get(name)
        if isinstance(expected, float):
            _require_close(actual, expected, f"Equation gate {name}")
        elif actual != expected:
            raise ProtocolError(f"Equation gate {name} is not frozen.")
    expected_gate_text = {
        "scruton_source": (
            "generated case equivalent mass, damping ratio, campaign air "
            "density, and diameter; no Scruton retuning"
        ),
        "equation": (
            "A/D=a_sat*sqrt(max(0,1-2*Sc_QBlade/"
            "(Ustar*H1*activation)))"
        ),
    }
    for name, expected in expected_gate_text.items():
        if gate.get(name) != expected:
            raise ProtocolError(f"Equation gate {name} is not frozen.")
    if gate.get("expected_active_reduced_velocities") != list(
        AMBIENT_V2_EXPECTED_ACTIVE_REDUCED_VELOCITIES
    ):
        raise ProtocolError("Equation gate active U* grid is not frozen.")
    expected_parameters = {
        "reduced_velocity_low": REDUCED_VELOCITY_LOW,
        "reduced_velocity_peak": REDUCED_VELOCITY_PEAK,
        "reduced_velocity_high": REDUCED_VELOCITY_HIGH,
        "aerodynamic_damping_gain_h1": AERODYNAMIC_DAMPING_GAIN_H1,
        "saturation_amplitude_ratio": SATURATION_AMPLITUDE_RATIO,
        "rms_time_constant_cycles": RMS_TIME_CONSTANT_CYCLES,
        "bandpass_damping_ratio": BANDPASS_DAMPING_RATIO,
        "max_abs_gain": MAX_ABS_GAIN,
        "max_abs_nodal_force_n": MAX_ABS_NODAL_FORCE_N,
    }
    parameters = gate.get("frozen_parameters")
    if not isinstance(parameters, dict) or set(parameters) != set(expected_parameters):
        raise ProtocolError("Equation gate frozen coefficient fields changed.")
    for name, expected in expected_parameters.items():
        _require_close(parameters.get(name), expected, f"Equation gate parameter {name}")
    expected_seed = {
        "schema_version": 1,
        "duration_cycles": 5.0,
        "target_amplitude_ratio": 0.005,
        "taper": "hann_sine_squared",
        "carrier": "natural_frequency",
    }
    if gate.get("startup_seed") != expected_seed:
        raise ProtocolError("Equation gate startup seed changed.")
    expected_limits = {
        "max_abs_gain": MAX_ABS_GAIN,
        "max_abs_nodal_force_n": MAX_ABS_NODAL_FORCE_N,
    }
    if gate.get("limit_caps") != expected_limits:
        raise ProtocolError("Equation gate response limit caps changed.")
    expected_caps_hash = sha256_payload(
        {
            str(sc): {
                str(u_star): cycles
                for u_star, cycles in sorted(rows.items())
            }
            for sc, rows in sorted(FROZEN_MAXIMUM_CYCLES.items())
        }
    )
    if gate.get("stationary_cycle_caps_sha256") != expected_caps_hash:
        raise ProtocolError("Equation gate stationary cycle caps changed.")

    table = gate.get("analytical_table")
    if not isinstance(table, list) or len(table) != len(REDUCED_VELOCITIES):
        raise ProtocolError("Equation gate analytical table is incomplete.")
    observed_active: List[float] = []
    for row, u_star in zip(table, REDUCED_VELOCITIES):
        if not isinstance(row, dict):
            raise ProtocolError("Equation gate analytical row is not an object.")
        if set(row) != {
            "reduced_velocity",
            "analytical_equilibrium_amplitude_over_diameter",
        }:
            raise ProtocolError("Equation gate analytical row fields changed.")
        _require_close(
            row.get("reduced_velocity"),
            u_star,
            "Equation gate analytical reduced velocity",
        )
        activation = smooth_activation(u_star)
        denominator = u_star * AERODYNAMIC_DAMPING_GAIN_H1 * activation
        expected_amplitude = (
            0.0
            if denominator <= 0.0
            else SATURATION_AMPLITUDE_RATIO
            * math.sqrt(
                max(
                    0.0,
                    1.0
                    - 2.0
                    * AMBIENT_V2_CALIBRATION_QBLADE_SCRUTON_NUMBER
                    / denominator,
                )
            )
        )
        amplitude = _require_close(
            row.get("analytical_equilibrium_amplitude_over_diameter"),
            expected_amplitude,
            "Equation gate analytical amplitude",
        )
        if amplitude >= AMBIENT_V2_ACTIVE_AMPLITUDE_RATIO:
            observed_active.append(u_star)
    if observed_active != list(AMBIENT_V2_EXPECTED_ACTIVE_REDUCED_VELOCITIES):
        raise ProtocolError("Equation gate analytical active grid changed.")
    return {
        "response_schema_version": 2,
        "v2": True,
        "response_model_semantics": dict(raw),
        "equation_reproduction_gate": dict(gate),
        "response_semantics_sha256": sha256_payload(
            {"semantics": raw, "equation_reproduction_gate": gate}
        ),
    }


# Preserve the private name used by older in-tree callers while exposing the
# validator to campaign loading and analysis code.
_validate_response_semantics = validate_response_semantics


def _require_exact_keys(
    value: Any, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label} must be an object.")
    if set(value) != expected:
        unexpected = sorted(set(value).difference(expected))
        missing = sorted(expected.difference(value))
        raise ProtocolError(
            f"{label} key schema differs; unexpected={unexpected}, missing={missing}."
        )
    return value


def _ambient_v2_float_tag(value: float) -> str:
    return f"{float(value):.5f}".rstrip("0").rstrip(".").replace(".", "p")


def ambient_v2_case_id(
    scruton_number: float, reduced_velocity: float, timestep_level: str
) -> str:
    if timestep_level not in {"dt", "dt_half"}:
        raise ProtocolError("Ambient-v2 timestep level must be dt or dt_half.")
    return (
        f"{AMBIENT_V2_CASE_ID_NAMESPACE}_sc"
        f"{_ambient_v2_float_tag(scruton_number)}_u"
        f"{_ambient_v2_float_tag(reduced_velocity)}_{timestep_level}"
    )


def validate_ambient_v2_reference_files(
    root: Path,
    manifest: Mapping[str, Any],
    file_index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Path]:
    reference = _require_exact_keys(
        manifest.get("reference"),
        {
            "doi",
            "experiment_doi",
            "height_m",
            "diameter_m",
            "reference_files",
            "figure9_publisher_sha256",
            "table1_publisher_sha256",
        },
        "Ambient-v2 reference",
    )
    expected_scalars = {
        "doi": "10.5194/wes-11-2585-2026",
        "experiment_doi": "10.1016/j.jweia.2020.104438",
        "height_m": HEIGHT_M,
        "diameter_m": DIAMETER_M,
        "figure9_publisher_sha256": (
            "e20f72c24c431b064615883965ef0f20d0f69b91bdefddf1c1ec52ef9cf9ee91"
        ),
        "table1_publisher_sha256": (
            "d24a65f63a33783ee6d69cc239915affa0c2cb6a43eb180cd1ee9db2ee7c4a22"
        ),
    }
    for name, expected in expected_scalars.items():
        if reference.get(name) != expected:
            raise ProtocolError(f"Ambient-v2 reference {name} is not frozen.")
    identities = reference.get("reference_files")
    if not isinstance(identities, dict) or set(identities) != set(
        AMBIENT_V2_REFERENCE_IDENTITIES
    ):
        raise ProtocolError("Ambient-v2 reference-file set is not frozen.")
    resolved: Dict[str, Path] = {}
    for filename, expected in AMBIENT_V2_REFERENCE_IDENTITIES.items():
        identity = identities.get(filename)
        if not isinstance(identity, dict) or identity != expected:
            raise ProtocolError(
                f"Ambient-v2 reference identity changed for {filename}."
            )
        verified = _verify_file_identity(
            Path(root), file_index, expected["path"], f"Reference {filename}"
        )
        if (
            verified["sha256"] != expected["sha256"]
            or verified["bytes"] != expected["bytes"]
        ):
            raise ProtocolError(
                f"Ambient-v2 campaign-local reference changed for {filename}."
            )
        path, _ = _safe_manifest_path(
            Path(root), expected["path"], f"Reference {filename}"
        )
        resolved[filename] = path
    return resolved


def _postfreeze_amplitude(
    qblade_scruton: float,
    reduced_velocity: float,
    frozen: Mapping[str, Any],
) -> float:
    u_star = np.asarray([float(reduced_velocity)], dtype=float)
    activation = np.zeros_like(u_star)
    low = float(frozen["reduced_velocity_low"])
    peak = float(frozen["reduced_velocity_peak"])
    high = float(frozen["reduced_velocity_high"])
    rising = (u_star > low) & (u_star < peak)
    falling = (u_star >= peak) & (u_star < high)
    rise_coordinate = (u_star[rising] - low) / (peak - low)
    fall_coordinate = (high - u_star[falling]) / (high - peak)
    activation[rising] = rise_coordinate**2 * (3.0 - 2.0 * rise_coordinate)
    activation[falling] = fall_coordinate**2 * (3.0 - 2.0 * fall_coordinate)
    activation[u_star == peak] = 1.0
    denominator = (
        u_star
        * float(frozen["aerodynamic_damping_gain_h1"])
        * activation
    )
    radicand = np.full_like(u_star, -np.inf)
    active = denominator > 0.0
    radicand[active] = 1.0 - 2.0 * float(qblade_scruton) / denominator[active]
    return float(
        float(frozen["saturation_amplitude_ratio"])
        * np.sqrt(np.maximum(0.0, radicand))[0]
    )


def expected_ambient_v2_postfreeze_gate(
    coefficients: Mapping[str, Any],
    *,
    figure10_path: Path,
    model_parameters_path: Path,
) -> Dict[str, Any]:
    peaks: Dict[Tuple[float, str], Tuple[float, float]] = {}
    with Path(figure10_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            peaks[(float(row["scruton_number"]), row["series"])] = (
                float(row["peak_tip_amplitude_over_diameter"]),
                float(row["digitization_uncertainty"]),
            )
    parameters: Dict[float, Dict[str, float]] = {}
    with Path(model_parameters_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parameters[float(row["scruton_number"])] = {
                name: float(value)
                for name, value in row.items()
                if name != "scruton_number"
            }
    if set(parameters) != set(STRUCTURES):
        raise ProtocolError("Ambient-v2 model-parameter source case set changed.")
    for sc, structure in STRUCTURES.items():
        expected = {
            "natural_frequency_hz": structure["natural_frequency_hz"],
            "equivalent_mass_kg_per_m": structure["equivalent_mass_kg_per_m"],
            "damping_ratio": structure["structural_damping_ratio"],
        }
        for name, value in expected.items():
            if parameters[sc].get(name) != value:
                raise ProtocolError(
                    f"Ambient-v2 model parameter {name} changed for Sc={sc:.2f}."
                )
    frozen = coefficients.get("frozen_parameters")
    if not isinstance(frozen, dict):
        raise ProtocolError("Ambient-v2 coefficients have no frozen_parameters.")
    comparisons = []
    for sc in (2.94, 7.02):
        target, uncertainty = peaks[(sc, "hgast_hartlen_currie")]
        u_star = parameters[sc]["u_star_max_figure_digitized"]
        qblade_scruton = float(STRUCTURES[sc]["qblade_implied_scruton_number"])
        prediction = _postfreeze_amplitude(
            qblade_scruton, u_star, frozen
        )
        tolerance = max(0.03, 0.15 * target)
        error = abs(prediction - target)
        comparisons.append(
            {
                "scruton_number": sc,
                "qblade_implied_scruton_number": qblade_scruton,
                "reference_peak_u_star": u_star,
                "postfreeze_prediction_peak_over_diameter": prediction,
                "hgast_reference_peak_over_diameter": target,
                "hgast_digitization_uncertainty": uncertainty,
                "absolute_error": error,
                "acceptance_tolerance": tolerance,
                "pass": error <= tolerance,
            }
        )
    passed = all(bool(row["pass"]) for row in comparisons)
    return {
        "schema": CAMPAIGN_SCHEMA,
        "gate": "post-freeze analytical held-out screen",
        "coefficient_fingerprint_sha256": coefficients[
            "coefficient_fingerprint_sha256"
        ],
        "coefficients_changed_after_held_out_access": False,
        "figure10_path": AMBIENT_V2_REFERENCE_IDENTITIES[
            "all_sc_figure10_peak_digitized.csv"
        ]["path"],
        "figure10_sha256": AMBIENT_V2_REFERENCE_IDENTITIES[
            "all_sc_figure10_peak_digitized.csv"
        ]["sha256"],
        "figure10_bytes": AMBIENT_V2_REFERENCE_IDENTITIES[
            "all_sc_figure10_peak_digitized.csv"
        ]["bytes"],
        "status": "pass" if passed else "fail",
        "pass": passed,
        "comparisons": comparisons,
        "decision": (
            "simple response closure passed the pre-QBlade held-out gate"
            if passed
            else "simple response closure failed without retuning; a phase-continuous wake oscillator is the next model rung"
        ),
    }


def validate_ambient_v2_postfreeze_gate(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    file_index: Mapping[str, Mapping[str, Any]],
    coefficients: Mapping[str, Any],
    reference_paths: Mapping[str, Path],
) -> Dict[str, Any]:
    policy = manifest["calibration_policy"]
    gate_relative = str(policy["postfreeze_analytical_gate_file"])
    gate_identity = _verify_file_identity(
        Path(root), file_index, gate_relative, "Post-freeze analytical gate"
    )
    gate_path, _ = _safe_manifest_path(
        Path(root), gate_relative, "Post-freeze analytical gate"
    )
    expected = expected_ambient_v2_postfreeze_gate(
        coefficients,
        figure10_path=reference_paths[
            "all_sc_figure10_peak_digitized.csv"
        ],
        model_parameters_path=reference_paths[
            "vlastos2026_model_parameters.csv"
        ],
    )
    expected_bytes = (
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    actual_bytes = gate_path.read_bytes()
    try:
        actual = json.loads(actual_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Post-freeze analytical gate is invalid JSON.") from exc
    if actual != expected or actual_bytes != expected_bytes:
        raise ProtocolError(
            "Ambient-v2 post-freeze gate does not exactly reproduce the frozen equation."
        )
    if expected["status"] != "fail" or expected["pass"] is not False:
        raise ProtocolError("Ambient-v2 post-freeze gate must retain its known failure.")
    if (
        policy.get("postfreeze_analytical_gate_status") != "fail"
        or policy.get("postfreeze_analytical_gate_sha256")
        != gate_identity["sha256"]
        or policy.get("held_out_targets_accessed_during_calibration") is not False
        or policy.get("coefficients_changed_after_held_out_access") is not False
        or policy.get("held_out_targets_used_for_calibration") is not False
        or policy.get("coefficients_changed_after_holdout_evaluation") is not False
    ):
        raise ProtocolError(
            "Ambient-v2 calibration policy does not preserve the frozen failed gate/no-retune decision."
        )
    return expected


_AMBIENT_V2_CASE_KEYS = {
    "case_id",
    "scruton_number",
    "qblade_implied_scruton_number",
    "scruton_relative_difference_percent",
    "role",
    "reduced_velocity",
    "mean_inflow_mps",
    "reynolds_number",
    "timestep_level",
    "timestep_s",
    "steps_per_natural_cycle",
    "num_steps",
    "duration_s",
    "duration_policy",
    "stationary_maximum_cycles",
    "structure",
    "sim_path",
    "parameter_file",
    "response_model_config",
    "coefficient_fingerprint_sha256",
}
_AMBIENT_V2_STRUCTURE_KEYS = {
    "scruton_number",
    "natural_frequency_hz",
    "equivalent_mass_kg_per_m",
    "damping_ratio",
    "g_prime_max",
    "h_prime_max",
    "u_star_max_figure_digitized",
}
_AMBIENT_V2_STRUCTURE_ROWS = {
    2.94: {
        "scruton_number": 2.94,
        "natural_frequency_hz": 20.02,
        "equivalent_mass_kg_per_m": 0.2311,
        "damping_ratio": 0.0030,
        "g_prime_max": 10.578932,
        "h_prime_max": 1.440342,
        "u_star_max_figure_digitized": 6.8,
    },
    5.18: {
        "scruton_number": 5.18,
        "natural_frequency_hz": 15.38,
        "equivalent_mass_kg_per_m": 0.4311,
        "damping_ratio": 0.0028,
        "g_prime_max": 7.908404,
        "h_prime_max": 1.797174,
        "u_star_max_figure_digitized": 6.5,
    },
    7.02: {
        "scruton_number": 7.02,
        "natural_frequency_hz": 13.18,
        "equivalent_mass_kg_per_m": 0.6027,
        "damping_ratio": 0.0027,
        "g_prime_max": 5.714756,
        "h_prime_max": 2.090286,
        "u_star_max_figure_digitized": 6.2,
    },
}


def validate_ambient_v2_manifest_contract(
    manifest: Mapping[str, Any],
) -> None:
    """Validate the complete v2 manifest schema and canonical semantics."""

    _require_exact_keys(
        manifest,
        {
            "schema",
            "scope",
            "reference",
            "calibration_policy",
            "numerics",
            "qblade_mapping",
            "cases",
            "files",
            "response_model_semantics",
            "equation_reproduction_gate",
            "runtime_contract",
        },
        "Ambient-v2 campaign manifest",
    )
    if manifest.get("schema") != CAMPAIGN_SCHEMA:
        raise ProtocolError("Ambient-v2 campaign schema changed.")
    if manifest.get("scope") != (
        "external hGAST response comparison and model-form assessment; not equation-level code "
        "verification, a resolved-wake model, or certification validation"
    ):
        raise ProtocolError("Ambient-v2 campaign scope changed.")
    policy = _require_exact_keys(
        manifest.get("calibration_policy"),
        {
            "calibration_scruton_number",
            "held_out_scruton_numbers",
            "coefficient_file",
            "coefficient_file_sha256",
            "coefficient_fingerprint_sha256",
            "held_out_targets_accessed_during_calibration",
            "postfreeze_analytical_gate_file",
            "postfreeze_analytical_gate_sha256",
            "postfreeze_analytical_gate_status",
            "coefficients_changed_after_held_out_access",
            "held_out_targets_used_for_calibration",
            "coefficients_changed_after_holdout_evaluation",
        },
        "Ambient-v2 calibration_policy",
    )
    expected_policy = {
        "calibration_scruton_number": 5.18,
        "held_out_scruton_numbers": [2.94, 7.02],
        "coefficient_file": "frozen_response_coefficients.json",
        "coefficient_fingerprint_sha256": (
            AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
        ),
        "held_out_targets_accessed_during_calibration": False,
        "postfreeze_analytical_gate_file": "postfreeze_analytical_gate.json",
        "postfreeze_analytical_gate_status": "fail",
        "coefficients_changed_after_held_out_access": False,
        "held_out_targets_used_for_calibration": False,
        "coefficients_changed_after_holdout_evaluation": False,
    }
    for name, expected in expected_policy.items():
        if policy.get(name) != expected:
            raise ProtocolError(
                f"Ambient-v2 calibration_policy.{name} is not frozen."
            )
    _require_sha256(
        policy.get("coefficient_file_sha256"),
        "Ambient-v2 coefficient file SHA",
    )
    _require_sha256(
        policy.get("postfreeze_analytical_gate_sha256"),
        "Ambient-v2 post-freeze gate SHA",
    )

    numerics = _require_exact_keys(
        manifest.get("numerics"),
        {
            "minimum_global_steps_per_cycle",
            "dt_halving_enabled",
            "duration_requested_s",
            "duration_policy",
            "stationary_protocol_schema",
            "stationary_protocol_compatible_full_grid",
            "qualification_input_eligible",
            "required_sil_vorlap_history_mode",
            "response_window_cycles",
        },
        "Ambient-v2 numerics",
    )
    if numerics.get("minimum_global_steps_per_cycle") != 50.0:
        raise ProtocolError("Ambient-v2 minimum steps per cycle changed.")
    if numerics.get("required_sil_vorlap_history_mode") != "compact":
        raise ProtocolError("Ambient-v2 history mode changed.")
    if numerics.get("response_window_cycles") != 40.0:
        raise ProtocolError("Ambient-v2 response window changed.")
    qualification = numerics.get("qualification_input_eligible") is True
    if qualification:
        expected_numerics = {
            "minimum_global_steps_per_cycle": 50.0,
            "dt_halving_enabled": True,
            "duration_requested_s": None,
            "duration_policy": "frozen_stationary_cycle_caps",
            "stationary_protocol_schema": PROTOCOL_SCHEMA,
            "stationary_protocol_compatible_full_grid": True,
            "qualification_input_eligible": True,
            "required_sil_vorlap_history_mode": "compact",
            "response_window_cycles": 40.0,
        }
        if numerics != expected_numerics:
            raise ProtocolError("Ambient-v2 qualification numerics are not canonical.")
    elif numerics.get("stationary_protocol_compatible_full_grid") is not False:
        raise ProtocolError(
            "Nonqualification ambient-v2 campaign cannot claim full-grid compatibility."
        )

    expected_mapping = {
        "representation": (
            "one isolated HAWT blade beam used as the horizontal cantilever cylinder"
        ),
        "inflow_direction": "+Y_g/+Y_b chord",
        "response_direction": (
            "global +X; QBlade X_b tip deflection; VorLap normal is -X"
        ),
        "native_aerodynamics": False,
        "persistent_fixed_wake_source": "disabled with force_scale=0",
        "startup_seed": {
            "duration_cycles": 5.0,
            "target_amplitude_ratio": 0.005,
            "waveform": "hann_sine_squared at the natural frequency",
            "nodal_weighting": "projection_mass_times_mode",
            "exactly_zero_after_seed": True,
        },
        "response_closure": (
            "frozen-coefficient response-aware single-mode aerodynamic damping"
        ),
        "scruton_mapping": {
            "definition": "Sc=4*pi*m_equivalent*zeta/(rho*D^2)",
            "decision": (
                "match the published Table 1 equivalent mass, damping, and natural frequency exactly; "
                "retain the existing case air density; report the resulting Sc mismatch explicitly"
            ),
            "air_density_kg_per_m3": AIR_DENSITY_KGPM3,
            "not_an_exact_scruton_match": True,
            "reason_no_density_retune": (
                "No single inferred density reconciles all three rounded Table 1 mass, damping, and Sc values, "
                "so changing density would trade one undocumented mismatch for another."
            ),
            "published_sc_used_for_published_regressions": True,
            "qblade_implied_sc_used_for_response_energy_balance": True,
        },
    }
    if manifest.get("qblade_mapping") != expected_mapping:
        raise ProtocolError("Ambient-v2 qblade_mapping is not canonical.")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ProtocolError("Ambient-v2 immutable files table is empty.")
    for index, row in enumerate(raw_files):
        _require_exact_keys(
            row, {"path", "sha256", "bytes"}, f"Ambient-v2 files[{index}]"
        )
    file_paths = [str(row["path"]) for row in raw_files]
    if file_paths != sorted(file_paths) or len(file_paths) != len(set(file_paths)):
        raise ProtocolError(
            "Ambient-v2 immutable file table order or uniqueness changed."
        )

    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ProtocolError("Ambient-v2 campaign has no cases.")
    if qualification and len(raw_cases) != 72:
        raise ProtocolError("Ambient-v2 qualification campaign must have 72 cases.")
    for index, entry in enumerate(raw_cases):
        _require_exact_keys(entry, _AMBIENT_V2_CASE_KEYS, f"Ambient-v2 cases[{index}]")
        structure = _require_exact_keys(
            entry.get("structure"),
            _AMBIENT_V2_STRUCTURE_KEYS,
            f"Ambient-v2 cases[{index}].structure",
        )
        try:
            sc = float(entry["scruton_number"])
            u_star = float(entry["reduced_velocity"])
            timestep = float(entry["timestep_s"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"Ambient-v2 case {index} has invalid numerics.") from exc
        if sc not in _AMBIENT_V2_STRUCTURE_ROWS or structure != (
            _AMBIENT_V2_STRUCTURE_ROWS[sc]
        ):
            raise ProtocolError(
                f"Ambient-v2 case {index} structural row is not canonical."
            )
        level = str(entry["timestep_level"])
        expected_id = ambient_v2_case_id(sc, u_star, level)
        if entry.get("case_id") != expected_id:
            raise ProtocolError(
                f"Ambient-v2 case_id does not reproduce its Sc/U*/timestep: {entry.get('case_id')!r}."
            )
        expected_role = "calibration" if sc == 5.18 else "held_out"
        expected_qblade_sc = float(STRUCTURES[sc]["qblade_implied_scruton_number"])
        expected_values = {
            "qblade_implied_scruton_number": expected_qblade_sc,
            "scruton_relative_difference_percent": (
                100.0 * (expected_qblade_sc - sc) / sc
            ),
            "role": expected_role,
            "mean_inflow_mps": u_star
            * float(structure["natural_frequency_hz"])
            * DIAMETER_M,
            "reynolds_number": u_star
            * float(structure["natural_frequency_hz"])
            * DIAMETER_M**2
            / AIR_KINEMATIC_VISCOSITY_M2PS,
            "steps_per_natural_cycle": 1.0
            / (float(structure["natural_frequency_hz"]) * timestep),
            "sim_path": f"cases/{expected_id}/{expected_id}.sim",
            "parameter_file": (
                f"cases/{expected_id}/model/Control/vorlap_qblade_external.json"
            ),
            "coefficient_fingerprint_sha256": (
                AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
            ),
        }
        for name, expected in expected_values.items():
            if entry.get(name) != expected:
                raise ProtocolError(
                    f"Ambient-v2 case {expected_id} field {name} is not canonical."
                )
        if qualification:
            expected_timestep = (
                COARSE_TIMESTEP_S if level == "dt" else FINE_TIMESTEP_S
            )
            if timestep != expected_timestep:
                raise ProtocolError(
                    f"Ambient-v2 qualification case {expected_id} timestep changed."
                )
            duration_bound = derive_duration_bound(sc, u_star)
            maximum = duration_bound["maximum_endpoint"]
            if (
                entry.get("duration_policy") != "frozen_stationary_cycle_cap"
                or entry.get("stationary_maximum_cycles")
                != duration_bound["maximum_cycles"]
                or entry.get("num_steps") != maximum[f"{level}_steps"]
                or entry.get("duration_s") != maximum["endpoint_time_s"]
            ):
                raise ProtocolError(
                    f"Ambient-v2 qualification case {expected_id} duration is not canonical."
                )
        else:
            requested_duration = numerics.get("duration_requested_s")
            if (
                not isinstance(requested_duration, (int, float))
                or isinstance(requested_duration, bool)
                or not math.isfinite(float(requested_duration))
                or float(requested_duration) <= 0.0
                or entry.get("duration_policy") != "explicit_fixed_diagnostic"
                or entry.get("stationary_maximum_cycles") is not None
            ):
                raise ProtocolError(
                    f"Ambient-v2 diagnostic case {expected_id} duration policy is not canonical."
                )
            expected_steps = int(math.ceil(float(requested_duration) / timestep))
            if (
                entry.get("num_steps") != expected_steps
                or entry.get("duration_s") != expected_steps * timestep
            ):
                raise ProtocolError(
                    f"Ambient-v2 diagnostic case {expected_id} duration is inconsistent."
                )
    if qualification:
        expected_case_order = [
            ambient_v2_case_id(sc, u_star, level)
            for sc in sorted(STRUCTURES)
            for u_star in REDUCED_VELOCITIES
            for level in ("dt", "dt_half")
        ]
        if [str(entry["case_id"]) for entry in raw_cases] != expected_case_order:
            raise ProtocolError(
                "Ambient-v2 qualification case order is not canonical."
            )


def ambient_v2_qblade_dependency_manifest(qblade_root: Path) -> List[Dict[str, Any]]:
    root = Path(qblade_root).resolve()
    libraries = root / "Libraries"
    wrapper = root / AMBIENT_V2_QBLADE_SIL_WRAPPER_IDENTITY["path"]
    if not libraries.is_dir() or not wrapper.is_file() or wrapper.is_symlink():
        raise ProtocolError("Pinned QBlade dependency tree is incomplete.")
    dependency_symlinks = [
        path for path in sorted(libraries.rglob("*")) if path.is_symlink()
    ]
    if dependency_symlinks:
        raise ProtocolError("Pinned QBlade dependency tree contains symlinks.")
    paths = [
        path
        for path in sorted(libraries.rglob("*"))
        if path.is_file()
    ]
    paths.append(wrapper)
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]
    by_path = {row["path"]: row for row in rows}
    if by_path.get(AMBIENT_V2_QBLADE_SIL_WRAPPER_IDENTITY["path"]) != (
        AMBIENT_V2_QBLADE_SIL_WRAPPER_IDENTITY
    ):
        raise ProtocolError("Pinned QBlade SIL wrapper identity changed.")
    if by_path.get(AMBIENT_V2_CHRONO_ENGINE_IDENTITY["path"]) != (
        AMBIENT_V2_CHRONO_ENGINE_IDENTITY
    ):
        raise ProtocolError("Pinned Chrono structural runtime identity changed.")
    return rows


def _runtime_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def ambient_v2_runtime_contract(campaign_root: Path) -> Dict[str, Any]:
    """Return portable identities for every launch-time implementation input."""

    repo_root = Path(__file__).resolve().parents[2]
    runner = repo_root / "run_headless" / "run_qblade_sim.py"
    result_filter = Path(campaign_root).resolve() / "cantilever_results.filter"
    if not runner.is_file():
        raise ProtocolError(f"Pinned QBlade runner is unavailable: {runner}")
    if not result_filter.is_file():
        raise ProtocolError(
            f"Campaign-local result filter is unavailable: {result_filter}"
        )
    if (
        sha256_file(result_filter) != AMBIENT_V2_RESULT_FILTER_IDENTITY["sha256"]
        or result_filter.stat().st_size
        != AMBIENT_V2_RESULT_FILTER_IDENTITY["bytes"]
    ):
        raise ProtocolError("Campaign-local result filter is not the frozen filter.")
    python_root = repo_root / "vorlap"
    python_assets = [
        {
            "relative_path": path.relative_to(python_root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(python_root.rglob("*.py"))
        if path.is_file()
    ]
    if not python_assets:
        raise ProtocolError("Pinned VorLap Python runtime is empty.")
    qblade_root = (
        repo_root / "run_headless" / "QBladeCE_2.0.9.7"
    )
    for label, expected in (
        ("QBlade library", AMBIENT_V2_QBLADE_LIBRARY_IDENTITY),
        ("VorLap/QBlade bridge", AMBIENT_V2_BRIDGE_LIBRARY_IDENTITY),
    ):
        path = qblade_root / str(expected["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected["sha256"]
            or path.stat().st_size != expected["bytes"]
        ):
            raise ProtocolError(f"Pinned {label} identity changed.")
    qblade_dependencies = ambient_v2_qblade_dependency_manifest(qblade_root)
    return {
        "schema": AMBIENT_V2_RUNTIME_CONTRACT_SCHEMA,
        "runner": {
            "path": "run_headless/run_qblade_sim.py",
            "sha256": sha256_file(runner),
            "bytes": runner.stat().st_size,
        },
        "result_filter": dict(AMBIENT_V2_RESULT_FILTER_IDENTITY),
        "vorlap_python_runtime_assets": python_assets,
        "vorlap_python_runtime_fingerprint": sha256_payload(python_assets),
        "qblade_library": dict(AMBIENT_V2_QBLADE_LIBRARY_IDENTITY),
        "bridge_library": dict(AMBIENT_V2_BRIDGE_LIBRARY_IDENTITY),
        "qblade_sil_wrapper": dict(
            AMBIENT_V2_QBLADE_SIL_WRAPPER_IDENTITY
        ),
        "qblade_runtime_dependencies": qblade_dependencies,
        "qblade_runtime_dependency_fingerprint": sha256_payload(
            qblade_dependencies
        ),
        "numerical_runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "packages": {
                name: _runtime_package_version(name)
                for name in ("numpy", "scipy", "h5py")
            },
            "reproducibility_scope": (
                "runtime-version provenance only; this contract does not assert "
                "cross-environment bitwise reproducibility; record a digest-pinned "
                "container at launch"
            ),
        },
    }


def validate_ambient_v2_runtime_contract(
    root: Path,
    manifest: Mapping[str, Any],
    file_index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    contract = manifest.get("runtime_contract")
    expected = ambient_v2_runtime_contract(Path(root))
    if contract != expected:
        raise ProtocolError(
            "Ambient-v2 runtime contract differs from the pinned runner/VorLap/QBlade identities."
        )
    filter_identity = _verify_file_identity(
        Path(root),
        file_index,
        str(expected["result_filter"]["path"]),
        "Campaign-local QBlade result filter",
    )
    if (
        filter_identity["sha256"] != expected["result_filter"]["sha256"]
        or filter_identity["bytes"] != expected["result_filter"]["bytes"]
    ):
        raise ProtocolError("Campaign-local result-filter identity changed.")
    return expected


def _case_binding(
    root: Path,
    file_index: Mapping[str, Mapping[str, Any]],
    entry: Mapping[str, Any],
    structure: Mapping[str, Any],
    maximum: Mapping[str, Any],
    response_semantics: Mapping[str, Any],
) -> Dict[str, Any]:
    case_id = str(entry.get("case_id", ""))
    if not case_id:
        raise ProtocolError("Campaign case is missing case_id.")
    timestep_level = entry.get("timestep_level")
    if timestep_level not in {"dt", "dt_half"}:
        raise ProtocolError(f"Case {case_id} has unsupported timestep_level.")
    expected_timestep = (
        COARSE_TIMESTEP_S if timestep_level == "dt" else FINE_TIMESTEP_S
    )
    _require_close(entry.get("timestep_s"), expected_timestep, f"Case {case_id} timestep_s")
    expected_steps = maximum[f"{timestep_level}_steps"]
    if entry.get("num_steps") != expected_steps:
        raise ProtocolError(
            f"Case {case_id} num_steps={entry.get('num_steps')!r} does not match "
            f"frozen maximum endpoint {expected_steps}."
        )
    _require_close(
        entry.get("duration_s"),
        float(maximum["endpoint_time_s"]),
        f"Case {case_id} duration_s",
    )
    _require_close(
        entry.get("qblade_implied_scruton_number"),
        float(structure["qblade_implied_scruton_number"]),
        f"Case {case_id} qblade_implied_scruton_number",
    )
    if entry.get("role") != structure["role"]:
        raise ProtocolError(f"Case {case_id} role does not match the frozen protocol.")
    raw_structure = entry.get("structure")
    if not isinstance(raw_structure, dict):
        raise ProtocolError(f"Case {case_id} has no structure object.")
    _require_close(
        raw_structure.get("natural_frequency_hz"),
        float(structure["natural_frequency_hz"]),
        f"Case {case_id} structure.natural_frequency_hz",
    )
    _require_close(
        raw_structure.get("damping_ratio"),
        float(structure["structural_damping_ratio"]),
        f"Case {case_id} structure.damping_ratio",
    )
    _require_close(
        raw_structure.get("equivalent_mass_kg_per_m"),
        float(structure["equivalent_mass_kg_per_m"]),
        f"Case {case_id} structure.equivalent_mass_kg_per_m",
    )

    sim_path = str(entry.get("sim_path", ""))
    parameter_file = str(entry.get("parameter_file", ""))
    expected_prefix = f"cases/{case_id}/"
    if not sim_path.startswith(expected_prefix) or not parameter_file.startswith(
        expected_prefix
    ):
        raise ProtocolError(f"Case {case_id} paths do not remain inside its case directory.")
    sim_identity = _verify_file_identity(
        root, file_index, sim_path, f"Case {case_id} simulation input"
    )
    sim_file, _sim_relative = _safe_manifest_path(
        root, sim_path, f"Case {case_id} simulation input"
    )
    sim_settings = _parse_required_qblade_settings(
        sim_file,
        (
            "ISOFFSHORE",
            "NUMTIMESTEPS",
            "TIMESTEP",
            "MEANINF",
            "TURBFILE",
            "INITIAL_YAW",
            "INITIAL_PITCH",
            "INITIAL_AZIMUTH",
            "STRSUBSTEP",
            "RELAXSTEPS",
            "RELAXTIMESTEP",
            "PRESCRIBETYPE",
            "RPMPRESCRIBED",
            "STRITERATIONS",
            "MODNEWTONITER",
            "DENSITYAIR",
            "VISCOSITYAIR",
            "GRAVITY",
            "INCLUDEAERO",
            "INCLUDEHYDRO",
            "RAMPUP",
            "ADDDAMP",
            "WNDTYPE",
            "HORANGLE",
            "VERTANGLE",
            "PROFILETYPE",
            "SHEAREXP",
            "GROUNDEFFECT",
            "GLOBPOS_X",
            "GLOBPOS_Y",
            "GLOBPOS_Z",
            "GLOBROT_X",
            "GLOBROT_Y",
            "GLOBROT_Z",
            "STOREFROM",
            "STOREREPLAY",
            "STOREAERO",
            "STOREBLADE",
            "STORESTRUCT",
            "STORESIM",
            "STOREHYDRO",
            "STORECONTROLLER",
            "CALCMODAL",
        ),
        f"Case {case_id} simulation input",
    )
    _require_blank_qblade_settings(
        sim_file,
        ("EVENTFILE", "LOADINGFILE", "SIMFILE", "MOTIONFILE"),
        f"Case {case_id} simulation input",
    )
    try:
        observed_num_steps_float = float(sim_settings["NUMTIMESTEPS"])
        observed_num_steps = int(observed_num_steps_float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError(
            f"Case {case_id} NUMTIMESTEPS is not an integer."
        ) from exc
    if observed_num_steps_float != observed_num_steps or observed_num_steps != expected_steps:
        raise ProtocolError(
            f"Case {case_id} NUMTIMESTEPS={sim_settings['NUMTIMESTEPS']!r} "
            f"does not match {expected_steps}."
        )
    _require_close(
        sim_settings["TIMESTEP"],
        expected_timestep,
        f"Case {case_id} simulation TIMESTEP",
    )
    expected_mean_inflow = (
        float(entry["reduced_velocity"])
        * float(structure["natural_frequency_hz"])
        * DIAMETER_M
    )
    _require_close(
        entry.get("mean_inflow_mps"),
        expected_mean_inflow,
        f"Case {case_id} mean_inflow_mps",
    )
    _require_close(
        sim_settings["MEANINF"],
        expected_mean_inflow,
        f"Case {case_id} simulation MEANINF",
    )
    for keyword, expected in {
        "ISOFFSHORE": 0.0,
        "DENSITYAIR": AIR_DENSITY_KGPM3,
        "VISCOSITYAIR": AIR_KINEMATIC_VISCOSITY_M2PS,
        "GRAVITY": 9.80665,
        "INITIAL_YAW": 0.0,
        "INITIAL_PITCH": 0.0,
        "INITIAL_AZIMUTH": 0.0,
        "STRSUBSTEP": 2.0,
        "RELAXSTEPS": 0.0,
        "RELAXTIMESTEP": 0.0,
        "PRESCRIBETYPE": 1.0,
        "RPMPRESCRIBED": 0.0,
        "STRITERATIONS": 5.0,
        "MODNEWTONITER": 0.0,
        "INCLUDEAERO": 0.0,
        "INCLUDEHYDRO": 0.0,
        "RAMPUP": 0.0,
        "ADDDAMP": 0.0,
        "WNDTYPE": 0.0,
        "HORANGLE": 90.0,
        "VERTANGLE": 0.0,
        "PROFILETYPE": 0.0,
        "SHEAREXP": 0.0,
        "GLOBPOS_X": 0.0,
        "GLOBPOS_Y": 0.0,
        "GLOBPOS_Z": 0.0,
        "GLOBROT_X": 0.0,
        "GLOBROT_Y": 0.0,
        "GLOBROT_Z": 0.0,
        "STOREFROM": 0.0,
    }.items():
        _require_close(
            sim_settings[keyword],
            expected,
            f"Case {case_id} simulation {keyword}",
        )
    for keyword, expected in {
        "GROUNDEFFECT": "false",
        "STOREREPLAY": "false",
        "STOREAERO": "false",
        "STOREBLADE": "false",
        "STORESTRUCT": "true",
        "STORESIM": "true",
        "STOREHYDRO": "false",
        "STORECONTROLLER": "false",
        "CALCMODAL": "false",
    }.items():
        if sim_settings[keyword].strip().lower() != expected:
            raise ProtocolError(
                f"Case {case_id} simulation {keyword} must be {expected}."
            )
    turbine_raw = Path(sim_settings["TURBFILE"])
    if turbine_raw.is_absolute() or ".." in turbine_raw.parts:
        raise ProtocolError(f"Case {case_id} TURBFILE is unsafe.")
    turbine_candidate = (Path(sim_path).parent / turbine_raw).as_posix()
    turbine_identity = _verify_file_identity(
        root,
        file_index,
        turbine_candidate,
        f"Case {case_id} turbine input",
    )
    if not turbine_candidate.startswith(expected_prefix):
        raise ProtocolError(f"Case {case_id} TURBFILE escapes its case directory.")
    turbine_path, turbine_relative = _safe_manifest_path(
        root, turbine_candidate, f"Case {case_id} turbine input"
    )
    config_identity = _verify_file_identity(
        root, file_index, parameter_file, f"Case {case_id} parameter config"
    )
    config_path, _relative = _safe_manifest_path(
        root, parameter_file, f"Case {case_id} parameter config"
    )
    config = _load_json_object(config_path, f"Case {case_id} parameter config")
    _require_close(
        config.get("force_scale"), 0.0, f"Case {case_id} force_scale"
    )
    _validate_runtime_mapping_config(
        config,
        config_path=config_path,
        sim_path=sim_file,
        timestep_s=expected_timestep,
        case_id=case_id,
    )
    response_model = _validate_response_model(
        config.get("response_model"),
        structure,
        case_id,
        response_schema_version=int(
            response_semantics["response_schema_version"]
        ),
    )
    if entry.get("response_model_config") != response_model:
        raise ProtocolError(
            f"Case {case_id} manifest/config response_model objects differ."
        )
    airfoil_source_binding = None
    if bool(response_semantics["v2"]):
        airfoil_source_binding = validate_ambient_v2_airfoil_source(
            root=root,
            file_index=file_index,
            case_id=case_id,
            parameter_file=parameter_file,
            config=config,
            coefficient_fingerprint=AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256,
        )
    structural_links = _validate_structural_and_blade_files(
        root=root,
        file_index=file_index,
        turbine_path=turbine_path,
        turbine_relative=turbine_relative,
        parameter_file=parameter_file,
        case_prefix=expected_prefix,
        structure=structure,
        case_id=case_id,
    )

    bundle_rows = _case_bundle_rows(case_id, file_index)
    bundle_hash = sha256_payload(bundle_rows)
    entry_hash = sha256_payload(entry)
    response_model_hash = sha256_payload(response_model)
    binding_payload = {
        "case_id": case_id,
        "case_manifest_entry_sha256": entry_hash,
        "case_input_bundle_sha256": bundle_hash,
        "config_sha256": config_identity["sha256"],
        "response_model_sha256": response_model_hash,
        "sim_sha256": sim_identity["sha256"],
        "turbine_sha256": turbine_identity["sha256"],
        "main_structure_sha256": structural_links["main_structure_sha256"],
        "beam_structure_sha256": structural_links["beam_structure_sha256"],
        "blade_geometry_sha256": structural_links["blade_geometry_sha256"],
    }
    if airfoil_source_binding is not None:
        binding_payload["airfoil_source_sha256"] = airfoil_source_binding[
            "sha256"
        ]
        binding_payload["airfoil_source_bytes"] = airfoil_source_binding[
            "bytes"
        ]
        binding_payload["airfoil_source_manifest_sha256"] = (
            airfoil_source_binding["manifest_sha256"]
        )
    return {
        **binding_payload,
        "case_binding_sha256": sha256_payload(binding_payload),
        "timestep_level": timestep_level,
        "timestep_s": expected_timestep,
        "num_steps": expected_steps,
        "duration_s": maximum["endpoint_time_s"],
    }


def _campaign_state(manifest_path: Path) -> Dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = _load_json_object(path, "campaign manifest")
    if manifest.get("schema") != CAMPAIGN_SCHEMA:
        raise ProtocolError("Unsupported Vlastos campaign schema.")
    root = path.parent
    file_index = _manifest_file_index(root, manifest)
    response_semantics = validate_response_semantics(manifest)
    reference_paths: Dict[str, Path] = {}
    if bool(response_semantics["v2"]):
        validate_ambient_v2_manifest_contract(manifest)
        validate_ambient_v2_campaign_tree(root, file_index)
        reference_paths = validate_ambient_v2_reference_files(
            root, manifest, file_index
        )
        validate_ambient_v2_runtime_contract(root, manifest, file_index)

    numerics = manifest.get("numerics")
    if not isinstance(numerics, dict):
        raise ProtocolError("Campaign has no numerics object.")
    _require_text(
        numerics.get("duration_policy"),
        "frozen_stationary_cycle_caps",
        "Campaign duration policy",
    )
    _require_boolean_text(
        numerics.get("stationary_protocol_compatible_full_grid"),
        True,
        "Campaign stationary full-grid compatibility",
    )
    _require_boolean_text(
        numerics.get("qualification_input_eligible"),
        True,
        "Campaign qualification-input eligibility",
    )
    _require_boolean_text(
        numerics.get("dt_halving_enabled"),
        True,
        "Campaign timestep-halving policy",
    )
    _require_text(
        numerics.get("required_sil_vorlap_history_mode"),
        "compact",
        "Campaign SIL-history mode",
    )

    calibration = manifest.get("calibration_policy")
    if not isinstance(calibration, dict):
        raise ProtocolError("Campaign has no calibration_policy object.")
    coefficient_file = str(calibration.get("coefficient_file", ""))
    coefficient_identity = _verify_file_identity(
        root, file_index, coefficient_file, "Frozen response-coefficient file"
    )
    if coefficient_identity["sha256"] != calibration.get("coefficient_file_sha256"):
        raise ProtocolError("Frozen response-coefficient hash differs from calibration_policy.")
    coefficient_path, _relative = _safe_manifest_path(
        root, coefficient_file, "Frozen response-coefficient file"
    )
    coefficients = _load_json_object(coefficient_path, "frozen response coefficients")
    coefficient_fingerprint = _require_sha256(
        calibration.get("coefficient_fingerprint_sha256"),
        "calibration coefficient fingerprint",
    )
    if coefficients.get("coefficient_fingerprint_sha256") != coefficient_fingerprint:
        raise ProtocolError("Frozen coefficient fingerprint differs from calibration_policy.")
    if bool(response_semantics["v2"]):
        if coefficient_fingerprint != (
            AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
        ):
            raise ProtocolError(
                "Ambient-v2 coefficient fingerprint is not the frozen portable identity."
            )
        coefficient_payload = dict(coefficients)
        coefficient_payload.pop("coefficient_fingerprint_sha256", None)
        if sha256_payload(coefficient_payload) != coefficient_fingerprint:
            raise ProtocolError(
                "Ambient-v2 coefficient object does not reproduce its portable fingerprint."
            )
        validate_ambient_v2_postfreeze_gate(
            root=root,
            manifest=manifest,
            file_index=file_index,
            coefficients=coefficients,
            reference_paths=reference_paths,
        )

    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ProtocolError("Campaign cases must be a list.")
    if len(raw_cases) != len(STRUCTURES) * len(REDUCED_VELOCITIES) * 2:
        raise ProtocolError("Stationary campaign must contain exactly 72 cases.")
    grouped: Dict[Tuple[float, float], Dict[str, Mapping[str, Any]]] = {}
    seen_ids = set()
    for raw_entry in raw_cases:
        if not isinstance(raw_entry, dict):
            raise ProtocolError("Every campaign case must be an object.")
        case_id = str(raw_entry.get("case_id", ""))
        if not case_id or case_id in seen_ids:
            raise ProtocolError(f"Campaign case_id is missing or duplicated: {case_id!r}")
        seen_ids.add(case_id)
        ambient_namespace_prefix = f"{AMBIENT_V2_CASE_ID_NAMESPACE}_"
        if bool(response_semantics["v2"]):
            if not case_id.startswith(ambient_namespace_prefix):
                raise ProtocolError(
                    f"Ambient-v2 case {case_id!r} is outside the frozen "
                    f"{AMBIENT_V2_CASE_ID_NAMESPACE!r} result/checkpoint namespace."
                )
        elif case_id.startswith(ambient_namespace_prefix):
            raise ProtocolError(
                f"Legacy case {case_id!r} uses the reserved ambient-v2 namespace."
            )
        if raw_entry.get("coefficient_fingerprint_sha256") != (
            coefficient_fingerprint
        ):
            raise ProtocolError(
                f"Case {case_id} coefficient fingerprint differs from the "
                "frozen calibration object."
            )
        sc = _match_frozen_value(
            raw_entry.get("scruton_number"), tuple(STRUCTURES), f"Case {case_id} Sc"
        )
        u_star = _match_frozen_value(
            raw_entry.get("reduced_velocity"),
            REDUCED_VELOCITIES,
            f"Case {case_id} U*",
        )
        level = raw_entry.get("timestep_level")
        if level not in {"dt", "dt_half"}:
            raise ProtocolError(f"Case {case_id} has invalid timestep level.")
        pair = grouped.setdefault((sc, u_star), {})
        if level in pair:
            raise ProtocolError(f"Duplicate {level} case for Sc={sc:.2f}, U*={u_star:.2f}.")
        pair[str(level)] = raw_entry

    expected_pairs = {
        (sc, u_star) for sc in STRUCTURES for u_star in REDUCED_VELOCITIES
    }
    if set(grouped) != expected_pairs:
        raise ProtocolError("Campaign case grid does not match the frozen 3x12x2 grid.")

    pair_rows = []
    for sc in sorted(STRUCTURES):
        structure = STRUCTURES[sc]
        for u_star in REDUCED_VELOCITIES:
            derivation = derive_duration_bound(sc, u_star)
            maximum = derivation["maximum_endpoint"]
            source_pair = grouped[(sc, u_star)]
            if set(source_pair) != {"dt", "dt_half"}:
                raise ProtocolError(
                    f"Sc={sc:.2f}, U*={u_star:.2f} lacks a complete timestep pair."
                )
            bindings = {
                level: _case_binding(
                    root,
                    file_index,
                    source_pair[level],
                    structure,
                    maximum,
                    response_semantics,
                )
                for level in ("dt", "dt_half")
            }
            if bindings["dt"]["duration_s"] != bindings["dt_half"]["duration_s"]:
                raise ProtocolError("Paired time steps do not share an exact endpoint.")
            pair_rows.append(
                {
                    "scruton_number": sc,
                    "role": structure["role"],
                    "reduced_velocity": u_star,
                    "duration_bound": derivation,
                    "evaluation_endpoints": evaluation_schedule(sc, u_star),
                    "cases": bindings,
                }
            )

    # Verify every declared immutable file, including non-case shared inputs.
    for relative in sorted(file_index):
        _verify_file_identity(root, file_index, relative, "Campaign input")

    state = {
        "manifest_sha256": sha256_file(path),
        "manifest_schema": manifest["schema"],
        "coefficient_file_sha256": coefficient_identity["sha256"],
        "coefficient_fingerprint_sha256": coefficient_fingerprint,
        "case_pairs": pair_rows,
    }
    if bool(response_semantics["v2"]):
        state["response_semantics_sha256"] = response_semantics[
            "response_semantics_sha256"
        ]
        state["response_model_semantics"] = response_semantics[
            "response_model_semantics"
        ]
        state["equation_reproduction_gate"] = response_semantics[
            "equation_reproduction_gate"
        ]
    return state


def _protocol_policy() -> Dict[str, Any]:
    return {
        "execution_mode": "fixed_full_cap_then_terminal_analysis",
        "adaptive_stop_or_continuation_enabled": False,
        "decision_ledger_qualification_enabled": ADAPTIVE_DECISION_SCHEMA_ENABLED,
        "planning_first_evaluation_cycles": FIRST_EVALUATION_CYCLES,
        "planning_extension_cycles": EXTENSION_CYCLES,
        "minimum_cap_cycles": EARLIEST_ACCEPTANCE_CYCLES,
        "terminal_failure_status": "incomplete_nonstationary",
        "endpoint_alignment": {
            "nominal_timestep_s": COARSE_TIMESTEP_S,
            "half_timestep_s": FINE_TIMESTEP_S,
            "rule": "fine_steps=2*nominal_steps at identical endpoint_time_ns",
        },
        "duration_derivation": {
            "response_ratio_equation": "R=Ustar*H1*activation/(2*Sc_QB)",
            "linear_exponent_equation": "g_cycle=2*pi*zeta*(R-1)",
            "active_envelope_equation": (
                "a(N)=a_eq/sqrt(1+((a_eq/a0)^2-1)*exp(-2*g_cycle*N))"
            ),
            "active_equilibrium_equation": "a_eq=a_sat*sqrt(1-1/R)",
            "active_seed_lower_bound_ratio": ACTIVE_SEED_LOWER_BOUND_RATIO,
            "requested_seed_target_ratio": REQUESTED_SEED_TARGET_RATIO,
            "requested_seed_upper_sensitivity_ratio": (
                REQUESTED_SEED_UPPER_SENSITIVITY_RATIO
            ),
            "stable_initial_amplitude_ratio": STABLE_INITIAL_AMPLITUDE_RATIO,
            "stable_initial_rationale": (
                "ten times the requested seed upper sensitivity to cover transient "
                "and off-branch contamination"
            ),
            "terminal_absolute_tolerance_ratio": (
                TERMINAL_ABSOLUTE_TOLERANCE_RATIO
            ),
            "terminal_relative_tolerance": TERMINAL_RELATIVE_TOLERANCE,
            "final_observation_cycles": FINAL_OBSERVATION_CYCLES,
            "round_up_cycles": EXTENSION_CYCLES,
            "minimum_cap_cycles": EARLIEST_ACCEPTANCE_CYCLES,
            "interpretation": (
                "planning bound only; measured stationarity gates control acceptance"
            ),
        },
        "qualification_rule": (
            "run every immutable input to its conservative fixed cap, then require "
            "hash-bound terminal stationarity and runtime checks"
        ),
    }


def build_protocol(manifest_path: Path) -> Dict[str, Any]:
    """Build a deterministic protocol object after validating all campaign inputs."""

    state = _campaign_state(manifest_path)
    campaign_binding = {
        key: state[key]
        for key in (
            "manifest_schema",
            "manifest_sha256",
            "coefficient_file_sha256",
            "coefficient_fingerprint_sha256",
        )
    }
    if "response_semantics_sha256" in state:
        campaign_binding["response_semantics_sha256"] = state[
            "response_semantics_sha256"
        ]
    payload = {
        "schema": PROTOCOL_SCHEMA,
        "scope": (
            "fixed conservative stationarity caps for the 72-case Vlastos "
            "response-aware campaign; not adaptive execution or response validation"
        ),
        "campaign_binding": campaign_binding,
        "policy": _protocol_policy(),
        "case_pairs": state["case_pairs"],
    }
    return {**payload, "protocol_sha256": sha256_payload(payload)}


def _validate_protocol_document(protocol: Mapping[str, Any]) -> None:
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ProtocolError("Unsupported stationary protocol schema.")
    supplied_hash = _require_sha256(
        protocol.get("protocol_sha256"), "protocol_sha256"
    )
    payload = dict(protocol)
    del payload["protocol_sha256"]
    observed_hash = sha256_payload(payload)
    if supplied_hash != observed_hash:
        raise ProtocolError(
            f"Protocol self-hash mismatch: expected {supplied_hash}, observed {observed_hash}."
        )
    if protocol.get("policy") != _protocol_policy():
        raise ProtocolError("Protocol policy differs from the frozen implementation.")


def validate_protocol(
    protocol: Mapping[str, Any], manifest_path: Path
) -> Dict[str, Any]:
    """Fail closed unless protocol, campaign, cases, and configs still match."""

    _validate_protocol_document(protocol)
    expected = build_protocol(manifest_path)
    if protocol != expected:
        raise ProtocolError(
            "Protocol content does not match the current hash-verified campaign."
        )
    return {
        "valid": True,
        "schema": PROTOCOL_SCHEMA,
        "protocol_sha256": protocol["protocol_sha256"],
        "campaign_manifest_sha256": protocol["campaign_binding"][
            "manifest_sha256"
        ],
        "case_count": 72,
        "case_pair_count": 36,
    }


def _pair_key(scruton_number: float, reduced_velocity: float) -> str:
    return f"sc{scruton_number:.2f}_u{reduced_velocity:.2f}"


def _protocol_pair(
    protocol: Mapping[str, Any], scruton_number: float, reduced_velocity: float
) -> Mapping[str, Any]:
    sc = _match_frozen_value(scruton_number, tuple(STRUCTURES), "scruton_number")
    u_star = _match_frozen_value(
        reduced_velocity, REDUCED_VELOCITIES, "reduced_velocity"
    )
    matches = [
        row
        for row in protocol.get("case_pairs", [])
        if row.get("scruton_number") == sc
        and row.get("reduced_velocity") == u_star
    ]
    if len(matches) != 1:
        raise ProtocolError(f"Protocol lacks unique pair {_pair_key(sc, u_star)}.")
    return matches[0]


def _decision_record(
    protocol: Mapping[str, Any],
    prior_records: Sequence[Mapping[str, Any]],
    *,
    scruton_number: float,
    reduced_velocity: float,
    evaluation_cycles: int,
    gate_results: Mapping[str, bool],
) -> Dict[str, Any]:
    pair = _protocol_pair(protocol, scruton_number, reduced_velocity)
    sc = float(pair["scruton_number"])
    u_star = float(pair["reduced_velocity"])
    pair_records = [
        row
        for row in prior_records
        if row.get("scruton_number") == sc
        and row.get("reduced_velocity") == u_star
    ]
    if pair_records and pair_records[-1].get("decision") != "continue":
        raise ProtocolError(f"Pair {_pair_key(sc, u_star)} is already terminal.")
    schedule = pair["evaluation_endpoints"]
    next_index = len(pair_records)
    if next_index >= len(schedule):
        raise ProtocolError(f"Pair {_pair_key(sc, u_star)} exhausted its cap.")
    endpoint = schedule[next_index]
    if evaluation_cycles != endpoint["evaluation_cycles"]:
        raise ProtocolError(
            f"Pair {_pair_key(sc, u_star)} next endpoint is "
            f"{endpoint['evaluation_cycles']} cycles, not {evaluation_cycles}."
        )
    if not isinstance(gate_results, dict) or not gate_results:
        raise ProtocolError("gate_results must be a nonempty object.")
    if any(not isinstance(name, str) or not name for name in gate_results):
        raise ProtocolError("Every gate name must be a nonempty string.")
    if any(type(value) is not bool for value in gate_results.values()):
        raise ProtocolError("Every gate result must be a boolean.")
    normalized_gates = {name: gate_results[name] for name in sorted(gate_results)}
    all_passed = all(normalized_gates.values())
    previous_count = (
        int(pair_records[-1]["consecutive_passing_endpoints"])
        if pair_records
        else 0
    )
    consecutive = previous_count + 1 if all_passed else 0
    maximum_cycles = int(pair["duration_bound"]["maximum_cycles"])
    if consecutive >= CONSECUTIVE_PASSING_ENDPOINTS:
        decision = "accept"
    elif evaluation_cycles == maximum_cycles:
        decision = "incomplete_nonstationary"
    else:
        decision = "continue"
    case_bindings = {
        level: {
            key: pair["cases"][level][key]
            for key in ("case_id", "case_binding_sha256", "config_sha256")
        }
        for level in ("dt", "dt_half")
    }
    payload = {
        "schema": LEDGER_RECORD_SCHEMA,
        "sequence": len(prior_records),
        "protocol_sha256": protocol["protocol_sha256"],
        "campaign_manifest_sha256": protocol["campaign_binding"][
            "manifest_sha256"
        ],
        "previous_record_sha256": (
            prior_records[-1]["record_sha256"] if prior_records else _GENESIS_SHA256
        ),
        "pair_key": _pair_key(sc, u_star),
        "scruton_number": sc,
        "reduced_velocity": u_star,
        "evaluation_cycles": evaluation_cycles,
        "endpoint_time_ns": endpoint["endpoint_time_ns"],
        "dt_steps": endpoint["dt_steps"],
        "dt_half_steps": endpoint["dt_half_steps"],
        "case_bindings": case_bindings,
        "gate_results": normalized_gates,
        "all_gates_passed": all_passed,
        "consecutive_passing_endpoints": consecutive,
        "decision": decision,
    }
    return {**payload, "record_sha256": sha256_payload(payload)}


def build_decision_record(
    protocol: Mapping[str, Any],
    prior_records: Sequence[Mapping[str, Any]],
    *,
    scruton_number: float,
    reduced_velocity: float,
    evaluation_cycles: int,
    gate_results: Mapping[str, bool],
) -> Dict[str, Any]:
    """Reject use of the unintegrated adaptive-decision prototype."""

    raise ProtocolError(
        "Adaptive decision records are disabled: current execution uses fixed "
        "full-cap inputs followed by the hash-bound terminal analyzer."
    )


def validate_ledger_records(
    records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate sequence, hash chain, bindings, schedules, and decisions."""

    _validate_protocol_document(protocol)
    if records:
        raise ProtocolError(
            "Adaptive decision ledgers are disabled for qualification until "
            "evidence-derived fixed gates are integrated."
        )
    if not isinstance(records, (list, tuple)):
        raise ProtocolError("Decision ledger records must be a sequence.")
    validated: List[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ProtocolError(f"Ledger record {index} is not an object.")
        if record.get("schema") != LEDGER_RECORD_SCHEMA:
            raise ProtocolError(f"Ledger record {index} has an unsupported schema.")
        expected = _decision_record(
            protocol,
            validated,
            scruton_number=record.get("scruton_number"),
            reduced_velocity=record.get("reduced_velocity"),
            evaluation_cycles=record.get("evaluation_cycles"),
            gate_results=record.get("gate_results"),
        )
        if record != expected:
            raise ProtocolError(
                f"Ledger record {index} does not match its hash-bound deterministic decision."
            )
        validated.append(record)
    terminal = sum(row["decision"] != "continue" for row in validated)
    return {
        "valid": True,
        "records": len(validated),
        "terminal_pairs": terminal,
        "ledger_fingerprint_sha256": sha256_payload(list(validated)),
        "ledger_tip_sha256": (
            validated[-1]["record_sha256"] if validated else _GENESIS_SHA256
        ),
    }


def read_ledger(path: Path) -> List[Dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        return []
    records = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ProtocolError(
                        f"Decision ledger contains blank line {line_number}."
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ProtocolError(
                        f"Decision ledger line {line_number} is not an object."
                    )
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Could not read decision ledger {source}: {exc}") from exc
    return records


def derive_publication_plan(
    protocol: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Reject the unintegrated adaptive-to-publication prototype."""

    raise ProtocolError(
        "Publication-plan derivation from adaptive decisions is disabled; use "
        "the immutable fixed-cap campaign and its terminal analysis."
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_output_path(output: Path, campaign_root: Path, label: str) -> Path:
    destination = Path(output).expanduser().resolve()
    if _is_within(destination, campaign_root.resolve()):
        raise ProtocolError(
            f"{label} must be outside the immutable generated campaign root."
        )
    return destination


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ProtocolError(f"Refusing to replace existing output: {destination}") from exc
    return destination


def _load_protocol(path: Path) -> Dict[str, Any]:
    return _load_json_object(Path(path), "stationary protocol")


def _load_gate_results(path: Path) -> Dict[str, bool]:
    value = _load_json_object(path, "gate results")
    if not value or any(type(item) is not bool for item in value.values()):
        raise ProtocolError("Gate-results JSON must map names to booleans.")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate protocol JSON.")
    generate.add_argument("--campaign-manifest", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate protocol/campaign.")
    validate.add_argument("--campaign-manifest", type=Path, required=True)
    validate.add_argument("--protocol", type=Path, required=True)
    validate.add_argument("--ledger", type=Path)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    campaign_manifest = args.campaign_manifest.expanduser().resolve()
    campaign_root = campaign_manifest.parent

    if args.command == "generate":
        output = _safe_output_path(args.output, campaign_root, "Protocol output")
        protocol = build_protocol(campaign_manifest)
        write_json_exclusive(output, protocol)
        validate_protocol(protocol, campaign_manifest)
        print(json.dumps({"status": "valid", "protocol": str(output)}, sort_keys=True))
        return 0

    protocol = _load_protocol(args.protocol)
    validation = validate_protocol(protocol, campaign_manifest)
    if args.command == "validate":
        if args.ledger is not None:
            validation["ledger"] = validate_ledger_records(
                read_ledger(args.ledger), protocol
            )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"Unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
