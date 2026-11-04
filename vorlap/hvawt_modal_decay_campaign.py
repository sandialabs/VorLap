"""Immutable H-VAWT aero-off Rayleigh-damping discriminator.

This is a deliberately small, no-inflow campaign.  It compares six exact
structural-input interpretations at the well-resolved 1-RPM mode 12; each
excited case has a byte-matched zero-force baseline.  The resulting twelve
cases distinguish the ambiguous substructure tail, station-vector Rayleigh
damping, file-scalar Rayleigh damping, and the residual all-beta-zero decay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .qblade_files import read_qblade_keyword, replace_qblade_keyword_file
from .qblade_modal_decay import ModalBurstPlan, SilModalBurstCoupler
from .qblade_nodes import (
    canonical_node_id,
    extract_structural_node_ids,
    position_channel_name,
    velocity_channel_name,
)
from .qblade_run import QBladeRunRequest, QBladeRunResult, run_qblade_case
from .qblade_run_support import (
    atomic_write_json,
    build_modal_native_bridge_runtime_identity,
    file_sha256,
    find_qblade_library,
    qblade_runtime_dependency_manifest,
)
from .qblade_sil import normalize_qblade_header, read_qblade_result_numeric_columns


CAMPAIGN_SCHEMA = "vorlap-hvawt-modal-decay-campaign-v3"
SOURCE_CONTRACT_SCHEMA = "vorlap-hvawt-modal-decay-source-contract-v1"
CASE_SCHEMA = "vorlap-hvawt-modal-decay-case-v3"
ANALYSIS_SCHEMA = "vorlap-hvawt-modal-decay-analysis-v3"
FULL_CASE_COUNT = 12
NOMINAL_AMPLITUDE_M = 0.001
TIMESTEP_S = 0.0005
ROTOR_SPEED_RPM = 1.0
TARGET_MODE = 12
VARIANTS = (
    "O",
    "S",
    "C20",
    "C20_station_zero",
    "C20_scalar_zero",
    "all_beta_zero",
)
BURST_CYCLES = 5
END_TIME_S = 6.8
MINIMUM_FREE_CYCLES = 20.0
ROTATION_SIGN = 1
AZIMUTH_REFERENCE_DEG = 0.0
TIME_CHANNEL = "Time [s]"
TIMESTEP_CHANNEL = "Timestep [-]"
AZIMUTH_CHANNEL = "LSS Azimuthal Pos. [deg]"
RPM_CHANNEL = "LSS Rpm [rpm]"
NOMINAL_STIFFNESS_LABEL = "nominal undamped 1-RPM stiffness scale"
RESULT_CHANNEL_COUNT = 238

_SUBSTRUCTURE_TAIL_VARIANT = {
    "O": "O",
    "S": "S",
    "C20": "C20",
    "C20_station_zero": "C20",
    "C20_scalar_zero": "C20",
    "all_beta_zero": "C20_ZERO",
}
_STATION_ZERO_VARIANTS = frozenset({"C20_station_zero", "all_beta_zero"})
_SCALAR_ZERO_VARIANTS = frozenset({"C20_scalar_zero", "all_beta_zero"})

MODAL_FREQUENCIES_HZ: dict[float, dict[int, float]] = {
    0.0: {6: 2.511557314896635, 12: 4.572512385909129},
    1.0: {6: 2.5115545017693828, 12: 4.572593676110874},
}
NOMINAL_1RPM_STIFFNESS_N_PER_M = {
    6: 84112.78376212968,
    12: 102171.33806478149,
}

_SIM_PATCHES: tuple[tuple[str, object], ...] = (
    ("PRESCRIBETYPE", 1),
    ("INITIAL_AZIMUTH", 0),
    ("RELAXSTEPS", 0),
    ("INCLUDEAERO", 0),
    ("INCLUDEHYDRO", 0),
    ("RAMPUP", 0),
    ("ADDDAMP", 0),
    ("WNDTYPE", 0),
    ("WNDNAME", ""),
    ("MEANINF", 0),
    ("STOREFROM", 0),
    ("STOREREPLAY", "false"),
    ("ISOFFSHORE", 0),
    ("WAKEINTERACTION", 0),
    ("SURF_CURR_U", 0),
    ("SUB_CURR_U", 0),
    ("SHORE_CURR_U", 0),
    ("EVENTFILE", ""),
    ("LOADINGFILE", ""),
    ("SIMFILE", ""),
    ("MOTIONFILE", ""),
    ("WAVEFILE", ""),
    ("MOORINGSYSTEM", ""),
)


@dataclass(frozen=True)
class DecayCase:
    case_id: str
    variant: str
    timestep_s: float
    rotor_speed_rpm: float
    role: str
    target_mode: int
    nominal_translation_m: float
    frequency_hz: float
    generalized_force_amplitude_n: float
    total_wrapper_updates: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModalDecayFit:
    classification: str
    frequency_hz: float
    damping_ratio: float
    normalized_rmse: float
    decay_rates_per_s: tuple[float, ...]
    peak_translation_m: float
    burst_end_translation_m: float
    available_free_cycles: float
    sample_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256(path: Path) -> str:
    identity = file_sha256(path)
    if identity is None:
        raise FileNotFoundError(path)
    return str(identity["sha256"])


def _canonical_sha256(payload: object) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _read_locked_json(path: Path, *, label: str) -> tuple[dict[str, object], dict[str, object]]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    before = resolved.stat()
    raw = resolved.read_bytes()
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
        or before.st_ino != after.st_ino
        or before.st_dev != after.st_dev
    ):
        raise RuntimeError(f"{label} changed while it was read: {resolved}")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {resolved}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved}")
    return parsed, {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _tree_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Immutable campaign input tree contains a symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path
    return result


def validate_source_case(
    source_root: Path,
    source_contract_path: Path,
) -> dict[str, object]:
    """Verify the source tree byte-for-byte before copying any candidate case."""

    root_candidate = Path(source_root).expanduser()
    if root_candidate.is_symlink():
        raise ValueError(f"Source case root must not be a symlink: {root_candidate}")
    root = root_candidate.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    contract, contract_identity = _read_locked_json(
        Path(source_contract_path),
        label="H-VAWT modal-decay source contract",
    )
    if contract.get("schema") != SOURCE_CONTRACT_SCHEMA:
        raise ValueError(f"Source contract schema must be {SOURCE_CONTRACT_SCHEMA!r}.")
    expected_files = contract.get("files_sha256")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("Source contract must bind a nonempty files_sha256 object.")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in expected_files.items()):
        raise ValueError("Source contract file paths and hashes must be strings.")
    observed_files = _tree_files(root)
    if set(observed_files) != set(expected_files):
        raise ValueError(
            "Source case file set does not match the exact contract: "
            f"missing={sorted(set(expected_files) - set(observed_files))}, "
            f"unexpected={sorted(set(observed_files) - set(expected_files))}."
        )
    mismatches = [
        relative
        for relative, path in observed_files.items()
        if _sha256(path) != str(expected_files[relative]).lower()
    ]
    if mismatches:
        raise ValueError(
            "Source case SHA-256 mismatch for: " + ", ".join(sorted(mismatches))
        )
    sim_relative = contract.get("sim_relative_path")
    if not isinstance(sim_relative, str) or sim_relative not in observed_files:
        raise ValueError("Source contract sim_relative_path is absent from files_sha256.")
    settings = contract.get("expected_sim_settings")
    if not isinstance(settings, dict):
        raise ValueError("Source contract expected_sim_settings must be an object.")
    setting_mismatches: list[str] = []
    for keyword, expected in settings.items():
        observed = read_qblade_keyword(observed_files[sim_relative], str(keyword))
        if observed != str(expected):
            setting_mismatches.append(
                f"{keyword}: expected {expected!r}, observed {observed!r}"
            )
    if setting_mismatches:
        raise ValueError(
            "Source case settings do not match the contract: "
            + "; ".join(setting_mismatches)
        )
    return {
        "schema": "vorlap-hvawt-modal-decay-source-audit-v1",
        "passed": True,
        "source_root": str(root),
        "contract": contract_identity,
        "file_count": len(observed_files),
        "source_tree_fingerprint": _canonical_sha256(
            [{"path": key, "sha256": expected_files[key]} for key in sorted(expected_files)]
        ),
        "sim_relative_path": sim_relative,
        "substructure_relative_path": contract.get("substructure_relative_path"),
        "legacy_external_config_relative_path": contract.get(
            "legacy_external_config_relative_path"
        ),
    }


def nominal_hann_force_amplitude_n(
    target_mode: int,
    nominal_translation_m: float,
) -> float:
    """Return the locked five-cycle no-tuning generalized-force scale."""

    if target_mode not in NOMINAL_1RPM_STIFFNESS_N_PER_M:
        raise ValueError(f"No locked nominal stiffness is available for mode {target_mode}.")
    translation = float(nominal_translation_m)
    if not math.isfinite(translation) or translation < 0.0:
        raise ValueError("Nominal translation must be finite and nonnegative.")
    stiffness = NOMINAL_1RPM_STIFFNESS_N_PER_M[target_mode]
    return 2.0 * stiffness * translation / (math.pi * BURST_CYCLES)


def numerical_hann_sdof_realized_amplitude_m(
    *,
    stiffness_n_per_m: float,
    mass_kg: float,
    force_amplitude_n: float,
    cycles: int = BURST_CYCLES,
    steps_per_cycle: int = 4000,
) -> float:
    """RK4 check of the continuous undamped resonant Hann-burst mapping."""

    stiffness = float(stiffness_n_per_m)
    mass = float(mass_kg)
    amplitude = float(force_amplitude_n)
    if stiffness <= 0.0 or mass <= 0.0 or cycles < 1 or steps_per_cycle < 100:
        raise ValueError("SDOF stiffness, mass, cycles, and resolution must be positive.")
    omega = math.sqrt(stiffness / mass)
    period = 2.0 * math.pi / omega
    duration = cycles * period
    dt = period / int(steps_per_cycle)
    steps = int(cycles * steps_per_cycle)
    x = 0.0
    v = 0.0

    def acceleration(time_s: float, displacement_m: float) -> float:
        hann = 0.5 * (1.0 - math.cos(2.0 * math.pi * time_s / duration))
        force = amplitude * hann * math.cos(omega * time_s)
        return (force - stiffness * displacement_m) / mass

    for index in range(steps):
        time_s = index * dt
        k1x = v
        k1v = acceleration(time_s, x)
        k2x = v + 0.5 * dt * k1v
        k2v = acceleration(time_s + 0.5 * dt, x + 0.5 * dt * k1x)
        k3x = v + 0.5 * dt * k2v
        k3v = acceleration(time_s + 0.5 * dt, x + 0.5 * dt * k2x)
        k4x = v + dt * k3v
        k4v = acceleration(time_s + dt, x + dt * k3x)
        x += dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        v += dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
    return math.hypot(x, v / omega)


def analytical_hann_sdof_realized_amplitude_m(
    *,
    stiffness_n_per_m: float,
    force_amplitude_n: float,
    cycles: int = BURST_CYCLES,
) -> float:
    """Exact post-burst amplitude for the locked undamped resonant SDOF."""

    stiffness = float(stiffness_n_per_m)
    amplitude = float(force_amplitude_n)
    if stiffness <= 0.0 or cycles < 1 or not math.isfinite(amplitude):
        raise ValueError("SDOF stiffness, force amplitude, and cycles must be valid.")
    return abs(amplitude) * math.pi * int(cycles) / (2.0 * stiffness)


def _case_id(variant: str, role: str) -> str:
    return f"m12_rpm1_dt0500us_{variant}_{role}"


def _make_case(*, variant: str, role: str) -> DecayCase:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown modal-decay variant {variant!r}.")
    if role not in {"excited", "baseline"}:
        raise ValueError(f"Unknown modal-decay role {role!r}.")
    frequency = MODAL_FREQUENCIES_HZ[ROTOR_SPEED_RPM][TARGET_MODE]
    total_updates = int(round(END_TIME_S / TIMESTEP_S)) + 2
    nominal = 0.0 if role == "baseline" else NOMINAL_AMPLITUDE_M
    force = 0.0 if role == "baseline" else nominal_hann_force_amplitude_n(
        TARGET_MODE, nominal
    )
    return DecayCase(
        case_id=_case_id(variant, role),
        variant=variant,
        timestep_s=TIMESTEP_S,
        rotor_speed_rpm=ROTOR_SPEED_RPM,
        role=role,
        target_mode=TARGET_MODE,
        nominal_translation_m=nominal,
        frequency_hz=frequency,
        generalized_force_amplitude_n=force,
        total_wrapper_updates=total_updates,
    )


def campaign_cases() -> tuple[DecayCase, ...]:
    cases = [
        _make_case(variant=variant, role=role)
        for variant in VARIANTS
        for role in ("baseline", "excited")
    ]
    if (
        len(cases) != FULL_CASE_COUNT
        or len({case.case_id for case in cases}) != FULL_CASE_COUNT
    ):
        raise RuntimeError("Internal modal-decay matrix is not the locked 12-case design.")
    free_cycles = (
        END_TIME_S - BURST_CYCLES / MODAL_FREQUENCIES_HZ[1.0][TARGET_MODE]
    ) * MODAL_FREQUENCIES_HZ[1.0][TARGET_MODE]
    if free_cycles < MINIMUM_FREE_CYCLES:
        raise RuntimeError("Locked modal-decay duration does not provide 20 free cycles.")
    return tuple(cases)


def _line_content_and_ending(line: bytes) -> tuple[bytes, bytes]:
    if line.endswith(b"\r\n"):
        return line[:-2], b"\r\n"
    if line.endswith((b"\n", b"\r")):
        return line[:-1], line[-1:]
    return line, b""


def _token_spans(line: bytes) -> list[re.Match[bytes]]:
    return list(re.finditer(rb"\S+", line))


def _subelements_layout(lines: Sequence[bytes]) -> tuple[int, list[int]]:
    try:
        table_index = next(
            index for index, line in enumerate(lines) if line.strip() == b"SUBELEMENTS"
        )
    except StopIteration as exc:
        raise ValueError("Substructure does not contain a SUBELEMENTS table.") from exc
    header_index = table_index + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1
    row_indices: list[int] = []
    for index in range(header_index + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped == b"SUBMEMBERS":
            break
        if not stripped:
            continue
        tokens = stripped.split()
        try:
            float(tokens[0])
        except (ValueError, IndexError):
            continue
        row_indices.append(index)
    if not row_indices:
        raise ValueError("Substructure SUBELEMENTS table has no numeric rows.")
    return header_index, row_indices


def apply_substructure_variant(path: Path, variant: str) -> dict[str, object]:
    """Apply the O/S/C20 tail transform without reformatting unrelated bytes."""

    tail_variant = _SUBSTRUCTURE_TAIL_VARIANT.get(variant, variant)
    if tail_variant not in {"O", "S", "C20", "C20_ZERO"}:
        raise ValueError(f"Unknown modal-decay substructure variant {variant!r}.")
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    header_index, row_indices = _subelements_layout(lines)
    header_content, header_ending = _line_content_and_ending(lines[header_index])
    header_spans = _token_spans(header_content)
    header = [match.group().decode("ascii") for match in header_spans]
    if len(header) != 21 or header[-3:] != ["DIA_[m]", "Drag[-]", "DAMP[-]"]:
        raise ValueError(
            "Frozen source SUBELEMENTS header must end DIA_[m], Drag[-], DAMP[-]."
        )
    rows_before: list[list[str]] = []
    rows_after: list[list[str]] = []
    for index in row_indices:
        content, ending = _line_content_and_ending(lines[index])
        spans = _token_spans(content)
        tokens = [match.group().decode("ascii") for match in spans]
        if len(tokens) != 21:
            raise ValueError("Frozen source SUBELEMENTS rows must contain exactly 21 values.")
        if not math.isclose(float(tokens[18]), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Frozen source SUBELEMENTS DIA column must equal 1.0 m.")
        if not math.isclose(float(tokens[19]), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Frozen source SUBELEMENTS column 19 must equal 1.0.")
        if not math.isclose(float(tokens[20]), 0.02, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Frozen source SUBELEMENTS column 20 must equal 0.02.")
        before = list(tokens)
        if tail_variant == "S":
            left = spans[19]
            right = spans[20]
            lines[index] = (
                content[: left.start()]
                + right.group()
                + content[left.end() : right.start()]
                + left.group()
                + content[right.end() :]
                + ending
            )
            tokens[19], tokens[20] = tokens[20], tokens[19]
        elif tail_variant in {"C20", "C20_ZERO"}:
            left = spans[19]
            right = spans[20]
            damping = (
                _zero_numeric_lexeme_same_width(right.group())
                if tail_variant == "C20_ZERO"
                else right.group()
            )
            lines[index] = (
                content[: left.start()]
                + damping
                + content[right.end() :]
                + ending
            )
            tokens[19] = damping.decode("ascii")
            tokens = tokens[:20]
        rows_before.append(before)
        rows_after.append(list(tokens))
    if tail_variant in {"C20", "C20_ZERO"}:
        left = header_spans[19]
        right = header_spans[20]
        lines[header_index] = (
            header_content[: left.start()]
            + right.group()
            + header_content[right.end() :]
            + header_ending
        )
    patched = original if tail_variant == "O" else b"".join(lines)
    if tail_variant != "O":
        path.write_bytes(patched)
    header_after_content, _ = _line_content_and_ending(lines[header_index])
    return {
        "schema": "vorlap-hvawt-modal-decay-substructure-variant-v3",
        "variant": variant,
        "tail_variant": tail_variant,
        "only_subelements_tail_changed": True,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "result_sha256": hashlib.sha256(patched).hexdigest(),
        "header_before": header,
        "header_after": [
            match.group().decode("ascii")
            for match in _token_spans(header_after_content)
        ],
        "rows_before": rows_before,
        "rows_after": rows_after,
        "contract": {
            "O": "original row: DIA=1.0, numeric column 19=1.0, column 20=0.02",
            "S": "swap only the final two numeric lexemes; retain every other byte",
            "C20": (
                "delete original Drag numeric lexeme plus following separator and the "
                "matching header token; retain every other byte"
            ),
            "C20_ZERO": (
                "apply canonical C20 layout and zero its substructure DAMP lexeme; "
                "retain every other byte"
            ),
        }[tail_variant],
    }


def _zero_numeric_lexeme_same_width(raw: bytes) -> bytes:
    text = raw.decode("ascii")
    if not math.isfinite(float(text)):
        raise ValueError(f"Rayleigh damping token is nonfinite: {text!r}.")
    exponent_at = max(text.find("e"), text.find("E"))
    if exponent_at >= 0:
        mantissa, exponent = text[:exponent_at], text[exponent_at:]
    else:
        mantissa, exponent = text, ""
    zero_mantissa = "".join("0" if char.isdigit() else char for char in mantissa)
    result = (zero_mantissa + exponent).encode("ascii")
    if len(result) != len(raw) or float(result) != 0.0:
        raise ValueError(f"Could not zero Rayleigh token byte-preservingly: {text!r}.")
    return result


def zero_rayleigh_scalar(path: Path) -> dict[str, object]:
    """Zero the unique file-level RAYLEIGHDMP value and no other bytes."""

    original = path.read_bytes()
    pattern = re.compile(
        rb"^(?P<lead>[ \t]*)(?P<value>\S+)(?P<sep>[ \t]+)RAYLEIGHDMP(?P<tail>[^\r\n]*)(?P<ending>\r?\n|\r)?$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(original))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one RAYLEIGHDMP scalar in {path}.")
    match = matches[0]
    old = match.group("value")
    new = _zero_numeric_lexeme_same_width(old)
    patched = original[: match.start("value")] + new + original[match.end("value") :]
    path.write_bytes(patched)
    return {
        "schema": "vorlap-hvawt-rayleigh-scalar-zero-v1",
        "path": str(path),
        "old_value": old.decode("ascii"),
        "new_value": new.decode("ascii"),
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "result_sha256": hashlib.sha256(patched).hexdigest(),
        "unchanged_byte_count": len(original) - len(old),
    }


def zero_rayleigh_station_vector(path: Path) -> dict[str, object]:
    """Zero every station RAYLEIGHDMP_Vec lexeme and preserve all other bytes."""

    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    header_indices = [
        index
        for index, line in enumerate(lines)
        if b"RAYLEIGHDMP_Vec" in _line_content_and_ending(line)[0]
    ]
    if len(header_indices) != 1:
        raise ValueError(f"Expected exactly one RAYLEIGHDMP_Vec header in {path}.")
    header_index = header_indices[0]
    changed: list[dict[str, object]] = []
    started = False
    row_column_count: int | None = None
    for index in range(header_index + 1, len(lines)):
        content, ending = _line_content_and_ending(lines[index])
        spans = _token_spans(content)
        if not spans:
            if started:
                break
            continue
        try:
            float(spans[0].group())
        except ValueError:
            if started:
                break
            continue
        started = True
        if row_column_count is None:
            row_column_count = len(spans)
        if len(spans) != row_column_count or row_column_count < 2:
            raise ValueError(f"Station row column count changed in {path} at line {index + 1}.")
        last = spans[-1]
        old = last.group()
        new = _zero_numeric_lexeme_same_width(old)
        lines[index] = content[: last.start()] + new + content[last.end() :] + ending
        changed.append(
            {"line_number": index + 1, "old_value": old.decode(), "new_value": new.decode()}
        )
    if not changed:
        raise ValueError(f"No station Rayleigh values were found in {path}.")
    patched = b"".join(lines)
    path.write_bytes(patched)
    return {
        "schema": "vorlap-hvawt-rayleigh-station-zero-v1",
        "path": str(path),
        "changed_station_count": len(changed),
        "changes": changed,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "result_sha256": hashlib.sha256(patched).hexdigest(),
        "byte_count_unchanged": len(patched) == len(original),
    }


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _result_channels(external_config: Mapping[str, object]) -> list[str]:
    rows = external_config.get("external_output_rows")
    if not isinstance(rows, list) or len(rows) != 117:
        raise ValueError("Modal-decay external config must contain exactly 117 output actions.")
    ordered_rows = sorted(rows, key=lambda value: int(value["index"]))
    if [int(row["index"]) for row in ordered_rows] != list(range(120, 237)):
        raise ValueError("Modal-decay output actions must occupy exact indices 120..236.")
    channels = [TIME_CHANNEL, TIMESTEP_CHANNEL, AZIMUTH_CHANNEL, RPM_CHANNEL]
    for row in ordered_rows:
        channels.append(position_channel_name(row))
    for row in ordered_rows:
        node_id = canonical_node_id(str(row["id"]), float(row["position"]))
        channels.append(velocity_channel_name(node_id, str(row["direction"])))
    if len(channels) != RESULT_CHANNEL_COUNT or len(set(channels)) != RESULT_CHANNEL_COUNT:
        raise ValueError(
            f"Modal-decay result filter must contain {RESULT_CHANNEL_COUNT} unique channels."
        )
    return channels


def _storage_channels(
    external_config: Mapping[str, object],
    result_channels: Sequence[str],
) -> list[str]:
    rows = external_config.get("external_input_rows")
    if not isinstance(rows, list) or len(rows) != 120:
        raise ValueError("Modal-decay external config must contain exactly 120 input rows.")
    ordered_rows = sorted(rows, key=lambda value: int(value["index"]))
    if [int(row["index"]) for row in ordered_rows] != list(range(120)):
        raise ValueError("Modal-decay external inputs must occupy exact indices 0..119.")
    result = list(result_channels)
    for row in ordered_rows:
        variable = str(row["variable"])
        if variable not in result:
            result.append(variable)
    if result != list(result_channels):
        raise ValueError(
            "Result filter is missing one or more exact external input/time channels."
        )
    return result


def _copy_locked_artifact(source: Path, destination: Path) -> dict[str, object]:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Campaign artifact must be a regular non-symlink file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": destination.name,
        "sha256": _sha256(destination),
        "size_bytes": destination.stat().st_size,
    }


def _normalize_inert_external_config(
    path: Path,
    *,
    qblade_root: Path,
) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Legacy external config is not an object: {path}")
    raw["force_scale"] = 0.0
    raw["include_mean_load"] = False
    raw["response_model"] = None
    raw["source_parameter_dir"] = str(path.parent.resolve())
    raw["python_path_entries"] = []
    raw["parameter_file"] = f"Control/{path.name}"
    bridge = (
        Path(qblade_root).resolve()
        / "ControllerFiles"
        / "libvorlap_qblade_bridge.so"
    )
    raw["native_bridge_runtime_identity"] = (
        build_modal_native_bridge_runtime_identity(bridge)
    )
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return raw


def _qblade_runtime_contract(qblade_root: Path) -> dict[str, object]:
    root = Path(qblade_root).expanduser().resolve()
    dependencies, dependency_fingerprint = qblade_runtime_dependency_manifest(root)
    library = find_qblade_library(root)
    library_identity = file_sha256(library)
    if library_identity is None:
        raise FileNotFoundError(library)
    return {
        "schema": "vorlap-hvawt-modal-qblade-runtime-contract-v1",
        "qblade_root": str(root),
        "qblade_library": {
            "path": library.name,
            "sha256": library_identity["sha256"],
            "size_bytes": library_identity["size_bytes"],
        },
        "dependency_fingerprint": dependency_fingerprint,
        "dependencies": dependencies,
    }


def _patch_case_sim(
    sim_path: Path,
    case: DecayCase,
    *,
    storage_filter_name: str,
) -> list[dict[str, str]]:
    patches: list[dict[str, str]] = []
    for keyword, value in _SIM_PATCHES:
        patches.append(replace_qblade_keyword_file(sim_path, keyword, value))
    for keyword, value in (
        ("NUMTIMESTEPS", case.total_wrapper_updates),
        ("TIMESTEP", case.timestep_s),
        ("RELAXTIMESTEP", case.timestep_s),
        ("RPMPRESCRIBED", case.rotor_speed_rpm),
        ("FILTERFILE", storage_filter_name),
    ):
        patches.append(replace_qblade_keyword_file(sim_path, keyword, value))
    return patches


def _contract_path_list(
    contract: Mapping[str, object],
    key: str,
    expected_files: Mapping[str, object],
) -> tuple[str, ...]:
    raw = contract.get(key)
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"Source contract {key} must be a nonempty path list.")
    paths = tuple(str(item) for item in raw)
    if len(paths) != len(set(paths)) or any(path not in expected_files for path in paths):
        raise ValueError(f"Source contract {key} contains duplicate or unbound paths.")
    return paths


def _verify_exact_copied_source(
    case_root: Path,
    expected_files: Mapping[str, object],
) -> None:
    observed = _tree_files(case_root)
    if set(observed) != set(expected_files):
        raise RuntimeError("A copied source case does not retain the contracted file set.")
    mismatches = [
        relative
        for relative, path in observed.items()
        if _sha256(path) != str(expected_files[relative])
    ]
    if mismatches:
        raise RuntimeError(
            "A copied source case changed before transformation: "
            + ", ".join(sorted(mismatches))
        )


def _validate_action_source_contract(
    external: Mapping[str, object],
    main_structural_path: Path,
) -> dict[str, object]:
    rows = external.get("external_output_rows")
    if not isinstance(rows, list) or len(rows) != 117:
        raise ValueError("Action-source audit requires exactly 117 output rows.")
    ordered = sorted(rows, key=lambda row: int(row["index"]))
    action_sites: list[str] = []
    for offset in range(0, len(ordered), 3):
        triplet = ordered[offset : offset + 3]
        if [str(row["direction"]).upper() for row in triplet] != ["X", "Y", "Z"]:
            raise ValueError("Each structural action site must contain ordered X/Y/Z actions.")
        ids = {str(row["id"]) for row in triplet}
        positions = {float(row["position"]) for row in triplet}
        if len(ids) != 1 or len(positions) != 1:
            raise ValueError("Each action triplet must target one identical structural site.")
        action_sites.append(canonical_node_id(ids.pop(), positions.pop()))
    structural_sites = extract_structural_node_ids(
        main_structural_path.read_text(encoding="utf-8", errors="strict"),
        include_tower_nodes=False,
    )
    if action_sites != structural_sites:
        raise ValueError(
            "The 117 global actions do not match the frozen 39-site structural output order."
        )
    return {
        "schema": "vorlap-hvawt-modal-action-source-audit-v1",
        "passed": True,
        "action_count": len(ordered),
        "site_count": len(action_sites),
        "directions_per_site": ["X", "Y", "Z"],
        "ordered_sites_sha256": _canonical_sha256(action_sites),
        "main_structural_file": str(main_structural_path),
        "main_structural_sha256": _sha256(main_structural_path),
    }


def generate_campaign(
    *,
    source_root: Path,
    source_contract_path: Path,
    output_root: Path,
    action_shapes_path: Path,
    force_duals_path: Path,
    qblade_root: Path,
) -> dict[str, object]:
    """Generate the complete sealed 12-case input tree without running QBlade."""

    source_audit = validate_source_case(source_root, source_contract_path)
    source_contract, source_contract_identity = _read_locked_json(
        source_contract_path,
        label="H-VAWT modal-decay source contract",
    )
    runtime_contract = _qblade_runtime_contract(qblade_root)
    output = Path(output_root).expanduser().resolve(strict=False)
    if output.exists():
        raise FileExistsError(
            f"Campaign output already exists and will not be mutated: {output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    cases_root = output / "cases"
    inputs.mkdir()
    cases_root.mkdir()
    shapes_identity = _copy_locked_artifact(
        Path(action_shapes_path).expanduser().resolve(),
        inputs / "hvawt_response_mode_shapes.csv",
    )
    duals_identity = _copy_locked_artifact(
        Path(force_duals_path).expanduser().resolve(),
        inputs / "hvawt_response_mode_force_duals.csv",
    )
    contract_identity_copy = _copy_locked_artifact(
        Path(source_contract_path).expanduser().resolve(),
        inputs / "source_contract.json",
    )

    sim_relative = str(source_contract["sim_relative_path"])
    substructure_relative = str(source_contract["substructure_relative_path"])
    external_relative = str(source_contract["legacy_external_config_relative_path"])
    expected_source_files = dict(source_contract["files_sha256"])
    station_paths = _contract_path_list(
        source_contract,
        "blade_strut_station_damping_relative_paths",
        expected_source_files,
    )
    scalar_paths = _contract_path_list(
        source_contract,
        "rayleigh_scalar_relative_paths",
        expected_source_files,
    )
    main_structural_relative = source_contract.get("main_structural_relative_path")
    if (
        not isinstance(main_structural_relative, str)
        or main_structural_relative not in expected_source_files
    ):
        raise ValueError("Source contract must bind main_structural_relative_path.")
    case_rows: list[dict[str, object]] = []
    source_path = Path(source_root).expanduser().resolve()
    source_structural_hashes = {
        relative: digest
        for relative, digest in expected_source_files.items()
        if str(relative).lower().endswith(".str")
    }
    for case in campaign_cases():
        case_root = cases_root / case.case_id
        shutil.copytree(source_path, case_root, symlinks=False)
        _verify_exact_copied_source(case_root, expected_source_files)
        sim_path = case_root / sim_relative
        substructure_path = case_root / substructure_relative
        external_path = case_root / external_relative
        variant_audit = apply_substructure_variant(substructure_path, case.variant)
        station_audits = (
            [zero_rayleigh_station_vector(case_root / relative) for relative in station_paths]
            if case.variant in _STATION_ZERO_VARIANTS
            else []
        )
        scalar_audits = (
            [zero_rayleigh_scalar(case_root / relative) for relative in scalar_paths]
            if case.variant in _SCALAR_ZERO_VARIANTS
            else []
        )
        external = _normalize_inert_external_config(
            external_path,
            qblade_root=Path(qblade_root),
        )
        action_source_audit = _validate_action_source_contract(
            external,
            case_root / str(main_structural_relative),
        )
        result_channels = _result_channels(external)
        storage_channels = _storage_channels(external, result_channels)
        result_filter = case_root / "modal_decay_result.filter"
        storage_filter = case_root / "modal_decay_storage.filter"
        _write_lines(result_filter, result_channels)
        _write_lines(storage_filter, storage_channels)
        sim_patches = _patch_case_sim(
            sim_path,
            case,
            storage_filter_name=storage_filter.name,
        )
        plan = ModalBurstPlan(
            timestep_s=case.timestep_s,
            total_wrapper_updates=case.total_wrapper_updates,
            bootstrap_zero=True,
            rotor_speed_rpm=case.rotor_speed_rpm,
            target_mode=case.target_mode,
            frequency_hz=case.frequency_hz,
            generalized_force_amplitude_n=case.generalized_force_amplitude_n,
            burst_start_time_s=0.0,
            burst_cycles=BURST_CYCLES,
            azimuth_reference_deg=AZIMUTH_REFERENCE_DEG,
            rotation_sign=ROTATION_SIGN,
        )
        plan_path = case_root / "modal_burst_plan.json"
        plan_payload = {
            **plan.as_dict(),
            "artifacts": {
                "external_config": {
                    "path": external_path.relative_to(case_root).as_posix(),
                    "sha256": _sha256(external_path),
                },
                "action_shapes": {
                    "path": os.path.relpath(inputs / shapes_identity["path"], case_root),
                    "sha256": shapes_identity["sha256"],
                },
                "force_duals": {
                    "path": os.path.relpath(inputs / duals_identity["path"], case_root),
                    "sha256": duals_identity["sha256"],
                },
            },
        }
        plan_path.write_text(
            json.dumps(plan_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        modal_coupler = SilModalBurstCoupler.from_plan_file(plan_path)
        structural_hashes = {
            relative: _sha256(case_root / relative)
            for relative in source_structural_hashes
        }
        expected_changed = set()
        if _SUBSTRUCTURE_TAIL_VARIANT[case.variant] != "O":
            expected_changed.add(substructure_relative)
        if case.variant in _STATION_ZERO_VARIANTS:
            expected_changed.update(station_paths)
        if case.variant in _SCALAR_ZERO_VARIANTS:
            expected_changed.update(scalar_paths)
        observed_changed = {
            relative
            for relative in source_structural_hashes
            if structural_hashes[relative] != source_structural_hashes[relative]
        }
        if observed_changed != expected_changed:
            raise RuntimeError(
                f"Case {case.case_id} structural diff scope mismatch: "
                f"expected={sorted(expected_changed)}, observed={sorted(observed_changed)}."
            )
        analytical_realized = analytical_hann_sdof_realized_amplitude_m(
            stiffness_n_per_m=NOMINAL_1RPM_STIFFNESS_N_PER_M[case.target_mode],
            force_amplitude_n=case.generalized_force_amplitude_n,
        )
        modal_mass = NOMINAL_1RPM_STIFFNESS_N_PER_M[case.target_mode] / (
            2.0 * math.pi * case.frequency_hz
        ) ** 2
        numerical_realized = numerical_hann_sdof_realized_amplitude_m(
            stiffness_n_per_m=NOMINAL_1RPM_STIFFNESS_N_PER_M[case.target_mode],
            mass_kg=modal_mass,
            force_amplitude_n=case.generalized_force_amplitude_n,
        )
        case_payload: dict[str, object] = {
            "schema": CASE_SCHEMA,
            **case.as_dict(),
            "sim_relative_path": sim_relative,
            "plan_relative_path": plan_path.relative_to(case_root).as_posix(),
            "result_filter_relative_path": result_filter.relative_to(case_root).as_posix(),
            "storage_filter_relative_path": storage_filter.relative_to(case_root).as_posix(),
            "expected_result_name": "modal_decay_results",
            "expected_summary_name": f"{sim_path.stem}_run_summary.json",
            "expected_result_file_name": "modal_decay_results.txt",
            "burst_cycles": BURST_CYCLES,
            "end_time_s": END_TIME_S,
            "last_physical_sample_time_s": (
                case.total_wrapper_updates - 2
            ) * case.timestep_s,
            "declared_free_cycles": (
                END_TIME_S - BURST_CYCLES / case.frequency_hz
            ) * case.frequency_hz,
            "minimum_required_free_cycles": MINIMUM_FREE_CYCLES,
            "force_scale_equation": "Fhat = 2*k_q_1rpm*q_nominal/(pi*5)",
            "stiffness_scale_label": NOMINAL_STIFFNESS_LABEL,
            "nominal_stiffness_n_per_m": NOMINAL_1RPM_STIFFNESS_N_PER_M[
                case.target_mode
            ],
            "same_force_scale_at_0_and_1_rpm": True,
            "analytical_undamped_realized_translation_m": analytical_realized,
            "numerical_undamped_realized_translation_m": numerical_realized,
            "variant_audit": variant_audit,
            "station_zero_audits": station_audits,
            "scalar_zero_audits": scalar_audits,
            "action_source_audit": action_source_audit,
            "modal_plan_identity_fingerprint": modal_coupler.checkpoint_identity()[
                "fingerprint"
            ],
            "sim_patches": sim_patches,
            "structural_hashes": structural_hashes,
            "expected_modified_structural_relative_paths": sorted(expected_changed),
            "observed_modified_structural_relative_paths": sorted(observed_changed),
            "all_other_structural_bytes_unchanged": True,
            "native_callback_requirement": "zero init and update callbacks",
            "legacy_external_source_inert": {
                "force_scale": external.get("force_scale"),
                "include_mean_load": external.get("include_mean_load"),
                "response_model": external.get("response_model"),
            },
            "paired_case_id": _case_id(
                case.variant,
                "excited" if case.role == "baseline" else "baseline",
            ),
        }
        case_metadata = case_root / "case.json"
        atomic_write_json(case_metadata, case_payload)
        case_rows.append(case_payload)

    generated_files = _tree_files(output)
    files_sha256 = {
        relative: _sha256(path)
        for relative, path in generated_files.items()
        if relative != "campaign_manifest.json"
    }
    generator_identity = file_sha256(Path(__file__).resolve())
    if generator_identity is None:
        raise FileNotFoundError(__file__)
    manifest: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "sealed": True,
        "campaign_root": str(output),
        "no_inflow_cases": True,
        "qualification_scope": {
            "description": "complete compact Rayleigh ambiguity classifier",
            "mode": TARGET_MODE,
            "rotor_speed_rpm": ROTOR_SPEED_RPM,
            "timestep_s": TIMESTEP_S,
            "nominal_translation_m": NOMINAL_AMPLITUDE_M,
            "variants": list(VARIANTS),
            "paired_zero_force_baseline_per_variant": True,
            "full_total": FULL_CASE_COUNT,
        },
        "source_audit": source_audit,
        "source_contract_original": source_contract_identity,
        "source_contract_copy": contract_identity_copy,
        "modal_artifacts": {
            "action_shapes": shapes_identity,
            "force_duals": duals_identity,
        },
        "full_case_count": FULL_CASE_COUNT,
        "cases": case_rows,
        "qblade_runtime_contract": runtime_contract,
        "files_sha256": files_sha256,
        "generator": {
            "path": Path(__file__).name,
            "sha256": generator_identity["sha256"],
            "size_bytes": generator_identity["size_bytes"],
        },
    }
    manifest["fingerprint"] = _canonical_sha256(manifest)
    atomic_write_json(output / "campaign_manifest.json", manifest)
    validate_source_case(source_root, source_contract_path)
    validate_campaign(output)
    return manifest


def validate_campaign(campaign_root: Path) -> dict[str, object]:
    """Rehash every sealed campaign input and reject added, missing, or linked files."""

    root_candidate = Path(campaign_root).expanduser()
    if root_candidate.is_symlink():
        raise ValueError(f"Campaign root must not be a symlink: {root_candidate}")
    root = root_candidate.resolve()
    manifest, _identity = _read_locked_json(
        root / "campaign_manifest.json",
        label="H-VAWT modal-decay campaign manifest",
    )
    if manifest.get("schema") != CAMPAIGN_SCHEMA or manifest.get("sealed") is not True:
        raise ValueError("Campaign manifest is not a sealed modal-decay v2 manifest.")
    if manifest.get("campaign_root") != str(root):
        raise ValueError("Sealed modal-decay campaign may not be moved after generation.")
    raw_fingerprint = manifest.get("fingerprint")
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("fingerprint", None)
    if raw_fingerprint != _canonical_sha256(fingerprint_payload):
        raise ValueError("Campaign manifest fingerprint is inconsistent.")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict):
        raise ValueError("Campaign manifest files_sha256 must be an object.")
    observed = _tree_files(root)
    observed.pop("campaign_manifest.json", None)
    if set(observed) != set(expected):
        raise ValueError(
            "Sealed campaign file set changed: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"unexpected={sorted(set(observed) - set(expected))}."
        )
    mismatches = [
        relative
        for relative, path in observed.items()
        if _sha256(path) != str(expected[relative])
    ]
    if mismatches:
        raise ValueError(
            "Sealed campaign input SHA-256 mismatch for: "
            + ", ".join(sorted(mismatches))
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != FULL_CASE_COUNT:
        raise ValueError("Sealed campaign must contain exactly 12 case records.")
    expected_cases = {case.case_id: case.as_dict() for case in campaign_cases()}
    observed_cases: dict[str, dict[str, object]] = {}
    for row in cases:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError("Sealed campaign contains a malformed case record.")
        case_id = str(row["case_id"])
        if case_id in observed_cases:
            raise ValueError(f"Sealed campaign repeats case {case_id!r}.")
        observed_cases[case_id] = row
    if set(observed_cases) != set(expected_cases):
        raise ValueError("Sealed campaign case IDs do not match the locked 12-case matrix.")
    for case_id, expected_case in expected_cases.items():
        if any(observed_cases[case_id].get(key) != value for key, value in expected_case.items()):
            raise ValueError(f"Sealed campaign case contract changed for {case_id}.")
    return manifest


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _case_map(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    rows = manifest.get("cases")
    if not isinstance(rows, list):
        raise ValueError("Campaign manifest has no case list.")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise ValueError("Campaign manifest contains a malformed case record.")
        result[str(row["case_id"])] = row
    return result


def run_campaign_case(
    *,
    campaign_root: Path,
    case_id: str,
    qblade_root: Path,
    results_root: Path,
    threads: int = 1,
    checkpoint_every: int = 1000,
    resume_from_checkpoint: bool = True,
) -> QBladeRunResult:
    """Run one locked discriminator case through the explicit modal coupling path."""

    campaign = Path(campaign_root).expanduser().resolve()
    results = Path(results_root).expanduser().resolve(strict=False)
    if _is_within(results, campaign) or _is_within(campaign, results):
        raise ValueError("Mutable result output must be outside the sealed campaign tree.")
    manifest = validate_campaign(campaign)
    expected_runtime = manifest.get("qblade_runtime_contract")
    observed_runtime = _qblade_runtime_contract(Path(qblade_root))
    if expected_runtime != observed_runtime:
        raise ValueError(
            "Live QBlade library/dependency bytes do not match the campaign runtime contract."
        )
    cases = _case_map(manifest)
    if case_id not in cases:
        raise KeyError(f"Unknown modal-decay campaign case {case_id!r}.")
    case = cases[case_id]
    case_root = campaign / "cases" / case_id
    output = results / case_id
    return run_qblade_case(
        QBladeRunRequest(
            sim_path=case_root / str(case["sim_relative_path"]),
            qblade_root=Path(qblade_root),
            output_dir=output,
            result_name=str(case["expected_result_name"]),
            steps=int(case["total_wrapper_updates"]),
            threads=int(threads),
            result_filter=case_root / str(case["result_filter_relative_path"]),
            checkpoint_every=int(checkpoint_every),
            resume_from_checkpoint=bool(resume_from_checkpoint),
            allow_partial=False,
            sil_modal_burst_plan=case_root / str(case["plan_relative_path"]),
            coupling_mode="sil-modal-burst",
            case_metadata_path=case_root / "case.json",
            message_callback=print,
        )
    )


def _load_mode_shape_vector(
    path: Path,
    *,
    rpm: float,
    target_mode: int,
    external_config: Mapping[str, object],
) -> np.ndarray:
    rows = external_config.get("external_output_rows")
    if not isinstance(rows, list) or len(rows) != 117:
        raise ValueError("Mode projection requires the exact 117-action layout.")
    ordered_actions = sorted(rows, key=lambda row: int(row["index"]))
    expected_sites: list[str] = []
    for offset in range(0, 117, 3):
        triplet = ordered_actions[offset : offset + 3]
        expected_sites.append(
            canonical_node_id(str(triplet[0]["id"]), float(triplet[0]["position"]))
        )
    selected: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if (
                math.isclose(float(row["rpm"]), float(rpm), rel_tol=0.0, abs_tol=1.0e-12)
                and int(row["tracked_mode"]) == int(target_mode)
            ):
                node = str(row["action_node_id"])
                if node in selected:
                    raise ValueError(f"Mode-shape artifact repeats action site {node}.")
                selected[node] = row
    if set(selected) != set(expected_sites) or len(expected_sites) != 39:
        raise ValueError("Mode-shape artifact does not bind the exact 39 action sites.")
    values: list[complex] = []
    for node in expected_sites:
        row = selected[node]
        for axis in ("x", "y", "z"):
            values.append(
                complex(
                    float(row[f"u{axis}_per_m_real"]),
                    float(row[f"u{axis}_per_m_imag"]),
                )
            )
    shape = np.asarray(values, dtype=np.complex128)
    basis = np.column_stack((shape.real, -shape.imag))
    if np.linalg.matrix_rank(basis) != 2 or np.linalg.cond(basis) > 1.0e4:
        raise ValueError("Selected complex mode shape is not a usable two-quadrature basis.")
    return shape


def project_modal_coordinates(
    *,
    global_position_residual_m: np.ndarray,
    global_velocity_residual_m_per_s: np.ndarray,
    azimuth_deg: np.ndarray,
    mode_shape: np.ndarray,
    rotor_speed_rpm: float = ROTOR_SPEED_RPM,
    rotation_sign: int = ROTATION_SIGN,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotate paired residuals to the body frame and project both q and qdot."""

    positions = np.asarray(global_position_residual_m, dtype=float)
    velocities = np.asarray(global_velocity_residual_m_per_s, dtype=float)
    azimuth = np.asarray(azimuth_deg, dtype=float)
    shape = np.asarray(mode_shape, dtype=np.complex128).reshape(-1)
    if (
        positions.ndim != 2
        or positions.shape != velocities.shape
        or positions.shape[1] != 117
        or positions.shape[0] != azimuth.size
        or shape.size != 117
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(velocities))
        or not np.all(np.isfinite(azimuth))
    ):
        raise ValueError("Modal projection arrays must be finite N-by-117 paired residuals.")
    angles = float(rotation_sign) * np.deg2rad(azimuth - AZIMUTH_REFERENCE_DEG)
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    pos = positions.reshape((-1, 39, 3))
    vel = velocities.reshape((-1, 39, 3))
    body_pos = np.empty_like(pos)
    body_vel = np.empty_like(vel)
    body_pos[:, :, 0] = cosine * pos[:, :, 0] + sine * pos[:, :, 1]
    body_pos[:, :, 1] = -sine * pos[:, :, 0] + cosine * pos[:, :, 1]
    body_pos[:, :, 2] = pos[:, :, 2]
    body_vel[:, :, 0] = cosine * vel[:, :, 0] + sine * vel[:, :, 1]
    body_vel[:, :, 1] = -sine * vel[:, :, 0] + cosine * vel[:, :, 1]
    body_vel[:, :, 2] = vel[:, :, 2]
    angular_speed = float(rotation_sign) * float(rotor_speed_rpm) * 2.0 * math.pi / 60.0
    # R^T v_global = v_body + Omega x x_body.
    body_vel[:, :, 0] += angular_speed * body_pos[:, :, 1]
    body_vel[:, :, 1] -= angular_speed * body_pos[:, :, 0]
    basis = np.column_stack((shape.real, -shape.imag))
    condition = float(np.linalg.cond(basis))
    pseudoinverse = np.linalg.pinv(basis, rcond=1.0e-12)
    q_coefficients = body_pos.reshape((-1, 117)) @ pseudoinverse.T
    qdot_coefficients = body_vel.reshape((-1, 117)) @ pseudoinverse.T
    q = q_coefficients[:, 0] + 1j * q_coefficients[:, 1]
    qdot = qdot_coefficients[:, 0] + 1j * qdot_coefficients[:, 1]
    return q, qdot, condition


def fit_modal_decay(
    time_s: Sequence[float],
    modal_coordinate_m: Sequence[complex],
    *,
    target_frequency_hz: float,
    burst_cycles: int = BURST_CYCLES,
) -> ModalDecayFit:
    """Fit the free complex-mode envelope and phase after the Hann burst."""

    time = np.asarray(time_s, dtype=float)
    coordinate = np.asarray(modal_coordinate_m, dtype=np.complex128)
    target = float(target_frequency_hz)
    if (
        time.ndim != 1
        or coordinate.shape != time.shape
        or time.size < 20
        or target <= 0.0
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(coordinate.real))
        or not np.all(np.isfinite(coordinate.imag))
        or np.any(np.diff(time) <= 0.0)
    ):
        raise ValueError("Decay fit requires finite, increasing, paired complex samples.")
    burst_end = float(burst_cycles) / target
    available_free_cycles = max(0.0, (float(time[-1]) - burst_end) * target)
    if available_free_cycles + 1.0e-9 < MINIMUM_FREE_CYCLES:
        raise ValueError(
            f"Decay record provides {available_free_cycles:.6g} free cycles; "
            f"at least {MINIMUM_FREE_CYCLES:g} are required."
        )
    amplitude = np.abs(coordinate)
    peak = float(np.max(amplitude))
    burst_index = int(np.argmin(np.abs(time - burst_end)))
    burst_amplitude = float(amplitude[burst_index])
    fit_start = burst_end + 0.25 / target
    mask = time >= fit_start
    fit_time = time[mask]
    fit_coordinate = coordinate[mask]
    fit_amplitude = np.abs(fit_coordinate)
    positive_floor = max(1.0e-12, peak * 1.0e-8)
    retained = fit_amplitude > positive_floor
    fit_time = fit_time[retained]
    fit_coordinate = fit_coordinate[retained]
    fit_amplitude = fit_amplitude[retained]
    if fit_time.size < 20:
        return ModalDecayFit(
            classification="unresolved_amplitude",
            frequency_hz=0.0,
            damping_ratio=float("nan"),
            normalized_rmse=float("inf"),
            decay_rates_per_s=(),
            peak_translation_m=peak,
            burst_end_translation_m=burst_amplitude,
            available_free_cycles=available_free_cycles,
            sample_count=int(fit_time.size),
        )
    shifted = fit_time - fit_time[0]
    log_amplitude = np.log(fit_amplitude)
    linear = np.polyfit(shifted, log_amplitude, 1)

    def envelope_residual(parameters: np.ndarray) -> np.ndarray:
        return parameters[0] - parameters[1] * shifted - log_amplitude

    optimized = least_squares(
        envelope_residual,
        np.array([float(linear[1]), float(-linear[0])]),
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=2000,
    )
    log_a0, decay_rate = (float(value) for value in optimized.x)
    predicted = np.exp(log_a0 - decay_rate * shifted)
    normalized_rmse = float(
        np.sqrt(np.mean((fit_amplitude - predicted) ** 2))
        / max(float(np.max(fit_amplitude)), 1.0e-15)
    )
    phase = np.unwrap(np.angle(fit_coordinate))
    phase_slope, _phase_offset = np.polyfit(shifted, phase, 1)
    frequency = abs(float(phase_slope)) / (2.0 * math.pi)
    damped_omega = 2.0 * math.pi * frequency
    natural_omega = math.hypot(damped_omega, decay_rate)
    damping_ratio = float(decay_rate / natural_omega) if natural_omega > 0.0 else float("nan")
    window_rates: list[float] = []
    for indices in np.array_split(np.arange(fit_time.size), 4):
        if indices.size < 2:
            continue
        slope, _ = np.polyfit(shifted[indices], log_amplitude[indices], 1)
        window_rates.append(float(-slope))
    if peak < 1.0e-8 or burst_amplitude < 1.0e-8:
        classification = "unresolved_amplitude"
    elif frequency < 0.5 * target or frequency > 1.5 * target:
        classification = "nonoscillatory_or_off_target"
    elif decay_rate < -0.02 * 2.0 * math.pi * target:
        classification = "growth"
    elif not math.isfinite(damping_ratio) or damping_ratio >= 0.8:
        classification = "overdamped_or_unresolved"
    elif normalized_rmse > 0.25:
        classification = "nonexponential"
    else:
        classification = "interpretable_underdamped"
    return ModalDecayFit(
        classification=classification,
        frequency_hz=frequency,
        damping_ratio=damping_ratio,
        normalized_rmse=normalized_rmse,
        decay_rates_per_s=tuple(window_rates),
        peak_translation_m=peak,
        burst_end_translation_m=burst_amplitude,
        available_free_cycles=available_free_cycles,
        sample_count=int(fit_time.size),
    )


def classify_rayleigh_discriminator(
    fits: Mapping[str, ModalDecayFit],
) -> dict[str, object]:
    """Interpret the six paired decay fits without silently tuning any input."""

    if set(fits) != set(VARIANTS):
        raise ValueError("Rayleigh discriminator requires one fit for every locked variant.")
    zeta = {variant: float(fits[variant].damping_ratio) for variant in VARIANTS}
    finite = all(math.isfinite(value) for value in zeta.values())
    interpretable = all(
        fit.classification == "interpretable_underdamped" for fit in fits.values()
    )
    parser_spread = max(zeta["O"], zeta["S"], zeta["C20"]) - min(
        zeta["O"], zeta["S"], zeta["C20"]
    ) if finite else float("nan")
    parser_tolerance = max(0.002, 0.15 * parser_spread) if finite else float("nan")
    if not finite:
        parser_semantics = "unresolved_nonfinite_fit"
    else:
        original_close = abs(zeta["O"] - zeta["C20"]) <= parser_tolerance
        swapped_close = abs(zeta["S"] - zeta["C20"]) <= parser_tolerance
        if original_close and not swapped_close:
            parser_semantics = "header_21st_token_is_damping"
        elif swapped_close and not original_close:
            parser_semantics = "fixed_20th_numeric_token_is_damping"
        elif original_close and swapped_close:
            parser_semantics = "tail_semantics_not_observable"
        else:
            parser_semantics = "tail_semantics_noncanonical_or_ambiguous"
    residual = zeta["all_beta_zero"] if finite else float("nan")
    scalar_removal_effect = (
        zeta["C20"] - zeta["C20_scalar_zero"] if finite else float("nan")
    )
    station_removal_effect = (
        zeta["C20"] - zeta["C20_station_zero"] if finite else float("nan")
    )
    attribution_tolerance = (
        max(0.002, 0.1 * abs(zeta["C20"] - residual)) if finite else float("nan")
    )
    if not finite:
        damping_source = "unresolved_nonfinite_fit"
    else:
        scalar_active = scalar_removal_effect > attribution_tolerance
        station_active = station_removal_effect > attribution_tolerance
        if min(scalar_removal_effect, station_removal_effect) < -attribution_tolerance:
            damping_source = "nonphysical_negative_removal_effect"
        elif scalar_active and station_active:
            damping_source = "both_file_scalar_and_station_vector"
        elif scalar_active:
            damping_source = "file_scalar_dominant"
        elif station_active:
            damping_source = "station_vector_dominant"
        else:
            damping_source = "rayleigh_beta_effect_not_resolved"
    complete = bool(finite and interpretable)
    selection_allowed = complete and parser_semantics not in {
        "tail_semantics_noncanonical_or_ambiguous",
        "unresolved_nonfinite_fit",
    } and damping_source in {
        "both_file_scalar_and_station_vector",
        "file_scalar_dominant",
        "station_vector_dominant",
    }
    return {
        "schema": "vorlap-hvawt-rayleigh-discriminator-classification-v2",
        "complete_and_interpretable": complete,
        "selection_allowed": selection_allowed,
        "selected_unambiguous_substructure_encoding": "C20" if complete else None,
        "parser_semantics": parser_semantics,
        "parser_absolute_damping_tolerance": parser_tolerance,
        "damping_source": damping_source,
        "attribution_absolute_damping_tolerance": attribution_tolerance,
        "effective_c20_damping_ratio": zeta["C20"],
        "all_beta_zero_residual_damping_ratio": residual,
        "file_scalar_removal_effect_damping_ratio": scalar_removal_effect,
        "station_vector_removal_effect_damping_ratio": station_removal_effect,
        "substructure_contribution_separately_identified": False,
        "scalar_station_interaction_separately_identified": False,
        "source_decision_uses_direct_removal_effects": True,
        "physical_rayleigh_increment_over_all_beta_zero": (
            zeta["C20"] - residual if finite else float("nan")
        ),
        "variant_damping_ratios": zeta,
    }


def _normalized_result_columns(path: Path) -> tuple[dict[str, list[float]], dict[str, str]]:
    columns = read_qblade_result_numeric_columns(path)
    if not columns:
        raise ValueError(f"QBlade result has no readable numeric columns: {path}")
    normalized: dict[str, str] = {}
    for original in columns:
        key = normalize_qblade_header(original)
        if key in normalized:
            raise ValueError(f"QBlade result repeats normalized column {key!r}: {path}")
        normalized[key] = original
    return columns, normalized


def _column_array(
    columns: Mapping[str, Sequence[float]],
    normalized: Mapping[str, str],
    expected: str,
) -> np.ndarray:
    source = normalized.get(normalize_qblade_header(expected))
    if source is None:
        raise ValueError(f"QBlade result is missing required channel {expected!r}.")
    values = np.asarray(columns[source], dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"QBlade result channel {expected!r} is not a finite vector.")
    return values


def _validate_completed_case_summary(
    summary: Mapping[str, object],
    *,
    case: Mapping[str, object],
    result_path: Path,
    runtime_dependency_fingerprint: str,
) -> dict[str, object]:
    required_equal = {
        "coupling_mode": "sil-modal-burst",
        "requested_steps": int(case["total_wrapper_updates"]),
        "completed_steps": int(case["total_wrapper_updates"]),
        "failed": False,
        "stopped": False,
        "allow_partial": False,
        "exported_results": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": summary.get(key)}
        for key, expected in required_equal.items()
        if summary.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"Case {case['case_id']} is not a completed strict modal run: "
            + json.dumps(mismatches, sort_keys=True)
        )
    modal = summary.get("sil_modal_burst")
    if not isinstance(modal, dict) or modal.get("enabled") is not True:
        raise ValueError(f"Case {case['case_id']} lacks typed modal-burst summary evidence.")
    final = modal.get("final_summary")
    if (
        modal.get("finalized") is not True
        or not isinstance(final, dict)
        or final.get("ended_with_explicit_zero") is not True
        or int(final.get("wrapper_updates", -1)) != int(case["total_wrapper_updates"])
    ):
        raise ValueError(f"Case {case['case_id']} did not finalize the exact force schedule.")
    native = summary.get("native_callback_evidence")
    if (
        not isinstance(native, dict)
        or int(native.get("update_init_count", 0) or 0) != 0
        or int(native.get("update_count", 0) or 0) != 0
        or native.get("log_file_exists") is True
    ):
        raise ValueError(f"Case {case['case_id']} has native callback contamination.")
    reproducibility = summary.get("reproducibility")
    if (
        not isinstance(reproducibility, dict)
        or reproducibility.get("qblade_runtime_dependency_fingerprint")
        != runtime_dependency_fingerprint
    ):
        raise ValueError(f"Case {case['case_id']} runtime dependencies are not campaign-bound.")
    result_identity = summary.get("result_file_identity")
    observed_identity = file_sha256(result_path)
    if (
        not isinstance(result_identity, dict)
        or observed_identity is None
        or result_identity.get("sha256") != observed_identity.get("sha256")
        or result_identity.get("size_bytes") != observed_identity.get("size_bytes")
    ):
        raise ValueError(f"Case {case['case_id']} result bytes do not match its run summary.")
    return {
        "summary_sha256": _canonical_sha256(summary),
        "result_sha256": observed_identity["sha256"],
        "result_size_bytes": observed_identity["size_bytes"],
        "zero_native_callbacks": True,
        "modal_schedule_finalized": True,
    }


def _load_completed_case_series(
    *,
    results_root: Path,
    case: Mapping[str, object],
    expected_channels: Sequence[str],
    runtime_dependency_fingerprint: str,
) -> dict[str, object]:
    case_results = results_root / str(case["case_id"])
    result_path = case_results / str(case["expected_result_file_name"])
    summary_path = case_results / str(case["expected_summary_name"])
    summary, summary_identity = _read_locked_json(
        summary_path,
        label=f"modal-decay run summary for {case['case_id']}",
    )
    evidence = _validate_completed_case_summary(
        summary,
        case=case,
        result_path=result_path,
        runtime_dependency_fingerprint=runtime_dependency_fingerprint,
    )
    columns, normalized = _normalized_result_columns(result_path)
    missing = [
        name
        for name in expected_channels
        if normalize_qblade_header(name) not in normalized
    ]
    if missing:
        raise ValueError(
            f"Case {case['case_id']} result is missing {len(missing)} locked channels."
        )
    arrays = [_column_array(columns, normalized, name) for name in expected_channels]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1 or next(iter(lengths)) < 20:
        raise ValueError(f"Case {case['case_id']} result columns are ragged or too short.")
    time = arrays[0]
    timestep_index = arrays[1]
    azimuth = arrays[2]
    rpm = arrays[3]
    if np.any(np.diff(time) <= 0.0):
        raise ValueError(f"Case {case['case_id']} time is not strictly increasing.")
    observed_dt = np.diff(time)
    expected_dt = float(case["timestep_s"])
    if not math.isclose(
        float(np.median(observed_dt)), expected_dt, rel_tol=2.0e-4, abs_tol=1.0e-9
    ):
        raise ValueError(f"Case {case['case_id']} exported timestep differs from its plan.")
    if float(time[-1]) + expected_dt < float(case["end_time_s"]):
        raise ValueError(f"Case {case['case_id']} result ends before the free-cycle gate.")
    if abs(float(np.median(rpm)) - float(case["rotor_speed_rpm"])) > 0.01:
        raise ValueError(f"Case {case['case_id']} exported RPM differs from the prescribed RPM.")
    return {
        "time": time,
        "timestep_index": timestep_index,
        "azimuth": azimuth,
        "rpm": rpm,
        "positions": np.column_stack(arrays[4:121]),
        "velocities": np.column_stack(arrays[121:238]),
        "summary_identity": summary_identity,
        "evidence": evidence,
    }


def _finite_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def analyze_campaign(
    *,
    campaign_root: Path,
    results_root: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Analyze all six excited/baseline pairs and classify Rayleigh semantics."""

    campaign = Path(campaign_root).expanduser().resolve()
    results = Path(results_root).expanduser().resolve()
    if _is_within(results, campaign) or _is_within(campaign, results):
        raise ValueError("Mutable result output must be outside the sealed campaign tree.")
    manifest = validate_campaign(campaign)
    case_by_id = _case_map(manifest)
    runtime = manifest.get("qblade_runtime_contract")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("dependency_fingerprint"), str):
        raise ValueError("Campaign runtime dependency contract is malformed.")
    dependency_fingerprint = str(runtime["dependency_fingerprint"])
    fits: dict[str, ModalDecayFit] = {}
    pair_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        baseline_case = case_by_id[_case_id(variant, "baseline")]
        excited_case = case_by_id[_case_id(variant, "excited")]
        case_root = campaign / "cases" / str(excited_case["case_id"])
        external_path = (
            case_root
            / str(manifest["source_audit"]["legacy_external_config_relative_path"])
        )
        external = json.loads(external_path.read_text(encoding="utf-8"))
        expected_channels = _result_channels(external)
        baseline = _load_completed_case_series(
            results_root=results,
            case=baseline_case,
            expected_channels=expected_channels,
            runtime_dependency_fingerprint=dependency_fingerprint,
        )
        excited = _load_completed_case_series(
            results_root=results,
            case=excited_case,
            expected_channels=expected_channels,
            runtime_dependency_fingerprint=dependency_fingerprint,
        )
        excited_time = np.asarray(excited["time"], dtype=float)
        baseline_time = np.asarray(baseline["time"], dtype=float)
        if excited_time.shape != baseline_time.shape or not np.allclose(
            excited_time,
            baseline_time,
            rtol=0.0,
            atol=max(1.0e-9, TIMESTEP_S * 2.0e-4),
        ):
            raise ValueError(f"Variant {variant} excited/baseline timestamps do not pair.")
        excited_azimuth = np.asarray(excited["azimuth"], dtype=float)
        baseline_azimuth = np.asarray(baseline["azimuth"], dtype=float)
        if not np.allclose(excited_azimuth, baseline_azimuth, rtol=0.0, atol=0.02):
            raise ValueError(f"Variant {variant} excited/baseline azimuths do not pair.")
        position_residual = np.asarray(excited["positions"]) - np.asarray(
            baseline["positions"]
        )
        velocity_residual = np.asarray(excited["velocities"]) - np.asarray(
            baseline["velocities"]
        )
        shape = _load_mode_shape_vector(
            campaign / "inputs" / str(manifest["modal_artifacts"]["action_shapes"]["path"]),
            rpm=float(excited_case["rotor_speed_rpm"]),
            target_mode=int(excited_case["target_mode"]),
            external_config=external,
        )
        coordinate, coordinate_rate, basis_condition = project_modal_coordinates(
            global_position_residual_m=position_residual,
            global_velocity_residual_m_per_s=velocity_residual,
            azimuth_deg=excited_azimuth,
            mode_shape=shape,
            rotor_speed_rpm=float(excited_case["rotor_speed_rpm"]),
        )
        fit = fit_modal_decay(
            excited_time,
            coordinate,
            target_frequency_hz=float(excited_case["frequency_hz"]),
        )
        fits[variant] = fit
        numerical_rate = np.gradient(coordinate, excited_time)
        free_mask = excited_time >= BURST_CYCLES / float(excited_case["frequency_hz"])
        denominator = max(float(np.linalg.norm(coordinate_rate[free_mask])), 1.0e-15)
        rate_error = float(
            np.linalg.norm(numerical_rate[free_mask] - coordinate_rate[free_mask])
            / denominator
        )
        pair_rows.append(
            {
                "variant": variant,
                "baseline_case_id": baseline_case["case_id"],
                "excited_case_id": excited_case["case_id"],
                "fit": fit.as_dict(),
                "mode_basis_condition_number": basis_condition,
                "position_velocity_projection_relative_rate_error": rate_error,
                "realized_peak_translation_relative_error_from_nominal": abs(
                    fit.peak_translation_m - NOMINAL_AMPLITUDE_M
                )
                / NOMINAL_AMPLITUDE_M,
                "baseline_evidence": baseline["evidence"],
                "excited_evidence": excited["evidence"],
            }
        )
    classification = classify_rayleigh_discriminator(fits)
    status = (
        "complete_interpretable"
        if classification["complete_and_interpretable"]
        else "complete_but_not_interpretable"
    )
    analysis: dict[str, object] = {
        "schema": ANALYSIS_SCHEMA,
        "status": status,
        "campaign_fingerprint": manifest["fingerprint"],
        "campaign_root": str(campaign),
        "results_root": str(results),
        "case_count": FULL_CASE_COUNT,
        "pair_count": len(VARIANTS),
        "no_inflow_cases": True,
        "force_scale_was_not_tuned": True,
        "force_scale_equation": "Fhat = 2*k_q_1rpm*q_nominal/(pi*5)",
        "pairs": pair_rows,
        "classification": classification,
        "claim_boundary": (
            "This classifies effective QBlade structural decay and its Rayleigh-beta "
            "increment relative to the all-beta-zero control; it is not an inflow or "
            "H-VAWT lock-in validation."
        ),
    }
    serializable = _finite_json(analysis)
    assert isinstance(serializable, dict)
    destination = (
        results / "rayleigh_discriminator_analysis.json"
        if output_path is None
        else Path(output_path).expanduser().resolve(strict=False)
    )
    if _is_within(destination, campaign):
        raise ValueError("Analysis output must be outside the sealed campaign tree.")
    atomic_write_json(destination, serializable)
    csv_path = destination.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "variant",
                "classification",
                "frequency_hz",
                "damping_ratio",
                "normalized_rmse",
                "peak_translation_m",
                "burst_end_translation_m",
                "available_free_cycles",
                "realized_peak_translation_relative_error_from_nominal",
            ),
        )
        writer.writeheader()
        for row in pair_rows:
            fit_row = dict(row["fit"])
            writer.writerow(
                {
                    "variant": row["variant"],
                    **{key: fit_row[key] for key in writer.fieldnames if key in fit_row},
                    "realized_peak_translation_relative_error_from_nominal": row[
                        "realized_peak_translation_relative_error_from_nominal"
                    ],
                }
            )
    return serializable


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate and seal the 12 cases")
    generate.add_argument("--source-root", type=Path, required=True)
    generate.add_argument("--source-contract", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--action-shapes", type=Path, required=True)
    generate.add_argument("--force-duals", type=Path, required=True)
    generate.add_argument("--qblade-root", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="rehash a sealed campaign")
    validate.add_argument("--campaign-root", type=Path, required=True)
    run = subparsers.add_parser("run-case", help="run one fixed campaign case")
    run.add_argument("--campaign-root", type=Path, required=True)
    run.add_argument("--case-id", required=True)
    run.add_argument("--qblade-root", type=Path, required=True)
    run.add_argument("--results-root", type=Path, required=True)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--checkpoint-every", type=int, default=1000)
    run.add_argument("--no-resume", action="store_true")
    analyze = subparsers.add_parser("analyze", help="analyze all six paired decays")
    analyze.add_argument("--campaign-root", type=Path, required=True)
    analyze.add_argument("--results-root", type=Path, required=True)
    analyze.add_argument("--output", type=Path)
    listing = subparsers.add_parser("list-cases", help="print the exact 12 case IDs")
    listing.add_argument("--campaign-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.command == "generate":
        payload = generate_campaign(
            source_root=args.source_root,
            source_contract_path=args.source_contract,
            output_root=args.output_root,
            action_shapes_path=args.action_shapes,
            force_duals_path=args.force_duals,
            qblade_root=args.qblade_root,
        )
        print(json.dumps({"fingerprint": payload["fingerprint"], "cases": FULL_CASE_COUNT}))
    elif args.command == "validate":
        payload = validate_campaign(args.campaign_root)
        print(json.dumps({"fingerprint": payload["fingerprint"], "cases": FULL_CASE_COUNT}))
    elif args.command == "run-case":
        result = run_campaign_case(
            campaign_root=args.campaign_root,
            case_id=args.case_id,
            qblade_root=args.qblade_root,
            results_root=args.results_root,
            threads=args.threads,
            checkpoint_every=args.checkpoint_every,
            resume_from_checkpoint=not args.no_resume,
        )
        print(json.dumps({"summary": str(result.summary_path), "failed": result.summary["failed"]}))
    elif args.command == "analyze":
        payload = analyze_campaign(
            campaign_root=args.campaign_root,
            results_root=args.results_root,
            output_path=args.output,
        )
        print(json.dumps({"status": payload["status"], "pairs": len(VARIANTS)}))
    elif args.command == "list-cases":
        if args.campaign_root is not None:
            validate_campaign(args.campaign_root)
        for case in campaign_cases():
            print(case.case_id)
    else:  # pragma: no cover - argparse enforces a known command
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    sys.exit(main())
