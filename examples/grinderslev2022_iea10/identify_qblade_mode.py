#!/usr/bin/env python3
"""Identify the QBlade first-edgewise coordinate from frozen structural probes.

The mode shape comes from the pre-existing 12 s axis-corrected-v2 coherent
structural-forcing probe.  The natural frequency comes independently from the
matching unloaded QBlade initialization decay.  Neither source uses the
Grinderslev aerodynamic-power curve or the response-aware 120 s result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXAMPLE_ROOT = Path(__file__).resolve().parent
BASE_CASE_ROOT = EXAMPLE_ROOT / "generated_case"
CASE_NAME = "grinderslev2022_p100_i50_iea10_single_blade"
BASE_CONFIG = (
    BASE_CASE_ROOT / CASE_NAME / "Control" / "vorlap_qblade_external.json"
)
GEOMETRY = BASE_CASE_ROOT / "iea10_blade_geometry.csv"
STRUCTURE = BASE_CASE_ROOT / CASE_NAME / "Structure" / "iea10_blade_hawc2.str"
DEFAULT_FORCED_HISTORY = (
    REPO_ROOT
    / "run_headless"
    / "runs"
    / "grinderslev2022_axis_corrected_v2_p100_i50_edge_12s"
    / "grinderslev2022_p100_i50_iea10_single_blade_results_sil_vorlap_history.csv"
)
DEFAULT_UNLOADED_RESULT = (
    REPO_ROOT
    / "run_headless"
    / "runs"
    / "grinderslev2022_axis_corrected_v2_p100_i50_unloaded_12s"
    / "grinderslev2022_p100_i50_iea10_single_blade_results.txt"
)
DEFAULT_OUTPUT_CSV = EXAMPLE_ROOT / "reference_data" / "qblade_first_edge_mode.csv"
DEFAULT_OUTPUT_JSON = (
    EXAMPLE_ROOT / "reference_data" / "qblade_first_edge_mode_identification.json"
)

EXPECTED_FORCED_HISTORY_SHA256 = (
    "c783084145188a480ab99a1c626737b648ebef785af280d28c4673fd0aeb5997"
)
EXPECTED_UNLOADED_RESULT_SHA256 = (
    "6237bd70415c6b6165576a4f1fae01e33f9f0084d7a806716cb74e10d3518f15"
)
EXPECTED_GEOMETRY_SHA256 = (
    "06d72b8f80e290ce0bac88011ad2a9698f5b42e1ae71da2c7f2b97f613c2c1b7"
)
EXPECTED_STRUCTURE_SHA256 = (
    "8455754fe52cfbb925c6a6432c48de7fb98ed1d29f09c5af899d19d0075a500c"
)
TARGET_FREQUENCY_HZ = 0.67
TIMESTEP_S = 0.006
EXPECTED_ROWS = 2000
MODE_WINDOW_STARTS_S = (0.0, 2.0, 4.0, 6.0)
FREQUENCY_WINDOWS_S = ((0.0, 12.0), (1.0, 12.0), (2.0, 12.0), (0.5, 10.0), (2.0, 10.0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 changed: expected {expected}, observed {observed}."
        )


def _node_token(node_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in node_id).strip("_")


def _load_public_quadrature() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    from examples.grinderslev2022_iea10 import response_aware_benchmark as benchmark

    geometry = benchmark._read_geometry(GEOMETRY)
    structural_fraction, structural_mass, rayleigh_beta_s = (
        benchmark._read_qblade_structure(STRUCTURE)
    )
    span_fraction = geometry["span_fraction"]
    tributary = benchmark._tributary_lengths(geometry["span_m"])
    mass_density = np.interp(span_fraction, structural_fraction, structural_mass)
    mass_weights = mass_density * tributary
    return (
        span_fraction,
        geometry["edge_mode_tip_normalized"],
        mass_weights,
        rayleigh_beta_s,
    )


def _load_node_ids() -> tuple[str, ...]:
    value = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    node_ids = tuple(str(item) for item in value.get("node_ids", ()))
    if len(node_ids) != 30 or len(set(node_ids)) != len(node_ids):
        raise ValueError("Pinned QBlade source configuration must contain 30 unique nodes.")
    return node_ids


def _read_forced_history(
    path: Path,
    node_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = tuple(_node_token(node_id) for node_id in node_ids)
    velocity_fields = [
        f"vel_{token}_{axis}_mps" for token in tokens for axis in "xyz"
    ]
    force_fields = [f"force_{token}_{axis}_n" for token in tokens for axis in "xyz"]
    time: list[float] = []
    velocity: list[list[float]] = []
    force: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step_index", "time_s", *velocity_fields, *force_fields}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Frozen structural history lacks required time/force/velocity fields.")
        for expected_step, row in enumerate(reader):
            if int(row["step_index"]) != expected_step:
                raise ValueError("Frozen structural-history step indices are not contiguous.")
            time.append(float(row["time_s"]))
            velocity.append([float(row[name]) for name in velocity_fields])
            force.append([float(row[name]) for name in force_fields])
    time_array = np.asarray(time, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float).reshape(-1, len(node_ids), 3)
    force_array = np.asarray(force, dtype=float).reshape(-1, len(node_ids), 3)
    if (
        time_array.size != EXPECTED_ROWS
        or np.any(~np.isfinite(time_array))
        or np.any(~np.isfinite(velocity_array))
        or np.any(~np.isfinite(force_array))
        or not np.allclose(np.diff(time_array), TIMESTEP_S, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("Frozen structural history has invalid size, time grid, or values.")
    return time_array, velocity_array, force_array


def _harmonic_shape(
    time_s: np.ndarray,
    velocity_mps: np.ndarray,
    force_n: np.ndarray,
    *,
    start_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    tip_norm = np.linalg.norm(force_n[:, -1, :], axis=1)
    direction_row = int(np.argmax(tip_norm))
    directions = np.zeros((force_n.shape[1], 3), dtype=float)
    for node in range(force_n.shape[1]):
        norm = float(np.linalg.norm(force_n[direction_row, node, :]))
        if norm > 0.0:
            directions[node, :] = force_n[direction_row, node, :] / norm
    selected = time_s >= float(start_s)
    time = time_s[selected]
    centered = time - float(np.mean(time))
    design = np.column_stack(
        (
            np.ones(time.size),
            centered,
            np.sin(2.0 * math.pi * TARGET_FREQUENCY_HZ * time),
            np.cos(2.0 * math.pi * TARGET_FREQUENCY_HZ * time),
        )
    )
    phasor = np.zeros(force_n.shape[1], dtype=complex)
    for node in range(force_n.shape[1]):
        if not np.any(directions[node, :]):
            continue
        directional_velocity = velocity_mps[selected, node, :] @ directions[node, :]
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, directional_velocity, rcond=None
        )
        if rank != 4 or np.any(~np.isfinite(coefficients)):
            raise ValueError("Structural harmonic fit is rank deficient or non-finite.")
        phasor[node] = complex(coefficients[2], coefficients[3])
    if abs(phasor[-1]) <= 0.0:
        raise ValueError("Structural harmonic fit has zero tip amplitude.")
    ratio = phasor / phasor[-1]
    mode = np.real(ratio)
    mode[0] = 0.0
    mode[-1] = 1.0
    phase = np.angle(ratio)
    if np.any(mode < 0.0) or np.any(np.diff(mode) < -1.0e-3):
        raise ValueError("Identified first-edgewise shape is not a positive unit-tip mode.")
    return mode, phase, float(time_s[direction_row])


def _read_unloaded_tip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    time: list[float] = []
    tip: list[float] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        next(handle)
        next(handle)
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Time~[s]", "Y_b~Tip~Trl.Def.~BLD_1~[m]"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Unloaded QBlade result lacks time or edgewise-tip deflection.")
        for row in reader:
            time.append(float(row["Time~[s]"]))
            tip.append(float(row["Y_b~Tip~Trl.Def.~BLD_1~[m]"]))
    time_array = np.asarray(time, dtype=float)
    tip_array = np.asarray(tip, dtype=float)
    if (
        time_array.size != EXPECTED_ROWS
        or np.any(~np.isfinite(time_array))
        or np.any(~np.isfinite(tip_array))
        or not np.allclose(np.diff(time_array), TIMESTEP_S, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("Unloaded QBlade result has invalid size, time grid, or values.")
    return time_array, tip_array


def _best_harmonic_frequency(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    start_s: float,
    end_s: float,
) -> tuple[float, float]:
    selected = (time_s >= start_s) & (time_s <= end_s)
    time = time_s[selected]
    values = values[selected]
    centered = time - float(np.mean(time))
    best_frequency = None
    best_sum_square = math.inf
    for frequency in np.linspace(0.64, 0.70, 6001):
        design = np.column_stack(
            (
                np.ones(time.size),
                centered,
                np.sin(2.0 * math.pi * frequency * time),
                np.cos(2.0 * math.pi * frequency * time),
            )
        )
        coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
        if rank != 4:
            raise ValueError("Unloaded frequency fit is rank deficient.")
        residual = values - design @ coefficients
        sum_square = float(np.dot(residual, residual))
        if sum_square < best_sum_square:
            best_sum_square = sum_square
            best_frequency = float(frequency)
    if best_frequency is None or not math.isfinite(best_sum_square):
        raise ValueError("Unloaded frequency identification failed.")
    return best_frequency, best_sum_square


def identify(
    forced_history: Path,
    unloaded_result: Path,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    _require_hash(forced_history, EXPECTED_FORCED_HISTORY_SHA256, "forced history")
    _require_hash(unloaded_result, EXPECTED_UNLOADED_RESULT_SHA256, "unloaded result")
    _require_hash(GEOMETRY, EXPECTED_GEOMETRY_SHA256, "public geometry")
    _require_hash(STRUCTURE, EXPECTED_STRUCTURE_SHA256, "QBlade structure")
    node_ids = _load_node_ids()
    span, public_mode, mass_weights, rayleigh_beta_s = _load_public_quadrature()
    time, velocity, force = _read_forced_history(forced_history, node_ids)
    unloaded_time, unloaded_tip = _read_unloaded_tip(unloaded_result)

    mode_rows: list[dict[str, Any]] = []
    primary_mode = None
    primary_phase = None
    direction_time = None
    public_mass = float(np.sum(mass_weights * public_mode**2))
    for start in MODE_WINDOW_STARTS_S:
        mode, phase, direction_time_s = _harmonic_shape(
            time, velocity, force, start_s=start
        )
        modal_mass = float(np.sum(mass_weights * mode**2))
        overlap = float(np.sum(mass_weights * public_mode * mode) / public_mass)
        mode_rows.append(
            {
                "start_s": start,
                "end_s": float(time[-1]),
                "modal_mass_kg": modal_mass,
                "public_projection_overlap": overlap,
                "max_abs_phase_offset_rad": float(np.max(np.abs(phase[1:]))),
            }
        )
        if start == MODE_WINDOW_STARTS_S[0]:
            primary_mode = mode
            primary_phase = phase
            direction_time = direction_time_s
    assert primary_mode is not None and primary_phase is not None and direction_time is not None

    frequency_rows = []
    for start, end in FREQUENCY_WINDOWS_S:
        frequency, residual = _best_harmonic_frequency(
            unloaded_time, unloaded_tip, start_s=start, end_s=end
        )
        frequency_rows.append(
            {
                "start_s": start,
                "end_s": end,
                "frequency_hz": frequency,
                "residual_sum_squares_m2": residual,
            }
        )
    natural_frequency_hz = float(frequency_rows[0]["frequency_hz"])
    damping_ratio = rayleigh_beta_s * math.pi * natural_frequency_hz
    modal_mass = float(mode_rows[0]["modal_mass_kg"])
    structural_power_coefficient = (
        damping_ratio
        * modal_mass
        * (2.0 * math.pi * natural_frequency_hz) ** 3
    )
    rows = [
        {
            "span_fraction": float(span[index]),
            "public_openfast_mode_tip_normalized": float(public_mode[index]),
            "qblade_mode_tip_normalized": float(primary_mode[index]),
            "qblade_phase_offset_rad": float(primary_phase[index]),
            "modal_mass_weight_kg": float(mass_weights[index]),
        }
        for index in range(span.size)
    ]
    metadata: dict[str, Any] = {
        "schema": "grinderslev2022-qblade-structural-coordinate-v1",
        "scope": (
            "QBlade first-edgewise structural-coordinate identification only; "
            "independent of the Grinderslev aerodynamic-power curve and 120 s response result"
        ),
        "forced_mode_method": (
            "project each stored nodal velocity onto the local chord direction recovered "
            "from the coherent frozen force; fit constant, linear trend, and 0.67 Hz "
            "sine/cosine; phase-align to the tip and take the real unit-tip ratio"
        ),
        "frequency_method": (
            "minimum-residual constant/trend/sine/cosine scan from 0.64 to 0.70 Hz "
            "in 1e-5 Hz increments on the independent unloaded tip decay"
        ),
        "direction_basis_sample_time_s": direction_time,
        "source_sha256": {
            "forced_sil_history": EXPECTED_FORCED_HISTORY_SHA256,
            "unloaded_qblade_result": EXPECTED_UNLOADED_RESULT_SHA256,
            "public_geometry": EXPECTED_GEOMETRY_SHA256,
            "qblade_structure": EXPECTED_STRUCTURE_SHA256,
        },
        "source_paths": {
            "forced_sil_history": str(forced_history.relative_to(REPO_ROOT)),
            "unloaded_qblade_result": str(unloaded_result.relative_to(REPO_ROOT)),
            "public_geometry": str(GEOMETRY.relative_to(REPO_ROOT)),
            "qblade_structure": str(STRUCTURE.relative_to(REPO_ROOT)),
        },
        "mode_window_sensitivity": mode_rows,
        "frequency_window_sensitivity": frequency_rows,
        "selected_window_s": [MODE_WINDOW_STARTS_S[0], float(time[-1])],
        "natural_frequency_hz": natural_frequency_hz,
        "rayleigh_stiffness_coefficient_s": rayleigh_beta_s,
        "modal_damping_ratio": damping_ratio,
        "public_openfast_modal_mass_kg": public_mass,
        "qblade_modal_mass_kg": modal_mass,
        "qblade_to_public_modal_mass_ratio": modal_mass / public_mass,
        "public_projection_overlap": float(
            np.sum(mass_weights * public_mode * primary_mode) / public_mass
        ),
        "structural_power_relation": "P_struct(A_tip) = zeta M_QBlade omega_QBlade^3 A_tip^2",
        "structural_power_coefficient_wpm2": structural_power_coefficient,
    }
    return rows, metadata


def _write_outputs(
    rows: Sequence[dict[str, float]],
    metadata: dict[str, Any],
    output_csv: Path,
    output_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0])
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: f"{float(row[name]):.17g}" for name in fieldnames})
    metadata = dict(metadata)
    metadata["mode_csv_sha256"] = sha256_file(output_csv)
    output_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forced-history", type=Path, default=DEFAULT_FORCED_HISTORY)
    parser.add_argument("--unloaded-result", type=Path, default=DEFAULT_UNLOADED_RESULT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows, metadata = identify(
        args.forced_history.resolve(), args.unloaded_result.resolve()
    )
    _write_outputs(rows, metadata, args.output_csv.resolve(), args.output_json.resolve())
    print(args.output_csv.resolve())
    print(args.output_json.resolve())


if __name__ == "__main__":
    main()
