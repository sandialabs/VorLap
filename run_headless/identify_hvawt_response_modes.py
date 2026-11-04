#!/usr/bin/env python3
"""Identify H-VAWT response modes at the 39 QBlade external-action locations.

The OWENS/GXBeam runner saves the complete mixed-state right eigenvector.  This
postprocessor interpolates its point states in component fraction, fixes the
complex phase deterministically, normalizes the maximum three-dimensional
translation amplitude over the QBlade action set to one metre, and converts the
runner's physical energy projections into generalized modal properties.

No QBlade simulation is launched by this script.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
from typing import Iterable, Sequence

import numpy as np


STATE_NAMES = (
    "ux",
    "uy",
    "uz",
    "theta_x",
    "theta_y",
    "theta_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "omega_x",
    "omega_y",
    "omega_z",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_BASIS_MODES = tuple(range(1, 13))
DAMPING_CANDIDATE_IDS = (
    "qblade_2_0_9_7_documented",
    "reference_header_intended",
    "qblade_2_0_4_archived_scalar",
    "legacy_converter_header_scalar",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        parts = resolved.parts
        for marker in ("run_headless", "examples", "tests"):
            if marker in parts:
                return Path(*parts[parts.index(marker) :]).as_posix()
    return resolved.name


def _artifact_identity(path: Path, label: str | None = None) -> dict[str, object]:
    return {
        "label": label or _portable_path_label(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _portable_manifest_label(label: str) -> str:
    path = Path(label)
    return _portable_path_label(path) if path.is_absolute() else path.as_posix()


def _modal_bundle_label(conversion: dict[str, object]) -> str:
    for source_label in conversion.get("inputs", {}):
        parts = Path(_portable_manifest_label(str(source_label))).parts
        for index in range(len(parts) - 2):
            if parts[index : index + 2] == ("run_headless", "runs"):
                return Path(*parts[index : index + 3], "owens_modal").as_posix()
    return "external_owens_modal_bundle"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _finite(value: str | float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, received {value!r}")
    return result


def _parse_list(value: str, cast: type[int] | type[float]) -> list[int] | list[float]:
    result = [cast(token.strip()) for token in value.split(",") if token.strip()]
    if not result:
        raise ValueError("A requested numeric list is empty")
    return result


def _complex_state(row: dict[str, str]) -> dict[str, complex]:
    return {
        name: complex(
            _finite(row[f"{name}_real"], f"{name}_real"),
            _finite(row[f"{name}_imag"], f"{name}_imag"),
        )
        for name in STATE_NAMES
    }


def _interpolate_state(
    lower: dict[str, complex], upper: dict[str, complex], weight_upper: float
) -> dict[str, complex]:
    return {
        name: (1.0 - weight_upper) * lower[name] + weight_upper * upper[name]
        for name in STATE_NAMES
    }


def _norm3(state: dict[str, complex], names: Sequence[str] = ("ux", "uy", "uz")) -> float:
    return math.sqrt(sum(abs(state[name]) ** 2 for name in names))


def _phase_residual_mod_pi(value: complex) -> float:
    """Phase distance from a real scalar, treating opposite signs as equivalent."""

    phase = math.atan2(value.imag, value.real)
    return 0.5 * math.atan2(math.sin(2.0 * phase), math.cos(2.0 * phase))


def _component_nodes(metadata_rows: Iterable[dict[str, str]]) -> dict[str, list[tuple[float, int]]]:
    result: dict[str, list[tuple[float, int]]] = {}
    blade = re.compile(r"^blade_(\d+)_([-+0-9.eE]+)$")
    strut = re.compile(r"^strut_(\d+)_blade_(\d+)_([-+0-9.eE]+)$")
    for row in metadata_rows:
        label = row["label"]
        match = blade.match(label)
        if match:
            key = f"BLD_{int(match.group(1))}"
            fraction = float(match.group(2))
        else:
            match = strut.match(label)
            if not match:
                continue
            key = f"STR_{int(match.group(1))}_{int(match.group(2))}"
            fraction = float(match.group(3))
        result.setdefault(key, []).append((fraction, int(row["node_id"])))
    for key, values in result.items():
        values.sort()
        fractions = [fraction for fraction, _ in values]
        if len(values) < 2 or any(b <= a for a, b in zip(fractions, fractions[1:])):
            raise ValueError(f"Converted component {key} lacks a strictly increasing interpolation mesh")
    return result


def _action_sites(config: dict[str, object]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], dict[str, object]] = {}
    for row_object in config.get("external_output_rows", []):
        row = dict(row_object)
        if row.get("action") != "ADDFORCE":
            continue
        if bool(row.get("local")):
            raise ValueError("The H-VAWT modal mapping requires global external forces")
        component = str(row["id"])
        fraction = _finite(row["position"], f"{component} action fraction")
        direction = str(row["direction"])
        if direction not in {"X", "Y", "Z"}:
            raise ValueError(f"Unsupported action direction {direction!r}")
        key = (component, fraction)
        site = grouped.setdefault(
            key,
            {
                "action_node_id": f"{component}_{fraction:.6f}",
                "component": component,
                "fraction": fraction,
                "output_indices": {},
            },
        )
        indices = site["output_indices"]
        if direction in indices:
            raise ValueError(f"Duplicate {direction} action at {component} fraction {fraction}")
        indices[direction] = int(row["index"])
    sites = sorted(grouped.values(), key=lambda item: (str(item["component"]), float(item["fraction"])))
    for site in sites:
        if set(site["output_indices"]) != {"X", "Y", "Z"}:
            raise ValueError(f"Action site {site['action_node_id']} does not have global X/Y/Z force rows")
    if len(sites) != 39:
        raise ValueError(f"Expected 39 H-VAWT external-action sites, found {len(sites)}")
    output_indices = [
        int(site["output_indices"][direction])
        for site in sites
        for direction in ("X", "Y", "Z")
    ]
    if len(set(output_indices)) != 117:
        raise ValueError("The 117 global action rows do not have unique output indices")
    if sorted(output_indices) != list(range(120, 237)):
        raise ValueError(
            "The H-VAWT force-dual contract requires contiguous QBlade output indices 120:236"
        )
    return sites


def _bracket(
    nodes: Sequence[tuple[float, int]], target: float, *, tolerance: float = 1.0e-10
) -> dict[str, float | int]:
    fractions = [fraction for fraction, _ in nodes]
    if target < fractions[0] - tolerance or target > fractions[-1] + tolerance:
        raise ValueError(
            f"Action fraction {target} lies outside converted component range "
            f"[{fractions[0]}, {fractions[-1]}]"
        )
    nearest = min(range(len(nodes)), key=lambda index: abs(fractions[index] - target))
    if abs(fractions[nearest] - target) <= tolerance:
        lower = upper = nearest
        weight = 0.0
    else:
        upper = bisect.bisect_right(fractions, target)
        lower = upper - 1
        weight = (target - fractions[lower]) / (fractions[upper] - fractions[lower])
    return {
        "lower_fraction": fractions[lower],
        "upper_fraction": fractions[upper],
        "lower_node_id": nodes[lower][1],
        "upper_node_id": nodes[upper][1],
        "weight_upper": weight,
        "nearest_fraction_error": abs(fractions[nearest] - target),
        "bracket_span": fractions[upper] - fractions[lower],
    }


def _frequency_table(path: Path) -> dict[float, list[float]]:
    rows = _rows(path)
    result: dict[float, list[float]] = {}
    for row in rows:
        rpm = _finite(row["rpm"], "Campbell RPM")
        columns = sorted(name for name in row if name.startswith("mode_") and name.endswith("_hz"))
        result[rpm] = [_finite(row[name], name) for name in columns]
    return result


def _nearest_rpm(values: Iterable[float], target: float) -> float:
    values = list(values)
    selected = min(values, key=lambda value: abs(value - target))
    if abs(selected - target) > 1.0e-9:
        raise ValueError(f"Requested RPM {target} is absent; nearest saved value is {selected}")
    return selected


def _normalize_action_mode(
    raw_states: Sequence[dict[str, complex]],
) -> dict[str, object]:
    anchor_options = [
        (abs(state[name]), site_index, name, state[name])
        for site_index, state in enumerate(raw_states)
        for name in ("ux", "uy", "uz")
    ]
    _, anchor_site_index, anchor_name, anchor = max(
        anchor_options, key=lambda item: item[0]
    )
    if abs(anchor) <= 0.0:
        raise ValueError("Mode has zero translation at every external-action location")
    phase = anchor.conjugate() / abs(anchor)
    q_raw = max(_norm3(state) for state in raw_states)
    if q_raw <= 0.0:
        raise ValueError("Mode has a nonpositive action-set normalization")
    scale = phase / q_raw
    states = [
        {name: value * scale for name, value in state.items()} for state in raw_states
    ]
    translations = [
        state[name] for state in states for name in ("ux", "uy", "uz")
    ]
    real_norm = math.sqrt(sum(value.real**2 for value in translations))
    imaginary_norm = math.sqrt(sum(value.imag**2 for value in translations))
    complex_norm = math.sqrt(sum(abs(value) ** 2 for value in translations))
    maximum_component = max(abs(value) for value in translations)
    nonzero = [
        value for value in translations if abs(value) >= 1.0e-12 * maximum_component
    ]
    residuals = [_phase_residual_mod_pi(value) for value in nonzero]
    weight_sum = sum(abs(value) ** 2 for value in nonzero)
    weighted_phase_rms = math.sqrt(
        sum(
            abs(value) ** 2 * residual**2
            for value, residual in zip(nonzero, residuals)
        )
        / weight_sum
    )
    material = [
        value for value in translations if abs(value) >= 0.05 * maximum_component
    ]
    material_residuals = [abs(_phase_residual_mod_pi(value)) for value in material]
    quadrature_to_in_phase = imaginary_norm / real_norm
    gate = (
        quadrature_to_in_phase <= 0.02
        and weighted_phase_rms <= 0.02
        and max(material_residuals, default=0.0) <= 0.05
    )
    return {
        "states": states,
        "translations": translations,
        "anchor_site_index": anchor_site_index,
        "anchor_name": anchor_name,
        "phase": phase,
        "q_raw": q_raw,
        "purity": {
            "component_count": len(translations),
            "imaginary_l2_norm_over_real_l2_norm": quadrature_to_in_phase,
            "real_part_relative_l2_approximation_error": imaginary_norm / complex_norm,
            "phase_residual_definition": "absolute phase distance modulo pi from the phase-anchored real axis",
            "maximum_phase_residual_rad_all_nonzero_components": max(
                abs(value) for value in residuals
            ),
            "amplitude_weighted_rms_phase_residual_rad": weighted_phase_rms,
            "material_component_amplitude_threshold_fraction_of_max": 0.05,
            "maximum_phase_residual_rad_material_components": max(
                material_residuals, default=0.0
            ),
            "phase_outlier_threshold_rad": 0.05,
            "phase_outlier_translation_energy_fraction": sum(
                abs(value) ** 2
                for value in nonzero
                if abs(_phase_residual_mod_pi(value)) > 0.05
            )
            / complex_norm**2,
            "single_real_scalar_gate": {
                "maximum_imaginary_over_real_l2": 0.02,
                "maximum_weighted_phase_rms_rad": 0.02,
                "maximum_material_component_phase_residual_rad": 0.05,
                "passed": gate,
            },
            "response_model_requirement": (
                "single real scalar using the real part of the phase-anchored action shape is defensible"
                if gate
                else "retain real/imaginary coordinate pair or a multimode representation"
            ),
        },
    }


def _minimum_norm_force_dual(
    phi: np.ndarray,
    *,
    target_modes: Sequence[int],
    basis_modes: Sequence[int],
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute F with Phi^H F=I by an explicit thin SVD."""

    if phi.shape != (117, len(basis_modes)):
        raise ValueError(f"Expected a 117x{len(basis_modes)} action basis, found {phi.shape}")
    u_matrix, singular_values, vh_matrix = np.linalg.svd(phi, full_matrices=False)
    tolerance = max(phi.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float("inf") if rank < len(basis_modes) else float(singular_values[0] / singular_values[-1])
    )
    if rank < len(basis_modes):
        dual = np.zeros_like(phi)
        residual = np.full((len(basis_modes), len(basis_modes)), np.nan + 0j)
    else:
        dual = (u_matrix / singular_values) @ vh_matrix
        residual = phi.conj().T @ dual - np.eye(len(basis_modes))
    residual_inf = (
        float(np.max(np.abs(residual)))
        if rank == len(basis_modes)
        else float("inf")
    )
    target_indices = [basis_modes.index(mode) for mode in target_modes]
    off_target = float("inf") if rank < len(basis_modes) else 0.0
    if rank == len(basis_modes):
        for target_index in target_indices:
            column = residual[:, target_index] + np.eye(len(basis_modes))[:, target_index]
            off_target = max(
                off_target,
                max(
                    (
                        abs(value)
                        for index, value in enumerate(column)
                        if index != target_index
                    ),
                    default=0.0,
                ),
            )
    gates = {
        "basis_mode_order": list(basis_modes),
        "action_row_order": "39 sites in qblade_component/fraction order, global X then Y then Z",
        "matrix_shape": list(phi.shape),
        "singular_values": [float(value) for value in singular_values],
        "rank_tolerance": float(tolerance),
        "rank": rank,
        "condition_number": condition,
        "maximum_biorthogonality_residual": residual_inf,
        "maximum_target_off_mode_generalized_force": float(off_target),
        "acceptance_gates": {
            "required_rank": len(basis_modes),
            "maximum_condition_number": 1.0e8,
            "maximum_biorthogonality_residual": 1.0e-8,
            "maximum_off_target_generalized_force": 0.01,
        },
    }
    gates["passed"] = bool(
        rank == len(basis_modes)
        and condition <= 1.0e8
        and residual_inf <= 1.0e-8
        and off_target <= 0.01
    )
    return dual, gates


def _group_complex_action_artifact(
    path: Path, *, force_dual: bool
) -> tuple[dict[tuple[float, int], np.ndarray], dict[tuple[float, int], tuple[tuple[object, ...], ...]]]:
    vectors: dict[tuple[float, int], list[complex]] = {}
    layouts: dict[tuple[float, int], list[tuple[object, ...]]] = {}
    mode_column = "target_mode" if force_dual else "tracked_mode"
    for row in _rows(path):
        key = (float(row["rpm"]), int(row[mode_column]))
        layout = (
            row["action_node_id"],
            row["qblade_component"],
            float(row["qblade_fraction"]),
            int(row["qblade_force_output_index_x"]),
            int(row["qblade_force_output_index_y"]),
            int(row["qblade_force_output_index_z"]),
        )
        layouts.setdefault(key, []).append(layout)
        if force_dual:
            names = ("x", "y", "z")
            vectors.setdefault(key, []).extend(
                complex(
                    _finite(
                        row[f"force_{name}_n_per_generalized_force_real"],
                        f"force {name} real",
                    ),
                    _finite(
                        row[f"force_{name}_n_per_generalized_force_imag"],
                        f"force {name} imag",
                    ),
                )
                for name in names
            )
        else:
            names = ("ux", "uy", "uz")
            vectors.setdefault(key, []).extend(
                complex(
                    _finite(row[f"{name}_per_m_real"], f"{name} real"),
                    _finite(row[f"{name}_per_m_imag"], f"{name} imag"),
                )
                for name in names
            )
    expected_modes = (6, 12) if force_dual else ACTION_BASIS_MODES
    expected_keys = {(rpm, mode) for rpm in (0.0, 1.0) for mode in expected_modes}
    if set(vectors) != expected_keys:
        raise ValueError(
            f"{path} has action keys {sorted(vectors)}, expected {sorted(expected_keys)}"
        )
    for key in expected_keys:
        if len(vectors[key]) != 117 or len(layouts[key]) != 39:
            raise ValueError(f"{path} key {key} does not contain 39 XYZ action sites")
        indices = [index for item in layouts[key] for index in item[3:6]]
        if sorted(indices) != list(range(120, 237)):
            raise ValueError(f"{path} key {key} does not retain force rows 120:236")
    return (
        {key: np.asarray(values, dtype=complex) for key, values in vectors.items()},
        {key: tuple(values) for key, values in layouts.items()},
    )


def _phase_aligned_vector_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if first.shape != second.shape:
        raise ValueError(f"Vector shapes differ: {first.shape} versus {second.shape}")
    inner = np.vdot(second, first)
    if abs(inner) == 0.0:
        raise ValueError("Independent evidence vectors are orthogonal; phase alignment is undefined")
    phase = inner / abs(inner)
    mac = abs(np.vdot(first, second)) ** 2 / (
        float(np.vdot(first, first).real) * float(np.vdot(second, second).real)
    )
    return {
        "mac": float(mac),
        "phase_alignment_real": float(phase.real),
        "phase_alignment_imag": float(phase.imag),
        "phase_aligned_relative_l2_difference": float(
            np.linalg.norm(first - phase * second) / np.linalg.norm(first)
        ),
        "direct_relative_l2_difference": float(
            np.linalg.norm(first - second) / np.linalg.norm(first)
        ),
    }


def compare_independent_action_evidence(
    original_shapes: Path,
    original_force_duals: Path,
    repeat_shapes: Path,
    repeat_force_duals: Path,
) -> dict[str, object]:
    """Recompute target-mode and cross-basis invariance from tracked CSV bytes."""

    shape_a, shape_layout_a = _group_complex_action_artifact(
        original_shapes, force_dual=False
    )
    shape_b, shape_layout_b = _group_complex_action_artifact(
        repeat_shapes, force_dual=False
    )
    force_a, force_layout_a = _group_complex_action_artifact(
        original_force_duals, force_dual=True
    )
    force_b, force_layout_b = _group_complex_action_artifact(
        repeat_force_duals, force_dual=True
    )
    cases: list[dict[str, object]] = []
    condition_numbers: list[dict[str, float]] = []
    for rpm in (0.0, 1.0):
        phi_a = np.column_stack([shape_a[(rpm, mode)] for mode in ACTION_BASIS_MODES])
        phi_b = np.column_stack([shape_b[(rpm, mode)] for mode in ACTION_BASIS_MODES])
        condition_numbers.append(
            {
                "rpm": rpm,
                "original_condition_number": float(np.linalg.cond(phi_a)),
                "repeat_condition_number": float(np.linalg.cond(phi_b)),
            }
        )
        for mode in (6, 12):
            if shape_layout_a[(rpm, mode)] != shape_layout_b[(rpm, mode)]:
                raise ValueError(f"Original/repeat shape layouts differ at RPM {rpm}, mode {mode}")
            if force_layout_a[(rpm, mode)] != force_layout_b[(rpm, mode)]:
                raise ValueError(f"Original/repeat force layouts differ at RPM {rpm}, mode {mode}")
            target_index = mode - 1
            original_on_repeat = phi_b.conj().T @ force_a[(rpm, mode)]
            repeat_on_original = phi_a.conj().T @ force_b[(rpm, mode)]
            cases.append(
                {
                    "rpm": rpm,
                    "target_mode": mode,
                    "action_shape": _phase_aligned_vector_metrics(
                        shape_a[(rpm, mode)], shape_b[(rpm, mode)]
                    ),
                    "force_dual": _phase_aligned_vector_metrics(
                        force_a[(rpm, mode)], force_b[(rpm, mode)]
                    ),
                    "original_force_on_repeat_basis_target_error": float(
                        abs(original_on_repeat[target_index] - 1.0)
                    ),
                    "original_force_on_repeat_basis_maximum_off_target": float(
                        max(
                            abs(value)
                            for index, value in enumerate(original_on_repeat)
                            if index != target_index
                        )
                    ),
                    "repeat_force_on_original_basis_target_error": float(
                        abs(repeat_on_original[target_index] - 1.0)
                    ),
                    "repeat_force_on_original_basis_maximum_off_target": float(
                        max(
                            abs(value)
                            for index, value in enumerate(repeat_on_original)
                            if index != target_index
                        )
                    ),
                }
            )
    summary = {
        "minimum_selected_action_shape_mac": min(
            case["action_shape"]["mac"] for case in cases
        ),
        "maximum_phase_aligned_selected_action_shape_relative_difference": max(
            case["action_shape"]["phase_aligned_relative_l2_difference"]
            for case in cases
        ),
        "minimum_selected_force_dual_mac": min(
            case["force_dual"]["mac"] for case in cases
        ),
        "maximum_phase_aligned_selected_force_dual_relative_difference": max(
            case["force_dual"]["phase_aligned_relative_l2_difference"]
            for case in cases
        ),
        "maximum_cross_basis_target_generalized_force_error": max(
            max(
                case["original_force_on_repeat_basis_target_error"],
                case["repeat_force_on_original_basis_target_error"],
            )
            for case in cases
        ),
        "maximum_cross_basis_off_target_generalized_force": max(
            max(
                case["original_force_on_repeat_basis_maximum_off_target"],
                case["repeat_force_on_original_basis_maximum_off_target"],
            )
            for case in cases
        ),
    }
    gates = {
        "minimum_selected_action_shape_mac": 0.999999,
        "maximum_phase_aligned_selected_action_shape_relative_difference": 1.0e-5,
        "minimum_selected_force_dual_mac": 0.999999,
        "maximum_phase_aligned_selected_force_dual_relative_difference": 1.0e-5,
        "maximum_cross_basis_target_generalized_force_error": 1.0e-5,
        "maximum_cross_basis_off_target_generalized_force": 1.0e-4,
    }
    passed = bool(
        summary["minimum_selected_action_shape_mac"] >= gates["minimum_selected_action_shape_mac"]
        and summary["maximum_phase_aligned_selected_action_shape_relative_difference"]
        <= gates["maximum_phase_aligned_selected_action_shape_relative_difference"]
        and summary["minimum_selected_force_dual_mac"] >= gates["minimum_selected_force_dual_mac"]
        and summary["maximum_phase_aligned_selected_force_dual_relative_difference"]
        <= gates["maximum_phase_aligned_selected_force_dual_relative_difference"]
        and summary["maximum_cross_basis_target_generalized_force_error"]
        <= gates["maximum_cross_basis_target_generalized_force_error"]
        and summary["maximum_cross_basis_off_target_generalized_force"]
        <= gates["maximum_cross_basis_off_target_generalized_force"]
    )
    return {
        "schema": "vorlap-hvawt-modal-independent-repeatability-v1",
        "condition_numbers": condition_numbers,
        "cases": cases,
        "summary": summary,
        "acceptance_gates": gates,
        "passed": passed,
    }


def identify(
    modal_dir: Path,
    external_config: Path,
    output_json: Path,
    output_csv: Path,
    output_duals_csv: Path | None = None,
    *,
    rpm: float,
    comparison_rpm: float,
    modes: Sequence[int],
    reviewed_frequencies: Path | None = None,
    tracked_evidence_dir: Path | None = None,
) -> dict[str, object]:
    modal_dir = modal_dir.resolve()
    if output_duals_csv is None:
        output_duals_csv = output_csv.with_name(
            f"{output_csv.stem}_force_duals{output_csv.suffix}"
        )
    input_names = (
        "conversion_manifest.json",
        "nodes.csv",
        "node_metadata.csv",
        "elements.csv",
        "element_damping_candidates.csv",
        "point_masses.csv",
        "campbell_frequencies.csv",
        "campbell_mode_shapes.csv",
        "campbell_mode_properties.csv",
        "campbell_mode_energy_by_component.csv",
        "campbell_mode_energy_by_element.csv",
        "campbell_mode_orthogonality.csv",
        "campbell_mode_tracking.csv",
        "owens_run_metadata.txt",
        "owens_request.json",
    )
    input_paths = {name: modal_dir / name for name in input_names}
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required modal artifacts are missing: " + ", ".join(missing))

    conversion = json.loads(input_paths["conversion_manifest.json"].read_text(encoding="utf-8"))
    conversion_inputs = conversion.get("inputs", {})
    conversion_sizes = conversion.get("input_sizes_bytes", {})
    if not conversion_inputs:
        raise ValueError("Conversion manifest has no source-input hashes")
    for label, digest in conversion_inputs.items():
        if Path(str(label)).is_absolute():
            raise ValueError(f"Conversion source label is not portable: {label}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError(f"Conversion source hash is invalid: {label}")
        if int(conversion_sizes.get(label, 0)) <= 0:
            raise ValueError(f"Conversion source size is missing or nonpositive: {label}")
    component_id_by_name = {
        str(name): int(component_id)
        for name, component_id in conversion.get("mesh", {}).get("component_ids", {}).items()
    }
    if not component_id_by_name:
        raise ValueError("Conversion manifest does not define mesh.component_ids")
    component_name_by_id = {
        component_id: name for name, component_id in component_id_by_name.items()
    }
    config = json.loads(external_config.read_text(encoding="utf-8"))
    sites = _action_sites(config)
    component_nodes = _component_nodes(_rows(input_paths["node_metadata.csv"]))
    node_rows = {int(row["node_id"]): row for row in _rows(input_paths["nodes.csv"])}

    mappings: list[dict[str, object]] = []
    for site in sites:
        component = str(site["component"])
        if component not in component_nodes:
            raise ValueError(f"No converted GXBeam component mesh matches QBlade action ID {component}")
        bracket = _bracket(component_nodes[component], float(site["fraction"]))
        lower_xyz = [
            _finite(node_rows[int(bracket["lower_node_id"])][column], column)
            for column in ("x_m", "y_m", "z_m")
        ]
        upper_xyz = [
            _finite(node_rows[int(bracket["upper_node_id"])][column], column)
            for column in ("x_m", "y_m", "z_m")
        ]
        weight = float(bracket["weight_upper"])
        mappings.append(
            {
                **site,
                **bracket,
                "xyz_m": [
                    (1.0 - weight) * lower_xyz[index] + weight * upper_xyz[index]
                    for index in range(3)
                ],
            }
        )

    shape_rows = _rows(input_paths["campbell_mode_shapes.csv"])
    shapes: dict[tuple[float, int, int], dict[str, complex]] = {}
    shape_eigen_indices: dict[tuple[float, int], int] = {}
    for row in shape_rows:
        key = (float(row["rpm"]), int(row["tracked_mode"]), int(row["node_id"]))
        if key in shapes:
            raise ValueError(f"Duplicate saved mode-shape row {key}")
        shapes[key] = _complex_state(row)
        mode_key = (key[0], key[1])
        eigen_index = int(row["eigen_index"])
        previous_index = shape_eigen_indices.setdefault(mode_key, eigen_index)
        if previous_index != eigen_index:
            raise ValueError(f"Inconsistent eigen index in saved mode shape {mode_key}")

    property_rows = _rows(input_paths["campbell_mode_properties.csv"])
    properties = {
        (float(row["rpm"]), int(row["tracked_mode"])): row for row in property_rows
    }
    component_energy_rows = _rows(input_paths["campbell_mode_energy_by_component.csv"])
    component_energies: dict[tuple[float, int], list[dict[str, str]]] = {}
    for row in component_energy_rows:
        key = (float(row["rpm"]), int(row["tracked_mode"]))
        component_energies.setdefault(key, []).append(row)
    element_energy_rows = _rows(input_paths["campbell_mode_energy_by_element.csv"])
    element_energies: dict[tuple[float, int], list[dict[str, str]]] = {}
    for row in element_energy_rows:
        key = (float(row["rpm"]), int(row["tracked_mode"]))
        element_energies.setdefault(key, []).append(row)
    damping_candidate_rows = _rows(input_paths["element_damping_candidates.csv"])
    damping_candidates_by_element = {
        int(row["element_id"]): row for row in damping_candidate_rows
    }
    if len(damping_candidates_by_element) != len(damping_candidate_rows):
        raise ValueError("Duplicate element damping-candidate rows were found")
    elements_by_id = {
        int(row["element_id"]): row for row in _rows(input_paths["elements.csv"])
    }
    if set(elements_by_id) != set(damping_candidates_by_element):
        raise ValueError("Element damping-candidate coverage differs from elements.csv")
    for element_id, candidate_row in damping_candidates_by_element.items():
        element_row = elements_by_id[element_id]
        if (
            int(candidate_row["component_id"]) != int(element_row["component_id"])
            or int(candidate_row["kind_id"]) != int(element_row["kind_id"])
        ):
            raise ValueError(f"Element {element_id} component/kind mapping is inconsistent")
        for candidate_id in DAMPING_CANDIDATE_IDS:
            if f"{candidate_id}_beta_s" not in candidate_row:
                raise ValueError(f"Damping candidate column {candidate_id!r} is absent")
    conversion_damping = conversion.get("damping_semantics", {})
    if conversion_damping.get("selected_damping_candidate_id") is not None:
        raise ValueError("Damping candidates must remain unresolved before the paired decay gate")
    frequencies = _frequency_table(input_paths["campbell_frequencies.csv"])
    selected_rpm = _nearest_rpm(frequencies, rpm)
    zero_rpm = _nearest_rpm(frequencies, comparison_rpm)
    zero_rpm_clusters: dict[int, list[int]] = {}
    for row in _rows(input_paths["campbell_mode_tracking.csv"]):
        if abs(float(row["rpm"]) - zero_rpm) <= 1.0e-9:
            zero_rpm_clusters.setdefault(int(row["reference_cluster"]), []).append(
                int(row["tracked_mode"])
            )
    degenerate_zero_rpm_clusters = [
        sorted(cluster)
        for _, cluster in sorted(zero_rpm_clusters.items())
        if len(cluster) > 1
    ]
    selected_mode_set = set(modes)
    orthogonality_evidence = [
        {
            "mode_a": int(row["mode_a"]),
            "mode_b": int(row["mode_b"]),
            "system_left_mass_right": {
                "real": _finite(
                    row["system_left_mass_right_real"], "system left-M-right real"
                ),
                "imag": _finite(
                    row["system_left_mass_right_imag"], "system left-M-right imag"
                ),
            },
            "physical_velocity_mass_mac": _finite(
                row["physical_velocity_mass_mac"], "physical velocity-mass MAC"
            ),
        }
        for row in _rows(input_paths["campbell_mode_orthogonality.csv"])
        if abs(float(row["rpm"]) - selected_rpm) <= 1.0e-9
        and int(row["mode_a"]) in selected_mode_set
        and int(row["mode_b"]) in selected_mode_set
    ]
    if not orthogonality_evidence:
        raise ValueError("No selected-RPM physical orthogonality evidence was saved")

    reviewed_delta: dict[int, float] = {}
    if reviewed_frequencies is not None:
        reviewed = _frequency_table(reviewed_frequencies)
        reviewed_selected = _nearest_rpm(reviewed, rpm)
        for mode in modes:
            delta = frequencies[selected_rpm][mode - 1] - reviewed[reviewed_selected][mode - 1]
            if abs(delta) > 1.0e-9:
                raise ValueError(
                    f"Mode {mode} regenerated frequency differs from the reviewed basis by {delta:.12g} Hz"
                )
            reviewed_delta[mode] = delta

    def action_states_at(saved_rpm: float, mode: int) -> list[dict[str, complex]]:
        states: list[dict[str, complex]] = []
        for mapping in mappings:
            lower_key = (saved_rpm, mode, int(mapping["lower_node_id"]))
            upper_key = (saved_rpm, mode, int(mapping["upper_node_id"]))
            if lower_key not in shapes or upper_key not in shapes:
                raise ValueError(f"Saved eigenvector does not cover action mapping for mode {mode}: {mapping}")
            states.append(
                _interpolate_state(
                    shapes[lower_key], shapes[upper_key], float(mapping["weight_upper"])
                )
            )
        return states

    basis_modes = list(ACTION_BASIS_MODES)
    if len(frequencies[selected_rpm]) < max(basis_modes):
        raise ValueError("The saved Campbell basis does not contain modes 1-12")
    evidence_rpms = tuple(dict.fromkeys((zero_rpm, selected_rpm)))
    normalized_action_modes: dict[tuple[float, int], dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    for saved_rpm in evidence_rpms:
        for basis_mode in basis_modes:
            normalized = _normalize_action_mode(action_states_at(saved_rpm, basis_mode))
            normalized_action_modes[(saved_rpm, basis_mode)] = normalized
            frequency = frequencies[saved_rpm][basis_mode - 1]
            for mapping, state in zip(mappings, normalized["states"]):
                indices = mapping["output_indices"]
                row: dict[str, object] = {
                    "rpm": saved_rpm,
                    "tracked_mode": basis_mode,
                    "eigen_index": shape_eigen_indices[(saved_rpm, basis_mode)],
                    "frequency_hz": frequency,
                    "action_node_id": mapping["action_node_id"],
                    "qblade_component": mapping["component"],
                    "qblade_fraction": mapping["fraction"],
                    "qblade_force_output_index_x": indices["X"],
                    "qblade_force_output_index_y": indices["Y"],
                    "qblade_force_output_index_z": indices["Z"],
                    "lower_node_id": mapping["lower_node_id"],
                    "lower_fraction": mapping["lower_fraction"],
                    "upper_node_id": mapping["upper_node_id"],
                    "upper_fraction": mapping["upper_fraction"],
                    "weight_upper": mapping["weight_upper"],
                    "x_m": mapping["xyz_m"][0],
                    "y_m": mapping["xyz_m"][1],
                    "z_m": mapping["xyz_m"][2],
                }
                for name in STATE_NAMES:
                    row[f"{name}_per_m_real"] = state[name].real
                    row[f"{name}_per_m_imag"] = state[name].imag
                csv_rows.append(row)

    force_dual_rows: list[dict[str, object]] = []
    force_dual_evidence: list[dict[str, object]] = []
    for saved_rpm in evidence_rpms:
        phi = np.column_stack(
            [
                np.asarray(
                    normalized_action_modes[(saved_rpm, mode)]["translations"],
                    dtype=complex,
                )
                for mode in basis_modes
            ]
        )
        real_basis_gate = all(
            bool(
                normalized_action_modes[(saved_rpm, mode)]["purity"]
                ["single_real_scalar_gate"]["passed"]
            )
            for mode in basis_modes
        )
        dual, dual_gates = _minimum_norm_force_dual(
            phi, target_modes=modes, basis_modes=basis_modes
        )
        if not dual_gates["passed"]:
            raise ValueError(
                f"The 12-mode action force-dual failed its SVD gates at {saved_rpm} RPM: "
                f"rank={dual_gates['rank']}, condition={dual_gates['condition_number']}, "
                f"residual={dual_gates['maximum_biorthogonality_residual']}"
            )
        dual_gates.update(
            {
                "rpm": saved_rpm,
                "complex_harmonic_convention": "Phi^H F = I; real and imaginary force coefficients are the cosine and negative-sine harmonic quadratures",
                "whole_basis_real_reduction_gate_passed": real_basis_gate,
                "real_reduction_gate_by_mode": [
                    {
                        "tracked_mode": mode,
                        **normalized_action_modes[(saved_rpm, mode)]["purity"][
                            "single_real_scalar_gate"
                        ],
                    }
                    for mode in basis_modes
                ],
                "real_reduction_policy": "A purely real force dual may replace the complex harmonic dual only when every mode in the 12-mode basis passes the serialized action-shape real-scalar gate.",
            }
        )
        force_dual_evidence.append(dual_gates)
        for target_mode in modes:
            target_column = basis_modes.index(target_mode)
            force_vector = dual[:, target_column]
            for site_index, mapping in enumerate(mappings):
                values = force_vector[3 * site_index : 3 * site_index + 3]
                indices = mapping["output_indices"]
                force_dual_rows.append(
                    {
                        "rpm": saved_rpm,
                        "target_mode": target_mode,
                        "target_basis_column_zero_based": target_column,
                        "action_site_row_zero_based": site_index,
                        "action_node_id": mapping["action_node_id"],
                        "qblade_component": mapping["component"],
                        "qblade_fraction": mapping["fraction"],
                        "qblade_force_output_index_x": indices["X"],
                        "qblade_force_output_index_y": indices["Y"],
                        "qblade_force_output_index_z": indices["Z"],
                        "force_x_n_per_generalized_force_real": values[0].real,
                        "force_x_n_per_generalized_force_imag": values[0].imag,
                        "force_y_n_per_generalized_force_real": values[1].real,
                        "force_y_n_per_generalized_force_imag": values[1].imag,
                        "force_z_n_per_generalized_force_real": values[2].real,
                        "force_z_n_per_generalized_force_imag": values[2].imag,
                    }
                )

    mode_summaries: list[dict[str, object]] = []
    for mode in modes:
        if not 1 <= mode <= len(frequencies[selected_rpm]):
            raise ValueError(f"Requested mode {mode} is outside the saved Campbell basis")
        raw_states = action_states_at(selected_rpm, mode)
        anchor_options = [
            (abs(state[name]), site_index, name, state[name])
            for site_index, state in enumerate(raw_states)
            for name in ("ux", "uy", "uz")
        ]
        _, anchor_site_index, anchor_name, anchor = max(anchor_options, key=lambda item: item[0])
        if abs(anchor) <= 0.0:
            raise ValueError(f"Mode {mode} has zero translation at every external-action location")
        phase = anchor.conjugate() / abs(anchor)
        q_raw = max(_norm3(state) for state in raw_states)
        if q_raw <= 0.0:
            raise ValueError(f"Mode {mode} has a nonpositive action-set normalization")
        scale = phase / q_raw
        normalized_states = [
            {name: value * scale for name, value in state.items()} for state in raw_states
        ]
        action_translations = [
            state[name] for state in normalized_states for name in ("ux", "uy", "uz")
        ]
        real_norm = math.sqrt(sum(value.real**2 for value in action_translations))
        imaginary_norm = math.sqrt(sum(value.imag**2 for value in action_translations))
        complex_norm = math.sqrt(sum(abs(value) ** 2 for value in action_translations))
        maximum_component = max(abs(value) for value in action_translations)
        nonzero_components = [
            value for value in action_translations if abs(value) >= 1.0e-12 * maximum_component
        ]
        phase_residuals = [_phase_residual_mod_pi(value) for value in nonzero_components]
        phase_weight_sum = sum(abs(value) ** 2 for value in nonzero_components)
        weighted_phase_rms = math.sqrt(
            sum(
                abs(value) ** 2 * residual**2
                for value, residual in zip(nonzero_components, phase_residuals)
            )
            / phase_weight_sum
        )
        material_components = [
            value for value in action_translations if abs(value) >= 0.05 * maximum_component
        ]
        material_phase_residuals = [
            abs(_phase_residual_mod_pi(value)) for value in material_components
        ]
        phase_outlier_energy_fraction = (
            sum(
                abs(value) ** 2
                for value in nonzero_components
                if abs(_phase_residual_mod_pi(value)) > 0.05
            )
            / complex_norm**2
        )
        quadrature_to_in_phase = imaginary_norm / real_norm
        real_scalar_gate = (
            quadrature_to_in_phase <= 0.02
            and weighted_phase_rms <= 0.02
            and max(material_phase_residuals, default=0.0) <= 0.05
        )

        property_key = (selected_rpm, mode)
        if property_key not in properties:
            raise ValueError(f"Mode-property row {property_key} is absent")
        prop = properties[property_key]
        frequency = frequencies[selected_rpm][mode - 1]
        if abs(_finite(prop["frequency_hz"], "property frequency") - frequency) > 1.0e-12:
            raise ValueError(f"Frequency disagreement between modal artifacts for mode {mode}")
        omega = 2.0 * math.pi * frequency
        velocity_mass_raw = _finite(prop["velocity_mass_raw"], "velocity mass numerator")
        stiffness_raw = _finite(prop["elastic_stiffness_raw"], "elastic stiffness numerator")
        modal_mass = velocity_mass_raw / (omega**2 * q_raw**2)
        modal_stiffness = stiffness_raw / q_raw**2

        energy_key = (selected_rpm, mode)
        if energy_key not in element_energies:
            raise ValueError(f"Element-energy rows {energy_key} are absent")
        mode_element_rows = element_energies[energy_key]
        seen_element_ids: set[int] = set()
        element_elastic_sum = 0.0
        component_elastic: dict[int, float] = {}
        for element_row in mode_element_rows:
            element_id = int(element_row["element_id"])
            if element_id in seen_element_ids:
                raise ValueError(f"Duplicate element-energy row for mode {mode}, element {element_id}")
            seen_element_ids.add(element_id)
            if element_id not in damping_candidates_by_element:
                raise ValueError(f"Element {element_id} has no damping-candidate row")
            elastic = _finite(element_row["elastic_stiffness_raw"], "element elastic stiffness")
            component_id = int(element_row["component_id"])
            candidate_row = damping_candidates_by_element[element_id]
            if (
                component_id != int(candidate_row["component_id"])
                or int(element_row["kind_id"]) != int(candidate_row["kind_id"])
            ):
                raise ValueError(f"Element-energy mapping is inconsistent for element {element_id}")
            element_elastic_sum += elastic
            component_elastic[component_id] = component_elastic.get(component_id, 0.0) + elastic
        if set(damping_candidates_by_element) != seen_element_ids:
            raise ValueError(f"Mode {mode} element-energy coverage is incomplete")
        if not math.isclose(
            element_elastic_sum, stiffness_raw, rel_tol=2.0e-11, abs_tol=2.0e-12
        ):
            raise ValueError(f"Element elastic-energy closure failed for mode {mode}")

        saved_component_elastic = {
            int(row["component_id"]): _finite(
                row["elastic_stiffness_raw"], "component elastic stiffness"
            )
            for row in component_energies.get(energy_key, [])
        }
        if set(saved_component_elastic) != set(component_elastic):
            raise ValueError(f"Element/component energy coverage differs for mode {mode}")
        for component_id, elastic in component_elastic.items():
            if not math.isclose(
                elastic,
                saved_component_elastic[component_id],
                rel_tol=2.0e-11,
                abs_tol=2.0e-12,
            ):
                raise ValueError(f"Element/component energy closure failed for mode {mode}")

        damping_projections: list[dict[str, object]] = []
        for candidate_id in DAMPING_CANDIDATE_IDS:
            column = f"{candidate_id}_beta_s"
            unavailable_elements: list[int] = []
            damping_raw = 0.0
            component_damping: dict[int, float] = {}
            component_beta_values: dict[int, set[float]] = {}
            for element_row in mode_element_rows:
                element_id = int(element_row["element_id"])
                candidate_row = damping_candidates_by_element[element_id]
                elastic = _finite(element_row["elastic_stiffness_raw"], "element elastic stiffness")
                raw_beta = candidate_row.get(column, "")
                if raw_beta is None or not str(raw_beta).strip():
                    if abs(elastic) > 1.0e-14 * max(1.0, stiffness_raw):
                        unavailable_elements.append(element_id)
                    continue
                beta = _finite(raw_beta, column)
                component_id = int(element_row["component_id"])
                damping_raw += beta * elastic
                component_damping[component_id] = (
                    component_damping.get(component_id, 0.0) + beta * elastic
                )
                component_beta_values.setdefault(component_id, set()).add(beta)
            available = not unavailable_elements
            modal_damping = damping_raw / q_raw**2 if available else None
            damping_ratio = (
                damping_raw * omega / (2.0 * velocity_mass_raw) if available else None
            )
            breakdown: list[dict[str, object]] = []
            if available:
                for component_id, elastic in component_elastic.items():
                    component_c = component_damping.get(component_id, 0.0)
                    zeta = component_c * omega / (2.0 * velocity_mass_raw)
                    breakdown.append(
                        {
                            "component": component_name_by_id[component_id],
                            "component_id": component_id,
                            "elastic_stiffness_raw": elastic,
                            "elastic_stiffness_fraction": elastic / stiffness_raw,
                            "candidate_beta_s_values": sorted(
                                component_beta_values.get(component_id, set())
                            ),
                            "strain_energy_weighted_effective_beta_s": (
                                0.0 if elastic == 0.0 else component_c / elastic
                            ),
                            "rayleigh_damping_raw_s_times_stiffness": component_c,
                            "projected_damping_ratio_contribution": zeta,
                            "fraction_of_total_projected_damping": (
                                0.0 if damping_ratio == 0.0 else zeta / damping_ratio
                            ),
                        }
                    )
                breakdown.sort(
                    key=lambda item: float(item["elastic_stiffness_fraction"]),
                    reverse=True,
                )
            damping_projections.append(
                {
                    "candidate_id": candidate_id,
                    "available": available,
                    "selected": False,
                    "evidence": conversion_damping.get("candidates", {}).get(candidate_id),
                    "unavailable_element_ids": unavailable_elements,
                    "generalized_rayleigh_damping_n_s_per_m": modal_damping,
                    "rayleigh_energy_projected_damping_ratio": damping_ratio,
                    "strain_energy_weighted_effective_beta_s": (
                        damping_raw / stiffness_raw if available else None
                    ),
                    "damping_component_breakdown": breakdown,
                }
            )

        comparison_states = action_states_at(zero_rpm, mode)
        comparison_anchor = max(
            (state[name] for state in comparison_states for name in ("ux", "uy", "uz")),
            key=abs,
        )
        comparison_phase = comparison_anchor.conjugate() / abs(comparison_anchor)
        comparison_q = max(_norm3(state) for state in comparison_states)
        vector_selected = [
            state[name] * scale
            for state in raw_states
            for name in ("ux", "uy", "uz")
        ]
        vector_comparison = [
            state[name] * comparison_phase / comparison_q
            for state in comparison_states
            for name in ("ux", "uy", "uz")
        ]
        inner = sum(a.conjugate() * b for a, b in zip(vector_comparison, vector_selected))
        norm_a = sum(abs(value) ** 2 for value in vector_comparison)
        norm_b = sum(abs(value) ** 2 for value in vector_selected)
        action_mac = abs(inner) ** 2 / (norm_a * norm_b)

        mode_summaries.append(
            {
                "tracked_mode": mode,
                "frequency_hz": frequency,
                "angular_frequency_rad_per_s": omega,
                "normalization": {
                    "generalized_coordinate": "maximum complex 3-D translational amplitude over the 39 QBlade external-action locations",
                    "generalized_coordinate_units": "m",
                    "normalization_value_m": 1.0,
                    "raw_action_coordinate": q_raw,
                    "phase_anchor_action_node_id": mappings[anchor_site_index]["action_node_id"],
                    "phase_anchor_component": anchor_name,
                    "phase_rotation_real": phase.real,
                    "phase_rotation_imag": phase.imag,
                    "phase_rule": "largest-magnitude action-set translation component is real and positive",
                },
                "action_translation_complex_mode_purity": {
                    "component_count": len(action_translations),
                    "imaginary_l2_norm_over_real_l2_norm": quadrature_to_in_phase,
                    "real_part_relative_l2_approximation_error": imaginary_norm / complex_norm,
                    "phase_residual_definition": "absolute phase distance modulo pi from the phase-anchored real axis",
                    "maximum_phase_residual_rad_all_nonzero_components": max(
                        abs(value) for value in phase_residuals
                    ),
                    "amplitude_weighted_rms_phase_residual_rad": weighted_phase_rms,
                    "material_component_amplitude_threshold_fraction_of_max": 0.05,
                    "maximum_phase_residual_rad_material_components": max(
                        material_phase_residuals, default=0.0
                    ),
                    "phase_outlier_threshold_rad": 0.05,
                    "phase_outlier_translation_energy_fraction": phase_outlier_energy_fraction,
                    "single_real_scalar_gate": {
                        "maximum_imaginary_over_real_l2": 0.02,
                        "maximum_weighted_phase_rms_rad": 0.02,
                        "maximum_material_component_phase_residual_rad": 0.05,
                        "passed": real_scalar_gate,
                    },
                    "response_model_requirement": (
                        "single real scalar using the real part of the phase-anchored action shape is defensible"
                        if real_scalar_gate
                        else "retain real/imaginary coordinate pair or a multimode representation"
                    ),
                },
                "generalized_modal_mass_kg": modal_mass,
                "generalized_modal_stiffness_n_per_m": modal_stiffness,
                "selected_damping_candidate_id": None,
                "generalized_rayleigh_damping_n_s_per_m": None,
                "rayleigh_energy_projected_damping_ratio": None,
                "strain_energy_weighted_effective_beta_s": None,
                "damping_candidates": damping_projections,
                "eigen_residual_relative": _finite(
                    prop["eigen_residual_relative"], "eigen residual"
                ),
                "left_right_mass_product": {
                    "real": _finite(prop["left_right_mass_real"], "left-right mass real"),
                    "imag": _finite(prop["left_right_mass_imag"], "left-right mass imag"),
                },
                "elastic_stiffness_to_velocity_inertia_ratio": _finite(
                    prop["stiffness_to_inertia_ratio"], "stiffness/inertia ratio"
                ),
                "comparison_to_zero_rpm": {
                    "comparison_rpm": zero_rpm,
                    "frequency_hz": frequencies[zero_rpm][mode - 1],
                    "frequency_shift_hz": frequency - frequencies[zero_rpm][mode - 1],
                    "frequency_shift_fraction": (
                        frequency / frequencies[zero_rpm][mode - 1] - 1.0
                    ),
                    "action_translation_mac": action_mac,
                },
                "reviewed_frequency_delta_hz": reviewed_delta.get(mode),
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(csv_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    output_duals_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_duals_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(force_dual_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(force_dual_rows)

    tracked_clean_checkout_evidence: dict[str, dict[str, object]] = {}
    independent_repeatability: dict[str, object] | None = None
    if tracked_evidence_dir is not None:
        tracked_evidence_dir = tracked_evidence_dir.resolve()
        tracked_names = (
            "hvawt_owens_request.json",
            "hvawt_owens_run_metadata.txt",
            "hvawt_campbell_mode_tracking.csv",
            "hvawt_campbell_mode_properties.csv",
            "hvawt_response_mode_shapes_repeat.csv",
            "hvawt_response_mode_force_duals_repeat.csv",
        )
        tracked_paths = {
            name: tracked_evidence_dir / name for name in tracked_names
        }
        missing_tracked = [
            str(path) for path in tracked_paths.values() if not path.is_file()
        ]
        if missing_tracked:
            raise FileNotFoundError(
                "Tracked clean-checkout modal evidence is missing: "
                + ", ".join(missing_tracked)
            )
        tracked_clean_checkout_evidence = {
            name: _artifact_identity(path)
            for name, path in tracked_paths.items()
        }
        independent_repeatability = compare_independent_action_evidence(
            output_csv,
            output_duals_csv,
            tracked_paths["hvawt_response_mode_shapes_repeat.csv"],
            tracked_paths["hvawt_response_mode_force_duals_repeat.csv"],
        )
        if not independent_repeatability["passed"]:
            raise ValueError(
                "Independent tracked action evidence failed its repeatability gates"
            )

    result: dict[str, object] = {
        "schema": "vorlap-hvawt-response-mode-identification-v2",
        "status": "undamped_modal_evidence_identified_damping_candidates_unresolved",
        "selected_rpm": selected_rpm,
        "selected_modes": list(modes),
        "saved_action_basis_modes": basis_modes,
        "saved_action_basis_rpms": list(evidence_rpms),
        "external_action_mapping": {
            "site_count": len(mappings),
            "method": "component-local linear interpolation of the saved GXBeam point state in normalized QBlade body fraction; the eigenproblem mesh is unchanged",
            "all_sites_bracketed": True,
            "exact_mesh_node_site_count": sum(
                float(mapping["bracket_span"]) == 0.0 for mapping in mappings
            ),
            "maximum_bracket_span_fraction": max(
                float(mapping["bracket_span"]) for mapping in mappings
            ),
            "maximum_nearest_node_fraction_error": max(
                float(mapping["nearest_fraction_error"]) for mapping in mappings
            ),
            "action_shape_artifact": _artifact_identity(output_csv),
            "action_shape_row_count": len(csv_rows),
            "qblade_global_force_output_indices": sorted(
                index
                for mapping in mappings
                for index in mapping["output_indices"].values()
            ),
        },
        "minimum_norm_force_duals": {
            "artifact": _artifact_identity(output_duals_csv),
            "row_count": len(force_dual_rows),
            "target_modes": list(modes),
            "basis_modes": basis_modes,
            "svd_evidence_by_rpm": force_dual_evidence,
            "zero_rpm_degenerate_mode_clusters": degenerate_zero_rpm_clusters,
            "repeatability_interpretation": "Individual vectors inside a degenerate zero-RPM cluster may rotate between eigensolves, so the raw 12-column condition number and bitwise CSV hashes are not invariants. Decay readiness is based on full rank/residual gates plus cross-regeneration invariance of the selected mode shapes, target force-dual columns, and their generalized-force action on the alternate basis.",
        },
        "independent_regeneration_evidence": independent_repeatability,
        "mode_state_convention": {
            "raw_vector": "GXBeam first-order mixed right eigenvector from K v + lambda M v = 0",
            "serialized_point_fields": "u, Wiener-Milenkovic theta, V, and Omega extracted by GXBeam.AssemblyState",
            "serialized_element_fields_used_for_properties": "Fi and Mi extracted by GXBeam.AssemblyState",
            "mass_projection": "m_q = (sum_e L v_e^H M_e_global v_e + sum_p v_p^H M_p_global v_p)/(omega^2 q_raw^2)",
            "stiffness_projection": "k_q = (sum_e L f_e^H C_e f_e)/q_raw^2",
            "rayleigh_projection": "c_q = (sum_e beta_e L f_e^H C_e f_e)/q_raw^2; zeta_q = c_q/(2 omega m_q)",
            "complex_quadratic_convention": "Hermitian transpose; physical velocity states are used directly rather than inferred from lambda times displacement",
            "units": {
                "q": "m",
                "m_q": "kg",
                "k_q": "N/m",
                "c_q": "N s/m",
                "beta_e": "s",
            },
        },
        "damping_evidence": {
            "type": "unselected QBlade stiffness-proportional Rayleigh candidate coefficients projected by per-element undamped modal strain energy",
            "selected_damping_candidate_id": None,
            "candidate_ids": list(DAMPING_CANDIDATE_IDS),
            "candidate_semantics": conversion_damping.get("candidates", {}),
            "source_semantics": conversion_damping,
            "numerical_undamped_eigenvalue_real_parts_used": False,
            "experimental_or_measured_damping": False,
            "cross_solver_gate": "No candidate is selected or called measured until an active QBlade binary discriminates the original and swapped SUBELEMENTS row order by paired decay.",
            "substructure_column_semantics_caveat": "QBlade 2.0.9.7 documentation and archived 2.0.4 code use zero-based row[19] as DAMP and optional row[20] as drag, while the local reference header labels row[19] Drag and row[20] DAMP. Both source projections are retained without changing the reference row.",
            "required_qblade_decay_probe": {
                "aerodynamics": "off/vacuum if supported; otherwise zero inflow with aerodynamic-force channels checked negligible",
                "rotor_speeds_rpm": [0.0, selected_rpm],
                "paired_reference_models_before_any_correction": [
                    "original reference row unchanged",
                    "temporary swapped final-two-column row with every other byte-equivalent physical input held fixed",
                ],
                "optional_parser_control": "A 20-column temporary row ending at documented DAMP may be added only as a parser control; it does not replace the original-versus-swapped pair.",
                "target_peak_action_translation_m": [0.0001, 0.0003, 0.001],
                "excitation": "one target at a time; apply the serialized 12-mode SVD force-dual column as a five-cycle narrowband Hann burst at the identified frequency, with complex real/imaginary coefficients implemented as harmonic quadratures, then set all 117 external-force rows exactly to zero",
                "excitation_gate": "The corresponding SVD evidence at that RPM must pass full-rank, condition-number, biorthogonality-residual, and off-target generalized-force gates before the decay is interpreted.",
                "minimum_unforced_cycles": {str(mode): (12 if mode == 6 else 20) for mode in modes},
                "time_steps_s": [0.001, 0.0005],
                "projection": "Use the saved body-fixed 39-site complex action basis and rotate it consistently to the QBlade global frame at nonzero RPM; project with the matching force/displacement dual convention.",
                "fit": "Use a damped sinusoid only when the candidate is underdamped; for zeta>=1 fit the two real decay poles. Estimate frequency or poles and damping independently for every amplitude, time step, row order, RPM, and target mode.",
                "acceptance_gates": {
                    "maximum_relative_frequency_change_between_time_steps": 0.005,
                    "maximum_absolute_damping_ratio_change_between_time_steps": 0.01,
                    "maximum_relative_frequency_span_across_amplitudes": 0.01,
                    "maximum_absolute_damping_ratio_span_across_amplitudes": 0.02,
                    "maximum_off_target_modal_amplitude_fraction": 0.10,
                    "minimum_selected_action_shape_mac_across_independent_regenerations": 0.999999,
                    "maximum_phase_aligned_selected_action_shape_relative_difference": 1.0e-5,
                    "maximum_phase_aligned_force_dual_relative_difference": 1.0e-5,
                    "maximum_cross_regeneration_generalized_force_target_error": 1.0e-5,
                    "maximum_cross_regeneration_off_target_generalized_force": 1.0e-4,
                    "row_order_discriminator": "The original and swapped models must yield statistically/numerically separable decay rates consistent with one serialized candidate; otherwise active-binary damping semantics remain unresolved.",
                },
                "repeatability_requirement": "After any runner, GXBeam, Julia, or modal-input change, regenerate an independent 0/1-RPM basis and reapply the selected-shape/force-dual cross-basis gates; do not reject an otherwise invariant target dual solely because a degenerate nuisance basis changes the raw full-matrix condition number.",
                "prohibition": "Do not tune a candidate coefficient from this comparison and do not correct the tracked source row until the paired results are archived.",
            },
        },
        "orthogonality_at_selected_rpm": orthogonality_evidence,
        "modes": mode_summaries,
        "provenance": {
            "identifier": _artifact_identity(Path(__file__)),
            "external_config": _artifact_identity(external_config),
            "modal_bundle_label": _modal_bundle_label(conversion),
            "modal_artifacts": {
                name: _artifact_identity(
                    path, f"{_modal_bundle_label(conversion)}/{name}"
                )
                for name, path in input_paths.items()
            },
            "reviewed_frequencies": (
                None
                if reviewed_frequencies is None
                else _artifact_identity(reviewed_frequencies)
            ),
            "python_runtime": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable_name": Path(sys.executable).name,
                "numpy_version": np.__version__,
            },
            "conversion_source_artifacts": [
                {
                    "label": _portable_manifest_label(str(label)),
                    "sha256": str(digest),
                    "size_bytes": int(conversion.get("input_sizes_bytes", {}).get(label, 0)),
                }
                for label, digest in sorted(conversion.get("inputs", {}).items())
            ],
            "tracked_clean_checkout_evidence": tracked_clean_checkout_evidence,
        },
        "limitations": [
            "The modal basis is the converted OWENS/GXBeam structural model, not a QBlade/Chrono eigenvector export.",
            "The positive-definite converted sectional mass convention intentionally differs from the indefinite literal QBlade Chrono construction documented by the converter.",
            "Action-location values use component-fraction interpolation of the converted point state; they are not new eigenproblem nodes.",
            "Every Rayleigh result is an unselected input-semantics candidate, not measured damping, and remains subject to the paired active-QBlade decay discriminator.",
            "Aerodynamic damping, response-dependent lock-in, turbulent forcing, and nonlinear limit-cycle behavior are absent from this structural modal identification.",
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modal-dir", type=Path, required=True)
    parser.add_argument("--external-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-duals-csv", type=Path)
    parser.add_argument("--rpm", type=float, default=1.0)
    parser.add_argument("--comparison-rpm", type=float, default=0.0)
    parser.add_argument("--modes", default="6,12")
    parser.add_argument("--reviewed-frequencies", type=Path)
    parser.add_argument(
        "--tracked-evidence-dir",
        type=Path,
        help="Tracked directory containing frozen OWENS provenance and independent-repeat action artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    modes = _parse_list(args.modes, int)
    if len(set(modes)) != len(modes) or any(mode < 1 for mode in modes):
        raise ValueError("Requested modes must be unique positive integers")
    result = identify(
        args.modal_dir,
        args.external_config,
        args.output_json,
        args.output_csv,
        args.output_duals_csv,
        rpm=args.rpm,
        comparison_rpm=args.comparison_rpm,
        modes=modes,
        reviewed_frequencies=args.reviewed_frequencies,
        tracked_evidence_dir=args.tracked_evidence_dir,
    )
    print(
        f"Identified {len(result['modes'])} modes at "
        f"{result['external_action_mapping']['site_count']} QBlade action locations."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
