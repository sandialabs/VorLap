"""Convert the mode-12 QBlade decay decision into two-mode damping evidence.

The decay campaign measures the active QBlade behavior at mode 12.  Once its
canonical C20 structural encoding and scalar/station precedence are resolved,
the tracked OWENS strain-energy projections provide the corresponding
stiffness-proportional Rayleigh input damping for modes 6 and 12.  Mode 6 is
therefore explicitly an energy projection, not a measured QBlade decay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence


SELECTION_SCHEMA = "vorlap-hvawt-modal-damping-selection-v2"
DOCUMENTED_SELECTION_BASIS = "qblade_parser_semantics_and_bounded_engineering_damping_assumption"
ANALYSIS_SCHEMA = "vorlap-hvawt-modal-decay-analysis-v3"
CLASSIFICATION_SCHEMA = "vorlap-hvawt-rayleigh-discriminator-classification-v2"
IDENTIFICATION_SCHEMA = "vorlap-hvawt-response-mode-identification-v2"
TARGET_MODES = (6, 12)
MAX_ALL_BETA_ZERO_DAMPING_RATIO = 0.001
MAX_MODE12_MEASURED_PROJECTED_ABS_ERROR = 0.01
_STATION_CANDIDATE = "reference_header_intended"
_SCALAR_CANDIDATE = "legacy_converter_header_scalar"
_VARIABLE_SECTION_COMPONENTS = {"blade", "lower_strut", "upper_strut"}
_STATION_PROJECTION_COMPONENTS = _VARIABLE_SECTION_COMPONENTS | {"substructure"}
ENGINEERING_SCALAR_BETA_S = 0.0011
_ENGINEERING_SCALAR_COMPONENTS = {"substructure", "torque_tube", "tower"}
_EXPECTED_COMPONENTS = {
    "blade",
    "lower_strut",
    "upper_strut",
    "torque_tube",
    "tower",
    "substructure",
    "drivetrain",
    "rigid_connector",
    "released_connector",
    "bearing",
}
_EXPECTED_MODE_VALUES = {
    6: (2.5115545017693828, 337.9307091909933, 84112.78376212968),
    12: (4.572593676110874, 123.99012430136153, 102171.33806478149),
}
_EXPECTED_IDENTIFICATION_SHA256 = (
    "da8ac3a8ffa64d99a9f1363b9bbf81dd06a5695c27c775193c54d5f4c85d3941"
)
_EXPECTED_IDENTIFICATION_SIZE_BYTES = 184137
_VARIANTS = {
    "O",
    "S",
    "C20",
    "C20_station_zero",
    "C20_scalar_zero",
    "all_beta_zero",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "label": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _nonnegative_float(value: object, *, label: str) -> float:
    result = _finite_float(value, label=label)
    if result < 0.0:
        raise ValueError(f"{label} must be nonnegative.")
    return result


def _require_close(actual: float, expected: float, *, label: str, atol: float = 1.0e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=atol):
        raise ValueError(f"{label} is inconsistent with its bound evidence.")


def _candidate(mode: Mapping[str, object], candidate_id: str) -> Mapping[str, object]:
    rows = mode.get("damping_candidates")
    if not isinstance(rows, list):
        raise ValueError("Mode identification lacks damping candidates.")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1 or matches[0].get("available") is not True:
        raise ValueError(
            f"Mode identification lacks one available {candidate_id!r} candidate."
        )
    return matches[0]


def _component_map(candidate: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = candidate.get("damping_component_breakdown")
    if not isinstance(rows, list):
        raise ValueError("Damping candidate lacks its component breakdown.")
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("component"), str):
            raise ValueError("Damping candidate has a malformed component row.")
        name = str(row["component"])
        if name in result:
            raise ValueError(f"Damping candidate repeats component {name!r}.")
        result[name] = row
    if set(result) != _EXPECTED_COMPONENTS:
        raise ValueError("Damping candidate does not contain the exact component set.")
    projected = _nonnegative_float(
        candidate.get("rayleigh_energy_projected_damping_ratio"),
        label="candidate projected damping ratio",
    )
    summed = 0.0
    for name, row in result.items():
        summed += _nonnegative_float(
            row.get("projected_damping_ratio_contribution"),
            label=f"{name} projected damping contribution",
        )
    _require_close(summed, projected, label="candidate component sum")
    return result


def _projected_zeta(
    mode: Mapping[str, object],
    *,
    damping_source: str,
) -> tuple[float, str]:
    station = _candidate(mode, _STATION_CANDIDATE)
    scalar = _candidate(mode, _SCALAR_CANDIDATE)
    station_components = _component_map(station)
    scalar_components = _component_map(scalar)
    if set(station_components) != set(scalar_components):
        raise ValueError("Scalar and station candidates do not cover the same components.")
    if damping_source == "station_vector_dominant":
        return (
            sum(
                _nonnegative_float(
                    station_components[name].get("projected_damping_ratio_contribution"),
                    label=f"{name} station projected damping contribution",
                )
                for name in _STATION_PROJECTION_COMPONENTS
            ),
            "c20_substructure_plus_station_vectors",
        )
    if damping_source == "file_scalar_dominant":
        return (
            _finite_float(
                scalar.get("rayleigh_energy_projected_damping_ratio"),
                label="file-scalar projected damping ratio",
            ),
            _SCALAR_CANDIDATE,
        )
    if damping_source != "both_file_scalar_and_station_vector":
        raise ValueError(
            "Damping source does not select a defensible scalar/station Rayleigh projection."
        )

    total = 0.0
    for name in sorted(station_components):
        source = scalar_components[name] if name in _VARIABLE_SECTION_COMPONENTS else station_components[name]
        total += _finite_float(
            source.get("projected_damping_ratio_contribution"),
            label=f"{name} projected damping contribution",
        )
        if name in _VARIABLE_SECTION_COMPONENTS:
            total += _finite_float(
                station_components[name].get("projected_damping_ratio_contribution"),
                label=f"{name} station projected damping contribution",
            )
    return total, "c20_additive_station_and_file_scalar"


def _validated_classification(classification: Mapping[str, object]) -> dict[str, object]:
    if classification.get("schema") != CLASSIFICATION_SCHEMA:
        raise ValueError("Unexpected Rayleigh classification schema.")
    if classification.get("complete_and_interpretable") is not True:
        raise ValueError("Rayleigh classification is not complete and interpretable.")
    ratios_raw = classification.get("variant_damping_ratios")
    if not isinstance(ratios_raw, Mapping) or set(ratios_raw) != _VARIANTS:
        raise ValueError("Rayleigh classification lacks the exact six variant ratios.")
    ratios = {
        name: _nonnegative_float(ratios_raw[name], label=f"{name} damping ratio")
        for name in _VARIANTS
    }
    if any(value >= 0.8 for value in ratios.values()):
        raise ValueError("Rayleigh variant damping is outside the interpretable range.")
    z0 = ratios["all_beta_zero"]
    scalar_removal_effect = ratios["C20"] - ratios["C20_scalar_zero"]
    station_removal_effect = ratios["C20"] - ratios["C20_station_zero"]
    checks = {
        "effective C20 damping": ("effective_c20_damping_ratio", ratios["C20"]),
        "all-beta-zero residual": ("all_beta_zero_residual_damping_ratio", z0),
        "file-scalar removal effect": (
            "file_scalar_removal_effect_damping_ratio",
            scalar_removal_effect,
        ),
        "station-vector removal effect": (
            "station_vector_removal_effect_damping_ratio",
            station_removal_effect,
        ),
        "physical Rayleigh increment": (
            "physical_rayleigh_increment_over_all_beta_zero",
            ratios["C20"] - z0,
        ),
    }
    for label, (key, expected) in checks.items():
        actual = _finite_float(classification.get(key), label=label)
        _require_close(actual, expected, label=label)
    if (
        classification.get("substructure_contribution_separately_identified") is not False
        or classification.get("scalar_station_interaction_separately_identified") is not False
        or classification.get("source_decision_uses_direct_removal_effects") is not True
    ):
        raise ValueError("Rayleigh source-identification boundary is inconsistent.")
    attribution_tolerance = _nonnegative_float(
        classification.get("attribution_absolute_damping_tolerance"),
        label="attribution damping tolerance",
    )
    expected_attribution_tolerance = max(0.002, 0.1 * abs(ratios["C20"] - z0))
    _require_close(
        attribution_tolerance,
        expected_attribution_tolerance,
        label="attribution damping tolerance",
    )
    if min(scalar_removal_effect, station_removal_effect) < -attribution_tolerance:
        expected_source = "nonphysical_negative_removal_effect"
    elif scalar_removal_effect > attribution_tolerance and station_removal_effect > attribution_tolerance:
        expected_source = "both_file_scalar_and_station_vector"
    elif scalar_removal_effect > attribution_tolerance:
        expected_source = "file_scalar_dominant"
    elif station_removal_effect > attribution_tolerance:
        expected_source = "station_vector_dominant"
    else:
        expected_source = "rayleigh_beta_effect_not_resolved"
    if classification.get("damping_source") != expected_source:
        raise ValueError("Rayleigh damping-source label is inconsistent with variant fits.")

    parser_spread = max(ratios[name] for name in ("O", "S", "C20")) - min(
        ratios[name] for name in ("O", "S", "C20")
    )
    parser_tolerance = max(0.002, 0.15 * parser_spread)
    _require_close(
        _nonnegative_float(
            classification.get("parser_absolute_damping_tolerance"),
            label="parser damping tolerance",
        ),
        parser_tolerance,
        label="parser damping tolerance",
    )
    original_close = abs(ratios["O"] - ratios["C20"]) <= parser_tolerance
    swapped_close = abs(ratios["S"] - ratios["C20"]) <= parser_tolerance
    expected_parser = (
        "header_21st_token_is_damping"
        if original_close and not swapped_close
        else "fixed_20th_numeric_token_is_damping"
        if swapped_close and not original_close
        else "tail_semantics_not_observable"
        if original_close and swapped_close
        else "tail_semantics_noncanonical_or_ambiguous"
    )
    if classification.get("parser_semantics") != expected_parser:
        raise ValueError("Rayleigh parser-semantics label is inconsistent with variant fits.")
    return {
        "ratios": ratios,
        "residual": z0,
        "physical_increment": ratios["C20"] - z0,
    }


def _validate_mode_identity(mode: Mapping[str, object], tracked_mode: int) -> tuple[float, float, float]:
    expected_frequency, expected_mass, expected_stiffness = _EXPECTED_MODE_VALUES[tracked_mode]
    frequency = _nonnegative_float(mode.get("frequency_hz"), label="mode frequency")
    mass = _nonnegative_float(mode.get("generalized_modal_mass_kg"), label="generalized modal mass")
    stiffness = _nonnegative_float(
        mode.get("generalized_modal_stiffness_n_per_m"),
        label="generalized modal stiffness",
    )
    if min(frequency, mass, stiffness) <= 0.0:
        raise ValueError("Mode frequency, mass, and stiffness must be positive.")
    _require_close(frequency, expected_frequency, label=f"mode {tracked_mode} frequency")
    _require_close(mass, expected_mass, label=f"mode {tracked_mode} mass")
    _require_close(stiffness, expected_stiffness, label=f"mode {tracked_mode} stiffness")
    if not math.isclose(
        stiffness,
        mass * (2.0 * math.pi * frequency) ** 2,
        rel_tol=0.002,
        abs_tol=0.2,
    ):
        raise ValueError(
            f"mode {tracked_mode} mass-stiffness-frequency identity is inconsistent."
        )
    return frequency, mass, stiffness


def derive_damping_selection(
    analysis: Mapping[str, object],
    identification: Mapping[str, object],
) -> dict[str, object]:
    """Return a fail-closed two-mode damping-selection payload."""

    if analysis.get("schema") != ANALYSIS_SCHEMA or analysis.get("status") != "complete_interpretable":
        raise ValueError("Rayleigh analysis is not complete and interpretable.")
    classification = analysis.get("classification")
    if not isinstance(classification, Mapping):
        raise ValueError("Rayleigh analysis lacks its classification block.")
    if (
        classification.get("selection_allowed") is not True
        or classification.get("selected_unambiguous_substructure_encoding") != "C20"
    ):
        raise ValueError("Rayleigh analysis does not select the canonical C20 encoding.")
    validated_classification = _validated_classification(classification)
    residual = float(validated_classification["residual"])
    if residual < 0.0 or residual > MAX_ALL_BETA_ZERO_DAMPING_RATIO:
        raise ValueError(
            "All-beta-zero residual damping exceeds the frozen physical-Rayleigh gate."
        )
    damping_source = classification.get("damping_source")
    if not isinstance(damping_source, str):
        raise ValueError("Rayleigh classification lacks a damping-source decision.")
    parser_semantics = classification.get("parser_semantics")
    if not isinstance(parser_semantics, str) or not parser_semantics:
        raise ValueError("Rayleigh classification lacks a parser-semantics decision.")
    if identification.get("schema") != IDENTIFICATION_SCHEMA:
        raise ValueError("Unexpected H-VAWT mode-identification schema.")
    if identification.get("selected_rpm") != 1.0 or identification.get("selected_modes") != [6, 12]:
        raise ValueError("Mode identification must contain the frozen 1-RPM modes 6 and 12.")
    raw_modes = identification.get("modes")
    if not isinstance(raw_modes, list):
        raise ValueError("Mode identification has no mode list.")
    selected_mode_rows = [
        row
        for row in raw_modes
        if isinstance(row, Mapping)
        and type(row.get("tracked_mode")) is int
        and int(row["tracked_mode"]) in TARGET_MODES
    ]
    mode_map = {int(row["tracked_mode"]): row for row in selected_mode_rows}
    if len(selected_mode_rows) != len(TARGET_MODES) or set(mode_map) != set(TARGET_MODES):
        raise ValueError("Mode identification must contain modes 6 and 12 exactly once.")

    effective_mode12 = float(validated_classification["ratios"]["C20"])
    measured_physical_mode12 = float(validated_classification["physical_increment"])
    modes: list[dict[str, object]] = []
    projection_method: str | None = None
    for tracked_mode in TARGET_MODES:
        mode = mode_map[tracked_mode]
        zeta, method = _projected_zeta(mode, damping_source=damping_source)
        if projection_method is None:
            projection_method = method
        elif projection_method != method:
            raise ValueError("Mode damping projections selected inconsistent semantics.")
        frequency, mass, stiffness = _validate_mode_identity(mode, tracked_mode)
        reconciliation_error = (
            abs(measured_physical_mode12 - zeta) if tracked_mode == 12 else None
        )
        if (
            reconciliation_error is not None
            and reconciliation_error > MAX_MODE12_MEASURED_PROJECTED_ABS_ERROR
        ):
            raise ValueError(
                "Measured mode-12 physical Rayleigh increment does not reconcile "
                "with the selected energy projection."
            )
        modes.append(
            {
                "tracked_mode": tracked_mode,
                "frequency_hz": frequency,
                "generalized_modal_mass_kg": mass,
                "generalized_modal_stiffness_n_per_m": stiffness,
                "rayleigh_energy_projected_damping_ratio": zeta,
                "generalized_rayleigh_damping_n_s_per_m": (
                    2.0 * zeta * 2.0 * math.pi * frequency * mass
                ),
                "qblade_effective_decay_damping_ratio": (
                    effective_mode12 if tracked_mode == 12 else None
                ),
                "qblade_physical_rayleigh_increment_damping_ratio": (
                    measured_physical_mode12 if tracked_mode == 12 else None
                ),
                "measured_projected_absolute_error": reconciliation_error,
                "evidence_role": (
                    "active_qblade_decay_plus_energy_projection"
                    if tracked_mode == 12
                    else "energy_projection_from_selected_qblade_input_semantics"
                ),
            }
        )

    payload: dict[str, object] = {
        "schema": SELECTION_SCHEMA,
        "status": "selected",
        "selected_structural_encoding": "C20",
        "parser_semantics": parser_semantics,
        "damping_source": damping_source,
        "projection_method": projection_method,
        "all_beta_zero_residual_damping_ratio": residual,
        "mode12_measured_projected_absolute_error_limit": (
            MAX_MODE12_MEASURED_PROJECTED_ABS_ERROR
        ),
        "modes": modes,
        "claim_boundary": (
            "Mode 12 includes an active-QBlade effective decay estimate. Mode 6 damping is "
            "the stiffness-energy projection of the selected Rayleigh input semantics and "
            "is not a measured QBlade decay or experimental damping ratio. Scalar and "
            "station activity use direct removal effects; substructure and scalar-station "
            "interaction are not separately identified, and their aggregate discrepancy "
            "is bounded only by the mode-12 measured-versus-projected gate."
        ),
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def derive_documented_damping_selection(
    identification: Mapping[str, object],
) -> dict[str, object]:
    """Select the corrected Rayleigh inputs without fitting a decay record.

    The original substructure file puts ``1.0 0.02`` after the diameter while
    its header labels those fields ``Drag DAMP``.  Both the archived QBlade
    parser and the current documentation define the opposite numerical order:
    the twentieth value is stiffness-proportional DAMP and the optional
    twenty-first value is aerodynamic drag.  The canonical C20 input therefore
    ends at DAMP=0.02.  With that correction, the numerical energy projection
    previously labelled ``reference_header_intended`` is the documented-parser
    projection.  However, beta is a stiffness-proportional coefficient in
    seconds rather than a direct fraction of critical damping.  The raw
    substructure/tower beta=0.02 and torque-tube beta=0.03 would project to
    implausibly high global damping.  For the bounded engineering pilot those
    scalar fallbacks are normalized to the model's existing beta=0.0011 s,
    while the distributed blade/strut station values are preserved.

    This function deliberately does not manufacture an effective QBlade decay
    measurement.  It selects model-input damping for the engineering pilot and
    leaves an active-binary smoke/decay check as verification evidence.
    """

    if identification.get("schema") != IDENTIFICATION_SCHEMA:
        raise ValueError("Unexpected H-VAWT mode-identification schema.")
    if identification.get("selected_rpm") != 1.0 or identification.get("selected_modes") != [6, 12]:
        raise ValueError("Mode identification must contain the frozen 1-RPM modes 6 and 12.")
    raw_modes = identification.get("modes")
    if not isinstance(raw_modes, list):
        raise ValueError("Mode identification has no mode list.")
    mode_map: dict[int, Mapping[str, object]] = {}
    for row in raw_modes:
        if not isinstance(row, Mapping) or type(row.get("tracked_mode")) is not int:
            continue
        tracked_mode = int(row["tracked_mode"])
        if tracked_mode in TARGET_MODES:
            if tracked_mode in mode_map:
                raise ValueError(f"Mode identification repeats mode {tracked_mode}.")
            mode_map[tracked_mode] = row
    if set(mode_map) != set(TARGET_MODES):
        raise ValueError("Mode identification must contain modes 6 and 12 exactly once.")

    modes: list[dict[str, object]] = []
    for tracked_mode in TARGET_MODES:
        mode = mode_map[tracked_mode]
        frequency, mass, stiffness = _validate_mode_identity(mode, tracked_mode)
        candidate = _candidate(mode, _STATION_CANDIDATE)
        components = _component_map(candidate)
        component_rows: list[dict[str, object]] = []
        beta_energy_sum = 0.0
        for name, component in components.items():
            energy_fraction = _nonnegative_float(
                component.get("elastic_stiffness_fraction"),
                label=f"mode {tracked_mode} {name} elastic-stiffness fraction",
            )
            source_beta = _nonnegative_float(
                component.get("strain_energy_weighted_effective_beta_s"),
                label=f"mode {tracked_mode} {name} effective beta",
            )
            selected_beta = (
                ENGINEERING_SCALAR_BETA_S
                if name in _ENGINEERING_SCALAR_COMPONENTS
                else source_beta
            )
            contribution = math.pi * frequency * energy_fraction * selected_beta
            beta_energy_sum += energy_fraction * selected_beta
            component_rows.append(
                {
                    "component": name,
                    "elastic_stiffness_fraction": energy_fraction,
                    "source_effective_beta_s": source_beta,
                    "selected_effective_beta_s": selected_beta,
                    "projected_damping_ratio_contribution": contribution,
                    "scalar_normalized": name in _ENGINEERING_SCALAR_COMPONENTS,
                }
            )
        zeta = math.pi * frequency * beta_energy_sum
        modes.append(
            {
                "tracked_mode": tracked_mode,
                "frequency_hz": frequency,
                "generalized_modal_mass_kg": mass,
                "generalized_modal_stiffness_n_per_m": stiffness,
                "rayleigh_energy_projected_damping_ratio": zeta,
                "selected_component_projection": component_rows,
                "generalized_rayleigh_damping_n_s_per_m": (
                    2.0 * zeta * 2.0 * math.pi * frequency * mass
                ),
                "qblade_effective_decay_damping_ratio": None,
                "qblade_physical_rayleigh_increment_damping_ratio": None,
                "measured_projected_absolute_error": None,
                "evidence_role": "energy_projection_from_documented_qblade_input_semantics",
            }
        )

    payload: dict[str, object] = {
        "schema": SELECTION_SCHEMA,
        "status": "selected",
        "selection_basis": DOCUMENTED_SELECTION_BASIS,
        "selected_structural_encoding": "C20",
        "parser_semantics": "fixed_20th_numeric_token_is_damping",
        "damping_source": "distributed_station_beta_with_normalized_scalar_fallback",
        "engineering_scalar_beta_s": ENGINEERING_SCALAR_BETA_S,
        "engineering_scalar_components": sorted(_ENGINEERING_SCALAR_COMPONENTS),
        "projection_method": (
            "corrected_c20_distributed_station_beta_plus_normalized_scalar_fallback"
        ),
        "modes": modes,
        "active_binary_verification": {
            "required_for_engineering_selection": False,
            "status": "pending_short_check",
            "scope": "confirm corrected C20 input imports and produces finite structural response",
        },
        "claim_boundary": (
            "These are stiffness-energy projections of an explicit engineering damping "
            "assumption, not measured decay ratios. The archived parser fixes the SUBELEMENTS "
            "field order. Distributed blade/strut station beta is retained, while the raw "
            "0.02/0.03-s component scalars are normalized to 0.0011 s because their direct "
            "Rayleigh projection is not physically plausible for an undamped steel assembly. "
            "A short active-binary check remains verification, not damping identification."
        ),
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def _validate_archived_parser_source(source: str) -> None:
    compact = "".join(source.split())
    required = (
        'FindNumericDataTable("SUBELEMENTS",subStructureStream,20',
        "DAMP=subElements.at(m).at(19);",
        "if(subElements.at(m).size()>20)DRAG=subElements.at(m).at(20);",
        "SetBeamRaleyghDamping(DAMP);",
    )
    missing = [snippet for snippet in required if "".join(snippet.split()) not in compact]
    if missing:
        raise ValueError("Archived QBlade source does not contain the required parser assignments.")


def _validate_c20_substructure(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        table_index = next(i for i, line in enumerate(lines) if line.strip() == "SUBELEMENTS")
    except StopIteration as exc:
        raise ValueError("C20 substructure lacks its SUBELEMENTS table.") from exc
    header_index = table_index + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1
    header = lines[header_index].split()
    if len(header) != 20 or header[-2:] != ["DIA_[m]", "DAMP[-]"]:
        raise ValueError("Selected C20 substructure does not use the canonical 20-field header.")
    row_count = 0
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if stripped == "SUBMEMBERS":
            break
        if not stripped:
            continue
        tokens = stripped.split()
        try:
            float(tokens[0])
        except (ValueError, IndexError):
            continue
        if len(tokens) != 20 or not math.isclose(
            float(tokens[19]), 0.02, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("Selected C20 row must contain 20 fields with DAMP=0.02.")
        row_count += 1
    if row_count == 0:
        raise ValueError("Selected C20 substructure has no numeric rows.")


def select_documented_from_files(
    *,
    campaign_root: Path,
    identification_path: Path,
    archived_strmodel_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Write a source-bound, input-defined Rayleigh selection artifact."""

    from .hvawt_modal_decay_campaign import validate_campaign

    campaign = Path(campaign_root).expanduser().resolve()
    manifest = validate_campaign(campaign)
    cases = manifest.get("cases")
    selected = [
        row
        for row in cases if isinstance(row, Mapping)
        and row.get("variant") == "C20" and row.get("role") == "baseline"
    ] if isinstance(cases, list) else []
    if len(selected) != 1:
        raise ValueError("Sealed campaign must contain one canonical C20 baseline case.")
    case = selected[0]
    case_root = campaign / "cases" / str(case["case_id"])
    modified = case.get("expected_modified_structural_relative_paths")
    if not isinstance(modified, list) or len(modified) != 1:
        raise ValueError("Canonical C20 case must identify one modified substructure file.")
    substructure = case_root / str(modified[0])
    _validate_c20_substructure(substructure)
    contract_identity = manifest.get("source_contract_copy")
    if not isinstance(contract_identity, Mapping) or not isinstance(
        contract_identity.get("path"), str
    ):
        raise ValueError("Sealed campaign lacks its copied source contract.")
    contract = _load_json(
        campaign / "inputs" / str(contract_identity["path"]),
        label="modal-decay source contract",
    )
    scalar_paths = contract.get("rayleigh_scalar_relative_paths")
    if not isinstance(scalar_paths, list) or not scalar_paths or not all(
        isinstance(value, str) and value for value in scalar_paths
    ):
        raise ValueError("Source contract lacks its Rayleigh scalar paths.")

    identification_file = Path(identification_path).expanduser().resolve()
    identification_identity = _file_identity(identification_file)
    if (
        identification_identity["sha256"] != _EXPECTED_IDENTIFICATION_SHA256
        or identification_identity["size_bytes"] != _EXPECTED_IDENTIFICATION_SIZE_BYTES
    ):
        raise ValueError("Mode-identification file does not match the frozen tracked evidence.")
    archived_source = Path(archived_strmodel_path).expanduser().resolve()
    source_text = archived_source.read_text(encoding="utf-8", errors="strict")
    _validate_archived_parser_source(source_text)

    payload = derive_documented_damping_selection(
        _load_json(identification_file, label="mode identification")
    )
    payload["source_case"] = {
        "campaign_root": str(campaign),
        "campaign_fingerprint": manifest["fingerprint"],
        "selected_case_id": case["case_id"],
        "selected_variant": "C20",
        "selected_sim_relative_path": case["sim_relative_path"],
        "selected_substructure_relative_path": modified[0],
        "normalized_scalar_relative_paths": list(scalar_paths),
    }
    payload["inputs"] = {
        "campaign_manifest": _file_identity(campaign / "campaign_manifest.json"),
        "selected_substructure": _file_identity(substructure),
        "mode_identification": identification_identity,
        "archived_qblade_strmodel_cpp": _file_identity(archived_source),
        "qblade_2_0_9_7_documentation": [
            "https://docs.qblade.org/src/user/turbine/substructure.html#substructure-definition",
            "https://docs.qblade.org/src/user/turbine/structure.html#structural-blade-and-strut-definition",
        ],
    }
    payload.pop("fingerprint", None)
    payload["fingerprint"] = _fingerprint(payload)
    destination = Path(output_path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return payload


def select_from_files(
    *,
    analysis_path: Path,
    identification_path: Path,
    output_path: Path,
) -> dict[str, object]:
    analysis_file = Path(analysis_path).expanduser().resolve()
    identification_file = Path(identification_path).expanduser().resolve()
    identification_identity = _file_identity(identification_file)
    if (
        identification_identity["sha256"] != _EXPECTED_IDENTIFICATION_SHA256
        or identification_identity["size_bytes"] != _EXPECTED_IDENTIFICATION_SIZE_BYTES
    ):
        raise ValueError("Mode-identification file does not match the frozen tracked evidence.")
    payload = derive_damping_selection(
        _load_json(analysis_file, label="Rayleigh analysis"),
        _load_json(identification_file, label="mode identification"),
    )
    payload["inputs"] = {
        "rayleigh_analysis": _file_identity(analysis_file),
        "mode_identification": identification_identity,
    }
    payload.pop("fingerprint", None)
    payload["fingerprint"] = _fingerprint(payload)
    destination = Path(output_path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind a completed QBlade Rayleigh discriminator to two-mode damping evidence."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--analysis", type=Path)
    source.add_argument("--documented-campaign-root", type=Path)
    parser.add_argument("--identification", type=Path, required=True)
    parser.add_argument("--archived-strmodel", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.documented_campaign_root is not None:
        if args.archived_strmodel is None:
            raise ValueError("--archived-strmodel is required for documented input selection.")
        payload = select_documented_from_files(
            campaign_root=args.documented_campaign_root,
            identification_path=args.identification,
            archived_strmodel_path=args.archived_strmodel,
            output_path=args.output,
        )
    else:
        if args.archived_strmodel is not None:
            raise ValueError("--archived-strmodel is only valid with documented selection.")
        payload = select_from_files(
            analysis_path=args.analysis,
            identification_path=args.identification,
            output_path=args.output,
        )
    print(json.dumps({"status": payload["status"], "fingerprint": payload["fingerprint"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
