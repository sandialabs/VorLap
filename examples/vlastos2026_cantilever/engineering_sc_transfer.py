#!/usr/bin/env python3
"""Frozen Belloli-informed Scruton transfer for the Vlastos VIV screen.

The underlying force implementation remains response-model schema v2.  This
module changes only the case-level closure parameters before QBlade starts:
the activation window follows the frozen Belloli peak-location relation, the
saturation amplitude follows the frozen Belloli peak-amplitude relation, and
``H1`` remains unchanged.  Consequently legacy schema-v1 and ambient
schema-v2 runtime behavior are not modified.

The Sc=2.94 and Sc=7.02 hGAST targets were available when the selected member
of a 20-form menu was chosen.  They are therefore transfer-tuning evidence,
not independent validation.  The generator creates inputs and a dry run plan;
this module intentionally provides no execution command.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import numbers
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

try:  # Package imports in tests and ``python -m`` use.
    from examples.vlastos2026_cantilever import response_benchmark
    from examples.vlastos2026_cantilever import stationary_campaign_protocol
    from vorlap import qblade_runtime
except ImportError:  # Direct execution from this example directory.
    import response_benchmark  # type: ignore[no-redef]
    import stationary_campaign_protocol  # type: ignore[no-redef]
    from vorlap import qblade_runtime  # type: ignore[no-redef]


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFINITION_PATH = HERE / "reference_data" / "belloli_sc_transfer_v1.json"

DEFINITION_SCHEMA = "vlastos2026-belloli-sc-transfer-definition-v1"
CAMPAIGN_SCHEMA = "vlastos2026-belloli-sc-transfer-campaign-v1"
ANALYTICAL_GATE_SCHEMA = "vlastos2026-belloli-sc-transfer-analytical-gate-v1"
PLAN_SCHEMA = "vlastos2026-belloli-sc-transfer-dry-run-plan-v1"
SELECTED_FORM_ID = "shift_1__h1_none__asat_1"
BASE_RESPONSE_SCHEMA_VERSION = 2
BASE_AERODYNAMIC_SPEED_BASIS = "declared_ambient_crossflow"
EXPECTED_DEFINITION_FINGERPRINT_SHA256 = (
    "49240a27fbc91648bd89c091d0348ef13a9dbec01fc0162d2bec58be0c31c6fa"
)

CALIBRATION_SC = 5.18
SCREEN_SCRUTON_NUMBERS = (2.94, 5.18, 7.02)
TRANSFER_TUNING_SCRUTON_NUMBERS = (2.94, 7.02)
CASE_GRID = {
    2.94: (6.25, 6.80),
    7.02: (6.20, 6.25),
}
TIMESTEP_LEVELS = ("dt", "dt_half")
EXPECTED_CASE_COUNT = 8

_TRANSFER_PARAMETER_NAMES = (
    "aerodynamic_damping_gain_h1",
    "saturation_amplitude_ratio",
    "reduced_velocity_low",
    "reduced_velocity_peak",
    "reduced_velocity_high",
)
_RUNTIME_PARAMETER_NAMES = _TRANSFER_PARAMETER_NAMES + (
    "rms_time_constant_cycles",
    "bandpass_damping_ratio",
    "max_abs_gain",
    "max_abs_nodal_force_n",
)


def canonical_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 for finite JSON-compatible data."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(raw: Any, *, label: str) -> PurePosixPath:
    text = str(raw)
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative POSIX path.")
    return path


def _write_json(path: Path, value: Any, *, replace: bool = False) -> Path:
    destination = Path(path)
    if destination.exists() and not replace:
        raise FileExistsError(f"Refusing to replace {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _finite_scruton_number(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError("scruton_number must be a real numeric value, not boolean.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("scruton_number must be finite.")
    return result


def load_transfer_definition(path: Path | None = None) -> dict[str, Any]:
    """Load and fully validate the frozen transfer definition."""

    source = DEFINITION_PATH if path is None else Path(path)
    definition = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(definition, dict):
        raise ValueError("Transfer definition must be a JSON object.")
    if definition.get("schema") != DEFINITION_SCHEMA:
        raise ValueError("Unsupported Scruton-transfer definition schema.")
    if definition.get("selected_form_id") != SELECTED_FORM_ID:
        raise ValueError("Transfer definition no longer selects the frozen form.")
    if definition.get("response_model_schema_version") != BASE_RESPONSE_SCHEMA_VERSION:
        raise ValueError("Transfer definition must retain response-model schema v2.")
    if definition.get("aerodynamic_speed_basis") != BASE_AERODYNAMIC_SPEED_BASIS:
        raise ValueError("Transfer definition must retain declared ambient crossflow.")

    supplied_fingerprint = definition.get("closure_fingerprint_sha256")
    payload = dict(definition)
    payload.pop("closure_fingerprint_sha256", None)
    observed_fingerprint = canonical_sha256(payload)
    if (
        supplied_fingerprint != EXPECTED_DEFINITION_FINGERPRINT_SHA256
        or observed_fingerprint != EXPECTED_DEFINITION_FINGERPRINT_SHA256
    ):
        raise ValueError("Scruton-transfer definition fingerprint changed.")

    domain = definition.get("application_scruton_domain")
    if not isinstance(domain, Mapping):
        raise ValueError("Transfer definition lacks an application domain.")
    if (
        float(domain.get("minimum", math.nan)) != min(SCREEN_SCRUTON_NUMBERS)
        or float(domain.get("maximum", math.nan)) != max(SCREEN_SCRUTON_NUMBERS)
        or domain.get("outside_domain_policy") != "raise"
    ):
        raise ValueError("Scruton-transfer application domain changed.")
    if float(definition.get("calibration_scruton_number", math.nan)) != CALIBRATION_SC:
        raise ValueError("Scruton-transfer calibration case changed.")

    base = definition.get("base_frozen_parameters")
    if not isinstance(base, Mapping) or set(base) != set(_RUNTIME_PARAMETER_NAMES):
        raise ValueError("Transfer definition has an incomplete base parameter set.")
    for name in _RUNTIME_PARAMETER_NAMES:
        value = float(base[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Base transfer parameter {name} must be positive and finite.")
    if (
        definition.get("base_coefficient_fingerprint_sha256")
        != response_benchmark.AMBIENT_V2_FROZEN_COEFFICIENT_FINGERPRINT_SHA256
    ):
        raise ValueError("Transfer definition is not bound to the frozen v2 coefficients.")

    source_identities = definition.get("source_identities")
    if not isinstance(source_identities, Mapping):
        raise ValueError("Transfer definition lacks source identities.")
    for filename in (
        "sc5p18_figure9_digitized.csv",
        "all_sc_figure10_peak_digitized.csv",
        "vlastos2026_model_parameters.csv",
    ):
        identity = source_identities.get(filename)
        if not isinstance(identity, Mapping):
            raise ValueError(f"Transfer definition lacks identity for {filename}.")
        reference = HERE / "reference_data" / filename
        if (
            sha256_file(reference) != identity.get("sha256")
            or reference.stat().st_size != int(identity.get("bytes", -1))
        ):
            raise ValueError(f"Transfer source identity changed: {filename}.")

    disclosure = definition.get("selection_disclosure")
    if (
        not isinstance(disclosure, Mapping)
        or disclosure.get("targets_available_before_selection") is not True
        or disclosure.get("independent_validation_claim_allowed") is not False
        or int(disclosure.get("candidate_menu_count", -1)) != 20
    ):
        raise ValueError("Transfer selection disclosure is incomplete.")
    return definition


def _transfer_trends(
    scruton_number: float,
    definition: Mapping[str, Any],
) -> tuple[float, float]:
    equations = definition["transfer_equations"]
    shift = equations["activation_shift"]
    amplitude = equations["saturation_amplitude_scaling"]
    intercept_u = float(shift["belloli_umax_intercept"])
    slope_u = float(shift["belloli_umax_slope_per_scruton"])
    intercept_a = float(amplitude["belloli_log_intercept"])
    slope_a = float(amplitude["belloli_log_slope"])
    u_at_sc = intercept_u + slope_u * scruton_number
    u_at_calibration = intercept_u + slope_u * CALIBRATION_SC
    a_at_sc = intercept_a + slope_a * math.log(scruton_number)
    a_at_calibration = intercept_a + slope_a * math.log(CALIBRATION_SC)
    if a_at_sc <= 0.0 or a_at_calibration <= 0.0:
        raise ValueError("Belloli amplitude trend is non-positive in the requested domain.")
    return u_at_sc - u_at_calibration, a_at_sc / a_at_calibration


def effective_frozen_parameters(
    scruton_number: Any,
    definition: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return the globally frozen parameter set for one published Sc label.

    The published Scruton label selects the Belloli scaling.  The existing
    QBlade-implied Scruton number remains the dynamic Sc in the analytical
    energy balance and is not retuned.
    """

    sc = _finite_scruton_number(scruton_number)
    frozen = load_transfer_definition() if definition is None else dict(definition)
    domain = frozen["application_scruton_domain"]
    minimum = float(domain["minimum"])
    maximum = float(domain["maximum"])
    if sc < minimum or sc > maximum:
        raise ValueError(
            f"scruton_number {sc:g} is outside the frozen [{minimum:g}, {maximum:g}] domain."
        )
    base = {name: float(frozen["base_frozen_parameters"][name]) for name in _RUNTIME_PARAMETER_NAMES}
    if sc == CALIBRATION_SC:
        # An explicit branch guarantees bit-identical base floats at Sc=5.18.
        return base
    activation_shift, saturation_scale = _transfer_trends(sc, frozen)
    base["saturation_amplitude_ratio"] *= saturation_scale
    for name in (
        "reduced_velocity_low",
        "reduced_velocity_peak",
        "reduced_velocity_high",
    ):
        base[name] += activation_shift
    return base


def effective_parameter_record(
    scruton_number: Any,
    definition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sc = _finite_scruton_number(scruton_number)
    frozen = load_transfer_definition() if definition is None else dict(definition)
    parameters = effective_frozen_parameters(sc, frozen)
    activation_shift, saturation_scale = _transfer_trends(sc, frozen)
    if sc == CALIBRATION_SC:
        activation_shift = 0.0
        saturation_scale = 1.0
    record = {
        "schema": "vlastos2026-belloli-sc-transfer-effective-parameters-v1",
        "transfer_definition_fingerprint_sha256": frozen[
            "closure_fingerprint_sha256"
        ],
        "published_scruton_number": sc,
        "activation_shift_u_star": activation_shift,
        "saturation_amplitude_scale": saturation_scale,
        "h1_scale": 1.0,
        "parameters": parameters,
    }
    record["effective_parameter_fingerprint_sha256"] = canonical_sha256(record)
    return record


def _reference_peaks() -> dict[tuple[float, str], tuple[float, float]]:
    return response_benchmark._load_figure10_peaks(
        HERE / "reference_data" / "all_sc_figure10_peak_digitized.csv"
    )


def analytical_dense_gate(
    definition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen transfer on the declared dense U* grid."""

    frozen = load_transfer_definition() if definition is None else dict(definition)
    acceptance = frozen["engineering_acceptance"]
    grid_low = float(acceptance["dense_grid_low_u_star"])
    grid_high = float(acceptance["dense_grid_high_u_star"])
    grid_step = float(acceptance["dense_grid_step_u_star"])
    dense_u = np.arange(grid_low, grid_high + 0.5 * grid_step, grid_step)
    structures = response_benchmark.load_structural_cases()
    references = _reference_peaks()
    base = {
        name: float(frozen["base_frozen_parameters"][name])
        for name in _RUNTIME_PARAMETER_NAMES
    }
    rows: list[dict[str, Any]] = []
    curve_records: dict[float, dict[str, np.ndarray]] = {}
    for sc in SCREEN_SCRUTON_NUMBERS:
        structure = structures[sc]
        dynamic_sc = response_benchmark.derived_scruton_number(structure)
        transfer = effective_frozen_parameters(sc, frozen)
        curves: dict[str, np.ndarray] = {}
        for model_name, parameters in (("original_schema_v2", base), ("belloli_sc_transfer", transfer)):
            curve = response_benchmark.analytical_equilibrium_amplitude_ratio(
                dynamic_sc,
                dense_u,
                aerodynamic_damping_gain_h1=parameters[
                    "aerodynamic_damping_gain_h1"
                ],
                saturation_amplitude_ratio=parameters[
                    "saturation_amplitude_ratio"
                ],
                reduced_velocity_low=parameters["reduced_velocity_low"],
                reduced_velocity_peak=parameters["reduced_velocity_peak"],
                reduced_velocity_high=parameters["reduced_velocity_high"],
            )
            curves[model_name] = curve
            peak_index = int(np.argmax(curve))
            predicted_peak = float(curve[peak_index])
            predicted_u = float(dense_u[peak_index])
            for reference_series in (
                "hgast_hartlen_currie",
                "experiment_lupi2021",
            ):
                target, uncertainty = references[(sc, reference_series)]
                if uncertainty != float(
                    acceptance["digitization_uncertainty_over_diameter"]
                ):
                    raise ValueError("Reference digitization uncertainty changed.")
                amplitude_error = abs(predicted_peak - target)
                amplitude_tolerance = (
                    float(acceptance["peak_relative_fraction"]) * target
                    + uncertainty
                )
                location_error = abs(
                    predicted_u - structure.u_star_max_figure_digitized
                )
                amplitude_pass = amplitude_error <= amplitude_tolerance
                location_pass = location_error <= float(
                    acceptance["peak_location_tolerance_u_star"]
                )
                rows.append(
                    {
                        "published_scruton_number": sc,
                        "qblade_implied_scruton_number": dynamic_sc,
                        "evidence_role": (
                            "calibration"
                            if sc == CALIBRATION_SC
                            else "transfer_tuning_engineering_screen"
                        ),
                        "model": model_name,
                        "reference_series": reference_series,
                        "reference_peak_amplitude_over_diameter": target,
                        "reference_digitization_uncertainty_over_diameter": uncertainty,
                        "reference_peak_u_star": structure.u_star_max_figure_digitized,
                        "predicted_peak_amplitude_over_diameter": predicted_peak,
                        "predicted_peak_u_star": predicted_u,
                        "peak_amplitude_absolute_error_over_diameter": amplitude_error,
                        "peak_amplitude_relative_error": amplitude_error / target,
                        "peak_amplitude_tolerance_over_diameter": amplitude_tolerance,
                        "peak_amplitude_pass": amplitude_pass,
                        "peak_location_absolute_error_u_star": location_error,
                        "peak_location_tolerance_u_star": float(
                            acceptance["peak_location_tolerance_u_star"]
                        ),
                        "peak_location_pass": location_pass,
                        "engineering_joint_pass": amplitude_pass and location_pass,
                    }
                )
        curve_records[sc] = {"u_star": dense_u, **curves}

    transfer_rows = [row for row in rows if row["model"] == "belloli_sc_transfer"]
    sc5_base = effective_frozen_parameters(CALIBRATION_SC, frozen)
    maximum_identity_difference = max(
        abs(sc5_base[name] - base[name]) for name in _RUNTIME_PARAMETER_NAMES
    )
    gate = {
        "schema": ANALYTICAL_GATE_SCHEMA,
        "status": (
            "pass"
            if all(bool(row["engineering_joint_pass"]) for row in transfer_rows)
            else "fail"
        ),
        "pass": all(bool(row["engineering_joint_pass"]) for row in transfer_rows),
        "scientific_scope": frozen["scientific_scope"],
        "independent_validation": False,
        "transfer_definition_fingerprint_sha256": frozen[
            "closure_fingerprint_sha256"
        ],
        "selected_form_id": frozen["selected_form_id"],
        "dense_grid": {
            "low_u_star": grid_low,
            "high_u_star": grid_high,
            "step_u_star": grid_step,
            "points": int(dense_u.size),
        },
        "acceptance": copy.deepcopy(acceptance),
        "sc5_exact_identity": maximum_identity_difference == 0.0,
        "sc5_maximum_parameter_absolute_difference": maximum_identity_difference,
        "comparisons": rows,
    }
    gate["assessment_fingerprint_sha256"] = canonical_sha256(gate)
    gate["_curve_records"] = curve_records
    return gate


def _portable_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    portable = dict(gate)
    portable.pop("_curve_records", None)
    return portable


def _write_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "published_scruton_number",
        "qblade_implied_scruton_number",
        "evidence_role",
        "model",
        "reference_series",
        "reference_peak_amplitude_over_diameter",
        "reference_peak_u_star",
        "predicted_peak_amplitude_over_diameter",
        "predicted_peak_u_star",
        "peak_amplitude_absolute_error_over_diameter",
        "peak_amplitude_relative_error",
        "peak_amplitude_tolerance_over_diameter",
        "peak_amplitude_pass",
        "peak_location_absolute_error_u_star",
        "peak_location_tolerance_u_star",
        "peak_location_pass",
        "engineering_joint_pass",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})


def _write_latex_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    transfer_rows = [
        row for row in rows if row["model"] == "belloli_sc_transfer"
    ]
    labels = {
        "hgast_hartlen_currie": "hGAST",
        "experiment_lupi2021": "Lupi experiment",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Belloli-informed analytical transfer screen. Sc=2.94 and 7.02 were used to select the transfer form and are tuning evidence, not independent validation. Errors are evaluated against $0.30A_{\rm target}+0.01D$ and $|\Delta U^*_{\rm peak}|\leq0.25$.}",
        r"\label{tab:belloli-sc-transfer-screen}",
        r"\begin{tabular}{ccrrrrc}",
        r"\hline",
        r"Sc & Reference & $A_{\rm target}/D$ & $A_{\rm pred}/D$ & Error (\%) & $|\Delta U^*|$ & Screen \\",
        r"\hline",
    ]
    for row in transfer_rows:
        lines.append(
            f"{row['published_scruton_number']:.2f} & "
            f"{labels[str(row['reference_series'])]} & "
            f"{row['reference_peak_amplitude_over_diameter']:.3f} & "
            f"{row['predicted_peak_amplitude_over_diameter']:.3f} & "
            f"{100.0 * row['peak_amplitude_relative_error']:.1f} & "
            f"{row['peak_location_absolute_error_u_star']:.3f} & "
            f"{'pass' if row['engineering_joint_pass'] else 'fail'} \\\\"
        )
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw_analytical_figure(path: Path, gate: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "transfer": "#0072B2",
        "original": "#7A7A7A",
        "hgast": "#D55E00",
        "lupi": "#009E73",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.25), sharex=True, sharey=True)
    comparisons = list(gate["comparisons"])
    curves = gate["_curve_records"]
    for axis, sc in zip(axes, SCREEN_SCRUTON_NUMBERS):
        record = curves[sc]
        axis.plot(
            record["u_star"],
            record["original_schema_v2"],
            color=colors["original"],
            linewidth=1.4,
            linestyle="--",
            label="VorLap analytical: original closure",
        )
        axis.plot(
            record["u_star"],
            record["belloli_sc_transfer"],
            color=colors["transfer"],
            linewidth=2.0,
            label="VorLap analytical: selected Sc transfer",
        )
        transfer_rows = [
            row
            for row in comparisons
            if row["published_scruton_number"] == sc
            and row["model"] == "belloli_sc_transfer"
        ]
        for row in transfer_rows:
            hgast = row["reference_series"] == "hgast_hartlen_currie"
            axis.errorbar(
                row["reference_peak_u_star"],
                row["reference_peak_amplitude_over_diameter"],
                yerr=row["reference_digitization_uncertainty_over_diameter"],
                color=colors["hgast"] if hgast else colors["lupi"],
                marker="D" if hgast else "o",
                markersize=5.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                capsize=2.2,
                linestyle="none",
                label=(
                    "hGAST numerical peak (digitized)"
                    if hgast
                    else "Lupi experimental peak (digitized)"
                ),
                zorder=5,
            )
        predicted = transfer_rows[0]
        axis.plot(
            predicted["predicted_peak_u_star"],
            predicted["predicted_peak_amplitude_over_diameter"],
            marker="s",
            color=colors["transfer"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=5.2,
            linestyle="none",
            zorder=6,
        )
        role = "calibration" if sc == CALIBRATION_SC else "transfer tuning"
        axis.set_title(f"Sc = {sc:.2f} ({role})", fontsize=8.3, fontweight="bold")
        axis.set_xlim(4.7, 7.8)
        axis.set_ylim(0.0, 0.62)
        axis.set_xticks((5.0, 6.0, 7.0))
        axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7.5)
        axis.set_xlabel(r"Reduced velocity, $U^*$", fontsize=8)
    axes[0].set_ylabel(r"Peak response, $A/D$", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
        fontsize=7.1,
    )
    fig.text(
        0.5,
        0.012,
        "Engineering transfer screen; Sc=2.94 and 7.02 informed form selection and are not independent validation.",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#333333",
    )
    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.19, top=0.75, wspace=0.10)
    fig.savefig(
        path,
        dpi=300,
        metadata={"Creator": "VorLap engineering_sc_transfer.py", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def render_analytical_artifacts(output: Path) -> dict[str, Path]:
    """Write the analytical gate, source table, LaTeX table, and PDF figure."""

    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    gate = analytical_dense_gate()
    if gate["pass"] is not True:
        raise RuntimeError("Frozen engineering transfer no longer passes its analytical screen.")
    portable_gate = _portable_gate(gate)
    gate_path = _write_json(
        destination / "analytical_gate.json", portable_gate, replace=True
    )
    csv_path = destination / "analytical_comparison.csv"
    _write_comparison_csv(csv_path, portable_gate["comparisons"])
    table_path = destination / "engineering_sc_transfer_table.tex"
    _write_latex_table(table_path, portable_gate["comparisons"])
    figure_path = destination / "engineering_sc_transfer_comparison.pdf"
    _draw_analytical_figure(figure_path, gate)
    return {
        "gate": gate_path,
        "comparison_csv": csv_path,
        "latex_table": table_path,
        "figure": figure_path,
    }


def _manifest_file_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        if path.name == "campaign_manifest.json":
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def _expected_case_keys() -> tuple[tuple[float, float, str], ...]:
    return tuple(
        (sc, u_star, level)
        for sc in TRANSFER_TUNING_SCRUTON_NUMBERS
        for u_star in CASE_GRID[sc]
        for level in TIMESTEP_LEVELS
    )


def _source_identities() -> dict[str, dict[str, Any]]:
    relative_paths = (
        "examples/vlastos2026_cantilever/engineering_sc_transfer.py",
        "examples/vlastos2026_cantilever/engineering_sc_transfer_run.py",
        "examples/vlastos2026_cantilever/response_benchmark.py",
        "examples/vlastos2026_cantilever/stationary_campaign_protocol.py",
        "vorlap/qblade_response_model.py",
        "vorlap/qblade_runtime.py",
        "run_headless/run_qblade_sim.py",
    )
    return {
        relative: {
            "path": relative,
            "sha256": sha256_file(REPO_ROOT / relative),
            "bytes": (REPO_ROOT / relative).stat().st_size,
        }
        for relative in relative_paths
    }


def _patch_case_for_transfer(
    campaign_root: Path,
    entry: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(entry))
    sc = float(result["scruton_number"])
    effective = effective_parameter_record(sc, definition)
    response_config = copy.deepcopy(dict(result["response_model_config"]))
    for name in _TRANSFER_PARAMETER_NAMES:
        response_config[name] = effective["parameters"][name]
    response_config["rms_time_constant_s"] = (
        effective["parameters"]["rms_time_constant_cycles"]
        / float(response_config["natural_frequency_hz"])
    )
    response_config["bandpass_damping_ratio"] = effective["parameters"][
        "bandpass_damping_ratio"
    ]
    response_config["max_abs_gain"] = effective["parameters"]["max_abs_gain"]
    response_config["max_abs_nodal_force_n"] = effective["parameters"][
        "max_abs_nodal_force_n"
    ]

    parameter_relative = _safe_relative_path(
        result["parameter_file"], label="case parameter_file"
    )
    parameter_path = campaign_root.joinpath(*parameter_relative.parts)
    runtime_config = json.loads(parameter_path.read_text(encoding="utf-8"))
    runtime_config["response_model"] = response_config
    parameter_path.write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    model = qblade_runtime._response_model_from_config(
        response_config, response_config["node_ids"]
    )
    if model is None:
        raise RuntimeError("Generated response model was unexpectedly disabled.")

    case_relative = _safe_relative_path(
        Path(str(result["sim_path"])).parent.as_posix(), label="case directory"
    )
    case_root = campaign_root.joinpath(*case_relative.parts)
    sources = sorted(case_root.rglob("CYLINDER.h5"))
    if len(sources) != 1:
        raise RuntimeError(f"Expected one CYLINDER.h5 for {result['case_id']}.")
    with h5py.File(sources[0], "r+") as source:
        source.attrs["response_closure"] = (
            "schema-v2 modal negative damping with frozen Belloli Scruton transfer"
        )
        source.attrs["base_coefficient_fingerprint_sha256"] = definition[
            "base_coefficient_fingerprint_sha256"
        ]
        source.attrs["sc_transfer_definition_fingerprint_sha256"] = definition[
            "closure_fingerprint_sha256"
        ]
        source.attrs["effective_parameter_fingerprint_sha256"] = effective[
            "effective_parameter_fingerprint_sha256"
        ]
        source.attrs["published_scruton_number"] = sc

    result["role"] = "transfer_tuning_engineering_screen"
    result["response_model_config"] = response_config
    result.pop("coefficient_fingerprint_sha256", None)
    result["base_coefficient_fingerprint_sha256"] = definition[
        "base_coefficient_fingerprint_sha256"
    ]
    result["transfer_definition_fingerprint_sha256"] = definition[
        "closure_fingerprint_sha256"
    ]
    result["effective_parameter_record"] = effective
    result["runtime_response_config_fingerprint_sha256"] = model.config_fingerprint
    return result


def generate_transfer_campaign(output: Path) -> Path:
    """Generate exactly eight immutable schema-v2 transfer-tuning inputs."""

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing campaign: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    definition = load_transfer_definition()
    gate = analytical_dense_gate(definition)
    if gate["pass"] is not True:
        raise RuntimeError("Refusing campaign generation after analytical gate failure.")

    source_velocities = tuple(sorted({value for values in CASE_GRID.values() for value in values}))
    with tempfile.TemporaryDirectory(
        prefix="vlastos_sc_transfer_stage_", dir=destination.parent
    ) as temporary:
        base_root = Path(temporary) / "base_campaign"
        base_manifest_path = response_benchmark.generate_campaign(
            base_root,
            reduced_velocities=source_velocities,
            timestep_s=stationary_campaign_protocol.COARSE_TIMESTEP_S,
            include_dt_halving=True,
            duration_s=None,
            selected_scruton_numbers=SCREEN_SCRUTON_NUMBERS,
            response_schema_version=2,
        )
        base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
        by_key = {
            (
                float(entry["scruton_number"]),
                float(entry["reduced_velocity"]),
                str(entry["timestep_level"]),
            ): entry
            for entry in base_manifest["cases"]
        }
        missing = set(_expected_case_keys()).difference(by_key)
        if missing:
            raise RuntimeError(f"Base schema-v2 generator omitted selected cases: {sorted(missing)}")

        destination.mkdir()
        shutil.copytree(base_root / "reference_data", destination / "reference_data")
        for filename in (
            "cantilever_results.filter",
            "cantilever_geometry_and_mode.csv",
            "frozen_response_coefficients.json",
        ):
            target_name = (
                "base_response_coefficients.json"
                if filename == "frozen_response_coefficients.json"
                else filename
            )
            shutil.copy2(base_root / filename, destination / target_name)
        _write_json(destination / "sc_transfer_definition.json", definition)
        _write_json(destination / "analytical_gate.json", _portable_gate(gate))

        selected_cases = []
        for key in _expected_case_keys():
            source_entry = by_key[key]
            source_case_relative = _safe_relative_path(
                Path(str(source_entry["sim_path"])).parent.as_posix(),
                label="base case directory",
            )
            source_case_root = base_root.joinpath(*source_case_relative.parts)
            target_case_root = destination.joinpath(*source_case_relative.parts)
            target_case_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_case_root, target_case_root)
            selected_cases.append(
                _patch_case_for_transfer(destination, source_entry, definition)
            )

    manifest: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "scope": (
            "eight-case Sc=2.94/7.02 Belloli-transfer tuning and engineering "
            "screen; not independent validation and not a QBlade result"
        ),
        "execution_state": "inputs_only_not_run",
        "case_count": len(selected_cases),
        "exact_case_grid": [
            {
                "published_scruton_number": sc,
                "reduced_velocity": u_star,
                "timestep_level": level,
            }
            for sc, u_star, level in _expected_case_keys()
        ],
        "base_response_model": {
            "schema_version": BASE_RESPONSE_SCHEMA_VERSION,
            "aerodynamic_speed_basis": BASE_AERODYNAMIC_SPEED_BASIS,
            "coefficient_fingerprint_sha256": definition[
                "base_coefficient_fingerprint_sha256"
            ],
            "runtime_code_modified_for_transfer": False,
        },
        "sc_transfer": {
            "definition_path": "sc_transfer_definition.json",
            "definition_fingerprint_sha256": definition[
                "closure_fingerprint_sha256"
            ],
            "selected_form_id": SELECTED_FORM_ID,
            "outside_domain_policy": "raise",
            "sc5_exact_identity": bool(gate["sc5_exact_identity"]),
            "targets_used_for_form_selection": [2.94, 7.02],
            "independent_validation": False,
        },
        "analytical_gate": {
            "path": "analytical_gate.json",
            "assessment_fingerprint_sha256": gate[
                "assessment_fingerprint_sha256"
            ],
            "status": gate["status"],
        },
        "numerics": {
            "timestep_s": stationary_campaign_protocol.COARSE_TIMESTEP_S,
            "timestep_half_s": stationary_campaign_protocol.FINE_TIMESTEP_S,
            "same_endpoint_dt_pair_required": True,
            "threads_must_be_less_than": 8,
        },
        "runtime_contract": copy.deepcopy(base_manifest["runtime_contract"]),
        "implementation_sources": _source_identities(),
        "cases": selected_cases,
    }
    manifest["files"] = _manifest_file_rows(destination)
    manifest["manifest_fingerprint_sha256"] = canonical_sha256(manifest)
    manifest_path = _write_json(destination / "campaign_manifest.json", manifest)
    validate_transfer_campaign(manifest_path)
    return manifest_path


def _verify_manifest_files(root: Path, rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("Campaign file manifest is empty.")
    verified: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Campaign file identity must be an object.")
        relative = _safe_relative_path(row.get("path"), label="manifest file path")
        relative_text = relative.as_posix()
        if relative_text in verified:
            raise ValueError(f"Duplicate manifest file identity: {relative_text}")
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Manifest file is missing or symlinked: {relative_text}")
        if path.stat().st_size != int(row.get("bytes", -1)):
            raise ValueError(f"Manifest byte count changed: {relative_text}")
        if sha256_file(path) != row.get("sha256"):
            raise ValueError(f"Manifest file hash changed: {relative_text}")
        verified[relative_text] = dict(row)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "campaign_manifest.json"
    }
    if observed != set(verified):
        raise ValueError("Campaign file set differs from its immutable manifest.")
    return verified


def validate_transfer_campaign(manifest_path: Path) -> dict[str, Any]:
    """Fail closed on any drift in an eight-case transfer campaign."""

    source = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError("Unsupported Scruton-transfer campaign schema.")
    supplied_fingerprint = manifest.get("manifest_fingerprint_sha256")
    payload = dict(manifest)
    payload.pop("manifest_fingerprint_sha256", None)
    if supplied_fingerprint != canonical_sha256(payload):
        raise ValueError("Scruton-transfer campaign fingerprint changed.")
    if manifest.get("execution_state") != "inputs_only_not_run":
        raise ValueError("Campaign execution state is not the frozen inputs-only state.")
    if int(manifest.get("case_count", -1)) != EXPECTED_CASE_COUNT:
        raise ValueError("Scruton-transfer campaign must contain exactly eight cases.")

    root = source.parent
    verified_files = _verify_manifest_files(root, manifest.get("files"))
    stationary_campaign_protocol.validate_ambient_v2_runtime_contract(
        root,
        manifest,
        verified_files,
    )
    definition_relative = _safe_relative_path(
        manifest["sc_transfer"]["definition_path"], label="transfer definition path"
    )
    campaign_definition = load_transfer_definition(
        root.joinpath(*definition_relative.parts)
    )
    tracked_definition = load_transfer_definition()
    if campaign_definition != tracked_definition:
        raise ValueError("Campaign transfer definition differs from the tracked source.")
    if manifest["sc_transfer"]["definition_fingerprint_sha256"] != campaign_definition[
        "closure_fingerprint_sha256"
    ]:
        raise ValueError("Campaign transfer fingerprint is inconsistent.")
    if manifest["sc_transfer"].get("independent_validation") is not False:
        raise ValueError("Campaign must not claim independent validation.")

    gate_relative = _safe_relative_path(
        manifest["analytical_gate"]["path"], label="analytical gate path"
    )
    stored_gate = json.loads(
        root.joinpath(*gate_relative.parts).read_text(encoding="utf-8")
    )
    expected_gate = _portable_gate(analytical_dense_gate(campaign_definition))
    if stored_gate != expected_gate or stored_gate.get("pass") is not True:
        raise ValueError("Campaign analytical gate is missing, changed, or failing.")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError("Campaign case list must contain exactly eight objects.")
    by_key: dict[tuple[float, float, str], Mapping[str, Any]] = {}
    pair_rows: dict[tuple[float, float], dict[str, Mapping[str, Any]]] = {}
    for entry in cases:
        if not isinstance(entry, Mapping):
            raise ValueError("Campaign case entry must be an object.")
        key = (
            float(entry["scruton_number"]),
            float(entry["reduced_velocity"]),
            str(entry["timestep_level"]),
        )
        if key in by_key:
            raise ValueError(f"Duplicate transfer campaign case: {key}")
        by_key[key] = entry
        pair_rows.setdefault(key[:2], {})[key[2]] = entry
        if entry.get("role") != "transfer_tuning_engineering_screen":
            raise ValueError("Transfer cases must retain their tuning/screen role.")
        expected_record = effective_parameter_record(key[0], campaign_definition)
        if entry.get("effective_parameter_record") != expected_record:
            raise ValueError(f"Effective parameter record changed for {entry['case_id']}.")
        response_config = entry.get("response_model_config")
        if not isinstance(response_config, Mapping):
            raise ValueError(f"Case {entry['case_id']} lacks a response config.")
        if (
            response_config.get("schema_version") != BASE_RESPONSE_SCHEMA_VERSION
            or response_config.get("aerodynamic_speed_basis")
            != BASE_AERODYNAMIC_SPEED_BASIS
        ):
            raise ValueError("Transfer case no longer uses ambient schema v2.")
        if response_config.get("enabled") is not True:
            raise ValueError("Transfer response model must be enabled.")
        for name in _TRANSFER_PARAMETER_NAMES:
            if float(response_config[name]) != expected_record["parameters"][name]:
                raise ValueError(f"Transfer parameter {name} changed in {entry['case_id']}.")
        parameter_relative = _safe_relative_path(
            entry["parameter_file"], label="case parameter_file"
        )
        runtime_config = json.loads(
            root.joinpath(*parameter_relative.parts).read_text(encoding="utf-8")
        )
        if runtime_config.get("response_model") != response_config:
            raise ValueError("Manifest and runtime response configurations differ.")
        if runtime_config.get("force_scale") != 0.0 or runtime_config.get(
            "include_mean_load"
        ) is not False:
            raise ValueError("Transfer campaign must retain response-only forcing.")
        model = qblade_runtime._response_model_from_config(
            response_config, response_config["node_ids"]
        )
        if model is None or model.config_fingerprint != entry.get(
            "runtime_response_config_fingerprint_sha256"
        ):
            raise ValueError("Runtime response configuration fingerprint changed.")
        expected_speed = (
            key[1]
            * float(entry["structure"]["natural_frequency_hz"])
            * float(response_config["reference_length_m"])
        )
        if not math.isclose(
            float(entry["mean_inflow_mps"]),
            expected_speed,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Mean inflow no longer reproduces the configured U*.")
        sim_relative = _safe_relative_path(entry["sim_path"], label="case sim_path")
        case_root = root.joinpath(*sim_relative.parts).parent
        sources = sorted(case_root.rglob("CYLINDER.h5"))
        if len(sources) != 1:
            raise ValueError("Transfer case must contain exactly one CYLINDER.h5.")
        source_relative = sources[0].relative_to(root).as_posix()
        if source_relative not in verified_files:
            raise ValueError("Transfer source is not in the immutable file manifest.")
        with h5py.File(sources[0], "r") as h5:
            if h5.attrs.get("sc_transfer_definition_fingerprint_sha256") != campaign_definition[
                "closure_fingerprint_sha256"
            ]:
                raise ValueError("Transfer source lacks the frozen closure fingerprint.")
            if h5.attrs.get("effective_parameter_fingerprint_sha256") != expected_record[
                "effective_parameter_fingerprint_sha256"
            ]:
                raise ValueError("Transfer source effective-parameter identity changed.")

    if set(by_key) != set(_expected_case_keys()):
        raise ValueError("Campaign does not match the exact eight-case grid.")
    for pair_key, levels in pair_rows.items():
        if set(levels) != set(TIMESTEP_LEVELS):
            raise ValueError(f"Case pair {pair_key} lacks dt or dt_half.")
        coarse = levels["dt"]
        fine = levels["dt_half"]
        if int(fine["num_steps"]) != 2 * int(coarse["num_steps"]):
            raise ValueError(f"Case pair {pair_key} does not halve its timestep exactly.")
        if not math.isclose(
            float(coarse["duration_s"]),
            float(fine["duration_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"Case pair {pair_key} does not share one endpoint.")

    for relative, identity in manifest.get("implementation_sources", {}).items():
        if not isinstance(identity, Mapping) or identity.get("path") != relative:
            raise ValueError("Implementation source identity is malformed.")
        current = REPO_ROOT / relative
        if (
            not current.is_file()
            or current.stat().st_size != int(identity.get("bytes", -1))
            or sha256_file(current) != identity.get("sha256")
        ):
            raise ValueError(f"Implementation source changed: {relative}")
    return manifest


def build_dry_run_plan(
    manifest_path: Path,
    *,
    execution_campaign_root: PurePosixPath,
    results_root: PurePosixPath,
    qblade_root: PurePosixPath,
    python_executable: PurePosixPath = PurePosixPath("/usr/bin/python3"),
    execution_repo_root: PurePosixPath = PurePosixPath("/work/VorLap"),
    threads: int = 4,
    checkpoint_every: int = 0,
) -> dict[str, Any]:
    """Build eight sequential commands without executing any solver."""

    manifest = validate_transfer_campaign(manifest_path)
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads < 8:
        raise ValueError("threads must be an integer from 1 through 7.")
    if (
        isinstance(checkpoint_every, bool)
        or not isinstance(checkpoint_every, int)
        or checkpoint_every < 0
    ):
        raise ValueError("checkpoint_every must be a non-negative integer.")
    for name, value in (
        ("execution_campaign_root", execution_campaign_root),
        ("results_root", results_root),
        ("qblade_root", qblade_root),
        ("python_executable", python_executable),
        ("execution_repo_root", execution_repo_root),
    ):
        if not isinstance(value, PurePosixPath) or not value.is_absolute():
            raise ValueError(f"{name} must be an absolute PurePosixPath.")
    commands = []
    for entry in manifest["cases"]:
        sim_relative = _safe_relative_path(entry["sim_path"], label="case sim_path")
        output = results_root / str(entry["case_id"])
        argv = [
            str(python_executable),
            str(execution_repo_root / "run_headless" / "run_qblade_sim.py"),
            "--sim",
            str(execution_campaign_root.joinpath(*sim_relative.parts)),
            "--qblade-root",
            str(qblade_root),
            "--output-dir",
            str(output),
            "--result-name",
            "result",
            "--result-filter",
            str(execution_campaign_root / "cantilever_results.filter"),
            "--threads",
            str(threads),
            "--sil-vorlap",
            "--coupling-mode",
            "sil-vorlap",
            "--sil-vorlap-history-mode",
            "compact",
        ]
        if checkpoint_every:
            argv.extend(
                (
                    "--checkpoint-every",
                    str(checkpoint_every),
                    "--checkpoint-dir",
                    str(output / "checkpoints"),
                )
            )
        commands.append({"case_id": entry["case_id"], "argv": argv})
    plan = {
        "schema": PLAN_SCHEMA,
        "dry_run": True,
        "execution_authorized": False,
        "scientific_scope": manifest["scope"],
        "campaign_manifest_sha256": sha256_file(Path(manifest_path).resolve()),
        "campaign_manifest_fingerprint_sha256": manifest[
            "manifest_fingerprint_sha256"
        ],
        "transfer_definition_fingerprint_sha256": manifest["sc_transfer"][
            "definition_fingerprint_sha256"
        ],
        "threads": threads,
        "checkpoint_every": checkpoint_every,
        "commands": commands,
    }
    plan["plan_fingerprint_sha256"] = canonical_sha256(plan)
    return plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    artifacts = subparsers.add_parser(
        "artifacts", help="Write the analytical comparison table and figure."
    )
    artifacts.add_argument("--output", type=Path, required=True)
    generate = subparsers.add_parser(
        "generate", help="Generate exactly eight QBlade inputs; never execute them."
    )
    generate.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate", help="Validate an immutable transfer campaign."
    )
    validate.add_argument("--campaign", type=Path, required=True)
    plan = subparsers.add_parser(
        "plan", help="Write a dry run plan; never execute it."
    )
    plan.add_argument("--campaign", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--execution-campaign-root", type=PurePosixPath, required=True)
    plan.add_argument("--results-root", type=PurePosixPath, required=True)
    plan.add_argument(
        "--qblade-root",
        type=PurePosixPath,
        default=PurePosixPath("/work/VorLap/run_headless/QBladeCE_2.0.9.7"),
    )
    plan.add_argument("--threads", type=int, default=4)
    plan.add_argument("--checkpoint-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "artifacts":
        result: Any = {
            name: str(path)
            for name, path in render_analytical_artifacts(args.output).items()
        }
    elif args.command == "generate":
        result = str(generate_transfer_campaign(args.output))
    elif args.command == "validate":
        manifest = validate_transfer_campaign(args.campaign)
        result = {
            "campaign": str(Path(args.campaign).resolve()),
            "case_count": manifest["case_count"],
            "status": "valid",
        }
    else:
        plan = build_dry_run_plan(
            args.campaign,
            execution_campaign_root=args.execution_campaign_root,
            results_root=args.results_root,
            qblade_root=args.qblade_root,
            threads=args.threads,
            checkpoint_every=args.checkpoint_every,
        )
        plan_output = args.output.expanduser().resolve()
        campaign_root = args.campaign.expanduser().resolve().parent
        try:
            plan_output.relative_to(campaign_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "Dry run plans must be written outside the immutable campaign tree."
            )
        _write_json(plan_output, plan)
        for row in plan["commands"]:
            print(shlex.join(row["argv"]))
        result = str(plan_output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
