#!/usr/bin/env python3
"""Audit a parked VorLap/QBlade campaign before using it as paper evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.qblade_nodes import (
    action_target_from_node_id as _shared_action_target_from_node_id,
    component_id_from_node_id as _shared_component_id_from_node_id,
)
from vorlap.qblade_batch import turbine_dir_for_sim
from vorlap.spectral_reconstruction import verify_frequency_band_spectral_content
from vorlap.dlc_manifest import (
    DLCManifestError,
    load_manifest as load_dlc_manifest,
    selected_row as selected_dlc_row,
    validate_wind_record,
)


SOURCE_COVERAGE_REQUIRED_COLUMNS = (
    "scenario",
    "case",
    "component",
    "wind_speed_mps",
    "azimuth_deg",
    "aoa_min_deg",
    "aoa_max_deg",
    "source_aoa_min_deg",
    "source_aoa_max_deg",
    "re_min",
    "re_max",
    "source_re_min",
    "source_re_max",
    "freq_min_hz",
    "freq_max_hz",
    "nyquist_hz",
    "nyquist_ratio",
    "max_nodal_force_n",
    "force_coefficient_max_abs",
    "source_force_coefficient_max_abs",
    "max_structural_velocity_mps",
    "force_power_mean_w",
    "force_power_max_abs_w",
    "samples",
    "node_samples",
)

SOURCE_COVERAGE_COUNT_COLUMNS = (
    "re_out_of_range_count",
    "aoa_out_of_range_count",
    "aliased_frequency_count",
    "source_clip_count",
    "force_coefficient_out_of_range_count",
    "nonfinite_count",
)

SOURCE_COVERAGE_OPTIONAL_COLUMNS = (
    "airfoil_id",
    "reynolds_model",
    "re_lower_bound_warning_count",
)

SOURCE_RANGE_EXCURSION_COUNT_COLUMNS = {
    "re_out_of_range_count",
    "aoa_out_of_range_count",
    "source_clip_count",
    "force_coefficient_out_of_range_count",
}

APPROVED_NACA0018_REYNOLDS_EXTENSION_GRID = (10_000.0, 500_000.0, 2_000_000.0)
APPROVED_NACA0018_REYNOLDS_PARENT_GRID = (30_000.0, 500_000.0, 2_000_000.0)
APPROVED_REYNOLDS_EXTENSION_GENERATOR = "scripts/extend_replicated_reynolds_grid.py"


def approved_reynolds_envelope_extension_matches(
    source_row: dict[str, Any],
    expected_parent_sha256: str,
) -> bool:
    """Accept the approved 30k-to-10k envelope-only NACA0018 source revision.

    The approved manifest hash remains the immutable parent used by already
    completed rows.  A later runtime HDF5 may differ only by extending the
    explicitly Reynolds-independent replicated grid to Re=10,000, provided
    its generator provenance names that exact parent hash.
    """
    if str(source_row.get("airfoil_id", "")) != "NACA0018":
        return False
    if str(source_row.get("reynolds_model", "")) != "replicated_reynolds_independent":
        return False
    if str(source_row.get("generator", "")) != "scripts/process_naluout_to_h5.py":
        return False
    processing = source_row.get("processing_config")
    if not isinstance(processing, dict):
        return False
    if str(processing.get("replicate_re_parent_sha256", "")) != expected_parent_sha256:
        return False
    if processing.get("reynolds_model") != "replicated_reynolds_independent":
        return False
    if processing.get("reynolds_envelope_extension_exact") is not True:
        return False
    if processing.get("reynolds_envelope_extension_generator") != APPROVED_REYNOLDS_EXTENSION_GENERATOR:
        return False
    try:
        reynolds = tuple(float(value) for value in source_row.get("reynolds", []))
        output_reynolds = tuple(float(value) for value in processing.get("output_reynolds", []))
        source_reynolds = tuple(float(value) for value in processing.get("source_reynolds", []))
        parent_reynolds = tuple(
            float(value) for value in processing.get("reynolds_envelope_parent_grid", [])
        )
    except (TypeError, ValueError):
        return False
    return (
        reynolds == APPROVED_NACA0018_REYNOLDS_EXTENSION_GRID
        and output_reynolds == APPROVED_NACA0018_REYNOLDS_EXTENSION_GRID
        and source_reynolds == (500_000.0,)
        and parent_reynolds == APPROVED_NACA0018_REYNOLDS_PARENT_GRID
    )

GEOMETRY_AUDIT_REQUIRED_COLUMNS = (
    "swap_index",
    "controller_index",
    "requested_node_id",
    "canonical_node_id",
    "controller_node_id",
    "action_id",
    "action_position",
    "force_swap_x",
    "force_swap_y",
    "force_swap_z",
    "velocity_swap_x",
    "velocity_swap_y",
    "velocity_swap_z",
    "vorlap_x_m",
    "vorlap_y_m",
    "vorlap_z_m",
    "chord_norm",
    "normal_norm",
    "span_norm",
    "chord_dot_normal",
    "chord_dot_span",
    "normal_dot_span",
    "basis_determinant",
    "basis_status",
)

GEOMETRY_AUDIT_NUMERIC_COLUMNS = (
    "swap_index",
    "controller_index",
    "action_position",
    "force_swap_x",
    "force_swap_y",
    "force_swap_z",
    "velocity_swap_x",
    "velocity_swap_y",
    "velocity_swap_z",
    "vorlap_x_m",
    "vorlap_y_m",
    "vorlap_z_m",
    "chord_norm",
    "normal_norm",
    "span_norm",
    "chord_dot_normal",
    "chord_dot_span",
    "normal_dot_span",
    "basis_determinant",
)

STABLE_SUPPORT_PRESERVED_STRUCTURAL_KEYWORDS = (
    "TWRHEIGHT",
    "TRQTBCLEAR",
    "HUBPOS",
    "TRQTBCONN",
    "RTRCLEAR",
)

STABLE_SUPPORT_SIM_KEYWORDS = ("RELAXSTEPS", "RELAXTIMESTEP", "RAMPUP", "ADDDAMP")

STRUCTURAL_QUALIFICATION_DEFAULT = "structural_model_qualification.json"
STRUCTURAL_MODEL_CHOICES = (
    "generated_patched_demonstration",
    "repaired_original_export",
    "source_exported_model",
)
SIGNED_CHANNEL_SELECTION_DEFAULT = "signed_channel_selection.json"
ULTIMATE_INPUT_BASIS_DEFAULT = "ultimate_input_basis.json"
ULTIMATE_INPUT_BASIS_REVIEW_STATUSES = frozenset({"reviewed", "reviewed_final", "final", "approved"})
PRODUCTION_ARCHIVE_MANIFEST_DEFAULT = "production_archive_manifest.json"
PRODUCTION_ARCHIVE_REQUIRED_CATEGORIES = (
    "audit_json",
    "bridge_log",
    "figure_provenance",
    "postprocess_csv",
    "qblade_log",
    "run_summary",
)
SPECTRAL_PSD_COMPARISON_DEFAULT = "postprocess/spectral_psd_comparison.csv"
SPECTRAL_PSD_COMPARISON_REQUIRED_COLUMNS = (
    "scenario",
    "case",
    "component",
    "channel",
    "source",
    "expected_frequency_hz",
    "peak_frequency_hz",
    "frequency_error_hz",
    "band_power_fraction",
    "peak_power",
    "passed",
    "artifact",
)
SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES = ("analytic", "offline", "sil", "qblade")

POSTPROCESS_REQUIRED_ARTIFACTS = (
    ("postprocess/completion_summary.csv", "csv"),
    ("postprocess/completion_table.tex", "file"),
    ("postprocess/rainflow_validation.csv", "rainflow_validation"),
    ("postprocess/summary_metrics.csv", "csv"),
    ("postprocess/signed_channel_metrics.csv", "csv"),
    ("postprocess/extreme_channel_metrics.csv", "csv"),
    ("postprocess/ultimate_allowables_template.csv", "csv"),
    ("postprocess/ultimate_margin_metrics.csv", "csv"),
    ("postprocess/del_comparison.csv", "csv"),
    ("postprocess/del_comparison_diagnostics.csv", "csv"),
    ("postprocess/baseline_identity_comparison.csv", "csv"),
    ("postprocess/baseline_identity_comparison_diagnostics.csv", "csv"),
    ("postprocess/signed_del_comparison.csv", "csv"),
    ("postprocess/signed_del_comparison_diagnostics.csv", "csv"),
    ("postprocess/signed_baseline_identity_comparison.csv", "csv"),
    ("postprocess/signed_baseline_identity_comparison_diagnostics.csv", "csv"),
    ("postprocess/source_coverage.csv", "csv"),
    ("postprocess/source_coverage_case_summary.csv", "csv"),
    ("postprocess/critical_case_summary.csv", "csv"),
    ("postprocess/critical_case_rainflow_ranges.csv", "csv"),
    ("postprocess/del_table.tex", "file"),
    ("postprocess/signed_del_table.tex", "file"),
    ("postprocess/parked_campaign_report.tex", "file"),
    ("postprocess/completion_heatmap.png", "file"),
    ("postprocess/signed_del_heatmap.png", "file"),
    ("postprocess/source_coverage_sanity.png", "file"),
    ("postprocess/critical_case_diagnostic.png", "file"),
)

POSTPROCESS_PAPER_CSV_SCHEMAS = {
    "postprocess/completion_summary.csv": (
        "scenario",
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "requested_steps",
        "completed_steps",
        "completed_time_s",
        "failed",
        "failure_message",
        "exported_results",
        "summary_file",
        "smoke_only",
        "postprocess_warnings",
    ),
    "postprocess/summary_metrics.csv": (
        "scenario",
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "blade",
        "requested_position",
        "force_position",
        "moment_position",
        "force_del_N",
        "moment_del_Nm",
        "force_rainflow_cycles",
        "moment_rainflow_cycles",
        "equivalent_cycles",
        "force_rms_N",
        "moment_rms_Nm",
        "force_max_N",
        "moment_max_Nm",
        "samples",
        "duration_s",
        "smoke_only",
        "postprocess_warnings",
        "result_file",
    ),
    "postprocess/signed_channel_metrics.csv": (
        "scenario",
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "blade",
        "requested_position",
        "position",
        "quantity",
        "axis",
        "unit",
        "channel",
        "signed_del",
        "rainflow_cycles",
        "equivalent_cycles",
        "rms",
        "mean",
        "min",
        "max",
        "samples",
        "duration_s",
        "smoke_only",
        "postprocess_warnings",
        "result_file",
    ),
    "postprocess/extreme_channel_metrics.csv": (
        "scenario",
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "blade",
        "position",
        "quantity",
        "axis",
        "unit",
        "channel",
        "min",
        "max",
        "max_abs",
        "samples",
        "duration_s",
        "smoke_only",
        "postprocess_warnings",
        "result_file",
    ),
    "postprocess/ultimate_allowables_template.csv": (
        "blade",
        "position",
        "quantity",
        "axis",
        "unit",
        "allowable_abs",
        "allowable_label",
        "load_factor",
        "resistance_factor",
        "notes",
    ),
    "postprocess/ultimate_margin_metrics.csv": (
        "scenario",
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "blade",
        "position",
        "quantity",
        "axis",
        "unit",
        "channel",
        "max_abs",
        "allowable_abs",
        "allowable_label",
        "load_factor",
        "resistance_factor",
        "factored_demand",
        "factored_allowable",
        "utilization",
        "reserve_factor",
        "ultimate_status",
        "postprocess_warnings",
        "result_file",
    ),
    "postprocess/source_coverage.csv": (
        "scenario",
        "case",
        "component",
        "wind_speed_mps",
        "azimuth_deg",
        "aoa_min_deg",
        "aoa_max_deg",
        "source_aoa_min_deg",
        "source_aoa_max_deg",
        "re_min",
        "re_max",
        "source_re_min",
        "source_re_max",
        "freq_min_hz",
        "freq_max_hz",
        "nyquist_hz",
        "nyquist_ratio",
        "max_nodal_force_n",
        "force_coefficient_max_abs",
        "source_force_coefficient_max_abs",
        "max_structural_velocity_mps",
        "force_power_mean_w",
        "force_power_max_abs_w",
        "re_out_of_range_count",
        "aoa_out_of_range_count",
        "aliased_frequency_count",
        "source_clip_count",
        "force_coefficient_out_of_range_count",
        "nonfinite_count",
        "samples",
        "node_samples",
        "summary_file",
    ),
    "postprocess/source_coverage_case_summary.csv": (
        "scenario",
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "min_nyquist_ratio",
        "max_force_coefficient_ratio",
        "max_aoa_margin_deg",
        "max_re_margin",
        "max_nodal_force_n",
        "max_structural_velocity_mps",
        "force_power_max_abs_w",
        "violation_count",
    ),
    "postprocess/del_comparison.csv": (
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "force_del_baseline_N",
        "force_del_vorlap_N",
        "force_delta_pct",
        "moment_del_baseline_Nm",
        "moment_del_vorlap_Nm",
        "moment_delta_pct",
        "equivalent_cycles_baseline",
        "equivalent_cycles_vorlap",
        "smoke_only",
        "postprocess_warnings",
    ),
    "postprocess/baseline_identity_comparison.csv": (
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "force_del_baseline_N",
        "force_del_identity_N",
        "force_identity_delta_pct",
        "moment_del_baseline_Nm",
        "moment_del_identity_Nm",
        "moment_identity_delta_pct",
        "equivalent_cycles_baseline",
        "equivalent_cycles_identity",
        "smoke_only",
        "postprocess_warnings",
    ),
    "postprocess/signed_del_comparison.csv": (
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "quantity",
        "axis",
        "unit",
        "channel_baseline",
        "channel_vorlap",
        "signed_del_baseline",
        "signed_del_vorlap",
        "signed_delta_pct",
        "rainflow_cycles_baseline",
        "rainflow_cycles_vorlap",
        "equivalent_cycles_baseline",
        "equivalent_cycles_vorlap",
        "smoke_only",
        "postprocess_warnings",
    ),
    "postprocess/signed_baseline_identity_comparison.csv": (
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "quantity",
        "axis",
        "unit",
        "channel_baseline",
        "channel_identity",
        "signed_del_baseline",
        "signed_del_identity",
        "signed_identity_delta_pct",
        "rainflow_cycles_baseline",
        "rainflow_cycles_identity",
        "equivalent_cycles_baseline",
        "equivalent_cycles_identity",
        "smoke_only",
        "postprocess_warnings",
    ),
    "postprocess/critical_case_summary.csv": (
        "case",
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "quantity",
        "axis",
        "unit",
        "channel_baseline",
        "channel_vorlap",
        "signed_del_baseline",
        "signed_del_vorlap",
        "signed_del_ratio",
        "signed_delta_pct",
        "rainflow_cycles_baseline",
        "rainflow_cycles_vorlap",
        "equivalent_cycles_baseline",
        "equivalent_cycles_vorlap",
        "postprocess_warnings",
        "selection_reason",
        "min_nyquist_ratio",
        "max_force_coefficient_ratio",
        "force_power_max_abs_w",
        "source_coverage_violation_count",
    ),
    "postprocess/critical_case_rainflow_ranges.csv": ("range", "count"),
}

POSTPROCESS_PAPER_COMPARISON_ARTIFACTS = {
    "postprocess/del_comparison.csv",
    "postprocess/baseline_identity_comparison.csv",
    "postprocess/signed_del_comparison.csv",
    "postprocess/signed_baseline_identity_comparison.csv",
}

POSTPROCESS_PAPER_INTERPRETED_ARTIFACTS = {
    *POSTPROCESS_PAPER_COMPARISON_ARTIFACTS,
    "postprocess/critical_case_summary.csv",
}

POSTPROCESS_PAPER_NUMERIC_COLUMNS = {
    "postprocess/del_comparison.csv": (
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "force_del_baseline_N",
        "force_del_vorlap_N",
        "moment_del_baseline_Nm",
        "moment_del_vorlap_Nm",
    ),
    "postprocess/baseline_identity_comparison.csv": (
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "force_del_baseline_N",
        "force_del_identity_N",
        "moment_del_baseline_Nm",
        "moment_del_identity_Nm",
    ),
    "postprocess/signed_del_comparison.csv": (
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "signed_del_baseline",
        "signed_del_vorlap",
        "rainflow_cycles_baseline",
        "rainflow_cycles_vorlap",
    ),
    "postprocess/signed_baseline_identity_comparison.csv": (
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "signed_del_baseline",
        "signed_del_identity",
        "rainflow_cycles_baseline",
        "rainflow_cycles_identity",
    ),
    "postprocess/critical_case_summary.csv": (
        "wind_speed_mps",
        "azimuth_deg",
        "position",
        "signed_del_baseline",
        "signed_del_vorlap",
        "rainflow_cycles_baseline",
        "rainflow_cycles_vorlap",
    ),
}

POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/del_comparison_diagnostics.csv"] = POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/del_comparison.csv"]
POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/baseline_identity_comparison_diagnostics.csv"] = POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/baseline_identity_comparison.csv"]
POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_del_comparison_diagnostics.csv"] = POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_del_comparison.csv"]
POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_baseline_identity_comparison_diagnostics.csv"] = POSTPROCESS_PAPER_CSV_SCHEMAS["postprocess/signed_baseline_identity_comparison.csv"]

CRITICAL_CASE_SELECTION_TEXT_COLUMNS = (
    "case",
    "quantity",
    "axis",
    "unit",
    "channel_baseline",
    "channel_vorlap",
    "postprocess_warnings",
)

CRITICAL_CASE_SELECTION_NUMERIC_COLUMNS = (
    "wind_speed_mps",
    "azimuth_deg",
    "position",
    "signed_del_baseline",
    "signed_del_vorlap",
    "signed_delta_pct",
    "rainflow_cycles_baseline",
    "rainflow_cycles_vorlap",
)

FORCE_SIGN_PROBE_REQUIRED_COLUMNS = (
    "scenario",
    "case",
    "node",
    "axis",
    "sign",
    "force_n",
    "output_dir",
    "summary_file",
    "result_file",
    "completed_steps",
    "failed",
    "exported_results",
    "observation_status",
    "delta_position_m",
    "paired_response_delta_m",
    "paired_passed",
    "passed",
)

FORCE_SIGN_PROBE_SIGNS = ("positive", "negative")
FORCE_SIGN_PROBE_INVALID_OBSERVATION_STATUSES = {
    "missing_position_channel",
    "insufficient_samples",
    "zero_or_unresolved_response",
}
FORCE_SIGN_PROBE_PAIR_RESPONSE_TOLERANCE_M = 1.0e-9

ULTIMATE_MARGIN_REQUIRED_COLUMNS = (
    "scenario",
    "case",
    "wind_speed_mps",
    "azimuth_deg",
    "blade",
    "position",
    "quantity",
    "axis",
    "unit",
    "channel",
    "max_abs",
    "allowable_abs",
    "load_factor",
    "resistance_factor",
    "factored_demand",
    "factored_allowable",
    "utilization",
    "reserve_factor",
    "ultimate_status",
)

ULTIMATE_MARGIN_NUMERIC_COLUMNS = (
    "wind_speed_mps",
    "azimuth_deg",
    "position",
    "max_abs",
    "allowable_abs",
    "load_factor",
    "resistance_factor",
    "factored_demand",
    "factored_allowable",
    "utilization",
)

ULTIMATE_EXTREME_REQUIRED_COLUMNS = (
    "scenario",
    "case",
    "wind_speed_mps",
    "azimuth_deg",
    "blade",
    "position",
    "quantity",
    "axis",
    "unit",
    "channel",
    "max_abs",
)

ULTIMATE_CHANNEL_KEY_COLUMNS = (
    "scenario",
    "case",
    "blade",
    "position",
    "quantity",
    "axis",
    "unit",
    "channel",
)

ULTIMATE_MARGIN_STATUS_VALUES = (
    "ok",
    "exceeds_allowable",
    "missing_allowable",
    "invalid_demand",
    "invalid_factor",
    "invalid_allowable",
)

@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    scenario: str = ""
    case: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "scenario": self.scenario,
            "case": self.case,
        }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scenarios(spec: str) -> list[str]:
    scenarios = [part for part in spec.replace(",", " ").split() if part]
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    return scenarios


def parse_numeric_grid_spec(spec: str) -> list[float]:
    text = str(spec).strip()
    if not text:
        return []
    if text.startswith("linspace:"):
        _, start, stop, count = text.split(":", 3)
        n = int(count)
        if n <= 0:
            raise ValueError("linspace count must be positive.")
        if n == 1:
            return [float(start)]
        start_value = float(start)
        stop_value = float(stop)
        step = (stop_value - start_value) / float(n - 1)
        return [start_value + i * step for i in range(n)]
    if text.startswith("range:"):
        _, start, stop, step = text.split(":", 3)
        start_value = float(start)
        stop_value = float(stop)
        step_value = float(step)
        if step_value == 0.0:
            raise ValueError("range step must be nonzero.")
        values: list[float] = []
        current = start_value
        tolerance = abs(step_value) * 1.0e-9 + 1.0e-12
        if step_value > 0.0:
            while current <= stop_value + tolerance:
                values.append(current)
                current += step_value
        else:
            while current >= stop_value - tolerance:
                values.append(current)
                current += step_value
        return values
    return [float(part) for part in text.replace(",", " ").split() if part]


def normalized_numeric_set(values: Any) -> set[float]:
    return {round(float(value), 9) for value in values}


def expected_scenarios(settings: dict[str, Any], args: argparse.Namespace) -> list[str]:
    scenario_names = settings.get("scenario_names")
    if isinstance(scenario_names, list) and scenario_names:
        return [str(name) for name in scenario_names]
    return parse_scenarios(args.scenarios)


def close_enough(left: float, right: float, *, rel: float = 1.0e-9, abs_tol: float = 1.0e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=rel, abs_tol=abs_tol)


def normalized_case_key(wind_speed: float, azimuth_deg: float) -> tuple[float, float]:
    return (round(float(wind_speed), 9), round(float(azimuth_deg), 9))


def planned_case_keys(settings: dict[str, Any]) -> set[tuple[float, float]]:
    planned = settings.get("planned_cases")
    if planned:
        return {
            normalized_case_key(case["wind_speed_mps"], case["parked_azimuth_deg"])
            for case in planned
        }
    return {
        normalized_case_key(wind_speed, azimuth)
        for wind_speed in settings.get("wind_speeds", [])
        for azimuth in settings.get("azimuths_deg", [])
    }


def issue(
    issues: list[AuditIssue],
    severity: str,
    code: str,
    message: str,
    *,
    scenario: str = "",
    case: str = "",
) -> None:
    issues.append(AuditIssue(severity=severity, code=code, message=message, scenario=scenario, case=case))


def source_coverage_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "source_coverage", None)
    if path is None:
        return args.run_root / "postprocess" / "source_coverage.csv"
    return Path(path)


def force_sign_probe_report_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "force_sign_probe_report", None)
    if path is None:
        return args.run_root / "sign_probes" / "force_sign_probe_report.csv"
    return Path(path)


def ultimate_margins_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "ultimate_margins", None)
    if path is None:
        return args.run_root / "postprocess" / "ultimate_margin_metrics.csv"
    return Path(path)


def ultimate_input_basis_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "ultimate_input_basis", None)
    if path is None:
        return args.run_root / ULTIMATE_INPUT_BASIS_DEFAULT
    return Path(path)


def production_archive_manifest_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "production_archive_manifest", None)
    if path is None:
        return args.run_root / PRODUCTION_ARCHIVE_MANIFEST_DEFAULT
    return Path(path)


def spectral_psd_comparison_path(args: argparse.Namespace) -> Path:
    path = getattr(args, "spectral_psd_comparison", None)
    if path is None:
        return args.run_root / SPECTRAL_PSD_COMPARISON_DEFAULT
    return Path(path)


def extreme_channel_metrics_path(args: argparse.Namespace) -> Path:
    return args.run_root / "postprocess" / "extreme_channel_metrics.csv"


def component_id_from_node_id(node_id: str) -> str:
    return _shared_component_id_from_node_id(node_id)


def vorlap_config_file_for_case(case: dict[str, Any]) -> Path | None:
    sim_path = Path(str(case.get("sim", "")))
    preferred = sorted(sim_path.parent.glob("*/Control/vorlap_qblade_external.json"))
    if preferred:
        return preferred[0]
    fallback = sorted(sim_path.parent.glob("**/vorlap_qblade_external.json"))
    if fallback:
        return fallback[0]
    return None


def expected_source_component_node_counts_for_case(
    case: dict[str, Any],
    issues: list[AuditIssue],
    *,
    scenario: str,
    case_label: str,
    require: bool,
) -> dict[str, int]:
    config_path = vorlap_config_file_for_case(case)
    if config_path is None:
        if require:
            issue(
                issues,
                "error",
                "missing_source_coverage_config",
                "Could not find vorlap_qblade_external.json to determine expected source-coverage components.",
                scenario=scenario,
                case=case_label,
            )
        return {}
    try:
        config = read_json(config_path)
    except Exception as exc:
        if require:
            issue(
                issues,
                "error",
                "unreadable_source_coverage_config",
                f"Could not read VorLap runtime config for source-coverage audit: {config_path}: {exc}",
                scenario=scenario,
                case=case_label,
            )
        return {}
    node_ids = config.get("node_ids")
    if not isinstance(node_ids, list):
        if require:
            issue(
                issues,
                "error",
                "source_coverage_config_schema_mismatch",
                "VorLap runtime config has no node_ids list for source-coverage component audit.",
                scenario=scenario,
                case=case_label,
            )
        return {}
    components: dict[str, int] = {}
    for node_id in node_ids:
        try:
            component = component_id_from_node_id(str(node_id))
            components[component] = components.get(component, 0) + 1
        except ValueError as exc:
            if require:
                issue(
                    issues,
                    "error",
                    "source_coverage_config_node_parse_failed",
                    str(exc),
                    scenario=scenario,
                    case=case_label,
                )
    if require and not components:
        issue(
            issues,
            "error",
            "source_coverage_config_has_no_components",
            "VorLap runtime config node_ids did not yield any expected source-coverage components.",
            scenario=scenario,
            case=case_label,
        )
    return components


def completed_steps_for_case(case: dict[str, Any], issues: list[AuditIssue], *, scenario: str, case_label: str) -> int | None:
    result_dir = Path(case.get("result_dir", ""))
    summary_path = first_summary_file(result_dir, issues, scenario=scenario, case_label=case_label)
    if summary_path is None:
        return None
    try:
        summary = read_json(summary_path)
        return int(summary.get("completed_steps", 0) or 0)
    except Exception as exc:
        issue(
            issues,
            "error",
            "unreadable_source_coverage_run_summary",
            f"Could not read run summary for source-coverage sample audit: {summary_path}: {exc}",
            scenario=scenario,
            case=case_label,
        )
        return None


def source_coverage_scenarios(settings: dict[str, Any], args: argparse.Namespace, issues: list[AuditIssue]) -> list[str]:
    scenarios = parse_scenarios(getattr(args, "source_coverage_scenarios", "vorlap"))
    known = set(expected_scenarios(settings, args))
    unknown = [scenario for scenario in scenarios if scenario not in known]
    if unknown:
        issue(issues, "error", "unknown_source_coverage_scenario", f"Source-coverage scenarios are not in the campaign plan: {', '.join(unknown)}.")
    return scenarios


def geometry_proof_scenarios(settings: dict[str, Any], args: argparse.Namespace, issues: list[AuditIssue]) -> list[str]:
    default = getattr(args, "source_coverage_scenarios", "vorlap")
    scenarios = parse_scenarios(getattr(args, "geometry_proof_scenarios", default))
    known = set(expected_scenarios(settings, args))
    unknown = [scenario for scenario in scenarios if scenario not in known]
    if unknown:
        issue(issues, "error", "unknown_geometry_proof_scenario", f"Geometry-proof scenarios are not in the campaign plan: {', '.join(unknown)}.")
    return scenarios


def spectral_force_scenarios(settings: dict[str, Any], args: argparse.Namespace, issues: list[AuditIssue]) -> list[str]:
    default = getattr(args, "source_coverage_scenarios", "vorlap")
    scenarios = parse_scenarios(getattr(args, "spectral_force_scenarios", default))
    known = set(expected_scenarios(settings, args))
    unknown = [scenario for scenario in scenarios if scenario not in known]
    if unknown:
        issue(issues, "error", "unknown_spectral_force_scenario", f"Spectral-force scenarios are not in the campaign plan: {', '.join(unknown)}.")
    return scenarios


def row_float(
    row: dict[str, str],
    name: str,
    issues: list[AuditIssue],
    *,
    scenario: str,
    case: str,
    component: str,
) -> float | None:
    raw = row.get(name, "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        issue(
            issues,
            "error",
            "invalid_source_coverage_value",
            f"Invalid source-coverage value {name}={raw!r} for component {component}.",
            scenario=scenario,
            case=case,
        )
        return None
    if not math.isfinite(value):
        issue(
            issues,
            "error",
            "nonfinite_source_coverage_value",
            f"Non-finite source-coverage value {name}={raw!r} for component {component}.",
            scenario=scenario,
            case=case,
        )
        return None
    return value


def row_count(
    row: dict[str, str],
    name: str,
    issues: list[AuditIssue],
    *,
    scenario: str,
    case: str,
    component: str,
) -> int | None:
    raw = row.get(name, "")
    if raw == "":
        return None
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        issue(
            issues,
            "error",
            "invalid_source_coverage_count",
            f"Invalid source-coverage count {name}={raw!r} for component {component}.",
            scenario=scenario,
            case=case,
        )
        return None
    if value < 0:
        issue(
            issues,
            "error",
            "negative_source_coverage_count",
            f"Negative source-coverage count {name}={raw!r} for component {component}.",
            scenario=scenario,
            case=case,
        )
        return None
    return value


def parse_axis_list(spec: str) -> list[str]:
    axes = [part.upper() for part in str(spec).replace(",", " ").split() if part]
    if not axes:
        raise ValueError("At least one force-sign probe axis is required.")
    bad = [axis for axis in axes if axis not in {"X", "Y", "Z"}]
    if bad:
        raise ValueError(f"Unknown force-sign probe axis/axes: {', '.join(bad)}.")
    return axes


def row_bool_text(row: dict[str, str], name: str) -> bool:
    return str(row.get(name, "")).strip().lower() in {"true", "1", "yes"}


def probe_row_float(row: dict[str, str], name: str) -> float:
    try:
        return float(row.get(name, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def parse_probe_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "local"}:
        return True
    if text in {"false", "0", "no", "global", ""}:
        return False
    return None


def path_matches(left: Any, right: Any) -> bool:
    try:
        return Path(str(left)).expanduser().resolve(strict=False) == Path(str(right)).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return str(left) == str(right)


def value_close_enough(raw: Any, expected: float) -> bool:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return False
    return close_enough(value, expected)


def action_target_from_node_id(node_id: str) -> tuple[str, float]:
    return _shared_action_target_from_node_id(node_id)


def nested_dict(data: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, dict) else None


def force_sign_probe_summary_matches_row(
    summary_file: Path,
    result_file: Path,
    row: dict[str, str],
    *,
    scenario: str,
    case: str,
    node: str,
    axis: str,
    sign: str,
    force: float,
    completed_steps: int,
    manifest_case: dict[str, Any] | None,
    issues: list[AuditIssue],
) -> bool:
    row_passed = True
    try:
        summary = read_json(summary_file)
    except Exception as exc:
        issue(
            issues,
            "error",
            "unreadable_force_sign_probe_summary",
            f"Could not read force-sign probe run summary: {summary_file}: {exc}",
            scenario=scenario,
            case=case,
        )
        return False
    if not isinstance(summary, dict):
        issue(
            issues,
            "error",
            "force_sign_probe_summary_schema_mismatch",
            f"Force-sign probe summary is not a JSON object: {summary_file}",
            scenario=scenario,
            case=case,
        )
        return False

    if summary.get("coupling_mode") != "sil-constant-force" or summary.get("sil_constant_force") is not True:
        issue(
            issues,
            "error",
            "force_sign_probe_summary_wrong_coupling_mode",
            "Force-sign probe summary was not produced by the sil-constant-force diagnostic path.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
    if bool(summary.get("failed", False)) != row_bool_text(row, "failed"):
        issue(
            issues,
            "error",
            "force_sign_probe_summary_status_mismatch",
            "Force-sign probe summary failed flag does not match the CSV row.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
    if bool(summary.get("exported_results", False)) != row_bool_text(row, "exported_results"):
        issue(
            issues,
            "error",
            "force_sign_probe_summary_status_mismatch",
            "Force-sign probe summary exported_results flag does not match the CSV row.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
    try:
        summary_steps = int(float(summary.get("completed_steps", "")))
    except (TypeError, ValueError):
        summary_steps = -1
    if summary_steps != completed_steps:
        issue(
            issues,
            "error",
            "force_sign_probe_summary_steps_mismatch",
            f"Force-sign probe summary completed_steps={summary.get('completed_steps')!r} does not match CSV completed_steps={completed_steps}.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
    try:
        update_count = int(float(summary.get("sil_constant_force_updates", "")))
    except (TypeError, ValueError):
        update_count = -1
    if update_count != completed_steps:
        issue(
            issues,
            "error",
            "force_sign_probe_update_count_mismatch",
            f"Force-sign probe applied {summary.get('sil_constant_force_updates')!r} constant-force updates for {completed_steps} completed steps.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
    if not path_matches(summary.get("result_file", ""), result_file):
        issue(
            issues,
            "error",
            "force_sign_probe_summary_result_mismatch",
            "Force-sign probe summary result_file does not match the CSV row.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
    if manifest_case is not None and not path_matches(summary.get("sim", ""), manifest_case.get("sim", "")):
        issue(
            issues,
            "error",
            "force_sign_probe_manifest_sim_mismatch",
            "Force-sign probe summary sim path does not match the generated case manifest.",
            scenario=scenario,
            case=case,
        )
        row_passed = False

    row_local = parse_probe_bool(row.get("local", "global"))
    if row_local is None:
        issue(
            issues,
            "error",
            "invalid_force_sign_probe_local_flag",
            f"Invalid force-sign probe local/global flag {row.get('local')!r}.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
    elif row_local:
        issue(
            issues,
            "error",
            "local_force_sign_probe_not_global",
            "Force-sign probe audit requires global ADDFORCE evidence because VorLap emits global force rows.",
            scenario=scenario,
            case=case,
        )
        row_passed = False

    try:
        expected_action_id, expected_position = action_target_from_node_id(node)
    except ValueError as exc:
        issue(issues, "error", "invalid_force_sign_probe_node", str(exc), scenario=scenario, case=case)
        return False

    actions = summary.get("sil_constant_force_actions")
    if not isinstance(actions, list) or len(actions) != 1 or not isinstance(actions[0], dict):
        issue(
            issues,
            "error",
            "force_sign_probe_summary_action_mismatch",
            "Force-sign probe summary must contain exactly one sil_constant_force_actions entry.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
        action: dict[str, Any] | None = None
    else:
        action = actions[0]
    if action is not None:
        action_local = parse_probe_bool(action.get("local", False))
        checks = {
            "action": action.get("action") == "ADDFORCE",
            "source_node_id": str(action.get("source_node_id", "")).strip() == node,
            "id": str(action.get("id", "")).strip() == expected_action_id,
            "direction": str(action.get("direction", "")).strip().upper() == axis,
            "position": value_close_enough(action.get("position"), expected_position),
            "force": value_close_enough(action.get("value"), force),
            "local": action_local is False,
        }
        if not all(checks.values()):
            failed_fields = ", ".join(name for name, passed in checks.items() if not passed)
            issue(
                issues,
                "error",
                "force_sign_probe_summary_action_mismatch",
                f"Force-sign probe summary action does not match the CSV row: {failed_fields}.",
                scenario=scenario,
                case=case,
            )
            row_passed = False

    try:
        max_abs_force = float(summary.get("sil_constant_force_max_abs_force", "nan"))
    except (TypeError, ValueError):
        max_abs_force = float("nan")
    if not close_enough(max_abs_force, abs(force)):
        issue(
            issues,
            "error",
            "force_sign_probe_summary_action_mismatch",
            "Force-sign probe summary max constant-force magnitude does not match the CSV row.",
            scenario=scenario,
            case=case,
        )
        row_passed = False

    observations = summary.get("force_sign_probe_observations")
    if not isinstance(observations, list) or len(observations) != 1 or not isinstance(observations[0], dict):
        issue(
            issues,
            "error",
            "force_sign_probe_summary_observation_mismatch",
            "Force-sign probe summary must contain exactly one force_sign_probe_observations entry.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
        observation: dict[str, Any] | None = None
    else:
        observation = observations[0]
    if observation is not None:
        observation_local = parse_probe_bool(observation.get("local", False))
        observation_status = str(observation.get("status", "")).strip()
        checks = {
            "source_node_id": str(observation.get("source_node_id", "")).strip() == node,
            "action_id": str(observation.get("action_id", "")).strip() == expected_action_id,
            "direction": str(observation.get("direction", "")).strip().upper() == axis,
            "position": value_close_enough(observation.get("position"), expected_position),
            "force": value_close_enough(observation.get("force_n"), force),
            "local": observation_local is False,
            "status": observation_status == str(row.get("observation_status", "")).strip(),
            "channel": bool(str(observation.get("observed_position_channel", "")).strip()),
        }
        if not all(checks.values()):
            failed_fields = ", ".join(name for name, passed in checks.items() if not passed)
            issue(
                issues,
                "error",
                "force_sign_probe_summary_observation_mismatch",
                f"Force-sign probe summary observation does not match the CSV row: {failed_fields}.",
                scenario=scenario,
                case=case,
            )
            row_passed = False
        try:
            delta_position = float(observation.get("delta_position_m", "nan"))
        except (TypeError, ValueError):
            delta_position = float("nan")
        if not math.isfinite(delta_position):
            issue(
                issues,
                "error",
                "force_sign_probe_summary_observation_sign_mismatch",
                "Force-sign probe summary observation does not show a finite displacement response.",
                scenario=scenario,
                case=case,
            )
            row_passed = False
        try:
            row_delta_position = float(row.get("delta_position_m", "nan"))
        except (TypeError, ValueError):
            row_delta_position = float("nan")
        if not value_close_enough(row_delta_position, delta_position):
            issue(
                issues,
                "error",
                "force_sign_probe_summary_observation_mismatch",
                "Force-sign probe summary delta_position_m does not match the CSV row.",
                scenario=scenario,
                case=case,
            )
            row_passed = False
        if observation_status in FORCE_SIGN_PROBE_INVALID_OBSERVATION_STATUSES:
            issue(
                issues,
                "error",
                "force_sign_probe_observation_failed",
                f"Force-sign probe observation status is {observation_status!r}.",
                scenario=scenario,
                case=case,
            )
            row_passed = False

    metadata = nested_dict(summary, "reproducibility", "case_metadata")
    metadata_probe = None if metadata is None else metadata.get("force_sign_probe")
    if not isinstance(metadata_probe, dict):
        issue(
            issues,
            "error",
            "force_sign_probe_summary_metadata_mismatch",
            "Force-sign probe summary is missing reproducibility.case_metadata.force_sign_probe.",
            scenario=scenario,
            case=case,
        )
        row_passed = False
    else:
        metadata_local = parse_probe_bool(metadata_probe.get("local", False))
        checks = {
            "scenario": str(metadata_probe.get("scenario", "")).strip() == scenario,
            "case": str(metadata_probe.get("case", "")).strip() == case,
            "node": str(metadata_probe.get("node", "")).strip() == node,
            "axis": str(metadata_probe.get("axis", "")).strip().upper() == axis,
            "sign": str(metadata_probe.get("sign", "")).strip().lower() == sign,
            "force": value_close_enough(metadata_probe.get("force_n"), force),
            "local": metadata_local is False,
        }
        if not all(checks.values()):
            failed_fields = ", ".join(name for name, passed in checks.items() if not passed)
            issue(
                issues,
                "error",
                "force_sign_probe_summary_metadata_mismatch",
                f"Force-sign probe summary metadata does not match the CSV row: {failed_fields}.",
                scenario=scenario,
                case=case,
            )
            row_passed = False
    metadata_case = None if metadata is None else metadata.get("case")
    if isinstance(metadata_case, dict) and str(metadata_case.get("label", "")).strip() != case:
        issue(
            issues,
            "error",
            "force_sign_probe_summary_metadata_mismatch",
            "Force-sign probe summary metadata case label does not match the CSV row.",
            scenario=scenario,
            case=case,
        )
        row_passed = False

    return row_passed


def audit_force_sign_probes(
    settings: dict[str, Any],
    args: argparse.Namespace,
    issues: list[AuditIssue],
    manifests: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    path = force_sign_probe_report_path(args)
    require = bool(getattr(args, "require_force_sign_probes", False)) or getattr(args, "stage", "") == "sign-probes"
    if not path.is_file():
        if require:
            issue(issues, "error", "missing_force_sign_probe_report", f"Missing force-sign probe report CSV: {path}")
        return None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [column for column in FORCE_SIGN_PROBE_REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            issue(
                issues,
                "error",
                "force_sign_probe_schema_mismatch",
                f"Force-sign probe CSV is missing required columns: {', '.join(missing_columns)}.",
            )
            return {"path": str(path), "rows": 0, "complete_nodes": []}
        rows = list(reader)

    expected = set(expected_scenarios(settings, args))
    expected_cases_by_scenario: dict[str, set[str]] = {}
    manifest_cases_by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    if manifests:
        for scenario, manifest in manifests.items():
            cases_by_label = {
                str(case.get("label", "")).strip(): case
                for case in manifest.get("cases", [])
                if str(case.get("label", "")).strip()
            }
            manifest_cases_by_scenario[scenario] = cases_by_label
            expected_cases_by_scenario[scenario] = set(cases_by_label)
    required_axes = parse_axis_list(getattr(args, "force_sign_required_axes", "X Y Z"))
    required_nodes = {
        str(node).strip()
        for node in getattr(args, "force_sign_required_node", [])
        if str(node).strip()
    }
    passed_keys: set[tuple[str, str, str, str, str]] = set()
    paired_candidates: dict[tuple[str, str, str, str], dict[str, tuple[float, float, dict[str, str]]]] = {}
    nodes: set[str] = set()

    for row in rows:
        scenario = str(row.get("scenario", "")).strip()
        case = str(row.get("case", "")).strip()
        node = str(row.get("node", "")).strip()
        axis = str(row.get("axis", "")).strip().upper()
        sign = str(row.get("sign", "")).strip().lower()
        row_passed = True
        nodes.add(node)
        if scenario not in expected:
            issue(issues, "warning", "unexpected_force_sign_probe_scenario", "Force-sign probe row scenario is not in the campaign settings.", scenario=scenario, case=case)
        expected_cases = expected_cases_by_scenario.get(scenario)
        if expected_cases is not None and case and case not in expected_cases:
            issue(
                issues,
                "error" if require else "warning",
                "unexpected_force_sign_probe_case",
                "Force-sign probe row case is not in the generated scenario manifest.",
                scenario=scenario,
                case=case,
            )
            row_passed = False
        if not node or not case:
            issue(issues, "error", "incomplete_force_sign_probe_identity", "Force-sign probe rows must include nonempty case and node fields.", scenario=scenario, case=case)
            continue
        if axis not in {"X", "Y", "Z"}:
            issue(issues, "error", "invalid_force_sign_probe_axis", f"Invalid force-sign probe axis {axis!r}.", scenario=scenario, case=case)
            continue
        if sign not in FORCE_SIGN_PROBE_SIGNS:
            issue(issues, "error", "invalid_force_sign_probe_sign", f"Invalid force-sign probe sign {sign!r}.", scenario=scenario, case=case)
            continue
        try:
            force = float(row.get("force_n", ""))
        except (TypeError, ValueError):
            issue(issues, "error", "invalid_force_sign_probe_force", f"Invalid force_n={row.get('force_n')!r}.", scenario=scenario, case=case)
            continue
        if not math.isfinite(force) or force == 0.0:
            issue(issues, "error", "invalid_force_sign_probe_force", f"force_n must be finite and nonzero, got {row.get('force_n')!r}.", scenario=scenario, case=case)
            row_passed = False
        elif sign == "positive" and force <= 0.0:
            issue(issues, "error", "force_sign_probe_force_sign_mismatch", f"Positive probe has force_n={force:g}.", scenario=scenario, case=case)
            row_passed = False
        elif sign == "negative" and force >= 0.0:
            issue(issues, "error", "force_sign_probe_force_sign_mismatch", f"Negative probe has force_n={force:g}.", scenario=scenario, case=case)
            row_passed = False

        try:
            completed_steps = int(float(row.get("completed_steps", "0")))
        except (TypeError, ValueError):
            completed_steps = 0
            issue(issues, "error", "invalid_force_sign_probe_steps", f"Invalid completed_steps={row.get('completed_steps')!r}.", scenario=scenario, case=case)
            row_passed = False

        summary_file = Path(str(row.get("summary_file", "")))
        result_file = Path(str(row.get("result_file", "")))
        if not summary_file.is_file():
            issue(issues, "error", "missing_force_sign_probe_summary", f"Missing force-sign probe run summary: {summary_file}", scenario=scenario, case=case)
            row_passed = False
        if not result_file.is_file():
            issue(issues, "error", "missing_force_sign_probe_result", f"Missing force-sign probe result file: {result_file}", scenario=scenario, case=case)
            row_passed = False
        if summary_file.is_file():
            manifest_case = manifest_cases_by_scenario.get(scenario, {}).get(case)
            row_passed = (
                force_sign_probe_summary_matches_row(
                    summary_file,
                    result_file,
                    row,
                    scenario=scenario,
                    case=case,
                    node=node,
                    axis=axis,
                    sign=sign,
                    force=force,
                    completed_steps=completed_steps,
                    manifest_case=manifest_case,
                    issues=issues,
                )
                and row_passed
            )
        if completed_steps < 1:
            issue(issues, "error", "force_sign_probe_no_completed_steps", "Force-sign probe completed no timesteps.", scenario=scenario, case=case)
            row_passed = False
        if row_bool_text(row, "failed"):
            issue(issues, "error", "force_sign_probe_failed_run", "Force-sign probe run is marked failed.", scenario=scenario, case=case)
            row_passed = False
        if not row_bool_text(row, "exported_results"):
            issue(issues, "error", "force_sign_probe_results_not_exported", "Force-sign probe results were not exported.", scenario=scenario, case=case)
            row_passed = False
        observation_status = str(row.get("observation_status", "")).strip()
        if observation_status in FORCE_SIGN_PROBE_INVALID_OBSERVATION_STATUSES:
            issue(
                issues,
                "error",
                "force_sign_probe_observation_failed",
                f"Force-sign probe observation status is {row.get('observation_status')!r}.",
                scenario=scenario,
                case=case,
            )
            row_passed = False
        delta_position = probe_row_float(row, "delta_position_m")
        if not math.isfinite(delta_position):
            issue(
                issues,
                "error",
                "invalid_force_sign_probe_delta",
                f"Invalid delta_position_m={row.get('delta_position_m')!r}.",
                scenario=scenario,
                case=case,
            )
            row_passed = False
        if not row_bool_text(row, "passed"):
            issue(issues, "error", "force_sign_probe_report_failed", f"Force-sign probe report marks this row failed: {row.get('issue', '')}", scenario=scenario, case=case)
            row_passed = False
        if not row_bool_text(row, "paired_passed"):
            issue(
                issues,
                "error",
                "force_sign_probe_pair_report_failed",
                f"Force-sign probe report marks the paired response failed: {row.get('paired_response_delta_m', '')}.",
                scenario=scenario,
                case=case,
            )
            row_passed = False
        if row_passed and scenario in expected and (expected_cases is None or case in expected_cases):
            paired_candidates.setdefault((node, axis, scenario, case), {})[sign] = (force, delta_position, row)

    for (node, axis, scenario, case), sign_data in sorted(paired_candidates.items()):
        positive = sign_data.get("positive")
        negative = sign_data.get("negative")
        if positive is None or negative is None:
            issue(
                issues,
                "error",
                "missing_force_sign_probe_pair",
                f"Missing positive/negative force-sign probe pair for node {node}, axis {axis}.",
                scenario=scenario,
                case=case,
            )
            continue
        pos_force, pos_delta, pos_row = positive
        neg_force, neg_delta, neg_row = negative
        force_delta = pos_force - neg_force
        response_delta = pos_delta - neg_delta
        reported_pair_deltas = [probe_row_float(pos_row, "paired_response_delta_m"), probe_row_float(neg_row, "paired_response_delta_m")]
        if any(not close_enough(value, response_delta) for value in reported_pair_deltas):
            issue(
                issues,
                "error",
                "force_sign_probe_pair_report_mismatch",
                f"Force-sign probe paired_response_delta_m does not match computed response {response_delta:g} m.",
                scenario=scenario,
                case=case,
            )
            continue
        if not math.isfinite(force_delta) or force_delta <= 0.0 or not math.isfinite(response_delta):
            issue(
                issues,
                "error",
                "invalid_force_sign_probe_pair",
                f"Invalid paired force-sign values for node {node}, axis {axis}.",
                scenario=scenario,
                case=case,
            )
            continue
        if response_delta <= FORCE_SIGN_PROBE_PAIR_RESPONSE_TOLERANCE_M:
            issue(
                issues,
                "error",
                "force_sign_probe_paired_response_failed",
                f"Paired +F/-F response is not positive for node {node}, axis {axis}: {response_delta:g} m.",
                scenario=scenario,
                case=case,
            )
            continue
        for sign in FORCE_SIGN_PROBE_SIGNS:
            passed_keys.add((node, axis, sign, scenario, case))

    complete_nodes: list[str] = []
    complete_node_cases: list[dict[str, str]] = []
    scenario_case_pairs = sorted({(scenario, case) for _node, _axis, _sign, scenario, case in passed_keys})
    for node in sorted(nodes):
        if not node:
            continue
        node_complete = False
        for scenario, case in scenario_case_pairs:
            if all((node, axis, sign, scenario, case) in passed_keys for axis in required_axes for sign in FORCE_SIGN_PROBE_SIGNS):
                node_complete = True
                complete_node_cases.append({"node": node, "scenario": scenario, "case": case})
                break
        if node_complete:
            complete_nodes.append(node)

    if required_nodes:
        for node in sorted(required_nodes):
            for axis in required_axes:
                for sign in FORCE_SIGN_PROBE_SIGNS:
                    if not any((node, axis, sign, scenario, case) in passed_keys for scenario, case in scenario_case_pairs):
                        issue(
                            issues,
                            "error",
                            "missing_force_sign_probe_axis",
                            f"Missing passing {sign} {axis} force-sign probe for node {node}.",
                        )
    elif require and not complete_nodes:
        issue(
            issues,
            "error",
            "missing_complete_force_sign_probe_node",
            f"No probed node has passing positive/negative results for axes: {', '.join(required_axes)}.",
        )

    return {
        "path": str(path),
        "rows": len(rows),
        "complete_nodes": complete_nodes,
        "complete_node_cases": complete_node_cases,
        "required_axes": required_axes,
        "required_nodes": sorted(required_nodes),
    }


def ultimate_margin_scenarios(settings: dict[str, Any], args: argparse.Namespace, issues: list[AuditIssue]) -> list[str]:
    scenarios = parse_scenarios(getattr(args, "ultimate_margin_scenarios", "vorlap"))
    known = set(expected_scenarios(settings, args))
    unknown = [scenario for scenario in scenarios if scenario not in known]
    if unknown:
        issue(issues, "error", "unknown_ultimate_margin_scenario", f"Ultimate-margin scenarios are not in the campaign plan: {', '.join(unknown)}.")
    return scenarios


def ultimate_row_float(
    row: dict[str, str],
    name: str,
    issues: list[AuditIssue],
    *,
    scenario: str,
    case: str,
    channel: str,
) -> float | None:
    raw = row.get(name, "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        issue(
            issues,
            "error",
            "invalid_ultimate_margin_value",
            f"Invalid ultimate-margin value {name}={raw!r} for channel {channel or '?'}.",
            scenario=scenario,
            case=case,
        )
        return None
    if not math.isfinite(value):
        issue(
            issues,
            "error",
            "nonfinite_ultimate_margin_value",
            f"Non-finite ultimate-margin value {name}={raw!r} for channel {channel or '?'}.",
            scenario=scenario,
            case=case,
        )
        return None
    return value


def ultimate_channel_key(
    row: dict[str, Any],
    *,
    position: float | None = None,
) -> tuple[str, str, str, float, str, str, str, str]:
    if position is None:
        position = float(row.get("position", ""))
    return (
        str(row.get("scenario", "")).strip(),
        str(row.get("case", "")).strip(),
        str(row.get("blade", "")).strip(),
        round(float(position), 6),
        str(row.get("quantity", "")).strip(),
        str(row.get("axis", "")).strip(),
        str(row.get("unit", "")).strip(),
        str(row.get("channel", "")).strip(),
    )


def ultimate_channel_label(key: tuple[str, str, str, float, str, str, str, str]) -> str:
    scenario, case, blade, position, quantity, axis, unit, channel = key
    return f"{scenario}/{case} {blade} pos {position:g} {quantity} {axis} [{unit}] {channel}"


def expected_ultimate_extreme_channels(
    args: argparse.Namespace,
    issues: list[AuditIssue],
    *,
    requested: set[str],
    expected_cases: set[tuple[str, str]],
    tolerance: float,
    require: bool,
) -> dict[tuple[str, str, str, float, str, str, str, str], float]:
    path = extreme_channel_metrics_path(args)
    if not path.is_file():
        if require:
            issue(
                issues,
                "error",
                "missing_ultimate_extreme_channels",
                f"Missing signed-channel extrema CSV needed to prove ultimate-margin channel coverage: {path}",
            )
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [column for column in ULTIMATE_EXTREME_REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            issue(
                issues,
                "error",
                "ultimate_extreme_schema_mismatch",
                f"Extreme-channel CSV is missing required columns for ultimate audit: {', '.join(missing_columns)}.",
            )
            return {}
        rows = list(reader)
    if not rows:
        issue(issues, "error", "ultimate_extreme_empty", f"Extreme-channel CSV has no rows: {path}")
        return {}

    expected: dict[tuple[str, str, str, float, str, str, str, str], float] = {}
    for row in rows:
        scenario = str(row.get("scenario", "")).strip()
        case = str(row.get("case", "")).strip()
        channel = str(row.get("channel", "")).strip()
        if scenario not in requested:
            continue
        if expected_cases and (scenario, case) not in expected_cases:
            continue
        if not case or not channel:
            issue(issues, "error", "incomplete_ultimate_extreme_identity", "Extreme-channel rows must include nonempty case and channel fields.", scenario=scenario, case=case)
            continue
        try:
            position = float(row.get("position", ""))
            max_abs = float(row.get("max_abs", ""))
        except (TypeError, ValueError):
            issue(issues, "error", "invalid_ultimate_extreme_value", f"Extreme-channel row has invalid position/max_abs for channel {channel}.", scenario=scenario, case=case)
            continue
        if not math.isfinite(position) or not math.isfinite(max_abs):
            issue(issues, "error", "nonfinite_ultimate_extreme_value", f"Extreme-channel row has non-finite position/max_abs for channel {channel}.", scenario=scenario, case=case)
            continue
        if max_abs < -tolerance:
            issue(issues, "error", "negative_ultimate_extreme_demand", f"Extreme-channel max_abs is negative for channel {channel}.", scenario=scenario, case=case)
            continue
        key = ultimate_channel_key(row, position=position)
        existing = expected.get(key)
        if existing is not None and not close_enough(existing, max_abs, rel=1.0e-7, abs_tol=tolerance):
            issue(
                issues,
                "error",
                "duplicate_ultimate_extreme_channel",
                f"Extreme-channel CSV has duplicate inconsistent rows for {ultimate_channel_label(key)}.",
                scenario=scenario,
                case=case,
            )
            continue
        expected[key] = max_abs
    if require and not expected:
        issue(
            issues,
            "error",
            "missing_ultimate_extreme_channel_rows",
            "Extreme-channel CSV has no rows for requested ultimate-margin scenarios/cases.",
        )
    return expected


def audit_ultimate_margins(
    settings: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    require = bool(getattr(args, "require_ultimate_margins", False))
    explicit_path = getattr(args, "ultimate_margins", None) is not None
    if not require and not explicit_path:
        return None

    path = ultimate_margins_path(args)
    if not path.is_file():
        if require:
            issue(issues, "error", "missing_ultimate_margins", f"Missing ultimate-margin CSV: {path}")
        return None

    requested_scenarios = ultimate_margin_scenarios(settings, args, issues)
    requested = set(requested_scenarios)
    expected_cases: set[tuple[str, str]] = set()
    expected_case_values: dict[tuple[str, str], tuple[float, float]] = {}
    for scenario in requested_scenarios:
        manifest = manifests.get(scenario)
        if manifest is None:
            continue
        for case in manifest.get("cases", []):
            label = str(case.get("label", ""))
            if label:
                expected_cases.add((scenario, label))
                expected_case_values[(scenario, label)] = normalized_case_key(case.get("wind_speed", math.nan), case.get("azimuth_deg", math.nan))

    tolerance = float(getattr(args, "ultimate_margin_tolerance", 1.0e-9))
    expected_extreme_demands = expected_ultimate_extreme_channels(
        args,
        issues,
        requested=requested,
        expected_cases=expected_cases,
        tolerance=tolerance,
        require=require,
    )

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [column for column in ULTIMATE_MARGIN_REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            issue(
                issues,
                "error",
                "ultimate_margin_schema_mismatch",
                f"Ultimate-margin CSV is missing required columns: {', '.join(missing_columns)}.",
            )
            return {"path": str(path), "rows": 0, "covered_cases": 0, "expected_cases": len(expected_cases)}
        rows = list(reader)

    if not rows:
        issue(issues, "error", "ultimate_margin_empty", f"Ultimate-margin CSV has no rows: {path}")
        return {"path": str(path), "rows": 0, "covered_cases": 0, "expected_cases": len(expected_cases)}

    covered_cases: set[tuple[str, str]] = set()
    covered_channel_keys: set[tuple[str, str, str, float, str, str, str, str]] = set()
    status_counts: dict[str, int] = {}
    fail_on_exceedance = bool(getattr(args, "fail_on_ultimate_exceedance", False))

    for row in rows:
        scenario = str(row.get("scenario", "")).strip()
        case = str(row.get("case", "")).strip()
        channel = str(row.get("channel", "")).strip()
        status = str(row.get("ultimate_status", "")).strip()
        if scenario not in requested:
            continue
        covered_cases.add((scenario, case))
        if not case or not channel:
            issue(issues, "error", "incomplete_ultimate_margin_identity", "Ultimate-margin rows must include nonempty case and channel fields.", scenario=scenario, case=case)
            continue
        if (scenario, case) not in expected_cases:
            issue(issues, "warning", "unexpected_ultimate_margin_case", "Ultimate-margin row is not in the expected campaign manifest.", scenario=scenario, case=case)
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in ULTIMATE_MARGIN_STATUS_VALUES:
            issue(issues, "error", "unknown_ultimate_status", f"Unknown ultimate_status={status!r}.", scenario=scenario, case=case)

        values = {
            name: ultimate_row_float(row, name, issues, scenario=scenario, case=case, channel=channel)
            for name in ULTIMATE_MARGIN_NUMERIC_COLUMNS
        }
        try:
            reserve_factor = float(row.get("reserve_factor", ""))
        except (TypeError, ValueError):
            reserve_factor = math.nan
            issue(issues, "error", "invalid_ultimate_margin_value", f"Invalid ultimate-margin value reserve_factor={row.get('reserve_factor')!r} for channel {channel}.", scenario=scenario, case=case)
        if any(value is None for value in values.values()):
            continue

        expected_case_key = expected_case_values.get((scenario, case))
        if expected_case_key is not None and normalized_case_key(values["wind_speed_mps"], values["azimuth_deg"]) != expected_case_key:
            issue(issues, "error", "ultimate_margin_case_mismatch", "Ultimate-margin wind speed or azimuth does not match the generated case manifest.", scenario=scenario, case=case)
        channel_key = ultimate_channel_key(row, position=values["position"])
        covered_channel_keys.add(channel_key)
        expected_extreme = expected_extreme_demands.get(channel_key)
        if expected_extreme is None and expected_extreme_demands:
            issue(
                issues,
                "error" if require else "warning",
                "unexpected_ultimate_margin_channel",
                f"Ultimate-margin row does not match any signed-channel extrema row: {ultimate_channel_label(channel_key)}.",
                scenario=scenario,
                case=case,
            )
        elif expected_extreme is not None and not close_enough(values["max_abs"], expected_extreme, rel=1.0e-7, abs_tol=tolerance):
            issue(
                issues,
                "error",
                "ultimate_margin_extreme_mismatch",
                f"Ultimate margin max_abs={values['max_abs']:g} does not match signed-channel extrema max_abs={expected_extreme:g}.",
                scenario=scenario,
                case=case,
            )
        if values["max_abs"] < -tolerance:
            issue(issues, "error", "negative_ultimate_demand", "Ultimate max_abs is negative.", scenario=scenario, case=case)
        if values["allowable_abs"] <= 0.0:
            issue(issues, "error", "missing_ultimate_allowable", "Ultimate margin is missing a positive allowable_abs.", scenario=scenario, case=case)
        if values["load_factor"] <= 0.0 or values["resistance_factor"] <= 0.0:
            issue(issues, "error", "invalid_ultimate_factor", "Ultimate load_factor and resistance_factor must be positive.", scenario=scenario, case=case)
        if values["factored_demand"] < -tolerance or values["factored_allowable"] <= 0.0:
            issue(issues, "error", "invalid_ultimate_factored_values", "Ultimate factored demand/allowable values are invalid.", scenario=scenario, case=case)
        if values["utilization"] < -tolerance:
            issue(issues, "error", "invalid_ultimate_utilization", "Ultimate utilization is negative.", scenario=scenario, case=case)
        if not math.isfinite(reserve_factor) and values["factored_demand"] > tolerance:
            issue(issues, "error", "nonfinite_ultimate_reserve_factor", "Ultimate reserve_factor is non-finite for nonzero demand.", scenario=scenario, case=case)

        expected_demand = values["max_abs"] * values["load_factor"]
        expected_allowable = values["allowable_abs"] * values["resistance_factor"]
        if math.isfinite(expected_demand) and not close_enough(values["factored_demand"], expected_demand, rel=1.0e-7, abs_tol=tolerance):
            issue(
                issues,
                "error",
                "ultimate_margin_calculation_inconsistent",
                f"factored_demand={values['factored_demand']:g} does not match max_abs*load_factor={expected_demand:g}.",
                scenario=scenario,
                case=case,
            )
        if math.isfinite(expected_allowable) and not close_enough(values["factored_allowable"], expected_allowable, rel=1.0e-7, abs_tol=tolerance):
            issue(
                issues,
                "error",
                "ultimate_margin_calculation_inconsistent",
                f"factored_allowable={values['factored_allowable']:g} does not match allowable_abs*resistance_factor={expected_allowable:g}.",
                scenario=scenario,
                case=case,
            )
        if values["factored_allowable"] > tolerance and math.isfinite(values["factored_allowable"]):
            expected_utilization = values["factored_demand"] / values["factored_allowable"]
            if math.isfinite(expected_utilization) and not close_enough(values["utilization"], expected_utilization, rel=1.0e-7, abs_tol=tolerance):
                issue(
                    issues,
                    "error",
                    "ultimate_margin_calculation_inconsistent",
                    f"utilization={values['utilization']:g} does not match factored_demand/factored_allowable={expected_utilization:g}.",
                    scenario=scenario,
                    case=case,
                )
        if values["factored_demand"] > tolerance and math.isfinite(values["factored_demand"]):
            expected_reserve_factor = values["factored_allowable"] / values["factored_demand"]
            if math.isfinite(expected_reserve_factor) and not close_enough(reserve_factor, expected_reserve_factor, rel=1.0e-7, abs_tol=tolerance):
                issue(
                    issues,
                    "error",
                    "ultimate_margin_calculation_inconsistent",
                    f"reserve_factor={reserve_factor:g} does not match factored_allowable/factored_demand={expected_reserve_factor:g}.",
                    scenario=scenario,
                    case=case,
                )

        if status in {"missing_allowable", "invalid_demand", "invalid_factor", "invalid_allowable"}:
            issue(issues, "error", "invalid_ultimate_margin_status", f"Ultimate margin row status is {status!r}.", scenario=scenario, case=case)
        elif status == "ok" and values["utilization"] > 1.0 + tolerance:
            issue(issues, "error", "ultimate_status_inconsistent", f"ultimate_status='ok' but utilization={values['utilization']:g}.", scenario=scenario, case=case)
        elif status == "exceeds_allowable":
            if values["utilization"] <= 1.0 + tolerance:
                issue(issues, "error", "ultimate_status_inconsistent", f"ultimate_status='exceeds_allowable' but utilization={values['utilization']:g}.", scenario=scenario, case=case)
            else:
                severity = "error" if fail_on_exceedance else "warning"
                issue(issues, severity, "ultimate_allowable_exceeded", f"Ultimate utilization {values['utilization']:g} exceeds 1.0.", scenario=scenario, case=case)

    missing_cases = sorted(expected_cases - covered_cases)
    for scenario, case in missing_cases:
        issue(
            issues,
            "error",
            "missing_ultimate_margin_case",
            "Ultimate-margin CSV has no row for this expected production case.",
            scenario=scenario,
            case=case,
        )

    missing_channels = sorted(expected_extreme_demands.keys() - covered_channel_keys)
    for key in missing_channels:
        scenario, case, *_ = key
        issue(
            issues,
            "error",
            "missing_ultimate_margin_channel",
            f"Ultimate-margin CSV has no row for signed-channel extrema row {ultimate_channel_label(key)}.",
            scenario=scenario,
            case=case,
        )

    return {
        "path": str(path),
        "rows": len(rows),
        "covered_cases": len(covered_cases),
        "expected_cases": len(expected_cases),
        "covered_channels": len(covered_channel_keys),
        "expected_channels": len(expected_extreme_demands),
        "scenarios": requested_scenarios,
        "status_counts": status_counts,
    }


def ultimate_basis_artifact_path(run_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path).strip()) if isinstance(raw_path, str) and raw_path.strip() else Path()
    return path if path.is_absolute() else run_root / path


def audit_completed_ultimate_allowables(path: Path, issues: list[AuditIssue]) -> dict[str, Any]:
    required_columns = (
        "blade",
        "position",
        "quantity",
        "axis",
        "unit",
        "allowable_abs",
        "load_factor",
        "resistance_factor",
    )
    if not path.is_file():
        issue(issues, "error", "ultimate_input_basis_missing_allowables", f"Ultimate allowables artifact does not exist: {path}")
        return {"path": str(path), "available": False, "rows": 0}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [column for column in required_columns if column not in fieldnames]
        if missing_columns:
            issue(
                issues,
                "error",
                "ultimate_input_basis_allowables_schema_mismatch",
                f"Ultimate allowables artifact is missing required column(s): {', '.join(missing_columns)}.",
            )
            return {"path": str(path), "available": True, "rows": 0}
        rows = list(reader)

    if not rows:
        issue(issues, "error", "ultimate_input_basis_allowables_empty", f"Ultimate allowables artifact has no rows: {path}")
        return {"path": str(path), "available": True, "rows": 0}

    valid_rows = 0
    for index, row in enumerate(rows, start=2):
        identity_missing = [
            column
            for column in ("blade", "quantity", "axis", "unit")
            if not str(row.get(column, "")).strip()
        ]
        if identity_missing:
            issue(
                issues,
                "error",
                "ultimate_input_basis_incomplete_allowable_identity",
                f"Ultimate allowables row {index} is missing identity field(s): {', '.join(identity_missing)}.",
            )
        try:
            position = float(row.get("position", ""))
            allowable = float(row.get("allowable_abs", ""))
            load_factor = float(row.get("load_factor", ""))
            resistance_factor = float(row.get("resistance_factor", ""))
        except (TypeError, ValueError):
            issue(issues, "error", "ultimate_input_basis_invalid_allowable_value", f"Ultimate allowables row {index} has a non-numeric position, allowable, or factor.")
            continue
        if not all(math.isfinite(value) for value in (position, allowable, load_factor, resistance_factor)):
            issue(issues, "error", "ultimate_input_basis_nonfinite_allowable_value", f"Ultimate allowables row {index} has a non-finite position, allowable, or factor.")
            continue
        if allowable <= 0.0:
            issue(issues, "error", "ultimate_input_basis_nonpositive_allowable", f"Ultimate allowables row {index} has allowable_abs <= 0.")
            continue
        if load_factor <= 0.0 or resistance_factor <= 0.0:
            issue(issues, "error", "ultimate_input_basis_nonpositive_factor", f"Ultimate allowables row {index} has nonpositive load_factor or resistance_factor.")
            continue
        if not identity_missing:
            valid_rows += 1

    if valid_rows == 0:
        issue(issues, "error", "ultimate_input_basis_no_valid_allowables", "Ultimate allowables artifact has no complete positive rows.")
    return {"path": str(path), "available": True, "rows": len(rows), "valid_rows": valid_rows}


def generated_draft_text(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith("draft:")


def audit_ultimate_input_basis(args: argparse.Namespace, issues: list[AuditIssue]) -> dict[str, Any] | None:
    require = bool(getattr(args, "require_ultimate_input_basis", False))
    path = ultimate_input_basis_path(args)
    if not path.is_file():
        if require:
            issue(issues, "error", "missing_ultimate_input_basis", f"Missing ultimate input basis manifest: {path}")
            return {"path": str(path), "available": False}
        return None

    try:
        payload = read_json(path)
    except Exception as exc:
        issue(issues, "error", "invalid_ultimate_input_basis", f"Could not read {path}: {exc}")
        return {"path": str(path), "available": False}
    if not isinstance(payload, dict):
        issue(issues, "error", "invalid_ultimate_input_basis", f"{path} must contain a JSON object.")
        return {"path": str(path), "available": False}

    if not nonempty_text(payload.get("basis_id")):
        issue(issues, "error", "ultimate_input_basis_missing_id", "Ultimate input basis manifest is missing basis_id.")

    review_status = str(payload.get("review_status", "")).strip().lower()
    if review_status not in ULTIMATE_INPUT_BASIS_REVIEW_STATUSES:
        accepted = ", ".join(sorted(ULTIMATE_INPUT_BASIS_REVIEW_STATUSES))
        issue(
            issues,
            "error",
            "ultimate_input_basis_not_reviewed",
            f"Ultimate input basis manifest must set review_status to one of: {accepted}.",
        )

    required_text_fields = (
        "allowable_type",
        "channel_mapping_basis",
        "stress_recovery_or_local_load_basis",
        "load_factor_basis",
        "resistance_factor_basis",
        "units_convention",
    )
    for field in required_text_fields:
        value = payload.get(field)
        if not nonempty_text(value):
            issue(issues, "error", "ultimate_input_basis_incomplete", f"{field} must be nonempty.")
        elif generated_draft_text(value):
            issue(issues, "error", "ultimate_input_basis_draft_text", f"{field} still contains generated DRAFT text.")

    raw_allowables = payload.get("allowables_artifact")
    raw_allowables_hash = payload.get("allowables_sha256")
    allowables_report: dict[str, Any] | None = None
    if not nonempty_text(raw_allowables):
        issue(issues, "error", "ultimate_input_basis_missing_allowables", "Ultimate input basis manifest is missing allowables_artifact.")
    else:
        allowables_path = ultimate_basis_artifact_path(args.run_root, raw_allowables)
        allowables_report = audit_completed_ultimate_allowables(allowables_path, issues)
        if not nonempty_text(raw_allowables_hash):
            issue(
                issues,
                "error",
                "ultimate_input_basis_missing_allowables_hash",
                "Ultimate input basis manifest is missing allowables_sha256.",
            )
        elif allowables_path.is_file():
            actual_hash = sha256_file(allowables_path)
            if str(raw_allowables_hash).strip().lower() != actual_hash:
                issue(
                    issues,
                    "error",
                    "ultimate_input_basis_allowables_hash_mismatch",
                    f"allowables_sha256 does not match {allowables_path}: expected {raw_allowables_hash}, observed {actual_hash}.",
                )

    return {
        "path": str(path),
        "available": True,
        "basis_id": str(payload.get("basis_id", "")),
        "review_status": str(payload.get("review_status", "")),
        "allowables": allowables_report,
    }


def geometry_audit_file_for_case(sim_path: Path) -> Path | None:
    """Find the generated per-node geometry/swap audit next to the runtime config."""
    case_dir = sim_path.parent
    preferred = sorted(case_dir.glob("*/Control/vorlap_qblade_geometry_audit.csv"))
    if preferred:
        return preferred[0]
    fallback = sorted(case_dir.glob("**/vorlap_qblade_geometry_audit.csv"))
    if fallback:
        return fallback[0]
    return None


def audit_geometry_file(path: Path, issues: list[AuditIssue], *, scenario: str, case_label: str) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [column for column in GEOMETRY_AUDIT_REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            issue(
                issues,
                "error",
                "geometry_audit_schema_mismatch",
                f"Geometry audit is missing required columns: {', '.join(missing_columns)}.",
                scenario=scenario,
                case=case_label,
            )
            return
        rows = list(reader)

    if not rows:
        issue(issues, "error", "geometry_audit_empty", f"Geometry audit has no node rows: {path}", scenario=scenario, case=case_label)
        return

    requested_nodes: set[str] = set()
    force_swaps: set[int] = set()
    velocity_swaps: set[int] = set()
    for row in rows:
        node_id = str(row.get("requested_node_id", "")).strip()
        if not node_id:
            issue(issues, "error", "geometry_audit_missing_node_id", "Geometry audit row has empty requested_node_id.", scenario=scenario, case=case_label)
        elif node_id in requested_nodes:
            issue(issues, "error", "geometry_audit_duplicate_node_id", f"Duplicate geometry audit node row: {node_id}.", scenario=scenario, case=case_label)
        requested_nodes.add(node_id)

        for column in GEOMETRY_AUDIT_NUMERIC_COLUMNS:
            raw = row.get(column, "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                issue(
                    issues,
                    "error",
                    "geometry_audit_invalid_numeric",
                    f"Geometry audit has invalid {column}={raw!r} for node {node_id or '?'}.",
                    scenario=scenario,
                    case=case_label,
                )
                continue
            if not math.isfinite(value):
                issue(
                    issues,
                    "error",
                    "geometry_audit_nonfinite_numeric",
                    f"Geometry audit has non-finite {column}={raw!r} for node {node_id or '?'}.",
                    scenario=scenario,
                    case=case_label,
                )
            if column.startswith("force_swap_") and value.is_integer():
                swap_index = int(value)
                if swap_index in force_swaps:
                    issue(issues, "error", "geometry_audit_duplicate_force_swap", f"Duplicate force swap index {swap_index}.", scenario=scenario, case=case_label)
                force_swaps.add(swap_index)
            if column.startswith("velocity_swap_") and value.is_integer():
                swap_index = int(value)
                if swap_index in velocity_swaps:
                    issue(issues, "error", "geometry_audit_duplicate_velocity_swap", f"Duplicate velocity swap index {swap_index}.", scenario=scenario, case=case_label)
                velocity_swaps.add(swap_index)

        if str(row.get("basis_status", "")).strip() != "ok":
            warning_text = str(row.get("basis_warnings", "")).strip()
            issue(
                issues,
                "error",
                "geometry_audit_basis_invalid",
                f"Geometry audit basis failed for node {node_id or '?'}: {warning_text or row.get('basis_status', '')}.",
                scenario=scenario,
                case=case_label,
            )


def geometry_audit_path_from_summary_or_case(summary: dict[str, Any], case: dict[str, Any]) -> Path | None:
    for key in ("sil_vorlap_geometry_audit_file", "geometry_audit_file", "geometry_audit"):
        raw_path = summary.get(key)
        if isinstance(raw_path, str) and raw_path.strip():
            path = Path(raw_path)
            if path.is_file():
                return path
            result_dir = Path(str(case.get("result_dir", "")))
            candidate = result_dir / path.name
            if candidate.is_file():
                return candidate
    sim_path = Path(str(case.get("sim", "")))
    if sim_path.is_file():
        return geometry_audit_file_for_case(sim_path)
    return None


def geometry_row_bool(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"true", "1", "yes"}


def geometry_row_float(
    row: dict[str, str],
    key: str,
    issues: list[AuditIssue],
    *,
    scenario: str,
    case_label: str,
    node_id: str,
) -> float | None:
    raw = row.get(key, "")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        issue(
            issues,
            "error",
            "geometry_position_proof_invalid_numeric",
            f"Geometry position proof has invalid {key}={raw!r} for node {node_id or '?'}.",
            scenario=scenario,
            case=case_label,
        )
        return None
    if not math.isfinite(value):
        issue(
            issues,
            "error",
            "geometry_position_proof_nonfinite_numeric",
            f"Geometry position proof has non-finite {key}={raw!r} for node {node_id or '?'}.",
            scenario=scenario,
            case=case_label,
        )
        return None
    return value


def audit_geometry_position_proof(
    settings: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    if not bool(getattr(args, "require_geometry_position_proof", False)):
        return None

    requested_scenarios = geometry_proof_scenarios(settings, args, issues)
    max_residual_m = float(getattr(args, "max_geometry_frame_fit_residual_m", 0.10))
    min_fit_points = int(getattr(args, "min_geometry_frame_fit_points", 3))
    allowed_missing = int(getattr(args, "allow_missing_geometry_reported_nodes", 0))
    checked_cases = 0
    checked_nodes = 0
    reported_nodes = 0
    fit_nodes = 0
    passing_cases = 0
    case_reports: list[dict[str, object]] = []

    for scenario in requested_scenarios:
        manifest = manifests.get(scenario)
        if manifest is None:
            issue(
                issues,
                "error",
                "geometry_position_proof_missing_manifest",
                "Geometry position proof scenario has no loaded case manifest.",
                scenario=scenario,
            )
            continue
        for case in manifest.get("cases", []):
            case_label = str(case.get("label", ""))
            summary_path = first_summary_file(Path(case.get("result_dir", "")), issues, scenario=scenario, case_label=case_label)
            if summary_path is None:
                continue
            try:
                summary = read_json(summary_path)
            except Exception as exc:
                issue(issues, "error", "geometry_position_summary_unreadable", f"Could not read run summary for geometry proof audit: {summary_path}: {exc}", scenario=scenario, case=case_label)
                continue
            audit_path = geometry_audit_path_from_summary_or_case(summary, case)
            if audit_path is None or not audit_path.is_file():
                issue(issues, "error", "missing_geometry_position_proof", "Missing QBlade-reported geometry audit CSV for this result case.", scenario=scenario, case=case_label)
                continue

            try:
                with audit_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = set(reader.fieldnames or [])
                    rows = list(reader)
            except Exception as exc:
                issue(issues, "error", "geometry_position_proof_unreadable", f"Could not read geometry position proof CSV {audit_path}: {exc}", scenario=scenario, case=case_label)
                continue
            required_columns = {
                "requested_node_id",
                "qblade_reported_available",
                "qblade_reported_x_m",
                "qblade_reported_y_m",
                "qblade_reported_z_m",
                "qblade_frame_fit_available",
                "qblade_frame_fit_x_m",
                "qblade_frame_fit_y_m",
                "qblade_frame_fit_z_m",
                "delta_reported_frame_fit_m",
            }
            missing_columns = sorted(required_columns - fieldnames)
            if missing_columns:
                issue(
                    issues,
                    "error",
                    "geometry_position_proof_schema_mismatch",
                    f"Geometry position proof is missing required columns: {', '.join(missing_columns)}.",
                    scenario=scenario,
                    case=case_label,
                )
                continue
            if not rows:
                issue(issues, "error", "geometry_position_proof_empty", f"Geometry position proof has no node rows: {audit_path}", scenario=scenario, case=case_label)
                continue

            checked_cases += 1
            case_errors_before = sum(1 for item in issues if item.severity == "error")
            case_reported = 0
            case_fit = 0
            case_missing = 0
            max_case_residual = 0.0
            max_residual_node = ""
            for row in rows:
                node_id = str(row.get("requested_node_id", "")).strip()
                checked_nodes += 1
                reported = geometry_row_bool(row, "qblade_reported_available")
                fit_available = geometry_row_bool(row, "qblade_frame_fit_available")
                if reported:
                    case_reported += 1
                    reported_nodes += 1
                    for column in ("qblade_reported_x_m", "qblade_reported_y_m", "qblade_reported_z_m"):
                        geometry_row_float(row, column, issues, scenario=scenario, case_label=case_label, node_id=node_id)
                else:
                    case_missing += 1
                if fit_available:
                    for column in ("qblade_frame_fit_x_m", "qblade_frame_fit_y_m", "qblade_frame_fit_z_m"):
                        geometry_row_float(row, column, issues, scenario=scenario, case_label=case_label, node_id=node_id)
                    residual = geometry_row_float(
                        row,
                        "delta_reported_frame_fit_m",
                        issues,
                        scenario=scenario,
                        case_label=case_label,
                        node_id=node_id,
                    )
                    if residual is None:
                        continue
                    case_fit += 1
                    fit_nodes += 1
                    if residual > max_case_residual:
                        max_case_residual = residual
                        max_residual_node = node_id
                    if residual > max_residual_m:
                        issue(
                            issues,
                            "error",
                            "geometry_position_fit_residual_exceeds_tolerance",
                            f"Geometry fitted-frame residual {residual:g} m for node {node_id or '?'} exceeds {max_residual_m:g} m.",
                            scenario=scenario,
                            case=case_label,
                        )

            if case_missing > allowed_missing:
                issue(
                    issues,
                    "error",
                    "geometry_position_missing_reported_nodes",
                    f"Geometry position proof has {case_missing} missing QBlade-reported node(s), above allowed {allowed_missing}.",
                    scenario=scenario,
                    case=case_label,
                )
            if case_fit < min_fit_points:
                issue(
                    issues,
                    "error",
                    "geometry_position_insufficient_frame_fit_points",
                    f"Geometry position proof has {case_fit} fitted-frame node(s), below required {min_fit_points}.",
                    scenario=scenario,
                    case=case_label,
                )
            if case_fit < case_reported - allowed_missing:
                issue(
                    issues,
                    "error",
                    "geometry_position_missing_frame_fit_nodes",
                    f"Geometry position proof has fitted-frame residuals for {case_fit}/{case_reported} reported node(s).",
                    scenario=scenario,
                    case=case_label,
                )
            case_has_errors = sum(1 for item in issues if item.severity == "error") > case_errors_before
            case_passed = (
                not case_has_errors
                and case_missing <= allowed_missing
                and case_fit >= min_fit_points
                and case_fit >= case_reported - allowed_missing
                and max_case_residual <= max_residual_m
            )
            if case_passed:
                passing_cases += 1
            case_reports.append(
                {
                    "scenario": scenario,
                    "case": case_label,
                    "path": str(audit_path),
                    "rows": len(rows),
                    "reported_nodes": case_reported,
                    "missing_reported_nodes": case_missing,
                    "frame_fit_nodes": case_fit,
                    "max_frame_fit_residual_m": max_case_residual,
                    "max_frame_fit_residual_node": max_residual_node,
                    "passed": case_passed,
                }
            )

    if checked_cases <= 0:
        issue(
            issues,
            "error",
            "geometry_position_proof_no_cases",
            "No geometry position proof cases were checked.",
        )
    if passing_cases <= 0:
        issue(
            issues,
            "error",
            "geometry_position_proof_no_passing_cases",
            "Geometry position proof requires at least one case with QBlade-reported positions and fitted-frame residuals inside tolerance.",
        )

    return {
        "scenarios": requested_scenarios,
        "checked_cases": checked_cases,
        "passing_cases": passing_cases,
        "checked_nodes": checked_nodes,
        "reported_nodes": reported_nodes,
        "frame_fit_nodes": fit_nodes,
        "max_geometry_frame_fit_residual_m": max_residual_m,
        "min_geometry_frame_fit_points": min_fit_points,
        "allow_missing_geometry_reported_nodes": allowed_missing,
        "cases": case_reports,
    }


def stable_support_patch_file_for_case(sim_path: Path) -> Path:
    return turbine_dir_for_sim(sim_path) / "Control" / "vorlap_stable_support_patch.json"


def stable_patch_row_map(rows: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        return {}
    mapped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            keyword = str(row.get("keyword", "")).strip()
            if keyword:
                mapped[keyword] = row
    return mapped


def patch_row_float(row: dict[str, Any], key: str) -> float:
    raw = row.get(key, "")
    return float(raw)


def audit_stable_support_patch_file(
    path: Path,
    settings: dict[str, Any],
    issues: list[AuditIssue],
    *,
    scenario: str,
    case_label: str,
) -> None:
    try:
        manifest = read_json(path)
    except Exception as exc:
        issue(
            issues,
            "error",
            "invalid_stable_support_patch_manifest",
            f"Could not read stable-support patch manifest: {exc}",
            scenario=scenario,
            case=case_label,
        )
        return

    if manifest.get("applied") is not True:
        issue(issues, "error", "stable_support_patch_not_applied", "Stable-support patch manifest is not marked applied.", scenario=scenario, case=case_label)
    if str(manifest.get("scope", "")).strip() != "generated_case_only":
        issue(issues, "error", "stable_support_patch_scope_invalid", "Stable-support patch manifest must be scoped to generated_case_only.", scenario=scenario, case=case_label)

    structural_rows = stable_patch_row_map(manifest.get("structural_patches"))
    sim_rows = stable_patch_row_map(manifest.get("sim_patches"))
    require_subfile_removal = bool(manifest.get("legacy_subfile_removal", True))
    missing_structural = [keyword for keyword in ("SUBFILE",) if keyword not in structural_rows]
    if require_subfile_removal and missing_structural:
        issue(
            issues,
            "error",
            "stable_support_patch_missing_structural_keyword",
            f"Stable-support patch manifest is missing structural keyword(s): {', '.join(missing_structural)}.",
            scenario=scenario,
            case=case_label,
        )
    for keyword in STABLE_SUPPORT_PRESERVED_STRUCTURAL_KEYWORDS:
        row = structural_rows.get(keyword)
        if row is None:
            continue
        old_value = str(row.get("old_value", "")).strip()
        new_value = str(row.get("new_value", "")).strip()
        if old_value != new_value:
            issue(
                issues,
                "error",
                "stable_support_patch_rotor_geometry_modified",
                f"{keyword} should be preserved by the stable-support patch, but changed from {old_value!r} to {new_value!r}.",
                scenario=scenario,
                case=case_label,
            )
    subfile = structural_rows.get("SUBFILE")
    if subfile is not None and str(subfile.get("new_value", "")).strip():
        issue(issues, "error", "stable_support_patch_subfile_not_removed", "SUBFILE should be removed in the generated stable-support patch.", scenario=scenario, case=case_label)

    missing_sim = [keyword for keyword in STABLE_SUPPORT_SIM_KEYWORDS if keyword not in sim_rows]
    if missing_sim:
        issue(
            issues,
            "error",
            "stable_support_patch_missing_sim_keyword",
            f"Stable-support patch manifest is missing simulation keyword(s): {', '.join(missing_sim)}.",
            scenario=scenario,
            case=case_label,
        )

    expected_sim = {
        "RELAXSTEPS": 0.0,
        "RELAXTIMESTEP": float(settings.get("timestep_s", math.nan)),
        "RAMPUP": float(settings.get("rampup_s", 0.0) or 0.0),
        "ADDDAMP": float(settings.get("add_damp_s", 0.0) or 0.0),
    }
    for keyword, expected in expected_sim.items():
        row = sim_rows.get(keyword)
        if row is None:
            continue
        try:
            value = patch_row_float(row, "new_value")
        except (TypeError, ValueError):
            issue(issues, "error", "stable_support_patch_invalid_value", f"{keyword} new_value is not numeric.", scenario=scenario, case=case_label)
            continue
        if not math.isfinite(expected) or not close_enough(value, expected):
            issue(
                issues,
                "error",
                "stable_support_patch_value_mismatch",
                f"{keyword} new_value {value:g} does not match campaign setting {expected:g}.",
                scenario=scenario,
                case=case_label,
            )

    expected_iterations = settings.get("str_iterations")
    if expected_iterations is not None and "STRITERATIONS" not in sim_rows:
        issue(issues, "error", "stable_support_patch_missing_sim_keyword", "Stable-support patch manifest is missing STRITERATIONS.", scenario=scenario, case=case_label)


def audit_plan(settings: dict[str, Any], args: argparse.Namespace, issues: list[AuditIssue]) -> None:
    duration = float(settings.get("duration_s", math.nan))
    store_from = float(settings.get("store_from_s", 0.0) or 0.0)
    retained_duration = duration - store_from
    timestep = float(settings.get("timestep_s", math.nan))
    timesteps = int(settings.get("timesteps", 0) or 0)
    rpm = float(settings.get("rpm", math.nan))
    wind_speeds = settings.get("wind_speeds", [])
    azimuths = settings.get("azimuths_deg", [])
    planned_keys = planned_case_keys(settings)

    if bool(getattr(args, "require_dlc_manifest_row", False)):
        archive_error = str(settings.get("dlc_manifest_row_error", "") or "").strip()
        archive = settings.get("dlc_manifest_row")
        if archive_error:
            issue(
                issues,
                "error",
                "dlc_manifest_row_archive_error",
                f"Could not read the selected DLC manifest-row archive: {archive_error}",
            )
        if not isinstance(archive, dict):
            issue(
                issues,
                "error",
                "missing_dlc_manifest_row",
                "Campaign settings do not contain a selected DLC manifest-row archive.",
            )
        else:
            try:
                manifest_path = Path(str(archive["manifest_path"]))
                manifest = load_dlc_manifest(manifest_path, require_approved=True)
                row_index = int(archive["row_index"])
                row = selected_dlc_row(manifest, row_index)
                if archive.get("manifest_sha256") != manifest["manifest_sha256"]:
                    raise DLCManifestError("Archived manifest_sha256 does not match the approved manifest.")
                if archive.get("row") != row:
                    raise DLCManifestError("Archived selected row does not match the approved manifest row.")
                expected_airfoil_hash = str(manifest["basis"]["airfoil_source_sha256"])
                source_rows = settings.get("airfoil_source_manifest", [])
                naca0018_rows = [
                    source_row
                    for source_row in source_rows
                    if isinstance(source_row, dict) and source_row.get("airfoil_id") == "NACA0018"
                ]
                if len(naca0018_rows) != 1:
                    raise DLCManifestError("Campaign source manifest must contain exactly one NACA0018 row.")
                observed_airfoil_hash = str(naca0018_rows[0].get("sha256", ""))
                if (
                    observed_airfoil_hash != expected_airfoil_hash
                    and not approved_reynolds_envelope_extension_matches(
                        naca0018_rows[0],
                        expected_airfoil_hash,
                    )
                ):
                    raise DLCManifestError(
                        "Campaign NACA0018 source hash does not match the approved DLC manifest "
                        "or its approved Re=10,000 Reynolds-independent envelope extension."
                    )
                actual_speeds = [float(value) for value in settings.get("wind_speeds", [])]
                actual_azimuths = [float(value) for value in settings.get("azimuths_deg", [])]
                if actual_speeds != [float(row["wind_speed_mps"])]:
                    raise DLCManifestError("Campaign wind speed does not match the selected DLC row.")
                if actual_azimuths != [float(row["parked_azimuth_deg"])]:
                    raise DLCManifestError("Campaign azimuth does not match the selected DLC row.")
                if not math.isclose(float(settings.get("rpm", math.nan)), float(row["rpm"])):
                    raise DLCManifestError("Campaign RPM does not match the selected DLC row.")
                if str(settings.get("wind_name", "")) != str(row["wind_file"]):
                    raise DLCManifestError("Campaign wind_name does not match the selected DLC row.")
                if str(Path(str(settings.get("run_root", args.run_root))).resolve()) != str(
                    Path(str(row["run_root"])).resolve()
                ):
                    raise DLCManifestError("Campaign run root does not match the selected DLC row.")
                execution = manifest["execution_settings"]
                if str(settings.get("wake_profile", "")) != str(execution["wake_profile"]):
                    raise DLCManifestError(
                        "Campaign wake profile does not match the approved DLC manifest."
                    )
                if str(Path(str(settings.get("result_filter", ""))).resolve()) != str(
                    Path(str(execution["result_filter_path"])).resolve()
                ):
                    raise DLCManifestError(
                        "Campaign QBlade result-filter path does not match the approved DLC manifest."
                    )
                if str(settings.get("result_filter_sha256", "")) != str(
                    execution["result_filter_sha256"]
                ):
                    raise DLCManifestError(
                        "Campaign QBlade result-filter hash does not match the approved DLC manifest."
                    )
                if str(settings.get("source_frequency_truncation", "")) != str(
                    execution["source_frequency_truncation"]
                ):
                    raise DLCManifestError(
                        "Campaign source-frequency truncation does not match the approved DLC manifest."
                    )
                if str(settings.get("source_bounds_policy", "")) != str(
                    execution["source_bounds_policy"]
                ):
                    raise DLCManifestError(
                        "Campaign source-bounds policy does not match the approved DLC manifest."
                    )
                if bool(getattr(args, "require_wind_record", False)):
                    validate_wind_record(row, manifest)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, DLCManifestError) as exc:
                issue(
                    issues,
                    "error",
                    "invalid_dlc_manifest_row",
                    f"Selected DLC manifest-row provenance is invalid: {exc}",
                )

    if bool(getattr(args, "require_airfoil_source_manifest", False)):
        manifest_error = str(settings.get("airfoil_source_manifest_error", "") or "").strip()
        manifest = settings.get("airfoil_source_manifest")
        if manifest_error:
            issue(
                issues,
                "error",
                "airfoil_source_manifest_generation_failed",
                f"Campaign airfoil-source manifest could not be generated: {manifest_error}",
            )
        if not isinstance(manifest, list) or not manifest:
            issue(
                issues,
                "error",
                "missing_airfoil_source_manifest",
                "Campaign settings do not contain a nonempty airfoil_source_manifest.",
            )
        else:
            for index, row in enumerate(manifest):
                if not isinstance(row, dict):
                    issue(
                        issues,
                        "error",
                        "invalid_airfoil_source_manifest_row",
                        f"Airfoil-source manifest row {index} is not an object.",
                    )
                    continue
                airfoil_id = str(row.get("airfoil_id", "") or "").strip()
                source_file = str(row.get("file", "") or "").strip()
                digest = str(row.get("sha256", "") or "").strip().lower()
                if not airfoil_id or not source_file:
                    issue(
                        issues,
                        "error",
                        "incomplete_airfoil_source_manifest_row",
                        f"Airfoil-source manifest row {index} is missing airfoil_id or file.",
                    )
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    issue(
                        issues,
                        "error",
                        "invalid_airfoil_source_sha256",
                        f"Airfoil-source manifest row {index} does not contain a valid SHA-256 digest.",
                    )
                try:
                    reynolds = [float(value) for value in row.get("reynolds", [])]
                    thickness_ratio = float(row.get("thickness_ratio", math.nan))
                except (TypeError, ValueError):
                    reynolds = []
                    thickness_ratio = math.nan
                if not reynolds or any(not math.isfinite(value) or value <= 0.0 for value in reynolds):
                    issue(
                        issues,
                        "error",
                        "invalid_airfoil_source_reynolds",
                        f"Airfoil-source manifest row {index} has no valid positive Reynolds grid.",
                    )
                if not math.isfinite(thickness_ratio) or thickness_ratio <= 0.0:
                    issue(
                        issues,
                        "error",
                        "invalid_airfoil_source_thickness",
                        f"Airfoil-source manifest row {index} has no valid positive thickness ratio.",
                    )

    if not wind_speeds:
        issue(issues, "error", "missing_wind_speeds", "Campaign settings do not define wind_speeds.")
    if not azimuths:
        issue(issues, "error", "missing_azimuths", "Campaign settings do not define fixed parked azimuths.")
    if not planned_keys:
        issue(issues, "error", "missing_planned_cases", "Campaign settings do not define any planned cases.")
    required_wind_speeds = str(getattr(args, "require_wind_speeds", "") or "").strip()
    production_grid_required = False
    if required_wind_speeds:
        production_grid_required = True
        try:
            expected_wind_speeds = normalized_numeric_set(parse_numeric_grid_spec(required_wind_speeds))
            actual_wind_speeds = normalized_numeric_set(wind_speeds)
        except (TypeError, ValueError) as exc:
            issue(issues, "error", "invalid_required_wind_speed_grid", f"Could not parse required wind-speed grid: {exc}")
        else:
            if actual_wind_speeds != expected_wind_speeds:
                issue(
                    issues,
                    "error",
                    "required_wind_speed_grid_mismatch",
                    "Campaign wind-speed grid does not match required production grid "
                    f"{sorted(expected_wind_speeds)}; found {sorted(actual_wind_speeds)}.",
                )
    required_azimuths = str(getattr(args, "require_azimuths", "") or "").strip()
    if required_azimuths:
        production_grid_required = True
        try:
            expected_azimuths = normalized_numeric_set(parse_numeric_grid_spec(required_azimuths))
            actual_azimuths = normalized_numeric_set(azimuths)
        except (TypeError, ValueError) as exc:
            issue(issues, "error", "invalid_required_azimuth_grid", f"Could not parse required parked-azimuth grid: {exc}")
        else:
            if actual_azimuths != expected_azimuths:
                issue(
                    issues,
                    "error",
                    "required_azimuth_grid_mismatch",
                    "Campaign parked-azimuth grid does not match required production grid "
                    f"{sorted(expected_azimuths)}; found {sorted(actual_azimuths)}.",
                )

    if production_grid_required and settings.get("qblade_steps_override") is not None:
        issue(
            issues,
            "error",
            "qblade_steps_override_in_production_plan",
            "Production-grid audits do not allow qblade_steps_override; unset QBLADE_STEPS or disable REQUIRE_PRODUCTION_MATRIX for smoke/debug runs.",
        )

    expected_count = len(wind_speeds) * len(azimuths)
    if planned_keys and expected_count and len(planned_keys) != expected_count:
        issue(
            issues,
            "error",
            "planned_case_grid_mismatch",
            f"Planned cases ({len(planned_keys)}) do not match wind-speed x azimuth grid ({expected_count}).",
        )
    if int(settings.get("planned_case_count_per_scenario", len(planned_keys)) or 0) != len(planned_keys):
        issue(
            issues,
            "error",
            "planned_case_count_mismatch",
            "planned_case_count_per_scenario does not match the planned case grid.",
        )
    scenarios = expected_scenarios(settings, args)
    expected_total = len(planned_keys) * len(scenarios)
    if int(settings.get("planned_case_count_total", expected_total) or 0) != expected_total:
        issue(
            issues,
            "error",
            "planned_total_count_mismatch",
            "planned_case_count_total does not match scenarios x planned cases.",
        )
    force_scales = settings.get("scenario_force_scales", {})
    couplings = settings.get("scenario_couplings", {})
    if scenarios and not isinstance(force_scales, dict):
        issue(issues, "error", "missing_scenario_force_scales", "scenario_force_scales must map each scenario to an expected force scale.")
    if scenarios and not isinstance(couplings, dict):
        issue(issues, "error", "missing_scenario_couplings", "scenario_couplings must map each scenario to an expected coupling path.")
    if isinstance(force_scales, dict):
        missing = [scenario for scenario in scenarios if scenario not in force_scales and f"{scenario}_force_scale" not in settings]
        if missing:
            issue(issues, "error", "missing_scenario_force_scale", f"Missing expected force scale for scenarios: {', '.join(missing)}.")
    if isinstance(couplings, dict):
        missing = [scenario for scenario in scenarios if scenario not in couplings and f"{scenario}_coupling" not in settings]
        if missing:
            issue(issues, "error", "missing_scenario_coupling", f"Missing expected coupling path for scenarios: {', '.join(missing)}.")

    if not math.isfinite(duration) or duration < args.min_duration:
        issue(issues, "error", "duration_too_short", f"Duration {duration:g} s is below required {args.min_duration:g} s.")
    if not math.isfinite(store_from) or store_from < 0.0 or (
        math.isfinite(duration) and store_from >= duration
    ):
        issue(
            issues,
            "error",
            "invalid_store_from",
            f"Store/exclusion start {store_from:g} s must be finite, non-negative, and below duration {duration:g} s.",
        )
    min_analyzed_duration = float(getattr(args, "min_analyzed_duration", 0.0) or 0.0)
    if (
        min_analyzed_duration > 0.0
        and (
            not math.isfinite(retained_duration)
            or retained_duration < min_analyzed_duration
        )
    ):
        issue(
            issues,
            "error",
            "analyzed_duration_too_short",
            f"Retained duration {retained_duration:g} s (= duration - store_from) "
            f"is below required {min_analyzed_duration:g} s.",
        )
    if not math.isfinite(timestep) or timestep <= 0.0:
        issue(issues, "error", "invalid_timestep", f"Invalid timestep {timestep!r}.")
    elif timestep > args.max_timestep:
        issue(issues, "error", "timestep_too_large", f"Timestep {timestep:g} s exceeds {args.max_timestep:g} s.")
    if math.isfinite(duration) and math.isfinite(timestep) and timestep > 0.0:
        required_steps = int(math.ceil(duration / timestep))
        if timesteps < required_steps:
            issue(
                issues,
                "error",
                "timesteps_too_low",
                f"NUMTIMESTEPS {timesteps} is below ceil(duration/timestep)={required_steps}.",
            )
    if not close_enough(rpm, args.expected_rpm):
        issue(issues, "error", "rpm_not_parked", f"RPM {rpm:g} does not match expected parked rpm {args.expected_rpm:g}.")
    if int(settings.get("rpm_prescribe_type", 1)) != 1:
        issue(
            issues,
            "error",
            "rpm_not_prescribed_for_whole_simulation",
            "Production fixed-speed cases require PRESCRIBETYPE=1 for the whole simulation.",
        )
    if bool(settings.get("quasi_static_third_rev", False)) and not args.allow_quasi_static:
        issue(issues, "error", "quasi_static_sweep", "Campaign is marked as a quasi-static azimuth sweep, not a fixed parked matrix.")
    if args.require_wind_name and not str(settings.get("wind_name", "")).strip():
        issue(issues, "error", "missing_wind_name", "A parked inflow DLC audit requires a nonempty wind_name template.")
    if int(settings.get("campaign_settings_version", 0) or 0) >= 2:
        environment_fields = {
            "horizontal_inflow_angle_deg": False,
            "vertical_inflow_angle_deg": False,
            "shear_exponent": True,
            "reference_height_m": True,
            "air_density_kg_m3": True,
            "air_kinematic_viscosity_m2_s": True,
        }
        environment_values: dict[str, float] = {}
        for field, positive in environment_fields.items():
            raw_value = settings.get(field)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = math.nan
            environment_values[field] = value
            if not math.isfinite(value) or (positive and value <= 0.0):
                issue(
                    issues,
                    "error",
                    "invalid_explicit_environment_setting",
                    f"Campaign setting {field} must be explicitly recorded and "
                    f"{'positive' if positive else 'finite'}; got {raw_value!r}.",
                )
        qblade_ref_height = environment_values["reference_height_m"]
        try:
            turbsim_ref_height = float(settings.get("turbsim_ref_height_m", math.nan))
        except (TypeError, ValueError):
            turbsim_ref_height = math.nan
        if math.isfinite(qblade_ref_height) and math.isfinite(turbsim_ref_height) and not close_enough(
            qblade_ref_height,
            turbsim_ref_height,
        ):
            issue(
                issues,
                "error",
                "qblade_turbsim_reference_height_mismatch",
                f"QBlade REFHEIGHT={qblade_ref_height:g} m differs from TurbSim RefHt={turbsim_ref_height:g} m.",
            )
        for index, record in enumerate(settings.get("turbsim_wind_files", []) or []):
            if not isinstance(record, dict) or bool(record.get("dry_run", False)):
                continue
            metadata_path = Path(str(record.get("metadata_file", "") or ""))
            expected_metadata_hash = str(record.get("metadata_file_sha256", "") or "")
            if not metadata_path.is_file() or not expected_metadata_hash:
                issue(
                    issues,
                    "error",
                    "missing_turbsim_provenance",
                    f"TurbSim record {index} is missing its immutable metadata file/hash: {metadata_path}.",
                )
                continue
            if sha256_file(metadata_path) != expected_metadata_hash:
                issue(
                    issues,
                    "error",
                    "turbsim_provenance_hash_mismatch",
                    f"TurbSim metadata hash no longer matches the campaign record: {metadata_path}.",
                )
                continue
            try:
                turbsim_metadata = read_json(metadata_path)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                issue(
                    issues,
                    "error",
                    "invalid_turbsim_provenance",
                    f"Could not read TurbSim metadata {metadata_path}: {exc}",
                )
                continue
            if turbsim_metadata.get("wind_file_sha256") != record.get("wind_file_sha256"):
                issue(
                    issues,
                    "error",
                    "turbsim_wind_hash_binding_mismatch",
                    f"TurbSim metadata and campaign record bind different wind hashes for record {index}.",
                )
            if turbsim_metadata.get("rendered_input_sha256") != record.get("input_file_sha256"):
                issue(
                    issues,
                    "error",
                    "turbsim_input_hash_binding_mismatch",
                    f"TurbSim metadata and campaign record bind different input hashes for record {index}.",
                )
            provenance_status = str(turbsim_metadata.get("provenance_status", ""))
            if provenance_status not in {"generated", "adopted_existing_exact_input"}:
                issue(
                    issues,
                    "error",
                    "invalid_turbsim_provenance_status",
                    f"TurbSim record {index} has unsupported provenance status {provenance_status!r}.",
                )


def expected_force_scale(settings: dict[str, Any], scenario: str) -> float | None:
    force_scales = settings.get("scenario_force_scales", {})
    if isinstance(force_scales, dict) and scenario in force_scales:
        return float(force_scales[scenario])
    legacy = settings.get(f"{scenario}_force_scale")
    return None if legacy is None else float(legacy)


def expected_coupling(settings: dict[str, Any], scenario: str) -> str | None:
    couplings = settings.get("scenario_couplings", {})
    if isinstance(couplings, dict) and scenario in couplings:
        return str(couplings[scenario])
    legacy = settings.get(f"{scenario}_coupling")
    if legacy is not None:
        return str(legacy)
    if scenario == "vorlap" and settings.get("coupling") is not None:
        return str(settings["coupling"])
    return None


def load_manifest(run_root: Path, scenario: str, issues: list[AuditIssue]) -> dict[str, Any] | None:
    path = run_root / scenario / "cases.json"
    if not path.is_file():
        issue(issues, "error", "missing_manifest", f"Missing case manifest: {path}", scenario=scenario)
        return None
    return read_json(path)


def audit_manifest(
    manifest: dict[str, Any],
    settings: dict[str, Any],
    args: argparse.Namespace,
    issues: list[AuditIssue],
    *,
    scenario: str,
) -> None:
    cases = manifest.get("cases", [])
    keys = planned_case_keys(settings)
    manifest_keys = {
        normalized_case_key(case.get("wind_speed", math.nan), case.get("azimuth_deg", math.nan))
        for case in cases
    }
    if len(cases) != len(keys):
        issue(
            issues,
            "error",
            "manifest_case_count_mismatch",
            f"Manifest has {len(cases)} cases; planned grid has {len(keys)}.",
            scenario=scenario,
        )
    if manifest_keys != keys:
        issue(issues, "error", "manifest_case_grid_mismatch", "Manifest wind-speed/azimuth grid does not match campaign settings.", scenario=scenario)

    manifest_settings = manifest.get("settings", {})
    if not close_enough(float(manifest_settings.get("rpm", math.nan)), args.expected_rpm):
        issue(issues, "error", "manifest_rpm_mismatch", "Manifest rpm is not the expected parked rpm.", scenario=scenario)
    if not close_enough(float(manifest_settings.get("timestep", math.nan)), float(settings.get("timestep_s", math.nan))):
        issue(issues, "error", "manifest_timestep_mismatch", "Manifest timestep does not match campaign settings.", scenario=scenario)
    if int(manifest_settings.get("timesteps", 0) or 0) != int(settings.get("timesteps", 0) or 0):
        issue(issues, "error", "manifest_timesteps_mismatch", "Manifest timesteps do not match campaign settings.", scenario=scenario)

    expected_scale = expected_force_scale(settings, scenario)
    if expected_scale is not None and not close_enough(float(manifest_settings.get("force_scale", math.nan)), float(expected_scale)):
        issue(issues, "error", "manifest_force_scale_mismatch", "Manifest force scale does not match campaign settings.", scenario=scenario)
    expected_mode = expected_coupling(settings, scenario)
    if expected_mode is not None and str(manifest_settings.get("coupling", "")) != expected_mode:
        issue(issues, "error", "manifest_coupling_mismatch", "Manifest coupling mode does not match campaign settings.", scenario=scenario)
    stable_support_patch = bool(settings.get("stable_support_patch", False))
    if "stable_support_patch" in manifest_settings:
        manifest_stable_support_patch = bool(manifest_settings.get("stable_support_patch", False))
        if manifest_stable_support_patch != stable_support_patch:
            issue(issues, "error", "manifest_stable_support_patch_mismatch", "Manifest stable-support patch flag does not match campaign settings.", scenario=scenario)
        stable_support_patch = manifest_stable_support_patch

    labels: set[str] = set()
    for case in cases:
        label = str(case.get("label", ""))
        if not label:
            issue(issues, "error", "missing_case_label", "Manifest case is missing a label.", scenario=scenario)
        elif label in labels:
            issue(issues, "error", "duplicate_case_label", f"Duplicate case label {label}.", scenario=scenario, case=label)
        labels.add(label)
        if args.stage in ("setup", "sign-probes", "results"):
            sim_path = Path(case.get("sim", ""))
            if not sim_path.is_file():
                issue(issues, "error", "missing_sim_file", f"Missing generated QBlade .sim: {case.get('sim')}", scenario=scenario, case=label)
                continue
            geometry_audit = geometry_audit_file_for_case(sim_path)
            if geometry_audit is None or not geometry_audit.is_file():
                issue(
                    issues,
                    "error",
                    "missing_geometry_audit",
                    f"Missing generated geometry/swap audit for case directory: {sim_path.parent}",
                    scenario=scenario,
                    case=label,
                )
            else:
                audit_geometry_file(geometry_audit, issues, scenario=scenario, case_label=label)
            if stable_support_patch:
                try:
                    patch_manifest = stable_support_patch_file_for_case(sim_path)
                except (OSError, ValueError) as exc:
                    issue(
                        issues,
                        "error",
                        "invalid_stable_support_patch_model",
                        f"Could not resolve the model for the stable-support patch: {exc}",
                        scenario=scenario,
                        case=label,
                    )
                    continue
                if not patch_manifest.is_file():
                    issue(
                        issues,
                        "error",
                        "missing_stable_support_patch_manifest",
                        f"Missing generated stable-support patch manifest: {patch_manifest}",
                        scenario=scenario,
                        case=label,
                    )
                else:
                    audit_stable_support_patch_file(patch_manifest, settings, issues, scenario=scenario, case_label=label)


def first_summary_file(result_dir: Path, issues: list[AuditIssue], *, scenario: str, case_label: str) -> Path | None:
    files = sorted(result_dir.glob("*_run_summary.json"))
    if not files:
        issue(issues, "error", "missing_run_summary", f"Missing run summary in {result_dir}.", scenario=scenario, case=case_label)
        return None
    if len(files) > 1:
        issue(issues, "warning", "multiple_run_summaries", f"Found multiple run summaries in {result_dir}; using {files[0].name}.", scenario=scenario, case=case_label)
    return files[0]


def audit_velocity_diagnostics(summary: dict[str, Any], issues: list[AuditIssue], *, scenario: str, case_label: str) -> None:
    diagnostics = summary.get("sil_vorlap_velocity_diagnostics")
    if not diagnostics:
        issue(issues, "warning", "missing_velocity_diagnostics", "SIL run summary has no structural-velocity diagnostics.", scenario=scenario, case=case_label)
        return
    if diagnostics.get("velocity_channel_count", 0) <= 0:
        issue(issues, "error", "no_velocity_channels", "SIL run recorded zero configured structural-velocity channels.", scenario=scenario, case=case_label)
    all_zero_axes = diagnostics.get("all_zero_axes", [])
    if all_zero_axes:
        issue(
            issues,
            "warning",
            "all_zero_velocity_axes",
            f"Structural velocity axes remained all-zero: {', '.join(all_zero_axes)}.",
            scenario=scenario,
            case=case_label,
        )
    all_zero_channels = diagnostics.get("all_zero_channels", [])
    if all_zero_channels:
        issue(
            issues,
            "warning",
            "all_zero_velocity_channels",
            f"{len(all_zero_channels)} structural velocity channels remained all-zero.",
            scenario=scenario,
            case=case_label,
        )


def audit_timing_diagnostics(
    summary: dict[str, Any],
    issues: list[AuditIssue],
    *,
    scenario: str,
    case_label: str,
    completed_steps: int,
) -> None:
    diagnostics = summary.get("sil_vorlap_timing_diagnostics")
    if not isinstance(diagnostics, dict):
        issue(issues, "error", "missing_timing_diagnostics", "SIL run summary has no timing diagnostics.", scenario=scenario, case=case_label)
        return
    if diagnostics.get("time_level") != "pre_advance_explicit":
        issue(
            issues,
            "error",
            "invalid_timing_time_level",
            f"SIL timing diagnostics report unexpected time_level={diagnostics.get('time_level')!r}.",
            scenario=scenario,
            case=case_label,
        )
    samples = int(diagnostics.get("samples", 0) or 0)
    if samples != int(completed_steps):
        issue(
            issues,
            "error",
            "timing_sample_count_mismatch",
            f"SIL timing diagnostics samples {samples} != completed steps {completed_steps}.",
            scenario=scenario,
            case=case_label,
        )
    if diagnostics.get("last_sample") is None:
        issue(issues, "error", "missing_timing_last_sample", "SIL timing diagnostics are missing the last sample.", scenario=scenario, case=case_label)
    first_samples = diagnostics.get("first_samples")
    if not isinstance(first_samples, list) or not first_samples:
        issue(issues, "error", "missing_timing_first_samples", "SIL timing diagnostics are missing first-sample records.", scenario=scenario, case=case_label)
    if int(diagnostics.get("qblade_time_available_count", 0) or 0) + int(diagnostics.get("qblade_time_missing_count", 0) or 0) != samples:
        issue(
            issues,
            "error",
            "timing_qblade_time_count_mismatch",
            "SIL timing diagnostics QBlade-time available/missing counts do not sum to sample count.",
            scenario=scenario,
            case=case_label,
        )
    export_timing = summary.get("qblade_export_time_diagnostics")
    if not isinstance(export_timing, dict) or not bool(export_timing.get("available", False)):
        issue(
            issues,
            "warning",
            "missing_qblade_export_time_diagnostics",
            "Run summary has no parseable exported QBlade Time [s] diagnostics.",
            scenario=scenario,
            case=case_label,
        )


def sil_vorlap_history_file_for_case(case: dict[str, Any], summary: dict[str, Any]) -> Path | None:
    result_dir = Path(str(case.get("result_dir", "")))
    candidates: list[Path] = []
    history = summary.get("sil_vorlap_history")
    if isinstance(history, dict) and history.get("path"):
        path = Path(str(history.get("path")))
        candidates.append(path)
        if not path.is_file():
            candidates.append(result_dir / path.name)
    result_name = str(case.get("result_name", "")).strip()
    if result_name:
        candidates.append(result_dir / f"{result_name}_sil_vorlap_history.csv")
    candidates.extend(sorted(result_dir.glob("*_sil_vorlap_history.csv")))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def force_history_columns_for_component(fieldnames: list[str], component: str) -> list[str]:
    prefix = f"force_{component}_"
    return sorted(
        column
        for column in fieldnames
        if column.startswith(prefix) and re.search(r"_[xyz]_n$", column)
    )


def read_history_time_and_rows(
    path: Path,
    issues: list[AuditIssue],
    *,
    scenario: str,
    case_label: str,
) -> tuple[list[str], list[dict[str, str]], list[float]] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        issue(issues, "error", "spectral_force_history_unreadable", f"Could not read SIL VorLap history {path}: {exc}", scenario=scenario, case=case_label)
        return None

    if not fieldnames:
        issue(issues, "error", "spectral_force_history_empty", f"SIL VorLap history has no header: {path}", scenario=scenario, case=case_label)
        return None
    if "time_s" not in fieldnames:
        issue(issues, "error", "spectral_force_history_missing_time", f"SIL VorLap history has no time_s column: {path}", scenario=scenario, case=case_label)
        return None
    if not rows:
        issue(issues, "error", "spectral_force_history_empty", f"SIL VorLap history has no rows: {path}", scenario=scenario, case=case_label)
        return None

    times: list[float] = []
    for row in rows:
        try:
            time_value = float(row.get("time_s", ""))
        except (TypeError, ValueError):
            issue(issues, "error", "spectral_force_history_invalid_time", f"SIL VorLap history has an invalid time_s value in {path}.", scenario=scenario, case=case_label)
            return None
        if not math.isfinite(time_value):
            issue(issues, "error", "spectral_force_history_nonfinite_time", f"SIL VorLap history has a non-finite time_s value in {path}.", scenario=scenario, case=case_label)
            return None
        times.append(time_value)
    return fieldnames, rows, times


def source_frequency_band(
    row: dict[str, Any],
    issues: list[AuditIssue],
    *,
    scenario: str,
    case_label: str,
    component: str,
) -> tuple[float, float] | None:
    try:
        fmin = float(row.get("freq_min_hz", 0.0) or 0.0)
        fmax = float(row.get("freq_max_hz", 0.0) or 0.0)
    except (TypeError, ValueError):
        issue(issues, "error", "spectral_force_invalid_source_frequency", f"{component} source frequency bounds are not numeric.", scenario=scenario, case=case_label)
        return None
    if not math.isfinite(fmin) or not math.isfinite(fmax):
        issue(issues, "error", "spectral_force_invalid_source_frequency", f"{component} source frequency bounds are not finite.", scenario=scenario, case=case_label)
        return None
    if fmax <= 0.0:
        issue(issues, "error", "spectral_force_invalid_source_frequency", f"{component} has no positive source frequency bandwidth.", scenario=scenario, case=case_label)
        return None
    if fmin > fmax:
        issue(issues, "error", "spectral_force_invalid_source_frequency", f"{component} source frequency min exceeds max.", scenario=scenario, case=case_label)
        return None
    return max(0.0, fmin), fmax


def audit_spectral_force_content(
    settings: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    if not bool(getattr(args, "require_spectral_force_content", False)):
        return None

    requested_scenarios = spectral_force_scenarios(settings, args, issues)
    min_samples = int(getattr(args, "min_spectral_force_samples", 16))
    min_dynamic_power = float(getattr(args, "min_spectral_force_dynamic_power", 1.0e-12))
    min_band_fraction = float(getattr(args, "min_spectral_force_band_power_fraction", 0.95))
    checked_cases = 0
    passing_cases = 0
    checked_components = 0
    passing_components = 0
    checked_force_channels = 0
    case_reports: list[dict[str, object]] = []
    component_reports: list[dict[str, object]] = []

    for scenario in requested_scenarios:
        manifest = manifests.get(scenario)
        if manifest is None:
            issue(
                issues,
                "error",
                "spectral_force_content_missing_manifest",
                "Spectral force-content scenario has no loaded case manifest.",
                scenario=scenario,
            )
            continue
        for case in manifest.get("cases", []):
            case_label = str(case.get("label", ""))
            summary_path = first_summary_file(Path(case.get("result_dir", "")), issues, scenario=scenario, case_label=case_label)
            if summary_path is None:
                continue
            try:
                summary = read_json(summary_path)
            except Exception as exc:
                issue(issues, "error", "spectral_force_summary_unreadable", f"Could not read run summary for spectral-force audit: {summary_path}: {exc}", scenario=scenario, case=case_label)
                continue
            if summary.get("coupling_mode") != "sil-vorlap":
                issue(issues, "error", "spectral_force_requires_sil_vorlap", "Spectral force-content audit requires a SIL VorLap run summary.", scenario=scenario, case=case_label)
                continue
            history_path = sil_vorlap_history_file_for_case(case, summary)
            if history_path is None:
                issue(issues, "error", "missing_spectral_force_history", "Missing recorded SIL VorLap force history CSV.", scenario=scenario, case=case_label)
                continue
            history_payload = read_history_time_and_rows(history_path, issues, scenario=scenario, case_label=case_label)
            if history_payload is None:
                continue
            fieldnames, rows, times = history_payload
            if len(rows) < min_samples:
                issue(
                    issues,
                    "error",
                    "spectral_force_history_too_short",
                    f"SIL VorLap history has {len(rows)} rows, below required {min_samples}.",
                    scenario=scenario,
                    case=case_label,
                )
                continue
            source_rows = summary.get("sil_vorlap_source_coverage")
            if not isinstance(source_rows, list) or not source_rows:
                issue(issues, "error", "missing_spectral_force_source_coverage", "Run summary has no SIL VorLap source-coverage rows for spectral-force audit.", scenario=scenario, case=case_label)
                continue

            checked_cases += 1
            case_errors_before = sum(1 for item in issues if item.severity == "error")
            case_components = 0
            case_passing_components = 0
            case_force_channels = 0
            for source_row in source_rows:
                if not isinstance(source_row, dict):
                    issue(issues, "error", "spectral_force_source_coverage_schema_mismatch", "SIL VorLap source-coverage row is not an object.", scenario=scenario, case=case_label)
                    continue
                component = str(source_row.get("component", "")).strip()
                if not component:
                    issue(issues, "error", "spectral_force_source_coverage_schema_mismatch", "SIL VorLap source-coverage row has no component.", scenario=scenario, case=case_label)
                    continue
                band = source_frequency_band(source_row, issues, scenario=scenario, case_label=case_label, component=component)
                if band is None:
                    continue
                force_columns = force_history_columns_for_component(fieldnames, component)
                checked_force_channels += len(force_columns)
                case_force_channels += len(force_columns)
                if not force_columns:
                    issue(issues, "error", "missing_spectral_force_component_history", f"No force history columns found for component {component}.", scenario=scenario, case=case_label)
                    continue

                component_errors_before = sum(1 for item in issues if item.severity == "error")
                active_channels = 0
                failed_channels: list[tuple[str, dict[str, object]]] = []
                strongest_channel = ""
                strongest_power = -math.inf
                strongest_peak: dict[str, object] | None = None
                for column in force_columns:
                    try:
                        values = [float(row.get(column, "")) for row in rows]
                    except (TypeError, ValueError):
                        issue(issues, "error", "spectral_force_history_invalid_value", f"Force history column {column} contains a non-numeric value.", scenario=scenario, case=case_label)
                        continue
                    try:
                        diagnostics = verify_frequency_band_spectral_content(
                            times,
                            values,
                            min_frequency_hz=band[0],
                            max_frequency_hz=band[1],
                            min_band_power_fraction=min_band_fraction,
                            min_total_power=min_dynamic_power,
                        )
                    except ValueError as exc:
                        issue(issues, "error", "spectral_force_history_invalid_signal", f"Could not analyze force history column {column}: {exc}", scenario=scenario, case=case_label)
                        continue
                    dynamic_power = float(diagnostics.get("total_dynamic_power", 0.0) or 0.0)
                    if dynamic_power <= min_dynamic_power:
                        continue
                    active_channels += 1
                    if dynamic_power > strongest_power:
                        strongest_power = dynamic_power
                        strongest_channel = column
                        strongest_peak = diagnostics.get("dominant_peak") if isinstance(diagnostics.get("dominant_peak"), dict) else None
                    if not bool(diagnostics.get("passed", False)):
                        failed_channels.append((column, diagnostics))

                component_has_errors = sum(1 for item in issues if item.severity == "error") > component_errors_before
                component_passed = active_channels > 0 and not failed_channels and not component_has_errors
                case_components += 1
                if component_passed:
                    passing_components += 1
                    case_passing_components += 1
                component_reports.append(
                    {
                        "scenario": scenario,
                        "case": case_label,
                        "component": component,
                        "force_columns": len(force_columns),
                        "active_force_channels": active_channels,
                        "source_freq_min_hz": band[0],
                        "source_freq_max_hz": band[1],
                        "strongest_channel": strongest_channel,
                        "strongest_dynamic_power": strongest_power if strongest_power > 0.0 else 0.0,
                        "strongest_peak": strongest_peak,
                        "passed": component_passed,
                    }
                )
                checked_components += 1
                if active_channels == 0:
                    issue(issues, "error", "spectral_force_no_dynamic_force_content", f"{component} has no active dynamic force channels above {min_dynamic_power:g} power.", scenario=scenario, case=case_label)
                for column, diagnostics in failed_channels:
                    dominant = diagnostics.get("dominant_peak")
                    dominant_text = ""
                    if isinstance(dominant, dict):
                        dominant_text = f" Dominant peak: {float(dominant.get('frequency_hz', math.nan)):g} Hz."
                    issue(
                        issues,
                        "error",
                        "spectral_force_content_out_of_source_band",
                        (
                            f"{component} force history column {column} has only "
                            f"{float(diagnostics.get('band_power_fraction', 0.0)):g} of dynamic power inside "
                            f"[{band[0]:g}, {band[1]:g}] Hz.{dominant_text}"
                        ),
                        scenario=scenario,
                        case=case_label,
                    )

            case_has_errors = sum(1 for item in issues if item.severity == "error") > case_errors_before
            case_passed = (
                not case_has_errors
                and case_components > 0
                and case_passing_components == case_components
                and case_force_channels > 0
            )
            if case_passed:
                passing_cases += 1
            case_reports.append(
                {
                    "scenario": scenario,
                    "case": case_label,
                    "components": case_components,
                    "passing_components": case_passing_components,
                    "force_channels": case_force_channels,
                    "passed": case_passed,
                }
            )

    if checked_cases <= 0:
        issue(
            issues,
            "error",
            "spectral_force_content_no_cases",
            "No spectral force-content cases were checked.",
        )
    if checked_components <= 0:
        issue(
            issues,
            "error",
            "spectral_force_content_no_components",
            "Spectral force-content proof requires at least one checked source-coverage component.",
        )
    if checked_force_channels <= 0:
        issue(
            issues,
            "error",
            "spectral_force_content_no_force_channels",
            "Spectral force-content proof requires at least one recorded SIL VorLap force-history channel.",
        )
    if passing_cases <= 0:
        issue(
            issues,
            "error",
            "spectral_force_content_no_passing_cases",
            "Spectral force-content proof requires at least one case whose active force channels stay inside source-frequency bands.",
        )

    return {
        "scenarios": requested_scenarios,
        "checked_cases": checked_cases,
        "passing_cases": passing_cases,
        "checked_components": checked_components,
        "passing_components": passing_components,
        "checked_force_channels": checked_force_channels,
        "min_samples": min_samples,
        "min_dynamic_power": min_dynamic_power,
        "min_band_power_fraction": min_band_fraction,
        "cases": case_reports,
        "components": component_reports,
    }


def spectral_psd_comparison_scenarios(
    settings: dict[str, Any],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> list[str]:
    default = "vorlap"
    scenarios = parse_scenarios(getattr(args, "spectral_psd_comparison_scenarios", default))
    known = set(expected_scenarios(settings, args))
    unknown = sorted(set(scenarios) - known)
    if unknown:
        issue(
            issues,
            "error",
            "unknown_spectral_psd_comparison_scenario",
            f"Spectral PSD comparison scenarios are not in the campaign plan: {', '.join(unknown)}.",
        )
    return [scenario for scenario in scenarios if scenario in known]


def spectral_psd_passed(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "ok"}


def audit_spectral_psd_comparison(
    settings: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    if not bool(getattr(args, "require_spectral_psd_comparison", False)):
        return None

    path = spectral_psd_comparison_path(args)
    requested_scenarios = spectral_psd_comparison_scenarios(settings, args, issues)
    max_frequency_error = float(getattr(args, "max_spectral_psd_frequency_error_hz", 0.05))
    min_band_fraction = float(getattr(args, "min_spectral_psd_band_power_fraction", 0.80))
    expected_cases = {
        (scenario, str(case.get("label", "")))
        for scenario in requested_scenarios
        for case in manifests.get(scenario, {}).get("cases", [])
    }
    missing_manifests = [scenario for scenario in requested_scenarios if scenario not in manifests]
    for scenario in missing_manifests:
        issue(
            issues,
            "error",
            "spectral_psd_comparison_missing_manifest",
            "Spectral PSD comparison scenario has no loaded case manifest.",
            scenario=scenario,
        )

    if not path.is_file():
        issue(
            issues,
            "error",
            "missing_spectral_psd_comparison",
            f"Missing spectral PSD comparison CSV: {path}",
        )
        return {
            "path": str(path),
            "scenarios": requested_scenarios,
            "expected_cases": len(expected_cases),
            "rows": 0,
            "checked_groups": 0,
            "complete_cases": 0,
            "required_sources": list(SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES),
            "max_frequency_error_hz": max_frequency_error,
            "min_band_power_fraction": min_band_fraction,
        }

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        issue(issues, "error", "spectral_psd_comparison_unreadable", f"Could not read spectral PSD comparison CSV {path}: {exc}")
        return {"path": str(path), "scenarios": requested_scenarios, "rows": 0}

    missing_columns = sorted(set(SPECTRAL_PSD_COMPARISON_REQUIRED_COLUMNS) - set(fieldnames))
    if missing_columns:
        issue(
            issues,
            "error",
            "spectral_psd_comparison_missing_columns",
            f"Spectral PSD comparison CSV is missing required columns: {', '.join(missing_columns)}.",
        )
        return {
            "path": str(path),
            "scenarios": requested_scenarios,
            "rows": len(rows),
            "checked_groups": 0,
            "complete_cases": 0,
            "required_sources": list(SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES),
            "max_frequency_error_hz": max_frequency_error,
            "min_band_power_fraction": min_band_fraction,
        }

    requested_set = set(requested_scenarios)
    groups: dict[tuple[str, str, str, str], set[str]] = {}
    row_cases: set[tuple[str, str]] = set()
    valid_rows = 0
    for index, row in enumerate(rows, start=2):
        scenario = str(row.get("scenario", "")).strip()
        case = str(row.get("case", "")).strip()
        if scenario not in requested_set:
            continue
        row_cases.add((scenario, case))
        component = str(row.get("component", "")).strip()
        channel = str(row.get("channel", "")).strip()
        source = str(row.get("source", "")).strip().lower()
        artifact = str(row.get("artifact", "")).strip()
        if (scenario, case) not in expected_cases:
            issue(
                issues,
                "error",
                "unknown_spectral_psd_comparison_case",
                f"Spectral PSD comparison row {index} references an unknown expected case.",
                scenario=scenario,
                case=case,
            )
        if not component or not channel:
            issue(
                issues,
                "error",
                "spectral_psd_comparison_missing_identity",
                f"Spectral PSD comparison row {index} is missing component or channel.",
                scenario=scenario,
                case=case,
            )
            continue
        if source not in SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES:
            issue(
                issues,
                "error",
                "spectral_psd_comparison_invalid_source",
                f"Spectral PSD comparison row {index} has unsupported source {source!r}.",
                scenario=scenario,
                case=case,
            )
            continue
        if not artifact:
            issue(
                issues,
                "error",
                "spectral_psd_comparison_missing_artifact",
                f"Spectral PSD comparison row {index} has no provenance artifact.",
                scenario=scenario,
                case=case,
            )

        try:
            expected_frequency = float(row.get("expected_frequency_hz", ""))
            peak_frequency = float(row.get("peak_frequency_hz", ""))
            frequency_error = float(row.get("frequency_error_hz", ""))
            band_fraction = float(row.get("band_power_fraction", ""))
            peak_power = float(row.get("peak_power", ""))
        except (TypeError, ValueError):
            issue(
                issues,
                "error",
                "spectral_psd_comparison_invalid_numeric",
                f"Spectral PSD comparison row {index} contains a nonnumeric frequency, power, or band fraction.",
                scenario=scenario,
                case=case,
            )
            continue
        if not all(
            math.isfinite(value)
            for value in (expected_frequency, peak_frequency, frequency_error, band_fraction, peak_power)
        ):
            issue(
                issues,
                "error",
                "spectral_psd_comparison_nonfinite_numeric",
                f"Spectral PSD comparison row {index} contains a nonfinite frequency, power, or band fraction.",
                scenario=scenario,
                case=case,
            )
            continue
        if expected_frequency <= 0.0 or peak_frequency <= 0.0 or frequency_error < 0.0:
            issue(
                issues,
                "error",
                "spectral_psd_comparison_invalid_frequency",
                f"Spectral PSD comparison row {index} must have positive frequencies and nonnegative frequency error.",
                scenario=scenario,
                case=case,
            )
        if peak_power < 0.0 or band_fraction < 0.0 or band_fraction > 1.0:
            issue(
                issues,
                "error",
                "spectral_psd_comparison_invalid_power",
                f"Spectral PSD comparison row {index} must have nonnegative power and 0<=band_power_fraction<=1.",
                scenario=scenario,
                case=case,
            )
        expected_error = abs(peak_frequency - expected_frequency)
        if not close_enough(frequency_error, expected_error, rel=1.0e-6, abs_tol=1.0e-9):
            issue(
                issues,
                "error",
                "spectral_psd_comparison_error_mismatch",
                f"Spectral PSD comparison row {index} frequency_error_hz does not match |peak-expected|.",
                scenario=scenario,
                case=case,
            )
        if frequency_error > max_frequency_error:
            issue(
                issues,
                "error",
                "spectral_psd_comparison_frequency_error_exceeds_tolerance",
                f"Spectral PSD comparison row {index} frequency error {frequency_error:g} Hz exceeds {max_frequency_error:g} Hz.",
                scenario=scenario,
                case=case,
            )
        if band_fraction < min_band_fraction:
            issue(
                issues,
                "error",
                "spectral_psd_comparison_band_power_too_low",
                f"Spectral PSD comparison row {index} band-power fraction {band_fraction:g} is below {min_band_fraction:g}.",
                scenario=scenario,
                case=case,
            )
        if not spectral_psd_passed(row.get("passed", "")):
            issue(
                issues,
                "error",
                "spectral_psd_comparison_not_passed",
                f"Spectral PSD comparison row {index} is not marked passed.",
                scenario=scenario,
                case=case,
            )

        groups.setdefault((scenario, case, component, channel), set()).add(source)
        valid_rows += 1

    complete_cases: set[tuple[str, str]] = set()
    required_sources = set(SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES)
    for (scenario, case, component, channel), sources in sorted(groups.items()):
        missing_sources = sorted(required_sources - sources)
        if missing_sources:
            issue(
                issues,
                "error",
                "spectral_psd_comparison_missing_source",
                (
                    f"Spectral PSD comparison for {component}/{channel} is missing required source(s): "
                    f"{', '.join(missing_sources)}."
                ),
                scenario=scenario,
                case=case,
            )
        else:
            if (scenario, case) in expected_cases:
                complete_cases.add((scenario, case))

    for scenario, case in sorted(expected_cases - row_cases):
        issue(
            issues,
            "error",
            "missing_spectral_psd_comparison_case",
            "Spectral PSD comparison CSV has no row for this expected case.",
            scenario=scenario,
            case=case,
        )
    for scenario, case in sorted(expected_cases - complete_cases):
        issue(
            issues,
            "error",
            "missing_complete_spectral_psd_comparison_group",
            "Expected case has no complete analytic/offline/SIL/QBlade PSD comparison group.",
            scenario=scenario,
            case=case,
        )
    if not expected_cases:
        issue(
            issues,
            "error",
            "spectral_psd_comparison_no_expected_cases",
            "Spectral PSD comparison proof requires at least one expected case from a loaded scenario manifest.",
        )
    if not groups:
        issue(
            issues,
            "error",
            "spectral_psd_comparison_no_groups",
            "Spectral PSD comparison proof requires at least one checked analytic/offline/SIL/QBlade comparison group.",
        )
    if not complete_cases:
        issue(
            issues,
            "error",
            "spectral_psd_comparison_no_complete_cases",
            "Spectral PSD comparison proof requires at least one complete passing case.",
        )

    return {
        "path": str(path),
        "scenarios": requested_scenarios,
        "missing_manifests": missing_manifests,
        "rows": len(rows),
        "valid_rows": valid_rows,
        "checked_groups": len(groups),
        "expected_cases": len(expected_cases),
        "complete_cases": len(complete_cases),
        "required_sources": list(SPECTRAL_PSD_COMPARISON_REQUIRED_SOURCES),
        "max_frequency_error_hz": max_frequency_error,
        "min_band_power_fraction": min_band_fraction,
    }


def native_update_log_lower_bound(completed_steps: int) -> int:
    completed = max(0, int(completed_steps))
    if completed <= 10:
        return completed
    if completed % 100 == 0:
        return completed
    return max(10, completed - completed % 100)


def audit_native_callback_evidence(
    evidence: Any,
    issues: list[AuditIssue],
    *,
    scenario: str,
    case_label: str,
    completed_steps: int,
) -> None:
    if not isinstance(evidence, dict):
        issue(issues, "error", "missing_native_callback_evidence", "qblade-external run summary is missing native callback evidence.", scenario=scenario, case=case_label)
        return
    if evidence.get("callback_block_present") is False:
        issue(
            issues,
            "error",
            "native_callback_block_missing",
            "QBlade turbine definition is missing a complete LIBFILE_1/LIBFUNCTION_1/LIBARRAYSIZE_1/LIBPARAMETERFILE_1 callback block.",
            scenario=scenario,
            case=case_label,
        )
    if evidence.get("library_file_exists") is False:
        candidates = evidence.get("library_candidate_names")
        search_roots = evidence.get("library_search_roots")
        issue(
            issues,
            "error",
            "native_library_missing",
            "The qblade-external callback block names a library that was not found under QBlade ControllerFiles or the QBlade root. "
            f"Configured library={evidence.get('library_stem')!r}; candidates={candidates!r}; search_roots={search_roots!r}.",
            scenario=scenario,
            case=case_label,
        )
    if bool(evidence.get("library_symbol_check_available", False)) and evidence.get("library_missing_required_symbols"):
        issue(
            issues,
            "error",
            "native_library_missing_required_symbols",
            (
                "The qblade-external callback library was found, but it does not export all symbols "
                f"QBlade expects from LIBFUNCTION_1={evidence.get('function_name')!r}: "
                f"{evidence.get('library_missing_required_symbols')!r}."
            ),
            scenario=scenario,
            case=case_label,
        )
    if not bool(evidence.get("update_init_seen", False)):
        issue(issues, "error", "native_update_init_missing", "Native bridge log does not show update_init.", scenario=scenario, case=case_label)
    try:
        observed_updates = int(evidence.get("update_count_observed_max", 0) or 0)
    except (TypeError, ValueError):
        observed_updates = 0
        issue(
            issues,
            "error",
            "native_update_count_unreadable",
            f"Native bridge log observed update count is not an integer: {evidence.get('update_count_observed_max')!r}.",
            scenario=scenario,
            case=case_label,
        )
    required_observed = native_update_log_lower_bound(completed_steps)
    if observed_updates < required_observed:
        issue(
            issues,
            "error",
            "native_update_count_missing",
            f"Native bridge log observed update count {observed_updates}, expected at least {required_observed}.",
            scenario=scenario,
            case=case_label,
        )
    if not bool(evidence.get("runtime_initialized_message_seen", False)):
        issue(issues, "error", "native_runtime_init_missing", "Native bridge log does not show VorLap runtime initialization.", scenario=scenario, case=case_label)
    if evidence.get("python_env_parse_errors"):
        issue(
            issues,
            "error",
            "native_python_env_diagnostics_unreadable",
            "Native bridge Python-environment diagnostics are present but could not be parsed as JSON.",
            scenario=scenario,
            case=case_label,
        )
    if not bool(evidence.get("python_env_before_import_seen", False)):
        issue(
            issues,
            "error",
            "native_python_env_before_import_missing",
            "Native bridge log does not show embedded-Python sys.path diagnostics before importing VorLap.",
            scenario=scenario,
            case=case_label,
        )
    if not bool(evidence.get("python_env_after_import_seen", False)):
        issue(
            issues,
            "error",
            "native_python_env_after_import_missing",
            "Native bridge log does not show embedded-Python diagnostics after importing VorLap.",
            scenario=scenario,
            case=case_label,
        )
    if not evidence.get("numpy_version") or not evidence.get("numpy_file"):
        issue(
            issues,
            "error",
            "native_numpy_diagnostics_missing",
            "Native bridge log does not identify the NumPy version and module path used by embedded Python.",
            scenario=scenario,
            case=case_label,
        )
    if not evidence.get("vorlap_qblade_runtime_file"):
        issue(
            issues,
            "error",
            "native_runtime_path_missing",
            "Native bridge log does not identify the imported vorlap.qblade_runtime module path.",
            scenario=scenario,
            case=case_label,
        )


def native_callback_proof_scenarios(
    settings: dict[str, Any],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> list[str]:
    default = "vorlap"
    scenarios = parse_scenarios(getattr(args, "native_callback_proof_scenarios", default))
    known = set(expected_scenarios(settings, args))
    unknown = sorted(set(scenarios) - known)
    if unknown:
        issue(
            issues,
            "error",
            "unknown_native_callback_proof_scenario",
            f"Native callback proof scenarios are not in the campaign plan: {', '.join(unknown)}.",
        )
    return [scenario for scenario in scenarios if scenario in known]


def native_callback_evidence_passes(evidence: Any, completed_steps: int) -> bool:
    scratch: list[AuditIssue] = []
    audit_native_callback_evidence(
        evidence,
        scratch,
        scenario="",
        case_label="",
        completed_steps=completed_steps,
    )
    return not [item for item in scratch if item.severity == "error"]


def audit_native_callback_proof(
    settings: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    if not bool(getattr(args, "require_native_callback_proof", False)):
        return None

    requested_scenarios = native_callback_proof_scenarios(settings, args, issues)
    checked_cases = 0
    qblade_external_cases = 0
    passing_cases = 0
    case_reports: list[dict[str, object]] = []

    for scenario in requested_scenarios:
        manifest = manifests.get(scenario)
        if manifest is None:
            issue(
                issues,
                "error",
                "native_callback_proof_missing_manifest",
                "Native callback proof scenario has no loaded case manifest.",
                scenario=scenario,
            )
            continue
        for case in manifest.get("cases", []):
            case_label = str(case.get("label", ""))
            summary_path = first_summary_file(Path(case.get("result_dir", "")), issues, scenario=scenario, case_label=case_label)
            if summary_path is None:
                continue
            try:
                summary = read_json(summary_path)
            except Exception as exc:
                issue(
                    issues,
                    "error",
                    "native_callback_proof_summary_unreadable",
                    f"Could not read native callback proof run summary: {summary_path}: {exc}",
                    scenario=scenario,
                    case=case_label,
                )
                continue
            checked_cases += 1
            coupling_mode = str(summary.get("coupling_mode", "")).strip()
            completed_steps = int(summary.get("completed_steps", 0) or 0)
            is_qblade_external = coupling_mode == "qblade-external"
            if is_qblade_external:
                qblade_external_cases += 1
            else:
                issue(
                    issues,
                    "error",
                    "native_callback_proof_requires_qblade_external",
                    f"Native callback proof requires coupling_mode='qblade-external', got {coupling_mode!r}.",
                    scenario=scenario,
                    case=case_label,
                )
            evidence_passed = is_qblade_external and completed_steps > 0 and native_callback_evidence_passes(
                summary.get("native_callback_evidence"),
                completed_steps,
            )
            if evidence_passed:
                passing_cases += 1
            case_reports.append(
                {
                    "scenario": scenario,
                    "case": case_label,
                    "summary": str(summary_path),
                    "coupling_mode": coupling_mode,
                    "completed_steps": completed_steps,
                    "update_init_seen": bool(
                        isinstance(summary.get("native_callback_evidence"), dict)
                        and summary["native_callback_evidence"].get("update_init_seen", False)
                    ),
                    "update_count_observed_max": (
                        summary.get("native_callback_evidence", {}).get("update_count_observed_max")
                        if isinstance(summary.get("native_callback_evidence"), dict)
                        else None
                    ),
                    "passed": evidence_passed,
                }
            )

    if checked_cases <= 0:
        issue(
            issues,
            "error",
            "native_callback_proof_no_cases",
            "No native callback proof cases were checked.",
        )
    if qblade_external_cases <= 0:
        issue(
            issues,
            "error",
            "native_callback_proof_no_qblade_external_cases",
            "Native callback proof requires at least one qblade-external case.",
        )
    if passing_cases <= 0:
        issue(
            issues,
            "error",
            "native_callback_proof_no_passing_cases",
            "Native callback proof requires at least one case with update_init and sampled update evidence.",
        )

    return {
        "scenarios": requested_scenarios,
        "checked_cases": checked_cases,
        "qblade_external_cases": qblade_external_cases,
        "passing_cases": passing_cases,
        "cases": case_reports,
    }


def _recorded_artifact_path(raw_path: object, *, base_dir: Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _recorded_int(value: object, *, default: int = -1) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def audit_checkpoint_and_project_artifacts(
    summary: dict[str, Any],
    settings: dict[str, Any],
    args: argparse.Namespace,
    issues: list[AuditIssue],
    *,
    result_dir: Path,
    scenario: str,
    case_label: str,
) -> None:
    """Validate restart and completed-project evidence requested by a campaign."""
    if bool(settings.get("store_project", False)):
        completed_project = _recorded_artifact_path(summary.get("stored_project"), base_dir=result_dir)
        if completed_project is None or not completed_project.is_file() or completed_project.stat().st_size <= 0:
            issue(
                issues,
                "error",
                "missing_completed_qblade_project",
                f"Campaign requested a completed QPR, but it is missing: {completed_project}.",
                scenario=scenario,
                case=case_label,
            )

    try:
        checkpoint_every = int(settings.get("checkpoint_every", 0) or 0)
    except (TypeError, ValueError):
        checkpoint_every = 0
    completed_steps = int(summary.get("completed_steps", 0) or 0)
    if checkpoint_every <= 0 or completed_steps <= 0:
        return

    restart = summary.get("restart")
    if not isinstance(restart, dict):
        issue(
            issues,
            "error",
            "missing_checkpoint_summary",
            "Checkpointing was requested, but the run summary has no restart evidence.",
            scenario=scenario,
            case=case_label,
        )
        return

    try:
        observed_checkpoint_every = int(restart.get("checkpoint_every", 0) or 0)
    except (TypeError, ValueError):
        observed_checkpoint_every = -1
    if observed_checkpoint_every != checkpoint_every:
        issue(
            issues,
            "error",
            "checkpoint_interval_mismatch",
            f"Run summary checkpoint_every={restart.get('checkpoint_every')!r}, expected {checkpoint_every}.",
            scenario=scenario,
            case=case_label,
        )
    if bool(getattr(args, "require_fresh_run", False)):
        fresh_violations = []
        # Auto-resume is deliberately enabled whenever checkpointing is on.
        # Freshness is proven by whether a checkpoint was actually loaded or
        # rejected, not by the presence of that harmless request flag.
        if restart.get("resume_checkpoint") is not None:
            fresh_violations.append("resume_checkpoint")
        if _recorded_int(restart.get("resume_start_step", 0)) != 0:
            fresh_violations.append("resume_start_step")
        if restart.get("resume_rejection") is not None:
            fresh_violations.append("resume_rejection")
        if fresh_violations:
            issue(
                issues,
                "error",
                "run_not_fresh",
                f"Fresh-root acceptance run contains restart state: {', '.join(fresh_violations)}.",
                scenario=scenario,
                case=case_label,
            )

    checkpoint_dir = _recorded_artifact_path(restart.get("checkpoint_dir"), base_dir=result_dir)
    metadata_path = _recorded_artifact_path(restart.get("latest_checkpoint"), base_dir=result_dir)
    project_path = _recorded_artifact_path(restart.get("latest_checkpoint_project"), base_dir=result_dir)
    if checkpoint_dir is None or not checkpoint_dir.is_dir():
        issue(
            issues,
            "error",
            "missing_checkpoint_directory",
            f"Checkpoint directory is missing: {checkpoint_dir}.",
            scenario=scenario,
            case=case_label,
        )
        return
    if metadata_path is None or not metadata_path.is_file():
        issue(
            issues,
            "error",
            "missing_checkpoint_metadata",
            f"Latest checkpoint metadata is missing: {metadata_path}.",
            scenario=scenario,
            case=case_label,
        )
        return
    if project_path is None or not project_path.is_file() or project_path.stat().st_size <= 0:
        issue(
            issues,
            "error",
            "missing_checkpoint_project",
            f"Latest checkpoint QPR is missing: {project_path}.",
            scenario=scenario,
            case=case_label,
        )

    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        issue(
            issues,
            "error",
            "invalid_checkpoint_metadata",
            f"Could not read checkpoint metadata {metadata_path}: {exc}",
            scenario=scenario,
            case=case_label,
        )
        return

    metadata_steps = _recorded_int(metadata.get("completed_steps"))
    restart_steps = _recorded_int(restart.get("latest_checkpoint_completed_steps"))
    if metadata_steps != completed_steps or restart_steps != completed_steps:
        issue(
            issues,
            "error",
            "checkpoint_completed_steps_mismatch",
            f"Latest checkpoint steps metadata={metadata_steps}, summary={restart_steps}, "
            f"completed run={completed_steps}.",
            scenario=scenario,
            case=case_label,
        )
    if _recorded_int(metadata.get("version")) < 3:
        issue(
            issues,
            "error",
            "legacy_checkpoint_metadata",
            f"Checkpoint {metadata_path} predates identity- and QPR-hash-aware metadata.",
            scenario=scenario,
            case=case_label,
        )
    expected_project_hash = str(metadata.get("qblade_project_sha256", "") or "")
    expected_project_size = _recorded_int(metadata.get("qblade_project_size_bytes"), default=0)
    if not expected_project_hash or expected_project_size <= 0:
        issue(
            issues,
            "error",
            "missing_checkpoint_project_binding",
            f"Checkpoint {metadata_path} does not record its QPR SHA-256 and size.",
            scenario=scenario,
            case=case_label,
        )
    elif project_path is not None and project_path.is_file():
        observed_project_size = project_path.stat().st_size
        observed_project_hash = sha256_file(project_path) if observed_project_size > 0 else ""
        if observed_project_size != expected_project_size or observed_project_hash != expected_project_hash:
            issue(
                issues,
                "error",
                "checkpoint_project_binding_mismatch",
                f"Checkpoint QPR {project_path} no longer matches its recorded hash/size.",
                scenario=scenario,
                case=case_label,
            )
    if _recorded_int(metadata.get("requested_steps")) != _recorded_int(summary.get("requested_steps")):
        issue(
            issues,
            "error",
            "checkpoint_requested_steps_mismatch",
            f"Checkpoint {metadata_path} requested_steps does not match the run summary.",
            scenario=scenario,
            case=case_label,
        )
    if str(metadata.get("coupling_mode", "")) != str(summary.get("coupling_mode", "")):
        issue(
            issues,
            "error",
            "checkpoint_coupling_mismatch",
            f"Checkpoint {metadata_path} coupling mode does not match the run summary.",
            scenario=scenario,
            case=case_label,
        )

    summary_fingerprint = str(restart.get("checkpoint_identity_fingerprint", "") or "")
    metadata_fingerprint = str(metadata.get("checkpoint_identity_fingerprint", "") or "")
    embedded_identity = metadata.get("checkpoint_identity")
    embedded_fingerprint = (
        str(embedded_identity.get("fingerprint", "") or "")
        if isinstance(embedded_identity, dict)
        else ""
    )
    if (
        not summary_fingerprint
        or metadata_fingerprint != summary_fingerprint
        or embedded_fingerprint != summary_fingerprint
    ):
        issue(
            issues,
            "error",
            "checkpoint_identity_mismatch",
            "Checkpoint identity fingerprints are missing or disagree between the run summary and metadata.",
            scenario=scenario,
            case=case_label,
        )
    export_only_recovery = bool(restart.get("completed_checkpoint_export_only", False))
    reproducibility = summary.get("reproducibility")
    if isinstance(embedded_identity, dict) and isinstance(reproducibility, dict):
        identity_fields = ["runtime_asset_fingerprint"]
        if not export_only_recovery:
            identity_fields.append("vorlap_python_runtime_fingerprint")
        for identity_field in identity_fields:
            if embedded_identity.get(identity_field) != reproducibility.get(identity_field):
                issue(
                    issues,
                    "error",
                    "checkpoint_runtime_identity_mismatch",
                    f"Checkpoint {identity_field} does not match the implementation/assets reported after the run.",
                    scenario=scenario,
                    case=case_label,
                )

    pointer_path = checkpoint_dir / "latest_checkpoint.json"
    if not pointer_path.is_file():
        issue(
            issues,
            "error",
            "missing_latest_checkpoint_pointer",
            f"Latest checkpoint pointer is missing: {pointer_path}.",
            scenario=scenario,
            case=case_label,
        )
    else:
        try:
            pointer = read_json(pointer_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issue(
                issues,
                "error",
                "invalid_latest_checkpoint_pointer",
                f"Could not read latest checkpoint pointer {pointer_path}: {exc}",
                scenario=scenario,
                case=case_label,
            )
        else:
            if (
                _recorded_int(pointer.get("completed_steps")) != completed_steps
                or str(pointer.get("checkpoint_identity_fingerprint", "") or "") != summary_fingerprint
                or str(pointer.get("qblade_project_sha256", "") or "") != expected_project_hash
                or _recorded_int(pointer.get("qblade_project_size_bytes"), default=0) != expected_project_size
            ):
                issue(
                    issues,
                    "error",
                    "latest_checkpoint_pointer_mismatch",
                    "latest_checkpoint.json does not identify the completed run's latest compatible checkpoint.",
                    scenario=scenario,
                    case=case_label,
                )

    expected_steps = list(range(checkpoint_every, completed_steps + 1, checkpoint_every))
    expected_steps.append(completed_steps)
    expected_retained = sorted(set(expected_steps))[-2:]
    observed_retained: list[int] = []
    for candidate in sorted(checkpoint_dir.glob("*_step_*.json")):
        try:
            candidate_metadata = read_json(candidate)
            candidate_step = int(candidate_metadata.get("completed_steps", 0) or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        candidate_project = _recorded_artifact_path(
            candidate_metadata.get("qblade_project"),
            base_dir=candidate.parent,
        )
        candidate_fingerprint = str(candidate_metadata.get("checkpoint_identity_fingerprint", "") or "")
        if candidate_fingerprint != summary_fingerprint:
            issue(
                issues,
                "error",
                "retained_checkpoint_identity_mismatch",
                f"Retained checkpoint {candidate} identity does not match the completed run.",
                scenario=scenario,
                case=case_label,
            )
            continue
        candidate_expected_hash = str(candidate_metadata.get("qblade_project_sha256", "") or "")
        candidate_expected_size = _recorded_int(candidate_metadata.get("qblade_project_size_bytes"), default=0)
        if (
            candidate_project is not None
            and candidate_project.is_file()
            and candidate_project.stat().st_size > 0
        ):
            candidate_size = candidate_project.stat().st_size
            candidate_hash = sha256_file(candidate_project)
            if (
                candidate_expected_hash
                and candidate_expected_size > 0
                and candidate_size == candidate_expected_size
                and candidate_hash == candidate_expected_hash
            ):
                observed_retained.append(candidate_step)
            else:
                issue(
                    issues,
                    "error",
                    "retained_checkpoint_project_binding_mismatch",
                    f"Retained checkpoint QPR {candidate_project} does not match metadata {candidate}.",
                    scenario=scenario,
                    case=case_label,
                )
    if sorted(set(observed_retained)) != expected_retained:
        issue(
            issues,
            "error",
            "checkpoint_retention_mismatch",
            f"Expected retained complete checkpoint steps {expected_retained}, found {sorted(set(observed_retained))}.",
            scenario=scenario,
            case=case_label,
        )


def audit_results(
    manifest: dict[str, Any],
    settings: dict[str, Any],
    args: argparse.Namespace,
    issues: list[AuditIssue],
    *,
    scenario: str,
) -> None:
    for case in manifest.get("cases", []):
        case_label = str(case.get("label", ""))
        result_dir = Path(case.get("result_dir", ""))
        summary_path = first_summary_file(result_dir, issues, scenario=scenario, case_label=case_label)
        if summary_path is None:
            continue
        summary = read_json(summary_path)
        requested_steps = int(summary.get("requested_steps", 0) or 0)
        completed_steps = int(summary.get("completed_steps", 0) or 0)
        expected_steps = int(
            settings.get("qblade_steps_override")
            if settings.get("qblade_steps_override") is not None
            else settings.get("timesteps", 0)
        )
        if requested_steps != expected_steps:
            issue(issues, "error", "requested_steps_mismatch", f"Requested steps {requested_steps} != expected {expected_steps}.", scenario=scenario, case=case_label)
        if completed_steps != requested_steps:
            issue(issues, "error", "incomplete_run", f"Completed {completed_steps}/{requested_steps} timesteps.", scenario=scenario, case=case_label)
        if bool(summary.get("failed", False)):
            issue(issues, "error", "failed_run", str(summary.get("failure_message", "Run summary is marked failed.")), scenario=scenario, case=case_label)
        if not bool(summary.get("exported_results", False)):
            issue(issues, "error", "results_not_exported", "Run summary says QBlade results were not exported.", scenario=scenario, case=case_label)
        result_file = Path(str(summary.get("result_file", "")))
        if not result_file.is_file():
            issue(issues, "error", "missing_result_file", f"Missing exported result file: {result_file}", scenario=scenario, case=case_label)
        audit_checkpoint_and_project_artifacts(
            summary,
            settings,
            args,
            issues,
            result_dir=result_dir,
            scenario=scenario,
            case_label=case_label,
        )

        reproducibility = summary.get("reproducibility")
        if not isinstance(reproducibility, dict):
            issue(issues, "error", "missing_reproducibility", "Run summary is missing reproducibility metadata.", scenario=scenario, case=case_label)
            continue
        case_settings = reproducibility.get("qblade_case_settings", {})
        expected_asset_fingerprint = str(case.get("runtime_asset_fingerprint", "") or "")
        observed_asset_fingerprint = str(reproducibility.get("runtime_asset_fingerprint", "") or "")
        runtime_assets = reproducibility.get("runtime_assets")
        if not observed_asset_fingerprint or not isinstance(runtime_assets, list) or not runtime_assets:
            issue(
                issues,
                "error",
                "missing_runtime_asset_fingerprint",
                "Run summary is missing the hashes of the actual QBlade/VorLap runtime assets.",
                scenario=scenario,
                case=case_label,
            )
        elif expected_asset_fingerprint and observed_asset_fingerprint != expected_asset_fingerprint:
            issue(
                issues,
                "error",
                "runtime_asset_fingerprint_mismatch",
                f"Run used runtime assets {observed_asset_fingerprint}, but setup recorded {expected_asset_fingerprint}.",
                scenario=scenario,
                case=case_label,
            )
        checks = {
            "NUMTIMESTEPS": settings.get("timesteps"),
            "TIMESTEP": settings.get("timestep_s"),
            "PRESCRIBETYPE": settings.get("rpm_prescribe_type", 1),
            "RPMPRESCRIBED": args.expected_rpm,
            "MEANINF": case.get("wind_speed"),
            "INITIAL_AZIMUTH": case.get("azimuth_deg"),
            "WNDTYPE": 1 if settings.get("wind_name") else 0,
            "HORANGLE": settings.get("horizontal_inflow_angle_deg", 0.0),
            "VERTANGLE": settings.get("vertical_inflow_angle_deg", 0.0),
            "SHEAREXP": settings.get("shear_exponent", 0.23),
            "REFHEIGHT": settings.get("reference_height_m", 18.0),
            "DENSITYAIR": settings.get("air_density_kg_m3", 1.0296),
            "VISCOSITYAIR": settings.get("air_kinematic_viscosity_m2_s", 1.647e-5),
        }
        if settings.get("result_filter"):
            storage_filter = case.get("storage_filter") or settings["result_filter"]
            checks["FILTERFILE"] = Path(str(storage_filter)).name
        if "store_from_s" in settings:
            checks["STOREFROM"] = settings["store_from_s"]
        for keyword, expected in checks.items():
            raw_value = case_settings.get(keyword)
            if raw_value is None:
                issue(issues, "error", "missing_qblade_case_setting", f"Missing QBlade setting {keyword}.", scenario=scenario, case=case_label)
                continue
            if keyword == "FILTERFILE":
                if str(raw_value) != str(expected):
                    issue(
                        issues,
                        "error",
                        "qblade_case_setting_mismatch",
                        f"{keyword}={raw_value!r} != expected {expected!r}.",
                        scenario=scenario,
                        case=case_label,
                    )
                continue
            try:
                observed = float(raw_value)
                expected_float = float(expected)
            except (TypeError, ValueError):
                issue(issues, "error", "invalid_qblade_case_setting", f"Invalid QBlade setting {keyword}={raw_value!r}.", scenario=scenario, case=case_label)
                continue
            if not close_enough(observed, expected_float):
                issue(issues, "error", "qblade_case_setting_mismatch", f"{keyword}={observed:g} != expected {expected_float:g}.", scenario=scenario, case=case_label)
        if "store_replay" in settings:
            expected_store_replay = bool(settings["store_replay"])
            observed_store_replay = parse_probe_bool(case_settings.get("STOREREPLAY"))
            if observed_store_replay is None:
                issue(
                    issues,
                    "error",
                    "invalid_qblade_case_setting",
                    f"Invalid QBlade setting STOREREPLAY={case_settings.get('STOREREPLAY')!r}.",
                    scenario=scenario,
                    case=case_label,
                )
            elif observed_store_replay != expected_store_replay:
                issue(
                    issues,
                    "error",
                    "qblade_case_setting_mismatch",
                    f"STOREREPLAY={observed_store_replay} != expected {expected_store_replay}.",
                    scenario=scenario,
                    case=case_label,
                )
        wind_name_template = str(settings.get("wind_name", "") or "")
        if wind_name_template:
            try:
                expected_wind_name = wind_name_template.format(
                    speed=float(case.get("wind_speed", 0.0)),
                    label=case_label,
                    speed_label=case.get("speed_label", ""),
                    azimuth=float(case.get("azimuth_deg", 0.0)),
                    azimuth_label=case.get("azimuth_label", ""),
                )
            except (KeyError, ValueError) as exc:
                issue(
                    issues,
                    "error",
                    "invalid_wind_name_template",
                    f"Could not format wind_name template {wind_name_template!r}: {exc}",
                    scenario=scenario,
                    case=case_label,
                )
            else:
                observed_wind_name = str(case_settings.get("WNDNAME", "") or "")
                if observed_wind_name != expected_wind_name:
                    issue(
                        issues,
                        "error",
                        "qblade_wind_name_mismatch",
                        f"WNDNAME={observed_wind_name!r} != expected {expected_wind_name!r}.",
                        scenario=scenario,
                        case=case_label,
                    )

        embedded_case = reproducibility.get("case_metadata", {}).get("case", {})
        if embedded_case.get("label") != case_label:
            issue(issues, "error", "case_metadata_mismatch", "Embedded case metadata does not match manifest label.", scenario=scenario, case=case_label)
        expected_mode = expected_coupling(settings, scenario)
        if expected_mode is not None and str(summary.get("coupling_mode", "")) != expected_mode:
            issue(
                issues,
                "error",
                "run_summary_coupling_mismatch",
                f"Run summary coupling_mode={summary.get('coupling_mode')!r} does not match expected {expected_mode!r}.",
                scenario=scenario,
                case=case_label,
            )
        try:
            fixed_speed = int(settings.get("rpm_prescribe_type", 1)) == 1
        except (TypeError, ValueError):
            fixed_speed = False
        if fixed_speed:
            rpm_diagnostics = summary.get("qblade_export_rpm_diagnostics")
            if not isinstance(rpm_diagnostics, dict) or not bool(rpm_diagnostics.get("available", False)):
                issue(
                    issues,
                    "error",
                    "missing_exported_rpm_diagnostics",
                    "Fixed-speed run summary has no parseable exported LSS Rpm [rpm] diagnostics.",
                    scenario=scenario,
                    case=case_label,
                )
            else:
                tolerance = float(settings.get("rpm_audit_tolerance_rpm", getattr(args, "rpm_tolerance", 0.01)))
                max_error = float(rpm_diagnostics.get("max_abs_error_rpm", math.inf))
                observed_expected = float(rpm_diagnostics.get("expected_rpm", math.nan))
                required_start = max(
                    float(settings.get("store_from_s", 0.0) or 0.0),
                    float(settings.get("rpm_audit_startup_allowance_s", 0.1) or 0.1),
                )
                observed_start = float(rpm_diagnostics.get("start_time_s", -math.inf))
                if not close_enough(observed_expected, float(args.expected_rpm)):
                    issue(
                        issues,
                        "error",
                        "exported_rpm_expected_value_mismatch",
                        f"RPM diagnostics expected {observed_expected:g} rpm, audit expects {args.expected_rpm:g} rpm.",
                        scenario=scenario,
                        case=case_label,
                    )
                if observed_start + 1.0e-12 < required_start:
                    issue(
                        issues,
                        "error",
                        "exported_rpm_startup_exclusion_too_short",
                        f"RPM diagnostics start at {observed_start:g} s, before required {required_start:g} s.",
                        scenario=scenario,
                        case=case_label,
                    )
                if not math.isfinite(max_error) or max_error > tolerance:
                    issue(
                        issues,
                        "error",
                        "exported_rpm_not_prescribed",
                        f"Exported LSS RPM differs from {args.expected_rpm:g} rpm by up to {max_error:g} rpm; tolerance is {tolerance:g} rpm.",
                        scenario=scenario,
                        case=case_label,
                    )
        native_evidence = summary.get("native_callback_evidence")
        if summary.get("coupling_mode") != "qblade-external" and isinstance(native_evidence, dict):
            update_init_count = int(native_evidence.get("update_init_count", 0) or 0)
            update_count = int(native_evidence.get("update_count", 0) or 0)
            if update_init_count or update_count:
                issue(
                    issues,
                    "error",
                    "unexpected_native_callback_activity",
                    f"{summary.get('coupling_mode')} run recorded native callback activity "
                    f"(update_init={update_init_count}, update={update_count}); forces may have been applied twice.",
                    scenario=scenario,
                    case=case_label,
                )
        if summary.get("coupling_mode") == "sil-vorlap":
            if int(summary.get("sil_vorlap_updates", -1)) != completed_steps:
                issue(issues, "error", "sil_update_count_mismatch", "SIL VorLap update count does not match completed steps.", scenario=scenario, case=case_label)
            audit_velocity_diagnostics(summary, issues, scenario=scenario, case_label=case_label)
            audit_timing_diagnostics(summary, issues, scenario=scenario, case_label=case_label, completed_steps=completed_steps)
        if summary.get("coupling_mode") == "qblade-external":
            audit_native_callback_evidence(
                summary.get("native_callback_evidence"),
                issues,
                scenario=scenario,
                case_label=case_label,
                completed_steps=completed_steps,
            )


def structural_model_qualification_path(args: argparse.Namespace) -> Path:
    configured = getattr(args, "structural_model_qualification", None)
    if configured is not None:
        return Path(configured)
    return args.run_root / STRUCTURAL_QUALIFICATION_DEFAULT


def qualification_path(run_root: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else run_root / path


def qualification_frequency_hz_rows(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    frequencies: list[float] = []
    for item in value:
        raw = item.get("frequency_hz") if isinstance(item, dict) else item
        try:
            frequency = float(raw)
        except (TypeError, ValueError):
            frequencies.append(math.nan)
            continue
        frequencies.append(frequency)
    return frequencies


def qualification_timestep_s(payload: dict[str, Any]) -> float:
    solver = payload.get("solver_settings")
    raw = payload.get("timestep_s")
    if raw is None and isinstance(solver, dict):
        raw = solver.get("timestep_s", solver.get("TIMESTEP"))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


def qualification_file_entries(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            entries.append({"path": item})
        elif isinstance(item, dict):
            entries.append(item)
    return entries


def audit_qualification_file_entries(
    entries: list[dict[str, Any]],
    run_root: Path,
    issues: list[AuditIssue],
    *,
    field: str,
    code_prefix: str,
    require_hash: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    if not entries:
        issue(issues, "error", f"missing_{code_prefix}", f"Structural qualification manifest is missing {field}.")
        return paths
    for index, entry in enumerate(entries):
        path = qualification_path(run_root, entry.get("path"))
        if path is None:
            issue(issues, "error", f"invalid_{code_prefix}_path", f"{field}[{index}] is missing a nonempty path.")
            continue
        paths.append(path)
        if not path.is_file():
            issue(issues, "error", f"missing_{code_prefix}_file", f"{field}[{index}] does not exist: {path}")
            continue
        expected_hash = entry.get("sha256")
        if expected_hash is None and require_hash:
            issue(issues, "error", f"missing_{code_prefix}_hash", f"{field}[{index}] is missing sha256 for {path}.")
            continue
        if expected_hash is not None:
            actual_hash = sha256_file(path)
            if str(expected_hash).strip().lower() != actual_hash:
                issue(
                    issues,
                    "error",
                    f"{code_prefix}_hash_mismatch",
                    f"{field}[{index}] sha256 does not match {path}: expected {expected_hash}, observed {actual_hash}.",
                )
    return paths


def audit_structural_model_qualification(
    settings: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    path = structural_model_qualification_path(args)
    require = bool(getattr(args, "require_structural_model_qualification", False))
    if not path.is_file():
        if require:
            issue(
                issues,
                "error",
                "missing_structural_model_qualification",
                f"Missing structural model qualification manifest: {path}",
            )
            return {"path": str(path), "available": False}
        return None

    try:
        payload = read_json(path)
    except Exception as exc:
        issue(issues, "error", "invalid_structural_model_qualification", f"Could not read {path}: {exc}")
        return {"path": str(path), "available": False}
    if not isinstance(payload, dict):
        issue(issues, "error", "invalid_structural_model_qualification", f"{path} must contain a JSON object.")
        return {"path": str(path), "available": False}

    model_id = str(payload.get("model_id", "")).strip()
    model_choice = str(payload.get("model_choice", "")).strip()
    if not model_id:
        issue(issues, "error", "structural_qualification_missing_model_id", "Structural qualification manifest is missing model_id.")
    if model_choice not in STRUCTURAL_MODEL_CHOICES:
        issue(
            issues,
            "error",
            "structural_qualification_invalid_model_choice",
            f"model_choice must be one of {', '.join(STRUCTURAL_MODEL_CHOICES)}.",
        )
    if payload.get("qualified") is not True:
        issue(issues, "error", "structural_qualification_not_marked_qualified", "Structural qualification manifest must set qualified=true.")

    frequencies = qualification_frequency_hz_rows(payload.get("modal_frequencies_hz", payload.get("modal_frequencies")))
    if not frequencies:
        issue(issues, "error", "structural_qualification_missing_modal_frequencies", "Structural qualification manifest is missing modal frequencies.")
    elif any((not math.isfinite(frequency)) or frequency <= 0.0 for frequency in frequencies):
        issue(issues, "error", "structural_qualification_invalid_modal_frequency", "Modal frequencies must be finite positive Hz values.")

    damping = payload.get("damping")
    if damping is None:
        damping = payload.get("damping_assumptions")
    if damping is None or (isinstance(damping, (str, list, dict)) and not damping):
        issue(issues, "error", "structural_qualification_missing_damping", "Structural qualification manifest is missing damping assumptions.")

    solver = payload.get("solver_settings")
    if not isinstance(solver, dict) or not solver:
        issue(issues, "error", "structural_qualification_missing_solver_settings", "Structural qualification manifest is missing solver_settings.")
        solver = {}

    timestep = qualification_timestep_s(payload)
    if not math.isfinite(timestep) or timestep <= 0.0:
        issue(issues, "error", "structural_qualification_invalid_timestep", "Structural qualification timestep must be finite and positive.")
    else:
        expected_timestep = float(settings.get("timestep_s", math.nan))
        if math.isfinite(expected_timestep) and not close_enough(timestep, expected_timestep):
            issue(
                issues,
                "error",
                "structural_qualification_timestep_mismatch",
                f"Structural qualification timestep {timestep:g} does not match campaign timestep {expected_timestep:g}.",
            )

    structural_entries = qualification_file_entries(payload.get("structural_files"))
    structural_paths = audit_qualification_file_entries(
        structural_entries,
        args.run_root,
        issues,
        field="structural_files",
        code_prefix="structural_qualification_structural",
    )

    patch_entries = qualification_file_entries(payload.get("patch_manifests"))
    patch_paths: list[Path] = []
    if model_choice == "generated_patched_demonstration":
        patch_paths = audit_qualification_file_entries(
            patch_entries,
            args.run_root,
            issues,
            field="patch_manifests",
            code_prefix="structural_qualification_patch_manifest",
        )
        for patch_path in patch_paths:
            if patch_path.is_file():
                audit_stable_support_patch_file(
                    patch_path,
                    settings,
                    issues,
                    scenario="structural-model",
                    case_label=patch_path.parent.parent.parent.name,
                )
    elif patch_entries:
        patch_paths = audit_qualification_file_entries(
            patch_entries,
            args.run_root,
            issues,
            field="patch_manifests",
            code_prefix="structural_qualification_patch_manifest",
        )

    baseline_identity = payload.get("baseline_identity_evidence")
    baseline_artifact = None
    if not isinstance(baseline_identity, dict):
        issue(
            issues,
            "error",
            "structural_qualification_missing_baseline_identity",
            "Structural qualification manifest is missing baseline_identity_evidence.",
        )
    else:
        status = str(baseline_identity.get("status", "")).strip().lower()
        if status != "passed":
            issue(
                issues,
                "error",
                "structural_qualification_baseline_identity_not_passed",
                f"baseline_identity_evidence.status must be 'passed', observed {status!r}.",
            )
        baseline_artifact = qualification_path(args.run_root, baseline_identity.get("artifact"))
        if baseline_artifact is None:
            issue(
                issues,
                "error",
                "structural_qualification_missing_baseline_identity_artifact",
                "baseline_identity_evidence is missing an artifact path.",
            )
        elif not baseline_artifact.is_file():
            issue(
                issues,
                "error",
                "structural_qualification_missing_baseline_identity_artifact",
                f"Baseline identity artifact does not exist: {baseline_artifact}",
            )
        elif baseline_artifact.stat().st_size <= 0:
            issue(
                issues,
                "error",
                "structural_qualification_empty_baseline_identity_artifact",
                f"Baseline identity artifact is empty: {baseline_artifact}",
            )
        elif baseline_artifact.suffix.lower() == ".csv":
            try:
                with baseline_artifact.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fieldnames = list(reader.fieldnames or [])
                    rows = list(reader)
            except Exception as exc:
                issue(
                    issues,
                    "error",
                    "structural_qualification_baseline_identity_artifact_unreadable",
                    f"Could not read baseline identity artifact {baseline_artifact}: {exc}",
                )
            else:
                if not fieldnames or not rows:
                    issue(
                        issues,
                        "error",
                        "structural_qualification_empty_baseline_identity_artifact",
                        f"Baseline identity CSV artifact must include a header and at least one data row: {baseline_artifact}",
                    )

    solver_compare_keys = ("TIMESTEP", "STRITERATIONS", "RAMPUP", "ADDDAMP")
    for scenario, manifest in manifests.items():
        for case in manifest.get("cases", []):
            case_label = str(case.get("label", ""))
            result_dir = Path(case.get("result_dir", ""))
            summary_path = first_summary_file(result_dir, issues, scenario=scenario, case_label=case_label)
            if summary_path is None:
                continue
            summary = read_json(summary_path)
            reproducibility = summary.get("reproducibility")
            case_settings = reproducibility.get("qblade_case_settings", {}) if isinstance(reproducibility, dict) else {}
            if not isinstance(case_settings, dict):
                continue
            for key in solver_compare_keys:
                if key not in solver:
                    continue
                observed_raw = case_settings.get(key)
                if observed_raw is None:
                    issue(issues, "error", "structural_qualification_case_setting_missing", f"Run summary missing QBlade case setting {key}.", scenario=scenario, case=case_label)
                    continue
                try:
                    expected = float(solver[key])
                    observed = float(observed_raw)
                except (TypeError, ValueError):
                    issue(issues, "error", "structural_qualification_case_setting_invalid", f"Cannot compare solver setting {key}.", scenario=scenario, case=case_label)
                    continue
                if not close_enough(observed, expected):
                    issue(
                        issues,
                        "error",
                        "structural_qualification_case_setting_mismatch",
                        f"Run summary {key}={observed:g} does not match structural qualification {expected:g}.",
                        scenario=scenario,
                        case=case_label,
                    )

    return {
        "path": str(path),
        "available": True,
        "model_id": model_id,
        "model_choice": model_choice,
        "modal_frequency_count": len(frequencies),
        "structural_file_count": len(structural_paths),
        "patch_manifest_count": len(patch_paths),
        "baseline_identity_artifact": None if baseline_artifact is None else str(baseline_artifact),
    }


def artifact_row_count(path: Path, issues: list[AuditIssue], *, artifact: str, kind: str) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                issue(issues, "error", "postprocess_artifact_empty", f"{kind} artifact has no header: {artifact}")
                return 0
            rows = list(reader)
    except Exception as exc:
        issue(issues, "error", "postprocess_artifact_unreadable", f"Could not read {kind} artifact {artifact}: {exc}")
        return 0
    if not rows:
        issue(issues, "error", "postprocess_artifact_empty", f"{kind} artifact has no data rows: {artifact}")
    return len(rows)


def audit_schema_csv_artifact(
    path: Path,
    issues: list[AuditIssue],
    *,
    artifact: str,
    required_columns: tuple[str, ...],
    disallow_smoke_only: bool,
    enforce_paper_eligibility: bool,
) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if not fieldnames:
                issue(issues, "error", "postprocess_artifact_empty", f"CSV artifact has no header: {artifact}")
                return 0
            missing_columns = [column for column in required_columns if column not in fieldnames]
            if missing_columns:
                issue(
                    issues,
                    "error",
                    "postprocess_artifact_schema_mismatch",
                    f"Postprocess artifact {artifact} is missing required columns: {', '.join(missing_columns)}.",
                )
            rows = list(reader)
    except Exception as exc:
        issue(issues, "error", "postprocess_artifact_unreadable", f"Could not read CSV artifact {artifact}: {exc}")
        return 0
    if not rows:
        issue(issues, "error", "postprocess_artifact_empty", f"CSV artifact has no data rows: {artifact}")
        return 0
    if disallow_smoke_only and "smoke_only" in fieldnames:
        smoke_rows = [row for row in rows if row_bool_text(row, "smoke_only")]
        if smoke_rows:
            issue(
                issues,
                "error",
                "postprocess_paper_artifact_contains_smoke_only",
                f"Paper-facing postprocess artifact {artifact} contains {len(smoke_rows)} smoke-only row(s).",
            )
    if (
        enforce_paper_eligibility
        and artifact in POSTPROCESS_PAPER_INTERPRETED_ARTIFACTS
        and "postprocess_warnings" in fieldnames
    ):
        warned_rows = [row for row in rows if str(row.get("postprocess_warnings", "")).strip()]
        if warned_rows:
            issue(
                issues,
                "error",
                "postprocess_paper_artifact_contains_warnings",
                f"Paper-facing postprocess artifact {artifact} contains {len(warned_rows)} row(s) with postprocess_warnings.",
            )
    numeric_columns = POSTPROCESS_PAPER_NUMERIC_COLUMNS.get(artifact, ())
    for column in numeric_columns:
        if column not in fieldnames:
            continue
        bad_count = 0
        for row in rows:
            try:
                value = float(row.get(column, ""))
            except (TypeError, ValueError):
                bad_count += 1
                continue
            if not math.isfinite(value):
                bad_count += 1
        if bad_count:
            issue(
                issues,
                "error",
                "postprocess_paper_artifact_nonfinite_value",
                f"Paper-facing postprocess artifact {artifact} has {bad_count} nonfinite or invalid {column} value(s).",
            )
    return len(rows)


def read_postprocess_csv_rows(path: Path, issues: list[AuditIssue], *, artifact: str) -> tuple[set[str], list[dict[str, str]]] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        issue(issues, "error", "postprocess_artifact_unreadable", f"Could not read CSV artifact {artifact}: {exc}")
        return None
    return fieldnames, rows


def postprocess_row_float(row: dict[str, str], column: str) -> float:
    try:
        return float(row.get(column, ""))
    except (TypeError, ValueError):
        return math.nan


def expected_critical_case_selection(signed_rows: list[dict[str, str]]) -> tuple[dict[str, str], float, str] | None:
    candidates: list[tuple[dict[str, str], float, float]] = []
    for row in signed_rows:
        baseline = postprocess_row_float(row, "signed_del_baseline")
        vorlap = postprocess_row_float(row, "signed_del_vorlap")
        if not math.isfinite(vorlap):
            continue
        ratio = vorlap / baseline if math.isfinite(baseline) and baseline > 0.0 else math.nan
        candidates.append((row, ratio, vorlap))
    if not candidates:
        return None
    finite_ratio = [candidate for candidate in candidates if math.isfinite(candidate[1])]
    if finite_ratio:
        row, ratio, _ = max(finite_ratio, key=lambda candidate: (candidate[1], candidate[2]))
        return row, ratio, "max_signed_del_ratio"
    row, _, _ = max(candidates, key=lambda candidate: candidate[2])
    return row, math.nan, "max_signed_del_vorlap_no_finite_baseline_ratio"


def postprocess_values_match(left: str, right: str, *, numeric: bool) -> bool:
    left_text = str(left).strip()
    right_text = str(right).strip()
    if not numeric:
        return left_text == right_text
    try:
        left_value = float(left_text)
        left_parsed = True
    except (TypeError, ValueError):
        left_value = math.nan
        left_parsed = False
    try:
        right_value = float(right_text)
        right_parsed = True
    except (TypeError, ValueError):
        right_value = math.nan
        right_parsed = False
    if not left_parsed or not right_parsed:
        return (not left_parsed) and (not right_parsed) and left_text == right_text
    if math.isfinite(left_value) and math.isfinite(right_value):
        return close_enough(left_value, right_value, rel=1.0e-8, abs_tol=1.0e-8)
    return not math.isfinite(left_value) and not math.isfinite(right_value)


def postprocess_rows_match(
    left: dict[str, str],
    right: dict[str, str],
    *,
    columns: tuple[str, ...],
    numeric_columns: set[str],
) -> bool:
    return all(
        postprocess_values_match(
            str(left.get(column, "")),
            str(right.get(column, "")),
            numeric=column in numeric_columns,
        )
        for column in columns
    )


def audit_signed_del_diagnostics_subset(run_root: Path, issues: list[AuditIssue]) -> dict[str, Any] | None:
    paper_artifact = "postprocess/signed_del_comparison.csv"
    diagnostics_artifact = "postprocess/signed_del_comparison_diagnostics.csv"
    paper_path = run_root / paper_artifact
    diagnostics_path = run_root / diagnostics_artifact
    if not paper_path.is_file() or not diagnostics_path.is_file():
        return None
    paper = read_postprocess_csv_rows(paper_path, issues, artifact=paper_artifact)
    diagnostics = read_postprocess_csv_rows(diagnostics_path, issues, artifact=diagnostics_artifact)
    if paper is None or diagnostics is None:
        return None
    paper_fieldnames, paper_rows = paper
    diagnostics_fieldnames, diagnostics_rows = diagnostics
    columns = POSTPROCESS_PAPER_CSV_SCHEMAS[paper_artifact]
    if not set(columns).issubset(paper_fieldnames) or not set(columns).issubset(diagnostics_fieldnames):
        return None
    numeric_columns = set(POSTPROCESS_PAPER_NUMERIC_COLUMNS[paper_artifact]) | {"signed_delta_pct"}
    missing_rows = [
        row
        for row in paper_rows
        if not any(postprocess_rows_match(row, diagnostic, columns=columns, numeric_columns=numeric_columns) for diagnostic in diagnostics_rows)
    ]
    if missing_rows:
        issue(
            issues,
            "error",
            "postprocess_signed_table_not_in_diagnostics",
            "signed_del_comparison.csv contains "
            f"{len(missing_rows)} row(s) not found in signed_del_comparison_diagnostics.csv.",
        )
    return {
        "paper_rows": len(paper_rows),
        "diagnostic_rows": len(diagnostics_rows),
        "missing_rows": len(missing_rows),
    }


def audit_postprocess_critical_case_selection(run_root: Path, issues: list[AuditIssue]) -> dict[str, Any] | None:
    signed_artifact = "postprocess/signed_del_comparison.csv"
    critical_artifact = "postprocess/critical_case_summary.csv"
    signed_path = run_root / signed_artifact
    critical_path = run_root / critical_artifact
    if not signed_path.is_file() or not critical_path.is_file():
        return None
    signed = read_postprocess_csv_rows(signed_path, issues, artifact=signed_artifact)
    critical = read_postprocess_csv_rows(critical_path, issues, artifact=critical_artifact)
    if signed is None or critical is None:
        return None
    signed_fieldnames, signed_rows = signed
    critical_fieldnames, critical_rows = critical
    signed_required = set(CRITICAL_CASE_SELECTION_TEXT_COLUMNS) | set(CRITICAL_CASE_SELECTION_NUMERIC_COLUMNS)
    signed_required.discard("signed_del_ratio")
    critical_required = (
        set(CRITICAL_CASE_SELECTION_TEXT_COLUMNS)
        | set(CRITICAL_CASE_SELECTION_NUMERIC_COLUMNS)
        | {"signed_del_ratio", "selection_reason"}
    )
    if not signed_required.issubset(signed_fieldnames) or not critical_required.issubset(critical_fieldnames):
        return None
    if len(critical_rows) != 1:
        issue(
            issues,
            "error",
            "postprocess_critical_case_row_count",
            f"critical_case_summary.csv should contain exactly one selected row; found {len(critical_rows)}.",
        )
        return {"critical_rows": len(critical_rows), "signed_rows": len(signed_rows)}
    expected = expected_critical_case_selection(signed_rows)
    if expected is None:
        issue(
            issues,
            "error",
            "postprocess_critical_case_no_candidate",
            "critical_case_summary.csv exists but signed_del_comparison.csv has no finite signed_del_vorlap candidate.",
        )
        return {"critical_rows": len(critical_rows), "signed_rows": len(signed_rows)}

    expected_row, expected_ratio, expected_reason = expected
    actual_row = critical_rows[0]
    mismatches: list[str] = []
    for column in CRITICAL_CASE_SELECTION_TEXT_COLUMNS:
        if str(actual_row.get(column, "")).strip() != str(expected_row.get(column, "")).strip():
            mismatches.append(column)
    for column in CRITICAL_CASE_SELECTION_NUMERIC_COLUMNS:
        actual = postprocess_row_float(actual_row, column)
        expected_value = postprocess_row_float(expected_row, column)
        if not (math.isfinite(actual) and math.isfinite(expected_value) and close_enough(actual, expected_value, rel=1.0e-8, abs_tol=1.0e-8)):
            mismatches.append(column)
    actual_reason = str(actual_row.get("selection_reason", "")).strip()
    if actual_reason != expected_reason:
        mismatches.append("selection_reason")
    actual_ratio = postprocess_row_float(actual_row, "signed_del_ratio")
    if math.isfinite(expected_ratio):
        if not (math.isfinite(actual_ratio) and close_enough(actual_ratio, expected_ratio, rel=1.0e-8, abs_tol=1.0e-8)):
            mismatches.append("signed_del_ratio")
    elif math.isfinite(actual_ratio):
        mismatches.append("signed_del_ratio")

    if mismatches:
        issue(
            issues,
            "error",
            "postprocess_critical_case_selection_mismatch",
            "critical_case_summary.csv does not match the selected signed_del_comparison.csv row "
            f"({expected_reason}); mismatched field(s): {', '.join(mismatches)}.",
        )
    return {
        "critical_rows": len(critical_rows),
        "signed_rows": len(signed_rows),
        "selection_reason": expected_reason,
    }


def audit_critical_case_rainflow_cycles(run_root: Path, issues: list[AuditIssue]) -> dict[str, Any] | None:
    critical_artifact = "postprocess/critical_case_summary.csv"
    rainflow_artifact = "postprocess/critical_case_rainflow_ranges.csv"
    critical_path = run_root / critical_artifact
    rainflow_path = run_root / rainflow_artifact
    if not critical_path.is_file() or not rainflow_path.is_file():
        return None
    critical = read_postprocess_csv_rows(critical_path, issues, artifact=critical_artifact)
    rainflow = read_postprocess_csv_rows(rainflow_path, issues, artifact=rainflow_artifact)
    if critical is None or rainflow is None:
        return None
    critical_fieldnames, critical_rows = critical
    rainflow_fieldnames, rainflow_rows = rainflow
    if "rainflow_cycles_vorlap" not in critical_fieldnames or "count" not in rainflow_fieldnames:
        return None
    if len(critical_rows) != 1:
        return None
    expected_cycles = postprocess_row_float(critical_rows[0], "rainflow_cycles_vorlap")
    if not math.isfinite(expected_cycles):
        return None
    total_count = 0.0
    bad_count_rows = 0
    for row in rainflow_rows:
        value = postprocess_row_float(row, "count")
        if math.isfinite(value):
            total_count += value
        else:
            bad_count_rows += 1
    if bad_count_rows:
        issue(
            issues,
            "error",
            "postprocess_critical_rainflow_invalid_count",
            f"critical_case_rainflow_ranges.csv has {bad_count_rows} invalid count value(s).",
        )
    elif not close_enough(total_count, expected_cycles, rel=1.0e-8, abs_tol=1.0e-8):
        issue(
            issues,
            "error",
            "postprocess_critical_rainflow_cycle_mismatch",
            "critical_case_rainflow_ranges.csv count sum does not match "
            f"critical_case_summary.csv rainflow_cycles_vorlap ({total_count:g} vs {expected_cycles:g}).",
        )
    return {
        "rainflow_rows": len(rainflow_rows),
        "rainflow_count_sum": total_count,
        "expected_cycles": expected_cycles,
    }


def signed_channel_selection_path(args: argparse.Namespace) -> Path:
    configured = getattr(args, "signed_channel_selection", None)
    if configured is not None:
        return Path(configured)
    return args.run_root / SIGNED_CHANNEL_SELECTION_DEFAULT


def signed_selection_artifact_path(run_root: Path, raw_path: object, default: str) -> Path:
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path)
    else:
        path = Path(default)
    return path if path.is_absolute() else run_root / path


def nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def selection_numeric_match(left: object, right: object) -> bool:
    try:
        return close_enough(float(left), float(right), rel=1.0e-8, abs_tol=1.0e-8)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def signed_channel_matches_row(selection: dict[str, Any], row: dict[str, str]) -> bool:
    for column in ("case", "quantity", "axis", "unit"):
        if column in selection and str(selection.get(column, "")).strip() != str(row.get(column, "")).strip():
            return False
    if "position" in selection and not selection_numeric_match(selection.get("position"), row.get("position")):
        return False
    if "channel_vorlap" in selection and str(selection.get("channel_vorlap", "")).strip() != str(row.get("channel_vorlap", "")).strip():
        return False
    if "channel" in selection:
        selected_channel = str(selection.get("channel", "")).strip()
        if selected_channel not in {
            str(row.get("channel_vorlap", "")).strip(),
            str(row.get("channel_baseline", "")).strip(),
        }:
            return False
    return True


def signed_channel_required_columns(selection: dict[str, Any]) -> list[str]:
    required = ["case", "position", "quantity", "axis", "unit"]
    if "channel" not in selection and "channel_vorlap" not in selection:
        required.append("channel")
    return [column for column in required if column not in selection or not str(selection.get(column, "")).strip()]


def audit_signed_channel_selection(
    settings: dict[str, Any],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    path = signed_channel_selection_path(args)
    require = bool(getattr(args, "require_signed_channel_selection", False))
    if not path.is_file():
        if require:
            issue(issues, "error", "missing_signed_channel_selection", f"Missing signed-channel selection manifest: {path}")
            return {"path": str(path), "available": False}
        return None

    try:
        payload = read_json(path)
    except Exception as exc:
        issue(issues, "error", "invalid_signed_channel_selection", f"Could not read {path}: {exc}")
        return {"path": str(path), "available": False}
    if not isinstance(payload, dict):
        issue(issues, "error", "invalid_signed_channel_selection", f"{path} must contain a JSON object.")
        return {"path": str(path), "available": False}

    if not nonempty_text(payload.get("selection_id")):
        issue(issues, "error", "signed_channel_selection_missing_id", "Signed-channel selection manifest is missing selection_id.")

    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        issue(issues, "error", "signed_channel_selection_missing_normalization", "Signed-channel selection manifest is missing normalization.")
        normalization = {}

    required_text_fields = (
        "metric",
        "baseline",
        "comparison",
        "reference_cycle_convention",
        "transient_removal",
        "mean_stress_assumption",
    )
    for field in required_text_fields:
        if not nonempty_text(normalization.get(field)):
            issue(
                issues,
                "error",
                "signed_channel_selection_incomplete_normalization",
                f"normalization.{field} must be nonempty.",
            )
    try:
        wohler = float(normalization.get("wohler_exponent"))
    except (TypeError, ValueError):
        wohler = math.nan
    if not math.isfinite(wohler) or wohler <= 0.0:
        issue(issues, "error", "signed_channel_selection_invalid_wohler", "normalization.wohler_exponent must be finite and positive.")
    else:
        expected_wohler = settings.get("wohler_exponent")
        if expected_wohler is not None:
            try:
                expected = float(expected_wohler)
            except (TypeError, ValueError):
                expected = math.nan
            if math.isfinite(expected) and not close_enough(wohler, expected):
                issue(
                    issues,
                    "error",
                    "signed_channel_selection_wohler_mismatch",
                    f"normalization.wohler_exponent={wohler:g} does not match campaign setting {expected:g}.",
                )

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    signed_path = signed_selection_artifact_path(
        args.run_root,
        artifacts.get("signed_del_comparison"),
        "postprocess/signed_del_comparison.csv",
    )
    critical_path = signed_selection_artifact_path(
        args.run_root,
        artifacts.get("critical_case_summary"),
        "postprocess/critical_case_summary.csv",
    )
    signed = read_postprocess_csv_rows(signed_path, issues, artifact="postprocess/signed_del_comparison.csv") if signed_path.is_file() else None
    if signed is None:
        issue(issues, "error", "signed_channel_selection_missing_artifact", f"Missing signed DEL comparison artifact: {signed_path}")
        signed_rows: list[dict[str, str]] = []
    else:
        _signed_fields, signed_rows = signed
        if not signed_rows:
            issue(
                issues,
                "error",
                "signed_channel_selection_empty_artifact",
                f"Signed DEL comparison artifact has no data rows: {signed_path}",
            )

    critical = read_postprocess_csv_rows(critical_path, issues, artifact="postprocess/critical_case_summary.csv") if critical_path.is_file() else None
    if critical is None:
        issue(issues, "error", "signed_channel_selection_missing_artifact", f"Missing critical-case summary artifact: {critical_path}")
        critical_rows: list[dict[str, str]] = []
    else:
        _critical_fields, critical_rows = critical
        if len(critical_rows) != 1:
            issue(
                issues,
                "error",
                "signed_channel_selection_critical_artifact_row_count",
                f"Critical-case summary artifact must contain exactly one selected row; found {len(critical_rows)}.",
            )

    selections = payload.get("selected_channels")
    if not isinstance(selections, list) or not selections:
        issue(issues, "error", "signed_channel_selection_missing_channels", "selected_channels must contain at least one channel.")
        selections = []

    matched_channels = 0
    critical_selected = 0
    for index, selection in enumerate(selections):
        if not isinstance(selection, dict):
            issue(issues, "error", "signed_channel_selection_invalid_channel", f"selected_channels[{index}] must be an object.")
            continue
        missing_columns = signed_channel_required_columns(selection)
        if missing_columns:
            issue(
                issues,
                "error",
                "signed_channel_selection_incomplete_channel",
                f"selected_channels[{index}] is missing required field(s): {', '.join(missing_columns)}.",
            )
            continue
        matched = [row for row in signed_rows if signed_channel_matches_row(selection, row)]
        if not matched:
            issue(
                issues,
                "error",
                "signed_channel_selection_channel_not_found",
                f"selected_channels[{index}] does not match any signed_del_comparison.csv row.",
            )
        else:
            matched_channels += 1
        if selection.get("critical") is True:
            critical_selected += 1
            critical_matches = [row for row in critical_rows if signed_channel_matches_row(selection, row)]
            if not critical_matches:
                issue(
                    issues,
                    "error",
                    "signed_channel_selection_critical_mismatch",
                    f"selected_channels[{index}] is marked critical but does not match critical_case_summary.csv.",
                )

    if critical_selected != 1:
        issue(
            issues,
            "error",
            "signed_channel_selection_critical_count",
            f"Exactly one selected channel must be marked critical; found {critical_selected}.",
        )

    if bool(payload.get("reports_stress_or_reserve_factors", False)):
        basis = payload.get("stress_or_reserve_factor_basis")
        if not isinstance(basis, dict):
            issue(
                issues,
                "error",
                "signed_channel_selection_missing_stress_basis",
                "reports_stress_or_reserve_factors=true requires stress_or_reserve_factor_basis.",
            )
        else:
            for field in ("allowables_artifact", "load_factor", "resistance_factor", "basis"):
                if field not in basis or not str(basis.get(field, "")).strip():
                    issue(
                        issues,
                        "error",
                        "signed_channel_selection_incomplete_stress_basis",
                        f"stress_or_reserve_factor_basis.{field} must be provided.",
                    )
            if nonempty_text(basis.get("allowables_artifact")):
                allowables_path = signed_selection_artifact_path(args.run_root, basis.get("allowables_artifact"), "")
                if not allowables_path.is_file():
                    issue(
                        issues,
                        "error",
                        "signed_channel_selection_missing_allowables_artifact",
                        f"Allowables artifact does not exist: {allowables_path}",
                    )

    return {
        "path": str(path),
        "available": True,
        "selection_id": str(payload.get("selection_id", "")),
        "selected_channels": len(selections),
        "matched_channels": matched_channels,
        "critical_selected": critical_selected,
        "signed_del_comparison": str(signed_path),
        "critical_case_summary": str(critical_path),
    }


def _audit_identity_delta_columns(
    rows: list[dict[str, str]],
    issues: list[AuditIssue],
    *,
    artifact: str,
    columns: tuple[str, ...],
    max_abs_delta_pct: float,
    issue_code: str,
) -> dict[str, Any]:
    max_abs_delta = 0.0
    max_rows: list[str] = []
    invalid_count = 0
    exceed_count = 0
    for row in rows:
        row_id = " ".join(
            str(row.get(column, "")).strip()
            for column in ("case", "position", "quantity", "axis")
            if str(row.get(column, "")).strip()
        )
        for column in columns:
            value = postprocess_row_float(row, column)
            if not math.isfinite(value):
                invalid_count += 1
                continue
            abs_value = abs(value)
            if abs_value > max_abs_delta:
                max_abs_delta = abs_value
                max_rows = [f"{row_id or '?'}:{column}"]
            elif close_enough(abs_value, max_abs_delta, rel=1.0e-12, abs_tol=1.0e-12):
                max_rows.append(f"{row_id or '?'}:{column}")
            if abs_value > max_abs_delta_pct:
                exceed_count += 1
    if invalid_count:
        issue(
            issues,
            "error",
            "baseline_identity_delta_nonfinite",
            f"{artifact} has {invalid_count} nonfinite baseline-vs-identity delta value(s).",
        )
    if exceed_count:
        issue(
            issues,
            "error",
            issue_code,
            f"{artifact} has {exceed_count} baseline-vs-identity delta value(s) exceeding {max_abs_delta_pct:g}%."
            f" Maximum absolute delta was {max_abs_delta:g}% at {', '.join(max_rows[:4])}.",
        )
    return {
        "rows": len(rows),
        "checked_columns": columns,
        "max_abs_delta_pct": max_abs_delta,
        "max_abs_delta_rows": max_rows[:8],
        "threshold_pct": max_abs_delta_pct,
        "invalid_count": invalid_count,
        "exceed_count": exceed_count,
    }


def audit_baseline_identity_drift(
    run_root: Path,
    args: argparse.Namespace,
    issues: list[AuditIssue],
    *,
    require_artifacts: bool = False,
) -> dict[str, Any] | None:
    """Require the zero-force bridge identity scenario to stay near no-SIL baseline."""
    checks = (
        (
            "postprocess/baseline_identity_comparison.csv",
            ("force_identity_delta_pct", "moment_identity_delta_pct"),
            float(getattr(args, "max_baseline_identity_delta_pct", 5.0)),
            "baseline_identity_delta_exceeds_tolerance",
        ),
        (
            "postprocess/signed_baseline_identity_comparison.csv",
            ("signed_identity_delta_pct",),
            float(getattr(args, "max_signed_baseline_identity_delta_pct", 5.0)),
            "signed_baseline_identity_delta_exceeds_tolerance",
        ),
    )
    report: dict[str, Any] = {}
    for artifact, columns, threshold, issue_code in checks:
        path = run_root / artifact
        if not path.is_file():
            if require_artifacts:
                issue(
                    issues,
                    "error",
                    "missing_baseline_identity_artifact",
                    f"Missing required baseline-identity artifact: {path}",
                )
            continue
        payload = read_postprocess_csv_rows(path, issues, artifact=artifact)
        if payload is None:
            continue
        fieldnames, rows = payload
        if not set(columns).issubset(fieldnames):
            if require_artifacts:
                missing_columns = [column for column in columns if column not in fieldnames]
                issue(
                    issues,
                    "error",
                    "baseline_identity_artifact_schema_mismatch",
                    f"Baseline-identity artifact {artifact} is missing required columns: {', '.join(missing_columns)}.",
                )
            continue
        if require_artifacts and not rows:
            issue(
                issues,
                "error",
                "baseline_identity_artifact_empty",
                f"Baseline-identity artifact {artifact} has no data rows.",
            )
        report[artifact] = _audit_identity_delta_columns(
            rows,
            issues,
            artifact=artifact,
            columns=columns,
            max_abs_delta_pct=threshold,
            issue_code=issue_code,
        )
    return report or None


def audit_rainflow_validation_artifact(path: Path, issues: list[AuditIssue], *, artifact: str) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                issue(issues, "error", "postprocess_artifact_empty", f"Rainflow validation artifact has no header: {artifact}")
                return 0
            rows = list(reader)
    except Exception as exc:
        issue(issues, "error", "postprocess_artifact_unreadable", f"Could not read rainflow validation artifact {artifact}: {exc}")
        return 0
    if not rows:
        issue(issues, "error", "postprocess_artifact_empty", f"Rainflow validation artifact has no data rows: {artifact}")
        return 0
    if "passed" not in (reader.fieldnames or []):
        issue(issues, "error", "rainflow_validation_schema_mismatch", f"Rainflow validation artifact has no passed column: {artifact}")
        return len(rows)
    failed = [row for row in rows if str(row.get("passed", "")).strip().lower() not in {"true", "1", "yes"}]
    if failed:
        issue(issues, "error", "rainflow_validation_failed", f"{len(failed)} rainflow validation row(s) failed in {artifact}.")
    return len(rows)


def audit_postprocess_artifacts(
    args: argparse.Namespace,
    issues: list[AuditIssue],
    settings: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not bool(getattr(args, "require_postprocess_artifacts", False)):
        return None
    settings = settings or {}
    enforce_paper_eligibility = not bool(
        getattr(args, "allow_diagnostic_postprocess_artifacts", False)
    )
    scenario_names = settings.get("scenario_names")
    selected_scenarios = set(str(name) for name in scenario_names) if isinstance(scenario_names, list) else set()
    skip_identity_artifacts = selected_scenarios and "identity" not in selected_scenarios
    identity_artifacts = {
        "postprocess/baseline_identity_comparison.csv",
        "postprocess/baseline_identity_comparison_diagnostics.csv",
        "postprocess/signed_baseline_identity_comparison.csv",
        "postprocess/signed_baseline_identity_comparison_diagnostics.csv",
    }
    checked: list[str] = []
    skipped: list[str] = []
    row_counts: dict[str, int] = {}
    for relative, kind in POSTPROCESS_REQUIRED_ARTIFACTS:
        if skip_identity_artifacts and relative in identity_artifacts:
            skipped.append(relative)
            continue
        path = args.run_root / relative
        checked.append(relative)
        if not path.is_file():
            issue(issues, "error", "missing_postprocess_artifact", f"Missing required postprocess artifact: {path}")
            continue
        if path.stat().st_size <= 0:
            issue(issues, "error", "postprocess_artifact_empty", f"Required postprocess artifact is empty: {path}")
            continue
        schema = POSTPROCESS_PAPER_CSV_SCHEMAS.get(relative)
        if kind == "csv" and schema is not None:
            row_counts[relative] = audit_schema_csv_artifact(
                path,
                issues,
                artifact=relative,
                required_columns=schema,
                disallow_smoke_only=(
                    enforce_paper_eligibility
                    and relative in POSTPROCESS_PAPER_COMPARISON_ARTIFACTS
                ),
                enforce_paper_eligibility=enforce_paper_eligibility,
            )
        elif kind == "csv":
            row_counts[relative] = artifact_row_count(path, issues, artifact=relative, kind="CSV")
        elif kind == "rainflow_validation":
            row_counts[relative] = audit_rainflow_validation_artifact(path, issues, artifact=relative)
    signed_diagnostics_subset = audit_signed_del_diagnostics_subset(args.run_root, issues)
    baseline_identity_drift = audit_baseline_identity_drift(args.run_root, args, issues)
    critical_case_selection = audit_postprocess_critical_case_selection(args.run_root, issues)
    critical_rainflow_cycles = audit_critical_case_rainflow_cycles(args.run_root, issues)
    report: dict[str, Any] = {
        "checked": len(checked),
        "artifacts": checked,
        "skipped": skipped,
        "row_counts": row_counts,
    }
    if signed_diagnostics_subset is not None:
        report["signed_diagnostics_subset"] = signed_diagnostics_subset
    if baseline_identity_drift is not None:
        report["baseline_identity_drift"] = baseline_identity_drift
    if critical_case_selection is not None:
        report["critical_case_selection"] = critical_case_selection
    if critical_rainflow_cycles is not None:
        report["critical_rainflow_cycles"] = critical_rainflow_cycles
    return report


def _resolve_archive_artifact_path(run_root: Path, raw_path: object) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = run_root / path
    return path


def expected_archive_run_summary_paths(
    manifests: dict[str, dict[str, Any]],
    issues: list[AuditIssue],
) -> list[Path]:
    paths: list[Path] = []
    for scenario, manifest in manifests.items():
        for case in manifest.get("cases", []):
            case_label = str(case.get("label", case.get("name", ""))).strip() or "unknown"
            raw_result_dir = str(case.get("result_dir", "")).strip()
            if not raw_result_dir:
                issue(
                    issues,
                    "error",
                    "production_archive_expected_run_summary_missing_dir",
                    "Case manifest has no result_dir for expected run-summary archive check.",
                    scenario=scenario,
                    case=case_label,
                )
                continue
            result_dir = Path(raw_result_dir)
            summaries = sorted(result_dir.glob("*_run_summary.json"))
            if not summaries:
                issue(
                    issues,
                    "error",
                    "production_archive_expected_run_summary_missing_file",
                    f"No run-summary file found in expected result directory: {result_dir}",
                    scenario=scenario,
                    case=case_label,
                )
                continue
            paths.extend(summaries)
    return paths


def audit_production_archive_manifest(
    settings: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    if not bool(getattr(args, "require_production_archive_manifest", False)):
        return None

    path = production_archive_manifest_path(args)
    expected_case_count = sum(len(manifest.get("cases", [])) for manifest in manifests.values())
    expected_run_summary_paths = expected_archive_run_summary_paths(manifests, issues)
    if not path.is_file():
        issue(issues, "error", "missing_production_archive_manifest", f"Missing production archive manifest: {path}")
        return {
            "path": str(path),
            "expected_cases": expected_case_count,
            "expected_run_summaries": len(expected_run_summary_paths),
            "artifacts": 0,
            "category_counts": {},
            "required_categories": list(PRODUCTION_ARCHIVE_REQUIRED_CATEGORIES),
        }

    try:
        payload = read_json(path)
    except Exception as exc:
        issue(issues, "error", "production_archive_manifest_unreadable", f"Could not read production archive manifest {path}: {exc}")
        return {
            "path": str(path),
            "expected_cases": expected_case_count,
            "expected_run_summaries": len(expected_run_summary_paths),
            "artifacts": 0,
            "category_counts": {},
            "required_categories": list(PRODUCTION_ARCHIVE_REQUIRED_CATEGORIES),
        }
    if not isinstance(payload, dict):
        issue(issues, "error", "production_archive_manifest_schema_mismatch", "Production archive manifest must be a JSON object.")
        return {
            "path": str(path),
            "expected_cases": expected_case_count,
            "expected_run_summaries": len(expected_run_summary_paths),
            "artifacts": 0,
            "category_counts": {},
            "required_categories": list(PRODUCTION_ARCHIVE_REQUIRED_CATEGORIES),
        }

    if not str(payload.get("archive_id", "")).strip():
        issue(issues, "error", "production_archive_manifest_missing_id", "Production archive manifest is missing archive_id.")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        issue(issues, "error", "production_archive_manifest_no_artifacts", "Production archive manifest must contain a nonempty artifacts list.")
        artifacts = []

    category_counts: dict[str, int] = {}
    archived_run_summary_paths: set[Path] = set()
    checked_artifacts = 0
    for index, item in enumerate(artifacts, start=1):
        if not isinstance(item, dict):
            issue(issues, "error", "production_archive_artifact_schema_mismatch", f"Archive artifact {index} is not a JSON object.")
            continue
        category = str(item.get("category", "")).strip()
        if not category:
            issue(issues, "error", "production_archive_artifact_missing_category", f"Archive artifact {index} is missing category.")
        artifact_path = _resolve_archive_artifact_path(args.run_root, item.get("path"))
        if artifact_path is None:
            issue(issues, "error", "production_archive_artifact_missing_path", f"Archive artifact {index} is missing path.")
            continue
        if not artifact_path.is_file():
            issue(issues, "error", "production_archive_artifact_missing_file", f"Archive artifact file does not exist: {artifact_path}")
            continue
        actual_size = artifact_path.stat().st_size
        if actual_size <= 0:
            issue(issues, "error", "production_archive_artifact_empty", f"Archive artifact file is empty: {artifact_path}")
        expected_sha = str(item.get("sha256", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            issue(issues, "error", "production_archive_artifact_missing_sha256", f"Archive artifact {index} is missing a valid sha256.")
        else:
            actual_sha = sha256_file(artifact_path)
            if actual_sha.lower() != expected_sha:
                issue(issues, "error", "production_archive_artifact_sha256_mismatch", f"Archive artifact hash mismatch: {artifact_path}")
        try:
            expected_size = int(item.get("bytes", ""))
        except (TypeError, ValueError):
            issue(issues, "error", "production_archive_artifact_missing_size", f"Archive artifact {index} is missing integer bytes.")
        else:
            if expected_size <= 0:
                issue(issues, "error", "production_archive_artifact_nonpositive_size", f"Archive artifact {index} has nonpositive bytes.")
            elif expected_size != actual_size:
                issue(
                    issues,
                    "error",
                    "production_archive_artifact_size_mismatch",
                    f"Archive artifact size mismatch for {artifact_path}: manifest {expected_size}, actual {actual_size}.",
                )
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
            if category == "run_summary":
                archived_run_summary_paths.add(artifact_path.resolve())
        checked_artifacts += 1

    for category in PRODUCTION_ARCHIVE_REQUIRED_CATEGORIES:
        if category_counts.get(category, 0) <= 0:
            issue(
                issues,
                "error",
                "production_archive_manifest_missing_category",
                f"Production archive manifest has no artifact in required category {category}.",
            )
    if expected_case_count > 0 and category_counts.get("run_summary", 0) < expected_case_count:
        issue(
            issues,
            "error",
            "production_archive_manifest_missing_run_summaries",
            (
                f"Production archive manifest has {category_counts.get('run_summary', 0)} run_summary artifact(s), "
                f"below expected case count {expected_case_count}."
            ),
        )
    for expected_summary in expected_run_summary_paths:
        if expected_summary.resolve() not in archived_run_summary_paths:
            issue(
                issues,
                "error",
                "production_archive_manifest_missing_expected_run_summary",
                f"Production archive manifest does not include expected run summary: {expected_summary}",
            )

    return {
        "path": str(path),
        "archive_id": str(payload.get("archive_id", "")),
        "expected_cases": expected_case_count,
        "expected_run_summaries": len(expected_run_summary_paths),
        "artifacts": checked_artifacts,
        "category_counts": category_counts,
        "required_categories": list(PRODUCTION_ARCHIVE_REQUIRED_CATEGORIES),
    }


def audit_source_coverage(
    settings: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    issues: list[AuditIssue],
) -> dict[str, Any] | None:
    path = source_coverage_path(args)
    require = bool(getattr(args, "require_source_coverage", False))
    if not path.is_file():
        if require:
            issue(issues, "error", "missing_source_coverage", f"Missing aerodynamic source-coverage CSV: {path}")
        return None

    requested_scenarios = source_coverage_scenarios(settings, args, issues)
    requested = set(requested_scenarios)
    expected_cases: set[tuple[str, str]] = set()
    expected_case_values: dict[tuple[str, str], tuple[float, float]] = {}
    expected_case_components: dict[tuple[str, str], set[str]] = {}
    expected_component_node_counts: dict[tuple[str, str, str], int] = {}
    expected_case_completed_steps: dict[tuple[str, str], int] = {}
    for scenario in requested_scenarios:
        manifest = manifests.get(scenario)
        if manifest is None:
            continue
        for case in manifest.get("cases", []):
            label = str(case.get("label", ""))
            if label:
                key = (scenario, label)
                expected_cases.add(key)
                expected_case_values[key] = normalized_case_key(case.get("wind_speed", math.nan), case.get("azimuth_deg", math.nan))
                component_counts = expected_source_component_node_counts_for_case(
                    case,
                    issues,
                    scenario=scenario,
                    case_label=label,
                    require=require,
                )
                if component_counts:
                    expected_case_components[key] = set(component_counts)
                    for component, node_count in component_counts.items():
                        expected_component_node_counts[(scenario, label, component)] = node_count
                if require:
                    completed_steps = completed_steps_for_case(case, issues, scenario=scenario, case_label=label)
                    if completed_steps is not None:
                        expected_case_completed_steps[key] = completed_steps

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [column for column in SOURCE_COVERAGE_REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            issue(
                issues,
                "error",
                "source_coverage_schema_mismatch",
                f"Source-coverage CSV is missing required columns: {', '.join(missing_columns)}.",
            )
            return {
                "path": str(path),
                "rows": 0,
                "covered_cases": 0,
                "expected_cases": len(expected_cases),
            }
        rows = list(reader)

    covered_cases: set[tuple[str, str]] = set()
    covered_components: dict[tuple[str, str], set[str]] = {}
    min_nyquist_ratio = float(getattr(args, "min_nyquist_ratio", 1.0))
    count_allowance = int(getattr(args, "allow_source_coverage_violations", 0))
    allow_source_range_excursions = bool(getattr(args, "allow_source_range_excursions", False))
    source_range_severity = "warning" if allow_source_range_excursions else "error"
    tolerance = float(getattr(args, "source_range_tolerance", 1.0e-9))
    source_manifest = settings.get("airfoil_source_manifest", [])
    if not isinstance(source_manifest, list):
        source_manifest = []
    source_models = {
        str(source.get("airfoil_id", "")).strip(): str(source.get("reynolds_model", "")).strip()
        for source in source_manifest
        if isinstance(source, dict) and str(source.get("airfoil_id", "")).strip()
    }
    re_lower_bound_warning_total = 0

    for row in rows:
        scenario = str(row.get("scenario", "")).strip()
        case = str(row.get("case", "")).strip()
        component = str(row.get("component", "")).strip()
        if not scenario or not case or not component:
            issue(
                issues,
                "error",
                "incomplete_source_coverage_identity",
                "Source-coverage rows must include nonempty scenario, case, and component fields.",
                scenario=scenario,
                case=case,
            )
            continue
        if scenario not in requested:
            continue
        key = (scenario, case)
        covered_cases.add(key)
        if key in expected_cases:
            covered_components.setdefault(key, set()).add(component)
            expected_components = expected_case_components.get(key)
            if expected_components and component not in expected_components:
                issue(
                    issues,
                    "error" if require else "warning",
                    "unexpected_source_coverage_component",
                    f"Source-coverage component {component!r} is not in the generated VorLap runtime config.",
                    scenario=scenario,
                    case=case,
                )
        if key not in expected_cases:
            issue(
                issues,
                "warning",
                "unexpected_source_coverage_case",
                f"Source-coverage row for component {component} is not in the expected campaign manifest.",
                scenario=scenario,
                case=case,
            )

        values = {
            name: row_float(row, name, issues, scenario=scenario, case=case, component=component)
            for name in SOURCE_COVERAGE_REQUIRED_COLUMNS
            if name not in {"scenario", "case", "component"}
        }
        if any(value is None for value in values.values()):
            continue

        expected_case_key = expected_case_values.get((scenario, case))
        if expected_case_key is not None and normalized_case_key(values["wind_speed_mps"], values["azimuth_deg"]) != expected_case_key:
            issue(
                issues,
                "error",
                "source_coverage_case_mismatch",
                "Source-coverage wind speed or azimuth does not match the generated case manifest.",
                scenario=scenario,
                case=case,
            )
        if values["aoa_min_deg"] > values["aoa_max_deg"] + tolerance:
            issue(issues, "error", "source_coverage_invalid_aoa_range", f"{component} AOA min exceeds AOA max.", scenario=scenario, case=case)
        if values["re_min"] > values["re_max"] + tolerance:
            issue(issues, "error", "source_coverage_invalid_re_range", f"{component} Reynolds min exceeds Reynolds max.", scenario=scenario, case=case)
        if values["freq_min_hz"] > values["freq_max_hz"] + tolerance:
            issue(issues, "error", "source_coverage_invalid_frequency_range", f"{component} frequency min exceeds frequency max.", scenario=scenario, case=case)
        if values["aoa_min_deg"] < values["source_aoa_min_deg"] - tolerance or values["aoa_max_deg"] > values["source_aoa_max_deg"] + tolerance:
            issue(
                issues,
                source_range_severity,
                "source_aoa_out_of_range",
                (
                    f"{component} AOA range [{values['aoa_min_deg']:g}, {values['aoa_max_deg']:g}] deg "
                    f"exceeds source range [{values['source_aoa_min_deg']:g}, {values['source_aoa_max_deg']:g}] deg."
                ),
                scenario=scenario,
                case=case,
            )
        lower_re_excursion = values["re_min"] < values["source_re_min"] - tolerance
        upper_re_excursion = values["re_max"] > values["source_re_max"] + tolerance
        airfoil_id = str(row.get("airfoil_id", "")).strip()
        reynolds_model = str(row.get("reynolds_model", "")).strip()
        lower_warning_count = row_count(
            row,
            "re_lower_bound_warning_count",
            issues,
            scenario=scenario,
            case=case,
            component=component,
        )
        lower_re_warning_allowed = (
            lower_re_excursion
            and not upper_re_excursion
            and lower_warning_count is not None
            and lower_warning_count > 0
            and reynolds_model == "replicated_reynolds_independent"
            and source_models.get(airfoil_id) == "replicated_reynolds_independent"
        )
        if lower_warning_count:
            re_lower_bound_warning_total += lower_warning_count
        if lower_re_warning_allowed:
            issue(
                issues,
                "warning",
                "source_re_lower_bound_warning",
                (
                    f"{component} Reynolds minimum {values['re_min']:g} is below source minimum "
                    f"{values['source_re_min']:g}; {lower_warning_count} node-samples were clipped "
                    "to an exactly replicated Reynolds-independent source plane."
                ),
                scenario=scenario,
                case=case,
            )
        elif lower_re_excursion or upper_re_excursion:
            issue(
                issues,
                source_range_severity,
                "source_re_out_of_range",
                (
                    f"{component} Reynolds range [{values['re_min']:g}, {values['re_max']:g}] "
                    f"exceeds source range [{values['source_re_min']:g}, {values['source_re_max']:g}]."
                ),
                scenario=scenario,
                case=case,
            )
        if lower_warning_count and not lower_re_warning_allowed:
            issue(
                issues,
                "error",
                "invalid_re_lower_bound_warning",
                (
                    f"{component} reports re_lower_bound_warning_count={lower_warning_count} without "
                    "a matching lower-only excursion and replicated_reynolds_independent source manifest."
                ),
                scenario=scenario,
                case=case,
            )
        if values["freq_max_hz"] > values["nyquist_hz"] + tolerance:
            issue(
                issues,
                "error",
                "source_frequency_above_nyquist",
                f"{component} max frequency {values['freq_max_hz']:g} Hz exceeds Nyquist {values['nyquist_hz']:g} Hz.",
                scenario=scenario,
                case=case,
            )
        if values["nyquist_ratio"] < min_nyquist_ratio - tolerance:
            issue(
                issues,
                "error",
                "source_nyquist_margin_too_low",
                f"{component} Nyquist ratio {values['nyquist_ratio']:g} is below required {min_nyquist_ratio:g}.",
                scenario=scenario,
                case=case,
            )
        if values["force_coefficient_max_abs"] > values["source_force_coefficient_max_abs"] + tolerance:
            issue(
                issues,
                source_range_severity,
                "source_force_coefficient_out_of_range",
                (
                    f"{component} max force coefficient {values['force_coefficient_max_abs']:g} "
                    f"exceeds source maximum {values['source_force_coefficient_max_abs']:g}."
                ),
                scenario=scenario,
                case=case,
            )
        if values["max_nodal_force_n"] < -tolerance:
            issue(issues, "error", "negative_max_nodal_force", f"{component} max_nodal_force_n is negative.", scenario=scenario, case=case)
        if values["max_structural_velocity_mps"] < -tolerance:
            issue(issues, "error", "negative_max_structural_velocity", f"{component} max_structural_velocity_mps is negative.", scenario=scenario, case=case)
        if values["force_power_max_abs_w"] < -tolerance:
            issue(issues, "error", "negative_force_power_max_abs", f"{component} force_power_max_abs_w is negative.", scenario=scenario, case=case)
        source_samples = int(round(values["samples"]))
        source_node_samples = int(round(values["node_samples"]))
        if not close_enough(values["samples"], source_samples, abs_tol=tolerance):
            issue(issues, "error", "source_coverage_sample_count_noninteger", f"{component} samples={values['samples']:g} is not an integer.", scenario=scenario, case=case)
        if not close_enough(values["node_samples"], source_node_samples, abs_tol=tolerance):
            issue(issues, "error", "source_coverage_node_sample_count_noninteger", f"{component} node_samples={values['node_samples']:g} is not an integer.", scenario=scenario, case=case)
        if source_samples < 1:
            issue(issues, "error", "source_coverage_no_samples", f"{component} has no source-coverage samples.", scenario=scenario, case=case)
        expected_steps = expected_case_completed_steps.get(key)
        if expected_steps is not None and source_samples != expected_steps:
            issue(
                issues,
                "error",
                "source_coverage_sample_count_mismatch",
                f"{component} source-coverage samples {source_samples} != completed steps {expected_steps}.",
                scenario=scenario,
                case=case,
            )
        expected_node_count = expected_component_node_counts.get((scenario, case, component))
        if expected_node_count is not None:
            expected_node_samples = source_samples * expected_node_count
            if source_node_samples != expected_node_samples:
                issue(
                    issues,
                    "error",
                    "source_coverage_node_sample_count_mismatch",
                    f"{component} node_samples {source_node_samples} != samples*expected_nodes {expected_node_samples}.",
                    scenario=scenario,
                    case=case,
                )

        for column in SOURCE_COVERAGE_COUNT_COLUMNS:
            count = row_count(row, column, issues, scenario=scenario, case=case, component=component)
            if count is not None and count > count_allowance:
                severity = (
                    "warning"
                    if allow_source_range_excursions and column in SOURCE_RANGE_EXCURSION_COUNT_COLUMNS
                    else "error"
                )
                issue(
                    issues,
                    severity,
                    "source_coverage_violation_count",
                    f"{component} reports {column}={count}, above allowed {count_allowance}.",
                    scenario=scenario,
                    case=case,
                )

    missing_cases = sorted(expected_cases - covered_cases)
    for scenario, case in missing_cases:
        issue(
            issues,
            "error",
            "missing_source_coverage_case",
            "Aerodynamic source-coverage CSV has no row for this expected production case.",
            scenario=scenario,
            case=case,
        )
    if require:
        for key, expected_components in sorted(expected_case_components.items()):
            missing_components = sorted(expected_components - covered_components.get(key, set()))
            for component in missing_components:
                issue(
                    issues,
                    "error",
                    "missing_source_coverage_component",
                    f"Aerodynamic source-coverage CSV has no row for expected component {component}.",
                    scenario=key[0],
                    case=key[1],
                )

    return {
        "path": str(path),
        "rows": len(rows),
        "covered_cases": len(covered_cases),
        "expected_cases": len(expected_cases),
        "covered_case_components": sum(len(components) for components in covered_components.values()),
        "expected_case_components": sum(len(components) for components in expected_case_components.values()),
        "expected_component_node_count_total": sum(expected_component_node_counts.values()),
        "re_lower_bound_warning_count": re_lower_bound_warning_total,
        "scenarios": requested_scenarios,
    }


def audit_campaign(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[AuditIssue] = []
    source_coverage_report: dict[str, Any] | None = None
    geometry_position_report: dict[str, Any] | None = None
    spectral_force_report: dict[str, Any] | None = None
    spectral_psd_comparison_report: dict[str, Any] | None = None
    native_callback_proof_report: dict[str, Any] | None = None
    force_sign_probe_report: dict[str, Any] | None = None
    ultimate_margin_report: dict[str, Any] | None = None
    ultimate_input_basis_report: dict[str, Any] | None = None
    baseline_identity_report: dict[str, Any] | None = None
    postprocess_artifact_report: dict[str, Any] | None = None
    production_archive_report: dict[str, Any] | None = None
    structural_model_qualification_report: dict[str, Any] | None = None
    signed_channel_selection_report: dict[str, Any] | None = None
    settings_path = args.run_root / "campaign_settings.json"
    if not settings_path.is_file():
        issue(issues, "error", "missing_campaign_settings", f"Missing campaign settings: {settings_path}")
        settings: dict[str, Any] = {}
    else:
        settings = read_json(settings_path)
        audit_plan(settings, args, issues)

    manifests: dict[str, dict[str, Any]] = {}
    if args.stage in ("setup", "sign-probes", "runtime", "results") and settings:
        for scenario in expected_scenarios(settings, args):
            manifest = load_manifest(args.run_root, scenario, issues)
            if manifest is None:
                continue
            manifests[scenario] = manifest
            audit_manifest(manifest, settings, args, issues, scenario=scenario)
    if args.stage == "sign-probes" and settings:
        force_sign_probe_report = audit_force_sign_probes(settings, args, issues, manifests)
    if args.stage in ("runtime", "results") and settings:
        for scenario, manifest in manifests.items():
            audit_results(manifest, settings, args, issues, scenario=scenario)
    if args.stage == "runtime" and settings:
        if bool(getattr(args, "require_baseline_identity_proof", False)):
            baseline_identity_report = audit_baseline_identity_drift(
                args.run_root,
                args,
                issues,
                require_artifacts=True,
            )
        postprocess_artifact_report = audit_postprocess_artifacts(args, issues, settings)
    if args.stage == "results" and settings:
        source_coverage_report = audit_source_coverage(settings, manifests, args, issues)
        geometry_position_report = audit_geometry_position_proof(settings, manifests, args, issues)
        spectral_force_report = audit_spectral_force_content(settings, manifests, args, issues)
        spectral_psd_comparison_report = audit_spectral_psd_comparison(settings, manifests, args, issues)
        native_callback_proof_report = audit_native_callback_proof(settings, manifests, args, issues)
        force_sign_probe_report = audit_force_sign_probes(settings, args, issues, manifests)
        ultimate_margin_report = audit_ultimate_margins(settings, manifests, args, issues)
        ultimate_input_basis_report = audit_ultimate_input_basis(args, issues)
        structural_model_qualification_report = audit_structural_model_qualification(settings, manifests, args, issues)
        signed_channel_selection_report = audit_signed_channel_selection(settings, args, issues)
        if bool(getattr(args, "require_baseline_identity_proof", False)):
            baseline_identity_report = audit_baseline_identity_drift(
                args.run_root,
                args,
                issues,
                require_artifacts=True,
            )
        postprocess_artifact_report = audit_postprocess_artifacts(args, issues, settings)
        production_archive_report = audit_production_archive_manifest(settings, manifests, args, issues)

    errors = [item for item in issues if item.severity == "error"]
    warnings = [item for item in issues if item.severity == "warning"]
    report = {
        "passed": not errors,
        "stage": args.stage,
        "run_root": str(args.run_root),
        "summary": {
            "errors": len(errors),
            "warnings": len(warnings),
            "issues": len(issues),
        },
        "issues": [item.as_dict() for item in issues],
    }
    if source_coverage_report is not None:
        report["source_coverage"] = source_coverage_report
    if geometry_position_report is not None:
        report["geometry_position_proof"] = geometry_position_report
    if spectral_force_report is not None:
        report["spectral_force_content"] = spectral_force_report
    if spectral_psd_comparison_report is not None:
        report["spectral_psd_comparison"] = spectral_psd_comparison_report
    if native_callback_proof_report is not None:
        report["native_callback_proof"] = native_callback_proof_report
    if force_sign_probe_report is not None:
        report["force_sign_probes"] = force_sign_probe_report
    if ultimate_margin_report is not None:
        report["ultimate_margins"] = ultimate_margin_report
    if ultimate_input_basis_report is not None:
        report["ultimate_input_basis"] = ultimate_input_basis_report
    if structural_model_qualification_report is not None:
        report["structural_model_qualification"] = structural_model_qualification_report
    if signed_channel_selection_report is not None:
        report["signed_channel_selection"] = signed_channel_selection_report
    if baseline_identity_report is not None:
        report["baseline_identity_proof"] = baseline_identity_report
    if postprocess_artifact_report is not None:
        report["postprocess_artifacts"] = postprocess_artifact_report
    if production_archive_report is not None:
        report["production_archive_manifest"] = production_archive_report
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/vorlap_parked_demo"))
    parser.add_argument(
        "--stage",
        choices=("plan", "setup", "sign-probes", "runtime", "results"),
        default="results",
        help=(
            "Audit stage. 'runtime' checks execution/checkpoint/postprocess integrity "
            "without applying production source/modal/ultimate evidence gates."
        ),
    )
    parser.add_argument("--scenarios", default="baseline vorlap")
    parser.add_argument("--min-duration", type=float, default=600.0)
    parser.add_argument(
        "--min-analyzed-duration",
        type=float,
        default=0.0,
        help="Minimum required duration after store_from/transient exclusion.",
    )
    parser.add_argument("--max-timestep", type=float, default=0.01)
    parser.add_argument("--expected-rpm", type=float, default=0.0)
    parser.add_argument(
        "--rpm-tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed exported LSS RPM deviation after startup/transient exclusion.",
    )
    parser.add_argument(
        "--require-wind-speeds",
        default="",
        help="Optional exact wind-speed grid required for production audits. Accepts lists, linspace:start:stop:count, or range:start:stop:step.",
    )
    parser.add_argument(
        "--require-azimuths",
        default="",
        help="Optional exact parked-azimuth grid required for production audits. Accepts lists, linspace:start:stop:count, or range:start:stop:step.",
    )
    parser.add_argument(
        "--require-airfoil-source-manifest",
        action="store_true",
        help="Require archived hashes, Reynolds grids, and thickness metadata for the HDF5 airfoil sources.",
    )
    parser.add_argument(
        "--require-dlc-manifest-row",
        action="store_true",
        help="Require the campaign to match one immutable row from an approved DLC manifest.",
    )
    parser.add_argument(
        "--require-wind-record",
        action="store_true",
        help="Require a hashed wind sidecar matching the approved DLC manifest row.",
    )
    parser.add_argument("--require-wind-name", action="store_true", help="Require a nonempty wind_name template for DLC audits.")
    parser.add_argument("--allow-quasi-static", action="store_true", help="Allow quasi-static azimuth-sweep diagnostics.")
    parser.add_argument(
        "--require-fresh-run",
        action="store_true",
        help="Reject any case that resumed or quarantined a prior checkpoint; intended for fresh-root acceptance trials.",
    )
    parser.add_argument(
        "--source-coverage",
        type=Path,
        default=None,
        help="Optional aerodynamic source-coverage CSV. Defaults to RUN_ROOT/postprocess/source_coverage.csv when present.",
    )
    parser.add_argument(
        "--require-source-coverage",
        action="store_true",
        help="Fail results audits when the aerodynamic source-coverage CSV is missing.",
    )
    parser.add_argument(
        "--source-coverage-scenarios",
        default="vorlap",
        help="Scenarios that must have aerodynamic source coverage. Defaults to the nominal VorLap scenario.",
    )
    parser.add_argument("--min-nyquist-ratio", type=float, default=1.0, help="Minimum allowed Nyquist/max-frequency ratio in source coverage rows.")
    parser.add_argument(
        "--allow-source-coverage-violations",
        type=int,
        default=0,
        help="Allowed count for explicit source coverage violation columns. Production evidence should normally keep this at zero.",
    )
    parser.add_argument(
        "--allow-source-range-excursions",
        action="store_true",
        help=(
            "Downgrade Re/AOA/force-coefficient source-grid excursions to warnings. "
            "Use only for exploratory runs with incomplete airfoil source grids."
        ),
    )
    parser.add_argument(
        "--source-range-tolerance",
        type=float,
        default=1.0e-9,
        help="Absolute tolerance for source range comparisons in aerodynamic source coverage rows.",
    )
    parser.add_argument(
        "--require-geometry-position-proof",
        action="store_true",
        help="Fail results audits unless QBlade-exported positions are recorded in geometry audit CSVs and fitted-frame residuals are within tolerance.",
    )
    parser.add_argument(
        "--geometry-proof-scenarios",
        default="vorlap",
        help="Scenarios that must have QBlade-reported geometry position proof. Defaults to the nominal VorLap scenario.",
    )
    parser.add_argument(
        "--max-geometry-frame-fit-residual-m",
        type=float,
        default=0.10,
        help="Maximum allowed fitted-frame residual for QBlade-reported geometry proof rows.",
    )
    parser.add_argument(
        "--min-geometry-frame-fit-points",
        type=int,
        default=3,
        help="Minimum fitted geometry points required for a frame-fit geometry proof.",
    )
    parser.add_argument(
        "--allow-missing-geometry-reported-nodes",
        type=int,
        default=0,
        help="Allowed count of geometry audit rows without QBlade-reported position channels.",
    )
    parser.add_argument(
        "--require-spectral-force-content",
        action="store_true",
        help="Fail results audits unless recorded SIL VorLap force histories have dynamic spectral power inside each component source-frequency band.",
    )
    parser.add_argument(
        "--spectral-force-scenarios",
        default="vorlap",
        help="Scenarios that must have recorded SIL VorLap spectral force-content evidence. Defaults to the nominal VorLap scenario.",
    )
    parser.add_argument(
        "--min-spectral-force-samples",
        type=int,
        default=16,
        help="Minimum SIL VorLap history samples required for spectral force-content evidence.",
    )
    parser.add_argument(
        "--min-spectral-force-dynamic-power",
        type=float,
        default=1.0e-12,
        help="Minimum positive-frequency force power required for a channel to count as active.",
    )
    parser.add_argument(
        "--min-spectral-force-band-power-fraction",
        type=float,
        default=0.95,
        help="Minimum active force-channel dynamic power fraction that must lie inside the source-frequency band.",
    )
    parser.add_argument(
        "--spectral-psd-comparison",
        type=Path,
        default=None,
        help=f"Optional analytic/offline/SIL/QBlade PSD comparison CSV. Defaults to RUN_ROOT/{SPECTRAL_PSD_COMPARISON_DEFAULT}.",
    )
    parser.add_argument(
        "--require-spectral-psd-comparison",
        action="store_true",
        help="Fail results audits unless an analytic/offline/SIL/QBlade PSD comparison CSV is complete and passing.",
    )
    parser.add_argument(
        "--spectral-psd-comparison-scenarios",
        default="vorlap",
        help="Scenarios that must have final PSD comparison evidence. Defaults to the nominal VorLap scenario.",
    )
    parser.add_argument(
        "--max-spectral-psd-frequency-error-hz",
        type=float,
        default=0.05,
        help="Maximum allowed absolute PSD peak-frequency error relative to the analytic expected frequency.",
    )
    parser.add_argument(
        "--min-spectral-psd-band-power-fraction",
        type=float,
        default=0.80,
        help="Minimum allowed PSD comparison band-power fraction for each analytic/offline/SIL/QBlade row.",
    )
    parser.add_argument(
        "--require-native-callback-proof",
        action="store_true",
        help="Fail results audits unless at least one requested case proves the qblade-external native callback path.",
    )
    parser.add_argument(
        "--native-callback-proof-scenarios",
        default="vorlap",
        help="Scenarios that must provide native callback proof evidence. Defaults to the nominal VorLap scenario.",
    )
    parser.add_argument(
        "--force-sign-probe-report",
        type=Path,
        default=None,
        help="Optional force-sign probe CSV. Defaults to RUN_ROOT/sign_probes/force_sign_probe_report.csv when present.",
    )
    parser.add_argument(
        "--require-force-sign-probes",
        action="store_true",
        help="Fail results audits when the force-sign probe report is missing or lacks a complete passing axis/sign set.",
    )
    parser.add_argument(
        "--require-postprocess-artifacts",
        action="store_true",
        help="Fail results audits when paper-facing postprocess CSV/report/plot artifacts are missing or empty.",
    )
    parser.add_argument(
        "--allow-diagnostic-postprocess-artifacts",
        action="store_true",
        help=(
            "Validate the postprocess bundle and numeric schemas while allowing smoke-only rows "
            "and diagnostic warning labels. Intended only for deliberately short acceptance trials."
        ),
    )
    parser.add_argument(
        "--require-baseline-identity-proof",
        action="store_true",
        help=(
            "Fail results audits unless baseline-vs-zero-force-identity postprocess "
            "artifacts exist and their DEL percent deltas stay within the configured thresholds."
        ),
    )
    parser.add_argument(
        "--production-archive-manifest",
        type=Path,
        default=None,
        help=f"Optional production archive manifest JSON. Defaults to RUN_ROOT/{PRODUCTION_ARCHIVE_MANIFEST_DEFAULT}.",
    )
    parser.add_argument(
        "--require-production-archive-manifest",
        action="store_true",
        help="Fail results audits unless logs, summaries, audits, postprocess CSVs, and figure provenance are listed with hashes in a manifest.",
    )
    parser.add_argument(
        "--max-baseline-identity-delta-pct",
        type=float,
        default=5.0,
        help="Maximum allowed absolute resultant DEL percent delta between no-SIL baseline and zero-force identity.",
    )
    parser.add_argument(
        "--max-signed-baseline-identity-delta-pct",
        type=float,
        default=5.0,
        help="Maximum allowed absolute signed-channel DEL percent delta between no-SIL baseline and zero-force identity.",
    )
    parser.add_argument(
        "--force-sign-required-axes",
        default="X Y Z",
        help="Axes that must have positive and negative passing sign probes. Defaults to X Y Z.",
    )
    parser.add_argument(
        "--force-sign-required-node",
        action="append",
        default=[],
        help="Specific QBlade node id that must have a complete passing force-sign probe set. Repeat for multiple nodes.",
    )
    parser.add_argument(
        "--ultimate-margins",
        type=Path,
        default=None,
        help="Optional ultimate-margin CSV. Defaults to RUN_ROOT/postprocess/ultimate_margin_metrics.csv when --require-ultimate-margins is set.",
    )
    parser.add_argument(
        "--require-ultimate-margins",
        action="store_true",
        help="Fail results audits when the ultimate-margin CSV is missing or has incomplete/invalid allowables or demand factors.",
    )
    parser.add_argument(
        "--ultimate-input-basis",
        type=Path,
        default=None,
        help=f"Optional ultimate input-basis JSON. Defaults to RUN_ROOT/{ULTIMATE_INPUT_BASIS_DEFAULT}.",
    )
    parser.add_argument(
        "--require-ultimate-input-basis",
        action="store_true",
        help="Fail results audits unless ultimate allowables, channel mapping, and factor basis are documented by a JSON manifest.",
    )
    parser.add_argument(
        "--structural-model-qualification",
        type=Path,
        default=None,
        help=f"Optional structural-model qualification JSON. Defaults to RUN_ROOT/{STRUCTURAL_QUALIFICATION_DEFAULT}.",
    )
    parser.add_argument(
        "--require-structural-model-qualification",
        action="store_true",
        help="Fail results audits unless the exact final QBlade structural model is qualified by a JSON manifest.",
    )
    parser.add_argument(
        "--signed-channel-selection",
        type=Path,
        default=None,
        help=f"Optional signed fatigue-channel selection JSON. Defaults to RUN_ROOT/{SIGNED_CHANNEL_SELECTION_DEFAULT}.",
    )
    parser.add_argument(
        "--require-signed-channel-selection",
        action="store_true",
        help="Fail results audits unless the final signed fatigue channels and normalization convention are documented by a JSON manifest.",
    )
    parser.add_argument(
        "--ultimate-margin-scenarios",
        default="vorlap",
        help="Scenarios that must have ultimate-margin rows. Defaults to the nominal VorLap scenario.",
    )
    parser.add_argument(
        "--ultimate-margin-tolerance",
        type=float,
        default=1.0e-9,
        help="Absolute tolerance for ultimate margin consistency checks.",
    )
    parser.add_argument(
        "--fail-on-ultimate-exceedance",
        action="store_true",
        help="Treat ultimate_status=exceeds_allowable as an audit error instead of a warning.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON audit output path.")
    parser.add_argument("--no-fail", action="store_true", help="Exit 0 even when audit errors are found.")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if args.source_coverage is not None:
        args.source_coverage = args.source_coverage.resolve()
    if args.force_sign_probe_report is not None:
        args.force_sign_probe_report = args.force_sign_probe_report.resolve()
    if args.ultimate_margins is not None:
        args.ultimate_margins = args.ultimate_margins.resolve()
    if args.ultimate_input_basis is not None:
        args.ultimate_input_basis = args.ultimate_input_basis.resolve()
    if args.structural_model_qualification is not None:
        args.structural_model_qualification = args.structural_model_qualification.resolve()
    if args.signed_channel_selection is not None:
        args.signed_channel_selection = args.signed_channel_selection.resolve()
    return args


def main() -> None:
    args = parse_args()
    report = audit_campaign(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"] and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
