#!/usr/bin/env python3
"""Assess the resolved-peak engineering lock-in campaign against Ehrmann data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import campaign
from vorlap.qblade_run_support import atomic_write_json

SCHEMA = "vorlap-ehrmann2014-engineering-validation-evidence-v1"
FIGURE6 = campaign.HERE / "reference_data/ehrmann2014_figure6_naca0018_digitized.csv"
FIGURE7 = (
    campaign.HERE / "reference_data/ehrmann2014_figure7_naca0018_peak_digitized.csv"
)
DIGITIZATION_SIGMA_A_OVER_C = 0.0006
MAX_AMPLITUDE_RELATIVE_ERROR = 0.30
MAX_PEAK_USTAR_RELATIVE_ERROR = 0.10
MAX_FREQUENCY_RELATIVE_ERROR = 0.10
MAX_DT_AMPLITUDE_DIFFERENCE_A_OVER_C = 0.01


class EngineeringEvidenceError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_execution_report(
    root: Path, report_path: Path, manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    if report_path.is_symlink() or not report_path.is_file():
        raise EngineeringEvidenceError("Engineering execution report is missing.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    contract = report.get("execution_contract")
    if (
        report.get("status") != "complete"
        or not isinstance(contract, Mapping)
        or report.get("campaign_fingerprint", manifest["fingerprint"])
        != manifest["fingerprint"]
        or contract.get("campaign_schema") != campaign.ENGINEERING_LOCKIN_SCHEMA
        or contract.get("calibration_sha256")
        != manifest["engineering_lockin_identity"]["sha256"]
        or contract.get("exact_raw_reuse_case_count") != 18
        or contract.get("new_raw_case_count") != 8
        or contract.get("hybrid_case_count") != 26
        or contract.get("complete_at_end") != 52
        or contract.get("nalu_required") is not False
    ):
        raise EngineeringEvidenceError("Engineering execution contract differs.")
    schema = contract.get("schema")
    if schema == "vorlap-ehrmann2014-engineering-flight-execution-v2":
        return contract
    if schema != "vorlap-ehrmann2014-engineering-local-docker-execution-v2":
        raise EngineeringEvidenceError("Engineering execution schema is unsupported.")
    container = contract.get("container")
    inspect_file = (
        container.get("inspect_file") if isinstance(container, Mapping) else None
    )
    if (
        contract.get("threads_per_case") != 1
        or not isinstance(container, Mapping)
        or container.get("platform") != "docker_linux_amd64"
        or container.get("nano_cpus") != 8_000_000_000
        or container.get("oom_killed") is not False
        or not isinstance(container.get("image_id"), str)
        or not str(container["image_id"]).startswith("sha256:")
        or not isinstance(inspect_file, Mapping)
        or set(inspect_file) != {"path", "sha256"}
    ):
        raise EngineeringEvidenceError("Local Docker execution contract differs.")
    inspect_path = Path(str(inspect_file["path"]))
    if not inspect_path.is_absolute():
        inspect_path = root / inspect_path
    if (
        inspect_path.is_symlink()
        or not inspect_path.is_file()
        or _sha256(inspect_path) != inspect_file["sha256"]
    ):
        raise EngineeringEvidenceError("Local Docker inspect identity differs.")
    return contract


def _read_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                key: (value if key in {"source_figure", "series"} else float(value))
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def _experimental_peak(figure7: Sequence[Mapping[str, Any]], md: float) -> float:
    x = np.asarray([float(row["mass_damping"]) for row in figure7])
    y = np.asarray([float(row["maximum_amplitude_over_chord"]) for row in figure7])
    if md <= x[0]:
        return float(y[0])
    if md >= x[-1]:
        return float(y[-1])
    return float(np.interp(md, x, y))


def _adjacent_lockin_points(rows: list[Mapping[str, Any]]) -> list[float]:
    ordered = sorted(rows, key=lambda row: float(row["reduced_velocity"]))
    peak = max(float(row["terminal_amplitude_over_chord"]) for row in ordered)
    active = [
        float(row["reduced_velocity"])
        for row in ordered
        if float(row["terminal_amplitude_over_chord"]) >= 0.5 * peak
        and row["terminal_dominant_frequency_over_fn"] is not None
        and abs(float(row["terminal_dominant_frequency_over_fn"]) - 1.0)
        <= MAX_FREQUENCY_RELATIVE_ERROR
    ]
    grid = list(campaign.ENGINEERING_REDUCED_VELOCITIES)
    active_set = set(active)
    return [
        value
        for index, value in enumerate(grid)
        if value in active_set
        and (
            (index > 0 and grid[index - 1] in active_set)
            or (index + 1 < len(grid) and grid[index + 1] in active_set)
        )
    ]


def assess_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    figure6: Sequence[Mapping[str, Any]],
    figure7: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(rows) != 52:
        raise EngineeringEvidenceError("Exactly 52 analyzed cases are required.")
    experiment_peak_ustar = float(
        max(figure6, key=lambda row: float(row["rms_acceleration_mps2"]))[
            "reduced_velocity"
        ]
    )
    results = []
    all_stationary = all(bool(row["stationarity_amplitude_pass"]) for row in rows)
    for md in campaign.MASS_DAMPINGS:
        nominal = [
            row
            for row in rows
            if float(row["mass_damping"]) == md
            and float(row["timestep_s"]) == campaign.TIMESTEP_S
        ]
        hybrid = [
            row
            for row in nominal
            if row["scenario"] == campaign.ENGINEERING_LOCKIN_SCENARIOS[1]
        ]
        raw = [
            row
            for row in nominal
            if row["scenario"] == campaign.ENGINEERING_LOCKIN_SCENARIOS[0]
        ]
        if len(hybrid) != 12 or len(raw) != 12:
            raise EngineeringEvidenceError(
                "Each damping case requires twelve raw/latest points."
            )
        peak = max(hybrid, key=lambda row: float(row["terminal_amplitude_over_chord"]))
        raw_peak = max(raw, key=lambda row: float(row["terminal_amplitude_over_chord"]))
        target = _experimental_peak(figure7, md)
        amplitude = float(peak["terminal_amplitude_over_chord"])
        amplitude_error = abs(amplitude - target)
        amplitude_tolerance = (
            MAX_AMPLITUDE_RELATIVE_ERROR * target + DIGITIZATION_SIGMA_A_OVER_C
        )
        peak_ustar = float(peak["reduced_velocity"])
        peak_frequency = peak["terminal_dominant_frequency_over_fn"]
        lockin_points = sorted(set(_adjacent_lockin_points(hybrid)))
        half_rows = [
            row
            for row in rows
            if float(row["mass_damping"]) == md
            and float(row["reduced_velocity"])
            == campaign.ENGINEERING_TIMESTEP_CHECK_USTAR
            and float(row["timestep_s"]) == campaign.HALF_TIMESTEP_S
            and row["scenario"] == campaign.ENGINEERING_LOCKIN_SCENARIOS[1]
        ]
        coarse_rows = [
            row
            for row in hybrid
            if float(row["reduced_velocity"])
            == campaign.ENGINEERING_TIMESTEP_CHECK_USTAR
        ]
        if len(half_rows) != 1 or len(coarse_rows) != 1:
            raise EngineeringEvidenceError("Missing paired-timestep engineering case.")
        half = half_rows[0]
        coarse = coarse_rows[0]
        dt_amplitude_difference = abs(
            float(half["terminal_amplitude_over_chord"])
            - float(coarse["terminal_amplitude_over_chord"])
        )
        half_frequency = half["terminal_dominant_frequency_over_fn"]
        coarse_frequency = coarse["terminal_dominant_frequency_over_fn"]
        dt_frequency_difference = (
            math.inf
            if half_frequency is None or coarse_frequency is None
            else abs(float(half_frequency) - float(coarse_frequency))
        )
        checks = {
            "peak_amplitude_within_30_percent_plus_digitization": (
                amplitude_error <= amplitude_tolerance
            ),
            "peak_reduced_velocity_within_10_percent": (
                abs(peak_ustar - experiment_peak_ustar) / experiment_peak_ustar
                <= MAX_PEAK_USTAR_RELATIVE_ERROR
            ),
            "peak_frequency_within_10_percent": (
                peak_frequency is not None
                and abs(float(peak_frequency) - 1.0) <= MAX_FREQUENCY_RELATIVE_ERROR
            ),
            "finite_adjacent_lockin_interval": len(lockin_points) >= 2,
            "paired_timestep_amplitude": (
                dt_amplitude_difference <= MAX_DT_AMPLITUDE_DIFFERENCE_A_OVER_C
            ),
            "paired_timestep_frequency": (
                dt_frequency_difference <= MAX_FREQUENCY_RELATIVE_ERROR
            ),
        }
        results.append(
            {
                "mass_damping": md,
                "experimental_peak_amplitude_over_chord": target,
                "latest_peak_amplitude_over_chord": amplitude,
                "raw_peak_amplitude_over_chord": float(
                    raw_peak["terminal_amplitude_over_chord"]
                ),
                "amplitude_absolute_error_over_chord": amplitude_error,
                "amplitude_tolerance_over_chord": amplitude_tolerance,
                "latest_peak_reduced_velocity": peak_ustar,
                "experimental_peak_reduced_velocity": experiment_peak_ustar,
                "latest_peak_frequency_over_fn": peak_frequency,
                "lockin_reduced_velocities": lockin_points,
                "dt_amplitude_difference_over_chord": dt_amplitude_difference,
                "dt_frequency_difference_over_fn": dt_frequency_difference,
                "checks": checks,
                "pass": all(checks.values()),
            }
        )
    checks = {
        "all_52_cases_stationary": all_stationary,
        "both_mass_damping_cases_pass": all(row["pass"] for row in results),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "pass": all(checks.values()),
        "checks": checks,
        "mass_damping_results": results,
    }


def _nominal(
    rows: Sequence[Mapping[str, Any]], mass_damping: float, scenario: str
) -> list[Mapping[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if float(row["mass_damping"]) == mass_damping
            and row["scenario"] == scenario
            and float(row["timestep_s"]) == campaign.TIMESTEP_S
        ),
        key=lambda row: float(row["reduced_velocity"]),
    )


def _derived_rms_acceleration(row: Mapping[str, Any]) -> float:
    frequency = row["terminal_dominant_frequency_hz"]
    if frequency is None:
        return 0.0
    return (
        (2.0 * math.pi * float(frequency)) ** 2
        * float(row["terminal_amplitude_m"])
        / math.sqrt(2.0)
    )


def _write_plot_data(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    figure6: Sequence[Mapping[str, Any]],
    figure7: Sequence[Mapping[str, Any]],
) -> None:
    columns = (
        "panel",
        "series",
        "mass_damping",
        "reduced_velocity",
        "timestep_s",
        "rms_acceleration_mps2",
        "amplitude_over_chord",
        "dominant_frequency_over_fn",
        "mass_damping_digitization_uncertainty",
        "amplitude_over_chord_digitization_uncertainty",
    )
    records: list[dict[str, object]] = []
    for row in figure6:
        records.append(
            {
                "panel": "acceleration_sweep",
                "series": "Ehrmann experiment (digitized)",
                "reduced_velocity": row["reduced_velocity"],
                "rms_acceleration_mps2": row["rms_acceleration_mps2"],
            }
        )
    for row in figure7:
        records.append(
            {
                "panel": "peak_amplitude",
                "series": "Ehrmann experiment (digitized)",
                "mass_damping": row["mass_damping"],
                "amplitude_over_chord": row["maximum_amplitude_over_chord"],
                "mass_damping_digitization_uncertainty": row[
                    "mass_damping_digitization_uncertainty"
                ],
                "amplitude_over_chord_digitization_uncertainty": row[
                    "amplitude_over_chord_digitization_uncertainty"
                ],
            }
        )
    labels = {
        campaign.ENGINEERING_LOCKIN_SCENARIOS[0]: "Raw QBlade",
        campaign.ENGINEERING_LOCKIN_SCENARIOS[1]: (
            "Latest VorLap-QBlade engineering bridge"
        ),
    }
    for row in rows:
        if float(row["timestep_s"]) != campaign.TIMESTEP_S:
            continue
        records.append(
            {
                "panel": "numerical_sweep",
                "series": labels[str(row["scenario"])],
                "mass_damping": row["mass_damping"],
                "reduced_velocity": row["reduced_velocity"],
                "timestep_s": row["timestep_s"],
                "rms_acceleration_mps2": _derived_rms_acceleration(row),
                "amplitude_over_chord": row["terminal_amplitude_over_chord"],
                "dominant_frequency_over_fn": row[
                    "terminal_dominant_frequency_over_fn"
                ],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in columns})


def _plot(
    rows: Sequence[Mapping[str, Any]],
    *,
    figure6: Sequence[Mapping[str, Any]],
    figure7: Sequence[Mapping[str, Any]],
    assessment: Mapping[str, Any],
    output_pdf: Path,
    output_png: Path,
) -> None:
    """Render the experimental, raw-QBlade, and latest-bridge series explicitly."""

    plt.rcParams.update(
        {
            "font.size": 8.3,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "pdf.fonttype": 42,
        }
    )
    raw_scenario, latest_scenario = campaign.ENGINEERING_LOCKIN_SCENARIOS
    navy = "#174A7E"
    orange = "#D55E00"
    gray = "#4D4D4D"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25), constrained_layout=True)

    ax = axes[0, 0]
    experiment_artist = ax.scatter(
        [row["reduced_velocity"] for row in figure6],
        [row["rms_acceleration_mps2"] for row in figure6],
        marker="v",
        s=21,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        label="Ehrmann experiment (digitized)",
        zorder=4,
    )
    series_artists: dict[str, Any] = {}
    for scenario, color, style, marker in (
        (raw_scenario, navy, "--", "s"),
        (latest_scenario, orange, "-", "o"),
    ):
        selected = _nominal(rows, 0.96, scenario)
        (series_artists[scenario],) = ax.plot(
            [row["reduced_velocity"] for row in selected],
            [_derived_rms_acceleration(row) for row in selected],
            linestyle=style,
            marker=marker,
            color=color,
            linewidth=1.35,
            markersize=3.2,
        )
    ax.set(
        xlabel=r"Reduced velocity, $U^*$",
        ylabel=r"RMS acceleration [m s$^{-2}$]",
        title=r"(a) Representative sweep; numerical $m^*\zeta=0.96$",
    )
    ax.set_xlim(3.0, 6.3)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    for mass_damping, alpha, fillstyle in ((0.96, 1.0, "full"), (1.44, 0.65, "none")):
        for scenario, color, style, marker in (
            (raw_scenario, navy, "--", "s"),
            (latest_scenario, orange, "-", "o"),
        ):
            selected = _nominal(rows, mass_damping, scenario)
            ax.plot(
                [row["reduced_velocity"] for row in selected],
                [row["terminal_amplitude_over_chord"] for row in selected],
                linestyle=style,
                marker=marker,
                color=color,
                alpha=alpha,
                fillstyle=fillstyle,
                linewidth=1.25,
                markersize=3.2,
            )
    ax.set(
        xlabel=r"Reduced velocity, $U^*$",
        ylabel=r"Terminal amplitude, $A/c$",
        title=r"(b) Filled: $m^*\zeta=0.96$; open: $1.44$",
    )
    ax.set_xlim(3.0, 6.3)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.errorbar(
        [row["mass_damping"] for row in figure7],
        [row["maximum_amplitude_over_chord"] for row in figure7],
        xerr=[float(row["mass_damping_digitization_uncertainty"]) for row in figure7],
        yerr=[
            float(row["amplitude_over_chord_digitization_uncertainty"])
            for row in figure7
        ],
        linestyle="none",
        marker="v",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor="black",
        ecolor="black",
        elinewidth=0.7,
        capsize=2,
    )
    for scenario, color, style, marker in (
        (raw_scenario, navy, "--", "s"),
        (latest_scenario, orange, "-", "o"),
    ):
        peaks = [
            max(
                float(row["terminal_amplitude_over_chord"])
                for row in _nominal(rows, mass_damping, scenario)
            )
            for mass_damping in campaign.MASS_DAMPINGS
        ]
        ax.plot(
            campaign.MASS_DAMPINGS,
            peaks,
            linestyle="none",
            marker=marker,
            color=color,
            markersize=4.0,
        )
    ax.set(
        xlabel=r"Mass--damping, $m^*\zeta$",
        ylabel=r"Maximum amplitude, $A_{\max}/c$",
        title="(c) Peak-amplitude comparison",
    )
    ax.set_xlim(0.9, 1.55)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.axhspan(
        1.0 - MAX_FREQUENCY_RELATIVE_ERROR,
        1.0 + MAX_FREQUENCY_RELATIVE_ERROR,
        color="#D9D9D9",
        alpha=0.65,
    )
    ax.axhline(1.0, color=gray, linestyle=":", linewidth=0.9)
    for mass_damping, marker, fillstyle in ((0.96, "o", "full"), (1.44, "s", "none")):
        selected = _nominal(rows, mass_damping, latest_scenario)
        ax.plot(
            [row["reduced_velocity"] for row in selected],
            [
                np.nan
                if row["terminal_dominant_frequency_over_fn"] is None
                else row["terminal_dominant_frequency_over_fn"]
                for row in selected
            ],
            color=orange,
            marker=marker,
            fillstyle=fillstyle,
            linewidth=1.25,
            markersize=3.5,
            label=rf"$m^*\zeta={mass_damping:.2f}$",
        )
    ax.set(
        xlabel=r"Reduced velocity, $U^*$",
        ylabel=r"Dominant frequency, $f/f_n$",
        title="(d) Latest-bridge lock-in frequency",
    )
    ax.set_xlim(3.0, 6.3)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower center", frameon=False, ncol=2)

    result = "PASS" if assessment["pass"] else "FAIL"
    fig.suptitle(
        f"Ehrmann NACA 0018: raw QBlade versus latest VorLap--QBlade bridge ({result})",
        fontsize=10.0,
        y=1.07,
    )
    fig.legend(
        [experiment_artist, series_artists[raw_scenario], series_artists[latest_scenario]],
        [
            "Ehrmann experiment (digitized)",
            "Raw QBlade",
            "Latest VorLap--QBlade engineering bridge",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
    )
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "VorLap Ehrmann engineering validation evidence builder",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build(
    campaign_root: Path,
    output: Path | None = None,
    execution_report: Path | None = None,
) -> Path:
    state = campaign.validate(campaign_root)
    root = Path(state["root"])
    manifest = state["manifest"]
    if manifest["schema"] != campaign.ENGINEERING_LOCKIN_SCHEMA:
        raise EngineeringEvidenceError("Engineering resolved-peak campaign required.")
    analysis_path = campaign.analyze(root)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis["status"] not in {
        "complete_simulation_comparison_experiment_digitization_pending",
        "complete_simulation_comparison_reference_overlay_tracked_separately",
    }:
        raise EngineeringEvidenceError("All 52 campaign cases must be complete.")
    if execution_report is None:
        local_report = root / "local_docker_execution_report.json"
        report_path = (
            local_report
            if local_report.is_file()
            else root / "flight_scheduler_report.json"
        )
    else:
        report_path = Path(execution_report).expanduser().resolve()
    contract = _validate_execution_report(root, report_path, manifest)
    assessment = assess_rows(
        analysis["cases"], figure6=_read_csv(FIGURE6), figure7=_read_csv(FIGURE7)
    )
    report = {
        "schema": SCHEMA,
        "status": assessment["status"],
        "pass": assessment["pass"],
        "comparison": {
            "experiment": "Ehrmann et al. NACA0018",
            "baseline": "Raw QBlade",
            "latest_model": "VorLap-QBlade engineering spectral negative damping",
        },
        "campaign_fingerprint": manifest["fingerprint"],
        "calibration_identity": manifest["engineering_lockin_identity"],
        "execution_contract": contract,
        "source_artifacts": {
            "campaign_manifest": {
                "path": "campaign_manifest.json",
                "sha256": _sha256(root / "campaign_manifest.json"),
            },
            "campaign_analysis": {
                "path": "analysis/ehrmann2014_analysis.json",
                "sha256": _sha256(analysis_path),
            },
            "campaign_case_metrics": {
                "path": "analysis/ehrmann2014_case_metrics.csv",
                "sha256": _sha256(root / "analysis/ehrmann2014_case_metrics.csv"),
            },
            "execution_report": {
                "path": report_path.relative_to(root).as_posix(),
                "sha256": _sha256(report_path),
            },
            "experiment_figure6_digitization": {
                "path": FIGURE6.relative_to(campaign.HERE).as_posix(),
                "sha256": _sha256(FIGURE6),
            },
            "experiment_figure7_digitization": {
                "path": FIGURE7.relative_to(campaign.HERE).as_posix(),
                "sha256": _sha256(FIGURE7),
            },
            "experiment_digitization_provenance": {
                "path": "reference_data/ehrmann2014_digitization_provenance.json",
                "sha256": _sha256(
                    campaign.HERE
                    / "reference_data/ehrmann2014_digitization_provenance.json"
                ),
            },
        },
        "acceptance": assessment,
        "claim_boundary": {
            "qualifies_only_ehrmann_naca0018_engineering_screen": True,
            "independent_cross_airfoil_validation": False,
            "hvawt_advancement_allowed": assessment["pass"],
            "nalu_required": False,
        },
    }
    destination = (
        root / "validation_evidence/ehrmann2014_engineering_validation.json"
        if output is None
        else Path(output).expanduser().resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure_pdf = destination.parent / "ehrmann2014_engineering_validation.pdf"
    figure_png = destination.parent / "ehrmann2014_engineering_validation.png"
    plot_data = destination.parent / "ehrmann2014_engineering_validation_plot_data.csv"
    compact_case_metrics = (
        destination.parent / "ehrmann2014_engineering_case_metrics.csv"
    )
    figure6 = _read_csv(FIGURE6)
    figure7 = _read_csv(FIGURE7)
    _write_plot_data(plot_data, rows=analysis["cases"], figure6=figure6, figure7=figure7)
    _plot(
        analysis["cases"],
        figure6=figure6,
        figure7=figure7,
        assessment=assessment,
        output_pdf=figure_pdf,
        output_png=figure_png,
    )
    shutil.copyfile(root / "analysis/ehrmann2014_case_metrics.csv", compact_case_metrics)

    def artifact_path(path: Path) -> str:
        return path.relative_to(destination.parent).as_posix()

    report["artifacts"] = {
        "comparison_pdf": {
            "path": artifact_path(figure_pdf),
            "sha256": _sha256(figure_pdf),
        },
        "comparison_png": {
            "path": artifact_path(figure_png),
            "sha256": _sha256(figure_png),
        },
        "plot_data_csv": {
            "path": artifact_path(plot_data),
            "sha256": _sha256(plot_data),
        },
        "case_metrics_csv": {
            "path": artifact_path(compact_case_metrics),
            "sha256": _sha256(compact_case_metrics),
        },
    }
    atomic_write_json(destination, report)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--execution-report", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        path = build(args.campaign_root, args.output, args.execution_report)
    except (
        campaign.CampaignError,
        EngineeringEvidenceError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {"path": str(path), "status": report["status"], "pass": report["pass"]},
            indent=2,
        )
    )
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
