"""Plot the completed eight-case coupled Scruton-transfer engineering screen."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


_EXPECTED_SCRUTON = (2.94, 7.02)
_TARGET_PEAK_U_STAR = {2.94: 6.8, 7.02: 6.2}
_COLORS = {"dt": "#0072B2", "dt_half": "#D55E00", "target": "#009E73"}


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _load_inputs(
    assessment_path: Path, response_csv_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assessment = json.loads(Path(assessment_path).read_text(encoding="utf-8"))
    if (
        assessment.get("schema")
        != "vlastos2026-belloli-sc-transfer-live-screen-v1"
        or assessment.get("evaluated") is not True
        or assessment.get("completed_case_count") != 8
    ):
        raise ValueError("The coupled transfer assessment must contain eight evaluated cases.")
    per_scruton = assessment.get("by_scruton_number")
    if not isinstance(per_scruton, list) or [
        _finite(row.get("scruton_number"), "assessment Scruton number")
        for row in per_scruton
    ] != list(_EXPECTED_SCRUTON):
        raise ValueError("The assessment Scruton-number rows are incomplete or reordered.")

    with Path(response_csv_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8 or len({row.get("case_id") for row in rows}) != 8:
        raise ValueError("The response CSV must contain exactly eight unique cases.")
    observed = {
        (
            _finite(row.get("scruton_number"), "response Scruton number"),
            _finite(row.get("reduced_velocity"), "response reduced velocity"),
            row.get("timestep_level"),
        )
        for row in rows
    }
    expected = {
        (2.94, 6.25, "dt"),
        (2.94, 6.25, "dt_half"),
        (2.94, 6.8, "dt"),
        (2.94, 6.8, "dt_half"),
        (7.02, 6.2, "dt"),
        (7.02, 6.2, "dt_half"),
        (7.02, 6.25, "dt"),
        (7.02, 6.25, "dt_half"),
    }
    if observed != expected:
        raise ValueError("The response CSV does not match the frozen eight-case screen.")
    for row in rows:
        amplitude = _finite(
            row.get("tip_peak_amplitude_over_diameter"), "response amplitude"
        )
        frequency = _finite(
            row.get("terminal_dominant_frequency_over_fn"), "frequency ratio"
        )
        if amplitude < 0.0 or frequency <= 0.0:
            raise ValueError("Response amplitudes and frequency ratios must be physical.")
    return assessment, rows


def draw_coupled_transfer_screen(
    assessment_path: Path, response_csv_path: Path, output_path: Path
) -> Path:
    """Render the paired-timestep QBlade amplitudes against the declared target peaks."""

    assessment, rows = _load_inputs(assessment_path, response_csv_path)
    import matplotlib.pyplot as plt

    by_scruton: Mapping[float, Mapping[str, Any]] = {
        float(row["scruton_number"]): row for row in assessment["by_scruton_number"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15))
    for axis, sc in zip(axes, _EXPECTED_SCRUTON):
        summary = by_scruton[sc]
        target = _finite(summary["target_peak_amplitude_over_diameter"], "target")
        tolerance = _finite(summary["peak_amplitude_tolerance"], "tolerance")
        axis.axhspan(
            max(0.0, target - tolerance),
            target + tolerance,
            color=_COLORS["target"],
            alpha=0.11,
            linewidth=0.0,
            label=r"Acceptance band about Lupi experiment ($30\%+0.01D$)",
        )
        for level, label, marker in (
            ("dt", "Response-aware VorLap–QBlade, nominal $\\Delta t$", "o"),
            ("dt_half", "Response-aware VorLap–QBlade, half $\\Delta t$", "s"),
        ):
            selected = sorted(
                (
                    (
                        _finite(row["reduced_velocity"], "reduced velocity"),
                        _finite(
                            row["tip_peak_amplitude_over_diameter"], "amplitude"
                        ),
                    )
                    for row in rows
                    if _finite(row["scruton_number"], "Scruton number") == sc
                    and row["timestep_level"] == level
                ),
                key=lambda item: item[0],
            )
            axis.plot(
                [item[0] for item in selected],
                [item[1] for item in selected],
                color=_COLORS[level],
                marker=marker,
                markersize=5.5,
                linewidth=1.5,
                label=label,
            )
        axis.errorbar(
            _TARGET_PEAK_U_STAR[sc],
            target,
            yerr=0.01,
            color=_COLORS["target"],
            marker="D",
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=2.5,
            linestyle="none",
            label="Lupi et al. experimental peak (digitized)",
            zorder=6,
        )
        selected_amplitude = _finite(
            summary["selected_peak_amplitude_over_diameter"], "selected amplitude"
        )
        selected_frequency = _finite(
            summary["selected_peak_frequency_over_fn"], "selected frequency"
        )
        error_percent = 100.0 * abs(selected_amplitude - target) / target
        axis.text(
            0.03,
            0.96,
            f"VorLap–QBlade selected: $A/D={selected_amplitude:.3f}$\n"
            rf"error $={error_percent:.1f}\%$, $f/f_n={selected_frequency:.3f}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 1.5,
            },
        )
        axis.set_title(f"Sc = {sc:.2f}", fontsize=9.0, fontweight="bold")
        axis.set_xlabel(r"Reduced velocity, $U^*$", fontsize=8.5)
        axis.set_ylabel(r"Peak response, $A/D$", fontsize=8.5)
        axis.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=8.0)
        x_values = sorted(
            {
                _finite(row["reduced_velocity"], "reduced velocity")
                for row in rows
                if _finite(row["scruton_number"], "Scruton number") == sc
            }
        )
        margin = max(0.04, 0.18 * (max(x_values) - min(x_values)))
        axis.set_xlim(min(x_values) - margin, max(x_values) + margin)
        axis.set_ylim(bottom=0.0)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=7.2,
    )
    fig.text(
        0.5,
        0.012,
        "Lines are response-aware VorLap–QBlade results; diamonds are digitized Lupi experiments. Targets informed model selection.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.73, wspace=0.22)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        destination,
        dpi=300,
        metadata={
            "Creator": "VorLap plot_engineering_sc_transfer_results.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--response-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(draw_coupled_transfer_screen(args.assessment, args.response_csv, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
