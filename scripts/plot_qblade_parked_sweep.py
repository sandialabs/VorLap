#!/usr/bin/env python3
"""Post-process VorLap/QBlade parked-load batch results."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_cache_root = Path(tempfile.gettempdir()) / "vorlap_matplotlib_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd


TIME_COLUMN = "Time~[s]"


@dataclass(frozen=True)
class LoadChannel:
    mode: str
    blade: int
    axis: str
    frame: str
    quantity: str
    level: str
    unit: str
    column: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Run root containing cases.json and results/<case> QBlade exports.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for plots and CSV output.")
    parser.add_argument("--mode", choices=("blade-pos", "joint-z"), default="blade-pos")
    parser.add_argument("--level", default="0.230", help="Blade station or joint z-height to summarize.")
    parser.add_argument("--axis", choices=("X", "Y", "Z"), default="Y")
    parser.add_argument("--frame", choices=("local", "global"), default="local")
    parser.add_argument("--quantity", choices=("force", "moment"), default="force")
    parser.add_argument("--time-min", type=float, default=None)
    parser.add_argument("--time-max", type=float, default=None)
    parser.add_argument(
        "--plot-case-count",
        type=int,
        default=3,
        help="Number of representative cases for which time-history plots are saved.",
    )
    parser.add_argument("--list-levels", action="store_true", help="List available levels from the first result file.")
    return parser.parse_args()


def load_manifest(run_root: Path) -> dict:
    manifest_path = run_root / "cases.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing batch manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def result_file_for_case(case: dict) -> Path | None:
    expected = Path(case["result_dir"]) / f"{case['result_name']}.txt"
    if expected.is_file():
        return expected
    result_dir = Path(case["result_dir"])
    matches = sorted(result_dir.glob("*.txt"))
    return matches[0] if matches else None


def load_qblade_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", skiprows=2)


def apply_time_window(df: pd.DataFrame, time_min: float | None, time_max: float | None) -> pd.DataFrame:
    if TIME_COLUMN not in df.columns:
        raise KeyError(f"Missing required time column: {TIME_COLUMN}")
    mask = pd.Series(True, index=df.index)
    if time_min is not None:
        mask &= df[TIME_COLUMN] >= time_min
    if time_max is not None:
        mask &= df[TIME_COLUMN] <= time_max
    return df.loc[mask].copy()


def parse_channels(columns: Iterable[str]) -> list[LoadChannel]:
    channels: list[LoadChannel] = []
    blade_pos_pattern = re.compile(
        r"^(?P<axis>[XYZ])_(?P<frame>[lg])~(?P<quantity>For\.|Mom\.)~"
        r"BLD_(?P<blade>[123])~pos~(?P<level>\d+\.\d+)~\[(?P<unit>[^\]]+)\]$"
    )
    joint_pattern = re.compile(
        r"^(?P<axis>[XYZ])_(?P<frame>[lg])~(?P<quantity>For\.|Mom\.)~"
        r"STR_\d+_(?P<blade>[123])~-~BLD_(?P=blade)~z=(?P<level>\d+\.\d+)m~"
        r"\[(?P<unit>[^\]]+)\]$"
    )
    for column in columns:
        match = blade_pos_pattern.match(column)
        if match:
            channels.append(
                LoadChannel(
                    mode="blade-pos",
                    blade=int(match.group("blade")),
                    axis=match.group("axis"),
                    frame={"l": "local", "g": "global"}[match.group("frame")],
                    quantity={"For.": "force", "Mom.": "moment"}[match.group("quantity")],
                    level=match.group("level"),
                    unit=match.group("unit"),
                    column=column,
                )
            )
            continue
        match = joint_pattern.match(column)
        if match:
            channels.append(
                LoadChannel(
                    mode="joint-z",
                    blade=int(match.group("blade")),
                    axis=match.group("axis"),
                    frame={"l": "local", "g": "global"}[match.group("frame")],
                    quantity={"For.": "force", "Mom.": "moment"}[match.group("quantity")],
                    level=match.group("level"),
                    unit=match.group("unit"),
                    column=column,
                )
            )
    return channels


def find_blade_channels(
    channels: Iterable[LoadChannel],
    *,
    mode: str,
    level: str,
    axis: str,
    frame: str,
    quantity: str,
) -> list[LoadChannel]:
    selected = [
        channel
        for channel in channels
        if channel.mode == mode
        and channel.level == level
        and channel.axis == axis
        and channel.frame == frame
        and channel.quantity == quantity
    ]
    selected.sort(key=lambda channel: channel.blade)
    return selected


def available_levels(channels: Iterable[LoadChannel], mode: str) -> list[str]:
    return sorted({channel.level for channel in channels if channel.mode == mode}, key=float)


def summarize_case(case: dict, df: pd.DataFrame, channels: list[LoadChannel]) -> dict:
    row = {
        "case_index": case["index"],
        "case_label": case["label"],
        "wind_speed_mps": case["wind_speed"],
        "sample_count": len(df),
        "time_start_s": float(df[TIME_COLUMN].iloc[0]),
        "time_end_s": float(df[TIME_COLUMN].iloc[-1]),
    }
    rms_values = []
    for channel in channels:
        values = pd.to_numeric(df[channel.column], errors="coerce")
        demeaned = values - values.mean()
        rms = float((demeaned.pow(2).mean()) ** 0.5)
        row[f"blade_{channel.blade}_mean_{channel.unit}"] = float(values.mean())
        row[f"blade_{channel.blade}_rms_{channel.unit}"] = rms
        row[f"blade_{channel.blade}_max_abs_{channel.unit}"] = float(values.abs().max())
        row[f"blade_{channel.blade}_peak_to_peak_{channel.unit}"] = float(values.max() - values.min())
        rms_values.append(rms)
    row["mean_blade_rms"] = float(sum(rms_values) / len(rms_values))
    row["max_blade_rms"] = float(max(rms_values))
    return row


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 200,
        }
    )


def plot_sweep(summary: pd.DataFrame, output_dir: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.plot(summary["wind_speed_mps"], summary["mean_blade_rms"], marker="o", lw=1.6, label="Mean blade RMS")
    ax.plot(summary["wind_speed_mps"], summary["max_blade_rms"], marker="s", lw=1.2, label="Max blade RMS")
    ax.set_xlabel("Mean inflow speed [m/s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"parked_load_sweep_rms.{ext}", bbox_inches="tight")
    plt.close(fig)


def representative_cases(cases: list[dict], count: int) -> list[dict]:
    if count <= 0 or not cases:
        return []
    if count >= len(cases):
        return cases
    if count == 1:
        return [cases[len(cases) // 2]]
    indices = sorted({round(i * (len(cases) - 1) / (count - 1)) for i in range(count)})
    return [cases[i] for i in indices]


def plot_time_history(case: dict, df: pd.DataFrame, channels: list[LoadChannel], output_dir: Path, ylabel: str) -> None:
    colors = ("#1b9e77", "#d95f02", "#7570b3")
    for demean in (False, True):
        fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
        for color, channel in zip(colors, channels):
            values = pd.to_numeric(df[channel.column], errors="coerce")
            if demean:
                values = values - values.mean()
            ax.plot(df[TIME_COLUMN], values, color=color, lw=1.0, label=f"Blade {channel.blade}")
        ax.axhline(0.0, color="0.65", lw=0.8, linestyle="--")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel if not demean else f"Alternating {ylabel}")
        ax.set_title(f"{case['label']}  U={case['wind_speed']:g} m/s")
        ax.grid(True, color="0.88", linewidth=0.6)
        ax.legend(frameon=False, ncol=3)
        suffix = "demeaned" if demean else "raw"
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"{case['label']}_blade_load_{suffix}.{ext}", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or run_root / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(run_root)

    loaded: list[tuple[dict, Path, pd.DataFrame, list[LoadChannel]]] = []
    missing: list[str] = []
    for case in sorted(manifest["cases"], key=lambda item: item["index"]):
        result_file = result_file_for_case(case)
        if result_file is None:
            missing.append(case["label"])
            continue
        df = apply_time_window(load_qblade_results(result_file), args.time_min, args.time_max)
        if df.empty:
            raise SystemExit(f"Requested time window left no rows for {result_file}")
        selected = find_blade_channels(
            parse_channels(df.columns),
            mode=args.mode,
            level=args.level,
            axis=args.axis,
            frame=args.frame,
            quantity=args.quantity,
        )
        if len(selected) != 3:
            levels = ", ".join(available_levels(parse_channels(df.columns), args.mode))
            if args.list_levels:
                print(f"{case['label']}: {levels}")
                return
            raise SystemExit(
                "Could not find three matching blade channels for "
                f"mode={args.mode}, level={args.level}, axis={args.axis}, "
                f"frame={args.frame}, quantity={args.quantity}. Available levels: {levels}"
            )
        loaded.append((case, result_file, df, selected))

    if args.list_levels:
        if not loaded:
            raise SystemExit("No result files were found.")
        levels = available_levels(parse_channels(loaded[0][2].columns), args.mode)
        print("\n".join(levels))
        return

    if missing:
        print("Missing result files for: " + ", ".join(missing))
    if not loaded:
        raise SystemExit("No QBlade result files were found to post-process.")

    configure_matplotlib()
    rows = [summarize_case(case, df, channels) for case, _result_file, df, channels in loaded]
    summary = pd.DataFrame(rows).sort_values("wind_speed_mps")
    summary_path = output_dir / "parked_load_summary.csv"
    summary.to_csv(summary_path, index=False)

    unit = loaded[0][3][0].unit
    quantity_label = "force" if args.quantity == "force" else "moment"
    ylabel = f"{args.axis} {args.frame} {quantity_label} RMS [{unit}]"
    plot_sweep(summary, output_dir, ylabel)
    for case, _result_file, df, channels in representative_cases(
        [item[0] for item in loaded], args.plot_case_count
    ):
        match = next(item for item in loaded if item[0]["index"] == case["index"])
        plot_time_history(match[0], match[2], match[3], output_dir, f"{args.axis} {args.frame} {quantity_label} [{unit}]")

    print(f"Wrote {summary_path}")
    print(f"Wrote plots under {output_dir}")


if __name__ == "__main__":
    main()
