#!/usr/bin/env python3
"""Plot Nalu force histories separately for each angle of attack.

The input layout is the same one accepted by ``process_naluout_to_h5.py``:
either ``--input-dir/*.dat`` or ``--input-dir/data_files/*.dat``.  Files whose
names start with ``pp_`` are ignored.

Each output figure contains:

* lift coefficient ``C_L`` versus time;
* drag coefficient ``C_D`` versus time;
* moment coefficient ``C_M`` versus time; and
* one-sided Hann-windowed FFT power spectral density curves for ``C_L``, ``C_D``,
  and ``C_M``.

The force normalization matches ``process_naluout_to_h5.py``:
``q = 0.5 * rho * Vinf**2 * chord * span`` and ``C_M = Mtz / q``.

Example from the VorLap repo root::

    .venv/bin/python scripts/plot_naluout_per_aoa.py \
        --input-dir scripts/NACA0018 \
        --output-dir scripts/NACA0018/figs_per_aoa \
        --airfoil-name NACA0018 \
        --re 5e5 \
        --initial-timestep-skip 1000

For a quick smoke test, restrict the run to selected angles::

    .venv/bin/python scripts/plot_naluout_per_aoa.py \
        --input-dir scripts/NACA0018 \
        --output-dir scripts/NACA0018/figs_per_aoa_smoke \
        --aoa 0 4 8 \
        --re 5e5 \
        --initial-timestep-skip 1000
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "scripts" / "NACA0018"


def positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected a positive finite value, got {text!r}.")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {text!r}.")
    return value


def parse_aoa_from_filename(path: Path) -> float:
    match = re.search(r"_(-?\d+(?:\.\d+)?)\.dat$", path.name)
    if match is None:
        raise ValueError(f"Could not infer AOA from file name: {path.name}")
    return float(match.group(1))


def aoa_file_label(aoa: float) -> str:
    """Return a filename-safe signed AOA label without +/- collisions."""
    if np.isclose(aoa, 0.0):
        return "p000"
    sign = "m" if aoa < 0.0 else "p"
    magnitude = abs(float(aoa))
    if np.isclose(magnitude, round(magnitude)):
        text = f"{int(round(magnitude)):03d}"
    else:
        text = f"{magnitude:.6g}".replace(".", "p")
    return f"{sign}{text}"


def discover_force_files(input_dir: Path) -> list[tuple[float, Path]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    direct = sorted(path for path in input_dir.iterdir() if path.suffix.lower() == ".dat")
    if direct:
        candidates = direct
    else:
        data_dir = input_dir / "data_files"
        candidates = sorted(path for path in data_dir.iterdir() if path.suffix.lower() == ".dat") if data_dir.is_dir() else []

    candidates = [path for path in candidates if not path.name.startswith("pp_")]
    if not candidates:
        raise FileNotFoundError(f"No primary .dat force histories found in {input_dir} or data_files/.")

    files: list[tuple[float, Path]] = []
    seen: dict[float, Path] = {}
    for path in candidates:
        aoa = parse_aoa_from_filename(path)
        key = round(aoa, 10)
        if key in seen:
            raise ValueError(f"Duplicate AOA {aoa:g}: {seen[key]} and {path}")
        seen[key] = path
        files.append((aoa, path))
    return sorted(files, key=lambda item: item[0])


def load_force_history(path: Path, *, min_samples: int, initial_timestep_skip: int) -> np.ndarray:
    """Load, sort, de-duplicate, startup-trim, and uniformize one history."""
    data = np.loadtxt(path, skiprows=1)
    if data.size == 0:
        raise ValueError(f"{path} contains a header but no data rows.")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < min_samples:
        raise ValueError(f"{path} has {data.shape[0]} samples; require at least {min_samples}.")
    if data.shape[1] <= 8:
        raise ValueError(f"{path} has {data.shape[1]} columns; require at least 9.")

    order = np.argsort(data[:, 0])
    data = data[order]
    time, unique_idx = np.unique(data[:, 0], return_index=True)
    data = data[unique_idx]
    if initial_timestep_skip >= time.size - 1:
        raise ValueError(
            f"{path} initial timestep skip {initial_timestep_skip} leaves fewer than two samples."
        )
    if initial_timestep_skip:
        data = data[initial_timestep_skip:]
        time = time[initial_timestep_skip:]

    dt = np.diff(time)
    if dt.size == 0 or not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError(f"{path} does not have increasing finite time values.")

    dt_med = float(np.median(dt))
    n_uniform = int(round((time[-1] - time[0]) / dt_med)) + 1
    uniform_time = time[0] + np.arange(n_uniform, dtype=float) * dt_med
    uniform_time = uniform_time[uniform_time <= time[-1] + 0.25 * dt_med]
    needs_resample = (
        uniform_time.size != time.size
        or not np.allclose(dt, dt_med, rtol=1.0e-3, atol=max(1.0e-10, 1.0e-6 * dt_med))
    )
    if needs_resample:
        uniform = np.empty((uniform_time.size, data.shape[1]), dtype=float)
        uniform[:, 0] = uniform_time
        for column in range(1, data.shape[1]):
            uniform[:, column] = np.interp(uniform_time, time, data[:, column])
        data = uniform
    else:
        data = data.copy()
        data[:, 0] = uniform_time

    if data.shape[0] < min_samples:
        raise ValueError(f"{path} has {data.shape[0]} uniform samples; require {min_samples}.")
    return data


def coefficient_history(
    data: np.ndarray,
    *,
    rho: float,
    vinf: float,
    chord: float,
    span: float,
) -> dict[str, np.ndarray]:
    dynamic_force = 0.5 * rho * vinf**2 * chord * span
    return {
        "time": data[:, 0],
        "CL": (data[:, 2] + data[:, 5]) / dynamic_force,
        "CD": (data[:, 1] + data[:, 4]) / dynamic_force,
        "CM": data[:, 8] / dynamic_force,
    }


def one_sided_psd(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=float)
    if values.size < 4:
        raise ValueError("Need at least four samples for a PSD.")
    window = np.hanning(values.size)
    centered = values - np.mean(values)
    spectrum = np.fft.rfft(centered * window)
    frequencies = np.fft.rfftfreq(values.size, dt)
    psd = (dt / np.sum(window**2)) * np.abs(spectrum) ** 2
    if values.size % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    return frequencies, psd


def select_plot_data(history: dict[str, np.ndarray], plot_samples: int) -> dict[str, np.ndarray]:
    if plot_samples == 0 or history["time"].size <= plot_samples:
        return history
    return {name: values[-plot_samples:] for name, values in history.items()}


def plot_one_aoa(
    output_dir: Path,
    *,
    airfoil_name: str,
    aoa: float,
    source_path: Path,
    history: dict[str, np.ndarray],
    reynolds: float,
    vinf: float,
    max_frequency: float | None,
    plot_format: str,
) -> Path:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config = Path(tempfile.gettempdir()) / "vorlap_matplotlib"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config)
    if "XDG_CACHE_HOME" not in os.environ:
        xdg_cache = Path(tempfile.gettempdir()) / "vorlap_xdg_cache"
        xdg_cache.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    time = history["time"]
    dt = float(np.median(np.diff(time)))
    frequencies, psd_cl = one_sided_psd(history["CL"], dt)
    _, psd_cd = one_sided_psd(history["CD"], dt)
    _, psd_cm = one_sided_psd(history["CM"], dt)
    positive = frequencies > 0.0
    if max_frequency is not None:
        positive &= frequencies <= max_frequency
    if not np.any(positive):
        raise ValueError(f"No PSD frequencies remain for max frequency {max_frequency}.")

    fig, axes = plt.subplots(4, 1, figsize=(10.5, 12.0), sharex=False)
    fig.suptitle(
        f"{airfoil_name} | AOA {aoa:g} deg | Re={reynolds:.5g} | Vinf={vinf:.5g} m/s",
        fontsize=14,
    )
    axes[0].plot(time, history["CL"], color="tab:blue", linewidth=0.8)
    axes[0].set_ylabel("$C_L$")
    axes[0].set_title("Lift coefficient")
    axes[1].plot(time, history["CD"], color="tab:orange", linewidth=0.8)
    axes[1].set_ylabel("$C_D$")
    axes[1].set_title("Drag coefficient")
    axes[2].plot(time, history["CM"], color="tab:green", linewidth=0.8)
    axes[2].set_ylabel("$C_M$")
    axes[2].set_title("Moment coefficient")
    axes[2].set_xlabel("Time (s)")
    axes[3].loglog(frequencies[positive], psd_cl[positive], label="$C_L$", linewidth=1.0)
    axes[3].loglog(frequencies[positive], psd_cd[positive], label="$C_D$", linewidth=1.0)
    axes[3].loglog(frequencies[positive], psd_cm[positive], label="$C_M$", linewidth=1.0)
    axes[3].set_xlabel("Frequency (Hz)")
    axes[3].set_ylabel("PSD (coefficient$^2$/Hz)")
    axes[3].set_title("One-sided power spectral density")
    axes[3].legend()
    for axis in axes[:3]:
        axis.grid(True, alpha=0.25)
    axes[3].grid(True, which="both", alpha=0.25)
    fig.text(0.01, 0.01, f"Source: {source_path.name}", ha="left", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{airfoil_name}_AOA_{aoa_file_label(aoa)}.{plot_format}"
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write one lift/drag/moment history plus PSD figure per Nalu AOA."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <input-dir>/figs_per_aoa.")
    parser.add_argument("--airfoil-name", default=None)
    parser.add_argument("--aoa", type=float, nargs="+", default=None, metavar="DEG", help="Subset of AOAs to plot.")
    parser.add_argument("--re", type=positive_float, default=5.0e5, help="Reynolds number used to derive Vinf.")
    parser.add_argument("--vinf", type=positive_float, default=None, help="Override Vinf [m/s] instead of deriving it from Re.")
    parser.add_argument("--chord", type=positive_float, default=1.0)
    parser.add_argument("--span", type=positive_float, default=4.0)
    parser.add_argument("--fluid-density", type=positive_float, default=1.2)
    parser.add_argument("--fluid-viscosity", type=positive_float, default=9.0e-6)
    parser.add_argument("--initial-timestep-skip", type=nonnegative_int, default=1000)
    parser.add_argument("--min-samples", type=nonnegative_int, default=1000)
    parser.add_argument("--plot-samples", type=nonnegative_int, default=8000, help="Trailing samples per figure; 0 uses all.")
    parser.add_argument("--max-frequency", type=positive_float, default=None, help="Optional PSD x-axis upper limit [Hz].")
    parser.add_argument("--plot-format", choices=("pdf", "png"), default="pdf")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = discover_force_files(args.input_dir)
    if args.airfoil_name is None:
        airfoil_name = re.sub(r"_[-+]?\d+(?:\.\d+)?$", "", files[0][1].stem)
    else:
        airfoil_name = args.airfoil_name

    requested = None if args.aoa is None else {round(value, 10) for value in args.aoa}
    selected = [(aoa, path) for aoa, path in files if requested is None or round(aoa, 10) in requested]
    if requested is not None:
        found = {round(aoa, 10) for aoa, _ in selected}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"Requested AOA files were not found: {missing}")
    if not selected:
        raise FileNotFoundError("No AOA files selected.")

    vinf = args.vinf or args.re * args.fluid_viscosity / (args.fluid_density * args.chord)
    reynolds = args.fluid_density * vinf * args.chord / args.fluid_viscosity
    output_dir = args.output_dir or args.input_dir / "figs_per_aoa"
    outputs: list[Path] = []
    for aoa, source_path in selected:
        data = load_force_history(
            source_path,
            min_samples=args.min_samples,
            initial_timestep_skip=args.initial_timestep_skip,
        )
        history = coefficient_history(
            data,
            rho=args.fluid_density,
            vinf=vinf,
            chord=args.chord,
            span=args.span,
        )
        history = select_plot_data(history, args.plot_samples)
        output = plot_one_aoa(
            output_dir,
            airfoil_name=airfoil_name,
            aoa=aoa,
            source_path=source_path,
            history=history,
            reynolds=reynolds,
            vinf=vinf,
            max_frequency=args.max_frequency,
            plot_format=args.plot_format,
        )
        outputs.append(output)
        print(f"{aoa:g} deg -> {output}")
    print(f"Generated {len(outputs)} figures in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
