#!/usr/bin/env python3
"""Convert Nalu force-history ``.dat`` files into VorLap airfoil FFT HDF5.

Setup once, from the VorLap repo root:

  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e ".[dev,docs,gui]"

Run the local NACA0018 conversion:

.venv/bin/python scripts/process_naluout_to_h5.py  --input-dir scripts/NACA0018 --airfoil-name NACA0018 --thickness 0.18 --re 5e5 --periodic-aoa --st-offset 0.0 --low-freq-skip 30 --initial-timestep-skip 1000


Other typical runs:

  # Flat folder with ``data_files/*.dat``; explicit Re gives Vinf from Re.
  .venv/bin/python scripts/process_naluout_to_h5.py \\
    --input-dir scripts/NACA0018 --airfoil-name NACA0018 --thickness 0.18 \\
    --re 5e5 --st-offset 0.07 --initial-timestep-skip 1000

  # Root folder with Reynolds subdirectories named like RE5_00E5.
  .venv/bin/python scripts/process_naluout_to_h5.py \\
    --input-dir data/airfoils/NALURuns/NACA0018 \\
    --airfoil-name NACA0018 --thickness 0.18 --symmetric-append

  # FFA-W3 source layout.
  .venv/bin/python scripts/process_naluout_to_h5.py \\
    --input-dir data/airfoils/2024_Ganesh_VIV_Paper_Data/ffa_data_files_ftt_160/ffa_w3_211 \\
    --n-freq 200 --output data/airfoils/ffa_w3_211.h5

  # Optional Swalwell et al. NACA0021 experimental overlays on the NACA0018
  # CF Strouhal/amplitude summary plots.
  .venv/bin/python scripts/process_naluout_to_h5.py \\
    --input-dir scripts/NACA0018 --airfoil-name NACA0018 --thickness 0.18 \\
    --re 5e5 --initial-timestep-skip 1000 \\
    --experimental-data-dir scripts/NACA0018/Swalwell_NACA0021_Exp_Data \\
    --experimental-psd-scale 1e-3 --periodic-aoa

  # Parked-DLC runtime source using the accepted Reynolds-independent model.
  .venv/bin/python scripts/process_naluout_to_h5.py \\
    --input-dir scripts/NACA0018 --airfoil-name NACA0018 --thickness 0.18 \\
    --re 5e5 --initial-timestep-skip 1000 --periodic-aoa --no-plots \\
    --replicate-re-grid "1e4 5e5 2e6" \\
    --replicate-re-justification \\
      "Conservative Reynolds-independent NACA0018 spectral model accepted for the parked DLC paper study; the exact Re=5e5 spectra are replicated down to Re=1e4 after the U03 coupled recovery reached Re=1.921e4 below the prior Re=2e4 envelope." \\
    --output examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/VorLapAirfoils/NACA0018.h5

  # For an approved envelope-only revision, use
  # scripts/extend_replicated_reynolds_grid.py instead. It verifies the parent
  # hash and preserves every non-Re dataset bit-for-bit.

Input layout:
  ``--input-dir`` may contain ``*.dat`` directly, ``data_files/*.dat``, or
  ``RE*`` child directories that contain either layout. File names must end in
  the angle of attack, e.g. ``NACA0018_84.dat`` or ``ffa_w3_211_144.dat``.
  For flat folders without RE children, pass ``--re``; otherwise the fallback
  Reynolds number is computed from ``--vinf-fallback``.

Main functions:
  ``discover_re_directories`` finds RE folders or a flat input.
  ``load_force_history`` drops startup samples, validates, and resamples time data.
  ``compute_fft`` returns DC-first spectra sorted by Hann-windowed power.
  ``build_airfoil_fft`` writes the HDF5 and provenance metadata.
  ``write_summary_plots`` writes the compact CF Strouhal/amplitude PDFs.

HDF5 output:
  ``Airfoilname``, ``Re``, ``Thickness``, and ``AOA`` plus 3-D datasets
  ``CL/CD/CM/CF`` x ``ST/Amp/Pha`` stored as ``[Re, AOA, frequency]``.
  Spectral index 0 is the signed mean coefficient and has Strouhal number 0.
  ``--periodic-aoa`` closes a full-circle source by copying the ``-180``
  spectra to a generated ``+180`` endpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Sequence

import h5py
import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal.windows import hann

REPO_ROOT = Path(__file__).resolve().parents[1]
NACA_LOCAL_INPUT = REPO_ROOT / "scripts" / "NACA0018"
FFA_INPUT = (
    REPO_ROOT
    / "data"
    / "airfoils"
    / "2024_Ganesh_VIV_Paper_Data"
    / "ffa_data_files_ftt_160"
    / "ffa_w3_211"
)
DEFAULT_INPUT_DIR = NACA_LOCAL_INPUT if NACA_LOCAL_INPUT.is_dir() else FFA_INPUT


@dataclass(frozen=True)
class ProcessingConfig:
    input_dir: Path
    source_path_root: Path | None
    output: Path | None
    plots_dir: Path | None
    experimental_data_dir: Path | None
    experimental_label: str | None
    experimental_psd_scale: float
    replicate_re_grid: tuple[float, ...] | None
    replicate_re_justification: str | None
    airfoil_name: str | None
    thickness_ratio: float | None
    re_override: float | None
    chord_m: float
    span_m: float
    fluid_density_kg_m3: float
    fluid_viscosity_pa_s: float
    vinf_fallback_m_s: float
    n_freq: int
    low_freq_skip: int
    min_samples: int
    initial_timestep_skip: int
    final_timestep_count: int | None
    st_offset: float
    summary_count: int
    plot_format: str
    write_plots: bool
    symmetric_append: bool
    periodic_aoa: bool
    allow_missing: bool


@dataclass(frozen=True)
class ReDirectory:
    path: Path
    reynolds: float
    source: str


@dataclass(frozen=True)
class TimeInfo:
    source_samples: int
    skipped_initial_steps: int
    samples: int
    time_start_s: float
    time_end_s: float
    source_dt_min_s: float
    source_dt_max_s: float
    dt_s: float
    resampled: bool


@dataclass(frozen=True)
class ExperimentalOverlay:
    """Headerless AOA/value tables for optional experimental plot overlays."""

    label: str
    source_dir: Path
    lift_st: np.ndarray
    drag_st: np.ndarray
    lift_psd: np.ndarray
    drag_psd: np.ndarray
    psd_scale: float
    source_files: tuple[Path, ...]


def parse_re_from_dirname(name: str) -> float:
    match = re.match(r"^RE(\d+)_([0-9]+)E([+-]?\d+)$", name)
    if not match:
        return float("nan")
    base, frac, exp = match.group(1), match.group(2), int(match.group(3))
    return float(f"{base}.{frac}e{exp}")


def parse_aoa_from_filename(path: Path) -> float:
    token = path.stem.split("_")[-1]
    numeric = re.sub(r"[^\d+\-.]+", "", token)
    if not numeric:
        raise ValueError(f"Could not parse AOA from file name: {path.name}")
    return float(numeric)


def infer_airfoil_name(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Could not infer airfoil name from file name: {path.name}")
    return "_".join(parts[:-1])


def infer_thickness_ratio(airfoil_name: str) -> float:
    naca = re.search(
        r"NACA\d{2}(\d{2})$", re.sub(r"[^A-Za-z0-9]+", "", airfoil_name.upper())
    )
    if naca:
        return float(naca.group(1)) / 100.0
    token = airfoil_name.split("_")[-1]
    numeric = re.sub(r"[^\d]+", "", token)
    if not numeric:
        raise ValueError(
            "Could not infer thickness ratio. Pass --thickness, e.g. 0.18."
        )
    return float(numeric) / 1000.0


def slug(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))


def load_experimental_table(path: Path) -> np.ndarray:
    """Load one headerless experimental AOA/value CSV table."""
    try:
        values = np.loadtxt(path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read experimental CSV {path}: {exc}") from exc
    if values.shape[1] != 2:
        raise ValueError(
            f"Experimental CSV {path} must have exactly two columns; got {values.shape[1]}."
        )
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Experimental CSV {path} must contain finite AOA/value rows.")
    return values[np.argsort(values[:, 0], kind="stable")]


def load_experimental_overlay(
    data_dir: Path,
    *,
    label: str | None = None,
    psd_scale: float = 1.0e-3,
) -> ExperimentalOverlay:
    """Load Swalwell-style lift/drag St and PSD peak tables.

    Figure 5(c) of Swalwell et al. reports its PSD ordinate as Power/Hz
    multiplied by 10^-3. The digitized CSV values are axis values, so only the
    PSD columns are multiplied by ``psd_scale``; Strouhal values are unchanged.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Experimental data directory does not exist: {data_dir}"
        )
    if not math.isfinite(psd_scale) or psd_scale <= 0.0:
        raise ValueError(
            f"Experimental PSD scale must be positive and finite; got {psd_scale!r}."
        )

    required = {
        "lift_st": "Lift_St_*.csv",
        "drag_st": "Drag_St_*.csv",
        "lift_psd": "Lift_PSD_*.csv",
        "drag_psd": "Drag_PSD_*.csv",
    }
    paths: dict[str, Path] = {}
    for key, pattern in required.items():
        matches = sorted(data_dir.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {pattern} file in {data_dir}; found {len(matches)}."
            )
        paths[key] = matches[0]

    if label is None:
        re_match = re.search(
            r"_RE([^.]*)\.csv$", paths["lift_st"].name, flags=re.IGNORECASE
        )
        re_label = re_match.group(1).replace("p", ".") if re_match else "unknown Re"
        label = f"Swalwell et al. NACA0021 (Re={re_label})"

    lift_psd = load_experimental_table(paths["lift_psd"])
    drag_psd = load_experimental_table(paths["drag_psd"])
    lift_psd[:, 1] *= psd_scale
    drag_psd[:, 1] *= psd_scale

    return ExperimentalOverlay(
        label=label,
        source_dir=data_dir,
        lift_st=load_experimental_table(paths["lift_st"]),
        drag_st=load_experimental_table(paths["drag_st"]),
        lift_psd=lift_psd,
        drag_psd=drag_psd,
        psd_scale=psd_scale,
        source_files=tuple(paths.values()),
    )


def dat_files_in(directory: Path) -> list[Path]:
    files = sorted(
        path for path in directory.iterdir() if path.suffix.lower() == ".dat"
    )
    if files:
        return files
    nested = directory / "data_files"
    if nested.is_dir():
        return sorted(
            path for path in nested.iterdir() if path.suffix.lower() == ".dat"
        )
    return []


def discover_re_directories(config: ProcessingConfig) -> list[ReDirectory]:
    if not config.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {config.input_dir}")

    re_dirs: list[ReDirectory] = []
    for path in sorted(config.input_dir.iterdir()):
        if path.is_dir():
            reynolds = parse_re_from_dirname(path.name)
            if math.isfinite(reynolds):
                re_dirs.append(ReDirectory(path, reynolds, "dirname"))
    if re_dirs:
        if config.re_override is not None:
            print(
                "Warning: --re ignored because RE* subdirectories were found.",
                file=sys.stderr,
            )
        return re_dirs

    if config.re_override is not None:
        reynolds = config.re_override
        source = "cli_re"
    else:
        reynolds = (
            config.fluid_density_kg_m3
            * config.vinf_fallback_m_s
            * config.chord_m
            / config.fluid_viscosity_pa_s
        )
        source = "fallback_vinf"
    return [ReDirectory(config.input_dir, reynolds, source)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_source_hash(rows: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_force_history(
    path: Path,
    *,
    min_samples: int,
    initial_timestep_skip: int,
    final_timestep_count: int | None = None,
) -> tuple[np.ndarray, TimeInfo]:
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < min_samples:
        raise ValueError(
            f"{path} has {data.shape[0]} samples; require at least {min_samples}."
        )
    if data.shape[1] <= 9:
        raise ValueError(f"{path} has {data.shape[1]} columns; require at least 10.")

    order = np.argsort(data[:, 0])
    data = data[order]
    time, unique_idx = np.unique(data[:, 0], return_index=True)
    data = data[unique_idx]
    source_samples = int(time.size)
    skipped_initial_steps = int(initial_timestep_skip)
    if final_timestep_count is not None:
        if initial_timestep_skip:
            raise ValueError(
                "initial_timestep_skip and final_timestep_count are mutually exclusive."
            )
        if final_timestep_count < 2 or final_timestep_count > time.size:
            raise ValueError(
                f"{path} cannot retain final {final_timestep_count} rows from {time.size}."
            )
        data = data[-final_timestep_count:]
        time = time[-final_timestep_count:]
        skipped_initial_steps = source_samples - final_timestep_count
        initial_timestep_skip = 0
    if initial_timestep_skip < 0:
        raise ValueError("initial_timestep_skip must be non-negative.")
    if initial_timestep_skip:
        if initial_timestep_skip >= time.size - 1:
            raise ValueError(
                f"{path} initial timestep skip {initial_timestep_skip} leaves fewer than two samples."
            )
        data = data[initial_timestep_skip:]
        time = time[initial_timestep_skip:]

    dt = np.diff(time)
    if dt.size == 0 or not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
        raise ValueError(f"{path} does not have increasing finite time values.")

    dt_med = float(np.median(dt))
    n_uniform = int(round((time[-1] - time[0]) / dt_med)) + 1
    uniform_time = time[0] + np.arange(n_uniform, dtype=float) * dt_med
    uniform_time = uniform_time[uniform_time <= time[-1] + 0.25 * dt_med]
    needs_resample = uniform_time.size != time.size or not np.allclose(
        dt, dt_med, rtol=1.0e-3, atol=max(1.0e-10, 1.0e-6 * dt_med)
    )
    if needs_resample:
        uniform = np.empty((uniform_time.size, data.shape[1]), dtype=float)
        uniform[:, 0] = uniform_time
        for col in range(1, data.shape[1]):
            uniform[:, col] = np.interp(uniform_time, time, data[:, col])
        data = uniform
    else:
        data = data.copy()
        data[:, 0] = uniform_time

    if data.shape[0] < min_samples:
        raise ValueError(
            f"{path} has {data.shape[0]} uniform samples; require {min_samples}."
        )

    info = TimeInfo(
        source_samples=source_samples,
        skipped_initial_steps=skipped_initial_steps,
        samples=int(data.shape[0]),
        time_start_s=float(data[0, 0]),
        time_end_s=float(data[-1, 0]),
        source_dt_min_s=float(np.min(dt)),
        source_dt_max_s=float(np.max(dt)),
        dt_s=float(data[1, 0] - data[0, 0]),
        resampled=bool(needs_resample),
    )
    return data, info


def load_dat(
    path: Path,
    *,
    min_samples: int,
    initial_timestep_skip: int = 0,
    final_timestep_count: int | None = None,
) -> np.ndarray:
    """Compatibility wrapper for callers that only need uniformized data."""
    data, _ = load_force_history(
        path,
        min_samples=min_samples,
        initial_timestep_skip=initial_timestep_skip,
        final_timestep_count=final_timestep_count,
    )
    return data


def compute_fft(
    signal: np.ndarray,
    dt: float,
    chord: float,
    aoa_deg: float,
    vinf: float,
    thickness_ratio: float,
    *,
    low_freq_skip: int = 10,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Return one-sided bins plus DC-first, power-ranked Strouhal/amplitude/phase.

    Positive-frequency amplitudes are doubled for their omitted negative-frequency
    partners.  DC and, for even-length records, the Nyquist bin are unpaired and
    therefore are not doubled.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size < 4:
        raise ValueError("signal must be one-dimensional with at least four samples.")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite.")
    if not math.isfinite(vinf) or vinf <= 0.0:
        raise ValueError("vinf must be positive and finite.")
    if not math.isfinite(chord) or chord <= 0.0:
        raise ValueError("chord must be positive and finite.")

    n_samples = values.size
    sample_rate = 1.0 / dt
    mean_amp = float(np.mean(values))
    demeaned = values - mean_amp

    window = hann(n_samples, sym=False)
    window_power = np.sum(window**2) / n_samples
    windowed_fft = rfft(window * demeaned)
    freqs = rfftfreq(n_samples, d=dt)
    power_density = (np.abs(windowed_fft) ** 2) / (
        sample_rate * n_samples * window_power
    )
    if n_samples % 2 == 0:
        power_density[1:-1] *= 2.0
    else:
        power_density[1:] *= 2.0
    power = power_density * (sample_rate / n_samples)

    raw_pos = rfft(demeaned)
    amps = np.abs(raw_pos) / n_samples
    if n_samples % 2 == 0:
        amps[1:-1] *= 2.0
    else:
        amps[1:] *= 2.0
    phases = np.angle(raw_pos)
    amps[0] = mean_amp
    phases[0] = 0.0

    first_peak = 1 + max(0, int(low_freq_skip))
    if power.size > first_peak:
        ranked = np.argsort(power[first_peak:])[::-1] + first_peak
        freqs_sorted = np.concatenate(([0.0], freqs[ranked]))
        amps_sorted = np.concatenate(([mean_amp], amps[ranked]))
        phases_sorted = np.concatenate(([0.0], phases[ranked]))
    else:
        freqs_sorted = np.array([0.0])
        amps_sorted = np.array([mean_amp])
        phases_sorted = np.array([0.0])

    projected_fraction = max(
        abs(math.sin(math.radians(aoa_deg))), float(thickness_ratio)
    )
    if projected_fraction <= 0.0:
        raise ValueError(
            "Strouhal reference length collapsed to zero. Pass --thickness."
        )
    strouhal = freqs_sorted * chord * projected_fraction / vinf
    return freqs, amps, phases, power, strouhal, amps_sorted, phases_sorted


def store_channel(
    arrays: dict[str, np.ndarray],
    prefix: str,
    i_re: int,
    i_aoa: int,
    signal: np.ndarray,
    *,
    dt: float,
    chord: float,
    aoa_deg: float,
    vinf: float,
    thickness_ratio: float,
    n_freq: int,
    low_freq_skip: int,
    st_offset: float,
) -> None:
    _, _, _, _, st, amp, phase = compute_fft(
        signal,
        dt,
        chord,
        aoa_deg,
        vinf,
        thickness_ratio,
        low_freq_skip=low_freq_skip,
    )
    if st.size < n_freq:
        raise ValueError(
            f"{prefix} spectrum has {st.size} frequency entries; require {n_freq}."
        )
    st = st[:n_freq].copy()
    st[1:] += st_offset
    arrays[f"{prefix}_ST"][i_re, i_aoa, :] = st
    arrays[f"{prefix}_Amp"][i_re, i_aoa, :] = amp[:n_freq]
    arrays[f"{prefix}_Pha"][i_re, i_aoa, :] = phase[:n_freq]


def sorted_arrays(
    arrays: dict[str, np.ndarray],
    aoa: np.ndarray,
    reynolds: np.ndarray,
    *,
    symmetric_append: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    aoa_order = np.argsort(aoa)
    re_order = np.argsort(reynolds)
    arrays = {
        name: values[re_order][:, aoa_order, :] for name, values in arrays.items()
    }
    aoa = aoa[aoa_order]
    reynolds = reynolds[re_order]

    if symmetric_append:
        if np.any(aoa < 0.0):
            raise ValueError(
                "--symmetric-append expects one-sided nonnegative AOA data."
            )
        mirror_idx = np.flatnonzero(~np.isclose(aoa, 0.0))[::-1]
        aoa = np.concatenate([-aoa[mirror_idx], aoa])
        for name, values in list(arrays.items()):
            mirrored = values[:, mirror_idx, :].copy()
            if name.startswith(("CL_Amp", "CM_Amp")):
                mirrored[:, :, 0] *= -1.0
            if name.startswith(("CL_Pha", "CM_Pha")) and mirrored.shape[2] > 1:
                mirrored[:, :, 1:] = (mirrored[:, :, 1:] + math.pi + math.pi) % (
                    2.0 * math.pi
                ) - math.pi
            arrays[name] = np.concatenate([mirrored, values], axis=1)
    return arrays, aoa, reynolds


def close_periodic_aoa_grid(
    arrays: dict[str, np.ndarray],
    aoa: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    """Close a near-complete ``[-180, 180]`` grid with a periodic endpoint.

    VorLap's flow-state angle comes from ``atan2`` and therefore lies in
    ``[-180, 180]``. A full-circle source commonly stores ``-180`` through
    ``176`` degrees because ``+180`` is the same orientation as ``-180``.
    Append the missing ``+180`` endpoint by copying the ``-180`` spectra so
    ordinary linear interpolation is continuous through the final grid cell.
    """
    aoa_values = np.asarray(aoa, dtype=float).reshape(-1)
    if aoa_values.size < 2:
        raise ValueError("--periodic-aoa requires at least two AOA grid points.")
    if np.any(~np.isfinite(aoa_values)) or np.any(np.diff(aoa_values) <= 0.0):
        raise ValueError(
            "--periodic-aoa requires a finite, strictly increasing AOA grid."
        )

    tolerance_deg = 1.0e-8
    if not math.isclose(float(aoa_values[0]), -180.0, abs_tol=tolerance_deg):
        raise ValueError(
            "--periodic-aoa requires the source AOA grid to start at -180 degrees; "
            f"found {aoa_values[0]:g}."
        )
    if math.isclose(float(aoa_values[-1]), 180.0, abs_tol=tolerance_deg):
        return arrays, aoa_values, False
    if aoa_values[-1] > 180.0:
        raise ValueError(
            "--periodic-aoa requires the source AOA grid to end at or below 180 degrees; "
            f"found {aoa_values[-1]:g}."
        )

    spacings = np.diff(aoa_values)
    seam_spacing = 180.0 - float(aoa_values[-1])
    largest_source_spacing = float(np.max(spacings))
    if seam_spacing > largest_source_spacing + tolerance_deg:
        raise ValueError(
            "--periodic-aoa source does not reach the periodic seam: "
            f"last AOA is {aoa_values[-1]:g} degrees, leaving a {seam_spacing:g}-degree gap "
            f"larger than the largest source spacing {largest_source_spacing:g} degrees."
        )

    closed_arrays: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        tensor = np.asarray(values)
        if tensor.ndim != 3 or tensor.shape[1] != aoa_values.size:
            raise ValueError(
                f"{name} must have shape [Re, AOA, frequency] with "
                f"{aoa_values.size} AOA rows; got {tensor.shape}."
            )
        closed_arrays[name] = np.concatenate([tensor, tensor[:, :1, :]], axis=1)
    return closed_arrays, np.concatenate([aoa_values, [180.0]]), True


def replicate_single_reynolds_grid(
    arrays: dict[str, np.ndarray],
    source_reynolds: np.ndarray,
    target_reynolds: Sequence[float],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Replicate one measured spectrum over an explicitly declared Re grid.

    This represents a Reynolds-independent modeling assumption; it does not
    synthesize Reynolds-resolved data. The source Reynolds number must remain
    one of the output grid points so the measured basis is not obscured.
    """
    source = np.asarray(source_reynolds, dtype=float).reshape(-1)
    target = np.asarray(target_reynolds, dtype=float).reshape(-1)
    if source.size != 1:
        raise ValueError(
            "--replicate-re-grid requires exactly one source Reynolds number; "
            f"found {source.size}."
        )
    if target.size < 2:
        raise ValueError(
            "--replicate-re-grid requires at least two Reynolds grid points."
        )
    if not np.all(np.isfinite(target)) or np.any(target <= 0.0):
        raise ValueError("--replicate-re-grid values must be positive and finite.")
    if np.any(np.diff(target) <= 0.0):
        raise ValueError(
            "--replicate-re-grid values must be unique and strictly increasing."
        )
    if not np.any(np.isclose(target, source[0], rtol=1.0e-12, atol=0.0)):
        raise ValueError(
            "--replicate-re-grid must include the source Reynolds number "
            f"{source[0]:g}."
        )
    replicated = {
        name: np.repeat(np.asarray(values, dtype=float), target.size, axis=0)
        for name, values in arrays.items()
    }
    return replicated, target


def default_output(input_dir: Path, airfoil_name: str) -> Path:
    return input_dir / f"{airfoil_name}.h5"


def default_plots_dir(output: Path) -> Path:
    return output.parent / "figs"


def split_experimental_branches(
    values: np.ndarray, *, aoa_tolerance_deg: float = 0.5
) -> list[np.ndarray]:
    """Split close AOA pairs such as primary/secondary experimental peaks."""
    if values.shape[0] == 0:
        return []
    groups: list[list[np.ndarray]] = []
    for row in values:
        if not groups or float(row[0] - groups[-1][-1][0]) > aoa_tolerance_deg:
            groups.append([row])
        else:
            groups[-1].append(row)

    max_group_size = max(len(group) for group in groups)
    branches: list[np.ndarray] = []
    for rank in range(max_group_size):
        rows = []
        for group in groups:
            ordered = sorted(group, key=lambda row: float(row[1]))
            if rank < len(ordered):
                rows.append(ordered[rank])
        if rows:
            branches.append(np.asarray(rows, dtype=float))
    return branches


def build_airfoil_fft(config: ProcessingConfig) -> dict[str, object]:
    re_dirs = discover_re_directories(config)
    base_files = dat_files_in(re_dirs[0].path)
    if not base_files:
        raise FileNotFoundError(
            f"No .dat files found in {re_dirs[0].path} or data_files/."
        )

    airfoil_name = config.airfoil_name or infer_airfoil_name(base_files[0])
    thickness = (
        config.thickness_ratio
        if config.thickness_ratio is not None
        else infer_thickness_ratio(airfoil_name)
    )
    if not math.isfinite(thickness) or thickness < 0.0:
        raise ValueError("--thickness must be finite and non-negative.")

    aoa = np.array([parse_aoa_from_filename(path) for path in base_files], dtype=float)
    reynolds = np.array([row.reynolds for row in re_dirs], dtype=float)
    shape = (len(re_dirs), len(base_files), config.n_freq)
    arrays = {
        f"{field}_{kind}": np.zeros(shape, dtype=float)
        for field in ("CL", "CD", "CM", "CF")
        for kind in ("ST", "Amp", "Pha")
    }
    source_rows: list[dict[str, object]] = []

    for i_re, re_dir in enumerate(re_dirs):
        vinf = (
            reynolds[i_re]
            * config.fluid_viscosity_pa_s
            / (config.fluid_density_kg_m3 * config.chord_m)
        )
        files_i = {path.name: path for path in dat_files_in(re_dir.path)}
        for i_aoa, base_file in enumerate(base_files):
            path = files_i.get(base_file.name)
            if path is None:
                message = f"AOA file {base_file.name!r} not found in {re_dir.path}."
                if config.allow_missing:
                    print(f"Warning: {message}", file=sys.stderr)
                    continue
                raise FileNotFoundError(message)

            data, time_info = load_force_history(
                path,
                min_samples=config.min_samples,
                initial_timestep_skip=config.initial_timestep_skip,
                final_timestep_count=config.final_timestep_count,
            )
            q = (
                0.5
                * config.fluid_density_kg_m3
                * vinf**2
                * config.chord_m
                * config.span_m
            )
            cl = (data[:, 2] + data[:, 5]) / q
            cd = (data[:, 1] + data[:, 4]) / q
            # Nalu integrated-force columns are Time,Fp{x,y,z},Fv{x,y,z},
            # Mt{x,y,z}. The aerodynamic pitching moment is Mtz and its
            # coefficient reference is q*c.
            cm = data[:, 9] / (q * config.chord_m)
            cf = np.sqrt(cd**2 + cl**2)

            common = {
                "dt": time_info.dt_s,
                "chord": config.chord_m,
                "aoa_deg": float(aoa[i_aoa]),
                "vinf": float(vinf),
                "thickness_ratio": thickness,
                "n_freq": config.n_freq,
                "low_freq_skip": config.low_freq_skip,
                "st_offset": config.st_offset,
            }
            store_channel(arrays, "CL", i_re, i_aoa, cl, **common)
            store_channel(arrays, "CD", i_re, i_aoa, cd, **common)
            store_channel(arrays, "CM", i_re, i_aoa, cm, **common)
            store_channel(arrays, "CF", i_re, i_aoa, cf, **common)

            if config.source_path_root is None:
                source_path = str(path.resolve())
                source_path_kind = "absolute"
            else:
                source_root = config.source_path_root.resolve()
                try:
                    source_path = path.resolve().relative_to(source_root).as_posix()
                except ValueError as exc:
                    raise ValueError(
                        f"Source file {path} is outside --source-path-root {source_root}."
                    ) from exc
                source_path_kind = "relative_to_source_path_root"
            source_rows.append(
                {
                    "path": source_path,
                    "path_kind": source_path_kind,
                    "sha256": sha256_file(path),
                    "reynolds": float(reynolds[i_re]),
                    "re_source": re_dir.source,
                    "aoa_deg": float(aoa[i_aoa]),
                    "source_samples": time_info.source_samples,
                    "skipped_initial_steps": time_info.skipped_initial_steps,
                    "samples": time_info.samples,
                    "time_start_s": time_info.time_start_s,
                    "time_end_s": time_info.time_end_s,
                    "dt_s": time_info.dt_s,
                    "source_dt_min_s": time_info.source_dt_min_s,
                    "source_dt_max_s": time_info.source_dt_max_s,
                    "resampled_to_uniform_time": time_info.resampled,
                    "vinf_m_s": float(vinf),
                }
            )

    arrays, aoa_sorted, source_re_sorted = sorted_arrays(
        arrays,
        aoa,
        reynolds,
        symmetric_append=config.symmetric_append,
    )
    periodic_aoa_endpoint_added = False
    if config.periodic_aoa:
        arrays, aoa_sorted, periodic_aoa_endpoint_added = close_periodic_aoa_grid(
            arrays,
            aoa_sorted,
        )
    re_sorted = source_re_sorted
    if config.replicate_re_grid is not None:
        arrays, re_sorted = replicate_single_reynolds_grid(
            arrays,
            source_re_sorted,
            config.replicate_re_grid,
        )
    output = config.output or default_output(config.input_dir, airfoil_name)
    plots_dir = config.plots_dir or default_plots_dir(output)
    experimental = (
        load_experimental_overlay(
            config.experimental_data_dir,
            label=config.experimental_label,
            psd_scale=config.experimental_psd_scale,
        )
        if config.experimental_data_dir is not None
        else None
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = combined_source_hash(source_rows)
    processing_config = {
        "input_dir": str(config.input_dir.resolve()),
        "experimental_data_dir": (
            str(config.experimental_data_dir.resolve())
            if config.experimental_data_dir is not None
            else None
        ),
        "experimental_label": experimental.label if experimental is not None else None,
        "experimental_psd_scale": (
            experimental.psd_scale if experimental is not None else None
        ),
        "source_reynolds": source_re_sorted.tolist(),
        "output_reynolds": re_sorted.tolist(),
        "reynolds_model": (
            "replicated_reynolds_independent"
            if config.replicate_re_grid is not None
            else "source_resolved"
        ),
        "replicate_re_grid": (
            list(config.replicate_re_grid)
            if config.replicate_re_grid is not None
            else None
        ),
        "replicate_re_justification": config.replicate_re_justification,
        "experimental_files": (
            [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in experimental.source_files
            ]
            if experimental is not None
            else []
        ),
        "airfoil_name": airfoil_name,
        "thickness_ratio": thickness,
        "re_override": config.re_override,
        "chord_m": config.chord_m,
        "span_m": config.span_m,
        "fluid_density_kg_m3": config.fluid_density_kg_m3,
        "fluid_viscosity_pa_s": config.fluid_viscosity_pa_s,
        "vinf_fallback_m_s": config.vinf_fallback_m_s,
        "n_freq": config.n_freq,
        "low_freq_skip": config.low_freq_skip,
        "min_samples": config.min_samples,
        "initial_timestep_skip": config.initial_timestep_skip,
        "final_timestep_count": config.final_timestep_count,
        "st_offset": config.st_offset,
        "summary_count": config.summary_count,
        "symmetric_append": config.symmetric_append,
        "periodic_aoa": config.periodic_aoa,
        "periodic_aoa_endpoint_added": periodic_aoa_endpoint_added,
        "allow_missing": config.allow_missing,
    }

    with h5py.File(output, "w") as h5:
        h5.create_dataset(
            "Airfoilname", data=np.array(airfoil_name, dtype=h5py.string_dtype("utf-8"))
        )
        h5.create_dataset("Re", data=re_sorted)
        h5.create_dataset("Thickness", data=np.array(thickness))
        h5.create_dataset("AOA", data=aoa_sorted)
        for name, values in arrays.items():
            h5.create_dataset(name, data=values)
        h5.attrs["generator"] = "scripts/process_naluout_to_h5.py"
        h5.attrs["source_data_sha256"] = source_hash
        h5.attrs["source_files_json"] = json.dumps(source_rows, sort_keys=True)
        h5.attrs["processing_config_json"] = json.dumps(
            processing_config, sort_keys=True
        )
        h5.attrs["reynolds_model"] = processing_config["reynolds_model"]
        if config.replicate_re_justification is not None:
            h5.attrs["reynolds_replication_justification"] = (
                config.replicate_re_justification
            )

    plot_paths: list[str] = []
    if config.write_plots:
        plot_paths = write_summary_plots(
            plots_dir,
            airfoil_name,
            aoa_sorted,
            re_sorted,
            arrays,
            count=config.summary_count,
            plot_format=config.plot_format,
            experimental=experimental,
        )

    return {
        "output": str(output),
        "plots": plot_paths,
        "airfoil_name": airfoil_name,
        "thickness_ratio": thickness,
        "reynolds": re_sorted.tolist(),
        "source_reynolds": source_re_sorted.tolist(),
        "reynolds_model": processing_config["reynolds_model"],
        "aoa_deg": aoa_sorted.tolist(),
        "periodic_aoa": config.periodic_aoa,
        "periodic_aoa_endpoint_added": periodic_aoa_endpoint_added,
        "source_files": len(source_rows),
        "resampled_source_files": sum(
            1 for row in source_rows if row["resampled_to_uniform_time"]
        ),
        "skipped_initial_steps": (
            config.initial_timestep_skip
            if config.final_timestep_count is None
            else "source_samples_minus_final_timestep_count"
        ),
        "final_timestep_count": config.final_timestep_count,
        "source_data_sha256": source_hash,
        "experimental_data_dir": (
            str(config.experimental_data_dir)
            if config.experimental_data_dir is not None
            else None
        ),
        "experimental_psd_scale": (
            experimental.psd_scale if experimental is not None else None
        ),
    }


def write_summary_plots(
    plots_dir: Path,
    airfoil_name: str,
    aoa: np.ndarray,
    reynolds: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    count: int,
    plot_format: str,
    experimental: ExperimentalOverlay | None = None,
) -> list[str]:
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

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: list[str] = []
    max_i = min(max(1, count), arrays["CF_ST"].shape[2])
    for i_re, reynolds_value in enumerate(reynolds):
        re_str = slug(f"{reynolds_value:.5g}")
        for field, ylabel, title, ylim, stem in (
            (
                "CF_ST",
                "Strouhal number based on CF",
                "Strouhal number based on CF",
                (0.0, 0.5),
                "STCF",
            ),
            ("CF_Amp", "Amplitude based on CF", "Amplitude based on CF", None, "AmpCF"),
        ):
            fig, ax = plt.subplots(figsize=(9.9, 5.9))
            for i_freq in range(1, max_i):
                ax.plot(
                    aoa,
                    arrays[field][i_re, :, i_freq],
                    marker="x",
                    linewidth=0.0,
                    markersize=4,
                    label=(
                        "VorLap NACA0018 CF spectral peaks"
                        if i_freq == 1
                        else "_nolegend_"
                    ),
                )
            ax.set_xlabel("AoA (deg)")
            ax.set_ylabel(
                "CF Fourier amplitude; experimental PSD (coefficient²/Hz)"
                if field == "CF_Amp" and experimental is not None
                else ylabel
            )
            ax.set_title(
                "CF Fourier amplitudes and experimental coefficient PSD peaks"
                if field == "CF_Amp" and experimental is not None
                else title
            )
            if ylim is not None:
                ax.set_ylim(*ylim)
            elif field == "CF_Amp":
                ax.set_ylim(bottom=0.0)
            ax.set_xlim(float(np.min(aoa)), float(np.max(aoa)))
            if experimental is not None:
                if field == "CF_ST":
                    for branch_index, branch in enumerate(
                        split_experimental_branches(experimental.lift_st)
                    ):
                        ax.plot(
                            branch[:, 0],
                            branch[:, 1],
                            linestyle="-",
                            marker="o",
                            linewidth=0.9,
                            markersize=3,
                            color="black",
                            label=(
                                f"{experimental.label} lift"
                                if branch_index == 0
                                else "_nolegend_"
                            ),
                        )
                    for branch_index, branch in enumerate(
                        split_experimental_branches(experimental.drag_st)
                    ):
                        ax.plot(
                            branch[:, 0],
                            branch[:, 1],
                            linestyle="-",
                            marker="s",
                            linewidth=0.9,
                            markersize=3,
                            color="black",
                            label=(
                                f"{experimental.label} drag"
                                if branch_index == 0
                                else "_nolegend_"
                            ),
                        )
                    ax.legend(loc="best", fontsize=8)
                else:
                    psd_scale_label = (
                        "10⁻³"
                        if math.isclose(experimental.psd_scale, 1.0e-3)
                        else f"{experimental.psd_scale:g}"
                    )
                    ax.plot(
                        experimental.lift_psd[:, 0],
                        experimental.lift_psd[:, 1],
                        linestyle="-",
                        marker="o",
                        linewidth=0.9,
                        markersize=3,
                        color="black",
                        label=(
                            f"{experimental.label} lift PSD "
                            f"(digitized ordinate ×{psd_scale_label})"
                        ),
                    )
                    ax.plot(
                        experimental.drag_psd[:, 0],
                        experimental.drag_psd[:, 1],
                        linestyle="-",
                        marker="s",
                        linewidth=0.9,
                        markersize=3,
                        color="black",
                        label=(
                            f"{experimental.label} drag PSD "
                            f"(digitized ordinate ×{psd_scale_label})"
                        ),
                    )
                    ax.legend(loc="best", fontsize=8)
                    ymax = max(
                        float(np.max(arrays[field][i_re, :, 1:max_i])),
                        float(np.max(experimental.lift_psd[:, 1])),
                        float(np.max(experimental.drag_psd[:, 1])),
                    )
                    ax.set_ylim(0.0, ymax * 1.05 if ymax > 0.0 else 1.0)
            path = plots_dir / f"{airfoil_name}{stem}_summary_Re{re_str}.{plot_format}"
            fig.savefig(path, bbox_inches="tight", dpi=150, transparent=True)
            plt.close(fig)
            plot_paths.append(str(path))
    return plot_paths


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive finite value, got {text!r}."
        )
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative finite value, got {text!r}."
        )
    return value


def finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"Expected a finite value, got {text!r}.")
    return value


def positive_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part) for part in text.replace(",", " ").split())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a whitespace/comma-separated positive-float list, got {text!r}."
        ) from exc
    if len(values) < 2:
        raise argparse.ArgumentTypeError("Expected at least two Reynolds grid values.")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError(
            "Reynolds grid values must be positive and finite."
        )
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError(
            "Reynolds grid values must be unique and strictly increasing."
        )
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Nalu force-history .dat files to a VorLap airfoil FFT HDF5 file.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Airfoil folder, RE* root, or folder with data_files/.",
    )
    parser.add_argument(
        "--source-path-root",
        type=Path,
        default=None,
        help=(
            "Optional common source root. Source-file paths embedded in the HDF5 "
            "are stored relative to this directory so outputs are location-independent."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5. Default: <input-dir>/<airfoil>.h5.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Summary plot directory. Default: <output-dir>/figs.",
    )
    parser.add_argument(
        "--experimental-data-dir",
        type=Path,
        default=None,
        help="Optional directory containing headerless Lift/Drag *_St_*.csv and *_PSD_*.csv overlay tables.",
    )
    parser.add_argument(
        "--experimental-label",
        default=None,
        help="Optional legend label for experimental overlay points.",
    )
    parser.add_argument(
        "--experimental-psd-scale",
        type=positive_float,
        default=1.0e-3,
        help=(
            "Multiplier applied to digitized experimental PSD ordinates "
            "(default: 1e-3 for Swalwell Figure 5(c); use 1 for already-scaled data)."
        ),
    )
    parser.add_argument(
        "--replicate-re-grid",
        type=positive_float_list,
        default=None,
        help=(
            "Explicit increasing Re grid over which a single-Re source spectrum is repeated. "
            "This declares a Reynolds-independent model and requires --replicate-re-justification."
        ),
    )
    parser.add_argument(
        "--replicate-re-justification",
        default=None,
        help="Required provenance statement when --replicate-re-grid is used.",
    )
    parser.add_argument(
        "--airfoil-name",
        "--airfoil",
        default=None,
        help="Airfoil name for HDF5. Defaults to file-name inference.",
    )
    parser.add_argument(
        "--thickness",
        type=nonnegative_float,
        default=None,
        help="Relative thickness t/c, e.g. 0.18 for NACA0018.",
    )
    parser.add_argument(
        "--re",
        type=positive_float,
        default=None,
        help="Reynolds number for flat inputs without RE* subdirectories.",
    )
    parser.add_argument(
        "--chord", type=positive_float, default=1.0, help="Reference chord length [m]."
    )
    parser.add_argument(
        "--span",
        type=positive_float,
        default=4.0,
        help="Reference span for force normalization [m].",
    )
    parser.add_argument(
        "--fluid-density",
        type=positive_float,
        default=1.2,
        help="Fluid density [kg/m^3].",
    )
    parser.add_argument(
        "--fluid-viscosity",
        type=positive_float,
        default=9.0e-6,
        help="Dynamic viscosity [Pa s].",
    )
    parser.add_argument(
        "--vinf-fallback",
        type=positive_float,
        default=2.0,
        help="Fallback speed if neither RE* folders nor --re are present [m/s].",
    )
    parser.add_argument(
        "--n-freq",
        type=int,
        default=200,
        help="Frequency entries to store, including DC.",
    )
    parser.add_argument(
        "--low-freq-skip",
        type=int,
        default=30,
        help="Non-DC FFT bins skipped before power ranking.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1000,
        help="Minimum raw/uniform samples required per .dat file.",
    )
    parser.add_argument(
        "--initial-timestep-skip",
        type=int,
        default=1000,
        help="Number of initial sorted/unique time rows dropped before resampling and FFT.",
    )
    parser.add_argument(
        "--final-timestep-count",
        type=int,
        default=None,
        help=(
            "Retain exactly the final N sorted/unique rows. Mutually exclusive "
            "with a nonzero --initial-timestep-skip."
        ),
    )
    parser.add_argument(
        "--st-offset",
        type=finite_float,
        default=0.0,
        help="Offset added to non-DC Strouhal values; use 0.07 to recreate the 2025 paper NACA0018 tuning.",
    )
    parser.add_argument(
        "--summary-count",
        type=int,
        default=30,
        help="Frequency entries shown in CF summary plots.",
    )
    parser.add_argument(
        "--plot-format",
        choices=("pdf", "png"),
        default="pdf",
        help="Summary plot format.",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip summary plot generation."
    )
    parser.add_argument(
        "--symmetric-append",
        action="store_true",
        help="Mirror positive-AOA data to negative AOA for one-sided symmetric sources.",
    )
    parser.add_argument(
        "--periodic-aoa",
        action="store_true",
        help=(
            "Close a full-circle -180..(<180) AOA grid by appending +180 as an "
            "exact periodic copy of -180."
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Warn and leave zeros for missing AOA files instead of failing.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    if args.n_freq < 1:
        raise ValueError("--n-freq must be at least 1.")
    if args.summary_count < 1:
        raise ValueError("--summary-count must be at least 1.")
    if args.low_freq_skip < 0:
        raise ValueError("--low-freq-skip must be non-negative.")
    if args.min_samples < 4:
        raise ValueError("--min-samples must be at least 4.")
    if args.initial_timestep_skip < 0:
        raise ValueError("--initial-timestep-skip must be non-negative.")
    if args.final_timestep_count is not None and args.final_timestep_count < 2:
        raise ValueError("--final-timestep-count must be at least 2.")
    if args.final_timestep_count is not None and args.initial_timestep_skip:
        raise ValueError("--final-timestep-count requires --initial-timestep-skip 0.")
    if (
        args.replicate_re_grid is not None
        and not str(args.replicate_re_justification or "").strip()
    ):
        raise ValueError(
            "--replicate-re-grid requires a nonempty --replicate-re-justification."
        )
    if args.replicate_re_grid is None and args.replicate_re_justification is not None:
        raise ValueError("--replicate-re-justification requires --replicate-re-grid.")
    return ProcessingConfig(
        input_dir=args.input_dir,
        source_path_root=args.source_path_root,
        output=args.output,
        plots_dir=args.plots_dir,
        experimental_data_dir=args.experimental_data_dir,
        experimental_label=args.experimental_label,
        experimental_psd_scale=args.experimental_psd_scale,
        replicate_re_grid=args.replicate_re_grid,
        replicate_re_justification=args.replicate_re_justification,
        airfoil_name=args.airfoil_name,
        thickness_ratio=args.thickness,
        re_override=args.re,
        chord_m=args.chord,
        span_m=args.span,
        fluid_density_kg_m3=args.fluid_density,
        fluid_viscosity_pa_s=args.fluid_viscosity,
        vinf_fallback_m_s=args.vinf_fallback,
        n_freq=args.n_freq,
        low_freq_skip=args.low_freq_skip,
        min_samples=args.min_samples,
        initial_timestep_skip=args.initial_timestep_skip,
        final_timestep_count=args.final_timestep_count,
        st_offset=args.st_offset,
        summary_count=args.summary_count,
        plot_format=args.plot_format,
        write_plots=not bool(args.no_plots),
        symmetric_append=bool(args.symmetric_append),
        periodic_aoa=bool(args.periodic_aoa),
        allow_missing=bool(args.allow_missing),
    )


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    summary = build_airfoil_fft(config_from_args(parse_args(argv)))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
