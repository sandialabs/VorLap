#!/usr/bin/env python3
"""Run the native-QBlade axis and free-decay checks for the Ehrmann surrogate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping

import numpy as np

from . import campaign
from vorlap.qblade_files import replace_qblade_keyword_text
from vorlap.qblade_loading_file import write_qblade_loading_file
from vorlap.qblade_run_support import atomic_write_json
from vorlap.qblade_sil import normalize_qblade_header, read_qblade_result_numeric_columns


SCHEMA = "vorlap-ehrmann2014-structural-verification-v1"
TIP_NODE = "BLD_1_1.000000"
GLOBAL_Y_POSITION = "Y_g Pos. BLD_1 pos 1.000 [m]"
BODY_TIP_CHANNELS = (
    "X_b Tip Trl.Def. BLD_1 [m]",
    "Y_b Tip Trl.Def. BLD_1 [m]",
)
PROBE_FORCE_N = 10.0
PROBE_STEPS = 1000
DECAY_STEPS = 4000
BURST_CYCLES = 3
BURST_FORCE_N = 5.0


class VerificationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch(path: Path, values: Mapping[str, object]) -> None:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        text, _old, count = replace_qblade_keyword_text(text, key, value)
        if count != 1:
            raise VerificationError(f"Could not patch {key} in {path}.")
    path.write_text(text, encoding="utf-8")


def _raw_case(manifest: Mapping[str, object], mass_damping: float) -> Mapping[str, object]:
    return next(
        row for row in manifest["cases"]
        if row["scenario"] == "raw_qblade"
        and row["mass_damping"] == mass_damping
        and row["reduced_velocity"] == 3.0
        and row["timestep_s"] == campaign.TIMESTEP_S
    )


def prepare(campaign_root: Path, output: Path) -> Path:
    state = campaign.validate(campaign_root)
    root = Path(output).expanduser().resolve()
    if root.exists():
        raise FileExistsError(root)
    (root / "cases").mkdir(parents=True)
    (root / "results").mkdir()
    cases: list[dict[str, object]] = []
    source_by_damping = {
        value: _raw_case(state["manifest"], value) for value in campaign.MASS_DAMPINGS
    }

    def stage(label: str, source: Mapping[str, object], steps: int) -> tuple[Path, Path]:
        source_dir = state["root"] / str(source["case_dir"])
        case_dir = root / "cases" / label
        shutil.copytree(source_dir, case_dir)
        sim = case_dir / Path(str(source["sim"])).name
        _patch(
            sim,
            {
                "OBJECTNAME": label,
                "INCLUDEAERO": 0,
                "MEANINF": 0.0,
                "NUMTIMESTEPS": steps,
                "STOREFROM": 0.0,
                "CALCMODAL": "false",
                "LOADINGFILE": "",
            },
        )
        return case_dir, sim

    for sign in (1, -1):
        label = "global_y_positive" if sign > 0 else "global_y_negative"
        _case_dir, sim = stage(label, source_by_damping[campaign.MASS_DAMPINGS[0]], PROBE_STEPS)
        cases.append(
            {
                "kind": "axis_probe",
                "label": label,
                "sim": str(sim.relative_to(root)),
                "result_dir": f"results/{label}",
                "result_name": f"{label}_results",
                "force_n": sign * PROBE_FORCE_N,
                "mass_damping": campaign.MASS_DAMPINGS[0],
            }
        )

    for mass_damping in campaign.MASS_DAMPINGS:
        label = f"decay_md{mass_damping:.2f}".replace(".", "p")
        case_dir, sim = stage(label, source_by_damping[mass_damping], DECAY_STEPS)
        timestep = campaign.TIMESTEP_S
        time = np.arange(DECAY_STEPS + 1, dtype=float) * timestep
        burst_duration = BURST_CYCLES / campaign.NATURAL_FREQUENCY_HZ
        phase = 2.0 * math.pi * campaign.NATURAL_FREQUENCY_HZ * time
        force_y = np.where(
            time <= burst_duration + 1.0e-12,
            BURST_FORCE_N * np.sin(phase) * np.sin(np.pi * time / burst_duration) ** 2,
            0.0,
        )
        force = np.zeros((time.size, 3, 1), dtype=float)
        force[:, 1, 0] = force_y
        loads = case_dir / "global_y_burst_loading.txt"
        write_qblade_loading_file(str(loads), time, force, [TIP_NODE], local=False)
        _patch(sim, {"LOADINGFILE": loads.name})
        cases.append(
            {
                "kind": "free_decay",
                "label": label,
                "sim": str(sim.relative_to(root)),
                "result_dir": f"results/{label}",
                "result_name": f"{label}_results",
                "loading_file": str(loads.relative_to(root)),
                "burst_end_s": burst_duration,
                "mass_damping": mass_damping,
                "target_damping_ratio": mass_damping / campaign.MASS_RATIO,
            }
        )
    manifest = {
        "schema": SCHEMA,
        "campaign_manifest_sha256": _sha256(state["root"] / "campaign_manifest.json"),
        "native_aerodynamics": False,
        "force_frame": "global_Y",
        "orthogonal_bending_stiffness_ratio_EIx_over_EIy": campaign.ORTHOGONAL_BENDING_STIFFNESS_RATIO,
        "cases": cases,
    }
    path = root / "structural_verification_manifest.json"
    atomic_write_json(path, manifest)
    return path


def _command(root: Path, row: Mapping[str, object], qblade_root: Path, threads: int) -> list[str]:
    command = [
        sys.executable,
        str(campaign.REPO_ROOT / "run_headless/run_qblade_sim.py"),
        "--sim", str(root / str(row["sim"])),
        "--qblade-root", str(Path(qblade_root).resolve()),
        "--output-dir", str(root / str(row["result_dir"])),
        "--result-name", str(row["result_name"]),
        "--result-filter", str(campaign.REPO_ROOT / "examples/vlastos2026_cantilever/cantilever_results.filter"),
        "--threads", str(threads),
        "--coupling-mode", "none",
    ]
    if row["kind"] == "axis_probe":
        command[command.index("none")] = "sil-constant-force"
        command.extend(("--sil-constant-force", f"{TIP_NODE}:Y:{row['force_n']}:global"))
    return command


def run(root: Path, qblade_root: Path, threads: int) -> None:
    if threads < 1 or threads > 4:
        raise VerificationError("Structural verification is capped at four QBlade threads.")
    root = Path(root).resolve()
    manifest = json.loads((root / "structural_verification_manifest.json").read_text())
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    for row in manifest["cases"]:
        subprocess.run(_command(root, row, qblade_root, threads), check=True, cwd=campaign.REPO_ROOT, env=env)


def _columns(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw = read_qblade_result_numeric_columns(path)
    normalized = {normalize_qblade_header(name): np.asarray(values, dtype=float) for name, values in raw.items()}
    if "Time [s]" not in normalized:
        raise VerificationError(f"Missing time column: {path}")
    return normalized["Time [s]"], normalized


def _paired_axis(root: Path, cases: list[Mapping[str, object]]) -> dict[str, object]:
    rows = [row for row in cases if row["kind"] == "axis_probe"]
    signals = []
    for row in rows:
        result = root / str(row["result_dir"]) / f"{row['result_name']}.txt"
        time, columns = _columns(result)
        signals.append((time, columns))
    if signals[0][0].shape != signals[1][0].shape or not np.allclose(signals[0][0], signals[1][0]):
        raise VerificationError("Positive/negative axis-probe time grids differ.")
    retain = signals[0][0] >= 0.1
    global_delta = signals[0][1][GLOBAL_Y_POSITION][retain] - signals[1][1][GLOBAL_Y_POSITION][retain]
    global_rms = float(np.sqrt(np.mean(global_delta**2)))
    candidates = []
    for channel in BODY_TIP_CHANNELS:
        delta = signals[0][1][channel][retain] - signals[1][1][channel][retain]
        rms = float(np.sqrt(np.mean(delta**2)))
        correlation = float(np.corrcoef(delta, global_delta)[0, 1])
        candidates.append({"channel": channel, "paired_rms_m": rms, "global_y_correlation": correlation})
    candidates.sort(key=lambda row: row["paired_rms_m"], reverse=True)
    winner, other = candidates
    passed = (
        global_rms > 1.0e-8
        and winner["channel"] == campaign.RESULT_CHANNEL
        and winner["paired_rms_m"] > 20.0 * max(other["paired_rms_m"], 1.0e-12)
        and winner["global_y_correlation"] > 0.99
    )
    return {
        "passed": passed,
        "native_aerodynamics": False,
        "applied_force": "+/- global Y",
        "global_y_position_channel": GLOBAL_Y_POSITION,
        "global_y_paired_rms_m": global_rms,
        "verified_body_tip_channel": winner["channel"],
        "campaign_result_channel": campaign.RESULT_CHANNEL,
        "candidate_channels": candidates,
    }


def _decay_metrics(time: np.ndarray, signal: np.ndarray, burst_end: float) -> dict[str, float]:
    retain = time >= burst_end + 0.15
    t = time[retain]
    y = signal[retain] - float(np.mean(signal[retain]))
    frequency = np.fft.rfftfreq(y.size, d=float(np.median(np.diff(t))))
    power = np.abs(np.fft.rfft(y)) ** 2
    band = (frequency >= 3.0) & (frequency <= 9.0)
    dominant = float(frequency[band][np.argmax(power[band])])
    absolute = np.abs(y)
    peaks = np.where((absolute[1:-1] > absolute[:-2]) & (absolute[1:-1] >= absolute[2:]))[0] + 1
    peaks = peaks[absolute[peaks] > max(float(np.max(absolute)) * 0.02, 1.0e-8)]
    if peaks.size < 8:
        raise VerificationError("Free decay has too few resolved envelope peaks.")
    slope, _intercept = np.polyfit(t[peaks], np.log(absolute[peaks]), 1)
    omega_d = 2.0 * math.pi * dominant
    damping = float(-slope / math.sqrt(omega_d**2 + slope**2))
    return {
        "dominant_frequency_hz": dominant,
        "estimated_damping_ratio": damping,
        "resolved_envelope_peaks": int(peaks.size),
        "initial_decay_amplitude_m": float(np.max(absolute)),
    }


def analyze(root: Path, output: Path | None = None) -> Path:
    root = Path(root).resolve()
    manifest = json.loads((root / "structural_verification_manifest.json").read_text())
    axis = _paired_axis(root, manifest["cases"])
    decays = []
    for row in manifest["cases"]:
        if row["kind"] != "free_decay":
            continue
        result = root / str(row["result_dir"]) / f"{row['result_name']}.txt"
        time, columns = _columns(result)
        metrics = _decay_metrics(time, columns[campaign.RESULT_CHANNEL], float(row["burst_end_s"]))
        target = float(row["target_damping_ratio"])
        metrics.update(
            {
                "mass_damping": row["mass_damping"],
                "target_damping_ratio": target,
                "frequency_pass": abs(metrics["dominant_frequency_hz"] - 6.0) / 6.0 <= 0.05,
                "damping_pass": abs(metrics["estimated_damping_ratio"] - target) <= 0.01,
            }
        )
        decays.append(metrics)
    fractions = np.linspace(0.0, 1.0, campaign.N_STATIONS)
    mode = campaign._first_mode(fractions)
    distributed_mass = campaign._surrogate_distributed_mass_kgpm(fractions)
    generalized_mass = float(distributed_mass * np.trapezoid(mode**2, fractions * campaign.SPAN_M))
    report = {
        "schema": SCHEMA,
        "status": "passed" if axis["passed"] and all(row["frequency_pass"] and row["damping_pass"] for row in decays) else "failed",
        "qualified_design": {
            "chord_m": campaign.CHORD_M,
            "span_m": campaign.SPAN_M,
            "station_count": campaign.N_STATIONS,
            "natural_frequency_hz": campaign.NATURAL_FREQUENCY_HZ,
            "mass_ratio": campaign.MASS_RATIO,
            "mass_damping_bracket": list(campaign.MASS_DAMPINGS),
            "crossflow_bending_axis": "EIy_global_Y",
            "orthogonal_bending_axis": "EIx_global_X",
            "EIx_over_EIy": campaign.ORTHOGONAL_BENDING_STIFFNESS_RATIO,
            "result_channel": campaign.RESULT_CHANNEL,
        },
        "axis_channel_probe": axis,
        "free_decay": decays,
        "generalized_mass_check": {
            "method": "generated_beam_mass_times_tip_normalized_Euler_Bernoulli_mode_quadrature; QBlade result export has no modal-mass channel",
            "generalized_mass_kg": generalized_mass,
            "target_generalized_mass_kg": campaign.PUBLISHED_EFFECTIVE_MASS_KGPM * campaign.SPAN_M,
            "implied_mass_ratio": campaign.MASS_RATIO,
        },
        "orthogonal_mode_design": {
            "EIx_over_EIy": campaign.ORTHOGONAL_BENDING_STIFFNESS_RATIO,
            "nominal_frequency_ratio": math.sqrt(campaign.ORTHOGONAL_BENDING_STIFFNESS_RATIO),
        },
    }
    path = root / "structural_verification_report.json" if output is None else Path(output).resolve()
    atomic_write_json(path, report)
    if report["status"] != "passed":
        raise VerificationError(f"Structural verification failed; inspect {path}.")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--qblade-root", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("analyze")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.campaign, args.output))
    elif args.command == "run":
        run(args.root, args.qblade_root, args.threads)
    else:
        print(analyze(args.root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
