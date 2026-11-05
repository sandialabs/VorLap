#!/usr/bin/env python3
"""Benchmark the frozen H-VAWT v8 adapter without launching QBlade."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vorlap.performance_timing import AggregatePerformanceTiming
from vorlap.qblade_hvawt_sil import HVAWTSilResponseAdapter
from vorlap.qblade_runtime import VorLapQBladeRuntime
from vorlap.qblade_source_coverage import (
    HVAWT_V8_COMPONENT_NODE_COUNTS,
    validate_complete_source_coverage,
)


SCHEMA = "vorlap-hvawt-v8-offline-benchmark-v1"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def numpy_configuration() -> dict[str, Any]:
    show = getattr(np.__config__, "show", None)
    if show is None:
        return {"numpy_version": np.__version__, "configuration": "unavailable"}
    try:
        configuration = show(mode="dicts")
    except TypeError:
        stream = io.StringIO()
        from contextlib import redirect_stdout

        with redirect_stdout(stream):
            show()
        configuration = stream.getvalue()
    return {"numpy_version": np.__version__, "configuration": configuration}


def load_body_positions(shapes_path: Path, node_ids: tuple[str, ...]) -> np.ndarray:
    positions: dict[str, tuple[float, float, float]] = {}
    with shapes_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["tracked_mode"]) != 6:
                continue
            positions[row["action_node_id"]] = (
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["z_m"]),
            )
    missing = [node_id for node_id in node_ids if node_id not in positions]
    if missing:
        raise ValueError(f"Mode-shape CSV lacks H-VAWT body positions: {missing[:4]}")
    result = np.asarray([positions[node_id] for node_id in node_ids], dtype=float)
    if result.shape != (39, 3) or np.any(~np.isfinite(result)):
        raise ValueError("H-VAWT benchmark body positions are invalid.")
    return result


def rotation_z(azimuth_deg: float) -> np.ndarray:
    angle = math.radians(azimuth_deg)
    return np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    parameter = Path(args.parameter_file).expanduser().resolve()
    model_path = Path(args.model_config).expanduser().resolve()
    if not parameter.is_file() or not model_path.is_file():
        raise FileNotFoundError("Benchmark parameter/model input is unavailable.")
    runtime = VorLapQBladeRuntime.from_qblade_config(str(parameter))
    raw_model = json.loads(model_path.read_text(encoding="utf-8"))
    adapter = HVAWTSilResponseAdapter(
        raw_config=raw_model,
        parameter_file=parameter,
        runtime=runtime,
        sample_period_s=float(args.dt),
    )
    timing = AggregatePerformanceTiming() if args.profile else None
    adapter.performance_timing = timing

    layout = runtime.layout
    basis = adapter.core.config.modal_basis
    control_dir = parameter.parent
    positions_body = load_body_positions(
        control_dir / raw_model["modal_artifacts"]["shapes"],
        basis.node_ids,
    )
    swap = np.zeros(runtime.swap_size, dtype=float)
    time_spec = layout[runtime.time_block]
    timestep_spec = layout[runtime.timestep_block]
    azimuth_spec = layout[runtime.azimuth_block]
    velocity_spec = layout[runtime.velocity_block]
    rpm_spec = layout[raw_model["rpm_block"]]
    position_spec = layout[raw_model["position_block"]]
    force_digest = hashlib.sha256(b"vorlap-hvawt-v8-force-trace-v1\0")
    maxima: list[float] = []

    started = time.perf_counter()
    for index in range(int(args.steps)):
        time_s = index * float(args.dt)
        azimuth = 6.0 * time_s
        rotation = rotation_z(azimuth)
        positions = positions_body @ rotation.T
        rigid = np.cross(
            np.broadcast_to([0.0, 0.0, 2.0 * math.pi / 60.0], positions.shape),
            positions,
        )
        qdot6 = 0.08 + 0.01 * ((index % 7) - 3)
        qdot12 = -0.04 + 0.005 * ((index % 5) - 2)
        mode6 = np.einsum("ab,ib->ia", rotation, basis.real_shape_body_per_m[:, :, 0])
        mode12 = np.einsum("ab,ib->ia", rotation, basis.real_shape_body_per_m[:, :, 1])
        velocity = rigid + qdot6 * mode6 + qdot12 * mode12
        swap[time_spec.offset] = time_s
        swap[timestep_spec.offset] = index
        swap[azimuth_spec.offset] = azimuth
        swap[velocity_spec.offset : velocity_spec.offset + velocity_spec.size] = (
            velocity.reshape(-1)
        )
        swap[rpm_spec.offset] = 1.0
        swap[position_spec.offset : position_spec.offset + position_spec.size] = (
            positions.reshape(-1)
        )
        force = adapter.step(swap)
        coverage_started_ns = timing.now_ns() if timing is not None else 0
        adapter.commit_source_coverage(step_index=index)
        if timing is not None:
            timing.record_since("source_coverage_aggregation", coverage_started_ns)
        canonical_force = np.ascontiguousarray(force, dtype="<f8")
        force_digest.update(canonical_force.tobytes())
        maxima.append(float(np.max(np.linalg.norm(force, axis=1))))
    elapsed = time.perf_counter() - started
    source_coverage = adapter.source_coverage()
    validate_complete_source_coverage(
        source_coverage,
        expected_samples=int(args.steps),
        component_node_counts=HVAWT_V8_COMPONENT_NODE_COUNTS,
    )
    checkpoint = adapter.checkpoint_state()
    checkpoint_bytes = json.dumps(
        checkpoint, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "label": args.label,
        "repository": git_identity(repo),
        "inputs": {
            "parameter_file": str(parameter),
            "parameter_sha256": sha256_file(parameter),
            "model_config": str(model_path),
            "model_config_sha256": sha256_file(model_path),
            "model_fingerprint": adapter.config_fingerprint,
        },
        "runtime": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT
            },
            "numpy": numpy_configuration(),
        },
        "scenario": "complete_v8_residual_plus_engineering_lockin_offline",
        "steps": int(args.steps),
        "dt_seconds": float(args.dt),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / int(args.steps),
        "maximum_nodal_force_n": max(maxima),
        "final_maximum_nodal_force_n": maxima[-1],
        "force_trace_sha256": force_digest.hexdigest(),
        "final_checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "source_coverage": source_coverage,
        "parity": {"tolerance": "exact_binary_equality", "status": "not_compared"},
        "performance_timing": None if timing is None else timing.as_dict(),
    }
    if args.reference_receipt:
        reference = json.loads(
            Path(args.reference_receipt).read_text(encoding="utf-8")
        )
        fields = ("force_trace_sha256", "final_checkpoint_sha256")
        mismatches = [field for field in fields if receipt[field] != reference.get(field)]
        receipt["parity"] = {
            "tolerance": "exact_binary_equality",
            "reference_receipt": str(Path(args.reference_receipt).resolve()),
            "compared_fields": list(fields),
            "mismatches": mismatches,
            "status": "passed" if not mismatches else "failed",
        }
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parameter-file", required=True)
    result.add_argument(
        "--model-config",
        default="run_headless/config/hvawt_response_model_v8_complete.json",
    )
    result.add_argument("--steps", type=int, default=1000)
    result.add_argument("--dt", type=float, default=0.0075)
    result.add_argument("--label", required=True)
    result.add_argument("--profile", action="store_true")
    result.add_argument("--reference-receipt")
    result.add_argument("--output")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.steps < 2 or not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("Benchmark requires at least two steps and positive finite dt.")
    try:
        receipt = benchmark(args)
    except Exception as exc:
        # Preserve an unfavorable or invalid run at the requested output path;
        # performance evidence must not disappear merely because a gate failed.
        receipt = {
            "schema": SCHEMA,
            "label": args.label,
            "scenario": "complete_v8_residual_plus_engineering_lockin_offline",
            "steps": int(args.steps),
            "dt_seconds": float(args.dt),
            "status": "failed",
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "parity": {
                "tolerance": "exact_binary_equality",
                "status": "failed",
            },
        }
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    print(encoded, end="")
    return 0 if receipt["parity"]["status"] != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
