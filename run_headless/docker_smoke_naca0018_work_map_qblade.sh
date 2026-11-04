#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${VORLAP_QBLADE_IMAGE:-vorlap-qblade-headless:amd64}"
PLATFORM="${VORLAP_QBLADE_PLATFORM:-linux/amd64}"

exec /opt/homebrew/bin/docker run --rm \
  --platform "${PLATFORM}" \
  --cpus 4 \
  --volume "${REPO_ROOT}:/work/VorLap" \
  --workdir /work/VorLap \
  --env QT_QPA_PLATFORM=offscreen \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    qblade_root=/work/VorLap/run_headless/QBladeCE_2.0.9.7
    smoke_root=/tmp/naca0018-work-map-qblade-smoke
    case_id=re140000_k0p45_a0p0500
    case_root=${smoke_root}/cases/${case_id}
    result_root=${smoke_root}/smoke_results
    export LD_LIBRARY_PATH="${qblade_root}:${qblade_root}/Libraries:${qblade_root}/ControllerFiles:${LD_LIBRARY_PATH:-}"
    python3 -m examples.ehrmann2014_naca0018.qblade_prescribed_motion generate \
      --campaign "${smoke_root}" \
      --qblade-root "${qblade_root}"
    python3 run_headless/run_qblade_sim.py \
      --sim "${case_root}/${case_id}.sim" \
      --qblade-root "${qblade_root}" \
      --output-dir "${result_root}" \
      --result-name smoke \
      --result-filter "${case_root}/results.filter" \
      --steps 2 \
      --threads 1 \
      --coupling-mode none \
      --case-metadata "${case_root}/case_metadata.json" \
      --progress-every 0
    python3 - "${smoke_root}" <<'"'"'PY'"'"'
import json
import math
from pathlib import Path
import sys

import numpy as np

from examples.ehrmann2014_naca0018 import qblade_prescribed_motion as qblade

root = Path(sys.argv[1])
manifest = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
case = manifest["cases"][qblade.PREFLIGHT_CASE_INDEX]
result = root / "smoke_results/smoke.txt"
summary_path = root / "smoke_results/re140000_k0p45_a0p0500_run_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
callback = summary.get("native_callback_evidence", {})
library = summary.get("reproducibility", {}).get("qblade", {}).get("library", {})
assert summary["failed"] is False and summary["stopped"] is False
assert summary["exported_results"] is True
assert summary["requested_steps"] == summary["completed_steps"] == 2
assert summary["threads"] == 1 and summary["coupling_mode"] == "none"
assert summary["sil_vorlap"] is False and summary["sil_constant_force"] is False
assert callback.get("callback_block_present") is False
assert callback.get("update_init_count", 0) == 0
assert callback.get("update_count_observed_max", 0) == 0
assert callback.get("runtime_initialized_message_seen") is False
assert library.get("sha256") == qblade.EXPECTED_QBLADE_LIBRARY_SHA256
columns = qblade._normalized_columns(result)
assert set(qblade.FILTER_CHANNELS).issubset(columns)
assert all(len(columns[name]) == 2 for name in qblade.FILTER_CHANNELS)
assert all(np.all(np.isfinite(columns[name])) for name in qblade.FILTER_CHANNELS)
assert abs(float(columns[qblade.TIME_CHANNEL][0])) <= 2.0e-11
columns = {name: values[1:] for name, values in columns.items()}
assert all(len(columns[name]) == 1 for name in qblade.FILTER_CHANNELS)
dt = float(case["time_step_s"])
times = np.asarray([dt])
q = float(case["amplitude_m"]) * np.sin(
    2.0 * math.pi * float(case["frequency_hz"]) * times
)
platform = columns[qblade.PLATFORM_Y_CHANNEL]
tip = columns[qblade.BLADE_TIP_Y_CHANNEL]
time_tolerance = np.maximum(2.0e-11, 6.0e-6 * np.maximum(times, dt))
assert np.all(np.abs(columns[qblade.TIME_CHANNEL] - times) <= time_tolerance)
assert np.max(np.abs(platform - q)) <= max(2.0e-9, 2.0e-5 * case["amplitude_m"])
assert np.max(np.abs((tip - tip[0]) - (platform - platform[0]))) <= 2.0e-6
assert np.max(np.abs(columns["LSS Rpm [rpm]"])) <= 1.0e-8
assert np.max(np.abs(columns["True Pitch Angle BLD_1 [deg]"])) <= 1.0e-8
assert np.max(np.abs(columns["True Yaw Angle [deg]"])) <= 1.0e-8
aoa = columns[qblade.AOA_CHANNEL]
reynolds = columns[qblade.REYNOLDS_CHANNEL]
assert abs(float(np.median(np.abs(aoa))) - 90.0) <= 12.0
assert float(np.max(np.abs(np.abs(aoa) - 90.0))) <= 12.0
assert abs(float(np.median(reynolds)) - case["reynolds_number"]) / case[
    "reynolds_number"
] <= 0.03
assert np.all(np.isfinite(columns[qblade.FORCE_CHANNEL]))
print(
    json.dumps(
        {
            "status": "complete_two_step_software_smoke_only",
            "case_index": qblade.PREFLIGHT_CASE_INDEX,
            "case_id": qblade.PREFLIGHT_CASE_ID,
            "qblade_library_sha256": library["sha256"],
            "qblade_aoa_deg": aoa.tolist(),
            "vorlap_declared_incidence_deg": qblade.EXPECTED_VORLAP_INCIDENCE_DEG,
            "panel_reynolds_number": reynolds.tolist(),
            "force_channel": qblade.FORCE_CHANNEL,
            "force_n": columns[qblade.FORCE_CHANNEL].tolist(),
            "callback_updates": 0,
        },
        indent=2,
    )
)
PY
  '
