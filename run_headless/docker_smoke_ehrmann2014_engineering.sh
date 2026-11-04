#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${VORLAP_QBLADE_IMAGE:-vorlap-qblade-headless:amd64}"
DOCKER="${DOCKER_EXE:-docker}"

"${DOCKER}" run --rm --platform linux/amd64 --cpus 4 \
  --volume "${ROOT}:/work/VorLap" \
  --workdir /work/VorLap \
  --env QT_QPA_PLATFORM=offscreen \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  "${IMAGE}" bash -lc '
    set -euo pipefail
    export LD_LIBRARY_PATH="/work/VorLap/run_headless/QBladeCE_2.0.9.7:/work/VorLap/run_headless/QBladeCE_2.0.9.7/Libraries:/work/VorLap/run_headless/QBladeCE_2.0.9.7/ControllerFiles:${LD_LIBRARY_PATH:-}"
    campaign_root="$(mktemp -d /tmp/ehrmann-engineering-smoke.XXXXXX)"
    python3 -m examples.ehrmann2014_naca0018.campaign generate \
      --output "${campaign_root}/campaign" \
      --runtime-vorlap-root /work/VorLap \
      --engineering-calibration /work/VorLap/examples/ehrmann2014_naca0018/engineering_lockin_calibration.json
    python3 -m examples.ehrmann2014_naca0018.campaign validate \
      --campaign "${campaign_root}/campaign"
    python3 -m examples.ehrmann2014_naca0018.campaign run \
      --campaign "${campaign_root}/campaign" \
      --qblade-root /work/VorLap/run_headless/QBladeCE_2.0.9.7 \
      --case-index 7 \
      --threads 1 \
      --qblade-steps 2000
    python3 - "${campaign_root}/campaign" <<"PY"
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "campaign_manifest.json").read_text())
row = manifest["cases"][7]
sim_stem = Path(row["sim"]).stem
summary_path = (
    root
    / row["result_dir"]
    / f"{sim_stem}_run_summary.json"
)
summary = json.loads(summary_path.read_text())
response = summary["sil_vorlap_response_diagnostics_summary"]
last = response["last"]
assert manifest["schema"] == "vorlap-ehrmann2014-naca0018-engineering-lockin-campaign-v3"
assert row["case_id"] == "md0p96_u4p50_dt1000us_hybrid"
assert summary["failed"] is False and summary["stopped"] is False
assert summary["completed_steps"] == summary["requested_steps"] == 2000
assert summary["exported_results"] is True
assert response["model_type"] == "single_mode_airfoil_engineering_lockin_v1"
assert response["residual_source_scale"] == 0.0
assert last["captured_active_harmonics_min"] >= 1
assert last["negative_to_structural_damping_ratio"] > 1.0
assert last["linear_lockin_available"] is True
assert last["predicted_equilibrium_amplitude_ratio"] > 0.0
assert last["current_saturation_factor"] < 1.0
assert summary["native_callback_evidence"]["update_count_observed_max"] == 0
print(json.dumps({
    "status": "passed",
    "campaign_fingerprint": manifest["fingerprint"],
    "case_id": row["case_id"],
    "completed_steps": summary["completed_steps"],
    "elapsed_seconds": summary["elapsed_seconds"],
    "captured_harmonics": last["captured_active_harmonics_max"],
    "negative_to_structural_damping_ratio": last["negative_to_structural_damping_ratio"],
    "predicted_equilibrium_amplitude_ratio": last["predicted_equilibrium_amplitude_ratio"],
}, indent=2))
PY
  '
