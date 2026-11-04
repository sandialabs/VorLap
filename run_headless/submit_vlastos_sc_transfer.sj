#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7%2
#SBATCH --time=23:59:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --job-name=vlastos-sc-transfer
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap_hvawt_delivery}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
TRANSFER_ROOT="${TRANSFER_ROOT:-/pscratch/kevmoor/vorlap_runs/vlastos2026_belloli_sc_transfer}"
MANIFEST="${MANIFEST:-${TRANSFER_ROOT}/campaign_manifest.json}"
CALIBRATION_READINESS="${CALIBRATION_READINESS:-/pscratch/kevmoor/vorlap_runs/vlastos2026_response_stationary_v2_linux/analysis/calibration_readiness.json}"
PLAN="${PLAN:-${TRANSFER_ROOT}/execution_plan.json}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

[[ -x "${PYTHON_EXE}" ]] || {
  echo "ERROR: Python environment is unavailable: ${PYTHON_EXE}" >&2
  exit 2
}
[[ -f "${MANIFEST}" ]] || {
  echo "ERROR: Transfer campaign manifest is unavailable: ${MANIFEST}" >&2
  exit 2
}
[[ -f "${CALIBRATION_READINESS}" ]] || {
  echo "ERROR: Passing 24-case calibration readiness is required: ${CALIBRATION_READINESS}" >&2
  exit 2
}
[[ -f "${PLAN}" ]] || {
  echo "ERROR: Generate the canonical transfer execution plan first: ${PLAN}" >&2
  exit 2
}
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo "ERROR: Submit this entry point as the declared 0-7 Slurm array." >&2
  exit 2
}
[[ "${THREADS}" == "4" ]] || {
  echo "ERROR: The frozen plan requires four QBlade threads per task; observed ${THREADS}." >&2
  exit 2
}

PLAN_CONTRACT="$("${PYTHON_EXE}" -c '
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    plan = json.load(stream)
threads = plan.get("threads")
commands = plan.get("commands")
if type(threads) is not int or type(commands) is not list:
    raise SystemExit("malformed plan thread/command contract")
print(f"{threads} {len(commands)}")
' "${PLAN}")" || {
  echo "ERROR: Could not read the canonical plan thread/command contract." >&2
  exit 2
}
[[ "${PLAN_CONTRACT}" == "4 8" ]] || {
  echo "ERROR: The canonical plan must bind exactly four threads and eight commands; observed ${PLAN_CONTRACT}." >&2
  exit 2
}

cd "${VORLAP_ROOT}"
export PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Running transfer-screen index ${SLURM_ARRAY_TASK_ID}; array concurrency is capped at two cases (eight cores total)."

exec "${PYTHON_EXE}" -m examples.vlastos2026_cantilever.engineering_sc_transfer_run run \
  --manifest "${MANIFEST}" \
  --calibration-readiness "${CALIBRATION_READINESS}" \
  --plan "${PLAN}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" \
  --acknowledge-transfer-tuning-screen
