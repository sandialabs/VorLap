#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --array=0-2%3
#SBATCH --time=47:59:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=vorlap-dlc-u03-recover
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

if [[ -n "${VORLAP_ROOT:-}" && -f "${VORLAP_ROOT}/run_headless/submit_parked_dlc_manifest.sj" ]]; then
  VORLAP_ROOT="$(cd "${VORLAP_ROOT}" && pwd -P)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_headless/submit_parked_dlc_manifest.sj" ]]; then
  VORLAP_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  VORLAP_ROOT="/pscratch/kevmoor/vorlap"
fi

PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
DLC_MANIFEST="${DLC_MANIFEST:-${VORLAP_ROOT}/run_headless/config/parked_dlc_matrix_approved.json}"

[[ -x "${PYTHON_EXE}" ]] || {
  echo "ERROR: Python environment is unavailable: ${PYTHON_EXE}" >&2
  exit 2
}
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo "ERROR: U03 recovery must run as the configured Slurm array." >&2
  exit 2
}

"${PYTHON_EXE}" "${VORLAP_ROOT}/run_headless/verify_u03_recovery.py" \
  --manifest "${DLC_MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --vorlap-root "${VORLAP_ROOT}"

export VORLAP_ROOT PYTHON_EXE DLC_MANIFEST
export DLC_PHASE=qblade
export RUN_PAPER_PREFLIGHT=1
export PRESERVE_COMPLETED_SCENARIOS=1

exec bash "${VORLAP_ROOT}/run_headless/submit_parked_dlc_manifest.sj"
