#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-11%8
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-rayleigh
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap_hvawt_decay}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_rayleigh_12case_inputs}"
RESULTS_ROOT="${RESULTS_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_rayleigh_12case_results}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"

[[ -x "${PYTHON_EXE}" ]] || {
  echo "ERROR: Python environment is unavailable: ${PYTHON_EXE}" >&2
  exit 2
}
[[ -d "${QBLADE_ROOT}" ]] || {
  echo "ERROR: QBlade root is unavailable: ${QBLADE_ROOT}" >&2
  exit 2
}
[[ -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]] || {
  echo "ERROR: Generate the path-bound campaign on native Linux before submission: ${CAMPAIGN_ROOT}" >&2
  exit 2
}
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo "ERROR: This entry point must run as the declared Slurm array." >&2
  exit 2
}

cd "${VORLAP_ROOT}"

"${PYTHON_EXE}" -m vorlap.hvawt_modal_decay_campaign validate \
  --campaign-root "${CAMPAIGN_ROOT}"

mapfile -t CASE_IDS < <(
  "${PYTHON_EXE}" -m vorlap.hvawt_modal_decay_campaign list-cases \
    --campaign-root "${CAMPAIGN_ROOT}"
)

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CASE_IDS[@]} )); then
  echo "ERROR: Array index ${SLURM_ARRAY_TASK_ID} is outside 0..$((${#CASE_IDS[@]} - 1))." >&2
  exit 2
fi

CASE_ID="${CASE_IDS[SLURM_ARRAY_TASK_ID]}"
echo "Running ${CASE_ID} as array index ${SLURM_ARRAY_TASK_ID}; one CPU per case, maximum eight concurrent cases."

exec "${PYTHON_EXE}" -m vorlap.hvawt_modal_decay_campaign run-case \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --case-id "${CASE_ID}" \
  --qblade-root "${QBLADE_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --threads 1 \
  --checkpoint-every "${CHECKPOINT_EVERY}"
