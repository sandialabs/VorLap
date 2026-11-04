#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SBATCH_EXE="${SBATCH_EXE:-sbatch}"
TRIAL_SCRIPT="${SCRIPT_DIR}/submit_changed_geometry_trial.sj"
POSTPROCESS_SCRIPT="${SCRIPT_DIR}/postprocess_changed_geometry_trial.sj"

command -v "${SBATCH_EXE}" >/dev/null 2>&1 || {
  echo "ERROR: sbatch executable not found: ${SBATCH_EXE}" >&2
  exit 2
}
[[ -f "${TRIAL_SCRIPT}" ]] || {
  echo "ERROR: trial submission script not found: ${TRIAL_SCRIPT}" >&2
  exit 2
}
[[ -f "${POSTPROCESS_SCRIPT}" ]] || {
  echo "ERROR: postprocess submission script not found: ${POSTPROCESS_SCRIPT}" >&2
  exit 2
}

trial_submission="$("${SBATCH_EXE}" --parsable "${TRIAL_SCRIPT}")"
trial_job_id="${trial_submission%%;*}"
[[ "${trial_job_id}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: could not parse trial job ID from sbatch output: ${trial_submission}" >&2
  exit 2
}

postprocess_submission="$(
  "${SBATCH_EXE}" \
    --parsable \
    --dependency="afterok:${trial_job_id}" \
    "${POSTPROCESS_SCRIPT}"
)"
postprocess_job_id="${postprocess_submission%%;*}"
[[ "${postprocess_job_id}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: could not parse postprocess job ID from sbatch output: ${postprocess_submission}" >&2
  exit 2
}

echo "Submitted changed-geometry trial array: ${trial_job_id}"
echo "Submitted postprocess job ${postprocess_job_id} with dependency afterok:${trial_job_id}"
echo "Run root: ${RUN_ROOT:-/pscratch/kevmoor/vorlap/run_headless/runs/vorlap_changed_geometry_trial}"
