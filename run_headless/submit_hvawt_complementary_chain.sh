#!/bin/bash
# Submit the reviewed H-VAWT authorization, U50 smoke, 82 complementary cases,
# and terminal analysis as one fail-fast dependency chain from a Flight login.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VORLAP_ROOT="${VORLAP_ROOT:-$(cd "${script_dir}/.." && pwd -P)}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3}"
CHECKPOINT_RECEIPT="${CHECKPOINT_RECEIPT:-${VORLAP_ROOT}/paper_drafts/NAWEA_QBladeBridge/generated/hvawt_turbulent_checkpoint_receipt.json}"
COMPLEMENTARY_ARRAY="1,3-8,10,12-17,19,21-26,28,30-35,37,39-44,46,48-53,55,57-62,64,66-80,82,84-89,91,93-101"

fail() { echo "ERROR: $*" >&2; exit 2; }
job_id() {
  local submission="$1"
  local identifier="${submission%%;*}"
  [[ "${identifier}" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response: ${submission}"
  printf '%s' "${identifier}"
}

command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable. Run this on Flight."
[[ -x "${PYTHON_EXE}" ]] || fail "Flight Python environment is unavailable: ${PYTHON_EXE}"
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root is unavailable: ${QBLADE_ROOT}"
[[ -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]] || fail "Sealed campaign is unavailable: ${CAMPAIGN_ROOT}"
[[ -f "${CHECKPOINT_RECEIPT}" ]] || fail "Reviewed checkpoint receipt is unavailable: ${CHECKPOINT_RECEIPT}"

"${PYTHON_EXE}" "${VORLAP_ROOT}/run_headless/hvawt_complementary_dlc_campaign.py" preflight-sources \
  --campaign-root "${CAMPAIGN_ROOT}"

authorization_submission="$(sbatch --parsable \
  --export=ALL,VORLAP_ROOT="${VORLAP_ROOT}",PYTHON_EXE="${PYTHON_EXE}",CAMPAIGN_ROOT="${CAMPAIGN_ROOT}",CHECKPOINT_RECEIPT="${CHECKPOINT_RECEIPT}" \
  "${VORLAP_ROOT}/run_headless/submit_hvawt_complementary_authorization.sj")"
AUTH_JOB="$(job_id "${authorization_submission}")"

u50_submission="$(sbatch --parsable --dependency="afterok:${AUTH_JOB}" \
  --export=ALL,VORLAP_ROOT="${VORLAP_ROOT}",PYTHON_EXE="${PYTHON_EXE}",QBLADE_ROOT="${QBLADE_ROOT}",CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  "${VORLAP_ROOT}/run_headless/submit_hvawt_complementary_u50_smoke.sj")"
U50_JOB="$(job_id "${u50_submission}")"

full_submission="$(sbatch --parsable --dependency="afterok:${U50_JOB}" \
  --array="${COMPLEMENTARY_ARRAY}" \
  --export=ALL,VORLAP_ROOT="${VORLAP_ROOT}",PYTHON_EXE="${PYTHON_EXE}",QBLADE_ROOT="${QBLADE_ROOT}",CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  "${VORLAP_ROOT}/run_headless/submit_hvawt_complementary_dlc_flight.sj")"
FULL_JOB="$(job_id "${full_submission}")"

analysis_submission="$(sbatch --parsable --dependency="afterany:${FULL_JOB}" \
  --export=ALL,VORLAP_ROOT="${VORLAP_ROOT}",PYTHON_EXE="${PYTHON_EXE}",CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  "${VORLAP_ROOT}/run_headless/submit_hvawt_complete_dlc_analysis.sj")"
ANALYSIS_JOB="$(job_id "${analysis_submission}")"

printf 'authorization=%s u50=%s complementary=%s analysis=%s\n' \
  "${AUTH_JOB}" "${U50_JOB}" "${FULL_JOB}" "${ANALYSIS_JOB}"
