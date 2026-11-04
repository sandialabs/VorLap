#!/bin/bash
# Operator entry point, following the recent Flight Git/Slurm workflow.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VORLAP_ROOT="$(cd "${script_dir}/.." && pwd -P)"
TASK_ROOT="${VORLAP_ROOT}/examples/fixed_azimuth_20260922"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
SOURCE_CAMPAIGN="${SOURCE_CAMPAIGN:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/u11p25_a147_fixed_$(date -u +%Y%m%dT%H%M%SZ)}"
FIXED_AZIMUTH_RESERVATION="${FIXED_AZIMUTH_RESERVATION-flight-cldera}"
fail() { echo "ERROR: $*" >&2; exit 2; }

for fixed_path in "${PYTHON_EXE}" "${QBLADE_ROOT}" "${SOURCE_CAMPAIGN}" "${CAMPAIGN_ROOT}"; do
  [[ "${fixed_path}" == /* ]] || fail "Use absolute paths for Flight overrides: ${fixed_path}"
done
command -v sbatch >/dev/null 2>&1 || fail "sbatch is unavailable. Run this on Flight."
[[ ! -e "${CAMPAIGN_ROOT}" ]] || fail "Use a new CAMPAIGN_ROOT: ${CAMPAIGN_ROOT}"
[[ "$(git -C "${VORLAP_ROOT}" branch --show-current)" == response-aware-viv-fatigue ]] || fail "Use branch response-aware-viv-fatigue."
git -C "${VORLAP_ROOT}" diff --quiet HEAD -- \
  vorlap run_headless/config run_headless/run_qblade_sim.py \
  run_headless/submit_fixed_azimuth_chain.sh examples/fixed_azimuth_20260922 \
  run_headless/flight_environment.sh || fail "Commit or preserve the local runtime/submission edits before submitting."
FLIGHT_SOURCE_COMMIT="$(git -C "${VORLAP_ROOT}" rev-parse HEAD)"

# Load the shared Flight Python environment for the prescribed parked cases.
source "${VORLAP_ROOT}/run_headless/flight_environment.sh"
[[ -x "${PYTHON_EXE}" ]] || fail "Flight Python environment is unavailable: ${PYTHON_EXE}"
qblade_library="${QBLADE_ROOT}/libQBladeCE_2.0.9.7.so.1.0.0"
[[ -f "${qblade_library}" ]] || fail "QBlade library is unavailable: ${qblade_library}"
[[ "$(sha256sum "${qblade_library}" | awk '{print $1}')" == 7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d ]] || fail "QBlade differs from the accepted Flight runtime."

export VORLAP_ROOT TASK_ROOT PYTHON_EXE QBLADE_ROOT SOURCE_CAMPAIGN CAMPAIGN_ROOT FLIGHT_SOURCE_COMMIT
cd "${VORLAP_ROOT}"
prepare_receipt="$(mktemp)"
trap 'rm -f "${prepare_receipt}"' EXIT
"${PYTHON_EXE}" "${TASK_ROOT}/campaign.py" prepare \
  --source-campaign "${SOURCE_CAMPAIGN}" --campaign-root "${CAMPAIGN_ROOT}" \
  --azimuth-deg 147 > "${prepare_receipt}"
mv "${prepare_receipt}" "${CAMPAIGN_ROOT}/preparation.json"
"${PYTHON_EXE}" "${TASK_ROOT}/campaign.py" validate \
  --campaign-root "${CAMPAIGN_ROOT}" > "${CAMPAIGN_ROOT}/validation.json"
mkdir -p "${CAMPAIGN_ROOT}/logs"
printf '%s\n' "${FLIGHT_SOURCE_COMMIT}" > "${CAMPAIGN_ROOT}/source_git_commit.txt"

options=(--parsable --export=ALL)
if [[ -n "${FIXED_AZIMUTH_RESERVATION}" ]]; then options+=(--reservation="${FIXED_AZIMUTH_RESERVATION}"); fi
array_reply="$(sbatch "${options[@]}" \
  --output="${CAMPAIGN_ROOT}/logs/simulation-%A_%a.out" \
  --error="${CAMPAIGN_ROOT}/logs/simulation-%A_%a.err" \
  "${TASK_ROOT}/submit_flight.sj")"
array_id="${array_reply%%;*}"
[[ "${array_id}" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch result: ${array_reply}"
printf 'array_job_id=%s\ncampaign_root=%s\nsource_commit=%s\nlogs=%s/logs/\n' \
  "${array_id}" "${CAMPAIGN_ROOT}" "${FLIGHT_SOURCE_COMMIT}" "${CAMPAIGN_ROOT}" \
  | tee "${CAMPAIGN_ROOT}/submission.txt"
