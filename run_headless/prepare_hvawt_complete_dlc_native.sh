#!/bin/bash
# Generate the exact 34-row/102-case H-VAWT campaign and bind the Flight runtime
# plus the narrowly scoped turbulent 20-case checkpoint authorization.

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3}"
RAYLEIGH_SELECTION="${RAYLEIGH_SELECTION:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_hybrid_prep_v8/hvawt_documented_rayleigh_selection.json}"
PILOT_CAMPAIGN_ROOT="${PILOT_CAMPAIGN_ROOT:-}"
CHECKPOINT_EVIDENCE="${CHECKPOINT_EVIDENCE:-${VORLAP_ROOT}/paper_drafts/NAWEA_QBladeBridge/analysis/hvawt_uniform_pilot_paper_receipt.json}"
REQUIRED_FLIGHT_HARDENING_COMMIT="142c186c9e43ec914219a9d0bdf7e554b35902e3"

fail() { echo "ERROR: $*" >&2; exit 2; }

is_known_cluster_runtime_artifact() {
  case "$1" in
    run_headless/QBladeCE_*/Binaries/XFoil | \
      run_headless/QBladeCE_*/ControllerFiles/libvorlap_qblade_bridge.so)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

check_tracked_checkout() {
  local tracked_status disallowed_status status_line tracked_path
  tracked_status="$(git -C "${VORLAP_ROOT}" status --porcelain --untracked-files=no)" ||
    fail "Could not inspect the tracked Flight checkout."
  disallowed_status=""
  while IFS= read -r status_line; do
    [[ -n "${status_line}" ]] || continue
    tracked_path="${status_line:3}"
    if ! is_known_cluster_runtime_artifact "${tracked_path}"; then
      disallowed_status+="${status_line}"$'\n'
    fi
  done <<<"${tracked_status}"
  if [[ -n "${disallowed_status}" ]]; then
    printf 'Disallowed tracked changes:\n%s' "${disallowed_status}" >&2
    fail "Tracked Flight checkout contains changes outside the two known cluster-built runtime artifacts."
  fi
  if [[ -n "${tracked_status}" ]]; then
    printf 'WARNING: Allowing known cluster-built tracked runtime artifacts; their actual bytes remain hash-bound:\n%s\n' "${tracked_status}" >&2
  fi
}

if ! type module >/dev/null 2>&1; then
  for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash /opt/cray/pe/modules/default/init/bash; do
    if [[ -r "${init_script}" ]]; then source "${init_script}"; break; fi
  done
fi
type module >/dev/null 2>&1 || fail "Flight module command is unavailable."
module load anaconda3/2025.12-2

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || fail "Prepare this native campaign on Flight Linux x86-64."
[[ -x "${PYTHON_EXE}" ]] || fail "Successful Flight Python environment is unavailable: ${PYTHON_EXE}"
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root is unavailable: ${QBLADE_ROOT}"
[[ -f "${RAYLEIGH_SELECTION}" ]] || fail "Rayleigh selection is unavailable: ${RAYLEIGH_SELECTION}"
[[ -f "${CHECKPOINT_EVIDENCE}" ]] || fail "Reviewed uniform checkpoint evidence is unavailable: ${CHECKPOINT_EVIDENCE}"
[[ -f "${VORLAP_ROOT}/run_headless/config/parked_dlc_matrix_approved.json" ]] || fail "Approved 34-row manifest is unavailable."
git -C "${VORLAP_ROOT}" merge-base --is-ancestor \
  "${REQUIRED_FLIGHT_HARDENING_COMMIT}" HEAD || \
  fail "Flight checkout does not contain the runtime-binding hardening commit ${REQUIRED_FLIGHT_HARDENING_COMMIT}."

check_tracked_checkout

export PYTHONPATH="${VORLAP_ROOT}:${QBLADE_ROOT}/SIL_Interface${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${QBLADE_ROOT}:${QBLADE_ROOT}/Libraries:${QBLADE_ROOT}/ControllerFiles:${QBLADE_ROOT}/Binaries:${QBLADE_ROOT}/Binaries/IPCBridge${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VORLAP_QBLADE_LOADER_ENV=1
cd "${VORLAP_ROOT}"

if [[ -e "${CAMPAIGN_ROOT}" ]]; then
  [[ -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]] || fail "Campaign root exists without a sealed manifest: ${CAMPAIGN_ROOT}"
  "${PYTHON_EXE}" run_headless/hvawt_complete_dlc_campaign.py validate \
    --campaign-root "${CAMPAIGN_ROOT}"
else
  "${PYTHON_EXE}" run_headless/hvawt_complete_dlc_campaign.py generate \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --rayleigh-selection "${RAYLEIGH_SELECTION}" \
    --runtime-vorlap-root "${VORLAP_ROOT}"
fi

"${PYTHON_EXE}" run_headless/hvawt_complete_dlc_campaign.py bind-runtime \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_complete_dlc_campaign.py bind-checkpoint-authorization \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --evidence "${CHECKPOINT_EVIDENCE}"

if [[ -n "${PILOT_CAMPAIGN_ROOT}" ]]; then
  "${PYTHON_EXE}" run_headless/hvawt_complete_dlc_campaign.py bind-authorization \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --pilot-campaign-root "${PILOT_CAMPAIGN_ROOT}"
else
  echo "The exact turbulent 20-case raw/complete checkpoint is authorized."
  echo "The remaining/full 102-case execution remains separately gated."
fi

echo "Prepared, runtime-bound, and checkpoint-authorized ${CAMPAIGN_ROOT}."
