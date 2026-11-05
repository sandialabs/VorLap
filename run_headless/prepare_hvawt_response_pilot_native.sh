#!/bin/bash
# Prepare the path-bound H-VAWT response pilot on the native Linux filesystem.

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
PREP_ROOT="${PREP_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_hybrid_prep_v8}"
PREFLIGHT_CAMPAIGN_ROOT="${PREFLIGHT_CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_hybrid_preflight_120s_v8}"
TWO_SPEED_TEST_CAMPAIGN_ROOT="${TWO_SPEED_TEST_CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_unified_h0_complete_two_speed_120s_test}"
FULL_CAMPAIGN_ROOT="${FULL_CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_hybrid_full_660s_v8}"
PREPARE_FULL_CAMPAIGN="${PREPARE_FULL_CAMPAIGN:-1}"
REQUIRED_FLIGHT_HARDENING_COMMIT="142c186c9e43ec914219a9d0bdf7e554b35902e3"

INPUT_ROOT="${VORLAP_ROOT}/run_headless/config/hvawt_pilot_hpc_inputs"
SOURCE_ARCHIVE="${VORLAP_PAPER_SOURCE_ARCHIVE:-${INPUT_ROOT}/U05p00ms_source_case.tar.gz}"
ARCHIVED_STRMODEL="${INPUT_ROOT}/StrModel_qblade_2.0.4.cpp"
CALIBRATED_STATIC_SPECTRUM="${VORLAP_ROOT}/examples/ReferenceHVAWTQBladeModel/10m_HVAWT_reference_turbine/VorLapAirfoils/NACA0018.h5"
SOURCE_STAGE="${PREP_ROOT}/source"
SOURCE_ROOT="${SOURCE_STAGE}/U05p00ms"
SOURCE_STATIC_SPECTRUM="${SOURCE_ROOT}/10m_HVAWT_reference_turbine/VorLapAirfoils/NACA0018.h5"
RAYLEIGH_CAMPAIGN_ROOT="${PREP_ROOT}/rayleigh_documented_inputs"
RAYLEIGH_SELECTION="${PREP_ROOT}/hvawt_documented_rayleigh_selection.json"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

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

[[ "$(uname -s)" == "Linux" ]] || fail "Preparation must run on native Linux."
case "$(uname -m)" in
  x86_64|amd64) ;;
  *) fail "Preparation requires native x86-64; observed $(uname -m)." ;;
esac
[[ -x "${PYTHON_EXE}" ]] || fail "Python environment is unavailable: ${PYTHON_EXE}"
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root is unavailable: ${QBLADE_ROOT}"
[[ -f "${SOURCE_ARCHIVE}" ]] || fail "H-VAWT source archive is unavailable: ${SOURCE_ARCHIVE}"
[[ -f "${ARCHIVED_STRMODEL}" ]] || fail "Archived parser source is unavailable: ${ARCHIVED_STRMODEL}"
[[ -f "${CALIBRATED_STATIC_SPECTRUM}" ]] || \
  fail "Tracked calibrated NACA0018 source is unavailable: ${CALIBRATED_STATIC_SPECTRUM}"
git -C "${VORLAP_ROOT}" merge-base --is-ancestor \
  "${REQUIRED_FLIGHT_HARDENING_COMMIT}" HEAD || \
  fail "Flight checkout does not contain the runtime-binding hardening commit ${REQUIRED_FLIGHT_HARDENING_COMMIT}."
check_tracked_checkout

observed_source_sha="$(sha256sum "${SOURCE_ARCHIVE}" | awk '{print $1}')"
observed_parser_sha="$(sha256sum "${ARCHIVED_STRMODEL}" | awk '{print $1}')"
observed_spectrum_sha="$(sha256sum "${CALIBRATED_STATIC_SPECTRUM}" | awk '{print $1}')"
[[ "${observed_source_sha}" == "6f6daf91d026087bb5f5acec6b313c7cead1a70faa1deded1819e15c60d7d403" ]] || \
  fail "H-VAWT source archive identity differs from the reviewed input."
[[ "${observed_parser_sha}" == "398bde198b8a3551cdc8469256be2d4b6cd8885ed1e003d93e7464cdf7b025e2" ]] || \
  fail "Archived QBlade parser identity differs from the reviewed input."
[[ "${observed_spectrum_sha}" == "64399e25dc5008caaed88d42e5cf24160c1045f17a282a06234648202cece369" ]] || \
  fail "Tracked NACA0018 identity differs from complete-hybrid schema v8."

[[ ! -e "${PREP_ROOT}" ]] || fail "Preparation root already exists: ${PREP_ROOT}"
[[ ! -e "${PREFLIGHT_CAMPAIGN_ROOT}" ]] || fail "Preflight campaign already exists: ${PREFLIGHT_CAMPAIGN_ROOT}"
[[ ! -e "${TWO_SPEED_TEST_CAMPAIGN_ROOT}" ]] || fail "Two-speed test campaign already exists: ${TWO_SPEED_TEST_CAMPAIGN_ROOT}"
if [[ "${PREPARE_FULL_CAMPAIGN}" == "1" ]]; then
  [[ ! -e "${FULL_CAMPAIGN_ROOT}" ]] || fail "Full campaign already exists: ${FULL_CAMPAIGN_ROOT}"
elif [[ "${PREPARE_FULL_CAMPAIGN}" != "0" ]]; then
  fail "PREPARE_FULL_CAMPAIGN must be 0 or 1."
fi
mkdir -p "${SOURCE_STAGE}"
tar -xzf "${SOURCE_ARCHIVE}" -C "${SOURCE_STAGE}"
[[ -f "${SOURCE_STATIC_SPECTRUM}" ]] || \
  fail "Extracted H-VAWT source lacks NACA0018.h5."
cp "${CALIBRATED_STATIC_SPECTRUM}" "${SOURCE_STATIC_SPECTRUM}"
staged_spectrum_sha="$(sha256sum "${SOURCE_STATIC_SPECTRUM}" | awk '{print $1}')"
[[ "${staged_spectrum_sha}" == "${observed_spectrum_sha}" ]] || \
  fail "Extracted H-VAWT source did not retain the calibrated NACA0018 bytes."

cd "${VORLAP_ROOT}"

"${PYTHON_EXE}" -m vorlap.hvawt_modal_decay_campaign generate \
  --source-root "${SOURCE_ROOT}" \
  --source-contract "${VORLAP_ROOT}/run_headless/config/hvawt_modal_decay_source_contract.json" \
  --output-root "${RAYLEIGH_CAMPAIGN_ROOT}" \
  --action-shapes "${VORLAP_ROOT}/run_headless/config/hvawt_response_mode_shapes.csv" \
  --force-duals "${VORLAP_ROOT}/run_headless/config/hvawt_response_mode_force_duals.csv" \
  --qblade-root "${QBLADE_ROOT}"

"${PYTHON_EXE}" -m vorlap.hvawt_modal_damping_selection \
  --documented-campaign-root "${RAYLEIGH_CAMPAIGN_ROOT}" \
  --archived-strmodel "${ARCHIVED_STRMODEL}" \
  --identification "${VORLAP_ROOT}/run_headless/config/hvawt_response_mode_identification.json" \
  --output "${RAYLEIGH_SELECTION}"

"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py generate \
  --campaign-root "${PREFLIGHT_CAMPAIGN_ROOT}" \
  --rayleigh-selection "${RAYLEIGH_SELECTION}" \
  --campaign-kind two_case_120s_preflight \
  --runtime-vorlap-root "${VORLAP_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py validate \
  --campaign-root "${PREFLIGHT_CAMPAIGN_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py bind-runtime \
  --campaign-root "${PREFLIGHT_CAMPAIGN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py list \
  --campaign-root "${PREFLIGHT_CAMPAIGN_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py generate \
  --campaign-root "${TWO_SPEED_TEST_CAMPAIGN_ROOT}" \
  --rayleigh-selection "${RAYLEIGH_SELECTION}" \
  --campaign-kind two_speed_120s_h0_complete_test \
  --runtime-vorlap-root "${VORLAP_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py validate \
  --campaign-root "${TWO_SPEED_TEST_CAMPAIGN_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py bind-runtime \
  --campaign-root "${TWO_SPEED_TEST_CAMPAIGN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py list \
  --campaign-root "${TWO_SPEED_TEST_CAMPAIGN_ROOT}"

if [[ "${PREPARE_FULL_CAMPAIGN}" == "1" ]]; then
  "${PYTHON_EXE}" run_headless/hvawt_response_pilot.py generate \
    --campaign-root "${FULL_CAMPAIGN_ROOT}" \
    --rayleigh-selection "${RAYLEIGH_SELECTION}" \
    --campaign-kind full_660s_resonance_sweep \
    --runtime-vorlap-root "${VORLAP_ROOT}"

  "${PYTHON_EXE}" run_headless/hvawt_response_pilot.py validate \
    --campaign-root "${FULL_CAMPAIGN_ROOT}"

  "${PYTHON_EXE}" run_headless/hvawt_response_pilot.py bind-runtime \
    --campaign-root "${FULL_CAMPAIGN_ROOT}" \
    --qblade-root "${QBLADE_ROOT}"

  "${PYTHON_EXE}" run_headless/hvawt_response_pilot.py list \
    --campaign-root "${FULL_CAMPAIGN_ROOT}"
fi

echo
echo "Prepared the exact two-case preflight at ${PREFLIGHT_CAMPAIGN_ROOT}."
echo "Prepared the four-case, two-speed matched H0/complete 120 s test at ${TWO_SPEED_TEST_CAMPAIGN_ROOT}."
if [[ "${PREPARE_FULL_CAMPAIGN}" == "1" ]]; then
  echo "Prepared the exact 20-case full campaign at ${FULL_CAMPAIGN_ROOT}."
fi
echo "Run the preflight first:"
echo "sbatch --export=ALL,VORLAP_ROOT=${VORLAP_ROOT},CAMPAIGN_ROOT=${PREFLIGHT_CAMPAIGN_ROOT} ${VORLAP_ROOT}/run_headless/submit_hvawt_response_preflight.sj"
echo "The four-node-array matched H0/complete test is prepared only as a contingency."
echo "Active plan: do not submit ${VORLAP_ROOT}/run_headless/submit_hvawt_two_speed_120s_test.sj."
if [[ "${PREPARE_FULL_CAMPAIGN}" == "1" ]]; then
  echo "After the preflight reports preflight_passed_complete_hybrid_screen, submit the 20-case controlled publication checkpoint:"
  echo "sbatch --export=ALL,VORLAP_ROOT=${VORLAP_ROOT},CAMPAIGN_ROOT=${FULL_CAMPAIGN_ROOT} ${VORLAP_ROOT}/run_headless/submit_hvawt_response_pilot.sj"
  echo "Submit run_headless/submit_hvawt_response_analysis.sj with an afterok dependency on that array job."
fi
