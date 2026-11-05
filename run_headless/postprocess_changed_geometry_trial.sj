#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --reservation=flight-cldera
#SBATCH --account=FY260158
#SBATCH --job-name=vorlap-geom-post
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

initial_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_DIR=""
for candidate in \
  "${VORLAP_ROOT:-}/run_headless" \
  "${SLURM_SUBMIT_DIR:-}/run_headless" \
  "${SLURM_SUBMIT_DIR:-}" \
  "${initial_script_dir}" \
  "$(pwd -P)/run_headless" \
  "$(pwd -P)"; do
  [[ -n "${candidate}" ]] || continue
  if [[ -f "${candidate}/submit_parked_demo_campaign.sj" ]]; then
    SCRIPT_DIR="$(cd "${candidate}" && pwd -P)"
    break
  fi
done
[[ -n "${SCRIPT_DIR}" ]] || {
  echo "ERROR: Could not resolve the VorLap run_headless checkout. Submit from the VorLap checkout or export VORLAP_ROOT=/absolute/path/to/VorLap." >&2
  exit 2
}
if [[ -z "${VORLAP_ROOT:-}" || ! -f "${VORLAP_ROOT}/run_headless/submit_parked_demo_campaign.sj" ]]; then
  VORLAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
export VORLAP_ROOT
PARENT_SUBMIT="${SCRIPT_DIR}/submit_parked_demo_campaign.sj"

export VORLAP_SUBMIT_SCRIPT="${VORLAP_SUBMIT_SCRIPT:-${PARENT_SUBMIT}}"
export RUN_ROOT="${RUN_ROOT:-/pscratch/kevmoor/vorlap/run_headless/runs/vorlap_changed_geometry_trial}"
export WIND_SPEEDS="${WIND_SPEEDS:-5 50}"
export QBLADE_AZIMUTHS="${QBLADE_AZIMUTHS:-0}"
export QBLADE_RPM="${QBLADE_RPM:-1}"
export QBLADE_DURATION="${QBLADE_DURATION:-660}"
export QBLADE_STORE_FROM="${QBLADE_STORE_FROM:-0}"
export QBLADE_TIMESTEP="${QBLADE_TIMESTEP:-0.0075}"
export CAMPAIGN_SCENARIOS="${CAMPAIGN_SCENARIOS:-baseline vorlap}"
export POSTPROCESS_MIN_FATIGUE_DURATION="${POSTPROCESS_MIN_FATIGUE_DURATION:-0}"
export POSTPROCESS_INCLUDE_SMOKE_IN_DEL="${POSTPROCESS_INCLUDE_SMOKE_IN_DEL:-1}"
export ONLY_POSTPROCESS=1
export RUN_FINAL_AUDIT="${RUN_FINAL_AUDIT:-0}"
export REQUIRE_PRODUCTION_MATRIX=0
export GENERATE_TURBSIM=0
export AUTO_SETUP_FIRST_RUN="${AUTO_SETUP_FIRST_RUN:-1}"
export AUTO_SETUP_PYTHON="${AUTO_SETUP_PYTHON:-1}"
export SKIP_VORLAP_INSTALL="${SKIP_VORLAP_INSTALL:-1}"
export SKIP_BRIDGE_BUILD=1
export RUN_CHANGED_GEOMETRY_ACCEPTANCE_AUDIT="${RUN_CHANGED_GEOMETRY_ACCEPTANCE_AUDIT:-1}"

echo "Changed-geometry postprocess: run_root=${RUN_ROOT} scenarios='${CAMPAIGN_SCENARIOS}' include_smoke=${POSTPROCESS_INCLUDE_SMOKE_IN_DEL} min_fatigue_duration=${POSTPROCESS_MIN_FATIGUE_DURATION}s."
bash "${PARENT_SUBMIT}"

if [[ "${RUN_CHANGED_GEOMETRY_ACCEPTANCE_AUDIT}" == "1" ]]; then
  if [[ -x "${VORLAP_ROOT}/.venv/bin/python" ]]; then
    TRIAL_AUDIT_PYTHON="${VORLAP_ROOT}/.venv/bin/python"
  else
    TRIAL_AUDIT_PYTHON="${PYTHON_EXE:-python3}"
  fi
  mkdir -p "${RUN_ROOT}/audits"
  identity_audit_flags=()
  normalized_scenarios=" ${CAMPAIGN_SCENARIOS//,/ } "
  if [[ "${normalized_scenarios}" == *" identity "* ]]; then
    identity_audit_flags+=(--require-baseline-identity-proof)
  fi

  # Preserve a broad report even when this deliberately short record trips
  # source/modal/fatigue gates that only close on the later full pilot.
  "${TRIAL_AUDIT_PYTHON}" "${SCRIPT_DIR}/audit_parked_campaign.py" \
    --run-root "${RUN_ROOT}" \
    --stage results \
    --scenarios "${CAMPAIGN_SCENARIOS}" \
    --min-duration 0 \
    --min-analyzed-duration 0 \
    --expected-rpm "${QBLADE_RPM}" \
    --rpm-tolerance "${RPM_AUDIT_TOLERANCE:-0.01}" \
    --require-wind-name \
    --require-airfoil-source-manifest \
    --require-geometry-position-proof \
    --require-postprocess-artifacts \
    "${identity_audit_flags[@]}" \
    --out "${RUN_ROOT}/audits/trial_diagnostic_results_audit.json" \
    --no-fail

  # This focused audit is strict about the acceptance-trial contract while
  # excluding production-only source/modal/ultimate evidence gates.
  echo "Auditing fresh-run checkpoints, completed QPRs, runtime identities, RPM, callbacks, and postprocess products."
  "${TRIAL_AUDIT_PYTHON}" "${SCRIPT_DIR}/audit_parked_campaign.py" \
    --run-root "${RUN_ROOT}" \
    --stage runtime \
    --scenarios "${CAMPAIGN_SCENARIOS}" \
    --min-duration 0 \
    --min-analyzed-duration 0 \
    --expected-rpm "${QBLADE_RPM}" \
    --rpm-tolerance "${RPM_AUDIT_TOLERANCE:-0.01}" \
    --require-wind-name \
    --require-airfoil-source-manifest \
    --require-fresh-run \
    --require-postprocess-artifacts \
    --allow-diagnostic-postprocess-artifacts \
    "${identity_audit_flags[@]}" \
    --out "${RUN_ROOT}/audits/changed_geometry_trial_acceptance.json"
  echo "Changed-geometry acceptance audit passed: ${RUN_ROOT}/audits/changed_geometry_trial_acceptance.json"
fi
