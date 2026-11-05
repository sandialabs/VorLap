#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --reservation=flight-cldera
#SBATCH --account=FY260158
#SBATCH --job-name=vorlap-geom-trial
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

# Two physical rows are submitted as separate array tasks so baseline and
# VorLap can remain sequential within each node while fitting the short queue.
# Identity is now an explicit opt-in diagnostic because the zero-force gate is
# closed for the current bridge/runtime. This remains an initialization trial,
# not fatigue evidence.

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
  echo "ERROR: Could not resolve the VorLap run_headless checkout from the Slurm spool copy. Submit from the VorLap checkout or export VORLAP_ROOT=/absolute/path/to/VorLap." >&2
  exit 2
}
if [[ -z "${VORLAP_ROOT:-}" || ! -f "${VORLAP_ROOT}/run_headless/submit_parked_demo_campaign.sj" ]]; then
  VORLAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
export VORLAP_ROOT
PARENT_SUBMIT="${SCRIPT_DIR}/submit_parked_demo_campaign.sj"

export VORLAP_SUBMIT_SCRIPT="${VORLAP_SUBMIT_SCRIPT:-${PARENT_SUBMIT}}"
export WIND_SPEEDS="${WIND_SPEEDS:-5 50}"
export PRODUCTION_WIND_SPEEDS="${PRODUCTION_WIND_SPEEDS:-${WIND_SPEEDS}}"
export QBLADE_AZIMUTHS="${QBLADE_AZIMUTHS:-0}"
export PRODUCTION_AZIMUTHS="${PRODUCTION_AZIMUTHS:-${QBLADE_AZIMUTHS}}"
export QBLADE_RPM="${QBLADE_RPM:-1}"
export PRODUCTION_RPM="${PRODUCTION_RPM:-${QBLADE_RPM}}"
export QBLADE_HORIZONTAL_INFLOW_ANGLE="${QBLADE_HORIZONTAL_INFLOW_ANGLE:-0.0}"
export QBLADE_VERTICAL_INFLOW_ANGLE="${QBLADE_VERTICAL_INFLOW_ANGLE:-0.0}"
export QBLADE_SHEAR_EXPONENT="${QBLADE_SHEAR_EXPONENT:-0.23}"
export QBLADE_REFERENCE_HEIGHT="${QBLADE_REFERENCE_HEIGHT:-18.0}"
export QBLADE_AIR_DENSITY="${QBLADE_AIR_DENSITY:-1.0296}"
export QBLADE_AIR_KINEMATIC_VISCOSITY="${QBLADE_AIR_KINEMATIC_VISCOSITY:-1.647e-5}"
export TURBSIM_REF_HEIGHT="${TURBSIM_REF_HEIGHT:-${QBLADE_REFERENCE_HEIGHT}}"
export QBLADE_DURATION="${QBLADE_DURATION:-660}"
export QBLADE_STORE_FROM="${QBLADE_STORE_FROM:-0}"
export QBLADE_TIMESTEP="${QBLADE_TIMESTEP:-0.0075}"
export QBLADE_STEPS="${QBLADE_STEPS:-500}"
export CAMPAIGN_SCENARIOS="${CAMPAIGN_SCENARIOS:-baseline vorlap}"
export CASE_WORKERS="${CASE_WORKERS:-1}"
export OMP_NUM_THREADS="${TRIAL_OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export QBLADE_CHECKPOINT_EVERY="${QBLADE_CHECKPOINT_EVERY:-250}"
# Replay is useful for this short geometry/debug trial, but remains opt-in in
# the production campaign entry points because of its storage cost.
export QBLADE_STORE_REPLAY="${QBLADE_STORE_REPLAY:-true}"
export QBLADE_STORE_PROJECT="${QBLADE_STORE_PROJECT:-true}"
export GENERATE_TURBSIM="${GENERATE_TURBSIM:-1}"
export AUTO_SETUP_FIRST_RUN="${AUTO_SETUP_FIRST_RUN:-1}"
export AUTO_SETUP_PYTHON="${AUTO_SETUP_PYTHON:-1}"
export AUTO_SETUP_TURBSIM="${AUTO_SETUP_TURBSIM:-1}"
export FORCE_GENERATE_TURBSIM="${FORCE_GENERATE_TURBSIM:-0}"
export FORCE_REINSTALL_SETUP="${FORCE_REINSTALL_SETUP:-0}"
export FORCE_CASE_SETUP="${FORCE_CASE_SETUP:-1}"
export VORLAP_FORCE_SETUP_TOKEN="${VORLAP_FORCE_SETUP_TOKEN:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}}"
if [[ "${FORCE_REINSTALL_SETUP}" == "1" ]]; then
  export FORCE_PYTHON_SETUP=1
  export FORCE_OPENFAST_TURBSIM_BUILD=1
  export FORCE_BRIDGE_BUILD=1
  export FORCE_CASE_SETUP=1
fi
if [[ "${FORCE_CASE_SETUP}" == "1" ]]; then
  export VORLAP_SERIALIZE_FORCE_SETUP="${VORLAP_SERIALIZE_FORCE_SETUP:-1}"
fi
export TURBSIM_WIND_ROOT="${TURBSIM_WIND_ROOT:-/pscratch/kevmoor/vorlap_wind_fields/changed_geometry_trial}"
export VORLAP_SOURCE_BOUNDS_POLICY="${VORLAP_SOURCE_BOUNDS_POLICY:-warn}"
export REQUIRE_PRODUCTION_MATRIX="${REQUIRE_PRODUCTION_MATRIX:-0}"
export RUN_PAPER_PREFLIGHT="${RUN_PAPER_PREFLIGHT:-0}"
export RUN_FINAL_AUDIT="${RUN_FINAL_AUDIT:-0}"
export SKIP_VORLAP_INSTALL="${SKIP_VORLAP_INSTALL:-1}"
export SKIP_BRIDGE_BUILD="${SKIP_BRIDGE_BUILD:-0}"
export RUN_ROOT="${RUN_ROOT:-/pscratch/kevmoor/vorlap/run_headless/runs/vorlap_changed_geometry_trial}"

echo "Changed-geometry trial: task=${SLURM_ARRAY_TASK_ID:-manual} winds='${WIND_SPEEDS}' scenarios='${CAMPAIGN_SCENARIOS}' steps=${QBLADE_STEPS} duration=${QBLADE_DURATION}s store_from=${QBLADE_STORE_FROM}s replay=${QBLADE_STORE_REPLAY} store_project=${QBLADE_STORE_PROJECT} checkpoint_every=${QBLADE_CHECKPOINT_EVERY} TurbSim=${GENERATE_TURBSIM} first_run_setup=${AUTO_SETUP_FIRST_RUN} force_reinstall=${FORCE_REINSTALL_SETUP} force_case_setup=${FORCE_CASE_SETUP} force_wind_regeneration=${FORCE_GENERATE_TURBSIM}."
exec bash "${PARENT_SUBMIT}"
