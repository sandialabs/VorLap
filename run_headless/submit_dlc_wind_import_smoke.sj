#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --reservation=flight-cldera
#SBATCH --account=FY260158
#SBATCH --job-name=vorlap-wind-smoke
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

# This proves TurbSim generation plus QBlade WNDTYPE=1/WNDNAME import for the
# changed geometry. It is intentionally too short for fatigue or spectra.

original_task="${SLURM_ARRAY_TASK_ID:-}"
[[ "${original_task}" =~ ^[01]$ ]] || {
  echo "ERROR: submit as the declared 0-1 Slurm array or set SLURM_ARRAY_TASK_ID to 0 or 1 for a dry run." >&2
  exit 2
}
wind_speeds=(5 50)
speed_labels=(U05p00ms U50p00ms)
wind_speed="${wind_speeds[original_task]}"
speed_label="${speed_labels[original_task]}"
seed=$((6633456 + original_task))

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
smoke_root="${WIND_SMOKE_ROOT:-/pscratch/kevmoor/vorlap/run_headless/runs/wind_import_smoke}"
wind_root="${WIND_SMOKE_FIELD_ROOT:-/pscratch/kevmoor/vorlap_wind_fields/import_smoke}"

export VORLAP_SUBMIT_SCRIPT="${PARENT_SUBMIT}"
export RUN_ROOT="${RUN_ROOT:-${smoke_root}/${speed_label}_s${seed}}"
export WIND_SPEEDS="${wind_speed}"
export PRODUCTION_WIND_SPEEDS="${wind_speed}"
export QBLADE_AZIMUTHS="0"
export PRODUCTION_AZIMUTHS="0"
export QBLADE_RPM="1"
export PRODUCTION_RPM="1"
export QBLADE_DURATION="${QBLADE_DURATION:-10}"
export QBLADE_STORE_FROM="0"
export QBLADE_TIMESTEP="${QBLADE_TIMESTEP:-0.0075}"
export QBLADE_STEPS="${QBLADE_STEPS:-20}"
export CAMPAIGN_SCENARIOS="baseline"
export CASE_WORKERS="1"
export OMP_NUM_THREADS="${WIND_SMOKE_OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
export QBLADE_CHECKPOINT_EVERY="0"
export QBLADE_STORE_REPLAY="false"
export GENERATE_TURBSIM="1"
export FORCE_GENERATE_TURBSIM="${FORCE_GENERATE_TURBSIM:-0}"
export AUTO_SETUP_TURBSIM="${AUTO_SETUP_TURBSIM:-0}"
export TURBSIM_WIND_ROOT="${wind_root}"
export TURBSIM_ANALYSIS_TIME="${TURBSIM_ANALYSIS_TIME:-10}"
export TURBSIM_TIMESTEP="${TURBSIM_TIMESTEP:-${QBLADE_TIMESTEP}}"
export TURBSIM_SEED_BASE="${seed}"
export TURBSIM_IEC_WIND_TYPE="${TURBSIM_IEC_WIND_TYPE:-NTM}"
export QBLADE_WIND_NAME="${wind_root}/${speed_label}/wind_import_${speed_label}_s${seed}.bts"
export REQUIRE_WIND_NAME="1"
export REQUIRE_PRODUCTION_MATRIX="0"
export AUDIT_MIN_ANALYZED_DURATION="0"
export RUN_PAPER_PREFLIGHT="0"
export RUN_FINAL_AUDIT="0"
export SKIP_POSTPROCESS="1"
export SKIP_VORLAP_INSTALL="${SKIP_VORLAP_INSTALL:-1}"
export SKIP_BRIDGE_BUILD="${SKIP_BRIDGE_BUILD:-1}"

# The parent must see a one-row campaign, not reinterpret the outer array index.
unset SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX

echo "Wind-import smoke: U=${wind_speed} m/s seed=${seed} wind=${QBLADE_WIND_NAME}"
exec bash "${PARENT_SUBMIT}"
