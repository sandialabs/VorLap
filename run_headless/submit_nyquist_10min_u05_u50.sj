#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=47:59:00
#SBATCH --account=FY260158
#SBATCH --job-name=vorlap-10m-rw2
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --array=0-1%2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

# Self-contained U05/U50, 11-minute simulated/10-minute retained reduced-wake pilot.
# Submit with:
#   sbatch run_headless/submit_nyquist_10min_u05_u50.sj
#
# Each array task owns one exclusive node and one wind speed. Baseline and
# VorLap run sequentially within that task so one QBlade process can use all
# CPUs reported by Slurm without oversubscription.

export VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
export QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
export TURBSIM_WIND_ROOT="${TURBSIM_WIND_ROOT:-/pscratch/kevmoor/vorlap_wind_fields/changed_geometry_trial}"
export VORLAP_PROFILE_RUN_ROOT="${VORLAP_PROFILE_RUN_ROOT:-${VORLAP_ROOT}/run_headless/runs/vorlap_nyquist_10min_U05_U50_reduced_wake_v3_filtered_20260803}"
export VORLAP_SUBMIT_SCRIPT="${VORLAP_ROOT}/run_headless/submit_parked_demo_campaign.sj"
export QBLADE_RESULT_FILTER="${QBLADE_RESULT_FILTER:-${VORLAP_ROOT}/run_headless/config/parked_fatigue_results.filter}"

[[ -f "${VORLAP_SUBMIT_SCRIPT}" ]] || {
  echo "ERROR: production campaign script not found: ${VORLAP_SUBMIT_SCRIPT}" >&2
  exit 2
}
[[ -f "${QBLADE_RESULT_FILTER}" ]] || {
  echo "ERROR: QBlade fatigue result filter not found: ${QBLADE_RESULT_FILTER}" >&2
  exit 2
}

profile_array_task_id="${SLURM_ARRAY_TASK_ID:-}"
case "${profile_array_task_id}" in
  0)
    profile_wind_speed="5"
    profile_wind_label="U05p00ms"
    profile_turbsim_seed="4433456"
    ;;
  1)
    profile_wind_speed="50"
    profile_wind_label="U50p00ms"
    profile_turbsim_seed="4433457"
    ;;
  *)
    echo "ERROR: expected Slurm array task 0 or 1; got '${profile_array_task_id:-unset}'." >&2
    exit 2
    ;;
esac

export RUN_ROOT="${VORLAP_PROFILE_RUN_ROOT}/${profile_wind_label}"
export WIND_SPEEDS="${profile_wind_speed}"
export PRODUCTION_WIND_SPEEDS="${profile_wind_speed}"
# The original fields were generated as a two-speed campaign. Preserve their
# case-indexed seeds even though each Slurm array task now runs one speed.
export TURBSIM_SEED_BASE="${profile_turbsim_seed}"
export QBLADE_AZIMUTHS="0"
export PRODUCTION_AZIMUTHS="0"
export QBLADE_RPM="1"
export PRODUCTION_RPM="1"
export QBLADE_DURATION="660"
export QBLADE_STORE_FROM="60"
# Keep one absolute-frequency sampling basis across wind speeds. The structural
# modal bands do not scale with wind speed; higher shed tones are removed by
# the common Nyquist policy rather than changing dt between paired cases.
export QBLADE_TIMESTEP="0.0075"
export POSTPROCESS_MIN_FATIGUE_DURATION="600"

export CAMPAIGN_SCENARIOS="baseline vorlap"
export VORLAP_QBLADE_COUPLING="sil-vorlap"
export VORLAP_FORCE_SCALE="1.0"
export VORLAP_SOURCE_FREQUENCY_TRUNCATION="nyquist"
export VORLAP_SOURCE_BOUNDS_POLICY="warn"
# Generated-case-only: 10D cutoff, Oye TF=8, bound/free core radii 1.0c/0.05c,
# and vortex viscosity 80, while preserving EF, panels, zones, 120k cap, and
# the 0.005 circulation threshold.
export QBLADE_WAKE_PROFILE="parked-reduced-v2"

export QBLADE_STORE_REPLAY="false"
export QBLADE_STORE_PROJECT="false"
export QBLADE_CHECKPOINT_EVERY="4000"
export CASE_WORKERS="1"
export VORLAP_AUTO_NODE_THREADS="1"
export QBLADE_USE_OPENCL="false"

export GENERATE_TURBSIM="1"
export ADOPT_EXISTING_TURBSIM="1"
export FORCE_GENERATE_TURBSIM="0"
export AUTO_SETUP_FIRST_RUN="0"
export AUTO_SETUP_TURBSIM="1"
export FORCE_PYTHON_SETUP="0"
export FORCE_OPENFAST_TURBSIM_BUILD="0"
export FORCE_BRIDGE_BUILD="0"
export FORCE_CASE_SETUP="0"
export SKIP_VORLAP_INSTALL="1"
export SKIP_BRIDGE_BUILD="1"
# Create missing cases on the first submission, then validate/reuse them so a
# repeated sbatch can resume compatible QPR checkpoints instead of replacing
# the generated case assets and invalidating the restart lineage.
# VorLap checkpoints created before the SIL runtime-filter fix are deliberately
# incompatible: setup is regenerated once and those VorLap cases restart at
# step zero. Completed baseline exports remain reusable.
export REUSE_SETUP="1"

export REQUIRE_PRODUCTION_MATRIX="1"
export REQUIRE_SOURCE_COVERAGE="1"
export REQUIRE_POSTPROCESS_ARTIFACTS="1"
export RUN_SIGN_PROBES="0"
export REQUIRE_FORCE_SIGN_PROBES="0"
export RUN_FINAL_AUDIT="1"
export RUN_PAPER_PREFLIGHT="1"
export SUBMIT_WIND_JOBS="0"
export SUBMIT_WIND_JOB_TIME="47:59:00"

export VORLAP_SLURM_MODULES="anaconda3/2025.12-2"
export SKIP_VORLAP_MODULE_LOAD="0"

# A stale smoke limit in the submission environment must not shorten this run.
unset QBLADE_STEPS

# The parent treats SLURM_ARRAY_TASK_ID as a case index and exits before its
# complete postprocessing path. This profile has already selected one speed, so
# clear only that marker and let each task complete its own full workflow.
export VORLAP_PROFILE_ARRAY_TASK_ID="${profile_array_task_id}"
unset SLURM_ARRAY_TASK_ID

echo "Nyquist 10-minute reduced-wake pilot: speed=${profile_wind_speed}m/s turbsim_seed=${TURBSIM_SEED_BASE} run_root=${RUN_ROOT} wake_profile=${QBLADE_WAKE_PROFILE} result_filter=${QBLADE_RESULT_FILTER} walltime=47:59:00 node_threads=auto scenarios='${CAMPAIGN_SCENARIOS}'."
exec bash "${VORLAP_SUBMIT_SCRIPT}"
