#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=47:59:00
#SBATCH --account=FY260158
#SBATCH --job-name=vorlap-oldwake-recover
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --array=0-1%2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

# Recover the completed, unfiltered baseline checkpoints from the original
# reference-wake U05/U10 submission, then create/run only the corresponding
# filtered VorLap cases. Repeating this submission skips a hash-verified
# baseline export. VorLap checkpoints created before the SIL runtime-filter
# fix restart at step zero; later corrected checkpoints resume normally.

export VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
export QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
export TURBSIM_WIND_ROOT="${TURBSIM_WIND_ROOT:-/pscratch/kevmoor/vorlap_wind_fields/changed_geometry_trial}"
export VORLAP_RECOVERY_RUN_ROOT="${VORLAP_RECOVERY_RUN_ROOT:-${VORLAP_ROOT}/run_headless/runs/vorlap_nyquist_10min_U05_U10_20260801}"
export QBLADE_RESULT_FILTER="${QBLADE_RESULT_FILTER:-${VORLAP_ROOT}/run_headless/config/parked_fatigue_results.filter}"
export PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"

case "${SLURM_ARRAY_TASK_ID:-}" in
  0)
    recovery_wind_speed="5"
    recovery_wind_label="U05p00ms"
    ;;
  1)
    recovery_wind_speed="10"
    recovery_wind_label="U10p00ms"
    ;;
  *)
    echo "ERROR: expected Slurm array task 0 or 1; got '${SLURM_ARRAY_TASK_ID:-unset}'." >&2
    exit 2
    ;;
esac

export RUN_ROOT="${VORLAP_RECOVERY_RUN_ROOT}/${recovery_wind_label}"
recovery_wind_file="${TURBSIM_WIND_ROOT}/${recovery_wind_label}/dlc6p4_${recovery_wind_label}.bts"

for required in \
  "${PYTHON_EXE}" \
  "${QBLADE_RESULT_FILTER}" \
  "${recovery_wind_file}" \
  "${RUN_ROOT}/campaign_settings.json" \
  "${RUN_ROOT}/baseline/cases.json" \
  "${RUN_ROOT}/baseline/results/${recovery_wind_label}/checkpoints/latest_checkpoint.json"; do
  [[ -e "${required}" ]] || {
    echo "ERROR: required recovery input is missing: ${required}" >&2
    exit 2
  }
done

recovery_threads=""
for variable in SLURM_CPUS_ON_NODE SLURM_JOB_CPUS_PER_NODE SLURM_CPUS_PER_TASK; do
  value="${!variable:-}"
  value="${value%%(*}"
  value="${value%%,*}"
  if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )); then
    recovery_threads="${value}"
    break
  fi
done
export OMP_NUM_THREADS="${recovery_threads:-1}"

batch_script="${VORLAP_ROOT}/run_headless/vorlap_qblade_batch.py"
campaign_script="${VORLAP_ROOT}/run_headless/run_parked_demo_campaign.py"
audit_script="${VORLAP_ROOT}/run_headless/audit_parked_campaign.py"

common_runtime_args=(
  --qblade-root "${QBLADE_ROOT}"
  --vorlap-root "${VORLAP_ROOT}"
  --python "${PYTHON_EXE}"
  --result-filter "${QBLADE_RESULT_FILTER}"
  --case-index 0
  --threads "${OMP_NUM_THREADS}"
  --progress-every 10
  --checkpoint-every 4000
  --skip-vorlap-install
  --skip-bridge-build
)

echo "Recovering reference-wake pilot: speed=${recovery_wind_speed}m/s run_root=${RUN_ROOT} threads=${OMP_NUM_THREADS} result_filter=${QBLADE_RESULT_FILTER}."

"${PYTHON_EXE}" "${batch_script}" recover-export \
  --run-root "${RUN_ROOT}/baseline" \
  --coupling none \
  "${common_runtime_args[@]}"

# The original submission stopped after the baseline export allocation failed.
# Setup regenerates only the VorLap scenario with FILTERFILE enabled and the
# original reference wake. Later invocations reuse it and resume compatible
# corrected checkpoints.
"${PYTHON_EXE}" "${batch_script}" run \
  --run-root "${RUN_ROOT}/vorlap" \
  --wind-speeds "${recovery_wind_speed}" \
  --azimuths 0 \
  --timesteps 88000 \
  --timestep 0.0075 \
  --store-from 60 \
  --store-replay false \
  --rpm 1 \
  --horizontal-inflow-angle 0 \
  --vertical-inflow-angle 0 \
  --shear-exponent 0.23 \
  --reference-height 18 \
  --air-density 1.0296 \
  --air-kinematic-viscosity 1.647e-5 \
  --wake-profile reference \
  --rampup 0 \
  --add-damp 0 \
  --force-scale 1 \
  --source-bounds-policy warn \
  --source-frequency-truncation nyquist \
  --wind-type 1 \
  --wind-name "${recovery_wind_file}" \
  --coupling sil-vorlap \
  "${common_runtime_args[@]}"

"${PYTHON_EXE}" "${campaign_script}" \
  --run-root "${RUN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}" \
  --vorlap-root "${VORLAP_ROOT}" \
  --python "${PYTHON_EXE}" \
  --wind-speeds "${recovery_wind_speed}" \
  --azimuths 0 \
  --duration 660 \
  --timestep 0.0075 \
  --store-from 60 \
  --store-replay false \
  --result-filter "${QBLADE_RESULT_FILTER}" \
  --rpm 1 \
  --wake-profile reference \
  --source-bounds-policy warn \
  --source-frequency-truncation nyquist \
  --checkpoint-every 4000 \
  --threads "${OMP_NUM_THREADS}" \
  --case-workers 1 \
  --scenarios "baseline vorlap" \
  --coupling sil-vorlap \
  --min-fatigue-duration 600 \
  --only-postprocess

mkdir -p "${RUN_ROOT}/audit"
"${PYTHON_EXE}" "${audit_script}" \
  --run-root "${RUN_ROOT}" \
  --stage results \
  --out "${RUN_ROOT}/audit/results_recovery_audit.json" \
  --scenarios "baseline vorlap" \
  --expected-rpm 1 \
  --min-duration 660 \
  --min-analyzed-duration 600 \
  --require-wind-name \
  --require-source-coverage \
  --require-postprocess-artifacts

echo "Reference-wake recovery workflow completed for ${recovery_wind_label}: ${RUN_ROOT}."
