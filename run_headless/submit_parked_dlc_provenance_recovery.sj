#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=vorlap-dlc-prov-recover
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

# Login-shell mode discovers rows failed solely by checkout-drift provenance,
# submits zero-timestep completed-checkpoint recovery for those rows, and chains
# aggregate-only postprocessing after every recovered row passes its final audit.

initial_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${VORLAP_ROOT:-}" && -f "${VORLAP_ROOT}/run_headless/recover_parked_dlc_provenance.py" ]]; then
  VORLAP_ROOT="$(cd "${VORLAP_ROOT}" && pwd -P)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_headless/recover_parked_dlc_provenance.py" ]]; then
  VORLAP_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
elif [[ -f "${initial_script_dir}/recover_parked_dlc_provenance.py" ]]; then
  VORLAP_ROOT="$(cd "${initial_script_dir}/.." && pwd -P)"
else
  echo "ERROR: Could not resolve the VorLap checkout; submit from the checkout or export VORLAP_ROOT." >&2
  exit 2
fi

SCRIPT_DIR="${VORLAP_ROOT}/run_headless"
RECOVERY_TOOL="${SCRIPT_DIR}/recover_parked_dlc_provenance.py"
RECOVERY_SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_parked_dlc_provenance_recovery.sj"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/submit_parked_dlc_manifest.sj"
MANIFEST_TOOL="${SCRIPT_DIR}/build_parked_dlc_manifest.py"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${SCRIPT_DIR}/QBladeCE_2.0.9.7}"
DLC_MANIFEST="${DLC_MANIFEST:-${SCRIPT_DIR}/config/parked_dlc_matrix_approved.json}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

absolute_file() {
  local path="$1"
  local directory
  [[ -f "${path}" ]] || return 1
  directory="$(cd "$(dirname "${path}")" && pwd -P)"
  printf '%s/%s\n' "${directory}" "$(basename "${path}")"
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

check_checkout() {
  local tracked_status disallowed_status status_line tracked_path
  [[ "${DLC_RECOVERY_REQUIRE_CLEAN_CHECKOUT:-1}" == "1" ]] || return 0
  tracked_status="$(git -C "${VORLAP_ROOT}" status --porcelain --untracked-files=no)" ||
    die "Could not inspect tracked checkout state at ${VORLAP_ROOT}."
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
    die "Provenance recovery requires a clean tracked checkout except for known cluster-built runtime artifacts."
  fi
  if [[ -n "${tracked_status}" ]]; then
    printf 'WARNING: Allowing known cluster-built tracked runtime artifacts:\n%s\n' "${tracked_status}" >&2
  fi
}

DLC_MANIFEST="$(absolute_file "${DLC_MANIFEST}")" ||
  die "Approved manifest not found: ${DLC_MANIFEST}"
[[ -x "${PYTHON_EXE}" ]] || die "Python environment is unavailable: ${PYTHON_EXE}"
[[ -d "${QBLADE_ROOT}" ]] || die "QBlade root is unavailable: ${QBLADE_ROOT}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  "${PYTHON_EXE}" "${MANIFEST_TOOL}" validate --manifest "${DLC_MANIFEST}"
  recovery_array="$(
    "${PYTHON_EXE}" "${RECOVERY_TOOL}" plan \
      --manifest "${DLC_MANIFEST}" \
      --format array
  )"
  manifest_run_root="$(
    "${PYTHON_EXE}" "${MANIFEST_TOOL}" root \
      --manifest "${DLC_MANIFEST}" \
      --kind run
  )"
  DLC_AGGREGATE_OUT="${DLC_AGGREGATE_OUT:-${manifest_run_root}/aggregate}"
  recovery_time="${DLC_RECOVERY_TIME:-04:00:00}"
  recovery_partition="${DLC_RECOVERY_PARTITION:-batch}"
  recovery_concurrency="${DLC_RECOVERY_CONCURRENCY:-6}"
  aggregate_time="${DLC_AGGREGATE_TIME:-02:00:00}"
  aggregate_partition="${DLC_AGGREGATE_PARTITION:-short}"
  job_prefix="${DLC_RECOVERY_JOB_PREFIX:-vorlap-dlc}"
  common_export="ALL,DLC_MANIFEST=${DLC_MANIFEST},DLC_AGGREGATE_OUT=${DLC_AGGREGATE_OUT},VORLAP_ROOT=${VORLAP_ROOT},PYTHON_EXE=${PYTHON_EXE},QBLADE_ROOT=${QBLADE_ROOT}"

  if [[ "${DLC_RECOVERY_SUBMIT_DRY_RUN:-0}" != "1" ]]; then
    check_checkout
  fi

  dependency=""
  if [[ -n "${recovery_array}" ]]; then
    if [[ "${DLC_RECOVERY_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
      printf 'sbatch --parsable --exclusive --array=%q --time=%q --partition=%q --job-name=%q --export=%q %q\n' \
        "${recovery_array}%${recovery_concurrency}" "${recovery_time}" "${recovery_partition}" \
        "${job_prefix}-prov-recover" "${common_export}" "${RECOVERY_SUBMIT_SCRIPT}"
      dependency="afterok:RECOVERY_ARRAY_JOB_ID"
    else
      recovery_job_id="$(
        sbatch --parsable \
          --exclusive \
          --array="${recovery_array}%${recovery_concurrency}" \
          --time="${recovery_time}" \
          --partition="${recovery_partition}" \
          --job-name="${job_prefix}-prov-recover" \
          --export="${common_export}" \
          "${RECOVERY_SUBMIT_SCRIPT}"
      )"
      echo "Submitted provenance-recovery array: ${recovery_job_id} (${recovery_array})"
      dependency="afterok:${recovery_job_id}"
    fi
  else
    echo "No recoverable failed result audits remain; submitting aggregation directly."
  fi

  aggregate_args=(
    sbatch --parsable
    "--time=${aggregate_time}"
    "--partition=${aggregate_partition}"
    "--job-name=${job_prefix}-aggregate-recovery"
    "--export=${common_export},DLC_PHASE=aggregate"
  )
  if [[ -n "${dependency}" ]]; then
    aggregate_args+=("--dependency=${dependency}")
  fi
  aggregate_args+=("${AGGREGATE_SCRIPT}")

  if [[ "${DLC_RECOVERY_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${aggregate_args[@]}"
    printf '\n'
  else
    aggregate_job_id="$("${aggregate_args[@]}")"
    echo "Submitted aggregation job: ${aggregate_job_id}"
  fi
  exit 0
fi

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] ||
  die "Recovery worker requires SLURM_ARRAY_TASK_ID."

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
recovery_threads="${recovery_threads:-1}"
export OMP_NUM_THREADS="${recovery_threads}"

exec "${PYTHON_EXE}" "${RECOVERY_TOOL}" run \
  --manifest "${DLC_MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --vorlap-root "${VORLAP_ROOT}" \
  --python-exe "${PYTHON_EXE}" \
  --qblade-root "${QBLADE_ROOT}" \
  --parent-submit "${SCRIPT_DIR}/submit_parked_demo_campaign.sj" \
  --threads "${recovery_threads}"
