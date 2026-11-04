#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=vorlap-dlc-row
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

# Login-shell submission mode creates a wind-generation array followed by a
# QBlade array that starts only after every wind row succeeds. The current
# measured full-run allocation is batch/47:59:00; both remain overridable.

initial_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${VORLAP_ROOT:-}" && -f "${VORLAP_ROOT}/run_headless/build_parked_dlc_manifest.py" ]]; then
  SCRIPT_DIR="$(cd "${VORLAP_ROOT}/run_headless" && pwd -P)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_headless/build_parked_dlc_manifest.py" ]]; then
  VORLAP_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
  SCRIPT_DIR="${VORLAP_ROOT}/run_headless"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/build_parked_dlc_manifest.py" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
  VORLAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
elif [[ -f "${initial_script_dir}/build_parked_dlc_manifest.py" ]]; then
  SCRIPT_DIR="${initial_script_dir}"
  VORLAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
else
  echo "ERROR: Could not resolve the VorLap run_headless checkout from the Slurm spool copy; submit from the checkout or export VORLAP_ROOT." >&2
  exit 2
fi
DLC_SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_parked_dlc_manifest.sj"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_EXE}" ]]; then
  PYTHON_EXE="${PYTHON_EXE_FALLBACK:-python3}"
fi
if [[ "${PYTHON_EXE}" == */* && -x "${PYTHON_EXE}" ]]; then
  python_dir="$(cd "$(dirname "${PYTHON_EXE}")" && pwd -P)"
  PYTHON_EXE="${python_dir}/$(basename "${PYTHON_EXE}")"
elif command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
  PYTHON_EXE="$(command -v "${PYTHON_EXE}")"
fi
MANIFEST_TOOL="${SCRIPT_DIR}/build_parked_dlc_manifest.py"
ROW_RUNNER="${SCRIPT_DIR}/run_parked_dlc_manifest_row.py"
DLC_MANIFEST="${DLC_MANIFEST:-${SCRIPT_DIR}/config/parked_dlc_matrix_approved.json}"
DLC_QBLADE_TIME="${DLC_QBLADE_TIME:-47:59:00}"
DLC_QBLADE_PARTITION="${DLC_QBLADE_PARTITION:-batch}"
if [[ -z "${SLURM_JOB_ID:-}" && -z "${DLC_PHASE:-}" ]]; then
  SUBMIT_DLC_MANIFEST="${SUBMIT_DLC_MANIFEST:-1}"
fi

die() {
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

absolute_file() {
  local path="$1"
  local directory
  [[ -f "${path}" ]] || return 1
  directory="$(cd "$(dirname "${path}")" && pwd -P)"
  printf '%s/%s\n' "${directory}" "$(basename "${path}")"
}

if [[ "${DLC_ALLOW_DRAFT:-0}" == "1" ]]; then
  [[ "${DLC_SUBMIT_DRY_RUN:-0}" == "1" || "${DLC_PARENT_DRY_RUN:-0}" == "1" ]] ||
    die "DLC_ALLOW_DRAFT=1 is permitted only for dry-run planning."
fi

DLC_MANIFEST="$(absolute_file "${DLC_MANIFEST}")" ||
  die "Approved manifest file not found: ${DLC_MANIFEST}. Build and review it after the pilot gates pass."
export DLC_MANIFEST

if [[ "${SUBMIT_DLC_MANIFEST:-0}" == "1" ]]; then
  [[ -z "${SLURM_JOB_ID:-}" ]] ||
    die "SUBMIT_DLC_MANIFEST=1 must run from a login shell, not Slurm job ${SLURM_JOB_ID}."
  if [[ "${DLC_SUBMIT_DRY_RUN:-0}" != "1" && "${DLC_REQUIRE_CLEAN_CHECKOUT:-1}" == "1" ]]; then
    tracked_status="$(git -C "${VORLAP_ROOT}" status --porcelain --untracked-files=no)" ||
      die "Could not inspect tracked checkout state at ${VORLAP_ROOT}."
    if [[ -n "${tracked_status}" ]]; then
      disallowed_status=""
      while IFS= read -r status_line; do
        [[ -n "${status_line}" ]] || continue
        tracked_path="${status_line:3}"
        if [[ "${DLC_ALLOW_DIRTY_RUNTIME_ARTIFACTS:-1}" != "1" ]] ||
          ! is_known_cluster_runtime_artifact "${tracked_path}"; then
          disallowed_status+="${status_line}"$'\n'
        fi
      done <<<"${tracked_status}"
      if [[ -n "${disallowed_status}" ]]; then
        printf 'Disallowed tracked changes:\n%s' "${disallowed_status}" >&2
        die "Production submission requires a clean tracked checkout except for known cluster-built runtime artifacts. Commit or restore other tracked changes, set DLC_ALLOW_DIRTY_RUNTIME_ARTIFACTS=1 for the known artifacts, or explicitly set DLC_REQUIRE_CLEAN_CHECKOUT=0."
      fi
      printf 'WARNING: Allowing known cluster-built tracked runtime artifacts:\n%s\n' "${tracked_status}" >&2
    fi
  fi

  if [[ "${DLC_ALLOW_DRAFT:-0}" == "1" ]]; then
    "${PYTHON_EXE}" "${MANIFEST_TOOL}" validate \
      --manifest "${DLC_MANIFEST}" --allow-draft
    row_count="$("${PYTHON_EXE}" "${MANIFEST_TOOL}" count \
      --manifest "${DLC_MANIFEST}" --allow-draft)"
  else
    "${PYTHON_EXE}" "${MANIFEST_TOOL}" validate --manifest "${DLC_MANIFEST}"
    row_count="$("${PYTHON_EXE}" "${MANIFEST_TOOL}" count --manifest "${DLC_MANIFEST}")"
  fi
  if [[ "${DLC_ALLOW_DRAFT:-0}" == "1" ]]; then
    "${PYTHON_EXE}" "${MANIFEST_TOOL}" disk-check \
      --manifest "${DLC_MANIFEST}" \
      --required-factor "${DLC_DISK_REQUIRED_FACTOR:-1.25}" \
      --allow-draft
  else
    "${PYTHON_EXE}" "${MANIFEST_TOOL}" disk-check \
      --manifest "${DLC_MANIFEST}" \
      --required-factor "${DLC_DISK_REQUIRED_FACTOR:-1.25}"
  fi
  [[ "${row_count}" =~ ^[1-9][0-9]*$ ]] || die "Invalid manifest row count: ${row_count}"
  array_max=$((row_count - 1))
  array_spec="0-${array_max}%${DLC_ARRAY_CONCURRENCY:-${row_count}}"
  if [[ "${DLC_ALLOW_DRAFT:-0}" == "1" ]]; then
    manifest_run_root="$("${PYTHON_EXE}" "${MANIFEST_TOOL}" root \
      --manifest "${DLC_MANIFEST}" --kind run --allow-draft)"
  else
    manifest_run_root="$("${PYTHON_EXE}" "${MANIFEST_TOOL}" root \
      --manifest "${DLC_MANIFEST}" --kind run)"
  fi
  DLC_AGGREGATE_OUT="${DLC_AGGREGATE_OUT:-${manifest_run_root}/aggregate}"
  wind_time="${DLC_WIND_TIME:-02:00:00}"
  wind_partition="${DLC_WIND_PARTITION:-short}"
  aggregate_time="${DLC_AGGREGATE_TIME:-02:00:00}"
  aggregate_partition="${DLC_AGGREGATE_PARTITION:-short}"
  job_prefix="${DLC_JOB_PREFIX:-vorlap-dlc}"
  common_export="ALL,DLC_MANIFEST=${DLC_MANIFEST},DLC_AGGREGATE_OUT=${DLC_AGGREGATE_OUT},VORLAP_ROOT=${VORLAP_ROOT},PYTHON_EXE=${PYTHON_EXE}"
  if [[ "${DLC_ALLOW_DRAFT:-0}" == "1" ]]; then
    common_export+=",DLC_ALLOW_DRAFT=1,DLC_PARENT_DRY_RUN=1"
  fi

  if [[ "${DLC_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
    if [[ "${DLC_SKIP_WIND_SUBMISSION:-0}" != "1" ]]; then
      printf 'sbatch --parsable --array=%q --time=%q --partition=%q --job-name=%q --export=%q %q\n' \
        "${array_spec}" "${wind_time}" "${wind_partition}" "${job_prefix}-wind" \
        "${common_export},DLC_PHASE=wind" "${DLC_SUBMIT_SCRIPT}"
      dependency="afterok:WIND_ARRAY_JOB_ID"
    else
      dependency=""
    fi
    qblade_args=(
      sbatch --parsable
      --exclusive
      "--array=${array_spec}"
      "--time=${DLC_QBLADE_TIME}"
      "--partition=${DLC_QBLADE_PARTITION}"
      "--job-name=${job_prefix}-qblade"
      "--export=${common_export},DLC_PHASE=qblade"
    )
    if [[ -n "${dependency}" ]]; then
      qblade_args+=("--dependency=${dependency}")
    fi
    printf '%q ' "${qblade_args[@]}"
    printf '%q\n' "${DLC_SUBMIT_SCRIPT}"
    printf 'sbatch --parsable --time=%q --partition=%q --job-name=%q --export=%q --dependency=%q %q\n' \
      "${aggregate_time}" "${aggregate_partition}" "${job_prefix}-aggregate" \
      "${common_export},DLC_PHASE=aggregate" "afterok:QBLADE_ARRAY_JOB_ID" \
      "${DLC_SUBMIT_SCRIPT}"
    exit 0
  fi

  qblade_dependency=""
  if [[ "${DLC_SKIP_WIND_SUBMISSION:-0}" != "1" ]]; then
    wind_job_id="$(
      sbatch --parsable \
        --array="${array_spec}" \
        --time="${wind_time}" \
        --partition="${wind_partition}" \
        --job-name="${job_prefix}-wind" \
        --export="${common_export},DLC_PHASE=wind" \
        "${DLC_SUBMIT_SCRIPT}"
    )"
    echo "Submitted wind array: ${wind_job_id}"
    qblade_dependency="afterok:${wind_job_id}"
  fi
  if [[ -n "${qblade_dependency}" ]]; then
    qblade_job_id="$(
      sbatch --parsable \
        --exclusive \
        --array="${array_spec}" \
        --time="${DLC_QBLADE_TIME}" \
        --partition="${DLC_QBLADE_PARTITION}" \
        --job-name="${job_prefix}-qblade" \
        --export="${common_export},DLC_PHASE=qblade" \
        --dependency="${qblade_dependency}" \
        "${DLC_SUBMIT_SCRIPT}"
    )"
  else
    qblade_job_id="$(
      sbatch --parsable \
        --exclusive \
        --array="${array_spec}" \
        --time="${DLC_QBLADE_TIME}" \
        --partition="${DLC_QBLADE_PARTITION}" \
        --job-name="${job_prefix}-qblade" \
        --export="${common_export},DLC_PHASE=qblade" \
        "${DLC_SUBMIT_SCRIPT}"
    )"
  fi
  echo "Submitted QBlade array: ${qblade_job_id}"
  aggregate_job_id="$(
    sbatch --parsable \
      --time="${aggregate_time}" \
      --partition="${aggregate_partition}" \
      --job-name="${job_prefix}-aggregate" \
      --export="${common_export},DLC_PHASE=aggregate" \
      --dependency="afterok:${qblade_job_id}" \
      "${DLC_SUBMIT_SCRIPT}"
  )"
  echo "Submitted aggregation job: ${aggregate_job_id}"
  exit 0
fi

DLC_PHASE="${DLC_PHASE:-}"
[[ "${DLC_PHASE}" == "wind" || "${DLC_PHASE}" == "qblade" || "${DLC_PHASE}" == "aggregate" ]] ||
  die "DLC_PHASE must be wind, qblade, or aggregate."
if [[ "${DLC_PHASE}" == "aggregate" ]]; then
  [[ "${DLC_ALLOW_DRAFT:-0}" != "1" ]] ||
    die "Aggregation is not allowed for a draft manifest."
  [[ -n "${DLC_AGGREGATE_OUT:-}" ]] ||
    die "DLC_AGGREGATE_OUT is required for aggregation."
  exec "${PYTHON_EXE}" "${SCRIPT_DIR}/aggregate_parked_dlc_manifest.py" \
    --manifest "${DLC_MANIFEST}" \
    --out-dir "${DLC_AGGREGATE_OUT}" \
    --wohler-exponent "${WOHLER_EXPONENT:-10}" \
    --equivalent-cycles "${POSTPROCESS_EQUIVALENT_CYCLES:-600}"
fi

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] ||
  die "Wind/QBlade row execution requires SLURM_ARRAY_TASK_ID."

runner_flags=(
  --manifest "${DLC_MANIFEST}"
  --index "${SLURM_ARRAY_TASK_ID}"
  --phase "${DLC_PHASE}"
)
if [[ "${DLC_PARENT_DRY_RUN:-0}" == "1" ]]; then
  runner_flags+=(--parent-dry-run)
fi
if [[ "${DLC_ALLOW_DRAFT:-0}" == "1" ]]; then
  runner_flags+=(--allow-draft)
fi

exec "${PYTHON_EXE}" "${ROW_RUNNER}" "${runner_flags[@]}"
