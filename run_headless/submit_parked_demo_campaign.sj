#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=47:59:00
#SBATCH --account=FY260158
#SBATCH --job-name=vorlap-qblade-dlc
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

# Use SUBMIT_WIND_JOBS=1 from a login shell to submit one independent one-node
# job per wind speed. Manual Slurm arrays are still supported with sbatch
# --array=0-$((N-1)); each array task consumes one planned wind-speed/azimuth
# case index and runs the selected scenarios for that case.

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

SERIALIZED_FORCE_SETUP_ROLE="none"
SERIALIZED_FORCE_SETUP_READY=""
SERIALIZED_FORCE_SETUP_FAILED=""
SERIALIZED_FORCE_SETUP_COMPLETE=0

atomic_setup_marker() {
  local path="$1"
  local message="$2"
  local temporary="${path}.tmp.${SLURM_JOB_ID:-$$}.${RANDOM}"
  printf '%s\n' "${message}" > "${temporary}"
  mv -f "${temporary}" "${path}"
}

disable_forced_setup_flags() {
  export FORCE_PYTHON_SETUP=0
  export FORCE_OPENFAST_TURBSIM_BUILD=0
  export FORCE_BRIDGE_BUILD=0
  export FORCE_CASE_SETUP=0
  export FORCE_GENERATE_TURBSIM=0
}

record_serialized_force_setup_failure() {
  local status="$1"
  if [[ "${SERIALIZED_FORCE_SETUP_ROLE}" == "leader" &&
        "${SERIALIZED_FORCE_SETUP_COMPLETE}" != "1" &&
        "${status}" != "0" &&
        -n "${SERIALIZED_FORCE_SETUP_FAILED}" ]]; then
    atomic_setup_marker \
      "${SERIALIZED_FORCE_SETUP_FAILED}" \
      "leader task ${SLURM_ARRAY_TASK_ID:-manual} exited with status ${status}"
  fi
}

prepare_serialized_force_setup() {
  [[ "${VORLAP_SERIALIZE_FORCE_SETUP:-0}" == "1" ]] || return 0
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || return 0
  [[ "${DRY_RUN_ONLY:-0}" != "1" ]] || return 0
  [[ "${SUBMIT_WIND_JOBS_DRY_RUN:-0}" != "1" ]] || return 0
  [[ "${ARRAY_CASE_DRY_RUN:-0}" != "1" ]] || return 0

  local token="${VORLAP_FORCE_SETUP_TOKEN:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}}"
  local safe_token="${token//[^A-Za-z0-9_.-]/_}"
  local state_dir="${VORLAP_ROOT}/.vorlap_setup/serialized-batch"
  local leader_task="${SLURM_ARRAY_TASK_MIN:-0}"
  local wait_seconds="${VORLAP_SERIALIZED_SETUP_WAIT_SECONDS:-1800}"
  local deadline

  [[ -n "${safe_token}" ]] ||
    die "Serialized forced setup requires VORLAP_FORCE_SETUP_TOKEN or a Slurm job ID."
  [[ "${wait_seconds}" =~ ^[0-9]+$ ]] && (( wait_seconds > 0 )) ||
    die "VORLAP_SERIALIZED_SETUP_WAIT_SECONDS must be a positive integer."

  mkdir -p "${state_dir}"
  SERIALIZED_FORCE_SETUP_READY="${state_dir}/${safe_token}.ready"
  SERIALIZED_FORCE_SETUP_FAILED="${state_dir}/${safe_token}.failed"

  if [[ "${SLURM_ARRAY_TASK_ID}" == "${leader_task}" ]]; then
    SERIALIZED_FORCE_SETUP_ROLE="leader"
    rm -f "${SERIALIZED_FORCE_SETUP_READY}" "${SERIALIZED_FORCE_SETUP_FAILED}"
    trap 'record_serialized_force_setup_failure "$?"' EXIT
    log "Array task ${SLURM_ARRAY_TASK_ID} is the one-time forced-setup leader for batch token ${token}."
    return 0
  fi

  SERIALIZED_FORCE_SETUP_ROLE="follower"
  disable_forced_setup_flags
  deadline=$((SECONDS + wait_seconds))
  log "Array task ${SLURM_ARRAY_TASK_ID} is waiting for one-time forced setup from leader task ${leader_task}."
  while [[ ! -f "${SERIALIZED_FORCE_SETUP_READY}" ]]; do
    if [[ -f "${SERIALIZED_FORCE_SETUP_FAILED}" ]]; then
      die "Serialized forced setup failed: $(<"${SERIALIZED_FORCE_SETUP_FAILED}")"
    fi
    (( SECONDS < deadline )) ||
      die "Timed out after ${wait_seconds}s waiting for serialized forced setup marker ${SERIALIZED_FORCE_SETUP_READY}."
    sleep 2
  done
  if [[ -x "${VORLAP_ROOT}/.venv/bin/python" ]]; then
    PYTHON_EXE="${VORLAP_ROOT}/.venv/bin/python"
    export PYTHON_EXE
  fi
  log "One-time forced setup is ready; task ${SLURM_ARRAY_TASK_ID} will reuse it."
}

complete_serialized_force_setup() {
  [[ "${SERIALIZED_FORCE_SETUP_ROLE}" == "leader" ]] || return 0
  local setup_cmd=("${campaign_cmd[@]}")
  local argument
  local has_reuse_setup=0
  if [[ "${FORCE_CASE_SETUP:-0}" == "1" ]]; then
    local filtered_setup_cmd=()
    for argument in "${setup_cmd[@]}"; do
      [[ "${argument}" == "--reuse-setup" ]] || filtered_setup_cmd+=("${argument}")
    done
    setup_cmd=("${filtered_setup_cmd[@]}")
    log "Forced generated-case refresh requested; the one-time setup will recopy reference-model assets."
  else
    for argument in "${setup_cmd[@]}"; do
      if [[ "${argument}" == "--reuse-setup" ]]; then
        has_reuse_setup=1
        break
      fi
    done
    if (( ! has_reuse_setup )); then
      setup_cmd+=(--reuse-setup)
    fi
  fi

  log "Running one-time serialized Python/TurbSim/bridge and case setup for this batch."
  "${setup_cmd[@]}" --setup-only
  local filtered_campaign_cmd=()
  for argument in "${campaign_cmd[@]}"; do
    case "${argument}" in
      --force-bridge-build|--force-generate-turbsim) ;;
      *) filtered_campaign_cmd+=("${argument}") ;;
    esac
  done
  campaign_cmd=("${filtered_campaign_cmd[@]}")
  atomic_setup_marker \
    "${SERIALIZED_FORCE_SETUP_READY}" \
    "leader task ${SLURM_ARRAY_TASK_ID:-manual} completed serialized setup"
  SERIALIZED_FORCE_SETUP_COMPLETE=1
  disable_forced_setup_flags
  log "One-time forced setup completed; releasing the remaining array tasks."
  log "Leader and follower case execution continues with all force flags disabled."
}

abspath_file() {
  local path="$1"
  local dir
  dir="$(cd "$(dirname "${path}")" && pwd -P)"
  printf '%s/%s\n' "${dir}" "$(basename "${path}")"
}

candidate_is_run_headless_script() {
  local path="$1"
  [[ -f "${path}" && -f "$(dirname "${path}")/run_parked_demo_campaign.py" ]]
}

resolve_submit_script() {
  local candidate
  for candidate in \
    "${VORLAP_SUBMIT_SCRIPT:-}" \
    "${BASH_SOURCE[0]:-}" \
    "${SLURM_SUBMIT_DIR:-}/run_headless/submit_parked_demo_campaign.sj" \
    "${SLURM_SUBMIT_DIR:-}/submit_parked_demo_campaign.sj" \
    "$(pwd -P)/run_headless/submit_parked_demo_campaign.sj" \
    "$(pwd -P)/submit_parked_demo_campaign.sj"; do
    [[ -n "${candidate}" ]] || continue
    if candidate_is_run_headless_script "${candidate}"; then
      abspath_file "${candidate}"
      return 0
    fi
  done
  return 1
}

qblade_library_exists() {
  local root="$1"
  compgen -G "${root}/*QBlade*.so*" >/dev/null ||
    compgen -G "${root}/*QBlade*.dll" >/dev/null ||
    compgen -G "${root}/*QBlade*.dylib" >/dev/null
}

qblade_root_preflight_needed() {
  [[ "${SKIP_QBLADE_ROOT_PREFLIGHT:-0}" == "1" ]] && return 1
  [[ "${WIND_ONLY:-0}" == "1" ]] && return 1
  [[ "${DRY_RUN_ONLY:-0}" == "1" ]] && return 1
  [[ "${SUBMIT_WIND_JOBS_DRY_RUN:-0}" == "1" ]] && return 1
  [[ "${ARRAY_CASE_DRY_RUN:-0}" == "1" ]] && return 1
  [[ "${ONLY_POSTPROCESS:-0}" == "1" ]] && return 1
  [[ "${PAPER_CHECKS_ONLY:-0}" == "1" ]] && return 1
  return 0
}

validate_qblade_root_preflight() {
  local root="$1"
  local missing=()
  if [[ ! -d "${root}" ]]; then
    missing+=("missing QBlade root directory: ${root}")
  else
    [[ -d "${root}/SIL_Interface" ]] || missing+=("missing QBlade SIL_Interface directory: ${root}/SIL_Interface")
    [[ -d "${root}/ControllerFiles" ]] || missing+=("missing QBlade ControllerFiles directory: ${root}/ControllerFiles")
    qblade_library_exists "${root}" || missing+=("missing QBlade shared library in ${root} matching one of: *QBlade*.so*, *QBlade*.dll, *QBlade*.dylib")
  fi

  if ((${#missing[@]})); then
    {
      echo "ERROR: QBlade root is incomplete: ${root}"
      printf '  - %s\n' "${missing[@]}"
      echo
      echo "This path must contain the complete QBlade CE runtime used by the headless bridge jobs."
      echo "Update the checkout so run_headless/QBladeCE_2.0.9.7 is present, copy a complete as-downloaded QBladeCE_2.0.9.7 directory to this path, or export QBLADE_ROOT=/absolute/path/to/a/complete/QBladeCE_2.0.9.7 before submitting."
      echo "Example:"
      echo "  mkdir -p \"$(dirname "${root}")\""
      echo "  rsync -a /path/to/working/QBladeCE_2.0.9.7/ \"${root}/\""
      echo
      echo "Set SKIP_QBLADE_ROOT_PREFLIGHT=1 only if a site wrapper makes QBLADE_ROOT visible later on the compute node."
    } >&2
    exit 2
  fi
}

resolve_turbsim_executable() {
  local exe="$1"
  [[ -n "${exe}" ]] || return 1
  if [[ "${exe}" == */* ]]; then
    printf '%s\n' "${exe}"
  else
    command -v "${exe}"
  fi
}

turbsim_command_works() {
  local exe resolved output
  exe="$1"
  resolved="$(resolve_turbsim_executable "${exe}" 2>/dev/null)" || return 1
  [[ -x "${resolved}" ]] || return 1
  if command -v ldd >/dev/null 2>&1 &&
     ldd "${resolved}" 2>/dev/null | grep -q 'not found'; then
    return 1
  fi
  if command -v timeout >/dev/null 2>&1; then
    output="$(timeout "${TURBSIM_PROBE_TIMEOUT:-15}" "${resolved}" -h 2>&1 || true)"
  else
    output="$("${resolved}" -h 2>&1 || true)"
  fi
  grep -Eqi 'TurbSim[^[:cntrl:]]*(version|v[0-9])' <<< "${output}"
}

setup_turbsim_if_needed() {
  [[ "${GENERATE_TURBSIM:-0}" == "1" ]] || return 0
  [[ "${DRY_RUN_ONLY:-0}" == "1" ]] && return 0
  [[ "${SUBMIT_WIND_JOBS_DRY_RUN:-0}" == "1" ]] && return 0
  [[ "${ARRAY_CASE_DRY_RUN:-0}" == "1" ]] && return 0
  local force_build="${FORCE_OPENFAST_TURBSIM_BUILD:-0}"
  if [[ "${force_build}" == "1" ]]; then
    log "Forced TurbSim reinstall/rebuild requested."
  elif turbsim_command_works "${TURBSIM_EXE:-}"; then
    log "TurbSim executable passed its startup check: ${TURBSIM_EXE}"
    return 0
  fi
  if [[ "${force_build}" != "1" && -n "${TURBSIM_EXE:-}" ]]; then
    log "WARNING: Configured TurbSim executable is missing or failed its startup check: ${TURBSIM_EXE}"
  fi
  local default_exe="${VORLAP_ROOT}/run_headless/openfast/install/bin/turbsim"
  if [[ "${force_build}" != "1" ]] && turbsim_command_works "${default_exe}"; then
    TURBSIM_EXE="${default_exe}"
    export TURBSIM_EXE
    log "TurbSim executable passed its startup check: ${TURBSIM_EXE}"
    return 0
  fi
  if [[ "${force_build}" != "1" && -e "${default_exe}" ]]; then
    log "WARNING: Default TurbSim executable exists but failed its startup check: ${default_exe}"
  fi
  if [[ "${AUTO_SETUP_TURBSIM:-1}" != "1" ]]; then
    echo "ERROR: GENERATE_TURBSIM=1 but no usable TurbSim executable was found. Set TURBSIM_EXE or enable AUTO_SETUP_TURBSIM=1." >&2
    exit 2
  fi
  local installer="${VORLAP_ROOT}/scripts/install_openfast_turbsim.sh"
  [[ -x "${installer}" ]] || {
    echo "ERROR: TurbSim installer not found or not executable: ${installer}" >&2
    exit 2
  }
  if [[ -z "${TURBSIM_EXE:-}" || "${TURBSIM_EXE}" != */* ]]; then
    TURBSIM_EXE="${default_exe}"
  fi
  export TURBSIM_EXE
  log "TurbSim is not usable; running automatic OpenFAST setup/repair."
  "${installer}"
  turbsim_command_works "${TURBSIM_EXE}" || {
    echo "ERROR: TurbSim installer did not produce a usable executable: ${TURBSIM_EXE}" >&2
    exit 2
  }
  log "Automatic TurbSim setup/repair succeeded: ${TURBSIM_EXE}"
}

setup_python_if_needed() {
  [[ "${AUTO_SETUP_FIRST_RUN:-0}" == "1" || "${FORCE_PYTHON_SETUP:-0}" == "1" ]] || return 0
  [[ "${DRY_RUN_ONLY:-0}" == "1" ]] && return 0
  [[ "${SUBMIT_WIND_JOBS_DRY_RUN:-0}" == "1" ]] && return 0
  [[ "${ARRAY_CASE_DRY_RUN:-0}" == "1" ]] && return 0
  local helper="${VORLAP_ROOT}/run_headless/python_env.sh"
  local setup_state_dir="${VORLAP_ROOT}/.vorlap_setup"
  local force_marker="${setup_state_dir}/python-force-token"
  local force_token="${VORLAP_FORCE_SETUP_TOKEN:-}"
  local force_setup_performed=0
  [[ -r "${helper}" ]] || {
    echo "ERROR: Python setup helper not found: ${helper}" >&2
    exit 2
  }
  mkdir -p "${setup_state_dir}"
  if command -v flock >/dev/null 2>&1; then
    exec 8>"${setup_state_dir}/python-setup.lock"
    log "Waiting for exclusive Python setup lock: ${setup_state_dir}/python-setup.lock"
    flock 8
  fi
  VORLAP_DIR="${VORLAP_ROOT}"
  export VORLAP_DIR AUTO_SETUP_PYTHON
  # shellcheck source=run_headless/python_env.sh
  source "${helper}"
  if [[ "${FORCE_PYTHON_SETUP:-0}" == "1" ]]; then
    if [[ -n "${force_token}" && -f "${force_marker}" && "$(<"${force_marker}")" == "${force_token}" ]]; then
      log "Forced Python setup already completed for job token ${force_token}; reusing it."
    else
      log "Forced VorLap Python environment reinstall/update requested."
      vorlap_setup_python_environment PYTHON_EXE
      force_setup_performed=1
    fi
  fi
  ensure_vorlap_python_environment \
    PYTHON_EXE \
    "VorLap parked campaign" \
    "Set AUTO_SETUP_FIRST_RUN=0 to disable first-run dependency installation." \
    1 \
    numpy scipy pandas h5py plotly matplotlib
  if [[ "${force_setup_performed}" == "1" && -n "${force_token}" ]]; then
    printf '%s\n' "${force_token}" > "${force_marker}"
  fi
  if command -v flock >/dev/null 2>&1; then
    flock -u 8
    exec 8>&-
  fi
  export PYTHON_EXE
}

load_requested_modules() {
  local modules="$1"
  local init_script module_name
  [[ -n "${modules}" ]] || return 0

  if ! type module >/dev/null 2>&1; then
    for init_script in \
      /etc/profile.d/modules.sh \
      /usr/share/Modules/init/bash \
      /usr/share/lmod/lmod/init/bash \
      /opt/cray/pe/modules/default/init/bash; do
      if [[ -r "${init_script}" ]]; then
        # shellcheck source=/dev/null
        source "${init_script}"
        break
      fi
    done
  fi

  if ! type module >/dev/null 2>&1; then
    log "Requested module(s) not loaded because the module command is unavailable: ${modules}"
    return 0
  fi

  modules="${modules//,/ }"
  for module_name in ${modules}; do
    [[ -n "${module_name}" ]] || continue
    log "Loading module ${module_name}."
    module load "${module_name}"
  done
}

should_load_requested_modules() {
  [[ "${SKIP_VORLAP_MODULE_LOAD:-0}" == "1" ]] && return 1
  [[ -n "${VORLAP_SLURM_MODULES:-}" ]] || return 1
  [[ "${LOAD_VORLAP_MODULES:-0}" == "1" ]] && return 0
  [[ -n "${SLURM_JOB_ID:-}" ]] && return 0
  [[ "${SUBMIT_WIND_JOBS:-0}" == "1" ]] && return 0
  return 1
}

detect_thread_count() {
  local var value
  for var in SLURM_CPUS_ON_NODE SLURM_JOB_CPUS_PER_NODE SLURM_CPUS_PER_TASK PBS_NP NSLOTS LSB_DJOB_NUMPROC; do
    value="${!var:-}"
    if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )); then
      printf '%s\n' "${value}"
      return
    fi
    if [[ "${value}" =~ ^([0-9]+) ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}"
      return
    fi
  done
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    value="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )); then
      printf '%s\n' "${value}"
      return
    fi
  fi
  printf '1\n'
}

duration_seconds() {
  local minutes="${QBLADE_DURATION_MINUTES:-${SIM_MINUTES:-}}"
  if [[ -n "${minutes}" ]]; then
    "${PYTHON_EXE}" -c 'import sys; print(f"{float(sys.argv[1]) * 60.0:.12g}")' "${minutes}"
    return
  fi
  printf '%s\n' "${QBLADE_DURATION:-660}"
}

speed_label() {
  "${PYTHON_EXE}" -c 'import sys; print(("U%05.2fms" % float(sys.argv[1])).replace(".", "p").replace("-", "m"))' "$1"
}

wind_speed_list() {
  "${PYTHON_EXE}" -c 'import sys
sys.path.insert(0, sys.argv[1])
import run_parked_demo_campaign as campaign
for speed in campaign.parse_wind_speeds(sys.argv[2]):
    print(f"{speed:.12g}")
' "${SCRIPT_DIR}" "${CAMPAIGN_WIND_SPEEDS}"
}

is_array_task() {
  [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]
}

is_array_leader() {
  if ! is_array_task; then
    return 0
  fi
  [[ "${SLURM_ARRAY_TASK_ID}" == "${SLURM_ARRAY_TASK_MIN:-0}" ]]
}

OWENS_PREPROCESS_DONE=0

run_owens_preprocess_once() {
  local mode="${RUN_OWENS_PREPROCESS:-auto}"
  local mode_lower
  mode_lower="$(printf '%s' "${mode}" | tr '[:upper:]' '[:lower:]')"
  case "${mode_lower}" in
    0|false|off|no) return 0 ;;
    1|true|on|yes|auto) ;;
    *) die "RUN_OWENS_PREPROCESS must be auto, 0, or 1 (Boolean aliases are accepted)." ;;
  esac
  [[ "${OWENS_PREPROCESS_DONE}" == "0" ]] || return 0
  is_array_leader || return 0
  [[ "${DRY_RUN_ONLY:-0}" != "1" ]] || return 0
  [[ "${SUBMIT_WIND_JOBS_DRY_RUN:-0}" != "1" ]] || return 0
  [[ "${ARRAY_CASE_DRY_RUN:-0}" != "1" ]] || return 0
  [[ "${ONLY_POSTPROCESS:-0}" != "1" ]] || return 0
  [[ "${PAPER_CHECKS_ONLY:-0}" != "1" ]] || return 0
  [[ "${WIND_ONLY:-0}" != "1" ]] || return 0

  local sim_file="${OWENS_QBLADE_SIM:-${VORLAP_ROOT}/examples/ReferenceHVAWTQBladeModel/baseline_10m_HVAWT_reference_turbine_Wwnd.sim}"
  local output_dir="${OWENS_OUTPUT_DIR:-${RUN_ROOT}/owens_modal}"
  local converter="${SCRIPT_DIR}/qblade_to_owens.py"
  if [[ ! -f "${sim_file}" || ! -f "${converter}" ]]; then
    if [[ "${mode_lower}" == "auto" ]]; then
      log "Skipping automatic OWENS preprocessing because its QBlade source model or converter is unavailable."
      OWENS_PREPROCESS_DONE=1
      return 0
    fi
    [[ -f "${sim_file}" ]] || die "OWENS QBlade source model not found: ${sim_file}"
    die "OWENS converter not found: ${converter}"
  fi

  local command=(
    "${PYTHON_EXE}" "${converter}" run
    --sim "${sim_file}"
    --output-dir "${output_dir}"
    --rpms "${OWENS_CAMPBELL_RPMS:-0,0.25,0.5,0.75,1,1.5,2,3,5,7.5,10}"
    --modes "${OWENS_CAMPBELL_MODES:-12}"
    --save-rpms "${OWENS_SAVE_RPMS:-0,1}"
    --save-modes "${OWENS_SAVE_MODES:-1,2,3,4,5,6,7,8,9,10,11,12}"
    --vtk-modes "${OWENS_VTK_MODES:-6,12}"
    --mode-scaling "${OWENS_MODE_SCALING:-3}"
    --vtk-steps "${OWENS_VTK_STEPS:-36}"
  )
  [[ -z "${OWENS_ROOT:-}" ]] || command+=(--owens-root "${OWENS_ROOT}")
  [[ -z "${JULIA:-}" ]] || command+=(--julia "${JULIA}")
  if [[ "${mode_lower}" == "auto" ]]; then
    command+=(--if-available)
  fi
  if [[ "${FORCE_OWENS_PREPROCESS:-0}" != "1" ]]; then
    command+=(--reuse-existing)
  fi
  if [[ "${OWENS_NO_GRAVITY:-0}" == "1" ]]; then
    command+=(--no-gravity)
  fi

  mkdir -p "${output_dir}/.matplotlib"
  log "Running one-time QBlade-to-OWENS structural preprocessing into ${output_dir}."
  MPLCONFIGDIR="${output_dir}/.matplotlib" "${command[@]}"
  OWENS_PREPROCESS_DONE=1
}

if ! SUBMIT_SCRIPT="$(resolve_submit_script)"; then
  echo "ERROR: Could not locate the real VorLap run_headless directory. If Slurm copied this script to /var/spool, submit it from the VorLap checkout or export VORLAP_SUBMIT_SCRIPT=/absolute/path/to/VorLap/run_headless/submit_parked_demo_campaign.sj." >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${SUBMIT_SCRIPT}")" && pwd -P)"
VORLAP_ROOT="${VORLAP_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PAPER_DIR="${PAPER_DIR:-${VORLAP_ROOT}/paper_drafts/NAWEA_QBladeBridge}"
PAPER_EVIDENCE_MATRIX="${PAPER_EVIDENCE_MATRIX:-${PAPER_DIR}/analysis/evidence_matrix.json}"
PAPER_TEX="${PAPER_TEX:-${PAPER_DIR}/NAWEA2026_QBladeBridge.tex}"
QBLADE_ROOT="${QBLADE_ROOT:-${SCRIPT_DIR}/QBladeCE_2.0.9.7}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/vorlap_parked_demo}"
AUDIT_ROOT="${AUDIT_ROOT:-${RUN_ROOT}/audits}"
VORLAP_AUTO_NODE_THREADS="${VORLAP_AUTO_NODE_THREADS:-1}"
if [[ "${VORLAP_AUTO_NODE_THREADS}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
  # Recompute inside every submitted job. This deliberately replaces any
  # OMP_NUM_THREADS value inherited from the login shell by --export=ALL.
  OMP_NUM_THREADS="$(detect_thread_count)"
elif [[ -z "${OMP_NUM_THREADS:-}" && "${SUBMIT_WIND_JOBS:-0}" != "1" ]]; then
  OMP_NUM_THREADS="$(detect_thread_count)"
fi
if [[ -n "${OMP_NUM_THREADS:-}" ]]; then
  export OMP_NUM_THREADS
fi
PRODUCTION_WIND_SPEEDS="${PRODUCTION_WIND_SPEEDS:-5 10 30 50}"
PRODUCTION_AZIMUTHS="${PRODUCTION_AZIMUTHS:-0}"
PRODUCTION_RPM="${PRODUCTION_RPM:-1}"
QBLADE_HORIZONTAL_INFLOW_ANGLE="${QBLADE_HORIZONTAL_INFLOW_ANGLE:-0.0}"
QBLADE_VERTICAL_INFLOW_ANGLE="${QBLADE_VERTICAL_INFLOW_ANGLE:-0.0}"
QBLADE_SHEAR_EXPONENT="${QBLADE_SHEAR_EXPONENT:-0.23}"
QBLADE_REFERENCE_HEIGHT="${QBLADE_REFERENCE_HEIGHT:-18.0}"
QBLADE_AIR_DENSITY="${QBLADE_AIR_DENSITY:-1.0296}"
QBLADE_AIR_KINEMATIC_VISCOSITY="${QBLADE_AIR_KINEMATIC_VISCOSITY:-1.647e-5}"
TURBSIM_REF_HEIGHT="${TURBSIM_REF_HEIGHT:-${QBLADE_REFERENCE_HEIGHT}}"
export QBLADE_HORIZONTAL_INFLOW_ANGLE QBLADE_VERTICAL_INFLOW_ANGLE QBLADE_SHEAR_EXPONENT
export QBLADE_REFERENCE_HEIGHT QBLADE_AIR_DENSITY QBLADE_AIR_KINEMATIC_VISCOSITY TURBSIM_REF_HEIGHT
CAMPAIGN_SCENARIOS="${CAMPAIGN_SCENARIOS:-baseline vorlap}"
CAMPAIGN_WIND_SPEEDS="${WIND_SPEEDS:-${PRODUCTION_WIND_SPEEDS}}"
CAMPAIGN_AZIMUTHS="${QBLADE_AZIMUTHS:-${PRODUCTION_AZIMUTHS}}"
CAMPAIGN_RPM="${QBLADE_RPM:-${PRODUCTION_RPM}}"
RUN_OWENS_PREPROCESS="${RUN_OWENS_PREPROCESS:-auto}"
VORLAP_SLURM_MODULES="${VORLAP_SLURM_MODULES-anaconda3/2025.12-2}"
VORLAP_PIP_TRUSTED_HOSTS="${VORLAP_PIP_TRUSTED_HOSTS-pypi.org files.pythonhosted.org pypi.python.org}"
VORLAP_PIP_NO_BUILD_ISOLATION="${VORLAP_PIP_NO_BUILD_ISOLATION:-1}"
SKIP_VORLAP_INSTALL="${SKIP_VORLAP_INSTALL:-1}"
VORLAP_SOURCE_BOUNDS_POLICY="${VORLAP_SOURCE_BOUNDS_POLICY:-warn}"
VORLAP_SOURCE_FREQUENCY_TRUNCATION="${VORLAP_SOURCE_FREQUENCY_TRUNCATION:-none}"
QBLADE_WAKE_PROFILE="${QBLADE_WAKE_PROFILE:-reference}"
QBLADE_RESULT_FILTER="${QBLADE_RESULT_FILTER:-}"
export VORLAP_SLURM_MODULES VORLAP_PIP_TRUSTED_HOSTS VORLAP_PIP_NO_BUILD_ISOLATION SKIP_VORLAP_INSTALL VORLAP_SOURCE_BOUNDS_POLICY VORLAP_SOURCE_FREQUENCY_TRUNCATION QBLADE_WAKE_PROFILE QBLADE_RESULT_FILTER
if [[ -n "${VORLAP_PIP_TRUSTED_HOSTS}" ]]; then
  PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-${VORLAP_PIP_TRUSTED_HOSTS//,/ }}"
  export PIP_TRUSTED_HOST
fi
if [[ -n "${AUDIT_EXPECTED_RPM:-}" ]]; then
  EXPECTED_AUDIT_RPM="${AUDIT_EXPECTED_RPM}"
elif [[ "${REQUIRE_PRODUCTION_MATRIX:-1}" == "1" ]]; then
  EXPECTED_AUDIT_RPM="${PRODUCTION_RPM}"
else
  EXPECTED_AUDIT_RPM="${CAMPAIGN_RPM}"
fi

if should_load_requested_modules; then
  load_requested_modules "${VORLAP_SLURM_MODULES}"
fi

if [[ -z "${PYTHON_EXE:-}" && -x "${VORLAP_ROOT}/.venv/bin/python" ]]; then
  PYTHON_EXE="${VORLAP_ROOT}/.venv/bin/python"
else
  PYTHON_EXE="${PYTHON_EXE:-python3}"
fi
prepare_serialized_force_setup
setup_python_if_needed
QBLADE_DURATION_SECONDS="$(duration_seconds)"
if [[ -n "${QBLADE_STORE_FROM:-}" ]]; then
  QBLADE_STORE_FROM_SECONDS="${QBLADE_STORE_FROM}"
elif [[ "${REQUIRE_PRODUCTION_MATRIX:-1}" == "1" ]]; then
  QBLADE_STORE_FROM_SECONDS="60"
else
  QBLADE_STORE_FROM_SECONDS="0"
fi
if [[ -n "${AUDIT_MIN_ANALYZED_DURATION:-}" ]]; then
  AUDIT_MIN_ANALYZED_DURATION_SECONDS="${AUDIT_MIN_ANALYZED_DURATION}"
elif [[ "${REQUIRE_PRODUCTION_MATRIX:-1}" == "1" ]]; then
  AUDIT_MIN_ANALYZED_DURATION_SECONDS="600"
else
  AUDIT_MIN_ANALYZED_DURATION_SECONDS="0"
fi

if qblade_root_preflight_needed; then
  validate_qblade_root_preflight "${QBLADE_ROOT}"
fi
setup_turbsim_if_needed

if [[ "${SUBMIT_WIND_JOBS:-0}" == "1" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" && "${ALLOW_NESTED_SUBMIT:-0}" != "1" ]]; then
    echo "ERROR: SUBMIT_WIND_JOBS=1 should be run from a login shell, not from inside Slurm job ${SLURM_JOB_ID}. Set ALLOW_NESTED_SUBMIT=1 to override." >&2
    exit 2
  fi
  submitted=0
  wind_job_time="${SUBMIT_WIND_JOB_TIME:-}"
  while IFS= read -r wind_speed; do
    [[ -n "${wind_speed}" ]] || continue
    label="$(speed_label "${wind_speed}")"
    job_run_root="${RUN_ROOT}/${label}"
    export_arg="ALL,SUBMIT_WIND_JOBS=0,WIND_SPEEDS=${wind_speed},PRODUCTION_WIND_SPEEDS=${wind_speed},RUN_ROOT=${job_run_root},QBLADE_DURATION=${QBLADE_DURATION_SECONDS},VORLAP_SUBMIT_SCRIPT=${SUBMIT_SCRIPT},VORLAP_ROOT=${VORLAP_ROOT},QBLADE_ROOT=${QBLADE_ROOT},SKIP_VORLAP_INSTALL=${SKIP_VORLAP_INSTALL},VORLAP_SOURCE_BOUNDS_POLICY=${VORLAP_SOURCE_BOUNDS_POLICY},VORLAP_SOURCE_FREQUENCY_TRUNCATION=${VORLAP_SOURCE_FREQUENCY_TRUNCATION},QBLADE_WAKE_PROFILE=${QBLADE_WAKE_PROFILE},VORLAP_AUTO_NODE_THREADS=${VORLAP_AUTO_NODE_THREADS}"
    log "Submitting wind speed ${wind_speed} m/s as ${label}; run root: ${job_run_root}"
    submit_args=(--job-name "${SBATCH_JOB_NAME_PREFIX:-vorlap-qblade}-${label}")
    if [[ -n "${wind_job_time}" ]]; then
      submit_args+=(--time "${wind_job_time}")
    fi
    submit_args+=(--export="${export_arg}" "${SUBMIT_SCRIPT}")
    if [[ "${SUBMIT_WIND_JOBS_DRY_RUN:-0}" == "1" ]]; then
      printf 'sbatch'
      printf ' %q' "${submit_args[@]}"
      printf '\n'
    else
      sbatch "${submit_args[@]}"
    fi
    submitted=$((submitted + 1))
  done <<< "$(wind_speed_list)"
  log "Submitted ${submitted} wind-speed job(s)."
  exit 0
fi

log "QBlade CPU policy: OMP_NUM_THREADS=${OMP_NUM_THREADS}; CASE_WORKERS=${CASE_WORKERS:-1}."

run_flags=()
if [[ -n "${QBLADE_STEPS:-}" ]]; then
  run_flags+=(--qblade-steps "${QBLADE_STEPS}")
fi
if [[ "${FORCE_BRIDGE_BUILD:-0}" == "1" ]]; then
  run_flags+=(--force-bridge-build)
fi
if [[ "${SKIP_VORLAP_INSTALL}" == "1" ]]; then
  run_flags+=(--skip-vorlap-install)
fi
if [[ "${SKIP_BRIDGE_BUILD:-0}" == "1" ]]; then
  run_flags+=(--skip-bridge-build)
fi
if [[ -n "${QBLADE_WIND_NAME:-}" ]]; then
  run_flags+=(--wind-name "${QBLADE_WIND_NAME}")
fi
if [[ "${GENERATE_TURBSIM:-0}" == "1" ]]; then
  run_flags+=(--generate-turbsim)
fi
if [[ "${FORCE_GENERATE_TURBSIM:-0}" == "1" ]]; then
  run_flags+=(--force-generate-turbsim)
fi
if [[ "${ADOPT_EXISTING_TURBSIM:-0}" == "1" ]]; then
  run_flags+=(--adopt-existing-turbsim)
fi
if [[ -n "${TURBSIM_EXE:-}" ]]; then
  run_flags+=(--turbsim-exe "${TURBSIM_EXE}")
fi
if [[ -n "${TURBSIM_TEMPLATE:-}" ]]; then
  run_flags+=(--turbsim-template "${TURBSIM_TEMPLATE}")
fi
if [[ -n "${TURBSIM_WIND_ROOT:-}" ]]; then
  run_flags+=(--wind-root "${TURBSIM_WIND_ROOT}")
fi
if [[ -n "${TURBSIM_TIMESTEP:-}" ]]; then
  run_flags+=(--turbsim-time-step "${TURBSIM_TIMESTEP}")
fi
if [[ -n "${TURBSIM_ANALYSIS_TIME:-}" ]]; then
  run_flags+=(--turbsim-analysis-time "${TURBSIM_ANALYSIS_TIME}")
fi
if [[ -n "${TURBSIM_USABLE_TIME:-}" ]]; then
  run_flags+=(--turbsim-usable-time "${TURBSIM_USABLE_TIME}")
fi
if [[ -n "${TURBSIM_NUM_GRID_Y:-}" ]]; then
  run_flags+=(--turbsim-num-grid-y "${TURBSIM_NUM_GRID_Y}")
fi
if [[ -n "${TURBSIM_NUM_GRID_Z:-}" ]]; then
  run_flags+=(--turbsim-num-grid-z "${TURBSIM_NUM_GRID_Z}")
fi
if [[ -n "${TURBSIM_GRID_WIDTH:-}" ]]; then
  run_flags+=(--turbsim-grid-width "${TURBSIM_GRID_WIDTH}")
fi
if [[ -n "${TURBSIM_GRID_HEIGHT:-}" ]]; then
  run_flags+=(--turbsim-grid-height "${TURBSIM_GRID_HEIGHT}")
fi
if [[ -n "${TURBSIM_HUB_HEIGHT:-}" ]]; then
  run_flags+=(--turbsim-hub-height "${TURBSIM_HUB_HEIGHT}")
fi
if [[ -n "${TURBSIM_REF_HEIGHT:-}" ]]; then
  run_flags+=(--turbsim-ref-height "${TURBSIM_REF_HEIGHT}")
fi
if [[ -n "${TURBSIM_TURB_MODEL:-}" ]]; then
  run_flags+=(--turbsim-turb-model "${TURBSIM_TURB_MODEL}")
fi
if [[ -n "${TURBSIM_IEC_STANDARD:-}" ]]; then
  run_flags+=(--turbsim-iec-standard "${TURBSIM_IEC_STANDARD}")
fi
if [[ -n "${TURBSIM_IEC_TURBC:-}" ]]; then
  run_flags+=(--turbsim-iec-turbc "${TURBSIM_IEC_TURBC}")
fi
if [[ -n "${TURBSIM_IEC_WIND_TYPE:-}" ]]; then
  run_flags+=(--turbsim-iec-wind-type "${TURBSIM_IEC_WIND_TYPE}")
fi
if [[ -n "${TURBSIM_SEED_BASE:-}" ]]; then
  run_flags+=(--turbsim-seed-base "${TURBSIM_SEED_BASE}")
fi
if [[ -n "${TURBSIM_SEED2:-}" ]]; then
  run_flags+=(--turbsim-seed2 "${TURBSIM_SEED2}")
fi
if [[ "${ALLOW_COARSE_WIND_TIMESTEP:-0}" == "1" ]]; then
  run_flags+=(--allow-coarse-wind-timestep)
fi
if [[ "${WIND_ONLY:-0}" == "1" ]]; then
  run_flags+=(--wind-only)
fi
if [[ -n "${ULTIMATE_ALLOWABLES:-}" ]]; then
  run_flags+=(--ultimate-allowables "${ULTIMATE_ALLOWABLES}")
fi
if [[ -n "${POSTPROCESS_LOCATIONS:-}" ]]; then
  run_flags+=(--postprocess-locations "${POSTPROCESS_LOCATIONS}")
fi
if [[ -n "${POSTPROCESS_BLADE:-}" ]]; then
  run_flags+=(--postprocess-blade "${POSTPROCESS_BLADE}")
fi
if [[ -n "${WOHLER_EXPONENT:-}" ]]; then
  run_flags+=(--wohler-exponent "${WOHLER_EXPONENT}")
fi
if [[ -n "${POSTPROCESS_EQUIVALENT_CYCLES:-}" ]]; then
  run_flags+=(--equivalent-cycles "${POSTPROCESS_EQUIVALENT_CYCLES}")
fi
if [[ "${DIAGNOSTIC_FORCE_SCALE:-0}" == "1" ]]; then
  run_flags+=(--diagnostic-force-scale)
fi
if [[ "${ALLOW_PARTIAL:-0}" == "1" ]]; then
  run_flags+=(--allow-partial)
fi
if [[ "${NATIVE_RUN_FULL_SIMULATION:-0}" == "1" ]]; then
  run_flags+=(--native-run-full-simulation)
fi
if [[ "${CONTINUE_ON_FAILURE:-0}" == "1" ]]; then
  run_flags+=(--continue-on-failure)
fi
if [[ "${ALLOW_COARSE_TIMESTEP:-0}" == "1" ]]; then
  run_flags+=(--allow-coarse-timestep)
fi
if [[ "${SKIP_POSTPROCESS:-0}" == "1" ]]; then
  run_flags+=(--skip-postprocess)
fi
if [[ "${QBLADE_STORE_PROJECT:-0}" == "1" || "${QBLADE_STORE_PROJECT:-false}" == "true" ]]; then
  run_flags+=(--store-project)
fi
if [[ "${POSTPROCESS_INCLUDE_SMOKE_IN_DEL:-0}" == "1" || "${POSTPROCESS_INCLUDE_SMOKE_IN_DEL:-false}" == "true" ]]; then
  run_flags+=(--include-smoke-in-del)
fi
if [[ "${REUSE_SETUP:-0}" == "1" ]]; then
  run_flags+=(--reuse-setup)
fi
if [[ "${PRESERVE_COMPLETED_SCENARIOS:-0}" == "1" ]]; then
  run_flags+=(--preserve-completed-scenarios)
fi

if ! is_array_leader && {
  [[ "${DRY_RUN_ONLY:-0}" == "1" ]] ||
  [[ "${SETUP_ONLY:-0}" == "1" ]] ||
  [[ "${ONLY_POSTPROCESS:-0}" == "1" ]] ||
  [[ "${PAPER_CHECKS_ONLY:-0}" == "1" ]];
}; then
  log "Skipping control-only mode on non-leader array task ${SLURM_ARRAY_TASK_ID}."
  exit 0
fi

effective_ultimate_allowables="${ULTIMATE_ALLOWABLES:-}"
if [[ -z "${effective_ultimate_allowables}" && -f "${RUN_ROOT}/campaign_settings.json" ]]; then
  effective_ultimate_allowables="$("${PYTHON_EXE}" -c 'import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    value = json.loads(path.read_text(encoding="utf-8")).get("ultimate_allowables") or ""
except Exception:
    value = ""
print(value)
' "${RUN_ROOT}/campaign_settings.json")"
fi
if [[ "${REQUIRE_ULTIMATE_MARGINS:-0}" == "1" ]]; then
  if [[ -z "${effective_ultimate_allowables}" ]]; then
    echo "ERROR: REQUIRE_ULTIMATE_MARGINS=1 requires ULTIMATE_ALLOWABLES or an archived campaign_settings.json ultimate_allowables path." >&2
    exit 2
  fi
  if [[ ! -f "${effective_ultimate_allowables}" ]]; then
    echo "ERROR: ultimate allowables file not found: ${effective_ultimate_allowables}" >&2
    exit 2
  fi
fi

campaign_cmd=(
  "${PYTHON_EXE}" "${SCRIPT_DIR}/run_parked_demo_campaign.py"
  --run-root "${RUN_ROOT}"
  --qblade-root "${QBLADE_ROOT}"
  --vorlap-root "${VORLAP_ROOT}"
  --python "${PYTHON_EXE}"
  --wind-speeds "${CAMPAIGN_WIND_SPEEDS}"
  --azimuths "${CAMPAIGN_AZIMUTHS}"
  --duration "${QBLADE_DURATION_SECONDS}"
  --timestep "${QBLADE_TIMESTEP:-0.0075}"
  --scenarios "${CAMPAIGN_SCENARIOS}"
  --rpm "${CAMPAIGN_RPM}"
  --store-from "${QBLADE_STORE_FROM_SECONDS}"
  --store-replay "${QBLADE_STORE_REPLAY:-false}"
  --checkpoint-every "${QBLADE_CHECKPOINT_EVERY:-4000}"
  --threads "${OMP_NUM_THREADS}"
  --case-workers "${CASE_WORKERS:-1}"
  --progress-every "${QBLADE_PROGRESS_EVERY:-10}"
  --coupling "${VORLAP_QBLADE_COUPLING:-sil-vorlap}"
  --identity-coupling "${VORLAP_QBLADE_IDENTITY_COUPLING:-sil-vorlap}"
  --vorlap-force-scale "${VORLAP_FORCE_SCALE:-1.0}"
  --source-bounds-policy "${VORLAP_SOURCE_BOUNDS_POLICY}"
  --source-frequency-truncation "${VORLAP_SOURCE_FREQUENCY_TRUNCATION}"
  --wake-profile "${QBLADE_WAKE_PROFILE}"
)
if [[ -n "${QBLADE_RESULT_FILTER}" ]]; then
  campaign_cmd+=(--result-filter "${QBLADE_RESULT_FILTER}")
fi
if ((${#run_flags[@]})); then
  campaign_cmd+=("${run_flags[@]}")
fi

complete_serialized_force_setup

if is_array_task; then
  run_owens_preprocess_once
fi

if is_array_task &&
   [[ "${DRY_RUN_ONLY:-0}" != "1" ]] &&
   [[ "${SETUP_ONLY:-0}" != "1" ]] &&
   [[ "${ONLY_POSTPROCESS:-0}" != "1" ]] &&
   [[ "${PAPER_CHECKS_ONLY:-0}" != "1" ]]; then
  log "Running Slurm array case index ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST:-one allocated node}; OMP_NUM_THREADS=${OMP_NUM_THREADS}."
  if [[ "${ARRAY_CASE_DRY_RUN:-0}" == "1" ]]; then
    "${campaign_cmd[@]}" --case-index "${SLURM_ARRAY_TASK_ID}" --dry-run
  else
    "${campaign_cmd[@]}" --case-index "${SLURM_ARRAY_TASK_ID}"
  fi
  exit 0
fi

audit_common=(
  "${PYTHON_EXE}" "${SCRIPT_DIR}/audit_parked_campaign.py"
  --run-root "${RUN_ROOT}"
  --expected-rpm "${EXPECTED_AUDIT_RPM}"
  --min-duration "${AUDIT_MIN_DURATION:-${QBLADE_DURATION_SECONDS}}"
  --min-analyzed-duration "${AUDIT_MIN_ANALYZED_DURATION_SECONDS}"
  --scenarios "${CAMPAIGN_SCENARIOS}"
)
if [[ "${REQUIRE_WIND_NAME:-0}" == "1" ]]; then
  audit_common+=(--require-wind-name)
fi
if [[ "${REQUIRE_PRODUCTION_MATRIX:-1}" == "1" ]]; then
  audit_common+=(--require-wind-speeds "${PRODUCTION_WIND_SPEEDS}")
  audit_common+=(--require-azimuths "${PRODUCTION_AZIMUTHS}")
  audit_common+=(--require-airfoil-source-manifest)
fi
if [[ "${REQUIRE_DLC_MANIFEST_ROW:-0}" == "1" ]]; then
  audit_common+=(--require-dlc-manifest-row)
fi
if [[ "${REQUIRE_WIND_RECORD:-0}" == "1" ]]; then
  audit_common+=(--require-wind-record)
fi

run_audit() {
  local stage="$1"
  local output_name="$2"
  shift 2
  "${audit_common[@]}" --stage "${stage}" --out "${AUDIT_ROOT}/${output_name}.json" "$@"
}

run_paper_submission_checks() {
  if [[ "${RUN_PAPER_PREFLIGHT:-1}" == "1" || "${REQUIRE_PAPER_EVIDENCE_MATRIX_READY:-0}" == "1" ]]; then
    paper_matrix_flags=(--matrix "${PAPER_EVIDENCE_MATRIX}" --root "${VORLAP_ROOT}")
    if [[ "${REQUIRE_PAPER_EVIDENCE_MATRIX_READY:-0}" == "1" ]]; then
      paper_matrix_flags+=(--require-submission-ready)
    fi
    "${PYTHON_EXE}" "${PAPER_DIR}/analysis/check_evidence_matrix.py" "${paper_matrix_flags[@]}"
  fi
  if [[ "${REQUIRE_PLACEHOLDER_FREE_SUBMISSION:-0}" == "1" ]]; then
    "${PYTHON_EXE}" "${PAPER_DIR}/analysis/check_submission_ready.py" --tex "${PAPER_TEX}"
  fi
}

build_archive_manifest() {
  if [[ "${GENERATE_PRODUCTION_ARCHIVE_MANIFEST:-${REQUIRE_PRODUCTION_ARCHIVE_MANIFEST:-0}}" == "1" ]]; then
    "${PYTHON_EXE}" "${SCRIPT_DIR}/build_production_archive_manifest.py" --run-root "${RUN_ROOT}"
  fi
}

build_signed_channel_selection() {
  if [[ "${GENERATE_SIGNED_CHANNEL_SELECTION:-0}" == "1" && ! -f "${RUN_ROOT}/signed_channel_selection.json" ]]; then
    "${PYTHON_EXE}" "${SCRIPT_DIR}/build_signed_channel_selection.py" --run-root "${RUN_ROOT}"
  fi
}

build_ultimate_input_basis() {
  if [[ "${GENERATE_ULTIMATE_INPUT_BASIS:-0}" == "1" && ! -f "${RUN_ROOT}/ultimate_input_basis.json" ]]; then
    ultimate_basis_flags=()
    if [[ -n "${effective_ultimate_allowables}" ]]; then
      ultimate_basis_flags+=(--allowables "${effective_ultimate_allowables}")
    fi
    "${PYTHON_EXE}" "${SCRIPT_DIR}/build_ultimate_input_basis.py" --run-root "${RUN_ROOT}" "${ultimate_basis_flags[@]}"
  fi
}

build_spectral_psd_comparison() {
  if [[ "${GENERATE_SPECTRAL_PSD_COMPARISON:-0}" == "1" && ! -f "${RUN_ROOT}/postprocess/spectral_psd_comparison.csv" ]]; then
    spectral_psd_flags=()
    if [[ -n "${SPECTRAL_PSD_COMPARISON_SPEC:-}" ]]; then
      spectral_psd_flags+=(--source-spec "${SPECTRAL_PSD_COMPARISON_SPEC}")
    fi
    if [[ -n "${SPECTRAL_PSD_MAX_FREQUENCY_ERROR_HZ:-}" ]]; then
      spectral_psd_flags+=(--max-frequency-error-hz "${SPECTRAL_PSD_MAX_FREQUENCY_ERROR_HZ}")
    fi
    if [[ -n "${SPECTRAL_PSD_MIN_BAND_POWER_FRACTION:-}" ]]; then
      spectral_psd_flags+=(--min-band-power-fraction "${SPECTRAL_PSD_MIN_BAND_POWER_FRACTION}")
    fi
    if [[ -n "${SPECTRAL_PSD_BAND_TOLERANCE_HZ:-}" ]]; then
      spectral_psd_flags+=(--band-tolerance-hz "${SPECTRAL_PSD_BAND_TOLERANCE_HZ}")
    fi
    "${PYTHON_EXE}" "${SCRIPT_DIR}/build_spectral_psd_comparison.py" --run-root "${RUN_ROOT}" "${spectral_psd_flags[@]}"
  fi
}

build_structural_model_qualification() {
  if [[ "${GENERATE_STRUCTURAL_MODEL_QUALIFICATION:-0}" == "1" && ! -f "${RUN_ROOT}/structural_model_qualification.json" ]]; then
    structural_flags=()
    if [[ -n "${STRUCTURAL_MODEL_SCENARIO:-}" ]]; then
      structural_flags+=(--scenario "${STRUCTURAL_MODEL_SCENARIO}")
    fi
    if [[ -n "${STRUCTURAL_MODEL_ID:-}" ]]; then
      structural_flags+=(--model-id "${STRUCTURAL_MODEL_ID}")
    fi
    if [[ -n "${STRUCTURAL_MODEL_CHOICE:-}" ]]; then
      structural_flags+=(--model-choice "${STRUCTURAL_MODEL_CHOICE}")
    fi
    for frequency in ${STRUCTURAL_MODAL_FREQUENCIES:-}; do
      structural_flags+=(--modal-frequency-hz "${frequency}")
    done
    if [[ "${STRUCTURAL_MODEL_QUALIFIED:-0}" == "1" ]]; then
      structural_flags+=(--qualified)
    fi
    if [[ "${STRUCTURAL_BASELINE_IDENTITY_PASSED:-0}" == "1" ]]; then
      structural_flags+=(--baseline-identity-passed)
    fi
    "${PYTHON_EXE}" "${SCRIPT_DIR}/build_structural_model_qualification.py" --run-root "${RUN_ROOT}" "${structural_flags[@]}"
  fi
}

results_audit_flags=()
if [[ "${REQUIRE_SOURCE_COVERAGE:-1}" == "1" ]]; then
  results_audit_flags+=(--require-source-coverage)
fi
if [[ "${REQUIRE_GEOMETRY_POSITION_PROOF:-0}" == "1" ]]; then
  results_audit_flags+=(--require-geometry-position-proof)
fi
if [[ "${REQUIRE_FORCE_SIGN_PROBES:-${RUN_SIGN_PROBES:-${SIGN_PROBES_ONLY:-0}}}" == "1" ]]; then
  results_audit_flags+=(--require-force-sign-probes)
fi
if [[ "${REQUIRE_POSTPROCESS_ARTIFACTS:-1}" == "1" ]]; then
  results_audit_flags+=(--require-postprocess-artifacts)
fi
if [[ "${REQUIRE_BASELINE_IDENTITY_PROOF:-0}" == "1" ]]; then
  results_audit_flags+=(--require-baseline-identity-proof)
fi
if [[ "${REQUIRE_PRODUCTION_ARCHIVE_MANIFEST:-0}" == "1" ]]; then
  results_audit_flags+=(--require-production-archive-manifest)
fi
if [[ "${REQUIRE_SPECTRAL_FORCE_CONTENT:-0}" == "1" ]]; then
  results_audit_flags+=(--require-spectral-force-content)
fi
if [[ "${REQUIRE_ULTIMATE_MARGINS:-0}" == "1" ]]; then
  results_audit_flags+=(--require-ultimate-margins)
fi
if [[ "${REQUIRE_ULTIMATE_INPUT_BASIS:-0}" == "1" ]]; then
  results_audit_flags+=(--require-ultimate-input-basis)
fi
if [[ "${REQUIRE_SPECTRAL_PSD_COMPARISON:-0}" == "1" ]]; then
  results_audit_flags+=(--require-spectral-psd-comparison)
fi
if [[ "${REQUIRE_NATIVE_CALLBACK_PROOF:-0}" == "1" ]]; then
  results_audit_flags+=(--require-native-callback-proof)
fi
if [[ "${REQUIRE_STRUCTURAL_MODEL_QUALIFICATION:-0}" == "1" ]]; then
  results_audit_flags+=(--require-structural-model-qualification)
fi
if [[ "${REQUIRE_SIGNED_CHANNEL_SELECTION:-0}" == "1" ]]; then
  results_audit_flags+=(--require-signed-channel-selection)
fi
if [[ "${FAIL_ON_ULTIMATE_EXCEEDANCE:-0}" == "1" ]]; then
  results_audit_flags+=(--fail-on-ultimate-exceedance)
fi
if [[ "${ALLOW_SOURCE_RANGE_EXCURSIONS:-0}" == "1" ]]; then
  results_audit_flags+=(--allow-source-range-excursions)
fi

if [[ "${PAPER_CHECKS_ONLY:-0}" == "1" ]]; then
  run_paper_submission_checks
  exit 0
fi

if [[ "${ONLY_POSTPROCESS:-0}" == "1" ]]; then
  "${campaign_cmd[@]}" --only-postprocess
  if [[ "${RUN_FINAL_AUDIT:-1}" == "1" ]]; then
    build_signed_channel_selection
    build_ultimate_input_basis
    build_spectral_psd_comparison
    build_structural_model_qualification
    build_archive_manifest
    if ((${#results_audit_flags[@]})); then
      run_audit results results_audit "${results_audit_flags[@]}"
    else
      run_audit results results_audit
    fi
    run_paper_submission_checks
  fi
  exit 0
fi

"${campaign_cmd[@]}" --dry-run

if [[ "${RUN_PLAN_AUDIT:-1}" == "1" ]]; then
  run_audit plan plan_audit
fi

if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${WIND_ONLY:-0}" == "1" ]]; then
  "${campaign_cmd[@]}"
  exit 0
fi

"${campaign_cmd[@]}" --setup-only

run_owens_preprocess_once

if [[ "${RUN_SETUP_AUDIT:-1}" == "1" ]]; then
  run_audit setup setup_audit
fi

if [[ "${SETUP_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${RUN_SIGN_PROBES:-${SIGN_PROBES_ONLY:-0}}" == "1" ]]; then
  sign_probe_flags=()
  if [[ -n "${SIGN_PROBE_CASE_LABEL:-}" ]]; then
    sign_probe_flags+=(--case-label "${SIGN_PROBE_CASE_LABEL}")
  else
    sign_probe_flags+=(--case-index "${SIGN_PROBE_CASE_INDEX:-0}")
  fi
  if [[ -n "${SIGN_PROBE_NODE:-}" ]]; then
    sign_probe_flags+=(--node "${SIGN_PROBE_NODE}")
  fi
  if [[ "${SIGN_PROBE_LOCAL:-0}" == "1" ]]; then
    sign_probe_flags+=(--local)
  fi
  sign_probe_cmd=(
    "${PYTHON_EXE}" "${SCRIPT_DIR}/run_force_sign_probe_campaign.py"
    --run-root "${RUN_ROOT}" \
    --scenario "${SIGN_PROBE_SCENARIO:-vorlap}" \
    --axes "${SIGN_PROBE_AXES:-X Y Z}" \
    --force "${SIGN_PROBE_FORCE:-1000}" \
    --steps "${SIGN_PROBE_STEPS:-200}" \
    --threads "${SIGN_PROBE_THREADS:-${OMP_NUM_THREADS}}" \
    --qblade-root "${QBLADE_ROOT}" \
    --vorlap-root "${VORLAP_ROOT}" \
    --python "${PYTHON_EXE}"
  )
  if ((${#sign_probe_flags[@]})); then
    sign_probe_cmd+=("${sign_probe_flags[@]}")
  fi
  "${sign_probe_cmd[@]}"
  sign_audit_flags=()
  if [[ -n "${SIGN_PROBE_REQUIRED_NODE:-${SIGN_PROBE_NODE:-}}" ]]; then
    sign_audit_flags+=(--force-sign-required-node "${SIGN_PROBE_REQUIRED_NODE:-${SIGN_PROBE_NODE}}")
  fi
  if ((${#sign_audit_flags[@]})); then
    run_audit sign-probes sign_probes_audit --scenarios "${SIGN_PROBE_SCENARIO:-vorlap}" "${sign_audit_flags[@]}"
  else
    run_audit sign-probes sign_probes_audit --scenarios "${SIGN_PROBE_SCENARIO:-vorlap}"
  fi
fi

if [[ "${SIGN_PROBES_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${REUSE_SETUP:-0}" == "1" ]]; then
  "${campaign_cmd[@]}"
else
  "${campaign_cmd[@]}" --reuse-setup
fi

if [[ "${RUN_FINAL_AUDIT:-1}" == "1" ]]; then
  build_signed_channel_selection
  build_ultimate_input_basis
  build_spectral_psd_comparison
  build_structural_model_qualification
  build_archive_manifest
  if ((${#results_audit_flags[@]})); then
    run_audit results results_audit "${results_audit_flags[@]}"
  else
    run_audit results results_audit
  fi
  run_paper_submission_checks
fi
