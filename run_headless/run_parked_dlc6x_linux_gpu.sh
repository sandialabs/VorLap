#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

detect_thread_count() {
  local var value
  for var in SLURM_CPUS_PER_TASK SLURM_CPUS_ON_NODE PBS_NP NSLOTS LSB_DJOB_NUMPROC; do
    value="${!var:-}"
    if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )); then
      printf '%s\n' "${value}"
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

append_flag_if_1() {
  local value="$1"
  local flag="$2"
  if [[ "${value}" == "1" ]]; then
    RUN_ARGS+=("${flag}")
  fi
}

check_python_environment() {
  local required_modules=(numpy scipy pandas h5py plotly)
  if [[ "${SKIP_POSTPROCESS}" != "1" && "${SETUP_ONLY}" != "1" && "${DRY_RUN_ONLY}" != "1" ]]; then
    required_modules+=(matplotlib)
  fi

  ensure_vorlap_python_environment \
    PYTHON_EXE \
    "parked QBlade/VorLap campaign" \
    "For a simulation-only campaign without diagnostic plots, set SKIP_POSTPROCESS=1." \
    1 \
    "${required_modules[@]}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORLAP_DIR="${VORLAP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=run_headless/python_env.sh
source "${SCRIPT_DIR}/python_env.sh"
RUN_ROOT_REL="${RUN_ROOT_REL:-run_headless/runs/parked_dlc6x_10min_linux_gpu}"
RUN_ROOT="${RUN_ROOT:-${VORLAP_DIR}/${RUN_ROOT_REL}}"
QBLADE_ROOT="${QBLADE_ROOT:-${QBLADE_ROOT_HOST:-${VORLAP_DIR}/run_headless/QBladeCE_2.0.9.7}}"
PYTHON_EXE="${PYTHON_EXE:-}"
if [[ -z "${PYTHON_EXE}" ]]; then
  if [[ -x "${VORLAP_DIR}/.venv/bin/python" ]]; then
    PYTHON_EXE="${VORLAP_DIR}/.venv/bin/python"
  else
    PYTHON_EXE="python3"
  fi
fi

[[ "$(uname -s)" == "Linux" ]] || die "This script is intended for Linux."
[[ -d "${VORLAP_DIR}" ]] || die "VorLap directory not found: ${VORLAP_DIR}"
[[ -d "${QBLADE_ROOT}" ]] || die "QBlade root not found: ${QBLADE_ROOT}"

export OMP_NUM_THREADS="${THREADS:-${OMP_NUM_THREADS:-$(detect_thread_count)}}"
export QBLADE_USE_OPENCL=1
export QBLADE_OPENCL_DEVICE="${OPENCL_DEVICE:-${QBLADE_OPENCL_DEVICE:-1}}"
export QBLADE_OPENCL_GROUP_SIZE="${OPENCL_GROUP_SIZE:-${QBLADE_OPENCL_GROUP_SIZE:-32}}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vorlap_matplotlib}"

WIND_SPEEDS="${WIND_SPEEDS:-5 10 30 50}"
QBLADE_AZIMUTHS="${QBLADE_AZIMUTHS:-0}"
QBLADE_RPM="${QBLADE_RPM:-1}"
QBLADE_DURATION="${QBLADE_DURATION:-660}"
QBLADE_TIMESTEP="${QBLADE_TIMESTEP:-0.0075}"
QBLADE_STORE_FROM="${QBLADE_STORE_FROM:-60}"
QBLADE_STORE_REPLAY="${QBLADE_STORE_REPLAY:-false}"
VORLAP_SOURCE_FREQUENCY_TRUNCATION="${VORLAP_SOURCE_FREQUENCY_TRUNCATION:-none}"
QBLADE_WAKE_PROFILE="${QBLADE_WAKE_PROFILE:-reference}"
QBLADE_RESULT_FILTER="${QBLADE_RESULT_FILTER:-}"
QBLADE_CHECKPOINT_EVERY="${QBLADE_CHECKPOINT_EVERY:-4000}"
QBLADE_PROGRESS_EVERY="${QBLADE_PROGRESS_EVERY:-1000}"
CASE_WORKERS="${CASE_WORKERS:-1}"
POSTPROCESS_LOCATIONS="${POSTPROCESS_LOCATIONS:-0.230 0.500 0.770 0.940}"
POSTPROCESS_BLADE="${POSTPROCESS_BLADE:-BLD_1}"
WOHLER_EXPONENT="${WOHLER_EXPONENT:-10}"
POSTPROCESS_MIN_FATIGUE_DURATION="${POSTPROCESS_MIN_FATIGUE_DURATION:-600}"

DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
SETUP_ONLY="${SETUP_ONLY:-0}"
ONLY_POSTPROCESS="${ONLY_POSTPROCESS:-0}"
SKIP_POSTPROCESS="${SKIP_POSTPROCESS:-0}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
RECORD_SIL_VORLAP_HISTORY="${RECORD_SIL_VORLAP_HISTORY:-0}"
REUSE_SETUP="${REUSE_SETUP:-1}"
FORCE_BRIDGE_BUILD="${FORCE_BRIDGE_BUILD:-0}"
SKIP_VORLAP_INSTALL="${SKIP_VORLAP_INSTALL:-1}"
SKIP_BRIDGE_BUILD="${SKIP_BRIDGE_BUILD:-1}"
GENERATE_TURBSIM="${GENERATE_TURBSIM:-0}"
FORCE_GENERATE_TURBSIM="${FORCE_GENERATE_TURBSIM:-0}"

RUN_ARGS=(
  "${PYTHON_EXE}" "${VORLAP_DIR}/run_headless/run_parked_demo_campaign.py"
  --run-root "${RUN_ROOT}"
  --qblade-root "${QBLADE_ROOT}"
  --vorlap-root "${VORLAP_DIR}"
  --python "${PYTHON_EXE}"
  --wind-speeds "${WIND_SPEEDS}"
  --azimuths "${QBLADE_AZIMUTHS}"
  --duration "${QBLADE_DURATION}"
  --timestep "${QBLADE_TIMESTEP}"
  --source-frequency-truncation "${VORLAP_SOURCE_FREQUENCY_TRUNCATION}"
  --wake-profile "${QBLADE_WAKE_PROFILE}"
  --rpm "${QBLADE_RPM}"
  --store-from "${QBLADE_STORE_FROM}"
  --store-replay "${QBLADE_STORE_REPLAY}"
  --checkpoint-every "${QBLADE_CHECKPOINT_EVERY}"
  --threads "${OMP_NUM_THREADS}"
  --case-workers "${CASE_WORKERS}"
  --progress-every "${QBLADE_PROGRESS_EVERY}"
  --postprocess-locations "${POSTPROCESS_LOCATIONS}"
  --postprocess-blade "${POSTPROCESS_BLADE}"
  --wohler-exponent "${WOHLER_EXPONENT}"
  --min-fatigue-duration "${POSTPROCESS_MIN_FATIGUE_DURATION}"
  --use-opencl
  --opencl-device "${QBLADE_OPENCL_DEVICE}"
  --opencl-group-size "${QBLADE_OPENCL_GROUP_SIZE}"
)

if [[ -n "${QBLADE_RESULT_FILTER}" ]]; then
  RUN_ARGS+=(--result-filter "${QBLADE_RESULT_FILTER}")
fi

append_flag_if_1 "${DRY_RUN_ONLY}" --dry-run
append_flag_if_1 "${SETUP_ONLY}" --setup-only
append_flag_if_1 "${ONLY_POSTPROCESS}" --only-postprocess
append_flag_if_1 "${SKIP_POSTPROCESS}" --skip-postprocess
append_flag_if_1 "${ALLOW_PARTIAL}" --allow-partial
append_flag_if_1 "${CONTINUE_ON_FAILURE}" --continue-on-failure
append_flag_if_1 "${RECORD_SIL_VORLAP_HISTORY}" --record-sil-vorlap-history
append_flag_if_1 "${REUSE_SETUP}" --reuse-setup
append_flag_if_1 "${FORCE_BRIDGE_BUILD}" --force-bridge-build
append_flag_if_1 "${SKIP_VORLAP_INSTALL}" --skip-vorlap-install
append_flag_if_1 "${SKIP_BRIDGE_BUILD}" --skip-bridge-build

if [[ -n "${ULTIMATE_ALLOWABLES:-}" ]]; then
  RUN_ARGS+=(--ultimate-allowables "${ULTIMATE_ALLOWABLES}")
fi

log "VorLap directory: ${VORLAP_DIR}"
log "Run root: ${RUN_ROOT}"
log "QBlade root: ${QBLADE_ROOT}"
log "Python: ${PYTHON_EXE}"
log "GPU/OpenCL device=${QBLADE_OPENCL_DEVICE}, group-size=${QBLADE_OPENCL_GROUP_SIZE}, OMP threads=${OMP_NUM_THREADS}"
log "Matrix: wind speeds='${WIND_SPEEDS}', initial azimuths='${QBLADE_AZIMUTHS}', rpm=${QBLADE_RPM}, duration=${QBLADE_DURATION}s, dt=${QBLADE_TIMESTEP}s"
log "Checkpoints every ${QBLADE_CHECKPOINT_EVERY} step(s); only the latest two checkpoint pairs are retained by the runner."
check_python_environment
ensure_vorlap_turbsim

append_flag_if_1 "${GENERATE_TURBSIM}" --generate-turbsim
append_flag_if_1 "${FORCE_GENERATE_TURBSIM}" --force-generate-turbsim
if [[ "${GENERATE_TURBSIM}" == "1" ]]; then
  [[ -n "${TURBSIM_EXE:-}" ]] && RUN_ARGS+=(--turbsim-exe "${TURBSIM_EXE}")
  [[ -n "${TURBSIM_TEMPLATE:-}" ]] && RUN_ARGS+=(--turbsim-template "${TURBSIM_TEMPLATE}")
  [[ -n "${TURBSIM_WIND_ROOT:-}" ]] && RUN_ARGS+=(--wind-root "${TURBSIM_WIND_ROOT}")
  [[ -n "${TURBSIM_TIMESTEP:-}" ]] && RUN_ARGS+=(--turbsim-time-step "${TURBSIM_TIMESTEP}")
  [[ -n "${TURBSIM_ANALYSIS_TIME:-}" ]] && RUN_ARGS+=(--turbsim-analysis-time "${TURBSIM_ANALYSIS_TIME}")
  [[ -n "${TURBSIM_USABLE_TIME:-}" ]] && RUN_ARGS+=(--turbsim-usable-time "${TURBSIM_USABLE_TIME}")
  [[ -n "${TURBSIM_NUM_GRID_Y:-}" ]] && RUN_ARGS+=(--turbsim-num-grid-y "${TURBSIM_NUM_GRID_Y}")
  [[ -n "${TURBSIM_NUM_GRID_Z:-}" ]] && RUN_ARGS+=(--turbsim-num-grid-z "${TURBSIM_NUM_GRID_Z}")
  [[ -n "${TURBSIM_GRID_WIDTH:-}" ]] && RUN_ARGS+=(--turbsim-grid-width "${TURBSIM_GRID_WIDTH}")
  [[ -n "${TURBSIM_GRID_HEIGHT:-}" ]] && RUN_ARGS+=(--turbsim-grid-height "${TURBSIM_GRID_HEIGHT}")
  [[ -n "${TURBSIM_HUB_HEIGHT:-}" ]] && RUN_ARGS+=(--turbsim-hub-height "${TURBSIM_HUB_HEIGHT}")
  [[ -n "${TURBSIM_REF_HEIGHT:-}" ]] && RUN_ARGS+=(--turbsim-ref-height "${TURBSIM_REF_HEIGHT}")
  [[ -n "${TURBSIM_TURB_MODEL:-}" ]] && RUN_ARGS+=(--turbsim-turb-model "${TURBSIM_TURB_MODEL}")
  [[ -n "${TURBSIM_IEC_STANDARD:-}" ]] && RUN_ARGS+=(--turbsim-iec-standard "${TURBSIM_IEC_STANDARD}")
  [[ -n "${TURBSIM_IEC_TURBC:-}" ]] && RUN_ARGS+=(--turbsim-iec-turbc "${TURBSIM_IEC_TURBC}")
  [[ -n "${TURBSIM_IEC_WIND_TYPE:-}" ]] && RUN_ARGS+=(--turbsim-iec-wind-type "${TURBSIM_IEC_WIND_TYPE}")
  [[ -n "${TURBSIM_SEED_BASE:-}" ]] && RUN_ARGS+=(--turbsim-seed-base "${TURBSIM_SEED_BASE}")
  [[ -n "${TURBSIM_SEED2:-}" ]] && RUN_ARGS+=(--turbsim-seed2 "${TURBSIM_SEED2}")
  append_flag_if_1 "${ALLOW_COARSE_WIND_TIMESTEP:-0}" --allow-coarse-wind-timestep
  log "TurbSim generation enabled; wind root='${TURBSIM_WIND_ROOT:-${RUN_ROOT}/wind_fields}', executable='${TURBSIM_EXE:-auto}'."
fi

"${RUN_ARGS[@]}"
