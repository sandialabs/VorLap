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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORLAP_DIR="${VORLAP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=run_headless/python_env.sh
source "${SCRIPT_DIR}/python_env.sh"
RUNNER_NAME="${RUNNER_NAME:-$(basename "${BASH_SOURCE[0]}")}"
BRIDGE_RUN_LABEL="${BRIDGE_RUN_LABEL:-sil}"
EXECUTION_BACKEND="${EXECUTION_BACKEND:-cpu}"
COUPLED_SCENARIO="${COUPLED_SCENARIO:-}"
COUPLED_COUPLING="${COUPLED_COUPLING:-sil-vorlap}"
COUPLED_RECORD_HISTORY="${COUPLED_RECORD_HISTORY:-1}"
NATIVE_RUN_FULL_SIMULATION="${NATIVE_RUN_FULL_SIMULATION:-0}"
FORCE_BRIDGE_BUILD="${FORCE_BRIDGE_BUILD:-0}"
SKIP_BRIDGE_BUILD="${SKIP_BRIDGE_BUILD:-1}"

if [[ -z "${COUPLED_SCENARIO}" ]]; then
  if [[ "${COUPLED_COUPLING}" == "qblade-external" ]]; then
    COUPLED_SCENARIO="qblade_driven_vorlap"
  else
    COUPLED_SCENARIO="sil_vorlap"
  fi
fi

RUN_ROOT_REL="${RUN_ROOT_REL:-run_headless/runs/codex_bridge_${BRIDGE_RUN_LABEL}_sample_U10p58_10s_linux_${EXECUTION_BACKEND}}"
RUN_ROOT_HOST="${VORLAP_DIR}/${RUN_ROOT_REL}"
LOG_DIR="${RUN_ROOT_HOST}/logs"
VERIFICATION_RUN_ROOT="${VERIFICATION_RUN_ROOT:-${VORLAP_DIR}/run_headless/runs/codex_bridge_oscillatory_only_U10p58_1s_docker}"
SUMMARY_BASENAME="${QBLADE_SUMMARY_BASENAME:-baseline_10m_HVAWT_reference_turbine_Wwnd}"
VERIFICATION_SUMMARY_BASENAME="${VERIFICATION_SUMMARY_BASENAME:-${SUMMARY_BASENAME}}"

QBLADE_ROOT_HOST="${QBLADE_ROOT_HOST:-${QBLADE_ROOT:-${VORLAP_DIR}/run_headless/QBladeCE_2.0.9.7}}"
PYTHON_HOST="${PYTHON_HOST:-}"
if [[ -z "${PYTHON_HOST}" ]]; then
  if [[ -x "${VORLAP_DIR}/.venv/bin/python" ]]; then
    PYTHON_HOST="${VORLAP_DIR}/.venv/bin/python"
  else
    PYTHON_HOST="python3"
  fi
fi

WIND_SPEED="${WIND_SPEED:-10.58}"
AZIMUTH="${AZIMUTH:-270}"
TIMESTEP="${TIMESTEP:-0.0075}"
TIMESTEPS="${TIMESTEPS:-1334}"
if [[ -z "${THREADS:-}" ]]; then
  if [[ "${EXECUTION_BACKEND}" == "gpu" ]]; then
    THREADS="16"
  else
    THREADS="$(detect_thread_count)"
  fi
fi
PROGRESS_EVERY="${PROGRESS_EVERY:-200}"
STORE_REPLAY="${STORE_REPLAY:-${QBLADE_STORE_REPLAY:-true}}"
if [[ "${NATIVE_RUN_FULL_SIMULATION}" == "1" ]]; then
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
else
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-${QBLADE_CHECKPOINT_EVERY:-200}}"
fi
RESUME_INITIALIZE="${RESUME_INITIALIZE:-0}"
OPENCL_DEVICE="${OPENCL_DEVICE:-1}"
OPENCL_GROUP_SIZE="${OPENCL_GROUP_SIZE:-32}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_COUPLED="${RUN_COUPLED:-${RUN_SIL:-1}}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_SETUP="${FORCE_SETUP:-1}"
if [[ -z "${GENERATE_PAPER:-}" ]]; then
  if [[ "${COUPLED_COUPLING}" == "sil-vorlap" ]]; then
    GENERATE_PAPER="1"
  else
    GENERATE_PAPER="0"
  fi
fi
COMPILE_PAPER="${COMPILE_PAPER:-1}"
REQUIRE_PAPER="${REQUIRE_PAPER:-0}"
POSTPROCESS_RESULTS="${POSTPROCESS_RESULTS:-1}"
POSTPROCESS_LOCATIONS="${POSTPROCESS_LOCATIONS:-0.230 0.500 0.770 0.940}"
POSTPROCESS_BLADE="${POSTPROCESS_BLADE:-BLD_1}"
POSTPROCESS_INCLUDE_SMOKE_IN_DEL="${POSTPROCESS_INCLUDE_SMOKE_IN_DEL:-1}"

[[ "$(uname -s)" == "Linux" ]] || die "This script is intended for a Linux machine."
if [[ "${EXECUTION_BACKEND}" == "gpu" ]]; then
  [[ "${USE_OPENCL:-1}" == "1" ]] || die "GPU runner requires USE_OPENCL=1."
else
  USE_OPENCL="${USE_OPENCL:-0}"
fi
[[ -d "${VORLAP_DIR}" ]] || die "VorLap directory not found: ${VORLAP_DIR}"
[[ -d "${VORLAP_DIR}/run_headless" ]] || die "run_headless directory not found under ${VORLAP_DIR}"
[[ -d "${QBLADE_ROOT_HOST}" ]] || die "QBlade root not found: ${QBLADE_ROOT_HOST}"
[[ -d "${QBLADE_ROOT_HOST}/SIL_Interface" ]] || die "QBlade SIL_Interface missing: ${QBLADE_ROOT_HOST}/SIL_Interface"
[[ -d "${QBLADE_ROOT_HOST}/ControllerFiles" ]] || die "QBlade ControllerFiles missing: ${QBLADE_ROOT_HOST}/ControllerFiles"
compgen -G "${QBLADE_ROOT_HOST}/libQBladeCE_*.so.1.0.0" >/dev/null || die "No QBlade shared library found in ${QBLADE_ROOT_HOST}"

mkdir -p "${LOG_DIR}"

prepend_path() {
  local var_name="$1"
  local value="$2"
  local current="${!var_name:-}"
  if [[ -n "${current}" ]]; then
    export "${var_name}=${value}:${current}"
  else
    export "${var_name}=${value}"
  fi
}

prepend_path PYTHONPATH "${VORLAP_DIR}"
prepend_path PYTHONPATH "${QBLADE_ROOT_HOST}/SIL_Interface"
prepend_path LD_LIBRARY_PATH "${QBLADE_ROOT_HOST}"
prepend_path LD_LIBRARY_PATH "${QBLADE_ROOT_HOST}/Libraries"
prepend_path LD_LIBRARY_PATH "${QBLADE_ROOT_HOST}/ControllerFiles"
prepend_path LD_LIBRARY_PATH "${QBLADE_ROOT_HOST}/Binaries"
prepend_path LD_LIBRARY_PATH "${QBLADE_ROOT_HOST}/Binaries/IPCBridge"
export OMP_NUM_THREADS="${THREADS}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/vorlap_matplotlib}"

summary_path() {
  local scenario="$1"
  printf '%s/%s/results/U10p58ms/%s_run_summary.json' "${RUN_ROOT_HOST}" "${scenario}" "${SUMMARY_BASENAME}"
}

is_completed() {
  local scenario="$1"
  local summary
  summary="$(summary_path "${scenario}")"
  [[ -f "${summary}" ]] || return 1
  "${PYTHON_HOST}" - "${summary}" "${TIMESTEPS}" <<'PY'
import json
import sys

path = sys.argv[1]
expected = int(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    summary = json.load(handle)
completed = int(summary.get("completed_steps") or 0)
requested = int(summary.get("requested_steps") or expected)
ok = completed >= min(expected, requested)
if summary.get("error"):
    ok = False
sys.exit(0 if ok else 1)
PY
}

case_field() {
  local scenario="$1"
  local field="$2"
  local manifest="${RUN_ROOT_HOST}/${scenario}/cases.json"
  [[ -f "${manifest}" ]] || die "Missing case manifest: ${manifest}"
  "${PYTHON_HOST}" - "${manifest}" "${field}" <<'PY'
import json
import sys

manifest_path, field = sys.argv[1:3]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
case = manifest["cases"][0]
print(case[field])
PY
}

check_python_environment() {
  local required_modules=(numpy scipy pandas h5py plotly)
  if [[ "${GENERATE_PAPER}" == "1" ]]; then
    required_modules+=(matplotlib)
  fi

  ensure_vorlap_python_environment \
    PYTHON_HOST \
    "native QBlade/VorLap run" \
    "For a simulation-only run without diagnostic plots, set GENERATE_PAPER=0." \
    1 \
    "${required_modules[@]}"
}

run_sample_postprocess() {
  [[ "${POSTPROCESS_RESULTS}" == "1" ]] || return 0

  ensure_vorlap_python_environment \
    PYTHON_HOST \
    "10 s sample postprocessing" \
    "Set GENERATE_PAPER=0 or POSTPROCESS_RESULTS=0 to skip postprocessing and diagnostic plot generation." \
    0 \
    numpy pandas matplotlib

  local postprocess_log="${LOG_DIR}/postprocess_10s_sample.log"
  local command=(
    "${PYTHON_HOST}" "${VORLAP_DIR}/run_headless/postprocess_parked_demo.py"
    --run-root "${RUN_ROOT_HOST}"
    --scenarios "baseline ${COUPLED_SCENARIO}"
    --comparison-baseline baseline
    --comparison-coupled "${COUPLED_SCENARIO}"
    --locations "${POSTPROCESS_LOCATIONS}"
    --blade "${POSTPROCESS_BLADE}"
  )
  if [[ "${POSTPROCESS_INCLUDE_SMOKE_IN_DEL}" == "1" ]]; then
    command+=(--include-smoke-in-del)
  fi

  log "Regenerating 10 s sample postprocess CSVs."
  if ! (cd "${VORLAP_DIR}" && "${command[@]}") 2>&1 | tee "${postprocess_log}"; then
    [[ "${REQUIRE_PAPER}" == "1" ]] && die "Cannot generate paper figures because sample postprocessing failed; see ${postprocess_log}."
    log "Skipping NAWEA sample diagnostics because sample postprocessing failed; see ${postprocess_log}."
    return 1
  fi
}

setup_scenario() {
  local scenario="$1"
  local force_scale="$2"
  local coupling="$3"
  local setup_log="${LOG_DIR}/${scenario}_setup.log"
  local setup_args=()
  if [[ "${FORCE_SETUP}" == "1" ]]; then
    setup_args+=(--force-setup)
  fi
  if [[ "${SKIP_BRIDGE_BUILD}" == "1" ]]; then
    setup_args+=(--skip-bridge-build)
  fi
  if [[ "${FORCE_BRIDGE_BUILD}" == "1" ]]; then
    setup_args+=(--force-bridge-build)
  fi

  log "Preparing ${scenario} case files."
  "${PYTHON_HOST}" "${VORLAP_DIR}/run_headless/vorlap_qblade_batch.py" setup \
    --run-root "${RUN_ROOT_HOST}/${scenario}" \
    --qblade-root "${QBLADE_ROOT_HOST}" \
    --vorlap-root "${VORLAP_DIR}" \
    --python "${PYTHON_HOST}" \
    --wind-speeds "${WIND_SPEED}" \
    --azimuths "${AZIMUTH}" \
    --timesteps "${TIMESTEPS}" \
    --timestep "${TIMESTEP}" \
    --store-from 0 \
    --store-replay "${STORE_REPLAY}" \
    --rpm 0 \
    --rampup 0 \
    --add-damp 0 \
    --force-scale "${force_scale}" \
    --threads "${THREADS}" \
    --progress-every "${PROGRESS_EVERY}" \
    --coupling "${coupling}" \
    --wind-type 0 \
    "${setup_args[@]}" \
    --skip-vorlap-install \
    2>&1 | tee "${setup_log}"
}

run_scenario() {
  local scenario="$1"
  local force_scale="$2"
  local coupling="$3"
  local record_history="$4"
  local run_log="${LOG_DIR}/${scenario}_10s_${EXECUTION_BACKEND}.log"

  if [[ "${SKIP_COMPLETED}" == "1" ]] && is_completed "${scenario}"; then
    log "Skipping ${scenario}; completed summary already exists at $(summary_path "${scenario}")."
    return
  fi

  setup_scenario "${scenario}" "${force_scale}" "${coupling}"

  local sim_path result_dir result_name metadata_path
  sim_path="$(case_field "${scenario}" sim)"
  result_dir="$(case_field "${scenario}" result_dir)"
  result_name="$(case_field "${scenario}" result_name)"
  metadata_path="${result_dir}/case_metadata.json"
  mkdir -p "${result_dir}"

  "${PYTHON_HOST}" - "${RUN_ROOT_HOST}/${scenario}/cases.json" "${metadata_path}" "${RUNNER_NAME}" <<'PY'
import json
import sys

manifest_path, metadata_path, runner_name = sys.argv[1:4]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
metadata = {
    "case": manifest["cases"][0],
    "manifest_settings": manifest.get("settings", {}),
    "manifest_created_unix": manifest.get("created_unix"),
    "runner": runner_name,
}
with open(metadata_path, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

  local command=(
    "${PYTHON_HOST}" "${VORLAP_DIR}/run_headless/run_qblade_sim.py"
    --qblade-root "${QBLADE_ROOT_HOST}"
    --sim "${sim_path}"
    --output-dir "${result_dir}"
    --threads "${THREADS}"
    --result-name "${result_name}"
    --coupling-mode "${coupling}"
    --case-metadata "${metadata_path}"
    --progress-every "${PROGRESS_EVERY}"
  )
  if [[ "${USE_OPENCL}" == "1" ]]; then
    command+=(--device "${OPENCL_DEVICE}" --group-size "${OPENCL_GROUP_SIZE}" --use-opencl)
  fi
  local use_native_full_simulation=0
  if [[ "${NATIVE_RUN_FULL_SIMULATION}" == "1" && "${coupling}" == "qblade-external" ]]; then
    use_native_full_simulation=1
  fi
  if [[ "${CHECKPOINT_EVERY}" != "0" && "${use_native_full_simulation}" != "1" ]]; then
    command+=(--checkpoint-every "${CHECKPOINT_EVERY}" --resume-from-checkpoint)
    if [[ "${RESUME_INITIALIZE}" == "1" ]]; then
      command+=(--resume-initialize)
    fi
  fi
  if [[ "${coupling}" == "sil-vorlap" ]]; then
    command+=(--sil-vorlap)
    if [[ "${record_history}" == "1" ]]; then
      command+=(--record-sil-vorlap-history)
    fi
  fi
  if [[ "${use_native_full_simulation}" == "1" ]]; then
    command+=(--native-run-full-simulation)
  fi

  if [[ "${USE_OPENCL}" == "1" ]]; then
    log "Starting GPU/OpenCL ${scenario}: U=${WIND_SPEED} m/s, azimuth=${AZIMUTH} deg, ${TIMESTEPS} steps at dt=${TIMESTEP} s."
    log "QBlade OpenCL device=${OPENCL_DEVICE}, group-size=${OPENCL_GROUP_SIZE}, OpenMP threads=${THREADS}. Writing log: ${run_log}"
  else
    log "Starting CPU/OpenMP ${scenario}: U=${WIND_SPEED} m/s, azimuth=${AZIMUTH} deg, ${TIMESTEPS} steps at dt=${TIMESTEP} s."
    log "QBlade/OpenMP threads=${THREADS}. Writing log: ${run_log}"
  fi
  "${command[@]}" 2>&1 | tee "${run_log}"

  is_completed "${scenario}" || die "${scenario} finished but no complete summary was found at $(summary_path "${scenario}")"
  log "Completed ${scenario}."
}

generate_paper_outputs() {
  [[ "${GENERATE_PAPER}" == "1" ]] || return 0
  local missing_sample=()
  local missing_verification=()
  local scenario summary
  for scenario in baseline "${COUPLED_SCENARIO}"; do
    summary="${VERIFICATION_RUN_ROOT}/${scenario}/results/U10p58ms/${VERIFICATION_SUMMARY_BASENAME}_run_summary.json"
    [[ -f "${summary}" ]] || missing_verification+=("${summary}")
  done
  for scenario in baseline "${COUPLED_SCENARIO}"; do
    summary="${RUN_ROOT_HOST}/${scenario}/results/U10p58ms/${SUMMARY_BASENAME}_run_summary.json"
    [[ -f "${summary}" ]] || missing_sample+=("${summary}")
  done
  if (( ${#missing_sample[@]} )); then
    log "Skipping NAWEA sample diagnostics; required sample summary missing: ${missing_sample[0]}"
    [[ "${REQUIRE_PAPER}" == "1" ]] && die "Cannot generate paper figures because ${#missing_sample[@]} required sample summary file(s) are missing."
    return
  fi
  if (( ${#missing_verification[@]} )); then
    log "Verification summaries are missing; sample diagnostics will still be regenerated."
    log "First missing verification summary: ${missing_verification[0]}"
    log "Set VERIFICATION_RUN_ROOT to an existing verification run to also regenerate the verification figure and paper PDF."
  fi

  log "Regenerating NAWEA bridge sample diagnostics."
  ensure_vorlap_python_environment \
    PYTHON_HOST \
    "NAWEA diagnostic plot generation" \
    "Set GENERATE_PAPER=0 to skip diagnostic plot generation." \
    0 \
    matplotlib
  run_sample_postprocess || return 0
  if ! (
    cd "${VORLAP_DIR}"
    VERIFICATION_RUN_ROOT="${VERIFICATION_RUN_ROOT}" SAMPLE_RUN_ROOT="${RUN_ROOT_HOST}" "${PYTHON_HOST}" paper_drafts/NAWEA_QBladeBridge/figure_generation/generate_bridge_figures.py
  ) 2>&1 | tee "${LOG_DIR}/generate_bridge_figures.log"; then
    [[ "${REQUIRE_PAPER}" == "1" ]] && die "Cannot generate paper figures; see ${LOG_DIR}/generate_bridge_figures.log."
    log "Skipping NAWEA bridge paper compile because figure generation failed; see ${LOG_DIR}/generate_bridge_figures.log."
    return
  fi

  if (( ${#missing_verification[@]} )); then
    [[ "${REQUIRE_PAPER}" == "1" ]] && die "Cannot compile paper because ${#missing_verification[@]} verification summary file(s) are missing."
    log "Skipping NAWEA bridge paper compile because verification summaries are missing."
    return
  fi

  [[ "${COMPILE_PAPER}" == "1" ]] || return 0
  log "Compiling NAWEA bridge paper PDF with latexmk -norc."
  (
    cd "${VORLAP_DIR}/paper_drafts/NAWEA_QBladeBridge"
    latexmk -norc -pdf -interaction=nonstopmode -halt-on-error NAWEA2026_QBladeBridge.tex
  ) 2>&1 | tee "${LOG_DIR}/latexmk_nawea_bridge.log"
}

log "VorLap directory: ${VORLAP_DIR}"
log "Run root: ${RUN_ROOT_HOST}"
log "QBlade root: ${QBLADE_ROOT_HOST}"
log "Python: ${PYTHON_HOST}"
log "Runner: ${RUNNER_NAME}"
log "Coupled scenario: ${COUPLED_SCENARIO} (${COUPLED_COUPLING}); native full simulation=${NATIVE_RUN_FULL_SIMULATION}."
if [[ "${USE_OPENCL}" == "1" ]]; then
  log "Using native Linux QBlade GPU/OpenCL execution with device ${OPENCL_DEVICE}, group-size ${OPENCL_GROUP_SIZE}, ${THREADS} OpenMP thread(s)."
else
  log "Using native Linux QBlade CPU/OpenMP execution with ${THREADS} thread(s)."
fi
log "QBlade restart checkpoints every ${CHECKPOINT_EVERY} step(s); STOREREPLAY=${STORE_REPLAY}."
check_python_environment

if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_scenario baseline 0 none 0
fi

if [[ "${RUN_COUPLED}" == "1" ]]; then
  run_scenario "${COUPLED_SCENARIO}" 1 "${COUPLED_COUPLING}" "${COUPLED_RECORD_HISTORY}"
fi

if [[ "${RUN_BASELINE}" == "1" && "${RUN_COUPLED}" == "1" ]]; then
  generate_paper_outputs
fi

log "Done. Outputs are under ${RUN_ROOT_HOST}."
