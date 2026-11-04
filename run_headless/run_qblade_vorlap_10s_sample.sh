#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORLAP_DIR="${VORLAP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=run_headless/python_env.sh
source "${SCRIPT_DIR}/python_env.sh"
RUN_ROOT_REL="${RUN_ROOT_REL:-run_headless/runs/codex_bridge_sample_U10p58_10s}"
RUN_ROOT_HOST="${VORLAP_DIR}/${RUN_ROOT_REL}"
LOG_DIR="${RUN_ROOT_HOST}/logs"
VERIFICATION_RUN_ROOT="${VERIFICATION_RUN_ROOT:-${VORLAP_DIR}/run_headless/runs/codex_bridge_oscillatory_only_U10p58_1s_docker}"
SUMMARY_BASENAME="${QBLADE_SUMMARY_BASENAME:-baseline_10m_HVAWT_reference_turbine_Wwnd}"
VERIFICATION_SUMMARY_BASENAME="${VERIFICATION_SUMMARY_BASENAME:-${SUMMARY_BASENAME}}"

IMAGE="${IMAGE:-vorlap-qblade-headless:amd64}"
PLATFORM="${PLATFORM:-linux/amd64}"
QBLADE_ROOT_HOST="${QBLADE_ROOT_HOST:-${VORLAP_DIR}/run_headless/QBladeCE_2.0.9.7}"

WIND_SPEED="${WIND_SPEED:-10.58}"
AZIMUTH="${AZIMUTH:-270}"
TIMESTEP="${TIMESTEP:-0.0075}"
TIMESTEPS="${TIMESTEPS:-1334}"
THREADS="${THREADS:-16}"
PROGRESS_EVERY="${PROGRESS_EVERY:-200}"
STORE_REPLAY="${STORE_REPLAY:-${QBLADE_STORE_REPLAY:-true}}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-${QBLADE_CHECKPOINT_EVERY:-200}}"
RESUME_INITIALIZE="${RESUME_INITIALIZE:-0}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_SIL="${RUN_SIL:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
FORCE_SETUP="${FORCE_SETUP:-1}"
REBUILD_IMAGE="${REBUILD_IMAGE:-0}"
GENERATE_PAPER="${GENERATE_PAPER:-1}"
COMPILE_PAPER="${COMPILE_PAPER:-1}"
REQUIRE_PAPER="${REQUIRE_PAPER:-0}"
POSTPROCESS_RESULTS="${POSTPROCESS_RESULTS:-1}"
POSTPROCESS_LOCATIONS="${POSTPROCESS_LOCATIONS:-0.230 0.500 0.770 0.940}"
POSTPROCESS_BLADE="${POSTPROCESS_BLADE:-BLD_1}"
POSTPROCESS_INCLUDE_SMOKE_IN_DEL="${POSTPROCESS_INCLUDE_SMOKE_IN_DEL:-1}"

PYTHON_HOST="${PYTHON_HOST:-}"
if [[ -z "${PYTHON_HOST}" ]]; then
  if [[ -x "${VORLAP_DIR}/.venv/bin/python" ]]; then
    PYTHON_HOST="${VORLAP_DIR}/.venv/bin/python"
  else
    PYTHON_HOST="python3"
  fi
fi

[[ -d "${VORLAP_DIR}" ]] || die "VorLap directory not found: ${VORLAP_DIR}"
[[ -d "${VORLAP_DIR}/run_headless" ]] || die "run_headless directory not found under ${VORLAP_DIR}"
[[ -d "${QBLADE_ROOT_HOST}" ]] || die "QBlade root not found: ${QBLADE_ROOT_HOST}"
[[ -d "${QBLADE_ROOT_HOST}/SIL_Interface" ]] || die "QBlade SIL_Interface missing: ${QBLADE_ROOT_HOST}/SIL_Interface"
[[ -d "${QBLADE_ROOT_HOST}/ControllerFiles" ]] || die "QBlade ControllerFiles missing: ${QBLADE_ROOT_HOST}/ControllerFiles"
compgen -G "${QBLADE_ROOT_HOST}/libQBladeCE_*.so.1.0.0" >/dev/null || die "No QBlade shared library found in ${QBLADE_ROOT_HOST}"

mkdir -p "${LOG_DIR}"

if [[ "${QBLADE_ROOT_HOST}" == "${VORLAP_DIR}"* ]]; then
  CONTAINER_QBLADE_ROOT="/work/VorLap${QBLADE_ROOT_HOST#${VORLAP_DIR}}"
  QBLADE_MOUNT_HOST=""
else
  CONTAINER_QBLADE_ROOT="/work/QBladeCE_2.0.9.7"
  QBLADE_MOUNT_HOST="${QBLADE_ROOT_HOST}"
fi

ensure_docker() {
  command -v docker >/dev/null 2>&1 || die "docker command not found."
  if docker info >/dev/null 2>&1; then
    return
  fi

  if command -v colima >/dev/null 2>&1; then
    log "Docker is not reachable; attempting to start Colima."
    if ! colima status >/dev/null 2>&1; then
      colima start
    fi
    docker context use colima >/dev/null 2>&1 || true
  fi

  docker info >/dev/null 2>&1 || die "Docker is still not reachable. Start Docker/Colima, then rerun this script."
}

prepare_ca_bundle() {
  local bundle="${VORLAP_DIR}/run_headless/docker_host_ca.crt"
  local candidate
  for candidate in \
    /opt/homebrew/etc/ca-certificates/cert.pem \
    /usr/local/etc/ca-certificates/cert.pem \
    /etc/ssl/cert.pem \
    /etc/ssl/certs/ca-certificates.crt; do
    if [[ -f "${candidate}" ]]; then
      cp "${candidate}" "${bundle}"
      return
    fi
  done
  : > "${bundle}"
}

ensure_image() {
  if [[ "${REBUILD_IMAGE}" != "1" ]] && docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    log "Using existing Docker image ${IMAGE}."
    return
  fi

  log "Building Docker image ${IMAGE} for ${PLATFORM}."
  prepare_ca_bundle
  if docker buildx version >/dev/null 2>&1; then
    docker buildx build \
      --platform "${PLATFORM}" \
      --load \
      -f "${VORLAP_DIR}/run_headless/Dockerfile" \
      -t "${IMAGE}" \
      "${VORLAP_DIR}/run_headless"
  else
    docker build \
      --platform "${PLATFORM}" \
      -f "${VORLAP_DIR}/run_headless/Dockerfile" \
      -t "${IMAGE}" \
      "${VORLAP_DIR}/run_headless"
  fi
}

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

check_host_paper_environment() {
  [[ "${GENERATE_PAPER}" == "1" ]] || return 0
  [[ "${RUN_BASELINE}" == "1" && "${RUN_SIL}" == "1" ]] || return 0

  ensure_vorlap_python_environment \
    PYTHON_HOST \
    "NAWEA diagnostic plot generation" \
    "For a simulation-only Docker run without diagnostic plots, set GENERATE_PAPER=0." \
    0 \
    numpy matplotlib
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
    --scenarios "baseline sil_vorlap"
    --comparison-baseline baseline
    --comparison-coupled sil_vorlap
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

docker_run_common() {
  local docker_args=(
    run --rm \
    --platform "${PLATFORM}" \
    --user "$(id -u):$(id -g)" \
    --volume "${VORLAP_DIR}:/work/VorLap" \
    --workdir /work/VorLap \
    --env QBLADE_ROOT="${CONTAINER_QBLADE_ROOT}" \
    --env VORLAP_ROOT=/work/VorLap \
    --env PYTHON_EXE=/usr/bin/python3 \
    --env OMP_NUM_THREADS="${THREADS}" \
    --env QT_QPA_PLATFORM=offscreen \
    --env MPLCONFIGDIR=/tmp/matplotlib
  )
  if [[ -n "${QBLADE_MOUNT_HOST}" ]]; then
    docker_args+=(--volume "${QBLADE_MOUNT_HOST}:${CONTAINER_QBLADE_ROOT}")
  fi
  docker_args+=("${IMAGE}")
  docker "${docker_args[@]}" "$@"
}

run_scenario() {
  local scenario="$1"
  local force_scale="$2"
  local coupling="$3"
  local record_history="$4"
  local log_file="${LOG_DIR}/${scenario}_10s.log"

  if [[ "${SKIP_COMPLETED}" == "1" ]] && is_completed "${scenario}"; then
    log "Skipping ${scenario}; completed summary already exists at $(summary_path "${scenario}")."
    return
  fi

  log "Starting ${scenario}: U=${WIND_SPEED} m/s, azimuth=${AZIMUTH} deg, ${TIMESTEPS} steps at dt=${TIMESTEP} s."
  log "Writing log: ${log_file}"

  local command=(
    /usr/bin/python3 run_headless/vorlap_qblade_batch.py run-all \
      --run-root "${RUN_ROOT_REL}/${scenario}" \
      --qblade-root "${CONTAINER_QBLADE_ROOT}" \
      --vorlap-root /work/VorLap \
      --python /usr/bin/python3 \
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
      --checkpoint-every "${CHECKPOINT_EVERY}" \
      --coupling "${coupling}" \
      --wind-type 0 \
      --skip-vorlap-install \
      --skip-bridge-build
  )
  if [[ "${record_history}" == "1" ]]; then
    command+=(--record-sil-vorlap-history)
  fi
  if [[ "${RESUME_INITIALIZE}" == "1" ]]; then
    command+=(--resume-initialize)
  fi
  if [[ "${FORCE_SETUP}" == "1" ]]; then
    command+=(--force-setup)
  fi

  docker_run_common "${command[@]}" 2>&1 | tee "${log_file}"

  is_completed "${scenario}" || die "${scenario} finished but no complete summary was found at $(summary_path "${scenario}")"
  log "Completed ${scenario}."
}

generate_paper_outputs() {
  [[ "${GENERATE_PAPER}" == "1" ]] || return 0
  local missing_sample=()
  local missing_verification=()
  local scenario summary
  for scenario in baseline sil_vorlap; do
    summary="${VERIFICATION_RUN_ROOT}/${scenario}/results/U10p58ms/${VERIFICATION_SUMMARY_BASENAME}_run_summary.json"
    [[ -f "${summary}" ]] || missing_verification+=("${summary}")
  done
  for scenario in baseline sil_vorlap; do
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
  check_host_paper_environment
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
log "Docker image: ${IMAGE}"
log "QBlade restart checkpoints every ${CHECKPOINT_EVERY} step(s); STOREREPLAY=${STORE_REPLAY}."
check_host_paper_environment

ensure_docker
ensure_image

if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_scenario baseline 0 none 0
fi

if [[ "${RUN_SIL}" == "1" ]]; then
  run_scenario sil_vorlap 1 sil-vorlap 1
fi

if [[ "${RUN_BASELINE}" == "1" && "${RUN_SIL}" == "1" ]]; then
  generate_paper_outputs
fi

log "Done. Outputs are under ${RUN_ROOT_HOST}."
