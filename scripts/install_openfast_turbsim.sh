#!/usr/bin/env bash
set -Eeuo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

detect_jobs() {
  local value
  for value in "${OPENFAST_BUILD_JOBS:-}" "${SLURM_CPUS_ON_NODE:-}" "${SLURM_CPUS_PER_TASK:-}"; do
    if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )); then
      printf '%s\n' "${value}"
      return
    fi
  done
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    value="$(sysctl -n hw.ncpu 2>/dev/null || true)"
    if [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 )); then
      printf '%s\n' "${value}"
      return
    fi
  fi
  printf '1\n'
}

resolve_executable() {
  local exe="$1"
  [[ -n "${exe}" ]] || return 1
  if [[ "${exe}" == */* ]]; then
    printf '%s\n' "${exe}"
  else
    command -v "${exe}"
  fi
}

turbsim_is_usable() {
  local exe resolved output
  exe="$1"
  resolved="$(resolve_executable "${exe}" 2>/dev/null)" || return 1
  [[ -x "${resolved}" ]] || return 1

  # A banner probe catches executable files that cannot actually start because
  # of missing runtime libraries or a damaged/incomplete installation.
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

find_built_turbsim() {
  local build_type candidate
  build_type="${OPENFAST_CMAKE_BUILD_TYPE:-Release}"

  # Prefer the deterministic runtime directory configured below, while also
  # supporting existing OpenFAST build trees and multi-config generators.
  for candidate in \
    "${OPENFAST_BUILD_DIR}/bin/turbsim" \
    "${OPENFAST_BUILD_DIR}/bin/${build_type}/turbsim" \
    "${OPENFAST_BUILD_DIR}/modules/turbsim/turbsim" \
    "${OPENFAST_BUILD_DIR}/modules/turbsim/${build_type}/turbsim"; do
    if [[ -f "${candidate}" && -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  # Some shared filesystems can leave a newly built target without execute
  # bits. Find it by filename first; the install step sets mode 0755.
  candidate="$(
    find "${OPENFAST_BUILD_DIR}" -type f -iname turbsim -size +0c -print -quit \
      2>/dev/null || true
  )"
  [[ -n "${candidate}" ]] || return 1
  printf '%s\n' "${candidate}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORLAP_DIR="${VORLAP_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OPENFAST_REPO_URL="${OPENFAST_REPO_URL:-https://github.com/OpenFAST/openfast.git}"
OPENFAST_REF="${OPENFAST_REF:-v5.0.0}"
OPENFAST_ROOT="${OPENFAST_ROOT:-${VORLAP_DIR}/run_headless/openfast}"
OPENFAST_SOURCE_DIR="${OPENFAST_SOURCE_DIR:-${OPENFAST_ROOT}/source}"
OPENFAST_BUILD_DIR="${OPENFAST_BUILD_DIR:-${OPENFAST_ROOT}/build}"
OPENFAST_INSTALL_DIR="${OPENFAST_INSTALL_DIR:-${OPENFAST_ROOT}/install}"
TURBSIM_EXE="${TURBSIM_EXE:-${OPENFAST_INSTALL_DIR}/bin/turbsim}"
FORCE_OPENFAST_TURBSIM_BUILD="${FORCE_OPENFAST_TURBSIM_BUILD:-0}"
VORLAP_FORCE_SETUP_TOKEN="${VORLAP_FORCE_SETUP_TOKEN:-}"
jobs="$(detect_jobs)"

if turbsim_is_usable "${TURBSIM_EXE}" && [[ "${FORCE_OPENFAST_TURBSIM_BUILD}" != "1" ]]; then
  log "Using verified TurbSim executable: ${TURBSIM_EXE}"
  printf '%s\n' "${TURBSIM_EXE}"
  exit 0
fi

if [[ -e "${TURBSIM_EXE}" && "${FORCE_OPENFAST_TURBSIM_BUILD}" != "1" ]]; then
  log "WARNING: Existing TurbSim executable failed its startup check: ${TURBSIM_EXE}"
  log "Attempting an automatic rebuild and replacement."
fi

command -v git >/dev/null 2>&1 || die "git is required to download OpenFAST."
command -v cmake >/dev/null 2>&1 || die "cmake is required to build TurbSim from OpenFAST."
command -v install >/dev/null 2>&1 || die "install is required to place the built TurbSim executable."

mkdir -p "${OPENFAST_ROOT}"
if command -v flock >/dev/null 2>&1; then
  lock_file="${OPENFAST_ROOT}/.turbsim-install.lock"
  force_marker="${OPENFAST_ROOT}/.turbsim-force-token"
  exec 9>"${lock_file}"
  log "Waiting for exclusive TurbSim setup lock: ${lock_file}"
  flock 9
  if [[ "${FORCE_OPENFAST_TURBSIM_BUILD}" == "1" &&
        -n "${VORLAP_FORCE_SETUP_TOKEN}" &&
        -f "${force_marker}" &&
        "$(<"${force_marker}")" == "${VORLAP_FORCE_SETUP_TOKEN}" ]] &&
     turbsim_is_usable "${TURBSIM_EXE}"; then
    log "Forced TurbSim setup already completed for job token ${VORLAP_FORCE_SETUP_TOKEN}; reusing ${TURBSIM_EXE}."
    printf '%s\n' "${TURBSIM_EXE}"
    exit 0
  fi
  if turbsim_is_usable "${TURBSIM_EXE}" && [[ "${FORCE_OPENFAST_TURBSIM_BUILD}" != "1" ]]; then
    log "Another task completed TurbSim setup while this task waited: ${TURBSIM_EXE}"
    printf '%s\n' "${TURBSIM_EXE}"
    exit 0
  fi
fi

if [[ ! -d "${OPENFAST_SOURCE_DIR}/.git" ]]; then
  log "Cloning OpenFAST ${OPENFAST_REF} from ${OPENFAST_REPO_URL} into ${OPENFAST_SOURCE_DIR}."
  git clone --depth 1 --branch "${OPENFAST_REF}" "${OPENFAST_REPO_URL}" "${OPENFAST_SOURCE_DIR}" || {
    log "Shallow branch/tag clone failed; retrying generic clone then checkout."
    rm -rf "${OPENFAST_SOURCE_DIR}"
    git clone "${OPENFAST_REPO_URL}" "${OPENFAST_SOURCE_DIR}"
    git -C "${OPENFAST_SOURCE_DIR}" checkout "${OPENFAST_REF}"
  }
else
  log "Using existing OpenFAST source tree: ${OPENFAST_SOURCE_DIR}"
  git -C "${OPENFAST_SOURCE_DIR}" fetch --tags --depth 1 origin "${OPENFAST_REF}" || true
  git -C "${OPENFAST_SOURCE_DIR}" checkout "${OPENFAST_REF}"
fi

mkdir -p "${OPENFAST_BUILD_DIR}" "${OPENFAST_INSTALL_DIR}"
log "Configuring OpenFAST TurbSim build."
cmake -S "${OPENFAST_SOURCE_DIR}" -B "${OPENFAST_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${OPENFAST_CMAKE_BUILD_TYPE:-Release}" \
  -DCMAKE_INSTALL_PREFIX="${OPENFAST_INSTALL_DIR}" \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${OPENFAST_BUILD_DIR}/bin"

log "Building TurbSim with ${jobs} parallel job(s)."
cmake --build "${OPENFAST_BUILD_DIR}" --target turbsim --parallel "${jobs}"

log "Installing the built TurbSim target directly into ${TURBSIM_EXE}."
built="$(find_built_turbsim || true)"
if [[ -z "${built}" ]]; then
  log "Files with TurbSim in their name found below the build directory:"
  find "${OPENFAST_BUILD_DIR}" -type f -iname '*turbsim*' -print 2>/dev/null || true
  die "TurbSim build finished but no non-empty TurbSim executable was found below ${OPENFAST_BUILD_DIR}."
fi
mkdir -p "$(dirname "${TURBSIM_EXE}")" ||
  die "Could not create the TurbSim install directory: $(dirname "${TURBSIM_EXE}")"
if [[ -e "${TURBSIM_EXE}" && "${built}" -ef "${TURBSIM_EXE}" ]]; then
  chmod 0755 "${TURBSIM_EXE}" ||
    die "Could not mark the built TurbSim executable as executable."
else
  install -m 0755 "${built}" "${TURBSIM_EXE}" ||
    die "Could not install the built TurbSim executable at ${TURBSIM_EXE}."
fi
log "Direct TurbSim installation succeeded: ${built} -> ${TURBSIM_EXE}"

turbsim_is_usable "${TURBSIM_EXE}" ||
  die "TurbSim installation produced ${TURBSIM_EXE}, but it failed the startup/banner check. Check compiler runtime modules and shared-library availability."
if [[ "${FORCE_OPENFAST_TURBSIM_BUILD}" == "1" && -n "${VORLAP_FORCE_SETUP_TOKEN}" ]]; then
  printf '%s\n' "${VORLAP_FORCE_SETUP_TOKEN}" > "${OPENFAST_ROOT}/.turbsim-force-token"
fi
log "TurbSim ready: ${TURBSIM_EXE}"
printf '%s\n' "${TURBSIM_EXE}"
