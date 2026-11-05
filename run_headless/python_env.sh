#!/usr/bin/env bash
# Shared Python environment bootstrap for headless QBlade/VorLap scripts.

vorlap_python_command_exists() {
  local python="$1"
  [[ -n "${python}" ]] || return 1
  if [[ "${python}" == */* ]]; then
    [[ -x "${python}" ]]
  else
    command -v "${python}" >/dev/null 2>&1
  fi
}

vorlap_python_has_module() {
  local python="$1"
  local module="$2"
  vorlap_python_command_exists "${python}" || return 1
  "${python}" - "${module}" >/dev/null 2>&1 <<'PY'
import importlib.util
import sys

module = sys.argv[1]
sys.exit(0 if importlib.util.find_spec(module) is not None else 1)
PY
}

vorlap_python_imports_bridge() {
  local python="$1"
  local root="$2"
  vorlap_python_command_exists "${python}" || return 1
  "${python}" - "${root}" >/dev/null 2>&1 <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
import vorlap
import vorlap.qblade_batch
import vorlap.qblade_run
PY
}

vorlap_should_use_repo_venv() {
  local python="$1"
  [[ "${VORLAP_ALLOW_SYSTEM_PIP:-0}" == "1" ]] && return 1
  if ! vorlap_python_command_exists "${python}"; then
    return 0
  fi
  case "${python}" in
    python|python3|/bin/python|/bin/python3|/usr/bin/python|/usr/bin/python3|/usr/local/bin/python|/usr/local/bin/python3)
      return 0
      ;;
  esac
  return 1
}

vorlap_pip_trusted_hosts() {
  local hosts="${VORLAP_PIP_TRUSTED_HOSTS:-pypi.org files.pythonhosted.org pypi.python.org}"
  local host
  hosts="${hosts//,/ }"
  for host in ${hosts}; do
    [[ -n "${host}" ]] || continue
    printf '%s\n' "${host}"
  done
}

vorlap_pip_trusted_host_args() {
  local host
  for host in $(vorlap_pip_trusted_hosts); do
    [[ -n "${host}" ]] || continue
    printf '%s\n' "--trusted-host"
    printf '%s\n' "${host}"
  done
}

vorlap_setup_python_environment() {
  local python_var="$1"
  local python="${!python_var:-}"
  local venv_dir="${VORLAP_VENV_DIR:-${VORLAP_DIR}/.venv}"
  local target_python="${python}"
  local bootstrap_python="${VORLAP_BOOTSTRAP_PYTHON:-python3}"
  local extras="${VORLAP_PYTHON_EXTRAS:-dev,docs,gui}"
  local pip_trusted_host_env=""
  local pip_trusted_args=()
  local pip_host

  [[ "${AUTO_SETUP_PYTHON:-1}" == "1" ]] || return 1
  [[ -n "${VORLAP_DIR:-}" ]] || die "VORLAP_DIR must be set before Python auto-setup."

  if vorlap_should_use_repo_venv "${python}"; then
    if [[ ! -x "${venv_dir}/bin/python" ]]; then
      command -v "${bootstrap_python}" >/dev/null 2>&1 || die "Cannot create ${venv_dir}; bootstrap Python not found: ${bootstrap_python}"
      log "Creating VorLap Python environment at ${venv_dir}."
      if ! "${bootstrap_python}" -m venv "${venv_dir}"; then
        log "Could not create ${venv_dir} with ${bootstrap_python} -m venv."
        log "Install the Python venv package for your platform, or set PYTHON_HOST/PYTHON_EXE to a ready environment."
        die "Python environment bootstrap failed."
      fi
    fi
    target_python="${venv_dir}/bin/python"
    printf -v "${python_var}" '%s' "${target_python}"
  fi

  while IFS= read -r pip_host; do
    [[ -n "${pip_host}" ]] || continue
    pip_trusted_args+=(--trusted-host "${pip_host}")
    if [[ -n "${pip_trusted_host_env}" ]]; then
      pip_trusted_host_env+=" ${pip_host}"
    else
      pip_trusted_host_env="${pip_host}"
    fi
  done < <(vorlap_pip_trusted_hosts)

  vorlap_python_command_exists "${target_python}" || die "Python executable not found: ${target_python}"
  if ! "${target_python}" -m pip --version >/dev/null 2>&1; then
    "${target_python}" -m ensurepip --upgrade >/dev/null 2>&1 || die "pip is not available for ${target_python}"
  fi

  log "Installing/updating VorLap Python dependencies in ${target_python}."
  log "Using pip trusted host(s): ${pip_trusted_host_env}"
  (
    cd "${VORLAP_DIR}"
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_TRUSTED_HOST="${pip_trusted_host_env}"
    "${target_python}" -m pip install "${pip_trusted_args[@]}" --upgrade pip setuptools wheel
    "${target_python}" -m pip install "${pip_trusted_args[@]}" -e ".[${extras}]"
  ) || die "Python dependency installation failed for ${target_python}."
}

ensure_vorlap_python_environment() {
  local python_var="$1"
  local context="$2"
  local skip_hint="$3"
  local check_bridge="$4"
  shift 4
  local required_modules=("$@")
  local missing=()
  local module
  local python="${!python_var:-}"
  local needs_setup=0

  if ! vorlap_python_command_exists "${python}"; then
    log "Python executable is not available: ${python:-<empty>}"
    needs_setup=1
  else
    for module in "${required_modules[@]}"; do
      if ! vorlap_python_has_module "${python}" "${module}"; then
        missing+=("${module}")
      fi
    done
    if (( ${#missing[@]} )); then
      log "Python environment is missing required module(s) for ${context}: ${missing[*]}"
      needs_setup=1
    fi
    if [[ "${check_bridge}" == "1" ]] && ! vorlap_python_imports_bridge "${python}" "${VORLAP_DIR}"; then
      log "Python can import packages but cannot import the VorLap bridge modules."
      needs_setup=1
    fi
  fi

  if (( needs_setup )); then
    log "Python before setup: ${python:-<empty>}"
    if [[ "${AUTO_SETUP_PYTHON:-1}" != "1" ]]; then
      log "Automatic setup is disabled by AUTO_SETUP_PYTHON=0."
      [[ -n "${skip_hint}" ]] && log "${skip_hint}"
      log "Manual setup from the VorLap root:"
      log "  ${python:-python3} -m pip install -e \".[${VORLAP_PYTHON_EXTRAS:-dev,docs,gui}]\""
      die "Python environment is not ready for ${context}."
    fi
    vorlap_setup_python_environment "${python_var}"
  fi

  python="${!python_var:-}"
  missing=()
  vorlap_python_command_exists "${python}" || die "Python executable is not available after setup: ${python:-<empty>}"
  for module in "${required_modules[@]}"; do
    if ! vorlap_python_has_module "${python}" "${module}"; then
      missing+=("${module}")
    fi
  done
  if (( ${#missing[@]} )); then
    log "Python is still missing required module(s): ${missing[*]}"
    [[ -n "${skip_hint}" ]] && log "${skip_hint}"
    log "Manual setup from the VorLap root:"
    log "  ${python} -m pip install -e \".[${VORLAP_PYTHON_EXTRAS:-dev,docs,gui}]\""
    die "Python environment is not ready for ${context}."
  fi
  if [[ "${check_bridge}" == "1" ]] && ! vorlap_python_imports_bridge "${python}" "${VORLAP_DIR}"; then
    log "Python cannot import VorLap bridge modules after setup."
    log "Manual setup from the VorLap root:"
    log "  ${python} -m pip install -e \".[${VORLAP_PYTHON_EXTRAS:-dev,docs,gui}]\""
    die "VorLap bridge import preflight failed."
  fi
  log "Python environment ready for ${context}: ${python}"
}

vorlap_default_turbsim_exe() {
  printf '%s\n' "${VORLAP_DIR}/run_headless/openfast/install/bin/turbsim"
}

vorlap_resolve_turbsim_executable() {
  local exe="$1"
  [[ -n "${exe}" ]] || return 1
  if [[ "${exe}" == */* ]]; then
    printf '%s\n' "${exe}"
  else
    command -v "${exe}"
  fi
}

vorlap_turbsim_command_works() {
  local exe resolved output
  exe="$1"
  resolved="$(vorlap_resolve_turbsim_executable "${exe}" 2>/dev/null)" || return 1
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

ensure_vorlap_turbsim() {
  [[ "${GENERATE_TURBSIM:-0}" == "1" ]] || return 0
  [[ "${DRY_RUN_ONLY:-0}" == "1" ]] && return 0
  [[ -n "${VORLAP_DIR:-}" ]] || die "VORLAP_DIR must be set before TurbSim auto-setup."

  local force_build="${FORCE_OPENFAST_TURBSIM_BUILD:-0}"
  if [[ "${force_build}" == "1" ]]; then
    log "Forced TurbSim reinstall/rebuild requested."
  elif vorlap_turbsim_command_works "${TURBSIM_EXE:-}"; then
    log "TurbSim executable passed its startup check: ${TURBSIM_EXE}"
    return 0
  fi
  if [[ "${force_build}" != "1" && -n "${TURBSIM_EXE:-}" ]]; then
    log "WARNING: Configured TurbSim executable is missing or failed its startup check: ${TURBSIM_EXE}"
  fi

  local default_exe
  default_exe="$(vorlap_default_turbsim_exe)"
  if [[ "${force_build}" != "1" ]] && vorlap_turbsim_command_works "${default_exe}"; then
    TURBSIM_EXE="${default_exe}"
    export TURBSIM_EXE
    log "TurbSim executable passed its startup check: ${TURBSIM_EXE}"
    return 0
  fi
  if [[ "${force_build}" != "1" && -e "${default_exe}" ]]; then
    log "WARNING: Default TurbSim executable exists but failed its startup check: ${default_exe}"
  fi

  if [[ "${AUTO_SETUP_TURBSIM:-1}" != "1" ]]; then
    die "GENERATE_TURBSIM=1 but no usable TurbSim executable was found. Set TURBSIM_EXE or enable AUTO_SETUP_TURBSIM=1."
  fi

  local installer="${VORLAP_DIR}/scripts/install_openfast_turbsim.sh"
  [[ -x "${installer}" ]] || die "TurbSim installer not found or not executable: ${installer}"
  if [[ -z "${TURBSIM_EXE:-}" || "${TURBSIM_EXE}" != */* ]]; then
    TURBSIM_EXE="${default_exe}"
  fi
  export TURBSIM_EXE
  log "TurbSim is not usable; running automatic OpenFAST setup/repair."
  "${installer}"
  vorlap_turbsim_command_works "${TURBSIM_EXE}" || die "TurbSim installer did not produce a usable executable: ${TURBSIM_EXE}"
  log "Automatic TurbSim setup/repair succeeded: ${TURBSIM_EXE}"
}
