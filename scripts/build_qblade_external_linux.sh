#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build/qblade_external_linux}"
INSTALL_DIR="${2:-}"
if [[ -z "${PYTHON_EXE:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_EXE="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_EXE="python3"
  fi
else
  PYTHON_EXE="${PYTHON_EXE}"
fi

echo "Building QBlade bridge with Python: ${PYTHON_EXE}"
"${PYTHON_EXE}" -c "import sys, numpy; print(f'Python executable: {sys.executable}'); print(f'Python version: {sys.version.split()[0]}'); print(f'NumPy version: {numpy.__version__}')" 

cmake -S "${ROOT_DIR}/qblade_external_bridge" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE="${PYTHON_EXE}"
cmake --build "${BUILD_DIR}" --config Release

LIB_PATH="$(find "${BUILD_DIR}" -maxdepth 3 -type f \( -name 'libvorlap_qblade_bridge.so' -o -name 'libvorlap_qblade_bridge.dylib' -o -name 'libvorlap_qblade_bridge.dll' \) | head -n 1)"
if [[ -z "${LIB_PATH}" ]]; then
  echo "Could not locate built shared library under ${BUILD_DIR}" >&2
  exit 1
fi

echo "Built library: ${LIB_PATH}"

if [[ -n "${INSTALL_DIR}" ]]; then
  mkdir -p "${INSTALL_DIR}"
  cp "${LIB_PATH}" "${INSTALL_DIR}/"
  echo "Copied to: ${INSTALL_DIR}"
fi
