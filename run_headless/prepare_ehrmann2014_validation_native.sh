#!/bin/bash
set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_validation_v1}"
RAW_REUSE_ROOT="${RAW_REUSE_ROOT:-}"

fail() { echo "ERROR: $*" >&2; exit 2; }
[[ "$(uname -s)" == "Linux" ]] || fail "Preparation must run on native Linux."
[[ "$(uname -m)" == "x86_64" ]] || fail "Preparation requires native x86-64."
python_candidates=(
  "${PYTHON_EXE:-}"
  "${VORLAP_FROZEN_PYTHON:-/pscratch/kevmoor/vorlap/.venv/bin/python}"
  "${VORLAP_ROOT}/.venv/bin/python"
  "${PYTHON_EXE_FALLBACK:-python3}"
)
PYTHON_EXE=""
for candidate in "${python_candidates[@]}"; do
  [[ -n "${candidate}" ]] || continue
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] || continue
    resolved_python="$(cd "$(dirname "${candidate}")" && pwd -P)/$(basename "${candidate}")"
  else
    resolved_python="$(command -v "${candidate}" 2>/dev/null)" || continue
  fi
  if PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${resolved_python}" -c \
    'import h5py, numpy, scipy; from examples.ehrmann2014_naca0018 import campaign; from vorlap import qblade_run' >/dev/null 2>&1; then
    PYTHON_EXE="${resolved_python}"
    break
  fi
  echo "WARNING: Python failed the Ehrmann import preflight: ${resolved_python}" >&2
done
[[ -n "${PYTHON_EXE}" ]] || fail "No Python can import the current Ehrmann/VorLap runtime."
export PYTHON_EXE
[[ ! -e "${CAMPAIGN_ROOT}" ]] || fail "Campaign root already exists: ${CAMPAIGN_ROOT}"

cd "${VORLAP_ROOT}"
echo "Using Ehrmann Python runtime: ${PYTHON_EXE}"
generate=(
  "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign generate
  --output "${CAMPAIGN_ROOT}"
  --runtime-vorlap-root "${VORLAP_ROOT}"
)
if [[ -n "${RAW_REUSE_ROOT}" ]]; then
  generate+=(--raw-reuse-root "${RAW_REUSE_ROOT}")
fi
"${generate[@]}"
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign validate \
  --campaign "${CAMPAIGN_ROOT}"
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign list \
  --campaign "${CAMPAIGN_ROOT}"

echo "Prepared the exact 40-case Ehrmann campaign at ${CAMPAIGN_ROOT}."
echo "Submit the shared-node 40-way one-core job (two hours):"
echo "sbatch --export=ALL,VORLAP_ROOT=${VORLAP_ROOT},CAMPAIGN_ROOT=${CAMPAIGN_ROOT} ${VORLAP_ROOT}/run_headless/submit_ehrmann2014_flight.sj"
echo "If the first allocation expires, use the identical checkpoint-resuming job with eight hours:"
echo "sbatch --time=08:00:00 --export=ALL,VORLAP_ROOT=${VORLAP_ROOT},CAMPAIGN_ROOT=${CAMPAIGN_ROOT} ${VORLAP_ROOT}/run_headless/submit_ehrmann2014_flight.sj"
