#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --reservation=flight-cldera
#SBATCH --account=FY260158
#SBATCH --job-name=vorlap-qblade
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --array=0-12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

initial_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_DIR=""
for candidate in \
  "${VORLAP_ROOT:-}/run_headless" \
  "${SLURM_SUBMIT_DIR:-}/run_headless" \
  "${SLURM_SUBMIT_DIR:-}" \
  "${initial_script_dir}" \
  "$(pwd -P)/run_headless" \
  "$(pwd -P)"; do
  [[ -n "${candidate}" ]] || continue
  if [[ -f "${candidate}/vorlap_qblade_batch.py" ]]; then
    SCRIPT_DIR="$(cd "${candidate}" && pwd -P)"
    break
  fi
done
[[ -n "${SCRIPT_DIR}" ]] || {
  echo "ERROR: Could not resolve the VorLap run_headless checkout from the Slurm spool copy. Submit from the VorLap checkout or export VORLAP_ROOT=/absolute/path/to/VorLap." >&2
  exit 2
}
if [[ -z "${VORLAP_ROOT:-}" || ! -f "${VORLAP_ROOT}/run_headless/vorlap_qblade_batch.py" ]]; then
  VORLAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi
QBLADE_ROOT="${QBLADE_ROOT:-${SCRIPT_DIR}/QBladeCE_2.0.9.7}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/vorlap_parked}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

if [[ -z "${PYTHON_EXE:-}" && -x "${VORLAP_ROOT}/.venv/bin/python" ]]; then
  PYTHON_EXE="${VORLAP_ROOT}/.venv/bin/python"
else
  PYTHON_EXE="${PYTHON_EXE:-python3}"
fi

setup_flags=()
if [[ "${FORCE_SETUP:-0}" == "1" ]]; then
  setup_flags+=(--force-setup)
fi
if [[ "${FORCE_BRIDGE_BUILD:-0}" == "1" ]]; then
  setup_flags+=(--force-bridge-build)
fi
if [[ "${SKIP_VORLAP_INSTALL:-0}" == "1" ]]; then
  setup_flags+=(--skip-vorlap-install)
fi
if [[ "${SKIP_BRIDGE_BUILD:-0}" == "1" ]]; then
  setup_flags+=(--skip-bridge-build)
fi

run_flags=()
if [[ -n "${QBLADE_STEPS:-}" ]]; then
  run_flags+=(--qblade-steps "${QBLADE_STEPS}")
fi
if [[ "${ALLOW_PARTIAL:-0}" == "1" ]]; then
  run_flags+=(--allow-partial)
fi
if [[ "${NATIVE_RUN_FULL_SIMULATION:-0}" == "1" ]]; then
  run_flags+=(--native-run-full-simulation)
fi

"${PYTHON_EXE}" "${SCRIPT_DIR}/vorlap_qblade_batch.py" run \
  --run-root "${RUN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}" \
  --vorlap-root "${VORLAP_ROOT}" \
  --python "${PYTHON_EXE}" \
  --wind-speeds "${WIND_SPEEDS:-linspace:2:20:13}" \
  --azimuths "${QBLADE_AZIMUTHS:-270}" \
  --coupling "${VORLAP_QBLADE_COUPLING:-sil-vorlap}" \
  --threads "${OMP_NUM_THREADS}" \
  "${setup_flags[@]}" \
  "${run_flags[@]}"
