#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-39%2
#SBATCH --time=23:59:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --job-name=ehrmann-viv
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_validation_v1}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

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
[[ -n "${PYTHON_EXE}" ]] || { echo "ERROR: No Python can import the current Ehrmann/VorLap runtime." >&2; exit 2; }
export PYTHON_EXE
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
  echo "ERROR: Ehrmann validation requires native Linux x86-64." >&2
  exit 2
}
[[ -d "${QBLADE_ROOT}" ]] || { echo "ERROR: QBlade root unavailable: ${QBLADE_ROOT}" >&2; exit 2; }
[[ -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]] || {
  echo "ERROR: Generate the campaign before submission: ${CAMPAIGN_ROOT}" >&2
  exit 2
}
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo "ERROR: Submit as the declared 0-39 Slurm array." >&2
  exit 2
}
[[ "${THREADS}" == "4" ]] || {
  echo "ERROR: Each QBlade case requires four threads; observed ${THREADS}." >&2
  exit 2
}

cd "${VORLAP_ROOT}"
echo "Using Ehrmann Python runtime: ${PYTHON_EXE}"
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign validate \
  --campaign "${CAMPAIGN_ROOT}"

export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Running Ehrmann case ${SLURM_ARRAY_TASK_ID}; two concurrent cases cap QBlade at eight cores."
exec "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign run \
  --campaign "${CAMPAIGN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" \
  --threads "${THREADS}" \
  --python "${PYTHON_EXE}" \
  --resume
