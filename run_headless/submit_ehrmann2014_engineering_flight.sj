#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=34
#SBATCH --ntasks-per-node=34
#SBATCH --cpus-per-task=1
#SBATCH --mem=68G
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=ehrmann-eng-lockin
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/ehrmann2014_engineering_lockin_v5}"
CALIBRATION="${CALIBRATION:-${VORLAP_ROOT}/examples/ehrmann2014_naca0018/engineering_lockin_calibration.json}"
RAW_REUSE_ROOT="${RAW_REUSE_ROOT:-/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_validation_v1/results}"
EHRMANN_ENGINEERING_DRY_RUN="${EHRMANN_ENGINEERING_DRY_RUN:-0}"
WORKERS=34
THREADS=1

fail() { echo "ERROR: $*" >&2; exit 2; }
[[ "${EHRMANN_ENGINEERING_DRY_RUN}" == "0" || "${EHRMANN_ENGINEERING_DRY_RUN}" == "1" ]] || \
  fail "EHRMANN_ENGINEERING_DRY_RUN must be 0 or 1."
[[ -f "${CALIBRATION}" && ! -L "${CALIBRATION}" ]] || \
  fail "Engineering calibration unavailable: ${CALIBRATION}"
[[ -d "${RAW_REUSE_ROOT}" && ! -L "${RAW_REUSE_ROOT}" ]] || \
  fail "Prior raw-QBlade results unavailable: ${RAW_REUSE_ROOT}"

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
    'import h5py,numpy,scipy; from examples.ehrmann2014_naca0018 import campaign; from run_headless import run_ehrmann2014_engineering_flight; from vorlap import qblade_run' >/dev/null 2>&1; then
    PYTHON_EXE="${resolved_python}"
    break
  fi
done
[[ -n "${PYTHON_EXE}" ]] || fail "No Python can import the engineering lock-in runtime."
export PYTHON_EXE
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
  fail "Ehrmann QBlade runs require native Linux x86-64."
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root unavailable: ${QBLADE_ROOT}"
[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || fail "Exactly one node is required."
[[ "${SLURM_NTASKS:-}" == "${WORKERS}" ]] || \
  fail "Expected ${WORKERS} allocated tasks; got ${SLURM_NTASKS:-unset}."
[[ "${SLURM_CPUS_PER_TASK:-}" == "${THREADS}" ]] || \
  fail "Expected one physical CPU per case; got ${SLURM_CPUS_PER_TASK:-unset}."
allocated_cpus="$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
[[ "${allocated_cpus}" == "${WORKERS}" ]] || fail "Expected 34 allocated CPUs."

physical_cores="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l | tr -d ' ')"
numa_domains="$(lscpu -p=NODE | awk -F, '!/^#/ && $1 >= 0 {print $1}' | sort -nu | wc -l | tr -d ' ')"
if [[ "${EHRMANN_ENGINEERING_DRY_RUN}" == "0" ]]; then
  [[ "${physical_cores}" == "112" ]] || fail "Expected 112 physical Flight cores; observed ${physical_cores}."
  [[ "${numa_domains}" == "8" ]] || fail "Expected 8 Flight NUMA domains; observed ${numa_domains}."
fi

cd "${VORLAP_ROOT}"
if [[ ! -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]]; then
  [[ ! -e "${CAMPAIGN_ROOT}" ]] || fail "Campaign root exists without a manifest: ${CAMPAIGN_ROOT}"
  mkdir -p "$(dirname "${CAMPAIGN_ROOT}")"
  staging="${CAMPAIGN_ROOT}.staging.${SLURM_JOB_ID:-$$}"
  [[ ! -e "${staging}" ]] || fail "Staging root already exists: ${staging}"
  "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign generate \
    --output "${staging}" \
    --runtime-vorlap-root "${VORLAP_ROOT}" \
    --raw-reuse-root "${RAW_REUSE_ROOT}" \
    --engineering-calibration "${CALIBRATION}"
  "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign validate --campaign "${staging}"
  mv -T -- "${staging}" "${CAMPAIGN_ROOT}"
fi
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign validate --campaign "${CAMPAIGN_ROOT}"

export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

scheduler=(
  "${PYTHON_EXE}" -m run_headless.run_ehrmann2014_engineering_flight
  --campaign "${CAMPAIGN_ROOT}"
  --qblade-root "${QBLADE_ROOT}"
  --calibration "${CALIBRATION}"
  --raw-reuse-root "${RAW_REUSE_ROOT}"
  --python "${PYTHON_EXE}"
  --workers "${WORKERS}"
  --threads "${THREADS}"
  --allocated-cpus "${allocated_cpus}"
)
if [[ "${EHRMANN_ENGINEERING_DRY_RUN}" == "1" ]]; then
  scheduler+=(--dry-run)
fi
"${scheduler[@]}"

if [[ "${EHRMANN_ENGINEERING_DRY_RUN}" == "0" ]]; then
  "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign analyze \
    --campaign "${CAMPAIGN_ROOT}"
  "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.build_engineering_validation_evidence \
    --campaign-root "${CAMPAIGN_ROOT}"
fi
