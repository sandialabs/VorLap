#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=224
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=naca0018-nalu-map
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/naca0018_prescribed_motion_nalu_v1}"
NALU_FLIGHT_ENV="${NALU_FLIGHT_ENV:-}"
NALU_EXECUTABLE="${NALU_EXECUTABLE:-}"
NALU_STAGE="${NALU_STAGE:-precursor}"
CORES_PER_NODE=112
fail() { echo "ERROR: $*" >&2; exit 2; }
[[ "${NALU_STAGE}" == "precursor" || "${NALU_STAGE}" == "motion" ]] || fail "NALU_STAGE must be precursor or motion."
if [[ -n "${NALU_FLIGHT_ENV}" ]]; then
  [[ -f "${NALU_FLIGHT_ENV}" ]] || fail "Nalu Flight environment unavailable: ${NALU_FLIGHT_ENV}"
  # shellcheck disable=SC1090
  source "${NALU_FLIGHT_ENV}"
fi
if [[ -n "${NALU_EXECUTABLE}" ]]; then
  if [[ "${NALU_EXECUTABLE}" == */* ]]; then
    [[ -x "${NALU_EXECUTABLE}" ]] || fail "Explicit NALU_EXECUTABLE is not executable: ${NALU_EXECUTABLE}"
    NALU_EXECUTABLE="$(readlink -f "${NALU_EXECUTABLE}")"
  else
    NALU_EXECUTABLE="$(command -v "${NALU_EXECUTABLE}" 2>/dev/null || true)"
  fi
else
  NALU_EXECUTABLE="$(command -v naluX 2>/dev/null || true)"
fi
[[ -n "${NALU_EXECUTABLE}" ]] || fail "No Nalu runtime found. Export NALU_EXECUTABLE, put naluX on PATH, or export NALU_FLIGHT_ENV to an accessible setup script."
[[ -x "${NALU_EXECUTABLE}" ]] || fail "Resolved naluX is not executable: ${NALU_EXECUTABLE}"
command -v flock >/dev/null 2>&1 || fail "flock is required for atomic Nalu result finalization."

python_candidates=("${PYTHON_EXE:-}" "${VORLAP_ROOT}/.venv/bin/python" "${VORLAP_FROZEN_PYTHON:-/pscratch/kevmoor/vorlap/.venv/bin/python}" "${PYTHON_EXE_FALLBACK:-python3}")
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
    'import h5py, scipy, yaml; from examples.ehrmann2014_naca0018 import nalu_prescribed_motion' >/dev/null 2>&1; then
    PYTHON_EXE="${resolved_python}"; break
  fi
done
[[ -n "${PYTHON_EXE}" ]] || fail "No Python can import the NACA0018 Nalu generator."
[[ "${SLURM_JOB_NUM_NODES:-}" =~ ^[248]$ ]] || fail "Production allocation must use exactly 2, 4, or 8 full nodes."
expected_tasks="$((SLURM_JOB_NUM_NODES * CORES_PER_NODE))"
[[ "${SLURM_NTASKS:-}" == "${expected_tasks}" ]] || fail "Expected ${expected_tasks} tasks; got ${SLURM_NTASKS:-unset}."
[[ "${SLURM_CPUS_PER_TASK:-}" == "1" ]] || fail "Expected one physical CPU per MPI rank."
physical_cores="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l | tr -d ' ')"
numa_domains="$(lscpu -p=NODE | awk -F, '!/^#/ && $1 >= 0 {print $1}' | sort -nu | wc -l | tr -d ' ')"
[[ "${physical_cores}" == "112" && "${numa_domains}" == "8" ]] || fail "Flight 112-core/8-NUMA topology check failed."
[[ -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]] || fail "Generate and scale-preflight the campaign first: ${CAMPAIGN_ROOT}"

PY=(env "PYTHONPATH=${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.nalu_prescribed_motion)
"${PY[@]}" validate --campaign "${CAMPAIGN_ROOT}"
"${PY[@]}" status --campaign "${CAMPAIGN_ROOT}" --no-write \
  --require-mesh-audit --require-scaling \
  --allocated-nodes "${SLURM_JOB_NUM_NODES}" --nalu-executable "${NALU_EXECUTABLE}"
export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ "${NALU_STAGE}" == "precursor" ]]; then
  [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-2]$ ]] || fail "Submit precursor stage with --array=0-2."
  stage=precursor
else
  [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([0-9]|[1-3][0-9]|4[0-6])$ ]] || fail "Submit the 47 remaining motions with --array=0-46."
  "${PY[@]}" status --campaign "${CAMPAIGN_ROOT}" --no-write \
    --require-mesh-audit --require-scaling \
    --require-precursors --require-motion-preflight --allocated-nodes "${SLURM_JOB_NUM_NODES}" \
    --nalu-executable "${NALU_EXECUTABLE}"
  stage=motion
  preflight_index="$("${PYTHON_EXE}" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["preflight"]["motion_case_index"])' \
    "${CAMPAIGN_ROOT}/campaign_manifest.json")"
  case_index="${SLURM_ARRAY_TASK_ID}"
  if (( case_index >= preflight_index )); then
    case_index="$((case_index + 1))"
  fi
fi

set +e
yaml_path="$("${PY[@]}" case-path --campaign "${CAMPAIGN_ROOT}" --stage "${stage}" \
  --index "${case_index:-${SLURM_ARRAY_TASK_ID}}" --skip-complete)"
path_status=$?
set -e
if [[ "${path_status}" == "10" ]]; then
  echo "${stage} array index ${SLURM_ARRAY_TASK_ID} is already complete."
  exit 0
fi
[[ "${path_status}" == "0" ]] || exit "${path_status}"
log_path="$(dirname "${yaml_path}")/log.out"
srun --exclusive --exact -N"${SLURM_JOB_NUM_NODES}" -n"${expected_tasks}" -c1 --cpu-bind=cores \
  "${NALU_EXECUTABLE}" -i "${yaml_path}" -o "${log_path}"
# Array elements finish independently.  Serialize the whole-campaign scan and
# atomic results-manifest publication so an earlier, incomplete scan cannot
# overwrite the final complete state after the last motion finishes.
exec 8>"${CAMPAIGN_ROOT}.status.lock"
flock 8
"${PY[@]}" status --campaign "${CAMPAIGN_ROOT}" --nalu-executable "${NALU_EXECUTABLE}"
flock -u 8
