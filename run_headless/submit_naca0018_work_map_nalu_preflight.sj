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
#SBATCH --job-name=naca0018-nalu-preflight
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/naca0018_prescribed_motion_nalu_v1}"
NALU_MESH="${NALU_MESH:-/gpfs/kevmoor/airfoil_vortexshedding_mesh_run_post_nalu/mesh_gen_scripts/grids/NACA0018-90.exo}"
NALU_MESH_SHA256="${NALU_MESH_SHA256:-}"
NALU_FLIGHT_ENV="${NALU_FLIGHT_ENV:-}"
NALU_EXECUTABLE="${NALU_EXECUTABLE:-}"
NALU_PREFLIGHT_STAGE="${NALU_PREFLIGHT_STAGE:-scaling}"
CORES_PER_NODE=112

fail() { echo "ERROR: $*" >&2; exit 2; }
[[ "${NALU_PREFLIGHT_STAGE}" == "scaling" || "${NALU_PREFLIGHT_STAGE}" == "motion" ]] || \
  fail "NALU_PREFLIGHT_STAGE must be scaling or motion."
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

python_candidates=(
  "${PYTHON_EXE:-}"
  "${VORLAP_ROOT}/.venv/bin/python"
  "${VORLAP_FROZEN_PYTHON:-/pscratch/kevmoor/vorlap/.venv/bin/python}"
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
    'import h5py, scipy, yaml; from examples.ehrmann2014_naca0018 import nalu_prescribed_motion' >/dev/null 2>&1; then
    PYTHON_EXE="${resolved_python}"
    break
  fi
done
[[ -n "${PYTHON_EXE}" ]] || fail "No Python can import PyYAML and the NACA0018 Nalu generator."

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || fail "Flight Linux x86-64 is required."
[[ "${SLURM_JOB_NUM_NODES:-}" =~ ^[248]$ ]] || fail "Scaling/preflight allocation must use exactly 2, 4, or 8 full nodes."
expected_tasks="$((SLURM_JOB_NUM_NODES * CORES_PER_NODE))"
[[ "${SLURM_NTASKS:-}" == "${expected_tasks}" ]] || \
  fail "Expected ${expected_tasks} tasks for ${SLURM_JOB_NUM_NODES} full nodes; got ${SLURM_NTASKS:-unset}."
[[ "${SLURM_CPUS_PER_TASK:-}" == "1" ]] || fail "Expected one physical CPU per MPI rank."
physical_cores="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l | tr -d ' ')"
numa_domains="$(lscpu -p=NODE | awk -F, '!/^#/ && $1 >= 0 {print $1}' | sort -nu | wc -l | tr -d ' ')"
[[ "${physical_cores}" == "${CORES_PER_NODE}" ]] || fail "Expected 112 physical cores per Flight node; observed ${physical_cores}."
[[ "${numa_domains}" == "8" ]] || fail "Expected 8 NUMA domains per Flight node; observed ${numa_domains}."
[[ -f "${NALU_MESH}" && ! -L "${NALU_MESH}" ]] || fail "Exact NACA0018 Exodus mesh unavailable: ${NALU_MESH}"
[[ "${NALU_MESH_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]] || fail "Export the audited 64-character NALU_MESH_SHA256."

mkdir -p "$(dirname "${CAMPAIGN_ROOT}")"
exec 9>"${CAMPAIGN_ROOT}.generation.lock"
command -v flock >/dev/null 2>&1 || fail "flock is required for safe campaign publication."
flock 9
if [[ ! -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]]; then
  [[ ! -e "${CAMPAIGN_ROOT}" ]] || fail "Campaign exists without a manifest: ${CAMPAIGN_ROOT}"
  PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_EXE}" -m \
    examples.ehrmann2014_naca0018.nalu_prescribed_motion generate \
    --campaign "${CAMPAIGN_ROOT}" --mesh "${NALU_MESH}" --mesh-sha256 "${NALU_MESH_SHA256}"
fi
flock -u 9

PY=(env "PYTHONPATH=${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.nalu_prescribed_motion)
"${PY[@]}" validate --campaign "${CAMPAIGN_ROOT}"
flock 9
"${PY[@]}" audit-mesh --campaign "${CAMPAIGN_ROOT}"
flock -u 9
export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [[ "${NALU_PREFLIGHT_STAGE}" == "scaling" ]]; then
  "${PY[@]}" status --campaign "${CAMPAIGN_ROOT}" --no-write --require-mesh-audit \
    --nalu-executable "${NALU_EXECUTABLE}"
  yaml_path="$("${PY[@]}" case-path --campaign "${CAMPAIGN_ROOT}" --stage scaling --nodes "${SLURM_JOB_NUM_NODES}" --reynolds 64000)"
  log_path="$(dirname "${yaml_path}")/log.out"
  start_epoch="$(date +%s)"
  srun --exclusive --exact -N"${SLURM_JOB_NUM_NODES}" -n"${expected_tasks}" -c1 --cpu-bind=cores \
    "${NALU_EXECUTABLE}" -i "${yaml_path}" -o "${log_path}"
  end_epoch="$(date +%s)"
  "${PY[@]}" record-scaling --campaign "${CAMPAIGN_ROOT}" \
    --nodes "${SLURM_JOB_NUM_NODES}" --reynolds 64000 --log "${log_path}" \
    --elapsed-seconds "$((end_epoch - start_epoch))" --nalu-executable "${NALU_EXECUTABLE}"
  exec 8>"${CAMPAIGN_ROOT}.status.lock"
  flock 8
  "${PY[@]}" status --campaign "${CAMPAIGN_ROOT}" --require-mesh-audit \
    --nalu-executable "${NALU_EXECUTABLE}"
  flock -u 8
  exit 0
fi

"${PY[@]}" status --campaign "${CAMPAIGN_ROOT}" --no-write \
  --require-mesh-audit --require-scaling \
  --require-precursors --allocated-nodes "${SLURM_JOB_NUM_NODES}" \
  --nalu-executable "${NALU_EXECUTABLE}"
set +e
yaml_path="$("${PY[@]}" case-path --campaign "${CAMPAIGN_ROOT}" --stage motion-preflight --skip-complete)"
path_status=$?
set -e
if [[ "${path_status}" == "10" ]]; then
  echo "Motion preflight is already complete."
  exit 0
fi
[[ "${path_status}" == "0" ]] || exit "${path_status}"
log_path="$(dirname "${yaml_path}")/log.out"
srun --exclusive --exact -N"${SLURM_JOB_NUM_NODES}" -n"${expected_tasks}" -c1 --cpu-bind=cores \
  "${NALU_EXECUTABLE}" -i "${yaml_path}" -o "${log_path}"
exec 8>"${CAMPAIGN_ROOT}.status.lock"
flock 8
"${PY[@]}" status --campaign "${CAMPAIGN_ROOT}" --require-mesh-audit --require-scaling \
  --require-precursors --require-motion-preflight --allocated-nodes "${SLURM_JOB_NUM_NODES}" \
  --nalu-executable "${NALU_EXECUTABLE}"
flock -u 8
