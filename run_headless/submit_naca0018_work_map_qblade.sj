#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=47
#SBATCH --ntasks-per-node=47
#SBATCH --cpus-per-task=1
#SBATCH --mem=94G
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=naca0018-qb-map
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/naca0018_work_map_qblade_v1}"
QBLADE_STAGE="${QBLADE_STAGE:-}"
QBLADE_WORK_MAP_DRY_RUN="${QBLADE_WORK_MAP_DRY_RUN:-0}"
THREADS=1
EXPECTED_NODE_PHYSICAL_CORES=112
EXPECTED_NUMA_DOMAINS=8

fail() { echo "ERROR: $*" >&2; exit 2; }
[[ "${QBLADE_WORK_MAP_DRY_RUN}" == "0" || "${QBLADE_WORK_MAP_DRY_RUN}" == "1" ]] || \
  fail "QBLADE_WORK_MAP_DRY_RUN must be exactly 0 or 1."
[[ "${QBLADE_STAGE}" == "preflight" || "${QBLADE_STAGE}" == "matrix" ]] || \
  fail "QBLADE_STAGE must be explicitly set to preflight or matrix."
if [[ "${QBLADE_STAGE}" == "preflight" ]]; then
  WORKERS=1
else
  WORKERS=47
fi
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
  fail "The frozen QBlade runtime requires native Linux x86-64."
[[ -d "${VORLAP_ROOT}" ]] || fail "VorLap root unavailable: ${VORLAP_ROOT}"
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root unavailable: ${QBLADE_ROOT}"

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
    'import numpy; from examples.ehrmann2014_naca0018 import qblade_prescribed_motion; from run_headless import run_naca0018_work_map_qblade_flight; from vorlap import qblade_run' >/dev/null 2>&1; then
    PYTHON_EXE="${resolved_python}"
    break
  fi
  echo "WARNING: Python failed the prescribed-motion QBlade import preflight: ${resolved_python}" >&2
done
[[ -n "${PYTHON_EXE}" ]] || fail "No Python can import the prescribed-motion QBlade runtime."
export PYTHON_EXE

[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || fail "Exactly one shared Flight node is required."
[[ "${SLURM_NTASKS:-}" == "${WORKERS}" ]] || \
  fail "Expected ${WORKERS} allocated tasks; got ${SLURM_NTASKS:-unset}."
[[ "${SLURM_CPUS_PER_TASK:-}" == "${THREADS}" ]] || \
  fail "Expected one physical CPU per QBlade case; got ${SLURM_CPUS_PER_TASK:-unset}."
[[ "${SLURM_NTASKS}" =~ ^[0-9]+$ && "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
  fail "Slurm task/CPU allocation fields must be positive integers."
expected_slots="$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
[[ "${expected_slots}" -eq "${WORKERS}" ]] || \
  fail "Expected ${WORKERS} fully assigned ${QBLADE_STAGE} CPU slots; got ${expected_slots}."
physical_cores="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l | tr -d ' ')"
numa_domains="$(lscpu -p=NODE | awk -F, '!/^#/ && $1 >= 0 {print $1}' | sort -nu | wc -l | tr -d ' ')"
[[ "${physical_cores}" == "${EXPECTED_NODE_PHYSICAL_CORES}" ]] || \
  fail "Flight topology changed: expected ${EXPECTED_NODE_PHYSICAL_CORES} physical cores, observed ${physical_cores}."
[[ "${numa_domains}" == "${EXPECTED_NUMA_DOMAINS}" ]] || \
  fail "Flight topology changed: expected ${EXPECTED_NUMA_DOMAINS} NUMA domains, observed ${numa_domains}."
echo "Flight topology verified: ${physical_cores} physical cores, ${numa_domains} NUMA domains."
echo "Shared allocation: requested slots=${expected_slots}; SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-unset}; SLURM_JOB_CPUS_PER_NODE=${SLURM_JOB_CPUS_PER_NODE:-unset}."

cd "${VORLAP_ROOT}"
echo "Using prescribed-motion Python runtime: ${PYTHON_EXE}"
if [[ ! -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]]; then
  [[ "${QBLADE_STAGE}" == "preflight" ]] || \
    fail "The matrix stage requires the preflight-generated campaign: ${CAMPAIGN_ROOT}"
  [[ ! -e "${CAMPAIGN_ROOT}" ]] || \
    fail "Campaign root exists without a manifest; refusing to replace it: ${CAMPAIGN_ROOT}"
  campaign_parent="$(dirname "${CAMPAIGN_ROOT}")"
  mkdir -p "${campaign_parent}"
  staging_root="${CAMPAIGN_ROOT}.staging.${SLURM_JOB_ID:-$$}"
  [[ ! -e "${staging_root}" ]] || fail "Campaign staging root already exists: ${staging_root}"
  echo "Campaign is absent; generating and validating the exact 48-case native-QBlade matrix."
  "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.qblade_prescribed_motion generate \
    --campaign "${staging_root}" \
    --qblade-root "${QBLADE_ROOT}"
  "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.qblade_prescribed_motion validate \
    --campaign "${staging_root}" \
    --qblade-root "${QBLADE_ROOT}"
  mv -T -- "${staging_root}" "${CAMPAIGN_ROOT}"
  echo "Published generated campaign: ${CAMPAIGN_ROOT}"
fi
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.qblade_prescribed_motion validate \
  --campaign "${CAMPAIGN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}"

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Launching the ${QBLADE_STAGE} stage with ${WORKERS} fully assigned one-core QBlade slots."
scheduler=(
  "${PYTHON_EXE}" -m run_headless.run_naca0018_work_map_qblade_flight
  --stage "${QBLADE_STAGE}"
  --campaign "${CAMPAIGN_ROOT}"
  --qblade-root "${QBLADE_ROOT}"
  --python "${PYTHON_EXE}"
  --workers "${WORKERS}"
  --threads "${THREADS}"
  --allocated-cpus "${expected_slots}"
)
if [[ "${QBLADE_WORK_MAP_DRY_RUN}" == "1" ]]; then
  scheduler+=(--dry-run)
fi
"${scheduler[@]}"

if [[ "${QBLADE_WORK_MAP_DRY_RUN}" == "0" && "${QBLADE_STAGE}" == "matrix" ]]; then
  [[ -f "${CAMPAIGN_ROOT}/qblade_results.json" ]] || \
    fail "The scheduler completed without final qblade_results.json."
  "${PYTHON_EXE}" - "${CAMPAIGN_ROOT}/qblade_results.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
assert value["schema"] == "vorlap-naca0018-qblade-prescribed-motion-results-v1"
assert value["status"] == "complete"
assert value["summary"] == {
    "required_cases": 48,
    "complete_cases": 48,
    "failed_cases": 0,
}
print(json.dumps({"qblade_results": str(path), "completed_cases": 48}, indent=2))
PY
fi
