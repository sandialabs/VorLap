#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=ehrmann-viv-node
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_validation_v1}"
RAW_REUSE_ROOT="${RAW_REUSE_ROOT:-}"
EHRMANN_FLIGHT_DRY_RUN="${EHRMANN_FLIGHT_DRY_RUN:-0}"
WORKERS=40
THREADS=1
EXPECTED_ALLOCATED_CPUS=40
EXPECTED_NODE_PHYSICAL_CORES=112
EXPECTED_NUMA_DOMAINS=8

fail() { echo "ERROR: $*" >&2; exit 2; }
[[ "${EHRMANN_FLIGHT_DRY_RUN}" == "0" || "${EHRMANN_FLIGHT_DRY_RUN}" == "1" ]] || \
  fail "EHRMANN_FLIGHT_DRY_RUN must be exactly 0 or 1."
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
    'import h5py, numpy, scipy; from examples.ehrmann2014_naca0018 import campaign; from run_headless import run_ehrmann2014_flight; from vorlap import qblade_run' >/dev/null 2>&1; then
    PYTHON_EXE="${resolved_python}"
    break
  fi
  echo "WARNING: Python failed the Ehrmann import preflight: ${resolved_python}" >&2
done
[[ -n "${PYTHON_EXE}" ]] || fail "No Python can import the current Ehrmann/VorLap runtime."
export PYTHON_EXE
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
  fail "Ehrmann validation requires native Linux x86-64."
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root unavailable: ${QBLADE_ROOT}"
[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || fail "Exactly one node is required."
[[ "${SLURM_NTASKS:-}" == "${WORKERS}" ]] || \
  fail "Expected ${WORKERS} allocated tasks; got ${SLURM_NTASKS:-unset}."
[[ "${SLURM_CPUS_PER_TASK:-}" == "${THREADS}" ]] || \
  fail "Expected ${THREADS} physical CPUs per case; got ${SLURM_CPUS_PER_TASK:-unset}."
[[ "${SLURM_NTASKS}" =~ ^[0-9]+$ && "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
  fail "Slurm task/CPU allocation fields must be positive integers."
allocated_cpus="$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
[[ "${allocated_cpus}" == "${EXPECTED_ALLOCATED_CPUS}" ]] || \
  fail "Expected ${EXPECTED_ALLOCATED_CPUS} campaign CPU slots from SLURM_NTASKS x SLURM_CPUS_PER_TASK; got ${allocated_cpus}."
node_visible_cpus="${SLURM_CPUS_ON_NODE:-unset}"
job_cpus_per_node="${SLURM_JOB_CPUS_PER_NODE:-unset}"

physical_cores="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l | tr -d ' ')"
numa_domains="$(lscpu -p=NODE | awk -F, '!/^#/ && $1 >= 0 {print $1}' | sort -nu | wc -l | tr -d ' ')"
if [[ "${EHRMANN_FLIGHT_DRY_RUN}" == "0" ]]; then
  [[ "${physical_cores}" == "${EXPECTED_NODE_PHYSICAL_CORES}" ]] || \
    fail "Flight topology changed: expected ${EXPECTED_NODE_PHYSICAL_CORES} physical cores, observed ${physical_cores}."
  [[ "${numa_domains}" == "${EXPECTED_NUMA_DOMAINS}" ]] || \
    fail "Flight topology changed: expected ${EXPECTED_NUMA_DOMAINS} NUMA domains, observed ${numa_domains}."
else
  echo "Dry run: observed local topology ${physical_cores} physical cores / ${numa_domains} NUMA domains; production still requires ${EXPECTED_NODE_PHYSICAL_CORES} / ${EXPECTED_NUMA_DOMAINS}."
fi

cd "${VORLAP_ROOT}"
echo "Using Ehrmann Python runtime: ${PYTHON_EXE}"
if [[ ! -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]]; then
  [[ ! -e "${CAMPAIGN_ROOT}" ]] || \
    fail "Campaign root exists without a manifest; refusing to replace it: ${CAMPAIGN_ROOT}"
  campaign_parent="$(dirname "${CAMPAIGN_ROOT}")"
  mkdir -p "${campaign_parent}"
  staging_root="${CAMPAIGN_ROOT}.staging.${SLURM_JOB_ID:-$$}"
  [[ ! -e "${staging_root}" ]] || \
    fail "Campaign staging root already exists: ${staging_root}"
  generate=(
    "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign generate
    --output "${staging_root}"
    --runtime-vorlap-root "${VORLAP_ROOT}"
  )
  if [[ -n "${RAW_REUSE_ROOT}" ]]; then
    generate+=(--raw-reuse-root "${RAW_REUSE_ROOT}")
  fi
  echo "Campaign is absent; generating and validating the exact 40-case campaign."
  "${generate[@]}"
  "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign validate \
    --campaign "${staging_root}"
  mv -T -- "${staging_root}" "${CAMPAIGN_ROOT}"
  echo "Published generated campaign: ${CAMPAIGN_ROOT}"
fi
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign validate \
  --campaign "${CAMPAIGN_ROOT}"

export OMP_NUM_THREADS="${THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ "${EHRMANN_FLIGHT_DRY_RUN}" == "0" ]]; then
  echo "Flight topology verified: ${physical_cores} physical cores, ${numa_domains} NUMA domains."
  echo "Slurm view: SLURM_CPUS_ON_NODE=${node_visible_cpus}, SLURM_JOB_CPUS_PER_NODE=${job_cpus_per_node}."
  echo "Launching ${WORKERS} dynamic one-core QBlade steps in ${allocated_cpus} requested campaign CPU slots."
else
  echo "Dry run: generating the complete 40-case srun launch plan without starting QBlade."
fi
scheduler=(
  "${PYTHON_EXE}" -m run_headless.run_ehrmann2014_flight
  --campaign "${CAMPAIGN_ROOT}"
  --qblade-root "${QBLADE_ROOT}"
  --python "${PYTHON_EXE}"
  --workers "${WORKERS}"
  --threads "${THREADS}"
  --allocated-cpus "${allocated_cpus}"
)
if [[ "${EHRMANN_FLIGHT_DRY_RUN}" == "1" ]]; then
  scheduler+=(--dry-run)
fi
"${scheduler[@]}"

if [[ "${EHRMANN_FLIGHT_DRY_RUN}" == "1" ]]; then
  "${PYTHON_EXE}" - "${CAMPAIGN_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
report = json.loads((root / "flight_scheduler_report.json").read_text(encoding="utf-8"))
assert len(manifest["cases"]) == 40
assert report["status"] == "dry_run"
assert len(report["cases"]) == 40
assert all(row["command"][0:7] == ["srun", "--exclusive", "--exact", "-N1", "-n1", "-c1", "--cpu-bind=cores"] for row in report["cases"])
print(json.dumps({
    "dry_run_verified": True,
    "campaign_fingerprint": manifest["fingerprint"],
    "case_count": len(manifest["cases"]),
    "planned_launch_count": len(report["cases"]),
}, indent=2))
PY
  exit 0
fi

"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign analyze \
  --campaign "${CAMPAIGN_ROOT}"
