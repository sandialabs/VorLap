#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=ehrmann-work-map
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/ehrmann2014_naca0018_work_map_validation_v2}"
WORK_MAP="${WORK_MAP:-}"
RAW_REUSE_ROOT="${RAW_REUSE_ROOT:-}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-${CAMPAIGN_ROOT}/validation_evidence}"
EHRMANN_WORK_MAP_FLIGHT_DRY_RUN="${EHRMANN_WORK_MAP_FLIGHT_DRY_RUN:-0}"
WORKERS=20
THREADS=1
EXPECTED_ALLOCATED_CPUS=20
EXPECTED_NODE_PHYSICAL_CORES=112
EXPECTED_NUMA_DOMAINS=8

fail() { echo "ERROR: $*" >&2; exit 2; }
[[ "${EHRMANN_WORK_MAP_FLIGHT_DRY_RUN}" == "0" || "${EHRMANN_WORK_MAP_FLIGHT_DRY_RUN}" == "1" ]] || \
  fail "EHRMANN_WORK_MAP_FLIGHT_DRY_RUN must be exactly 0 or 1."
[[ -n "${WORK_MAP}" ]] || fail "WORK_MAP is required for the schema-v2 Ehrmann run."
[[ -f "${WORK_MAP}" && ! -L "${WORK_MAP}" ]] || \
  fail "WORK_MAP must be a regular non-symlink file: ${WORK_MAP}"
[[ -n "${RAW_REUSE_ROOT}" ]] || \
  fail "RAW_REUSE_ROOT is required and must contain the 20 frozen raw result directories."
[[ -d "${RAW_REUSE_ROOT}" && ! -L "${RAW_REUSE_ROOT}" ]] || \
  fail "RAW_REUSE_ROOT must be a real non-symlink directory: ${RAW_REUSE_ROOT}"

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
    'import h5py, matplotlib, numpy, scipy; from examples.ehrmann2014_naca0018 import campaign, build_validation_evidence; from run_headless import run_ehrmann2014_work_map_flight; from vorlap import qblade_run' >/dev/null 2>&1; then
    PYTHON_EXE="${resolved_python}"
    break
  fi
  echo "WARNING: Python failed the Ehrmann work-map import preflight: ${resolved_python}" >&2
done
[[ -n "${PYTHON_EXE}" ]] || fail "No Python can import the current Ehrmann work-map runtime."
export PYTHON_EXE
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || \
  fail "Ehrmann work-map validation requires native Linux x86-64."
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root unavailable: ${QBLADE_ROOT}"
[[ "${SLURM_JOB_NUM_NODES:-}" == "1" ]] || fail "Exactly one node is required."
[[ "${SLURM_NTASKS:-}" == "${WORKERS}" ]] || \
  fail "Expected ${WORKERS} allocated tasks; got ${SLURM_NTASKS:-unset}."
[[ "${SLURM_CPUS_PER_TASK:-}" == "${THREADS}" ]] || \
  fail "Expected ${THREADS} physical CPU per case; got ${SLURM_CPUS_PER_TASK:-unset}."
[[ "${SLURM_NTASKS}" =~ ^[0-9]+$ && "${SLURM_CPUS_PER_TASK}" =~ ^[0-9]+$ ]] || \
  fail "Slurm task/CPU allocation fields must be positive integers."
allocated_cpus="$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
[[ "${allocated_cpus}" == "${EXPECTED_ALLOCATED_CPUS}" ]] || \
  fail "Expected ${EXPECTED_ALLOCATED_CPUS} campaign CPU slots; got ${allocated_cpus}."
node_visible_cpus="${SLURM_CPUS_ON_NODE:-unset}"
job_cpus_per_node="${SLURM_JOB_CPUS_PER_NODE:-unset}"

physical_cores="$(lscpu -p=CORE,SOCKET | awk -F, '!/^#/ {print $1 "," $2}' | sort -u | wc -l | tr -d ' ')"
numa_domains="$(lscpu -p=NODE | awk -F, '!/^#/ && $1 >= 0 {print $1}' | sort -nu | wc -l | tr -d ' ')"
if [[ "${EHRMANN_WORK_MAP_FLIGHT_DRY_RUN}" == "0" ]]; then
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
  echo "Campaign is absent; generating schema-v2 from the exact map and 20 frozen raw cases."
  "${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign generate \
    --output "${staging_root}" \
    --runtime-vorlap-root "${VORLAP_ROOT}" \
    --raw-reuse-root "${RAW_REUSE_ROOT}" \
    --work-map "${WORK_MAP}"
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

if [[ "${EHRMANN_WORK_MAP_FLIGHT_DRY_RUN}" == "0" ]]; then
  echo "Flight topology verified: ${physical_cores} physical cores, ${numa_domains} NUMA domains."
  echo "Slurm view: SLURM_CPUS_ON_NODE=${node_visible_cpus}, SLURM_JOB_CPUS_PER_NODE=${job_cpus_per_node}."
  echo "Launching at most 20 one-core work-map hybrid steps; all 20 requested slots are occupied on a fresh campaign."
else
  echo "Dry run: generating the hybrid-only srun launch plan without starting QBlade."
fi
scheduler=(
  "${PYTHON_EXE}" -m run_headless.run_ehrmann2014_work_map_flight
  --campaign "${CAMPAIGN_ROOT}"
  --qblade-root "${QBLADE_ROOT}"
  --work-map "${WORK_MAP}"
  --raw-reuse-root "${RAW_REUSE_ROOT}"
  --python "${PYTHON_EXE}"
  --workers "${WORKERS}"
  --threads "${THREADS}"
  --allocated-cpus "${allocated_cpus}"
)
if [[ "${EHRMANN_WORK_MAP_FLIGHT_DRY_RUN}" == "1" ]]; then
  scheduler+=(--dry-run)
fi
"${scheduler[@]}"

if [[ "${EHRMANN_WORK_MAP_FLIGHT_DRY_RUN}" == "1" ]]; then
  "${PYTHON_EXE}" - "${CAMPAIGN_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.loads((root / "campaign_manifest.json").read_text(encoding="utf-8"))
report = json.loads((root / "flight_scheduler_report.json").read_text(encoding="utf-8"))
contract = report["execution_contract"]
assert manifest["schema"] == "vorlap-ehrmann2014-naca0018-work-map-campaign-v2"
assert report["schema"] == "vorlap-ehrmann2014-flight-scheduler-v1"
assert report["status"] == "dry_run"
assert contract["exact_raw_reuse_case_count"] == 20
assert contract["hybrid_case_count"] == 20
assert contract["hybrid_complete_at_start"] + len(report["cases"]) == 20
assert all(row["case_id"].endswith("_hybrid") for row in report["cases"])
assert all(row["command"][0:7] == ["srun", "--exclusive", "--exact", "-N1", "-n1", "-c1", "--cpu-bind=cores"] for row in report["cases"])
print(json.dumps({
    "dry_run_verified": True,
    "campaign_fingerprint": manifest["fingerprint"],
    "exact_raw_reuse_case_count": 20,
    "complete_hybrid_case_count": contract["hybrid_complete_at_start"],
    "planned_hybrid_launch_count": len(report["cases"]),
}, indent=2))
PY
  exit 0
fi

"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.campaign analyze \
  --campaign "${CAMPAIGN_ROOT}"
mkdir -p "${EVIDENCE_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CAMPAIGN_ROOT}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
"${PYTHON_EXE}" -m examples.ehrmann2014_naca0018.build_validation_evidence \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --evidence-root "${EVIDENCE_ROOT}" \
  --no-paper-copy
"${PYTHON_EXE}" - "${EVIDENCE_ROOT}/ehrmann2014_validation_metrics.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
metrics = json.loads(path.read_text(encoding="utf-8"))
assert metrics["schema"] == "vorlap-ehrmann2014-work-map-validation-evidence-v2"
assert metrics["completed_cases"] == 40
print(json.dumps({
    "validation_evidence": str(path),
    "status": metrics["status"],
    "primary_acceptance_pass": metrics["primary_acceptance"]["pass"],
    "failed_checks": metrics["primary_acceptance"]["failed_checks"],
}, indent=2))
if metrics["primary_acceptance"]["pass"] is not True:
    raise SystemExit(2)
PY
