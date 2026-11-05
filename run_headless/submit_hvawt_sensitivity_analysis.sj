#!/bin/bash
# Deep-audit all six terminal cases, evaluate the preregistered numerical gates
# and response-ablation diagnostics, and publish the claim-bounded handoff.
# This job has no QBlade execution path.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-sens-analysis
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
BASE_CAMPAIGN_ROOT="${BASE_CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3}"
SENSITIVITY_ROOT="${SENSITIVITY_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_targeted_sensitivity_v1}"

fail() { echo "ERROR: $*" >&2; exit 2; }

if ! type module >/dev/null 2>&1; then
  for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash /opt/cray/pe/modules/default/init/bash; do
    if [[ -r "${init_script}" ]]; then source "${init_script}"; break; fi
  done
fi
type module >/dev/null 2>&1 || fail "Flight module command is unavailable."
module load anaconda3/2025.12-2

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || fail "Native Linux x86-64 is required."
[[ -x "${PYTHON_EXE}" ]] || fail "Flight Python environment is unavailable: ${PYTHON_EXE}"
[[ -s "${BASE_CAMPAIGN_ROOT}/campaign_manifest.json" ]] || fail "Accepted v3 campaign is unavailable."
[[ -s "${SENSITIVITY_ROOT}/sensitivity_campaign_manifest.json" ]] || fail "Sensitivity campaign is unavailable."
[[ -s "${SENSITIVITY_ROOT}/sensitivity_execution_authorization.json" ]] || fail "Sensitivity execution authorization is unavailable."

export VORLAP_ROOT PYTHON_EXE BASE_CAMPAIGN_ROOT SENSITIVITY_ROOT
export PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${VORLAP_ROOT}"
"${PYTHON_EXE}" run_headless/hvawt_sensitivity_campaign.py analyze \
  --campaign-root "${SENSITIVITY_ROOT}" \
  --base-campaign-root "${BASE_CAMPAIGN_ROOT}" \
  --output "${SENSITIVITY_ROOT}/sensitivity_analysis_v1.json"

"${PYTHON_EXE}" run_headless/hvawt_sensitivity_campaign.py promote \
  --campaign-root "${SENSITIVITY_ROOT}" \
  --analysis "${SENSITIVITY_ROOT}/sensitivity_analysis_v1.json" \
  --output-dir "${SENSITIVITY_ROOT}/paper_export_sensitivity_v1"

echo "Completed six-case sensitivity audit, gate evaluation, and compact promotion."
