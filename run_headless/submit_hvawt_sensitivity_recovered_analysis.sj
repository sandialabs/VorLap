#!/bin/bash
# Reproduce the audit receipt, analyze six cases, and publish versioned outputs.
# This launcher has no simulation executable, archive path, or Slurm array.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-sens-analysis-v2
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
[[ -x "${PYTHON_EXE}" ]] || fail "Flight Python is unavailable: ${PYTHON_EXE}"
[[ -s "${BASE_CAMPAIGN_ROOT}/campaign_manifest.json" ]] || fail "Accepted v3 campaign is unavailable."
[[ -s "${SENSITIVITY_ROOT}/sensitivity_campaign_manifest.json" ]] || fail "Sensitivity campaign is unavailable."
[[ -s "${SENSITIVITY_ROOT}/sensitivity_terminal_audit_recovery_v2.json" ]] || fail "Sensitivity recovery receipt is unavailable."
[[ ! -e "${SENSITIVITY_ROOT}/sensitivity_analysis_v1.json" ]] || fail "Unversioned sensitivity analysis exists; stop for review."
[[ ! -e "${SENSITIVITY_ROOT}/paper_export_sensitivity_v1" ]] || fail "Unversioned sensitivity promotion exists; stop for review."

export VORLAP_ROOT PYTHON_EXE BASE_CAMPAIGN_ROOT SENSITIVITY_ROOT
export PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SENSITIVITY_ROOT}/.matplotlib_recovered_v2}"
mkdir -p "${MPLCONFIGDIR}"

cd "${VORLAP_ROOT}"
"${PYTHON_EXE}" run_headless/recover_hvawt_sensitivity_results.py analyze \
  --campaign-root "${SENSITIVITY_ROOT}" \
  --base-campaign-root "${BASE_CAMPAIGN_ROOT}"
"${PYTHON_EXE}" run_headless/recover_hvawt_sensitivity_results.py promote \
  --campaign-root "${SENSITIVITY_ROOT}"

echo "Completed recovered sensitivity analysis and versioned paper promotion."
