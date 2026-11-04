#!/bin/bash
# Submit with --dependency=afterok:<full-array-job-id>.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-v8-analysis
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_hybrid_full_660s_v8}"

fail() { echo "ERROR: $*" >&2; exit 2; }

if ! type module >/dev/null 2>&1; then
  for init_script in \
    /etc/profile.d/modules.sh \
    /usr/share/Modules/init/bash \
    /usr/share/lmod/lmod/init/bash \
    /opt/cray/pe/modules/default/init/bash; do
    if [[ -r "${init_script}" ]]; then
      source "${init_script}"
      break
    fi
  done
fi
type module >/dev/null 2>&1 || fail "Flight module command is unavailable."
module load anaconda3/2025.12-2
[[ -x "${PYTHON_EXE}" ]] || { echo "ERROR: Successful Flight Python environment is unavailable: ${PYTHON_EXE}" >&2; exit 2; }

export PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${VORLAP_ROOT}"
"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py validate --campaign-root "${CAMPAIGN_ROOT}"
"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py analyze --campaign-root "${CAMPAIGN_ROOT}"
