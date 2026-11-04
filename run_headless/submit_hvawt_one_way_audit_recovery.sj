#!/bin/bash
# Read-only audit of the 34 completed one-way cases plus one additive receipt.
# No QBlade executable, simulation runner, archive path, or Slurm array is used.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-oneway-recover
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3}"

fail() { echo "ERROR: $*" >&2; exit 2; }

if ! type module >/dev/null 2>&1; then
  for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash /opt/cray/pe/modules/default/init/bash; do
    if [[ -r "${init_script}" ]]; then source "${init_script}"; break; fi
  done
fi
type module >/dev/null 2>&1 || fail "Flight module command is unavailable."
module load anaconda3/2025.12-2

[[ -x "${PYTHON_EXE}" ]] || fail "Flight Python environment is unavailable: ${PYTHON_EXE}"
[[ -s "${CAMPAIGN_ROOT}/campaign_manifest.json" ]] || fail "Sealed campaign is unavailable: ${CAMPAIGN_ROOT}"
[[ -s "${CAMPAIGN_ROOT}/analysis.json" ]] || fail "Original 68/102 analysis is unavailable."
[[ ! -e "${CAMPAIGN_ROOT}/analysis_recovered_v3.json" ]] || fail "Recovered analysis already exists; use the analysis launcher's idempotent verification."

export PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "${VORLAP_ROOT}"

"${PYTHON_EXE}" run_headless/recover_hvawt_completed_one_way.py preflight \
  --campaign-root "${CAMPAIGN_ROOT}"
"${PYTHON_EXE}" run_headless/recover_hvawt_completed_one_way.py recover \
  --campaign-root "${CAMPAIGN_ROOT}"

echo "Recovered 34 completed one-way audit-only cases without running QBlade."
