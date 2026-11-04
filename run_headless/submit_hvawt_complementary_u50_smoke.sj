#!/bin/bash
# Four exact U50 complete-path steps using the sealed scientific executor.
# This gates input/runtime identity, finite nonzero residual/total force, and
# the recorded energy-closure checks; it is not response or load evidence.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-cont-u50-smoke
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3}"
THREADS="${SLURM_CPUS_ON_NODE:-}"
THREADS="${THREADS%%(*}"

fail() { echo "ERROR: $*" >&2; exit 2; }

if ! type module >/dev/null 2>&1; then
  for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash /opt/cray/pe/modules/default/init/bash; do
    if [[ -r "${init_script}" ]]; then source "${init_script}"; break; fi
  done
fi
type module >/dev/null 2>&1 || fail "Flight module command is unavailable."
module load anaconda3/2025.12-2

[[ "${SLURM_JOB_NUM_NODES:-0}" == "1" ]] || fail "The U50 smoke requires one node."
[[ "${SLURM_NTASKS:-0}" == "1" ]] || fail "The U50 smoke uses one direct process."
[[ "${THREADS}" == "112" ]] || fail "The exclusive node reports ${SLURM_CPUS_ON_NODE:-unset} CPUs instead of the proven 112."
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || fail "Native Linux x86-64 is required."
[[ -x "${PYTHON_EXE}" ]] || fail "Successful Flight Python environment is unavailable: ${PYTHON_EXE}"
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root is unavailable: ${QBLADE_ROOT}"
[[ -f "${CAMPAIGN_ROOT}/complementary_execution_authorization.json" ]] || fail "Bind the reviewed turbulent-checkpoint continuation before the U50 smoke."

export PYTHONPATH="${VORLAP_ROOT}:${QBLADE_ROOT}/SIL_Interface${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${QBLADE_ROOT}:${QBLADE_ROOT}/Libraries:${QBLADE_ROOT}/ControllerFiles:${QBLADE_ROOT}/Binaries:${QBLADE_ROOT}/Binaries/IPCBridge${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VORLAP_QBLADE_LOADER_ENV=1
export OMP_NUM_THREADS="${THREADS}"
cd "${VORLAP_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_complementary_dlc_campaign.py smoke-u50 \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}" \
  --threads "${THREADS}" \
  --python "${PYTHON_EXE}"
