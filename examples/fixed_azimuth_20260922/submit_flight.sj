#!/bin/bash
# One matched model per node. Submit through submit_fixed_azimuth_chain.sh.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --array=0-2%3
#SBATCH --time=47:59:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --job-name=viv-fixed-u11p25

set -euo pipefail
VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
: "${CAMPAIGN_ROOT:?Set CAMPAIGN_ROOT to the fresh prepared fixed-azimuth campaign}"
: "${FLIGHT_SOURCE_COMMIT:?Submit through run_headless/submit_fixed_azimuth_chain.sh}"
: "${SLURM_ARRAY_TASK_ID:?Submit the declared three-task array}"
[[ "$(git -C "${VORLAP_ROOT}" rev-parse HEAD)" == "${FLIGHT_SOURCE_COMMIT}" ]] || {
  echo "Source commit changed after submission; prepare a fresh campaign." >&2; exit 2;
}
git -C "${VORLAP_ROOT}" diff --quiet HEAD -- \
  vorlap run_headless/config run_headless/run_qblade_sim.py \
  run_headless/submit_fixed_azimuth_chain.sh examples/fixed_azimuth_20260922 \
  run_headless/flight_environment.sh || {
  echo "Runtime/submission files changed after submission." >&2; exit 2;
}
THREADS="${SLURM_CPUS_ON_NODE:-}"
THREADS="${THREADS%%(*}"
[[ "${SLURM_JOB_NUM_NODES:-0}" == "1" && "${SLURM_NTASKS:-0}" == "1" && "${THREADS}" == "112" ]] || {
  echo "Expected the proven one-process, exclusive 112-CPU Flight node." >&2; exit 2;
}
if ! type module >/dev/null 2>&1; then
  for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash /opt/cray/pe/modules/default/init/bash; do
    if [[ -r "${init_script}" ]]; then source "${init_script}"; break; fi
  done
fi
module load anaconda3/2025.12-2
library="${QBLADE_ROOT}/libQBladeCE_2.0.9.7.so.1.0.0"
expected_sha="7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
[[ "$(sha256sum "${library}" | awk '{print $1}')" == "${expected_sha}" ]] || {
  echo "QBlade runtime differs from the verified campaign." >&2; exit 2;
}
export VORLAP_ROOT PYTHON_EXE QBLADE_ROOT CAMPAIGN_ROOT
export PYTHONPATH="${VORLAP_ROOT}:${QBLADE_ROOT}/SIL_Interface${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${QBLADE_ROOT}:${QBLADE_ROOT}/Libraries:${QBLADE_ROOT}/ControllerFiles:${QBLADE_ROOT}/Binaries:${QBLADE_ROOT}/Binaries/IPCBridge${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VORLAP_QBLADE_LOADER_ENV=1
export QT_QPA_PLATFORM=offscreen PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "${VORLAP_ROOT}"
exec "${PYTHON_EXE}" examples/fixed_azimuth_20260922/campaign.py run \
  --campaign-root "${CAMPAIGN_ROOT}" --qblade-root "${QBLADE_ROOT}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" --threads "${THREADS}"
