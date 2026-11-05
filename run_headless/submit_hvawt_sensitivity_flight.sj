#!/bin/bash
# Run the six sealed targeted-sensitivity cases, one direct QBlade process per
# exclusive Flight node. Partial attempts are archived and restarted fresh;
# no QBlade continuation checkpoint is accepted as scientific evidence.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --array=0-5%6
#SBATCH --time=47:59:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-sensitivity
#SBATCH --output=slurm-%A_%a.out
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
SENSITIVITY_ROOT="${SENSITIVITY_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_targeted_sensitivity_v1}"
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

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "Submit this file as its declared 0-5 array."
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || fail "Invalid array index: ${SLURM_ARRAY_TASK_ID}."
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 6 )) || fail "Array index must be in 0..5."
[[ "${SLURM_JOB_NUM_NODES:-0}" == "1" ]] || fail "Each case requires exactly one Flight node."
[[ "${SLURM_NTASKS:-0}" == "1" ]] || fail "Each case uses one direct QBlade process."
[[ "${THREADS}" == "112" ]] || fail "The exclusive node reports ${SLURM_CPUS_ON_NODE:-unset} CPUs instead of the proven 112."
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || fail "Native Linux x86-64 is required."
[[ -x "${PYTHON_EXE}" ]] || fail "Flight Python environment is unavailable: ${PYTHON_EXE}"
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root is unavailable: ${QBLADE_ROOT}"
[[ -s "${SENSITIVITY_ROOT}/sensitivity_campaign_manifest.json" ]] || fail "Sensitivity campaign is unavailable."
[[ -s "${SENSITIVITY_ROOT}/flight_runtime_identity.json" ]] || fail "Sensitivity runtime binding is unavailable."
[[ -s "${SENSITIVITY_ROOT}/sensitivity_smoke_receipt.json" ]] || fail "Six-case smoke receipt is unavailable."
[[ -s "${SENSITIVITY_ROOT}/sensitivity_execution_authorization.json" ]] || fail "Sensitivity execution authorization is unavailable."

qblade_library="${QBLADE_ROOT}/libQBladeCE_2.0.9.7.so.1.0.0"
expected_qblade_sha256="7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
[[ -f "${qblade_library}" ]] || fail "QBlade library is unavailable: ${qblade_library}"
observed_qblade_sha256="$(sha256sum "${qblade_library}" | awk '{print $1}')"
[[ "${observed_qblade_sha256}" == "${expected_qblade_sha256}" ]] || fail "QBlade library SHA-256 differs from the bound Flight runtime."

export VORLAP_ROOT PYTHON_EXE QBLADE_ROOT SENSITIVITY_ROOT
export PYTHONPATH="${VORLAP_ROOT}:${QBLADE_ROOT}/SIL_Interface${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${QBLADE_ROOT}:${QBLADE_ROOT}/Libraries:${QBLADE_ROOT}/ControllerFiles:${QBLADE_ROOT}/Binaries:${QBLADE_ROOT}/Binaries/IPCBridge${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VORLAP_QBLADE_LOADER_ENV=1
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${VORLAP_ROOT}"
echo "Running sealed targeted sensitivity case ${SLURM_ARRAY_TASK_ID} with one direct QBlade process."
"${PYTHON_EXE}" run_headless/hvawt_sensitivity_campaign.py run \
  --campaign-root "${SENSITIVITY_ROOT}" \
  --qblade-root "${QBLADE_ROOT}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" \
  --threads "${THREADS}" \
  --python "${PYTHON_EXE}"
