#!/bin/bash
# Run only the 82 sealed H-VAWT cases outside the reviewed 20-case checkpoint.
# Scientific case construction/execution remains the byte-identical
# implementation recorded by the original campaign manifest.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --array=1,3-8,10,12-17,19,21-26,28,30-35,37,39-44,46,48-53,55,57-62,64,66-80,82,84-89,91,93-101
#SBATCH --time=47:59:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-complementary-dlc
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3}"
THREADS="${SLURM_CPUS_ON_NODE:-}"
THREADS="${THREADS%%(*}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-24000}"

fail() { echo "ERROR: $*" >&2; exit 2; }

if ! type module >/dev/null 2>&1; then
  for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash /opt/cray/pe/modules/default/init/bash; do
    if [[ -r "${init_script}" ]]; then source "${init_script}"; break; fi
  done
fi
type module >/dev/null 2>&1 || fail "Flight module command is unavailable."
module load anaconda3/2025.12-2

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "Submit this file as its declared 82-case complementary array."
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || fail "Invalid array index: ${SLURM_ARRAY_TASK_ID}."
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 102 )) || fail "Array index must be in 0..101."
case ",0,2,9,11,18,20,27,29,36,38,45,47,54,56,63,65,81,83,90,92," in
  *",${SLURM_ARRAY_TASK_ID},"*) fail "Case ${SLURM_ARRAY_TASK_ID} belongs to the immutable 20-case checkpoint." ;;
esac
[[ "${SLURM_JOB_NUM_NODES:-0}" == "1" ]] || fail "Each case requires exactly one Flight node."
[[ "${SLURM_NTASKS:-0}" == "1" ]] || fail "Each case uses one direct QBlade process."
[[ "${THREADS}" == "112" ]] || fail "The exclusive node reports ${SLURM_CPUS_ON_NODE:-unset} CPUs instead of the proven 112."
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || fail "Native Linux x86-64 is required."
[[ -x "${PYTHON_EXE}" ]] || fail "Successful Flight Python environment is unavailable: ${PYTHON_EXE}"
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root is unavailable: ${QBLADE_ROOT}"
[[ -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]] || fail "Sealed campaign is unavailable: ${CAMPAIGN_ROOT}"
[[ -f "${CAMPAIGN_ROOT}/flight_runtime_identity.json" ]] || fail "The sealed Flight runtime is not bound."
[[ -f "${CAMPAIGN_ROOT}/complementary_execution_authorization.json" ]] || fail "The reviewed complementary authorization is missing."
[[ -f "${CAMPAIGN_ROOT}/u50_complete_four_step_smoke.json" ]] || fail "Run the reviewed U50 four-step smoke before the long array."

qblade_library="${QBLADE_ROOT}/libQBladeCE_2.0.9.7.so.1.0.0"
expected_qblade_sha256="7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
[[ -f "${qblade_library}" ]] || fail "QBlade library is unavailable: ${qblade_library}"
observed_qblade_sha256="$(sha256sum "${qblade_library}" | awk '{print $1}')"
[[ "${observed_qblade_sha256}" == "${expected_qblade_sha256}" ]] || fail "QBlade library SHA-256 differs from the sealed Flight runtime."

export VORLAP_ROOT PYTHON_EXE QBLADE_ROOT CAMPAIGN_ROOT
export PYTHONPATH="${VORLAP_ROOT}:${QBLADE_ROOT}/SIL_Interface${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${QBLADE_ROOT}:${QBLADE_ROOT}/Libraries:${QBLADE_ROOT}/ControllerFiles:${QBLADE_ROOT}/Binaries:${QBLADE_ROOT}/Binaries/IPCBridge${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VORLAP_QBLADE_LOADER_ENV=1
export OMP_NUM_THREADS="${THREADS}"

cd "${VORLAP_ROOT}"
echo "Using sealed scientific executor with reviewed continuation authorization: complementary case=${SLURM_ARRAY_TASK_ID}."
"${PYTHON_EXE}" run_headless/hvawt_complementary_dlc_campaign.py run \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" \
  --threads "${THREADS}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --progress-every "${PROGRESS_EVERY}" \
  --python "${PYTHON_EXE}"
