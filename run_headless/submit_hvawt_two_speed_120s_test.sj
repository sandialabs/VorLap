#!/bin/bash
# Four matched H0/complete-v8, 120 s tests at the frozen mode-6 and mode-12
# lock-in peaks. Each array task owns one exclusive Flight node and one QBlade
# process. The 12 h wall allows margin above the historical 6.92 h native
# 16,000-step result while keeping both hypothesis pairs independent.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --array=0-3%4
#SBATCH --time=12:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-v8-2speed
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_unified_h0_complete_two_speed_120s_test}"
THREADS="${SLURM_CPUS_ON_NODE:-}"
THREADS="${THREADS%%(*}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-24000}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000}"

fail() { echo "ERROR: $*" >&2; exit 2; }

load_successful_flight_module() {
  local init_script
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
}

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "Submit this file as its declared 0-3 array."
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-3]$ ]] || fail "Array index must be 0, 1, 2, or 3."
[[ "${SLURM_JOB_NUM_NODES:-0}" == "1" ]] || fail "Each case requires exactly one Flight node."
[[ "${SLURM_NTASKS:-0}" == "1" ]] || fail "Each case uses one direct QBlade process."
[[ "${THREADS}" == "112" ]] || fail "The exclusive node reports ${SLURM_CPUS_ON_NODE:-unset} CPUs instead of 112."
[[ "${CHECKPOINT_EVERY}" == "24000" ]] || fail "Flight recovery saves must use the frozen 24,000-step interval."
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || fail "Native Linux x86-64 is required."

load_successful_flight_module
[[ -x "${PYTHON_EXE}" ]] || fail "Successful Flight Python environment is unavailable: ${PYTHON_EXE}"
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root is unavailable: ${QBLADE_ROOT}"
[[ -f "${CAMPAIGN_ROOT}/campaign_manifest.json" ]] || fail "Generate the sealed two-speed campaign first: ${CAMPAIGN_ROOT}"
[[ -f "${CAMPAIGN_ROOT}/flight_qblade_runtime_identity.json" ]] || fail "Prepare and bind the two-speed runtime receipt before submission."

qblade_library="${QBLADE_ROOT}/libQBladeCE_2.0.9.7.so.1.0.0"
expected_qblade_sha256="7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
[[ -f "${qblade_library}" ]] || fail "QBlade library is unavailable: ${qblade_library}"
observed_qblade_sha256="$(sha256sum "${qblade_library}" | awk '{print $1}')"
[[ "${observed_qblade_sha256}" == "${expected_qblade_sha256}" ]] || fail "QBlade library SHA-256 differs from the successful Flight runtime."

export VORLAP_ROOT PYTHON_EXE QBLADE_ROOT CAMPAIGN_ROOT
export PYTHONPATH="${VORLAP_ROOT}:${QBLADE_ROOT}/SIL_Interface${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${QBLADE_ROOT}:${QBLADE_ROOT}/Libraries:${QBLADE_ROOT}/ControllerFiles:${QBLADE_ROOT}/Binaries:${QBLADE_ROOT}/Binaries/IPCBridge${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VORLAP_QBLADE_LOADER_ENV=1
export VORLAP_HVAWT_PROFILE="${VORLAP_HVAWT_PROFILE:-1}"
export OMP_NUM_THREADS="${THREADS}"

cd "${VORLAP_ROOT}"
"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py validate --campaign-root "${CAMPAIGN_ROOT}"
"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py bind-runtime --campaign-root "${CAMPAIGN_ROOT}" --qblade-root "${QBLADE_ROOT}"
"${PYTHON_EXE}" run_headless/hvawt_response_pilot.py run \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}" \
  --case-index "${SLURM_ARRAY_TASK_ID}" \
  --threads "${THREADS}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --progress-every "${PROGRESS_EVERY}" \
  --archive-partial-and-restart \
  --python "${PYTHON_EXE}"

echo "Completed matched H0/complete test case ${SLURM_ARRAY_TASK_ID}; analyze after all four array tasks finish."
