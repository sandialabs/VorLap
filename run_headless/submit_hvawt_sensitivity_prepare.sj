#!/bin/bash
# Generate, deep-validate, smoke, and authorize the separate six-case
# sensitivity campaign on a scheduled Flight node. This job performs no
# production-length simulation.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-sens-prepare
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
QBLADE_ROOT="${QBLADE_ROOT:-${VORLAP_ROOT}/run_headless/QBladeCE_2.0.9.7}"
BASE_CAMPAIGN_ROOT="${BASE_CAMPAIGN_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_complete_dlc_three_path_v3}"
SENSITIVITY_ROOT="${SENSITIVITY_ROOT:-/pscratch/kevmoor/vorlap_runs/hvawt_targeted_sensitivity_v1}"
PREPARING_ROOT="${SENSITIVITY_ROOT}.preparing.${SLURM_JOB_ID}"
INSTALL_LOCK="${SENSITIVITY_ROOT}.install.lock"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

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
[[ -d "${QBLADE_ROOT}" ]] || fail "QBlade root is unavailable: ${QBLADE_ROOT}"
[[ -s "${BASE_CAMPAIGN_ROOT}/campaign_manifest.json" ]] || fail "Accepted v3 campaign is unavailable."
[[ -s "${BASE_CAMPAIGN_ROOT}/hvawt_sensitivity_selection_v1.json" ]] || fail "Accepted row-selection receipt is unavailable."
[[ ! -e "${SENSITIVITY_ROOT}" ]] || fail "Final sensitivity root already exists: ${SENSITIVITY_ROOT}"
[[ ! -e "${PREPARING_ROOT}" ]] || fail "Preparation staging root already exists: ${PREPARING_ROOT}"
[[ ! -e "${INSTALL_LOCK}" ]] || fail "Sensitivity install lock already exists: ${INSTALL_LOCK}"

qblade_library="${QBLADE_ROOT}/libQBladeCE_2.0.9.7.so.1.0.0"
expected_qblade_sha256="7d4d07401dddf65dcf25d7cb7d809a78fbf8ab9ecad629aaaf6d2a4308a0764d"
[[ -f "${qblade_library}" ]] || fail "QBlade library is unavailable: ${qblade_library}"
observed_qblade_sha256="$(sha256sum "${qblade_library}" | awk '{print $1}')"
[[ "${observed_qblade_sha256}" == "${expected_qblade_sha256}" ]] || fail "QBlade library SHA-256 differs from the accepted Flight runtime."

export VORLAP_ROOT PYTHON_EXE QBLADE_ROOT BASE_CAMPAIGN_ROOT SENSITIVITY_ROOT
export PYTHONPATH="${VORLAP_ROOT}:${QBLADE_ROOT}/SIL_Interface${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${QBLADE_ROOT}:${QBLADE_ROOT}/Libraries:${QBLADE_ROOT}/ControllerFiles:${QBLADE_ROOT}/Binaries:${QBLADE_ROOT}/Binaries/IPCBridge${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VORLAP_QBLADE_LOADER_ENV=1
export OMP_NUM_THREADS="${THREADS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$(dirname "${SENSITIVITY_ROOT}")"
cd "${VORLAP_ROOT}"

"${PYTHON_EXE}" run_headless/hvawt_sensitivity_campaign.py prepare \
  --campaign-root "${PREPARING_ROOT}" \
  --base-campaign-root "${BASE_CAMPAIGN_ROOT}" \
  --qblade-root "${QBLADE_ROOT}" \
  --threads "${THREADS}" \
  --python "${PYTHON_EXE}"

# Every campaign-owned artifact identity is relative to the campaign root, so
# an atomic same-filesystem rename preserves the receipts while preventing a
# partially prepared root from becoming executable.
install_lock_acquired=0
release_install_lock() {
  if [[ "${install_lock_acquired}" == "1" ]]; then
    rmdir "${INSTALL_LOCK}" || true
  fi
}
trap release_install_lock EXIT
mkdir "${INSTALL_LOCK}" || fail "Another sensitivity preparation owns the install lock."
install_lock_acquired=1
[[ ! -e "${SENSITIVITY_ROOT}" ]] || fail "Final sensitivity root appeared during preparation."
mv "${PREPARING_ROOT}" "${SENSITIVITY_ROOT}"
rmdir "${INSTALL_LOCK}"
install_lock_acquired=0
trap - EXIT
"${PYTHON_EXE}" run_headless/hvawt_sensitivity_campaign.py validate \
  --campaign-root "${SENSITIVITY_ROOT}"

echo "Prepared, smoked, and authorized six sensitivity cases without production execution."
