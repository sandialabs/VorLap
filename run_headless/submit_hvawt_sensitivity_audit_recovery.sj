#!/bin/bash
# Read-only audit of six completed sensitivity cases plus one additive receipt.
# This launcher has no simulation executable, archive path, or Slurm array.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-sens-recover
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
PYTHON_EXE="${PYTHON_EXE:-${VORLAP_ROOT}/.venv/bin/python}"
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
[[ -s "${SENSITIVITY_ROOT}/sensitivity_campaign_manifest.json" ]] || fail "Sensitivity campaign is unavailable."
[[ -s "${SENSITIVITY_ROOT}/sensitivity_execution_authorization.json" ]] || fail "Sensitivity authorization is unavailable."
[[ ! -e "${SENSITIVITY_ROOT}/sensitivity_analysis_v1.json" ]] || fail "Unversioned sensitivity analysis exists; stop for review."
[[ ! -e "${SENSITIVITY_ROOT}/paper_export_sensitivity_v1" ]] || fail "Unversioned sensitivity promotion exists; stop for review."

export VORLAP_ROOT PYTHON_EXE SENSITIVITY_ROOT
export PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/kevmoor-hvawt-sensitivity-recovery-${SLURM_JOB_ID:-manual}}"
mkdir -p "${MPLCONFIGDIR}"

cd "${VORLAP_ROOT}"
"${PYTHON_EXE}" run_headless/recover_hvawt_sensitivity_results.py preflight \
  --campaign-root "${SENSITIVITY_ROOT}"
"${PYTHON_EXE}" run_headless/recover_hvawt_sensitivity_results.py recover \
  --campaign-root "${SENSITIVITY_ROOT}"

echo "Recovered two completed half-timestep audit-only cases without rerunning a simulation."
