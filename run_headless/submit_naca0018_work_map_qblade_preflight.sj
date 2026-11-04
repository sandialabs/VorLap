#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=naca0018-qb-preflight
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

VORLAP_ROOT="${VORLAP_ROOT:-/pscratch/kevmoor/vorlap}"
export VORLAP_ROOT
export QBLADE_STAGE=preflight

common="${VORLAP_ROOT}/run_headless/submit_naca0018_work_map_qblade.sj"
[[ -f "${common}" ]] || {
  echo "ERROR: Common prescribed-motion QBlade wrapper unavailable: ${common}" >&2
  exit 2
}
exec /bin/bash "${common}"
