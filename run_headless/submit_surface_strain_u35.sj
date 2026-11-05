#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=47:59:00
#SBATCH --account=FY260158
#SBATCH --job-name=vorlap-u35-surface
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevmoor@sandia.gov

set -euo pipefail

# Re-run only the governing full-campaign realization with dense structural
# output.  The wind, duration, timestep, idling speed, wake reduction, and SIL
# source policies match the approved U35/seed-4433458 production case.  A new
# root is required because the original result filter retained only four BLD_1
# stations and missing channels cannot be recovered from its compact export.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VORLAP_ROOT="${VORLAP_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"

export VORLAP_ROOT
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/parked_dlc_surface_strain_u35_s4433458_20260808}"
export WIND_SPEEDS="35"
export QBLADE_AZIMUTHS="0"
export CAMPAIGN_SCENARIOS="baseline vorlap"
export QBLADE_RPM="1"
export QBLADE_DURATION="660"
export QBLADE_TIMESTEP="0.0075"
export QBLADE_STORE_FROM="60"
export QBLADE_STORE_REPLAY="false"
export QBLADE_CHECKPOINT_EVERY="4000"
export QBLADE_PROGRESS_EVERY="10"
export QBLADE_WIND_NAME="/pscratch/kevmoor/vorlap_wind_fields/dlc6p4/U35p00ms/dlc6p4_U35p00ms_s4433458.bts"
export QBLADE_FULL_STRUCTURE_STRAIN_OUTPUT="true"
export QBLADE_RESULT_FILTER=""
export QBLADE_STORE_PROJECT="0"
export QBLADE_WAKE_PROFILE="parked-reduced-v2"
export VORLAP_SOURCE_BOUNDS_POLICY="raise"
export VORLAP_SOURCE_FREQUENCY_TRUNCATION="nyquist"
export SKIP_VORLAP_INSTALL="1"
export SKIP_POSTPROCESS="1"
export RUN_OWENS_PREPROCESS="0"
export RUN_PLAN_AUDIT="0"
export RUN_SETUP_AUDIT="0"
export RUN_FINAL_AUDIT="0"
export REQUIRE_PRODUCTION_MATRIX="0"
export AUDIT_MIN_ANALYZED_DURATION="600"

exec bash "${SCRIPT_DIR}/submit_parked_demo_campaign.sj"
