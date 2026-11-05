#!/bin/bash
# Verify the one-way recovery receipt, build v3 analysis, then postprocess 102 cases.
# This is analysis-only: it contains no QBlade execution path or Slurm array.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --partition=batch
#SBATCH --account=FY260158
#SBATCH --qos=normal
#SBATCH --reservation=flight-cldera
#SBATCH --job-name=hvawt-recovered-analysis
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
[[ -s "${CAMPAIGN_ROOT}/one_way_audit_recovery_receipt.json" ]] || fail "Phase-1 recovery receipt is unavailable."

export PYTHONPATH="${VORLAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CAMPAIGN_ROOT}/.matplotlib_recovered_v3}"
mkdir -p "${MPLCONFIGDIR}"
cd "${VORLAP_ROOT}"

# Reproduce Phase 1 from the unchanged terminal artifacts before publishing v3.
"${PYTHON_EXE}" run_headless/recover_hvawt_completed_one_way.py verify \
  --campaign-root "${CAMPAIGN_ROOT}"
"${PYTHON_EXE}" run_headless/recover_hvawt_completed_one_way.py analyze \
  --campaign-root "${CAMPAIGN_ROOT}"

# Validate the complete ordered matrix before any derived postprocessor writes.
"${PYTHON_EXE}" \
  paper_drafts/NAWEA_QBladeBridge/analysis/prepare_unified_hvawt_paper_results_v2.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --preflight-only

scenarios="raw_qblade one_way_spectral_vorlap complete_hybrid_v8"
for comparison in raw_vs_one_way raw_vs_complete one_way_vs_complete; do
  case "${comparison}" in
    raw_vs_one_way) baseline=raw_qblade; coupled=one_way_spectral_vorlap ;;
    raw_vs_complete) baseline=raw_qblade; coupled=complete_hybrid_v8 ;;
    one_way_vs_complete) baseline=one_way_spectral_vorlap; coupled=complete_hybrid_v8 ;;
    *) fail "Unknown comparison ${comparison}." ;;
  esac
  "${PYTHON_EXE}" run_headless/postprocess_parked_demo.py \
    --run-root "${CAMPAIGN_ROOT}" \
    --out-dir "${CAMPAIGN_ROOT}/postprocess_${comparison}" \
    --scenarios "${scenarios}" \
    --comparison-baseline "${baseline}" \
    --comparison-coupled "${coupled}" \
    --min-fatigue-samples 80000 \
    --min-fatigue-duration 600 \
    --min-rainflow-cycles 10
done

"${PYTHON_EXE}" \
  paper_drafts/NAWEA_QBladeBridge/analysis/prepare_unified_hvawt_paper_results_v2.py \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --output-dir "${CAMPAIGN_ROOT}/paper_export_recovered_v2"

echo "Completed recovered matched raw/one-way/complete analysis and paper export."
