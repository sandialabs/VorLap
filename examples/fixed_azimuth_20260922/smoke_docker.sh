#!/bin/bash
# Real QBlade import/integration verification with steady U11.25 inflow only.
# Production always uses the original matched TurbSim bytes; this smoke is not
# fatigue evidence and cannot be submitted as a production campaign.
set -euo pipefail
VORLAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOURCE_CAMPAIGN_REL="${SOURCE_CAMPAIGN_REL:-run_headless/runs/hvawt_complete_dlc_three_path_v3}"
SMOKE_ROOT_REL="${SMOKE_ROOT_REL:-run_headless/runs/fixed_azimuth_20260922_native_smoke}"
exec /opt/homebrew/bin/docker run --rm --platform linux/amd64 --cpus 4 \
  --volume "${VORLAP_ROOT}:/work/VorLap" --workdir /work/VorLap \
  --env QT_QPA_PLATFORM=offscreen --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env "SOURCE_CAMPAIGN_REL=${SOURCE_CAMPAIGN_REL}" --env "SMOKE_ROOT_REL=${SMOKE_ROOT_REL}" \
  --env "AZIMUTH_DEG=${AZIMUTH_DEG:-147}" \
  "${VORLAP_QBLADE_IMAGE:-vorlap-qblade-headless:amd64}" bash -lc '
    set -euo pipefail
    export VORLAP_ROOT=/work/VorLap
    export QBLADE_ROOT=/work/VorLap/run_headless/QBladeCE_2.0.9.7
    export PYTHONPATH="${VORLAP_ROOT}:${QBLADE_ROOT}/SIL_Interface"
    export LD_LIBRARY_PATH="${QBLADE_ROOT}:${QBLADE_ROOT}/Libraries:${QBLADE_ROOT}/ControllerFiles:${QBLADE_ROOT}/Binaries:${QBLADE_ROOT}/Binaries/IPCBridge"
    export VORLAP_QBLADE_LOADER_ENV=1
    python3 examples/fixed_azimuth_20260922/campaign.py prepare \
      --source-campaign "$SOURCE_CAMPAIGN_REL" --campaign-root "$SMOKE_ROOT_REL" \
      --azimuth-deg "$AZIMUTH_DEG" --smoke-uniform > /tmp/fixed_prepare.json
    for index in 0 1 2; do
      python3 examples/fixed_azimuth_20260922/campaign.py run \
        --campaign-root "$SMOKE_ROOT_REL" --qblade-root "$QBLADE_ROOT" \
        --case-index "$index" --threads 4 --smoke-steps 160
    done
  '
