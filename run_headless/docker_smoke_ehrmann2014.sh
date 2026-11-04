#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORLAP_ROOT="${VORLAP_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
IMAGE="${IMAGE:-vorlap-qblade-headless:amd64}"
PLATFORM="${PLATFORM:-linux/amd64}"
SMOKE_HOST_ROOT="$(mktemp -d /private/tmp/vorlap-ehrmann-smoke.XXXXXX)"
chmod 0777 "${SMOKE_HOST_ROOT}"
trap 'rm -rf "${SMOKE_HOST_ROOT}"' EXIT

docker run --rm \
  --platform "${PLATFORM}" \
  --cpus 4 \
  --volume "${VORLAP_ROOT}:/work/VorLap" \
  --volume "${SMOKE_HOST_ROOT}:/work/smoke" \
  --workdir /work/VorLap \
  --env QT_QPA_PLATFORM=offscreen \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env OMP_NUM_THREADS=4 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    export LD_LIBRARY_PATH="/work/VorLap/run_headless/QBladeCE_2.0.9.7:/work/VorLap/run_headless/QBladeCE_2.0.9.7/Libraries:/work/VorLap/run_headless/QBladeCE_2.0.9.7/ControllerFiles:${LD_LIBRARY_PATH:-}"
    python3 -m examples.ehrmann2014_naca0018.campaign generate \
      --output /work/smoke/campaign \
      --runtime-vorlap-root /work/VorLap
    python3 -m examples.ehrmann2014_naca0018.campaign validate \
      --campaign /work/smoke/campaign
    python3 -m examples.ehrmann2014_naca0018.campaign run \
      --campaign /work/smoke/campaign \
      --qblade-root /work/VorLap/run_headless/QBladeCE_2.0.9.7 \
      --case-index 1 \
      --threads 4 \
      --python /usr/bin/python3 \
      --qblade-steps 2
  '
