#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${VORLAP_QBLADE_IMAGE:-vorlap-qblade-headless:amd64}"
PLATFORM="${VORLAP_QBLADE_PLATFORM:-linux/amd64}"

exec /opt/homebrew/bin/docker run --rm \
  --platform "${PLATFORM}" \
  --cpus 4 \
  --volume "${REPO_ROOT}:/work/VorLap" \
  --workdir /work/VorLap \
  --env QT_QPA_PLATFORM=offscreen \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  --env OMP_NUM_THREADS=4 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  "${IMAGE}" \
  bash -lc 'set -euo pipefail; export LD_LIBRARY_PATH="/work/VorLap/run_headless/QBladeCE_2.0.9.7:/work/VorLap/run_headless/QBladeCE_2.0.9.7/Libraries:/work/VorLap/run_headless/QBladeCE_2.0.9.7/ControllerFiles:${LD_LIBRARY_PATH:-}"; python3 -m examples.ehrmann2014_naca0018.work_map_smoke --root /tmp/airfoil-work-map-smoke --qblade-root /work/VorLap/run_headless/QBladeCE_2.0.9.7 --runtime-vorlap-root /work/VorLap --python /usr/bin/python3 --threads 4 --steps 100'
