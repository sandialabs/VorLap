#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORLAP_ROOT="${VORLAP_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
IMAGE="${IMAGE:-vorlap-qblade-headless:amd64}"
PLATFORM="${PLATFORM:-linux/amd64}"
VERIFY_HOST_ROOT="${VERIFY_HOST_ROOT:-$(mktemp -d /private/tmp/vorlap-ehrmann-structure.XXXXXX)}"
CONTAINER_NAME="vorlap-ehrmann-structure-verify-$$"
mkdir -p "${VERIFY_HOST_ROOT}"
chmod 0777 "${VERIFY_HOST_ROOT}"
trap 'docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true' EXIT

run_status=0
docker run --name "${CONTAINER_NAME}" \
  --platform "${PLATFORM}" \
  --cpus 4 \
  --volume "${VORLAP_ROOT}:/work/VorLap" \
  --workdir /work/VorLap \
  --env QT_QPA_PLATFORM=offscreen \
  --env OMP_NUM_THREADS=4 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    export LD_LIBRARY_PATH="/work/VorLap/run_headless/QBladeCE_2.0.9.7:/work/VorLap/run_headless/QBladeCE_2.0.9.7/Libraries:/work/VorLap/run_headless/QBladeCE_2.0.9.7/ControllerFiles:${LD_LIBRARY_PATH:-}"
    python3 -m examples.ehrmann2014_naca0018.campaign generate \
      --output /tmp/verify/campaign \
      --runtime-vorlap-root /work/VorLap
    python3 -m examples.ehrmann2014_naca0018.structural_verification prepare \
      --campaign /tmp/verify/campaign \
      --output /tmp/verify/structural
    python3 -m examples.ehrmann2014_naca0018.structural_verification run \
      --root /tmp/verify/structural \
      --qblade-root /work/VorLap/run_headless/QBladeCE_2.0.9.7 \
      --threads 4
    python3 -m examples.ehrmann2014_naca0018.structural_verification analyze \
      --root /tmp/verify/structural
  ' || run_status=$?

docker cp "${CONTAINER_NAME}:/tmp/verify/structural" "${VERIFY_HOST_ROOT}/structural"
if [[ "${run_status}" -ne 0 ]]; then
  echo "ERROR: structural verification failed; retained artifacts: ${VERIFY_HOST_ROOT}/structural" >&2
  exit "${run_status}"
fi

echo "Ehrmann structural verification artifacts: ${VERIFY_HOST_ROOT}/structural"
