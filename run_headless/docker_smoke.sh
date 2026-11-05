#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VORLAP_ROOT="${VORLAP_ROOT:-${REPO_ROOT}}"
IMAGE="${IMAGE:-vorlap-qblade-headless:amd64}"
PLATFORM="${PLATFORM:-linux/amd64}"
RUN_ROOT="${RUN_ROOT:-runs/docker_parked_demo}"
WIND_SPEEDS="${WIND_SPEEDS:-5}"
QBLADE_STEPS="${QBLADE_STEPS:-2}"
QBLADE_CHECKPOINT_EVERY="${QBLADE_CHECKPOINT_EVERY:-0}"
QBLADE_DURATION="${QBLADE_DURATION:-60}"
QBLADE_TIMESTEP="${QBLADE_TIMESTEP:-0.0075}"
VORLAP_SOURCE_FREQUENCY_TRUNCATION="${VORLAP_SOURCE_FREQUENCY_TRUNCATION:-none}"
CAMPAIGN_SCENARIOS="${CAMPAIGN_SCENARIOS:-baseline vorlap}"
QBLADE_WAKE_PROFILE="${QBLADE_WAKE_PROFILE:-reference}"
QBLADE_RESULT_FILTER="${QBLADE_RESULT_FILTER:-${REPO_ROOT}/run_headless/config/parked_fatigue_results.filter}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
HOST_CA_BUNDLE="${HOST_CA_BUNDLE:-}"
DOCKER_CA_BUNDLE="${SCRIPT_DIR}/docker_host_ca.crt"

if [[ -z "${QBLADE_ROOT:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/QBladeCE_2.0.9.7" ]]; then
    QBLADE_ROOT="${SCRIPT_DIR}/QBladeCE_2.0.9.7"
  elif [[ -d "${REPO_ROOT}/QBladeCE_2.0.9.7" ]]; then
    QBLADE_ROOT="${REPO_ROOT}/QBladeCE_2.0.9.7"
  else
    QBLADE_ROOT="${SCRIPT_DIR}/QBladeCE_2.0.9.7"
  fi
fi
QBLADE_ROOT="$(cd "${QBLADE_ROOT}" && pwd)"
QBLADE_RESULT_FILTER="$(cd "$(dirname "${QBLADE_RESULT_FILTER}")" && pwd)/$(basename "${QBLADE_RESULT_FILTER}")"

if [[ "${QBLADE_RESULT_FILTER}" == "${REPO_ROOT}"* ]]; then
  CONTAINER_QBLADE_RESULT_FILTER="/work/VorLap${QBLADE_RESULT_FILTER#${REPO_ROOT}}"
else
  echo "ERROR: QBLADE_RESULT_FILTER must be inside VORLAP_ROOT for docker_smoke.sh: ${QBLADE_RESULT_FILTER}" >&2
  exit 2
fi

if [[ "${QBLADE_ROOT}" == "${REPO_ROOT}"* ]]; then
  CONTAINER_QBLADE_ROOT="/work/VorLap${QBLADE_ROOT#${REPO_ROOT}}"
  qblade_mounts=()
else
  CONTAINER_QBLADE_ROOT="/work/QBladeCE_2.0.9.7"
  qblade_mounts=(--volume "${QBLADE_ROOT}:${CONTAINER_QBLADE_ROOT}")
fi

if [[ -z "${HOST_CA_BUNDLE}" ]]; then
  for candidate in \
    /opt/homebrew/etc/ca-certificates/cert.pem \
    /usr/local/etc/ca-certificates/cert.pem \
    /etc/ssl/cert.pem \
    /etc/ssl/certs/ca-certificates.crt; do
    if [[ -f "${candidate}" ]]; then
      HOST_CA_BUNDLE="${candidate}"
      break
    fi
  done
fi

if [[ -n "${HOST_CA_BUNDLE}" && -f "${HOST_CA_BUNDLE}" ]]; then
  cp "${HOST_CA_BUNDLE}" "${DOCKER_CA_BUNDLE}"
else
  : > "${DOCKER_CA_BUNDLE}"
fi

if docker buildx version >/dev/null 2>&1; then
  docker buildx build \
    --platform "${PLATFORM}" \
    --load \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE}" \
    "${SCRIPT_DIR}"
else
  docker build \
    --platform "${PLATFORM}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE}" \
    "${SCRIPT_DIR}"
fi

docker run --rm \
  --platform "${PLATFORM}" \
  --user "$(id -u):$(id -g)" \
  --volume "${VORLAP_ROOT}:/work/VorLap" \
  ${qblade_mounts[@]+"${qblade_mounts[@]}"} \
  --workdir /work/VorLap \
  --env QBLADE_ROOT="${CONTAINER_QBLADE_ROOT}" \
  --env VORLAP_ROOT=/work/VorLap \
  --env QBLADE_RESULT_FILTER="${CONTAINER_QBLADE_RESULT_FILTER}" \
  --env PYTHON_EXE=/usr/bin/python3 \
  --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  --env QBLADE_CHECKPOINT_EVERY="${QBLADE_CHECKPOINT_EVERY}" \
  --env CAMPAIGN_SCENARIOS="${CAMPAIGN_SCENARIOS}" \
  --env QBLADE_WAKE_PROFILE="${QBLADE_WAKE_PROFILE}" \
  --env QT_QPA_PLATFORM=offscreen \
  --env MPLCONFIGDIR=/tmp/matplotlib \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    python3 -m py_compile run_headless/*.py
    export LD_LIBRARY_PATH=\"\${QBLADE_ROOT}:\${QBLADE_ROOT}/Libraries:\${QBLADE_ROOT}/ControllerFiles:\${LD_LIBRARY_PATH:-}\"
    ldd \"\$(ls \${QBLADE_ROOT}/libQBladeCE_*.so.1.0.0 | head -1)\"
    run_headless/run_parked_demo_campaign.py \
      --run-root '${RUN_ROOT}' \
      --qblade-root \"\${QBLADE_ROOT}\" \
      --vorlap-root /work/VorLap \
      --python /usr/bin/python3 \
      --wind-speeds '${WIND_SPEEDS}' \
      --duration '${QBLADE_DURATION}' \
      --timestep '${QBLADE_TIMESTEP}' \
      --source-frequency-truncation '${VORLAP_SOURCE_FREQUENCY_TRUNCATION}' \
      --qblade-steps '${QBLADE_STEPS}' \
      --threads '${OMP_NUM_THREADS}' \
      --case-workers 1 \
      --skip-vorlap-install \
      --force-bridge-build
  "
