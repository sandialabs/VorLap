#!/bin/bash
# Produce all 48 exact-runtime-base prescribed-motion histories locally.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${VORLAP_QBLADE_IMAGE:-vorlap-qblade-headless:amd64}"
PLATFORM="${VORLAP_QBLADE_PLATFORM:-linux/amd64}"
CPU_LIMIT="${VORLAP_QBLADE_LOCAL_CPUS:-8}"
CAMPAIGN_RELATIVE="${VORLAP_QBLADE_WORK_MAP_CAMPAIGN_RELATIVE:-run_headless/runs/naca0018_work_map_qblade_v2}"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ "${CPU_LIMIT}" =~ ^[1-8]$ ]] || fail "VORLAP_QBLADE_LOCAL_CPUS must be an integer in [1, 8]."
[[ "${CAMPAIGN_RELATIVE}" != /* && "${CAMPAIGN_RELATIVE}" != *..* ]] || \
  fail "Campaign path must be a canonical repository-relative path."
command -v /opt/homebrew/bin/docker >/dev/null 2>&1 || fail "Docker CLI is unavailable."
image_id="$(/opt/homebrew/bin/docker image inspect "${IMAGE}" --format '{{.Id}}')"
[[ "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "Docker image ID is not a SHA-256 identity."

exec /opt/homebrew/bin/docker run --rm \
  --platform "${PLATFORM}" \
  --cpus "${CPU_LIMIT}" \
  --volume "${REPO_ROOT}:/work/VorLap" \
  --workdir /work/VorLap \
  --env QT_QPA_PLATFORM=offscreen \
  --env OMP_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 \
  --env "VORLAP_CONTAINER_IMAGE_ID=${image_id}" \
  --env "VORLAP_LOCAL_CPU_LIMIT=${CPU_LIMIT}" \
  --env "VORLAP_CAMPAIGN_RELATIVE=${CAMPAIGN_RELATIVE}" \
  "${IMAGE}" \
  bash -lc '
    set -euo pipefail
    campaign="/work/VorLap/${VORLAP_CAMPAIGN_RELATIVE}"
    qblade_root=/work/VorLap/run_headless/QBladeCE_2.0.9.7
    export LD_LIBRARY_PATH="${qblade_root}:${qblade_root}/Libraries:${qblade_root}/ControllerFiles:${LD_LIBRARY_PATH:-}"
    if [[ ! -f "${campaign}/campaign_manifest.json" ]]; then
      [[ ! -e "${campaign}" ]] || {
        echo "ERROR: campaign exists without a manifest: ${campaign}" >&2
        exit 2
      }
      python3 -m examples.ehrmann2014_naca0018.qblade_prescribed_motion generate \
        --campaign "${campaign}" --qblade-root "${qblade_root}"
    fi
    python3 -m examples.ehrmann2014_naca0018.qblade_prescribed_motion validate \
      --campaign "${campaign}" --qblade-root "${qblade_root}"
    python3 -m run_headless.run_naca0018_work_map_qblade_local \
      --stage preflight --campaign "${campaign}" --qblade-root "${qblade_root}" \
      --python /usr/bin/python3 --workers 1 --cpu-limit "${VORLAP_LOCAL_CPU_LIMIT}" \
      --container-image-id "${VORLAP_CONTAINER_IMAGE_ID}"
    python3 -m run_headless.run_naca0018_work_map_qblade_local \
      --stage matrix --campaign "${campaign}" --qblade-root "${qblade_root}" \
      --python /usr/bin/python3 --workers "${VORLAP_LOCAL_CPU_LIMIT}" \
      --cpu-limit "${VORLAP_LOCAL_CPU_LIMIT}" \
      --container-image-id "${VORLAP_CONTAINER_IMAGE_ID}"
    python3 -m examples.ehrmann2014_naca0018.qblade_prescribed_motion status \
      --campaign "${campaign}"
    python3 -m examples.ehrmann2014_naca0018.qblade_prescribed_motion finalize \
      --campaign "${campaign}"
  '
