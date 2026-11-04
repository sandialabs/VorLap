#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUNNER_NAME="run_qblade_vorlap_sil_10s_sample_linux_cpu.sh"
export BRIDGE_RUN_LABEL="${BRIDGE_RUN_LABEL:-sil}"
export EXECUTION_BACKEND="${EXECUTION_BACKEND:-cpu}"
export COUPLED_SCENARIO="${COUPLED_SCENARIO:-sil_vorlap}"
export COUPLED_COUPLING="${COUPLED_COUPLING:-sil-vorlap}"
export COUPLED_RECORD_HISTORY="${COUPLED_RECORD_HISTORY:-1}"
export NATIVE_RUN_FULL_SIMULATION="${NATIVE_RUN_FULL_SIMULATION:-0}"
export SKIP_BRIDGE_BUILD="${SKIP_BRIDGE_BUILD:-1}"

exec "${SCRIPT_DIR}/run_qblade_vorlap_10s_sample_linux_common.sh" "$@"
