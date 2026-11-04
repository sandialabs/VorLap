#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf 'Usage: %s PROJECT.qpr OUTPUT_DIR\n' "$(basename "$0")" >&2
}

if (( $# != 2 )); then
  usage
  exit 2
fi

project="$1"
output_dir="$2"
qblade_root="${QBLADE_ROOT:?QBLADE_ROOT must name the QBlade CE directory}"
qblade_bin="${qblade_root}/QBladeCE_2.0.9.7"
display_name="${QBLADE_GUI_DISPLAY:-:99}"
screen_geometry="${QBLADE_GUI_SCREEN:-2400x1600x24}"
runtime_dir="${XDG_RUNTIME_DIR:-/tmp/qblade-runtime-${UID}}"

[[ -x "${qblade_bin}" ]] || {
  printf 'ERROR: QBlade executable is unavailable: %s\n' "${qblade_bin}" >&2
  exit 2
}
[[ -s "${project}" ]] || {
  printf 'ERROR: QBlade project is unavailable or empty: %s\n' "${project}" >&2
  exit 2
}

mkdir -p "${output_dir}" "${runtime_dir}"
chmod 700 "${runtime_dir}"

export DISPLAY="${display_name}"
export XDG_RUNTIME_DIR="${runtime_dir}"
export QT_QPA_PLATFORM=xcb
export LIBGL_ALWAYS_SOFTWARE=1

Xvfb "${display_name}" \
  -screen 0 "${screen_geometry}" \
  +extension GLX \
  +render \
  -noreset \
  >"${output_dir}/xvfb.log" 2>&1 &
xvfb_pid=$!

cleanup() {
  kill "${xvfb_pid}" 2>/dev/null || true
  wait "${xvfb_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

display_ready=0
for _ in $(seq 1 50); do
  if DISPLAY="${display_name}" xdotool getmouselocation --shell >/dev/null 2>&1; then
    display_ready=1
    break
  fi
  sleep 0.1
done
if (( display_ready != 1 )); then
  printf 'ERROR: Xvfb did not become ready on %s.\n' "${display_name}" >&2
  exit 2
fi

"${qblade_bin}" "${project}" >"${output_dir}/qblade_gui.log" 2>&1 &
qblade_pid=$!
wait "${qblade_pid}"
