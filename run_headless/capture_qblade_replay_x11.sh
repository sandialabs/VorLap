#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  printf 'Usage: %s OUTPUT_DIR\n' "$(basename "$0")" >&2
}

if (( $# != 1 )); then
  usage
  exit 2
fi

output_dir="$1"
display_name="${QBLADE_GUI_DISPLAY:-:99}"
screen_size="${QBLADE_CAPTURE_SCREEN_SIZE:-2400x1600}"
viewport_crop="${QBLADE_CAPTURE_VIEWPORT_CROP:-1854:1400:546:177}"
capture_seconds="${QBLADE_CAPTURE_SECONDS:-40}"
frame_rate="${QBLADE_CAPTURE_FRAME_RATE:-30}"

for executable in ffmpeg xdotool; do
  command -v "${executable}" >/dev/null 2>&1 || {
    printf 'ERROR: required executable is unavailable: %s\n' "${executable}" >&2
    exit 2
  }
done
mkdir -p "${output_dir}"
export DISPLAY="${display_name}"

window_id=""
for _ in $(seq 1 120); do
  window_id="$(xdotool search --name 'QBlade' 2>/dev/null | tail -n 1 || true)"
  if [[ -n "${window_id}" ]]; then
    break
  fi
  sleep 0.25
done
if [[ -z "${window_id}" ]]; then
  printf 'ERROR: no QBlade GUI window appeared on %s.\n' "${display_name}" >&2
  exit 2
fi

xdotool windowfocus --sync "${window_id}"
sleep 1

# Physical X11 coordinates are for QBlade CE 2.0.9.7 at 2400 x 1600. Enter
# Turbine Simulation, preserve the native turbine/blade surfaces and vortex
# filaments, hide the text overlay and wake particles, and color filaments by
# circulation. The stored structural deformation is never magnified.
xdotool mousemove 642 65 click 1
sleep 8
xdotool mousemove 400 668 click 1
xdotool mousemove 144 887 click 1
xdotool mousemove 401 944 click 1

# Focus the replay slider and use Home so the capture starts at stored step 0
# without relying on a click near the slider's narrow left endpoint.
xdotool mousemove 1500 157 click 1
xdotool key Home
sleep 3

ffmpeg -y \
  -f x11grab \
  -draw_mouse 0 \
  -framerate "${frame_rate}" \
  -video_size "${screen_size}" \
  -i "${display_name}.0+0,0" \
  -t "${capture_seconds}" \
  -vf "crop=${viewport_crop}" \
  -c:v libx264 \
  -preset medium \
  -crf 17 \
  -pix_fmt yuv420p \
  "${output_dir}/qblade_replay_raw.mp4" \
  >"${output_dir}/ffmpeg_capture.log" 2>&1 &
ffmpeg_pid=$!

sleep 1
xdotool mousemove 930 157 click 1
wait "${ffmpeg_pid}"

ffmpeg -y \
  -f x11grab \
  -draw_mouse 0 \
  -video_size "${screen_size}" \
  -i "${display_name}.0+0,0" \
  -frames:v 1 \
  -vf "crop=${viewport_crop}" \
  "${output_dir}/qblade_replay_final.png" \
  >"${output_dir}/ffmpeg_still.log" 2>&1
