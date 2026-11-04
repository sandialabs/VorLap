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
frame_count="${QBLADE_SAMPLE_FRAME_COUNT:-60}"
stored_step_count="${QBLADE_STORED_STEP_COUNT:-534}"
render_wait="${QBLADE_SAMPLE_RENDER_WAIT:-0.35}"
key_delay_ms="${QBLADE_SAMPLE_KEY_DELAY_MS:-1}"
configure_visuals="${QBLADE_SAMPLE_CONFIGURE_VISUALS:-true}"
hide_text_only="${QBLADE_SAMPLE_HIDE_TEXT_ONLY:-false}"

if (( frame_count < 2 || stored_step_count < 2 )); then
  printf 'ERROR: frame and stored-step counts must both be at least two.\n' >&2
  exit 2
fi
for executable in ffmpeg xdotool; do
  command -v "${executable}" >/dev/null 2>&1 || {
    printf 'ERROR: required executable is unavailable: %s\n' "${executable}" >&2
    exit 2
  }
done

mkdir -p "${output_dir}"
if compgen -G "${output_dir}/frame_*.png" >/dev/null; then
  printf 'ERROR: sampled frame files already exist under %s.\n' "${output_dir}" >&2
  exit 2
fi
export DISPLAY="${display_name}"

window_id="$(xdotool search --name 'QBlade CE v2.0.9.7b_beta unix' 2>/dev/null | tail -n 1 || true)"
if [[ -z "${window_id}" ]]; then
  printf 'ERROR: no QBlade main window appeared on %s.\n' "${display_name}" >&2
  exit 2
fi
xdotool windowfocus --sync "${window_id}"

# Physical X11 coordinates for QBlade CE 2.0.9.7 at 2400 x 1600.
xdotool mousemove 642 65 click 1
sleep 8
if [[ "${configure_visuals}" == "true" ]]; then
  xdotool mousemove 400 668 click 1
  xdotool mousemove 144 887 click 1
  xdotool mousemove 401 944 click 1
  xdotool mousemove 144 1276 click 1
  xdotool mousemove 144 639 click 1
  sleep 5
  xdotool mousemove 1500 800 click 4
  sleep 3
elif [[ "${hide_text_only}" == "true" ]]; then
  # Cosmetic-only export option: remove the GUI text overlay while retaining
  # every native wake-visualization toggle, color, filament, and particle.
  xdotool mousemove 400 668 click 1
  sleep 2
fi

# QBlade's slider ignores absolute track clicks and advances by a page step.
# Home plus exact Right-key increments therefore provides deterministic state
# selection without interpolating or magnifying the stored replay.
xdotool mousemove 1500 157 click 1
xdotool key Home
sleep 2

map_path="${output_dir}/sampled_frame_map.tsv"
printf 'frame_index\tstored_step_index\n' >"${map_path}"
current_step=0
last_step=$((stored_step_count - 1))
for (( frame_index = 0; frame_index < frame_count; frame_index++ )); do
  target_step=$(( (frame_index * last_step + (frame_count - 1) / 2) / (frame_count - 1) ))
  increment=$((target_step - current_step))
  if (( increment > 0 )); then
    xdotool key --repeat "${increment}" --delay "${key_delay_ms}" Right
  fi
  current_step="${target_step}"
  sleep "${render_wait}"
  printf -v frame_name 'frame_%04d.png' "${frame_index}"
  ffmpeg -loglevel error -y \
    -f x11grab \
    -draw_mouse 0 \
    -video_size "${screen_size}" \
    -i "${display_name}.0+0,0" \
    -frames:v 1 \
    -vf "crop=${viewport_crop}" \
    "${output_dir}/${frame_name}"
  printf '%d\t%d\n' "${frame_index}" "${target_step}" >>"${map_path}"
done
