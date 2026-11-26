#!/usr/bin/env bash
# Make the README hero GIF from a real-hand trial recording.
#
# The trial videos are written on the lab machine by `./leap.sh trial` with LEAP_USE_VISION=1
# (outputs/cube_vision/<trial folder>/rgb.mp4 = raw RealSense colour, 1280x720) and are not in git.
# Good candidate: 25 Nov trial 1 (C5 seed 3, full 60 s, 16 quarter turns, no drop):
#   outputs/cube_vision/live_20261125_111603/rgb.mp4
#
# Usage (from anywhere):
#   scripts/make_hero_gif.sh INPUT.mp4 [START_S] [DURATION_S] [SPEED] [CROP]
#     START_S     seconds into INPUT where the clip starts            (default 15)
#     DURATION_S  seconds of real time in the clip                     (default 24)
#     SPEED       playback speed-up; label it in the README caption    (default 3, so 24 s -> 8 s)
#     CROP        ffmpeg crop w:h:x:y in input pixels                  (default 880:720:400:0,
#                 the tracker's hand ROI from configs/hand.env)
#   OUT=path.gif overrides the output (default docs/figures/readme/real_hand_c5.gif);
#   WIDTH=px overrides the width (default 440).
#
# The defaults give an 8 s, 10 fps GIF of about 6 MB. Keep it under ~8 MB so the README loads
# quickly; lower DURATION_S or WIDTH if needed.
set -euo pipefail

in=${1:?"usage: $0 INPUT.mp4 [START_S] [DURATION_S] [SPEED] [CROP]"}
start=${2:-15}
dur=${3:-24}
speed=${4:-3}
crop=${5:-880:720:400:0}
root=$(cd "$(dirname "$0")/.." && pwd)
out=${OUT:-$root/docs/figures/readme/real_hand_c5.gif}
width=${WIDTH:-440}

mkdir -p "$(dirname "$out")"
ffmpeg -v error -y -ss "$start" -t "$dur" -i "$in" \
  -vf "crop=${crop},setpts=PTS/${speed},fps=10,scale=${width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
  -loop 0 "$out"
echo "wrote $out ($(du -h "$out" | cut -f1))"
