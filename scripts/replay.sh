#!/usr/bin/env bash
# Watch the exact stable 30 s continuous cube-yaw trajectory (model_400.pt, seed 11).
#   scripts/replay.sh                         # full 30 s
#   scripts/replay.sh --start-time 5.0        # skip the grasp-settling transient
#   scripts/replay.sh --start-time 5 --playback-speed 0.5 --loop
source "$(dirname "$0")/_common.sh"
cd "$ROOT"
uv run "$VIEWER_PY" simulation/replay_exact_mjlab_yaw.py "$@"
