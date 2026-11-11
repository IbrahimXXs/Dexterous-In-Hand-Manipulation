#!/usr/bin/env bash
# Run the exported actor (checkpoint/policy_weights.npz) LIVE in native MuJoCo on CPU.
# Lighter approximation of the MJLab task: can look more jittery than the replay.
#   scripts/live_policy.sh --duration-s 30
source "$(dirname "$0")/_common.sh"
cd "$ROOT"
uv run "$VIEWER_PY" simulation/run_yaw_rotation.py "$@"
