#!/usr/bin/env bash
# Run the trained policy LIVE in the full MJLab environment it was trained in (exact physics,
# actuator model, delays and observation history), in MuJoCo's native viewer.
#   scripts/play_mjlab.sh                                   # checkpoint/model_400.pt
#   scripts/play_mjlab.sh --checkpoint-file /abs/path/to/model_XXXX.pt   # your own run
# Requires scripts/setup_training.sh first. Uses CUDA if available, otherwise CPU.
source "$(dirname "$0")/_common.sh"
cd "$TRAINING"
PY_DIR="$TRAINING/.venv/bin"
[ -x "$PY_DIR/python" ] || { echo "❌ Run scripts/setup_training.sh first." >&2; exit 1; }
if [ "$(uname -s)" = "Darwin" ]; then PY="$PY_DIR/mjpython"; else PY="$PY_DIR/python"; fi
DEVICE="cpu"; "$PY_DIR/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && DEVICE="cuda:0"
ARGS=("$@"); printf '%s\n' "$@" | grep -q -- '--checkpoint-file' || ARGS=(--checkpoint-file "$ROOT/checkpoint/model_400.pt" ${ARGS[@]+"${ARGS[@]}"})
exec "$PY" scripts/play_policy.py Mjlab-Leap-Right-HandCube-MBZUAI-Rotate --device "$DEVICE" --viewer native "${ARGS[@]}"
