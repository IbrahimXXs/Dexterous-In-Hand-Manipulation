#!/usr/bin/env bash
# Install the heavy MJLab training stack (Python 3.10 + MJLab + MuJoCo-Warp + PyTorch)
# into training_source/project/.venv. Needed for scripts/train.sh.
# Real training wants Linux + NVIDIA/CUDA; on macOS this runs on CPU only (smoke tests).
# NOTE: pyproject.toml/uv.lock were patched (stable mujoco 3.8.1 + mujoco-warp 3.8.1 + warp-lang 1.13.0)
# because the course's pinned MuJoCo nightly was deleted upstream; originals kept as *.course-original.
source "$(dirname "$0")/_common.sh"
banner "MJLab training environment setup  ->  $TRAINING"
[ "$(uname -s)" = "Darwin" ] && echo "ℹ️  macOS: CPU only. Training is far faster on a Linux/NVIDIA machine."
cd "$TRAINING"
uv sync
echo "✅ Done. Next: scripts/train.sh --help"
