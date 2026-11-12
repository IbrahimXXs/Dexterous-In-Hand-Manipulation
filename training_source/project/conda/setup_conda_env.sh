#!/usr/bin/env bash
# Create the "leap-estimator" conda environment: the full MJLab training
# stack (from the project's uv.lock), CUDA-12.8 PyTorch, the estimator-study
# extras, the portable viewer / hardware-bridge dependencies, and the Warp
# compatibility hook.  Linux + NVIDIA driver >= 570 (CUDA 12.8).
#
#   bash training_source/project/conda/setup_conda_env.sh [env-name]
set -euo pipefail
ENV_NAME=${1:-leap-estimator}
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(dirname "$HERE")"

conda create -y -n "$ENV_NAME" python=3.10 pip
PREFIX=$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')
PY="$PREFIX/bin/python"
"$PY" -m pip install -q uv

# 1. Exact locked training stack (MJLab v1.1.1, mujoco 3.8.1, mujoco-warp
#    3.8.1, warp-lang 1.13.0, rsl-rl-lib 4.0.1, numpy 1.26.4, ...) and the
#    project itself in editable mode.
(cd "$PROJECT" && UV_PROJECT_ENVIRONMENT="$PREFIX" "$PREFIX/bin/uv" sync --frozen --inexact)

# 2. The lock resolves torch 2.12.0 (+cu130), which needs a CUDA 13 driver.
#    Replace it with the newest CUDA 12.8 build (driver 570.x).
"$PREFIX/bin/uv" pip install --python "$PY" \
  "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

# 3. Study extras + portable viewer / hardware bridge dependencies.
"$PREFIX/bin/uv" pip install --python "$PY" \
  tensorboard matplotlib pytest "pyzmq>=25,<28" "dynamixel-sdk>=3.7,<4"

# 4. MJLab v1.1.1 reads warp.context.*, which Warp >= 1.12 no longer binds;
#    without this every CUDA run fails (see leap_warp_compat.py).
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
cp "$HERE/leap_warp_compat.py" "$SITE/"
echo "import leap_warp_compat" > "$SITE/leap_warp_compat.pth"

"$PY" - <<'EOF'
import torch, warp, mjlab, rsl_rl, mujoco
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "mujoco", mujoco.__version__)
import warp as wp
assert hasattr(wp, "context"), "warp.context hook not active"
EOF
echo "Done.  conda activate $ENV_NAME"
