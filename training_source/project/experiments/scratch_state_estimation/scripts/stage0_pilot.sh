#!/usr/bin/env bash
# Stage-0 pilot: from-scratch PPO, 200 iterations, 4096 envs, seed 1.
#   P1 original clipped actions + adaptive LR (as shipped)
#   P2 tanh-squashed actions   + adaptive LR
#   P3 tanh-squashed actions   + fixed LR 3e-4
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}; export MUJOCO_GL=egl PYTHONUNBUFFERED=1
L=experiments/scratch_state_estimation/logs
COMMON="--env.scene.num-envs 4096 --agent.max-iterations 200 --agent.seed 1 --zero-actor-output True"
CUDA_VISIBLE_DEVICES=${G1:-1} $PY scripts/train.py Mjlab-Leap-Scratch-A-ClipAction $COMMON --agent.run-name pilot_P1_clip_adaptive > $L/pilot_P1.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G2:-2} $PY scripts/train.py Mjlab-Leap-Scratch-A $COMMON --agent.run-name pilot_P2_tanh_adaptive > $L/pilot_P2.log 2>&1 &
CUDA_VISIBLE_DEVICES=${G3:-3} $PY scripts/train.py Mjlab-Leap-Scratch-A $COMMON --agent.algorithm.schedule fixed --agent.algorithm.learning-rate 3e-4 --agent.run-name pilot_P3_tanh_fixed3e-4 > $L/pilot_P3.log 2>&1 &
wait
echo PILOT_DONE
