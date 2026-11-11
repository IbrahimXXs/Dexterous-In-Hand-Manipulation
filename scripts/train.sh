#!/usr/bin/env bash
# Train / fine-tune the continuous-yaw policy (task Mjlab-Leap-Right-HandCube-MBZUAI-Rotate).
#   scripts/train.sh --help                                              # every overridable option
#   scripts/train.sh --checkpoint-file "$PWD/checkpoint/model_400.pt"    # fine-tune the shipped policy
#   scripts/train.sh --env.scene.num-envs 64 --agent.max-iterations 5    # quick CPU smoke test
# Reward terms: training_source/project/src/in_hand_rotation_mjlab/tasks/hand_cube/hand_cube_env_cfg.py
# Runs are written to training_source/project/logs/rsl_rl/leap_right_hand_cube_mbzuai_rotate/<timestamp>/
# Requires scripts/setup_training.sh first.
source "$(dirname "$0")/_common.sh"
cd "$TRAINING"
uv run python scripts/train.py Mjlab-Leap-Right-HandCube-MBZUAI-Rotate "$@"
