#!/usr/bin/env bash
# Train one study condition for one seed on one GPU (resumable).
#
#   GPU=1 bash train_condition.sh A 1
#   GPU=1 bash train_condition.sh D 2
#   GPU=2 LEAP_SCRATCH_ESTIMATOR=.../estimator_B.pt bash train_condition.sh C 3
#
# Identical for all conditions: 4096 envs, STUDY_ITERATIONS PPO iterations
# (rl_cfg.py), zero-initialised actor output, matched seed.  If a run with
# the same name already has checkpoints, training resumes from the latest one.
set -uo pipefail
COND=$1; SEED=$2
cd "$(dirname "$0")/../../.."
PY=${PY:-python}; export MUJOCO_GL=egl PYTHONUNBUFFERED=1
NAME="study_${COND}_seed${SEED}"
LOG=experiments/scratch_state_estimation/logs/train_${COND}_seed${SEED}.log
EXISTING=$(ls -d logs/rsl_rl/scratch_${COND}/*_${NAME} 2>/dev/null | tail -1)
ARGS=(Mjlab-Leap-Scratch-$COND --env.scene.num-envs 4096 --agent.seed "$SEED" --agent.run-name "$NAME")
if [[ -n "$EXISTING" ]] && ls "$EXISTING"/model_*.pt >/dev/null 2>&1; then
  LAST=$(ls "$EXISTING"/model_*.pt | sort -V | tail -1)
  DONE=$(basename "$LAST" .pt | cut -d_ -f2)
  echo "[resume] $LAST" | tee -a "$LOG"
  TOTAL=$($PY -c "from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_scratch_study.rl_cfg import STUDY_ITERATIONS as N; print(N)")
  ARGS+=(--checkpoint-file "$LAST" --agent.max-iterations $((TOTAL - DONE)))
else
  ARGS+=(--zero-actor-output True)
fi
CUDA_VISIBLE_DEVICES=${GPU:-1} $PY scripts/train.py "${ARGS[@]}" >> "$LOG" 2>&1
echo "[exit $?] $NAME" | tee -a "$LOG"
