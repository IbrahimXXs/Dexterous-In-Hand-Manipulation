#!/usr/bin/env bash
# Wait for A/D seeds 2 and 3 to reach the budget and exit, then run Stage-2 train/val collection + estimator training.
cd "$(dirname "$0")/../../.."
E=experiments/scratch_state_estimation
d() { ls -d logs/rsl_rl/scratch_$1/*_study_$1_seed$2 | tail -1; }
until [[ -f $(d A 2)/model_1000.pt && -f $(d A 3)/model_1000.pt && -f $(d D 2)/model_1000.pt && -f $(d D 3)/model_1000.pt ]]; do sleep 60; done
while pgrep -f -- "--agent.run-name study_[AD]_seed[23]" >/dev/null; do sleep 10; done
echo "$(date) A/D seeds 2,3 done; starting stage 2 trainval"
MODE=trainval GPUS="2 3" A1=$(d A 2) A2=$(d A 3) PY=${PY} bash $E/scripts/stage2_estimator.sh
echo "$(date) STAGE2_TRAINVAL_DONE"
