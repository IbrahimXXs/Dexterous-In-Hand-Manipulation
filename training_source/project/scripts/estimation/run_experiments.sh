#!/usr/bin/env bash
# End-to-end reproduction of the estimator study (Tasks 2-6).
# Run from training_source/project with the leap-estimator env active.
#
#   GPUS="1 2 3" bash scripts/estimation/run_experiments.sh
#
# Steps can be skipped by exporting SKIP_<STEP>=1 (COLLECT, ESTIMATOR, PPO, EVAL).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
read -r -a G <<< "${GPUS:-0 1 2}"
ROOT_CKPT="$(cd ../.. && pwd)/checkpoint/model_400.pt"
ITERS=${ITERS:-300}
NUM_ENVS=${NUM_ENVS:-4096}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1

# Task 2 - dataset (train 6 seeds x 2048 envs, val 1, test 2 seeds x 1024 envs, 30 s each)
if [[ -z "${SKIP_COLLECT:-}" ]]; then
  GPUS="${G[*]}" PY="$PY" bash scripts/estimation/collect_all.sh
fi

# Task 3/4 - estimator training + independent evaluation
if [[ -z "${SKIP_ESTIMATOR:-}" ]]; then
  CUDA_VISIBLE_DEVICES=${G[0]} "$PY" scripts/estimation/train_estimator.py --run-name mlp_baseline
  BEST=$(ls -td data/runs/estimator/*_mlp_baseline | head -1)/best.pt
  mkdir -p data/estimator_models && cp "$BEST" data/estimator_models/cube_state_estimator.pt
  CUDA_VISIBLE_DEVICES=${G[0]} "$PY" scripts/estimation/evaluate_estimator.py \
    --estimator data/estimator_models/cube_state_estimator.pt --output-dir data/eval/estimator
fi

# Task 5/6 - checkpoint migration and PPO fine-tuning (A' control, C full state, D contacts)
if [[ -z "${SKIP_PPO:-}" ]]; then
  for m in full contact; do
    "$PY" scripts/estimation/migrate_checkpoint.py --estimator data/estimator_models/cube_state_estimator.pt \
      --feature-mode $m --output data/estimator_models/model_400_estimator_$m.pt
  done
  mkdir -p data/runs/ppo
  CUDA_VISIBLE_DEVICES=${G[0]} "$PY" scripts/train.py Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-FixedReset \
    --checkpoint-file "$ROOT_CKPT" --env.scene.num-envs $NUM_ENVS --agent.max-iterations $ITERS --agent.seed 7 \
    > data/runs/ppo/A_prime_control.log 2>&1 &
  CUDA_VISIBLE_DEVICES=${G[1]:-${G[0]}} "$PY" scripts/train.py Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-Estimator \
    --checkpoint-file "$PWD/data/estimator_models/model_400_estimator_full.pt" --env.scene.num-envs $NUM_ENVS \
    --agent.max-iterations $ITERS --agent.seed 7 > data/runs/ppo/C_estimator_full.log 2>&1 &
  CUDA_VISIBLE_DEVICES=${G[2]:-${G[0]}} "$PY" scripts/train.py Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-EstContact \
    --checkpoint-file "$PWD/data/estimator_models/model_400_estimator_contact.pt" --env.scene.num-envs $NUM_ENVS \
    --agent.max-iterations $ITERS --agent.seed 7 > data/runs/ppo/D_estimator_contact.log 2>&1 &
  wait
fi

# Task 6 - evaluation under identical held-out conditions
if [[ -z "${SKIP_EVAL:-}" ]]; then
  last() { ls -td logs/rsl_rl/$1/*/ | head -1; }
  final() { ls "$(last "$1")"model_*.pt | sort -V | tail -1; }
  A2=$(final leap_right_mbzuai_rotate_fixedreset)
  C=$(final leap_right_mbzuai_rotate_estimator_full)
  D=$(final leap_right_mbzuai_rotate_estimator_contact)
  PRESETS="nominal heavy_slippery light_grippy high_delay weak_motors"
  EST=data/estimator_models/cube_state_estimator.pt
  run_eval() {  # gpu name ckpt
    CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/estimation/evaluate_policies.py --policy "$2=$3" \
      --seeds 5001 5002 --presets $PRESETS --estimator $EST --output-dir data/eval/policies/$2 \
      > data/eval/policies_$2.log 2>&1
  }
  mkdir -p data/eval/policies
  run_eval ${G[0]} A_model400 "$ROOT_CKPT" &
  run_eval ${G[1]:-${G[0]}} Aprime_control "$A2" &
  run_eval ${G[2]:-${G[0]}} C_estimator "$C" &
  wait
  run_eval ${G[0]} D_contact "$D" &
  CUDA_VISIBLE_DEVICES=${G[1]:-${G[0]}} "$PY" scripts/estimation/evaluate_policies.py \
    --policy A_model400="$ROOT_CKPT" --seeds 5001 5002 --presets nominal --env-variant original \
    --output-dir data/eval/policies/A_model400_original_env > data/eval/policies_A_original.log 2>&1 &
  wait
  "$PY" scripts/estimation/plot_results.py \
    --eval data/eval/policies/{A_model400,Aprime_control,C_estimator,D_contact}/results.json \
    --runs A_prime="$(last leap_right_mbzuai_rotate_fixedreset)" C_estimator="$(last leap_right_mbzuai_rotate_estimator_full)" \
           D_contact="$(last leap_right_mbzuai_rotate_estimator_contact)" \
    --output-dir data/eval/report
  "$PY" scripts/estimation/export_estimator_policy.py --checkpoint "$C" --output data/estimator_models/estimator_policy_full.pt
fi
echo "all done"
