#!/usr/bin/env bash
# Collect the train / val / test splits with model_400.pt.
# Splits are separated by seed (independent episodes, env instances and
# startup randomisation).  Runs two seeds concurrently, one per GPU.
#
#   GPUS="1 2" NUM_ENVS=2048 NUM_STEPS=600 scripts/estimation/collect_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
OUT=${OUT:-data/estimator}
NUM_ENVS=${NUM_ENVS:-2048}
NUM_STEPS=${NUM_STEPS:-600}
read -r -a GPUS <<< "${GPUS:-0}"
TRAIN_SEEDS=${TRAIN_SEEDS:-"1001 1002 1003 1004 1005 1006"}
VAL_SEEDS=${VAL_SEEDS:-"2001"}
TEST_SEEDS=${TEST_SEEDS:-"3001 3002"}

jobs_list=()
for s in $TRAIN_SEEDS; do jobs_list+=("train $s $NUM_ENVS"); done
for s in $VAL_SEEDS;   do jobs_list+=("val $s $((NUM_ENVS / 2))"); done
for s in $TEST_SEEDS;  do jobs_list+=("test $s $((NUM_ENVS / 2))"); done

mkdir -p "$OUT/logs"
run_queue() {  # $1 = gpu, remaining = job indices
  local gpu=$1; shift
  for idx in "$@"; do
    read -r split seed n <<< "${jobs_list[$idx]}"
    [[ -f "$OUT/$split/seed_$seed/meta.json" ]] && { echo "skip $split/$seed"; continue; }
    echo "[gpu $gpu] $split seed $seed ($n envs)"
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl "$PY" scripts/estimation/collect_estimator_data.py \
      --split "$split" --seed "$seed" --num-envs "$n" --num-steps "$NUM_STEPS" \
      --output-root "$OUT" --overwrite > "$OUT/logs/${split}_${seed}.log" 2>&1
    grep '\[collect\] wrote' "$OUT/logs/${split}_${seed}.log"
  done
}
pids=()
for g in "${!GPUS[@]}"; do
  mine=()
  for i in "${!jobs_list[@]}"; do (( i % ${#GPUS[@]} == g )) && mine+=("$i"); done
  run_queue "${GPUS[$g]}" "${mine[@]}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "collection finished"
