#!/usr/bin/env bash
# Validation learning curves + checkpoint selection for the given runs (sequential, one GPU).
#   GPU=1 bash run_curves.sh A1 A2 D3 ...   (condition letter + seed)
cd "$(dirname "$0")/../../.."
E=experiments/scratch_state_estimation
PY=${PY:-python}; export MUJOCO_GL=egl PYTHONUNBUFFERED=1
for r in "$@"; do
  c=${r:0:1}; s=${r:1:1}
  d=$(ls -d logs/rsl_rl/scratch_${c}/*_study_${c}_seed${s} | tail -1)
  CUDA_VISIBLE_DEVICES=${GPU:-1} $PY $E/scripts/study_eval.py curve --run "$d" --cond $c --name ${c}_seed${s} \
    --iterations ${ITERS:-100 200 300 400 500 600 700 800 900 1000} --include-last --num-envs 1024 --out $E/results/curves \
    >> $E/logs/curves_${c}_seed${s}.log 2>&1
  echo "curve $r exit $?"
done
