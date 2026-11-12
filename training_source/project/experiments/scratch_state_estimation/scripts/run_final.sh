#!/usr/bin/env bash
# Held-out test evaluation of the validation-selected checkpoint of each run.
#   GPU=3 bash run_final.sh A1 D1 ...
cd "$(dirname "$0")/../../.."
E=experiments/scratch_state_estimation
PY=${PY:-python}; export MUJOCO_GL=egl PYTHONUNBUFFERED=1
for r in "$@"; do
  c=${r:0:1}; s=${r:1:1}
  ck=$($PY -c "import json;print(json.load(open('$E/results/curves/${c}_seed${s}.json'))['selected_checkpoint'])")
  echo "$(date) final ${c}_seed${s} <- $ck"
  CUDA_VISIBLE_DEVICES=${GPU:-1} $PY $E/scripts/study_eval.py final --policy ${c}_seed${s}=$c:$ck \
    --estimator $E/models/estimator_B.pt --num-envs ${NENV:-1024} --out $E/results/final >> $E/logs/final_${c}_seed${s}.log 2>&1
  echo "final $r exit $?"
done
