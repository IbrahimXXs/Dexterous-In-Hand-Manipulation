#!/usr/bin/env bash
# Stage 2 (Experiment B): dataset from Experiment-A checkpoints, new estimator.
#
# Data sources (policy x simulation seed; splits never share a sim seed, and
# the test split uses a policy that never produced training data):
#   train: A seed2 + A seed3 (A1, A2), checkpoints {100, 300, 600, final=1000}, 1024 envs,
#          half the envs sample from the policy Gaussian (std x1); plus the
#          final checkpoints with std x2 (more slips / drift / drops)
#   val:   A seed2 + A seed3, checkpoints {mid, final}, 256 envs
#   test:  A seed1 (A3; held-out policy), {early, mid, late, final}, 512 envs,
#          plus final with std x2
# Each run: 600 control steps (30 s), training env (noise, delays, DR).
#
#   MODE=trainval GPUS="2 3" A1=<run dir> A2=<run dir> bash stage2_estimator.sh   # collect + train
#   MODE=test     GPUS="1"   A3=<run dir>                 bash stage2_estimator.sh   # test data only
# (A1, A2 = training-data policies; A3 = held-out test policy.)
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}; export MUJOCO_GL=egl PYTHONUNBUFFERED=1
E=experiments/scratch_state_estimation
ROOT=$E/data/estimator
read -r -a G <<< "${GPUS:-1 2 3}"
ITS=(${ITS:-100 300 600 FINAL})
ck() {  # run_dir iteration|FINAL -> checkpoint path
  if [[ $2 == FINAL ]]; then ls "$1"/model_*.pt | sort -V | tail -1; else echo "$1/model_$2.pt"; fi
}
MODE=${MODE:-all}
jobs_file=$(mktemp)
if [[ $MODE != test ]]; then
s=11000
for r in "$A1" "$A2"; do for it in "${ITS[@]}"; do
  echo "train $((s++)) 1024 1.0 $(ck $r $it)" >> $jobs_file; done
  echo "train $((s++)) 512 2.0 $(ck $r FINAL)" >> $jobs_file
done
s=12000
for r in "$A1" "$A2"; do for it in 300 FINAL; do
  echo "val $((s++)) 256 1.0 $(ck $r $it)" >> $jobs_file; done; done
fi
if [[ $MODE != trainval ]]; then
s=13000
for it in "${ITS[@]}"; do echo "test $((s++)) 512 1.0 $(ck $A3 $it)" >> $jobs_file; done
echo "test $((s++)) 512 2.0 $(ck $A3 FINAL)" >> $jobs_file
fi
mkdir -p $E/data; cat $jobs_file >> $E/data/collection_plan.txt

i=0
while read -r split seed nenv scale ckpt; do
  gpu=${G[$((i % ${#G[@]}))]}; i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$gpu $PY scripts/estimation/collect_estimator_data.py --task Mjlab-Leap-Scratch-A \
      --checkpoint "$ckpt" --split $split --seed $seed --num-envs $nenv --num-steps 600 --noise-scale $scale \
      --output-root $ROOT > $E/logs/collect_${split}_${seed}.log 2>&1; echo "collect $split $seed exit $?" ) &
  if (( i % ${#G[@]} == 0 )); then wait; fi
done < $jobs_file
wait

[[ $MODE == test ]] && exit 0
# Train the estimator (velocity weight raised to 1.0: velocity targets are
# RMS-normalised, so all continuous targets get equal weight).
CUDA_VISIBLE_DEVICES=${G[0]} $PY scripts/estimation/train_estimator.py --data-root $ROOT \
  --log-root $E/runs/estimator --run-name estimator_B --w-vel 1.0 --skip-test > $E/logs/train_estimator.log 2>&1
BEST=$(ls -td $E/runs/estimator/*_estimator_B | head -1)/best.pt
cp "$BEST" $E/models/estimator_B.pt
echo "estimator -> $E/models/estimator_B.pt"
