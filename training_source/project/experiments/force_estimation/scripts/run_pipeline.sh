#!/usr/bin/env bash
#SBATCH --job-name=force_est
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80000
#SBATCH --time=08:00:00
#SBATCH --output=experiments/force_estimation/logs/slurm_%j.out
#
# Full fingertip-force pipeline: dataset -> training -> selection -> evaluation -> preservation check.
# Resumable: finished shards (meta.json) and finished runs (summary.json) are skipped.
# Uses only existing, read-only study checkpoints; starts no PPO training.
#
#   cd training_source/project && sbatch experiments/force_estimation/scripts/run_pipeline.sh
#   (or: CUDA_VISIBLE_DEVICES=<g> bash experiments/force_estimation/scripts/run_pipeline.sh)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
E=experiments/force_estimation
PY=${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export LEAP_SCRATCH_ESTIMATOR=experiments/scratch_state_estimation/models/estimator_B.pt  # read-only, C actors only
NPAR=${NPAR:-4}
mkdir -p $E/logs $E/data
echo "$(date) pipeline start host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?}" | tee -a $E/logs/pipeline.log

RA=logs/rsl_rl/scratch_A/2026-11-12_00-48-01_study_A
RC=logs/rsl_rl/scratch_C/2026-11-12_19-06-10_study_C
RD=logs/rsl_rl/scratch_D/2026-11-12_00-48-01_study_D
ck() { echo "${1}_seed${2}/model_${3}.pt"; }   # run-prefix seed iteration

JOBS=$E/data/collection_plan.txt
: > $JOBS.tmp
add() {  # split seed cond run-prefix policy-seed it envs noise preset
  echo "$1 $2 $3 $(ck $4 $5 $6) ${3}_seed${5}_it${6}$([[ $8 != 1.0 ]] && echo _std$8) $7 $8 $9" >> $JOBS.tmp
}
# train: training policies A2 A3 C2 D2, checkpoints 300/600/1000 (+1000 with std x2)
s=21000
for P in "A $RA 2" "A $RA 3" "C $RC 2" "D $RD 2"; do set -- $P
  for it in 300 600 1000; do add train $((s++)) $1 $2 $3 $it 192 1.0 nominal; done
  add train $((s++)) $1 $2 $3 1000 192 2.0 nominal
done
# val: same policies, new sim seeds
s=22000
for P in "A $RA 2" "A $RA 3" "C $RC 2" "D $RD 2"; do set -- $P
  for it in 600 1000; do add val $((s++)) $1 $2 $3 $it 64 1.0 nominal; done
done
# test: HELD-OUT policies A1 C3 D1 (never used for training/validation)
s=23000
for P in "A $RA 1" "C $RC 3" "D $RD 1"; do set -- $P
  for it in 300 1000; do add test $((s++)) $1 $2 $3 $it 160 1.0 nominal; done
  add test $((s++)) $1 $2 $3 1000 160 2.0 nominal
done
# test_seen: training policies, final checkpoints, new sim seeds
s=24000
for P in "A $RA 2" "C $RC 2" "D $RD 2"; do set -- $P; add test_seen $((s++)) $1 $2 $3 1000 160 1.0 nominal; done
# robustness: held-out policies under the study's robustness presets
s=25000
for pr in heavy_slippery light_grippy high_delay weak_motors obs_noise_x3; do
  for P in "A $RA 1" "C $RC 3" "D $RD 1"; do set -- $P; add robust_$pr $((s++)) $1 $2 $3 1000 64 1.0 $pr; done
done
mv $JOBS.tmp $JOBS
echo "$(wc -l < $JOBS) collection runs" | tee -a $E/logs/pipeline.log

collect_one() {
  read -r split seed cond ckpt name envs noise preset <<< "$1"
  $PY $E/scripts/collect_force_data.py --condition $cond --checkpoint $ckpt --policy-name $name --split $split \
    --seed $seed --preset $preset --num-envs $envs --num-steps 600 --noise-scale $noise --out $E/data \
    > $E/logs/collect_${split}_${seed}.log 2>&1
  echo "collect $split $seed $name exit $?"
}
export -f collect_one; export PY E
cat $JOBS | xargs -P $NPAR -I{} bash -c 'collect_one "{}"' | tee -a $E/logs/pipeline.log
echo "$(date) collection done" | tee -a $E/logs/pipeline.log

for cfg in "direct 0" "direct 1" "hurdle 0" "hurdle 1"; do set -- $cfg
  name=${1}_s${2}
  [[ -f $E/runs/$name/summary.json ]] && { echo "skip $name"; continue; }
  $PY $E/scripts/train_force_estimator.py --head $1 --seed $2 --run-name $name --steps 40000 \
    > $E/logs/train_$name.log 2>&1
  echo "train $name exit $?" | tee -a $E/logs/pipeline.log
done
$PY $E/scripts/select_force_model.py >> $E/logs/pipeline.log 2>&1
$PY $E/scripts/evaluate_force_estimator.py > $E/logs/evaluate.log 2>&1
echo "evaluate exit $?" | tee -a $E/logs/pipeline.log
$PY $E/scripts/inference_example.py > $E/logs/inference_example.log 2>&1
echo "inference example exit $?" | tee -a $E/logs/pipeline.log
bash $E/scripts/check_preservation.sh > $E/logs/preservation_check.log 2>&1
echo "preservation check exit $?" | tee -a $E/logs/pipeline.log
$PY -m pytest -q tests/test_force_estimation.py tests/test_scratch_study.py tests/test_estimation.py > $E/logs/pytest_final.log 2>&1
echo "pytest exit $?" | tee -a $E/logs/pipeline.log
echo "$(date) PIPELINE_DONE" | tee -a $E/logs/pipeline.log
