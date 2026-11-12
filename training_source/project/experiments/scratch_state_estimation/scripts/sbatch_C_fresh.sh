#!/usr/bin/env bash
#SBATCH --job-name=leap_C
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclude=gpu-02
# Experiment C, one seed, trained FROM SCRATCH WITHOUT INTERRUPTION (never resumed:
# RSL-RL resume does not restore the adaptive learning rate, curriculum EMA or RNG),
# followed by its validation curve, held-out test and feature ablation.
#   sbatch --export=ALL,SEED=1 sbatch_C_fresh.sh
# Uses the fixed EstimatedStateMLPModel (estimator outside the module tree).
set -uo pipefail
P=/home/ibrahim.alsarraj/Documents/Dexterous-In-Hand-Manipulation/training_source/project
E=$P/experiments/scratch_state_estimation
cd $P
export PY=/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python MUJOCO_GL=egl PYTHONUNBUFFERED=1
export LEAP_SCRATCH_ESTIMATOR=$E/models/estimator_B.pt
NAME=study_C_seed${SEED}; LOG=$E/logs/train_C_seed${SEED}.log
G=${CUDA_VISIBLE_DEVICES:-0}
echo "$(date) job ${SLURM_JOB_ID:-?} host $(hostname) gpu $G" | tee -a $E/logs/phase3_C.log
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
# wait (up to 3 h) until the allocated GPU has >= 8 GB free, so training is not OOM-killed midway
for i in $(seq 1 36); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $G | head -1)
  [[ $free -ge 8000 ]] && break; echo "gpu $G free ${free} MiB < 8000, waiting"; sleep 300
done
if [[ ! -f $E/results/curves/C_seed${SEED}.json ]]; then
  # never resume: archive any partial run of this seed (e.g. after a node failure / requeue)
  for d in $(ls -d logs/rsl_rl/scratch_C/*_${NAME} 2>/dev/null); do
    [[ -f $d/model_1000.pt ]] && { echo "complete run exists: $d"; continue; }
    mkdir -p logs/rsl_rl/scratch_C_INTERRUPTED; mv "$d" logs/rsl_rl/scratch_C_INTERRUPTED/; echo "archived partial $d"
  done
  if ! ls logs/rsl_rl/scratch_C/*_${NAME}/model_1000.pt >/dev/null 2>&1; then
    [[ -f $LOG ]] && mv $LOG $LOG.prev_$(date +%s)
    CUDA_VISIBLE_DEVICES=$G $PY scripts/train.py Mjlab-Leap-Scratch-C --env.scene.num-envs 4096 --agent.seed $SEED \
      --agent.run-name $NAME --zero-actor-output True >> $LOG 2>&1
    echo "[train exit $?] $NAME" | tee -a $LOG
  fi
fi
ls logs/rsl_rl/scratch_C/*_${NAME}/model_1000.pt || { echo "no model_1000 for $NAME"; exit 1; }
GPU=$G bash $E/scripts/run_curves.sh C${SEED}
GPU=$G bash $E/scripts/run_final.sh C${SEED}
ck=$($PY -c "import json;print(json.load(open('$E/results/curves/C_seed${SEED}.json'))['selected_checkpoint'])")
CUDA_VISIBLE_DEVICES=$G $PY $E/scripts/study_eval.py ablation --policy C_seed${SEED}=C:$ck --num-envs 1024 --out $E/results/ablation >> $E/logs/ablation_C_seed${SEED}.log 2>&1
echo "ablation C${SEED} exit $?"
# extra work assigned to this job (e.g. remaining D ablations)
for r in ${EXTRA_ABL:-}; do
  c=${r:0:1}; s=${r:1:1}
  ck=$($PY -c "import json;print(json.load(open('$E/results/curves/${c}_seed${s}.json'))['selected_checkpoint'])")
  CUDA_VISIBLE_DEVICES=$G $PY $E/scripts/study_eval.py ablation --policy ${c}_seed${s}=$c:$ck --num-envs 1024 --out $E/results/ablation >> $E/logs/ablation_${c}_seed${s}.log 2>&1
  echo "ablation $r exit $?"
done
echo "$(date) JOB_DONE C seed $SEED" | tee -a $E/logs/phase3_C.log
