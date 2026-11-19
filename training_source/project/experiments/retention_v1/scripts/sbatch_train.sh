#!/usr/bin/env bash
#SBATCH --job-name=retv1_A1
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=1-00:00:00
#SBATCH --output=experiments/retention_v1/logs/slurm_%j.out
# Retention-V1 seed-1 run, from scratch (never resumed), then its validation curve.
#   cd training_source/project && sbatch experiments/retention_v1/scripts/sbatch_train.sh
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
PY=${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python}
E=experiments/retention_v1
SEED=${SEED:-1}
NAME=retv1_A_seed${SEED}
LOG=$E/logs/train_A_seed${SEED}.log
fail() { echo "[preflight FAIL] $*" | tee -a "$LOG"; exit 2; }
sha256sum --quiet -c experiments/hwlimit_ac/cache/hwlimit_grasp_cache.sha256 || fail "grasp cache changed"
EXISTING=$(ls -d logs/rsl_rl/retention_v1_A/*_${NAME} 2>/dev/null | grep -v INTERRUPTED | tail -1)
[[ -z "$EXISTING" ]] || fail "run exists: $EXISTING (never resumed)"
echo "$(date) start $NAME host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} job ${SLURM_JOB_ID:-none}" | tee -a "$LOG"
$PY scripts/train.py Mjlab-Leap-RetV1-A --env.scene.num-envs 4096 --agent.seed "$SEED" \
  --agent.run-name "$NAME" --zero-actor-output True >> "$LOG" 2>&1
echo "$(date) exit $? $NAME" | tee -a "$LOG"
$PY $E/scripts/curves.py eval --run R$SEED >> $E/logs/curves_R$SEED.log 2>&1
echo "$(date) curves exit $?" | tee -a "$LOG"
