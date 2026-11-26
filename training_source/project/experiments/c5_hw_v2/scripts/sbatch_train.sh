#!/usr/bin/env bash
#SBATCH --job-name=c5hw2_train
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=1-00:00:00
#SBATCH --output=experiments/c5_hw_v2/logs/slurm_%j.out
# The single C5-HW-V2 run (PPO seed 1), from scratch (never resumed, never initialised from C5 / C5-EFFORT),
# then its validation curve.  Same command as c5_hw_v1/scripts/sbatch_train.sh, task Mjlab-Leap-C5HW2.
#   cd training_source/project && SEED=1 sbatch experiments/c5_hw_v2/scripts/sbatch_train.sh
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
PY=${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python}
E=experiments/c5_hw_v2
SEED=${SEED:-1}
[[ "$SEED" == 1 ]] || { echo "only seed 1 is approved (single-seed exploratory run)"; exit 2; }
NAME=c5hw2_seed${SEED}
LOG=$E/logs/train_C5HW2_seed${SEED}.log
fail() { echo "[preflight FAIL] $*" | tee -a "$LOG"; exit 2; }
sha256sum --quiet -c experiments/hwlimit_ac/cache/hwlimit_grasp_cache.sha256 || fail "grasp cache changed"
sha256sum --quiet -c $E/logs/frozen.sha256 || fail "frozen config/cache/fit/protocol changed"
EXISTING=$(ls -d logs/rsl_rl/c5_hw_v2/*_${NAME} 2>/dev/null | grep -v INTERRUPTED | tail -1)
[[ -z "$EXISTING" ]] || fail "run exists: $EXISTING (never resumed)"
echo "$(date) start $NAME host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} job ${SLURM_JOB_ID:-none}" | tee -a "$LOG"
$PY scripts/train.py Mjlab-Leap-C5HW2 --env.scene.num-envs 4096 --agent.seed "$SEED" \
  --agent.run-name "$NAME" --zero-actor-output True >> "$LOG" 2>&1
echo "$(date) exit $? $NAME" | tee -a "$LOG"
$PY $E/scripts/curves.py eval --run C5HW2_${SEED} >> $E/logs/curves_C5HW2_${SEED}.log 2>&1
echo "$(date) curves exit $?" | tee -a "$LOG"
