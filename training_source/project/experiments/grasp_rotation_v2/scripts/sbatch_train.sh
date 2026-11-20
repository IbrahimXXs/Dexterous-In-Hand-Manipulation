#!/usr/bin/env bash
#SBATCH --job-name=grv2_train
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=1-00:00:00
#SBATCH --output=experiments/grasp_rotation_v2/logs/slurm_%j.out
# One Grasp-Rotation-V2 candidate run, from scratch (never resumed), then its validation curve.
#   cd training_source/project && CAND=P0 SEED=1 sbatch experiments/grasp_rotation_v2/scripts/sbatch_train.sh
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
PY=${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python}
E=experiments/grasp_rotation_v2
CAND=${CAND:?CAND=P0|C5}
SEED=${SEED:-1}
NAME=grv2_${CAND}_seed${SEED}
LOG=$E/logs/train_${CAND}_seed${SEED}.log
fail() { echo "[preflight FAIL] $*" | tee -a "$LOG"; exit 2; }
sha256sum --quiet -c experiments/hwlimit_ac/cache/hwlimit_grasp_cache.sha256 || fail "grasp cache changed"
sha256sum --quiet -c $E/logs/protocol.sha256 || fail "PROTOCOL.md changed after pre-registration"
EXISTING=$(ls -d logs/rsl_rl/grasp_rotation_v2_${CAND}/*_${NAME} 2>/dev/null | grep -v INTERRUPTED | tail -1)
[[ -z "$EXISTING" ]] || fail "run exists: $EXISTING (never resumed)"
echo "$(date) start $NAME host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} job ${SLURM_JOB_ID:-none}" | tee -a "$LOG"
$PY scripts/train.py Mjlab-Leap-GRV2-${CAND} --env.scene.num-envs 4096 --agent.seed "$SEED" \
  --agent.run-name "$NAME" --zero-actor-output True >> "$LOG" 2>&1
echo "$(date) exit $? $NAME" | tee -a "$LOG"
$PY $E/scripts/curves.py eval --run ${CAND}_${SEED} >> $E/logs/curves_${CAND}_${SEED}.log 2>&1
echo "$(date) curves exit $?" | tee -a "$LOG"
