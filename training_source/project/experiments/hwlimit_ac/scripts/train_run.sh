#!/usr/bin/env bash
# Train one hardware-limit study run from scratch on the visible GPU.  Never resumes.
#
#   bash experiments/hwlimit_ac/scripts/train_run.sh A 1
#   bash experiments/hwlimit_ac/scripts/train_run.sh C 2
#
# Identical for A and C: task Mjlab-Leap-HwLim-<cond>, 4096 envs, HW_ITERATIONS
# PPO updates (rl_cfg.py), zero-initialised actor output, --agent.seed <seed>
# (also the env seed).  Refuses to start if a run with the same name exists,
# if estimator B or the grasp cache differ from their recorded sha256, or if the
# cache is missing.  An interrupted run is never resumed: its directory is
# renamed to *_INTERRUPTED and the run is retrained from scratch.
set -uo pipefail
COND=$1; SEED=$2
cd "$(dirname "$0")/../../.."
E=experiments/hwlimit_ac
PY=${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python}
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
NAME="hwlim_${COND}_seed${SEED}"
LOG=$E/logs/train_${COND}_seed${SEED}.log
fail() { echo "[preflight FAIL] $*" | tee -a "$LOG"; exit 2; }
[[ $COND == A || $COND == C ]] || fail "condition must be A or C"
sha256sum --quiet -c $E/preservation/estimator_B.sha256 || fail "estimator_B.pt changed"
sha256sum --quiet -c $E/cache/hwlimit_grasp_cache.sha256 || fail "grasp cache changed"
EXISTING=$(ls -d logs/rsl_rl/hwlim_${COND}/*_${NAME} 2>/dev/null | grep -v INTERRUPTED | tail -1)
[[ -z "$EXISTING" ]] || fail "run exists: $EXISTING (never resumed; rename it *_INTERRUPTED to retrain)"
echo "$(date) start $NAME host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} job ${SLURM_JOB_ID:-none}" | tee -a "$LOG"
$PY scripts/train.py Mjlab-Leap-HwLim-$COND --env.scene.num-envs 4096 --agent.seed "$SEED" \
  --agent.run-name "$NAME" --zero-actor-output True >> "$LOG" 2>&1
rc=$?
echo "$(date) exit $rc $NAME" | tee -a "$LOG"
exit $rc
