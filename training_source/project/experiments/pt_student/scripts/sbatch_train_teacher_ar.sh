#!/usr/bin/env bash
#SBATCH --job-name=pt_teacher_ar
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=1-00:00:00
#SBATCH --output=experiments/pt_student/logs/slurm_%j.out
# PT-student OPTIONAL action-rate variant (PLAN.md 4.3): the teacher of sbatch_train_teacher.sh with one added reward
# term (task Mjlab-Leap-PT-Teacher-AR, weight -0.075 on the squared change of the squashed action); all else equal.
# From scratch, never resumed, training seed 1, 4096 environments, 3001 updates, zero-initialised actor output.
#   cd training_source/project && sbatch experiments/pt_student/scripts/sbatch_train_teacher_ar.sh
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
PY=${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python}
E=experiments/pt_student
NAME=pt_teacher_ar_seed1
LOG=$E/logs/train_teacher_ar_seed1.log
fail() { echo "[preflight FAIL] $*" | tee -a "$LOG"; exit 2; }
sha256sum --quiet -c experiments/hwlimit_ac/cache/hwlimit_grasp_cache.sha256 || fail "grasp cache changed"
(cd $E && sha256sum --quiet -c logs/plan.sha256) || fail "PLAN.md changed after pre-registration"
EXISTING=$(ls -d logs/rsl_rl/pt_student_teacher_ar/*_${NAME} 2>/dev/null | tail -1)
[[ -z "$EXISTING" ]] || fail "run exists: $EXISTING (never resumed)"
echo "$(date) start $NAME host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} job ${SLURM_JOB_ID:-none}" | tee -a "$LOG"
$PY scripts/train.py Mjlab-Leap-PT-Teacher-AR --env.scene.num-envs 4096 --agent.seed 1 \
  --agent.run-name "$NAME" --zero-actor-output True >> "$LOG" 2>&1
echo "$(date) exit $? $NAME" | tee -a "$LOG"
