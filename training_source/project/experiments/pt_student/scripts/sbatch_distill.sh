#!/usr/bin/env bash
#SBATCH --job-name=pt_distill
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000
#SBATCH --time=1-00:00:00
#SBATCH --output=experiments/pt_student/logs/slurm_%j.out
# One PT-student distillation (PLAN.md section 2): DAgger from the final teacher, training seed 1, 1500 updates.
#   cd training_source/project && STUDENT=S-TF sbatch experiments/pt_student/scripts/sbatch_distill.sh
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
PY=${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python}
E=experiments/pt_student
STUDENT=${STUDENT:?STUDENT=S-TF|S-TF-noaux|S-MLP}
(cd $E && sha256sum --quiet -c logs/plan.sha256) || { echo "PLAN.md changed"; exit 2; }
[[ ! -e $E/runs/$STUDENT/final.pt ]] || { echo "run exists: $E/runs/$STUDENT (never resumed)"; exit 2; }
TEACHER=$(ls logs/rsl_rl/pt_student_teacher/*_pt_teacher_seed1/model_3000.pt)
echo "$(date) start $STUDENT teacher $TEACHER host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} job ${SLURM_JOB_ID:-none}"
$PY $E/scripts/distill.py --student "$STUDENT" --teacher "$TEACHER" --updates 1500
echo "$(date) exit $?"
