#!/usr/bin/env bash
#SBATCH --job-name=c5hw_eval
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000
#SBATCH --time=12:00:00
#SBATCH --output=experiments/c5_hw_v1/logs/slurm_%j.out
# One evaluation worker; several workers share an item list via claim files.
#   cd training_source/project && ITEMS=experiments/c5_hw_v1/items/effort_v1_eval.json sbatch experiments/c5_hw_v1/scripts/sbatch_eval.sh
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
echo "$(date) worker job ${SLURM_JOB_ID:-local} host $(hostname) items ${ITEMS}"
${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python} experiments/c5_hw_v1/scripts/eval_items.py --items "$ITEMS"
echo "$(date) exit $?"
