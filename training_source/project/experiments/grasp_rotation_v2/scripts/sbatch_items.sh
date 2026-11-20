#!/usr/bin/env bash
#SBATCH --job-name=gr2_items
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000
#SBATCH --time=12:00:00
#SBATCH --output=experiments/grasp_rotation_v2/logs/slurm_%j.out
# One rollout worker; several workers share an item list via claim files.
#   cd training_source/project && ITEMS=experiments/grasp_rotation_v2/items/p1_dev.json sbatch experiments/grasp_rotation_v2/scripts/sbatch_items.sh
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
echo "$(date) worker job ${SLURM_JOB_ID:-local} host $(hostname) items ${ITEMS}"
${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python} experiments/grasp_rotation_v2/scripts/run_items.py --items "$ITEMS"
echo "$(date) exit $?"
