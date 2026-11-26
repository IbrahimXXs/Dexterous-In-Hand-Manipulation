#!/usr/bin/env bash
#SBATCH --job-name=c5hwv2
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000
#SBATCH --time=12:00:00
#SBATCH --output=experiments/c5_hw_v2/logs/slurm_%j.out
# One C5-HW-V2 evaluation-only worker (no training): runs CMD (a python command line, relative to the project).
#   cd training_source/project && CMD="experiments/c5_hw_v2/scripts/response_fit.py record --effort default" sbatch experiments/c5_hw_v2/scripts/sbatch_job.sh
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
echo "$(date) job ${SLURM_JOB_ID:-local} host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?}: $CMD"
${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python} $CMD
echo "$(date) exit $?"
