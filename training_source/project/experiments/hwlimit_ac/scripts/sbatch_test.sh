#!/usr/bin/env bash
#SBATCH --job-name=hwlim_test
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24000
#SBATCH --time=12:00:00
#SBATCH --output=experiments/hwlimit_ac/logs/slurm_%j.out
# One held-out-test worker (test_eval.py; workers share the item list via claim files).
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
echo "$(date) test worker job ${SLURM_JOB_ID:-local} host $(hostname)"
${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python} experiments/hwlimit_ac/scripts/test_eval.py
echo "$(date) exit $?"
