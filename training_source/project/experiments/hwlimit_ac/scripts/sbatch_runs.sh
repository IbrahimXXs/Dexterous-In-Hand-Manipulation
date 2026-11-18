#!/usr/bin/env bash
#SBATCH --job-name=hwlim_ac
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24000
#SBATCH --time=1-00:00:00
#SBATCH --output=experiments/hwlimit_ac/logs/slurm_%j.out
#
# Runs the given study runs concurrently on this job's single GPU (co-location
# changes wall-clock only, not the training; every run is identical otherwise).
#
#   cd training_source/project && sbatch experiments/hwlimit_ac/scripts/sbatch_runs.sh A1 C1 A2
set -uo pipefail
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
echo "$(date) job ${SLURM_JOB_ID:-local} host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} runs: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for r in "$@"; do
  bash experiments/hwlimit_ac/scripts/train_run.sh "${r:0:1}" "${r:1}" &
  sleep 60   # stagger start-up (compilation, memory)
done
wait
echo "$(date) training finished: $*"
# validation curves (PROTOCOL.md Section 2), sequentially on this GPU after training
for r in "$@"; do
  ${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python} experiments/hwlimit_ac/scripts/curves.py eval --run "$r" \
    >> experiments/hwlimit_ac/logs/curves_$r.log 2>&1
  echo "$(date) curves $r exit $?"
done
echo "$(date) job done: $*"
