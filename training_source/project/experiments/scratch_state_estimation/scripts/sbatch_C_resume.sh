#!/usr/bin/env bash
#SBATCH --job-name=leap_C_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
# Resume Experiment C seeds 1 and 3 after the gpu-01 node failure (08:47).
# train_condition.sh resumes from the latest checkpoint (C1: model_850, C3: model_50)
# up to the fixed budget (final checkpoint model_1000).
cd /home/ibrahim.alsarraj/Documents/Dexterous-In-Hand-Manipulation/training_source/project/experiments/scratch_state_estimation/scripts
export PY=/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python
export LEAP_SCRATCH_ESTIMATOR=$(cd ..; pwd)/models/estimator_B.pt
G=${CUDA_VISIBLE_DEVICES:-0}
echo "host $(hostname) CUDA_VISIBLE_DEVICES=$G"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
GPU=$G bash train_condition.sh C 1
GPU=$G bash train_condition.sh C 3
echo C_RESUME_DONE
