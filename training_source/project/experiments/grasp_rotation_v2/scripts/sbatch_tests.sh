#!/bin/bash
#SBATCH -J gr2_tests
#SBATCH -p ws-ia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH -o experiments/grasp_rotation_v2/logs/slurm_%j.out
# Full test suite (all studies + Grasp-Rotation-V2) on one GPU node.
cd /home/ibrahim.alsarraj/Documents/Dexterous-In-Hand-Manipulation/training_source/project
export MUJOCO_GL=egl
${PY:-/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python} -m pytest tests -q -p no:cacheprovider 2>&1 | tee experiments/grasp_rotation_v2/logs/pytest_full.log
