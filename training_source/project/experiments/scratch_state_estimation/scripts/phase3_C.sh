#!/usr/bin/env bash
# Phase 3: Experiment C (estimated state, frozen estimator_B) x seeds 1-3.
# Seeds 1,2 start immediately on GPUs 3,1; seed 3 starts on whichever finishes first.
# (GPU 2 is occupied by another user's vLLM server; GPUs 0,4-7 belong to other users.)
set -uo pipefail
cd "$(dirname "$0")"
export LEAP_SCRATCH_ESTIMATOR=$(cd ..; pwd)/models/estimator_B.pt
GPU=3 bash train_condition.sh C 1 & p1=$!
GPU=1 bash train_condition.sh C 2 & p2=$!
wait -n $p1 $p2
if kill -0 $p1 2>/dev/null; then g=1; else g=3; fi
GPU=$g bash train_condition.sh C 3
wait
echo PHASE3_DONE
