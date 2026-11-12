#!/usr/bin/env bash
# Phase 1: Experiments A (baseline) and D (oracle) x seeds 1-3, 2 runs per GPU.
set -uo pipefail
cd "$(dirname "$0")"
for s in 1 2 3; do
  g=$(( s ))   # GPUs 1, 2, 3
  GPU=$g bash train_condition.sh A $s &
  GPU=$g bash train_condition.sh D $s &
done
wait
echo PHASE1_DONE
