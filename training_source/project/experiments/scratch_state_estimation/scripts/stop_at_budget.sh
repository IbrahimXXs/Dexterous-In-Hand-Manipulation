#!/usr/bin/env bash
# Stop phase-1 runs (started with a 1500 cap) right after model_1000.pt.
cd "$(dirname "$0")/../../.."
left="A1 A2 A3 D1 D2 D3"
while [[ -n "$left" ]]; do
  new=""
  for r in $left; do
    c=${r:0:1}; s=${r:1:1}
    d=$(ls -d logs/rsl_rl/scratch_${c}/*_study_${c}_seed${s} 2>/dev/null | tail -1)
    if [[ -n "$d" && -f "$d/model_1000.pt" ]]; then
      sleep 5  # let the save finish
      pkill -f -- "--agent.run-name study_${c}_seed${s}" && echo "$(date) stopped study_${c}_seed${s} at model_1000.pt"
    else new="$new $r"; fi
  done
  left=$(echo $new)
  [[ -n "$left" ]] && sleep 20
done
echo BUDGET_STOP_DONE
