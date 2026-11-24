#!/usr/bin/env bash
# C5-HW-EFFORT-V1: verify that nothing that existed before this milestone changed.  Baseline commit
# f8e47b8 (audit + spec).  Checks: (1) no tracked file outside this milestone's new paths differs
# from the baseline (completed experiments, GRV2 env/reward code, deploy kit source and limits,
# hand.env, raw hardware logs, audit outputs, checkpoint manifests); (2) sha256 of all 974 RSL-RL
# checkpoints that existed before training (untracked logs/rsl_rl); (3) checkpoint manifests of
# A/R1/C5/P0 re-verified file by file; (4) kit logs + audit outputs hashed before this milestone;
# (5) grasp cache.
set -uo pipefail
cd "$(dirname "$0")/../../../../.."   # repository root
T=training_source/project
P=$T/experiments/c5_hw_v1/preservation
B=$(cat $P/baseline_commit.txt)
fail=0
echo "== tracked files outside this milestone vs baseline commit $B"
if git merge-base --is-ancestor "$B" HEAD && git diff --quiet "$B" -- . \
    ":!$T/experiments/c5_hw_v1" ":!$T/src/in_hand_rotation_mjlab/c5_hw_v1" \
    ":!$T/src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_c5_hw_v1" \
    ":!$T/tests/test_c5_hw_v1.py"; then
  echo "OK: no tracked file outside this milestone's paths differs from $B"
else
  echo "FAIL: files outside this milestone changed"; git diff --stat "$B" -- . ":!$T/experiments/c5_hw_v1" | tail -8; fail=1
fi
echo "== untracked files outside this milestone's paths (new files elsewhere)"
U=$(git status --porcelain -uall | grep '^??' | awk '{print $2}' | grep -vE "^$T/(experiments/c5_hw_v1/|src/in_hand_rotation_mjlab/c5_hw_v1/|src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_c5_hw_v1/|tests/test_c5_hw_v1.py)" | grep -v __pycache__)
if [[ -z "$U" ]]; then echo "OK: none"; else echo "NOTE: $U"; fi
echo "== RSL-RL checkpoints that existed before this milestone"
if (cd $T && sha256sum --quiet -c experiments/c5_hw_v1/preservation/rsl_rl_pt_before.sha256); then
  echo "OK: $(wc -l < $P/rsl_rl_pt_before.sha256) checkpoints unchanged"; else echo "FAIL: checkpoint changed"; fail=1; fi
echo "== checkpoint manifests (A/hwlimit_ac, R1/retention_v1, C5+P0/grasp_rotation_v2)"
for M in hwlimit_ac retention_v1 grasp_rotation_v2; do
  D=$T/experiments/$M/checkpoints
  bad=0; n=0
  while IFS='|' read -r f h _; do
    f=$(echo "$f" | xargs); h=$(echo "$h" | xargs)
    [[ "$f" == \#* || -z "$h" || ! "$h" =~ ^[0-9a-f]{64}$ ]] && continue
    n=$((n+1)); [[ "$(sha256sum "$D/$f" | cut -d' ' -f1)" == "$h" ]] || { echo "FAIL: $D/$f"; bad=1; }
  done < $D/MANIFEST.txt
  [[ $bad == 0 ]] && echo "OK: $M manifest, $n files match" || fail=1
done
echo "== deploy-kit logs and audit outputs (hashed before this milestone)"
if sha256sum --quiet -c $P/kit_logs_audit_before.sha256; then echo "OK: $(wc -l < $P/kit_logs_audit_before.sha256) files unchanged"
else echo "FAIL: kit log / audit output changed"; fail=1; fi
echo "== grasp cache"
if (cd $T && sha256sum --quiet -c experiments/hwlimit_ac/cache/hwlimit_grasp_cache.sha256); then
  echo "OK: grasp cache unchanged"; else echo "FAIL: grasp cache changed"; fail=1; fi
exit $fail
