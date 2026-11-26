#!/usr/bin/env bash
# C5-HW-V2: verify that nothing that existed before this milestone changed (pattern of c5_hw_v1/scripts/check_preservation.sh).
# Baseline commit 8bdcbee.  Checks: (1) no tracked file outside this milestone's new paths differs from the
# baseline (completed experiments incl. C5-HW-EFFORT-V1, GRV2 env/reward code, deploy kit source, guards, limits,
# hand.env, raw hardware logs, trial CSVs, audit outputs, checkpoint manifests, frozen protocols); (2) sha256 of
# all RSL-RL checkpoints that existed before this milestone; (3) checkpoint manifests of A/R1/C5/P0/C5-EFFORT
# re-verified file by file; (4) kit logs, audit outputs and kit policies hashed before this milestone (nothing
# was copied into policies/); (5) grasp cache.
set -uo pipefail
cd "$(dirname "$0")/../../../../.."   # repository root
T=training_source/project
P=$T/experiments/c5_hw_v2/preservation
B=$(cat $P/baseline_commit.txt)
fail=0
NEW=(":!$T/experiments/c5_hw_v2" ":!$T/src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_c5_hw_v2"
     ":!$T/tests/test_c5_hw_v2.py")
echo "== tracked files outside this milestone vs baseline commit $B"
if git merge-base --is-ancestor "$B" HEAD && git diff --quiet "$B" -- . "${NEW[@]}"; then
  echo "OK: no tracked file outside this milestone's paths differs from $B"
else
  echo "FAIL: files outside this milestone changed"; git diff --stat "$B" -- . "${NEW[@]}" | tail -8; fail=1
fi
echo "== untracked files outside this milestone's paths (new files elsewhere)"
U=$(git status --porcelain -uall | grep '^??' | awk '{print $2}' | grep -vE "^$T/(experiments/c5_hw_v2/|src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_c5_hw_v2/|tests/test_c5_hw_v2.py)" | grep -v __pycache__)
if [[ -z "$U" ]]; then echo "OK: none"; else echo "NOTE: $U"; fi
echo "== RSL-RL checkpoints that existed before this milestone"
if (cd $T && sha256sum --quiet -c experiments/c5_hw_v2/preservation/rsl_rl_pt_before.sha256); then
  echo "OK: $(wc -l < $P/rsl_rl_pt_before.sha256) checkpoints unchanged"; else echo "FAIL: checkpoint changed"; fail=1; fi
echo "== checkpoint manifests (A/hwlimit_ac, R1/retention_v1, C5+P0/grasp_rotation_v2, C5-EFFORT/c5_hw_v1)"
for M in hwlimit_ac retention_v1 grasp_rotation_v2 c5_hw_v1; do
  D=$T/experiments/$M/checkpoints
  bad=0; n=0
  while IFS='|' read -r f h _; do
    f=$(echo "$f" | xargs); h=$(echo "$h" | xargs)
    [[ "$f" == \#* || -z "$h" || ! "$h" =~ ^[0-9a-f]{64}$ ]] && continue
    n=$((n+1)); [[ "$(sha256sum "$D/$f" | cut -d' ' -f1)" == "$h" ]] || { echo "FAIL: $D/$f"; bad=1; }
  done < $D/MANIFEST.txt
  [[ $bad == 0 ]] && echo "OK: $M manifest, $n files match" || fail=1
done
echo "== deploy-kit logs, audit outputs and kit policies (hashed before this milestone)"
if sha256sum --quiet -c $P/kit_logs_audit_policies_before.sha256; then echo "OK: $(wc -l < $P/kit_logs_audit_policies_before.sha256) files unchanged"
else echo "FAIL: kit log / audit output / kit policy changed"; fail=1; fi
N_NOW=$(find LEAP_Hand_RL_Deploy_Kit/logs LEAP_Hand_RL_Deploy_Kit/audit LEAP_Hand_RL_Deploy_Kit/policies -type f | wc -l)
echo "   (files now under kit logs/audit/policies: $N_NOW; new lab session files are not this milestone's)"
echo "== hand.env"
echo "   sha256 $(sha256sum LEAP_Hand_RL_Deploy_Kit/configs/hand.env | cut -c1-16)... (baseline 68cc0663e9cb200f...)"
git diff --quiet "$B" -- LEAP_Hand_RL_Deploy_Kit/configs/hand.env && echo "OK: hand.env unchanged" || { echo "FAIL: hand.env changed"; fail=1; }
echo "== grasp cache"
if (cd $T && sha256sum --quiet -c experiments/hwlimit_ac/cache/hwlimit_grasp_cache.sha256); then
  echo "OK: grasp cache unchanged"; else echo "FAIL: grasp cache changed"; fail=1; fi
exit $fail
