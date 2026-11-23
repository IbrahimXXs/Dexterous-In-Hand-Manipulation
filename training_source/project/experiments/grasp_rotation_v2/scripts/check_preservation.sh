#!/usr/bin/env bash
# Grasp-Rotation-V2: verify that nothing that existed before this study changed.  Baseline commit
# a2c01e2 (Retention-V1).  Checks: (1) no file outside this study's folders differs from the
# baseline commit (covers all completed experiments, frozen checkpoints, the grasp cache, deploy kit,
# hardware logs); (2) every RSL-RL checkpoint of the earlier studies (Retention-V1's baseline list +
# the Retention-V1 runs; the R1 selected/final hashes equal retention_v1/checkpoints/MANIFEST.txt);
# (3) the grasp cache hash; (4) Retention-V1's checks of estimator B, kit symlinks and datasets.
set -uo pipefail
cd "$(dirname "$0")/../../../../.."   # repository root
P=training_source/project/experiments/grasp_rotation_v2/preservation
B=$(cat $P/baseline_commit.txt)
fail=0
echo "== files outside this study vs baseline commit $B"
if git merge-base --is-ancestor "$B" HEAD && git diff --quiet "$B" -- . \
    ':!training_source/project/experiments/grasp_rotation_v2' \
    ':!training_source/project/src/in_hand_rotation_mjlab/grasp_rotation_v2' \
    ':!training_source/project/src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_grasp_rotation_v2' \
    ':!training_source/project/tests/test_grasp_rotation_v2.py' ':!.gitignore'; then
  echo "OK: no file outside this study's folders differs from $B (.gitignore checked below)"
else
  echo "FAIL: files outside this study changed"; git diff --stat "$B" | tail -5; fail=1
fi
echo "== .gitignore: only appended lines allowed (this study's data/ and claim folders)"
if git diff -U0 "$B" -- .gitignore | grep -q '^-[^-]'; then echo "FAIL: .gitignore lines removed"; fail=1
else echo "OK: .gitignore has only added lines: $(git diff -U0 "$B" -- .gitignore | grep -c '^+[^+]')"; fi
echo "== RSL-RL checkpoints of the earlier studies (sha256)"
R=training_source/project/experiments/retention_v1/preservation/rsl_rl_pt.sha256
if sha256sum --quiet -c $R && sha256sum --quiet -c $P/rsl_rl_pt_retention_v1.sha256; then
  echo "OK: $(( $(wc -l < $R) + $(wc -l < $P/rsl_rl_pt_retention_v1.sha256) )) checkpoints unchanged"
else
  echo "FAIL: checkpoint changed"; fail=1
fi
echo "== grasp cache"
if (cd training_source/project && sha256sum --quiet -c experiments/hwlimit_ac/cache/hwlimit_grasp_cache.sha256); then
  echo "OK: grasp cache unchanged"; else echo "FAIL: grasp cache changed"; fail=1; fi
echo "== estimator B, kit symlinks, datasets (via Retention-V1's script)"
out=$(bash training_source/project/experiments/retention_v1/scripts/check_preservation.sh 2>&1)
echo "$out" | grep -E '^(OK|FAIL)' | grep -vE 'tracked file' | sed 's/^/   /'
echo "$out" | grep -E '^FAIL' | grep -vqE 'tracked file' && fail=1
exit $fail
