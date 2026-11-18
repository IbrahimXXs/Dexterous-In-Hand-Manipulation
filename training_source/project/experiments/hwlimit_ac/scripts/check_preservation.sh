#!/usr/bin/env bash
# Verify that nothing that existed before this study changed: every git-tracked file at the
# baseline commit (incl. the A/B/C/D study, force study, deploy kit and hardware logs), every
# RSL-RL checkpoint, estimator B and the three datasets.  Baselines in preservation/.
set -uo pipefail
cd "$(dirname "$0")/../../../../.."   # repository root
P=training_source/project/experiments/hwlimit_ac/preservation
fail=0
echo "== tracked files at baseline commit $(cat $P/baseline_commit.txt | cut -c1-7): sha256"
if sha256sum --quiet -c <(grep -v "  .gitignore$" $P/tracked.sha256); then echo "OK: all $(grep -vc "  .gitignore$" $P/tracked.sha256) tracked files unchanged (.gitignore checked by diff below)"; else echo "FAIL: tracked file changed"; fail=1; fi
echo "== RSL-RL checkpoints that existed at baseline: sha256"
if sha256sum --quiet -c $P/rsl_rl_pt.sha256; then echo "OK: all $(wc -l < $P/rsl_rl_pt.sha256) checkpoints unchanged"; else echo "FAIL: checkpoint changed"; fail=1; fi
echo "== estimator B"
(cd training_source/project && sha256sum --quiet -c experiments/hwlimit_ac/preservation/estimator_B.sha256) && echo "OK: estimator_B.pt unchanged" || { echo "FAIL: estimator_B.pt changed"; fail=1; }
echo "== datasets (path size mtime)"
for d in training_source/project/experiments/scratch_state_estimation/data training_source/project/data training_source/project/experiments/force_estimation/data; do
  ref=$P/stat_$(echo $d | tr / _).txt
  if diff -q <(find $d -type f -printf "%p %s %T@\n" 2>/dev/null | sort) $ref >/dev/null; then echo "OK: $d unchanged"; else echo "FAIL: $d changed"; fail=1; fi
done
echo "== git: modified tracked files (expected: none, or .gitignore with appended lines only)"
git status --porcelain | grep -v '^??' || true
git diff -U0 .gitignore | grep '^-[^-]' && { echo "FAIL: .gitignore lines removed"; fail=1; }
exit $fail
