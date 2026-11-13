#!/usr/bin/env bash
# Verify that the completed A/B/C/D study, original checkpoints and datasets are unchanged.
# Baseline fingerprints were recorded before any force-estimation work (preservation/).
set -uo pipefail
cd "$(dirname "$0")/../../../../.."   # repository root
P=training_source/project/experiments/force_estimation/preservation
fail=0
echo "== tracked files (git HEAD 310ce80 at baseline): sha256"
if sha256sum --quiet -c <(grep -v "  .gitignore$" $P/tracked.sha256); then echo "OK: all $(grep -vc "  .gitignore$" $P/tracked.sha256) tracked files unchanged (.gitignore excluded: one line added for the force dataset)"; else echo "FAIL: tracked file changed"; fail=1; fi
echo "== RSL-RL checkpoints (logs/rsl_rl, incl. A/C/D runs): sha256"
if sha256sum --quiet -c $P/rsl_rl_pt.sha256; then echo "OK: all $(wc -l < $P/rsl_rl_pt.sha256) checkpoints unchanged"; else echo "FAIL: checkpoint changed"; fail=1; fi
echo "== datasets (path size mtime)"
for f in scratch_data_stat.txt:training_source/project/experiments/scratch_state_estimation/data proj_data_stat.txt:training_source/project/data; do
  ref=${f%%:*}; dir=${f#*:}
  if diff -q <(find $dir -type f -printf "%p %s %T@\n" 2>/dev/null | sort) $P/$ref >/dev/null; then echo "OK: $dir unchanged"; else echo "FAIL: $dir changed"; fail=1; fi
done
echo "== git: modified tracked files (expected: only .gitignore)"
git status --porcelain | grep -v '^??' || true
exit $fail
