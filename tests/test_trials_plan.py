"""LEAP_Hand_RL_Deploy_Kit/kit/trials.py schedules: the DRAFT final3 plan is balanced; earlier plans are unchanged."""

from __future__ import annotations

import collections
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "LEAP_Hand_RL_Deploy_Kit" / "kit" / "trials.py"


def schedule(plan: str, blocks: int, tmp_path: Path) -> list[str]:
  env = {**os.environ, "LEAP_TRIAL_PLAN": plan, "LEAP_TRIALS_CSV": str(tmp_path / f"{plan}.csv")}
  out = subprocess.run([sys.executable, str(TRIALS), "plan", "--blocks", str(blocks)], env=env, check=True,
                       capture_output=True, text=True).stdout
  return [line.split()[3] for line in out.splitlines() if line.strip()]


def test_final3_is_a_balanced_williams_square(tmp_path: Path):
  s = schedule("final3", 12, tmp_path)
  assert len(s) == 36
  assert collections.Counter(s) == {"M400": 12, "A3": 12, "C5s3": 12}
  seqs = [tuple(s[i:i + 3]) for i in range(0, 36, 3)]
  assert len(set(seqs)) == 6  # all six orders of three arms, each twice
  for pos in range(3):  # every arm equally often in every position
    assert collections.Counter(q[pos] for q in seqs) == {"M400": 4, "A3": 4, "C5s3": 4}
  pairs = collections.Counter((q[i], q[i + 1]) for q in seqs for i in range(2))
  assert len(pairs) == 6 and set(pairs.values()) == {4}  # every arm follows every other arm equally often


def test_earlier_plans_unchanged(tmp_path: Path):
  assert schedule("nov20", 5, tmp_path)[:8] == ["A3", "C5", "R1", "P0", "C5", "P0", "A3", "R1"]
  assert len(schedule("nov20", 5, tmp_path)) == 20
  assert schedule("c5seeds", 5, tmp_path)[:4] == ["C5s3", "C5", "C5", "C5s3"]
  assert len(schedule("c5seeds", 5, tmp_path)) == 20


def test_unknown_plan_refused(tmp_path: Path):
  env = {**os.environ, "LEAP_TRIAL_PLAN": "nope", "LEAP_TRIALS_CSV": str(tmp_path / "x.csv")}
  r = subprocess.run([sys.executable, str(TRIALS), "plan"], env=env, capture_output=True, text=True)
  assert r.returncode != 0 and "final3" in r.stderr
