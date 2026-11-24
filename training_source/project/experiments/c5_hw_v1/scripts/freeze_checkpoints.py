#!/usr/bin/env python3
"""Copy the EFFORT_1 selected (max 30-s validation turns, ties -> later) and final checkpoints and the
run's params into experiments/c5_hw_v1/checkpoints/ and write MANIFEST.txt (never overwrites).

    python experiments/c5_hw_v1/scripts/curves.py assess && python experiments/c5_hw_v1/scripts/freeze_checkpoints.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

E = Path("experiments/c5_hw_v1")


def sha(p: Path) -> str:
  return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
  conv = json.loads((E / "results" / "convergence.json").read_text())["runs"]["EFFORT_1"]
  run_dir = Path(json.loads((E / "results" / "curves" / "EFFORT_1.json").read_text())["run_dir"])
  sel = conv["selected_iteration"]
  out = E / "checkpoints"
  out.mkdir(exist_ok=True)
  lines = ["# C5-HW-EFFORT-V1, single training seed (PPO seed 1). Selected (max 30-s validation turns, seed 7001; ties -> later) and final.",
           "# copy | sha256 | source"]
  for tag, it in (("selected", sel), ("final", 3000)):
    src = run_dir / f"model_{it}.pt"
    dst = out / f"EFFORT_1_{tag}_model_{it}.pt"
    if not dst.exists():
      shutil.copy2(src, dst)
    assert sha(dst) == sha(src)
    lines.append(f"{dst.name} | {sha(dst)} | {src}")
  pd = out / "EFFORT_1_params"
  pd.mkdir(exist_ok=True)
  for f in ("env.yaml", "agent.yaml"):
    if not (pd / f).exists():
      shutil.copy2(run_dir / "params" / f, pd / f)
    lines.append(f"EFFORT_1_params/{f} | {sha(pd / f)} | {run_dir / 'params' / f}")
  (out / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
