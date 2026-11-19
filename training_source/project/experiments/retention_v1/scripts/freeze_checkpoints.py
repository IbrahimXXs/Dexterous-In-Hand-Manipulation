#!/usr/bin/env python3
"""Copy the selected (validation rule) and final checkpoints of each Retention-V1 run into
checkpoints/ with sha256 (MANIFEST.txt) and the run's params/.  Run after `curves.py assess`."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

E = Path("experiments/retention_v1")


def main():
  conv = json.loads((E / "results" / "convergence.json").read_text())["runs"]
  lines = ["# Selected (max 30-s validation turns, seed 7001, nominal; ties -> later) and final checkpoints", "# copy | sha256 | source"]
  (E / "checkpoints").mkdir(exist_ok=True)
  for run, v in conv.items():
    if v.get("selected_iteration") is None:
      continue
    rd = Path(json.loads((E / "results" / "curves" / f"{run}.json").read_text())["run_dir"])
    for kind, it in (("selected", v["selected_iteration"]), ("final", 3000)):
      src, dst = rd / f"model_{it}.pt", E / "checkpoints" / f"{run}_{kind}_model_{it}.pt"
      shutil.copy2(src, dst)
      h = hashlib.sha256(dst.read_bytes()).hexdigest()
      assert h == hashlib.sha256(src.read_bytes()).hexdigest()
      lines.append(f"{dst.name} | {h} | {src}")
    (E / "checkpoints" / f"{run}_params").mkdir(exist_ok=True)
    for f in ("env.yaml", "agent.yaml"):
      shutil.copy2(rd / "params" / f, E / "checkpoints" / f"{run}_params" / f)
  (E / "checkpoints" / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
