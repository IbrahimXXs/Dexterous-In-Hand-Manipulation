#!/usr/bin/env python3
"""After `curves.py assess`: copy each V2 run's selected and final checkpoints (+ params) into
checkpoints/ with sha256 (MANIFEST.txt), and write the Phase 4 held-out items for the
selected checkpoints (items/phase4.json): Setting A nominal (6201, 6202) and presets (6201),
Setting B nominal grasp and robust subset (6201, 6202).

    python experiments/grasp_rotation_v2/scripts/freeze_and_items.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

E = Path("experiments/grasp_rotation_v2")


def main():
  conv = json.loads((E / "results" / "convergence.json").read_text())["runs"]
  (E / "checkpoints").mkdir(exist_ok=True)
  lines = ["# Selected (max 30-s validation turns, seed 7001; ties -> later) and final checkpoints", "# copy | sha256 | source"]
  sel = {}
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
      if kind == "selected":
        sel[run.split("_")[0]] = str(dst)
    (E / "checkpoints" / f"{run}_params").mkdir(exist_ok=True)
    for f in ("env.yaml", "agent.yaml"):
      shutil.copy2(rd / "params" / f, E / "checkpoints" / f"{run}_params" / f)
  (E / "checkpoints" / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))
  rank = json.loads((E / "results" / "grasp_ranking.json").read_text())
  nominal = json.loads((E / "results" / "perturbation.json").read_text())["nominal_grasp_B"]
  subset = ",".join(map(str, rank["robust_subset_C"]))
  items = []
  for c, ck in sel.items():
    base = dict(policy=ck, num_envs=2048, steps=1200, pose_range=None)
    for s in (6201, 6202):
      items.append(dict(base, key=f"ho_A_{c}_nominal_s{s}", seed=s, preset="nominal", assign=None))
    for pr in ("heavy_slippery", "high_delay", "obs_noise_x3"):
      items.append(dict(base, key=f"ho_A_{c}_{pr}_s6201", seed=6201, preset=pr, assign=None))
    items.append(dict(base, key=f"ho_N_{c}_nominal_s6201", seed=6201, preset="nominal", assign=None,
                      terminations_off=["cube_pose_deviation"], record_quat=True))
    for s in (6201, 6202):
      items.append(dict(base, key=f"ho_B_{c}_nominal_s{s}", seed=s, preset="nominal", assign=f"grasps:{nominal}"))
      items.append(dict(base, key=f"ho_C_{c}_nominal_s{s}", seed=s, preset="nominal", assign=f"grasps:{subset}"))
  (E / "items" / "phase4.json").write_text(json.dumps(items, indent=1))
  print(len(items), "Phase 4 items")


if __name__ == "__main__":
  main()
