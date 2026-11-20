#!/usr/bin/env python3
"""Phase 1 perturbation tolerance (seed 6103): the first 5 robust-subset grasps x levels L0-L3 x
A1-A3, R1.  Selects the nominal grasp B (highest mean 60-s turns at L2, PROTOCOL.md) and
writes the held-out items for conditions B and D (items/heldout_BD.json).

    python experiments/grasp_rotation_v2/scripts/analyze_perturbation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

E = Path("experiments/grasp_rotation_v2")
POL = ("A1", "A2", "A3", "R1")
LEVELS = ("L0", "L1", "L2", "L3")


def main():
  meta = json.loads((E / "results" / "perturbation_levels.json").read_text())
  top5 = meta["top5"]
  res = {"top5": top5, "levels": meta["levels"], "per_level": {}}
  S = np.zeros((len(LEVELS), len(top5), len(POL)))
  P = np.zeros_like(S)
  T30 = np.zeros_like(S)
  for li, lv in enumerate(LEVELS):
    for pi, p in enumerate(POL):
      tr = np.load(E / "data" / f"p1pert_{p}_{lv}_s6103.npz")
      e, code, dt = tr["end_step"], tr["end_code"], float(tr["dt"])
      for gi, g in enumerate(top5):
        k = tr["cache_id"] == g
        S[li, gi, pi] = tr["yaw_inc"][:, k].astype(np.float64).sum(0).mean() / (2 * math.pi)
        T30[li, gi, pi] = tr["yaw_inc"][:600, k].astype(np.float64).sum(0).mean() / (2 * math.pi)
        P[li, gi, pi] = 1 - (((e[k] >= 0) & (code[k] >= 2)).mean())
  for li, lv in enumerate(LEVELS):
    res["per_level"][lv] = {str(g): {"S_T60_mean": float(S[li, gi].mean()), "T30_mean": float(T30[li, gi].mean()),
                                    "P60_mean": float(P[li, gi].mean()), "S_per_policy": S[li, gi].round(3).tolist()}
                            for gi, g in enumerate(top5)}
    res["per_level"][lv]["all5"] = {"S_T60_mean": float(S[li].mean()), "T30_mean": float(T30[li].mean()), "P60_mean": float(P[li].mean())}
  li2 = LEVELS.index("L2")
  nominal = int(top5[int(np.argmax(S[li2].mean(-1)))])
  res["nominal_grasp_B"] = nominal
  res["rule"] = "highest mean 60-s turns over A1-A3, R1 at L2 (PROTOCOL.md Phase 1)"
  (E / "results" / "perturbation.json").write_text(json.dumps(res, indent=1))
  for lv in LEVELS:
    print(lv, " ".join(f"g{g}: S {res['per_level'][lv][str(g)]['S_T60_mean']:.2f} P60 {res['per_level'][lv][str(g)]['P60_mean']:.3f}" for g in top5),
          "| all5 S %.2f P60 %.3f" % (res["per_level"][lv]["all5"]["S_T60_mean"], res["per_level"][lv]["all5"]["P60_mean"]))
  print("nominal grasp B:", nominal)
  L2 = meta["levels"]["L2"]
  items = []
  for s in (6201, 6202):
    for p in POL:
      items.append(dict(key=f"ho_B_{p}_nominal_s{s}", policy=p, seed=s, num_envs=2048, steps=1200, preset="nominal",
                        assign=f"grasps:{nominal}", pose_range=None))
      items.append(dict(key=f"ho_D_{p}_nominal_s{s}", policy=p, seed=s, num_envs=2048, steps=1200, preset="nominal",
                        assign=f"grasps:{nominal}", pose_range=L2))
  for pr in ("heavy_slippery", "high_delay", "obs_noise_x3"):
    for p in POL:
      items.append(dict(key=f"ho_B_{p}_{pr}_s6201", policy=p, seed=6201, num_envs=2048, steps=1200, preset=pr,
                        assign=f"grasps:{nominal}", pose_range=None))
  (E / "items" / "heldout_BD.json").write_text(json.dumps(items, indent=1))
  print(len(items), "held-out B/D items written")


if __name__ == "__main__":
  main()
